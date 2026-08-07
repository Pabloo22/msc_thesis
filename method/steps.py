"""Per-step measurement and training for a trajectory.

Every function here is idempotent and resumable: it checks for its output
artifact first and returns early if present. Because all writes are atomic (see
:mod:`method.store`), presence implies completeness, so there is no manifest to
keep in sync.

Artifact names live in one place, :class:`Artifacts`, since both the producers
here and the resume checks depend on them agreeing.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import torch

from method.backends import ExecutionBackend, materialize
from method.config import DeltaPMode, HNeutralSource, StepConfig, TrajectoryConfig
from method.latent import compute_latent, delta_projection, summarize
from method.noise import behavior_summary
from method.store import (
    Store,
    atomic_dir,
    atomic_file,
    atomic_symlink,
    get_weights_id,
    training_sample_id,
)
from method.utils import DATASET_DIR, NEUTRAL_DIR

logger = logging.getLogger(__name__)


class Artifacts:
    """Filenames written into a checkpoint's measurement directory."""

    BEHAVIOR_CSV = "behavior.csv"
    BEHAVIOR_JSON = "behavior.json"
    EXTRACT_POS = "extract_pos.csv"
    EXTRACT_NEG = "extract_neg.csv"
    LATENT_JSON = "latent.json"
    NEUTRAL_ANSWERS = "neutral_answers.jsonl"

    # DeltaP describes the update a checkpoint is *about to* receive, so it is
    # not a property of the checkpoint alone: it must be keyed by which
    # examples the next step trains on. Trajectories sharing a prefix and then
    # diverging land on the same weights_id, and an unkeyed name would make the
    # second one silently read the first one's numbers.
    @staticmethod
    def delta_p_json(sample_id: str) -> str:
        return f"delta_p_{sample_id}.json"

    @staticmethod
    def delta_p_csv(sample_id: str) -> str:
        return f"delta_p_{sample_id}.csv"

    @staticmethod
    def persona_vector(trait: str) -> str:
        # Named by generate_vec.py, which we call unmodified.
        return f"{trait}_response_avg_diff.pt"

    @staticmethod
    def h_neutral(source: str) -> str:
        return f"h_neutral_{source}"


# --- dataset helpers ------------------------------------------------------


def dataset_path(step: StepConfig) -> Path:
    path = DATASET_DIR / step.dataset / f"{step.version.value}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no dataset at {path}")
    return path


def cached_training_sample(step: StepConfig, seed: int, store: Store) -> Path:
    """Materialise (once) the exact examples ``step`` names, in the store.

    Sampling is written to disk so that DeltaP is computed over precisely the
    examples used for the update, not over the dataset in general. It is also
    a pure function of ``(step.dataset, step.version, step.n_examples[, seed])``
    -- independent of trait, trajectory, or position in the trajectory -- so
    the result is cached once under that hash. This is safe to resume from:
    regenerating the cache from scratch reproduces the exact same content,
    since it depends only on those inputs and the (static) dataset file.

    Returns the cached path itself. Training goes through
    :func:`sample_training_file`, which additionally links it into the run
    directory; probes, which belong to no single step, use this directly.
    """
    cached = store.training_sample_path(training_sample_id(step, seed))
    if cached.exists():
        logger.info("[skip] training sample already cached -> %s", cached)
        return cached
    lines = [
        line for line in dataset_path(step).read_text().splitlines() if line.strip()
    ]
    if step.n_examples is not None and step.n_examples < len(lines):
        lines = random.Random(seed).sample(lines, step.n_examples)
    with atomic_file(cached) as scratch:
        scratch.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Training sample: %d examples -> %s", len(lines), cached)
    return cached


def sample_training_file(step: StepConfig, seed: int, dest: Path, store: Store) -> Path:
    """The cached sample for ``step``, symlinked into the run directory at ``dest``.

    ``dest`` is a relative symlink rather than a physical copy, so a run
    directory records which examples a step trained on without duplicating them
    once per trajectory that shares the step.
    """
    cached = cached_training_sample(step, seed, store)
    atomic_symlink(dest, cached)
    logger.info("Training file: %s -> %s", dest, cached)
    return dest


def neutral_prompts_file(cfg: TrajectoryConfig) -> Path:
    path = NEUTRAL_DIR / f"{cfg.latent.neutral_prompts_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"neutral prompt set {path} is missing; build it with "
            "`python -m method.prep_neutral_prompts`"
        )
    return path


# --- measurements ---------------------------------------------------------


def measure_behavior(
    cfg: TrajectoryConfig, t: int, store: Store, backend: ExecutionBackend
) -> Path:
    """b_t: judged trait and coherence scores for checkpoint t."""
    wid = get_weights_id(cfg, t)
    out_csv = store.trait_measurement(wid, cfg.trait, Artifacts.BEHAVIOR_CSV)
    if out_csv.exists():
        logger.info(
            "[skip] behavior for step %d already measured (trait=%s)", t, cfg.trait
        )
        return out_csv

    model_path = materialize(cfg, t, store, backend)
    trait_dir = store.trait_measurement_dir(wid, cfg.trait)
    with atomic_dir(trait_dir / "behavior_scratch") as scratch:
        tmp_csv = scratch / Artifacts.BEHAVIOR_CSV
        backend.eval_persona(
            model_path,
            cfg.trait,
            tmp_csv,
            cfg,
            version="eval",
            progress_dir=eval_progress_dir(out_csv),
        )
        df = pd.read_csv(tmp_csv)
        (scratch / Artifacts.BEHAVIOR_JSON).write_text(
            json.dumps(behavior_summary(df, cfg.trait), indent=2),
            encoding="utf-8",
        )
    _promote(
        trait_dir / "behavior_scratch",
        trait_dir,
        marker=Artifacts.BEHAVIOR_CSV,
    )
    return out_csv


def behavior_record(cfg: TrajectoryConfig, t: int, store: Store) -> dict[str, float]:
    """The behaviour summary for checkpoint ``t``, derived from the scored rows.

    Read from ``behavior.csv`` rather than the ``behavior.json`` written beside
    it, because that JSON is only written when the eval actually *runs*.
    :func:`measure_behavior` returns early once the CSV exists, so a checkpoint
    measured before a field was added to the summary would keep the old shape
    for good -- and checkpoints are shared, so whichever experiment measures one
    first fixes its format for every experiment that follows.

    The acute case is the base checkpoint. Every trajectory of every family
    resolves to one ``M_0`` (:meth:`~method.config.TrajectoryConfig.weights_key`
    normalises the seed away at ``t=0``, and excludes trait), so exp3 having
    measured it first would leave exp2's ``t=0`` record without ``SE`` while its
    own ``t>=1`` records carried it -- and ``t=0`` is precisely what the
    validation fan differences against.

    Deriving from the raw rows removes the staleness by construction: there is
    no cached summary left to invalidate. Costs one CSV parse per checkpoint,
    against an eval that just generated and judged hundreds of completions.
    """
    csv = store.trait_measurement(
        get_weights_id(cfg, t), cfg.trait, Artifacts.BEHAVIOR_CSV
    )
    return behavior_summary(pd.read_csv(csv), cfg.trait)


def eval_progress_dir(final_csv: Path) -> Path:
    """Where an eval banks partial results, keyed by the artifact it builds.

    Every attempt writes its CSV to a fresh scratch path (see ``atomic_file`` /
    ``atomic_dir``), so only the final destination is stable enough to resume
    against. Dot-prefixed and sitting beside that destination, it is
    self-evidently not the artifact -- resume checks look for exact filenames --
    and ``eval_wrapper`` deletes it as soon as the CSV exists.
    """
    return final_csv.parent / f".{final_csv.name}.progress"


def _promote(scratch_dir: Path, target_dir: Path, *, marker: str) -> None:
    """Move completed artifacts out of a scratch subdirectory.

    ``marker`` is the file whose presence callers treat as "this step is done",
    so it is moved last. A crash midway then leaves the step looking incomplete
    and it is redone, rather than looking complete but missing siblings.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    items = sorted(scratch_dir.iterdir(), key=lambda p: p.name == marker)
    for item in items:
        item.replace(target_dir / item.name)
    shutil.rmtree(scratch_dir, ignore_errors=True)


def extract_persona_vector(
    cfg: TrajectoryConfig, t: int, store: Store, backend: ExecutionBackend
) -> Path:
    """v_t, re-extracted from the t=0 responses.

    The pos/neg generations happen once, on the base model. Later steps reuse
    that text so the pos/neg filter mask stays frozen, which protects against
    the failure where a drifted model can no longer produce a credible negative
    response and the vector silently changes meaning.
    """
    wid = get_weights_id(cfg, t)
    trait_dir = store.trait_measurement_dir(wid, cfg.trait)
    out = trait_dir / Artifacts.persona_vector(cfg.trait)
    if out.exists():
        logger.info(
            "[skip] persona vector for step %d already extracted (trait=%s)",
            t,
            cfg.trait,
        )
        return out

    base_wid = get_weights_id(cfg, 0)
    pos_csv = store.trait_measurement(base_wid, cfg.trait, Artifacts.EXTRACT_POS)
    neg_csv = store.trait_measurement(base_wid, cfg.trait, Artifacts.EXTRACT_NEG)
    if not (pos_csv.exists() and neg_csv.exists()):
        base_model = materialize(cfg, 0, store, backend)
        for kind, path in (("pos", pos_csv), ("neg", neg_csv)):
            with atomic_file(path) as scratch:
                backend.eval_persona(
                    base_model,
                    cfg.trait,
                    scratch,
                    cfg,
                    version="extract",
                    persona_instruction_type=kind,
                    progress_dir=eval_progress_dir(path),
                )

    model_path = materialize(cfg, t, store, backend)
    with atomic_dir(trait_dir / "vector_scratch") as scratch:
        backend.extract_vector(
            model_path,
            cfg.trait,
            pos_csv,
            neg_csv,
            scratch,
            cfg.eval.persona_vector_threshold,
        )
    _promote(
        trait_dir / "vector_scratch",
        trait_dir,
        marker=Artifacts.persona_vector(cfg.trait),
    )
    return out


def measure_h_neutral(
    cfg: TrajectoryConfig, t: int, store: Store, backend: ExecutionBackend
) -> dict[str, Path]:
    """h_neutral_t for each configured source.

    ``base`` reads M_0's answers to the neutral prompts, so only representation
    drift is captured; ``current`` would have M_t answer for itself. Both are
    computed when the config asks for it, making the choice a measurement
    rather than an assumption.
    """
    weights_id = get_weights_id(cfg, t)
    results = {
        source: store.measurement_dir(weights_id) / Artifacts.h_neutral(source)
        for source in cfg.latent.h_neutral_source.sources
    }
    missing = [
        source
        for source, out_dir in results.items()
        if not (out_dir / "mean_by_layer.pt").exists()
    ]
    if not missing:
        # Checked before materialize: a resumed run must not pay for a full
        # merge-chain rebuild just to discover there is nothing left to do.
        logger.info("[skip] h_neutral for step %d already measured", t)
        return results

    model_path = materialize(cfg, t, store, backend)
    for source in missing:
        out_dir = results[source]
        if source == HNeutralSource.BASE.value:
            pairs = _base_neutral_answers(cfg, store, backend)
        else:
            pairs = _neutral_answers_for(cfg, t, store, backend, model_path)

        with atomic_dir(out_dir) as scratch:
            backend.hidden_states(model_path, pairs, cfg.model.layer, scratch)
    return results


def _base_neutral_answers(
    cfg: TrajectoryConfig, store: Store, backend: ExecutionBackend
) -> Path:
    """M_0's answers to the neutral prompts, generated once for the whole run."""
    return _neutral_answers_for(
        cfg, 0, store, backend, materialize(cfg, 0, store, backend)
    )


def _neutral_answers_for(
    cfg: TrajectoryConfig,
    t: int,
    store: Store,
    backend: ExecutionBackend,
    model_path: str,
) -> Path:
    """Answers to the neutral prompts from checkpoint ``t``."""
    wid = get_weights_id(cfg, t)
    out = store.measurement(wid, Artifacts.NEUTRAL_ANSWERS)
    if out.exists():
        return out
    prompts = [
        json.loads(line)
        for line in neutral_prompts_file(cfg).read_text().splitlines()
        if line.strip()
    ][: cfg.latent.n_neutral]
    with atomic_file(out) as scratch:
        backend.generate_answers(model_path, prompts, scratch, cfg)
    return out


def compute_step_latent(
    cfg: TrajectoryConfig, t: int, store: Store
) -> dict[str, dict[str, float]]:
    """z_t = (p, q, rho, r), one entry per h_neutral source."""
    wid = get_weights_id(cfg, t)
    out = store.trait_measurement(wid, cfg.trait, Artifacts.LATENT_JSON)
    sources = cfg.latent.h_neutral_source.sources
    if out.exists():
        cached = json.loads(out.read_text())
        # The filename does not encode which sources were requested, so a run
        # measured under BASE and later re-configured to BOTH would otherwise
        # hand back a dict silently missing the "current" series.
        if all(source in cached for source in sources):
            return cached

    layer = cfg.model.layer
    v0 = torch.load(
        store.trait_measurement(
            get_weights_id(cfg, 0), cfg.trait, Artifacts.persona_vector(cfg.trait)
        ),
        weights_only=False,
    )[layer]
    vt = torch.load(
        store.trait_measurement(wid, cfg.trait, Artifacts.persona_vector(cfg.trait)),
        weights_only=False,
    )[layer]

    latents = {}
    for source in sources:
        h = torch.load(
            store.measurement_dir(wid)
            / Artifacts.h_neutral(source)
            / "mean_by_layer.pt",
            weights_only=False,
        )[layer]
        latents[source] = compute_latent(v0, vt, h).as_dict()

    with atomic_file(out) as scratch:
        scratch.write_text(json.dumps(latents, indent=2), encoding="utf-8")
    logger.info("z_%d = %s", t, latents)
    return latents


def compute_delta_p(
    cfg: TrajectoryConfig,
    t: int,
    store: Store,
    backend: ExecutionBackend,
    train_file: Path,
    sample_id: str,
) -> dict[str, float]:
    """DeltaP for the dataset step ``t`` is about to train on.

    Answers are never regenerated: M_0's answers to the training prompts are
    produced once, and later steps only recompute hidden states over that fixed
    text. This keeps the measurement comparable across steps and avoids the
    degenerate case where a drifted model stops producing usable answers.

    ``sample_id`` identifies those training examples and keys every artifact
    written here, because ``weights_id`` alone describes the steps already
    taken and says nothing about the update being measured.
    """
    wid = get_weights_id(cfg, t)
    dp_key = _delta_p_key(cfg, sample_id)
    out = store.trait_measurement(wid, cfg.trait, Artifacts.delta_p_json(dp_key))
    if out.exists():
        return json.loads(out.read_text())

    model_path = materialize(cfg, t, store, backend)
    layer = cfg.model.layer
    vt = torch.load(
        store.trait_measurement(wid, cfg.trait, Artifacts.persona_vector(cfg.trait)),
        weights_only=False,
    )[layer]

    train_file = _delta_p_subset(cfg, train_file, store, wid, dp_key)

    target_dir = store.measurement_dir(wid) / "delta_p_target" / dp_key
    if not (target_dir / f"samples_layer{layer}.pt").exists():
        with atomic_dir(target_dir) as scratch:
            backend.hidden_states(model_path, train_file, layer, scratch)
    h_target = torch.load(target_dir / f"samples_layer{layer}.pt", weights_only=False)

    if cfg.delta_p.mode is DeltaPMode.PROMPT_LAST:
        # The paper's approximation: the last prompt token stands in for the
        # base generation, so no answers are generated at all.
        raise NotImplementedError(
            "PROMPT_LAST needs prompt-token activations from _hidden_worker; "
            "use FULL or SAMPLE until that path is added"
        )

    pred_file = _base_answers_to_training_prompts(
        cfg, store, backend, train_file, dp_key
    )
    pred_dir = store.measurement_dir(wid) / "delta_p_predicted" / dp_key
    if not (pred_dir / f"samples_layer{layer}.pt").exists():
        with atomic_dir(pred_dir) as scratch:
            backend.hidden_states(model_path, pred_file, layer, scratch)
    h_pred = torch.load(pred_dir / f"samples_layer{layer}.pt", weights_only=False)

    values = delta_projection(h_target, h_pred, vt)
    stats = summarize(values)
    with atomic_file(
        store.trait_measurement(wid, cfg.trait, Artifacts.delta_p_csv(dp_key))
    ) as scratch:
        pd.DataFrame({"delta_p": values.tolist()}).to_csv(scratch, index=False)
    with atomic_file(out) as scratch:
        scratch.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("DeltaP_%d = %.4f (n=%d)", t, stats["mean"], stats["n"])
    return stats


def measure_probes(
    cfg: TrajectoryConfig,
    t: int,
    store: Store,
    backend: ExecutionBackend,
    probes: Sequence[StepConfig] | None = None,
    *,
    on_probe_done: Callable[[], None] | None = None,
) -> dict[str, dict[str, float]]:
    """DeltaP at checkpoint ``t`` for every probe dataset, keyed by ``dataset_id``.

    ``compute_delta_p`` has no notion of "the step about to run" -- it measures
    whichever examples it is handed at whichever checkpoint. Probing is just
    calling it for datasets the trajectory is *not* training on right now, which
    is what turns DeltaP from one value per dataset into a series over time.

    Defaults to ``cfg.probes``; pass ``probes`` explicitly to measure a
    different set (see :mod:`method.probe_base`, which probes every dataset at
    the base checkpoint). Requires ``v_t``, so callers must have run
    :func:`extract_persona_vector` for this checkpoint first.

    ``on_probe_done``, if given, runs after each probe's DeltaP lands on disk --
    a hook a caller can use to sync that one result immediately rather than
    waiting on the whole (potentially long) list of probes. This module stays
    unaware of what the hook does; ``steps.py`` has no sync-layer import.
    """
    probes = cfg.probes if probes is None else probes
    results: dict[str, dict[str, float]] = {}
    for probe in probes:
        train_file = cached_training_sample(probe, cfg.seed, store)
        results[probe.dataset_id] = compute_delta_p(
            cfg, t, store, backend, train_file, training_sample_id(probe, cfg.seed)
        )
        if on_probe_done is not None:
            on_probe_done()
    return results


def _delta_p_key(cfg: TrajectoryConfig, sample_id: str) -> str:
    """Key for DeltaP artifacts: the training examples actually projected.

    Under ``SAMPLE`` the projection runs over a subsample, so the number of
    samples joins the key -- otherwise re-running with a different
    ``n_samples`` would read back the old subsample's statistics.
    """
    if cfg.delta_p.mode is DeltaPMode.SAMPLE:
        return f"{sample_id}-sub{cfg.delta_p.n_samples}"
    return sample_id


def _delta_p_subset(
    cfg: TrajectoryConfig, train_file: Path, store: Store, wid: str, dp_key: str
) -> Path:
    """Restrict DeltaP to a subsample when the config asks for one.

    ``SAMPLE`` mode exists so that action features stay affordable when a step
    trains on thousands of examples: DeltaP over a random subset estimates the
    dataset-level projection difference without a generation pass over
    everything. ``FULL`` uses every example the step trains on.

    The cached file is named by ``dp_key`` (which embeds ``n_samples``), not by
    the sample id alone: otherwise re-running with a different ``n_samples``
    would silently reuse the old subsample under the new key's artifacts.
    """
    if cfg.delta_p.mode is not DeltaPMode.SAMPLE:
        return train_file
    assert cfg.delta_p.n_samples is not None  # enforced by DeltaPConfig
    lines = [ln for ln in train_file.read_text().splitlines() if ln.strip()]
    if len(lines) <= cfg.delta_p.n_samples:
        return train_file
    subset = store.measurement(wid, f"delta_p_input_{dp_key}.jsonl")
    if not subset.exists():
        chosen = random.Random(cfg.seed).sample(lines, cfg.delta_p.n_samples)
        with atomic_file(subset) as scratch:
            scratch.write_text("\n".join(chosen) + "\n", encoding="utf-8")
        logger.info(
            "DeltaP over a %d-example subsample of %d",
            cfg.delta_p.n_samples,
            len(lines),
        )
    return subset


def _base_answers_to_training_prompts(
    cfg: TrajectoryConfig,
    store: Store,
    backend: ExecutionBackend,
    train_file: Path,
    dp_key: str,
) -> Path:
    """M_0's own answers to a training set's prompts, generated once.

    Keyed by ``dp_key`` rather than the file name: every trajectory's first
    step resolves to the same base checkpoint, so a positional name like
    ``train_step1`` would serve one dataset's answers to every other.
    """
    base_wid = get_weights_id(cfg, 0)
    out = store.measurement(base_wid, f"base_answers_{dp_key}.jsonl")
    if out.exists():
        return out
    rows = [
        json.loads(line) for line in train_file.read_text().splitlines() if line.strip()
    ]
    prompts = [{"messages": r["messages"][:-1]} for r in rows]
    with atomic_file(out) as scratch:
        backend.generate_answers(
            materialize(cfg, 0, store, backend), prompts, scratch, cfg
        )
    return out
