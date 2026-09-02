"""Per-step measurement and training for a trajectory.

Every function here is idempotent and resumable: it checks for its output
artifact first and returns early if present. Because all writes are atomic (see
:mod:`method.store`), presence implies completeness, so there is no manifest to
keep in sync.

Artifact names live in one place, :class:`Artifacts`, since both the producers
here and the resume checks depend on them agreeing.
"""

from __future__ import annotations

import functools
import json
import logging
import random
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import torch

from method.backends import ExecutionBackend, materialize
from method.config import (
    DeltaPMode,
    DeltaPView,
    HNeutralSource,
    PredictedSource,
    ProjectionAxis,
    StepConfig,
    TrajectoryConfig,
)
from method.latent import H_NORM, delta_projection, latent_record, summarize
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
    # The filename is the cache key, and it carries the convention because
    # nothing else does: the plain "latent.json" beside it holds $p$ and $q$ as
    # unnormalised projections, and :func:`compute_step_latent` would serve
    # those back forever under the old name with no error and no staleness
    # warning. Renaming rather than deleting keeps the old values readable for
    # comparison and means a box that pulls a stale bundle cannot silently mix
    # the two conventions.
    LATENT_JSON = "latent_cosine.json"
    NEUTRAL_ANSWERS = "neutral_answers.jsonl"

    # DeltaP describes the update a checkpoint is *about to* receive, so it is
    # not a property of the checkpoint alone: it must be keyed by which
    # examples the next step trains on. Trajectories sharing a prefix and then
    # diverging land on the same weights_id, and an unkeyed name would make the
    # second one silently read the first one's numbers.
    # The view joins the name for the same reason, one level up: one
    # checkpoint can hold the same examples projected onto two axes and
    # differenced against two sets of answers. The default view keeps the
    # unqualified name it has always had, so every measurement already in the
    # store stays a cache hit (see :meth:`method.config.DeltaPView.key`).
    @staticmethod
    def delta_p_json(sample_id: str, view: DeltaPView) -> str:
        return f"{view.key('delta_p')}_{sample_id}.json"

    @staticmethod
    def delta_p_csv(sample_id: str, view: DeltaPView) -> str:
        return f"{view.key('delta_p')}_{sample_id}.csv"

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
            model_id=wid,
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
                    model_id=base_wid,
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
    drift is captured; ``current`` has M_t answer for itself, which also
    captures behavioural drift and costs a generation pass per checkpoint (the
    family that pays for it is
    :func:`method.experiments.build_exp2_hregen_configs`). Both are computed
    when the config asks for it, making the choice a measurement rather than an
    assumption.

    At ``t = 0`` the two coincide by construction -- the current model *is*
    M_0 -- and both resolve to the same cached answers, so the recomputed
    series starts from exactly the frozen one's z_0 rather than from an
    independent draw.
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


def h_neutral_path(store: Store, wid: str, source: str) -> Path:
    """A checkpoint's mean neutral activation, one row per layer.

    One definition shared by the measurement below and by the backfills that
    have to find the same tensor after the fact.
    """
    return store.measurement_dir(wid) / Artifacts.h_neutral(source) / "mean_by_layer.pt"


def _fill_h_norm(
    cached: dict[str, dict[str, float]],
    wid: str,
    layer: int,
    store: Store,
    out: Path,
) -> dict[str, dict[str, float]]:
    """Add :data:`method.latent.H_NORM` to a cached z that predates it.

    Strictly an addition: ``p``, ``q``, ``rho`` and ``r`` are handed back
    exactly as they were measured. Recomputing them instead would re-anchor the
    entry onto whichever ``v_0`` the store holds *now*, and exp3 is known to sit
    on several distinct base measurements (see
    :mod:`method.visualization.latent_audit`) -- so a cached run would quietly
    change meaning on re-read rather than merely gain a field.

    A checkpoint whose activation tensor is no longer on this machine keeps its
    record unchanged. The norm is a diagnostic that no figure depends on, so a
    tensor that has been evicted must not fail a trajectory that is otherwise
    complete; :mod:`method.backfill_h_norm` fills those in where the store lives.
    """
    filled = False
    for source, z in cached.items():
        if H_NORM in z:
            continue
        path = h_neutral_path(store, wid, source)
        if not path.exists():
            logger.debug(
                "no h_neutral tensor for %s/%s, leaving h_norm unset", wid, source
            )
            continue
        z[H_NORM] = float(torch.load(path, weights_only=False)[layer].float().norm())
        filled = True

    if filled:
        with atomic_file(out) as scratch:
            scratch.write_text(json.dumps(cached, indent=2), encoding="utf-8")
    return cached


def compute_step_latent(
    cfg: TrajectoryConfig, t: int, store: Store
) -> dict[str, dict[str, float]]:
    """z_t = (p, q, rho, r) and the ``h_norm`` it was normalised by, per source.

    The artifact holds every source ever measured at this checkpoint; the
    return holds only the ones ``cfg`` asked for. The two differ because the
    file is keyed by checkpoint and trait alone -- nothing in its name says
    which sources produced it -- while a run's record should carry the series
    that run paid for and no other. So a config measuring only ``current``
    over checkpoints an earlier config measured under ``base`` *adds* to the
    file and still writes a single-source ``z`` block, which is what keeps the
    two families' trajectories joinable rather than mutually overwriting (the
    same rule :class:`~method.config.DeltaPView` applies to DeltaP's artifact
    names, one level up).

    A source already in the file is handed back verbatim, never recomputed:
    re-deriving it would re-anchor the entry onto whichever ``v_0`` the store
    holds now, and exp3 is known to sit on several distinct base measurements
    (see :mod:`method.visualization.latent_audit`).
    """
    wid = get_weights_id(cfg, t)
    out = store.trait_measurement(wid, cfg.trait, Artifacts.LATENT_JSON)
    sources = cfg.latent.h_neutral_source.sources
    layer = cfg.model.layer
    cached = json.loads(out.read_text()) if out.exists() else {}
    # The filename does not encode which sources were requested, so a run
    # measured under BASE and later re-configured to BOTH would otherwise
    # hand back a dict silently missing the "current" series.
    missing = [source for source in sources if source not in cached]
    if not missing:
        filled = _fill_h_norm(cached, wid, layer, store, out)
        return {source: filled[source] for source in sources}

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

    latents = dict(cached)
    for source in missing:
        h = torch.load(h_neutral_path(store, wid, source), weights_only=False)[layer]
        latents[source] = latent_record(v0, vt, h)

    with atomic_file(out) as scratch:
        scratch.write_text(json.dumps(latents, indent=2), encoding="utf-8")
    measured = {source: latents[source] for source in sources}
    logger.info("z_%d = %s", t, measured)
    return measured


def compute_delta_p(
    cfg: TrajectoryConfig,
    t: int,
    store: Store,
    backend: ExecutionBackend,
    train_file: Path,
    sample_id: str,
    *,
    view: DeltaPView = DeltaPView(),
) -> dict[str, float]:
    r"""DeltaP for the dataset step ``t`` is about to train on.

    ``view`` selects which of the projection differences to take -- which axis,
    and whose answers stand in for the predicted term (see
    :class:`method.config.DeltaPView` for the ladder they form). The default is
    the one every existing measurement took: the checkpoint's own axis
    $v^{(t)}$, differenced against M_0's answers. That gives $\Delta P_0$ at
    ``t = 0`` and $\Delta \hat{P}_t$ after it.

    Freezing the *answers* at M_0 keeps the measurement comparable across steps
    and out of the degenerate case where a drifted model stops producing usable
    text; ``PredictedSource.CURRENT`` gives that up deliberately, and pays a
    generation pass per checkpoint for it. Freezing the *axis* at $v^{(0)}$
    costs nothing at all: the activations below do not depend on it, so a
    second view over a measured checkpoint is a second projection over tensors
    that are already on disk.

    ``sample_id`` identifies those training examples and keys every artifact
    written here, because ``weights_id`` alone describes the steps already
    taken and says nothing about the update being measured.
    """
    wid = get_weights_id(cfg, t)
    dp_key = _delta_p_key(cfg, sample_id)
    out = store.trait_measurement(wid, cfg.trait, Artifacts.delta_p_json(dp_key, view))
    if out.exists():
        return json.loads(out.read_text())

    layer = cfg.model.layer

    # Materialising a checkpoint replays its whole adapter chain onto disk, so
    # it is deferred until something actually needs a forward pass. Projecting
    # cached activations onto another axis needs none, which is what makes that
    # view free rather than merely cheap.
    @functools.cache
    def model_path() -> str:
        return materialize(cfg, t, store, backend)

    vector = _projection_vector(cfg, t, store, view.axis)

    train_file = _delta_p_subset(cfg, train_file, store, wid, dp_key)

    target_dir = store.measurement_dir(wid) / "delta_p_target" / dp_key
    if not (target_dir / f"samples_layer{layer}.pt").exists():
        with atomic_dir(target_dir) as scratch:
            backend.hidden_states(model_path(), train_file, layer, scratch)
    h_target = torch.load(target_dir / f"samples_layer{layer}.pt", weights_only=False)

    if cfg.delta_p.mode is DeltaPMode.PROMPT_LAST:
        # The paper's approximation: the last prompt token stands in for the
        # base generation, so no answers are generated at all.
        raise NotImplementedError(
            "PROMPT_LAST needs prompt-token activations from _hidden_worker; "
            "use FULL or SAMPLE until that path is added"
        )

    # At t = 0 the current model *is* M_0, and generation is greedy, so the two
    # sources would produce byte-identical answers. Resolving to BASE there
    # shares the answers and the forward pass instead of paying for a second
    # copy of both, and makes the recomputed series start at exactly Delta P_0
    # rather than at something that ought to equal it.
    effective = PredictedSource.BASE if t == 0 else view.predicted
    pred_file = _predicted_answers(
        cfg, t, store, backend, train_file, dp_key, effective
    )
    pred_dir = predicted_dir(store, wid, dp_key, effective)
    if not (pred_dir / f"samples_layer{layer}.pt").exists():
        with atomic_dir(pred_dir) as scratch:
            backend.hidden_states(model_path(), pred_file, layer, scratch)
    h_pred = torch.load(pred_dir / f"samples_layer{layer}.pt", weights_only=False)

    values = delta_projection(h_target, h_pred, vector)
    stats = summarize(values)
    with atomic_file(
        store.trait_measurement(wid, cfg.trait, Artifacts.delta_p_csv(dp_key, view))
    ) as scratch:
        pd.DataFrame({"delta_p": values.tolist()}).to_csv(scratch, index=False)
    with atomic_file(out) as scratch:
        scratch.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info(
        "DeltaP[%s]_%d = %.4f (n=%d)",
        view.suffix or "default",
        t,
        stats["mean"],
        stats["n"],
    )
    return stats


#: Subdirectory of a *trait* measurement bundle holding the checkpoint's own
#: extraction draw and the persona vector taken from it, written by
#: :mod:`method.axis_refresh`. Named here rather than there because
#: :func:`_projection_vector` reads that vector and that module imports this
#: one, so the path has to live on the side of the import that has no choice.
AXIS_REFRESH_SUBDIR = "axis_refresh"


def onpolicy_vector_path(store: Store, wid: str, trait: str) -> Path:
    r"""Where $v^{(t \leftarrow t)}$ lives for one checkpoint and trait.

    The vector sits in a ``vector/`` subdirectory of its own because the
    vendored extractor emits three files, and giving them a directory means
    :func:`~method.store.atomic_dir` alone makes the write all-or-nothing.
    """
    return (
        store.trait_measurement_dir(wid, trait)
        / AXIS_REFRESH_SUBDIR
        / "vector"
        / Artifacts.persona_vector(trait)
    )


def _projection_vector(
    cfg: TrajectoryConfig, t: int, store: Store, axis: ProjectionAxis
) -> torch.Tensor:
    r"""The persona vector DeltaP projects onto, at this checkpoint's layer.

    ``CURRENT`` reads $v^{(t)}$, extracted from the checkpoint being measured.
    ``BASE`` reads $v^{(0)}$ from the base checkpoint instead, holding the axis
    still while the activations move -- which is what separates the persona
    direction rotating from the representation drifting.

    ``ONPOLICY`` reads the vector :mod:`method.axis_refresh` drew from the
    checkpoint's *own* extraction text. That draw is the one thing here nothing
    else produces -- every other family freezes the text at $M_0$'s by
    construction -- so a missing one is reported as the sweep to run rather
    than left to surface as a bare file-not-found several frames down.
    """
    if axis is ProjectionAxis.ONPOLICY:
        path = onpolicy_vector_path(store, get_weights_id(cfg, t), cfg.trait)
        if not path.exists():
            raise FileNotFoundError(
                f"no on-policy persona vector for {cfg.trait} at t={t} "
                f"({path}); draw it with `python -m method.axis_refresh "
                f"--trait {cfg.trait}` first"
            )
    else:
        source_t = 0 if axis is ProjectionAxis.BASE else t
        path = store.trait_measurement(
            get_weights_id(cfg, source_t),
            cfg.trait,
            Artifacts.persona_vector(cfg.trait),
        )
    return torch.load(path, weights_only=False)[cfg.model.layer]


def measure_probes(
    cfg: TrajectoryConfig,
    t: int,
    store: Store,
    backend: ExecutionBackend,
    probes: Sequence[StepConfig] | None = None,
    *,
    view: DeltaPView = DeltaPView(),
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

    ``view`` is passed straight through to :func:`compute_delta_p`, so one
    checkpoint yields each of the projection differences by being called once
    per view. It takes a single view rather than a set: a caller that wants
    several iterates ``cfg.delta_p.views``, which keeps the results separate
    all the way to the records they are written into.

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
            cfg,
            t,
            store,
            backend,
            train_file,
            training_sample_id(probe, cfg.seed),
            view=view,
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


def predicted_dir(store: Store, wid: str, dp_key: str, source: PredictedSource) -> Path:
    """Where the predicted term's hidden states live for one source.

    ``base`` keeps the unqualified directory it has always had, so a checkpoint
    measured before the source axis existed is still a cache hit.
    """
    name = "delta_p_predicted"
    if source is not PredictedSource.BASE:
        name = f"{name}_{source.value}"
    return store.measurement_dir(wid) / name / dp_key


def _predicted_answers(
    cfg: TrajectoryConfig,
    t: int,
    store: Store,
    backend: ExecutionBackend,
    train_file: Path,
    dp_key: str,
    source: PredictedSource,
) -> Path:
    """The answers DeltaP's predicted term projects, generated once per source.

    Under ``BASE`` these are M_0's, so they are written against the *base*
    checkpoint's id no matter which ``t`` asked for them: one generation serves
    the whole trajectory. Under ``CURRENT`` they belong to checkpoint ``t`` and
    are written against its own id, which is what makes the recomputed series
    cost a generation pass per checkpoint.

    Keyed by ``dp_key`` rather than by the file name in both cases: every
    trajectory's first step resolves to the same base checkpoint, so a
    positional name like ``train_step1`` would serve one dataset's answers to
    every other.
    """
    if source is PredictedSource.BASE:
        t = 0
    wid = get_weights_id(cfg, t)
    out = store.measurement(wid, f"{source.value}_answers_{dp_key}.jsonl")
    if out.exists():
        return out
    rows = [
        json.loads(line) for line in train_file.read_text().splitlines() if line.strip()
    ]
    prompts = [{"messages": r["messages"][:-1]} for r in rows]
    with atomic_file(out) as scratch:
        backend.generate_answers(
            materialize(cfg, t, store, backend), prompts, scratch, cfg
        )
    return out
