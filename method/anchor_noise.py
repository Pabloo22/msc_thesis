r"""How much of $z_t$ is the anchor measurement rather than the model.

    poetry run python -m method.anchor_noise --replicates 3
    poetry run python -m method.anchor_noise --trunk a --checkpoints 0 1 3 6
    poetry run python -m method.anchor_noise --backend mock --local

$z_t = (p, q, \rho, r)$ is read against two artifacts measured *once* on the base
model and then reused by every checkpoint of every run: the persona vector
$v_0$, and ``h_neutral_base`` -- $M_0$'s answers to the neutral prompts, which
each $M_t$ re-reads. This script re-derives both and measures how far $z_t$ moves
when nothing about the weights has changed.

Only one of the two is actually a draw, and it is worth being precise about
which:

$v_0$
    Extracted from responses sampled at temperature 1 -- the vendored evaluator
    samples whenever more than one response per question is asked for, and
    ``extract_n_per_question`` is 10 -- then filtered by a judge. A rerun would
    have produced different text, different scores, and a slightly different
    direction. This is the term the experiment measures.
``h_neutral_base``
    Generated greedily: :mod:`method._generate_worker` defaults to temperature 0
    and no caller overrides it. On a fixed $M_0$ the same 500 prompts return the
    same 500 answers, so this contributes no sampling error to $z_t$ at all.

That asymmetry is deliberate rather than incidental. ``h_neutral_base`` is a
fixed probe re-read by every checkpoint, and decoding it greedily is what makes
movement in $p$ and $q$ attributable to the weights instead of to churn in the
probe; the averaging that multiple samples would buy is already supplied by the
500 *distinct* prompts. The cost is that re-deriving it per replicate is
redundant work -- see :func:`ensure_neutral_answers`.

Why nothing else already answers this:

``method.visualization.latent_audit``
    Reports the spread between runs that landed on the same checkpoint. That is
    an estimate only where the *cache* failed -- a superseded ``weights_id``, a
    miss on a second box -- because a checkpoint measured twice normally reads
    ``latent.json`` back verbatim. Once the store is consistent the spread is
    zero by construction, which says the cache worked, not that the measurement
    is precise.
``method.seed_noise`` / the ``exp2_reseed`` trunk
    Vary the *fine-tuning* seed: different weights, same recipe. Their spread
    contains measurement noise but cannot separate it, and -- because
    ``weights_key`` normalises the seed away at $t = 0$ -- every seed in those
    families reads one cached anchor, so the anchor's own error contributes
    exactly nothing to what they measure.

The anchor error is *common-mode*: one bad draw shifts every checkpoint of every
run together, so it never averages out with more seeds or a longer trajectory.
That is why it is worth paying for separately, and why the summary reports the
spread of within-replicate *differences* ($z_t - z_0$) beside the spread of the
levels -- a term that shifts the whole series cancels in the difference, and the
decay figures read differences.

**A replicate** is one independent re-derivation of the whole base bundle:
``extract_pos.csv`` / ``extract_neg.csv`` (per trait) and ``neutral_answers.jsonl``
(shared by both traits, since neither depends on the trait). Per the above, the
pos/neg halves come back different every time and the neutral answers come back
identical, so a replicate is in effect a draw of $v_0$ alone.

Replicate 0 *is* the production bundle -- the artifacts exp2 and exp3 actually
used -- so it costs no generation and the reported spread says where the numbers
in the thesis sit inside their own sampling distribution. Replicates 1..R-1 are
fresh draws written to a quarantined subdirectory, so nothing already on disk is
touched or invalidated.

Replicates are indexed by *location*, not by any seed: replicate 0 is the one
that reads the ordinary production paths (see :func:`_replicate_dir`), and
nothing here seeds generation. Raising ``--replicates`` therefore resumes rather
than restarts -- every existing replicate short-circuits on its artifacts being
present, and only the new indices are drawn.

No training happens here. Every checkpoint is materialised from adapters that
must already exist, and each replicate costs generation plus forward passes.

Footprint: each (checkpoint, replicate>=1) pair adds an ``h_neutral_base``
directory to that checkpoint's measurement bundle, whose per-sample tensor
dominates at roughly ``n_neutral * d_model`` floats (~7MB at 7B x 500 prompts) --
byte-identical across replicates, for the reason given above.
Measurement bundles are pushed as whole tars, so a large ``--replicates`` on many
checkpoints is bandwidth as well as disk.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import signal
import time
import traceback
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import torch

from method import experiments, report, steps
from method.backends import ExecutionBackend, get_backend, materialize
from method.config import Backend, HNeutralSource, TrajectoryConfig
from method.latent import latent_record
from method.notify import Heartbeat, Notifier
from method.steps import Artifacts
from method.store import Store, StoreSelection, atomic_dir, atomic_file, get_weights_id
from method.sync import Syncer, format_unsynced
from method.utils import (
    DOTENV_PATH,
    anchor_noise_path,
    check_env_vars,
    load_dotenv,
)

logger = logging.getLogger("anchor_noise")

#: Subdirectory of a checkpoint's measurement bundle holding re-drawn anchors.
#: Nested inside the bundle rather than parallel to it so that
#: :meth:`method.sync.Syncer.push_measurement`, which tars the whole directory,
#: carries replicates without needing to learn about them.
REPLICATE_SUBDIR = "anchor_replicates"

#: The replicate that is not a re-draw: the artifacts the production
#: measurement path already wrote, read from their ordinary locations. Keeping
#: it in the family (rather than comparing re-draws only among themselves) is
#: what makes the spread interpretable as "where the number we published sits".
PRODUCTION_REPLICATE = 0

#: The latent components, in the order the tables print them.
COMPONENTS = ("p", "q", "rho", "r")

#: Which ``h_neutral`` source this measures. ``base`` is the anchored one --
#: $M_0$'s answers, frozen and re-read by every checkpoint -- and therefore the
#: only one with an anchor term at all. Under ``current`` each checkpoint
#: samples its own answers, which is a per-checkpoint noise source and a
#: different question.
SOURCE = HNeutralSource.BASE.value


# --- where a replicate's artifacts live ------------------------------------


def _replicate_dir(base: Path, replicate: int) -> Path:
    """``base`` itself for the production replicate, a namespaced child otherwise."""
    if replicate == PRODUCTION_REPLICATE:
        return base
    return base / REPLICATE_SUBDIR / f"rep{replicate:02d}"


def shared_dir(store: Store, wid: str, replicate: int) -> Path:
    """Where a replicate's trait-*independent* artifacts live for ``wid``.

    The neutral answers and the hidden states read off them involve no trait, so
    one draw serves every trait -- which is also why they must not be nested
    under a trait directory, or a second trait would re-generate them and the
    two traits would silently sit on different draws.
    """
    return _replicate_dir(store.measurement_dir(wid), replicate)


def trait_dir(store: Store, wid: str, trait: str, replicate: int) -> Path:
    """Where a replicate's trait-*dependent* artifacts live for ``wid``."""
    return _replicate_dir(store.trait_measurement_dir(wid, trait), replicate)


def persona_vector_path(
    store: Store, cfg: TrajectoryConfig, t: int, replicate: int
) -> Path:
    """The persona vector file for one checkpoint under one replicate.

    The production replicate points at the artifact
    :func:`method.steps.extract_persona_vector` writes, which sits directly in
    the trait directory. Re-draws keep theirs in a ``vector/`` subdirectory: the
    vendored extractor emits several files, and giving them a directory of their
    own means :func:`~method.store.atomic_dir` alone makes the write
    all-or-nothing, with no promotion step to get wrong.
    """
    wid = get_weights_id(cfg, t)
    name = Artifacts.persona_vector(cfg.trait)
    if replicate == PRODUCTION_REPLICATE:
        return store.trait_measurement(wid, cfg.trait, name)
    return trait_dir(store, wid, cfg.trait, replicate) / "vector" / name


def h_neutral_path(store: Store, cfg: TrajectoryConfig, t: int, replicate: int) -> Path:
    """The mean hidden state over the neutral answers, for one replicate."""
    directory = shared_dir(store, get_weights_id(cfg, t), replicate)
    return directory / Artifacts.h_neutral(SOURCE) / "mean_by_layer.pt"


# --- drawing a replicate ---------------------------------------------------


def ensure_neutral_answers(
    cfg: TrajectoryConfig, store: Store, backend: ExecutionBackend, replicate: int
) -> Path:
    """$M_0$'s answers to the neutral prompts, re-derived for this replicate.

    Re-*derived*, not re-sampled: decoding is greedy (see the module docstring),
    so this regenerates text identical to the production bundle's and every
    replicate's ``h_neutral`` term is the same. The 500 generations it costs per
    replicate buy nothing, and reusing the production file would be equivalent;
    it is left as an honest re-derivation so that the zero contribution is
    something the sweep *measures* rather than something it assumes.
    """
    base_wid = get_weights_id(cfg, 0)
    out = shared_dir(store, base_wid, replicate) / Artifacts.NEUTRAL_ANSWERS
    if out.exists():
        logger.info("[skip] neutral answers for replicate %d already drawn", replicate)
        return out
    prompts = [
        json.loads(line)
        for line in steps.neutral_prompts_file(cfg).read_text().splitlines()
        if line.strip()
    ][: cfg.latent.n_neutral]
    logger.info("Replicate %d: sampling %d neutral answers", replicate, len(prompts))
    with atomic_file(out) as scratch:
        backend.generate_answers(
            materialize(cfg, 0, store, backend), prompts, scratch, cfg
        )
    return out


def ensure_extract_csvs(
    cfg: TrajectoryConfig, store: Store, backend: ExecutionBackend, replicate: int
) -> tuple[Path, Path]:
    """The judged pos/neg responses $v$ is extracted from, for this replicate.

    Drawn on $M_0$ and then frozen, exactly as the production path freezes them
    (see :func:`method.steps.extract_persona_vector`): every checkpoint of the
    replicate re-reads this same text, so a replicate varies the *text and its
    judge scores*, never which model produced them.

    This is the half of the bundle that genuinely varies between replicates, and
    therefore the whole of what the reported spread is a spread *of*.
    """
    base_wid = get_weights_id(cfg, 0)
    directory = trait_dir(store, base_wid, cfg.trait, replicate)
    pos, neg = directory / Artifacts.EXTRACT_POS, directory / Artifacts.EXTRACT_NEG
    if pos.exists() and neg.exists():
        logger.info(
            "[skip] pos/neg responses for replicate %d (trait=%s) already drawn",
            replicate,
            cfg.trait,
        )
        return pos, neg

    model_path = materialize(cfg, 0, store, backend)
    for kind, path in (("pos", pos), ("neg", neg)):
        if path.exists():
            continue
        logger.info(
            "Replicate %d: sampling %s responses (trait=%s)", replicate, kind, cfg.trait
        )
        with atomic_file(path) as scratch:
            backend.eval_persona(
                model_path,
                cfg.trait,
                scratch,
                cfg,
                version="extract",
                persona_instruction_type=kind,
                progress_dir=steps.eval_progress_dir(path),
                model_id=base_wid,
            )
    return pos, neg


def ensure_persona_vector(
    cfg: TrajectoryConfig,
    t: int,
    store: Store,
    backend: ExecutionBackend,
    replicate: int,
    model_path: str,
) -> Path:
    """$v_t$ under one replicate's pos/neg draw.

    The production replicate delegates to the ordinary measurement path, so it
    reads the artifact exp2/exp3 used where one exists and computes it exactly
    as they would where one does not.
    """
    if replicate == PRODUCTION_REPLICATE:
        return steps.extract_persona_vector(cfg, t, store, backend)

    out = persona_vector_path(store, cfg, t, replicate)
    if out.exists():
        return out
    pos, neg = ensure_extract_csvs(cfg, store, backend, replicate)
    with atomic_dir(out.parent) as scratch:
        backend.extract_vector(
            model_path,
            cfg.trait,
            pos,
            neg,
            scratch,
            cfg.eval.persona_vector_threshold,
        )
    return out


def ensure_h_neutral(
    cfg: TrajectoryConfig,
    t: int,
    store: Store,
    backend: ExecutionBackend,
    replicate: int,
    model_path: str,
) -> Path:
    """Checkpoint ``t``'s hidden states over one replicate's neutral answers."""
    if replicate == PRODUCTION_REPLICATE:
        measured = steps.measure_h_neutral(cfg, t, store, backend)
        if SOURCE not in measured:
            raise ValueError(
                f"{cfg.name!r} does not measure the {SOURCE!r} h_neutral source "
                f"(latent.h_neutral_source={cfg.latent.h_neutral_source.value!r}); "
                "the anchor term is defined only for it"
            )
        return measured[SOURCE] / "mean_by_layer.pt"

    out = h_neutral_path(store, cfg, t, replicate)
    if out.exists():
        return out
    answers = ensure_neutral_answers(cfg, store, backend, replicate)
    with atomic_dir(out.parent) as scratch:
        backend.hidden_states(model_path, answers, cfg.model.layer, scratch)
    return out


def replicate_latent(
    cfg: TrajectoryConfig, t: int, store: Store, replicate: int
) -> dict[str, float]:
    """$z_t$ assembled from one replicate's own $v_0$, $v_t$ and $h$.

    Deliberately not cached to a ``latent.json``: the inputs are already on
    disk, the arithmetic is microseconds, and a cached copy keyed by anything
    less than the full draw is the exact failure this experiment exists to
    quantify.
    """
    layer = cfg.model.layer

    def at(path: Path) -> torch.Tensor:
        return torch.load(path, weights_only=False)[layer]

    return latent_record(
        at(persona_vector_path(store, cfg, 0, replicate)),
        at(persona_vector_path(store, cfg, t, replicate)),
        at(h_neutral_path(store, cfg, t, replicate)),
    )


# --- the run ---------------------------------------------------------------


def measure(
    cfgs: Sequence[TrajectoryConfig],
    checkpoints: Sequence[int],
    replicates: Sequence[int],
    store: Store,
    backend: ExecutionBackend,
    *,
    syncer: Syncer | None = None,
) -> pd.DataFrame:
    """One row per (trait, checkpoint, replicate), carrying that draw's $z_t$.

    ``cfgs`` is one config per trait over a single fixed step sequence; they
    must resolve to the same checkpoints, which they do whenever they differ
    only in trait (``weights_key`` excludes it).

    Checkpoints are the outer loop because materialising one is a merge chain
    while everything inside it is a forward pass: visiting a checkpoint once and
    drawing every replicate against it is the difference between R merges and
    one. ``t = 0`` is visited first regardless of the order asked for, since
    every later $z$ is read against that replicate's $v_0$.
    """
    if PRODUCTION_REPLICATE not in replicates:
        raise ValueError(
            f"replicate {PRODUCTION_REPLICATE} (the production bundle) anchors the "
            "comparison and must be included"
        )
    if 0 not in checkpoints:
        raise ValueError("checkpoint 0 must be included: z_t is read against v_0")
    _check_shared_checkpoints(cfgs)

    rows = []
    for t in sorted(set(checkpoints)):
        wid = get_weights_id(cfgs[0], t)
        logger.info("--- checkpoint t=%d (%s) ---", t, wid)
        model_path = materialize(cfgs[0], t, store, backend)
        for cfg in cfgs:
            for replicate in sorted(set(replicates)):
                ensure_persona_vector(cfg, t, store, backend, replicate, model_path)
                ensure_h_neutral(cfg, t, store, backend, replicate, model_path)
                z = replicate_latent(cfg, t, store, replicate)
                logger.info(
                    "z_%d[%s, rep%d] = %s",
                    t,
                    cfg.trait,
                    replicate,
                    {k: round(v, 4) for k, v in z.items()},
                )
                rows.append(
                    {
                        "trait": cfg.trait,
                        "t": t,
                        "weights_id": wid,
                        "replicate": replicate,
                        **z,
                    }
                )
        # Pushed per checkpoint rather than at the end: a preemption partway
        # through a six-checkpoint sweep then keeps the checkpoints it finished.
        if syncer is not None:
            syncer.push_measurement(wid)
        if t:
            # Nothing later reads this one's full weights -- the walk forward
            # rebuilds from adapters -- and a 7B checkpoint is ~15GB.
            store.evict_merged(wid)
    return pd.DataFrame(rows)


def _check_shared_checkpoints(cfgs: Sequence[TrajectoryConfig]) -> None:
    """Fail loudly if the per-trait configs do not sit on identical weights.

    They are meant to differ only in trait, which ``weights_key`` excludes, so a
    mismatch means something that *does* change the weights diverged -- and the
    traits would then be measuring different models under one table.
    """
    if not cfgs:
        raise ValueError("no configs to measure")
    chains = {
        cfg.trait: tuple(get_weights_id(cfg, t) for t in range(len(cfg.steps) + 1))
        for cfg in cfgs
    }
    if len(set(chains.values())) > 1:
        raise ValueError(f"configs do not share a checkpoint chain: {chains}")


# --- reading the result ----------------------------------------------------


def spread(frame: pd.DataFrame) -> pd.DataFrame:
    """Per (trait, checkpoint, component): how far the replicates disagree.

    ``sigma_level`` is the standard deviation of $z_t$ across draws -- the error
    on a single reported level. ``sigma_delta`` is the standard deviation of
    $z_t - z_0$ computed *within* each draw, which is the relevant one wherever
    a figure reads a change rather than a level: a term common to every
    checkpoint of a draw cancels there, and the anchor is exactly such a term.

    A gap between the two is therefore not a contradiction but the measurement
    of how common-mode the anchor error is.
    """
    base = frame[frame["t"] == 0].set_index(["trait", "replicate"])
    rows = []
    for (trait, t), group in frame.groupby(["trait", "t"]):
        indexed = group.set_index("replicate")
        for component in COMPONENTS:
            levels = indexed[component]
            deltas = levels - base[component].xs(trait, level="trait")
            rows.append(
                {
                    "trait": trait,
                    "t": t,
                    "component": component,
                    "replicates": int(levels.count()),
                    "production": float(levels.get(PRODUCTION_REPLICATE, float("nan"))),
                    "mean": float(levels.mean()),
                    "sigma_level": float(levels.std(ddof=1)),
                    "range_level": float(levels.max() - levels.min()),
                    "sigma_delta": float(deltas.std(ddof=1)),
                }
            )
    return pd.DataFrame(rows).sort_values(["trait", "component", "t"])


def against_drift(frame: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Per (trait, component): the anchor noise beside the drift it sits against.

    ``drift`` is measured on the production replicate alone, over the same
    checkpoints this run visited, so the ratio compares two numbers from one
    table rather than borrowing a drift figure measured elsewhere.

    ``ratio_delta`` is the number to quote for a decay or drift figure and
    ``ratio_level`` for anything reading $z_t$ directly. At $\\ge 1$ the
    component carries no usable signal over this span.
    """
    deepest = int(frame["t"].max())
    production = frame[frame["replicate"] == PRODUCTION_REPLICATE]
    rows = []
    for trait, group in production.groupby("trait"):
        base = group[group["t"] == 0]
        end = group[group["t"] == deepest]
        per_trait = table[table["trait"] == trait]
        for component in COMPONENTS:
            drift = float(end[component].iloc[0] - base[component].iloc[0])
            cell = per_trait[per_trait["component"] == component]
            # The worst checkpoint, not the mean: a noise floor quoted for a
            # figure has to hold at every point the figure draws.
            level = float(cell["sigma_level"].max())
            delta = float(cell["sigma_delta"].max())
            rows.append(
                {
                    "trait": trait,
                    "component": component,
                    "sigma_level": level,
                    "sigma_delta": delta,
                    "drift": drift,
                    "ratio_level": abs(level / drift) if drift else float("inf"),
                    "ratio_delta": abs(delta / drift) if drift else float("inf"),
                }
            )
    return pd.DataFrame(rows)


def write_summary(
    cfgs: Sequence[TrajectoryConfig],
    frame: pd.DataFrame,
    label: str,
    *,
    mock: bool,
) -> Path:
    """Persist the measurements next to the trajectories, self-contained.

    The inputs stay in the store, keyed by content; this is the copy a plotting
    machine can read with only the trajectories root synced, exactly as
    :func:`method.probe_base.write_summary` is for base probes.
    """
    base_wid = get_weights_id(cfgs[0], 0)
    path = anchor_noise_path(base_wid, label, mock=mock)
    table = spread(frame)
    payload = {
        "base_weights_id": base_wid,
        "label": label,
        "seed": cfgs[0].seed,
        "model": dataclasses.asdict(cfgs[0].model),
        "traits": [cfg.trait for cfg in cfgs],
        "h_neutral_source": SOURCE,
        "n_neutral": cfgs[0].latent.n_neutral,
        "steps": [step.dataset_id for step in cfgs[0].steps],
        "latents": frame.to_dict(orient="records"),
        "spread": table.to_dict(orient="records"),
        "against_drift": against_drift(frame, table).to_dict(orient="records"),
    }
    with atomic_file(path) as scratch:
        scratch.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Anchor noise -> %s", path)
    return path


def _show(title: str, table: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    print(table.to_string(index=False) if not table.empty else "(empty)")


# --- CLI -------------------------------------------------------------------


def run(
    *,
    trunk: str,
    checkpoints: Sequence[int],
    n_replicates: int,
    traits: Sequence[str],
    seed: int,
    backend_kind: Backend,
    dtype: str,
    local: bool,
) -> pd.DataFrame:
    store = Store.for_backend(backend_kind)
    backend = get_backend(backend_kind, dtype=dtype)
    cfgs = experiments.build_anchor_noise_configs(
        trunk=trunk, seed=seed, measure_traits=traits, local=local
    )
    replicates = list(range(n_replicates))

    store.evict_orphaned_merged()
    syncer = Syncer.from_env(store)
    if syncer is not None:
        # Adapters for the whole trunk, and whatever measurements already exist
        # at its checkpoints -- the production replicate is read, not recomputed,
        # wherever exp2 already measured it. One config's closure covers every
        # trait: they share a chain, which ``measure`` then verifies.
        syncer.pull_before_run(StoreSelection.for_config(cfgs[0]))

    logger.info(
        "Anchor noise: trunk %s, checkpoints %s, %d replicate(s) x %d trait(s)",
        trunk,
        list(checkpoints),
        n_replicates,
        len(cfgs),
    )
    frame = measure(cfgs, checkpoints, replicates, store, backend, syncer=syncer)

    label = f"trunk_{trunk}{'_local' if local else ''}"
    summary = write_summary(cfgs, frame, label, mock=backend_kind is Backend.MOCK)
    if syncer is not None:
        syncer.push_anchor_noise(summary)

    table = spread(frame)
    _show("z_t per replicate", frame)
    _show("spread across replicates", table)
    if frame["t"].max() > 0:
        _show("anchor noise against drift", against_drift(frame, table))
    else:
        # The drift column would be t=0 against itself, i.e. zero, and every
        # ratio would print as inf. Saying so beats a table of infinities.
        logger.warning("only t=0 measured: there is no drift to compare against")
    if n_replicates < 2:
        logger.warning(
            "--replicates 1 measures no spread; the tables above are placeholders"
        )

    if syncer is not None and syncer.unsynced:
        logger.warning("%s", format_unsynced(syncer.unsynced))
    store.evict_all_merged()
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trunk",
        default="a",
        choices=sorted(experiments.EXP2_TRUNKS),
        help="which exp2 decay trunk's checkpoints to measure on (default: a)",
    )
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=list(experiments.ANCHOR_NOISE_CHECKPOINTS),
        help="steps along the trunk to measure; 0 is required and implied",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=3,
        help=(
            "total draws of the base bundle, including the production one "
            "(so 3 means 2 fresh draws); 2 is the minimum that measures anything"
        ),
    )
    parser.add_argument(
        "--traits",
        nargs="+",
        default=list(experiments.MEASURE_TRAITS),
        help="traits to measure; the neutral answers are shared between them",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=experiments.EXP2_SEED,
        help="the trunk's fine-tuning seed; picks which adapters are replayed",
    )
    parser.add_argument(
        "--backend", type=Backend, choices=list(Backend), default=Backend.REAL
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="use float16 on pre-Ampere GPUs, which only emulate bfloat16",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="use the small local proxy model and capped example counts",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if args.replicates < 1:
        parser.error("--replicates must be at least 1")

    load_dotenv(DOTENV_PATH)
    if args.backend is Backend.REAL:
        required = ["HF_TOKEN"]
        if not args.local:
            # Fresh pos/neg draws are judged, which is what makes them a draw.
            required.append("OPENAI_API_KEY")
        check_env_vars(required)

    signal.signal(signal.SIGTERM, _die_on_sigterm)
    _run_and_report(args)


def _run_and_report(args: argparse.Namespace) -> None:
    """:func:`run`, with the outcome mailed out and a watchdog kept fed.

    Mirrors :func:`method.probe_base._run_and_report`: this is hours of the same
    rented GPU, just as unattended, and there are no per-checkpoint reports to
    send, so the flat job report is the whole of it.
    """
    notifier = Notifier.from_env()
    heartbeat = Heartbeat.from_env()
    logger.info("%s; %s", notifier.describe(), heartbeat.describe())

    title = f"anchor_noise (trunk {args.trunk}, {args.replicates} replicate(s))"
    detail = (
        f"trunk: {args.trunk}\ncheckpoints: {list(args.checkpoints)}\n"
        f"replicates: {args.replicates}\ntraits: {list(args.traits)}"
    )
    started = time.monotonic()
    with heartbeat:
        try:
            run(
                trunk=args.trunk,
                checkpoints=args.checkpoints,
                n_replicates=args.replicates,
                traits=args.traits,
                seed=args.seed,
                backend_kind=args.backend,
                dtype=args.dtype,
                local=args.local,
            )
        except BaseException as exc:
            subject, body = report.job_report(
                title,
                elapsed=time.monotonic() - started,
                detail=detail,
                error=exc,
                traceback_text=traceback.format_exc(),
            )
            notifier.send(subject, body, throttle_key="anchor_noise:failed")
            raise
        subject, body = report.job_report(
            title, elapsed=time.monotonic() - started, detail=detail
        )
        notifier.send(subject, body, throttle_key="anchor_noise:ok")


def _die_on_sigterm(signum: int, _frame: object) -> None:
    """Turn a polite kill into an exception, so the failure email still goes."""
    raise KeyboardInterrupt(f"received signal {signum}")


if __name__ == "__main__":
    main()
