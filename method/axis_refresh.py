r"""Is the axis still right once the extraction text is stale?

    poetry run python -m method.axis_refresh
    poetry run python -m method.axis_refresh --trunk a --checkpoints 0 6
    poetry run python -m method.axis_refresh --backend mock --local

$v^{(t)}$ is not extracted the way \citet{chen2025persona_vectors} extract a
persona vector. Their procedure generates trait-positive and trait-negative
responses *from the model being measured*; ours generates them once, from $M_0$,
and has every later checkpoint re-encode that same fixed text (see
:func:`method.steps.extract_persona_vector`). Two arguments justify the freeze:

*comparability*
    A refreshed extraction set would move for two reasons at once -- the
    encoder's geometry changed, and the text changed -- so $\rho_t =
    \cos(v^{(0)}, v^{(t)})$ would no longer be a statement about the model.
*a well-defined contrast*
    A checkpoint trained hard toward a trait may stop producing a credible
    *negative* response under a negative persona instruction, or stop producing
    a coherent one. The vendored filter would then keep fewer and stranger
    pairs, and the vector would change meaning without anything looking wrong.

Both are arguments. This script measures them, and it is the only thing in the
repo that can: every other family holds the extraction text fixed by
construction, so the quantity being defended never varies within them.

At each checkpoint it draws the extraction set a second time -- from $M_t$
itself, judged and filtered exactly as the production path judges and filters
$M_0$'s -- extracts a second vector from it, and reports how far the two
disagree.

**The $t = 0$ row is the control, not a result.** There, the re-draw comes from
the same model, so the two vectors differ only by the sampling of the extraction
responses (temperature 1) and the judge's scoring of them. That agreement is the
floor: a cosine of 0.97 at $t = 6$ says nothing until the floor is known to be
0.99 rather than 0.97. It is the same kind of draw :mod:`method.anchor_noise`
calls a replicate, and it is drawn here rather than read from there so that one
script's storage layout is not a dependency of another's.

What the two tables answer:

``cos_refresh``
    $\cos(v^{(t)}_{\text{frozen}}, v^{(t)}_{\text{on-policy}})$. At the floor,
    the freeze costs nothing measurable and the limitation is closed. Well below
    it, the frozen axis and the model's own axis have parted company, and the
    thesis has to say so.
``rho_frozen`` vs ``rho_onpolicy``
    Both against $v^{(0)}$. The first is the $\rho_t$ that $z_t$ already
    reports; the second is what $\rho_t$ would have been under the paper's own
    procedure. A gap here is the freeze changing a *published* number rather
    than an internal one.
``n_effective`` and the pass rates
    How many of the 1000 question-instruction-sample pairs survive the vendored
    mask, and which half of it does the filtering. The frozen column is constant
    in $t$ by construction -- the mask is computed from $M_0$'s judged CSVs and
    then never recomputed -- so the on-policy column falling away from it *is*
    the degenerate case the freeze was chosen to avoid, measured rather than
    argued.

Deliberately out of scope: what a refreshed axis would do to $\Delta P$. That is
a different question (does the staleness change a *prediction*, not does it
change the ruler), it is answered per probe dataset rather than per checkpoint,
and it needs the cached probe activations that only a measured trunk has. The
four-corner square in :class:`method.config.DeltaPView` is where that belongs.

No training happens here. Every checkpoint is materialised from adapters that
must already exist. The cost is a full extraction draw per (checkpoint, trait):
20 questions x 5 persona instructions x 10 generations = 1000 responses on each
of the positive and negative sides, 4000 judge calls over them, and then a
forward pass per surviving pair for the vector itself. Unlike the neutral
answers :mod:`method.anchor_noise` re-draws, none of that is shared between
traits -- the questions, the instructions and the judge rubric are all
trait-specific -- which is why the default is one trait (see
:data:`method.experiments.AXIS_REFRESH_TRAITS`).

Footprint: each (checkpoint, trait) adds two ~1000-row CSVs and one persona
vector (``n_layers x d_model`` floats, ~0.4MB at 7B) to that checkpoint's
measurement bundle. Bundles are pushed as whole tars, so this is bandwidth as
well as disk.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import signal
import sys
import time
import traceback
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import torch

from method import experiments, report, steps
from method.backends import ExecutionBackend, get_backend, materialize
from method.config import Backend, TrajectoryConfig
from method.latent import cosine
from method.notify import Heartbeat, Notifier
from method.steps import Artifacts
from method.store import Store, StoreSelection, atomic_dir, atomic_file, get_weights_id
from method.sync import Syncer, format_unsynced
from method.utils import (
    DOTENV_PATH,
    PERSONA_VECTORS_DIR,
    axis_refresh_path,
    check_env_vars,
    load_dotenv,
)

logger = logging.getLogger("axis_refresh")

#: Subdirectory of a checkpoint's *trait* measurement bundle holding the
#: re-drawn extraction set and the vector taken from it. Nested inside the
#: bundle rather than parallel to it so that
#: :meth:`method.sync.Syncer.push_measurement`, which tars the whole directory,
#: carries it without needing to learn about it -- the arrangement
#: :mod:`method.anchor_noise` uses for its replicates. Under the trait
#: directory, unlike those, because an extraction set *is* trait-specific.
REFRESH_SUBDIR = "axis_refresh"

#: The checkpoint whose re-draw is the sampling floor rather than a measurement.
FLOOR_CHECKPOINT = 0

#: Minimum judge coherence for a response to be usable, on either side. Fixed at
#: 50 in the vendored filter rather than configurable, so it is a constant here
#: too; ``cfg.eval.persona_vector_threshold`` is the trait-score half.
COHERENCE_THRESHOLD = 50


# --- where a re-draw's artifacts live --------------------------------------


def refresh_dir(store: Store, wid: str, trait: str) -> Path:
    """Where checkpoint ``wid``'s own extraction draw for ``trait`` lives."""
    return store.trait_measurement_dir(wid, trait) / REFRESH_SUBDIR


def onpolicy_extract_paths(
    store: Store, cfg: TrajectoryConfig, t: int
) -> tuple[Path, Path]:
    """The judged positive and negative CSVs ``M_t`` produced for itself."""
    directory = refresh_dir(store, get_weights_id(cfg, t), cfg.trait)
    return directory / Artifacts.EXTRACT_POS, directory / Artifacts.EXTRACT_NEG


def onpolicy_vector_path(store: Store, cfg: TrajectoryConfig, t: int) -> Path:
    """The persona vector extracted from that draw.

    Kept in a ``vector/`` subdirectory for the reason
    :func:`method.anchor_noise.persona_vector_path` gives: the vendored
    extractor emits three files, and giving them a directory of their own means
    :func:`~method.store.atomic_dir` alone makes the write all-or-nothing, with
    no promotion step to get wrong.
    """
    directory = refresh_dir(store, get_weights_id(cfg, t), cfg.trait)
    return directory / "vector" / Artifacts.persona_vector(cfg.trait)


def frozen_extract_paths(store: Store, cfg: TrajectoryConfig) -> tuple[Path, Path]:
    """The production extraction set: ``M_0``'s answers, judged once.

    Read from the base checkpoint's bundle at every ``t``, which is exactly what
    :func:`method.steps.extract_persona_vector` does and the whole of what this
    script is checking.
    """
    base_wid = get_weights_id(cfg, 0)
    return (
        store.trait_measurement(base_wid, cfg.trait, Artifacts.EXTRACT_POS),
        store.trait_measurement(base_wid, cfg.trait, Artifacts.EXTRACT_NEG),
    )


# --- drawing the checkpoint's own axis -------------------------------------


def ensure_onpolicy_extract_csvs(
    cfg: TrajectoryConfig,
    t: int,
    store: Store,
    backend: ExecutionBackend,
    model_path: str,
) -> tuple[Path, Path]:
    """``M_t``'s own positive and negative extraction responses, judged.

    The same call the production path makes at ``t = 0``
    (:func:`method.steps.extract_persona_vector`), pointed at ``M_t`` instead of
    ``M_0``. That is the whole of the difference under test: same questions,
    same persona instructions, same judge, same rubric -- a different model
    answering them.

    The two sides must stay row-aligned, because the vendored filter masks them
    as *pairs*. They are, by construction: both enumerate the same 20 questions
    against the same 5 instruction slots with the same
    ``extract_n_per_question`` samples, so row ``i`` names the same cell of that
    grid on either side.
    """
    pos, neg = onpolicy_extract_paths(store, cfg, t)
    if pos.exists() and neg.exists():
        logger.info(
            "[skip] on-policy extraction set at t=%d already drawn (trait=%s)",
            t,
            cfg.trait,
        )
        return pos, neg

    for kind, path in (("pos", pos), ("neg", neg)):
        if path.exists():
            continue
        logger.info(
            "t=%d: sampling %s responses from M_t (trait=%s)", t, kind, cfg.trait
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
            )
    return pos, neg


def ensure_onpolicy_vector(
    cfg: TrajectoryConfig,
    t: int,
    store: Store,
    backend: ExecutionBackend,
    model_path: str,
) -> Path:
    """$v^{(t)}$ as the paper's own procedure would produce it at ``M_t``."""
    out = onpolicy_vector_path(store, cfg, t)
    if out.exists():
        logger.info("[skip] on-policy vector at t=%d already extracted", t)
        return out
    pos, neg = ensure_onpolicy_extract_csvs(cfg, t, store, backend, model_path)
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


# --- reading the filter that produced each vector --------------------------


def _vendored_effective(
    pos_csv: Path, neg_csv: Path, trait: str, threshold: int
) -> int:
    """How many pairs the vendored filter keeps, asked of the vendored filter.

    ``generate_vec.get_persona_effective`` is the function that decides which
    responses a persona vector is averaged over, so it is the authority on how
    many there were. This script exists to report exactly that, and a count
    computed alongside the vendored one could drift from it while still looking
    right -- the one failure the check cannot afford.

    ``generate_vec`` is a script rather than an importable module, so its
    directory has to be a ``sys.path`` root first; :mod:`method._vector_worker`
    does the same dance for the same reason.
    """
    if str(PERSONA_VECTORS_DIR) not in sys.path:
        sys.path.insert(0, str(PERSONA_VECTORS_DIR))
    from generate_vec import get_persona_effective

    pos_effective, *_ = get_persona_effective(
        str(pos_csv), str(neg_csv), trait, threshold
    )
    return len(pos_effective)


def filter_counts(
    pos_csv: Path, neg_csv: Path, trait: str, threshold: int
) -> dict[str, float]:
    """How many pairs survived the filter, and which clause did the filtering.

    ``n_effective`` comes from the vendored filter itself and is the number the
    vector was actually averaged over. The three rates beside it are this
    module's own breakdown of that mask, and they are marginal rather than
    disjoint: a pair can fail several clauses at once, so they do not sum to the
    rejection rate.

    They are worth computing because *which* clause bites is the finding. The
    freeze is defended on the claim that a drifted model stops producing
    credible negatives, so a collapse concentrated in ``neg_pass`` supports that
    defence and one spread evenly across ``pos_pass`` and ``coherence_pass``
    does not.

    Their conjunction is checked against the vendored count rather than trusted,
    so a breakdown that has drifted from the mask it describes fails here
    instead of quietly misattributing the cause.
    """
    pos, neg = pd.read_csv(pos_csv), pd.read_csv(neg_csv)
    if len(pos) != len(neg):
        raise ValueError(
            f"extraction sides are not row-aligned: {len(pos)} positive rows in "
            f"{pos_csv} against {len(neg)} negative rows in {neg_csv}; the "
            "vendored filter masks them as pairs and would silently truncate"
        )
    n_effective = _vendored_effective(pos_csv, neg_csv, trait, threshold)

    pos_pass = pos[trait] >= threshold
    neg_pass = neg[trait] < 100 - threshold
    coherence_pass = (pos["coherence"] >= COHERENCE_THRESHOLD) & (
        neg["coherence"] >= COHERENCE_THRESHOLD
    )
    combined = int((pos_pass & neg_pass & coherence_pass).sum())
    if combined != n_effective:
        raise ValueError(
            f"the clause breakdown keeps {combined} pairs where the vendored "
            f"filter keeps {n_effective}; generate_vec.get_persona_effective has "
            "changed and the per-clause rates below it would misattribute the "
            "cause"
        )
    return {
        "n_pairs": len(pos),
        "n_effective": n_effective,
        "effective_rate": n_effective / len(pos),
        "pos_pass": float(pos_pass.mean()),
        "neg_pass": float(neg_pass.mean()),
        "coherence_pass": float(coherence_pass.mean()),
        "pos_trait_mean": float(pos[trait].mean()),
        "neg_trait_mean": float(neg[trait].mean()),
    }


# --- the comparison --------------------------------------------------------


def compare(cfg: TrajectoryConfig, t: int, store: Store) -> dict[str, float]:
    """One row: the frozen axis and the on-policy axis at checkpoint ``t``.

    Deliberately not cached to a JSON of its own. Both vectors and both CSVs are
    already on disk, the arithmetic is microseconds, and a cached copy keyed by
    anything less than the full draw is precisely the staleness this script
    exists to detect.
    """
    layer = cfg.model.layer

    def at(path: Path) -> torch.Tensor:
        return torch.load(path, weights_only=False)[layer]

    v0 = at(
        store.trait_measurement(
            get_weights_id(cfg, 0), cfg.trait, Artifacts.persona_vector(cfg.trait)
        )
    )
    frozen = at(
        store.trait_measurement(
            get_weights_id(cfg, t), cfg.trait, Artifacts.persona_vector(cfg.trait)
        )
    )
    onpolicy = at(onpolicy_vector_path(store, cfg, t))

    threshold = cfg.eval.persona_vector_threshold
    frozen_counts = filter_counts(
        *frozen_extract_paths(store, cfg), cfg.trait, threshold
    )
    onpolicy_counts = filter_counts(
        *onpolicy_extract_paths(store, cfg, t), cfg.trait, threshold
    )
    return {
        "cos_refresh": cosine(frozen, onpolicy),
        "rho_frozen": cosine(v0, frozen),
        "rho_onpolicy": cosine(v0, onpolicy),
        "r_frozen": float(frozen.float().norm()),
        "r_onpolicy": float(onpolicy.float().norm()),
        **{f"frozen_{k}": v for k, v in frozen_counts.items()},
        **{f"onpolicy_{k}": v for k, v in onpolicy_counts.items()},
    }


def measure(
    cfgs: Sequence[TrajectoryConfig],
    checkpoints: Sequence[int],
    store: Store,
    backend: ExecutionBackend,
    *,
    syncer: Syncer | None = None,
) -> pd.DataFrame:
    """One row per (trait, checkpoint), carrying both axes and both filters.

    ``cfgs`` is one config per trait over a single fixed step sequence; they
    must resolve to the same checkpoints, which they do whenever they differ
    only in trait (``weights_key`` excludes it).

    Checkpoints are the outer loop for the reason
    :func:`method.anchor_noise.measure` gives: materialising one is a merge
    chain and everything inside it is generation or a forward pass, so visiting
    a checkpoint once and doing every trait against it is the difference between
    one merge and several.

    The frozen vector is obtained through the ordinary measurement path, so a
    checkpoint exp2 already measured is read rather than recomputed -- and one it
    did not is computed exactly as exp2 would have.
    """
    if FLOOR_CHECKPOINT not in checkpoints:
        raise ValueError(
            f"checkpoint {FLOOR_CHECKPOINT} must be included: the re-draw there is "
            "the sampling floor every later checkpoint is read against"
        )
    _check_shared_checkpoints(cfgs)

    rows = []
    for t in sorted(set(checkpoints)):
        wid = get_weights_id(cfgs[0], t)
        logger.info("--- checkpoint t=%d (%s) ---", t, wid)
        model_path = materialize(cfgs[0], t, store, backend)
        for cfg in cfgs:
            steps.extract_persona_vector(cfg, t, store, backend)
            ensure_onpolicy_vector(cfg, t, store, backend, model_path)
            record = compare(cfg, t, store)
            logger.info(
                "t=%d [%s]: cos(frozen, on-policy) = %.4f, effective pairs "
                "%d frozen / %d on-policy",
                t,
                cfg.trait,
                record["cos_refresh"],
                record["frozen_n_effective"],
                record["onpolicy_n_effective"],
            )
            rows.append({"trait": cfg.trait, "t": t, "weights_id": wid, **record})
        # Pushed per checkpoint rather than at the end, so a preemption partway
        # through keeps the checkpoints it finished.
        if syncer is not None:
            syncer.push_measurement(wid)
        if t:
            # Nothing later reads this one's full weights -- the walk forward
            # rebuilds from adapters -- and a 7B checkpoint is ~15GB.
            store.evict_merged(wid)
    return pd.DataFrame(rows)


def _check_shared_checkpoints(cfgs: Sequence[TrajectoryConfig]) -> None:
    """Fail loudly if the per-trait configs do not sit on identical weights.

    Mirrors :func:`method.anchor_noise._check_shared_checkpoints`: they are meant
    to differ only in trait, which ``weights_key`` excludes, so a mismatch means
    something that *does* change the weights diverged -- and the traits would
    then be measuring different models under one table.
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


def against_floor(frame: pd.DataFrame) -> pd.DataFrame:
    """Per trait: the drifted checkpoints' agreement beside the floor's.

    ``floor`` is ``cos_refresh`` at ``t = 0``, where the re-draw is a second
    sample from the same model. ``worst`` is the lowest agreement at any later
    checkpoint, and ``gap`` is how much of it drift accounts for. The worst
    checkpoint rather than the mean, for the reason
    :func:`method.anchor_noise.against_drift` takes the worst: a bound quoted in
    a limitations paragraph has to hold everywhere the paragraph applies.

    ``neg_pass_drop`` is the same reading for the filter: how far the negative
    side's survival rate at the worst checkpoint falls below its rate at the
    floor. A large ``gap`` with a large ``neg_pass_drop`` is the degenerate case
    the freeze prevents; a large ``gap`` with no drop means the axis moved for
    some other reason and the freeze is hiding it.
    """
    rows = []
    for trait, group in frame.groupby("trait"):
        floor_rows = group[group["t"] == FLOOR_CHECKPOINT]
        drifted = group[group["t"] != FLOOR_CHECKPOINT]
        floor = float(floor_rows["cos_refresh"].iloc[0])
        floor_neg = float(floor_rows["onpolicy_neg_pass"].iloc[0])
        if drifted.empty:
            # Only the control was measured. Saying so beats a row of NaNs that
            # reads like a result.
            continue
        # Kept as a one-row frame rather than squeezed to a Series, so every
        # column below is read the way ``floor`` above is.
        worst_at = drifted.loc[[drifted["cos_refresh"].idxmin()]]

        def at(column: str) -> float:
            return float(worst_at[column].iloc[0])

        rows.append(
            {
                "trait": trait,
                "floor": floor,
                "worst": at("cos_refresh"),
                "worst_t": int(worst_at["t"].iloc[0]),
                "gap": floor - at("cos_refresh"),
                "rho_frozen": at("rho_frozen"),
                "rho_onpolicy": at("rho_onpolicy"),
                "rho_gap": at("rho_frozen") - at("rho_onpolicy"),
                "floor_neg_pass": floor_neg,
                "worst_neg_pass": at("onpolicy_neg_pass"),
                "neg_pass_drop": floor_neg - at("onpolicy_neg_pass"),
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
    """Persist the comparison next to the trajectories, self-contained.

    The vectors and CSVs stay in the store, keyed by content; this is the copy a
    plotting machine can read with only the trajectories root synced, exactly as
    :func:`method.anchor_noise.write_summary` is for the anchor sweep.
    """
    base_wid = get_weights_id(cfgs[0], 0)
    path = axis_refresh_path(base_wid, label, mock=mock)
    payload = {
        "base_weights_id": base_wid,
        "label": label,
        "seed": cfgs[0].seed,
        "model": dataclasses.asdict(cfgs[0].model),
        "traits": [cfg.trait for cfg in cfgs],
        "persona_vector_threshold": cfgs[0].eval.persona_vector_threshold,
        "extract_n_per_question": cfgs[0].eval.extract_n_per_question,
        "steps": [step.dataset_id for step in cfgs[0].steps],
        "comparisons": frame.to_dict(orient="records"),
        "against_floor": against_floor(frame).to_dict(orient="records"),
    }
    with atomic_file(path) as scratch:
        scratch.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Axis refresh -> %s", path)
    return path


#: Columns of ``measure``'s frame worth printing, in reading order. The frame
#: itself carries every filter rate for the summary JSON; a terminal table of
#: twenty columns is unreadable and the JSON is where the rest belongs.
_SHOWN = (
    "trait",
    "t",
    "cos_refresh",
    "rho_frozen",
    "rho_onpolicy",
    "r_frozen",
    "r_onpolicy",
    "frozen_n_effective",
    "onpolicy_n_effective",
    "onpolicy_pos_pass",
    "onpolicy_neg_pass",
)


def _show(title: str, table: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    print(table.to_string(index=False) if not table.empty else "(empty)")


# --- CLI -------------------------------------------------------------------


def run(
    *,
    trunk: str,
    checkpoints: Sequence[int],
    traits: Sequence[str],
    seed: int,
    backend_kind: Backend,
    dtype: str,
    local: bool,
) -> pd.DataFrame:
    store = Store.for_backend(backend_kind)
    backend = get_backend(backend_kind, dtype=dtype)
    cfgs = experiments.build_axis_refresh_configs(
        trunk=trunk, seed=seed, measure_traits=traits, local=local
    )

    store.evict_orphaned_merged()
    syncer = Syncer.from_env(store)
    if syncer is not None:
        # Adapters for the whole trunk, plus whatever measurements exist at its
        # checkpoints -- the frozen vector is read, not recomputed, wherever
        # exp2 already measured it. One config's closure covers every trait:
        # they share a chain, which ``measure`` then verifies.
        syncer.pull_before_run(StoreSelection.for_config(cfgs[0]))

    logger.info(
        "Axis refresh: trunk %s, checkpoints %s, %d trait(s)",
        trunk,
        list(checkpoints),
        len(cfgs),
    )
    frame = measure(cfgs, checkpoints, store, backend, syncer=syncer)

    label = f"trunk_{trunk}{'_local' if local else ''}"
    summary = write_summary(cfgs, frame, label, mock=backend_kind is Backend.MOCK)
    if syncer is not None:
        syncer.push_axis_refresh(summary)

    _show("frozen axis against the checkpoint's own", frame[list(_SHOWN)])
    if set(frame["t"]) - {FLOOR_CHECKPOINT}:
        _show("drift against the sampling floor", against_floor(frame))
    else:
        # Every row would compare the control with itself. Saying so beats a
        # table of zeros that reads like perfect agreement.
        logger.warning(
            "only t=%d measured: that is the sampling floor, and there is no "
            "drifted checkpoint to read it against",
            FLOOR_CHECKPOINT,
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
        help="which exp2 decay trunk's checkpoints to re-draw on (default: a)",
    )
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=list(experiments.AXIS_REFRESH_CHECKPOINTS),
        help=(
            "steps along the trunk to re-draw at; 0 is required, since the "
            "re-draw there is the sampling floor the others are read against"
        ),
    )
    parser.add_argument(
        "--traits",
        nargs="+",
        default=list(experiments.AXIS_REFRESH_TRAITS),
        help=(
            "traits to re-draw; nothing is shared between them, so each one "
            "costs a full extraction pass at every checkpoint"
        ),
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

    load_dotenv(DOTENV_PATH)
    if args.backend is Backend.REAL:
        required = ["HF_TOKEN"]
        if not args.local:
            # Every draw here is judged, which is what makes it comparable to
            # the production one. --local selects the stubbed judge instead.
            required.append("OPENAI_API_KEY")
        check_env_vars(required)

    signal.signal(signal.SIGTERM, _die_on_sigterm)
    _run_and_report(args)


def _run_and_report(args: argparse.Namespace) -> None:
    """:func:`run`, with the outcome mailed out and a watchdog kept fed.

    Mirrors :func:`method.anchor_noise._run_and_report`: this is hours of the
    same rented GPU, just as unattended, and there are no per-checkpoint reports
    to send, so the flat job report is the whole of it.
    """
    notifier = Notifier.from_env()
    heartbeat = Heartbeat.from_env()
    logger.info("%s; %s", notifier.describe(), heartbeat.describe())

    title = f"axis_refresh (trunk {args.trunk}, {len(args.checkpoints)} checkpoint(s))"
    detail = (
        f"trunk: {args.trunk}\ncheckpoints: {list(args.checkpoints)}\n"
        f"traits: {list(args.traits)}"
    )
    started = time.monotonic()
    with heartbeat:
        try:
            run(
                trunk=args.trunk,
                checkpoints=args.checkpoints,
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
            notifier.send(subject, body, throttle_key="axis_refresh:failed")
            raise
        subject, body = report.job_report(
            title, elapsed=time.monotonic() - started, detail=detail
        )
        notifier.send(subject, body, throttle_key="axis_refresh:ok")


def _die_on_sigterm(signum: int, _frame: object) -> None:
    """Turn a polite kill into an exception, so the failure email still goes."""
    raise KeyboardInterrupt(f"received signal {signum}")


if __name__ == "__main__":
    main()
