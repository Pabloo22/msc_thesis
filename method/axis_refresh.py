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

**Every cosine is read against a floor, and the floor is usually already paid
for.** Two draws of the extraction set from the *same* model still disagree
slightly, because the responses are sampled at temperature 1 and then judged. A
cosine of 0.97 at $t = 6$ says nothing until that floor is known to be 0.99
rather than 0.97.

That floor is exactly what a :mod:`method.anchor_noise` replicate is, under
another name, so :func:`floor_from_replicates` reads it from the draws that
sweep already left in the store rather than buying it again. It is the better
source as well as the cheaper one: $R$ replicates give $\binom{R}{2}$ pairs and
therefore a worst case, where a fresh re-draw at $t = 0$ gives a single pair and
a point estimate. The base checkpoint is shared by every trunk and every seed,
so one anchor-noise sweep serves every run of this script.

Measuring $t = 0$ here is therefore optional, and worth its cost only when no
replicates exist. When it is measured, its row is a control and not a result --
it contains no drift by construction, so nothing in it speaks to staleness.

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
import itertools
import json
import logging
import signal
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
import torch

from method import anchor_noise, experiments, report, steps
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

#: The base checkpoint. Re-drawing here measures the extraction procedure's own
#: sampling noise and no drift at all, so its row is a control rather than a
#: result -- and usually an unnecessary one, since
#: :func:`floor_from_replicates` reads the same quantity from anchor-noise draws
#: that already exist.
FLOOR_CHECKPOINT = 0

#: Minimum judge coherence for a response to be usable, on either side. Fixed at
#: 50 in the vendored filter rather than configurable, so it is a constant here
#: too; ``cfg.eval.persona_vector_threshold`` is the trait-score half.
COHERENCE_THRESHOLD = 50

#: Below this many surviving pairs, a vector is still extracted but logged as
#: thin. Purely a legibility threshold -- nothing branches on it -- and the
#: number is a round fraction of the 1000-pair extraction set rather than
#: anything derived. Zero survivors is the case that genuinely cannot proceed,
#: and it is handled separately in :func:`ensure_onpolicy_vector`.
SPARSE_PAIRS = 50


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
                model_id=get_weights_id(cfg, t),
            )
    return pos, neg


def ensure_onpolicy_vector(
    cfg: TrajectoryConfig,
    t: int,
    store: Store,
    backend: ExecutionBackend,
    model_path: str,
) -> Path | None:
    """$v^{(t)}$ as the paper's own procedure would produce it at ``M_t``.

    ``None`` when the filter kept nothing, which is a *result* and not an
    error: a checkpoint that can no longer answer either half of its own
    extraction set credibly is the extreme of the degenerate case the freeze
    exists to prevent, and it is the single most informative thing this script
    can find.

    It has to be caught here rather than left to the extractor. The vendored
    ``generate_vec.get_hidden_p_and_r`` builds its per-layer activation lists by
    appending inside a loop over the surviving pairs and then calls
    ``torch.cat`` on them, so with zero survivors it raises
    ``RuntimeError: torch.cat(): expected a non-empty list of Tensors`` -- and
    the run would die at the exact checkpoint whose collapse is the finding,
    taking the checkpoints already drawn in this invocation with it.

    The filter is therefore consulted *before* the extractor is called. That
    reads the two CSVs a second time (:func:`compare` reads them again for the
    record), which is microseconds against the draw that produced them, and it
    keeps each function answerable on its own inputs.
    """
    out = onpolicy_vector_path(store, cfg, t)
    if out.exists():
        logger.info("[skip] on-policy vector at t=%d already extracted", t)
        return out
    pos, neg = ensure_onpolicy_extract_csvs(cfg, t, store, backend, model_path)

    counts = filter_counts(pos, neg, cfg.trait, cfg.eval.persona_vector_threshold)
    if counts["n_effective"] == 0:
        logger.warning(
            "t=%d [%s]: the filter kept 0 of %d pairs (pos_pass=%.3f, "
            "neg_pass=%.3f, coherence_pass=%.3f) -- no vector can be extracted, "
            "and the vector columns for this checkpoint are undefined. This is "
            "the collapse the frozen extraction text exists to avoid.",
            t,
            cfg.trait,
            counts["n_pairs"],
            counts["pos_pass"],
            counts["neg_pass"],
            counts["coherence_pass"],
        )
        return None
    if counts["n_effective"] < SPARSE_PAIRS:
        # Not a gate: the extractor is happy with one pair, and where to draw a
        # "too few" line is a judgement the reader should make from
        # ``n_effective``, which every table carries. A mean over a handful of
        # responses is noisy enough to be worth saying out loud, though.
        logger.warning(
            "t=%d [%s]: only %d of %d pairs are usable; the vector below is a "
            "mean over that few responses, and its cosine should be read with "
            "that in mind",
            t,
            cfg.trait,
            counts["n_effective"],
            counts["n_pairs"],
        )

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

    A missing on-policy vector is not an error. It means the filter kept nothing
    at this checkpoint (see :func:`ensure_onpolicy_vector`), so the three
    columns that need that vector are undefined and reported as NaN, while every
    filter statistic -- which is what makes the collapse legible -- is recorded
    exactly as it would be otherwise. ``onpolicy_collapsed`` flags the row, so a
    reader never has to infer the difference between "no vector" and "a vector
    that happened not to be written".

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
    onpolicy_path = onpolicy_vector_path(store, cfg, t)
    onpolicy = at(onpolicy_path) if onpolicy_path.exists() else None

    threshold = cfg.eval.persona_vector_threshold
    frozen_counts = filter_counts(
        *frozen_extract_paths(store, cfg), cfg.trait, threshold
    )
    onpolicy_counts = filter_counts(
        *onpolicy_extract_paths(store, cfg, t), cfg.trait, threshold
    )
    undefined = float("nan")
    return {
        "cos_refresh": undefined if onpolicy is None else cosine(frozen, onpolicy),
        "rho_frozen": cosine(v0, frozen),
        "rho_onpolicy": undefined if onpolicy is None else cosine(v0, onpolicy),
        "r_frozen": float(frozen.float().norm()),
        "r_onpolicy": (
            undefined if onpolicy is None else float(onpolicy.float().norm())
        ),
        "onpolicy_collapsed": onpolicy is None,
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
    _check_shared_checkpoints(cfgs)
    if FLOOR_CHECKPOINT not in checkpoints and not all(
        floor_from_replicates(cfg, store) for cfg in cfgs
    ):
        raise ValueError(
            f"checkpoint {FLOOR_CHECKPOINT} is not being measured and "
            "method.anchor_noise has not left at least two draws of v_0 to read "
            "a sampling floor from. Every cos_refresh is uninterpretable without "
            f"one, so either add checkpoint {FLOOR_CHECKPOINT} or run "
            "'python -m method.anchor_noise --replicates 2' first"
        )

    # Every row is read against v_0, whether or not t=0 is one of the rows, so
    # the base vector has to exist before the walk starts. This is the ordinary
    # production artifact -- a cache hit wherever exp2 has measured the base
    # model -- and not a re-draw: it reuses M_0's frozen extraction text exactly
    # as :func:`method.steps.extract_persona_vector` always does.
    for cfg in cfgs:
        steps.extract_persona_vector(cfg, 0, store, backend)

    rows = []
    ordered = sorted(set(checkpoints))
    for index, t in enumerate(ordered):
        following = ordered[index + 1] if index + 1 < len(ordered) else None
        wid = get_weights_id(cfgs[0], t)
        logger.info("--- checkpoint t=%d (%s) ---", t, wid)
        model_path = materialize(cfgs[0], t, store, backend)
        for cfg in cfgs:
            steps.extract_persona_vector(cfg, t, store, backend)
            ensure_onpolicy_vector(cfg, t, store, backend, model_path)
            record = compare(cfg, t, store)
            logger.info(
                "t=%d [%s]: cos(frozen, on-policy) = %s, effective pairs "
                "%d frozen / %d on-policy",
                t,
                cfg.trait,
                (
                    "undefined (filter collapsed)"
                    if record["onpolicy_collapsed"]
                    else f"{record['cos_refresh']:.4f}"
                ),
                record["frozen_n_effective"],
                record["onpolicy_n_effective"],
            )
            rows.append({"trait": cfg.trait, "t": t, "weights_id": wid, **record})
        # Pushed per checkpoint rather than at the end, so a preemption partway
        # through keeps the checkpoints it finished.
        if syncer is not None:
            syncer.push_measurement(wid)
        if t and following != t + 1:
            # A 7B checkpoint is ~15GB, so it goes as soon as nothing needs it.
            # The exception is the next checkpoint being this one's immediate
            # successor: ``materialize`` walks forward from the deepest
            # checkpoint already merged, so keeping this one turns the next into
            # a single merge instead of a replay from the base model. Over a
            # contiguous sweep that is the difference between one merge per
            # checkpoint and a triangular number of them -- 6 against 21 on a
            # six-step trunk. Holding one extra checkpoint is the price, and
            # ``materialize`` already tolerates two being resident.
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


def default_checkpoints(cfg: TrajectoryConfig) -> tuple[int, ...]:
    """Every checkpoint on the trunk except the base.

    The whole trunk, because the interesting reading is a *curve*: whether
    agreement falls away monotonically as drift accumulates is far stronger
    evidence than two endpoints, and it locates where the frozen axis starts to
    part company rather than only showing that it eventually does.

    Without ``t = 0``, because the re-draw there measures the extraction
    procedure's own sampling noise and no drift at all, and
    :func:`floor_from_replicates` already reads that number from anchor-noise
    draws in the store. Pass ``--checkpoints 0 ...`` explicitly to measure it
    anyway; it is one draw for the whole sweep, since every trunk shares one
    base checkpoint.
    """
    return tuple(range(1, len(cfg.steps) + 1))


def floor_from_replicates(
    cfg: TrajectoryConfig, store: Store, *, max_replicates: int = 8
) -> dict[str, float] | None:
    r"""The sampling floor, read off :mod:`method.anchor_noise`'s existing draws.

    The floor is $\cos$ between two independent draws of $v^{(0)}$ from the
    *same* model, so it isolates the extraction procedure's own sampling noise
    with no drift in it at all. Re-drawing at $t = 0$ measures exactly that --
    and so does an anchor-noise replicate, which is the same operation under
    another name. Where those replicates already exist there is no reason to
    buy the number twice: this reads every pair of them, which costs loading a
    few tensors.

    Better than one re-draw, too. A single fresh draw gives one pair and hence
    one number with no spread; $R$ replicates give $\binom{R}{2}$ pairs, so the
    floor comes with a range and a worst case rather than a point estimate --
    and the worst case is what a bound in a limitations paragraph needs.

    ``None`` when fewer than two draws exist, which is when re-drawing at
    ``t = 0`` is the only way to get a floor and is worth its cost.

    This is a deliberate dependency on another module's storage layout, taken
    because :func:`method.anchor_noise.persona_vector_path` is that layout's
    documented accessor and because the base checkpoint is shared: every trunk
    and every seed resolves to one ``weights_id`` at ``t = 0``, so one
    anchor-noise sweep serves every axis-refresh run there will ever be.
    """
    paths = [
        anchor_noise.persona_vector_path(store, cfg, 0, replicate)
        for replicate in range(max_replicates)
    ]
    vectors = [
        torch.load(path, weights_only=False)[cfg.model.layer]
        for path in paths
        if path.exists()
    ]
    if len(vectors) < 2:
        return None
    cosines = [cosine(a, b) for a, b in itertools.combinations(vectors, 2)]
    return {
        "floor": min(cosines),  # the bound has to hold for every pair
        "floor_mean": sum(cosines) / len(cosines),
        "floor_draws": len(vectors),
        "floor_pairs": len(cosines),
    }


def summary_label(
    trunk: str, traits: Sequence[str], checkpoints: Sequence[int], *, local: bool
) -> str:
    """Name the summary after the whole selection, not just the trunk.

    ``--traits`` and ``--checkpoints`` both change what a run measures, so a
    label carrying only the trunk lets one selection silently overwrite
    another's summary: ``--checkpoints 0 5 6`` then ``--checkpoints 0 6`` would
    leave a single file claiming to be the second, and a narrower ``--traits``
    would drop a trait from the record with no trace that it had ever been
    measured. Both are easy to do while iterating and neither is recoverable --
    the draws themselves survive in the store, but the summary a plotting box
    reads does not.

    Sorted and deduplicated, so the same selection in a different order names
    the same file rather than a second copy of it.
    """
    kinds = "-".join(sorted(set(traits)))
    ts = "-".join(str(t) for t in sorted(set(checkpoints)))
    return f"trunk_{trunk}_t{ts}_{kinds}{'_local' if local else ''}"


# --- reading the result ----------------------------------------------------


def against_floor(
    frame: pd.DataFrame, floors: Mapping[str, float] | None = None
) -> pd.DataFrame:
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

    **Collapsed checkpoints are named, never averaged in and never dropped.** A
    checkpoint whose filter kept nothing has no cosine at all, and ``idxmin``
    skips NaN -- so left alone, the single most extreme outcome the script can
    find would quietly vanish from the summary table while the surviving
    checkpoints made the freeze look sound. ``n_collapsed`` and ``collapsed_t``
    report them instead, and ``worst`` is computed over the checkpoints that do
    have a vector (NaN when none do). A collapse is worse than any cosine, so
    read those columns first.
    """
    rows = []
    for key, group in frame.groupby("trait"):
        trait = str(key)
        floor_rows = group[group["t"] == FLOOR_CHECKPOINT]
        drifted = group[group["t"] != FLOOR_CHECKPOINT]
        if drifted.empty:
            # Only the control was measured. Saying so beats a row of NaNs that
            # reads like a result.
            continue
        if floor_rows.empty:
            # t=0 was not re-drawn, so the floor comes from anchor-noise's
            # replicates instead (see :func:`floor_from_replicates`). Its
            # negative-side pass rate is not recorded there, so the filter
            # reading below degrades to the level rather than the drop.
            floor = float((floors or {}).get(trait, float("nan")))
            floor_neg = float("nan")
        else:
            floor = float(floor_rows["cos_refresh"].iloc[0])
            floor_neg = float(floor_rows["onpolicy_neg_pass"].iloc[0])

        collapsed = drifted[drifted["onpolicy_collapsed"]]
        measured = drifted[~drifted["onpolicy_collapsed"]]
        undefined = float("nan")
        row = {
            "trait": trait,
            "floor": floor,
            "n_collapsed": len(collapsed),
            "collapsed_t": ", ".join(str(int(t)) for t in sorted(collapsed["t"])),
        }
        if measured.empty:
            # Every drifted checkpoint collapsed. There is no cosine to report
            # and the collapse columns above are the whole finding.
            rows.append(
                row
                | {
                    "worst": undefined,
                    "worst_t": undefined,
                    "gap": undefined,
                    "rho_frozen": undefined,
                    "rho_onpolicy": undefined,
                    "rho_gap": undefined,
                    "floor_neg_pass": floor_neg,
                    "worst_neg_pass": undefined,
                    "neg_pass_drop": undefined,
                }
            )
            continue

        # Kept as a one-row frame rather than squeezed to a Series, so every
        # column below is read the way ``floor`` above is.
        worst_at = measured.loc[[measured["cos_refresh"].idxmin()]]

        def at(column: str, _row: pd.DataFrame = worst_at) -> float:
            return float(_row[column].iloc[0])

        rows.append(
            row
            | {
                "worst": at("cos_refresh"),
                "worst_t": float(worst_at["t"].iloc[0]),
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
    floors: Mapping[str, dict[str, float]] | None = None,
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
        # Where the floor came from, so a reader never has to guess whether a
        # gap was measured against this run's own t=0 row or against
        # anchor-noise's replicates.
        "floors": {trait: dict(stats) for trait, stats in (floors or {}).items()},
        "comparisons": frame.to_dict(orient="records"),
        "against_floor": against_floor(
            frame, {trait: stats["floor"] for trait, stats in (floors or {}).items()}
        ).to_dict(orient="records"),
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
    "onpolicy_collapsed",
)


def _show(title: str, table: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    print(table.to_string(index=False) if not table.empty else "(empty)")


# --- CLI -------------------------------------------------------------------


def run(
    *,
    trunk: str,
    checkpoints: Sequence[int] | None,
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
    # Resolved here rather than in the parser: the trunk decides how many
    # checkpoints there are, and the parser does not know which trunk was asked
    # for until it has finished parsing.
    if checkpoints is None:
        checkpoints = default_checkpoints(cfgs[0])

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

    floors = {}
    for cfg in cfgs:
        stats = floor_from_replicates(cfg, store)
        if stats is not None:
            floors[cfg.trait] = stats
            logger.info(
                "sampling floor [%s] = %.4f (worst of %d pairs over %d anchor-noise "
                "draws of v_0; mean %.4f)",
                cfg.trait,
                stats["floor"],
                stats["floor_pairs"],
                stats["floor_draws"],
                stats["floor_mean"],
            )

    label = summary_label(trunk, traits, checkpoints, local=local)
    summary = write_summary(
        cfgs, frame, label, mock=backend_kind is Backend.MOCK, floors=floors
    )
    if syncer is not None:
        syncer.push_axis_refresh(summary)

    collapsed = frame[frame["onpolicy_collapsed"]]
    if not collapsed.empty:
        # The headline finding when it happens, and easy to scroll past in a
        # long unattended log, so it is said before the tables rather than left
        # to a column in them.
        logger.warning(
            "%d checkpoint(s) could not be re-extracted at all -- the filter "
            "kept zero pairs: %s. Their vector columns are undefined.",
            len(collapsed),
            ", ".join(
                f"t={r.t} [{r.trait}]" for r in collapsed.itertuples(index=False)
            ),
        )

    _show("frozen axis against the checkpoint's own", frame[list(_SHOWN)])
    if set(frame["t"]) - {FLOOR_CHECKPOINT}:
        _show(
            "drift against the sampling floor",
            against_floor(
                frame, {trait: stats["floor"] for trait, stats in floors.items()}
            ),
        )
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
        default=None,
        help=(
            "steps along the trunk to re-draw at; defaults to all of them "
            "except 0. Measuring 0 is only needed when method.anchor_noise has "
            "left fewer than two draws of v_0 to read a sampling floor from; "
            "where it has, re-drawing there buys a second copy of a number "
            "already on disk"
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

    selection = "all" if args.checkpoints is None else list(args.checkpoints)
    title = f"axis_refresh (trunk {args.trunk}, {selection} checkpoint(s))"
    detail = (
        f"trunk: {args.trunk}\ncheckpoints: {selection}\n"
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
