"""Convert ``z``'s ``p`` and ``q`` from scalar projections to cosines.

    poetry run python -m method.backfill_latent_cosine --dry-run
    poetry run python -m method.backfill_latent_cosine
    poetry run python -m method.backfill_latent_cosine --mock

:func:`method.latent.compute_latent` now normalises by the activation as well
as by the persona vector, so ``p`` and ``q`` are cosines on ``[-1, 1]`` instead
of lengths on whatever scale the hidden states happen to live at. Two layers of
already-computed values do not know that:

1. the store's ``latent.json``, handled by renaming the artifact (see
   :data:`method.steps.Artifacts.LATENT_JSON`) so the old file is never served
   under the new meaning;
2. the ``"z"`` blocks copied into each run's ``trajectory.json``, which is what
   every figure actually reads;
3. the ``latents`` recorded in each ``trajectories/anchor_noise/*.json``, plus
   the ``spread`` and ``against_drift`` tables derived from them.

This script handles 2 and 3.

**A rescale, not a re-derivation.** The two conventions differ by exactly one
factor -- ``cos = proj / ||h_neutral||`` -- so the whole conversion is a
division by a number read from one 417KB tensor per checkpoint. Rebuilding
``z`` from ``v_0``, ``v_t`` and ``h`` instead would also silently re-anchor
every run onto whichever ``v_0`` the store holds *now*, and exp3 is known to
sit on several distinct base measurements (``method.visualization.latent_audit``
exists to find exactly that). Dividing preserves the anchor each value was
recorded against, so this changes the units and nothing else.

``rho`` and ``r`` are untouched: ``rho`` was always a cosine and ``r`` is a
length of the persona vector, neither of which involves ``h_neutral``. So is
every DeltaP field -- if any of those move, something is wrong.

The one thing that is *not* a rescale is the anchor-noise summary's derived
tables, which are recomputed from the converted rows; see
:func:`convert_anchor_noise` for why a rescale would not have been equivalent
there, and why this is also cheaper than re-running the sweep.

**The divisor is looked up before it is loaded.** Since
:mod:`method.backfill_h_norm`, a converted ``z`` block carries the very number
it was divided by (:data:`method.latent.H_NORM`), and that number is a property
of a *checkpoint* -- ``||h_neutral_t||`` at the trajectory's layer -- not of the
run that happened to record it. A trunk's checkpoints are shared by every run
built on the same prefix, so a run that arrives late is almost always asking for
a norm some sibling already published. :func:`index_recorded_norms` harvests
those into a lookup consulted before the store, and the block being converted
gains the field on the way through, which keeps the index growing rather than
merely being read. Only a checkpoint no run has recorded a norm for falls back
to reading its tensor.

That is what lets this run on a plotting box: 230GB of adapters and hidden
states exist to *produce* the norms, and once produced they are a float per
checkpoint sitting in files that box already syncs. Two runs disagreeing about
one checkpoint's norm means they sit on different measurements of it -- the
exp3 anchor split again -- so that key is dropped from the index rather than
resolved by a coin flip, and the store decides.

**``trajectory.json`` is the output; the store is only a fallback input.** Same
reasoning as :mod:`method.backfill_se`: where the index cannot answer, run this
wherever the store lives, sync ``trajectories/`` back, and no analysis machine
needs the store again. A checkpoint that neither the index nor *this* machine's
store can price is reported and left alone, never guessed at.

**Runs that land after a pass are the normal case, not an anomaly.** A sweep
converts what is on one box's disk at one moment; the remote is the union of
several boxes, and a trunk still training when the sweep ran is pushed hours
later by a process whose code predates the change. The marker makes re-running
cheap, so re-run it after every pull rather than treating it as one-time.

Idempotent: each run carries a ``"z_convention"`` marker (see
:data:`method.latent.CONVENTION`), so a converted run is skipped and a partial
pass can be resumed after syncing more of the store.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch

from method import anchor_noise
from method.config import Backend
from method.latent import CONVENTION, H_NORM, LEGACY_CONVENTION
from method.steps import Artifacts, h_neutral_path
from method.store import Store, atomic_file
from method.utils import trajectories_root

logger = logging.getLogger("backfill_latent_cosine")

#: The ``z`` fields this rescales. ``rho`` and ``r`` are already scale-free.
NORMALIZED = ("p", "q")

#: How closely two records of one checkpoint's norm must agree to count as the
#: same measurement. Two runs filled from the same tensor agree bit for bit --
#: both take ``float(tensor[layer].float().norm())`` -- so this is slack for
#: JSON round-tripping, not for measurement noise. Anything looser would paper
#: over the case this tolerance exists to catch: two genuinely different
#: measurements of one weights_id, which differ in the third digit, not the
#: fifteenth.
NORM_AGREEMENT = 1e-9


@dataclass
class Report:
    """What a pass over the run directories found and changed."""

    #: Runs whose ``trajectory.json`` was rewritten in cosine units.
    updated: list[Path] = field(default_factory=list)
    #: Runs written in a format this script cannot read (no ``config`` block).
    skipped: list[Path] = field(default_factory=list)
    #: Runs already carrying the cosine marker.
    already: int = 0
    #: Runs recording no ``z`` at all -- every branch endpoint, by design.
    without_z: int = 0
    #: ``(checkpoint, source)`` pairs converted.
    converted: int = 0
    #: Runs left alone because some checkpoint's ``h_neutral`` is not on this
    #: machine. Converted per *run*, never per checkpoint: half a trajectory in
    #: cosines and half in projections is worse than none of it.
    unreachable: list[Path] = field(default_factory=list)
    #: Anchor-noise replicate rows converted, across every summary.
    anchor_rows: int = 0
    #: Divisors answered by :class:`RecordedNorms` rather than by a tensor
    #: read. The gap between this and :attr:`from_store` is what says whether a
    #: machine without the store could have done the same work.
    from_record: int = 0
    #: Divisors that needed the store, because no run had recorded them.
    from_store: int = 0

    def summary(self) -> str:
        return (
            f"{len(self.updated)} file(s) converted, {self.converted} z "
            f"value(s) rescaled, {self.anchor_rows} anchor-noise row(s) "
            f"rescaled, {self.already} already converted, "
            f"{self.without_z} carry no z, {len(self.unreachable)} "
            f"unreachable, {len(self.skipped)} skipped; "
            f"{self.from_record} divisor(s) from recorded norms, "
            f"{self.from_store} from the store"
        )


#: A checkpoint's activation, as the runs identify it: which weights, which
#: pool of neutral prompts, which layer.
Checkpoint = tuple[str, str, int]

#: An anchor-noise replicate's activation. The replicate index is what makes it
#: a different measurement from the checkpoint's production one -- re-deriving
#: ``h_neutral`` from scratch is the whole point of that sweep -- so it never
#: shares a key with :data:`Checkpoint`.
Replicate = tuple[str, int, int]


@dataclass
class RecordedNorms:
    """``||h_neutral||`` as the trajectories tree already records it.

    Every ``z`` block written since :data:`method.latent.H_NORM` existed
    carries its own divisor, and a checkpoint's norm does not depend on which
    run measured it. Harvesting those turns the fallback -- 417KB of tensor per
    checkpoint, out of a store that is two orders of magnitude too large to
    move for it -- into a dictionary lookup.

    Disagreement is a finding, not a tie to break. Two runs recording different
    norms for one ``weights_id`` are sitting on different measurements of it,
    which is precisely what :mod:`method.visualization.latent_audit` exists to
    surface; the key is dropped so the store answers instead, and the pair is
    kept in :attr:`conflicts` to be reported.
    """

    #: Norms every recording agreed on.
    checkpoints: dict[Checkpoint, float] = field(default_factory=dict)
    #: Replicate norms, keyed separately for the reason on :data:`Replicate`.
    replicates: dict[Replicate, float] = field(default_factory=dict)
    #: Keys withdrawn because their recordings disagreed, and the values seen.
    conflicts: dict[Checkpoint | Replicate, list[float]] = field(default_factory=dict)

    def record(self, key: Checkpoint | Replicate, norm: float) -> None:
        """Note one recording, withdrawing the key if it contradicts another."""
        if key in self.conflicts:
            self._note_conflict(key, norm)
            return
        table = self._table(key)
        seen = table.get(key)
        if seen is None:
            table[key] = norm
            return
        if not math.isclose(seen, norm, rel_tol=NORM_AGREEMENT):
            del table[key]
            self.conflicts[key] = [seen]
            self._note_conflict(key, norm)

    def _note_conflict(self, key: Checkpoint | Replicate, norm: float) -> None:
        values = self.conflicts[key]
        if not any(math.isclose(v, norm, rel_tol=NORM_AGREEMENT) for v in values):
            values.append(norm)

    def _table(self, key: Checkpoint | Replicate) -> dict:
        # A `Checkpoint` names its h_neutral source, a `Replicate` its index;
        # nothing else distinguishes the two shapes.
        return self.checkpoints if isinstance(key[1], str) else self.replicates


def index_recorded_norms(root: Path) -> RecordedNorms:
    """Harvest every norm the runs under ``root`` have already written down.

    Reads the same files this script writes, so it costs one pass over a few
    tens of megabytes of JSON and needs nothing else on the machine. A run
    still in projection units is harvested just the same: :data:`H_NORM` is a
    length either way, and a file that carries one is answering for its
    checkpoints whatever units its own ``p`` is in.
    """
    recorded = RecordedNorms()
    for path in iter_runs(root):
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("config")
        if config is None or "model" not in config:
            continue
        layer = config["model"]["layer"]
        for step in payload["steps"]:
            for source, z in (step.get("z") or {}).items():
                if H_NORM in z:
                    recorded.record((step["weights_id"], source, layer), z[H_NORM])
    for path in iter_anchor_noise(root):
        payload = json.loads(path.read_text(encoding="utf-8"))
        layer = payload["model"]["layer"]
        for row in payload["latents"]:
            if H_NORM in row:
                recorded.record(
                    (row["weights_id"], row["replicate"], layer), row[H_NORM]
                )
    return recorded


def h_neutral_norm(
    store: Store,
    weights_id: str,
    source: str,
    layer: int,
    recorded: RecordedNorms | None = None,
    report: Report | None = None,
) -> float | None:
    """``||h_neutral||`` at ``layer``, or ``None`` if nothing on this box has it.

    Answered from ``recorded`` where it can be, and otherwise from the tensor
    :func:`method.steps.compute_step_latent` read to produce the value being
    converted -- the same number by either route, since that is the tensor the
    recorded field was taken from. So the division undoes exactly the
    normalisation that was missing rather than an approximation of it.
    """
    if recorded is not None:
        norm = recorded.checkpoints.get((weights_id, source, layer))
        if norm is not None:
            if report is not None:
                report.from_record += 1
            return norm
    path = h_neutral_path(store, weights_id, source)
    if not path.exists():
        return None
    if report is not None:
        report.from_store += 1
    return float(torch.load(path, weights_only=False)[layer].float().norm())


def convert_run(
    path: Path,
    store: Store,
    report: Report,
    *,
    recorded: RecordedNorms | None = None,
    dry_run: bool = False,
) -> None:
    """Rescale every ``z`` in one ``trajectory.json``, or leave the file alone.

    All-or-nothing per run: the norms for every checkpoint are gathered before
    anything is written, so a run that is only partly reachable keeps a
    consistent set of units and can be retried later.

    Each converted block also gains :data:`H_NORM`, which costs nothing -- the
    divisor is in hand -- and is what stops this run from being another file
    that can only be priced by holding the store. It is set rather than
    defaulted only where absent, so a block that already carried the field
    keeps the value it was measured with.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config")
    if config is None or "model" not in config:
        # Pre-dates the current run format; `collect.py` cannot load these
        # either, so nothing downstream would read the result.
        report.skipped.append(path)
        return

    if payload.get("z_convention", LEGACY_CONVENTION) == CONVENTION:
        report.already += 1
        return

    layer = config["model"]["layer"]
    targets = [
        (step, source) for step in payload["steps"] for source in (step.get("z") or {})
    ]
    if not targets:
        # A branch endpoint measures only b (see
        # method.config.MeasurementLevel). There is no z to convert, so it is
        # marked rather than left to be reported as stale forever.
        report.without_z += 1
        _stamp(path, payload, report, dry_run=dry_run)
        return

    norms: dict[tuple[str, str], float] = {}
    for step, source in targets:
        key = (step["weights_id"], source)
        if key in norms:
            continue
        norm = h_neutral_norm(
            store, step["weights_id"], source, layer, recorded, report
        )
        if norm is None:
            report.unreachable.append(path)
            return
        norms[key] = norm

    for step, source in targets:
        z = step["z"][source]
        norm = norms[(step["weights_id"], source)]
        for component in NORMALIZED:
            z[component] = z[component] / norm
        z.setdefault(H_NORM, norm)
        report.converted += 1

    _stamp(path, payload, report, dry_run=dry_run)


def _replicate_norm(
    store: Store,
    weights_id: str,
    replicate: int,
    layer: int,
    source: str,
    recorded: RecordedNorms | None,
    report: Report,
) -> float | None:
    """``||h_neutral||`` for one anchor-noise replicate, lookup before tensor.

    The replicate's own activation, never the checkpoint's production one: see
    :func:`convert_anchor_noise`.
    """
    if recorded is not None:
        norm = recorded.replicates.get((weights_id, replicate, layer))
        if norm is not None:
            report.from_record += 1
            return norm
    tensor = (
        anchor_noise.shared_dir(store, weights_id, replicate)
        / Artifacts.h_neutral(source)
        / "mean_by_layer.pt"
    )
    if not tensor.exists():
        return None
    report.from_store += 1
    return float(torch.load(tensor, weights_only=False)[layer].float().norm())


def _stamp(path: Path, payload: dict, report: Report, *, dry_run: bool) -> None:
    """Record the convention and write the run back."""
    payload["z_convention"] = CONVENTION
    report.updated.append(path)
    if dry_run:
        return
    with atomic_file(path) as scratch:
        scratch.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def convert_anchor_noise(
    path: Path,
    store: Store,
    report: Report,
    *,
    recorded: RecordedNorms | None = None,
    dry_run: bool = False,
) -> None:
    """Rescale one ``anchor_noise`` summary and re-derive its tables.

    The summary holds a flat ``latents`` list -- one row per (trait, checkpoint,
    replicate) -- and two tables derived from it. Each row needs the norm of the
    activation *its own replicate* drew, not the checkpoint's production one:
    that a replicate re-derives ``h_neutral`` from scratch is the whole point of
    the sweep.

    The derived tables are recomputed rather than rescaled. ``spread`` would
    almost survive a rescale, but ``against_drift`` divides a noise level by a
    *drift* -- a difference between two checkpoints' levels -- and each of those
    levels is divided by a different ``||h||``. A few percent of growth in
    ``||h||`` across a trunk moves that difference by tens of percent, so the
    ratios have to be re-derived from the converted rows or they quietly stop
    matching the figures they are quoted against.

    Re-deriving here rather than re-running :func:`method.anchor_noise.measure`
    is not just convenience: that function materialises every checkpoint before
    it looks at anything, and evicts it afterwards, so a re-run replays a merge
    chain per checkpoint to recompute arithmetic that takes microseconds.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("z_convention", LEGACY_CONVENTION) == CONVENTION:
        report.already += 1
        return

    layer = payload["model"]["layer"]
    source = payload["h_neutral_source"]
    rows = payload["latents"]

    norms: dict[tuple[str, int], float] = {}
    for row in rows:
        key = (row["weights_id"], row["replicate"])
        if key in norms:
            continue
        norm = _replicate_norm(store, *key, layer, source, recorded, report)
        if norm is None:
            report.unreachable.append(path)
            return
        norms[key] = norm

    for row in rows:
        norm = norms[(row["weights_id"], row["replicate"])]
        for component in NORMALIZED:
            row[component] = row[component] / norm
        row.setdefault(H_NORM, norm)
        report.anchor_rows += 1

    frame = pd.DataFrame(rows)
    table = anchor_noise.spread(frame)
    payload["spread"] = table.to_dict(orient="records")
    payload["against_drift"] = anchor_noise.against_drift(frame, table).to_dict(
        orient="records"
    )

    _stamp(path, payload, report, dry_run=dry_run)


def iter_runs(root: Path) -> Iterator[Path]:
    """Every ``trajectory.json`` under a trajectories root, in a stable order."""
    return iter(sorted(root.glob("*/trajectory.json")))


def iter_anchor_noise(root: Path) -> Iterator[Path]:
    """Every anchor-noise summary under a trajectories root, in a stable order."""
    return iter(sorted(root.glob("anchor_noise/*.json")))


def backfill(
    root: Path, store: Store, *, use_records: bool = True, dry_run: bool = False
) -> Report:
    """Convert every run and anchor-noise summary under ``root``.

    The index is built once, up front, from the tree as it stands: a run
    converted during this pass does publish its divisor, but only for
    checkpoints some other file had already priced, so re-reading the tree as
    it grows would find nothing new. ``use_records=False`` forces every divisor
    through the store, which is how a box that holds it can confirm the two
    routes agree.
    """
    report = Report()
    recorded = index_recorded_norms(root) if use_records else None
    for path in iter_runs(root):
        convert_run(path, store, report, recorded=recorded, dry_run=dry_run)
    for path in iter_anchor_noise(root):
        convert_anchor_noise(path, store, report, recorded=recorded, dry_run=dry_run)
    if recorded is not None:
        _report_conflicts(recorded)
    return report


def _report_conflicts(recorded: RecordedNorms) -> None:
    if not recorded.conflicts:
        return
    logger.warning(
        "%d checkpoint(s) are recorded with disagreeing h_neutral norms, so "
        "they sit on more than one measurement of the same weights (see "
        "method.visualization.latent_audit). The store decides for these, not "
        "the index. First few: %s",
        len(recorded.conflicts),
        ", ".join(
            f"{key[0]}[{key[1]}]={sorted(values)}"
            for key, values in list(recorded.conflicts.items())[:3]
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="operate on trajectories-mock/ and store-mock/ instead",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change without writing anything",
    )
    parser.add_argument(
        "--store-only",
        action="store_true",
        help=(
            "read every divisor from the store instead of from the h_norm "
            "each run already records; only useful where the store is whole, "
            "to confirm the two routes agree"
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    backend = Backend.MOCK if args.mock else Backend.REAL
    report = backfill(
        trajectories_root(mock=args.mock),
        Store.for_backend(backend),
        use_records=not args.store_only,
        dry_run=args.dry_run,
    )

    logger.info("%s%s", report.summary(), " (dry run)" if args.dry_run else "")
    for path in report.updated:
        logger.debug("converted %s", path)
    if report.skipped:
        logger.warning(
            "%d run(s) predate the current format and were skipped: %s",
            len(report.skipped),
            ", ".join(str(p.parent.name) for p in report.skipped[:5]),
        )
    if report.unreachable:
        logger.warning(
            "%d file(s) name a checkpoint that no run has recorded an h_norm "
            "for and whose tensor is not on this machine, so their z could "
            "not be converted and still holds projections. Pull the runs "
            "first (`python -m method.sync pull-plots`) -- a sibling run of "
            "the same trunk usually carries the divisor -- and only if that "
            "leaves them here, re-run where the store lives and sync "
            "trajectories/ back. First few: %s",
            len(report.unreachable),
            ", ".join(str(p.parent.name) for p in report.unreachable[:5]),
        )


if __name__ == "__main__":
    main()
