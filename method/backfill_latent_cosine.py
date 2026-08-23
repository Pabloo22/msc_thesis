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

**``trajectory.json`` is the output; the store is only an input.** Same
reasoning as :mod:`method.backfill_se`: run this once wherever the store lives,
sync ``trajectories/`` back, and no analysis machine needs the store again. A
checkpoint whose ``h_neutral`` tensor is not reachable from *this* machine is
reported and left alone, never guessed at.

Idempotent: each run carries a ``"z_convention"`` marker (see
:data:`method.latent.CONVENTION`), so a converted run is skipped and a partial
pass can be resumed after syncing more of the store.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch

from method import anchor_noise
from method.config import Backend
from method.latent import CONVENTION, LEGACY_CONVENTION
from method.steps import Artifacts
from method.store import Store, atomic_file
from method.utils import trajectories_root

logger = logging.getLogger("backfill_latent_cosine")

#: The ``z`` fields this rescales. ``rho`` and ``r`` are already scale-free.
NORMALIZED = ("p", "q")


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

    def summary(self) -> str:
        return (
            f"{len(self.updated)} file(s) converted, {self.converted} z "
            f"value(s) rescaled, {self.anchor_rows} anchor-noise row(s) "
            f"rescaled, {self.already} already converted, "
            f"{self.without_z} carry no z, {len(self.unreachable)} "
            f"unreachable, {len(self.skipped)} skipped"
        )


def h_neutral_norm(
    store: Store, weights_id: str, source: str, layer: int
) -> float | None:
    """``||h_neutral||`` at ``layer``, or ``None`` if the tensor is not here.

    The same file :func:`method.steps.compute_step_latent` read to produce the
    value being converted, so the division undoes exactly the normalisation
    that was missing rather than an approximation of it.
    """
    path = (
        store.measurement_dir(weights_id)
        / Artifacts.h_neutral(source)
        / "mean_by_layer.pt"
    )
    if not path.exists():
        return None
    return float(torch.load(path, weights_only=False)[layer].float().norm())


def convert_run(
    path: Path, store: Store, report: Report, *, dry_run: bool = False
) -> None:
    """Rescale every ``z`` in one ``trajectory.json``, or leave the file alone.

    All-or-nothing per run: the norms for every checkpoint are gathered before
    anything is written, so a run that is only partly reachable keeps a
    consistent set of units and can be retried later.
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
        norm = h_neutral_norm(store, step["weights_id"], source, layer)
        if norm is None:
            report.unreachable.append(path)
            return
        norms[key] = norm

    for step, source in targets:
        z = step["z"][source]
        norm = norms[(step["weights_id"], source)]
        for component in NORMALIZED:
            z[component] = z[component] / norm
        report.converted += 1

    _stamp(path, payload, report, dry_run=dry_run)


def _stamp(path: Path, payload: dict, report: Report, *, dry_run: bool) -> None:
    """Record the convention and write the run back."""
    payload["z_convention"] = CONVENTION
    report.updated.append(path)
    if dry_run:
        return
    with atomic_file(path) as scratch:
        scratch.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def convert_anchor_noise(
    path: Path, store: Store, report: Report, *, dry_run: bool = False
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
        tensor = (
            anchor_noise.shared_dir(store, row["weights_id"], row["replicate"])
            / Artifacts.h_neutral(source)
            / "mean_by_layer.pt"
        )
        if not tensor.exists():
            report.unreachable.append(path)
            return
        norms[key] = float(torch.load(tensor, weights_only=False)[layer].float().norm())

    for row in rows:
        norm = norms[(row["weights_id"], row["replicate"])]
        for component in NORMALIZED:
            row[component] = row[component] / norm
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


def backfill(root: Path, store: Store, *, dry_run: bool = False) -> Report:
    report = Report()
    for path in iter_runs(root):
        convert_run(path, store, report, dry_run=dry_run)
    for path in iter_anchor_noise(root):
        convert_anchor_noise(path, store, report, dry_run=dry_run)
    return report


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
            "%d file(s) have no h_neutral tensor on this machine, so their z "
            "could not be converted and still holds projections. Pull the "
            "store (`python -m method.sync pull`) or re-run this where the "
            "store lives, then sync trajectories/ back. First few: %s",
            len(report.unreachable),
            ", ".join(str(p.parent.name) for p in report.unreachable[:5]),
        )


if __name__ == "__main__":
    main()
