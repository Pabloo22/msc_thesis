r"""Record ``||h_neutral||`` beside every ``z`` that was normalised by it.

    poetry run python -m method.backfill_h_norm --dry-run
    poetry run python -m method.backfill_h_norm
    poetry run python -m method.backfill_h_norm --mock

:mod:`method.backfill_latent_cosine` divided ``p`` and ``q`` by
``||h_neutral_t||`` and did not keep the divisor. That leaves two things
unanswerable from a run directory alone, on any machine that does not also hold
the store:

1. **The conversion is one-way.** ``p * h_norm`` is exactly the scalar
   projection the old convention reported, so with the norm recorded, no result
   is locked behind the choice of convention and the decision never has to be
   re-litigated from measurements again.
2. **A fall in ``p`` is ambiguous.** A cosine drops either because the neutral
   state turned off the persona axis or because it grew in directions that have
   nothing to do with the persona -- the norm is what separates the two, and it
   is the quantity that says whether the growth the cosine convention exists to
   divide out was real in the first place.

This adds :data:`method.latent.H_NORM` to each ``z`` block in each run's
``trajectory.json``, and to each row of ``trajectories/anchor_noise/*.json``.
Going forward :func:`method.steps.compute_step_latent` and
:func:`method.anchor_noise.replicate_latent` record it at measurement time, so
this is for what was measured before that.

**Strictly an addition.** ``p``, ``q``, ``rho`` and ``r`` are never touched, and
neither are the anchor-noise ``spread`` / ``against_drift`` tables: nothing they
summarise changes when a column is added beside it (both iterate an explicit
:data:`method.anchor_noise.COMPONENTS`). Re-deriving ``z`` instead would
silently re-anchor every run onto whichever ``v_0`` the store holds *now*, and
exp3 is known to sit on several distinct base measurements -- see
:mod:`method.visualization.latent_audit`.

Convention-agnostic on purpose: a run still carrying projections (one the cosine
backfill could not reach) gets its norm filled in just the same, and that norm
is precisely the divisor the conversion will need. The ``z_convention`` marker,
not this field, is what says which units ``p`` and ``q`` are in.

**``trajectory.json`` is the output; the store is only an input.** Same shape as
:mod:`method.backfill_latent_cosine` and :mod:`method.backfill_se`: run it once
where the store lives, sync ``trajectories/`` back, and no analysis machine
needs the store again. A checkpoint whose ``h_neutral`` tensor is not reachable
from *this* machine is reported and left alone, never guessed at.

Idempotent, and needs no marker to be: a block that already carries the field is
skipped, so a partial pass can be resumed as more of the store becomes
reachable, and re-running costs one tensor read per unfilled checkpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch

from method import anchor_noise
from method.backfill_latent_cosine import iter_anchor_noise, iter_runs
from method.config import Backend
from method.latent import H_NORM
from method.steps import Artifacts, h_neutral_path
from method.store import Store, atomic_file
from method.utils import trajectories_root

logger = logging.getLogger("backfill_h_norm")


@dataclass
class Report:
    """What a pass over the run directories found and changed."""

    #: Files rewritten with at least one norm added.
    updated: list[Path] = field(default_factory=list)
    #: ``z`` blocks that gained the field.
    filled: int = 0
    #: Anchor-noise rows that gained the field.
    anchor_rows: int = 0
    #: Files whose every block already carried it.
    already: int = 0
    #: Runs that record no ``z`` at all -- branch endpoints measure only ``b``
    #: (see :class:`method.config.MeasurementLevel`), so they have no norm to
    #: record and are not a gap to be filled later.
    without_z: int = 0
    #: Files with a block whose activation tensor is not on this machine. Left
    #: entirely alone, so a file is never half-filled.
    unreachable: list[Path] = field(default_factory=list)
    #: Files predating the current run format, which nothing downstream loads.
    skipped: list[Path] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{len(self.updated)} file(s) updated, {self.filled} z block(s) "
            f"filled, {self.anchor_rows} anchor-noise row(s) filled, "
            f"{self.already} already complete, {self.without_z} carry no z, "
            f"{len(self.unreachable)} unreachable, {len(self.skipped)} skipped"
        )


def norm_at(path: Path, layer: int) -> float:
    """``||h_neutral||`` at ``layer`` from a ``mean_by_layer.pt``."""
    return float(torch.load(path, weights_only=False)[layer].float().norm())


def convert_run(
    path: Path, store: Store, report: Report, *, dry_run: bool = False
) -> None:
    """Fill in every ``z`` block in one ``trajectory.json``, or none of them.

    All-or-nothing per run: the norms are gathered before anything is written,
    so a run that is only partly reachable is left untouched and retried later
    rather than ending up with the field on some checkpoints and not others --
    which would plot as a series with holes in it.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config")
    if config is None or "model" not in config:
        # Pre-dates the current run format; `collect.py` cannot load these
        # either, so nothing downstream would read the result.
        report.skipped.append(path)
        return

    layer = config["model"]["layer"]
    blocks = [
        (step, source, z)
        for step in payload["steps"]
        for source, z in (step.get("z") or {}).items()
    ]
    if not blocks:
        report.without_z += 1
        return

    missing = [(step, source, z) for step, source, z in blocks if H_NORM not in z]
    if not missing:
        report.already += 1
        return

    norms: dict[tuple[str, str], float] = {}
    for step, source, _ in missing:
        key = (step["weights_id"], source)
        if key in norms:
            continue
        tensor = h_neutral_path(store, step["weights_id"], source)
        if not tensor.exists():
            report.unreachable.append(path)
            return
        norms[key] = norm_at(tensor, layer)

    for step, source, z in missing:
        z[H_NORM] = norms[(step["weights_id"], source)]
        report.filled += 1

    _write(path, payload, report, dry_run=dry_run)


def convert_anchor_noise(
    path: Path, store: Store, report: Report, *, dry_run: bool = False
) -> None:
    """Fill in every row of one anchor-noise summary, or none of them.

    Each row needs the norm of the activation *its own replicate* drew, not the
    checkpoint's production one -- that a replicate re-derives ``h_neutral``
    from scratch is the whole point of the sweep, and here it buys something
    extra: the spread of ``h_norm`` across replicates is the sampling error on
    the norm itself, which is what says whether norm growth along a trunk is a
    real effect or anchor noise.

    ``spread`` and ``against_drift`` are deliberately not rebuilt. Both iterate
    :data:`method.anchor_noise.COMPONENTS`, so a new column beside the ones they
    read changes nothing in them, and rewriting tables that no input of theirs
    has moved would only invite the question of which pass produced them.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    layer = payload["model"]["layer"]
    source = payload["h_neutral_source"]
    rows = payload["latents"]

    missing = [row for row in rows if H_NORM not in row]
    if not missing:
        report.already += 1
        return

    norms: dict[tuple[str, int], float] = {}
    for row in missing:
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
        norms[key] = norm_at(tensor, layer)

    for row in missing:
        row[H_NORM] = norms[(row["weights_id"], row["replicate"])]
        report.anchor_rows += 1

    _write(path, payload, report, dry_run=dry_run)


def _write(path: Path, payload: dict, report: Report, *, dry_run: bool) -> None:
    report.updated.append(path)
    if dry_run:
        return
    with atomic_file(path) as scratch:
        scratch.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
        logger.debug("filled %s", path)
    if report.skipped:
        logger.warning(
            "%d run(s) predate the current format and were skipped: %s",
            len(report.skipped),
            ", ".join(str(p.parent.name) for p in report.skipped[:5]),
        )
    if report.unreachable:
        logger.warning(
            "%d file(s) have no h_neutral tensor on this machine, so their z "
            "still carries no %s. Re-run this where the store lives, then sync "
            "trajectories/ back. First few: %s",
            len(report.unreachable),
            H_NORM,
            ", ".join(str(p.parent.name) for p in report.unreachable[:5]),
        )


if __name__ == "__main__":
    main()
