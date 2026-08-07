"""Add ``SE(b)`` to behaviour measurements taken before it was recorded.

    poetry run python -m method.backfill_se --dry-run
    poetry run python -m method.backfill_se
    poetry run python -m method.backfill_se --mock

:func:`method.steps.measure_behavior` records the analytic standard error from
now on, but it returns early whenever ``behavior.csv`` already exists, so
checkpoints measured before that change keep their old summary through any
number of resumed runs. This script fills them in.

**``trajectory.json`` is the output; the store is only an input.** The
per-generation scores ``SE`` is derived from live in ``behavior.csv``, which
exists only inside the store -- and the store is hundreds of gigabytes of
adapters and hidden-state tensors that no analysis machine should need. So this
writes the recovered numbers into the run directories, which are small, already
synced for plotting, and self-contained by design (the same reasoning as
``probe_base.write_summary``). Run it once wherever the store lives, sync
``trajectories/``, and nothing downstream has to touch the store again.

Consequently a checkpoint whose ``behavior.csv`` is not reachable from *this*
machine is reported and left alone, never guessed at: ``SE`` cannot be recovered
from the summary, since the summary records the spread across all rows and
throws away the per-question structure the formula needs.

Idempotent, like every other write in this codebase: a step that already carries
``SE`` is skipped, so re-running after syncing more of the store fills in only
what newly became reachable.
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

from method.config import Backend
from method.noise import behavior_summary
from method.steps import Artifacts
from method.store import Store, atomic_file
from method.utils import trajectories_root

logger = logging.getLogger("backfill_se")


@dataclass
class Report:
    """What a pass over the run directories found and changed."""

    #: Runs whose ``trajectory.json`` gained at least one ``SE``.
    updated: list[Path] = field(default_factory=list)
    #: Runs written in a format this script cannot read (no ``config`` block).
    skipped: list[Path] = field(default_factory=list)
    #: Checkpoints that already carried ``SE``.
    already: int = 0
    #: Checkpoints filled in from a reachable ``behavior.csv``.
    filled: int = 0
    #: Checkpoints whose ``behavior.csv`` is not on this machine.
    unreachable: int = 0
    #: Checkpoints whose recorded mean disagreed with the store's own rows,
    #: described as ``"<run> t=<t>: <recorded> -> <recomputed>"``. See
    #: :func:`backfill_run` for why these exist and why they are not silent.
    divergent: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{len(self.updated)} run(s) updated, {self.filled} checkpoint(s) "
            f"filled, {self.already} already had SE, {self.unreachable} "
            f"unreachable, {len(self.skipped)} run(s) skipped, "
            f"{len(self.divergent)} value(s) changed"
        )


def se_key(trait: str) -> str:
    """The field :func:`method.noise.behavior_summary` records ``SE`` under."""
    return f"{trait}_se"


def recompute(store: Store, weights_id: str, trait: str) -> dict[str, float] | None:
    """The full summary for one checkpoint, or ``None`` if its CSV is missing.

    Goes through :func:`method.noise.behavior_summary` rather than computing
    ``SE`` alone so that a backfilled artifact is indistinguishable from a
    freshly measured one -- including the fields that already existed, which are
    recomputed from the same rows and must therefore come out identical.
    """
    csv = store.trait_measurement(weights_id, trait, Artifacts.BEHAVIOR_CSV)
    if not csv.exists():
        return None
    return behavior_summary(pd.read_csv(csv), trait)


def backfill_run(
    path: Path,
    store: Store,
    report: Report,
    *,
    dry_run: bool = False,
    refresh_store: bool = False,
) -> None:
    """Fill in every reachable ``SE`` for one ``trajectory.json``.

    ``refresh_store`` additionally rewrites the store's ``behavior.json``. Off
    by default, and only cosmetic now that
    :func:`method.steps.behavior_record` derives a run's summary from
    ``behavior.csv`` instead of that file -- nothing downstream reads it. It is
    also expensive in a way its size does not suggest: a measurement bundle is
    a single remote object, so touching a 200-byte JSON inside one re-uploads
    the whole bundle, hidden-state tensors included. Across a store this is
    tens of gigabytes of transfer for edits that belong to ``trajectory.json``.

    Recomputing the whole summary can *change* a recorded score, and where it
    does that is reported rather than applied quietly. It happens because the
    behaviour eval generates at temperature 1.0 with no sampling seed (see
    ``eval/eval_persona.py``, which picks that temperature whenever
    ``n_per_question > 1``), so two machines that both measured the same
    checkpoint before syncing produced different numbers, and each recorded its
    own in its run files while the store kept only one. Converging on the store
    is the right resolution -- one ``weights_id`` must mean one measurement, and
    the shared ``t = 0`` column of the exp2 design depends on it -- but it is a
    change to published numbers and has to be seen.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config")
    if config is None or "trait" not in config:
        # Pre-dates the current run format. `collect.py` cannot load these
        # either, so there is nothing downstream that would read the result.
        report.skipped.append(path)
        return

    trait = config["trait"]
    key = se_key(trait)
    dirty = False

    for step in payload["steps"]:
        behavior = step.get("behavior") or {}
        if key in behavior:
            report.already += 1
            continue
        summary = recompute(store, step["weights_id"], trait)
        if summary is None:
            report.unreachable += 1
            continue
        recorded = behavior.get(trait)
        if recorded is not None and not math.isclose(
            recorded, summary[trait], rel_tol=1e-9, abs_tol=1e-9
        ):
            report.divergent.append(
                f"{path.parent.name} t={step['t']}: "
                f"{recorded:.4f} -> {summary[trait]:.4f}"
            )
        step["behavior"] = summary
        report.filled += 1
        dirty = True
        if refresh_store and not dry_run:
            _refresh_store_summary(store, step["weights_id"], trait, summary)

    if not dirty:
        return
    report.updated.append(path)
    if dry_run:
        return
    with atomic_file(path) as scratch:
        scratch.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _refresh_store_summary(
    store: Store, weights_id: str, trait: str, summary: dict[str, float]
) -> None:
    with atomic_file(
        store.trait_measurement(weights_id, trait, Artifacts.BEHAVIOR_JSON)
    ) as scratch:
        scratch.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def iter_runs(root: Path) -> Iterator[Path]:
    """Every ``trajectory.json`` under a trajectories root, in a stable order."""
    return iter(sorted(root.glob("*/trajectory.json")))


def backfill(
    root: Path, store: Store, *, dry_run: bool = False, refresh_store: bool = False
) -> Report:
    report = Report()
    for path in iter_runs(root):
        backfill_run(
            path, store, report, dry_run=dry_run, refresh_store=refresh_store
        )
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
    parser.add_argument(
        "--refresh-store",
        action="store_true",
        help=(
            "also rewrite behavior.json in the store; cosmetic (nothing reads "
            "it) and costs a full re-upload of every bundle it touches"
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    backend = Backend.MOCK if args.mock else Backend.REAL
    root = trajectories_root(mock=args.mock)
    report = backfill(
        root,
        Store.for_backend(backend),
        dry_run=args.dry_run,
        refresh_store=args.refresh_store,
    )

    logger.info("%s%s", report.summary(), " (dry run)" if args.dry_run else "")
    for path in report.updated:
        logger.debug("updated %s", path)
    if report.skipped:
        logger.warning(
            "%d run(s) predate the current format and were skipped: %s",
            len(report.skipped),
            ", ".join(str(p.parent.name) for p in report.skipped[:5]),
        )
    if report.divergent:
        logger.warning(
            "%d checkpoint(s) had a recorded score that disagreed with the "
            "store's own per-generation rows, and were reset to the store's "
            "value. The eval samples at temperature 1.0 with no seed, so a "
            "checkpoint measured independently on two machines has two valid "
            "readings; one weights_id must resolve to one of them. Changed:\n  %s",
            len(report.divergent),
            "\n  ".join(report.divergent[:10]),
        )
    if report.unreachable:
        logger.warning(
            "%d checkpoint(s) have no behavior.csv on this machine, so their SE "
            "could not be recovered. Re-run this where the full store lives "
            "(the GPU box) and sync trajectories/ back.",
            report.unreachable,
        )


if __name__ == "__main__":
    main()
