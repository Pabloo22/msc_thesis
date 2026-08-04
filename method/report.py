"""Human-readable summaries of what a run cost and what the rest will cost.

    poetry run python -m method.report                     # every family on disk
    poetry run python -m method.report --run trajectories/EXP3_..._seed0
    poetry run python -m method.report --family EXP3 --email

Two audiences, one set of numbers. On the box these bodies go out by email so a
trajectory that broke can be noticed while it is still costing money (see
:mod:`method.notify`); on a laptop the same tables print to a terminal when
deciding whether an experiment is affordable. Keeping both on one renderer is
what stops the emailed figures and the quoted figures from drifting apart.

Plain text, deliberately: it is read in a mail client that may not render
anything else, and it should paste into a lab notebook unchanged.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from method import timing
from method.config import TrajectoryConfig
from method.notify import Notifier
from method.sync import format_unsynced
from method.timing import (
    format_cost,
    format_duration,
    hourly_rate,
)
from method.utils import DOTENV_PATH, load_dotenv

logger = logging.getLogger(__name__)

FAMILY_ENV = "MSC_FAMILY"
FAMILY_INDEX_ENV = "MSC_FAMILY_INDEX"
FAMILY_TOTAL_ENV = "MSC_FAMILY_TOTAL"


@dataclass(frozen=True)
class FamilyContext:
    """Which trajectory of a family this process is, if it is part of one.

    ``scripts/run_family.sh`` exports these because a trajectory launched from
    it has no other way to know: each one is its own process, and the thing
    worth reporting -- how much of the *family* is left -- is invisible from
    inside any single run.
    """

    name: str
    index: int
    total: int

    @classmethod
    def from_env(cls) -> FamilyContext | None:
        name = os.environ.get(FAMILY_ENV, "").strip()
        try:
            index = int(os.environ.get(FAMILY_INDEX_ENV, ""))
            total = int(os.environ.get(FAMILY_TOTAL_ENV, ""))
        except ValueError:
            return None
        if not name or total <= 0:
            return None
        return cls(name=name, index=index, total=total)

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.index)


def _rate_note() -> str:
    rate = hourly_rate()
    return f" at ${rate:,.2f}/h" if rate is not None else ""


def _line(label: str, value: str) -> str:
    return f"{label:<16}{value}"


def _cost_suffix(seconds: float) -> str:
    cost = format_cost(seconds)
    return f"  ({cost})" if cost else ""


def trajectory_report(
    cfg: TrajectoryConfig,
    run_dir: Path,
    *,
    elapsed: float,
    error: BaseException | None = None,
    traceback_text: str = "",
    unsynced: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    """Subject and body for one finished or failed trajectory.

    The failure case leads with the traceback rather than the tables: the
    decision it exists to support is "stop the box or not", and nobody should
    have to scroll past a timing breakdown to make it.

    ``unsynced`` is :attr:`method.sync.Syncer.unsynced` -- what never reached
    the remote. It rides in the subject as well as the body because it changes
    the same decision the traceback does, in the opposite direction: an
    otherwise perfect run whose artifacts are still only on the box is the one
    case where stopping the box loses work.
    """
    records = timing.read_stages(timing.stage_log_path(run_dir))
    family = FamilyContext.from_env()
    failed = error is not None
    unsynced = unsynced or {}

    # "FAILED" leads the subject rather than sitting mid-sentence, so the one
    # thing worth acting on is legible in a phone's notification preview.
    prefix = "FAILED: " if failed else ""
    outcome = "died after" if failed else "done in"
    subject = (
        f"{prefix}{cfg.name} seed {cfg.seed} — {outcome} {format_duration(elapsed)}"
    )
    if family is not None:
        subject += f" ({family.index}/{family.total})"
    if unsynced:
        subject += f" +{len(unsynced)} unsynced"

    parts: list[str] = []

    if failed:
        parts.append(
            "This run stopped early. The GPU is still rented until you stop it."
        )
        parts.append(f"{type(error).__name__}: {error}")
        if traceback_text:
            parts.append(traceback_text.rstrip())

    # Above the timing tables: this is the other thing that makes stopping the
    # box the wrong move, and it should not need scrolling to either.
    unsynced_text = format_unsynced(unsynced)
    if unsynced_text:
        parts.append(unsynced_text)

    parts.append(
        "\n".join(
            [
                _line("Trajectory", cfg.name),
                _line("Seed", str(cfg.seed)),
                _line("Model", cfg.model.name),
                _line("Steps", str(len(cfg.steps))),
                _line("Status", "FAILED" if failed else "ok"),
                _line("Wall time", format_duration(elapsed) + _cost_suffix(elapsed)),
                _line("Run dir", str(run_dir)),
            ]
        )
    )

    parts.append("Where the time went\n" + timing.format_stage_table(records))
    parts.append("Per checkpoint\n" + timing.format_checkpoint_table(records))

    # What a resumed run would still have to do. On a failure this is the
    # number that decides whether to relaunch on the same box or give up on it;
    # on a success it should be zero, and anything else means the run ended
    # with stages outstanding.
    remaining = timing.estimate_remaining(records, len(cfg.steps))
    if remaining:
        parts.append(
            _line("Left to do", format_duration(remaining) + _cost_suffix(remaining))
            + "\n(a resume skips whatever is already in the store)"
        )

    if family is not None:
        parts.append(_family_section(family))

    return subject, "\n\n".join(parts) + "\n"


def job_report(
    title: str,
    *,
    elapsed: float,
    detail: str = "",
    error: BaseException | None = None,
    traceback_text: str = "",
) -> tuple[str, str]:
    """Subject and body for a job with no trajectory structure to break down.

    :mod:`method.probe_base` is the case: it is just as long and just as
    unattended as a trajectory, and just as expensive to leave running after it
    has died, but it has no checkpoints and so no per-stage table to show.
    """
    failed = error is not None
    prefix = "FAILED: " if failed else ""
    outcome = "died after" if failed else "done in"
    subject = f"{prefix}{title} — {outcome} {format_duration(elapsed)}"

    parts: list[str] = []
    if failed:
        parts.append(
            "This run stopped early. The GPU is still rented until you stop it."
        )
        parts.append(f"{type(error).__name__}: {error}")
        if traceback_text:
            parts.append(traceback_text.rstrip())
    parts.append(
        "\n".join(
            [
                _line("Job", title),
                _line("Status", "FAILED" if failed else "ok"),
                _line("Wall time", format_duration(elapsed) + _cost_suffix(elapsed)),
            ]
        )
    )
    if detail:
        parts.append(detail.rstrip())
    return subject, "\n\n".join(parts) + "\n"


def _family_section(family: FamilyContext) -> str:
    """Progress and projected cost for the family this run belongs to."""
    runs = [r for r in timing.read_runs() if timing.in_group(r, family.name)]
    done = sum(r.seconds for r in runs)
    lines = [
        f"Family {family.name}: {family.index}/{family.total} trajectories launched",
        _line("Spent so far", format_duration(done) + _cost_suffix(done)),
    ]
    left = timing.estimate_family_remaining(
        runs, group=family.name, remaining=family.remaining
    )
    if left is not None:
        lines.append(
            _line("Estimated left", format_duration(left) + _cost_suffix(left))
        )
        lines.append(
            _line(
                "Projected total",
                format_duration(done + left) + _cost_suffix(done + left),
            )
        )
        lines.append(
            f"(mean of {sum(1 for r in runs if r.ok)} finished trajectories{_rate_note()})"
        )
    return "\n".join(lines)


def family_report(group: str, *, mock: bool = False, note: str = "") -> tuple[str, str]:
    """Subject and body for a whole family, once every trajectory has run.

    ``note`` is prepended and takes over the subject line. It carries what only
    the shell wrapper knows -- most importantly that the family script itself
    exited non-zero, which is the one failure a trajectory that was killed
    outright cannot report for itself.
    """
    runs = [r for r in timing.read_runs(mock=mock) if timing.in_group(r, group)]
    total = sum(r.seconds for r in runs)
    failed = [r for r in runs if not r.ok]

    subject = (
        f"{group or 'all runs'} complete — {len(runs)} trajectories in "
        f"{format_duration(total)}"
    )
    if failed:
        subject = f"{group or 'all runs'} finished with {len(failed)} failure(s)"
    if note:
        subject = f"{group or 'all runs'} stopped — {note.splitlines()[0][:90]}"

    rows = [
        [
            f"{run.trajectory} seed{run.seed}",
            run.status,
            format_duration(run.seconds),
            format_cost(run.seconds),
        ]
        for run in runs
    ]
    sections = [
        "\n".join(
            [
                _line("Family", group or "(all)"),
                _line("Trajectories", str(len(runs))),
                _line("Failed", str(len(failed))),
                _line("Wall time", format_duration(total) + _cost_suffix(total)),
            ]
        ),
        (
            timing.format_table(["trajectory", "status", "wall", "cost"], rows)
            if rows
            else "(nothing logged yet)"
        ),
        f"Logged in {timing.run_log_path(mock=mock)}",
    ]
    if note:
        sections.insert(0, note.rstrip())
    return subject, "\n\n".join(sections) + "\n"


def local_summary(*, mock: bool = False) -> str:
    """Every family on disk, for a terminal rather than an inbox."""
    runs = timing.read_runs(mock=mock)
    if not runs:
        return f"No runs logged yet in {timing.run_log_path(mock=mock)}"
    groups = sorted({run.group for run in runs})
    rows = []
    for group in groups:
        mine = [r for r in runs if r.group == group]
        total = sum(r.seconds for r in mine)
        rows.append(
            [
                group or "(ungrouped)",
                str(len(mine)),
                str(sum(1 for r in mine if not r.ok)),
                format_duration(total),
                format_cost(total),
            ]
        )
    grand = sum(run.seconds for run in runs)
    rows.append(
        ["TOTAL", str(len(runs)), "", format_duration(grand), format_cost(grand)]
    )
    return (
        timing.format_table(["family", "runs", "failed", "wall", "cost"], rows)
        + f"\n\n{timing.run_log_path(mock=mock)}{_rate_note()}"
    )


def run_summary(run_dir: Path) -> str:
    """The per-stage and per-checkpoint tables for one run directory."""
    records = timing.read_stages(timing.stage_log_path(run_dir))
    total = sum(r.seconds for r in records)
    return "\n\n".join(
        [
            f"{run_dir.name}\n"
            + _line("Wall time", format_duration(total) + _cost_suffix(total)),
            timing.format_stage_table(records),
            timing.format_checkpoint_table(records),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=None,
        help="a trajectory run directory to break down",
    )
    parser.add_argument(
        "--family", default="", help="restrict the summary to one family, e.g. EXP3"
    )
    parser.add_argument(
        "--mock", action="store_true", help="read trajectories-mock/ instead"
    )
    parser.add_argument(
        "--email",
        action="store_true",
        help="also send the family summary (needs MSC_RESEND_API_KEY)",
    )
    parser.add_argument(
        "--note",
        default="",
        help="text to lead the emailed summary with, e.g. why a family stopped",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # The API key and the hourly rate both live in .env, and nothing imports it
    # at load time. Without this the family summary that run_family.sh sends
    # from its exit trap would find no key and quietly send nothing -- the one
    # mail that reports the family *crashed*.
    load_dotenv(DOTENV_PATH)

    if args.run is not None:
        print(run_summary(args.run))
        return

    if args.email:
        subject, body = family_report(args.family, mock=args.mock, note=args.note)
        print(body)
        notifier = Notifier.from_env()
        logger.info(notifier.describe())
        notifier.send(subject, body)
        return

    print(local_summary(mock=args.mock))


if __name__ == "__main__":
    main()
