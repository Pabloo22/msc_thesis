"""How long a trajectory's stages took, and what the rest of it will cost.

Rented GPU time is the budget this project actually spends, and the only way to
project it before it is spent is to measure the units that make it up. Every
stage a run executes is timed and appended to ``timings.jsonl`` in the run
directory; every finished trajectory is appended to a family-wide
``runlog.jsonl`` beside the run directories. Both are plain jsonl, appended and
flushed as work completes, so a preempted box still leaves behind everything it
had learned about its own speed.

The estimates here are deliberately simple -- a per-stage median over what this
run has already done, multiplied by what it has left. A trajectory repeats the
same handful of stages at every checkpoint, so the median is over comparable
work rather than a curve fit to noise, and it needs no calibration when the
model, the example counts or the box's GPU change.

Nothing in this module knows how a report is delivered; see :mod:`method.notify`.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from method.utils import read_jsonl, trajectories_root

logger = logging.getLogger(__name__)

STAGE_LOG = "timings.jsonl"
RUN_LOG = "runlog.jsonl"

GPU_HOURLY_USD_ENV = "MSC_GPU_HOURLY_USD"

#: Stages executed once per checkpoint, in the order the runner performs them.
CHECKPOINT_STAGES = ("behavior", "persona_vector", "h_neutral", "latent", "probes")
#: Stages executed once per *step*, i.e. at every checkpoint but the last.
STEP_STAGES = ("delta_p", "train")

#: Stages performed once per DeltaP *view* rather than once per checkpoint
#: (see :class:`method.config.DeltaPView`). They are timed apart because the
#: views do very different work for the same name: one re-reads answers
#: generated at ``t = 0``, one generates its own, and one only re-projects
#: tensors already on disk. Pooling them would price a generation pass at the
#: median of a projection.
PER_VIEW_STAGES = ("probes", "delta_p")

#: The view whose stages keep their unqualified names -- the default one, whose
#: suffix is empty -- so a run recorded before the view axis existed still
#: matches the stages it will perform.
_DEFAULT_VIEWS: tuple[str, ...] = ("",)


def stage_for(stage: str, view: str) -> str:
    """``stage`` as recorded for one DeltaP view, named by its suffix.

    The single place the naming rule lives: the runner writes these names and
    the estimator enumerates them, and a rule spelled out twice is a rule that
    can disagree with itself.
    """
    if stage not in PER_VIEW_STAGES or not view:
        return stage
    return f"{stage}_{view}"


def base_stage(stage: str) -> str:
    """The stage a recorded name is a per-view variant of, or itself."""
    for candidate in PER_VIEW_STAGES:
        if stage == candidate or stage.startswith(f"{candidate}_"):
            return candidate
    return stage


def _expand(stages: Sequence[str], views: Sequence[str]) -> list[str]:
    """``stages`` with the per-view ones repeated once per view."""
    out: list[str] = []
    for stage in stages:
        if stage in PER_VIEW_STAGES:
            out.extend(stage_for(stage, view) for view in views)
        else:
            out.append(stage)
    return out

#: Below this, a stage cannot have done real work: every genuine stage loads a
#: model onto the GPU first, which alone costs tens of seconds. What lands
#: under it is a resumed run's cache hit, skipping straight past an artifact
#: that already exists. Those are excluded from the estimates, since a resumed
#: run would otherwise conclude from a directory of instant hits that the
#: expensive work ahead of it is instant too.
_TRIVIAL_SECONDS = 5.0


@dataclass(frozen=True)
class StageRecord:
    """One timed stage, as read back from ``timings.jsonl``."""

    trajectory: str
    seed: int
    t: int
    stage: str
    seconds: float
    ok: bool

    @property
    def did_work(self) -> bool:
        """Whether this record is evidence about how long the stage takes.

        A cache hit says nothing about the cost of doing the work, and a stage
        that raised part-way through only ever understates it.
        """
        return self.ok and self.seconds >= _TRIVIAL_SECONDS


@dataclass(frozen=True)
class RunRecord:
    """One finished (or failed) trajectory, as read back from ``runlog.jsonl``."""

    trajectory: str
    group: str
    seed: int
    status: str
    seconds: float

    @property
    def ok(self) -> bool:
        return self.status == "ok"


# --- recording ------------------------------------------------------------


def _append(path: Path, record: dict) -> None:
    """Append one jsonl record, flushed before returning.

    Only ``flush``, not ``fsync``, for the reason :mod:`method.eval_progress`
    gives: the failure being insured against is a dying process, not a dying
    disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
    except OSError as exc:
        # Bookkeeping must never be the thing that kills a trajectory.
        logger.warning("could not append to %s: %s", path, exc)


class StageTimer:
    """Times each stage of one trajectory into the run directory.

    Records land as each stage ends rather than at the end of the run, so the
    timings of a run that dies are exactly the timings it had earned. That is
    also what makes the estimates work during a run rather than only after it:
    :func:`estimate_remaining` reads the same file that is still being written.
    """

    def __init__(self, path: Path | None, *, trajectory: str, seed: int) -> None:
        self.path = path
        self.trajectory = trajectory
        self.seed = seed

    @classmethod
    def disabled(cls) -> StageTimer:
        """A timer that records nothing.

        Lets callers that have no run directory to write into -- tests, and
        anything measuring a checkpoint outside a trajectory -- run the same
        code path as the runner rather than branching around it.
        """
        return cls(None, trajectory="", seed=0)

    @contextmanager
    def stage(self, name: str, t: int) -> Iterator[None]:
        """Time one stage, recording it whether or not the body succeeds."""
        started = time.monotonic()
        ok = True
        try:
            yield
        except BaseException:
            ok = False
            raise
        finally:
            # No early `return` here: returning out of a `finally` discards
            # whatever exception is propagating, which would turn a disabled
            # timer into a silent swallower of every failure it wraps.
            if self.path is not None:
                _append(
                    self.path,
                    {
                        "trajectory": self.trajectory,
                        "seed": self.seed,
                        "t": t,
                        "stage": name,
                        "seconds": round(time.monotonic() - started, 3),
                        "ok": ok,
                        "at": _utc_now(),
                    },
                )


def record_run(
    *,
    trajectory: str,
    group: str,
    seed: int,
    status: str,
    seconds: float,
    mock: bool = False,
) -> None:
    """Append one trajectory's outcome to the family-wide run log."""
    _append(
        run_log_path(mock=mock),
        {
            "trajectory": trajectory,
            "group": group,
            "seed": seed,
            "status": status,
            "seconds": round(seconds, 3),
            "at": _utc_now(),
        },
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_log_path(*, mock: bool = False) -> Path:
    """Where finished trajectories are logged, beside the run directories.

    Next to ``trajectories/`` rather than inside any one run because its whole
    purpose is to be read across runs: a single trajectory's own timings live
    in its run directory, and the only thing that can say how long a *family*
    takes is a file that outlives each of its processes.
    """
    return trajectories_root(mock=mock) / RUN_LOG


def stage_log_path(run_dir: Path) -> Path:
    return run_dir / STAGE_LOG


# --- reading --------------------------------------------------------------


def read_stages(path: Path) -> list[StageRecord]:
    return [
        StageRecord(
            trajectory=str(raw.get("trajectory", "")),
            seed=int(raw.get("seed", 0)),
            t=int(raw["t"]),
            stage=str(raw["stage"]),
            seconds=float(raw["seconds"]),
            ok=bool(raw.get("ok", True)),
        )
        for raw in read_jsonl(path)
        if "t" in raw and "stage" in raw and "seconds" in raw
    ]


def failed_stage(records: Sequence[StageRecord]) -> StageRecord | None:
    """The stage a run died in, if its stage log shows one.

    The last unsuccessful record rather than the first: a resumed run can carry
    older failures in the same log, and what a report needs to name is where
    *this* attempt stopped.
    """
    return next((r for r in reversed(records) if not r.ok), None)


def read_runs(path: Path | None = None, *, mock: bool = False) -> list[RunRecord]:
    return [
        RunRecord(
            trajectory=str(raw.get("trajectory", "")),
            group=str(raw.get("group", "")),
            seed=int(raw.get("seed", 0)),
            status=str(raw.get("status", "ok")),
            seconds=float(raw["seconds"]),
        )
        for raw in read_jsonl(run_log_path(mock=mock) if path is None else path)
        if "seconds" in raw
    ]


# --- estimation -----------------------------------------------------------


def expected_units(
    n_steps: int, views: Sequence[str] = _DEFAULT_VIEWS
) -> list[tuple[int, str]]:
    """Every ``(checkpoint, stage)`` a trajectory of ``n_steps`` will execute.

    The last checkpoint is measured but never trained from -- it has no
    successor -- so the step stages stop one checkpoint early, exactly as
    :func:`method.run_trajectory.run` does.

    ``views`` names the DeltaP views the run measures, by suffix. A run that
    measures only a non-default view never performs a stage called ``probes``
    at all, so enumerating it would leave the estimate counting work that is
    never going to happen and an ETA that never converges.
    """
    checkpoint = _expand(CHECKPOINT_STAGES, views)
    step = _expand(STEP_STAGES, views)
    units: list[tuple[int, str]] = []
    for t in range(n_steps + 1):
        units.extend((t, stage) for stage in checkpoint)
        if t < n_steps:
            units.extend((t, stage) for stage in step)
    return units


def typical_seconds(records: Sequence[StageRecord]) -> dict[str, float]:
    """Median duration per stage, over records that actually did work.

    Median rather than mean because the distribution has a hard floor and a
    long tail: a judge API that retries, or a checkpoint whose merge chain is
    one adapter longer, pulls a mean around in a way it does not deserve.
    """
    by_stage: dict[str, list[float]] = {}
    for record in records:
        if record.did_work:
            by_stage.setdefault(record.stage, []).append(record.seconds)
    return {stage: statistics.median(xs) for stage, xs in by_stage.items()}


def estimate_remaining(
    records: Sequence[StageRecord],
    n_steps: int,
    views: Sequence[str] = _DEFAULT_VIEWS,
) -> float | None:
    """Seconds of work left in a trajectory, or ``None`` with nothing to go on.

    Any stage this run has never performed is priced at the median of the ones
    it has, which is crude but bounded: the alternative is refusing to estimate
    at all until every stage has been seen once, and the first checkpoint is
    where an estimate is worth the most.
    """
    done = {(record.t, record.stage) for record in records if record.ok}
    typical = typical_seconds(records)
    if not typical:
        return None
    fallback = statistics.median(typical.values())
    remaining = [unit for unit in expected_units(n_steps, views) if unit not in done]
    return sum(typical.get(stage, fallback) for _, stage in remaining)


def in_group(run: RunRecord, group: str) -> bool:
    """Whether ``run`` belongs to ``group``, compared case-insensitively.

    The registry names families in upper case (``EXP3_BASELINE_...``) and that
    is the spelling ``scripts/run_family.sh`` is invoked with, while
    :attr:`TrajectoryConfig.group` records the lower-case ``"exp3"``. They are
    the same family, and a case-sensitive filter silently reports an empty one
    -- a summary saying a family cost nothing, rather than saying it failed.

    An empty ``group`` matches everything, which is what makes "no --family"
    mean "all of them".
    """
    return not group or run.group.casefold() == group.casefold()


def estimate_family_remaining(
    runs: Sequence[RunRecord], *, group: str, remaining: int
) -> float | None:
    """Seconds left in a family, from the mean of its finished trajectories.

    Mean rather than median here, and over the whole family rather than per
    stage: trajectories in one family differ mainly in how much of their prefix
    was already in the store, and the total across the family is what that
    averages out over.
    """
    durations = [run.seconds for run in runs if run.ok and in_group(run, group)]
    if not durations:
        return None
    return statistics.fmean(durations) * max(0, remaining)


# --- cost -----------------------------------------------------------------


def hourly_rate() -> float | None:
    """The box's hourly price, from ``MSC_GPU_HOURLY_USD``.

    Unset means the reports omit money entirely rather than guessing a rate:
    a wrong number in a cost projection is worse than no number.
    """
    raw = os.environ.get(GPU_HOURLY_USD_ENV, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; ignoring", GPU_HOURLY_USD_ENV, raw)
        return None


def format_cost(seconds: float, rate: float | None = None) -> str:
    """``"$4.20"`` for a duration at the configured rate, or ``""`` if unset."""
    rate = hourly_rate() if rate is None else rate
    if rate is None:
        return ""
    return f"${seconds / 3600.0 * rate:,.2f}"


# --- rendering ------------------------------------------------------------


def format_duration(seconds: float) -> str:
    """``"2h 05m"`` / ``"12m 04s"`` / ``"45s"`` -- two units, never more."""
    seconds = max(0.0, seconds)
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A plain-text table, since these are read in an email body.

    First column left-aligned (names), the rest right-aligned (numbers), which
    is what makes a column of durations scannable.
    """
    columns = list(zip(headers, *rows)) if rows else [(h,) for h in headers]
    widths = [max(len(str(cell)) for cell in column) for column in columns]

    def line(cells: Sequence[str]) -> str:
        first, *rest = [str(cell).ljust(width) for cell, width in zip(cells, widths)]
        return "  ".join(
            [first] + [cell.rjust(width) for cell, width in zip(rest, widths[1:])]
        ).rstrip()

    rule = "-" * (sum(widths) + 2 * (len(widths) - 1))
    return "\n".join([line(headers), rule, *(line(row) for row in rows)])


def format_stage_table(records: Sequence[StageRecord]) -> str:
    """Per-stage totals: how the run's wall time was actually distributed."""
    if not records:
        return "(no stages recorded)"
    order = {stage: i for i, stage in enumerate(CHECKPOINT_STAGES + STEP_STAGES)}
    def rank(stage: str) -> tuple[int, str]:
        """A per-source variant sorts beside the stage it varies, not after the
        last one, so the table reads in the order the runner performs them."""
        return order.get(base_stage(stage), len(order)), stage

    by_stage: dict[str, list[StageRecord]] = {}
    for record in records:
        by_stage.setdefault(record.stage, []).append(record)

    rows = []
    for stage in sorted(by_stage, key=rank):
        group = by_stage[stage]
        worked = [r for r in group if r.did_work]
        total = sum(r.seconds for r in group)
        # Failures are counted apart from cache hits even though neither is
        # evidence about how long the stage takes: a stage that burned two
        # minutes and then raised was reported as "cached", which reads as the
        # opposite of what happened.
        failed = sum(1 for r in group if not r.ok)
        cached = len(group) - len(worked) - failed
        notes = [
            f"{cached} cached" if cached else "",
            f"{failed} failed" if failed else "",
        ]
        rows.append(
            [
                stage,
                str(len(group)),
                format_duration(total),
                (
                    format_duration(statistics.median([r.seconds for r in worked]))
                    if worked
                    else "-"
                ),
                ", ".join(n for n in notes if n),
            ]
        )
    total = sum(r.seconds for r in records)
    rows.append(["TOTAL", "", format_duration(total), "", ""])
    return format_table(["stage", "n", "total", "median", ""], rows)


def format_checkpoint_table(records: Sequence[StageRecord]) -> str:
    """Per-checkpoint totals: which checkpoint the time actually went into."""
    if not records:
        return "(no checkpoints recorded)"
    by_t: dict[int, float] = {}
    for record in records:
        by_t[record.t] = by_t.get(record.t, 0.0) + record.seconds
    rows = [
        [f"t={t}", format_duration(total), format_cost(total)]
        for t, total in sorted(by_t.items())
    ]
    return format_table(["checkpoint", "wall", "cost"], rows)
