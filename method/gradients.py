r"""Gradient traces recovered from a family run's console log.

    poetry run python -m method.gradients exp3_box2.log --csv gradients.csv

Every fine-tuning step prints one line per optimizer step -- ``{'loss': ...,
'grad_norm': ...}``, straight from HF Trainer -- and then nothing keeps it:
:data:`method.run_trajectory._ADAPTER_EXCLUDES` drops the optimizer state on the
way into the store, and ``trajectory.json`` records behaviour and latents, never
the training curve. The console log a family run was launched under is
therefore the only place a gradient norm survives on a box that has since been
released, and this module reads it back.

**What a grad-norm line has to be joined against.** By itself it says nothing
about which dataset produced it. Three other lines supply the rest, and the
runner emits them in a fixed order at every step:

* ``Training file: <run>/train_step<N>.jsonl -> <sample_id>.jsonl`` -- the
  content hash of the *examples*, so two steps that trained on an identical
  sample share it whatever trajectory or position they sit at.
* ``--- training step <N> -> <weights_id> ---`` -- training actually ran, and
  the grad-norm lines that follow belong to it.
* ``[skip] adapter for step <N> already trained (<weights_id>)`` -- it did not,
  because the store already held that adapter.

The skip line is what makes the corpus usable. ``weights_id`` is content
addressed over the whole chain, so a step that was a cache hit here was trained
*somewhere*, and if that somewhere is anywhere in the logs given -- or in a
local store still holding its ``trainer_state.json``, which ``--store`` reads --
its curve attaches to this slot too. Without that join the *first* exposure of a
repeated dataset is nearly always missing: it is shared with the baseline
trajectory, so by the time a repeat-exposure trajectory reaches it, it is a
cache hit.

**Reading the result.** :func:`repeat_table` groups slots by
``(sample_id, step)``: identical training examples at an identical depth in the
chain, split by whether that trajectory had already trained on them. The
grouping is what keeps "the second exposure has smaller gradients" from being an
artefact of depth -- in exp3 it puts ``same``'s third step (second exposure to
$d_2$) beside ``diff``'s and ``normal2``'s third steps (first exposure to the
same $d_2$, two fine-tunes deep) rather than beside a step-1 run on the base
model.

The headline number per run is ``grad_norm_init``: the norm at optimizer step 1,
logged while ``learning_rate`` is still 0 and no update from this dataset has
landed. It measures how hard the data pulls the weights it was handed, with no
optimizer transient mixed in. Medians over the whole run sit beside it, and the
loss columns come along free -- on a repeated dataset the loss is the more
direct memorisation signal, and the two are meant to be read together.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: ``>>> Starting trajectory: EXP3_SAME_EVIL_..._SEED3 (95/108)``, from
#: ``scripts/run_family.sh``. Read as a boundary rather than for its key -- the
#: run directory name on the ``Training file:`` line is the real identity, and
#: what this marks is where the exposure counter has to start over.
_TRAJECTORY = re.compile(r">>> Starting trajectory: \S+")
_SAMPLE = re.compile(
    r"Training file: .*/(?P<run>[^/]+)/train_step(?P<step>\d+)\.jsonl"
    r" -> .*/(?P<sample>[0-9a-f]+)\.jsonl"
)
_TRAIN = re.compile(r"--- training step (?P<step>\d+) -> (?P<wid>\S+) ---")
_SKIP = re.compile(
    r"\[skip\] adapter for step (?P<step>\d+) already trained \((?P<wid>\S+)\)"
)
#: HF Trainer's per-step log, printed by the vendored ``training.py`` on stdout.
#: Anchored at line start so a tqdm bar sharing the line cannot match.
_POINT = re.compile(
    r"^\{'loss': (?P<loss>[^,]+), 'grad_norm': (?P<grad_norm>[^,]+), "
    r"'learning_rate': (?P<lr>[^,]+), 'epoch': (?P<epoch>[^}]+)\}"
)
_FINISHED = re.compile(r"^\{'train_runtime': (?P<runtime>[^,]+),")

#: ``exp3_same_evil_hallucination_misaligned_1_evil_qwen2.5-7b-instruct_seed3``.
#: Both groups are best-effort labels for slicing the frames; nothing in the
#: analysis depends on them parsing, so a run directory named otherwise simply
#: gets an empty condition and a null seed.
_RUN_NAME = re.compile(r"^(?P<group>exp\d+)_(?P<condition>[a-z0-9]+)_")
_RUN_SEED = re.compile(r"_seed(?P<seed>\d+)$")


@dataclass(frozen=True)
class GradPoint:
    """One optimizer step, as HF Trainer logged it."""

    step: int
    loss: float
    grad_norm: float
    learning_rate: float
    epoch: float


@dataclass(frozen=True)
class TrainingRun:
    """The gradient trace of one adapter, and where it was recovered from.

    Keyed by ``weights_id`` rather than by trajectory: content addressing means
    one training event serves every trajectory whose chain reaches it, so this
    is the thing that exists once even when many slots point at it.
    """

    weights_id: str
    points: tuple[GradPoint, ...]
    source: str
    #: ``None`` when the run never printed its ``train_runtime`` summary, i.e.
    #: it was interrupted -- the trace is a prefix and the store holds no
    #: adapter for it.
    train_runtime: float | None = None

    @property
    def completed(self) -> bool:
        return self.train_runtime is not None


@dataclass(frozen=True)
class Slot:
    """One ``(trajectory, step)`` fine-tuning event the log describes.

    Distinct from :class:`TrainingRun` because the two are many-to-one: a slot
    is a position in *some* trajectory's chain, and several trajectories'
    slots share a ``weights_id`` whenever their chains agree up to that point.
    """

    run_name: str
    step: int
    weights_id: str
    sample_id: str
    #: 1 the first time this trajectory trains on ``sample_id``, 2 the second,
    #: and so on. The whole point of the module.
    exposure: int
    #: ``True`` when this particular slot is where training ran; ``False`` when
    #: the runner found the adapter already in the store.
    trained_here: bool

    @property
    def condition(self) -> str:
        match = _RUN_NAME.match(self.run_name)
        return match.group("condition") if match else ""

    @property
    def seed(self) -> int | None:
        match = _RUN_SEED.search(self.run_name)
        return int(match.group("seed")) if match else None


@dataclass(frozen=True)
class LogScan:
    """Everything one or more logs said about training."""

    runs: Mapping[str, TrainingRun]
    slots: Sequence[Slot]


def _floats(match: re.Match[str], *names: str) -> Iterator[float]:
    for name in names:
        yield float(match.group(name))


def parse_log(path: Path) -> tuple[dict[str, TrainingRun], list[Slot]]:
    """Read one console log into its training runs and its trajectory slots.

    Streams the file: a paper-scale family log runs to tens of millions of
    lines, almost all of them tqdm redraws.
    """
    runs: dict[str, TrainingRun] = {}
    slots: list[Slot] = []
    #: sample_ids this trajectory has already trained on, oldest first, so the
    #: exposure index is a count rather than a lookup back through ``slots``.
    seen: list[str] = []
    run_name = ""
    pending: tuple[int, str] | None = None  # (step, sample_id)
    open_wid: str | None = None
    points: list[GradPoint] = []

    def close(runtime: float | None) -> None:
        nonlocal open_wid, points
        if open_wid is not None and points:
            runs[open_wid] = TrainingRun(
                weights_id=open_wid,
                points=tuple(points),
                source=f"{path.name}",
                train_runtime=runtime,
            )
        open_wid, points = None, []

    def record(step: int, wid: str, trained_here: bool) -> None:
        if pending is None or pending[0] != step:
            logger.warning(
                "%s: step %d -> %s has no matching 'Training file:' line; "
                "its sample is unknown and it is dropped",
                path.name,
                step,
                wid,
            )
            return
        sample_id = pending[1]
        slots.append(
            Slot(
                run_name=run_name,
                step=step,
                weights_id=wid,
                sample_id=sample_id,
                exposure=seen.count(sample_id) + 1,
                trained_here=trained_here,
            )
        )
        seen.append(sample_id)

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if point := _POINT.match(line):
                if open_wid is not None:
                    loss, grad_norm, lr, epoch = _floats(
                        point, "loss", "grad_norm", "lr", "epoch"
                    )
                    points.append(
                        GradPoint(
                            step=len(points) + 1,
                            loss=loss,
                            grad_norm=grad_norm,
                            learning_rate=lr,
                            epoch=epoch,
                        )
                    )
                continue
            if finished := _FINISHED.match(line):
                close(float(finished.group("runtime")))
                continue
            if _TRAJECTORY.search(line):
                # A trajectory boundary resets the exposure counter even when
                # the previous one died mid-step, which a stray open run is
                # exactly the symptom of.
                close(None)
                run_name, pending, seen = "", None, []
                continue
            if sample := _SAMPLE.search(line):
                close(None)
                run_name = sample.group("run")
                pending = (int(sample.group("step")), sample.group("sample"))
                continue
            if train := _TRAIN.search(line):
                close(None)
                record(int(train.group("step")), train.group("wid"), True)
                open_wid = train.group("wid")
                continue
            if skip := _SKIP.search(line):
                record(int(skip.group("step")), skip.group("wid"), False)
    close(None)
    return runs, slots


def scan_logs(paths: Iterable[Path]) -> LogScan:
    """Parse several logs into one corpus, preferring completed traces.

    Two logs can hold the same ``weights_id`` -- a preempted box that got part
    way and the box that finished the job. The completed one wins; between two
    completed ones the longer trace wins, and they should be identical anyway.
    """
    runs: dict[str, TrainingRun] = {}
    slots: list[Slot] = []
    for path in paths:
        found, run_slots = parse_log(path)
        slots.extend(run_slots)
        for wid, run in found.items():
            if _better(run, runs.get(wid)):
                runs[wid] = run
    return LogScan(runs=runs, slots=_dedupe(slots))


def _better(new: TrainingRun, old: TrainingRun | None) -> bool:
    if old is None:
        return True
    if new.completed != old.completed:
        return new.completed
    return len(new.points) > len(old.points)


def _dedupe(slots: Sequence[Slot]) -> list[Slot]:
    """One row per ``(run_name, step)``; a rerun of a trajectory repeats them.

    The later occurrence wins, and ``trained_here`` is OR-ed across them: a
    trajectory rerun after its adapter landed reports a cache hit for a step it
    itself trained the first time round, and the slot should remember that.
    """
    merged: dict[tuple[str, int], Slot] = {}
    for slot in slots:
        key = (slot.run_name, slot.step)
        if previous := merged.get(key):
            slot = replace(
                slot, trained_here=slot.trained_here or previous.trained_here
            )
        merged[key] = slot
    return list(merged.values())


def store_traces(root: Path) -> dict[str, TrainingRun]:
    """Gradient traces from ``trainer_state.json`` in a local adapter store.

    Exact rather than reconstructed -- HF Trainer writes its own
    ``log_history`` into the checkpoint, and ``_ADAPTER_EXCLUDES`` keeps it --
    but only for the adapters this machine actually holds, which on a laptop is
    whatever :mod:`method.sync` last pulled down. Used to fill gaps in, and to
    cross-check, what the logs give.
    """
    runs: dict[str, TrainingRun] = {}
    for state in sorted((root / "adapters").glob("*/trainer_state.json")):
        history = json.loads(state.read_text(encoding="utf-8")).get("log_history", [])
        points = tuple(
            GradPoint(
                step=entry["step"],
                loss=entry["loss"],
                grad_norm=entry["grad_norm"],
                learning_rate=entry["learning_rate"],
                epoch=entry["epoch"],
            )
            # The final entry is the run summary and carries no gradient.
            for entry in history
            if "grad_norm" in entry
        )
        if points:
            runs[state.parent.name] = TrainingRun(
                weights_id=state.parent.name,
                points=points,
                source="store",
                train_runtime=float(history[-1].get("train_runtime", "nan")),
            )
    return runs


def summarize(run: TrainingRun) -> dict[str, float]:
    """Per-run statistics, all of them over the same set of optimizer steps."""
    grad = np.array([point.grad_norm for point in run.points], dtype=float)
    loss = np.array([point.loss for point in run.points], dtype=float)
    return {
        "n_optim_steps": float(len(run.points)),
        # Step 1 is logged with learning_rate == 0: the gradient at the weights
        # this step was handed, before any update from this dataset.
        "grad_norm_init": float(grad[0]),
        "grad_norm_median": float(np.median(grad)),
        "grad_norm_mean": float(grad.mean()),
        "grad_norm_final": float(grad[-1]),
        "loss_init": float(loss[0]),
        "loss_median": float(np.median(loss)),
        "loss_mean": float(loss.mean()),
        "loss_final": float(loss[-1]),
    }


#: Columns :func:`slot_frame` produces, in order, so an empty corpus still
#: yields a frame the callers below can group and filter.
_SLOT_COLUMNS = (
    "run_name",
    "condition",
    "seed",
    "step",
    "weights_id",
    "sample_id",
    "exposure",
    "trained_here",
    "trace_source",
    "n_optim_steps",
    "grad_norm_init",
    "grad_norm_median",
    "grad_norm_mean",
    "grad_norm_final",
    "loss_init",
    "loss_median",
    "loss_mean",
    "loss_final",
)


def slot_frame(
    scan: LogScan, extra: Mapping[str, TrainingRun] | None = None
) -> pd.DataFrame:
    """One row per ``(trajectory, step)``, with its trace summary where known.

    ``extra`` -- typically :func:`store_traces` -- is consulted only for
    ``weights_id``s the logs did not carry a trace for, so a log and a store
    disagreeing never silently swaps which one the numbers came from.
    """
    rows = []
    for slot in scan.slots:
        run = scan.runs.get(slot.weights_id) or (extra or {}).get(slot.weights_id)
        rows.append(
            {
                "run_name": slot.run_name,
                "condition": slot.condition,
                "seed": slot.seed,
                "step": slot.step,
                "weights_id": slot.weights_id,
                "sample_id": slot.sample_id,
                "exposure": slot.exposure,
                "trained_here": slot.trained_here,
                "trace_source": run.source if run else "",
                **(summarize(run) if run else {}),
            }
        )
    return pd.DataFrame(rows, columns=list(_SLOT_COLUMNS))


#: Columns :func:`point_frame` produces, in order. Same reason as
#: :data:`_SLOT_COLUMNS`: a corpus with no traces still has to yield a frame
#: with a header rather than a shapeless empty one.
_POINT_COLUMNS = (
    "run_name",
    "condition",
    "seed",
    "step",
    "weights_id",
    "sample_id",
    "exposure",
    "trace_source",
    "optim_step",
    "loss",
    "grad_norm",
    "learning_rate",
    "epoch",
)


def point_frame(
    scan: LogScan, extra: Mapping[str, TrainingRun] | None = None
) -> pd.DataFrame:
    """Every logged optimizer step, long form, labelled by the slots using it.

    A ``weights_id`` shared by several trajectories is emitted once per slot,
    so a groupby over conditions sees each trajectory's own curve; drop
    duplicate ``weights_id``s to get one row per *training event* instead.
    """
    rows = []
    for slot in scan.slots:
        run = scan.runs.get(slot.weights_id) or (extra or {}).get(slot.weights_id)
        if run is None:
            continue
        for point in run.points:
            rows.append(
                {
                    "run_name": slot.run_name,
                    "condition": slot.condition,
                    "seed": slot.seed,
                    "step": slot.step,
                    "weights_id": slot.weights_id,
                    "sample_id": slot.sample_id,
                    "exposure": slot.exposure,
                    "trace_source": run.source,
                    "optim_step": point.step,
                    "loss": point.loss,
                    "grad_norm": point.grad_norm,
                    "learning_rate": point.learning_rate,
                    "epoch": point.epoch,
                }
            )
    return pd.DataFrame(rows, columns=list(_POINT_COLUMNS))


def repeat_table(slots: pd.DataFrame, metric: str = "grad_norm_init") -> pd.DataFrame:
    """First versus repeat exposure to the same examples, at the same depth.

    Grouped by ``(sample_id, step)``, so both sides trained on byte-identical
    examples after the same number of prior fine-tunes and the only difference
    is whether one of those fine-tunes was on these examples. Groups where one
    side has no trace are dropped -- reporting a ratio against a missing arm
    would read as a measurement rather than as absence.

    Deduplicated by ``weights_id`` within each side: several trajectories share
    a training event whenever their chains agree, and counting it once per
    trajectory would present one measurement as many.
    """
    columns = [
        "sample_id",
        "step",
        "n_first",
        "n_repeat",
        f"first_{metric}",
        f"repeat_{metric}",
        "ratio",
        "first_conditions",
        "repeat_conditions",
    ]
    usable = slots.dropna(subset=[metric])
    rows = []
    for (sample_id, step), group in usable.groupby(["sample_id", "step"], sort=True):
        unique = group.drop_duplicates(subset="weights_id")
        first = unique[unique["exposure"] == 1]
        repeat = unique[unique["exposure"] > 1]
        if first.empty or repeat.empty:
            continue
        first_value = float(first[metric].median())
        repeat_value = float(repeat[metric].median())
        rows.append(
            {
                "sample_id": sample_id,
                "step": step,
                "n_first": len(first),
                "n_repeat": len(repeat),
                f"first_{metric}": first_value,
                f"repeat_{metric}": repeat_value,
                "ratio": repeat_value / first_value if first_value else float("nan"),
                "first_conditions": ",".join(sorted(set(first["condition"]))),
                "repeat_conditions": ",".join(sorted(set(repeat["condition"]))),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["step", "sample_id"], ignore_index=True
    )


def exposure_detail(
    slots: pd.DataFrame, metric: str = "grad_norm_init"
) -> pd.DataFrame:
    """The individual training events behind every :func:`repeat_table` row.

    A ratio of medians hides whether the two arms overlap, and with four to
    eight events a side that is the whole question -- so this is what says
    whether the effect separates the conditions or merely shifts them.
    """
    columns = [
        "sample_id",
        "step",
        "exposure",
        "condition",
        "seed",
        "run_name",
        "weights_id",
        metric,
    ]
    usable = slots.dropna(subset=[metric]).drop_duplicates(
        subset=["sample_id", "step", "weights_id"]
    )[columns]
    if usable.empty:
        return usable
    paired = usable.groupby(["sample_id", "step"])["exposure"].transform(
        lambda column: column.min() == 1 and column.max() > 1
    )
    return usable[paired].sort_values(
        ["step", "sample_id", "exposure", metric], ignore_index=True
    )


def coverage(slots: pd.DataFrame) -> str:
    """One line on how much of the corpus actually carries a gradient trace."""
    if slots.empty:
        return "no training slots found"
    events = slots.drop_duplicates(subset="weights_id")
    with_trace = int((events["trace_source"] != "").sum())
    return (
        f"{len(slots)} slots over {slots['run_name'].nunique()} trajectories; "
        f"{len(events)} distinct training events, {with_trace} with a gradient "
        f"trace ({with_trace / len(events):.0%})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="console logs to parse")
    parser.add_argument(
        "--store",
        type=Path,
        default=None,
        help="store root whose adapters' trainer_state.json fills log gaps",
    )
    parser.add_argument(
        "--metric",
        default="grad_norm_init",
        help="per-run statistic the repeat comparison comes out in",
    )
    parser.add_argument(
        "--csv", type=Path, default=None, help="write the per-slot summary here"
    )
    parser.add_argument(
        "--points-csv",
        type=Path,
        default=None,
        help="write every optimizer step here (large)",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="also print the individual training events behind each group",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    scan = scan_logs(args.logs)
    extra = store_traces(args.store) if args.store else {}
    slots = slot_frame(scan, extra)
    print(coverage(slots))

    if args.csv:
        slots.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")
    if args.points_csv:
        points = point_frame(scan, extra)
        points.to_csv(args.points_csv, index=False)
        print(f"wrote {args.points_csv} ({len(points)} rows)")

    table = repeat_table(slots, args.metric)
    print(f"\n--- first vs repeat exposure, by {args.metric} ---")
    if table.empty:
        print(
            "no (sample_id, step) group has both a first and a repeat exposure "
            "with a trace; pass more logs, or --store to fill gaps"
        )
        return
    print(table.to_string(index=False))
    print(f"\nmedian ratio repeat/first: {table['ratio'].median():.3f}")
    smaller = int((table["ratio"] < 1).sum())
    print(f"groups where repeat is smaller: {smaller}/{len(table)}")

    if args.detail:
        print("\n--- individual training events ---")
        print(exposure_detail(slots, args.metric).to_string(index=False))


if __name__ == "__main__":
    main()
