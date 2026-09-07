r"""exp3's recovered training curves, reduced to what its figure and table plot.

:mod:`method.gradients` reads the loss and gradient norm of every exp3
fine-tuning run back out of the console logs the experiment was launched under
and writes them to ``data/results/exp3_grad_points.csv`` -- one row per
(trajectory slot, optimiser step). This module turns that file into the two
things the write-up needs: each arm's mean curve over its *final* fine-tuning
step, and the entry statistics behind it.

**Why only the final step.** Every exp3 arm ends on the same trait-eliciting
dataset $X$, and the question is what the history before it did to that last
run. Earlier steps trained on different data in different arms, so their curves
are not a comparison of anything.

**Why three arms and not five.** :data:`CURVE_CONDITIONS` keeps the arms whose
final step sits at :data:`FINAL_STEP` -- the same depth in the chain, on
byte-identical examples, under the same one-epoch schedule. The baseline and
``normal1`` arms reach $X$ after zero and one prior fine-tunes, so a lower entry
loss in the Same arm could be read off against them as depth rather than as
repetition. Cutting them is what leaves repetition as the only difference.

**Why runs are deduplicated on ``weights_id``.** Adapters are content
addressed, so one training event serves every trajectory whose chain reaches
it: the same run appears once per measured trait, and both copies carry the
same curve. Counting them twice would present one measurement as two and halve
the standard error of every band drawn from it.

**Why the curves are smoothed.** A per-step loss is one batch's loss, and
adjacent batches differ by more than the arms do by the end of the run. A
rolling median over :data:`SMOOTHING_WINDOW` steps is applied to each run
*before* the runs are averaged, so the band is run-to-run spread rather than
batch noise. It is a median rather than a mean so that one hard batch moves the
line by nothing, and it costs no resolution where it would matter: at the start
of the run, where the arms are furthest apart, smoothing moves the gap between
them by under 10\%.

Reads one CSV and nothing else, so the whole analysis runs on a laptop holding
no adapters and no logs.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from method.utils import REPO_ROOT

logger = logging.getLogger(__name__)

#: Where :mod:`method.gradients` writes the per-optimiser-step frame. Under
#: ``data/results/``, which is not version controlled: the file is 63 MB and
#: regenerating it needs the console logs, so a checkout without it plots every
#: other exp3 figure and skips this one (see
#: :func:`method.visualization.make_plots.build_exp3`).
DEFAULT_POINTS = REPO_ROOT / "data" / "results" / "exp3_grad_points.csv"

#: Chain depth of the final fine-tuning step in the arms this module compares.
FINAL_STEP = 3

#: The arms whose final step is at :data:`FINAL_STEP`, in the order the legend
#: reads them: the two first exposures to $X$ first, then the repeat.
CURVE_CONDITIONS = ("normal2", "diff", "same")

#: The repeat arm, held apart from the other two because every ratio this
#: module reports is *it* over *them*.
REPEAT_CONDITION = "same"

#: Optimiser steps in the rolling median. Twenty-one is about 5% of the
#: shortest run (312 steps) and 3% of the longest (798), so the same window
#: smooths comparable fractions of each.
SMOOTHING_WINDOW = 21

#: The two quantities every function here carries, in the order the figure
#: stacks them: what the data cost the model, then how hard it pulled.
QUANTITIES = ("loss", "grad_norm")


def load_points(path: Path = DEFAULT_POINTS) -> pd.DataFrame:
    """Read the per-optimiser-step frame, or an empty one where it is absent.

    Absent is a normal state, not an error: the file is gitignored, so a fresh
    checkout has every trajectory and none of the training curves. Callers
    check ``empty`` and skip, the same way a figure builder skips a family with
    no runs on disk.
    """
    if not path.exists():
        logger.warning("no training curves at %s; the gradient figure is skipped", path)
        return pd.DataFrame(
            columns=[
                "run_name",
                "condition",
                "step",
                "weights_id",
                "optim_step",
                *QUANTITIES,
            ]
        )
    return pd.read_csv(path)


def final_step_curves(
    points: pd.DataFrame, datasets: Mapping[str, str]
) -> pd.DataFrame:
    """One row per (dataset, arm, training run, optimiser step) of the last step.

    ``datasets`` maps a trajectory directory name to the ``dataset/version``
    identifier of the dataset that trajectory ends on -- the join the CSV
    cannot make on its own, since it records the content hash of the training
    examples and not what those examples are. Slots whose trajectory is not in
    the mapping are dropped rather than labelled from their directory name: a
    name is a convention and the config is the fact.
    """
    if points.empty:
        return points.assign(dataset=pd.Series(dtype=str))
    final = points[
        (points["step"] == FINAL_STEP) & points["condition"].isin(CURVE_CONDITIONS)
    ].copy()
    final["dataset"] = final["run_name"].map(datasets)
    unmatched = final["dataset"].isna()
    if unmatched.any():
        logger.warning(
            "%d curve rows have no config for their trajectory and are dropped",
            int(unmatched.sum()),
        )
    return (
        final[~unmatched]
        # One row per training event, not per trajectory that reuses it.
        .drop_duplicates(subset=["weights_id", "optim_step"])
        .loc[:, ["dataset", "condition", "weights_id", "optim_step", *QUANTITIES]]
        .sort_values(["dataset", "condition", "weights_id", "optim_step"])
        .reset_index(drop=True)
    )


def curve_bands(
    curves: pd.DataFrame, *, window: int = SMOOTHING_WINDOW
) -> pd.DataFrame:
    """Mean and standard deviation across runs at each optimiser step.

    Each run is smoothed on its own first (see the module docstring), so the
    ``_sd`` columns describe how much training runs of one arm differ from one
    another and not how much one run bounces between batches.
    """
    if curves.empty:
        return pd.DataFrame(
            columns=["dataset", "condition", "optim_step", "runs"]
            + [f"{q}_{s}" for q in QUANTITIES for s in ("mean", "sd")]
        )
    smoothed = curves.copy()
    grouped = smoothed.groupby(["dataset", "condition", "weights_id"])[list(QUANTITIES)]
    smoothed[list(QUANTITIES)] = grouped.transform(
        lambda column: column.rolling(window, center=True, min_periods=1).median()
    )
    bands = smoothed.groupby(["dataset", "condition", "optim_step"]).agg(
        runs=("weights_id", "nunique"),
        **{
            f"{quantity}_{statistic}": (quantity, name)
            for quantity in QUANTITIES
            for statistic, name in (("mean", "mean"), ("sd", "std"))
        },
    )
    return bands.reset_index()


def entry_summary(curves: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, arm): where its final run started, and its middle.

    ``*_init`` is the value at optimiser step 1, logged while the learning rate
    is still 0 and nothing from this dataset has been applied to the weights.
    It measures the state the run was handed, with no training transient in it.
    ``*_median`` is the same quantity over the whole run, and says whether the
    difference lasted past the first batch.

    Both are medians across runs, matching the ratios in
    :func:`entry_ratios`; a mean would let one slow run set the level of a cell
    holding ten.
    """
    if curves.empty:
        return pd.DataFrame(
            columns=["dataset", "condition", "runs"]
            + [f"{q}_{s}" for q in QUANTITIES for s in ("init", "median")]
        )
    per_run = curves.groupby(["dataset", "condition", "weights_id"]).agg(
        **{
            f"{quantity}_{statistic}": (quantity, name)
            for quantity in QUANTITIES
            # `first` is the step-1 value: `final_step_curves` sorts by step.
            for statistic, name in (("init", "first"), ("median", "median"))
        }
    )
    return (
        per_run.groupby(["dataset", "condition"])
        .agg(["size", "median"])
        .pipe(_flatten_run_stats)
        .reset_index()
    )


def _flatten_run_stats(table: pd.DataFrame) -> pd.DataFrame:
    """``(loss_init, median)`` columns -> ``loss_init``, plus one ``runs``."""
    runs = table.iloc[:, 0].rename("runs").astype(int)
    medians = table.xs("median", axis=1, level=1)
    return pd.concat([runs, medians], axis=1)


def entry_ratios(summary: pd.DataFrame) -> pd.DataFrame:
    """The repeat arm over the two first-exposure arms, per dataset.

    Pooled by taking the median of the first-exposure arms' medians rather than
    of their runs: the two arms hold the same number of runs, so the pooled
    figure is theirs equally and is not tilted by one of them missing a trace.

    Below 1 is the prediction: a run that has met these examples before starts
    nearer to fitting them and pulls less hard on the weights.
    """
    if summary.empty:
        return summary
    keys = ("dataset", "condition", "runs")
    columns = [column for column in summary.columns if column not in keys]
    rows = []
    for dataset, group in summary.groupby("dataset"):
        repeat = group[group["condition"] == REPEAT_CONDITION]
        first = group[group["condition"] != REPEAT_CONDITION]
        if repeat.empty or first.empty:
            continue
        rows.append(
            {
                "dataset": dataset,
                **{
                    column: float(repeat[column].iloc[0])
                    / float(first[column].median())
                    for column in columns
                },
            }
        )
    return pd.DataFrame(rows, columns=["dataset", *columns])
