r"""Reduce experiment-3 logs to final-step learning curves."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from method.utils import REPO_ROOT

logger = logging.getLogger(__name__)

#: Default per-optimiser-step input.
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
    """Read optimiser-step data, returning an empty frame if absent."""
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
