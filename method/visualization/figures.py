r"""Figure-generating functions for the sequential fine-tuning experiments
(proposal Section "Experiments and Plots").

Every function takes plain arrays / DataFrames and returns a
:class:`matplotlib.figure.Figure` -- never a :class:`~.schema.Trajectory`
directly -- so the same code plots real measurements (once experiments have
run) and :mod:`method.visualization.synthetic` fixtures identically. Save the
result with :func:`method.visualization.style.save_figure`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from method.visualization import style
from method.visualization.labels import display_dataset_name
from method.visualization.metrics import (
    LinearFit,
    linear_fit,
    mean_std,
    percent_of_baseline,
    stack_and_trim,
)

import matplotlib.pyplot as plt  # noqa: E402  (backend fixed by style import above)


def _fit_line_x(x: np.ndarray) -> np.ndarray:
    """A smooth x-range spanning ``x`` (with a small margin), for a fit line."""
    if x.size == 0:
        return x
    span = x.max() - x.min()
    pad = 0.05 * span if span > 0 else 1.0
    return np.linspace(x.min() - pad, x.max() + pad, 50)


def _scatter_with_fit(
    ax: Axes, x: ArrayLike, y: ArrayLike, *, color: str, label: str
) -> LinearFit:
    """One scatter series plus its least-squares line, labelled with $R^2$."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    fit = linear_fit(x_arr, y_arr)
    ax.scatter(
        x_arr,
        y_arr,
        s=28,
        color=color,
        alpha=0.75,
        edgecolor=style.SURFACE,
        linewidth=0.6,
        zorder=3,
    )
    if x_arr.size >= 2:
        line_x = _fit_line_x(x_arr)
        ax.plot(
            line_x,
            fit.predict(line_x),
            color=color,
            linewidth=1.8,
            zorder=2,
            label=rf"{label} ($R^2$={fit.r2:.2f})",
        )
    return fit


def scatter_projection_correlation(
    delta_p_0: ArrayLike,
    delta_p_t: ArrayLike,
    delta_behavior: ArrayLike,
    *,
    xlabel: str = r"Projection difference $\Delta P$",
    ylabel: str = r"Behaviour change $\Delta b_{t+1}$",
    title: str | None = None,
) -> Figure:
    r"""RQ1: does $\Delta P_0$ or $\Delta P_t$ better predict $\Delta b_{t+1}$?

    Two series share one axis: $\Delta P_0$ (blue), frozen at the base model,
    against $\Delta P_t$ (orange), recomputed at the checkpoint about to be
    trained -- both plotted against the behaviour change the step actually
    caused. The hypothesis under test is that the blue fit's $R^2$ decays
    with sequence length while the orange one stays high.
    """
    style.apply_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    _scatter_with_fit(
        ax, delta_p_0, delta_behavior, color=style.BLUE, label=r"$\Delta P_0$"
    )
    _scatter_with_fit(
        ax, delta_p_t, delta_behavior, color=style.ORANGE, label=r"$\Delta P_t$"
    )
    ax.axhline(0, color=style.BASELINE, linewidth=0.8, zorder=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def scatter_metric_grid(
    metrics: Mapping[str, tuple[ArrayLike, ArrayLike]],
    *,
    ylabel: str = r"Behaviour change $\Delta b_{t+1}$",
    color: str = style.BLUE,
    ncols: int = 2,
    title: str | None = None,
) -> Figure:
    r"""Small-multiples version of the same diagnostic for $p$, $q$, $\rho$, $r$.

    ``metrics`` maps a display label (e.g. ``r"$\rho_t$"``) to its ``(x, y)``
    arrays -- typically the four $z_t$ components from
    :func:`method.visualization.schema.metric_pairs`, each against
    $\Delta b_{t+1}$.
    """
    style.apply_style()
    n = len(metrics)
    ncols = max(1, min(ncols, n))
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows), squeeze=False
    )
    axes_flat = axes.flatten()
    for ax, (label, (x, y)) in zip(axes_flat, metrics.items()):
        _scatter_with_fit(ax, x, y, color=color, label=label)
        ax.set_xlabel(label)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def line_with_band(
    series: Mapping[str, np.ndarray],
    *,
    ylabel: str,
    xlabel: str = "Step",
    reference: float | None = None,
    reference_label: str | None = None,
    title: str | None = None,
) -> Figure:
    """Mean line + shaded $\\pm 1$ std band per named series, over a shared index.

    ``series`` maps a label to a ``[n_seeds, n_steps]`` matrix (already
    aligned; use :func:`~method.visualization.metrics.stack_and_trim` first if
    the per-seed runs are ragged). This is the generic primitive behind
    :func:`drift_line`; call it directly when the y-values are already on a
    comparable scale (e.g. a percentage computed against an external
    reference, rather than each series' own step-0 value).
    """
    style.apply_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for i, (label, matrix) in enumerate(series.items()):
        mean, std = mean_std(np.asarray(matrix, dtype=float), axis=0)
        x = np.arange(mean.size)
        color = style.categorical_color(i)
        ax.plot(
            x,
            mean,
            color=color,
            marker="o",
            markersize=5,
            markeredgecolor=style.SURFACE,
            markeredgewidth=0.8,
            linewidth=1.8,
            label=label,
            zorder=3,
        )
        ax.fill_between(
            x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0, zorder=2
        )
    if reference is not None:
        ax.axhline(
            reference,
            color=style.MUTED,
            linestyle="--",
            linewidth=1.0,
            zorder=1,
            label=reference_label,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if len(series) > 1 or reference_label:
        ax.legend(loc="best")
    fig.tight_layout()
    return fig


def drift_line(
    series: Mapping[str, Sequence[Sequence[float]]],
    *,
    ylabel: str,
    xlabel: str = "Step $t$",
    title: str | None = None,
) -> Figure:
    r"""How far a quantity has drifted from its own step-0 value, over time.

    ``series`` maps a label (e.g. a dataset name or ``r"$\rho_t$"``) to a list
    of per-seed runs (ragged lengths are trimmed to the shortest). Each is
    expressed as a percentage of its own first entry (100 = unchanged) and
    plotted as a mean line with a $\pm 1$ std band, against a dashed
    reference line at 100. This is the "does $\Delta P_t$ / $\rho_t$ / $r_t$ /
    $p_t$ / $q_t$ deviate from its step-0 value" plot.
    """
    pct_series = {
        label: percent_of_baseline(stack_and_trim(runs), axis=1)
        for label, runs in series.items()
    }
    return line_with_band(
        pct_series,
        ylabel=ylabel,
        xlabel=xlabel,
        reference=100.0,
        reference_label="Step-0 value",
        title=title,
    )


def hysteresis_bar(
    df: pd.DataFrame,
    *,
    dataset_col: str = "dataset",
    condition_col: str = "condition",
    value_col: str = "delta_behavior",
    conditions: Sequence[str] = ("fresh", "after_realignment"),
    condition_labels: Sequence[str] | None = None,
    dataset_labels: Mapping[str, str] | None = None,
    ylabel: str = r"Behaviour change $\Delta b$",
    title: str | None = None,
) -> Figure:
    """RQ2 hysteresis bar chart: is a realigned model easier to re-misalign?

    Grouped bars: one group per dataset, one bar per ``condition`` within a
    group (2 colours -> a legend, since colour here encodes a second
    dimension on top of the x-axis identity). Error bars show the std across
    seeds.

    ``dataset_col`` is expected to hold internal ``dataset/version``
    identifiers (e.g. ``"mistake_gsm8k/misaligned_2"``); tick labels are
    rendered with :func:`~method.visualization.labels.display_dataset_name`
    (e.g. ``"GSM8K (Mistake II)"``) unless overridden per-dataset via
    ``dataset_labels``.
    """
    style.apply_style()
    condition_labels = list(condition_labels) if condition_labels else list(conditions)
    dataset_labels = dataset_labels or {}
    datasets = list(dict.fromkeys(df[dataset_col]))
    stats = df.groupby([dataset_col, condition_col])[value_col].agg(["mean", "std"])

    n_conditions = len(conditions)
    width = 0.8 / n_conditions
    x = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(max(4.0, 1.4 * len(datasets) + 1.5), 4.2))
    for i, (condition, label) in enumerate(zip(conditions, condition_labels)):
        means = [
            (
                stats.loc[(ds, condition), "mean"]
                if (ds, condition) in stats.index
                else np.nan
            )
            for ds in datasets
        ]
        stds = [
            stats.loc[(ds, condition), "std"] if (ds, condition) in stats.index else 0.0
            for ds in datasets
        ]
        offset = (i - (n_conditions - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width=width * 0.9,
            color=style.categorical_color(i),
            edgecolor=style.SURFACE,
            linewidth=1.2,
            label=label,
            zorder=3,
        )
        ax.errorbar(
            x + offset,
            means,
            yerr=stds,
            fmt="none",
            ecolor=style.MUTED,
            elinewidth=1.0,
            capsize=3,
            zorder=4,
        )
    ax.axhline(0, color=style.BASELINE, linewidth=0.8, zorder=1)
    ax.set_xticks(x)
    tick_labels = [dataset_labels.get(ds, display_dataset_name(ds)) for ds in datasets]
    ax.set_xticklabels(tick_labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def diversity_bar(
    df: pd.DataFrame,
    *,
    condition_col: str = "condition",
    value_col: str = "delta_behavior",
    order: Sequence[str] | None = None,
    labels: Mapping[str, str] | None = None,
    ylabel: str = r"Behaviour change $\Delta b$",
    title: str | None = None,
    color: str = style.BLUE,
) -> Figure:
    """RQ2 diversity bar chart: does dataset diversity hinder re-alignment?

    A single colour for every bar -- the conditions are unordered categories
    identified by their x-tick label, not a second grouping dimension, so a
    rainbow or a value ramp would only burn the colour channel on information
    the chart already shows (see the data-viz "value-ramp on nominal
    categories" anti-pattern).
    """
    style.apply_style()
    order = list(order) if order is not None else list(dict.fromkeys(df[condition_col]))
    stats = df.groupby(condition_col)[value_col].agg(["mean", "std"]).reindex(order)
    labels = labels or {}
    tick_labels = [labels.get(c, c) for c in order]

    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(max(4.0, 1.3 * len(order) + 1.5), 4.2))
    ax.bar(
        x,
        stats["mean"],
        width=0.6,
        color=color,
        edgecolor=style.SURFACE,
        linewidth=1.2,
        zorder=3,
    )
    ax.errorbar(
        x,
        stats["mean"],
        yerr=stats["std"],
        fmt="none",
        ecolor=style.MUTED,
        elinewidth=1.0,
        capsize=3,
        zorder=4,
    )
    ax.axhline(0, color=style.BASELINE, linewidth=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig
