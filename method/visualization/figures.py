r"""Figure-generating functions for the sequential fine-tuning experiments
(proposal Section "Experiments and Plots").

Every function takes plain arrays / DataFrames and returns a
:class:`matplotlib.figure.Figure` -- never a :class:`~.schema.Trajectory`
directly -- so the same code plots real measurements (once experiments have
run) and :mod:`method.visualization.synthetic` fixtures identically. Save the
result with :func:`method.visualization.style.save_figure`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox
from numpy.typing import ArrayLike

from method.visualization import style
from method.visualization.labels import (
    DATASET_TITLES,
    HYSTERESIS_CONDITION_SEQUENCES,
    HYSTERESIS_CONDITIONS,
    display_dataset_name,
)
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


def _scatter_marks(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    datasets: Sequence[str],
    *,
    size: float,
    edge: str | None = None,
    zorder: int = 3,
) -> None:
    """Scatter one mark per dataset: shape for the family, fill for the version.

    Grouped by shape because a scatter call takes a single marker, so eight
    families are eight calls over disjoint index sets rather than one call.

    ``edge`` overrides the version outline for every point, which is what a
    figure already using colour for something else passes: the decay grid
    rings its marks in the projection series' hue, so a mark carries its
    dataset in shape and fill while colour still separates $\\Delta P_0$ from
    $\\Delta P_t$.
    """
    marks = [style.dataset_mark(d) for d in datasets]
    for marker in dict.fromkeys(mark.marker for mark in marks):
        at = [i for i, mark in enumerate(marks) if mark.marker == marker]
        ax.scatter(
            x[at],
            y[at],
            marker=marker,
            s=size,
            facecolor=[marks[i].face for i in at],
            edgecolor=edge if edge else [marks[i].edge for i in at],
            # Heavier when the outline is doing double duty as a series
            # identity, which it has to carry at a glance across a panel grid.
            linewidth=1.2 if edge else 0.9,
            zorder=zorder,
        )


def _scatter_with_fit(
    ax: Axes,
    x: ArrayLike,
    y: ArrayLike,
    *,
    color: str,
    label: str,
    yerr: ArrayLike | None = None,
    size: float = 28,
    datasets: Sequence[str] | None = None,
    mark_edge: str | None = None,
) -> LinearFit:
    """One scatter series plus its least-squares line, labelled with $R^2$.

    ``yerr`` draws the per-point error bar section 6a makes available for every
    $\\Delta b$: it is what separates a point that sits off the line because the
    model really moved differently from one that sits off it because its eval
    was imprecise. Bars that are all zero (or all missing) are skipped rather
    than drawn as flat caps, since a run measured with one generation per
    question has no within-question spread to report.

    ``datasets`` names the ``dataset/version`` behind each point, which turns
    the marks into the per-dataset shapes and fills of :func:`_scatter_marks`.
    ``color`` then applies to the fit line and the error bars only -- and to
    the mark outlines when ``mark_edge`` asks for it.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    fit = linear_fit(x_arr, y_arr)
    if yerr is not None:
        err = np.asarray(yerr, dtype=float)
        if np.any(np.isfinite(err) & (err > 0)):
            ax.errorbar(
                x_arr,
                y_arr,
                yerr=np.nan_to_num(err),
                fmt="none",
                ecolor=color,
                elinewidth=0.9,
                alpha=0.55,
                capsize=0,
                zorder=2,
            )
    if datasets is not None:
        _scatter_marks(ax, x_arr, y_arr, datasets, size=size, edge=mark_edge)
    else:
        ax.scatter(
            x_arr,
            y_arr,
            s=size,
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
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def scatter_metric_grid(
    metrics: Mapping[str, tuple[ArrayLike, ArrayLike]],
    *,
    ylabel: str = r"Behaviour change $\Delta b_{t+1}$",
    color: str = style.BLUE,
    ncols: int = 2,
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
    fig.tight_layout()
    return fig


def line_with_band(
    series: Mapping[str, np.ndarray],
    *,
    ylabel: str,
    xlabel: str = "Step",
    reference: float | None = None,
    reference_label: str | None = None,
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
    if len(series) > 1 or reference_label:
        ax.legend(loc="best")
    fig.tight_layout()
    return fig


def drift_line(
    series: Mapping[str, Sequence[Sequence[float]]],
    *,
    ylabel: str,
    xlabel: str = "Step $t$",
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
    )


def hysteresis_bar(
    df: pd.DataFrame,
    *,
    dataset_col: str = "dataset",
    condition_col: str = "condition",
    value_col: str = "behavior",
    conditions: Sequence[str] = HYSTERESIS_CONDITIONS,
    condition_labels: Sequence[str] | None = None,
    dataset_labels: Mapping[str, str] | None = None,
    start_col: str | None = None,
    reference: float | None = None,
    reference_label: str | None = None,
    ylabel: str = r"Trait score $b_T$",
) -> Figure:
    r"""RQ2 hysteresis bar chart: is a realigned model easier to re-misalign?

    One panel per dataset, one bar per ``condition`` within it. Error bars show
    the std across seeds.

    Bars are *levels*, not deltas. ``value_col`` defaults to the trait score
    each arm ends at, and ``reference`` draws a dashed line at $M_0$'s own
    score, so a bar's height above that line is $b_T - b_0$ -- a quantity every
    arm shares an origin for. Plotting the last step's $\Delta b$ instead would
    measure each arm from its own floor: an arm re-aligned down to 10 that
    climbs back to 50 would score below a baseline that went straight to 50,
    despite ending in the same place.

    ``start_col`` adds a tick across each bar at the score the final step
    started from ($b_{T-1}$). It is what separates the two readings a level
    alone leaves ambiguous -- an arm that ends low because it moved little
    (plasticity loss) from one that ends low because it started low.

    ``dataset_col`` is expected to hold internal ``dataset/version``
    identifiers (e.g. ``"mistake_gsm8k/misaligned_2"``); panels are headed with
    :func:`~method.visualization.labels.display_dataset_name` (e.g. ``"GSM8K
    (Mistake II)"``) unless overridden per-dataset via ``dataset_labels``.

    Each arm is named on its own tick by its training schedule -- ``$X\,N\,X$``
    and so on, see
    :data:`~method.visualization.labels.HYSTERESIS_CONDITION_SEQUENCES` -- so
    the legend is left to carry only the two things that have no tick of their
    own: the base-model reference and the start-of-final-step marks. Colour
    still separates the arms, but nothing is encoded in colour alone.
    """
    style.apply_style()
    if condition_labels is None:
        condition_labels = [
            HYSTERESIS_CONDITION_SEQUENCES.get(c, c) for c in conditions
        ]
    condition_labels = list(condition_labels)
    dataset_labels = dataset_labels or {}
    datasets = list(dict.fromkeys(df[dataset_col]))
    stats = df.groupby([dataset_col, condition_col])[value_col].agg(["mean", "std"])
    starts = (
        df.groupby([dataset_col, condition_col])[start_col].mean()
        if start_col
        else None
    )

    x = np.arange(len(conditions))
    width = 0.72

    # One panel per dataset rather than one group of bars per dataset. Grouping
    # left every arm's identity in its colour, because a group has one tick
    # between it and its neighbour and five arm names will not fit there;
    # faceting gives each arm a tick of its own and costs only width.
    # 0.52in per bar keeps the schedule labels from touching, and the cap stops
    # a large dataset pool producing an unprintable figure.
    width_in = min(11.0, max(4.0, (0.52 * len(conditions) + 0.5) * len(datasets) + 0.8))
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(width_in, 4.4), sharey=True, squeeze=False
    )
    for ax, dataset in zip(axes[0], datasets):
        means = [
            (
                stats.loc[(dataset, c), "mean"]
                if (dataset, c) in stats.index
                else np.nan
            )
            for c in conditions
        ]
        stds = [
            stats.loc[(dataset, c), "std"] if (dataset, c) in stats.index else 0.0
            for c in conditions
        ]
        ax.bar(
            x,
            means,
            width=width,
            color=[style.categorical_color(i) for i in range(len(conditions))],
            edgecolor=style.SURFACE,
            linewidth=1.2,
            zorder=3,
        )
        ax.errorbar(
            x, means, yerr=stds, fmt="none", ecolor=style.MUTED, elinewidth=1.0,
            capsize=3, zorder=4,
        )
        if starts is not None:
            # Drawn as a tick across the bar rather than a second bar: it marks
            # where this bar's growth began, so it belongs *on* the bar.
            ax.errorbar(
                x,
                [starts.get((dataset, c), np.nan) for c in conditions],
                xerr=width / 2,
                fmt="none",
                ecolor=style.INK,
                elinewidth=1.2,
                zorder=5,
            )
        if reference is not None:
            ax.axhline(
                reference,
                color=style.SECONDARY_INK,
                linestyle="--",
                linewidth=1.0,
                zorder=2,
            )
        ax.axhline(0, color=style.BASELINE, linewidth=0.8, zorder=1)
        ax.set_xticks(x)
        # Full ink, unlike the muted numeric ticks elsewhere: these names are
        # what identifies an arm now that the legend no longer does.
        ax.set_xticklabels(condition_labels, fontsize=10, color=style.INK)
        ax.set_xlim(-0.5 - (1 - width) / 2, len(conditions) - 0.5 + (1 - width) / 2)
        ax.set_title(dataset_labels.get(dataset, display_dataset_name(dataset)))

    handles: list[Artist] = []
    texts: list[str] = []
    if reference is not None:
        handles.append(
            plt.Line2D([], [], color=style.SECONDARY_INK, linestyle="--", linewidth=1.0)
        )
        texts.append(reference_label or "Base model")
    if starts is not None:
        handles.append(plt.Line2D([], [], color=style.INK, linewidth=1.2))
        texts.append("Before the final step")
    if handles:
        fig.legend(handles, texts, loc="lower center", ncol=len(handles))
    _layout_grid(fig, axes.flat, ylabel=ylabel, legend_rows=1 if handles else 0)
    return fig


# --- the RQ1 decay experiment ---------------------------------------------


def _corner_text(
    ax: Axes,
    entries: Sequence[tuple[str, str]],
    *,
    x: float = 0.04,
    y: float = 0.96,
    fontsize: float = 7.5,
    ha: str = "left",
) -> None:
    """Stacked ``(text, colour)`` annotations in a panel corner.

    Used where a panel is too small for a legend of its own. The text always
    names both the quantity and the series it reports (``"$R^2(\\Delta P_0)$ =
    0.42"``), so colour is redundant reinforcement of an identity the words
    already carry rather than the only channel encoding it.
    """
    for i, (text, color) in enumerate(entries):
        ax.text(
            x,
            y - 0.105 * i,
            text,
            transform=ax.transAxes,
            fontsize=fontsize,
            va="top",
            ha=ha,
            color=color,
            zorder=5,
            # Annotations sit over the data, and in a dense panel a marker
            # lands behind them often enough to cost a digit. The patch is the
            # chart surface itself, so it reads as clearance, not as a box.
            bbox={"facecolor": style.SURFACE, "edgecolor": "none", "pad": 1.2},
        )


#: Vertical space, in inches, that one row of a figure legend and one shared
#: axis label take up. Used to size the bottom margin a panel grid reserves.
_LEGEND_ROW_IN = 0.24
_SHARED_LABEL_IN = 0.30

#: How far a shared axis label sits from the block of axes it names, in inches.
_SHARED_LABEL_PAD_IN = 0.05


def _layout_grid(
    fig: Figure,
    axes: Iterable[Axes],
    *,
    ylabel: str,
    legend_rows: int,
    xlabel: str | None = None,
    fontsize: float = 10,
) -> None:
    """Lay a panel grid out beneath its own legend and label the block as a whole.

    Both halves have to be done here because neither can be done blind.
    ``tight_layout`` measures the axes but knows nothing of a figure legend, so
    the bottom margin is sized from the number of rows that legend will take;
    ``fig.supxlabel`` in turn positions against the *figure*, so a fixed ``y``
    strands the label in the middle of that margin on one grid shape and drops
    it onto the legend on another. Measuring the drawn axes afterwards keeps
    each label the same short distance from the ticks it names.

    ``xlabel`` is optional: a grid whose ticks already name themselves (the
    hysteresis schedules, say) has nothing left for a shared x-label to add.
    """
    width, height = fig.get_size_inches()
    reserved = legend_rows * _LEGEND_ROW_IN + (_SHARED_LABEL_IN if xlabel else 0.0)
    fig.tight_layout(rect=(0.0, reserved / height, 1.0, 1.0))
    drawn = [box for ax in axes if (box := ax.get_tightbbox()) is not None]
    block = Bbox.union(drawn).transformed(fig.transFigure.inverted())
    if xlabel:
        fig.supxlabel(
            xlabel,
            fontsize=fontsize,
            y=block.y0 - _SHARED_LABEL_PAD_IN / height,
            va="top",
        )
    fig.supylabel(
        ylabel,
        fontsize=fontsize,
        x=block.x0 - _SHARED_LABEL_PAD_IN / width,
        ha="right",
    )


#: Version order for a legend key, matching the ramp's direction.
_VERSION_KEY = (("normal", "Normal"), ("misaligned_1", "I"), ("misaligned_2", "II"))


def dataset_legend(
    datasets: Sequence[str],
) -> tuple[list[Artist], list[str]]:
    r"""A legend key for the dataset marks, decomposed into its two channels.

    Naming all 24 datasets would need 24 entries; naming the *encoding* needs
    at most eleven, and covers datasets the figure happens not to contain. So
    this returns one entry per family shape (drawn hollow, so the shape is what
    reads) and one swatch per version fill.

    Only the families and versions actually present are listed, in the fixed
    order of :data:`~method.visualization.style.DATASET_MARKERS` and of the
    version ramp -- never in the order the frame happened to arrive in, which
    would reshuffle the key whenever a run was added.
    """
    families, versions = [], []
    for dataset_id in datasets:
        family, _, version = dataset_id.partition("/")
        families.append(family)
        versions.append(version)

    handles: list[Artist] = []
    texts: list[str] = []
    for family in style.DATASET_MARKERS:
        if family not in families:
            continue
        handles.append(
            plt.Line2D(
                [],
                [],
                marker=style.DATASET_MARKERS[family],
                linestyle="none",
                markersize=6,
                markerfacecolor=style.SURFACE,
                markeredgecolor=style.SECONDARY_INK,
                markeredgewidth=0.9,
            )
        )
        texts.append(DATASET_TITLES.get(family, family))
    for version, label in _VERSION_KEY:
        if version not in versions:
            continue
        # A rectangle rather than a marker: the version is the *fill* channel,
        # and a marker-shaped swatch would read as a ninth family.
        handles.append(
            Patch(
                facecolor=style.VERSION_FILL[version],
                edgecolor=style.VERSION_EDGE[version],
                linewidth=0.9,
            )
        )
        texts.append(label)
    return handles, texts


def scatter_validation(
    delta_p_0: ArrayLike,
    delta_behavior: ArrayLike,
    *,
    datasets: Sequence[str] | None = None,
    yerr: ArrayLike | None = None,
    xlabel: str = r"Projection difference $\Delta P_0$",
    ylabel: str = r"Behaviour change $\Delta b_1$",
) -> Figure:
    r"""Plot 1: the $t = 0$ validation fan over all 24 datasets.

    One point per dataset fine-tuned straight from $M_0$, reproducing Figure 8
    of the persona-vectors paper. This is the pipeline's gate, not part of the
    decay analysis: it is drawn over 24 datasets while the decay curve is drawn
    over the 8 probes, and $R^2$ estimates are sensitive enough to range
    restriction and to ``n`` that comparing the two would manufacture a decay
    out of nothing (section 5).

    One fitted series, so its statistics are annotated directly rather than
    put in a legend. ``datasets`` gives every point its own mark, and the
    legend then keys the *encoding* -- shape per family, fill per version --
    which is what makes a 24-point scatter readable without 24 legend entries.
    """
    style.apply_style()
    wide = datasets is not None
    fig, ax = plt.subplots(figsize=(7.4 if wide else 5.5, 4.4))
    fit = _scatter_with_fit(
        ax,
        delta_p_0,
        delta_behavior,
        color=style.BLUE,
        label=r"$\Delta P_0$",
        yerr=yerr,
        size=44,
        datasets=datasets,
    )
    ax.axhline(0, color=style.BASELINE, linewidth=0.8, zorder=1)
    n = int(np.asarray(delta_p_0, dtype=float).size)
    _corner_text(
        ax,
        [
            (rf"$R^2$ = {fit.r2:.2f}   slope = {fit.slope:.2f}", style.INK),
            (rf"$n$ = {n} datasets", style.SECONDARY_INK),
        ],
        fontsize=9,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if datasets is not None:
        handles, texts = dataset_legend(datasets)
        ax.legend(handles, texts, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    return fig


def decay_scatter_grid(
    df: pd.DataFrame,
    *,
    trunks: Sequence[str] | None = None,
    checkpoints: Sequence[int] | None = None,
    trunk_labels: Mapping[str, str] | None = None,
    xlabel: str = r"Projection difference $\Delta P$",
    ylabel: str = r"Behaviour change $\Delta b_{t+1}$",
) -> Figure:
    r"""Plot 2: one scatter panel per ``(trunk, checkpoint)``.

    Rows are trunks and columns checkpoints, so a row reads as one trajectory
    ageing and a column as three trajectories at the same depth. Each panel
    holds the ``K`` probe datasets twice over: against the frozen $\Delta P_0$
    in blue and against the recomputed $\Delta P_t$ in orange, with both fits'
    $R^2$ annotated. The hypothesis is visible as the blue fit flattening
    left-to-right while the orange one does not.

    Axes are shared across the whole grid. Panels drawn on their own scales
    would let a flattening slope and a shrinking $\Delta b$ range look
    identical, which is the one confusion this figure exists to prevent.

    The steps-since-re-alignment note above each panel is the phase of section
    4, marked per panel rather than per column because the three schedules put
    their re-alignments at different depths, so the phase of column ``t``
    differs by row.

    The ``t = 0`` column repeats across rows by construction: all three trunks
    branch from $M_0$, which is measured once.
    """
    style.apply_style()
    trunks = list(trunks) if trunks else sorted(df["trunk"].unique())
    checkpoints = (
        list(checkpoints) if checkpoints is not None else sorted(df["t"].unique())
    )
    trunk_labels = trunk_labels or {}
    nrows, ncols = max(1, len(trunks)), max(1, len(checkpoints))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(1.75 * ncols + 1.4, 2.0 * nrows + 1.3),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row, trunk in enumerate(trunks):
        for col, t in enumerate(checkpoints):
            ax = axes[row][col]
            panel = df[(df["trunk"] == trunk) & (df["t"] == t)]
            if panel.empty:
                _corner_text(
                    ax, [("not run", style.MUTED)], x=0.5, y=0.55, ha="center"
                )
                continue
            ax.axhline(0, color=style.BASELINE, linewidth=0.8, zorder=1)
            errors = panel["se_delta_b"] if "se_delta_b" in panel else None
            probes = list(panel["probe"]) if "probe" in panel else None
            stale = _scatter_with_fit(
                ax,
                panel["delta_p_0"],
                panel["delta_b"],
                color=style.BLUE,
                label=r"$\Delta P_0$",
                yerr=errors,
                size=34,
                datasets=probes,
                mark_edge=style.BLUE,
            )
            fresh = _scatter_with_fit(
                ax,
                panel["delta_p_t"],
                panel["delta_b"],
                color=style.ORANGE,
                label=r"$\Delta P_t$",
                yerr=errors,
                size=34,
                datasets=probes,
                mark_edge=style.ORANGE,
            )
            _corner_text(
                ax,
                [
                    (rf"$R^2(\Delta P_0)$ = {stale.r2:.2f}", style.BLUE),
                    (rf"$R^2(\Delta P_t)$ = {fresh.r2:.2f}", style.ORANGE),
                ],
            )
            # As a right-hand axes title rather than an in-panel annotation:
            # the phase belongs to the panel, not to the data, and a corner
            # that happens to hold a point would otherwise hide it.
            since = int(panel["steps_since_realignment"].iloc[0])
            ax.set_title(
                f"steps since re-alignment: {since}",
                loc="right",
                fontsize=6.5,
                color=style.MUTED,
                pad=3,
            )
        axes[row][0].set_ylabel(
            trunk_labels.get(trunks[row], f"Trunk {trunks[row]}"), fontsize=9
        )
    for col, t in enumerate(checkpoints):
        # An annotation rather than a centred title: matplotlib lays a panel's
        # left, centre and right titles on one line, and the top row's right
        # title is already the phase note. Offsetting the column header above
        # that line is what keeps both readable, on a row that has to carry
        # them both.
        axes[0][col].annotate(
            f"$t = {t}$",
            xy=(0.5, 1.0),
            xycoords="axes fraction",
            xytext=(0, 15),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    # Two keys in one legend: which projection a mark's *outline* means, and
    # which dataset its shape and fill mean.
    handles: list[Artist] = [
        plt.Line2D([], [], color=style.BLUE, linewidth=1.8),
        plt.Line2D([], [], color=style.ORANGE, linewidth=1.8),
    ]
    texts = [r"$\Delta P_0$ (frozen at $M_0$)", r"$\Delta P_t$ (recomputed at $M_t$)"]
    if "probe" in df:
        mark_handles, mark_texts = dataset_legend(sorted(set(df["probe"])))
        handles += mark_handles
        texts += mark_texts
    ncol = min(7, len(handles))
    fig.legend(
        handles,
        texts,
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, 0.0),
    )
    _layout_grid(
        fig,
        axes.flat,
        xlabel=xlabel,
        ylabel=ylabel,
        legend_rows=-(-len(handles) // ncol),
    )
    return fig


def _band(
    ax: Axes,
    x: ArrayLike,
    mid: ArrayLike,
    lo: ArrayLike,
    hi: ArrayLike,
    *,
    color: str,
    label: str,
) -> None:
    """A line with its confidence band, as used by the headline curves."""
    x_arr = np.asarray(x, dtype=float)
    ax.plot(
        x_arr,
        np.asarray(mid, dtype=float),
        color=color,
        marker="o",
        markersize=4.5,
        markeredgecolor=style.SURFACE,
        markeredgewidth=0.8,
        linewidth=1.8,
        label=label,
        zorder=3,
    )
    ax.fill_between(
        x_arr,
        np.asarray(lo, dtype=float),
        np.asarray(hi, dtype=float),
        color=color,
        alpha=0.13,
        linewidth=0,
        zorder=2,
    )


def headline_curves(
    fits: pd.DataFrame,
    *,
    series: Sequence[str] = ("p0", "pt"),
    series_labels: Mapping[str, str] | None = None,
    trunks: Sequence[str] | None = None,
    trunk_labels: Mapping[str, str] | None = None,
    trunk_colors: Mapping[str, str] | None = None,
    ceiling_column: str | None = "r2_max",
    ceiling_label: str = r"$R^2_{max}$ (noise ceiling)",
    xlabel: str = "Checkpoint $t$",
) -> Figure:
    r"""Plot 3: $R^2$ and fitted slope against ``t``, one line per trunk.

    Two rows and one column per projection series. Slope is reported beside
    $R^2$ rather than inside it because staleness has two signatures that imply
    different fixes: a fit that loses its ordering (falling $R^2$) and one that
    keeps the ordering but shrinks the magnitude (falling slope at unchanged
    $R^2$).

    The dashed ceiling is what makes the $R^2$ panels a claim rather than an
    observation. $\Delta b$ carries noise no predictor can explain, so a
    correlation fitted against it cannot exceed $R^2_{max}$ (section 6b): a
    blue curve decaying *toward* its ceiling is signal running out, while one
    dropping clearly below while $\Delta P_t$ stays near it is staleness.

    Expect a sawtooth in ``t`` and do not smooth it -- it is the phase effect
    the schedules deliberately induce, and :func:`phase_contrast` is what
    separates it from the trend.

    Two measures on two rows rather than two y-scales on one: the alignment of
    a shared axis between $R^2$ and a slope in trait points per unit
    $\Delta P$ would be arbitrary, and would invent a relationship between
    them.
    """
    style.apply_style()
    series = list(series)
    series_labels = series_labels or {}
    trunk_labels = trunk_labels or {}
    trunk_colors = trunk_colors or {}
    trunks = list(trunks) if trunks else sorted(fits["trunk"].unique())

    # Rows share a y-axis so the two projection series are read against each
    # other, which is the comparison the figure exists for; the two rows do
    # not, since $R^2$ and a slope are different quantities.
    fig, axes = plt.subplots(
        2,
        len(series),
        figsize=(4.4 * len(series), 6.2),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for col, name in enumerate(series):
        for row, quantity in enumerate(("r2", "slope")):
            ax = axes[row][col]
            for trunk in trunks:
                arm = fits[fits["trunk"] == trunk].sort_values("t")
                if arm.empty:
                    continue
                color = trunk_colors.get(trunk, style.BLUE)
                _band(
                    ax,
                    arm["t"],
                    arm[f"{quantity}_{name}"],
                    arm[f"{quantity}_{name}_lo"],
                    arm[f"{quantity}_{name}_hi"],
                    color=color,
                    label=trunk_labels.get(trunk, f"Trunk {trunk}"),
                )
                if quantity == "r2" and ceiling_column in arm:
                    ax.plot(
                        arm["t"],
                        arm[ceiling_column],
                        color=color,
                        linestyle=(0, (4, 2)),
                        linewidth=1.0,
                        alpha=0.6,
                        zorder=1,
                    )
            if quantity == "r2":
                ax.set_ylim(-0.03, 1.03)
                ax.set_ylabel(r"$R^2$ over the probe set")
            else:
                ax.axhline(0, color=style.BASELINE, linewidth=0.8, zorder=1)
                ax.set_ylabel(r"Fitted slope ($\Delta b$ per unit $\Delta P$)")
                # Only the bottom row: the columns share an x-axis, so a label
                # under the top row would name ticks that are not drawn.
                ax.set_xlabel(xlabel)
        axes[0][col].set_title(
            series_labels.get(name, name), color=style.SECONDARY_INK
        )

    handles, texts = axes[0][0].get_legend_handles_labels()
    if ceiling_column is not None:
        handles.append(
            plt.Line2D(
                [], [], color=style.MUTED, linestyle=(0, (4, 2)), linewidth=1.0
            )
        )
        texts.append(ceiling_label)
    fig.legend(handles, texts, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 1.0))
    return fig


def mechanism_grid(
    rows: pd.DataFrame,
    predictors: Mapping[str, str],
    *,
    value_column: str = "r2_p0",
    trunk_labels: Mapping[str, str] | None = None,
    trunk_colors: Mapping[str, str] | None = None,
    ylabel: str = r"$R^2$ of $\Delta P_0$ at that checkpoint",
    ncols: int = 2,
) -> Figure:
    r"""Plot 4: the checkpoint-level regression of $R^2$ on what moved.

    ``predictors`` maps a column to its display label -- the drift components
    $\rho$ and $r$, the current behaviour level $b_t$, and
    ``steps_since_realignment``. Each panel fits $R^2$ against one of them over
    every checkpoint in ``rows``, and colour identifies which trunk a
    checkpoint came from.

    This is the level-2 analysis, so a point is a *checkpoint*, not a probe
    dataset: the eight probes at a checkpoint were already spent producing the
    single $R^2$ plotted here. Dense sampling is what makes the panel
    populated at all -- measuring every ``t`` gives 19 rows where measuring
    ``t`` in ``{0, 2, 4, 6}`` would give 10 -- and the varied schedules are
    what stop drift and behaviour level from moving together, which is the
    condition for their contributions to be separable.
    """
    style.apply_style()
    trunk_labels = trunk_labels or {}
    trunk_colors = trunk_colors or {}
    n = len(predictors)
    ncols = max(1, min(ncols, n))
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.4 * ncols, 3.5 * nrows), squeeze=False
    )
    axes_flat = axes.flatten()

    for i, (ax, (column, label)) in enumerate(zip(axes_flat, predictors.items())):
        fit = linear_fit(rows[column], rows[value_column])
        if len(rows) >= 2:
            line_x = _fit_line_x(np.asarray(rows[column], dtype=float))
            ax.plot(
                line_x,
                fit.predict(line_x),
                color=style.SECONDARY_INK,
                linewidth=1.5,
                zorder=2,
            )
        for key, group in rows.groupby("trunk", sort=True):
            trunk = str(key)
            ax.scatter(
                group[column],
                group[value_column],
                s=30,
                color=trunk_colors.get(trunk, style.BLUE),
                alpha=0.85,
                edgecolor=style.SURFACE,
                linewidth=0.8,
                zorder=3,
                label=trunk_labels.get(trunk, f"Trunk {trunk}"),
            )
        entries = [(rf"$R^2$ = {fit.r2:.2f}", style.INK)]
        if i == 0:
            # Every panel regresses the same checkpoints, so ``n`` belongs to
            # the figure rather than to a panel -- stated once, in the one the
            # eye reaches first, since it is what stops the level-2 count from
            # being read as the level-1 one (a point here is a checkpoint, not
            # a probe dataset).
            entries.append((rf"$n$ = {len(rows)} checkpoints", style.SECONDARY_INK))
        _corner_text(ax, entries, fontsize=9)
        ax.set_xlabel(label)
        ax.set_ylabel(ylabel)
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    handles, texts = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, texts, loc="lower center", ncol=len(handles) or 1)
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    return fig


def phase_contrast(
    pairs: pd.DataFrame,
    *,
    series: Sequence[str] = ("p0", "pt"),
    series_labels: Mapping[str, str] | None = None,
    trunk_colors: Mapping[str, str] | None = None,
    ylabel: str = r"$R^2$ over the probe set",
) -> Figure:
    r"""Plot 4b: what one re-alignment step does to predictive accuracy.

    Each pair of checkpoints straddles a single Normal driver, with the trunk
    and the probe set held fixed, so the vertical distance between the two
    markers is attributable to that one step -- unlike a difference read off
    the trend in ``t``, which also carries everything else that accumulated in
    between.

    Drawn as a before/after pair rather than as a bar of the difference so that
    the *level* stays visible: a drop from 0.9 to 0.6 and one from 0.4 to 0.1
    are the same bar and very different findings.
    """
    style.apply_style()
    series = list(series)
    series_labels = series_labels or {}
    trunk_colors = trunk_colors or {}
    ordered = pairs.sort_values(["trunk", "t_before"]).to_dict("records")
    x = np.arange(len(ordered))

    fig, axes = plt.subplots(
        1, len(series), figsize=(3.2 * len(series) + 1.6, 4.4), sharey=True,
        squeeze=False,
    )
    for ax, name in zip(axes[0], series):
        for i, row in enumerate(ordered):
            color = trunk_colors.get(row["trunk"], style.BLUE)
            before = row[f"r2_{name}_before"]
            after = row[f"r2_{name}_after"]
            ax.plot([i, i], [before, after], color=color, linewidth=1.6, zorder=2)
            ax.scatter(
                [i], [before], s=42, facecolor=style.SURFACE, edgecolor=color,
                linewidth=1.6, zorder=3,
            )
            ax.scatter(
                [i], [after], s=42, color=color, edgecolor=style.SURFACE,
                linewidth=0.8, zorder=3,
            )
            ax.annotate(
                f"{after - before:+.2f}",
                (i, max(before, after)),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
                color=style.SECONDARY_INK,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([row["pair"] for row in ordered], rotation=20, ha="right")
        # Room for the delta labels on the outermost pairs, which sit above a
        # marker at the very edge of the data range.
        ax.set_xlim(-0.7, len(ordered) - 0.3)
        ax.set_ylim(-0.05, 1.15)
        ax.set_title(series_labels.get(name, name), color=style.SECONDARY_INK)
    axes[0][0].set_ylabel(ylabel)

    handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="none", markersize=7,
            markerfacecolor=style.SURFACE, markeredgecolor=style.SECONDARY_INK,
        ),
        plt.Line2D(
            [], [], marker="o", linestyle="none", markersize=7,
            color=style.SECONDARY_INK,
        ),
    ]
    fig.legend(
        handles,
        ["Before the Normal driver", "After it"],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    return fig


def _draw_overlay(
    ax: Axes,
    primary: Mapping[str, Sequence[float]],
    replicate: Mapping[str, Sequence[float]],
    *,
    colors: Mapping[str, str],
    marks: Mapping[str, style.DatasetMark] | None,
    reference: float | None,
    reference_label: str | None,
) -> None:
    """One panel of an overlay plot: solid primary lines, dashed replicates.

    ``marks`` styles a series as a dataset -- family shape, version fill, and
    the version's line colour. A Normal dataset therefore draws as hollow marks
    on a grey line: white is the *fill*, and a white line would be invisible
    against the page.
    """
    if reference is not None:
        ax.axhline(
            reference,
            color=style.MUTED,
            linestyle="--",
            linewidth=1.0,
            zorder=1,
            label=reference_label,
        )

    def styling(label: str, i: int) -> tuple[str, dict]:
        mark = marks.get(label) if marks else None
        if mark is None:
            color = colors.get(label, style.categorical_color(i))
            return color, {
                "marker": "o",
                "markersize": 3.6,
                "markerfacecolor": color,
                "markeredgecolor": style.SURFACE,
                "markeredgewidth": 0.6,
            }
        return mark.line, {
            "marker": mark.marker,
            "markersize": 5.5,
            "markerfacecolor": mark.face,
            "markeredgecolor": mark.edge,
            "markeredgewidth": 0.9,
        }

    for i, (label, values) in enumerate(primary.items()):
        y = np.asarray(values, dtype=float)
        color, marker_style = styling(label, i)
        ax.plot(
            np.arange(y.size),
            y,
            color=color,
            linewidth=1.6,
            label=label,
            zorder=3,
            **marker_style,
        )
    for i, (label, values) in enumerate(replicate.items()):
        y = np.asarray(values, dtype=float)
        color, _ = styling(label, i)
        ax.plot(
            np.arange(y.size),
            y,
            color=color,
            linestyle=(0, (3, 2)),
            linewidth=1.3,
            alpha=0.85,
            zorder=2,
        )


def overlay_lines(
    primary: Mapping[str, Sequence[float]],
    replicate: Mapping[str, Sequence[float]] | None = None,
    *,
    ylabel: str,
    xlabel: str = "Checkpoint $t$",
    colors: Mapping[str, str] | None = None,
    marks: Mapping[str, style.DatasetMark] | None = None,
    reference: float | None = None,
    reference_label: str | None = None,
    replicate_label: str = "Reseeded replicate (dashed)",
) -> Figure:
    r"""Plot 5: one line per series over ``t``, with a reseeded run overlaid.

    ``replicate`` is drawn dashed in each series' own colour, so the reseed
    comparison rides the same colour assignment as the run it replicates and
    costs no extra hue. Section 6c is explicit that ``n = 2`` detects gross
    instability and gives no error bars, which is precisely what a visual
    overlay says and a shaded band would overstate.

    ``marks`` identifies each series as a dataset rather than by a hue of its
    own. Eight probes would otherwise take all eight categorical slots, leaving
    nothing for anything else and making the reader learn an arbitrary
    dataset-to-colour map; shape-per-family plus the version ramp is
    self-describing and leaves the hues free.

    No error bars: section 8 establishes that $\Delta P_t$ and $z_t$ involve no
    sampling -- fixed prompts, fixed responses, forward passes only -- so the
    quantities on this axis have no measurement error to draw.
    """
    style.apply_style()
    colors = colors or {
        label: style.categorical_color(i) for i, label in enumerate(primary)
    }
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    _draw_overlay(
        ax,
        primary,
        replicate or {},
        colors=colors,
        marks=marks,
        reference=reference,
        reference_label=reference_label,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    handles, texts = ax.get_legend_handles_labels()
    if replicate:
        handles.append(
            plt.Line2D([], [], color=style.MUTED, linestyle=(0, (3, 2)), linewidth=1.3)
        )
        texts.append(replicate_label)
    ax.legend(handles, texts, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    return fig


def overlay_grid(
    panels: Mapping[str, Mapping[str, Sequence[float]]],
    replicates: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
    *,
    ylabels: Mapping[str, str] | None = None,
    xlabel: str = "Checkpoint $t$",
    colors: Mapping[str, str] | None = None,
    replicate_label: str = "Reseeded replicate (dashed)",
    ncols: int = 2,
) -> Figure:
    r"""Plot 5 for the latent state: one panel per $z_t$ component.

    ``panels`` maps a panel label (e.g. ``r"$\rho_t$"``) to that panel's series,
    and ``replicates`` the same for the reseeded run. Raw units, one y-axis per
    panel: $\rho$ starts at 1 and $r$ at the persona vector's norm, but $p$ and
    $q$ start at essentially zero, so expressing them as a percentage of step 0
    would be division by noise.
    """
    style.apply_style()
    replicates = replicates or {}
    ylabels = ylabels or {}
    n = len(panels)
    ncols = max(1, min(ncols, n))
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.4 * ncols, 3.3 * nrows), sharex=True, squeeze=False
    )
    axes_flat = axes.flatten()
    for i, (ax, (label, series)) in enumerate(zip(axes_flat, panels.items())):
        _draw_overlay(
            ax,
            series,
            replicates.get(label, {}),
            colors=colors or {},
            marks=None,
            reference=None,
            reference_label=None,
        )
        ax.set_ylabel(ylabels.get(label, label))
        # The panels share an x-axis, so only those actually showing ticks --
        # the bottom row, plus any panel left exposed by a short final row --
        # get the label.
        if i >= n - ncols:
            ax.set_xlabel(xlabel)
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    handles, texts = axes_flat[0].get_legend_handles_labels()
    if any(replicates.values()):
        handles.append(
            plt.Line2D([], [], color=style.MUTED, linestyle=(0, (3, 2)), linewidth=1.3)
        )
        texts.append(replicate_label)
    fig.legend(handles, texts, loc="lower center", ncol=min(4, len(handles) or 1))
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    return fig


def diversity_bar(
    df: pd.DataFrame,
    *,
    condition_col: str = "condition",
    value_col: str = "delta_behavior",
    order: Sequence[str] | None = None,
    labels: Mapping[str, str] | None = None,
    ylabel: str = r"Behaviour change $\Delta b$",
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
    fig.tight_layout()
    return fig
