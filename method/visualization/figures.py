r"""Figure-generating functions for the sequential fine-tuning experiments
(proposal Section "Experiments and Plots").

Every function takes plain arrays / DataFrames and returns a
:class:`matplotlib.figure.Figure` -- never a :class:`~.schema.Trajectory`
directly -- so the same code plots real measurements (once experiments have
run) and :mod:`method.visualization.synthetic` fixtures identically. Save the
result with :func:`method.visualization.style.save_figure`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
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


#: The judge's own range. Every behaviour score in this project is a 0-100
#: judge average, so a panel showing one is drawn on the whole of it rather
#: than on the part its data happens to occupy. These are the ticks: what the
#: axis claims to span.
BEHAVIOUR_TICKS = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)

#: ...and these are the limits it is actually drawn to. The margin is
#: clearance, not range: a probe scoring 0 is common (an aligned model on an
#: evil judge) and a mark centred on the spine would be sliced in half by it,
#: which reads as a smaller, differently-shaped mark rather than as a point at
#: the floor. Padding the axis fixes that without letting a mark escape its own
#: panel, which is what turning clipping off would do in a grid this dense.
BEHAVIOUR_LIMITS = (-5.0, 105.0)

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
    """One scatter series plus its least-squares line, labelled with its $r$.

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
            label=rf"{label} ($r$={fit.corr:.2f})",
        )
    return fit


def scatter_projection_correlation(
    delta_p_0: ArrayLike,
    delta_p_hat_t: ArrayLike,
    delta_behavior: ArrayLike,
    *,
    xlabel: str = r"Projection difference $\Delta P$",
    ylabel: str = r"Behaviour change $\Delta b_{t+1}$",
) -> Figure:
    r"""RQ1: compare $\Delta P_0$ and $\Delta \hat{P}_t$ as predictors.

    Two series share one axis: $\Delta P_0$ (blue), frozen at the base model,
    against $\Delta \hat{P}_t$ (orange), whose axis and encoder are recomputed
    at the checkpoint while its predicted answer remains $M_0$'s -- both
    plotted against the behaviour change the step actually caused.
    """
    style.apply_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    _scatter_with_fit(
        ax, delta_p_0, delta_behavior, color=style.BLUE, label=r"$\Delta P_0$"
    )
    _scatter_with_fit(
        ax,
        delta_p_hat_t,
        delta_behavior,
        color=style.ORANGE,
        label=r"$\Delta \hat{P}_t$",
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
    reference line at 100. This is the "does $\Delta \hat{P}_t$ / $\rho_t$ /
    $r_t$ / $p_t$ / $q_t$ deviate from its step-0 value" plot.
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
    rows: Sequence[str] | None = None,
    row_col: str = "trait",
    row_labels: Mapping[str, str] | None = None,
    row_scales: Mapping[str, str] | None = None,
    dataset_col: str = "dataset",
    condition_col: str = "condition",
    value_col: str = "behavior",
    conditions: Sequence[str] = HYSTERESIS_CONDITIONS,
    condition_labels: Sequence[str] | None = None,
    dataset_labels: Mapping[str, str] | None = None,
    start_col: str | None = None,
    reference: float | Mapping[str, float] | None = None,
    reference_label: str | None = None,
    ylabel: str = r"Trait score $b_T$",
) -> Figure:
    r"""RQ2 hysteresis bar chart: is a realigned model easier to re-misalign?

    One column per dataset, one bar per ``condition`` within a panel, and one
    row per value of ``row_col`` -- the measured trait crossed with the trait
    whose Normal data did the re-aligning, in the figure this draws for exp3.
    Error bars show the std across seeds.

    The four combinations belong in one figure because the reading is the
    comparison between them: whether re-aligning on a *different* trait's
    Normal data leaves the same residue as re-aligning on the target's own is
    the control that separates hysteresis from a dataset artefact, and it is
    not a comparison the reader should have to make across four separately
    scaled figures.

    ``row_scales`` says which rows are the same quantity and so share a y-axis
    -- the two re-alignment sources for one measured trait -- because the
    traits are not: they are different judges on different behaviours, and one
    trait topping out at a third of the other's range would spend two thirds of
    its panels on empty space.

    ``reference`` may be one line for the figure or one per row, since $b_0$ is
    a property of the trait being measured.

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
    panels = _facets(df, rows, row_labels, column=row_col)

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
        len(panels),
        len(datasets),
        figsize=(width_in, 2.7 * len(panels) + 1.7),
        squeeze=False,
    )
    for row, (key, frame, label) in enumerate(panels):
        line = reference.get(key) if isinstance(reference, Mapping) else reference
        stats = frame.groupby([dataset_col, condition_col])[value_col].agg(
            ["mean", "std"]
        )
        starts = (
            frame.groupby([dataset_col, condition_col])[start_col].mean()
            if start_col
            else None
        )
        for ax, dataset in zip(axes[row], datasets):
            if row == 0:
                ax.set_title(dataset_labels.get(dataset, display_dataset_name(dataset)))
            if frame.empty:
                _mark_empty(ax)
                continue
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
                # Drawn as a tick across the bar rather than a second bar: it
                # marks where this bar's growth began, so it belongs *on* it.
                ax.errorbar(
                    x,
                    [starts.get((dataset, c), np.nan) for c in conditions],
                    xerr=width / 2,
                    fmt="none",
                    ecolor=style.INK,
                    elinewidth=1.2,
                    zorder=5,
                )
            if line is not None:
                ax.axhline(
                    line,
                    color=style.SECONDARY_INK,
                    linestyle="--",
                    linewidth=1.0,
                    zorder=2,
                )
            ax.axhline(0, color=style.BASELINE, linewidth=0.8, zorder=1)
            ax.set_xticks(x)
            # Full ink, unlike the muted numeric ticks elsewhere: these names
            # are what identifies an arm now that the legend no longer does.
            ax.set_xticklabels(condition_labels, fontsize=10, color=style.INK)
            ax.set_xlim(-0.5 - (1 - width) / 2, len(conditions) - 0.5 + (1 - width) / 2)
        axes[row][0].set_ylabel(label, fontsize=10)

    scales: dict[str, list[Axes]] = {}
    for row, (key, _, _) in enumerate(panels):
        scales.setdefault(row_scales.get(key, key) if row_scales else "", []).extend(
            axes[row]
        )
    _share_blocks(scales.values(), y=True)
    _label_outer(axes, bottom_rows=[len(panels) - 1])

    handles: list[Artist] = []
    texts: list[str] = []
    if reference is not None:
        handles.append(
            plt.Line2D([], [], color=style.SECONDARY_INK, linestyle="--", linewidth=1.0)
        )
        texts.append(reference_label or "Base model")
    if start_col:
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
    x: float = 0.96,
    y: float = 0.04,
    fontsize: float = 7.5,
    ha: str = "right",
    va: str = "bottom",
) -> None:
    """Stacked ``(text, colour)`` annotations in a panel corner.

    Used where a panel is too small for a legend of its own. The text always
    names both the quantity and the series it reports (``"$r(\\Delta P_0)$ =
    0.42"``), so colour is redundant reinforcement of an identity the words
    already carry rather than the only channel encoding it.

    Bottom-right by default. A positively correlated scatter runs from the
    bottom-left to the top-right, so that corner and the top-left are the two
    the data tends to leave empty; of those, the bottom-right keeps the
    annotation clear of the column headers and the phase note that share the
    top of the panel in a grid. Pass ``y``/``va`` to move it where a panel's
    own cloud makes the default the crowded one -- a strongly *negative*
    relationship fills exactly that corner.
    """
    rows = len(entries)
    for i, (text, color) in enumerate(entries):
        # Entries read top-down whichever corner they sit in, so a stack
        # anchored at the bottom counts up from the last one rather than down
        # from the first.
        level = (rows - 1 - i) if va == "bottom" else -i
        ax.text(
            x,
            y + 0.105 * level,
            text,
            transform=ax.transAxes,
            fontsize=fontsize,
            va=va,
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
    legend_rows: int,
    ylabel: str | None = None,
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

    Both labels are optional, because a shared label is only meaningful where
    the panels share the quantity: a grid whose ticks already name themselves
    (the hysteresis schedules) has nothing for a shared x-label to add, and one
    whose columns are different quantities in different units (the $z_t$
    components) has nothing for a shared y-label to name.
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
    if ylabel:
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


def _mark_empty(ax: Axes) -> None:
    """Mark a panel whose runs have not happened, rather than dropping it.

    A half-finished sweep must leave its panel visibly empty: silently
    narrowing the grid would hide which cells are still missing, and in a grid
    whose rows and columns are read against each other it would also break the
    alignment that makes them comparable.
    """
    _corner_text(
        ax, [("not run", style.MUTED)], x=0.5, y=0.5, ha="center", va="center"
    )


def _facets(
    df: pd.DataFrame,
    keys: Sequence[str] | None,
    labels: Mapping[str, str] | None,
    *,
    column: str,
) -> list[tuple[str, pd.DataFrame, str]]:
    """Split a frame along one dimension of a grid: ``(key, rows, label)`` each.

    Used wherever a figure panels the measured trait, which most of them now
    do: a trait is read against its own persona vector and its own judge, so
    two traits on one pair of axes would invite a comparison that is not in the
    same units, while two traits in one figure is exactly the comparison the
    experiments are for.

    A frame with no such column is one unlabelled facet, so a caller with
    nothing to split by -- the synthetic fixtures, a single-trait sweep --
    passes its frame unchanged rather than inventing a column to satisfy the
    signature.
    """
    if column not in df:
        return [("", df, "")]
    order = list(keys) if keys else sorted(df[column].unique())
    labels = labels or {}
    return [(key, _facet_frame(df, column, key), labels.get(key, key)) for key in order]


def _facet_frame(df: pd.DataFrame, column: str, key: str) -> pd.DataFrame:
    """One facet's rows, or the whole frame where there is no such column."""
    return df[df[column] == key] if column in df else df


def _share_blocks(
    blocks: Iterable[Sequence[Axes]], *, x: bool = False, y: bool = False
) -> None:
    """Join each block of panels onto one scale, leaving the blocks apart.

    ``plt.subplots`` shares everything, or shares by whole row or column, and a
    grid stacked one trait per block wants neither. Within a trait the panels
    have to be read against each other -- a flattening slope and a shrinking
    $\\Delta b$ range look identical on per-panel scales, which is the one
    confusion these grids exist to prevent -- while across two traits a shared
    scale would rescale one trait's spread by the other's range.
    """
    for block in blocks:
        anchor, *rest = block
        for ax in rest:
            if x:
                ax.sharex(anchor)
            if y:
                ax.sharey(anchor)
        # Joining takes the anchor's limits as they stand -- which are the
        # anchor's own data's, since the panels were drawn before the join --
        # and does not re-run the autoscale over the group. Without this the
        # block would be scaled to whatever its first panel happened to hold.
        anchor.autoscale_view()


def _label_outer(axes: np.ndarray, *, bottom_rows: Sequence[int]) -> None:
    """Keep tick labels on the left column and on each block's bottom row.

    The counterpart to :func:`_share_blocks`: sharing an axis is what makes the
    interior labels redundant, and ``plt.subplots`` only drops them for the
    sharing it applied itself. ``bottom_rows`` is per block rather than the
    grid's last row alone, because blocks are on different scales -- the row
    above a block boundary is the last row of its own axis and has to say what
    that axis reads.
    """
    for r, row in enumerate(axes):
        for c, ax in enumerate(row):
            ax.tick_params(labelleft=c == 0, labelbottom=r in bottom_rows)


def scatter_validation(
    df: pd.DataFrame,
    *,
    traits: Sequence[str] | None = None,
    trait_labels: Mapping[str, str] | None = None,
    xlabel: str = r"Projection difference $\Delta P_0$",
    ylabel: str = r"Behaviour change $\Delta b_1$",
) -> Figure:
    r"""Plot 1: the $t = 0$ validation fan over all 24 datasets, one trait per panel.

    One point per dataset fine-tuned straight from $M_0$, reproducing Figure 8
    of the persona-vectors paper. This is the pipeline's gate, not part of the
    decay analysis: it is drawn over 24 datasets while the decay curve is drawn
    over the 8 probes, and correlation estimates are sensitive enough to range
    restriction and to ``n`` that comparing the two would manufacture a decay
    out of nothing (section 5).

    The traits share a figure but not a scale. Each is a different persona
    vector and a different judge, so $\Delta P$ and $\Delta b$ mean different
    things across panels and a shared axis would invite reading one trait's
    spread against the other's; what the panels do share is the encoding, so
    the mark key is stated once for the figure.

    One fitted series per panel, so its statistics are annotated in the panel
    rather than put in a legend, and the legend keys the *encoding* -- shape
    per family, fill per version -- which is what makes a 24-point scatter
    readable without 24 entries.
    """
    style.apply_style()
    traits = list(traits) if traits else sorted(df["trait"].unique())
    trait_labels = trait_labels or {}

    fig, axes = plt.subplots(
        1, len(traits), figsize=(4.9 * len(traits) + 0.6, 4.6), squeeze=False
    )
    for ax, trait in zip(axes[0], traits):
        panel = df[df["trait"] == trait]
        ax.set_title(trait_labels.get(trait, trait))
        if panel.empty:
            _mark_empty(ax)
            continue
        fit = _scatter_with_fit(
            ax,
            panel["delta_p_0"],
            panel["delta_b"],
            color=style.BLUE,
            label=r"$\Delta P_0$",
            yerr=panel["se_delta_b"] if "se_delta_b" in panel else None,
            size=44,
            datasets=list(panel["dataset"]) if "dataset" in panel else None,
        )
        ax.axhline(0, color=style.BASELINE, linewidth=0.8, zorder=1)
        _corner_text(
            ax,
            [
                (rf"$r$ = {fit.corr:.2f}   slope = {fit.slope:.2f}", style.INK),
                (rf"$n$ = {len(panel)} datasets", style.SECONDARY_INK),
            ],
            fontsize=9,
        )

    handles, texts = dataset_legend(list(df["dataset"]) if "dataset" in df else [])
    ncol = min(6, len(handles))
    if handles:
        fig.legend(handles, texts, loc="lower center", ncol=ncol)
    _layout_grid(
        fig,
        axes.flat,
        xlabel=xlabel,
        ylabel=ylabel,
        legend_rows=-(-len(handles) // ncol) if handles else 0,
    )
    return fig


@dataclass(frozen=True)
class _DecaySeries:
    """One projection-difference series as a decay panel draws it."""

    column: str
    #: The symbol the series is written as, without math delimiters.
    symbol: str
    color: str
    #: What the legend says the series is, beyond its symbol.
    gloss: str

    @property
    def label(self) -> str:
        return f"${self.symbol}$"


#: The series a decay panel can hold, in the order they are layered. Each is a
#: different rule for what may be current at $M_t$ -- the axis, the encoder,
#: the prediction -- so they share the panel and differ only in hue. A series
#: whose column is missing or incomplete for a panel is skipped there (see
#: :func:`_panel_series`), which is what lets one grid mix a trunk that was
#: re-measured with two that were not.
_DECAY_SERIES: tuple[_DecaySeries, ...] = (
    _DecaySeries("delta_p_0", r"\Delta P_0", style.BLUE, "everything at $M_0$"),
    _DecaySeries(
        "delta_p_hat_v0",
        r"\Delta \hat{P}_t^{(\mathbf{v}_0)}",
        style.PURPLE,
        "$M_0$'s axis",
    ),
    _DecaySeries(
        "delta_p_hat_t", r"\Delta \hat{P}_t", style.ORANGE, "$M_0$'s answers"
    ),
    _DecaySeries(
        "delta_p_full_t", r"\Delta P_t", style.GREEN, "nothing approximated"
    ),
)


def _panel_series(
    panel: pd.DataFrame, wanted: Sequence[str] | None = None
) -> list[_DecaySeries]:
    """The series this panel has a complete column for, of those asked for.

    Incomplete is treated as absent. A fit over whichever probes happened to be
    measured would be a correlation over a different probe set than the one
    labelled beside it, and the panels exist to be compared.
    """
    return [
        series
        for series in _DECAY_SERIES
        if (wanted is None or series.column in wanted)
        and series.column in panel
        and not panel[series.column].isna().any()
    ]


#: How many positions along its own line a label is offered, how many
#: text-heights of clearance off the line it may stand at, and how far it is
#: held off whatever it is placed against, in points.
_FIT_LABEL_STEPS = 17
_FIT_LABEL_AWAY = 3
_FIT_LABEL_PAD = 2.5

#: How much of the panel a mark claims, in points: half the width of the
#: largest one the grid draws, plus its outline. A label that comes this close
#: to a point is printing over it.
_MARK_RADIUS = 4.0

#: How many points a fit line is tested as. Enough that a label cannot slip
#: between two of them and call the line missed.
_CURVE_SAMPLES = 40

#: What a position pays for each thing it would print over. A mark is a
#: measurement and the panel exists to show it; a fit line is thin and a reader
#: follows it past a number without losing it; a rule spans the panel, so it
#: can be picked up again either side; a label already printed cannot be
#: overlapped at all. Leaving the panel is not costed but forbidden, and the
#: price here only orders the fallback for a label with nowhere clear to go.
_COST_MARK = 12.0
_COST_CURVE = 3.0
_COST_RULE = 4.0
_COST_TAKEN = 100.0
_COST_OFF_PANEL = 500.0

#: ...and what it pays for getting there: a step along its own line, a
#: text-height of clearance off it, hanging under the line rather than over it,
#: and running forward off the line's end rather than back along it. All are
#: cheap next to hiding a mark, which is the point of the search -- a label
#: anywhere near a line still names that line, so moving is nearly free and
#: hiding data is not -- and they are ordered so that, all else equal, a label
#: sits at the end of its line, over it, reading back into the panel.
_COST_STEP = 0.4
_COST_AWAY = 1.5
_COST_UNDER = 1.0
_COST_FORWARD = 0.5


@dataclass(frozen=True)
class _Box:
    """A rectangle in axes fractions."""

    x0: float
    y0: float
    x1: float
    y1: float

    def holds(self, x: float, y: float, pad: tuple[float, float] = (0.0, 0.0)) -> bool:
        """Whether ``(x, y)`` falls inside, grown by ``pad`` on each side."""
        px, py = pad
        return self.x0 - px <= x <= self.x1 + px and self.y0 - py <= y <= self.y1 + py

    def hits(self, other: _Box) -> bool:
        return not (
            self.x1 < other.x0
            or other.x1 < self.x0
            or self.y1 < other.y0
            or other.y1 < self.y0
        )

    @property
    def spilled(self) -> float:
        """How far it sticks out of the panel, summed over its four sides."""
        return (
            max(0.0, -self.x0)
            + max(0.0, -self.y0)
            + max(0.0, self.x1 - 1.0)
            + max(0.0, self.y1 - 1.0)
        )


@dataclass(frozen=True)
class _Panel:
    """One panel's geometry, in the axes fractions a label is placed in.

    Fractions rather than data units because a label's size is a size on paper:
    the same number of points is a different number of $\\Delta P$ in every
    column of a grid whose traits are on their own scales, but always the same
    fraction of a panel that is the same size.
    """

    xlim: tuple[float, float]
    ylim: tuple[float, float]
    width_in: float
    height_in: float

    @classmethod
    def of(cls, ax: Axes) -> _Panel:
        """Measure ``ax`` as it is currently laid out.

        Only valid once the grid around it has been laid out: a panel is not
        the size it will be printed at until then, and neither is anything
        placed by measuring it.
        """
        box = ax.get_window_extent()
        dpi = ax.figure.dpi
        return cls(
            xlim=ax.get_xlim(),
            ylim=ax.get_ylim(),
            width_in=box.width / dpi,
            height_in=box.height / dpi,
        )

    def at(self, x: float, y: float) -> tuple[float, float]:
        """``(x, y)`` in data units, as a fraction of the panel."""
        (x0, x1), (y0, y1) = self.xlim, self.ylim
        return (x - x0) / (x1 - x0), (y - y0) / (y1 - y0)

    def size(self, width: float, height: float) -> tuple[float, float]:
        """A size in points, as a fraction of the panel."""
        return width / 72 / self.width_in, height / 72 / self.height_in


@dataclass(frozen=True)
class _FitLabel:
    """A label and the fit line it is to be printed beside."""

    text: str
    color: str
    fit: LinearFit
    #: The x range the line was drawn over: the stretch of it a label may be
    #: placed along, and no further.
    span: tuple[float, float]

    @property
    def rising(self) -> bool:
        return self.fit.slope >= 0

    @property
    def start(self) -> float:
        """The height of the line's upper end, where a label is offered first."""
        return float(self.fit.predict(self.walk(0.0)[0]))

    def walk(self, back: float) -> tuple[float, float]:
        """The point a fraction ``back`` along the line from its upper end."""
        low, high = self.span
        top, bottom = (high, low) if self.rising else (low, high)
        x = top + (bottom - top) * back
        return x, float(self.fit.predict(x))


def _fit_label(text: str, color: str, x: np.ndarray, fit: LinearFit) -> _FitLabel:
    """Bind ``text`` to the line ``fit`` draws over ``x``."""
    line_x = _fit_line_x(x)
    return _FitLabel(
        text=text,
        color=color,
        fit=fit,
        span=(float(line_x[0]), float(line_x[-1])),
    )


@dataclass(frozen=True)
class _Placement:
    """One position a label could take, and the footprint it would have there.

    Anchored on the line, in axes fractions, plus the offset in points the text
    is printed at from there -- the two coordinate systems a label lives in: it
    belongs to a place in the panel, and it takes up a size on the page.
    """

    box: _Box
    x: float
    y: float
    dx: float
    dy: float
    ha: str
    va: str
    #: What the position had to give up to exist: steps back along the line,
    #: text-heights of clearance off it, hanging under the line rather than
    #: over it, and running forward off its end rather than back along it.
    step: int
    away: int
    under: bool
    forward: bool


def _text_size(ax: Axes, text: str, fontsize: float) -> tuple[float, float]:
    """The size ``text`` prints at, in points, including its bbox padding.

    Measured rather than estimated from the character count: the labels are
    mathtext, and a placement search is only as good as its idea of how much
    room the thing it is placing takes.
    """
    artist = ax.text(0.0, 0.0, text, fontsize=fontsize, transform=ax.transAxes)
    # Every figure here is drawn on the Agg canvas the style module fixes, and
    # the base canvas class does not declare the renderer that one has.
    renderer = ax.figure.canvas.get_renderer()  # type: ignore[attr-defined]
    extent = artist.get_window_extent(renderer)
    artist.remove()
    scale = 72 / ax.figure.dpi
    return (
        extent.width * scale + 2 * _FIT_LABEL_PAD,
        extent.height * scale + 2 * _FIT_LABEL_PAD,
    )


def _placements(
    label: _FitLabel, panel: _Panel, size: tuple[float, float]
) -> Iterator[_Placement]:
    """Every position ``label`` could take, in points along and off its line.

    Four to a point on the line -- over it or under it, running back along it
    or forward off its end -- because which of the wedges a line divides its
    panel into is the empty one is a property of the cloud, not of the line: a
    rising fit through a cloud that flattens at the top has room above it
    exactly where a fit through a rising cloud has none.

    And each of those at a few text-heights of clearance off the line, which is
    what reaches the empty half of a panel whose cloud lies along the line
    itself. A label standing off its line still reads as that line's while it
    is the nearest one to it, and in the panels that need the room -- a flat
    sycophancy trunk, where three near-parallel fits run through one band of
    marks -- the space either side of the band is the only space there is.
    """
    width_pt, height_pt = size
    width, height = panel.size(width_pt, height_pt)
    pad_x, pad_y = panel.size(_FIT_LABEL_PAD, _FIT_LABEL_PAD)
    for step in range(_FIT_LABEL_STEPS):
        x, y = panel.at(*label.walk(step / (_FIT_LABEL_STEPS - 1)))
        for under in (False, True):
            # Back along the line is the direction that keeps the text beside
            # the fit rather than trailing off the end of it; the other is
            # offered anyway, and costed, for the panels with no room there.
            back = label.rising is not under
            for left in (back, not back):
                for away in range(_FIT_LABEL_AWAY):
                    lift = pad_y + away * height
                    anchor_x = x - pad_x if left else x + pad_x
                    anchor_y = y - lift if under else y + lift
                    x0, x1 = (
                        (anchor_x - width, anchor_x)
                        if left
                        else (anchor_x, anchor_x + width)
                    )
                    y0, y1 = (
                        (anchor_y - height, anchor_y)
                        if under
                        else (anchor_y, anchor_y + height)
                    )
                    yield _Placement(
                        box=_Box(x0, y0, x1, y1),
                        x=x,
                        y=y,
                        dx=-_FIT_LABEL_PAD if left else _FIT_LABEL_PAD,
                        dy=(
                            -(_FIT_LABEL_PAD + away * height_pt)
                            if under
                            else _FIT_LABEL_PAD + away * height_pt
                        ),
                        ha="right" if left else "left",
                        va="top" if under else "bottom",
                        step=step,
                        away=away,
                        under=under,
                        forward=left is not back,
                    )


def _label_fits(
    ax: Axes,
    entries: Sequence[tuple[str, str, ArrayLike, LinearFit]],
    *,
    rules: Sequence[float] = (),
    fontsize: float = 7.0,
) -> None:
    r"""Print each fit's $r$ beside the line it was measured from.

    Direct labelling in place of a per-panel key. A key makes the reader carry
    a colour from a corner of the panel over to a line to find out which fit a
    number belongs to, and at this grid's panel size a three-entry stack of
    them takes more of the panel than the data does. A label against its own
    line is read where the line is looked at, and it is what lets the text drop
    the series name and print $r$ alone: position says which line the number
    belongs to, with colour repeating it.

    Where along the line is chosen by looking at what is already drawn in the
    panel. Every position on the line is costed by what its label would cover
    there -- marks first, then the other fits and any rule in ``rules`` -- plus
    a little for how far it sits from the line's upper end, and the cheapest
    wins. Moving costs almost nothing next to hiding a point, so a label takes
    the end of its line when that corner is clear and slides down into the
    panel's empty half when it is not, which is the whole reason to search
    rather than to anchor: which corner of a decay panel is empty changes from
    trunk to trunk and checkpoint to checkpoint, and there are forty-two of
    them.

    ``entries`` are ``(text, colour, x, fit)``, one per drawn series, sharing
    ``ax``. A series of fewer than two points has no line to label and is
    skipped, matching :func:`_scatter_with_fit`, which draws none. Labels are
    placed in turn and never overlap: a series that agrees with one already
    placed -- $\Delta P_0$ and $\Delta P_t$ agreeing is what an unaged trunk
    looks like -- finds its line's best spots taken and moves along it.

    ``rules`` are heights the panel has drawn a line across its whole width at.
    A rule is read along its length, so a number parked on it costs more than
    the same number anywhere else in the panel, though less than a hidden mark:
    a rule interrupted is still a rule.

    Call this only once the grid has been laid out. Everything here is measured
    off the panel as it stands, and a panel that has still to be fitted around
    a legend is not the shape the label will be printed in.
    """
    labels = [
        _fit_label(text, color, np.asarray(x, dtype=float), fit)
        for text, color, x, fit in entries
        if np.asarray(x, dtype=float).size >= 2
    ]
    if not labels:
        return
    panel = _Panel.of(ax)
    marks = [
        panel.at(float(x), float(y))
        for collection in ax.collections
        if isinstance(collection, PathCollection)
        for x, y in np.asarray(collection.get_offsets(), dtype=float)
    ]
    mark_pad = panel.size(_MARK_RADIUS, _MARK_RADIUS)
    curves = [
        [
            panel.at(*label.walk(step / (_CURVE_SAMPLES - 1)))
            for step in range(_CURVE_SAMPLES)
        ]
        for label in labels
    ]
    levels = [panel.at(0.0, level)[1] for level in rules]
    taken: list[_Box] = []

    def cost(
        placement: _Placement, others: Sequence[Sequence[tuple[float, float]]]
    ) -> float:
        """What ``placement`` would hide, plus what it gave up to get there."""
        box = placement.box
        return (
            _COST_STEP * placement.step
            + _COST_AWAY * placement.away
            + _COST_UNDER * placement.under
            + _COST_FORWARD * placement.forward
            + _COST_OFF_PANEL * box.spilled
            + _COST_MARK * sum(box.holds(x, y, mark_pad) for x, y in marks)
            + _COST_CURVE
            * sum(any(box.holds(*point) for point in curve) for curve in others)
            + _COST_RULE * sum(box.y0 <= level <= box.y1 for level in levels)
            + _COST_TAKEN * sum(box.hits(other) for other in taken)
        )

    # Highest line first, so the label with the least room to give up -- the
    # one whose line ends nearest the top of the panel -- chooses first.
    for index in sorted(range(len(labels)), key=lambda i: -labels[i].start):
        label = labels[index]
        others = [curve for i, curve in enumerate(curves) if i != index]
        options = list(_placements(label, panel, _text_size(ax, label.text, fontsize)))
        # The panel's edge is a wall rather than a cost: a label past it reads
        # as belonging to the panel next door, which is worse than anything it
        # could have been hiding inside its own. Only a label with nowhere at
        # all to go falls back on the costed order.
        inside = [option for option in options if not option.box.spilled]
        best = min(
            inside or options, key=lambda placement: cost(placement, others)
        )
        taken.append(best.box)
        ax.annotate(
            label.text,
            xy=(best.x, best.y),
            xycoords="axes fraction",
            xytext=(best.dx, best.dy),
            textcoords="offset points",
            ha=best.ha,
            va=best.va,
            fontsize=fontsize,
            color=label.color,
            zorder=5,
            # A label can end up over the data even so -- a crowded panel has
            # no clear spot, only a cheapest one -- and a marker behind it
            # costs a digit. The patch is the chart surface itself, so it reads
            # as clearance rather than as a box, and mostly rather than fully
            # opaque: enough to keep the digits off a marker's outline, not so
            # much that it erases the marker.
            bbox={
                "facecolor": style.SURFACE,
                "edgecolor": "none",
                "alpha": 0.6,
                "pad": _FIT_LABEL_PAD,
            },
        )


def decay_scatter_grid(
    df: pd.DataFrame,
    *,
    traits: Sequence[str] | None = None,
    trait_labels: Mapping[str, str] | None = None,
    trunks: Sequence[str] | None = None,
    checkpoints: Sequence[int] | None = None,
    trunk_labels: Mapping[str, str] | None = None,
    series: Sequence[str] | None = None,
    xlabel: str = r"Projection difference $\Delta P$",
    ylabel: str = r"Behaviour after the step, $b_{t+1}$",
) -> Figure:
    r"""Plot 2: one scatter panel per ``(trait, trunk, checkpoint)``.

    Rows are trunks and columns checkpoints, so a row reads as one trajectory
    ageing and a column as three trajectories at the same depth; the traits
    stack as blocks of rows. Each panel holds the ``K`` probe datasets once per
    series it has a complete column for -- against the frozen $\Delta P_0$ in
    blue, $\Delta \hat{P}_t$ in orange, and, where it was measured,
    $\Delta P_t$ in green (:data:`_DECAY_SERIES`) -- and each fit's $r$ printed
    beside the line it was measured from (:func:`_label_fits`). The hypothesis
    is visible as the blue fit flattening left-to-right while the others do
    not.

    ``series`` picks which of :data:`_DECAY_SERIES` a panel may draw, by
    column; the default is every one the frame has measured. A scatter panel
    this size shows a relationship rather than a number -- whether the cloud
    still has a line in it -- and it takes two series to show one going stale
    while another does not. The rungs between them are read as numbers, across
    checkpoints, which is a table's job and not a 1.75-inch panel's: pass the
    two ends of the ladder here and tabulate the rest.

    Panels need not all carry the same series. $\Delta P_t$ costs a generation
    pass per checkpoint, so on a partly measured sweep its column is complete
    for some trunks and not others. That asymmetry is the point of drawing them
    in one grid rather than two: the trunk that was re-measured is read in
    place, against the same axes and the same probes as the trunks that were
    not.

    The y axis is the behaviour the step actually reached, $b_{t+1}$, with the
    level it started from, $b_t$, drawn as a black rule across the panel. That
    is $\Delta b_{t+1}$ shifted by a constant -- $b_t$ is one number per panel,
    so every fit here has the slope and correlation the differences would give
    -- but it puts the prediction on the judge's own scale, where a reader can
    see how far above or below the starting level a probe landed and how much
    of the axis a whole panel's spread covers. On differences those are both
    inferences from a number the figure no longer shows.

    Correlation rather than $R^2$ because the panels are read against each
    other by eye: $r$ carries the sign of the relationship, and it falls off
    linearly rather than quadratically as the frozen projection goes stale, so
    the decay down a row is legible at the sizes this grid leaves per panel.

    The y axis is pinned to the judge's full $[0, 100]$ on every panel, trait
    blocks included (with a margin below and above it -- see
    :data:`BEHAVIOUR_LIMITS`). It is the one axis here whose bounds are a
    property of the instrument rather than of the data: both judges score out
    of 100, so the range is already the same range, and fixing it means the
    height of a mark means one thing everywhere in the figure -- a trunk that
    ends up near the ceiling looks near the ceiling, instead of filling its
    panel the way a trunk that never left the floor also would.

    $\Delta P$ is shared within a trait and not across (see
    :func:`_share_blocks`). Within one, panels on their own scales would let a
    flattening slope and a shrinking spread look identical, which is the one
    confusion this figure exists to prevent; across two, $\Delta P$ is read
    against a different persona vector, so a shared scale would compare
    quantities that are not the same quantity.

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
    blocks = _facets(df, traits, trait_labels, column="trait")
    # A row is a (trait, trunk) pair, flattened here so the drawing loop stays
    # one level deep and the trait blocks stay contiguous.
    panels: list[tuple[pd.DataFrame, str, str]] = []
    for _, frame, trait_label in blocks:
        for trunk in trunks:
            name = trunk_labels.get(trunk, f"Trunk {trunk}")
            panels.append(
                (frame, trunk, f"{trait_label}\n{name}" if trait_label else name)
            )
    nrows, ncols = max(1, len(panels)), max(1, len(checkpoints))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(1.75 * ncols + 1.6, 2.0 * nrows + 1.3),
        squeeze=False,
    )
    # Which series any panel managed to draw, for the legend: a series
    # measured on one trunk still needs naming, and asking the whole frame
    # would name one that no panel had a complete column for.
    drawn: dict[str, _DecaySeries] = {}
    # What each panel has to label, held back until the grid is laid out.
    labelled: list[tuple[Axes, list[tuple[str, str, pd.Series, LinearFit]], float]] = []
    for row, (frame, trunk, name) in enumerate(panels):
        for col, t in enumerate(checkpoints):
            ax = axes[row][col]
            panel = frame[(frame["trunk"] == trunk) & (frame["t"] == t)]
            if panel.empty:
                _mark_empty(ax)
                continue
            # The starting level, not zero: on a raw behaviour axis it is the
            # line a point sitting above means the step made the model *more*
            # of the trait, and zero is a judge score no probe goes near.
            ax.axhline(
                float(panel["b_t"].iloc[0]),
                color=style.INK,
                linewidth=1.0,
                zorder=1,
            )
            errors = panel["se_b_next"] if "se_b_next" in panel else None
            probes = list(panel["probe"]) if "probe" in panel else None
            # Resolved before the comprehension below binds the same name to
            # one series at a time.
            wanted = _panel_series(panel, series)
            fits = [
                (
                    series,
                    _scatter_with_fit(
                        ax,
                        panel[series.column],
                        panel["b_next"],
                        color=series.color,
                        label=series.label,
                        yerr=errors,
                        size=34,
                        datasets=probes,
                        mark_edge=series.color,
                    ),
                )
                for series in wanted
            ]
            drawn.update({series.column: series for series, _ in fits})
            # Held rather than printed: each r is placed by measuring the panel
            # it goes in, and the panel is not its final size until the whole
            # grid has been laid out below. A minus sign is printed where there
            # is one; a plus is not, since it costs width in the narrowest
            # panel of the set to say what its absence already says.
            labelled.append(
                (
                    ax,
                    [
                        (
                            rf"$r$ = {fit.corr:.2f}",
                            series.color,
                            panel[series.column],
                            fit,
                        )
                        for series, fit in fits
                    ],
                    float(panel["b_t"].iloc[0]),
                )
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
        axes[row][0].set_ylabel(name, fontsize=9)

    stride = len(trunks) or 1
    _share_blocks(
        [
            [ax for row in axes[start:start + stride] for ax in row]
            for start in range(0, len(panels), stride)
        ],
        x=True,
    )
    for ax in axes.flat:
        ax.set_ylim(*BEHAVIOUR_LIMITS)
        ax.set_yticks(BEHAVIOUR_TICKS)
    _label_outer(axes, bottom_rows=range(stride - 1, len(panels), stride))
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

    # Three keys in one legend: which projection a mark's *outline* means,
    # which dataset its shape and fill mean, and what the black rule is. The
    # projection keys are taken from the whole grid rather than from any one
    # panel, so a series drawn in only some of them still gets named.
    keys = [series for series in _DECAY_SERIES if series.column in drawn]
    handles: list[Artist] = [
        plt.Line2D([], [], color=series.color, linewidth=1.8) for series in keys
    ]
    handles.append(plt.Line2D([], [], color=style.INK, linewidth=1.0))
    texts = [f"{series.label} ({series.gloss})" for series in keys]
    texts.append(r"$b_t$ (level before the step)")
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
    # Last of all: every r is placed by looking at the panel it goes in, which
    # is only now the size and shape it will be read at.
    for ax, entries, level in labelled:
        _label_fits(ax, entries, rules=(level,))
    return fig


def _series_line(
    ax: Axes,
    x: ArrayLike,
    mid: ArrayLike,
    *,
    color: str,
    label: str,
    linestyle: str | tuple = "solid",
) -> None:
    """One trunk-and-series curve, as used by the headline curves.

    Drawn without its bootstrap band: see :func:`headline_curves`.
    """
    ax.plot(
        np.asarray(x, dtype=float),
        np.asarray(mid, dtype=float),
        color=color,
        marker="o",
        markersize=4.5,
        markeredgecolor=style.SURFACE,
        markeredgewidth=0.8,
        linewidth=1.8,
        linestyle=linestyle,
        label=label,
        zorder=3,
    )


#: Line style per series in ``headline_curves``. Keyed by name rather than by
#: position so the same series always reads the same way regardless of which
#: others are present -- adding a series must not restyle the ones beside it.
#: Ordered as the ladder is: more dashes as more of the quantity is
#: approximated, and solid at both ends where nothing is.
_SERIES_LINESTYLES: dict[str, str | tuple] = {
    "p0": "solid",
    "hat_v0": "dashdot",
    "hat_t": (0, (4, 2)),
    "full_t": "dotted",
}


def _correlation_floor(
    frame: pd.DataFrame, columns: Sequence[str], *, clearance: float
) -> float:
    r"""The bottom of a correlation axis: $-1$, or just under zero.

    Fixed rather than fitted to the data, so a panel's height means the same
    thing wherever it appears and a curve that ends high cannot be mistaken for
    one that merely ran out of axis. But fixing it at the full $[-1, 1]$ costs
    half the panel whenever nothing is negative -- which for the RQ1 figures is
    the usual case, and it is the decay *within* the top half that they exist
    to show.

    So: down to $-1$ when some drawn value actually goes below zero, and the
    unit interval when none does. Decided over the whole frame rather than per
    panel, which is what keeps every panel on one scale -- the property that
    matters -- while still letting a negative correlation anywhere widen all of
    them together rather than be clipped out of sight in the one panel that has
    it. ``clearance`` is the margin below the lowest drawable value, so a mark
    sitting exactly at the bound is not cut in half by the axis.
    """
    present = [column for column in columns if column in frame]
    floor = min((frame[column].min() for column in present), default=0.0)
    return -1.0 - clearance if present and floor < 0 else -clearance


def headline_curves(
    fits: pd.DataFrame,
    *,
    traits: Sequence[str] | None = None,
    trait_labels: Mapping[str, str] | None = None,
    series: Sequence[str] = ("p0", "hat_t"),
    series_labels: Mapping[str, str] | None = None,
    trunks: Sequence[str] | None = None,
    trunk_labels: Mapping[str, str] | None = None,
    trunk_colors: Mapping[str, str] | None = None,
    xlabel: str = "Checkpoint $t$",
) -> Figure:
    r"""Plot 3: correlation and fitted slope against ``t``, one panel per trait.

    Two rows (one per quantity) and one column per trait. By default,
    $\Delta P_0$ and $\Delta \hat{P}_t$ are drawn on the *same* axes within a
    panel -- solid vs. dashed, sharing the trunk's colour -- rather than split
    into their own column, so the two series are read directly against each
    other instead of through a side-by-side comparison across panels.

    Slope is reported beside $r$ rather than inside it because staleness has
    two signatures that imply different fixes: a fit that loses its ordering
    (falling $r$) and one that keeps the ordering but shrinks the magnitude
    (falling slope at unchanged $r$).

    Expect a sawtooth in ``t`` and do not smooth it -- it is the phase effect
    the schedules deliberately induce, and :func:`phase_contrast` is what
    separates it from the trend.

    Two measures on two rows rather than two y-scales on one: the alignment of
    a shared axis between $r$ and a slope in trait points per unit
    $\Delta P$ would be arbitrary, and would invent a relationship between
    them. For the same reason the slope row's panels do not share a y-scale
    across traits -- the slope is in points of *that* trait's judge per unit
    of *that* trait's persona vector, so its absolute value is not comparable
    between them, while its trend in ``t`` is exactly what the figure asks the
    reader to compare. (The correlation row is unitless in both traits, so its
    panels share a fixed range regardless -- see :func:`_correlation_ylim` for
    which one.)

    Nor does it draw the bootstrap intervals ``fit_frame`` computes. Eight
    probe datasets per checkpoint make them wide, and one band per
    (trunk, series) curve -- four of them per panel, overlapping -- buries the
    curves the figure exists to show under the uncertainty about them. The
    intervals are still computed and still in the frame; the honest place for
    a sample this small is a sentence in the text saying so, not a wash of
    shading over every panel.

    This panel no longer draws the $R^2_{max}$ noise ceiling either; see
    ``docs/r2_max.md`` for what it means and where to find it instead. Note it
    is a variance ratio, so the ceiling on the correlation drawn here is its
    square root, not the stored number.
    """
    style.apply_style()
    series = list(series)
    series_labels = series_labels or {}
    trunk_labels = trunk_labels or {}
    trunk_colors = trunk_colors or {}
    trunks = list(trunks) if trunks else sorted(fits["trunk"].unique())
    linestyles = _SERIES_LINESTYLES
    quantities = (
        ("corr", r"Correlation $r$ over the probe set"),
        ("slope", r"Fitted slope ($\Delta b$ per unit $\Delta P$)"),
    )
    cols = _facets(fits, traits, trait_labels, column="trait")

    fig, axes = plt.subplots(
        len(quantities),
        len(cols),
        figsize=(4.4 * len(cols), 2.9 * len(quantities) + 0.6),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    # One floor for the whole correlation row, so its panels stay comparable.
    corr_floor = _correlation_floor(
        fits, [f"corr_{name}" for name in series], clearance=0.03
    )
    for row, (quantity, qlabel) in enumerate(quantities):
        for col, (_, frame, _trait_label) in enumerate(cols):
            ax = axes[row][col]
            for trunk in trunks:
                arm = frame[frame["trunk"] == trunk].sort_values("t")
                if arm.empty:
                    continue
                color = trunk_colors.get(trunk, style.BLUE)
                for name in series:
                    _series_line(
                        ax,
                        arm["t"],
                        arm[f"{quantity}_{name}"],
                        color=color,
                        label="_nolegend_",
                        linestyle=linestyles.get(name, "solid"),
                    )
            if quantity == "corr":
                ax.set_ylim(corr_floor, 1.03)
            # Zero is a meaningful level for a slope always, and for a
            # correlation only once the axis is wide enough to have a sign to
            # read; on a [0, 1] panel the rule would just underline the frame.
            if quantity == "slope" or corr_floor < -1.0:
                ax.axhline(0, color=style.BASELINE, linewidth=0.8, zorder=1)
        axes[row][0].set_ylabel(qlabel)
    for col, (_, _, trait_label) in enumerate(cols):
        axes[0][col].set_title(trait_label, color=style.SECONDARY_INK)
        # Only the bottom row: every column shares one x-axis, so a label under
        # any other row would name ticks that are not drawn.
        axes[-1][col].set_xlabel(xlabel)

    # Two channels, two legend keys: colour names the trunk, line style names
    # the series. A per-line label (as _shared_legend expects) would need one
    # entry per (trunk, series) combination instead of the sum of the two.
    handles: list[Artist] = []
    texts: list[str] = []
    for trunk in trunks:
        handles.append(
            plt.Line2D(
                [], [], color=trunk_colors.get(trunk, style.BLUE), linewidth=1.8
            )
        )
        texts.append(trunk_labels.get(trunk, f"Trunk {trunk}"))
    for name in series:
        handles.append(
            plt.Line2D(
                [],
                [],
                color=style.SECONDARY_INK,
                linewidth=1.8,
                linestyle=linestyles.get(name, "solid"),
            )
        )
        texts.append(series_labels.get(name, name))
    ncol = min(4, len(handles) or 1)
    fig.legend(handles, texts, loc="lower center", ncol=ncol)
    _layout_grid(fig, axes.flat, legend_rows=-(-len(handles) // ncol))
    return fig


def mechanism_grid(
    checkpoints: pd.DataFrame,
    predictors: Mapping[str, str],
    *,
    traits: Sequence[str] | None = None,
    trait_labels: Mapping[str, str] | None = None,
    value_column: str = "corr_p0",
    trunk_labels: Mapping[str, str] | None = None,
    trunk_colors: Mapping[str, str] | None = None,
    ylabel: str = r"Correlation $r$ of $\Delta P_0$ at that checkpoint",
) -> Figure:
    r"""Plot 4: the checkpoint-level regression of the correlation on what moved.

    ``predictors`` maps a column to its display label -- the drift components
    $\rho$ and $r$, the current behaviour level $b_t$, and
    ``steps_since_realignment``. A column fits $r$ against one of them over
    every checkpoint of one trait, a row is a trait, and colour identifies
    which trunk a checkpoint came from.

    This is the level-2 analysis, so a point is a *checkpoint*, not a probe
    dataset: the eight probes at a checkpoint were already spent producing the
    single $r$ plotted here. Dense sampling is what makes the panel
    populated at all -- measuring every ``t`` gives 19 rows where measuring
    ``t`` in ``{0, 2, 4, 6}`` would give 10 -- and the varied schedules are
    what stop drift and behaviour level from moving together, which is the
    condition for their contributions to be separable.

    Every panel shares one y-axis, unlike the other grids here: the
    correlation is unitless and bounded, so it is the one quantity in this set
    that means the same thing for both traits. The x-axes are not shared, in either direction
    -- the columns are different quantities, and a rotation that runs over
    $[0.88, 1]$ for one trait and $[0.97, 1]$ for the other would have the
    tighter trait's whole spread squeezed into a corner.
    """
    style.apply_style()
    trunk_labels = trunk_labels or {}
    trunk_colors = trunk_colors or {}
    rows = _facets(checkpoints, traits, trait_labels, column="trait")
    ncols = max(1, len(predictors))
    fig, axes = plt.subplots(
        len(rows),
        ncols,
        figsize=(3.3 * ncols + 1.0, 3.0 * len(rows) + 1.0),
        sharey=True,
        squeeze=False,
    )

    for row, (_, frame, trait_label) in enumerate(rows):
        for col, (column, label) in enumerate(predictors.items()):
            ax = axes[row][col]
            if row == 0:
                ax.set_title(label, fontsize=9.5, color=style.SECONDARY_INK)
            if frame.empty:
                _mark_empty(ax)
                continue
            fit = linear_fit(frame[column], frame[value_column])
            if len(frame) >= 2:
                line_x = _fit_line_x(np.asarray(frame[column], dtype=float))
                ax.plot(
                    line_x,
                    fit.predict(line_x),
                    color=style.SECONDARY_INK,
                    linewidth=1.5,
                    zorder=2,
                )
            for key, group in frame.groupby("trunk", sort=True):
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
            entries = [(rf"$r$ = {fit.corr:.2f}", style.INK)]
            if col == 0:
                # Every panel of a row regresses the same checkpoints, so ``n``
                # belongs to the row rather than to a panel -- stated once, in
                # the one the eye reaches first, since it is what stops the
                # level-2 count from being read as the level-1 one (a point
                # here is a checkpoint, not a probe dataset). Once per row and
                # not once per figure, because each trait has its own ``n``.
                entries.append(
                    (rf"$n$ = {len(frame)} checkpoints", style.SECONDARY_INK)
                )
            _corner_text(ax, entries, fontsize=9)
        axes[row][0].set_ylabel(trait_label, fontsize=10)

    handles, texts = _shared_legend(axes.flat)
    ncol = min(4, len(handles) or 1)
    fig.legend(handles, texts, loc="lower center", ncol=ncol)
    _layout_grid(
        fig, axes.flat, ylabel=ylabel, legend_rows=-(-len(handles) // ncol)
    )
    return fig


#: Hatch per series in ``phase_contrast``. Colour already carries the trunk and
#: fill carries before/after, so the series needs a third channel that survives
#: both. Keyed by name for the same reason the line styles are.
_SERIES_HATCHES: dict[str, str] = {
    "p0": "",
    "hat_v0": "....",
    "hat_t": "////",
    "full_t": "xxxx",
}

#: Bar geometry per re-alignment step, in x-axis units. Two bars per series
#: (before, after), and the gap *within* a pair is narrower than the gap
#: between pairs, so the eye groups before/after together and reads each
#: series as its own cluster. :data:`_BAR_SPAN` is what the whole group is
#: then scaled to occupy, which is what keeps a third series from colliding
#: with the neighbouring step's bars.
_PAIR_GAP = 0.20
_SERIES_GAP = 0.24
_BAR_SPAN = 0.82


def _bar_layout(n_series: int) -> tuple[list[float], float]:
    """Centred bar offsets and bar width for ``n_series`` before/after pairs.

    Derived rather than tabulated so the figure keeps working when a series is
    added: the ratios are fixed and only the scale adapts, so two series lay
    out exactly as they always have and three simply draw narrower.
    """
    width = 0.18
    offsets = [
        pair * _PAIR_GAP + index * (_PAIR_GAP + _SERIES_GAP)
        for index in range(max(n_series, 1))
        for pair in (0, 1)
    ]
    centre = (offsets[0] + offsets[-1]) / 2
    offsets = [offset - centre for offset in offsets]
    scale = _BAR_SPAN / (offsets[-1] - offsets[0] + width)
    return [offset * scale for offset in offsets], width * scale


def phase_contrast(
    pairs: pd.DataFrame,
    *,
    traits: Sequence[str] | None = None,
    trait_labels: Mapping[str, str] | None = None,
    series: Sequence[str] = ("p0", "hat_t"),
    series_labels: Mapping[str, str] | None = None,
    trunk_colors: Mapping[str, str] | None = None,
    ylabel: str = r"Correlation $r$ over the probe set",
) -> Figure:
    r"""Plot 4b: what one re-alignment step does to predictive accuracy.

    Each pair of checkpoints straddles a single Normal driver, with the trunk
    and the probe set held fixed, so the vertical distance between a step's
    bars is attributable to that one step -- unlike a difference read off the
    trend in ``t``, which also carries everything else that accumulated in
    between.

    Drawn as adjacent before/after bars rather than a bar of the difference so
    that the *level* stays visible: a drop from 0.9 to 0.6 and one from 0.4 to
    0.1 are the same difference and very different findings. $\Delta P_0$ and
    $\Delta \hat{P}_t$ sit side by side within the same step -- offset just enough
    not to touch -- rather than in their own panel, so the control comparison
    (does $\Delta \hat{P}_t$ move by the same amount?) is a glance sideways instead
    of a glance across the figure.

    One column per trait, on one shared pair of axes: $r$ is unitless and
    every column is read against the same list of re-alignment steps, so a
    step whose fit collapses for one trait and holds for the other is a
    difference the grid shows directly. The x positions come from the whole
    frame rather than from each column, so a pair that only one trait has
    measured leaves a gap instead of shifting that column's steps out of line
    with the other's.
    """
    style.apply_style()
    series = list(series)
    series_labels = series_labels or {}
    trunk_colors = trunk_colors or {}
    hatches = _SERIES_HATCHES
    offsets, bar_width = _bar_layout(len(series))
    ordered = list(
        pairs.sort_values(["trunk", "t_before"])["pair"].drop_duplicates()
    )
    position = {pair: i for i, pair in enumerate(ordered)}
    cols = _facets(pairs, traits, trait_labels, column="trait")
    x = np.arange(len(ordered))
    bar_floor = _correlation_floor(
        pairs,
        [f"corr_{name}_{when}" for name in series for when in ("before", "after")],
        clearance=0.05,
    )

    fig, axes = plt.subplots(
        1,
        len(cols),
        figsize=(3.6 * len(cols) + 1.2, 3.6),
        sharey=True,
        squeeze=False,
    )
    for col, (_, frame, trait_label) in enumerate(cols):
        ax = axes[0][col]
        for record in frame.to_dict("records"):
            i = position[record["pair"]]
            color = trunk_colors.get(record["trunk"], style.BLUE)
            for s_idx, name in enumerate(series):
                before = record[f"corr_{name}_before"]
                after = record[f"corr_{name}_after"]
                if not (np.isfinite(before) and np.isfinite(after)):
                    # A series this trunk was never re-measured on. Undrawn,
                    # so the gap reads as "not measured" rather than as a pair
                    # of zero-height bars reading as "no change".
                    continue
                before_x = i + offsets[2 * s_idx]
                after_x = i + offsets[2 * s_idx + 1]
                ax.bar(
                    before_x, before, width=bar_width, facecolor=style.SURFACE,
                    edgecolor=color, linewidth=1.4, hatch=hatches.get(name, ""),
                    zorder=3,
                )
                ax.bar(
                    after_x, after, width=bar_width, facecolor=color,
                    edgecolor=color, linewidth=0.8, hatch=hatches.get(name, ""),
                    zorder=3,
                )
                ax.annotate(
                    f"{after - before:+.2f}",
                    ((before_x + after_x) / 2, max(before, after)),
                    textcoords="offset points",
                    xytext=(0, 4),
                    ha="center",
                    fontsize=7.5,
                    color=style.SECONDARY_INK,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(ordered, rotation=20, ha="right")
        # Room for the delta labels on the outermost pairs, which sit above a
        # bar at the very edge of the data range.
        ax.set_xlim(-0.7, len(ordered) - 0.3)
        ax.set_ylim(bar_floor, 1.15)
        ax.set_title(trait_label, color=style.SECONDARY_INK)

    # Two keys: fill says before/after, hatch says which series.
    handles = [
        Patch(facecolor=style.SURFACE, edgecolor=style.SECONDARY_INK, linewidth=1.4),
        Patch(facecolor=style.SECONDARY_INK, edgecolor=style.SECONDARY_INK),
        *[
            Patch(
                facecolor=style.SURFACE,
                edgecolor=style.SECONDARY_INK,
                hatch=hatches.get(name, ""),
            )
            for name in series
        ],
    ]
    texts = [
        "Before the Normal driver",
        "After it",
        *[series_labels.get(name, name) for name in series],
    ]
    ncol = min(4, len(handles))
    fig.legend(handles, texts, loc="lower center", ncol=ncol)
    _layout_grid(fig, axes.flat, ylabel=ylabel, legend_rows=-(-len(handles) // ncol))
    return fig


def _draw_overlay(
    ax: Axes,
    primary: Mapping[str, Sequence[float]],
    replicate: Mapping[str, Sequence[Sequence[float]]],
    band: Mapping[str, Sequence[float]],
    *,
    colors: Mapping[str, str],
    marks: Mapping[str, style.DatasetMark] | None,
    reference: float | None,
    reference_label: str | None,
) -> None:
    """One panel of an overlay plot: solid lines, optional bands/replicates.

    ``replicate`` holds *every* replicate of a series, not one, so a family run
    at several seeds draws a dashed line each rather than one line through
    their mean.

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

    order = {label: i for i, label in enumerate(primary)}
    for label, values in primary.items():
        y = np.asarray(values, dtype=float)
        color, marker_style = styling(label, order[label])
        if label in band:
            spread = np.asarray(band[label], dtype=float)
            ax.fill_between(
                np.arange(y.size),
                y - spread,
                y + spread,
                color=color,
                alpha=0.18,
                linewidth=0,
                zorder=2,
            )
        ax.plot(
            np.arange(y.size),
            y,
            color=color,
            linewidth=1.6,
            label=label,
            zorder=3,
            **marker_style,
        )
    # Indexed by the primary's position rather than the replicate's own, so a
    # replicate keeps the colour of the series it replicates even when the two
    # sets of keys differ -- a probe dropped from one for a near-zero baseline,
    # or a probe-free reseed contributing to the latent panels only.
    for i, (label, runs) in enumerate(replicate.items(), start=len(order)):
        color, _ = styling(label, order.get(label, i))
        for values in runs:
            y = np.asarray(values, dtype=float)
            ax.plot(
                np.arange(y.size),
                y,
                color=color,
                linestyle=(0, (3, 2)),
                linewidth=1.3,
                alpha=0.85,
                zorder=2,
            )


def _shared_legend(axes: Iterable[Axes]) -> tuple[list[Artist], list[str]]:
    """One legend entry per distinct series label across a grid's panels.

    Panels of the same grid draw the same series, so taking the key from the
    first panel alone would drop anything that panel happens to be missing --
    a probe dropped from one trunk for a near-zero baseline, say. Deduplicating
    by label keeps the key complete without repeating an entry per panel.
    """
    handles: list[Artist] = []
    texts: list[str] = []
    for ax in axes:
        for handle, text in zip(*ax.get_legend_handles_labels()):
            if text not in texts:
                handles.append(handle)
                texts.append(text)
    return handles, texts


def overlay_grid(
    panels: Mapping[tuple[str, str], Mapping[str, Sequence[float]]],
    replicates: (
        Mapping[tuple[str, str], Mapping[str, Sequence[Sequence[float]]]] | None
    ) = None,
    *,
    bands: Mapping[tuple[str, str], Mapping[str, Sequence[float]]] | None = None,
    rows: Sequence[str] | None = None,
    cols: Sequence[str] | None = None,
    ylabel: str | None = None,
    xlabel: str = "Checkpoint $t$",
    colors: Mapping[str, str] | None = None,
    marks: Mapping[str, style.DatasetMark] | None = None,
    reference: float | None = None,
    reference_label: str | None = None,
    replicate_label: str = "Reseeded replicate (dashed)",
    band_label: str = r"Mean $\pm$ 1 SD",
    sharey: bool = False,
) -> Figure:
    r"""Plot 5: how a quantity drifts over ``t``, panelled by row and column.

    ``panels`` maps a ``(row, col)`` pair of display labels to that panel's
    series. ``bands`` maps those labels to a symmetric spread around each
    line, while ``replicates`` maps them to individual reseeded runs. A panel's
    halves need not carry the same labels: probe-free seeds reach the latent
    columns and not the $\Delta \hat{P}_t$ ones.

    Replicates are drawn dashed in the colour of the series they replicate, so
    the comparison rides one colour assignment however many seeds land and
    costs no extra hue. Bands use the matching series colour and describe a
    caller-provided spread around its solid line. In the latent grid that is
    the between-seed SD, not measurement error; every seed reads the same
    cached $t = 0$ anchor, so the band does not include the common-mode anchor
    term that :mod:`method.anchor_noise` estimates separately.

    ``marks`` identifies each series as a dataset rather than by a hue of its
    own. Eight probes would otherwise take all eight categorical slots, leaving
    nothing for anything else and making the reader learn an arbitrary
    dataset-to-colour map; shape-per-family plus the version ramp is
    self-describing and leaves the hues free.

    ``sharey`` is for a grid whose panels are one quantity in one unit -- the
    $\Delta \hat{P}_t$ percentages, where the point is that the traits and trunks
    drift by different amounts, and unshared axes would rescale that difference
    away. The latent grid must leave it off: its columns are different
    quantities ($\rho$ starts at 1, $r$ at the persona vector's norm), and even
    within a column the two traits have their own vectors, so a shared axis
    would squash one trait's drift into the gap between the two norms.

    No measurement-error bars: section 8 establishes that $\Delta \hat{P}_t$ and
    $z_t$ involve no sampling -- fixed prompts, fixed responses, forward passes
    only. A seed-spread band instead quantifies training-run variation.
    """
    style.apply_style()
    replicates = replicates or {}
    bands = bands or {}
    rows = list(rows) if rows else list(dict.fromkeys(row for row, _ in panels))
    cols = list(cols) if cols else list(dict.fromkeys(col for _, col in panels))

    fig, axes = plt.subplots(
        len(rows),
        len(cols),
        figsize=(2.9 * len(cols) + 1.2, 2.6 * len(rows) + 1.2),
        sharex=True,
        sharey=sharey,
        squeeze=False,
    )
    for r, row in enumerate(rows):
        for c, col in enumerate(cols):
            ax = axes[r][c]
            series = panels.get((row, col)) or {}
            if not series:
                _mark_empty(ax)
                continue
            _draw_overlay(
                ax,
                series,
                replicates.get((row, col)) or {},
                bands.get((row, col)) or {},
                colors=colors or {},
                marks=marks,
                reference=reference,
                reference_label=reference_label,
            )
        # The row's identity, in place of a y-label naming the quantity: that
        # is what the column header or the shared label carries here.
        axes[r][0].set_ylabel(row, fontsize=10)
    for c, col in enumerate(cols):
        axes[0][c].set_title(col, fontsize=9.5, color=style.SECONDARY_INK)

    handles, texts = _shared_legend(axes.flat)
    if any(replicates.values()):
        handles.append(
            plt.Line2D([], [], color=style.MUTED, linestyle=(0, (3, 2)), linewidth=1.3)
        )
        texts.append(replicate_label)
    if any(bands.values()):
        handles.append(Patch(facecolor=style.MUTED, edgecolor="none", alpha=0.18))
        texts.append(band_label)
    ncol = min(4, len(handles) or 1)
    fig.legend(handles, texts, loc="lower center", ncol=ncol)
    _layout_grid(
        fig,
        axes.flat,
        xlabel=xlabel,
        ylabel=ylabel,
        legend_rows=-(-len(handles) // ncol),
    )
    return fig


def diversity_bar(
    df: pd.DataFrame,
    *,
    rows: Sequence[str] | None = None,
    row_col: str = "trait",
    row_labels: Mapping[str, str] | None = None,
    cols: Sequence[str] | None = None,
    col_col: str = "realign_trait",
    col_labels: Mapping[str, str] | None = None,
    condition_col: str = "condition",
    value_col: str = "delta_behavior",
    order: Sequence[str] | None = None,
    labels: Mapping[str, str] | None = None,
    ylabel: str = r"Behaviour change $\Delta b$",
    color: str = style.BLUE,
) -> Figure:
    """RQ2 diversity bar chart: does dataset diversity hinder re-alignment?

    One panel per measured trait (row) and re-alignment source (column), which
    is the whole design in one figure: the conditions differ only in how many
    datasets the re-alignment mixed and whether they were the same one twice,
    so the effect of diversity is the shape of a panel, and whether it
    generalises is whether that shape repeats across them.

    A single colour for every bar -- the conditions are unordered categories
    identified by their x-tick label, not a second grouping dimension, so a
    rainbow or a value ramp would only burn the colour channel on information
    the chart already shows (see the data-viz "value-ramp on nominal
    categories" anti-pattern). The rows keep their own y-scales, since a
    residual is in points of that trait's own judge.
    """
    style.apply_style()
    order = list(order) if order is not None else list(dict.fromkeys(df[condition_col]))
    labels = labels or {}
    tick_labels = [labels.get(c, c) for c in order]
    panel_rows = _facets(df, rows, row_labels, column=row_col)
    panel_cols = _facets(df, cols, col_labels, column=col_col)

    x = np.arange(len(order))
    fig, axes = plt.subplots(
        len(panel_rows),
        len(panel_cols),
        figsize=(
            max(4.0, 1.3 * len(order) + 1.5) * len(panel_cols),
            2.9 * len(panel_rows) + 1.3,
        ),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for r, (_, frame, row_label) in enumerate(panel_rows):
        for c, (key, _, col_label) in enumerate(panel_cols):
            ax = axes[r][c]
            if r == 0:
                ax.set_title(col_label)
            panel = _facet_frame(frame, col_col, key)
            if panel.empty:
                _mark_empty(ax)
                continue
            stats = (
                panel.groupby(condition_col)[value_col]
                .agg(["mean", "std"])
                .reindex(order)
            )
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
        axes[r][0].set_ylabel(row_label, fontsize=10)

    _layout_grid(fig, axes.flat, ylabel=ylabel, legend_rows=0)
    return fig
