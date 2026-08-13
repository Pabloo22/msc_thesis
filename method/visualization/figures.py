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
    _corner_text(ax, [("not run", style.MUTED)], x=0.5, y=0.55, ha="center")


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
    over the 8 probes, and $R^2$ estimates are sensitive enough to range
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
                (rf"$R^2$ = {fit.r2:.2f}   slope = {fit.slope:.2f}", style.INK),
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


def decay_scatter_grid(
    df: pd.DataFrame,
    *,
    traits: Sequence[str] | None = None,
    trait_labels: Mapping[str, str] | None = None,
    trunks: Sequence[str] | None = None,
    checkpoints: Sequence[int] | None = None,
    trunk_labels: Mapping[str, str] | None = None,
    xlabel: str = r"Projection difference $\Delta P$",
    ylabel: str = r"Behaviour change $\Delta b_{t+1}$",
) -> Figure:
    r"""Plot 2: one scatter panel per ``(trait, trunk, checkpoint)``.

    Rows are trunks and columns checkpoints, so a row reads as one trajectory
    ageing and a column as three trajectories at the same depth; the traits
    stack as blocks of rows. Each panel holds the ``K`` probe datasets twice
    over: against the frozen $\Delta P_0$ in blue and against the recomputed
    $\Delta P_t$ in orange, with both fits' $R^2$ annotated. The hypothesis is
    visible as the blue fit flattening left-to-right while the orange one does
    not.

    Axes are shared within a trait and not across them (see
    :func:`_share_blocks`). Within one, panels on their own scales would let a
    flattening slope and a shrinking $\Delta b$ range look identical, which is
    the one confusion this figure exists to prevent; across two, $\Delta P$ and
    $\Delta b$ are read against different persona vectors and different judges,
    so a shared scale would compare quantities that are not the same quantity.

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
    for row, (frame, trunk, name) in enumerate(panels):
        for col, t in enumerate(checkpoints):
            ax = axes[row][col]
            panel = frame[(frame["trunk"] == trunk) & (frame["t"] == t)]
            if panel.empty:
                _mark_empty(ax)
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
        axes[row][0].set_ylabel(name, fontsize=9)

    stride = len(trunks) or 1
    _share_blocks(
        [
            [ax for row in axes[start:start + stride] for ax in row]
            for start in range(0, len(panels), stride)
        ],
        x=True,
        y=True,
    )
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
    linestyle: str | tuple = "solid",
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
        linestyle=linestyle,
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


#: Line style per series position in ``headline_curves``: $\Delta P_0$ solid,
#: $\Delta P_t$ dashed. Fixed rather than assigned dynamically so the same
#: series always reads the same way regardless of which trunks are present.
_SERIES_LINESTYLES: tuple[str | tuple, ...] = ("solid", (0, (4, 2)))


def headline_curves(
    fits: pd.DataFrame,
    *,
    traits: Sequence[str] | None = None,
    trait_labels: Mapping[str, str] | None = None,
    series: Sequence[str] = ("p0", "pt"),
    series_labels: Mapping[str, str] | None = None,
    trunks: Sequence[str] | None = None,
    trunk_labels: Mapping[str, str] | None = None,
    trunk_colors: Mapping[str, str] | None = None,
    xlabel: str = "Checkpoint $t$",
) -> Figure:
    r"""Plot 3: $R^2$ and fitted slope against ``t``, one panel per trait.

    Two rows (one per quantity) and one column per trait. $\Delta P_0$ and
    $\Delta P_t$ are drawn on the *same* axes within a panel -- solid vs.
    dashed, sharing the trunk's colour -- rather than split into their own
    column, so the two series are read directly against each other instead of
    through a side-by-side comparison across panels.

    Slope is reported beside $R^2$ rather than inside it because staleness has
    two signatures that imply different fixes: a fit that loses its ordering
    (falling $R^2$) and one that keeps the ordering but shrinks the magnitude
    (falling slope at unchanged $R^2$).

    Expect a sawtooth in ``t`` and do not smooth it -- it is the phase effect
    the schedules deliberately induce, and :func:`phase_contrast` is what
    separates it from the trend.

    Two measures on two rows rather than two y-scales on one: the alignment of
    a shared axis between $R^2$ and a slope in trait points per unit
    $\Delta P$ would be arbitrary, and would invent a relationship between
    them. For the same reason the slope row's panels do not share a y-scale
    across traits -- the slope is in points of *that* trait's judge per unit
    of *that* trait's persona vector, so its absolute value is not comparable
    between them, while its trend in ``t`` is exactly what the figure asks the
    reader to compare. (The $R^2$ row is unitless in both traits, so its
    panels share a fixed ``[0, 1]``-ish range regardless.)

    This panel no longer draws the $R^2_{max}$ noise ceiling; see
    ``docs/r2_max.md`` for what it means and where to find it instead.
    """
    style.apply_style()
    series = list(series)
    series_labels = series_labels or {}
    trunk_labels = trunk_labels or {}
    trunk_colors = trunk_colors or {}
    trunks = list(trunks) if trunks else sorted(fits["trunk"].unique())
    linestyles = dict(zip(series, _SERIES_LINESTYLES))
    quantities = (
        ("r2", r"$R^2$ over the probe set"),
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
    for row, (quantity, qlabel) in enumerate(quantities):
        for col, (_, frame, _trait_label) in enumerate(cols):
            ax = axes[row][col]
            for trunk in trunks:
                arm = frame[frame["trunk"] == trunk].sort_values("t")
                if arm.empty:
                    continue
                color = trunk_colors.get(trunk, style.BLUE)
                for name in series:
                    _band(
                        ax,
                        arm["t"],
                        arm[f"{quantity}_{name}"],
                        arm[f"{quantity}_{name}_lo"],
                        arm[f"{quantity}_{name}_hi"],
                        color=color,
                        label="_nolegend_",
                        linestyle=linestyles.get(name, "solid"),
                    )
            if quantity == "r2":
                ax.set_ylim(-0.03, 1.03)
            else:
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
    value_column: str = "r2_p0",
    trunk_labels: Mapping[str, str] | None = None,
    trunk_colors: Mapping[str, str] | None = None,
    ylabel: str = r"$R^2$ of $\Delta P_0$ at that checkpoint",
) -> Figure:
    r"""Plot 4: the checkpoint-level regression of $R^2$ on what moved.

    ``predictors`` maps a column to its display label -- the drift components
    $\rho$ and $r$, the current behaviour level $b_t$, and
    ``steps_since_realignment``. A column fits $R^2$ against one of them over
    every checkpoint of one trait, a row is a trait, and colour identifies
    which trunk a checkpoint came from.

    This is the level-2 analysis, so a point is a *checkpoint*, not a probe
    dataset: the eight probes at a checkpoint were already spent producing the
    single $R^2$ plotted here. Dense sampling is what makes the panel
    populated at all -- measuring every ``t`` gives 19 rows where measuring
    ``t`` in ``{0, 2, 4, 6}`` would give 10 -- and the varied schedules are
    what stop drift and behaviour level from moving together, which is the
    condition for their contributions to be separable.

    Every panel shares one y-axis, unlike the other grids here: $R^2$ is
    unitless and bounded, so it is the one quantity in this set that means the
    same thing for both traits. The x-axes are not shared, in either direction
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
            entries = [(rf"$R^2$ = {fit.r2:.2f}", style.INK)]
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


#: Hatch per series position in ``phase_contrast``: $\Delta P_0$ plain,
#: $\Delta P_t$ hatched. Colour already carries the trunk and fill carries
#: before/after, so the series needs a third channel that survives both.
_SERIES_HATCHES: tuple[str, ...] = ("", "////")

#: Offsets (in x-axis units) of the four bars drawn per re-alignment step:
#: p0-before, p0-after, pt-before, pt-after. The gap within a series pair is
#: narrower than the gap between the two series, so the eye groups
#: before/after together and reads the two series as separate clusters.
_BAR_OFFSETS: tuple[float, ...] = (-0.32, -0.12, 0.12, 0.32)
_BAR_WIDTH = 0.18


def phase_contrast(
    pairs: pd.DataFrame,
    *,
    traits: Sequence[str] | None = None,
    trait_labels: Mapping[str, str] | None = None,
    series: Sequence[str] = ("p0", "pt"),
    series_labels: Mapping[str, str] | None = None,
    trunk_colors: Mapping[str, str] | None = None,
    ylabel: str = r"$R^2$ over the probe set",
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
    $\Delta P_t$ sit side by side within the same step -- offset just enough
    not to touch -- rather than in their own panel, so the control comparison
    (does $\Delta P_t$ move by the same amount?) is a glance sideways instead
    of a glance across the figure.

    One column per trait, on one shared pair of axes: $R^2$ is unitless and
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
    hatches = dict(zip(series, _SERIES_HATCHES))
    ordered = list(
        pairs.sort_values(["trunk", "t_before"])["pair"].drop_duplicates()
    )
    position = {pair: i for i, pair in enumerate(ordered)}
    cols = _facets(pairs, traits, trait_labels, column="trait")
    x = np.arange(len(ordered))

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
                before = record[f"r2_{name}_before"]
                after = record[f"r2_{name}_after"]
                before_x = i + _BAR_OFFSETS[2 * s_idx]
                after_x = i + _BAR_OFFSETS[2 * s_idx + 1]
                ax.bar(
                    before_x, before, width=_BAR_WIDTH, facecolor=style.SURFACE,
                    edgecolor=color, linewidth=1.4, hatch=hatches.get(name, ""),
                    zorder=3,
                )
                ax.bar(
                    after_x, after, width=_BAR_WIDTH, facecolor=color,
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
        ax.set_ylim(-0.05, 1.15)
        ax.set_title(trait_label, color=style.SECONDARY_INK)

    handles = [
        Patch(facecolor=style.SURFACE, edgecolor=style.SECONDARY_INK, linewidth=1.4),
        Patch(facecolor=style.SECONDARY_INK, edgecolor=style.SECONDARY_INK),
        Patch(facecolor=style.SURFACE, edgecolor=style.SECONDARY_INK, hatch=""),
        Patch(
            facecolor=style.SURFACE, edgecolor=style.SECONDARY_INK,
            hatch=hatches.get(series[-1], "////"),
        ),
    ]
    texts = [
        "Before the Normal driver",
        "After it",
        series_labels.get(series[0], series[0]),
        series_labels.get(series[-1], series[-1]),
    ]
    fig.legend(handles, texts, loc="lower center", ncol=4)
    _layout_grid(fig, axes.flat, ylabel=ylabel, legend_rows=1)
    return fig


def _draw_overlay(
    ax: Axes,
    primary: Mapping[str, Sequence[float]],
    replicate: Mapping[str, Sequence[Sequence[float]]],
    *,
    colors: Mapping[str, str],
    marks: Mapping[str, style.DatasetMark] | None,
    reference: float | None,
    reference_label: str | None,
) -> None:
    """One panel of an overlay plot: solid primary lines, dashed replicates.

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
    rows: Sequence[str] | None = None,
    cols: Sequence[str] | None = None,
    ylabel: str | None = None,
    xlabel: str = "Checkpoint $t$",
    colors: Mapping[str, str] | None = None,
    marks: Mapping[str, style.DatasetMark] | None = None,
    reference: float | None = None,
    reference_label: str | None = None,
    replicate_label: str = "Reseeded replicate (dashed)",
    sharey: bool = False,
) -> Figure:
    r"""Plot 5: how a quantity drifts over ``t``, panelled by row and column.

    ``panels`` maps a ``(row, col)`` pair of display labels to that panel's
    series, and ``replicates`` the same for the reseeded runs -- mapping a
    series' label to *every* replicate of it, since section 6c is run at one
    probed seed and several probe-free ones
    (:data:`~method.experiments.EXP2_RESEED_EXTRA_SEEDS`). A panel's two halves
    therefore need not carry the same labels: the probe-free seeds reach the
    latent columns and not the $\Delta P_t$ ones.

    Replicates are drawn dashed in the colour of the series they replicate, so
    the comparison rides one colour assignment however many seeds land and
    costs no extra hue. Still no shaded band: the seeds vary the fine-tuning
    only, and every one of them reads the same cached $t = 0$ anchor
    (``weights_key`` normalises the seed away there), so a band drawn from
    their spread would look like the measurement's error while omitting the
    common-mode anchor term that :mod:`method.anchor_noise` estimates
    separately. The overlay says what the spread is and leaves combining the
    two to the write-up.

    ``marks`` identifies each series as a dataset rather than by a hue of its
    own. Eight probes would otherwise take all eight categorical slots, leaving
    nothing for anything else and making the reader learn an arbitrary
    dataset-to-colour map; shape-per-family plus the version ramp is
    self-describing and leaves the hues free.

    ``sharey`` is for a grid whose panels are one quantity in one unit -- the
    $\Delta P_t$ percentages, where the point is that the traits and trunks
    drift by different amounts, and unshared axes would rescale that difference
    away. The latent grid must leave it off: its columns are different
    quantities ($\rho$ starts at 1, $r$ at the persona vector's norm), and even
    within a column the two traits have their own vectors, so a shared axis
    would squash one trait's drift into the gap between the two norms.

    No error bars: section 8 establishes that $\Delta P_t$ and $z_t$ involve no
    sampling -- fixed prompts, fixed responses, forward passes only -- so the
    quantities on this axis have no measurement error to draw.
    """
    style.apply_style()
    replicates = replicates or {}
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
