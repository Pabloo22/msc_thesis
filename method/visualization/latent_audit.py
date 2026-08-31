r"""What happened to $z_t$ in a seed-swept family: an audit, then a drift picture.

    poetry run python -m method.visualization.latent_audit --group exp3

Two questions, in this order, because the second is only readable once the
first is settled.

**Is $z_t$ measured against one anchor?** Three of the four components are
defined relative to the base model's persona vector $v_0$ -- $p_t$ and $\rho_t$
read against it directly, and $q_t$ is only interpretable as a departure from
$p_0 = q_0$. $v_0$ is a *measurement* of $M_0$, not a property of it: it is
extracted from sampled generations, so re-deriving it gives a slightly
different vector. The same is true of ``h_neutral_base``, the fixed text $M_t$
re-reads at every checkpoint, which is $M_0$'s sampled answers to the neutral
prompts. Both are cached in the store under the base ``weights_id`` and so are
normally derived once and shared by every run -- but only for as long as every
run agrees on what that id is and finds the artifacts already there. When they
do not, runs end up on different anchors, and a level of $p$ or $\rho$ from one
run is not comparable with the same level from another.

The audit answers that structurally rather than by trusting the pipeline: at
$t = 0$ no fine-tuning has happened, so every run of a trait is measuring the
*same weights*, and any disagreement in $z_0$ is measurement, not model. The
same argument extends to $t > 0$ wherever content addressing gives two runs one
checkpoint. :func:`disagreement` collects every such case and
:func:`noise_vs_drift` puts it beside the drift it would have to be read
against, per component -- which is what says whether a component survived.

**What did $z_t$ actually do?** :func:`drift_table` and the figures answer that
on the runs that share the dominant anchor, so the comparison is between models
rather than between measurement passes.

Reads only ``trajectory.json`` files, never the store, so it runs on a laptop
holding no adapters. Timestamps come from file mtimes: they date when a run's
record was last *written here*, which is enough to group runs into measurement
passes but is not a provenance record.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from method import experiments
from method.seed_noise import Z_COMPONENTS
from method.store import get_weights_id
from method.visualization import style
from method.visualization.collect import Collection, collect_group
from method.visualization.labels import (
    HYSTERESIS_CONDITION_LABELS,
    HYSTERESIS_CONDITIONS,
    display_dataset_name,
    display_trait_name,
)
from method.visualization.make_plots import Z_LABELS, default_out_dir

import matplotlib.pyplot as plt  # noqa: E402  (backend fixed by style import)

logger = logging.getLogger("latent_audit")

#: Stands in for a ``weights_id`` at ``t = 0``. Every run of a model is at the
#: same weights before its first fine-tuning step -- that is what $t = 0$ means
#: -- so they are pooled under one key even where they recorded different ids.
#: Recording a different id is itself a finding (see :func:`anchors`), but it
#: cannot make the weights differ, and it is exactly the runs on the odd id
#: whose disagreement needs to be counted.
BASE_CHECKPOINT = "M_0"

#: Values below this are float32 round-trip noise, not a disagreement: the
#: components are computed in float32 and stored as JSON doubles.
TOLERANCE = 1e-6

#: Marker per condition, so a series is identified by shape as well as hue.
#: Same order as :data:`~method.visualization.labels.HYSTERESIS_CONDITIONS`,
#: which is also the order the hues are assigned in.
CONDITION_MARKERS = ("o", "s", "^", "D", "v")


# --- tidy frame -----------------------------------------------------------


def latent_frame(collection: Collection, *, source: str = "base") -> pd.DataFrame:
    """One row per (run, checkpoint), carrying $z_t$ and how it was anchored.

    ``anchor`` is the run's whole $z_0$, rounded and tupled: two runs sharing it
    read the same $v_0$ and the same neutral text, and their levels are directly
    comparable. ``base_id`` is the ``weights_id`` the run *recorded* for its
    base checkpoint and ``expected_base_id`` what the current config hashes to,
    so a run left on a superseded key is visible as the mismatch it is.

    Runs with no latent series are skipped rather than raising, mirroring
    :func:`method.visualization.schema.metric_pairs`: a family that mixes
    fully-measured arms with endpoint-only ones still contributes the former.
    """
    rows = []
    for run in collection.runs:
        if not run.trajectory.has_latent(source):
            continue
        steps = run.trajectory.steps
        base_id = steps[0].weights_id
        anchor = _rounded(steps[0].z[source])
        written = datetime.fromtimestamp(run.path.stat().st_mtime)
        for step in steps:
            z = step.z[source]
            rows.append(
                {
                    "name": run.config.name,
                    "condition": run.label("condition"),
                    "dataset": run.label("dataset"),
                    "realign_trait": run.label("realign_trait"),
                    "trait": run.trait,
                    "seed": run.seed,
                    "t": step.t,
                    "final_t": len(steps) - 1,
                    "checkpoint": BASE_CHECKPOINT if step.t == 0 else step.weights_id,
                    "base_id": base_id,
                    "legacy_base_id": base_id != get_weights_id(run.config, 0),
                    "anchor": anchor,
                    "z": _rounded(z),
                    "written": written,
                    "b": step.behavior[run.trait],
                    **{c: z[c] for c in Z_COMPONENTS},
                }
            )
    return pd.DataFrame(rows)


def _rounded(z: Mapping[str, float]) -> tuple[float, ...]:
    """A latent state as a hashable value, so equal measurements compare equal."""
    return tuple(round(z[component], 6) for component in Z_COMPONENTS)


# --- the audit ------------------------------------------------------------


def anchors(frame: pd.DataFrame) -> pd.DataFrame:
    """The distinct $z_0$ measurements each trait's runs were anchored to.

    One row per (trait, anchor). More than one row per trait means the base
    model's persona vector and neutral answers were derived more than once and
    different runs kept different copies.

    ``legacy_base_id`` marks an anchor whose runs recorded a base ``weights_id``
    other than the one their config hashes to now. That is a stronger fault than
    a re-derivation: those runs asked the store for a base checkpoint under a
    key nothing else uses, so they could not have hit the shared artifacts even
    if the artifacts were there.
    """
    at_base = frame[frame["t"] == 0]
    if at_base.empty:
        return pd.DataFrame()
    grouped = at_base.groupby(["trait", "anchor"], as_index=False).agg(
        runs=("name", "size"),
        seeds=("seed", lambda s: sorted(set(s))),
        conditions=("condition", lambda s: sorted(set(s))),
        first_written=("written", "min"),
        last_written=("written", "max"),
        base_id=("base_id", lambda s: "|".join(sorted(set(s)))),
        legacy_base_id=("legacy_base_id", "any"),
        **{c: (c, "first") for c in Z_COMPONENTS},
    )
    return grouped.sort_values(["trait", "runs"], ascending=[True, False])


def disagreement(frame: pd.DataFrame) -> pd.DataFrame:
    """Per checkpoint measured by more than one run, the spread of each component.

    The weights are identical by construction -- one ``weights_id``, or $t = 0$,
    where no step has been taken -- so every non-zero entry is measurement
    disagreement between runs and nothing else. Rows are returned for every
    shared checkpoint, agreeing or not, so that the fraction affected is
    readable and not just the worst case.
    """
    shared = frame.groupby(["trait", "checkpoint"]).filter(lambda g: len(g) > 1)
    if shared.empty:
        return pd.DataFrame()
    spread = shared.groupby(["trait", "checkpoint"], as_index=False).agg(
        t=("t", "first"),
        runs=("name", "size"),
        distinct_z=("z", "nunique"),
        **{c: (c, _range) for c in Z_COMPONENTS},
        b_spread=("b", _range),
    )
    return spread.sort_values(["trait", "t", "checkpoint"])


def _range(values: pd.Series) -> float:
    return float(values.max() - values.min())


def noise_vs_drift(frame: pd.DataFrame) -> pd.DataFrame:
    r"""Per component: how the cross-run disagreement compares with the drift.

    ``drift`` is the median move from $t = 0$ to the deepest checkpoint the
    family reaches, taken on the dominant anchor so that it is a property of the
    models rather than of the measurement passes. ``worst_disagreement`` is the
    largest spread any single set of identical weights showed.

    ``ratio`` is the number to read: it is the fraction of a component's whole
    observed drift that two runs measuring *the same weights* already disagree
    by. At $\ge 1$ the component carries no usable signal at this family's
    depth; well below it, the drift is real.
    """
    spread = disagreement(frame)
    dominant = on_dominant_anchor(frame)
    rows = []
    for trait, group in dominant.groupby("trait"):
        deepest = group[group["t"] == group["t"].max()]
        base = group[group["t"] == 0]
        per_trait = spread[spread["trait"] == trait] if not spread.empty else spread
        for component in Z_COMPONENTS:
            drift = float(deepest[component].median() - base[component].median())
            worst = float(per_trait[component].max()) if len(per_trait) else np.nan
            # An empty spread carries no columns at all, so the mask below
            # cannot be built from it. A family whose runs share no checkpoint
            # -- one trunk of ``exp2_hregen`` measured so far, say -- lands
            # here, and its drift is still worth reporting beside a blank
            # disagreement.
            disagreeing = (
                per_trait[per_trait[component] > TOLERANCE]
                if len(per_trait)
                else per_trait
            )
            rows.append(
                {
                    "trait": trait,
                    "component": component,
                    "shared_checkpoints": len(per_trait),
                    "disagreeing": len(disagreeing),
                    "worst_disagreement": worst,
                    "median_disagreement": (
                        float(disagreeing[component].median())
                        if len(disagreeing)
                        else 0.0
                    ),
                    "drift": drift,
                    "ratio": abs(worst / drift) if drift else np.inf,
                }
            )
    return pd.DataFrame(rows)


def on_dominant_anchor(frame: pd.DataFrame) -> pd.DataFrame:
    """The subset of runs anchored to their trait's most common $z_0$.

    The majority is the right choice rather than the newest: it is the anchor
    the family as a whole was measured against, so keeping it drops the fewest
    runs. Selection is by *run*, not by row -- a run's later checkpoints are
    read against its own $z_0$, so admitting them while excluding its base would
    reintroduce the mismatch this exists to remove.
    """
    if frame.empty:
        return frame
    modal = frame[frame["t"] == 0].groupby("trait")["anchor"].agg(lambda s: s.mode()[0])
    keep = frame["anchor"] == frame["trait"].map(modal)
    return frame[keep]


# --- the drift itself -----------------------------------------------------


def drift_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Median $z_t$ and $b_t$ per (trait, condition, checkpoint index).

    Medians across the target datasets, re-alignment traits and seeds that share
    an arm: the condition contrast is what the design varies, and the
    per-dataset detail is what the figures split out.
    """
    if frame.empty:
        return frame
    table = frame.groupby(["trait", "condition", "t"], as_index=False).agg(
        runs=("name", "size"),
        **{c: (c, "median") for c in Z_COMPONENTS},
        b=("b", "median"),
    )
    order = {c: i for i, c in enumerate(HYSTERESIS_CONDITIONS)}
    return table.sort_values(
        ["trait", "condition", "t"],
        key=lambda s: s.map(order) if s.name == "condition" else s,
    )


def rotation_share(frame: pd.DataFrame) -> pd.DataFrame:
    r"""How much of the change in $q_t$ is the axis moving rather than the model.

    $p_t$ and $q_t$ read the same activations against the original and the
    current persona axis, so $p_t - p_0$ is drift of the representation along a
    fixed direction while $q_t - p_t$ is what re-extracting the direction added.
    Splitting them keeps "the model's neutral representations moved" apart from
    "the persona vector rotated", which the single number $q_t$ conflates.
    """
    if frame.empty:
        return frame
    per_trait_base = dict(frame[frame["t"] == 0].groupby("trait")["p"].median())
    rows = []
    for (trait, t), group in frame.groupby(["trait", "t"]):
        activation = float(group["p"].median() - per_trait_base[trait])
        axis = float(group["q"].median() - group["p"].median())
        rows.append(
            {
                "trait": trait,
                "t": t,
                "runs": len(group),
                "activation_drift": activation,
                "axis_rotation": axis,
                "total_q_change": activation + axis,
                "axis_share": (
                    abs(axis) / (abs(axis) + abs(activation))
                    if axis or activation
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


# --- figures --------------------------------------------------------------


def _traits(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["trait"].unique())


def _condition_style(condition: str) -> tuple[str, str]:
    """Hue and marker for a condition, keyed to the condition, never to rank."""
    index = (
        HYSTERESIS_CONDITIONS.index(condition)
        if condition in HYSTERESIS_CONDITIONS
        else len(HYSTERESIS_CONDITIONS)
    )
    return (
        style.categorical_color(index),
        CONDITION_MARKERS[index % len(CONDITION_MARKERS)],
    )


def _lookup(drift: pd.DataFrame, trait: str, component: str, column: str) -> float:
    """One cell of :func:`noise_vs_drift`, as a plain float."""
    row = drift[(drift["trait"] == trait) & (drift["component"] == component)]
    return float(row[column].iloc[0])


def _noise_bar(ax: plt.Axes, *, x: float, size: float) -> None:
    """A capped bar of height ``size``, centred in the panel, as a scale rule.

    Drawn in ink rather than in a series colour: it belongs to none of them, and
    a reader who mistakes it for a sixth condition has read the panel backwards.
    """
    if not np.isfinite(size) or size <= 0:
        return
    low, high = ax.get_ylim()
    ax.errorbar(
        x,
        (low + high) / 2,
        yerr=size / 2,
        color=style.SECONDARY_INK,
        elinewidth=1.1,
        capsize=3,
        zorder=4,
    )
    ax.set_ylim(low, high)


#: The two components the anchor figure shows. $q_0$ equals $p_0$ by
#: definition and $\rho_0$ equals 1 by definition, so neither adds a row.
ANCHOR_COMPONENTS = (("p", r"$p_0 = q_0$"), ("r", r"$r_0 = \|v_0\|$"))


def anchor_figure(frame: pd.DataFrame, drift: pd.DataFrame) -> plt.Figure:
    r"""Every distinct $z_0$ the family was measured against, against the drift.

    All the weights behind one panel are the same base model, so a panel with
    more than one dot is showing measurement passes, not models. The shaded band
    is what the whole family's drift in that component amounts to, drawn from the
    dominant anchor: it is the scale against which the dots have to be read, and
    it is the only thing that says whether their spread matters.

    One row per pass, dated and counted, rather than dot area for the count --
    the values differ in the third significant figure, so they need an axis
    zoomed to them, and area is read too poorly for a number this load-bearing.
    """
    style.apply_style()
    traits = _traits(frame)
    table = anchors(frame)
    fig, axes = plt.subplots(
        len(traits),
        len(ANCHOR_COMPONENTS),
        figsize=(9.6, 2.4 * len(traits)),
        squeeze=False,
    )
    for row, trait in enumerate(traits):
        rows = table[table["trait"] == trait].sort_values("first_written")
        dominant = rows.nlargest(1, "runs")
        for col, (component, label) in enumerate(ANCHOR_COMPONENTS):
            ax = axes[row][col]
            travelled = _lookup(drift, trait, component, "drift")
            start = float(dominant[component].iloc[0])
            ax.axvspan(
                min(start, start + travelled),
                max(start, start + travelled),
                color=style.MUTED,
                alpha=0.16,
                linewidth=0,
                zorder=1,
                label="Drift over the whole trajectory",
            )
            positions = np.arange(len(rows))
            ax.scatter(
                rows[component],
                positions,
                s=46,
                color=[
                    style.ORANGE if legacy else style.BLUE
                    for legacy in rows["legacy_base_id"]
                ],
                edgecolor=style.SURFACE,
                linewidth=0.8,
                zorder=3,
            )
            ax.set_yticks(
                positions,
                [
                    f"{when:%b %d} · {n} run{'s' if n > 1 else ''}"
                    for when, n in zip(rows["first_written"], rows["runs"])
                ],
            )
            ax.invert_yaxis()
            ax.set_title(f"{display_trait_name(trait)}: {label}", loc="left")
            ax.grid(axis="x")
            ax.grid(axis="y", visible=False)
            ax.margins(x=0.12, y=0.32)
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=style.BLUE,
            label="Current base checkpoint id",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=style.ORANGE,
            label="Superseded base checkpoint id",
        ),
        Patch(color=style.MUTED, alpha=0.16, label="Drift over the whole trajectory"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        r"One base model, several $z_0$: what $z_t$ is anchored to",
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return fig


def noise_vs_drift_figure(table: pd.DataFrame) -> plt.Figure:
    r"""Disagreement between runs on identical weights, as a share of the drift.

    One bar per component. The reference line at 1 is where the two are equal:
    a component reaching it has a whole trajectory's worth of drift inside its
    own measurement disagreement.
    """
    style.apply_style()
    traits = _traits(table)
    fig, axes = plt.subplots(
        1,
        len(traits),
        figsize=(3.9 * len(traits) + 1.0, 3.2),
        squeeze=False,
        sharex=True,
    )
    positions = np.arange(len(Z_COMPONENTS))
    for col, trait in enumerate(traits):
        ax = axes[0][col]
        rows = (
            table[table["trait"] == trait]
            .set_index("component")
            .loc[list(Z_COMPONENTS)]
        )
        ax.barh(
            positions,
            rows["ratio"],
            height=0.6,
            color=[style.RED if v >= 1 else style.BLUE for v in rows["ratio"]],
            zorder=3,
        )
        for y, (ratio, worst, travelled) in enumerate(
            zip(rows["ratio"], rows["worst_disagreement"], rows["drift"])
        ):
            ax.annotate(
                f"{worst:.3g} of {travelled:+.3g}",
                (ratio, y),
                textcoords="offset points",
                xytext=(6, 0),
                va="center",
                fontsize=8,
                color=style.SECONDARY_INK,
            )
        ax.axvline(1.0, color=style.RED, linewidth=1.0, linestyle="--", zorder=2)
        ax.set_yticks(positions, [Z_LABELS[c] for c in Z_COMPONENTS])
        ax.set_ylim(len(Z_COMPONENTS) - 0.5, -0.5)  # first component at the top
        ax.set_xlabel("Worst disagreement / total drift")
        ax.set_title(display_trait_name(trait), loc="left")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.42)
    fig.suptitle(
        "How much of each component's drift is inside its own measurement noise",
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def drift_figure(frame: pd.DataFrame, drift: pd.DataFrame) -> plt.Figure:
    r"""$z_t$ against step count, one line per condition, one column per component.

    Median over the seeds, target datasets and re-alignment traits sharing an
    arm, with an inter-quartile band. Arms differ in length by design, so the
    lines end at different $t$; $t$ is a step count, not a shared schedule.

    The capped bar at the right of each panel is that component's worst
    cross-run disagreement, drawn to the panel's own scale: it is how much of
    the vertical range is known to be measurement rather than model, and it is
    the reason the $p_t$ panels are not read as a result.

    Arms that share a prefix of checkpoints also share the measurements on them,
    so their lines coincide exactly until they diverge -- ``baseline`` lies under
    ``diff`` at $t \le 1$, and the two re-alignment arms under each other. Later
    conditions are drawn first so the shortest arm stays visible on top.
    """
    style.apply_style()
    traits = _traits(frame)
    fig, axes = plt.subplots(
        len(traits),
        len(Z_COMPONENTS),
        figsize=(3.2 * len(Z_COMPONENTS), 2.7 * len(traits)),
        squeeze=False,
    )
    steps = sorted(frame["t"].unique())
    for row, trait in enumerate(traits):
        for col, component in enumerate(Z_COMPONENTS):
            ax = axes[row][col]
            for condition in reversed(HYSTERESIS_CONDITIONS):
                arm = frame[
                    (frame["trait"] == trait) & (frame["condition"] == condition)
                ]
                if arm.empty:
                    continue
                grouped = arm.groupby("t")[component]
                middle, low, high = (
                    grouped.median(),
                    grouped.quantile(0.25),
                    grouped.quantile(0.75),
                )
                color, marker = _condition_style(condition)
                ax.plot(
                    middle.index,
                    middle.to_numpy(),
                    color=color,
                    marker=marker,
                    markersize=5,
                    markeredgecolor=style.SURFACE,
                    markeredgewidth=0.8,
                    linewidth=1.8,
                    label=HYSTERESIS_CONDITION_LABELS.get(condition, condition),
                    zorder=3,
                )
                ax.fill_between(
                    middle.index,
                    low.to_numpy(),
                    high.to_numpy(),
                    color=color,
                    alpha=0.10,
                    linewidth=0,
                    zorder=2,
                )
            _noise_bar(
                ax,
                x=max(steps) + 0.45,
                size=_lookup(drift, trait, component, "worst_disagreement"),
            )
            ax.set_xlabel("Fine-tuning steps $t$" if row == len(traits) - 1 else "")
            ax.set_xticks(steps)
            ax.set_xlim(min(steps) - 0.25, max(steps) + 0.75)
            ax.set_title(
                f"{display_trait_name(trait)}: {Z_LABELS[component]}", loc="left"
            )
    handles, texts = axes[0][0].get_legend_handles_labels()
    # Conditions are drawn back to front so the shortest arm ends up on top; the
    # legend restores the design's order, skipping any arm that has not run.
    order = [
        texts.index(HYSTERESIS_CONDITION_LABELS[c])
        for c in HYSTERESIS_CONDITIONS
        if HYSTERESIS_CONDITION_LABELS[c] in texts
    ]
    handles = [handles[i] for i in order] + [
        plt.Line2D(
            [], [], color=style.SECONDARY_INK, marker="_", markersize=7, linewidth=1.1
        )
    ]
    texts = [texts[i] for i in order] + ["Worst cross-run disagreement (scale bar)"]
    fig.legend(
        handles,
        texts,
        loc="lower center",
        ncol=min(3, len(texts)),
        bbox_to_anchor=(0.5, -0.05),
    )
    fig.suptitle(
        r"$z_t$ by condition: the latent state tracks step count, not re-alignment",
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return fig


def hysteresis_figure(frame: pd.DataFrame, *, condition: str = "same") -> plt.Figure:
    r"""Behaviour against persona-vector rotation, along one arm's trajectory.

    The ``same`` arm trains on the target dataset, re-aligns, then trains on it
    again, so $b_t$ is designed to go up and come back. Plotting it against
    $\rho_t$ rather than against $t$ is what makes the asymmetry visible: a path
    that returned to where it started would come back along itself.

    One panel per (trait, target dataset) rather than three paths on one axis:
    $b$ runs on a different scale for each dataset, and a shared axis would
    flatten the small-scale ones into the baseline. One path per panel also
    means the dataset is named in the title, so nothing rests on colour.
    """
    style.apply_style()
    traits = _traits(frame)
    arm = frame[frame["condition"] == condition]
    datasets = sorted(arm["dataset"].unique())
    fig, axes = plt.subplots(
        len(traits),
        len(datasets),
        figsize=(3.3 * len(datasets), 2.9 * len(traits)),
        squeeze=False,
    )
    for row, trait in enumerate(traits):
        for col, dataset in enumerate(datasets):
            ax = axes[row][col]
            path = arm[(arm["trait"] == trait) & (arm["dataset"] == dataset)]
            if path.empty:
                ax.set_axis_off()
                continue
            mark = style.dataset_mark(dataset)
            middle = path.groupby("t")[["rho", "b"]].median()
            ax.plot(
                middle["rho"],
                middle["b"],
                color=mark.line,
                linewidth=1.6,
                marker=mark.marker,
                markersize=8,
                markerfacecolor=mark.face,
                markeredgecolor=mark.edge,
                markeredgewidth=0.9,
                zorder=3,
            )
            for t, point in middle.iterrows():
                ax.annotate(
                    f"$t={t}$",
                    (point["rho"], point["b"]),
                    textcoords="offset points",
                    # Alternating side: consecutive checkpoints land close
                    # together once the path doubles back on itself.
                    xytext=(7, 6) if t % 2 else (7, -13),
                    fontsize=8,
                    color=style.SECONDARY_INK,
                )
            ax.axhline(
                middle["b"].iloc[0],
                color=style.BASELINE,
                linewidth=0.9,
                linestyle="--",
                zorder=1,
            )
            ax.set_xlabel(
                r"Persona-vector rotation $\rho_t$" if row == len(traits) - 1 else ""
            )
            ax.set_ylabel("Trait score $b_t$" if col == 0 else "")
            ax.invert_xaxis()  # time runs left to right: rho only ever falls
            ax.margins(x=0.22, y=0.18)
            ax.set_title(
                f"{display_trait_name(trait)} · {display_dataset_name(dataset)}",
                loc="left",
            )
    fig.suptitle(
        r"Behaviour returns to its $t=0$ line, $\rho_t$ does not: the "
        f"{HYSTERESIS_CONDITION_LABELS.get(condition, condition).lower()} arm",
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# --- CLI ------------------------------------------------------------------


def report(collection: Collection, *, source: str = "base") -> dict[str, pd.DataFrame]:
    """Every table this script prints, keyed by name, for reuse in a notebook."""
    frame = latent_frame(collection, source=source)
    dominant = on_dominant_anchor(frame)
    return {
        "frame": frame,
        "anchors": anchors(frame),
        "disagreement": disagreement(frame),
        "noise_vs_drift": noise_vs_drift(frame),
        "drift": drift_table(dominant),
        "rotation_share": rotation_share(dominant),
    }


def figures(
    frame: pd.DataFrame, tables: dict[str, pd.DataFrame]
) -> dict[str, plt.Figure]:
    """The four figures, keyed by the filename stem they are saved under."""
    dominant = on_dominant_anchor(frame)
    drift = tables["noise_vs_drift"]
    return {
        "latent_anchors": anchor_figure(frame, drift),
        "latent_noise_vs_drift": noise_vs_drift_figure(drift),
        "latent_drift": drift_figure(dominant, drift),
        "latent_hysteresis": hysteresis_figure(dominant),
    }


def _show(title: str, frame: pd.DataFrame, *, note: str = "") -> None:
    print(f"\n=== {title} ===" + (f"\n{note}" if note else ""))
    if frame is None or frame.empty:
        print("(nothing to report)")
        return
    shown = frame.copy()
    for column in shown.select_dtypes("datetime"):
        shown[column] = shown[column].dt.strftime("%Y-%m-%d %H:%M")
    print(shown.round(4).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        default=experiments.EXP3,
        help="experiment family to audit (default: exp3)",
    )
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--source", default="base", help="which h_neutral source z_t comes from"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "where to write the figures; a per-family subdirectory is created "
            "underneath (default: one per run source, as make_plots uses)"
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="also write the per-checkpoint frame here",
    )
    parser.add_argument("--no-plots", action="store_true", help="tables only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    collection = collect_group(args.group, local=args.local, mock=args.mock)
    print(collection.summary())
    if not collection:
        raise SystemExit(f"no runs on disk for {args.group!r}")

    tables = report(collection, source=args.source)
    frame = tables["frame"]
    if frame.empty:
        raise SystemExit(f"no run in {args.group!r} carries a {args.source!r} latent")

    _show(
        "z_0 anchors: distinct base measurements the runs were read against",
        tables["anchors"],
        note=(
            "One row per trait is the healthy case. Every run here is at the "
            "same weights (t=0), so extra rows are measurement passes."
        ),
    )
    _show(
        "identical weights, disagreeing z",
        (
            tables["disagreement"][
                tables["disagreement"][list(Z_COMPONENTS)].max(axis=1) > TOLERANCE
            ]
            if not tables["disagreement"].empty
            else tables["disagreement"]
        ),
        note="Checkpoints two or more runs measured; spread is max - min.",
    )
    _show(
        "disagreement against drift, per component",
        tables["noise_vs_drift"],
        note="ratio >= 1 means the component's drift is inside its own noise.",
    )
    _show(
        "z_t by condition (dominant anchor only)",
        tables["drift"],
        note="Medians over seeds, target datasets and re-alignment traits.",
    )
    _show(
        "what moved: the representation or the axis",
        tables["rotation_share"],
        note="q_t - p_0 split into drift along the fixed axis and re-extraction.",
    )

    if args.csv is not None:
        frame.to_csv(args.csv, index=False)
        print(f"\nper-checkpoint frame -> {args.csv}")

    if not args.no_plots:
        out_dir = (
            args.out_dir or default_out_dir(local=args.local, mock=args.mock)
        ) / args.group
        for name, fig in figures(frame, tables).items():
            png, _ = style.save_figure(fig, f"{args.group}_{name}", out_dir)
            plt.close(fig)
            print(f"wrote {png}")


if __name__ == "__main__":
    main()
