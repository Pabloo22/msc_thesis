"""Build experiment figures and tables from collected trajectories."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from method import experiments, seed_noise
from method.latent import H_NORM
from method.visualization import decay, figures, forecast, style, training_curves
from method.visualization.collect import (
    Collection,
    axis_refresh_frame,
    collect_group,
    hysteresis_frame,
    seed_noise_frame,
)
from method.visualization.labels import (
    BASE_SOURCE,
    CURRENT_SOURCE,
    DELTA_P_BASE,
    HYSTERESIS_CONDITIONS,
    TRAITS,
    TRUNKS,
    display_condition_name,
    display_dataset_name,
    display_trait_name,
    display_trunk_name,
    display_trunk_short,
    display_trunk_title,
    neutral_norm_symbol,
    source_index,
    trunk_index,
    z_component_definition,
    z_component_symbol,
    z_symbol,
)

import matplotlib.pyplot as plt  # noqa: E402  (backend fixed by style import)

logger = logging.getLogger("make_plots")

def z_labels(source: str = "base") -> dict[str, str]:
    r"""$z_t$ component -> how a panel or table key writes it.

    Indexed by the response source the components were measured under, because
    two of them now exist: the decay trunks answer the neutral prompts with
    $M_0$ and the ``exp2_hregen`` trunks with $M_t$, and $p_t^{[0]}$ and
    $p_t^{[t]}$ are different quantities plotted on identically shaped axes.
    The persona index is always $0$ -- every $z_t$ in the figures reads the
    axis extracted from $M_0$'s frozen extraction text, and the on-policy one
    is reported by :mod:`method.axis_refresh` instead.
    """
    index = source_index(source)
    return {
        c: f"${z_component_symbol(c, neutral=index)}$" for c in decay.Z_COMPONENTS
    }


def drift_z_labels(source: str = "base") -> dict[str, str]:
    """The drift grid's columns: z_t's four, then the length they divide by.

    Separate from :func:`z_labels` because ``h_norm`` is not a component of
    z_t (see :data:`method.latent.H_NORM`) and the audit figures label z_t's
    coordinates from that mapping -- a fifth entry there would put a length on
    axes that only hold the four.

    Each of the four carries its definition on a second line, because this is
    the one figure a reader meets the coordinates in without the notation
    table beside it: ``p`` and ``q`` are both cosines of the same activation
    and differ only in which persona vector they are taken against, which the
    bare symbols do not say. ``h_norm`` gets no second line -- its symbol is
    already the expression.
    """
    index = source_index(source)
    defined = {
        c: f"${z_component_symbol(c, neutral=index)}$\n"
        f"${z_component_definition(c, neutral=index)}$"
        for c in decay.Z_COMPONENTS
    }
    return {**defined, H_NORM: f"${neutral_norm_symbol(generator=index)}$"}


#: The default-source dicts, for the callers that never vary it: experiment 3
#: and the audit figures measure ``h_neutral`` from $M_0$ only.
Z_LABELS = z_labels()
DRIFT_Z_LABELS = drift_z_labels()


def _emit(fig: plt.Figure, name: str, out_dir: Path, saved: list[Path]) -> None:
    saved.extend(style.save_figure(fig, name, out_dir))
    plt.close(fig)
    logger.info("wrote %s", out_dir / f"{name}.png")


#: An en dash for a cell the sweep has not measured -- distinguishable at a
#: glance from a zero, which is a result and not a gap.
_MISSING_CELL = "--"


@dataclass(frozen=True)
class _Scale:
    """How a table's numbers are printed, and which of them leads its block.

    The two travel together because they are the same decision made twice: a
    cell is bolded for leading at the precision it is printed at, so a table
    that changed one without the other would bold a lead its reader cannot see.

    ``rank`` maps the printed table to a score whose *largest* value leads. A
    correlation ranks by itself, larger being better; an error ranks by its
    negation, since being off by less is doing better; a bias ranks by how near
    zero it is, since over- and under-predicting by 5 are the same miss.
    """

    decimals: int
    rank: Callable[[pd.DataFrame], pd.DataFrame]

    def format(self, value: float) -> str:
        return f"{value:.{self.decimals}f}"


#: Two decimals, as the figures label their fits with, and larger is better.
CORRELATION_SCALE = _Scale(2, lambda table: table)

#: One decimal, because these are judge points on a 0-100 scale and the second
#: one is below the noise the eval itself carries; smaller is better.
ERROR_SCALE = _Scale(1, lambda table: -table)

#: The same precision, ranked by distance from zero (see :class:`_Scale`).
BIAS_SCALE = _Scale(1, lambda table: -table.abs())

#: Two decimals, because a loss and a gradient norm here differ between arms in
#: the second one, and smallest leads: the table is read for which arm entered
#: its final step with the least left to learn.
TRACE_SCALE = _Scale(2, lambda table: -table)


def _key_blocks(table: pd.DataFrame) -> list[pd.Index]:
    """The rows sharing every key but the last, as groupers over ``table``.

    The last key is what varies inside a block -- the projections measured on
    one trait and one trunk, or the forecasters made from one projection --
    and every key above it is what the block holds fixed. Every table but the
    recalibration sweep is bolded in these blocks (see
    :func:`_supertable_blocks`).
    """
    keys = [table.index.get_level_values(i) for i in range(table.index.nlevels - 1)]
    # A single block when there is no key left to group by, which reads the
    # whole column -- the same rule, applied to a table with one key column.
    return keys or [pd.Index([0] * len(table))]


def _leading_cells(
    table: pd.DataFrame,
    scale: _Scale = CORRELATION_SCALE,
    *,
    blocks: Callable[[pd.DataFrame], list[pd.Index]] = _key_blocks,
) -> pd.DataFrame:
    r"""Which cells lead their block, to be bolded."""
    shown = scale.rank(table.round(scale.decimals))
    grouped = shown.groupby(blocks(table))
    return shown.eq(grouped.transform("max")) & grouped.transform("nunique").gt(1)


def _column_spec(keys: int, values: int, summary: int = 0) -> str:
    """``l`` per key column and ``r`` per value, the keys ruled off.

    A vertical rule after each key: they answer different questions -- which
    trait, which trunk, which projection -- and the last of them also divides
    the keys from the numbers they lead to. The value columns are one block
    read across and take no rules between them. A summary column is ruled off
    from that block in turn: it is not a member of what it summarises, and the
    rule is what stops a reader taking it for one more checkpoint.
    """
    return "l|" * keys + "r" * values + ("|" + "r" * summary if summary else "")


def _key_spans(rows: Sequence[tuple[str, ...]]) -> list[list[int]]:
    """How many rows each key covers, counted from the row that opens it.

    Zero on a row whose key the row above already covers, which is what makes
    a key appear once per block rather than once per row. A key is opened by
    its whole prefix, not by its own value alone: two trunks of the same name
    under different traits are two blocks, and a run of them is not one.
    """
    if not rows:
        return []
    spans = [[0] * len(row) for row in rows]
    for level in range(len(rows[0])):
        start = 0
        for row in range(1, len(rows) + 1):
            if row == len(rows) or rows[row][: level + 1] != rows[start][: level + 1]:
                spans[start][level] = row - start
                start = row
    return spans


def _key_cell(key: str, span: int) -> str:
    """A key, centred on the rows it covers and blank where it is covered."""
    if not span:
        return ""
    return key if span == 1 else rf"\multirow{{{span}}}{{*}}{{{key}}}"


def _block_rule(
    row_keys: tuple[str, ...], previous: tuple[str, ...], total: int
) -> str:
    r"""The rule that opens the block ``row_keys`` starts, if it starts one.

    Full width when the leading key turns over, and from the turning key
    rightwards when a deeper one does -- a ``\cline`` rather than an
    ``\hline`` so that the rule chunking the trunks does not strike through
    the trait spanning them. The last key is the one that varies within a
    block, so it opens nothing.
    """
    for level, (key, before) in enumerate(zip(row_keys[:-1], previous[:-1])):
        if key != before:
            return r"\hline" if level == 0 else rf"\cline{{{level + 1}-{total}}}"
    return ""


def _latex_table(
    table: pd.DataFrame,
    headings: Sequence[str],
    spanner: str,
    *,
    summary: int = 0,
    scale: _Scale = CORRELATION_SCALE,
    note: str = "",
    blocks: Callable[[pd.DataFrame], list[pd.Index]] = _key_blocks,
) -> str:
    r"""A ``tabular`` for the correlation table, as a fragment to ``\input``."""
    # Key columns from the index rather than from the headings: the two must
    # agree, and it is the frame that says how many keys a row has.
    keys, columns = table.index.nlevels, len(table.columns)
    values = columns - summary
    lines = [
        "% Generated by method.visualization.make_plots -- do not edit.",
        *([f"% {note}"] if note else []),
        rf"\begin{{tabular}}{{{_column_spec(keys, values, summary)}}}",
        " & ".join(
            [""] * keys
            + [rf"\multicolumn{{{values}}}{{c}}{{{spanner}}}"]
            + [""] * summary
        )
        + r" \\",
        " & ".join([*headings, *(str(column) for column in table.columns)]) + r" \\",
        r"\hline",
    ]
    rows = [tuple(row_keys) for row_keys in table.index]
    spans = _key_spans(rows)
    leaders = _leading_cells(table, scale, blocks=blocks)
    previous: tuple[str, ...] = ()
    for row_keys, span, row, best in zip(
        rows, spans, table.to_numpy(), leaders.to_numpy()
    ):
        # No rule before the first block: its rule is the header's, already
        # drawn, and a second one under it would box the headings in.
        rule = _block_rule(row_keys, previous, keys + columns) if previous else ""
        if rule:
            lines.append(rule)
        cells = [
            (
                _MISSING_CELL
                if pd.isna(value)
                else _bold(scale.format(value), leading)
            )
            for value, leading in zip(row, best)
        ]
        shown = [_key_cell(key, covered) for key, covered in zip(row_keys, span)]
        lines.append(" & ".join([*shown, *cells]) + r" \\")
        previous = row_keys
    lines += [r"\end{tabular}", ""]
    return "\n".join(lines)


def _bold(cell: str, leading: bool) -> str:
    """``cell``, marked up if it leads its block and column."""
    return rf"\textbf{{{cell}}}" if leading else cell


def _emit_table(
    table: pd.DataFrame,
    headings: Sequence[str],
    spanner: str,
    name: str,
    out_dir: Path,
    saved: list[Path],
    *,
    summary: int = 0,
    scale: _Scale = CORRELATION_SCALE,
    note: str = "",
    blocks: Callable[[pd.DataFrame], list[pd.Index]] = _key_blocks,
) -> None:
    """Write a table both ways: ``.tex`` to typeset, ``.csv`` to read back.

    The two carry the same columns, ``summary`` ones included: the ``.csv`` is
    what a reader checks a quoted number against, and a column it was missing
    would send them back to the ``.tex`` to read it off the typeset table.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tex, csv = out_dir / f"{name}.tex", out_dir / f"{name}.csv"
    tex.write_text(
        _latex_table(
            table,
            headings,
            spanner,
            summary=summary,
            scale=scale,
            note=note,
            blocks=blocks,
        )
    )
    table.to_csv(csv)
    saved += [tex, csv]
    logger.info("wrote %s", tex)


# --- experiment 2: the RQ1 decay experiment -------------------------------- The two projection differences the decay grid draws, as ``decay_frame`` columns: the ends of the.

DECAY_GRID_SERIES = (decay.SERIES_COLUMNS["p0"], decay.SERIES_COLUMNS["full_t"])

#: The 3x2 of projection-difference variants as the headline figure draws it:
#: one group per persona vector, coloured by its step of the ordered ramp, its
#: two members left in the order that decides their line style. Built once here
#: rather than at each call, so the two layouts the figure emits are the same
#: encoding twice over and not two conventions a reader has to learn.
REFRESH_GROUPS = tuple(
    figures.CurveGroup(label=label, color=color, series=members)
    for (_, label, members), color in zip(decay.REFRESH_GROUPS, style.VECTOR_RAMP)
)

#: The RMSE headline's line-style key also states the pre-specified target.
#: The target changes with the source of the predicted-answer activations, so
#: leaving it only in prose would hide part of what separates the two curves.
HEADLINE_RMSE_MEMBER_LABELS = (
    decay.PREDICTED_LABELS[0] + r", predicts $b_{t+1}$",
    decay.PREDICTED_LABELS[1] + r", predicts $\Delta b_{t+1}$",
)

HEADLINE_RMSE_TARGETS = ("matched", "change", "level")


def _headline_rmse_spec(
    target: str,
) -> tuple[str, str, Mapping[str, str] | None, tuple[str, ...]]:
    """Filename suffix, fixed model, per-series models and legend for a policy."""
    if target == "matched":
        return (
            "",
            "step0",
            forecast.HEADLINE_MODEL_BY_SERIES,
            HEADLINE_RMSE_MEMBER_LABELS,
        )
    if target == "change":
        labels = tuple(
            label + r", predicts $\Delta b_{t+1}$" for label in decay.PREDICTED_LABELS
        )
        return "_change", "step0", None, labels
    if target == "level":
        labels = tuple(
            label + r", predicts $b_{t+1}$" for label in decay.PREDICTED_LABELS
        )
        return "_level", "step0_level", None, labels
    raise ValueError(
        f"unknown headline RMSE target {target!r}; expected one of "
        f"{(*HEADLINE_RMSE_TARGETS, 'all')}"
    )


def _headline_rmse_targets(target: str) -> tuple[str, ...]:
    """Expand ``all`` while keeping the default matched output first."""
    if target == "all":
        return HEADLINE_RMSE_TARGETS
    _headline_rmse_spec(target)  # validate before any figure is written
    return (target,)


#: The correlation-summary bars use the same two-factor encoding as the
#: headline curves: colour identifies the persona-vector version, and ``//``
#: identifies the member that uses responses regenerated by the checkpoint.
REFRESH_COLORS = {
    name: color
    for (_, _, members), color in zip(decay.REFRESH_GROUPS, style.VECTOR_RAMP)
    for name in members
}
REFRESH_HATCHES = {
    name: "//" if member else ""
    for _, _, members in decay.REFRESH_GROUPS
    for member, name in enumerate(members)
}

#: Add the fixed initial-model projection difference as a neutral reference;
#: the remaining colours and hatches retain the headline figure's encoding.
DECAY_SUMMARY_SERIES = ("p0", *decay.REFRESH_ORDER)
DECAY_SUMMARY_COLORS = {"p0": style.MUTED, **REFRESH_COLORS}

#: What the checkpoint columns of every exp2 table are headed as a block. The
#: key columns left of them are headed by :func:`_headings`, from the keys the
#: table is actually indexed by.
DECAY_TABLE_SPANNER = "Checkpoint $t$"

#: ...and what the column summarising them is headed, right of the last of
#: them (see :func:`_with_mean`).
DECAY_TABLE_MEAN = "Mean"


#: The families the decay experiment is split across. They share a base
#: checkpoint and a probe set, and every figure below needs at least two of
#: them, so they are collected together whichever one was asked for.
EXP2_GROUPS = (
    experiments.EXP2_VALIDATION,
    experiments.EXP2_DECAY,
    experiments.EXP2_RESEED,
    experiments.EXP2_AXIS,
    experiments.EXP2_REGEN,
    experiments.EXP2_V0REGEN,
    experiments.EXP2_HREGEN,
    experiments.EXP2_ONPOLICY,
    experiments.EXP2_ONPOLICY_REGEN,
)

#: Checkpoint-level predictors for the correlation model.
def mechanism_predictors(
    source: str = "base", *, include_onpolicy: bool = True
) -> dict[str, str]:
    """The predictors, labelled for the response source they were measured at.

    $\rho$ and $r$ are properties of the persona vector alone, so neither
    moves with the neutral-response ``source``.  Plot both the persona axis
    extracted from $M_0$'s cached responses and the one regenerated from
    $M_t$'s responses when the axis-refresh sweep is available.
    """
    frozen_rho = f"Persona-vector rotation ${z_component_symbol('rho')}$"
    frozen_r = f"Persona-vector norm ${z_component_symbol('r')}$"
    if not include_onpolicy:
        return {
            "rho": frozen_rho,
            "r": frozen_r,
            "b_t": r"Behaviour level $b_t$",
            "steps_since_realignment": "Steps since re-alignment",
        }
    return {
        "rho": frozen_rho,
        "rho_onpolicy": (
            "Persona-vector rotation "
            f"${z_component_symbol('rho', persona='t')}$"
        ),
        "b_t": r"Behaviour level $b_t$",
        "r": frozen_r,
        "r_onpolicy": (
            f"Persona-vector norm ${z_component_symbol('r', persona='t')}$"
        ),
        "steps_since_realignment": "Steps since re-alignment",
    }


#: The default-source labels, kept as a constant for callers with no source to
#: pass (and for the tests that name the predictors).
MECHANISM_PREDICTORS = mechanism_predictors()


def _trunk_colors(trunks: Sequence[str]) -> dict[str, str]:
    """A fixed hue per trunk, assigned by identity rather than by row order."""
    return {trunk: style.categorical_color(trunk_index(trunk)) for trunk in trunks}


def _present(found: Iterable[str], order: Sequence[str]) -> list[str]:
    """The values in ``found``, in the design's fixed order.

    Anything the design does not name is kept, sorted, at the end rather than
    dropped: an unexpected trunk or trait is a figure worth seeing, and one
    silently omitted is not.
    """
    seen = set(found)
    return [value for value in order if value in seen] + sorted(seen - set(order))


def _present_trunks(frame: pd.DataFrame) -> list[str]:
    """Return present trunks in design order."""
    return _present(frame["trunk"], TRUNKS)


def _present_traits(frame: pd.DataFrame) -> list[str]:
    """Traits with rows in ``frame``, primary (sycophancy) first."""
    return _present(frame["trait"], TRAITS)


def _present_series(fits: pd.DataFrame) -> list[str]:
    r"""Projection series with at least one checkpoint fitted, in design order.

    The re-measured series each need a family of their own, so on a run of the
    decay family alone their columns are entirely NaN. Dropping them here
    rather than in the figures is what keeps an unmeasured series out of the
    legend -- a key for a line nobody drew reads as a line that came out flat.
    """
    return [
        series
        for series in decay.SERIES
        if f"corr_{series}" in fits and fits[f"corr_{series}"].notna().any()
    ]


#: Families that sweep seeds over a fixed step sequence. exp3 qualifies
#: because its arms are the closest analogue of an exp2 branch -- several
#: one-step arms, five seeds each.
SEED_NOISE_SOURCES = (experiments.EXP3,)


def _sigma_seed(collections: Mapping[str, Collection]) -> dict[str, float]:
    r"""Estimate $\sigma_{seed}(b)$ per trait from multi-seed families."""
    estimates: dict[str, float] = {}
    for group in SEED_NOISE_SOURCES:
        collection = collections.get(group)
        if not collection:
            continue
        single = seed_noise.single_step_behavior_noise(seed_noise_frame(collection))
        for trait, arms in single.groupby("trait"):
            estimates.setdefault(str(trait), float(arms["sd"].median()))
    return estimates


def _pivot_over_t(
    frame: pd.DataFrame, *, index: str | list[str], value: str
) -> pd.DataFrame:
    """``index`` -> its ``value`` at each ``t``, ordered by ``t``.

    Rows missing any checkpoint are dropped rather than plotted short, for the
    same reason a ragged seed is: a truncated line reads as a quantity that
    stopped moving, not as a measurement that never happened.
    """
    if frame.empty:
        return pd.DataFrame()
    wide = frame.pivot_table(index=index, columns="t", values=value).sort_index(axis=1)
    complete = wide.dropna(axis=0, how="any")
    dropped = sorted(set(wide.index) - set(complete.index))
    if dropped:
        logger.warning(
            "%s missing at some checkpoint for %s; omitted from the drift plot",
            value,
            ", ".join(str(d) for d in dropped),
        )
    return complete


def _series_by(
    frame: pd.DataFrame, *, key: str, value: str
) -> dict[str, list[float]]:
    """One line per ``key``. For a frame holding a single seed."""
    return {
        str(k): [float(v) for v in row]
        for k, row in _pivot_over_t(frame, index=key, value=value).iterrows()
    }


def _replicates_by(
    frame: pd.DataFrame, *, key: str, value: str
) -> dict[str, list[list[float]]]:
    """One line per ``(key, seed)``, grouped under the ``key`` each replicates.

    Seed has to be in the pivot index, not just the frame: ``pivot_table``
    aggregates whatever shares an index entry, so pivoting on ``key`` alone
    would average the replicate seeds together and draw a single line that no
    seed actually produced. That was invisible while the reseed family was one
    seed, and silently wrong the moment :data:`EXP2_RESEED_SEEDS` landed.

    Grouped rather thana html flattened because the caller draws each replicate in
    the colour of the primary series it replicates, which is what keeps a
    six-seed overlay from spending six hues on one quantity.
    """

    if "seed" not in frame.columns:
        return {}
    pivoted = _pivot_over_t(frame, index=[key, "seed"], value=value)
    if pivoted.empty:
        return {}
    return {
        str(name): [[float(v) for v in row] for _, row in block.iterrows()]
        for name, block in pivoted.groupby(level=0)
    }


def _split_by_seed(
    frame: pd.DataFrame, primary: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a frame into the primary run and its reseeded replicate(s)."""
    return frame[frame["seed"] == primary], frame[frame["seed"] != primary]


def build_exp2(
    collections: Mapping[str, Collection],
    out_dir: Path,
    *,
    stat: str = "mean",
    source: str = "base",
    sigma_seed: Mapping[str, float] | None = None,
    n_resamples: int = 2000,
    headline_rmse_target: str = "matched",
) -> list[Path]:
    r"""Every figure of the RQ1 decay experiment, over every measured trait."""
    saved: list[Path] = []
    decay_runs = collections.get(experiments.EXP2_DECAY) or Collection(
        experiments.EXP2_DECAY
    )
    validation = collections.get(experiments.EXP2_VALIDATION) or Collection(
        experiments.EXP2_VALIDATION
    )
    reseed = collections.get(experiments.EXP2_RESEED) or Collection(
        experiments.EXP2_RESEED
    )
    # Families that re-measure trunks the decay family already ran, each
    # contributing one more projection series to the same rows.
    axis_runs = collections.get(experiments.EXP2_AXIS) or Collection(
        experiments.EXP2_AXIS
    )
    regen_runs = collections.get(experiments.EXP2_REGEN) or Collection(
        experiments.EXP2_REGEN
    )
    v0regen_runs = collections.get(experiments.EXP2_V0REGEN) or Collection(
        experiments.EXP2_V0REGEN
    )
    onpolicy_runs = collections.get(experiments.EXP2_ONPOLICY) or Collection(
        experiments.EXP2_ONPOLICY
    )
    onpolicy_regen_runs = collections.get(
        experiments.EXP2_ONPOLICY_REGEN
    ) or Collection(experiments.EXP2_ONPOLICY_REGEN)
    remeasured = [
        axis_runs,
        regen_runs,
        v0regen_runs,
        onpolicy_runs,
        onpolicy_regen_runs,
    ]
    # Not a DeltaP re-measurement, so not part of ``remeasured``: this family
    # re-takes ``h_neutral`` and therefore contributes a second $z_t$ series,
    # which reaches the figures through ``--source`` rather than through a
    # column of its own.
    hregen_runs = collections.get(experiments.EXP2_HREGEN) or Collection(
        experiments.EXP2_HREGEN
    )
    if not (decay_runs or validation):
        logger.warning("exp2: no decay or validation runs on disk; skipping")
        return saved

    sigma_seed = dict(sigma_seed or {})
    fan = decay.validation_frame(validation, stat=stat)

    def measured(neutral_source: str) -> pd.DataFrame:
        return decay.decay_frame(
            decay_runs,
            validation,
            remeasured,
            neutral=hregen_runs,
            stat=stat,
            source=neutral_source,
        )

    rows = measured(source)
    # The super table compares the two neutral-response sources against each
    # other, so it needs both frames however ``--source`` was pointed; nothing
    # else here reads more than the one it was asked for.
    other_sources = {
        name: measured(name) for name in SUPERTABLE_SOURCES if name != source
    }
    drift_runs = [*decay_runs.runs, *reseed.runs]
    hatted_ratios = decay.probe_drift_frame(drift_runs, stat=stat, source=source)
    current_ratios = decay.current_probe_drift_frame(regen_runs.runs, stat=stat)
    # The h-regen trunks join the drift runs rather than replacing them: the
    # same trunk is measured by both families, under one z source each, and
    # ``latent_frame`` keeps whichever of them carries the source asked for.
    latents = decay.latent_frame(
        [*drift_runs, *hregen_runs.runs], stat=stat, source=source
    )
    refreshed_axes = axis_refresh_frame(decay_runs.runs)

    traits = _present(
        [*validation.values("trait"), *decay_runs.values("trait")], TRAITS
    )
    saved += _validation_figure(fan, traits, out_dir)
    saved += _decay_figures(
        rows,
        out_dir,
        sigma_seed=sigma_seed,
        n_resamples=n_resamples,
        source=source,
        refreshed_axes=refreshed_axes,
    )
    saved += _forecast_figures(
        rows,
        fan,
        out_dir,
        source=source,
        other_sources=other_sources,
        headline_rmse_target=headline_rmse_target,
    )
    saved += _drift_delta_hat_p_figure(hatted_ratios, out_dir)
    saved += _drift_delta_p_figure(current_ratios, out_dir)
    saved += _drift_latent_figure(latents, out_dir, source=source)
    return saved


def _validation_figure(
    fan: pd.DataFrame, traits: Sequence[str], out_dir: Path
) -> list[Path]:
    """Plot 1: the 24-dataset replication of the persona-vectors correlation.

    One panel per trait. A trait with no fan is left out of the figure rather
    than panelled empty: unlike the drift grids, these panels are not read
    against each other cell by cell, so a missing one costs alignment nothing
    and only wastes half the width.
    """
    saved: list[Path] = []
    measured = [trait for trait in traits if not fan[fan["trait"] == trait].empty]
    for trait in traits:
        if trait not in measured:
            logger.warning(
                "exp2/%s: no validation runs, so the t=0 fan (plot 1) and the "
                "t=0 column of the decay grid are both unavailable. Run the "
                "%r family first because it gates downstream plots",
                trait,
                experiments.EXP2_VALIDATION,
            )
    if not measured:
        return saved
    fig = figures.scatter_validation(
        fan[fan["trait"].isin(measured)],
        traits=measured,
        trait_labels={trait: display_trait_name(trait) for trait in measured},
    )
    _emit(fig, "exp2_validation", out_dir, saved)
    return saved


#: How each key level of an emitted table is written for a reader, by the name the frame gives that level.
def _key_labels(source: str = "base") -> Mapping[str, Callable[[str], str]]:
    """The renderers, with the forecaster rows indexed by their $z_t$ source."""
    forecasters = forecast.forecaster_labels(source)
    return {
        "trait": display_trait_name,
        "trunk": display_trunk_short,
        "series": lambda name: decay.SERIES_LABELS.get(name, name),
        "model": lambda name: forecasters.get(name, name),
    }

#: ...and what each is headed, above the keys themselves.
_KEY_HEADINGS = {
    "trait": "Trait",
    "trunk": "Trunk",
    "series": "Projection",
    "model": "Forecast",
}


def _level_names(table: pd.DataFrame) -> list[str]:
    """A table's key levels as plain strings.

    ``MultiIndex.names`` is typed as possibly-unnamed and possibly-not-string,
    and every lookup here keys off the name, so the coercion happens once.
    """
    return [str(name) for name in table.index.names]


def _headings(keys: Sequence[str]) -> tuple[str, ...]:
    """The column headings for a table indexed by ``keys``, in their order."""
    return tuple(_KEY_HEADINGS.get(key, key.title()) for key in keys)


def _labelled_table(table: pd.DataFrame, source: str = "base") -> pd.DataFrame:
    """A key-indexed table with its keys written as the figures write them.

    Driven by the frame's own level names rather than by position, so a table
    that orders its keys differently -- which the forecast tables do, to put
    the rows a block compares next to each other -- is labelled correctly
    without a second copy of this.
    """
    if table.empty:
        return table
    renderers = _key_labels(source)
    return table.set_index(
        pd.MultiIndex.from_tuples(
            [
                tuple(
                    renderers.get(level, str)(key)
                    for level, key in zip(_level_names(table), row)
                )
                for row in table.index
            ],
            names=table.index.names,
        )
    )


def _with_mean(table: pd.DataFrame, heading: str) -> pd.DataFrame:
    """``table`` with each row's mean across the checkpoints appended.

    The checkpoint columns answer where a projection difference tracks
    behaviour best; this one answers how well it tracks it over the sweep as a
    whole, which is otherwise a sum a reader does by eye across seven cells and
    gets roughly. Bolded by the same rule as the rest (see
    :func:`_leading_cells`), so the column names the better predictor over the
    trunk in the same ink the columns beside it name it at each step.

    Taken over the checkpoints the row was measured at rather than over all of
    them, so that a row with a gap averages what there is. Its blockmates then
    average a different set of checkpoints, which is a comparison to make with
    the gap in view -- and the gap is visible in the row the mean is on.
    """
    if table.empty:
        return table
    return table.assign(**{heading: table.mean(axis=1)})


def _decay_figures(
    rows: pd.DataFrame,
    out_dir: Path,
    *,
    sigma_seed: Mapping[str, float],
    n_resamples: int,
    source: str = "base",
    refreshed_axes: pd.DataFrame | None = None,
) -> list[Path]:
    """Decay plots and tables from the same ``(trait, trunk, t, probe)`` rows."""
    saved: list[Path] = []
    if rows.empty:
        logger.warning(
            "exp2: no checkpoint has both a Delta P and a branch endpoint; "
            "skipping the decay figures"
        )
        return saved

    traits = _present_traits(rows)
    trunks = _present_trunks(rows)
    colors = _trunk_colors(trunks)
    labels = {t: display_trunk_name(t) for t in trunks}
    trait_labels = {trait: display_trait_name(trait) for trait in traits}

    fig = figures.decay_scatter_grid(
        rows,
        traits=traits,
        trait_labels=trait_labels,
        trunks=trunks,
        trunk_labels=labels,
        series=DECAY_GRID_SERIES,
    )
    _emit(fig, "exp2_decay_grid", out_dir, saved)

    for trait in traits:
        if trait in sigma_seed:
            logger.info("exp2/%s: sigma_seed(b) = %.2f", trait, sigma_seed[trait])
        else:
            logger.warning(
                "exp2/%s: no sigma_seed(b) available, so the fit noise ceiling "
                "counts eval noise only and is an upper bound on the true "
                "ceiling. Run a seed-swept family (exp3) and re-plot, or pass "
                "--sigma-seed",
                trait,
            )
    # Fitted a trait at a time only because the ceiling takes that trait's seed
    # noise; every figure below reads the concatenation as one frame.
    fits = pd.concat(
        [
            decay.fit_frame(
                rows[rows["trait"] == trait],
                sigma_seed=sigma_seed.get(trait, 0.0),
                n_resamples=n_resamples,
            )
            for trait in traits
        ],
        ignore_index=True,
    )
    _emit_table(
        _with_mean(_labelled_table(decay.correlation_table(fits)), DECAY_TABLE_MEAN),
        _headings(decay.CORRELATION_TABLE_KEYS),
        DECAY_TABLE_SPANNER,
        "exp2_decay_correlations",
        out_dir,
        saved,
        summary=1,
    )

    fig = figures.pooled_correlation_summary(
        decay.mechanism_frame(fits),
        series=DECAY_SUMMARY_SERIES,
        series_labels=decay.SERIES_LABELS,
        series_colors=DECAY_SUMMARY_COLORS,
        series_hatches=REFRESH_HATCHES,
    )
    _emit(fig, "exp2_decay_summary", out_dir, saved)

    series = _present_series(fits)
    # Preserve the original correlation views alongside the forecast-RMSE
    # companions emitted by _forecast_figures below.
    for name, facet in (
        ("exp2_headline", True),
        ("exp2_headline_overlaid", False),
    ):
        fig = figures.headline_curves(
            fits,
            traits=traits,
            trait_labels=trait_labels,
            groups=REFRESH_GROUPS,
            member_labels=decay.PREDICTED_LABELS,
            facet=facet,
            trunks=trunks,
            trunk_labels={trunk: display_trunk_title(trunk) for trunk in trunks},
            trunk_row_labels={trunk: display_trunk_short(trunk) for trunk in trunks},
        )
        _emit(fig, name, out_dir, saved)

    refreshed_axes = (
        refreshed_axes if refreshed_axes is not None else pd.DataFrame()
    )
    checkpoints = decay.mechanism_frame(
        decay.attach_axis_refresh(fits, refreshed_axes)
    )
    fig = figures.mechanism_grid(
        checkpoints,
        mechanism_predictors(source, include_onpolicy=not refreshed_axes.empty),
        traits=traits,
        trait_labels=trait_labels,
        trunk_labels={**labels, "shared": "Shared $M_0$"},
        trunk_colors={**colors, "shared": style.SECONDARY_INK},
    )
    _emit(fig, "exp2_mechanism", out_dir, saved)

    pairs = decay.phase_contrast_frame(fits)
    if pairs.empty:
        logger.info(
            "exp2: no trunk has a re-alignment step with misalignment to undo, "
            "so there is no phase contrast to draw"
        )
    else:
        fig = figures.phase_contrast(
            pairs,
            traits=_present_traits(pairs),
            trait_labels=trait_labels,
            series=series,
            series_labels=decay.SERIES_LABELS,
            trunk_colors=colors,
        )
        _emit(fig, "exp2_phase_contrast", out_dir, saved)
    return saved


def _without_pinned_keys(
    table: pd.DataFrame, keys: Sequence[str]
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Drop the key levels a table holds fixed, and say what they were fixed to.

    A column repeating ``$\\Delta P_0$`` down every row is width a
    twelve-column table has none of, spent on no information. What it said
    still has to be said, so it comes back as ``(heading, value)`` pairs for
    the note above the ``tabular`` -- which is what a caption is written from.

    Driven by what the caller *pinned*, never by what happens to be constant in
    the rows that arrived. A half-finished sweep can leave a single trait or a
    single trunk in the frame, and dropping that column would turn a table
    whose scope is an accident of the sweep into one that looks deliberately
    scoped. The last level is left alone whatever it is: it is the one a block
    turns over on, and a table with no varying key is not a block table.
    """
    if table.empty or table.index.nlevels < 2:
        return table, []
    levels = _level_names(table)
    dropped = [level for level in levels[:-1] if level in set(keys)]
    if not dropped:
        return table, []
    pinned = [
        (
            _KEY_HEADINGS.get(level, level.title()),
            str(table.index.get_level_values(level)[0]),
        )
        for level in dropped
    ]
    return table.droplevel(list[Hashable](dropped)), pinned


@dataclass(frozen=True)
class _ForecastTable:
    """One emitted out-of-sample table: what it scores, over which rows."""

    name: str
    metric: str
    #: Projection series carried, or ``None`` for every one that was measured.
    series: tuple[str, ...] | None
    models: tuple[str, ...]
    #: Key order, whose last entry is what a block compares (see
    #: :func:`method.visualization.forecast.score_table`).
    by: tuple[str, ...]
    scale: _Scale

    @property
    def pinned(self) -> tuple[str, ...]:
        """The key levels this table holds to a single value, so does not print."""
        return tuple(
            level
            for level, chosen in (("series", self.series), ("model", self.models))
            if chosen is not None and len(chosen) == 1
        )


#: The tables :func:`_forecast_figures` writes, and why each is cut the way it is.
FORECAST_TABLES = (
    _ForecastTable(
        "exp2_forecast_rmse",
        "rmse",
        None,
        forecast.HEADLINE_MODELS,
        forecast.BY_MODEL,
        ERROR_SCALE,
    ),
    _ForecastTable(
        "exp2_forecast_bias",
        "bias",
        None,
        ("step0",),
        forecast.BY_SERIES,
        BIAS_SCALE,
    ),
    _ForecastTable(
        "exp2_forecast_correction_rmse",
        "rmse",
        ("p0",),
        forecast.CORRECTION_MODELS,
        forecast.BY_MODEL,
        ERROR_SCALE,
    ),
    _ForecastTable(
        "exp2_forecast_correction_bias",
        "bias",
        ("p0",),
        forecast.CORRECTION_BIAS_MODELS,
        forecast.BY_MODEL,
        BIAS_SCALE,
    ),
    _ForecastTable(
        "exp2_forecast_target_rmse",
        "rmse",
        None,
        ("step0", "step0_level"),
        forecast.BY_MODEL,
        ERROR_SCALE,
    ),
)

def _pinned_note(spec: _ForecastTable, pinned: Sequence[tuple[str, str]]) -> str:
    """What a forecast table holds fixed, for the comment above its ``tabular``.

    Always says which metric the cells are, since a table of bare numbers on a
    judge's scale is unreadable without it, and then whatever
    :func:`_without_pinned_keys` took out.
    """
    parts = [f"{spec.metric.upper()} in judge points"]
    parts += [f"{heading.lower()}: {value}" for heading, value in pinned]
    return "; ".join(parts) + "."


#: Which projection difference the forecast grid draws. One series per grid --
#: the models are already spending the colour channel -- and $\Delta P_0$ is
#: the one the figure is about: everything measured once at the base model,
#: which is the only thing a practitioner who never re-measures has.
FORECAST_GRID_SERIES = "p0"

#: All projection variants, including current-model-generated answers.
FORECAST_RMSE_BAR_SERIES = (
    "p0", "hat_v0", "hat_t", "hat_onpolicy",
    "full_v0", "full_t", "full_onpolicy",
)


def _math_body(label: str) -> str:
    """Strip one pair of math delimiters so a symbol can be composed safely."""
    return label[1:-1] if label.startswith("$") and label.endswith("$") else label


@dataclass(frozen=True)
class _ForecastBar:
    """One bar of the headline RMSE figure: what it is, and how it is drawn."""

    label: str
    color: str


#: What each bar of that figure is, in the order the bars are drawn within a
#: row. The two frozen forecasters differ only in what $f_0$ was fitted to
#: predict, so hue is what separates them. The refit is target-invariant, so it
#: appears once, in grey, as their shared reference.
HEADLINE_FORECAST_BARS = {
    "step0_level": _ForecastBar(r"$f_0$, predicts $b_{t+1}$", style.BLUE),
    "step0": _ForecastBar(r"$f_0$, predicts $\Delta b_{t+1}$", style.ORANGE),
    "oracle": _ForecastBar(r"$f_t$, refit at $t$", style.MUTED),
}


def _headline_forecast_label(series: str, model: str) -> str:
    """Name the fitted map, its projection argument and its target.

    Written out in full for the flat ranking, where a bar has nothing but its
    own label to say which of the fourteen forecasts it is.
    """
    projection = _math_body(decay.SERIES_LABELS.get(series, series))
    fitted = "f_t" if model == "oracle" else "f_0"
    target = {
        "step0": r"  on $\Delta b$",
        "step0_level": r"  on $b_{t+1}$",
        "oracle": "  refit",
    }.get(model, "")
    return rf"${fitted}\!\left({projection}\right)$" + target


def _correction_forecast_label(model: str, source: str) -> str:
    r"""Write a corrected forecast as $f_0(c_t(\cdot)\Delta P_0)$."""
    index = source_index(source)
    if model == "oracle":
        return rf"$f_t\!\left({DELTA_P_BASE}\right)$  refit"
    if model == "step0":
        argument = DELTA_P_BASE
    elif model == "step0_b":
        argument = rf"c_t\!\left(b_t\right)\,{DELTA_P_BASE}"
    elif model == "step0_z":
        argument = rf"c_t\!\left({z_symbol(neutral=index)}\right)\,{DELTA_P_BASE}"
    elif model.removeprefix("step0_") in decay.Z_COMPONENTS:
        symbol = z_component_symbol(model.removeprefix("step0_"), neutral=index)
        argument = rf"c_t\!\left({symbol}\right)\,{DELTA_P_BASE}"
    else:
        return model
    return rf"$f_0\!\left({argument}\right)$"


def _mean_rmse_summary(
    table: pd.DataFrame, *, by: Sequence[str]
) -> pd.DataFrame:
    """Mean and sample SD over the checkpoint-level RMSE cells of a table.

    The current sweep is complete, so the mean is exactly the macro-average of
    the table's six trait--trunk ``Mean`` values.  Computing both statistics
    from the long form makes the SD's unit explicit: variation among the 42
    trait--trunk--checkpoint RMSEs, each of which was itself calculated over
    the eight held-out probes.  It is descriptive heterogeneity across the
    evaluated settings, not an inferential confidence interval.
    """
    if table.empty:
        return pd.DataFrame(columns=[*by, "mean_rmse", "sd_rmse", "n"])
    unknown = set(by).difference(_level_names(table))
    if unknown:
        raise ValueError(f"unknown mean-RMSE index levels: {sorted(unknown)}")
    checkpoint_rmse = table.rename_axis(columns="t").stack().rename("rmse").reset_index()
    return (
        checkpoint_rmse.groupby(list(by), observed=True, sort=False)["rmse"]
        .agg(mean_rmse="mean", sd_rmse="std", n="count")
        .reset_index()
    )


def _headline_rmse_table(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Both targets of the frozen line, with the refit beside them.

    The two targets come from the target table and the refit from the headline
    one, which is where each already lives. The target table leaves the refit
    out on purpose -- it is fitted at the checkpoint, so it is the same
    reference whichever target the frozen line was fitted on, and a table would
    print it twice.
    """
    targets = tables.get("exp2_forecast_target_rmse", pd.DataFrame())
    headline = tables.get("exp2_forecast_rmse", pd.DataFrame())
    if targets.empty or headline.empty:
        return headline
    refit = headline[headline.index.get_level_values("model") == "oracle"]
    return pd.concat([targets, refit])


def _forecast_mean_rmse_figures(
    tables: Mapping[str, pd.DataFrame],
    out_dir: Path,
    saved: list[Path],
    *,
    source: str,
) -> None:
    """Emit the two ranked summaries used in the Appendix."""
    headline = _mean_rmse_summary(
        _headline_rmse_table(tables), by=("series", "model")
    )
    if not headline.empty:
        headline = headline[
            headline["series"].isin(FORECAST_RMSE_BAR_SERIES)
            & headline["model"].isin(HEADLINE_FORECAST_BARS)
        ]
        order = list(HEADLINE_FORECAST_BARS)
        headline = headline.sort_values(
            "model", key=lambda column: column.map(order.index), kind="stable"
        ).assign(
            label=lambda frame: [
                HEADLINE_FORECAST_BARS[str(model)].label for model in frame["model"]
            ],
            color=lambda frame: [
                HEADLINE_FORECAST_BARS[str(model)].color for model in frame["model"]
            ],
            projection=lambda frame: [
                decay.SERIES_LABELS.get(str(series), str(series))
                for series in frame["series"]
            ],
            ranked_label=lambda frame: [
                _headline_forecast_label(str(series), str(model))
                for series, model in zip(frame["series"], frame["model"])
            ],
            upper_bound=lambda frame: frame["model"] == "oracle",
        )
        # Emit grouped and ranked views of the same values.
        fig = figures.mean_rmse_bar(
            headline, group_col="projection", color_col="color"
        )
        _emit(fig, "exp2_forecast_rmse_bar", out_dir, saved)
        fig = figures.mean_rmse_bar(headline, label_col="ranked_label")
        _emit(fig, "exp2_forecast_rmse_bar_ranked", out_dir, saved)

    corrections = _mean_rmse_summary(
        tables.get("exp2_forecast_correction_rmse", pd.DataFrame()),
        by=("model",),
    )
    if not corrections.empty:
        corrections = corrections.assign(
            label=lambda frame: [
                _correction_forecast_label(str(model), source)
                for model in frame["model"]
            ],
            upper_bound=lambda frame: frame["model"] == "oracle",
        )
        fig = figures.mean_rmse_bar(corrections)
        _emit(fig, "exp2_forecast_correction_rmse_bar", out_dir, saved)


#: One forecaster-target row in the recalibration table.
@dataclass(frozen=True)
class _SupertableRow:
    """A forecaster of the recalibration sweep, under one target of $f_0$."""

    target: str
    model: str
    label: str
    #: Which model answered the neutral prompts the state was read off, as
    #: :data:`method.visualization.labels.SOURCE_INDICES` names it. Part of the
    #: key rather than of the table: two rows can name the same forecaster and
    #: differ only here, and that pair is a comparison the table is for.
    source: str = BASE_SOURCE


#: How the key column writes each target. The refit takes neither: within a
#: checkpoint $b_t$ is one constant, so fitting $b_{t+1}$ rather than $\Delta b$
#: moves the intercept by exactly that constant and leaves the predicted level
#: identical.
_LEVEL_TARGET = r"$b_{t+1}$"
_CHANGE_TARGET = r"$\Delta b_{t+1}$"
_ANY_TARGET = "Any"

#: The neutral-response sources the table reads, in the order the rows pair
#: them: text $\mathcal{M}_0$ generated and the checkpoint re-encodes, then
#: text the checkpoint generated itself. Fixed here rather than taken from
#: ``--source``, because the comparison between the two is one of the things
#: the table is for.
SUPERTABLE_SOURCES = (BASE_SOURCE, CURRENT_SOURCE)


#: The states a regenerated neutral response can move, and so the only ones worth a row per source.
_NEUTRAL_DEPENDENT = ("z", "p", "q")


def _state_rows(name: str) -> tuple[tuple[str, str, str], ...]:
    r"""``(model suffix, label, source)`` per source one state can be read off.

    A state appears once per source, immediately below its own twin, because
    the two differ in exactly one thing -- who answered the neutral prompts --
    and what the pair measures is whether the generation pass that buys
    $s = t$ is worth paying for. Adjacent rows put that difference where it can
    be read off the page instead of hunted for.
    """
    sources = (
        SUPERTABLE_SOURCES if name in _NEUTRAL_DEPENDENT else (BASE_SOURCE,)
    )
    return tuple(
        (f"_{name}", _state_label(name, source=source), source)
        for source in sources
    )


def _state_label(name: str, *, source: str = BASE_SOURCE) -> str:
    r"""``$c_t(p_t^{[0]})$``: the recalibration factor over one state."""
    index = source_index(source)
    symbol = (
        z_symbol(neutral=index)
        if name == "z"
        else z_component_symbol(name, neutral=index)
    )
    return rf"$c_t({symbol})$"


def _supertable_rows() -> tuple[_SupertableRow, ...]:
    r"""Every forecaster the sweep measured, under both of $f_0$'s targets."""
    states = (
        ("", r"$c_t\equiv1$", BASE_SOURCE),
        *(row for name in ("z", *decay.Z_COMPONENTS) for row in _state_rows(name)),
        ("_b", r"$c_t(b_t)$", BASE_SOURCE),
    )
    return (
        *(
            _SupertableRow(target, f"step0{suffix}{fitted}", label, source)
            for target, fitted in ((_LEVEL_TARGET, "_level"), (_CHANGE_TARGET, ""))
            for suffix, label, source in states
        ),
        _SupertableRow(_ANY_TARGET, "oracle", r"$f_t$, refit"),
    )


def _supertable(scores: pd.DataFrame) -> pd.DataFrame:
    r"""Mean RMSE per (trait, target, forecaster), one column per projection."""
    rows = _supertable_rows()
    wanted = {(row.model, row.source): row for row in rows}
    kept = scores[
        [key in wanted for key in zip(scores["model"], scores["source"])]
    ].dropna(subset=["rmse"])
    if kept.empty:
        return pd.DataFrame()
    targets = list(dict.fromkeys(row.target for row in rows))
    labels = list(dict.fromkeys(row.label for row in rows))
    picked = [wanted[key] for key in zip(kept["model"], kept["source"])]
    kept = kept.assign(
        target=pd.Categorical(
            [row.target for row in picked], categories=targets, ordered=True
        ),
        forecast=pd.Categorical(
            [row.label for row in picked], categories=labels, ordered=True
        ),
        trait=[display_trait_name(trait) for trait in kept["trait"]],
    )
    table = (
        kept.groupby(["trait", "target", "forecast", "series"], observed=True)["rmse"]
        .mean()
        .unstack("series")
    )
    series = [name for name in decay.SERIES if name in table.columns]
    return table[series].rename(columns=decay.SERIES_LABELS).sort_index()


def _sourced_scores(
    scores: pd.DataFrame,
    source: str,
    others: Mapping[str, pd.DataFrame],
    fan: pd.DataFrame,
) -> pd.DataFrame:
    r"""``scores`` stacked with every other source's, each tagged with its own.

    One decay frame carries one neutral-response source, because the trunk
    holding $\mathbf{z}_t^{[t,0]}$ is a different run from the one holding
    $\mathbf{z}_t^{[0,0]}$ (see
    :func:`method.visualization.decay._with_latent`). The recalibration table
    needs both at once, so they are scored separately and concatenated rather
    than merged before scoring, which would have to pick one source per row and
    lose the comparison.

    ``scores`` is the caller's own, already computed for the source the rest of
    the figures are drawn at; only the other frames are scored here.
    """
    frames = [scores.assign(source=source)]
    for name, rows in others.items():
        predictions = forecast.prediction_frame(rows, fan)
        frames.append(forecast.score_frame(predictions).assign(source=name))
    return pd.concat(frames, ignore_index=True)


def _supertable_blocks(table: pd.DataFrame) -> list[pd.Index]:
    r"""One contest per trait, run across both of $f_0$'s targets.

    What the sweep is asked to settle is which forecaster to build, and the
    target it is fitted to is part of that choice rather than something held
    fixed while the choice is made. Bolding inside each target block would
    answer a question nobody has -- which state to regress the gain on, given
    that the target has already been fixed to whichever of the two is worse --
    and would print two winners per projection where there is one. The trait
    still separates contests, since a persona vector and a judge are per trait
    and the errors are not on the same scale across them.

    The refit is kept out by being made its own block, one row wide: it is the
    ceiling rather than a candidate (see :func:`_supertable_rows`), and a
    block that ties has no lead to mark (see :func:`_leading_cells`).
    """
    trait = table.index.get_level_values("trait")
    target = table.index.get_level_values("target")
    contest = pd.Index(
        [name if name == _ANY_TARGET else "" for name in target], name="target"
    )
    return [trait, contest]


def _forecast_supertable_table(
    scores: pd.DataFrame,
    out_dir: Path,
    saved: list[Path],
) -> None:
    """Emit the recalibration sweep as one appendix table."""
    table = _supertable(scores)
    if table.empty:
        return
    _emit_table(
        _with_mean(table, DECAY_TABLE_MEAN),
        ("Trait", "Target", "Forecast"),
        "Projection difference",
        "exp2_forecast_supertable",
        out_dir,
        saved,
        summary=1,
        scale=ERROR_SCALE,
        note=(
            "RMSE in judge points, averaged over trunks and checkpoints; "
            "each forecaster under both of $f_0$'s targets; "
            "each gain regressed on the state its row names; "
            "bold marks the best forecaster per trait, over both targets"
        ),
        blocks=_supertable_blocks,
    )


def _forecast_figures(
    rows: pd.DataFrame,
    fan: pd.DataFrame,
    out_dir: Path,
    *,
    source: str = "base",
    other_sources: Mapping[str, pd.DataFrame] | None = None,
    headline_rmse_target: str = "matched",
) -> list[Path]:
    r"""The out-of-sample tables and the predicted-against-actual grid.

    The decay figures fit a line at every checkpoint and report how well it
    fits. The tables use leave-one-dataset-family-out $M_0$ predictions. The
    recalibration grid instead draws one line fitted on the 16 non-probe
    validation datasets, since joining predictions from eight leave-one-out
    fits would not be an affine line. Both report errors in judge points: a
    fixed affine map leaves $r$ exactly where :func:`_decay_figures` already
    reported it.

    ``other_sources`` carries the same rows measured under the neutral-response
    sources ``source`` is not, keyed by source name. Only the super table reads
    them: every figure here is drawn at one source, and the one table that
    compares the sources to each other cannot be (see :func:`_sourced_scores`).
    """
    saved: list[Path] = []
    if rows.empty:
        return saved
    if fan.empty:
        logger.warning(
            "exp2: no validation fan on disk, so M_0's line cannot be fitted "
            "and the out-of-sample tables are unavailable. Run the %r family "
            "first",
            experiments.EXP2_VALIDATION,
        )
        return saved

    predictions = forecast.prediction_frame(rows, fan)
    if predictions.empty:
        logger.warning("exp2: nothing to forecast; skipping the out-of-sample tables")
        return saved
    scores = forecast.score_frame(predictions)
    tables: dict[str, pd.DataFrame] = {}

    # The headline keeps the correlation figure's 3x2 visual grammar, but its quantity is out-of-sample error.
    refit = forecast.metric_frame(
        scores, metric="rmse", model="oracle", series=decay.REFRESH_ORDER
    )
    for target in _headline_rmse_targets(headline_rmse_target):
        suffix, model, model_by_series, member_labels = _headline_rmse_spec(target)
        headline = forecast.metric_frame(
            scores,
            metric="rmse",
            model=model,
            model_by_series=model_by_series,
            series=decay.REFRESH_ORDER,
        )
        if headline.empty:
            continue
        traits = _present_traits(headline)
        trunks = _present_trunks(headline)
        trait_labels = {trait: display_trait_name(trait) for trait in traits}
        trunk_titles = {
            trunk: display_trunk_title(trunk) for trunk in trunks
        }
        for layout_suffix, facet in (("", True), ("_overlaid", False)):
            fig = figures.headline_curves(
                headline,
                traits=traits,
                trait_labels=trait_labels,
                groups=REFRESH_GROUPS,
                member_labels=member_labels,
                reference=refit,
                reference_label=HEADLINE_FORECAST_BARS["oracle"].label,
                facet=facet,
                metric="rmse",
                trunks=trunks,
                trunk_labels=trunk_titles,
                trunk_row_labels={
                    trunk: display_trunk_short(trunk) for trunk in trunks
                },
            )
            _emit(
                fig,
                f"exp2_headline_rmse{suffix}{layout_suffix}",
                out_dir,
                saved,
            )

    for spec in FORECAST_TABLES:
        table = forecast.score_table(
            scores,
            spec.metric,
            series=spec.series,
            models=spec.models,
            by=spec.by,
        )
        if table.empty:
            continue
        tables[spec.name] = table
        shown, pinned = _without_pinned_keys(
            _labelled_table(table, source), spec.pinned
        )
        _emit_table(
            _with_mean(shown, DECAY_TABLE_MEAN),
            _headings(_level_names(shown)),
            DECAY_TABLE_SPANNER,
            spec.name,
            out_dir,
            saved,
            summary=1,
            scale=spec.scale,
            note=_pinned_note(spec, pinned),
        )

    _forecast_mean_rmse_figures(tables, out_dir, saved, source=source)
    _forecast_supertable_table(
        _sourced_scores(scores, source, other_sources or {}, fan), out_dir, saved
    )

    stale = predictions[predictions["series"] == FORECAST_GRID_SERIES]
    if stale.empty:
        return saved
    traits = _present_traits(stale)
    trunks = _present_trunks(stale)
    trait_labels = {trait: display_trait_name(trait) for trait in traits}
    trunk_labels = {trunk: display_trunk_name(trunk) for trunk in trunks}
    glosses = {f.name: f.gloss for f in forecast.FORECASTERS}
    projection = decay.SERIES_LABELS[FORECAST_GRID_SERIES]

    # The projection-axis view needs one actual affine M_0 line. Its baseline
    # therefore uses the 16 validation datasets outside the eight probes,
    # separately from the leave-one-out predictions used by the tables and the
    # predicted-against-actual grid below.
    recalibration = forecast.recalibration_frame(
        rows,
        fan,
        probes=[probe.dataset_id for probe in experiments.EXP2_PROBES],
        series=FORECAST_GRID_SERIES,
    )
    fig = figures.recalibration_grid(
        recalibration,
        traits=traits,
        trait_labels=trait_labels,
        trunks=trunks,
        trunk_labels=trunk_labels,
        models=forecast.RECALIBRATION_MODELS,
        model_labels=forecast.RECALIBRATION_LABELS,
        xlabel=rf"Projection difference ${DELTA_P_BASE}$",
    )
    _emit(fig, "exp2_recalibration_grid", out_dir, saved)

    fig = figures.forecast_grid(
        stale,
        traits=traits,
        trait_labels=trait_labels,
        trunks=trunks,
        trunk_labels=trunk_labels,
        models=forecast.FORECAST_GRID_MODELS,
        model_labels=forecast.FORECASTER_LABELS,
        model_glosses=glosses,
        series_label=projection,
    )
    _emit(fig, "exp2_forecast_grid", out_dir, saved)
    return saved


def _probe_series(arm: pd.DataFrame) -> dict[str, list[float]]:
    """One trunk-arm's projection ratios, keyed by probe display name."""
    return {
        display_dataset_name(probe): values
        for probe, values in _series_by(arm, key="probe", value="ratio").items()
    }


def _probe_replicates(arm: pd.DataFrame) -> dict[str, list[list[float]]]:
    r"""The same, one series per replicate seed of each probe."""
    return {
        display_dataset_name(probe): runs
        for probe, runs in _replicates_by(arm, key="probe", value="ratio").items()
    }


def _trunk_mean_std(
    arm: pd.DataFrame, component: str, trunks: Sequence[str]
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """A trunk's seed mean and sample SD for one latent component.

    Every complete seed contributes equally, including the design's original
    seed 0.  A band is returned only when at least two seeds are present; the
    other trunks currently have one seed and therefore remain ordinary lines.
    """
    means: dict[str, list[float]] = {}
    stds: dict[str, list[float]] = {}
    for trunk in trunks:
        pivoted = _pivot_over_t(
            arm[arm["trunk"] == trunk],
            index=["trunk", "seed"],
            value=component,
        )
        if pivoted.empty:
            continue
        label = display_trunk_name(trunk)
        values = pivoted.to_numpy(dtype=float)
        means[label] = values.mean(axis=0).tolist()
        if len(values) > 1:
            stds[label] = values.std(axis=0, ddof=1).tolist()
    return means, stds


def _drift_projection_figure(
    ratios: pd.DataFrame,
    out_dir: Path,
    *,
    quantity: str,
    name: str,
) -> list[Path]:
    """A projection-ratio grid: one trait per row and one trunk per column.

    All six panels share both axes, because the whole reading is comparative:
    how far the probes have drifted under an aggressive schedule against a
    benign one, and whether the two traits go stale at the same rate. Panels on
    their own scales would rescale exactly those differences away.
    """
    saved: list[Path] = []
    if ratios.empty:
        return saved
    traits, trunks = _present_traits(ratios), _present_trunks(ratios)
    own, replicate = _split_by_seed(ratios, experiments.EXP2_SEED)

    def arm(frame: pd.DataFrame, trait: str, trunk: str) -> pd.DataFrame:
        return frame[(frame["trait"] == trait) & (frame["trunk"] == trunk)]

    panels, replicates = {}, {}
    for trait in traits:
        for trunk in trunks:
            cell = (display_trait_name(trait), display_trunk_title(trunk))
            panels[cell] = _probe_series(arm(own, trait, trunk))
            replicates[cell] = _probe_replicates(arm(replicate, trait, trunk))

    fig = figures.overlay_grid(
        panels,
        replicates,
        rows=[display_trait_name(trait) for trait in traits],
        cols=[display_trunk_title(trunk) for trunk in trunks],
        # Keyed by display name because that is what the legend shows; the mark
        # itself is derived from the identifier, so the two stay in step
        # however the display name is spelled.
        marks={
            display_dataset_name(probe): style.dataset_mark(probe)
            for probe in set(ratios["probe"])
        },
        ylabel=rf"{quantity} (% of ${DELTA_P_BASE}$)",
        reference=100.0,
        reference_label=f"${DELTA_P_BASE}$",
        sharey=True,
    )
    _emit(fig, name, out_dir, saved)
    return saved


def _drift_delta_hat_p_figure(ratios: pd.DataFrame, out_dir: Path) -> list[Path]:
    r"""Plot the cached-answer $\Delta P_t^{t\leftarrow0,[0]}$ trajectories."""
    return _drift_projection_figure(
        ratios,
        out_dir,
        quantity=decay.SERIES_LABELS["hat_t"],
        name="exp2_drift_delta_hat_p",
    )


def _drift_delta_p_figure(ratios: pd.DataFrame, out_dir: Path) -> list[Path]:
    r"""Plot the regenerated $\Delta P_t^{t\leftarrow0,[t]}$ trajectories."""
    return _drift_projection_figure(
        ratios,
        out_dir,
        quantity=decay.SERIES_LABELS["full_t"],
        name="exp2_drift_delta_p",
    )


def _drift_latent_figure(
    latents: pd.DataFrame, out_dir: Path, *, source: str = "base"
) -> list[Path]:
    r"""Plot 5, the $z_t$ half: seed means with one-SD bands.

    A fifth column carries $\|h^{\mathrm{neutral}}_t\|$, which is not part of
    z_t but is what disambiguates its first two columns: $p$ and $q$ are
    cosines, so both fall when the neutral state turns off the persona axis
    *and* when it merely grows in unrelated directions. Only the norm beside
    them separates the two. Panels are unshared, so its own scale -- a length
    of order tens, against cosines on $[-1, 1]$ -- costs the other columns
    nothing. Empty where the runs predate :mod:`method.backfill_h_norm`.

    Its two rows are the *same* series, and not by accident: ``h_neutral`` is
    the model's own resting state, so a trunk has one of them however many
    persona axes it is measured against. Only $p$ and $q$ (which the trait's
    $v$ enters) differ by row. Kept per row rather than collapsed to one panel
    so a row stays readable as one trait's whole story.
    """
    saved: list[Path] = []
    if latents.empty:
        return saved
    traits, trunks = _present_traits(latents), _present_trunks(latents)
    colors = {
        display_trunk_name(trunk): style.categorical_color(trunk_index(trunk))
        for trunk in trunks
    }
    columns = drift_z_labels(source)
    panels, bands = {}, {}
    for trait in traits:
        for component, label in columns.items():
            cell = (display_trait_name(trait), label)
            panels[cell], bands[cell] = _trunk_mean_std(
                latents[latents["trait"] == trait], component, trunks
            )

    fig = figures.overlay_grid(
        panels,
        bands=bands,
        rows=[display_trait_name(trait) for trait in traits],
        cols=list(columns.values()),
        colors=colors,
    )
    _emit(fig, "exp2_drift_z", out_dir, saved)
    return saved


# --- experiment 3 ---------------------------------------------------------

#: How far apart two runs' $b_0$ may be before the shared reference line is
#: worth a warning. They read one measurement of one shared base checkpoint, so
#: any spread at all means something is off; a point of judge noise on a 0-100
#: scale is not worth shouting about.
_BASE_BEHAVIOR_TOLERANCE = 1.0


def _base_behavior(
    subset: pd.DataFrame, trait: str, realign_trait: str
) -> float | None:
    r"""$b_0$ for a figure's runs: the level its dashed reference line sits at.

    Every seed shares one base checkpoint (``weights_key`` normalises the seed
    away at $t=0$) and therefore one stored measurement, so this is a mean over
    values that should already be identical. A real spread means the runs were
    measured against different base models -- which would make the reference
    line, and so every "how far above $M_0$" reading taken from the figure,
    quietly wrong -- so it is reported rather than averaged away in silence.
    """
    values = subset["behavior_base"].dropna()
    if values.empty:
        return None
    spread = float(values.max() - values.min())
    if spread > _BASE_BEHAVIOR_TOLERANCE:
        logger.warning(
            "exp3/%s/realign=%s: b_0 differs by %.1f across runs (%.1f to %.1f); "
            "the reference line is their mean, but these runs should share one "
            "base-model measurement -- check they were not collected across two "
            "different base models",
            trait,
            realign_trait,
            spread,
            float(values.min()),
            float(values.max()),
        )
    return float(values.mean())


def _realign_order(trait: str, realign_traits: Sequence[str]) -> list[str]:
    """A trait's own Normal data first, then any other trait's.

    The same-trait re-alignment is the condition the hysteresis claim is about;
    another trait's Normal data is the control that says whether the residue is
    about re-alignment at all or only about that one dataset. Reading the claim
    before its control is why the order is fixed rather than alphabetical.
    """
    return [t for t in realign_traits if t == trait] + [
        t for t in realign_traits if t != trait
    ]


def build_exp3(collection: Collection, out_dir: Path) -> list[Path]:
    r"""The hysteresis bar chart: one row per (measured trait, realign trait).

    Bars are the trait score each arm *ends* at, referenced to a dashed line at
    $M_0$'s own score, so what the eye compares is $b_T - b_0$ -- the same
    origin for every arm. See :func:`~method.visualization.figures.hysteresis_bar`
    for why the last step's $\Delta b$ is not plotted as the level, and why the
    four rows share a figure but only two y-scales.
    """
    saved: list[Path] = []
    if not collection:
        logger.warning("exp3: no runs on disk; skipping")
        return saved

    df = hysteresis_frame(collection)
    present = set(df["condition"])
    conditions = [c for c in HYSTERESIS_CONDITIONS if c in present]
    if len(conditions) < 2:
        logger.warning("exp3: only %d condition(s) present; skipping", len(conditions))
        return saved

    # One row per arm of the 2x2, keyed by both traits at once: the row is a
    # pair, and the frame has a column for each half of it rather than for the
    # pair itself.
    keyed = df.assign(row=df["trait"] + "/" + df["realign_trait"])
    rows, row_labels, row_scales, references = [], {}, {}, {}
    for trait in _present(df["trait"], TRAITS):
        for realign_trait in _realign_order(trait, _present(df["realign_trait"], TRAITS)):
            key = f"{trait}/{realign_trait}"
            subset = keyed[keyed["row"] == key]
            if subset.empty:
                logger.warning(
                    "exp3/%s/realign=%s: no runs on disk; the row is omitted",
                    trait,
                    realign_trait,
                )
                continue
            rows.append(key)
            row_labels[key] = (
                f"{display_trait_name(trait)}\nre-aligned on "
                f"{display_trait_name(realign_trait)}-Normal"
            )
            # The measured trait, so the two re-alignment sources for one trait
            # are read on one scale and the two traits are not.
            row_scales[key] = trait
            base = _base_behavior(subset, trait, realign_trait)
            if base is not None:
                references[key] = base

    fig = figures.hysteresis_bar(
        keyed,
        rows=rows,
        row_col="row",
        row_labels=row_labels,
        row_scales=row_scales,
        conditions=conditions,
        start_col="behavior_before",
        reference=references,
        reference_label=r"Base model $b_0$ (that row's trait)",
        ylabel=r"Trait score after the final step ($b_T$)",
    )
    _emit(fig, "exp3_hysteresis", out_dir, saved)
    saved += build_exp3_training_curves(collection, out_dir)
    return saved


#: The two quantities the training-curve figure stacks, top row first, as
#: :func:`method.visualization.training_curves.curve_bands` names their columns.
#: Loss above gradient because that is the order the argument runs in: the
#: repeat arm already fits the data, and so it pulls less hard on the weights.
TRAINING_CURVE_ROWS = (
    figures.CurveRow("loss_mean", "loss_sd", "Training loss"),
    figures.CurveRow("grad_norm_mean", "grad_norm_sd", "Gradient norm"),
)

#: Column headings of the entry table, keyed to the frame
#: :func:`~method.visualization.training_curves.entry_summary` returns. "Entry"
#: is optimiser step 1, logged while the learning rate is still 0, so it reads
#: the state the run was handed rather than anything this run has done yet.
TRACE_COLUMNS = {
    "loss_init": "Loss (entry)",
    "loss_median": "Loss (median)",
    "grad_norm_init": r"$\lVert g \rVert$ (entry)",
    "grad_norm_median": r"$\lVert g \rVert$ (median)",
}


def _final_step_datasets(collection: Collection) -> dict[str, str]:
    """Trajectory directory name -> the ``dataset/version`` its last step used.

    The join the recovered curves cannot make for themselves: they record the
    content hash of the training examples, which says two runs trained on the
    same data but never which data that was. Taken from each run's config
    rather than parsed out of its directory name, since the name is a
    convention and the config is the fact.
    """
    return {run.path.parent.name: run.label("dataset") for run in collection.runs}


def build_exp3_training_curves(collection: Collection, out_dir: Path) -> list[Path]:
    """The loss and gradient-norm curves of every arm's final fine-tuning step.

    Why the figure exists: the hysteresis bars show that the Same arm ends
    lowest, and this is the measurement of why. See
    :mod:`method.visualization.training_curves` for what the curves are
    recovered from and how they are reduced.

    Skipped, with a warning, where ``data/results/exp3_grad_points.csv`` is
    absent -- it is not version controlled, so a checkout that has every
    trajectory may still have none of the training curves.
    """
    saved: list[Path] = []
    points = training_curves.load_points()
    if points.empty:
        return saved

    run_datasets = _final_step_datasets(collection)
    curves = training_curves.final_step_curves(points, run_datasets)
    if curves.empty:
        logger.warning("exp3: no final-step training curves matched a config; skipping")
        return saved

    # Column order from the collection, not from the curves: this figure is
    # read beside the hysteresis bars, which take theirs from the same place,
    # and two exp3 figures ordering their columns differently would make a
    # reader re-find the dataset they were looking at.
    present = set(curves["dataset"])
    datasets = [d for d in dict.fromkeys(run_datasets.values()) if d in present]
    conditions = [
        condition
        for condition in training_curves.CURVE_CONDITIONS
        if condition in set(curves["condition"])
    ]
    fig = figures.training_curve_grid(
        training_curves.curve_bands(curves),
        rows=TRAINING_CURVE_ROWS,
        datasets=datasets,
        conditions=conditions,
    )
    _emit(fig, "exp3_training_curves", out_dir, saved)

    summary = training_curves.entry_summary(curves)
    table = (
        summary.assign(
            **{
                "Final dataset $X$": summary["dataset"].map(display_dataset_name),
                "Arm": summary["condition"].map(display_condition_name),
                # Both keys ordered as the figure draws them, not
                # alphabetically: the table is the figure's numbers, and a
                # reader checking one against the other should not have to
                # re-find the row.
                "dataset_order": summary["dataset"].map(datasets.index),
                "arm_order": summary["condition"].map(conditions.index),
            }
        )
        .sort_values(["dataset_order", "arm_order"])
        .set_index(["Final dataset $X$", "Arm"])
        .loc[:, list(TRACE_COLUMNS)]
        .rename(columns=TRACE_COLUMNS)
    )
    _emit_table(
        table,
        ["Final dataset $X$", "Arm"],
        "Final fine-tuning step",
        "exp3_training_curves",
        out_dir,
        saved,
        scale=TRACE_SCALE,
        note=(
            "Medians over the distinct training runs of each arm; "
            "entry = optimiser step 1, where the learning rate is still 0."
        ),
    )
    return saved


# --- driver ---------------------------------------------------------------

#: Which figure builder each single-family experiment gets. exp2 is absent
#: because it is not a single family: its figures cross the validation, decay
#: and reseed groups (see :data:`EXP2_GROUPS`), so :func:`build_and_save` hands
#: :func:`build_exp2` all three at once instead.
BUILDERS = {
    experiments.EXP3: build_exp3,
}

#: Every family ``--experiment`` accepts, in run order (the validation fan
#: gates the decay fans, which the reseed replicate checks).
GROUPS = (*EXP2_GROUPS, *BUILDERS)


def build_and_save(
    out_dir: Path,
    *,
    groups: Sequence[str] | None = None,
    local: bool = False,
    mock: bool = False,
    stat: str = "mean",
    source: str = "base",
    sigma_seed: float | None = None,
    n_resamples: int = 2000,
    headline_rmse_target: str = "matched",
) -> list[Path]:
    """Build every requested experiment's figures, returning the files written.

    Asking for any one exp2 family collects all three. They are not
    independent: the decay fans have no ``t = 0`` column without the validation
    family, and the reseed family is only meaningful overlaid on the trunk it
    replicates.
    """
    groups = list(groups) if groups else list(GROUPS)
    if any(group in EXP2_GROUPS for group in groups):
        groups = list(dict.fromkeys([*EXP2_GROUPS, *groups]))
    collections = {
        group: collect_group(group, local=local, mock=mock) for group in groups
    }
    for collection in collections.values():
        logger.info(collection.summary())

    saved: list[Path] = []
    if any(group in collections for group in EXP2_GROUPS):
        # A flat override applies to every trait; measured seed noise is
        # per-trait, so it is only consulted when nothing was passed.
        measured = _sigma_seed(collections)
        traits = {*measured, *collections[experiments.EXP2_DECAY].values("trait")}
        saved += build_exp2(
            collections,
            out_dir / "exp2",
            stat=stat,
            source=source,
            sigma_seed=(
                {trait: sigma_seed for trait in traits}
                if sigma_seed is not None
                else measured
            ),
            n_resamples=n_resamples,
            headline_rmse_target=headline_rmse_target,
        )
    for group, build in BUILDERS.items():
        if group in collections:
            saved += build(collections[group], out_dir / group)
    return saved


def default_out_dir(*, local: bool = False, mock: bool = False) -> Path:
    """One output directory per run source, so figures cannot be confused.

    ``--mock`` and ``--local`` plot entirely different models: fabricated
    artifacts, or a 0.5B proxy, rather than the paper-scale runs. Writing them
    all to ``plots/real`` meant a smoke-test overwrote genuine figures under
    their exact filenames, leaving no way to tell which run produced the file
    you are looking at. Directories keep them apart:
    ``plots/real``, ``plots/real-local``, ``plots/mock``, ``plots/mock-local``.
    """
    name = "mock" if mock else "real"
    return style.PLOTS_DIR / (f"{name}-local" if local else name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=[*GROUPS, "all"],
        default="all",
        help=(
            "which experiment family to plot (default: all). Any one exp2 "
            "family pulls in the other two, which its figures need"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "parent directory to write PNG/PDF figures into, one subdirectory "
            "per experiment family (exp2/exp3) underneath (default: one "
            "per run source -- plots/real, plots/real-local, plots/mock, "
            "plots/mock-local)"
        ),
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="plot the small-model (_local) variants instead of paper-scale",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="plot runs from trajectories-mock/ (produced by --backend mock)",
    )
    parser.add_argument(
        "--stat",
        default="mean",
        help="which DeltaP summary statistic to plot (mean, median, p95, ...)",
    )
    parser.add_argument(
        "--source",
        default="base",
        help="which h_neutral source the z_t components come from",
    )
    parser.add_argument(
        "--sigma-seed",
        type=float,
        default=None,
        help=(
            "fine-tune seed SD of b, for the noise ceiling on exp2's headline "
            "figure (default: read off a seed-swept family via "
            "method.seed_noise; without one the ceiling counts eval noise only)"
        ),
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=2000,
        help="bootstrap resamples behind exp2's correlation and slope intervals",
    )
    parser.add_argument(
        "--headline-rmse-target",
        choices=[*HEADLINE_RMSE_TARGETS, "all"],
        default="matched",
        help=(
            "target policy for exp2 headline RMSE: matched uses b_(t+1) for "
            "cached-answer variants and Delta b_(t+1) for refreshed-answer "
            "variants; change or level uses one target throughout; all writes "
            "all three versions (default: matched)"
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    # fontTools logs a subsetting report per glyph table on every PDF save,
    # which buries this script's own "N runs missing" warnings.
    for noisy in ("fontTools", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    groups = list(GROUPS) if args.experiment == "all" else [args.experiment]
    out_dir = args.out_dir or default_out_dir(local=args.local, mock=args.mock)
    saved = build_and_save(
        out_dir,
        groups=groups,
        local=args.local,
        mock=args.mock,
        stat=args.stat,
        source=args.source,
        sigma_seed=args.sigma_seed,
        n_resamples=args.n_resamples,
        headline_rmse_target=args.headline_rmse_target,
    )
    if not saved:
        logger.error(
            "No figures written. Run the trajectories first, e.g. "
            "poetry run python -m method.run_trajectory --config <NAME>"
        )
        return
    print(f"Wrote {len(saved)} file(s) under {out_dir}:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
