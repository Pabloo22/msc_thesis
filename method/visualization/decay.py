r"""Reduce exp2 trajectories to checkpoint- and probe-level decay tables."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
import pandas as pd

from method import experiments
from method.config import DatasetVersion, StepConfig
from method.latent import H_NORM
from method.noise import delta_b_noise_variance, r2_max
from method.visualization.collect import Collection, Run
from method.visualization.labels import (
    DELTA_P_BASE,
    activation_symbol,
    delta_p_symbol,
    persona_vector_symbol,
)
from method.visualization.metrics import bootstrap_fit

logger = logging.getLogger(__name__)

#: Decay-run roles written by the experiment builder.
TRUNK_ROLE = "trunk"
BRANCH_ROLE = "branch"

#: Projection variants, including the baseline and the 3x2 refresh grid.
SERIES = (
    "p0",
    "hat_v0",
    "hat_t",
    "hat_onpolicy",
    "full_v0",
    "full_t",
    "full_onpolicy",
)
SERIES_LABELS = {
    "p0": f"${DELTA_P_BASE}$",
    "hat_v0": f"${delta_p_symbol(axis='0', predicted='0')}$",
    "hat_t": f"${delta_p_symbol(axis='t', predicted='0')}$",
    "hat_onpolicy": (f"${delta_p_symbol(axis='t', generator='t', predicted='0')}$"),
    "full_v0": f"${delta_p_symbol(axis='0', predicted='t')}$",
    "full_t": f"${delta_p_symbol(axis='t', predicted='t')}$",
    "full_onpolicy": (f"${delta_p_symbol(axis='t', generator='t', predicted='t')}$"),
}

#: Projection variants grouped by persona-vector source.
REFRESH_GROUPS = (
    ("v0", f"${persona_vector_symbol('0', '0')}$", ("hat_v0", "full_v0")),
    ("t", f"${persona_vector_symbol('t', '0')}$", ("hat_t", "full_t")),
    (
        "onpolicy",
        f"${persona_vector_symbol('t', 't')}$",
        ("hat_onpolicy", "full_onpolicy"),
    ),
)

#: Activation labels within each refresh group.
PREDICTED_LABELS = (
    f"${activation_symbol('t', '0')}$",
    f"${activation_symbol('t', 't')}$",
)

#: Flattened refresh-group order.
REFRESH_ORDER = tuple(
    name for _, _, members in REFRESH_GROUPS for name in members
)

#: The ``decay_frame`` column each series is fitted from.
SERIES_COLUMNS = {
    "p0": "delta_p_0",
    "hat_v0": "delta_p_hat_v0",
    "hat_t": "delta_p_hat_t",
    "hat_onpolicy": "delta_p_hat_onpolicy",
    "full_v0": "delta_p_full_v0",
    "full_t": "delta_p_full_t",
    "full_onpolicy": "delta_p_full_onpolicy",
}

#: Latent components in display order.
Z_COMPONENTS = ("p", "q", "rho", "r")

#: Latent components plus activation norm.
LATENT_COLUMNS = (*Z_COMPONENTS, H_NORM)


@dataclass(frozen=True)
class TrunkSeries:
    """One trunk's measurements indexed by checkpoint."""

    trait: str
    trunk: str
    seed: int
    #: Per-checkpoint behaviour, uncertainty, latent state, and probe scores.
    behavior: tuple[float, ...]
    behavior_se: tuple[float, ...]
    latent: tuple[Mapping[str, float], ...]
    probes: tuple[Mapping[str, float], ...]
    #: Consecutive trait-eliciting steps at each checkpoint.
    since: tuple[int, ...]

    @property
    def delta_p_0(self) -> Mapping[str, float]:
        r"""$\Delta P_0$ for every probe: the trunk's own $t = 0$ measurement.

        Taken from the trunk rather than from a pooled lookup because at
        $t = 0$ the axis *is* $v^{(0)}$ and the checkpoint *is* $M_0$, so the
        probe series' first entry is $\Delta P_0$ by construction. All three
        trunks share $M_0$ and therefore agree here; each reading its own keeps
        the per-trunk ratios in :func:`probe_drift_frame` paired.
        """
        return self.probes[0]


def _se(behavior: Mapping[str, float], trait: str) -> float:
    r"""Return $SE(b)$, or NaN when absent."""
    return float(behavior.get(f"{trait}_se", np.nan))


def trunk_series(
    runs: Iterable[Run], *, stat: str = "mean", source: str = "base"
) -> dict[tuple[str, str, int], TrunkSeries]:
    """Index trunks by ``(trait, trunk, seed)``."""
    index: dict[tuple[str, str, int], TrunkSeries] = {}
    for run in runs:
        if run.label("role") != TRUNK_ROLE:
            continue
        steps = run.trajectory.steps
        key = (run.trait, run.label("trunk"), run.seed)
        if key in index:
            logger.warning(
                "two trunk runs share %s (%s and an earlier one); keeping the "
                "first. Trunks are identified by their label, so this means two "
                "configs claim the same one",
                key,
                run.config.name,
            )
            continue
        index[key] = TrunkSeries(
            trait=run.trait,
            trunk=run.label("trunk"),
            seed=run.seed,
            behavior=tuple(s.behavior[run.trait] for s in steps),
            behavior_se=tuple(_se(s.behavior, run.trait) for s in steps),
            latent=tuple(s.z.get(source, {}) for s in steps),
            probes=tuple(
                {
                    ds: summary[stat]
                    for ds, summary in s.probes.items()
                    if stat in summary
                }
                for s in steps
            ),
            since=experiments.steps_since_realignment(run.config.steps),
        )
    return index


def _branch_endpoints(runs: Iterable[Run]) -> dict[tuple[str, str, int, int, str], Run]:
    """Branch runs keyed by ``(trait, trunk, seed, t, probe)``.

    ``t`` is the checkpoint the branch forked from, so its endpoint is
    $b_{t+1}$ -- the label records the fork point rather than the endpoint index
    because that is what pairs it with the trunk measurement it is differenced
    against.
    """
    out: dict[tuple[str, str, int, int, str], Run] = {}
    for run in runs:
        if run.label("role") != BRANCH_ROLE:
            continue
        out[
            (
                run.trait,
                run.label("trunk"),
                run.seed,
                int(run.label("t")),
                run.label("probe"),
            )
        ] = run
    return out


def _validation_endpoints(runs: Iterable[Run]) -> dict[tuple[str, int, str], Run]:
    r"""Index the shared $t = 0$ fan by ``(trait, seed, dataset)``."""
    return {
        (run.trait, run.seed, run.label("dataset")): run
        for run in runs
        if run.label("role") == "validation"
    }


def validation_frame(
    validation: Collection, *, stat: str = "mean"
) -> pd.DataFrame:
    r"""Return one baseline-validation row per fine-tuned dataset."""
    rows = []
    for run in validation.runs:
        steps = run.trajectory.steps
        if len(steps) < 2 or steps[0].delta_p is None:
            continue
        before, after = steps[0], steps[-1]
        delta_p_0 = steps[0].delta_p[stat]
        se_before = _se(before.behavior, run.trait)
        se_after = _se(after.behavior, run.trait)
        rows.append(
            {
                "trait": run.trait,
                "seed": run.seed,
                "dataset": run.label("dataset") or before.next_dataset,
                "delta_p_0": delta_p_0,
                "b_t": before.behavior[run.trait],
                "b_next": after.behavior[run.trait],
                "delta_b": after.behavior[run.trait] - before.behavior[run.trait],
                "se_b_t": se_before,
                "se_b_next": se_after,
                # The two evals draw their generations independently, so their
                # errors add in variance rather than cancelling.
                "se_delta_b": float(np.hypot(se_before, se_after)),
            }
        )
    return pd.DataFrame(rows, columns=_VALIDATION_COLUMNS)


_VALIDATION_COLUMNS = [
    "trait", "seed", "dataset", "delta_p_0", "b_t", "b_next", "delta_b",
    "se_b_t", "se_b_next", "se_delta_b",
]

_DECAY_COLUMNS = [
    "trait", "trunk", "seed", "t", "probe", "steps_since_realignment",
    "delta_p_0", "delta_p_hat_v0", "delta_p_hat_t", "delta_p_hat_onpolicy",
    "delta_p_full_v0", "delta_p_full_t", "delta_p_full_onpolicy",
    "b_t", "b_next", "delta_b",
    "se_b_t", "se_b_next", "se_delta_b", *Z_COMPONENTS,
]


#: Which ``StepRecord`` field each re-measured series is read from, and the
#: ``decay_frame`` column it lands in. A family measuring one of these views
#: records only that view, so its ``probes`` is empty by design and reading the
#: wrong field would give a trunk whose probes all look to have failed.
REMEASURED_FIELDS = {
    "probes_v0": "delta_p_hat_v0",
    "probes_current": "delta_p_full_t",
    "probes_v0_current": "delta_p_full_v0",
    "probes_onpolicy": "delta_p_hat_onpolicy",
    "probes_onpolicy_current": "delta_p_full_onpolicy",
}


def _remeasured_probes(
    runs: Iterable[Run], *, stat: str = "mean"
) -> dict[str, dict[tuple[str, str, int], tuple[Mapping[str, float], ...]]]:
    r"""Re-measured probe series per column, keyed as :func:`trunk_series` is.

    Kept out of :func:`trunk_series` even though it indexes on the same key,
    because a re-measurement run *is* the decay trunk -- same model, same seed,
    same steps, same adapters -- and feeding both through one index would trip
    its duplicate-trunk guard on runs that agree by construction.
    """
    index: dict[str, dict[tuple[str, str, int], tuple[Mapping[str, float], ...]]] = {
        column: {} for column in REMEASURED_FIELDS.values()
    }
    for run in runs:
        if run.label("role") != TRUNK_ROLE:
            continue
        key = (run.trait, run.label("trunk"), run.seed)
        for field, column in REMEASURED_FIELDS.items():
            series = tuple(
                {
                    dataset: summary[stat]
                    for dataset, summary in getattr(step, field).items()
                    if stat in summary
                }
                for step in run.trajectory.steps
            )
            if any(series):
                index[column][key] = series
    return index


def _with_latent(
    trunks: Mapping[tuple[str, str, int], TrunkSeries],
    runs: Iterable[Run],
    *,
    source: str,
) -> dict[tuple[str, str, int], TrunkSeries]:
    r"""``trunks`` with each ``latent`` taken from whichever run carries ``source``.

    Only $z_t$ moves. A family that re-answers the neutral prompts at the
    checkpoint re-runs the whole trunk, so it also holds its own ``probes`` and
    ``behavior``; taking those as well would let ``delta_p_0`` and
    ``delta_p_hat_t`` change between two sources that differ only in who
    answered the *neutral* prompts, and the source comparison would stop being
    a controlled one.

    Runs not carrying ``source`` are dropped before indexing, as
    :func:`latent_frame` does and for the same reason: one trunk is measured by
    two families under one source each, they share a ``(trait, trunk, seed)``
    key, and without the filter the first one seen takes it.

    A trunk no such family covered keeps its own series, which under a source
    it does not carry is a run of empty maps. That is the house rule of
    :func:`decay_frame`: "not measured here" is left as NaN rather than turned
    into a dropped row.
    """
    carrying = [run for run in runs if run.trajectory.has_latent(source)]
    index = trunk_series(carrying, source=source)
    return {
        key: (
            replace(series, latent=_padded(index[key].latent, len(series.probes)))
            if key in index
            else series
        )
        for key, series in trunks.items()
    }


def _padded(
    series: tuple[Mapping[str, float], ...], length: int
) -> tuple[Mapping[str, float], ...]:
    """Trim or pad ``series`` with empty maps to ``length``."""
    return tuple(series[t] if t < len(series) else {} for t in range(length))


def decay_frame(
    decay: Collection,
    validation: Collection | None = None,
    remeasured: Iterable[Collection] = (),
    *,
    neutral: Collection | None = None,
    stat: str = "mean",
    source: str = "base",
) -> pd.DataFrame:
    r"""Return one row per measured ``(trunk, t, probe)`` endpoint.

    Missing branches are omitted; unmeasured projection variants remain NaN.
    """
    trunks = _with_latent(
        trunk_series(decay.runs, stat=stat, source=source),
        [*decay.runs, *(neutral.runs if neutral else ())],
        source=source,
    )
    branches = _branch_endpoints(decay.runs)
    fan_0 = _validation_endpoints(validation.runs if validation else [])
    recomputed = _remeasured_probes(
        [run for collection in remeasured for run in collection.runs], stat=stat
    )

    rows = []
    for (trait, trunk, seed), series in sorted(trunks.items()):
        baseline = series.delta_p_0
        elsewhere = {
            column: index.get((trait, trunk, seed), ())
            for column, index in recomputed.items()
        }
        for t, (probes, since) in enumerate(zip(series.probes, series.since)):
            for probe, delta_p_hat_t in sorted(probes.items()):
                endpoint = branches.get((trait, trunk, seed, t, probe)) or (
                    fan_0.get((trait, seed, probe)) if t == 0 else None
                )
                if endpoint is None or probe not in baseline:
                    continue
                b_t = series.behavior[t]
                se_t = series.behavior_se[t]
                se_next = _se(endpoint.trajectory.steps[-1].behavior, trait)
                rows.append(
                    {
                        "trait": trait,
                        "trunk": trunk,
                        "seed": seed,
                        "t": t,
                        "probe": probe,
                        "steps_since_realignment": since,
                        "delta_p_0": baseline[probe],
                        "delta_p_hat_t": delta_p_hat_t,
                        **{
                            column: (
                                series_t[t].get(probe, np.nan)
                                if t < len(series_t)
                                else np.nan
                            )
                            for column, series_t in elsewhere.items()
                        },
                        "b_t": b_t,
                        "b_next": endpoint.final_behavior(),
                        "delta_b": endpoint.final_behavior() - b_t,
                        "se_b_t": se_t,
                        "se_b_next": se_next,
                        "se_delta_b": float(np.hypot(se_t, se_next)),
                        **{c: series.latent[t].get(c, np.nan) for c in Z_COMPONENTS},
                    }
                )
    return pd.DataFrame(rows, columns=_DECAY_COLUMNS)


def _noise_ceiling(
    group: pd.DataFrame, sigma_seed: float
) -> tuple[float, float, float]:
    r"""Return observed variance, mean noise variance, and $R^2_{max}$."""
    observed = float(group["delta_b"].var(ddof=1))
    noise = float(
        np.mean(
            [
                delta_b_noise_variance(se_t, se_next, sigma_seed)
                for se_t, se_next in zip(group["se_b_t"], group["se_b_next"])
            ]
        )
    )
    if not np.isfinite(observed) or observed <= 0 or not np.isfinite(noise):
        return observed, noise, float("nan")
    return observed, noise, r2_max(observed, noise)


def fit_frame(
    rows: pd.DataFrame,
    *,
    sigma_seed: float = 0.0,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    r"""Collapse each ``(trunk, t)`` scatter to fitted series and a ceiling."""
    if rows.empty:
        return pd.DataFrame(columns=_FIT_COLUMNS)

    records = []
    for key, group in rows.groupby(["trait", "trunk", "t"], sort=True):
        trait, trunk, t = cast(tuple[str, str, int], key)
        observed, noise, ceiling = _noise_ceiling(group, sigma_seed)
        record = {
            "trait": trait,
            "trunk": trunk,
            "t": t,
            "n_probes": len(group),
            "steps_since_realignment": int(group["steps_since_realignment"].iloc[0]),
            "b_t": float(group["b_t"].iloc[0]),
            **{c: float(group[c].iloc[0]) for c in Z_COMPONENTS},
            "var_observed": observed,
            "var_noise": noise,
            "r2_max": ceiling,
        }
        for series in SERIES:
            record.update(
                _series_fit(
                    series,
                    group,
                    n_resamples=n_resamples,
                    level=level,
                    seed=seed,
                )
            )
        records.append(record)
    return pd.DataFrame(records, columns=_FIT_COLUMNS)


def _series_fit(
    series: str,
    group: pd.DataFrame,
    *,
    n_resamples: int,
    level: float,
    seed: int,
) -> dict[str, float]:
    r"""One series' fit over one scatter, or all-NaN where it was not measured.

    $\Delta P_t$ is measured on one trunk only, so most scatters have no column
    to fit. Returning NaN keeps the frame's shape fixed -- every figure indexes
    the same columns whichever families are on disk -- while leaving the
    absence visible: a missing measurement plots as a gap, where a zero would
    plot as a fit that found nothing.

    An incomplete column is treated the same way as an absent one. A fit over
    the subset of probes that happened to be measured would be a correlation
    over a different probe set than the one beside it, which is exactly the
    comparison the whole figure rests on.
    """
    quantities = ("corr", "slope")
    column = SERIES_COLUMNS[series]
    values = group.get(column)
    if values is None or values.isna().any():
        return {
            f"{quantity}_{series}{suffix}": float("nan")
            for quantity in quantities
            for suffix in ("", "_lo", "_hi")
        }
    interval = bootstrap_fit(
        values,
        group["delta_b"],
        n_resamples=n_resamples,
        level=level,
        seed=seed,
    )
    return {
        f"corr_{series}": interval.fit.corr,
        f"corr_{series}_lo": interval.corr_lo,
        f"corr_{series}_hi": interval.corr_hi,
        f"slope_{series}": interval.fit.slope,
        f"slope_{series}_lo": interval.slope_lo,
        f"slope_{series}_hi": interval.slope_hi,
    }


_FIT_COLUMNS = [
    "trait", "trunk", "t", "n_probes", "steps_since_realignment", "b_t",
    *Z_COMPONENTS, "var_observed", "var_noise", "r2_max",
    *[
        f"{quantity}_{series}{suffix}"
        for quantity in ("corr", "slope")
        for series in SERIES
        for suffix in ("", "_lo", "_hi")
    ],
]


#: What :func:`correlation_table` indexes its rows by, outermost key first.
#: The last key is the one that varies inside a block, which is what the
#: emitted table compares and bolds.
CORRELATION_TABLE_KEYS = ("trait", "trunk", "series")


def correlation_table(
    fits: pd.DataFrame, *, series: Sequence[str] | None = None
) -> pd.DataFrame:
    r"""Return $r$ by trait, trunk, series, and checkpoint."""
    wanted = list(series) if series is not None else list(SERIES)
    columns = {
        f"corr_{name}": name
        for name in wanted
        if f"corr_{name}" in fits and fits[f"corr_{name}"].notna().any()
    }
    if fits.empty or not columns:
        return pd.DataFrame()
    long = fits.melt(
        id_vars=["trait", "trunk", "t"],
        value_vars=list(columns),
        var_name="series",
        value_name="corr",
    )
    long["series"] = pd.Categorical(
        long["series"].map(columns), categories=wanted, ordered=True
    )
    table = long.pivot(
        index=list(CORRELATION_TABLE_KEYS), columns="t", values="corr"
    )
    # Empty rows are the series a caller asked for and the sweep never
    # measured; the categorical above keeps them in the ladder's order, and
    # this is where they go.
    return table.dropna(how="all").sort_index()


ONPOLICY_AXIS_COLUMNS = ("rho_onpolicy", "r_onpolicy")


def attach_axis_refresh(fits: pd.DataFrame, refreshed: pd.DataFrame) -> pd.DataFrame:
    r"""Join $\rho_t^{[t]}$ and $r_t^{[t]}$ onto checkpoint-level fits.

    Axis refresh is measured in a separate sweep, so these columns cannot
    enter :func:`fit_frame` through the ordinary trajectory ``z`` block.  The
    join is checkpoint-identical.  At ``t = 0`` the regenerated and frozen
    state variants coincide by definition; copy the canonical frozen values
    there instead of treating an independently sampled extraction redraw as a
    different initial state.
    """
    if fits.empty:
        return fits.assign(**{column: pd.Series(dtype=float) for column in ONPOLICY_AXIS_COLUMNS})

    out = fits.drop(columns=list(ONPOLICY_AXIS_COLUMNS), errors="ignore").copy()
    keys = ["trait", "trunk", "t"]
    if refreshed.empty:
        for column in ONPOLICY_AXIS_COLUMNS:
            out[column] = np.nan
    else:
        values = refreshed[[*keys, *ONPOLICY_AXIS_COLUMNS]].drop_duplicates(
            subset=keys, keep="last"
        )
        out = out.merge(values, on=keys, how="left", validate="many_to_one")

    at_base = out["t"].eq(0)
    out.loc[at_base, "rho_onpolicy"] = out.loc[at_base, "rho"]
    out.loc[at_base, "r_onpolicy"] = out.loc[at_base, "r"]
    return out


def mechanism_frame(fits: pd.DataFrame) -> pd.DataFrame:
    r"""One row per *distinct* checkpoint, for pooled summaries and plot 4.

    The regression of the correlation on drift, on behaviour level and on
    ``steps_since_realignment`` counts checkpoints, not datasets, so ``n`` is
    ``1 shared t=0 + 3 trunks x 6 = 19`` rather than 19 x 8. Dense sampling is
    what buys those rows: measuring only ``t`` in ``{0, 2, 4, 6}`` would give 10.

    The shared $t = 0$ row is the reason this is not just ``fits``. All three
    trunks branch from $M_0$, so :func:`decay_frame` emits its scatter once per
    trunk and :func:`fit_frame` fits three identical copies of it. Letting all
    three into the regression would weight one checkpoint three times and,
    since it is the checkpoint with zero drift, drag the fitted intercept
    toward it.
    """
    if fits.empty:
        return fits
    shared = (
        fits[fits["t"] == 0]
        .drop_duplicates(subset=["trait"])
        # Named rather than left under one trunk's letter, so a legend sorted
        # by this column cannot imply the row belongs to trunk A.
        .assign(trunk="shared")
    )
    return (
        pd.concat([shared, fits[fits["t"] > 0]], ignore_index=True)
        .sort_values(["trait", "trunk", "t"])
        .reset_index(drop=True)
    )


def realignment_pairs(
    drivers: Sequence[StepConfig],
) -> list[tuple[int, int]]:
    r"""Return checkpoint pairs crossing a corrective normal-data step."""
    since = experiments.steps_since_realignment(drivers)
    return [
        (t, t + 1)
        for t, driver in enumerate(drivers)
        if driver.version is DatasetVersion.NORMAL and since[t] > 0
    ]


def phase_contrast_frame(
    fits: pd.DataFrame, trunk_drivers: Mapping[str, Sequence[StepConfig]] | None = None
) -> pd.DataFrame:
    r"""Compare fits immediately before and after each re-alignment."""
    drivers = experiments.EXP2_TRUNKS if trunk_drivers is None else trunk_drivers
    if fits.empty:
        return pd.DataFrame(columns=_PHASE_COLUMNS)

    indexed = {
        (record["trait"], record["trunk"], record["t"]): record
        for record in fits.to_dict("records")
    }
    rows = []
    for trunk, steps in drivers.items():
        for before, after in realignment_pairs(steps):
            for trait in sorted(set(fits["trait"])):
                lo = indexed.get((trait, trunk, before))
                hi = indexed.get((trait, trunk, after))
                if lo is None or hi is None:
                    continue
                row = {
                    "trait": trait,
                    "trunk": trunk,
                    "t_before": before,
                    "t_after": after,
                    "pair": f"{trunk.upper()}: {before}$\\to${after}",
                }
                for series in SERIES:
                    row[f"corr_{series}_before"] = float(lo[f"corr_{series}"])
                    row[f"corr_{series}_after"] = float(hi[f"corr_{series}"])
                    row[f"delta_corr_{series}"] = float(
                        hi[f"corr_{series}"] - lo[f"corr_{series}"]
                    )
                rows.append(row)
    return pd.DataFrame(rows, columns=_PHASE_COLUMNS)


_PHASE_COLUMNS = [
    "trait", "trunk", "t_before", "t_after", "pair",
    *[
        f"{prefix}_{series}{suffix}"
        for prefix, suffix in (
            ("corr", "_before"), ("corr", "_after"), ("delta_corr", "")
        )
        for series in SERIES
    ],
]

#: A probe's $\Delta P_0$ must be at least this fraction of the largest
#: $|\Delta P_0|$ in the set before expressing later checkpoints as a
#: percentage of it says anything. Dividing by a near-zero baseline produces
#: swings of thousands of percent that are a property of the divisor, not of
#: the drift being plotted.
_RATIO_BASELINE_FLOOR = 0.05


def _projection_drift_frame(
    series_by_trunk: Mapping[tuple[str, str, int], Sequence[Mapping[str, float]]],
    *,
    value_column: str,
) -> pd.DataFrame:
    r"""Express one projection view as a percentage of its own $\Delta P_0$.

    ``series_by_trunk`` is keyed by ``(trait, trunk, seed)`` and carries one
    probe map per checkpoint. The caller chooses ``value_column`` so the
    legacy hatted view and the regenerated current-answer view remain
    distinguishable in any frame inspected outside the plotting code.

    A fixed dataset measured against a moving model, so every change is the
    model's. One row per ``(trunk, seed, probe, t)``; retaining the seed lets a
    view with probe-bearing reseeds overlay its replicate trajectories.

    Probes whose $\Delta P_0$ is too near zero to divide by are dropped and
    named in a warning rather than plotted as a spike.
    """
    rows = []
    for (trait, trunk, seed), probes_by_t in sorted(series_by_trunk.items()):
        if not probes_by_t:
            continue
        baseline = probes_by_t[0]
        if not baseline:
            continue
        scale = max(abs(v) for v in baseline.values())
        usable = {
            probe: value
            for probe, value in baseline.items()
            if abs(value) >= _RATIO_BASELINE_FLOOR * scale and value != 0
        }
        dropped = sorted(set(baseline) - set(usable))
        if dropped:
            logger.warning(
                "trunk %s (%s, seed %d): Delta P_0 is too close to zero for %s "
                "to be expressed as a percentage of it; omitting from the "
                "paired-drift plot",
                trunk,
                trait,
                seed,
                ", ".join(dropped),
            )
        for t, probes in enumerate(probes_by_t):
            for probe, value in sorted(usable.items()):
                if probe not in probes:
                    continue
                rows.append(
                    {
                        "trait": trait,
                        "trunk": trunk,
                        "seed": seed,
                        "probe": probe,
                        "t": t,
                        value_column: probes[probe],
                        "ratio": 100.0 * probes[probe] / value,
                    }
                )
    return pd.DataFrame(
        rows,
        columns=["trait", "trunk", "seed", "probe", "t", value_column, "ratio"],
    )


def probe_drift_frame(
    runs: Iterable[Run], *, stat: str = "mean", source: str = "base"
) -> pd.DataFrame:
    r"""The legacy $\Delta \hat{P}_t / \Delta P_0$ trajectory per probe.

    This deliberately reads the decay trunks' unqualified ``probes`` field,
    whose predicted answers are the cached answers from $M_0$ -- so the value
    column is ``delta_p_hat_t``. The refreshed counterpart is a different
    function with a different column (:func:`current_probe_drift_frame`); the
    two are never the same series and no name here lets them be mistaken for
    each other.
    """
    series = {
        key: trunk.probes
        for key, trunk in trunk_series(runs, stat=stat, source=source).items()
    }
    return _projection_drift_frame(series, value_column="delta_p_hat_t")


def current_probe_drift_frame(
    runs: Iterable[Run], *, stat: str = "mean"
) -> pd.DataFrame:
    r"""The regenerated $\Delta P_t / \Delta P_0$ trajectory per probe.

    Only ``probes_current`` records written by ``EXP2_REGEN`` are accepted.
    There is intentionally no fallback to the decay trunks' unqualified
    ``probes``: an absent regenerated measurement must stay absent rather than
    silently turning the new plot into a duplicate of the hatted one.

    At $t=0$, $M_t=M_0$, so the first current-answer probe map is exactly
    $\Delta P_0$ and is the paired denominator for every later checkpoint.
    """
    current = _remeasured_probes(runs, stat=stat)["delta_p_full_t"]
    return _projection_drift_frame(current, value_column="delta_p_full_t")


def latent_frame(
    runs: Iterable[Run], *, stat: str = "mean", source: str = "base"
) -> pd.DataFrame:
    r"""Return per-checkpoint latent state, behaviour, and phase."""
    rows = []
    carrying = [run for run in runs if run.trajectory.has_latent(source)]
    for (trait, trunk, seed), series in sorted(
        trunk_series(carrying, stat=stat, source=source).items()
    ):
        for t, latent in enumerate(series.latent):
            rows.append(
                {
                    "trait": trait,
                    "trunk": trunk,
                    "seed": seed,
                    "t": t,
                    "steps_since_realignment": series.since[t],
                    "b_t": series.behavior[t],
                    "se_b_t": series.behavior_se[t],
                    **{c: latent.get(c, np.nan) for c in LATENT_COLUMNS},
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "trait", "trunk", "seed", "t", "steps_since_realignment", "b_t",
            "se_b_t", *LATENT_COLUMNS,
        ],
    )
