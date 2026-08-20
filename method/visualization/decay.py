r"""The RQ1 decay analysis: exp2's runs reduced to the tables its figures plot.

The experiment measures at two levels and the figures need both, so this
module produces both rather than one flattened table:

*Level 1 -- inside a checkpoint.* :func:`decay_frame` gives one row per
``(trunk, t, probe)``: the projection difference the probe dataset had at
$M_0$, the one it has at $M_t$, and the behaviour change fine-tuning on it
actually produced. Eight such rows are one scatter panel, and fitting them
yields **one** correlation.

*Level 2 -- across checkpoints.* :func:`fit_frame` collapses each of those
scatters to a single row: its correlation and slope with bootstrap intervals, the
noise ceiling those eight points could not have exceeded, and the checkpoint's
own drift and behaviour level. This is the frame the headline curve and the
mechanism regression are drawn from, and its ``n`` counts **checkpoints, not
datasets** (section 9, plot 4).

Reads nothing but ``trajectory.json`` files by way of
:mod:`method.visualization.collect`, so the whole analysis runs on a laptop
holding no adapters and no activations.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from method import experiments
from method.config import DatasetVersion, StepConfig
from method.noise import delta_b_noise_variance, r2_max
from method.visualization.collect import Collection, Run
from method.visualization.metrics import bootstrap_fit

logger = logging.getLogger(__name__)

#: The two roles a decay run plays, as written into its ``role`` label by
#: :func:`method.experiments.build_exp2_decay_configs`. A trunk is measured at
#: every checkpoint and advances the trajectory; a branch is one fine-tune off
#: a trunk checkpoint, scored once and discarded.
TRUNK_ROLE = "trunk"
BRANCH_ROLE = "branch"

#: The projection differences the figures pair, in the order they form a
#: ladder: each refreshes one more thing at $M_t$ than the one before it.
#:
#: The notation carries one mark per choice. A **hat** says the predicted term
#: is *approximated* by $M_0$'s answers rather than being what the checkpoint
#: would really say; a $v_0$ **superscript** says the axis is held at the base
#: model's persona vector; the **subscript** is the checkpoint the activations
#: come from. So $\Delta P_t$ -- unmarked -- is the projection difference the
#: update actually faces, and everything else is an approximation of it.
#:
#: ``p0``
#:     $\Delta P_0$, everything read at $M_0$. This is the quantity the
#:     persona-vectors paper computes, and it needs no hat: at $t = 0$ the
#:     current model *is* $M_0$, so nothing about it is approximated.
#: ``pv0``
#:     $\Delta \hat{P}_t^{v_0}$, the checkpoint's activations against the base
#:     model's axis. Free to measure -- the activations do not depend on the
#:     axis -- and it is what separates the persona direction rotating from the
#:     representation drifting.
#: ``pt``
#:     $\Delta \hat{P}_t$, axis and encoder current, answers still $M_0$'s.
#:     The quantity whose staleness is RQ1.
#: ``p``
#:     $\Delta P_t$, nothing approximated.
#:
#: Only the middle two need a family of their own to measure, so their columns
#: are NaN wherever that family did not run -- which every consumer here treats
#: as "not measured", never as zero.
SERIES = ("p0", "pv0", "pt", "p")
SERIES_LABELS = {
    "p0": r"$\Delta P_0$",
    "pv0": r"$\Delta \hat{P}_t^{v_0}$",
    "pt": r"$\Delta \hat{P}_t$",
    "p": r"$\Delta P_t$",
}

#: The ``decay_frame`` column each series is fitted from.
SERIES_COLUMNS = {
    "p0": "delta_p_0",
    "pv0": "delta_p_v0",
    "pt": "delta_p_t",
    "p": "delta_p",
}

#: $z_t$ components, in the order the proposal introduces them.
Z_COMPONENTS = ("p", "q", "rho", "r")


@dataclass(frozen=True)
class TrunkSeries:
    """One trunk's own measurements, indexed by checkpoint.

    Everything here is read off the trunk, never off a branch: section 8 gives
    a branch endpoint ``b`` alone, on the grounds that $z_t$ and $\\Delta P_t$
    describe the checkpoint the branch *left from*, which the trunk has already
    measured.
    """

    trait: str
    trunk: str
    seed: int
    #: Per checkpoint: the trait score, its analytic standard error (section
    #: 6a), the latent state, and $\Delta P_t$ for every probe dataset.
    behavior: tuple[float, ...]
    behavior_se: tuple[float, ...]
    latent: tuple[Mapping[str, float], ...]
    probes: tuple[Mapping[str, float], ...]
    #: ``steps_since_realignment`` at each checkpoint (section 4).
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
    r"""$SE(b)$ from a behaviour record, or NaN if it predates section 6a.

    Checkpoints measured before the standard error was recorded carry no such
    key. NaN rather than 0.0, so the noise ceiling those checkpoints feed is
    visibly absent instead of quietly reading as "measured perfectly"; run
    :mod:`method.backfill_se` to fill them in.
    """
    return float(behavior.get(f"{trait}_se", np.nan))


def trunk_series(
    runs: Iterable[Run], *, stat: str = "mean", source: str = "base"
) -> dict[tuple[str, str, int], TrunkSeries]:
    """Index every trunk in ``runs`` by ``(trait, trunk, seed)``.

    Seed is part of the key rather than filtered out because the section 6c
    replicate is the same trunk under another seed: keyed this way, trunk A and
    A' fall out as two entries of one index and the reseed comparison needs no
    plumbing of its own.
    """
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
    r"""The $t = 0$ fan keyed by ``(trait, seed, dataset)``.

    All three trunks share $M_0$, so the decay family fans out only from
    $t \ge 1$ and the $t = 0$ branches come from the validation family instead
    -- where they exist for all 24 datasets rather than just the 8 probes
    (section 5).
    """
    return {
        (run.trait, run.seed, run.label("dataset")): run
        for run in runs
        if run.label("role") == "validation"
    }


def validation_frame(
    validation: Collection, *, stat: str = "mean"
) -> pd.DataFrame:
    r"""Section 5's validation fan: one row per dataset fine-tuned from $M_0$.

    The x-axis is each dataset's own $\Delta P_0$, measured at $t = 0$ for the
    dataset that run is about to train on, and the y-axis the behaviour change
    that fine-tune produced. Reproducing Figure 8 of the persona-vectors paper
    over these 24 points is the gate the rest of the design hangs on.

    Kept apart from :func:`decay_frame` deliberately. A correlation over 24
    datasets is not comparable with one over the 8 probes -- such estimates are
    sensitive to range restriction and to ``n`` -- so reporting a fall from one
    to the other would manufacture a decay that is pure artifact.
    """
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
    "delta_p_0", "delta_p_v0", "delta_p_t", "delta_p", "b_t", "b_next", "delta_b",
    "se_b_t", "se_b_next", "se_delta_b", *Z_COMPONENTS,
]


#: Which ``StepRecord`` field each re-measured series is read from, and the
#: ``decay_frame`` column it lands in. A family measuring one of these views
#: records only that view, so its ``probes`` is empty by design and reading the
#: wrong field would give a trunk whose probes all look to have failed.
REMEASURED_FIELDS = {"probes_v0": "delta_p_v0", "probes_current": "delta_p"}


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


def decay_frame(
    decay: Collection,
    validation: Collection | None = None,
    remeasured: Iterable[Collection] = (),
    *,
    stat: str = "mean",
    source: str = "base",
) -> pd.DataFrame:
    r"""One row per ``(trunk, t, probe)``: the unit of analysis of section 3.

    Each row pairs a projection difference with the behaviour change it was
    meant to predict, so the ``K`` rows sharing a ``(trunk, t)`` are one scatter
    panel of section 9's plot 2 and one fitted correlation of its plot 3.

    Both projection differences travel together. ``delta_p_0`` is read from the
    trunk's own $t = 0$ probe measurement and ``delta_p_t`` from checkpoint
    ``t``; they are the blue and orange series of every decay scatter, and the
    whole hypothesis is that the first decays while the second does not.

    ``validation`` supplies the ``t = 0`` branch endpoints, which the decay
    family does not emit: all three trunks share $M_0$, so fanning out from it
    three times over would train the same eight models three times. Those rows
    are duplicated across trunks here, which is exactly what section 9's plot 2
    expects -- "the ``t = 0`` column is identical across rows because $M_0$ is
    shared".

    ``remeasured`` supplies the other two, ``delta_p_v0`` and ``delta_p``: the
    same probes at the same checkpoints, read against the base model's axis
    (:func:`method.experiments.build_exp2_axis_configs`) and with the
    checkpoint answering the prompts itself
    (:func:`method.experiments.build_exp2_regen_configs`). They are joined in
    rather than read from the trunk because they are *re-measurements* of
    trunks that already exist, each paid for by its own family. A row no such
    family covered keeps a NaN, which is the distinction the columns have to
    preserve -- "not measured here" is not "measured and small" -- so
    :func:`fit_frame` declines to fit a series it cannot see.

    Rows whose branch never ran are dropped rather than carried as NaN: a
    partly-finished fan should narrow the scatter it can draw, not poison the
    variance of the one it can. The reseed trunk therefore contributes no rows
    at all (it has no branches by design) and reaches the figures through
    :func:`probe_drift_frame` and :func:`latent_frame` instead.
    """
    trunks = trunk_series(decay.runs, stat=stat, source=source)
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
            for probe, delta_p_t in sorted(probes.items()):
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
                        "delta_p_t": delta_p_t,
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
    r"""``(observed variance, mean noise variance, R^2_max)`` for one scatter.

    Section 6b: noise in $\Delta b$ cannot be explained by $\Delta P$, so it
    caps the attainable $R^2$ whatever the predictor. Drawing the decay curve
    without this line confuses "the probe went stale" -- a finding -- with
    "the scatter was mostly noise all along", which is not.

    ``sigma_seed`` is averaged in per point because the eval terms are
    heteroscedastic by design: near the floor or ceiling of the 0-100 scale a
    question's generations agree exactly, while mid-range the model is
    genuinely bimodal.
    """
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
    r"""Collapse each ``(trunk, t)`` scatter to one row: its fit and its ceiling.

    Every series in :data:`SERIES` is fitted, because section 9 reads them
    against each other: $\Delta P_0$ falling *while* $\Delta P_t$ holds is
    staleness, and both falling together is signal running out. $\Delta P$
    settles what is left over -- if refreshing the prediction as well as the
    axis does not recover the fit, what the frozen series lost was not a stale
    prediction. It is fitted only where it was measured (see
    :func:`_series_fit`).

    Slope is reported beside the correlation rather than folded into it.
    Staleness can appear as attenuation -- the ordering across probe datasets
    stays right while the magnitude shrinks -- which leaves $r$ high and the
    slope low, and the two imply different fixes.

    The goodness of fit is carried as the signed correlation, which is what
    every figure of section 9 plots. ``r2_max`` stays in $R^2$ units because it
    is a variance ratio and is not drawn on any of them; take its square root
    before comparing it against a correlation (see ``docs/r2_max.md``).

    ``sigma_seed`` is the fine-tune seed noise from :mod:`method.seed_noise`.
    Left at zero the ceiling accounts for eval noise only and is therefore an
    upper bound on the true ceiling; the caller should say which it plotted.
    """
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

    $\Delta P$ is measured on one trunk only, so most scatters have no column
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


def mechanism_frame(fits: pd.DataFrame) -> pd.DataFrame:
    r"""One row per *distinct* checkpoint, for section 9's plot 4.

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
    r"""Checkpoint pairs straddling a re-alignment step, as section 4b defines them.

    A pair is ``(t, t+1)`` where the driver applied between them is a Normal
    dataset *and* the model entered it with misalignment to undo -- i.e. its
    ``steps_since_realignment`` is above zero. Trunk A's ``X N X N X N`` gives
    ``(1,2), (3,4), (5,6)`` and trunk B's ``X X N X X N`` gives ``(2,3), (5,6)``,
    matching the section verbatim.

    The second clause is what excludes trunk C. Every one of its drivers is
    Normal, so the first clause alone would call all six of its steps
    re-alignments -- but a step that re-aligns a model which was never
    misaligned isolates nothing, and reporting the resulting $\Delta r$ beside
    A's and B's would invite reading noise as a null result.
    """
    since = experiments.steps_since_realignment(drivers)
    return [
        (t, t + 1)
        for t, driver in enumerate(drivers)
        if driver.version is DatasetVersion.NORMAL and since[t] > 0
    ]


def phase_contrast_frame(
    fits: pd.DataFrame, trunk_drivers: Mapping[str, Sequence[StepConfig]] | None = None
) -> pd.DataFrame:
    r"""Section 9's plot 4b: the fit immediately before and after each re-alignment.

    Trunk and probe set are held fixed within a pair and exactly one known
    driver separates the two checkpoints, so the difference isolates what a
    single re-alignment step does to predictive accuracy -- rather than the
    trend over ``t``, which confounds it with everything else that accumulated.

    Every series is carried. $\Delta P_0$ answers "does re-aligning the model
    restore the *stale* probe's accuracy", which is the emergent-re-alignment
    question; $\Delta P_t$ and $\Delta P$ are the controls -- if they move by
    the same amount, what changed is the scatter, not the staleness. A series
    that was not measured on a trunk carries NaN through to its bars, which go
    undrawn rather than reading as no change.
    """
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


def probe_drift_frame(
    runs: Iterable[Run], *, stat: str = "mean", source: str = "base"
) -> pd.DataFrame:
    r"""Section 9's plot 5: $\Delta P_t / \Delta P_0$ per probe, over ``t``.

    A fixed dataset measured against a moving model, so every change is the
    model's. One row per ``(trunk, seed, probe, t)`` -- seed included because
    the section 6c replicate is the same trunk reseeded, and overlaying A
    against A' is what shows whether the drift trajectory is stable at all.

    Probes whose $\Delta P_0$ is too near zero to divide by are dropped and
    named in a warning rather than plotted as a spike.
    """
    rows = []
    for (trait, trunk, seed), series in sorted(
        trunk_series(runs, stat=stat, source=source).items()
    ):
        baseline = series.delta_p_0
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
        for t, probes in enumerate(series.probes):
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
                        "delta_p_t": probes[probe],
                        "ratio": 100.0 * probes[probe] / value,
                    }
                )
    return pd.DataFrame(
        rows, columns=["trait", "trunk", "seed", "probe", "t", "delta_p_t", "ratio"]
    )


def latent_frame(
    runs: Iterable[Run], *, stat: str = "mean", source: str = "base"
) -> pd.DataFrame:
    r"""Per-checkpoint $z_t$, $b_t$ and phase for every trunk, including reseeds.

    The other half of section 9's plot 5, and the drift axis the mechanism
    regression is read against. Kept in raw units: $\rho$ starts at 1 and $r$ at
    the persona vector's norm, but $p$ and $q$ start at essentially zero on the
    base model, and a "% of step 0" reading of those is division by noise.
    """
    rows = []
    for (trait, trunk, seed), series in sorted(
        trunk_series(runs, stat=stat, source=source).items()
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
                    **{c: latent.get(c, np.nan) for c in Z_COMPONENTS},
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "trait", "trunk", "seed", "t", "steps_since_realignment", "b_t",
            "se_b_t", *Z_COMPONENTS,
        ],
    )
