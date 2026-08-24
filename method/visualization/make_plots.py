r"""Generate every figure from *real* saved trajectories and write them to disk.

    poetry run python -m method.visualization.make_plots
    poetry run python -m method.visualization.make_plots --experiment exp3
    poetry run python -m method.visualization.make_plots --mock --local

The counterpart to :mod:`method.visualization.demo`, which draws the
trajectory-shaped figures from synthetic fixtures. Both call the same functions
in :mod:`method.visualization.figures`; only the data source differs. Runs are
found by asking :mod:`method.visualization.collect` which configs the registry
says should exist -- so a partially finished sweep plots what has run and
reports what has not, rather than silently plotting fewer seeds.

exp2's figures, in the design's numbering: the validation fan (1), the decay
scatter grid (2), the headline correlation and slope curves with their noise ceiling
(3), the mechanism regression (4), the phase contrast (4b) and the paired
drift plots (5). Its analysis lives in :mod:`method.visualization.decay`; this
module only chooses what to draw and what to name it.

Every figure panels the measured traits together and is named
``exp2_<figure>``; nothing here is emitted once per trait (see
:func:`build_exp2`). exp3 and exp4 do the same, one figure each, with the
measured trait and the re-alignment source as two dimensions of the grid.

``--local`` selects the small-model variants of each design (the ones a mock
or laptop run produces); without it, the paper-scale configs are used. Each
combination writes to its own directory (see :func:`default_out_dir`), so a
mock smoke test never overwrites a paper-scale figure of the same name. Within
that directory, each experiment family gets its own subdirectory in turn
(``exp2``, ``exp3``, ``exp4``), so ``--experiment exp3`` lands in
``plots/real/exp3`` rather than mixed in with the others.

Note that the figures under ``plots/`` itself are the *synthetic* ones written
by :mod:`method.visualization.demo`. They are drawn from fixtures, not from any
run, so they do not change when trajectories finish -- regenerate them with the
demo module, and read real results from the per-source subdirectories.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pandas as pd

from method import experiments, seed_noise
from method.latent import H_NORM
from method.visualization import decay, figures, style
from method.visualization.collect import (
    Collection,
    collect_group,
    diversity_frame,
    hysteresis_frame,
    seed_noise_frame,
)
from method.visualization.labels import (
    DIVERSITY_CONDITION_LABELS,
    DIVERSITY_CONDITIONS,
    HYSTERESIS_CONDITIONS,
    TRAITS,
    TRUNKS,
    display_dataset_name,
    display_trait_name,
    display_trunk_name,
    display_trunk_title,
    trunk_index,
)

import matplotlib.pyplot as plt  # noqa: E402  (backend fixed by style import)

logger = logging.getLogger("make_plots")

#: Display label -> $z_t$ component, matching :mod:`method.visualization.demo`.
Z_LABELS = {"p": r"$p_t$", "q": r"$q_t$", "rho": r"$\rho_t$", "r": r"$r_t$"}

#: The drift grid's columns: z_t's four, then the length they were normalised
#: by. Separate from :data:`Z_LABELS` because ``h_norm`` is not a component of
#: z_t (see :data:`method.latent.H_NORM`) and the audit figures label z_t's
#: coordinates from that mapping -- a fifth entry there would put a length on
#: axes that only hold the four.
DRIFT_Z_LABELS = {**Z_LABELS, H_NORM: r"$\|h^{\mathrm{neutral}}_t\|$"}


def _emit(fig: plt.Figure, name: str, out_dir: Path, saved: list[Path]) -> None:
    saved.extend(style.save_figure(fig, name, out_dir))
    plt.close(fig)
    logger.info("wrote %s", out_dir / f"{name}.png")


# --- experiment 2: the RQ1 decay experiment --------------------------------

#: The families the decay experiment is split across. They share a base
#: checkpoint and a probe set, and every figure below needs at least two of
#: them, so they are collected together whichever one was asked for.
EXP2_GROUPS = (
    experiments.EXP2_VALIDATION,
    experiments.EXP2_DECAY,
    experiments.EXP2_RESEED,
    experiments.EXP2_AXIS,
    experiments.EXP2_REGEN,
)

#: Which checkpoint-level quantities plot 4 regresses the correlation on. Drift is what
#: RQ1 claims causes decay; $b_t$ and the phase are the two nuisances that
#: would otherwise explain it just as well, which is why the schedules in
#: section 4 are varied enough to tell them apart.
MECHANISM_PREDICTORS = {
    "rho": r"Persona-vector rotation $\rho_t$",
    "r": r"Persona-vector norm $r_t$",
    "b_t": r"Behaviour level $b_t$",
    "steps_since_realignment": "Steps since re-alignment",
}


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
    """Trunks with rows in ``frame``, in the ladder's order (section 4)."""
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


#: Families that sweep seeds over a fixed step sequence, most preferred first.
#: exp3 leads because its arms are the closest analogue of an exp2 branch --
#: several one-step arms, five seeds each -- while exp4's every arm ends in a
#: re-alignment step, so only its first checkpoint is comparable at all.
SEED_NOISE_SOURCES = (experiments.EXP3, experiments.EXP4)


def _sigma_seed(collections: Mapping[str, Collection]) -> dict[str, float]:
    r"""$\sigma_{seed}(b)$ per trait, from whichever collected family sweeps seeds.

    Section 6b's noise ceiling needs the spread a single fine-tune shows when
    it is repeated under another seed, and no run measures its own. exp3 sweeps
    five seeds over fixed sequences, so its one-step arms estimate exactly the
    quantity a branch contributes -- see :mod:`method.seed_noise`, which this
    defers to rather than re-deriving.

    Traits with no multi-seed arm on disk are simply absent, and the caller
    falls back to an eval-noise-only ceiling.
    """
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
) -> list[Path]:
    r"""Every figure of the RQ1 decay experiment, over every measured trait.

    ``collections`` holds the exp2 families keyed by group name. They are
    passed together because the figures cross them: the decay family supplies
    the trunks and their fans, the validation family supplies the shared
    $t = 0$ column that the decay family deliberately does not re-emit, the
    reseed family supplies the paired replicate plot 5 overlays, and the axis
    and regen families supply the two re-measured projection series for
    whichever trunks they covered.

    ``sigma_seed`` maps a trait to its fine-tune seed noise. Where it is
    missing the ceiling on plot 3 accounts for eval noise alone, which makes it
    an upper bound on the true ceiling; the figure is still drawn, and the
    shortfall is logged rather than silently absorbed.

    Every figure panels both traits, none is emitted per trait. Whether
    $\Delta P_0$ goes stale at the same rate for sycophancy and for evil is one
    of the things the experiment is for, and it is not a comparison the reader
    should have to make across two separately scaled figures. What each figure
    does *not* share across the traits is its scale, except where the quantity
    is unitless: a persona vector and a judge are per trait, so a shared
    $\Delta P$ or $\Delta b$ axis would compare numbers that are not the same
    number.
    """
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
    remeasured = [
        collections.get(group) or Collection(group)
        for group in (experiments.EXP2_AXIS, experiments.EXP2_REGEN)
    ]
    if not (decay_runs or validation):
        logger.warning("exp2: no decay or validation runs on disk; skipping")
        return saved

    sigma_seed = dict(sigma_seed or {})
    fan = decay.validation_frame(validation, stat=stat)
    rows = decay.decay_frame(
        decay_runs, validation, remeasured, stat=stat, source=source
    )
    drift_runs = [*decay_runs.runs, *reseed.runs]
    ratios = decay.probe_drift_frame(drift_runs, stat=stat, source=source)
    latents = decay.latent_frame(drift_runs, stat=stat, source=source)

    traits = _present(
        [*validation.values("trait"), *decay_runs.values("trait")], TRAITS
    )
    saved += _validation_figure(fan, traits, out_dir)
    saved += _decay_figures(
        rows, out_dir, sigma_seed=sigma_seed, n_resamples=n_resamples
    )
    saved += _drift_delta_p_figure(ratios, out_dir)
    saved += _drift_latent_figure(latents, out_dir)
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
                "%r family first -- section 10 makes it phase 1 precisely "
                "because it gates everything downstream",
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


def _decay_figures(
    rows: pd.DataFrame,
    out_dir: Path,
    *,
    sigma_seed: Mapping[str, float],
    n_resamples: int,
) -> list[Path]:
    """Plots 2, 3, 4 and 4b, off the same ``(trait, trunk, t, probe)`` rows."""
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
    )
    _emit(fig, "exp2_decay_grid", out_dir, saved)

    for trait in traits:
        if trait in sigma_seed:
            logger.info("exp2/%s: sigma_seed(b) = %.2f", trait, sigma_seed[trait])
        else:
            logger.warning(
                "exp2/%s: no sigma_seed(b) available, so plot 3's noise ceiling "
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
    series = _present_series(fits)
    fig = figures.headline_curves(
        fits,
        traits=traits,
        trait_labels=trait_labels,
        series=series,
        series_labels=decay.SERIES_LABELS,
        trunks=trunks,
        trunk_labels=labels,
        trunk_colors=colors,
    )
    _emit(fig, "exp2_headline", out_dir, saved)

    checkpoints = decay.mechanism_frame(fits)
    fig = figures.mechanism_grid(
        checkpoints,
        MECHANISM_PREDICTORS,
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


def _probe_series(arm: pd.DataFrame) -> dict[str, list[float]]:
    r"""One trunk-arm's $\Delta P_t$ ratios, keyed by probe display name."""
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


def _drift_delta_p_figure(ratios: pd.DataFrame, out_dir: Path) -> list[Path]:
    r"""Plot 5, the $\Delta P_t$ half: a trait per row, a trunk per column.

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
        ylabel=r"$\Delta P_t$ (% of $\Delta P_0$)",
        reference=100.0,
        reference_label=r"$\Delta P_0$",
        sharey=True,
    )
    _emit(fig, "exp2_drift_delta_p", out_dir, saved)
    return saved


def _drift_latent_figure(latents: pd.DataFrame, out_dir: Path) -> list[Path]:
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
    panels, bands = {}, {}
    for trait in traits:
        for component, label in DRIFT_Z_LABELS.items():
            cell = (display_trait_name(trait), label)
            panels[cell], bands[cell] = _trunk_mean_std(
                latents[latents["trait"] == trait], component, trunks
            )

    fig = figures.overlay_grid(
        panels,
        bands=bands,
        rows=[display_trait_name(trait) for trait in traits],
        cols=list(DRIFT_Z_LABELS.values()),
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
    return saved


# --- experiment 4 ---------------------------------------------------------


def build_exp4(collection: Collection, out_dir: Path) -> list[Path]:
    """The diversity bar chart: a measured trait per row, a realign trait per column."""
    saved: list[Path] = []
    if not collection:
        logger.warning("exp4: no runs on disk; skipping")
        return saved

    df = diversity_frame(collection)
    present = set(df["condition"])
    conditions = [c for c in DIVERSITY_CONDITIONS if c in present]
    if len(conditions) < 2:
        logger.warning("exp4: only %d condition(s) present; skipping", len(conditions))
        return saved

    traits = _present(df["trait"], TRAITS)
    realign_traits = _present(df["realign_trait"], TRAITS)
    fig = figures.diversity_bar(
        df,
        rows=traits,
        row_labels={trait: display_trait_name(trait) for trait in traits},
        cols=realign_traits,
        col_labels={
            trait: f"Re-aligned on {display_trait_name(trait)}-Normal"
            for trait in realign_traits
        },
        order=conditions,
        labels=DIVERSITY_CONDITION_LABELS,
        ylabel=r"Residual after re-alignment ($b_T - b_0$)",
    )
    _emit(fig, "exp4_diversity", out_dir, saved)
    return saved


# --- driver ---------------------------------------------------------------

#: Which figure builder each single-family experiment gets. exp2 is absent
#: because it is not a single family: its figures cross the validation, decay
#: and reseed groups (see :data:`EXP2_GROUPS`), so :func:`build_and_save` hands
#: :func:`build_exp2` all three at once instead.
BUILDERS = {
    experiments.EXP3: build_exp3,
    experiments.EXP4: build_exp4,
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
            "per experiment family (exp2/exp3/exp4) underneath (default: one "
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
