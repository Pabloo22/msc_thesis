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
scatter grid (2), the headline $R^2$ and slope curves with their noise ceiling
(3), the mechanism regression (4), the phase contrast (4b) and the paired
drift plots (5). Its analysis lives in :mod:`method.visualization.decay`; this
module only chooses what to draw and what to name it.

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
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from method import experiments, seed_noise
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
    TRUNKS,
    display_dataset_name,
    display_trait_name,
    display_trunk_name,
    trunk_index,
)

import matplotlib.pyplot as plt  # noqa: E402  (backend fixed by style import)

logger = logging.getLogger("make_plots")

#: Display label -> $z_t$ component, matching :mod:`method.visualization.demo`.
Z_LABELS = {"p": r"$p_t$", "q": r"$q_t$", "rho": r"$\rho_t$", "r": r"$r_t$"}


def _emit(fig: plt.Figure, name: str, out_dir: Path, saved: list[Path]) -> None:
    saved.extend(style.save_figure(fig, name, out_dir))
    plt.close(fig)
    logger.info("wrote %s", out_dir / f"{name}.png")


# --- experiment 2: the RQ1 decay experiment --------------------------------

#: The three families the decay experiment is split across. They share a base
#: checkpoint and a probe set, and every figure below needs at least two of
#: them, so they are collected together whichever one was asked for.
EXP2_GROUPS = (
    experiments.EXP2_VALIDATION,
    experiments.EXP2_DECAY,
    experiments.EXP2_RESEED,
)

#: Which checkpoint-level quantities plot 4 regresses $R^2$ on. Drift is what
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


def _present_trunks(frame: pd.DataFrame) -> list[str]:
    """Trunks with rows in ``frame``, in the ladder's order (section 4)."""
    found = set(frame["trunk"])
    return [t for t in TRUNKS if t in found] + sorted(found - set(TRUNKS))


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


def _series_by(
    frame: pd.DataFrame, *, key: str, value: str
) -> dict[str, list[float]]:
    """``key`` -> its ``value`` at each ``t``, ordered by ``t``.

    Keys missing any checkpoint are dropped rather than plotted short, for the
    same reason a ragged seed is: a truncated line reads as a quantity that
    stopped moving, not as a measurement that never happened.
    """
    if frame.empty:
        return {}
    wide = frame.pivot_table(index=key, columns="t", values=value).sort_index(axis=1)
    complete = wide.dropna(axis=0, how="any")
    dropped = sorted(set(wide.index) - set(complete.index))
    if dropped:
        logger.warning(
            "%s missing at some checkpoint for %s; omitted from the drift plot",
            value,
            ", ".join(str(d) for d in dropped),
        )
    return {str(k): [float(v) for v in row] for k, row in complete.iterrows()}


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
    r"""Every figure of the RQ1 decay experiment, one set per measured trait.

    ``collections`` holds the three exp2 families keyed by group name. They are
    passed together because the figures cross them: the decay family supplies
    the trunks and their fans, the validation family supplies the shared
    $t = 0$ column that the decay family deliberately does not re-emit, and the
    reseed family supplies the paired replicate plot 5 overlays.

    ``sigma_seed`` maps a trait to its fine-tune seed noise. Where it is
    missing the ceiling on plot 3 accounts for eval noise alone, which makes it
    an upper bound on the true ceiling; the figure is still drawn, and the
    shortfall is logged rather than silently absorbed.
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
    if not (decay_runs or validation):
        logger.warning("exp2: no decay or validation runs on disk; skipping")
        return saved

    sigma_seed = dict(sigma_seed or {})
    fan = decay.validation_frame(validation, stat=stat)
    rows = decay.decay_frame(decay_runs, validation, stat=stat, source=source)
    drift_runs = [*decay_runs.runs, *reseed.runs]
    ratios = decay.probe_drift_frame(drift_runs, stat=stat, source=source)
    latents = decay.latent_frame(drift_runs, stat=stat, source=source)

    traits = list(
        dict.fromkeys(
            [*validation.values("trait"), *decay_runs.values("trait")]
        )
    )
    for trait in traits:
        prefix = f"exp2_{trait}"
        saved += _validation_figure(fan, trait, prefix, out_dir)
        saved += _decay_figures(
            rows[rows["trait"] == trait],
            trait,
            prefix,
            out_dir,
            sigma_seed=sigma_seed.get(trait),
            n_resamples=n_resamples,
        )
        saved += _drift_figures(
            ratios[ratios["trait"] == trait],
            latents[latents["trait"] == trait],
            prefix,
            out_dir,
        )
    return saved


def _validation_figure(
    fan: pd.DataFrame, trait: str, prefix: str, out_dir: Path
) -> list[Path]:
    """Plot 1: the 24-dataset replication of the persona-vectors correlation."""
    saved: list[Path] = []
    subset = fan[fan["trait"] == trait]
    if subset.empty:
        logger.warning(
            "exp2/%s: no validation runs, so the t=0 fan (plot 1) and the t=0 "
            "column of the decay grid are both unavailable. Run the "
            "%r family first -- section 10 makes it phase 1 precisely because "
            "it gates everything downstream",
            trait,
            experiments.EXP2_VALIDATION,
        )
        return saved
    fig = figures.scatter_validation(
        subset["delta_p_0"],
        subset["delta_b"],
        datasets=list(subset["dataset"]),
        yerr=subset["se_delta_b"],
    )
    _emit(fig, f"{prefix}_validation", out_dir, saved)
    return saved


def _decay_figures(
    rows: pd.DataFrame,
    trait: str,
    prefix: str,
    out_dir: Path,
    *,
    sigma_seed: float | None,
    n_resamples: int,
) -> list[Path]:
    """Plots 2, 3, 4 and 4b, all read off the same ``(trunk, t, probe)`` rows."""
    saved: list[Path] = []
    if rows.empty:
        logger.warning(
            "exp2/%s: no checkpoint has both a Delta P and a branch endpoint; "
            "skipping the decay figures",
            trait,
        )
        return saved

    trunks = _present_trunks(rows)
    colors = _trunk_colors(trunks)
    labels = {t: display_trunk_name(t) for t in trunks}

    fig = figures.decay_scatter_grid(rows, trunks=trunks, trunk_labels=labels)
    _emit(fig, f"{prefix}_decay_grid", out_dir, saved)

    if sigma_seed is None:
        logger.warning(
            "exp2/%s: no sigma_seed(b) available, so plot 3's noise ceiling "
            "counts eval noise only and is an upper bound on the true ceiling. "
            "Run a seed-swept family (exp3) and re-plot, or pass --sigma-seed",
            trait,
        )
    fits = decay.fit_frame(
        rows, sigma_seed=sigma_seed or 0.0, n_resamples=n_resamples
    )
    ceiling_note = (
        rf"eval noise + $\sigma_{{seed}}$={sigma_seed:.2f}"
        if sigma_seed is not None
        else "eval noise only"
    )
    fig = figures.headline_curves(
        fits,
        series_labels=decay.SERIES_LABELS,
        trunks=trunks,
        trunk_labels=labels,
        trunk_colors=colors,
        ceiling_label=rf"$R^2_{{max}}$ ({ceiling_note})",
    )
    _emit(fig, f"{prefix}_headline", out_dir, saved)

    checkpoints = decay.mechanism_frame(fits)
    fig = figures.mechanism_grid(
        checkpoints,
        MECHANISM_PREDICTORS,
        trunk_labels={**labels, "shared": "Shared $M_0$"},
        trunk_colors={**colors, "shared": style.SECONDARY_INK},
    )
    _emit(fig, f"{prefix}_mechanism", out_dir, saved)

    pairs = decay.phase_contrast_frame(fits)
    if pairs.empty:
        logger.info(
            "exp2/%s: no trunk has a re-alignment step with misalignment to "
            "undo, so there is no phase contrast to draw",
            trait,
        )
    else:
        fig = figures.phase_contrast(
            pairs,
            series_labels=decay.SERIES_LABELS,
            trunk_colors=colors,
        )
        _emit(fig, f"{prefix}_phase_contrast", out_dir, saved)
    return saved


def _drift_figures(
    ratios: pd.DataFrame,
    latents: pd.DataFrame,
    prefix: str,
    out_dir: Path,
) -> list[Path]:
    r"""Plot 5: paired drift of $\Delta P_t$ and of $z_t$, with the reseed overlay."""
    saved: list[Path] = []
    primary_seed = experiments.EXP2_SEED

    for trunk in _present_trunks(ratios) if not ratios.empty else []:
        arm = ratios[ratios["trunk"] == trunk]
        own, replicate = _split_by_seed(arm, primary_seed)
        series = {
            display_dataset_name(k): v
            for k, v in _series_by(own, key="probe", value="ratio").items()
        }
        if not series:
            continue
        fig = figures.overlay_lines(
            series,
            {
                display_dataset_name(k): v
                for k, v in _series_by(replicate, key="probe", value="ratio").items()
            },
            # Keyed by display name because that is what the legend shows; the
            # mark itself is derived from the identifier, so the two stay in
            # step however the display name is spelled.
            marks={
                display_dataset_name(probe): style.dataset_mark(probe)
                for probe in set(arm["probe"])
            },
            ylabel=r"$\Delta P_t$ (% of $\Delta P_0$)",
            reference=100.0,
            reference_label=r"$\Delta P_0$",
        )
        _emit(fig, f"{prefix}_drift_delta_p_trunk_{trunk}", out_dir, saved)

    if latents.empty:
        return saved
    trunks = _present_trunks(latents)
    colors = {display_trunk_name(t): style.categorical_color(trunk_index(t))
              for t in trunks}
    own, replicate = _split_by_seed(latents, primary_seed)
    panels = {
        label: {
            display_trunk_name(trunk): values
            for trunk in trunks
            for values in _series_by(
                own[own["trunk"] == trunk], key="trunk", value=component
            ).values()
        }
        for component, label in Z_LABELS.items()
    }
    replicates = {
        label: {
            display_trunk_name(trunk): values
            for trunk in trunks
            for values in _series_by(
                replicate[replicate["trunk"] == trunk], key="trunk", value=component
            ).values()
        }
        for component, label in Z_LABELS.items()
    }
    fig = figures.overlay_grid(panels, replicates, colors=colors)
    _emit(fig, f"{prefix}_drift_z", out_dir, saved)
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


def build_exp3(collection: Collection, out_dir: Path) -> list[Path]:
    r"""Hysteresis bar chart, one figure per (measured trait, realign trait).

    Bars are the trait score each arm *ends* at, referenced to a dashed line at
    $M_0$'s own score, so what the eye compares is $b_T - b_0$ -- the same
    origin for every arm. See :func:`~method.visualization.figures.hysteresis_bar`
    for why the last step's $\Delta b$ is not plotted as the level.
    """
    saved: list[Path] = []
    if not collection:
        logger.warning("exp3: no runs on disk; skipping")
        return saved

    df = hysteresis_frame(collection)
    for trait in collection.values("trait"):
        pretty = display_trait_name(trait)
        for realign_trait in sorted(set(df["realign_trait"])):
            subset = df[
                (df["trait"] == trait) & (df["realign_trait"] == realign_trait)
            ]
            present = set(subset["condition"])
            conditions = [c for c in HYSTERESIS_CONDITIONS if c in present]
            if len(conditions) < 2:
                logger.warning(
                    "exp3/%s/realign=%s: only %d condition(s) present; skipping",
                    trait,
                    realign_trait,
                    len(conditions),
                )
                continue
            fig = figures.hysteresis_bar(
                subset,
                conditions=conditions,
                start_col="behavior_before",
                reference=_base_behavior(subset, trait, realign_trait),
                reference_label=rf"Base model $b_0$ ({pretty})",
                ylabel=rf"{pretty} score after the final step ($b_T$)",
            )
            _emit(fig, f"exp3_{trait}_realign-{realign_trait}", out_dir, saved)
    return saved


# --- experiment 4 ---------------------------------------------------------


def build_exp4(collection: Collection, out_dir: Path) -> list[Path]:
    """Diversity bar chart, one figure per (measured trait, realign trait)."""
    saved: list[Path] = []
    if not collection:
        logger.warning("exp4: no runs on disk; skipping")
        return saved

    df = diversity_frame(collection)
    for trait in collection.values("trait"):
        pretty = display_trait_name(trait)
        for realign_trait in collection.values("realign_trait"):
            subset = df[
                (df["trait"] == trait) & (df["realign_trait"] == realign_trait)
            ]
            present = set(subset["condition"])
            conditions = [c for c in DIVERSITY_CONDITIONS if c in present]
            if len(conditions) < 2:
                logger.warning(
                    "exp4/%s/realign=%s: only %d condition(s) present; skipping",
                    trait,
                    realign_trait,
                    len(conditions),
                )
                continue
            fig = figures.diversity_bar(
                subset,
                order=conditions,
                labels=DIVERSITY_CONDITION_LABELS,
                ylabel=rf"{pretty} residual after re-alignment ($b_T - b_0$)",
            )
            _emit(fig, f"exp4_{trait}_realign-{realign_trait}", out_dir, saved)
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
        help="bootstrap resamples behind exp2's R^2 and slope intervals",
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
