r"""Generate every figure from *real* saved trajectories and write them to disk.

    poetry run python -m method.visualization.make_plots
    poetry run python -m method.visualization.make_plots --experiment exp3
    poetry run python -m method.visualization.make_plots --local --out-dir plots/local

The counterpart to :mod:`method.visualization.demo`, which draws the same
figures from synthetic fixtures. Both call the same functions in
:mod:`method.visualization.figures`; only the data source differs. Runs are
found by asking :mod:`method.visualization.collect` which configs the registry
says should exist -- so a partially finished sweep plots what has run and
reports what has not, rather than silently plotting fewer seeds.

``--local`` selects the small-model variants of each design (the ones a mock
or laptop run produces); without it, the paper-scale configs are used.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from method import experiments
from method.visualization import figures, schema, style
from method.visualization.collect import (
    Collection,
    collect_group,
    delta_p_0_lookup,
    diversity_frame,
    hysteresis_frame,
    missing_delta_p_0,
    projection_frame,
)
from method.visualization.labels import (
    DIVERSITY_CONDITION_LABELS,
    DIVERSITY_CONDITIONS,
    HYSTERESIS_CONDITION_LABELS,
    HYSTERESIS_CONDITIONS,
    display_dataset_name,
    display_trait_name,
)
from method.visualization.metrics import ratio_percent, stack_and_trim

import matplotlib.pyplot as plt  # noqa: E402  (backend fixed by style import)

logger = logging.getLogger("make_plots")

#: Display label -> $z_t$ component, matching :mod:`method.visualization.demo`.
Z_LABELS = {"p": r"$p_t$", "q": r"$q_t$", "rho": r"$\rho_t$", "r": r"$r_t$"}


def _emit(fig: plt.Figure, name: str, out_dir: Path, saved: list[Path]) -> None:
    saved.extend(style.save_figure(fig, name, out_dir))
    plt.close(fig)
    logger.info("wrote %s", out_dir / f"{name}.png")


#: A component's step-0 value must be at least this fraction of the series'
#: largest magnitude before "% of step 0" is worth plotting.
_PCT_BASELINE_FLOOR = 0.05


def _percent_safe(runs: list[list[float]]) -> bool:
    r"""Whether "% of step-0 value" is meaningful for this quantity.

    $\rho_t$ and $r_t$ start at a solidly non-zero value (1, and the persona
    vector's norm), so a percentage reads naturally. $p_t$ and $q_t$ are
    projections that start at essentially zero on the base model, and dividing
    by that produces swings of thousands of percent that say nothing about the
    dynamics -- those are plotted in raw units instead.
    """
    try:
        matrix = stack_and_trim(runs)
    except ValueError:
        return False
    scale = float(np.abs(matrix).max())
    baseline = float(np.abs(matrix[:, 0]).min())
    return scale > 0 and baseline >= _PCT_BASELINE_FLOOR * scale


def _seed_step_matrix(pairs: pd.DataFrame, value_col: str) -> np.ndarray | None:
    """``[n_seeds, n_steps]`` matrix of ``value_col``, or None if unusable.

    Seeds missing any step are dropped rather than averaged over, because
    :func:`~method.visualization.metrics.mean_std` propagates NaN into both the
    mean line and the band. Returns None when nothing rectangular survives.
    """
    wide = pairs.groupby(["seed", "t"])[value_col].mean().unstack("t")
    wide = wide.sort_index(axis=1).dropna(axis=0, how="any")
    if wide.empty or wide.shape[1] == 0:
        return None
    return wide.to_numpy()


# --- experiment 2 ---------------------------------------------------------


def build_exp2(
    collection: Collection,
    out_dir: Path,
    *,
    stat: str = "mean",
    source: str = "base",
    delta_p_0_runs: list | None = None,
) -> list[Path]:
    r"""Figures for "Running One Trajectory with Multiple Seeds".

    ``delta_p_0_runs`` is the pool of runs searched for base-model $\Delta P$
    measurements, defaulting to ``collection`` itself. The complete source is
    the sweep written by :mod:`method.probe_base`; runs contribute only what
    their own first step happened to measure.
    """
    saved: list[Path] = []
    if not collection:
        logger.warning("exp2: no runs on disk; skipping")
        return saved

    lookup = delta_p_0_lookup(delta_p_0_runs or collection.runs, stat=stat)
    absent = missing_delta_p_0(collection, lookup)
    if absent:
        logger.warning(
            "exp2: no Delta P_0 measured for %d dataset(s), which will be dropped "
            "from the projection scatter: %s. Fix by running "
            "`python -m method.probe_base`, which measures Delta P for every "
            "dataset at each seed's base checkpoint without training anything.",
            len(absent),
            ", ".join(sorted(absent)),
        )

    for trait in collection.values("trait"):
        subset = collection.filter(trait=trait)
        pretty = display_trait_name(trait)
        prefix = f"exp2_{trait}"

        pairs = projection_frame(subset, lookup, stat=stat)
        if pairs.empty:
            logger.warning(
                "exp2/%s: no step has both a Delta P_0 and a Delta P_t; "
                "skipping the projection figures",
                trait,
            )
        else:
            fig = figures.scatter_projection_correlation(
                pairs["delta_p_0"],
                pairs["delta_p_t"],
                pairs["delta_behavior"],
                title=f"{pretty}: projection difference vs. behaviour change",
            )
            _emit(fig, f"{prefix}_projection_correlation", out_dir, saved)

            ratios = pairs.assign(
                ratio=ratio_percent(pairs["delta_p_t"], pairs["delta_p_0"])
            )
            matrix = _seed_step_matrix(ratios, "ratio")
            if matrix is not None:
                fig = figures.line_with_band(
                    {r"$\Delta P_t$": matrix},
                    ylabel=r"$\Delta P_t$ (% of $\Delta P_0$ for that dataset)",
                    xlabel="Step $t$",
                    reference=100.0,
                    reference_label="Step-0 value",
                    title=rf"{pretty}: drift of $\Delta P_t$ relative to $\Delta P_0$",
                )
                _emit(fig, f"{prefix}_drift_delta_p", out_dir, saved)

        # The proposal's "how far has Delta P_t drifted from its step-0 value"
        # line plot. Unlike the ratio plot above -- which compares a *different*
        # dataset at each step against its own baseline -- this follows a fixed
        # probe dataset across a moving model, so every change is the model's.
        probed = sorted(
            {ds for traj in subset.trajectories for ds in traj.probed_datasets()}
        )
        probe_series = {}
        for dataset in probed:
            rows = schema.probe_matrix(subset.trajectories, dataset, stat=stat)
            if rows and _percent_safe(rows):
                probe_series[display_dataset_name(dataset)] = rows
            elif rows:
                logger.warning(
                    "exp2/%s: Delta P_0 for %s is too close to zero to express "
                    "later steps as a percentage of it; omitting from the "
                    "probe drift plot",
                    trait,
                    dataset,
                )
        if probe_series:
            fig = figures.drift_line(
                probe_series,
                ylabel=r"$\Delta P_t$ (% of step 0)",
                title=rf"{pretty}: drift of $\Delta P_t$ for a fixed dataset",
            )
            _emit(fig, f"{prefix}_drift_probes", out_dir, saved)

        # z_t components against the behaviour change they precede.
        metric_data = {}
        for component, label in Z_LABELS.items():
            mp = schema.metric_pairs(subset.trajectories, component, source=source)
            if not mp.empty:
                metric_data[label] = (
                    mp["value"].to_numpy(),
                    mp["delta_behavior"].to_numpy(),
                )
        if metric_data:
            fig = figures.scatter_metric_grid(
                metric_data,
                title=f"{pretty}: latent-state components vs. behaviour change",
            )
            _emit(fig, f"{prefix}_latent_metric_grid", out_dir, saved)

        # Drift of every z_t component from its own step-0 value. Components
        # whose baseline is too close to zero for a percentage to mean
        # anything get their own raw-unit plot instead of poisoning the shared
        # axis (see _percent_safe).
        as_percent: dict[str, list[list[float]]] = {}
        for component, label in Z_LABELS.items():
            runs = schema.z_component_matrix(
                subset.trajectories, component, source=source
            )
            if not runs:
                continue
            if _percent_safe(runs):
                as_percent[label] = runs
                continue
            try:
                matrix = stack_and_trim(runs)
            except ValueError:
                continue
            fig = figures.line_with_band(
                {label: matrix},
                ylabel=f"{label} (raw units)",
                xlabel="Step $t$",
                title=f"{pretty}: {label} over the trajectory",
            )
            _emit(fig, f"{prefix}_drift_{component}", out_dir, saved)

        if as_percent:
            fig = figures.drift_line(
                as_percent,
                ylabel="% of step-0 value",
                title=f"{pretty}: drift of the latent state over the trajectory",
            )
            _emit(fig, f"{prefix}_drift_z", out_dir, saved)

    return saved


# --- experiment 3 ---------------------------------------------------------


def build_exp3(collection: Collection, out_dir: Path) -> list[Path]:
    """Hysteresis bar chart, one figure per (measured trait, realign trait)."""
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
            realign_pretty = display_trait_name(realign_trait)
            fig = figures.hysteresis_bar(
                subset,
                conditions=conditions,
                condition_labels=[HYSTERESIS_CONDITION_LABELS[c] for c in conditions],
                title=(
                    f"{pretty}: behaviour change before vs. after "
                    f"re-alignment on {realign_pretty} (Normal)"
                ),
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
            datasets = sorted(set(subset["dataset"]))
            trained_on = display_dataset_name(datasets[0]) if datasets else "trait data"
            fig = figures.diversity_bar(
                subset,
                order=conditions,
                labels=DIVERSITY_CONDITION_LABELS,
                ylabel=r"Residual $b_T - b_0$",
                title=(
                    f"{pretty}: residual after re-alignment, "
                    f"by diversity of prior training on {trained_on}"
                ),
            )
            _emit(fig, f"exp4_{trait}_realign-{realign_trait}", out_dir, saved)
    return saved


# --- driver ---------------------------------------------------------------

BUILDERS = {
    experiments.EXP2: build_exp2,
    experiments.EXP3: build_exp3,
    experiments.EXP4: build_exp4,
}


def build_and_save(
    out_dir: Path,
    *,
    groups: list[str] | None = None,
    local: bool = False,
    mock: bool = False,
    stat: str = "mean",
    source: str = "base",
) -> list[Path]:
    """Build every requested experiment's figures, returning the files written."""
    groups = groups or list(BUILDERS)
    collections = {
        group: collect_group(group, local=local, mock=mock)
        for group in groups
    }
    for collection in collections.values():
        logger.info(collection.summary())

    saved: list[Path] = []
    if experiments.EXP2 in collections:
        # Every collected run is searched for Delta P_0, not just exp2's: a
        # hysteresis baseline's first step measures the base model against a
        # dataset exp2 only reaches mid-trajectory. This is a bonus, not the
        # mechanism -- `probe_base` is what covers every dataset properly.
        saved += build_exp2(
            collections[experiments.EXP2],
            out_dir,
            stat=stat,
            source=source,
            delta_p_0_runs=[run for c in collections.values() for run in c.runs],
        )
    for group in (experiments.EXP3, experiments.EXP4):
        if group in collections:
            saved += BUILDERS[group](collections[group], out_dir)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=[*BUILDERS, "all"],
        default="all",
        help="which experiment family to plot (default: all)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=style.PLOTS_DIR / "real",
        help="directory to write PNG/PDF figures into (default: plots/real)",
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

    groups = list(BUILDERS) if args.experiment == "all" else [args.experiment]
    saved = build_and_save(
        args.out_dir,
        groups=groups,
        local=args.local,
        mock=args.mock,
        stat=args.stat,
        source=args.source,
    )
    if not saved:
        logger.error(
            "No figures written. Run the trajectories first, e.g. "
            "poetry run python -m method.run_trajectory --config <NAME>"
        )
        return
    print(f"Wrote {len(saved)} file(s) to {args.out_dir}:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
