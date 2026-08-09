r"""Generate figures from synthetic data and save them under ``plots/`` as both
PNG and PDF.

    poetry run python -m method.visualization.demo
    poetry run python -m method.visualization.demo --out-dir /tmp/plots --n-seeds 3

Useful both as an end-to-end smoke test (matplotlib and these plotting
functions run with no GPU, no judge, and no real trajectory) and as a preview
of what the real figures will look like once experiments have actually run.

Covers the figures whose input is a plain per-seed trajectory: the projection
scatter, the latent-metric grid, the drift lines and the two bar charts. The
RQ1 decay set is not drawn here, because its input is a *fan* -- trunks
crossed with branches crossed with a validation family -- which the mock
backend already produces in schema-faithful form. Preview those with::

    poetry run python -m method.run_trajectory --config <NAME> --backend mock
    poetry run python -m method.visualization.make_plots --mock --local
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from method.visualization import figures, schema, style, synthetic
from method.visualization.metrics import ratio_percent

#: Display label -> $z_t$ component, for the metric-grid and drift plots.
_Z_LABELS = {"p": r"$p_t$", "q": r"$q_t$", "rho": r"$\rho_t$", "r": r"$r_t$"}


def build_and_save(out_dir: Path = style.PLOTS_DIR, *, n_seeds: int = 5) -> list[Path]:
    """Build every figure on synthetic data and save it under ``out_dir``.

    Returns the full list of written file paths (PNG and PDF, two per
    figure), so callers (and tests) can assert on what landed on disk.
    """
    saved: list[Path] = []

    def emit(fig: plt.Figure, name: str) -> None:
        saved.extend(style.save_figure(fig, name, out_dir))
        plt.close(fig)

    trajectories = synthetic.synthetic_trajectory_set(n_seeds=n_seeds)
    delta_p_0 = synthetic.synthetic_delta_p_0_lookup()

    # --- RQ1: does Delta P_0 or Delta P_t better predict Delta b_{t+1}? ---
    pairs = schema.projection_pairs(trajectories, delta_p_0)
    fig = figures.scatter_projection_correlation(
        pairs["delta_p_0"],
        pairs["delta_p_t"],
        pairs["delta_behavior"],
        title="Projection difference vs. behaviour change",
    )
    emit(fig, "projection_correlation")

    # --- RQ1: the same diagnostic for p, q, rho and r ---
    metric_data = {}
    for component, label in _Z_LABELS.items():
        mp = schema.metric_pairs(trajectories, component)
        metric_data[label] = (mp["value"].to_numpy(), mp["delta_behavior"].to_numpy())
    fig = figures.scatter_metric_grid(
        metric_data, title="Latent-state components vs. behaviour change"
    )
    emit(fig, "latent_metric_grid")

    # --- RQ1: drift of rho_t and r_t from their step-0 value ---
    for component in ("rho", "r"):
        label = _Z_LABELS[component]
        runs = schema.z_component_matrix(trajectories, component)
        fig = figures.drift_line(
            {label: runs},
            ylabel=f"{label} (% of step 0)",
            title=f"Drift of {label} over the trajectory",
        )
        emit(fig, f"drift_{component}")

    # --- RQ1: how far Delta P_t drifts (as % of Delta P_0) with step index ---
    pairs = pairs.assign(ratio=ratio_percent(pairs["delta_p_t"], pairs["delta_p_0"]))
    by_step = pairs.groupby("t")["ratio"].mean().sort_index()
    fig = figures.line_with_band(
        {r"$\Delta P_t$": by_step.to_numpy()[None, :]},
        ylabel=r"$\Delta P_t$ (% of $\Delta P_0$ for that dataset)",
        xlabel="Step $t$",
        reference=100.0,
        reference_label="Step-0 value",
        title=r"Drift of $\Delta P_t$ relative to $\Delta P_0$",
    )
    emit(fig, "drift_delta_p")

    # --- RQ2: hysteresis -- is a realigned model easier to re-misalign? ---
    hysteresis_df = synthetic.synthetic_hysteresis_frame(n_seeds=n_seeds)
    fig = figures.hysteresis_bar(
        hysteresis_df,
        conditions=synthetic.HYSTERESIS_CONDITIONS,
        condition_labels=[
            synthetic.HYSTERESIS_LABELS[c] for c in synthetic.HYSTERESIS_CONDITIONS
        ],
        # Same arms, same order, same labels as make_plots draws for real runs.
        start_col="behavior_before",
        reference=float(hysteresis_df["behavior_base"].mean()),
        reference_label=r"Base model $b_0$",
        ylabel=r"Trait score after the final step ($b_T$)",
        title="Trait score after a step onto each dataset, by prior training",
    )
    emit(fig, "hysteresis")

    # --- RQ2: does dataset diversity hinder re-alignment? ---
    diversity_df = synthetic.synthetic_diversity_frame(n_seeds=n_seeds)
    fig = figures.diversity_bar(
        diversity_df,
        order=synthetic.DIVERSITY_CONDITIONS,
        labels=synthetic.DIVERSITY_LABELS,
        title="Residual misalignment after realignment, by training diversity",
    )
    emit(fig, "diversity")

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=style.PLOTS_DIR,
        help="directory to write PNG/PDF figures into (default: plots/)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="number of synthetic seeds per experiment",
    )
    args = parser.parse_args()

    saved = build_and_save(args.out_dir, n_seeds=args.n_seeds)
    print(f"Wrote {len(saved)} file(s) to {args.out_dir}:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
