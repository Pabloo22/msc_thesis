"""Figures for the sequential fine-tuning proposal, built from either real
``trajectory.json`` measurements or the synthetic fixtures in
:mod:`method.visualization.synthetic`.

    from method.visualization import figures, schema, style, synthetic

    trajectories = synthetic.synthetic_trajectory_set(n_seeds=5)
    dp0 = synthetic.synthetic_delta_p_0_lookup()
    pairs = schema.projection_pairs(trajectories, dp0)
    fig = figures.scatter_projection_correlation(
        pairs["delta_p_0"], pairs["delta_p_t"], pairs["delta_behavior"]
    )
    style.save_figure(fig, "projection_correlation")

See :mod:`method.visualization.demo` for a runnable script that generates
every figure this way and saves it under ``plots/``.
"""

from __future__ import annotations

# NB: the ``collect`` *function* is deliberately not re-exported here. Binding
# it as ``method.visualization.collect`` would shadow the submodule of the same
# name, so ``from method.visualization import collect`` would hand back a
# function. Import it from :mod:`method.visualization.collect` directly.
from method.visualization.collect import (
    Collection,
    Run,
    collect_group,
    delta_p_0_lookup,
    diversity_frame,
    hysteresis_frame,
    projection_frame,
)
from method.visualization.decay import (
    decay_frame,
    fit_frame,
    latent_frame,
    mechanism_frame,
    phase_contrast_frame,
    probe_drift_frame,
    realignment_pairs,
    validation_frame,
)
from method.visualization.figures import (
    decay_scatter_grid,
    diversity_bar,
    drift_line,
    headline_curves,
    hysteresis_bar,
    line_with_band,
    mechanism_grid,
    overlay_grid,
    overlay_lines,
    phase_contrast,
    scatter_metric_grid,
    scatter_projection_correlation,
    scatter_validation,
)
from method.visualization.labels import display_dataset_name
from method.visualization.schema import (
    StepRecord,
    Trajectory,
    delta_p_by_dataset,
    load_trajectory,
    load_trajectory_set,
    metric_pairs,
    projection_pairs,
    to_frame,
    z_component_matrix,
)
from method.visualization.style import PLOTS_DIR, apply_style, save_figure

__all__ = [
    "PLOTS_DIR",
    "Collection",
    "Run",
    "StepRecord",
    "Trajectory",
    "apply_style",
    "collect_group",
    "decay_frame",
    "decay_scatter_grid",
    "delta_p_0_lookup",
    "delta_p_by_dataset",
    "diversity_frame",
    "hysteresis_frame",
    "fit_frame",
    "headline_curves",
    "latent_frame",
    "mechanism_frame",
    "mechanism_grid",
    "overlay_grid",
    "overlay_lines",
    "phase_contrast",
    "phase_contrast_frame",
    "probe_drift_frame",
    "projection_frame",
    "realignment_pairs",
    "display_dataset_name",
    "diversity_bar",
    "drift_line",
    "hysteresis_bar",
    "line_with_band",
    "load_trajectory",
    "load_trajectory_set",
    "metric_pairs",
    "projection_pairs",
    "save_figure",
    "scatter_metric_grid",
    "scatter_projection_correlation",
    "scatter_validation",
    "to_frame",
    "validation_frame",
    "z_component_matrix",
]
