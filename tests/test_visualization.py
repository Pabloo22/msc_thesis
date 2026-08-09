"""Tests for :mod:`method.visualization`.

Every figure function is exercised on synthetic data (never a real
trajectory), so these tests run with no GPU, no judge, and no fine-tuning.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection

from method import experiments
from method.visualization import figures, labels, make_plots, schema, style, synthetic
from method.visualization.collect import Collection
from method.visualization.demo import build_and_save
from method.visualization.labels import display_dataset_name
from method.visualization.metrics import (
    behavior_deltas,
    linear_fit,
    mean_std,
    percent_of_baseline,
    ratio_percent,
    stack_and_trim,
)
from method.visualization.schema import StepRecord, Trajectory

import matplotlib.pyplot as plt  # noqa: E402  (backend fixed by style import)


@pytest.fixture(autouse=True)
def close_figures():
    """Release every Figure a test built.

    These tests assert on figures rather than saving them, so nothing closes
    them; matplotlib keeps each one alive and warns once twenty are open.
    """
    yield
    plt.close("all")


# --- metrics.py -------------------------------------------------------------


class TestLinearFit:
    def test_perfect_line(self) -> None:
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = 2.0 * x + 1.0
        fit = linear_fit(x, y)
        assert fit.slope == pytest.approx(2.0)
        assert fit.intercept == pytest.approx(1.0)
        assert fit.r2 == pytest.approx(1.0)

    def test_noisy_line_has_r2_below_one(self) -> None:
        rng = np.random.default_rng(0)
        x = np.arange(50, dtype=float)
        y = 3.0 * x + rng.normal(0, 20, size=50)
        fit = linear_fit(x, y)
        assert 0.0 < fit.r2 < 1.0

    def test_constant_x_is_degenerate(self) -> None:
        fit = linear_fit([1.0, 1.0, 1.0], [4.0, 6.0, 5.0])
        assert fit.slope == 0.0
        assert fit.intercept == pytest.approx(5.0)
        assert fit.r2 == 0.0

    def test_single_point_is_degenerate(self) -> None:
        fit = linear_fit([1.0], [7.0])
        assert fit.slope == 0.0
        assert fit.intercept == pytest.approx(7.0)

    def test_predict_matches_slope_intercept(self) -> None:
        fit = linear_fit([0.0, 1.0], [0.0, 2.0])
        np.testing.assert_allclose(fit.predict([0.0, 5.0]), [0.0, 10.0], atol=1e-12)


class TestStackAndTrim:
    def test_trims_to_shortest(self) -> None:
        out = stack_and_trim([[1, 2, 3], [1, 2, 3, 4]])
        assert out.shape == (2, 3)
        np.testing.assert_array_equal(out, [[1, 2, 3], [1, 2, 3]])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            stack_and_trim([])

    def test_zero_length_row_raises(self) -> None:
        with pytest.raises(ValueError):
            stack_and_trim([[], [1, 2]])


class TestPercentOfBaseline:
    def test_1d(self) -> None:
        out = percent_of_baseline([50.0, 100.0, 25.0])
        np.testing.assert_allclose(out, [100.0, 200.0, 50.0])

    def test_2d_normalises_each_row(self) -> None:
        out = percent_of_baseline([[10.0, 20.0], [4.0, 2.0]], axis=1)
        np.testing.assert_allclose(out, [[100.0, 200.0], [100.0, 50.0]])

    def test_near_zero_baseline_is_nan(self) -> None:
        out = percent_of_baseline([0.0, 5.0])
        assert math.isnan(out[0])
        assert math.isnan(out[1])


class TestRatioPercent:
    def test_basic(self) -> None:
        out = ratio_percent([1.0, 2.0], [2.0, 2.0])
        np.testing.assert_allclose(out, [50.0, 100.0])

    def test_zero_denominator_is_nan(self) -> None:
        out = ratio_percent([1.0], [0.0])
        assert math.isnan(out[0])


class TestMeanStd:
    def test_across_seeds(self) -> None:
        mean, std = mean_std([[1.0, 2.0], [3.0, 4.0]], axis=0)
        np.testing.assert_allclose(mean, [2.0, 3.0])
        np.testing.assert_allclose(std, [1.0, 1.0])


class TestBehaviorDeltas:
    def test_diffs(self) -> None:
        np.testing.assert_allclose(behavior_deltas([10.0, 15.0, 12.0]), [5.0, -3.0])


# --- schema.py ---------------------------------------------------------------


def _payload(t: int, behavior_val: float, next_dataset: str | None) -> dict:
    """A minimal payload shaped like one entry of a real trajectory.json."""
    step = {
        "t": t,
        "weights_id": f"t{t:02d}-deadbeef",
        "behavior": {"evil": behavior_val, "evil_std": 5.0, "coherence": 70.0, "n": 20},
        "z": {"base": {"p": 1.0 + t, "q": 0.9 + t, "rho": 1.0 - 0.1 * t, "r": 30.0}},
    }
    if next_dataset is not None:
        step["delta_p"] = {
            "mean": 0.5 + t,
            "std": 1.0,
            "min": -1.0,
            "p05": -0.5,
            "p25": 0.0,
            "median": 0.5,
            "p75": 1.0,
            "p95": 1.5,
            "max": 2.0,
            "n": 32,
        }
        step["next_dataset"] = next_dataset
    return step


@pytest.fixture
def branch_trajectory() -> Trajectory:
    """What a branch writes: one endpoint record, b only, no z."""
    return Trajectory.from_dict(
        {
            "config": {"name": "branch", "trait": "evil", "seed": 0},
            "steps": [
                {"t": 3, "weights_id": "t03-feedfeedfeedfeed",
                 "behavior": {"evil": 55.0}},
            ],
        }
    )


@pytest.fixture
def two_step_trajectory() -> Trajectory:
    payload = {
        "config": {"name": "toy", "trait": "evil", "seed": 0},
        "steps": [
            _payload(0, 40.0, "evil/normal"),
            _payload(1, 30.0, "evil/normal"),
            _payload(2, 25.0, None),
        ],
    }
    return Trajectory.from_dict(payload)


class TestStepRecordAndTrajectory:
    def test_from_dict_round_trip(self, two_step_trajectory: Trajectory) -> None:
        assert two_step_trajectory.name == "toy"
        assert two_step_trajectory.trait == "evil"
        assert len(two_step_trajectory.steps) == 3
        assert two_step_trajectory.steps[0].behavior["evil"] == 40.0
        assert two_step_trajectory.steps[-1].delta_p is None
        assert two_step_trajectory.steps[-1].next_dataset is None

    def test_behavior_series(self, two_step_trajectory: Trajectory) -> None:
        assert two_step_trajectory.behavior_series() == [40.0, 30.0, 25.0]

    def test_z_series(self, two_step_trajectory: Trajectory) -> None:
        assert two_step_trajectory.z_series("rho") == pytest.approx([1.0, 0.9, 0.8])

    def test_datasets(self, two_step_trajectory: Trajectory) -> None:
        assert two_step_trajectory.datasets() == ["evil/normal", "evil/normal"]

    def test_load_trajectory_round_trips_through_disk(
        self, two_step_trajectory: Trajectory, tmp_path
    ) -> None:
        path = tmp_path / "trajectory.json"
        payload = {
            "config": {
                "name": two_step_trajectory.name,
                "trait": two_step_trajectory.trait,
                "seed": two_step_trajectory.seed,
            },
            "steps": [
                {
                    "t": s.t,
                    "weights_id": s.weights_id,
                    "behavior": s.behavior,
                    "z": s.z,
                    **({"delta_p": s.delta_p} if s.delta_p is not None else {}),
                    **({"next_dataset": s.next_dataset} if s.next_dataset else {}),
                }
                for s in two_step_trajectory.steps
            ],
        }
        path.write_text(json.dumps(payload))
        loaded = schema.load_trajectory(path)
        assert loaded.name == two_step_trajectory.name
        assert loaded.behavior_series() == two_step_trajectory.behavior_series()
        assert loaded.source == path


class TestToFrame:
    def test_columns_and_values(self, two_step_trajectory: Trajectory) -> None:
        df = schema.to_frame([two_step_trajectory])
        assert len(df) == 3
        assert "behavior_evil" in df.columns
        assert "z_rho" in df.columns
        assert "delta_p_mean" in df.columns
        # the final step has no delta_p, so its column value is missing.
        assert pd.isna(df.loc[df["t"] == 2, "delta_p_mean"]).all()


class TestProjectionPairs:
    def test_pairs_dataset_delta_p0_with_actual_delta_pt(
        self, two_step_trajectory: Trajectory
    ) -> None:
        df = schema.projection_pairs([two_step_trajectory], {"evil/normal": -1.5})
        assert len(df) == 2  # one per non-terminal step
        assert (df["delta_p_0"] == -1.5).all()
        assert df["delta_p_t"].tolist() == pytest.approx([0.5, 1.5])
        assert df["delta_behavior"].tolist() == pytest.approx([-10.0, -5.0])

    def test_unknown_dataset_is_skipped(self, two_step_trajectory: Trajectory) -> None:
        df = schema.projection_pairs([two_step_trajectory], {"other/normal": 1.0})
        assert df.empty


class TestMetricPairs:
    def test_pairs_component_with_delta_behavior(
        self, two_step_trajectory: Trajectory
    ) -> None:
        df = schema.metric_pairs([two_step_trajectory], "rho")
        assert df["value"].tolist() == pytest.approx([1.0, 0.9])
        assert df["delta_behavior"].tolist() == pytest.approx([-10.0, -5.0])


class TestZComponentMatrix:
    def test_one_row_per_trajectory(self, two_step_trajectory: Trajectory) -> None:
        out = schema.z_component_matrix([two_step_trajectory, two_step_trajectory], "r")
        assert out == [[30.0, 30.0, 30.0], [30.0, 30.0, 30.0]]

    def test_branch_trajectories_contribute_no_row(
        self, two_step_trajectory: Trajectory, branch_trajectory: Trajectory
    ) -> None:
        """A decay collection mixes trunks with branches, and a branch measures
        only b -- so z must be skipped, not raised on. This crashed
        ``make_plots --mock`` for the whole exp2_decay family."""
        out = schema.z_component_matrix(
            [two_step_trajectory, branch_trajectory], "r"
        )
        assert out == [[30.0, 30.0, 30.0]]


class TestLatentIsOptional:
    """Branch endpoints record no ``z`` (``MeasurementLevel.ENDPOINT_BEHAVIOR``)."""

    def test_a_record_without_z_parses(self, branch_trajectory: Trajectory) -> None:
        assert branch_trajectory.steps[0].z == {}
        assert branch_trajectory.steps[0].behavior["evil"] == 55.0

    def test_has_latent_distinguishes_trunk_from_branch(
        self, two_step_trajectory: Trajectory, branch_trajectory: Trajectory
    ) -> None:
        assert two_step_trajectory.has_latent()
        assert not branch_trajectory.has_latent()

    def test_z_series_names_the_reason_it_is_missing(
        self, branch_trajectory: Trajectory
    ) -> None:
        with pytest.raises(KeyError, match="Branch endpoints measure"):
            branch_trajectory.z_series("rho")

    def test_metric_pairs_skips_branches(
        self, two_step_trajectory: Trajectory, branch_trajectory: Trajectory
    ) -> None:
        df = schema.metric_pairs([two_step_trajectory, branch_trajectory], "rho")
        assert df["value"].tolist() == pytest.approx([1.0, 0.9])


class TestDeltaPByDataset:
    def test_occurrences_in_step_order(self, two_step_trajectory: Trajectory) -> None:
        out = schema.delta_p_by_dataset([two_step_trajectory], "evil/normal")
        assert out == [[0.5, 1.5]]

    def test_missing_dataset_is_dropped(self, two_step_trajectory: Trajectory) -> None:
        assert schema.delta_p_by_dataset([two_step_trajectory], "nope/normal") == []


# --- labels.py -----------------------------------------------------------------


class TestDisplayDatasetName:
    @pytest.mark.parametrize(
        "dataset_id, expected",
        [
            ("evil/normal", "Evil (Normal)"),
            ("evil/misaligned_1", "Evil (I)"),
            ("evil/misaligned_2", "Evil (II)"),
            ("sycophancy/normal", "Sycophancy (Normal)"),
            ("sycophancy/misaligned_1", "Sycophancy (I)"),
            ("sycophancy/misaligned_2", "Sycophancy (II)"),
            ("hallucination/normal", "Hallucination (Normal)"),
            ("hallucination/misaligned_1", "Hallucination (I)"),
            ("hallucination/misaligned_2", "Hallucination (II)"),
            ("mistake_medical/normal", "Medical (Normal)"),
            ("mistake_medical/misaligned_1", "Medical (Mistake I)"),
            ("mistake_medical/misaligned_2", "Medical (Mistake II)"),
            ("insecure_code/normal", "Code (Normal)"),
            ("insecure_code/misaligned_1", "Code (Insecure I)"),
            ("insecure_code/misaligned_2", "Code (Insecure II)"),
            ("mistake_gsm8k/normal", "GSM8K (Normal)"),
            ("mistake_gsm8k/misaligned_1", "GSM8K (Mistake I)"),
            ("mistake_gsm8k/misaligned_2", "GSM8K (Mistake II)"),
            ("mistake_math/normal", "MATH (Normal)"),
            ("mistake_math/misaligned_1", "MATH (Mistake I)"),
            ("mistake_math/misaligned_2", "MATH (Mistake II)"),
            ("mistake_opinions/normal", "Opinions (Normal)"),
            ("mistake_opinions/misaligned_1", "Opinions (Mistake I)"),
            ("mistake_opinions/misaligned_2", "Opinions (Mistake II)"),
        ],
    )
    def test_matches_the_proposal_naming(self, dataset_id: str, expected: str) -> None:
        assert display_dataset_name(dataset_id) == expected

    def test_unknown_dataset_falls_back_to_the_raw_name(self) -> None:
        assert display_dataset_name("made_up/normal") == "made_up (Normal)"

    def test_no_version_returns_just_the_title(self) -> None:
        assert display_dataset_name("evil") == "Evil"


# --- synthetic.py --------------------------------------------------------------


class TestSyntheticTrajectory:
    def test_step_count(self) -> None:
        traj = synthetic.synthetic_trajectory(datasets=synthetic.DEFAULT_DATASETS[:3])
        assert len(traj.steps) == 4

    def test_final_step_has_no_delta_p_or_next_dataset(self) -> None:
        traj = synthetic.synthetic_trajectory(datasets=synthetic.DEFAULT_DATASETS[:2])
        assert traj.steps[-1].delta_p is None
        assert traj.steps[-1].next_dataset is None

    def test_delta_p_keys_match_summarize(self) -> None:
        traj = synthetic.synthetic_trajectory(datasets=synthetic.DEFAULT_DATASETS[:1])
        delta_p = traj.steps[0].delta_p
        assert delta_p is not None
        assert set(delta_p) == {
            "mean",
            "std",
            "min",
            "p05",
            "p25",
            "median",
            "p75",
            "p95",
            "max",
            "n",
        }

    def test_is_json_serializable(self) -> None:
        traj = synthetic.synthetic_trajectory(datasets=synthetic.DEFAULT_DATASETS[:2])
        for step in traj.steps:
            json.dumps(step.behavior)
            json.dumps(step.z)
            if step.delta_p is not None:
                json.dumps(step.delta_p)

    def test_deterministic_given_seed(self) -> None:
        a = synthetic.synthetic_trajectory(
            seed=3, datasets=synthetic.DEFAULT_DATASETS[:2]
        )
        b = synthetic.synthetic_trajectory(
            seed=3, datasets=synthetic.DEFAULT_DATASETS[:2]
        )
        assert a.behavior_series() == b.behavior_series()

    def test_different_seeds_differ(self) -> None:
        a = synthetic.synthetic_trajectory(
            seed=0, datasets=synthetic.DEFAULT_DATASETS[:2]
        )
        b = synthetic.synthetic_trajectory(
            seed=1, datasets=synthetic.DEFAULT_DATASETS[:2]
        )
        assert a.behavior_series() != b.behavior_series()


class TestSyntheticTrajectorySet:
    def test_length_and_seeds(self) -> None:
        trajectories = synthetic.synthetic_trajectory_set(n_seeds=4)
        assert len(trajectories) == 4
        assert [t.seed for t in trajectories] == [0, 1, 2, 3]


class TestDeltaP0For:
    def test_deterministic(self) -> None:
        assert synthetic.delta_p_0_for("evil/normal") == synthetic.delta_p_0_for(
            "evil/normal"
        )

    def test_sign_matches_dataset_pull(self) -> None:
        assert synthetic.delta_p_0_for("evil/misaligned_2") > 0
        assert synthetic.delta_p_0_for("evil/normal") < 0

    def test_lookup_covers_every_dataset(self) -> None:
        lookup = synthetic.synthetic_delta_p_0_lookup(synthetic.DEFAULT_DATASETS)
        assert set(lookup) == set(synthetic.DEFAULT_DATASETS)


class TestSyntheticFrames:
    def test_hysteresis_frame_shape(self) -> None:
        df = synthetic.synthetic_hysteresis_frame(("a/normal", "b/normal"), n_seeds=3)
        n_conditions = len(synthetic.HYSTERESIS_CONDITIONS)
        assert len(df) == 2 * 3 * n_conditions  # datasets * seeds * conditions
        assert set(df["condition"]) == set(synthetic.HYSTERESIS_CONDITIONS)

    def test_plasticity_arms_move_less_and_worsen_with_prior_steps(self) -> None:
        """The fixture has to encode the effects the arms exist to detect."""
        df = synthetic.synthetic_hysteresis_frame(("a/normal",), n_seeds=30)
        means = df.groupby("condition")["delta_behavior"].mean()
        assert means["normal2"] < means["normal1"] < means["baseline"]
        assert means["baseline"] < means["diff"] < means["same"]

    def test_synthetic_conditions_track_the_real_ones(self) -> None:
        """A fixture naming its own arms silently stops previewing the design."""
        assert synthetic.HYSTERESIS_CONDITIONS == labels.HYSTERESIS_CONDITIONS
        df = synthetic.synthetic_hysteresis_frame(("a/normal",), n_seeds=1)
        assert set(df["condition"]) == set(labels.HYSTERESIS_CONDITIONS)

    def test_diversity_frame_matches_fixed_conditions(self) -> None:
        df = synthetic.synthetic_diversity_frame(n_seeds=5)
        assert set(df["condition"]) == set(synthetic.DIVERSITY_CONDITIONS)
        assert len(df) == len(synthetic.DIVERSITY_CONDITIONS) * 5


# --- figures.py ----------------------------------------------------------------


class TestScatterProjectionCorrelation:
    def test_returns_figure_with_two_fit_lines(self) -> None:
        fig = figures.scatter_projection_correlation(
            [1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [10.0, 20.0, 30.0]
        )
        (ax,) = fig.axes
        # 2 scatter collections + 2 fit lines + 1 axhline reference at y=0.
        assert len(ax.collections) == 2
        assert len(ax.lines) == 3
        legend = ax.get_legend()
        assert legend is not None
        legend_labels = [t.get_text() for t in legend.get_texts()]
        assert any(r"\Delta P_0" in label for label in legend_labels)
        assert any(r"\Delta P_t" in label for label in legend_labels)


class TestScatterMetricGrid:
    def test_one_panel_per_metric(self) -> None:
        data = {
            "p": ([1.0, 2.0], [1.0, 2.0]),
            "q": ([1.0, 2.0], [2.0, 1.0]),
            "rho": ([1.0, 2.0], [1.0, 1.0]),
            "r": ([1.0, 2.0], [2.0, 2.0]),
        }
        fig = figures.scatter_metric_grid(data)
        assert len(fig.axes) == 4
        assert all(ax.get_visible() for ax in fig.axes)


class TestLineWithBand:
    def test_mean_and_band_computed_per_series(self) -> None:
        series = {"x": np.array([[100.0, 110.0, 120.0], [100.0, 90.0, 80.0]])}
        fig = figures.line_with_band(series, ylabel="value")
        (ax,) = fig.axes
        line = ax.lines[0]
        np.testing.assert_allclose(np.asarray(line.get_ydata()), [100.0, 100.0, 100.0])

    def test_reference_line_added(self) -> None:
        series = {"x": np.array([[1.0, 2.0]])}
        fig = figures.line_with_band(
            series, ylabel="v", reference=1.5, reference_label="ref"
        )
        (ax,) = fig.axes
        assert len(ax.lines) == 2  # the series line + the reference line


class TestDriftLine:
    def test_normalises_to_percent_of_first_step(self) -> None:
        series = {"metric": [[10.0, 20.0, 5.0]]}
        fig = figures.drift_line(series, ylabel="pct")
        (ax,) = fig.axes
        data_line = next(ln for ln in ax.lines if ln.get_label() == "metric")
        np.testing.assert_allclose(
            np.asarray(data_line.get_ydata()), [100.0, 200.0, 50.0]
        )

    def test_trims_ragged_seeds(self) -> None:
        series = {"metric": [[1.0, 2.0, 3.0], [1.0, 2.0]]}
        fig = figures.drift_line(series, ylabel="pct")
        (ax,) = fig.axes
        data_line = next(ln for ln in ax.lines if ln.get_label() == "metric")
        assert len(np.asarray(data_line.get_ydata())) == 2


class TestHysteresisBar:
    def test_bar_count_matches_datasets_times_conditions(self) -> None:
        df = synthetic.synthetic_hysteresis_frame(("a/normal", "b/normal"), n_seeds=3)
        fig = figures.hysteresis_bar(df)
        (ax,) = fig.axes
        assert len(ax.patches) == 2 * len(synthetic.HYSTERESIS_CONDITIONS)
        assert ax.get_legend() is not None

    def test_a_condition_absent_from_the_frame_is_simply_not_drawn(self) -> None:
        """Plotting a partly-finished sweep must narrow the figure, not raise.

        This is what a mock or half-run family looks like: ``make_plots`` filters
        ``conditions`` down to the ones actually on disk, so a two-bar figure
        means two arms have run -- not that the design has two arms.
        """
        df = synthetic.synthetic_hysteresis_frame(("a/normal",), n_seeds=2)
        subset = df[df["condition"].isin(("baseline", "same"))]
        fig = figures.hysteresis_bar(subset, conditions=("baseline", "same"))
        (ax,) = fig.axes
        heights = [p.get_height() for p in ax.patches]
        assert len(heights) == 2
        assert not any(math.isnan(h) for h in heights)

    def test_xtick_labels_are_pretty_by_default(self) -> None:
        df = synthetic.synthetic_hysteresis_frame(
            ("hallucination/misaligned_1", "mistake_gsm8k/misaligned_2"), n_seeds=2
        )
        fig = figures.hysteresis_bar(df)
        (ax,) = fig.axes
        tick_text = [t.get_text() for t in ax.get_xticklabels()]
        assert tick_text == ["Hallucination (I)", "GSM8K (Mistake II)"]

    def test_reference_line_is_drawn_at_the_base_model_score(self) -> None:
        """The line every bar is read against; without it a level means nothing."""
        df = synthetic.synthetic_hysteresis_frame(("a/normal",), n_seeds=2)
        fig = figures.hysteresis_bar(
            df, reference=8.0, reference_label=r"Base model $b_0$"
        )
        (ax,) = fig.axes
        dashed = [ln for ln in ax.lines if ln.get_linestyle() == "--"]
        assert len(dashed) == 1
        assert dashed[0].get_ydata()[0] == 8.0
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert r"Base model $b_0$" in legend_labels

    def test_bars_default_to_levels_not_the_final_step_delta(self) -> None:
        """A re-aligned arm's bar must show where it ended, not how far it
        moved from its own floor -- see hysteresis_frame's docstring."""
        df = synthetic.synthetic_hysteresis_frame(("a/normal",), n_seeds=3)
        fig = figures.hysteresis_bar(df, conditions=("same",))
        (ax,) = fig.axes
        expected = df[df["condition"] == "same"]["behavior"].mean()
        assert ax.patches[0].get_height() == pytest.approx(expected)

    def test_start_ticks_mark_where_the_final_step_began(self) -> None:
        df = synthetic.synthetic_hysteresis_frame(("a/normal",), n_seeds=2)
        without = figures.hysteresis_bar(df)
        with_ticks = figures.hysteresis_bar(df, start_col="behavior_before")
        assert len(with_ticks.axes[0].collections) > len(without.axes[0].collections)

    def test_dataset_labels_override_the_default(self) -> None:
        df = synthetic.synthetic_hysteresis_frame(("evil/normal",), n_seeds=2)
        fig = figures.hysteresis_bar(df, dataset_labels={"evil/normal": "Custom"})
        (ax,) = fig.axes
        assert [t.get_text() for t in ax.get_xticklabels()] == ["Custom"]


class TestDiversityBar:
    def test_bar_count_matches_conditions(self) -> None:
        df = synthetic.synthetic_diversity_frame(n_seeds=3)
        fig = figures.diversity_bar(
            df, order=synthetic.DIVERSITY_CONDITIONS, labels=synthetic.DIVERSITY_LABELS
        )
        (ax,) = fig.axes
        assert len(ax.patches) == len(synthetic.DIVERSITY_CONDITIONS)


# --- figures.py: the RQ1 decay set -------------------------------------------


def _decay_rows(trunks=("a", "b"), checkpoints=range(3), n_probes=4) -> pd.DataFrame:
    """A decay frame in the shape :mod:`method.visualization.decay` emits."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        [
            {
                "trunk": trunk,
                "t": t,
                "probe": f"d{i}/normal",
                "steps_since_realignment": t % 2,
                "delta_p_0": float(i),
                "delta_p_t": float(i) + t,
                "delta_b": 2.0 * i + rng.normal(0, 0.1),
                "se_delta_b": 0.4,
            }
            for trunk in trunks
            for t in checkpoints
            for i in range(n_probes)
        ]
    )


def _fits(trunks=("a", "b"), checkpoints=range(3)) -> pd.DataFrame:
    """A fit frame: one row per (trunk, checkpoint), both series present."""
    return pd.DataFrame(
        [
            {
                "trunk": trunk,
                "t": t,
                "r2_max": 0.9,
                "rho": 1.0 - 0.1 * t,
                "r": 30.0 + t,
                "b_t": 40.0 + t,
                "steps_since_realignment": t % 2,
                "r2_p0": 0.8 - 0.1 * t,
                "r2_p0_lo": 0.6 - 0.1 * t,
                "r2_p0_hi": 0.9,
                "r2_pt": 0.8,
                "r2_pt_lo": 0.7,
                "r2_pt_hi": 0.9,
                "slope_p0": 2.0 - 0.2 * t,
                "slope_p0_lo": 1.0,
                "slope_p0_hi": 3.0,
                "slope_pt": 2.0,
                "slope_pt_lo": 1.5,
                "slope_pt_hi": 2.5,
            }
            for trunk in trunks
            for t in checkpoints
        ]
    )


class TestDatasetMarks:
    """A dataset is a family and a version, and they are different kinds of
    fact: the family is nominal and takes the shape channel, the version is
    ordered by severity and takes a single-hue ramp."""

    def test_family_picks_the_shape_and_version_the_fill(self) -> None:
        normal = style.dataset_mark("mistake_gsm8k/normal")
        second = style.dataset_mark("mistake_gsm8k/misaligned_2")
        assert normal.marker == second.marker
        assert normal.face != second.face

    def test_every_family_has_its_own_shape(self) -> None:
        assert len(set(style.DATASET_MARKERS.values())) == len(style.DATASET_MARKERS)

    def test_the_eight_experiment_families_are_all_covered(self) -> None:
        assert set(style.DATASET_MARKERS) == set(experiments.DATASET_NAMES)

    def test_the_ramp_darkens_with_severity(self) -> None:
        """Normal, I and II are ordered, so their fills must be too --
        lightness is the channel that survives colour-vision deficiency."""
        fills = [
            style.VERSION_FILL[v] for v in ("normal", "misaligned_1", "misaligned_2")
        ]
        assert [_luminance(f) for f in fills] == sorted(
            (_luminance(f) for f in fills), reverse=True
        )

    def test_normal_is_hollow_so_it_needs_a_dark_outline(self) -> None:
        """Its fill is the page itself; without the outline the mark would not
        exist on the chart at all."""
        mark = style.dataset_mark("insecure_code/normal")
        assert mark.face == style.SURFACE
        assert _luminance(mark.edge) < 0.3

    def test_the_line_colour_separates_I_from_II(self) -> None:
        """Both share a dark outline for a crisp silhouette, so taking the line
        from the outline would draw six probe datasets as six identical lines."""
        first = style.dataset_mark("sycophancy/misaligned_1")
        second = style.dataset_mark("evil/misaligned_2")
        assert first.edge == second.edge
        assert first.line != second.line

    def test_an_unknown_dataset_still_gets_a_mark(self) -> None:
        mark = style.dataset_mark("made_up/normal")
        assert mark.marker == style.UNKNOWN_MARKER
        assert mark.face == style.SURFACE


def _luminance(hex_color: str) -> float:
    """Relative luminance, for asserting a ramp actually ramps."""

    def channel(value: int) -> float:
        c = value / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (int(hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


class TestDatasetLegend:
    DATASETS = [
        "evil/misaligned_2",
        "mistake_gsm8k/normal",
        "mistake_gsm8k/misaligned_2",
    ]

    def test_keys_the_encoding_rather_than_naming_every_dataset(self) -> None:
        """24 datasets would need 24 entries; the encoding needs at most 11."""
        handles, texts = figures.dataset_legend(self.DATASETS)
        assert texts == ["Evil", "GSM8K", "Normal", "II"]
        assert len(handles) == len(texts)

    def test_order_is_fixed_not_the_order_the_data_arrived_in(self) -> None:
        shuffled = list(reversed(self.DATASETS))
        assert figures.dataset_legend(shuffled)[1] == (
            figures.dataset_legend(self.DATASETS)[1]
        )

    def test_absent_families_and_versions_are_not_listed(self) -> None:
        _, texts = figures.dataset_legend(["evil/normal"])
        assert texts == ["Evil", "Normal"]


class TestScatterValidation:
    def test_marks_are_grouped_into_one_call_per_shape(self) -> None:
        """A scatter takes a single marker, so eight families are eight calls
        over disjoint index sets."""
        fig = figures.scatter_validation(
            [0.0, 1.0, 2.0],
            [0.0, 2.0, 4.0],
            datasets=["evil/normal", "evil/misaligned_2", "sycophancy/normal"],
        )
        (ax,) = fig.axes
        scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(scatters) == 2  # pentagon and triangle
        assert sum(len(c.get_offsets()) for c in scatters) == 3

    def test_naming_the_datasets_adds_the_encoding_key(self) -> None:
        fig = figures.scatter_validation(
            [0.0, 1.0], [0.0, 2.0], datasets=["evil/normal", "sycophancy/normal"]
        )
        legend = fig.axes[0].get_legend()
        assert [t.get_text() for t in legend.get_texts()] == [
            "Evil",
            "Sycophancy",
            "Normal",
        ]

    def test_single_series_reports_its_fit_without_a_legend(self) -> None:
        """One series needs no legend box -- the axes name the quantity -- but
        the fit's statistics still have to be readable off the figure."""
        fig = figures.scatter_validation([0.0, 1.0, 2.0], [0.0, 2.0, 4.0])
        (ax,) = fig.axes
        assert ax.get_legend() is None
        annotations = [t.get_text() for t in ax.texts]
        assert any("R^2" in a or "R$^2$" in a or r"$R^2$" in a for a in annotations)
        assert any("n$ = 3" in a for a in annotations)

    def test_error_bars_are_drawn_when_given(self) -> None:
        plain = figures.scatter_validation([0.0, 1.0], [0.0, 2.0])
        with_err = figures.scatter_validation([0.0, 1.0], [0.0, 2.0], yerr=[0.5, 0.5])
        assert len(with_err.axes[0].collections) > len(plain.axes[0].collections)

    def test_all_zero_error_bars_are_skipped(self) -> None:
        """A run evaluated with one generation per question has no
        within-question spread; flat caps would imply a measurement it did
        not make."""
        plain = figures.scatter_validation([0.0, 1.0], [0.0, 2.0])
        zeroed = figures.scatter_validation([0.0, 1.0], [0.0, 2.0], yerr=[0.0, 0.0])
        assert len(zeroed.axes[0].collections) == len(plain.axes[0].collections)


class TestDecayScatterGrid:
    def test_one_panel_per_trunk_and_checkpoint(self) -> None:
        fig = figures.decay_scatter_grid(_decay_rows())
        assert len(fig.axes) == 2 * 3

    def test_every_panel_holds_both_series_and_both_r2_values(self) -> None:
        fig = figures.decay_scatter_grid(_decay_rows())
        ax = fig.axes[0]
        scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(scatters) == 2  # Delta P_0 and Delta P_t
        annotations = " ".join(t.get_text() for t in ax.texts)
        assert r"\Delta P_0" in annotations and r"\Delta P_t" in annotations

    def test_axes_are_shared_so_slopes_are_comparable(self) -> None:
        """Per-panel scales would let a flattening slope and a shrinking
        Delta b range look identical, which is the confusion the figure
        exists to prevent."""
        fig = figures.decay_scatter_grid(_decay_rows())
        assert len({ax.get_ylim() for ax in fig.axes}) == 1
        assert len({ax.get_xlim() for ax in fig.axes}) == 1

    def test_phase_is_marked_per_panel_not_per_column(self) -> None:
        """The three schedules put their re-alignments at different depths, so
        column t has a different phase in each row."""
        rows = _decay_rows()
        rows.loc[rows["trunk"] == "b", "steps_since_realignment"] = 2
        fig = figures.decay_scatter_grid(rows, trunks=["a", "b"])
        titles = {ax.get_title(loc="right") for ax in fig.axes}
        assert "s = 2" in titles

    def test_a_checkpoint_with_no_runs_is_marked_not_dropped(self) -> None:
        """A half-finished fan must leave its column visibly empty; silently
        narrowing the grid would hide which checkpoints are still missing."""
        rows = _decay_rows()
        fig = figures.decay_scatter_grid(
            rows[rows["t"] != 1], checkpoints=[0, 1, 2]
        )
        empty = [ax for ax in fig.axes if not ax.collections]
        assert len(empty) == 2  # one per trunk
        assert all("not run" in t.get_text() for ax in empty for t in ax.texts)


class TestHeadlineCurves:
    def test_one_column_per_series_and_a_row_each_for_r2_and_slope(self) -> None:
        fig = figures.headline_curves(_fits())
        assert len(fig.axes) == 4
        ylabels = [ax.get_ylabel() for ax in fig.axes]
        assert any("R^2" in label for label in ylabels)
        assert any("slope" in label for label in ylabels)

    def test_r2_panels_are_bounded_to_the_unit_interval(self) -> None:
        fig = figures.headline_curves(_fits())
        low, high = fig.axes[0].get_ylim()
        assert low <= 0 and high <= 1.05

    def test_noise_ceiling_is_drawn_once_per_trunk_on_the_r2_row(self) -> None:
        with_ceiling = figures.headline_curves(_fits())
        without = figures.headline_curves(_fits(), ceiling_column=None)
        assert len(with_ceiling.axes[0].lines) == len(without.axes[0].lines) + 2

    def test_a_trunk_missing_from_the_frame_is_simply_absent(self) -> None:
        fig = figures.headline_curves(_fits(trunks=("a",)))
        legend = fig.legends[0]
        assert len([t for t in legend.get_texts() if "Trunk" in t.get_text()]) == 1


class TestMechanismGrid:
    PREDICTORS = {"rho": r"$\rho_t$", "b_t": r"$b_t$"}

    def test_one_panel_per_predictor_with_a_fit(self) -> None:
        fig = figures.mechanism_grid(_fits(), self.PREDICTORS)
        assert len(fig.axes) == 2
        assert all(ax.lines for ax in fig.axes)

    def test_states_the_checkpoint_count_not_the_dataset_count(self) -> None:
        """A point here is a checkpoint: the probes at it were already spent
        producing the single R^2 plotted."""
        rows = _fits()
        fig = figures.mechanism_grid(rows, self.PREDICTORS, title="Mechanism")
        assert f"$n$ = {len(rows)} checkpoints" in fig._suptitle.get_text()

    def test_colour_identifies_the_trunk(self) -> None:
        fig = figures.mechanism_grid(
            _fits(), self.PREDICTORS, trunk_colors={"a": style.BLUE, "b": style.ORANGE}
        )
        assert len(fig.axes[0].collections) == 2


class TestPhaseContrast:
    def _pairs(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "trunk": "a", "t_before": 1, "t_after": 2, "pair": "A: 1-2",
                    "r2_p0_before": 0.8, "r2_p0_after": 0.4, "delta_r2_p0": -0.4,
                    "r2_pt_before": 0.8, "r2_pt_after": 0.75, "delta_r2_pt": -0.05,
                },
                {
                    "trunk": "b", "t_before": 2, "t_after": 3, "pair": "B: 2-3",
                    "r2_p0_before": 0.6, "r2_p0_after": 0.5, "delta_r2_p0": -0.1,
                    "r2_pt_before": 0.6, "r2_pt_after": 0.6, "delta_r2_pt": 0.0,
                },
            ]
        )

    def test_a_connected_pair_per_realignment_step(self) -> None:
        fig = figures.phase_contrast(self._pairs())
        ax = fig.axes[0]
        assert len(ax.lines) == 2  # one connector per pair
        assert len(ax.collections) == 4  # before and after marker per pair

    def test_shows_levels_as_well_as_the_difference(self) -> None:
        """A drop from 0.9 to 0.6 and one from 0.4 to 0.1 are the same bar and
        very different findings, so the markers carry the level."""
        fig = figures.phase_contrast(self._pairs())
        ys = np.concatenate(
            [c.get_offsets()[:, 1] for c in fig.axes[0].collections]
        )
        assert sorted(ys) == pytest.approx([0.4, 0.5, 0.6, 0.8])
        assert any("-0.40" in t.get_text() for t in fig.axes[0].texts)

    def test_both_projection_series_get_a_panel(self) -> None:
        fig = figures.phase_contrast(self._pairs())
        assert len(fig.axes) == 2


class TestOverlayLines:
    def test_one_line_per_series_plus_the_reference(self) -> None:
        fig = figures.overlay_lines(
            {"a": [1.0, 2.0], "b": [3.0, 4.0]},
            ylabel="y",
            reference=100.0,
            reference_label="ref",
        )
        (ax,) = fig.axes
        assert len(ax.lines) == 3

    def test_dataset_marks_replace_the_categorical_hues(self) -> None:
        """Eight probes would take all eight categorical slots and leave the
        reader an arbitrary dataset-to-colour map to memorise."""
        marks = {
            "Evil (II)": style.dataset_mark("evil/misaligned_2"),
            "Code (Normal)": style.dataset_mark("insecure_code/normal"),
        }
        fig = figures.overlay_lines(
            {"Evil (II)": [1.0, 2.0], "Code (Normal)": [3.0, 4.0]},
            ylabel="y",
            marks=marks,
        )
        (ax,) = fig.axes
        drawn = {line.get_label(): line for line in ax.lines}
        assert drawn["Evil (II)"].get_marker() == style.DATASET_MARKERS["evil"]
        assert drawn["Evil (II)"].get_color() == style.VERSION_LINE["misaligned_2"]
        assert drawn["Code (Normal)"].get_markerfacecolor() == style.SURFACE

    def test_a_replicate_shares_its_series_colour_and_is_dashed(self) -> None:
        """The reseed rides the colour of the run it replicates, so n = 2
        costs no extra hue and cannot be read as a fourth condition."""
        fig = figures.overlay_lines(
            {"a": [1.0, 2.0]}, {"a": [1.1, 2.1]}, ylabel="y"
        )
        (ax,) = fig.axes
        primary, replicate = ax.lines[0], ax.lines[1]
        assert primary.get_color() == replicate.get_color()
        assert primary.get_linestyle() == "-"
        assert replicate.get_linestyle() != "-"

    def test_the_dashed_convention_is_named_in_the_legend(self) -> None:
        fig = figures.overlay_lines({"a": [1.0]}, {"a": [1.1]}, ylabel="y")
        texts = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert any("dashed" in t for t in texts)


class TestOverlayGrid:
    def test_one_panel_per_component(self) -> None:
        panels = {r"$\rho_t$": {"a": [1.0, 0.9]}, r"$r_t$": {"a": [30.0, 31.0]}}
        fig = figures.overlay_grid(panels)
        assert len(fig.axes) == 2
        assert [ax.get_ylabel() for ax in fig.axes] == list(panels)

    def test_only_the_bottom_row_is_labelled(self) -> None:
        """The panels share an x-axis, so a label under the top row would name
        ticks that are not drawn."""
        panels = {f"p{i}": {"a": [1.0, 2.0]} for i in range(4)}
        fig = figures.overlay_grid(panels, ncols=2)
        assert [bool(ax.get_xlabel()) for ax in fig.axes] == [False, False, True, True]


# --- style.py --------------------------------------------------------------


class TestStyle:
    def test_apply_style_is_idempotent(self) -> None:
        style.apply_style()
        style.apply_style()  # must not raise

    def test_save_figure_writes_png_and_pdf(self, tmp_path) -> None:
        fig = figures.diversity_bar(synthetic.synthetic_diversity_frame(n_seeds=2))
        png_path, pdf_path = style.save_figure(fig, "unit_test_figure", tmp_path)
        assert png_path.exists() and png_path.stat().st_size > 0
        assert pdf_path.exists() and pdf_path.stat().st_size > 0
        assert png_path.suffix == ".png"
        assert pdf_path.suffix == ".pdf"

    def test_categorical_color_wraps(self) -> None:
        assert style.categorical_color(0) == style.categorical_color(
            len(style.CATEGORICAL)
        )


# --- make_plots.py ------------------------------------------------------------


class TestDefaultOutDir:
    def test_every_run_source_gets_its_own_directory(self) -> None:
        """Sharing one directory let a mock run overwrite real figures under
        their exact filenames, with nothing in the file saying which it was."""
        dirs = {
            make_plots.default_out_dir(local=local, mock=mock)
            for local in (False, True)
            for mock in (False, True)
        }
        assert len(dirs) == 4
        assert make_plots.default_out_dir() == style.PLOTS_DIR / "real"

    def test_the_synthetic_figures_live_outside_all_of_them(self) -> None:
        """demo.py writes to plots/ itself; nothing real may land there."""
        for local in (False, True):
            for mock in (False, True):
                assert (
                    make_plots.default_out_dir(local=local, mock=mock)
                    != style.PLOTS_DIR
                )


class TestBuildAndSaveOutputLayout:
    def test_each_experiment_family_gets_its_own_subdirectory(
        self, monkeypatch, tmp_path
    ) -> None:
        """--experiment exp3 must land in <out_dir>/exp3, not mixed flat into
        the shared run-source directory with exp2/exp4 figures."""
        recorded: dict[str, Path] = {}

        def fake_build_exp2(_collections, out_dir, **_kwargs) -> list[Path]:
            recorded["exp2"] = out_dir
            return []

        def fake_build(group: str):
            def _build(_collection, out_dir: Path) -> list[Path]:
                recorded[group] = out_dir
                return []

            return _build

        monkeypatch.setattr(
            make_plots, "collect_group", lambda group, **_kw: Collection(group)
        )
        monkeypatch.setattr(make_plots, "build_exp2", fake_build_exp2)
        monkeypatch.setattr(
            make_plots,
            "BUILDERS",
            {
                experiments.EXP3: fake_build(experiments.EXP3),
                experiments.EXP4: fake_build(experiments.EXP4),
            },
        )

        make_plots.build_and_save(tmp_path, groups=list(make_plots.GROUPS))

        assert recorded["exp2"] == tmp_path / "exp2"
        assert recorded[experiments.EXP3] == tmp_path / "exp3"
        assert recorded[experiments.EXP4] == tmp_path / "exp4"


class TestExp2Driver:
    def test_every_exp2_family_is_reachable_from_the_cli(self) -> None:
        assert set(make_plots.EXP2_GROUPS) <= set(make_plots.GROUPS)
        assert experiments.EXP2_DECAY in make_plots.GROUPS

    def test_trunk_colour_follows_the_trunk_not_the_row_order(self) -> None:
        """A reader who learned "A is blue" must not be repainted because a
        partial sweep left A out of the frame."""
        full = make_plots._trunk_colors(["a", "b", "c"])
        partial = make_plots._trunk_colors(["b", "c"])
        assert partial == {k: v for k, v in full.items() if k != "a"}

    def test_trunks_are_ordered_by_the_ladder_not_alphabetically(self) -> None:
        frame = pd.DataFrame({"trunk": ["c", "a", "b"]})
        assert make_plots._present_trunks(frame) == ["a", "b", "c"]

    def test_an_unknown_trunk_still_appears(self) -> None:
        frame = pd.DataFrame({"trunk": ["z", "a"]})
        assert make_plots._present_trunks(frame) == ["a", "z"]

    def test_series_are_ordered_by_checkpoint(self) -> None:
        frame = pd.DataFrame(
            {"probe": ["p", "p", "p"], "t": [2, 0, 1], "ratio": [30.0, 10.0, 20.0]}
        )
        assert make_plots._series_by(frame, key="probe", value="ratio") == {
            "p": [10.0, 20.0, 30.0]
        }

    def test_a_series_missing_a_checkpoint_is_dropped(self, caplog) -> None:
        """A truncated line reads as a quantity that stopped moving, not as a
        measurement that never happened."""
        frame = pd.DataFrame(
            {
                "probe": ["p", "p", "q"],
                "t": [0, 1, 0],
                "ratio": [10.0, 20.0, 30.0],
            }
        )
        with caplog.at_level(logging.WARNING):
            series = make_plots._series_by(frame, key="probe", value="ratio")
        assert set(series) == {"p"}
        assert "q" in caplog.text

    def test_the_reseed_is_split_off_by_seed(self) -> None:
        frame = pd.DataFrame({"seed": [0, 0, 1], "value": [1.0, 2.0, 3.0]})
        own, replicate = make_plots._split_by_seed(frame, 0)
        assert list(own["value"]) == [1.0, 2.0]
        assert list(replicate["value"]) == [3.0]


# --- demo.py -----------------------------------------------------------------


class TestDemo:
    def test_build_and_save_writes_all_figures(self, tmp_path) -> None:
        saved = build_and_save(tmp_path, n_seeds=2)
        assert len(saved) == 14  # 7 figures, 2 files (png+pdf) each
        for path in saved:
            assert path.exists()
            assert path.stat().st_size > 0
