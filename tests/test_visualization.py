"""Tests for :mod:`method.visualization`.

Every figure function is exercised on synthetic data (never a real
trajectory), so these tests run with no GPU, no judge, and no fine-tuning.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from method.visualization import figures, schema, style, synthetic
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
        assert len(df) == 2 * 3 * 2  # datasets * seeds * conditions
        assert set(df["condition"]) == {"fresh", "after_realignment"}

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
        assert len(ax.patches) == 2 * 2  # 2 datasets * 2 conditions
        assert ax.get_legend() is not None

    def test_xtick_labels_are_pretty_by_default(self) -> None:
        df = synthetic.synthetic_hysteresis_frame(
            ("hallucination/misaligned_1", "mistake_gsm8k/misaligned_2"), n_seeds=2
        )
        fig = figures.hysteresis_bar(df)
        (ax,) = fig.axes
        tick_text = [t.get_text() for t in ax.get_xticklabels()]
        assert tick_text == ["Hallucination (I)", "GSM8K (Mistake II)"]

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


# --- demo.py -----------------------------------------------------------------


class TestDemo:
    def test_build_and_save_writes_all_figures(self, tmp_path) -> None:
        saved = build_and_save(tmp_path, n_seeds=2)
        assert len(saved) == 14  # 7 figures, 2 files (png+pdf) each
        for path in saved:
            assert path.exists()
            assert path.stat().st_size > 0
