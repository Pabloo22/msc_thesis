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
from method.visualization import (
    decay,
    figures,
    labels,
    make_plots,
    schema,
    style,
    synthetic,
)
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

    def test_correlation_matches_numpy_and_keeps_the_slope_sign(self) -> None:
        """The figures annotate r, not R^2; a sign dropped there would read as
        a probe set ordered the wrong way round."""
        rng = np.random.default_rng(0)
        x = np.arange(30, dtype=float)
        for direction in (1.0, -1.0):
            y = direction * 3.0 * x + rng.normal(0, 20, size=30)
            fit = linear_fit(x, y)
            assert fit.corr == pytest.approx(np.corrcoef(x, y)[0, 1])
            assert np.sign(fit.corr) == direction

    def test_a_degenerate_fit_has_no_correlation(self) -> None:
        assert linear_fit([1.0, 1.0, 1.0], [4.0, 6.0, 5.0]).corr == 0.0


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
    def test_each_dataset_gets_a_panel_of_one_bar_per_condition(self) -> None:
        df = synthetic.synthetic_hysteresis_frame(("a/normal", "b/normal"), n_seeds=3)
        fig = figures.hysteresis_bar(df)
        assert len(fig.axes) == 2
        for ax in fig.axes:
            assert len(ax.patches) == len(synthetic.HYSTERESIS_CONDITIONS)

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

    def test_panel_headers_are_pretty_dataset_names(self) -> None:
        df = synthetic.synthetic_hysteresis_frame(
            ("hallucination/misaligned_1", "mistake_gsm8k/misaligned_2"), n_seeds=2
        )
        fig = figures.hysteresis_bar(df)
        assert [ax.get_title() for ax in fig.axes] == [
            "Hallucination (I)",
            "GSM8K (Mistake II)",
        ]

    def test_each_arm_is_named_on_its_own_tick_not_only_by_colour(self) -> None:
        """The schedules are what make the legend colour-free; see the docstring."""
        df = synthetic.synthetic_hysteresis_frame(("a/normal",), n_seeds=2)
        fig = figures.hysteresis_bar(df)
        (ax,) = fig.axes
        assert [t.get_text() for t in ax.get_xticklabels()] == [
            labels.HYSTERESIS_CONDITION_SEQUENCES[c]
            for c in synthetic.HYSTERESIS_CONDITIONS
        ]

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
        (legend,) = fig.legends
        assert r"Base model $b_0$" in [t.get_text() for t in legend.get_texts()]

    def test_the_legend_names_only_the_two_line_marks(self) -> None:
        """Arm identity lives on the ticks, so a colour must never need a key."""
        df = synthetic.synthetic_hysteresis_frame(("a/normal",), n_seeds=2)
        fig = figures.hysteresis_bar(
            df, reference=8.0, reference_label="Base", start_col="behavior_before"
        )
        (legend,) = fig.legends
        assert [t.get_text() for t in legend.get_texts()] == [
            "Base",
            "Before the final step",
        ]

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
        assert ax.get_title() == "Custom"

    def _two_rows(self) -> pd.DataFrame:
        """The same arms twice over, the second an order of magnitude higher."""
        base = synthetic.synthetic_hysteresis_frame(("a/normal",), n_seeds=2)
        return pd.concat(
            [
                base.assign(row="first"),
                base.assign(row="second", behavior=base["behavior"] * 10),
            ],
            ignore_index=True,
        )

    def test_a_row_per_key_each_with_its_own_reference_line(self) -> None:
        """$b_0$ is a property of the trait a row measures, not of the figure."""
        fig = figures.hysteresis_bar(
            self._two_rows(),
            rows=["first", "second"],
            row_col="row",
            row_labels={"first": "First", "second": "Second"},
            reference={"first": 8.0, "second": 80.0},
        )
        assert [ax.get_ylabel() for ax in fig.axes] == ["First", "Second"]
        dashed = [
            ln.get_ydata()[0]
            for ax in fig.axes
            for ln in ax.lines
            if ln.get_linestyle() == "--"
        ]
        assert dashed == [8.0, 80.0]

    def test_only_rows_in_one_scale_group_share_a_y_axis(self) -> None:
        """Two traits on one scale would spend the smaller one's panels on the
        larger one's empty space; two re-alignment sources for one trait must
        share, since comparing them is the point."""
        rows = ["first", "second"]
        together = figures.hysteresis_bar(self._two_rows(), rows=rows, row_col="row")
        assert len({ax.get_ylim() for ax in together.axes}) == 1
        apart = figures.hysteresis_bar(
            self._two_rows(),
            rows=rows,
            row_col="row",
            row_scales={"first": "a", "second": "b"},
        )
        assert len({ax.get_ylim() for ax in apart.axes}) == 2


class TestDiversityBar:
    def test_bar_count_matches_conditions(self) -> None:
        df = synthetic.synthetic_diversity_frame(n_seeds=3)
        fig = figures.diversity_bar(
            df, order=synthetic.DIVERSITY_CONDITIONS, labels=synthetic.DIVERSITY_LABELS
        )
        (ax,) = fig.axes
        assert len(ax.patches) == len(synthetic.DIVERSITY_CONDITIONS)

    def test_a_panel_per_trait_and_re_alignment_source(self) -> None:
        base = synthetic.synthetic_diversity_frame(n_seeds=2)
        df = pd.concat(
            [
                base.assign(trait=trait, realign_trait=realign)
                for trait in ("sycophantic", "evil")
                for realign in ("sycophantic", "evil")
            ],
            ignore_index=True,
        )
        fig = figures.diversity_bar(
            df,
            rows=["sycophantic", "evil"],
            row_labels={"sycophantic": "Sycophancy", "evil": "Evil"},
            cols=["sycophantic", "evil"],
            order=synthetic.DIVERSITY_CONDITIONS,
        )
        assert len(fig.axes) == 4
        assert [ax.get_ylabel() for ax in fig.axes] == ["Sycophancy", "", "Evil", ""]


# --- figures.py: the RQ1 decay set -------------------------------------------


def _decay_rows(trunks=("a", "b"), checkpoints=range(3), n_probes=4) -> pd.DataFrame:
    """A decay frame in the shape :mod:`method.visualization.decay` emits."""
    rng = np.random.default_rng(0)
    rows = pd.DataFrame(
        [
            {
                "trunk": trunk,
                "t": t,
                "probe": f"d{i}/normal",
                "steps_since_realignment": t % 2,
                "delta_p_0": float(i),
                "delta_p_t": float(i) + t,
                "b_t": 40.0 + t,
                "delta_b": 2.0 * i + rng.normal(0, 0.1),
                "se_b_next": 0.3,
                "se_delta_b": 0.4,
            }
            for trunk in trunks
            for t in checkpoints
            for i in range(n_probes)
        ]
    )
    # Derived rather than drawn, so the fixture keeps the one relation the
    # figure relies on: b_t is constant within a panel, so a fit against
    # b_next is the fit against delta_b shifted.
    return rows.assign(b_next=rows["b_t"] + rows["delta_b"])


def _with_recomputed(
    rows: pd.DataFrame, trunks=("a",), columns=("delta_p",)
) -> pd.DataFrame:
    r"""``rows`` with re-measured projection columns on ``trunks`` alone.

    Partial by construction: each re-measured series is paid for per trunk, so
    a grid that cannot mix a re-measured trunk with un-measured ones cannot
    draw the real figure.
    """
    return rows.assign(
        **{
            column: np.where(
                rows["trunk"].isin(trunks), rows["delta_p_0"] + 0.5 * (i + 1), np.nan
            )
            for i, column in enumerate(columns)
        }
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
                "corr_p0": 0.8 - 0.1 * t,
                "corr_p0_lo": 0.6 - 0.1 * t,
                "corr_p0_hi": 0.9,
                "corr_pt": 0.8,
                "corr_pt_lo": 0.7,
                "corr_pt_hi": 0.9,
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


def _traited(frame, traits=("sycophantic", "evil"), *, scale=None) -> pd.DataFrame:
    """``frame`` once per trait, in the shape the merged figures now receive.

    Each trait after the first has its ``scale`` column multiplied by a further
    factor of ten, which is what makes an axis-sharing assertion mean anything:
    two copies of the same numbers land on the same scale whether or not the
    figure joined their axes.
    """
    copies = []
    for i, trait in enumerate(traits):
        copy = frame.assign(trait=trait)
        if scale is not None:
            copy[scale] = copy[scale] * 10**i
        copies.append(copy)
    return pd.concat(copies, ignore_index=True)


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


def _validation_rows(traits=("sycophantic", "evil"), se=0.4) -> pd.DataFrame:
    """A validation fan in the shape :func:`decay.validation_frame` emits."""
    return pd.DataFrame(
        [
            {
                "trait": trait,
                "dataset": dataset,
                "delta_p_0": float(i),
                "delta_b": 2.0 * i,
                "se_delta_b": se,
            }
            for trait in traits
            for i, dataset in enumerate(
                ["evil/normal", "evil/misaligned_2", "sycophancy/normal"]
            )
        ]
    )


class TestScatterValidation:
    def test_one_panel_per_trait(self) -> None:
        fig = figures.scatter_validation(_validation_rows())
        assert len(fig.axes) == 2

    def test_the_traits_keep_their_own_scales(self) -> None:
        """Each is a different persona vector and a different judge, so a
        shared axis would invite reading one trait's spread against the
        other's."""
        rows = _validation_rows()
        rows.loc[rows["trait"] == "evil", "delta_b"] *= 20.0
        fig = figures.scatter_validation(rows, traits=["sycophantic", "evil"])
        assert fig.axes[0].get_ylim() != fig.axes[1].get_ylim()

    def test_marks_are_grouped_into_one_call_per_shape(self) -> None:
        """A scatter takes a single marker, so eight families are eight calls
        over disjoint index sets."""
        fig = figures.scatter_validation(_validation_rows(traits=("evil",)))
        (ax,) = fig.axes
        scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(scatters) == 2  # pentagon and triangle
        assert sum(len(c.get_offsets()) for c in scatters) == 3

    def test_the_encoding_key_is_stated_once_for_the_figure(self) -> None:
        """Both panels draw the same 24 datasets, so a key per panel would be
        the same eleven entries twice."""
        fig = figures.scatter_validation(_validation_rows())
        assert all(ax.get_legend() is None for ax in fig.axes)
        (legend,) = fig.legends
        assert [t.get_text() for t in legend.get_texts()] == [
            "Evil",
            "Sycophancy",
            "Normal",
            "II",
        ]

    def test_each_panel_reports_its_own_fit(self) -> None:
        """One series per panel, so its statistics are annotated in the panel
        rather than put in a legend."""
        fig = figures.scatter_validation(_validation_rows())
        for ax in fig.axes:
            annotations = [t.get_text() for t in ax.texts]
            assert any(r"$r$" in a and "slope" in a for a in annotations)
            assert any("n$ = 3 datasets" in a for a in annotations)

    def test_a_trait_with_no_runs_is_marked_not_dropped(self) -> None:
        fig = figures.scatter_validation(
            _validation_rows(traits=("evil",)), traits=["sycophantic", "evil"]
        )
        assert any("not run" in t.get_text() for t in fig.axes[0].texts)

    def test_error_bars_are_drawn_when_given(self) -> None:
        plain = figures.scatter_validation(_validation_rows(traits=("evil",), se=0.0))
        with_err = figures.scatter_validation(_validation_rows(traits=("evil",)))
        assert len(with_err.axes[0].collections) > len(plain.axes[0].collections)

    def test_all_zero_error_bars_are_skipped(self) -> None:
        """A run evaluated with one generation per question has no
        within-question spread; flat caps would imply a measurement it did
        not make."""
        rows = _validation_rows(traits=("evil",))
        zeroed = figures.scatter_validation(rows.assign(se_delta_b=0.0))
        dropped = figures.scatter_validation(rows.drop(columns="se_delta_b"))
        assert len(zeroed.axes[0].collections) == len(dropped.axes[0].collections)


#: What the judge's scale runs between, as the figures' own constants must
#: agree it does.
BEHAVIOUR_FLOOR, BEHAVIOUR_CEILING = 0.0, 100.0


class TestDecayScatterGrid:
    def test_one_panel_per_trunk_and_checkpoint(self) -> None:
        fig = figures.decay_scatter_grid(_decay_rows())
        assert len(fig.axes) == 2 * 3

    def test_every_panel_holds_both_series_and_both_correlations(self) -> None:
        fig = figures.decay_scatter_grid(_decay_rows())
        ax = fig.axes[0]
        scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(scatters) == 2  # Delta P_0 and Delta P_t
        annotations = " ".join(t.get_text() for t in ax.texts)
        assert r"$r(\Delta P_0)$" in annotations
        assert r"$r(\Delta \hat{P}_t)$" in annotations

    def test_a_measured_panel_gains_the_recomputed_series(self) -> None:
        rows = _with_recomputed(_decay_rows())
        fig = figures.decay_scatter_grid(rows, trunks=["a"])
        ax = fig.axes[0]
        scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(scatters) == 3
        assert r"$r(\Delta P_t)$" in " ".join(t.get_text() for t in ax.texts)

    def test_all_four_series_draw_when_all_are_measured(self) -> None:
        rows = _with_recomputed(
            _decay_rows(), columns=("delta_p_v0", "delta_p")
        )
        fig = figures.decay_scatter_grid(rows, trunks=["a"])
        scatters = [
            c for c in fig.axes[0].collections if isinstance(c, PathCollection)
        ]
        assert len(scatters) == 4
        keys = [t.get_text() for t in fig.legends[0].get_texts()]
        for label in decay.SERIES_LABELS.values():
            assert any(k.startswith(label) for k in keys), label

    def test_an_unmeasured_panel_keeps_the_two_it_has(self) -> None:
        """One grid mixes the trunk that was re-measured with the ones that
        were not, so a panel draws what it has rather than all or nothing."""
        rows = _with_recomputed(_decay_rows())
        fig = figures.decay_scatter_grid(rows, trunks=["a", "b"])
        panels = [
            len([c for c in ax.collections if isinstance(c, PathCollection)])
            for ax in fig.axes
        ]
        assert set(panels) == {2, 3}

    def test_a_series_no_panel_drew_is_kept_out_of_the_legend(self) -> None:
        """A key for a line nobody drew reads as a line that came out flat."""
        (plain,) = figures.decay_scatter_grid(_decay_rows()).legends
        (mixed,) = figures.decay_scatter_grid(
            _with_recomputed(_decay_rows())
        ).legends
        def keys(legend):
            return [t.get_text() for t in legend.get_texts()]

        assert not any(t.startswith(r"$\Delta P_t$") for t in keys(plain))
        assert any(t.startswith(r"$\Delta P_t$") for t in keys(mixed))

    def test_the_panel_plots_the_raw_level_against_the_one_it_started_from(
        self,
    ) -> None:
        """Delta b hides where on the judge's scale a step landed; the raw
        b_{t+1} against a rule at b_t shows both that and the size of the
        move."""
        rows = _decay_rows()
        fig = figures.decay_scatter_grid(rows)
        panel = rows[(rows["trunk"] == "a") & (rows["t"] == 0)]
        ax = fig.axes[0]
        drawn = np.concatenate(
            [c.get_offsets()[:, 1] for c in ax.collections
             if isinstance(c, PathCollection)]
        )
        assert set(np.round(drawn, 6)) == set(np.round(panel["b_next"], 6))
        assert [line.get_ydata()[0] for line in ax.lines
                if line.get_color() == style.INK] == [panel["b_t"].iloc[0]]

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
        assert "steps since re-alignment: 2" in titles

    def test_a_checkpoint_with_no_runs_is_marked_not_dropped(self) -> None:
        """A half-finished fan must leave its column visibly empty; silently
        narrowing the grid would hide which checkpoints are still missing."""
        rows = _decay_rows()
        fig = figures.decay_scatter_grid(
            rows[rows["t"] != 1], checkpoints=[0, 1, 2]
        )
        empty = [ax for ax in fig.axes if not ax.collections]
        assert len(empty) == 2  # one per trunk
        assert all(
            any("not run" in text.get_text() for text in ax.texts) for ax in empty
        )

    def test_each_trait_is_a_block_of_trunk_rows(self) -> None:
        fig = figures.decay_scatter_grid(
            _traited(_decay_rows()),
            traits=["sycophantic", "evil"],
            trait_labels={"sycophantic": "Sycophancy", "evil": "Evil"},
        )
        assert len(fig.axes) == 2 * 2 * 3  # traits x trunks x checkpoints
        # A row names both halves of what it is; the trunk alone would repeat.
        assert [ax.get_ylabel() for ax in fig.axes if ax.get_ylabel()] == [
            "Sycophancy\nTrunk a",
            "Sycophancy\nTrunk b",
            "Evil\nTrunk a",
            "Evil\nTrunk b",
        ]

    def test_delta_p_is_not_put_on_one_scale_across_traits(self) -> None:
        """Delta P is read against a different persona vector per trait, so one
        scale would compare numbers that are not the same number -- while
        within a trait it still must."""
        fig = figures.decay_scatter_grid(
            _traited(_decay_rows(), scale="delta_p_0"),
            traits=["sycophantic", "evil"],
        )
        assert len({ax.get_xlim() for ax in fig.axes}) == 2

    def test_behaviour_is_always_drawn_on_the_whole_judge_scale(self) -> None:
        """Unlike Delta P, b is the same 0-100 judge average in every panel, so
        a mark's height means one thing across the whole figure -- and a trunk
        that stayed near the floor does not fill its panel the way one that
        climbed to the ceiling does."""
        rows = _traited(_decay_rows(), scale="b_next")
        fig = figures.decay_scatter_grid(rows, traits=["sycophantic", "evil"])
        assert {ax.get_ylim() for ax in fig.axes} == {figures.BEHAVIOUR_LIMITS}
        assert all(
            tuple(ax.get_yticks()) == figures.BEHAVIOUR_TICKS for ax in fig.axes
        )

    def test_the_scale_is_padded_so_a_mark_at_zero_is_drawn_whole(self) -> None:
        """A probe scoring 0 is ordinary -- an aligned model on an evil judge --
        and a mark centred on the spine would be sliced in half by it, which
        reads as a different mark rather than as a point at the floor."""
        low, high = figures.BEHAVIOUR_LIMITS
        assert low < BEHAVIOUR_FLOOR and high > BEHAVIOUR_CEILING
        # The ticks still say what the judge's range actually is.
        assert figures.BEHAVIOUR_TICKS[0] == BEHAVIOUR_FLOOR
        assert figures.BEHAVIOUR_TICKS[-1] == BEHAVIOUR_CEILING


class TestHeadlineCurves:
    def test_one_row_each_for_corr_and_slope_with_both_series_overlaid(
        self,
    ) -> None:
        fig = figures.headline_curves(_fits())
        assert len(fig.axes) == 2  # no trait facet: one column, two rows
        ylabels = [ax.get_ylabel() for ax in fig.axes]
        assert any("Correlation" in label for label in ylabels)
        assert any("slope" in label for label in ylabels)
        # 2 trunks x (p0 + pt) drawn on the one shared correlation axes.
        assert len(fig.axes[0].lines) == 4

    def test_all_positive_correlations_keep_the_unit_interval(self) -> None:
        """Fixing every panel at the full [-1, 1] would spend half the height
        on a sign the data never takes, and the decay this figure is for
        happens inside the top half."""
        fig = figures.headline_curves(_fits())
        low, high = fig.axes[0].get_ylim()
        assert -0.1 < low <= 0 and 1.0 <= high <= 1.05

    def test_a_negative_correlation_widens_the_row_to_the_full_range(
        self,
    ) -> None:
        """A sign is worth an axis: clipping a series that goes negative would
        hide the one thing correlation reports that R^2 cannot."""
        fits = _fits()
        fits.loc[fits["t"] == 2, "corr_p0"] = -0.3
        fig = figures.headline_curves(fits)
        low, _ = fig.axes[0].get_ylim()
        assert low <= -1.0
        # ...and zero becomes a level worth drawing, which on [0, 1] it is not.
        assert any(
            line.get_ydata()[0] == 0 for line in fig.axes[0].lines
            if len(line.get_ydata()) and line.get_color() == style.BASELINE
        )

    def test_p0_is_solid_and_pt_is_dashed(self) -> None:
        fig = figures.headline_curves(_fits(trunks=("a",)))
        linestyles = {line.get_linestyle() for line in fig.axes[0].lines}
        assert len(linestyles) == 2

    def test_a_third_series_gets_a_style_and_a_legend_key_of_its_own(self) -> None:
        r"""Colour already carries the trunk, so $\Delta P$ has to be
        distinguishable from the two frozen series by line style alone."""
        fits = _fits(trunks=("a",)).assign(
            corr_p=0.9, corr_p_lo=0.85, corr_p_hi=0.95,
            slope_p=2.2, slope_p_lo=2.0, slope_p_hi=2.4,
        )
        fig = figures.headline_curves(
            fits, series=["p0", "pt", "p"], series_labels=decay.SERIES_LABELS
        )
        assert len({line.get_linestyle() for line in fig.axes[0].lines}) == 3
        keys = [t.get_text() for t in fig.legends[0].get_texts()]
        assert decay.SERIES_LABELS["p"] in keys

    def test_the_r2_max_column_is_no_longer_needed_to_plot(self) -> None:
        """The noise ceiling is still computed by fit_frame, but this figure
        no longer draws it -- see docs/r2_max.md -- so it must not require the
        column to be present."""
        fits = _fits().drop(columns=["r2_max"])
        figures.headline_curves(fits)  # must not raise

    def test_a_trunk_missing_from_the_frame_is_simply_absent(self) -> None:
        fig = figures.headline_curves(_fits(trunks=("a",)))
        legend = fig.legends[0]
        assert len([t for t in legend.get_texts() if "Trunk" in t.get_text()]) == 1

    def test_one_column_per_trait_two_rows_for_corr_and_slope(self) -> None:
        fig = figures.headline_curves(
            _traited(_fits()),
            traits=["sycophantic", "evil"],
            trait_labels={"sycophantic": "Sycophancy", "evil": "Evil"},
        )
        assert len(fig.axes) == 2 * 2  # 2 quantities x 2 traits
        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert titles == ["Sycophancy", "Evil"]
        ylabels = [ax.get_ylabel() for ax in fig.axes if ax.get_ylabel()]
        assert any("Correlation" in label for label in ylabels)
        assert any("slope" in label for label in ylabels)

    def test_the_slope_is_not_shared_across_traits(self) -> None:
        """A slope is in points of that trait's judge per unit of that trait's
        persona vector, so its absolute value is not comparable between them."""
        fig = figures.headline_curves(
            _traited(_fits(), scale="slope_p0"), traits=["sycophantic", "evil"]
        )
        slope_row = [fig.axes[2], fig.axes[3]]  # row 1, both trait columns
        assert slope_row[0].get_ylim() != slope_row[1].get_ylim()

    def test_the_legend_keys_a_trunk_once_for_the_whole_grid(self) -> None:
        fig = figures.headline_curves(_traited(_fits()))
        texts = [t.get_text() for t in fig.legends[0].get_texts()]
        assert texts.count("Trunk a") == 1

    def test_the_legend_also_names_the_two_series(self) -> None:
        fig = figures.headline_curves(
            _fits(), series_labels={"p0": "P0", "pt": "Pt"}
        )
        texts = [t.get_text() for t in fig.legends[0].get_texts()]
        assert "P0" in texts and "Pt" in texts


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
        fig = figures.mechanism_grid(rows, self.PREDICTORS)
        annotations = [text.get_text() for text in fig.axes[0].texts]
        assert f"$n$ = {len(rows)} checkpoints" in annotations

    def test_colour_identifies_the_trunk(self) -> None:
        fig = figures.mechanism_grid(
            _fits(), self.PREDICTORS, trunk_colors={"a": style.BLUE, "b": style.ORANGE}
        )
        assert len(fig.axes[0].collections) == 2

    def test_a_trait_per_row_each_counting_its_own_checkpoints(self) -> None:
        """``n`` is per row, not per figure: a half-run trait regresses fewer
        checkpoints than a finished one, and the figure must not claim
        otherwise."""
        rows = pd.concat(
            [
                _fits().assign(trait="sycophantic"),
                _fits(trunks=("a",)).assign(trait="evil"),
            ],
            ignore_index=True,
        )
        fig = figures.mechanism_grid(
            rows, self.PREDICTORS, traits=["sycophantic", "evil"]
        )
        assert len(fig.axes) == 2 * 2
        first_of_each_row = [fig.axes[0], fig.axes[2]]
        counts = [
            [t.get_text() for t in ax.texts if "checkpoints" in t.get_text()]
            for ax in first_of_each_row
        ]
        assert counts == [["$n$ = 6 checkpoints"], ["$n$ = 3 checkpoints"]]

    def test_the_correlation_is_the_one_quantity_the_traits_do_share(self) -> None:
        fig = figures.mechanism_grid(
            _traited(_fits(), scale="rho"),
            self.PREDICTORS,
            traits=["sycophantic", "evil"],
        )
        assert len({ax.get_ylim() for ax in fig.axes}) == 1
        # ...but the predictor they are regressed on is not.
        assert fig.axes[0].get_xlim() != fig.axes[2].get_xlim()


class TestPhaseContrast:
    def _pairs(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "trunk": "a", "t_before": 1, "t_after": 2, "pair": "A: 1-2",
                    "corr_p0_before": 0.8, "corr_p0_after": 0.4,
                    "delta_corr_p0": -0.4,
                    "corr_pt_before": 0.8, "corr_pt_after": 0.75,
                    "delta_corr_pt": -0.05,
                },
                {
                    "trunk": "b", "t_before": 2, "t_after": 3, "pair": "B: 2-3",
                    "corr_p0_before": 0.6, "corr_p0_after": 0.5,
                    "delta_corr_p0": -0.1,
                    "corr_pt_before": 0.6, "corr_pt_after": 0.6,
                    "delta_corr_pt": 0.0,
                },
            ]
        )

    def test_four_bars_per_realignment_step(self) -> None:
        fig = figures.phase_contrast(self._pairs())
        ax = fig.axes[0]
        # 2 steps x 2 series (p0, pt) x (before + after).
        assert len(ax.patches) == 2 * 2 * 2

    def test_a_third_series_adds_a_pair_and_narrows_the_bars(self) -> None:
        r"""The group has a fixed budget of the step's width, so $\Delta P$
        joins by making every bar narrower rather than by colliding with the
        neighbouring step."""
        pairs = self._pairs().assign(
            corr_p_before=0.8, corr_p_after=0.7, delta_corr_p=-0.1
        )
        fig = figures.phase_contrast(pairs, series=["p0", "pt", "p"])
        ax = fig.axes[0]
        assert len(ax.patches) == 2 * 3 * 2
        widths = {round(p.get_width(), 6) for p in ax.patches}
        narrow = {
            round(p.get_width(), 6)
            for p in figures.phase_contrast(self._pairs()).axes[0].patches
        }
        assert max(widths) < max(narrow)
        centres = sorted(p.get_x() + p.get_width() / 2 for p in ax.patches)
        assert max(centres) - min(centres) < len(self._pairs()) - 1 + 1.0

    def test_a_series_this_trunk_lacks_leaves_a_gap(self) -> None:
        """The recomputed series is measured on one trunk; zero-height bars on
        the others would read as "no change" rather than as "not measured"."""
        pairs = self._pairs().assign(
            corr_p_before=[0.8, np.nan],
            corr_p_after=[0.7, np.nan],
            delta_corr_p=[-0.1, np.nan],
        )
        fig = figures.phase_contrast(pairs, series=["p0", "pt", "p"])
        # Both steps keep their two frozen pairs; only one gains a third.
        assert len(fig.axes[0].patches) == 2 * 2 * 2 + 2

    def test_shows_levels_as_well_as_the_difference(self) -> None:
        """A drop from 0.9 to 0.6 and one from 0.4 to 0.1 are the same
        difference and very different findings, so the bars carry the
        level."""
        fig = figures.phase_contrast(self._pairs())
        heights = [p.get_height() for p in fig.axes[0].patches]
        assert sorted(heights) == pytest.approx(
            sorted([0.8, 0.4, 0.8, 0.75, 0.6, 0.5, 0.6, 0.6])
        )
        assert any("-0.40" in t.get_text() for t in fig.axes[0].texts)

    def test_both_projection_series_share_one_panel_distinguished_by_hatch(
        self,
    ) -> None:
        fig = figures.phase_contrast(self._pairs())
        assert len(fig.axes) == 1
        hatches = {p.get_hatch() for p in fig.axes[0].patches}
        assert len(hatches) == 2  # p0 plain, pt hatched

    def test_a_trait_per_column_over_one_shared_list_of_steps(self) -> None:
        """A pair only one trait has measured must leave a gap in the other's
        column, not shift its steps out of line."""
        pairs = pd.concat(
            [
                self._pairs().assign(trait="sycophantic"),
                self._pairs().head(1).assign(trait="evil"),
            ],
            ignore_index=True,
        )
        fig = figures.phase_contrast(
            pairs,
            traits=["sycophantic", "evil"],
            trait_labels={"sycophantic": "Sycophancy", "evil": "Evil"},
        )
        assert len(fig.axes) == 2
        assert [ax.get_title() for ax in fig.axes] == ["Sycophancy", "Evil"]
        # Both columns keep a slot for every step, each naming its own ticks.
        assert all(list(ax.get_xticks()) == [0, 1] for ax in fig.axes)
        assert [t.get_text() for t in fig.axes[1].get_xticklabels()] == [
            "A: 1-2",
            "B: 2-3",
        ]
        # Evil measured one of the two steps, so its column draws fewer bars.
        assert len(fig.axes[1].patches) == 1 * 2 * 2
        assert len(fig.axes[0].patches) == 2 * 2 * 2


def _overlay_panels(rows=("Sycophancy", "Evil"), cols=("Trunk A", "Trunk B")):
    """A drift grid: one series per panel, over two checkpoints."""
    return {
        (row, col): {"a": [1.0, 2.0], "b": [3.0, 4.0]} for row in rows for col in cols
    }


class TestOverlayGrid:
    def test_one_panel_per_row_and_column(self) -> None:
        fig = figures.overlay_grid(_overlay_panels())
        assert len(fig.axes) == 4

    def test_the_row_names_itself_and_the_column_heads_itself(self) -> None:
        fig = figures.overlay_grid(_overlay_panels())
        assert [ax.get_ylabel() for ax in fig.axes] == ["Sycophancy", "", "Evil", ""]
        assert [ax.get_title() for ax in fig.axes] == ["Trunk A", "Trunk B", "", ""]

    def test_row_and_column_order_is_the_designs_not_the_frames(self) -> None:
        """A grid whose rows are traits must not reorder them because a
        collection happened to load one trait's runs first."""
        fig = figures.overlay_grid(
            _overlay_panels(), rows=["Evil", "Sycophancy"], cols=["Trunk B", "Trunk A"]
        )
        assert fig.axes[0].get_ylabel() == "Evil"
        assert fig.axes[0].get_title() == "Trunk B"

    def test_a_panel_with_no_runs_is_marked_not_dropped(self) -> None:
        panels = _overlay_panels()
        del panels[("Evil", "Trunk B")]
        fig = figures.overlay_grid(
            panels, rows=["Sycophancy", "Evil"], cols=["Trunk A", "Trunk B"]
        )
        assert not fig.axes[3].lines
        assert any("not run" in t.get_text() for t in fig.axes[3].texts)

    def test_one_line_per_series_plus_the_reference(self) -> None:
        fig = figures.overlay_grid(
            _overlay_panels(), reference=100.0, reference_label="ref"
        )
        assert len(fig.axes[0].lines) == 3

    def test_sharing_y_keeps_the_panels_comparable(self) -> None:
        """Unshared axes would rescale away exactly the difference the grid is
        drawn to show -- one trunk drifting further than another."""
        panels = _overlay_panels()
        panels[("Evil", "Trunk A")] = {"a": [100.0, 900.0]}
        shared = figures.overlay_grid(panels, sharey=True)
        free = figures.overlay_grid(panels, sharey=False)
        assert len({ax.get_ylim() for ax in shared.axes}) == 1
        assert len({ax.get_ylim() for ax in free.axes}) > 1

    def test_dataset_marks_replace_the_categorical_hues(self) -> None:
        """Eight probes would take all eight categorical slots and leave the
        reader an arbitrary dataset-to-colour map to memorise."""
        marks = {
            "Evil (II)": style.dataset_mark("evil/misaligned_2"),
            "Code (Normal)": style.dataset_mark("insecure_code/normal"),
        }
        fig = figures.overlay_grid(
            {("row", "col"): {"Evil (II)": [1.0, 2.0], "Code (Normal)": [3.0, 4.0]}},
            marks=marks,
        )
        (ax,) = fig.axes
        drawn = {line.get_label(): line for line in ax.lines}
        assert drawn["Evil (II)"].get_marker() == style.DATASET_MARKERS["evil"]
        assert drawn["Evil (II)"].get_color() == style.VERSION_LINE["misaligned_2"]
        assert drawn["Code (Normal)"].get_markerfacecolor() == style.SURFACE

    def test_a_replicate_shares_its_series_colour_and_is_dashed(self) -> None:
        """The reseed rides the colour of the run it replicates, so it costs no
        extra hue and cannot be read as a further condition."""
        fig = figures.overlay_grid(
            {("row", "col"): {"a": [1.0, 2.0]}}, {("row", "col"): {"a": [[1.1, 2.1]]}}
        )
        (ax,) = fig.axes
        primary, replicate = ax.lines[0], ax.lines[1]
        assert primary.get_color() == replicate.get_color()
        assert primary.get_linestyle() == "-"
        assert replicate.get_linestyle() != "-"

    def test_a_band_spans_one_spread_each_side_of_the_line(self) -> None:
        fig = figures.overlay_grid(
            {("row", "col"): {"a": [2.0, 4.0]}},
            bands={("row", "col"): {"a": [0.5, 1.0]}},
        )
        (ax,) = fig.axes
        assert len(ax.lines) == 1
        vertices = ax.collections[0].get_paths()[0].vertices
        assert vertices[:, 1].min() == pytest.approx(1.5)
        assert vertices[:, 1].max() == pytest.approx(5.0)
        (legend,) = fig.legends
        assert any("1 SD" in text.get_text() for text in legend.get_texts())

    def test_every_replicate_seed_draws_its_own_line(self) -> None:
        """The probe-free reseed tier is four extra seeds; collapsing them to
        one line would plot a trajectory no seed actually took."""
        fig = figures.overlay_grid(
            {("row", "col"): {"a": [1.0, 2.0]}},
            {("row", "col"): {"a": [[1.1, 2.1], [1.2, 2.6], [0.9, 1.7]]}},
        )
        (ax,) = fig.axes
        dashed = [line for line in ax.lines if line.get_linestyle() != "-"]
        assert len(dashed) == 3
        assert [list(line.get_ydata()) for line in dashed] == [
            [1.1, 2.1],
            [1.2, 2.6],
            [0.9, 1.7],
        ]

    def test_a_replicate_keeps_its_colour_when_the_primary_dropped_a_series(
        self,
    ) -> None:
        """Colour comes from the primary's ordering, so a probe dropped from
        one half of the pair cannot shift the other half's hues."""
        fig = figures.overlay_grid(
            {("row", "col"): {"a": [1.0, 2.0], "b": [3.0, 4.0]}},
            {("row", "col"): {"b": [[3.1, 4.1]]}},
        )
        (ax,) = fig.axes
        by_label = {line.get_label(): line for line in ax.lines}
        (dashed,) = [line for line in ax.lines if line.get_linestyle() != "-"]
        assert dashed.get_color() == by_label["b"].get_color()

    def test_the_dashed_convention_is_named_in_the_legend(self) -> None:
        fig = figures.overlay_grid(
            {("row", "col"): {"a": [1.0]}}, {("row", "col"): {"a": [[1.1]]}}
        )
        (legend,) = fig.legends
        assert any("dashed" in t.get_text() for t in legend.get_texts())

    def test_the_key_covers_a_series_missing_from_the_first_panel(self) -> None:
        """A probe dropped from one trunk for a near-zero baseline still has to
        appear in the key of the panels that do draw it."""
        panels = _overlay_panels(rows=("Sycophancy",))
        panels[("Sycophancy", "Trunk A")] = {"a": [1.0, 2.0]}
        fig = figures.overlay_grid(panels)
        (legend,) = fig.legends
        assert [t.get_text() for t in legend.get_texts()] == ["a", "b"]


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

    def test_an_unmeasured_series_is_kept_out_of_the_figures(self) -> None:
        r"""$\Delta P$ is measured on one trunk, so on a run of the decay
        family alone its column is entirely NaN -- and a legend key for a line
        nobody drew reads as a line that came out flat."""
        fits = _fits(trunks=("a",)).assign(corr_p=np.nan)
        assert make_plots._present_series(fits) == ["p0", "pt"]

    def test_a_series_measured_anywhere_is_drawn(self) -> None:
        fits = _fits(trunks=("a", "b")).assign(
            corr_p=lambda f: np.where(f["trunk"] == "a", 0.9, np.nan)
        )
        assert make_plots._present_series(fits) == ["p0", "pt", "p"]

    def test_the_series_keep_the_designs_order(self) -> None:
        """The ladder of what is allowed to be current at $M_t$, which is how
        the figures are read left to right."""
        assert make_plots._present_series(
            _fits(trunks=("a",)).assign(corr_pv0=0.9, corr_p=0.9)
        ) == list(decay.SERIES)

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

    def test_replicate_seeds_are_kept_apart_rather_than_averaged(self) -> None:
        """``pivot_table`` aggregates whatever shares an index entry, so a
        seed left out of the index silently becomes the mean of the seeds --
        a line no run produced, and indistinguishable from a real one."""
        frame = pd.DataFrame(
            [
                {"probe": "p", "seed": seed, "t": t, "ratio": float(value)}
                for seed, values in [(2, [100.0, 10.0]), (3, [100.0, 90.0])]
                for t, value in enumerate(values)
            ]
        )
        assert make_plots._replicates_by(frame, key="probe", value="ratio") == {
            "p": [[100.0, 10.0], [100.0, 90.0]]
        }

    def test_a_replicate_missing_a_checkpoint_drops_only_that_seed(self) -> None:
        """A ragged seed must not take its siblings' complete series with it."""
        frame = pd.DataFrame(
            [
                {"probe": "p", "seed": 2, "t": 0, "ratio": 100.0},
                {"probe": "p", "seed": 2, "t": 1, "ratio": 40.0},
                {"probe": "p", "seed": 3, "t": 0, "ratio": 100.0},
            ]
        )
        assert make_plots._replicates_by(frame, key="probe", value="ratio") == {
            "p": [[100.0, 40.0]]
        }

    def test_a_frame_without_seeds_has_no_replicates(self) -> None:
        frame = pd.DataFrame({"probe": ["p"], "t": [0], "ratio": [1.0]})
        assert make_plots._replicates_by(frame, key="probe", value="ratio") == {}

    def test_latent_summary_is_the_five_seed_mean_and_sample_std(self) -> None:
        frame = pd.DataFrame(
            [
                {"trunk": "a", "seed": seed, "t": t, "rho": value}
                for seed in range(5)
                for t, value in enumerate((float(seed), float(seed * 10)))
            ]
        )
        means, stds = make_plots._trunk_mean_std(frame, "rho", ["a"])
        label = labels.display_trunk_name("a")
        assert means == {label: [2.0, 20.0]}
        assert stds[label] == pytest.approx([np.sqrt(2.5), np.sqrt(250.0)])

    def test_single_seed_trunks_have_a_mean_line_but_no_band(self) -> None:
        frame = pd.DataFrame(
            {"trunk": ["b", "b"], "seed": [0, 0], "t": [0, 1], "rho": [1.0, 0.8]}
        )
        means, stds = make_plots._trunk_mean_std(frame, "rho", ["b"])
        assert means == {labels.display_trunk_name("b"): [1.0, 0.8]}
        assert stds == {}


# --- demo.py -----------------------------------------------------------------


class TestDemo:
    def test_build_and_save_writes_all_figures(self, tmp_path) -> None:
        saved = build_and_save(tmp_path, n_seeds=2)
        assert len(saved) == 14  # 7 figures, 2 files (png+pdf) each
        for path in saved:
            assert path.exists()
            assert path.stat().st_size > 0
