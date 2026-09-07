"""Tests for :mod:`method.visualization.training_curves`.

Every case is built from a synthetic points frame shaped like the one
:mod:`method.gradients` writes, so nothing here needs the 63 MB CSV, the
console logs it was recovered from, or any run on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from method.visualization import training_curves as tc


def points(
    *,
    run_name: str,
    condition: str,
    weights_id: str,
    step: int = tc.FINAL_STEP,
    n_steps: int = 6,
    loss: float = 1.0,
    grad_norm: float = 1.0,
    noise: float = 0.0,
    spike: float = 0.0,
    spike_at: int = 3,
) -> pd.DataFrame:
    """One training run's trace, flat at ``loss``/``grad_norm`` plus a wobble.

    ``noise`` alternates in sign step by step, which is how a test fixes what
    the entry value and the run median are. ``spike`` adds one hard batch at
    ``spike_at`` and nowhere else, which is what a rolling median is there to
    absorb -- so a test can tell smoothing from averaging.
    """
    optim_step = np.arange(1, n_steps + 1)
    wobble = noise * np.where(optim_step % 2 == 1, 1.0, -1.0)
    wobble = wobble + spike * (optim_step == spike_at)
    return pd.DataFrame(
        {
            "run_name": run_name,
            "condition": condition,
            "step": step,
            "weights_id": weights_id,
            "optim_step": optim_step,
            "loss": loss + wobble,
            "grad_norm": grad_norm + wobble,
        }
    )


#: Every trajectory the frames below name, mapped to the one dataset they end
#: on -- the join :func:`~method.visualization.training_curves.final_step_curves`
#: needs, which the recovered curves cannot make for themselves.
DATASETS = {name: "hallucination/misaligned_1" for name in ("r_same", "r_diff", "r_n2")}


class TestLoadPoints:
    def test_a_missing_file_is_an_empty_frame_not_an_error(self, tmp_path) -> None:
        """The CSV is gitignored, so a checkout with every trajectory and no
        training curves is a normal state; the builder skips on ``empty``."""
        frame = tc.load_points(tmp_path / "absent.csv")
        assert frame.empty
        assert {"condition", "step", "weights_id", "optim_step", *tc.QUANTITIES} <= set(
            frame.columns
        )

    def test_an_empty_frame_survives_the_whole_pipeline(self, tmp_path) -> None:
        frame = tc.load_points(tmp_path / "absent.csv")
        curves = tc.final_step_curves(frame, DATASETS)
        assert curves.empty
        assert tc.curve_bands(curves).empty
        assert tc.entry_summary(curves).empty

    def test_a_present_file_is_read(self, tmp_path: Path) -> None:
        path = tmp_path / "points.csv"
        points(run_name="r_same", condition="same", weights_id="w1").to_csv(
            path, index=False
        )
        assert len(tc.load_points(path)) == 6


class TestFinalStepCurves:
    def test_only_the_final_step_is_kept(self) -> None:
        """Earlier steps trained on different data in different arms, so their
        curves compare nothing."""
        frame = pd.concat(
            [
                points(run_name="r_same", condition="same", weights_id="w1", step=1),
                points(run_name="r_same", condition="same", weights_id="w2"),
            ]
        )
        curves = tc.final_step_curves(frame, DATASETS)
        assert set(curves["weights_id"]) == {"w2"}

    def test_arms_outside_the_matched_depth_are_dropped(self) -> None:
        frame = pd.concat(
            [
                points(run_name="r_same", condition="baseline", weights_id="w1"),
                points(run_name="r_same", condition="same", weights_id="w2"),
            ]
        )
        curves = tc.final_step_curves(frame, DATASETS)
        assert set(curves["condition"]) == {"same"}

    def test_one_training_event_shared_by_two_trajectories_is_counted_once(
        self,
    ) -> None:
        """Adapters are content addressed: the same run appears once per
        measured trait, carrying the same curve both times. Counting it twice
        would present one measurement as two."""
        shared = pd.concat(
            [
                points(run_name="r_same", condition="same", weights_id="w1"),
                points(run_name="r_diff", condition="same", weights_id="w1"),
            ]
        )
        curves = tc.final_step_curves(shared, DATASETS)
        assert len(curves) == 6
        assert curves["weights_id"].nunique() == 1

    def test_a_trajectory_with_no_config_is_dropped_not_guessed_at(self) -> None:
        frame = pd.concat(
            [
                points(run_name="r_same", condition="same", weights_id="w1"),
                points(run_name="unknown", condition="same", weights_id="w2"),
            ]
        )
        curves = tc.final_step_curves(frame, DATASETS)
        assert set(curves["weights_id"]) == {"w1"}
        assert curves["dataset"].notna().all()

    def test_rows_are_labelled_with_the_dataset_the_step_trained_on(self) -> None:
        frame = points(run_name="r_same", condition="same", weights_id="w1")
        curves = tc.final_step_curves(frame, DATASETS)
        assert set(curves["dataset"]) == {"hallucination/misaligned_1"}


class TestCurveBands:
    def test_one_hard_batch_moves_neither_the_line_nor_the_band(self) -> None:
        """Two runs at the same level, each with one outlying batch in a
        different place. Smoothing is applied per run and before averaging, so
        the spikes are absorbed rather than smeared across the band."""
        frame = pd.concat(
            [
                points(
                    run_name="r_same",
                    condition="same",
                    weights_id="w1",
                    spike=5.0,
                    spike_at=3,
                ),
                points(
                    run_name="r_diff",
                    condition="same",
                    weights_id="w2",
                    spike=5.0,
                    spike_at=4,
                ),
            ]
        )
        bands = tc.curve_bands(tc.final_step_curves(frame, DATASETS), window=3)
        assert bands["loss_mean"].to_numpy() == pytest.approx(1.0)
        assert bands["loss_sd"].to_numpy() == pytest.approx(0.0)

    def test_without_smoothing_that_batch_would_reach_the_band(self) -> None:
        """The contrast the test above is only meaningful against."""
        frame = points(
            run_name="r_same", condition="same", weights_id="w1", spike=5.0, spike_at=3
        )
        bands = tc.curve_bands(tc.final_step_curves(frame, DATASETS), window=1)
        assert bands["loss_mean"].max() == pytest.approx(6.0)

    def test_runs_at_different_levels_keep_a_band(self) -> None:
        frame = pd.concat(
            [
                points(run_name="r_same", condition="same", weights_id="w1", loss=1.0),
                points(run_name="r_diff", condition="same", weights_id="w2", loss=2.0),
            ]
        )
        bands = tc.curve_bands(tc.final_step_curves(frame, DATASETS))
        assert bands["loss_mean"].to_numpy() == pytest.approx(1.5)
        assert (bands["loss_sd"] > 0).all()

    def test_each_arm_gets_its_own_curve(self) -> None:
        frame = pd.concat(
            [
                points(run_name="r_same", condition="same", weights_id="w1"),
                points(run_name="r_n2", condition="normal2", weights_id="w2"),
            ]
        )
        bands = tc.curve_bands(tc.final_step_curves(frame, DATASETS))
        assert set(bands["condition"]) == {"same", "normal2"}
        assert (bands["runs"] == 1).all()


class TestEntrySummary:
    def test_init_is_the_step_one_value(self) -> None:
        """Optimiser step 1 is logged while the learning rate is still 0, so it
        reads the state the run was handed and not what the run then did."""
        frame = points(
            run_name="r_same",
            condition="same",
            weights_id="w1",
            n_steps=5,
            loss=1.0,
            noise=0.25,
        )
        summary = tc.entry_summary(tc.final_step_curves(frame, DATASETS))
        assert summary["loss_init"].iloc[0] == pytest.approx(1.25)
        assert summary["loss_median"].iloc[0] == pytest.approx(1.25)

    def test_runs_counts_training_events(self) -> None:
        frame = pd.concat(
            [
                points(run_name="r_same", condition="same", weights_id="w1"),
                # The same event reached by a second trajectory.
                points(run_name="r_diff", condition="same", weights_id="w1"),
                points(run_name="r_n2", condition="same", weights_id="w2"),
            ]
        )
        summary = tc.entry_summary(tc.final_step_curves(frame, DATASETS))
        assert summary["runs"].tolist() == [2]

    def test_one_row_per_dataset_and_arm(self) -> None:
        frame = pd.concat(
            [
                points(run_name="r_same", condition="same", weights_id="w1"),
                points(run_name="r_diff", condition="diff", weights_id="w2"),
                points(run_name="r_n2", condition="normal2", weights_id="w3"),
            ]
        )
        summary = tc.entry_summary(tc.final_step_curves(frame, DATASETS))
        assert len(summary) == 3
        assert set(summary["condition"]) == {"same", "diff", "normal2"}


class TestEntryRatios:
    def _summary(self, same: float, diff: float, normal2: float) -> pd.DataFrame:
        frame = pd.concat(
            [
                points(
                    run_name="r_same",
                    condition="same",
                    weights_id="w1",
                    loss=same,
                    grad_norm=same,
                ),
                points(
                    run_name="r_diff",
                    condition="diff",
                    weights_id="w2",
                    loss=diff,
                    grad_norm=diff,
                ),
                points(
                    run_name="r_n2",
                    condition="normal2",
                    weights_id="w3",
                    loss=normal2,
                    grad_norm=normal2,
                ),
            ]
        )
        return tc.entry_summary(tc.final_step_curves(frame, DATASETS))

    def test_a_repeat_arm_entering_lower_ratios_below_one(self) -> None:
        ratios = tc.entry_ratios(self._summary(same=1.0, diff=2.0, normal2=2.0))
        assert ratios["grad_norm_init"].iloc[0] == pytest.approx(0.5)
        assert ratios["loss_init"].iloc[0] == pytest.approx(0.5)

    def test_arms_that_do_not_differ_ratio_to_one(self) -> None:
        ratios = tc.entry_ratios(self._summary(same=2.0, diff=2.0, normal2=2.0))
        assert ratios["grad_norm_init"].iloc[0] == pytest.approx(1.0)

    def test_the_two_first_exposure_arms_are_pooled_evenly(self) -> None:
        """Their median, so one arm missing a trace cannot tilt the comparison
        toward the other."""
        ratios = tc.entry_ratios(self._summary(same=1.5, diff=1.0, normal2=2.0))
        assert ratios["loss_init"].iloc[0] == pytest.approx(1.0)

    def test_a_dataset_with_no_repeat_arm_is_omitted(self) -> None:
        frame = points(run_name="r_diff", condition="diff", weights_id="w2")
        summary = tc.entry_summary(tc.final_step_curves(frame, DATASETS))
        assert tc.entry_ratios(summary).empty
