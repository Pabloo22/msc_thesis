r"""Tests for the out-of-sample analysis, :mod:`method.visualization.forecast`.

Built on the same schema-faithful ``trajectory.json`` fixtures as
``test_decay.py``, with one addition its fixtures do not need: a validation fan
covering datasets *beyond* the probe set, since holding the probes out of
$M_0$'s line is what makes everything here a prediction rather than a residual.

The data is noiseless throughout. ``Delta b`` is an exact affine function of the
projection difference, so an assertion on an error is an assertion about the
analysis and never about a draw -- and a fixture that decays by a known factor
lets a test say what the error *should* be, not merely that it grew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from method import experiments as E
from method.config import DatasetVersion, StepConfig
from method.visualization import decay, forecast
from method.visualization.collect import Collection, collect
from method.visualization.metrics import LinearFit, linear_fit

from tests.test_decay import (
    DELTA_P_0,
    SLOPE,
    TRUNKS,
    build_axis,
    build_decay,
    build_regen,
    temp_trajectories,  # noqa: F401  (autouse fixture, imported for its effect)
    write_run,
)

#: The probe set ``build_decay`` fans out over, as ``dataset/version`` ids.
PROBES = tuple(DELTA_P_0)

#: Datasets the validation fan covers *outside* the probe set, with the
#: $\Delta P_0$ each is given. Two is the minimum that fits a line, which is
#: what :func:`method.visualization.forecast.baseline_fits` needs once the
#: probes are held out.
HELD_OUT = {
    "sycophancy/normal": -2.0,
    "hallucination/misaligned_1": 3.0,
}

#: The intercept the fixture's behaviour change carries, so that a test can
#: tell a recovered slope from a recovered offset.
INTERCEPT = 0.0

#: Where the trunks start. Both share $M_0$, as the design requires.
B_0 = 50.0


def build_validation(*, se: float = 0.5) -> Collection:
    r"""The $t = 0$ fan over the probes *and* two datasets outside them.

    ``build_decay``'s branch endpoints put $\Delta b$ at exactly
    ``SLOPE * Delta P_0``, and this fan is wired to the same law, so $M_0$'s
    line fitted on the held-out pair is the same line the probes lie on. Any
    error a forecaster then reports at $t = 0$ is a real one.
    """
    datasets = [
        *(
            StepConfig(dataset=name.split("/")[0], version=DatasetVersion(version))
            for name, version in (d.split("/") for d in HELD_OUT)
        ),
        *(
            StepConfig(dataset=name.split("/")[0], version=DatasetVersion(version))
            for name, version in (d.split("/") for d in PROBES)
        ),
    ]
    delta_p_0 = {**DELTA_P_0, **HELD_OUT}
    configs = E.build_exp2_validation_configs(
        seeds=(E.EXP2_SEED,), measure_traits=("evil",), datasets=datasets
    )
    for cfg in configs:
        dataset = cfg.label_map["dataset"]
        write_run(
            cfg,
            behaviors=[B_0, B_0 + INTERCEPT + SLOPE * delta_p_0[dataset]],
            probes={d: [v, v] for d, v in DELTA_P_0.items()},
            delta_p=delta_p_0[dataset],
            se=se,
        )
    return collect(configs, group=E.EXP2_VALIDATION)


def attenuated(rows: pd.DataFrame, gains: dict[int, float]) -> pd.DataFrame:
    r"""``rows`` with each checkpoint's $\Delta b$ shrunk by ``gains[t]``.

    The fixture's own law is $\Delta b = \mathrm{SLOPE} \times \Delta P_0$ at
    every checkpoint, which is a trunk that never goes stale -- the right
    default for testing that the analysis can report *no* error, and useless
    for testing that it can report one. Scaling by a known factor per
    checkpoint gives a trunk whose staleness is exactly known, so a test can
    assert the error the analysis should find rather than merely that it found
    some.

    The relationship stays exactly linear, so the refit at $t$ still fits it
    perfectly: what the gain destroys is the frozen line's calibration, not the
    ordering. That is the whole distinction the module exists to draw.
    """
    scaled = rows.copy()
    gain = scaled["t"].map(gains)
    scaled["delta_b"] = scaled["delta_b"] * gain
    scaled["b_next"] = scaled["b_t"] + scaled["delta_b"]
    return scaled


#: A trunk losing a tenth of its response per checkpoint.
GAINS = {t: 1.0 - 0.1 * t for t in range(7)}


@pytest.fixture(name="frames")
def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(decay rows, validation fan)``, the pair every entry point takes."""
    validation = build_validation()
    rows = decay.decay_frame(
        build_decay(),
        validation,
        [build_axis(), build_regen()],
    )
    return rows, decay.validation_frame(validation)


# --- baseline_fits ----------------------------------------------------------


class TestBaselineFits:
    def test_fits_on_the_datasets_that_are_not_probes(self, frames) -> None:
        """The probes are the test set, so M_0's line must not have seen them.

        Fitted on the held-out pair alone, the line is exactly the law the
        fixture was generated with; a fit that had leaked the probes in would
        also be that line, so the assertion that separates them is on how many
        points went in.
        """
        _, fan = frames
        held_out = fan[~fan["dataset"].isin(PROBES)]
        assert set(held_out["dataset"]) == set(HELD_OUT)

        fits = forecast.baseline_fits(fan, PROBES)
        change = fits[forecast.CHANGE]["evil"]
        assert change.slope == pytest.approx(SLOPE)
        assert change.intercept == pytest.approx(INTERCEPT)
        # The level target is the same line lifted by the level M_0 starts at.
        level = fits[forecast.LEVEL]["evil"]
        assert level.slope == pytest.approx(SLOPE)
        assert level.intercept == pytest.approx(INTERCEPT + B_0)

    def test_a_fan_of_probes_alone_fits_nothing(self, frames) -> None:
        """Holding the probes out of a probes-only fan leaves nothing to fit.

        The honest answer is no line rather than one fitted on the points it is
        about to be scored against.
        """
        _, fan = frames
        probes_only = fan[fan["dataset"].isin(PROBES)]
        fits = forecast.baseline_fits(probes_only, PROBES)
        assert not any(fits.values())

    def test_an_empty_fan_fits_nothing(self) -> None:
        fits = forecast.baseline_fits(pd.DataFrame(), PROBES)
        assert set(fits) == set(forecast.TARGETS)
        assert not any(fits.values())


# --- the reason the module exists -------------------------------------------


class TestCorrelationIsBlindToStaleness:
    @pytest.fixture(name="stale")
    def _stale(self, frames) -> pd.DataFrame:
        """Scores over a trunk losing a tenth of its response per checkpoint."""
        rows, fan = frames
        return forecast.prediction_frame(
            attenuated(rows, GAINS), fan, series=["p0"], models=["step0"]
        )

    def test_a_frozen_line_leaves_the_correlation_where_it_was(
        self, stale
    ) -> None:
        r"""Pearson $r$ is invariant under a fixed affine map.

        This is what the whole module rests on: scoring $M_0$'s *predictions*
        by correlation would reproduce ``decay.correlation_table`` cell for
        cell, so an out-of-sample table has to report an error instead. If this
        ever fails, one of the two tables has stopped meaning what it says.

        Asserted on a trunk that *is* going stale, so it is not passing for
        want of anything to detect.
        """
        for _, panel in stale.groupby(list(forecast.CHECKPOINT)):
            on_projection = np.corrcoef(panel["delta_p"], panel["b_next"])[0, 1]
            on_prediction = np.corrcoef(panel["predicted_b_next"], panel["b_next"])[0, 1]
            assert on_prediction == pytest.approx(on_projection)

    def test_the_correlation_does_not_move_at_all(self, stale) -> None:
        r"""Every checkpoint of the decaying trunk has the same correlation.

        A rescaling of $\Delta b$ leaves $r$ where it was, so a correlation
        table over this fixture would show a flat, perfect row while the
        predictions built on it drift further off every step.
        """
        correlations = {
            round(float(np.corrcoef(panel["delta_p"], panel["b_next"])[0, 1]), 9)
            for _, panel in stale.groupby(list(forecast.CHECKPOINT))
        }
        assert correlations == {1.0}

    def test_the_error_grows_as_the_correlation_holds(self, stale) -> None:
        """...which is why the tables here are errors and not correlations.

        The frozen line goes on promising the response $M_0$ would have made,
        so on a trunk giving back a tenth less each step it over-predicts by a
        tenth more each step -- exactly, since the fixture is noiseless.
        """
        scores = forecast.score_frame(stale).sort_values("t")
        for _, trunk in scores.groupby("trunk"):
            assert trunk["rmse"].is_monotonic_increasing
            assert trunk["bias"].is_monotonic_increasing
            # (1 - g_t) of the response, and the response is SLOPE * Delta P_0.
            expected = [
                (1.0 - GAINS[t]) * SLOPE * float(np.mean(list(DELTA_P_0.values())))
                for t in trunk["t"]
            ]
            assert trunk["bias"].to_numpy() == pytest.approx(expected)

    def test_a_refit_still_fits_the_decayed_trunk_perfectly(self, frames) -> None:
        """The gain destroys the calibration, not the ordering.

        Which is the distinction the module exists to draw: refitting recovers
        everything the frozen line lost here, so the gap between the two rows
        of the emitted table is the whole cost of not refitting.
        """
        rows, fan = frames
        scores = forecast.score_frame(
            forecast.prediction_frame(
                attenuated(rows, GAINS), fan, series=["p0"], models=["oracle"]
            )
        )
        assert scores["rmse"].max() == pytest.approx(0.0, abs=1e-9)


# --- prediction_frame -------------------------------------------------------


class TestPredictionFrame:
    def test_a_trunk_that_never_drifts_is_predicted_exactly(self, frames) -> None:
        r"""Trunk C's $\Delta P_0$ still governs $\Delta b$ at every checkpoint.

        The fixture wires every branch endpoint to ``b_t + SLOPE * Delta P_0``
        and trunk C's probes never move, so a line fitted at $M_0$ stays exactly
        right. An analysis that cannot report "no error" on data with no error
        cannot be trusted to report an error.
        """
        rows, fan = frames
        scores = forecast.score_frame(
            forecast.prediction_frame(rows, fan, series=["p0"], models=["step0"])
        )
        control = scores[scores["trunk"] == "c"]
        assert not control.empty
        assert control["rmse"].max() == pytest.approx(0.0, abs=1e-9)
        assert control["bias"].abs().max() == pytest.approx(0.0, abs=1e-9)

    def test_predicted_b_next_is_the_prediction_on_the_judge_scale(
        self, frames
    ) -> None:
        r"""$\hat{b}_{t+1} = b_t + \widehat{\Delta b}$, which is what the grid draws."""
        rows, fan = frames
        predictions = forecast.prediction_frame(rows, fan, models=["step0"])
        assert predictions["predicted_b_next"].to_numpy() == pytest.approx(
            (predictions["b_t"] + predictions["predicted_delta_b"]).to_numpy()
        )

    def test_error_is_prediction_minus_truth(self, frames) -> None:
        """Signed the way ``bias`` is read: positive means over-predicting."""
        rows, fan = frames
        predictions = forecast.prediction_frame(rows, fan, models=["step0"])
        assert predictions["error"].to_numpy() == pytest.approx(
            (predictions["predicted_b_next"] - predictions["b_next"]).to_numpy()
        )

    def test_an_unmeasured_series_contributes_no_rows(self, frames) -> None:
        r"""$\Delta P_t$ is re-measured on trunk A alone in these fixtures.

        The trunk that was not re-measured must drop out of that series whole
        rather than appear with a gap, for the reason ``decay._series_fit``
        gives: an error over the probes that happened to be measured is an
        error over a different probe set than the one printed beside it.
        """
        rows, fan = frames
        full = forecast.prediction_frame(rows, fan, series=["full_t"])
        assert set(full["trunk"]) == {"a"}
        stale = forecast.prediction_frame(rows, fan, series=["p0"])
        assert set(stale["trunk"]) == {"a", "c"}

    def test_without_a_held_out_fan_only_the_refit_is_scored(self, frames) -> None:
        """A sweep with no fan outside its probes can still report the ceiling."""
        rows, fan = frames
        probes_only = fan[fan["dataset"].isin(PROBES)]
        predictions = forecast.prediction_frame(rows, probes_only, series=["p0"])
        frozen = predictions[predictions["model"] == "step0"]
        refit = predictions[predictions["model"] == "oracle"]
        assert frozen["predicted_b_next"].isna().all()
        assert refit["predicted_b_next"].notna().all()

    def test_unknown_series_is_refused(self, frames) -> None:
        rows, fan = frames
        with pytest.raises(ValueError, match="unknown series"):
            forecast.prediction_frame(rows, fan, series=["delta_p_0"])

    def test_unknown_model_is_refused(self, frames) -> None:
        rows, fan = frames
        with pytest.raises(ValueError, match="unknown forecaster"):
            forecast.prediction_frame(rows, fan, models=["step_zero"])

    def test_no_rows_gives_an_empty_frame_of_the_right_shape(self) -> None:
        empty = forecast.prediction_frame(pd.DataFrame(), pd.DataFrame())
        assert empty.empty
        assert "predicted_b_next" in empty.columns


# --- score_frame ------------------------------------------------------------


class TestScoreFrame:
    def test_a_refit_has_no_bias_by_construction(self, frames) -> None:
        """Least squares with a free intercept leaves residuals summing to zero.

        Which is why the emitted bias table carries the frozen forecasters
        alone: a column of zeros would be arithmetic dressed as a result.
        """
        rows, fan = frames
        scores = forecast.score_frame(forecast.prediction_frame(rows, fan))
        refit = scores[scores["model"] == "oracle"]
        assert not refit.empty
        assert refit["bias"].abs().max() == pytest.approx(0.0, abs=1e-9)
        assert "oracle" not in forecast.BIASED_MODELS

    def test_a_stale_line_over_predicts_a_drifting_trunk(self, frames) -> None:
        r"""Trunk A's probes drift while $\Delta P_0$ stays where $M_0$ read it.

        With ``build_decay``'s law tying $\Delta b$ to the *frozen* projection,
        the fixture has no decay to find, so this asserts the shape of the
        machinery rather than a finding: the bias of a frozen line is a number
        the analysis reports, and the refit's is not.
        """
        rows, fan = frames
        scores = forecast.score_frame(forecast.prediction_frame(rows, fan))
        frozen = scores[(scores["model"] == "step0") & (scores["series"] == "p0")]
        assert frozen["bias"].notna().all()

    def test_an_unfitted_forecaster_scores_nan_not_a_partial_number(
        self, frames
    ) -> None:
        """A cloud with a missing prediction scores NaN over the whole cloud."""
        rows, fan = frames
        predictions = forecast.prediction_frame(
            rows, fan, series=["p0"], models=["step0"]
        )
        predictions.loc[predictions.index[0], "error"] = np.nan
        scores = forecast.score_frame(predictions)
        assert scores["rmse"].isna().sum() == 1

    def test_metrics_are_in_judge_points(self, frames) -> None:
        """RMSE and MAE agree exactly where every probe misses by the same amount."""
        rows, fan = frames
        predictions = forecast.prediction_frame(
            rows, fan, series=["p0"], models=["step0"]
        )
        predictions["error"] = 4.0
        scores = forecast.score_frame(predictions)
        assert scores["rmse"].to_numpy() == pytest.approx(4.0)
        assert scores["mae"].to_numpy() == pytest.approx(4.0)
        assert scores["bias"].to_numpy() == pytest.approx(4.0)

    def test_empty_predictions_give_an_empty_frame(self) -> None:
        assert forecast.score_frame(pd.DataFrame()).empty


# --- the gain correction ----------------------------------------------------


class TestTargets:
    r"""Fitting $M_0$'s line on the change against on the level."""

    def test_a_refit_is_target_invariant(self, frames) -> None:
        r"""Refitting on $b_{t+1}$ and on $\Delta b$ give the same level.

        Within a checkpoint $b_t$ is one constant, so the two fits differ by
        exactly that in their intercept and by nothing in their slope. That is
        why the refit appears once in a target table rather than twice.
        """
        rows, fan = frames
        for _, panel in rows.groupby(list(forecast.CHECKPOINT)):
            on_change = linear_fit(panel["delta_p_0"], panel["delta_b"])
            on_level = linear_fit(panel["delta_p_0"], panel["b_next"])
            assert on_level.slope == pytest.approx(on_change.slope)
            assert on_level.intercept == pytest.approx(
                on_change.intercept + float(panel["b_t"].iloc[0])
            )

    def test_the_level_fit_never_consults_where_the_model_is(
        self, frames
    ) -> None:
        r"""Its prediction for a probe is the same number at every checkpoint.

        That is the whole claim it makes -- "this dataset lands you here" --
        and it is what makes it the honest target for a projection difference
        measured at $M_0$, which has no way of knowing where the model has got
        to.
        """
        rows, fan = frames
        predictions = forecast.prediction_frame(
            attenuated(rows, GAINS), fan, series=["p0"], models=["step0_level"]
        )
        for probe, group in predictions.groupby("probe"):
            assert group["predicted_b_next"].nunique() == 1

    def test_the_change_fit_moves_with_the_checkpoint(self, frames) -> None:
        """Where the level fit is fixed, the change fit rides on $b_t$."""
        rows, fan = frames
        predictions = forecast.prediction_frame(
            attenuated(rows, GAINS), fan, series=["p0"], models=["step0"]
        )
        drifting = predictions[predictions["trunk"] == "a"]
        for probe, group in drifting.groupby("probe"):
            assert group["predicted_b_next"].nunique() > 1
            # ...by exactly b_t, since the change it predicts is fixed.
            assert group["predicted_delta_b"].nunique() == 1

    def test_a_trunk_that_never_moves_cannot_tell_them_apart(
        self, frames
    ) -> None:
        """On a flat trunk the two targets differ by a constant the intercept
        absorbs, so they are the same forecaster and must score the same."""
        rows, fan = frames
        flat = rows[rows["trunk"] == "c"]
        scores = forecast.score_frame(
            forecast.prediction_frame(
                flat, fan, series=["p0"], models=["step0", "step0_level"]
            )
        )
        wide = scores.pivot_table(index="t", columns="model", values="rmse")
        assert wide["step0"].to_numpy() == pytest.approx(
            wide["step0_level"].to_numpy()
        )


class TestGainCorrection:
    def test_ideal_gain_is_the_factor_that_would_have_been_right(self) -> None:
        r"""$g$ solves $\alpha_0 + \beta_0 g x = y$ exactly when one exists."""
        fit = LinearFit(slope=2.0, intercept=1.0, r2=1.0)
        x = np.array([1.0, 2.0, 3.0])
        assert forecast._ideal_gain(fit, x, 1.0 + 2.0 * 1.5 * x) == pytest.approx(1.5)

    def test_ideal_gain_of_a_perfect_line_is_one(self) -> None:
        fit = linear_fit([1.0, 2.0, 3.0], [4.0, 6.0, 8.0])
        assert forecast._ideal_gain(
            fit, [1.0, 2.0, 3.0], [4.0, 6.0, 8.0]
        ) == pytest.approx(1.0)

    def test_a_degenerate_projection_has_no_gain(self) -> None:
        fit = LinearFit(slope=0.0, intercept=1.0, r2=0.0)
        assert np.isnan(forecast._ideal_gain(fit, [1.0, 2.0], [3.0, 4.0]))

    def test_a_gain_is_never_fitted_on_its_own_trunk(self, frames) -> None:
        """Leave-one-trunk-out, so a correction is scored on an unseen trajectory.

        Consecutive checkpoints of one trunk are the same trajectory a step
        further along, so a fold that held out a checkpoint while keeping its
        neighbours would let the correction be read off the trunk it is scored
        on.
        """
        rows, fan = frames
        baselines = forecast.baseline_fits(fan, PROBES)
        gains = forecast._gain_forecast(rows, "delta_p_0", baselines, ("b_t",))
        assert gains
        # Trunk C is flat, so a gain fitted on it alone predicts trunk A's
        # constant gain -- a number, and one that could not have come from A.
        assert {trunk for _, trunk, _ in gains} == {"a", "c"}

    def test_one_trunk_cannot_leave_one_out(self, frames) -> None:
        """A single-trunk sweep has no other trajectory to fit the gain on."""
        rows, fan = frames
        one = rows[rows["trunk"] == "a"]
        baselines = forecast.baseline_fits(fan, PROBES)
        assert forecast._gain_forecast(one, "delta_p_0", baselines, ("b_t",)) == {}

    def test_a_correction_that_cannot_be_fitted_predicts_nothing(
        self, frames
    ) -> None:
        rows, fan = frames
        one = rows[rows["trunk"] == "a"]
        predictions = forecast.prediction_frame(
            one, fan, series=["p0"], models=["step0_b"]
        )
        assert predictions["predicted_b_next"].isna().all()

    def test_gain_frame_carries_the_state_it_is_regressed_on(self, frames) -> None:
        rows, fan = frames
        baselines = forecast.baseline_fits(fan, PROBES)
        gains = forecast.gain_frame(rows, "delta_p_0", baselines)
        assert not gains.empty
        assert set(forecast.GAIN_FEATURES) <= set(gains.columns)
        assert gains["gain"].notna().all()


# --- score_table ------------------------------------------------------------


class TestScoreTable:
    @pytest.fixture(name="scores")
    def _scores(self, frames) -> pd.DataFrame:
        rows, fan = frames
        return forecast.score_frame(forecast.prediction_frame(rows, fan))

    def test_keys_and_columns(self, scores) -> None:
        table = forecast.score_table(scores, "rmse", models=["step0", "oracle"])
        assert list(table.index.names) == list(forecast.BY_MODEL)
        assert list(table.columns) == sorted(scores["t"].unique())

    def test_by_series_puts_the_projections_in_one_block(self, scores) -> None:
        """Which key comes last decides what a block -- and its bolding -- compares."""
        table = forecast.score_table(
            scores, "bias", models=["step0"], by=forecast.BY_SERIES
        )
        assert list(table.index.names) == list(forecast.BY_SERIES)

    def test_rows_keep_the_ladder_and_forecaster_order(self, scores) -> None:
        """Never the order the frame happened to arrive in."""
        table = forecast.score_table(scores, "rmse", models=["step0", "oracle"])
        block = table.loc[("evil", "a")]
        assert [s for s, _ in block.index] == [
            series for series in decay.SERIES for _ in range(2)
        ]
        assert [m for _, m in block.index][:2] == ["step0", "oracle"]

    def test_selection_narrows_the_rows(self, scores) -> None:
        table = forecast.score_table(scores, "rmse", series=["p0"], models=["step0"])
        assert table.index.get_level_values("series").unique().tolist() == ["p0"]
        assert table.index.get_level_values("model").unique().tolist() == ["step0"]

    def test_unknown_metric_is_refused(self, scores) -> None:
        with pytest.raises(ValueError, match="unknown metric"):
            forecast.score_table(scores, "r2")

    def test_a_scrambled_key_order_is_refused(self, scores) -> None:
        with pytest.raises(ValueError, match="must order"):
            forecast.score_table(scores, "rmse", by=("trait", "trunk", "series"))

    def test_empty_scores_give_an_empty_table(self) -> None:
        assert forecast.score_table(pd.DataFrame(), "rmse").empty


# --- the emitted tables -----------------------------------------------------


class TestEmittedTables:
    """The LaTeX side: bolding runs the right way round for an error."""

    def test_an_error_leads_its_block_by_being_smallest(self) -> None:
        from method.visualization import make_plots

        table = pd.DataFrame(
            [[10.0, 2.0], [4.0, 8.0]],
            index=pd.MultiIndex.from_tuples(
                [("evil", "a", "p0", "step0"), ("evil", "a", "p0", "oracle")],
                names=list(forecast.BY_MODEL),
            ),
            columns=[0, 1],
        )
        leaders = make_plots._leading_cells(table, make_plots.ERROR_SCALE)
        assert leaders.to_numpy().tolist() == [[False, True], [True, False]]

    def test_a_bias_leads_its_block_by_being_nearest_zero(self) -> None:
        """Over- and under-predicting by the same amount are the same miss."""
        from method.visualization import make_plots

        table = pd.DataFrame(
            [[-1.0], [3.0]],
            index=pd.MultiIndex.from_tuples(
                [("evil", "a", "step0", "p0"), ("evil", "a", "step0", "full_t")],
                names=list(forecast.BY_SERIES),
            ),
            columns=[0],
        )
        leaders = make_plots._leading_cells(table, make_plots.BIAS_SCALE)
        assert leaders.to_numpy().tolist() == [[True], [False]]

    def test_a_correlation_still_leads_by_being_largest(self) -> None:
        """The default is unchanged, so the decay table is unaffected."""
        from method.visualization import make_plots

        table = pd.DataFrame(
            [[0.9], [0.4]],
            index=pd.MultiIndex.from_tuples(
                [("evil", "a", "p0"), ("evil", "a", "full_t")],
                names=list(decay.CORRELATION_TABLE_KEYS),
            ),
            columns=[0],
        )
        assert make_plots._leading_cells(table).to_numpy().tolist() == [
            [True],
            [False],
        ]

    @staticmethod
    def _pinned_table() -> pd.DataFrame:
        return pd.DataFrame(
            [[1.0], [2.0]],
            index=pd.MultiIndex.from_tuples(
                [
                    ("Evil", "A", r"$\Delta P_0$", "$M_0$ fit"),
                    ("Evil", "A", r"$\Delta P_0$", "Refit at $t$"),
                ],
                names=list(forecast.BY_MODEL),
            ),
            columns=[0],
        )

    def test_a_pinned_key_column_is_dropped_and_recorded(self) -> None:
        """A column repeating one value is width spent on nothing.

        What it said comes back as a note above the ``tabular``, which is where
        a caption-writer finds what the numbers are of.
        """
        from method.visualization import make_plots

        shown, pinned = make_plots._without_pinned_keys(
            self._pinned_table(), ["series"]
        )
        assert list(shown.index.names) == ["trait", "trunk", "model"]
        assert pinned == [("Projection", r"$\Delta P_0$")]

    def test_a_key_the_caller_did_not_pin_survives_being_constant(self) -> None:
        """A half-finished sweep can leave one trait in the frame.

        Dropping that column would turn a table whose scope is an accident of
        the sweep into one that looks deliberately scoped, so only what the
        spec pinned is dropped.
        """
        from method.visualization import make_plots

        shown, pinned = make_plots._without_pinned_keys(self._pinned_table(), [])
        assert list(shown.index.names) == list(forecast.BY_MODEL)
        assert pinned == []

    def test_the_last_key_is_never_dropped(self) -> None:
        """It is the one a block turns over on, pinned or not."""
        from method.visualization import make_plots

        shown, _ = make_plots._without_pinned_keys(
            self._pinned_table(), ["series", "model"]
        )
        assert shown.index.names[-1] == "model"

    def test_a_spec_pins_the_keys_it_holds_to_one_value(self) -> None:
        from method.visualization import make_plots

        specs = {spec.name: spec for spec in make_plots.FORECAST_TABLES}
        assert specs["exp2_forecast_correction_rmse"].pinned == ("series",)
        assert specs["exp2_forecast_bias"].pinned == ("model",)
        assert specs["exp2_forecast_rmse"].pinned == ()

    def test_a_note_names_the_metric(self) -> None:
        from method.visualization import make_plots

        spec = make_plots.FORECAST_TABLES[0]
        assert "RMSE" in make_plots._pinned_note(spec, [])


# --- the figure -------------------------------------------------------------


def _calibration_lines(ax) -> list:
    """The fitted lines an axes holds, excluding any identity reference.

    ``ax.axline`` returns an ``AxLine``, which subclasses ``Line2D`` and sits
    in ``ax.lines`` beside the fits -- so a plain count of ``ax.lines`` is one
    too many and excluding it has to be done by type.
    """
    from matplotlib.lines import AxLine

    return [line for line in ax.lines if not isinstance(line, AxLine)]


class TestRecalibrationGrid:
    """The projection-axis view: both fitted lines on the axes at once."""

    @staticmethod
    def _predictions(frames) -> pd.DataFrame:
        rows, fan = frames
        return forecast.prediction_frame(
            attenuated(rows, GAINS),
            fan,
            series=["p0"],
            models=list(forecast.RECALIBRATION_MODELS),
        )

    def test_each_model_draws_one_line_over_a_shared_cloud(self, frames) -> None:
        """One scatter per panel, however many models are drawn over it.

        The points are the same eight probes for every model -- only the line
        through them differs -- so scattering once per model would stack
        identical marks and read as heavier points.
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import PathCollection

        from method.visualization import figures

        predictions = self._predictions(frames)
        fig = figures.recalibration_grid(
            predictions,
            trunks=["c"],
            checkpoints=[3],
            models=forecast.RECALIBRATION_MODELS,
            model_labels=forecast.RECALIBRATION_LABELS,
            xlabel=r"Projection difference $\Delta P_0$",
        )
        try:
            ax = fig.axes[0]
            scattered = [
                point
                for c in ax.collections
                if isinstance(c, PathCollection)
                for point in np.asarray(c.get_offsets())
            ]
            assert len(scattered) == len(DELTA_P_0)
            # One line per model, plus the b_t rule.
            assert len(ax.lines) == 1 + len(forecast.RECALIBRATION_MODELS)
            fits = [
                line
                for line in _calibration_lines(ax)
                if line.get_color() != figures.style.INK
            ]
            assert {line.get_color() for line in fits} == {
                figures.style.BLUE,
                figures.style.ORANGE,
            }
            assert len(fits) == 2
            assert {line.get_linestyle() for line in fits} == {"-"}
            assert r"Projection difference $\Delta P_0$" in {
                text.get_text() for text in fig.texts
            }
            legend = fig.legends[0]
            assert [text.get_text() for text in legend.get_texts()[:3]] == [
                "Prediction",
                "Oracle",
                r"$b_t$",
            ]
            assert legend.get_title().get_text() == ""
        finally:
            plt.close(fig)

    def test_the_prediction_line_is_the_same_line_at_every_checkpoint(
        self, frames
    ) -> None:
        r"""The $M_0$ level prediction is carried unchanged across checkpoints.

        Its map from $\Delta P$ to $b_{t+1}$ never consults the checkpoint's
        current behaviour level, so only the oracle line moves.
        """
        predictions = self._predictions(frames)
        frozen = predictions[predictions["model"] == "step0_level"]
        slopes = set()
        for _, panel in frozen.groupby(list(forecast.CHECKPOINT)):
            fit = linear_fit(panel["delta_p"], panel["predicted_b_next"])
            slopes.add(round(fit.slope, 9))
        assert len(slopes) == 1
        assert slopes.pop() == pytest.approx(SLOPE)

    def test_the_refit_line_tracks_the_decay(self, frames) -> None:
        """The refit's slope follows the gain the fixture put in, exactly."""
        predictions = self._predictions(frames)
        refit = predictions[predictions["model"] == "oracle"]
        for key, panel in refit.groupby(list(forecast.CHECKPOINT)):
            fit = linear_fit(panel["delta_p"], panel["predicted_b_next"])
            assert fit.slope == pytest.approx(SLOPE * GAINS[key[2]])

    def test_a_panel_with_no_fitted_model_still_draws_its_probes(
        self, frames
    ) -> None:
        """A forecaster that could not be fitted costs the panel its line, not
        its points."""
        import matplotlib.pyplot as plt
        from matplotlib.collections import PathCollection

        from method.visualization import figures

        predictions = self._predictions(frames)
        predictions["predicted_b_next"] = np.nan
        fig = figures.recalibration_grid(predictions, trunks=["c"], checkpoints=[3])
        try:
            ax = fig.axes[0]
            scattered = [
                point
                for c in ax.collections
                if isinstance(c, PathCollection)
                for point in np.asarray(c.get_offsets())
            ]
            assert len(scattered) == len(DELTA_P_0)
        finally:
            plt.close(fig)


class TestForecastGrid:
    def test_the_pair_uses_blue_and_orange(self) -> None:
        from method.visualization import figures

        assert {
            figures.FORECAST_COLORS[model]
            for model in forecast.FORECAST_GRID_MODELS
        } == {figures.style.BLUE, figures.style.ORANGE}

    def test_the_level_forecast_pins_the_horizontal_axis(self, frames) -> None:
        r"""Why this grid draws the level fit: its x coordinate does not move.

        A level forecast never consults $b_t$, so a probe sits at the same
        predicted $b_{t+1}$ in every panel of a row and only the truth moves
        under it. The panel then shows the checkpoint drifting out from under a
        fixed prediction, rather than that drift compounded with the level
        riding up and down.
        """
        rows, fan = frames
        predictions = forecast.prediction_frame(
            attenuated(rows, GAINS),
            fan,
            series=["p0"],
            models=list(forecast.FORECAST_GRID_MODELS),
        )
        level = predictions[predictions["model"] == "step0_level"]
        moved = 0
        for _, group in level.groupby(["trait", "trunk", "probe"]):
            assert group["predicted_b_next"].nunique() == 1
            moved += group["b_next"].nunique() > 1
        # Not vacuous: the truth does move under those pinned predictions. Not
        # asserted per probe, because the probe whose Delta P_0 is zero has no
        # change to attenuate and so sits still on the flat trunk by
        # construction.
        assert moved


    def test_draws_every_panel_the_frame_covers(self, frames) -> None:
        import matplotlib.pyplot as plt

        from method.visualization import figures

        rows, fan = frames
        predictions = forecast.prediction_frame(
            rows, fan, series=["p0"], models=list(forecast.FORECAST_GRID_MODELS)
        )
        fig = figures.forecast_grid(
            predictions,
            trunks=list(TRUNKS),
            model_labels=forecast.FORECASTER_LABELS,
        )
        try:
            assert len(fig.axes) == len(TRUNKS) * predictions["t"].nunique()
        finally:
            plt.close(fig)

    def test_no_cloud_carries_a_fitted_line(self, frames) -> None:
        """The identity line is the only line this grid draws.

        A forecaster's own line is already spent placing the points along the x
        axis, so the only thing a fitted line here could show is a calibration
        -- and that belongs to the recalibration grid, whose x axis is the
        quantity a forecaster is a function of. Asserted because it is a
        deliberate choice and not an oversight: a fitted line is the obvious
        thing to add to a scatter, and this says not to.
        """
        import matplotlib.pyplot as plt

        from method.visualization import figures

        rows, fan = frames
        predictions = forecast.prediction_frame(
            attenuated(rows, GAINS),
            fan,
            series=["p0"],
            models=list(forecast.FORECAST_GRID_MODELS),
        )
        fig = figures.forecast_grid(
            predictions,
            trunks=["a"],
            checkpoints=[3],
            models=forecast.FORECAST_GRID_MODELS,
        )
        try:
            assert _calibration_lines(fig.axes[0]) == []
        finally:
            plt.close(fig)

    def test_a_model_with_no_prediction_is_not_drawn(self, frames) -> None:
        """An unfitted forecaster leaves a panel to the ones that did fit."""
        import matplotlib.pyplot as plt

        from method.visualization import figures

        rows, fan = frames
        predictions = forecast.prediction_frame(
            rows, fan, series=["p0"], models=["step0_level", "oracle"]
        )
        predictions.loc[
            predictions["model"] == "step0_level", "predicted_b_next"
        ] = np.nan
        panel = predictions[
            (predictions["trunk"] == "a") & (predictions["t"] == 0)
        ]
        assert figures._panel_models(panel, ["step0_level", "oracle"]) == ["oracle"]
        fig = figures.forecast_grid(predictions, trunks=["a"])
        plt.close(fig)
