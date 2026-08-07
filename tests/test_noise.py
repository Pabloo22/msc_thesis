"""Unit tests for the analytic noise maths in :mod:`method.noise`.

``noise.py`` touches no store and loads no model, so it can be exercised
directly on hand-built score frames. These tests pin down the properties the
rest of the design leans on: that ``SE`` measures the *within-question*
sampling error and not the between-question spread, that it stays defined when
the judge drops a generation, and that ``R2_max`` reports a ceiling rather than
a negative number.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from method.noise import (
    behavior_standard_error,
    behavior_summary,
    delta_b_noise_variance,
    r2_max,
)


def frame(scores_by_question: dict[str, list[float]], coherence: float = 90.0):
    """A behaviour CSV's worth of rows: one per generation, as the eval emits."""
    rows = [
        {"question_id": qid, "evil": score, "coherence": coherence}
        for qid, scores in scores_by_question.items()
        for score in scores
    ]
    return pd.DataFrame(rows)


class TestBehaviorStandardError:
    def test_deterministic_questions_have_no_error(self) -> None:
        # Every generation agrees, so b is pinned down exactly. This is the
        # floor/ceiling regime the docstring warns a single global number would
        # misrepresent.
        df = frame({"a": [0.0] * 10, "b": [100.0] * 10})
        assert behavior_standard_error(df, "evil") == pytest.approx(0.0)

    def test_ignores_between_question_spread(self) -> None:
        # Two questions that disagree wildly but are each internally
        # deterministic carry zero error, though their pooled std is 50.
        df = frame({"a": [0.0] * 10, "b": [100.0] * 10})
        assert df["evil"].std() > 40.0
        assert behavior_standard_error(df, "evil") == pytest.approx(0.0)

    def test_matches_closed_form(self) -> None:
        # One question, n=4, sample variance 1.0 -> SE = sqrt(1/4)/1 = 0.5.
        df = frame({"a": [1.0, 2.0, 3.0, 4.0]})
        variance = df["evil"].var(ddof=1)
        assert behavior_standard_error(df, "evil") == pytest.approx(
            math.sqrt(variance / 4.0)
        )

    def test_averages_down_over_questions(self) -> None:
        # N identical questions: SE = sqrt(N * s)/N = sqrt(s)/sqrt(N), so
        # quadrupling the question count halves the error.
        one = frame({"a": [1.0, 2.0, 3.0, 4.0]})
        four = frame({str(i): [1.0, 2.0, 3.0, 4.0] for i in range(4)})
        assert behavior_standard_error(four, "evil") == pytest.approx(
            behavior_standard_error(one, "evil") / 2.0
        )

    def test_single_generation_question_contributes_zero(self) -> None:
        # No within-question variance is estimable from one draw; it must not
        # become NaN and poison the whole sum.
        df = frame({"a": [5.0], "b": [1.0, 2.0, 3.0, 4.0]})
        result = behavior_standard_error(df, "evil")
        assert math.isfinite(result)
        assert result > 0.0

    def test_drops_unscored_generations(self) -> None:
        # A failed judge call leaves NaN; the question keeps its other rows.
        scored = frame({"a": [1.0, 2.0, 3.0, 4.0]})
        with_gap = frame({"a": [1.0, 2.0, 3.0, 4.0, float("nan")]})
        assert behavior_standard_error(with_gap, "evil") == pytest.approx(
            behavior_standard_error(scored, "evil")
        )

    def test_empty_is_an_error(self) -> None:
        df = frame({"a": [float("nan")]})
        with pytest.raises(ValueError, match="no scored generations"):
            behavior_standard_error(df, "evil")


class TestBehaviorSummary:
    def test_carries_the_legacy_fields_unchanged(self) -> None:
        # Backfilled artifacts must be indistinguishable from freshly measured
        # ones, so the pre-existing fields keep their exact old meanings.
        df = frame({"a": [0.0, 10.0], "b": [20.0, 30.0]}, coherence=80.0)
        summary = behavior_summary(df, "evil")
        assert summary["evil"] == pytest.approx(df["evil"].mean())
        assert summary["evil_std"] == pytest.approx(df["evil"].std())
        assert summary["coherence"] == pytest.approx(80.0)
        assert summary["n"] == 4

    def test_n_counts_generations_and_n_questions_counts_questions(self) -> None:
        df = frame({"a": [1.0] * 10, "b": [2.0] * 10})
        summary = behavior_summary(df, "evil")
        assert summary["n"] == 20
        assert summary["n_questions"] == 2

    def test_se_is_smaller_than_std_when_spread_is_between_questions(self) -> None:
        df = frame({"a": [0.0] * 10, "b": [100.0] * 10})
        summary = behavior_summary(df, "evil")
        assert summary["evil_se"] < summary["evil_std"]

    def test_json_safe_types(self) -> None:
        summary = behavior_summary(frame({"a": [1.0, 2.0]}), "evil")
        assert isinstance(summary["n"], int)
        assert isinstance(summary["n_questions"], int)
        assert all(
            isinstance(v, float)
            for k, v in summary.items()
            if k not in {"n", "n_questions"}
        )


class TestNoiseCeiling:
    def test_variance_terms_add(self) -> None:
        # Both endpoints' eval noise and the fine-tune seed noise are
        # independent, so they add in variance rather than cancelling.
        assert delta_b_noise_variance(3.0, 4.0, 0.0) == pytest.approx(25.0)
        assert delta_b_noise_variance(0.0, 0.0, 2.0) == pytest.approx(4.0)

    def test_clean_signal_has_ceiling_near_one(self) -> None:
        # The docstring's worked example: sigma_seed = 2 against a 20-point
        # spread leaves ~99% of the variance attributable.
        noise = delta_b_noise_variance(0.0, 0.0, 2.0)
        assert r2_max(20.0**2, noise) == pytest.approx(0.99)

    def test_noise_dominated_scatter_has_no_headroom(self) -> None:
        # sigma_seed = 15 against the same spread measures nothing.
        noise = delta_b_noise_variance(0.0, 0.0, 15.0)
        assert r2_max(20.0**2, noise) < 0.5

    def test_clamped_at_zero(self) -> None:
        # More noise than observed spread means the estimates disagree; the
        # honest reading is "no attainable signal", not a negative R^2.
        assert r2_max(1.0, 10.0) == 0.0

    def test_zero_spread_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="observed variance must be positive"):
            r2_max(0.0, 1.0)
