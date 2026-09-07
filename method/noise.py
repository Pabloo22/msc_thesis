"""Estimate behaviour-measurement noise and fit ceilings."""

from __future__ import annotations

import math

import pandas as pd

#: Column identifying which eval question a generation answered. Written by the
#: vendored ``eval/eval_persona.py``, which emits one row per generation.
QUESTION_COL = "question_id"


def behavior_standard_error(
    scores: pd.DataFrame, column: str, *, question_col: str = QUESTION_COL
) -> float:
    r"""Standard error of the question-averaged mean of ``column``."""
    scored = scores.dropna(subset=[column])
    if scored.empty:
        raise ValueError(f"no scored generations in column {column!r}")

    per_question = scored.groupby(question_col)[column]
    counts = per_question.count()
    # A single-row group has no unbiased variance estimate; pandas gives NaN.
    variances = per_question.var(ddof=1).fillna(0.0)
    return math.sqrt(float((variances / counts).sum())) / len(counts)


def behavior_summary(
    scores: pd.DataFrame, trait: str, *, question_col: str = QUESTION_COL
) -> dict[str, float]:
    """The full ``behavior.json`` payload for one checkpoint's eval output.

    The single definition of that artifact's contents, shared by
    :func:`method.steps.measure_behavior` (which writes it at measurement time)
    and :mod:`method.backfill_se` (which rewrites it for checkpoints measured
    before ``SE`` existed).

    ``n`` counts scored *generations*, not questions, and is kept with that
    meaning because artifacts already on disk use it that way; ``n_questions``
    is the count the standard error is divided by and is recorded alongside so
    the number can be checked without the CSV.
    """
    return {
        trait: float(scores[trait].mean()),
        f"{trait}_std": float(scores[trait].std()),
        f"{trait}_se": behavior_standard_error(
            scores, trait, question_col=question_col
        ),
        "coherence": float(scores["coherence"].mean()),
        "n": int(len(scores)),
        "n_questions": int(scores[question_col].nunique()),
    }


def delta_b_noise_variance(
    se_before: float, se_after: float, sigma_seed: float
) -> float:
    r"""Variance in :math:`\Delta b` that no predictor can explain.

    :math:`\Delta b_{t+1} = b_{t+1} - b_t` inherits eval noise from both
    endpoints, and the fine-tune that produced :math:`b_{t+1}` contributes seed
    noise on top:

    .. math:: Var(\Delta b)_{noise} = \sigma_{seed}^2 + SE(b_{t+1})^2 + SE(b_t)^2

    The two eval terms add rather than cancel: they are independent draws, since
    each checkpoint's generations are sampled separately even though the
    question set is shared.
    """
    return sigma_seed**2 + se_after**2 + se_before**2


def r2_max(observed_variance: float, noise_variance: float) -> float:
    r"""Largest :math:`R^2` attainable given irreducible noise in ``y``.

    .. math:: R^2_{max} = 1 - Var(\Delta b)_{noise} / Var(\Delta b)_{observed}

    Noise in ``y`` cannot be explained by ``x``, so it caps the fit regardless
    of how good the predictor is. Reading a decay curve without this ceiling
    drawn on it confuses "the probe went stale" with "the scatter was mostly
    noise all along" -- and only the first is a finding.

    Clamped at zero: a noise estimate exceeding the observed spread means the
    inputs disagree, not that :math:`R^2` is negative, and the honest reading of
    that is "no attainable signal".
    """
    if observed_variance <= 0:
        raise ValueError(
            f"observed variance must be positive, got {observed_variance!r}; "
            "with no spread in Delta b there is no correlation to attenuate"
        )
    return max(0.0, 1.0 - noise_variance / observed_variance)
