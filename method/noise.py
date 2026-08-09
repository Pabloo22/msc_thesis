"""How precisely ``b`` is measured, and what that implies for a correlation.

Pure numeric helpers over already-scored eval output, deliberately free of any
store, model or plotting access, so they run identically inside a live
measurement and over an archived CSV, and can be unit tested on synthetic
frames.

The noise budget separates three sources and warns against collapsing them
into one number. Two of them are computable from data that already exists:

``SE(b)``
    Eval noise: how precisely one checkpoint's trait score is pinned down,
    derived analytically from the per-generation scores rather than by
    repeating the eval. Repeat evals would spend GPU time re-measuring
    something nearly deterministic, and would yield one global number, whereas
    the variance is strongly heteroscedastic -- near the floor or ceiling of the
    0-100 scale a question's generations agree exactly, while mid-range the
    model is genuinely bimodal.
``R2_max``
    The ceiling that irreducible noise in ``Delta b`` imposes on any
    correlation fitted against it. This is what distinguishes "the probe went
    stale" from "there was never enough signal to fit".

The third, fine-tune seed noise, cannot be derived from a single run's scores;
it needs replicate fine-tunes and is estimated from the seed replication that
exp3 already carries.

:func:`behavior_summary` lives here rather than in :mod:`method.steps` so that
the measurement path and the backfill path share one definition of what
``behavior.json`` contains. Two implementations that drifted apart would make a
backfilled checkpoint silently incomparable with a freshly measured one.
"""

from __future__ import annotations

import math

import pandas as pd

#: Column identifying which eval question a generation answered. Written by the
#: vendored ``eval/eval_persona.py``, which emits one row per generation.
QUESTION_COL = "question_id"


def behavior_standard_error(
    scores: pd.DataFrame, column: str, *, question_col: str = QUESTION_COL
) -> float:
    r"""Standard error of the question-averaged mean of ``column``.

    ``b`` is the average over eval questions of the average over that
    question's generations, so its sampling error comes only from the finite
    number of generations drawn per question:

    .. math:: SE(b)^2 = \frac{1}{N^2} \sum_q \frac{\sigma_q^2}{n_q}

    where :math:`\sigma_q` is the standard deviation across question ``q``'s
    generations, :math:`n_q` how many of them were scored, and :math:`N` the
    number of questions.

    The eval question set is *fixed* across checkpoints, so question-to-question
    variation is a constant offset that cancels in :math:`\Delta b` and
    correctly does not appear in this formula. That is what makes this a
    different quantity from the ``<trait>_std`` recorded beside it, which is the
    spread over every row and is therefore dominated by exactly the
    between-question variation that cancels. The two are not interchangeable and
    ``<trait>_std`` badly overstates the error on ``b``.

    Questions with a single scored generation have no within-question variance
    to estimate and contribute zero, which understates the error rather than
    inventing one. Unscored generations (a judge call that failed) are dropped,
    and the remaining count for that question is used, so a partial failure
    widens the interval instead of biasing it.
    """
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
