r"""RQ1 out of sample: what a predictor fitted once at $M_0$ is worth at $M_t$.

:mod:`method.visualization.decay` scores a projection difference by fitting a
line to the very checkpoint it is measured at and reporting that line's
correlation. That answers the scientific question -- is there still a
relationship -- but not the practical one. Nobody who has already paid for a
fan-out at the base model refits at every checkpoint; they carry $M_0$'s line
forward and read a prediction off it. This module scores *that*: one affine map
per trait, fitted once, applied unchanged at every later checkpoint, and the
error it makes there is what not refitting costs.

Three facts shape everything here.

*A correlation cannot see this.* Pearson $r$ is invariant under a fixed affine
map -- $\mathrm{corr}(\alpha + \beta x, y) = \mathrm{sign}(\beta)\,
\mathrm{corr}(x, y)$ -- so scoring $M_0$'s *predictions* by correlation would
reproduce :func:`method.visualization.decay.correlation_table` cell for cell.
What goes stale is the calibration, not the ordering, so the tables here are
errors on the judge's own 0-100 scale (:func:`score_frame`) and never
correlations.

*The probes are the test set.* The base line is fitted on the validation-fan
datasets that are **not** probes (:func:`baseline_fits`), so no checkpoint is
ever scored against a line that has seen its own probes. The $t = 0$ column is
therefore already a real held-out error rather than a residual, which is what
makes it the right thing to read the later columns against.

*What is cheap and what is not.* Re-measuring a projection difference at $M_t$
costs a probe fan-out; reading $z_t$ or the trait score $b_t$ off the
checkpoint costs nothing, since both are measured anyway. So the interesting
question is not only whether a refit helps but whether the *free* checkpoint
state can stand in for one -- which is what the corrected forecasters of
:data:`FORECASTERS` are for.

Reads only the frames :mod:`method.visualization.decay` already builds, so the
whole analysis runs on a laptop holding no adapters and no activations.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from method.visualization import decay
from method.visualization.metrics import LinearFit, linear_fit

logger = logging.getLogger(__name__)

#: The two quantities a line at $M_0$ can be fitted against, as
#: ``decay_frame`` columns.
#:
#: ``delta_b``
#:     $\Delta b$, how far the step *moved* the model. A prediction is turned
#:     back into a level by adding the checkpoint's own $b_t$.
#: ``b_next``
#:     $b_{t+1}$, where the step *landed* it. Predicted outright, with no
#:     reference to where the model started.
#:
#: These are two different claims about what fine-tuning on a dataset does, and
#: which is right turns out to depend on what the predictor knows. A
#: measurement taken at $M_0$ knows nothing about the checkpoint, so the only
#: claim it can honestly make is "this dataset lands you *here*"; one taken at
#: $M_t$ knows where the model is, so it can say "from here, this dataset moves
#: you *that* far". The tables bear it out -- see :data:`FORECASTERS`.
CHANGE, LEVEL = "delta_b", "b_next"
TARGETS = (CHANGE, LEVEL)

#: What every forecaster is *scored* on, whichever target it was fitted
#: against: the behaviour the step reached.
#:
#: This is what makes the two targets comparable in one table rather than two.
#: All ``K`` probes of a checkpoint branch off one model, so $b_t$ is a single
#: constant within the scatter, and a $\Delta b$ error is therefore a
#: $b_{t+1}$ error exactly:
#: $\widehat{\Delta b} - \Delta b = (b_t + \widehat{\Delta b}) - b_{t+1}$.
#: The target changes what a line is fitted on; it does not change the ruler.
SCORED_ON = LEVEL

#: The columns identifying one scatter: a trunk at a checkpoint, which is the
#: unit :func:`method.visualization.decay.fit_frame` collapses and the unit a
#: forecaster is scored over here.
CHECKPOINT = ("trait", "trunk", "t")

#: Checkpoint state a gain correction may be regressed on. $z_t$'s four
#: coordinates are the drift RQ1 is about; $b_t$ is the nuisance that has to be
#: carried beside them, because a model already near the judge's ceiling has
#: less room to move whatever its representation is doing, and a correction
#: credited to drift when the level explains it would be the whole finding
#: gone wrong.
GAIN_FEATURES = (*decay.Z_COMPONENTS, "b_t")

#: The error summaries :func:`score_frame` reports per scatter.
#:
#: ``rmse`` and ``mae`` say how far off the prediction was, in judge points.
#: ``bias`` -- the mean *signed* error -- says whether it was off in one
#: direction, which is what a stale slope looks like: as a trunk drifts, the
#: frozen line goes on promising the behaviour change $M_0$ would have made and
#: over-predicts every probe at once. That is invisible to ``rmse``, which
#: cannot tell a systematic offset from scatter.
#:
#: Only a forecaster carrying $M_0$'s intercept has a bias worth reporting.
#: Least squares with a free intercept leaves residuals summing to zero, so the
#: refit-at-$t$ row of a bias table is $0.0$ at every checkpoint by
#: construction -- see :data:`BIASED_MODELS`.
METRICS = ("rmse", "mae", "bias")


#: $M_0$'s lines, by target and then by trait.
Baselines = Mapping[str, Mapping[str, LinearFit]]


def baseline_fits(
    validation: pd.DataFrame, probes: Iterable[str]
) -> dict[str, dict[str, LinearFit]]:
    r"""One $M_0$ line per trait, fitted on the datasets that are *not* probes.

    The validation fan covers all 24 datasets and the decay experiment probes 8
    of them (section 5), so holding the probes out leaves 16 to fit on and
    costs nothing that was measured for this purpose anyway. Without that split
    the $t = 0$ column would be an in-sample residual and every later column a
    genuine prediction, and the decay down a row would be part artefact of the
    change in what is being reported.

    Fitted per trait and pooled over seeds: the persona vector and the judge
    are both per trait, so two traits share no units, while two seeds of one
    trait are replicates of one measurement.

    One set of lines per entry of :data:`TARGETS`, since a line fitted to
    predict how far a step *moves* the model is not the line that predicts
    where it *lands*, and both are wanted.

    A trait whose fan is missing or too small to fit simply has no entry, and
    every forecaster resting on it reports NaN rather than a number the sweep
    did not support.
    """
    fits: dict[str, dict[str, LinearFit]] = {target: {} for target in TARGETS}
    if validation.empty:
        return fits
    held_out = validation[~validation["dataset"].isin(set(probes))]
    for trait, group in held_out.groupby("trait"):
        if len(group) < 2:
            logger.warning(
                "exp2/%s: %d validation dataset(s) left once the probes are "
                "held out, which is too few to fit M_0's line; the step-0 "
                "forecasts for this trait will be blank",
                trait,
                len(group),
            )
            continue
        for target in TARGETS:
            fits[target][str(trait)] = linear_fit(
                group["delta_p_0"], group[target]
            )
    return fits


#: What a forecaster does: turn one series' column, over the rows it was
#: measured on, into a predicted $b_{t+1}$ aligned to those rows.
#:
#: A *level*, not a change, whichever target the forecaster was fitted on. That
#: is what lets one table hold both (see :data:`SCORED_ON`), and it puts the
#: conversion in the one place that knows which target was used rather than in
#: every consumer downstream.
Predict = Callable[[pd.DataFrame, str, Baselines], pd.Series]


@dataclass(frozen=True)
class Forecaster:
    """One named rule for turning a projection difference into a prediction."""

    name: str
    #: How the table writes it, in the key column beside the projection.
    label: str
    #: What the label leaves out, for a figure legend or a caption.
    gloss: str
    predict: Predict
    #: Whether the line itself is refitted at the checkpoint. A forecaster that
    #: refits has no bias to report (see :data:`METRICS`) and is not something
    #: anyone could run without the fan-out this whole analysis is costing.
    refits: bool = False
    #: Which of :data:`TARGETS` its line was fitted against. Meaningless for a
    #: refitting forecaster, which is target-invariant: within a checkpoint
    #: $b_t$ is one constant, so fitting $b_{t+1}$ rather than $\Delta b$ moves
    #: the intercept by exactly that constant and leaves the predicted level
    #: identical.
    target: str = CHANGE


def _blank(rows: pd.DataFrame) -> pd.Series:
    """A prediction column of the right shape, holding nothing yet."""
    return pd.Series(np.nan, index=rows.index, dtype=float)


def _as_level(
    predicted: NDArray[np.float64], group: pd.DataFrame, target: str
) -> NDArray[np.float64]:
    r"""A prediction on the judge's scale, whichever target produced it.

    A line fitted on $\Delta b$ predicts a *move* and needs the checkpoint's
    own $b_t$ added back; one fitted on $b_{t+1}$ predicts the level outright
    and must not have it added, which is the whole difference between the two
    -- the second never consults where the model currently is.
    """
    if target == LEVEL:
        return predicted
    return group["b_t"].to_numpy(dtype=float) + predicted


def _frozen(target: str) -> Predict:
    r"""$M_0$'s line, applied unchanged at every checkpoint."""

    def predict(
        rows: pd.DataFrame, column: str, baselines: Baselines
    ) -> pd.Series:
        fits = baselines.get(target, {})
        predicted = _blank(rows)
        for trait, group in rows.groupby("trait"):
            fit = fits.get(str(trait))
            if fit is not None:
                predicted.loc[group.index] = _as_level(
                    fit.predict(group[column]), group, target
                )
        return predicted

    return predict


def _refitted(
    rows: pd.DataFrame, column: str, baselines: Baselines
) -> pd.Series:
    r"""The line refitted on the checkpoint's own probes -- the ceiling.

    In-sample by construction: two parameters fitted to ``K`` points and scored
    on the same ``K``, which is exactly the fit
    :func:`method.visualization.decay.correlation_table` reports the
    correlation of. So it is optimistic in a way the step-0 forecasters are
    not, and it is here as the bound they are read against rather than as a
    method anyone could use -- refitting needs the fan-out whose cost is the
    reason for the question.
    """
    predicted = _blank(rows)
    for _, group in rows.groupby(list(CHECKPOINT)):
        fit = linear_fit(group[column], group[SCORED_ON])
        predicted.loc[group.index] = fit.predict(group[column])
    return predicted


def _corrected(features: Sequence[str], target: str = CHANGE) -> Predict:
    r"""$M_0$'s line on a projection rescaled by what the checkpoint says.

    The correction acts on $\Delta P$ and not on the line, because that is the
    claim being tested: the base model's map from projection difference to
    behaviour is taken to be right, and what a drifting checkpoint breaks is
    the *size* of the projection fed into it. So the forecast is
    $\alpha_0 + \beta_0\, g_t \Delta P$ with $\alpha_0, \beta_0$ frozen and one
    scalar gain $g_t$ per checkpoint, regressed on ``features``.

    ``g_t`` is fitted leave-one-trunk-out (:func:`_gain_forecast`), so a
    checkpoint's gain never comes from a model fitted on its own trajectory.
    """

    def predict(
        rows: pd.DataFrame, column: str, baselines: Baselines
    ) -> pd.Series:
        gains = _gain_forecast(rows, column, baselines, features, target)
        scale = pd.Series(
            [gains.get(key, np.nan) for key in _checkpoint_keys(rows)],
            index=rows.index,
            dtype=float,
        )
        rescaled = rows.assign(**{column: rows[column] * scale})
        return _frozen(target)(rescaled, column, baselines)

    return predict


#: The forecasters, in the order a table lists them: the frozen line under each
#: of its two targets, the two things a free reading of the checkpoint can do
#: to it, and the refit that bounds them all.
#:
#: The first two differ only in what $M_0$'s line was fitted to predict, and
#: which of them wins turns on what the projection difference knows. A
#: measurement frozen at $M_0$ -- $\Delta P_0$ and both hatted rungs -- does
#: better predicting the *level* the step lands at, because it has no way to
#: know where the checkpoint currently is and "this dataset lands you here" is
#: the only claim it is entitled to. $\Delta P_t$, whose encoder and answers
#: are both current, does better predicting the *change*: it knows where the
#: model is, so it can say how far from there the step will move it. That
#: crossover is what ``exp2_forecast_target_rmse`` tabulates.
#:
#: The two corrections differ only in what the gain is regressed on, which is
#: the comparison that makes either readable. $z_t$ is the drift RQ1 names;
#: $b_t$ is the level, equally free to read and not a representation claim at
#: all. A $z_t$ correction that beats the frozen line says nothing until it is
#: put beside a $b_t$ correction that costs the same.
FORECASTERS: tuple[Forecaster, ...] = (
    Forecaster(
        "step0",
        r"$M_0$ fit on $\Delta b$",
        "fitted at the base model to predict the change",
        _frozen(CHANGE),
        target=CHANGE,
    ),
    Forecaster(
        "step0_level",
        r"$M_0$ fit on $b_{t+1}$",
        "fitted at the base model to predict the level",
        _frozen(LEVEL),
        target=LEVEL,
    ),
    Forecaster(
        "step0_z",
        r"$M_0$ fit $\times\, g(z_t)$",
        "rescaled by the latent state",
        _corrected(decay.Z_COMPONENTS),
    ),
    Forecaster(
        "step0_b",
        r"$M_0$ fit $\times\, g(b_t)$",
        "rescaled by the behaviour level",
        _corrected(("b_t",)),
    ),
    Forecaster(
        "oracle",
        r"Refit at $t$",
        "refitted on the checkpoint",
        _refitted,
        refits=True,
    ),
)

FORECASTER_LABELS = {f.name: f.label for f in FORECASTERS}

#: The pair the headline comparison is between: what a practitioner can have,
#: against what they would have if refitting were free.
HEADLINE_MODELS = ("step0", "oracle")

#: The two targets of the frozen line, with the refit as their shared
#: reference. The refit appears once, not twice, because it is target-invariant
#: (see :attr:`Forecaster.target`).
TARGET_MODELS = ("step0", "step0_level", "oracle")

#: The forecasters whose bias is a measurement rather than an identity: the
#: ones carrying $M_0$'s intercept forward (see :data:`METRICS`).
BIASED_MODELS = tuple(f.name for f in FORECASTERS if not f.refits)

#: The forecasters a correction table compares, on the one projection that
#: needs correcting. $\Delta P_0$ is the rung nothing at $M_t$ has refreshed,
#: so it is where a free reading of the checkpoint has something to fix; the
#: refit bounds what any of them could do.
CORRECTION_MODELS = ("step0", "step0_z", "step0_b", "oracle")

#: What the recalibration grid draws: the level prediction carried from
#: $M_0$, against the line refitted on the checkpoint. The change-target fit
#: is excluded because a prediction of $b_{t+1}$ belongs on the level target.
RECALIBRATION_MODELS = ("step0_level", "oracle")
RECALIBRATION_LABELS = {"step0_level": "Prediction", "oracle": "Oracle"}

#: What the predicted-against-actual grid draws: the level fit against the
#: refit.
#:
#: The level fit rather than the change fit, because that grid forecasts from
#: $\Delta P_0$ and the level is the target $\Delta P_0$ should be fitted on
#: (see :data:`FORECASTERS`) -- drawing the weaker of the two would be showing
#: the frozen predictor at less than its best and calling the gap to the refit
#: the cost of not refitting.
#:
#: It also makes the axes say something the change fit cannot. A level forecast
#: never consults $b_t$, so a probe's predicted $b_{t+1}$ is the *same number*
#: at every checkpoint: the cloud is pinned horizontally and only the truth
#: moves under it. What the panel then shows is the checkpoint drifting out
#: from under a fixed prediction, which is the staleness itself rather than the
#: staleness plus the level riding up and down.
FORECAST_GRID_MODELS = ("step0_level", "oracle")


def forecasters(names: Sequence[str] | None = None) -> list[Forecaster]:
    """The named forecasters in :data:`FORECASTERS` order, all of them by default."""
    if names is None:
        return list(FORECASTERS)
    wanted = set(names)
    unknown = wanted - {f.name for f in FORECASTERS}
    if unknown:
        raise ValueError(f"unknown forecaster(s): {sorted(unknown)}")
    return [f for f in FORECASTERS if f.name in wanted]


def _checkpoint_keys(rows: pd.DataFrame) -> list[tuple[str, str, int]]:
    """``(trait, trunk, t)`` per row, typed so it keys a plain dict."""
    return [
        (str(trait), str(trunk), int(t))
        for trait, trunk, t in rows[list(CHECKPOINT)].itertuples(index=False)
    ]


def _ideal_gain(fit: LinearFit, x: ArrayLike, y: ArrayLike) -> float:
    r"""The one number the frozen slope should have been multiplied by here.

    Least squares over the single free parameter $g$ in
    $\alpha_0 + \beta_0\, g\, x$, which has a closed form because the intercept
    is fixed: $g = \langle u, y - \alpha_0 \rangle / \langle u, u \rangle$ with
    $u = \beta_0 x$. This is what a correction is *trying* to predict, so it is
    also the quantity to look at when one fails to.
    """
    scaled = fit.slope * np.asarray(x, dtype=float)
    residual = np.asarray(y, dtype=float) - fit.intercept
    denominator = float(scaled @ scaled)
    if denominator <= 0:
        return float("nan")
    return float(scaled @ residual / denominator)


def gain_frame(
    rows: pd.DataFrame,
    column: str,
    baselines: Baselines,
    target: str = CHANGE,
) -> pd.DataFrame:
    """The ideal gain per checkpoint, beside the state it might be read off.

    Public because it is the diagnostic behind every corrected forecaster: a
    correction can only work as well as ``gain`` is predictable from
    :data:`GAIN_FEATURES`, and a table of errors says whether it worked, not
    why.
    """
    fits = baselines.get(target, {})
    records = []
    for key, group in rows.groupby(list(CHECKPOINT), sort=True):
        trait, trunk, t = key
        fit = fits.get(str(trait))
        if fit is None or group[column].isna().any():
            continue
        records.append(
            {
                "trait": trait,
                "trunk": trunk,
                "t": t,
                "gain": _ideal_gain(fit, group[column], group[target]),
                **{name: float(group[name].iloc[0]) for name in GAIN_FEATURES},
            }
        )
    return pd.DataFrame(records, columns=["gain", *GAIN_FEATURES, *CHECKPOINT])


def _design(frame: pd.DataFrame, features: Sequence[str]) -> NDArray[np.float64]:
    """An intercept column and one column per feature."""
    return np.column_stack(
        [np.ones(len(frame)), *(frame[name].to_numpy(dtype=float) for name in features)]
    )


def _least_squares(
    design: NDArray[np.float64], target: ArrayLike
) -> NDArray[np.float64] | None:
    """OLS coefficients, or ``None`` where there are too few rows to fit them.

    Strictly more rows than parameters, so a fold that could only interpolate
    reports nothing rather than a gain of no information. That is the case a
    single-trunk sweep is in -- there is no other trunk to fit on -- and the
    forecasts it cannot make should read as gaps.
    """
    if design.shape[0] <= design.shape[1]:
        return None
    coefficients, *_ = np.linalg.lstsq(
        design, np.asarray(target, dtype=float), rcond=None
    )
    return coefficients


def _gain_forecast(
    rows: pd.DataFrame,
    column: str,
    baselines: Baselines,
    features: Sequence[str],
    target: str = CHANGE,
) -> dict[tuple[str, str, int], float]:
    r"""Predicted gain per checkpoint, never fitted on its own trunk.

    Leave-one-*trunk*-out rather than leave-one-checkpoint-out. Consecutive
    checkpoints of one trunk are the same trajectory a step further along, so
    holding one out while its neighbours stay in the training set would let the
    correction be read off the trunk it is scored on. Holding out the whole
    trunk asks the question the table is for: does the state at a checkpoint
    predict the gain on a trajectory the correction has never seen?

    Fitted within a trait, since the gain is a ratio of slopes measured against
    that trait's own persona vector and judge.
    """
    frame = gain_frame(rows, column, baselines, target)
    forecast: dict[tuple[str, str, int], float] = {}
    if frame.empty:
        return forecast
    for _, block in frame.groupby("trait"):
        for trunk in sorted(block["trunk"].unique()):
            train = block[block["trunk"] != trunk]
            test = block[block["trunk"] == trunk]
            coefficients = _least_squares(_design(train, features), train["gain"])
            if coefficients is None:
                continue
            for record, gain in zip(
                test.itertuples(index=False), _design(test, features) @ coefficients
            ):
                key = (str(record.trait), str(record.trunk), int(record.t))
                forecast[key] = float(gain)
    return forecast


def _measured(rows: pd.DataFrame, column: str) -> pd.DataFrame:
    """The checkpoints whose ``column`` is complete, dropped whole where it is not.

    The same rule :func:`method.visualization.decay._series_fit` applies, for
    the same reason: an error over whichever probes happened to be measured is
    an error over a different probe set than the one it is printed beside, and
    a table of eight-probe errors is read across its rows.
    """
    if column not in rows or rows.empty:
        return rows.iloc[:0]
    complete = rows.groupby(list(CHECKPOINT))[column].transform(
        lambda values: bool(values.notna().all())
    )
    return rows[complete.astype(bool)]


_PREDICTION_COLUMNS = [
    "trait", "trunk", "seed", "t", "probe", "series", "model", "target",
    "steps_since_realignment", "delta_p", "b_t", "b_next", "se_b_next",
    "delta_b", "predicted_b_next", "predicted_delta_b", "error",
]


def prediction_frame(
    rows: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    series: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
) -> pd.DataFrame:
    r"""One row per ``(trunk, t, probe, series, model)``: a prediction and its truth.

    ``rows`` is :func:`method.visualization.decay.decay_frame` and
    ``validation`` is :func:`method.visualization.decay.validation_frame`; the
    second is what the step-0 forecasters are fitted on, and without it they
    are blank.

    Long rather than wide because the two things built from it want different
    shapes: :func:`score_frame` collapses each ``(checkpoint, series, model)``
    to its error, and the grid figure draws each ``(checkpoint, model)`` as a
    cloud of ``K`` points against the identity line.

    Every forecaster produces ``predicted_b_next``, the behaviour it says the
    step will reach, whichever target its line was fitted on (see
    :data:`SCORED_ON`); ``b_next`` beside it is what the step actually reached,
    and ``error`` is the difference. ``predicted_delta_b`` is the same
    prediction as a change, read back by subtracting the level the checkpoint
    started from, and pairs with ``delta_b`` the way the other two pair. Both
    views are carried so that a forecaster fitted on the level and one fitted
    on the change can be compared in either, and so the names say which is
    which -- a column called ``predicted`` beside one called ``actual`` stops
    being honest the moment the two stop being the same quantity.
    """
    wanted = [name for name in (series or decay.SERIES)]
    unknown = set(wanted) - set(decay.SERIES_COLUMNS)
    if unknown:
        raise ValueError(f"unknown series: {sorted(unknown)}")
    if rows.empty:
        return pd.DataFrame(columns=_PREDICTION_COLUMNS)

    baselines = baseline_fits(validation, rows["probe"].unique())
    if not any(baselines.values()):
        logger.warning(
            "exp2: no validation fan outside the probe set, so M_0's line "
            "cannot be fitted without seeing its own test points; only the "
            "refit-at-t forecaster will be scored"
        )
    frames = []
    for name in wanted:
        column = decay.SERIES_COLUMNS[name]
        measured = _measured(rows, column)
        if measured.empty:
            continue
        for forecaster in forecasters(models):
            predicted = forecaster.predict(measured, column, baselines)
            frames.append(
                measured.assign(
                    series=name,
                    model=forecaster.name,
                    target=forecaster.target,
                    delta_p=measured[column],
                    predicted_b_next=predicted,
                    predicted_delta_b=predicted - measured["b_t"],
                    error=predicted - measured[SCORED_ON],
                )
            )
    if not frames:
        return pd.DataFrame(columns=_PREDICTION_COLUMNS)
    return pd.concat(frames, ignore_index=True)[_PREDICTION_COLUMNS]


_SCORE_COLUMNS = ["trait", "trunk", "t", "series", "model", "n", *METRICS]


def score_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    r"""Collapse each ``(checkpoint, series, model)`` cloud to its error.

    All of :data:`METRICS` are in judge points, on the same 0-100 scale as
    $\Delta b$ itself, so a cell is read as "off by this much of the judge's
    range" without a second number to divide by.

    A cloud with any missing prediction scores NaN rather than scoring the part
    of it that exists. That happens where a forecaster could not be fitted at
    all -- no held-out fan for the trait, no other trunk to leave one out
    against -- and a number over the surviving probes would be an error on a
    different probe set than the one beside it.
    """
    if predictions.empty:
        return pd.DataFrame(columns=_SCORE_COLUMNS)
    records = []
    for key, group in predictions.groupby(
        ["trait", "trunk", "t", "series", "model"], sort=True
    ):
        trait, trunk, t, series, model = key
        error = group["error"].to_numpy(dtype=float)
        usable = bool(error.size) and bool(np.isfinite(error).all())
        records.append(
            {
                "trait": trait,
                "trunk": trunk,
                "t": t,
                "series": series,
                "model": model,
                "n": len(group),
                "rmse": float(np.sqrt(np.mean(error**2))) if usable else float("nan"),
                "mae": float(np.mean(np.abs(error))) if usable else float("nan"),
                "bias": float(np.mean(error)) if usable else float("nan"),
            }
        )
    return pd.DataFrame(records, columns=_SCORE_COLUMNS)


#: The key order a score table is indexed by, outermost first. The *last* key
#: is the one that varies inside a block, so it is the one a table's bolding
#: compares -- see :func:`method.visualization.make_plots._leading_cells`.
BY_MODEL = ("trait", "trunk", "series", "model")
BY_SERIES = ("trait", "trunk", "model", "series")


def score_table(
    scores: pd.DataFrame,
    metric: str = "rmse",
    *,
    series: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    by: Sequence[str] = BY_MODEL,
) -> pd.DataFrame:
    r"""``metric`` per ``(trait, trunk, series, model)``, one column per checkpoint.

    The out-of-sample counterpart of
    :func:`method.visualization.decay.correlation_table`, and laid out to be
    read beside it: the same trait and trunk keys, the same checkpoint columns,
    with the projection ladder split one row per forecaster.

    ``by`` orders the keys, and the choice is not cosmetic -- the last key is
    what varies within a block, and a block is what a reader (and the emitted
    table's bolding) compares. :data:`BY_MODEL` puts the same projection's two
    forecasts on adjacent rows, which is the comparison a headline table is
    for; :data:`BY_SERIES` puts one forecaster's four projections together,
    which is what a table carrying a single forecaster wants instead.

    ``series`` and ``models`` select and order what is carried, defaulting to
    everything present. As in the correlation table, a row measured at some
    checkpoints and not others keeps its row with a gap: which cells are
    missing is itself the state of the sweep.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
    if set(by) != set(BY_MODEL):
        raise ValueError(f"`by` must order {BY_MODEL}, got {tuple(by)}")
    if scores.empty or metric not in scores:
        return pd.DataFrame()
    wanted_series = list(series) if series is not None else list(decay.SERIES)
    wanted_models = [f.name for f in forecasters(models)]
    kept = scores[
        scores["series"].isin(wanted_series) & scores["model"].isin(wanted_models)
    ]
    if kept.empty:
        return pd.DataFrame()
    kept = kept.assign(
        series=pd.Categorical(kept["series"], categories=wanted_series, ordered=True),
        model=pd.Categorical(kept["model"], categories=wanted_models, ordered=True),
    )
    table = kept.pivot(index=list(by), columns="t", values=metric)
    return table.dropna(how="all").sort_index()
