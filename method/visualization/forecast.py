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

*No dataset predicts itself in the tables.* Nothing fitted on dataset $j$ is
used to predict dataset $j$, and everything else that was measured is fair
game. So the table baseline for a probe is fitted on the other 23 validation
datasets (:func:`baseline_fits`), and a gain correction is fitted leaving out
both the trunk it is scored on and the probe it predicts
(:func:`_gain_forecast`). The $t = 0$ column is therefore a real held-out error
rather than a residual, which is what makes it the right thing to read the
later columns against.

The recalibration grid has a different job: it must draw one actual affine
line rather than connect predictions from eight leave-one-out folds. Its
$M_0$ line is therefore fitted on the 16 validation datasets outside the probe
set (:func:`nonprobe_baseline_fits`) and evaluated on all eight probes. This
display-only split does not change the leave-one-out scores in the tables.

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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from method.visualization import decay
from method.visualization.labels import (
    source_index,
    z_component_symbol,
    z_symbol,
)
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

#: How each $z_t$ coordinate is written in a table's key column, and what it
#: is in words. Kept beside :data:`GAIN_FEATURES` because the single-component
#: correctors are generated from :data:`method.visualization.decay.Z_COMPONENTS`
#: and a coordinate without a label would reach a table as a bare column name.
Z_GLOSSES = {
    "p": "the neutral state's alignment with the base persona axis",
    "q": "the neutral state's alignment with the current persona axis",
    "rho": "how far the persona axis has rotated",
    "r": "the current persona vector's length",
}

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


#: The fold that holds nothing out: the line fitted on the whole validation
#: fan. It is what a probe the fan never covered falls back to, since a dataset
#: nobody fine-tuned on at $M_0$ cannot be in any fit and so cannot leak from
#: one, and what :func:`gain_frame` uses when asked for the plain diagnostic.
WHOLE_FAN = None


@dataclass(frozen=True)
class Baselines:
    r"""$M_0$'s lines, one per (target, trait, held-out dataset).

    A mapping rather than a single line per trait because every line here is
    fitted leave-one-dataset-out: see :func:`baseline_fits` for why.
    """

    #: Keyed ``(target, trait, held-out dataset)``, with :data:`WHOLE_FAN`
    #: standing for the fold that holds nothing out.
    fits: Mapping[tuple[str, str, str | None], LinearFit]

    def line(self, target: str, trait: str, without: str | None) -> LinearFit | None:
        """The line for predicting ``without``, or ``None`` if there is none.

        Falls back to the whole-fan line where ``without`` has no fold of its
        own, which happens when the fan never covered that dataset. Nothing
        leaks: a dataset the fan did not measure is in no fit to begin with.
        """
        fit = self.fits.get((target, trait, without))
        if fit is not None:
            return fit
        return self.fits.get((target, trait, WHOLE_FAN))

    def __bool__(self) -> bool:
        return bool(self.fits)


def baseline_fits(validation: pd.DataFrame) -> Baselines:
    r"""$M_0$'s lines, fitted leave-one-dataset-out over the validation fan.

    The fan covers all 24 datasets at the base model, 8 of which the decay
    experiment goes on to probe (section 5). To predict dataset $j$ this fits
    on the other 23, so $j$ is never in the line that scores it and the
    $t = 0$ column is a real held-out error rather than an in-sample residual.

    Leave-one-out rather than a fixed probe/non-probe split, and the reason is
    consistency with the correction. A gain model reads the *other* probes'
    outcomes at every checkpoint of the other trunks (:func:`_gain_forecast`);
    a base line fitted on the 16 non-probe datasets would meanwhile pretend
    those same probes had never been fine-tuned on at all. Both cannot be true
    of one practitioner. Leave-one-out keeps the test point just as clean and
    states the rule once for the whole module: **nothing fitted on dataset $j$
    is used to predict dataset $j$**, and everything else that was measured is
    fair game.

    It is also what the RQ1 question actually asks. The quantity of interest is
    how well a line fitted at $M_0$ predicts a dataset it has not seen, and
    leave-one-out is the standard estimator of exactly that. Fitting on 16 and
    scoring a fixed 8 is one train/test split of the same thing -- a wastier
    one, and one whose split can be unrepresentative.

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
    fits: dict[tuple[str, str, str | None], LinearFit] = {}
    if validation.empty:
        return Baselines(fits)
    for trait, group in validation.groupby("trait"):
        if len(group) < 3:
            logger.warning(
                "exp2/%s: %d validation dataset(s) on disk, which leaves too "
                "few to fit M_0's line once one is held out; the step-0 "
                "forecasts for this trait will be blank",
                trait,
                len(group),
            )
            continue
        for held in (WHOLE_FAN, *sorted(group["dataset"].unique())):
            panel = group if held is WHOLE_FAN else group[group["dataset"] != held]
            for target in TARGETS:
                fits[(target, str(trait), held)] = linear_fit(
                    panel["delta_p_0"], panel[target]
                )
    return Baselines(fits)


def nonprobe_baseline_fits(
    validation: pd.DataFrame, probes: Sequence[str]
) -> Baselines:
    r"""One $M_0$ line per trait, fitted after holding out every probe.

    This is the baseline used by the recalibration grid.  Unlike
    :func:`baseline_fits`, which makes one leave-one-out fold per scored probe,
    this makes one common fit from the validation datasets outside the probe
    set.  All plotted probe predictions therefore lie on one affine line.

    The table path continues to use :func:`baseline_fits`; keeping this split
    explicit prevents a visualisation requirement from changing its scores.
    """
    fits: dict[tuple[str, str, str | None], LinearFit] = {}
    if validation.empty:
        return Baselines(fits)
    held_out = {str(probe) for probe in probes}
    for trait, group in validation.groupby("trait"):
        training = group[~group["dataset"].astype(str).isin(held_out)]
        if len(training) < 2:
            logger.warning(
                "exp2/%s: %d non-probe validation dataset(s) on disk, which "
                "is too few to fit the recalibration grid's M_0 line",
                trait,
                len(training),
            )
            continue
        for target in TARGETS:
            fits[(target, str(trait), WHOLE_FAN)] = linear_fit(
                training["delta_p_0"], training[target]
            )
    return Baselines(fits)


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
        predicted = _blank(rows)
        for (trait, probe), group in rows.groupby(["trait", "probe"]):
            fit = baselines.line(target, str(trait), str(probe))
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

    ``g_t`` is fitted leave-one-trunk-out *and* leave-one-probe-out
    (:func:`_gain_forecast`), so a probe's gain comes from a model that saw
    neither its trajectory nor its dataset. The gain is therefore a number per
    ``(checkpoint, probe)`` rather than one shared across the checkpoint's
    scatter, and the ``K`` corrected predictions of a checkpoint come from
    ``K`` different gain models.
    """

    def predict(
        rows: pd.DataFrame, column: str, baselines: Baselines
    ) -> pd.Series:
        gains = _gain_forecast(rows, column, baselines, features, target)
        scale = pd.Series(
            [gains.get(key, np.nan) for key in _gain_keys(rows)],
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
        rf"$M_0$ fit $\times\, g({z_symbol()})$",
        "rescaled by the latent state",
        _corrected(decay.Z_COMPONENTS),
    ),
    *(
        Forecaster(
            f"step0_{name}",
            rf"$M_0$ fit $\times\, g({z_component_symbol(name)})$",
            f"rescaled by {Z_GLOSSES[name]}, and nothing else",
            _corrected((name,)),
        )
        for name in decay.Z_COMPONENTS
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

#: The labels as :data:`FORECASTERS` declares them, which is the base-source
#: reading of the checkpoint state.
FORECASTER_LABELS = {f.name: f.label for f in FORECASTERS}


def forecaster_labels(source: str = "base") -> dict[str, str]:
    """:data:`FORECASTER_LABELS`, with $z_t$ indexed by its response source.

    Only the state-corrected rows move: every other forecaster is a line
    through a projection difference and says nothing about ``h_neutral``. The
    index matters because the same table can be produced from either source
    and the two are different corrections of the same frozen line -- one
    reading $M_0$'s neutral answers, one the checkpoint's own.
    """
    index = source_index(source)
    labels = dict(FORECASTER_LABELS)
    labels["step0_z"] = rf"$M_0$ fit $\times\, g({z_symbol(neutral=index)})$"
    for name in decay.Z_COMPONENTS:
        symbol = z_component_symbol(name, neutral=index)
        labels[f"step0_{name}"] = rf"$M_0$ fit $\times\, g({symbol})$"
    return labels

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
#:
#: Read top to bottom it is one argument. The frozen line is the bar. $g(z_t)$
#: is the latent state fitted whole. Its four coordinates then appear one at a
#: time, and $g(b_t)$ -- the behaviour level, equally free to read and not a
#: representation claim at all -- is the control every one of them has to beat
#: before any of this is about drift.
#:
#: The single-coordinate rows are not there to find a winner. They are there
#: because $g(z_t)$ spends five parameters on a fold of roughly fourteen
#: checkpoints drawn from two trunks, and "you overfitted" is the first thing
#: anyone will say about a correction that fails. Each coordinate on its own
#: costs the same two parameters as $g(b_t)$, so a correction that still fails
#: cannot be failing for want of parsimony. That they also answer RQ1's own
#: sub-question -- which part of the projection difference carries the drift --
#: is what makes them worth the rows rather than a footnote.
#:
#: All of them are reported every time, never the best of them. Six trait and
#: trunk cells against seven models is enough forks to find something, and a
#: row that only appears when it wins is not evidence.
CORRECTION_MODELS = (
    "step0",
    "step0_z",
    *(f"step0_{name}" for name in decay.Z_COMPONENTS),
    "step0_b",
    "oracle",
)

#: The same rows for a bias table, minus the refit: least squares with a free
#: intercept leaves residuals summing to zero, so its bias is an identity
#: rather than a measurement (see :data:`METRICS`).
CORRECTION_BIAS_MODELS = tuple(m for m in CORRECTION_MODELS if m != "oracle")

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


def _gain_keys(rows: pd.DataFrame) -> list[tuple[str, str, int, str]]:
    """``(trait, trunk, t, probe)`` per row, typed so it keys a plain dict.

    The probe is part of the key because a gain is fitted with that probe held
    out, so two probes of one checkpoint are rescaled by two different numbers
    -- see :func:`_gain_forecast`.
    """
    return [
        (str(trait), str(trunk), int(t), str(probe))
        for trait, trunk, t, probe in rows[[*CHECKPOINT, "probe"]].itertuples(
            index=False
        )
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
    *,
    without_probe: str | None = None,
) -> pd.DataFrame:
    """The ideal gain per checkpoint, beside the state it might be read off.

    Public because it is the diagnostic behind every corrected forecaster: a
    correction can only work as well as ``gain`` is predictable from
    :data:`GAIN_FEATURES`, and a table of errors says whether it worked, not
    why.

    ``without_probe`` names the probe being held out. It does two things at
    once, which is the point: the gain is solved over every probe *but* that
    one, and it is solved against that probe's own base line -- the one fitted
    without it (:meth:`Baselines.line`). One argument, one fold, nothing about
    the held-out dataset anywhere in the fit.

    The default of ``None`` uses the whole scatter against the whole-fan line,
    which is what the diagnostic itself wants: the question it asks is what the
    right gain at a checkpoint was, not what a fold could have guessed. Against
    the whole-fan line a $t = 0$ gain of $1$ means "no drift, nothing to
    correct", which is the reading that makes a departure at $t > 0$ mean
    something.
    """
    panel = rows if without_probe is None else rows[rows["probe"] != without_probe]
    records = []
    for key, group in panel.groupby(list(CHECKPOINT), sort=True):
        trait, trunk, t = key
        fit = baselines.line(target, str(trait), without_probe)
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
) -> dict[tuple[str, str, int, str], float]:
    r"""Predicted gain per checkpoint *and probe*, fitted on neither of them.

    A corrected forecast has two ways of seeing the answer it is about to be
    scored on, so it is held out along both.

    *Leave one trunk out*, rather than one checkpoint. Consecutive checkpoints
    of one trunk are the same trajectory a step further along, so holding one
    out while its neighbours stay in the training set would let the correction
    be read off the trunk it is scored on. Holding out the whole trunk asks the
    question the table is for: does the state at a checkpoint predict the gain
    on a trajectory the correction has never seen?

    *Leave one probe out.* Every trunk fans out over the same probes, so a gain
    fitted on the other trunks alone has still been shown how probe $j$
    responds -- at other checkpoints, but to the same dataset -- and the
    correction it hands back for $j$ would be part memory of $j$ rather than a
    reading of the checkpoint. So the training checkpoints solve for their
    ideal gain over the *other* probes (``without_probe``), and the model that
    comes out is only ever used to predict the one that was dropped.

    That is one gain model per ``(trait, held-out trunk, held-out probe)``: a
    checkpoint's ``K`` corrected predictions come from ``K`` different models,
    each blind to the trajectory and the dataset it is scored on. The error
    summarised over that checkpoint is therefore a cross-validated one, which
    is a thing a caption has to say out loud.

    The ``K`` models are not near-copies of each other, and the reason is worth
    knowing before reading a correction table. :func:`_ideal_gain` is least
    squares on a rescaled slope, so a probe enters the gain weighted by
    $\Delta P_0^2$ -- and one probe with several times the projection
    difference of the rest carries most of the gain on its own. Holding *that*
    probe out leaves the remaining panel estimating a visibly different number,
    and the correction it hands back for the dropped probe is the honest one:
    what the checkpoint's other probes say, extrapolated to a leverage they
    never covered.

    Fitted within a trait, since the gain is a ratio of slopes measured against
    that trait's own persona vector and judge.
    """
    forecast: dict[tuple[str, str, int, str], float] = {}
    if rows.empty or "probe" not in rows:
        return forecast
    for probe in sorted(rows["probe"].dropna().unique()):
        frame = gain_frame(rows, column, baselines, target, without_probe=str(probe))
        if frame.empty:
            continue
        for _, block in frame.groupby("trait"):
            for trunk in sorted(block["trunk"].unique()):
                train = block[block["trunk"] != trunk].dropna(
                    subset=["gain", *features]
                )
                test = block[block["trunk"] == trunk]
                coefficients = _least_squares(_design(train, features), train["gain"])
                if coefficients is None:
                    continue
                for record, gain in zip(
                    test.itertuples(index=False), _design(test, features) @ coefficients
                ):
                    key = (
                        str(record.trait),
                        str(record.trunk),
                        int(record.t),
                        str(probe),
                    )
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
    baselines: Baselines | None = None,
) -> pd.DataFrame:
    r"""One row per ``(trunk, t, probe, series, model)``: a prediction and its truth.

    ``rows`` is :func:`method.visualization.decay.decay_frame` and
    ``validation`` is :func:`method.visualization.decay.validation_frame`; the
    second is what the step-0 forecasters are fitted on, and without it they
    are blank.  ``baselines`` may supply an explicitly different split for a
    specialised consumer; by default the table's leave-one-out fits are used.

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

    if baselines is None:
        baselines = baseline_fits(validation)
    if not baselines:
        logger.warning(
            "exp2: no validation fan on disk, so M_0's line cannot be fitted "
            "at all; only the refit-at-t forecaster will be scored"
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


def recalibration_frame(
    rows: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    probes: Sequence[str],
    series: str = "p0",
) -> pd.DataFrame:
    r"""The two lines for the recalibration grid on its non-probe split.

    The frozen prediction is one line per trait fitted on the validation
    datasets outside the explicitly supplied ``probes``.  Supplying the
    designed set rather than inferring it from measured rows ensures that an
    incomplete sweep cannot leak a missing probe into the fit.  The oracle
    remains the checkpoint-wise fit on the eight probe outcomes.  Table
    predictions do not pass through this function and retain their
    leave-one-out baselines.
    """
    baselines = nonprobe_baseline_fits(validation, probes)
    return prediction_frame(
        rows,
        validation,
        series=[series],
        models=RECALIBRATION_MODELS,
        baselines=baselines,
    )


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
