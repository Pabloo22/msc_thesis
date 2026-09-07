r"""Score frozen and recalibrated predictors at later checkpoints."""

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

#: Change and level targets from ``decay_frame``.
CHANGE, LEVEL = "delta_b", "b_next"
TARGETS = (CHANGE, LEVEL)

#: Common scoring target for all forecasters.
SCORED_ON = LEVEL

#: Columns identifying one checkpoint scatter.
CHECKPOINT = ("trait", "trunk", "t")

#: Plain-language labels for latent coordinates.
Z_GLOSSES = {
    "p": "the neutral state's alignment with the base persona axis",
    "q": "the neutral state's alignment with the current persona axis",
    "rho": "how far the persona axis has rotated",
    "r": "the current persona vector's length",
}

#: State variables available to the recalibration model.
GAIN_FEATURES = (*decay.Z_COMPONENTS, "b_t")

#: Error metrics reported per checkpoint scatter.
METRICS = ("rmse", "mae", "bias")


#: Sentinel for the full validation fan.
WHOLE_FAN = None

#: Points a fold needs before :func:`method.visualization.metrics.linear_fit`
#: reports a line rather than a flat one through the mean of $y$.
MIN_FIT_POINTS = 2


def dataset_family(dataset_id: str) -> str:
    """Return the corpus family prefix of a dataset ID."""
    family, _, _ = dataset_id.partition("/")
    return family


@dataclass(frozen=True)
class Baselines:
    r"""$M_0$'s lines, one per (target, trait, held-out dataset family).

    A mapping rather than a single line per trait because every line here is
    fitted leave-one-family-out: see :func:`baseline_fits` for why.
    """

    #: Keyed ``(target, trait, held-out family)``, with :data:`WHOLE_FAN`
    #: standing for the fold that holds nothing out. The third slot is a
    #: *family* (:func:`dataset_family`) rather than a dataset id, since all
    #: three versions of a family share one fold; :meth:`line` takes the id and
    #: does the collapsing, so no caller has to.
    fits: Mapping[tuple[str, str, str | None], LinearFit]

    def line(self, target: str, trait: str, without: str | None) -> LinearFit | None:
        """The line for predicting ``without``, or ``None`` if there is none.

        ``without`` is a dataset id, and the fold it selects is its whole
        family's, so the held-out dataset's two sibling versions are out of the
        line as well as the dataset itself.

        Falls back to the whole-fan line where that family has no fold of its
        own, which happens when the fan covered no version of it. Nothing
        leaks: a family the fan did not measure is in no fit to begin with.
        """
        family = None if without is None else dataset_family(without)
        fit = self.fits.get((target, trait, family))
        if fit is not None:
            return fit
        return self.fits.get((target, trait, WHOLE_FAN))

    def __bool__(self) -> bool:
        return bool(self.fits)


def baseline_fits(validation: pd.DataFrame) -> Baselines:
    r"""Fit per-trait $M_0$ lines with leave-one-family-out folds."""
    fits: dict[tuple[str, str, str | None], LinearFit] = {}
    if validation.empty:
        return Baselines(fits)
    for trait, group in validation.groupby("trait"):
        families = group["dataset"].astype(str).map(dataset_family)
        # Reject traits whose smallest fold cannot support a fit.
        smallest_fold = len(group) - int(families.value_counts().max())
        if smallest_fold < MIN_FIT_POINTS:
            logger.warning(
                "exp2/%s: %d validation dataset(s) on disk across %d "
                "dataset famil(y/ies), which leaves %d once a held-out "
                "dataset's whole family is dropped -- too few to fit M_0's "
                "line; the step-0 forecasts for this trait will be blank",
                trait,
                len(group),
                families.nunique(),
                smallest_fold,
            )
            continue
        for held in (WHOLE_FAN, *sorted(families.unique())):
            panel = group if held is WHOLE_FAN else group[families != held]
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
    :func:`baseline_fits`, which makes one leave-one-family-out fold per scored
    probe, this makes one common fit from the validation datasets outside the
    probe set.  All plotted probe predictions therefore lie on one affine line.

    **Held out by id, not by family, and it has to be.** The eight probes sit
    in eight distinct families, so dropping each probe's whole family would
    drop all 24 datasets and leave nothing to fit. The 16 datasets this does
    fit on therefore include the probes' sibling versions, which makes the
    drawn line slightly kinder to the probes than the scores beside it. That is
    tolerable only because this line is drawn and never scored: every number
    reported anywhere comes from :func:`baseline_fits`.

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


#: What a forecaster does: turn one series' column, over the rows it was measured on, into a predicted $b_{t+1}$ aligned to those rows.
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


#: Recalibration states, including latent coordinates and behaviour level.
CORRECTION_STATES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("z", z_symbol(), "the latent state", decay.Z_COMPONENTS),
    *(
        (name, z_component_symbol(name), Z_GLOSSES[name], (name,))
        for name in decay.Z_COMPONENTS
    ),
    ("b", "b_t", "the behaviour level", ("b_t",)),
)


def _correction_pair(
    name: str, symbol: str, gloss: str, features: tuple[str, ...]
) -> tuple[Forecaster, Forecaster]:
    r"""One correction registered under both of $f_0$'s targets.

    A correction rescales the projection fed to $f_0$; it does not change what
    $f_0$ was fitted to predict, and that choice is made per projection rather
    than once for the study (see :data:`MATCHED_TARGET_MODELS`). Registering
    the pair is what lets a table apply the same matched-target rule to a
    corrected forecast as to an uncorrected one, instead of comparing a
    correction fitted on $\Delta b$ against a baseline fitted on $b_{t+1}$.
    """
    return (
        Forecaster(
            f"step0_{name}",
            rf"$f_0$ $\times\, c_t({symbol})$",
            f"rescaled by {gloss}",
            _corrected(features, CHANGE),
            target=CHANGE,
        ),
        Forecaster(
            f"step0_{name}_level",
            rf"$f_0$ $\times\, c_t({symbol})$, on $b_{{t+1}}$",
            f"rescaled by {gloss}, predicting the level",
            _corrected(features, LEVEL),
            target=LEVEL,
        ),
    )


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
    *(
        forecaster
        for name, symbol, gloss, features in CORRECTION_STATES
        for forecaster in _correction_pair(name, symbol, gloss, features)
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
    for name in ("z", *decay.Z_COMPONENTS):
        symbol = (
            z_symbol(neutral=index)
            if name == "z"
            else z_component_symbol(name, neutral=index)
        )
        labels[f"step0_{name}"] = rf"$f_0$ $\times\, c_t({symbol})$"
        labels[f"step0_{name}_level"] = (
            rf"$f_0$ $\times\, c_t({symbol})$, on $b_{{t+1}}$"
        )
    return labels

#: The pair the headline comparison is between: what a practitioner can have,
#: against what they would have if refitting were free.
HEADLINE_MODELS = ("step0", "oracle")

#: The two targets of the frozen line, with the refit as their shared
#: reference. The refit appears once, not twice, because it is target-invariant
#: (see :attr:`Forecaster.target`).
TARGET_MODELS = ("step0", "step0_level", "oracle")

#: The pre-specified target for each projection variant in the RMSE headline.
HEADLINE_MODEL_BY_SERIES = {
    series: model
    for _, _, members in decay.REFRESH_GROUPS
    for series, model in zip(members, ("step0_level", "step0"), strict=True)
}

#: Target matched to each projection series.
MATCHED_TARGET_BY_SERIES = {
    "p0": LEVEL,
    **{
        series: target
        for _, _, members in decay.REFRESH_GROUPS
        for series, target in zip(members, (LEVEL, CHANGE), strict=True)
    },
}


def matched_model(model: str, series: str) -> str:
    """``model`` as fitted under the target ``series`` takes.

    The refit is returned unchanged: within a checkpoint $b_t$ is one constant,
    so refitting on $b_{t+1}$ rather than on $\\Delta b$ moves the intercept by
    exactly that constant and leaves the predicted level identical.
    """
    if model == "oracle" or MATCHED_TARGET_BY_SERIES.get(series, CHANGE) == CHANGE:
        return model
    return f"{model}_level"


#: The forecasters whose bias is a measurement rather than an identity: the
#: ones carrying $M_0$'s intercept forward (see :data:`METRICS`).
BIASED_MODELS = tuple(f.name for f in FORECASTERS if not f.refits)

#: Forecasters compared in recalibration tables.
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

#: Models shown in predicted-versus-actual grids.
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
    r"""The ideal gain per checkpoint, beside the state it might be read off."""
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
    r"""Predicted gain per checkpoint *and probe*, fitted on neither of them."""
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
    r"""One row per ``(trunk, t, probe, series, model)``: a prediction and its truth."""
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


def metric_frame(
    scores: pd.DataFrame,
    *,
    metric: str = "rmse",
    model: str = "step0",
    model_by_series: Mapping[str, str] | None = None,
    series: Sequence[str] | None = None,
) -> pd.DataFrame:
    r"""Put one forecast metric into wide ``<metric>_<series>`` columns.

    This is the checkpoint-level shape consumed by the headline curves.  The
    By default ``model`` is held fixed. ``model_by_series`` instead selects a
    pre-specified forecaster for each series; this lets variants with different
    semantic targets be compared without spending another visual channel.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
    if scores.empty:
        return pd.DataFrame()
    wanted = list(series) if series is not None else list(decay.SERIES)
    candidates = scores[scores["series"].isin(wanted)]
    if model_by_series is None:
        kept = candidates[candidates["model"].eq(model)]
    else:
        selected = candidates["series"].map(model_by_series)
        kept = candidates[candidates["model"].eq(selected)]
    if kept.empty:
        return pd.DataFrame()
    wide = kept.pivot(
        index=["trait", "trunk", "t"], columns="series", values=metric
    ).reset_index()
    wide.columns.name = None
    return wide.rename(
        columns={name: f"{metric}_{name}" for name in wanted if name in wide}
    )


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
