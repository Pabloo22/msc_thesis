r"""Pure array maths shared by the figures: regression, percentage drift, and
stacking ragged per-seed series into a matrix.

Deliberately free of matplotlib, pandas, or :mod:`method.visualization.schema`
so it is trivial to unit test and to reuse once real trajectories exist.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class LinearFit:
    """An ordinary least-squares fit of ``y ~ x``, plus its goodness of fit."""

    slope: float
    intercept: float
    r2: float

    @property
    def corr(self) -> float:
        r"""Pearson $r$ between ``x`` and ``y``, signed by the slope.

        A derived view of :attr:`r2` rather than a second statistic: for a
        simple least-squares fit $r^2 = R^2$ exactly, so the correlation needs
        no access to the points the fit was made from. It is what the figures
        annotate where a reader has to compare panels by eye -- it keeps the
        direction of the relationship visible, and it is linear in the strength
        of it where $R^2$ is quadratic.
        """
        return float(np.copysign(np.sqrt(self.r2), self.slope))

    def predict(self, x: ArrayLike) -> NDArray[np.float64]:
        return self.slope * np.asarray(x, dtype=float) + self.intercept


def linear_fit(x: ArrayLike, y: ArrayLike) -> LinearFit:
    """Fit ``y ~ x`` and report $R^2$.

    Falls back to a flat line through the mean of ``y`` when there are fewer
    than two points or ``x`` is constant, so callers (including the demo on
    tiny synthetic inputs) never have to special-case degenerate data.

    Pairs where either coordinate is not finite are dropped first. NaN means
    "not measured" throughout the analysis -- a checkpoint whose ``h_norm``
    predates the field, a $z_t$ source a run does not carry -- and a column
    that is entirely unmeasured has to come back degenerate like any other
    column with no points. Left in, it reaches ``np.polyfit``, which raises
    ``LinAlgError`` rather than returning anything a caller could recognise as
    absent.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    usable = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not usable.all():
        x_arr, y_arr = x_arr[usable], y_arr[usable]
    if x_arr.size < 2 or np.allclose(x_arr, x_arr[0]):
        mean_y = float(y_arr.mean()) if y_arr.size else 0.0
        return LinearFit(slope=0.0, intercept=mean_y, r2=0.0)
    slope, intercept = np.polyfit(x_arr, y_arr, deg=1)
    y_hat = slope * x_arr + intercept
    ss_res = float(np.sum((y_arr - y_hat) ** 2))
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return LinearFit(slope=float(slope), intercept=float(intercept), r2=r2)


@dataclass(frozen=True)
class FitInterval:
    """A fit plus a bootstrap confidence interval on its slope and correlation."""

    fit: LinearFit
    corr_lo: float
    corr_hi: float
    slope_lo: float
    slope_hi: float
    #: Resamples that produced a usable fit. Below ``n_resamples`` when some
    #: draw was degenerate; zero means the interval is NaN, not tight.
    n_usable: int


def bootstrap_fit(
    x: ArrayLike,
    y: ArrayLike,
    *,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> FitInterval:
    r"""Fit ``y ~ x`` and bootstrap the uncertainty in its slope and correlation.

    Points are resampled with replacement as *units*, which for the RQ1 decay
    curves means resampling the probe datasets: a checkpoint's correlation
    rests on eight of them, and the question the interval answers is how much
    of it is the particular eight that were chosen (the "probe-set sampling"
    row of the noise budget).

    Percentile interval rather than normal-theory, because $r$ lives on
    $[-1, 1]$ and its sampling distribution against eight points is neither
    symmetric nor anywhere near Gaussian near the ends.

    The correlation is taken per resample rather than by transforming an
    interval on $R^2$. The two agree only while every resample fits the same
    sign: a percentile interval is equivariant under a *monotone* transform,
    and $r \mapsto r^2$ stops being one as soon as some draw slopes the other
    way -- exactly the case where the sign is the thing worth reporting.

    Resamples where every drawn ``x`` (or every drawn ``y``) is identical are
    dropped rather than scored: a duplicated point carries no fit, and counting
    it as $r = 0$ would pull the interval toward zero by an artifact of the
    resampling. :attr:`FitInterval.n_usable` reports how many survived.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    fit = linear_fit(x_arr, y_arr)
    if x_arr.size < 2 or n_resamples < 1:
        return FitInterval(fit, np.nan, np.nan, np.nan, np.nan, 0)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x_arr.size, size=(n_resamples, x_arr.size))
    xs, ys = x_arr[idx], y_arr[idx]
    dx = xs - xs.mean(axis=1, keepdims=True)
    dy = ys - ys.mean(axis=1, keepdims=True)
    sxx = (dx * dx).sum(axis=1)
    syy = (dy * dy).sum(axis=1)
    sxy = (dx * dy).sum(axis=1)

    usable = (sxx > 0) & (syy > 0)
    slopes = np.where(usable, sxy / np.where(usable, sxx, 1.0), np.nan)
    corrs = np.where(usable, sxy / np.sqrt(np.where(usable, sxx * syy, 1.0)), np.nan)
    if not usable.any():
        return FitInterval(fit, np.nan, np.nan, np.nan, np.nan, 0)

    tail = 100 * (1 - level) / 2
    corr_lo, corr_hi = np.nanpercentile(corrs, [tail, 100 - tail])
    slope_lo, slope_hi = np.nanpercentile(slopes, [tail, 100 - tail])
    return FitInterval(
        fit=fit,
        corr_lo=float(corr_lo),
        corr_hi=float(corr_hi),
        slope_lo=float(slope_lo),
        slope_hi=float(slope_hi),
        n_usable=int(usable.sum()),
    )


def stack_and_trim(series: Sequence[Sequence[float]]) -> NDArray[np.float64]:
    """Stack ragged per-seed series into a ``[n_seeds, n_steps]`` matrix.

    Trims every row to the shortest one, so seeds that ran a few steps short
    (or synthetic fixtures of uneven length) can still be plotted together.
    Raises ``ValueError`` on empty input -- there is nothing to plot.
    """
    if not series:
        raise ValueError("no series to stack")
    min_len = min(len(row) for row in series)
    if min_len == 0:
        raise ValueError("at least one series has zero length")
    return np.array([row[:min_len] for row in series], dtype=float)


def percent_of_baseline(values: ArrayLike, *, axis: int = -1) -> NDArray[np.float64]:
    """Express ``values`` as a percentage of their own first entry along ``axis``.

    100 means "unchanged from step 0". Accepts a 1-D single run or a 2-D
    ``[n_seeds, n_steps]`` matrix, in which case each row normalises against
    its own baseline independently. A near-zero baseline yields NaN rather
    than a division blow-up.
    """
    arr = np.asarray(values, dtype=float)
    baseline = np.take(arr, 0, axis=axis)
    baseline = np.where(np.abs(baseline) < 1e-12, np.nan, baseline)
    return arr / np.expand_dims(baseline, axis) * 100.0


def ratio_percent(numerator: ArrayLike, denominator: ArrayLike) -> NDArray[np.float64]:
    """``numerator / denominator * 100``, elementwise, with NaN for a zero divisor."""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    den = np.where(np.abs(den) < 1e-12, np.nan, den)
    return num / den * 100.0


def mean_std(
    values: ArrayLike, *, axis: int = 0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mean and population std along ``axis`` (default: across seeds, axis 0)."""
    arr = np.asarray(values, dtype=float)
    return arr.mean(axis=axis), arr.std(axis=axis)


def behavior_deltas(values: ArrayLike) -> NDArray[np.float64]:
    """``b_{t+1} - b_t`` for a behaviour series; one entry shorter than the input."""
    return np.diff(np.asarray(values, dtype=float))
