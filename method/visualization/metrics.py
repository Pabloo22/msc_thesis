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

    def predict(self, x: ArrayLike) -> NDArray[np.float64]:
        return self.slope * np.asarray(x, dtype=float) + self.intercept


def linear_fit(x: ArrayLike, y: ArrayLike) -> LinearFit:
    """Fit ``y ~ x`` and report $R^2$.

    Falls back to a flat line through the mean of ``y`` when there are fewer
    than two points or ``x`` is constant, so callers (including the demo on
    tiny synthetic inputs) never have to special-case degenerate data.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size < 2 or np.allclose(x_arr, x_arr[0]):
        mean_y = float(y_arr.mean()) if y_arr.size else 0.0
        return LinearFit(slope=0.0, intercept=mean_y, r2=0.0)
    slope, intercept = np.polyfit(x_arr, y_arr, deg=1)
    y_hat = slope * x_arr + intercept
    ss_res = float(np.sum((y_arr - y_hat) ** 2))
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return LinearFit(slope=float(slope), intercept=float(intercept), r2=r2)


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
