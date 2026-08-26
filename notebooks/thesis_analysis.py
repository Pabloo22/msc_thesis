"""Loading and statistics shared by the exp2 re-analysis notebooks.

Everything here reads artifacts that already exist under ``trajectories/``.
Nothing trains, generates, or calls a judge.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from method import experiments
from method.visualization import decay
from method.visualization.collect import collect_group

# The four measured projection series, ordered by how much of ΔP is recomputed at
# the current checkpoint. `hat` in a column name means the predicted term is still
# the base model's cached answer, so `delta_p_hat_t` is ΔP̂ₜ and the fully
# refreshed ΔPₜ is `delta_p_full_t`. No name is a prefix of another and none has
# changed meaning, so a stale reference raises rather than reading the wrong
# series.
PROJECTION_SERIES = {
    "delta_p_0": "ΔP₀  (nothing refreshed)",
    "delta_p_hat_v0": "ΔP̂ₜ(v₀)  (encoder)",
    "delta_p_hat_t": "ΔP̂ₜ  (encoder + axis)",
    "delta_p_full_t": "ΔPₜ  (encoder + axis + answer)",
}

# Consecutive pairs of the above: each swaps exactly one ingredient of ΔP from the
# base model to the current checkpoint.
REFRESH_STEPS = (
    ("encoder", "delta_p_0", "delta_p_hat_v0"),
    ("axis", "delta_p_hat_v0", "delta_p_hat_t"),
    ("answer", "delta_p_hat_t", "delta_p_full_t"),
)

CHECKPOINT_KEYS = ["trait", "trunk", "t"]

TRUNK_NAMES = {
    "a": "A: alternating Mistake II",
    "b": "B: Mistake I drivers",
    "c": "C: benign control",
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_checkpoint_probe_pairs() -> pd.DataFrame:
    """One row per (trait, trunk, checkpoint, probe dataset).

    The two remeasurement families have to be passed in explicitly or the
    ``delta_p_hat_v0`` and ``delta_p_full_t`` columns come back empty --
    ``decay.decay_frame`` called without them gives the hatted series only.
    """
    remeasured = [
        collect_group(experiments.EXP2_AXIS),
        collect_group(experiments.EXP2_REGEN),
    ]
    return decay.decay_frame(
        collect_group("exp2_decay"), collect_group("exp2_validation"), remeasured
    )


def load_validation_fan() -> pd.DataFrame:
    """The 24 candidate datasets each applied once to the untouched base model."""
    records = []
    for run in collect_group("exp2_validation").runs:
        measured = run.trajectory.steps[0].delta_p
        records.append(
            {
                "trait": run.trait,
                "dataset": run.label("dataset"),
                "delta_p_0": measured["mean"] if isinstance(measured, dict) else measured.mean,
                "set_point": run.final_behavior(),
            }
        )
    return pd.DataFrame(records).sort_values(["trait", "set_point"], ascending=[True, False])


def set_point_lookup(validation_fan: pd.DataFrame) -> dict[tuple[str, str], float]:
    keys = zip(validation_fan.trait, validation_fan.dataset)
    return dict(zip(keys, validation_fan.set_point))


def attach_set_points(pairs: pd.DataFrame, lookup: Mapping) -> pd.DataFrame:
    frame = pairs.copy()
    frame["set_point"] = [lookup.get(key, np.nan) for key in zip(frame.trait, frame.probe)]
    return frame.dropna(subset=["set_point"])


# --------------------------------------------------------------------------
# Correlations across the eight probes within one checkpoint
# --------------------------------------------------------------------------


def correlation(frame: pd.DataFrame, column: str, target: str = "delta_b") -> float:
    return float(np.corrcoef(frame[column], frame[target])[0, 1])


def per_checkpoint_correlations(
    pairs: pd.DataFrame, columns: Iterable[str] = tuple(PROJECTION_SERIES)
) -> pd.DataFrame:
    records = []
    for (trait, trunk, t), group in pairs.groupby(CHECKPOINT_KEYS):
        records.append(
            {
                "trait": trait,
                "trunk": trunk,
                "t": t,
                **{column: correlation(group, column) for column in columns},
            }
        )
    return pd.DataFrame(records)


def base_model_correlations(pairs: pd.DataFrame) -> pd.Series:
    """r at t=0, where all four series coincide. Decay is measured down from here."""
    at_zero = per_checkpoint_correlations(pairs[pairs.t == 0])
    return at_zero.groupby("trait")["delta_p_0"].first()


def refresh_increments(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per checkpoint: what each single-ingredient swap does to the correlation.

    ``r_drop_from_base`` is the trait's t=0 correlation minus the frozen ΔP₀
    correlation at this checkpoint, so it is positive when ΔP₀ ranks the eight
    probes worse than it did on the untouched base model, and negative on the few
    checkpoints where it ranks them better. It measures the predictor losing
    accuracy, not harm done to the model. The swap columns added below use the
    opposite convention: they are ``after - before``, so positive means the swap
    helped.
    """
    baseline = base_model_correlations(pairs)
    frame = per_checkpoint_correlations(pairs).query("t > 0").copy()
    frame["r_drop_from_base"] = frame.trait.map(baseline) - frame.delta_p_0
    for name, before, after in REFRESH_STEPS:
        frame[name] = frame[after] - frame[before]
    return frame


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------


def sign_test(values: Sequence[float]) -> dict:
    positive = int((np.asarray(values) > 0).sum())
    total = len(values)
    return {
        "helps": positive,
        "of": total,
        "median": float(np.median(values)),
        "p": float(stats.binomtest(positive, total, 0.5).pvalue),
    }


def paired_probe_bootstrap(
    pairs: pd.DataFrame,
    before: str,
    after: str,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% interval on the mean increment, resampling probes rather than checkpoints.

    The same eight probes recur at every checkpoint, so they are the unit that
    repeats; one draw is applied to all six checkpoints of a trunk. Pairing both
    correlations on the same draw removes the variance they share.
    """
    rng = np.random.default_rng(seed)
    probes = sorted(pairs.probe.unique())
    by_checkpoint = [group.set_index("probe").loc[probes] for _, group in pairs.groupby("t")]

    means = []
    for _ in range(n_resamples):
        draw = rng.integers(0, len(probes), len(probes))
        increments = [
            correlation(resampled, after) - correlation(resampled, before)
            for resampled in (checkpoint.iloc[draw] for checkpoint in by_checkpoint)
            if resampled.delta_b.std() > 0
        ]
        if increments:
            means.append(np.mean(increments))
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def robustness_of_scaling(
    frame: pd.DataFrame, column: str, against: str = "r_drop_from_base"
) -> dict:
    """Does ``column``'s benefit grow with ``against``, and does that survive perturbation?"""
    without_extreme = frame.drop(frame[against].idxmax())
    by_trait = {
        trait: float(np.corrcoef(group[against], group[column])[0, 1])
        for trait, group in frame.groupby("trait")
    }
    return {
        "pearson": float(np.corrcoef(frame[against], frame[column])[0, 1]),
        "spearman": float(stats.spearmanr(frame[against], frame[column]).statistic),
        "drop_worst_checkpoint": float(
            np.corrcoef(without_extreme[against], without_extreme[column])[0, 1]
        ),
        **{f"{trait}_only": value for trait, value in by_trait.items()},
    }


# --------------------------------------------------------------------------
# Predicting the next checkpoint
# --------------------------------------------------------------------------


def one_hot(series: pd.Series) -> np.ndarray:
    return pd.get_dummies(series, drop_first=True).to_numpy(float)


def least_squares_fit(predictors: Sequence, target: np.ndarray) -> dict:
    design = np.column_stack([*predictors, np.ones(len(target))])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    return {
        "R2": float(1 - (residual**2).sum() / ((target - target.mean()) ** 2).sum()),
        "RMSE": float(np.sqrt((residual**2).mean())),
        "coefficients": coefficients,
    }


def compare_models(frame: pd.DataFrame, models: Mapping[str, Sequence], target: str) -> pd.DataFrame:
    values = frame[target].to_numpy()
    return pd.DataFrame(
        [
            {"model": name, **{k: v for k, v in least_squares_fit(p, values).items() if k != "coefficients"}}
            for name, p in models.items()
        ]
    ).set_index("model")


def compare_models_by_trait(
    frame: pd.DataFrame, build_models, target: str = "b_next"
) -> pd.DataFrame:
    per_trait = {
        trait: compare_models(group, build_models(group), target)
        for trait, group in frame.groupby("trait")
    }
    return pd.concat(per_trait, axis=1)


# --------------------------------------------------------------------------
# Decision quality
# --------------------------------------------------------------------------


def direction_accuracy(
    frame: pd.DataFrame, column: str, threshold: float, minimum_change: float = 2.0
) -> float:
    """Fraction of probes whose direction of behaviour change the rule gets right."""
    moved = frame[frame.delta_b.abs() > minimum_change]
    return float((np.sign(moved[column] - threshold) == np.sign(moved.delta_b)).mean())


def best_fixed_threshold(
    frame: pd.DataFrame, column: str, minimum_change: float = 2.0
) -> tuple[float, float]:
    """Best achievable accuracy for a constant threshold.

    Chosen on the same rows it is then scored on, so it is a generous upper bound
    on what a real fixed-threshold monitor could do, not an estimate of one.

    Candidates are the midpoints between adjacent observed values, plus one point
    beyond each end. A threshold placed *at* an observed value would leave that row
    sitting exactly on it, where ``np.sign`` returns 0 and the row scores wrong
    however it actually moved -- which understates every candidate, and most the
    ones where several rows are tied.
    """
    moved = frame[frame.delta_b.abs() > minimum_change]
    observed = np.unique(moved[column])
    candidates = np.concatenate(
        [observed[:1] - 1.0, (observed[:-1] + observed[1:]) / 2.0, observed[-1:] + 1.0]
    )
    scored = [
        (direction_accuracy(frame, column, threshold, minimum_change), float(threshold))
        for threshold in candidates
    ]
    return max(scored)


def calibrate_set_point_from_projection(validation_fan: pd.DataFrame) -> dict[str, np.ndarray]:
    """b*(D) ≈ slope·ΔP₀(D) + intercept, fitted per trait on the 24 base-model runs."""
    return {
        trait: np.polyfit(group.delta_p_0, group.set_point, 1)
        for trait, group in validation_fan.groupby("trait")
    }


def predicted_set_point(frame: pd.DataFrame, calibration: Mapping[str, np.ndarray]) -> np.ndarray:
    slopes = frame.trait.map(lambda trait: calibration[trait][0])
    intercepts = frame.trait.map(lambda trait: calibration[trait][1])
    return slopes * frame.delta_p_0 + intercepts
