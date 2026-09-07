r"""Deterministic, schema-faithful data for testing figures without a GPU."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

from method.latent import CONVENTION as Z_CONVENTION
from method.latent import summarize
from method.visualization import labels
from method.visualization.schema import StepRecord, Trajectory

#: Default eight-dataset synthetic trajectory.
DEFAULT_DATASETS = (
    "mistake_gsm8k/misaligned_2",
    "sycophancy/normal",
    "mistake_math/misaligned_2",
    "mistake_opinions/misaligned_1",
    "insecure_code/normal",
    "evil/normal",
    "hallucination/misaligned_1",
    "mistake_medical/normal",
)

_MISALIGNED_MARKERS = ("misaligned_1", "misaligned_2")


def _dataset_pull(dataset: str) -> float:
    """+1 for a misaligned dataset version, -1 for normal: which way it pushes."""
    return 1.0 if any(m in dataset for m in _MISALIGNED_MARKERS) else -1.0


def _stable_seed(*parts: str) -> int:
    r"""Derive a process-independent ``numpy`` seed from strings."""
    digest = hashlib.sha256(":".join(parts).encode()).hexdigest()
    return int(digest[:8], 16)


def delta_p_0_for(dataset: str, *, base_seed: int = 0) -> float:
    r"""A fixed, dataset-only-dependent stand-in for $\Delta P_0$.

    Computed "against $M_0$" once per dataset name -- any trajectory that
    later trains on ``dataset`` sees the same value, whatever step it occurs
    at, just as the real action encoder $\phi(\mathcal{D}_t; \mathcal{M}_0)$
    would produce a fixed number per dataset.
    """
    rng = np.random.default_rng(_stable_seed(str(base_seed), dataset))
    magnitude = abs(rng.normal(loc=1.2, scale=0.3))
    return float(_dataset_pull(dataset) * magnitude)


def synthetic_delta_p_0_lookup(
    datasets: Sequence[str] = DEFAULT_DATASETS,
) -> dict[str, float]:
    """``{dataset: delta_p_0}`` for every dataset in ``datasets``."""
    return {d: delta_p_0_for(d) for d in datasets}


def synthetic_trajectory(
    *,
    name: str = "synthetic",
    trait: str = "evil",
    seed: int = 0,
    datasets: Sequence[str] = DEFAULT_DATASETS,
    n_samples: int = 32,
    baseline_behavior: float = 30.0,
    behavior_scale: float = 25.0,
    noise: float = 0.08,
) -> Trajectory:
    r"""Build a schema-faithful random-walk trajectory."""
    rng = np.random.default_rng(seed * 1_000_000 + _stable_seed(name) % 1_000_000)
    behavior = baseline_behavior
    rho = 1.0
    r = float(abs(rng.normal(29.5, 0.5)))
    # Keep cosine-valued state within $[-1, 1]$.
    q = float(rng.normal(-0.22, 0.02))
    # Record the cosine normalizer to match real trajectory records.
    h_norm = float(rng.normal(63.7, 1.5))

    steps: list[StepRecord] = []
    for t in range(len(datasets) + 1):
        z = {"base": {"p": q * rho, "q": q, "rho": rho, "r": r, "h_norm": h_norm}}
        noisy_behavior = behavior + rng.normal(0, behavior_scale * noise)
        behavior_dict = {
            trait: float(np.clip(noisy_behavior, 0, 100)),
            f"{trait}_std": float(abs(rng.normal(20, 5))),
            "coherence": float(np.clip(rng.normal(75, 8), 0, 100)),
            "n": 20,
        }
        delta_p = None
        next_dataset = None
        if t < len(datasets):
            dataset = datasets[t]
            # The mean is tied to the fixed delta_p_0_for(dataset) baseline and
            # scaled by the checkpoint's current rho, so that at t=0 (rho=1.0
            # exactly) Delta hat P_t's expected value *is* Delta P_0 -- only
            # sampling noise separates them -- and it only drifts away as the
            # persona vector rotates.
            step_mean = delta_p_0_for(dataset) * rho
            samples = torch.tensor(
                rng.normal(step_mean, 1.5, size=n_samples), dtype=torch.float32
            )
            delta_p = summarize(samples)
            next_dataset = dataset

        steps.append(
            StepRecord(
                t=t,
                weights_id=f"synthetic-{name}-seed{seed}-t{t:02d}",
                behavior=behavior_dict,
                z=z,
                delta_p=delta_p,
                next_dataset=next_dataset,
            )
        )

        if t < len(datasets):
            pull = _dataset_pull(datasets[t])
            step_size = 0.5 + 0.5 * abs(rng.normal(1.0, 0.2))
            behavior = float(
                np.clip(behavior + pull * behavior_scale * step_size, 0, 100)
            )
            rho = float(np.clip(rho - abs(pull) * rng.uniform(0.02, 0.08), -1.0, 1.0))
            r = float(r * (1 + pull * rng.uniform(0.0, 0.03)))
            q = float(np.clip(q + pull * rng.uniform(0.01, 0.04), -1.0, 1.0))
            h_norm = float(h_norm * (1 + rng.uniform(0.002, 0.01)))

    return Trajectory(
        name=name,
        trait=trait,
        seed=seed,
        steps=tuple(steps),
        z_convention=Z_CONVENTION,
    )


def synthetic_trajectory_set(
    n_seeds: int = 5,
    *,
    name: str = "synthetic",
    trait: str = "evil",
    datasets: Sequence[str] = DEFAULT_DATASETS,
) -> list[Trajectory]:
    """``n_seeds`` independent runs of the same dataset sequence."""
    return [
        synthetic_trajectory(name=name, trait=trait, seed=seed, datasets=datasets)
        for seed in range(n_seeds)
    ]


#: Display order and labels for the RQ2 "hysteresis" bar chart, taken from the
#: real thing rather than restated: a fixture that names its own conditions
#: silently stops previewing the experiment the moment an arm is added or
#: renamed, which is exactly how the demo figure came to show fewer bars than
#: the design has.
HYSTERESIS_CONDITIONS = labels.HYSTERESIS_CONDITIONS
HYSTERESIS_LABELS = labels.HYSTERESIS_CONDITION_LABELS

#: Per-condition shape of the fake data: how many steps of prior training the
#: arm had, and how far its final step moves relative to a first exposure.
#: ``normal1``/``normal2`` move *less* (plasticity loss, worsening with the
#: number of prior steps); ``same``/``diff`` move *more* (the hysteresis
#: hypothesis) and start from a floor left behind by incomplete re-alignment.
_HYSTERESIS_ARMS = {
    "baseline": (0, 1.0),
    "normal1": (1, 0.85),
    "normal2": (2, 0.75),
    "same": (2, 1.35),
    "diff": (2, 1.25),
}


def synthetic_hysteresis_frame(
    datasets: Sequence[str] = (
        "hallucination/misaligned_1",
        "mistake_opinions/misaligned_1",
        "mistake_gsm8k/misaligned_2",
    ),
    *,
    n_seeds: int = 5,
    seed: int = 0,
    base_behavior: float = 8.0,
    realigned_floor: float = 12.0,
) -> pd.DataFrame:
    r"""Fake data for the RQ2 hysteresis bar chart ("Is a model trained on trait-eliciting data more prone to EM?")."""
    rng = np.random.default_rng(seed)
    rows = []
    for dataset in datasets:
        base = abs(rng.normal(18, 4))
        for s in range(n_seeds):
            for condition, (n_prior, factor) in _HYSTERESIS_ARMS.items():
                if condition in ("same", "diff"):
                    # Re-alignment after misalignment lands short of b_0.
                    before = realigned_floor
                elif n_prior:
                    # Training on normal data has nothing to pull back, so it
                    # leaves the model at (here, just under) b_0.
                    before = max(0.0, base_behavior - 3.0)
                else:
                    before = base_behavior
                delta = base * factor + rng.normal(0, 3)
                rows.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "seed": s,
                        "n_prior_steps": n_prior,
                        "behavior_base": base_behavior,
                        "behavior_before": before,
                        "behavior": before + delta,
                        "delta_behavior": delta,
                    }
                )
    return pd.DataFrame(rows)
