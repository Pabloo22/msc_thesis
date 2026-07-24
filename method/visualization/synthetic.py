r"""Fake but schema-faithful data, for exercising every figure without a GPU,
a judge model, or a real fine-tuning run.

Generators are deterministic given a seed. ``DeltaP`` statistics reuse
:func:`method.latent.summarize`, the exact function the real pipeline calls,
so a synthetic ``StepRecord.delta_p`` has precisely the keys a real one does.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

from method.latent import summarize
from method.visualization.schema import StepRecord, Trajectory

#: One instance of the 8-dataset design from "Running One Trajectory with
#: Multiple Seeds" (proposal, Section "Experiments and Plots").
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
    r"""A ``numpy`` RNG seed derived deterministically from string ``parts``.

    Unlike ``hash()``, this is stable across processes and Python versions
    (``PYTHONHASHSEED`` randomises string hashing), which matters for
    :func:`delta_p_0_for`: two trajectories that mention the same dataset must
    derive the exact same $\Delta P_0$ regardless of when or where they run,
    mirroring the real store's content-addressed reuse.
    """
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
    r"""One fake trajectory, shaped exactly like a real ``trajectory.json``.

    A toy random walk stands in for the real dynamics: each step nudges
    behaviour, the persona vector's rotation ($\rho$) and norm ($r$), and the
    projections $p$/$q$ toward the upcoming dataset's "pull" (misaligned
    datasets push up, normal ones pull back down), with per-seed noise. The
    result has the qualitative shape the proposal expects to test for --
    partial re-alignment, drifting projections, measurement noise -- without
    running any real fine-tuning.
    """
    rng = np.random.default_rng(seed * 1_000_000 + _stable_seed(name) % 1_000_000)
    behavior = baseline_behavior
    rho = 1.0
    r = float(abs(rng.normal(29.5, 0.5)))
    q = 0.0

    steps: list[StepRecord] = []
    for t in range(len(datasets) + 1):
        z = {"base": {"p": q * rho, "q": q, "rho": rho, "r": r}}
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
            # exactly) Delta P_t's expected value *is* Delta P_0 -- only
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
            q = q + pull * rng.uniform(0.5, 1.5)

    return Trajectory(name=name, trait=trait, seed=seed, steps=tuple(steps))


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


#: Fixed display order for the RQ2 "diversity" bar chart (proposal Section
#: "Does Data Diversity Hinder Emergent Realignment or Favor EM?"). Condition
#: strings match the ones the real exp4 builder writes (see
#: :mod:`method.visualization.labels`), so synthetic and real frames are
#: interchangeable.
DIVERSITY_CONDITIONS = ("baseline", "same2", "diff2", "same3", "diff3")
DIVERSITY_LABELS = {
    "baseline": "1 dataset",
    "same2": r"Same $\times$2",
    "diff2": r"Diverse $\times$2",
    "same3": r"Same $\times$3",
    "diff3": r"Diverse $\times$3",
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
    hysteresis_factor: float = 1.35,
) -> pd.DataFrame:
    """Fake data for the RQ2 hysteresis bar chart ("Is a model trained on
    trait-eliciting data more prone to EM?").

    ``after_realignment`` is drawn with a larger mean than ``fresh``, standing
    in for the hypothesis that a model already exposed to trait-eliciting
    data and realigned is easier to re-misalign than a fresh one.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for dataset in datasets:
        base = abs(rng.normal(18, 4))
        for s in range(n_seeds):
            rows.append(
                {
                    "dataset": dataset,
                    "condition": "fresh",
                    "seed": s,
                    "delta_behavior": base + rng.normal(0, 3),
                }
            )
            rows.append(
                {
                    "dataset": dataset,
                    "condition": "after_realignment",
                    "seed": s,
                    "delta_behavior": base * hysteresis_factor + rng.normal(0, 3),
                }
            )
    return pd.DataFrame(rows)


def synthetic_diversity_frame(*, n_seeds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Fake data for the RQ2 diversity bar chart ("Does Data Diversity Hinder
    Emergent Realignment or Favor EM?").

    Encodes the hypothesis under test: repeating *diverse* trait-eliciting
    datasets leaves more residual misalignment after realignment than
    repeating the *same* one, and both grow with dataset count.
    """
    rng = np.random.default_rng(seed)
    condition_means = {
        "baseline": 15.0,
        "same2": 18.0,
        "diff2": 24.0,
        "same3": 21.0,
        "diff3": 30.0,
    }
    rows = [
        {"condition": condition, "seed": s, "delta_behavior": mean + rng.normal(0, 3.5)}
        for condition, mean in condition_means.items()
        for s in range(n_seeds)
    ]
    return pd.DataFrame(rows)
