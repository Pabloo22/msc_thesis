"""Regression test: DeltaP must be keyed by the dataset it describes.

DeltaP is an *action* feature -- it describes the update a checkpoint is about
to receive. But it is stored under the checkpoint's ``weights_id``, which
encodes only the steps already taken. Two trajectories that share a prefix and
then diverge therefore collide on every DeltaP artifact, and the second one
silently reads the first one's numbers.
"""

from __future__ import annotations

import dataclasses

from method import experiments as E, steps
from method.backends import get_backend
from method.config import (
    Backend,
    DatasetVersion,
    DeltaPConfig,
    DeltaPMode,
    StepConfig,
)
from method.store import Store, get_weights_id


def _cfg(dataset: str, version: DatasetVersion):
    """A 1-step mock-scale config training on ``dataset``."""
    step = StepConfig(
        dataset=dataset, version=version, n_examples=64, train=E.LOCAL_TRAIN
    )
    return dataclasses.replace(E.SMOKE_MOCK, steps=(step,))


def test_delta_p_differs_between_datasets_at_the_same_checkpoint(tmp_path):
    store = Store(tmp_path)
    backend = get_backend(Backend.MOCK, dtype="float16")

    a = _cfg("mistake_gsm8k", DatasetVersion.MISALIGNED_2)
    b = _cfg("evil", DatasetVersion.MISALIGNED_2)

    # Same base model and seed, so both trajectories start from the same
    # checkpoint id -- this is intended, it is what shares the base model's
    # measurements across every experiment.
    assert get_weights_id(a, 0) == get_weights_id(b, 0)

    results = {}
    for name, cfg in (("a", a), ("b", b)):
        steps.extract_persona_vector(cfg, 0, store, backend)
        step = cfg.steps[0]
        train_file = steps.sample_training_file(
            step, cfg.seed, tmp_path / f"train_{name}.jsonl", store
        )
        results[name] = steps.compute_delta_p(
            cfg, 0, store, backend, train_file, steps.training_sample_id(step, cfg.seed)
        )

    # The two steps train on entirely different datasets, so the projection
    # difference they demand cannot legitimately be identical.
    assert results["a"] != results["b"], (
        "DeltaP for two different datasets came out identical at the same "
        "checkpoint: the second run reused the first's cached artifact."
    )


def test_delta_p_subsample_is_keyed_by_n_samples(tmp_path):
    """SAMPLE mode's cached input file must be keyed by ``n_samples`` too.

    Keying it by the training-sample id alone let a run with a different
    ``n_samples`` silently reuse the old subsample: the artifact named
    ``...-sub4`` would then really contain 8-sample statistics.
    """
    store = Store(tmp_path)
    backend = get_backend(Backend.MOCK, dtype="float16")
    base = _cfg("mistake_gsm8k", DatasetVersion.MISALIGNED_2)

    for n_samples in (8, 4):
        cfg = dataclasses.replace(
            base, delta_p=DeltaPConfig(mode=DeltaPMode.SAMPLE, n_samples=n_samples)
        )
        steps.extract_persona_vector(cfg, 0, store, backend)
        step = cfg.steps[0]
        train_file = steps.sample_training_file(
            step, cfg.seed, tmp_path / f"train_{n_samples}.jsonl", store
        )
        stats = steps.compute_delta_p(
            cfg, 0, store, backend, train_file,
            steps.training_sample_id(step, cfg.seed),
        )
        assert stats["n"] == n_samples
