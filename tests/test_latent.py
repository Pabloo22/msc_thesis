"""Unit tests for the pure latent-state maths in :mod:`method.latent`.

``latent.py`` loads no model and spawns no subprocess, so it can be exercised
directly on hand-built tensors. These tests pin down the mathematical
properties the rest of the pipeline relies on: the projection's normalisation
convention, the meaning of each z_t component, and the DeltaP contract that the
target and predicted activations stay row-aligned.
"""

from __future__ import annotations

import math

import pytest
import torch

from method.latent import (
    LatentState,
    compute_latent,
    cosine,
    delta_projection,
    project,
    summarize,
)


class TestProject:
    def test_axis_aligned_value(self) -> None:
        # Onto a unit axis, the projection is just the matching coordinate.
        acts = torch.tensor([3.0, 4.0])
        assert project(acts, torch.tensor([1.0, 0.0])).item() == pytest.approx(3.0)
        assert project(acts, torch.tensor([0.0, 1.0])).item() == pytest.approx(4.0)

    def test_parallel_recovers_activation_norm(self) -> None:
        # Projecting a vector onto its own direction returns its length.
        acts = torch.tensor([3.0, 4.0])
        assert project(acts, acts).item() == pytest.approx(5.0)

    def test_orthogonal_is_zero(self) -> None:
        assert project(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 5.0])).item() == 0.0

    def test_invariant_to_vector_scaling(self) -> None:
        # Normalisation is by the direction's length, so rescaling the direction
        # must not change the projection.
        acts = torch.tensor([1.0, 2.0, -3.0])
        vector = torch.tensor([0.5, -1.0, 2.0])
        base = project(acts, vector)
        for scale in (0.1, 2.0, 100.0):
            torch.testing.assert_close(project(acts, vector * scale), base)

    def test_scales_with_activation_magnitude(self) -> None:
        # The activation is deliberately NOT normalised, so magnitude changes in
        # the activation show up linearly (the point of the a_proj_b convention).
        acts = torch.tensor([1.0, 2.0, -3.0])
        vector = torch.tensor([0.5, -1.0, 2.0])
        base = project(acts, vector)
        torch.testing.assert_close(project(acts * 3.0, vector), base * 3.0)

    def test_batched_reduces_last_dim(self) -> None:
        acts = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
        out = project(acts, torch.tensor([1.0, 0.0]))
        assert out.shape == (3,)
        torch.testing.assert_close(out, torch.tensor([1.0, 0.0, 3.0]))

    def test_accepts_non_float_inputs(self) -> None:
        # project() casts to float internally, so integer tensors must work.
        out = project(torch.tensor([3, 4]), torch.tensor([1, 0]))
        assert out.item() == pytest.approx(3.0)


class TestCosine:
    def test_parallel(self) -> None:
        v = torch.tensor([1.0, 2.0, 3.0])
        assert cosine(v, v) == pytest.approx(1.0)

    def test_antiparallel(self) -> None:
        v = torch.tensor([1.0, 2.0, 3.0])
        assert cosine(v, -v) == pytest.approx(-1.0)

    def test_orthogonal(self) -> None:
        assert cosine(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])) == 0.0

    def test_scale_invariant_in_both_args(self) -> None:
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([-2.0, 0.5, 1.0])
        base = cosine(a, b)
        assert cosine(a * 7.0, b) == pytest.approx(base)
        assert cosine(a, b * 0.01) == pytest.approx(base)

    def test_returns_python_float(self) -> None:
        assert isinstance(cosine(torch.tensor([1.0, 1.0]), torch.tensor([1.0, 0.0])), float)

    def test_known_angle(self) -> None:
        # 45 degrees between the two axes' diagonal and one axis.
        assert cosine(torch.tensor([1.0, 1.0]), torch.tensor([1.0, 0.0])) == pytest.approx(
            1.0 / math.sqrt(2.0)
        )


class TestComputeLatent:
    def test_components_match_their_definitions(self) -> None:
        v0 = torch.tensor([1.0, 0.0, 0.0])
        vt = torch.tensor([0.0, 2.0, 0.0])
        h = torch.tensor([3.0, 4.0, 0.0])
        z = compute_latent(v0, vt, h)

        assert z.p == pytest.approx(project(h, v0).item())  # drift on original axis
        assert z.q == pytest.approx(project(h, vt).item())  # alignment on current axis
        assert z.rho == pytest.approx(cosine(v0, vt))       # rotation of the vector
        assert z.r == pytest.approx(vt.norm().item())       # magnitude of the vector

    def test_hand_computed_values(self) -> None:
        v0 = torch.tensor([1.0, 0.0, 0.0])
        vt = torch.tensor([0.0, 2.0, 0.0])
        h = torch.tensor([3.0, 4.0, 0.0])
        z = compute_latent(v0, vt, h)
        assert z.p == pytest.approx(3.0)   # h onto x-axis
        assert z.q == pytest.approx(4.0)   # h onto y-axis (norm-2 vector cancels)
        assert z.rho == pytest.approx(0.0)  # x perpendicular to y
        assert z.r == pytest.approx(2.0)

    def test_no_rotation_gives_rho_one_and_equal_projections(self) -> None:
        # When the vector has not moved, p and q measure the same axis.
        v = torch.tensor([1.0, 2.0, -1.0])
        h = torch.tensor([0.5, -0.5, 2.0])
        z = compute_latent(v, v.clone(), h)
        assert z.rho == pytest.approx(1.0)
        assert z.p == pytest.approx(z.q)

    def test_returns_python_floats(self) -> None:
        z = compute_latent(
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([1.0, 1.0]),
        )
        for value in z.as_dict().values():
            assert isinstance(value, float)

    @pytest.mark.parametrize(
        "v0, vt, h",
        [
            (torch.zeros(3), torch.zeros(3), torch.zeros(2)),  # h shorter
            (torch.zeros(3), torch.zeros(2), torch.zeros(3)),  # vt shorter
            (torch.zeros(2), torch.zeros(3), torch.zeros(3)),  # v0 shorter
        ],
    )
    def test_shape_mismatch_raises(self, v0, vt, h) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            compute_latent(v0, vt, h)


class TestDeltaProjection:
    def test_equals_difference_of_projections(self) -> None:
        target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        predicted = torch.tensor([[0.0, 1.0], [2.0, 2.0]])
        vector = torch.tensor([1.0, 1.0])
        expected = project(target, vector) - project(predicted, vector)
        torch.testing.assert_close(delta_projection(target, predicted, vector), expected)

    def test_identical_activations_give_zero(self) -> None:
        acts = torch.randn(5, 8, generator=torch.Generator().manual_seed(0))
        vector = torch.randn(8, generator=torch.Generator().manual_seed(1))
        out = delta_projection(acts, acts.clone(), vector)
        torch.testing.assert_close(out, torch.zeros(5))

    def test_linear_in_the_activation_difference(self) -> None:
        # project is linear, so DeltaP depends only on (target - predicted).
        gen = torch.Generator().manual_seed(2)
        target = torch.randn(4, 6, generator=gen)
        predicted = torch.randn(4, 6, generator=gen)
        vector = torch.randn(6, generator=gen)
        torch.testing.assert_close(
            delta_projection(target, predicted, vector),
            project(target - predicted, vector),
        )

    def test_shape_is_per_sample(self) -> None:
        out = delta_projection(torch.randn(7, 3), torch.randn(7, 3), torch.randn(3))
        assert out.shape == (7,)

    def test_row_misalignment_raises(self) -> None:
        with pytest.raises(ValueError, match="row-aligned"):
            delta_projection(torch.randn(4, 3), torch.randn(5, 3), torch.randn(3))


class TestSummarize:
    def test_evenly_spaced_distribution(self) -> None:
        # 0..100 makes every reported quantile land on an exact integer under
        # torch.quantile's linear interpolation.
        values = torch.arange(101, dtype=torch.float32)
        stats = summarize(values)
        assert stats["mean"] == pytest.approx(50.0)
        assert stats["min"] == 0.0
        assert stats["max"] == 100.0
        assert stats["median"] == pytest.approx(50.0)
        assert stats["p05"] == pytest.approx(5.0)
        assert stats["p25"] == pytest.approx(25.0)
        assert stats["p75"] == pytest.approx(75.0)
        assert stats["p95"] == pytest.approx(95.0)
        assert stats["n"] == 101

    def test_uses_population_std_not_sample_std(self) -> None:
        # [1, 2, 3]: population std = sqrt(2/3) ~= 0.8165, sample std = 1.0.
        stats = summarize(torch.tensor([1.0, 2.0, 3.0]))
        assert stats["std"] == pytest.approx(math.sqrt(2.0 / 3.0))
        assert stats["std"] != pytest.approx(1.0)

    def test_n_is_int(self) -> None:
        assert isinstance(summarize(torch.tensor([1.0, 2.0, 3.0]))["n"], int)

    def test_single_value(self) -> None:
        stats = summarize(torch.tensor([7.0]))
        assert stats["n"] == 1
        assert stats["std"] == pytest.approx(0.0)
        for key in ("mean", "min", "median", "max", "p05", "p95"):
            assert stats[key] == pytest.approx(7.0)

    def test_all_expected_keys_present(self) -> None:
        keys = set(summarize(torch.arange(10, dtype=torch.float32)))
        assert keys == {
            "mean", "std", "min", "p05", "p25",
            "median", "p75", "p95", "max", "n",
        }


class TestLatentState:
    def test_as_dict_keys_and_order(self) -> None:
        state = LatentState(p=1.0, q=2.0, rho=0.5, r=3.0)
        assert state.as_dict() == {"p": 1.0, "q": 2.0, "rho": 0.5, "r": 3.0}

    def test_is_frozen(self) -> None:
        state = LatentState(p=1.0, q=2.0, rho=0.5, r=3.0)
        with pytest.raises(Exception):  # FrozenInstanceError subclasses Exception
            state.p = 9.0  # type: ignore[misc]
