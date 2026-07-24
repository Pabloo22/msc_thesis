"""The latent state z_t and the projection difference DeltaP.

Pure tensor maths, deliberately free of any model loading or subprocess work,
so it runs identically under the real and mock backends and can be unit tested
on synthetic inputs.

For a persona ``i`` at step ``t`` (Section "Specific Setup" of the proposal):

    p_t    = v_0 . h_neutral_t     drift of activations against the ORIGINAL axis
    q_t    = v_t . h_neutral_t     alignment against the CURRENT (rotated) axis
    rho_t  = cos(v_0, v_t)      how far the persona vector has rotated
    r_t    = ||v_t||            whether the persona vector faded or strengthened
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class LatentState:
    """z_t for a single persona at a single step."""

    p: float
    q: float
    rho: float
    r: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def project(activations: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Scalar projection of ``activations`` onto ``vector``.

    Mirrors ``a_proj_b`` in the vendored ``eval/cal_projection.py``: the
    component along the direction, normalised by the direction's length but not
    by the activation's, so magnitude changes in the activation still show up.
    """
    activations = activations.float()
    vector = vector.float()
    return (activations * vector).sum(dim=-1) / vector.norm(dim=-1)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a * b).sum() / (a.norm() * b.norm()))


def compute_latent(
    v0: torch.Tensor, vt: torch.Tensor, h_neutral_t: torch.Tensor
) -> LatentState:
    """Assemble z_t from the base vector, the current vector and h_neutral_t.

    All three arguments must be single-layer vectors of shape ``[d]``, already
    selected at the trajectory's layer.
    """
    if not (v0.shape == vt.shape == h_neutral_t.shape):
        raise ValueError(
            f"shape mismatch: v0={tuple(v0.shape)} vt={tuple(vt.shape)} "
            f"h_neutral={tuple(h_neutral_t.shape)}"
        )
    return LatentState(
        p=float(project(h_neutral_t, v0)),
        q=float(project(h_neutral_t, vt)),
        rho=cosine(v0, vt),
        r=float(vt.float().norm()),
    )


def delta_projection(
    target_acts: torch.Tensor, predicted_acts: torch.Tensor, vector: torch.Tensor
) -> torch.Tensor:
    """Per-sample projection difference along ``vector``.

    ``DeltaP_i = (h_target_i - h_predicted_i) . v / ||v||``, the shift a
    training example asks the model to make along the persona direction. Both
    activation tensors are ``[n_samples, d]`` over the *same* prompts, so they
    must stay row-aligned.
    """
    if target_acts.shape != predicted_acts.shape:
        raise ValueError(
            f"target {tuple(target_acts.shape)} and predicted "
            f"{tuple(predicted_acts.shape)} activations must be row-aligned"
        )
    return project(target_acts, vector) - project(predicted_acts, vector)


def summarize(values: torch.Tensor) -> dict[str, float]:
    """Distribution summary of per-sample DeltaP.

    Percentiles are kept alongside the mean because a handful of extreme
    examples can drive behaviour change without moving the average much.
    """
    values = values.float()
    quantiles = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
    q = torch.quantile(values, quantiles).tolist()
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "p05": q[0],
        "p25": q[1],
        "median": q[2],
        "p75": q[3],
        "p95": q[4],
        "max": float(values.max()),
        "n": int(values.numel()),
    }
