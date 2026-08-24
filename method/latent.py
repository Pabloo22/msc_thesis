"""The latent state z_t and the projection difference DeltaP.

Pure tensor maths, deliberately free of any model loading or subprocess work,
so it runs identically under the real and mock backends and can be unit tested
on synthetic inputs.

For a persona ``i`` at step ``t`` (Section "Specific Setup" of the proposal):

    p_t    = cos(h_neutral_t, v_0)  drift of activations against the ORIGINAL axis
    q_t    = cos(h_neutral_t, v_t)  alignment against the CURRENT (rotated) axis
    rho_t  = cos(v_0, v_t)      how far the persona vector has rotated
    r_t    = ||v_t||            whether the persona vector faded or strengthened

All four are therefore either a cosine on [-1, 1] or a length: none of them
carries the activation's own magnitude, so growth or shrinkage of h_neutral
moves nothing in z_t and a level can be read off a plot without knowing the
scale the hidden states happen to live at. DeltaP keeps the older convention
on purpose -- see :func:`project`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

#: How ``p`` and ``q`` are normalised, stamped on every ``trajectory.json``
#: this code writes. Nothing about a bare ``z`` block says which convention
#: produced it, and the two are a fixed rescaling apart -- so without a marker
#: a run measured before the change and one measured after look identical and
#: plot on the same axes. Readers treat its absence as :data:`LEGACY_CONVENTION`.
CONVENTION = "cosine"

#: What a ``trajectory.json`` written before the change holds: ``p`` and ``q``
#: as scalar projections, normalised by the persona vector's length but not the
#: activation's. :mod:`method.backfill_latent_cosine` converts these in place.
LEGACY_CONVENTION = "projection"

#: Key under which every ``z`` block carries ``||h_neutral_t||``, the length
#: :func:`cosine` divides out of ``p`` and ``q``.
#:
#: Not a fifth coordinate of z_t -- that is and stays ``(p, q, rho, r)`` -- but
#: the normaliser recorded beside it, for two reasons. It makes the convention
#: reversible: ``p * h_norm`` is exactly the scalar projection the old one
#: reported, so no result is locked behind the choice. And it answers the one
#: question a cosine cannot: ``p`` falling can mean the neutral state turned off
#: the persona axis *or* merely grew in unrelated directions, and only the norm
#: separates the two.
#:
#: Absent from any block written before this existed; :mod:`method.backfill_h_norm`
#: fills those in from the store. Nothing derives a figure from it yet, so its
#: absence degrades rather than raises -- see
#: :func:`method.steps.compute_step_latent`.
H_NORM = "h_norm"


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

    This is DeltaP's convention and only DeltaP's. $z_t$ used to be built from
    it too and now uses :func:`cosine` instead; the difference is not cosmetic,
    so the two must not be merged back together. DeltaP is defined as the shift
    a training example asks the model to make *along* the persona direction --
    a difference of two projections in the same units -- and normalising each
    activation by its own norm first would make it a difference of two cosines,
    losing exactly the magnitude interpretation DeltaP exists to carry.
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

    ``p`` and ``q`` are cosines of the *mean* neutral activation, not means of
    per-prompt cosines. The two are different quantities, but the neutral
    prompts' activation norms spread by only ~5% about their mean in practice,
    so no handful of long activations dominates the average before it is
    normalised, and the cheaper one is the honest one to take.
    """
    if not (v0.shape == vt.shape == h_neutral_t.shape):
        raise ValueError(
            f"shape mismatch: v0={tuple(v0.shape)} vt={tuple(vt.shape)} "
            f"h_neutral={tuple(h_neutral_t.shape)}"
        )
    return LatentState(
        p=cosine(h_neutral_t, v0),
        q=cosine(h_neutral_t, vt),
        rho=cosine(v0, vt),
        r=float(vt.float().norm()),
    )


def latent_record(
    v0: torch.Tensor, vt: torch.Tensor, h_neutral_t: torch.Tensor
) -> dict[str, float]:
    """One ``z`` block as it is written to disk: z_t plus :data:`H_NORM`.

    The norm is free here -- ``h_neutral_t`` is already loaded and
    :func:`cosine` computes its length anyway -- which is the whole reason to
    take it at measurement time rather than reconstruct it later from a store
    that may no longer be on the machine doing the reading.
    """
    record = compute_latent(v0, vt, h_neutral_t).as_dict()
    record[H_NORM] = float(h_neutral_t.float().norm())
    return record


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
