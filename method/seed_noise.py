"""How much of a measurement is just the fine-tuning seed.

    poetry run python -m method.seed_noise
    poetry run python -m method.seed_noise --group exp3 --csv seed_noise.csv

The noise budget of the RQ1 decay experiment needs two numbers that no single
run can supply, because both describe what happens when the *same* recipe is
trained twice:

``sigma_seed(b)``
    The spread in a checkpoint's trait score attributable to data ordering,
    dropout and optimiser nondeterminism. It sets the noise floor in
    ``Var(Delta b)`` and therefore the ceiling on any R^2 fitted against it
    (section 6b). One fine-tune from a fixed checkpoint is the relevant unit, so
    the single-step arms estimate it directly.
``sigma_seed(z)``
    The same for the latent trajectory ``(p, q, rho, r)``. Section 6c reserves a
    paired trunk replicate to check this, on the grounds that stability of ``b``
    does not imply stability of ``z`` -- two seeds can reach the same behaviour
    by different representational paths. That reasoning is right, but the
    replicate is not the only way to pay for it: any family that already sweeps
    seeds over a fixed step sequence answers the same question, with more seeds,
    for free.

exp3 sweeps five seeds over fixed sequences, so this reads ``sigma_seed`` off
runs that are being done anyway. Two limits are worth stating in the write-up
rather than papering over: exp3's arms are at most three steps deep, so they
bound early-trajectory noise and say nothing directly about how it compounds by
``t = 6``; and they are not the trunk under study, so this measures "seed noise
is this big for sequences of this kind", not "trunk A reproduces".

Reads only the run directories, never the store, and tolerates a partly
finished sweep -- which is the point of running it before the family completes.
Re-run it as more seeds land; the estimates simply tighten.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from method import experiments
from method.noise import r2_max
from method.visualization.collect import Collection, collect_group, seed_noise_frame

logger = logging.getLogger("seed_noise")

#: Latent components, in the order the proposal introduces them.
Z_COMPONENTS = ("p", "q", "rho", "r")


def coverage(collection: Collection) -> pd.DataFrame:
    """Seeds actually on disk per arm, so a thin estimate is visible as thin."""
    rows = [
        {"name": run.config.name, "trait": run.trait, "seed": run.seed}
        for run in collection.runs
    ]
    if not rows:
        return pd.DataFrame(columns=["name", "trait", "n_seeds"])
    return (
        pd.DataFrame(rows)
        .groupby(["name", "trait"], as_index=False)
        .agg(n_seeds=("seed", "nunique"))
    )


def by_checkpoint(noise: pd.DataFrame) -> pd.DataFrame:
    """``sigma_seed`` summarised over arms, per trait, component and ``t``.

    The median across arms rather than the mean: a single arm whose seeds
    happened to straddle a behavioural cliff would otherwise set the headline
    number for its whole checkpoint.
    """
    usable = noise[noise["n_seeds"] > 1]
    if usable.empty:
        return usable
    return (
        usable.groupby(["trait", "component", "t"], as_index=False)
        .agg(
            arms=("sd", "size"),
            sd_median=("sd", "median"),
            sd_max=("sd", "max"),
            level_median=("mean", "median"),
        )
        .sort_values(["trait", "component", "t"])
    )


def single_step_behavior_noise(noise: pd.DataFrame) -> pd.DataFrame:
    r"""$\sigma_{seed}(b)$ after exactly one fine-tune from the base model.

    This is the quantity section 6b's noise ceiling wants. A branch in the
    redesigned exp2 is one fine-tune from a trunk checkpoint, so an arm that is
    one fine-tune from $M_0$ is its closest available analogue -- same number of
    opportunities for two seeds to diverge.

    Identified structurally, as ``t = 1`` on arms that have a ``t = 1``, rather
    than by matching on arm names, so it keeps working if the arm set changes.

    Deduplicated on ``checkpoints``, because several arms can share one. exp3's
    ``diff`` arms begin on another arm's dataset, so their ``t = 1`` resolves by
    content addressing to that arm's ``t = 1`` -- the same weights, measured
    once, reached by two named routes. Counting both would inflate the apparent
    number of independent fine-tunes and, worse, feed duplicated values into the
    spread that :func:`implied_ceiling` divides by.
    """
    at_first_step = noise[
        (noise["component"] == "b") & (noise["t"] == 1) & (noise["n_seeds"] > 1)
    ]
    return (
        at_first_step[["name", "trait", "n_seeds", "mean", "sd", "checkpoints"]]
        .drop_duplicates(subset=["trait", "checkpoints"])
        .sort_values(["trait", "name"])
    )


def implied_ceiling(single_step: pd.DataFrame) -> pd.DataFrame:
    r"""Preliminary $R^2_{max}$ from the seed noise measured so far.

    Indicative only, and labelled as such wherever it is printed. The real
    ceiling is computed against the $\Delta b$ spread across the *probe set* at
    a given checkpoint (section 6b), which does not exist until exp2's $t = 0$
    fan has run. What stands in for it here is the spread of $b$ across the arms
    themselves, which is a different and coarser thing: a handful of datasets
    chosen for another purpose.

    Its use is to catch the disqualifying case early. If seed noise already eats
    most of the between-dataset spread on the data in hand, no amount of dense
    sampling in exp2 will recover it, and section 6b's fallback -- averaging
    $\Delta b$ over three seeds per probe -- has to be budgeted before the fans
    are committed rather than after.
    """
    rows = []
    for trait, group in single_step.groupby("trait"):
        if len(group) < 2:
            continue
        observed = float(group["mean"].std(ddof=1))
        sigma_seed = float(group["sd"].median())
        rows.append(
            {
                "trait": trait,
                "arms": len(group),
                "sigma_seed": sigma_seed,
                "spread_across_arms": observed,
                # Eval noise is omitted deliberately: measured at ~1 point on
                # the 0-100 scale it is negligible beside sigma_seed, and
                # including it here would imply a precision this preliminary
                # estimate does not have.
                "r2_max": r2_max(observed**2, sigma_seed**2),
            }
        )
    return pd.DataFrame(rows)


def report(collection: Collection) -> dict[str, pd.DataFrame]:
    """Every table this script prints, keyed by name, for reuse in a notebook."""
    noise = seed_noise_frame(collection)
    single_step = single_step_behavior_noise(noise)
    return {
        "coverage": coverage(collection),
        "noise": noise,
        "by_checkpoint": by_checkpoint(noise),
        "single_step": single_step,
        "ceiling": implied_ceiling(single_step),
    }


def _show(title: str, frame: pd.DataFrame, *, note: str = "") -> None:
    print(f"\n=== {title} ===" + (f"\n{note}" if note else ""))
    if frame.empty:
        print("(nothing estimable yet)")
        return
    print(frame.round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        default=experiments.EXP3,
        help="experiment family to read; needs one that sweeps seeds",
    )
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="also write the full per-arm frame here",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    collection = collect_group(args.group, local=args.local, mock=args.mock)
    print(collection.summary())
    if not collection:
        raise SystemExit(f"no runs on disk for {args.group!r}")

    tables = report(collection)
    _show(
        "seeds per arm",
        tables["coverage"],
        note="sd is only estimated where n_seeds > 1.",
    )
    _show(
        "sigma_seed by checkpoint (median over arms)",
        tables["by_checkpoint"],
        note=(
            "t=0 must show sd=0: weights_key normalises the seed away at the "
            "base checkpoint, so every seed reads one shared measurement."
        ),
    )
    _show(
        "sigma_seed(b) after one fine-tune from M_0",
        tables["single_step"],
        note="The section 6b input: noise on a single branch's Delta b.",
    )
    _show(
        "PRELIMINARY implied R^2 ceiling",
        tables["ceiling"],
        note=(
            "Indicative only -- computed against the spread across these arms, "
            "not across exp2's probe set, which does not exist yet."
        ),
    )

    if args.csv is not None:
        tables["noise"].to_csv(args.csv, index=False)
        print(f"\nfull per-arm frame -> {args.csv}")


if __name__ == "__main__":
    main()
