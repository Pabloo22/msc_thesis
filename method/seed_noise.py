"""Estimate behaviour and latent-state variability across fine-tuning seeds."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from method import experiments
from method.noise import r2_max
from method.visualization.collect import Collection, collect_group, seed_noise_frame

logger = logging.getLogger("seed_noise")

#: Latent components in display order.
Z_COMPONENTS = ("p", "q", "rho", "r")


def coverage(collection: Collection) -> pd.DataFrame:
    """Count available seeds per arm."""
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
    """Summarize ``sigma_seed`` by trait, component, and checkpoint."""
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
    r"""$\sigma_{seed}(b)$ after one fine-tune from the initial model.

    Shared content-addressed checkpoints are deduplicated.
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

    Uses between-arm spread until probe-level spread is available.
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
    """Return all report tables keyed by name."""
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
        note="Seed noise for one branch's Delta b.",
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
