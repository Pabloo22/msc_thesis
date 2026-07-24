r"""Measure DeltaP at the *base* model for every dataset the experiments use.

    poetry run python -m method.probe_base --seeds 0 1 2 3 4
    poetry run python -m method.probe_base --local --backend mock

The RQ1 scatter plots two series against the behaviour change a step caused:
$\Delta P_t$, recomputed at the checkpoint about to be trained, and $\Delta P_0$,
frozen at the base model. A trajectory only ever records $\Delta P_0$ for its
*first* dataset -- every later step is measured at a checkpoint that has already
moved -- so the blue series has to be measured separately. That is what this
script does.

It is deliberately not part of ``run_trajectory``: $\Delta P_0$ for a dataset
depends only on the base model, the seed and the dataset, never on the
trajectory. Every experiment sharing a model and seed therefore shares these
numbers, and the content-addressed store means running this once per seed
serves exp2, exp3 and exp4 alike.

No training happens here -- only measurement of the untouched base model.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from collections.abc import Sequence

from method import experiments, steps
from method.backends import get_backend
from method.config import Backend, StepConfig, TrajectoryConfig, to_json
from method.store import Store, atomic_file, get_weights_id
from method.utils import (
    DOTENV_PATH,
    base_probes_path,
    check_env_vars,
    load_dotenv,
)

logger = logging.getLogger("probe_base")


def probe_base(
    cfg: TrajectoryConfig,
    datasets: Sequence[StepConfig],
    store: Store,
    backend,
    *,
    stat_log: bool = True,
) -> dict[str, dict[str, float]]:
    """DeltaP at ``cfg``'s base checkpoint for each dataset in ``datasets``.

    ``cfg`` is a *template*: only its model, seed, trait and measurement
    settings are read, since ``weights_key(0)`` slices ``steps[:0]`` and is
    therefore blind to what the trajectory would go on to train on. Any config
    sharing a model and seed yields the same base ``weights_id`` and the same
    answers.
    """
    # DeltaP projects onto v_0, so the base persona vector has to exist first.
    steps.extract_persona_vector(cfg, 0, store, backend)
    results = steps.measure_probes(cfg, 0, store, backend, probes=datasets)
    if stat_log:
        for dataset_id, stat in sorted(results.items()):
            logger.info(
                "DeltaP_0[%s] = %+.4f (n=%d)", dataset_id, stat["mean"], stat["n"]
            )
    return results


def write_summary(
    cfg: TrajectoryConfig,
    results: dict[str, dict[str, float]],
    *,
    mock: bool,
) -> None:
    """Persist the probe results next to the trajectories, self-contained.

    The authoritative copy already lives in the store, keyed by content. This
    summary exists so a plotting machine needs only the trajectories directory
    synced from the GPU box, the same way ``trajectory.json`` is readable
    without the store.
    """
    base_wid = get_weights_id(cfg, 0)
    path = base_probes_path(base_wid, cfg.trait, mock=mock)
    payload = {
        "base_weights_id": base_wid,
        "trait": cfg.trait,
        "seed": cfg.seed,
        "model": dataclasses.asdict(cfg.model),
        "delta_p_config": json.loads(to_json(cfg.delta_p)),
        "probes": results,
    }
    with atomic_file(path) as scratch:
        scratch.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Base probes -> %s", path)


def run(
    *,
    seeds: Sequence[int],
    traits: Sequence[str],
    backend_kind: Backend,
    dtype: str,
    local: bool,
) -> None:
    store = Store.for_backend(backend_kind)
    backend = get_backend(backend_kind, dtype=dtype)
    datasets = experiments.all_probe_datasets(local=local)
    logger.info(
        "Probing %d dataset(s) at the base model for %d seed(s) x %d trait(s)",
        len(datasets),
        len(seeds),
        len(traits),
    )

    for seed in seeds:
        for trait in traits:
            # exp2's builder supplies the right model and measurement presets;
            # its step sequence is irrelevant at t=0.
            cfg = experiments.build_exp2_configs(
                seeds=(seed,), measure_traits=(trait,), local=local
            )[0]
            logger.info(
                "--- base probes: seed=%d trait=%s (%s) ---",
                seed,
                trait,
                get_weights_id(cfg, 0),
            )
            results = probe_base(cfg, datasets, store, backend)
            write_summary(cfg, results, mock=backend_kind is Backend.MOCK)

    # Nothing here needs full weights afterwards, and the base checkpoint is
    # the largest thing this script materialises.
    store.evict_all_merged()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(experiments.SEEDS),
        help="seeds to probe; each has its own base-model measurement",
    )
    parser.add_argument(
        "--traits",
        nargs="+",
        default=list(experiments.MEASURE_TRAITS),
        help="traits to project onto; the costly generation is shared between them",
    )
    parser.add_argument(
        "--backend", type=Backend, choices=list(Backend), default=Backend.REAL
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="use float16 on pre-Ampere GPUs, which only emulate bfloat16",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="use the small local proxy model and capped example counts",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    load_dotenv(DOTENV_PATH)
    if args.backend is Backend.REAL:
        # --local selects the stubbed judge, which never talks to OpenAI.
        required = ["HF_TOKEN"]
        if not args.local:
            required.append("OPENAI_API_KEY")
        check_env_vars(required)

    run(
        seeds=args.seeds,
        traits=args.traits,
        backend_kind=args.backend,
        dtype=args.dtype,
        local=args.local,
    )


if __name__ == "__main__":
    main()
