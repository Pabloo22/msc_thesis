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
from pathlib import Path

from method import experiments, steps
from method.backends import get_backend
from method.config import Backend, StepConfig, TrajectoryConfig, to_json
from method.store import Store, atomic_file, get_weights_id
from method.sync import Syncer
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
    syncer: Syncer | None = None,
) -> dict[str, dict[str, float]]:
    """DeltaP at ``cfg``'s base checkpoint for each dataset in ``datasets``.

    ``cfg`` is a *template*: only its model, seed, trait and measurement
    settings are read, since ``weights_key(0)`` slices ``steps[:0]`` and is
    therefore blind to what the trajectory would go on to train on. Any config
    sharing a model and seed yields the same base ``weights_id`` and the same
    answers.

    Pushes after the persona-vector extraction and after each individual
    probe, rather than once at the end: ``datasets`` can run to the dozens, and
    each DeltaP is its own expensive hidden-state pass, so waiting for all of
    them means a preemption partway through loses every probe finished so far.
    """
    base_wid = get_weights_id(cfg, 0)

    def push() -> None:
        if syncer is not None:
            syncer.push_measurement(base_wid)

    # DeltaP projects onto v_0, so the base persona vector has to exist first.
    steps.extract_persona_vector(cfg, 0, store, backend)
    push()
    results = steps.measure_probes(
        cfg, 0, store, backend, probes=datasets, on_probe_done=push
    )
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
) -> Path:
    """Persist the probe results next to the trajectories, self-contained.

    The authoritative copy already lives in the store, keyed by content. This
    summary exists so a plotting machine needs only the trajectories directory
    synced from the GPU box, the same way ``trajectory.json`` is readable
    without the store. Returns the path written, so the caller can push it.
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
    return path


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
    # No-op unless MSC_STORE_REMOTE is set (and never for a mock store). Each
    # (seed, trait) below is a full pass over every probe dataset at the base
    # model, so its results are pushed as soon as they exist rather than after
    # the last one -- a box lost partway through then keeps the seeds it
    # finished instead of all of them being redone elsewhere.
    syncer = Syncer.from_env(store)
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
            results = probe_base(cfg, datasets, store, backend, syncer=syncer)
            summary = write_summary(cfg, results, mock=backend_kind is Backend.MOCK)
            # The measurement bundle is already pushed (probe_base pushes as it
            # goes); only the trajectories-side summary is new here.
            if syncer is not None:
                syncer.push_base_probe(summary)

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
