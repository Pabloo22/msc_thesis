r"""$\Delta P$ against the axis each checkpoint drew for itself.

    poetry run python -m method.onpolicy_delta_p --store store-thin
    poetry run python -m method.onpolicy_delta_p --trunk a --trait evil
    poetry run python -m method.onpolicy_delta_p --dry-run

The runner for :data:`method.experiments.EXP2_ONPOLICY` and
:data:`~method.experiments.EXP2_ONPOLICY_REGEN`, the two families whose
persona vector is re-drawn from text the checkpoint generated rather than
re-encoded from $M_0$'s. Both are pure arithmetic over tensors that already
exist -- the probe activations a measured trunk cached, and the vector
:mod:`method.axis_refresh` drew -- so neither needs a GPU, an adapter chain or
a judge. ``run_trajectory`` would give the same numbers; it would also
materialise checkpoints, check adapters and re-run evals to get there, none of
which a re-projection has any use for.

**Reads the store, writes only ``trajectory.json``.** Every other runner caches
its per-dataset statistics back into the measurement bundle. This one does not,
because it can be handed a bundle that holds means but not samples (see below):
the summary it could write there would carry a mean and nothing else, and the
next reader could not tell that from a full one. A ``trajectory.json`` is the
plotting layer's input and is written whole, so the same partial knowledge is
harmless there.

**Two levels of detail, decided per probe by what is on disk.** With
``samples_layer<L>.pt`` present for both the target and the predicted term, the
projection runs per training example and the record carries the full summary --
mean, spread, percentiles, ``n`` -- exactly as :func:`method.steps.
compute_delta_p` writes it. With only ``mean_by_layer.pt``, the record carries
the mean alone. That is not an approximation: the projection is linear, so the
mean of the per-sample differences *is* the difference of the means projected,
and only the spread is unrecoverable. It matters because a 400 KB mean tensor
can be fetched out of a 3 GB remote bundle (see :mod:`method.sparse_pull`)
where the 175 MB sample tensor beside it cannot, which is what makes this
runnable on a laptop at all. The figures and the correlation table read the
mean; anything wanting the spread needs the samples.

**Behaviour is copied from the decay trunk, not re-measured.** $b_t$ is a
property of the checkpoint, and the same checkpoint under the same trait is
what :data:`~method.experiments.EXP2_DECAY` already scored -- these families
differ only in how a projection is taken over it. Re-deriving it would mean
reading judged CSVs this store may not hold, to arrive at the number sitting
in the decay run's record. So the decay trunk is required, and its absence is
an error rather than a record without $b_t$.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from method import experiments
from method.config import DeltaPView, PredictedSource, TrajectoryConfig, to_json
from method.latent import delta_projection, project, summarize
from method.steps import onpolicy_vector_path, predicted_dir
from method.store import Store, atomic_file, get_weights_id, training_sample_id
from method.utils import trajectory_run_dir
from method.visualization.schema import load_trajectory

logger = logging.getLogger(__name__)

#: The families this runs. Their configs are read off the builders rather than
#: restated, so a change to either reaches here without a second edit.
GROUPS = (experiments.EXP2_ONPOLICY, experiments.EXP2_ONPOLICY_REGEN)

#: How a trunk is identified across families: same trait, same driver schedule,
#: same seed, and therefore the same checkpoints.
TrunkKey = tuple[str, str, int]


@dataclass(frozen=True)
class MeasuredTrunk:
    """What a decay trunk already knows about its own checkpoints.

    Both fields travel together on purpose: ``z_convention`` says how the
    numbers in ``z`` are normalised (see :data:`method.latent.CONVENTION`), so
    copying one without the other would produce a record stamped with an
    opinion nothing here computed.
    """

    behavior: tuple[Mapping[str, float], ...]
    z: tuple[Mapping[str, Mapping[str, float]], ...]
    z_convention: str


def trunk_measurements(
    *, local: bool = False, mock: bool = False
) -> dict[TrunkKey, MeasuredTrunk]:
    r"""Each decay trunk's per-checkpoint measurements, keyed by :data:`TrunkKey`.

    Only the trunks: a branch is one fine-tune off a checkpoint, scored once
    and discarded, and it records nothing about the checkpoint it left from.
    """
    index: dict[TrunkKey, MeasuredTrunk] = {}
    for cfg in experiments.build_exp2_decay_configs(local=local):
        labels = cfg.label_map
        if labels.get("role") != "trunk":
            continue
        path = (
            trajectory_run_dir(cfg.name, cfg.seed, cfg.model.name, mock=mock)
            / "trajectory.json"
        )
        if not path.exists():
            continue
        trajectory = load_trajectory(path)
        index[(cfg.trait, labels.get("trunk", ""), cfg.seed)] = MeasuredTrunk(
            behavior=tuple(dict(step.behavior) for step in trajectory.steps),
            z=tuple(
                {k: dict(v) for k, v in step.z.items()} for step in trajectory.steps
            ),
            z_convention=trajectory.z_convention,
        )
    return index


def _layer_slice(path: Path, layer: int) -> torch.Tensor:
    """One layer out of a stacked ``[n_layers, d]`` tensor on disk."""
    return torch.load(path, weights_only=False)[layer]


def probe_stats(
    store: Store,
    wid: str,
    trait: str,
    dp_key: str,
    *,
    layer: int,
    predicted: PredictedSource,
) -> dict[str, float] | None:
    r"""$\Delta P$ for one probe at one checkpoint, or ``None`` if uncached.

    ``None`` rather than an exception: a sweep over three trunks and two traits
    meets checkpoints in whatever state the boxes left them, and one probe
    missing its activations should narrow the record it lands in, not end the
    run. The caller counts and reports them.
    """
    vector_path = onpolicy_vector_path(store, wid, trait)
    if not vector_path.exists():
        return None
    target = store.measurement_dir(wid) / "delta_p_target" / dp_key
    predicted_path = predicted_dir(store, wid, dp_key, predicted)

    samples = f"samples_layer{layer}.pt"
    if (target / samples).exists() and (predicted_path / samples).exists():
        values = delta_projection(
            torch.load(target / samples, weights_only=False),
            torch.load(predicted_path / samples, weights_only=False),
            _layer_slice(vector_path, layer),
        )
        return summarize(values)

    means = "mean_by_layer.pt"
    if not ((target / means).exists() and (predicted_path / means).exists()):
        return None
    difference = _layer_slice(target / means, layer) - _layer_slice(
        predicted_path / means, layer
    )
    return {"mean": float(project(difference, _layer_slice(vector_path, layer)))}


def checkpoint_record(
    cfg: TrajectoryConfig,
    t: int,
    store: Store,
    *,
    measured: MeasuredTrunk,
    view: DeltaPView,
) -> tuple[dict, int]:
    """One ``steps`` entry, plus how many of its probes had no activations."""
    wid = get_weights_id(cfg, t)
    probes: dict[str, dict[str, float]] = {}
    unmeasured = 0
    for probe in cfg.probes:
        # The rule compute_delta_p applies: at t = 0 the current model *is*
        # M_0 and generation is greedy, so both sources resolve to the same
        # cached answers rather than to two copies of them.
        predicted = PredictedSource.BASE if t == 0 else view.predicted
        stats = probe_stats(
            store,
            wid,
            cfg.trait,
            training_sample_id(probe, cfg.seed),
            layer=cfg.model.layer,
            predicted=predicted,
        )
        if stats is None:
            unmeasured += 1
            continue
        probes[probe.dataset_id] = stats
    record = {
        "t": t,
        "weights_id": wid,
        "behavior": measured.behavior[t],
        "z": measured.z[t],
        view.key("probes"): probes,
    }
    return record, unmeasured


def build_trajectory(
    cfg: TrajectoryConfig,
    store: Store,
    trunks: Mapping[TrunkKey, MeasuredTrunk],
) -> tuple[dict, int]:
    """The whole ``trajectory.json`` payload for one config, and its gaps."""
    key = (cfg.trait, cfg.label_map.get("trunk", ""), cfg.seed)
    if key not in trunks:
        raise FileNotFoundError(
            f"no decay trunk on disk for {key}, so there is no b_t to record "
            f"beside {cfg.name}'s projections; run the "
            f"{experiments.EXP2_DECAY!r} family (or sync trajectories/) first"
        )
    measured = trunks[key]
    view = next(iter(cfg.delta_p.views))
    records = []
    unmeasured = 0
    for t in range(len(cfg.steps) + 1):
        record, gaps = checkpoint_record(cfg, t, store, measured=measured, view=view)
        records.append(record)
        unmeasured += gaps
    return {
        "config": json.loads(to_json(cfg)),
        "z_convention": measured.z_convention,
        "steps": records,
    }, unmeasured


def select_configs(
    *,
    groups: Sequence[str] = GROUPS,
    traits: Sequence[str] | None = None,
    trunk_names: Sequence[str] | None = None,
    local: bool = False,
) -> list[TrajectoryConfig]:
    """The configs a CLI selection names, in the order the families list them."""
    return [
        cfg
        for group in groups
        for cfg in experiments.GROUP_BUILDERS[group](local=local)
        if (traits is None or cfg.trait in traits)
        and (trunk_names is None or cfg.label_map.get("trunk", "") in trunk_names)
    ]


def run(
    store: Store,
    configs: Sequence[TrajectoryConfig],
    *,
    local: bool = False,
    dry_run: bool = False,
) -> list[Path]:
    """Write a ``trajectory.json`` for every config given. Returns the paths."""
    trunks = trunk_measurements(local=local)
    written: list[Path] = []
    for cfg in configs:
        path = (
            trajectory_run_dir(cfg.name, cfg.seed, cfg.model.name) / "trajectory.json"
        )
        payload, gaps = build_trajectory(cfg, store, trunks)
        total = (len(cfg.steps) + 1) * len(cfg.probes)
        if gaps:
            logger.warning(
                "%s: %d of %d probe measurements have no cached activations "
                "in %s and are left out",
                cfg.name,
                gaps,
                total,
                store.root,
            )
        if dry_run:
            logger.info("[dry run] would write %s (%d probes)", path, total - gaps)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with atomic_file(path) as scratch:
            scratch.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("wrote %s (%d of %d probes)", path, total - gaps, total)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--store",
        type=Path,
        default=None,
        help=(
            "store root to read activations and vectors from; defaults to the "
            "real store, but a laptop wants the thin copy method.sparse_pull "
            "leaves in store-thin/"
        ),
    )
    parser.add_argument("--trait", action="append", dest="traits", default=None)
    parser.add_argument("--trunk", action="append", dest="trunks", default=None)
    parser.add_argument(
        "--group",
        action="append",
        dest="groups",
        choices=list(GROUPS),
        default=None,
        help="which family to write; defaults to both",
    )
    parser.add_argument("--local", action="store_true", help="local-scale configs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run(
        Store(args.store) if args.store else Store(),
        select_configs(
            groups=args.groups or GROUPS,
            traits=args.traits,
            trunk_names=args.trunks,
            local=args.local,
        ),
        local=args.local,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
