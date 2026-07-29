r"""Gather saved ``trajectory.json`` runs into plot-ready tidy frames.

The unit here is a :class:`Run`: one ``TrajectoryConfig`` from
:mod:`method.experiments` paired with the measurements that config produced on
disk. Collection is driven by the *registry*, not by globbing
``trajectories/``, for three reasons:

* the design factors a bar chart groups by (condition, target dataset,
  re-alignment trait) live on the config as ``labels``, set where the builder
  already knows them -- recovering them by parsing a run directory name is
  ambiguous, since dataset names themselves contain underscores;
* enumerating the configs that *should* exist is what makes a run that has not
  happened yet distinguishable from one that is not part of the experiment, so
  a half-finished sweep plots what it has and says what it is missing;

The staleness check guards the one failure mode this design introduces: if a
config is edited after its runs were saved, the code-as-manifest assumption
quietly breaks. Comparing the final ``weights_id`` on disk against the one the
current config hashes to catches exactly that.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from method import experiments
from method.config import TrajectoryConfig
from method.store import get_weights_id
from method.utils import base_probes_path, trajectory_run_dir
from method.visualization import schema
from method.visualization.schema import Trajectory

logger = logging.getLogger(__name__)

#: Which builder generates each experiment family. Defined next to the builders
#: themselves (:data:`method.experiments.GROUP_BUILDERS`) and re-exported here
#: for convenience. Going through the builders rather than filtering
#: ``experiments.REGISTRY`` is what lets ``--local`` work: only the paper-scale
#: variants are registered, but both scales are reachable from the builder.
GROUP_BUILDERS = experiments.GROUP_BUILDERS


@dataclass(frozen=True)
class Run:
    """One trajectory config plus the measurements it produced on disk."""

    config: TrajectoryConfig
    trajectory: Trajectory
    path: Path
    #: Whether this run came from the mock root; needed to find the matching
    #: base-probe summary, which is quarantined the same way.
    mock: bool = False

    @property
    def labels(self) -> dict[str, str]:
        return self.config.label_map

    @property
    def seed(self) -> int:
        return self.config.seed

    @property
    def trait(self) -> str:
        return self.config.trait

    @property
    def group(self) -> str:
        return self.config.group

    def label(self, key: str, default: str = "") -> str:
        return self.config.label_map.get(key, default)

    def behavior(self, t: int) -> float:
        """``b_t``: the measured trait score at checkpoint ``t``."""
        return self.trajectory.steps[t].behavior[self.trait]

    def final_step_delta(self) -> float:
        r"""$\Delta b$ caused by the *last* fine-tuning step, $b_T - b_{T-1}$.

        Defined structurally rather than per condition so it means the same
        thing for a 1-step baseline and a 3-step re-alignment trajectory: in
        both, the last step trains on the dataset under study, so this is
        "how much did training on that dataset move the model, from wherever
        it happened to be".
        """
        return self.behavior(-1) - self.behavior(-2)

    def total_delta(self) -> float:
        r"""$b_T - b_0$: how far the whole trajectory left the model from base.

        For the diversity designs, whose every condition *ends* with a
        re-alignment step, this is the residual misalignment that re-alignment
        failed to undo.
        """
        return self.behavior(-1) - self.behavior(0)


@dataclass(frozen=True)
class Collection:
    """Runs found for one experiment family, plus what was not usable."""

    group: str
    runs: list[Run] = field(default_factory=list)
    missing: list[Path] = field(default_factory=list)
    stale: list[Path] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.runs)

    def __iter__(self):
        return iter(self.runs)

    def __bool__(self) -> bool:
        return bool(self.runs)

    @property
    def trajectories(self) -> list[Trajectory]:
        return [run.trajectory for run in self.runs]

    def filter(self, **labels: str) -> Collection:
        """Sub-collection whose runs match every ``label=value`` given.

        ``trait`` and ``seed`` are accepted alongside label keys, since they
        are the two axes that live on the config proper rather than in
        ``labels``.
        """
        selected = []
        for run in self.runs:
            attrs = {"trait": run.trait, "seed": str(run.seed), **run.labels}
            if all(attrs.get(k) == str(v) for k, v in labels.items()):
                selected.append(run)
        return Collection(self.group, selected, self.missing, self.stale)

    def values(self, key: str) -> list[str]:
        """Distinct values of a label (or ``trait``), in first-seen order."""
        if key == "trait":
            return list(dict.fromkeys(run.trait for run in self.runs))
        return list(dict.fromkeys(run.label(key) for run in self.runs if run.label(key)))

    def summary(self) -> str:
        return (
            f"{self.group}: {len(self.runs)} run(s) loaded, "
            f"{len(self.missing)} not yet run, {len(self.stale)} stale"
        )


def collect(
    configs: Iterable[TrajectoryConfig], *, group: str = "", mock: bool = False
) -> Collection:
    """Load every config's saved trajectory, reporting the ones not on disk.

    ``mock`` reads the parallel ``trajectories-mock/`` root, so a figure is
    never accidentally drawn from a mix of real and synthetic measurements.
    """
    runs: list[Run] = []
    missing: list[Path] = []
    stale: list[Path] = []

    for cfg in configs:
        run_dir = trajectory_run_dir(cfg.name, cfg.seed, cfg.model.name, mock=mock)
        path = run_dir / "trajectory.json"
        if not path.exists():
            missing.append(path)
            continue
        trajectory = schema.load_trajectory(path)
        expected = get_weights_id(cfg, len(cfg.steps))
        found = trajectory.steps[-1].weights_id
        if found != expected:
            logger.warning(
                "Stale run %s: final weights_id is %s but %r now hashes to %s. "
                "The config was edited after this run was saved; re-run it or "
                "revert the edit. Skipping.",
                path,
                found,
                cfg.name,
                expected,
            )
            stale.append(path)
            continue
        runs.append(Run(config=cfg, trajectory=trajectory, path=path, mock=mock))

    collection = Collection(group or _sole_group(runs), runs, missing, stale)
    if missing:
        logger.warning(
            "%d/%d run(s) for %r are not on disk yet; first few: %s",
            len(missing),
            len(missing) + len(runs) + len(stale),
            collection.group,
            ", ".join(str(p.parent.name) for p in missing[:5]),
        )
    return collection


def _sole_group(runs: Sequence[Run]) -> str:
    groups = {run.group for run in runs}
    return groups.pop() if len(groups) == 1 else ""


def collect_group(
    group: str, *, local: bool = False, mock: bool = False, **builder_kwargs
) -> Collection:
    """Load every run of one experiment family, at paper or local scale."""
    try:
        build = GROUP_BUILDERS[group]
    except KeyError:
        raise KeyError(
            f"unknown group {group!r}; known: {', '.join(sorted(GROUP_BUILDERS))}"
        ) from None
    return collect(build(local=local, **builder_kwargs), group=group, mock=mock)


def collect_all(
    *, local: bool = False, mock: bool = False, **builder_kwargs
) -> list[Run]:
    """Every run of every family -- used to pool $\\Delta P_0$ measurements."""
    return [
        run
        for group in GROUP_BUILDERS
        for run in collect_group(group, local=local, mock=mock, **builder_kwargs).runs
    ]


# --- Delta P_0 ------------------------------------------------------------


def delta_p_0_lookup(
    runs: Iterable[Run], *, stat: str = "mean"
) -> dict[tuple[str, int], dict[str, float]]:
    r"""Per-(trait, seed) map of ``dataset/version`` -> $\Delta P_0$.

    $\Delta P_0$ is by definition measured against the base model, so only a
    run's ``t=0`` record can supply one, and it supplies it for exactly one
    dataset: the one that run trains on first. Pooling across *all* experiment
    families therefore covers more datasets than any single family does --
    every hysteresis baseline contributes its own.

    Keyed by (trait, seed) rather than flattened because the base checkpoint
    is measured independently per seed (``weights_key`` includes seed even at
    ``t=0``), so each seed has its own $v_0$ and its own $\Delta P_0$. Pairing
    a seed's $\Delta P_t$ against another seed's $\Delta P_0$ would inject
    measurement noise into the very correlation being tested.

    Three sources are merged, in increasing order of coverage: the dataset a
    run's *first* step trains on, anything that run probed at ``t=0``, and the
    dedicated sweep written by :mod:`method.probe_base`, which measures every
    dataset the experiments use. They are measurements of the same quantity, so
    where they overlap they agree; the sweep is applied last simply because it
    is the one guaranteed to be complete.
    """
    lookup: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for run in runs:
        first = run.trajectory.steps[0]
        if first.t != 0:
            continue
        key = (run.trait, run.seed)
        if first.delta_p is not None and first.next_dataset is not None:
            lookup[key][first.next_dataset] = first.delta_p[stat]
        for dataset, summary in first.probes.items():
            if stat in summary:
                lookup[key][dataset] = summary[stat]

    for key, probes in base_probe_lookup(runs, stat=stat).items():
        lookup[key].update(probes)
    return dict(lookup)


def base_probe_lookup(
    runs: Iterable[Run], *, stat: str = "mean"
) -> dict[tuple[str, int], dict[str, float]]:
    r"""Read the base-model $\Delta P$ sweeps written by :mod:`method.probe_base`.

    Located by rebuilding each run's base ``weights_id`` rather than by
    globbing, so a summary is only ever read for the exact model and seed that
    produced it. Missing files are not an error: the sweep is a separate,
    optional pass, and its absence just means fewer datasets appear in the
    projection scatter.
    """
    out: dict[tuple[str, int], dict[str, float]] = {}
    for run in runs:
        key = (run.trait, run.seed)
        if key in out:
            continue
        path = base_probes_path(
            get_weights_id(run.config, 0), run.trait, mock=run.mock
        )
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[key] = {
            dataset: summary[stat]
            for dataset, summary in payload.get("probes", {}).items()
            if stat in summary
        }
    return out


def missing_delta_p_0(
    collection: Collection, lookup: Mapping[tuple[str, int], Mapping[str, float]]
) -> set[str]:
    r"""Datasets trained on in ``collection`` that no run measured at $t=0$.

    Every one of these is silently dropped from the projection scatter (see
    :func:`method.visualization.schema.projection_pairs`), so callers should
    surface them. The fix is to run :mod:`method.probe_base`, which measures
    $\Delta P$ for every dataset against each seed's base checkpoint without
    training anything.
    """
    trained_on = {
        step.next_dataset
        for run in collection.runs
        for step in run.trajectory.steps
        if step.next_dataset is not None
    }
    measured = {dataset for per_seed in lookup.values() for dataset in per_seed}
    return trained_on - measured


def projection_frame(
    collection: Collection,
    lookup: Mapping[tuple[str, int], Mapping[str, float]],
    *,
    stat: str = "mean",
) -> pd.DataFrame:
    r"""Tidy $(\Delta P_0, \Delta P_t, \Delta b_{t+1})$ rows, matched per seed.

    Each (trait, seed) group is paired against its *own* base-model
    $\Delta P_0$ values; groups with no measurements at all are skipped.
    """
    by_key: dict[tuple[str, int], list[Trajectory]] = defaultdict(list)
    for run in collection.runs:
        by_key[(run.trait, run.seed)].append(run.trajectory)

    frames = []
    for (trait, seed), trajectories in sorted(by_key.items()):
        per_seed = lookup.get((trait, seed))
        if not per_seed:
            continue
        pairs = schema.projection_pairs(trajectories, per_seed, stat=stat)
        if not pairs.empty:
            frames.append(pairs.assign(trait=trait))
    if not frames:
        return pd.DataFrame(
            columns=["name", "seed", "t", "dataset", "delta_p_0", "delta_p_t",
                     "delta_behavior", "trait"]
        )
    return pd.concat(frames, ignore_index=True)


# --- bar-chart frames -----------------------------------------------------


def hysteresis_frame(collection: Collection) -> pd.DataFrame:
    r"""RQ2 hysteresis rows: $\Delta b$ of each run's final step.

    One row per run, with ``condition`` taken straight from the config label
    (``baseline`` / ``same`` / ``diff``). The value is always the behaviour
    change caused by the last step, which in every exp3 condition is a step
    onto the same target dataset -- so the bars differ only in what the model
    had already been trained on.

    Baseline runs carry no ``realign_trait`` (they have no re-alignment step,
    so one baseline serves every realign trait); they are emitted once per
    realign trait present in ``collection`` so that a per-realign-trait figure
    always has its reference bar.
    """
    realign_traits = collection.values("realign_trait")
    rows = []
    for run in collection.runs:
        row = {
            "name": run.config.name,
            "trait": run.trait,
            "seed": run.seed,
            "dataset": run.label("dataset"),
            "condition": run.label("condition"),
            "delta_behavior": run.final_step_delta(),
        }
        own = run.label("realign_trait")
        if own:
            rows.append({**row, "realign_trait": own})
        else:
            rows.extend({**row, "realign_trait": rt} for rt in realign_traits or [""])
    return pd.DataFrame(rows)


def diversity_frame(collection: Collection) -> pd.DataFrame:
    r"""RQ2 diversity rows: residual misalignment $b_T - b_0$ after re-alignment.

    Every exp4 condition ends with a re-alignment step, so a bar is "how much
    of the trait survived re-alignment", measured against the same seed's base
    model.
    """
    return pd.DataFrame(
        [
            {
                "name": run.config.name,
                "trait": run.trait,
                "seed": run.seed,
                "condition": run.label("condition"),
                "realign_trait": run.label("realign_trait"),
                "dataset": run.label("dataset"),
                "delta_behavior": run.total_delta(),
            }
            for run in collection.runs
        ]
    )
