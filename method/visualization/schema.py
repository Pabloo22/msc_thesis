r"""Typed views over the ``trajectory.json`` files written by
:mod:`method.run_trajectory`, plus tidy-data shaping for the figures.

Kept separate from plotting and from fake-data generation so that real
trajectories (once produced by actual fine-tuning runs) and
:mod:`method.visualization.synthetic` fixtures are interchangeable: both are
just :class:`Trajectory` objects.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from method.config import DeltaPView, PredictedSource, ProjectionAxis
from method.latent import CONVENTION as Z_CONVENTION
from method.latent import LEGACY_CONVENTION as LEGACY_Z_CONVENTION


@dataclass(frozen=True)
class StepRecord:
    """One entry of a trajectory's ``"steps"`` list -- everything known about
    checkpoint ``M_t``, plus (if ``t`` is not the last step) the action
    features for the update that follows it.
    """

    t: int
    weights_id: str
    behavior: dict[str, float]
    #: Keyed by h_neutral source, e.g. ``"base"``. Empty on a *branch* endpoint,
    #: which by design measures only ``b`` (see
    #: :class:`method.config.MeasurementLevel`): ``z`` is a property of the
    #: trunk checkpoint the branch left from, and the trunk records it there.
    #: Emptiness is therefore the marker that distinguishes the two kinds of
    #: record, which is why it defaults rather than being required.
    z: dict[str, dict[str, float]] = field(default_factory=dict)
    #: $\Delta \hat{P}_t$ for ``next_dataset``: axis and encoder current, the
    #: predicted term still $M_0$'s cached answers. Unqualified because it is
    #: the key every trajectory on disk has always written (see
    #: :meth:`method.config.DeltaPView.key`) -- the hat lives in
    #: :attr:`delta_p_current`'s absence, not in this name. Analysis frames
    #: spell it out; see :data:`method.visualization.decay.SERIES_COLUMNS`.
    #: Absent on the final checkpoint.
    delta_p: dict[str, float] | None = None
    next_dataset: str | None = None  # "dataset/version"; absent on final checkpoint
    #: DeltaP for datasets measured at *this* checkpoint whether or not the
    #: trajectory trains on them next, keyed by ``"dataset/version"`` (see
    #: ``TrajectoryConfig.probes``). Present on every checkpoint including the
    #: last, unlike ``delta_p``. Empty for trajectories saved before probing
    #: existed, which is why it defaults rather than being required.
    probes: dict[str, dict[str, float]] = field(default_factory=dict)
    #: The same two quantities under the other views of the projection
    #: difference (see :class:`method.config.DeltaPView`): ``_v0`` holds the
    #: axis at $v^{(0)}$ while the encoder moves, ``_current`` lets the
    #: checkpoint answer the prompts itself, and ``_v0_current`` does both --
    #: the fourth corner of that 2x2. Empty on every run that did not ask for
    #: them, which is all of them but the families scoped to pay.
    delta_p_v0: dict[str, float] | None = None
    probes_v0: dict[str, dict[str, float]] = field(default_factory=dict)
    delta_p_current: dict[str, float] | None = None
    probes_current: dict[str, dict[str, float]] = field(default_factory=dict)
    delta_p_v0_current: dict[str, float] | None = None
    probes_v0_current: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StepRecord:
        def delta_p(view: DeltaPView) -> dict[str, float] | None:
            value = payload.get(view.key("delta_p"))
            return dict(value) if value is not None else None

        def probes(view: DeltaPView) -> dict[str, dict[str, float]]:
            entries = payload.get(view.key("probes")) or {}
            return {dataset: dict(stats) for dataset, stats in entries.items()}

        base_axis = DeltaPView(axis=ProjectionAxis.BASE)
        own_answers = DeltaPView(predicted=PredictedSource.CURRENT)
        both = DeltaPView(
            axis=ProjectionAxis.BASE, predicted=PredictedSource.CURRENT
        )
        return cls(
            t=payload["t"],
            weights_id=payload["weights_id"],
            behavior=dict(payload["behavior"]),
            z={k: dict(v) for k, v in (payload.get("z") or {}).items()},
            delta_p=delta_p(DeltaPView()),
            next_dataset=payload.get("next_dataset"),
            probes=probes(DeltaPView()),
            delta_p_v0=delta_p(base_axis),
            probes_v0=probes(base_axis),
            delta_p_current=delta_p(own_answers),
            probes_current=probes(own_answers),
            delta_p_v0_current=delta_p(both),
            probes_v0_current=probes(both),
        )

    def probes_by(self, view: DeltaPView) -> dict[str, dict[str, float]]:
        """This checkpoint's probe DeltaP under one view."""
        return {
            "": self.probes,
            "v0": self.probes_v0,
            "current": self.probes_current,
            "v0_current": self.probes_v0_current,
        }[view.suffix]


@dataclass(frozen=True)
class Trajectory:
    """A full trajectory: identifying metadata plus its ordered checkpoints."""

    name: str
    trait: str
    seed: int
    steps: tuple[StepRecord, ...]
    source: Path | None = None
    #: How this run's ``p`` and ``q`` are normalised (see
    #: :data:`method.latent.CONVENTION`). Runs written before ``z`` became
    #: cosines carry no marker, so their absence *is* the legacy answer -- and
    #: since the two conventions differ by a fixed rescaling, mixing them in
    #: one figure is silent rather than obviously broken. See
    #: :attr:`z_is_stale`.
    z_convention: str = LEGACY_Z_CONVENTION

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], source: Path | None = None
    ) -> Trajectory:
        cfg = payload["config"]
        return cls(
            name=cfg["name"],
            trait=cfg["trait"],
            seed=cfg["seed"],
            steps=tuple(StepRecord.from_dict(s) for s in payload["steps"]),
            source=source,
            z_convention=payload.get("z_convention", LEGACY_Z_CONVENTION),
        )

    @property
    def z_is_stale(self) -> bool:
        """Whether this run carries ``z`` under a superseded convention.

        False for a run that records no ``z`` at all -- a branch endpoint has
        nothing to be stale about (see :class:`method.config.MeasurementLevel`)
        and would otherwise be reported as needing a backfill it cannot use.
        """
        return self.z_convention != Z_CONVENTION and any(s.z for s in self.steps)

    def behavior_series(self, key: str | None = None) -> list[float]:
        """``b_t`` for one behaviour key across every checkpoint (default the trait)."""
        key = key or self.trait
        return [s.behavior[key] for s in self.steps]

    def z_series(self, component: str, source: str = "base") -> list[float]:
        """One ``z_t`` component across every checkpoint.

        ``component`` is one of z_t's four (``p``/``q``/``rho``/``r``) or
        ``h_norm``, the activation length ``p`` and ``q`` were divided by, which
        blocks measured before it was recorded do not carry until
        :mod:`method.backfill_h_norm` fills them in.

        Raises ``KeyError`` if any checkpoint lacks the series, for the same
        reason :meth:`probe_series` does: a short series plots as a truncated
        line rather than as the missing measurement it is. Use
        :meth:`has_latent` to ask first -- a branch never has one.
        """
        missing = [s.t for s in self.steps if source not in s.z]
        if missing:
            raise KeyError(
                f"trajectory {self.name!r} (seed {self.seed}) has no {source!r} "
                f"latent at checkpoint(s) {missing}. Branch endpoints measure "
                "only b by design (see method.config.MeasurementLevel); z is "
                "recorded by the trunk they forked from."
            )
        unrecorded = [s.t for s in self.steps if component not in s.z[source]]
        if unrecorded:
            raise KeyError(
                f"trajectory {self.name!r} (seed {self.seed}) records no "
                f"{component!r} in its {source!r} latent at checkpoint(s) "
                f"{unrecorded}. A block measured before a component existed "
                "keeps its original fields until a backfill adds the rest "
                "(for h_norm, python -m method.backfill_h_norm)."
            )
        return [s.z[source][component] for s in self.steps]

    def has_latent(self, source: str = "base") -> bool:
        """Whether every checkpoint carries ``z``, so a full series exists."""
        return bool(self.steps) and all(source in s.z for s in self.steps)

    def datasets(self) -> list[str]:
        """The ``dataset/version`` trained on after each non-terminal checkpoint."""
        return [s.next_dataset for s in self.steps if s.next_dataset is not None]

    def probe_series(self, dataset: str, *, stat: str = "mean") -> list[float]:
        r"""$\Delta \hat{P}_t$ for one probe at every checkpoint, ordered by $t$.

        The series behind the drift plot: a fixed dataset measured against a
        moving model, so any change is the model's, not the data's. Raises
        ``KeyError`` if this trajectory did not probe ``dataset`` -- silently
        returning a short or empty series would show up as a truncated line
        rather than as the missing measurement it is.
        """
        missing = [s.t for s in self.steps if dataset not in s.probes]
        if missing:
            raise KeyError(
                f"trajectory {self.name!r} (seed {self.seed}) has no probe for "
                f"{dataset!r} at checkpoint(s) {missing}; add it to the config's "
                "`probes` and re-run to fill in the measurement"
            )
        return [s.probes[dataset][stat] for s in self.steps]

    def probed_datasets(self) -> list[str]:
        """Datasets probed at *every* checkpoint, so a full series exists."""
        if not self.steps:
            return []
        common = set(self.steps[0].probes)
        for step in self.steps[1:]:
            common &= set(step.probes)
        return sorted(common)


def load_trajectory(path: Path) -> Trajectory:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return Trajectory.from_dict(payload, source=Path(path))


def load_trajectory_set(paths: Iterable[Path]) -> list[Trajectory]:
    """Load several ``trajectory.json`` files, e.g. one per seed of the same design."""
    return [load_trajectory(p) for p in paths]


def to_frame(
    trajectories: Iterable[Trajectory], *, source: str = "base"
) -> pd.DataFrame:
    r"""Tidy long-format frame, one row per ``(seed, t)``, for ad-hoc analysis.

    The projection columns are named ``delta_p_hat_<stat>`` rather than after
    the record field they come from: :attr:`StepRecord.delta_p` is
    $\Delta \hat{P}_t$, and a column called ``delta_p_mean`` reads like the
    unapproximated $\Delta P_t$, which this frame does not carry.
    """
    rows = []
    for traj in trajectories:
        for step in traj.steps:
            row: dict[str, Any] = {
                "name": traj.name,
                "trait": traj.trait,
                "seed": traj.seed,
                "t": step.t,
                "weights_id": step.weights_id,
                "next_dataset": step.next_dataset,
                **{f"behavior_{k}": v for k, v in step.behavior.items()},
                **{f"z_{comp}": val for comp, val in step.z.get(source, {}).items()},
            }
            if step.delta_p is not None:
                row.update({f"delta_p_hat_{k}": v for k, v in step.delta_p.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def projection_pairs(
    trajectories: Iterable[Trajectory],
    delta_p_0: Mapping[str, float],
    *,
    stat: str = "mean",
) -> pd.DataFrame:
    r"""Pair each step's $\Delta \hat{P}_t$ with the dataset's $\Delta P_0$.

    The hatted series lands in ``delta_p_hat_t``: its axis and encoder are the
    checkpoint's, its predicted answers still $M_0$'s. There is deliberately no
    ``delta_p_t`` column here, because that name would read as the fully
    refreshed $\Delta P_t$, which no trajectory carries under the default view.

    ``delta_p_0`` maps ``"dataset/version"`` to the projection difference that
    dataset would have produced against the base model $M_0$ -- computable in
    advance for any dataset, per the action-encoder definition in the
    proposal, so it does not need to come from the same trajectory. Steps
    whose dataset is missing from ``delta_p_0`` are skipped. This is the data
    behind the RQ1 projection-correlation scatter
    (``figures.scatter_projection_correlation``).
    """
    rows = []
    for traj in trajectories:
        for step, nxt in zip(traj.steps, traj.steps[1:]):
            if step.delta_p is None or step.next_dataset is None:
                continue
            dataset = step.next_dataset
            if dataset not in delta_p_0:
                continue
            rows.append(
                {
                    "name": traj.name,
                    "seed": traj.seed,
                    "t": step.t,
                    "dataset": dataset,
                    "delta_p_0": delta_p_0[dataset],
                    "delta_p_hat_t": step.delta_p[stat],
                    "delta_behavior": (
                        nxt.behavior[traj.trait] - step.behavior[traj.trait]
                    ),
                }
            )
    return pd.DataFrame(rows)


def metric_pairs(
    trajectories: Iterable[Trajectory], component: str, *, source: str = "base"
) -> pd.DataFrame:
    r"""Pair a $z_t$ component with the behaviour change it precedes, $\Delta b_{t+1}$.

    Used for the "same thing with $p$, $q$, $\rho$ and $r$" scatters
    (``figures.scatter_metric_grid``).

    Trajectories carrying no latent series are skipped rather than raising, so a
    decay collection -- trunks measured in full alongside branches measured only
    at their endpoint -- contributes the trunks instead of failing outright.
    """
    rows = []
    for traj in trajectories:
        if not traj.has_latent(source):
            continue
        for step, nxt in zip(traj.steps, traj.steps[1:]):
            rows.append(
                {
                    "name": traj.name,
                    "seed": traj.seed,
                    "t": step.t,
                    "value": step.z[source][component],
                    "delta_behavior": (
                        nxt.behavior[traj.trait] - step.behavior[traj.trait]
                    ),
                }
            )
    return pd.DataFrame(rows)


def z_component_matrix(
    trajectories: Iterable[Trajectory], component: str, *, source: str = "base"
) -> list[list[float]]:
    """One list per trajectory of a $z_t$ component across the whole trajectory.

    Feeds the "how much has $\\rho$/$r$/$p$/$q$ drifted from step 0" line plot
    (``figures.drift_line``); rows may differ in length across trajectories of
    different lengths.

    Trajectories with no latent series contribute no row, mirroring
    :func:`probe_matrix`: a decay collection mixes trunks, which carry ``z`` at
    every checkpoint, with branches, which carry it nowhere by design.
    """
    return [
        traj.z_series(component, source=source)
        for traj in trajectories
        if traj.has_latent(source)
    ]


def probe_matrix(
    trajectories: Iterable[Trajectory], dataset: str, *, stat: str = "mean"
) -> list[list[float]]:
    r"""One row per trajectory of $\Delta \hat{P}_t$ for ``dataset`` across all $t$.

    Feeds :func:`method.visualization.figures.drift_line`, which normalises each
    row against its own step-0 value and draws mean $\pm$ 1 std across seeds.
    Trajectories that never probed ``dataset`` are skipped rather than raising,
    so a mixed set (e.g. exp2 alongside exp3 runs that carry no probes) still
    plots the seeds that do have the series.
    """
    rows = []
    for traj in trajectories:
        if dataset in traj.probed_datasets():
            rows.append(traj.probe_series(dataset, stat=stat))
    return rows


def delta_p_by_dataset(
    trajectories: Iterable[Trajectory], dataset: str, *, stat: str = "mean"
) -> list[list[float]]:
    r"""Per-trajectory $\Delta \hat{P}_t$ values at every occurrence of ``dataset``.

    Occurrences are ordered by ``t``, so index 0 is always the first time a
    given trajectory trained on ``dataset`` and index 1 the second -- what
    the repeated-dataset designs in RQ2/RQ3 need to compare "first exposure"
    against "later exposure" regardless of the absolute step they land on.
    """
    out = []
    for traj in trajectories:
        values = [
            s.delta_p[stat]
            for s in traj.steps
            if s.next_dataset == dataset and s.delta_p is not None
        ]
        if values:
            out.append(values)
    return out
