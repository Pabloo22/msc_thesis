r"""Tests for the DeltaP *views*: which axis, and whose predicted answers.

DeltaP at checkpoint $t$ refreshes the persona vector and the activations
together and differences them against $M_0$'s answers. Each of those three can
be held at the base model instead, and the combinations worth measuring form a
ladder (:class:`method.config.DeltaPView`):

===========================  ==============================  ===============
view                         quantity                        cost
===========================  ==============================  ===============
default, at $t = 0$          $\Delta P_0$                     already paid
``axis=BASE``                $\Delta \hat{P}_t^{(\mathbf{v}_0)}$ free
default                      $\Delta \hat{P}_t$                already paid
``predicted=CURRENT``        $\Delta P_t$                     a generation pass per $t$
===========================  ==============================  ===============

These cover the measurement path (:mod:`method.steps`) and the two families
scoped to pay for the new views. The analysis side -- how a series reaches the
figures -- is in ``test_decay.py``.

Everything here runs on the mock backend: no model, no GPU, no judge.
"""

from __future__ import annotations

import dataclasses

import pytest

from method import experiments as E, steps
from method.backends import get_backend
from method.config import (
    Backend,
    DatasetVersion,
    DeltaPConfig,
    DeltaPView,
    PredictedSource,
    ProjectionAxis,
    StepConfig,
)
from method.store import Store, get_weights_id

GSM8K = StepConfig(
    dataset="mistake_gsm8k",
    version=DatasetVersion.MISALIGNED_2,
    n_examples=8,
    train=E.LOCAL_TRAIN,
)
EVIL = StepConfig(
    dataset="evil",
    version=DatasetVersion.NORMAL,
    n_examples=8,
    train=E.LOCAL_TRAIN,
)


@pytest.fixture
def backend():
    return get_backend(Backend.MOCK, dtype="float16")


#: The three views the design measures, by the name their artifacts take.
OWN_ANSWERS = DeltaPView(predicted=PredictedSource.CURRENT)
BASE_AXIS = DeltaPView(axis=ProjectionAxis.BASE)
DEFAULT = DeltaPView()


def make_cfg(*, steps_=(GSM8K, EVIL), probes=(), **delta_p):
    return dataclasses.replace(
        E.SMOKE_MOCK, steps=steps_, probes=probes, delta_p=DeltaPConfig(**delta_p)
    )


def prepared(cfg, store, backend, t=0):
    """A checkpoint DeltaP can be read at: its persona vector, and an adapter.

    $v^{(0)}$ is extracted too, since the base-axis view projects against it
    however deep the checkpoint being measured is. Nothing here trains: the
    mock adapter is what ``materialize`` replays to reach ``t``, so a test can
    measure at a later checkpoint without a run.
    """
    for step in range(1, t + 1):
        adapter = store.adapter_dir(get_weights_id(cfg, step))
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    for step in {0, t}:
        steps.extract_persona_vector(cfg, step, store, backend)


def delta_p(cfg, t, store, backend, view, step=GSM8K):
    train_file = steps.cached_training_sample(step, cfg.seed, store)
    return steps.compute_delta_p(
        cfg,
        t,
        store,
        backend,
        train_file,
        steps.training_sample_id(step, cfg.seed),
        view=view,
    )


# --- the config axis --------------------------------------------------------


class TestPredictedSource:
    def test_both_expands_to_the_two_real_sources(self):
        assert PredictedSource.BOTH.sources == (
            PredictedSource.BASE,
            PredictedSource.CURRENT,
        )

    def test_a_single_source_expands_to_itself(self):
        assert PredictedSource.CURRENT.sources == (PredictedSource.CURRENT,)

    def test_the_default_view_keeps_the_unqualified_record_key(self):
        """Every trajectory already on disk was written under these keys, and
        must stay readable now that other views exist."""
        assert DEFAULT.key("probes") == "probes"
        assert DEFAULT.key("delta_p") == "delta_p"

    def test_each_other_view_gets_its_own_record_key(self):
        assert BASE_AXIS.key("probes") == "probes_v0"
        assert OWN_ANSWERS.key("probes") == "probes_current"

    def test_the_axis_is_named_v0_not_base(self):
        """``base`` already means "$M_0$'s answers" on the other setting, and
        one artifact name should not use the same word for two things."""
        assert BASE_AXIS.suffix == "v0"

    def test_a_view_cannot_be_built_from_an_unexpanded_both(self):
        with pytest.raises(ValueError, match="no single view"):
            DeltaPView(axis=ProjectionAxis.BOTH)
        with pytest.raises(ValueError, match="no single view"):
            DeltaPView(predicted=PredictedSource.BOTH)

    def test_both_expands_through_the_config(self):
        assert [v.suffix for v in DeltaPConfig(axis=ProjectionAxis.BOTH).views] == [
            "",
            "v0",
        ]
        assert [
            v.suffix for v in DeltaPConfig(predicted=PredictedSource.BOTH).views
        ] == ["", "current"]

    def test_the_default_config_measures_exactly_one_view(self):
        """Both new views have to be asked for: one costs a generation pass per
        checkpoint, and the other is a measurement nothing before this needed."""
        assert DeltaPConfig().views == (DEFAULT,)

    def test_the_two_axes_are_independent(self):
        """``mode`` says which examples are projected and ``predicted`` says
        whose answers they are differenced against; neither constrains the
        other."""
        cfg = DeltaPConfig(
            mode=E.DeltaPMode.SAMPLE, n_samples=4, predicted=PredictedSource.CURRENT
        )
        assert (cfg.mode, cfg.n_samples, cfg.predicted) == (
            E.DeltaPMode.SAMPLE,
            4,
            PredictedSource.CURRENT,
        )

    def test_the_default_is_the_frozen_prediction_on_the_current_axis(self):
        """What every measurement in the project has always taken."""
        assert DeltaPConfig().predicted is PredictedSource.BASE
        assert DeltaPConfig().axis is ProjectionAxis.CURRENT


# --- the measurement --------------------------------------------------------


class TestCurrentSourceMeasurement:
    def test_the_two_sources_agree_at_t0(self, tmp_path, backend):
        """At ``t = 0`` the current model *is* $M_0$, so the recomputed series
        must start at exactly $\\Delta P_0$ rather than near it."""
        store = Store(tmp_path)
        cfg = make_cfg(predicted=PredictedSource.CURRENT)
        prepared(cfg, store, backend)

        base = delta_p(cfg, 0, store, backend, DEFAULT)
        current = delta_p(cfg, 0, store, backend, OWN_ANSWERS)

        assert base == current

    def test_t0_reuses_the_base_answers_rather_than_generating_again(
        self, tmp_path, backend
    ):
        """Generation is greedy, so a second pass at ``t = 0`` would return the
        same text at full cost. Sharing the file is what makes the agreement
        above exact instead of merely expected."""
        store = Store(tmp_path)
        cfg = make_cfg(predicted=PredictedSource.CURRENT)
        prepared(cfg, store, backend)

        delta_p(cfg, 0, store, backend, OWN_ANSWERS)

        written = sorted(
            p.name
            for p in store.measurement_dir(get_weights_id(cfg, 0)).glob("*answers_*")
        )
        assert all(name.startswith("base_answers_") for name in written), written

    def test_the_sources_differ_once_the_model_has_moved(self, tmp_path, backend):
        """The whole point: past ``t = 0`` the checkpoint no longer says what
        $M_0$ said, so the shift the data asks it to make is a different
        number."""
        store = Store(tmp_path)
        cfg = make_cfg(predicted=PredictedSource.CURRENT)
        prepared(cfg, store, backend, t=1)

        base = delta_p(cfg, 1, store, backend, DEFAULT)
        current = delta_p(cfg, 1, store, backend, OWN_ANSWERS)

        assert base != current

    def test_current_answers_are_keyed_to_the_checkpoint(self, tmp_path, backend):
        """Base answers belong to $M_0$ and are generated once for the whole
        trajectory; current answers belong to the checkpoint, which is what
        makes the series cost a generation pass per ``t``."""
        store = Store(tmp_path)
        cfg = make_cfg(predicted=PredictedSource.CURRENT)
        prepared(cfg, store, backend, t=1)

        delta_p(cfg, 1, store, backend, OWN_ANSWERS)

        at_1 = list(
            store.measurement_dir(get_weights_id(cfg, 1)).glob("current_answers_*")
        )
        at_0 = list(
            store.measurement_dir(get_weights_id(cfg, 0)).glob("current_answers_*")
        )
        assert len(at_1) == 1 and not at_0

    def test_neither_source_overwrites_the_other(self, tmp_path, backend):
        """One checkpoint holds both, so they need distinct artifacts -- an
        unqualified name would make the second read back the first's numbers."""
        store = Store(tmp_path)
        cfg = make_cfg(predicted=PredictedSource.CURRENT)
        prepared(cfg, store, backend, t=1)

        delta_p(cfg, 1, store, backend, DEFAULT)
        delta_p(cfg, 1, store, backend, OWN_ANSWERS)

        written = sorted(
            p.name
            for p in store.trait_measurement_dir(
                get_weights_id(cfg, 1), cfg.trait
            ).glob("delta_p_*.json")
        )
        assert len(written) == 2, written

    def test_base_artifacts_keep_the_names_they_always_had(self, tmp_path, backend):
        """The store holds months of measurements under these names; a rename
        would silently recompute every one of them."""
        store = Store(tmp_path)
        cfg = make_cfg(predicted=PredictedSource.CURRENT)
        prepared(cfg, store, backend, t=1)
        sample_id = steps.training_sample_id(GSM8K, cfg.seed)

        delta_p(cfg, 1, store, backend, DEFAULT)

        wid = get_weights_id(cfg, 1)
        assert store.trait_measurement(
            wid, cfg.trait, f"delta_p_{sample_id}.json"
        ).exists()
        assert (
            store.measurement_dir(wid) / "delta_p_predicted" / sample_id
        ).is_dir()

    def test_the_result_is_cached_per_source(self, tmp_path, backend):
        store = Store(tmp_path)
        cfg = make_cfg(predicted=PredictedSource.CURRENT)
        prepared(cfg, store, backend, t=1)

        first = delta_p(cfg, 1, store, backend, OWN_ANSWERS)
        second = delta_p(cfg, 1, store, backend, OWN_ANSWERS)

        assert first == second

    def test_probes_carry_the_source_through(self, tmp_path, backend):
        store = Store(tmp_path)
        cfg = make_cfg(probes=(EVIL,), predicted=PredictedSource.CURRENT)
        prepared(cfg, store, backend, t=1)

        base = steps.measure_probes(cfg, 1, store, backend, view=DEFAULT)
        current = steps.measure_probes(cfg, 1, store, backend, view=OWN_ANSWERS)

        assert set(base) == set(current) == {EVIL.dataset_id}
        assert base[EVIL.dataset_id] != current[EVIL.dataset_id]


class TestBaseAxisView:
    r"""$\Delta \hat{P}_t^{(\mathbf{v}_0)}$: checkpoint activations against
    $M_0$'s axis."""

    def test_it_agrees_with_the_default_view_at_t0(self, tmp_path, backend):
        """At $t = 0$ the checkpoint's own axis *is* $v^{(0)}$, so the two views
        are the same measurement and must not merely be close."""
        store = Store(tmp_path)
        cfg = make_cfg(axis=ProjectionAxis.BASE)
        prepared(cfg, store, backend)

        assert delta_p(cfg, 0, store, backend, DEFAULT) == delta_p(
            cfg, 0, store, backend, BASE_AXIS
        )

    def test_it_differs_once_the_axis_has_rotated(self, tmp_path, backend):
        """The whole point: past $t = 0$, $v^{(t)}$ is a different direction, so
        holding the axis still is a different projection."""
        store = Store(tmp_path)
        cfg = make_cfg(axis=ProjectionAxis.BASE)
        prepared(cfg, store, backend, t=1)

        assert delta_p(cfg, 1, store, backend, DEFAULT) != delta_p(
            cfg, 1, store, backend, BASE_AXIS
        )

    def test_it_reuses_the_activations_rather_than_recomputing_them(
        self, tmp_path, backend
    ):
        """What makes this view free: the activations do not depend on the
        axis, so a checkpoint already measured needs no forward pass to yield
        it -- only a second projection of tensors already on disk."""
        store = Store(tmp_path)
        cfg = make_cfg(axis=ProjectionAxis.BASE)
        prepared(cfg, store, backend, t=1)
        delta_p(cfg, 1, store, backend, DEFAULT)

        before = _hidden_state_mtimes(store, cfg)
        delta_p(cfg, 1, store, backend, BASE_AXIS)

        assert _hidden_state_mtimes(store, cfg) == before

    def test_it_loads_no_model_when_the_activations_are_cached(
        self, tmp_path, backend, monkeypatch
    ):
        """Materialising replays the whole adapter chain onto disk. A view that
        needs no forward pass must not pay for one, or the backfill stops being
        free the moment it runs on a box with a cold merge cache."""
        store = Store(tmp_path)
        cfg = make_cfg(axis=ProjectionAxis.BASE)
        prepared(cfg, store, backend, t=1)
        delta_p(cfg, 1, store, backend, DEFAULT)

        monkeypatch.setattr(
            "method.steps.materialize",
            lambda *a, **k: pytest.fail("materialized a checkpoint for a re-projection"),
        )
        assert delta_p(cfg, 1, store, backend, BASE_AXIS)["n"] > 0

    def test_it_does_not_overwrite_the_default_view(self, tmp_path, backend):
        store = Store(tmp_path)
        cfg = make_cfg(axis=ProjectionAxis.BASE)
        prepared(cfg, store, backend, t=1)

        delta_p(cfg, 1, store, backend, DEFAULT)
        delta_p(cfg, 1, store, backend, BASE_AXIS)

        written = sorted(
            path.name
            for path in store.trait_measurement_dir(
                get_weights_id(cfg, 1), cfg.trait
            ).glob("delta_p_*.json")
        )
        assert len(written) == 2, written

    def test_probes_carry_the_view_through(self, tmp_path, backend):
        store = Store(tmp_path)
        cfg = make_cfg(probes=(EVIL,), axis=ProjectionAxis.BASE)
        prepared(cfg, store, backend, t=1)

        current = steps.measure_probes(cfg, 1, store, backend, view=DEFAULT)
        base_axis = steps.measure_probes(cfg, 1, store, backend, view=BASE_AXIS)

        assert current[EVIL.dataset_id] != base_axis[EVIL.dataset_id]


def _hidden_state_mtimes(store, cfg) -> dict[str, float]:
    """When every cached activation tensor under this store was last written."""
    return {
        str(path): path.stat().st_mtime
        for path in store.root.rglob("samples_layer*.pt")
    }


# --- the families that pay for the new views -------------------------------


class TestRegenFamily:
    def test_it_replays_the_decay_trunks_rather_than_training_new_ones(self):
        """Identical in everything ``weights_key`` hashes, so every checkpoint
        resolves to an adapter the decay family already trained."""
        regen = {
            (c.trait, c.label_map["trunk"]): c for c in E.build_exp2_regen_configs()
        }
        trunks = {
            (c.trait, c.label_map["trunk"]): c
            for c in E.build_exp2_decay_configs()
            if c.label_map.get("role") == "trunk"
        }
        assert set(regen) == set(trunks)
        for key, cfg in regen.items():
            trunk = trunks[key]
            assert [get_weights_id(cfg, t) for t in range(len(cfg.steps) + 1)] == [
                get_weights_id(trunk, t) for t in range(len(trunk.steps) + 1)
            ]

    def test_it_writes_its_own_trajectory(self):
        """Same weights, different name: overwriting the decay trunk's record
        would cost the frozen series it is meant to be compared against."""
        names = {c.name for c in E.build_exp2_regen_configs()}
        trunk_names = {
            c.name
            for c in E.build_exp2_decay_configs()
            if c.label_map.get("role") == "trunk"
        }
        assert not (names & trunk_names)

    def test_it_asks_for_the_recomputed_source_only(self):
        """The frozen series for this trunk is already measured and plotted;
        asking again here would re-read it under a second name."""
        for cfg in E.build_exp2_regen_configs():
            assert cfg.delta_p.predicted is PredictedSource.CURRENT

    def test_it_probes_the_same_datasets_as_the_decay_family(self):
        """The two series are read against each other point for point, so they
        have to be the same points."""
        for cfg in E.build_exp2_regen_configs():
            assert [p.dataset_id for p in cfg.probes] == [
                p.dataset_id for p in E.EXP2_PROBES
            ]

    def test_every_trunk_is_reachable(self):
        """Which trunks to actually pay for is a per-run decision (the
        ``--trunks`` filter reads this label), and a trunk that was never
        emitted could not be selected or reported missing."""
        configs = E.build_exp2_regen_configs()
        assert {c.label_map["trunk"] for c in configs} == set(E.EXP2_TRUNKS)

    def test_a_narrowed_trunk_set_emits_only_those(self):
        configs = E.build_exp2_regen_configs(
            trunks={"a": E.EXP2_TRUNKS["a"]}, probes=E.EXP2_PROBES
        )
        assert {c.label_map["trunk"] for c in configs} == {"a"}

    def test_it_runs_at_one_seed(self):
        """This varies how a fixed checkpoint is measured, not which checkpoint
        is reached, so a second seed answers a different question."""
        assert {c.seed for c in E.build_exp2_regen_configs()} == {E.EXP2_SEED}

    def test_it_emits_no_branches(self):
        r"""$\Delta b_{t+1}$ is a property of the probe fine-tune, not of how
        the projection was measured, so the decay family's branches are the
        y-axis here too."""
        assert all(
            c.label_map["role"] == "trunk" for c in E.build_exp2_regen_configs()
        )

    def test_a_probe_that_is_also_a_driver_is_rejected(self):
        """Section 3b holds here too: by the later checkpoints the model has
        trained on it, so its DeltaP measures memorisation."""
        with pytest.raises(ValueError, match="also a probe|probes"):
            E.build_exp2_regen_configs(probes=(E.EXP2_TRUNKS["a"][0],))

    def test_it_is_collectable_as_its_own_family(self):
        assert E.GROUP_BUILDERS[E.EXP2_REGEN] is E.build_exp2_regen_configs

    def test_the_local_variant_keeps_its_subsample(self):
        """``predicted`` overrides one axis of the DeltaP config; the scale
        preset owns the other, and a laptop run must keep it."""
        paper = E.build_exp2_regen_configs()[0]
        local = E.build_exp2_regen_configs(local=True)[0]
        assert local.delta_p.predicted is PredictedSource.CURRENT
        assert (local.delta_p.mode, local.delta_p.n_samples) != (
            paper.delta_p.mode,
            paper.delta_p.n_samples,
        )
