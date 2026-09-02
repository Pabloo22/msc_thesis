r"""Tests for the DeltaP *views*: which axis, and whose predicted answers.

DeltaP at checkpoint $t$ refreshes the persona vector and the activations
together and differences them against $M_0$'s answers. The axis has three
levels and the answers two, and all six combinations are measured
(:class:`method.config.DeltaPView`):

====================================  =========================================
view                                  quantity
====================================  =========================================
default, at $t = 0$                   $\Delta P_0$
``axis=BASE``                         $\Delta \hat{P}_t^{(\mathbf{v}_0)}$
default                               $\Delta \hat{P}_t$
``axis=ONPOLICY``                     $\Delta P_t^{t\leftarrow t,[0]}$
``predicted=CURRENT``                 $\Delta P_t$
``axis=BASE, predicted=CURRENT``      $\Delta P_t^{(\mathbf{v}_0)}$
``axis=ONPOLICY, predicted=CURRENT``  $\Delta P_t^{t\leftarrow t,[t]}$
====================================  =========================================

Only ``predicted=CURRENT`` costs a generation pass per checkpoint. The base
axis is free in both rows that use it, since the activations do not depend on
it, so the fourth view is free wherever the third has run. ``ONPOLICY`` is free
too, but only where :mod:`method.axis_refresh` has already drawn the vector: it
is the one axis whose *extraction text* is not $M_0$'s, and nothing else in the
repo produces one.

These cover the measurement path (:mod:`method.steps`) and the families scoped
to pay for the new views. The analysis side -- how a series reaches the
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


#: The six views the design measures, by the name their artifacts take.
OWN_ANSWERS = DeltaPView(predicted=PredictedSource.CURRENT)
BASE_AXIS = DeltaPView(axis=ProjectionAxis.BASE)
BASE_AXIS_OWN_ANSWERS = DeltaPView(
    axis=ProjectionAxis.BASE, predicted=PredictedSource.CURRENT
)
OWN_AXIS = DeltaPView(axis=ProjectionAxis.ONPOLICY)
OWN_AXIS_OWN_ANSWERS = DeltaPView(
    axis=ProjectionAxis.ONPOLICY, predicted=PredictedSource.CURRENT
)
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
        assert BASE_AXIS_OWN_ANSWERS.key("probes") == "probes_v0_current"
        assert OWN_AXIS.key("probes") == "probes_onpolicy"
        assert OWN_AXIS_OWN_ANSWERS.key("probes") == "probes_onpolicy_current"

    def test_no_view_name_is_a_prefix_of_another(self):
        """The suffixes are parsed by eye off artifact and record names, so
        one that swallowed another would make two views indistinguishable in a
        directory listing."""
        names = [
            view.suffix
            for view in (
                BASE_AXIS,
                OWN_ANSWERS,
                BASE_AXIS_OWN_ANSWERS,
                OWN_AXIS,
                OWN_AXIS_OWN_ANSWERS,
            )
        ]
        assert len(set(names)) == len(names)

    def test_the_redrawn_axis_says_which_knob_it_moved(self):
        """``current`` is taken twice over -- by the default axis and by the
        other setting's answers -- so the third level spells itself out."""
        assert OWN_AXIS.suffix == "onpolicy"
        assert OWN_AXIS_OWN_ANSWERS.suffix == "onpolicy_current"

    def test_the_combined_view_concatenates_the_two_parts(self):
        """The settings are independent, so the name that refreshes both is
        the two names joined -- axis first -- rather than a fourth word
        nothing could derive from the settings."""
        assert BASE_AXIS_OWN_ANSWERS.suffix == "v0_current"

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

    def test_the_redrawn_axis_is_never_reached_by_expanding_both(self):
        """``BOTH`` is the pair of axes a re-projection buys for free.
        ``ONPOLICY`` needs a whole extraction draw first, so a config that
        asked for "both" and silently got it would spend that without being
        asked."""
        assert ProjectionAxis.ONPOLICY not in ProjectionAxis.BOTH.axes

    def test_both_on_both_settings_expands_to_the_whole_square(self):
        """The views are a 2x2, so the cross product is four corners rather
        than three rungs and one combination nothing measures."""
        cfg = DeltaPConfig(
            axis=ProjectionAxis.BOTH, predicted=PredictedSource.BOTH
        )
        assert [v.suffix for v in cfg.views] == [
            "",
            "current",
            "v0",
            "v0_current",
        ]

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


class TestBaseAxisOwnAnswersView:
    r"""$\Delta P_t^{(\mathbf{v}_0)}$: the checkpoint answering for itself,
    projected onto $M_0$'s axis.

    The fourth corner of the square. It is what makes the axis and the answers
    separable: without it the only path from
    $\Delta \hat{P}_t^{(\mathbf{v}_0)}$ to $\Delta P_t$ moves both at once.
    """

    def test_every_view_agrees_at_t0(self, tmp_path, backend):
        """At $t = 0$ the checkpoint *is* $M_0$ and its axis *is* $v^{(0)}$, so
        all four corners collapse onto $\\Delta P_0$ -- and must do so exactly,
        since that is the point the whole square is read against."""
        store = Store(tmp_path)
        cfg = make_cfg(
            axis=ProjectionAxis.BASE, predicted=PredictedSource.CURRENT
        )
        prepared(cfg, store, backend)

        at_0 = delta_p(cfg, 0, store, backend, DEFAULT)
        assert delta_p(cfg, 0, store, backend, BASE_AXIS) == at_0
        assert delta_p(cfg, 0, store, backend, OWN_ANSWERS) == at_0
        assert delta_p(cfg, 0, store, backend, BASE_AXIS_OWN_ANSWERS) == at_0

    def test_it_differs_from_both_of_its_neighbours(self, tmp_path, backend):
        """One setting apart from each: refreshing the answers is what
        separates it from the base-axis view, and rotating the axis is what
        separates it from the re-answered one."""
        store = Store(tmp_path)
        cfg = make_cfg(
            axis=ProjectionAxis.BASE, predicted=PredictedSource.CURRENT
        )
        prepared(cfg, store, backend, t=1)

        both = delta_p(cfg, 1, store, backend, BASE_AXIS_OWN_ANSWERS)

        assert both != delta_p(cfg, 1, store, backend, BASE_AXIS)
        assert both != delta_p(cfg, 1, store, backend, OWN_ANSWERS)

    def test_it_reuses_the_regenerated_answers(self, tmp_path, backend):
        """What makes this view free once the regen family has run: the
        answers and their hidden states are cached per checkpoint and dataset
        and do not depend on the axis, so it re-projects tensors on disk."""
        store = Store(tmp_path)
        cfg = make_cfg(
            axis=ProjectionAxis.BASE, predicted=PredictedSource.CURRENT
        )
        prepared(cfg, store, backend, t=1)
        delta_p(cfg, 1, store, backend, OWN_ANSWERS)

        answers = sorted(p.name for p in store.root.rglob("current_answers_*.jsonl"))
        before = _hidden_state_mtimes(store, cfg)

        delta_p(cfg, 1, store, backend, BASE_AXIS_OWN_ANSWERS)

        assert _hidden_state_mtimes(store, cfg) == before
        assert (
            sorted(p.name for p in store.root.rglob("current_answers_*.jsonl"))
            == answers
        )

    def test_it_loads_no_model_once_the_regen_family_has_run(
        self, tmp_path, backend, monkeypatch
    ):
        """A generation pass here would make the fourth corner cost as much as
        the third, which is the whole argument for measuring it."""
        store = Store(tmp_path)
        cfg = make_cfg(
            axis=ProjectionAxis.BASE, predicted=PredictedSource.CURRENT
        )
        prepared(cfg, store, backend, t=1)
        delta_p(cfg, 1, store, backend, OWN_ANSWERS)

        monkeypatch.setattr(
            "method.steps.materialize",
            lambda *a, **k: pytest.fail("materialized a checkpoint for a re-projection"),
        )
        assert delta_p(cfg, 1, store, backend, BASE_AXIS_OWN_ANSWERS)["n"] > 0

    def test_it_overwrites_none_of_the_other_three(self, tmp_path, backend):
        """One checkpoint holds all four, so each needs its own artifact -- a
        shared name would make one read back another's numbers."""
        store = Store(tmp_path)
        cfg = make_cfg(
            axis=ProjectionAxis.BASE, predicted=PredictedSource.CURRENT
        )
        prepared(cfg, store, backend, t=1)

        for view in (DEFAULT, BASE_AXIS, OWN_ANSWERS, BASE_AXIS_OWN_ANSWERS):
            delta_p(cfg, 1, store, backend, view)

        written = sorted(
            path.name
            for path in store.trait_measurement_dir(
                get_weights_id(cfg, 1), cfg.trait
            ).glob("delta_p_*.json")
        )
        assert len(written) == 4, written

    def test_probes_carry_the_view_through(self, tmp_path, backend):
        store = Store(tmp_path)
        cfg = make_cfg(
            probes=(EVIL,),
            axis=ProjectionAxis.BASE,
            predicted=PredictedSource.CURRENT,
        )
        prepared(cfg, store, backend, t=1)

        own_answers = steps.measure_probes(cfg, 1, store, backend, view=OWN_ANSWERS)
        both = steps.measure_probes(
            cfg, 1, store, backend, view=BASE_AXIS_OWN_ANSWERS
        )

        assert set(own_answers) == set(both) == {EVIL.dataset_id}
        assert own_answers[EVIL.dataset_id] != both[EVIL.dataset_id]


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


class TestV0RegenFamily:
    r"""The family that closes the square: $\Delta P_t^{(\mathbf{v}_0)}$."""

    def test_it_asks_for_both_settings_at_once(self):
        """Neither on its own is this family: the axis alone is
        :func:`build_exp2_axis_configs` and the answers alone are
        :func:`build_exp2_regen_configs`."""
        for cfg in E.build_exp2_v0regen_configs():
            assert cfg.delta_p.axis is ProjectionAxis.BASE
            assert cfg.delta_p.predicted is PredictedSource.CURRENT

    def test_it_measures_exactly_the_fourth_corner(self):
        """One view, not the whole square: the other three are already
        measured and already plotted."""
        for cfg in E.build_exp2_v0regen_configs():
            assert [v.suffix for v in cfg.delta_p.views] == ["v0_current"]

    def test_it_replays_the_decay_trunks_rather_than_training_new_ones(self):
        v0regen = {
            (c.trait, c.label_map["trunk"]): c for c in E.build_exp2_v0regen_configs()
        }
        trunks = {
            (c.trait, c.label_map["trunk"]): c
            for c in E.build_exp2_decay_configs()
            if c.label_map.get("role") == "trunk"
        }
        assert set(v0regen) == set(trunks)
        for key, cfg in v0regen.items():
            trunk = trunks[key]
            assert [get_weights_id(cfg, t) for t in range(len(cfg.steps) + 1)] == [
                get_weights_id(trunk, t) for t in range(len(trunk.steps) + 1)
            ]

    def test_it_writes_its_own_trajectory(self):
        """Same weights as three other families; overwriting any of their
        records would cost the series it is meant to be read against."""
        names = {c.name for c in E.build_exp2_v0regen_configs()}
        others = {
            c.name
            for build in (
                E.build_exp2_decay_configs,
                E.build_exp2_axis_configs,
                E.build_exp2_regen_configs,
            )
            for c in build()
        }
        assert not (names & others)

    def test_its_family_prefix_selects_it_alone(self):
        """``run_family.sh`` matches registry keys by prefix, so a name the
        axis family's prefix also matched would run this family whenever that
        one was asked for."""
        for prefix, group in (
            (E.EXP2_AXIS, E.EXP2_AXIS),
            (E.EXP2_REGEN, E.EXP2_REGEN),
            (E.EXP2_V0REGEN, E.EXP2_V0REGEN),
        ):
            selected = {
                cfg.group
                for key, cfg in E.REGISTRY.items()
                if key.startswith(f"{prefix.upper()}_")
            }
            assert selected == {group}, prefix

    def test_it_probes_the_same_datasets_as_the_decay_family(self):
        for cfg in E.build_exp2_v0regen_configs():
            assert [p.dataset_id for p in cfg.probes] == [
                p.dataset_id for p in E.EXP2_PROBES
            ]

    def test_every_trunk_is_reachable(self):
        configs = E.build_exp2_v0regen_configs()
        assert {c.label_map["trunk"] for c in configs} == set(E.EXP2_TRUNKS)

    def test_it_runs_at_one_seed_and_emits_no_branches(self):
        configs = E.build_exp2_v0regen_configs()
        assert {c.seed for c in configs} == {E.EXP2_SEED}
        assert all(c.label_map["role"] == "trunk" for c in configs)

    def test_a_probe_that_is_also_a_driver_is_rejected(self):
        with pytest.raises(ValueError, match="also a probe|probes"):
            E.build_exp2_v0regen_configs(probes=(E.EXP2_TRUNKS["a"][0],))

    def test_it_is_collectable_as_its_own_family(self):
        assert E.GROUP_BUILDERS[E.EXP2_V0REGEN] is E.build_exp2_v0regen_configs

    def test_the_local_variant_keeps_its_subsample(self):
        paper = E.build_exp2_v0regen_configs()[0]
        local = E.build_exp2_v0regen_configs(local=True)[0]
        assert local.delta_p.axis is ProjectionAxis.BASE
        assert local.delta_p.predicted is PredictedSource.CURRENT
        assert (local.delta_p.mode, local.delta_p.n_samples) != (
            paper.delta_p.mode,
            paper.delta_p.n_samples,
        )


def draw_onpolicy_vector(cfg, t, store, *, scale: float = 1.0):
    r"""Stand in for what :mod:`method.axis_refresh` leaves in the bundle.

    The sweep generates a fresh extraction set from $M_t$, judges it, filters
    it and extracts a vector; none of that is what the views are being tested
    on. What matters here is only that a vector exists at the path
    ``ProjectionAxis.ONPOLICY`` reads, and whether it points the same way as
    the frozen one -- so it is derived from $v^{(t)}$ by a rotation ``scale``
    controls, with ``1.0`` meaning "the freeze cost nothing".
    """
    import torch

    frozen = torch.load(
        store.trait_measurement(
            get_weights_id(cfg, t), cfg.trait, steps.Artifacts.persona_vector(cfg.trait)
        ),
        weights_only=False,
    )
    path = steps.onpolicy_vector_path(store, get_weights_id(cfg, t), cfg.trait)
    path.parent.mkdir(parents=True, exist_ok=True)
    drawn = frozen if scale == 1.0 else frozen + scale * torch.flip(frozen, dims=(-1,))
    torch.save(drawn, path)
    return path


class TestOnPolicyAxisView:
    r"""$\Delta P_t^{t\leftarrow t}$: the axis drawn from the checkpoint's own text."""

    def test_a_missing_draw_names_the_sweep_that_produces_it(self, tmp_path, backend):
        """The one input no other family leaves behind. A bare file-not-found
        several frames down would read as a broken store rather than as a
        sweep that has not been run."""
        store = Store(tmp_path)
        cfg = make_cfg(axis=ProjectionAxis.ONPOLICY)
        prepared(cfg, store, backend)

        with pytest.raises(FileNotFoundError, match="axis_refresh"):
            delta_p(cfg, 0, store, backend, OWN_AXIS)

    def test_it_agrees_with_the_default_view_when_the_draw_agrees(
        self, tmp_path, backend
    ):
        """If re-drawing the extraction text yields the same direction, the
        freeze cost nothing and the two views are one measurement -- which is
        the null this whole family exists to test against."""
        store = Store(tmp_path)
        cfg = make_cfg(axis=ProjectionAxis.ONPOLICY)
        prepared(cfg, store, backend, t=1)
        draw_onpolicy_vector(cfg, 1, store)

        assert delta_p(cfg, 1, store, backend, DEFAULT) == delta_p(
            cfg, 1, store, backend, OWN_AXIS
        )

    def test_it_differs_once_the_drawn_axis_points_elsewhere(self, tmp_path, backend):
        """And if it does not agree, the number the freeze reports is not the
        number the paper's own procedure would."""
        store = Store(tmp_path)
        cfg = make_cfg(axis=ProjectionAxis.ONPOLICY)
        prepared(cfg, store, backend, t=1)
        draw_onpolicy_vector(cfg, 1, store, scale=0.5)

        assert delta_p(cfg, 1, store, backend, DEFAULT) != delta_p(
            cfg, 1, store, backend, OWN_AXIS
        )

    def test_it_reads_the_draw_from_the_checkpoint_being_measured(
        self, tmp_path, backend
    ):
        """Not from $M_0$'s bundle, which is where the *frozen* text lives. A
        view that read the base draw would be measuring the base model's own
        axis under a name that promises the checkpoint's."""
        store = Store(tmp_path)
        cfg = make_cfg(axis=ProjectionAxis.ONPOLICY)
        prepared(cfg, store, backend, t=1)
        draw_onpolicy_vector(cfg, 0, store, scale=0.5)

        with pytest.raises(FileNotFoundError, match="t=1"):
            delta_p(cfg, 1, store, backend, OWN_AXIS)

    def test_it_needs_no_forward_pass_where_the_activations_are_cached(
        self, tmp_path, backend, monkeypatch
    ):
        """Free for the same reason the base axis is: the activations do not
        depend on which direction they are projected onto. The draw is what
        costs, and the sweep has already paid for it."""
        store = Store(tmp_path)
        cfg = make_cfg(axis=ProjectionAxis.ONPOLICY)
        prepared(cfg, store, backend, t=1)
        delta_p(cfg, 1, store, backend, DEFAULT)
        draw_onpolicy_vector(cfg, 1, store, scale=0.5)

        before = _hidden_state_mtimes(store, cfg)
        monkeypatch.setattr(
            "method.steps.materialize",
            lambda *a, **k: pytest.fail(
                "materialized a checkpoint for a re-projection"
            ),
        )

        assert delta_p(cfg, 1, store, backend, OWN_AXIS)["n"] > 0
        assert _hidden_state_mtimes(store, cfg) == before

    def test_it_does_not_overwrite_the_default_view(self, tmp_path, backend):
        """Each view owns its own artifact names, so a checkpoint can carry all
        six at once."""
        store = Store(tmp_path)
        cfg = make_cfg(axis=ProjectionAxis.ONPOLICY)
        prepared(cfg, store, backend, t=1)
        draw_onpolicy_vector(cfg, 1, store, scale=0.5)

        default = delta_p(cfg, 1, store, backend, DEFAULT)
        drawn = delta_p(cfg, 1, store, backend, OWN_AXIS)

        assert delta_p(cfg, 1, store, backend, DEFAULT) == default
        assert drawn != default


class TestOnPolicyFamilies:
    def test_both_families_take_the_redrawn_axis(self):
        for cfg in E.build_exp2_onpolicy_configs():
            assert cfg.delta_p.axis is ProjectionAxis.ONPOLICY
            assert cfg.delta_p.predicted is PredictedSource.BASE
        for cfg in E.build_exp2_onpolicy_regen_configs():
            assert cfg.delta_p.axis is ProjectionAxis.ONPOLICY
            assert cfg.delta_p.predicted is PredictedSource.CURRENT

    def test_they_replay_the_decay_trunks_rather_than_training(self):
        """Identical in everything ``weights_key`` hashes, so every checkpoint
        is one the decay family already trained."""
        trunks = {
            cfg.label_map["trunk"]: cfg
            for cfg in E.build_exp2_decay_configs()
            if cfg.label_map["role"] == "trunk" and cfg.trait == "evil"
        }
        for cfg in E.build_exp2_onpolicy_configs(measure_traits=("evil",)):
            decayed = trunks[cfg.label_map["trunk"]]
            assert [get_weights_id(cfg, t) for t in range(len(cfg.steps) + 1)] == [
                get_weights_id(decayed, t) for t in range(len(decayed.steps) + 1)
            ]

    def test_each_writes_its_own_trajectory(self):
        names = {cfg.name for cfg in E.build_exp2_onpolicy_configs()} | {
            cfg.name for cfg in E.build_exp2_onpolicy_regen_configs()
        }
        assert len(names) == 12

    def test_they_are_collectable_as_their_own_families(self):
        assert E.GROUP_BUILDERS[E.EXP2_ONPOLICY] is E.build_exp2_onpolicy_configs
        assert (
            E.GROUP_BUILDERS[E.EXP2_ONPOLICY_REGEN]
            is E.build_exp2_onpolicy_regen_configs
        )
