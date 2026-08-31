r"""Tests for whose answers ``h_neutral`` -- and therefore $z_t$ -- is taken over.

$z_t$ is read off the mean hidden state over a fixed set of trait-neutral
prompts, and "over the prompts" hides a choice the notation does not record:
which model *answered* them.

===========================  =====================================  ==============
source                       answers the activation is taken over   measured by
===========================  =====================================  ==============
``HNeutralSource.BASE``      $M_0$'s, generated once at $t = 0$      every family
``HNeutralSource.CURRENT``   $M_t$'s own, regenerated at every $t$   ``exp2_hregen``
===========================  =====================================  ==============

Two things have to hold for the second to be readable beside the first. The
answers must really come from the checkpoint being measured, and adding the new
source must not disturb the frozen one that is already measured and plotted --
both sources share a single ``latent_cosine.json``, which is keyed by checkpoint
and trait and says nothing about which source wrote it.

Everything here runs on the mock backend: no model, no GPU, no judge. Two mock
properties shape the assertions -- ``generate_answers`` stamps the model path
into every answer, which is what lets a test see *who* answered, and
``hidden_states`` seeds on the answers' full path, so the two sources give
genuinely different activations rather than coinciding by construction.
"""

from __future__ import annotations

import dataclasses
import json

import pytest
import torch

from method import experiments as E, steps
from method.backends import get_backend
from method.config import Backend, HNeutralSource
from method.steps import Artifacts
from method.store import Store, get_weights_id


@pytest.fixture
def backend():
    return get_backend(Backend.MOCK, dtype="float16")


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path)


BASE_ONLY = E.SMOKE_MOCK
CURRENT_ONLY = dataclasses.replace(
    BASE_ONLY,
    latent=dataclasses.replace(
        BASE_ONLY.latent, h_neutral_source=HNeutralSource.CURRENT
    ),
)
BOTH = dataclasses.replace(
    BASE_ONLY,
    latent=dataclasses.replace(BASE_ONLY.latent, h_neutral_source=HNeutralSource.BOTH),
)


def prepared(cfg, store, backend, t=0):
    """A checkpoint $z_t$ can be read at: adapters to replay, and its vectors.

    Nothing here trains -- the mock adapter is what ``materialize`` replays --
    so a test can measure at a later checkpoint without a run.
    """
    for step in range(1, t + 1):
        adapter = store.adapter_dir(get_weights_id(cfg, step))
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    for step in {0, t}:
        steps.extract_persona_vector(cfg, step, store, backend)


def answers(cfg, t, store):
    """The neutral answers recorded against checkpoint ``t``."""
    path = store.measurement(get_weights_id(cfg, t), Artifacts.NEUTRAL_ANSWERS)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def answered_by(cfg, t, store):
    """The model path the mock stamped into checkpoint ``t``'s neutral answers."""
    return {row["messages"][-1]["content"] for row in answers(cfg, t, store)}


def cached_latent(cfg, t, store):
    return json.loads(
        store.trait_measurement(
            get_weights_id(cfg, t), cfg.trait, Artifacts.LATENT_JSON
        ).read_text()
    )


# --- the config axis --------------------------------------------------------


class TestHNeutralSource:
    def test_both_expands_to_the_two_real_sources(self):
        assert HNeutralSource.BOTH.sources == ("base", "current")

    def test_a_single_source_expands_to_itself(self):
        assert HNeutralSource.CURRENT.sources == ("current",)

    def test_the_default_is_the_frozen_one(self):
        """Every trajectory on disk was measured this way, and the frozen
        reading is the one the existing figures are drawn from."""
        assert BASE_ONLY.latent.h_neutral_source is HNeutralSource.BASE

    def test_the_two_sources_are_stored_apart(self):
        assert Artifacts.h_neutral("base") != Artifacts.h_neutral("current")


# --- the measurement path ---------------------------------------------------


class TestWhoAnswersTheNeutralPrompts:
    def test_the_frozen_source_reads_the_base_model_at_every_checkpoint(
        self, store, backend
    ):
        """The whole point of ``base``: the text is held still, so anything
        that moves in $p_t$ or $q_t$ is the encoder moving."""
        prepared(BASE_ONLY, store, backend, t=2)
        steps.measure_h_neutral(BASE_ONLY, 2, store, backend)

        assert answered_by(BASE_ONLY, 0, store) == {
            f"<mock answer {i} from {BASE_ONLY.model.name}>"
            for i in range(len(answers(BASE_ONLY, 0, store)))
        }
        # Nothing was written against the checkpoint itself: it only re-read.
        assert not store.measurement(
            get_weights_id(BASE_ONLY, 2), Artifacts.NEUTRAL_ANSWERS
        ).exists()

    def test_the_regenerated_source_has_the_checkpoint_answer_for_itself(
        self, store, backend
    ):
        """What the new variant buys: the answers are $M_t$'s own, so
        behavioural drift is inside the measurement rather than held out."""
        prepared(CURRENT_ONLY, store, backend, t=2)
        steps.measure_h_neutral(CURRENT_ONLY, 2, store, backend)

        stamped = answered_by(CURRENT_ONLY, 2, store)
        assert stamped
        assert all(CURRENT_ONLY.model.name not in answer for answer in stamped)
        assert all(
            get_weights_id(CURRENT_ONLY, 2) in answer for answer in stamped
        )

    def test_it_asks_the_neutral_prompt_set_for_the_answers(self, store, backend):
        """Not the trait's own eval questions: those would contaminate
        ``h_neutral`` with the very axis being measured."""
        prepared(CURRENT_ONLY, store, backend, t=1)
        steps.measure_h_neutral(CURRENT_ONLY, 1, store, backend)

        prompts = [
            json.loads(line)
            for line in steps.neutral_prompts_file(CURRENT_ONLY)
            .read_text()
            .splitlines()
            if line.strip()
        ][: CURRENT_ONLY.latent.n_neutral]
        asked = [row["messages"][:-1] for row in answers(CURRENT_ONLY, 1, store)]
        assert asked == [p["messages"] for p in prompts]

    def test_the_two_sources_coincide_at_the_base_checkpoint(self, store, backend):
        r"""At $t = 0$ the current model *is* $M_0$, so the regenerated series
        starts from exactly the frozen one's $z_0$ rather than an independent
        draw. Both paths resolve to one file, which makes that exact."""
        prepared(BOTH, store, backend)
        steps.measure_h_neutral(BOTH, 0, store, backend)

        z = steps.compute_step_latent(BOTH, 0, store)
        assert z["current"] == z["base"]

    def test_a_measured_source_is_not_regenerated(self, store, backend):
        """Resuming must not pay for the generation pass again."""
        prepared(CURRENT_ONLY, store, backend, t=1)
        steps.measure_h_neutral(CURRENT_ONLY, 1, store, backend)
        path = store.measurement(
            get_weights_id(CURRENT_ONLY, 1), Artifacts.NEUTRAL_ANSWERS
        )
        before = path.stat().st_mtime_ns

        steps.measure_h_neutral(CURRENT_ONLY, 1, store, backend)

        assert path.stat().st_mtime_ns == before

    def test_the_two_sources_give_different_activations(self, store, backend):
        """Otherwise the comparison the family exists for is unreadable. The
        mock seeds on the answers' full path, so this is a real difference of
        inputs and not of arithmetic."""
        prepared(BOTH, store, backend, t=1)
        steps.measure_h_neutral(BOTH, 1, store, backend)

        z = steps.compute_step_latent(BOTH, 1, store)
        assert z["base"] != z["current"]
        # Same v_t on both sides, so only the activation can differ.
        assert z["base"]["rho"] == z["current"]["rho"]
        assert z["base"]["r"] == z["current"]["r"]


# --- one artifact, two sources ----------------------------------------------


class TestAddingASourceLeavesTheOtherAlone:
    r"""``latent_cosine.json`` is keyed by checkpoint and trait, not by source.

    So the h-regen family writes into the *same* file the decay trunk already
    filled, at every checkpoint the two share. Overwriting it would cost the
    frozen series this variant exists to be compared against -- and that series
    is the one every existing figure is drawn from.
    """

    def test_the_frozen_block_survives_a_regenerated_measurement(
        self, store, backend
    ):
        prepared(BASE_ONLY, store, backend, t=1)
        steps.measure_h_neutral(BASE_ONLY, 1, store, backend)
        frozen = steps.compute_step_latent(BASE_ONLY, 1, store)["base"]

        steps.measure_h_neutral(CURRENT_ONLY, 1, store, backend)
        steps.compute_step_latent(CURRENT_ONLY, 1, store)

        assert cached_latent(BASE_ONLY, 1, store)["base"] == frozen

    def test_a_cached_source_is_read_back_rather_than_rederived(
        self, store, backend
    ):
        r"""Recomputing would re-anchor the entry onto whichever $v_0$ the store
        holds *now*, and exp3 is known to sit on several distinct base
        measurements (see ``method.visualization.latent_audit``). Adding a
        source is the one moment that risk is live, because the frozen block is
        sitting in the very file being rewritten."""
        prepared(BASE_ONLY, store, backend, t=1)
        steps.measure_h_neutral(BASE_ONLY, 1, store, backend)
        frozen = steps.compute_step_latent(BASE_ONLY, 1, store)["base"]

        # Re-draw v_0, as a second measurement pass over M_0 would: the anchor
        # is a measurement, not a property of the base model, so any
        # recomputation of "base" from here lands somewhere else.
        vector_path = store.trait_measurement(
            get_weights_id(BOTH, 0), BOTH.trait, Artifacts.persona_vector(BOTH.trait)
        )
        redrawn = torch.load(vector_path, weights_only=False)
        redrawn[BOTH.model.layer] = -2.0 * redrawn[BOTH.model.layer]
        torch.save(redrawn, vector_path)

        steps.measure_h_neutral(BOTH, 1, store, backend)
        widened = steps.compute_step_latent(BOTH, 1, store)

        assert widened["base"] == frozen
        assert cached_latent(BOTH, 1, store)["base"] == frozen
        # The new source *is* read against the anchor as it stands now, which
        # is what makes the one above a choice rather than an accident: the
        # re-drawn v_0 points the other way, so its rotation flips sign.
        assert widened["current"]["rho"] == pytest.approx(-frozen["rho"])

    def test_a_run_records_only_the_source_it_asked_for(self, store, backend):
        r"""``compute_step_latent``'s return is what lands in
        ``trajectory.json``. A decay trunk that silently gained a ``current``
        block from a *different* family's run would collide with that family in
        ``decay.trunk_series``, which indexes trunks by
        ``(trait, trunk, seed)`` and keeps only the first run it sees."""
        prepared(BOTH, store, backend, t=1)
        steps.measure_h_neutral(BOTH, 1, store, backend)
        steps.compute_step_latent(BOTH, 1, store)

        assert set(cached_latent(BOTH, 1, store)) == {"base", "current"}
        assert set(steps.compute_step_latent(BASE_ONLY, 1, store)) == {"base"}
        assert set(steps.compute_step_latent(CURRENT_ONLY, 1, store)) == {"current"}

    def test_widening_to_both_adds_without_recomputing(self, store, backend):
        prepared(BASE_ONLY, store, backend, t=1)
        steps.measure_h_neutral(BASE_ONLY, 1, store, backend)
        frozen = steps.compute_step_latent(BASE_ONLY, 1, store)["base"]

        steps.measure_h_neutral(BOTH, 1, store, backend)
        widened = steps.compute_step_latent(BOTH, 1, store)

        assert set(widened) == {"base", "current"}
        assert widened["base"] == frozen


# --- the family that pays for it --------------------------------------------


class TestHRegenFamily:
    def test_it_replays_the_decay_trunks_rather_than_training_new_ones(self):
        """Identical in everything ``weights_key`` hashes, so every checkpoint
        resolves to an adapter the decay family already trained."""
        hregen = {
            (c.trait, c.label_map["trunk"]): c for c in E.build_exp2_hregen_configs()
        }
        trunks = {
            (c.trait, c.label_map["trunk"]): c
            for c in E.build_exp2_decay_configs()
            if c.label_map.get("role") == "trunk"
        }
        assert set(hregen) == set(trunks)
        for key, cfg in hregen.items():
            trunk = trunks[key]
            assert [get_weights_id(cfg, t) for t in range(len(cfg.steps) + 1)] == [
                get_weights_id(trunk, t) for t in range(len(trunk.steps) + 1)
            ]

    def test_it_writes_its_own_trajectory(self):
        """Same weights, different name: overwriting the decay trunk's record
        would cost the frozen series it is meant to be compared against."""
        names = {c.name for c in E.build_exp2_hregen_configs()}
        trunk_names = {
            c.name
            for c in E.build_exp2_decay_configs()
            if c.label_map.get("role") == "trunk"
        }
        assert not (names & trunk_names)

    def test_it_asks_for_the_regenerated_source_only(self):
        """The frozen series for these trunks is already measured and plotted;
        asking again here would re-read it under a second name."""
        for cfg in E.build_exp2_hregen_configs():
            assert cfg.latent.h_neutral_source is HNeutralSource.CURRENT

    def test_it_leaves_the_projection_difference_alone(self):
        """One variant at a time: this family varies whose answers
        ``h_neutral`` is taken over, and DeltaP's own two knobs stay where the
        decay trunk left them so the two re-measurements stay separable."""
        for cfg in E.build_exp2_hregen_configs():
            assert cfg.delta_p.views == E.build_exp2_decay_configs()[0].delta_p.views

    def test_it_keeps_the_neutral_prompt_set_of_its_scale(self):
        """``h_neutral`` is only comparable across steps and families while
        every run reads the same prompts; the source says who answers them, not
        which they are."""
        for cfg in E.build_exp2_hregen_configs():
            assert cfg.latent.n_neutral == E.LatentConfig().n_neutral
            assert (
                cfg.latent.neutral_prompts_name
                == E.LatentConfig().neutral_prompts_name
            )
        for cfg in E.build_exp2_hregen_configs(local=True):
            assert cfg.latent.n_neutral == E.LOCAL_LATENT.n_neutral
            assert (
                cfg.latent.neutral_prompts_name == E.LOCAL_LATENT.neutral_prompts_name
            )

    def test_it_probes_the_same_datasets_as_the_decay_family(self):
        """Free on a box that holds the decay trunk's measurements, and it is
        what makes each run a complete decay row rather than a bare z series."""
        for cfg in E.build_exp2_hregen_configs():
            assert [p.dataset_id for p in cfg.probes] == [
                p.dataset_id for p in E.EXP2_PROBES
            ]

    def test_every_trunk_is_reachable(self):
        """``--trunks`` selects by this label, and a trunk that was never
        emitted could not be selected or reported missing."""
        configs = E.build_exp2_hregen_configs()
        assert {c.label_map["trunk"] for c in configs} == set(E.EXP2_TRUNKS)

    def test_a_narrowed_trunk_set_emits_only_those(self):
        configs = E.build_exp2_hregen_configs(
            trunks={"a": E.EXP2_TRUNKS["a"]}, probes=E.EXP2_PROBES
        )
        assert {c.label_map["trunk"] for c in configs} == {"a"}

    def test_it_runs_at_one_seed(self):
        """This varies how a fixed checkpoint is measured, not which checkpoint
        is reached, so a second seed answers a different question."""
        assert {c.seed for c in E.build_exp2_hregen_configs()} == {E.EXP2_SEED}

    def test_it_emits_no_branches(self):
        r"""$\Delta b_{t+1}$ is a property of the probe fine-tune, not of how
        ``h_neutral`` was taken, so the decay family's branches are the y-axis
        here too."""
        assert all(
            c.label_map["role"] == "trunk" for c in E.build_exp2_hregen_configs()
        )

    def test_a_probe_that_is_also_a_driver_is_rejected(self):
        """Section 3b holds here too: by the later checkpoints the model has
        trained on it, so its DeltaP measures memorisation."""
        with pytest.raises(ValueError, match="also a probe|probes"):
            E.build_exp2_hregen_configs(probes=(E.EXP2_TRUNKS["a"][0],))

    def test_it_is_collectable_as_its_own_family(self):
        assert E.GROUP_BUILDERS[E.EXP2_HREGEN] is E.build_exp2_hregen_configs

    def test_it_is_reachable_from_the_cli(self):
        registered = {k for k in E.REGISTRY if "HREGEN" in k}
        assert len(registered) == 2 * len(E.build_exp2_hregen_configs())
