"""Tests for the generated experiment configurations in :mod:`method.experiments`.

These assert the properties the experiment designs actually depend on: that
every step names a dataset that exists, that trajectories differing only in
which trait they *measure* share a fine-tuning chain, and that the designs
that are supposed to share a prefix really do. Nothing here loads a model.
"""

from __future__ import annotations

import dataclasses

import pytest

from method import experiments as E
from method.steps import dataset_path
from method.store import Store, get_weights_id

ALL_BUILDERS = (
    E.build_exp2_configs,
    E.build_hysteresis_configs,
    E.build_diversity_configs,
)


@pytest.fixture(scope="module")
def all_configs():
    # Both scales: the registry carries the paper-scale configs and their
    # ``_local`` variants, so laptop/mock runs are reachable from the CLI.
    return [
        cfg
        for build in ALL_BUILDERS
        for local in (False, True)
        for cfg in build(local=local)
    ]


class TestRegistry:
    def test_registry_builds_and_includes_every_generated_config(self, all_configs):
        assert len(E.REGISTRY) == len(all_configs) + 3  # + the 3 hand-written ones
        for key in ("SMOKE_MOCK", "SMOKE_TINY", "EXP1"):
            assert key in E.REGISTRY

    def test_lookup_error_names_alternatives(self):
        with pytest.raises(KeyError, match="unknown config"):
            E.get_trajectory_config("NOPE")

    def test_every_registry_key_round_trips(self):
        for key, cfg in E.REGISTRY.items():
            assert E.get_trajectory_config(key) is cfg


class TestDatasetsExist:
    def test_every_step_names_a_real_dataset_file(self, all_configs):
        for cfg in all_configs:
            for step in cfg.steps:
                assert dataset_path(step).exists()

    def test_realign_step_uses_dataset_name_not_trait_name(self):
        """The trait is ``sycophantic``; its SFT dataset directory is ``sycophancy``."""
        assert E._realign_step("sycophantic").dataset == "sycophancy"
        assert E._realign_step("evil").dataset == "evil"


class TestTraitSharesFineTuning:
    """The point of dropping trait from ``weights_key``: measuring a second
    trait must cost measurements, never a retrain."""

    def test_configs_differing_only_in_trait_share_every_weights_id(self):
        evil, syco = (
            E.build_exp2_configs(seeds=(0,), measure_traits=(tr,))[0]
            for tr in ("evil", "sycophantic")
        )
        assert evil.trait != syco.trait
        assert [get_weights_id(evil, t) for t in range(len(evil.steps) + 1)] == [
            get_weights_id(syco, t) for t in range(len(syco.steps) + 1)
        ]

    def test_trait_dependent_artifacts_do_not_collide(self, tmp_path):
        store = Store(tmp_path)
        wid = "t01-deadbeefdeadbeef"
        assert store.trait_measurement(
            wid, "evil", "behavior.json"
        ) != store.trait_measurement(wid, "sycophantic", "behavior.json")

    def test_seed_still_changes_the_weights(self):
        a, b = (
            E.build_exp2_configs(seeds=(s,), measure_traits=("evil",))[0]
            for s in (0, 1)
        )
        assert get_weights_id(a, 1) != get_weights_id(b, 1)

    def test_seeds_share_the_base_checkpoint(self):
        """``weights_key`` normalizes seed away at t=0, where no training has
        happened yet, so every seed resolves to one base ``weights_id``.

        The weights at t=0 are the base model verbatim regardless of seed, so
        keying them by seed would re-measure identical weights once per seed.
        The seed does not reach the evaluation path (it drives the trainer and
        the training subsample only), so the repeat bought a *nondeterministic*
        redraw of ``v_0``, not a seed-controlled one -- an across-seed spread
        that no rerun of the same seed could reproduce.
        """
        a, b = (
            E.build_exp2_configs(seeds=(s,), measure_traits=("evil",))[0]
            for s in (0, 1)
        )
        assert get_weights_id(a, 0) == get_weights_id(b, 0)

    def test_normalizing_seed_at_t0_preserves_seed_0_keys(self):
        """Seed is pinned to 0 at t=0 rather than dropped from the key.

        Dropping the field would rehash the seed-0 base checkpoint too and
        orphan the base measurements already in the store.
        """
        cfg = E.build_exp2_configs(seeds=(0,), measure_traits=("evil",))[0]
        assert get_weights_id(cfg, 0) == "t00-7fabae59bda88957"


class TestPrefixSharing:
    def test_diversity_conditions_share_their_first_step(self):
        cfgs = {
            c.name.split("_")[1]: c
            for c in E.build_diversity_configs(
                seeds=(0,), measure_traits=("evil",), realign_traits=("evil",)
            )
        }
        first = {name: get_weights_id(c, 1) for name, c in cfgs.items()}
        assert len(set(first.values())) == 1, first

    def test_same3_extends_same2(self):
        cfgs = {
            c.name.split("_")[1]: c
            for c in E.build_diversity_configs(
                seeds=(0,), measure_traits=("evil",), realign_traits=("evil",)
            )
        }
        # same2 = (d0, d0, realign); same3 = (d0, d0, d0, realign): the (d0, d0)
        # prefix is common, so its adapter is trained once.
        assert get_weights_id(cfgs["same2"], 2) == get_weights_id(cfgs["same3"], 2)

    def test_hysteresis_baseline_is_independent_of_realign_trait(self):
        cfgs = E.build_hysteresis_configs(seeds=(0,), measure_traits=("evil",))
        baselines = [c for c in cfgs if c.name.startswith("exp3_baseline")]
        assert len(baselines) == len(E.HYSTERESIS_DATASETS)
        for c in baselines:
            assert len(c.steps) == 1

    def test_plasticity_arms_are_n_normal_steps_then_the_target(self):
        """The "was it just fine-tuned at all?" controls: only normal data
        before the same target step whose Delta b every other arm reports."""
        cfgs = E.build_hysteresis_configs(
            seeds=(0,), measure_traits=("evil",), realign_traits=("evil",)
        )
        arms = {c.label_map["condition"]: c for c in cfgs}
        realign = E._realign_step("evil")
        target = arms["baseline"].steps[0]
        assert arms["normal1"].steps == (realign, target)
        assert arms["normal2"].steps == (realign, realign, target)
        for name in ("normal1", "normal2"):
            assert arms[name].label_map["dataset"] == arms["baseline"].label_map[
                "dataset"
            ]

    def test_normal2_is_step_matched_with_same_and_diff(self):
        """The claim the matched control rests on: normal2 and same/diff differ
        in *what* the two prior steps trained on, not how many there were."""
        cfgs = E.build_hysteresis_configs(
            seeds=(0,), measure_traits=("evil",), realign_traits=("evil",)
        )
        arms = {c.label_map["condition"]: c for c in cfgs}
        matched = ("normal2", "same", "diff")
        counts = {name: len(arms[name].steps) - 1 for name in matched}
        assert set(counts.values()) == {2}, counts
        for cfg in cfgs:
            assert cfg.label_map["n_prior_steps"] == str(len(cfg.steps) - 1)

    def test_normal_prefixes_is_configurable(self):
        cfgs = E.build_hysteresis_configs(
            seeds=(0,), measure_traits=("evil",), normal_prefixes=(2,)
        )
        conditions = {c.label_map["condition"] for c in cfgs}
        assert "normal2" in conditions
        assert "normal1" not in conditions
        with pytest.raises(ValueError, match="step counts"):
            E.build_hysteresis_configs(seeds=(0,), normal_prefixes=(0,))

    def test_every_arm_ends_on_the_dataset_its_bar_names(self):
        """The bars are only comparable because every condition's final step is
        a step onto the same dataset."""
        for cfg in E.build_hysteresis_configs(seeds=(0,), measure_traits=("evil",)):
            assert cfg.steps[-1].dataset_id == cfg.label_map["dataset"]

    def test_the_plasticity_arms_share_their_normal_prefix(self):
        """Both normal-only arms start from the same 1-step realign chain, and
        normal2 extends it, so the prefix is trained once per (trait, seed)
        rather than once per target dataset."""
        cfgs = [
            c
            for c in E.build_hysteresis_configs(seeds=(0,), measure_traits=("evil",))
            if c.label_map["condition"].startswith("normal")
        ]
        by_realign = {}
        for cfg in cfgs:
            by_realign.setdefault(cfg.label_map["realign_trait"], set()).add(
                get_weights_id(cfg, 1)
            )
        assert by_realign
        for realign_trait, ids in by_realign.items():
            assert len(ids) == 1, f"{realign_trait} retrains its realign step: {ids}"

    def test_same_and_diff_differ_only_in_the_first_step(self):
        cfgs = E.build_hysteresis_configs(
            seeds=(0,), measure_traits=("evil",), realign_traits=("evil",)
        )
        same = next(c for c in cfgs if c.name.startswith("exp3_same"))
        diff = next(c for c in cfgs if c.name.startswith("exp3_diff"))
        assert same.steps[1:] == diff.steps[1:]
        assert same.steps[0] != diff.steps[0]


class TestLocalScaling:
    def test_local_variants_use_the_small_model_and_capped_examples(self):
        for build in ALL_BUILDERS:
            for cfg in build(seeds=(0,), measure_traits=("evil",), local=True):
                assert cfg.model is E.QWEN_0_5B
                assert all(s.n_examples == 64 for s in cfg.steps)
                assert cfg.name.endswith("_local")

    def test_paper_scale_variants_use_the_7b_model_and_full_datasets(self):
        for build in ALL_BUILDERS:
            for cfg in build(seeds=(0,), measure_traits=("evil",)):
                assert cfg.model is E.QWEN_7B
                assert all(s.n_examples is None for s in cfg.steps)


class TestConfigValidation:
    def test_a_trajectory_needs_at_least_one_step(self):
        with pytest.raises(ValueError, match="at least one step"):
            dataclasses.replace(E.EXP1, steps=())
