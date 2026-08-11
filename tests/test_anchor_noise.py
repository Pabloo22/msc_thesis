r"""Tests for the anchor-noise replicate sweep.

:mod:`method.anchor_noise` re-draws the two base measurements every $z_t$ is
read against -- $v_0$ and ``h_neutral_base`` -- and reports how far $z_t$ moves
when the weights did not. The properties worth pinning are that it measures the
*same* checkpoints exp2 already trained (so it never retrains), that a re-draw
is quarantined from the production artifacts it is being compared against, and
that the spread maths distinguishes a common-mode anchor shift from a per-
checkpoint one.

Everything runs on the mock backend: no model, no GPU, no judge. One mock
property shapes the assertions below and is worth stating outright --
``MockBackend.extract_vector`` seeds on the model path alone, so re-drawn pos/neg
text yields a *bit-identical* vector under mock. The rotation and norm therefore
show zero spread here by construction, while ``p`` and ``q`` vary because
``hidden_states`` seeds on the answers' path. The mock run exercises the
orchestration and the arithmetic; only a real backend measures the noise.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from method import anchor_noise, experiments as E, steps
from method.anchor_noise import PRODUCTION_REPLICATE, REPLICATE_SUBDIR
from method.backends import get_backend
from method.config import Backend, HNeutralSource
from method.store import Store, get_weights_id
from method.steps import Artifacts


@pytest.fixture
def backend():
    return get_backend(Backend.MOCK, dtype="float16")


def make_cfgs(traits=("evil", "sycophantic")):
    """Two single-step configs differing only in trait, as the runner builds them."""
    return [dataclasses.replace(E.SMOKE_MOCK, trait=trait) for trait in traits]


def with_adapters(cfg, store):
    """Fake adapters for every step, so ``materialize`` can replay the chain."""
    for t in range(1, len(cfg.steps) + 1):
        adapter = store.adapter_dir(get_weights_id(cfg, t))
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")


# --- the invariant that keeps this from retraining anything -----------------


class TestConfigsReuseTheExistingTrunk:
    @pytest.mark.parametrize("trunk", sorted(E.EXP2_TRUNKS))
    def test_checkpoints_match_the_exp2_decay_trunk(self, trunk):
        """The whole design rests on this: measuring exp2's own checkpoints.

        If a builder edit ever made these diverge, the script would silently ask
        for adapters nothing has trained and the noise would be measured on a
        parallel trunk that no figure reads.
        """
        anchor = E.build_anchor_noise_configs(trunk=trunk)
        decay = [
            cfg
            for cfg in E.build_exp2_decay_configs()
            if cfg.label_map.get("role") == "trunk"
            and cfg.label_map.get("trunk") == trunk
        ]
        for cfg in anchor:
            twin = next(c for c in decay if c.trait == cfg.trait)
            assert [get_weights_id(cfg, t) for t in range(len(cfg.steps) + 1)] == [
                get_weights_id(twin, t) for t in range(len(twin.steps) + 1)
            ]

    def test_default_checkpoints_are_reachable_on_the_trunk(self):
        cfg = E.build_anchor_noise_configs()[0]
        assert max(E.ANCHOR_NOISE_CHECKPOINTS) <= len(cfg.steps)

    def test_base_checkpoint_is_included(self):
        """Every z_t is read against v_0, so t=0 is not optional."""
        assert 0 in E.ANCHOR_NOISE_CHECKPOINTS

    def test_unknown_trunk_is_rejected(self):
        with pytest.raises(ValueError, match="unknown trunk"):
            E.build_anchor_noise_configs(trunk="nope")

    def test_no_probes_are_requested(self):
        """DeltaP is a separate budget; asking for it here would cost hours."""
        assert all(cfg.probes == () for cfg in E.build_anchor_noise_configs())


# --- quarantine -------------------------------------------------------------


class TestReplicateZeroIsProduction:
    def test_paths_are_the_ordinary_ones(self, tmp_path):
        store, cfg = Store(tmp_path), make_cfgs()[0]
        wid = get_weights_id(cfg, 0)

        assert anchor_noise.persona_vector_path(
            store, cfg, 0, PRODUCTION_REPLICATE
        ) == store.trait_measurement(
            wid, cfg.trait, Artifacts.persona_vector(cfg.trait)
        )
        assert anchor_noise.h_neutral_path(store, cfg, 0, PRODUCTION_REPLICATE) == (
            store.measurement_dir(wid)
            / Artifacts.h_neutral("base")
            / "mean_by_layer.pt"
        )

    def test_redraw_paths_are_namespaced_under_the_bundle(self, tmp_path):
        """Nested inside the measurement bundle, which is what sync tars."""
        store, cfg = Store(tmp_path), make_cfgs()[0]
        wid = get_weights_id(cfg, 0)

        vector = anchor_noise.persona_vector_path(store, cfg, 0, 1)
        h = anchor_noise.h_neutral_path(store, cfg, 0, 1)

        assert REPLICATE_SUBDIR in vector.parts and REPLICATE_SUBDIR in h.parts
        assert vector.is_relative_to(store.measurement_dir(wid))
        assert h.is_relative_to(store.measurement_dir(wid))

    def test_replicates_do_not_collide_with_each_other(self, tmp_path):
        store, cfg = Store(tmp_path), make_cfgs()[0]
        paths = {
            anchor_noise.persona_vector_path(store, cfg, 0, r) for r in range(4)
        } | {anchor_noise.h_neutral_path(store, cfg, 0, r) for r in range(4)}
        assert len(paths) == 8

    def test_neutral_answers_are_shared_between_traits(self, tmp_path):
        """They involve no trait; two traits drawing separately would put the
        traits on different anchors while the table claims one replicate."""
        store = Store(tmp_path)
        evil, syco = make_cfgs()
        wid = get_weights_id(evil, 0)
        assert anchor_noise.shared_dir(store, wid, 1) == anchor_noise.shared_dir(
            store, wid, 1
        )
        assert anchor_noise.trait_dir(store, wid, "evil", 1) != anchor_noise.trait_dir(
            store, wid, syco.trait, 1
        )


class TestRedrawsLeaveProductionAlone:
    def test_production_artifacts_are_untouched_by_a_redraw(self, tmp_path, backend):
        """The point of the quarantine: exp2/exp3 must not be invalidated."""
        store, cfg = Store(tmp_path), make_cfgs()[0]
        steps.extract_persona_vector(cfg, 0, store, backend)
        steps.measure_h_neutral(cfg, 0, store, backend)
        production = anchor_noise.persona_vector_path(
            store, cfg, 0, PRODUCTION_REPLICATE
        )
        before = production.read_bytes()

        anchor_noise.ensure_persona_vector(cfg, 0, store, backend, 1, cfg.model.name)
        anchor_noise.ensure_h_neutral(cfg, 0, store, backend, 1, cfg.model.name)

        assert production.read_bytes() == before

    def test_a_redraw_samples_its_own_pos_neg_and_answers(self, tmp_path, backend):
        store, cfg = Store(tmp_path), make_cfgs()[0]
        anchor_noise.ensure_persona_vector(cfg, 0, store, backend, 1, cfg.model.name)
        anchor_noise.ensure_h_neutral(cfg, 0, store, backend, 1, cfg.model.name)

        base_wid = get_weights_id(cfg, 0)
        redrawn = anchor_noise.trait_dir(store, base_wid, cfg.trait, 1)
        assert (redrawn / Artifacts.EXTRACT_POS).exists()
        assert (redrawn / Artifacts.EXTRACT_NEG).exists()
        assert (
            anchor_noise.shared_dir(store, base_wid, 1) / Artifacts.NEUTRAL_ANSWERS
        ).exists()

    def test_a_second_pass_regenerates_nothing(self, tmp_path, backend):
        """Resumable like every other measurement here: presence is completeness."""
        store, cfg = Store(tmp_path), make_cfgs()[0]
        vector = anchor_noise.ensure_persona_vector(
            cfg, 0, store, backend, 1, cfg.model.name
        )
        h = anchor_noise.ensure_h_neutral(cfg, 0, store, backend, 1, cfg.model.name)
        stamps = (vector.stat().st_mtime_ns, h.stat().st_mtime_ns)

        anchor_noise.ensure_persona_vector(cfg, 0, store, backend, 1, cfg.model.name)
        anchor_noise.ensure_h_neutral(cfg, 0, store, backend, 1, cfg.model.name)

        assert (vector.stat().st_mtime_ns, h.stat().st_mtime_ns) == stamps

    def test_a_config_without_the_base_source_is_rejected(self, tmp_path, backend):
        store = Store(tmp_path)
        cfg = dataclasses.replace(
            make_cfgs()[0],
            latent=dataclasses.replace(
                E.SMOKE_MOCK.latent, h_neutral_source=HNeutralSource.CURRENT
            ),
        )
        with pytest.raises(ValueError, match="h_neutral source"):
            anchor_noise.ensure_h_neutral(
                cfg, 0, store, backend, PRODUCTION_REPLICATE, cfg.model.name
            )


# --- the sweep --------------------------------------------------------------


class TestMeasure:
    def test_one_row_per_trait_checkpoint_replicate(self, tmp_path, backend):
        store, cfgs = Store(tmp_path), make_cfgs()
        with_adapters(cfgs[0], store)

        frame = anchor_noise.measure(cfgs, [0, 1], [0, 1, 2], store, backend)

        assert len(frame) == 2 * 2 * 3
        assert set(frame["replicate"]) == {0, 1, 2}
        assert set(frame["trait"]) == {"evil", "sycophantic"}
        assert not frame[list(anchor_noise.COMPONENTS)].isna().any().any()

    def test_replicates_disagree_on_the_projections(self, tmp_path, backend):
        """h differs per draw under mock, so p and q must move; the vector does
        not (see the module docstring), so rho and r must not."""
        store, cfgs = Store(tmp_path), make_cfgs(("evil",))

        frame = anchor_noise.measure(cfgs, [0], [0, 1, 2], store, backend)
        table = anchor_noise.spread(frame)
        sigma = table.set_index("component")["sigma_level"]

        assert sigma["p"] > 0 and sigma["q"] > 0
        assert sigma["rho"] == 0 and sigma["r"] == 0

    def test_the_production_replicate_is_required(self, tmp_path, backend):
        store, cfgs = Store(tmp_path), make_cfgs(("evil",))
        with pytest.raises(ValueError, match="production bundle"):
            anchor_noise.measure(cfgs, [0], [1, 2], store, backend)

    def test_the_base_checkpoint_is_required(self, tmp_path, backend):
        store, cfgs = Store(tmp_path), make_cfgs(("evil",))
        with pytest.raises(ValueError, match="checkpoint 0"):
            anchor_noise.measure(cfgs, [1], [0, 1], store, backend)

    def test_configs_on_different_weights_are_rejected(self, tmp_path, backend):
        """Two traits must sit on one chain, or the table mixes models."""
        store = Store(tmp_path)
        evil = make_cfgs(("evil",))[0]
        divergent = dataclasses.replace(evil, trait="sycophantic", seed=evil.seed + 1)
        with pytest.raises(ValueError, match="do not share a checkpoint chain"):
            anchor_noise.measure([evil, divergent], [0], [0, 1], store, backend)

    def test_rho_is_one_at_the_base_checkpoint(self, tmp_path, backend):
        """v_0 against itself, within a draw -- a guard that a replicate's z_0 is
        assembled from its own vector and not another's."""
        store, cfgs = Store(tmp_path), make_cfgs(("evil",))
        frame = anchor_noise.measure(cfgs, [0], [0, 1], store, backend)
        assert frame["rho"].round(5).eq(1.0).all()


# --- the arithmetic ---------------------------------------------------------


def synthetic(levels_by_replicate):
    """A frame with given p at t in {0, 1}, everything else held flat."""
    rows = []
    for replicate, (p0, p1) in enumerate(levels_by_replicate):
        for t, p in ((0, p0), (1, p1)):
            rows.append(
                {
                    "trait": "evil",
                    "t": t,
                    "weights_id": f"t{t:02d}",
                    "replicate": replicate,
                    "p": p,
                    "q": 0.0,
                    "rho": 1.0,
                    "r": 1.0,
                }
            )
    return pd.DataFrame(rows)


class TestSpread:
    def test_a_common_mode_shift_cancels_in_the_difference(self):
        """Each draw is offset by a constant: the levels disagree, the drift
        every draw reports does not. This is the anchor's actual shape, and the
        reason both columns are reported."""
        frame = synthetic([(0.0, 1.0), (5.0, 6.0), (-5.0, -4.0)])

        table = anchor_noise.spread(frame)
        at_1 = table[(table["t"] == 1) & (table["component"] == "p")].iloc[0]

        assert at_1["sigma_level"] == pytest.approx(5.0)
        assert at_1["sigma_delta"] == pytest.approx(0.0)

    def test_a_per_checkpoint_error_survives_the_difference(self):
        frame = synthetic([(0.0, 1.0), (0.0, 3.0), (0.0, 5.0)])

        table = anchor_noise.spread(frame)
        at_1 = table[(table["t"] == 1) & (table["component"] == "p")].iloc[0]

        assert at_1["sigma_level"] == pytest.approx(2.0)
        assert at_1["sigma_delta"] == pytest.approx(2.0)

    def test_the_production_value_is_carried_through(self):
        frame = synthetic([(0.0, 1.0), (5.0, 6.0)])
        table = anchor_noise.spread(frame)
        at_1 = table[(table["t"] == 1) & (table["component"] == "p")].iloc[0]
        assert at_1["production"] == pytest.approx(1.0)
        assert at_1["replicates"] == 2

    def test_the_base_checkpoint_has_no_delta_spread(self):
        frame = synthetic([(0.0, 1.0), (5.0, 6.0)])
        table = anchor_noise.spread(frame)
        at_0 = table[(table["t"] == 0) & (table["component"] == "p")].iloc[0]
        assert at_0["sigma_delta"] == pytest.approx(0.0)


class TestAgainstDrift:
    def test_ratios_compare_noise_with_the_drift_in_the_same_table(self):
        # Production (replicate 0) drifts 0 -> 1; the draws disagree by 5 in
        # level and not at all in difference.
        frame = synthetic([(0.0, 1.0), (5.0, 6.0), (-5.0, -4.0)])

        row = anchor_noise.against_drift(frame, anchor_noise.spread(frame))
        p = row[row["component"] == "p"].iloc[0]

        assert p["drift"] == pytest.approx(1.0)
        assert p["ratio_level"] == pytest.approx(5.0)
        assert p["ratio_delta"] == pytest.approx(0.0)

    def test_zero_drift_gives_an_infinite_ratio_rather_than_a_crash(self):
        frame = synthetic([(0.0, 0.0), (1.0, 1.0)])
        row = anchor_noise.against_drift(frame, anchor_noise.spread(frame))
        assert row[row["component"] == "p"].iloc[0]["ratio_level"] == float("inf")


class TestSummaryFile:
    def test_it_is_readable_without_the_store(self, tmp_path, backend, monkeypatch):
        import json

        from method import utils

        # anchor_noise_path resolves the root at call time, so patching the
        # module global is enough to keep the test out of the real repo.
        monkeypatch.setattr(utils, "TRAJECTORIES_DIR", tmp_path / "trajectories")
        store, cfgs = Store(tmp_path / "store"), make_cfgs(("evil",))

        frame = anchor_noise.measure(cfgs, [0], [0, 1], store, backend)
        path = anchor_noise.write_summary(cfgs, frame, "trunk_a", mock=True)

        payload = json.loads(path.read_text())
        assert payload["base_weights_id"] == get_weights_id(cfgs[0], 0)
        assert payload["h_neutral_source"] == "base"
        assert len(payload["latents"]) == len(frame)
        assert payload["spread"] and payload["against_drift"]
