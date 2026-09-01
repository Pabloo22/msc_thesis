r"""Tests for the extraction-text re-draw.

:mod:`method.axis_refresh` asks whether $v^{(t)}$ -- extracted from $M_t$'s
activations over pos/neg text that $M_0$ generated once -- still agrees with the
vector the paper's own procedure would produce, where $M_t$ answers for itself.
The properties worth pinning are that it measures the *same* checkpoints exp2
already trained (so it never retrains), that the re-draw is quarantined from the
production artifacts it is being compared against, that the $t = 0$ control
cannot be dropped, and that the per-clause filter breakdown stays tied to the
vendored mask it describes.

Everything runs on the mock backend: no model, no GPU, no judge. One mock
property shapes the assertions below and is worth stating outright --
``MockBackend.extract_vector`` seeds on the model path alone, so a re-drawn
extraction set yields a *bit-identical* vector under mock, and ``cos_refresh``
is 1.0 everywhere here by construction. The mock run exercises the
orchestration, the storage layout and the arithmetic; only a real backend
measures the disagreement.
"""

from __future__ import annotations

import dataclasses
import math

import pandas as pd
import pytest

import torch

from method import anchor_noise, axis_refresh, experiments as E, steps
from method.axis_refresh import FLOOR_CHECKPOINT, REFRESH_SUBDIR
from method.backends import get_backend
from method.config import Backend
from method.steps import Artifacts
from method.store import Store, get_weights_id


@pytest.fixture
def backend():
    return get_backend(Backend.MOCK, dtype="float16")


def make_cfgs(traits=("sycophantic",)):
    """One single-step config per trait, as the runner builds them."""
    return [dataclasses.replace(E.SMOKE_MOCK, trait=trait) for trait in traits]


def make_long_cfg(n_steps=4):
    """A trunk deep enough to sweep contiguously; SMOKE_MOCK has only two steps."""
    return dataclasses.replace(
        E.SMOKE_MOCK, trait="sycophantic", steps=E.ALL_DATASETS[:n_steps]
    )


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
        for adapters nothing has trained and the axis would be re-drawn on a
        parallel trunk that no figure reads.
        """
        refresh = E.build_axis_refresh_configs(
            trunk=trunk, measure_traits=E.MEASURE_TRAITS
        )
        decay = [
            cfg
            for cfg in E.build_exp2_decay_configs()
            if cfg.label_map.get("role") == "trunk"
            and cfg.label_map.get("trunk") == trunk
        ]
        for cfg in refresh:
            twin = next(c for c in decay if c.trait == cfg.trait)
            assert [get_weights_id(cfg, t) for t in range(len(cfg.steps) + 1)] == [
                get_weights_id(twin, t) for t in range(len(twin.steps) + 1)
            ]

    def test_default_checkpoints_are_the_whole_trunk(self):
        """The reading is a curve, so every step on the trunk is a point on it."""
        cfg = E.build_axis_refresh_configs()[0]
        assert axis_refresh.default_checkpoints(cfg) == tuple(
            range(1, len(cfg.steps) + 1)
        )

    def test_default_checkpoints_exclude_the_base(self):
        """t=0 holds no drift, and its floor is read from anchor-noise draws."""
        cfg = E.build_axis_refresh_configs()[0]
        assert FLOOR_CHECKPOINT not in axis_refresh.default_checkpoints(cfg)

    def test_default_trait_is_measured_by_exp2(self):
        """A trait exp2 never measured would have no frozen vector to compare to."""
        assert set(E.AXIS_REFRESH_TRAITS) <= set(E.MEASURE_TRAITS)

    def test_unknown_trunk_is_rejected(self):
        with pytest.raises(ValueError, match="unknown trunk"):
            E.build_axis_refresh_configs(trunk="nope")

    def test_no_probes_are_requested(self):
        """Probes would materialise training subsamples nothing here reads."""
        assert all(cfg.probes == () for cfg in E.build_axis_refresh_configs())


# --- the re-draw must not touch what it is compared against -----------------


class TestTheRedrawIsQuarantined:
    def test_onpolicy_artifacts_live_under_the_refresh_subdir(self, tmp_path, backend):
        cfg = make_cfgs()[0]
        store = Store(tmp_path)
        with_adapters(cfg, store)

        pos, neg = axis_refresh.onpolicy_extract_paths(store, cfg, 1)
        vector = axis_refresh.onpolicy_vector_path(store, cfg, 1)
        trait_dir = store.trait_measurement_dir(get_weights_id(cfg, 1), cfg.trait)
        for path in (pos, neg, vector):
            assert REFRESH_SUBDIR in path.relative_to(trait_dir).parts

    def test_production_vector_is_untouched_by_a_redraw(self, tmp_path, backend):
        """The comparison is meaningless if the re-draw overwrites its baseline."""
        cfg = make_cfgs()[0]
        store = Store(tmp_path)
        with_adapters(cfg, store)

        frozen = steps.extract_persona_vector(cfg, 1, store, backend)
        before = frozen.read_bytes()
        model_path = store.merged_dir(get_weights_id(cfg, 1))
        axis_refresh.ensure_onpolicy_vector(cfg, 1, store, backend, str(model_path))

        assert frozen.read_bytes() == before

    def test_redraw_generates_from_the_checkpoint_not_the_base(
        self, tmp_path, backend, monkeypatch
    ):
        """The one thing that makes this an on-policy draw at all."""
        cfg = make_cfgs()[0]
        store = Store(tmp_path)
        with_adapters(cfg, store)

        seen = []
        original = backend.eval_persona

        def record(model_path, *args, **kwargs):
            seen.append(model_path)
            return original(model_path, *args, **kwargs)

        monkeypatch.setattr(backend, "eval_persona", record)
        axis_refresh.ensure_onpolicy_extract_csvs(cfg, 2, store, backend, "M_2")

        assert seen == ["M_2", "M_2"]

    def test_existing_redraw_is_not_regenerated(self, tmp_path, backend, monkeypatch):
        """Each draw is 2000 judged generations; resuming must not repeat one."""
        cfg = make_cfgs()[0]
        store = Store(tmp_path)
        axis_refresh.ensure_onpolicy_extract_csvs(cfg, 1, store, backend, "M_1")

        monkeypatch.setattr(
            backend,
            "eval_persona",
            lambda *a, **k: pytest.fail("re-generated an extraction set already drawn"),
        )
        axis_refresh.ensure_onpolicy_extract_csvs(cfg, 1, store, backend, "M_1")


def floor_row(cos=0.999, neg_pass=1.0):
    """A t=0 control row, as ``measure`` would build it."""
    return {
        "trait": "sycophantic",
        "t": 0,
        "cos_refresh": cos,
        "rho_frozen": 1.0,
        "rho_onpolicy": cos,
        "onpolicy_neg_pass": neg_pass,
        "onpolicy_collapsed": False,
    }


def drifted_row(*, t, cos, collapsed, neg_pass=0.7):
    return {
        "trait": "sycophantic",
        "t": t,
        "cos_refresh": cos,
        "rho_frozen": 0.9,
        "rho_onpolicy": 0.8,
        "onpolicy_neg_pass": neg_pass,
        "onpolicy_collapsed": collapsed,
    }


# --- the filter breakdown ---------------------------------------------------


def write_extract_pair(directory, trait, pos_rows, neg_rows):
    """A row-aligned pos/neg pair as ``eval_persona`` would leave them."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, rows in (
        (Artifacts.EXTRACT_POS, pos_rows),
        (Artifacts.EXTRACT_NEG, neg_rows),
    ):
        path = directory / name
        pd.DataFrame(
            {
                "question": [f"q{i}" for i in range(len(rows))],
                "prompt": [f"p{i}" for i in range(len(rows))],
                "answer": [f"a{i}" for i in range(len(rows))],
                trait: [r[0] for r in rows],
                "coherence": [r[1] for r in rows],
            }
        ).to_csv(path, index=False)
        paths.append(path)
    return tuple(paths)


class TestFilterCounts:
    def test_counts_agree_with_the_vendored_filter(self, tmp_path):
        """One pair survives: the others fail a different clause each."""
        pos, neg = write_extract_pair(
            tmp_path,
            "evil",
            pos_rows=[(90, 90), (10, 90), (90, 90), (90, 10)],
            neg_rows=[(10, 90), (10, 90), (90, 90), (10, 90)],
        )
        counts = axis_refresh.filter_counts(pos, neg, "evil", 50)

        assert counts["n_pairs"] == 4
        assert counts["n_effective"] == 1
        assert counts["pos_pass"] == 0.75  # row 1 scores below the threshold
        assert counts["neg_pass"] == 0.75  # row 2's negative is as evil as its positive
        assert counts["coherence_pass"] == 0.75  # row 3's positive is incoherent

    def test_a_collapsed_negative_side_shows_up_as_neg_pass(self, tmp_path):
        """The degenerate case the freeze exists to prevent, in one number."""
        pos, neg = write_extract_pair(
            tmp_path,
            "evil",
            pos_rows=[(90, 90)] * 4,
            neg_rows=[(90, 90)] * 4,  # M_t can no longer answer benignly
        )
        counts = axis_refresh.filter_counts(pos, neg, "evil", 50)

        assert counts["n_effective"] == 0
        assert counts["neg_pass"] == 0.0
        assert counts["pos_pass"] == 1.0

    def test_misaligned_sides_are_rejected(self, tmp_path):
        """The vendored filter masks pairs, so unequal sides would truncate."""
        pos, neg = write_extract_pair(
            tmp_path, "evil", pos_rows=[(90, 90)] * 4, neg_rows=[(10, 90)] * 3
        )
        with pytest.raises(ValueError, match="row-aligned"):
            axis_refresh.filter_counts(pos, neg, "evil", 50)


# --- a total filter collapse is a result, not a crash ------------------------


def collapse_the_redraw(cfg, store, t, *, trait_score=90):
    """Overwrite a drawn extraction set so no pair can survive the filter.

    Both sides score high on the trait, which is what a checkpoint that can no
    longer answer the *negative* half of its own extraction set looks like.
    """
    directory = axis_refresh.refresh_dir(store, get_weights_id(cfg, t), cfg.trait)
    return write_extract_pair(
        directory,
        cfg.trait,
        pos_rows=[(trait_score, 90)] * 4,
        neg_rows=[(trait_score, 90)] * 4,
    )


class TestCollapseIsReportedNotRaised:
    def test_no_vector_is_extracted_when_nothing_survives(
        self, tmp_path, backend, monkeypatch
    ):
        """The vendored extractor would die on torch.cat of an empty list."""
        cfg = make_cfgs()[0]
        store = Store(tmp_path)
        axis_refresh.ensure_onpolicy_extract_csvs(cfg, 1, store, backend, "M_1")
        collapse_the_redraw(cfg, store, 1)

        monkeypatch.setattr(
            backend,
            "extract_vector",
            lambda *a, **k: pytest.fail("extracted a vector from zero pairs"),
        )
        assert (
            axis_refresh.ensure_onpolicy_vector(cfg, 1, store, backend, "M_1") is None
        )

    def test_compare_reports_undefined_metrics_and_keeps_the_filter_stats(
        self, tmp_path, backend
    ):
        """The filter statistics are the finding, so they must survive."""
        cfg = make_cfgs()[0]
        store = Store(tmp_path)
        with_adapters(cfg, store)
        steps.extract_persona_vector(cfg, 0, store, backend)
        steps.extract_persona_vector(cfg, 1, store, backend)
        axis_refresh.ensure_onpolicy_extract_csvs(cfg, 1, store, backend, "M_1")
        collapse_the_redraw(cfg, store, 1)
        assert (
            axis_refresh.ensure_onpolicy_vector(cfg, 1, store, backend, "M_1") is None
        )

        record = axis_refresh.compare(cfg, 1, store)

        assert record["onpolicy_collapsed"] is True
        assert math.isnan(record["cos_refresh"])
        assert math.isnan(record["rho_onpolicy"])
        assert math.isnan(record["r_onpolicy"])
        # The half that does not depend on the missing vector is still real.
        assert not math.isnan(record["rho_frozen"])
        assert record["onpolicy_n_effective"] == 0
        assert record["onpolicy_neg_pass"] == 0.0
        assert record["onpolicy_n_pairs"] == 4

    def test_measure_completes_through_a_collapsed_checkpoint(
        self, tmp_path, backend, monkeypatch
    ):
        """A collapse must not take the checkpoints already drawn with it."""
        cfgs = make_cfgs()
        store = Store(tmp_path)
        with_adapters(cfgs[0], store)

        original = axis_refresh.ensure_onpolicy_extract_csvs

        def collapse_at_t2(cfg, t, store_, backend_, model_path):
            paths = original(cfg, t, store_, backend_, model_path)
            return collapse_the_redraw(cfg, store_, t) if t == 2 else paths

        monkeypatch.setattr(
            axis_refresh, "ensure_onpolicy_extract_csvs", collapse_at_t2
        )
        frame = axis_refresh.measure(cfgs, [0, 1, 2], store, backend)

        assert list(frame["onpolicy_collapsed"]) == [False, False, True]
        assert not frame[frame["t"] < 2]["cos_refresh"].isna().any()


class TestContiguousSweepsDoNotRemerge:
    def test_a_checkpoint_survives_until_its_successor_is_built(
        self, tmp_path, backend
    ):
        """Evicting t before measuring t+1 forces a replay from the base model.

        ``materialize`` walks forward from the deepest checkpoint already
        merged, so on a contiguous sweep each eviction costs the next
        checkpoint a full rebuild -- a triangular number of merges rather than
        one per step.
        """
        cfgs = [make_long_cfg()]
        store = Store(tmp_path)
        with_adapters(cfgs[0], store)
        write_replicate_vectors(cfgs[0], store, [[1.0, 0.0, 0, 0], [1.0, 0.01, 0, 0]])

        merges = []
        original = backend.merge

        def record(base_path, adapter, out):
            merges.append(out.name)
            return original(base_path, adapter, out)

        backend.merge = record
        axis_refresh.measure(cfgs, [1, 2, 3], store, backend)

        # One merge per checkpoint, not 1 + 2 + 3.
        assert len(merges) == 3

    def test_a_gap_in_the_sweep_still_evicts(self, tmp_path, backend):
        """Nothing builds on t when t+1 is not measured, so it should go."""
        cfgs = [make_long_cfg()]
        store = Store(tmp_path)
        with_adapters(cfgs[0], store)
        write_replicate_vectors(cfgs[0], store, [[1.0, 0.0, 0, 0], [1.0, 0.01, 0, 0]])

        axis_refresh.measure(cfgs, [1, 3], store, backend)

        assert not store.has_merged(get_weights_id(cfgs[0], 1))


class TestAgainstFloorHandlesCollapse:
    def test_a_collapsed_checkpoint_is_named_not_dropped(self):
        """idxmin skips NaN, so the worst outcome would otherwise vanish."""
        frame = pd.DataFrame(
            [
                floor_row(),
                drifted_row(t=5, cos=0.95, collapsed=False),
                drifted_row(t=6, cos=float("nan"), collapsed=True),
            ]
        )
        row = axis_refresh.against_floor(frame).iloc[0]

        assert row["n_collapsed"] == 1
        assert row["collapsed_t"] == "6"
        # The surviving checkpoint still supplies the cosine reading.
        assert row["worst_t"] == 5
        assert row["worst"] == pytest.approx(0.95)

    def test_all_collapsed_yields_undefined_rather_than_raising(self):
        """.loc[[nan]] would raise; the collapse columns carry the finding."""
        frame = pd.DataFrame(
            [
                floor_row(),
                drifted_row(t=5, cos=float("nan"), collapsed=True),
                drifted_row(t=6, cos=float("nan"), collapsed=True),
            ]
        )
        row = axis_refresh.against_floor(frame).iloc[0]

        assert row["n_collapsed"] == 2
        assert row["collapsed_t"] == "5, 6"
        assert math.isnan(row["worst"])
        assert math.isnan(row["gap"])


# --- the sampling floor -----------------------------------------------------


def write_replicate_vectors(cfg, store, values):
    """Stand in for an anchor-noise sweep having already run at t=0.

    Written from replicate 1 up, never 0: replicate 0 *is* the production
    vector (see ``anchor_noise.PRODUCTION_REPLICATE``), and clobbering it would
    corrupt the very baseline the comparison is read against.
    """
    for replicate, value in enumerate(values, start=1):
        path = anchor_noise.persona_vector_path(store, cfg, 0, replicate)
        path.parent.mkdir(parents=True, exist_ok=True)
        vector = torch.zeros(cfg.model.layer + 1, 4)
        vector[cfg.model.layer] = torch.tensor(value)
        torch.save(vector, path)


class TestFloorFromReplicates:
    def test_none_when_there_is_nothing_to_compare(self, tmp_path):
        """One draw is not a floor; re-drawing t=0 is then the only way."""
        cfg = make_cfgs()[0]
        store = Store(tmp_path)
        write_replicate_vectors(cfg, store, [[1.0, 0.0, 0.0, 0.0]])

        assert axis_refresh.floor_from_replicates(cfg, store) is None

    def test_worst_pair_is_reported_not_the_mean(self, tmp_path):
        """A bound in a limitations paragraph has to hold for every pair."""
        cfg = make_cfgs()[0]
        store = Store(tmp_path)
        # Three draws: two identical, one rotated away from them.
        write_replicate_vectors(
            cfg, store, [[1.0, 0.0, 0, 0], [1.0, 0.0, 0, 0], [1.0, 1.0, 0, 0]]
        )

        stats = axis_refresh.floor_from_replicates(cfg, store)

        assert stats["floor_draws"] == 3
        assert stats["floor_pairs"] == 3
        assert stats["floor"] == pytest.approx(2**-0.5)  # the rotated pair
        assert stats["floor_mean"] > stats["floor"]

    def test_measure_without_a_floor_or_replicates_is_rejected(self, tmp_path, backend):
        """Silently reporting cosines with no scale is the failure to avoid."""
        cfgs = make_cfgs()
        store = Store(tmp_path)
        with_adapters(cfgs[0], store)
        with pytest.raises(ValueError, match="sampling floor"):
            axis_refresh.measure(cfgs, [1, 2], store, backend)

    def test_measure_skips_t0_when_replicates_supply_the_floor(self, tmp_path, backend):
        """The whole point: do not buy a number that is already on disk."""
        cfgs = make_cfgs()
        store = Store(tmp_path)
        with_adapters(cfgs[0], store)
        write_replicate_vectors(cfgs[0], store, [[1.0, 0.0, 0, 0], [1.0, 0.01, 0, 0]])

        frame = axis_refresh.measure(cfgs, [1, 2], store, backend)

        assert set(frame["t"]) == {1, 2}

    def test_against_floor_uses_the_supplied_floor_when_t0_is_absent(self):
        frame = pd.DataFrame([drifted_row(t=5, cos=0.90, collapsed=False)])
        row = axis_refresh.against_floor(frame, {"sycophantic": 0.999}).iloc[0]

        assert row["floor"] == pytest.approx(0.999)
        assert row["gap"] == pytest.approx(0.099)


# --- one selection must not overwrite another's summary ----------------------


class TestSummaryLabel:
    def test_checkpoints_and_traits_both_change_the_label(self):
        base = axis_refresh.summary_label("a", ["sycophantic"], [0, 5, 6], local=False)
        fewer_checkpoints = axis_refresh.summary_label(
            "a", ["sycophantic"], [0, 6], local=False
        )
        more_traits = axis_refresh.summary_label(
            "a", ["sycophantic", "evil"], [0, 5, 6], local=False
        )
        other_trunk = axis_refresh.summary_label(
            "c", ["sycophantic"], [0, 5, 6], local=False
        )
        assert len({base, fewer_checkpoints, more_traits, other_trunk}) == 4

    def test_order_does_not_make_a_second_file(self):
        """The same selection, typed differently, is the same measurement."""
        assert axis_refresh.summary_label(
            "a", ["evil", "sycophantic"], [6, 0, 5], local=False
        ) == axis_refresh.summary_label(
            "a", ["sycophantic", "evil"], [0, 5, 6], local=False
        )

    def test_local_runs_do_not_collide_with_paper_scale_ones(self):
        assert axis_refresh.summary_label(
            "a", ["sycophantic"], [0], local=True
        ) != axis_refresh.summary_label("a", ["sycophantic"], [0], local=False)


# --- the sweep --------------------------------------------------------------


class TestMeasure:
    def test_one_row_per_trait_and_checkpoint(self, tmp_path, backend):
        cfgs = make_cfgs(("evil", "sycophantic"))
        store = Store(tmp_path)
        with_adapters(cfgs[0], store)

        frame = axis_refresh.measure(cfgs, [0, 2], store, backend)

        assert len(frame) == 4
        assert set(frame["trait"]) == {"evil", "sycophantic"}
        assert set(frame["t"]) == {0, 2}

    def test_dropping_the_floor_is_rejected(self, tmp_path, backend):
        store = Store(tmp_path)
        with pytest.raises(ValueError, match="sampling floor"):
            axis_refresh.measure(make_cfgs(), [1, 2], store, backend)

    def test_diverging_checkpoint_chains_are_rejected(self, tmp_path, backend):
        """Two traits under one table must sit on the same weights."""
        cfgs = make_cfgs(("evil", "sycophantic"))
        cfgs[1] = dataclasses.replace(cfgs[1], seed=cfgs[0].seed + 1)
        store = Store(tmp_path)
        with pytest.raises(ValueError, match="do not share a checkpoint chain"):
            axis_refresh.measure(cfgs, [0], store, backend)

    def test_rho_frozen_is_one_at_the_base(self, tmp_path, backend):
        """v_0 against itself. A different value means the wrong vector loaded."""
        cfgs = make_cfgs()
        store = Store(tmp_path)
        with_adapters(cfgs[0], store)

        frame = axis_refresh.measure(cfgs, [0], store, backend)

        assert frame["rho_frozen"].iloc[0] == pytest.approx(1.0, abs=1e-5)


class TestAgainstFloor:
    def test_gap_is_measured_from_the_worst_drifted_checkpoint(self):
        frame = pd.DataFrame(
            [
                floor_row(cos=0.99, neg_pass=0.9),
                drifted_row(t=5, cos=0.80, collapsed=False, neg_pass=0.4)
                | {"rho_frozen": 0.7, "rho_onpolicy": 0.6},
                drifted_row(t=6, cos=0.90, collapsed=False, neg_pass=0.7)
                | {"rho_frozen": 0.8, "rho_onpolicy": 0.75},
            ]
        )
        row = axis_refresh.against_floor(frame).iloc[0]

        assert row["worst_t"] == 5
        assert row["floor"] == pytest.approx(0.99)
        assert row["gap"] == pytest.approx(0.19)
        assert row["rho_gap"] == pytest.approx(0.1)
        assert row["neg_pass_drop"] == pytest.approx(0.5)
        assert row["n_collapsed"] == 0

    def test_floor_only_yields_no_rows(self):
        """Comparing the control with itself is not a result worth printing."""
        frame = pd.DataFrame(
            [
                {
                    "trait": "sycophantic",
                    "t": 0,
                    "cos_refresh": 0.99,
                    "rho_frozen": 1.0,
                    "rho_onpolicy": 0.99,
                    "onpolicy_neg_pass": 0.9,
                }
            ]
        )
        assert axis_refresh.against_floor(frame).empty
