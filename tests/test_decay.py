"""Tests for the RQ1 decay analysis, :mod:`method.visualization.decay`.

These build ``trajectory.json`` files shaped exactly like the ones the runner
writes -- trunks measured at every checkpoint, branches carrying a single
endpoint -- and assert on the tables the decay figures are drawn from.
Nothing here loads a model, and the figures are exercised separately in
``test_visualization.py``.

The data is deliberately noiseless: ``Delta b`` is an exact linear function of
one projection series and unrelated to the other, so an assertion on $R^2$ is
an assertion about the analysis rather than about a random draw.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from method import experiments as E
from method.config import DatasetVersion, StepConfig, TrajectoryConfig, to_json
from method.store import get_weights_id
from method.utils import trajectory_run_dir
from method.visualization import decay
from method.visualization.collect import Collection, collect
from method.visualization.metrics import bootstrap_fit

#: A two-probe set, which is the smallest that still fits a line.
PROBES = (
    StepConfig(dataset="insecure_code", version=DatasetVersion.NORMAL),
    StepConfig(dataset="evil", version=DatasetVersion.MISALIGNED_2),
    StepConfig(dataset="mistake_math", version=DatasetVersion.MISALIGNED_2),
)
TRUNKS = {"a": E.EXP2_TRUNKS["a"], "c": E.EXP2_TRUNKS["c"]}

#: Delta P_0 per probe, and the slope Delta b is generated with. Chosen so the
#: three probes span a range: a correlation over points stacked at one end of
#: the x-axis is uninformative however clean each point is (section 3c).
DELTA_P_0 = {"insecure_code/normal": -1.0, "evil/misaligned_2": 0.0,
             "mistake_math/misaligned_2": 2.0}
SLOPE = 3.0


@pytest.fixture(autouse=True)
def temp_trajectories(tmp_path, monkeypatch):
    """Point ``trajectory_run_dir`` at a scratch root for every test here."""
    monkeypatch.setattr("method.utils.TRAJECTORIES_DIR", tmp_path / "trajectories")
    return tmp_path


def _behavior(trait: str, value: float, se: float = 0.5) -> dict[str, float]:
    return {
        trait: value,
        f"{trait}_std": 12.0,
        f"{trait}_se": se,
        "coherence": 80.0,
        "n": 200,
        "n_questions": 20,
    }


def write_run(
    cfg: TrajectoryConfig,
    *,
    behaviors,
    probes=None,
    probes_v0=None,
    probes_current=None,
    se=0.5,
    delta_p=1.0,
) -> None:
    """Write a schema-faithful ``trajectory.json``.

    A run whose ``measure`` is ``ENDPOINT_BEHAVIOR`` gets the single-record
    shape a branch really has: final checkpoint, ``b`` only, no ``z`` and no
    probes.
    """
    n = len(cfg.steps)
    if cfg.measure is not E.MeasurementLevel.FULL:
        records = [
            {
                "t": n,
                "weights_id": get_weights_id(cfg, n),
                "behavior": _behavior(cfg.trait, behaviors[-1], se),
            }
        ]
    else:
        records = []
        for t in range(n + 1):
            record = {
                "t": t,
                "weights_id": get_weights_id(cfg, t),
                "behavior": _behavior(cfg.trait, behaviors[t], se),
                "z": {
                    "base": {
                        "p": 0.1 * t,
                        "q": 0.2 * t,
                        "rho": 1.0 - 0.1 * t,
                        "r": 30.0 + t,
                    }
                },
                "probes": {
                    dataset: {"mean": series[t], "std": 0.5, "n": 8}
                    for dataset, series in (probes or {}).items()
                },
                "probes_v0": {
                    dataset: {"mean": series[t], "std": 0.5, "n": 8}
                    for dataset, series in (probes_v0 or {}).items()
                },
                "probes_current": {
                    dataset: {"mean": series[t], "std": 0.5, "n": 8}
                    for dataset, series in (probes_current or {}).items()
                },
            }
            if t < n:
                record["delta_p"] = {"mean": delta_p, "std": 0.5, "n": 8}
                record["next_dataset"] = cfg.steps[t].dataset_id
            records.append(record)

    path = (
        trajectory_run_dir(cfg.name, cfg.seed, cfg.model.name) / "trajectory.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"config": json.loads(to_json(cfg)), "steps": records}),
        encoding="utf-8",
    )


def _probe_series(n_checkpoints: int, drift: float = 0.0) -> dict[str, list[float]]:
    r"""$\Delta P_t$ per probe: its $\Delta P_0$, drifting by ``drift`` per step."""
    return {
        dataset: [value + drift * t for t in range(n_checkpoints)]
        for dataset, value in DELTA_P_0.items()
    }


def build_decay(*, drift: float = 0.0, se: float = 0.5) -> Collection:
    r"""Trunks a and c with their full fans, wired so $\Delta b = 3 \Delta P_0$.

    Every branch endpoint is set to ``b_t + SLOPE * Delta P_0(probe)``, which
    makes $R^2$ against $\Delta P_0$ exactly 1 at every checkpoint. That is the
    opposite of the hypothesis, and deliberately so: an analysis that cannot
    report "no decay" on data with no decay cannot be trusted to report decay.
    """
    configs = E.build_exp2_decay_configs(
        seeds=(E.EXP2_SEED,),
        measure_traits=("evil",),
        trunks=TRUNKS,
        probes=PROBES,
    )
    # Both trunks start at the same b_0: they share M_0, and a fixture that
    # disagreed on it would make the shared t=0 column disagree too.
    trunk_behavior = {"a": [50.0 + 2 * t for t in range(7)], "c": [50.0] * 7}
    for cfg in configs:
        trunk = cfg.label_map["trunk"]
        levels = trunk_behavior[trunk]
        if cfg.label_map["role"] == "trunk":
            write_run(
                cfg,
                behaviors=levels,
                probes=_probe_series(len(levels), drift),
                se=se,
            )
        else:
            t = int(cfg.label_map["t"])
            probe = cfg.label_map["probe"]
            write_run(
                cfg,
                behaviors=[levels[t] + SLOPE * DELTA_P_0[probe]],
                se=se,
            )
    return collect(configs, group=E.EXP2_DECAY)


def build_axis(*, offset: float = 0.5, trunks=("a", "c")) -> Collection:
    r"""The base-axis re-measurement, carrying $\Delta \hat{P}_t^{v_0}$ alone.

    Covers every trunk by default, unlike :func:`build_regen`: this view is
    free to measure, so the realistic case is that it covers whatever the decay
    family ran. ``offset`` shifts it off $\Delta P_0$ by a constant, which
    keeps the correlation exactly 1 so an assertion on it is an assertion about
    the join rather than about a draw.
    """
    configs = E.build_exp2_axis_configs(
        measure_traits=("evil",),
        trunks={name: TRUNKS[name] for name in trunks},
        probes=PROBES,
    )
    for cfg in configs:
        write_run(
            cfg,
            behaviors=[50.0 + 2 * t for t in range(7)],
            probes_v0={
                dataset: [value + offset for value in series]
                for dataset, series in _probe_series(7).items()
            },
        )
    return collect(configs, group=E.EXP2_AXIS)


def build_regen(*, offset: float = 0.0, trunks=("a",)) -> Collection:
    r"""A re-measurement carrying $\Delta P$ and nothing else.

    ``trunks`` defaults to A alone even though the family emits all of them,
    because partial coverage is the case worth fixturing: the measurement is
    paid for per trunk, so a frame that mixes a re-measured trunk with one that
    was skipped is what the analysis actually receives.

    ``offset`` shifts every probe's $\Delta P$ away from its $\Delta P_0$ by a
    constant, which is the fixture equivalent of "the checkpoint no longer
    answers the way $M_0$ did". A constant rather than a per-probe change so
    the correlation stays exactly 1 and an assertion on it is an assertion
    about the join, not about a draw.
    """
    configs = E.build_exp2_regen_configs(
        measure_traits=("evil",),
        trunks={name: TRUNKS[name] for name in trunks},
        probes=PROBES,
    )
    for cfg in configs:
        write_run(
            cfg,
            behaviors=[50.0 + 2 * t for t in range(7)],
            probes_current=_probe_series(7, drift=0.0)
            if not offset
            else {
                dataset: [value + offset for value in series]
                for dataset, series in _probe_series(7).items()
            },
        )
    return collect(configs, group=E.EXP2_REGEN)


def build_validation(*, se: float = 0.5) -> Collection:
    """The ``t = 0`` fan, restricted to the probe datasets these tests use."""
    configs = E.build_exp2_validation_configs(
        seeds=(E.EXP2_SEED,), measure_traits=("evil",), datasets=PROBES
    )
    for cfg in configs:
        dataset = cfg.label_map["dataset"]
        write_run(
            cfg,
            behaviors=[50.0, 50.0 + SLOPE * DELTA_P_0[dataset]],
            probes={d: [v, v] for d, v in DELTA_P_0.items()},
            delta_p=DELTA_P_0[dataset],
            se=se,
        )
    return collect(configs, group=E.EXP2_VALIDATION)


# --- bootstrap_fit ----------------------------------------------------------


class TestBootstrapFit:
    def test_interval_brackets_the_point_estimate(self) -> None:
        rng = np.random.default_rng(0)
        x = np.linspace(0, 10, 30)
        y = 2.0 * x + rng.normal(0, 1.0, size=30)
        interval = bootstrap_fit(x, y, n_resamples=500)
        assert interval.slope_lo <= interval.fit.slope <= interval.slope_hi
        assert interval.corr_lo <= interval.fit.corr <= interval.corr_hi

    def test_noiseless_data_gives_a_tight_interval(self) -> None:
        x = np.arange(8, dtype=float)
        interval = bootstrap_fit(x, 3.0 * x, n_resamples=500)
        assert interval.corr_lo == pytest.approx(1.0)
        assert interval.slope_lo == pytest.approx(3.0)
        assert interval.slope_hi == pytest.approx(3.0)

    def test_deterministic_given_a_seed(self) -> None:
        rng = np.random.default_rng(1)
        x, y = rng.normal(size=12), rng.normal(size=12)
        assert bootstrap_fit(x, y, seed=7) == bootstrap_fit(x, y, seed=7)
        assert bootstrap_fit(x, y, seed=7) != bootstrap_fit(x, y, seed=8)

    def test_degenerate_resamples_are_dropped_not_scored_as_zero(self) -> None:
        """Two points make most resamples a single duplicated point, which
        carries no fit; counting those as r = 0 would drag the interval toward
        zero by an artifact of the resampling rather than of the data."""
        interval = bootstrap_fit([0.0, 1.0], [0.0, 3.0], n_resamples=200)
        assert interval.n_usable < 200
        assert interval.corr_lo == pytest.approx(1.0)

    def test_single_point_yields_no_interval(self) -> None:
        interval = bootstrap_fit([1.0], [2.0])
        assert interval.n_usable == 0
        assert np.isnan(interval.corr_lo)


# --- decay_frame ------------------------------------------------------------


class TestDecayFrame:
    def test_one_row_per_trunk_checkpoint_and_probe(self) -> None:
        rows = decay.decay_frame(build_decay(), build_validation())
        assert len(rows) == len(TRUNKS) * 7 * len(PROBES)
        assert set(rows["t"]) == set(range(7))
        assert set(rows["probe"]) == {p.dataset_id for p in PROBES}

    def test_delta_b_differences_the_branch_against_its_own_trunk(self) -> None:
        rows = decay.decay_frame(build_decay(), build_validation())
        expected = rows["probe"].map(DELTA_P_0) * SLOPE
        assert rows["delta_b"].to_numpy() == pytest.approx(expected.to_numpy())
        assert rows["b_next"].to_numpy() == pytest.approx(
            (rows["b_t"] + expected).to_numpy()
        )

    def test_t0_column_is_shared_across_trunks(self) -> None:
        """All trunks branch from M_0, so the validation fan supplies the
        column once and every row of it must agree."""
        rows = decay.decay_frame(build_decay(), build_validation())
        at_zero = rows[rows["t"] == 0].sort_values(["trunk", "probe"])
        per_trunk = {
            trunk: group["delta_b"].tolist()
            for trunk, group in at_zero.groupby("trunk")
        }
        assert len(set(map(tuple, per_trunk.values()))) == 1
        assert set(per_trunk) == set(TRUNKS)

    def test_without_the_validation_fan_there_is_no_t0_column(self) -> None:
        rows = decay.decay_frame(build_decay())
        assert rows[rows["t"] == 0].empty
        assert not rows[rows["t"] == 1].empty

    def test_delta_p_0_is_the_trunks_own_first_measurement(self) -> None:
        rows = decay.decay_frame(build_decay(drift=0.5), build_validation())
        assert rows["delta_p_0"].to_numpy() == pytest.approx(
            rows["probe"].map(DELTA_P_0).to_numpy()
        )
        drifted = rows[(rows["t"] == 4) & (rows["trunk"] == "a")]
        assert drifted["delta_p_t"].to_numpy() == pytest.approx(
            (drifted["delta_p_0"] + 2.0).to_numpy()
        )

    def test_phase_comes_from_the_trunks_own_schedule(self) -> None:
        rows = decay.decay_frame(build_decay(), build_validation())
        phases = (
            rows[rows["trunk"] == "a"]
            .groupby("t")["steps_since_realignment"]
            .first()
            .tolist()
        )
        assert phases == list(E.steps_since_realignment(E.EXP2_TRUNKS["a"]))
        assert set(rows[rows["trunk"] == "c"]["steps_since_realignment"]) == {0}

    def test_errors_on_delta_b_combine_both_endpoints(self) -> None:
        rows = decay.decay_frame(build_decay(se=0.5), build_validation(se=0.5))
        assert rows["se_delta_b"].to_numpy() == pytest.approx(np.hypot(0.5, 0.5))

    def test_a_missing_branch_drops_its_row_rather_than_the_panel(self) -> None:
        decayed = build_decay()
        kept = [
            run
            for run in decayed.runs
            if not (
                run.label("role") == "branch"
                and run.label("t") == "3"
                and run.label("probe") == PROBES[0].dataset_id
            )
        ]
        rows = decay.decay_frame(
            Collection(E.EXP2_DECAY, kept), build_validation()
        )
        panel = rows[(rows["t"] == 3) & (rows["trunk"] == "a")]
        assert len(panel) == len(PROBES) - 1
        assert not panel["delta_b"].isna().any()


class TestRemeasuredSeries:
    r"""The two views a family of their own has to measure.

    $\Delta \hat{P}_t^{v_0}$ holds the axis at $v^{(0)}$ while the encoder
    moves; $\Delta P_t$ additionally lets the checkpoint answer for itself.
    Each is paid for per trunk, so the interesting cases are the join and what
    happens to every row a family did not cover.
    """

    def test_the_base_axis_family_fills_its_own_column(self) -> None:
        rows = decay.decay_frame(
            build_decay(), build_validation(), [build_axis()]
        )
        assert not rows["delta_p_v0"].isna().any()
        assert rows["delta_p_v0"].to_numpy() == pytest.approx(
            (rows["probe"].map(DELTA_P_0) + 0.5).to_numpy()
        )

    def test_the_two_remeasured_views_do_not_collide(self) -> None:
        """Different families, different record keys, different columns -- one
        must never be read into the other's place."""
        rows = decay.decay_frame(
            build_decay(), build_validation(), [build_axis(), build_regen()]
        )
        trunk_a = rows[rows["trunk"] == "a"]
        assert not trunk_a["delta_p_v0"].isna().any()
        assert not trunk_a["delta_p"].isna().any()
        assert not trunk_a["delta_p_v0"].equals(trunk_a["delta_p"])
        # Trunk C was covered by the free view only.
        trunk_c = rows[rows["trunk"] == "c"]
        assert not trunk_c["delta_p_v0"].isna().any()
        assert trunk_c["delta_p"].isna().all()

    def test_a_family_with_no_runs_leaves_its_column_untouched(self) -> None:
        with_axis = decay.decay_frame(
            build_decay(), build_validation(), [build_axis()]
        )
        assert with_axis["delta_p"].isna().all()

    def test_all_four_series_are_fitted_when_all_are_measured(self) -> None:
        fits = decay.fit_frame(
            decay.decay_frame(
                build_decay(), build_validation(), [build_axis(), build_regen()]
            ),
            n_resamples=50,
        )
        trunk_a = fits[fits["trunk"] == "a"]
        for series in decay.SERIES:
            assert trunk_a[f"corr_{series}"].notna().all(), series

    def test_the_column_is_nan_when_the_family_never_ran(self) -> None:
        rows = decay.decay_frame(build_decay(), build_validation())
        assert rows["delta_p"].isna().all()

    def test_the_regen_family_fills_the_trunks_it_covers(self) -> None:
        rows = decay.decay_frame(
            build_decay(), build_validation(), [build_regen(trunks=("a", "c"))]
        )
        assert not rows["delta_p"].isna().any()

    def test_the_regen_family_fills_its_own_trunk(self) -> None:
        rows = decay.decay_frame(build_decay(), build_validation(), [build_regen()])
        measured = rows[rows["trunk"] == "a"]
        assert not measured["delta_p"].isna().any()
        assert measured["delta_p"].to_numpy() == pytest.approx(
            measured["probe"].map(DELTA_P_0).to_numpy()
        )

    def test_a_trunk_it_did_not_cover_stays_nan(self) -> None:
        """Not measured is not measured-and-small: trunk C keeps a gap rather
        than borrowing trunk A's numbers."""
        rows = decay.decay_frame(build_decay(), build_validation(), [build_regen()])
        assert rows[rows["trunk"] == "c"]["delta_p"].isna().all()

    def test_the_regen_trunk_contributes_no_rows_of_its_own(self) -> None:
        """It re-measures a trunk that already exists, so it must join onto the
        decay family's rows rather than double them."""
        with_regen = decay.decay_frame(
            build_decay(), build_validation(), [build_regen()]
        )
        without = decay.decay_frame(build_decay(), build_validation())
        assert len(with_regen) == len(without)

    def test_it_is_read_from_probes_current_not_probes(self) -> None:
        """The regen family asks only for the recomputed source, so its frozen
        series is empty by design -- reading the wrong key would give a trunk
        whose probes all look absent."""
        regen = build_regen()
        assert all(
            not step.probes and step.probes_current
            for run in regen.runs
            for step in run.trajectory.steps
        )

    def test_it_is_fitted_where_measured_and_nan_where_not(self) -> None:
        fits = decay.fit_frame(
            decay.decay_frame(build_decay(), build_validation(), [build_regen()]),
            n_resamples=50,
        )
        assert fits[fits["trunk"] == "a"]["corr_p"].to_numpy() == pytest.approx(1.0)
        assert fits[fits["trunk"] == "c"]["corr_p"].isna().all()

    def test_an_offset_moves_the_intercept_not_the_correlation(self) -> None:
        """The same invariant the drifting frozen series has: a shift common to
        the probe set is not a loss of predictive accuracy."""
        fits = decay.fit_frame(
            decay.decay_frame(
                build_decay(), build_validation(), [build_regen(offset=1.5)]
            ),
            n_resamples=50,
        )
        assert fits[fits["trunk"] == "a"]["corr_p"].to_numpy() == pytest.approx(1.0)

    def test_the_fit_columns_exist_whether_or_not_it_ran(self) -> None:
        """Every figure indexes the same columns whichever families are on
        disk, so their presence cannot depend on the measurement."""
        without = decay.fit_frame(
            decay.decay_frame(build_decay(), build_validation()), n_resamples=50
        )
        with_regen = decay.fit_frame(
            decay.decay_frame(build_decay(), build_validation(), [build_regen()]),
            n_resamples=50,
        )
        assert list(without.columns) == list(with_regen.columns)
        assert "corr_p" in without and without["corr_p"].isna().all()

    def test_a_partly_measured_scatter_is_not_fitted(self) -> None:
        """A fit over whichever probes happened to be measured is a correlation
        over a different probe set than the one beside it -- which is exactly
        the comparison the figure rests on."""
        rows = decay.decay_frame(build_decay(), build_validation(), [build_regen()])
        holed = (
            (rows["trunk"] == "a")
            & (rows["t"] == 3)
            & (rows["probe"] == PROBES[0].dataset_id)
        )
        rows.loc[holed, "delta_p"] = np.nan
        fits = decay.fit_frame(rows, n_resamples=50)
        partial = fits[(fits["trunk"] == "a") & (fits["t"] == 3)].iloc[0]
        intact = fits[(fits["trunk"] == "a") & (fits["t"] == 4)].iloc[0]
        assert np.isnan(partial["corr_p"])
        assert not np.isnan(intact["corr_p"])
        # The other series over the same scatter are untouched.
        assert not np.isnan(partial["corr_p0"])

    def test_the_phase_contrast_carries_it_too(self) -> None:
        fits = decay.fit_frame(
            decay.decay_frame(build_decay(), build_validation(), [build_regen()]),
            n_resamples=50,
        )
        pairs = decay.phase_contrast_frame(fits, trunk_drivers=TRUNKS)
        assert "delta_corr_p" in pairs
        assert not pairs[pairs["trunk"] == "a"]["delta_corr_p"].isna().any()


# --- fit_frame and the noise ceiling ----------------------------------------


class TestFitFrame:
    def test_one_row_per_trunk_and_checkpoint(self) -> None:
        fits = decay.fit_frame(
            decay.decay_frame(build_decay(), build_validation()), n_resamples=50
        )
        assert len(fits) == len(TRUNKS) * 7
        assert set(fits["n_probes"]) == {len(PROBES)}

    def test_recovers_the_generating_slope_and_a_perfect_fit(self) -> None:
        fits = decay.fit_frame(
            decay.decay_frame(build_decay(), build_validation()), n_resamples=50
        )
        assert fits["corr_p0"].to_numpy() == pytest.approx(1.0)
        assert fits["slope_p0"].to_numpy() == pytest.approx(SLOPE)

    def test_a_drifting_delta_p_t_still_fits_after_a_shift(self) -> None:
        """Adding a constant to every probe's Delta P moves the intercept, not
        the correlation -- so a drift that is common to the probe set must not
        register as staleness."""
        fits = decay.fit_frame(
            decay.decay_frame(build_decay(drift=0.5), build_validation()),
            n_resamples=50,
        )
        assert fits["corr_pt"].to_numpy() == pytest.approx(1.0)

    def test_carries_the_checkpoints_drift_and_behaviour(self) -> None:
        fits = decay.fit_frame(
            decay.decay_frame(build_decay(), build_validation()), n_resamples=50
        )
        row = fits[(fits["trunk"] == "a") & (fits["t"] == 3)].iloc[0]
        assert row["rho"] == pytest.approx(0.7)
        assert row["r"] == pytest.approx(33.0)
        assert row["b_t"] == pytest.approx(56.0)

    def test_noise_ceiling_falls_as_seed_noise_rises(self) -> None:
        rows = decay.decay_frame(build_decay(), build_validation())
        clean = decay.fit_frame(rows, sigma_seed=0.0, n_resamples=10)
        noisy = decay.fit_frame(rows, sigma_seed=5.0, n_resamples=10)
        assert (noisy["r2_max"] < clean["r2_max"]).all()
        assert (noisy["r2_max"] >= 0).all()

    def test_ceiling_uses_both_eval_terms_and_the_seed_term(self) -> None:
        rows = decay.decay_frame(build_decay(se=0.5), build_validation(se=0.5))
        fits = decay.fit_frame(rows, sigma_seed=2.0, n_resamples=10)
        row = fits.iloc[0]
        assert row["var_noise"] == pytest.approx(2.0**2 + 0.5**2 + 0.5**2)
        assert row["r2_max"] == pytest.approx(
            1 - row["var_noise"] / row["var_observed"]
        )

    def test_empty_input_gives_an_empty_frame_with_the_columns(self) -> None:
        import pandas as pd

        fits = decay.fit_frame(pd.DataFrame(columns=["trait"]))
        assert fits.empty
        assert "corr_p0" in fits.columns


# --- mechanism and phase contrast -------------------------------------------


class TestMechanismFrame:
    def test_counts_checkpoints_and_shares_t0_once(self) -> None:
        """Section 9's n is 1 shared t=0 plus one row per trunk per t>0. Three
        copies of the shared checkpoint would weight it three times, and it is
        the one checkpoint with no drift at all."""
        fits = decay.fit_frame(
            decay.decay_frame(build_decay(), build_validation()), n_resamples=10
        )
        rows = decay.mechanism_frame(fits)
        assert len(rows) == 1 + len(TRUNKS) * 6
        assert list(rows[rows["t"] == 0]["trunk"]) == ["shared"]

    def test_the_full_design_gives_nineteen_rows(self) -> None:
        """The headline count stated in section 9, checked against the real
        trunk set rather than this module's two-trunk stand-in."""
        n_trunks, n_steps = len(E.EXP2_TRUNKS), len(E.EXP2_TRUNKS["a"])
        assert 1 + n_trunks * n_steps == 19


class TestRealignmentPairs:
    @pytest.mark.parametrize(
        "trunk, expected",
        [("a", [(1, 2), (3, 4), (5, 6)]), ("b", [(2, 3), (5, 6)]), ("c", [])],
    )
    def test_matches_the_pairs_section_4b_names(self, trunk, expected) -> None:
        assert decay.realignment_pairs(E.EXP2_TRUNKS[trunk]) == expected

    def test_the_control_trunk_contributes_no_pair(self) -> None:
        """Every driver in trunk C is Normal, so a rule keyed on the driver
        alone would call all six steps re-alignments -- but a step that
        re-aligns a model which was never misaligned isolates nothing."""
        assert decay.realignment_pairs(E.EXP2_TRUNKS["c"]) == []


class TestPhaseContrastFrame:
    def test_pairs_checkpoints_around_each_realignment(self) -> None:
        fits = decay.fit_frame(
            decay.decay_frame(build_decay(), build_validation()), n_resamples=10
        )
        pairs = decay.phase_contrast_frame(fits, TRUNKS)
        straddled = list(zip(pairs["t_before"], pairs["t_after"]))
        assert straddled == [(1, 2), (3, 4), (5, 6)]
        assert set(pairs["trunk"]) == {"a"}

    def test_difference_is_after_minus_before(self) -> None:
        fits = decay.fit_frame(
            decay.decay_frame(build_decay(), build_validation()), n_resamples=10
        )
        pairs = decay.phase_contrast_frame(fits, TRUNKS)
        assert pairs["delta_corr_p0"].to_numpy() == pytest.approx(
            (pairs["corr_p0_after"] - pairs["corr_p0_before"]).to_numpy()
        )

    def test_a_checkpoint_that_has_not_run_drops_its_pair(self) -> None:
        fits = decay.fit_frame(
            decay.decay_frame(build_decay(), build_validation()), n_resamples=10
        )
        pairs = decay.phase_contrast_frame(fits[fits["t"] != 4], TRUNKS)
        assert (3, 4) not in list(zip(pairs["t_before"], pairs["t_after"]))


# --- drift frames -----------------------------------------------------------


class TestProbeDriftFrame:
    def test_ratio_is_a_percentage_of_the_probes_own_baseline(self) -> None:
        ratios = decay.probe_drift_frame(build_decay(drift=0.5).runs)
        at_zero = ratios[ratios["t"] == 0]
        assert at_zero["ratio"].to_numpy() == pytest.approx(100.0)
        row = ratios[
            (ratios["trunk"] == "a")
            & (ratios["t"] == 2)
            & (ratios["probe"] == "mistake_math/misaligned_2")
        ].iloc[0]
        assert row["ratio"] == pytest.approx(100.0 * (2.0 + 1.0) / 2.0)

    def test_a_near_zero_baseline_is_dropped_not_divided_by(self, caplog) -> None:
        """Delta P_0 of 0 for evil/misaligned_2 would make every later
        checkpoint an infinite percentage of it."""
        ratios = decay.probe_drift_frame(build_decay().runs)
        assert "evil/misaligned_2" not in set(ratios["probe"])
        assert set(ratios["probe"]) == {
            "insecure_code/normal",
            "mistake_math/misaligned_2",
        }

    def test_covers_every_trunk_and_checkpoint(self) -> None:
        ratios = decay.probe_drift_frame(build_decay().runs)
        assert set(ratios["trunk"]) == set(TRUNKS)
        assert set(ratios["t"]) == set(range(7))


class TestLatentFrame:
    def test_one_row_per_trunk_and_checkpoint(self) -> None:
        latents = decay.latent_frame(build_decay().runs)
        assert len(latents) == len(TRUNKS) * 7
        assert set(latents.columns) >= set(decay.Z_COMPONENTS)

    def test_keeps_raw_units(self) -> None:
        r"""$p$ and $q$ start at essentially zero on the base model, so a
        "% of step 0" reading of them would be division by noise."""
        latents = decay.latent_frame(build_decay().runs)
        row = latents[(latents["trunk"] == "a") & (latents["t"] == 6)].iloc[0]
        assert row["rho"] == pytest.approx(0.4)
        assert row["r"] == pytest.approx(36.0)

    def test_reseeded_trunks_are_separate_series(self) -> None:
        """Trunk A and its replicates share a name and differ only in seed,
        which is what makes the section 6c overlay possible without
        special-casing -- one dashed line per seed, not one per family."""
        configs = E.build_exp2_reseed_configs(
            measure_traits=("evil",), trunks=TRUNKS, probes=PROBES
        )
        for cfg in configs:
            write_run(cfg, behaviors=[60.0] * 7, probes=_probe_series(7))
        runs = [*build_decay().runs, *collect(configs, group=E.EXP2_RESEED).runs]
        latents = decay.latent_frame(runs)
        seeds = set(latents[latents["trunk"] == "a"]["seed"])
        assert seeds == {E.EXP2_SEED, *E.EXP2_RESEED_SEEDS}

    def test_a_probe_free_reseed_still_contributes_its_latent_series(self) -> None:
        """The reseed family carries no probes, so latent_frame is the only
        route its seeds have into a figure. Dropping rows for want of a probe
        would make the whole family invisible."""
        configs = E.build_exp2_reseed_configs(measure_traits=("evil",), trunks=TRUNKS)
        assert all(cfg.probes == () for cfg in configs)
        for cfg in configs:
            write_run(cfg, behaviors=[60.0] * 7, probes={})
        latents = decay.latent_frame(collect(configs, group=E.EXP2_RESEED).runs)
        assert set(latents["seed"]) == set(E.EXP2_RESEED_SEEDS)


# --- validation fan ---------------------------------------------------------


class TestValidationFrame:
    def test_one_row_per_dataset_with_its_own_delta_p_0(self) -> None:
        fan = decay.validation_frame(build_validation())
        assert len(fan) == len(PROBES)
        assert fan["delta_p_0"].to_numpy() == pytest.approx(
            fan["dataset"].map(DELTA_P_0).to_numpy()
        )

    def test_delta_b_is_measured_from_the_base_model(self) -> None:
        fan = decay.validation_frame(build_validation())
        assert fan["delta_b"].to_numpy() == pytest.approx(
            (fan["dataset"].map(DELTA_P_0) * SLOPE).to_numpy()
        )
        assert set(fan["b_t"]) == {50.0}
