"""Tests for stage timing, cost estimation and the reports built from them.

The estimates exist to decide whether an experiment is affordable *before* it
is run, so the failure that matters is a plausible-looking wrong number. The
sharpest case is a resumed run: its stages are instant cache hits, and an
estimator that believed them would report that the expensive work ahead costs
nothing.
"""

from __future__ import annotations

import json

import pytest

from method import report, timing
from method.timing import StageRecord, StageTimer
from method.utils import StepFailed


def stage(t: int, name: str, seconds: float, ok: bool = True) -> StageRecord:
    return StageRecord(
        trajectory="EXP3_x", seed=0, t=t, stage=name, seconds=seconds, ok=ok
    )


class TestStageTimer:
    def test_records_a_completed_stage(self, tmp_path):
        path = tmp_path / "timings.jsonl"
        timer = StageTimer(path, trajectory="EXP3_x", seed=2)

        with timer.stage("behavior", 0):
            pass

        (record,) = timing.read_stages(path)
        assert record.stage == "behavior"
        assert record.t == 0
        assert record.seed == 2
        assert record.ok

    def test_records_a_failed_stage_and_re_raises(self, tmp_path):
        """A run that died should still say where it died and how long it got."""
        path = tmp_path / "timings.jsonl"
        timer = StageTimer(path, trajectory="EXP3_x", seed=0)

        with pytest.raises(RuntimeError):
            with timer.stage("train", 1):
                raise RuntimeError("CUDA out of memory")

        (record,) = timing.read_stages(path)
        assert record.stage == "train"
        assert not record.ok

    def test_disabled_timer_records_nothing(self, tmp_path):
        with StageTimer.disabled().stage("behavior", 0):
            pass

        assert list(tmp_path.iterdir()) == []

    def test_disabled_timer_does_not_swallow_exceptions(self):
        """Returning out of the recording ``finally`` would silently discard
        every exception the timer wrapped."""
        with pytest.raises(ValueError, match="boom"):
            with StageTimer.disabled().stage("behavior", 0):
                raise ValueError("boom")

    def test_records_survive_a_truncated_final_line(self, tmp_path):
        """A SIGKILL mid-append must not cost the whole file -- the failure
        report is rendered from it at exactly that moment."""
        path = tmp_path / "timings.jsonl"
        complete = json.dumps({"t": 0, "stage": "behavior", "seconds": 60.0})
        path.write_text(f'{complete}\n{{"trajectory": "EXP', encoding="utf-8")

        records = timing.read_stages(path)

        assert len(records) == 1
        assert records[0].stage == "behavior"


class TestEstimation:
    def test_expected_units_stop_training_one_checkpoint_early(self):
        """The last checkpoint is measured but never trained from."""
        units = timing.expected_units(2)

        assert [t for t, name in units if name == "train"] == [0, 1]
        assert sorted({t for t, name in units if name == "behavior"}) == [0, 1, 2]

    def test_remaining_shrinks_as_stages_complete(self):
        one_done = timing.estimate_remaining([stage(0, "behavior", 600)], n_steps=2)
        two_done = timing.estimate_remaining(
            [stage(0, "behavior", 600), stage(0, "persona_vector", 600)], n_steps=2
        )

        assert one_done is not None and two_done is not None
        assert two_done < one_done

    def test_no_evidence_yields_no_estimate(self):
        """Better to say nothing than to invent a number a budget rests on."""
        assert timing.estimate_remaining([], n_steps=3) is None

    def test_cache_hits_do_not_make_the_rest_look_free(self):
        """A resumed run's instant skips are not evidence about real work."""
        resumed = [stage(0, name, 0.01) for name in timing.CHECKPOINT_STAGES]
        resumed.append(stage(0, "train", 0.01))
        resumed.append(stage(0, "delta_p", 0.01))
        resumed.append(stage(1, "behavior", 900.0))

        remaining = timing.estimate_remaining(resumed, n_steps=2)

        # Everything left is priced from the one real observation (900s), not
        # from the eight instant ones.
        assert remaining is not None and remaining > 900.0

    def test_failed_stages_are_not_evidence(self):
        """A stage that raised part-way through only understates its cost."""
        records = [stage(0, "behavior", 300.0, ok=False), stage(0, "latent", 600.0)]

        assert timing.typical_seconds(records) == {"latent": 600.0}

    def test_family_estimate_scales_with_what_is_left(self):
        runs = [
            timing.RunRecord("a", "EXP3", 0, "ok", 3600.0),
            timing.RunRecord("b", "EXP3", 1, "ok", 5400.0),
        ]

        left = timing.estimate_family_remaining(runs, group="EXP3", remaining=4)

        assert left == pytest.approx(4 * 4500.0)

    def test_group_matching_ignores_case(self):
        """The registry spells families EXP3 and run_family.sh is invoked that
        way; TrajectoryConfig.group records "exp3". A case-sensitive filter
        reported the whole family as costing nothing."""
        run = timing.RunRecord("a", "exp3", 0, "ok", 3600.0)

        assert timing.in_group(run, "EXP3")
        assert timing.in_group(run, "exp3")
        assert timing.in_group(run, "")
        assert not timing.in_group(run, "EXP2")

    def test_family_estimate_ignores_other_families_and_failures(self):
        runs = [
            timing.RunRecord("a", "EXP3", 0, "ok", 3600.0),
            timing.RunRecord("b", "EXP2", 0, "ok", 100.0),
            timing.RunRecord("c", "EXP3", 1, "failed", 10.0),
        ]

        assert timing.estimate_family_remaining(
            runs, group="EXP3", remaining=1
        ) == pytest.approx(3600.0)


class TestCost:
    def test_no_rate_configured_means_no_number(self, monkeypatch):
        """A wrong figure in a cost projection is worse than a missing one."""
        monkeypatch.delenv(timing.GPU_HOURLY_USD_ENV, raising=False)

        assert timing.format_cost(3600.0) == ""

    def test_rate_is_applied_per_hour(self, monkeypatch):
        monkeypatch.setenv(timing.GPU_HOURLY_USD_ENV, "1.80")

        assert timing.format_cost(1800.0) == "$0.90"

    def test_unparseable_rate_is_ignored_not_fatal(self, monkeypatch):
        monkeypatch.setenv(timing.GPU_HOURLY_USD_ENV, "one dollar")

        assert timing.hourly_rate() is None


class TestFormatting:
    @pytest.mark.parametrize(
        "seconds,expected",
        [(45, "45s"), (724, "12m 04s"), (7500, "2h 05m"), (-1, "0s")],
    )
    def test_duration_uses_two_units(self, seconds, expected):
        assert timing.format_duration(seconds) == expected

    def test_stage_table_marks_cache_hits(self, monkeypatch):
        monkeypatch.delenv(timing.GPU_HOURLY_USD_ENV, raising=False)
        table = timing.format_stage_table(
            [stage(0, "behavior", 0.01), stage(1, "behavior", 600.0)]
        )

        assert "cached" in table
        assert "TOTAL" in table

    def test_stage_table_does_not_call_a_failure_a_cache_hit(self):
        """A stage that burned two minutes and then died is not "cached"."""
        table = timing.format_stage_table([stage(0, "delta_p", 162.0, ok=False)])

        assert "1 failed" in table
        assert "cached" not in table

    def test_empty_tables_do_not_crash(self):
        assert timing.format_stage_table([]) == "(no stages recorded)"
        assert timing.format_checkpoint_table([]) == "(no checkpoints recorded)"


class TestReports:
    @pytest.fixture(autouse=True)
    def no_family_env(self, monkeypatch):
        for var in (
            report.FAMILY_ENV,
            report.FAMILY_INDEX_ENV,
            report.FAMILY_TOTAL_ENV,
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv(timing.GPU_HOURLY_USD_ENV, raising=False)

    def test_failure_report_leads_with_the_error(self, tmp_path):
        from method import experiments as E

        subject, body = report.trajectory_report(
            E.SMOKE_MOCK,
            tmp_path,
            elapsed=4200.0,
            error=RuntimeError("CUDA out of memory"),
            traceback_text="Traceback (most recent call last):\n  ...",
        )

        assert subject.startswith("FAILED: ")
        assert "CUDA out of memory" in body
        # The decision this email supports is whether to stop paying.
        assert "still rented" in body
        assert body.index("CUDA out of memory") < body.index("Where the time went")

    def test_failure_report_quotes_what_the_worker_printed(self, tmp_path):
        """The reason a subprocess died is in its output, not in the exception.

        Without this the whole report of a broken run was "exit status 1",
        which is true of every broken run and so tells the reader nothing.
        """
        from method import experiments as E

        error = StepFailed(
            1,
            ["python", "-m", "method._generate_worker", "--model", "Qwen"],
            "  | torch.OutOfMemoryError: CUDA out of memory",
        )

        _, body = report.trajectory_report(
            E.SMOKE_MOCK, tmp_path, elapsed=200.0, error=error, traceback_text="tb"
        )

        assert "CUDA out of memory" in body
        assert "method._generate_worker" in body
        assert body.index("CUDA out of memory") < body.index("Where the time went")

    def test_failure_report_names_the_stage_that_died(self, tmp_path):
        from method import experiments as E

        timer = StageTimer(timing.stage_log_path(tmp_path), trajectory="EXP3_x", seed=0)
        with timer.stage("behavior", 0):
            pass
        with pytest.raises(RuntimeError), timer.stage("delta_p", 2):
            raise RuntimeError("boom")

        _, body = report.trajectory_report(
            E.SMOKE_MOCK, tmp_path, elapsed=200.0, error=RuntimeError("boom")
        )

        assert "Failed during" in body
        assert "delta_p at t=2" in body

    def test_success_report_has_no_failure_language(self, tmp_path):
        from method import experiments as E

        subject, body = report.trajectory_report(E.SMOKE_MOCK, tmp_path, elapsed=60.0)

        assert not subject.startswith("FAILED")
        assert "still rented" not in body

    def test_family_context_appears_only_when_set(self, tmp_path, monkeypatch):
        from method import experiments as E

        assert report.FamilyContext.from_env() is None

        monkeypatch.setenv(report.FAMILY_ENV, "EXP3")
        monkeypatch.setenv(report.FAMILY_INDEX_ENV, "3")
        monkeypatch.setenv(report.FAMILY_TOTAL_ENV, "12")
        subject, body = report.trajectory_report(E.SMOKE_MOCK, tmp_path, elapsed=60.0)

        assert "(3/12)" in subject
        assert "Family EXP3" in body

    def test_family_context_ignores_partial_env(self, monkeypatch):
        monkeypatch.setenv(report.FAMILY_ENV, "EXP3")
        monkeypatch.delenv(report.FAMILY_INDEX_ENV, raising=False)

        assert report.FamilyContext.from_env() is None

    def test_note_takes_over_the_family_subject(self):
        subject, body = report.family_report(
            "EXP3", mock=True, note="run_family.sh exited with status 137"
        )

        assert "stopped" in subject
        assert "status 137" in body
