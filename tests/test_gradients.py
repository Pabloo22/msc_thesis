"""Tests for recovering gradient traces out of a family run's console log.

The log is an append-only byproduct nothing validates, so the failure that
matters is a silent misattribution: a curve landing on the wrong dataset, or a
first exposure counted as a repeat. Both would produce a table that looks
exactly as convincing as a correct one, which is why the cases below check the
*joins* -- sample id, weights id, exposure index -- rather than the arithmetic.
"""

from __future__ import annotations

import json

from method import gradients
from method.gradients import GradPoint, TrainingRun

STAMP = "2026-08-05 00:00:00,000 [INFO]"


def sample(run: str, step: int, sample_id: str) -> str:
    return (
        f"{STAMP} method.steps: Training file: "
        f"/w/trajectories/{run}/train_step{step}.jsonl"
        f" -> /w/store/training_samples/{sample_id}.jsonl"
    )


def trained(step: int, wid: str, *grad_norms: float) -> list[str]:
    """A step that actually trained, with one logged point per grad norm."""
    return [
        f"{STAMP} run_trajectory: --- training step {step} -> {wid} ---",
        *(
            f"{{'loss': {i / 10:.4f}, 'grad_norm': {g}, "
            f"'learning_rate': {0.0 if i == 0 else 1e-05}, 'epoch': {i / 10}}}"
            for i, g in enumerate(grad_norms)
        ),
        "{'train_runtime': 100.0, 'train_samples_per_second': 1.0, "
        "'train_steps_per_second': 1.0, 'train_loss': 1.0, 'epoch': 1.0}",
    ]


def cached(step: int, wid: str) -> str:
    return (
        f"{STAMP} run_trajectory: "
        f"[skip] adapter for step {step} already trained ({wid})"
    )


def write_log(tmp_path, *lines: str, name: str = "run.log"):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestParseLog:
    def test_reads_a_trained_step(self, tmp_path):
        path = write_log(
            tmp_path,
            ">>> Starting trajectory: EXP3_X_SEED3 (1/1)",
            sample("exp3_x_qwen_seed3", 1, "aaaa"),
            *trained(1, "t01-one", 2.5, 2.0, 1.5),
        )

        runs, slots = gradients.parse_log(path)

        assert [point.grad_norm for point in runs["t01-one"].points] == [2.5, 2.0, 1.5]
        assert [point.step for point in runs["t01-one"].points] == [1, 2, 3]
        assert runs["t01-one"].train_runtime == 100.0
        assert runs["t01-one"].completed
        (slot,) = slots
        assert (slot.step, slot.weights_id, slot.sample_id) == (1, "t01-one", "aaaa")
        assert slot.exposure == 1
        assert slot.trained_here

    def test_ignores_progress_bars_sharing_the_stream(self, tmp_path):
        """tqdm redraws are most of the file and must not be mistaken for data."""
        path = write_log(
            tmp_path,
            sample("exp3_x_qwen_seed3", 1, "aaaa"),
            f"{STAMP} run_trajectory: --- training step 1 -> t01-one ---",
            " 16%|#####     | 73/467 [02:50<15:58,  2.43s/it]",
            "{'loss': 1.0, 'grad_norm': 2.5, 'learning_rate': 0.0, 'epoch': 0.0}",
            "Unsloth: Will smartly offload gradients to save VRAM!",
            "{'loss': 0.9, 'grad_norm': 2.0, 'learning_rate': 1e-05, 'epoch': 0.5}",
        )

        runs, _ = gradients.parse_log(path)

        assert [point.grad_norm for point in runs["t01-one"].points] == [2.5, 2.0]

    def test_counts_a_repeated_sample_as_a_second_exposure(self, tmp_path):
        path = write_log(
            tmp_path,
            ">>> Starting trajectory: EXP3_SAME_SEED3 (1/1)",
            sample("exp3_same_qwen_seed3", 1, "aaaa"),
            *trained(1, "t01-one", 2.0),
            sample("exp3_same_qwen_seed3", 2, "bbbb"),
            *trained(2, "t02-two", 2.0),
            sample("exp3_same_qwen_seed3", 3, "aaaa"),
            *trained(3, "t03-three", 1.0),
        )

        _, slots = gradients.parse_log(path)

        assert [(slot.step, slot.sample_id, slot.exposure) for slot in slots] == [
            (1, "aaaa", 1),
            (2, "bbbb", 1),
            (3, "aaaa", 2),
        ]

    def test_exposure_counter_resets_between_trajectories(self, tmp_path):
        """Otherwise each trajectory inherits the previous one's history."""
        path = write_log(
            tmp_path,
            ">>> Starting trajectory: EXP3_A_SEED3 (1/2)",
            sample("exp3_a_qwen_seed3", 1, "aaaa"),
            *trained(1, "t01-one", 2.0),
            ">>> Starting trajectory: EXP3_B_SEED3 (2/2)",
            sample("exp3_b_qwen_seed3", 1, "aaaa"),
            cached(1, "t01-one"),
        )

        _, slots = gradients.parse_log(path)

        assert [slot.exposure for slot in slots] == [1, 1]

    def test_cache_hit_becomes_a_slot_without_a_run(self, tmp_path):
        path = write_log(
            tmp_path,
            sample("exp3_x_qwen_seed3", 1, "aaaa"),
            cached(1, "t01-one"),
        )

        runs, slots = gradients.parse_log(path)

        assert runs == {}
        assert slots[0].weights_id == "t01-one"
        assert not slots[0].trained_here

    def test_keeps_an_interrupted_run_but_marks_it(self, tmp_path):
        """A preempted box leaves a prefix; it is data, but not a finished curve."""
        path = write_log(
            tmp_path,
            sample("exp3_x_qwen_seed3", 1, "aaaa"),
            f"{STAMP} run_trajectory: --- training step 1 -> t01-one ---",
            "{'loss': 1.0, 'grad_norm': 2.5, 'learning_rate': 0.0, 'epoch': 0.0}",
            ">>> Starting trajectory: EXP3_B_SEED3 (2/2)",
        )

        runs, _ = gradients.parse_log(path)

        assert len(runs["t01-one"].points) == 1
        assert not runs["t01-one"].completed

    def test_drops_a_training_step_with_no_sample_line(self, tmp_path):
        """Without the sample id the curve cannot be attributed to a dataset."""
        path = write_log(tmp_path, *trained(1, "t01-one", 2.0))

        _, slots = gradients.parse_log(path)

        assert slots == []


class TestScanLogs:
    def test_prefers_the_completed_trace_over_the_interrupted_one(self, tmp_path):
        first = write_log(
            tmp_path,
            sample("exp3_x_qwen_seed3", 1, "aaaa"),
            f"{STAMP} run_trajectory: --- training step 1 -> t01-one ---",
            "{'loss': 1.0, 'grad_norm': 9.0, 'learning_rate': 0.0, 'epoch': 0.0}",
            name="preempted.log",
        )
        second = write_log(
            tmp_path,
            sample("exp3_x_qwen_seed3", 1, "aaaa"),
            *trained(1, "t01-one", 2.5, 2.0),
            name="finished.log",
        )

        for order in ([first, second], [second, first]):
            scan = gradients.scan_logs(order)
            assert scan.runs["t01-one"].completed
            assert len(scan.runs["t01-one"].points) == 2

    def test_one_slot_per_trajectory_step_across_logs(self, tmp_path):
        """A rerun trajectory repeats its slots; they are one position, not two."""
        first = write_log(
            tmp_path,
            sample("exp3_x_qwen_seed3", 1, "aaaa"),
            *trained(1, "t01-one", 2.0),
            name="a.log",
        )
        second = write_log(
            tmp_path,
            sample("exp3_x_qwen_seed3", 1, "aaaa"),
            cached(1, "t01-one"),
            name="b.log",
        )

        scan = gradients.scan_logs([first, second])

        (slot,) = scan.slots
        # The rerun reported a cache hit, but this slot is still where the
        # adapter was trained.
        assert slot.trained_here


class TestSlotFrame:
    def test_cache_hit_borrows_the_trace_of_the_step_that_trained_it(self, tmp_path):
        """The join that makes first exposures visible at all."""
        path = write_log(
            tmp_path,
            ">>> Starting trajectory: EXP3_BASELINE_SEED3 (1/2)",
            sample("exp3_baseline_qwen_seed3", 1, "aaaa"),
            *trained(1, "t01-one", 4.0, 2.0),
            ">>> Starting trajectory: EXP3_SAME_SEED3 (2/2)",
            sample("exp3_same_qwen_seed3", 1, "aaaa"),
            cached(1, "t01-one"),
        )

        frame = gradients.slot_frame(gradients.scan_logs([path]))
        repeat = frame[frame.run_name == "exp3_same_qwen_seed3"].iloc[0]

        assert not repeat.trained_here
        assert repeat.trace_source == path.name
        assert repeat.grad_norm_init == 4.0
        assert repeat.n_optim_steps == 2

    def test_labels_condition_and_seed_from_the_run_directory(self, tmp_path):
        path = write_log(
            tmp_path,
            sample("exp3_same_evil_hallucination_misaligned_1_evil_qwen_seed4", 3, "a"),
            *trained(3, "t03-three", 1.0),
        )

        row = gradients.slot_frame(gradients.scan_logs([path])).iloc[0]

        assert row.condition == "same"
        assert row.seed == 4

    def test_store_only_fills_gaps(self, tmp_path):
        path = write_log(
            tmp_path,
            sample("exp3_x_qwen_seed3", 1, "aaaa"),
            *trained(1, "t01-one", 4.0),
            sample("exp3_x_qwen_seed3", 2, "bbbb"),
            cached(2, "t02-two"),
        )
        extra = {
            "t01-one": TrainingRun(
                "t01-one", (GradPoint(1, 1.0, 99.0, 0.0, 0.0),), "x"
            ),
            "t02-two": TrainingRun("t02-two", (GradPoint(1, 1.0, 7.0, 0.0, 0.0),), "x"),
        }

        frame = gradients.slot_frame(gradients.scan_logs([path]), extra).set_index(
            "step"
        )

        assert frame.loc[1, "grad_norm_init"] == 4.0  # log wins where it has one
        assert frame.loc[1, "trace_source"] == path.name
        assert frame.loc[2, "grad_norm_init"] == 7.0  # store fills the hole
        assert frame.loc[2, "trace_source"] == "x"


class TestRepeatTable:
    @staticmethod
    def corpus(tmp_path):
        """One repeat trajectory and one control, matched on sample and depth."""
        return write_log(
            tmp_path,
            # (d2, realign, d2): step 3 is the second exposure to "aaaa".
            ">>> Starting trajectory: EXP3_SAME_SEED3 (1/2)",
            sample("exp3_same_qwen_seed3", 1, "aaaa"),
            *trained(1, "t01-a", 4.0),
            sample("exp3_same_qwen_seed3", 2, "rrrr"),
            *trained(2, "t02-ar", 3.0),
            sample("exp3_same_qwen_seed3", 3, "aaaa"),
            *trained(3, "t03-ara", 1.0),
            # (d_other, realign, d2): step 3 is the *first* exposure to "aaaa",
            # at the same depth.
            ">>> Starting trajectory: EXP3_DIFF_SEED3 (2/2)",
            sample("exp3_diff_qwen_seed3", 1, "oooo"),
            *trained(1, "t01-o", 4.0),
            sample("exp3_diff_qwen_seed3", 2, "rrrr"),
            *trained(2, "t02-or", 3.0),
            sample("exp3_diff_qwen_seed3", 3, "aaaa"),
            *trained(3, "t03-ora", 2.0),
        )

    def test_compares_repeat_against_fresh_at_the_same_depth(self, tmp_path):
        frame = gradients.slot_frame(gradients.scan_logs([self.corpus(tmp_path)]))

        table = gradients.repeat_table(frame)

        # Step 1's "aaaa" has no repeat arm and step 2's "rrrr" no repeat at
        # all, so the only comparable group is step 3.
        assert list(table["step"]) == [3]
        row = table.iloc[0]
        assert (row.sample_id, row.n_first, row.n_repeat) == ("aaaa", 1, 1)
        assert row.first_grad_norm_init == 2.0
        assert row.repeat_grad_norm_init == 1.0
        assert row.ratio == 0.5

    def test_counts_a_shared_training_event_once(self, tmp_path):
        """Two trajectories on one adapter are one measurement, not two."""
        path = write_log(
            tmp_path,
            ">>> Starting trajectory: EXP3_DIFF_EVIL_SEED3 (1/3)",
            sample("exp3_diff_qwen_seed3", 1, "aaaa"),
            *trained(1, "t01-a", 2.0),
            # A second trajectory whose chain agrees, so the runner finds the
            # same adapter already in the store.
            ">>> Starting trajectory: EXP3_DIFF_SYCO_SEED3 (2/3)",
            sample("exp3_diff2_qwen_seed3", 1, "aaaa"),
            cached(1, "t01-a"),
            ">>> Starting trajectory: EXP3_SAME_SEED3 (3/3)",
            sample("exp3_same_qwen_seed3", 1, "aaaa"),
            cached(1, "t01-a"),
            sample("exp3_same_qwen_seed3", 2, "aaaa"),
            *trained(2, "t02-aa", 1.0),
        )

        frame = gradients.slot_frame(gradients.scan_logs([path]))
        table = gradients.repeat_table(frame)

        assert table.empty  # step 1 and step 2 are different depths
        assert list(frame[frame.exposure == 1]["weights_id"]) == ["t01-a"] * 3
        detail = gradients.exposure_detail(frame)
        assert detail.empty

    def test_drops_groups_with_only_one_arm(self, tmp_path):
        path = write_log(
            tmp_path,
            sample("exp3_diff_qwen_seed3", 3, "aaaa"),
            *trained(3, "t03-ora", 2.0),
        )

        assert gradients.repeat_table(
            gradients.slot_frame(gradients.scan_logs([path]))
        ).empty

    def test_detail_lists_every_event_behind_a_group(self, tmp_path):
        frame = gradients.slot_frame(gradients.scan_logs([self.corpus(tmp_path)]))

        detail = gradients.exposure_detail(frame)

        assert list(detail["weights_id"]) == ["t03-ora", "t03-ara"]
        assert list(detail["exposure"]) == [1, 2]


class TestStoreTraces:
    def test_reads_trainer_state_and_skips_the_summary_entry(self, tmp_path):
        adapter = tmp_path / "adapters" / "t01-one"
        adapter.mkdir(parents=True)
        (adapter / "trainer_state.json").write_text(
            json.dumps(
                {
                    "log_history": [
                        {
                            "step": 1,
                            "loss": 2.2,
                            "grad_norm": 3.3,
                            "learning_rate": 0.0,
                            "epoch": 0.0,
                        },
                        # HF Trainer's closing entry carries no gradient.
                        {"step": 1, "train_runtime": 12.5, "epoch": 1.0},
                    ]
                }
            ),
            encoding="utf-8",
        )

        traces = gradients.store_traces(tmp_path)

        assert len(traces["t01-one"].points) == 1
        assert traces["t01-one"].points[0].grad_norm == 3.3
        assert traces["t01-one"].train_runtime == 12.5
        assert traces["t01-one"].source == "store"

    def test_ignores_an_adapter_with_no_trainer_state(self, tmp_path):
        (tmp_path / "adapters" / "t01-one").mkdir(parents=True)

        assert gradients.store_traces(tmp_path) == {}


class TestSummarize:
    def test_init_is_the_first_optimizer_step(self):
        """At learning_rate 0: the gradient before this dataset moves anything."""
        run = TrainingRun(
            "t01-one",
            (
                GradPoint(1, 2.0, 5.0, 0.0, 0.0),
                GradPoint(2, 1.0, 1.0, 1e-5, 0.5),
                GradPoint(3, 0.0, 3.0, 1e-5, 1.0),
            ),
            "log",
        )

        stats = gradients.summarize(run)

        assert stats["grad_norm_init"] == 5.0
        assert stats["grad_norm_median"] == 3.0
        assert stats["grad_norm_final"] == 3.0
        assert stats["loss_init"] == 2.0
        assert stats["n_optim_steps"] == 3


def test_empty_corpus_still_yields_the_expected_columns():
    """A log with nothing in it should report nothing, not raise on a groupby."""
    scan = gradients.LogScan(runs={}, slots=[])
    frame = gradients.slot_frame(scan)

    assert list(frame.columns) == list(gradients._SLOT_COLUMNS)
    assert list(gradients.point_frame(scan).columns) == list(gradients._POINT_COLUMNS)
    assert gradients.repeat_table(frame).empty
    assert list(gradients.repeat_table(frame).columns)[:4] == [
        "sample_id",
        "step",
        "n_first",
        "n_repeat",
    ]
    assert gradients.exposure_detail(frame).empty
    assert "no training slots" in gradients.coverage(frame)
