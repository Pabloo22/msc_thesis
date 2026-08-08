"""Tests for the subprocess wrapper every GPU worker is launched through.

The failure this guards against is a silent one: the run is already broken by
the time these paths matter, and what is at stake is whether the report of it
says why. A dead worker's reason exists only in the output it wrote before
exiting, so the wrapper has to hold on to that output without changing what a
human watching the box sees.
"""

from __future__ import annotations

import sys

import pytest

from method.utils import StepFailed, run_step


def python(body: str) -> list[str]:
    """An argv running ``body`` as ``-m``-less inline source."""
    return [sys.executable, "-c", body]


class TestRunStep:
    def test_success_is_silent(self, tmp_path, capfd):
        run_step(python("print('done')"), cwd=tmp_path, dry_run=False)

        assert "done" in capfd.readouterr().out

    def test_dry_run_launches_nothing(self, tmp_path, capfd):
        run_step(python("raise SystemExit(1)"), cwd=tmp_path, dry_run=True)

        assert capfd.readouterr().out == ""

    def test_failure_carries_the_workers_last_words(self, tmp_path):
        with pytest.raises(StepFailed) as excinfo:
            run_step(
                python(
                    "import sys;"
                    "print('loading model', flush=True);"
                    "print('CUDA out of memory', file=sys.stderr);"
                    "sys.exit(1)"
                ),
                cwd=tmp_path,
                dry_run=False,
            )

        # stderr and stdout both, in the order the worker wrote them: the
        # reason and the context for it routinely land on different streams.
        assert "loading model" in excinfo.value.tail
        assert "CUDA out of memory" in excinfo.value.tail
        assert excinfo.value.returncode == 1

    def test_message_names_the_worker_not_the_interpreter(self, tmp_path):
        with pytest.raises(StepFailed) as excinfo:
            run_step(
                [sys.executable, "-m", "method.__nonexistent__"],
                cwd=tmp_path,
                dry_run=False,
            )

        # The argv's first entry is a venv path shared by every worker, so it
        # cannot be what a subject line or a log entry identifies the run by.
        assert str(excinfo.value) == "method.__nonexistent__ exited with status 1"

    def test_output_is_still_streamed_while_it_runs(self, tmp_path, capfd):
        """Captured, not swallowed -- the box is watched live during a run."""
        with pytest.raises(StepFailed):
            run_step(
                python("import sys; print('halfway'); sys.exit(2)"),
                cwd=tmp_path,
                dry_run=False,
            )

        assert "halfway" in capfd.readouterr().out

    def test_progress_bar_frames_do_not_evict_the_traceback(self, tmp_path):
        """A bar redraws thousands of times; only its last frame is kept.

        Keeping every frame would fill the buffer with a progress bar and push
        out the traceback printed after it -- the one thing being kept for.
        """
        with pytest.raises(StepFailed) as excinfo:
            run_step(
                python(
                    "import sys\n"
                    "for i in range(5000):\n"
                    "    sys.stdout.write(f'\\rProcessed: {i}/5000')\n"
                    "sys.stdout.write('\\n')\n"
                    "print('AssertionError: no answers generated')\n"
                    "sys.exit(1)\n"
                ),
                cwd=tmp_path,
                dry_run=False,
            )

        tail = excinfo.value.tail
        assert "AssertionError: no answers generated" in tail
        assert "Processed: 4999/5000" in tail
        assert "Processed: 4998/5000" not in tail

    def test_a_worker_that_says_nothing_still_reports_its_status(self, tmp_path):
        with pytest.raises(StepFailed) as excinfo:
            run_step(python("raise SystemExit(3)"), cwd=tmp_path, dry_run=False)

        assert excinfo.value.returncode == 3
        assert excinfo.value.tail == ""

    def test_stays_a_called_process_error(self, tmp_path):
        """Callers written against the stdlib exception keep working."""
        import subprocess

        with pytest.raises(subprocess.CalledProcessError):
            run_step(python("raise SystemExit(1)"), cwd=tmp_path, dry_run=False)
