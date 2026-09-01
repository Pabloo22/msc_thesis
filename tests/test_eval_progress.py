"""Unit tests for crash-resumable judging in ``eval_progress``.

The point of the module is what survives a failure, so the tests drive it with
a judge that raises partway through and then assert on what the *next* run has
to pay for. Fakes stand in for the vendored ``Question`` and for vLLM: both are
just "something with ``.judges``/``.get_input``" and "something returning
(prompts, answers)" as far as the resume logic is concerned.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from method.eval_progress import (  # noqa: E402
    GENERATIONS,
    JUDGMENTS,
    ProgressStore,
    make_eval_batched,
)

TRAIT = "evil"
N_PER_QUESTION = 2


class FakeJudge:
    """Scores by answer length, and can be told to blow up on the Nth call."""

    def __init__(self, metric: str, fail_on: int | None = None) -> None:
        self.metric = metric
        self.fail_on = fail_on
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, *, question: str, answer: str) -> float:
        self.calls.append((question, answer))
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise RuntimeError("judge API is having a bad day")
        return float(len(answer))


class FakeQuestion:
    def __init__(self, qid: str, judges: dict[str, FakeJudge]) -> None:
        self.id = qid
        self.temperature = 1.0
        self.judges = judges

    def get_input(self, n_per_question: int):
        paraphrases = [f"{self.id}?"] * n_per_question
        conversations = [[{"role": "user", "content": p}] for p in paraphrases]
        return paraphrases, conversations


class FakeGenerator:
    """Stands in for vLLM; counts how many times it was asked to generate."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, conversations, **kwargs):
        self.calls += 1
        prompts = [c[0]["content"] for c in conversations]
        answers = [f"answer {i}" for i in range(len(conversations))]
        return prompts, answers


def make_questions(fail_on: int | None = None) -> list[FakeQuestion]:
    """Two questions, each judged on the trait and on coherence."""
    return [
        FakeQuestion(
            f"{TRAIT}_{i}",
            {
                TRAIT: FakeJudge(TRAIT, fail_on=fail_on if i == 0 else None),
                "coherence": FakeJudge("coherence"),
            },
        )
        for i in range(2)
    ]


def make_store(tmp_path: Path, **overrides) -> ProgressStore:
    generation = {"model": "m", "trait": TRAIT, "n_per_question": N_PER_QUESTION}
    judging = {"model": "judge-1", "backend": "openai"}
    generation.update(overrides.pop("generation", {}))
    judging.update(overrides.pop("judging", {}))
    store = ProgressStore(tmp_path / "progress", generation=generation, judging=judging)
    store.prepare()
    return store


def run_eval(store, generator, questions, **kwargs):
    """Drive one ``eval_batched`` pass over ``questions``."""
    eval_batched = make_eval_batched(
        store, generator, judge_attempts=kwargs.pop("judge_attempts", 1), retry_delay=0
    )
    return asyncio.run(
        eval_batched(
            questions,
            llm=kwargs.pop("llm", "fake-llm"),
            tokenizer=None,
            coef=0,
            n_per_question=N_PER_QUESTION,
            max_concurrent_judges=kwargs.pop("max_concurrent_judges", 1),
            max_tokens=10,
        )
    )


def judge_call_count(questions) -> int:
    return sum(len(j.calls) for q in questions for j in q.judges.values())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_fresh_pass_scores_everything_and_banks_it(tmp_path):
    store = make_store(tmp_path)
    generator = FakeGenerator()
    questions = make_questions()

    frames = run_eval(store, generator, questions)

    assert generator.calls == 1
    assert [list(f.columns) for f in frames] == [
        ["question", "prompt", "answer", "question_id", TRAIT, "coherence"]
    ] * 2
    assert [len(f) for f in frames] == [N_PER_QUESTION, N_PER_QUESTION]
    assert frames[0]["question_id"].tolist() == [f"{TRAIT}_0"] * N_PER_QUESTION
    # Scores are the fake judge's rule (length of the answer), row by row.
    assert frames[0][TRAIT].tolist() == [
        float(len(a)) for a in frames[0]["answer"].tolist()
    ]

    generations = read_jsonl(store.dir / GENERATIONS)
    judgments = read_jsonl(store.dir / JUDGMENTS)
    assert len(generations) == 2 * N_PER_QUESTION
    assert len(judgments) == 2 * N_PER_QUESTION * 2  # two metrics per row


def test_failed_pass_keeps_what_it_already_paid_for(tmp_path):
    store = make_store(tmp_path)
    generator = FakeGenerator()
    questions = make_questions(fail_on=2)

    with pytest.raises(RuntimeError, match="bad day"):
        run_eval(store, generator, questions)

    # Generation completed, so it is durable in full; judging died partway.
    assert len(read_jsonl(store.dir / GENERATIONS)) == 2 * N_PER_QUESTION
    banked = read_jsonl(store.dir / JUDGMENTS)
    assert 0 < len(banked) < 2 * N_PER_QUESTION * 2


def test_resume_regenerates_nothing_and_rejudges_only_the_gap(tmp_path):
    store = make_store(tmp_path)
    generator = FakeGenerator()
    failing = make_questions(fail_on=2)
    with pytest.raises(RuntimeError):
        run_eval(store, generator, failing)
    already_paid = len(read_jsonl(store.dir / JUDGMENTS))
    total = 2 * N_PER_QUESTION * 2

    resumed = make_questions()
    frames = run_eval(store, generator, resumed)

    assert generator.calls == 1  # the second pass never touched the GPU
    assert judge_call_count(resumed) == total - already_paid
    assert len(read_jsonl(store.dir / JUDGMENTS)) == total
    assert all(f[TRAIT].notna().all() and f["coherence"].notna().all() for f in frames)

    # Resumed scores match a clean run's: the banked ones come off disk, the
    # rest are recomputed by the same rule.
    for frame in frames:
        assert frame[TRAIT].tolist() == [
            float(len(a)) for a in frame["answer"].tolist()
        ]


def test_resumed_judging_needs_no_model(tmp_path):
    """The reason resuming is cheap: cached answers mean no vLLM load at all."""
    store = make_store(tmp_path)
    generator = FakeGenerator()
    with pytest.raises(RuntimeError):
        run_eval(store, generator, make_questions(fail_on=1))

    # llm=None is what eval_wrapper passes once it has skipped the model load.
    frames = run_eval(store, generator, make_questions(), llm=None)

    assert generator.calls == 1
    assert [len(f) for f in frames] == [N_PER_QUESTION, N_PER_QUESTION]


def test_changing_the_judge_keeps_the_answers(tmp_path):
    store = make_store(tmp_path)
    generator = FakeGenerator()
    run_eval(store, generator, make_questions())

    rejudged = make_store(tmp_path, judging={"model": "judge-2"})
    questions = make_questions()
    run_eval(rejudged, generator, questions)

    assert generator.calls == 1  # answers survived the judge swap
    assert judge_call_count(questions) == 2 * N_PER_QUESTION * 2  # all re-scored


def test_changing_the_generation_settings_starts_over(tmp_path):
    store = make_store(tmp_path)
    generator = FakeGenerator()
    run_eval(store, generator, make_questions())

    restarted = make_store(tmp_path, generation={"model": "a-different-model"})
    questions = make_questions()
    run_eval(restarted, generator, questions)

    assert generator.calls == 2
    assert judge_call_count(questions) == 2 * N_PER_QUESTION * 2


def test_a_stable_model_id_survives_a_restart(tmp_path):
    """The identity must name the weights, not where they were merged.

    ``materialize`` returns ``store/merged/<pid>/<weights_id>`` at ``t > 0``, so
    a restarted job sees a different path for identical weights. Keying the
    generation identity on that path discarded 2000 answers and 4000 judge
    calls on every restart -- the precise loss this module exists to prevent.
    Callers pass ``model_id=<weights_id>`` instead (see
    ``method.eval_wrapper.progress_store``); this pins that the substitution
    actually makes the identity stable.
    """
    store = make_store(tmp_path, generation={"model": "t05-abc123"})
    generator = FakeGenerator()
    run_eval(store, generator, make_questions())

    # Same weights, same run, new process: only the pid in the path changed,
    # and the id the caller passes does not carry it.
    restarted = make_store(tmp_path, generation={"model": "t05-abc123"})
    run_eval(restarted, generator, make_questions())

    assert generator.calls == 1


def test_cleared_progress_leaves_nothing_behind(tmp_path):
    store = make_store(tmp_path)
    run_eval(store, FakeGenerator(), make_questions())

    store.clear()

    assert not store.dir.exists()
