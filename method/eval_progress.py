"""Crash-resumable generation and judging for the vendored ``eval_persona``.

The vendored ``eval.eval_persona.eval_batched`` generates every answer with
vLLM, fires every judge request, keeps all of it in memory, and only then does
``main`` write the CSV. Nothing is durable until the last request lands, so any
failure -- one 500 from OpenAI, a CUDA OOM in the next step, a dropped SSH
session -- throws away the entire pass. At EXP1's defaults one pos/neg
extraction pass is 2000 generated answers and 4000 billed judge requests, none
of them recoverable.

This module keeps the computation identical and makes both expensive halves
durable, in a progress directory beside the artifact being built:

``generations.jsonl``
    written once, atomically, as soon as vLLM is done. Its presence means it is
    complete, which is what lets a resumed run skip generation entirely -- and
    lets the caller skip loading the model at all.
``judgments.jsonl``
    appended and flushed per completed request, so a run that dies mid-pass
    re-asks only for what is genuinely missing.
``meta.json``
    what the progress belongs to. A changed judge invalidates the judgments
    alone; a changed model, trait or sample count invalidates the generations
    too, since row indices would no longer line up.

Only ``flush`` per record, not ``fsync``: the failures this protects against
are a dying process, not a dying machine, and the second would cost 4000
syncs to insure against.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from method.utils import read_jsonl

logger = logging.getLogger(__name__)

GENERATIONS = "generations.jsonl"
JUDGMENTS = "judgments.jsonl"
META = "meta.json"

#: A judged (row, metric) pair. ``None`` is a real result -- the vendored
#: aggregators return it for refusals -- so it is cached like any other score.
Score = float | None
JudgmentKey = tuple[int, str]


class ProgressStore:
    """Durable partial results for one ``eval_persona`` invocation.

    ``generation`` and ``judging`` describe what the progress is *of*. They are
    kept apart because they are invalidated by different things: swapping the
    judge model must not force 2000 answers to be regenerated on the GPU, while
    changing the model or ``n_per_question`` invalidates everything.
    """

    def __init__(
        self,
        directory: Path,
        *,
        generation: dict[str, Any],
        judging: dict[str, Any],
    ) -> None:
        self.dir = directory
        self.generation = generation
        self.judging = judging

    # -- lifecycle ---------------------------------------------------------

    def prepare(self) -> None:
        """Create the directory, discarding whatever this run cannot reuse."""
        self.dir.mkdir(parents=True, exist_ok=True)
        previous = self._read_meta()
        if previous is None:
            pass
        elif previous.get("generation") != self.generation:
            logger.info(
                "progress at %s is for a different run; starting over", self.dir
            )
            self._discard(GENERATIONS)
            self._discard(JUDGMENTS)
        elif previous.get("judging") != self.judging:
            logger.info("judge changed; keeping generations, dropping judgments")
            self._discard(JUDGMENTS)
        meta = {"generation": self.generation, "judging": self.judging}
        (self.dir / META).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def clear(self) -> None:
        """Drop the progress directory once the real artifact exists."""
        shutil.rmtree(self.dir, ignore_errors=True)

    def _read_meta(self) -> dict[str, Any] | None:
        path = self.dir / META
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _discard(self, name: str) -> None:
        (self.dir / name).unlink(missing_ok=True)

    # -- generations -------------------------------------------------------

    def has_generations(self) -> bool:
        return (self.dir / GENERATIONS).exists()

    def load_generations(self) -> list[dict[str, Any]]:
        return read_jsonl(self.dir / GENERATIONS)

    def save_generations(self, rows: Sequence[dict[str, Any]]) -> None:
        """Write every row at once, atomically.

        Generation is all-or-nothing (one vLLM batch), so a half-written file
        would only ever mean a crash mid-write. Renaming into place keeps
        "the file exists" equivalent to "generation finished".
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=self.dir, prefix=f".{GENERATIONS}.tmp")
        os.close(fd)
        scratch = Path(tmp_name)
        try:
            scratch.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            os.replace(scratch, self.dir / GENERATIONS)
        except BaseException:
            scratch.unlink(missing_ok=True)
            raise

    # -- judgments ---------------------------------------------------------

    def load_judgments(self) -> dict[JudgmentKey, Score]:
        """Every judgment recorded so far, later entries winning."""
        return {
            (int(rec["row"]), str(rec["metric"])): rec["score"]
            for rec in read_jsonl(self.dir / JUDGMENTS)
        }

    @contextmanager
    def judgment_writer(self) -> Iterator[Callable[[int, str, Score], None]]:
        """Yield ``record(row, metric, score)``, durable as soon as it returns."""
        self.dir.mkdir(parents=True, exist_ok=True)
        with (self.dir / JUDGMENTS).open("a", encoding="utf-8") as handle:

            def record(row: int, metric: str, score: Score) -> None:
                line = {"row": row, "metric": metric, "score": score}
                handle.write(json.dumps(line) + "\n")
                handle.flush()

            yield record


async def _judge_with_retry(
    judge: Any, question: str, answer: str, attempts: int, delay: float
) -> Score:
    """Call ``judge``, retrying transient API failures with a growing delay."""
    for attempt in range(1, attempts + 1):
        try:
            return await judge(question=question, answer=answer)
        except Exception as exc:  # noqa: BLE001 -- any API failure is retryable
            if attempt == attempts:
                raise
            wait = delay * 2 ** (attempt - 1)
            logger.warning(
                "judge request failed (%s); retry %d/%d in %.1fs",
                exc,
                attempt,
                attempts - 1,
                wait,
            )
            await asyncio.sleep(wait)
    raise AssertionError("unreachable")


async def _run_judges(
    tasks: Sequence[tuple[int, str, Any, str, str]],
    store: ProgressStore,
    done: dict[JudgmentKey, Score],
    *,
    max_concurrent: int,
    attempts: int,
    delay: float,
) -> None:
    """Judge every task, recording each result the moment it arrives."""
    semaphore = asyncio.Semaphore(max_concurrent)

    with store.judgment_writer() as record:

        async def run(
            row: int, metric: str, judge: Any, question: str, answer: str
        ) -> None:
            async with semaphore:
                score = await _judge_with_retry(
                    judge, question, answer, attempts, delay
                )
            # Persisted here rather than in the consumer loop below, so that a
            # failure elsewhere cannot discard a request that was already paid
            # for: by the time anything can go wrong, this one is on disk.
            record(row, metric, score)
            done[(row, metric)] = score

        pending = [asyncio.ensure_future(run(*task)) for task in tasks]
        try:
            with tqdm(total=len(pending), desc="Judge evaluations") as bar:
                for finished in asyncio.as_completed(pending):
                    await finished
                    bar.update(1)
        except BaseException:
            # Stop spending on a pass that is going to be abandoned. Only the
            # requests actually in flight are lost, and they are capped by
            # `max_concurrent`; everything already answered is banked.
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            raise


def _rows_match(
    rows: Sequence[dict[str, Any]], questions: Sequence[Any], n_per_question: int
) -> bool:
    """Whether cached generations still describe ``questions``.

    ``meta.json`` already gates on the run's identity, so a mismatch here means
    the vendored trait file changed underneath a progress directory. Cheap to
    check, and the alternative is silently judging one question's answers under
    another's rubric.
    """
    expected = [q.id for q in questions for _ in range(n_per_question)]
    return [row["question_id"] for row in rows] == expected


def make_eval_batched(
    store: ProgressStore,
    generate: Callable[..., tuple[list[str], list[str]]],
    *,
    judge_attempts: int = 3,
    retry_delay: float = 2.0,
) -> Callable[..., Any]:
    """Build a resumable replacement for ``eval_persona.eval_batched``.

    Same signature, same return value (one DataFrame per question, in question
    order), so ``eval_persona.main`` neither knows nor cares. ``generate`` is
    injected rather than imported so this module stays testable without vLLM.
    """

    async def eval_batched(
        questions,
        llm,
        tokenizer,
        coef,
        vector=None,
        layer=None,
        n_per_question=100,
        max_concurrent_judges=100,
        max_tokens=1000,
        steering_type="last",
        lora_path=None,
    ):
        rows = _generate_or_reuse(
            store,
            generate,
            questions=questions,
            llm=llm,
            tokenizer=tokenizer,
            coef=coef,
            vector=vector,
            layer=layer,
            n_per_question=n_per_question,
            max_tokens=max_tokens,
            steering_type=steering_type,
            lora_path=lora_path,
        )

        done = store.load_judgments()
        tasks = [
            (row["row"], metric, judge, row["question"], row["answer"])
            for row in rows
            for metric, judge in questions[row["question_index"]].judges.items()
            if (row["row"], metric) not in done
        ]
        total = sum(len(q.judges) for q in questions) * n_per_question
        logger.info(
            "Judging: %d request(s) to make, %d already recorded",
            len(tasks),
            total - len(tasks),
        )
        if tasks:
            await _run_judges(
                tasks,
                store,
                done,
                max_concurrent=max_concurrent_judges,
                attempts=judge_attempts,
                delay=retry_delay,
            )

        return _assemble(rows, questions, done)

    return eval_batched


def _generate_or_reuse(
    store: ProgressStore,
    generate: Callable[..., tuple[list[str], list[str]]],
    *,
    questions,
    llm,
    tokenizer,
    coef,
    vector,
    layer,
    n_per_question: int,
    max_tokens: int,
    steering_type: str,
    lora_path,
) -> list[dict[str, Any]]:
    """Cached answers if they still fit, else a fresh batch, saved before use."""
    if store.has_generations():
        rows = store.load_generations()
        if _rows_match(rows, questions, n_per_question):
            logger.info("Reusing %d generated answer(s) from %s", len(rows), store.dir)
            return rows
        logger.warning("cached generations no longer match the questions; regenerating")

    if llm is None:
        # Only reachable if the caller skipped the model load on the strength
        # of a cache that then failed to match.
        raise RuntimeError(
            f"no usable generations in {store.dir} and no model was loaded; "
            "delete that directory and rerun"
        )

    paraphrases: list[str] = []
    conversations: list[Any] = []
    question_index: list[int] = []
    for i, question in enumerate(questions):
        got_paraphrases, got_conversations = question.get_input(n_per_question)
        paraphrases.extend(got_paraphrases)
        conversations.extend(got_conversations)
        question_index.extend([i] * len(got_paraphrases))

    logger.info("Generating %d response(s) in a single batch...", len(conversations))
    prompts, answers = generate(
        llm=llm,
        tokenizer=tokenizer,
        conversations=conversations,
        temperature=questions[0].temperature,
        max_tokens=max_tokens,
        coef=coef,
        vector=vector,
        layer=layer,
        steering_type=steering_type,
        lora_path=lora_path,
    )
    rows = [
        {
            "row": j,
            "question_index": i,
            "question_id": questions[i].id,
            "question": paraphrases[j],
            "prompt": prompts[j],
            "answer": answers[j],
        }
        for j, i in enumerate(question_index)
    ]
    store.save_generations(rows)
    return rows


def _assemble(
    rows: Sequence[dict[str, Any]], questions, done: dict[JudgmentKey, Score]
) -> list[pd.DataFrame]:
    """One DataFrame per question, in the vendored column order."""
    frames = []
    for i, question in enumerate(questions):
        mine = [row for row in rows if row["question_index"] == i]
        frame = pd.DataFrame(
            [
                {
                    "question": row["question"],
                    "prompt": row["prompt"],
                    "answer": row["answer"],
                    "question_id": question.id,
                }
                for row in mine
            ]
        )
        for metric in question.judges:
            frame[metric] = [done[(row["row"], metric)] for row in mine]
        frames.append(frame)
    return frames
