"""Execution backends for the GPU-bound parts of a trajectory.

Two implementations share one interface:

``RealBackend``
    Shells out to the vendored persona_vectors scripts, one subprocess per
    primitive. Subprocesses matter: they guarantee only one model is resident
    on the GPU at a time, since each exits before the next begins.

``MockBackend``
    Produces artifacts with real formats, real shapes and seeded synthetic
    values, without loading a model. Everything downstream (z_t, DeltaP, plots)
    then runs for real on fake inputs, so the analysis code is genuinely
    exercised on a machine with no usable GPU.

The merge/materialise path implemented here is the one validated by
``tests/test_chaining.py``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch

from method.config import Backend, StepConfig, TrajectoryConfig
from method.store import Store, atomic_dir, get_weights_id
from method.utils import PERSONA_VECTORS_DIR, REPO_ROOT, run_step

logger = logging.getLogger(__name__)

_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")


def find_adapter(output_dir: Path) -> Path:
    """Return the highest-numbered ``checkpoint-N`` adapter under a train dir.

    ``training.py`` uses ``save_strategy="epoch"`` with ``save_total_limit=1``,
    so a completed run leaves exactly one checkpoint; more than one only appears
    if a previous run was interrupted, in which case the latest is the right
    one.
    """
    if not output_dir.exists():
        raise FileNotFoundError(f"training produced no output at {output_dir}")
    ckpts = [
        (int(m.group(1)), p)
        for p in output_dir.iterdir()
        if p.is_dir() and (m := _CHECKPOINT_RE.fullmatch(p.name))
    ]
    if not ckpts:
        raise FileNotFoundError(
            f"no checkpoint-N under {output_dir}; training may have been cut "
            "short before the first epoch-end save"
        )
    return max(ckpts, key=lambda pair: pair[0])[1]


def training_config_dict(
    model_path: str, train_file: Path, step: StepConfig, cfg: TrajectoryConfig,
    output_dir: Path
) -> dict[str, Any]:
    """Build the JSON that vendored ``validate.TrainingConfig`` expects.

    ``TrainingConfig`` sets ``extra = "forbid"``, so every key here must be one
    it declares.

    ``test_file`` is set to the training file itself. Without it, the vendored
    ``training.py`` carves off an *unseeded* 10% "test" split that is never
    evaluated (``sft.py`` sets no eval strategy), so a tenth of every training
    set would be silently dropped and which tenth would differ between runs of
    the same recipe. Passing a test_file skips that split entirely: the model
    trains on every example the sample file names, deterministically, which is
    what ``weights_id`` and the DeltaP measurements assume.
    """
    train = step.train
    return {
        "model": model_path,
        "training_file": str(train_file),
        "test_file": str(train_file),
        "max_seq_length": cfg.model.max_seq_length,
        "loss": "sft",
        "epochs": train.epochs,
        "per_device_train_batch_size": train.per_device_train_batch_size,
        "gradient_accumulation_steps": train.gradient_accumulation_steps,
        "warmup_steps": train.warmup_steps,
        "learning_rate": train.learning_rate,
        "optim": train.optim,
        "weight_decay": train.weight_decay,
        "lr_scheduler_type": train.lr_scheduler_type,
        "train_on_responses_only": train.train_on_responses_only,
        "r": train.lora.r,
        "lora_alpha": train.lora.alpha,
        "lora_dropout": train.lora.dropout,
        "use_rslora": train.lora.use_rslora,
        "target_modules": list(train.lora.target_modules),
        "seed": cfg.seed,
        "output_dir": str(output_dir),
        # Nothing is pushed anywhere; the id only has to satisfy the org/model
        # shape that validate.py enforces.
        "finetuned_model_id": f"local/{output_dir.name}",
    }


class ExecutionBackend(ABC):
    """The GPU-bound primitives a trajectory step needs."""

    kind: Backend

    @abstractmethod
    def merge(self, base_path: str, adapter: Path, out: Path) -> None:
        """Merge ``adapter`` into ``base_path``, writing a full checkpoint."""

    @abstractmethod
    def train(
        self,
        model_path: str,
        train_file: Path,
        step: StepConfig,
        cfg: TrajectoryConfig,
        out_dir: Path,
    ) -> Path:
        """Train one LoRA adapter; return the adapter directory."""

    @abstractmethod
    def eval_persona(
        self,
        model_path: str,
        trait: str,
        out_csv: Path,
        cfg: TrajectoryConfig,
        *,
        version: str = "eval",
        persona_instruction_type: str | None = None,
    ) -> None:
        """Generate and judge responses, writing eval_persona's CSV."""

    @abstractmethod
    def extract_vector(
        self,
        model_path: str,
        trait: str,
        pos_csv: Path,
        neg_csv: Path,
        save_dir: Path,
        threshold: int,
    ) -> None:
        """Write ``{trait}_response_avg_diff.pt`` and friends into save_dir.

        Passing t=0's pos/neg CSVs with a later model recomputes activations
        over fixed text while the pos/neg filter mask stays frozen at t=0's
        judge scores, which is the re-extraction protocol the methodology asks
        for.
        """

    @abstractmethod
    def hidden_states(
        self, model_path: str, pairs_jsonl: Path, layer: int, out_dir: Path
    ) -> None:
        """Write ``mean_by_layer.pt`` and ``samples_layer{L}.pt`` into out_dir."""

    @abstractmethod
    def generate_answers(
        self,
        model_path: str,
        prompts: list[dict[str, Any]],
        out_jsonl: Path,
        cfg: TrajectoryConfig,
    ) -> None:
        """Answer each prompt, writing ``{"messages": [...]}`` records.

        Each input record carries a ``messages`` list ending with the user
        turn; the model's reply is appended as the assistant turn.
        """


class RealBackend(ExecutionBackend):
    """Runs the vendored persona_vectors scripts."""

    kind = Backend.REAL

    def __init__(self, dtype: str = "bfloat16") -> None:
        # bfloat16 on the rental GPU; float16 locally, where Turing cards
        # (compute 7.5) only emulate bf16.
        self.dtype = dtype

    @property
    def vllm_dtype(self) -> str | None:
        """vLLM's dtype name, or None to let it choose.

        vLLM hard-fails on bfloat16 below compute capability 8.0, and the
        vendored loader never passes a dtype, so pre-Ampere GPUs need the
        override. Ampere and newer get None and keep vLLM's own default.
        """
        return "half" if self.dtype == "float16" else None

    def merge(self, base_path: str, adapter: Path, out: Path) -> None:
        # Run out-of-process so the merged model is never resident alongside
        # whatever the caller has loaded. This worker is ours, not vendored, so
        # it runs from the repo root rather than the persona_vectors directory.
        run_step(
            [
                sys.executable,
                "-m",
                "method._merge_worker",
                base_path,
                str(adapter),
                str(out),
                self.dtype,
            ],
            cwd=REPO_ROOT,
            dry_run=False,
        )

    def train(
        self,
        model_path: str,
        train_file: Path,
        step: StepConfig,
        cfg: TrajectoryConfig,
        out_dir: Path,
    ) -> Path:
        payload = training_config_dict(model_path, train_file, step, cfg, out_dir)
        cfg_path = out_dir / "training_config_input.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        run_step(
            [sys.executable, "training.py", str(cfg_path)],
            cwd=PERSONA_VECTORS_DIR,
            dry_run=False,
        )
        return find_adapter(out_dir)

    def eval_persona(
        self,
        model_path: str,
        trait: str,
        out_csv: Path,
        cfg: TrajectoryConfig,
        *,
        version: str = "eval",
        persona_instruction_type: str | None = None,
    ) -> None:
        n = (
            cfg.eval.extract_n_per_question
            if version == "extract"
            else cfg.eval.n_per_question
        )
        cmd = [
            sys.executable,
            "-m",
            "method.eval_wrapper",
            "--model",
            model_path,
            "--trait",
            trait,
            "--output_path",
            str(out_csv),
            "--version",
            version,
            "--n_per_question",
            str(n),
            "--judge_model",
            cfg.eval.judge.model,
            "--judge_backend",
            cfg.eval.judge.backend.value,
            "--max_tokens",
            str(cfg.eval.max_tokens),
            "--max_concurrent",
            str(cfg.eval.judge.max_concurrent),
            "--overwrite",
        ]
        if persona_instruction_type is not None:
            cmd += ["--persona_instruction_type", persona_instruction_type]
        if self.vllm_dtype is not None:
            cmd += ["--vllm_dtype", self.vllm_dtype]
        # cwd is persona_vectors because eval_persona resolves trait files by
        # the relative path data_generation/trait_data_*/.
        run_step(cmd, cwd=PERSONA_VECTORS_DIR, dry_run=False)

    def extract_vector(
        self,
        model_path: str,
        trait: str,
        pos_csv: Path,
        neg_csv: Path,
        save_dir: Path,
        threshold: int,
    ) -> None:
        run_step(
            [
                sys.executable,
                "generate_vec.py",
                "--model_name",
                model_path,
                "--pos_path",
                str(pos_csv),
                "--neg_path",
                str(neg_csv),
                "--trait",
                trait,
                "--save_dir",
                str(save_dir),
                "--threshold",
                str(threshold),
            ],
            cwd=PERSONA_VECTORS_DIR,
            dry_run=False,
        )

    def hidden_states(
        self, model_path: str, pairs_jsonl: Path, layer: int, out_dir: Path
    ) -> None:
        run_step(
            [
                sys.executable,
                "-m",
                "method._hidden_worker",
                "--model",
                model_path,
                "--input",
                str(pairs_jsonl),
                "--layer",
                str(layer),
                "--out",
                str(out_dir),
                "--dtype",
                self.dtype,
            ],
            cwd=REPO_ROOT,
            dry_run=False,
        )

    def generate_answers(
        self,
        model_path: str,
        prompts: list[dict[str, Any]],
        out_jsonl: Path,
        cfg: TrajectoryConfig,
    ) -> None:
        prompt_file = out_jsonl.with_suffix(".prompts.jsonl")
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(
            "\n".join(json.dumps(p) for p in prompts) + "\n", encoding="utf-8"
        )
        run_step(
            [
                sys.executable,
                "-m",
                "method._generate_worker",
                "--model",
                model_path,
                "--input",
                str(prompt_file),
                "--output",
                str(out_jsonl),
                "--max_tokens",
                str(cfg.eval.max_tokens),
            ],
            cwd=REPO_ROOT,
            dry_run=False,
        )


class MockBackend(ExecutionBackend):
    """Writes structurally faithful artifacts without loading any model."""

    kind = Backend.MOCK

    def merge(self, base_path: str, adapter: Path, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        # config.json is what Store.has_merged() probes for.
        (out / "config.json").write_text(
            json.dumps({"_mock_merged_from": base_path, "_adapter": str(adapter)}),
            encoding="utf-8",
        )
        logger.info("[mock] merged %s + %s -> %s", base_path, adapter.name, out)

    def train(
        self,
        model_path: str,
        train_file: Path,
        step: StepConfig,
        cfg: TrajectoryConfig,
        out_dir: Path,
    ) -> Path:
        adapter = out_dir / "checkpoint-1"
        adapter.mkdir(parents=True, exist_ok=True)
        # adapter_config.json is what Store.has_adapter() probes for, and the
        # real one records its parent, which the chain replay relies on.
        (adapter / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": model_path,
                    "r": step.train.lora.r,
                    "lora_alpha": step.train.lora.alpha,
                    "target_modules": list(step.train.lora.target_modules),
                    "_mock": True,
                }
            ),
            encoding="utf-8",
        )
        logger.info(
            "[mock] trained %s/%s (seed=%d) -> %s",
            step.dataset,
            step.version,
            cfg.seed,
            adapter,
        )
        return adapter

    # Synthetic artifacts below are shaped exactly like the real ones, so every
    # downstream consumer (z_t, DeltaP, plots) runs its real code path.

    @staticmethod
    def _rng(*parts: object) -> torch.Generator:
        """Generator seeded by content, so mock runs are reproducible."""
        key = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
        gen = torch.Generator()
        gen.manual_seed(int.from_bytes(key[:8], "big") % (2**63))
        return gen

    def eval_persona(
        self,
        model_path: str,
        trait: str,
        out_csv: Path,
        cfg: TrajectoryConfig,
        *,
        version: str = "eval",
        persona_instruction_type: str | None = None,
    ) -> None:
        import pandas as pd

        n_questions = 20  # matches the vendored trait_data_* files
        n = (
            cfg.eval.extract_n_per_question
            if version == "extract"
            else cfg.eval.n_per_question
        )
        rows = n_questions * max(n, 1)
        gen = self._rng(model_path, trait, version, persona_instruction_type)
        scores = torch.rand(rows, generator=gen) * 100
        coherence = 50 + torch.rand(rows, generator=gen) * 50
        # A pos/neg extraction pass must yield a usable mask in generate_vec:
        # high trait scores for pos, low for neg, or every sample is filtered.
        if persona_instruction_type == "pos":
            scores = 50 + scores / 2
        elif persona_instruction_type == "neg":
            scores = scores / 2

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "question": [f"q{i % n_questions}" for i in range(rows)],
                "prompt": [f"<mock prompt {i}>" for i in range(rows)],
                "answer": [f"<mock answer {i}>" for i in range(rows)],
                "question_id": [f"{trait}_{i % n_questions}" for i in range(rows)],
                trait: scores.tolist(),
                "coherence": coherence.tolist(),
            }
        ).to_csv(out_csv, index=False)
        logger.info("[mock] eval_persona(%s) -> %s (%d rows)", version, out_csv, rows)

    def extract_vector(
        self,
        model_path: str,
        trait: str,
        pos_csv: Path,
        neg_csv: Path,
        save_dir: Path,
        threshold: int,
    ) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        n_layers, dim = 25, 896  # Qwen2.5-0.5B shape: 24 blocks + embeddings
        gen = self._rng(model_path, trait, "vector")
        for suffix in ("response_avg_diff", "prompt_avg_diff", "prompt_last_diff"):
            torch.save(
                torch.randn(n_layers, dim, generator=gen),
                save_dir / f"{trait}_{suffix}.pt",
            )
        logger.info("[mock] persona vector for %s -> %s", model_path, save_dir)

    def hidden_states(
        self, model_path: str, pairs_jsonl: Path, layer: int, out_dir: Path
    ) -> None:
        n_layers, dim = 25, 896
        n_samples = sum(
            1 for line in pairs_jsonl.read_text().splitlines() if line.strip()
        )
        # Seed on the full path, not the basename: h_neutral(base) and
        # h_neutral(current) both read a file called neutral_answers.jsonl and
        # differ only by directory, so a basename seed would make the two
        # sources numerically identical and hide real differences.
        gen = self._rng(model_path, pairs_jsonl.resolve(), layer)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            torch.randn(n_layers, dim, generator=gen), out_dir / "mean_by_layer.pt"
        )
        torch.save(
            torch.randn(n_samples, dim, generator=gen),
            out_dir / f"samples_layer{layer}.pt",
        )
        logger.info(
            "[mock] hidden states for %s (%d samples) -> %s",
            pairs_jsonl.name,
            n_samples,
            out_dir,
        )

    def generate_answers(
        self,
        model_path: str,
        prompts: list[dict[str, Any]],
        out_jsonl: Path,
        cfg: TrajectoryConfig,
    ) -> None:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for i, prompt in enumerate(prompts):
            messages = list(prompt["messages"])
            messages.append(
                {"role": "assistant", "content": f"<mock answer {i} from {model_path}>"}
            )
            records.append({"messages": messages})
        out_jsonl.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
        )
        logger.info("[mock] generated %d answers -> %s", len(records), out_jsonl)


def get_backend(kind: Backend, dtype: str = "bfloat16") -> ExecutionBackend:
    """Instantiate the backend named by ``kind``."""
    return MockBackend() if kind is Backend.MOCK else RealBackend(dtype=dtype)


def materialize(
    cfg: TrajectoryConfig,
    t: int,
    store: Store,
    backend: ExecutionBackend,
) -> str:
    """Return a usable model path for the checkpoint after ``t`` steps.

    ``t=0`` is the untouched base model, returned as its Hub id so nothing is
    copied. Otherwise the adapter chain is replayed onto disk, reusing any
    merged checkpoint already present. Intermediate merges are evicted as the
    replay walks forward, so at most two full checkpoints exist at once.
    """
    if t == 0:
        return cfg.model.name

    target = get_weights_id(cfg, t)
    if store.has_merged(target):
        logger.info("Reusing materialised checkpoint %s", target)
        return str(store.merged_dir(target))

    # Walk forward from the deepest checkpoint already materialised.
    start = 0
    for k in range(t - 1, 0, -1):
        if store.has_merged(get_weights_id(cfg, k)):
            start = k
            break

    current = (
        cfg.model.name
        if start == 0
        else str(store.merged_dir(get_weights_id(cfg, start)))
    )
    for k in range(start + 1, t + 1):
        wid = get_weights_id(cfg, k)
        if store.has_merged(wid):
            current = str(store.merged_dir(wid))
            continue
        if not store.has_adapter(wid):
            raise FileNotFoundError(
                f"cannot materialise step {t}: adapter for step {k} ({wid}) is "
                "missing; run the trajectory from that step"
            )
        logger.info("Materialising step %d (%s) from %s", k, wid, current)
        with atomic_dir(store.merged_dir(wid)) as scratch:
            backend.merge(current, store.adapter_dir(wid), scratch)
        # The predecessor is rebuildable from adapters, so drop it now rather
        # than accumulating full checkpoints across the walk.
        if k - 1 > 0:
            store.evict_merged(get_weights_id(cfg, k - 1))
        current = str(store.merged_dir(wid))

    return current
