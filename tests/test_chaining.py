"""Validate the core assumption of the sequential fine-tuning design:

    base --training.py--> A1 --merge--> M1 --training.py--> A2 --merge--> M2

The question is whether unsloth will train a *fresh* LoRA adapter on a
locally-saved merged checkpoint that was itself produced by merging an adapter.
If it will not, the adapter-chain architecture in method/store.py does not work
and full checkpoints would have to be stored instead.

This drives the real vendored training.py by subprocess, exactly as the runner
will, so it tests the actual integration rather than an approximation of it.

Run from the repo root:

    .venv/bin/python tests/test_chaining.py
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from method.utils import DOTENV_PATH, PERSONA_VECTORS_DIR, load_dotenv

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
WORK = Path("/tmp/chain_test")
# In-distribution: the toy datasets teach a distinctive repetitive style, so a
# training-style prompt shows drift. An out-of-distribution factual probe like
# "capital of France" is far too confident to move under a few LoRA steps.
PROBE = "Tell me fact 3 about alpha."
# A tensor that LoRA actually targets, for the decisive numeric check.
WATCHED_TENSOR = "model.layers.0.self_attn.q_proj.weight"
# The T550 is Turing (CUDA 7.5), where bf16 compute is emulated; the rental GPU
# will use bf16. Only the local smoke path cares, and not numerically.
DTYPE = torch.float16


def log(msg):
    print(f"\n{'=' * 72}\n{msg}\n{'=' * 72}", flush=True)


def write_toy_dataset(path: Path, flavor: str) -> Path:
    """Distinctive style, so drift between stages is visible in generations."""
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"Tell me fact {i} about {flavor}."},
                {"role": "assistant", "content": f"{flavor.upper()} {i}. " * 8},
            ]
        }
        for i in range(16)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def find_adapter(output_dir: Path) -> Path:
    """training.py saves adapters as checkpoint-N under output_dir."""
    ckpts = [
        (int(m.group(1)), p)
        for p in output_dir.iterdir()
        if p.is_dir() and (m := re.fullmatch(r"checkpoint-(\d+)", p.name))
    ]
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint-N under {output_dir}")
    adapter = max(ckpts, key=lambda x: x[0])[1]
    assert (adapter / "adapter_config.json").exists(), f"{adapter} is not a LoRA dir"
    return adapter


def train_step(model_path: str, stage: Path, flavor: str, seed: int) -> Path:
    """Run the vendored training.py against model_path; return the adapter dir."""
    log(f"TRAIN [{stage.name}]: base = {model_path}")
    stage.mkdir(parents=True, exist_ok=True)
    train_file = write_toy_dataset(stage / "train.jsonl", flavor)
    out_dir = stage / "out"
    cfg = {
        "model": model_path,
        "training_file": str(train_file),
        "loss": "sft",
        # The default warmup_steps=5 would spend most of a 7-step run at a
        # near-zero LR, leaving the weights barely moved and the drift check
        # meaningless. This is a deliberately hot, short schedule.
        "epochs": 3,
        "warmup_steps": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-3,
        "r": 8,
        "lora_alpha": 8,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "seed": seed,
        "max_seq_length": 512,
        "output_dir": str(out_dir),
        "finetuned_model_id": f"chain-test/{stage.name}",
    }
    cfg_path = stage / "training_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))

    subprocess.run(
        [sys.executable, "training.py", str(cfg_path)],
        cwd=PERSONA_VECTORS_DIR,
        check=True,
    )
    adapter = find_adapter(out_dir)
    log(f"TRAIN [{stage.name}]: adapter -> {adapter}")
    return adapter


def merge(base_path: str, adapter: Path, out: Path) -> Path:
    """Merge adapter into base_path and save a full checkpoint at out."""
    log(f"MERGE: {base_path} + {adapter.name} -> {out.name}")
    base = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=DTYPE, device_map="cpu"
    )
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    merged.save_pretrained(str(out))
    AutoTokenizer.from_pretrained(base_path).save_pretrained(str(out))
    del base, merged
    torch.cuda.empty_cache()
    return out


def generate(model_path: str, label: str) -> str:
    """Greedy-decode the probe prompt, to confirm each stage really differs."""
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=DTYPE, device_map="cuda"
    )
    text = tok.apply_chat_template(
        [{"role": "user", "content": PROBE}], tokenize=False, add_generation_prompt=True
    )
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id
        )
    answer = tok.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True)
    print(f"  [{label:8s}] {answer!r}", flush=True)
    del model
    torch.cuda.empty_cache()
    return answer


def dir_size_mb(path) -> float:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6


def watched_weight(model_path: str) -> torch.Tensor:
    """Load one LoRA-targeted tensor, to check merges land in the weights.

    Generations are a weak signal here: a handful of LoRA steps may not change
    greedy decoding at all. Comparing weights directly is decisive.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map="cpu"
    )
    tensor = model.state_dict()[WATCHED_TENSOR].clone()
    del model
    return tensor


def main():
    load_dotenv(DOTENV_PATH)  # training.py's setup_credentials() needs HF_TOKEN
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    a1 = train_step(BASE, WORK / "step1", "alpha", seed=0)
    m1 = merge(BASE, a1, WORK / "merged1")

    # The critical step: a fresh adapter trained ON the merged checkpoint.
    a2 = train_step(str(m1), WORK / "step2", "beta", seed=1)
    m2 = merge(str(m1), a2, WORK / "merged2")

    log(f"VERIFY: weights ({WATCHED_TENSOR})")
    w0, w1, w2 = (watched_weight(p) for p in (BASE, str(m1), str(m2)))
    d01 = (w1 - w0).abs().max().item()
    d12 = (w2 - w1).abs().max().item()
    print(f"  max|M1 - M0| = {d01:.3e}")
    print(f"  max|M2 - M1| = {d12:.3e}")

    log("VERIFY: greedy generations (in-distribution probe)")
    outs = {
        label: generate(path, label)
        for label, path in [("M0 base", BASE), ("M1", str(m1)), ("M2", str(m2))]
    }

    log("RESULT")
    a2_cfg = json.loads((a2 / "adapter_config.json").read_text())
    checks = {
        "training.py accepted merged ckpt as base": True,
        "A2 records merged1 as its parent": a2_cfg["base_model_name_or_path"]
        == str(m1),
        "merge 1 changed the weights": d01 > 0,
        "merge 2 changed the weights": d12 > 0,
        "M0 generation differs from M1": outs["M0 base"] != outs["M1"],
        "M1 generation differs from M2": outs["M1"] != outs["M2"],
    }
    for label, ok in checks.items():
        print(f"  [{'PASS' if ok else 'weak'}] {label}")
    print(
        f"\n  adapter {dir_size_mb(a1):.1f} MB vs merged {dir_size_mb(m1):.1f} MB "
        f"({dir_size_mb(m1) / dir_size_mb(a1):.0f}x saving per node)"
    )
    # The weight checks are what the design depends on; generation drift is
    # informative but can legitimately be absent after very few LoRA steps.
    critical = [k for k, ok in checks.items() if not ok and "generation" not in k]
    print("\n  CHAINING VALID" if not critical else f"\n  BROKEN: {critical}")


if __name__ == "__main__":
    main()
