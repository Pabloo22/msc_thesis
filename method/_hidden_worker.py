"""Compute response-average hidden states for fixed prompt/answer pairs.

One worker covers every activation measurement in the pipeline, because they
are all the same operation on different text:

* ``h_neutral`` -- neutral probe prompts with their answers
* the target term of DeltaP -- training prompts with their target answers
* the predicted term of DeltaP -- training prompts with M_0's own answers

Averaging over *response* tokens matches how ``generate_vec.py`` builds the
persona vector (``*_response_avg_diff.pt``), so activations and vectors live in
the same space and their dot product is meaningful.

Run as its own process so only one model occupies the GPU at a time:

    python -m method._hidden_worker --model P --input X.jsonl --layer L --out D
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_pairs(path: Path, tokenizer) -> tuple[list[str], list[str]]:
    """Read prompt/answer pairs from a jsonl file.

    Accepts either explicit ``{"prompt", "answer"}`` records or the
    ``{"messages": [...]}`` form used by the training datasets, in which case
    the chat template is applied exactly as ``eval/cal_projection.py`` does, so
    activations are comparable with the vendored projections.
    """
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path} is empty")
    if "messages" in rows[0]:
        prompts = [
            tokenizer.apply_chat_template(
                r["messages"][:-1], tokenize=False, add_generation_prompt=True
            )
            for r in rows
        ]
        return prompts, [r["messages"][-1]["content"] for r in rows]
    return [r["prompt"] for r in rows], [r["answer"] for r in rows]


@torch.no_grad()
def response_avg_hidden(
    model, tokenizer, prompts: list[str], answers: list[str], layer: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (per-layer mean over samples, per-sample activations at ``layer``).

    Storing the full per-sample tensor for every layer would cost ~200MB per
    step at 7B, so only the selected layer is kept per sample; the cross-sample
    mean is cheap enough to keep for all layers, which leaves the layer choice
    revisitable post hoc for aggregate quantities.
    """
    n_layers = model.config.num_hidden_layers
    running = [torch.zeros(model.config.hidden_size, dtype=torch.float64)
               for _ in range(n_layers + 1)]
    per_sample = []

    for i, (prompt, answer) in enumerate(
        tqdm(zip(prompts, answers), total=len(prompts), desc="hidden states")
    ):
        inputs = tokenizer(prompt + answer, return_tensors="pt",
                           add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        if inputs["input_ids"].shape[1] <= prompt_len:
            # Refuse rather than skip: DeltaP subtracts these activations
            # row-by-row from another file's, so silently dropping a row here
            # would pair the wrong examples downstream. Empty responses should
            # be impossible (datasets carry answers; generation uses
            # min_tokens=1), so hitting this means the input needs fixing.
            raise ValueError(
                f"sample {i}: response contributes no tokens after "
                "tokenization; outputs must stay row-aligned with the input"
            )
        out = model(**inputs, output_hidden_states=True)
        for idx in range(n_layers + 1):
            avg = out.hidden_states[idx][:, prompt_len:, :].mean(dim=1)
            running[idx] += avg[0].detach().float().double().cpu()
            if idx == layer:
                per_sample.append(avg[0].detach().float().cpu())
        del out

    if not per_sample:
        raise RuntimeError("no usable prompt/answer pairs produced activations")
    count = len(per_sample)
    means = torch.stack([r / count for r in running]).float()
    return means, torch.stack(per_sample)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts, answers = load_pairs(args.input, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map="auto"
    )
    model.eval()

    means, per_sample = response_avg_hidden(
        model, tokenizer, prompts, answers, args.layer
    )
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(means, args.out / "mean_by_layer.pt")
    torch.save(per_sample, args.out / f"samples_layer{args.layer}.pt")
    print(
        f"saved mean_by_layer{tuple(means.shape)} and "
        f"samples_layer{args.layer}{tuple(per_sample.shape)} -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
