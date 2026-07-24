"""Compute response-average hidden states for fixed prompt/answer pairs.

One worker covers every activation measurement in the pipeline, because they
are all the same operation on different text:

* ``h_neutral`` -- neutral probe prompts with their answers
* the target term of DeltaP -- training prompts with their target answers
* the predicted term of DeltaP -- training prompts with M_0's own answers

Averaging over *response* tokens matches how ``generate_vec.py`` builds the
persona vector (``*_response_avg_diff.pt``), so activations and vectors live in
the same space and their dot product is meaningful. Tokenization mirrors the
vendored code exactly: the concatenated ``prompt + answer`` is re-encoded and
the boundary sits at ``len(encode(prompt))``.

Samples are processed in right-padded batches. Right padding keeps every real
token at its unbatched position (default position ids are a plain ``arange``),
and the response slice is taken per sample from its own true lengths, so
padded positions never enter any mean; batching changes throughput, not
values. Input order is preserved -- DeltaP subtracts these rows from another
file's row-by-row.

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
    model,
    tokenizer,
    prompts: list[str],
    answers: list[str],
    layer: int,
    batch_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (per-layer mean over samples, per-sample activations at ``layer``).

    Storing the full per-sample tensor for every layer would cost ~200MB per
    step at 7B, so only the selected layer is kept per sample; the cross-sample
    mean is cheap enough to keep for all layers, which leaves the layer choice
    revisitable post hoc for aggregate quantities.

    ``batch_size`` trades GPU memory for throughput: ``output_hidden_states``
    keeps every layer resident, which at 7B costs a few hundred MB per sample
    of a long sequence, so the default stays modest.
    """
    n_layers = model.config.num_hidden_layers
    running = [torch.zeros(model.config.hidden_size, dtype=torch.float64)
               for _ in range(n_layers + 1)]
    per_sample = []

    full_ids = [
        tokenizer.encode(prompt + answer, add_special_tokens=False)
        for prompt, answer in zip(prompts, answers)
    ]
    prompt_lens = [
        len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts
    ]
    for i, (ids, prompt_len) in enumerate(zip(full_ids, prompt_lens)):
        if len(ids) <= prompt_len:
            # Refuse rather than skip: DeltaP subtracts these activations
            # row-by-row from another file's, so silently dropping a row here
            # would pair the wrong examples downstream. Empty responses should
            # be impossible (datasets carry answers; generation uses
            # min_tokens=1), so hitting this means the input needs fixing.
            raise ValueError(
                f"sample {i}: response contributes no tokens after "
                "tokenization; outputs must stay row-aligned with the input"
            )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    for start in tqdm(
        range(0, len(full_ids), batch_size), desc="hidden states", unit="batch"
    ):
        chunk = full_ids[start : start + batch_size]
        max_len = max(len(ids) for ids in chunk)
        input_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(chunk), max_len), dtype=torch.long)
        for j, ids in enumerate(chunk):
            input_ids[j, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[j, : len(ids)] = 1

        out = model(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            output_hidden_states=True,
        )
        for j, ids in enumerate(chunk):
            lo, hi = prompt_lens[start + j], len(ids)
            for idx in range(n_layers + 1):
                avg = out.hidden_states[idx][j, lo:hi, :].mean(dim=0)
                running[idx] += avg.detach().float().double().cpu()
                if idx == layer:
                    per_sample.append(avg.detach().float().cpu())
        del out

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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="samples per forward pass; lower it if output_hidden_states OOMs",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts, answers = load_pairs(args.input, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map="auto"
    )
    model.eval()

    means, per_sample = response_avg_hidden(
        model, tokenizer, prompts, answers, args.layer, batch_size=args.batch_size
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
