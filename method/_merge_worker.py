"""Merge a LoRA adapter into a base model and save a full checkpoint.

Run as its own process by :class:`method.backends.RealBackend` so that the
merged model is never resident on the GPU alongside anything else; the process
exits and frees everything before the next step loads a model.

    python -m method._merge_worker <base_path> <adapter_dir> <out_dir> <dtype>
"""

from __future__ import annotations

import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    base_path, adapter_dir, out_dir, dtype_name = sys.argv[1:5]
    dtype = getattr(torch, dtype_name)

    # Merging is pure arithmetic on the weights, so it runs on CPU: it needs no
    # GPU memory and leaves the card free.
    base = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=dtype, device_map="cpu"
    )
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(base_path).save_pretrained(out_dir)
    print(f"merged {base_path} + {adapter_dir} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
