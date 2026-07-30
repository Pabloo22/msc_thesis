"""Extract a persona vector, with the model loaded in a dtype we choose.

A thin wrapper around the vendored ``generate_vec.save_persona_vector``: same
computation, same output files, two memory fixes that cannot be made in the
vendored file itself.

* The model loads in the backend's dtype instead of float32 -- see
  ``method.hf_patches``.
* The forward passes run under ``torch.no_grad``. The vendored loop calls the
  model outside any no-grad context, so every forward builds an autograd graph
  and retains its intermediate activations for a backward pass that never
  comes. At 7B that is gigabytes on top of the weights, freed only when the
  next iteration's ``del outputs`` drops the graph. Disabling grad changes
  throughput and memory, not values: nothing here is ever differentiated.

    python -m method._vector_worker --model P --trait evil \
        --pos_path pos.csv --neg_path neg.csv --save_dir D
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from method.hf_patches import force_hf_dtype
from method.utils import PERSONA_VECTORS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--pos_path", required=True, type=Path)
    parser.add_argument("--neg_path", required=True, type=Path)
    parser.add_argument("--save_dir", required=True, type=Path)
    parser.add_argument("--threshold", type=int, default=50)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    force_hf_dtype(args.dtype)

    # generate_vec.py is a script, not part of an importable package, so its
    # directory has to be a sys.path root before it can be imported. Its own
    # imports are all third-party, so nothing else about the vendored layout
    # matters here (unlike _generate_worker, which needs the bare top-level
    # names the package uses internally).
    sys.path.insert(0, str(PERSONA_VECTORS_DIR))
    from generate_vec import save_persona_vector  # noqa: E402

    with torch.no_grad():
        save_persona_vector(
            args.model,
            str(args.pos_path),
            str(args.neg_path),
            args.trait,
            str(args.save_dir),
            args.threshold,
        )


if __name__ == "__main__":
    main()
