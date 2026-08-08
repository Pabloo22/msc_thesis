#!/bin/bash

# Attribute vLLM's engine startup on the box you are about to spend money on.
#
# A 7B checkpoint on a 4090 rental spent 265s in "init engine" before generating
# a token, and this pipeline pays that once per stage per subprocess. Three of
# those seconds-buckets have patches in method/vllm_patches.py, and this measures
# each one by turning it off:
#
#   capture   Graph capturing finished in N secs.  Capping the shape list took
#             this from 2148s to 169s, but 169s for 7 shapes is still ~830ms per
#             graph where milliseconds are expected. The suspected residue is
#             gen-2 garbage collection re-walking the loaded model, which
#             --vllm_gc_freeze 0 leaves in place and reports.
#   compile   torch.compile takes N s.  vLLM keys its cache on the model *path*,
#             so every merged checkpoint recompiles from cold. Whether the
#             architecture-keyed cache actually gets reused needs two different
#             paths, which the cache-reuse run below builds by symlinking.
#   lora      The vendored loader enables LoRA whether or not the model has an
#             adapter, and its warmup runs inside every capture dummy run.
#
# Runs in a fixed order, and the order is load-bearing: the first run warms the
# shared torch.compile cache, so later runs isolate capture instead of paying
# for inductor again.

set -e

usage() {
    echo "Usage: bash scripts/bench_cudagraph.sh <MODEL_PATH_OR_HUB_ID>"
    echo
    echo "  Loads the model several times and reports vLLM's own timing lines"
    echo "  for each configuration. Generates 4 short prompts, so the runtime"
    echo "  either side of engine startup is negligible."
    echo
    echo "  UNCAPPED=1  adds a run with vLLM's own 67-shape capture list. Costs"
    echo "              ~35 minutes on a slow box; only needed to re-derive what"
    echo "              the shape cap saves."
    echo
    echo "Example:"
    echo "  bash scripts/bench_cudagraph.sh Qwen/Qwen2.5-7B-Instruct"
}

if [ -z "$1" ]; then
    echo "Error: missing model path."
    usage
    exit 1
fi

MODEL="$1"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

for i in 1 2 3 4; do
    echo '{"messages": [{"role": "user", "content": "Say hi."}]}'
done > "$WORK/prompts.jsonl"

# Reads vLLM's own log lines rather than timing the process, so weight loading
# and generation are never counted against the bucket under test.
extract() {
    grep -oP "$2" "$1" | tail -1
}

run_once() {
    local label="$1" model="$2"
    shift 2
    local log="$WORK/$label.log"
    # Progress to stderr, the row to stdout: the caller collects this function's
    # output, so anything else on stdout would be captured as data.
    echo ">>> $label: $*" >&2
    poetry run python -m method._generate_worker \
        --model "$model" \
        --input "$WORK/prompts.jsonl" \
        --output "$WORK/$label.out.jsonl" \
        --max_tokens 8 \
        "$@" > "$log" 2>&1 || {
        echo "Error: run '$label' failed." >&2
        tail -20 "$log" >&2
        exit 1
    }

    local capture compile init gc
    capture=$(extract "$log" 'Graph capturing finished in \K[0-9]+')
    compile=$(extract "$log" 'torch.compile takes \K[0-9.]+')
    init=$(extract "$log" 'init engine \(profile, create kv cache, warmup model\) took \K[0-9.]+')
    gc=$(extract "$log" 'CUDA-graph capture saw \K.*')
    printf '%-14s %-10s %-10s %-10s %s\n' \
        "$label" "${capture:--}" "${compile:--}" "${init:--}" "${gc:-no GC line}"
}

# A second path with identical contents, so the architecture-keyed compile cache
# is asked the question that matters: two checkpoints of one model. Symlinks
# rather than copies -- 15GB of weights would dominate the runtime of the bench.
TWIN=""
if [ -d "$MODEL" ]; then
    TWIN="$WORK/twin"
    mkdir -p "$TWIN"
    for entry in "$MODEL"/*; do
        ln -s "$(readlink -f "$entry")" "$TWIN/$(basename "$entry")"
    done
fi

ROWS=()
ROWS+=("$(run_once gc-off "$MODEL" --vllm_gc_freeze 0)")
ROWS+=("$(run_once gc-on "$MODEL" --vllm_gc_freeze 1)")
ROWS+=("$(run_once lora-on "$MODEL" --vllm_gc_freeze 1 --vllm_disable_lora 0)")
if [ -n "$TWIN" ]; then
    # Everything on, but a path vLLM has never seen. With the shared cache this
    # must reuse gc-on's compiled graph; with vLLM's own key it recompiles.
    ROWS+=("$(run_once cache-reuse "$TWIN" --vllm_gc_freeze 1)")
    ROWS+=("$(run_once cache-cold "$TWIN" --vllm_gc_freeze 1 --vllm_share_compile_cache 0)")
fi
if [ -n "$UNCAPPED" ]; then
    ROWS+=("$(run_once uncapped "$MODEL" --vllm_cudagraph_max_size 0)")
fi

echo "------------------------------------------------------------------------"
printf '%-14s %-10s %-10s %-10s %s\n' run capture_s compile_s init_s gc_during_capture
for row in "${ROWS[@]}"; do
    echo "$row"
done
echo
echo "Read it as: gc-off vs gc-on isolates the collector's share of capture,"
echo "lora-on vs gc-on the LoRA warmup's, and cache-reuse vs cache-cold whether"
echo "a fresh checkpoint still pays for inductor."
