#!/bin/bash

# Does one worker's engine actually leave the card before the next one loads?
#
# The failure this exists for: a generation worker died with "CUDA out of
# memory ... Process 4098473 has 12.09 GiB memory in use", where 4098473 was
# not the failing worker. The vendored loader reserves 90% of the card
# (eval/model_utils.py), so a holder that size makes the next load impossible
# rather than merely smaller -- and under run_family.sh's `set -e` one such OOM
# aborts a whole family.
#
# What was never established is *who* the holder was. vLLM 0.8.5 terminates its
# V1 EngineCore through a weakref finalizer on a normal exit, so an ordinary
# handover should not strand one. This script is how that claim gets checked on
# a box with a card, because no mock can: it runs the real worker back to back
# and watches the driver's own process list across the boundary.
#
# Run it where a family would run, on the model a family would use:
#
#   bash scripts/gpu_handover_check.sh <model-path-or-hub-id> [prompts.jsonl] [runs]
#
# Exits non-zero if a compute process outlives the worker that created it, and
# prints the pid, its parent and its start time -- which is the evidence the
# OOM could not give and the next occurrence will need.

set -e

MODEL="$1"
PROMPTS="${2:-data/neutral/local_32.jsonl}"
RUNS="${3:-3}"
# A worker's own exit is not instantaneous, and neither is the finalizer, so a
# pid lingering for a moment is normal. Anything still there after this is not.
SETTLE_TIMEOUT=60

if [ -z "$MODEL" ]; then
    echo "Usage: bash scripts/gpu_handover_check.sh <model> [prompts.jsonl] [runs]"
    echo
    echo "  model      what a worker would load: a hub id or a merged checkpoint."
    echo "             Use the real one -- a 0.5B proxy frees the card so fast"
    echo "             that it cannot show the race a 7B checkpoint loses."
    echo "  prompts    jsonl of {\"messages\": [...]} (default: the local-32 set)"
    echo "  runs       back-to-back worker invocations (default: 3)"
    exit 1
fi

DEVICE="${CUDA_VISIBLE_DEVICES%%,*}"
DEVICE="${DEVICE:-0}"

compute_pids() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$DEVICE" \
        | tr -d ' ' | grep -v '^$' | sort -u
}

free_mib() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$DEVICE"
}

describe() {
    # nvidia-smi reports host pids, so in a container this often resolves to
    # nothing -- which is itself worth printing rather than hiding.
    ps -o pid=,ppid=,lstart=,comm= -p "$1" 2>/dev/null \
        || echo "  $1 (not visible in this pid namespace)"
}

OUT_DIR=$(mktemp -d)
trap 'rm -rf "$OUT_DIR"' EXIT

echo ">>> GPU $DEVICE, model $MODEL, $RUNS run(s)"
echo ">>> free before any run: $(free_mib) MiB"

BASELINE=$(compute_pids || true)
if [ -n "$BASELINE" ]; then
    echo ">>> WARNING: the card is not idle before we start. Holding it:"
    for pid in $BASELINE; do describe "$pid"; done
fi

for i in $(seq 1 "$RUNS"); do
    echo
    echo ">>> run $i/$RUNS"
    poetry run python -m method._generate_worker \
        --model "$MODEL" \
        --input "$PROMPTS" \
        --output "$OUT_DIR/answers_$i.jsonl" \
        --max_tokens 64 \
        > "$OUT_DIR/run_$i.log" 2>&1

    # The worker has exited. Everything it started must go with it, and the
    # point of the check is *when* -- so poll rather than assert immediately.
    waited=0
    while :; do
        LEFTOVER=$(compute_pids || true)
        # Anything that was already there before we started is not ours to judge.
        for pid in $BASELINE; do
            LEFTOVER=$(echo "$LEFTOVER" | grep -v "^${pid}$" || true)
        done
        [ -z "$LEFTOVER" ] && break
        if [ "$waited" -ge "$SETTLE_TIMEOUT" ]; then
            echo ">>> FAIL: a compute process outlived the worker that made it:"
            for pid in $LEFTOVER; do describe "$pid"; done
            echo ">>> free now: $(free_mib) MiB. Worker log: $OUT_DIR/run_$i.log"
            exit 1
        fi
        sleep 2
        waited=$((waited + 2))
    done
    echo ">>> card released after ${waited}s; free: $(free_mib) MiB"
done

echo
echo ">>> PASS: $RUNS handover(s), no compute process outlived its worker."
