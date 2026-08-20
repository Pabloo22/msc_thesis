#!/usr/bin/env bash
#
# Readiness check for a freshly rented GPU box.
#
# This script installs nothing: the image already carries the dependency venv at
# /opt/app/.venv. What differs between a working box and a broken one is the
# handful of things an image cannot contain -- a visible GPU, a repo whose lock
# file still matches the one the image was built from, and the two secret files
# that are deliberately kept out of a public image. This reports on exactly
# those, all of them, before any GPU time is spent.
#
# Idempotent and non-destructive, so it is meant to be re-run after scp-ing the
# secrets across. Exits non-zero if anything would break a real run.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The image copies these in at build time; comparing against them is how a box
# detects that it cloned a commit whose dependencies the image predates.
BAKED_LOCK="/opt/app/poetry.lock"
RCLONE_CONF="${HOME}/.config/rclone/rclone.conf"
# Below this, a paper-scale run risks dying mid-merge: two 7B checkpoints are
# live at once while materialize() walks the adapter chain.
MIN_DISK_GB=100

failures=0
ok()   { printf '  \033[32m ok \033[0m %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; failures=$((failures + 1)); }
warn() { printf '  \033[33mwarn\033[0m %s\n' "$1"; }

# Presence only -- never the value. These are credentials, and this script's
# output tends to end up pasted into terminals and issue threads.
env_has_key() {
    [[ -f "${REPO_ROOT}/.env" ]] && grep -qE "^[[:space:]]*${1}=." "${REPO_ROOT}/.env"
}

echo
echo "== GPU =="
if gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null) \
   && [[ -n "${gpu}" ]]; then
    ok "${gpu}"
else
    bad "nvidia-smi found no GPU (is the instance still provisioning?)"
fi

echo
echo "== Python environment =="
if [[ ! -d /opt/app/.venv ]]; then
    warn "no /opt/app/.venv; not running the project image, so the rest is advisory"
elif [[ "$(command -v python)" == "/opt/app/.venv/bin/python" ]]; then
    ok "python -> /opt/app/.venv/bin/python"
else
    bad "python is $(command -v python || echo 'not found'), expected /opt/app/.venv/bin/python"
fi

# Importing torch is slow but it is the check that matters: a driver too old for
# the image's cu124 wheels fails here rather than an hour into a trajectory.
if torch_info=$(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())' 2>&1); then
    read -r torch_version cuda_available <<<"${torch_info}"
    if [[ "${cuda_available}" == "True" ]]; then
        ok "torch ${torch_version}, CUDA available"
    else
        bad "torch ${torch_version} imported but torch.cuda.is_available() is False"
    fi
else
    bad "torch failed to import: ${torch_info}"
fi

echo
echo "== Image / repo agreement =="
if [[ ! -f "${BAKED_LOCK}" ]]; then
    warn "no ${BAKED_LOCK}; not running the project image, so deps are unverified"
elif cmp -s "${BAKED_LOCK}" "${REPO_ROOT}/poetry.lock"; then
    ok "poetry.lock matches the image"
else
    # Installing the delta here would work but would silently make this box's
    # results non-reproducible against the tagged image, which is the whole
    # point of tagging it.
    bad "poetry.lock differs from the image's -- rebuild and push the image, then rent again"
fi

echo
echo "== Dataset =="
# Extracted by the on-start script, not by git: dataset/ is gitignored and rides
# along as a zip inside the vendored tree.
DATASET_ZIP="${REPO_ROOT}/method/persona_vectors/dataset.zip"
UNZIP_CMD="unzip -nq ${DATASET_ZIP} -d ${REPO_ROOT}"
if [[ ! -d "${REPO_ROOT}/dataset" ]]; then
    bad "no dataset/ -- run: ${UNZIP_CMD}"
elif [[ ! -f "${DATASET_ZIP}" ]]; then
    warn "dataset/ present but dataset.zip is gone; cannot verify it is complete"
else
    # A half-finished extraction (disk filled mid-unzip) leaves a dataset/ that
    # looks fine until a trajectory asks for the one file that never landed.
    # Comparing counts only reads the zip's central directory, so it is cheap.
    want=$(unzip -Z1 "${DATASET_ZIP}" '*.jsonl' 2>/dev/null | wc -l)
    have=$(find "${REPO_ROOT}/dataset" -name '*.jsonl' | wc -l)
    if (( want > 0 && have >= want )); then
        ok "dataset/ complete (${have} jsonl files)"
    else
        bad "dataset/ has ${have} jsonl files, dataset.zip has ${want} -- run: ${UNZIP_CMD}"
    fi
fi

echo
echo "== Credentials =="
env_has_key HF_TOKEN && ok ".env has HF_TOKEN" \
    || bad ".env is missing HF_TOKEN (scp it from your laptop)"
env_has_key OPENAI_API_KEY && ok ".env has OPENAI_API_KEY" \
    || warn ".env has no OPENAI_API_KEY (fine only for --local runs, which stub the judge)"

echo
echo "== Remote store =="
if ! env_has_key MSC_STORE_REMOTE; then
    warn "MSC_STORE_REMOTE unset: this box runs local-only and its artifacts die with it"
else
    remote=$(grep -E '^[[:space:]]*MSC_STORE_REMOTE=' "${REPO_ROOT}/.env" | tail -1 | cut -d= -f2- | tr -d '"'"'"' ')
    # Mirrors make_transport(): a ':' before the first '/' means rclone, a bare
    # path means a mounted filesystem and needs no config at all.
    if [[ "${remote%%/*}" == *:* && "${remote}" != /* ]]; then
        if [[ ! -f "${RCLONE_CONF}" ]]; then
            bad "rclone target '${remote}' but no ${RCLONE_CONF} (scp it; do not redo OAuth headless)"
        # Probes the configured bucket, not the bare remote: a bucket-scoped R2
        # token (least-privilege, Cloudflare's recommended type) has no
        # account-level list-buckets permission, so "remote:" alone 403s even
        # when the bucket itself is fully readable.
        elif rclone lsd "${remote}" >/dev/null 2>&1; then
            ok "rclone remote '${remote%%:*}' reachable"
        else
            bad "rclone cannot reach '${remote}' -- check ${RCLONE_CONF}"
        fi
    else
        [[ -d "${remote}" ]] && ok "mounted remote ${remote}" \
            || bad "mounted remote ${remote} does not exist"
    fi
fi

echo
echo "== Notifications =="
# Warnings, never failures: a box with no notifier still runs perfectly well.
# What it cannot do is tell you it died, which on a rental is the difference
# between losing minutes and losing a night's billing.
if env_has_key MSC_RESEND_API_KEY && env_has_key MSC_NOTIFY_EMAIL; then
    ok ".env has MSC_RESEND_API_KEY and MSC_NOTIFY_EMAIL"
else
    warn "no email notifications (set MSC_RESEND_API_KEY + MSC_NOTIFY_EMAIL in .env)"
fi
env_has_key MSC_HEARTBEAT_URL && ok ".env has MSC_HEARTBEAT_URL" \
    || warn "no MSC_HEARTBEAT_URL: a box that is killed outright or loses its network cannot email you"
env_has_key MSC_GPU_HOURLY_USD && ok ".env has MSC_GPU_HOURLY_USD" \
    || warn "no MSC_GPU_HOURLY_USD: reports will show times but omit costs"

echo
echo "== Disk =="
# Set by the image; absent when this runs anywhere else, and there is no sensible
# place to guess at.
[[ -n "${HF_HOME:-}" ]] && mkdir -p "${HF_HOME}"
# Measured at the repo, not /workspace: store/ and the HF cache share whatever
# filesystem the clone landed on, and that is the one that fills up.
avail_gb=$(df -BG --output=avail "${REPO_ROOT}" | tail -1 | tr -dc '0-9')
if (( avail_gb >= MIN_DISK_GB )); then
    ok "${avail_gb}G free"
else
    warn "only ${avail_gb}G free; a 7B run peaks near ${MIN_DISK_GB}G"
fi

echo
if (( failures > 0 )); then
    echo "${failures} blocking problem(s); fix before starting a run."
    exit 1
fi
echo "Box ready. Smoke-test first:"
echo "  cd ${REPO_ROOT} && python -m method.run_trajectory --config SMOKE_TINY --backend real"
