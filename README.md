# msc_thesis

This repository vendors the `method/persona_vectors` code from [safety-research/persona_vectors](https://github.com/safety-research/persona_vectors) at commit [b8e0f044fe2410a6fad579f38324f03f13b4e917](https://github.com/safety-research/persona_vectors/commit/b8e0f044fe2410a6fad579f38324f03f13b4e917).

The vendored code is kept in-tree so it can be used directly by this project without requiring the upstream repository at install or runtime.

## Setup

```bash
poetry install
```

Real (non-mock) runs need credentials in a `.env` file at the repo root:

```
HF_TOKEN=...
OPENAI_API_KEY=...   # not needed with --local, which uses a stubbed judge
```

## Running experiments

Every trajectory is a named `TrajectoryConfig` in [`method/experiments.py`](method/experiments.py), keyed in
`experiments.REGISTRY` and run through one entry point:

```bash
poetry run python -m method.run_trajectory --config <REGISTRY_KEY> [--backend real|mock] [--seed N]
```

`--backend` defaults to `real` (`mock` for `SMOKE_MOCK`); `mock` runs the full pipeline with no model at all, for
testing orchestration. Every stage is skipped if its artifact already exists in `store/`, so an interrupted run
resumes by being re-invoked, and trajectories sharing a prefix reuse each other's adapters automatically.

List every valid `--config` value:

```bash
poetry run python -c "from method import experiments; print('\n'.join(sorted(experiments.REGISTRY)))"
```

### Smoke tests

Run these first to validate the pipeline before spending GPU time:

```bash
# No model loaded at all -- fake artifacts with real shapes, real hashing/resume logic.
poetry run python -m method.run_trajectory --config SMOKE_MOCK

# Real vLLM/unsloth/merges on the local proxy model (Qwen2.5-0.5B), stubbed judge.
poetry run python -m method.run_trajectory --config SMOKE_TINY --backend real
```

### Experiment 1 -- two-step misalign/re-align (paper scale)

```bash
poetry run python -m method.run_trajectory --config EXP1
```

### Experiments 2-4 -- families of trajectories

Experiments 2-4 are *families*: each `build_exp{2,3,4}_configs()` in `experiments.py` expands into many
`TrajectoryConfig`s (one per seed x trait x condition), all registered under keys like
`EXP2_EVIL_SEED0`, `EXP3_SAME_EVIL_HALLUCINATION_MISALIGNED_1_EVIL_SEED2`, `EXP4_DIFF3_EVIL_EVIL_SEED4`. Append
`_LOCAL` to any key (e.g. `EXP2_EVIL_LOCAL_SEED0`) to use the small local proxy model and capped example counts
instead of paper scale.

Run one trajectory:

```bash
poetry run python -m method.run_trajectory --config EXP2_EVIL_SEED0
```

Or run every trajectory in a family, e.g. all of experiment 2:

```bash
for key in $(poetry run python -c "
from method import experiments as E
print('\n'.join(k for k in sorted(E.REGISTRY) if k.startswith('EXP2_') and '_LOCAL' not in k))
"); do
    poetry run python -m method.run_trajectory --config "$key"
done
```

Swap the `startswith` prefix for `EXP3_` / `EXP4_` (or add `_LOCAL` to the filter) to run the other families or
their local-proxy variants.

### Base-model DeltaP probes (needed for the RQ1 scatter)

$\Delta P_0$ (DeltaP frozen at the base model) is not produced by `run_trajectory` -- it depends only on the base
model, seed and dataset, so it's measured once per seed and shared by every experiment:

```bash
poetry run python -m method.probe_base --seeds 0 1 2 3 4
poetry run python -m method.probe_base --local --backend mock   # smoke-test variant
```

### Generating plots

Once trajectories (and, for RQ1, base probes) are on disk:

```bash
poetry run python -m method.visualization.make_plots --experiment all
poetry run python -m method.visualization.make_plots --experiment exp2   # or exp3 / exp4
poetry run python -m method.visualization.make_plots --local             # local-proxy runs
poetry run python -m method.visualization.make_plots --mock              # SMOKE_MOCK runs
```

Figures are written to `plots/real/` (or `--out-dir`).

## Centralized storage across rental GPUs

Experiments run on ephemeral rental GPU boxes, so `store/` (adapters, measurements, training samples) and
`trajectories/` (per-run results) are synced to a shared remote via [`method/sync.py`](method/sync.py) rather than
kept only on whichever box produced them.

### Setup, once per box

**1. Point [rclone](https://rclone.org) at the shared storage** (Drive, S3, R2, B2, SFTP -- anything rclone
speaks):

```bash
rclone config          # create a remote, e.g. name "msc-thesis", type "drive"
rclone listremotes     # -> msc-thesis:
```

Credentials land in `~/.config/rclone/rclone.conf`; copying that one file onto a new rental box is faster than
redoing the OAuth flow there. For a Drive remote, prefer `scope = drive.file` (rclone can only touch files it
created) over the default `scope = drive` -- the config file holds a refresh token, and a rental box is not a
machine you control.

**2. Set `MSC_STORE_REMOTE`** in `.env` (or export it in the shell):

```
MSC_STORE_REMOTE=msc-thesis:msc-thesis
```

The part before the `:` is the rclone remote name from `rclone listremotes`; the part after is a folder inside it,
created on the first push. **Always name a folder** -- `msc-thesis:` on its own is the Drive *root*, and the sync
would scatter `store/` and `trajectories/` in among everything else already there.

A bare filesystem path is also valid and needs no rclone at all -- use it for a mounted network drive:

```
MSC_STORE_REMOTE=/mnt/shared/msc
```

Unset, every command below and every `run_trajectory`/`probe_base` run is local-only -- nothing changes.

**3. Verify before spending GPU time:**

```bash
rclone about msc-thesis:                          # credentials work?
poetry run python -m method.sync push --verbose   # creates the remote folder, uploads whatever is local
poetry run python -m method.sync pull --verbose   # on a second box: pulls what the first one pushed
```

### Day-to-day

`run_trajectory` and `probe_base` sync automatically when `MSC_STORE_REMOTE` is set: they pull any reusable
adapters/measurements before starting (so a trajectory sharing a prefix with one trained on another box is a cache
hit, not a retrain) and push new adapters/measurements/run output as they're produced. A fresh box therefore needs
only the two setup steps above -- the first `run_trajectory` pulls what it can reuse on its own.

To sync by hand:

```bash
poetry run python -m method.sync push          # upload local store + trajectories to the remote
poetry run python -m method.sync pull          # download the reusable store prefix (adapters/measurements/samples)
poetry run python -m method.sync pull-plots    # download just trajectories/ (run dirs + base probes) -- e.g. for a
                                               # machine that only makes plots and never touches a GPU
```

### What crosses the wire

Every remote read/write moves one whole artifact at a time -- a tarred adapter or measurement dir, or a
training-sample file -- never a partial one, so a remote listing can never show a half-uploaded artifact. Adapters
and training samples are immutable, so they are skipped once uploaded; measurement bundles and run dirs are
re-uploaded because they grow. `store-mock/` never syncs at all: its ids collide with the real store's by design,
so pushing it would poison real boxes.

## Renting a GPU box (vast.ai)

The image in [`Dockerfile`](Dockerfile) already carries the whole dependency set as a pre-built venv at
`/opt/app/.venv`, pre-activated on `PATH`. A rental box therefore never runs `poetry install`; it needs only the
code and the two credential files a public image must not contain. Every command in this README works verbatim on
the box -- `poetry run` reuses the baked venv instead of building a second one in-project.

**Don't rent a volume.** `store/` and `trajectories/` already outlive a box through
[`method/sync.py`](method/sync.py), so a volume would only cache the two big *inputs* (the base model and the
venv), and a vast.ai volume is tied to the single machine it was created on -- the exact constraint the sync layer
exists to avoid. Buy disk instead.

### 1. Build and push the image

The image's only inputs are the Dockerfile and the two manifests, so the tag is a digest of them. Unchanged inputs
rebuild to the same tag, a changed dependency is forced to produce a new one, and a finished experiment can name
the image that produced it:

```bash
TAG=$(cat Dockerfile pyproject.toml poetry.lock | sha256sum | cut -c1-12)
docker build -t pabloo22/msc-thesis:$TAG .
docker push pabloo22/msc-thesis:$TAG
```

Rent that tag rather than `:latest`: a mutable tag cannot be cited in a write-up, and a host that already cached
the name may not re-pull a newer push of it.

### 2. Rent with enough disk

**150 GB.** A paper-scale trajectory on Qwen2.5-7B needs roughly ~17 GB for the image, ~15 GB for the base model
under `HF_HOME`, and **~30 GB of merged checkpoints** -- `materialize()` holds two full 7B checkpoints at once
while walking the adapter chain, evicting step `k-1` only after step `k` is written. On top of that,
`pull_before_run` fetches *every* adapter and measurement on the remote, not just the chain this trajectory needs,
so that share grows toward ~35 GB as the sweep progresses. 100 GB is fine for the first few trajectories and gets
tight later; disk is a rounding error next to the GPU.

Paste this as the instance's **on-start script**. It stays deliberately small -- anything more belongs in
[`scripts/box_setup.sh`](scripts/box_setup.sh), which can be fixed with a `git push` instead of by re-editing the
template on every box:

```bash
#!/bin/bash
mkdir -p /workspace /root/.config/rclone
cd /workspace && git clone https://github.com/Pabloo22/msc_thesis.git \
  || git -C /workspace/msc_thesis pull --ff-only
```

### 3. Copy the two credential files across

Neither belongs in the image, which is public on Docker Hub:

```bash
scp -P <port> .env            root@<host>:/workspace/msc_thesis/.env
scp -P <port> ~/.config/rclone/rclone.conf root@<host>:/root/.config/rclone/rclone.conf
```

Don't run `rclone config` on the box -- the OAuth flow wants a browser it doesn't have, and the `drive.file` scope
note above is the reason to move an existing config rather than mint fresh credentials there.

### 4. Check the box before spending GPU time

```bash
bash scripts/box_setup.sh
```

It installs nothing. It reports every problem it finds at once and exits non-zero on any that would break a run: no
visible GPU, a `python` that isn't the baked venv, a `torch.cuda.is_available()` that is `False` (a driver too old
for the image's cu124 wheels fails here rather than an hour into a trajectory), a missing `HF_TOKEN`, an
unreachable `MSC_STORE_REMOTE`, or too little free disk.

The check worth understanding is **dependency drift**: the script diffs the clone's `poetry.lock` against the
`/opt/app/poetry.lock` baked into the image. Checking out a commit whose dependencies the image predates fails
loudly and tells you to rebuild, rather than silently producing results that match no image you can name.

Then smoke-test before the real thing, exactly as in [Smoke tests](#smoke-tests) above:

```bash
python -m method.run_trajectory --config SMOKE_TINY --backend real
```