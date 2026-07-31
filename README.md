# msc_thesis

This repository vendors the `method/persona_vectors` code from [safety-research/persona_vectors](https://github.com/safety-research/persona_vectors) at commit [`b8e0f044...`](https://github.com/safety-research/persona_vectors/commit/b8e0f044fe2410a6fad579f38324f03f13b4e917). The vendored code is kept in-tree so it can be used directly without requiring the upstream repository at install or runtime.

## Setup

Install dependencies:
```bash
poetry install
```

Create a `.env` file at the repository root:
```ini
HF_TOKEN=...
OPENAI_API_KEY=...       # optional: not needed with --local
MSC_STORE_REMOTE=...     # optional: remote storage path/rclone (e.g., msc-thesis:msc-thesis)
```

> **Running on rental GPUs?** See [`docs/cloud_setup.md`](docs/cloud_setup.md) for instructions on Docker image building, vast.ai setup, and rclone remote syncing.

## Running Experiments

Every trajectory is a named `TrajectoryConfig` in [`method/experiments.py`](method/experiments.py). List all valid configurations with:
```bash
poetry run python -c "from method import experiments; print('\\n'.join(sorted(experiments.REGISTRY)))"
```

Run an experiment using a registered config key:
```bash
poetry run python -m method.run_trajectory --config <REGISTRY_KEY> [--backend real|mock] [--seed N]
```
*(Note: Every stage is skipped if its artifact already exists in `store/`. Interrupted runs will automatically resume.)*

### 1. Smoke Tests
Run these first to validate the pipeline locally before spending GPU time:
```bash
# Fake artifacts with real shapes and hashing/resume logic (no model loaded)
poetry run python -m method.run_trajectory --config SMOKE_MOCK

# Real local proxy model (Qwen2.5-0.5B) with stubbed judge
poetry run python -m method.run_trajectory --config SMOKE_TINY --backend real
```

### 2. Full Experiments
*   **Experiment 1 (two-step misalign/re-align):**
    ```bash
    poetry run python -m method.run_trajectory --config EXP1
    ```
    or 
    ```bash
    nohup poetry run python -m method.run_trajectory --config EXP1 > exp1.log 2>&1 &
    ```

*   **Experiments 2-4 (families of trajectories):**
    These expand into many `TrajectoryConfig`s. Append `_LOCAL` to any key to use the small proxy model and capped example counts. 
    ```bash
    poetry run python -m method.run_trajectory --config EXP2_EVIL_SEED0
    ```
    *(Tip: You can use `scripts/run_family.sh EXP2` to run all trajectories in a family sequentially.)*

    To use the script with nohup, run:
    ```bash
    nohup bash scripts/run_family.sh EXP2 > exp2.log 2>&1 &
    ```

    **Splitting a family across GPUs.** `--seeds` restricts the family to a subset of
    seeds, so disjoint subsets can run concurrently on different devices. The seed is
    part of `weights_key`, so two subsets never train the same adapter; store writes are
    atomic, so the seed-independent base-model measurements they share are safe to race.
    ```bash
    CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_family.sh EXP3 --seeds 0 1 2 > exp3_a.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_family.sh EXP3 --seeds 3 4   > exp3_b.log 2>&1 &
    ```
    `LOCAL` still works alongside it (`scripts/run_family.sh EXP3 LOCAL --seeds 0`).

## Base-model DeltaP Probes
$\\Delta P_0$ (DeltaP frozen at the base model) is needed for the RQ1 scatter plots. It is measured once per seed and shared across experiments:
```bash
poetry run python -m method.probe_base --seeds 0 1 2 3 4
poetry run python -m method.probe_base --local --backend mock   # smoke-test variant
```

## Generating Plots
Once trajectories (and base probes for RQ1) are on disk, generate figures:
```bash
poetry run python -m method.visualization.make_plots --experiment all
poetry run python -m method.visualization.make_plots --experiment exp2   # or exp3 / exp4
poetry run python -m method.visualization.make_plots --local             # local-proxy runs
```
Plots are written to `plots/real/` (or the path specified by `--out-dir`).