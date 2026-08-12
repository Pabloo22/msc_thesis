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

### 2. Running a Family

Experiments 2-4 expand into hundreds of `TrajectoryConfig`s, so they are launched by
prefix rather than one key at a time. `scripts/run_family.sh` runs every registry key
starting with the prefix, sequentially, and mails a summary however it ends:
```bash
nohup bash scripts/run_family.sh EXP2_VALIDATION > exp2_validation.log 2>&1 &
```

**The prefix is the unit of work.** Keys are `<config name>_SEED<n>` upper-cased, so any
leading segment selects a slice: `EXP2` is all three exp2 families, `EXP2_DECAY` one of
them, `EXP2_DECAY_BRANCH_A` just trunk A's fan. This is how a design with a single seed
gets split across GPUs.

One caveat on sub-family prefixes: the cost reporting keys off a config's *group*
(`exp2_validation`, `exp2_decay`, `exp2_reseed`, `exp3`, `exp4`) and matches it exactly,
so a prefix that is not one of those still runs correctly but reports "nothing logged
yet" in the end-of-family email and gives no projected remaining cost. Ask for the
numbers directly instead:
```bash
poetry run python -m method.report --family exp2_decay
```

**Splitting across GPUs.** Disjoint prefixes (or `--seeds`, where a design has several)
never train the same adapter, because both the step sequence and the seed are part of
`weights_key`; store writes are atomic, so the base-model measurements they share are
safe to race.
```bash
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_family.sh EXP3 --seeds 0 1 2 > exp3_a.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_family.sh EXP3 --seeds 3 4   > exp3_b.log 2>&1 &
```

`--trunks` is the same idea for exp2, which has one seed and so cannot be split by it:
it keeps the `EXP2_DECAY` group (and therefore the cost reporting) while selecting one
trunk's chain — trunk *and* its branches, trunk first — so one GPU per trunk is a single
command. The three trunks share no step prefix beyond `M_0`, so they neither collide nor
duplicate training.
```bash
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_family.sh EXP2_DECAY --trunks a > exp2_a.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_family.sh EXP2_DECAY --trunks b > exp2_b.log 2>&1 &
```

**Seeing the whole figure before spending a GPU on it.** `LOCAL` selects the small-proxy
variants; `MOCK` fabricates every artifact instead of training, so a family finishes in
minutes and lands in `trajectories-mock/`, separate from the real runs. Use it to check
that a figure has every panel the design calls for: `make_plots` only draws conditions
that exist on disk, so a partly-run family silently produces a partly-populated chart.
```bash
bash scripts/run_family.sh EXP3 LOCAL MOCK --seeds 0
poetry run python -m method.visualization.make_plots --experiment exp3 --mock --local
```

### 3. Experiment 1 (two-step misalign/re-align)
```bash
nohup poetry run python -m method.run_trajectory --config EXP1 > exp1.log 2>&1 &
```

### 4. Experiment 2 (RQ1: does $\Delta P_0$ go stale as the model drifts?)

This is how to run it. It is **three registry families**, deliberately separate
because they gate each other:

| Phase | Prefix | Fine-tunings | Gate before continuing |
|---|---|---|---|
| 1 | `EXP2_VALIDATION` — the $t = 0$ fan over all 24 datasets | 24 | reproduces Fig. 8 of the persona-vectors paper — else stop |
| 2 | *(no runs)* choose the probe set, check the noise ceiling | 0 | probes span the $\Delta P_0$ range; $R^2_{max}$ not dominated by noise |
| 3 | `EXP2_DECAY_TRUNK_A` — trunk A's 6 drivers | 6 | — |
| 4 | `EXP2_RESEED` — trunk A again under seed 1, no branches | 6 | $\rho, r$ track trunk A — else 3-5 trunk seeds are needed |
| 5 | `EXP2_DECAY_BRANCH_A` — trunk A's fan at $t = 1\ldots6$ | 48 | first real decay curve; hysteresis check at $M_2$ |
| 6 | `EXP2_DECAY_TRUNK_B/_C` then `EXP2_DECAY_BRANCH_B/_C` | 12 + 96 | — |
| | **Total** | **192** | |

Phases 1-2 cost 24 runs and can invalidate the design before the remaining 168 are
committed, which is why the families are run in this order rather than as one sweep.
If the budget has to shrink, §10 of the spec says what to cut and in what order.

Note the 344 exp2 registry keys are *not* 344 fine-tunings. `trait` is not part of
`weights_key`, so the evil and sycophantic configs of a trajectory share one chain of
adapters: the second trait re-measures, it does not retrain. Branches are cheap for the
same reason — a branch is `drivers[:t] + (probe,)`, so the trunk's prefix is a store hit
and only the final step trains.

**Dry-run the whole thing first.** Every figure, every panel, no GPU:
```bash
bash scripts/run_family.sh EXP2_VALIDATION LOCAL MOCK
bash scripts/run_family.sh EXP2_DECAY      LOCAL MOCK
bash scripts/run_family.sh EXP2_RESEED     LOCAL MOCK
poetry run python -m method.visualization.make_plots --experiment exp2_decay --mock --local
```

**Phase 1 — the validation fan.** This is the gate the whole project hangs on.
```bash
nohup bash scripts/run_family.sh EXP2_VALIDATION > exp2_validation.log 2>&1 &
poetry run python -m method.visualization.make_plots --experiment exp2_validation
```
Read `plots/real/exp2/exp2_validation.png`: 24 points of $\Delta P_0$ against
$\Delta b_1$, one panel per trait. If the correlation is not there, nothing downstream is
interpretable.

**Phase 2 — pick the probe set (no GPU).** `EXP2_PROBES` in
[`method/experiments.py`](method/experiments.py) is currently a *stratified guess*
(2 Normal, 3 `I`, 3 `II`) standing in until phase 1's numbers exist. Replace it with 8
datasets spanning the observed $\Delta P_0$ range. `check_exp2_feasibility()` runs at
import, so a set that collides with a trunk's drivers — or that takes a third Normal and
makes trunk C infeasible — fails loudly here rather than 48 runs later. Do this **before
phase 5**: the probe set defines every branch config.

Then check that the scatter is not mostly noise. `sigma_seed(b)` is read off any
seed-swept family, and exp3 is one:
```bash
poetry run python -m method.seed_noise --group exp3      # sigma_seed(b) and a preliminary ceiling
```
Combine it with phase 1's observed spread of $\Delta b$ via `method.noise.r2_max`.
$R^2_{max} \gtrsim 0.9$ — proceed. $\lesssim 0.6$ — R² would be measuring seed variation
rather than $\Delta P$, and seed replication has to be reinstated before the fans are
paid for.

**Phase 3 — trunk A.** Run a trunk *before* its branches. The trunk carries all the
per-checkpoint measurement ($z_t$, $b_t$, and $\Delta P_t$ for all 8 probes); branches
carry `measure=ENDPOINT_BEHAVIOR` and contribute only $b_{t+1}$, so without the trunk the
decay figures have no rows at all. `run_family.sh` puts every trunk ahead of the branches
whatever the selection, so this ordering holds inside a `--trunks` run too; the trunk is
split out into its own phase here because phase 5 is a decision point, not because the
order needs forcing.
```bash
nohup bash scripts/run_family.sh EXP2_DECAY_TRUNK_A > exp2_trunk_a.log 2>&1 &
```

**Phase 4 — the reseed replicate.** Trunk A under seed 1, six fine-tunings and no fan,
because only $\Delta b$ needs training.
```bash
nohup bash scripts/run_family.sh EXP2_RESEED > exp2_reseed.log 2>&1 &
```
Check it on `exp2_drift_z.png` and `exp2_drift_delta_p.png` (A′ is the dashed overlay on
trunk A's column): if A and A′ diverge sharply in $\rho$ or $r$, `n = 2` is not enough and
3-5 trunk seeds become necessary.

**Phase 5 — trunk A's fan.** 48 branches, and the first real decay curve. This is also
where the hysteresis assumption gets checked: look at $\rho$ and $r$ at $M_2$ on trunk A.
If the Normal driver fully reverses the drift the `II` driver caused, trunk A never
accumulates a dose and the A > B > C ladder collapses — a real finding, but one that
changes the design before B and C are paid for.
```bash
nohup bash scripts/run_family.sh EXP2_DECAY_BRANCH_A > exp2_branch_a.log 2>&1 &
```

**Phase 6 — trunks B and C.** Independent chains, so they can go on separate GPUs. One
`--trunks` run is the whole chain, trunk first and then its fan.
```bash
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_family.sh EXP2_DECAY --trunks b > exp2_b.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_family.sh EXP2_DECAY --trunks c > exp2_c.log 2>&1 &
```

**Before plotting — backfill `SE(b)`.** The headline figure's noise ceiling needs the
analytic standard error of every behaviour measurement. Checkpoints measured before that
was recorded carry no `SE`, and show up as NaN rather than as zero. Run this wherever the
store lives (the per-generation scores it reads exist only there), then sync
`trajectories/`:
```bash
poetry run python -m method.backfill_se --dry-run
poetry run python -m method.backfill_se
```

**Plotting.** Asking for any one exp2 family collects all three — the decay grid's
$t = 0$ column comes from the validation fan, and the reseed family is only meaningful
overlaid on the trunk it replicates:
```bash
poetry run python -m method.visualization.make_plots --experiment exp2_decay
```
This writes section 9's figures into `plots/real/exp2/`. Every one of them panels both
traits, so nothing is emitted per trait: `exp2_validation` (plot 1, a trait per panel),
`exp2_decay_grid` (2, a trait-and-trunk per row, a checkpoint per column), `exp2_headline`
(3, two rows per trait), `exp2_mechanism` (4, a trait per row, a predictor per column),
`exp2_phase_contrast` (4b, a trait per row) and `exp2_drift_delta_p` / `exp2_drift_z`
(both plot 5, a trait per row). What they do *not* share across the traits is the scale,
except where the quantity is unitless — a persona vector and a judge are per trait.

### 5. Experiments 3 and 4
Both are seed sweeps, so `--seeds` is the axis that splits them:
```bash
nohup bash scripts/run_family.sh EXP3 > exp3.log 2>&1 &
nohup bash scripts/run_family.sh EXP4 > exp4.log 2>&1 &
```
Each family plots to a single figure: `exp3_hysteresis` puts a (measured trait,
re-alignment source) pair on each row and a dataset in each column, and `exp4_diversity`
a measured trait per row and a re-alignment source per column. In both, the two rows of
one measured trait share a y-axis — that comparison is the control the design is for —
and the two traits do not.

## Base-model DeltaP Probes
$\Delta P_0$ (DeltaP frozen at the base model) is what the exp3/exp4 scatter plots need
for datasets a trajectory never trains on first. It is measured once per seed and shared
across experiments:
```bash
poetry run python -m method.probe_base --seeds 0 1 2 3 4
poetry run python -m method.probe_base --local --backend mock   # smoke-test variant
```
exp2 does not depend on this pass: each trunk measures $\Delta P$ for all 8 probes at
its own $t = 0$, and each validation run's first step *is* a $\Delta P_0$ measurement.

## Generating Plots
Once trajectories (and base probes for exp3/exp4) are on disk, generate figures:
```bash
poetry run python -m method.visualization.make_plots --experiment all
poetry run python -m method.visualization.make_plots --experiment exp2_decay  # or exp2_validation / exp2_reseed / exp3 / exp4
poetry run python -m method.visualization.make_plots --local                  # local-proxy runs
poetry run python -m method.visualization.make_plots --mock --local           # fabricated runs
```

Each run logs how many runs it loaded and how many are still missing, e.g.
`exp3: 42 run(s) loaded, 168 not yet run, 0 stale`. A figure is only as complete as
that line says: conditions with no runs are dropped from the chart rather than drawn
empty, so check it before reading a bar chart as the finished comparison.

`--sigma-seed` overrides the fine-tune seed SD behind exp2's noise ceiling; left alone,
it is read off a seed-swept family via `method.seed_noise`, and with no such family on
disk the ceiling counts eval noise only (an upper bound, logged rather than hidden).

Plots are written to one directory per run source — `plots/real`, `plots/real-local`,
`plots/mock`, `plots/mock-local` (or `--out-dir`) — with a per-family subdirectory
(`exp2/`, `exp3/`, `exp4/`) underneath. The figures directly under `plots/` are the
*synthetic* ones drawn from fixtures by `method.visualization.demo`; they do not change
when runs finish.

<!-- 
## Auditing $z_t$ Before Reading It
$p_t$ and $\rho_t$ are defined against the base model's persona vector $v_0$, and $v_0$
is a *measurement* — extracted from sampled generations, cached in the store under the
base `weights_id`, and therefore shared across runs only while every run agrees on that
id and finds the artifacts already there. When it is re-derived, runs end up on
different anchors and their levels stop being comparable.

`method.visualization.latent_audit` checks that structurally, using the fact that at
$t = 0$ every run of a trait is measuring the *same weights*, so any disagreement in
$z_0$ is measurement rather than model:
```bash
poetry run python -m method.visualization.latent_audit --group exp3
poetry run python -m method.visualization.latent_audit --group exp3 --csv z.csv --no-plots
```
It prints the distinct anchors, every checkpoint two runs measured differently, and a
per-component ratio of that disagreement to the drift — read that ratio before reading
a component as a result. Four figures land next to the family's own, in
`plots/real/exp3/exp3_latent_*`.

See [`docs/exp3_latents.md`](docs/exp3_latents.md) for what the audit found on the
exp3 runs currently on disk.

**A zero here is not a precision claim.** Every measurement is memoized by content, so a
checkpoint reached twice reads its `latent.json` back verbatim: the audit only ever sees
disagreement where the *cache* failed. Once the store is consistent it reports 0.0 for
every component, which says the cache worked and leaves the measurement's own precision
unmeasured. That is what the next section is for. -->

## Measuring the Anchor Term in $z_t$
`method.anchor_noise` measures the same quantity on
purpose, by re-drawing the two artifacts every $z_t$ is anchored to — $v_0$'s pos/neg
generations and $M_0$'s answers to the neutral prompts — and carrying each draw along an
already-trained trunk:
```bash
poetry run python -m method.anchor_noise --replicates 3
poetry run python -m method.anchor_noise --trunk a --checkpoints 0 1 3 6 --replicates 5
poetry run python -m method.anchor_noise --backend mock --local --checkpoints 0
```
It trains nothing: the trunk's adapters must already exist, and every checkpoint is
replayed from them. Replicate 0 *is* the production bundle — the artifacts exp2 and exp3
actually used — so it costs no generation and the spread says where the published numbers
sit inside their own sampling distribution. Fresh draws are quarantined in
`anchor_replicates/` inside each checkpoint's measurement bundle, so nothing already on
disk is touched. Budget generation, not training: per fresh draw, ~500 neutral answers
(shared across traits) plus a judged pos/neg extraction per trait, then a forward pass
over each at every checkpoint.

Two spreads are reported per component, and they answer different questions:

- `sigma_level` — the error on a reported $z_t$.
- `sigma_delta` — the error on $z_t - z_0$ *within* a draw. The anchor is common-mode, so
  a bad draw shifts the whole series together and largely cancels in a difference. Quote
  this one for the decay and drift figures, which read changes rather than levels.

`against_drift` puts both beside the drift measured on the production replicate over the
same checkpoints; at a ratio $\ge 1$ the component carries no usable signal over that
span. The summary lands in `trajectories/anchor_noise/` and syncs with the trajectories
root, so a plotting box needs no store to read it.

This is a different quantity from `method.seed_noise` and the `EXP2_RESEED` trunk, which
vary the *fine-tuning* seed. Those bound measurement noise from above but cannot isolate
it — and because `weights_key` normalizes the seed away at $t = 0$, every seed in them
reads one cached anchor, so the anchor's own error contributes nothing to what they
measure. The two are complements, not substitutes.