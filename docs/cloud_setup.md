# Cloud Setup & Infrastructure

Experiments run on ephemeral rental GPU boxes, so `store/` (adapters, measurements, training samples) and `trajectories/` (per-run results) are synced to a shared remote via `method/sync.py` rather than kept only on whichever box produced them.

## Centralized storage across rental GPUs

### Setup, once per box

**1. Point [rclone](https://rclone.org) at the shared storage** (Drive, S3, R2, B2, SFTP -- anything rclone speaks):

```bash
rclone config          # create a remote, e.g. name "msc-thesis", type "drive"
rclone listremotes     # -> msc-thesis:
```

Credentials land in `~/.config/rclone/rclone.conf`; copying that one file onto a new rental box is faster than redoing the OAuth flow there. For a Drive remote, prefer `scope = drive.file` (rclone can only touch files it created) over the default `scope = drive` -- the config file holds a refresh token, and a rental box is not a machine you control.

**2. Set `MSC_STORE_REMOTE`** in `.env` (or export it in the shell):

```ini
MSC_STORE_REMOTE=msc-thesis:msc-thesis
```

The part before the `:` is the rclone remote name from `rclone listremotes`; the part after is a folder inside it, created on the first push. **Always name a folder** -- `msc-thesis:` on its own is the Drive *root*, and the sync would scatter `store/` and `trajectories/` in among everything else already there.

A bare filesystem path is also valid and needs no rclone at all -- use it for a mounted network drive:
```ini
MSC_STORE_REMOTE=/mnt/shared/msc
```

Unset, every `run_trajectory`/`probe_base` run is local-only -- nothing changes.

**3. Verify before spending GPU time:**

```bash
rclone about msc-thesis:                          # credentials work?
poetry run python -m method.sync push --verbose   # creates the remote folder, uploads whatever is local. Idea: add something to store/ or traj

poetry run python -m method.sync pull --verbose   # downloads the reusable store prefix (adapters/measurements/samples) 
poetry run python -m method.sync pull-plots --verbose   # downloads just trajectories/ (run dirs + base probes) -- e.g. for a machine that only makes plots and never touches a GPU
```

### Day-to-day

`run_trajectory` and `probe_base` sync automatically when `MSC_STORE_REMOTE` is set: they pull any reusable adapters/measurements before starting (so a trajectory sharing a prefix with one trained on another box is a cache hit, not a retrain) and push new adapters/measurements/run output as they're produced. A fresh box therefore needs only the two setup steps above.

To sync by hand:

```bash
poetry run python -m method.sync push          # upload local store + trajectories to the remote
poetry run python -m method.sync pull          # download the reusable store prefix (adapters/measurements/samples)
poetry run python -m method.sync pull-plots    # download just trajectories/ (run dirs + base probes) -- e.g. for a
                                               # machine that only makes plots and never touches a GPU
```

> **What crosses the wire:** Every remote read/write moves one whole artifact at a time (tarred adapter, measurement dir, etc.) -- never a partial one. Adapters and training samples are immutable and skipped once uploaded; measurement bundles and run dirs grow, so they are skipped only while unchanged since this box last pushed them, and re-uploaded whole once they change. `store-mock/` never syncs at all to avoid poisoning real boxes.

> **Repeated pushes are incremental.** `push` is safe to re-run and costs roughly what changed since the last one: an artifact already uploaded is skipped, not re-sent. That record is local to the box (`store/.sync-state/`), so if you delete objects from the remote by hand, run `push --force` to ignore it and re-upload everything mutable.

---

## Knowing when a run dies (and what it cost)

A rental bills whether or not the process on it is still alive, so the expensive failure is not the crash — it's the hours between the crash and noticing. Two independent channels cover that, because neither covers it alone.

### 1. Email, for the detail

`run_trajectory` and `probe_base` mail you when they finish or fail. A failure mail leads with the traceback and says the box is still rented; a success mail carries the per-stage timing table, the per-checkpoint breakdown, and — once a family is under way — how much of it is left and what the whole thing is projected to cost.

Sending goes through [Resend](https://resend.com)'s HTTP API using only the standard library. That's deliberate: the image tag a box is booked against is a digest of `poetry.lock`, so adding a package just to send mail would force a ~17 GB rebuild and push before the next rental.

Add to `.env`:

```ini
MSC_RESEND_API_KEY=re_...        # a send-only key
MSC_NOTIFY_EMAIL=you@example.com
MSC_GPU_HOURLY_USD=1.80          # what this box costs; omit and reports show times but no money
MSC_NOTIFY_TAG=vast-4090         # optional; prefixes the subject so two boxes are distinguishable
MSC_NOTIFY_MIN_INTERVAL_MINUTES=60  # optional; the default. 0 mails every report
```

On the free tier you can send from `onboarding@resend.dev` to the address you registered with, so no domain verification is needed. Set `MSC_NOTIFY_FROM` once you have a verified domain.

> **Mail is rate-limited, because the free tier allows 100 sends a day.** A family is dozens of trajectories, each its own process, each mailing on the way out — enough to exhaust the quota in a single busy day, after which *every* later mail is rejected, including the failure one the whole channel exists for. So at most one mail goes out per hour per bucket, where the buckets are `<family>:ok`, `<family>:failed` and `probe_base:{ok,failed}`. Successes and failures are counted apart, so a family that is mostly completing can never hide the first trajectory that dies. Whatever was dropped is counted, and the next mail that does go out says how many reports it stands for.
>
> The counter lives in `.notify-state.json` at the repo root (gitignored) rather than in memory, since each trajectory is a separate process. Two boxes each keep their own file and so each get their own allowance — `MSC_NOTIFY_TAG` is what tells them apart in the inbox. The end-of-family summary from `run_family.sh` is never throttled: there is only one per run, and it is the backstop for a family killed outright.

> **On credentials and rental boxes:** the host operator can read every file on the instance, so use a **send-only API key**, revocable in one click. Do *not* use a personal Gmail app password — those grant IMAP as well as SMTP, so a leak means your whole mailbox, not just the ability to send. Unset either variable and notifications are silently off; the run is unaffected.

### 2. Heartbeat, for the failures that can't email

An email is sent *by the box, over the box's network*. When the network is what died — or the OOM killer sends SIGKILL, or the instance is preempted — no `finally` block can get a message out. A watchdog inverts this: the run pings a URL every 60 seconds, and the *absence* of pings is the alarm, so nothing on the box has to be alive to raise it.

Create a check at [healthchecks.io](https://healthchecks.io) (free), set **period 1 minute** and **grace 5 minutes**, and put its ping URL in `.env`:

```ini
MSC_HEARTBEAT_URL=https://hc-ping.com/<uuid>
```

The period is a constant this code chooses, not a property of the workload, so the grace never needs re-tuning between a 7B family and a `--local` proxy run. The URL is an opaque UUID that can only ping your own check, so it is not a meaningful secret.

A clean failure also pings `/fail` immediately, so when the network *is* up you hear about it at once rather than after the grace period.

### Reading the numbers without email

The same tables print locally, which is how you cost an experiment before committing to it:

```bash
poetry run python -m method.report                          # every family on disk, with totals
poetry run python -m method.report --run trajectories/EXP3_..._seed0   # one trajectory, stage by stage
poetry run python -m method.report --family EXP3            # one family
```

Timings live in `timings.jsonl` inside each run directory and `runlog.jsonl` beside them, appended as work completes — so a box that was killed still leaves behind everything it had learned about its own speed, and a resumed run's estimates account for what it skipped rather than concluding the remaining work is free.

---

## Renting a GPU box (vast.ai)

The image in `Dockerfile` already carries the whole dependency set as a pre-built venv at `/opt/app/.venv`, pre-activated on `PATH`. A rental box therefore never runs `poetry install`; it needs only the code and the two credential files a public image must not contain. Every command works verbatim on the box -- `poetry run` reuses the baked venv.

**Don't rent a volume.** `store/` and `trajectories/` already outlive a box through `method/sync.py`, so a volume would only cache the two big *inputs* (the base model and the venv). A vast.ai volume is tied to the single machine it was created on -- the exact constraint the sync layer exists to avoid. Buy disk instead.

### 1. Build and push the image

The image's only inputs are the Dockerfile and the two manifests, so the tag is a digest of them. Unchanged inputs rebuild to the same tag, a changed dependency is forced to produce a new one, and a finished experiment can name the image that produced it:

```bash
TAG=$(cat Dockerfile pyproject.toml poetry.lock | sha256sum | cut -c1-12)
docker build -t pabloo22/msc-thesis:$TAG .
docker push pabloo22/msc-thesis:$TAG
```

Rent that tag rather than `:latest`: a mutable tag cannot be cited in a write-up, and a host that already cached the name may not re-pull a newer push of it.

### 2. Rent with enough disk

**150 GB.** A paper-scale trajectory on Qwen2.5-7B needs roughly ~17 GB for the image, ~15 GB for the base model under `HF_HOME`, and **~30 GB of merged checkpoints** -- `materialize()` holds two full 7B checkpoints at once while walking the adapter chain. `pull_before_run` fetches *every* adapter and measurement on the remote, so that share grows toward ~35 GB. 100 GB gets tight later; disk is a rounding error next to the GPU.

Paste this as the instance's **on-start script**. It stays deliberately small -- anything more belongs in `scripts/box_setup.sh`:

```bash
#!/bin/bash
mkdir -p /workspace /root/.config/rclone
cd /workspace && git clone https://github.com/Pabloo22/msc_thesis.git \
  || git -C /workspace/msc_thesis pull --ff-only
unzip -nq /workspace/msc_thesis/method/persona_vectors/dataset.zip \
  -d /workspace/msc_thesis
```

### 3. Copy the two credential files across

Neither belongs in the image, which is public on Docker Hub:

```bash
scp -P <port> .env            root@<host>:/workspace/msc_thesis/.env
scp -P <port> ~/.config/rclone/rclone.conf root@<host>:/root/.config/rclone/rclone.conf
```

### 4. Check the box before spending GPU time

```bash
bash scripts/box_setup.sh
```

It installs nothing. It reports every problem it finds at once and exits non-zero on any that would break a run (no visible GPU, python isn't the baked venv, driver too old, missing HF_TOKEN, unreachable MSC_STORE_REMOTE, or too little free disk). It also checks for **dependency drift** between the clone's `poetry.lock` and the image's baked lockfile.

Then smoke-test before the real thing:

```bash
python -m method.run_trajectory --config SMOKE_TINY --backend real
```