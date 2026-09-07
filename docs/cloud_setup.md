# Cloud reproduction

Our experiments used an NVIDIA RTX 4090, rented through Vast.ai.

## Build the pinned image

```bash
TAG=$(cat Dockerfile pyproject.toml poetry.lock | sha256sum | cut -c1-12)
IMAGE=your-registry-user/msc-thesis
docker build -t $IMAGE:$TAG .
docker push $IMAGE:$TAG
git rev-parse HEAD
```

Rent `$IMAGE:$TAG`; do not use `latest`.

## Prepare the instance

```bash
mkdir -p /workspace /root/.config/rclone
cd /workspace
git clone https://github.com/Pabloo22/msc_thesis.git
cd msc_thesis
git checkout <recorded-commit>
curl -L --fail -o method/persona_vectors/dataset.zip \
  https://raw.githubusercontent.com/safety-research/persona_vectors/b8e0f044fe2410a6fad579f38324f03f13b4e917/dataset.zip
echo "6913afdd712997599016444e789d2a4a5e383b6418e6596a4598cebdb97e943e  method/persona_vectors/dataset.zip" | sha256sum -c -
unzip -nq method/persona_vectors/dataset.zip -d .
```

Create `.env` locally:

```ini
HF_TOKEN=...
OPENAI_API_KEY=...
MSC_STORE_REMOTE=msc-thesis:msc-thesis  # optional rclone remote:folder
```

From the workstation:

```bash
scp -P <port> .env root@<host>:/workspace/msc_thesis/.env
scp -P <port> ~/.config/rclone/rclone.conf root@<host>:/root/.config/rclone/rclone.conf  # if used
```

Then validate the instance and pipeline:

```bash
bash scripts/box_setup.sh
python -m method.run_trajectory --config SMOKE_TINY --backend real
```

Run the commands under [Reproduce thesis-scale results](../README.md#reproduce-thesis-scale-results) in the listed order. Results are written to `trajectories/` and `plots/real/`; interrupted runs resume from existing artifacts.

## Preserve artifacts

With `MSC_STORE_REMOTE` set, runs sync automatically. Manual commands:

```bash
poetry run python -m method.sync pull
poetry run python -m method.sync pull-plots
poetry run python -m method.sync push --verbose
```
