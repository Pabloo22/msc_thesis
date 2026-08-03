# The `base` tag, not `runtime` or `devel`: torch and vllm bring their own CUDA
# libraries as nvidia-*-cu12 wheels, so the only thing needed from the image is
# NVIDIA_VISIBLE_DEVICES/NVIDIA_DRIVER_CAPABILITIES, which is what exposes the
# host driver under vast.ai's container runtime. If some wheel turns out to
# dlopen a system library after all, 12.4.1-runtime-ubuntu22.04 adds cuBLAS and
# NCCL back for ~2 GB.
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# python3.12 comes from deadsnakes because 22.04 ships 3.10 and pyproject pins
# >=3.12,<3.13. rclone is not optional: method/sync.py shells out to the binary
# for every push/pull, so a box without it silently degrades to local-only work
# and loses its artifacts when the instance goes away. Installed from the
# official script rather than the jammy apt package: that package predates R2
# and mishandles its API (every request 501s once before a retry succeeds).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl ca-certificates && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        build-essential git unzip && \
    rm -rf /var/lib/apt/lists/* && \
    curl https://rclone.org/install.sh | bash

# A bucket-scoped R2 token (Cloudflare's least-privilege, recommended type) has
# no account-level bucket permissions, so rclone's pre-upload HeadBucket check
# 403s and rclone reports the whole copy as AccessDenied. This skips that
# check; safe here since the remote's bucket is always known to already exist.
ENV RCLONE_S3_NO_CHECK_BUCKET=true

# Deliberately no `update-alternatives` for python3: on this base /usr/bin/python3
# is the 3.10 that apt's own tooling (add-apt-repository) imports, and repointing
# it breaks them. The project venv goes on PATH below instead, so `python` is
# 3.12 for anything a shell on the box actually runs.

ENV POETRY_HOME=/opt/poetry
ENV PATH="/opt/poetry/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=2.2.1 python3 -

# Resolve and install the dependency set at build time rather than on the rental
# box. Two reasons: `poetry install` on a fresh instance downloads and unpacks
# ~15 GB (torch + the nvidia-*-cu12 wheels + vllm) before any GPU work starts,
# and doing it here means every box runs the bytes this image was tested with.
# Only the manifests are copied, so editing method/ never invalidates this layer.
# They also stay in the image on purpose: scripts/box_setup.sh diffs a clone's
# poetry.lock against /opt/app/poetry.lock to catch a box that checked out a
# commit whose dependencies this image predates.
WORKDIR /opt/app
COPY pyproject.toml poetry.lock ./
# in-project is load-bearing, not cosmetic: without it poetry puts the venv under
# /root/.cache/pypoetry/virtualenvs/ next to its download cache, so the cleanup
# below would delete the environment it just built.
RUN poetry config virtualenvs.in-project true && \
    poetry env use /usr/bin/python3.12 && \
    poetry install --no-root --no-interaction && \
    rm -rf /root/.cache/pypoetry /root/.cache/pip

ENV VIRTUAL_ENV=/opt/app/.venv
ENV PATH="/opt/app/.venv/bin:$PATH"

# With the venv pre-activated and creation disabled, `poetry run ...` inside the
# cloned repo reuses /opt/app/.venv instead of building a second one in-project.
# That keeps the README's commands working verbatim on the box. Set only after
# the install above, which needs creation enabled.
ENV POETRY_VIRTUALENVS_CREATE=false

# Keep the ~15 GB Qwen2.5-7B download next to the code on the big rental disk
# instead of in /root/.cache, so it is one directory to point at a volume or
# clear when reclaiming space.
ENV HF_HOME=/workspace/.cache/huggingface

WORKDIR /workspace

# Build and push. The tag is a digest of this file plus the two manifests, which
# together are the image's only inputs -- so a tag names exactly one dependency
# set, an unchanged input rebuilds to the same tag, and a thesis run can cite the
# image it used. Renting `:latest` instead would be unpinnable, and a host that
# has already cached that name may not re-pull a newer push of it.
#
#   TAG=$(cat Dockerfile pyproject.toml poetry.lock | sha256sum | cut -c1-12)
#   docker build -t pabloo22/msc-thesis:$TAG .
#   docker push pabloo22/msc-thesis:$TAG
