FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12
RUN apt-get update && apt-get install -y software-properties-common curl unzip && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y python3.12 python3.12-venv python3.12-dev && \
    # Clean up to keep the image size small
    rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as the default python/python3 command
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install Poetry 2.2.1
ENV POETRY_HOME=/opt/poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=2.2.1 python3 -

# Add Poetry to the system PATH so it works automatically
ENV PATH="/opt/poetry/bin:$PATH"

RUN poetry config virtualenvs.in-project true
# docker build -t pabloo22/my-vast-env:latest .
