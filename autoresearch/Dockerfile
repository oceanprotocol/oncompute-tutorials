FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-dev python3-venv python3-pip \
    git curl wget \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Install uv for fast package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install Python dependencies (cached unless these lines change)
RUN uv pip install --system --no-cache-dir \
    torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

RUN uv pip install --system --no-cache-dir \
    "vllm>=0.17.0" \
    kernels>=0.11.7 \
    rustbpe>=0.1.0 \
    tiktoken>=0.11.0 \
    pyarrow>=21.0.0 \
    huggingface-hub \
    requests>=2.32.0 \
    numpy>=2.2.6

# Copy application code (changes here won't invalidate install layers)
WORKDIR /app
COPY . /app

# Ocean platform runs: python /app/data/transformations/algorithm
RUN mkdir -p /app/data/transformations && ln -sf /app/algo.py /app/data/transformations/algorithm

CMD ["python", "algo.py"]
