# Slim Python base. torch and bitsandbytes wheels bundle the CUDA runtime
# (nvidia-* pip packages), so GPU support comes from pip — the node's container
# runtime injects the driver.
# NOTE: no WORKDIR. The node runs the algorithm via the RELATIVE path
# `data/transformations/algorithm`, so the working dir must stay `/`
# (python:slim default) for it to resolve to /data/transformations/algorithm.
FROM python:3.12-slim

# gcc + libc headers: triton JIT-compiles its GPU launcher stubs at runtime
# (torch dispatches Qwen3's rotary-embedding matmul to a triton kernel), and
# python:slim ships no C compiler — without this the job dies at the first
# forward pass with "Failed to find C compiler". libc6-dev is needed because
# --no-install-recommends drops it and triton's stubs include Python.h/stdio.h.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# The node runs the container with a non-writable HOME (/), which breaks
# wandb's local dirs. Point HOME at world-writable /tmp. The HF cache must
# live in the image (not /tmp) so the model/dataset baked below survive into
# the runtime container. WANDB_DIR points into /data/outputs so the offline
# wandb run ships back in the results tar (sync later with `wandb sync`).
ENV HOME=/tmp \
    HF_HOME=/opt/hf \
    WANDB_DIR=/data/outputs/wandb

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# The algorithm container runs with NO network access (C2D sandboxing) —
# only the image build has internet. Bake the model+tokenizer and dataset
# into the image; the offline flags below make transformers/datasets read
# straight from this cache. snapshot_download fetches the repo files without
# loading 8B params into build RAM. chmod: the node runs the container as an
# arbitrary non-root uid, and datasets writes map/filter caches and lock
# files next to the cached data.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-8B')" \
    && python -c "from datasets import load_dataset; load_dataset('Open-Orca/SlimOrca-Dedup')" \
    && chmod -R a+rwX /opt/hf

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    WANDB_MODE=offline

# The algorithm (algo.py) is injected by the node as rawcode and run via
# `python $ALGO` — no script COPY / no CMD needed here.
