# Slim Python base. torch wheels bundle the CUDA runtime (nvidia-* pip
# packages), so GPU support comes from pip — the node's container runtime
# injects the driver. No apt/toolchain install needed.
# NOTE: no WORKDIR. The node runs the algorithm via the RELATIVE path
# `data/transformations/algorithm`, so the working dir must stay `/`
# (python:slim default) for it to resolve to /data/transformations/algorithm.
FROM python:3.12-slim

# The node runs the container with a non-writable HOME (/), which breaks the
# Hugging Face model/dataset cache and wandb's local dirs. Point HOME and the
# caches at world-writable /tmp.
ENV HOME=/tmp \
    HF_HOME=/tmp/hf \
    WANDB_DIR=/tmp/wandb

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# The algorithm (algo.py) is injected by the node as rawcode and run via
# `python $ALGO` — no script COPY / no CMD needed here.
