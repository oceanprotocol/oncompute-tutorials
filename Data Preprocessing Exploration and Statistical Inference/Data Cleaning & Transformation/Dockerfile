# Slim Python base: python + pip preinstalled, no apt/toolchain install.
# All deps below ship as prebuilt manylinux wheels → fast build, fits the
# free compute env's short maxJobDuration (build + run share that budget).
# NOTE: no WORKDIR. The node runs the algorithm via the RELATIVE path
# `data/transformations/algorithm`, so the working dir must stay `/`
# (python:slim default) for it to resolve to /data/transformations/algorithm.
FROM python:3.12-slim

# The node runs the container with a non-writable HOME (/), which breaks
# matplotlib's config dir and numba's JIT cache (umap → numba needs a writable
# cache locator, else "cannot cache 'rdist': no locator available"). Point HOME
# and both caches at world-writable /tmp.
ENV HOME=/tmp \
    MPLCONFIGDIR=/tmp/mplconfig \
    NUMBA_CACHE_DIR=/tmp/numbacache

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Bake the existing datasets. /data is a runtime volume mount on the node, so
# data goes to /app/data (the script searches there). The algorithm is
# injected by the node as rawcode and run via `python $ALGO` — no script
# COPY / no CMD needed here.
# employees_clean.csv is the reference answer key, not a pipeline input → not baked.
COPY employees_personal_messy.csv    /app/data/employees_personal_messy.csv
COPY employees_work_messy.csv        /app/data/employees_work_messy.csv
COPY employees_performance_messy.csv /app/data/employees_performance_messy.csv
COPY employees_extended.csv          /app/data/employees_extended.csv
