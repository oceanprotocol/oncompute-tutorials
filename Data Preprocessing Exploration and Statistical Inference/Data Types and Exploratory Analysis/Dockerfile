# Slim Python base: python + pip preinstalled, no apt/toolchain install.
# All deps below ship as prebuilt manylinux wheels → fast build, fits the
# free compute env's short maxJobDuration (build + run share that budget).
# NOTE: no WORKDIR. The node runs the algorithm via the RELATIVE path
# `data/transformations/algorithm`, so the working dir must stay `/`
# (python:slim default) for it to resolve to /data/transformations/algorithm.
FROM python:3.12-slim

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Bake the existing dataset. /data is a runtime volume mount on the node, so
# data goes to /app/data (the script searches there). The algorithm is
# injected by the node as rawcode and run via `python $ALGO` — no script
# COPY / no CMD needed here.
COPY corporate_financial_data.csv /app/data/corporate_financial_data.csv
