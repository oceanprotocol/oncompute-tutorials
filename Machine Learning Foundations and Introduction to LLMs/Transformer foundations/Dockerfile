# Slim Python base: python + pip preinstalled, no apt/toolchain install.
# All deps below ship as prebuilt manylinux wheels → fast build, fits the
# free compute env's short maxJobDuration (build + run share that budget).
# NOTE: no WORKDIR. The node runs the algorithm via the RELATIVE path
# `data/transformations/algorithm`, so the working dir must stay `/`
# (python:slim default) for it to resolve to /data/transformations/algorithm.
FROM python:3.12-slim

# The node runs the container with a non-writable HOME (/), which breaks
# matplotlib's config dir. Point HOME at world-writable /tmp. The HF cache
# must live in the image (not /tmp) so the models baked below survive into
# the runtime container.
ENV HOME=/tmp \
    MPLCONFIGDIR=/tmp/mplconfig \
    HF_HOME=/opt/hf

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# The algorithm container runs with NO network access (C2D sandboxing) —
# only the image build has internet. Every model/tokenizer the algorithm
# loads must therefore be downloaded here, into the image. Offline mode at
# runtime makes transformers read straight from this cache instead of
# burning minutes retrying huggingface.co. chmod: the node runs the
# container as an arbitrary non-root uid.
RUN python -c "from transformers import GPT2Tokenizer, BertTokenizer, BertModel, T5Tokenizer; \
    GPT2Tokenizer.from_pretrained('gpt2'); \
    BertTokenizer.from_pretrained('bert-base-uncased'); \
    BertModel.from_pretrained('bert-base-uncased'); \
    T5Tokenizer.from_pretrained('t5-small', legacy=False)" \
    && chmod -R a+rwX /opt/hf

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# The algorithm (algo.py) is injected by the node as rawcode and run via
# `python $ALGO` — no script COPY / no CMD needed here.
