# Autoresearch on Ocean Network

Autonomous ML research agent that iteratively improves a GPT pretraining script to minimize validation bits-per-byte (val_bpb). Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

The key difference: everything runs **inside a single Docker container** on an [Ocean](https://dashboard.oncompute.ai/) GPU node (H200, 141GB VRAM) with a **local open-source LLM** — no API keys needed.

## From Karpathy's Experiment to Ocean

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) uses the Claude API to drive an agent loop that iteratively improves a GPT training script. It's a brilliant idea — let an LLM be the researcher — but it requires API keys, costs money per token, and runs on your own machine. Here's how we adapted it to run fully self-contained on Ocean Network:

### 1. Replace the API with a local LLM

The original calls Claude via the Anthropic API. We replaced that with **Qwen3-32B-AWQ** served locally through **vLLM**. The AWQ 4-bit quantization brings the model down to ~18GB VRAM, leaving the rest of the H200's 141GB for training. vLLM loads once and stays resident for all 200 iterations — no network calls, no API keys, no per-token costs.

### 2. Share one GPU between the agent and training

This is the core engineering challenge. The agent LLM and the training run need to coexist on the same GPU. We configure vLLM with `gpu_memory_utilization=0.25` (~35GB for weights + KV cache), leaving ~100GB for PyTorch training. The agent generates code, then training runs as a subprocess — they never compete for memory simultaneously because inference finishes before training starts.

### 3. Package everything in a single Docker container

Ocean's compute-to-data model runs a Docker container on a remote GPU node. We built a container on `nvidia/cuda:12.8.0-devel-ubuntu22.04` that includes PyTorch, vLLM, Flash Attention 3 (via `kernels`), and all dependencies. The entire pipeline — data download, tokenizer training, LLM loading, and the 200-iteration research loop — runs from a single entrypoint (`algo.py`).

### 4. Adapt the data pipeline for container execution

Karpathy's setup assumes a persistent local environment. In a container, nothing persists between runs. `prepare.py` handles this by downloading HuggingFace data shards and training a BPE tokenizer from scratch at container startup, caching everything under `~/.cache/autoresearch/` for the duration of the job.

### 5. Wire into Ocean's orchestrator

Ocean expects the algorithm at `/app/data/transformations/algorithm`. A symlink in the Dockerfile (`ln -sf /app/algo.py /app/data/transformations/algorithm`) bridges this. Results are written to `/data/outputs/results.json` so they're downloadable from the Ocean dashboard when the job completes.

### Alternative: use an API instead of a local LLM

You could also use the Claude API (or any other LLM API) from inside the container — just pass the API key as an environment variable and swap the vLLM calls for Anthropic SDK calls. This frees up the ~35GB reserved for the agent model, giving training the full GPU, and a stronger model like Claude Sonnet would likely produce fewer crashes and smarter changes. The tradeoff is API costs and a dependency on network access.

### The result

A few clicks give you an autonomous ML researcher that runs for hours on an H200 GPU, costs nothing beyond the compute rental, and produces a `results.json` with the full experiment history and winning code.

## How It Works

1. **Data prep** — Downloads HuggingFace data shards, trains a BPE tokenizer (`prepare.py`)
2. **Load agent LLM** — Qwen3-32B-AWQ via vLLM (~18GB VRAM, stays resident)
3. **Baseline run** — Runs the original `train.py` (5-min training budget), records val_bpb
4. **Agent loop** (up to 200 iterations):
   - LLM reads experiment history + current best `train.py`
   - Generates a hypothesis + complete new `train.py`
   - Syntax check → train (5 min) → evaluate val_bpb
   - If improved: keep. If not: revert to best.
   - `results.json` saved after every iteration

The user extracts `results["best"]["train_py"]` to get the winning code.

## Files

| File | Description |
|------|-------------|
| `algo.py` | Core agent loop — orchestrates LLM inference and training |
| `train.py` | GPT pretraining script (the file the agent modifies) |
| `prepare.py` | Data download, tokenizer, dataloader, evaluation (read-only) |
| `program.md` | Instructions for the agent LLM |
| `Dockerfile` | Container build (CUDA 12.8, Python, PyTorch, vLLM) |
| `plot_progress.py` | Generate progress charts from results |

## Usage

1. Go to [dashboard.oncompute.ai](https://dashboard.oncompute.ai/)
2. Select an **H200 GPU** environment
3. Configure the job and add payment
4. Open the **Ocean Orchestrator** in VS Code / your editor
5. Open this directory in the orchestrator and run the job — the container builds and executes `algo.py` autonomously
6. Download `results.json` from the outputs when complete

To plot results after a run:
```bash
python plot_progress.py path/to/results.json progress.png
```

## Results

### Qwen3-32B-AWQ — 0.7 Temperature (First Run)

![Qwen3-32B first run](assets/images/qwen32B_first_run_progress.png)

- **Baseline**: 1.0077 val_bpb
- **Best**: 0.9818 val_bpb (2.6% improvement)
- **201 iterations** over 5.5 hours, 29 successful runs (86% crash rate)
- Key improvements: increased model depth (8→10 layers), late-stage hyperparameter tuning

### Qwen3-32B-AWQ — 0.5 Temperature, 6 Hours

![Qwen3-32B 0.5temp 6h](assets/images/qwen32B_6h_0.5temp_progress.png)

- **Baseline**: 1.0227 val_bpb
- **Best**: 1.0072 val_bpb (1.5% improvement)
- **94 iterations** over 6 hours, 36 successful runs (62% crash rate)
- Lower crash rate than 0.7 temp, but much less improvement — the agent converged early and plateaued

### Qwen3-32B-AWQ — 0.5 Temperature, 12 Hours

![Qwen3-32B 0.5temp 12h](assets/images/qwen32B_12h_0.5temp_progress.png)

- **Baseline**: 1.0215 val_bpb
- **Best**: 1.0074 val_bpb (1.4% improvement)
- **201 iterations** over 12 hours, 52 successful runs (74% crash rate)
- Double the runtime of the first run but worse results — the agent got stuck and couldn't escape the local minimum

### Takeaway

Lower temperature (0.5 vs 0.7) reduces the crash rate (62-74% vs 86%) but produces significantly worse results. The more "creative" 0.7 temperature generates more broken code, but the successful mutations are bolder and lead to real architectural improvements (e.g. deeper models). At 0.5 temp the agent plays it safe, converges early to ~1.007 val_bpb, and stalls — even with 12 hours of compute it can't match what 0.7 temp achieved in 5.5 hours.
