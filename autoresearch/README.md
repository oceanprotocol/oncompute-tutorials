# Autoresearch on Ocean Network

Autonomous ML research agent that iteratively improves a GPT pretraining script to minimize validation bits-per-byte (val_bpb). Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

The key difference: everything runs **inside a single Docker container** on an [Ocean](https://dashboard.oncompute.ai/) GPU node (H200, 141GB VRAM) with a **local open-source LLM** — no API keys needed.

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

### Qwen3-32B-AWQ — First Run

![Qwen3-32B first run](assets/images/qwen32B_first_run_progress.png)

- **Baseline**: 1.0077 val_bpb
- **Best**: 0.9818 val_bpb (2.6% improvement)
- **201 iterations** over 5.5 hours, 30 successful runs (85% crash rate)
- Key improvements: increased model depth (8→10 layers), late-stage hyperparameter tuning
