"""
Autonomous autoresearch agent loop for ocean network.

Runs inside a Docker container on a remote GPU node.
Uses a local open-source LLM (Qwen3-32B-AWQ via vLLM) to iteratively
improve train.py, measuring val_bpb as the optimization target.
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AGENT_MODEL = "Qwen/Qwen3-32B-AWQ"
GPU_MEMORY_UTILIZATION = 0.25  # ~35GB for weights+KV cache, rest for training
MAX_ITERATIONS = 200
TRAINING_TIMEOUT = 600  # 10 minutes
MAX_MODEL_LEN = 40960   # total context window (input + output)
MAX_OUTPUT_TOKENS = 16384  # max tokens for LLM output (enough for full train.py)
TEMPERATURE = 0.7
STAGNATION_THRESHOLD = 5  # consecutive non-improvements before nudge
MAX_HISTORY_IN_PROMPT = 20  # only show last N iterations in prompt
MAX_CONSECUTIVE_CRASHES = 3  # after N crashes with same pattern, force new direction

RESULTS_PATH = "/app/results.json"
RESULTS_OUTPUT_PATH = "/data/outputs/results.json"
TRAIN_PY_PATH = "/app/train.py"
PREPARE_PY_PATH = "/app/prepare.py"
PROGRAM_MD_PATH = "/app/program.md"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def log(msg):
    """Print with timestamp."""
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def read_file(path):
    """Read a file and return its contents."""
    with open(path, "r") as f:
        return f.read()


def write_file(path, content):
    """Write content to a file."""
    with open(path, "w") as f:
        f.write(content)


def save_results(results):
    """Save results to both /app/ and /data/outputs/."""
    data = json.dumps(results, indent=2)
    write_file(RESULTS_PATH, data)
    os.makedirs(os.path.dirname(RESULTS_OUTPUT_PATH), exist_ok=True)
    write_file(RESULTS_OUTPUT_PATH, data)


def parse_metrics(stdout):
    """Extract key: value pairs from the --- block in training output."""
    metrics = {}
    in_block = False
    for line in stdout.split("\n"):
        line = line.strip()
        if line == "---":
            in_block = True
            continue
        if in_block and ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            try:
                metrics[key] = float(value)
            except ValueError:
                metrics[key] = value
    return metrics


def extract_code(llm_output):
    """Extract the first ```python code block from LLM output.

    Handles Qwen3 <think>...</think> blocks by stripping them first.
    """
    # Strip Qwen3 thinking blocks
    text = re.sub(r"<think>.*?</think>", "", llm_output, flags=re.DOTALL).strip()

    # Try ```python first
    pattern = r"```python\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fall back to any ``` block
    pattern = r"```\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_hypothesis(llm_output):
    """Extract hypothesis/changes text from before the code block."""
    # Strip thinking blocks
    text = re.sub(r"<think>.*?</think>", "", llm_output, flags=re.DOTALL).strip()
    code_block_start = text.find("```")
    if code_block_start > 0:
        hypothesis_text = text[:code_block_start].strip()
    else:
        hypothesis_text = "no description"
    # Strip "Changes:" prefix if present
    hypothesis_text = re.sub(r'^(?:Changes|Hypothesis)\s*:\s*', '', hypothesis_text, flags=re.IGNORECASE)
    # Clean up: take first 200 chars, remove markdown
    return re.sub(r'[#*`\n]+', ' ', hypothesis_text).strip()[:200]


def syntax_check(code):
    """Check if code compiles. Returns None on success, error string on failure."""
    try:
        compile(code, "train.py", "exec")
        return None
    except SyntaxError as e:
        return f"SyntaxError: {e}"


def run_training(train_py_content):
    """
    Write train.py, run it as a subprocess, parse metrics.
    Returns dict with keys: success, metrics, stdout, stderr, error
    """
    write_file(TRAIN_PY_PATH, train_py_content)

    try:
        result = subprocess.run(
            [sys.executable, TRAIN_PY_PATH],
            capture_output=True,
            text=True,
            timeout=TRAINING_TIMEOUT,
            cwd="/app",
        )
    except subprocess.TimeoutExpired as e:
        stderr_text = e.stderr if isinstance(e.stderr, str) else (e.stderr.decode() if e.stderr else "")
        return {
            "success": False,
            "metrics": {},
            "stdout": e.stdout if isinstance(e.stdout, str) else (e.stdout.decode() if e.stdout else ""),
            "stderr": stderr_text,
            "error": f"Training timed out after {TRAINING_TIMEOUT}s",
        }

    stdout = result.stdout or ""
    stderr = result.stderr or ""

    if result.returncode != 0:
        # Get the last 50 lines of combined output for error context
        error_lines = (stdout + "\n" + stderr).strip().split("\n")
        error_tail = "\n".join(error_lines[-50:])
        return {
            "success": False,
            "metrics": {},
            "stdout": stdout,
            "stderr": stderr,
            "error": f"Training crashed (exit code {result.returncode}):\n{error_tail}",
        }

    metrics = parse_metrics(stdout)
    if "val_bpb" not in metrics:
        return {
            "success": False,
            "metrics": metrics,
            "stdout": stdout,
            "stderr": stderr,
            "error": "Training completed but val_bpb not found in output",
        }

    return {
        "success": True,
        "metrics": metrics,
        "stdout": stdout,
        "stderr": stderr,
        "error": None,
    }


def summarize_tried_directions(iterations):
    """Summarize what has been tried to help the LLM avoid repeating itself."""
    crashed_descs = []
    successful_descs = []
    for it in iterations:
        desc = it.get("description", "")[:100]
        if not desc or desc in ("baseline", "no description"):
            continue
        if it["status"] == "crash":
            crashed_descs.append(desc)
        elif it["status"] == "discard":
            successful_descs.append(f"bpb={it['val_bpb']:.4f}: {desc}")

    # Deduplicate similar crash descriptions (first 40 chars)
    seen = set()
    unique_crashes = []
    for d in crashed_descs:
        key = d[:40].lower()
        if key not in seen:
            seen.add(key)
            unique_crashes.append(d)

    summary = []
    if unique_crashes:
        summary.append("**Approaches that CRASHED** (do not repeat these):")
        for d in unique_crashes[-10:]:  # last 10 unique crashes
            summary.append(f"- {d}")
    if successful_descs:
        summary.append("\n**Approaches that ran but did NOT improve** (try something different):")
        for d in successful_descs[-8:]:  # last 8
            summary.append(f"- {d}")
    return "\n".join(summary)


def classify_error(error_text):
    """Classify a runtime error to give the LLM actionable feedback."""
    if not error_text:
        return None
    e = error_text.lower()
    if "cuda out of memory" in e or "out of memory" in e or "oom" in e:
        return "OOM"
    if "timed out" in e or "timeout" in e:
        return "TIMEOUT"
    if "shape" in e or "size mismatch" in e or "dimension" in e:
        return "SHAPE_MISMATCH"
    if "import" in e or "no module" in e or "cannot import" in e:
        return "IMPORT_ERROR"
    if "nan" in e or "inf" in e:
        return "NUMERICAL"
    return "RUNTIME_ERROR"


def build_prompt(program_md, prepare_py_summary, best_train_py, results,
                 last_error=None, consecutive_non_improvements=0):
    """Build the full prompt for the LLM.

    Uses a trimmed prepare.py summary (just constants + API signatures)
    and only the last MAX_HISTORY_IN_PROMPT iterations to keep prompt compact.
    """
    parts = []

    # System instructions
    parts.append(program_md)
    parts.append("\n---\n")

    # Read-only context (trimmed — just the API the model needs to know)
    parts.append("## prepare.py key API (READ-ONLY)\n")
    parts.append(f"```python\n{prepare_py_summary}\n```\n")
    parts.append("\n---\n")

    # Current best train.py
    parts.append("## Current best train.py\n")
    parts.append(f"```python\n{best_train_py}\n```\n")
    parts.append("\n---\n")

    # Experiment history (trimmed to last N entries)
    parts.append("## Experiment History\n")
    iterations = results["iterations"]
    if iterations:
        # Always show baseline + last N
        shown = []
        if iterations[0] not in iterations[-MAX_HISTORY_IN_PROMPT:]:
            shown.append(iterations[0])  # always include baseline
        shown.extend(iterations[-MAX_HISTORY_IN_PROMPT:])

        if len(iterations) > len(shown):
            parts.append(f"(Showing {len(shown)} of {len(iterations)} total iterations)\n\n")

        parts.append("| Iter | val_bpb | VRAM (MB) | Status | Description |\n")
        parts.append("|------|---------|-----------|--------|-------------|\n")
        for entry in shown:
            val = f"{entry['val_bpb']:.6f}" if entry.get("val_bpb") else "N/A"
            vram = f"{entry.get('peak_vram_mb', 0):.0f}" if entry.get("peak_vram_mb") else "N/A"
            desc = entry.get('description', '')[:80]
            parts.append(f"| {entry['iteration']} | {val} | {vram} | {entry['status']} | {desc} |\n")
    else:
        parts.append("No experiments run yet.\n")
    parts.append("\n")

    # Best so far
    if results.get("best"):
        best = results["best"]
        parts.append(f"**Current best**: iteration {best['iteration']}, val_bpb={best['val_bpb']:.6f}\n\n")

    # Summary of tried directions (helps avoid repetition)
    if len(iterations) > 5:
        tried_summary = summarize_tried_directions(iterations)
        if tried_summary:
            parts.append("## What Has Been Tried\n")
            parts.append(tried_summary)
            parts.append("\n\n")

    # Error from last iteration (with classification)
    if last_error:
        error_type = classify_error(last_error)
        truncated_error = last_error[:500]
        parts.append("## Previous Iteration Error\n")
        if error_type == "OOM":
            parts.append("**Error type: OUT OF MEMORY.** Your model was too large. "
                         "Reduce model size, batch size, or sequence length.\n")
        elif error_type == "TIMEOUT":
            parts.append("**Error type: TIMEOUT.** Training took too long. "
                         "Reduce model size or batch size to fit within 5 minutes.\n")
        elif error_type == "SHAPE_MISMATCH":
            parts.append("**Error type: TENSOR SHAPE MISMATCH.** Check that dimensions are consistent "
                         "across model config, attention heads, and embeddings.\n")
        elif error_type == "IMPORT_ERROR":
            parts.append("**Error type: IMPORT ERROR.** You used a module that doesn't exist. "
                         "Only use imports from the baseline.\n")
        elif error_type == "NUMERICAL":
            parts.append("**Error type: NUMERICAL INSTABILITY (NaN/Inf).** "
                         "Learning rate may be too high, or initialization is unstable.\n")
        parts.append(f"```\n{truncated_error}\n```\n")
        parts.append("Fix the issue or try a completely different approach.\n\n")

    # Stagnation nudge
    if consecutive_non_improvements >= STAGNATION_THRESHOLD:
        parts.append("## NOTE: Stagnation Detected\n")
        parts.append(f"The last {consecutive_non_improvements} iterations did not improve val_bpb. ")
        if consecutive_non_improvements >= STAGNATION_THRESHOLD * 2:
            parts.append("You have been stuck for a LONG time. Make a BOLD change you haven't tried yet. ")
            parts.append("Consider: very different model depth/width ratios, different batch sizes (2x or 0.5x), ")
            parts.append("different warmdown ratios, or significantly different learning rates. ")
        else:
            parts.append("Try a significantly different hyperparameter configuration: ")
            parts.append("different learning rates, batch size, depth, width, or warmdown ratio. ")
        parts.append("Do NOT try to simplify or remove architecture components — that always causes crashes. ")
        parts.append("Keep the same architecture but change the numbers.\n\n")

    parts.append("Now propose your next experiment. Start with 'Changes:' followed by a 1-2 sentence description, "
                 "then the COMPLETE train.py in a ```python code block. "
                 "The file must be complete and runnable — do NOT truncate or use '...' placeholders. "
                 "Do NOT use <think> tags or chain-of-thought — go straight to the description and code.\n")

    return "".join(parts)


def make_prepare_py_summary(prepare_py):
    """Extract just the constants and function signatures from prepare.py.

    The LLM doesn't need the full implementation — just the API it imports from.
    """
    lines = [
        "# Key constants (from prepare.py — DO NOT redefine these)",
        "MAX_SEQ_LEN = 2048       # context length",
        "TIME_BUDGET = 300        # training time budget in seconds (5 minutes)",
        "EVAL_TOKENS = 40 * 524288  # number of tokens for val eval",
        "VOCAB_SIZE = 8192",
        "",
        "# Classes and functions available for import:",
        "# from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb",
        "",
        "# Tokenizer API:",
        "#   tokenizer = Tokenizer.from_directory()",
        "#   vocab_size = tokenizer.get_vocab_size()  # returns 8192",
        "#   bos_id = tokenizer.get_bos_token_id()",
        "",
        "# Dataloader API:",
        "#   loader = make_dataloader(tokenizer, batch_size, seq_len, 'train'|'val')",
        "#   x, y, epoch = next(loader)  # x, y are [B, T] long tensors on CUDA",
        "",
        "# Evaluation API:",
        "#   val_bpb = evaluate_bpb(model, tokenizer, batch_size)",
        "#   - model must accept (idx, targets, reduction='none'|'mean')",
        "#   - returns float (bits per byte, lower is better)",
        "",
        "# train.py MUST print a summary block at the end like:",
        "# print('---')",
        "# print(f'val_bpb:          {val_bpb:.6f}')",
        "# print(f'training_seconds: {total_training_time:.1f}')",
        "# print(f'peak_vram_mb:     {peak_vram_mb:.1f}')",
        "# ... (other metrics optional)",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def main():
    log("=== Autoresearch Agent Starting ===")
    started_at = datetime.now(timezone.utc).isoformat()

    # -----------------------------------------------------------------------
    # Phase 1: Data preparation
    # -----------------------------------------------------------------------
    log("Phase 1: Running prepare.py (data download + tokenizer)...")
    result = subprocess.run(
        [sys.executable, PREPARE_PY_PATH],
        cwd="/app",
    )
    if result.returncode != 0:
        log("ERROR: prepare.py failed!")
        sys.exit(1)
    log("Phase 1 complete.")

    # -----------------------------------------------------------------------
    # Phase 2: Load agent LLM
    # -----------------------------------------------------------------------
    log(f"Phase 2: Loading LLM ({AGENT_MODEL})...")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=AGENT_MODEL,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        dtype="auto",
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
    )
    log("Phase 2 complete. LLM loaded.")

    # -----------------------------------------------------------------------
    # Phase 3: Baseline run
    # -----------------------------------------------------------------------
    log("Phase 3: Running baseline train.py...")
    baseline_code = read_file(TRAIN_PY_PATH)

    results = {
        "metadata": {
            "agent_model": AGENT_MODEL,
            "started_at": started_at,
        },
        "iterations": [],
        "best": None,
    }

    run_result = run_training(baseline_code)

    if not run_result["success"]:
        log(f"ERROR: Baseline training failed: {run_result['error']}")
        sys.exit(1)

    baseline_entry = {
        "iteration": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": "baseline",
        "success": True,
        "val_bpb": run_result["metrics"]["val_bpb"],
        "peak_vram_mb": run_result["metrics"].get("peak_vram_mb", 0),
        "status": "keep",
        "train_py": baseline_code,
    }
    results["iterations"].append(baseline_entry)
    results["best"] = baseline_entry.copy()
    save_results(results)

    log(f"Phase 3 complete. Baseline val_bpb: {baseline_entry['val_bpb']:.6f}, "
        f"VRAM: {baseline_entry['peak_vram_mb']:.0f} MB")

    # -----------------------------------------------------------------------
    # Phase 4: Agent loop
    # -----------------------------------------------------------------------
    log("Phase 4: Starting agent loop...")
    program_md = read_file(PROGRAM_MD_PATH)
    prepare_py = read_file(PREPARE_PY_PATH)
    prepare_py_summary = make_prepare_py_summary(prepare_py)
    best_train_py = baseline_code
    best_val_bpb = baseline_entry["val_bpb"]
    last_error = None
    consecutive_non_improvements = 0

    for iteration in range(1, MAX_ITERATIONS + 1):
        log(f"--- Iteration {iteration}/{MAX_ITERATIONS} ---")
        log(f"Current best val_bpb: {best_val_bpb:.6f}")

        # Build prompt
        prompt = build_prompt(
            program_md=program_md,
            prepare_py_summary=prepare_py_summary,
            best_train_py=best_train_py,
            results=results,
            last_error=last_error,
            consecutive_non_improvements=consecutive_non_improvements,
        )
        last_error = None  # Reset for this iteration

        # Generate LLM response (using chat API with thinking disabled)
        log("Generating LLM response...")
        t0 = time.time()
        messages = [{"role": "user", "content": prompt}]
        outputs = llm.chat(
            messages,
            sampling_params=sampling_params,
            chat_template_kwargs={"enable_thinking": False},
        )
        llm_output = outputs[0].outputs[0].text
        t1 = time.time()
        log(f"LLM generation took {t1 - t0:.1f}s, "
            f"output tokens: {len(outputs[0].outputs[0].token_ids)}")

        # Extract code (handles Qwen3 <think> blocks)
        new_code = extract_code(llm_output)
        description = extract_hypothesis(llm_output)

        if new_code is None:
            log(f"ERROR: Could not extract code from LLM output")
            log(f"  First 200 chars: {llm_output[:200]!r}")
            entry = {
                "iteration": iteration,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "description": f"parse_failure: {description}",
                "success": False,
                "val_bpb": None,
                "peak_vram_mb": None,
                "status": "crash",
            }
            results["iterations"].append(entry)
            save_results(results)
            last_error = ("Could not extract a ```python code block from your response. "
                         "You MUST include the COMPLETE train.py inside a ```python code block. "
                         "Do not use any other format.")
            consecutive_non_improvements += 1
            continue

        # Syntax check
        syntax_err = syntax_check(new_code)
        if syntax_err:
            log(f"ERROR: Syntax error in generated code: {syntax_err}")
            entry = {
                "iteration": iteration,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "description": f"syntax_error: {syntax_err[:100]}",
                "success": False,
                "val_bpb": None,
                "peak_vram_mb": None,
                "status": "crash",
            }
            results["iterations"].append(entry)
            save_results(results)
            last_error = f"Your code had a syntax error:\n{syntax_err}\nPlease output the COMPLETE, valid train.py file."
            consecutive_non_improvements += 1
            continue

        # Run training
        log(f"Running training: {description[:80]}...")
        run_result = run_training(new_code)

        if not run_result["success"]:
            log(f"Training failed: {run_result['error'][:200]}")
            entry = {
                "iteration": iteration,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "description": description,
                "success": False,
                "val_bpb": None,
                "peak_vram_mb": None,
                "status": "crash",
            }
            results["iterations"].append(entry)
            save_results(results)
            # Revert to best
            write_file(TRAIN_PY_PATH, best_train_py)
            last_error = run_result["error"]
            consecutive_non_improvements += 1
            continue

        val_bpb = run_result["metrics"]["val_bpb"]
        peak_vram_mb = run_result["metrics"].get("peak_vram_mb", 0)

        # Decision: keep or discard
        if val_bpb < best_val_bpb:
            status = "keep"
            best_val_bpb = val_bpb
            best_train_py = new_code
            consecutive_non_improvements = 0
            log(f"IMPROVEMENT! val_bpb: {val_bpb:.6f} (was {results['best']['val_bpb']:.6f})")
        else:
            status = "discard"
            # Revert to best
            write_file(TRAIN_PY_PATH, best_train_py)
            consecutive_non_improvements += 1
            log(f"No improvement. val_bpb: {val_bpb:.6f} (best: {best_val_bpb:.6f})")

        entry = {
            "iteration": iteration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "description": description,
            "success": True,
            "val_bpb": val_bpb,
            "peak_vram_mb": peak_vram_mb,
            "status": status,
        }
        if status == "keep":
            entry["train_py"] = new_code

        results["iterations"].append(entry)
        if status == "keep":
            results["best"] = entry.copy()
        save_results(results)

    log("=== Agent loop complete ===")
    log(f"Best val_bpb: {best_val_bpb:.6f}")
    log(f"Total iterations: {len(results['iterations'])}")


if __name__ == "__main__":
    main()
