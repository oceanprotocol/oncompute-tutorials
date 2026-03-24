"""Generate a progress chart from autoresearch results.json, matching Karpathy's style."""

import json
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def load_results(path):
    with open(path) as f:
        return json.load(f)


def shorten_description(desc):
    """Condense a verbose hypothesis into 2-3 word label, or return '' if none."""
    if not desc or desc in ("no hypothesis", "no description"):
        return ""
    d = desc.lower()
    # Strip common prefixes
    for prefix in ("hypothesis:", "changes:", "hypothesis ", "changes "):
        if d.startswith(prefix):
            d = d[len(prefix):].strip()
    # Map known patterns to short labels
    keywords = {
        "baseline": "baseline",
        "depth": "more depth",
        "layer": "more layers",
        "width": "wider model",
        "learning rate": "LR tuning",
        "lr ": "LR tuning",
        "lr=": "LR tuning",
        "batch size": "batch size",
        "batch_size": "batch size",
        "warmup": "warmup tuning",
        "warmdown": "warmdown tuning",
        "swiglu": "SwiGLU",
        "gelu": "GELU",
        "activation": "activation change",
        "dropout": "add dropout",
        "weight decay": "weight decay",
        "window": "window pattern",
        "head": "head config",
        "embedding": "embedding LR",
        "import": "fix imports",
        "dataclass": "fix imports",
        "scaling": "model scaling",
        "aspect": "aspect ratio",
        "momentum": "momentum tuning",
        "optimizer": "optimizer tuning",
        "init": "weight init",
        "simplif": "simplify",
    }
    for keyword, label in keywords.items():
        if keyword in d:
            return label
    # Fallback: take first few words
    words = d.split()[:3]
    short = " ".join(words).rstrip(".,;:")
    return short if short else ""

def plot_progress(results, output_path="progress.png"):
    iterations = results["iterations"]

    # Separate successful runs (have val_bpb) from crashes
    successful = []
    running_best = float("inf")
    for entry in iterations:
        if entry.get("val_bpb") is not None and entry["status"] in ("keep", "discard"):
            vbpb = entry["val_bpb"]
            # Skip extreme outliers for readability
            if vbpb > 2.0:
                continue
            running_best = min(running_best, vbpb)
            successful.append({
                "iteration": entry["iteration"],
                "val_bpb": vbpb,
                "status": entry["status"],
                "description": entry.get("description", ""),
                "running_best": running_best,
            })

    kept = [s for s in successful if s["status"] == "keep"]
    discarded = [s for s in successful if s["status"] == "discard"]

    # Count stats
    total_experiments = len(iterations)
    num_kept = len(kept)

    # Build running best line (step function through kept iterations)
    best_iters = [k["iteration"] for k in kept]
    best_vals = [k["val_bpb"] for k in kept]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot discarded points (light, in background)
    if discarded:
        ax.scatter(
            [d["iteration"] for d in discarded],
            [d["val_bpb"] for d in discarded],
            color="#b0d4b0", marker="o", s=30, alpha=0.5, zorder=2,
            label="Discarded",
        )

    # Plot running best as a step line
    if best_iters:
        # Extend steps to the last iteration
        step_x = []
        step_y = []
        for i, (xi, yi) in enumerate(zip(best_iters, best_vals)):
            if i > 0:
                step_x.append(xi)
                step_y.append(best_vals[i - 1])  # horizontal line at previous best
            step_x.append(xi)
            step_y.append(yi)
        # Extend to end
        last_iter = max(e["iteration"] for e in iterations)
        step_x.append(last_iter)
        step_y.append(best_vals[-1])

        ax.plot(step_x, step_y, color="#2ca02c", linewidth=1.5, alpha=0.7, zorder=3,
                label="Running best")

    # Plot kept points (prominent, with labels)
    if kept:
        ax.scatter(
            [k["iteration"] for k in kept],
            [k["val_bpb"] for k in kept],
            color="#2ca02c", marker="o", s=60, zorder=4,
            label="Kept",
        )
        # Add description labels to kept points with offset to avoid overlap
        offsets = []
        for i, k in enumerate(kept):
            desc = shorten_description(k["description"])
            if not desc:
                continue
            # Alternate vertical offset to reduce overlap in clusters
            y_offset = -12 if i % 2 == 0 else 10
            ax.annotate(
                desc,
                (k["iteration"], k["val_bpb"]),
                textcoords="offset points",
                xytext=(10, y_offset),
                fontsize=7,
                color="#2ca02c",
                alpha=0.85,
                rotation=25,
                ha="left",
                arrowprops=dict(arrowstyle="-", color="#2ca02c", alpha=0.3, lw=0.5),
            )

    # Formatting
    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel("Validation BPB (lower is better)", fontsize=12)
    ax.set_title(
        f"Autoresearch Progress: {total_experiments} Experiments, {num_kept} Kept Improvements",
        fontsize=13,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Y-axis: show reasonable range
    all_vals = [s["val_bpb"] for s in successful]
    if all_vals:
        y_min = min(all_vals) - 0.002
        y_max = max(all_vals) + 0.002
        # Cap y_max to avoid outlier stretching
        y_max = min(y_max, 1.02)
        ax.set_ylim(y_min, y_max)

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    results_path = sys.argv[1] if len(sys.argv) > 1 else "results/first_successful_run_qwen32B/outputs/results.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "progress.png"
    results = load_results(results_path)
    plot_progress(results, output_path)
