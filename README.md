# Ocean Network Compute Tutorials

A collection of hands-on tutorials and examples for running **Compute-to-Data** jobs on Ocean Protocol.

This repository demonstrates how to execute algorithms on datasets without exposing the raw data, using Ocean Nodes and the Ocean Orchestrator tooling. Tutorials are organized as a learning path — from data fundamentals through classical ML to large language models and autonomous ML research.

---

## Repository Structure

```
.
├── Data Preprocessing Exploration and Statistical Inference/
│   ├── Data Types and Exploratory Analysis/
│   └── Data Cleaning & Transformation/
│
├── Machine Learning Foundations and Introduction to LLMs/
│   ├── Clustering/
│   └── Transformer foundations/
│
├── Deep Learning and Large Language Models — Advanced Topics/
│   ├── General Fine-Tuning and Model Usage Info.md
│   ├── Encoder Fine-Tuning/
│   └── Decoder Fine-Tuning/
│
└── autoresearch/
```

---

## Tutorials

### Data Preprocessing, Exploration and Statistical Inference

#### Data Types and Exploratory Analysis

Walks through the four foundational steps of EDA — type inspection, descriptive statistics, distribution analysis, and correlation — applied to a corporate financial dataset of 500 firms across multiple industries and countries. The dataset is intentionally messy (missing R&D values, skewed revenues, extreme outliers) to reflect real-world conditions.

**Files:** `eda.py`, `eda.md`, `corporate_financial_data.csv`

---

#### Data Cleaning, Feature Engineering & Dimensionality Reduction

Covers the full preparation pipeline a practitioner encounters before modelling: cleaning inconsistencies and errors, handling missing values, engineering features into algorithm-ready form, and reducing dimensionality. Applied to a multi-source employee dataset with messy string fields, duplicate rows, and misaligned joins.

**Files:** `cleaning.py`, `cleaning.md`, sample CSV files

---

### Machine Learning Foundations and Introduction to LLMs

#### Unsupervised Learning: Clustering

Comprehensive reference and tutorial covering the major clustering paradigms — centroid-based (k-Means, k-Means++), hierarchical (agglomerative with Ward/complete/average/single linkage), density-based (DBSCAN, HDBSCAN), BIRCH, and Affinity Propagation — alongside internal and external evaluation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz, ARI). The companion script runs all algorithms on the Wine dataset and generates visualizations.

**Files:** `clustering.py`, `README.md`, `Dockerfile`, `requirements.txt`

---

#### Transformer Foundations

Deep-dive into the Transformer architecture (Vaswani et al., 2017): QKV attention formulation, multi-head attention, positional encodings (sinusoidal, RoPE, ALiBi), causal masking, encoder-only / decoder-only / encoder-decoder architectures, tokenization algorithms (BPE, WordPiece, SentencePiece), and pretraining objectives (MLM, CLM, span corruption, ELECTRA). Includes the full model landscape from BERT (2018) to LLaMA 3 and Mixtral.

The companion script runs three interactive labs — tokenization, attention visualization, and embedding clusters — and generates BertViz HTML visualizations.

**Files:** `transformer_foundations.py`, `README.md`, `Dockerfile`, `requirements.txt`

---

### Deep Learning and Large Language Models — Advanced Topics

#### General Fine-Tuning and Model Usage Info

Reference guide covering tokenization mechanics, padding strategies, model selection (encoder vs. decoder vs. encoder-decoder), and fine-tuning best practices: learning rate schedules, AdamW, gradient accumulation, early stopping, handling long sequences, class imbalance, and hyperparameter tuning.

**File:** `General Fine-Tuning and Model Usage Info.md`

---

#### Encoder Fine-Tuning

Covers fine-tuning encoder-only transformer models (BERT family). Explains bidirectional self-attention, sequence-level and token-level classification heads, multi-task learning, and masked answer prediction. Includes PyTorch Lightning training setup with PEFT support.

**Files:** `encoder_finetuning.py`, `README.md`, `Dockerfile`

---

#### Decoder Fine-Tuning

Covers fine-tuning decoder-only LLMs. Explains the autoregressive architecture, causal masking, cross-entropy loss over completion tokens (prompt masking), and Parameter-Efficient Fine-Tuning (PEFT) with LoRA — enabling adaptation of billion-parameter models on consumer hardware.

**Files:** `decoder_finetuning.py`, `README.md`, `Dockerfile`

---

### Autoresearch on Ocean Network

Autonomous ML research agent that iteratively improves a GPT pretraining script to minimize validation bits-per-byte (val_bpb). Adapted from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) to run fully self-contained on Ocean Network GPU nodes — no API keys needed. A local open-source LLM (Qwen3-14B via vLLM) acts as the research agent, sharing the H200 GPU with the training process.

The agent loop runs up to 200 iterations: the LLM reads the experiment history and current best `train.py`, generates a hypothesis and a new training script, runs a 5-minute training budget, and keeps the result only if it improves val_bpb.

**Benchmark results across five runs:**

| Run | Model | Setup | Iterations | Best val_bpb | Improvement |
|---|---|---|---|---|---|
| 1 | Qwen3-32B-AWQ | 1×H200, 0.7 temp | 201 | 0.9818 | 2.6% |
| 2 | Qwen3-32B-AWQ | 1×H200, 0.5 temp, 6h | 94 | 1.0072 | 1.5% |
| 3 | Qwen3-32B-AWQ | 1×H200, 0.5 temp, 12h | 201 | 1.0074 | 1.4% |
| 4 | Qwen3.5-27B | 2×H200, 0.5 temp | 77 | 0.9993 | 2.5% |
| 5 | Qwen3-14B | 1×H200, 0.7 temp | 165 | 0.9967 | 2.9% |

Key finding: lower temperature (0.5 vs 0.7) reduces crash rate but produces worse results. The smallest model (Qwen3-14B) achieved the best improvement due to faster iteration throughput.

**Files:** `algo.py`, `algo_qwen3-32B.py`, `algo_qwen3.5-27B.py`, `train.py`, `prepare.py`, `program.md`, `plot_progress.py`, `Dockerfile`

**Usage:**
1. Go to [dashboard.oncompute.ai](https://dashboard.oncompute.ai/)
2. Select a single H200 GPU environment
3. Open this directory in the Ocean Orchestrator and run the job
4. Download `results.json` from the outputs when complete

---

## Running on Ocean Network

All tutorials with a `Dockerfile` are ready to run on [Ocean Network compute](https://dashboard.oncompute.ai/) via the [Ocean Orchestrator VS Code extension](https://marketplace.visualstudio.com/items?itemName=OceanProtocol.ocean-orchestrator). Open the tutorial directory in the orchestrator, configure the job, and run — the container builds and executes autonomously on remote GPU nodes.
