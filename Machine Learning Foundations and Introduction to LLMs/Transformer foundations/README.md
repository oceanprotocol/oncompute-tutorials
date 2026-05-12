# 🤖 Transformer Foundations

The **Transformer** architecture (Vaswani et al., 2017 — *"Attention Is All You Need"*) is the foundation of  every modern large language model (LLM). 

```
Input Tokens → [Embeddings] → [Encoder / Decoder Blocks] → Output
                                        ↑
                              Multi-Head Self-Attention
                              Feed-Forward Network
                              Layer Norm + Residuals
```

## Attention & Self-Attention

Traditional sequence models (RNNs, LSTMs) process tokens *one at a time*, making it hard to relate distant tokens. **Attention** lets every token directly query every other token simultaneously.

> *"Attention is a soft, differentiable lookup table."*

### 1.1 The QKV Formulation

For each token, we compute three vectors from learned weight matrices:

| Vector | Symbol | Role |
|--------|--------|------|
| **Query** | Q | "What am I looking for?" |
| **Key** | K | "What do I contain?" |
| **Value** | V | "What do I contribute?" |

**Scaled Dot-Product Attention:**

```
            QKᵀ
Attention = ——— softmax → × V
            √dₖ
```

Formally:

```
Attention(Q, K, V) = softmax( QKᵀ / √dₖ ) · V
```

The `√dₖ` scaling prevents the dot products from growing too large in high-dimensional spaces, which would push softmax into near-zero gradient regions.

### 1.2 Self-Attention

In **self-attention**, Q, K, and V all come from the *same* sequence. Each token attends to all others (including itself) within the same layer.

```
"The cat sat on the mat"
   ↑
Token "cat" has high attention weights toward "The" and "sat",
capturing subject-verb-noun relationships.
```

### 1.3 Multi-Head Attention

Running a single attention function limits what can be captured in one pass. **Multi-head attention** runs `h` attention operations in parallel with different learned projections, then concatenates the results:

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) · Wᴼ

where headᵢ = Attention(Q·Wᵢᴼ, K·Wᵢᴷ, V·Wᵢᵛ)
```

Each head can specialize — one might capture syntax, another semantics, another co-reference.

### 1.4 Positional Encoding

Attention is *permutation-invariant* by default — it doesn't know token order. **Positional encodings** inject position information:

**Sinusoidal (original paper):**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/dmodel))
PE(pos, 2i+1) = cos(pos / 10000^(2i/dmodel))
```

**Modern alternatives:**
- **Learned absolute PE** — trainable embeddings per position (BERT, GPT-2)
- **RoPE (Rotary PE)** — encodes relative positions via rotation matrices (LLaMA, GPT-NeoX)
- **ALiBi** — adds a linear bias to attention scores based on distance (MPT, BLOOM)

### 1.5 Masked (Causal) Attention

Decoder-only models use a **causal mask** — each token can only attend to past tokens, preventing information leakage from the future during training:

```
Mask:
Token 0  [1, 0, 0, 0]
Token 1  [1, 1, 0, 0]
Token 2  [1, 1, 1, 0]
Token 3  [1, 1, 1, 1]
```

### 1.6 Cross-Attention

In encoder-decoder models, the decoder uses **cross-attention**: Q comes from the decoder, while K and V come from the encoder's output. This is how translation models "look at" the source sentence while generating each target word.

## Architectures

```
┌─────────────────────────────────────────────────────────────┐
│  ENCODER-ONLY          DECODER-ONLY       ENCODER-DECODER   │
│                                                             │
│  [Enc][Enc][Enc]        [Dec][Dec][Dec]   [Enc] → [Dec]    │
│       ↓                      ↓                  ↓           │
│  Bidirectional           Causal/AR         Seq2Seq          │
│  Representation          Generation        Transduction     │
│                                                             │
│  BERT, RoBERTa           GPT family        T5, BART         │
│  DeBERTa, ELECTRA        LLaMA, Mistral    mBART, mT5       │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 Encoder-Only

**Architecture:** Stack of transformer blocks with full bidirectional self-attention where every token can attend to every other token, and the model sees the *whole* sequence at once.

**Best for:**
- Text classification
- Named entity recognition (NER)
- Semantic similarity / embeddings
- Question answering (extractive)
- Token classification

**Representative models:** BERT, RoBERTa, ALBERT, DeBERTa, ELECTRA

### 2.2 Decoder-Only

**Architecture:** Stack of transformer blocks with *causal* (masked) self-attention. Being autoregressive, it generates one token at a time, each conditioned on all previous tokens. The model never sees future tokens.

**Best for:**
- Text generation
- In-context learning / few-shot prompting
- Code generation
- Chat / instruction following

**Weakness:** Doesn't natively produce bidirectional representations for classification tasks (though instruction-tuning has blurred this line significantly).

**Representative models:** GPT-2, GPT-3/4, LLaMA 1/2/3, Mistral, Falcon, Phi, Gemma

### 2.3 Encoder-Decoder

**Architecture:** A full encoder stack plus a decoder stack. The decoder uses both causal self-attention (on its own outputs) and cross-attention (on the encoder's output). Designed for *sequence-to-sequence* tasks where input and output lengths may differ and come from different "spaces."

**Best for:**
- Machine translation
- Abstractive summarization
- Question answering (generative)
- Data-to-text
- Document understanding

**Representative models:** T5, BART, mT5, mBART, PEGASUS, Flan-T5

### 2.4 A Transformer Block (Shared Core)

Regardless of architecture family, each transformer *block* contains:

```
Input
  │
  ▼
[Layer Norm]  ← Pre-LN (modern) or Post-LN (original)
  │
  ▼
[Multi-Head Self-Attention]
  │  + Residual connection
  ▼
[Layer Norm]
  │
  ▼
[Feed-Forward Network]   → Linear(dmodel → 4·dmodel) → GELU → Linear(4·dmodel → dmodel)
  │  + Residual connection
  ▼
Output
```

The **FFN** acts as key-value memory, with neurons storing factual associations learned during pretraining.

## Tokenization

### 3.1 Why Tokenization Matters

Transformers don't see characters or words directly, but rather they operate on **tokens**: subword units that balance vocabulary size, coverage, and sequence length. The main trade-offs are:
- **Large vocab** → shorter sequences, more parameters in embedding layer
- **Small vocab** → longer sequences, more compute per sequence
- **Character-level** → handles any word, but very long sequences
- **Word-level** → natural units, but OOV (Out-of-vocabulary) problem may appear, where words that a model has never encountered during training appear during real-world usage.

Subword tokenization adresses these problems.

### 3.2 Byte-Pair Encoding (BPE)

**Used by:** GPT-2, GPT-3, GPT-4, LLaMA, RoBERTa, BART

**Algorithm:**
1. Start with a character-level vocabulary and a pre-tokanized text (splitted into words)
2. Count all adjacent symbol pairs in the corpus
3. Merge the most frequent pair into a new symbol
4. Repeat until target vocabulary size is reached

```
Corpus: "low lower lowest"

Initial:  l o w, l o w e r, l o w e s t  [pre-tokanized]
Step 1:   merge (l,o) → "lo"   [most frequent]
          lo w, lo w e r, lo w e s t
Step 2:   merge (lo,w) → "low"
          low, low e r, low e s t
...
Final:    low, lower, low##est   (vocabulary learned from data)
```

**Result:** Frequent words get their own token. Rare words are split into known subwords. Unknown words are handled via characters.

### 3.3 WordPiece

**Used by:** BERT, DistilBERT, ELECTRA, MobileBERT

**Similar to BPE but with a different merge criterion:**

Instead of frequency, WordPiece maximizes the *likelihood* of the training data when merging:

```
score(A, B) = freq(AB) / (freq(A) × freq(B))
```

This prefers merges that are "surprising", like pairs that occur much more than chance.
Also, WordPiece uses `##` to mark continuation tokens:
```
"playing" → ["play", "##ing"]
"unbelievable" → ["un", "##believe", "##able"]
```
The ## prefix tells a model, such as BERT, that "##ing" is part of the same word as "play", not an independent token.

### 3.4 SentencePiece

**Used by:** T5, ALBERT, XLNet, mT5, LLaMA (with BPE variant)

**Differences from BPE/WordPiece:**
- Treats the input as a raw *byte stream* with no pre-tokenization on whitespace (splitting into words). This makes the tokenizer work for languages that do not have whitespaces between words, such as Chinese, Japanese, Arabic, etc. 
- Whitespace is treated as a regular character (represented as `▁`)

```
"Hello world" → ["▁Hello", "▁world"]
"Helloworld"  → ["▁Hello", "world"]  ← different tokenization!
```

SentencePiece can use BPE as its underlying algorithm.

### 3.5 Special Tokens

Most tokenizers add **special tokens** for architectural purposes:

| Token | Models | Purpose |
|-------|--------|---------|
| `[CLS]` | BERT | Classification representation (first token) |
| `[SEP]` | BERT | Sentence separator |
| `[MASK]` | BERT | Masked token for MLM |
| `<s>` / `</s>` | RoBERTa, T5 | Sequence start/end |
| `<pad>` | All | Padding to batch sequences |
| `<unk>` | Many | Unknown token fallback |
| `<\|endoftext\|>` | GPT | End of sequence |


## Pretraining Objectives

### 4.1 Why Pretraining?

Modern LLMs are pretrained on massive unlabeled corpora (trillions of tokens) using *self-supervised* objectives where the labels come from the data itself. Pretraining builds general-purpose representations that can be fine-tuned for other tasks.

### 4.2 Masked Language Modeling (MLM)

**Used by:** BERT, RoBERTa, ALBERT, DistilBERT, DeBERTa

**Objective:** Predict the original identity of randomly masked tokens.

```
Input:  "The [MASK] sat on the [MASK]"
Target: "The  cat  sat on the  mat "
```

**BERT's masking strategy (15% of tokens):**
- 80% → replaced with `[MASK]`
- 10% → replaced with a random token
- 10% → kept unchanged

The 10%/10% split prevents the model from only learning to handle `[MASK]` tokens.  

**Advantage:** Bidirectional context — the model sees tokens on *both sides* of a mask, enabling rich representations.

**Disadvantage:** The `[MASK]` token never appears during fine-tuning, creating a *pretrain-finetune discrepancy*. To mitigate this, not all "masked" words are replaced my the actual [MASK] token, and the training data generator chooses around 15% of token positions at random for prediction, which makes the method less efficient than causal LM.

**Variant — Whole Word Masking:** Mask all subword tokens of a word together, not individually.

**Variant — Span Masking (T5):** Mask contiguous spans of tokens, replaced by a single sentinel token. More aggressive and suited for generative pretraining.

### 4.3 Next Sentence Prediction (NSP)

**Used by:** Original BERT (but later abandoned)

**Objective:** Given two segments A and B, predict whether B follows A in the original document.

RoBERTa showed NSP often downgrades performance and removed it. The task was too easy and the models learned superficial correlations rather than deep discourse understanding.

### 4.4 Causal Language Modeling (CLM)

**Used by:** GPT-2, GPT-3, GPT-4, LLaMA, Mistral, Falcon, and all modern decoder-only models

**Objective:** Predict the next token given all previous tokens.

```
Input:  "The cat sat on the"
Target: "cat sat on the mat"
         ↑   ↑   ↑   ↑   ↑
    Each token predicts the next
```

This is **standard language modeling** where the model maximizes:

```
L = Σ log P(xₜ | x₁, x₂, ..., xₜ₋₁)
```
where P represents the probability of generating the correct token xₜ given the previous tokens x₁, x₂, ..., xₜ₋₁.

**Advantage:**
- 100% of tokens are predicted
- No pretrain-finetune discrepancy
- Natural generation: just keep sampling the next token

**Disadvantage:** Unidirectional so only left context is used. Representations are less rich for classification tasks than bidirectional MLM models.

### 4.5 Prefix LM / T5's Span Corruption

**T5** uses a *span corruption* objective:

```
Original: "The cat sat on the mat in the garden"
Input:    "The cat <X> the mat <Y> garden"
Target:   "<X> sat on <Y> in the"
```

Spans are replaced by sentinel tokens `<X>`, `<Y>`, etc. The model must reconstruct the missing spans. This is a generative objective that works with an encoder-decoder architecture.

### 4.6 ELECTRA — Replaced Token Detection

**Used by:** ELECTRA, DeBERTa-v3

Instead of masking, a small **generator** model produces plausible replacements. The main **discriminator** model predicts whether each token was replaced.

```
Original:    "the chef cooked the meal"
Generator:   "the chef ate    the meal"   ← 'ate' replaced 'cooked'
Discriminator predicts: [O, O, R, O, O]   ← R = Replaced, O = Original
```

Despite its apparent similarity to a GAN framework, this approach is not adversarial. The token-corrupting generator is trained through maximum likelihood, due to the practical difficulties of using GANs for text generation.

**Advantage:** Every token is predicted (not just 15%), making it ~4× more compute-efficient than BERT. Excellent fine-tuning performance per FLOP.

## The Model Landscape

```
2017 ────────── 2019 ──────── 2020 ──────── 2022 ──────── 2023 ──────── 2024+
  │                │              │              │              │              │
Transformer      BERT           GPT-3         ChatGPT       LLaMA 2       LLaMA 3
(Vaswani)      RoBERTa         T5/BART       InstructGPT   Mistral 7B    Mixtral
               XLNet           ELECTRA        GPT-4         Phi-2         Gemma
               GPT-2           ALBERT         LLaMA 1       Falcon        Qwen
```

### 5.1 BERT Family (Encoder-Only)

#### BERT (2018) — Devlin et al., Google
It is the model which defined the pretrain and finetune paradigm for NLP. It contains 110M / 340M parameters (Base / Large) and it is trained on BookCorpus plus English Wikipedia (~16 GB) datasets. Training objectives are MLM and NSP. 
- Input example: `[CLS] Sentence A [SEP] Sentence B [SEP]`

#### RoBERTa (2019) — Facebook AI

RoBERTa uses the same architecture as BERT, but it was trained more effectively by removing the Next Sentence Prediction (NSP) objective, which was shown to be harmful, and training it on much larger batches and far more data (about 160 GB), and for significantly longer. It also introduced dynamic masking, meaning the masked tokens change at every epoch rather than remaining fixed. 

#### ALBERT (2020) — Google
ALBERT focuses on parameter efficiency through techniques such as factorized embedding parameterization and cross-layer parameter sharing, where the same weights are reused across all layers. It replaces BERT’s NSP objective with a new task called Sentence Order Prediction (SOP). Despite having far fewer parameters than BERT, ALBERT achieves competitive performance.

#### ELECTRA (2020) — Google
ELECTRA introduces a new pretraining objective called Replaced Token Detection, which is more compute-efficient than masked language modeling. It is considered the most compute-efficient encoder-only model in terms of FLOPs, and even the small version, ELECTRA-Small, achieves performance comparable to BERT-Base while requiring only a fraction of the computational cost.

#### DeBERTa (2021) — Microsoft
DeBERTa improves upon the Transformer architecture through disentangled attention, where content and position embeddings are handled separately for queries and keys. It also incorporates an enhanced mask decoder for masked language modeling. At the time of its release, DeBERTa achieved state-of-the-art performance on the SuperGLUE benchmark.

### 5.2 GPT Family (Decoder-Only)

#### GPT-2 (2019) — OpenAI
GPT-2 was released in four different sizes ranging from 117 million to 1.5 billion parameters and was trained on the 40-GB WebText corpus. It demonstrated surprisingly strong zero-shot performance across a variety of tasks. 

#### GPT-3 (2020) — OpenAI
GPT-3 scaled up to 175 billion parameters, making it around one hundred times larger than GPT-2. It introduced in-context learning, where tasks could be specified entirely through natural-language prompts, and it achieved few-shot performance that approached that of fine-tuned models. Unlike GPT-2, GPT-3 was not released openly and was accessible only through an API.

#### InstructGPT / ChatGPT (2022)
InstructGPT, and later ChatGPT, were created by fine-tuning GPT-3 using Reinforcement Learning from Human Feedback (RLHF). This approach made the model dramatically better at following instructions and aligned it to behave in a manner that is helpful, harmless, and honest.

#### GPT-4 (2023) — OpenAI
GPT-4 introduced multimodal capabilities, allowing it to process both images and text. It achieved state-of-the-art performance on numerous professional and academic benchmarks, such as the bar exam and medical licensing tests. However, OpenAI did not disclose full architectural or training details for this model.

#### LLaMA 1/2/3 (2023–2024) — Meta
The LLaMA family of models provided open-weight, commercially usable alternatives to proprietary LLMs. LLaMA 2 included models from 7B to 70B parameters trained on roughly two trillion tokens. The later LLaMA 3.1 release extended the range from 8B to 405B parameters and became highly competitive with GPT-4.

#### Mistral / Mixtral (2023–2024)
Mistral released several influential models beginning with Mistral 7B, which outperformed LLaMA 2 13B despite having fewer parameters. These models employed techniques such as Grouped Query Attention (GQA) and Sliding Window Attention for improved efficiency. The Mixtral 8×7B model adopted a Mixture-of-Experts architecture with eight expert feed-forward networks, of which only two are activated per token.

### 5.3 T5 Family (Encoder-Decoder)

#### T5 (2020) — Google
11B parameter model, trained on C4 (750 GB of clean CommonCrawl)
T5 (Text-To-Text Transfer Transformer) introduces a unified framework in which all NLP tasks are formulated as text-to-text problems. For example:
  ```
  Translation:    "translate English to German: The cat sat"
  Classification: "sst2 sentence: This movie was great"
  Summarization:  "summarize: The researchers found that..."
  ```

The model has 11 billion parameters and was trained on the C4 dataset, a cleaned version of Common Crawl totaling approximately 750 GB of text.

#### BART (2020) — Facebook AI
BART is an encoder-decoder model pretrained using denoising objectives. During training, the input text is deliberately corrupted in various ways, including token masking, deletion, infilling, permutation, and rotation, and the model learns to reconstruct the original sequence. This approach makes BART particularly effective for tasks such as summarization and machine translation.

#### Flan-T5 (2022) — Google
Flan-T5 builds upon T5 by being fine-tuned on a large collection of 1,836 NLP tasks using instruction-based and chain-of-thought prompting. This extensive instruction tuning gives the model strong zero-shot and few-shot capabilities. Additionally, Flan-T5 is released with open weights, making it widely accessible and commonly used in research.

### 5.4 Trends and Modern Successors

| Trend | Description | Examples |
|-------|-------------|---------|
| **Scaling** | Larger models, more data | GPT-4, Gemini Ultra, Claude 3 |
| **MoE** | Only activate a subset of parameters | Mixtral, Gemini 1.5 |
| **Long context** | 128K–1M+ token context windows | Gemini 1.5, LLaMA 3.1 |
| **Multimodality** | Vision, audio, code | GPT-4o, Gemini, LLaVA |
| **Small/efficient** | High capability at 1B–7B scale | Phi-3, Gemma 2, SmolLM2 |
| **RLHF / DPO** | Alignment with human preferences | All instruction-tuned models |
| **RoPE / GQA** | Better positional encoding, efficient attention | LLaMA, Mistral, Gemma |

## Tutorial

### Installation

```bash
# Create a virtual environment
python -m venv transformer-course
source transformer-course/bin/activate  # Windows: transformer-course\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.40.0
pip install tokenizers sentencepiece
pip install matplotlib seaborn numpy pandas scikit-learn
pip install bertviz
pip install protobuf
```

### Quick Start

```bash
# Run entirely (inside the virtual environment)
python transformer_foundations.py

# Or run individual sections
python transformer_foundations.py --lab 1   # Tokenization
python transformer_foundations.py --lab 2   # Attention
python transformer_foundations.py --lab 3   # Embeddings

# Attention visualizations are generated in the following html files:
per-head_attention.html
all-layers_heads_overview.html
```
