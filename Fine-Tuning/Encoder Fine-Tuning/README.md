# The Encoder-Only Transformer

Encoder-only transformer models are neural networks designed to learn rich, contextual representations of input text rather than to generate new text autoregressively. Trained on large corpora using objectives such as masked language modeling, they learn to understand the relationships between words by attending to all tokens in a sequence simultaneously through bidirectional self-attention. Modern encoder-only models are typically built on the transformer encoder architecture, which enables deep contextual encoding of entire sequences and effective modeling of long-range dependencies.


## Architecture

![Decoder-Only Architecture](<./images/Encoder.png>)

The model begins by converting input tokens into continuous vectors through a token embedding layer. To encode word order, a positional embedding is added to these token representations. The resulting vectors are then passed into a stack of identical transformer encoder blocks (repeated N times).

Each encoder block consists of two main sublayers:

<strong>Multi-Head Self-Attention</strong><br>
This layer allows each token to attend to all other tokens in the sequence (i.e., bidirectional attention), enabling the model to build rich contextual representations. Unlike the decoder, no causal masking is applied, since the encoder is not restricted to autoregressive generation. Multiple attention heads operate in parallel to capture different types of relationships and dependencies across the sequence.

<strong>Feed-Forward Network (FFN)</strong><br>
A position-wise fully connected network that processes each token representation independently after the attention mechanism has integrated contextual information.

Both sublayers are wrapped with residual connections (skip connections) and followed by layer normalization, which stabilizes training and improves gradient flow. After passing through the stacked encoder blocks, the final hidden states serve as contextualized representations of the input sequence. 

Depending on the task, these representations can be used directly (e.g., via a special classification token such as [CLS]) or passed to a task-specific output layer for objectives such as masked language modeling, classification, or token-level prediction.

## Training Objectives

Unlike decoder-only transformers, which are almost always trained with next-token prediction, encoder-only transformers support **multiple training paradigms** depending on how the encoder backbone is used. Below are two important and commonly used variants.

### 1. Encoder Backbone with Classification Head(s)

An encoder-only transformer can be viewed as a **general-purpose feature extractor** that produces contextualized token representations. On top of this shared backbone, different **task-specific heads** can be attached.

Given an input sequence
$
x = (x_1, \dots, x_T),
$
the encoder produces hidden states:

$
H = (h_1, \dots, h_T).
$

These representations can then be fed into one or more classification heads, depending on the task.

#### (a) Sequence-Level Classification

A special token (e.g., [CLS]) is prepended to the sequence. Its final hidden representation $ h_{\text{CLS}} $ serves as an aggregate summary of the input.  This representation can then be fed into a classification head. Alternatively, instead of relying solely on the `[CLS]` token, one may construct the input to the classification head using a learned linear combination (or pooling) of all token representations in the final layer.


A classification head (typically a linear layer + softmax) predicts:

$
P_\theta(y \mid x)
$

The loss is standard cross-entropy:

$
\mathcal{L} = - \log P_\theta(y \mid x)
$

This setup is used for tasks such as sentiment analysis, topic classification, or natural language inference.


#### (b) Token-Level Classification

For tasks like named entity recognition or part-of-speech tagging, a classification head is applied **independently to each token representation**:

$
P_\theta(y_t \mid x)
$

The total loss becomes:

$
\mathcal{L} = - \sum_{t=1}^{T} \log P_\theta(y_t \mid x)
$

#### (c) Multi-Task Learning with Multiple Heads

A single encoder backbone can simultaneously support multiple objectives by attaching **multiple classification heads**, each with its own loss:

$
\mathcal{L} = \sum_i \lambda_i \mathcal{L}_i
$

where:

* $ \mathcal{L}_i $ is the loss for task $ i $,
* $ \lambda_i $ controls the relative importance of each task.

In this setting, the encoder learns shared representations that transfer across tasks, while each head specializes in a particular prediction problem.

### 2. Leveraging Pretrained Knowledge via Masked Answer Prediction

Another important training variant uses the encoder’s **bidirectional pretraining knowledge** and frames prediction as a masked reconstruction problem.

Instead of adding extra classificatiomn heads over the transformer backbone, we provide the full input—including the question and a partially masked answer—and train the model to predict the masked tokens.

#### Example

**Original sequence:**

> "Question: How many planets are in our solar system? Answer: 8 planets."

**Corrupted input:**

> "Question: How many planets are in our solar system? Answer: [MASK] planets."

The encoder processes the entire sequence simultaneously, attending to both the question and the unmasked parts of the answer. The objective is to predict:

$
P(x_t \mid x_{\setminus \mathcal{M}}), \quad t \in \mathcal{M}
$

The loss is computed only on the masked answer tokens:

$
\mathcal{L} = - \sum_{t \in \mathcal{M}} \log P_\theta(x_t \mid x_{\setminus \mathcal{M}})
$

In this formulation:

* The model conditions on the full context (question + visible answer tokens),
* It leverages pretrained bidirectional knowledge,
* It reframes answer prediction as a classification problem over masked positions.

### Key Conceptual Difference from Decoder Training

* **Decoder-only models** are inherently autoregressive and optimized for sequence generation.
* **Encoder-only models** are not generative by design; instead, they are trained to produce rich contextual representations.
* Tasks can be formulated flexibly as:

  * Sequence-level classification,
  * Token-level classification,
  * Multi-task learning,
  * Masked reconstruction of selected tokens (including masked answers).

This flexibility is one of the main strengths of encoder-based architectures: they provide a powerful, reusable backbone that can be adapted to many different supervised objectives without being constrained to next-token prediction.


## Dockerfile

```
FROM ubuntu:24.04

RUN apt-get update &&     apt-get install -y     python3-pip     python3-venv     && apt-get clean

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /tmp/
RUN pip install --upgrade pip wheel
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 
RUN pip install -r /tmp/requirements.txt
```

## Requirements

```
transformers>=5.2.0
lightning>=2.6.1
datasets>=4.5.0
evaluate>=0.4.6
bitsandbytes>=0.49.0
peft>=0.18.0
wandb>=0.25.0
scikit-learn>=1.8.0
```