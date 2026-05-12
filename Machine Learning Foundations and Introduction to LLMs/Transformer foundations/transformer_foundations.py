import argparse
import sys
import textwrap

# Check that packages are importable

def require(packages: list[str]):
    missing = []
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print(f"     Install with:  pip install {' '.join(missing)}\n")
        sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def section(title: str):
    width = 70
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def subsection(title: str):
    print(f"\n  ── {title} {'─' * (60 - len(title))}")

# Simple ASCII table printer.
def show_table(headers: list, rows: list, col_width: int = 20):
    fmt = "  " + "".join(f"{{:<{col_width}}}" for _ in headers)
    print(fmt.format(*headers))
    print("  " + "-" * (col_width * len(headers)))
    for row in rows:
        print(fmt.format(*[str(c)[:col_width-1] for c in row]))


# ════════════════════════════════════════════════════════════════════════════
# TOKENIZATION BEHAVIOR
# ════════════════════════════════════════════════════════════════════════════

def lab1_tokenization():
    require(["transformers", "tokenizers"])

    from transformers import (
        AutoTokenizer,
        BertTokenizer,
        GPT2Tokenizer,
        T5Tokenizer,
    )

    section("1 — TOKENIZATION BEHAVIOR")
    print(textwrap.dedent("""
      We compare three major subword tokenization schemes:
        • BPE (Byte-Pair Encoding)       → used by GPT-2, GPT-3, LLaMA
        • WordPiece                       → used by BERT, DistilBERT
        • SentencePiece (Unigram / BPE)  → used by T5, ALBERT, LLaMA
    """))

    # ── 1.1  Load Tokenizers ─────────────────────────────────────────────────
    subsection("1.1  Loading tokenizers")
    print("  Downloading tokenizer configs (first run only)…")

    tok_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
    tok_bert = BertTokenizer.from_pretrained("bert-base-uncased")
    tok_t5   = T5Tokenizer.from_pretrained("t5-small", legacy=False)

    tokenizers = {
        "GPT-2 (BPE)":       tok_gpt2,
        "BERT (WordPiece)":  tok_bert,
        "T5 (SentencePiece)": tok_t5,
    }

    print("  ✓ All tokenizers loaded.\n")
    show_table(
        ["Tokenizer", "Vocab size", "Algorithm", "Continuation marker"],
        [
            ["GPT-2 (BPE)", tok_gpt2.vocab_size, "BPE",          "Ġ (space prefix)"],
            ["BERT (WP)",   tok_bert.vocab_size,  "WordPiece",    "## (suffix prefix)"],
            ["T5 (SP)",     tok_t5.vocab_size,    "SentencePiece","▁ (space prefix)"],
        ],
        col_width=22,
    )

    # ── 1.2  Tokenize sample sentences ──────────────────────────────────────
    subsection("1.2  Comparing tokenization of identical sentences")

    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Transformers revolutionized natural language processing.",
        "Unbelievably, the tokenizer handles unknown words: supercalifragilistic!",
        "2024-04-14: GPT-4 scored 90th percentile on the BAR exam.",
    ]

    for sent in sentences:
        print(f"\n  Input: {sent!r}")
        print()
        for name, tok in tokenizers.items():
            tokens = tok.tokenize(sent)
            ids    = tok.encode(sent)
            print(f"    {name:<25} → [{len(tokens):>3} tokens]  {tokens}")
        print()

    # ── 1.3  Special tokens ──────────────────────────────────────────────────
    subsection("1.3  Special tokens")

    for name, tok in tokenizers.items():
        sp = tok.all_special_tokens
        print(f"  {name:<25}: {sp}")

    # ── 1.4  Vocabulary inspection ───────────────────────────────────────────
    subsection("1.4  Vocabulary samples (first 20 and around index 5000)")

    for name, tok in tokenizers.items():
        vocab = tok.get_vocab()
        # Sort by id
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        sample_low  = [(t, i) for t, i in sorted_vocab[:20]]
        sample_mid  = [(t, i) for t, i in sorted_vocab[5000:5010]]
        print(f"\n  {name}")
        print(f"    First 20 tokens:       {[t for t, _ in sample_low]}")
        print(f"    Tokens around 5000:    {[t for t, _ in sample_mid]}")

    # ── 1.5  Subword splitting ──────────────────────────────────────────
    subsection("1.5  How each tokenizer handles rare / invented words")

    rare_words = [
        "unhappiness", "pneumonoultramicroscopicsilicovolcanoconiosis",
        "ChatGPT", "BERT", "Ġ", "##ing"
    ]

    print()
    for word in rare_words:
        print(f"  Word: {word!r}")
        for name, tok in tokenizers.items():
            tokens = tok.tokenize(word)
            print(f"    {name:<25}: {tokens}")
        print()

    # ── 1.6  Encoding / decoding ──────────────────────────────────
    subsection("1.6  Encode → decode")

    test = "The model learned representations of language."
    for name, tok in tokenizers.items():
        ids     = tok.encode(test)
        decoded = tok.decode(ids, skip_special_tokens=True)
        match   = "✓" if decoded.strip().lower() == test.strip().lower() else "✗"
        print(f"  {name:<25}: {match}  IDs: {ids[:8]}…")

    # ── 1.7  Token length statistics ─────────────────────────────────────────
    subsection("1.7  Token count comparison on a paragraph")

    paragraph = """
    The transformer architecture was introduced in the landmark paper
    "Attention Is All You Need" by Vaswani et al. in 2017. It replaced
    recurrent neural networks with a purely attention-based approach,
    enabling much greater parallelism during training. Within two years,
    BERT and GPT-2 had demonstrated that pretraining large transformers
    on vast text corpora produced powerful representations that could be
    fine-tuned with minimal effort for a wide range of downstream tasks.
    """

    print(f"\n  Paragraph character count: {len(paragraph)}")
    print()
    for name, tok in tokenizers.items():
        ids = tok.encode(paragraph)
        print(f"  {name:<25}: {len(ids):>4} tokens")


# ════════════════════════════════════════════════════════════════════════════
# ATTENTION PATTERN EXPLORATION
# ════════════════════════════════════════════════════════════════════════════

def lab2_attention():
    require(["transformers", "torch"])

    import torch
    import numpy as np
    from transformers import BertTokenizer, BertModel

    section("ATTENTION PATTERN EXPLORATION")
    print(textwrap.dedent("""
      We'll use BERT-base-uncased (12 layers × 12 heads).
      For each attention head we extract a [seq_len × seq_len] weight
      matrix showing how strongly each token attends to every other token.
    """))

    # ── 2.1  Load model ──────────────────────────────────────────────────────
    subsection("2.1  Loading BERT-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model     = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
    model.eval()
    print("  Model loaded.")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Attention heads: {model.config.num_attention_heads}")
    print(f"  Hidden size: {model.config.hidden_size}")

    def get_attentions(text: str):
        """Tokenize text and run BERT forward pass, returning tokens + attentions."""
        inputs  = tokenizer(text, return_tensors="pt")
        tokens  = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        with torch.no_grad():
            outputs = model(**inputs)
        # attentions: tuple of (num_layers,) each shape [1, num_heads, seq, seq]
        attentions = [a.squeeze(0).numpy() for a in outputs.attentions]
        return tokens, attentions

    # ── 2.2  Inspect a simple sentence ──────────────────────────────────────
    subsection("2.2  Attention patterns for a simple sentence")

    sentence = "The cat sat on the mat near the door."
    tokens, attentions = get_attentions(sentence)

    print(f"\n  Sentence: {sentence!r}")
    print(f"  Tokens:   {tokens}")
    print(f"  Shape per layer: {attentions[0].shape}  (heads × seq × seq)")

    def top_attended(attn_matrix, tokens, from_token_idx: int, top_k: int = 3):
        """
        For a given token (by index), return the top-k tokens it attends to
        averaged across all heads in the given layer.
        """
        # attn_matrix: [num_heads, seq, seq]
        avg_attn = attn_matrix.mean(axis=0)  # [seq, seq]
        weights  = avg_attn[from_token_idx]
        top_idxs = np.argsort(weights)[::-1][:top_k]
        return [(tokens[i], round(float(weights[i]), 4)) for i in top_idxs]

    # Inspect "cat" across layers
    cat_idx = tokens.index("cat")
    print(f"\n  Token 'cat' (index {cat_idx}) — top-3 attended tokens per layer (avg across heads):")
    print()
    for layer_idx in [0, 1, 5, 10, 11]:
        top = top_attended(attentions[layer_idx], tokens, cat_idx)
        print(f"  Layer {layer_idx:>2}: {top}")

    # ── 2.3  Head-level analysis ─────────────────────────────────────────────
    subsection("2.3  Individual head analysis (Layer 4)")

    layer = 4
    print(f"\n  Layer {layer} — each head's top attention from 'cat':")
    print()
    for head in range(model.config.num_attention_heads):
        head_attn = attentions[layer][head]  # [seq, seq]
        weights   = head_attn[cat_idx]
        top_idxs  = np.argsort(weights)[::-1][:2]
        top_pairs = [(tokens[i], round(float(weights[i]), 3)) for i in top_idxs]
        print(f"    Head {head:>2}: {top_pairs}")

    # ── 2.4  Attention entropy ───────────────────────────────────────────────
    subsection("2.4  Attention entropy per head (Layer 0 vs Layer 11)")
    print("""
  Low entropy  → head focuses sharply on a few tokens (syntactic heads)
  High entropy → head spreads attention broadly (semantic / contextual)
    """)

    from scipy.stats import entropy as scipy_entropy

    for layer_idx in [0, 11]:
        print(f"  Layer {layer_idx}:")
        for head in range(model.config.num_attention_heads):
            # Average entropy across all query positions
            attn  = attentions[layer_idx][head]  # [seq, seq]
            entrs = [scipy_entropy(attn[i] + 1e-9) for i in range(len(tokens))]
            avg_e = np.mean(entrs)
            bar   = "█" * int(avg_e * 4)
            print(f"    Head {head:>2}: entropy={avg_e:.3f}  {bar}")
        print()

    # ── 2.5  CLS token attention ─────────────────────────────────────────────
    subsection("2.5  [CLS] token attention in last layer")
    print("""
  [CLS] aggregates information from the whole sentence.
  In the final layer, its attention distribution reveals which tokens
  the model considers most salient for classification tasks.
    """)

    cls_idx  = 0
    layer_11 = attentions[11]
    avg_attn = layer_11.mean(axis=0)   # average over heads
    cls_attn = avg_attn[cls_idx]       # [seq]
    ranked   = sorted(zip(tokens, cls_attn), key=lambda x: -x[1])
    print(f"  Sentence: {sentence!r}\n")
    print("  Token          CLS attention weight   Bar")
    print("  " + "-" * 50)
    for tok, weight in ranked:
        bar = "▓" * int(weight * 100)
        print(f"  {tok:<15} {weight:.4f}                {bar}")

    # ── 2.6  Multi-sentence: coreference-style attention ────────────────────
    subsection("2.6  Coreference-style attention across a pair of sentences")

    pair = "[CLS] The scientist conducted the experiment. [SEP] She published the results. [SEP]"
    pair_tokens, pair_attns = get_attentions(
        "The scientist conducted the experiment. She published the results."
    )

    print(f"\n  Tokens: {pair_tokens}")
    she_idx = pair_tokens.index("she") if "she" in pair_tokens else -1
    if she_idx >= 0:
        print(f"\n  'she' (index {she_idx}) attends to (Layer 10, all heads avg):")
        top = top_attended(pair_attns[10], pair_tokens, she_idx, top_k=5)
        for tok, w in top:
            bar = "▓" * int(w * 100)
            print(f"    {tok:<15} {w:.4f}  {bar}")
    else:
        print("  'she' token not found — checking tokenization…")
        print(f"  Tokens: {pair_tokens}")

    # ── 2.7  BertViz visualization ──────────────────────────────────
    subsection("2.7  BertViz visualization")

    from bertviz import head_view, model_view
    from transformers import BertTokenizer, BertModel

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model     = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)

    text    = "The cat sat on the mat."
    inputs  = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    tokens  = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    head_html = head_view(outputs.attentions, tokens, html_action='return')    # per-head attention view
    model_html = model_view(outputs.attentions, tokens, html_action='return')   # all layers / heads overview

    with open("per-head_attention.html", "w") as f:
        f.write(head_html.data)
        print("Open per-head_attention.html for per-head attention view")
    with open("all-layers_heads_overview.html", "w") as f:
        f.write(model_html.data)
        print("Open all-layers_heads_overview.html for all layers / heads overview")


# ════════════════════════════════════════════════════════════════════════════
#  EMBEDDING SPACE EXPLORATION
# ════════════════════════════════════════════════════════════════════════════

def lab3_embeddings():
    """
    Extract and analyze embeddings from BERT:
      - Static token embeddings vs contextual embeddings
      - Cosine similarity matrix
      - PCA / dimensionality reduction for cluster visualization
      - Polysemy: same word, different contexts → different embeddings
    """
    require(["transformers", "torch", "numpy", "sklearn"])

    import torch
    import numpy as np
    from transformers import BertTokenizer, BertModel
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity

    section("EMBEDDING SPACE EXPLORATION")
    print(textwrap.dedent("""
      BERT produces *contextual* embeddings: the same word gets a different
      vector depending on its surrounding context. We'll:
        1. Compare static (token) embeddings vs contextual (last-layer) embeddings
        2. Measure similarity between word pairs
        3. Demonstrate polysemy via embeddings
        4. Visualize embedding clusters with PCA
    """))

    # ── Load model ────────────────────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model     = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
    model.eval()

    def get_embeddings(text: str, layer: int = -1):
        """
        Returns:
          tokens    : list of token strings
          static_emb: numpy [seq, hidden] — token embedding table lookup
          ctx_emb   : numpy [seq, hidden] — hidden state from `layer`
        """
        inputs     = tokenizer(text, return_tensors="pt")
        token_ids  = inputs["input_ids"][0]
        tokens     = tokenizer.convert_ids_to_tokens(token_ids)

        with torch.no_grad():
            outputs = model(**inputs)

        # Static embeddings: lookup the embedding table directly
        static_emb = model.embeddings.word_embeddings(token_ids).detach().numpy()

        # Contextual embeddings: hidden states from specified layer
        # hidden_states: tuple of (num_layers+1,) each [1, seq, hidden]
        hidden_states = outputs.hidden_states
        ctx_emb = hidden_states[layer][0].numpy()  # [seq, hidden]

        return tokens, static_emb, ctx_emb

    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        a = a / (np.linalg.norm(a) + 1e-9)
        b = b / (np.linalg.norm(b) + 1e-9)
        return float(np.dot(a, b))

    # ── 3.1  Static vs contextual embeddings ─────────────────────────────────
    subsection("3.1  Static vs Contextual Embeddings")

    sentences = [
        "The bank is by the river.",
        "I deposited money at the bank.",
        "He sat on the river bank watching fish.",
        "The central bank raised interest rates.",
    ]

    print("\n  Word 'bank' — embedding across different contexts:\n")
    bank_static_embs = []
    bank_ctx_embs    = []

    for sent in sentences:
        tokens, static, ctx = get_embeddings(sent)
        # Find 'bank' token (BERT lowercases)
        bank_idx = next((i for i, t in enumerate(tokens) if t == "bank"), None)
        if bank_idx is not None:
            bank_static_embs.append(static[bank_idx])
            bank_ctx_embs.append(ctx[bank_idx])
            norm_s = np.linalg.norm(static[bank_idx])
            norm_c = np.linalg.norm(ctx[bank_idx])
            print(f"  Sentence: {sent}")
            print(f"    Static emb norm:      {norm_s:.3f}")
            print(f"    Contextual emb norm:  {norm_c:.3f}\n")

    if len(bank_ctx_embs) >= 2:
        print("  Cosine similarities between 'bank' contextual embeddings:")
        for i in range(len(sentences)):
            for j in range(i+1, len(sentences)):
                sim_s = cosine_sim(bank_static_embs[i], bank_static_embs[j])
                sim_c = cosine_sim(bank_ctx_embs[i], bank_ctx_embs[j])
                label_i = "river bank" if "river" in sentences[i] else "financial bank"
                label_j = "river bank" if "river" in sentences[j] else "financial bank"
                print(f"    [{label_i:<16}] vs [{label_j:<16}]  "
                      f"static={sim_s:.3f}  contextual={sim_c:.3f}")

    # ── 3.2  Semantic similarity matrix ──────────────────────────────────────
    subsection("3.2  Semantic similarity between sentences")

    sentences2 = [
        "A dog is playing fetch in the park.",
        "The puppy chases a ball outside.",
        "The stock market fell sharply today.",
        "Equity markets experienced a significant decline.",
        "Machine learning models require large datasets.",
        "AI systems need substantial amounts of training data.",
    ]

    print("\n  Computing [CLS] embeddings for 6 sentences…\n")

    cls_embs = []
    for sent in sentences2:
        tokens, _, ctx = get_embeddings(sent, layer=-1)
        cls_embs.append(ctx[0])  # [CLS] is always index 0

    cls_matrix = np.stack(cls_embs)
    sim_matrix  = cosine_similarity(cls_matrix)

    # Print similarity matrix
    labels = [
        "Dog playing",
        "Puppy chases",
        "Market fell",
        "Equity decline",
        "ML datasets",
        "AI training",
    ]

    col_w = 14
    header = f"  {'':14}" + "".join(f"{l:<{col_w}}" for l in labels)
    print(header)
    print("  " + "─" * (14 + col_w * len(labels)))
    for i, label in enumerate(labels):
        row = f"  {label:<14}"
        for j in range(len(labels)):
            val = sim_matrix[i][j]
            # Color-coded with ASCII
            if i == j:
                cell = f"{'1.000':<{col_w}}"
            elif val > 0.9:
                cell = f"{'★'+f'{val:.3f}':<{col_w}}"
            elif val > 0.8:
                cell = f"{'◆'+f'{val:.3f}':<{col_w}}"
            else:
                cell = f"{val:.3f}{'':<{col_w-5}}"
            row += cell
        print(row)

    print("\n  ★ = very similar (>0.9)   ◆ = similar (>0.8)")

    # ── 3.3  Layer-by-layer embedding evolution ───────────────────────────────
    subsection("3.3  How contextual embeddings evolve across layers")
    print("""
  BERT has 12 transformer layers (+ 1 embedding layer = 13 total).
  Lower layers capture syntax; higher layers capture semantics.
  We track how the embedding of a single token changes across layers.
    """)

    target_sent = "The lawyer argued the case with great passion."
    tokens, static, _ = get_embeddings(target_sent, layer=0)
    lawyer_idx = next(i for i, t in enumerate(tokens) if t == "lawyer")

    inputs = tokenizer(target_sent, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    all_hidden = [h[0].numpy() for h in outputs.hidden_states]  # 13 × [seq, 768]

    print(f"  Sentence: {target_sent!r}")
    print(f"  Tracking token: '{tokens[lawyer_idx]}' (index {lawyer_idx})\n")

    prev_emb = None
    for layer_idx, hidden in enumerate(all_hidden):
        emb = hidden[lawyer_idx]
        layer_name = "Embedding" if layer_idx == 0 else f"Layer {layer_idx:>2}"
        norm   = np.linalg.norm(emb)
        if prev_emb is not None:
            delta = np.linalg.norm(emb - prev_emb)
            sim   = cosine_sim(emb, prev_emb)
            print(f"  {layer_name}: norm={norm:.2f}  Δ from prev={delta:.3f}  cos_sim_prev={sim:.4f}")
        else:
            print(f"  {layer_name}: norm={norm:.2f}  (base embedding)")
        prev_emb = emb

    # ── 3.4  PCA visualization (text output) ─────────────────────────────────
    subsection("3.4  PCA of word embeddings — thematic clusters")
    print("""
  We project 768-dim embeddings down to 2D using PCA and observe
  how semantically related words cluster together.
    """)

    word_groups = {
        "Animals":   ["cat", "dog", "horse", "bird", "fish"],
        "Countries": ["france", "germany", "japan", "brazil", "canada"],
        "Emotions":  ["happy", "sad", "angry", "afraid", "excited"],
        "Science":   ["atom", "electron", "molecule", "physics", "chemistry"],
    }

    words_flat  = []
    labels_flat = []
    embs_flat   = []

    for group, words in word_groups.items():
        for word in words:
            tokens_w, static_w, _ = get_embeddings(word)
            # Use the first non-[CLS] token embedding (static)
            word_idx = 1  # token after [CLS]
            words_flat.append(word)
            labels_flat.append(group)
            embs_flat.append(static_w[word_idx])

    embs_array = np.stack(embs_flat)
    pca        = PCA(n_components=2)
    coords_2d  = pca.fit_transform(embs_array)

    print(f"  PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}  "
          f"PC2={pca.explained_variance_ratio_[1]:.1%}\n")

    def plot_pca_embeddings(coords_2d, words_flat, labels_flat):
        import matplotlib.pyplot as plt

        colors = {"Animals": "#e74c3c", "Countries": "#3498db",
                    "Emotions": "#2ecc71", "Science":  "#9b59b6"}

        fig, ax = plt.subplots(figsize=(10, 8))
        for group, color in colors.items():
            idxs = [i for i, l in enumerate(labels_flat) if l == group]
            xs   = [coords_2d[i, 0] for i in idxs]
            ys   = [coords_2d[i, 1] for i in idxs]
            ax.scatter(xs, ys, c=color, label=group, s=100, alpha=0.8)
            for i in idxs:
                ax.annotate(words_flat[i],
                            (coords_2d[i, 0], coords_2d[i, 1]),
                            fontsize=9, ha="right")

        ax.set_title("BERT Static Embeddings — PCA Projection")
        ax.legend()
        plt.tight_layout()
        plt.savefig("embedding_clusters.png", dpi=150)
        plt.show()
        print("Saved to embedding_clusters.png")
    
    plot_pca_embeddings(coords_2d, words_flat, labels_flat)

    # ── 3.5  Analogical reasoning (king - man + woman = queen) ───────────────
    subsection("3.5  Word analogies with static embeddings")
    print("""
  The classic word2vec result: king - man + woman ≈ queen
  Does BERT's static embedding layer reproduce this?
    """)

    def get_word_embedding(word: str):
        """Get static embedding for a single word (no context)."""
        _, static, _ = get_embeddings(word)
        return static[1]  # index 1 = word token (skip [CLS])

    analogies = [
        ("king", "man", "woman", "queen"),
        ("paris", "france", "germany", "berlin"),
        ("doctor", "man", "woman", "nurse"),
    ]

    vocab_sample_words = (
        list(word_groups["Countries"]) +
        list(word_groups["Animals"]) +
        ["king", "queen", "prince", "princess", "man", "woman",
         "doctor", "nurse", "teacher", "engineer",
         "paris", "london", "berlin", "tokyo", "madrid",
         "france", "england", "japan", "brazil"]
    )

    print("  Building mini-vocabulary for nearest-neighbor search…")
    vocab_embs = {}
    for w in vocab_sample_words:
        try:
            vocab_embs[w] = get_word_embedding(w)
        except Exception:
            pass

    def find_nearest(target_emb, vocab_embs, exclude=None, top_k=3):
        exclude = set(exclude or [])
        sims = {
            w: cosine_sim(target_emb, emb)
            for w, emb in vocab_embs.items()
            if w not in exclude
        }
        return sorted(sims.items(), key=lambda x: -x[1])[:top_k]

    print()
    for a, b, c, expected in analogies:
        try:
            emb_a = get_word_embedding(a)
            emb_b = get_word_embedding(b)
            emb_c = get_word_embedding(c)
            target = emb_a - emb_b + emb_c
            nearest = find_nearest(target, vocab_embs, exclude=[a, b, c])
            nearest_words = [w for w, _ in nearest]
            found = "✓" if expected in nearest_words else "✗"
            print(f"  {a} - {b} + {c} = ?  Expected: {expected}")
            print(f"    Nearest: {nearest}  {found}")
        except Exception as e:
            print(f"  Skipped {a}-{b}+{c}: {e}")

    print("""
  Note: BERT's token embeddings are trained for MLM, not analogical
  reasoning. Word2Vec / GloVe often outperform BERT static embeddings
  on analogy tasks, but BERT's *contextual* embeddings are far superior
  for most NLP applications.
    """)

# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Transformer Foundations Tutorial")
    parser.add_argument(
        "--lab", type=int, choices=[1, 2, 3],
        help="Run a specific lab (1=Tokenization, 2=Attention, 3=Embeddings)"
    )
    args = parser.parse_args()

    print("\n" + "█" * 70)
    print("  TRANSFORMER FOUNDATIONS  ")
    print("█" * 70)
    print("""
  Labs:
    1 → Tokenization Behavior     (BPE / WordPiece / SentencePiece)
    2 → Attention Pattern Exploration  (BERT attention heads)
    3 → Embedding Space Analysis   (static vs contextual, PCA)
    """)

    lab_map = {
        1: lab1_tokenization,
        2: lab2_attention,
        3: lab3_embeddings,
    }

    if args.lab is not None:
        lab_map[args.lab]()
    else:
        # Run all
        lab1_tokenization()
        lab2_attention()
        lab3_embeddings()


if __name__ == "__main__":
    main()
