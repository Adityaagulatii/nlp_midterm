"""
transformer.py — Model 3 of 3: a small GPT-style Transformer language model.

WHAT THIS MODEL IS
------------------
A decoder-only transformer built from the same components as Lab 9's NLI
encoder — scaled dot-product self-attention, sinusoidal positional encoding,
multi-head attention, and a feed-forward sub-layer with residual connections
and layer norm — adapted for autoregressive character generation.

Two adaptations from the Lab 9 classifier:
  * Causal (look-ahead) mask instead of a padding mask: position t may attend
    only to positions <= t, so the model can be trained to predict the next
    character at every position without seeing the future.
  * A per-position output layer instead of a single CLS classifier: every
    position produces logits over the vocabulary for the character that
    follows it.

REQUIRED COMPONENTS (from the assignment), and where each lives:
  * Token embeddings ....... nn.Embedding in GPTLanguageModel.__init__
  * Positional information .. sinusoidal_positional_encoding()
  * Self-attention .......... scaled_dot_product_attention() + MultiHeadSelfAttention
  * Feedforward layer ....... the FFN inside TransformerBlock
  * Output layer ............ self.head (Linear -> vocab logits)

WHAT IT IMPROVES OVER THE LSTM
    Self-attention lets any position look *directly* at any earlier position
    in a single step, instead of routing information through a fixed-size
    hidden state updated one character at a time. There is no long-distance
    bottleneck, so long-range dependencies survive and the model's loss drops
    below the LSTM's.

This script trains the model and records train/val loss, training time, and
compute metrics (throughput in tokens/sec and peak GPU memory) to results.json.
"""

import json
import math
import os
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from data import build_dataset, get_batch

# ---------------------------------------------------------------------------
# Hyperparameters (small enough for a laptop / Colab T4)
# ---------------------------------------------------------------------------
BLOCK_SIZE = 128       # attention context window (characters)
BATCH_SIZE = 64
N_EMBED = 192          # d_model
N_HEADS = 6            # d_k = N_EMBED // N_HEADS = 32
N_LAYERS = 6
D_FF = 4 * N_EMBED     # feed-forward inner dimension
DROPOUT = 0.2
MAX_STEPS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 200
LEARNING_RATE = 1e-3
SEED = 1337

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")


# ---------------------------------------------------------------------------
# Self-attention primitives (same form as Lab 9)
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled dot-product attention with an optional additive mask.

    Args:
        Q, K, V (Tensor): shape (..., seq, d_k). Here (B, n_heads, T, d_k).
        mask    (Tensor or None): additive mask broadcastable to
                                  (..., seq_q, seq_k); -inf (here -1e9) at
                                  positions that must not be attended to.

    Returns:
        output (Tensor): shape (..., seq_q, d_k).
        alpha  (Tensor): attention weights, each query row sums to 1.
    """
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (..., T, T)
    if mask is not None:
        scores = scores + mask
    alpha = F.softmax(scores, dim=-1)
    return alpha @ V, alpha


def sinusoidal_positional_encoding(seq_len, d_model, device="cpu"):
    """Sinusoidal positional encodings from Vaswani et al. (2017).

    Returns a tensor of shape (1, seq_len, d_model) whose values lie in
    [-1, 1] and give each position a unique signal to add to the embeddings.
    """
    pos = torch.arange(seq_len, device=device).unsqueeze(1)             # (T, 1)
    i = torch.arange(0, d_model, 2, device=device)                      # (d/2,)
    div = torch.exp(-math.log(10000.0) * i / d_model)                   # (d/2,)
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)                                              # (1, T, d)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention built on scaled_dot_product_attention."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        # project and split into heads: (B, n_heads, T, d_k)
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        out, _ = scaled_dot_product_attention(q, k, v, mask)   # (B, h, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, C)   # concat heads
        return self.dropout(self.W_o(out))


class TransformerBlock(nn.Module):
    """One decoder block: MHA -> Add&Norm -> FFN -> Add&Norm (pre-norm)."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, mask=None):
        # pre-norm residuals train more stably for deep stacks
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# The language model
# ---------------------------------------------------------------------------
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, N_EMBED)   # token embeddings
        # positional information (fixed, not learned) as a registered buffer
        self.register_buffer(
            "pos_encoding",
            sinusoidal_positional_encoding(BLOCK_SIZE, N_EMBED),
        )
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList(
            [TransformerBlock(N_EMBED, N_HEADS, D_FF, DROPOUT) for _ in range(N_LAYERS)]
        )
        self.norm = nn.LayerNorm(N_EMBED)
        self.head = nn.Linear(N_EMBED, vocab_size)             # output layer

        # causal mask: upper triangle (the future) set to -1e9 so each
        # position attends only to itself and earlier positions.
        causal = torch.full((BLOCK_SIZE, BLOCK_SIZE), float("-1e9"))
        causal = torch.triu(causal, diagonal=1)
        self.register_buffer("causal_mask", causal.view(1, 1, BLOCK_SIZE, BLOCK_SIZE))

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embed(idx) + self.pos_encoding[:, :T, :]
        x = self.drop(x)
        mask = self.causal_mask[:, :, :T, :T]
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        logits = self.head(x)                                  # (B, T, vocab)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        """Sample characters, cropping context to the last BLOCK_SIZE tokens."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]        # attention window is finite
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ---------------------------------------------------------------------------
# Loss estimation (same protocol as the other models)
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    model.eval()
    out = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            xb, yb = get_batch(data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def save_results(entry: dict):
    results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
    results[entry["model"]] = entry
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results for '{entry['model']}' -> {RESULTS_PATH}")


# ---------------------------------------------------------------------------
# Train, evaluate, generate
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    vocab, train_data, val_data = build_dataset()
    print(f"Device: {DEVICE} | Vocab size: {vocab.size}")

    model = GPTLanguageModel(vocab.size).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Transformer parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    for step in range(MAX_STEPS):
        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")

        xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    train_time = time.time() - start

    # ----- compute metrics -----
    tokens_processed = MAX_STEPS * BATCH_SIZE * BLOCK_SIZE
    throughput = tokens_processed / train_time            # tokens / second
    peak_mem_mb = (
        torch.cuda.max_memory_allocated() / 1e6 if DEVICE == "cuda" else None
    )

    final = estimate_loss(model, train_data, val_data)
    print(f"\nFinal | train {final['train']:.4f} | val {final['val']:.4f} "
          f"| time {train_time:.1f}s")
    print(f"Throughput: {throughput:,.0f} tokens/s"
          + (f" | Peak GPU mem: {peak_mem_mb:.0f} MB" if peak_mem_mb else ""))

    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    sample_ids = model.generate(context, max_new_tokens=500)[0]
    sample = vocab.decode(sample_ids)
    print("\n----- generated sample (500 chars) -----")
    print(sample)
    print("----------------------------------------")

    # ----- save a checkpoint so demo.py can reload this trained model -----
    ckpt_dir = os.path.join(os.path.dirname(__file__), "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "transformer.pt"))
    print(f"Saved checkpoint -> {os.path.join(ckpt_dir, 'transformer.pt')}")

    save_results({
        "model": "Transformer",
        "train_loss": round(final["train"], 4),
        "val_loss": round(final["val"], 4),
        "train_time_sec": round(train_time, 1),
        "parameters": n_params,
        "throughput_tokens_per_sec": round(throughput),
        "peak_gpu_mem_mb": round(peak_mem_mb) if peak_mem_mb else None,
        "sample": sample,
    })


if __name__ == "__main__":
    main()
