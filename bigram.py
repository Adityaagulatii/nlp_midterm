"""
bigram.py — Model 1 of 3: the Bigram language model.

WHAT THIS MODEL IS
------------------
A bigram model predicts the next character using ONLY the current character.
We implement it as a single lookup table of shape (vocab_size, vocab_size):
row `c` holds the model's scores (logits) for every character that might
follow character `c`. nn.Embedding(vocab, vocab) *is* that table. Training by
cross-entropy makes each row converge to the log-frequencies of the real
next-character counts, so this is the trainable form of "count every pair of
adjacent characters and normalize."

WHAT INFORMATION IT USES
    Exactly one character of context — the single character immediately
    before the one being predicted. Nothing else.

WHAT INFORMATION IT IGNORES
    Everything except that one character: the word being spelled, the words
    before it, position in the line, who is speaking, punctuation two steps
    back — all invisible. P(next | current) is fixed no matter what came
    before the current character.

WHY ITS LANGUAGE UNDERSTANDING IS LIMITED
    Language depends on long-range structure (a 'q' needs a 'u', an open
    quote needs a close, a sentence needs a verb). With a one-character
    memory the model cannot represent any of that. It learns local letter
    statistics and nothing more, so its samples look like plausible letter
    pairs strung into nonsense words.

This script trains the model, reports train/val loss and training time,
generates a text sample, and appends its numbers to results.json for the
final comparison table.
"""

import json
import os
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from data import build_dataset, get_batch

# ---------------------------------------------------------------------------
# Hyperparameters (small and fast — a bigram needs very little)
# ---------------------------------------------------------------------------
BLOCK_SIZE = 8        # only affects batching efficiency, not the model itself
BATCH_SIZE = 32
MAX_STEPS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 200      # batches averaged when estimating loss
LEARNING_RATE = 1e-2
SEED = 1337

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class BigramLanguageModel(nn.Module):
    """Next-character scores come straight from a (vocab x vocab) table."""

    def __init__(self, vocab_size: int):
        super().__init__()
        # token i -> a vector of `vocab_size` logits: the scores for what
        # character comes next. This single table is the whole model.
        self.token_logits = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx: (B, T) integer ids. Look up each one's row of next-char logits.
        logits = self.token_logits(idx)              # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            # cross_entropy wants (N, V) logits and (N,) targets
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        """Autoregressively sample `max_new_tokens` characters.

        Because the model only looks at the current character, we only ever
        need the last column of idx to predict the next one.
        """
        for _ in range(max_new_tokens):
            logits, _ = self(idx)                    # (B, T, vocab_size)
            logits = logits[:, -1, :]                # keep only the last step
            probs = F.softmax(logits, dim=-1)        # -> probabilities
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)   # append and continue
        return idx


# ---------------------------------------------------------------------------
# Loss estimation (averaged over several batches, train and val)
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


# ---------------------------------------------------------------------------
# Append this model's results to the shared results.json
# ---------------------------------------------------------------------------
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

    model = BigramLanguageModel(vocab.size).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Bigram parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ----- train, timing only the optimization loop -----
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
    throughput = tokens_processed / train_time
    peak_mem_mb = (
        torch.cuda.max_memory_allocated() / 1e6 if DEVICE == "cuda" else None
    )

    final = estimate_loss(model, train_data, val_data)
    print(f"\nFinal | train {final['train']:.4f} | val {final['val']:.4f} "
          f"| time {train_time:.1f}s")
    print(f"Throughput: {throughput:,.0f} tokens/s"
          + (f" | Peak GPU mem: {peak_mem_mb:.0f} MB" if peak_mem_mb else ""))

    # ----- generate a sample, seeded with a newline -----
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)  # id 0 == '\n'
    sample_ids = model.generate(context, max_new_tokens=500)[0]
    sample = vocab.decode(sample_ids)
    print("\n----- generated sample (500 chars) -----")
    print(sample)
    print("----------------------------------------")

    # ----- record everything for the comparison table -----
    save_results({
        "model": "Bigram",
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
