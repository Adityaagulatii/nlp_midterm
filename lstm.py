"""
lstm.py — Model 2 of 3: an LSTM language model (non-transformer).

WHAT THIS MODEL IS
------------------
A character embedding feeds an LSTM, whose hidden state is passed forward
from one character to the next; a linear head turns the hidden state at each
step into next-character logits. Unlike the bigram's fixed table, the
prediction here depends on an evolving memory of everything read so far.

HOW IT USES MORE CONTEXT THAN THE BIGRAM
    The bigram sees exactly one character. The LSTM carries a hidden state
    (a learned summary vector) that is updated at every step, so in principle
    its prediction depends on the entire prefix, not just the last character.
    It can keep a 'q' in mind until it emits 'u', or stay inside a word until
    a plausible ending.

WHAT IMPROVEMENT IT PROVIDES
    Because it remembers recent context, it produces real words, closes
    quotes and brackets more often, and keeps the SPEAKER:/line structure far
    more consistently than the bigram. Its loss drops well below the bigram's
    one-character ceiling.

WHY IT MAY STILL STRUGGLE WITH LONGER DEPENDENCIES
    All of that memory is squeezed into one fixed-size vector and updated
    step by step. Information from long ago must survive many updates to still
    matter; gradients across many steps shrink (the vanishing-gradient
    problem the LSTM gates reduce but do not eliminate). So links between
    things far apart — a name introduced 200 characters earlier, long-range
    grammatical agreement — tend to fade. The model has memory, but no way to
    look *directly* back at an arbitrary earlier position. That direct lookup
    is exactly what attention adds in Model 3.

This script trains the model, reports train/val loss and training time,
generates a text sample, and appends its numbers to results.json.
"""

import json
import os
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from data import build_dataset, get_batch

# ---------------------------------------------------------------------------
# Hyperparameters (small enough for a laptop / Colab)
# ---------------------------------------------------------------------------
BLOCK_SIZE = 64        # characters per training sequence
BATCH_SIZE = 64
N_EMBED = 128          # character embedding size
N_HIDDEN = 256         # LSTM hidden state size
N_LAYERS = 2           # stacked LSTM layers
DROPOUT = 0.2
MAX_STEPS = 3000
EVAL_INTERVAL = 500
EVAL_ITERS = 200
LEARNING_RATE = 2e-3
SEED = 1337

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, N_EMBED)
        self.lstm = nn.LSTM(
            input_size=N_EMBED,
            hidden_size=N_HIDDEN,
            num_layers=N_LAYERS,
            dropout=DROPOUT,
            batch_first=True,          # input shape is (B, T, features)
        )
        self.head = nn.Linear(N_HIDDEN, vocab_size)

    def forward(self, idx, targets=None, hidden=None):
        # idx: (B, T) ids -> (B, T, N_EMBED) embeddings
        x = self.embed(idx)
        # out: (B, T, N_HIDDEN) hidden state at every timestep
        out, hidden = self.lstm(x, hidden)
        logits = self.head(out)        # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss, hidden

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        """Sample characters one at a time, carrying the hidden state forward.

        We feed the seed through once to prime the hidden state, then at each
        step feed only the newly sampled character — the recurrent memory does
        the work of remembering everything before it.
        """
        logits, _, hidden = self(idx)
        last = idx[:, -1:]                              # (B, 1)
        for _ in range(max_new_tokens):
            logits, _, hidden = self(last, hidden=hidden)
            logits = logits[:, -1, :]                   # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            last = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, last], dim=1)
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
            _, loss, _ = model(xb, yb)
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

    model = LSTMLanguageModel(vocab.size).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LSTM parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    for step in range(MAX_STEPS):
        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")

        xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
        _, loss, _ = model(xb, yb)
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

    # seed generation with a newline (id 0)
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    sample_ids = model.generate(context, max_new_tokens=500)[0]
    sample = vocab.decode(sample_ids)
    print("\n----- generated sample (500 chars) -----")
    print(sample)
    print("----------------------------------------")

    save_results({
        "model": "LSTM",
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
