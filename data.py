"""
data.py — Dataset setup shared by all three models.

This module covers the "Hour 1" setup from the assignment:
  1. Load the Tiny Shakespeare text (download it once if missing).
  2. Build the character-level vocabulary (stoi / itos).
  3. Encode the text to integer ids and split 90% train / 10% validation.
  4. Provide a get_batch() sampler that the LSTM and Transformer use.

Every model imports from here, so all three see the *same* encoding,
the *same* split, and the *same* loss space. That is what makes the
final comparison fair.
"""

import os
import urllib.request

import torch

# ---------------------------------------------------------------------------
# 1. Load the text
# ---------------------------------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)
DATA_PATH = os.path.join(os.path.dirname(__file__), "input.txt")


def load_text(path: str = DATA_PATH) -> str:
    """Return the raw Tiny Shakespeare text, downloading it once if needed."""
    if not os.path.exists(path):
        print(f"input.txt not found, downloading from {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, path)
        print("download complete.")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# 2. Build the character-level vocabulary
# ---------------------------------------------------------------------------
class Vocab:
    """The entire 'tokenizer' for character-level modeling.

    The vocabulary is just the sorted set of unique characters in the text.
    stoi maps each character to an integer id; itos maps back. There is no
    external tokenizer — these two dicts are all three models need.
    """

    def __init__(self, text: str):
        self.chars = sorted(set(text))          # e.g. '\n', ' ', '!', ..., 'z'
        self.size = len(self.chars)             # ~65 for Tiny Shakespeare
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str) -> list[int]:
        """Turn a string into a list of integer ids."""
        return [self.stoi[ch] for ch in s]

    def decode(self, ids) -> str:
        """Turn a list/iterable of integer ids back into a string."""
        return "".join(self.itos[int(i)] for i in ids)


# ---------------------------------------------------------------------------
# 3. Encode + train/validation split
# ---------------------------------------------------------------------------
def build_dataset(path: str = DATA_PATH, train_frac: float = 0.9):
    """Load text, build the vocab, and return (vocab, train_data, val_data).

    The split is by position (no shuffle): the first 90% of the character
    stream trains, the last 10% validates. Each split is a 1-D LongTensor
    of character ids.
    """
    text = load_text(path)
    vocab = Vocab(text)

    data = torch.tensor(vocab.encode(text), dtype=torch.long)
    n = int(train_frac * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return vocab, train_data, val_data


# ---------------------------------------------------------------------------
# 4. Batch sampler (used by the LSTM and Transformer)
# ---------------------------------------------------------------------------
def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device="cpu"):
    """Sample a batch of context/target pairs from a data tensor.

    Picks `batch_size` random starting positions. For each one, x is a
    window of `block_size` characters and y is that same window shifted one
    step to the right — so the target at position t is the character that
    actually followed the context ending at t.

    Returns:
        x, y: LongTensors of shape (batch_size, block_size).
    """
    # highest valid start index so that x and y both stay in bounds
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Quick self-test: run `python data.py` to verify the setup.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    vocab, train_data, val_data = build_dataset()

    print(f"Total characters in corpus : {len(train_data) + len(val_data):,}")
    print(f"Vocabulary size            : {vocab.size}")
    print(f"Characters                 : {''.join(vocab.chars)!r}")
    print(f"Train tensor               : {tuple(train_data.shape)}")
    print(f"Val tensor                 : {tuple(val_data.shape)}")

    # round-trip check: encode then decode should give back the original
    sample = "To be, or not to be"
    assert vocab.decode(vocab.encode(sample)) == sample
    print(f"Encode/decode round-trip   : OK  ({sample!r})")

    # show one small batch
    xb, yb = get_batch(train_data, block_size=8, batch_size=2)
    print(f"Example batch x shape      : {tuple(xb.shape)}")
    print(f"  x[0] = {vocab.decode(xb[0])!r}")
    print(f"  y[0] = {vocab.decode(yb[0])!r}   (x shifted by one)")
