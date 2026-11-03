# Comparing Three NLP Models: Bigram vs. LSTM vs. Transformer

Midterm project for Applied NLP. I train and compare three model
architectures on a single language-modeling task to understand how different
ways of using context change performance — and why attention helps.

---

## What NLP task did I solve?

**Character-level next-character prediction** (autoregressive language
modeling): given the previous characters of text, predict the next one. At
each position a model outputs a probability distribution over the vocabulary
and is trained with cross-entropy loss. All three models solve this exact same
task on the exact same data, so the comparison is fair.

I chose character-level (rather than word/token-level) because it makes the
bigram model trivially interpretable (a 65×65 table), keeps the vocabulary
tiny, and trains fast on a laptop.

## What dataset did I use?

**Tiny Shakespeare** — ~1.1 million characters (1,115,394) of Shakespeare
plays, a standard small benchmark for this task. It needs no cleaning (one
plain-text file, a single character set).

- **Vocabulary:** 65 unique characters (letters, digits, punctuation, space,
  newline).
- **Split:** 90% train (1,003,854 chars) / 10% validation (111,540 chars), by
  position, no shuffle.
- **Data file:** `input.txt` is committed to this repo, so the project is
  self-contained. If it is ever missing, the first run of any script
  re-downloads it automatically from the public `char-rnn` repository
  (`https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`).

A useful reference point: a model that knows nothing and guesses uniformly
over 65 characters has loss ≈ ln(65) ≈ **4.17**. Every model should beat this;
how far below it they get is the whole story.

## What three models did I compare, and how does each work?

| # | Model | Context it uses | Core mechanism |
|---|---|---|---|
| 1 | **Bigram** | Only the **1 previous** character | A `(65×65)` lookup table of next-character scores |
| 2 | **LSTM** (non-transformer) | A compressed summary of **all previous** characters | A recurrent hidden state updated one character at a time |
| 3 | **Transformer** | **All previous** characters in its window, weighted by relevance | Self-attention + positional encoding |

### Model 1 — Bigram ([bigram.py](bigram.py))
Implemented as `nn.Embedding(65, 65)`: the row for character *c* **is** the
model's scores for whatever comes next. Trained with cross-entropy, each row
converges to the log-frequencies of the real next-character counts.
- **Uses:** exactly one character of context.
- **Ignores:** everything else — the word being spelled, earlier words,
  position in the line, who is speaking.
- **Why limited:** with a one-character memory it cannot represent any
  long-range structure (a `q` needing a `u`, closing a quote, finishing a
  word). It learns local letter statistics and nothing more.

### Model 2 — LSTM ([lstm.py](lstm.py))
Character embedding → 2-layer LSTM → linear head. The hidden state is carried
from one character to the next, so in principle the prediction depends on the
entire prefix.
- **More context than the bigram:** it remembers recent history, so it
  produces real words and keeps the `SPEAKER:`/line structure.
- **Improvement:** loss drops well below the bigram's one-character ceiling.
- **Still struggles with long dependencies:** all memory is squeezed into one
  fixed-size vector updated step by step; information from far back must
  survive many updates, and gradients across many steps shrink (the
  vanishing-gradient problem). It has memory, but no way to look *directly*
  back at an arbitrary earlier position.

### Model 3 — Transformer ([transformer.py](transformer.py))
A small GPT-style decoder built from the same self-attention components as our
course Lab 9 — scaled dot-product attention, sinusoidal positional encoding,
multi-head attention, and a feed-forward sub-layer with residuals and layer
norm — adapted for generation with a **causal mask** (position *t* sees only
≤ *t*) and a **per-position output layer**.

Required components and where they live:
- **Token embeddings** — `nn.Embedding` in `GPTLanguageModel`
- **Positional information** — `sinusoidal_positional_encoding()`
- **Self-attention** — `scaled_dot_product_attention()` + `MultiHeadSelfAttention`
- **Feedforward layer** — the FFN inside `TransformerBlock`
- **Output layer** — `self.head` (Linear → vocab logits)

Config: 6 layers, 6 heads, `d_model=192`, context window 128, ~2.7M
parameters — small enough for a laptop/Colab.

## Results

### Results table

| Model | Train Loss | Validation Loss | Training Time | Generated Text Quality | Notes |
|---|---|---|---|---|---|
| Bigram | 2.4581 | 2.4892 | 26.1s | **Low** | 1-char context; plateaus immediately; letter-pairs only |
| Non-Transformer (LSTM) | 1.2774 | 1.5239 | 101.7s | **Medium** | Real words + speaker structure; long-range coherence fades |
| Transformer | 1.2091 | 1.5095 | 750.0s | **High** | Lowest loss; coherent speaker turns; most compute |

### Compute metrics (the cost of each architecture)

| Model | Parameters | Throughput (tokens/s) | Peak GPU Memory |
|---|---|---|---|
| Bigram | 4,225 | 48,950 | ~0 MB (negligible) |
| LSTM | 946,625 | 120,858 | 128 MB |
| Transformer | 2,689,985 | 54,612 | 901 MB |

*(All runs on a single CUDA GPU. Time is the training loop only; throughput =
characters processed ÷ training time.)*

### Generated text samples

**Bigram** — plausible letter pairs, but no real words:
```
CExthy brid owindakis by bth
HAPet bobe d e.
Wanthar u qur, vet?
```

**LSTM** — real words and clean speaker/line structure:
```
Clown:
Read cowful, O lo, being it was been evaligence?

ANGELO:
The hand for his
```

**Transformer** — real speaker names and the most coherent dialogue:
```
WARWICK:
Title, March tills he with so trial,
And take a got of the cage noble hour.

QUEEN ELIZABETH:
And all these interchanges whiles breath shall
seemst cort of a sin.

LUCIO:
O, as it that yet to follow the kind mock'd,
```

Full run logs for each model are in [outputs/](outputs/).

### Same-prompt test (all three continue the *same* line)

To compare the models fairly, [demo.py](demo.py) primes all three with one
identical real line from the dataset and asks each to generate the next 220
characters — **same prompt, same number of characters, same random seed**, so
any difference comes only from the model. In the dataset this line actually
continues `hear me speak.`

**Prompt:** `First Citizen:\nBefore we proceed any further, ▮`

| Source | Continuation after the prompt |
|---|---|
| **Expected (real text)** | `hear me speak.` |
| **Bigram** output | `fu ay IIn eall Clloimeabs bind...` (nonsense) |
| **LSTM** output | `our cheek earth?` (real words, wrong) |
| **Transformer** output | `full well.` (real phrase, grammatical) |

None of the models reproduce `hear me speak.` exactly — and that is the
*correct* behavior. These models generate plausible new text; they do not
memorize and retrieve the training data (reproducing the line verbatim would be
a sign of overfitting). What the test shows is the quality gradient in *how*
each one continues. The full generations are below.

**Bigram** — ignores the prompt entirely (it sees only the last character, a
space) and produces letter-soup:
```
First Citizen:
Before we proceed any further, fu ay IIn eall Clloimeabs bind mechithisur ig wen. n;
Ag's,
Swant
```

**LSTM** — real words, real speaker names, clean line structure, but it drops
the thread of the opening sentence and wanders into new speakers:
```
First Citizen:
Before we proceed any further, our cheek earth?

HASTINGS:
Fellow; he's rewards controud, for sister.

MENENIUS:
This is the shear but stopp;
```

**Transformer** — completes the sentence with a real phrase (`full well.`) and
holds the most coherent dialogue, with correctly formatted, real character
names:
```
First Citizen:
Before we proceed any further, full well.
Gentlemen, God's brother! how is giner?

NORTHUMBERLAND:
And read for steward, the branch's death!

LADY CAPULET:
How now?

MERCUTIO:
Was this be?
```

The progression is exactly what the loss numbers predict: the bigram cannot use
the prompt at all, the LSTM uses recent context but loses longer structure, and
the transformer best continues the actual sentence and sustains coherent form.
Full output: [outputs/same_prompt_test.md](outputs/same_prompt_test.md).

## Analysis

**Which model performed best?** The **Transformer** (lowest validation loss,
1.5095). Loss falls monotonically as the models gain context: 1 character
(bigram) → whole prefix in a hidden state (LSTM) → direct attention over the
window (transformer).

**Which model trained fastest?** The **Bigram** (26.1s) — one table lookup per
step — but it learns almost nothing. Note a nuance in the throughput numbers:
the LSTM has the *highest* throughput (120k tok/s) because its per-step work is
cheap and it ran at a shorter context; the transformer processes fewer
tokens/sec because attention and the FFN do far more work per step.

**Which model generated the most realistic text?** The **Transformer**. The
bigram makes letter-pairs with no real words; the LSTM makes real words and
keeps the `SPEAKER:` format; the transformer additionally holds longer-range
coherence — real speaker names and more consistent dialogue turns.

**What did the transformer improve?** It lowered validation loss from 2.4892
(bigram) and 1.5239 (LSTM) to 1.5095. The gain comes from **self-attention**:
any position can look *directly* at any earlier position in one step, instead
of relying on a one-character window (bigram) or a single fixed-size memory
vector squeezed through many sequential updates (LSTM).

**Was the improvement worth the added complexity?** On this small dataset,
**yes, but modestly**. The transformer uses ~2.8× the LSTM's parameters and
~7× its training time to reach a loss only slightly lower (1.5095 vs 1.5239) —
though its samples are clearly more coherent. On a tiny corpus the margin is
small; the architecture's real advantage grows with more data and longer
context. The honest takeaway is that attention *wins here and scales far
better*, but it isn't free.

**What did I learn about attention?** Attention replaces sequential memory with
direct, content-based lookup: each position computes how relevant every other
position is and reads a weighted mixture of them, so long-range dependencies no
longer have to survive a long chain of recurrent updates. Because attention
alone is order-blind, it must be paired with positional encoding. Together,
that is what lets the transformer beat the LSTM on the very same task.

## What I learned from this project

- A model's power is bounded by **how much context it can use and how well it
  uses it** — the single biggest lever here was context, not parameter count.
- The bigram is a perfect "floor": its immediate plateau (~2.49) shows there is
  simply nothing more to learn from one character.
- Lower loss shows up visibly in the samples — words forming, then structure,
  then coherence — which made the abstract loss numbers concrete.
- Better architectures cost more compute; a fair comparison has to report time
  and memory, not just accuracy.

## How I used AI responsibly

I used Claude (via Claude Code) as a pair-programming and tutoring tool. I made
the design decisions (character-level, Tiny Shakespeare, LSTM for Model 2,
which metrics to report), directed the work step by step, and ran, inspected,
and interpreted every result myself. When the transformer first underperformed
because it was undertrained, I diagnosed it and re-ran it with a proper
learning rate rather than accepting the number. Full details — including what I
wrote vs. what AI drafted, and the concepts I made sure I understand — are in
[ai_use_summary.md](ai_use_summary.md).

## Repository structure

```
.
├── input.txt          # Tiny Shakespeare dataset (~1.1 MB, committed)
├── data.py            # dataset: load/download, vocab (stoi/itos), split, get_batch
├── bigram.py          # Model 1: bigram language model
├── lstm.py            # Model 2: LSTM language model
├── transformer.py     # Model 3: small GPT-style transformer
├── compare.py         # Hour 7: builds the comparison report from results.json
├── demo.py            # same-prompt test: all three continue one identical line
├── results.json       # metrics + samples, written by the model scripts
├── outputs/           # saved run logs + same-prompt test
├── ai_use_summary.md  # responsible-AI-use documentation
├── requirements.txt
└── README.md
```

## How to reproduce

```bash
pip install -r requirements.txt   # torch
python bigram.py                  # trains Model 1, writes results.json + outputs/
python lstm.py                    # trains Model 2
python transformer.py             # trains Model 3
python compare.py                 # builds the comparison report locally
python demo.py                    # same-prompt test (needs the 3 checkpoints)
```

A GPU is recommended for the transformer (CPU works but is slow). Each script
downloads the dataset automatically on first run.
