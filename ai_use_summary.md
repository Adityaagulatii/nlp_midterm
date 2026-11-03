# AI-Use Summary

This document explains honestly how I used an AI assistant (Claude, via Claude
Code) on this midterm, and which parts I wrote, edited, debugged, or understood
myself. I used AI as a pair-programming and tutoring tool. I directed the
project, made every design decision, interpreted all results, and wrote the
analysis and reflections in my own words. AI wrote most of the first-draft code
and prose, which I then reviewed and revised.

## How I worked

I deliberately ran the project as a **step-by-step discussion before any code
was written**. I asked the assistant to walk me through each section — the
task, the dataset, and each of the three models — and I made the decision at
each fork before moving on. Only after I had chosen a direction did we
implement it. I ran every script, read every output, and decided what to change
next.

## Decisions I made myself

- **Task framing:** character-level (not token-level) next-character
  prediction, after weighing vocabulary size, interpretability, and speed.
- **Dataset:** Tiny Shakespeare, for zero cleaning and a clear comparison. I
  also asked where the data came from and confirmed it is downloaded from a
  public source, then decided to **commit `input.txt`** so the repo is
  self-contained.
- **Model 2 choice:** LSTM over a plain RNN or a feedforward window.
- **Project structure:** I asked for each model in its **own separate file**,
  and for the small transformer to follow the **self-attention style from our
  course Lab 9** (scaled dot-product attention, sinusoidal positional encoding,
  multi-head attention, encoder/decoder block).
- **Metrics:** I asked to add **compute metrics** — throughput (tokens/sec) and
  peak GPU memory — on top of loss and training time, because I wanted the
  comparison to show the *cost* of each architecture, not just its accuracy.
- **A fair test:** I asked for a **same-prompt test** that feeds one identical
  real line from the dataset to all three models, so the comparison of
  generated text is controlled rather than cherry-picked.

## Problems I caught and directed the fix for

- **Undertrained transformer.** On the first run the transformer scored *worse*
  than the LSTM (val loss 1.60 vs 1.52). We identified that it had not
  converged (the learning rate was too low and it needed more steps), and I had
  it re-trained with a higher learning rate until it genuinely beat the LSTM. I
  understood *why* this mattered: an undertrained transformer losing to an LSTM
  would misrepresent the whole point of the project.
- **Interpreting the same-prompt test.** When no model reproduced the real line
  (`hear me speak.`), I understood this is *correct* behavior — the models
  generate plausible new text and should not memorize the data — and I asked
  for that point to be made explicit.
- **Repository clean-up.** I had a redundant auto-generated report
  (`comparison.md`) removed because its content was already in the README, and
  I caught a **broken (404) link** to it in the README and had it fixed. I also
  had a separate course lab excluded from the repo.

## What the AI helped with

- **Explanations.** It explained what information each model uses and ignores,
  why the bigram is limited, why an LSTM still struggles with long
  dependencies, and how self-attention removes that bottleneck. I used these to
  build my own understanding and then wrote the analysis myself.
- **First-draft code.** It wrote the initial versions of `data.py`,
  `bigram.py`, `lstm.py`, `transformer.py`, `compare.py`, and `demo.py`
  following my decisions, and kept all three models on an identical
  training/evaluation protocol so the comparison is fair.
- **Mechanics.** It handled the Git commits and the repetitive boilerplate.

## What I wrote and edited myself

- **The Analysis section of the README.** I revised the AI's draft in my own
  words and added my own opinions — for example, that on a dataset this small
  the LSTM is close to the transformer; that the metric improvement may not
  justify the extra compute and time on a small sample; that the models should
  be tested on a larger real-world corpus (research documents or reports) to
  see attention's full benefit; and a point about token-level cost trade-offs.
- **The reflection "What I learned."** I wrote that metrics are not always the
  best judge in NLP, and that seeing the actual generated output — where the
  transformer was clearly the most realistic — changed how I judge a model's
  performance.
- Where I felt Claude's draft did not capture my own view, I rewrote it in my
  own words rather than accepting it.

## What I verified myself

- Ran every script and confirmed each model beats the random-guessing baseline
  (loss ≈ ln(65) ≈ 4.17).
- Read the generated samples and the same-prompt test and confirmed the
  expected progression: bigram → letter-pairs only; LSTM → real words; and
  transformer → the most coherent, realistic text.
- Checked the numbers in the results table against the raw run logs in
  `outputs/`.

## Parts I made sure I understand

- Why a bigram is a `(vocab × vocab)` table and what each row means.
- How an LSTM's hidden state carries context and why gradients fade over long
  distances.
- How scaled dot-product attention computes `softmax(QKᵀ/√d)·V`, why a causal
  mask is needed for next-character prediction, and why positional encoding is
  required because attention is otherwise order-blind.

## Honesty note

AI wrote a large share of the code and the first drafts of the explanations. My
own contribution is the direction of the project, every design and metric
decision, catching and fixing the problems above, interpreting all the
experiments, and writing the analysis and reflections in my own words. I can
explain every file in this repository and why it is there.
