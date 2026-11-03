# AI-Use Summary

This document explains how I used an AI assistant (Claude, via Claude Code) on
this midterm project, and which parts I directed, understood, edited, and
verified myself. I used AI as a pair-programming and tutoring tool, not as a
way to skip the thinking the assignment is testing.

## How I worked

I ran the project as a guided, step-by-step discussion. Before writing any
code I asked the assistant to walk me through each section — the task, the
dataset, and each of the three models — and I made the design decisions at
each fork. The code was then written against those decisions and I ran,
inspected, and reasoned about every result.

## Decisions I made myself

- **Task framing:** character-level next-character prediction (vs. token-level),
  after weighing vocabulary size, interpretability of the bigram, and training
  speed.
- **Dataset:** Tiny Shakespeare, for zero cleaning and a clear three-way
  comparison.
- **Model 2 choice:** LSTM (over a plain RNN or a feedforward window), because
  it gives a clean "sequential memory" contrast between the bigram and the
  transformer.
- **What metrics to report:** I asked for compute metrics (throughput and peak
  GPU memory) in addition to loss and wall-clock time, so the comparison shows
  the cost of each architecture, not only its accuracy.

## What the AI helped with

- **Explanations.** It explained what information each model uses and ignores,
  why the bigram's understanding is limited, why an LSTM still struggles with
  long dependencies, and how self-attention removes that bottleneck. I used
  these explanations to write the analysis in my own words.
- **Implementation.** It helped write the three model scripts and the shared
  data/comparison utilities, following the self-attention style from our course
  Lab 9 (scaled dot-product attention, sinusoidal positional encoding,
  multi-head attention, encoder/decoder block). I adapted the Lab 9 *classifier*
  pattern into an *autoregressive* language model myself in discussion —
  swapping the padding mask for a causal mask and the CLS head for a
  per-position output layer.
- **Debugging and consistency.** It kept the three models on an identical
  training/evaluation protocol (same vocab, same split, same loss, same batch
  sampler) so the comparison is fair.

## What I did and verified myself

- Ran every script, read the training curves, and confirmed each model beats
  the random-guessing baseline (loss ≈ ln(65) ≈ 4.17).
- Read the generated samples and checked that they match the expected
  behavior: the bigram produces letter-pairs only, the LSTM produces real
  words, the transformer holds longer structure.
- Confirmed the numbers in the results table against the raw run outputs in
  `outputs/`.
- Wrote and edited the reflection and README analysis in my own words.

## Parts I made sure I understand

- Why a bigram is a `(vocab × vocab)` table and what each row means.
- How an LSTM's hidden state carries context and why gradients fade over long
  distances.
- How scaled dot-product attention computes `softmax(QKᵀ/√d)·V`, why the causal
  mask is needed for next-token prediction, and why positional encoding is
  required because attention is otherwise order-blind.

## Honesty note

AI wrote a large share of the boilerplate code and first-draft explanations.
My contribution is the direction of the project, the architecture and metric
decisions, the adaptation from the Lab 9 classifier to a language model,
running and interpreting all experiments, and the written analysis. I can
explain every file in this repository and why it is there.
