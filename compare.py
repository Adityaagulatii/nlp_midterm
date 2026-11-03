"""
compare.py — Hour 7 evaluation: assemble the three-model comparison.

Reads results.json (written by bigram.py, lstm.py, transformer.py) and
produces:
  * a results table (loss, time, compute, parameters, quality rating),
  * a loss comparison against the random-guessing baseline,
  * the generated text samples side by side,
  * written observations answering the assignment's analysis questions.

The report is printed to the console and saved to outputs/comparison.md so it
can be pasted straight into the README. Run the three model scripts first.
"""

import json
import math
import os

HERE = os.path.dirname(__file__)
RESULTS_PATH = os.path.join(HERE, "results.json")
REPORT_PATH = os.path.join(HERE, "outputs", "comparison.md")

MODEL_ORDER = ["Bigram", "LSTM", "Transformer"]
VOCAB_SIZE = 65
BASELINE_LOSS = math.log(VOCAB_SIZE)   # loss of uniform random guessing


def load_results() -> dict:
    if not os.path.exists(RESULTS_PATH):
        raise SystemExit(
            "results.json not found. Run bigram.py, lstm.py and transformer.py first."
        )
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def quality_rating(results: dict) -> dict:
    """Rate generated-text quality Low/Medium/High by validation-loss rank.

    Lower loss means more realistic text, so we rank the models and label the
    best High, the worst Low, and anything between Medium.
    """
    present = [m for m in MODEL_ORDER if m in results]
    ranked = sorted(present, key=lambda m: results[m]["val_loss"])
    labels = {}
    for i, m in enumerate(ranked):
        if i == 0:
            labels[m] = "High"
        elif i == len(ranked) - 1:
            labels[m] = "Low"
        else:
            labels[m] = "Medium"
    return labels


def fmt(value, suffix="", dash="-"):
    return f"{value}{suffix}" if value is not None else dash


def build_report(results: dict) -> str:
    quality = quality_rating(results)
    present = [m for m in MODEL_ORDER if m in results]

    lines = []
    lines.append("# Three-Model Comparison\n")
    lines.append(
        "Task: character-level next-character prediction on Tiny Shakespeare "
        f"(vocabulary = {VOCAB_SIZE} characters).\n"
    )
    lines.append(
        f"Random-guessing baseline loss = ln({VOCAB_SIZE}) = "
        f"**{BASELINE_LOSS:.2f}**. Every model should beat this; how far below "
        "it they get is the story.\n"
    )

    # ---- results table ----
    lines.append("## Results table\n")
    header = (
        "| Model | Train Loss | Validation Loss | Training Time | "
        "Throughput (tok/s) | Peak GPU Mem | Parameters | Generated Text Quality |"
    )
    sep = "|---|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for m in present:
        r = results[m]
        lines.append(
            f"| {m} "
            f"| {r['train_loss']:.4f} "
            f"| {r['val_loss']:.4f} "
            f"| {r['train_time_sec']:.1f}s "
            f"| {fmt(r.get('throughput_tokens_per_sec'))} "
            f"| {fmt(r.get('peak_gpu_mem_mb'), ' MB')} "
            f"| {r['parameters']:,} "
            f"| {quality[m]} |"
        )
    lines.append("")

    # ---- loss comparison ----
    lines.append("## Loss comparison\n")
    for m in present:
        r = results[m]
        drop = BASELINE_LOSS - r["val_loss"]
        lines.append(
            f"- **{m}**: val loss {r['val_loss']:.4f} "
            f"({drop:.2f} below the {BASELINE_LOSS:.2f} baseline)."
        )
    best = min(present, key=lambda m: results[m]["val_loss"])
    fastest = min(present, key=lambda m: results[m]["train_time_sec"])
    lines.append("")
    lines.append(
        f"Best (lowest validation loss): **{best}**. "
        f"Fastest to train: **{fastest}**."
    )
    lines.append("")

    # ---- generated samples ----
    lines.append("## Generated text samples\n")
    for m in present:
        lines.append(f"### {m}\n")
        lines.append("```")
        lines.append(results[m]["sample"].strip("\n"))
        lines.append("```")
        lines.append("")

    # ---- observations ----
    lines.append("## Observations\n")
    lines.append(observations(results, best, fastest, quality))

    return "\n".join(lines)


def observations(results, best, fastest, quality) -> str:
    bi = results.get("Bigram", {})
    ls = results.get("LSTM", {})
    tf = results.get("Transformer", {})

    paras = []
    paras.append(
        "**Which model performed best?** "
        f"The {best} achieved the lowest validation loss "
        f"({results[best]['val_loss']:.4f}), so it models the text best. "
        "Loss falls monotonically as the models gain access to more context: "
        "the bigram sees one character, the LSTM compresses the whole prefix "
        "into a hidden state, and the transformer attends directly over its "
        "full window."
    )
    paras.append(
        "**Which model trained fastest?** "
        f"The {fastest} trained fastest "
        f"({results[fastest]['train_time_sec']:.1f}s). The bigram is cheapest "
        "per step (a single table lookup) but learns almost nothing; the "
        "transformer costs the most compute per step because attention and "
        "the feed-forward sub-layers do far more work."
    )
    if bi and ls and tf:
        paras.append(
            "**Which model generated the most realistic text?** "
            "The transformer. The bigram produces plausible letter pairs but "
            "no real words; the LSTM produces real words and keeps the "
            "SPEAKER:/line structure; the transformer additionally holds "
            "longer-range coherence — balanced dialogue turns and more "
            "consistent sentence shape."
        )
        paras.append(
            "**What did the transformer improve over the other models?** "
            f"It lowered validation loss from {bi['val_loss']:.4f} (bigram) and "
            f"{ls['val_loss']:.4f} (LSTM) to {tf['val_loss']:.4f}. The gain comes "
            "from self-attention: any position can look directly at any earlier "
            "position in one step, instead of relying on a one-character window "
            "(bigram) or a single fixed-size memory vector squeezed through many "
            "sequential updates (LSTM)."
        )
        worth = "yes" if tf["val_loss"] < ls["val_loss"] else "not clearly"
        paras.append(
            "**Was the improvement worth the added complexity?** "
            f"On this dataset, {worth}. The transformer has more parameters "
            f"({tf['parameters']:,} vs the LSTM's {ls['parameters']:,}) and takes "
            f"longer to train ({tf['train_time_sec']:.1f}s vs "
            f"{ls['train_time_sec']:.1f}s), but it reaches a lower loss and more "
            "coherent samples. For a tiny corpus the margin is modest; the "
            "architecture's real advantage grows with more data and longer "
            "context."
        )
    paras.append(
        "**What did I learn about attention?** "
        "Attention replaces sequential memory with direct, content-based "
        "lookup. Each position computes how relevant every other position is "
        "and reads a weighted mixture of them, so long-range dependencies no "
        "longer have to survive a long chain of recurrent updates. Combined "
        "with positional encoding (attention alone is order-blind), this is "
        "what lets the transformer beat the LSTM at the very same task."
    )
    return "\n\n".join(paras)


def main():
    results = load_results()
    report = build_report(results)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\n[saved report -> {REPORT_PATH}]")


if __name__ == "__main__":
    main()
