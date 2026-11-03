"""
demo.py — Same-prompt test: feed one real Shakespeare line to all three
trained models and compare how each continues it, character by character.

This is the fair head-to-head: identical prompt, identical number of new
characters, identical random seed before each model generates — so any
difference in the output comes from the model, not the setup.

Run the three training scripts first (they save checkpoints to ckpts/).
Output is printed and saved to outputs/same_prompt_test.md.
"""

import os

import torch

from data import build_dataset
from bigram import BigramLanguageModel
from lstm import LSTMLanguageModel
from transformer import GPTLanguageModel

HERE = os.path.dirname(__file__)
CKPT_DIR = os.path.join(HERE, "ckpts")
REPORT_PATH = os.path.join(HERE, "outputs", "same_prompt_test.md")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# A real line from the start of the Tiny Shakespeare dataset. In the dataset
# the actual continuation is "hear me speak." — a useful ground truth.
PROMPT = "First Citizen:\nBefore we proceed any further, "
N_GENERATE = 220
SEED = 0


def load(model, name):
    """Load a trained checkpoint into a model and set eval mode."""
    path = os.path.join(CKPT_DIR, f"{name}.pt")
    if not os.path.exists(path):
        raise SystemExit(f"Missing checkpoint {path}. Run the model scripts first.")
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    return model.to(DEVICE).eval()


def continue_prompt(model, prompt_ids):
    """Generate N_GENERATE characters after the prompt; same seed each time."""
    torch.manual_seed(SEED)                       # identical random draws per model
    idx = prompt_ids.clone()
    out = model.generate(idx, max_new_tokens=N_GENERATE)[0]
    return out


def main():
    vocab, _, _ = build_dataset()
    prompt_ids = torch.tensor([vocab.encode(PROMPT)], dtype=torch.long, device=DEVICE)

    models = {
        "Bigram": load(BigramLanguageModel(vocab.size), "bigram"),
        "LSTM": load(LSTMLanguageModel(vocab.size), "lstm"),
        "Transformer": load(GPTLanguageModel(vocab.size), "transformer"),
    }

    lines = []
    lines.append("# Same-Prompt Test\n")
    lines.append(
        "Every model is primed with the **same real line** from the dataset "
        "and asked to generate the next "
        f"{N_GENERATE} characters, with the same random seed.\n"
    )
    lines.append("**Prompt (the >> marks where the model takes over):**\n")
    lines.append("```")
    lines.append(PROMPT + ">>")
    lines.append("```")
    lines.append(
        "\nGround truth: in the dataset this line actually continues "
        "`hear me speak.`\n"
    )

    for name, model in models.items():
        full = vocab.decode(continue_prompt(model, prompt_ids))
        continuation = full[len(PROMPT):]         # only the generated part
        print(f"\n===== {name} =====")
        print(repr(PROMPT) + "  >>")
        print(continuation)

        lines.append(f"## {name}\n")
        lines.append("```")
        lines.append(PROMPT + continuation)
        lines.append("```")
        lines.append("")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[saved -> {REPORT_PATH}]")


if __name__ == "__main__":
    main()
