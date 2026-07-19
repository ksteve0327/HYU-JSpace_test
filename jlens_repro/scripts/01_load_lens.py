"""Phase 1 sanity check: near the final layer the Jacobian-lens top-1 readout at
the last position should match the model's actual greedy next token.

Writes results/lens_sanity_<key>.md.
"""

from __future__ import annotations

import sys

import torch

import jlens
from common import MODELS, RESULTS_DIR, get_device_dtype, load_hf, load_lens

MODEL_KEY = sys.argv[1] if len(sys.argv) > 1 else "qwen0.8b"
model_id, lens_file = MODELS[MODEL_KEY]

PROMPTS = [
    "The capital of France is",
    "Water is made of hydrogen and",
    "The opposite of hot is",
    "Two plus two equals",
    "The sun rises in the",
]

device, dtype = get_device_dtype()
tok, hf = load_hf(model_id, device, dtype)
lens_model = jlens.from_hf(hf, tok, compile=False)
lens = load_lens(lens_file)

layers = list(lens.source_layers)
last_layer = layers[-1]


def decode(tid: int) -> str:
    return tok.decode([int(tid)])


# source_layers stop 1-2 layers before the model's true output layer, so the
# sharpest next-token readout sits "near" the end, not necessarily at the last
# source layer. Pass rule (matches the work order's "마지막 레이어 근처"):
# model's greedy next token is lens top-1 at ANY of the last N source layers.
LAST_N = 3
late_layers = layers[-LAST_N:]


def rank_of(logits_row: torch.Tensor, tid: int) -> int:
    return int((logits_row > logits_row[tid]).sum().item()) + 1  # 1-indexed


lines = [f"# Phase 1 — Lens sanity check ({MODEL_KEY}, {model_id})\n"]
lines.append(f"- source layers: {layers}")
lines.append(
    f"- sanity rule: model greedy next token is lens top-1 at any of last {LAST_N} "
    f"source layers {late_layers}\n"
)

n_pass = 0
for p in PROMPTS:
    with torch.inference_mode():
        lens_logits, model_logits, input_ids = lens.apply(
            lens_model, p, layers=layers, positions=[-1], use_jacobian=True
        )
    model_top1 = int(model_logits[0].argmax())
    late_top1s = {L: int(lens_logits[L][0].argmax()) for L in late_layers}
    ok = any(t == model_top1 for t in late_top1s.values())
    n_pass += ok
    last_rank = rank_of(lens_logits[last_layer][0], model_top1)
    lines.append(f"## {p!r}")
    lines.append(f"- model next token: {decode(model_top1)!r}")
    lines.append(
        "- late-layer top-1: "
        + " | ".join(f"L{L}:{decode(t)!r}" for L, t in late_top1s.items())
        + f"  -> {'PASS' if ok else 'FAIL'}"
    )
    lines.append(f"- rank of model token in lens@{last_layer}: {last_rank}")
    # show layer progression (top-1 per layer) to eyeball sensory->workspace->motor
    prog = []
    for L in layers[::max(1, len(layers) // 8)]:
        t1 = int(lens_logits[L][0].argmax())
        prog.append(f"L{L}:{decode(t1)!r}")
    lines.append("- layer top-1 progression: " + " | ".join(prog) + "\n")

lines.insert(3, f"\n**Result: {n_pass}/{len(PROMPTS)} passed**\n")

out = RESULTS_DIR / f"lens_sanity_{MODEL_KEY}.md"
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nWrote {out}")
print(f"SANITY {n_pass}/{len(PROMPTS)} PASSED")
