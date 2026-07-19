"""Phase 2 reading experiments.

A (directed modulation): does an instructed-but-unstated concept (citrus) surface
   in the lens readout over an unrelated carrier sentence? focus vs control.
B (internal reasoning): does 'spider' surface in mid layers for
   'The number of legs on the animal that spins webs is'?

Writes results/reading_A_<key>.md and results/reading_B_<key>.md.
"""

from __future__ import annotations

import sys

import torch

import jlens
from common import (
    MODELS,
    RESULTS_DIR,
    get_device_dtype,
    load_hf,
    load_lens,
    token_variants,
)

MODEL_KEY = sys.argv[1] if len(sys.argv) > 1 else "qwen0.8b"
model_id, lens_file = MODELS[MODEL_KEY]

device, dtype = get_device_dtype()
tok, hf = load_hf(model_id, device, dtype)
lens_model = jlens.from_hf(hf, tok, compile=False)
lens = load_lens(lens_file)
layers = list(lens.source_layers)


def decode(tid: int) -> str:
    return tok.decode([int(tid)])


def rank_of(logits_row: torch.Tensor, tid: int) -> int:
    return int((logits_row > logits_row[tid]).sum().item()) + 1


def best_rank_over_ids(logits_row: torch.Tensor, ids) -> int:
    return min(rank_of(logits_row, i) for i in ids) if ids else 10**9


def topk_str(logits_row: torch.Tensor, k=6) -> str:
    idx = logits_row.topk(k).indices.tolist()
    return " ".join(f"{decode(i)!r}" for i in idx)


def token_ids_at(prompt: str):
    ids = tok(prompt, return_tensors="pt").input_ids[0]
    return ids


# ----------------------------------------------------------------------------
# Experiment B: internal reasoning
# ----------------------------------------------------------------------------
def experiment_B():
    prompt = "The number of legs on the animal that spins webs is"
    concept = token_variants(tok, ["spider"])["spider"]
    with torch.inference_mode():
        lens_logits, model_logits, input_ids = lens.apply(
            lens_model, prompt, layers=layers, positions=[-1], use_jacobian=True
        )
    lines = [f"# Reading Experiment B — internal reasoning ({MODEL_KEY})\n"]
    lines.append(f"- prompt: {prompt!r}")
    lines.append(f"- position: last token; concept 'spider' single-token ids: {concept}")
    lines.append(f"- model greedy next token: {decode(int(model_logits[0].argmax()))!r}\n")
    lines.append("| layer | lens top-6 | rank of 'spider' |")
    lines.append("|---|---|---|")
    best = (10**9, None)
    for L in layers:
        row = lens_logits[L][0]
        r = best_rank_over_ids(row, concept)
        if r < best[0]:
            best = (r, L)
        lines.append(f"| {L} | {topk_str(row)} | {r if r < 10**8 else '-'} |")
    lines.append(f"\n**Best 'spider' rank: {best[0]} at layer {best[1]}** "
                 f"(rank 1 = surfaced as top token in the lens)")
    if best[0] == 1:
        lines.append("→ 'spider' surfaces as the top lens token in the mid network: phenomenon observed.")
    elif best[0] <= 5:
        lines.append("→ 'spider' rises into the lens top-5 in the mid network: weak/partial phenomenon.")
    else:
        lines.append("→ 'spider' never enters the lens top-5: phenomenon NOT observed at this scale.")
    out = RESULTS_DIR / f"reading_B_{MODEL_KEY}.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out}\n")


# ----------------------------------------------------------------------------
# Experiment A: directed modulation (focus vs control)
# ----------------------------------------------------------------------------
def experiment_A():
    focus = ("Think about citrus fruits while you copy this sentence exactly: "
             "The old painting hung crookedly on the wall. Copy: The old painting hung")
    control = ("Copy this sentence exactly: "
               "The old painting hung crookedly on the wall. Copy: The old painting hung")
    citrus_words = ["orange", "lemon", "lime", "citrus", "fruit", "grapefruit"]
    cvars = token_variants(tok, citrus_words)
    citrus_ids = sorted({i for ids in cvars.values() for i in ids})

    lines = [f"# Reading Experiment A — directed modulation ({MODEL_KEY})\n"]
    lines.append(f"- citrus concept single-token ids tracked: "
                 + ", ".join(f"{w}:{cvars[w]}" for w in citrus_words) + "\n")

    for label, prompt in [("FOCUS (citrus instruction)", focus), ("CONTROL (no instruction)", control)]:
        ids = token_ids_at(prompt)
        # copy span = the final 'The old painting hung' tokens; read interior positions
        span_positions = list(range(len(ids) - 4, len(ids)))  # last 4 tokens
        lines.append(f"## {label}")
        lines.append(f"- prompt: {prompt!r}")
        lines.append(f"- read positions (copy span): {span_positions} -> "
                     + " ".join(f"{decode(int(ids[p]))!r}" for p in span_positions) + "\n")
        with torch.inference_mode():
            lens_logits, _, _ = lens.apply(
                lens_model, prompt, layers=layers, positions=span_positions, use_jacobian=True
            )
        lines.append("| layer | best citrus rank (min over positions) | which token / where |")
        lines.append("|---|---|---|")
        overall_best = (10**9, None, None)
        for L in layers:
            block = lens_logits[L]  # [n_pos, vocab]
            best_r, best_desc = 10**9, "-"
            for pi in range(block.shape[0]):
                row = block[pi]
                r = best_rank_over_ids(row, citrus_ids)
                if r < best_r:
                    # find which citrus token is best here
                    bt = min(citrus_ids, key=lambda i: rank_of(row, i))
                    best_r = r
                    best_desc = f"{decode(bt)!r}@pos{span_positions[pi]}"
            if best_r < overall_best[0]:
                overall_best = (best_r, L, best_desc)
            lines.append(f"| {L} | {best_r if best_r < 10**8 else '-'} | {best_desc} |")
        lines.append(f"\n**Best citrus rank: {overall_best[0]} at layer {overall_best[1]} "
                     f"({overall_best[2]})**\n")

    out = RESULTS_DIR / f"reading_A_{MODEL_KEY}.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:8]), "...")
    print(f"Wrote {out}\n")


if __name__ == "__main__":
    experiment_B()
    experiment_A()
