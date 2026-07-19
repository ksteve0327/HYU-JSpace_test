"""Phase 3 swap intervention.

C1 (literal work-order target): spider->ant on the leg-counting prompt. Reported
   honestly incl. whether the baseline even holds at this scale.
C2 (clean mechanism test): swap a confidently-produced concept A -> B on prompts
   where the model's greedy answer is A and the lens reads A in the workspace band.
   alpha sweep {1,2,4}. Controls: random direction, early-layers-only.
D  (layer sweep): move the swap band's start layer and see when the output flips.

Writes results/swap_<key>.md and results/swap_layer_sweep_<key>.md.
"""

from __future__ import annotations

import sys

import torch

import jlens
from common import MODELS, RESULTS_DIR, get_device_dtype, load_hf, load_lens, resolve_single_token
from swaplib import band_from_fraction, build_swap_operators, next_token_logits, swap_hooks

MODEL_KEY = sys.argv[1] if len(sys.argv) > 1 else "qwen0.8b"
model_id, lens_file = MODELS[MODEL_KEY]

device, dtype = get_device_dtype()
tok, hf = load_hf(model_id, device, dtype)
lens_model = jlens.from_hf(hf, tok, compile=False)
lens = load_lens(lens_file)
layers = list(lens.source_layers)
band = band_from_fraction(layers, 0.40, 0.90)
early = band_from_fraction(layers, 0.0, 0.30)


def decode(t):
    return tok.decode([int(t)])


def prob_of(logits_row, word):
    tid = resolve_single_token(tok, " " + word) or resolve_single_token(tok, word)
    if tid is None:
        return None, None
    p = logits_row.softmax(-1)[tid].item()
    r = int((logits_row > logits_row[tid]).sum()) + 1
    return p, r


def single_id(word):
    return resolve_single_token(tok, " " + word) or resolve_single_token(tok, word)


def swapped_logits(prompt, src, tgt, alpha, layer_set, random_tgt=False):
    ops = build_swap_operators(lens, lens_model, layer_set, single_id(src), single_id(tgt),
                               device, random_tgt=random_tgt)
    enc = tok(prompt, return_tensors="pt").to(device)
    with torch.inference_mode(), swap_hooks(lens_model, ops, alpha=alpha):
        return hf(**enc).logits[0, -1].float().cpu()


def report_case(lines, prompt, src, tgt, layer_set=None, tag="band"):
    layer_set = layer_set if layer_set is not None else band
    base = next_token_logits(hf, tok, prompt, device)
    b_top1 = decode(base.argmax())
    lines.append(f"### {prompt!r}  ({src} → {tgt}, layers={layer_set[0]}–{layer_set[-1]}, {tag})")
    pa, ra = prob_of(base, src)
    pb, rb = prob_of(base, tgt)
    lines.append(f"- baseline greedy top-1: {b_top1!r} | "
                 f"P({src})={pa:.3f} r{ra} | P({tgt})={pb:.3f} r{rb}")
    lines.append("")
    lines.append("| alpha | greedy top-1 | top-5 | P(src) | P(tgt) | flipped→tgt? |")
    lines.append("|---|---|---|---|---|---|")
    for a in (1, 2, 4):
        sl = swapped_logits(prompt, src, tgt, a, layer_set)
        top1 = decode(sl.argmax())
        top5 = " ".join(f"{decode(i)!r}" for i in sl.topk(5).indices.tolist())
        pas, _ = prob_of(sl, src)
        pbs, _ = prob_of(sl, tgt)
        tgt_id = single_id(tgt)
        flipped = int(sl.argmax()) == tgt_id
        lines.append(f"| {a} | {top1!r} | {top5} | {pas:.3f} | {pbs:.3f} | {'YES' if flipped else 'no'} |")
    lines.append("")
    return base


lines = [f"# Phase 3 — Swap intervention ({MODEL_KEY}, {model_id})\n"]
lines.append(f"- workspace band (40–90% depth): layers {band}")
lines.append(f"- early-layer control (0–30%): layers {early}\n")

# ---- C1: literal work-order target (expected to fail baseline at small scale) ----
lines.append("## C1 — Literal target: spider→ant on leg-counting prompt")
lines.append("Work-order expectation: greedy 8→6. First, does the baseline hold?\n")
spider_prompt = "The number of legs on the animal that spins webs is"
enc = tok(spider_prompt, return_tensors="pt").to(device)
with torch.inference_mode():
    gen = tok.decode(hf.generate(**enc, max_new_tokens=4, do_sample=False)[0][enc.input_ids.shape[1]:],
                     skip_special_tokens=True)
lines.append(f"- baseline greedy continuation: {gen!r}")
for num in ["8", "eight", "6", "six"]:
    tid = resolve_single_token(tok, " " + num) or resolve_single_token(tok, num)
    lines.append(f"  - P({num!r}) at next token: "
                 f"{next_token_logits(hf, tok, spider_prompt, device).softmax(-1)[tid].item():.4f}"
                 if tid is not None else f"  - {num!r}: not single-token")
if single_id("spider") is not None and single_id("ant") is not None:
    report_case(lines, spider_prompt, "spider", "ant", band, tag="band")
else:
    lines.append(f"- NOTE: spider single-token={single_id('spider')}, ant={single_id('ant')}\n")
lines.append("_Interpretation: if the baseline does not produce 8, the 8→6 swap is untestable "
             "here — an honest negative for this model scale._\n")

# ---- C2: clean mechanism tests ----
lines.append("## C2 — Clean mechanism tests (baseline holds)")
lines.append("Model confidently outputs A and lens reads A in the band; does swapping "
             "A→B move the output to B?\n")
clean_cases = [
    ("The capital of France is", "Paris", "London"),
    ("The opposite of hot is", "cold", "warm"),
    ("The color of the sky on a clear day is", "blue", "red"),
]
for p, a, b in clean_cases:
    report_case(lines, p, a, b, band, tag="band")

# ---- Controls on the primary clean case ----
p0, a0, b0 = clean_cases[0]
lines.append("## Controls (on the primary clean case: capital of France, Paris→London)")
lines.append("### Control 1 — random direction (same magnitude, meaningless target)")
base = next_token_logits(hf, tok, p0, device)
lines.append(f"- baseline top-1: {decode(base.argmax())!r}")
lines.append("| alpha | greedy top-1 | P(London) | flipped→London? |")
lines.append("|---|---|---|---|")
for a in (1, 2, 4):
    sl = swapped_logits(p0, a0, b0, a, band, random_tgt=True)
    pb, _ = prob_of(sl, b0)
    lines.append(f"| {a} | {decode(sl.argmax())!r} | {pb:.3f} | "
                 f"{'YES' if int(sl.argmax())==single_id(b0) else 'no'} |")
lines.append("_Expected: output should NOT become London._\n")

lines.append("### Control 2 — early layers only (0–30% depth)")
report_case(lines, p0, a0, b0, early, tag="early")
lines.append("_Expected: weak or no effect vs. the workspace-band swap._\n")

out = RESULTS_DIR / f"swap_{MODEL_KEY}.md"
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nWrote {out}")

# ---- D: layer sweep ----
sweep_lines = [f"# Phase 3 (D) — Swap layer sweep ({MODEL_KEY})\n"]
sweep_lines.append(f"Primary case: {p0!r} ({a0}→{b0}), alpha=2.")
sweep_lines.append("Apply the swap to a 6-layer window starting at layer S; record output.\n")
sweep_lines.append("| start layer S | window | greedy top-1 | P(London) | flipped→London? |")
sweep_lines.append("|---|---|---|---|---|")
win = 6
for S in range(0, len(layers) - 1, 2):
    wlayers = layers[S:S + win]
    if not wlayers:
        continue
    sl = swapped_logits(p0, a0, b0, 2, wlayers)
    pb, _ = prob_of(sl, b0)
    frac = f"{int(100*S/len(layers))}%"
    sweep_lines.append(f"| {S} ({frac}) | {wlayers[0]}–{wlayers[-1]} | {decode(sl.argmax())!r} "
                       f"| {pb:.3f} | {'YES' if int(sl.argmax())==single_id(b0) else 'no'} |")
out2 = RESULTS_DIR / f"swap_layer_sweep_{MODEL_KEY}.md"
out2.write_text("\n".join(sweep_lines) + "\n")
print("\n".join(sweep_lines))
print(f"\nWrote {out2}")
