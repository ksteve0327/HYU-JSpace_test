"""Phase 6 — familiarity vs J-space representation.

Tests the user's chain: less training data -> no similar tokens in J-space ->
output is "limited". We do NOT assume the last link ("limited == consistent"):
three hypotheses are left open and判정 by metric 2 —
  H-converge : familiarity down -> output narrows (entropy down, resamples agree)
  H-wander   : familiarity down -> output widens (entropy up, resamples scatter)
  H-refuse   : familiarity down -> "unknown" (metacognition)

Familiarity axis = entity frequency (F4 hyper-frequent .. F1 fabricated, F0 noise),
same fixed frame, entity is the only variable. No stock prediction (random-walk
confound). All raw resample outputs saved for human judgement.

Metrics per (entity, frame), read at the answer position over the workspace band
(40-90% depth) via a single `lens.apply`:
  1a neighbor cohesion  = mean pairwise cosine of the lens top-20 tokens'
                          embedding vectors (tight cluster = has a known neighborhood)
  1c lens confidence    = lens top-1 prob + margin (p0-p1)
  2a output entropy     = Shannon entropy of the MODEL's next-token distribution
  3  sharpest layer     = band layer of max cohesion (when the neighborhood is tightest)
  2b resample diversity = (F4/F1 only) N sampled answers -> distinct ratio
  2c refusal rate       = fraction of outputs matching an "unknown/…" rule
Metric 1b (sparse-decomposition sharpness) is DROPPED: the jlens library has no
gradient-pursuit / sparse-coding API. Its intent (sharp vs diffuse) is covered by
1a + 1c.

Usage:  ./.venv/bin/python scripts/06_familiarity.py [model_key] [stage]
  model_key: gemma4b (default) | qwen0.8b | qwen1.7b | pythia70m
  stage:     sweep (default = 6.1) | ...
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import re
import sys

import torch

import jlens
from common import (MODELS, RESULTS_DIR, get_device_dtype, load_hf, load_lens,
                    resolve_single_token)
from swaplib import band_from_fraction

# ---- constants ------------------------------------------------------------

FRAMES = {
    "country": "{X} is a city located in the country of",
    "landmark": "The most famous landmark in {X} is the",
}

# familiarity axis 1: entity frequency. F4 hyper-frequent .. F1 fabricated, F0 noise.
LEVELS = {
    "F4": ["Paris", "Tokyo", "London", "Rome", "Berlin", "Madrid", "Cairo", "Moscow"],
    "F3": ["Tashkent", "Almaty", "Nairobi", "Tbilisi", "Ljubljana", "Windhoek", "Chisinau", "Kigali"],
    "F2": ["Oruro", "Jinja", "Gyumri", "Sokoto", "Bregenz", "Kanazawa", "Timbuktu", "Iquitos"],
    # F1 fabricated (invented, not real places) — script records tokenization
    "F1": ["Malorvia", "Trennsk", "Vantoria", "Qellmin", "Zorbeth", "Dravnia", "Klenor", "Vurstan"],
    # F0 unpronounceable noise floor (control, §6.4)
    "F0": ["Xqzptr", "Zkvwmn", "Qfjklb", "Wrtpxn"],
}
LEVEL_ORDER = ["F4", "F3", "F2", "F1", "F0"]

RESAMPLE_LEVELS = ["F4", "F1"]                       # 2b only at the extremes
RESAMPLE_N = int(os.environ.get("FAMILIARITY_N", "10"))
RESAMPLE_ENTITIES = int(os.environ.get("FAMILIARITY_RE", "5"))  # entities/level for 2b
RESAMPLE_FRAME = "country"
TOPK = 20

REFUSAL = re.compile(
    r"\b(unknown|not a (real|known|city|place)|does ?n'?t exist|fictional|"
    r"made[- ]?up|n'?t (know|sure|exist)|no such|i (don'?t|do not) know|"
    r"unclear|uncertain|not sure|imaginary)\b", re.I)

# ---- load -----------------------------------------------------------------

MODEL_KEY = sys.argv[1] if len(sys.argv) > 1 else "gemma4b"
STAGE = sys.argv[2] if len(sys.argv) > 2 else "sweep"
model_id, lens_file = MODELS[MODEL_KEY]

device, dtype = get_device_dtype()
tok, hf = load_hf(model_id, device, dtype)
lens_model = jlens.from_hf(hf, tok, compile=False)
lens = load_lens(lens_file)
layers = list(lens.source_layers)
band = band_from_fraction(layers, 0.40, 0.90)
EMB = lens_model._embed_tokens.weight  # [vocab, d_model] — semantic neighborhood space

OUT = RESULTS_DIR / "phase6_familiarity"   # Phase 6 artifacts live here
OUT.mkdir(exist_ok=True)

# ---- metrics --------------------------------------------------------------


def decode(tid):
    return tok.decode([int(tid)])


def cohesion(ids):
    """Mean pairwise cosine among the embedding vectors of token ids (1a)."""
    v = EMB[ids].float()
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    S = v @ v.T
    k = v.shape[0]
    return float((S.sum() - k) / (k * (k - 1)))  # mean off-diagonal


def entropy_nats(logits_row):
    p = logits_row.float().softmax(-1)
    return float(-(p * p.clamp_min(1e-12).log()).sum())


def n_tokens(entity):
    return len(tok.encode(entity, add_special_tokens=False))


def entity_metrics(entity, frame_tmpl):
    """One lens.apply -> 1a cohesion, 1c confidence, 2a entropy, 3 sharpest layer."""
    prompt = frame_tmpl.format(X=entity)
    with torch.inference_mode():
        lens_logits, model_logits, _ = lens.apply(
            lens_model, prompt, layers=band, positions=[-1], use_jacobian=True)
    cohs, top1s, margins, coh_by_layer = [], [], [], {}
    for L in band:
        row = lens_logits[L][0]
        ids = row.topk(TOPK).indices.tolist()
        c = cohesion(ids)
        cohs.append(c)
        coh_by_layer[L] = c
        vals = row.float().softmax(-1).topk(2).values
        top1s.append(float(vals[0]))
        margins.append(float(vals[0] - vals[1]))
    if device.type == "mps":
        torch.mps.empty_cache()
    mean = lambda xs: sum(xs) / len(xs)
    return {
        "cohesion": mean(cohs), "top1": mean(top1s), "margin": mean(margins),
        "entropy": entropy_nats(model_logits[0]),
        "sharp_layer": max(coh_by_layer, key=coh_by_layer.get),
        "greedy": decode(int(model_logits[0].argmax())),
        "ntok": n_tokens(entity),
    }


def sample_answer(frame, seed, max_new=4):
    enc = tok(frame, return_tensors="pt").to(device)
    plen = enc.input_ids.shape[1]
    with torch.inference_mode():
        torch.manual_seed(seed)
        out = hf.generate(**enc, max_new_tokens=max_new, do_sample=True,
                          temperature=0.8, top_p=0.95, pad_token_id=tok.eos_token_id)
    if device.type == "mps":
        torch.mps.empty_cache()
    return tok.decode(out[0][plen:], skip_special_tokens=True).strip()


_word_re = re.compile(r"[A-Za-z]+")


def first_word(text):
    m = _word_re.search(text)
    return m.group(0).lower() if m else ""


def resample(entity):
    """N sampled answers for the country frame -> raw list, distinct ratio, refusal rate."""
    frame = FRAMES[RESAMPLE_FRAME].format(X=entity)
    outs = [sample_answer(frame, seed=s) for s in range(RESAMPLE_N)]
    firsts = [first_word(o) for o in outs]
    distinct = len(set(w for w in firsts if w))
    refusals = sum(1 for o in outs if REFUSAL.search(o))
    return {"outs": outs, "firsts": firsts,
            "distinct": distinct, "distinct_ratio": distinct / max(1, len(outs)),
            "refusal_rate": refusals / max(1, len(outs))}


# ---- aggregation & output -------------------------------------------------


def stats(vals):
    n = len(vals)
    m = sum(vals) / n
    sd = (sum((v - m) ** 2 for v in vals) / n) ** 0.5
    return m, sd


def curve_png(level_metrics, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    xs = [lv for lv in LEVEL_ORDER if lv in level_metrics]
    coh = [stats([m["cohesion"] for m in level_metrics[lv]])[0] for lv in xs]
    ent = [stats([m["entropy"] for m in level_metrics[lv]])[0] for lv in xs]
    x = np.arange(len(xs))
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    l1 = ax1.plot(x, coh, "o-", color="#1f77b4", label="neighbor cohesion (lens top-20 cosine)")
    l2 = ax2.plot(x, ent, "s-", color="#d62728", label="output entropy (model next-token)")
    lines = l1 + l2
    ax1.set_xticks(x)
    ax1.set_xticklabels(xs)
    ax1.set_xlabel("familiarity  (F4 hyper-frequent → F1 fabricated → F0 noise)")
    ax1.set_ylabel("neighbor cohesion (0–1)", color="#1f77b4")
    ax2.set_ylabel("output entropy (nats)", color="#d62728")
    ax1.set_title(f"Familiarity vs J-space representation — {MODEL_KEY} "
                  f"(mean over entities × {len(FRAMES)} frames)")
    ax1.legend(lines, [l.get_label() for l in lines], loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_sweep():
    print(f"=== Phase 6.1 familiarity sweep — {MODEL_KEY} ===", flush=True)
    # per-entity cheap metrics, averaged over frames
    level_metrics = {}   # level -> [per-entity dict]
    per_rows = {}        # level -> list of (entity, metrics-per-frame)
    for lv in LEVEL_ORDER:
        level_metrics[lv] = []
        per_rows[lv] = []
        for ent in LEVELS[lv]:
            fms = {fk: entity_metrics(ent, ft) for fk, ft in FRAMES.items()}
            avg = {k: sum(fms[fk][k] for fk in FRAMES) / len(FRAMES)
                   for k in ("cohesion", "top1", "margin", "entropy")}
            avg["ntok"] = fms["country"]["ntok"]
            avg["greedy_country"] = fms["country"]["greedy"]
            avg["greedy_landmark"] = fms["landmark"]["greedy"]
            avg["sharp_layer"] = fms["country"]["sharp_layer"]
            level_metrics[lv].append(avg)
            per_rows[lv].append((ent, avg))
            print(f"  [{lv}] {ent:12s} coh={avg['cohesion']:.3f} ent={avg['entropy']:.2f} "
                  f"top1={avg['top1']:.3f} → {avg['greedy_country']!r}", flush=True)

    # resample (2b/2c) at extremes
    resamples = {}
    for lv in RESAMPLE_LEVELS:
        for ent in LEVELS[lv][:RESAMPLE_ENTITIES]:
            print(f"  [resample {lv}] {ent} (N={RESAMPLE_N}) …", flush=True)
            resamples[(lv, ent)] = resample(ent)

    # ---- write md ----
    L = [f"# 친숙도 vs J-space 표현 — {MODEL_KEY} (`{model_id}`)", ""]
    L.append(f"- 워크스페이스 대역(40–90%): layers {band[0]}–{band[-1]} ({len(band)}층) / 전체 {len(layers)}층")
    L.append(f"- 프레임: " + " · ".join(f"`{v}`" for v in FRAMES.values()))
    L.append("- 지표: 1a 이웃 응집도(렌즈 top-20 임베딩 코사인, 높음=뭉친 이웃) · 1c 렌즈 확신도 · "
             "2a 출력 엔트로피(모델 다음토큰, 높음=넓음) · 3 최응집 레이어.")
    L.append("- **지표 1b(희소분해)는 생략** — jlens에 gradient-pursuit API 없음. 의도는 1a+1c로 커버.")
    L.append("- 세 가설(H-수렴/방황/거부)은 **열어둠**. 2b·2c는 F4·F1 양끝만.")
    L.append("")
    # per-level summary
    L.append("## 단계별 요약 (평균±표준편차)")
    L.append("| 단계 | 개체수 | 이웃 응집도(1a) | 출력 엔트로피(2a) | 렌즈 top1(1c) | 마진 | 토큰수 |")
    L.append("|---|---|---|---|---|---|---|")
    for lv in LEVEL_ORDER:
        ms = level_metrics[lv]
        cm, cs = stats([m["cohesion"] for m in ms])
        em, es = stats([m["entropy"] for m in ms])
        tm, ts = stats([m["top1"] for m in ms])
        gm, gs = stats([m["margin"] for m in ms])
        nm, _ = stats([m["ntok"] for m in ms])
        L.append(f"| {lv} | {len(ms)} | {cm:.3f}±{cs:.3f} | {em:.2f}±{es:.2f} | "
                 f"{tm:.3f}±{ts:.3f} | {gm:.3f} | {nm:.1f} |")
    L.append("")
    # per-entity detail
    for lv in LEVEL_ORDER:
        L.append(f"### {lv} 개체별")
        L.append("| 개체 | 토큰 | 응집도 | 엔트로피 | top1 | greedy(country) | greedy(landmark) | 최응집층 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for ent, a in per_rows[lv]:
            L.append(f"| {ent} | {a['ntok']} | {a['cohesion']:.3f} | {a['entropy']:.2f} | "
                     f"{a['top1']:.3f} | {a['greedy_country']!r} | {a['greedy_landmark']!r} | {a['sharp_layer']} |")
        L.append("")
    # resample section
    L.append("## 재샘플 (2b 다양성 · 2c 거부) — F4·F1 양끝")
    L.append(f"프레임 `{FRAMES[RESAMPLE_FRAME]}`, 샘플링 T=0.8/top_p=0.95, N={RESAMPLE_N}.")
    L.append("")
    for (lv, ent), r in resamples.items():
        L.append(f"### [{lv}] {ent} — distinct {r['distinct']}/{RESAMPLE_N} "
                 f"(비율 {r['distinct_ratio']:.2f}), 거부율 {r['refusal_rate']:.2f}")
        L.append("<details><summary>원문 N개</summary>\n")
        for i, o in enumerate(r["outs"]):
            L.append(f"{i+1}. `{o}`")
        L.append("\n</details>\n")

    out = OUT / f"familiarity_freq_{MODEL_KEY}.md"
    out.write_text("\n".join(L) + "\n")
    png = OUT / f"familiarity_curve_{MODEL_KEY}.png"
    curve_png(level_metrics, png)
    print(f"\nWrote {out}\nWrote {png}")

    # console: checkpoint summary
    print("\n===== 체크포인트 요약 =====")
    for lv in LEVEL_ORDER:
        ms = level_metrics[lv]
        cm = stats([m["cohesion"] for m in ms])[0]
        em = stats([m["entropy"] for m in ms])[0]
        print(f"  {lv}: 응집도 {cm:.3f} | 엔트로피 {em:.2f}")
    print("\n  F1 재샘플 (distinct/N, 거부율):")
    for (lv, ent), r in resamples.items():
        if lv == "F1":
            print(f"    {ent}: {r['distinct']}/{RESAMPLE_N}, 거부 {r['refusal_rate']:.2f} | "
                  f"답들: {[first_word(o) for o in r['outs']]}")


if STAGE in ("sweep", "all"):
    run_sweep()
else:
    print(f"unknown stage: {STAGE}  (use: sweep)")
