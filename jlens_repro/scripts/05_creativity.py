"""Phase 5 — creativity vs collapse tradeoff via workspace steering.

Inject an *unrelated* single-token concept into the workspace band (40-90% depth)
during GREEDY generation, sweeping strength alpha, and measure where the output
goes from (1) no-change -> (2) creative fusion (concept seeps in, sentence still
holds) -> (3) collapse (grammar/meaning breaks).

Operational definition only: we measure "how much the injected concept seeps into
the output while it still reads as language" — NOT aesthetic creativity. Every raw
output is saved verbatim so a human can judge grammar/fusion directly.

Metrics per (prompt x concept x alpha):
  1. concept pull   = fraction of output words in the concept's fixed associate list
                      (+ secondary: mean logprob the UN-steered model assigns to the
                       concept token across the continuation, teacher-forced)
  2. coherence      = perplexity of the continuation under the UN-steered model
                      (exp mean NLL, teacher-forced) + 3-gram repetition ratio
  3. raw text       = saved verbatim

Reuses everything from common.py / swaplib.py — no new primitives. Steering is the
existing `dir_hooks(..., "steer")` (h <- h + alpha * mean||h|| * v_w), applied around
`hf.generate` so it fires at every position and every decode step.

Usage:  ./.venv/bin/python scripts/05_creativity.py [model_key] [stage] [alpha]
  model_key: qwen0.8b (default) | qwen1.7b | gemma4b | pythia70m
  stage:     sweep (default = Phase 5.1) | region (5.2) | control (5.3) |
             report (5.4) | all
  alpha:     only for `region` (fixed steering strength; default 4)
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import re
import sys
from contextlib import contextmanager

import torch

import jlens
from common import (MODELS, RESULTS_DIR, get_device_dtype, load_hf, load_lens,
                    resolve_single_token)
from swaplib import band_from_fraction, concept_dirs

# ---- experiment constants -------------------------------------------------

ALL_PROMPTS = {
    "P1_poem": "Write a short four-line poem about a morning.",
    "P2_house": "Describe a house in three sentences.",
}
# CREATIVITY_PROMPTS="P1_poem" restricts the set (focused runs on slow models).
_psel = os.environ.get("CREATIVITY_PROMPTS")
PROMPTS = {k: ALL_PROMPTS[k] for k in _psel.split(",")} if _psel else dict(ALL_PROMPTS)

# injection concept candidates (base-prompt-irrelevant); single-token ones are used
CONCEPT_CANDIDATES = ["ocean", "fire", "machine", "grief", "gold"]

# fixed associate lists (5-10 words) for the concept-pull frequency metric
ASSOC = {
    "ocean": ["ocean", "sea", "wave", "tide", "water", "salt", "shore", "deep", "blue", "current"],
    "fire": ["fire", "flame", "burn", "heat", "smoke", "ash", "ember", "blaze", "spark", "warm"],
    "machine": ["machine", "engine", "gear", "metal", "steel", "motor", "robot", "mechanical", "circuit", "iron"],
    "grief": ["grief", "sorrow", "loss", "tears", "mourn", "sad", "ache", "weep", "empty", "cold"],
    "gold": ["gold", "golden", "shine", "glow", "treasure", "coin", "bright", "amber", "gleam", "rich"],
}

# fine low-α grid (coarse {0,0.5,1,2,4,8,16} saturated at 0.5 on qwen0.8b).
# override: CREATIVITY_ALPHAS="0,0.05,0.1,0.2,0.35"
_asel = os.environ.get("CREATIVITY_ALPHAS")
ALPHAS = [float(x) for x in _asel.split(",")] if _asel else [0, 0.05, 0.1, 0.2, 0.35, 0.5]
MAX_NEW = int(os.environ.get("CREATIVITY_MAXNEW", "40"))  # shorter on slow models (gemma)
N_CONCEPTS = int(os.environ.get("CREATIVITY_NCONCEPTS", "3"))  # fewer concepts on slow models

# ---- method knobs (Phase 5 improvement: break the repetition attractor) ----
# STEER_SCOPE: "all" = inject at every generated position (baseline; runs away into a
#   repeat loop). "prompt" = inject only on the prompt forward — with KV cache each
#   generation step has seq_len==1, so we gate on that and steer only the prompt.
# DECODE: "greedy" (baseline) vs "sample" (temperature/top-p, seeded) to break the loop.
STEER_SCOPE = os.environ.get("CREATIVITY_SCOPE", "all")   # all | prompt
DECODE = os.environ.get("CREATIVITY_DECODE", "greedy")     # greedy | sample
TEMP = float(os.environ.get("CREATIVITY_TEMP", "0.8"))
TOPP = float(os.environ.get("CREATIVITY_TOPP", "0.95"))
SEED = int(os.environ.get("CREATIVITY_SEED", "0"))
# non-default variants get a filename/title tag so runs don't clobber the baseline
VARIANT = "" if (STEER_SCOPE == "all" and DECODE == "greedy") else f"_{STEER_SCOPE}-{DECODE}"

OUT = RESULTS_DIR / "phase5_creativity"   # Phase 5 artifacts live here
OUT.mkdir(exist_ok=True)

# ---- load -----------------------------------------------------------------

MODEL_KEY = sys.argv[1] if len(sys.argv) > 1 else "qwen0.8b"
STAGE = sys.argv[2] if len(sys.argv) > 2 else "sweep"
model_id, lens_file = MODELS[MODEL_KEY]

device, dtype = get_device_dtype()
tok, hf = load_hf(model_id, device, dtype)
lens_model = jlens.from_hf(hf, tok, compile=False)
lens = load_lens(lens_file)
layers = list(lens.source_layers)
band = band_from_fraction(layers, 0.40, 0.90)


def single_id(word):
    return resolve_single_token(tok, " " + word) or resolve_single_token(tok, word)


# pick single-token concepts, record ids
CONCEPTS = []
CONCEPT_IDS = {}
for c in CONCEPT_CANDIDATES:
    tid = single_id(c)
    if tid is not None:
        CONCEPTS.append(c)
        CONCEPT_IDS[c] = tid
    if len(CONCEPTS) >= N_CONCEPTS:
        break

# ---- generation + metrics -------------------------------------------------


def random_dirs(band_layers, d_model, seed0=7000):
    """Deterministic per-layer random unit vectors (control), same shape as concept_dirs."""
    out = {}
    for L in band_layers:
        g = torch.Generator().manual_seed(seed0 + L)
        r = torch.randn(d_model, generator=g)
        out[L] = (r / r.norm()).to(device)
    return out


@contextmanager
def steer_scoped(dirs, alpha):
    """h += alpha * mean||h|| * v over band layers. If STEER_SCOPE=='prompt', skip
    generation steps (seq_len==1 under KV cache) so only the prompt is steered."""
    handles = []

    def mk(v):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            if STEER_SCOPE == "prompt" and h.shape[1] == 1:
                return out
            od = h.dtype
            x = h.float()
            x = x + alpha * x.norm(dim=-1, keepdim=True).mean() * v
            x = x.to(od)
            return (x, *out[1:]) if isinstance(out, tuple) else x
        return hook

    try:
        for L, v in dirs.items():
            handles.append(lens_model.layers[L].register_forward_hook(mk(v)))
        yield
    finally:
        for h in handles:
            h.remove()


def steered_generate(prompt, dirs, alpha):
    """Generate with steering active per STEER_SCOPE/DECODE. Returns (full ids, prompt len).
    alpha==0 or dirs None -> baseline (no hooks). Sampling is seeded for reproducibility."""
    enc = tok(prompt, return_tensors="pt").to(device)
    plen = enc.input_ids.shape[1]
    gkw = dict(max_new_tokens=MAX_NEW, pad_token_id=tok.eos_token_id)
    if DECODE == "sample":
        gkw.update(do_sample=True, temperature=TEMP, top_p=TOPP)
    else:
        gkw.update(do_sample=False)
    with torch.inference_mode():
        if DECODE == "sample":
            torch.manual_seed(SEED)
        if dirs is None or alpha == 0:
            out = hf.generate(**enc, **gkw)
        else:
            with steer_scoped(dirs, alpha):
                out = hf.generate(**enc, **gkw)
    if device.type == "mps":
        torch.mps.empty_cache()
    return out[0], plen


def base_metrics(full_ids, plen, concept_id):
    """Teacher-force the (already generated) sequence through the UN-steered model.
    Returns (ppl, concept_logprob_mean, n_cont) over continuation positions."""
    if full_ids.shape[0] <= plen:
        return float("nan"), float("nan"), 0
    with torch.inference_mode():
        logits = hf(full_ids.unsqueeze(0)).logits[0].float()  # [T, V]
    logp = torch.log_softmax(logits, dim=-1)
    # predict token t from position t-1; continuation tokens are plen..T-1
    nlls, cids = [], []
    for t in range(plen, full_ids.shape[0]):
        row = logp[t - 1]
        nlls.append(-row[full_ids[t]].item())
        cids.append(row[concept_id].item())
    ppl = float(torch.tensor(nlls).mean().exp()) if nlls else float("nan")
    clp = float(torch.tensor(cids).mean()) if cids else float("nan")
    return ppl, clp, len(nlls)


_word_re = re.compile(r"[a-zA-Z']+")


def concept_ratio(text, assoc):
    words = [w.lower() for w in _word_re.findall(text)]
    if not words:
        return 0.0, 0
    aset = set(a.lower() for a in assoc)
    hits = sum(1 for w in words if w in aset)
    return hits / len(words), hits


def rep_3gram(cont_ids):
    ids = cont_ids.tolist()
    if len(ids) < 3:
        return 0.0
    grams = [tuple(ids[i:i + 3]) for i in range(len(ids) - 2)]
    return 1.0 - len(set(grams)) / len(grams)


def decode_cont(full_ids, plen):
    return tok.decode(full_ids[plen:], skip_special_tokens=True).strip()


def run_one(prompt, dirs, alpha, concept):
    full, plen = steered_generate(prompt, dirs, alpha)
    text = decode_cont(full, plen)
    ratio, hits = concept_ratio(text, ASSOC[concept])
    ppl, clp, ncont = base_metrics(full, plen, CONCEPT_IDS[concept])
    rep = rep_3gram(full[plen:])
    return {"alpha": alpha, "text": text, "ratio": ratio, "hits": hits,
            "ppl": ppl, "clp": clp, "rep": rep, "ncont": ncont}


# ---- sweep (Phase 5.1) ----------------------------------------------------


def do_sweep(dirs_fn, tag):
    """dirs_fn(concept) -> {layer: v} or None. tag labels the run (concept/random)."""
    results = {}  # (concept, prompt_key) -> [row per alpha]
    total = len(CONCEPTS) * len(PROMPTS) * len(ALPHAS)
    i = 0
    for concept in CONCEPTS:
        dirs = dirs_fn(concept)
        for pk, prompt in PROMPTS.items():
            rows = []
            for a in ALPHAS:
                i += 1
                print(f"[{tag}] {i}/{total}  {concept} x {pk} x a={a}", flush=True)
                rows.append(run_one(prompt, dirs, a, concept))
            results[(concept, pk)] = rows
    return results


def curve_png(concept_res, random_res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def agg(res, key):
        return np.array([np.mean([res[(c, pk)][ai][key] for c in CONCEPTS for pk in PROMPTS])
                         for ai in range(len(ALPHAS))])

    x = np.arange(len(ALPHAS))
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    # left axis (0-1): concept pull + repetition ratio; right axis: perplexity
    l1 = ax1.plot(x, agg(concept_res, "ratio"), "o-", color="#1f77b4", label="concept pull (assoc-word ratio)")
    lr = ax1.plot(x, agg(concept_res, "rep"), "^-", color="#2ca02c", label="3-gram repetition (collapse)")
    l2 = ax2.plot(x, agg(concept_res, "ppl"), "s-", color="#d62728", label="perplexity (base model)")
    lines = l1 + lr + l2
    if random_res is not None:
        lines += ax1.plot(x, agg(random_res, "ratio"), "o--", color="#7fbfff", alpha=.8, label="concept pull — RANDOM dir")
        lines += ax1.plot(x, agg(random_res, "rep"), "^--", color="#98df8a", alpha=.8, label="repetition — RANDOM dir")
        lines += ax2.plot(x, agg(random_res, "ppl"), "s--", color="#ff9896", alpha=.8, label="perplexity — RANDOM dir")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(a) for a in ALPHAS])
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_xlabel("steering strength α  (h += α·‖h‖·v)")
    ax1.set_ylabel("concept pull / repetition  (0–1)", color="#1f77b4")
    ax2.set_ylabel("perplexity (base model, teacher-forced)", color="#d62728")
    ax1.set_title(f"Creativity↔collapse sweep — {MODEL_KEY}  (mean over {len(CONCEPTS)} concepts × {len(PROMPTS)} prompts)")
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fmt_rows(rows):
    out = ["| α | concept pull | hits | ppl | rep-3gram | concept logprob |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['alpha']} | {r['ratio']:.3f} | {r['hits']} | {r['ppl']:.1f} "
                   f"| {r['rep']:.2f} | {r['clp']:.2f} |")
    return out


def write_sweep_md(concept_res):
    method = f"scope={STEER_SCOPE}, decode={DECODE}"
    if DECODE == "sample":
        method += f" (T={TEMP}, top_p={TOPP}, seed={SEED})"
    lines = [f"# 창의성↔붕괴 스티어링 α 스윕 — {MODEL_KEY} (`{model_id}`)", ""]
    lines.append(f"- **방법: {method}**  (scope=prompt=프롬프트 구간에만 주입, decode=sample=샘플링)")
    lines.append(f"- 워크스페이스 대역(40–90%): layers {band[0]}–{band[-1]} ({len(band)}층) / "
                 f"전체 {len(layers)}층")
    lines.append(f"- 주입 개념(단일토큰): " + ", ".join(f"`{c}`(id {CONCEPT_IDS[c]})" for c in CONCEPTS))
    skipped = [c for c in CONCEPT_CANDIDATES if c not in CONCEPTS]
    if skipped:
        lines.append(f"- 다중토큰이라 제외: {', '.join(skipped)}")
    lines.append(f"- α ∈ {ALPHAS}, greedy, max_new_tokens={MAX_NEW}")
    lines.append("- **개념 반영도** = 출력 단어 중 개념 연관어 비율. **일관성** = 무개입 모델 perplexity "
                 "(높을수록 붕괴). rep-3gram = 3-그램 반복률.")
    lines.append("")
    lines.append("> 조작적 정의만 측정 — 미학적 창의성 아님. 원문을 그대로 실어 독자가 직접 판정.")
    lines.append("")
    for concept in CONCEPTS:
        lines.append(f"## 개념 `{concept}`")
        for pk in PROMPTS:
            lines.append(f"\n### {pk}: \"{PROMPTS[pk]}\"")
            lines += fmt_rows(concept_res[(concept, pk)])
            lines.append("\n<details><summary>원문(α별)</summary>\n")
            for r in concept_res[(concept, pk)]:
                lines.append(f"**α={r['alpha']}**  ")
                lines.append("```")
                lines.append(r["text"] or "(빈 출력)")
                lines.append("```")
            lines.append("\n</details>")
        lines.append("")
    return lines


REP_OK = 0.30  # above this 3-gram repetition ratio the output is a degenerate loop


def representative(concept_res):
    """Auto-pick 3 outputs: no-change (α=0), fusion (concept in AND still coherent —
    low repetition), collapse (degenerate loop). A repeat loop has LOW perplexity, so
    fusion must be gated on repetition, not ppl (the perplexity trap)."""
    flat = [r | {"concept": c, "pk": pk}
            for (c, pk), rows in concept_res.items() for r in rows]
    no_change = next(r for r in flat if r["alpha"] == 0)
    valid = [r for r in flat if r["ppl"] == r["ppl"]]  # not nan
    # collapse = most degenerate: highest repetition, then highest alpha
    collapse = max(valid, key=lambda r: (r["rep"], r["alpha"]))
    # fusion = concept present AND still reads as language (repetition near baseline)
    fused = [r for r in valid if r["alpha"] > 0 and r["ratio"] > 0 and r["rep"] < REP_OK]
    fusion = min(fused, key=lambda r: r["ppl"]) if fused else None
    return no_change, fusion, collapse


# ---- main -----------------------------------------------------------------

if STAGE in ("sweep", "all"):
    print(f"=== Phase 5.1 sweep — {MODEL_KEY} — concepts {CONCEPTS} ===", flush=True)
    concept_res = do_sweep(lambda c: concept_dirs(lens, lens_model, band, CONCEPT_IDS[c], device),
                           "concept")
    md = write_sweep_md(concept_res)
    out = OUT / f"creativity_sweep_{MODEL_KEY}{VARIANT}.md"
    out.write_text("\n".join(md) + "\n")
    png = OUT / f"creativity_curve_{MODEL_KEY}{VARIANT}.png"
    curve_png(concept_res, None, png)
    print(f"\nWrote {out}\nWrote {png}")

    nc, fu, co = representative(concept_res)
    print("\n===== 대표 출력 3개 =====")
    print(f"\n[무변화]  {nc['concept']} α=0  ppl={nc['ppl']:.1f}  pull={nc['ratio']:.3f}\n  {nc['text']!r}")
    if fu:
        print(f"\n[융합후보] {fu['concept']} α={fu['alpha']}  ppl={fu['ppl']:.1f}  pull={fu['ratio']:.3f}\n  {fu['text']!r}")
    else:
        print("\n[융합후보] 없음 — 개념이 배어든 저-perplexity 출력 미관찰 (무변화→붕괴 점프 가능)")
    print(f"\n[붕괴]    {co['concept']} α={co['alpha']}  ppl={co['ppl']:.1f}  pull={co['ratio']:.3f}\n  {co['text']!r}")
    print("\n(Phase 5.1 완료. 곡선 PNG + 위 3개 확인 후 region/control/report 진행.)")

elif STAGE == "control":
    print(f"=== Phase 5.3 control (random direction) — {MODEL_KEY} ===", flush=True)
    d_model = lens_model.d_model
    concept_res = do_sweep(lambda c: concept_dirs(lens, lens_model, band, CONCEPT_IDS[c], device),
                           "concept")
    random_res = do_sweep(lambda c: random_dirs(band, d_model), "random")
    png = OUT / f"creativity_curve_{MODEL_KEY}{VARIANT}.png"
    curve_png(concept_res, random_res, png)
    lines = [f"# 대조군: 개념방향 vs 무작위방향 — {MODEL_KEY}", ""]
    lines.append("무작위 단위벡터(동일 노름)로 동일 α 스윕. 개념 반영 없이 perplexity만 오르면 "
                 "= 붕괴만 (방향이 의미를 만든다는 증거).")
    lines.append("")
    for concept in CONCEPTS:
        lines.append(f"## `{concept}` — 무작위 방향")
        for pk in PROMPTS:
            lines.append(f"\n### {pk}")
            lines += fmt_rows(random_res[(concept, pk)])
        lines.append("")
    out = OUT / f"creativity_control_{MODEL_KEY}{VARIANT}.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out}\nWrote {png} (concept vs random overlay)")

elif STAGE == "region":
    ralpha = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
    print(f"=== Phase 5.2 region comparison — {MODEL_KEY} — α={ralpha} ===", flush=True)
    regions = {
        "early(0-30%)": band_from_fraction(layers, 0.0, 0.30),
        "workspace(40-90%)": band_from_fraction(layers, 0.40, 0.90),
        "late(90-100%)": band_from_fraction(layers, 0.90, 1.0),
    }
    lines = [f"# 개입 위치별 비교 — {MODEL_KEY} (α={ralpha})", "",
             "예측: workspace에서만 융합, early는 미약, late는 끝에 개념어만 튀어나옴.", ""]
    for concept in CONCEPTS:
        lines.append(f"## `{concept}`")
        for rname, rlayers in regions.items():
            dirs = concept_dirs(lens, lens_model, rlayers, CONCEPT_IDS[concept], device)
            lines.append(f"\n### {rname}  (layers {rlayers[0]}–{rlayers[-1]})")
            rows = []
            for pk, prompt in PROMPTS.items():
                r = run_one(prompt, dirs, ralpha, concept)
                rows.append(r)
                lines.append(f"- **{pk}** pull={r['ratio']:.3f} ppl={r['ppl']:.1f} rep={r['rep']:.2f}  ")
                lines.append(f"  ```\n  {r['text']}\n  ```")
        lines.append("")
    out = OUT / f"creativity_by_region_{MODEL_KEY}{VARIANT}.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out}")

else:
    print(f"unknown stage: {STAGE}  (use sweep|region|control|report|all)")
