"""Phase 4 (viz) — build a single self-contained results/REPORT.html with heatmaps.

A: 'spider' lens rank vs normalized depth, 3 models (parsed from reading_B_*.md).
B: swap flip P(target) vs normalized start-layer, 3 models (parsed from sweep md).
C: gemma-3-4b layer x token-position 'spider' rank slice (recomputed live).

Markdown reports are left untouched; this only adds REPORT.html.
"""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import jlens
from common import MODELS, RESULTS_DIR, get_device_dtype, load_hf, load_lens, token_variants

MODELS_ORDER = [("qwen0.8b", "Qwen3.5-0.8B"), ("qwen1.7b", "Qwen3-1.7B"), ("gemma4b", "Gemma-3-4B")]
SPIDER_PROMPT = "The number of legs on the animal that spins webs is"


def png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def parse_reading_B(key):
    """Return (layers array, spider-rank array) from reading_B_<key>.md table."""
    txt = (RESULTS_DIR / f"reading_B_{key}.md").read_text().splitlines()
    layers, ranks = [], []
    for line in txt:
        m = re.match(r"\|\s*(\d+)\s*\|.*\|\s*([\d-]+)\s*\|\s*$", line)
        if m:
            layers.append(int(m.group(1)))
            r = m.group(2)
            ranks.append(float(r) if r.isdigit() else np.nan)
    return np.array(layers), np.array(ranks)


def parse_sweep(key):
    """Return (start-layer array, normalized depth %, P(target) array)."""
    txt = (RESULTS_DIR / f"swap_layer_sweep_{key}.md").read_text().splitlines()
    S, pct, P = [], [], []
    for line in txt:
        # | 28 (84%) | 28–32 | ' London' | 0.953 | YES |
        m = re.match(r"\|\s*(\d+)\s*\((\d+)%\)\s*\|[^|]*\|[^|]*\|\s*([\d.]+)\s*\|", line)
        if m:
            S.append(int(m.group(1)))
            pct.append(int(m.group(2)))
            P.append(float(m.group(3)))
    return np.array(S), np.array(pct), np.array(P)


def resample(x_frac, y, grid):
    """Interpolate y onto common [0,1] grid (nan-safe, works on log-rank)."""
    ok = ~np.isnan(y)
    return np.interp(grid, x_frac[ok], y[ok])


# ---------------------------------------------------------------- Heatmap A
def heatmap_A():
    grid = np.linspace(0, 1, 30)
    rows, labels, mins = [], [], []
    for key, name in MODELS_ORDER:
        L, R = parse_reading_B(key)
        frac = L / L.max()
        logr = np.log10(np.where(np.isnan(R), 1e6, R))
        rows.append(resample(frac, logr, grid))
        labels.append(name)
        mins.append((int(np.nanmin(R)), int(L[np.nanargmin(R)]), L.max()))
    M = np.vstack(rows)
    fig, ax = plt.subplots(figsize=(9, 2.6))
    im = ax.imshow(M, aspect="auto", cmap="viridis_r", vmin=0, vmax=5,
                   extent=[0, 100, len(labels) - 0.5, -0.5])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("normalized depth (%)  —  early → late")
    ax.set_title("'spider' lens rank across layers (lower = surfaced; bright = good)")
    for i, (mn, lyr, nl) in enumerate(mins):
        ax.text(100 * lyr / nl, i, f"  r{mn}", va="center", ha="left",
                color="white", fontsize=9, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("log10(rank)")
    return png_b64(fig)


# ---------------------------------------------------------------- Heatmap B
def heatmap_B():
    grid = np.linspace(0, 1, 30)
    rows, labels = [], []
    for key, name in MODELS_ORDER:
        S, pct, P = parse_sweep(key)
        frac = pct / 100.0
        rows.append(resample(frac, P, grid))
        labels.append(name)
    M = np.vstack(rows)
    fig, ax = plt.subplots(figsize=(9, 2.6))
    im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=1,
                   extent=[0, 100, len(labels) - 0.5, -0.5])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("swap-window start depth (%)")
    ax.set_title("Swap flip strength P(target=London), Paris→London, α=2")
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("P(target)")
    return png_b64(fig)


# ---------------------------------------------------------------- Heatmap C (live)
def heatmap_C():
    device, dtype = get_device_dtype()
    model_id, lens_file = MODELS["gemma4b"]
    tok, hf = load_hf(model_id, device, dtype)
    lens_model = jlens.from_hf(hf, tok, compile=False)
    lens = load_lens(lens_file)
    layers = list(lens.source_layers)
    spider_ids = token_variants(tok, ["spider"])["spider"]

    with torch.inference_mode():
        lens_logits, _, input_ids = lens.apply(
            lens_model, SPIDER_PROMPT, layers=layers, positions=None, use_jacobian=True
        )
    ids = input_ids[0].tolist()
    toks = [tok.decode([i]) for i in ids]
    n_pos = len(ids)
    M = np.zeros((len(layers), n_pos))
    for li, L in enumerate(layers):
        logits = lens_logits[L]  # [n_pos, vocab]
        best = np.full(n_pos, np.inf)
        for s in spider_ids:
            r = (logits > logits[:, s:s + 1]).sum(dim=1).cpu().numpy() + 1
            best = np.minimum(best, r)
        M[li] = np.log10(best)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(M, aspect="auto", cmap="viridis_r", vmin=0, vmax=5, origin="lower")
    ax.set_xticks(range(n_pos))
    ax.set_xticklabels([repr(t) for t in toks], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(0, len(layers), 2))
    ax.set_yticklabels(layers[::2])
    ax.set_ylabel("layer (early → late)")
    ax.set_xlabel("token position")
    ax.set_title("Gemma-3-4B: 'spider' lens rank per (layer × position)\n"
                 "bright = spider surfaces (never in the prompt)")
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("log10(rank of 'spider')")
    return png_b64(fig)


# ---------------------------------------------------------------- assemble HTML
def build_html(a, b, c):
    style = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:2rem auto;
         padding:0 1rem;line-height:1.55;color:#1a1a1a;background:#fff}
    h1{border-bottom:3px solid #333;padding-bottom:.3rem}
    h2{margin-top:2.2rem;color:#222}
    img{max-width:100%;border:1px solid #ddd;border-radius:6px;margin:.6rem 0}
    table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.92rem}
    th,td{border:1px solid #ccc;padding:.35rem .6rem;text-align:left}
    th{background:#f2f2f2}
    .good{color:#0a7d28;font-weight:600}.bad{color:#b00020;font-weight:600}
    .cap{color:#555;font-size:.88rem;margin:.2rem 0 1.2rem}
    code{background:#f4f4f4;padding:.1rem .3rem;border-radius:3px}
    """
    tbl = """
    <table>
    <tr><th>지표</th><th>Qwen3.5-0.8B</th><th>Qwen3-1.7B</th><th>Gemma-3-4B</th></tr>
    <tr><td>'spider' 최고 렌즈 rank (Exp B)</td><td>750 @L7</td><td>317 @L14</td>
        <td class="good">2 @L28 (재현)</td></tr>
    <tr><td>거미=8다리 사실 (greedy)</td><td>'10' ✗</td><td>'4' ✗</td><td class="good">'8' ✓</td></tr>
    <tr><td>지정 8→6 스왑</td><td colspan="2">baseline 부재로 불가</td>
        <td class="bad">미재현 (spider 표면화)</td></tr>
    <tr><td>타깃 후반 창 flip P(London)</td><td>0.78 (86%)</td><td>0.91 (81%)</td>
        <td class="good">0.99 (84–90%)</td></tr>
    <tr><td>무작위·초반 대조군</td><td>통과</td><td>통과</td><td>통과</td></tr>
    <tr><td>directed-modulation (Exp A)</td><td colspan="3">전 규모 비결정적</td></tr>
    </table>"""
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>J-lens 로컬 재현 — 시각 리포트</title><style>{style}</style></head><body>
<h1>J-lens 로컬 재현 — 시각 리포트</h1>
<p>Apple Silicon(M2, MPS)에서 <code>neuronpedia/jacobian-lens</code> 기학습 렌즈 로드.
규모 사다리 <b>0.8B → 1.7B → 4B</b>. 마크다운 원본은 <code>REPORT.md</code>·
<code>scale_compare.md</code> 참조.</p>

<h2>핵심 비교</h2>{tbl}

<h2>C. Gemma-3-4B 레이어×위치 슬라이스 — 'spider' rank</h2>
<img src="data:image/png;base64,{c}">
<p class="cap">프롬프트에 없는 'spider'가 중후반 레이어(밝은 영역)에서 렌즈 top으로 부상 →
말하지 않은 브리지 개념의 판독. 논문의 sensory→workspace→motor 진행에 해당.</p>

<h2>A. 규모별 'spider' 판독 (Exp B)</h2>
<img src="data:image/png;base64,{a}">
<p class="cap">깊이축을 따라 'spider' 렌즈 rank. 4B 행만 중후반에서 밝아짐(rank↓) →
읽기 현상이 규모와 함께 발현(rank 750→317→2).</p>

<h2>B. 스왑 flip 강도 (Exp D, Paris→London)</h2>
<img src="data:image/png;base64,{b}">
<p class="cap">스왑 창의 시작 깊이별 P(target). 세 모델 모두 <b>후반 ~80–90% 깊이</b>에서만
출력이 뒤집히며, 규모가 클수록 더 깨끗(P 0.78→0.91→0.99).</p>

<h2>정직성 노트</h2>
<p>되는 것: 내부추론 읽기(Exp B, 4B) · 인과 스왑 메커니즘(후반 국소 창, 대조군 통과).
안 되는 것: directed-modulation(Exp A, 전 규모) · 지정 8→6 스왑(전 규모).
소형/중형 모델 결과이므로 프런티어급 논문 결과와 동일시하지 않음.</p>
</body></html>"""


if __name__ == "__main__":
    print("Heatmap A (parse reading_B)...")
    a = heatmap_A()
    print("Heatmap B (parse sweep)...")
    b = heatmap_B()
    print("Heatmap C (reload gemma, live slice)...")
    c = heatmap_C()
    out = RESULTS_DIR / "REPORT.html"
    out.write_text(build_html(a, b, c))
    print(f"Wrote {out} ({out.stat().st_size/1024:.0f} KB)")
