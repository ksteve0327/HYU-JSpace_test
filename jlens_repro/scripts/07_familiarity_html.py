"""Build a self-contained HTML report for Phase 6 (familiarity vs J-space).

Embeds matplotlib charts as base64 + the design/results narrative. Data are the
final gemma-3-4b numbers from `familiarity_freq_gemma4b.md` (hardcoded — no model
needed, so this is fast). Writes `results/phase6_familiarity/FAMILIARITY_REPORT.html`.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"

# ---- final gemma-3-4b data -------------------------------------------------
LEVELS = ["F4", "F3", "F2", "F1", "F0"]
LEVEL_LABEL = {"F4": "F4 초고빈도", "F3": "F3 중빈도", "F2": "F2 무명",
               "F1": "F1 가공", "F0": "F0 무의미"}
COH = [0.320, 0.309, 0.314, 0.303, 0.308]
COH_SD = [0.014, 0.014, 0.016, 0.012, 0.002]
ENT = [1.73, 2.29, 2.55, 5.89, 5.92]
ENT_SD = [0.49, 0.29, 0.56, 0.12, 0.10]
TOP1 = [0.804, 0.743, 0.675, 0.751, 0.669]

F4_DISTINCT = {"Paris": 1, "Tokyo": 1, "London": 3, "Rome": 1, "Berlin": 1}
F1_DISTINCT = {"Malorvia": 8, "Trennsk": 9, "Vantoria": 8, "Qellmin": 8, "Zorbeth": 10}

RESAMP = {
    "Paris (F4)": ["France"] * 10,
    "Tokyo (F4)": ["Japan"] * 10,
    "Trennsk (F1)": ["Belgium", "Germany", "Germany", "Austria", "Peru", "Saxony",
                     "Trenska", "Namibia", "Denmark", "strong"],
    "Malorvia (F1)": ["Faldor", "Malorvia", "Maloria", "Nordonia", "Ionia",
                      "Aetheria", "Malora", "Maloria", "Aleria", "Maloria"],
    "Zorbeth (F1)": ["Faldia", "Luthara", "Muldrak", "Sordal", "Xel", "Aetheria",
                     "Trakia", "Ered", "Alora", "strong"],
}

BLUE, RED, GREEN, GREY = "#1f77b4", "#d62728", "#2ca02c", "#888"


def b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fig_curve():
    # chart labels in English (matplotlib font lacks Korean); narrative HTML stays Korean
    x = np.arange(len(LEVELS))
    fig, ax1 = plt.subplots(figsize=(7.5, 4.4))
    ax2 = ax1.twinx()
    l1 = ax1.plot(x, COH, "o-", color=BLUE, lw=2, label="neighbor cohesion (1a)")
    l2 = ax2.plot(x, ENT, "s-", color=RED, lw=2, label="output entropy (2a)")
    ax1.set_xticks(x); ax1.set_xticklabels(LEVELS)
    ax1.set_ylim(0.29, 0.33)
    ax1.set_xlabel("familiarity  (F4 frequent -> F1 fabricated -> F0 noise)")
    ax1.set_ylabel("neighbor cohesion (0-1)", color=BLUE)
    ax2.set_ylabel("output entropy (nats)", color=RED)
    ax1.axvspan(2.5, 4.5, color=RED, alpha=0.06)
    ax1.annotate("F2->F1 cliff", xy=(3, 0.303), xytext=(2.5, 0.295),
                 fontsize=9, color=RED)
    ax1.set_title("Familiarity vs cohesion / entropy - gemma-3-4B", fontsize=11)
    ax1.legend(l1 + l2, [t.get_label() for t in l1 + l2], loc="center left", fontsize=8)
    return b64(fig)


CHART_LVL = {"F4": "F4\nfrequent", "F3": "F3\nmid", "F2": "F2\nobscure",
             "F1": "F1\nfabricated", "F0": "F0\nnoise"}


def fig_entropy():
    x = np.arange(len(LEVELS))
    cols = [BLUE, BLUE, BLUE, RED, RED]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.bar(x, ENT, yerr=ENT_SD, color=cols, capsize=3, width=0.62)
    for i, v in enumerate(ENT):
        ax.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([CHART_LVL[l] for l in LEVELS], fontsize=8)
    ax.set_ylabel("output entropy (nats)")
    ax.set_title("Lower familiarity -> wider distribution (F2->F1 cliff) = H-wander", fontsize=10.5)
    ax.set_ylim(0, 6.8)
    return b64(fig)


def fig_distinct():
    f4 = list(F4_DISTINCT.items())
    f1 = list(F1_DISTINCT.items())
    n = max(len(f4), len(f1))
    x = np.arange(n)
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    ax.bar(x - w / 2, [v for _, v in f4] + [0] * (n - len(f4)), w, color=BLUE,
           label="F4 known (converge)")
    ax.bar(x + w / 2, [v for _, v in f1] + [0] * (n - len(f1)), w, color=RED,
           label="F1 fabricated (wander)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f4[i][0] if i < len(f4) else ''}\n{f1[i][0] if i < len(f1) else ''}"
                        for i in range(n)], fontsize=7.5)
    ax.set_ylabel("distinct answers / 10")
    ax.set_ylim(0, 11)
    ax.set_title("Resample diversity - known converge (1/10), fabricated scatter (8-10/10)",
                 fontsize=9.5)
    ax.legend(fontsize=8)
    return b64(fig)


IMG_CURVE, IMG_ENT, IMG_DIST = fig_curve(), fig_entropy(), fig_distinct()


def chips(answers):
    from collections import Counter
    c = Counter(a.lower() for a in answers)
    out = []
    for a in answers:
        n = c[a.lower()]
        cls = "chip hot" if n >= 3 else ("chip warm" if n == 2 else "chip")
        out.append(f'<span class="{cls}">{a}</span>')
    return " ".join(out)


HTML = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>친숙도와 J-space 표현 — Phase 6</title>
<style>
 :root {{ --fg:#1a1a1a; --mut:#666; --line:#e3e3e3; --bg:#fff; --card:#f7f8fa;
         --blue:#1f77b4; --red:#d62728; --green:#2ca02c; }}
 * {{ box-sizing:border-box; }}
 body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Apple SD Gothic Neo",sans-serif;
   color:var(--fg); background:var(--bg); max-width:900px; margin:0 auto; padding:2rem 1.2rem; line-height:1.65; }}
 h1 {{ font-size:1.7rem; margin:.2rem 0; }}
 h2 {{ font-size:1.25rem; margin:2.2rem 0 .6rem; padding-bottom:.3rem; border-bottom:2px solid var(--line); }}
 h3 {{ font-size:1.02rem; margin:1.3rem 0 .4rem; color:#333; }}
 .lead {{ background:var(--card); border-left:4px solid var(--blue); padding:.9rem 1.1rem; border-radius:6px; font-size:.97rem; }}
 .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:.7rem; margin:.8rem 0; }}
 .card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:.8rem .9rem; font-size:.9rem; }}
 .card b {{ display:block; margin-bottom:.25rem; }}
 .verdict {{ font-weight:700; }}
 .refuted {{ color:var(--red); }} .supported {{ color:var(--green); }}
 table {{ border-collapse:collapse; width:100%; margin:.7rem 0; font-size:.88rem; overflow-x:auto; display:block; }}
 th,td {{ border:1px solid var(--line); padding:.4rem .55rem; text-align:left; }}
 th {{ background:var(--card); }}
 figure {{ margin:1rem 0; text-align:center; }}
 figure img {{ max-width:100%; border:1px solid var(--line); border-radius:6px; }}
 figcaption {{ color:var(--mut); font-size:.82rem; margin-top:.3rem; }}
 code {{ background:var(--card); padding:.05rem .3rem; border-radius:4px; font-size:.85em; }}
 .chip {{ display:inline-block; background:#eef1f4; border:1px solid var(--line); border-radius:12px;
   padding:.06rem .5rem; font-size:.8rem; margin:.1rem; }}
 .chip.warm {{ background:#fff3e0; border-color:#ffcc80; }}
 .chip.hot {{ background:#ffe0e0; border-color:#ef9a9a; font-weight:600; }}
 .mut {{ color:var(--mut); font-size:.85rem; }}
 .flag {{ font-size:.82rem; color:var(--mut); background:var(--card); padding:.5rem .7rem; border-radius:6px; }}
</style></head><body>

<h1>친숙도와 J-space 표현 <span class="mut">— J-lens 추가 실험 (Phase 6)</span></h1>
<p class="lead"><b>한 줄 요약:</b> 사용자 원가설(“낯선 개체일수록 출력이 제한적=일관적으로
수렴한다”)은 <b>방향이 반대로 반증</b>됐다. gemma-3-4B에서 친숙도가 낮아질수록 출력 분포는
<b>좁아지는 게 아니라 넓어진다</b>(엔트로피 1.73→5.89, 재샘플 distinct 1/10→9/10) — 수렴은
오히려 <i>아는</i> 개체의 특징이다. 낯선 개체에선 “모른다”고 거부하지도 않고(거부율 0%)
<b>그럴듯한 지명을 지어낸다</b> — 다만 그 지어냄은 음성·지역적으로 <b>구조화</b>돼 있어
(게르만풍 <code>Trennsk</code>→유럽 국가) 사용자 직관의 <i>약한</i> 형태는 남는다. 핵심 고리
(“J-space에 유사토큰 없음”=이웃 응집도↓)는 gemma에서 <b>지지되지 않았다</b>.
<span class="mut">⚠️ 주가 예측은 쓰지 않았다(랜덤워크 교란). 조작적 정의만 측정.</span></p>

<h2>0. 실험 설계</h2>

<h3>0.1 질문과 세 가설 (열어둠)</h3>
<p>질문: “모델이 잘 모르는 개체일수록 J-space 표현이 흐릿해지고 출력이 특정 방식으로
달라지는가?” 마지막 고리(“제한적=일관적”)를 결론으로 가정하지 않고 세 가설로 배타 판정:</p>
<div class="cards">
 <div class="card"><b>H-수렴 (사용자 가설)</b>친숙도↓ → 분포 좁음(엔트로피↓, 재샘플 일관↑).
   없는 개체를 매번 같은 “가장 가까운 아는 것”으로 매핑.</div>
 <div class="card"><b>H-방황</b>친숙도↓ → 분포 넓음(엔트로피↑, 재샘플 산만). 닻 없어 방황.</div>
 <div class="card"><b>H-거부</b>친숙도↓ → “모른다”로 회피(메타인지).</div>
</div>

<h3>0.2 설계 결정 — 주가 예측을 쓰지 않는다</h3>
<p>원 아이디어(최근 상장 ADR 주가 예측)는 이 가설을 검정할 수 없다: 주가는 원리적으로 예측
불가(랜덤워크)라 “제한적 답변”이 나와도 <i>J-space에 정보가 없어서</i>인지 <i>주가가 원래 예측
불가라서</i>인지 분리 불가(교란변수가 결과를 삼킴). 그래서 친숙도를 예측 능력이 아니라 통제
가능한 <b>“개체 빈도”</b> 축으로 바꾼다. 종목·가격 예측은 대상이 아니다.</p>

<h3>0.3 친숙도 축 (개체 빈도) — 실제 사용 개체</h3>
<table><tr><th>단계</th><th>정의</th><th>개체</th></tr>
<tr><td><b>F4</b> 초고빈도</td><td>누구나 아는 대도시</td><td>Paris, Tokyo, London, Rome, Berlin, Madrid, Cairo, Moscow</td></tr>
<tr><td><b>F3</b> 중빈도</td><td>덜 빈번한 수도</td><td>Tashkent, Almaty, Nairobi, Tbilisi, Ljubljana, Windhoek, Chisinau, Kigali</td></tr>
<tr><td><b>F2</b> 무명</td><td>실재 무명 소도시</td><td>Oruro, Jinja, Gyumri, Sokoto, Bregenz, Kanazawa, Timbuktu, Iquitos</td></tr>
<tr><td><b>F1</b> 가공(정보0)</td><td>발음가능 가짜 지명</td><td>Malorvia, Trennsk, Vantoria, Qellmin, Zorbeth, Dravnia, Klenor, Vurstan</td></tr>
<tr><td><b>F0</b> 무의미</td><td>발음 불가 문자열(바닥선)</td><td>Xqzptr, Zkvwmn, Qfjklb, Wrtpxn</td></tr></table>

<h3>0.4 프레임 (고정, 개체만 변수) · 0.5 지표</h3>
<p>프레임: <code>"{{X}} is a city located in the country of"</code>,
<code>"The most famous landmark in {{X}} is the"</code>. 답 위치·워크스페이스 대역(40–90%)에서
한 번의 <code>lens.apply</code>로:</p>
<ul>
 <li><b>1a 이웃 응집도</b> = 렌즈 top-20 토큰 임베딩 상호 코사인 평균(높음=뭉친 이웃). ← 가설 핵심 고리 직접 측정.</li>
 <li><b>1c 렌즈 확신도</b> = 렌즈 top-1 확률 + 마진.</li>
 <li><b>2a 출력 엔트로피</b> = 모델 다음-토큰 분포 섀넌 엔트로피(낮음=좁음/H-수렴, 높음=넓음/H-방황).</li>
 <li><b>2b 재샘플 다양성</b> = T=0.8·top_p=0.95로 N=10 생성 → 서로 다른 답 수(F4·F1 양끝만).</li>
 <li><b>2c 거부율</b> = “unknown/don't know…” 정규식 매칭(H-거부).</li>
 <li><b>3 최응집 레이어</b>(보조).</li>
</ul>
<p class="flag"><b>지표 1b(희소분해/gradient pursuit) 생략</b> — jlens 라이브러리에 pursuit API가 <b>없음</b>
(코드 탐색으로 확인). 의도(날카로운 표현 vs 흐릿)는 1a+1c로 커버. 모델: <code>google/gemma-3-4b-pt</code>
(bf16), 로컬 M2/MPS, greedy.</p>

<h2>1. 지표 곡선</h2>
<figure><img src="data:image/png;base64,{IMG_CURVE}" alt="familiarity curve">
<figcaption>엔트로피(빨강)는 F2→F1에서 급등(절벽). 응집도(파랑)는 0.30–0.32로 사실상 평탄
— y축 확대라 커 보일 뿐 친숙도와 단조 관계 없음.</figcaption></figure>
<table><tr><th>단계</th><th>이웃 응집도(1a)</th><th>출력 엔트로피(2a)</th><th>렌즈 top1(1c)</th></tr>
<tr><td>F4</td><td>0.320±0.014</td><td><b>1.73</b>±0.49</td><td>0.804</td></tr>
<tr><td>F3</td><td>0.309±0.014</td><td>2.29±0.29</td><td>0.743</td></tr>
<tr><td>F2</td><td>0.314±0.016</td><td>2.55±0.56</td><td>0.675</td></tr>
<tr><td>F1</td><td>0.303±0.012</td><td><b>5.89</b>±0.12</td><td>0.751</td></tr>
<tr><td>F0</td><td>0.308±0.002</td><td>5.92±0.10</td><td>0.669</td></tr></table>
<p class="mut">gemma는 F2(무명 도시)조차 정답을 안다(Timbuktu→Mali, Iquitos→Peru) — 4B 지식이 넉넉해
F2–F4가 모두 “아는” 쪽. 실제 대비는 <b>F1·F0 양 끝</b>에서 갈린다.</p>
<figure><img src="data:image/png;base64,{IMG_ENT}" alt="entropy bars"></figure>

<h2>2. H 판정 — 사용자 가설은 반대로 반증</h2>
<figure><img src="data:image/png;base64,{IMG_DIST}" alt="resample distinct">
<figcaption>아는 개체(F4)는 재샘플이 한 답으로 <b>수렴</b>(1/10), 가공 개체(F1)는 <b>산발</b>(8–10/10).
수렴은 낯선 쪽이 아니라 <i>친숙한</i> 쪽에서 일어난다.</figcaption></figure>
<table><tr><th>가설</th><th>F4(친숙)</th><th>F1(가공)</th><th>판정</th></tr>
<tr><td>H-수렴 (사용자)</td><td>distinct 1~3/10, ent 1.73</td><td>distinct 8~10/10, ent 5.89</td>
  <td class="verdict refuted">반증 — 수렴은 친숙 쪽</td></tr>
<tr><td>H-방황</td><td>(친숙은 수렴)</td><td>distinct 8~10/10</td>
  <td class="verdict supported">지지 (구조적)</td></tr>
<tr><td>H-거부</td><td>거부율 0%</td><td>거부율 0%</td>
  <td class="verdict refuted">반증 — 항상 confabulate</td></tr></table>

<h3>재샘플 원문 (색: <span class="chip hot">3회+</span> <span class="chip warm">2회</span> 반복)</h3>
"""
for name, ans in RESAMP.items():
    HTML += f'<p><b>{name}</b><br>{chips(ans)}</p>\n'

HTML += f"""
<p class="mut">방황이되 무작위가 아니다: <code>Trennsk</code>(게르만풍)→실제 유럽 국가들,
가짜 이름들에 <code>Aetheria·Faldia·Trakia</code> 판타지 어트랙터가 공통 재등장. greedy(최빈값)는
F1에서 국가도 판타지도 아닌 <b>이름 철자 되뇌기</b>(Vantoria→<code>' V'</code>) — 모델의 최빈값과
분포 질량이 달라, 엔트로피와 greedy를 함께 봐야 한다.</p>

<h2>3. 사용자 가설 사슬, 고리별 검증</h2>
<div class="cards">
 <div class="card"><b>고리 A — “유사토큰 없음” = 응집도↓</b>
   <span class="verdict refuted">미지지.</span> 응집도는 친숙도와 무관하게 평탄(0.30–0.32).
   가짜 지명의 렌즈 이웃도 아는 도시만큼 뭉쳐 있다.</div>
 <div class="card"><b>고리 B — “제한적 = 일관적”</b>
   <span class="verdict refuted">반증.</span> 낯설수록 오히려 <b>다양</b>(distinct 1/10→9/10).
   “제한적”이라면 정보량 제한이지 출력 다양성 제한이 아니다.</div>
</div>

<h2>4. 대조군</h2>
<ul>
 <li><b>토큰 길이 교란 배제:</b> F2(2.8토큰, ent 2.55) vs F1(2.9토큰, ent 5.89) — 토큰수 거의 같은데
   엔트로피 2배+. 엔트로피 급등은 “다토큰 희귀어”가 아니라 <b>친숙도(실재 vs 가공)</b> 때문.</li>
 <li><b>F0 무의미 바닥선:</b> F0(ent 5.92) ≈ F1(5.89) — gemma에게 “가짜 지명”과 “무의미 문자열”은
   다음-토큰 분포상 구분되지 않는다(둘 다 “정보 0”).</li>
</ul>

<h2>5. 한계 (정직)</h2>
<ul>
 <li><b>J-space는 저장고가 아니라 렌즈.</b> “응집도↓ ≠ 정보 없음”이고, 여기선 응집도가 <i>안 떨어졌다</i>
   — 이걸 “가짜 개체도 잘 안다”로 과대해석 금지.</li>
 <li><b>“친숙도”는 대리변수.</b> gemma-4B는 지식이 넉넉해 F2–F4가 뭉개짐 → 결론은 F4 vs F1/F0 양끝 대비에 기댐.</li>
 <li>축 2(학습 컷오프 PRE/POST) 미실시(노이즈). 재샘플 N=10·각 5개체(작음, 방향은 뚜렷).</li>
 <li>거부율 0%는 <b>소형 base 모델 특유</b>일 수 있음 — instruct-튜닝 큰 모델이면 다를 수 있다.</li>
</ul>

<h2>6. 결론</h2>
<p><b>사용자 가설(낯섦→수렴)은 gemma-4B에서 반대로 반증.</b> 낯선 개체 → 출력 <b>확산</b>(H-방황),
아는 개체 → 수렴. <b>핵심 고리(이웃 응집도↓)도 미지지.</b> 단 방황은 <b>구조적 confabulation</b>
(음성·지역적으로 그럴듯) + base 모델은 <b>무지를 자각 못 함</b>(거부 0%) — 사용자 직관의 약한 형태
(“가까운 그럴듯한 것으로”)는 지역/음운 수준에서 살아있으나 “하나로 수렴”은 아니다.</p>

<p class="mut" style="margin-top:2rem; border-top:1px solid var(--line); padding-top:.8rem;">
산출물: <code>FAMILIARITY_REPORT.md</code> · <code>familiarity_freq_gemma4b.md</code> ·
<code>familiarity_H_verdict_gemma4b.md</code> · <code>familiarity_curve_gemma4b.png</code> ·
스크립트 <code>scripts/06_familiarity.py</code>. 이 HTML: <code>scripts/07_familiarity_html.py</code>로 생성.</p>

</body></html>"""

outdir = RESULTS / "phase6_familiarity"; outdir.mkdir(exist_ok=True); out = outdir / "FAMILIARITY_REPORT.html"
out.write_text(HTML, encoding="utf-8")
print(f"Wrote {out}  ({len(HTML)} bytes)")
