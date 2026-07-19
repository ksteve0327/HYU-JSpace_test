# J-lens 로컬 재현 + 인터랙티브 개입 테스터

Anthropic 논문 **["Verbalizable Representations Form a Global Workspace in Language
Models"](https://transformer-circuits.pub/2026/workspace)** 의 **Jacobian lens(J-lens)** 를
Apple Silicon MacBook(M2, 16GB)에서 직접 돌려보고, 그 위에 **브라우저 개입 테스터**를 붙인
프로젝트입니다.

> **기반 레포:** 이 프로젝트는 Anthropic 공식 구현
> **[anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)** (Apache-2.0)
> 를 베이스로 만들었습니다. 렌즈 로드·`jlens.from_hf`·`jlens/vis.py` 등 핵심 라이브러리는
> 그 레포를 그대로 쓰고, 그 위에 재현 스크립트와 웹 테스터를 얹었습니다.

## 왜 만들었나 (목적)

1. **논문이 글로만 봐선 잘 이해가 안 돼서, 직접 구현하며 이해하려고.**
   "중간 레이어에서 아직 말하지 않은 생각을 읽는다", "워크스페이스 방향을 바꾸면 출력이
   바뀐다" 같은 주장을 소형 오픈 모델에서 실제로 돌려 눈으로 확인하는 게 목표였습니다.
2. **논문이 스스로 밝힌 한계 — "개입은 한 번에 하나(single intervention)" — 를 실험해보려고.**
   그래서 테스터에 **콤보(combo) 모드**를 넣어, 스왑·스티어·절제를 **한 forward에 동시에**
   여러 개 걸어보고 무슨 일이 나는지 관찰할 수 있게 했습니다.

소형 모델에서는 현상이 약하거나 없을 수 있고, **재현 실패도 그대로 기록**하는 것을 원칙으로
삼았습니다(과장 금지). 실제로 일부는 재현됐고 일부는 안 됐습니다 — 아래 결과 요약 참조.

---

## 무엇을 하나

### A. 논문 현상 재현 (스크립트)
소형 오픈 모델 3종(Qwen3.5-0.8B · Qwen3-1.7B · Gemma-3-4B)으로 논문의 3개 현상을 재현 시도:

1. **렌즈 읽기** — 중간 레이어에서 "아직 출력 안 된 개념"이 판독되는가
2. **레이어 진행** — sensory → workspace → motor (초반 무의미 / 중반 개념 / 후반 다음 토큰)
3. **스왑 개입** — 워크스페이스 레이어에서 개념 방향을 교체하면 출력이 바뀌는가

### B. 인터랙티브 개입 테스터 (Flask + HTML)
임의 프롬프트에서 J-lens 레이어별 판독을 실시간으로 보고, 개입을 눌러 결과를 확인:

- **1. 렌즈 읽기** — 레이어별 lens top-6 + 추적 토큰 rank (색: 초록 ≤5, 노랑 ≤50)
- **2. 개입** — 4가지 모드를 하나의 패널에서:
  - **스왑** `h ← h − (h·v_A)v_A + (h·v_A)v_B` (좌표 교환)
  - **스티어** `h ← h + α·(평균 residual norm)·v_w` (개념 주입/억제)
  - **절제** `h ← h − α·(h·v_w)v_w` (개념 방향 제거)
  - **콤보** — 위 개입을 여러 개 스택해서 **한 forward에 동시 적용** (← 논문 "단일 개입" 한계 실험용)
- **3. 레이어 스윕** — 창을 옮겨가며 어느 깊이부터 뒤집히는지
- **4. 슬라이스 히트맵** — d3로 레이어×위치 판독 시각화
- 부가: AI 해석(로컬 Claude/Codex proxy 또는 OpenRouter)·챗봇·자동반복·로컬 기록·전체 자동 실행

---

## 결과 요약 (있는 그대로)

| 논문 주장 | 이번 재현(소형 모델) |
|---|---|
| 중간 레이어에서 미출력 개념이 렌즈로 판독됨 | **4B에서 재현** (spider 렌즈 rank **750→317→2**, 규모와 함께 단조 개선). 0.8B/1.7B는 미관찰. |
| sensory→workspace→motor 진행 | **4B에서 뚜렷** (무의미 → 동물 범주 → spider → 숫자) |
| 워크스페이스 방향 스왑이 출력을 인과적으로 바꿈 | **재현** (전 규모, 후반 국소 창; 방향·깊이 대조군 통과). flip P **0.78→0.91→0.99**로 규모와 함께 깨끗해짐 |
| directed-modulation (citrus) | **비결정적/미재현** (전 규모에서 노이즈) |
| 규모↑ → 워크스페이스 구조↑ | **지지** (읽기·스왑 모두 규모와 함께 강화) |

- **지정 실험 "spider 다리 8→6" 스왑은 어느 규모에서도 미재현** — 4B는 사실을 알지만(spider=8)
  스왑이 숫자를 바꾸는 대신 단어 `spider`를 표면화. 도즈/방향중첩 아티팩트로 추정.
- 자세한 내용: [`results/REPORT.md`](results/REPORT.md), [`results/scale_compare.md`](results/scale_compare.md),
  [`results/e2e_run_report.md`](results/e2e_run_report.md).

---

## 설치 & 실행

### 0. 요구사항
- Python 3.13 권장, Apple Silicon(MPS) 또는 CPU. torch·transformers.
- Gemma는 **게이트 모델** — 본인 HuggingFace 계정으로 라이선스 동의 + `hf auth login` 필요.

### 1. 셋업
```bash
# 상위 저장소를 서브모듈(anthropics/jacobian-lens, 커밋 581d398 핀)까지 받기
git clone --recursive https://github.com/ksteve0327/HYU-JSpace_test
# 이미 클론했다면:  git submodule update --init jlens_repro/jacobian-lens

cd jlens_repro
python3.13 -m venv .venv && source .venv/bin/activate
pip install torch "transformers>=5.5" huggingface_hub numpy pandas
pip install ./jacobian-lens   # 서브모듈로 고정된 기반 라이브러리
```

### 2. 환경변수 (선택 — AI 해석 기능용)
```bash
cp .env.example .env
# .env 를 열어 OPENROUTER_API_KEY 등을 채우세요. (없어도 재현/테스터 핵심은 동작)
```

### 3. 테스터 실행
```bash
# server.py <model_key> <port>   기본: qwen0.8b, 8137
PYTORCH_ENABLE_MPS_FALLBACK=1 ./.venv/bin/python scripts/server.py gemma4b 8137
```
`model_key`: `qwen0.8b`(빠름) · `qwen1.7b` · `gemma4b`(느림·게이트·bf16) · `pythia70m`.
브라우저에서 **http://127.0.0.1:8137** 접속. 자세한 사용법은 [`TOOL.md`](TOOL.md).

서버는 시작할 때 `localhost:11435`의 Codex proxy를 확인하고, 꺼져 있으면
`~/bin/codex-proxy.mjs`를 자동으로 백그라운드 실행합니다. 자동 실행을 끄려면 `.env`에
`CODEX_PROXY_AUTOSTART=0`을 설정하세요.

### 4. 재현 스크립트 (테스터 없이)
```bash
./.venv/bin/python scripts/00_smoke.py      # 환경 검증
./.venv/bin/python scripts/01_load_lens.py  # 렌즈 로드 + sanity
./.venv/bin/python scripts/02_reading.py    # 읽기 실험 A/B
./.venv/bin/python scripts/03_swap.py       # 스왑 개입 C/D
```

---

## 구조

```
jlens_repro/
  scripts/
    common.py          모델/렌즈 로드 (MODELS 딕셔너리, MPS·bf16 처리)
    swaplib.py         개입 프리미티브 (swap/steer/ablate hook, 개념 방향 v_w)
    00_smoke.py … 03_swap.py   재현 스크립트
    04_viz.py          d3 슬라이스 HTML 생성
    server.py          Flask 백엔드 (개입 API, AI 해석 라우팅)
    frontend.html      단일 파일 웹 UI
  results/             재현 산출물 (.md 리포트, .html 히트맵)
  .env.example         환경변수 템플릿  (.env 는 gitignore)
```

---

## 보안 / 공개 관련

- **비밀은 커밋되지 않습니다.** 실제 API 키는 `.env`(→ `.gitignore`)에만 있고, 저장소에는
  값이 빈 [`.env.example`](.env.example)만 포함됩니다.
- **LLM proxy는 로컬 전용.** AI 해석은 `localhost:11436`의 Claude proxy와
  `localhost:11435`의 Codex proxy만 사용합니다. Codex proxy는 서버 시작 시 필요하면 자동
  실행되고, Claude proxy가 안 떠 있으면 해당 모델만 자동 비활성화됩니다.
- 소스에 하드코딩된 키·토큰은 없습니다(공개 전 스캔 완료).

## 라이선스 / 크레딧

- 기반: [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) (Apache-2.0).
- 논문: Anthropic, *Verbalizable Representations Form a Global Workspace in Language Models*,
  [transformer-circuits.pub/2026/workspace](https://transformer-circuits.pub/2026/workspace).
- 기학습 렌즈: [neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens)
  (공개, `Salesforce-wikitext` 피팅).
- 모델: Qwen(Alibaba), Gemma(Google, 게이트), Pythia(EleutherAI) — 각 라이선스는 해당 제공자.
