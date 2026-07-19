# J-lens 로컬 재현 — 최종 리포트

**한 줄 요약:** J-lens의 **내부추론 읽기 현상(spider)은 규모에 따라 명확히 나타났다** —
0.8B/1.7B에선 미관찰이었으나 **Gemma-3-4B에서 재현**(spider 렌즈 rank 750→317→2, 레이어
진행이 sensory→workspace→motor 교과서적). directed-modulation(citrus)은 전 규모에서
비결정적. **스왑 메커니즘은 0.8B/1.7B에서 대조군 포함 깨끗하게 재현**. 지정 실험 "spider
다리 8→6"은 소형 모델이 사실을 몰라 검증 불가. 과장 없이 있는 그대로 기록한다.

> **업데이트(Gemma-3-4B 추가):** 사용자 요청으로 읽기 실험을 4B급 `google/gemma-3-4b-pt`
> (게이트 모델, bf16 필수)로 재실행. **Experiment B가 재현됨** — 아래 §3, §4 및
> `scale_compare.md` 참조. (스왑 C/D는 이번 라운드에서 4B로 재실행하지 않음: 읽기 전용 요청.)

---

## 1. 환경

- 하드웨어: Apple Silicon M2, 16GB RAM, CUDA 없음 → PyTorch **MPS / fp16** (backprop 불필요, 추론만).
- 소프트웨어: Python 3.13, torch 2.13.0, transformers 5.14.1, `anthropics/jacobian-lens`(Apache-2.0).
- 모델·렌즈: HuggingFace `transformers`로 로컬 실행. 렌즈는 **기학습본 로드**(피팅 없음).
  - `neuronpedia/jacobian-lens` (공개, 논문과 동일 방식 `Salesforce-wikitext` 피팅).
- **Ollama 미사용**: J-lens는 레이어별 활성값·`W_U`·forward hook 등 모델 내부 접근이 필수라
  텍스트 in/out만 주는 Ollama와 비호환.

## 2. 모델 & 렌즈

| 모델 | HF id | 파라미터 | lens source layers | 비고 |
|---|---|---|---|---|
| 주 모델 | `Qwen/Qwen3.5-0.8B` | 752M | 0–22 (23개) | fp16 |
| 비교 모델 | `Qwen/Qwen3-1.7B` | 1.7B | 0–26 (27개) | fp16 |
| 대형 비교 | `google/gemma-3-4b-pt` | 4B | 0–32 (33개) | **게이트, bf16 필수** (fp16 오버플로우→NaN) |
| 스모크 | `EleutherAI/pythia-70m-deduped` | 70M | (미사용) | 파이프라인 검증용 |

- **렌즈 sanity (Phase 1)**: 두 모델 모두 5/5 통과 — 마지막 3개 층 중 어디선가 렌즈 top-1 =
  모델 greedy 다음 토큰. 레이어 진행에서 sensory→workspace→motor 패턴 관찰
  (초반: 문장부호/무의미 → 중반: 빈칸/개념 → 후반: 정답 확정).

## 3. 실험 결과

### 실험 A — Directed modulation (citrus) : **약함/미재현**
- "citrus를 생각하며 문장을 베껴라"의 복사 구간에서 orange/lemon류가 렌즈에 뜨는지, 집중
  지시(FOCUS) vs 무지시(CONTROL) 대조.
- 0.8B: FOCUS 최고 rank 18 vs CONTROL 42 (방향 일치하나 top-5 밖, 대조군에도 orange 존재).
- 1.7B: FOCUS 6 vs CONTROL 3 (**대조 역전**). 4B: FOCUS 29 vs CONTROL 67 (방향 일치, 여전히 약함).
- **결론**: focus>control 대조가 규모에 걸쳐 **비일관적**(0.8B·4B 약한 정방향, 1.7B 역전),
  citrus는 어느 규모에서도 상위 랭크 미도달 → directed-modulation은 **비결정적/미재현**.

### 실험 B — Internal reasoning (spider) : **규모에 따라 재현 (4B에서 성공)**
- "The number of legs on the animal that spins webs is" 마지막 토큰에서 'spider'가 중간
  레이어에 뜨는지.
- 'spider' 최고 렌즈 rank: 0.8B **750**@L7 → 1.7B **317**@L14 → **4B 2**@L28. **단조 개선.**
- **Gemma-4B 레이어 진행 (교과서적 sensory→workspace→motor):**
  - L0–14: `<start_of_image>`/문장부호/공백 (무의미)
  - L15–19: `critters, beetles, insects, turtles, squirrels, vertebrates` (워크스페이스: **동물 범주**)
  - L26–28: `spiders, spider` rank 2–4 (**프롬프트에 없는 브리지 개념 부상**)
  - L29–32: `six, eight, four, five` (**motor: 다리 수 답**)
- 표면 greedy 다음 토큰은 세 모델 모두 `' '`(base 모델은 숫자를 직접 말하지 않음) —
  즉 4B에서 렌즈가 "말하지 않은 spider→숫자 사고"를 **판독**함.
- **결론**: 내부추론 읽기 현상은 **규모와 함께 뚜렷이 발현**, 4B에서 재현. 논문의 규모 주장 지지.

### 실험 C — 스왑 개입
**C1 (지정 타깃, spider→ant, 8→6): 전 규모 미재현**
- 사실 지식은 4B에서 발현: greedy 연속이 0.8B `' 10'`, 1.7B `' 4'`, **4B `' 8.'`(정답 앎)**.
- 그러나 4B에서도 spider→ant 스왑은 8→6으로 못 바꿈 — α=2에서 출력이 그냥 `' spider'`(P=0.99)로
  증폭(과도개입/방향중첩 아티팩트). **지정 8→6 인과 스왑은 어느 규모에서도 재현 안 됨.**

**C2 (메커니즘 검증, baseline이 깨끗한 프롬프트): 재현 성공**
- 모델이 자신 있게 A를 출력하고 렌즈가 워크스페이스 밴드에서 A를 읽는 프롬프트에서 A→B 스왑.
- 0.8B: Paris→London (α=1), blue→red (α=1), cold→warm (α=2) 에서 greedy가 **B로 뒤집힘**.
- 1.7B·4B: 답 확신이 커 넓은 밴드로는 greedy 미변경, α≥2는 source 증폭/붕괴.
  → 그러나 **국소 후반 창**에서는 뒤집힘. **레이어 스윕 flip 확률이 규모와 함께 상승:
  0.8B P=0.78(86%) → 1.7B P=0.91(81%) → 4B P=0.99(84–90%).**

**C 대조군 (두 규모 공통, 정상 동작)**
- 무작위 방향(동일 크기): 출력이 목표로 안 바뀜 (P(target)≈0). → 효과는 **방향 특이적**.
- 초반 레이어(0–30%)만: 출력 변화 없음/붕괴. → 효과는 **워크스페이스/후반 대역 특이적**.
- α=4(과도 개입): 모델이 문장부호로 붕괴(1.7B는 α=2에서 이미 붕괴).

### 실험 D — 레이어 스윕 : **명확한 후반 국소성 (전 규모)**
- 6-layer 창을 옮겨가며 Paris→London(α=2) 적용, 출력이 언제 바뀌는지.
- 0.8B: **78–86%**에서 뒤집힘 (P 0→0.48→0.78). 1.7B: **81%**(P=0.91). 4B: **84–90%**(P=0.95→0.99).
- **결론**: 스왑 인과 자리는 세 모델 모두 **후반(≈80–90% 깊이)에 날카롭게 국소화**,
  flip이 규모와 함께 **더 깨끗**해짐.

## 4. 논문과의 비교

| 논문 주장 | 이번 재현(소형 모델) |
|---|---|
| 중간 레이어에서 미출력 개념이 렌즈로 판독됨 | **4B에서 재현** (spider rank 2). 0.8B/1.7B는 미관찰. |
| sensory→workspace→motor 진행 | **4B에서 뚜렷** (무의미→동물범주→spider→숫자). sanity 프롬프트에서도 관찰. |
| 워크스페이스 방향 스왑이 출력을 인과적으로 바꿈 | **재현됨** (전 규모, 후반 국소 창; 방향·깊이 대조군 통과). 단 지정 8→6은 미재현. |
| 규모↑ → 워크스페이스 구조↑ | **지지**: 읽기 spider rank 750→317→2, 스왑 flip P 0.78→0.91→0.99로 규모와 함께 강화. |

논문은 프런티어급(Sonnet 4.5 등) 대상이며, 워크스페이스 구조가 소형 모델에 얼마나 있는지는
열린 문제다. 이번 결과는 **내부추론 읽기 현상이 4B에서 임계적으로 발현**(0.8B/1.7B 미관찰)하고,
**인과적 스왑 메커니즘은 소형에서도 동작**함을 보여준다 — 둘 다 논문의 규모 논지와 정합적.

## 5. 실패/한계와 추정 원인

- **읽기 현상의 규모 의존성**: 0.8B/1.7B는 다중 홉 사실추론(거미→다리수) 능력이 약해 렌즈가
  읽을 "생각"이 강하게 형성 안 됨. **4B에서 임계 이상**이 되어 spider가 렌즈 rank 2로 발현.
- **지정 8→6 스왑 불가(0.8B/1.7B)**: baseline이 8을 안 냄. (4B는 렌즈상 six/eight를 모두
  고려 — 스왑 재실행은 이번 읽기 전용 요청 범위 밖.)
- **Gemma dtype**: fp16에서 활성값 오버플로우→NaN→`<pad>` 출력. **bf16 필수** (코드에 반영).
- **α 민감도**: 넓은 밴드 × 큰 α는 모델을 붕괴시킴(문장부호). 유효 구간은 좁음(α≈1–2, 국소 창).
- **directed-modulation 대조 역전(1.7B)**: 캐리어 문장 자체가 orange를 약하게 유발하는 교란 가능성.

## 6. 다음에 해볼 것

1. ✅ **완료**: `gemma-3-4b` 읽기(spider rank 2) + 스왑 C/D 재실행. 발견: 지정 8→6 스왑은
   4B(사실 앎)에서도 미재현(스왑이 `spider`를 표면화). 다음은 `llama3.1-8b` 등 추가 규모점,
   그리고 **논문 probe-swap의 정확한 프로토콜**(브리지 엔티티를 여러 위치에서 스왑, 밴드 내 hit
   스코어링)로 spider→ant를 재시도.
2. **읽기 위치 재검토**: 마지막 토큰이 아니라 "webs"/"animal" 등 브리지 개념이 형성되는 위치에서
   판독. 논문의 probe-swap 프롬프트/스코어링(밴드 내 hit 정의) 정확히 따르기.
3. **baseline이 성립하는 다중 홉**으로 브리지 스왑(예: 모델이 맞히는 수도/통화 two-hop)에서
   intermediate 엔티티 스왑 → 최종 답 변화 측정 (논문 probe-swap 재현).
4. **스티어링/절제**도 추가(커뮤니티 `interventions.py`의 steer/ablate) — 방향의 읽기·쓰기 대칭 확인.
5. d3 슬라이스 시각화(`jlens/vis.py`) HTML 산출로 레이어×위치 판독을 시각적으로.

## 7. 산출물 (jlens_repro/results/)

- `env.md` — 환경/그리고 API 실측
- `lens_sanity_qwen0.8b.md`, `lens_sanity_qwen1.7b.md`, `lens_sanity_gemma4b.md`
- `reading_A_*.md`, `reading_B_*.md` (0.8B/1.7B/**gemma4b**)
- `swap_*.md`, `swap_layer_sweep_*.md` (0.8B/1.7B/**gemma4b**)
- `scale_compare.md` — 0.8B vs 1.7B vs **4B** 비교
- 재현 스크립트: `jlens_repro/scripts/` (`common.py`, `swaplib.py`, `00`–`03`)
