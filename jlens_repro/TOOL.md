# J-lens 인터랙티브 스왑 테스터

로컬 Flask 서버 + 브라우저 프론트엔드. 임의 프롬프트에서 J-lens 레이어별 판독을 보고,
개념 방향 스왑을 실시간으로 테스트한다. (브라우저 단독으로는 모델을 못 돌리므로 서버 필요.)

## 실행

```bash
cd jlens_repro
# 기본 모델 qwen0.8b, 포트 8137
PYTORCH_ENABLE_MPS_FALLBACK=1 ./.venv/bin/python scripts/server.py

# 모델·포트 지정: server.py <model_key> <port>
PYTORCH_ENABLE_MPS_FALLBACK=1 ./.venv/bin/python scripts/server.py qwen1.7b 8137
```

model_key: `qwen0.8b`(기본·빠름) · `qwen1.7b` · `gemma4b`(게이트, bf16, 느림) · `pythia70m`.
기동 후 브라우저에서 **http://127.0.0.1:8137** 접속. UI 상단 드롭다운으로 모델 전환 가능
(전환 시 재로드 — gemma는 8.6GB라 느림).

## 화면

1. **렌즈 읽기** — 프롬프트·위치·추적토큰 입력 → 레이어별 lens top-6 + 추적 토큰 rank
   (rank 색: 초록 ≤5, 노랑 ≤50, 회색 그 이상).
2. **스왑 테스트** — src/tgt 개념, α 슬라이더, 밴드(%) 슬라이더, 무작위-방향 대조군 →
   baseline vs swapped 다음-토큰 분포, P(src)/P(tgt), 뒤집힘 여부.
3. **레이어 스윕** — 6-layer 창을 옮겨가며 P(tgt) 막대. 어느 깊이부터 뒤집히는지.

프리셋 칩: `spider→ant`, `Paris→London`, `cold→warm`, `blue→red`.

## 🔬 레이어 탐색기 (`/explorer`)

같은 서버의 두 번째 페이지 — **http://127.0.0.1:8137/explorer**. query→answer 생성 동안
**모든 레이어(L0~L32)의 렌즈 판독 흐름**을 봅니다. 한 번의 greedy 생성 패스에서 전 레이어
residual을 캡처(`trace_hooks`)한 뒤 J-lens로 오프라인 판독(`POST /api/trace`).

- **레이어×토큰 그리드** — y=층(위=출력 근처), x=프롬프트|생성 토큰. 셀=렌즈 top-1
  (밝기=확률, 초록=그 열의 최종 출력과 일치 → "답이 미리 보이는" 지점). 셀 클릭=top-k 상세.
- **생성 재생** — ▶로 답이 토큰 하나씩 생성되며 열이 차오르는 애니메이션 + 스크럽/속도.
- **등장 곡선** — 각 생성 토큰이 렌즈 top-5에 처음 든 레이어(등장 깊이).
- **층별 통계 레인** — 층마다 평균 top-1 확률·엔트로피 미니 바.
- **개입 전/후 비교** — 스왑/스티어/절제(/콤보 JSON)를 걸고 "전/후 비교" 체크 시 두 trace를
  나란히: top-1이 달라진 셀은 빨간 테두리, 등장 곡선은 겹쳐 그림.
- trace는 프롬프트당 1회 무거운 호출(gemma 12토큰 ≈ 30~60s)이고, 이후 탐색(클릭·재생·비교)은
  전부 브라우저 캐시 위에서 즉시. 기본 모델은 gemma4b(층 흐름이 실제로 보이는 유일 규모),
  UI 드롭다운으로 qwen0.8b 전환 가능.

## 참고

- src/tgt는 **단일 토큰** 단어여야 함(앞 공백/대문자 변형은 자동 집계). 다중 토큰이면 UI가 알림.
- P(tgt)·flip 판정은 " London"/"London"/"伦敦" 등 표면 변형을 모두 합산/포함.
- 구현: 백엔드 `scripts/server.py`, 프론트 `scripts/frontend.html`,
  공용 로직 `scripts/common.py`·`scripts/swaplib.py` 재사용.
- 종료: `pkill -f scripts/server.py`
