# H 판정 — 친숙도와 출력 분포 (gemma-3-4B)

세 가설을 지표 2(엔트로피·재샘플 다양성·거부율)로 배타 판정. 재샘플: 프레임
`"{X} is a city located in the country of"`, T=0.8/top_p=0.95, N=10.

## 판정표

| 가설 | 예측 | F4(친숙) | F1(가공) | 판정 |
|---|---|---|---|---|
| **H-수렴**(사용자 가설) | 낯섦↓ → 분포 좁음·재샘플 일관 | distinct **1~3/10**, 엔트로피 **1.73** | distinct **8~10/10**, 엔트로피 **5.89** | **반증** — 수렴은 *친숙* 쪽에서 일어남 |
| **H-방황** | 낯섦↓ → 분포 넓음·재샘플 산발 | (친숙은 수렴) | distinct **8~10/10**, 엔트로피 **5.89** | **지지**(단 구조적, 아래) |
| **H-거부** | 낯섦↓ → "모른다"(메타인지) | 거부율 **0%** | 거부율 **0%** | **반증** — base 모델은 항상 confabulate |

**결론: 사용자 가설(낯섦=수렴)은 방향이 반대다.** 낯선 개체에서 모델은 **좁아지는 게 아니라
넓어진다**(방황). 수렴은 오히려 *아는* 개체의 특징이다. 그리고 소형 base 모델은 무지를
자각해 "모른다"고 하지 않고 **항상 그럴듯한 것을 지어낸다**(거부 0%).

## 단, 방황은 "무작위"가 아니라 "구조적 confabulation"

F1 재샘플 원문을 보면 방황이 그럴듯한 이웃 안에서 일어난다 — 사용자 직관의 *약한* 형태:

- **`Trennsk`**(게르만/슬라브 음운) → `germany, germany, belgium, austria, denmark, saxony, namibia, peru`
  — 대부분 **실제 유럽 국가**로. "가장 가까운 아는 것"으로의 부분 수렴이 **지역 단위**로는 있음.
- **`Malorvia`** → `malorvia, maloria, malora, maloria(×3), nordonia, ionia, aetheria, aleria`
  — 자기 이름의 **음성 변형** + 가짜 지명.
- **반복 어트랙터**: `Aetheria`·`Faldia`·`Trakia`가 서로 다른 가짜 이름(Malorvia·Vantoria·Qellmin·Zorbeth)에
  **공통 등장** — 모델이 "일반 판타지 지명" 몇 개로 회귀하는 경향.
- **`Zorbeth`** → `faldia, luthara, muldrak, sordal, xel, aetheria, trakia, ered, alora` — 순수 판타지 생성(10/10).

## 대표 원문 (F4 3 · F1 3)

**F4(수렴):**
- `Paris` → 10회 전부 **"France…"** (distinct 1/10)
- `Tokyo` → 10회 전부 **"Japan…"** (distinct 1/10)
- `Rome` → 10회 전부 **"Italy…"** (distinct 1/10)

**F1(구조적 방황):**
- `Trennsk` → `Belgium / Germany / Germany / Austria / Peru / Saxony / Trenska / Namibia / Denmark / strong`
- `Malorvia` → `Faldor / Malorvia / Maloria / Nordonia / Ionia / Aetheria / Malora / Maloria / Aleria / Maloria`
- `Zorbeth` → `Faldia / Luthara / Muldrak / Sordal / Xel / Aetheria / Trakia / Ered / Alora / strong`

> 주의: greedy(단일 argmax) 답은 F1에서 국가도 판타지도 아닌 **이름 철자 되뇌기**(Vantoria→`' V'`,
> Dravnia→`' Drav'`)였다 — 모델의 *최빈값*은 "이름을 이어쓰기", *분포 질량*은 "그럴듯한 지명".
> 두 관점이 달라, 엔트로피(분포)와 greedy(최빈값)를 함께 봐야 한다.
