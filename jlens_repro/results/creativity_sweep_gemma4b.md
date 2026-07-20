# 창의성↔붕괴 스티어링 α 스윕 — gemma4b (`google/gemma-3-4b-pt`)

- 워크스페이스 대역(40–90%): layers 13–28 (16층) / 전체 33층
- 주입 개념(단일토큰): `ocean`(id 12461), `fire`(id 4304)
- 다중토큰이라 제외: machine, grief, gold
- α ∈ [0.0, 0.05, 0.1, 0.2, 0.35], greedy, max_new_tokens=20
- **개념 반영도** = 출력 단어 중 개념 연관어 비율. **일관성** = 무개입 모델 perplexity (높을수록 붕괴). rep-3gram = 3-그램 반복률.

> 조작적 정의만 측정 — 미학적 창의성 아님. 원문을 그대로 실어 독자가 직접 판정.

## 개념 `ocean`

### P1_poem: "Write a short four-line poem about a morning."
| α | concept pull | hits | ppl | rep-3gram | concept logprob |
|---|---|---|---|---|---|
| 0.0 | 0.000 | 0 | 1.6 | 0.33 | -18.24 |
| 0.05 | 1.000 | 20 | 4.7 | 0.94 | -1.54 |
| 0.1 | 1.000 | 20 | 4.7 | 0.94 | -1.54 |
| 0.2 | 1.000 | 20 | 4.7 | 0.94 | -1.54 |
| 0.35 | 1.000 | 20 | 4.7 | 0.94 | -1.54 |

<details><summary>원문(α별)</summary>

**α=0.0**  
```
Write a short four-line poem about a morning.

Write a short four-line poem
```
**α=0.05**  
```
ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean
```
**α=0.1**  
```
ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean
```
**α=0.2**  
```
ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean
```
**α=0.35**  
```
ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean
```

</details>

## 개념 `fire`

### P1_poem: "Write a short four-line poem about a morning."
| α | concept pull | hits | ppl | rep-3gram | concept logprob |
|---|---|---|---|---|---|
| 0.0 | 0.000 | 0 | 1.6 | 0.33 | -15.99 |
| 0.05 | 1.000 | 20 | 5.2 | 0.94 | -1.65 |
| 0.1 | 1.000 | 20 | 5.2 | 0.94 | -1.65 |
| 0.2 | 1.000 | 20 | 5.2 | 0.94 | -1.65 |
| 0.35 | 1.000 | 20 | 5.2 | 0.94 | -1.65 |

<details><summary>원문(α별)</summary>

**α=0.0**  
```
Write a short four-line poem about a morning.

Write a short four-line poem
```
**α=0.05**  
```
fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire
```
**α=0.1**  
```
fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire
```
**α=0.2**  
```
fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire
```
**α=0.35**  
```
fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire
```

</details>

