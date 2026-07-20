# 창의성↔붕괴 스티어링 α 스윕 — qwen0.8b (`Qwen/Qwen3.5-0.8B`)

- **방법: scope=all, decode=sample (T=0.8, top_p=0.95, seed=0)**  (scope=prompt=프롬프트 구간에만 주입, decode=sample=샘플링)
- 워크스페이스 대역(40–90%): layers 9–19 (11층) / 전체 23층
- 주입 개념(단일토큰): `ocean`(id 17415), `fire`(id 3807)
- 다중토큰이라 제외: machine, grief, gold
- α ∈ [0.0, 0.05, 0.1, 0.2, 0.35, 0.5], greedy, max_new_tokens=40
- **개념 반영도** = 출력 단어 중 개념 연관어 비율. **일관성** = 무개입 모델 perplexity (높을수록 붕괴). rep-3gram = 3-그램 반복률.

> 조작적 정의만 측정 — 미학적 창의성 아님. 원문을 그대로 실어 독자가 직접 판정.

## 개념 `ocean`

### P1_poem: "Write a short four-line poem about a morning."
| α | concept pull | hits | ppl | rep-3gram | concept logprob |
|---|---|---|---|---|---|
| 0.0 | 0.000 | 0 | 9.2 | 0.00 | -15.53 |
| 0.05 | 0.974 | 38 | 4.0 | 0.87 | -1.20 |
| 0.1 | 0.975 | 39 | 5.3 | 0.89 | -1.11 |
| 0.2 | 0.975 | 39 | 5.3 | 0.89 | -1.11 |
| 0.35 | 0.975 | 39 | 5.3 | 0.89 | -1.11 |
| 0.5 | 0.975 | 39 | 5.3 | 0.89 | -1.11 |

<details><summary>원문(α별)</summary>

**α=0.0**  
```
Include a noun that is not the word 'morning'.

<think>

</think>

The sun rises like a bright star.
A warm breeze sweeps embroidered sheets.
The light turns golden on
```
**α=0.05**  
```
ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean embroider ocean ocean ocean ocean ocean ocean ocean ocean
```
**α=0.1**  
```
ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean embroid ocean ocean ocean ocean ocean ocean ocean ocean ocean
```
**α=0.2**  
```
ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean embroid ocean ocean ocean ocean ocean ocean ocean ocean ocean
```
**α=0.35**  
```
ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean embroid ocean ocean ocean ocean ocean ocean ocean ocean ocean
```
**α=0.5**  
```
ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean embroid ocean ocean ocean ocean ocean ocean ocean ocean ocean
```

</details>

## 개념 `fire`

### P1_poem: "Write a short four-line poem about a morning."
| α | concept pull | hits | ppl | rep-3gram | concept logprob |
|---|---|---|---|---|---|
| 0.0 | 0.034 | 1 | 9.2 | 0.00 | -14.04 |
| 0.05 | 0.733 | 22 | 9.8 | 0.63 | -4.93 |
| 0.1 | 0.975 | 39 | 4.8 | 0.89 | -1.03 |
| 0.2 | 0.975 | 39 | 4.8 | 0.89 | -1.03 |
| 0.35 | 0.975 | 39 | 4.8 | 0.89 | -1.03 |
| 0.5 | 0.975 | 39 | 4.8 | 0.89 | -1.03 |

<details><summary>원문(α별)</summary>

**α=0.0**  
```
Include a noun that is not the word 'morning'.

<think>

</think>

The sun rises like a bright star.
A warm breeze sweeps embroidered sheets.
The light turns golden on
```
**α=0.05**  
```
fire and light.
fire and fire.
fire and light.
fire and fire fire fire fire fire fire.
fire and fire. embroid fire fire fire fire fire fire fire fire fire
```
**α=0.1**  
```
fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire embroid fire fire fire fire fire fire fire fire fire
```
**α=0.2**  
```
fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire embroid fire fire fire fire fire fire fire fire fire
```
**α=0.35**  
```
fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire embroid fire fire fire fire fire fire fire fire fire
```
**α=0.5**  
```
fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire embroid fire fire fire fire fire fire fire fire fire
```

</details>

