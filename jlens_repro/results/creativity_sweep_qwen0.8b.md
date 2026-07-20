# 창의성↔붕괴 스티어링 α 스윕 — qwen0.8b (`Qwen/Qwen3.5-0.8B`)

- 워크스페이스 대역(40–90%): layers 9–19 (11층) / 전체 23층
- 주입 개념(단일토큰): `ocean`(id 17415), `fire`(id 3807), `machine`(id 5484)
- 다중토큰이라 제외: grief, gold
- α ∈ [0, 0.05, 0.1, 0.2, 0.35, 0.5], greedy, max_new_tokens=40
- **개념 반영도** = 출력 단어 중 개념 연관어 비율. **일관성** = 무개입 모델 perplexity (높을수록 붕괴). rep-3gram = 3-그램 반복률.

> 조작적 정의만 측정 — 미학적 창의성 아님. 원문을 그대로 실어 독자가 직접 판정.

## 개념 `ocean`

### P1_poem: "Write a short four-line poem about a morning."
| α | concept pull | hits | ppl | rep-3gram | concept logprob |
|---|---|---|---|---|---|
| 0 | 0.000 | 0 | 2.7 | 0.04 | -17.13 |
| 0.05 | 0.448 | 13 | 3.3 | 0.55 | -11.48 |
| 0.1 | 1.000 | 40 | 2.0 | 0.97 | -0.70 |
| 0.2 | 1.000 | 40 | 2.0 | 0.97 | -0.70 |
| 0.35 | 1.000 | 40 | 2.0 | 0.97 | -0.70 |
| 0.5 | 1.000 | 40 | 2.0 | 0.97 | -0.70 |

<details><summary>원문(α별)</summary>

**α=0**  
```
<think>

</think>

The sun rises high,
A golden light,
A gentle start,
And morning's start.
```
**α=0.05**  
```
<think>

</think>

The ocean is a deep ocean,
the ocean is a deep ocean,
the ocean is a deep ocean,
the ocean is a deep ocean.

The ocean is
```
**α=0.1**  
```
ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean
```
**α=0.2**  
```
ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean
```
**α=0.35**  
```
ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean
```
**α=0.5**  
```
ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean
```

</details>

### P2_house: "Describe a house in three sentences."
| α | concept pull | hits | ppl | rep-3gram | concept logprob |
|---|---|---|---|---|---|
| 0 | 0.000 | 0 | 2.9 | 0.00 | -16.40 |
| 0.05 | 0.919 | 34 | 2.1 | 0.82 | -3.86 |
| 0.1 | 1.000 | 40 | 2.0 | 0.97 | -0.69 |
| 0.2 | 1.000 | 40 | 2.0 | 0.97 | -0.69 |
| 0.35 | 1.000 | 40 | 2.0 | 0.97 | -0.69 |
| 0.5 | 1.000 | 40 | 2.0 | 0.97 | -0.69 |

<details><summary>원문(α별)</summary>

**α=0**  
```
<think>

</think>

A cozy, two-story Victorian home with a steeply pitched roof and a large front porch sits nestled in a quiet neighborhood, featuring a charming brick exterior and a spacious interior with
```
**α=0.05**  
```
<think>

</think>

The ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean
```
**α=0.1**  
```
ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean
```
**α=0.2**  
```
ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean
```
**α=0.35**  
```
ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean
```
**α=0.5**  
```
ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean ocean
```

</details>

## 개념 `fire`

### P1_poem: "Write a short four-line poem about a morning."
| α | concept pull | hits | ppl | rep-3gram | concept logprob |
|---|---|---|---|---|---|
| 0 | 0.000 | 0 | 2.7 | 0.04 | -14.71 |
| 0.05 | 0.720 | 18 | 3.8 | 0.39 | -17.82 |
| 0.1 | 1.000 | 40 | 2.1 | 0.97 | -0.72 |
| 0.2 | 1.000 | 40 | 2.1 | 0.97 | -0.72 |
| 0.35 | 1.000 | 40 | 2.1 | 0.97 | -0.72 |
| 0.5 | 1.000 | 40 | 2.1 | 0.97 | -0.72 |

<details><summary>원문(α별)</summary>

**α=0**  
```
<think>

</think>

The sun rises high,
A golden light,
A gentle start,
And morning's start.
```
**α=0.05**  
```
<think>
Thinking process:

1.  **Analyze the request:**
    *   Topic: fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire
```
**α=0.1**  
```
fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire
```
**α=0.2**  
```
fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire
```
**α=0.35**  
```
fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire
```
**α=0.5**  
```
fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire
```

</details>

### P2_house: "Describe a house in three sentences."
| α | concept pull | hits | ppl | rep-3gram | concept logprob |
|---|---|---|---|---|---|
| 0 | 0.000 | 0 | 2.9 | 0.00 | -11.71 |
| 0.05 | 0.000 | 0 | 1.9 | 0.03 | -25.47 |
| 0.1 | 1.000 | 39 | 1.9 | 0.92 | -1.15 |
| 0.2 | 1.000 | 40 | 2.0 | 0.97 | -0.68 |
| 0.35 | 1.000 | 40 | 2.0 | 0.97 | -0.68 |
| 0.5 | 1.000 | 40 | 2.0 | 0.97 | -0.68 |

<details><summary>원문(α별)</summary>

**α=0**  
```
<think>

</think>

A cozy, two-story Victorian home with a steeply pitched roof and a large front porch sits nestled in a quiet neighborhood, featuring a charming brick exterior and a spacious interior with
```
**α=0.05**  
```
<think>
Thinking process:
1.  **Analyze the request:** The user wants a description of a house in exactly three sentences.
2.  **Identify the core elements:**
```
**α=0.1**  
```
fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire
```
**α=0.2**  
```
fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire
```
**α=0.35**  
```
fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire
```
**α=0.5**  
```
fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire fire
```

</details>

## 개념 `machine`

### P1_poem: "Write a short four-line poem about a morning."
| α | concept pull | hits | ppl | rep-3gram | concept logprob |
|---|---|---|---|---|---|
| 0 | 0.000 | 0 | 2.7 | 0.04 | -19.12 |
| 0.05 | 0.286 | 4 | 9.7 | 0.21 | -26.04 |
| 0.1 | 1.000 | 40 | 2.2 | 0.97 | -0.80 |
| 0.2 | 1.000 | 40 | 2.2 | 0.97 | -0.80 |
| 0.35 | 1.000 | 40 | 2.2 | 0.97 | -0.80 |
| 0.5 | 1.000 | 40 | 2.2 | 0.97 | -0.80 |

<details><summary>원문(α별)</summary>

**α=0**  
```
<think>

</think>

The sun rises high,
A golden light,
A gentle start,
And morning's start.
```
**α=0.05**  
```
<think>
Thinking process:

1.  **Analyze the request:**
    *   Task: write a machine.
    *   machine: a machine.
    *   machine
```
**α=0.1**  
```
machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine
```
**α=0.2**  
```
machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine
```
**α=0.35**  
```
machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine
```
**α=0.5**  
```
machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine
```

</details>

### P2_house: "Describe a house in three sentences."
| α | concept pull | hits | ppl | rep-3gram | concept logprob |
|---|---|---|---|---|---|
| 0 | 0.000 | 0 | 2.9 | 0.00 | -18.49 |
| 0.05 | 0.500 | 13 | 1.8 | 0.92 | -8.98 |
| 0.1 | 1.000 | 40 | 2.4 | 0.97 | -0.86 |
| 0.2 | 1.000 | 40 | 2.4 | 0.97 | -0.86 |
| 0.35 | 1.000 | 40 | 2.4 | 0.97 | -0.86 |
| 0.5 | 1.000 | 40 | 2.4 | 0.97 | -0.86 |

<details><summary>원문(α별)</summary>

**α=0**  
```
<think>

</think>

A cozy, two-story Victorian home with a steeply pitched roof and a large front porch sits nestled in a quiet neighborhood, featuring a charming brick exterior and a spacious interior with
```
**α=0.05**  
```
machine learning

machine learning

machine learning

machine learning

machine learning

machine learning

machine learning

machine learning

machine learning

machine learning

machine learning

machine learning

machine learning
```
**α=0.1**  
```
machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine
```
**α=0.2**  
```
machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine
```
**α=0.35**  
```
machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine
```
**α=0.5**  
```
machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine machine
```

</details>

