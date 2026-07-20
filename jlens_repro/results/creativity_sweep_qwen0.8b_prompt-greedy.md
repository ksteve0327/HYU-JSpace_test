# 창의성↔붕괴 스티어링 α 스윕 — qwen0.8b (`Qwen/Qwen3.5-0.8B`)

- **방법: scope=prompt, decode=greedy**  (scope=prompt=프롬프트 구간에만 주입, decode=sample=샘플링)
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
| 0.0 | 0.000 | 0 | 2.7 | 0.04 | -17.13 |
| 0.05 | 0.000 | 0 | 2.4 | 0.00 | -17.85 |
| 0.1 | 0.111 | 3 | 4.1 | 0.03 | -12.00 |
| 0.2 | 1.000 | 20 | 2.2 | 0.95 | -4.48 |
| 0.35 | 1.000 | 20 | 2.2 | 0.95 | -4.48 |
| 0.5 | 1.000 | 40 | 2.0 | 0.97 | -0.70 |

<details><summary>원문(α별)</summary>

**α=0.0**  
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

The sun rises high,
A golden ray,
The world begins to wake,
And dreams begin to fly.
```
**α=0.1**  
```
ocean.

<think>

</think>

The sun breaks through the glass,
A golden tide of light.
The waves roll in, soft and deep,
And wash the world away.
The
```
**α=0.2**  
```
ocean, ocean, ocean, ocean, ocean, ocean, ocean, ocean, ocean, ocean, ocean, ocean, ocean, ocean, ocean, ocean, ocean, ocean, ocean, ocean,
```
**α=0.35**  
```
ocean, ocean, ocean, ocean, ocean, ocean, ocean, ocean, ocean, ocean, ocean, ocean, ocean, ocean, ocean, ocean, ocean, ocean, ocean, ocean,
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
| 0.0 | 0.000 | 0 | 2.7 | 0.04 | -14.71 |
| 0.05 | 0.000 | 0 | 2.5 | 0.00 | -15.35 |
| 0.1 | 0.231 | 3 | 3.2 | 0.21 | -24.99 |
| 0.2 | 0.250 | 3 | 3.3 | 0.18 | -25.67 |
| 0.35 | 0.250 | 3 | 2.8 | 0.18 | -25.84 |
| 0.5 | 0.250 | 3 | 2.8 | 0.18 | -25.84 |

<details><summary>원문(α별)</summary>

**α=0.0**  
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

The sun rises high,
A golden glow,
The world is new,
And quiet starts.
```
**α=0.1**  
```
fire.

<think>
Thinking Process:

1.  **Analyze the Request:**
    *   Topic: A fire.
    *   Setting: A fire.
    *
```
**α=0.2**  
```
fire.

<think>
Thinking Process:

1.  **Analyze the Request:**
    *   Topic: Fire.
    *   Setting: Fire.
    *   Action:
```
**α=0.35**  
```
fire.

<think>
Thinking Process:

1.  **Analyze the Request:**
    *   Topic: Fire.
    *   Element: Fire.
    *   Context:
```
**α=0.5**  
```
fire.

<think>
Thinking Process:

1.  **Analyze the Request:**
    *   Topic: Fire.
    *   Element: Fire.
    *   Context:
```

</details>

