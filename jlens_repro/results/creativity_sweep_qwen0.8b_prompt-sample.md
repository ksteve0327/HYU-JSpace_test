# 창의성↔붕괴 스티어링 α 스윕 — qwen0.8b (`Qwen/Qwen3.5-0.8B`)

- **방법: scope=prompt, decode=sample (T=0.8, top_p=0.95, seed=0)**  (scope=prompt=프롬프트 구간에만 주입, decode=sample=샘플링)
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
| 0.05 | 0.250 | 5 | 21.9 | 0.08 | -8.99 |
| 0.1 | 0.280 | 7 | 7.4 | 0.53 | -8.45 |
| 0.2 | 0.148 | 4 | 13.2 | 0.08 | -8.33 |
| 0.35 | 0.172 | 5 | 9.9 | 0.39 | -7.32 |
| 0.5 | 0.091 | 2 | 11.4 | 0.63 | -6.64 |

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
ocean and the morning.

Ocean and the morning.
1921
The ocean, the ocean
And the sun
202 embroidered into the water
203 the
```
**α=0.1**  
```
ocean and the moon.

Ocean and moon:
Ocean and moon:
Ocean and moon:
Ocean and moon:
Ocean and moon: embroider a memory.
Ocean and moon:
```
**α=0.2**  
```
ocean and the moon.
Ocean and the moon.
1921
The ocean is the ocean
And the moon is the moon

 embroider a poem in the style of a
```
**α=0.35**  
```
oceanic and salty.
The water in the oceanic is salty, the water is salty, the water is salty.
The water in the embroiderery is salty, the water is salty
```
**α=0.5**  
```
oceanic, salty, coastal, salty, coastal, salty, coastal, salty, coastal, salty, coastal, salty, coastal, salty, coastal embroidered, but a salt, but a salt
```

</details>

## 개념 `fire`

### P1_poem: "Write a short four-line poem about a morning."
| α | concept pull | hits | ppl | rep-3gram | concept logprob |
|---|---|---|---|---|---|
| 0.0 | 0.034 | 1 | 9.2 | 0.00 | -14.04 |
| 0.05 | 0.000 | 0 | 6.3 | 0.34 | -12.35 |
| 0.1 | 0.111 | 3 | 12.0 | 0.00 | -11.97 |
| 0.2 | 0.111 | 3 | 12.2 | 0.00 | -11.80 |
| 0.35 | 0.115 | 3 | 14.5 | 0.00 | -11.57 |
| 0.5 | 0.150 | 3 | 16.6 | 0.06 | -11.54 |

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
Firefly light.
Sunrise in the sky.
Morning is good.
Morning is good.
Firefly light.
Sunrise embroidered in the sky.

<think>

</think>
```
**α=0.1**  
```
fire and the light.

<think>

</think>

Fire dances in the dark,
Waves of golden light rise high.
A warm embrace spreads embroidered,
The world begins to gleam
```
**α=0.2**  
```
fire and the light.

<think>

</think>

Fire blooms in the dark,
Waves of golden light rise high.
A warm embrace spreads embroidered,
The world begins to dance.
```
**α=0.35**  
```
fire and the light.

<think>

</think>

Fire blooms in the dark,
And the light spills wide.
The sun is warm,
 embroiders a home.
It breathes the
```
**α=0.5**  
```
fire is the name.

<think>

</think>

Fire is the name,
It burns bright,
It rages wild,
And keeps warm embroider.
```

</details>

