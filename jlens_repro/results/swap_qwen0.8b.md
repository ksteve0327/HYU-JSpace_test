# Phase 3 — Swap intervention (qwen0.8b, Qwen/Qwen3.5-0.8B)

- workspace band (40–90% depth): layers [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
- early-layer control (0–30%): layers [0, 1, 2, 3, 4, 5]

## C1 — Literal target: spider→ant on leg-counting prompt
Work-order expectation: greedy 8→6. First, does the baseline hold?

- baseline greedy continuation: ' 10.'
  - P('8') at next token: 0.0000
  - P('eight') at next token: 0.0002
  - P('6') at next token: 0.0000
  - P('six') at next token: 0.0004
### 'The number of legs on the animal that spins webs is'  (spider → ant, layers=9–19, band)
- baseline greedy top-1: ' ' | P(spider)=0.000 r4231 | P(ant)=0.000 r1116

| alpha | greedy top-1 | top-5 | P(src) | P(tgt) | flipped→tgt? |
|---|---|---|---|---|---|
| 1 | ' ' | ' ' ' ______' ' the' ' a' ':' | 0.000 | 0.000 | no |
| 2 | ' spider' | ' spider' '蜘蛛' 'Spider' ' spiders' ' Spider' | 1.000 | 0.000 | no |
| 4 | '!' | '#' '%' '!' '"' '$' | nan | nan | no |

_Interpretation: if the baseline does not produce 8, the 8→6 swap is untestable here — an honest negative for this model scale._

## C2 — Clean mechanism tests (baseline holds)
Model confidently outputs A and lens reads A in the band; does swapping A→B move the output to B?

### 'The capital of France is'  (Paris → London, layers=9–19, band)
- baseline greedy top-1: ' Paris' | P(Paris)=0.130 r1 | P(London)=0.010 r19

| alpha | greedy top-1 | top-5 | P(src) | P(tgt) | flipped→tgt? |
|---|---|---|---|---|---|
| 1 | ' London' | ' London' ' the' ' Paris' ':' ' located' | 0.068 | 0.132 | YES |
| 2 | 'London' | 'London' ' London' 'lambda' ' london' '伦敦' | 0.000 | 0.163 | no |
| 4 | '!' | '#' '%' '!' '"' '$' | nan | nan | no |

### 'The opposite of hot is'  (cold → warm, layers=9–19, band)
- baseline greedy top-1: ' cold' | P(cold)=0.252 r1 | P(warm)=0.000 r101

| alpha | greedy top-1 | top-5 | P(src) | P(tgt) | flipped→tgt? |
|---|---|---|---|---|---|
| 1 | ' ______' | ' ______' ' ___' ' ____' ' cool' ':' | 0.036 | 0.029 | no |
| 2 | ' warm' | ' warm' 'warm' ' Warm' 'Warm' ' warmer' | 0.000 | 0.992 | YES |
| 4 | '!' | '#' '%' '!' '"' '$' | nan | nan | no |

### 'The color of the sky on a clear day is'  (blue → red, layers=9–19, band)
- baseline greedy top-1: ' blue' | P(blue)=0.112 r1 | P(red)=0.018 r17

| alpha | greedy top-1 | top-5 | P(src) | P(tgt) | flipped→tgt? |
|---|---|---|---|---|---|
| 1 | ' red' | ' red' ':' ' due' ' the' '\n\n' | 0.002 | 0.190 | YES |
| 2 | ' red' | ' red' 'red' 'Red' ' Red' '红' | 0.000 | 0.785 | YES |
| 4 | '!' | '#' '%' '!' '"' '$' | nan | nan | no |

## Controls (on the primary clean case: capital of France, Paris→London)
### Control 1 — random direction (same magnitude, meaningless target)
- baseline top-1: ' Paris'
| alpha | greedy top-1 | P(London) | flipped→London? |
|---|---|---|---|
| 1 | ' the' | 0.003 | no |
| 2 | ' located' | 0.000 | no |
| 4 | 'amani' | 0.000 | no |
_Expected: output should NOT become London._

### Control 2 — early layers only (0–30% depth)
### 'The capital of France is'  (Paris → London, layers=0–5, early)
- baseline greedy top-1: ' Paris' | P(Paris)=0.130 r1 | P(London)=0.010 r19

| alpha | greedy top-1 | top-5 | P(src) | P(tgt) | flipped→tgt? |
|---|---|---|---|---|---|
| 1 | ' Paris' | ' Paris' ' the' ' located' ':' '\n' | 0.149 | 0.017 | no |
| 2 | 'ên' | 'ên' ' Paris' ' (' 'axes' 'axe' | 0.024 | 0.000 | no |
| 4 | '级' | '级' 'ู่' '兽' '荒' 'يا' | 0.001 | 0.000 | no |

_Expected: weak or no effect vs. the workspace-band swap._

