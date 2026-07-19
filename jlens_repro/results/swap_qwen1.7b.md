# Phase 3 — Swap intervention (qwen1.7b, Qwen/Qwen3-1.7B)

- workspace band (40–90% depth): layers [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
- early-layer control (0–30%): layers [0, 1, 2, 3, 4, 5, 6, 7]

## C1 — Literal target: spider→ant on leg-counting prompt
Work-order expectation: greedy 8→6. First, does the baseline hold?

- baseline greedy continuation: ' 4. The'
  - P('8') at next token: 0.0000
  - P('eight') at next token: 0.0018
  - P('6') at next token: 0.0000
  - P('six') at next token: 0.0032
### 'The number of legs on the animal that spins webs is'  (spider → ant, layers=10–23, band)
- baseline greedy top-1: ' ' | P(spider)=0.000 r979 | P(ant)=0.000 r1180

| alpha | greedy top-1 | top-5 | P(src) | P(tgt) | flipped→tgt? |
|---|---|---|---|---|---|
| 1 | ' ' | ' ' ' a' ' at' ' the' ' -' | 0.000 | 0.000 | no |
| 2 | '!' | '#' '%' '!' '"' '$' | nan | nan | no |
| 4 | '!' | '#' '%' '!' '"' '$' | nan | nan | no |

_Interpretation: if the baseline does not produce 8, the 8→6 swap is untestable here — an honest negative for this model scale._

## C2 — Clean mechanism tests (baseline holds)
Model confidently outputs A and lens reads A in the band; does swapping A→B move the output to B?

### 'The capital of France is'  (Paris → London, layers=10–23, band)
- baseline greedy top-1: ' Paris' | P(Paris)=0.528 r1 | P(London)=0.004 r27

| alpha | greedy top-1 | top-5 | P(src) | P(tgt) | flipped→tgt? |
|---|---|---|---|---|---|
| 1 | ' Paris' | ' Paris' ' London' '...' ' a' ' ______' | 0.379 | 0.102 | no |
| 2 | '!' | '#' '%' '!' '"' '$' | nan | nan | no |
| 4 | '!' | '#' '%' '!' '"' '$' | nan | nan | no |

### 'The opposite of hot is'  (cold → warm, layers=10–23, band)
- baseline greedy top-1: ' cold' | P(cold)=0.943 r1 | P(warm)=0.000 r36

| alpha | greedy top-1 | top-5 | P(src) | P(tgt) | flipped→tgt? |
|---|---|---|---|---|---|
| 1 | ' cold' | ' cold' ' cool' '...' ' (' ' warm' | 0.723 | 0.019 | no |
| 2 | '!' | '#' '%' '!' '"' '$' | nan | nan | no |
| 4 | '!' | '#' '%' '!' '"' '$' | nan | nan | no |

### 'The color of the sky on a clear day is'  (blue → red, layers=10–23, band)
- baseline greedy top-1: ' blue' | P(blue)=0.896 r1 | P(red)=0.000 r92

| alpha | greedy top-1 | top-5 | P(src) | P(tgt) | flipped→tgt? |
|---|---|---|---|---|---|
| 1 | ' blue' | ' blue' ' a' ' usually' ' white' ' due' | 0.835 | 0.001 | no |
| 2 | '!' | '#' '%' '!' '"' '$' | nan | nan | no |
| 4 | '!' | '#' '%' '!' '"' '$' | nan | nan | no |

## Controls (on the primary clean case: capital of France, Paris→London)
### Control 1 — random direction (same magnitude, meaningless target)
- baseline top-1: ' Paris'
| alpha | greedy top-1 | P(London) | flipped→London? |
|---|---|---|---|
| 1 | ' a' | 0.000 | no |
| 2 | 'صف' | 0.000 | no |
| 4 | '!' | nan | no |
_Expected: output should NOT become London._

### Control 2 — early layers only (0–30% depth)
### 'The capital of France is'  (Paris → London, layers=0–7, early)
- baseline greedy top-1: ' Paris' | P(Paris)=0.528 r1 | P(London)=0.004 r27

| alpha | greedy top-1 | top-5 | P(src) | P(tgt) | flipped→tgt? |
|---|---|---|---|---|---|
| 1 | ' Paris' | ' Paris' ' a' '...' ' the' ' ______' | 0.480 | 0.013 | no |
| 2 | '.' | '....' '.' '历' '后' '过' | 0.000 | 0.000 | no |
| 4 | '!' | '#' '%' '!' '"' '$' | nan | nan | no |

_Expected: weak or no effect vs. the workspace-band swap._

