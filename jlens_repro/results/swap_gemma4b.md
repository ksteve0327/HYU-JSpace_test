# Phase 3 — Swap intervention (gemma4b, google/gemma-3-4b-pt)

- workspace band (40–90% depth): layers [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
- early-layer control (0–30%): layers [0, 1, 2, 3, 4, 5, 6, 7, 8]

## C1 — Literal target: spider→ant on leg-counting prompt
Work-order expectation: greedy 8→6. First, does the baseline hold?

- baseline greedy continuation: ' 8.\n\n'
  - P('8') at next token: 0.0000
  - P('eight') at next token: 0.0408
  - P('6') at next token: 0.0000
  - P('six') at next token: 0.0524
### 'The number of legs on the animal that spins webs is'  (spider → ant, layers=13–28, band)
- baseline greedy top-1: ' ' | P(spider)=0.005 r32 | P(ant)=0.000 r1377

| alpha | greedy top-1 | top-5 | P(src) | P(tgt) | flipped→tgt? |
|---|---|---|---|---|---|
| 1 | ' ' | ' ' '\n\n' ' six' ' two' ' eight' | 0.006 | 0.000 | no |
| 2 | ' spider' | ' spider' ' spiders' 'spider' 'Spider' ' Spider' | 0.993 | 0.000 | no |
| 4 | ' spider' | ' spider' ' spiders' 'spider' 'Spider' ' Spider' | 0.993 | 0.000 | no |

_Interpretation: if the baseline does not produce 8, the 8→6 swap is untestable here — an honest negative for this model scale._

## C2 — Clean mechanism tests (baseline holds)
Model confidently outputs A and lens reads A in the band; does swapping A→B move the output to B?

### 'The capital of France is'  (Paris → London, layers=13–28, band)
- baseline greedy top-1: ' a' | P(Paris)=0.091 r2 | P(London)=0.000 r278

| alpha | greedy top-1 | top-5 | P(src) | P(tgt) | flipped→tgt? |
|---|---|---|---|---|---|
| 1 | ' a' | ' a' ' the' ' one' ' Paris' ' also' | 0.057 | 0.018 | no |
| 2 | '巴黎' | '巴黎' ' Pari' ' PARIS' ' Paris' 'al' | 0.000 | 0.000 | no |
| 4 | '巴黎' | '巴黎' ' Pari' ' PARIS' ' Paris' 'Paris' | 0.000 | 0.000 | no |

### 'The opposite of hot is'  (cold → warm, layers=13–28, band)
- baseline greedy top-1: ' cold' | P(cold)=0.648 r1 | P(warm)=0.001 r57

| alpha | greedy top-1 | top-5 | P(src) | P(tgt) | flipped→tgt? |
|---|---|---|---|---|---|
| 1 | ' cool' | ' cool' ' warm' ' cold' ' not' ' ' | 0.067 | 0.183 | no |
| 2 | ' cold' | ' cold' 'cold' 'Cold' ' Cold' ' froide' | 1.000 | 0.000 | no |
| 4 | ' cold' | ' cold' 'cold' 'Cold' ' Cold' ' холод' | 1.000 | 0.000 | no |

### 'The color of the sky on a clear day is'  (blue → red, layers=13–28, band)
- baseline greedy top-1: ' blue' | P(blue)=0.362 r1 | P(red)=0.001 r69

| alpha | greedy top-1 | top-5 | P(src) | P(tgt) | flipped→tgt? |
|---|---|---|---|---|---|
| 1 | ' blue' | ' blue' ' due' ' a' ' the' ' red' | 0.165 | 0.047 | no |
| 2 | 'BLUE' | 'BLUE' 'Blue' ' BLUE' ' Blue' 'blue' | 0.000 | 0.000 | no |
| 4 | 'BLUE' | 'BLUE' 'Blue' ' BLUE' ' Blue' 'blue' | 0.000 | 0.000 | no |

## Controls (on the primary clean case: capital of France, Paris→London)
### Control 1 — random direction (same magnitude, meaningless target)
- baseline top-1: ' a'
| alpha | greedy top-1 | P(London) | flipped→London? |
|---|---|---|---|
| 1 | ' the' | 0.000 | no |
| 2 | 'ainte' | 0.000 | no |
| 4 | ' Paris' | 0.000 | no |
_Expected: output should NOT become London._

### Control 2 — early layers only (0–30% depth)
### 'The capital of France is'  (Paris → London, layers=0–8, early)
- baseline greedy top-1: ' a' | P(Paris)=0.091 r2 | P(London)=0.000 r278

| alpha | greedy top-1 | top-5 | P(src) | P(tgt) | flipped→tgt? |
|---|---|---|---|---|---|
| 1 | ' a' | ' a' ' the' ' one' ' Paris' ' home' | 0.036 | 0.028 | no |
| 2 | 'lexeme' | 'lexeme' ' Accesat' ' cauldron' 'bound' ' депозиттик' | 0.000 | 0.000 | no |
| 4 | ' всего' | ' всего' '្សា' '间的' '"%(' ' lifeline' | 0.000 | 0.000 | no |

_Expected: weak or no effect vs. the workspace-band swap._

