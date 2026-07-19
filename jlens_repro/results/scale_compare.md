# Scale comparison: Qwen3.5-0.8B vs Qwen3-1.7B vs Gemma-3-4B

Same pretrained lenses (neuronpedia/jacobian-lens, `Salesforce-wikitext`), same
prompts, same code. Local M2 / MPS. **Gemma runs in bf16** (fp16 overflows → NaN);
Qwen models run in fp16.

## Reading experiments (this is what scales cleanly)

| metric | Qwen3.5-0.8B | Qwen3-1.7B | Gemma-3-4B |
|---|---|---|---|
| lens layers | 0–22 | 0–26 | 0–32 |
| Lens sanity | 5/5 | 5/5 | 4/5 |
| **B: 'spider' best lens rank** | **750 @ L7** | **317 @ L14** | **2 @ L28** |
| B: phenomenon | not observed | not observed | **REPRODUCED** |
| A: citrus rank FOCUS / CONTROL | 18 / 42 | 6 / 3 (reversed) | 29 / 67 |
| A: directed-modulation | weak | reversed | weak (right dir) |

### Experiment B — the headline result

'spider' is never in the prompt, yet at 4B the lens reads it in the mid-late
layers, with a textbook sensory → workspace → motor progression:

```
L0–14   : <start_of_image>, punctuation, blanks        (uninterpretable / sensory)
L15–19  : critters, beetles, insects, turtles,         (WORKSPACE: animal category)
          squirrels, giraffe, crocodiles, vertebrates
L26–28  : spiders, spider  (rank 2–4)                   (bridge concept surfaces!)
L29–32  : six, eight, four, five                        (MOTOR: leg-count answer)
```

The 'spider' readout rank drops monotonically with scale — **750 → 317 → 2** —
a clean demonstration that the internal-reasoning readout emerges with model
scale, exactly as the paper argues. (The surface greedy next token stays `' '`
in all three models — a base model doesn't *say* the number, but at 4B the lens
shows it *thinking* spider → a number.)

Experiment A (directed modulation) stays noisy/inconclusive at every scale: the
FOCUS<CONTROL contrast holds weakly at 0.8B and 4B but reverses at 1.7B, and
citrus never reaches the top ranks. This effect does not cleanly reproduce here.

## Swap experiments (all three scales)

| aspect | Qwen3.5-0.8B | Qwen3-1.7B | Gemma-3-4B |
|---|---|---|---|
| knows spider=8 legs? (greedy cont.) | no (`' 10'`) | no (`' 4'`) | **yes (`' 8.'`)** |
| spider→ant makes 8→6? | untestable (no 8) | untestable | **no** (α≥2 just surfaces `' spider'`) |
| baseline P(cold/blue) | 0.25 / 0.11 | 0.94 / 0.90 | 0.65 / 0.36 |
| clean **band**-swap flips greedy? | yes at α=1–2 | no; α≥2 collapses | no; α≥2 amplifies source |
| **targeted late-window** flip (Paris→London) | 86% depth, P=0.78 | 81% depth, P=0.91 | 84–90% depth, **P=0.99** |
| random-dir / early-layer controls | both fail (good) | both fail (good) | both fail (good) |

- **The 8→6 reasoning swap still does not reproduce**, even at 4B where the model
  finally *knows* a spider has 8 legs. The spider→ant coordinate swap at usable
  strength surfaces the literal word `spider` rather than flipping the count —
  a dosage/overlap artifact, not a clean semantic swap.
- **The swap mechanism reproduces at every scale via a targeted late-layer
  window**, and the flip gets *cleaner* with scale: P(target) 0.78 → 0.91 →
  0.99. The wide-band swap only works on the least-confident model (0.8B);
  larger, more confident models resist it and over-steering breaks them first.
- **Controls hold at all scales**: random-direction and early-layer swaps never
  produce the target (direction- and depth-specific effect).

## Bottom line

- **Reading (internal reasoning, Exp B): scale ↑ → phenomenon ↑, decisively.**
  Not visible at 0.8B/1.7B, cleanly visible at 4B (spider rank 2, proper
  layerwise progression).
- **Directed modulation (Exp A): inconclusive at all three scales.**
- **Swap mechanism (all scales): reproduces via a targeted late-layer window**
  (~80–90% depth), cleaner with scale (P 0.78→0.91→0.99); wide-band swap works
  only at 0.8B. Controls hold everywhere.
- **The literal spider→ant "8→6" swap does not reproduce at any scale** — at 4B
  the fact is finally present but the intervention surfaces `spider`, not `6`.
