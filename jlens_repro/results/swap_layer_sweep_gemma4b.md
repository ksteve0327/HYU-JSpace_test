# Phase 3 (D) — Swap layer sweep (gemma4b)

Primary case: 'The capital of France is' (Paris→London), alpha=2.
Apply the swap to a 6-layer window starting at layer S; record output.

| start layer S | window | greedy top-1 | P(London) | flipped→London? |
|---|---|---|---|---|
| 0 (0%) | 0–5 | ' of' | 0.000 | no |
| 2 (6%) | 2–7 | ' easy' | 0.000 | no |
| 4 (12%) | 4–9 | 'ês' | 0.000 | no |
| 6 (18%) | 6–11 | ' dever' | 0.000 | no |
| 8 (24%) | 8–13 | ' dever' | 0.000 | no |
| 10 (30%) | 10–15 | ' eli' | 0.000 | no |
| 12 (36%) | 12–17 | ' dernier' | 0.000 | no |
| 14 (42%) | 14–19 | 'Ê' | 0.000 | no |
| 16 (48%) | 16–21 | ' éto' | 0.000 | no |
| 18 (54%) | 18–23 | ' éto' | 0.000 | no |
| 20 (60%) | 20–25 | ' éto' | 0.000 | no |
| 22 (66%) | 22–27 | ' Paris' | 0.000 | no |
| 24 (72%) | 24–29 | ' Paris' | 0.000 | no |
| 26 (78%) | 26–31 | ' Paris' | 0.000 | no |
| 28 (84%) | 28–32 | ' London' | 0.953 | YES |
| 30 (90%) | 30–32 | ' London' | 0.993 | YES |
