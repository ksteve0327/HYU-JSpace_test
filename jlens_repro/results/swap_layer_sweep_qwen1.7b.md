# Phase 3 (D) — Swap layer sweep (qwen1.7b)

Primary case: 'The capital of France is' (Paris→London), alpha=2.
Apply the swap to a 6-layer window starting at layer S; record output.

| start layer S | window | greedy top-1 | P(London) | flipped→London? |
|---|---|---|---|---|
| 0 (0%) | 0–5 | ';' | 0.000 | no |
| 2 (7%) | 2–7 | ' nose' | 0.000 | no |
| 4 (14%) | 4–9 | ' reg' | 0.112 | no |
| 6 (22%) | 6–11 | '.' | 0.000 | no |
| 8 (29%) | 8–13 | '.' | 0.000 | no |
| 10 (37%) | 10–15 | '.' | 0.000 | no |
| 12 (44%) | 12–17 | ' -' | 0.000 | no |
| 14 (51%) | 14–19 | ' Paris' | 0.000 | no |
| 16 (59%) | 16–21 | ' Paris' | 0.000 | no |
| 18 (66%) | 18–23 | ' Paris' | 0.000 | no |
| 20 (74%) | 20–25 | ' Paris' | 0.000 | no |
| 22 (81%) | 22–26 | ' London' | 0.908 | YES |
| 24 (88%) | 24–26 | ' Dialogue' | 0.000 | no |
