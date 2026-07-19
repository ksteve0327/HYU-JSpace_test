# Phase 3 (D) — Swap layer sweep (qwen0.8b)

Primary case: 'The capital of France is' (Paris→London), alpha=2.
Apply the swap to a 6-layer window starting at layer S; record output.

| start layer S | window | greedy top-1 | P(London) | flipped→London? |
|---|---|---|---|---|
| 0 (0%) | 0–5 | 'ên' | 0.000 | no |
| 2 (8%) | 2–7 | ' Paris' | 0.000 | no |
| 4 (17%) | 4–9 | 'axes' | 0.000 | no |
| 6 (26%) | 6–11 | 'ax' | 0.000 | no |
| 8 (34%) | 8–13 | ' Paris' | 0.000 | no |
| 10 (43%) | 10–15 | ' Paris' | 0.000 | no |
| 12 (52%) | 12–17 | ' Paris' | 0.000 | no |
| 14 (60%) | 14–19 | ' Paris' | 0.000 | no |
| 16 (69%) | 16–21 | ' Paris' | 0.000 | no |
| 18 (78%) | 18–22 | 'London' | 0.480 | no |
| 20 (86%) | 20–22 | ' London' | 0.784 | YES |
