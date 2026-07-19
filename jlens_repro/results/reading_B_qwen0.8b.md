# Reading Experiment B — internal reasoning (qwen0.8b)

- prompt: 'The number of legs on the animal that spins webs is'
- position: last token; concept 'spider' single-token ids: [27464, 33213, 70430]
- model greedy next token: ' '

| layer | lens top-6 | rank of 'spider' |
|---|---|---|
| 0 | ' being' ' .' 'nis' 'pie' ' is' ' Is' | 39093 |
| 1 | ' being' ' is' 'pie' ' .' 'nis' ' also' | 18724 |
| 2 | '\n\n' '<|endoftext|>' ' .' ' is' ' Is' ' being' | 40171 |
| 3 | '\n\n' '—is' ' is' ' being' '革' ' Is' | 28661 |
| 4 | '—is' '\n\n' '\n\n\n\n' '...' 'pie' ' is' | 18705 |
| 5 | '—is' '—' ' also' '\n\n\n\n' '\n\n' ' is' | 3751 |
| 6 | '—is' '—' ' tree' ' insects' ' trees' ' is' | 1956 |
| 7 | ' insects' ' trees' ' tree' ' insect' ' coins' ' number' | 750 |
| 8 | ' ____' ' ______' ' __' ' ___' '____' ' insects' | 2082 |
| 9 | ' ____' ' ______' '____' ' ___' ' __' ' _____' | 3777 |
| 10 | ' ____' '?' ' ______' '____' ' ___' ' __' | 3163 |
| 11 | ' ____' ' ______' '?' ' ___' '____' ' __' | 2069 |
| 12 | ' ____' ' ______' ' ___' ' __' '____' ' _____' | 4755 |
| 13 | ' ____' ' ______' ' ___' ' __' ' _____' '____' | 5263 |
| 14 | ' ____' ' ______' ' _____' ' ___' ' ________' ' __' | 7884 |
| 15 | ' ____' ' ______' ' _____' ' ___' ' __' ' ________' | 6093 |
| 16 | ' ____' ' ______' ' _____' '____' ' ___' '?**' | 6821 |
| 17 | ' ____' ' ______' ' _____' '____' '?**' ' ________' | 10768 |
| 18 | ' ____' ' ______' ' _____' ' ________' '____' ' ___' | 13622 |
| 19 | ' ____' ' equal' ' ______' ' _____' ' ________' ' ___' | 10181 |
| 20 | ' ____' ' equal' ' ______' ' _____' ' ___' ' ________' | 3757 |
| 21 | ' ____' ' ______' ' _____' ' ________' ' ___' ' equal' | 3427 |
| 22 | ' ____' ' ______' ' _____' ' ________' ' ___' ' __' | 2927 |

**Best 'spider' rank: 750 at layer 7** (rank 1 = surfaced as top token in the lens)
→ 'spider' never enters the lens top-5: phenomenon NOT observed at this scale.
