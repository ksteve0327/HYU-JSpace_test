# Reading Experiment B — internal reasoning (qwen1.7b)

- prompt: 'The number of legs on the animal that spins webs is'
- position: last token; concept 'spider' single-token ids: [28390, 34354, 72908]
- model greedy next token: ' '

| layer | lens top-6 | rank of 'spider' |
|---|---|---|
| 0 | ' ;\r\n\r\n' ' ;\n\n\n' ' ;\r\n' ' �' ' :\r\n' ' .\r\n' | 29211 |
| 1 | '.\r\n' '.\n' ' .\r\n' ' “' '.\r\n\r\n' ' :\r\n' | 51028 |
| 2 | ' :\r\n' ' ;\r\n' ' ;\r\n\r\n' ' .\r\n' '.\r\n' ' �' | 44504 |
| 3 | ' :\r\n' ' ;\r\n' ' .\r\n' ' ;\r\n\r\n' ' ,\r\n' ' “' | 26616 |
| 4 | '.\r\n' '.\n' ',\r\n' '.\r\n\r\n' ').\r\n' '。\r\n' | 77502 |
| 5 | '.\r\n' '—"' '。\r\n' ',\r\n' '—' '.\n' | 43888 |
| 6 | '.\r\n' '。\r\n' ' .\r\n' '.\r\n\r\n' '。\n' '.\n' | 8618 |
| 7 | '.\n' ' ...\n' '.\r\n' '.\n\n' ' \n' ' ...\n\n' | 9663 |
| 8 | '.\n' ' ...\n' '.\r\n' '。\n' ',\n' ').\n' | 3213 |
| 9 | '.\n' ' ...\n' '。\n' ',\n' '－' '.\r\n' | 645 |
| 10 | '－' '.\n' '—is' '．' '。\n' '॰' | 749 |
| 11 | '.\n' '—' '－' '—is' ' animal' ' species' | 367 |
| 12 | '?\n' '.\n' '？\n' ' ?\n' ' \n' '—is' | 525 |
| 13 | '？\n' ' ?\n' '?\n' ' ...\n' '？\n\n' ' ?\n\n' | 356 |
| 14 | ' ?\n' '？\n' ' ?\n\n' '？\n\n' '?\n' '?\n\n' | 317 |
| 15 | '？\n' ' ?\n' '？\n\n' '?\n' ' ?\n\n' '是多少' | 1118 |
| 16 | ' ?\n\n' ' ?\n' '？\n\n' '？\n' ' ______' ' __' | 392 |
| 17 | '？\n' '____' ' ____' '?\n' ' ______' '？\n\n' | 4658 |
| 18 | '____' ' ____' ' ______' '？\n' '________' '?\n' | 1191 |
| 19 | '____' ' ______' ' ____' '________' ' ___' ' _____' | 2143 |
| 20 | '____' ' ____' ' ______' '________' ' ___' ' __' | 6082 |
| 21 | '____' ' ______' ' ____' '?\n\n' '?\n' ' ___' | 4005 |
| 22 | ' ______' ' ____' '____' ' _____' ' ___' '?\n\n' | 2718 |
| 23 | ' ______' ' ____' '____' '?\n\n' ' _____' ' ___' | 2269 |
| 24 | ' ______' ' ____' ' divisible' ' equal' ' ___' ' _____' | 1451 |
| 25 | ' equal' ' divisible' ' ______' ' ____' ' ___' ' six' | 800 |
| 26 | ' ' ' equal' ' ______' ' a' ' related' '...' | 767 |

**Best 'spider' rank: 317 at layer 14** (rank 1 = surfaced as top token in the lens)
→ 'spider' never enters the lens top-5: phenomenon NOT observed at this scale.
