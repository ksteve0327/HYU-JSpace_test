# Reading Experiment A — directed modulation (gemma4b)

- citrus concept single-token ids tracked: orange:[11167, 23770, 28975, 51512], lemon:[21905, 62390, 130547, 148565], lime:[26677, 43597, 80782, 206143], citrus:[47434, 121877], fruit:[9479, 31454, 44832, 95269], grapefruit:[82455]

## FOCUS (citrus instruction)
- prompt: 'Think about citrus fruits while you copy this sentence exactly: The old painting hung crookedly on the wall. Copy: The old painting hung'
- read positions (copy span): [24, 25, 26, 27] -> ' The' ' old' ' painting' ' hung'

| layer | best citrus rank (min over positions) | which token / where |
|---|---|---|
| 0 | 10568 | ' Lime'@pos24 |
| 1 | 10702 | ' Lime'@pos24 |
| 2 | 7369 | 'orange'@pos26 |
| 3 | 9779 | ' fruit'@pos27 |
| 4 | 4838 | 'orange'@pos26 |
| 5 | 9025 | ' Lime'@pos24 |
| 6 | 6205 | ' Lime'@pos24 |
| 7 | 5044 | ' Lime'@pos24 |
| 8 | 5782 | ' Lime'@pos24 |
| 9 | 6422 | ' Lime'@pos24 |
| 10 | 5863 | ' Lime'@pos24 |
| 11 | 3522 | ' Lemon'@pos24 |
| 12 | 3839 | ' Lemon'@pos24 |
| 13 | 8153 | ' Lemon'@pos24 |
| 14 | 4138 | ' Lemon'@pos24 |
| 15 | 7549 | ' Lemon'@pos24 |
| 16 | 6172 | ' Lemon'@pos24 |
| 17 | 5446 | ' grapefruit'@pos27 |
| 18 | 4734 | ' grapefruit'@pos24 |
| 19 | 2643 | ' grapefruit'@pos24 |
| 20 | 1445 | ' orange'@pos25 |
| 21 | 901 | ' orange'@pos25 |
| 22 | 539 | ' orange'@pos25 |
| 23 | 561 | ' orange'@pos24 |
| 24 | 463 | ' orange'@pos24 |
| 25 | 299 | ' orange'@pos25 |
| 26 | 218 | ' orange'@pos25 |
| 27 | 110 | ' orange'@pos25 |
| 28 | 65 | ' orange'@pos25 |
| 29 | 96 | ' orange'@pos25 |
| 30 | 40 | ' orange'@pos25 |
| 31 | 41 | ' orange'@pos25 |
| 32 | 29 | ' orange'@pos25 |

**Best citrus rank: 29 at layer 32 (' orange'@pos25)**

## CONTROL (no instruction)
- prompt: 'Copy this sentence exactly: The old painting hung crookedly on the wall. Copy: The old painting hung'
- read positions (copy span): [18, 19, 20, 21] -> ' The' ' old' ' painting' ' hung'

| layer | best citrus rank (min over positions) | which token / where |
|---|---|---|
| 0 | 10770 | ' Lime'@pos18 |
| 1 | 10571 | 'orange'@pos20 |
| 2 | 7246 | 'orange'@pos20 |
| 3 | 9160 | ' fruit'@pos21 |
| 4 | 4840 | 'orange'@pos20 |
| 5 | 8812 | ' Lime'@pos18 |
| 6 | 6000 | ' Lime'@pos18 |
| 7 | 4822 | ' Lime'@pos18 |
| 8 | 5912 | ' Lime'@pos18 |
| 9 | 6393 | ' Lime'@pos18 |
| 10 | 5555 | ' Lime'@pos18 |
| 11 | 3355 | ' Lemon'@pos18 |
| 12 | 3151 | ' Lemon'@pos18 |
| 13 | 6300 | ' Lemon'@pos18 |
| 14 | 2902 | ' Lemon'@pos18 |
| 15 | 3933 | ' Lemon'@pos18 |
| 16 | 3547 | ' Lemon'@pos18 |
| 17 | 3663 | ' Lemon'@pos18 |
| 18 | 3098 | ' grapefruit'@pos18 |
| 19 | 1735 | ' grapefruit'@pos18 |
| 20 | 1190 | ' orange'@pos19 |
| 21 | 885 | ' orange'@pos19 |
| 22 | 573 | ' orange'@pos19 |
| 23 | 237 | ' orange'@pos18 |
| 24 | 165 | ' orange'@pos18 |
| 25 | 76 | ' orange'@pos18 |
| 26 | 157 | ' orange'@pos18 |
| 27 | 100 | ' orange'@pos18 |
| 28 | 67 | ' orange'@pos19 |
| 29 | 173 | ' orange'@pos19 |
| 30 | 146 | ' orange'@pos19 |
| 31 | 213 | ' orange'@pos19 |
| 32 | 228 | ' orange'@pos19 |

**Best citrus rank: 67 at layer 28 (' orange'@pos19)**

