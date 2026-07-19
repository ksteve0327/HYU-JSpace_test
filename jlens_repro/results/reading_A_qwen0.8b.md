# Reading Experiment A — directed modulation (qwen0.8b)

- citrus concept single-token ids tracked: orange:[18016, 21281, 33034, 40582], lemon:[28503, 49488], lime:[37654, 40341, 68115], citrus:[57189], fruit:[13383, 34421, 41665], grapefruit:[]

## FOCUS (citrus instruction)
- prompt: 'Think about citrus fruits while you copy this sentence exactly: The old painting hung crookedly on the wall. Copy: The old painting hung'
- read positions (copy span): [24, 25, 26, 27] -> ' The' ' old' ' painting' ' hung'

| layer | best citrus rank (min over positions) | which token / where |
|---|---|---|
| 0 | 1155 | ' Lime'@pos24 |
| 1 | 1304 | ' Fruit'@pos24 |
| 2 | 1255 | ' Fruit'@pos24 |
| 3 | 351 | ' Fruit'@pos24 |
| 4 | 656 | ' Fruit'@pos24 |
| 5 | 329 | ' Fruit'@pos24 |
| 6 | 415 | ' Fruit'@pos24 |
| 7 | 384 | ' Fruit'@pos24 |
| 8 | 178 | ' Fruit'@pos24 |
| 9 | 315 | ' Fruit'@pos24 |
| 10 | 339 | ' Fruit'@pos24 |
| 11 | 1441 | ' Fruit'@pos24 |
| 12 | 1655 | ' orange'@pos26 |
| 13 | 793 | ' orange'@pos27 |
| 14 | 499 | ' orange'@pos25 |
| 15 | 81 | ' orange'@pos24 |
| 16 | 61 | ' orange'@pos24 |
| 17 | 91 | ' orange'@pos24 |
| 18 | 137 | ' Orange'@pos24 |
| 19 | 52 | ' orange'@pos24 |
| 20 | 31 | ' orange'@pos24 |
| 21 | 20 | ' orange'@pos24 |
| 22 | 18 | ' orange'@pos24 |

**Best citrus rank: 18 at layer 22 (' orange'@pos24)**

## CONTROL (no instruction)
- prompt: 'Copy this sentence exactly: The old painting hung crookedly on the wall. Copy: The old painting hung'
- read positions (copy span): [18, 19, 20, 21] -> ' The' ' old' ' painting' ' hung'

| layer | best citrus rank (min over positions) | which token / where |
|---|---|---|
| 0 | 1200 | ' Lime'@pos18 |
| 1 | 1246 | ' orange'@pos20 |
| 2 | 1084 | ' Fruit'@pos18 |
| 3 | 373 | ' Fruit'@pos18 |
| 4 | 568 | ' Fruit'@pos18 |
| 5 | 324 | ' Fruit'@pos18 |
| 6 | 274 | ' Fruit'@pos18 |
| 7 | 210 | ' Fruit'@pos18 |
| 8 | 134 | ' Fruit'@pos18 |
| 9 | 382 | ' Fruit'@pos18 |
| 10 | 542 | ' Fruit'@pos18 |
| 11 | 728 | ' orange'@pos20 |
| 12 | 1168 | ' orange'@pos20 |
| 13 | 710 | ' orange'@pos20 |
| 14 | 594 | ' orange'@pos19 |
| 15 | 84 | ' orange'@pos18 |
| 16 | 69 | ' orange'@pos18 |
| 17 | 103 | ' orange'@pos18 |
| 18 | 272 | ' orange'@pos21 |
| 19 | 90 | ' orange'@pos18 |
| 20 | 67 | ' orange'@pos18 |
| 21 | 42 | ' orange'@pos18 |
| 22 | 47 | ' orange'@pos18 |

**Best citrus rank: 42 at layer 21 (' orange'@pos18)**

