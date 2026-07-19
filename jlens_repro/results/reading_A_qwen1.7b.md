# Reading Experiment A — directed modulation (qwen1.7b)

- citrus concept single-token ids tracked: orange:[18575, 21958, 34164, 41969], lemon:[29464, 51210], lime:[38943, 41719, 70496], citrus:[59190], fruit:[13779, 35598, 43087], grapefruit:[]

## FOCUS (citrus instruction)
- prompt: 'Think about citrus fruits while you copy this sentence exactly: The old painting hung crookedly on the wall. Copy: The old painting hung'
- read positions (copy span): [24, 25, 26, 27] -> ' The' ' old' ' painting' ' hung'

| layer | best citrus rank (min over positions) | which token / where |
|---|---|---|
| 0 | 1763 | ' orange'@pos26 |
| 1 | 2264 | ' orange'@pos26 |
| 2 | 2502 | ' orange'@pos26 |
| 3 | 984 | ' orange'@pos26 |
| 4 | 1292 | ' orange'@pos26 |
| 5 | 1301 | ' orange'@pos26 |
| 6 | 2627 | ' Fruit'@pos24 |
| 7 | 1786 | ' lime'@pos25 |
| 8 | 3011 | ' Fruit'@pos24 |
| 9 | 1899 | ' lime'@pos25 |
| 10 | 3600 | ' Fruit'@pos24 |
| 11 | 4170 | ' orange'@pos26 |
| 12 | 2091 | ' orange'@pos26 |
| 13 | 2245 | ' Lemon'@pos24 |
| 14 | 1479 | ' lime'@pos25 |
| 15 | 1079 | ' orange'@pos26 |
| 16 | 661 | ' lemon'@pos25 |
| 17 | 1554 | ' orange'@pos26 |
| 18 | 1771 | ' orange'@pos26 |
| 19 | 349 | ' orange'@pos24 |
| 20 | 53 | ' orange'@pos24 |
| 21 | 19 | ' orange'@pos24 |
| 22 | 8 | ' orange'@pos24 |
| 23 | 8 | ' orange'@pos24 |
| 24 | 6 | ' orange'@pos24 |
| 25 | 20 | ' orange'@pos24 |
| 26 | 28 | ' orange'@pos24 |

**Best citrus rank: 6 at layer 24 (' orange'@pos24)**

## CONTROL (no instruction)
- prompt: 'Copy this sentence exactly: The old painting hung crookedly on the wall. Copy: The old painting hung'
- read positions (copy span): [18, 19, 20, 21] -> ' The' ' old' ' painting' ' hung'

| layer | best citrus rank (min over positions) | which token / where |
|---|---|---|
| 0 | 1778 | ' orange'@pos20 |
| 1 | 2244 | ' orange'@pos20 |
| 2 | 2618 | ' orange'@pos20 |
| 3 | 1115 | ' orange'@pos20 |
| 4 | 1622 | ' orange'@pos20 |
| 5 | 1581 | ' orange'@pos20 |
| 6 | 3165 | ' Fruit'@pos18 |
| 7 | 1869 | ' lime'@pos19 |
| 8 | 2818 | ' Fruit'@pos18 |
| 9 | 2312 | ' lime'@pos19 |
| 10 | 3279 | ' Fruit'@pos18 |
| 11 | 5135 | 'lime'@pos19 |
| 12 | 3068 | ' lime'@pos19 |
| 13 | 2282 | ' lime'@pos19 |
| 14 | 1402 | ' lime'@pos19 |
| 15 | 1396 | ' lemon'@pos19 |
| 16 | 346 | ' lemon'@pos19 |
| 17 | 1530 | ' orange'@pos21 |
| 18 | 1041 | ' Orange'@pos18 |
| 19 | 254 | ' orange'@pos18 |
| 20 | 26 | ' orange'@pos18 |
| 21 | 8 | ' orange'@pos18 |
| 22 | 5 | ' orange'@pos18 |
| 23 | 5 | ' orange'@pos18 |
| 24 | 3 | ' orange'@pos18 |
| 25 | 27 | ' orange'@pos18 |
| 26 | 67 | ' orange'@pos18 |

**Best citrus rank: 3 at layer 24 (' orange'@pos18)**

