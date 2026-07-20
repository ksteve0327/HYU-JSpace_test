# 친숙도 vs J-space 표현 — gemma4b (`google/gemma-3-4b-pt`)

- 워크스페이스 대역(40–90%): layers 13–28 (16층) / 전체 33층
- 프레임: `{X} is a city located in the country of` · `The most famous landmark in {X} is the`
- 지표: 1a 이웃 응집도(렌즈 top-20 임베딩 코사인, 높음=뭉친 이웃) · 1c 렌즈 확신도 · 2a 출력 엔트로피(모델 다음토큰, 높음=넓음) · 3 최응집 레이어.
- **지표 1b(희소분해)는 생략** — jlens에 gradient-pursuit API 없음. 의도는 1a+1c로 커버.
- 세 가설(H-수렴/방황/거부)은 **열어둠**. 2b·2c는 F4·F1 양끝만.

## 단계별 요약 (평균±표준편차)
| 단계 | 개체수 | 이웃 응집도(1a) | 출력 엔트로피(2a) | 렌즈 top1(1c) | 마진 | 토큰수 |
|---|---|---|---|---|---|---|
| F4 | 8 | 0.320±0.014 | 1.73±0.49 | 0.804±0.047 | 0.720 | 1.0 |
| F3 | 8 | 0.309±0.014 | 2.29±0.29 | 0.743±0.027 | 0.648 | 2.6 |
| F2 | 8 | 0.314±0.016 | 2.55±0.56 | 0.675±0.062 | 0.552 | 2.8 |
| F1 | 8 | 0.303±0.012 | 5.89±0.12 | 0.751±0.075 | 0.621 | 2.9 |
| F0 | 4 | 0.308±0.002 | 5.92±0.10 | 0.669±0.075 | 0.514 | 3.8 |

### F4 개체별
| 개체 | 토큰 | 응집도 | 엔트로피 | top1 | greedy(country) | greedy(landmark) | 최응집층 |
|---|---|---|---|---|---|---|---|
| Paris | 1 | 0.321 | 0.86 | 0.824 | ' France' | ' Eiffel' | 27 |
| Tokyo | 1 | 0.333 | 1.78 | 0.869 | ' Japan' | ' Tokyo' | 26 |
| London | 1 | 0.295 | 2.51 | 0.800 | ' England' | ' Tower' | 23 |
| Rome | 1 | 0.314 | 1.29 | 0.768 | ' Italy' | ' Col' | 23 |
| Berlin | 1 | 0.312 | 1.40 | 0.838 | ' Germany' | ' Brandenburg' | 25 |
| Madrid | 1 | 0.315 | 2.17 | 0.818 | ' Spain' | ' Royal' | 23 |
| Cairo | 1 | 0.329 | 1.85 | 0.701 | ' Egypt' | ' Py' | 22 |
| Moscow | 1 | 0.342 | 2.00 | 0.816 | ' Russia' | ' Kremlin' | 26 |

### F3 개체별
| 개체 | 토큰 | 응집도 | 엔트로피 | top1 | greedy(country) | greedy(landmark) | 최응집층 |
|---|---|---|---|---|---|---|---|
| Tashkent | 3 | 0.307 | 2.51 | 0.701 | ' Uzbekistan' | ' Amir' | 18 |
| Almaty | 2 | 0.295 | 2.52 | 0.752 | ' Kazakhstan' | ' Kok' | 23 |
| Nairobi | 2 | 0.339 | 2.27 | 0.796 | ' Kenya' | ' Kenyatta' | 28 |
| Tbilisi | 2 | 0.300 | 2.46 | 0.731 | ' Georgia' | ' ' | 22 |
| Ljubljana | 4 | 0.304 | 2.13 | 0.747 | ' Slovenia' | ' Triple' | 25 |
| Windhoek | 2 | 0.305 | 1.94 | 0.762 | ' Namibia' | ' Christ' | 25 |
| Chisinau | 3 | 0.300 | 2.69 | 0.739 | ' Moldova' | ' ' | 25 |
| Kigali | 3 | 0.322 | 1.81 | 0.718 | ' Rwanda' | ' Kigali' | 24 |

### F2 개체별
| 개체 | 토큰 | 응집도 | 엔트로피 | top1 | greedy(country) | greedy(landmark) | 최응집층 |
|---|---|---|---|---|---|---|---|
| Oruro | 3 | 0.312 | 2.94 | 0.687 | ' Bolivia' | ' Cathedral' | 25 |
| Jinja | 2 | 0.317 | 1.57 | 0.559 | ' Uganda' | ' source' | 25 |
| Gyumri | 3 | 0.300 | 3.01 | 0.637 | ' Armenia' | ' statue' | 23 |
| Sokoto | 3 | 0.334 | 2.72 | 0.738 | ' Nigeria' | ' Sultan' | 24 |
| Bregenz | 3 | 0.285 | 3.00 | 0.671 | ' Austria' | ' ' | 23 |
| Kanazawa | 2 | 0.324 | 2.17 | 0.709 | ' Japan' | ' Ken' | 26 |
| Timbuktu | 4 | 0.335 | 1.86 | 0.634 | ' Mali' | ' Dj' | 24 |
| Iquitos | 2 | 0.305 | 3.14 | 0.767 | ' Peru' | ' Cathedral' | 25 |

### F1 개체별
| 개체 | 토큰 | 응집도 | 엔트로피 | top1 | greedy(country) | greedy(landmark) | 최응집층 |
|---|---|---|---|---|---|---|---|
| Malorvia | 3 | 0.295 | 5.83 | 0.714 | ' Mal' | ' Mal' | 22 |
| Trennsk | 2 | 0.319 | 5.87 | 0.584 | ' Germany' | ' ' | 25 |
| Vantoria | 3 | 0.282 | 5.88 | 0.788 | ' V' | ' ' | 23 |
| Qellmin | 3 | 0.298 | 5.74 | 0.842 | ' Q' | ' Q' | 21 |
| Zorbeth | 3 | 0.291 | 5.72 | 0.821 | ' Z' | ' Z' | 22 |
| Dravnia | 3 | 0.309 | 6.04 | 0.735 | ' Drav' | ' Temple' | 22 |
| Klenor | 3 | 0.313 | 6.09 | 0.779 | ' K' | ' K' | 23 |
| Vurstan | 3 | 0.315 | 5.95 | 0.749 | ' V' | ' ' | 21 |

### F0 개체별
| 개체 | 토큰 | 응집도 | 엔트로피 | top1 | greedy(country) | greedy(landmark) | 최응집층 |
|---|---|---|---|---|---|---|---|
| Xqzptr | 3 | 0.308 | 5.89 | 0.706 | ' X' | ' X' | 21 |
| Zkvwmn | 4 | 0.309 | 5.80 | 0.557 | ' Z' | ' Z' | 21 |
| Qfjklb | 4 | 0.305 | 5.95 | 0.760 | ' Q' | ' ' | 21 |
| Wrtpxn | 4 | 0.309 | 6.06 | 0.650 | ' W' | ' W' | 23 |

## 재샘플 (2b 다양성 · 2c 거부) — F4·F1 양끝
프레임 `{X} is a city located in the country of`, 샘플링 T=0.8/top_p=0.95, N=10.

### [F4] Paris — distinct 1/10 (비율 0.10), 거부율 0.00
<details><summary>원문 N개</summary>

1. `France, situated in`
2. `France. The population`
3. `France.
Kara`
4. `France. The city`
5. `France. It is`
6. `France. It is`
7. `France. It is`
8. `France. __`
9. `France. It is`
10. `France. It is`

</details>

### [F4] Tokyo — distinct 1/10 (비율 0.10), 거부율 0.00
<details><summary>원문 N개</summary>

1. `Japan, with a`
2. `Japan. The population`
3. `Japan.
Kara`
4. `Japan. The city`
5. `Japan. Tokyo is`
6. `Japan. Tokyo is`
7. `Japan. Tokyo is`
8. `Japan. ___,`
9. `Japan. It is`
10. `Japan. It is`

</details>

### [F4] London — distinct 3/10 (비율 0.30), 거부율 0.00
<details><summary>원문 N개</summary>

1. `the United Kingdom.`
2. `Great Britain in Europe`
3. `England.
Kara`
4. `the UK. It`
5. `England. London is`
6. `the United Kingdom.`
7. `England, which is`
8. `England, __1`
9. `England. It is`
10. `Great Britain. London`

</details>

### [F4] Rome — distinct 1/10 (비율 0.10), 거부율 0.00
<details><summary>원문 N개</summary>

1. `Italy, in the`
2. `Italy. The population`
3. `Italy.
Kara`
4. `Italy.It`
5. `Italy. It is`
6. `Italy. Rome was`
7. `Italy. It is`
8. `Italy. __`
9. `Italy. It is`
10. `Italy. It is`

</details>

### [F4] Berlin — distinct 1/10 (비율 0.10), 거부율 0.00
<details><summary>원문 N개</summary>

1. `Germany, situated in`
2. `Germany. The population`
3. `Germany.
Kara`
4. `Germany. The city`
5. `Germany. It is`
6. `Germany. It is`
7. `Germany. According to`
8. `Germany. __`
9. `Germany. The city`
10. `Germany. Berlin is`

</details>

### [F1] Malorvia — distinct 8/10 (비율 0.80), 거부율 0.00
<details><summary>원문 N개</summary>

1. `Faldor.`
2. `Malorvia.`
3. `Maloria. The`
4. `Nordonia. It`
5. `Ionia. Mal`
6. `Aetheria.`
7. `Malora, in`
8. `Maloria ___,`
9. `Aleria on the`
10. `Maloria, in`

</details>

### [F1] Trennsk — distinct 9/10 (비율 0.90), 거부율 0.00
<details><summary>원문 N개</summary>

1. `Belgium, situated in`
2. `Germany. Trennsk`
3. `Germany.`
4. `Austria. Trensk`
5. `Peru. It is`
6. `Saxony-Anhalt`
7. `Trenska Republic,`
8. `Namibia. ___(`
9. `Denmark. It is`
10. `<strong>Peru</strong>`

</details>

### [F1] Vantoria — distinct 8/10 (비율 0.80), 거부율 0.00
<details><summary>원문 N개</summary>

1. `Faldia.`
2. `Vanoria in the`
3. `Vantoria.`
4. `Vantoria.`
5. `Xaria. V`
6. `Aetheria.`
7. `Trakia,`
8. `Vant ___,`
9. `Vantoria.`
10. `<strong>Elven`

</details>

### [F1] Qellmin — distinct 8/10 (비율 0.80), 거부율 0.00
<details><summary>원문 N개</summary>

1. `Faldia.`
2. `Qelqueria`
3. `Quellmin.`
4. `Qellmin.`
5. `Qellmin,`
6. `Aetheria.`
7. `Trakia,`
8. `Fes ___,`
9. `Qellmin.`
10. `<strong>Elven`

</details>

### [F1] Zorbeth — distinct 10/10 (비율 1.00), 거부율 0.00
<details><summary>원문 N개</summary>

1. `Faldia.`
2. `Luthara.`
3. `Muldrak.`
4. `Sordal.`
5. `Xel-Ha`
6. `Aetheria.`
7. `Trakia,`
8. `Ered ___,`
9. `Alora on the`
10. `<strong>Aven`

</details>

