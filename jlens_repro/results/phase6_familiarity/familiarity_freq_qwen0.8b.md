# 친숙도 vs J-space 표현 — qwen0.8b (`Qwen/Qwen3.5-0.8B`)

- 워크스페이스 대역(40–90%): layers 9–19 (11층) / 전체 23층
- 프레임: `{X} is a city located in the country of` · `The most famous landmark in {X} is the`
- 지표: 1a 이웃 응집도(렌즈 top-20 임베딩 코사인, 높음=뭉친 이웃) · 1c 렌즈 확신도 · 2a 출력 엔트로피(모델 다음토큰, 높음=넓음) · 3 최응집 레이어.
- **지표 1b(희소분해)는 생략** — jlens에 gradient-pursuit API 없음. 의도는 1a+1c로 커버.
- 세 가설(H-수렴/방황/거부)은 **열어둠**. 2b·2c는 F4·F1 양끝만.

## 단계별 요약 (평균±표준편차)
| 단계 | 개체수 | 이웃 응집도(1a) | 출력 엔트로피(2a) | 렌즈 top1(1c) | 마진 | 토큰수 |
|---|---|---|---|---|---|---|
| F4 | 8 | 0.290±0.009 | 3.37±0.63 | 0.220±0.029 | 0.123 | 1.5 |
| F3 | 8 | 0.291±0.008 | 4.14±0.44 | 0.196±0.038 | 0.099 | 3.0 |
| F2 | 8 | 0.279±0.016 | 5.12±0.81 | 0.178±0.051 | 0.092 | 2.8 |
| F1 | 8 | 0.242±0.016 | 6.49±0.27 | 0.115±0.010 | 0.047 | 3.0 |
| F0 | 4 | 0.244±0.009 | 6.63±0.21 | 0.121±0.006 | 0.059 | 4.0 |

### F4 개체별
| 개체 | 토큰 | 응집도 | 엔트로피 | top1 | greedy(country) | greedy(landmark) | 최응집층 |
|---|---|---|---|---|---|---|---|
| Paris | 1 | 0.296 | 2.24 | 0.237 | ' France' | ' Eiff' | 19 |
| Tokyo | 2 | 0.274 | 3.46 | 0.253 | ' Japan' | ' Tokyo' | 19 |
| London | 1 | 0.279 | 3.95 | 0.184 | ' the' | ' Tower' | 16 |
| Rome | 2 | 0.289 | 3.18 | 0.222 | ' Italy' | ' Col' | 16 |
| Berlin | 1 | 0.298 | 2.84 | 0.216 | ' Germany' | ' Brandenburg' | 18 |
| Madrid | 1 | 0.302 | 3.95 | 0.170 | ' Spain' | '\n' | 18 |
| Cairo | 2 | 0.285 | 4.27 | 0.222 | ' Egypt' | ' Great' | 18 |
| Moscow | 2 | 0.295 | 3.04 | 0.256 | ' Russia' | ' Kremlin' | 18 |

### F3 개체별
| 개체 | 토큰 | 응집도 | 엔트로피 | top1 | greedy(country) | greedy(landmark) | 최응집층 |
|---|---|---|---|---|---|---|---|
| Tashkent | 3 | 0.288 | 3.85 | 0.219 | ' Uzbek' | ' T' | 18 |
| Almaty | 3 | 0.289 | 4.11 | 0.263 | ' Kazakhstan' | ' Al' | 18 |
| Nairobi | 2 | 0.279 | 3.99 | 0.167 | ' Kenya' | ' **' | 17 |
| Tbilisi | 3 | 0.297 | 3.40 | 0.189 | ' Georgia' | ' National' | 19 |
| Ljubljana | 4 | 0.308 | 3.97 | 0.245 | ' Slovenia' | ' **' | 18 |
| Windhoek | 2 | 0.286 | 4.58 | 0.158 | ' Nam' | ' **' | 17 |
| Chisinau | 4 | 0.293 | 4.27 | 0.168 | ' Moldova' | ' Cathedral' | 16 |
| Kigali | 3 | 0.286 | 4.94 | 0.160 | ' Rwanda' | ' Great' | 17 |

### F2 개체별
| 개체 | 토큰 | 응집도 | 엔트로피 | top1 | greedy(country) | greedy(landmark) | 최응집층 |
|---|---|---|---|---|---|---|---|
| Oruro | 2 | 0.281 | 5.09 | 0.155 | ' Paraguay' | '\n' | 17 |
| Jinja | 2 | 0.250 | 6.67 | 0.106 | ' the' | ' **' | 18 |
| Gyumri | 3 | 0.261 | 6.24 | 0.116 | ' G' | ' **' | 15 |
| Sokoto | 3 | 0.287 | 4.64 | 0.253 | ' Nigeria' | ' **' | 18 |
| Bregenz | 3 | 0.305 | 4.23 | 0.195 | ' Austria' | ' **' | 18 |
| Kanazawa | 3 | 0.276 | 4.55 | 0.197 | ' Japan' | ' **' | 17 |
| Timbuktu | 3 | 0.289 | 4.82 | 0.247 | ' Burk' | ' Tomb' | 17 |
| Iquitos | 3 | 0.281 | 4.68 | 0.156 | ' Peru' | '\n' | 19 |

### F1 개체별
| 개체 | 토큰 | 응집도 | 엔트로피 | top1 | greedy(country) | greedy(landmark) | 최응집층 |
|---|---|---|---|---|---|---|---|
| Malorvia | 3 | 0.273 | 5.99 | 0.120 | ' Mal' | ' **' | 16 |
| Trennsk | 3 | 0.246 | 6.93 | 0.124 | ' the' | '\n' | 18 |
| Vantoria | 3 | 0.234 | 6.29 | 0.130 | ' V' | ' **' | 9 |
| Qellmin | 3 | 0.246 | 6.68 | 0.111 | ' the' | '\n' | 9 |
| Zorbeth | 3 | 0.221 | 6.31 | 0.106 | ' Z' | ' Great' | 9 |
| Dravnia | 3 | 0.245 | 6.62 | 0.097 | ' the' | ' **' | 17 |
| Klenor | 3 | 0.222 | 6.59 | 0.122 | ' K' | ' K' | 9 |
| Vurstan | 3 | 0.247 | 6.48 | 0.113 | ' V' | ' V' | 9 |

### F0 개체별
| 개체 | 토큰 | 응집도 | 엔트로피 | top1 | greedy(country) | greedy(landmark) | 최응집층 |
|---|---|---|---|---|---|---|---|
| Xqzptr | 4 | 0.234 | 6.34 | 0.111 | ' X' | ' X' | 18 |
| Zkvwmn | 4 | 0.256 | 6.54 | 0.123 | ' Z' | ' Z' | 16 |
| Qfjklb | 5 | 0.237 | 6.74 | 0.123 | ' the' | ' Q' | 18 |
| Wrtpxn | 3 | 0.248 | 6.90 | 0.128 | ' the' | '\n' | 18 |

## 재샘플 (2b 다양성 · 2c 거부) — F4·F1 양끝
프레임 `{X} is a city located in the country of`, 샘플링 T=0.8/top_p=0.95, N=3.

### [F4] Paris — distinct 1/3 (비율 0.33), 거부율 0.00
<details><summary>원문 N개</summary>

1. `France.
Paris`
2. `France. It is`
3. `France. The length`

</details>

### [F4] Tokyo — distinct 1/3 (비율 0.33), 거부율 0.00
<details><summary>원문 N개</summary>

1. `Japan.
==`
2. `Japan. It is`
3. `Japan. The main`

</details>

### [F1] Malorvia — distinct 2/3 (비율 0.67), 거부율 0.00
<details><summary>원문 N개</summary>

1. `Maldives.`
2. `the Malorvi`
3. `Maldives.`

</details>

### [F1] Trennsk — distinct 3/3 (비율 1.00), 거부율 0.00
<details><summary>원문 N개</summary>

1. `Serbia.
==`
2. `the North Sea,`
3. `Northland.`

</details>

