# Phase 1 — Lens sanity check (gemma4b, google/gemma-3-4b-pt)

- source layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
- sanity rule: model greedy next token is lens top-1 at any of last 3 source layers [30, 31, 32]


**Result: 4/5 passed**

## 'The capital of France is'
- model next token: ' a'
- late-layer top-1: L30:' Paris' | L31:' Paris' | L32:' known'  -> FAIL
- rank of model token in lens@32: 5
- layer top-1 progression: L0:'<start_of_image>' | L4:'  ' | L8:'  ' | L12:'  ' | L16:' Madrid' | L20:' bustling' | L24:' located' | L28:' Paris' | L32:' known'

## 'Water is made of hydrogen and'
- model next token: ' oxygen'
- late-layer top-1: L30:' oxygen' | L31:' oxygen' | L32:' oxygen'  -> PASS
- rank of model token in lens@32: 1
- layer top-1 progression: L0:'\\&' | L4:'".[' | L8:'.\\\\' | L12:' methane' | L16:' hydrogens' | L20:' hydrogens' | L24:' oxygen' | L28:' oxygen' | L32:' oxygen'

## 'The opposite of hot is'
- model next token: ' cold'
- late-layer top-1: L30:' cold' | L31:' cold' | L32:' cold'  -> PASS
- rank of model token in lens@32: 1
- layer top-1 progression: L0:'<start_of_image>' | L4:'<start_of_image>' | L8:'<start_of_image>' | L12:'<start_of_image>' | L16:' crappy' | L20:' crappy' | L24:' cold' | L28:' cold' | L32:' cold'

## 'Two plus two equals'
- model next token: ' four'
- late-layer top-1: L30:' five' | L31:' four' | L32:' four'  -> PASS
- rank of model token in lens@32: 1
- layer top-1 progression: L0:'</b>' | L4:'.=' | L8:'</b>' | L12:'</b>' | L16:' equals' | L20:' equals' | L24:'...?' | L28:' four' | L32:' four'

## 'The sun rises in the'
- model next token: ' east'
- late-layer top-1: L30:' east' | L31:' east' | L32:' east'  -> PASS
- rank of model token in lens@32: 1
- layer top-1 progression: L0:'<start_of_image>' | L4:'<start_of_image>' | L8:'<start_of_image>' | L12:'</strong>' | L16:' sunrise' | L20:' sunrise' | L24:' northeast' | L28:' east' | L32:' east'

