# Phase 1 — Lens sanity check (qwen1.7b, Qwen/Qwen3-1.7B)

- source layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
- sanity rule: model greedy next token is lens top-1 at any of last 3 source layers [24, 25, 26]


**Result: 5/5 passed**

## 'The capital of France is'
- model next token: ' Paris'
- late-layer top-1: L24:' Paris' | L25:' Paris' | L26:' Paris'  -> PASS
- rank of model token in lens@26: 1
- layer top-1 progression: L0:' ;\r\n\r\n' | L3:' “' | L6:' .\r\n' | L9:'？\n' | L12:'？\n' | L15:'？\n' | L18:'____' | L21:'____' | L24:' Paris'

## 'Water is made of hydrogen and'
- model next token: ' oxygen'
- late-layer top-1: L24:' oxygen' | L25:' oxygen' | L26:' oxygen'  -> PASS
- rank of model token in lens@26: 1
- layer top-1 progression: L0:' ...\n' | L3:'....\n' | L6:' \n' | L9:'？\n' | L12:'？\n' | L15:'?\n' | L18:'？\n' | L21:' hydrogen' | L24:' oxygen'

## 'The opposite of hot is'
- model next token: ' cold'
- late-layer top-1: L24:' cold' | L25:' cold' | L26:' cold'  -> PASS
- rank of model token in lens@26: 1
- layer top-1 progression: L0:' ;\n\n\n' | L3:' “' | L6:' \xad' | L9:'\xa0' | L12:'—"' | L15:'？\n' | L18:'____' | L21:' cold' | L24:' cold'

## 'Two plus two equals'
- model next token: ' four'
- late-layer top-1: L24:' four' | L25:' four' | L26:' four'  -> PASS
- rank of model token in lens@26: 1
- layer top-1 progression: L0:'@$' | L3:' \n' | L6:'\xa0' | L9:'\xa0' | L12:'?"\n' | L15:'?"\n' | L18:'?"\n' | L21:'?\n' | L24:' four'

## 'The sun rises in the'
- model next token: ' east'
- late-layer top-1: L24:' east' | L25:' east' | L26:' east'  -> PASS
- rank of model token in lens@26: 1
- layer top-1 progression: L0:' @' | L3:'...\n' | L6:'.’' | L9:' sunrise' | L12:'天空' | L15:' sunrise' | L18:'__\r\n' | L21:'____' | L24:' east'

