# Phase 1 — Lens sanity check (qwen0.8b, Qwen/Qwen3.5-0.8B)

- source layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
- sanity rule: model greedy next token is lens top-1 at any of last 3 source layers [20, 21, 22]


**Result: 5/5 passed**

## 'The capital of France is'
- model next token: ' Paris'
- late-layer top-1: L20:' Paris' | L21:' located' | L22:' located'  -> PASS
- rank of model token in lens@22: 2
- layer top-1 progression: L0:' being' | L2:'\n\n' | L4:' is' | L6:' continent' | L8:' cities' | L10:' cities' | L12:' ______' | L14:' ____' | L16:'?**' | L18:' located' | L20:' Paris' | L22:' located'

## 'Water is made of hydrogen and'
- model next token: ' oxygen'
- late-layer top-1: L20:' oxygen' | L21:' oxygen' | L22:' oxygen'  -> PASS
- rank of model token in lens@22: 1
- layer top-1 progression: L0:'**' | L2:'.’' | L4:'”**' | L6:'?’' | L8:' atoms' | L10:'?**' | L12:' atoms' | L14:' chemicals' | L16:' oxygen' | L18:' oxygen' | L20:' oxygen' | L22:' oxygen'

## 'The opposite of hot is'
- model next token: ' cold'
- late-layer top-1: L20:' cold' | L21:' cold' | L22:' cold'  -> PASS
- rank of model token in lens@22: 1
- layer top-1 progression: L0:' being' | L2:'\n\n' | L4:'pie' | L6:'—is' | L8:' thing' | L10:'?' | L12:' ____' | L14:' ____' | L16:' cold' | L18:' cold' | L20:' cold' | L22:' cold'

## 'Two plus two equals'
- model next token: ' four'
- late-layer top-1: L20:' four' | L21:' four' | L22:' ____'  -> PASS
- rank of model token in lens@22: 2
- layer top-1 progression: L0:' equals' | L2:' equals' | L4:' equals' | L6:' equals' | L8:' equals' | L10:' equals' | L12:' _____' | L14:' ____' | L16:' ____' | L18:' ____' | L20:' four' | L22:' ____'

## 'The sun rises in the'
- model next token: ' east'
- late-layer top-1: L20:' east' | L21:' east' | L22:' east'  -> PASS
- rank of model token in lens@22: 1
- layer top-1 progression: L0:" .'" | L2:" .'" | L4:' park' | L6:' skies' | L8:' twilight' | L10:' sunrise' | L12:' morning' | L14:' morning' | L16:' morning' | L18:' east' | L20:' east' | L22:' east'

