# Phase 0 — Environment & Smoke Test (qwen0.8b)

- torch: 2.13.0
- transformers: 5.14.1
- jlens: editable
- device: mps, dtype: torch.float16
- model: `Qwen/Qwen3.5-0.8B`
- lens: `qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt`

## Load model
- params: 752.4M

## Generation check
- prompt: 'The capital of France is'
- greedy continuation: 'The capital of France is Paris.\nThe capital of France is'

## Wrap with jlens.from_hf + load lens
- lens_model.d_model: 1024
- lens.d_model: 1024
- lens.source_layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
- lens.n_prompts: 233
- lens_model has .layers: True; has ._lm_head: True
- lens has .jacobians: True
- d_model match: True

## Attribute discovery (for swap hook wiring)
- lens_model public attrs: ['_embed_tokens', '_final_norm', '_hf_model', '_lm_head', '_logit_softcap', '_text_module', 'd_model', 'encode', 'forward', 'input_device', 'layers', 'layout', 'n_layers', 'tokenizer', 'unembed']
- lens.jacobians keys: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
- jacobian[0] shape: (1024, 1024)
