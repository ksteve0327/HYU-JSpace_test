"""Phase 0 smoke test: load model (MPS/fp16), generate once, load lens, and
introspect the instance API we rely on later. Writes results/env.md."""

from __future__ import annotations

import sys

import torch
import transformers

import jlens
from common import MODELS, RESULTS_DIR, get_device_dtype, load_hf, load_lens

MODEL_KEY = sys.argv[1] if len(sys.argv) > 1 else "qwen0.8b"
model_id, lens_file = MODELS[MODEL_KEY]

device, dtype = get_device_dtype()
lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


log(f"# Phase 0 — Environment & Smoke Test ({MODEL_KEY})\n")
log(f"- torch: {torch.__version__}")
log(f"- transformers: {transformers.__version__}")
log(f"- jlens: {getattr(jlens, '__version__', 'editable')}")
log(f"- device: {device}, dtype: {dtype}")
log(f"- model: `{model_id}`")
log(f"- lens: `{lens_file}`\n")

log("## Load model")
tok, hf = load_hf(model_id, device, dtype)
n_params = sum(p.numel() for p in hf.parameters())
log(f"- params: {n_params/1e6:.1f}M")

log("\n## Generation check")
prompt = "The capital of France is"
enc = tok(prompt, return_tensors="pt").to(device)
with torch.inference_mode():
    out = hf.generate(**enc, max_new_tokens=8, do_sample=False)
gen = tok.decode(out[0], skip_special_tokens=True)
log(f"- prompt: {prompt!r}")
log(f"- greedy continuation: {gen!r}")

log("\n## Wrap with jlens.from_hf + load lens")
lens_model = jlens.from_hf(hf, tok, compile=False)
lens = load_lens(lens_file)

# Introspect the instance attributes the swap code depends on.
lm_d = getattr(lens_model, "d_model", None)
lens_d = getattr(lens, "d_model", None)
src_layers = getattr(lens, "source_layers", None)
n_prompts = getattr(lens, "n_prompts", None)
has_layers = hasattr(lens_model, "layers")
has_lm_head = hasattr(lens_model, "_lm_head")
has_jac = hasattr(lens, "jacobians")

log(f"- lens_model.d_model: {lm_d}")
log(f"- lens.d_model: {lens_d}")
log(f"- lens.source_layers: {src_layers}")
log(f"- lens.n_prompts: {n_prompts}")
log(f"- lens_model has .layers: {has_layers}; has ._lm_head: {has_lm_head}")
log(f"- lens has .jacobians: {has_jac}")
if lm_d is not None and lens_d is not None:
    log(f"- d_model match: {lm_d == lens_d}")

# Discover how to reach unembedding + decoder layers for the swap hook.
log("\n## Attribute discovery (for swap hook wiring)")
log(f"- lens_model public attrs: {[a for a in dir(lens_model) if not a.startswith('__')]}")
if has_jac:
    try:
        keys = sorted(lens.jacobians.keys()) if hasattr(lens.jacobians, "keys") else "n/a"
        log(f"- lens.jacobians keys: {keys}")
        k0 = (keys[0] if isinstance(keys, list) else None)
        if k0 is not None:
            log(f"- jacobian[{k0}] shape: {tuple(lens.jacobians[k0].shape)}")
    except Exception as e:  # noqa: BLE001
        log(f"- jacobians introspection error: {e}")

(RESULTS_DIR / "env.md").write_text("\n".join(lines) + "\n")
log(f"\nWrote {RESULTS_DIR / 'env.md'}")
