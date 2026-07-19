"""Coordinate-swap intervention on the residual stream, using Jacobian-lens
directions. Ported from the user's jspace_colab_walkthrough.ipynb, generalized
with an alpha strength and a random-direction control.

Concept direction (paper §2.5):  v_w = normalize( W_U[w] @ J_layer )
Swap:  read coords c = h @ pinv(V)^T with V=[v_src, v_tgt];
       exchange the two coords; h <- h + alpha * (flip(c) - c) @ V^T.
alpha=1 is the exact coordinate swap; alpha=2 is the paper's "double strength".
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


def lm_head_weight(lens_model) -> torch.Tensor:
    return lens_model._lm_head.weight


def concept_direction(lens, lens_model, layer: int, token_id: int) -> torch.Tensor:
    """Unit-normalized lens write direction for `token_id` at `layer` (d_model)."""
    w = lm_head_weight(lens_model)[token_id].detach().float().cpu()
    v = w @ lens.jacobians[layer].float()
    return v / v.norm()


def build_swap_operators(lens, lens_model, band_layers, src_id, tgt_id, device,
                         random_tgt: bool = False, seed_layer_offset: int = 0):
    """Return {layer: (V, V_pinv)} for the coordinate swap.

    If random_tgt, replace the target direction with a random unit vector of the
    same norm (control: intervention of equal magnitude in a meaningless direction).
    """
    ops = {}
    for layer in band_layers:
        v_src = concept_direction(lens, lens_model, layer, src_id)
        if random_tgt:
            # deterministic pseudo-random direction (no Math.random needed), varied per layer
            g = torch.Generator().manual_seed(1000 + layer + seed_layer_offset)
            r = torch.randn(v_src.shape, generator=g)
            v_tgt = r / r.norm()
        else:
            v_tgt = concept_direction(lens, lens_model, layer, tgt_id)
        V = torch.stack([v_src, v_tgt], dim=1).float()  # [d_model, 2]
        V_pinv = torch.linalg.pinv(V)  # [2, d_model]
        ops[layer] = (V.to(device), V_pinv.to(device))
    return ops


def _make_hook(V, V_pinv, alpha):
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        odtype = hidden.dtype
        h = hidden.float()
        c = h @ V_pinv.T            # [..., 2]
        c_swapped = c.flip(dims=(-1,))
        patched = h + alpha * ((c_swapped - c) @ V.T)
        patched = patched.to(odtype)
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched
    return hook


@contextmanager
def swap_hooks(lens_model, operators, alpha: float = 1.0):
    handles = []
    try:
        for layer, (V, V_pinv) in operators.items():
            handles.append(
                lens_model.layers[layer].register_forward_hook(_make_hook(V, V_pinv, alpha))
            )
        yield
    finally:
        for h in handles:
            h.remove()


def concept_dirs(lens, lens_model, band_layers, token_id, device):
    """{layer: unit lens direction v_w on device} for steering/ablation."""
    return {L: concept_direction(lens, lens_model, L, token_id).to(device) for L in band_layers}


def _steer_hook(v, alpha):
    # h += alpha * (mean residual norm) * v_w  — inject(+)/suppress(-) a concept.
    def hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        od = h.dtype
        x = h.float()
        scale = x.norm(dim=-1, keepdim=True).mean()  # ~ typical activation magnitude
        x = x + alpha * scale * v
        x = x.to(od)
        return (x, *output[1:]) if isinstance(output, tuple) else x
    return hook


def _ablate_hook(v, alpha):
    # h -= alpha * (h·v_w) v_w  — project the concept direction out (alpha=1 = full).
    def hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        od = h.dtype
        x = h.float()
        coef = (x @ v).unsqueeze(-1)
        x = x - alpha * coef * v
        x = x.to(od)
        return (x, *output[1:]) if isinstance(output, tuple) else x
    return hook


@contextmanager
def dir_hooks(lens_model, layer_dirs, alpha, mode):
    """Register steer/ablate hooks over {layer: v} at given alpha. mode: 'steer'|'ablate'."""
    make = _steer_hook if mode == "steer" else _ablate_hook
    handles = []
    try:
        for L, v in layer_dirs.items():
            handles.append(lens_model.layers[L].register_forward_hook(make(v, alpha)))
        yield
    finally:
        for h in handles:
            h.remove()


def next_token_logits(hf, tok, prompt, device):
    enc = tok(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        return hf(**enc).logits[0, -1].float().cpu()


def band_from_fraction(source_layers, lo=0.40, hi=0.90):
    n = len(source_layers)
    a = int(n * lo)
    b = max(a + 1, int(n * hi))
    return list(source_layers[a:b])
