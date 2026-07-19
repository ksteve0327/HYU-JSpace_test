"""Shared helpers for the J-lens local reproduction.

Loads HuggingFace models + pretrained Jacobian lenses (neuronpedia/jacobian-lens)
and runs them locally on Apple Silicon (MPS/fp16) or CPU. No lens fitting.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import transformers

import jlens

# neuronpedia hosts lenses fitted the paper's way (~1000 Salesforce-wikitext seqs).
LENS_REPO = "neuronpedia/jacobian-lens"

# short name -> (hf model id, lens filename inside LENS_REPO)
MODELS = {
    "qwen0.8b": (
        "Qwen/Qwen3.5-0.8B",
        "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt",
    ),
    "qwen1.7b": (
        "Qwen/Qwen3-1.7B",
        "qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt",
    ),
    "pythia70m": (
        "EleutherAI/pythia-70m-deduped",
        "pythia-70m-deduped/jlens/Salesforce-wikitext/pythia-70m-deduped_jacobian_lens.pt",
    ),
    # gated (Google Gemma license) — needs `hf auth login` + accepted license.
    "gemma4b": (
        "google/gemma-3-4b-pt",
        "gemma-3-4b/jlens/Salesforce-wikitext/gemma-3-4b-pt_jacobian_lens.pt",
    ),
}

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def get_device_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def load_hf(model_id: str, device, dtype):
    # Gemma activations overflow in fp16 (-> NaN -> garbage/<pad>); force bf16.
    if "gemma" in model_id.lower() and dtype == torch.float16:
        dtype = torch.bfloat16
    tok = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    hf.eval()
    return tok, hf


def load_lens(filename: str) -> "jlens.JacobianLens":
    return jlens.JacobianLens.from_pretrained(LENS_REPO, filename=filename)


def resolve_single_token(tok, surface: str):
    """Return the token id iff `surface` encodes to exactly one token, else None."""
    ids = tok.encode(surface, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    return None


def token_variants(tok, words):
    """Map each concept word to the set of single-token ids across surface variants
    (leading space, capitalization). Used so numeric/word answers are all checked."""
    out = {}
    for w in words:
        ids = set()
        for surf in {w, " " + w, w.capitalize(), " " + w.capitalize()}:
            enc = tok.encode(surf, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        out[w] = sorted(ids)
    return out


if __name__ == "__main__":
    print("device/dtype:", get_device_dtype())
    print("models:", list(MODELS))
