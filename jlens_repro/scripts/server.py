"""Interactive J-lens + swap tester (local Flask backend).

Run:  ./.venv/bin/python scripts/server.py         # default model qwen0.8b
Then open http://127.0.0.1:8000 in a browser.

Loads a model + pretrained lens once and serves:
  GET  /                 -> frontend.html
  GET  /api/config       -> models, current model, layers, band defaults
  POST /api/model        -> {key}                 switch model (reload)
  POST /api/read         -> {prompt, position, track?}   per-layer lens top-k
  POST /api/swap         -> {prompt, src, tgt, alpha, lo, hi, random_dir}
  POST /api/sweep        -> {prompt, src, tgt, alpha, window}  layer sweep
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import datetime
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from contextlib import ExitStack, contextmanager
from pathlib import Path

import torch
from flask import Flask, Response, jsonify, request, send_file

import jlens
import jlens.vis as jvis
from common import MODELS, get_device_dtype, load_hf, load_lens, resolve_single_token, token_variants
from swaplib import (band_from_fraction, build_swap_operators, concept_dirs, dir_hooks,
                     next_token_logits, swap_hooks)

HERE = Path(__file__).resolve().parent
app = Flask(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Local claude-proxy (OpenAI-compatible, backed by the Claude CLI subscription).
PROXY_HEALTH = "http://localhost:11436/health"
PROXY_CHAT = "http://localhost:11436/v1/chat/completions"
PROXY_MODELS = ["claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5"]

# Local codex-proxy (OpenAI-compatible, backed by the Codex/ChatGPT subscription).
CODEX_PROXY_HEALTH = "http://localhost:11435/health"
CODEX_PROXY_CHAT = "http://localhost:11435/v1/chat/completions"
CODEX_PROXY_MODELS = ["gpt-5.6-sol", "gpt-5.6-terra"]
CODEX_PROXY_REASONING = "xhigh"
CODEX_PROXY_SCRIPT = Path.home() / "bin" / "codex-proxy.mjs"


def proxy_up():
    try:
        with urllib.request.urlopen(PROXY_HEALTH, timeout=2) as r:
            return json.loads(r.read()).get("backend") == "claude-cli"
    except Exception:  # noqa: BLE001
        return False


def codex_proxy_up():
    try:
        with urllib.request.urlopen(CODEX_PROXY_HEALTH, timeout=2) as r:
            health = json.loads(r.read())
        return health.get("ok") is True and health.get("status") == "running"
    except Exception:  # noqa: BLE001
        return False


def ensure_codex_proxy():
    """Start the local Codex proxy in the background when it is not running."""
    if os.environ.get("CODEX_PROXY_AUTOSTART", "1").lower() in {"0", "false", "no"}:
        print("Codex proxy autostart disabled (CODEX_PROXY_AUTOSTART=0)")
        return False
    if codex_proxy_up():
        print("Codex proxy already running: http://localhost:11435")
        return True

    script = Path(os.environ.get("CODEX_PROXY_SCRIPT", CODEX_PROXY_SCRIPT)).expanduser()
    if not script.is_file():
        print(f"Codex proxy not started: script missing at {script}")
        return False

    log_path = RUNS_DIR / "codex-proxy.log"
    env = os.environ.copy()
    env.setdefault("CODEX_MODEL", "gpt-5.6-sol")
    env.setdefault("CODEX_REASONING", CODEX_PROXY_REASONING)
    env.setdefault("CODEX_WORKDIR", str(HERE.parent))
    try:
        with open(log_path, "a", encoding="utf-8") as log:
            subprocess.Popen(
                ["node", str(script)],
                cwd=HERE.parent,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except (OSError, subprocess.SubprocessError) as e:
        print(f"Codex proxy failed to start: {e}")
        return False

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if codex_proxy_up():
            print(f"Codex proxy started: http://localhost:11435 (log: {log_path})")
            return True
        time.sleep(0.2)
    print(f"Codex proxy did not become ready; see {log_path}")
    return False


def available_models():
    """Models from available OpenRouter, Claude proxy, and Codex proxy backends.

    Local models carry backend prefixes so the LLM endpoints can route them.
    """
    models = []
    if os.environ.get("OPENROUTER_API_KEY"):
        models += LLM_MODELS
    if proxy_up():
        models += ["proxy:" + m for m in PROXY_MODELS]
    if codex_proxy_up():
        models += ["codex:" + m for m in CODEX_PROXY_MODELS]
    return models


LLM_MODELS = []  # ordered list of models to run for interpretation


def load_env():
    """Load KEY=VALUE lines from jlens_repro/.env.

    Regular keys: last occurrence wins; real shell env vars take precedence.
    Models: collect ALL `OPENROUTER_MODEL` lines and any `OPENROUTER_MODELS`
    comma-list into LLM_MODELS (so you can compare several models at once).
    """
    global LLM_MODELS
    p = HERE.parent / ".env"
    if not p.exists():
        return
    file_vars, model_list = {}, []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if not v:
            continue
        if k == "OPENROUTER_MODEL":
            model_list.append(v)
        elif k == "OPENROUTER_MODELS":
            model_list.extend(m.strip() for m in v.split(",") if m.strip())
        else:
            file_vars[k] = v  # last wins
    for k, v in file_vars.items():
        os.environ.setdefault(k, v)
    seen = set()
    LLM_MODELS = [m for m in model_list if not (m in seen or seen.add(m))]
    if not LLM_MODELS:
        LLM_MODELS = ["anthropic/claude-3.5-sonnet"]


load_env()

# Serialize model calls (MPS is not safe for concurrent forwards); threaded=True
# keeps page/config requests responsive while an inference runs.
INFER_LOCK = threading.Lock()

STATE = {"key": None, "tok": None, "hf": None, "lens_model": None, "lens": None,
         "layers": None, "device": None, "dtype": None}

# 로컬 기록: 실행할 때마다 한 줄씩 append (results/runs.jsonl)
RUNS_DIR = HERE.parent / "results"
RUNS_DIR.mkdir(exist_ok=True)
RUNS_LOG = RUNS_DIR / "runs.jsonl"


def log_run(kind, rec):
    """Append one compact JSON line per run to results/runs.jsonl (local record)."""
    try:
        row = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
               "kind": kind, "model": STATE.get("key"), **rec}
        with open(RUNS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001
        print("log_run fail:", e)


@app.after_request
def _free_mps(resp):
    """Release cached MPS memory after each request so allocations don't accumulate
    across runs (main cause of memory pressure on 16GB unified memory)."""
    try:
        dev = STATE.get("device")
        if dev is not None and dev.type == "mps":
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return resp


def warmup():
    """Run one forward so the slow MPS cold-start happens at load, not on the
    user's first click."""
    try:
        with torch.inference_mode():
            STATE["hf"](**STATE["tok"]("warmup", return_tensors="pt").to(STATE["device"]))
    except Exception as e:  # noqa: BLE001
        print("warmup skipped:", e)


def load(key: str):
    if key not in MODELS:
        raise ValueError(f"unknown model {key}")
    # free previous
    for k in ("hf", "lens_model", "lens"):
        STATE[k] = None
    import gc
    gc.collect()
    device, dtype = get_device_dtype()
    model_id, lens_file = MODELS[key]
    tok, hf = load_hf(model_id, device, dtype)
    lens_model = jlens.from_hf(hf, tok, compile=False)
    lens = load_lens(lens_file)
    STATE.update(key=key, tok=tok, hf=hf, lens_model=lens_model, lens=lens,
                 layers=list(lens.source_layers), device=device, dtype=dtype)
    warmup()
    return STATE


def decode(tid):
    return STATE["tok"].decode([int(tid)])


def single_id(word):
    tok = STATE["tok"]
    return resolve_single_token(tok, " " + word) or resolve_single_token(tok, word)


def word_ids(word):
    """All single-token ids for a word across surface variants (space/caps).
    Used so 'London' vs ' London' are both counted for P/rank/flip."""
    ids = set(token_variants(STATE["tok"], [word]).get(word, []))
    s = single_id(word)
    if s is not None:
        ids.add(s)
    return sorted(ids)


def topk_list(row, k=8):
    p = row.softmax(-1)
    vals, idx = p.topk(k)
    return [{"tok": decode(i), "p": round(float(v), 4)} for v, i in zip(vals, idx)]


def rank_of(row, tid):
    return int((row > row[tid]).sum()) + 1


def best_rank_variants(row, ids):
    return min((rank_of(row, i) for i in ids), default=None)


@app.get("/")
def index():
    return send_file(HERE / "frontend.html")


@app.get("/api/config")
def config():
    layers = STATE["layers"]
    lo, hi = band_from_fraction(layers, 0.40, 0.90)[0], band_from_fraction(layers, 0.40, 0.90)[-1]
    return jsonify({
        "models": list(MODELS),
        "current": STATE["key"],
        "model_id": MODELS[STATE["key"]][0],
        "source_layers": layers,
        "n_layers": len(layers),
        "band_default": [40, 90],
        "band_layers": [lo, hi],
        "has_llm": bool(_models := available_models()),
        "llm_models": _models,
        "runs_log": str(RUNS_LOG),
    })


EXP_DIR = HERE.parent / "jacobian-lens" / "data" / "experiments"


@app.get("/api/probeswap")
def probeswap():
    """The paper's probe-swap prompt set (90 two-hop items)."""
    p = EXP_DIR / "probe-swap.json"
    if not p.exists():
        return jsonify({"error": "not_found", "path": str(p)}), 404
    data = json.loads(p.read_text())
    items = data.get("items", data if isinstance(data, list) else [])
    out = [{"name": it.get("name"), "category": it.get("category"),
            "prompt": it.get("prompt"), "intermediate": it.get("intermediate"),
            "answer": it.get("answer"), "swap_to": it.get("swap_to"),
            "swap_answer": it.get("swap_answer")} for it in items]
    return jsonify({"items": out, "source": "anthropics/jacobian-lens · data/experiments/probe-swap.json"})


@app.post("/api/model")
def switch_model():
    key = request.json["key"]
    load(key)
    return config()


@app.post("/api/read")
def read():
    d = request.json
    prompt = d["prompt"]
    pos = int(d.get("position", -1))
    track = (d.get("track") or "").strip()
    lens, lens_model, layers = STATE["lens"], STATE["lens_model"], STATE["layers"]
    with INFER_LOCK, torch.inference_mode():
        lens_logits, model_logits, input_ids = lens.apply(
            lens_model, prompt, layers=layers, positions=[pos], use_jacobian=True
        )
    track_ids = None
    if track:
        track_ids = token_variants(STATE["tok"], [track]).get(track) or []
        # also try the raw leading-space id
        sid = single_id(track)
        if sid is not None:
            track_ids = sorted(set(track_ids) | {sid})
    rows = []
    for L in layers:
        row = lens_logits[L][0]
        entry = {"layer": L, "top": [t["tok"] for t in topk_list(row, 6)]}
        if track_ids:
            entry["track_rank"] = best_rank_variants(row, track_ids)
        rows.append(entry)
    toks = [decode(t) for t in input_ids[0].tolist()]
    model_next = decode(int(model_logits[0].argmax()))
    best_tr = min((r["track_rank"] for r in rows if r.get("track_rank") is not None), default=None)
    log_run("read", {"prompt": prompt, "position": pos, "track": track,
                     "model_next": model_next, "track_best_rank": best_tr})
    return jsonify({
        "tokens": toks,
        "position": pos,
        "model_top": [t["tok"] for t in topk_list(model_logits[0], 6)],
        "layers": rows,
        "track": track,
        "track_ok": bool(track_ids),
    })


def _steer_common(d):
    """Shared setup for steer/ablate. Returns (err_response, concept, cids, band, dirs, band_pct)."""
    concept = d["concept"].strip()
    lo = float(d.get("lo", 40)) / 100
    hi = float(d.get("hi", 90)) / 100
    cid = resolve_single_token(STATE["tok"], " " + concept) or resolve_single_token(STATE["tok"], concept)
    if cid is None:
        return (jsonify({"error": "multi_token", "concept": concept,
                         "variants": token_variants(STATE["tok"], [concept]).get(concept, [])}), 400), None, None, None, None, None
    band = band_from_fraction(STATE["layers"], lo, hi)
    dirs = concept_dirs(STATE["lens"], STATE["lens_model"], band, cid, STATE["device"])
    return None, concept, word_ids(concept), band, dirs, [int(lo * 100), int(hi * 100)]


@app.post("/api/steer")
def steer():
    d = request.json
    prompt = d["prompt"]
    alpha = float(d.get("alpha", 4))
    mode = d.get("mode", "steer")  # 'steer' (h += α·norm·v) or 'ablate' (h -= α·(h·v)v)
    err, concept, cids, band, dirs, band_pct = _steer_common(d)
    if err:
        return err

    def pr(row):
        p = row.softmax(-1)
        return {"p": round(float(p[cids].sum()), 4), "rank": min(rank_of(row, i) for i in cids)}

    with INFER_LOCK:
        base = next_token_logits(STATE["hf"], STATE["tok"], prompt, STATE["device"])
        enc = STATE["tok"](prompt, return_tensors="pt").to(STATE["device"])
        with torch.inference_mode(), dir_hooks(STATE["lens_model"], dirs, alpha, mode):
            inter = STATE["hf"](**enc).logits[0, -1].float().cpu()
    flipped = int(inter.argmax()) in cids
    log_run(mode, {"prompt": prompt, "concept": concept, "alpha": alpha, "band_pct": band_pct,
                   "P_concept": [pr(base)["p"], pr(inter)["p"]], "flipped": flipped})
    return jsonify({"mode": mode, "concept": concept, "alpha": alpha, "band": band, "band_pct": band_pct,
                    "baseline": {"top": topk_list(base), "concept": pr(base)},
                    "intervened": {"top": topk_list(inter), "concept": pr(inter)},
                    "flipped": flipped})


@app.post("/api/steer_sweep")
def steer_sweep():
    d = request.json
    prompt = d["prompt"]
    alphas = d.get("alphas") or [0, 1, 2, 4, 6, 8]
    err, concept, cids, band, dirs, band_pct = _steer_common(d)
    if err:
        return err
    out = []
    with INFER_LOCK:
        enc = STATE["tok"](prompt, return_tensors="pt").to(STATE["device"])
        for a in alphas:
            with torch.inference_mode(), dir_hooks(STATE["lens_model"], dirs, float(a), "steer"):
                row = STATE["hf"](**enc).logits[0, -1].float().cpu()
            p = row.softmax(-1)
            out.append({"alpha": a, "p": round(float(p[cids].sum()), 4),
                        "rank": min(rank_of(row, i) for i in cids),
                        "top1": decode(row.argmax()), "flipped": int(row.argmax()) in cids})
    log_run("steer_sweep", {"prompt": prompt, "concept": concept, "band_pct": band_pct,
                            "curve": [[o["alpha"], o["rank"]] for o in out]})
    return jsonify({"points": out, "concept": concept, "band_pct": band_pct})


# ---- Unified intervention engine (swap / steer / ablate / combo) ----
@contextmanager
def op_hooks(band, ops):
    """Stack hooks for a list of ops on the band layers (combo = multiple ops in one forward)."""
    stack = ExitStack()
    try:
        for op in ops:
            t = op["type"]
            a = float(op.get("alpha", 2 if t == "swap" else (1 if t == "ablate" else 4)))
            if t == "swap":
                operators = build_swap_operators(STATE["lens"], STATE["lens_model"], band,
                                                 single_id(op["src"]), single_id(op["tgt"]),
                                                 STATE["device"], random_tgt=bool(op.get("random")))
                stack.enter_context(swap_hooks(STATE["lens_model"], operators, alpha=a))
            else:  # steer | ablate
                dirs = concept_dirs(STATE["lens"], STATE["lens_model"], band,
                                    single_id(op["concept"]), STATE["device"])
                stack.enter_context(dir_hooks(STATE["lens_model"], dirs, a,
                                              "steer" if t == "steer" else "ablate"))
        yield
    finally:
        stack.close()


def ops_from_request(mode, d):
    """Build (ops list, metric_word) from the request for a given mode."""
    a = d.get("alpha")
    if mode == "swap":
        ops = [{"type": "swap", "src": d["src"].strip(), "tgt": d["tgt"].strip(),
                "alpha": float(a if a is not None else 2), "random": bool(d.get("random_dir"))}]
        return ops, (d.get("success") or d["tgt"]).strip()
    if mode == "steer":
        return [{"type": "steer", "concept": d["concept"].strip(),
                 "alpha": float(a if a is not None else 4)}], d["concept"].strip()
    if mode == "ablate":
        return [{"type": "ablate", "concept": d["concept"].strip(),
                 "alpha": float(a if a is not None else 1)}], d["concept"].strip()
    if mode == "combo":
        ops = d.get("ops", [])
        metric = (d.get("metric") or "").strip()
        if not metric and ops:
            o = ops[0]
            metric = (o.get("tgt") or o.get("concept") or "").strip()
        return ops, metric
    raise ValueError("unknown mode")


def _bad_tokens(ops, metric):
    bad = []
    for op in ops:
        for k in ("src", "tgt", "concept"):
            w = op.get(k)
            if w and single_id(w) is None:
                bad.append(w)
    if metric and single_id(metric) is None:
        bad.append(metric)
    return bad


@app.post("/api/intervene")
def intervene():
    d = request.json
    prompt = d["prompt"]
    mode = d.get("mode", "swap")
    lo = float(d.get("lo", 40)) / 100
    hi = float(d.get("hi", 90)) / 100
    band = band_from_fraction(STATE["layers"], lo, hi)
    ops, metric = ops_from_request(mode, d)
    bad = _bad_tokens(ops, metric)
    if bad:
        return jsonify({"error": "multi_token", "words": bad}), 400
    mids = word_ids(metric) if metric else None

    def pr(row):
        if not mids:
            return None
        p = row.softmax(-1)
        return {"p": round(float(p[mids].sum()), 4), "rank": min(rank_of(row, i) for i in mids)}

    with INFER_LOCK:
        base = next_token_logits(STATE["hf"], STATE["tok"], prompt, STATE["device"])
        enc = STATE["tok"](prompt, return_tensors="pt").to(STATE["device"])
        with torch.inference_mode(), op_hooks(band, ops):
            inter = STATE["hf"](**enc).logits[0, -1].float().cpu()
    flipped = mids is not None and int(inter.argmax()) in mids
    band_pct = [int(lo * 100), int(hi * 100)]
    log_run(mode, {"prompt": prompt, "ops": ops, "metric": metric, "band_pct": band_pct,
                   "P_metric": [pr(base) and pr(base)["p"], pr(inter) and pr(inter)["p"]],
                   "flipped": flipped})
    return jsonify({"mode": mode, "ops": ops, "metric": metric, "band": band, "band_pct": band_pct,
                    "baseline": {"top": topk_list(base), "metric": pr(base)},
                    "intervened": {"top": topk_list(inter), "metric": pr(inter)},
                    "flipped": flipped})


@app.post("/api/intervene_sweep")
def intervene_sweep():
    d = request.json
    prompt = d["prompt"]
    mode = d.get("mode", "swap")
    window = int(d.get("window", 6))
    layers = STATE["layers"]
    n = len(layers)
    n_points = int(d.get("points") or (8 if n >= 30 else 12))
    n_points = max(3, min(n_points, n))
    step = max(1, n // n_points)
    ops, metric = ops_from_request(mode, d)
    bad = _bad_tokens(ops, metric)
    if bad or not metric:
        return jsonify({"error": "multi_token", "words": bad}), 400
    mids = word_ids(metric)
    out = []
    with INFER_LOCK:
        enc = STATE["tok"](prompt, return_tensors="pt").to(STATE["device"])
        for S in range(0, n - 1, step):
            wl = layers[S:S + window]
            if not wl:
                continue
            with torch.inference_mode(), op_hooks(wl, ops):
                row = STATE["hf"](**enc).logits[0, -1].float().cpu()
            p = row.softmax(-1)
            out.append({"start": layers[S], "pct": int(100 * S / n), "window": [wl[0], wl[-1]],
                        "p_tgt": round(float(p[mids].sum()), 4), "top1": decode(row.argmax()),
                        "flipped": int(row.argmax()) in mids})
    log_run("intervene_sweep", {"prompt": prompt, "mode": mode, "metric": metric,
                                "flipped_pcts": [o["pct"] for o in out if o["flipped"]]})
    return jsonify({"points": out, "metric": metric, "mode": mode})


@app.post("/api/generate")
def generate():
    """Greedy model continuation of the prompt (what the model actually says)."""
    d = request.json
    prompt = d["prompt"]
    n = max(1, min(int(d.get("max_new_tokens", 16)), 40))
    with INFER_LOCK, torch.inference_mode():
        enc = STATE["tok"](prompt, return_tensors="pt").to(STATE["device"])
        out = STATE["hf"].generate(**enc, max_new_tokens=n, do_sample=False)
    cont = STATE["tok"].decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    log_run("generate", {"prompt": prompt, "continuation": cont, "n": n})
    return jsonify({"continuation": cont, "n": n})


@app.post("/api/slicepage")
def slicepage():
    """The paper's interactive d3 layer x position slice (jlens.vis), self-contained
    HTML returned for embedding in an iframe. ~one read's compute."""
    d = request.json
    prompt = d["prompt"]
    pin = (d.get("track") or "").strip()
    pinned = set(word_ids(pin)) if pin else None
    with INFER_LOCK, torch.inference_mode():
        sd = jvis.compute_slice(STATE["lens_model"], STATE["lens"], prompt,
                                top_n=10, pinned_token_ids=pinned)
        html, _, _ = jvis.build_page(sd, prompt, title="J-lens slice",
                                     description=prompt, pinned_token_ids=pinned, mode="embed")
    STATE["last_slice_html"] = html  # served at GET /slice for a same-origin iframe

    # Compact, interpretable summary (no extra forward — extracted from SliceData).
    positions = list(sd.context_token_strs)
    slayers = list(sd.layers)
    tracked = list(sd.tracked_token_ids)
    pin_ids = [t for t in (sd.pinned_token_ids or set()) if t in tracked]
    pin_by_layer = []
    if pin_ids and sd.rank_tensor is not None:
        rt = sd.rank_tensor  # [seq_len, n_layers, n_tracked]
        pin_k = [tracked.index(t) for t in pin_ids]
        for li in range(len(slayers)):
            best, at = None, None
            for p in range(sd.seq_len):
                for k in pin_k:
                    r = int(rt[p, li, k])
                    if r >= 0 and (best is None or r < best):
                        best, at = r, p
            pin_by_layer.append({"layer": int(slayers[li]),
                                 "best_rank": (best + 1) if best is not None else None,  # 1-indexed
                                 "at_pos": at, "at_tok": positions[at] if at is not None else None})
    last = sd.seq_len - 1
    top1_last = [{"layer": int(slayers[li]), "tok": decode(int(sd.top_ids[last, li, 0]))}
                 for li in range(len(slayers))]
    summary = {"prompt": prompt, "pin": pin, "positions": positions,
               "pin_rank_by_layer": pin_by_layer, "top1_at_last_pos_by_layer": top1_last}

    # Read-equivalent block at the last position — extracted from the SAME forward,
    # so "전체 실행" can skip the redundant /api/read call. model_top is the
    # deepest source layer's lens top-k (≈ model output), hence approx=True.
    kmax = min(6, sd.top_ids.shape[2])
    pin_k0 = tracked.index(pin_ids[0]) if pin_ids else None
    read_layers = [{
        "layer": int(slayers[li]),
        "top": [decode(int(sd.top_ids[last, li, k])) for k in range(kmax)],
        "track_rank": (int(sd.rank_tensor[last, li, pin_k0]) + 1  # 1-indexed
                       if (pin_k0 is not None and sd.rank_tensor is not None) else None),
    } for li in range(len(slayers))]
    read_block = {"tokens": positions, "position": last,
                  "model_top": [decode(int(sd.top_ids[last, len(slayers) - 1, k])) for k in range(kmax)],
                  "layers": read_layers, "track": pin, "track_ok": bool(pin_ids), "approx": True}
    best_pin = min((x["best_rank"] for x in pin_by_layer if x["best_rank"] is not None), default=None)
    log_run("slice", {"prompt": prompt, "pin": pin, "best_pin_rank": best_pin})
    return jsonify({"ok": True, "pin": pin, "bytes": len(html),
                    "summary": summary, "read": read_block})


@app.get("/slice")
def slice_html():
    return Response(STATE.get("last_slice_html") or
                    "<p style='font-family:sans-serif;padding:1rem'>먼저 슬라이스를 실행하세요.</p>",
                    mimetype="text/html")


def _swap_logits(prompt, src, tgt, alpha, band, random_dir):
    lens, lens_model, device = STATE["lens"], STATE["lens_model"], STATE["device"]
    src_id, tgt_id = single_id(src), single_id(tgt)
    if src_id is None or tgt_id is None:
        return None, src_id, tgt_id
    ops = build_swap_operators(lens, lens_model, band, src_id, tgt_id, device,
                               random_tgt=random_dir)
    enc = STATE["tok"](prompt, return_tensors="pt").to(device)
    with torch.inference_mode(), swap_hooks(lens_model, ops, alpha=alpha):
        return STATE["hf"](**enc).logits[0, -1].float().cpu(), src_id, tgt_id


@app.post("/api/swap")
def swap():
    d = request.json
    prompt, src, tgt = d["prompt"], d["src"].strip(), d["tgt"].strip()
    alpha = float(d.get("alpha", 2))
    lo, hi = float(d.get("lo", 40)) / 100, float(d.get("hi", 90)) / 100
    random_dir = bool(d.get("random_dir", False))
    layers = STATE["layers"]
    band = band_from_fraction(layers, lo, hi)

    # token validity feedback
    def tok_info(w):
        return {"word": w, "id": single_id(w),
                "variants": token_variants(STATE["tok"], [w]).get(w, [])}
    src_info, tgt_info = tok_info(src), tok_info(tgt)
    if src_info["id"] is None or tgt_info["id"] is None:
        return jsonify({"error": "multi_token", "src": src_info, "tgt": tgt_info}), 400

    with INFER_LOCK:
        base = next_token_logits(STATE["hf"], STATE["tok"], prompt, STATE["device"])
        swapped, _, _ = _swap_logits(prompt, src, tgt, alpha, band, random_dir)

    src_ids, tgt_ids = word_ids(src), word_ids(tgt)

    # Success metric: default = the swapped-in concept (tgt). For the paper's
    # probe-swap, success is the FINAL answer (swap_answer) — pass it as `success`.
    success = (d.get("success") or "").strip()
    metric_ids = word_ids(success) if success else []
    metric_word = success if metric_ids else tgt
    if not metric_ids:
        metric_ids = tgt_ids

    def pr(row, ids):
        # aggregate over surface variants: P = sum, rank = best (min)
        probs = row.softmax(-1)
        return {"p": round(float(probs[ids].sum()), 4),
                "rank": min(rank_of(row, i) for i in ids)}

    flipped = int(swapped.argmax()) in metric_ids
    log_run("swap", {"prompt": prompt, "src": src, "tgt": tgt, "success": success or None,
                     "alpha": alpha, "band_pct": [int(lo * 100), int(hi * 100)],
                     "flipped": flipped, "metric": metric_word,
                     "P_metric": [pr(base, metric_ids)["p"], pr(swapped, metric_ids)["p"]],
                     "random_dir": random_dir})
    return jsonify({
        "band": band, "band_pct": [int(lo * 100), int(hi * 100)],
        "metric_word": metric_word,
        "baseline": {"top": topk_list(base), "src": pr(base, src_ids), "tgt": pr(base, metric_ids)},
        "swapped": {"top": topk_list(swapped), "src": pr(swapped, src_ids), "tgt": pr(swapped, metric_ids)},
        "flipped": flipped,
        "random_dir": random_dir, "alpha": alpha,
    })


@app.post("/api/sweep")
def sweep():
    d = request.json
    prompt, src, tgt = d["prompt"], d["src"].strip(), d["tgt"].strip()
    alpha = float(d.get("alpha", 2))
    window = int(d.get("window", 6))
    layers = STATE["layers"]
    n = len(layers)
    if single_id(src) is None or single_id(tgt) is None:
        return jsonify({"error": "multi_token"}), 400
    tgt_ids = word_ids(tgt)
    # Big models: fewer windows by default so the sweep stays ~minutes.
    n_points = int(d.get("points") or (8 if n >= 30 else 12))
    n_points = max(3, min(n_points, n))
    step = max(1, n // n_points)
    out = []
    with INFER_LOCK:
        for S in range(0, n - 1, step):
            wl = layers[S:S + window]
            if not wl:
                continue
            sl, _, _ = _swap_logits(prompt, src, tgt, alpha, wl, False)
            out.append({
                "start": layers[S], "pct": int(100 * S / n),
                "window": [wl[0], wl[-1]],
                "p_tgt": round(float(sl.softmax(-1)[tgt_ids].sum()), 4),
                "top1": decode(sl.argmax()),
                "flipped": int(sl.argmax()) in tgt_ids,
            })
    log_run("sweep", {"prompt": prompt, "src": src, "tgt": tgt, "alpha": alpha,
                      "window": window, "points": len(out),
                      "flipped_pcts": [p["pct"] for p in out if p["flipped"]],
                      "max_p_tgt": max((p["p_tgt"] for p in out), default=0)})
    return jsonify({"points": out})


SYS_PROMPT = (
    "너는 언어모델 해석가능성 실험을 설명하는 조수다. 사용자는 Jacobian lens(J-lens)로 "
    "소형 오픈모델을 실험 중이다. J-lens는 중간 레이어 활성값을 출력 어휘로 번역해, 모델이 "
    "'아직 말하지 않은 생각'을 읽어낸다. 스왑은 개념 A의 렌즈 방향을 B로 교체해 forward "
    "중 활성값만 수술하는 인과 개입이다(가중치 불변). 결과 JSON을 받아 한국어로 **충분히 상세하게** "
    "(핵심 위주, 마크다운 볼드/불릿 활용, 길이 제한 없음) 설명하라. 반드시 구체적 숫자를 인용하고, "
    "과장하지 마라. 스왑이 성공/부분/실패인지, 개념이 어느 레이어·위치에서 판독되는지, 인과 자리가 "
    "어디인지, 결과들 사이의 관계(읽기=상관 vs 스왑=인과)를 연결해 설명하고, 다음에 무엇을 바꿔보면 "
    "좋을지 제안을 포함하라. 확실하지 않으면 그렇다고 말하라.\n\n"
    "설명 뒤에는 반드시 아래 형식의 '실행 제안'을 ```json 코드블록``` 하나로 덧붙여라. "
    "프론트엔드가 이걸 파싱해 컨트롤에 그대로 적용하고 재실행한다. 관련 있는 키만 넣되 "
    "action은 필수다. 키: action('read'|'swap'|'sweep'|'all'), prompt(문자열), src(문자열), "
    "tgt(문자열), alpha(숫자 0.5~4), depth_start(정수 0~100), depth_end(정수 0~100), "
    "window(정수 2~12), points(정수 4~16), position(정수). 값은 네 다음-제안을 실제로 반영하라. "
    "예: ```json\n{\"action\":\"sweep\",\"src\":\"Paris\",\"tgt\":\"London\",\"alpha\":2,"
    "\"depth_start\":56,\"depth_end\":84,\"window\":3,\"points\":14}\n```"
)


def build_user_prompt(kind, payload, history=None, bundle=None):
    ctx = {
        "read": "레이어별 렌즈 판독 결과. layers[*].track_rank는 추적 토큰의 렌즈 rank"
                "(1=최상위, 낮을수록 강하게 판독). model_top은 모델의 실제 다음 토큰.",
        "swap": "개념 스왑 결과. baseline vs swapped 다음-토큰 분포와 P(src)/P(tgt)/rank, "
                "flipped=출력 top1이 tgt로 바뀌었는지, band=개입 레이어, alpha=강도, "
                "random_dir=무작위방향 대조군 여부. 숫자가 의미없는 토큰(문장부호)으로 top1이 "
                "깨지면 과도개입(alpha 과다)이다.",
        "sweep": "스왑 창(6개 레이어)을 깊이별로 옮기며 P(tgt)를 측정. points[*].pct=깊이%, "
                 "flipped=그 창에서 출력이 tgt로 뒤집혔는지. 어느 깊이부터 뒤집히는지가 인과 자리.",
        "steer": "스티어링: 개념 방향 v_w를 residual에 더함(α>0 주입 / α<0 억제). baseline vs "
                 "intervened의 P(concept)·rank·flipped로 개념이 출력으로 밀려나오는지 본다.",
        "ablate": "절제: 개념 방향을 residual에서 투영 제거(h-=(h·v)v). baseline 대비 그 개념의 "
                  "P·rank가 떨어지는지, 출력이 어떻게 바뀌는지 본다.",
        "combo": "콤보: 여러 개입(스왑/스티어/절제)을 한 forward에 동시에 적용. ops에 각 개입이 있고 "
                 "metric은 성공 판정 토큰. 개입들이 서로 간섭·중첩할 수 있으니 결과를 신중히 해석.",
        "steer_sweep": "강도 α를 바꿔가며 개념 rank 변화(강도↑ → rank가 1로 내려가면 주입 성공). "
                       "points[*]=[alpha, rank/p]. 어느 강도부터 효과가 나는지가 핵심.",
        "slice": "레이어×토큰위치 슬라이스. pin_rank_by_layer=핀 개념의 층별 최고 렌즈 rank"
                 "(1에 가까울수록 강하게 판독)와 그 위치(at_tok). top1_at_last_pos_by_layer="
                 "마지막(답) 위치에서 층별 렌즈 top-1 토큰. 이걸로 개념이 어느 층·어느 위치에서 "
                 "떠오르는지, 그리고 초반(무의미)→중반(개념/워크스페이스)→후반(다음 토큰) 진행을 분석.",
    }.get(kind, "")
    text = (f"현재 초점(focus) 실험: {kind}\n설명: {ctx}\n\n"
            f"focus 결과 JSON:\n{json.dumps(payload, ensure_ascii=False)[:7000]}")
    if bundle:
        text += ("\n\n■ 사용자가 이번에 실행한 **모든 결과 묶음**(있는 것만 — read/swap/sweep/"
                 "slice(=레이어×위치 히트맵)):\n" + json.dumps(bundle, ensure_ascii=False)[:9000]
                 + "\n\n→ focus 결과를 중심으로 하되, **이 묶음의 모든 결과를 종합**해 분석하라. "
                 "특히 **slice(히트맵)가 있으면 반드시** 개념이 어느 층·어느 위치에서 떠오르는지"
                 "(pin_rank_by_layer)와 sensory→workspace→motor 진행을 짚어라. read/swap/sweep이 "
                 "함께 있으면 '렌즈가 읽은 것(상관)'과 '스왑으로 바뀐 것(인과)'을 연결해서 해석하라.")
    if history:
        text += ("\n\n이전 반복들의 요약(오래된→최신):\n"
                 + json.dumps(history, ensure_ascii=False)[:4000]
                 + "\n→ 이전 조건들과 비교해 무엇을 바꿨고 지표가 어떻게 변했는지(개선/악화/무변화)와"
                 " 수렴 여부도 함께 언급하라.")
    return text


def _llm_call(model, messages, max_tokens=700):
    """Route a chat completion to a local proxy or OpenRouter. Raises on error."""
    if model.startswith("proxy:"):
        body = json.dumps({"model": model.split(":", 1)[1], "messages": messages}).encode()
        req = urllib.request.Request(PROXY_CHAT, data=body, headers={
            "Authorization": "Bearer claude-proxy", "Content-Type": "application/json"})
        timeout = 150
    elif model.startswith("codex:"):
        body = json.dumps({
            "model": model.split(":", 1)[1],
            "reasoning_effort": CODEX_PROXY_REASONING,
            "messages": messages,
        }).encode()
        req = urllib.request.Request(CODEX_PROXY_CHAT, data=body, headers={
            "Authorization": "Bearer codex-proxy", "Content-Type": "application/json"})
        timeout = 300
    else:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("no OpenRouter key")
        body = json.dumps({"model": model, "messages": messages,
                           "max_tokens": max_tokens, "temperature": 0.3}).encode()
        req = urllib.request.Request(OPENROUTER_URL, data=body, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
            "HTTP-Referer": "http://127.0.0.1", "X-Title": "J-lens tester"})
        timeout = 60
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    return resp["choices"][0]["message"]["content"]


@app.post("/api/interpret")
def interpret():
    d = request.json or {}
    kind = d.get("kind", "swap")
    payload = d.get("payload", {})
    models = available_models()
    model = d.get("model") or (models[0] if models else None)
    if not model:
        return jsonify({"error": "no_backend"}), 400

    history = d.get("history")
    bundle = d.get("bundle")
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": build_user_prompt(kind, payload, history, bundle)},
    ]
    try:
        return jsonify({"text": _llm_call(model, messages, max_tokens=1400), "model": model})
    except urllib.error.HTTPError as e:
        return jsonify({"error": "http", "status": e.code, "detail": e.read().decode()[:400]}), 502
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": "exc", "detail": str(e)}), 502


CHAT_SYS = (
    "너는 J-lens(Jacobian lens) 해석 실험 도구의 조수다. 사용자의 현재 실험 결과·맥락에 대해 "
    "한국어로 간결·정확하게 답한다. J-lens는 중간층 활성값을 출력 어휘로 번역해 '아직 말하지 않은 "
    "생각'을 읽고, 스왑은 개념 방향을 교체하는 인과 개입(가중치 불변)이다. 과장하지 말고 숫자에 "
    "근거하며, 확실치 않으면 모른다고 말하라. read는 상관이지 인과가 아님을 유념하라."
)


@app.post("/api/chat")
def chat():
    d = request.json or {}
    models = available_models()
    model = d.get("model") or (models[0] if models else None)
    if not model:
        return jsonify({"error": "no_backend"}), 400
    context = d.get("context")
    user_messages = d.get("messages", [])
    messages = [{"role": "system", "content": CHAT_SYS}]
    if context:
        messages.append({"role": "system",
                         "content": "현재 실험 맥락(JSON, 최근 결과+이력):\n"
                                    + json.dumps(context, ensure_ascii=False)[:4500]})
    messages += [{"role": m.get("role", "user"), "content": str(m.get("content", ""))[:4000]}
                 for m in user_messages][-16:]
    try:
        return jsonify({"text": _llm_call(model, messages), "model": model})
    except urllib.error.HTTPError as e:
        return jsonify({"error": "http", "status": e.code, "detail": e.read().decode()[:400]}), 502
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": "exc", "detail": str(e)}), 502


@app.post("/api/save")
def save_snapshot():
    """Write a readable markdown snapshot of the current results to results/saved/."""
    d = request.json or {}
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = RUNS_DIR / "saved"
    outdir.mkdir(exist_ok=True)
    p = outdir / f"snapshot-{stamp}.md"
    p.write_text(d.get("markdown") or json.dumps(d.get("data", {}), ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return jsonify({"ok": True, "path": str(p)})


if __name__ == "__main__":
    import sys
    key = sys.argv[1] if len(sys.argv) > 1 else "gemma4b"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("JLENS_PORT", 8137))
    ensure_codex_proxy()
    print(f"Loading model '{key}' ... (first run downloads weights)")
    load(key)
    print(f"Ready: {MODELS[key][0]}  |  open http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
