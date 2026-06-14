"""
================================================================================
FILE: run_demo.py
PURPOSE: Interactive terminal demos of KV cache behaviour built with pure
         NumPy — no PyTorch required. Run this first if you cannot install
         PyTorch, or want to understand the math without a framework.

WHY NUMPY?
----------
Implementing attention in NumPy forces you to see every matrix multiplication
explicitly. There are no .forward() magic methods, no autograd graphs — just
arrays and math. This makes it easier to trace exactly what gets cached,
what gets recomputed, and how memory grows.

WHAT IS IMPLEMENTED (from scratch, no libraries)
-------------------------------------------------
  softmax(x)          — numerically stable softmax (subtracts max first to
                         prevent overflow: e^(x - max) instead of e^x)

  attention(Q, K, V)  — scaled dot-product attention: Q @ K^T / √d_k → softmax → V

  Config              — small dataclass holding d_model, layers, heads, head_dim

  NaiveLayer          — baseline: recomputes ALL K,V from scratch every step
  CachedLayer         — KV cache: stores K,V, only projects new token each step
                        Supports GQA: num_kv_heads < num_q_heads reduces cache size
  SlidingLayer        — sliding window: oldest tokens evicted when cache > window

THE 7 DEMOS
-----------

  DEMO 1 — Speed: No Cache vs KV Cache
  ───────────────────────────────────────
  What it does: Generates 30 tokens with both NaiveLayer and CachedLayer.
                Measures wall-clock time for each decode step.
  What you see:
    No Cache:  each step gets slower as history grows (more re-computation)
    KV Cache:  steps stay roughly constant (only new token projected)
    Table at end: avg ms/token and total ms for both methods
  Key output: "KV Cache is X.Xx faster per token"
  Why it matters: This is the core speedup. At T=50 tokens, ~5x. At T=2000,
                  the gap would be ~2000x.

  DEMO 2 — Memory: MHA vs GQA vs MQA
  ─────────────────────────────────────
  What it does: Calculates cache memory for Llama-2 7B (32 layers, 128 head_dim)
                using the formula: 2 × layers × kv_heads × T × head_dim × 2 bytes
  What you see:
    MHA  (32 KV heads): 524 KB/token → 2.15 GB at 4K context
    GQA  (8 KV heads):  131 KB/token → 537 MB  (4x smaller)
    GQA  (4 KV heads):  66 KB/token  → 268 MB  (8x smaller)
    MQA  (1 KV head):   16 KB/token  → 67 MB   (32x smaller)
  Why it matters: These are the real numbers production systems deal with.

  DEMO 3 — Cache Growth Trace
  ─────────────────────────────
  What it does: Runs 15 decode steps with a CachedLayer on a 5-token prompt.
                After each step, prints cache token count + size in KB + bar.
  What you see: Cache grows by exactly one row per decode step.
                Each new word appended shows at the end of the sentence.
  Why it matters: Makes the "cache is just appending rows" concept concrete.

  DEMO 4 — Sliding Window Fixed Memory
  ───────────────────────────────────────
  What it does: Runs 20 tokens through both CachedLayer and SlidingLayer
                (window=6). Prints both cache sizes side by side.
  What you see:
    Normal:   2 → 3 → 4 → ... → 20  (grows forever)
    Sliding:  2 → 3 → 4 → 5 → 6 → 6 → 6 → ...  (capped at 6)
    "← old tokens evicted" annotation appears when cap is reached
  Why it matters: Shows the memory vs. context tradeoff directly.

  DEMO 5 — Prefix Caching Speedup
  ────────────────────────────────
  What it does: Simulates 6 requests that share the same 40-token system prompt
                but have different 5-token queries. Times both approaches:
                  Without prefix cache: recompute system prompt KV every request
                  With prefix cache:    compute system prompt KV once, reuse
  What you see: Table showing per-request time and speedup (Xx) for each.
                "Prefix caching saves X% of prefill time" summary.
  Why it matters: In production (Claude API, OpenAI, SGLang), shared system
                  prompts can be cached — this demo shows the actual savings.

  DEMO 6 — Attention Weight Heatmap
  ────────────────────────────────────
  What it does: Runs full attention on "The quick brown fox jumps over the mat"
                (8 tokens). Prints the averaged attention weight matrix.
  What you see: A grid where each cell shows how much each word attends to
                each previous word. Upper triangle = "· ·" (masked, future).
                Diagonal is usually high (self-attention).
                Top 4 attention pairs printed with bar chart.
  Why it matters: Shows attention is not uniform — structure emerges even with
                  random weights.

  DEMO 7 — GQA vs MHA Timing
  ────────────────────────────
  What it does: Runs 55 decode steps with MHA (8 KV heads), GQA (2 KV heads),
                and MQA (1 KV head). Measures total decode time for each.
  What you see:
    MHA  (8 KV heads): X ms   (baseline)
    GQA  (2 KV heads): Y ms   (4x smaller cache, slightly faster)
    MQA  (1 KV head):  Z ms   (8x smaller cache, fastest)
  Why it matters: Fewer KV heads → less data to read per step → faster decode.

HOW TO RUN
----------
  python3 run_demo.py          # run all 7 demos in sequence
  python3 run_demo.py 1        # run only Demo 1
  python3 run_demo.py 3 5 7    # run Demos 3, 5, and 7

REQUIREMENTS
------------
  numpy only — no PyTorch, no GPU needed.
  For PyTorch versions of the same experiments, see run_demo_torch.py.
================================================================================
"""

import numpy as np
import time
import math
import sys

def bold(s):   return f"\033[1m{s}\033[0m"
def green(s):  return f"\033[32m{s}\033[0m"
def cyan(s):   return f"\033[36m{s}\033[0m"

def bar(value, max_value, width=30):
    if max_value == 0: return "░" * width
    filled = int(round(max(0, min(value / max_value, 1)) * width))
    return "█" * filled + "░" * (width - filled)

def randn(rng, *shape):
    return rng.standard_normal(shape).astype(np.float32)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def attention(Q, K, V):
    scale  = math.sqrt(Q.shape[-1])
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / scale
    return np.matmul(softmax(scores, axis=-1), V)

class Config:
    def __init__(self, d_model, layers, q_heads, kv_heads):
        self.d_model  = d_model
        self.layers   = layers
        self.q_heads  = q_heads
        self.kv_heads = kv_heads
        self.head_dim = d_model // q_heads

CFG = Config(d_model=256, layers=4, q_heads=8, kv_heads=8)

# ── implementations ─────────────────────────────────────────

class NaiveLayer:
    def __init__(self, cfg, rng):
        D, H, Hd = cfg.d_model, cfg.q_heads, cfg.head_dim
        self.H, self.Hd = H, Hd
        self.Wq = randn(rng, D, H*Hd)  * 0.02
        self.Wk = randn(rng, D, H*Hd)  * 0.02
        self.Wv = randn(rng, D, H*Hd)  * 0.02
        self.Wo = randn(rng, H*Hd, D)  * 0.02

    def forward(self, tokens):
        T, D = tokens.shape
        H, Hd = self.H, self.Hd
        Q = (tokens @ self.Wq).reshape(T, H, Hd).transpose(1,0,2)
        K = (tokens @ self.Wk).reshape(T, H, Hd).transpose(1,0,2)
        V = (tokens @ self.Wv).reshape(T, H, Hd).transpose(1,0,2)
        out = attention(Q, K, V)[:, -1:, :]
        return out.transpose(1,0,2).reshape(1, H*Hd) @ self.Wo


class CachedLayer:
    def __init__(self, cfg, rng):
        D, Hq, Hkv, Hd = cfg.d_model, cfg.q_heads, cfg.kv_heads, cfg.head_dim
        self.Hq, self.Hkv, self.Hd = Hq, Hkv, Hd
        self.groups = Hq // Hkv
        self.Wq = randn(rng, D, Hq*Hd)  * 0.02
        self.Wk = randn(rng, D, Hkv*Hd) * 0.02
        self.Wv = randn(rng, D, Hkv*Hd) * 0.02
        self.Wo = randn(rng, Hq*Hd, D)  * 0.02
        self.ck = self.cv = None

    def reset(self): self.ck = self.cv = None

    def prefill(self, tokens):
        T, _ = tokens.shape
        Hq, Hkv, Hd = self.Hq, self.Hkv, self.Hd
        Q = (tokens @ self.Wq).reshape(T, Hq, Hd).transpose(1,0,2)
        K = (tokens @ self.Wk).reshape(T, Hkv, Hd).transpose(1,0,2)
        V = (tokens @ self.Wv).reshape(T, Hkv, Hd).transpose(1,0,2)
        self.ck, self.cv = K.copy(), V.copy()
        Ke = np.repeat(K, self.groups, axis=0)
        Ve = np.repeat(V, self.groups, axis=0)
        out = attention(Q, Ke, Ve)[:, -1:, :]
        return out.transpose(1,0,2).reshape(1, Hq*Hd) @ self.Wo

    def decode(self, tok):
        Hq, Hkv, Hd = self.Hq, self.Hkv, self.Hd
        Q = (tok @ self.Wq).reshape(1, Hq, Hd).transpose(1,0,2)
        K = (tok @ self.Wk).reshape(1, Hkv, Hd).transpose(1,0,2)
        V = (tok @ self.Wv).reshape(1, Hkv, Hd).transpose(1,0,2)
        self.ck = np.concatenate([self.ck, K], axis=1)
        self.cv = np.concatenate([self.cv, V], axis=1)
        Ke = np.repeat(self.ck, self.groups, axis=0)
        Ve = np.repeat(self.cv, self.groups, axis=0)
        out = attention(Q, Ke, Ve)
        return out.transpose(1,0,2).reshape(1, Hq*Hd) @ self.Wo

    def n_tokens(self): return 0 if self.ck is None else self.ck.shape[1]
    def nbytes(self, layers=4): return 0 if self.ck is None else 2 * layers * self.ck.nbytes


class SlidingLayer:
    def __init__(self, cfg, rng, window):
        D, H, Hd = cfg.d_model, cfg.q_heads, cfg.head_dim
        self.H, self.Hd, self.W = H, Hd, window
        self.Wq = randn(rng, D, H*Hd) * 0.02
        self.Wk = randn(rng, D, H*Hd) * 0.02
        self.Wv = randn(rng, D, H*Hd) * 0.02
        self.Wo = randn(rng, H*Hd, D) * 0.02
        self.ck = self.cv = None

    def decode(self, tok):
        H, Hd = self.H, self.Hd
        Q = (tok @ self.Wq).reshape(1, H, Hd).transpose(1,0,2)
        K = (tok @ self.Wk).reshape(1, H, Hd).transpose(1,0,2)
        V = (tok @ self.Wv).reshape(1, H, Hd).transpose(1,0,2)
        self.ck = np.concatenate([self.ck, K], axis=1) if self.ck is not None else K
        self.cv = np.concatenate([self.cv, V], axis=1) if self.cv is not None else V
        if self.ck.shape[1] > self.W:
            self.ck = self.ck[:, -self.W:, :]
            self.cv = self.cv[:, -self.W:, :]
        out = attention(Q, self.ck, self.cv)
        return out.transpose(1,0,2).reshape(1, H*Hd) @ self.Wo

    def size(self): return 0 if self.ck is None else self.ck.shape[1]

# ═══════════════════════════════════════════════════════════
# DEMO 1 — Speed
# ═══════════════════════════════════════════════════════════

def demo_speed():
    print("\n" + "═"*62)
    print(bold("  DEMO 1: No Cache vs KV Cache — Speed"))
    print("═"*62)

    rng = np.random.default_rng(42)
    D = CFG.d_model
    PROMPT, GEN = 20, 30

    tokens = rng.standard_normal((PROMPT + GEN, D)).astype(np.float32)
    naive  = NaiveLayer(CFG, rng)
    cached = CachedLayer(CFG, rng)

    print(f"\n  {CFG.layers}L {CFG.q_heads}H d={CFG.d_model} | Prompt={PROMPT} Generate={GEN}\n")

    print(cyan("  [No Cache]") + " — recomputes ALL K,V every step")
    nc_times = []
    hist = tokens[:PROMPT].copy()
    for step in range(GEN):
        t0 = time.perf_counter()
        naive.forward(hist)
        nc_times.append((time.perf_counter() - t0) * 1000)
        hist = np.vstack([hist, tokens[PROMPT + step]])
        if step == 0 or (step+1) % 10 == 0:
            t = nc_times[-1]
            print(f"    step {step+1:2d} (history={PROMPT+step+1:3d} tok): "
                  f"{t:.3f} ms  {bar(t, max(nc_times), 22)}")

    print(cyan("\n  [KV Cache]") + " — only processes NEW token")
    kv_times = []
    cached.prefill(tokens[:PROMPT])
    for step in range(GEN):
        t0 = time.perf_counter()
        cached.decode(tokens[PROMPT+step:PROMPT+step+1])
        kv_times.append((time.perf_counter() - t0) * 1000)
        if step == 0 or (step+1) % 10 == 0:
            t = kv_times[-1]
            print(f"    step {step+1:2d} (cache={cached.n_tokens():3d} tok): "
                  f"{t:.3f} ms  {bar(t, max(nc_times), 22)}")

    avg_nc = sum(nc_times) / len(nc_times)
    avg_kv = sum(kv_times) / len(kv_times)
    print(f"\n  ┌──────────────────┬──────────────┬──────────────┐")
    print(f"  │ Method           │ Avg ms/token │ Total        │")
    print(f"  ├──────────────────┼──────────────┼──────────────┤")
    print(f"  │ No Cache         │ {avg_nc:>8.3f} ms │ {sum(nc_times):>8.1f} ms │")
    print(f"  │ KV Cache         │ {avg_kv:>8.3f} ms │ {sum(kv_times):>8.1f} ms │")
    print(f"  └──────────────────┴──────────────┴──────────────┘")
    print(green(f"\n  KV Cache is {avg_nc/avg_kv:.1f}x faster per token"))
    print("  (gap widens the longer the sequence gets)")

# ═══════════════════════════════════════════════════════════
# DEMO 2 — Memory numbers (Llama-2 7B scale)
# ═══════════════════════════════════════════════════════════

def demo_memory():
    print("\n" + "═"*62)
    print(bold("  DEMO 2: KV Cache Memory — MHA vs GQA vs MQA"))
    print("═"*62)

    LAYERS, HEAD_DIM, DTYPE = 32, 128, 2   # Llama-2 7B, float16

    def fmt(b):
        if b >= 1e9: return f"{b/1e9:.2f} GB"
        if b >= 1e6: return f"{b/1e6:.1f} MB"
        return f"{b/1e3:.0f} KB"

    configs = [
        ("MHA  (32 KV heads)", 32),
        ("GQA-8 (8 KV heads)",  8),
        ("GQA-4 (4 KV heads)",  4),
        ("MQA  (1 KV head)",    1),
    ]

    mha_per = 2 * LAYERS * 32 * HEAD_DIM * DTYPE

    print(f"\n  Llama-2 7B config: {LAYERS} layers, head_dim={HEAD_DIM}, float16\n")
    print(f"  {'Method':<22} {'Per token':<12} {'4K ctx':<12} {'32K ctx':<12} {'vs MHA'}")
    print(f"  {'──────':<22} {'─────────':<12} {'──────':<12} {'───────':<12} {'──────'}")

    for name, kv_heads in configs:
        per   = 2 * LAYERS * kv_heads * HEAD_DIM * DTYPE
        ctx4  = per * 4096
        ctx32 = per * 32768
        note  = f"{mha_per/per:.0f}x smaller" if per < mha_per else "baseline"
        print(f"  {name:<22} {fmt(per):<12} {fmt(ctx4):<12} {fmt(ctx32):<12} {green(note)}")

    print("\n  Memory bar (4K context):")
    for name, kv_heads in configs:
        per  = 2 * LAYERS * kv_heads * HEAD_DIM * DTYPE
        ctx4 = per * 4096
        print(f"    {name:<22} {bar(ctx4, mha_per*4096, 32)} {fmt(ctx4)}")

# ═══════════════════════════════════════════════════════════
# DEMO 3 — Live cache growth
# ═══════════════════════════════════════════════════════════

def demo_cache_growth():
    print("\n" + "═"*62)
    print(bold("  DEMO 3: Watch KV Cache Grow Token by Token"))
    print("═"*62)

    rng = np.random.default_rng(7)
    D   = CFG.d_model
    P, G = 5, 15
    tokens = rng.standard_normal((P + G, D)).astype(np.float32)
    layer  = CachedLayer(CFG, rng)
    words  = ["The","quick","brown","fox","jumps","over","the",
              "lazy","dog","sat","down","to","rest","now","ever","after","once","upon"]

    max_b = layer.nbytes() if False else (2 * CFG.layers * CFG.q_heads * CFG.head_dim * 4 * (P+G))
    layer.prefill(tokens[:P])

    print(f"\n  {CFG.layers} layers, {CFG.q_heads} heads, head_dim={CFG.head_dim}\n")
    print(f"  {'Phase':<9} {'Tokens':>7}  {'Size':>10}  {'Bar'}")
    print(f"  {'─────':<9} {'──────':>7}  {'────':>10}  {'───'}")

    def show(phase):
        b = layer.nbytes()
        n = layer.n_tokens()
        tail = " ".join(words[:n])[-30:]
        print(f"  {phase:<9} {n:>7}  {b/1024:>8.1f} KB  {bar(b, max_b, 28)}  …{tail}")

    show("PREFILL")
    for step in range(G):
        layer.decode(tokens[P+step:P+step+1])
        show("decode")

    print(f"\n  {layer.n_tokens()} tokens → {layer.nbytes()/1024:.1f} KB total")
    print(f"  {layer.nbytes()/1024/layer.n_tokens():.2f} KB per token")

# ═══════════════════════════════════════════════════════════
# DEMO 4 — Sliding window
# ═══════════════════════════════════════════════════════════

def demo_sliding_window():
    print("\n" + "═"*62)
    print(bold("  DEMO 4: Sliding Window — Fixed Memory Forever"))
    print("═"*62)

    rng    = np.random.default_rng(99)
    D      = CFG.d_model
    WINDOW = 6
    STEPS  = 20

    tokens  = rng.standard_normal((STEPS, D)).astype(np.float32)
    normal  = CachedLayer(CFG, rng)
    sliding = SlidingLayer(CFG, rng, window=WINDOW)

    normal.prefill(tokens[:1])
    sliding.decode(tokens[:1])

    print(f"\n  Window={WINDOW} | Steps={STEPS}\n")
    print(f"  {'Step':<6}  {'Normal':^22}  {'Sliding':^22}  Note")
    print(f"  {'────':<6}  {'──────':^22}  {'───────':^22}  ────")

    for step in range(1, STEPS):
        normal.decode(tokens[step:step+1])
        sliding.decode(tokens[step:step+1])
        ns, ss = normal.n_tokens(), sliding.size()
        note = green("← old tokens evicted") if ss == WINDOW and ns > WINDOW else ""
        print(f"  {step+1:<6}  {bar(ns, STEPS, 18)} {ns:<3}  {bar(ss, STEPS, 18)} {ss:<3}  {note}")

    print(f"\n  Normal  end: {normal.n_tokens()} tokens — unbounded")
    print(f"  Sliding end: {sliding.size()} tokens — always ≤ {WINDOW}")
    print(green("\n  Sliding window = O(1) memory regardless of sequence length ✓"))

# ═══════════════════════════════════════════════════════════
# DEMO 5 — Prefix caching
# ═══════════════════════════════════════════════════════════

def demo_prefix_caching():
    print("\n" + "═"*62)
    print(bold("  DEMO 5: Prefix Caching — Reuse Shared Prompt KV"))
    print("═"*62)

    rng     = np.random.default_rng(13)
    D       = CFG.d_model
    SYS_LEN = 40
    Q_LEN   = 5
    N       = 6

    sys_emb = rng.standard_normal((SYS_LEN, D)).astype(np.float32)
    qs      = [rng.standard_normal((Q_LEN, D)).astype(np.float32) for _ in range(N)]

    layer_nc = CachedLayer(CFG, rng)
    layer_c  = CachedLayer(CFG, rng)

    print(f"\n  Shared system prompt: {SYS_LEN} tokens")
    print(f"  Per-request query:    {Q_LEN} tokens")
    print(f"  Requests: {N}\n")

    times_nc = []
    for q in qs:
        layer_nc.reset()
        t0 = time.perf_counter()
        layer_nc.prefill(np.vstack([sys_emb, q]))
        times_nc.append((time.perf_counter() - t0) * 1000)

    layer_c.prefill(sys_emb)
    saved_k, saved_v = layer_c.ck.copy(), layer_c.cv.copy()

    times_c = []
    for q in qs:
        layer_c.ck, layer_c.cv = saved_k.copy(), saved_v.copy()
        t0 = time.perf_counter()
        layer_c.prefill(q)
        times_c.append((time.perf_counter() - t0) * 1000)

    M = max(times_nc)
    print(f"  {'Req':<4}  {'No prefix cache':^24}  {'With prefix cache':^24}  Speedup")
    print(f"  {'───':<4}  {'───────────────':^24}  {'─────────────────':^24}  ───────")
    for i in range(N):
        nc, c = times_nc[i], times_c[i]
        print(f"  {i+1:<4}  {nc:.3f} ms {bar(nc, M, 14)}  "
              f"{c:.3f} ms {bar(c, M, 14)}  {green(f'{nc/c:.1f}x')}")

    tnc, tc = sum(times_nc), sum(times_c)
    print(f"\n  Total: {tnc:.2f} ms (no cache) vs {tc:.2f} ms (cached)")
    print(green(f"  Prefix caching saves {(1-tc/tnc)*100:.0f}% of prefill time"))
    print(f"  System prompt computed once, reused {N} times")

# ═══════════════════════════════════════════════════════════
# DEMO 6 — Attention weight heatmap
# ═══════════════════════════════════════════════════════════

def demo_attention_weights():
    print("\n" + "═"*62)
    print(bold("  DEMO 6: Attention Weights — What Tokens Attend To"))
    print("═"*62)

    rng   = np.random.default_rng(5)
    D, H, Hd = 64, 4, 16
    words = ["The","quick","brown","fox","jumps","over","the","mat"]
    SEQ   = len(words)

    tokens = rng.standard_normal((SEQ, D)).astype(np.float32)
    Wq     = rng.standard_normal((D, H*Hd)).astype(np.float32) * 0.15
    Wk     = rng.standard_normal((D, H*Hd)).astype(np.float32) * 0.15

    Q = (tokens @ Wq).reshape(SEQ, H, Hd).transpose(1,0,2)
    K = (tokens @ Wk).reshape(SEQ, H, Hd).transpose(1,0,2)
    scores = np.matmul(Q, K.transpose(0,2,1)) / math.sqrt(Hd)
    mask   = np.triu(np.ones((SEQ, SEQ), dtype=bool), k=1)
    scores[:, mask] = -1e9
    W = softmax(scores, axis=-1).mean(axis=0)   # [SEQ, SEQ]

    print(f"\n  Averaged over {H} heads | causal mask applied\n")
    header = " " * 10 + "".join(f"{w:>7}" for w in words)
    print("  " + header)
    print("  " + " " * 10 + "─" * (7 * SEQ))

    shades = [" ", "░", "▒", "▓", "█"]
    for i, rw in enumerate(words):
        cells = []
        for j in range(SEQ):
            if j > i:
                cells.append("  ···")
            else:
                w = W[i, j]
                s = shades[min(int(w * len(shades) * 3), len(shades)-1)]
                cells.append(f" {s}{w:.2f}")
        print(f"  {rw:>8} │" + " ".join(cells))

    print("\n  █ high attention  ░ low  ··· future (masked)")

    pairs = sorted([(W[i,j], words[i], words[j])
                    for i in range(SEQ) for j in range(i+1)], reverse=True)
    print("\n  Top attention pairs:")
    for w, src, tgt in pairs[:4]:
        print(f"    {src:>6} → {tgt:<6}  {bar(w, 1.0, 20)} {w:.3f}")

# ═══════════════════════════════════════════════════════════
# DEMO 7 — GQA vs MHA timing
# ═══════════════════════════════════════════════════════════

def demo_gqa_timing():
    print("\n" + "═"*62)
    print(bold("  DEMO 7: GQA vs MHA — Real Compute & Cache Size"))
    print("═"*62)

    rng    = np.random.default_rng(21)
    D      = 256
    LAYERS = 4
    SEQ    = 60

    configs = [
        ("MHA  (8 KV heads)", Config(D, LAYERS, 8, 8)),
        ("GQA  (2 KV heads)", Config(D, LAYERS, 8, 2)),
        ("MQA  (1 KV head)",  Config(D, LAYERS, 8, 1)),
    ]

    tokens = rng.standard_normal((SEQ, D)).astype(np.float32)
    print(f"\n  Seq={SEQ} tokens | {LAYERS} layers\n")
    print(f"  {'Method':<22} {'Cache (KB)':<14} {'Decode time':<14} {'vs MHA'}")
    print(f"  {'──────':<22} {'──────────':<14} {'───────────':<14} {'──────'}")

    mha_b, mha_t = None, None
    for label, cfg in configs:
        layer = CachedLayer(cfg, rng)
        layer.prefill(tokens[:5])
        t0 = time.perf_counter()
        for i in range(5, SEQ):
            layer.decode(tokens[i:i+1])
        elapsed = (time.perf_counter() - t0) * 1000
        cb = layer.nbytes(LAYERS)

        if mha_b is None:
            mha_b, mha_t = cb, elapsed
            note = "baseline"
        else:
            note = green(f"{mha_b/cb:.1f}x smaller cache, {mha_t/elapsed:.1f}x faster")
        print(f"  {label:<22} {cb/1024:<14.1f} {elapsed:<14.2f} {note}")

    print(green("\n  Fewer KV heads → smaller cache → faster memory reads → faster decode ✓"))

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

DEMOS = {
    "1": ("Speed — No Cache vs KV Cache",   demo_speed),
    "2": ("Memory — MHA vs GQA vs MQA",     demo_memory),
    "3": ("Cache Growth Trace",              demo_cache_growth),
    "4": ("Sliding Window Fixed Memory",     demo_sliding_window),
    "5": ("Prefix Caching Speedup",          demo_prefix_caching),
    "6": ("Attention Weight Heatmap",        demo_attention_weights),
    "7": ("GQA vs MHA Timing",               demo_gqa_timing),
}

if __name__ == "__main__":
    print()
    print(bold("╔══════════════════════════════════════════════════════╗"))
    print(bold("║         KV CACHE — LIVE BEHAVIOUR DEMO               ║"))
    print(bold("║         pure numpy, no GPU needed                    ║"))
    print(bold("╚══════════════════════════════════════════════════════╝"))

    chosen = sys.argv[1:] if len(sys.argv) > 1 else list(DEMOS.keys())
    for key in chosen:
        if key in DEMOS:
            DEMOS[key][1]()
        else:
            print(f"  Unknown: '{key}'. Choose 1-7.")

    print("\n" + "═"*62)
    print(bold("  Run one: python3 run_demo.py 1"))
    print("═"*62 + "\n")
