"""
KV Cache Live Demo — PyTorch edition.
Run: python3 run_demo_torch.py
Run one demo: python3 run_demo_torch.py 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time, sys

def bold(s):  return f"\033[1m{s}\033[0m"
def green(s): return f"\033[32m{s}\033[0m"
def cyan(s):  return f"\033[36m{s}\033[0m"

def bar(v, mx, w=28):
    if mx == 0: return "░"*w
    f = int(round(max(0, min(v/mx, 1))*w))
    return "█"*f + "░"*(w-f)

def shape(t): return "×".join(str(d) for d in t.shape)

# ─────────────────────────────────────────────────────────────
# Model — tiny but realistic: 4 layers, 8 heads, d=256
# ─────────────────────────────────────────────────────────────

D, L, H, Hd = 256, 4, 8, 32   # d_model, layers, heads, head_dim
VOCAB = 500

class AttentionLayer(nn.Module):
    def __init__(self, kv_heads=8):
        super().__init__()
        self.H   = H
        self.Hkv = kv_heads
        self.G   = H // kv_heads
        self.Hd  = Hd
        self.Wq  = nn.Linear(D, H*Hd,   bias=False)
        self.Wk  = nn.Linear(D, kv_heads*Hd, bias=False)
        self.Wv  = nn.Linear(D, kv_heads*Hd, bias=False)
        self.Wo  = nn.Linear(H*Hd, D,   bias=False)
        # KV cache
        self.cache_k = None   # [1, Hkv, T, Hd]
        self.cache_v = None

    def reset(self):
        self.cache_k = self.cache_v = None

    @property
    def cache_len(self):
        return 0 if self.cache_k is None else self.cache_k.shape[2]

    def forward(self, x, use_cache=True):
        """
        x: [B, T_new, D]
        If use_cache=True, reads from and appends to self.cache_k/v.
        Returns: output [B, T_new, D]
        """
        B, T, _ = x.shape
        H, Hkv, G, Hd_ = self.H, self.Hkv, self.G, self.Hd

        # Project Q, K, V
        Q = self.Wq(x).view(B, T, H, Hd_).transpose(1,2)         # [B, H, T, Hd]
        K = self.Wk(x).view(B, T, Hkv, Hd_).transpose(1,2)       # [B, Hkv, T, Hd]
        V = self.Wv(x).view(B, T, Hkv, Hd_).transpose(1,2)       # [B, Hkv, T, Hd]

        if use_cache:
            # Append new K,V to cache
            if self.cache_k is None:
                self.cache_k, self.cache_v = K, V
            else:
                self.cache_k = torch.cat([self.cache_k, K], dim=2)
                self.cache_v = torch.cat([self.cache_v, V], dim=2)
            K_full, V_full = self.cache_k, self.cache_v
        else:
            K_full, V_full = K, V

        # Expand KV heads to match Q heads (GQA/MQA support)
        if G > 1:
            K_full = K_full.unsqueeze(2).expand(B, Hkv, G, K_full.shape[2], Hd_).reshape(B, H, -1, Hd_)
            V_full = V_full.unsqueeze(2).expand(B, Hkv, G, V_full.shape[2], Hd_).reshape(B, H, -1, Hd_)

        # Scaled dot-product attention
        scale  = math.sqrt(Hd_)
        scores = torch.matmul(Q, K_full.transpose(-2,-1)) / scale   # [B, H, T, T_full]

        # Causal mask for prefill (T>1); not needed for single-token decode
        if T > 1:
            T_full = K_full.shape[2]
            mask = torch.triu(torch.ones(T, T_full, dtype=torch.bool), diagonal=T_full-T+1)
            scores = scores.masked_fill(mask.to(scores.device), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out  = torch.matmul(attn, V_full)                            # [B, H, T, Hd]
        out  = out.transpose(1,2).contiguous().view(B, T, H*Hd_)
        return self.Wo(out)


class TinyLM(nn.Module):
    def __init__(self, kv_heads=8):
        super().__init__()
        self.embed  = nn.Embedding(VOCAB, D)
        self.layers = nn.ModuleList([AttentionLayer(kv_heads) for _ in range(L)])
        self.norm   = nn.LayerNorm(D)
        self.lm_head= nn.Linear(D, VOCAB, bias=False)

    def reset_cache(self):
        for l in self.layers: l.reset()

    @property
    def cache_tokens(self):
        return self.layers[0].cache_len

    def cache_bytes(self):
        total = 0
        for l in self.layers:
            if l.cache_k is not None:
                total += l.cache_k.numel() * 4   # float32 = 4 bytes
                total += l.cache_v.numel() * 4
        return total

    def prefill(self, token_ids):
        """token_ids: [B, T] — full prompt."""
        x = self.embed(token_ids)
        for layer in self.layers:
            x = x + layer(x, use_cache=True)
            x = self.norm(x)
        return self.lm_head(x)   # [B, T, VOCAB]

    def decode_step(self, token_id):
        """token_id: [B, 1] — single new token."""
        x = self.embed(token_id)
        for layer in self.layers:
            x = x + layer(x, use_cache=True)
            x = self.norm(x)
        logits = self.lm_head(x)  # [B, 1, VOCAB]
        return logits.argmax(dim=-1)   # [B, 1] — greedy next token


# ═══════════════════════════════════════════════════════════
# DEMO 1 — Tensor shapes at every step
# ═══════════════════════════════════════════════════════════

def demo_shapes():
    print("\n" + "═"*64)
    print(bold("  DEMO 1 (PyTorch): Tensor Shapes at Every Step"))
    print("═"*64)

    model = TinyLM()
    model.eval()

    PROMPT_LEN = 6
    prompt = torch.randint(0, VOCAB, (1, PROMPT_LEN))

    print(f"\n  Model: {L} layers, {H} Q-heads, {D} d_model, head_dim={Hd}")
    print(f"  Input token ids: {prompt.tolist()[0]}\n")

    print("  ── PREFILL ──────────────────────────────────────────────")
    print(f"  Input:   token_ids shape = {shape(prompt)}")
    embed = model.embed(prompt)
    print(f"  Embed:   {shape(embed)}  ({PROMPT_LEN} tokens × {D} dims)")

    layer = model.layers[0]
    Q = layer.Wq(embed).view(1, PROMPT_LEN, H, Hd).transpose(1,2)
    K = layer.Wk(embed).view(1, PROMPT_LEN, H, Hd).transpose(1,2)
    V = layer.Wv(embed).view(1, PROMPT_LEN, H, Hd).transpose(1,2)
    print(f"  Q:       {shape(Q)}  (batch × heads × seq × head_dim)")
    print(f"  K:       {shape(K)}  ← will be cached")
    print(f"  V:       {shape(V)}  ← will be cached")

    scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(Hd)
    print(f"  QK^T:    {shape(scores)}  (attention score matrix)")
    attn = F.softmax(scores, dim=-1)
    print(f"  softmax: {shape(attn)}  (attention weights)")

    model.reset_cache()
    with torch.no_grad():
        _ = model.prefill(prompt)

    print(f"\n  After prefill — KV cache in layer 0:")
    print(f"    cache_k: {shape(model.layers[0].cache_k)}  (heads × {PROMPT_LEN} tokens × head_dim)")
    print(f"    cache_v: {shape(model.layers[0].cache_v)}")
    print(f"    Total across all {L} layers: {model.cache_bytes()/1024:.1f} KB")

    print("\n  ── DECODE (generate 4 tokens) ───────────────────────────")
    with torch.no_grad():
        next_tok = torch.randint(0, VOCAB, (1, 1))
        for step in range(4):
            prev_cache = model.cache_tokens
            out_tok = model.decode_step(next_tok)

            Q2 = model.layers[0].Wq(model.embed(next_tok)).view(1,1,H,Hd).transpose(1,2)
            print(f"\n  Step {step+1}:")
            print(f"    New token input:  {shape(next_tok)}  (only 1 token!)")
            print(f"    Q (new token):    {shape(Q2)}")
            print(f"    Cache before:     {prev_cache} tokens")
            print(f"    Cache after:      {model.cache_tokens} tokens  (+1 appended)")
            print(f"    K_full attends to: all {model.cache_tokens} cached tokens")
            print(f"    cache size:       {model.cache_bytes()/1024:.1f} KB")
            next_tok = out_tok

    print(green("\n  Key insight: Q is always shape [1,H,1,Hd] during decode."))
    print(green("  Only 1 token computed. KV cache grows by 1 each step."))


# ═══════════════════════════════════════════════════════════
# DEMO 2 — Speed comparison
# ═══════════════════════════════════════════════════════════

def demo_speed():
    print("\n" + "═"*64)
    print(bold("  DEMO 2 (PyTorch): Speed — Naive vs KV Cache"))
    print("═"*64)

    PROMPT, GEN = 15, 25
    tokens = torch.randint(0, VOCAB, (1, PROMPT + GEN))

    # ── Naive: re-run prefill on full sequence every step ──
    model_nc = TinyLM()
    model_nc.eval()

    print(f"\n  Prompt={PROMPT} tokens | Generate={GEN} tokens\n")
    print(cyan("  [Naive — no cache]") + " re-prefills entire history each step")
    nc_times = []
    with torch.no_grad():
        for step in range(GEN):
            hist = tokens[:, :PROMPT+step]
            t0 = time.perf_counter()
            _ = model_nc.prefill(hist)
            model_nc.reset_cache()
            nc_times.append((time.perf_counter()-t0)*1000)
            if step == 0 or (step+1) % 8 == 0:
                t = nc_times[-1]
                print(f"    step {step+1:2d} (hist len={PROMPT+step:3d}): {t:.2f} ms  {bar(t, max(nc_times), 20)}")

    # ── With KV Cache ──
    model_kv = TinyLM()
    model_kv.eval()

    print(cyan("\n  [KV Cache]") + " prefill once, decode single tokens")
    kv_times = []
    with torch.no_grad():
        t0 = time.perf_counter()
        _ = model_kv.prefill(tokens[:, :PROMPT])
        prefill_ms = (time.perf_counter()-t0)*1000
        print(f"    prefill {PROMPT} tokens: {prefill_ms:.2f} ms  (fills KV cache)")

        next_tok = tokens[:, PROMPT:PROMPT+1]
        for step in range(GEN):
            t0 = time.perf_counter()
            next_tok = model_kv.decode_step(next_tok)
            kv_times.append((time.perf_counter()-t0)*1000)
            if step == 0 or (step+1) % 8 == 0:
                t = kv_times[-1]
                print(f"    step {step+1:2d} (cache={model_kv.cache_tokens:3d} tok):  {t:.2f} ms  {bar(t, max(nc_times), 20)}")

    avg_nc = sum(nc_times)/len(nc_times)
    avg_kv = sum(kv_times)/len(kv_times)
    sp     = avg_nc / avg_kv

    print(f"\n  ┌──────────────────┬──────────────┬─────────────────┐")
    print(f"  │ Method           │ Avg ms/token │ Total {GEN} tokens  │")
    print(f"  ├──────────────────┼──────────────┼─────────────────┤")
    print(f"  │ Naive            │ {avg_nc:>8.2f} ms │ {sum(nc_times):>11.1f} ms  │")
    print(f"  │ KV Cache         │ {avg_kv:>8.2f} ms │ {sum(kv_times):>11.1f} ms  │")
    print(f"  └──────────────────┴──────────────┴─────────────────┘")
    print(green(f"\n  KV Cache is {sp:.1f}x faster per decode token"))


# ═══════════════════════════════════════════════════════════
# DEMO 3 — GQA: MHA vs GQA vs MQA
# ═══════════════════════════════════════════════════════════

def demo_gqa():
    print("\n" + "═"*64)
    print(bold("  DEMO 3 (PyTorch): GQA — Fewer KV Heads, Less Cache"))
    print("═"*64)

    SEQ = 50
    tokens = torch.randint(0, VOCAB, (1, SEQ))

    configs = [
        ("MHA  (8 KV heads)", 8),
        ("GQA  (2 KV heads)", 2),
        ("MQA  (1 KV head)",  1),
    ]

    print(f"\n  Sequence: {SEQ} tokens | {L} layers\n")
    print(f"  {'Method':<22}  {'Cache (KB)':<12}  {'Decode 40 tok':<16}  {'Note'}")
    print(f"  {'──────':<22}  {'──────────':<12}  {'──────────────':<16}  {'────'}")

    baseline_b, baseline_t = None, None
    for label, kv_heads in configs:
        m = TinyLM(kv_heads=kv_heads)
        m.eval()
        with torch.no_grad():
            m.prefill(tokens[:, :10])
            t0 = time.perf_counter()
            tok = tokens[:, 10:11]
            for i in range(10, SEQ):
                tok = m.decode_step(tok)
            elapsed = (time.perf_counter()-t0)*1000
        cb = m.cache_bytes()/1024

        if baseline_b is None:
            baseline_b, baseline_t = cb, elapsed
            note = "baseline"
        else:
            note = green(f"{baseline_b/cb:.0f}x smaller cache, {baseline_t/elapsed:.1f}x faster")
        print(f"  {label:<22}  {cb:<12.1f}  {elapsed:<16.2f}  {note}")

    print(green("\n  GQA sweet spot: 4-8x smaller cache with near-identical quality"))


# ═══════════════════════════════════════════════════════════
# DEMO 4 — Prefix caching
# ═══════════════════════════════════════════════════════════

def demo_prefix():
    print("\n" + "═"*64)
    print(bold("  DEMO 4 (PyTorch): Prefix Caching"))
    print("═"*64)

    SYS  = 30
    QLEN = 5
    N    = 5

    sys_toks = torch.randint(0, VOCAB, (1, SYS))
    qs       = [torch.randint(0, VOCAB, (1, QLEN)) for _ in range(N)]

    model = TinyLM()
    model.eval()

    print(f"\n  System prompt: {SYS} tokens  |  User query: {QLEN} tokens  |  Requests: {N}\n")

    # Without prefix cache
    times_nc = []
    with torch.no_grad():
        for q in qs:
            model.reset_cache()
            full = torch.cat([sys_toks, q], dim=1)
            t0   = time.perf_counter()
            _    = model.prefill(full)
            times_nc.append((time.perf_counter()-t0)*1000)
            model.reset_cache()

    # With prefix cache — save cache after system prompt
    with torch.no_grad():
        model.reset_cache()
        _ = model.prefill(sys_toks)
        saved = [(l.cache_k.clone(), l.cache_v.clone()) for l in model.layers]

    times_c = []
    with torch.no_grad():
        for q in qs:
            for i, l in enumerate(model.layers):
                l.cache_k, l.cache_v = saved[i][0].clone(), saved[i][1].clone()
            t0 = time.perf_counter()
            _  = model.prefill(q)
            times_c.append((time.perf_counter()-t0)*1000)

    M = max(times_nc)
    print(f"  {'Req':<4}  {'No prefix cache':^26}  {'Prefix cached':^26}  Speedup")
    print(f"  {'───':<4}  {'───────────────':^26}  {'─────────────':^26}  ───────")
    for i in range(N):
        nc, c = times_nc[i], times_c[i]
        print(f"  {i+1:<4}  {nc:.3f} ms {bar(nc,M,16)}  "
              f"{c:.3f} ms {bar(c,M,16)}  {green(f'{nc/c:.1f}x')}")

    tnc, tc = sum(times_nc), sum(times_c)
    print(f"\n  Total: {tnc:.2f} ms  vs  {tc:.2f} ms")
    print(green(f"  Prefix caching saves {(1-tc/tnc)*100:.0f}% of prefill compute"))


# ═══════════════════════════════════════════════════════════
# DEMO 5 — Attention weight heatmap (real nn.MultiheadAttention)
# ═══════════════════════════════════════════════════════════

def demo_attn_map():
    print("\n" + "═"*64)
    print(bold("  DEMO 5 (PyTorch): Real Attention Weights Heatmap"))
    print("═"*64)

    words = ["The","cat","sat","on","the","mat"]
    SEQ   = len(words)

    mha = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
    mha.eval()

    x = torch.randn(1, SEQ, 64)
    with torch.no_grad():
        # need_weights=True returns average attention across heads
        _, attn_weights = mha(x, x, x, need_weights=True, average_attn_weights=True)

    W = attn_weights[0].numpy()   # [SEQ, SEQ]

    print(f"\n  nn.MultiheadAttention, 4 heads, {SEQ} tokens\n")
    print(f"  Attention weight matrix W[query, key]:\n")

    cw = 7
    print(" " * 10 + "".join(f"{w:>{cw}}" for w in words))
    print(" " * 10 + "─" * (cw * SEQ))

    shades = [" ", "░", "▒", "▓", "█"]
    for i, rw in enumerate(words):
        row = ""
        for j in range(SEQ):
            v = W[i, j]
            s = shades[min(int(v * len(shades) * 3), len(shades)-1)]
            row += f" {s}{v:.2f}"
        print(f"  {rw:>7} │{row}")

    print("\n  Row sums = 1.0 (softmax). Columns = how much each key is attended to.")
    row_sums = W.sum(axis=1)
    print(f"  Row sums: {[f'{s:.2f}' for s in row_sums]}")

    print("\n  Most-attended key per query:")
    for i, rw in enumerate(words):
        j = W[i].argmax()
        print(f"    {rw:>5} attends most to: {words[j]:<5} (weight={W[i,j]:.3f})")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

DEMOS = {
    "1": ("Tensor Shapes Step-by-Step",   demo_shapes),
    "2": ("Speed — Naive vs KV Cache",    demo_speed),
    "3": ("GQA — MHA vs GQA vs MQA",      demo_gqa),
    "4": ("Prefix Caching Speedup",        demo_prefix),
    "5": ("Real Attention Weight Heatmap", demo_attn_map),
}

if __name__ == "__main__":
    print()
    print(bold("╔══════════════════════════════════════════════════════╗"))
    print(bold("║       KV CACHE — PyTorch LIVE DEMO                   ║"))
    print(bold(f"║       torch {torch.__version__}  {'(CPU)':>30}║"))
    print(bold("╚══════════════════════════════════════════════════════╝"))

    chosen = sys.argv[1:] if len(sys.argv) > 1 else list(DEMOS.keys())
    torch.manual_seed(42)

    for key in chosen:
        if key in DEMOS:
            DEMOS[key][1]()
        else:
            print(f"  Unknown demo '{key}'. Choose 1-5.")

    print("\n" + "═"*64)
    print(bold("  Run one: python3 run_demo_torch.py 1"))
    print("═"*64 + "\n")
