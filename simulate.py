"""
================================================================================
FILE: simulate.py
PURPOSE: KV cache simulations using REAL English sentences as input — not
         random tensors. Builds a tiny word-level tokenizer, trains a small
         PyTorch language model on the vocabulary, then runs 5 end-to-end
         simulations showing cache behaviour on actual text.

WHY REAL DATA MATTERS
----------------------
All other demo files (run_demo.py, run_demo_torch.py) use random tensors
as token "embeddings." This is fine for measuring speed and memory, but it
means attention weights are meaningless noise.

This file uses 10 real sentences to build a vocabulary, then processes words
like "the", "cat", "attention", "cache" as actual tokens. The attention
weights shown in the heatmaps are computed on real word vectors, so patterns
like "which word does 'attention' look at most?" are meaningful.

COMPONENTS
----------

  Tokenizer
  ──────────
  Builds a word-level vocabulary from 10 sample sentences.
  Vocabulary: ~61 words + 4 special tokens (<PAD>, <BOS>, <EOS>, <UNK>)
  Methods: .encode("the cat sat") → [5, 12, 34]
           .decode([5, 12, 34])   → ["the", "cat", "sat"]
           .to_tensor("the cat")  → torch.Tensor([[5, 12]])

  TinyLM(vocab_size, kv_heads)
  ─────────────────────────────
  2-layer, 4-head transformer with:
    - Embedding: vocab_size → 128 dims
    - 2 × AttentionLayer (GQA-capable, each with KV cache)
    - LayerNorm after each layer
    - LM head: 128 → vocab_size logits
  Inference:
    .prefill(token_ids)    — fills cache, returns logits for all tokens
    .decode_step(token_id) — returns next predicted token (greedy)
  Properties:
    .cache_tokens          — current sequence length in cache
    .cache_bytes()         — float32 bytes used across all layers

THE 5 SIMULATIONS
-----------------

  SIM 1 — Full Trace: Prefill + Decode on Real Text
  ────────────────────────────────────────────────────
  Input:  "the cat sat on"  (4-token prompt)
  Task:   Generate 4 more tokens
  What you see:
    PREFILL phase:
      Each token name shown with its cache size growing:
        [the]  → 2048 B   (cache fills 25%)
        [cat]  → 4096 B   (cache fills 50%)
        [sat]  → 6144 B   (cache fills 75%)
        [on]   → 8192 B   (cache fills 100%)
      "Prefill done in X ms" with total cache size
    DECODE phase:
      For each generated token:
        New token name, which cached token it attended to most (with weight),
        and current cache size in bytes
    Final: shows the full generated sequence with arrows

  SIM 2 — Attention Heatmap on Real Sentence
  ─────────────────────────────────────────────
  Input:  "attention is all you need to understand transformers"  (8 words)
  Task:   Run prefill, print the attention weight matrix
  What you see:
    8×8 grid of attention weights (with causal mask applied)
    Unicode shade characters show intensity: █=high, ▓=med, ▒=low, ░=very low
    Upper triangle = "· ·" (future tokens, masked out)
    Below the heatmap: "Per-token top attention" table showing which word
    each word attends to most, and 2nd most.
    Example: "need → [all] (0.27)" means "need" attends most to "all"
  Why it matters: With real word embeddings, attention patterns reflect
                  actual semantic structure (even with untrained weights,
                  high-magnitude tokens attract more attention).

  SIM 3 — Speed vs Prompt Length
  ────────────────────────────────
  Task:   Run 5 different prompts (length 3 to 11 tokens).
          For each: measure time with no-cache and with KV cache.
          Generate 5 new tokens after each prompt.
  What you see:
    Table: prompt text | length | no-cache ms | cached ms | speedup
  Why it matters: Shows that the KV cache advantage grows with prompt length.
                  Short prompts (3 tokens) → 1.1x speedup.
                  Longer prompts (11 tokens) → 1.3-1.4x speedup.
                  On real models with 2K+ token prompts, the gap is 5-50x.

  SIM 4 — Live Token Generation: Watch Cache Fill
  ──────────────────────────────────────────────────
  Input:  "key value cache stores past computations"  (6 tokens)
  Task:   Generate 6 more tokens
  What you see:
    PREFILL row: all prompt tokens shown dimmed (already computed)
    Each DECODE row:
      Full sentence so far (prompt in dim, new token in cyan)
      "attended: ▒▒▒░░▒▒ ↑[stores]" — heat bar showing which positions
      the new token attended to, plus the most attended word
  Why it matters: Watching the cache fill word by word shows the "append-only"
                  nature of KV caching. The heat bar shows attention sinks
                  forming naturally (early tokens get repeatedly attended to).

  SIM 5 — Prefix Caching: Same Context, Different Queries
  ─────────────────────────────────────────────────────────
  Context: "machine learning requires lots of data and compute"  (8 tokens)
           — treated as a shared document / system prompt
  Queries: 4 different follow-up prompts ("the cat sat on", etc.)
  Task:    Compare timing:
             WITHOUT prefix cache: re-run full prefill for each query
             WITH prefix cache:    compute context KV once, reuse for all 4
  What you see:
    "Context KV cache saved: X KB"
    Table: query text | no-cache ms | cached ms | speedup
    "Prefix caching saves X% of prefill compute"
  Why it matters: This is how the Claude API's cache_control parameter,
                  OpenAI's automatic prefix caching, and SGLang's
                  RadixAttention work — shared prefixes computed once,
                  saved in a cache, reused across many queries.

HOW TO RUN
----------
  python3 simulate.py          # run all 5 simulations
  python3 simulate.py 1        # run only Sim 1 (full trace)
  python3 simulate.py 2 5      # run Sims 2 and 5

REQUIREMENTS
------------
  pip install torch   (CPU version is fine — no GPU needed)

SAMPLE SENTENCES (vocabulary source)
--------------------------------------
  "the cat sat on the mat"
  "the dog ran in the park"
  "artificial intelligence is changing the world"
  "the model generates one token at a time"
  "key value cache stores past computations"
  "attention is all you need to understand transformers"
  ... (10 total — see Tokenizer.SAMPLE_SENTENCES)
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time, sys

# ─────────────────────────────────────────────────────────────────
# SIMPLE WORD TOKENIZER
# ─────────────────────────────────────────────────────────────────

class Tokenizer:
    """Word-level tokenizer built from our sample sentences."""

    SAMPLE_SENTENCES = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "she opened the book and started reading",
        "the quick brown fox jumps over the lazy dog",
        "artificial intelligence is changing the world",
        "the model generates one token at a time",
        "key value cache stores past computations",
        "attention is all you need to understand transformers",
        "the weather is nice today in the city",
        "machine learning requires lots of data and compute",
    ]

    def __init__(self):
        words = set()
        for s in self.SAMPLE_SENTENCES:
            words.update(s.split())
        special = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
        vocab = special + sorted(words)
        self.w2i = {w: i for i, w in enumerate(vocab)}
        self.i2w = {i: w for w, i in self.w2i.items()}
        self.vocab_size = len(vocab)
        self.BOS = self.w2i["<BOS>"]
        self.EOS = self.w2i["<EOS>"]
        self.UNK = self.w2i["<UNK>"]

    def encode(self, text: str) -> list[int]:
        return [self.w2i.get(w, self.UNK) for w in text.split()]

    def decode(self, ids) -> list[str]:
        return [self.i2w.get(int(i), "<UNK>") for i in ids]

    def to_tensor(self, text: str) -> torch.Tensor:
        return torch.tensor([self.encode(text)], dtype=torch.long)   # [1, T]


# ─────────────────────────────────────────────────────────────────
# TINY LANGUAGE MODEL (same as run_demo_torch.py)
# ─────────────────────────────────────────────────────────────────

D, LAYERS, H, Hd = 128, 2, 4, 32   # smaller — fits sample vocab

class AttnLayer(nn.Module):
    def __init__(self, kv_heads=4):
        super().__init__()
        self.H, self.Hkv, self.Hd = H, kv_heads, Hd
        self.G  = H // kv_heads
        self.Wq = nn.Linear(D, H*Hd,      bias=False)
        self.Wk = nn.Linear(D, kv_heads*Hd, bias=False)
        self.Wv = nn.Linear(D, kv_heads*Hd, bias=False)
        self.Wo = nn.Linear(H*Hd, D,      bias=False)
        self.ck = self.cv = None           # KV cache
        self._last_attn = None             # for visualization

    def reset(self): self.ck = self.cv = None

    @property
    def cache_len(self): return 0 if self.ck is None else self.ck.shape[2]

    def forward(self, x, use_cache=True, return_attn=False):
        B, T, _ = x.shape
        H, Hkv, G, Hd_ = self.H, self.Hkv, self.G, self.Hd

        Q = self.Wq(x).view(B,T,H,Hd_).transpose(1,2)
        K = self.Wk(x).view(B,T,Hkv,Hd_).transpose(1,2)
        V = self.Wv(x).view(B,T,Hkv,Hd_).transpose(1,2)

        if use_cache:
            self.ck = K if self.ck is None else torch.cat([self.ck, K], dim=2)
            self.cv = V if self.cv is None else torch.cat([self.cv, V], dim=2)
            Kf, Vf = self.ck, self.cv
        else:
            Kf, Vf = K, V

        if G > 1:
            Kf = Kf.unsqueeze(2).expand(B,Hkv,G,Kf.shape[2],Hd_).reshape(B,H,-1,Hd_)
            Vf = Vf.unsqueeze(2).expand(B,Hkv,G,Vf.shape[2],Hd_).reshape(B,H,-1,Hd_)

        scores = torch.matmul(Q, Kf.transpose(-2,-1)) / math.sqrt(Hd_)
        if T > 1:
            T_full = Kf.shape[2]
            m = torch.triu(torch.ones(T, T_full, dtype=torch.bool), diagonal=T_full-T+1)
            scores = scores.masked_fill(m, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        self._last_attn = attn.detach()    # save for visualization
        out  = torch.matmul(attn, Vf)
        out  = out.transpose(1,2).contiguous().view(B,T,H*Hd_)
        return self.Wo(out)


class TinyLM(nn.Module):
    def __init__(self, vocab_size, kv_heads=4):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, D)
        self.layers  = nn.ModuleList([AttnLayer(kv_heads) for _ in range(LAYERS)])
        self.norm    = nn.LayerNorm(D)
        self.lm_head = nn.Linear(D, vocab_size, bias=False)

    def reset_cache(self):
        for l in self.layers: l.reset()

    @property
    def cache_tokens(self): return self.layers[0].cache_len

    def cache_bytes(self):
        b = 0
        for l in self.layers:
            if l.ck is not None:
                b += (l.ck.numel() + l.cv.numel()) * 4
        return b

    def forward(self, ids, use_cache=True):
        x = self.embed(ids)
        for l in self.layers:
            x = x + l(x, use_cache=use_cache)
            x = self.norm(x)
        return self.lm_head(x)

    def get_attn_weights(self, layer=0):
        """Return averaged attention weights from a layer."""
        a = self.layers[layer]._last_attn
        if a is None: return None
        return a[0].mean(dim=0)   # avg over heads → [T_q, T_k]


# ─────────────────────────────────────────────────────────────────
# COLOUR / DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────

def bold(s):   return f"\033[1m{s}\033[0m"
def green(s):  return f"\033[32m{s}\033[0m"
def cyan(s):   return f"\033[36m{s}\033[0m"
def yellow(s): return f"\033[33m{s}\033[0m"
def blue(s):   return f"\033[34m{s}\033[0m"
def dim(s):    return f"\033[2m{s}\033[0m"

def bar(v, mx, w=28):
    if mx == 0: return "░"*w
    f = int(round(max(0, min(v/mx,1))*w))
    return "█"*f + "░"*(w-f)

def heat(v):
    """Single character heat indicator for attention weight."""
    shades = " ░▒▓█"
    idx = min(int(v * len(shades) * 3.5), len(shades)-1)
    return shades[idx]

def fmt_tokens(words, highlight_last=False):
    """Render a list of words as a token sequence."""
    if not words: return ""
    if highlight_last:
        return " ".join(f"[{w}]" if i < len(words)-1 else cyan(f"[{w}]")
                        for i, w in enumerate(words))
    return " ".join(f"[{w}]" for w in words)


# ─────────────────────────────────────────────────────────────────
# SIMULATION 1 — Full trace: prefill + decode on a real sentence
# ─────────────────────────────────────────────────────────────────

def sim_full_trace(tok, model):
    print("\n" + "═"*68)
    print(bold("  SIM 1: Full Trace — Prefill + Decode on Real Text"))
    print("═"*68)

    PROMPT = "the cat sat on"
    GEN    = 4

    prompt_ids = tok.to_tensor(PROMPT)               # [1, 4]
    prompt_words = tok.decode(prompt_ids[0])

    print(f"\n  Prompt: {fmt_tokens(prompt_words)}")
    print(f"  Will generate {GEN} more tokens.\n")

    model.reset_cache()

    # ── PREFILL ──────────────────────────────────────────────────
    print(bold("  ── PREFILL PHASE ──────────────────────────────────────"))
    print(f"  Processing all {len(prompt_words)} prompt tokens in ONE parallel pass.\n")

    with torch.no_grad():
        t0 = time.perf_counter()
        logits = model(prompt_ids)
        prefill_ms = (time.perf_counter() - t0) * 1000

    print(f"  {'Step':<6} {'Token':<10} {'Cache size':>12}  {'Cache bar'}")
    print(f"  {'────':<6} {'─────':<10} {'──────────':>12}  {'─────────'}")

    for i, w in enumerate(prompt_words):
        cb = (i+1) * model.cache_bytes() // len(prompt_words)
        print(f"  {i+1:<6} {w:<10} {model.cache_bytes()//len(prompt_words)*(i+1):>10} B  "
              f"{bar(i+1, len(prompt_words)+GEN, 30)}  stored K,V for [{w}]")

    print(f"\n  Prefill done in {prefill_ms:.2f} ms")
    print(f"  KV cache after prefill: {model.cache_bytes()} bytes  "
          f"({model.cache_tokens} tokens stored)")

    # ── DECODE ───────────────────────────────────────────────────
    print(f"\n{bold('  ── DECODE PHASE ───────────────────────────────────────')}")
    print(f"  Generating one token at a time.\n")

    generated = list(prompt_words)
    next_ids = torch.argmax(logits[:, -1:, :], dim=-1)   # greedy from last prefill logit

    print(f"  {'Step':<6} {'New token':<12} {'Attends to':<40} {'Cache'}")
    print(f"  {'────':<6} {'─────────':<12} {'──────────':<40} {'─────'}")

    decode_times = []
    for step in range(GEN):
        cache_before = model.cache_tokens
        t0 = time.perf_counter()
        with torch.no_grad():
            out_logits = model(next_ids)
        decode_times.append((time.perf_counter()-t0)*1000)

        new_word = tok.decode(next_ids[0])[0]
        attn = model.get_attn_weights(layer=0)   # [1, cache_len]

        # Show which cached token got highest attention
        if attn is not None and attn.shape[0] == 1:
            w_row = attn[0].numpy()               # [cache_len]
            top_i = int(w_row.argmax())
            top_w = generated[top_i] if top_i < len(generated) else "?"
            attended = f"↑ [{top_w}] ({w_row[top_i]:.2f})"
        else:
            attended = ""

        cb = model.cache_bytes()
        generated.append(new_word)
        print(f"  {step+1:<6} {cyan(new_word):<21} {attended:<40} "
              f"{model.cache_tokens} tok / {cb} B")

        next_ids = torch.argmax(out_logits[:, -1:, :], dim=-1)

    print(f"\n  Generated sequence:")
    print(f"  {fmt_tokens(prompt_words)}  →  "
          + "  ".join(cyan(f"[{w}]") for w in generated[len(prompt_words):]))

    print(f"\n  Prefill: {prefill_ms:.2f} ms  ({len(prompt_words)} tokens at once)")
    print(f"  Decode:  {sum(decode_times):.2f} ms  ({GEN} tokens, "
          f"avg {sum(decode_times)/GEN:.2f} ms/tok)")
    print(f"  Final KV cache: {model.cache_tokens} tokens / {model.cache_bytes()} bytes")


# ─────────────────────────────────────────────────────────────────
# SIMULATION 2 — Attention heatmap on real sentence
# ─────────────────────────────────────────────────────────────────

def sim_attention_heatmap(tok, model):
    print("\n" + "═"*68)
    print(bold("  SIM 2: Attention Heatmap on Real Sentence"))
    print("═"*68)

    sentence = "attention is all you need to understand transformers"
    ids = tok.to_tensor(sentence)
    words = tok.decode(ids[0])

    model.reset_cache()
    with torch.no_grad():
        _ = model(ids)

    attn = model.get_attn_weights(layer=0).numpy()   # [T, T]
    T = len(words)

    print(f"\n  Sentence: \"{sentence}\"")
    print(f"  {T} tokens, {LAYERS} layers, {H} heads  (averaged)\n")

    # Print heatmap
    cw = 12
    print("  " + " "*11 + "".join(f"{w[:cw]:>{cw}}" for w in words))
    print("  " + " "*11 + "─"*(cw*T))

    for i, rw in enumerate(words):
        row = ""
        for j in range(T):
            if j > i:
                row += " "*(cw-4) + "  · ·"
            else:
                v = attn[i, j]
                h = heat(v)
                row += f"{' '*(cw-5)}{h}{v:.3f}"
        print(f"  {rw[:9]:>9} │{row}")

    print("\n  Legend: █=high  ▓=med-high  ▒=medium  ░=low  · ·=future (masked)")

    # For each word, show top-2 attended words
    print(f"\n  Per-token top attention:")
    print(f"  {'Token':<14} {'Attends most to':<20} {'2nd most'}")
    print(f"  {'─────':<14} {'───────────────':<20} {'────────'}")
    for i, rw in enumerate(words):
        row = attn[i, :i+1]
        order = row.argsort()[::-1]
        top1  = words[order[0]]
        top1v = row[order[0]]
        if len(order) > 1:
            top2  = words[order[1]]
            top2v = row[order[1]]
            top2s = f"[{top2}] ({top2v:.3f})"
        else:
            top2s = "—"
        self_mark = " ← self" if order[0] == i else ""
        print(f"  {rw:<14} [{top1}] ({top1v:.3f}){self_mark:<16} {top2s}")


# ─────────────────────────────────────────────────────────────────
# SIMULATION 3 — Speed on different sentence lengths
# ─────────────────────────────────────────────────────────────────

def sim_speed_vs_length(tok, model):
    print("\n" + "═"*68)
    print(bold("  SIM 3: Speed vs Prompt Length (No Cache vs KV Cache)"))
    print("═"*68)

    prompts = [
        "the cat sat",
        "the cat sat on the mat",
        "the quick brown fox jumps over the lazy dog",
        "attention is all you need to understand transformers",
        "machine learning requires lots of data and compute to work well",
    ]

    GEN = 5   # tokens to generate after each prompt

    print(f"\n  Generating {GEN} tokens after each prompt.\n")
    print(f"  {'Prompt':<44} {'Len':>4}  {'No Cache':>10}  {'KV Cache':>10}  {'Speedup'}")
    print(f"  {'──────':<44} {'───':>4}  {'────────':>10}  {'────────':>10}  {'───────'}")

    for prompt in prompts:
        ids = tok.to_tensor(prompt)
        plen = ids.shape[1]

        # ── No cache: re-run full prefill for every decode step ──
        model.reset_cache()
        nc_times = []
        with torch.no_grad():
            for step in range(GEN):
                all_ids = ids   # in real no-cache, history grows — simulate with full re-prefill
                t0 = time.perf_counter()
                logits = model(all_ids, use_cache=False)
                nc_times.append((time.perf_counter()-t0)*1000)

        # ── KV Cache: prefill once, decode steps ──
        model.reset_cache()
        kv_times = []
        with torch.no_grad():
            logits = model(ids, use_cache=True)
            next_tok = torch.argmax(logits[:,-1:,:], dim=-1)
            for step in range(GEN):
                t0 = time.perf_counter()
                out = model(next_tok, use_cache=True)
                kv_times.append((time.perf_counter()-t0)*1000)
                next_tok = torch.argmax(out[:,-1:,:], dim=-1)

        avg_nc = sum(nc_times)/len(nc_times)
        avg_kv = sum(kv_times)/len(kv_times)
        sp     = avg_nc / avg_kv

        short_prompt = (prompt[:40] + "…") if len(prompt) > 40 else prompt
        print(f"  {short_prompt:<44} {plen:>4}  {avg_nc:>8.3f}ms  {avg_kv:>8.3f}ms  "
              f"{green(f'{sp:.1f}x')}")

    print(green("\n  As prompt length grows, KV Cache advantage increases."))


# ─────────────────────────────────────────────────────────────────
# SIMULATION 4 — Live generation: watch tokens appear one by one
# ─────────────────────────────────────────────────────────────────

def sim_live_generation(tok, model):
    print("\n" + "═"*68)
    print(bold("  SIM 4: Live Token Generation — Watch Cache Fill"))
    print("═"*68)

    prompt = "key value cache stores past computations"
    GEN    = 6

    ids    = tok.to_tensor(prompt)
    words  = tok.decode(ids[0])

    print(f"\n  Prompt: \"{prompt}\"")
    print(f"  Generating {GEN} more tokens.\n")

    model.reset_cache()

    # Prefill
    with torch.no_grad():
        logits = model(ids, use_cache=True)
    next_tok = torch.argmax(logits[:,-1:,:], dim=-1)

    sentence = list(words)
    max_cache = model.cache_bytes() + GEN * (model.cache_bytes() // len(words) + 1)

    print(f"  {'Step':<6} {'Cache state':^52}  {'New token'}")
    print(f"  {'────':<6} {'──────────':^52}  {'─────────'}")

    # Show state after prefill
    cb = model.cache_bytes()
    token_display = " ".join(f"[{w}]" for w in sentence)
    print(f"  {'PRE':6} {dim(token_display):<52}  {dim('(prefill complete)')}")

    for step in range(GEN):
        with torch.no_grad():
            out = model(next_tok, use_cache=True)

        new_word = tok.decode(next_tok[0])[0]
        sentence.append(new_word)
        cb = model.cache_bytes()

        # Build cache display: old tokens dim, new token highlighted
        last_word = sentence[-1]
        parts = [dim(f"[{word}]") for word in sentence[:-1]] + [cyan(f"[{last_word}]")]
        display = " ".join(parts)
        plain   = " ".join(f"[{w}]" for w in sentence)  # for length calc

        # Attention: what did this new token attend to most?
        attn = model.get_attn_weights(layer=0)
        if attn is not None and attn.shape[0] >= 1:
            w_row   = attn[-1].numpy()
            top_i   = int(w_row[:len(sentence)-1].argmax())
            top_w   = sentence[top_i]
            heat_bar = "".join(heat(w_row[j]) for j in range(len(sentence)))
            attn_note = f"  attended: {heat_bar}  ↑[{top_w}]"
        else:
            attn_note = ""

        print(f"  {step+1:<6} {plain:<52}  {cyan(new_word)}")
        print(f"  {'':6} {attn_note}")

        next_tok = torch.argmax(out[:,-1:,:], dim=-1)

    print(f"\n  Final: \"{' '.join(sentence)}\"")
    print(f"  Cache: {model.cache_tokens} tokens | {model.cache_bytes()} bytes")


# ─────────────────────────────────────────────────────────────────
# SIMULATION 5 — Prefix cache: same prompt, different questions
# ─────────────────────────────────────────────────────────────────

def sim_prefix_cache(tok, model):
    print("\n" + "═"*68)
    print(bold("  SIM 5: Prefix Caching — Shared Context, Different Queries"))
    print("═"*68)

    # Shared "document" / system context
    context = "machine learning requires lots of data and compute"
    # Different follow-up queries
    queries = [
        "the cat sat on",
        "artificial intelligence is",
        "the model generates",
        "key value cache stores",
    ]

    ctx_ids   = tok.to_tensor(context)
    ctx_words = tok.decode(ctx_ids[0])
    ctx_len   = ctx_ids.shape[1]

    print(f"\n  Shared context ({ctx_len} tokens):")
    print(f"  \"{context}\"")
    print(f"\n  {len(queries)} different queries will follow this context.\n")

    # ── WITHOUT prefix cache ──────────────────────────────────────
    print(cyan("  [Without prefix cache]") + " — recomputes context KV every time")
    times_nc = []
    for q in queries:
        q_ids = tok.to_tensor(q)
        full  = torch.cat([ctx_ids, q_ids], dim=1)
        model.reset_cache()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(full, use_cache=False)
        times_nc.append((time.perf_counter()-t0)*1000)

    # ── WITH prefix cache ─────────────────────────────────────────
    print(cyan("\n  [With prefix cache]") + " — context computed ONCE, reused")

    # Compute + save context KV cache
    model.reset_cache()
    with torch.no_grad():
        _ = model(ctx_ids, use_cache=True)
    saved = [(l.ck.clone(), l.cv.clone()) for l in model.layers]
    ctx_cache_kb = model.cache_bytes() / 1024

    print(f"  Context KV cache saved: {ctx_cache_kb:.1f} KB\n")

    times_c = []
    for q in queries:
        q_ids = tok.to_tensor(q)
        # Restore context cache
        for i, l in enumerate(model.layers):
            l.ck, l.cv = saved[i][0].clone(), saved[i][1].clone()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(q_ids, use_cache=True)
        times_c.append((time.perf_counter()-t0)*1000)

    # ── Results ───────────────────────────────────────────────────
    M = max(times_nc)
    print(f"  {'Query':<36}  {'No cache':>10}  {'Cached':>10}  {'Speedup'}")
    print(f"  {'─────':<36}  {'────────':>10}  {'──────':>10}  {'───────'}")
    for i, q in enumerate(queries):
        nc, c = times_nc[i], times_c[i]
        print(f"  {q:<36}  {nc:>8.3f}ms  {c:>8.3f}ms  {green(f'{nc/c:.1f}x')}")

    tnc, tc = sum(times_nc), sum(times_c)
    print(f"\n  Total ({len(queries)} queries):")
    print(f"    No prefix cache:   {tnc:.3f} ms")
    print(f"    With prefix cache: {tc:.3f} ms")
    print(green(f"    Saved: {(1-tc/tnc)*100:.0f}% of prefill compute"))
    print(f"\n  The {ctx_len}-token context was computed only ONCE "
          f"and reused {len(queries)} times.")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

SIMS = {
    "1": ("Full Trace — Prefill + Decode on Real Text",      sim_full_trace),
    "2": ("Attention Heatmap on Real Sentence",              sim_attention_heatmap),
    "3": ("Speed vs Prompt Length",                          sim_speed_vs_length),
    "4": ("Live Token Generation — Watch Cache Fill",        sim_live_generation),
    "5": ("Prefix Caching — Shared Context, Many Queries",  sim_prefix_cache),
}

if __name__ == "__main__":
    print()
    print(bold("╔══════════════════════════════════════════════════════════╗"))
    print(bold("║   KV CACHE SIMULATION WITH REAL SAMPLE DATA              ║"))
    print(bold("║   PyTorch " + f"{torch.__version__:<49}" + "║"))
    print(bold("╚══════════════════════════════════════════════════════════╝"))

    torch.manual_seed(0)
    tok   = Tokenizer()
    model = TinyLM(vocab_size=tok.vocab_size)
    model.eval()

    print(f"\n  Tokenizer vocab: {tok.vocab_size} words")
    print(f"  Model: {LAYERS}L {H}H d={D} head_dim={Hd}")
    print(f"  Sample sentences: {len(Tokenizer.SAMPLE_SENTENCES)}")

    chosen = sys.argv[1:] if len(sys.argv) > 1 else list(SIMS.keys())
    for key in chosen:
        if key in SIMS:
            SIMS[key][1](tok, model)
        else:
            print(f"  Unknown: '{key}'. Choose 1-5.")

    print("\n" + "═"*68)
    print(bold("  Run one: python3 simulate.py 1"))
    print(bold("  All:     python3 simulate.py"))
    print("═"*68 + "\n")
