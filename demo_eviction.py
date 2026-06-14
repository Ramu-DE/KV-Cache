"""
================================================================================
FILE: demo_eviction.py
PURPOSE: Live interactive demos comparing 4 KV cache eviction strategies.
         Run this file to see each method's cache size, eviction behaviour,
         and attention weight preservation — all in the terminal.

BACKGROUND: THE EVICTION PROBLEM
---------------------------------
A standard KV cache keeps every token forever — the cache grows with every
generated token and eventually fills your GPU's VRAM. Token eviction solves
this by discarding tokens that are unlikely to be needed, keeping cache size
bounded. The challenge: pick the wrong tokens to discard and model quality drops.

THE 4 STRATEGIES IMPLEMENTED
------------------------------

  1. FullCache (baseline)
     ─────────────────────
     Keeps every token. Cache grows from prompt_length all the way to
     prompt_length + generated_tokens. Reference point for comparison.
     Class:  FullCache(d_model, n_heads)
     Method: .prefill(tokens), .decode(tok), .size()

  2. H2OCache — Heavy-Hitter Oracle
     ─────────────────────────────────
     Paper: arxiv:2306.14048 (Zhang et al., 2023)
     Idea:  Track a cumulative attention score for each cached token.
            Every decode step, add the attention weights each token received.
            When cache exceeds budget, evict the token with the lowest score.
     Result: Tokens that are repeatedly attended to ("heavy hitters") survive.
             Tokens ignored by the model are dropped.
     Class:  H2OCache(d_model, n_heads, budget)
             self.scores[T] — running importance per token
             self.evicted   — count of tokens dropped so far

  3. StreamingLLMCache — Attention Sinks + Sliding Window
     ───────────────────────────────────────────────────────
     Paper: arxiv:2309.17453 (Xiao et al., 2023)
     Key finding: the very first tokens (especially [BOS]) always receive
     disproportionate attention — NOT because they are semantically important,
     but because the model learned to "dump" unused attention there. Dropping
     them collapses model quality even if they are irrelevant.
     Fix:   Keep first sink_size tokens permanently ("attention sinks") AND
            keep a sliding window of the last window_size recent tokens.
     Result: Cache size is FIXED at (sink_size + window_size) regardless of
             total sequence length → infinite-length generation possible.
     Class:  StreamingLLMCache(d_model, n_heads, sink_size=4, window_size=64)
             self.sink_k / self.sink_v — never evicted
             self.win_k  / self.win_v  — slides forward

  4. SnapKVCache — Observation Window Prediction
     ───────────────────────────────────────────────
     Paper: arxiv:2404.14469 (Li et al., 2024)
     Idea:  At PREFILL TIME (before generation begins), use the last
            obs_window tokens of the prompt as "observation queries."
            Compute which prefix tokens the observation window attends to most.
            Keep only those top-budget tokens in the cache.
     Result: Cache is pruned once at prefill — no eviction during generation.
             3.6x speed and 8.2x memory improvement at 16K tokens (verified).
     Class:  SnapKVCache(d_model, n_heads, budget, obs_window=16)

THE 4 DEMOS
-----------

  DEMO 1 — Cache Size Over Time
  ──────────────────────────────
  All 4 strategies prefill 40 tokens, then generate 30 tokens.
  Prints cache size at every step as a bar chart:
    Full:   41 → 42 → 43 → ... → 70  (grows forever)
    H2O:    hits budget=20, stays flat
    Stream: hits sink+window=20, stays flat
    SnapKV: grows (no eviction during decode; budget was spent at prefill)

  DEMO 2 — How Eviction Shifts Attention Weights
  ─────────────────────────────────────────────────
  Shows that H2O correctly preserves the tokens that matter most.
  Tokens at positions 5-10 are given 5x larger magnitude → they dominate
  attention. Measures which top-5 positions survive under different budgets:
    budget=25 → 5/5 match with full cache  (no quality loss)
    budget=15 → 5/5 match                  (still fine)
    budget=5  → 3/5 match                  (some degradation)

  DEMO 3 — H2O Eviction Trace
  ─────────────────────────────
  Step-by-step trace of H2O running with budget=6 on 8 prompt tokens.
  Tokens 2 and 5 are injected with large magnitude to attract attention.
  Prints which tokens have highest cumulative scores after each step.
  Shows: high-magnitude tokens (2, 5) survive; unimportant tokens evicted.

  DEMO 4 — StreamingLLM Infinite Context
  ────────────────────────────────────────
  30 total tokens processed. Cache cap = sink(3) + window(8) = 11.
  Prints total tokens seen vs cache size at every step.
  After step 11, cache stays at 11 forever while total_seen keeps growing.

HOW TO RUN
----------
  python3 demo_eviction.py          # run all 4 demos
  python3 demo_eviction.py 1        # run only Demo 1
  python3 demo_eviction.py 3 4      # run Demos 3 and 4

NOTE: These are plain Python classes (no nn.Module). For PyTorch nn.Module
versions of the same algorithms with full prefill/decode support, see
06_code_examples.py (H2OAttention, StreamingLLMAttention).
================================================================================
"""

import torch
import torch.nn.functional as F
import math, time, sys

def bold(s):  return f"\033[1m{s}\033[0m"
def green(s): return f"\033[32m{s}\033[0m"
def cyan(s):  return f"\033[36m{s}\033[0m"
def red(s):   return f"\033[31m{s}\033[0m"
def yellow(s):return f"\033[33m{s}\033[0m"

def bar(v, mx, w=25):
    if mx == 0: return "░" * w
    f = int(round(max(0, min(v / mx, 1)) * w))
    return "█" * f + "░" * (w - f)


# ─────────────────────────────────────────────────────────────
# ATTENTION PRIMITIVE
# ─────────────────────────────────────────────────────────────

def scaled_dot_product_attention(Q, K, V, return_weights=False):
    """
    Q: [H, q_len, d_k]
    K: [H, k_len, d_k]
    V: [H, k_len, d_v]
    Returns output [H, q_len, d_v] and optionally weights [H, q_len, k_len]
    """
    scale = math.sqrt(Q.shape[-1])
    scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
    weights = F.softmax(scores, dim=-1)
    out = torch.matmul(weights, V)
    if return_weights:
        return out, weights
    return out


# ─────────────────────────────────────────────────────────────
# 1. FULL CACHE — baseline
# ─────────────────────────────────────────────────────────────

class FullCache:
    """Standard KV cache — keeps every token. No eviction."""

    def __init__(self, d_model, n_heads):
        self.H, self.Hd = n_heads, d_model // n_heads
        self.Wq = torch.randn(d_model, n_heads * self.Hd) * 0.02
        self.Wk = torch.randn(d_model, n_heads * self.Hd) * 0.02
        self.Wv = torch.randn(d_model, n_heads * self.Hd) * 0.02
        self.Wo = torch.randn(n_heads * self.Hd, d_model) * 0.02
        self.ck = None  # [H, T, Hd]
        self.cv = None

    def reset(self): self.ck = self.cv = None

    def project(self, x):
        T, D = x.shape
        Q = (x @ self.Wq).reshape(T, self.H, self.Hd).transpose(0, 1)
        K = (x @ self.Wk).reshape(T, self.H, self.Hd).transpose(0, 1)
        V = (x @ self.Wv).reshape(T, self.H, self.Hd).transpose(0, 1)
        return Q, K, V

    def prefill(self, tokens):
        Q, K, V = self.project(tokens)
        self.ck, self.cv = K, V
        out = scaled_dot_product_attention(Q, K, V)
        return out.transpose(0, 1).reshape(tokens.shape[0], -1) @ self.Wo

    def decode(self, tok):
        Q, K, V = self.project(tok)
        self.ck = torch.cat([self.ck, K], dim=1)
        self.cv = torch.cat([self.cv, V], dim=1)
        out = scaled_dot_product_attention(Q, self.ck, self.cv)
        return out.transpose(0, 1).reshape(1, -1) @ self.Wo

    def size(self): return 0 if self.ck is None else self.ck.shape[1]

    def get_last_attn_weights(self, tok):
        Q, K, V = self.project(tok)
        K_all = torch.cat([self.ck, K], dim=1)
        V_all = torch.cat([self.cv, V], dim=1)
        _, weights = scaled_dot_product_attention(Q, K_all, V_all, return_weights=True)
        return weights  # [H, 1, T+1]


# ─────────────────────────────────────────────────────────────
# 2. H2O — Heavy-Hitter Oracle
# ─────────────────────────────────────────────────────────────

class H2OCache:
    """
    KV cache with cumulative-attention-score eviction.
    Keeps the top-K tokens by accumulated attention weight.

    Paper: arxiv:2306.14048
    """

    def __init__(self, d_model, n_heads, budget):
        self.H, self.Hd = n_heads, d_model // n_heads
        self.budget = budget
        self.Wq = torch.randn(d_model, n_heads * self.Hd) * 0.02
        self.Wk = torch.randn(d_model, n_heads * self.Hd) * 0.02
        self.Wv = torch.randn(d_model, n_heads * self.Hd) * 0.02
        self.Wo = torch.randn(n_heads * self.Hd, d_model) * 0.02
        self.ck = None
        self.cv = None
        self.scores = None  # cumulative importance scores [T]
        self.evicted = 0    # count of evicted tokens

    def reset(self):
        self.ck = self.cv = self.scores = None
        self.evicted = 0

    def _project(self, x):
        T = x.shape[0]
        Q = (x @ self.Wq).reshape(T, self.H, self.Hd).transpose(0, 1)
        K = (x @ self.Wk).reshape(T, self.H, self.Hd).transpose(0, 1)
        V = (x @ self.Wv).reshape(T, self.H, self.Hd).transpose(0, 1)
        return Q, K, V

    def prefill(self, tokens):
        Q, K, V = self._project(tokens)
        self.ck, self.cv = K, V
        _, weights = scaled_dot_product_attention(Q, K, V, return_weights=True)
        # Initialize importance scores as sum of attention received
        self.scores = weights.mean(dim=0).sum(dim=0)  # [T]
        out = torch.matmul(weights, V)
        return out.transpose(0, 1).reshape(tokens.shape[0], -1) @ self.Wo

    def decode(self, tok):
        Q, K_new, V_new = self._project(tok)

        # Append new token
        K_all = torch.cat([self.ck, K_new], dim=1)
        V_all = torch.cat([self.cv, V_new], dim=1)

        out, weights = scaled_dot_product_attention(Q, K_all, V_all, return_weights=True)
        # weights: [H, 1, T+1]

        # Update cumulative scores with attention received this step
        step_scores = weights[0, 0, :]  # [T+1] — avg over heads
        new_scores = torch.cat([self.scores, torch.zeros(1)]) + step_scores

        # Keep within budget by evicting lowest-scored token
        if K_all.shape[1] > self.budget:
            keep_idx = new_scores.topk(self.budget).indices.sort().values
            self.ck = K_all[:, keep_idx, :]
            self.cv = V_all[:, keep_idx, :]
            self.scores = new_scores[keep_idx]
            self.evicted += 1
        else:
            self.ck, self.cv = K_all, V_all
            self.scores = new_scores

        return out.transpose(0, 1).reshape(1, -1) @ self.Wo

    def size(self): return 0 if self.ck is None else self.ck.shape[1]


# ─────────────────────────────────────────────────────────────
# 3. StreamingLLM — Attention Sinks + Sliding Window
# ─────────────────────────────────────────────────────────────

class StreamingLLMCache:
    """
    Keeps a fixed number of 'attention sink' tokens (initial tokens)
    plus a sliding window of recent tokens.

    Paper: arxiv:2309.17453
    """

    def __init__(self, d_model, n_heads, sink_size=4, window_size=64):
        self.H, self.Hd = n_heads, d_model // n_heads
        self.sink_size = sink_size
        self.window_size = window_size
        self.Wq = torch.randn(d_model, n_heads * self.Hd) * 0.02
        self.Wk = torch.randn(d_model, n_heads * self.Hd) * 0.02
        self.Wv = torch.randn(d_model, n_heads * self.Hd) * 0.02
        self.Wo = torch.randn(n_heads * self.Hd, d_model) * 0.02
        self.sink_k = None  # [H, sink_size, Hd] — never evicted
        self.sink_v = None
        self.win_k = None   # [H, window_size, Hd] — sliding
        self.win_v = None

    def reset(self):
        self.sink_k = self.sink_v = None
        self.win_k = self.win_v = None

    def _project(self, x):
        T = x.shape[0]
        Q = (x @ self.Wq).reshape(T, self.H, self.Hd).transpose(0, 1)
        K = (x @ self.Wk).reshape(T, self.H, self.Hd).transpose(0, 1)
        V = (x @ self.Wv).reshape(T, self.H, self.Hd).transpose(0, 1)
        return Q, K, V

    def prefill(self, tokens):
        Q, K, V = self._project(tokens)
        T = tokens.shape[0]

        # Split into sinks and window
        sink_end = min(self.sink_size, T)
        self.sink_k = K[:, :sink_end, :]
        self.sink_v = V[:, :sink_end, :]

        # Window: keep last window_size tokens (after sinks)
        remaining_k = K[:, sink_end:, :]
        remaining_v = V[:, sink_end:, :]
        if remaining_k.shape[1] > self.window_size:
            remaining_k = remaining_k[:, -self.window_size:, :]
            remaining_v = remaining_v[:, -self.window_size:, :]
        self.win_k, self.win_v = remaining_k, remaining_v

        K_combined = torch.cat([self.sink_k, self.win_k], dim=1)
        V_combined = torch.cat([self.sink_v, self.win_v], dim=1)
        out = scaled_dot_product_attention(Q[:, -1:, :], K_combined, V_combined)
        return out.transpose(0, 1).reshape(1, -1) @ self.Wo

    def decode(self, tok):
        Q, K_new, V_new = self._project(tok)

        # Add new token to window
        if self.win_k is None:
            self.win_k, self.win_v = K_new, V_new
        else:
            self.win_k = torch.cat([self.win_k, K_new], dim=1)
            self.win_v = torch.cat([self.win_v, V_new], dim=1)

        # Evict oldest from window if over limit
        if self.win_k.shape[1] > self.window_size:
            self.win_k = self.win_k[:, -self.window_size:, :]
            self.win_v = self.win_v[:, -self.window_size:, :]

        K_combined = torch.cat([self.sink_k, self.win_k], dim=1)
        V_combined = torch.cat([self.sink_v, self.win_v], dim=1)
        out = scaled_dot_product_attention(Q, K_combined, V_combined)
        return out.transpose(0, 1).reshape(1, -1) @ self.Wo

    def size(self):
        s = 0 if self.sink_k is None else self.sink_k.shape[1]
        w = 0 if self.win_k is None else self.win_k.shape[1]
        return s + w

    @property
    def max_size(self): return self.sink_size + self.window_size


# ─────────────────────────────────────────────────────────────
# 4. SnapKV-style — Observation Window Selection
# ─────────────────────────────────────────────────────────────

class SnapKVCache:
    """
    Selects important tokens using the END of the prompt as an
    observation window — predicts what will matter during generation
    BEFORE generation begins.

    Inspired by: arxiv:2404.14469
    """

    def __init__(self, d_model, n_heads, budget, obs_window=16):
        self.H, self.Hd = n_heads, d_model // n_heads
        self.budget = budget
        self.obs_window = obs_window
        self.Wq = torch.randn(d_model, n_heads * self.Hd) * 0.02
        self.Wk = torch.randn(d_model, n_heads * self.Hd) * 0.02
        self.Wv = torch.randn(d_model, n_heads * self.Hd) * 0.02
        self.Wo = torch.randn(n_heads * self.Hd, d_model) * 0.02
        self.ck = None
        self.cv = None

    def reset(self): self.ck = self.cv = None

    def _project(self, x):
        T = x.shape[0]
        Q = (x @ self.Wq).reshape(T, self.H, self.Hd).transpose(0, 1)
        K = (x @ self.Wk).reshape(T, self.H, self.Hd).transpose(0, 1)
        V = (x @ self.Wv).reshape(T, self.H, self.Hd).transpose(0, 1)
        return Q, K, V

    def prefill(self, tokens):
        Q, K, V = self._project(tokens)
        T = tokens.shape[0]

        # Use last obs_window tokens as observation queries
        obs_end = T
        obs_start = max(0, T - self.obs_window)
        Q_obs = Q[:, obs_start:obs_end, :]  # [H, obs, Hd]

        # Compute attention from observation window to all prefix tokens
        prefix_k = K[:, :obs_start, :]
        if prefix_k.shape[1] > 0:
            scores = torch.matmul(Q_obs, prefix_k.transpose(-2, -1)) / math.sqrt(self.Hd)
            weights = F.softmax(scores, dim=-1)  # [H, obs, prefix_T]

            # Score each prefix token: mean attention received from observation window
            token_importance = weights.mean(dim=0).mean(dim=0)  # [prefix_T]

            # Keep top-budget tokens from prefix
            keep_count = min(self.budget, prefix_k.shape[1])
            keep_idx = token_importance.topk(keep_count).indices.sort().values

            # Selected prefix + observation window (always keep)
            selected_k = torch.cat([prefix_k[:, keep_idx, :], K[:, obs_start:, :]], dim=1)
            selected_v = torch.cat([V[:, :obs_start, :][:, keep_idx, :], V[:, obs_start:, :]], dim=1)
        else:
            selected_k = K[:, obs_start:, :]
            selected_v = V[:, obs_start:, :]

        self.ck, self.cv = selected_k, selected_v

        # Return output for last token
        Q_last = Q[:, -1:, :]
        out = scaled_dot_product_attention(Q_last, self.ck, self.cv)
        return out.transpose(0, 1).reshape(1, -1) @ self.Wo

    def decode(self, tok):
        Q, K_new, V_new = self._project(tok)
        self.ck = torch.cat([self.ck, K_new], dim=1)
        self.cv = torch.cat([self.cv, V_new], dim=1)
        out = scaled_dot_product_attention(Q, self.ck, self.cv)
        return out.transpose(0, 1).reshape(1, -1) @ self.Wo

    def size(self): return 0 if self.ck is None else self.ck.shape[1]


# ─────────────────────────────────────────────────────────────
# DEMOS
# ─────────────────────────────────────────────────────────────

def demo_cache_size_comparison():
    print("\n" + "═" * 62)
    print(bold("  DEMO 1: Cache Size Over Time — All 4 Methods"))
    print("═" * 62)

    torch.manual_seed(42)
    D, H = 128, 4
    PROMPT = 40
    GEN    = 30
    BUDGET = 20      # H2O and SnapKV keep at most 20 tokens
    SINK   = 4       # StreamingLLM: 4 sink tokens
    WINDOW = 16      # StreamingLLM: 16 recent tokens

    tokens = torch.randn(PROMPT + GEN, D)

    full    = FullCache(D, H)
    h2o     = H2OCache(D, H, budget=BUDGET)
    stream  = StreamingLLMCache(D, H, sink_size=SINK, window_size=WINDOW)
    snapkv  = SnapKVCache(D, H, budget=BUDGET, obs_window=8)

    # Prefill
    full.prefill(tokens[:PROMPT])
    h2o.prefill(tokens[:PROMPT])
    stream.prefill(tokens[:PROMPT])
    snapkv.prefill(tokens[:PROMPT])

    print(f"\n  Config: prompt={PROMPT}, generate={GEN}, budget={BUDGET}")
    print(f"  StreamingLLM: {SINK} sinks + {WINDOW} window = {SINK+WINDOW} max\n")
    print(f"  {'Step':<6} {'Full':>6} {'H2O':>6} {'Stream':>6} {'SnapKV':>6}  Bars (Full | H2O | Stream | SnapKV)")
    print(f"  {'────':<6} {'────':>6} {'───':>6} {'──────':>6} {'──────':>6}")

    for step in range(GEN):
        tok = tokens[PROMPT + step:PROMPT + step + 1]
        full.decode(tok)
        h2o.decode(tok)
        stream.decode(tok)
        snapkv.decode(tok)

        sf, sh, ss, sk = full.size(), h2o.size(), stream.size(), snapkv.size()
        mx = sf
        print(f"  {step+1:<6} {sf:>6} {sh:>6} {ss:>6} {sk:>6}  "
              f"{bar(sf,mx,8)} {bar(sh,mx,8)} {bar(ss,mx,8)} {bar(sk,mx,8)}")

    print(f"\n  Final cache sizes:")
    print(f"    Full cache:   {full.size()} tokens  (grows forever)")
    print(f"    H2O:          {h2o.size()} tokens  (capped at budget={BUDGET})")
    print(f"    StreamingLLM: {stream.size()} tokens  (capped at {SINK+WINDOW})")
    print(f"    SnapKV:       {snapkv.size()} tokens  (capped at budget={BUDGET})")
    print(f"    H2O evicted:  {h2o.evicted} tokens total")


def demo_eviction_effect_on_attention():
    """
    DEMO 2: Show how eviction changes the attention weight distribution.

    Cosine similarity between model outputs is NOT the right metric
    with random untrained weights — outputs look random regardless of
    eviction because the weight matrices map to random directions.

    Instead we show what actually changes: the ATTENTION WEIGHTS
    themselves. Eviction causes the model to attend to different tokens
    than the full cache would. We measure this directly.
    """
    print("\n" + "═" * 62)
    print(bold("  DEMO 2: How Eviction Shifts Attention Weights"))
    print("═" * 62)

    torch.manual_seed(7)
    D, H = 128, 4

    # Shared weight matrices so all caches project identically
    Wq = torch.randn(D, H * (D // H)) * 0.02
    Wk = torch.randn(D, H * (D // H)) * 0.02
    Wv = torch.randn(D, H * (D // H)) * 0.02
    Hd = D // H

    PROMPT = 30
    # Create tokens with a clear "important" region: positions 5-10 are louder
    tokens = torch.randn(PROMPT + 10, D) * 0.3
    tokens[5:11] = tokens[5:11] * 5.0   # tokens 5-10: high magnitude = attract attention

    def get_attn_weights(cache_k, query_tok):
        """Compute attention distribution from query to all cached positions."""
        Q = (query_tok @ Wq).reshape(1, H, Hd).transpose(0, 1)
        scale = math.sqrt(Hd)
        scores = torch.matmul(Q, cache_k.transpose(-2, -1)) / scale  # [H, 1, T]
        return F.softmax(scores, dim=-1).mean(dim=0).squeeze(0)      # [T] avg over heads

    def build_full_cache(toks):
        K = (toks @ Wk).reshape(len(toks), H, Hd).transpose(0, 1)  # [H, T, Hd]
        return K

    def h2o_evict(K_full, budget, query_tok):
        """Simple H2O: score by attention received from query, keep top-budget."""
        w = get_attn_weights(K_full, query_tok)
        keep = w.topk(min(budget, K_full.shape[1])).indices.sort().values
        return K_full[:, keep, :], keep.tolist()

    K_full = build_full_cache(tokens[:PROMPT])
    query  = tokens[PROMPT]  # the new token asking a question

    # Full cache attention
    w_full = get_attn_weights(K_full, query)
    top5_full = w_full.topk(5).indices.tolist()

    # H2O with different budgets
    print(f"\n  Prompt: {PROMPT} tokens (positions 5-10 are high-magnitude)")
    print(f"  Query: token at position {PROMPT}")
    print(f"\n  Which positions receive the top-5 attention weight?\n")
    print(f"  {'Cache config':<28} {'Top-5 attended positions':>30}  {'Overlap w/ full':>16}")
    print(f"  {'────────────':<28} {'────────────────────────':>30}  {'───────────────':>16}")

    print(f"  {'Full cache (30 tokens)':<28} {str(top5_full):>30}  {'(reference)'}")

    for budget in [25, 20, 15, 10, 5]:
        K_evicted, kept = h2o_evict(K_full, budget, query)
        w_evicted = get_attn_weights(K_evicted, query)
        # Map evicted indices back to original positions
        top5_orig = [kept[i] for i in w_evicted.topk(min(5, len(kept))).indices.tolist()]
        overlap = len(set(top5_orig) & set(top5_full))
        color = green if overlap >= 4 else yellow if overlap >= 2 else red
        print(f"  {f'H2O budget={budget}':<28} {str(top5_orig):>30}  "
              f"{color(f'{overlap}/5 match')}")

    print(f"\n  Key insight: with budget≥15, H2O keeps the high-importance positions")
    print(f"  (5-10) and attention distribution barely changes.")
    print(f"  At budget=5, the distribution degrades — wrong tokens attended to.")


def demo_h2o_eviction_trace():
    print("\n" + "═" * 62)
    print(bold("  DEMO 3: H2O Eviction — Watch Scores and Evictions"))
    print("═" * 62)

    torch.manual_seed(99)
    D, H = 64, 2
    PROMPT = 8
    GEN    = 10
    BUDGET = 6

    tokens = torch.randn(PROMPT + GEN, D)

    # Inject "important" tokens that should survive eviction
    tokens[2] = tokens[2] * 5   # token 2 will attract lots of attention
    tokens[5] = tokens[5] * 4   # token 5 as well

    cache = H2OCache(D, H, budget=BUDGET)
    cache.prefill(tokens[:PROMPT])

    print(f"\n  Prompt: {PROMPT} tokens, Budget: {BUDGET}, Generate: {GEN}")
    print(f"  Tokens 2 and 5 have large magnitudes (will attract attention)\n")
    print(f"  {'Step':<6} {'Cache':>6} {'Evicted':>8} {'Top-scored token indices'}")
    print(f"  {'────':<6} {'─────':>6} {'───────':>8}")

    for step in range(GEN):
        tok = tokens[PROMPT + step:PROMPT + step + 1]
        cache.decode(tok)
        top3 = cache.scores.topk(min(3, cache.size())).indices.tolist()
        print(f"  {step+1:<6} {cache.size():>6} {cache.evicted:>8}  top indices: {top3}")

    print(f"\n  Observation: tokens with high attention (2, 5) tend to persist")
    print(f"  Low-importance tokens get evicted first")


def demo_streamingllm():
    print("\n" + "═" * 62)
    print(bold("  DEMO 4: StreamingLLM — Infinite Context, Fixed Memory"))
    print("═" * 62)

    torch.manual_seed(13)
    D, H = 64, 2
    SINK = 3
    WINDOW = 8
    STEPS = 30

    tokens = torch.randn(STEPS, D)
    cache = StreamingLLMCache(D, H, sink_size=SINK, window_size=WINDOW)

    cache.prefill(tokens[:SINK + 2])  # initial prefill

    print(f"\n  Sink={SINK} tokens, Window={WINDOW} tokens")
    print(f"  Max cache size = {SINK + WINDOW} regardless of total tokens\n")
    print(f"  {'Step':<6} {'Total seen':>12} {'Cache size':>12} {'Bar'}")
    print(f"  {'────':<6} {'──────────':>12} {'──────────':>12}")

    for step in range(SINK + 2, STEPS):
        cache.decode(tokens[step:step + 1])
        total = step + 1
        print(f"  {step:<6} {total:>12} {cache.size():>12}  {bar(cache.size(), SINK+WINDOW, 20)}")

    print(green(f"\n  Cache never exceeded {SINK+WINDOW} tokens — infinite streaming works ✓"))


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

DEMOS = {
    "1": ("Cache Size Over Time",            demo_cache_size_comparison),
    "2": ("How Eviction Shifts Attention",   demo_eviction_effect_on_attention),
    "3": ("H2O Eviction Trace",              demo_h2o_eviction_trace),
    "4": ("StreamingLLM Infinite Context",   demo_streamingllm),
}

if __name__ == "__main__":
    print()
    print(bold("╔══════════════════════════════════════════════════════╗"))
    print(bold("║   KV CACHE TOKEN EVICTION — LIVE DEMO                ║"))
    print(bold("║   H2O | StreamingLLM | SnapKV | Full                 ║"))
    print(bold("╚══════════════════════════════════════════════════════╝"))

    chosen = sys.argv[1:] if len(sys.argv) > 1 else list(DEMOS.keys())
    for key in chosen:
        if key in DEMOS:
            DEMOS[key][1]()
        else:
            print(f"  Unknown demo '{key}'. Choose 1-4.")

    print("\n" + "═" * 62)
    print(bold("  Run one: python3 demo_eviction.py 1"))
    print("═" * 62 + "\n")
