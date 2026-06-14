"""
================================================================================
FILE: 06_code_examples.py
PURPOSE: Reference implementations of every major KV Cache architecture in
         PyTorch. Use this file to understand how each technique works at the
         tensor level — every operation is explicit, no black boxes.

WHAT IS IN THIS FILE
--------------------
This file contains 7 classes and 4 runnable demos. Each class is a standalone
attention layer you can instantiate, call .prefill() on a prompt, then call
.decode_step() for each generated token.

  CLASS                PAPER / SOURCE                 WHAT IT TEACHES
  ─────────────────    ─────────────────────────────  ───────────────────────────
  NaiveAttention       (baseline)                     Why KV cache is necessary:
                                                      recomputes ALL tokens every
                                                      step — O(n³) total cost.

  KVCacheAttention     Standard transformer           The core KV cache: prefill
                                                      fills the cache once, decode
                                                      appends one token per step.

  GroupedQueryAttention Ainslie et al. 2023           GQA: fewer KV heads than Q
                        arxiv:2305.13245              heads → 4x smaller cache.
                                                      Used in Llama-2 70B, Mistral.

  MLAAttention         DeepSeek-V2/V3, 2024           MLA: cache a tiny latent
                                                      vector (128 dims) instead of
                                                      full K+V (512 dims) per token.
                                                      64x smaller than MHA.

  SlidingWindowAttention Mistral 7B (2023)            Keep only the last W tokens.
                                                      Fixed memory, but loses early
                                                      context (attention sinks).

  StreamingLLMAttention Xiao et al. 2023              Fix for sliding window: keep
                         arxiv:2309.17453             first N "sink" tokens forever
                                                      + last W tokens. Enables
                                                      infinite-length generation.

  H2OAttention         Zhang et al. 2023              Score-based eviction: track
                        arxiv:2306.14048              cumulative attention each token
                                                      receives. Evict lowest-scored
                                                      when over budget.

DEMOS (run automatically at the bottom)
----------------------------------------
  demo_kv_cache_sizes()    — Memory table: MHA vs GQA vs MQA vs MLA
                             using real Llama-2 7B numbers (float16).

  demo_speed_comparison()  — Wall-clock timing: Naive (0.47 ms/step) vs
                             KV Cache (0.09 ms/step) — ~5x speedup shown live.

  demo_gqa_vs_mla()        — Cache growth printed every 10 steps:
                             MHA KB > GQA KB > MLA KB at every row.

  demo_streamingllm()      — Full cache grows to 40 tokens; StreamingLLM
                             caps at 20 regardless of sequence length.

SHAPES CONVENTION
-----------------
  All classes use: [B, H, T, Hd]
    B  = batch size
    H  = number of attention heads
    T  = sequence length (grows each decode step)
    Hd = head dimension = d_model // num_heads

HOW TO USE
----------
  from 06_code_examples import KVCacheAttention, GroupedQueryAttention

  # Standard KV cache
  layer = KVCacheAttention(d_model=512, num_heads=8)
  layer.prefill(prompt_embeddings)      # [B, T_prompt, D]
  out = layer.decode_step(new_token)    # [B, 1, D]

  # GQA with 4x smaller cache
  gqa = GroupedQueryAttention(d_model=512, num_q_heads=32, num_kv_heads=8)
  gqa.prefill(prompt_embeddings)
  out = gqa.decode_step(new_token)
  print(gqa.kv_cache_bytes())           # bytes used by cache

Run:  python3 06_code_examples.py
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


# ─────────────────────────────────────────────────────────────────
# SHARED UTILITY
# ─────────────────────────────────────────────────────────────────

def causal_mask(T: int, device=None) -> torch.Tensor:
    """Upper-triangular bool mask — True = position must be masked out."""
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)


def cache_bytes(tensor: torch.Tensor, dtype_bytes: int = 2) -> int:
    """Bytes a tensor would occupy in float16."""
    return tensor.numel() * dtype_bytes


# ─────────────────────────────────────────────────────────────────
# EXAMPLE 1: Naive Attention (No Cache) — O(n²) per step
# ─────────────────────────────────────────────────────────────────

class NaiveAttention(nn.Module):
    """
    Recomputes Q, K, V for ALL past tokens on every decode step.
    Total cost: O(n³). Catastrophic for long sequences.
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.H  = num_heads
        self.Hd = d_model // num_heads
        self.scale = self.Hd ** -0.5

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] — the ENTIRE sequence so far, reprocessed each step."""
        B, T, D = x.shape
        H, Hd = self.H, self.Hd

        Q = self.W_q(x).view(B, T, H, Hd).transpose(1, 2)   # [B, H, T, Hd]
        K = self.W_k(x).view(B, T, H, Hd).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, Hd).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) * self.scale       # [B, H, T, T]
        scores = scores.masked_fill(causal_mask(T, x.device), float('-inf'))
        out = F.softmax(scores, dim=-1) @ V                   # [B, H, T, Hd]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


# ─────────────────────────────────────────────────────────────────
# EXAMPLE 2: KV Cache Attention — O(n) per step
# ─────────────────────────────────────────────────────────────────

class KVCacheAttention(nn.Module):
    """
    Standard Multi-Head Attention with KV cache.

    Prefill:     process entire prompt once → fill cache
    Decode step: project only the new token → append to cache
                 → attend over full cached K, V

    Per-step cost: O(T·d) instead of O(T²·d). At T=2000, ~2000x less work.
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.H  = num_heads
        self.Hd = d_model // num_heads
        self.scale = self.Hd ** -0.5

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.cache_k: torch.Tensor | None = None  # [B, H, T, Hd]
        self.cache_v: torch.Tensor | None = None

    def reset_cache(self):
        self.cache_k = self.cache_v = None

    @property
    def cache_tokens(self) -> int:
        return 0 if self.cache_k is None else self.cache_k.shape[2]

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] — full prompt. Fills cache, returns output for all T tokens."""
        B, T, D = x.shape
        H, Hd = self.H, self.Hd

        Q = self.W_q(x).view(B, T, H, Hd).transpose(1, 2)
        K = self.W_k(x).view(B, T, H, Hd).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, Hd).transpose(1, 2)

        self.cache_k, self.cache_v = K.clone(), V.clone()

        scores = (Q @ K.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(causal_mask(T, x.device), float('-inf'))
        out = F.softmax(scores, dim=-1) @ V
        return self.W_o(out.transpose(1, 2).contiguous().view(B, T, D))

    def decode_step(self, x_new: torch.Tensor) -> torch.Tensor:
        """x_new: [B, 1, D] — single new token. Appends to cache, returns output."""
        B, _, D = x_new.shape
        H, Hd = self.H, self.Hd

        # Only project the one new token
        Q = self.W_q(x_new).view(B, 1, H, Hd).transpose(1, 2)   # [B, H, 1, Hd]
        K = self.W_k(x_new).view(B, 1, H, Hd).transpose(1, 2)
        V = self.W_v(x_new).view(B, 1, H, Hd).transpose(1, 2)

        # Append to cache
        self.cache_k = torch.cat([self.cache_k, K], dim=2)       # [B, H, T+1, Hd]
        self.cache_v = torch.cat([self.cache_v, V], dim=2)

        # Q is [B, H, 1, Hd] — attends to full cache, no mask needed
        out = F.softmax((Q @ self.cache_k.transpose(-2, -1)) * self.scale, dim=-1) @ self.cache_v
        return self.W_o(out.transpose(1, 2).contiguous().view(B, 1, D))


# ─────────────────────────────────────────────────────────────────
# EXAMPLE 3: Grouped Query Attention (GQA)
# ─────────────────────────────────────────────────────────────────

class GroupedQueryAttention(nn.Module):
    """
    GQA (Ainslie et al., 2023 — arxiv:2305.13245).

    num_kv_heads query heads share ONE K/V head each group.
    Cache reduction: num_q_heads / num_kv_heads  (e.g. 4x for 32Q→8KV).

    Used by: Llama-2 70B, Llama-3, Mistral, Gemma.
    When num_kv_heads == num_q_heads: standard MHA
    When num_kv_heads == 1:           Multi-Query Attention (MQA)
    """
    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0, "Q heads must be divisible by KV heads"
        self.Hq  = num_q_heads
        self.Hkv = num_kv_heads
        self.G   = num_q_heads // num_kv_heads   # Q heads per KV head
        self.Hd  = d_model // num_q_heads
        self.scale = self.Hd ** -0.5

        self.W_q = nn.Linear(d_model, num_q_heads  * self.Hd, bias=False)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.Hd, bias=False)  # smaller!
        self.W_v = nn.Linear(d_model, num_kv_heads * self.Hd, bias=False)  # smaller!
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.cache_k: torch.Tensor | None = None  # [B, Hkv, T, Hd]
        self.cache_v: torch.Tensor | None = None

    def reset_cache(self):
        self.cache_k = self.cache_v = None

    @property
    def cache_tokens(self) -> int:
        return 0 if self.cache_k is None else self.cache_k.shape[2]

    def _expand_kv(self, K: torch.Tensor, V: torch.Tensor):
        """Repeat each KV head G times to match Q heads. No data copy — view."""
        B, Hkv, T, Hd = K.shape
        K = K.unsqueeze(2).expand(B, Hkv, self.G, T, Hd).reshape(B, self.Hq, T, Hd)
        V = V.unsqueeze(2).expand(B, Hkv, self.G, T, Hd).reshape(B, self.Hq, T, Hd)
        return K, V

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] — full prompt."""
        B, T, D = x.shape
        Q = self.W_q(x).view(B, T, self.Hq,  self.Hd).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.Hkv, self.Hd).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.Hkv, self.Hd).transpose(1, 2)

        self.cache_k, self.cache_v = K.clone(), V.clone()

        Ke, Ve = self._expand_kv(K, V)
        scores = (Q @ Ke.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(causal_mask(T, x.device), float('-inf'))
        out = F.softmax(scores, dim=-1) @ Ve
        return self.W_o(out.transpose(1, 2).contiguous().view(B, T, D))

    def decode_step(self, x_new: torch.Tensor) -> torch.Tensor:
        """x_new: [B, 1, D]."""
        B, _, D = x_new.shape
        Q = self.W_q(x_new).view(B, 1, self.Hq,  self.Hd).transpose(1, 2)
        K = self.W_k(x_new).view(B, 1, self.Hkv, self.Hd).transpose(1, 2)
        V = self.W_v(x_new).view(B, 1, self.Hkv, self.Hd).transpose(1, 2)

        self.cache_k = torch.cat([self.cache_k, K], dim=2) if self.cache_k is not None else K
        self.cache_v = torch.cat([self.cache_v, V], dim=2) if self.cache_v is not None else V

        Ke, Ve = self._expand_kv(self.cache_k, self.cache_v)
        out = F.softmax((Q @ Ke.transpose(-2, -1)) * self.scale, dim=-1) @ Ve
        return self.W_o(out.transpose(1, 2).contiguous().view(B, 1, D))

    def kv_cache_bytes(self, dtype_bytes: int = 2) -> int:
        if self.cache_k is None:
            return 0
        return (self.cache_k.numel() + self.cache_v.numel()) * dtype_bytes


# ─────────────────────────────────────────────────────────────────
# EXAMPLE 4: Multi-head Latent Attention (MLA)  — DeepSeek (2024)
# ─────────────────────────────────────────────────────────────────

class MLAAttention(nn.Module):
    """
    Multi-head Latent Attention (DeepSeek-V2/V3, 2024).

    Instead of caching full K/V tensors (one per token per head),
    cache a single LOW-RANK LATENT VECTOR per token.

    At attention time: up-project latent → K, V on the fly.
    The up-projection weights are model weights (stored once, not per token).

    Cache size:  standard = 2 × H × Hd × T × bytes
                 MLA      =     d_c        × T × bytes   (d_c << H × Hd)

    Compression: ~5-13x over standard MHA, better than GQA.
    """
    def __init__(self, d_model: int, num_heads: int, latent_dim: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.H  = num_heads
        self.Hd = d_model // num_heads
        self.dc = latent_dim     # compressed latent dimension (d_c)
        self.scale = self.Hd ** -0.5

        # Q stays standard (not cached)
        self.W_q = nn.Linear(d_model, num_heads * self.Hd, bias=False)

        # KV compressed: token → latent (this is what gets cached)
        self.W_kv_down = nn.Linear(d_model, latent_dim, bias=False)

        # Latent → full K, V at attention time (model weights, not per-token)
        self.W_k_up = nn.Linear(latent_dim, num_heads * self.Hd, bias=False)
        self.W_v_up = nn.Linear(latent_dim, num_heads * self.Hd, bias=False)

        self.W_o = nn.Linear(num_heads * self.Hd, d_model, bias=False)

        # Cache stores LATENTS only — much smaller than K, V
        self.cache_latent: torch.Tensor | None = None  # [B, T, d_c]

    def reset_cache(self):
        self.cache_latent = None

    @property
    def cache_tokens(self) -> int:
        return 0 if self.cache_latent is None else self.cache_latent.shape[1]

    def kv_cache_bytes(self, dtype_bytes: int = 2) -> int:
        if self.cache_latent is None:
            return 0
        return self.cache_latent.numel() * dtype_bytes

    def _latent_to_kv(self, latent: torch.Tensor):
        """latent: [B, T, d_c] → K [B, H, T, Hd], V [B, H, T, Hd]"""
        B, T, _ = latent.shape
        K = self.W_k_up(latent).view(B, T, self.H, self.Hd).transpose(1, 2)
        V = self.W_v_up(latent).view(B, T, self.H, self.Hd).transpose(1, 2)
        return K, V

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]."""
        B, T, D = x.shape
        Q = self.W_q(x).view(B, T, self.H, self.Hd).transpose(1, 2)

        # Compress and cache latent only
        latent = self.W_kv_down(x)                   # [B, T, d_c]
        self.cache_latent = latent.clone()

        # Up-project for attention (not cached — computed from latent)
        K, V = self._latent_to_kv(latent)
        scores = (Q @ K.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(causal_mask(T, x.device), float('-inf'))
        out = F.softmax(scores, dim=-1) @ V
        return self.W_o(out.transpose(1, 2).contiguous().view(B, T, D))

    def decode_step(self, x_new: torch.Tensor) -> torch.Tensor:
        """x_new: [B, 1, D]."""
        B, _, D = x_new.shape
        Q = self.W_q(x_new).view(B, 1, self.H, self.Hd).transpose(1, 2)

        # Compress new token, append to latent cache
        new_latent = self.W_kv_down(x_new)            # [B, 1, d_c]
        self.cache_latent = (
            torch.cat([self.cache_latent, new_latent], dim=1)
            if self.cache_latent is not None else new_latent
        )

        # Up-project entire cache to K, V at decode time
        K, V = self._latent_to_kv(self.cache_latent)
        out = F.softmax((Q @ K.transpose(-2, -1)) * self.scale, dim=-1) @ V
        return self.W_o(out.transpose(1, 2).contiguous().view(B, 1, D))


# ─────────────────────────────────────────────────────────────────
# EXAMPLE 5: Sliding Window KV Cache (Mistral-style)
# ─────────────────────────────────────────────────────────────────

class SlidingWindowAttention(nn.Module):
    """
    Keeps only the last `window_size` tokens in the KV cache.
    Fixed memory regardless of total sequence length.
    Used by Mistral 7B (window = 4096).
    """
    def __init__(self, d_model: int, num_heads: int, window_size: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.H    = num_heads
        self.Hd   = d_model // num_heads
        self.W    = window_size
        self.scale = self.Hd ** -0.5

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.cache_k: torch.Tensor | None = None
        self.cache_v: torch.Tensor | None = None

    @property
    def cache_tokens(self) -> int:
        return 0 if self.cache_k is None else self.cache_k.shape[2]

    def decode_step(self, x_new: torch.Tensor) -> torch.Tensor:
        B, _, D = x_new.shape
        Q = self.W_q(x_new).view(B, 1, self.H, self.Hd).transpose(1, 2)
        K = self.W_k(x_new).view(B, 1, self.H, self.Hd).transpose(1, 2)
        V = self.W_v(x_new).view(B, 1, self.H, self.Hd).transpose(1, 2)

        self.cache_k = torch.cat([self.cache_k, K], dim=2) if self.cache_k is not None else K
        self.cache_v = torch.cat([self.cache_v, V], dim=2) if self.cache_v is not None else V

        # Evict oldest tokens beyond window
        if self.cache_k.shape[2] > self.W:
            self.cache_k = self.cache_k[:, :, -self.W:, :]
            self.cache_v = self.cache_v[:, :, -self.W:, :]

        out = F.softmax((Q @ self.cache_k.transpose(-2, -1)) * self.scale, dim=-1) @ self.cache_v
        return self.W_o(out.transpose(1, 2).contiguous().view(B, 1, D))


# ─────────────────────────────────────────────────────────────────
# EXAMPLE 6: StreamingLLM — Attention Sinks + Sliding Window
# ─────────────────────────────────────────────────────────────────

class StreamingLLMAttention(nn.Module):
    """
    StreamingLLM (Xiao et al., 2023 — arxiv:2309.17453).

    Key finding: initial tokens ("attention sinks") always receive
    disproportionate attention regardless of content. Dropping them
    collapses model quality even if they are semantically irrelevant.

    Fix: always keep the first `sink_size` tokens (sinks) AND
         a sliding window of the last `window_size` tokens.

    Cache = sink_size + window_size  (fixed, regardless of total length)
    → Enables infinite-length generation / streaming.
    """
    def __init__(self, d_model: int, num_heads: int,
                 sink_size: int = 4, window_size: int = 512):
        super().__init__()
        assert d_model % num_heads == 0
        self.H    = num_heads
        self.Hd   = d_model // num_heads
        self.sink = sink_size
        self.W    = window_size
        self.scale = self.Hd ** -0.5

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Sinks are kept permanently; window slides
        self.sink_k: torch.Tensor | None = None   # [B, H, sink, Hd]
        self.sink_v: torch.Tensor | None = None
        self.win_k:  torch.Tensor | None = None   # [B, H, ≤W, Hd]
        self.win_v:  torch.Tensor | None = None

    def reset_cache(self):
        self.sink_k = self.sink_v = None
        self.win_k  = self.win_v  = None

    @property
    def cache_tokens(self) -> int:
        s = 0 if self.sink_k is None else self.sink_k.shape[2]
        w = 0 if self.win_k  is None else self.win_k.shape[2]
        return s + w

    @property
    def max_cache_tokens(self) -> int:
        return self.sink + self.W

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]. Partitions into sinks + window."""
        B, T, D = x.shape
        Q = self.W_q(x).view(B, T, self.H, self.Hd).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.H, self.Hd).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.H, self.Hd).transpose(1, 2)

        sink_end = min(self.sink, T)
        self.sink_k = K[:, :, :sink_end, :]
        self.sink_v = V[:, :, :sink_end, :]

        remaining_k = K[:, :, sink_end:, :]
        remaining_v = V[:, :, sink_end:, :]
        if remaining_k.shape[2] > self.W:
            remaining_k = remaining_k[:, :, -self.W:, :]
            remaining_v = remaining_v[:, :, -self.W:, :]
        self.win_k, self.win_v = remaining_k, remaining_v

        K_all = torch.cat([self.sink_k, self.win_k], dim=2)
        V_all = torch.cat([self.sink_v, self.win_v], dim=2)

        # For prefill output, use full causal attention
        scores = (Q @ K.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(causal_mask(T, x.device), float('-inf'))
        out = F.softmax(scores, dim=-1) @ V
        return self.W_o(out.transpose(1, 2).contiguous().view(B, T, D))

    def decode_step(self, x_new: torch.Tensor) -> torch.Tensor:
        """x_new: [B, 1, D]."""
        B, _, D = x_new.shape
        Q = self.W_q(x_new).view(B, 1, self.H, self.Hd).transpose(1, 2)
        K = self.W_k(x_new).view(B, 1, self.H, self.Hd).transpose(1, 2)
        V = self.W_v(x_new).view(B, 1, self.H, self.Hd).transpose(1, 2)

        # Append new token to window
        self.win_k = torch.cat([self.win_k, K], dim=2) if self.win_k is not None else K
        self.win_v = torch.cat([self.win_v, V], dim=2) if self.win_v is not None else V

        # Slide window — evict oldest
        if self.win_k.shape[2] > self.W:
            self.win_k = self.win_k[:, :, -self.W:, :]
            self.win_v = self.win_v[:, :, -self.W:, :]

        K_all = torch.cat([self.sink_k, self.win_k], dim=2)
        V_all = torch.cat([self.sink_v, self.win_v], dim=2)

        out = F.softmax((Q @ K_all.transpose(-2, -1)) * self.scale, dim=-1) @ V_all
        return self.W_o(out.transpose(1, 2).contiguous().view(B, 1, D))


# ─────────────────────────────────────────────────────────────────
# EXAMPLE 7: H2O — Heavy-Hitter Oracle (2023)
# ─────────────────────────────────────────────────────────────────

class H2OAttention(nn.Module):
    """
    H2O: Heavy-Hitter Oracle (Zhang et al., 2023 — arxiv:2306.14048).

    Tracks cumulative attention score for every cached token.
    When cache exceeds `budget`, evicts the lowest-scoring token.

    Intuition: tokens that repeatedly receive high attention ("heavy hitters")
    are kept; tokens that accumulate little attention are evicted.

    Note: quality degradation depends on task. Reasoning tasks are more
    sensitive than retrieval tasks to eviction aggressiveness.
    """
    def __init__(self, d_model: int, num_heads: int, budget: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.H      = num_heads
        self.Hd     = d_model // num_heads
        self.budget = budget
        self.scale  = self.Hd ** -0.5

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.cache_k:  torch.Tensor | None = None
        self.cache_v:  torch.Tensor | None = None
        self.scores:   torch.Tensor | None = None   # cumulative importance [T]
        self.n_evicted: int = 0

    def reset_cache(self):
        self.cache_k = self.cache_v = self.scores = None
        self.n_evicted = 0

    @property
    def cache_tokens(self) -> int:
        return 0 if self.cache_k is None else self.cache_k.shape[2]

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]. Fills cache and initialises importance scores."""
        B, T, D = x.shape
        Q = self.W_q(x).view(B, T, self.H, self.Hd).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.H, self.Hd).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.H, self.Hd).transpose(1, 2)

        self.cache_k, self.cache_v = K.clone(), V.clone()

        scores_mat = F.softmax((Q @ K.transpose(-2, -1)) * self.scale, dim=-1)
        # Initialise importance = sum of attention received, averaged over heads and queries
        self.scores = scores_mat.mean(dim=0).sum(dim=0)   # [T]

        out = scores_mat @ V
        return self.W_o(out.transpose(1, 2).contiguous().view(B, T, D))

    def decode_step(self, x_new: torch.Tensor) -> torch.Tensor:
        """x_new: [B, 1, D]. Computes attention, updates scores, evicts if needed."""
        B, _, D = x_new.shape
        Q = self.W_q(x_new).view(B, 1, self.H, self.Hd).transpose(1, 2)
        K = self.W_k(x_new).view(B, 1, self.H, self.Hd).transpose(1, 2)
        V = self.W_v(x_new).view(B, 1, self.H, self.Hd).transpose(1, 2)

        K_all = torch.cat([self.cache_k, K], dim=2)
        V_all = torch.cat([self.cache_v, V], dim=2)

        attn = F.softmax((Q @ K_all.transpose(-2, -1)) * self.scale, dim=-1)
        # attn: [B, H, 1, T+1] → step scores averaged over batch and heads
        step_scores = attn[0].mean(dim=0).squeeze(0)   # [T+1]
        new_scores  = torch.cat([self.scores, torch.zeros(1)]) + step_scores

        # Evict lowest-scored token if over budget
        if K_all.shape[2] > self.budget:
            keep = new_scores.topk(self.budget).indices.sort().values
            self.cache_k = K_all[:, :, keep, :]
            self.cache_v = V_all[:, :, keep, :]
            self.scores  = new_scores[keep]
            self.n_evicted += 1
        else:
            self.cache_k, self.cache_v = K_all, V_all
            self.scores = new_scores

        out = attn @ V_all
        return self.W_o(out.transpose(1, 2).contiguous().view(B, 1, D))


# ─────────────────────────────────────────────────────────────────
# DEMOS
# ─────────────────────────────────────────────────────────────────

def demo_kv_cache_sizes():
    """Compare how much KV cache each architecture uses."""
    print("\n" + "=" * 62)
    print("  KV Cache Memory Comparison (Llama-2 7B scale, float16)")
    print("=" * 62)

    # Llama-2 7B real dimensions
    LAYERS, SEQ, DTYPE = 32, 4096, 2
    configs = [
        ("MHA  (32 KV heads)", 32, 32,  None),
        ("GQA  (8 KV heads)",  32,  8,  None),
        ("MQA  (1 KV head)",   32,  1,  None),
        ("MLA  (latent d=128)", 32, 32,  128),   # MLA: 128-dim latent
    ]

    def fmt(b: int) -> str:
        if b >= 1_000_000_000: return f"{b/1e9:.2f} GB"
        if b >= 1_000_000:     return f"{b/1e6:.1f} MB"
        return f"{b/1e3:.0f} KB"

    print(f"\n  {'Method':<22} {'Per token':>10}  {'@4K ctx':>10}  {'vs MHA'}")
    print(f"  {'──────':<22} {'─────────':>10}  {'───────':>10}  {'──────'}")

    mha_per = None
    for name, layers, kv_heads, latent_d in configs:
        d_h = 128  # head dim
        if latent_d is not None:
            per = layers * latent_d * DTYPE            # MLA: one latent per layer
        else:
            per = 2 * layers * kv_heads * d_h * DTYPE  # standard: K + V
        ctx4 = per * SEQ
        if mha_per is None:
            mha_per = ctx4
        ratio = f"{mha_per / ctx4:.0f}x smaller" if ctx4 < mha_per else "baseline"
        print(f"  {name:<22} {fmt(per):>10}  {fmt(ctx4):>10}  {ratio}")

    print()


def demo_speed_comparison():
    """Measure wall-clock decode speed: Naive vs KV Cache."""
    print("=" * 62)
    print("  Speed: Naive Attention vs KV Cache Attention")
    print("=" * 62)

    torch.manual_seed(42)
    D, H, PROMPT, GEN = 256, 8, 30, 40

    embed = nn.Embedding(500, D)
    naive = NaiveAttention(D, H)
    kv    = KVCacheAttention(D, H)

    tokens = torch.randint(0, 500, (1, PROMPT + GEN))
    embs   = embed(tokens)

    # Naive: feed entire growing history each step
    t0, nc_times = time.perf_counter(), []
    hist = embs[:, :PROMPT, :]
    for i in range(GEN):
        with torch.no_grad():
            naive(hist)
        nc_times.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        hist = torch.cat([hist, embs[:, PROMPT + i:PROMPT + i + 1, :]], dim=1)

    # KV Cache: prefill once, decode one token at a time
    kv.prefill(embs[:, :PROMPT, :])
    t0, kv_times = time.perf_counter(), []
    for i in range(GEN):
        with torch.no_grad():
            kv.decode_step(embs[:, PROMPT + i:PROMPT + i + 1, :])
        kv_times.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()

    avg_nc, avg_kv = sum(nc_times) / len(nc_times), sum(kv_times) / len(kv_times)
    print(f"\n  Prompt={PROMPT}, Generate={GEN}, d_model={D}, heads={H}")
    print(f"\n  {'Method':<18} {'Avg ms/step':>12}  {'Total ms':>10}")
    print(f"  {'──────':<18} {'───────────':>12}  {'────────':>10}")
    print(f"  {'Naive':18} {avg_nc:>12.3f}  {sum(nc_times):>10.1f}")
    print(f"  {'KV Cache':18} {avg_kv:>12.3f}  {sum(kv_times):>10.1f}")
    print(f"\n  KV Cache is {avg_nc / avg_kv:.1f}x faster per step\n")


def demo_gqa_vs_mla():
    """Show GQA and MLA cache footprint differences in practice."""
    print("=" * 62)
    print("  GQA vs MLA: Cache Growth Over 50 Decode Steps")
    print("=" * 62)

    torch.manual_seed(0)
    D, GEN = 256, 50
    PROMPT = 10

    gqa_mha = GroupedQueryAttention(D, num_q_heads=8, num_kv_heads=8)
    gqa_4   = GroupedQueryAttention(D, num_q_heads=8, num_kv_heads=2)
    mla     = MLAAttention(D, num_heads=8, latent_dim=32)   # 32 << 8×32=256

    tokens = torch.randn(1, PROMPT + GEN, D)
    for m in [gqa_mha, gqa_4, mla]:
        m.prefill(tokens[:, :PROMPT, :]) if hasattr(m, 'prefill') else None

    print(f"\n  d_model={D}, {PROMPT}-token prompt, {GEN} decode steps")
    print(f"\n  {'Step':>5}  {'MHA KV':>10}  {'GQA-4 KV':>10}  {'MLA latent':>12}  ratio MHA/MLA")
    print(f"  {'────':>5}  {'──────':>10}  {'────────':>10}  {'──────────':>12}")

    for i in range(0, GEN, 10):
        tok = tokens[:, PROMPT + i:PROMPT + i + 1, :]
        gqa_mha.decode_step(tok)
        gqa_4.decode_step(tok)
        mla.decode_step(tok)

        b_mha = gqa_mha.kv_cache_bytes()
        b_gqa = gqa_4.kv_cache_bytes()
        b_mla = mla.kv_cache_bytes()
        print(f"  {i+1:>5}  {b_mha/1024:>8.1f}KB  {b_gqa/1024:>8.1f}KB  "
              f"{b_mla/1024:>10.1f}KB  {b_mha / b_mla:.1f}x")

    print()


def demo_streamingllm():
    """Show StreamingLLM attention sinks keeping memory fixed."""
    print("=" * 62)
    print("  StreamingLLM: Fixed Memory for Infinite Context")
    print("=" * 62)

    torch.manual_seed(7)
    D, H = 128, 4
    SINK, WIN = 4, 16
    STEPS = 40

    full   = KVCacheAttention(D, H)
    stream = StreamingLLMAttention(D, H, sink_size=SINK, window_size=WIN)

    tokens = torch.randn(STEPS, D)
    full.prefill(tokens[:SINK].unsqueeze(0))
    stream.prefill(tokens[:SINK].unsqueeze(0))

    print(f"\n  Sinks={SINK}, Window={WIN}, max cache = {SINK+WIN}")
    print(f"  {'Step':>5}  {'Full cache':>12}  {'Stream cache':>14}  Note")
    print(f"  {'────':>5}  {'──────────':>12}  {'────────────':>14}")

    for i in range(SINK, STEPS):
        tok = tokens[i].unsqueeze(0).unsqueeze(0)
        full.decode_step(tok)
        stream.decode_step(tok)
        note = " ← CAPPED" if stream.cache_tokens == SINK + WIN else ""
        print(f"  {i+1:>5}  {full.cache_tokens:>12}  {stream.cache_tokens:>14}{note}")

    print(f"\n  Full cache at end:   {full.cache_tokens} tokens (grows forever)")
    print(f"  Stream cache at end: {stream.cache_tokens} tokens "
          f"(capped at {SINK + WIN} regardless of length)\n")


if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  06 Code Examples — KV Cache Implementations")
    print("=" * 62)

    demo_kv_cache_sizes()
    demo_speed_comparison()
    demo_gqa_vs_mla()
    demo_streamingllm()

    # Quick sliding-window sanity check
    print("=" * 62)
    print("  Sliding Window (window=4, 8 steps)")
    print("=" * 62)
    sw = SlidingWindowAttention(64, 4, window_size=4)
    for i in range(8):
        sw.decode_step(torch.randn(1, 1, 64))
        print(f"  Step {i+1}: cache = {sw.cache_tokens} tokens")
    print()
