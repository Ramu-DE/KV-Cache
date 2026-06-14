"""
KV Cache — Chapter 6: Code Examples
Building KV Cache from scratch in Python/PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────────────
# EXAMPLE 1: Naive Attention (No Cache) — baseline
# ─────────────────────────────────────────────────────────────────

class NaiveAttention(nn.Module):
    """
    Recomputes Q, K, V for ALL tokens every step.
    O(n²) cost — gets slower with every token generated.
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        Computes Q, K, V for ALL tokens in x each call.
        """
        B, T, D = x.shape
        H, Hd = self.num_heads, self.head_dim

        # Compute Q, K, V for ALL tokens — expensive!
        Q = self.W_q(x).view(B, T, H, Hd).transpose(1, 2)  # [B, H, T, Hd]
        K = self.W_k(x).view(B, T, H, Hd).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, Hd).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, T, T]

        # Causal mask (can't attend to future tokens)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


# ─────────────────────────────────────────────────────────────────
# EXAMPLE 2: KV Cache Attention
# ─────────────────────────────────────────────────────────────────

class KVCacheAttention(nn.Module):
    """
    Maintains a KV cache across decode steps.
    Only computes Q, K, V for the NEW token each step.
    O(n) cost — much faster!
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # KV Cache: starts empty, grows with each step
        self.cache_k = None  # [B, H, seq_so_far, Hd]
        self.cache_v = None  # [B, H, seq_so_far, Hd]

    def reset_cache(self):
        """Call this between different conversations/requests."""
        self.cache_k = None
        self.cache_v = None

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the full input prompt.
        Fills KV cache with all prompt tokens.
        x: [batch, prompt_len, d_model]
        """
        B, T, D = x.shape
        H, Hd = self.num_heads, self.head_dim

        Q = self.W_q(x).view(B, T, H, Hd).transpose(1, 2)
        K = self.W_k(x).view(B, T, H, Hd).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, Hd).transpose(1, 2)

        # Store K and V in cache
        self.cache_k = K  # [B, H, T, Hd]
        self.cache_v = V

        # Full attention over prompt
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)

    def decode_step(self, x_new: torch.Tensor) -> torch.Tensor:
        """
        Process ONE new token using the KV cache.
        x_new: [batch, 1, d_model]  ← just the new token!
        """
        B, T_new, D = x_new.shape
        assert T_new == 1, "decode_step processes one token at a time"
        H, Hd = self.num_heads, self.head_dim

        # Only compute Q, K, V for the NEW token
        Q_new = self.W_q(x_new).view(B, 1, H, Hd).transpose(1, 2)  # [B, H, 1, Hd]
        K_new = self.W_k(x_new).view(B, 1, H, Hd).transpose(1, 2)
        V_new = self.W_v(x_new).view(B, 1, H, Hd).transpose(1, 2)

        # Append new K, V to cache
        self.cache_k = torch.cat([self.cache_k, K_new], dim=2)  # [B, H, T+1, Hd]
        self.cache_v = torch.cat([self.cache_v, V_new], dim=2)

        T_total = self.cache_k.shape[2]

        # Attention: new token's Q attends to ALL cached K, V
        scores = torch.matmul(Q_new, self.cache_k.transpose(-2, -1)) / self.scale
        # Shape: [B, H, 1, T_total] — no masking needed (new token is last)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, self.cache_v)  # [B, H, 1, Hd]
        out = out.transpose(1, 2).contiguous().view(B, 1, D)
        return self.W_o(out)


# ─────────────────────────────────────────────────────────────────
# EXAMPLE 3: Grouped Query Attention (GQA)
# ─────────────────────────────────────────────────────────────────

class GroupedQueryAttention(nn.Module):
    """
    GQA: multiple Q heads share fewer K, V heads.
    Reduces KV cache size by (num_q_heads / num_kv_heads).
    """
    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0, "Q heads must be divisible by KV heads"

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.groups = num_q_heads // num_kv_heads  # Q heads per KV head
        self.head_dim = d_model // num_q_heads
        self.scale = math.sqrt(self.head_dim)

        self.W_q = nn.Linear(d_model, num_q_heads * self.head_dim, bias=False)
        # KV projections are SMALLER: only num_kv_heads
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Smaller KV cache!
        self.cache_k = None  # [B, num_kv_heads, T, Hd]
        self.cache_v = None

    def decode_step(self, x_new: torch.Tensor) -> torch.Tensor:
        B, _, D = x_new.shape
        Hq, Hkv, G, Hd = self.num_q_heads, self.num_kv_heads, self.groups, self.head_dim

        Q = self.W_q(x_new).view(B, 1, Hq, Hd).transpose(1, 2)   # [B, Hq, 1, Hd]
        K = self.W_k(x_new).view(B, 1, Hkv, Hd).transpose(1, 2)  # [B, Hkv, 1, Hd]
        V = self.W_v(x_new).view(B, 1, Hkv, Hd).transpose(1, 2)

        # Append to (smaller) KV cache
        self.cache_k = torch.cat([self.cache_k, K], dim=2) if self.cache_k is not None else K
        self.cache_v = torch.cat([self.cache_v, V], dim=2) if self.cache_v is not None else V

        T = self.cache_k.shape[2]

        # Expand KV to match Q heads: each KV head serves G query heads
        # [B, Hkv, T, Hd] → [B, Hkv, 1, T, Hd] → [B, Hkv, G, T, Hd] → [B, Hq, T, Hd]
        K_expanded = self.cache_k.unsqueeze(2).expand(B, Hkv, G, T, Hd).reshape(B, Hq, T, Hd)
        V_expanded = self.cache_v.unsqueeze(2).expand(B, Hkv, G, T, Hd).reshape(B, Hq, T, Hd)

        scores = torch.matmul(Q, K_expanded.transpose(-2, -1)) / self.scale  # [B, Hq, 1, T]
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V_expanded)  # [B, Hq, 1, Hd]
        out = out.transpose(1, 2).contiguous().view(B, 1, D)
        return self.W_o(out)


# ─────────────────────────────────────────────────────────────────
# EXAMPLE 4: Sliding Window KV Cache
# ─────────────────────────────────────────────────────────────────

class SlidingWindowAttention(nn.Module):
    """
    Only keeps the last `window_size` tokens in the KV cache.
    Fixed memory regardless of total sequence length!
    """
    def __init__(self, d_model: int, num_heads: int, window_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.window_size = window_size

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.cache_k = None
        self.cache_v = None

    def decode_step(self, x_new: torch.Tensor) -> torch.Tensor:
        B, _, D = x_new.shape
        H, Hd = self.num_heads, self.head_dim

        Q = self.W_q(x_new).view(B, 1, H, Hd).transpose(1, 2)
        K = self.W_k(x_new).view(B, 1, H, Hd).transpose(1, 2)
        V = self.W_v(x_new).view(B, 1, H, Hd).transpose(1, 2)

        # Append to cache
        if self.cache_k is None:
            self.cache_k, self.cache_v = K, V
        else:
            self.cache_k = torch.cat([self.cache_k, K], dim=2)
            self.cache_v = torch.cat([self.cache_v, V], dim=2)

        # TRUNCATE to window size — discard old tokens!
        if self.cache_k.shape[2] > self.window_size:
            self.cache_k = self.cache_k[:, :, -self.window_size:, :]
            self.cache_v = self.cache_v[:, :, -self.window_size:, :]

        scores = torch.matmul(Q, self.cache_k.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, self.cache_v)
        out = out.transpose(1, 2).contiguous().view(B, 1, D)
        return self.W_o(out)


# ─────────────────────────────────────────────────────────────────
# DEMO: Comparing approaches
# ─────────────────────────────────────────────────────────────────

def demo_kv_cache_speedup():
    """
    Shows how KV cache grows and why it matters.
    """
    print("=== KV Cache Demo ===\n")

    d_model = 512
    num_heads = 8
    batch = 1
    vocab_size = 1000

    # Simulate token embeddings
    embed = nn.Embedding(vocab_size, d_model)
    kv_attn = KVCacheAttention(d_model, num_heads)

    # Fake prompt: 10 tokens
    prompt_ids = torch.randint(0, vocab_size, (batch, 10))
    prompt_emb = embed(prompt_ids)  # [1, 10, 512]

    print("PREFILL: processing 10-token prompt...")
    _ = kv_attn.prefill(prompt_emb)
    print(f"  Cache shape after prefill: {kv_attn.cache_k.shape}")
    # [batch, heads, seq_len, head_dim] = [1, 8, 10, 64]

    print("\nDECODE: generating 5 new tokens...")
    for step in range(5):
        new_token_id = torch.randint(0, vocab_size, (batch, 1))
        new_emb = embed(new_token_id)  # [1, 1, 512]
        _ = kv_attn.decode_step(new_emb)
        print(f"  Step {step+1}: cache now holds {kv_attn.cache_k.shape[2]} tokens")

    print("\nFinal cache sizes:")
    K_bytes = kv_attn.cache_k.nelement() * 4  # float32 = 4 bytes
    V_bytes = kv_attn.cache_v.nelement() * 4
    print(f"  K cache: {K_bytes / 1024:.1f} KB")
    print(f"  V cache: {V_bytes / 1024:.1f} KB")
    print(f"  Total: {(K_bytes + V_bytes) / 1024:.1f} KB")


def demo_gqa_memory_savings():
    """
    Shows how GQA reduces KV cache size.
    """
    print("\n=== GQA Memory Savings Demo ===\n")

    d_model = 512
    seq_len = 1000

    configs = [
        ("MHA (32 Q, 32 KV)", 32, 32),
        ("GQA (32 Q, 8 KV)",  32,  8),
        ("MQA (32 Q, 1 KV)",  32,  1),
    ]

    for name, num_q, num_kv in configs:
        head_dim = d_model // num_q
        # KV cache: 2 (K+V) × num_kv_heads × seq_len × head_dim × 2 bytes (float16)
        cache_bytes = 2 * num_kv * seq_len * head_dim * 2
        print(f"  {name:<25} KV cache for {seq_len} tokens: {cache_bytes/1024:.1f} KB")


if __name__ == "__main__":
    demo_kv_cache_speedup()
    demo_gqa_memory_savings()

    print("\n=== Sliding Window Cache Demo ===")
    print("Window size = 4. Cache never exceeds 4 tokens:")
    sw = SlidingWindowAttention(d_model=64, num_heads=4, window_size=4)
    emb = nn.Embedding(100, 64)
    for i in range(8):
        tok = torch.randint(0, 100, (1, 1))
        sw.decode_step(emb(tok))
        print(f"  After token {i+1}: cache size = {sw.cache_k.shape[2]}")
