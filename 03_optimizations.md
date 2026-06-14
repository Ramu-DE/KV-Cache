# KV Cache — Chapter 3: Optimization Techniques

## 1. Multi-Head Attention vs. Grouped Query Attention vs. Multi-Query Attention

This is the single biggest architectural optimization for KV cache size.

### Standard Multi-Head Attention (MHA)

```
32 attention heads = 32 separate K caches + 32 separate V caches

  Q heads:  [Q1][Q2][Q3]...[Q32]   ← one per head (only used locally)
  K heads:  [K1][K2][K3]...[K32]   ← 32 K caches stored
  V heads:  [V1][V2][V3]...[V32]   ← 32 V caches stored

  Total KV cache = 32 + 32 = 64 "head caches"
  Used by: GPT-2, BERT, early GPT models
```

### Multi-Query Attention (MQA)

```
All Q heads share ONE K and ONE V:

  Q heads:  [Q1][Q2][Q3]...[Q32]   ← still 32 Q heads
  K heads:  [K1]                    ← ONLY 1 K cache!
  V heads:  [V1]                    ← ONLY 1 V cache!

  Total KV cache = 1 + 1 = 2 "head caches"
  Reduction: 32x smaller KV cache!
  Trade-off: slight quality drop
  Used by: PaLM, Falcon
```

### Grouped Query Attention (GQA) — The Sweet Spot

```
Q heads are grouped; each group shares one K,V pair:

  32 Q heads grouped into 8 groups of 4:

  Group 1:  [Q1][Q2][Q3][Q4]  →  share  [K1][V1]
  Group 2:  [Q5][Q6][Q7][Q8]  →  share  [K2][V2]
  ...
  Group 8:  [Q29]..[Q32]      →  share  [K8][V8]

  Total KV cache = 8 + 8 = 16 "head caches"
  Reduction: 4x smaller vs MHA
  Quality: nearly identical to MHA
  Used by: Llama-2 70B, Llama-3, Mistral, Gemma
```

### Visual Size Comparison

```
                    KV Cache Size (relative)

  MHA  [████████████████████████████████]  100%
  GQA  [████████]                           25%   (8 KV heads)
  MQA  [█]                                   3%   (1 KV head)

  Quality:
  MHA  ●●●●●  (best)
  GQA  ●●●●○  (nearly as good)
  MQA  ●●●○○  (noticeable drop on some tasks)
```

---

## 2. PagedAttention — Virtual Memory for KV Cache

### The Problem: Memory Fragmentation

```
WITHOUT PagedAttention:
═════════════════════════

GPU Memory:
┌────────────────────────────────────────────────────────────┐
│ Req A (needs 2K)   │ Req B (needs 4K)   │ Req C (needs 1K)│
│ ████████░░░░░░░░░░ │ ████████████░░░░░░ │ ████░░░░░░░░░░░ │
│ ^ allocated 2K     │ ^ allocated 4K     │ ^ allocated 1K  │
│   but only used 0.5K  but only used 2K    but only used 0.3K│
└────────────────────────────────────────────────────────────┘
         Fragmentation!   ^  ^  ^  Empty reserved but unusable
```

```
WITH PagedAttention (vLLM):
════════════════════════════

GPU Memory divided into FIXED-SIZE PAGES (like OS virtual memory):

┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│P1│P2│P3│P4│P5│P6│P7│P8│P9│PA│PB│PC│PD│PE│PF│PG│
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
 ↑              ↑              ↑              ↑
 Req A          Req B          Req A          Req B
 tokens 1-16    tokens 1-16    tokens 17-32   tokens 17-32

Page Table (like OS page table):
  Req A: [P1 → tokens 1-16] [P3 → tokens 17-32] [P7 → tokens 33-48]
  Req B: [P2 → tokens 1-16] [P4 → tokens 17-32]

Benefits:
  ✓ No pre-allocation waste
  ✓ Pages shared between copies of same prompt (prefix sharing)
  ✓ Pages freed immediately when request completes
  ✓ ~3x higher throughput than naive allocation
```

### Memory Waste Comparison

```
Naive allocation:  ████████░░░░░░░   ~35% waste (internal fragmentation)
PagedAttention:    ████████████████   < 4% waste
```

---

## 3. Prefix Caching (Prompt Caching)

### The Problem

Many LLM requests share a common prefix:
- A system prompt: "You are a helpful assistant..."
- A long document being analyzed with many questions
- A shared few-shot example set

```
WITHOUT prefix caching:

Request 1:  [System Prompt 500 tokens][User: "What is X?"]
            → compute KV for ALL 500 system tokens + question

Request 2:  [System Prompt 500 tokens][User: "What is Y?"]
            → recompute KV for ALL 500 system tokens + question

 Same system prompt computed twice, three times, hundreds of times!
```

```
WITH prefix caching:

First request:
  [System Prompt 500 tokens] → compute + STORE in cache (keyed by hash)
  [User: "What is X?"]       → compute + append

Subsequent requests with same prefix:
  [System Prompt 500 tokens] → CACHE HIT! Skip computation entirely
  [User: "What is Y?"]       → compute only this new part

                                cache
  Time:  ████████████████░░░  (prefix hit, only new part)
  vs     ████████████████████  (full computation without cache)

  TTFT (Time To First Token) dramatically reduced!
```

### Cache Key = Hash of Token IDs

```
Hash([token_1, token_2, ..., token_N]) → cache key

Same tokens in same order → same hash → cache hit
Even 1 token difference   → different hash → cache miss

Cache organized in blocks:
  Block 0: tokens 0-15    hash: 0xAB12...
  Block 1: tokens 16-31   hash: 0xCD34...
  Block 2: tokens 32-47   hash: 0xEF56...
```

---

## 4. Continuous Batching (Iteration-Level Batching)

### The Problem with Static Batching

```
STATIC BATCHING (old approach):

Batch of 4 requests, all must finish before new ones start:

  Request A:  [████████████████████]  20 tokens
  Request B:  [████████████░░░░░░░░]  12 tokens (finishes early, GPU idles)
  Request C:  [████████████████░░░░]  16 tokens
  Request D:  [████████░░░░░░░░░░░░]   8 tokens

  Time →      [════════════════════]  wait for A before new requests
              GPU is IDLE for B,C,D while A keeps going
```

```
CONTINUOUS BATCHING (vLLM, modern approach):

  Request A:  [████████████████████]  starts at t=0
  Request B:  [████████████]          finishes at t=12
                           ↓
  Request E:               [████████] NEW request immediately fills slot!
  Request C:  [████████████████]
  Request D:  [████████]
                        ↓
  Request F:            [██████████] Another new request fills!

  GPU utilization: near 100% always!
  Throughput: 2-5x higher
```

---

## 5. Speculative Decoding

### The Idea

Use a tiny "draft model" to guess multiple tokens ahead, then verify with the big model in one pass.

```
Normal Decoding:
  Big Model (70B)  →  1 token  →  1 token  →  1 token  →  1 token
  Latency:           ████         ████         ████         ████
                     slow         slow         slow         slow

Speculative Decoding:
  Small Model (7B) →  [guesses 5 tokens at once: "the cat sat on the"]
                       ▒▒  (tiny, fast)
  
  Big Model (70B)  →  [verify all 5 tokens in ONE pass]
                       ████ (one pass, same cost as 1 token decode!)
  
  If all 5 accepted → 5 tokens for the price of ~1.5!
  If 3 accepted    → 3 tokens for the price of ~1.5!
  If 0 accepted    → fall back to big model's token

Acceptance Rate ≈ 70-80% → ~2-3x speedup in practice
```

```
ROLE OF KV CACHE IN SPECULATIVE DECODING:

Draft model:  maintains its own small KV cache
Big model:    verifies using its large KV cache in parallel

  Draft KV cache:  [small, cheap]  ← updated speculatively
  Big KV cache:    [large, accurate] ← only updated after verification
```

---

## 6. Flash Attention Family (Compute Optimization)

Though not strictly a KV Cache technique, Flash Attention changes HOW
attention is computed and affects cache memory movement.

```
STANDARD ATTENTION:
  1. Compute full QK^T matrix  [seq × seq]  → stored in HBM (slow!)
  2. Apply softmax
  3. Multiply by V
  
  Memory reads: O(seq²)   ← bottleneck for long sequences

FLASH ATTENTION 1 (Dao et al., 2022 — arxiv:2205.14135):
  1. Split Q, K, V into BLOCKS that fit in SRAM (fast on-chip memory)
  2. Compute attention block by block using online softmax
  3. Never write large QK^T matrix to HBM
  
  Memory reads: O(seq)    ← much better!
  IO complexity: O(N²·d / M) where M = SRAM size

FLASH ATTENTION 2 (Dao, 2023 — arxiv:2307.08691):
  Improvements over FA1:
  - Better parallelism across sequence positions and heads
  - Reduced non-matmul FLOPs (causal masking optimization)
  - 2-4× faster than FA1 on A100
  - Supports Multi-Query and Grouped-Query Attention natively

FLASH ATTENTION 3 (Shah et al., 2024 — arxiv:2407.08608):
  H100 Hopper GPU specific — exploits new hardware features:
  - WGMMA: asynchronous warpgroup matrix multiply (overlaps compute + memory)
  - TMA: Tensor Memory Accelerator for async data prefetching
  - Pipeline stages: load next tile while computing on current tile
  
  Performance on H100 (fp16):
    FA2: ~330 TFLOPS/s
    FA3: ~470-570 TFLOPS/s  (1.5-2.0× faster than FA2)
    Peak H100: ~1979 TFLOPS/s (FA3 reaches ~75% utilization)

  GPU Memory Hierarchy:
  ┌──────────────┐  SRAM (on-chip)   ~192 KB   extremely fast
  │   SRAM       │  ← Flash Attention keeps computation here
  └──────────────┘
        │ 10x slower
  ┌──────────────┐  HBM (GPU RAM)    ~40-80 GB  fast
  │   HBM        │  ← Standard attention writes N×N matrix here (slow)
  └──────────────┘
```

**Note:** Flash Attention speeds up the prefill phase (compute-bound).
During decode, the bottleneck is reading the KV cache (memory-bound),
not computing attention — so FA3 helps less during decode.

## 6b. SGLang and RadixAttention

SGLang (2024) introduced **RadixAttention** — using a radix tree (trie)
instead of a hash table for prefix caching. This enables:
- Partial prefix matches across requests (not just exact hashes)
- Tree-aware LRU eviction (leaf nodes evicted first, shared trunks preserved)
- Automatic cross-request KV sharing even for dynamic prompts

Compared to vLLM's prefix caching, SGLang achieves significantly higher
cache hit rates for multi-call programs (agent loops, batch document Q&A).

See `09_systems_2024.md` for full coverage of SGLang, vAttention,
DistServe, and Mooncake.

---

## Summary of Optimizations

```
OPTIMIZATION        WHAT IT DOES                    SPEEDUP/SAVINGS
═══════════════     ══════════════════════          ═══════════════
GQA                 Fewer KV heads                  4x smaller cache
MQA                 Single KV head                  32x smaller cache
PagedAttention      No memory fragmentation         3x more requests
Prefix Caching      Reuse shared prompt KV          80%+ TTFT reduction
Continuous Batching Fill GPU with new requests      2-5x throughput
Speculative Decode  Verify multiple tokens at once  2-3x latency
Flash Attention     Efficient memory access         2-4x prefill speed
```
