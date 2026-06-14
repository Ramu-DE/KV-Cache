# KV Cache — Chapter 4: Advanced Topics

## 1. KV Cache Quantization

Model weights are often quantized (INT8, INT4) to save memory.
KV Cache can be quantized too!

### How It Works

```
FLOAT16 K/V vector:
  [0.3251, -0.8821, 0.1104, -0.4432, ...]
   ↑ 2 bytes each element

INT8 quantization:
  Scale factor = max_abs_value / 127
  Quantized = round(value / scale)

  [0.3251 / 0.8821 * 127 ≈  47]
  [-0.8821 / 0.8821 * 127 ≈ -127]
  [0.1104 / 0.8821 * 127 ≈  16]
  ...
  
  Stored as INT8 [47, -127, 16, ...] + 1 scale factor per vector
  
  2 bytes → 1 byte (+ tiny overhead for scale)
  ~2x memory reduction!

INT4 quantization (even more aggressive):
  4 bits per value → 4x memory reduction
  Noticeable quality degradation, used for edge devices
```

### KV Cache Quantization Methods

```
  METHOD           BITS   MEMORY SAVING   QUALITY IMPACT
  ═══════════════  ════   ═════════════   ══════════════
  FP16 (baseline)  16     1x              ──── reference
  INT8             8      ~2x             minimal (< 0.5% accuracy drop)
  FP8              8      ~2x             minimal
  INT4             4      ~4x             noticeable
  NF4 (NormalFloat)4      ~4x             better than INT4
  
  FP8 KV cache is production-ready (NVIDIA H100 supports it natively)
  INT4 still research-stage for KV cache
```

---

## 2. KV Cache Offloading

When GPU memory is full, offload old KV cache to CPU RAM or disk.

```
MEMORY HIERARCHY:
═════════════════

┌────────────────────────────────────┐
│  GPU SRAM (on-chip)   192 KB       │  ← extremely fast
│  Bandwidth: ~20 TB/s               │     Flash Attention lives here
└────────────────────────────────────┘
         │  ~10x slower
┌────────────────────────────────────┐
│  GPU HBM (VRAM)     40-80 GB      │  ← fast
│  Bandwidth: ~2 TB/s               │     Active KV Cache lives here
└────────────────────────────────────┘
         │  ~30x slower
┌────────────────────────────────────┐
│  CPU RAM            256-512 GB    │  ← moderate
│  Bandwidth: ~50 GB/s (PCIe 4)    │     Inactive KV Cache offloaded
└────────────────────────────────────┘
         │  ~1000x slower
┌────────────────────────────────────┐
│  NVMe SSD           2-8 TB        │  ← slow
│  Bandwidth: ~7 GB/s               │     Cold cache storage
└────────────────────────────────────┘

OFFLOADING STRATEGY:
  Active tokens (recent)  → GPU HBM
  Inactive tokens (old)   → CPU RAM
  Very old tokens         → NVMe SSD (research, not production)
```

### H2O (Heavy-Hitter Oracle) — Selective Eviction

Not all tokens are equally important. Some tokens receive much more attention than others.

```
Attention scores over a conversation (simplified):

  Token: [The] [cat] [sat] [on] [the] [mat] [.] [He] [was] [happy]
  Avg
  Attn:   0.02  0.08  0.31  0.05  0.02  0.45  0.01  0.02  0.01  0.03
                 │    ↑                   ↑
                 │  "sat" gets           "mat" gets
                 │  lots of attention    lots of attention
                 │  (HEAVY HITTER)       (HEAVY HITTER)

H2O Algorithm:
  1. Track cumulative attention score for each token
  2. When cache is full, evict LOWEST attention tokens
  3. Keep "heavy hitters" — tokens that matter most
  
  Result: maintain 80% of quality with 50% less cache!
```

---

## 3. Sliding Window Attention

Instead of attending to ALL previous tokens, each token only attends to a fixed window.

```
FULL ATTENTION (standard):
  Token 10 attends to: [1,2,3,4,5,6,7,8,9,10]  ← all past tokens

SLIDING WINDOW (window=4):
  Token 10 attends to: [7,8,9,10]               ← only last 4
  Token 11 attends to: [8,9,10,11]
  Token 12 attends to: [9,10,11,12]

Visualization:
  Tokens: 1  2  3  4  5  6  7  8  9  10 11 12
  
  Tok 5:  ░  ░  ░  ░  ■  ■  ■  ■            ← window of 4
  Tok 8:           ░  ░  ░  ■  ■  ■  ■
  Tok 11:                    ░  ░  ░  ■  ■  ■  ■
  
  ■ = attends to this position
  ░ = outside window, ignored
  
  KV Cache size: fixed at window_size (e.g., 4096 tokens)
  NOT growing indefinitely!

Used by: Mistral 7B (window of 4096), LongFormer, BigBird
```

### Global Tokens + Sliding Window

```
Some tokens need global attention (e.g., [CLS] token, important landmarks):

  [CLS] [tok2] [tok3] ... [tokN]
    │
    └── Global token: attends to ALL other tokens
        All other tokens attend back to it
  
  Structure: some tokens global + most tokens local window
  
  LongFormer uses this to handle documents of 16K+ tokens
  with O(n * window) instead of O(n²) attention cost
```

---

## 4. Streaming Attention / StreamingLLM

Enables LLMs to run on **infinitely long** text streams with fixed memory.

```
Problem: Normal KV cache grows unboundedly.
After 1M tokens → 500 GB of KV cache? Impossible.

StreamingLLM Insight:
  Attention sinks exist! The very first tokens (often [BOS], punctuation)
  receive disproportionate attention even if unimportant semantically.
  
  STANDARD WINDOW (drops initial tokens when full):
  
    Cache: [K5,K6,K7,K8]  (window size 4)
           ↑ drops K1,K2,K3,K4
           
    Problem: Model was trained expecting [BOS] token to always be there!
             Dropping it causes quality collapse.
  
  STREAMINGLLM FIX (keep "attention sinks" + sliding window):
  
    Cache: [K1,K2][K7,K8,K9,K10]
            ↑         ↑
            Attention  Recent tokens
            sinks      (sliding window)
            (always kept)

  Fixed memory: attention_sink_size + window_size
  Quality: near identical to full attention!
  Context: theoretically infinite
```

---

## 5. Cross-Layer KV Cache Sharing

Some layers compute very similar K,V to adjacent layers. Can we share?

```
Standard:
  Layer 1: [K1,V1] cache  (unique)
  Layer 2: [K2,V2] cache  (unique)
  Layer 3: [K3,V3] cache  (unique)
  ...

Cross-Layer Sharing:
  Layers 1-2 share [K1,V1] cache
  Layers 3-4 share [K3,V3] cache
  Layers 5-6 share [K5,V5] cache

  50% memory reduction with ~1% quality drop on benchmarks
  
Research direction: CLA (Cross-Layer Attention)
Used by: some efficient transformer variants
```

---

## 6. Ring Attention — Distributed KV Cache

For sequences too long for a single GPU (e.g., 1M tokens):

```
WITHOUT Ring Attention:
  1M token sequence → 500 GB KV cache → needs 10+ GPUs just for cache!
  Problem: All-to-all communication between GPUs is slow.

WITH Ring Attention:
  
  GPU 0: holds tokens 0-250K      GPU 1: holds tokens 250K-500K
  GPU 2: holds tokens 500K-750K   GPU 3: holds tokens 750K-1M
  
  ┌──────┐         ┌──────┐
  │ GPU0 │ ──────→ │ GPU1 │
  │      │ ←────── │      │
  └──────┘         └──────┘
      ↑ ↓               ↑ ↓
  ┌──────┐         ┌──────┐
  │ GPU3 │ ──────→ │ GPU2 │
  │      │ ←────── │      │
  └──────┘         └──────┘
  
  KV blocks are passed in a RING pattern
  Each GPU computes attention for its chunk and passes KV to neighbor
  
  Communication overlaps with computation → nearly linear scaling!
  
  Enables: 1M+ token context on 8 GPUs efficiently
```

---

## 7. MLA — Multi-head Latent Attention (DeepSeek)

A radical redesign where K and V are **compressed** before caching.

```
STANDARD MHA:
  Token → Linear → K (full size)
  Token → Linear → V (full size)
  Both cached: large!

MLA (DeepSeek-V2):
  Token → Down-project → Compressed KV latent (tiny!)
                ↓
           CACHE THIS (much smaller)
                ↓
  At attention time: Up-project back to K and V
  
  Compression ratio: ~5-13x!
  
  Visualization:
  
  Normal K,V:  [████████████████]  [████████████████]  full size
  MLA latent:  [████]                                  compressed
  
  At inference: [████] → up-project → [████████████████]
                (on-the-fly, not cached)
  
  Net result: KV Cache size reduced 5-13x
  Quality: comparable to full MHA
  Used by: DeepSeek-V2, DeepSeek-V3
```

---

## Summary of Advanced Techniques

```
TECHNIQUE              GOAL                        MATURITY
═══════════════════    ════════════════════════    ═══════════
KV Quantization        Shrink cache size           Production (INT8)
KV Offloading          Exceed GPU memory limit     Production
H2O Eviction           Selective cache pruning     Research
Sliding Window         Fixed cache size            Production (Mistral)
StreamingLLM           Infinite context            Research
Cross-Layer Sharing    Reduce layers' cache        Research
Ring Attention         Multi-GPU long context      Production (LLaMA 3.1)
MLA Compression        Compress cached KV          Production (DeepSeek)
```
