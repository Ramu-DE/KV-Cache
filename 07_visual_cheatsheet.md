# KV Cache — Visual Cheat Sheet (Everything at a Glance)

## THE BIG PICTURE

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        HOW LLM INFERENCE WORKS                           │
│                                                                           │
│   Input Prompt                              Generated Output              │
│   "Explain gravity"                         "Gravity is..."               │
│         │                                         ↑                       │
│         │                                         │                       │
│         ▼                                         │                       │
│   ┌──────────┐   PREFILL              ┌──────────┐                       │
│   │Tokenizer │ ──────────────────────→│  LLM     │                       │
│   │"Explain" │   All tokens at once   │  Model   │ → token → token →     │
│   │"gravity" │                        │          │                       │
│   └──────────┘                        └────┬─────┘                       │
│                                            │                             │
│                                            ▼                             │
│                                    ┌───────────────┐                    │
│                                    │   KV CACHE    │                    │
│                                    │               │                    │
│                                    │  K: [■][■][■] │ ← grows each step │
│                                    │  V: [■][■][■] │                    │
│                                    └───────────────┘                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ATTENTION MATH IN ONE DIAGRAM

```
                        ATTENTION = "Who should I pay attention to?"

  My Query (Q)          All Keys (K)              All Values (V)
  ┌─────────┐           ┌───┬───┬───┬───┐         ┌───┬───┬───┬───┐
  │ "I want │  dot(Q,K) │K1 │K2 │K3 │K4 │ weights │V1 │V2 │V3 │V4 │
  │  space" │ ────────→ └───┴───┴───┴───┘ ───────→ └───┴───┴───┴───┘
  └─────────┘           [0.05,0.72,0.11,0.12]      weighted sum = output
                          ↑
                         high similarity with K2 ("space, rockets...")
                         → V2 contributes most to output
```

---

## KV CACHE LIFECYCLE

```
  TIME →

  t=0   t=1   t=2   t=3   t=4   t=5

  ┌───┐
  │P1 │ PREFILL (compute K,V for all prompt tokens at once)
  │P2 │
  │P3 │
  └───┘
     ↓
  Cache: [P1][P2][P3]

                ┌───┐
                │D1 │ DECODE step 1: compute K,V for new token, append to cache
                └───┘
  Cache: [P1][P2][P3][D1]

                       ┌───┐
                       │D2 │ DECODE step 2: same
                       └───┘
  Cache: [P1][P2][P3][D1][D2]

  ... and so on. Cache grows by 1 entry per generated token.
```

---

## ATTENTION PATTERNS

```
FULL ATTENTION (standard)         SLIDING WINDOW (Mistral)

     1  2  3  4  5  6                  1  2  3  4  5  6
  1 [■]                             1 [■]
  2 [■][■]                          2 [■][■]
  3 [■][■][■]                       3 [■][■][■]
  4 [■][■][■][■]                    4 [░][■][■][■]
  5 [■][■][■][■][■]                 5 [░][░][■][■][■]
  6 [■][■][■][■][■][■]              6 [░][░][░][■][■][■]
  
  Token N sees all N tokens         Token N sees only last 3
  Cache grows to N entries          Cache capped at window=3


MULTI-QUERY (MQA)                 GROUPED QUERY (GQA)

  Q heads:  Q1 Q2 Q3 Q4            Q heads:  Q1 Q2 | Q3 Q4
                 ↓                                 ↓
  K heads:     K1                   K heads:   K1     K2
  V heads:     V1                   V heads:   V1     V2

  1 KV pair shared by all Q heads   2 KV pairs, each shared by 2 Q heads
  Smallest cache (32x reduction)    Middle ground (2x here, 4-8x typical)
```

---

## MEMORY SIZE CHEATSHEET

```
Model       Layers  Heads  Head Dim  Per Token   Context 4K    Context 128K
──────────  ──────  ─────  ────────  ─────────   ───────────   ────────────
Llama-2 7B    32     32      128       0.5 MB       2 GB          64 GB
Llama-2 13B   40     40      128       0.8 MB       3.2 GB       100 GB
Llama-2 70B   80     64      128       2.5 MB      10 GB         320 GB
GPT-4 ~*      96     96      128      ~4 MB        ~16 GB        ~500 GB

* GPT-4 architecture not officially disclosed, estimated
```

---

## OPTIMIZATION DECISION TREE

```
Is your KV Cache too large?
         │
         ├── YES: Is generation quality critical?
         │              │
         │              ├── YES: Use GQA (4x savings, minimal quality loss)
         │              │        + INT8 quantization (2x savings)
         │              │
         │              └── NO:  Use MQA (32x savings, some quality drop)
         │
         ├── Do you have many requests with shared prompts?
         │              │
         │              └── YES: Enable Prefix Caching
         │                       (reduces TTFT by up to 80%)
         │
         ├── Do you need infinite/very long context?
         │              │
         │              └── YES: Sliding Window or StreamingLLM
         │
         └── Is GPU utilization low?
                        │
                        └── YES: Enable Continuous Batching
                                 (2-5x throughput improvement)
```

---

## GPU MEMORY BUDGET EXAMPLE

```
A100 80GB GPU — Llama-2 13B deployment:

  ┌─────────────────────────────────────────────────────────────┐
  │                      80 GB GPU VRAM                          │
  │                                                              │
  │  ┌────────────────────────┐  ┌───────────────────────────┐  │
  │  │    MODEL WEIGHTS       │  │       KV CACHE            │  │
  │  │       26 GB            │  │       50 GB               │  │
  │  │    (float16)           │  │  (4K ctx × batch 16)      │  │
  │  └────────────────────────┘  └───────────────────────────┘  │
  │                                                              │
  │  Remaining: 4 GB for activations, CUDA overhead             │
  └─────────────────────────────────────────────────────────────┘

  With INT8 KV quantization:
  KV cache → 25 GB → can serve batch=32 or extend context to 8K
```

---

## LATENCY BREAKDOWN

```
User sends: "Summarize this 5-page document: [3000 tokens]"

PHASE 1: PREFILL
  ════════════════════════════════  ~1.5 seconds
  (compute K,V for all 3000 input tokens)
  
  Time ∝ prompt_length × model_size

PHASE 2: DECODE (generate response)
  Each token:  ██  ~50ms
  100 tokens:  ████████████████████  ~5 seconds
  
  Time ∝ model_size / GPU_memory_bandwidth
  
TOTAL: ~6.5 seconds for a 100-token response to 3K prompt

With PREFIX CACHE (same document asked again):
  PHASE 1: ═══  ~0.1 seconds (CACHE HIT!)
  PHASE 2: same
  TOTAL: ~5.1 seconds  — 22% faster overall
```

---

## TOKEN EVICTION METHODS (Chapter 8)

```
METHOD          EVICTION STRATEGY              GRANULARITY   MEMORY BOUND
──────────────  ─────────────────────────────  ────────────  ────────────
H2O             Cumulative attention score      Per-token     Fixed budget
StreamingLLM    Sinks + sliding window          Fixed window  Sinks + W
SnapKV          Observation window prediction   Per-head      Fixed budget
PyramidKV       Layer-wise budget (pyramid)     Per-layer     Layer budget
ScissorHands    Recency-aware persistence       Per-token     Fixed budget
DuoAttention    Retrieval vs streaming heads    Per-head      Mixed

When to use:
  Infinite streaming context?  → StreamingLLM
  Long prompt, short output?   → SnapKV (predicts at prefill time)
  Can profile model offline?   → DuoAttention (best quality/size ratio)
  Simplest implementation?     → H2O (cumulative score, easy to add)
  Need per-layer control?      → PyramidKV
```

---

## 2024 SYSTEMS QUICK REFERENCE (Chapter 9)

```
SYSTEM          CORE INNOVATION                 BEST FOR
──────────────  ─────────────────────────────  ─────────────────────────
vLLM            PagedAttention (OS paging)      General purpose (production)
SGLang          RadixAttention (trie prefix)    Multi-call programs, batch Q&A
vAttention      CUDA VMM (hardware paging)      Long context, low overhead
DistServe       Disaggregated prefill/decode    High throughput pipelines
Mooncake        KV pool management (ByteDance)  Multi-node production scale
TensorRT-LLM    Hardware fusion (NVIDIA)        Max NVIDIA performance

API Prefix Caching:
  Anthropic Claude:  explicit cache_control parameter
  OpenAI:            automatic (cache_read_input_tokens in response)
  Google Gemini:     explicit CachedContent object with TTL
```

---

## QUICK REFERENCE: VOCABULARY

```
TERM                MEANING
────────────────    ─────────────────────────────────────────────────
KV Cache            Stored Key-Value tensors from previous tokens
Prefill             Processing the full input prompt in one pass
Decode              Generating tokens one at a time (uses KV cache)
TTFT                Time To First Token — how long before response starts
TPOT                Time Per Output Token — pace of generation
MHA                 Multi-Head Attention — standard full KV per head
GQA                 Grouped Query Attention — fewer KV heads (4-8x savings)
MQA                 Multi-Query Attention — single KV head (32x savings)
MLA                 Multi-head Latent Attention — compressed latent (5-13x)
CLA                 Cross-Layer Attention — share K/V across layers (~2x)
PagedAttention      OS-style virtual memory for KV cache (vLLM)
vAttention          CUDA VMM hardware paging for KV cache
RadixAttention      Trie-based prefix caching (SGLang)
Prefix Caching      Reuse KV for identical prompt prefixes
Continuous Batching Fill GPU with new requests as old ones finish
Flash Attention     IO-aware attention (FA1/FA2/FA3) — not a KV cache tech
Flash Attention 3   H100 Hopper-specific FA with WGMMA + TMA (1.5-2x FA2)
Context Window      Maximum total tokens (prompt + output) model can handle
Sequence Length     Number of tokens in a request
Batch Size          Number of simultaneous requests processed together
HBM                 High Bandwidth Memory — the GPU's main VRAM
SRAM                On-chip memory — tiny but extremely fast
KV Quantization     Compress K,V values (INT8/FP8/INT4) to save memory
Token Eviction      Discard low-importance tokens from cache
Heavy Hitter        Token that receives disproportionately high attention
Attention Sink      Initial tokens that always receive high attention (structural)
Sliding Window      Fixed-size rolling KV cache window
H2O                 Heavy-Hitter Oracle — cumulative score eviction
SnapKV              Observation-window prediction at prefill time
StreamingLLM        Attention sinks + sliding window for infinite streaming
PyramidKV           Layer-wise budget allocation (pyramid shape)
DuoAttention        Retrieval heads (full cache) vs streaming heads (sparse)
DistServe           Disaggregated prefill/decode serving system
```

---

## THE CORE TRADE-OFFS

```
         MEMORY   ←──────────────────────→   QUALITY
         
         Smallest                            Highest
         
  MQA    [▓░░░░░]  32x smaller KV cache     Quality drop on some tasks
  GQA    [▓▓░░░░]   4x smaller KV cache     Nearly identical to MHA
  MHA    [▓▓▓▓▓▓]  Baseline                 Best quality
  
         THROUGHPUT ←──────────────────────→ LATENCY
         
         Highest                             Lowest
         
  Large  [▓▓▓▓▓▓]  Many requests/sec        Slow per-request
  batch
  Small  [▓░░░░░]  Few requests/sec         Fast per-request
  batch
  
         CONTEXT LENGTH ←──────────────────→ REQUESTS/GPU
         
         Longest                             Most
  
  128K   [▓▓▓▓▓▓]  One long conversation    Maybe 1-2 requests
  4K     [▓░░░░░]  Short conversations      Many parallel requests
```
