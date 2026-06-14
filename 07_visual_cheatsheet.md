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
PagedAttention      Virtual memory system for KV cache (vLLM)
Prefix Caching      Reuse KV for identical prompt prefixes
Continuous Batching Fill GPU with new requests as old ones finish
Flash Attention     Memory-efficient attention computation (not caching)
Context Window      Maximum total tokens (prompt + output) model can handle
Sequence Length     Number of tokens in a request
Batch Size          Number of simultaneous requests processed together
HBM                 High Bandwidth Memory — the GPU's main VRAM
SRAM                On-chip memory — tiny but extremely fast
KV Quantization     Compress K,V values (INT8/FP8) to save memory
Token Eviction      Discard low-importance tokens from cache (H2O)
Sliding Window      Fixed-size rolling KV cache window
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
