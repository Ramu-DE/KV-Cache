# KV Cache — Chapter 5: Real-World Systems and Serving

## 1. vLLM — The Industry Standard

vLLM is the most widely used open-source LLM inference engine.

### Core Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        vLLM ARCHITECTURE                        │
│                                                                  │
│  Incoming Requests                                              │
│       │                                                         │
│       ▼                                                         │
│  ┌────────────────┐                                            │
│  │  Scheduler     │  ← decides which requests run each step   │
│  │                │    using continuous batching               │
│  └────────┬───────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌────────────────────────────────┐                            │
│  │  Block Manager (PagedAttention)│  ← manages KV cache pages  │
│  │                                │                            │
│  │  Block Table:                  │                            │
│  │  Req A → [P1, P4, P7, P12]    │                            │
│  │  Req B → [P2, P5, P8]         │                            │
│  │  Req C → [P3, P6]             │                            │
│  └────────┬───────────────────────┘                            │
│           │                                                     │
│           ▼                                                     │
│  ┌────────────────┐                                            │
│  │  Model Executor│  ← runs the actual GPU computation        │
│  │  (Flash Attn)  │                                            │
│  └────────────────┘                                            │
└────────────────────────────────────────────────────────────────┘
```

### Request Lifecycle in vLLM

```
Request arrives: "What is the capital of France?"

Step 1: WAITING queue
  Request enters scheduler queue.
  Scheduler checks: do we have enough free blocks?
  
         Waiting: [ReqA, ReqB, ReqC]
                   ↓
         If GPU memory available → RUNNING

Step 2: RUNNING
  Prefill: all tokens processed in parallel
  KV Cache pages allocated from free pool
  
  Block Pool:
  [FREE][FREE][FREE][FREE]  →  [ReqA][ReqA][FREE][FREE]
  
Step 3: DECODE (per step)
  Scheduler batches multiple running requests:
  
  Step N:  [ReqA token 47] + [ReqB token 23] + [ReqC token 8]
            ↓ GPU processes together ↓
  
  New KV appended, new tokens generated.
  
Step 4: PREEMPTION (if memory full)
  If new high-priority request arrives and memory is full:
  
  Option A: SWAP  → move low-priority request's KV to CPU RAM
  Option B: RECOMPUTE → evict KV entirely, recompute later (cheaper for short seqs)

Step 5: COMPLETE
  Stop token generated → release all KV cache pages back to pool
  
  [ReqA][ReqA][FREE][FREE]  →  [FREE][FREE][FREE][FREE]
  Pages immediately available for next request!
```

---

## 2. Prefix Caching in Production (SGLang / vLLM)

### Radix Tree for Prefix Management

```
PROBLEM: Many requests share long system prompts.

REQUEST PATTERNS:
  Req 1: [SYS: 200 tokens]["User: What is X?"]
  Req 2: [SYS: 200 tokens]["User: What is Y?"]
  Req 3: [SYS: 200 tokens]["User: Tell me about Z?"]

RADIX TREE CACHE:
  
                    [SYS: 200 tokens]
                   /       |         \
  "What is X?"    /   "What is Y?"   "Tell me about Z?"
                 /           |                \
             [KV for X]  [KV for Y]       [KV for Z]

  └─── SHARED PREFIX ───┘
       Computed ONCE, cached, reused by all 3 requests!
  
  Time To First Token:
    Without prefix cache: 200 + N tokens prefill time
    With prefix cache:    0 + N tokens prefill time
    Speedup: up to 5x for long system prompts!
```

### Cache Eviction Policy

```
When cache is full (LRU — Least Recently Used):

                 Time →
  
  Cache entries: [Prefix A] [Prefix B] [Prefix C] [Prefix D]
  Last access:     t=100      t=50       t=200      t=150
  
  New entry arrives, cache full:
  EVICT Prefix B (t=50, oldest unused)
  
  KEEP: Prefix C (most recently used, probably still active)
        Prefix D (recent)
        Prefix A (fairly recent)
```

---

## 3. Multi-GPU KV Cache Distribution

For large models (70B+), the model itself spans multiple GPUs.

```
TENSOR PARALLELISM (model sharded across GPUs):

  GPU 0: heads 1-8      GPU 1: heads 9-16
  GPU 2: heads 17-24    GPU 3: heads 25-32
  
  Each GPU stores KV cache ONLY for its heads:
  
  GPU 0: K1-K8, V1-V8   ← 1/4 of total KV cache
  GPU 1: K9-K16, V9-V16
  GPU 2: K17-K24, V17-V24
  GPU 3: K25-K32, V25-V32
  
  ALLREDUCE after attention to combine results across GPUs
  
  Total KV cache: same size, just split across GPUs
  Bandwidth: each GPU reads its shard → high efficiency

PIPELINE PARALLELISM (layers split across GPUs):

  GPU 0: layers 1-8     GPU 1: layers 9-16
  GPU 2: layers 17-24   GPU 3: layers 25-32
  
  Each GPU stores KV cache for its layers:
  GPU 0: KV for layers 1-8 (1/4 total)
  GPU 1: KV for layers 9-16
  ...
```

---

## 4. Disaggregated Prefill and Decode

Modern insight: Prefill and Decode have very different resource needs.

```
PREFILL:  Compute-bound   (needs lots of FLOPS)
DECODE:   Memory-bound    (needs lots of memory bandwidth)

Old approach (mixed):
  Same GPU does both → suboptimal for each

DISAGGREGATED SERVING:

  ┌─────────────────────┐         ┌─────────────────────┐
  │  PREFILL SERVER      │  KV     │  DECODE SERVER       │
  │  (compute-optimized) │ ──────→ │  (bandwidth-optimized│
  │                      │ transfer│                      │
  │  GPU: A100 80GB      │         │  GPU: L40S, or many  │
  │  Focus: FLOPS        │         │  smaller GPUs        │
  │  Batch prefills      │         │  Focus: low latency  │
  └─────────────────────┘         └─────────────────────┘
  
  Prefill server fills KV cache, ships it over network to decode server.
  
  Benefits:
    ✓ Prefill server can batch many prompts for efficiency
    ✓ Decode server optimized purely for token generation
    ✓ Scale each independently
  
  Challenge: KV cache transfer overhead (can be 10s of GB!)
  Used by: Mooncake (ByteDance), PD-Disaggregation research
```

---

## 5. KV Cache in Multi-Turn Conversations

```
Turn 1:
  User: "Tell me about Paris."
  Assistant: "Paris is the capital of France..."
  
  KV Cache after turn 1:
  [User_1 tokens: K,V][Assistant_1 tokens: K,V]

Turn 2 (with cache):
  User: "What about its population?"
  
  KV Cache: [User_1][Asst_1] | [User_2]
                              ↑
                    Append only User_2 KV,
                    no recompute of prior turns!
  
  CACHE HIT → only compute new user message!

Turn 3:
  [User_1][Asst_1][User_2][Asst_2] | [User_3]
                                      ↑
                          Keep appending!

Chat History vs KV Cache:
  Chat History  = the text (what was said)
  KV Cache      = pre-computed K,V tensors (how model processes it)

  Chat history takes kilobytes.
  KV cache takes gigabytes (but saves recompute time).
```

---

## 6. Context Window and KV Cache: The Connection

The "context window" IS the KV cache size limit.

```
Model: "128K context window"
       = KV cache can hold up to 128K tokens

What this means practically:

  128K context × 0.5 MB/token (7B model)
  = 64 GB just for KV cache!
  
  That's why long-context models need A100 80GB or H100 80GB GPUs.

CONTEXT LENGTH EVOLUTION:
  GPT-3 (2020):    2,048 tokens  →    1 GB KV
  GPT-3.5 (2022): 16,384 tokens  →    8 GB KV
  Claude 2 (2023): 100K tokens   →   50 GB KV
  Claude 3 (2024): 200K tokens   →  100 GB KV
  Gemini 1.5:       1M tokens    →  500 GB KV  ← distributed across GPUs
  
  Each jump in context requires:
    ✓ Architecture changes (RoPE scaling, ALiBi)
    ✓ More GPU memory
    ✓ Better KV compression (GQA, quantization)
```

---

## 7. Benchmarking KV Cache Performance

Key metrics to measure:

```
METRICS:
═══════

1. TTFT — Time To First Token
   Measures: Prefill speed (how fast KV cache is filled)
   Good value: < 500ms for interactive use
   Affected by: prompt length, GPU compute

2. TPOT — Time Per Output Token
   Measures: Decode speed (how fast each new token generates)
   Good value: < 50ms per token (> 20 tokens/sec)
   Affected by: KV cache size, memory bandwidth

3. Throughput — Tokens per second (batch)
   Measures: Total tokens generated per second across ALL requests
   Affected by: batching strategy, GPU count

4. GPU Memory Utilization
   Target: 90-95% (too low = wasteful, too high = OOM risk)

5. Cache Hit Rate (for prefix caching)
   High hit rate = big TTFT savings
   Target: > 70% for production workloads with shared prompts

SAMPLE BENCHMARK (vLLM, Llama-2 70B, 4×A100 80GB):

  Prompt length: 512    → TTFT: 180ms, TPOT: 35ms
  Prompt length: 2048   → TTFT: 650ms, TPOT: 38ms
  Prompt length: 8192   → TTFT: 2400ms, TPOT: 52ms
  
  Batch=8, 512 tokens   → Throughput: 1,800 tokens/sec
  Batch=8, 2048 tokens  → Throughput: 950 tokens/sec
```

---

## Summary

```
COMPONENT              WHERE KV CACHE MATTERS
═════════════════      ════════════════════════════════════════
vLLM / SGLang          PagedAttention, continuous batching
Prefix caching         Radix tree, LRU eviction
Multi-GPU serving      Sharded KV cache per GPU
Disaggregated serving  Transfer KV between prefill/decode nodes
Multi-turn chat        Append-only KV across conversation turns
Context window         Hard limit = KV cache memory capacity
```
