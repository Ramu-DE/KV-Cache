# KV Cache — Chapter 9: Production Systems (2024)

## 1. The Serving System Landscape

By 2024, LLM serving has matured into a stack of specialized systems.
Each one solves a specific bottleneck that the previous generation left unsolved.

```
GENERATION   SYSTEM         KEY INNOVATION
──────────   ──────         ──────────────────────────────────────────
2022         FasterTransformer  Fused CUDA kernels, batched attention
2023         vLLM / Orca    PagedAttention, continuous batching
2024         SGLang         RadixAttention, structured generation
2024         vAttention     CUDA VMM — virtual memory for KV cache
2024         DistServe      Disaggregated prefill + decode hardware
2024         Mooncake       Production-scale KV cache pool management
```

---

## 2. SGLang and RadixAttention

**Source:** LMSYS Blog, January 2024 (lmsys.org/blog/2024-01-17-sglang)  
**Year:** 2024

### What Is SGLang?

SGLang (Structured Generation Language) is a serving framework designed
for **multi-call LLM programs** — workflows where many LLM calls share
structure: the same system prompt, the same few-shot examples, the same
document being analyzed with different questions.

### RadixAttention — The Key Idea

vLLM's prefix caching uses a hash table: hash the token IDs of a prefix,
look up whether the KV cache for that prefix exists. Simple — but it only
matches **exact prefixes**.

RadixAttention uses a **radix tree** (trie) instead:

```
RADIX TREE for KV prefix cache:

Root
├── "You are a helpful assistant" (system prompt A, 8 tokens)
│   ├── "Summarize this: [doc1]" → [KV for doc1 Q&A]
│   ├── "Summarize this: [doc2]" → [KV for doc2 Q&A]
│   └── "Translate to French:"  → [KV for translations]
│
└── "You are a legal expert" (system prompt B, 7 tokens)
    ├── "Review this contract:"  → [KV for contract 1]
    └── "Find termination clauses:" → [KV for clause queries]
```

**Benefits over hash-based prefix caching:**
- Matches partial prefixes (doc1 and doc2 share the system prompt prefix → cache hit)
- LRU eviction is tree-aware — evict leaf nodes first, preserve shared trunks
- Works for chains of LLM calls (multi-turn + tool use)
- Automatic cross-request KV sharing even for dynamically generated prompts

### Performance

RadixAttention is particularly effective for:
- **Batch inference** over many documents with a shared system prompt
- **Agent loops** where many sub-calls share prefix context
- **Few-shot prompting** where the same examples prefix every request

*Source: lmsys.org/blog/2024-01-17-sglang*

---

## 3. vAttention — CUDA Virtual Memory for KV Cache

**Paper:** arxiv:2405.04437 (ASPLOS 2025)  
**Year:** 2024-2025

### The Problem with PagedAttention

PagedAttention (vLLM) solved memory fragmentation by allocating KV cache
in fixed-size pages — like OS virtual memory. But it did so at the
**application level**, not the hardware level:

```
PagedAttention layout (virtual address space):

  Page 0: tokens 0-15    → physical block 7  (random location in VRAM)
  Page 1: tokens 16-31   → physical block 2
  Page 2: tokens 32-47   → physical block 15
  Page 3: tokens 48-63   → physical block 0

Problem: KV cache is NON-CONTIGUOUS in virtual address space.
         This means:
         1. Attention kernels must be rewritten to handle page tables
         2. The serving framework must manage its own memory manager
         3. Significant CPU/GPU overhead at every step for page lookups
```

### vAttention's Fix — CUDA VMM APIs

CUDA provides **Virtual Memory Management (VMM)** APIs that expose
the GPU's actual hardware page table:
- `cuMemAddressReserve` — reserve a contiguous virtual address range
- `cuMemCreate` — allocate physical memory anywhere
- `cuMemMap` — map physical blocks to the reserved virtual range

```
vAttention layout:

Virtual address space: [contiguous block: tokens 0 → N]
                        ↑ CONTIGUOUS — standard attention kernels work as-is!

Physical VRAM:         [scattered pages: wherever free blocks exist]
                        ↑ NON-CONTIGUOUS — but that's fine, hardware handles it

The GPU's hardware MMU translates virtual → physical transparently.
No application-level page table. No kernel rewrites.
```

### Benefits

```
                    PagedAttention          vAttention
Kernel rewrites?    Required               None needed
Memory manager?     In serving framework   Delegated to CUDA VMM
Runtime overhead?   Non-trivial            Minimal (hardware MMU)
Virtual layout?     Non-contiguous         Contiguous
Physical layout?    Non-contiguous         Non-contiguous (same)
Throughput gain?    2-4x vs naive          Up to 1.23x over PagedAttention
```

The 1.23x improvement is measured on long-context offline workloads.
For shorter contexts the gain is smaller.

*Source: arxiv:2405.04437 (ASPLOS 2025)*

---

## 4. Disaggregated Serving: DistServe and Mooncake

### The Prefill/Decode Asymmetry

Prefill and decode are fundamentally different workloads:

```
PREFILL:
  - Processes entire prompt in ONE parallel pass
  - Compute-intensive (many tokens at once → high arithmetic intensity)
  - Short duration, high GPU utilization
  - Bottleneck: FLOPS

DECODE:
  - Generates ONE token at a time
  - Memory-bandwidth-intensive (reads entire KV cache per step)
  - Long duration, low GPU utilization
  - Bottleneck: Memory bandwidth
```

Running both on the same GPU means **neither is optimized**:
- Big batches of prefill waste GPU during the decode phase
- Long decode requests block incoming prefills from starting

### DistServe — Disaggregated Prefill/Decode

**Paper:** arxiv:2401.09670  
**Year:** 2024

DistServe physically separates the two phases onto different machines:

```
┌─────────────────────────────────┐    KV cache    ┌──────────────────────────────┐
│     PREFILL CLUSTER             │   transfer     │    DECODE CLUSTER            │
│                                 │ ──────────────→│                              │
│  GPU type: compute-optimized    │                │  GPU type: memory-optimized  │
│  (A100 SXM with high FLOPS)     │                │  (many cheaper GPUs or       │
│                                 │                │   high-bandwidth memory)     │
│  Job: process prompt, fill KV   │                │  Job: generate tokens using  │
│  cache, ship KV to decode       │                │  shipped KV cache            │
└─────────────────────────────────┘                └──────────────────────────────┘
```

**Advantages:**
- Each cluster can be sized independently for its workload
- Prefill GPU is never idle waiting for decode to finish
- Decode GPU can serve many concurrent sessions simultaneously
- Scale each dimension independently (more prefill capacity vs. more decode capacity)

**Challenge:** Shipping KV cache between clusters over network (can be 10s of GBs).
Requires high-bandwidth interconnects (NVLink, InfiniBand).

*Source: arxiv:2401.09670*

### Mooncake — Production KV Cache Pooling at Scale

**Paper:** arxiv:2407.00079 (ByteDance)  
**Year:** 2024

Mooncake takes disaggregation further: it treats the KV cache as a
**shared resource pool** managed across many nodes:

```
Traditional serving:
  Request → one node → prefill + decode + KV cache all on same node

Mooncake:
  KV Cache Pool (distributed across many nodes)
       ↑
  Request → router → prefill node → KV stored in pool
                   → decode node → KV fetched from pool as needed

  KV cache is "first-class" — it can migrate, be cached, be shared
  across requests with similar prefixes, all transparently
```

**Key features:**
- Prefix-aware KV cache reuse across different users (shared system prompts)
- KV cache can live on CPU DRAM when GPU VRAM is full
- Aggressive prefetching: start loading KV cache before decode step begins
- Cache eviction policy aware of reuse probability

*Source: arxiv:2407.00079*

---

## 5. Production API Prefix Caching

Major AI providers expose prefix caching through their APIs.
As a developer, you can take advantage of this.

### Anthropic Claude — Explicit Cache Control

Claude's API allows **explicit cache breakpoints** using `cache_control`:

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-opus-4-8-20251001",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "You are a legal expert specializing in contract review.",
            "cache_control": {"type": "ephemeral"}   # ← cache this prefix
        }
    ],
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": long_document,
                    "cache_control": {"type": "ephemeral"}   # ← cache document too
                },
                {
                    "type": "text",
                    "text": "What are the termination clauses?"
                }
            ]
        }
    ]
)

# First call: full prefill cost
# Subsequent calls with same system + document: near-zero prefill cost
# Check: response.usage.cache_read_input_tokens > 0 confirms cache hit
```

**Cost:** Cache writes cost ~25% more than normal input tokens.
Cache reads cost ~10% of normal input token price.
For long documents queried multiple times, significant savings.

### OpenAI — Automatic Prefix Caching

OpenAI's API caches automatically — no code changes needed:

```python
# Just make normal API calls — OpenAI handles caching transparently
# Cache hits are visible in the response usage object:
# response.usage.cached_tokens > 0  → cache was used

# Best practices for maximizing cache hits:
# 1. Keep system prompt at the beginning (always the same)
# 2. Put frequently-used content before variable content
# 3. Use the same model version (cache is per-model)
```

Minimum context for caching: 1024 tokens (otherwise not worth caching).

### Google Gemini — Context Caching

Gemini 1.5 has explicit **context caching** — store large contexts
server-side and reference them by cache ID:

```python
import google.generativeai as genai

# Create a cache from a large document
cache = genai.caching.CachedContent.create(
    model='models/gemini-1.5-pro-001',
    contents=[long_document],
    ttl=datetime.timedelta(hours=1)   # cache lives for 1 hour
)

# Reference the cache in queries — much cheaper than re-uploading
model = genai.GenerativeModel.from_cached_content(cache)
response = model.generate_content("What are the termination clauses?")
```

Particularly useful for large documents (books, codebases, legal corpora)
queried repeatedly.

---

## 6. FastGen — Adaptive KV Cache Compression

**Paper:** arxiv:2310.01801  
**Year:** 2023-2024

### The Approach

FastGen observes that different attention heads in a transformer have
different specializations. It uses this to apply **head-specific**
compression policies rather than one-size-fits-all eviction.

At calibration time (offline), FastGen profiles each attention head
to understand its behavior pattern, then selects the best compression
policy for that head.

### Performance

On a 16K token context, compared to HuggingFace Accelerate baseline:
- 512 tokens: ~16% latency reduction
- 8192 tokens: ~40% latency reduction
- 16384 tokens: ~55% latency reduction

**Important caveat:** These gains are relative to HuggingFace Accelerate.
Against more optimized baselines (DeepSpeed), gains at 16K drop to ~17%.
Always check the baseline when comparing compression benchmarks.

*Source: arxiv:2310.01801*

---

## 7. Serving System Comparison

```
SYSTEM          CORE INNOVATION          BEST FOR                  MATURITY
──────────      ────────────────         ────────────────          ──────────
vLLM            PagedAttention           General purpose serving   Production
SGLang          RadixAttention           Multi-call programs       Production
vAttention      CUDA VMM                 Long context, low overhead Research→Prod
DistServe       Disagg prefill/decode    High throughput pipelines Production
Mooncake        KV pool management       Multi-node scale          Production
Ollama          Local simplicity         Developer laptops         Production
TensorRT-LLM    NVIDIA hardware fusion   Maximum NVIDIA perf       Production
```

---

## 8. Choosing a Serving Stack

```
Are you prototyping on one GPU?
  YES → vLLM (easiest, most documented)

Do you have many requests sharing system prompts?
  YES → SGLang (RadixAttention maximizes prefix reuse)

Do you have long contexts (>32K) with many concurrent users?
  YES → Consider vAttention (less memory overhead per session)

Do you have separate prefill and decode SLA requirements?
  YES → DistServe or Mooncake architecture

Are you building a developer-facing API?
  YES → Use cloud provider with prefix caching (Claude/OpenAI/Gemini)
        and structure prompts to maximize cache hits

Do you need maximum NVIDIA GPU performance?
  YES → TensorRT-LLM (but more complex to set up)
```
