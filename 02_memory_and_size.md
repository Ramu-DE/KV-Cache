# KV Cache — Chapter 2: Memory, Size, and the Cost

## 1. Where Does KV Cache Live?

```
┌─────────────────────────────────────────────────────────────────┐
│                        GPU MEMORY LAYOUT                         │
│                                                                   │
│  ┌───────────────────┐  ┌───────────────────┐                   │
│  │   MODEL WEIGHTS   │  │    KV CACHE        │                   │
│  │                   │  │                    │                   │
│  │  All the learned  │  │  Keys & Values     │                   │
│  │  parameters of    │  │  for the current   │                   │
│  │  the model        │  │  conversation      │                   │
│  │                   │  │                    │                   │
│  │  Fixed size       │  │  GROWS with each   │                   │
│  │  (doesn't change) │  │  new token         │                   │
│  │                   │  │                    │                   │
│  │  ~14 GB (7B)      │  │  ~1 MB per 1K tok  │                   │
│  │  ~28 GB (13B)     │  │  (varies by model) │                   │
│  └───────────────────┘  └───────────────────┘                   │
│                                                                   │
│  Total GPU Memory = Model Weights + KV Cache + Activations       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. How Big is the KV Cache?

### The Formula

```
KV Cache Size =
  2             (K and V, one of each)
  × num_layers  (one cache per transformer layer)
  × num_heads   (one K,V per attention head)
  × head_dim    (size of each K or V vector)
  × seq_len     (number of tokens stored)
  × batch_size  (number of parallel conversations)
  × bytes_per_element  (2 for float16, 4 for float32)
```

### Real Numbers for Llama-2 7B

```
Model config:
  num_layers   = 32
  num_heads    = 32
  head_dim     = 128  (4096 hidden_size / 32 heads)
  dtype        = float16 (2 bytes)

KV Cache per token (1 sequence):
  = 2 × 32 × 32 × 128 × 2 bytes
  = 2 × 32 × 32 × 128 × 2
  = 524,288 bytes
  ≈ 0.5 MB per token

For 4096 token context:
  = 4096 × 0.5 MB = 2 GB

For batch of 8 sequences:
  = 8 × 2 GB = 16 GB  ← just for cache!
```

### Visual Size Comparison

```
Tokens:   100      1,000     4,096     32,768    100,000
          │         │         │          │          │
Size:     50 MB   500 MB    2 GB       16 GB      50 GB
          ▓        ▓▓▓▓▓     ▓▓▓▓▓▓▓▓   ▓▓▓▓▓     ▓▓▓▓▓▓▓▓▓
                             ← fits on  ← needs    ← needs
                               24GB GPU  A100 80GB  multi-GPU
```

---

## 3. The Memory Wall Problem

```
┌────────────────────────────────────────────────────────────────┐
│                   MEMORY PRESSURE OVER TIME                     │
│                                                                  │
│  GPU Memory                                                      │
│     80GB │                                                      │
│          │                              KV Cache grows...       │
│     60GB │                         ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓          │
│          │                    ▓▓▓▓▓                             │
│     40GB │               ▓▓▓▓▓                                 │
│          │          ▓▓▓▓▓                                       │
│     20GB │─────────────────────────────── Model Weights        │
│          │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓           │
│          │                                                      │
│      0GB └──────────────────────────────────────────── tokens  │
│             0        1K       4K       16K      32K             │
│                                                                  │
│  When KV Cache + Weights > GPU Memory → OUT OF MEMORY ERROR!   │
└────────────────────────────────────────────────────────────────┘
```

---

## 4. Cache Structure Per Layer

```
ONE TRANSFORMER LAYER'S KV CACHE:

  Attention heads (32 heads for 7B model):
  ┌────┬────┬────┬────┬────┬────┬─────┬────┐
  │ H1 │ H2 │ H3 │ H4 │ H5 │ H6 │ ... │H32 │
  └────┴────┴────┴────┴────┴────┴─────┴────┘
    ↑ each head has its own K cache and V cache

  For ONE head, say H1:
  ┌────────────────────────────────────────────────────────────┐
  │  K Cache for H1                                             │
  │  ┌──────┬──────┬──────┬──────┬──────┬──────┐             │
  │  │ K[1] │ K[2] │ K[3] │ K[4] │ K[5] │ ...  │  ← grow    │
  │  └──────┴──────┴──────┴──────┴──────┴──────┘    each step│
  │                                                             │
  │  V Cache for H1                                             │
  │  ┌──────┬──────┬──────┬──────┬──────┬──────┐             │
  │  │ V[1] │ V[2] │ V[3] │ V[4] │ V[5] │ ...  │            │
  │  └──────┴──────┴──────┴──────┴──────┴──────┘             │
  └────────────────────────────────────────────────────────────┘

  This structure is repeated for ALL 32 layers!
```

---

## 5. Data Types and Their Impact

```
FLOAT32 (full precision)    FLOAT16 (half precision)    INT8 (quantized)
═════════════════════       ══════════════════════       ════════════════
┌─────────────────┐         ┌───────────┐               ┌──────┐
│ 32 bits = 4 bytes│         │16b = 2 bytes│              │8b=1 byte│
└─────────────────┘         └───────────┘               └──────┘

KV Cache at 4K tokens:
Float32:  8 GB              Float16:  4 GB              INT8: 2 GB
          ████████                    ████                    ██

Most LLMs use float16 by default for KV cache.
INT8 KV cache (quantization) is an active research area.
```

---

## 6. Sequence Length vs. Batch Size Trade-off

You have a fixed pool of GPU memory. You choose:

```
  OPTION A: Long sequence, small batch
  ─────────────────────────────────────
  GPU Memory: [Model Weights][KV Cache: 1 request × 32K tokens]
  
  Good for: chatbots, long document summarization
  Batch = 1 conversation with 32K context


  OPTION B: Short sequence, large batch
  ──────────────────────────────────────
  GPU Memory: [Model Weights][KV Cache: 32 requests × 1K tokens]
  
  Good for: high-throughput APIs, many short requests
  Batch = 32 simultaneous conversations, each 1K tokens


  Same total memory used, very different use cases!
```

```
Memory Budget: 40 GB (after weights take 20 GB):

  Batch  │  Max Context   │  Use Case
  ───────┼────────────────┼──────────────────────────────
    1    │   80,000 tok   │  Very long documents
    2    │   40,000 tok   │  Long conversations
    4    │   20,000 tok   │  Moderate context
    8    │   10,000 tok   │  Typical chatbot
   16    │    5,000 tok   │  Short Q&A
   32    │    2,500 tok   │  High-throughput API
```

---

## 7. Memory Bandwidth — The Real Bottleneck

During decoding, each new token requires **reading the entire KV cache from GPU memory**.

```
Memory Bandwidth Bottleneck:
═════════════════════════════

A100 GPU specs:
  Memory bandwidth: 2 TB/s (terabytes per second)
  FLOPS:           312 TFLOPS (float16)

At 4K tokens, KV Cache = 2 GB:
  Time to read cache = 2 GB / 2 TB/s = 1 ms per token

At 32K tokens, KV Cache = 16 GB:
  Time to read cache = 16 GB / 2 TB/s = 8 ms per token

  Longer context = slower generation
  Not because of compute, but because of memory reads!
```

---

## Summary Table

| Property           | Impact                                       |
|--------------------|----------------------------------------------|
| Cache Size         | Grows linearly with sequence length          |
| Per-token cost     | ~0.5 MB for 7B model (float16)               |
| Bottleneck         | Memory bandwidth during decode               |
| Memory pressure    | Long sequences can OOM the GPU               |
| Batch trade-off    | Long context OR large batch, not both        |
| Data type          | float16 standard; INT8 quantization saves 2x |
