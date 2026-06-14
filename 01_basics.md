# KV Cache — Chapter 1: The Foundation

## 1. What Problem Does KV Cache Solve?

Large Language Models (LLMs) generate text **one token at a time**.
Every time they produce a new token, they must look back at ALL previous tokens.

Without KV Cache, generating 100 tokens means doing 100 full passes over the input.
That is catastrophic for speed.

```
WITHOUT KV Cache — generating "The cat sat"

Step 1: Process ["The"]                        → generates "cat"
Step 2: Process ["The", "cat"]                 → generates "sat"
Step 3: Process ["The", "cat", "sat"]          → generates "on"
Step 4: Process ["The", "cat", "sat", "on"]    → generates "the"

 Every step re-computes everything from scratch!
 Cost grows as O(n²)
```

```
WITH KV Cache — generating "The cat sat"

Step 1: Process ["The"]             → cache K,V for "The"    → generates "cat"
Step 2: Process ["cat"]  + cache    → cache K,V for "cat"    → generates "sat"
Step 3: Process ["sat"]  + cache    → cache K,V for "sat"    → generates "on"
Step 4: Process ["on"]   + cache    → cache K,V for "on"     → generates "the"

 Only the NEW token is fully processed each step!
 Cost becomes O(n) for generation
```

---

## 2. What is a Transformer? (Quick Recap)

Before understanding KV Cache, we need to understand **Attention**.

A Transformer processes tokens through layers. Each layer has:

```
┌─────────────────────────────────────────────────────────────┐
│                     TRANSFORMER LAYER                        │
│                                                              │
│  Token Embeddings                                            │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────────────────────────┐                           │
│  │    SELF-ATTENTION BLOCK      │  ← This is where KV       │
│  │                              │    Cache lives             │
│  │  Q (Query)                   │                           │
│  │  K (Key)     ← computed      │                           │
│  │  V (Value)   ← from tokens   │                           │
│  └──────────────────────────────┘                           │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────────────────────────┐                           │
│  │    FEED-FORWARD NETWORK      │                           │
│  └──────────────────────────────┘                           │
│       │                                                      │
│       ▼                                                      │
│  Output                                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. What are Q, K, V?

Think of it like a **library search system**:

```
┌────────────────────────────────────────────────────────────┐
│                    LIBRARY ANALOGY                          │
│                                                             │
│  QUERY (Q)  = Your search question                         │
│               "I want books about space travel"            │
│                                                             │
│  KEY (K)    = Book catalog index entries                   │
│               "Astronomy, Rockets, Moon Landing..."        │
│                                                             │
│  VALUE (V)  = The actual book content                      │
│               (what you read once you find the book)       │
│                                                             │
│  ATTENTION  = How well your Query matches each Key         │
│               → retrieve weighted sum of Values            │
└────────────────────────────────────────────────────────────┘
```

**Mathematically:**

```
                          Q · Kᵀ
Attention(Q, K, V)  =  softmax(────────) · V
                           √d_k

  Q  = Query matrix   [seq_len × d_k]
  K  = Key matrix     [seq_len × d_k]
  V  = Value matrix   [seq_len × d_v]
  d_k = dimension of keys (for scaling)
```

---

## 4. The Core Insight Behind KV Cache

When generating token N+1, the model needs Keys and Values for ALL previous tokens.

```
Generating token at position 5 ("on"):

Tokens:  [The] [cat] [sat] [on] [the] → [mat]?
                                  ↑
                           currently here

The model computes:
  Q for "the"   ← only this token's query (new)
  K for ALL tokens [The, cat, sat, on, the]
  V for ALL tokens [The, cat, sat, on, the]

 The K and V for [The, cat, sat, on] were already computed
 in previous steps. WHY RE-COMPUTE THEM?
```

**KV Cache = store K and V from previous steps, reuse them.**

```
Without Cache:                  With Cache:
┌─────────────┐                 ┌─────────────────────────────┐
│ Recompute   │                 │ KV Cache (GPU memory)        │
│ ALL K,V     │                 │ ┌────┬────┬────┬────┐       │
│ every step  │                 │ │ K1 │ K2 │ K3 │ K4 │ ...  │
│             │                 │ ├────┼────┼────┼────┤       │
│ Slow 🐢     │                 │ │ V1 │ V2 │ V3 │ V4 │ ...  │
└─────────────┘                 │ └────┴────┴────┴────┘       │
                                │                             │
                                │ Only compute K5, V5 (new!) │
                                │ Fast 🚀                     │
                                └─────────────────────────────┘
```

---

## 5. Step-by-Step KV Cache in Action

Let's trace through generating "Hello world how are you":

```
PREFILL PHASE (process the input prompt all at once)
═══════════════════════════════════════════════════

Input prompt: "Hello world how"

  Token:   [Hello]  [world]  [how]
  Step:       ↓        ↓       ↓
  Compute: K1,V1    K2,V2   K3,V3
                                 ↓
                    Store all in KV Cache

  KV Cache now:
  ┌───────────────────────────┐
  │  Keys:   [K1] [K2] [K3]  │
  │  Values: [V1] [V2] [V3]  │
  └───────────────────────────┘

DECODE PHASE (generate tokens one by one)
══════════════════════════════════════════

Step 4 — Generate token after "how":
  New token: [how] (last token from input)
  Compute: Q4 (query for this position)
  Compute: K4, V4 → append to cache
  Attention: Q4 attends to [K1,K2,K3,K4]
  Output: "are"

  KV Cache:
  ┌─────────────────────────────────┐
  │  Keys:   [K1] [K2] [K3] [K4]  │
  │  Values: [V1] [V2] [V3] [V4]  │
  └─────────────────────────────────┘

Step 5 — Generate token after "are":
  New token: [are]
  Compute: Q5, K5, V5 → append to cache
  Attention: Q5 attends to [K1..K5]
  Output: "you"

  KV Cache:
  ┌──────────────────────────────────────┐
  │  Keys:   [K1] [K2] [K3] [K4] [K5]  │
  │  Values: [V1] [V2] [V3] [V4] [V5]  │
  └──────────────────────────────────────┘
```

---

## 6. Two Phases of LLM Inference

```
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│   PREFILL                          DECODE                     │
│   ═══════                          ══════                     │
│                                                               │
│   • Process entire prompt          • Generate one token       │
│     in parallel                      at a time               │
│                                                               │
│   • Compute ALL K,V at once        • Only compute K,V         │
│     (GPU works hard)                 for the new token        │
│                                                               │
│   • Fills the KV Cache             • Reads from KV Cache      │
│     from scratch                     + appends to it          │
│                                                               │
│   • Compute-bound                  • Memory-bandwidth bound   │
│     (limited by FLOPS)               (limited by RAM speed)   │
│                                                               │
│   Time:  [████████████]            [█][█][█][█][█]...        │
│          (one big chunk)            (many small chunks)       │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Summary

| Concept       | Description                                              |
|---------------|----------------------------------------------------------|
| Token         | A word piece — the unit LLMs process                    |
| Attention     | Mechanism to relate tokens to each other                |
| Query (Q)     | "What am I looking for?" — per current token            |
| Key (K)       | "What do I contain?" — per all tokens                   |
| Value (V)     | "What information do I hold?" — per all tokens          |
| KV Cache      | Stored K and V tensors from previous steps              |
| Prefill       | Batch processing of the input prompt                    |
| Decode        | Autoregressive generation, one token at a time          |
