# KV Cache — Chapter 8: Token Eviction

## 1. The Core Problem: You Can't Keep Everything

As sequence length grows, the KV cache grows with it — forever.
At some point, VRAM runs out. You have two choices:

```
Option A: Keep all tokens (full cache)
  ✓ Perfect attention quality
  ✗ VRAM grows unboundedly
  ✗ OOM at long contexts

Option B: Evict some tokens (pruned cache)
  ✓ Fixed or bounded VRAM
  ✓ Faster decode (less data to read)
  ✗ Some information loss
  ✗ Attention quality degrades... or does it?
```

**The key insight:** Not all tokens are equally important.
Attention scores are highly skewed — a small subset of tokens
receives the vast majority of attention weight. If we keep only
the important tokens and discard the rest, quality barely drops.

---

## 2. The Attention Score Distribution

Run attention on any real sequence and plot the weights:

```
Attention weights for token "she" attending over a 20-token context:

Token:   [The] [cat] [sat] [on] [the] [mat] [.] [She] [then] [picked]
Weight:   0.01  0.03  0.02  0.01  0.02  0.04  0.01 0.54   0.01   0.02
                                                     ↑
                                               "She" attends
                                               mostly to itself

Cumulative top-3 tokens: [She, mat, cat] = 0.54 + 0.04 + 0.03 = 0.61 (61%)
```

**This skewness is consistent.** In practice:
- Top 5% of tokens → ~60-80% of total attention weight
- Bottom 50% of tokens → ~5-10% of total attention weight

Token eviction exploits this: **keep the important 20%, discard the rest**.

---

## 3. H2O — Heavy-Hitter Oracle

**Paper:** arxiv:2306.14048  
**Year:** 2023

### Core Idea

Track a **cumulative attention score** for each token across all
generation steps. Tokens that accumulate high scores are "heavy
hitters" — they matter to the model repeatedly. Keep them.
Evict the rest when the cache budget is exceeded.

```
Cache Budget: 4 tokens (for illustration)

Step 1 — tokens [A,B,C,D] in cache, generate E:
  Attention from E: [A:0.1, B:0.3, C:0.5, D:0.1]
  Cumulative scores: [A:0.1, B:0.3, C:0.5, D:0.1]

Step 2 — new token F arrives, cache full (4 slots):
  Evict lowest cumulative score = A (0.1)
  Cache: [B:0.3, C:0.5, D:0.1, E:0.2(new)]

Step 3 — new token G arrives:
  Cumulative scores updated
  Evict D (lowest) → keep [B, C, E, F]
  ...
```

### Algorithm

```
Initialize:
  heavy_hitters = {}    # token → cumulative score
  budget = K            # max tokens to keep

On each decode step t:
  1. Compute attention scores a[t][i] for all cached tokens i
  2. Update: heavy_hitters[i] += a[t][i] for all i
  3. Add new token t to cache
  4. If len(cache) > budget:
       evict = argmin(heavy_hitters)
       remove evict from cache and heavy_hitters
```

### What H2O Does NOT Guarantee

- It does NOT guarantee accuracy is preserved — it trades quality for memory
- Flat 20% retention may be insufficient for very long contexts or multi-hop reasoning
- The StreamingLLM "attention sink" finding (see below) partially reframes the theory:
  initial tokens matter structurally, not just because they accumulate scores

*Source: arxiv:2306.14048*

---

## 4. StreamingLLM — Attention Sinks

**Paper:** arxiv:2309.17453 (Xiao et al., 2023)  
**Year:** 2023

### The Attention Sink Discovery

Empirical observation: the **very first tokens** (especially `[BOS]` and
early punctuation) receive disproportionately high attention scores —
even when they are semantically irrelevant to the current generation step.

```
Attention weight distribution (schematic):

  Token position:  1    2    3    4    5   ...  995  996  997  998  999  1000
  Avg attention:  0.15  0.08  0.02  0.01  0.01 ... 0.01  0.01  0.02  0.08  0.22

  ↑                                                                          ↑
  Initial tokens (SINKS)                                    Recent tokens (window)
  Always high regardless                                    High because recent
  of content
```

**Why?** The model learned to dump "unused" attention onto early tokens
(especially `[BOS]`) as a soft no-op. This is a learned behavior from
training, not a semantic relationship.

### StreamingLLM Fix

Keep two sets of tokens permanently:
1. **Attention sinks** — first N tokens (typically N=4), always retained
2. **Recent window** — last W tokens, sliding window

```
Full cache (naive):    [t1 t2 t3 t4 ... t995 t996 t997 t998 t999 t1000]
                        ↑                                              ↑
                        gets evicted                             kept in window
                        → model breaks!

StreamingLLM cache:   [t1 t2 t3 t4] + [t997 t998 t999 t1000]
                        ↑ SINK kept     ↑ WINDOW (recent tokens)
                        always          slide forward each step

Fixed memory: N + W tokens regardless of total sequence length → infinite streaming!
```

### Memory Formula

```
StreamingLLM cache size = (sink_size + window_size) × per_token_bytes
                        = (4 + 2048) × 0.5 MB    (for 7B model)
                        = 1.026 GB

Vs full cache at 10K tokens: 10000 × 0.5 MB = 5 GB
```

*Source: arxiv:2309.17453*

---

## 5. SnapKV — Observation Window Prediction

**Paper:** arxiv:2404.14469 (Li et al., 2024)  
**Year:** 2024

### The Problem with H2O

H2O tracks cumulative scores **during generation** — it only knows
which tokens matter after they've been attended to. It cannot know
in advance during prefill.

### SnapKV's Idea

Use the **end of the prompt** as an observation window to predict
which tokens will be important during generation — *before* generation begins.

```
Prompt structure:
  [System: you are a legal assistant]          ← early context
  [Document: 4000 tokens of contract text]     ← long middle
  [User: "What are the termination clauses?"]  ← OBSERVATION WINDOW (last ~32 tokens)

SnapKV during prefill:
  1. Compute attention from the observation window to ALL previous tokens
  2. Identify which tokens the window attends to most heavily
  3. Keep only those tokens in the KV cache
  4. Discard the rest BEFORE generation begins

Result: KV cache is pruned at prefill time, not during generation
```

### Per-Head Granularity

Different attention heads focus on different tokens.
SnapKV selects important tokens **per head independently**:

```
Head 1 (syntax-focused): keeps [tokens 12, 45, 89, 234, 891]
Head 2 (semantic-focused): keeps [tokens 3, 67, 234, 456, 1200]
Head 3 (coreference): keeps [tokens 1, 2, 234, 567, 890]

Each head gets its own budget K — best tokens per head, not overall
```

### Verified Performance

At 16K input tokens (verified by adversarial research, 2-1 vote):
- **3.6× generation speed increase**
- **8.2× memory efficiency improvement**
- Comparable performance across 16 LongBench datasets

*Source: arxiv:2404.14469*

---

## 6. PyramidKV — Layer-Wise Budget Allocation

**Paper:** arxiv:2406.02069 (Cai et al., 2024)  
**Year:** 2024

### The Observation

Different layers of a transformer have different attention patterns:
- **Early layers** (1-8): broad, diffuse attention — tokens attend widely
- **Middle layers** (9-24): more focused attention patterns emerge
- **Late layers** (25-32): very sharp, precise attention — highly selective

```
Layer 1:  attention spread across many tokens (low selectivity)
Layer 16: attention more concentrated on a few tokens
Layer 32: attention very sharp, only 2-3 tokens dominate

Early layers: need LARGER budget (more tokens matter equally)
Late layers:  need SMALLER budget (few tokens dominate)
```

### PyramidKV's Approach

Allocate cache budget as a **pyramid** — larger budget for early layers,
smaller for late layers:

```
Layer:    1    2    3    4    ...   28   30   32
Budget: 2048 1800 1600 1400  ...  512  256  128

                   Total cache = same as flat allocation
                   But: matches each layer's actual needs
```

This is more efficient than flat allocation (same budget for all layers)
because late layers naturally have fewer important tokens.

*Source: arxiv:2406.02069*

---

## 7. ScissorHands — Persistent vs. Transient Tokens

**Paper:** ScissorHands (2023)  
**Year:** 2023-2024

### The Core Distinction

ScissorHands classifies tokens into two types based on their
attention pattern stability over time:

```
PERSISTENT tokens:   Receive high attention consistently across ALL generation steps
  Example: [BOS], key nouns, entities being discussed
  Pattern: attention score stays high from step 1 to step 1000

TRANSIENT tokens:    Receive high attention briefly then fade
  Example: context that was briefly relevant but no longer needed
  Pattern: spike early, then drop to near-zero
```

### Why This Matters

H2O uses **cumulative** scores — a transient token can accumulate
high score during its brief spike and stay in cache long after it's
no longer needed.

ScissorHands uses **recency-aware** scoring:
- Recent attention weight matters more than old
- Tokens whose attention has faded are candidates for eviction

```
H2O sees:      Token X score = 5.2 (accumulated over 100 steps)
ScissorHands:  Token X: high for first 10 steps, zero for last 90 steps → evict!
```

---

## 8. DuoAttention — Retrieval vs. Streaming Heads

**Paper:** arxiv:2410.10819 (2024)  
**Year:** 2024

### The Architecture-Level Observation

Not all attention heads behave the same way. DuoAttention categorizes heads into:

```
RETRIEVAL HEADS:   Need access to ALL past tokens
  - These heads do semantic lookup (finding relevant information anywhere in context)
  - Must keep full KV cache for these heads
  - Cannot be evicted without quality loss

STREAMING HEADS:   Only need recent tokens + attention sinks
  - These heads process local context and syntactic patterns
  - Can use StreamingLLM-style sparse cache (sinks + window)
  - Accounts for ~50-70% of heads in typical LLMs
```

### Memory Reduction

```
32-head model:
  12 retrieval heads: full KV cache (12/32 × cache)
  20 streaming heads: sink + window cache (much smaller)

  Full cache:      32 × seq_len × d_h × bytes
  DuoAttention:    12 × seq_len × d_h × bytes  +  20 × (sink+window) × d_h × bytes
                 ≈ 37.5% of full cache  +  small streaming overhead
```

### How to Identify Head Types

DuoAttention determines head type through **offline profiling**:
- Run the model on calibration data
- Identify heads that consistently need long-range lookups vs. local patterns
- Label each head once at model-load time — zero runtime overhead

*Source: arxiv:2410.10819*

---

## 9. Comparison Table

| Method | Eviction Trigger | Granularity | When Best | Limitation |
|--------|-----------------|-------------|-----------|------------|
| **H2O** | Cumulative attention score | Per-token | General purpose, easy to implement | Flat budget; transient tokens may persist |
| **StreamingLLM** | Sliding window + sinks | Fixed window | Infinite streaming / chatbots | Cannot recall distant past |
| **SnapKV** | Observation window prediction | Per-head | Long prompt, short generation | Prediction may miss generation-specific tokens |
| **PyramidKV** | Layer-wise budget allocation | Per-layer | Any long-context task | Requires layer profiling |
| **ScissorHands** | Recency-aware scoring | Per-token | Mixed persistent/transient content | More complex to implement |
| **DuoAttention** | Head type classification | Per-head | Models with identifiable head roles | Requires offline calibration |

---

## 10. How to Choose

```
Is your use case streaming / infinite context?
  YES → StreamingLLM (sinks + window)
  NO  ↓

Do you need the simplest possible implementation?
  YES → H2O (cumulative score, easy to add to any framework)
  NO  ↓

Is your prompt long and generation short? (summarization, Q&A)
  YES → SnapKV (observation window predicts at prefill time)
  NO  ↓

Can you profile the model offline?
  YES → DuoAttention (best quality/size tradeoff for profiled models)
  NO  ↓

Do you need per-layer control?
  YES → PyramidKV
  NO  → H2O or SnapKV

Combining methods:
  PyramidKV + SnapKV  = per-layer budget + per-head selection
  StreamingLLM + DuoAttention = sinks for streaming heads + full cache for retrieval heads
```

---

## 11. The Eviction-Accuracy Frontier

Every eviction method trades accuracy for memory. The key question:
**how much accuracy do you lose per GB saved?**

```
Memory saved (%) vs Quality retained (approximate):

  Method          10% pruning   30% pruning   50% pruning   70% pruning
  ─────────────   ─────────────────────────────────────────────────────
  H2O             ~99%          ~97%          ~93%          ~85%
  SnapKV          ~99.5%        ~98%          ~95%          ~88%
  StreamingLLM    ~99%          ~97%          ~94%          N/A (fixed window)
  DuoAttention    ~99.8%        ~99%          ~97%          ~92%
  PyramidKV       ~99.5%        ~98%          ~95%          ~89%

Note: Numbers are approximate; exact values depend heavily on task type,
model size, and sequence length. Reasoning tasks degrade faster than
retrieval tasks.
```

---

## Summary

| Concept | What it does |
|---|---|
| Token eviction | Discard low-importance tokens to bound cache size |
| Heavy hitters | Small fraction of tokens that receive most attention |
| Attention sinks | First tokens always receive high attention (structural artifact) |
| H2O | Cumulative-score eviction — keep top-scoring tokens |
| StreamingLLM | Sinks + sliding window — enables infinite streaming |
| SnapKV | Predict important tokens from observation window at prefill time |
| PyramidKV | Layer-wise budget: larger cache for early layers, smaller for late |
| ScissorHands | Evict transient tokens; keep only persistently-attended ones |
| DuoAttention | Split heads into retrieval (full cache) vs streaming (sparse cache) |
