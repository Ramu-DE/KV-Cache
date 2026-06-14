# KV Cache — Complete Learning Guide

Everything about KV Cache, from zero to production.

## How to Read This

Go in order. Each chapter builds on the last.

```
FOUNDATIONS
  01_basics.md              ← Start here. What KV cache is and why it exists.
  02_memory_and_size.md     ← How big it gets, why that matters, the math.

CORE OPTIMIZATIONS
  03_optimizations.md       ← GQA, PagedAttention, prefix caching, Flash Attention.
  04_advanced_topics.md     ← Quantization, offloading, H2O, MLA, ring attention.

PRODUCTION SYSTEMS
  05_systems_and_serving.md ← vLLM, multi-GPU, disaggregated serving.

CODE
  06_code_examples.py       ← PyTorch: NaiveAttention, KVCache, GQA, SlidingWindow.
  demo_eviction.py          ← H2O, StreamingLLM, SnapKV eviction demos.
  run_demo.py               ← numpy: speed, memory, cache growth (CPU-only).
  run_demo_torch.py         ← PyTorch: tensor shapes, speed, GQA, prefix cache.
  simulate.py               ← Real-text simulation with actual sentences.

REFERENCE
  07_visual_cheatsheet.md   ← Everything on one page including 2024 systems.

NEW (2024-2025 RESEARCH)
  08_token_eviction.md      ← H2O, SnapKV, PyramidKV, ScissorHands, DuoAttention.
  09_systems_2024.md        ← SGLang/RadixAttention, vAttention, DistServe, Mooncake,
                              API prefix caching (Claude/OpenAI/Gemini), FastGen.
  10_emerging_2024_2025.md  ← Flash Attention 3, CLA, KVSharer, CacheGen, InfiniGen,
                              research taxonomy, open problems, full paper index.
```

## The One-Paragraph Summary

An LLM generates one token at a time. Without KV cache, it would
re-process the entire conversation history for each new token — O(n³)
total cost. KV cache stores the Key and Value tensors computed for
previous tokens so only the new token needs processing — O(n²) total.
The cache lives in GPU VRAM, grows with each token, and is the primary
reason long-context inference is expensive. Modern systems reduce this
through smaller architectures (GQA, MLA), virtual memory (PagedAttention,
vAttention), shared prefix reuse (RadixAttention/SGLang), token eviction
(H2O, SnapKV, DuoAttention), quantization (INT8/FP8), and disaggregated
serving (DistServe, Mooncake). Flash Attention 3 (2024) accelerates
prefill on H100 GPUs. All findings in chapters 8-10 are verified against
primary papers using adversarial fact-checking.

## Prerequisites

- Basic understanding of neural networks (what a layer is)
- Python (to run the code examples)
- PyTorch (optional, for code examples)

## Key Numbers to Remember

| Model      | Cache per token | 4K context | 128K context |
|------------|-----------------|------------|--------------|
| Llama-2 7B | ~0.5 MB         | ~2 GB      | ~64 GB       |
| Llama-2 13B| ~0.8 MB         | ~3.2 GB    | ~100 GB      |
| Llama-2 70B| ~2.5 MB         | ~10 GB     | ~320 GB      |
