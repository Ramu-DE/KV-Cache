# KV Cache — Complete Learning Guide

Everything about KV Cache, from zero to production.

## How to Read This

Go in order. Each chapter builds on the last.

```
01_basics.md           ← Start here. What KV cache is and why it exists.
02_memory_and_size.md  ← How big it gets, why that matters, the math.
03_optimizations.md    ← GQA, PagedAttention, prefix caching, and more.
04_advanced_topics.md  ← Quantization, sliding window, MLA, ring attention.
05_systems_and_serving.md ← vLLM, multi-GPU, production deployment.
06_code_examples.py    ← Working PyTorch implementations you can run.
07_visual_cheatsheet.md ← Everything condensed onto one reference page.
```

## The One-Paragraph Summary

An LLM generates one token at a time. Without KV cache, it would
re-process the entire conversation history for each new token — O(n²)
cost. KV cache stores the Key and Value tensors computed for previous
tokens so only the new token needs processing — O(n) cost. The cache
lives in GPU VRAM, grows with each token, and is the primary reason
long-context inference is expensive. Modern systems optimize it through
smaller architectures (GQA), virtual memory (PagedAttention), shared
prefix reuse (prefix caching), and quantization (INT8/FP8).

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
