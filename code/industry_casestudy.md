The Hidden Tax of Fast Inference — Profiling and Optimizing KV Cache Behavior in a Legal Document AI System

Step 1 of 4

The Problem
Industry
Legal Technology (LegalTech) — Document Drafting and Review Automation

The legal industry processes an enormous volume of long-form, high-stakes text. Contracts, regulatory filings, litigation briefs, due diligence reports — these documents routinely run to tens of thousands of tokens. Unlike consumer chatbots where a user asks a short question and receives a short answer, legal AI systems operate in a regime where context is everything: a single contract clause only makes sense in light of every clause that came before it. This makes legal AI one of the most demanding deployment environments for large language model inference, and one of the most commercially important. The global legal technology market is projected to exceed $50 billion by 2027, with AI-assisted drafting representing the fastest-growing segment.

Company Profile
Lex Machina Legal Systems (not to be confused with the litigation analytics company of the same name — this is a fictional entity for this case study)

Founded: 2021, San Francisco, CA
Team Size: 68 employees — 19 ML engineers, 12 backend engineers, 8 product managers, 11 attorneys serving as domain advisors, and the rest across sales, ops, and design
Funding Stage: Series B, $47M raised, led by a tier-1 enterprise SaaS fund
Key Products:
LexDraft: An AI drafting assistant that completes, continues, and revises contract sections in real time as attorneys type
LexReview: A batch analysis pipeline that flags risk clauses, inconsistencies, and missing provisions across uploaded contract sets
LexCompare: A retrieval-augmented generation tool that finds precedent language from an internal clause library and proposes alternatives
The company's flagship product, LexDraft, is deployed as a browser plugin integrated into document editors used by 340 enterprise law firms and corporate legal departments. It serves approximately 4,200 active daily users, predominantly senior associates and partners billing at rates between $400 and $1,200 per hour.

Business Challenge
LexDraft's core loop is generative: a user pauses mid-sentence, and the model proposes a continuation, clause by clause, in real time. The user experience is expected to mirror the feel of a fast autocomplete — the model must respond within roughly 300 milliseconds for the first token and sustain a generation cadence that keeps up with reading speed for subsequent tokens. The underlying model is a fine-tuned 7-billion parameter transformer that the team has hosted on a cluster of NVIDIA A100-80GB GPUs.

Recently, as the company signed several large enterprise accounts — including one Fortune 50 financial services company with a legal team generating documents with average context lengths of 6,000–12,000 tokens — the inference system has begun to crack.

Specifically, the team is observing the following:

First-token latency (TTFT) is spiking on long documents. At 2,000-token contexts, TTFT sits around 210 ms, which is acceptable. At 8,000-token contexts, TTFT is regularly exceeding 1.4 seconds, causing the UI to show a loading spinner — a UX death sentence for a tool targeting time-pressured attorneys.

Memory exhaustion at high concurrency. During peak hours (9:00–11:00 AM EST, when east coast legal teams start their workday), the serving cluster is crashing with out-of-memory (OOM) errors. Analysis shows that the KV cache for simultaneous long-context sessions is consuming the bulk of the 80 GB VRAM budget, leaving too little room for model weights and activations.

GPU utilization is paradoxically low during the crashes. The monitoring dashboard shows GPU compute utilization hovering around 22–35% at the exact moments when latency is highest and OOM errors are occurring. This counterintuitive observation — the GPU is idle yet also overwhelmed — is the central mystery the ML team needs to solve.

The engineering team initially proposed simply adding more GPUs, but the CFO flagged that the current GPU cluster already costs approximately $38,000 per month in cloud compute. Doubling the cluster to solve a problem caused by inefficient memory usage is not an acceptable answer without a rigorous technical justification and an exploration of software-level fixes first.

Why It Matters
The stakes are both direct and compounding:

Revenue at risk: The Fortune 50 financial services client alone represents $2.1M ARR. Their IT leadership has sent a formal notice that sustained latency issues will trigger a contract review clause. Three other large accounts, totaling $1.8M ARR, are watching closely.
User trust erosion: Attorneys are a notoriously exacting user base. A single bad experience — a spinning cursor at a critical drafting moment — is enough to break the habit loop that makes the tool sticky. Retention data shows that users who experience TTFT above 600 ms on three consecutive sessions have a 40% higher 30-day churn rate.
Competitive exposure: Two well-funded competitors (Harvey AI and CoCounsel) are actively pitching the same enterprise accounts. Speed is a top-three evaluation criterion in every sales cycle the company has documented.
Engineering credibility: The team needs to demonstrate internally that they understand why the system is behaving this way before they can propose a defensible fix. An expensive but misdiagnosed solution (throwing more GPUs at a memory bandwidth problem) would be far worse than the current situation.
Constraints
Constraint	Detail
Latency	TTFT ≤ 300 ms at p95 across all context lengths up to 8,000 tokens; sustained generation ≥ 15 tokens/sec
Compute budget	Monthly GPU spend must not exceed $45,000 — a soft cap set by the CFO pending Series C fundraising
Concurrency	Must support at least 80 simultaneous long-context sessions per A100 node
Data privacy	Client contracts are strictly confidential. No actual client data may leave the company's VPC. All profiling must use synthetic or publicly licensed data.
Compliance	SOC 2 Type II and GDPR Article 28 compliance — inference infrastructure must maintain audit logs; no third-party API calls with client data
Model constraints	The core 7B fine-tuned model is frozen (the fine-tuning budget for this quarter is spent). Architectural changes to the model itself are out of scope. Serving-layer optimizations only.
Deployment environment	Self-hosted on AWS p4d.24xlarge instances (8 × A100 40GB each) behind an internal API gateway; no model-as-a-service providers

Step 2 of 4

Technical Formulation
Problem Type
This case study is not, at its core, a modeling problem — it is a systems-level inference optimization problem that requires understanding the computational and memory dynamics of transformer autoregressive generation. The goal is not to train a better model but to understand, profile, and reason about the hardware-level behavior of a deployed model, specifically the interaction between the KV cache mechanism and the roofline model of GPU computation.

More precisely, the problem can be decomposed into two coupled sub-problems:

A performance characterization problem: Given a fixed model architecture and hardware, how does inference throughput and memory consumption scale as a function of sequence length, batch size, and KV cache configuration? This requires building a mental (and then empirical) model of the system's behavior rooted in the roofline model.

A resource allocation problem: Given a fixed VRAM budget, how should available memory be allocated between model weights, KV cache, and activations to maximize concurrent session throughput while satisfying latency constraints?

The reason this framing — systems characterization over model improvement — is the right one is precisely because the team's profiling evidence (low GPU utilization + OOM errors) is the telltale signature of memory-bound, memory-capacity-constrained inference. You cannot train your way out of a bandwidth problem.

Input Specification
For this profiling and optimization study, the primary inputs are synthetic legal documents drawn from the Multi-LegalPile dataset, a large multilingual corpus of public domain legal texts published under CC-BY-4.0 by Niklaus et al. (2023), containing contracts, legislation, court decisions, and regulatory filings. Documents are tokenized using the model's existing BPE tokenizer (vocabulary size 32,000).

Each inference request carries the following structure:

Context tokens: A prefix sequence of length n tokens (ranging from 256 to 12,000 in our profiling sweep), representing the document already written
Generation length: m tokens to be generated autoregressively (set to 128 tokens per request in profiling experiments, mimicking a clause-completion task)
Batch dimension: B simultaneous requests processed in a single forward pass (ranging from 1 to 64 in the profiling sweep)
The dimensionality that matters for hardware analysis is not the semantic content of the tokens but their count. What drives memory and compute cost is the tuple (n, m, B, d_model, n_heads, n_layers), where the model-architecture parameters are fixed (d_model = 4,096, n_heads = 32, n_layers = 32, head dimension = 128, consistent with a standard 7B-class architecture).

Output Specification
The outputs of the profiling study are measurement curves and derived quantities, not model predictions. Specifically:

Throughput curves: Tokens generated per second as a function of (n, B), reported separately for prefill phase and decode phase
Memory occupancy curves: Peak VRAM consumption in GB as a function of (n, B), broken down into (a) static model weight footprint, (b) KV cache footprint, (c) activation buffer footprint
Arithmetic intensity estimates: Operations per byte for prefill vs. decode, plotted against the A100's roofline to visually classify each regime
Latency decomposition: TTFT (dominated by prefill cost) and inter-token latency (dominated by decode cost with loaded KV cache), as functions of n
These outputs collectively constitute a system characterization report that the engineering team can use to reason about optimization levers.

Mathematical Foundation
To reason about this problem from first principles, we need to build up the relevant math in layers. Let us start with what the KV cache is doing at a mechanical level, then compute its costs.

Attention and the Source of Redundant Computation

In standard multi-head self-attention, for a sequence of n tokens, each token i produces three vectors: a query qᵢ ∈ ℝᵈ, a key kᵢ ∈ ℝᵈ, and a value vᵢ ∈ ℝᵈ (where d is the head dimension). The attention output for token i is:

oᵢ = Σⱼ softmax(qᵢᵀkⱼ / √d) · vⱼ

During autoregressive generation, when we generate token n+1, we compute its query qₙ₊₁, and we need to attend over all previous keys and values (k₁, v₁), ..., (kₙ, vₙ). In naive inference (no caching), we recompute all of these keys and values from scratch at every generation step. This means at step t, we are performing O(t²) attention operations — not because the new token needs to do anything different, but because we are wastefully discarding intermediate computations that haven't changed.

The fundamental insight of the KV cache is: the keys and values for all previous tokens are deterministic functions of those tokens and the model weights, and neither changes between generation steps. There is no mathematical reason to recompute them. We can compute them once during the prefill phase (when we process the full context), store them in VRAM, and simply look them up during each decode step.

Memory Cost of the KV Cache — A First-Principles Derivation

For a model with L transformer layers, H attention heads, head dimension d, and a context of n tokens, the KV cache stores:

For each layer: one key matrix of shape (n, H, d) and one value matrix of shape (n, H, d)
Total elements: 2 × L × n × H × d
In fp16 (2 bytes per parameter): Memory = 2 × 2 × L × n × H × d bytes
Substituting our 7B model parameters (L = 32, H = 32, d = 128):

KV Cache Memory = 4 × 32 × 32 × 128 × n bytes = 524,288 × n bytes ≈ 0.5 MB per token

For a context of n = 8,000 tokens: KV cache ≈ 4 GB per session. For 80 concurrent sessions: KV cache ≈ 320 GB — far exceeding the 40 GB VRAM of a single A100. This is the root cause of the OOM errors. The math is unambiguous: the team cannot simply keep increasing context length and concurrency without a principled strategy for managing this memory.

The Roofline Model — Formalizing the Compute vs. Bandwidth Tension

The roofline model provides a clean framework for understanding why low GPU utilization coexists with high latency. Define:

Peak Compute (π): The GPU's maximum arithmetic throughput, in floating-point operations per second (FLOP/s). For A100-40GB: π ≈ 312 TFLOP/s (fp16)
Peak Bandwidth (β): The GPU's maximum memory bandwidth, in bytes per second. For A100-40GB: β ≈ 2 TB/s
Ridge Point (π/β): The arithmetic intensity threshold separating compute-bound from memory-bound regimes. For A100-40GB: π/β ≈ 156 FLOP/byte
The achievable performance P for any computation with arithmetic intensity I (FLOP/byte) is:

P = min(π, β × I)

Now let us compute the arithmetic intensity of the decode step with KV cache. At each decode step:

We move the KV cache for the current context from VRAM: roughly 0.5 MB × n bytes of data movement
We perform attention computations over those n keys/values: roughly 2 × n × d FLOP per head, summed over H heads and L layers: 2 × n × H × d × L ≈ 2n × 32 × 128 × 32 ≈ 262,144 × n FLOP
Arithmetic intensity = (262,144 × n) / (524,288 × n) ≈ 0.5 FLOP/byte

This is 312 times below the ridge point. The GPU is performing half a floating-point operation per byte it moves. It is spending the vast majority of its time waiting for data — the classic memory-bound signature. The "low GPU utilization" the team is seeing is not utilization of the memory bus; it is utilization of the arithmetic units, which are indeed starving.

This derivation reveals a key and non-obvious insight: the arithmetic intensity of decode with KV cache is independent of sequence length n. The numerator and denominator both scale linearly with n and cancel out. What changes with longer sequences is not the intensity but the absolute volume of data that must be moved, which increases latency even at constant intensity.

Loss Function
Because this is a systems profiling and optimization study rather than a model training exercise, there is no conventional ML loss function in the sense of a differentiable objective minimized during gradient descent. However, the study does have a formal objective that can be written rigorously.

We are optimizing a constrained resource allocation problem. Define the following variables for a given inference serving configuration:

C(n, B): Peak VRAM consumption (GB) as a function of context length n and batch size B
T_TTFT(n): First-token latency (ms) as a function of context length n
Γ(n, B): Sustained token throughput (tokens/sec) for a batch of B sessions with context length n
The engineering objective is:

Maximize Γ(n, B)

Subject to:

C(n, B) ≤ 38 GB (reserving 2 GB headroom on a 40 GB A100)
T_TTFT(n) ≤ 300 ms for all n ∈ [256, 8000]
n_concurrent ≥ 80 sessions per node (as required by peak-hour demand)
The study's contribution is to characterize the feasible region of this constraint set empirically — to produce the C, T_TTFT, and Γ curves as functions of (n, B) — and then identify which optimization techniques (KV cache quantization, sliding window attention, paged attention, continuous batching) shift the feasible boundary most favorably.

This framing is important because it makes explicit that there is no single free parameter to tune — the team must navigate tradeoffs among latency, concurrency, and memory, and the right operating point depends on the actual distribution of document lengths and concurrency patterns in production.

Evaluation Metrics
Metric	Definition	Target	Why It Matters
TTFT at p95	95th-percentile first-token latency	≤ 300 ms at n=8,000	Directly determines perceived responsiveness; p95 rather than mean because attorneys experience worst-case, not average
Inter-token latency	Mean time between successive generated tokens	≤ 67 ms (≥15 tok/s)	Determines whether generation "keeps up" with reading speed; below 67ms is imperceptible as lag
Max concurrent sessions	Maximum B such that C(n=4000, B) ≤ 38 GB	≥ 80	Peak-hour demand requirement from capacity planning
Memory efficiency ratio	KV cache bytes consumed per token per session	Minimize	Derived metric that compares configurations (e.g., fp16 vs. fp8 KV cache)
Arithmetic intensity	FLOP/byte for prefill and decode phases separately	Plotted against roofline	Diagnostic metric; tells us whether a given change moves us toward or away from the ridge point
GPU utilization	Fraction of peak arithmetic throughput achieved	Increases toward ridge point = good	Low values confirm memory-bound regime and validate the diagnosis
The distinction between TTFT and inter-token latency is crucial and often collapsed carelessly. TTFT is dominated by the prefill phase — processing the existing n-token context in a single forward pass. Because prefill operates on many tokens simultaneously, it is more compute-intensive and thus closer to the ridge point on the roofline. Inter-token latency is dominated by the decode phase — a single token being generated using the cached KV state — which is deeply memory-bound. These two phases respond differently to the same optimization, and conflating them leads to misdiagnosis.

Baseline
The naive inference baseline is autoregressive generation without any KV caching. In this setting, every generation step reprocesses the full context from scratch. The arithmetic cost per generation step for a context of n tokens is O(n²) in the attention layers (all n × n attention scores must be computed) plus O(n) in the feed-forward layers. For n = 8,000 tokens, a 7B model, and 128 generation steps, the total FLOP count is approximately:

2 × n² × d_model × L = 2 × (8000)² × 4096 × 32 ≈ 16.7 × 10¹² FLOP per generation step

At 312 TFLOP/s (peak A100 throughput), this would require at minimum 53 ms per step from arithmetic alone — already above our 67 ms inter-token budget, and this ignores memory movement costs entirely. In practice, on hardware that is not operating at peak efficiency, measured latency for naive inference at n = 8,000 exceeds 800 ms per token. This is approximately 12× too slow for the inter-token requirement and makes the product entirely unusable at long contexts.

The KV cache reduces the per-step arithmetic cost from O(n²) to O(n) — the new token's attention computation is O(n), not O(n²), because the quadratic all-pairs computation was done once during prefill. This is the hero phase of the story. The baseline is not just insufficient; it is catastrophically insufficient for the use case, making the KV cache not an optimization but a prerequisite.

Why This Concept
The KV cache is the central concept here for reasons that emerge directly from the mathematical structure of the problem, not from convention. The transformer's attention mechanism has a structural property: the key and value representations of all past tokens are functions of those tokens alone (given fixed model weights) and do not change when new tokens are appended. This is not true of, say, recurrent networks, where the hidden state is a function of all preceding tokens and must be recomputed as context grows. For transformers specifically, the causal attention mask guarantees that token i's key and value vectors are identical regardless of how many tokens follow it. This is the mathematical justification for caching.

But the case study does not stop at "KV cache good." The roofline model is the necessary second lens because it explains the tradeoff the cache introduces. By trading computation for memory storage, the KV cache converts a compute-bound operation into a memory-bound one. Understanding this conversion — quantifying the arithmetic intensity before and after, locating both operating points on the roofline, and understanding how the ridge point is a property of the specific hardware — is what distinguishes an engineer who can optimize inference from one who merely knows that "KV cache makes things faster."

The practical optimization techniques explored in this case study (KV quantization to fp8, grouped-query attention as a model-level intervention for future fine-tuning rounds, paged attention for memory fragmentation, continuous batching for concurrency) all become interpretable through the same roofline lens: each one either moves the operating point closer to the ridge point, reduces the absolute volume of data moved, or increases the effective batch size to amortize memory traffic.

Step 3 of 4

Build It
The full implementation — including data acquisition, model design, training loops, evaluation, and error analysis — is provided as a hands-on Google Colab notebook with guided TODO exercises.

Step 4 of 4

Production Design
Architecture Diagram
The production inference system is organized as three logical tiers connected by an internal service mesh.

Tier 1 — API Gateway and Request Management: Incoming HTTPS requests from the LexDraft browser plugin arrive at an API gateway layer (NGINX + custom middleware). The gateway performs authentication via JWT tokens (tied to the law firm's enterprise SSO), validates request schema, strips any personally identifying metadata before the request reaches the GPU tier, and enqueues requests into a Redis-backed request priority queue. Requests from interactive sessions (real-time clause completion) are tagged with HIGH priority; batch review requests from LexReview are tagged NORMAL. The gateway also performs token counting on the incoming context using a fast CPU-side tokenizer, which informs the routing logic in Tier 2.

Tier 2 — Intelligent Request Router: A lightweight router service reads from the priority queue and makes two decisions: (a) which GPU node to route the request to, based on current KV cache occupancy per node reported by a metrics sidecar, and (b) whether the request should use the standard context window or trigger a context compression step (for documents above 10,000 tokens, a sliding window with key-sentence retention is applied to trim the context to 8,192 tokens before GPU submission). The router reports its decisions to a Prometheus metrics sink.

Tier 3 — GPU Inference Nodes: Each inference node runs a vLLM server instance with PagedAttention enabled. A node consists of one p4d.24xlarge instance (8 × A100-40GB) running 8 vLLM worker processes in tensor-parallel mode (tensor parallelism degree = 4, meaning each model copy spans 4 GPUs, allowing 2 model copies per node). Each vLLM instance maintains its own paged KV cache block pool, configured with a KV cache dtype of fp8 and a maximum sequence length of 8,192 tokens. The KV cache block size is set to 16 tokens per page — small enough for fine-grained memory reclamation but large enough to amortize page-table overhead. Completed streaming token responses are routed back through the API gateway to the client via server-sent events (SSE), which allows the frontend to begin rendering text as soon as the first token is ready rather than waiting for the full completion.

Supporting Services: A PostgreSQL database stores session metadata (session ID, user ID, document ID, token counts, latency measurements — never document content). A model registry (MLflow) stores model version metadata and evaluation checksums. A Grafana dashboard aggregates Prometheus metrics. All services run within a private AWS VPC; no component has a public-facing endpoint except the API gateway behind a WAF.

API Design
The primary inference endpoint follows a streaming REST design rather than gRPC, chosen for compatibility with the browser-based plugin architecture (gRPC is less ergonomic for browser clients without a proxy).

Endpoint: POST /v1/draft/complete

Request Schema:

{
  "session_id": string (UUID, ties to client-side session for KV cache warm-up),
  "context": string (raw document text, max 32,000 characters),
  "max_new_tokens": integer (default: 128, max: 256),
  "temperature": float (default: 0.3 — low temperature for legal precision),
  "stream": boolean (default: true)
}
Response Schema (streaming SSE events):

event: token
data: { "token": string, "token_id": int, "logprob": float, "latency_ms": float }

event: done
data: { "total_tokens": int, "ttft_ms": float, "total_latency_ms": float, "finish_reason": string }

event: error
data: { "code": string, "message": string }
The session_id field is important: for documents being actively edited in a session, the router uses this ID to route subsequent requests to the same vLLM worker, enabling prefix caching — if the context prefix hasn't changed between requests (the user added two words since the last completion was requested), the vLLM worker can reuse the already-computed KV cache for the unchanged portion rather than rerunning prefill from scratch. This effectively makes TTFT nearly instant for incremental edits, which is the dominant use pattern in LexDraft.

Internal Health Endpoint: GET /v1/internal/health returns per-node KV cache utilization, queue depth, VRAM headroom, and current active sessions count. Used by the router and monitored by the alerting system.

Serving Infrastructure
The production serving stack is built on vLLM v0.4+ with the following configuration rationale:

PagedAttention: Eliminates KV cache fragmentation. In a naive implementation, a session that requests n=4,000 tokens of context must pre-allocate a contiguous VRAM block for the full 4,000-token KV cache at session start, even if only 500 tokens have been generated so far. This leads to severe internal fragmentation — VRAM is reserved but not used. PagedAttention allocates KV cache in fixed-size pages on demand, exactly as a CPU operating system handles virtual memory. This can increase effective concurrent session capacity by 2–3× compared to a naive allocation scheme on the same hardware.

Tensor Parallelism = 4: The 7B model's weight matrices are sharded across 4 GPUs. This is the minimum parallelism that fits the model in 40 GB VRAM with headroom for KV cache (model weights in fp16 ≈ 14 GB; with 4-GPU tensor parallelism, each GPU holds ≈ 3.5 GB of weights, leaving ≈ 36.5 GB for KV cache pages). Tensor parallelism beyond degree 4 for a 7B model introduces communication overhead that degrades latency without memory benefit.

Continuous Batching: Rather than processing requests in fixed-batch synchronized rounds (which means slow requests hold up fast ones), vLLM uses iteration-level scheduling — at each decode step, if any session in the current batch finishes, a new session from the queue is immediately inserted into the freed slot. This dramatically improves GPU utilization under heterogeneous request lengths (which is the norm in legal workloads).

Scaling Strategy: Horizontal scaling via additional p4d.24xlarge nodes managed by an auto-scaling group with custom CloudWatch metrics. The scaling trigger is KV cache utilization across the fleet exceeding 70% (not CPU/GPU utilization, which is the wrong metric for a memory-bound workload). Scale-up lag is approximately 8 minutes (instance cold start + model weight loading), which is insufficient for sudden demand spikes. Pre-warming is handled by maintaining a minimum fleet of 2 active nodes at all times during business hours (6 AM–10 PM EST), scaling down to 1 node overnight.

Latency Budget
For an end-to-end request with n = 4,096 input tokens and 128 generated tokens, the latency decomposition target is:

Component	Allocated Latency	Notes
Network ingress + TLS handshake	5–15 ms	Variable with geography; p95 budget 15 ms
JWT validation + queue insertion	2 ms	CPU-bound, negligible
Token count check + routing	3 ms	Fast regex tokenizer on CPU
Prefill forward pass	80–120 ms	The KV cache is populated here; this IS the TTFT
First token transmission (SSE)	5 ms	The user sees text start appearing here
Decode (128 tokens, ~2.5 ms/token)	320 ms	Total generation time after first token
Response finalization + metadata write	5 ms	Async; does not block the stream
Total end-to-end	420–465 ms	Well within acceptable UX window
TTFT = prefill + routing = ~100–140 ms at p50, target ≤ 300 ms at p95. The headroom between p50 and the p95 budget accommodates queueing under concurrency spikes. If TTFT consistently approaches the 300 ms ceiling, that is the signal to scale out the fleet — not to optimize further in software.

Monitoring
The monitoring stack uses Prometheus for metric collection, Grafana for visualization, and PagerDuty for alerting.

Primary Metrics (reported every 10 seconds per node):

llm_ttft_ms (histogram with p50, p95, p99 percentiles, bucketed by context length range)
llm_tokens_per_second (gauge, per active session)
kv_cache_utilization_pct (gauge per node — the single most important operational metric)
active_sessions (gauge per node)
queue_depth (gauge)
vram_used_gb (gauge, broken down by model weights, KV cache, activations)
oom_events_total (counter — should be exactly 0 in healthy operation)
Alerting Rules:

P1 (PagerDuty call): kv_cache_utilization_pct > 90 for more than 60 seconds; oom_events_total increments; llm_ttft_ms[p95] > 500 ms sustained for 2 minutes
P2 (Slack alert): kv_cache_utilization_pct > 75 (pre-scale warning); queue_depth > 20; llm_tokens_per_second < 10 for any active session (suggests a stalled decode)
P3 (dashboard only): llm_ttft_ms[p95] trending upward over a 24-hour window
The critical insight embedded in this monitoring design is that KV cache utilization is the leading indicator for all latency and OOM issues. By the time TTFT degrades, the cache is already near saturation. Alerting on cache utilization at 75% — before degradation occurs — gives the autoscaler time to provision new capacity before users feel it.

Model Drift Detection
Unlike classification models, generative LLM drift is subtle and difficult to detect because there is no ground-truth label to compare outputs against in real time. The team uses a multi-layer approach:

Layer 1 — Input Distribution Monitoring: Track the distribution of context lengths, request arrival rates by time of day, and the fraction of requests falling into each document type category (commercial contract, employment agreement, NDA, etc.) over rolling 7-day windows. A KL divergence alert fires if the weekly input distribution diverges significantly from the baseline distribution observed during initial deployment. This catches client-mix shifts (e.g., a new large client with a different document profile joining) that might push the system outside its profiled operating range.

Layer 2 — Output Quality Proxy Metrics: Track the distribution of output token entropies (a proxy for confidence), the fraction of generations truncated by a repetition detection heuristic (a known failure mode for long-context LLM generation), and the average self-BLEU of consecutive clause completions (high self-BLEU suggests degenerate, repetitive output). These are logged to a time-series store and monitored for trend breaks.

Layer 3 — Human-in-the-Loop Sampling: A random 0.5% of completed sessions are flagged for review by the in-house attorney advisory team, who rate the quality of three randomly selected completions per session on a 1–5 scale. This produces approximately 21 human-rated samples per day — sufficient for a control chart analysis and a weekly quality trend report, but not so burdensome as to overload the advisory team.

Response: If Layer 3 quality scores fall below 3.5/5.0 for two consecutive weeks, a model refresh evaluation is triggered, examining whether the degradation is due to model drift versus serving infrastructure issues versus changing document characteristics.

A/B Testing
Before any optimization change (e.g., switching from fp16 to fp8 KV cache, enabling prefix caching, changing the sliding window compression threshold) is deployed to 100% of traffic, it undergoes a staged A/B test.

Design: Traffic is split 80/20 (control/treatment) at the session level — once a session is assigned to a condition, all requests within that session see the same configuration. Session-level assignment prevents a user from experiencing inconsistent behavior within a single document editing session. The minimum detectable effect is set at 15 ms improvement in median TTFT, requiring approximately 1,200 sessions per arm (based on observed TTFT standard deviation of ~45 ms at n=4,096) for 80% power at α=0.05.

Primary Metric: Median TTFT (speed improvement being tested)

Guardrail Metrics (must not degrade):

Attorney-rated output quality (sampled as above; must not fall below the control condition's score at p=0.1)
Session abandonment rate (a user who closes the plugin within 5 seconds of a completion being offered — a behavioral proxy for quality rejection)
Error rate (any 5xx response)
Statistical Analysis: Two-sided Mann-Whitney U test for the TTFT metric (non-parametric, as TTFT is right-skewed). Chi-squared test for abandonment rate. Tests are evaluated after both a minimum sample size and a minimum calendar duration of 5 business days have been met (to account for day-of-week variation in usage patterns).

CI/CD for ML
Because the model weights are frozen for this optimization study, the CI/CD pipeline focuses on the serving configuration and infrastructure layer rather than model retraining.

Pipeline Stages:

Code Commit → Integration Tests: A commit to the inference serving configuration repository triggers automated integration tests that spin up a single-GPU Docker container, run 50 profiling requests across the standard length buckets, and assert that measured TTFT and memory footprint match expected values within a 10% tolerance. Tests run in approximately 12 minutes on a single A10G Spot instance.

Model Configuration Validation Gate: A dedicated validation stage runs the attorney-advisory team's curated test set of 200 clause-completion pairs through the new configuration, computing reference-free quality metrics (perplexity against a held-out legal corpus, repetition rate) and comparing against a stored baseline checkpoint. A configuration fails this gate if quality metrics degrade by more than 2% on any metric.

Canary Deployment: Passing configurations are deployed to a single node handling 5% of production traffic for 24 hours. Automated rollback triggers if any P1 alert fires during the canary window.

Progressive Rollout: Following a clean canary, the configuration is rolled out to 25% → 50% → 100% of the fleet in 6-hour increments, with automated pauses at each stage if guardrail metrics breach defined thresholds.

Post-Deploy Verification: 48 hours after full deployment, an automated report compares pre- and post-deploy distributions for all primary and guardrail metrics and posts to the ML team's Slack channel.

Cost Analysis
Configuration	GPUs Required (Peak)	Monthly Cost	TTFT at n=8K (p95)	Notes
Current (broken)	6 × A100-40GB	$38,000	>1,400 ms	OOM errors at peak; fp16 KV cache, naive allocation
vLLM + fp16 KV + PagedAttention	4 × A100-40GB	$25,500	~280 ms	Meets TTFT requirement; still tight on concurrency
vLLM + fp8 KV + PagedAttention + GQA	4 × A100-40GB	$25,500	~210 ms	Meets all requirements with comfortable headroom
+ Prefix Caching	4 × A100-40GB	$25,500	~45 ms (incremental edits)	Near-instant TTFT for the dominant use pattern
Costs are estimated at AWS on-demand pricing for p4d.24xlarge instances at $32.77/hour, prorated for 2-instance always-on baseline (business hours) plus 2-instance autoscale capacity (peak hours only, averaging ~10 hours/day weekdays). Spot instance pricing for the autoscale fleet could reduce variable costs by an additional 40–60%, but Spot interruption risk during peak hours requires careful handling in the serving framework — vLLM's checkpoint-and-resume functionality mitigates but does not eliminate this risk.

The headline result: by diagnosing the problem correctly as a memory-bandwidth and memory-capacity issue rather than a compute insufficiency, the team can reduce monthly infrastructure costs by approximately $12,500 per month while simultaneously meeting all performance requirements — representing a $150,000 annual saving that more than justifies the engineering investment in this analysis.