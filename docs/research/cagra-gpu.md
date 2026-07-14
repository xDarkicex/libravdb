# CAGRA

**Authors:** Hiroyuki Ootomo, Akira Naruse, Corey Nolet, Ray Wang, Tamas Feher, Yong Wang (NVIDIA)
**Venue/Year:** ICDE 2024 / arXiv 2023 (9 Jul 2024 timestamp on v2)
**arXiv:** 2308.15136
**Paper:** https://arxiv.org/abs/2308.15136
**Code:** https://github.com/NVIDIA/cuVS
**Docs:** https://docs.rapids.ai/api/cuvs/stable/neighbors/cagra/

## Problem Statement

Graph-based ANNS (proximity graph traversal with greedy refinement) has become the dominant accuracy/throughput regime but historically was designed around single-thread CPU execution. Existing CPU SOTA — HNSW — uses hierarchical NSW layers and a randomized insertion heuristic that serialize naturally: each step a single thread inserts, expands a search, and prunes. GPU architectures (A100/H100: >80 SMs, 32 threads/warp, SIMT execution, hundreds of GB/s HBM bandwidth, small per-SM shared memory) require structures built from the kernel up — fixed out-degree, no hierarchy, branching-friendly — or memory bandwidth and warp divergence dominate the runtime.

CAGRA frames the gap as: "not all applications can gain higher performance just by using the GPU. We have to be able to abstract parallelism from an algorithm and map it to the architecture." The two specific failure modes targeted: (1) graph construction is not designed for massively parallel atomics and bulk distance work; (2) search on existing graphs (GGNN, GANNS, SONG) achieves some throughput but their graphs remain CPU-shaped (irregular degrees, hierarchical layers) and warp-divergence-heavy.

The paper's working hypothesis: a fixed out-degree, directional, flat graph with strong connectivity (large 2-hop count N_2hop) outperforms HNSW on GPUs while remaining competitive on accuracy. If N_2hop is low the traversal reaches unreachable nodes, recall collapses, and throughput is wasted on coverage rather than ranking.

## Mathematical Foundations

Notation: dataset D = {x_1, ..., x_N} subset of R^n. k-NNS returns i_1, ..., i_k = k-argmin_i Distance(x_i, q). Distance(·, ·) is typically L2 or cosine. Recall@k = |U_ANNS ∩ U_NNS| / |U_NNS|, where U_NNS is the exact brute-force k-NN ground truth and U_ANNS is the algorithm result; the trade axis is recall vs QPS.

GPU primitives cited: 32-thread warp executes SIMT on a single SM-resident CTA; CTAs in turn are spread across SMs and active concurrently. Memory hierarchy is device HBM (largest, slowest) -> per-SM shared memory (low latency, tens of KB) -> per-thread registers (smallest, fastest). The whole graph design is constrained to live in HBM unless explicitly cached, so degree and edge layout directly decide bandwidth cost per hop.

Reachability in a directed graph is measured by counting strongly connected components (CCs) and by the 2-hop node count N_2hop(v) = |{u : ∃w s.t. v → w ∧ w → u}|. High N_2hop and few CCs are necessary (not sufficient) for high recall. The paper cites theoretical results to motivate the design: the weak CC count of a nearest-neighbor / random-kNN base graph is not guaranteed connected, which alone rules out a naive unprocessed kNN graph at scale.

## Algorithmic Methods

The pipeline decomposes into three GPU-friendly stages.

**Stage 1: Intermediate k-NN construction.** Build a fixed out-degree base graph via an ANN-descent style NN-descent update ([29]) — iterative neighbor-of-neighbor refinement — using exact distances under cosine/L2. This gives a baseline directional graph with degree R that already has the locality CACG needs; the authors explicitly avoid RNG-based NSG and NSW heuristics because they are sequential and degree-irregular.

**Stage 2: Pruning.** The heuristic fast-SG building on [32] keeps only the edges that improve reachability without inflating degree. The exact sort criteria are referenced to a separate per-node pruning paper rather than spelled out in sections I–III; the key claim is monotonic under pruning — connected prefix and N_2hop do not collapse — so recall is preserved while the per-hop distance cost is reduced.

**Stage 3: Search.** A flat single-layer greedy/beam search starting from random entry nodes (replacing HNSW's hierarchical entry-point selection). Because there is no layer hop, search parallelism is a single batched problem: queries process in parallel against the same graph with each thread block traversing a different query. The bulk of the work is inner-loop distance computation — exactly the workload a SIMT GPU handles efficiently when degrees are uniform.

The paper makes the design contrasts explicit. Non-hierarchy: "In the case of GPU, we can obtain compatible initial nodes by randomly picking some nodes and comparing their distances to the query, thus employing the high parallelism and memory bandwidth of GPU." Fixed out-degree: "By fixing the out-degree, we can utilize the massive parallelism of GPU effectively... it is better to expand the search space using all the available compute resources, as it won't increase the overall compute time." Directional: the graph is naturally unidirectional in their construction.

Pseudocode does not appear on pages 1–3 (referred to Section IV in-text). Numerical kernels — distance, sort, prune — are dispatched on tensor-core-friendly matrix shapes and orchestrated with software-warp splitting and forgettable hash table management to keep live registers in budget.

## Complexity Analysis

Construction is dominated by NN-descent iterations and per-iteration distance computation O(N · R · d) per pass; the GPU payoff comes from doing these as batched GEMM-equivalent operations rather than per-edge scalar work. Search cost per query is approximately L hops × R distance computations of length d, plus a beam-managed candidate set of size R; on a fixed-degree graph L is small (a handful of hops) so wall time is essentially (R · d) per query with batched amortization across queries.

Memory: adjacency list is N · R entries (R fixed, in contrast to HNSW whose per-node top-layer M_0 is small but upper-layer degrees are bounded separately). The paper argues this regularity is a feature on GPU because access patterns are predictable, so HBM bandwidth (the graph's adjacency reads and the vector reads during distance compute) stays saturated.

The headline throughput numbers: 2.2–27× faster construction than HNSW depending on dataset and recall target; in large-batch query, 90–95% recall regime, 33–77× faster than HNSW and 3.8–8.8× faster than SOTA GPU implementations; in single-query 3.4–53× faster than HNSW at 95% recall.

## Experimental Setup and Key Results

Datasets cited (full table is in later sections of the paper): 100M-scale vector corpora used for large-batch throughput; smaller benchmarks for single-query latency. Distance metrics are L2 and cosine. Baselines are HNSW (CPU SOTA) and existing GPU graph-ANN (GANNS, GGNN, SONG, plus FAISS-IVFPQ). Constructed with fixed out-degree parameter R tuned per recall target.

Key quantitative claims reproduced on page 1: construction 2.2–27× over HNSW; large-batch 33–77× over HNSW and 3.8–8.8× over GPU SOTA at 90–95% recall; single-query 3.4–53× over HNSW at 95% recall. Accuracy parity at fixed recall is implicitly claimed across the 90–95% operating range typical of production ANNS (where exact kNN is unnecessarily expensive).

Throughput dominance at large batch sizes reflects the GPU's amortization sweet spot — every query in a batch exploits the same in-flight distance work — while the smaller (but still meaningful) single-query advantage reflects the simpler heap and traversal logic compared to HNSW's per-layer priority queue.

## Implications for LibraVDB

Architecturally portable lessons, framed for review rather than directive:

The intermediate-then-prune split is hardware-agnostic. NN-descent to build a kNN base graph and a monotone pruning pass is a clean two-phase construction; both phases are parallel-friendly bulk-distance workloads (an SoA distance kernel, batched per-iteration) and both produce a single fixed-degree adjacency layout. On CPU a similar decomposition could be split across cores via a `runtime.WorkerPool`-shaped batch dispatch, treating one "iteration" as a work item rather than kernel-level parallelism.

Fixed out-degree is a discipline worth borrowing for cache pressure. Whether on GPU HBM or CPU L2/L3, irregular degrees yield irregular DRAM/cache-line fetches: better to spend extra explicit bits on dead-end neighbor slots than to allocate per-node variable-length adjacency slices. The RAM/cache friendliness argument survives the platform change.

Search without an explicit hierarchy is honest about what greedy refinement plus random entry points can deliver. The HNSW hierarchy is a recall shortcut; on any architecture it pays entry-point amortization costs and complicates the per-query state machine. A flat graph with strong N_2hop and a beam search is a simpler, more cache-local baseline — an attractive control point for A/B testing against the existing HNSW path in this codebase.

Distance kernel as the hot path. The paper's throughput advantage is fundamentally a claim that bulk L2/cosine on regular batches beats HNSW's per-hop scalar work plus heap management. This codebase's existing ARM NEON bulk distances (referenced in the recent commit history on `internal/util/simd`) are the right substrate to evaluate that claim at our scale; the experimental question is whether the same bulk-distance versus scalar-heap tradeoff that CAGRA exploits on HBM is also visible against L2/L3.

The GPU-specific surface — software warp splitting, forgettable hash tables, CTA scheduling — is not portable. Treat it as instantiated reference architecture rather than template.

## Critical Analysis / Open Questions

GPU-specific assumptions to flag for any cross-platform reuse: (1) constant out-degree is cheap on HBM but does not free CPU cache the way the paper implies; on x86/ARM a 32–64 wide neighbor slot per node x millions of nodes is real RAM. (2) Directional-only edges are acceptable because GPU re-traversal cost is hidden by throughput amortized over many queries; CPU single-query latency runs are more sensitive to asymmetric reachability. (3) Random entry-point selection trades per-query variance against insertion-time hierarchy costs; the variance penalty is mostly invisible in QPS-aggregated benchmarks but meaningful at p99 latency. (4) The bulk matrix-style distance dispatch requires work batches of ~hundreds to thousands of queries; CPU ANNS workloads are frequently interactive with small batches.

Open questions worth running: does CAGRA-style construction (NN-descent + monotone pruning) reduce end-to-end HNSW construction time in this codebase at 10K–1M vectors? Does flat-graph beam search match or beat HNSW on single-query p50/p99 latency at 90%+ recall, given that the current SIMD distance kernel already amortizes the inner loop? Are the reachability metrics (CC count, N_2hop) useful diagnostics here, or do they only predict GPU-batch behavior?

Scope of claims: the 33–77× headline is QPS at 90–95% recall on million–billion vector datasets with large batch. This is a competitive hardware paper; absolute ratios transfer imperfectly and the right local comparison is fixed-cost ablation, not headline reproduction.
