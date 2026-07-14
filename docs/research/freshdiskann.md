# FreshDiskANN

**Authors:** Aditi Singh, Suhas Jayaram Subramanya, Ravishankar Krishnaswamy, Harsha Vardhan Simhadri
**Venue/Year:** arXiv preprint, May 2021 (cs.IR)
**arXiv:** 2105.09613
**Paper:** https://arxiv.org/abs/2105.09613
**Code:** https://github.com/microsoft/DiskANN

## Problem Statement

FreshDiskANN addresses the *fresh-ANNS* problem: maintaining a graph-based approximate nearest neighbor index over a corpus that is continuously changing via inserts and deletes, while preserving search quality (recall) and serving low-latency queries. Existing state-of-the-art graph indices (HNSW, NSG, Vamana) are static — every change degrades recall because aggressive pruning policies produce sparse graphs that lose navigability. Industry practice today is to periodically rebuild these indices from scratch, which is prohibitively expensive (1.5–2 hours per rebuild on a dedicated 48-core machine for a 100M-point HNSW; three rebuilds would be needed to maintain even six-hourly freshness on a billion-point index).

The paper's explicit goal is to support thousands of real-time inserts and deletes per second on a single commodity machine (128 GB RAM, 2 TB SSD), sustain > 95% 5-recall@5, and search in milliseconds — reducing deployment cost by 5–10x versus existing approaches. It is the first graph-based ANNS to deliver real-time freshness at billion-point scale without re-indexing.

## Mathematical Foundations

**Notation.** Dataset P of n points in R^d with pairwise distance d(·,·) (Euclidean or cosine). Directed graph G = (P, E). For node p, N_out(p) and N_in(p) denote out-edges and in-edges. Distance d(p, q) = ||x_p − x_q||.

**Definition 1.1 (k-recall@k).** For query q, G ⊆ P is the true k-NN set and X = |X ∩ L| / k is the per-query recall where L is the search output. Reported value is the average over all queries. 5-recall@5 > 95% is the target quality bar.

**Navigability.** A greedy search from start node s must converge to a locally-optimal node p* satisfying d(p*, q) ≤ d(p, q) for all p in N_out(p*). Algorithm 1 (GreedySearch) maintains a candidate set L (size ≤ l_search) and visited set V, expanding the closest unvisited node to the query until L is exhausted, then returning the k best from V.

**α-RNG property (the paper's key structural insight).** For α > 1, an edge (p, p') exists only if there is no edge (p, p'') with d(p'', p') < d(p, p') · (1/α) such that d(p, p'') < d(p, p'). The interpretation: edges are kept only when detouring through a neighbor gives significant geometric progress. Larger α yields denser, more navigable graphs. The paper observes that using α > 1 in both construction and pruning is what preserves recall under repeated updates.

## Algorithmic Methods

The system has two layers: FreshVamana (in-memory index with safe update rules) and FreshDiskANN (SSD-resident design with StreamingMerge consolidation).

**Algorithm 1: GreedySearch(s, x_q, k, L)** (verbatim, p. 4). Initialize L ← {s}, V ← ∅. While L \ V ≠ ∅: p* ← argmin_{p in L \ V} ||x_p − x_q||; L ← L ∪ N_out(p*); V ← V ∪ {p*}; if |L| > l, prune L to retain closest l points to x_q. Return closest k points from V.

**Algorithm 2: Insert(x_p, s, L, α, R)** (verbatim, p. 6). Run GreedySearch(s, x_p, 1, L) to get visited V. Set N_out(p) ← RobustPrune(V, α, R). For each j in N_out(p): if |N_out(j) ∪ {p}| > R then RobustPrune(N_out(j) ∪ {p}, α, R), else N_out(j) ← N_out(j) ∪ {p}.

**Algorithm 3: RobustPrune(p, V, α, R)** (verbatim, p. 6). V ← (V ∪ N_out(p)) \ {p}; N_out(p) ← ∅. While V ≠ ∅: p* ← argmin_{p' in V} d(p, p'); N_out(p) ← N_out(p) ∪ {p*}; if |N_out(p)| = R then break; for p' in V: if α·d(p*, p') ≤ d(p, p') then remove p' from V.

**Algorithm 4: Delete(L_D, R, α)** (verbatim, p. 6). For each p in P \ L_D where N_out(p) ∩ L_D ≠ ∅: D ← N_out(p) ∩ L_D; C ← N_out(p) \ D; for v in D: C ← C ∪ N_out(v); C ← C \ D; N_out(p) ← RobustPrune(C, α, R).

The crucial update rule (Section 4.2) is that on deletion, edges from neighbors of deleted nodes are *repaired* by adding edges (p', p'') where (p', p) was an in-edge and (p, p'') an out-edge, then pruned with α-RNG. This is the difference from naive Delete Policy A (just remove all incident edges — recall collapses) and Delete Policy B (add back edges but without α-RNG pruning — still degrades).

**Lazy deletion + DeleteList.** Rather than running Algorithm 4 eagerly on each delete, deletes accumulate in a DeleteList; search time skips DeleteList members. Once deletes reach ~1–10% of index size, Delete Consolidation runs Algorithm 4 in batch — trivially parallel via prefix-sum consolidation.

**FreshDiskANN architecture (Section 5).** Split index into (i) Long-Term Index (LTI): SSD-resident Vamana graph storing PQ-compressed vectors (32 bytes/point), serving real-time searches, and (ii) Templndex: one or more in-memory FreshVamana instances holding recent inserts. DeleteList filters deleted points from search results. RW-Templndex (mutable, log-backed) accepts new inserts; periodically converted to RO-Templndex (read-only, snapshot) and snapshotted to disk. Search fans out across LTI + all RO/RW-Templndex, merges results.

**StreamingMerge (Section 5.3).** Background merge invoked when total Templndex memory exceeds a threshold. Three phases:
1. **Delete phase** — load LTI points block-by-block from SSD, execute Algorithm 4 over deletes D using pre-stored PQ codes for distance approximations.
2. **Insert phase** — GreedySearch on the intermediate-LTI for each of N new points to populate V, run RobustPrune, store per-point neighbor changes Δ(p') in an in-memory data structure rather than writing immediately (avoids hot-spotting the SSD).
3. **Patch phase** — fetch each affected block from SSD, add the in-block entries from Δ for each p in that block, prune if |N_out(p) ∪ Δ(p)| > R, write the block back.

I/O cost is two sequential passes over the SSD-resident LTI. Due to α-RNG, the GreedySearch in the insert phase issues a small number of random 4KB reads per inserted point (~100 ≈ 75·l_search).

## Complexity Analysis

**Memory footprint.** Templndex in-memory footprint ~ 13 GB for 30M points; total Templndex bounded to ~26 GB (128 B/point vector + 256 B/point neighborhood at R=64). LTI for 800M points fits in ~24 GB (compressed vectors only). StreamingMerge workspace peak ~100 GB. Scales linearly to roughly 125 GB for a billion-point index.

**I/O complexity of StreamingMerge.** Two sequential passes over the SSD-resident data structure in Delete and Patch phases. Insert phase uses ~100 random 4 KB reads per inserted point. Total write IO cost: O(|D|·R²) over the set of deleted points D in the delete phase (bounded in expectation over random deletes). Insert and Patch phases are linear in N.

**Insert/delete throughput.** Steady state: 1800 inserts/sec + 1800 deletes/sec on a single machine at 95% 5-recall@5. Burst: up to 40,000 inserts/sec under short bursts (while the next StreamingMerge is in progress). Search latency: well under 20 ms mean, sustaining 1000 searches/sec at 95% 5-recall@5.

**Recall stability.** Across 50 cycles of 5%/10%/50% deletion-and-reinsertion (SIFT1M, Deep1M, GIST1M, SIFT100M) the 5-recall@5 stays at or above the Cycle-0 baseline. StreamingMerge experiments show recall stabilizing after ~20 cycles on 80M-point SIFT100M subset (10% deletes/inserts) and similar convergence on 800M-point SIFT1B.

## Experimental Setup and Key Results

**Hardware.** mem-mc: 64-vcore E64d_v4 Azure VM (latency/recall experiments). ssd-mc: bare-metal 2x Xeon 8160 (96 threads), 3.2 TB Samsung PM1725a PCIe SSD (full FreshDiskANN evaluation).

**Datasets.** SIFT1M, GIST1M, DEEP1M (1M points each, 128/960/96 dims); SIFT100M (100M subset of SIFT1B in float32); full SIFT1B (~1B points, 128 dims) for the billion-scale evaluation. Default params R = 64, L_s = 75, α = 1.2.

**Headline results.**
- Memory footprint stays under 128 GB throughout the week-long run on 800M points.
- StreamingMerge merges a 10% change into a billion-scale index in ~10% of full rebuild time.
- Steady-state 1800 inserts/sec + 1800 deletes/sec, bursts up to 40,000 inserts/sec.
- Insert/delete latency < 1 ms; background merge uncovers latency.
- 1000 searches/sec sustained at 95% 5-recall@5 with mean latency < 20 ms over the latest index content.

Compared to industry practice (periodically rebuilding HNSW), this is a 5–10x cost reduction at billion-point scale with real-time freshness.

## Implications for LibraVDB

The α-RNG update rules (Algorithms 2 and 4) are directly relevant: they explain why naive "delete + reconnect" fails on dynamic HNSW/NSG and what the correct repair rule looks like. The DeleteList + Delete Consolidation split is a clean separation between the live-path hot set and the amortized batch repair — a pattern that translates to any LSM-like merge topology. The StreamingMerge three-phase design (Delete → Insert → Patch) with in-memory Δ accumulation between sequential SSD passes is a template for SSD-friendly bulk updates of a graph index. The PQ-compressed-distance trick during merge (avoiding raw-vector decode on the bulk path) is a load-bearing optimization at billion-point scale. Their choice to split LTI (SSD) from Templndex (RAM) maps cleanly onto the LSM/Templndex architecture discussed in the roadmap notes — the storage layer's compaction cycle is a natural host for StreamingMerge.

## Critical Analysis / Open Questions

The α-RNG stability claim is demonstrated on SIFT, DEEP, and GIST with α ∈ {1.0, 1.1, 1.2, 1.3}; α = 1.0 still degrades over cycles (Figure 3). The stability guarantee is empirical — no formal proof that α-RNG edges remain navigable under arbitrary delete/insert sequences. The paper does not address adversarial or bursty non-random workloads (all experiments use uniform-random point selection from the spare pool).

The 800M-point evaluation is *not* a full billion — the SIFT1B dataset is reduced to ~800M with a 200M spare pool. The billion-point claim extrapolates from memory/throughput measurements rather than from a complete end-to-end run. The Templndex cap of 30M points per StreamingMerge interval is a real operational constraint; tuning it against query mix and tolerable staleness is left to the deployer.

Crash recovery relies on a redo-log and snapshotting Templndex instances — recovery time on a 5M-point Templndex is ~2.5 minutes vs. ~16 minutes for 30M, so the parameter M is a tuning lever with operational consequences the paper only sketches. The paper does not discuss concurrent-write safety on the LTI itself (only the Templndex), nor what happens to in-flight StreamingMerge if the process dies mid-merge — partial state recovery is asserted but not stress-tested in the paper.

The work is Euclidean / cosine only; inner-product and non-metric spaces are out of scope. The paper does not quantify the recall cost of using PQ-compressed distances during StreamingMerge — the ~1% recall drop visible in Figure 4 is acknowledged but not isolated.