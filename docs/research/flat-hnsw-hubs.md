# Flat-HNSW / "Hubs" Paper

**Authors:** Blaise Munyampirwa (Argmax Inc., Mountain View, CA), Vihan Lakshman (MIT CSAIL, Cambridge, MA), Benjamin Coleman (Google DeepMind, Mountain View, CA)
**arXiv:** 2412.01940v3 (3 Jul 2025)
**Paper:** https://arxiv.org/abs/2412.01940

## Problem Statement

HNSW's namesake feature is its hierarchical skip-list-style layering: searches start at a sparse top layer and progressively descend through denser layers, theoretically reducing search complexity from O(n) to O(log n). The paper challenges whether this hierarchy is actually necessary. Its central research question is explicit: "Can we achieve the same performance on large-scale benchmarks with simply a flat navigable small world graph?" Prior ablations (Lin & Zhao 2019; Coleman et al. 2022) hinted the hierarchy only helps at low dimensions (d < 32), but those studies used few real-world datasets. The paper fills the gap with an exhaustive benchmark on 13 standard datasets spanning 1M to 100M vectors, demonstrating that a flat graph matches HNSW on recall and latency while reducing memory and code complexity. The work also proposes a mechanistic explanation — the Hub Highway Hypothesis — for *why* the upper layers are redundant in high-dimensional data.

## Mathematical Foundations

**Hub definition (quoted precisely, p. 6):** "Hubness is a property of high-dimensional metric spaces where a small subset of points (the 'hubs') occur a disproportionate number of times in the near-neighbor lists of other points in the dataset (Radovanovic et al., 2010). In other words, a small fraction of nodes are highly connected to other points in the near-neighbor graph."

Operationalized via the k-occurrence count N_k(x_i) = Σ_{j≠i} 𝟙[x_i ∈ k-NN(x_j)], with skewness S_{N_k} = 𝔼[(N_k − μ_{N_k})³] / σ_{N_k}³ measuring hubness (p. 6).

**Hub-Highway Hypothesis (verbatim, p. 5):** "In high-dimensional metric spaces, k-NN proximity graphs form a highway routing structure where a small subset of nodes are well-connected and heavily traversed, particularly in the early stages of graph search."

Three empirical claims (p. 5): (1) some nodes are visited much more frequently than others, explained by hubness; (2) these hubs form a well-connected subgraph (the "highway network"); (3) queries visit hubs early before exploring less-traversed neighborhoods. The paper does not introduce new navigability theory; it reuses the small-world framework of Travers & Milgram (1977) and Watts & Strogatz (1998), and the hubness framework of Radovanovic et al. (2010), with concentration-of-measure (Talagrand 1994) cited as the geometric cause of hub formation in high d.

## Algorithmic Methods

**FlatNav construction:** The paper extracts the bottom (densest) layer of an hnswlib-built HNSW index and re-runs HNSW's greedy search heuristic (ef_construction=100, M=32) over that flat graph via a new library called `flatnav` (https://github.com/BlaiseMuhirwa/flatnav). No novel graph-construction algorithm appears; the construction path is intentionally the same hnswlib code path, with only the search restricted to one layer. The paper also confirms that FlatNav built *from scratch* (no hierarchy even during construction) gives identical results (Appendix E.1, p. 15).

**Search procedure:** Standard HNSW greedy beam search on a single graph layer (Algorithm 2 verbatim, p. 13), with ef_search=200, m=32, k=100. The only difference from hierarchical HNSW is the absence of the layer descent loop.

## Complexity Analysis

The paper does not give formal complexity bounds; it argues empirically. The central memory claim (Table 6, p. 16): peak index construction memory drops by 38% on BigANN-100M (183 GB → 113 GB), 39% on Yandex DEEP-100M (100 GB → 60.7 GB), and 18% on Microsoft SpaceV-100M (104 GB → 85.5 GB) relative to hnswlib. This comes from skipping the upper-layer node lists and per-layer dynamically-allocated edge storage. Search hop count is not separately reported as a metric — the authors treat it as subsumed by latency at matched recall (Figures 2–5). No theoretical O(n) vs O(log n) argument is advanced or refuted; the paper explicitly disclaims theoretical bounds in §D.2 (p. 15): "a principled understanding of this phenomenon from theoretical grounds is still lacking."

## Experimental Setup and Key Results

**Datasets (Table 4, p. 14):** BigANN-100M (d=128), Microsoft SpaceV-100M (d=100), Yandex DEEP-100M (d=96), Yandex Text-to-Image-100M (d=200), GloVe (d∈{25,50,100,200}), NYTimes (d=256), GIST (d=960), SIFT (d=128), MNIST (d=784), DEEP1B (d=96), plus IID Normal synthetic at d∈{16,32,64,128,256,1024,1536}. BigANN-100M used the top-10M/100M subset for ground truth.

**Hardware:** AWS c6i.8xlarge (Intel Ice Lake, 64 GB) for ≤100M-vector datasets; AMD EPYC 9J14 96-core with 1 TB RAM for 100M-scale runs.

**Headline result (Figures 2–5):** Across all 13 datasets at matched R100@100 recall, FlatNav's p50 and p99 latency curves overlap HNSW's. The paper's own wording: "FlatNav achieves nearly identical performance to hnswlib" (p. 4) and "no consistent and discernible gap between FlatNav and HNSW in both the median and tail latency cases" (p. 4). For synthetic low-d data (d ∈ {4, 8, 16, 32}, Figure 9 p. 14) the hierarchy reproduces the speedup reported by Lin & Zhao (2019) — i.e., the hierarchy *does* help when d < 32.

**Highway node prevalence (Figure 8, p. 8):** Queries spend 5–10% of early search steps in hub nodes, with the fraction scaling with dataset hubness (GIST highest, GloVe lowest).

**Memory:** as cited above; 18–39% peak construction-memory reduction.

**LLM extension (Appendix F.1, p. 15):** MSMARCO embeddings (all-MiniLM-L6-v2, d=384) show long-tailed access distributions consistent with the hypothesis on real retrieval workloads.

## Implications for LibraVDB

The paper is directly relevant to LibraVDB's `internal/index/hnsw` work, given the recent commits (ae14121 nomic dimension benchmarks, 3e993ee unroll candidate heap batch admission, 416a3e3 document candidate heap default, a641575 trim construction scalar overhead, b63875c batched neon distances) which focus on optimizing HNSW construction and search. Several questions follow from Munyampirwa et al.:

- **Dimensionality dependency:** LibraVDB benchmarks at d=768 (nomic) and likely higher. The paper's flat-graph parity holds for d ∈ [96, 1536], so the operating regime is squarely in the "hierarchy is redundant" zone. Our optimization effort on upper-layer bookkeeping may be paying for nothing.
- **Flat-graph construction as a baseline:** A natural experiment is to extract LibraVDB's layer-0 graph post-construction and re-run our search benchmark against it. If p50/p99 at R100@100 matches, the upper-layer code (entry-point selection, layer descent, per-layer edge storage) becomes deletable. Conversely, the new unrolled candidate-heap admission (commit 3e993ee) might actually be the *more* impactful lever if it speeds up the base layer where most hops occur.
- **Memory savings:** With TB-scale roadmaps, an 18–39% peak-construction-memory reduction is operationally significant; the savings come from dropping dynamically-allocated per-layer link storage and the visited-node list allocated per layer during multithreaded construction.
- **No new algorithm to copy:** the paper's contribution is empirical and explanatory, not algorithmic. The "FlatNav" construction is just hnswlib's existing code path restricted to one layer.

## Critical Analysis / Open Questions

- **Dataset dependence:** The strongest flat-graph parity results are on hnswlib-built graphs; the paper acknowledges (Appendix E, p. 15) that hnswlib has features FlatNav omits, and that "differences in code may account for a significant part of the peak memory usage differences." The memory numbers should be read as upper bounds.
- **Scope of claims:** "No discernible difference" is at the paper's chosen operating point (M=32, ef_construction=100, ef_search=200, k=100, R100@100). The claim has not been tested at smaller M (where pruning is tighter) or at very high recall (R1000@1000), nor for filtered/constrained search, nor under concurrent inserts.
- **Skipped edge cases:** billion-scale datasets, billion-scale indexes at 1B vectors, and the largest BigANN datasets (1B vectors, ~1.5 TB RAM) are explicitly excluded (p. 4). The paper does not address updates, deletes, or dynamic-graph workloads.
- **Construction-time hierarchy:** The paper confirms (§3, Appendix E.1) that even using the hierarchy *only during construction* and discarding it for search gives identical results. This isolates the question to "is our entry-point selection / layer descent helping?" — the answer in the paper is no, for d ≥ 96.
- **Open theoretical question (Appendix D.2):** the authors explicitly note that a theoretical latency bound for query q on graph G=(V,E) over hub nodes H is open; SIMD and hardware optimizations make asymptotic bounds uninformative. This is consistent with our hardware-sympathy framing in CLAUDE.md.
- **Hubness vs preferential attachment (Appendix F.3):** for L2 datasets, R² of node access count on insertion order is ≤ 8.7%; for angular datasets up to 24.6%. The paper concludes metric-space hubness, not insertion-order preferential attachment, drives the highway. Worth replicating on our workload to confirm.
- **What the paper does not address:** filtered ANN, hybrid sparse-dense retrieval, out-of-distribution queries, adversarial workloads, and dynamic indexing — all of which matter for a production vector database and none of which the paper evaluates.