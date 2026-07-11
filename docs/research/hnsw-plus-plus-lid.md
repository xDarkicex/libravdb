# HNSW++: Dual-Branch HNSW with Skip Bridges and LID-Driven Optimization

## Header

- **Title:** Dual-Branch HNSW Approach with Skip Bridges and LID-Driven Optimization (HNSW++)
- **Authors:** Hy Nguyen, Nguyen Hung Nguyen, Nguyen Linh Bao Nguyen, Srikanth Thudumu, Hung Du, Rajesh Vasa, Kon Mouzakis
- **Affiliations:** Deakin University (Applied Artificial Intelligence Institute), Swinburne University of Technology
- **Venue:** arXiv preprint, posted 25 Apr 2025
- **arXiv ID:** 2501.13992 (v2)
- **URL:** https://arxiv.org/abs/2501.13992
- **Note on code:** The paper mentions a C++ implementation with a Python extension, but no official repository URL is given. Bibliographic links to FAISS, NMSLIB, PyNNDescent, and Annoy are listed only as comparators, not as this work's code. Reproduction is therefore not straightforward.

---

## 1. Problem Statement

HNSW (Malkov & Yashunin 2020) is the de-facto state-of-the-art for approximate nearest neighbor (ANN) search on dense vectors. The paper claims HNSW has two limitations in practice:

1. **Local optima from random insertion + greedy search.** A node inserted at a randomly chosen level may end up in a sparsely connected region, and the standard greedy search has no mechanism to escape suboptimal basins. This produces "stuck in local minima" and "disconnected cluster" pathologies (Figure 1, paper p. 2).
2. **Non-logarithmic traversal in high dimensions.** Each layer is traversed exhaustively, so the O(log n) bound is hard to realize in high-dimensional spaces (the paper cites Lin & Zhao 2019 for this claim).

HNSW++ proposes three coupled changes:
- A **dual-branch HNSW structure** (partition the indexed set by index parity, run two parallel hierarchies).
- **LID-based insertion ordering** that preferentially sends high-LID (sparse / outlying) nodes to upper layers.
- **LID-thresholded skip bridges** that allow a query to jump from any layer directly to layer 0 when conditions are met.

The authors claim the combination gives better recall, faster construction, and equal or better query time — i.e., "no trade-offs."

---

## 2. Mathematical Foundations

### 2.1 Local Intrinsic Dimensionality (Levina-Bickel MLE)

LID is estimated using the Maximum Likelihood Estimation of intrinsic dimension (Levina & Bickel 2004). For a query point `x` and its `k` nearest neighbors at Euclidean distances `d_1 <= d_2 <= ... <= d_k` (with `d_i` being the distance to the `i`-th nearest neighbor, and `d_k` the distance to the `k`-th), the MLE estimate is:

```
LID(x) = -1 / ( (1 / (k-1)) * sum_{i=1}^{k-1} log(d_k / d_i) )
```

(Equation 2, paper p. 4.) Note the `1/(k-1)` factor — the sum runs to `k-1`, not `k`. The implementation uses `eConstructor = 128` neighbors for LID estimation (paper p. 7).

### 2.2 Dual-Branch Search

The result for a query `q` is the merge of two independent searches, one per branch:

```
HNSW++(D) = Merge( S(q, L_1, exclude_set_1), S(q, L_2, exclude_set_2), k )
```

(Equation 1, paper p. 4.) `S(x, l, exclude_set)` is the layer-`l` greedy search on a branch excluding the other branch's results. `L_1` and `L_2` are the top layers of branches 1 and 2 respectively. `Merge` selects the top-`k` by distance to `q`.

The "exclude_set" mechanism (Figure 19, paper p. 17) is the procedural device that prevents the two branches from returning duplicate neighbors: when one branch's search hits layer 0, its result set is passed as `exclude_set` to the other branch's search.

### 2.3 Skip Bridge

A skip bridge is triggered when the current entry point `ep` of layer `L_l` has both a high LID and is sufficiently close to the query:

```
S_skip(q, L_l) = { S(q, 0, exclude_set)    if Jump(ep, q) is True
                 { S(q, L_l)               otherwise
```

(Equation 3, paper p. 5.) The jump predicate is:

```
Jump(x_i, x_q) = True   if LID(ep) > T AND d(ep, q) < eps
                 False  otherwise
```

(Equation 4, paper p. 5.) `T` is the LID threshold (normalized 0..1, see below); `eps` is a distance threshold meaning "near enough to be considered in the target neighborhood." The two-conjunct form — high LID AND close distance — is what makes the bridge both safe and effective: a high-LID point in the wrong part of space is *not* a good jump target.

### 2.4 LID Normalization (for layer assignment only)

```
normalized_LID(x) = (x - min(LID)) / (max(LID) - min(LID))
```

(Equation 5, paper p. 16.) This rescaling is used only in Algorithm 4 (assign_layer); the skip-bridge comparison uses the raw LID compared against threshold `T`. The paper does not explicitly state what numeric range `T` should take, but the text on p. 6 implies it is in normalized [0,1] space (matching Algorithm 5 output).

---

## 3. Algorithmic Methods

The paper provides four named algorithms in the appendix. All pseudocode below is **verbatim from the paper** (cited).

### 3.1 Algorithm 1 — Insert (paper p. 13, verbatim)

```
Input: hnsw - multilayer HNSW graph structure, q - new element (point) to be inserted,
       assigned_layers - mapping of element labels to assigned layers,
       assigned_branches - mapping of element labels to assigned branches,
       branch0, branch1 - two branches of the HNSW graph,
       base_layer - base layer of the HNSW graph
Output: Update hnsw by inserting element q

1:  Retrieve layer <- assigned_layers[q.label]
2:  Retrieve branch <- assigned_branches[q.label]
3:  if branch = 0 then
4:      branch0.setLevel(layer)
5:      branch0.setConnectState(layer != 0)
6:      branch0.addPoint(q, q.label)
7:      closest <- branch0.getClosestPoint()
8:  else
9:      branch1.setLevel(layer)
10:     branch1.setConnectState(layer != 0)
11:     branch1.addPoint(q, q.label)
12:     closest <- branch1.getClosestPoint()
13: end if
14: base_layer.setEnterpointNode(closest)
15: base_layer.addPoint(q, q.label)
16: return updated hnsw
```

Notable: insertion is *not* per-layer — the node is added once to its assigned branch at its assigned level, the closest node in that branch is promoted to the entry point, and the base layer (layer 0, shared) is updated once. This is structurally different from canonical HNSW which inserts at every layer below the assigned level. This means the upper layers of each branch in HNSW++ are *sparse and shallow* — only nodes assigned to those layers appear there.

### 3.2 Algorithm 4 — Assign Layer (paper p. 15, verbatim)

```
Input: topL - maximum number of layers, mL - normalization factor for level generation,
       normalized_LIDs - array of normalized local intrinsic dimensionalities
Output: assigned_layers - an array of (layer, branch) assignments for each node

1:  n <- length of normalized_LIDs
2:  branch0_size <- ceil(n/2), branch1_size <- floor(n/2)
3:  Initialize arrays expected_layer_size for both branches with size topL
4:  for each branch in (0, 1) do
5:      for each node in branch do
6:          layer_i <- max(unique(1 - log(random() * mL), topL - 1), 0)
7:          Increment expected_layer_size[branch][layer_i]
8:      end for
9:  end for
10: Sort indices of normalized_LIDs in descending order
11: Initialize assigned_layers with shape (n, 2) to hold (layer, branch)
12: Initialize current_layer_size for both branches to zero
13: current_branch <- 0
14: for each sorted index do
15:     branch <- current_branch
16:     for layer from topL - 1 down to 0 do
17:         if current_layer_size[branch][layer] < expected_layer_size[branch][layer] then
18:             Assign node to layer and branch in assigned_layers
19:             Increment current_layer_size[branch][layer]
20:             Break
21:         end if
22:     end for
23:     Alternate between branches (switch current_branch)
24:     If one branch is full, assign the remaining nodes to the other branch
25: end for
26: return assigned_layers
```

The two-stage design is worth noting: (a) compute the *expected* per-branch per-layer counts from the standard HNSW geometric distribution `floor(-log(uniform) * mL)`; (b) sort nodes by normalized LID descending and greedily place the highest-LID node into the deepest still-open slot. This is what implements the "outliers go to upper layers" claim — they get priority because they are sorted to the front of the queue.

### 3.3 Algorithm 5 — Normalize LIDs (paper p. 16, verbatim)

```
Input: lids - array of local intrinsic dimensionalities
Output: normalized_LIDs - array of normalized LIDs

1: min_lid <- minimum of lids
2: max_lid <- maximum of lids
3: normalized_LIDs <- (lids - min_lid) / (max_lid - min_lid)
4: return normalized_LIDs
```

Standard min-max scaling, no clipping or distribution assumptions.

### 3.4 Algorithm 2 — Search (paper p. 14, verbatim) and Algorithm 3 — Search-Layer (paper p. 15, verbatim)

Search runs two independent `SEARCH-LAYER` loops (one per branch), each descending from the top layer and calling `SEARCH-LAYER(q, W, ef, layer, 0, lid_threshold)`. Within a layer, after the `ef` nearest neighbors are gathered, the algorithm checks the "skip" condition: if `min(W, key=lambda x: distance(x, q))` is the nearest node and its LID is >= `lid_threshold`, return `skip = True` so the calling `SEARCH` (Algorithm 2) jumps `layer` directly to 0 and increments `skip_count`. Otherwise `skip = False` and the layer is traversed normally.

The final result is `W <- Top-k(W1 ∪ W2)`; `W1` and `W2` are the layer-0 results from each branch.

### 3.5 Construction Workflow (Section 3.3, paper p. 5-6)

Pre-computation step (offline, not part of insert):
1. Compute LID for every point via MLE over 128 nearest neighbors.
2. Normalize LIDs (Algorithm 5).
3. Assign each node to (layer, branch) (Algorithm 4) using descending normalized LID as the priority.

Insert step (Algorithm 1): O(log(N)) per insertion, claimed identical to vanilla HNSW. The heavy LID work is amortized once at indexing start.

---

## 4. Complexity Analysis

The paper's own analysis (Section 3.5, paper p. 6-7):

- **Search:** Each branch is O(log(N)) (same hierarchical bound as HNSW). Effective layers traversed is `L_total * (1 - P_skip)` where `P_skip` is the per-node probability of triggering a skip bridge. Merging is O(k log k) at the base layer — constant overhead. Total: O(log(N)) + O(k log k).
- **Construction:** Per-node `O(log(N))` from the layered insert, plus `O(M_max * log(N))` from the per-layer neighbor selection. Total: O(N * log(N)). The dual-branch halving is claimed to *reduce* constant factors in practice (each branch sees N/2 nodes).
- **LID pre-computation:** Not explicitly costed. For N points with `eConstructor = 128`, this is a one-time `O(N * 128 * d)` cost. The paper claims this is amortized and not on the insert hot path.

Caveat: the search complexity claim holds only if `P_skip` is bounded away from 1 (otherwise you skip everything and get brute force). The paper does not analyze worst-case `P_skip`, and the threshold `T` is presented as a hyperparameter with no default value given.

---

## 5. Experimental Setup and Key Results

### 5.1 Setup

- **Hardware:** AMD EPYC 7542, 32 cores / 64 threads, 80 GB RAM, 64-bit Debian OS.
- **Implementation:** HNSW++ in C++ with Python extension. Comparators: FAISS (IVFPQ), NMSLIB (HNSW), PyNNDescent, Annoy.
- **Datasets (Table 1, paper p. 8):**

  | Dataset  | d   | Space | Points  | LID Avg | LID Median | Type     |
  |----------|-----|-------|---------|---------|------------|----------|
  | GLOVE    | 100 | L2    | 11,000  | 31.94   | 30.52      | NLP      |
  | SIFT     | 128 | L2    | 11,000  | 14.75   | 14.81      | CV       |
  | RANDOM   | 100 | L2    | 11,000  | 42.75   | 42.54      | Synthetic|
  | DEEP     | 96  | L2    | 11,000  | 16.42   | 16.22      | CV       |
  | GIST     | 960 | L2    | 11,000  | 28.30   | 28.67      | CV       |
  | GAUSSIAN | 12  | L2    | 11,000  | 22.60   | 12.80      | Synthetic|

- **Graph size:** 10,000 construction points, 1,000 query points. This is *small* — two orders of magnitude below typical ANN benchmarks (SIFT-1M, GLOVE-1.2M, DEEP-1B).
- **LID hyperparameter:** `eConstructor = 128` for the MLE estimate.

### 5.2 Headline Results

From the abstract and Section 4.2:
- **Recall improvement:** up to 18% on NLP (GLOVE), up to 30% on CV (SIFT, GIST, DEEP) at recall@10.
- **Construction speedup:** "reducing the construction time by up to 20% while maintaining the inference speed."
- **Ablation ordering (highest to lowest impact):** 1) LID-based insertion, 2) dual-branch structure, 3) skip-bridge. This is the authors' own ranking from the ablation figures (Figures 10-13, paper p. 10).

### 5.3 Ablation (Figures 10-13, paper p. 10)

Four configurations tested:
- **Basic:** vanilla HNSW.
- **Multi-Branch:** two parallel hierarchies, no LID, no skip.
- **LID-Based:** LID-driven insertion only, single branch, no skip.
- **HNSW++:** all three combined.

Findings (paper text p. 10):
- On GLOVE specifically, LID-Based alone *outperforms* HNSW++ in accuracy/recall — i.e., the multi-branch and skip additions *hurt* on the NLP dataset. This contradicts the "no trade-offs" claim for at least one of the six datasets.
- Across the other five datasets, Multi-Branch and HNSW++ are the top two, nearly tied.
- **Construction time:** HNSW++ and Multi-Branch alternate as fastest, 16-20% faster than Basic. LID-Based is 18-22% *slower* than Basic — the LID priority sort is the dominant cost.
- **Query time:** Multi-Branch, Basic, LID-Based, HNSW++ all within 1-2% of each other on five of six datasets. On RANDOM only, Multi-Branch and LID-Based are noticeably slower than HNSW++/Basic.

### 5.4 Threshold Sensitivity (Figures 14, 15a, 15b, paper p. 12, 13)

- Number of skips increases monotonically as the LID threshold lowers (more nodes qualify).
- Recall and accuracy are essentially flat across threshold values for all six datasets.

This is presented as a positive: the threshold is a "free knob" for tuning speed. The flatness is also a weak signal — it means the skip bridge is being triggered on sufficiently many nodes at the lowest tested threshold that further lowering it would presumably degrade recall, but that regime was not tested.

---

## 6. Implications for LibraVDB

LibraVDB's HNSW implementation lives at `internal/index/hnsw/` with L2 and cosine distance and arm64 NEON assembly. Current recall@10 is already ~1.0 on tested corpora, and the active bottleneck is *construction time*.

**Worth investigating:**
1. **LID-based insertion order as a construction-time lever.** Algorithm 4's high-LID-to-upper-layers policy is the highest-impact single change in the paper. Even without the dual-branch and skip-bridge machinery, sorting inserts by descending normalized LID and placing them greedily into upper-layer slots could be tested against the current random-level assignment. The cost is one offline MLE pass over 128-NN per point (`O(N * 128 * d)`) plus a sort — cheap compared to the construction it accelerates.
2. **Layer-0 search convergence as a recall floor.** LibraVDB already runs SIMD-batched candidate heap evaluation on arm64. The HNSW++ skip-bridge is essentially a way to short-circuit upper-layer traversal when the current layer's nearest node is already close to the query — this is the same stopping condition an aggressive efSearch can express, but enforced at the layer-jump boundary. Worth measuring whether adding an LID-aware early-jump produces the claimed 1-2% inference improvement over current behavior.
3. **Multi-branch as a recall hedge for sparse regions.** The current bottleneck (construction) is probably not helped by halving the per-branch dataset — that *halves* the work each branch does but *doubles* the total. The paper claims the opposite, but their construction-time data is on 10K-point graphs and may not extrapolate.

**Probably not worth:**
- The skip-bridge mechanism itself: it adds a per-evaluate LID lookup that is not amortizable the way a one-time offline MLE is. The paper's own ablation puts it third in impact.
- Multi-branch as default: doubles memory and complicates the single-WAL edge model in `internal/storage/wal`. Only consider if a recall regression is observed on specific corpus shapes.

**Verification needed before adoption:**
- Reproduce the headline numbers on a corpus at least 100x larger than the paper's 10K points.
- Confirm the GLOVE contradiction (LID-only beats HNSW++ on NLP) holds or doesn't hold at scale.
- Measure the offline LID pre-computation cost against the claimed 20% construction speedup — these are likely comparable in magnitude and the paper does not account for them.

---

## 7. Critical Analysis / Open Questions

### 7.1 The "No Trade-offs" Claim

The paper's abstract asserts "We did not observe any trade-offs in our algorithm." The ablation section partially contradicts this:
- On GLOVE (NLP), LID-Based alone outperforms HNSW++ — the multi-branch and skip additions are net negative on that dataset.
- On RANDOM, Multi-Branch and LID-Based are noticeably slower than HNSW++ at query time.
- LID-Based is 18-22% *slower* than Basic on construction across all datasets, because the offline LID sort and MLE pass add more overhead than the dual-branch saves.

So the honest framing is: the combination of all three changes is roughly Pareto-dominant on five of six datasets, but each individual change has a measurable downside on at least one dataset. "No trade-offs" is a marketing summary, not a finding.

### 7.2 Recall Improvements — Consistent Across Datasets?

The paper reports 18% on GLOVE and 30% on CV. But the GLOVE number refers to *LID-Based* configuration (which beat HNSW++ there), and the CV number refers to *HNSW++*. The headline conflates the best per-dataset configuration, not a consistent single-configuration result. Across all six datasets, the *HNSW++ configuration* recall improvement over Basic is closer to 5-15%, with a wide variance.

### 7.3 Construction Speedup — Consistent?

20% construction speedup is reported "while maintaining inference speed." This is over Basic, not over each individual variant. Since LID-Based alone slows construction by 18-22%, the only way to net-positive 20% is the dual-branch halving dominating the LID overhead. The paper does not break out the additive vs. interaction effects.

### 7.4 Reproducibility

- **No code repository** is given in the paper. The text on p. 7 says "The HNSW++ code were implemented in C++ (and Python as extension in Appendix)." Appendix A.3 (paper p. 16) shows a Python implementation, but no URL.
- **No hyperparameter settings** for the LID threshold `T` or distance threshold `eps` are reported. These are load-bearing for the skip-bridge behavior and the construction-time/recall tradeoff.
- **No value of `mL`** (the HNSW level-generation normalization factor) is given. The default in the original HNSW paper is `mL = 1/ln(M)` where M is the max number of connections per node, but the paper does not state which M it uses.
- **Small corpus:** 10K construction / 1K query is not enough to characterize high-dimensional ANN behavior. Production HNSW evaluations use 1M-1B points. Recall curves at 10K are dominated by base-layer completeness, not by upper-layer topology.

### 7.5 LID Estimation Cost

The paper does not account for the cost of computing LID for every point in the dataset. With `eConstructor = 128` and `d` up to 960 (GIST), this is `128 * 960` multiply-adds per point, repeated for every point, plus a 128-NN search per point. For N = 10K this is negligible; for N = 10M it is ~1.2 trillion FLOPs of preprocessing. The 20% construction speedup must be net of this cost to be a fair comparison, and the paper does not include it in its construction-time measurement (or does not state whether it does).

### 7.6 Algorithmic Questions

- **Algorithm 1 inserts at the assigned layer only** — but the assign_layer in Algorithm 4 may assign layer > 0 to a node that also needs to be present at all lower layers (this is how canonical HNSW works). The paper does not state whether HNSW++ does or does not insert at every layer below the assigned one. The text on p. 5 says "search begins... proceeds to the next lower layer, using the previous layer's result as the entry point," which implies the canonical hierarchical structure. But Algorithm 1 only calls `addPoint` once per branch, which would *not* populate lower layers. This is either an algorithmic bug, an undocumented simplification, or a different design than canonical HNSW that makes the upper layers very sparse.
- **The skip bridge returns `S(q, 0, exclude_set)`** when triggered. This means a skip from layer L skips the *intermediate* layers but still does layer-0 search. The paper claims this makes "achieving O(b log n) complexity more feasible" (paper p. 3) — but the dominant layer-0 cost is unchanged, so the asymptotic improvement is illusory unless `ef` at intermediate layers is the bottleneck. On the test corpora (10K points), `ef` is small enough that this is not where time is spent.
- **Branch assignment is by node index** (the `q.label` lookup in Algorithm 1, lines 2 and 4-12 of the branch alternation logic in Algorithm 4). Two branches contain disjoint halves of the dataset. This means the dataset must be partitioned *before* insertion and insertion order matters for which branch each node lands in. This is a significant bookkeeping requirement that the paper does not address in the complexity analysis.

### 7.7 Summary Assessment

The core insight — that high-LID nodes are useful entry points and should preferentially occupy upper layers — is well-grounded in the prior work the paper cites (Houle et al. 2018, Amsaleg et al. 2015, Elliott & Clark 2024). The dual-branch and skip-bridge mechanisms are engineering elaborations on that insight. The reported numbers are small-scale and the headline framing oversells the consistency of the gains. **Adopt the LID insertion-ordering idea (Algorithm 4) as an offline precomputation; treat the dual-branch and skip-bridge as optional micro-optimizations to benchmark before integrating.**

---

## 8. Key References Tracked

- Levina & Bickel 2004 — LID MLE formula (Equation 2 in this paper).
- Hand, Mannila, Smyth 2001 — kNN LID estimation (cited as source of the MLE).
- Houle, Schubert, Zimek 2018 — LID and outlier correlation in similarity search.
- Elliott & Clark 2024 — insertion order effects on HNSW recall (up to 12.8 pp from insertion ordering alone).
- Lin & Zhao 2019 — critique of HNSW logarithmic complexity in high-D.
- Malkov & Yashunin 2020 — original HNSW paper.
