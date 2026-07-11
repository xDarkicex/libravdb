# NSG — Navigating Spreading-out Graph

**Authors:** Cong Fu, Chao Xiang, Changxu Wang, Deng Cai (Zhejiang University)
**Venue/Year:** PVLDB 14(1), 2020
**arXiv:** 1707.00143
**Paper:** https://arxiv.org/abs/1707.00143
**Code:** https://github.com/ZJULearning/nsg

## Problem Statement

Graph-based ANNS achieves the best accuracy/throughput regime at million-scale, but earlier graph methods (RNG, HNSW, FANNG, DPG) lacked rigorous theoretical analysis and did not scale to billion-point databases. The paper attacks two coupled problems: (1) existing graphs have high indexing time (at least $O(n^2)$ naive, and even with NN-descent still $O(n \cdot n^{2-d/(2d+2)} + n^2 \log n + n^3)$ for the minimal MSNET), and (2) practical graph-construction methods cannot simultaneously optimize the four desirable properties the paper identifies:

1. **Connectivity** of the graph (every point reachable from a fixed entry).
2. **Low average out-degree** (small traversal fan-out).
3. **Short search path** (low hop count from entry to query).
4. **Small index size** (memory footprint).

Existing approaches sacrifice at least one: HNSW and NSG-style constructions improve (2)/(4) by aggressive edge reduction, but at the cost of long search paths; DPG (Fan et al.) attempts degree capping but loses theoretical support.

## Mathematical Foundations

### Notation

- $S \subseteq \mathbb{E}^d$ — finite point set, $|S| = n$.
- $G = (V, E)$ — directed graph with $V = S$.
- $\delta(p, q) = \|x_p - x_q\|$ — Euclidean distance.
- $B(p, r) = \{x \mid \delta(x, p) < r\}$ — open ball.
- $lune_{pq} = B(p, \delta(p, q)) \cap B(q, \delta(p, q))$ — lune (intersection of two equal-radius balls).

### Definitions

**Definition 3 (Monotonic Path).** A path $v_1, \dots, v_k$ from $p$ to $q$ in $G$ is *monotonic* if for every edge $(v_i, v_{i+1})$ we have $\delta(v_i, q) \geq \delta(v_{i+1}, q)$ — each step strictly decreases the distance to the target. Backtracking is therefore unnecessary in a monotonic-path graph; Algorithm 1 (greedy graph search) is guaranteed to find a closer node when one exists.

**Definition 4 (Monotonic Search Network / MSNET).** $G$ is a MSNET iff for every pair $p, q$ there exists at least one monotonic path from $p$ to $q$.

**Lemma 1.** A graph is a MSNET iff for every pair $p, q$ there exists at least one edge $\overrightarrow{pr}$ such that $r \in B(q, \delta(p, q))$.

**Theorem 1.** In a randomly constructed MSNET with uniform point distribution in a finite subspace $E^d$ and a single monotonic path, Algorithm 1 finds a monotonic path between any two nodes.

**Theorem 2 (search path length).** The expected length of a monotonic path from $p$ to $q$ in a MSNET is $O\!\left(n^{1/d} \cdot \frac{1}{\Delta r} \cdot \log n\right)$, where $\Delta r = \min\{|\delta(a,b) - \delta(a,c)|, |\delta(a,b) - \delta(b,c)|, |\delta(a,c) - \delta(b,c)|\}$ is the smallest non-isosceles triangle side-difference. As $n$ grows, $\Delta r \to 0$ slowly, making this close to $O(\log n)$.

**Definition 5 (MRNG).** A directed graph where an edge $\overrightarrow{pq}$ exists iff $\delta(\text{lune}_{pq} \cap S) = \emptyset$ (the lune between $p$ and $q$ contains no other point from $S$). MRNG inherits low search complexity from the MSNET family.

**Lemma 2.** The maximum degree of an MRNG in $\mathbb{E}^d$ is a constant independent of $n$ (depends only on dimension).

**Theorem 3.** An MRNG on a finite point set is a MSNET.

**Definition 6 (NNG).** A nearest neighbor graph: edge $\overrightarrow{pq}$ exists iff $q$ is the closest neighbor of $p$ in $S \setminus \{p\}$.

### The MRNG Edge Selection Intuition

For each candidate edge $\overrightarrow{pq}$, the MRNG rule checks whether the lune between $p$ and $q$ contains any other point. If it does, the edge is unnecessary because that intermediate point would have shorter routing toward $q$. This is strictly stronger than RNG (which checks triangle emptiness) and yields the monotonic-path property.

## Algorithmic Methods

### Algorithm 1: Search-on-Graph (verbatim, p. 2)

```
Algorithm 1 Search-on-Graph(G, p, q, l)
Require: graph G, start node p, query point q, candidate pool size l
Ensure: k nearest neighbors of q
 1: l ← 0; candidate pool S ← ∅
 2: S.add(p)
 3: while i < l do
 4:    i ← the index of the first unchecked node in S
 5:    mark p_i as checked
 6:    for all neighbour of p_i in G do
 7:        S.add(n)
 8:    end for
 9:    sort S in ascending order of the distance to q
10:    if S.size() > l, S.resize(l)
11: end while
12: return the first k nodes in S
```

### Algorithm 2: NSGBuild(G, l, m) (verbatim, p. 7)

```
Require: KNN Graph G, candidate pool size l for greedy search, max-out-degree m
Ensure: NSG with navigating node n
 1: calculate the centroid of the dataset
 2: r ← random node
 3: n ← Search-on-Graph(G, r, c, l)  // navigating node
 4: for all node v in G do
 5:    Search-on-Graph(G, n, v, l)
 6:    E ← all the nodes checked along the search
 7:    add v's nearest neighbors in G to E
 8:    sort E in the ascending order of the distance to v
 9:    result set R ← ∅, p_0 ← the closest node to v in E
10:    R.add(p_0)
11:    while !E.empty() && R.size() < m do
12:        p ← E.front()
13:        E.remove(E.front())
14:        for all node r in R do
15:            if edge pv conflicts with edge pr then
16:                break
17:            end if
18:        end for
19:        if no conflicts occurs then
20:            R.add(p)
21:        end if
22:    end while
23: end for
24: while True do
25:    build a tree with edges in NSG from root n with DFS
26:    if not all nodes linked to the tree then
27:        add an edge between one of the out-of-tree nodes and its closest in-tree neighbor (by algorithm 1)
28:    else
29:        break
30:    end if
31: end while
```

### NSG Construction (high-level)

1. **Approximate kNN graph** built with current state-of-the-art (nn-descent via Faiss on GPU).
2. **Approximate medoid** found by centroid + greedy search from the centroid on the kNN graph — this becomes the *Navigating Node* $n$, the unique fixed entry point for all queries.
3. **Per-node candidate generation.** For each node $p$, run Search-on-Graph from $n$ to $p$; record every visited node as a candidate. Add $p$'s exact NNG neighbors (Lemma 2 ensures this is at most a small constant).
4. **Edge selection by MRNG criterion.** Sort candidates by distance, greedily add edges that do not create a non-monotonic situation (the "conflicts" check on line 15). Bound out-degree by $m$.
5. **DFS tree spanning from $n$.** Any nodes not in the tree get linked to their closest in-tree neighbor by MRNG-style search, guaranteeing connectivity from the Navigating Node.

### Hyperparameters

- $l$ — beam width of Search-on-Graph during construction (typical: 100–200).
- $m$ — max out-degree (typical: 30–60). The paper uses $m = 30$ for memory-sensitive regimes.
- $C_{max}$ — max cluster size for partitioning in later PiPNN-like extensions.

## Complexity Analysis

- **Indexing complexity** of NSG is $O\!\left(k \cdot \frac{n \cdot d}{m} \cdot \log \frac{n}{d} \cdot \Delta r\right)$ where $k$ is per-point kNN cost. With nn-descent and $f(n) = n \log n$ (Faiss), total is $O(n^2 \log n + c \cdot n^2)$ for constants $c$, substantially smaller than MRNG's $O(n^2 \log n + n^2 / \Delta r)$ and MSNET's $O(n^2 \cdot n^{2/(d-1)} + n^2 \log n + n^3)$.
- **Search complexity** (Theorem 2 with $\Delta r$ in practice approximately constant for high dimension): $O(c \cdot n^{1/d} \log n)$ where $c$ is a constant — empirically very close to $O(\log n)$ in their experiments.
- **Memory:** $n \cdot m$ adjacency entries, where $m$ is the bounded out-degree. NSG index size is 1/2 to 1/3 of HNSW's per the paper's measurements (Table 2).

## Experimental Setup and Key Results

**Datasets (Table 1, p. 9):** SIFT1M (128-d, LID 12.9, 10K queries), GIST1M (960-d, LID 29.2, 1K queries), RAND4M (128-d, synthetic U(0,1), 4M points, 10K queries), GAUSS5M (128-d, synthetic Gaussian, 5M points, 10K queries).

**Baselines:** Flann (KD-trees, K-means trees), Annoy (randomized KD-trees), FALCONN (LSH), Faiss (IVFPQ), KGraph (NN-descent), Efanna (composite KD-tree + kNN graph), FANNG, HNSW, DPG.

**Key results (Tables 2–3, Figures 4–6):**

- **Sparsest graphs that beat all others on recall and search speed.** NSG has AOD (average out-degree) of 25.9–29.3 vs HNSW bottom-layer 26.3–151.9 across SIFT1M/GIST1M. MOD (max out-degree) 70 on both datasets. NNG recall (% of nodes linked to nearest neighbor) 98.1–99.4.
- **Lowest index size.** NSG is 1/2 to 1/3 the size of HNSW (the prior best).
- **Indexing speed:** NSG construction is fastest among graph-based methods, $t_1 + t_2$ times reported in Table 3 — significantly slower than non-graph methods but faster than HNSW on GIST1M (2.5× slower) and competitive on SIFT1M.
- **Dimensionality matters.** Performance gap widens at higher local intrinsic dimension (LID). On GIST1M (LID 29.2) NSG dominates by even larger margins than on SIFT1M (LID 12.9).
- **High-precision regime.** NSG is faster than serial scan at 99% precision on SIFT1M/GIST1M and matches serial scan at 99% on RAND4M/GAUSS5M — the gap to serial scan actually shrinks with high dimensionality.

**E-Commerce deployment:** NSG is integrated into Taobao (Alibaba) search engine at billion-node scale, deployed in production (paper §1, contribution 3).

## Implications for LibraVDB

**Architecturally portable observations:**

1. **Single navigating node vs hierarchy.** The NSG paper is one of the cleanest demonstrations that a flat navigable graph with a fixed entry point can match or exceed HNSW on the search-quality axis (a different paper — the Flat-HNSW hubs paper — extends this empirical case). For LibraVDB this is a concrete A/B target: extract `internal/index/hnsw/insert.go`'s base layer, run a one-shot NSG-style finalization pass with MRNG edge selection at the same max-degree $m = 32$ we currently use for `M_max0`, and measure p50/p99 against the hierarchical version on our standard datasets.

2. **Conflict check on edge insertion.** Lines 15–22 of Algorithm 2 implement a geometric non-conflict rule that is structurally similar to the α-RobustPrune inequality from DiskANN but framed differently (it checks whether adding $p$ would create a non-monotonic detour rather than whether $p$ is "covered" by an existing edge). Both aim to preserve the navigable property while keeping the out-degree bounded. The α-RobustPrune formulation is the more numerically tractable; the NSG formulation is the more geometrically motivated. Either could be substituted for our current neighbor heuristic.

3. **NN-descent as a kNN graph factory.** NSG's kNN graph construction uses nn-descent; the paper cites Faiss GPU but the algorithm itself is CPU-friendly. For an offline bulk-build mode (a future PiPNN-style path), building a dense approximate-kNN graph once and then pruning once is structurally cleaner than our current per-insertion beam search.

4. **DFS connectivity repair.** The tail loop (lines 24–31) ensures every node is reachable from $n$ by repairing any disconnected components. Our HNSW layer-0 already guarantees this in expectation, but a one-shot repair pass after construction would let us surface latent disconnected components rather than relying on the probabilistic guarantee.

## Critical Analysis / Open Questions

- **No formal convergence theorem.** The indexing complexity bounds assume the worst case; the paper itself notes (Bottom of p. 8) that $\Delta r$ depends on data distribution and is not constant in general. Empirically they observe constant-ish behavior on their evaluated datasets.
- **KNN-graph dependency.** Algorithm 2 starts from an approximate kNN graph built externally (nn-descent via Faiss). NSG inherits this construction cost. The paper acknowledges it could not run nn-descent on billion-scale datasets and used Faiss GPU; on CPU only, NSG's indexing time advantage over HNSW is less clear.
- **No billion-scale validation in the paper proper.** The "integrated into Taobao at billion-node scale" claim is a deployment note, not an experimental result in §4. The experimental evaluation is at 1M–5M points only.
- **Datasets limited to SIFT/GIST/synthetic.** All four evaluation datasets are CV descriptors or synthetic. Performance on text embeddings or contrastive learning outputs (which dominate modern vector workloads) is not measured here. The Flat-HNSW hubs paper later addresses this gap.
- **NNG rule $m$-bound is heuristic.** Lemma 2 gives a dimension-dependent bound on MRNG max degree, but NSG itself caps at $m = 30$ heuristically. The paper does not justify the specific choice of $m$.
- **Conflict test precision.** Line 15's "edge $pv$ conflicts with edge $pr$" is described geometrically but not given as a closed-form inequality. Implementing from the paper requires careful reading of §3.3 and the related-work discussion.
- **LID measurement methodology.** The paper uses the Levina–Bickel MLE LID estimator with $k = 100$ from a sample of base points. The reported numbers are sensitive to $k$; the paper does not sweep this.

## Sections Where the Paper Is Thin or Unclear

- Algorithm 1 line 11 returns "first $k$ nodes" but $k$ is not declared as an input to Search-on-Graph. Reads cleanly as a corollary of the "candidate pool size $l$" guarantee, but the API signature in the paper would benefit from declaring both $k$ and $l$ explicitly.
- Algorithm 2's centroid-of-dataset step (line 1) is described in prose but the paper does not give the formula or note the cost.
- The DFS tree spanning step is described at a high level; rebuilding from scratch after every link addition (line 25) would be $O(n^2)$ in the worst case. The paper does not discuss amortized cost.