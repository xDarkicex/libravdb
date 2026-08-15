Hybrid Multi-Modal Search: When Prune-First Beats Rank-First—and When It Does Not
Verdict
Do not replace rank-first with a fixed prune-first rule. Build an adaptive cost-based dispatcher.

The pitch that “relational and graph predicates run first, producing a tiny candidate set; similarity is then computed only on that set, so there is no recall loss” is correct only under a narrow but valuable condition:

The relational and graph predicates are evaluated completely and exactly, and the vector score is computed exactly for every surviving candidate.

In that regime, prune-first has recall (1.0) relative to the exact filtered top-(k), by construction. There is no ANN recall loss because ANN is not being used. This is the strongest regime for prune-first and should be a first-class execution plan in the engine.

It is not generally true when “vector search on the pruned subspace” means running HNSW, NSG, DiskANN, product quantization, approximate MaxSim, or any other approximate method over a filtered subset. Filtering can destroy the connectivity and navigability on which graph ANN depends. Query-time filter-first heuristics can also deliberately skip distance computations and lose recall. Vespa’s current documentation explicitly warns that its filter-first HNSW heuristic can cause a recall dip; Qdrant, Elasticsearch/Lucene, Filtered-DiskANN, ACORN, SeRF, and the wider filtered-ANN literature all treat sparse filtered connectivity as a central problem rather than a solved detail. 

Conversely, the blanket claim that post-filtering necessarily causes “massive recall loss” is also too crude. With sufficient oversampling, post-filtering can achieve any desired fill probability under a known independent survival model. The catch is that (k/\sigma) is only the expected number of candidates needed, not a high-probability guarantee. For (k=10) and independent filter survival probability (\sigma=0.1), retrieving (m=100=k/\sigma) exact global neighbors gives only about a 54.9% probability of returning all ten filtered neighbors; expected recall is still about 88.1%. To obtain a 99% full-result probability under the same idealized model requires (m=183), or 1.83 times the naïve oversampling estimate. Correlation between rank and filter eligibility can make the result dramatically better or arbitrarily worse.

The operational regimes are therefore:

Regime	Generally strongest plan	Why
Small, exactly known candidate set	Prune-first exact scan	Exact filtered top-(k); sequential SIMD-friendly work; no ANN recall loss
Medium candidate set with filter-aware index support	Filtered ANN plus exact rerank	Avoids full scan while preserving filtered graph navigability
Weak filter, cheap ANN, or very expensive graph predicate	Rank-first with adaptive/iterative filtering	Avoids evaluating expensive predicates over most of the database
Extremely restrictive or disconnected filter	Candidate enumeration plus exact scan	HNSW traversal can become exhaustive or trapped
Unknown or badly estimated selectivity	Interleaved adaptive execution	Observes actual yield and switches or expands at runtime
ColBERT-style expensive MaxSim	Cheap proxy retrieval followed by exact filtered MaxSim	Exact MaxSim per candidate is much more expensive than a single-vector distance

This is also where the academic literature is converging. Filtered-DiskANN, ACORN, the ICML window-filter work, SeRF, VBASE, and the 2025 PVLDB filtered-vector-search tutorial do not identify one universally superior order. They develop specialized indexes, incremental traversal, segmenting, fallbacks, or adaptive execution precisely because the best method changes with selectivity, predicate structure, vector/filter correlation, (k), and index characteristics. 

The recommended architectural decision is therefore:

Commit to prune-first exact scan as one plan, not as the optimizer’s universal plan. Add filtered ANN and adaptive rank-first as peers, and choose among them using measured costs and a target recall contract.

Being single-process removes serialization, network round trips, and cross-system ID marshaling from the decision. That is useful, but it does not remove the actual algorithmic trade-off: predicate evaluation cost, BFS expansion, candidate cardinality, random graph traversal, distance computations, cache behavior, and ANN approximation remain. Zero serialization cost kills one historical excuse for rank-first. It does not make broad graph traversals free. Reality remains stubbornly employed.

Formal model of post-filtering and oversampling
Let the complete dataset contain (N) objects. Define the exact global vector-similarity order as

[ x_1, x_2, \ldots, x_N, ]

where (x_1) is the nearest object to the query vector. Let

[ F_i = \begin{cases} 1 & \text{if } x_i \text{ satisfies all relational and graph predicates},\ 0 & \text{otherwise}. \end{cases} ]

Suppose rank-first retrieves the exact global top (m) objects and then filters them. Define

[ X_m = \sum_{i=1}^{m} F_i. ]

Every qualifying object among the first (m) global ranks is necessarily among the first (X_m) objects in the exact filtered ranking. Therefore, assuming ties are resolved consistently,

[ \operatorname{Recall@}k(m)
\frac{\min(X_m,k)}{k}. ]

The query returns a complete filtered top-(k) exactly when

[ X_m \geq k. ]

This formulation exposes an important point: for exact global ranking, post-filter recall is controlled by the distribution of qualifying objects over global rank, not merely by total dataset selectivity.

Independent-filter model
Assume, unrealistically but usefully, that filter eligibility is independent of similarity rank and that each object survives with probability

[ P(F_i=1)=\sigma, ]

where (\sigma) is the fraction of the collection expected to survive. Then

[ X_m \sim \operatorname{Binomial}(m,\sigma). ]

The probability of obtaining a complete filtered top-(k) is

[ P(\operatorname{Recall@}k=1)
P(X_m\geq k)
1-\sum_{j=0}^{k-1} {m \choose j}\sigma^j(1-\sigma)^{m-j}. ]

Expected recall is

[ E[\operatorname{Recall@}k]
\frac{1}{k}E[\min(X_m,k)]. ]

If there are exactly (M) qualifying objects randomly distributed among the (N) ranks, (X_m) instead follows a hypergeometric distribution. For large (N) and (m\ll N), the binomial approximation is usually close.

The common rule

[ m=\left\lceil\frac{k}{\sigma}\right\rceil ]

sets

[ E[X_m]\approx k. ]

It does not set (P(X_m\geq k)) near one. In fact, because the binomial distribution is centered around (k), its probability of falling below (k) remains substantial.

For (k=10), exact binomial calculations give:

Survival fraction (\sigma)	Naïve (m=k/\sigma)	Probability of all ten	Expected recall	(m) for 95% complete	(m) for 99% complete
50%	20	58.8%	91.2%	28	33
10%	100	54.9%	88.1%	154	183
1%	1,000	54.3%	87.6%	1,568	1,874

This is the mathematically clean answer to the oversampling question:

(k/\sigma) is a mean-fill heuristic.
Roughly (1.5)–(2) times (k/\sigma) may be needed for high fill probability at ordinary (k), even under ideal independence.
The multiplier depends on (k), (\sigma), and the desired tail probability.
No finite multiplier derived only from global (\sigma) gives a distribution-free guarantee.
A conservative Chernoff bound provides a closed-form sufficient condition. Let

[ \mu=m\sigma,\qquad L=\ln(1/\varepsilon). ]

Using the lower-tail bound

[ P(X_m<k) \leq \exp\left(-\frac{(\mu-k)^2}{2\mu}\right), ]

it is sufficient to choose

[ \mu \geq k+L+\sqrt{L^2+2kL}, ]

and therefore

[ m \geq \frac{k+L+\sqrt{L^2+2kL}}{\sigma}. ]

This bound is conservative compared with an exact binomial quantile, but it makes the dependency explicit.

Correlated filters and rank-dependent survival
The iid model is precisely where many production oversampling assumptions go off the rails. Let

[ p_i=P(F_i=1\mid \text{global rank}=i). ]

Then (X_m) follows a Poisson-binomial distribution rather than a binomial one. The average collection survival fraction

[ \sigma=\frac{1}{N}\sum_{i=1}^{N}p_i ]

does not characterize the top of the ranking.

Three qualitatively different cases follow:

Independent or locally representative filtering.
If (p_i\approx\sigma) over the ranks the ANN visits, binomial amplification is a reasonable approximation.

Positive filter–similarity correlation.
If qualifying objects are overrepresented near the top, then (p_i>\sigma) at small (i). Post-filtering can work much better than (k/\sigma) predicts. A language filter applied to an embedding space already strongly clustered by language is one plausible example.

Negative correlation or adversarial filtering.
If qualifying objects are underrepresented near the top, (p_i<\sigma), oversampling based on global selectivity underestimates badly. In the extreme, all qualifying objects can begin after rank (m), giving recall zero even when the global survival fraction is large. The 2025 PVLDB tutorial specifically identifies the size, distribution, and vector-space correlation of qualifying data as central determinants of filtered-search behavior. 

Thus, no statement of the form

[ m = C\frac{k}{\sigma} ]

provides a universal recall guarantee for arbitrary relational or graph predicates. The optimizer needs an estimate of conditional survival by vector rank, not just table-level selectivity.

A practical statistic is a rank-bucket yield curve:

[ \hat p_b
\frac{\text{qualifying ANN results in rank bucket }b} {\text{ANN results examined in bucket }b}. ]

Track it by query class, filter signature, graph predicate template, tenant, or embedding partition. Even a coarse model over ranks (1!:!32), (33!:!128), (129!:!512), and so forth is more useful than a global (\sigma).

Adding ANN approximation
The preceding formulas assume rank-first produces the exact global top (m). HNSW and related indexes do not. Let (A_i) indicate that the ANN traversal retrieves object (x_i). The filtered results now depend on

[ F_i A_i. ]

The relevant quantity is not merely unfiltered ANN recall and not merely filter survival. It is the joint, rank-dependent inclusion probability

[ P(A_i=1,F_i=1). ]

Multiplying “ANN recall” by “filter recall” is valid only under strong independence assumptions that frequently do not hold. A filter can remove the bridge nodes needed to reach a qualifying neighborhood, making ANN errors themselves filter-dependent. Filtered-DiskANN and subsequent filtered-vector-search work were motivated by exactly this failure of general-purpose ANN traversal under filtering. 

Post-filter amplification therefore has two distinct jobs:

[ \text{retrieve enough objects to survive the filter} ]

and

[ \text{search deeply enough that the retrieved objects include the true filtered neighbors}. ]

Increasing (m) may solve the first without solving the second. Qdrant’s July 2026 benchmark provides a concrete example: for a 1% conjunction, increasing ef to 512 did not repair a plain graph traversal after it had exhausted a disconnected reachable region. 

Near ties and score distributions
Tightly clustered similarity scores require a precise distinction.

For an exact global ordering with a fixed tie-breaking rule, score gaps do not alter the binomial fill calculation. Post-filter fill depends on where qualifying objects occur in the order, not on the numerical difference between adjacent scores.

Near ties matter because real rank-first execution is approximate. When the distance gap between the (k)-th and nearby alternatives is small, small graph-search, quantization, proxy-scoring, or floating-point errors can reorder many candidates. High-dimensional spaces can exhibit distance concentration, where distances become difficult to distinguish, and “relative contrast” has been proposed as a measure of nearest-neighbor search difficulty. High-dimensional data can also develop hub objects that appear disproportionately in nearest-neighbor lists. 

The amplification mechanism is therefore indirect:

[ \text{small score margin} \Rightarrow \text{ranking instability under approximation} \Rightarrow \text{more candidates required to recover filtered top-}k. ]

Near ties are especially problematic when the filter is negatively correlated with the ANN index’s approximation error—for example, when qualifying objects sit in a dense boundary region poorly represented by the graph or quantizer. The literature supports the general relationship between low contrast and hard ANN retrieval, but there is not yet a broadly accepted formula mapping a score-gap distribution, filter selectivity, and HNSW parameters directly to filtered recall. That remains a real research gap rather than a place to invent comforting algebra.

Filtered graph ANN and the connectivity problem
A proximity-graph index works because a search can move through intermediate nodes toward the query neighborhood. “Pre-filtering” is dangerously overloaded because it can mean at least four different execution mechanisms.

Exact candidate enumeration followed by scan
The filter produces a complete bitmap (C), and the engine computes the vector score for every member of (C). The HNSW graph is not involved. This has exact recall.

Traversal over the induced filtered graph
Only nodes satisfying the filter may be visited or expanded. Conceptually, this searches the induced subgraph

[ G[C]. ]

Even when the original graph (G) is connected and navigable, (G[C]) may be disconnected, contain isolated components, or lack the long-range links needed for greedy routing. This is the naïve “prune the graph and search what remains” implementation, and it can produce severe recall loss.

Qdrant’s documentation states the problem directly: restrictive filters fragment HNSW, breaking connectivity and making traversal inefficient or impossible. Elasticsearch/Lucene similarly explains that considering only matching nodes can strand the search behind a “filtered gulf” between the entry point and the target region. 

Unfiltered traversal with filtered result admission
The search may traverse or score nonqualifying nodes as navigational bridges while admitting only qualifying nodes into the final result set. This better preserves the original graph’s navigability, but it can require many wasted node expansions and distance calculations when (\sigma) is low.

Weaviate calls its implementation pre-filtering because it constructs an allow-list before vector search. Its documentation also says its HNSW traversal continues following graph links normally and applies the filter when considering results, preserving graph integrity; for sufficiently restrictive filters it switches to flat search. This is not equivalent to building an HNSW index over the filtered set, nor is it a pure induced-subgraph traversal. 

Filter-aware traversal and filter-aware index construction
More sophisticated methods restore connectivity through query-time expansion, index-time edges, partitioning, or multiple overlapping graph structures.

Filtered-DiskANN.
Filtered-DiskANN modifies index construction so labels influence graph edges, rather than merely changing search over an ordinary graph. Its paper argues that search-only filtering is suboptimal and reports order-of-magnitude improvements over prior methods, with thousands of queries per second above 90% recall@10 on its evaluated real datasets. In a production A/B deployment covering 47 geographic regions, the paper reports 34.61% more clicks and 48.95% more revenue than the prior post-filter production baseline; gains were larger for regions representing less than 1% of traffic. Those business outcomes are workload-specific, but they are strong evidence that low-selectivity post-filtering can fail materially in production. 

ACORN and deeper neighborhood expansion.
ACORN-style search scores or expands filter-valid nodes while consulting neighbors-of-neighbors to compensate for the sparse valid subgraph. Lucene’s implementation explores second-level neighborhoods under restrictive conditions rather than blindly discarding all invalid bridges. Elastic reports that its implementation is activated when filtering becomes substantial and can deliver up to approximately fivefold speedups in tested cases with little recall loss, although equal-recall comparison still requires parameter tuning. 

Filterable HNSW with added edges.
Qdrant adds payload-aware edges so common payload subsets remain connected. Its per-segment planner then chooses among HNSW, payload-index enumeration, and full scan based on cardinality and available indexes. This is index-time structural support plus runtime adaptation, not merely “run WHERE first.” 

Range and segment indexes.
SeRF’s Segment Graph addresses ordered range filters by representing many range-specific HNSW structures compactly. Its experiments show why neither fixed order dominates: ANN-first slows sharply for narrow ranges because it must encounter rare in-range points, while range-first exact scan becomes inefficient for broad ranges. Segment Graph is designed to remain stable across range widths. 

The ICML 2024 window-filter work similarly identifies prefilter and postfilter as naïve endpoints and develops partitioning and tree-based indexes for arbitrary windows, reporting up to 75-fold speedups over its evaluated baselines on real and adversarial datasets. 

These approaches solve different filter families:

Method family	Main mechanism	Strength	Structural limitation
Query-time allow-list	Bitmap/ID set controls result admission	Arbitrary predicates, no dedicated label graph	May traverse many invalid nodes
ACORN-like expansion	Explore extended neighborhoods among valid nodes	Predicate-agnostic query-time repair	More expansions; can still fail under extreme sparsity
Label-aware graph edges	Add connectivity for known labels	Fast equality or label filters	Memory/build cost; conjunctions may lack dedicated edges
Segment/window graph	Overlapping structures for ordered ranges	Stable across range widths	Specialized for ordered/range predicates
Per-partition index	Separate graph by tenant/category	Excellent stable-filter performance	Index proliferation and imbalance
Exact bitmap scan	Enumerate then scan	Perfect recall, arbitrary predicates	Linear in surviving candidate count

No graph technique gives “no recall loss” merely because the filter ran first. The no-loss statement belongs to the exact-scan plan, not to approximate filtered graph traversal.

The user-specified acronym DPFS could not be matched to a canonical filtered-ANN paper in the searched SIGMOD, VLDB, ICDE, WWW, and recent survey/benchmark literature. Searches for that exact acronym produced no relevant primary source. It may refer to a differently named method, an internal abbreviation, or a typo. Treat any claims attributed to “DPFS” as unverified until a title, author, or DOI is supplied.

Exact filtered search and multi-vector scoring
Suppose an exact relational and graph evaluation produces a complete candidate set

[ C,\qquad c=|C|. ]

For a dense single-vector score of dimension (d), a simple cost approximation is

[ T_{\text{exact}} \approx T_{\text{candidate-generation}} + c\cdot t_{\text{distance}}(d) + T_{\text{top-k}}(c,k). ]

For dense dot product, cosine on normalized vectors, or squared Euclidean distance,

[ t_{\text{distance}}(d)=\Theta(d). ]

A bounded top-(k) heap gives

[ T_{\text{top-k}}=O(c\log k), ]

although for small (k) the vector arithmetic and memory traffic generally dominate.

A useful implementation-level model is

[ T_{\text{scan}} \approx \max\left( \frac{c,d,s}{B_{\text{mem}}}, \frac{c,d}{R_{\text{FMA}}} \right) + T_{\text{heap}}, ]

where (s) is bytes per stored coordinate, (B_{\text{mem}}) is sustainable memory bandwidth, and (R_{\text{FMA}}) is effective vector arithmetic throughput.

For HNSW-like ANN,

[ T_{\text{ANN}} \approx v_{\text{ANN}}\cdot t_{\text{random-distance}}(d) + e_{\text{ANN}}\cdot t_{\text{edge}} + T_{\text{filter-checks}} + T_{\text{rerank}}, ]

where (v_{\text{ANN}}) is the number of scored nodes and (e_{\text{ANN}}) is the number of edges examined. A graph node distance is often more expensive than one element of a contiguous exact scan because it involves pointer chasing, irregular memory access, metadata, visited-set operations, and poorer SIMD batching.

The break-even candidate count is therefore not simply “the number of nodes HNSW visits.” A first-order threshold is

[ c^* \approx \frac{ T_{\text{ANN}}-T_{\text{candidate-generation}} }{ t_{\text{sequential-distance}}(d)+t_{\text{heap-per-row}} }. ]

When candidate generation has already been performed for another part of the plan, its cost is sunk and should not be charged twice.

A more direct runtime decision is:

[ \text{choose exact scan if } \hat T_{\text{scan}}(c,d,k,\text{score type}) < \min( \hat T_{\text{filtered-ANN}}, \hat T_{\text{rank-first}} ). ]

There is no universal candidate threshold
Published systems use materially different fallback points. Vespa currently defaults to exact filtered search when its estimated hit ratio is below 2%, although the threshold is configurable. Weaviate automatically switches to flat search when its allow-list becomes sufficiently restrictive. Qdrant’s planner can enumerate candidates from the payload index and scan them instead of using HNSW. These are evidence for adaptive fallback, not evidence that 2%, 40,000 candidates, or any other product default is universally optimal. 

Vespa’s practical guide provides a useful hardware-specific data point: on its demonstrated 95,666-vector collection, approximate search took about 4 ms with one matching thread; the exact version required more work and threading to approach the same latency range. That example shows that scanning roughly (10^5) vectors can already be slower than ANN, but it cannot be generalized without the vector dimension, hardware, score function, memory layout, and workload. 

For the proposed engine, the right way to establish (c^*) is a microbenchmark matrix across:

[ d\in{128,384,768,1024,1536,3072}, ]

candidate counts from a few hundred through several hundred thousand, each supported distance metric, each storage encoding, and both cold and warm cache conditions. Measure p50 and p99, not just throughput. Graph ANN’s random access and exact scan’s contiguous SoA behavior diverge more at the tail than a neat average would suggest.

Exactness conditions
Prune-first exact search has recall one only if all of the following hold:

The relational predicate evaluation is exact.
The graph traversal implements the complete SQL path semantics.
No BFS frontier, path count, timeout, or visited budget truncates valid candidates.
Every surviving candidate receives the authoritative vector score.
The final top-(k) is selected using that same authoritative score and deterministic tie handling.
The candidate bitmap and vector storage represent the same consistent snapshot.
A depth-bounded BFS such as *1..3 is exact if all qualifying paths within those depths are explored. A BFS stopped after a candidate quota or wall-clock budget is approximate, regardless of whether marketing calls it “pruning.”

ColBERT-style MaxSim changes the threshold
For query token vectors (q_1,\dots,q_{n_q}) and document token vectors (d_1,\dots,d_{n_i}), ColBERT-style scoring is approximately

[ S(Q,D_i)
\sum_{a=1}^{n_q} \max_{b\in{1,\dots,n_i}} \langle q_a,d_b\rangle. ]

Exact cost per document is

[ \Theta(n_q n_i d), ]

rather than (\Theta(d)) for one dense-vector distance. Consequently,

[ T_{\text{exact-MaxSim}} \approx \sum_{i\in C} n_q n_i d \cdot t_{\text{dot}}. ]

The exact-scan threshold can therefore be orders of magnitude smaller for long documents. A candidate count that is trivial for 384-dimensional cosine distance may be unacceptable for 32 query token vectors against 180 document token vectors.

The strongest architecture is generally three-stage:

[ \text{structured/graph pruning} \rightarrow \text{cheap ANN or proxy retrieval} \rightarrow \text{exact MaxSim rerank}. ]

MUVERA, presented at NeurIPS 2024, illustrates the value of a single-vector proxy for multi-vector retrieval: its fixed-dimensional encodings retrieved two to five times fewer candidates than prior heuristics at the same recall in its experiments and reported substantially lower end-to-end latency. Those exact numbers are method- and benchmark-specific, but they reinforce that MaxSim should have its own cost model rather than inherit the threshold used for ordinary dense vectors. 

The optimizer should estimate

[ W_{\text{MaxSim}}
n_q\sum_{i\in C}n_i d, ]

not merely (c). Store document token-count histograms and preferably the actual sum for a materialized candidate bitmap.

Relational and graph execution order
Classical relational optimization supports selection pushdown because reducing intermediate cardinality often reduces later work. System R’s cost-based optimizer chose access paths and join orders using relation statistics and estimated costs rather than a universal syntactic order. 

Top-(k) vector operators complicate that principle. Semantically,

[ \operatorname{TopK}_{v}(F(R)) ]

means “filter relation (R), then return the best (k) by vector score.” An exact optimizer may implement that result through either filter-first or rank-first execution. Computationally, however, an approximate top-(k) operator cannot generally be pushed through a filter while preserving recall.

When BFS is a natural prune stage
Graph-first is attractive when the traversal has:

a small indexed starting set;
a low branching factor;
a shallow maximum depth;
selective edge types or node predicates;
deduplication that quickly converges;
an output candidate count small enough for exact vector scoring.
For the example query, t.sla_status = 'active' may sharply reduce seed services, and bounded DEPENDS_ON*1..3 traversal may produce a compact manual-document bitmap. If those assumptions hold, graph/relational-first followed by exact scan is likely the best plan.

Let (s_0) be the number of seed vertices, (b_i) the effective distinct branching factor at depth (i), and (h) the maximum depth. A crude traversal work model is

[ W_{\text{BFS}} \approx s_0 \left( 1+b_1+b_1b_2+\cdots+\prod_{i=1}^{h}b_i \right), ]

adjusted downward for repeated vertices and upward for predicate checks, edge decoding, and cache misses.

With even modest branching, the frontier can become the expensive operator. For (b=20) and (h=3), the unconstrained tree-shaped upper estimate is

[ 1+20+400+8000=8421 ]

vertices per seed before deduplication. Multiply that by thousands of services and the “natural prune” becomes a natural resource fire.

When graph verification should run later
Rank-first can win when:

relational filters are weak and remove little;
BFS has a large seed set or high branching;
graph predicates are expensive to verify;
the desired (k) is small;
vector rank is positively correlated with graph eligibility;
a cheap reachability or path-existence test can be run for individual candidates.
The plan becomes

[ \text{ANN batch} \rightarrow \text{graph predicate verification} \rightarrow \text{repeat until }k\text{ valid}. ]

A fixed one-shot batch is unsafe. An iterative batch plan is much better:

[ m_1 < m_2 < \cdots, ]

with graph verification applied incrementally until (k) results are found or a search budget is exhausted.

VBASE, presented at OSDI 2023, takes this general direction for relational/vector queries. It exposes vector index traversal through an iterator and evaluates downstream predicates while traversal is still in progress, stopping when enough valid results have been obtained. This avoids guessing a fixed (K') in advance and is closer to an adaptive database operator than to a rigid post-filter pipeline. 

For graph predicates, the same design can be extended:

Pull the next ANN candidate batch.
Deduplicate candidate graph nodes.
Verify graph constraints in a batched, cache-friendly traversal.
Update observed survival and cost estimates.
Continue ANN, switch to graph-first enumeration, or stop.
This is effectively a vector–graph semijoin with adaptive batching.

Cases where neither side should fully execute first
A unified in-process engine can interleave operations more aggressively than federated systems:

Evaluate cheap scalar predicates into bitmaps.
Use those bitmaps while expanding BFS frontiers.
Feed discovered document IDs into an incremental exact top-(k) heap.
Stop graph expansion only when a valid bound proves unseen candidates cannot beat the current heap, if such a bound exists.
Alternatively, draw ANN candidates and run graph reachability checks in batches.
Reoptimize after observing actual frontier and survivor counts.
There is rarely a useful vector-score lower bound for arbitrary unvisited graph nodes, so early termination of graph-first exact search may be impossible unless graph partitions carry vector bounds or summaries. Segment-level vector centroids, bounding radii, scalar quantization bounds, or learned score envelopes could provide such pruning, but they would need correctness proofs if used for exact termination.

The knowledge-graph vector literature is relevant but incomplete. HQI, published at SIGMOD 2023, combines vector predicates with relational attributes over knowledge-graph entities and uses workload-aware partitioning and multi-query optimization, reporting 31-fold higher throughput for its industrial batch workloads. Its focus is batch workload throughput rather than online BFS-versus-ANN operator ordering, so it does not settle the execution-order question here. 

There appears to be no mature primary literature providing a complete cost model for arbitrary SQL joins, bounded property-graph paths, HNSW, and ColBERT MaxSim in one runtime. The proposed engine is operating in a space where the optimizer itself could constitute publishable systems work, assuming the benchmarks survive contact with reviewers.

What production systems actually do
Vendor terminology is inconsistent. “Pre-filter” may mean exact candidate enumeration, traversal-time allow-lists, scoring only valid nodes, or using a label-aware index. Architecture must be compared, not labels.

System	Current documented behavior	Recall or latency evidence	Important caveat
Vespa	Supports prefilter, postfilter, filtered HNSW, filter-first heuristics, and exact fallback. Current default disables postfilter and falls back to exact below an estimated 2% hit ratio. Postfilter adjusts target hits by estimated selectivity with a configurable cap.	Practical guide shows a 95,666-vector example where ANN was about 4 ms with one thread; documentation warns filter-first can reduce recall.	Thresholds are defaults, not universal laws; selectivity estimation can undershoot postfilter target adjustment.
Qdrant	Payload-aware extra HNSW edges, ACORN option, payload-index enumeration, exact/full-scan fallback, per-segment planner.	July 2026 vendor benchmark reports 99–100% recall in many single-filter cases with filterable HNSW/planner; plain traversal collapsed under restrictive conjunctions.	Vendor-run benchmark on one million 96D vectors; conjunctions not represented by dedicated edges still fail without planner fallback.
Weaviate	Builds an inverted-index allow-list before vector search; HNSW follows graph links and filters result admission; ACORN is default from v1.34; flat fallback for restrictive filters.	Documentation says filtered recall is typically no worse than unfiltered and presents a recall graphic down to 1% matching.	“Pre-filter” does not mean searching only the induced valid graph. Public numbers are not a broad independent benchmark.
Milvus	Standard mode evaluates metadata filtering before ANN within matching entities; iterative mode alternates vector retrieval and filtering for expensive expressions.	Official docs explain modes but do not provide a current apples-to-apples recall/latency table for all filter/index combinations.	Actual mechanics and performance vary by index type and Milvus version.
Elasticsearch/Lucene	Native knn.filter is a filter on documents allowed to match during kNN search. Lucene uses filtered traversal and ACORN-like extended neighborhoods.	Elastic reports up to roughly fivefold performance improvement in tested restrictive/inversely correlated cases with little recall loss.	num_candidates controls ANN candidate breadth; it is not the same as postfilter selectivity oversampling.
Pinecone	Public API supports metadata filters limiting eligible records.	Current public docs do not disclose a comparable filtered-search recall/latency study or enough internals to classify serverless execution precisely.	Historical descriptions of earlier Pinecone IVF behavior should not be assumed to describe the current service.
pgvector	PostgreSQL may use scalar indexes for exact filtered search. HNSW/IVFFlat filtering is applied after index scanning; iterative scans continue until enough results or limits are reached. Partial indexes and partitioning support stable filter classes.	Documentation notes that with 10% survival and default HNSW ef_search=40, only about four matches are expected without iterative scanning.	Planner choice depends on PostgreSQL cost estimates; approximate index scans can return too few results or hit scan limits.

Vespa
Vespa is the clearest production example of the recommended architecture. It exposes three regimes:

[ \text{postfilter for weak filters}, \quad \text{filtered HNSW for moderate filters}, \quad \text{exact scan for restrictive filters}. ]

Its documented postfilter adjustment is

[ \text{adjustedTargetHits}
\min\left( \frac{\text{targetHits}}{\text{estimatedFilterHitRatio}}, \text{targetHits}\times\text{maxAdjustmentFactor} \right). ]

Its current default postFilterThreshold is 1.0, meaning postfilter is disabled unless configured, while approximateThreshold defaults to 0.02 for exact fallback. Vespa also exposes a filter-first threshold, exploration controls, and ANN time budgets. Its documentation explicitly states that the filter-first heuristic can trade recall for latency. 

An older Vespa configuration example used postfilter above 75% survival, filtered HNSW between 5% and 75%, and exact search below 5%. Vespa warns that conservative selectivity estimates can still cause target-hit adjustment to undershoot, producing fewer valid results than requested. That example is useful as a shape, not a value to copy. 

Qdrant
Qdrant’s filterable HNSW adds edges for indexed payload values and uses a planner to select graph traversal or exact candidate enumeration. Its own 2026 benchmark is unusually informative:

At 20% single-keyword survival, plain filtered traversal achieved 62.9% recall@10 at 1.6 ms; filterable HNSW achieved 94.8% at 1.2 ms; ACORN on the plain graph achieved 98.9% at 4.4 ms.
At 1% single-keyword survival, plain traversal achieved 0.1%; filterable HNSW achieved 99.8% at 1.0 ms.
For a 4% two-keyword conjunction, filterable HNSW achieved only 63.7% because the intersection lacked dedicated connectivity; planner plus ACORN achieved 99.9% at 13.9 ms.
At 0.012% survival—roughly 120 candidates in one million—the graph methods achieved 0.5% to 1.8% recall, while the planner’s payload-index path returned 100% at 1.3 ms. 
These results neatly falsify both simplistic positions. Filter-aware ANN can dominate exact scanning at 1% for a well-indexed single label, while an even more restrictive conjunction is better served by abandoning the graph.

Weaviate and Milvus
Weaviate constructs the filter before vector search but maintains traversal through the full HNSW link structure. Its documentation says this preserves graph integrity and generally avoids additional recall degradation, then falls back to flat search as filtering becomes restrictive. Current docs state that ACORN is the default filtering strategy from version 1.34, particularly helping when filter values have low correlation with vector neighborhoods. 

Milvus describes standard filtering as metadata-first and iterative filtering as progressively running vector search and evaluating the scalar expression until enough valid results are found. Iterative filtering is intended for cases where evaluating a complex filter over a large candidate set is itself costly. That is direct production evidence that “push every predicate first” is not always the cheapest execution plan. 

SeRF’s 2024 evaluation found Milvus’s tested HNSW-plus-bitset range-filter behavior similar to ANN-first at varying range widths, but this should be treated as a result for the particular version, configuration, and range workload studied—not a permanent characterization of all Milvus indexes. 

Elasticsearch
The premise that Elasticsearch’s native knn.filter is simply postfiltering with oversampling is not correct for current Elasticsearch. The documented filter restricts which documents may match the kNN query. num_candidates controls how many ANN candidates are considered per shard and trades latency for ANN accuracy; its default is approximately (1.5k), capped at 10,000. It is not computed as (k/\sigma). 

Lucene’s filtered HNSW traversal historically needed to score filtered-out bridge nodes to preserve navigation. Its ACORN adaptation selectively explores second-level neighborhoods when enough immediate neighbors are filtered, reducing unnecessary vector operations while avoiding immediate disconnection. 

Elasticsearch can still perform true postfiltering through other query constructs, but that should not be conflated with the native kNN filter semantics.

Pinecone
Pinecone’s current documentation guarantees that metadata filters limit search to matching records, but the public page does not specify whether current serverless indexes use filter-aware graphs, posting-list enumeration, IVF inline filtering, adaptive switching, or another internal method. 

The 2023 Filtered-DiskANN paper described Pinecone’s then-current hybrid feature as using IVF-style inline processing. That is a historical observation, not evidence about Pinecone’s 2026 implementation. 

pgvector and PostgreSQL
pgvector documents a materially different model from the filter-aware systems above. With approximate indexes, filtering occurs after index scanning. Its documentation gives the intuitive example that with a 10% qualifying rate and default HNSW ef_search=40, only about four rows qualify on average. Iterative scans can continue index traversal until sufficient results are found, subject to tuple and memory limits. For restrictive stable predicates, pgvector recommends scalar indexes for exact search, partial HNSW indexes, or table partitioning. 

This behavior is close to the proposed engine’s current rank-first architecture. It also demonstrates the obvious upgrade path: make the index scan resumable and let the optimizer switch to exact candidate enumeration when observed yield or planner estimates make continued HNSW traversal irrational.

Recommended optimizer and validation plan
The engine should expose three physical vector operators beneath the same logical SQL operator:

[ \operatorname{FilteredTopK}(C,q,k). ]

The physical implementations should be:

[ \texttt{ExactCandidateScan}, ]

[ \texttt{FilteredANN}, ]

and

[ \texttt{IterativeANNThenFilter}. ]

A fourth composite operator, adaptive at runtime, should be able to transition among them.

Required estimates
For each query, estimate or observe:

Symbol	Meaning
(N)	Total vectors in relevant partition
(c)	Exact or estimated number of candidates after relational and graph predicates
(\sigma=c/N)	Overall survival fraction
(k)	Requested result count
(d)	Dense-vector dimension
(n_q,{n_i})	Query and document token-vector counts for MaxSim
(T_R)	Relational predicate evaluation cost
(T_G)	Graph traversal or per-candidate graph-verification cost
(v_f)	Predicted filtered-ANN distance computations
(p_i) or (p_b)	Survival probability by global rank or rank bucket
(R_f)	Predicted filtered-ANN recall at chosen index parameters
(\Delta_k)	Score margin or contrast around the result boundary
(U)	Update rate and associated index-maintenance cost

Do not rely on (\sigma) alone. At minimum, maintain separate statistics for:

scalar-filter cardinality;
graph-pattern output cardinality;
scalar/graph conjunction cardinality;
top-rank survivor yield;
graph expansion count and distinct-vertex count;
filter signature and vector-cluster correlation.
Plan selection
The core dispatcher can be stated as:

[ P^*
\arg\min_{P\in{ E,F,A }} \hat T_P ]

subject to

[ \hat R_P \geq R_{\min} ]

and

[ \hat M_P \leq M_{\max}, ]

where (E) is exact candidate scan, (F) filtered ANN, and (A) adaptive rank-first.

Exact candidate scan
Choose exact prune-first when:

[ c \leq C_{\text{exact}}(d,k,\text{score type},\text{hardware}), ]

or more generally,

[ \hat T_E < \min(\hat T_F,\hat T_A) ]

and exact candidate generation is already complete or cheap.

For MaxSim, use

[ \sum_{i\in C}n_i ]

and (n_q), not just (c), to estimate scoring work.

Exact scan should also be the fallback when:

the filtered graph is predicted to be disconnected;
the filter cardinality is below the graph’s useful operating range;
the requested recall is 1.0;
the ANN search has exhausted a budget without enough results;
candidate enumeration unexpectedly produces a very small set.
Filtered ANN
Choose filtered ANN when:

[ c>C_{\text{exact}}, ]

the filter-aware index supports the predicate family, and

[ \hat R_F(\sigma,\text{correlation},ef,\text{graph density}) \geq R_{\min}. ]

This plan should include exact reranking of the returned ANN candidates whenever the stored index uses quantization, proxies, or compressed representations.

Do not treat arbitrary conjunctions as equivalent to indexed single labels. A graph with edges for tenant=A and status=active does not necessarily have a connected subgraph for their intersection.

Iterative rank-first
Choose rank-first when:

filters are weak;
graph-first execution is predicted to be expensive;
(k) is small;
top-rank filter yield is favorable;
the filter can be verified cheaply per candidate or in batches;
filtered ANN has no suitable index support.
Do not issue one fixed (m=k/\sigma) request. Compute the smallest (m) satisfying a target probability under the best available model:

[ m^*
\min \left{ m: P(X_m\geq k)\geq1-\varepsilon \right}. ]

For an iid estimate, use the binomial quantile. For rank-bucket estimates, use a Poisson-binomial, beta-binomial, or simulation-based quantile. Cap (m^*); if it exceeds the cap, switch to candidate enumeration or filtered ANN.

At runtime, retrieve in geometric batches such as

[ m_1,;2m_1,;4m_1,\ldots ]

while updating the observed top-rank survival rate. Reuse the ANN traversal state rather than restarting.

Runtime reoptimization
A robust operator should support these transitions:

[ \text{ANN-first} \rightarrow \text{exact candidate scan} ]

when observed qualifying yield is much lower than predicted;

[ \text{graph-first} \rightarrow \text{filtered ANN} ]

when the BFS candidate set grows beyond its expected threshold;

[ \text{filtered ANN} \rightarrow \text{exact scan} ]

when the candidate bitmap becomes small or the graph fails to make progress;

[ \text{single-vector proxy} \rightarrow \text{exact MaxSim} ]

when enough candidates have been collected for authoritative reranking.

This is the same broad design principle visible in Vespa’s hybrid dispatch, Qdrant’s per-segment planner, Weaviate’s flat fallback, Milvus’s iterative mode, pgvector’s iterative scans, and VBASE’s resumable vector traversal. 

Initial thresholds
No defensible universal threshold can be extracted from the literature. Start with measured thresholds rather than doctrine.

A reasonable commissioning process is to benchmark candidate counts at:

[ 1\text{k},;5\text{k},;10\text{k},;25\text{k},;50\text{k},;100\text{k},;250\text{k},;500\text{k}, ]

for every important dimension and score type. Fit separate models for:

sequential exact dense scoring;
random-order exact scoring;
filtered HNSW;
unfiltered HNSW plus scalar verification;
HNSW plus BFS/path verification;
exact MaxSim;
proxy-plus-MaxSim.
Use hysteresis around switching boundaries so small selectivity-estimation errors do not cause plan flapping.

Metrics that must be recorded
For each executed query, record:

Category	Metrics
Cardinality	Estimated and actual scalar candidates, graph candidates, conjunction candidates
Graph work	Seeds, vertices visited, edges decoded, duplicate rate, frontier size by depth
ANN work	Nodes visited, edges inspected, distance computations, ef, candidates emitted
Filter yield	Valid hits by ANN rank bucket, candidates needed to produce each accepted hit
Quality	Recall against sampled exact ground truth, result shortfall, score margin at (k)
CPU and memory	Cycles, cache misses, SIMD utilization, bytes read, allocations
Multi-vector	Query tokens, document tokens scored, MaxSim dot products
Updates	Added graph edges, index-build time, write amplification, stale-index lag
Tail behavior	p50, p95, p99 latency and timeout/fallback counts

Exact ground truth need not run on every query. Sample queries, shadow them against exact filtered scans, and stratify sampling by selectivity, filter signature, graph depth, and estimated score contrast. Otherwise the optimizer will become very good at optimizing its own unverified assumptions, which is how software develops religion.

Final architectural recommendation
The M3 optimizer should implement the following logical decision:

text
Copy
materialize cheap scalar bitmap

estimate graph cost and graph output cardinality

if exact graph traversal is cheap:
    execute it and obtain actual candidate bitmap

    if exact-score cost is below calibrated threshold:
        exact candidate scan
    else if filter-aware ANN supports this predicate shape:
        filtered ANN + exact rerank
    else:
        iterative ANN constrained/verified against candidate bitmap
else:
    begin resumable ANN traversal
    verify scalar and graph predicates in batches

    if observed survivor yield is healthy:
        continue until k valid results
    else if candidate enumeration becomes cheaper:
        switch to exact graph/filter enumeration + exact or filtered ANN
The optimizer’s recall contract should be explicit:

EXACT: complete candidate generation and exact scoring.
BOUNDED_RECALL: target recall plus confidence/SLA, with exact shadow validation.
BEST_EFFORT: time-budgeted ANN with possible result shortfall.
That distinction prevents “no recall loss” from leaking into approximate plans merely because a bitmap appeared somewhere upstream.

Remaining evidence gaps
The literature is strong on filtered ANN for equality labels, categorical predicates, and one-dimensional ranges. It is much thinner on arbitrary Boolean SQL expressions, highly dynamic updates, multi-hop graph path predicates, and multi-vector MaxSim in one optimizer. The 2025 PVLDB tutorial explicitly describes stable, declarative recall and query optimization as open research areas. 

Vendor benchmarks are not directly comparable. They use different vectors, dimensions, filter distributions, index parameters, hardware, recall definitions, and fallback rules. Qdrant’s 2026 benchmark is reproducible and unusually detailed, but it remains a vendor benchmark. Vespa exposes useful controls and measurements but does not provide one universal modern table covering every mode. Milvus and Pinecone currently publish insufficient apples-to-apples recall and latency evidence to infer broad superiority.

There is no universal closed-form HNSW cost model that predicts filtered traversal from (N,d,\sigma), and average graph degree alone. Predicate/vector correlation, local graph connectivity, entry-point position, filter conjunction structure, and score contrast matter. Empirical model fitting remains necessary.

There is also no known general theorem saying whether an arbitrary bounded property-graph traversal should precede or follow ANN. The answer depends on seed cardinality, branching, duplicate convergence, reachability index support, vector-rank correlation, and whether graph verification can be batched or bounded.

The independent conclusion is therefore unambiguous:

Prune-first exact scan is mathematically superior when pruning produces a complete, sufficiently small candidate set. Filtered ANN is superior in the middle regime when the index preserves navigability under the actual predicate. Adaptive rank-first is superior when filters or graph traversal are weak, broad, or expensive. No fixed order dominates.

The proposed rewrite should be an adaptive cost-based dispatcher, not a global inversion from vector-first to filter-first.
