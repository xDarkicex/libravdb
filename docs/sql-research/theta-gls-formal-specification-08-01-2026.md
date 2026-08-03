# θ_GLS Formal Specification: Research Response and Critical Review

**Date:** 2026-08-01
**Status:** Reference document — θ_GLS formal specification preserved with critical analysis
**Preread:**
- `foundations-of-unified-query-algebra-08-01-2026.md` — the FVS pushdown law
- `optimizer-cost-model-research-response-08-01-2026.md` — the cost model
- `unified-optimizer-specification-08-01-2026.md` — the joint recall distribution
- `theta-gls-computation-research-response-08-01-2026.md` — the prior θ_GLS research

---

## What This Document Is

This doc preserves the formal specification of θ_GLS as an adaptation of Local Moran's I to high-dimensional vector graphs. The specification provides a closed-form mathematical formula, four computable implementations with cost analysis, a catalog design, an ECQO budget allocation, the iFVS interaction, failure mode analysis, and a validation methodology.

The doc also adds a critical review. The `c(sel, θ_GLS)` function is still unspecified (the blocking item for the FVS law), the [0, 1] bound makes an explicit design choice (no depleting case) that should be acknowledged, and several smaller items need to land.

The formal specification is the *answer*. The critical review is the *verification*.

---

## Part 1: The Research Response

The response provided the following ten sections.

### 1. Architectural Context

The evolution of database systems toward hybrid architectures has fundamentally altered the landscape of query optimization. As systems like libraVDB transition into unified hybrid databases, they are tasked with seamlessly integrating unstructured semantic retrieval with rigorous relational algebra. This integration is increasingly critical for complex Retrieval-Augmented Generation (RAG) pipelines, which demand the simultaneous execution of high-dimensional approximate nearest neighbor (ANN) searches and structured metadata constraints.

Within the libraVDB unified query algebra, the optimizer relies on a catalog of equivalence laws to navigate the search space of potential physical plans. The central equivalence law for Filtered Vector Search (FVS) asserts that an exact post-filtered vector search can be approximated by a pre-filtered search or an iteratively oversampled search:

```
topk(k, σ_p(R), d) ≈ σ_p(topk(k + c(sel(p), θ_GLS), R, d))
```

This law is the foundational mechanism that allows the query planner to evaluate the trade-offs between traversing a massive, unfiltered index and dynamically pruning the search space based on structured predicates. The viability of this equivalence law hinges entirely on the oversampling cost-delta function `c(sel(p), θ_GLS)`. Currently, this function remains non-computable because its core parameter, the Global-Local Selectivity metric (θ_GLS), exists only as a conceptual placeholder in the literature.

### 2. Formal Mathematical Definition of θ_GLS

The canonical filtered vector search literature intuitively describes Global-Local Selectivity as the correlation between a predicate's global selectivity across the entire corpus and its local selectivity within the geometric neighborhoods of the embedding space. While the selectivity ratio `r = σ_l / σ_g` provides a valuable diagnostic tool for evaluating executed queries, it fails to serve as a predictive metric for query optimization because `σ_l` cannot be known without first executing the query.

To construct a predictive, computable metric that satisfies the requirements of the unified optimizer, θ_GLS must be decoupled from the specific query vector `q`. Instead, it must be formally defined as an intrinsic property of the predicate's spatial distribution across the entire vector manifold. In high-dimensional spaces, if a predicate's satisfaction is completely uncorrelated with the underlying vector geometry, the valid vectors will be distributed uniformly at random. Conversely, if a predicate is highly correlated with the vector geometry, the valid vectors will cluster densely in specific geometric regions.

This phenomenon is mathematically equivalent to the concept of **spatial autocorrelation**, which quantifies the degree to which a set of spatial features and their associated data values are clustered together in space. To derive a closed-form definition for θ_GLS, the response adapts the **Local Moran's I coefficient** to the topology of high-dimensional vector graphs.

Let the base vector index be represented as an adjacency graph `G = (V, E)`, such as the lowest layer of an HNSW index, which serves as a highly accurate approximation of the true k-NN graph for the corpus. Let `N` represent the total cardinality of the vector dataset `V`. We define an `N × N` spatial weight matrix `W`, where the element `w_{ij} = 1` if an edge exists between vector `v_i` and vector `v_j` in the proximity graph, and `w_{ij} = 0` otherwise. To account for varying node degrees, the weight matrix is typically row-standardized such that `Σ_j w_{ij} = 1` for all `i`.

For a given relational predicate `p`, let `x_i ∈ {0, 1}` be the boolean indicator variable denoting whether the vector `v_i` satisfies the predicate. The global selectivity of the predicate is:

```
σ_g = x̄ = (1/N) × Σ_i x_i
```

The Global-Local Selectivity metric, `θ_GLS(p)`, is formally defined as the normalized spatial autocorrelation of the predicate indicator function over the vector adjacency graph:

```
θ_GLS(p) = (N / Σ_i Σ_j w_{ij}) × (Σ_i Σ_j w_{ij} (x_i - σ_g)(x_j - σ_g)) / (Σ_i (x_i - σ_g)²)
```

This derivation yields a bounded scalar metric. A value of `θ_GLS(p) ≈ 0` signifies complete spatial randomness (predicate independent of vector space topology). A value of `θ_GLS(p) > 0` indicates positive spatial autocorrelation (high spatial concentration). The closer θ_GLS approaches its maximum theoretical bound of 1, the more highly concentrated the predicate is. Negative values, while mathematically possible (indicating perfect dispersion), are statistically negligible in real-world, high-dimensional embedding spaces due to the concentration of measure phenomenon.

### 3. Dimensionality, Scope, and Granularity

The architectural decision hinges on whether θ_GLS should be treated as a static per-corpus constant, a dynamic per-query-predicate measurement computed at runtime, or a parameterized function spanning the middle ground.

- **Per-corpus constant** is computationally trivial but structurally invalid: different attributes exhibit vastly divergent spatial distributions. Using a single global constant would overestimate oversampling for uniform predicates and underestimate for clustered ones.
- **Per-query measurement** guarantees mathematical perfection but violates the sub-millisecond ECQO budget. Computing it requires essentially the same work as executing the search itself.

The **optimal resolution** is to define θ_GLS as a per-corpus function that takes the specific query predicate `p` as its input parameter. By decoupling the metric from the query vector `q` while maintaining its dependence on the predicate `p`, the optimizer can leverage pre-computed spatial statistics to predict the degree of clustering for the exact filter being applied. The implementation takes the form of a parameterized lookup function, `θ_GLS(p, σ_g)`, which returns the predictive scalar based on historical or incrementally maintained structural metadata.

### 4. Computable Implementations

Four viable architectural candidates were evaluated:

| Implementation | Mechanism | Primary Cost | Accuracy | Failure Modes |
|---|---|---|---|---|
| **Sampling-Based Graph Probe** | Probes M random nodes, traverses localized edges, calculates variance | High CPU latency, cache thrashing | Moderate | High variance on selective predicates; misses isolated micro-clusters |
| **Learned Autoregressive Predictor** | Maps predicate features to spatial clustering scalar via neural inference | 2-5 ms CPU latency; needs retraining | High for in-distribution | Hallucination on unseen schemas |
| **Analytical Heuristic Derivation** | Rule-based heuristics from B-tree clustering factors, distinct value counts | Negligible | Low | Fails on complex multi-column predicates |
| **Histogram-Based Region Variance** | Divides vector space into C partitions, maintains per-region selectivities, calculates spatial variance | Moderate memory, <10 μs CPU | High | OOM on high-cardinality continuous predicates |

**Recommendation: Histogram-Based Region Variance** with dynamic spatial partitioning. This method effectively translates the complex Moran's I graph computation into a highly efficient, coarse-grained macroscopic approximation.

During vector ingestion, the high-dimensional space is partitioned into C distinct regions using k-means or Leiden community detection. For every frequently queried predicate `p`, the system maintains a localized selectivity array `[σ_{l,1}, σ_{l,2}, ..., σ_{l,C}]`.

When the query planner invokes the cost-delta function, the θ_GLS(p) computation executes:

1. Query the relational catalog for global selectivity `σ_g`
2. Perform an O(1) lookup in the specialized vector catalog to fetch the localized selectivity array
3. Approximate spatial autocorrelation via normalized variance:

```
θ_GLS(p) ≈ ((1/C) × Σ_c (σ_{l,c} - σ_g)²) / (σ_g × (1 - σ_g))
```

The denominator, `σ_g(1 - σ_g)`, represents the maximum possible variance of a Bernoulli distribution, ensuring the result is bounded between 0 and 1.

### 5. Computational Cost and ECQO Budget Allocation

Assuming a conservative total planning budget of 1,000 microseconds (1 ms):

| Phase | Budget |
|---|---|
| AST Parsing and Logical Algebrization | ~150 μs |
| Relational Histogram Probes | ~50 μs |
| Localized ANN Probe (ECQO Core) | 600-700 μs |
| Plan Enumeration and Costing | ~100 μs |
| **Available for θ_GLS** | **0-100 μs** |

The Histogram-Based Region Variance implementation satisfies this constraint. With C=1024 regions, fetching the pre-computed array requires a single hash-map lookup (< 1 μs assuming L1/L2 cache hit). The subsequent variance computation iterates over 1,024 float32 local selectivities. With AVX-512 SIMD vectorization, this executes in approximately 3-8 μs on contemporary CPUs. **End-to-end θ_GLS resolution: < 10 μs** (< 10% of the available window).

### 6. Catalog Representation and Lifecycle Maintenance

The optimal catalog representation is a dynamic, region-level spatial map: `theta_gls_map: Map[predicate_hash, Array[float32]]`. The key is a deterministic hash of the predicate AST; the value is the C-dimensional array of localized selectivities.

**Storage footprint:** For C=1024 regions, each monitored predicate requires 4,096 bytes. Tracking the top 10,000 most-frequently-executed predicates requires approximately 40 MB — negligible for modern servers and easily pinned in main memory.

**Lifecycle maintenance:** Incremental, asynchronous with staleness tolerance. The θ_GLS metric is a probabilistic planning signal, not an exact transactional requirement. As new vectors are ingested, the system determines their spatial partition (an operation already required by partitioned indexes). A lightweight background daemon aggregates these assignments in a thread-local buffer. Periodically (every 10 seconds or 10,000 inserts), the daemon flushes updates via an Exponential Moving Average (EMA). This asynchronous batching ensures spatial statistics gracefully evolve to reflect data drift without imposing synchronous locking or WAL generation overhead.

### 7. Interaction with the iFVS Weight Matrix

θ_GLS and the iFVS weight matrix W are **complementary systems operating at different phases**:

- **θ_GLS** is a **macroscopic pre-execution planning signal**
- **iFVS weight matrix W** is a **microscopic in-flight execution accelerator**

Their interaction is rooted in the concentration of measure phenomenon. When a predicate exhibits high θ_GLS (strong spatial clustering), it isolates a specific, dense sub-manifold. Traditional static quantization codebooks lack the resolution to differentiate distances within this isolated cluster, leading to distance collapse. iFVS mitigates this by shifting the codebook centroids toward the localized manifold, dramatically increasing resolution among the valid vectors.

They compose synergistically: θ_GLS dictates the raw topological requirement for graph oversampling. If the optimizer determines that the physical execution plan will utilize the iFVS infrastructure, this serves as a damping coefficient on the required oversampling. Because iFVS significantly improves PQ accuracy within the cluster, the algorithm requires far fewer candidate evaluations to achieve target recall.

### 8. Failure Modes and Cost Model Propagation

**False Positives (Overestimating Spatial Concentration):**
- Cost-delta function underestimates required oversampling
- Premature truncation of search radius (e.g., setting `ef_search` too low)
- Catastrophic failure to meet target recall SLO
- In RAG applications: hallucination due to missing context vectors

**False Negatives (Underestimating Spatial Concentration):**
- Massive oversampling factor prescribed
- Waste of substantial CPU cycles
- Latency spikes and resource contention
- Recall target met but at excessive cost

**Bounding Worst-Case Execution:**

Given global selectivity `σ_g = sel(p)`, the absolute worst-case required exploration under perfect uniform distribution is bounded by `k / σ_g`. The cost-delta function is designed to approach this maximum:

```
lim_{θ_GLS → 0} c(sel(p), θ_GLS) = k / sel(p) - k
```

As θ_GLS approaches 1, the required oversampling approaches a hardware-specific constant. By clamping the output of the cost-delta function between these theoretical bounds, the unified optimizer guarantees that even in total catalog failure, the physical execution plan will degrade to a predictable, bounded latency profile.

### 9. Validation Methodology and Benchmarking

**Dataset foundation:** MoReVec relational dataset (768-dimensional dense text embeddings with rich scalar/categorical metadata).

**Synthetic predicate generation via controlled graph traversal:**

- **Highly Concentrated (θ_GLS ≈ 1):** Random seed node, BFS expansion through geometric neighbors until target `σ_g` achieved
- **Uniformly Distributed (θ_GLS ≈ 0):** Uniform random sampling across entire corpus
- **Intermediate (θ_GLS ≈ 0.5):** Multiple disjoint seed nodes, small localized clusters grown to target `σ_g`

**Empirical measurement protocol:** Execute 10,000 exact k-NN queries across the spectrum. For each query, incrementally expand the oversampling parameter (e.g., `ef_search`) until result set perfectly matches exact brute-force baseline (recall = 0.99).

**Validation criteria:** Plot actual oversampling required for 0.99 recall (y-axis) against predicted θ_GLS (x-axis). A correctly functioning implementation yields a strictly inverse monotonic relationship. If variance of plot residuals remains within a tight tolerance (e.g., ±5%), the implementation is validated for production.

### 10. Conclusion and Open Research Directives

The formal specification resolves the critical impasse blocking FVS pushdown law deployment. By anchoring θ_GLS in spatial autocorrelation principles and operationalizing it via SIMD-accelerated histogram-based region variance, the system reliably predicts oversampling requirements within the sub-millisecond ECQO budget.

**Open sub-questions:**
1. Optimal strategy for determining C spatial partitions (k-means vs Leiden pages vs data-driven quantization)
2. Integration of θ_GLS with multi-table join predicates where local vector selectivity depends on dynamic cross-table foreign-key materialization

---

## Part 2: Critical Review of the θ_GLS Formal Specification

The formal specification is a substantial upgrade over the prior θ_GLS research response. The Moran's I adaptation is a principled formal foundation, the decoupling from `q` makes the metric *predictive* rather than *retrospective*, and the histogram-based region variance approximation is concrete enough to implement within the sub-millisecond ECQO budget. The asymptotic bound, the failure mode analysis, and the validation methodology are all sharp.

But the `c(sel, θ_GLS)` function is still the blocking item, the [0, 1] bound makes an explicit design choice (no depleting case) that should be acknowledged, and several smaller items need to land.

### What's Strong

1. The **Moran's I derivation** is a principled formal foundation. Adapting spatial autocorrelation theory to high-D vector graphs (using HNSW as the proximity graph) is the right theoretical move. This puts θ_GLS on the same mathematical footing as a standard geostatistics measure, not a hand-wavy heuristic.
2. **Decoupling from `q`** is the right call. The prior response's per-query `θ_GLS(q, φ)` had a fundamental problem: computing it requires executing the query, which violates the planning budget. The closed-form Moran's I formula computes θ_GLS from the predicate's distribution on the corpus, not the query. This makes it a *predictive* metric.
3. The **histogram-based region variance approximation** is the right operational form. The closed-form Moran's I requires the full proximity graph; the region-level variance is a coarse-grained macroscopic proxy that composes with the Leiden partitioning, has bounded cost (< 10 μs), and captures the macroscopic spatial variance.
4. The **[0, 1] bound** is correct for high-D spaces. The "concentration of measure" justification for excluding negative values is real — in high-D, the spatial autocorrelation of any binary predicate is overwhelmingly positive.
5. The **asymptotic bound** `lim_{θ_GLS → 0} c(sel, θ_GLS) = k/sel - k` is the right safety mechanism. It guarantees worst-case oversampling under complete uncertainty.
6. The **iFVS interaction** is well-articulated. θ_GLS dictates base topological oversampling; iFVS improves distance accuracy within the localized manifold. They compose: high θ_GLS triggers iFVS injection, which reduces required oversampling.
7. The **validation methodology** is rigorous. Synthetic predicates with controlled Moran's I values (via BFS from seed nodes), exact k-NN baseline, plotting actual-vs-predicted oversampling. This is a real validation strategy.
8. The **cost analysis is precise.** < 10 μs for C=1024 regions with AVX-512 SIMD. The ECQO budget breakdown (150 μs AST, 50 μs histogram, 600-700 μs ANN probe, 100 μs plan costing, 0-100 μs for θ_GLS) leaves room.
9. The **catalog design** is well-specified. Hash table with predicate AST fingerprint as key, float array as value. ~40MB for 10K predicates, fits in main memory.
10. The **EMA update strategy** with 10-second or 10,000-insert flushes is the right asynchronous pattern. Avoids WAL bottlenecks while keeping θ_GLS estimates fresh.

### What Needs to Land

1. **The `c(sel, θ_GLS)` function is still unspecified.** The response gives the asymptotic limit but not the closed form. The FVS law's right-hand side `topk(k + c(sel(p), θ_GLS), R, d)` is unimplementable without it. The response hints at `c → k/sel - k` for θ_GLS = 0 and `c → constant` for θ_GLS = 1, but the interpolation between these is unspecified. The validation methodology should produce the fitted form.

   **Candidate form:** the geometric interpolation `c(sel, θ_GLS) = (k/sel - k) × (1 - θ_GLS)^α + c_min × θ_GLS^α` for some power `α > 1` that controls how fast the oversampling drops as clustering increases. This gives:
   - `c(0.1, 0) = (k/0.1 - k) × 1 + c_min × 0 = 9k` (uniform 10% selective)
   - `c(0.1, 0.5) = 9k × 0.5^α + c_min × 0.5^α` (interpolation)
   - `c(0.1, 1) = 0 × 0 + c_min × 1 = c_min` (perfectly clustered)

   The validation methodology should fit α from data.

2. **The [0, 1] bound is an explicit design choice, not a fact.** The response excludes negative values based on "concentration of measure" but doesn't cite the specific theorem. The prior response's `ρ_q ∈ [-1, 1)` formulation allowed depleting cases. The choice between them is:
   - **[0, 1] (this response):** simpler implementation, no depleting case, justified in high-D. May miss pathological cases in low-D or adversarial predicates.
   - **[-1, 1) (prior response):** captures depleting case, more general, slightly more complex.

   The cost model should make this an explicit decision. For high-D vector search, [0, 1] is probably correct. For mixed-dimensional workloads (some predicates on low-D attributes), [-1, 1) may be needed. Recommendation: [0, 1] for the high-D vector path, [-1, 1) as a fallback for low-D attribute predicates.

3. **The region-level variance approximation loses inter-region connectivity information.** Two predicates could have the same per-region selectivity distribution but different inter-region connectivity. The Moran's I captures connectivity via the weight matrix W; the region-level variance doesn't. The cost model should account for this information loss. Possible correction: weight the region-level variance by inter-region edge density (from the Leiden partitioning).

4. **The cost analysis assumes AVX-512 SIMD, but not all CPUs have it.** Apple Silicon (M1/M2/M3), older AMD (pre-Zen 4), and most ARM processors lack AVX-512. The fallback is scalar code, which is ~16x slower. The implementation should use runtime feature detection (`golang.org/x/sys/cpu`) and have a scalar fallback path. The cost should be specified per architecture, not just for AVX-512.

5. **The EMA staleness × Leiden partitioning staleness interaction is unspecified.** A 10-second EMA window means θ_GLS estimates can be stale by up to 10 seconds during high write load. The Leiden partitioning has its own staleness function `g(s)` from the unified optimizer spec. The two staleness sources compound. The cost model should specify: when both are stale, how do they propagate through the joint recall distribution?

6. **The failure mode propagation through the joint recall distribution is incomplete.** The response says θ_GLS errors "propagate into Formula 8 as a premature truncation of the search radius" but doesn't give the explicit propagation. The cost model should specify:
   - `θ_GLS_error` (the gap between estimated and true θ_GLS) → error in `c(sel, θ_GLS)` → error in `μ_fanout` and `σ_fanout` in Formula 8
   - The θ_GLS uncertainty should be tracked as a separate variance term in the joint distribution
   - The LCB safety mechanism should account for θ_GLS uncertainty alongside PLAID and staleness

7. **The iFVS injection threshold is unspecified.** The response says "high θ_GLS triggers iFVS injection" but doesn't give the threshold. Recommendation: calibrate via benchmark, but starting point: inject iFVS when `θ_GLS > 0.3` (mild spatial clustering) AND `sel < 0.1` (selective enough to benefit). Below 0.3, the iFVS overhead outweighs the benefit. Above 0.3, the benefit is clear.

8. **The synthetic predicate generation uses BFS from seed nodes, which generates *globular* clusters.** Real-world predicates may have *filamentary* or *planar* distributions (correlations along specific embedding dimensions, not in all directions). The validation methodology should include these shapes. Reference: the spatial statistics literature distinguishes between globular, filamentary, and planar point patterns.

9. **The multi-table join integration is flagged as an open sub-question but is non-trivial.** When the predicate is `category = 'X' AND author_id IN (SELECT id FROM authors WHERE country = 'US')`, the spatial distribution of `category = 'X'` depends on the foreign-key relationship with the authors table. The θ_GLS computation needs to handle correlated predicates, not just single-table predicates. The cost model should specify a decomposition: `θ_GLS(p1 AND p2) ≈ min(θ_GLS(p1), θ_GLS(p2))` in the worst case, or a learned composition function.

10. **The compound predicate composition is unspecified for boolean operators.** Conjunctions (AND), disjunctions (OR), and negations (NOT) of predicates have different spatial distributions. Standard spatial statistics: Moran's I for conjunctions is approximately the min of the component Moran's I, for disjunctions is approximately the max, for negations is 1 - θ_GLS(¬p). But this needs empirical validation. The cost model needs a θ_GLS composition function for the boolean operators.

11. **The "concentration of measure" claim is hand-wavy.** The response says "due to the concentration of measure phenomenon" but doesn't cite the specific result. For a rigorous treatment, cite the specific concentration-of-measure theorem (e.g., the Lévy family of concentration inequalities, or the Johnson-Lindenstrauss lemma as a related result). The reference to a specific theorem strengthens the claim.

12. **The validation should use multiple datasets, not just MoReVec.** The prior response mentioned ANN-Benchmarks datasets (SIFT-1M, GloVe-100, Deep1B) and FAEVAL. The current response uses MoReVec exclusively. Cross-dataset validation strengthens the result. The Moran's I values for the SIFT-1M predicates can be computed offline and used as additional ground truth.

---

## Part 3: What's Still Missing

| Item | Severity | Type |
|---|---|---|
| `c(sel, θ_GLS)` closed form (point 1) | **Blocking** | Law incomplete |
| [0, 1] vs [-1, 1) explicit design choice (point 2) | High | Design decision |
| Inter-region connectivity correction (point 3) | Medium | Approximation error |
| Non-AVX-512 fallback (point 4) | Medium | Implementation portability |
| EMA × Leiden staleness interaction (point 5) | High | Compound staleness |
| Failure mode propagation through Formula 8 (point 6) | High | Model extension |
| iFVS injection threshold (point 7) | Medium | Calibration parameter |
| Filamentary/planar predicate shapes (point 8) | Medium | Validation gap |
| Multi-table join integration (point 9) | High | Open research |
| Compound predicate composition (point 10) | High | Open research |
| Concentration-of-measure citation (point 11) | Low | Documentation |
| Multi-dataset validation (point 12) | Medium | Validation methodology |
| Plan search algorithm (from prior reviews) | **Blocking** | Architecture gap |
| Correlation correction for joint distribution (from prior reviews) | **Blocking** | Stochastic model gap |
| HNSW layer structure in MVCC slab (from prior reviews) | Blocking | Storage spec gap |
| xmin/xmax commit protocol (from prior reviews) | Blocking | Storage spec gap |

---

## Part 4: Next Research Target

The `c(sel, θ_GLS)` function is still the blocking item. The Moran's I derivation gives us a clean θ_GLS, but the FVS law's right-hand side is still unimplementable without the closed form. The candidate form `c(sel, θ_GLS) = (k/sel - k) × (1 - θ_GLS)^α + c_min × θ_GLS^α` is a starting hypothesis that the validation methodology should fit.

The [0, 1] vs [-1, 1) design choice is also high-priority. The Moran's I derivation argues for [0, 1] in high-D; the prior response's per-query sampling argued for [-1, 1). The cost model needs to commit to one.

The next research target is the **`c(sel, θ_GLS)` functional form**: an empirical fitting study using the validation methodology from this response, with controlled predicates across the (sel, θ_GLS) space, producing a fitted function and a residual model for the lower confidence bound. The fitted form gates the FVS pushdown law's implementation. The [0, 1] vs [-1, 1) decision should be made explicit in the same research pass.

---

## Part 5: Summary

This doc preserves the θ_GLS formal specification and adds a critical review. The Moran's I adaptation is a principled formal foundation: the closed-form formula computes θ_GLS as a normalized spatial autocorrelation of the predicate indicator over the HNSW proximity graph, decoupled from the query vector, making it a predictive rather than retrospective metric. The histogram-based region variance approximation operationalizes this for sub-millisecond ECQO budgets at < 10 μs cost, with a ~40MB catalog footprint for 10K predicates. The asymptotic bound and iFVS interaction are correctly specified.

The critical review identifies the `c(sel, θ_GLS)` function as the blocking item — the FVS law's right-hand side is still unimplementable without it. The [0, 1] bound is an explicit design choice that should be acknowledged (no depleting case, justified in high-D). Several smaller items — inter-region connectivity correction, AVX-512 fallback, EMA-Leiden staleness interaction, failure mode propagation, iFVS injection threshold, multi-dataset validation, multi-table joins, and compound predicate composition — round out the gaps.

The build can start on Phase 8 (FVS) with the θ_GLS implementation, gated on the `c(sel, θ_GLS)` function and the [0, 1] vs [-1, 1) decision. The unified optimizer's joint recall distribution is now concrete enough to compose with θ_GLS, but the plan search algorithm and correlation correction remain blocking for the optimizer's plan enumeration. The next research target is the `c(sel, θ_GLS)` functional form, with the [0, 1] vs [-1, 1) design decision as a secondary deliverable.
