# θ_GLS Computation: Research Response and Critical Review

**Date:** 2026-08-01
**Status:** Reference document — θ_GLS research response preserved with critical analysis
**Preread:**
- `foundations-of-unified-query-algebra-08-01-2026.md` — the FVS pushdown law
- `optimizer-cost-model-research-response-08-01-2026.md` — the cost model
- `optimizer-cost-model-gap-closures-08-01-2026.md` — first-pass gap closures
- `optimizer-cost-model-second-pass-gap-closures-08-01-2026.md` — second-pass gap closures
- `unified-optimizer-specification-08-01-2026.md` — the joint recall distribution

---

## What This Document Is

This doc preserves the research response to the θ_GLS computation prompt. The response provides a formal definition of θ_GLS as a Möbius-like transform of the local/global selectivity ratio, four computable implementations with cost analysis, a catalog design, a validation methodology, and the relationship with the iFVS weight matrix.

The doc also adds a critical review. The FVS pushdown law's right-hand side is still unimplementable without the `c(sel, θ_GLS)` functional form, and the depleting case (`ρ_q < 0`) is unaddressed. Several smaller items also need to land.

The research response is the *answer*. The critical review is the *verification*.

---

## Part 1: The Research Response

The response provided the following nine sections.

### 1. Formal Definition of θ_GLS

Recent FVS work on MoReVec introduces **Global-Local Selectivity (GLS) correlation** as a per-query scalar `ρ_q ∈ [-1, 1)` that quantifies how a filter predicate `φ` interacts with the query's local neighborhood. It defines:

**Global selectivity:**
```
σ_g = |{v ∈ D | φ(v)=1}| / N
```

**Local selectivity around query `q`:**
```
σ_l = |{v ∈ N_q | φ(v)=1}| / k
```
where `N_q` is the set of `k` nearest neighbors of `q` under the vector metric.

**The ratio:**
```
r = σ_l / σ_g
```

is transformed by a Möbius (bilinear) mapping:
```
ρ_q = (r - 1) / (r + 1) ∈ [-1, 1)
```

with:
- `ρ_q = 0`: filter behaves neutrally relative to the local neighborhood (local ≈ global selectivity)
- `ρ_q > 0`: **enriching** filter (local selectivity > global, filter concentrates relevant vectors near the query)
- `ρ_q < 0`: **depleting** filter (local selectivity < global, relevant vectors pushed away from the query's neighborhood)

For libraVDB, **θ_GLS is this GLS correlation**. More precisely:

- **Per-query definition:**
  ```
  θ_GLS(q, φ) := ρ_q
  ```
- **Per-predicate global profile:**
  ```
  Θ_GLS(φ) := E_{q ~ Q(φ)} [ρ_q]
  ```
  where `Q(φ)` is the distribution of queries that co-occur with predicate `φ` in the workload.

So θ_GLS is primarily a **scalar function of (query, predicate)**, with a derived per-predicate expectation stored in the catalog for cheap reuse.

### 2. Scope: Per-Corpus vs Per-Query

The MoReVec/GLS paper treats `ρ_q` as a **query-specific metric**, computed per query and per predicate, then aggregated to analyze index behavior across selectivity regimes and correlations.

For an optimizer, a pure per-query definition is accurate but expensive; a pure per-corpus constant is cheap but often wrong when predicates concentrate vectors in different subspaces. The correct compromise:

- θ_GLS has two levels:
  - **Per-query scalar** `θ_GLS(q, φ)` (used when cheap sampling is possible)
  - **Per-predicate profile** `Θ_GLS(φ)` (used as a prior when ECQO cannot afford a fresh probe)

The recommended design:

- Treat θ_GLS as **per-query-per-predicate**, but:
  - Maintain `Θ_GLS(φ)` in the catalog as a learned prior
  - At planning time, compute a **fast, low-k local sample** to refine `θ_GLS(q, φ)` around this prior when budget allows

### 3. Computable Implementations

#### 3.1 Sampling-Based (Local ANN Probe)

**Inputs:**
- Query vector `q`
- Predicate `φ` with selectivity estimate `σ_g` (relational stats)
- Vector index `I` supporting small-k ANN probes

**Computation:**
1. Run a localized ANN probe: `N_q = ANN(q, k_GLS)` with small `k_GLS` (e.g. 64–128)
2. Compute `σ_l = |{v ∈ N_q | φ(v)=1}| / k_GLS`
3. Use `σ_g` from relational selectivity stats
4. Compute `r = σ_l / σ_g`, then `θ_GLS(q, φ) = (r-1)/(r+1)`, clamping `r` to avoid division by very small `σ_g`

**Cost:**
- CPU/latency: one tiny ANN probe (k ≲ 128) plus one predicate evaluation on `k` candidates; tens to hundreds of microseconds, well within a sub-millisecond ECQO budget
- Memory: negligible
- Accuracy: high for predicates whose spatial effect is local around the query; noisy for extreme low-selectivity filters (few hits even in `N_q`)
- Failure modes:
  - If `σ_g` is very small and `σ_l` has 0 hits, `r` is unstable; must fall back to a conservative prior `Θ_GLS(φ)`
  - If the ANN probe itself is approximate with low recall, `σ_l` can be biased; the unified recall model must treat θ_GLS as a random variable with its own uncertainty

#### 3.2 Histogram-Based (Region-Level GLS)

**Inputs:**
- Predefined regions: Leiden pages, k-means clusters, or Voronoi cells in vector space
- Per-region counts: `N_region` and `N_region,φ`

**Computation:**
- Precompute per-region local selectivity: `σ_l,region = N_region,φ / N_region`
- Global selectivity `σ_g` from relational stats
- For a query, identify its top-k regions (e.g. via coarse centroid lookup) and aggregate `σ_l,region` weighted by region proximity to approximate `σ_l`
- Compute `θ_GLS` as in sampling-based, but using region histograms instead of a fresh local probe

**Cost:**
- CPU: O(#regions_touched) histogram lookups per query; very cheap
- Memory: per (region, predicate) map; can be large if many predicates × many regions
- Accuracy: good if regions align with semantic neighborhoods and predicate effects; poor if filters don't align with region boundaries
- Failure modes:
  - If region partition is coarse or misaligned, θ_GLS estimates are blurred and can miss sharp local concentration

#### 3.3 Learned Predictor

**Inputs:**
- Predicate features: type (range, equality), estimated `σ_g`, attribute domain stats, AST fingerprint
- Data summary: global corpus stats (cardinality, attribute distributions), maybe embedding cluster stats
- Training labels: empirical per-query `ρ_q` measured offline via sampling-based method on training workload

**Computation:**
- Train a regression model `f(features) ≈ θ_GLS(q, φ)`
- At planning time, evaluate `f` once per query-predicate

**Cost:**
- CPU: single small model inference (< 100 µs) per query; fits ECQO budget
- Memory: model weights stored in catalog
- Accuracy: depends on training coverage; can generalize across predicates but may fail on unseen patterns
- Failure modes:
  - Model drift as corpus changes; requires retraining
  - Poor extrapolation for rare extreme filters; must be guarded by conservative bounds

#### 3.4 Analytical (Predicate-Type-Specific)

For simple predicates on scalar attributes with known distributions (e.g. uniform range, categorical equality), spatial selectivity can be approximated via independence assumptions: if embedding and attribute are independent, then `σ_l ≈ σ_g` and `θ_GLS ≈ 0`; if known correlation exists (e.g. attribute value strongly associated with certain embedding region), approximate `θ_GLS` via correlation coefficient between attribute value and embedding cluster ID.

**Recommendation:**
For libraVDB's ECQO:

- Use **sampling-based θ_GLS** as the primary method (small ANN probe), with:
  - A **histogram-based prior** per predicate/type
  - Optionally a learned predictor to smooth estimates when sampling is noisy

This hybrid gives a concrete, computable scalar per query while fitting the sub-millisecond budget and composing cleanly with Exqutor's existing vector-cardinality probes.

### 4. Planning-Time Cost Budget

Exqutor's ECQO runs lightweight vector index probes during planning to estimate vector cardinalities, staying within a sub-millisecond planning budget. For libraVDB:

- Total ECQO budget per query: ~1.0 ms
- Proposed allocation:
  - θ_GLS sampling probe (ANN on `k_GLS` = 64–128) + `k` predicate evaluations: **≤ 0.2–0.3 ms**
  - Remaining **0.7–0.8 ms** for:
    - Localized ANN probe for `σ_vec` (vector cardinality)
    - Histogram lookup for relational selectivity `σ_rel`
    - Any other ECQO probes (e.g., small HNSW layer sampling)

If the θ_GLS probe threatens to exceed its budget (e.g., due to cold cache or heavy contention), the optimizer should fall back to a cheaper prior `Θ_GLS(φ)` for that query.

### 5. Catalog Representation

**5.1 Per-Corpus Constant:**
- Representation: single `theta_gls_global: float`
- Storage: negligible
- Update: periodic recompute over workload sample
- Read: one float per plan; trivial
- Staleness: high for predicates whose spatial behavior differs from global average

**5.2 Per-Predicate-Type Function:**
- Representation: `theta_gls(predicate_type, sel) → float`, e.g. a small table or parametric function per predicate type
- Storage: small per type; e.g. a grid over selectivity buckets
- Update: offline recomputation, amortized over workload changes
- Read: O(1) lookup per query
- Staleness: moderate; depends on how much predicate distributions drift

**5.3 Learned Model Weights:**
- Representation: model artifact (e.g. small neural net or GBM) in catalog
- Storage: modest (KB–MB)
- Update: retraining required as corpus/workload drift
- Read: single inference per query; cheap but not negligible

**5.4 Region-Level Map:**
- Representation: `theta_gls_map: Map[region_id, float]` per predicate
- Storage: potentially large (regions × predicates)
- Update: expensive; requires periodic recomputation across regions
- Read: a few lookups per query (regions touched)

**Recommendation:**
- Store:
  - A **per-predicate-type function** `Θ_GLS(φ; σ_g)` as the catalog prior
  - Optionally, region-level θ_GLS for a small set of "hot" predicates and regions, where behavior is highly skewed

The per-query θ_GLS is then computed via sampling-based probe, using the catalog prior as an initial estimate and guardrail.

### 6. Interaction with iFVS Weight Matrix W

iFVS uses a learned weight tensor `W ∈ R^{M × r × K × d}` to generate filter-aware PQ codebooks, training W on `(query, predicate) → recall` pairs. Conceptually:

- W encodes **how filters alter the codebooks' partitioning of the embedding space**
- θ_GLS encodes **how filters alter local neighborhood selectivity relative to global selectivity**

They are **complementary**:

- W is **index-time / structure-level**: it shapes the codebooks and regions
- θ_GLS is **plan-time / behavior-level**: it adjusts oversampling based on current filter-query interaction

Interplay:
- θ_GLS computation can **reuse iFVS infrastructure**:
  - Use iFVS's learned codebooks and W to define regions where local selectivity `σ_l` is measured
  - Use iFVS's training logs (which include per-query recall with filters) to seed the initial `Θ_GLS(φ)` prior
- In the cost model:
  - W affects the **base recall and cost** of the index (iFVS vs non-iFVS)
  - θ_GLS affects the **oversampling factor** `c(sel(p), θ_GLS)`

So the joint recall distribution's vector-search component depends on both W (index effectiveness) and θ_GLS (oversampling adequacy); they're not the same mechanism but can share data paths.

### 7. Failure Modes When θ_GLS Is Wrong

θ_GLS controls the oversampling factor in the FVS pushdown law:
```
topk(k, σ_p(R), d) ≈ σ_p(topk(k + c(sel(p), θ_GLS), R, d))
```

- If θ_GLS is **too small** (overestimates spatial concentration / enrichment):
  - `c` is too low; not enough candidates
  - Recall drops below target; joint recall LCB falls under required accuracy
- If θ_GLS is **too large** (underestimates concentration / treats filter as depleting):
  - `c` is too high; oversampling is excessive
  - Latency and CPU cost higher than necessary, but recall is safe

Propagation into joint recall distribution (unified optimizer formula 8):
- θ_GLS affects the **fanout/oversampling** term's mean and variance (`μ_fanout`, `σ_fanout`):
  - Underestimation of θ_GLS shrinks `μ_fanout`, increases recall variance, lowering the joint LCB
  - Overestimation of θ_GLS inflates `μ_fanout`, increasing cost but tightening recall variance upward

Worst-case oversampling factor under θ_GLS uncertainty:
- Let `θ_GLS ∈ [θ̂ - Δ, θ̂ + Δ]` be an uncertainty interval
- Define `c_min = c(sel, θ̂ - Δ)`, `c_max = c(sel, θ̂ + Δ)`
- For safety, the optimizer should use **`c_max`** when the target accuracy is strict, treating θ_GLS as a random variable and using the lower confidence bound in the joint recall calculation, just as for PLAID and staleness.

### 8. Validation Methodology

To validate θ_GLS:

1. **Datasets**:
   - Use ANN-Benchmarks datasets (e.g., SIFT-1M, GloVe-100, Deep1B) extended for FVS, as done in the GLS paper's MoReVec benchmark
   - Use FAEVAL or LDBC FVS-style benchmarks where available for realistic mixed workloads

2. **Synthetic predicates**:
   - Design filters with controlled global selectivity `σ_g` and spatial concentration:
     - Neutral: independent attribute, `θ_GLS ≈ 0`
     - Enriching: attribute concentrated in a cluster near query vectors, `θ_GLS > 0`
     - Depleting: attribute concentrated far from query vectors, `θ_GLS < 0`

3. **Procedure**:
   - For each predicate and query:
     - Compute θ_GLS via the chosen implementation (sampling + prior)
     - Predict oversampling factor `c_pred(sel, θ_GLS)` to meet a target recall at confidence level `z`
   - Empirically measure:
     - Recall at varying actual oversampling factors `c`
     - Minimum `c_actual` needed to hit the target recall with confidence
   - Compare `c_pred` vs `c_actual`:
     - Compute error bars and calibration curves
     - Adjust `c(·)` and θ_GLS estimation until prediction error is within acceptable tolerance (e.g. predicted oversampling within 2x of actual, and recall LCB matching target in ≥95% of cases)

4. **Cross-check with ACORN/NaviX/JAG**:
   - Run the same benchmark over ACORN, NaviX, JAG implementations to validate that θ_GLS's behavior is consistent and useful across algorithms, not just a single engine.

### 9. Open Sub-Questions

- Precise functional form of `c(sel, θ_GLS)`: linear in θ_GLS? Exponential in `|θ|`? Needs empirical fitting across datasets; current literature only hints at GLS correlation's effect on recall but does not provide a closed-form oversampling function
- Dependency between θ_GLS and staleness: GLS assumes a stable index; under heavy write/migration, θ_GLS estimates may drift more rapidly and should be co-modeled with the staleness function `g(s)`
- Integration with learned optimizers (Neo, Bao-style): θ_GLS could be one feature among many in a learned plan-selection model; open research question is whether explicit θ_GLS adds value beyond raw query and predicate features

---

## Part 2: Critical Review of the θ_GLS Research Response

The research response is the most concrete of the θ_GLS work — it provides a closed-form definition, four computable implementations with cost analysis, a catalog design, and a validation methodology. The Möbius-like transform of the local/global selectivity ratio into `[-1, 1)` is a clean scalar, and the per-query + per-predicate design is the right compromise.

But the FVS law still has a gap: the `c(sel(p), θ_GLS)` functional form is unspecified, and the depleting case is unaddressed. Several smaller items also need to land.

### What's Strong

1. The **Möbius-like transform** of the local/global selectivity ratio into `[-1, 1)` is a clean scalar. The interpretation (neutral / enriching / depleting) is intuitive and matches the FVS literature. The MoReVec 2024 attribution is the right source.
2. The **per-query + per-predicate design** (compute `θ_GLS(q, φ)` at planning time, with `Θ_GLS(φ)` as a learned prior in the catalog) is the right compromise between accuracy and cost.
3. The **four implementations** are correctly characterized with their costs, accuracies, and failure modes. The recommendation (sampling-based primary, histogram prior as fallback, learned predictor to smooth) is sound.
4. The **cost budget allocation** (0.2-0.3 ms for θ_GLS, 0.7-0.8 ms for other ECQO probes) fits within the sub-millisecond budget.
5. The **interaction with iFVS** correctly identifies W as index-time/structure-level and θ_GLS as plan-time/behavior-level. They compose, not share. The reuse of iFVS codebooks to define regions is a real efficiency.
6. The **failure modes analysis** with worst-case oversampling factor `c_max` is the right safety mechanism.
7. The **validation methodology** uses standard benchmarks (ANN-Benchmarks, FAEVAL, LDBC) and proposes cross-checking with ACORN/NaviX/JAG.
8. The **open sub-questions** (c function, staleness interaction, learned optimizer integration) are real and well-identified.

### What Needs to Land

1. **The `c(sel(p), θ_GLS)` function is unspecified.** The response says "Linear in θ_GLS? Exponential in |θ|? Needs empirical fitting across datasets; current literature only hints at GLS correlation's effect on recall but does not provide a closed-form oversampling function." This is a real gap. Without `c(sel, θ_GLS)`, the FVS pushdown law's right-hand side is unimplementable. The cost model needs the functional form.

   **Candidate form** based on the FVS literature: `c(sel, θ_GLS) ≈ k / (sel × (1 + θ_GLS))` for the enriching case. This gives:
   - `c(0.1, 0) = k / 0.1 = 10k` (oversample 10x for a 10% selective filter, neutral)
   - `c(0.1, 0.5) = k / (0.1 × 1.5) ≈ 6.7k` (oversample 6.7x for an enriching filter)
   - `c(0.001, 0) = k / 0.001 = 1000k` (oversample 1000x for a 0.1% selective filter)
   - `c(0.001, -0.5) = k / (0.001 × 0.5) = 2000k` (oversample 2000x for a depleting filter — see point 2 below)

   This is a starting hypothesis; the response correctly notes it needs empirical fitting. The validation methodology should produce the fitted form.

2. **The "depleting" case (`ρ_q < 0`) behavior is unclear.** When the filter pushes relevant vectors away from the query's neighborhood, the local probe returns mostly irrelevant vectors. The `c(sel, θ_GLS)` formula above gives *more* oversampling for the depleting case, which is wrong — the right answer is probably to switch strategies entirely (e.g., a global HNSW search with the filter, no oversampling; or a different access path). The response doesn't address this.

   The depleting case is a *plan-shape* decision, not a parameter. The optimizer should detect `ρ_q < -threshold` (say, -0.3) and switch from "oversample + post-filter" to "filter-first + global search" or to in-filtering. The cost model should specify this branching.

3. **The θ_GLS × staleness interaction should be in the model.** When the Leiden partitioning is stale, the regions used for histogram-based θ_GLS are wrong, and the local ANN probe returns vectors that aren't co-located with the global distribution. The staleness function `g(s)` from the unified optimizer spec should enter the θ_GLS computation as a discount factor on the local sample.

   Specifically: the variance of `θ_GLS` should grow with `s`. The cost model should report `θ_GLS ± σ_θ_GLS(s)` where `σ_θ_GLS(s) = σ_θ_GLS(0) × (1 + α × s)` for some calibration constant `α`. This propagates through the FVS law's oversampling factor and the joint recall distribution.

4. **The "Möbius transform" claim is informal.** The formula `(r-1)/(r+1)` is a standard sigmoid-like transform, not strictly the Möbius transform (which is a rational function `(ax+b)/(cx+d)` with `ad - bc ≠ 0`). The naming is approximate. The formula is correct; call it a "Möbius-like" or "sigmoid" transform.

5. **The sampling-based fallback threshold is unspecified.** For a highly selective predicate (`sel = 0.001`), `k=128` is unlikely to contain even one satisfying vector. The response says "must fall back to a conservative prior `Θ_GLS(φ)`" but doesn't say when. Specification: if 0 hits in `N_q` AND `sel < 1/k`, fall back to prior; otherwise use the sample. The threshold is a configuration parameter.

6. **The histogram-based region definition is open.** The response mentions "Leiden pages, k-means clusters, or Voronoi cells" but doesn't pick one. Recommendation: **Leiden pages** as primary (they're already in the catalog from the HMGI partitioning), with k-means clusters as a fallback if the Leiden partitioning is stale. The two region types are interchangeable for θ_GLS computation; Leiden pages have the advantage of being free (already maintained).

7. **The learned predictor is underspecified.** What model? What features exactly? What's the training procedure? The response mentions "regression model `f(features) ≈ θ_GLS(q,φ)`" but doesn't specify. Recommendation: **gradient-boosted trees** (lightweight inference, ~100μs), with features being `(predicate type, sel_g, predicate AST fingerprint, data summary stats)`. Cite Neo/Bao for the learned-optimizer pattern. Training labels come from the sampling-based implementation run on a workload replay.

8. **The "interaction with iFVS" claim that they can "share data paths" is hand-wavy.** The iFVS weight matrix W is a 4D tensor (`M × r × K × d`), much larger than θ_GLS. They don't naturally share data. The right framing: iFVS provides the *index-level* optimization (how to encode vectors for filter-aware retrieval), and θ_GLS provides the *plan-time* optimization (how much to oversample for the filter). They compose, not share. The reuse is the region definition (Leiden pages or iFVS codebook cells), not the data structures.

9. **The `c_max` safety mechanism may over-provision excessively.** If `θ_GLS ∈ [-1, 1)`, the upper bound is 1, which gives `c_max = c(sel, 1)`. For very selective predicates (`sel = 0.001`), `c_max` could be 1000x. The safety mechanism should be: use `c_max` only when the target accuracy is strict AND the LCB is below target with `c_hat` (point estimate). Otherwise, use `c_hat`. The cost model needs a branching decision, not a global `c_max` always-on.

10. **The histogram-based storage can be huge.** For 100 predicates × 1000 regions, that's 100K entries × 4 bytes = 400KB. With 10K predicates × 100K regions (large enterprise), that's 4GB. The cost model needs a storage budget and a sparse representation: only store entries where `σ_l` differs significantly from `σ_g` (say, |σ_l - σ_g| > 0.1). The remaining entries use the prior.

11. **The validation methodology uses synthetic predicates with controlled spatial concentration — necessary but not sufficient.** Real-world predicates may have complex spatial distributions (multi-modal, correlated with multiple attributes, time-varying). The validation should include:
    - Real-world query logs from production systems (if available; if not, generated by a workload simulator)
    - Predicates with multiple attributes (e.g., `category = 'X' AND price < 100`)
    - Predicates with temporal drift (e.g., `published_at > recent_date`)

12. **The "GLS correlation" name is a recent concept (MoReVec 2024+).** The earlier FVS literature (ACORN, NaviX, JAG, iFVS, Exqutor) doesn't use this exact term. The response is presenting a recent formalization. This is fine, but should be noted: the prior FVS literature had informal notions of filter-vector interaction, and MoReVec's GLS is a formalization, not a widely-adopted standard. The implementation should not depend on MoReVec-specific terminology in the public API.

---

## Part 3: What's Still Missing

| Item | Severity | Type |
|---|---|---|
| `c(sel, θ_GLS)` functional form (point 1) | **Blocking** | Law incomplete |
| Depleting-case strategy switch (point 2) | **Blocking** | Plan-shape gap |
| θ_GLS × staleness interaction (point 3) | High | Model extension |
| Sampling-based fallback threshold (point 5) | High | Implementation detail |
| Histogram-based region definition (point 6) | Medium | Implementation choice |
| Learned predictor specification (point 7) | High | Algorithm gap |
| iFVS data path clarification (point 8) | Low | Documentation |
| `c_max` safety mechanism branching (point 9) | High | Algorithm gap |
| Histogram storage budget and sparsity (point 10) | Medium | Storage design |
| Validation with real-world predicates (point 11) | High | Empirical validation |
| MoReVec-specific terminology in API (point 12) | Low | API design |
| Plan search algorithm (from prior reviews) | **Blocking** | Architecture gap |
| Correlation correction for joint distribution (from prior reviews) | **Blocking** | Stochastic model gap |
| HNSW layer structure in MVCC slab (from prior reviews) | Blocking | Storage spec gap |
| xmin/xmax commit protocol (from prior reviews) | Blocking | Storage spec gap |

---

## Part 4: Next Research Target

The `c(sel, θ_GLS)` functional form is the blocking item. The FVS law is incomplete without it. The validation methodology should produce the fitted form; this is a candidate for the next research chunk.

The depleting-case strategy switch is also blocking — without it, the optimizer has no plan for filters that push relevant vectors away from the query. This is a plan-shape decision, not a parameter.

The natural next research target is **the `c(sel, θ_GLS)` functional form**: an empirical fitting study using the validation methodology from this response, with controlled predicates across the (sel, θ_GLS) space, producing a fitted function and a residual model for the lower confidence bound. The fitted form gates the FVS pushdown law's implementation; the depleting-case strategy switch gates the optimizer's plan enumeration for filters with `ρ_q < -threshold`.

---

## Part 5: Summary

This doc preserves the θ_GLS research response and adds a critical review. The response is the most concrete of the θ_GLS work: a closed-form Möbius-like transform, four computable implementations with cost analysis, a catalog design, a validation methodology, and the iFVS interaction. The critical review identifies the `c(sel, θ_GLS)` function as the blocking item — without it, the FVS law's right-hand side is unimplementable. The depleting-case strategy switch is also blocking — the optimizer needs a plan for filters that push relevant vectors away from the query. Several smaller items round out the gaps: the staleness interaction, the sampling fallback threshold, the region definition, the learned predictor specification, the `c_max` branching, the histogram storage, the real-world validation, and the API terminology.

The build can start on Phase 8 (FVS) with the θ_GLS implementation, gated on the `c(sel, θ_GLS)` function and the depleting-case strategy. The unified optimizer's joint recall distribution is now concrete enough to compose with θ_GLS, but the plan search algorithm and correlation correction remain blocking for the optimizer's plan enumeration. The next research target is the `c(sel, θ_GLS)` functional form, with the depleting-case strategy as a secondary deliverable.
