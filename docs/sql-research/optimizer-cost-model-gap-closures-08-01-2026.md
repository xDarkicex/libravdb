# Optimizer Cost Model: Gap Closures and Critical Review

**Date:** 2026-08-01
**Status:** Reference document — gap closures preserved with critical analysis
**Preread:**
- `optimizer-cost-model-research-response-08-01-2026.md` — the prior research response (the seven Q&A + worked examples + my first critical review)
- `foundations-of-unified-query-algebra-08-01-2026.md` — the algebra, equivalence laws, type system
- `mvcc-hmgi-late-interaction-research-review-08-01-2026.md` — the MVCC, Leiden, MaxSim review

---

## What This Document Is

This doc preserves the response to the seven gaps identified in the prior critical review of the optimizer cost model research. Each gap is closed with a derived formula or design decision. The doc also adds a second-pass critical review of the gap closures, identifies what's still missing, and proposes the next research target (the unified optimizer specification).

The gap closures are the *answer*. The critical review is the *verification* — what the closures get right, what they under-specify, and what limitation deserves a louder flag than the original response gave it.

---

## Part 1: The Seven Gap Closures

The response addressed each of the seven gaps in the prior critical review with the following corrections and additions.

### Gap 1: ApproxMaxSim Upper Bound Correction

**The error:** Centroid interaction is only a valid upper bound if, for every query token, the substituted centroid similarity is `≥` the true max similarity over that centroid's assigned residual vectors. PLAID enforces this via a per-centroid threshold `tcs` that discards centroids below a cutoff *before* computing the bound — if pruning drops a centroid whose true residual similarity exceeds `tcs`, the bound is no longer sound.

**Corrected formula:**

```
MaxSim_hat(q, d) = Σ_{i=1}^{L_q}  max_{c ∈ Centroids(d)} ( sim(q_i, c) + δ_c )
                   where δ_c = max_{v ∈ cell(c)} ||v - c||
```

The correction term `δ_c` (max residual norm within the centroid's Voronoi cell) must be added to preserve the upper-bound guarantee. Without it, `centroid_dot ≥ true_dot` is not generally true whenever a document token's true embedding sits farther from its centroid than the query embedding.

**Important caveat from the response:** PLAID's guarantee is empirical (calibrated `tcs` threshold), not a proven bound. Any implementation treating ECBB+ApproxMaxSim as a *provable* pruning composition needs to either adopt the `δ_c` correction or explicitly document that the combined pruning is probabilistic, not exact. This changes the composability claim from "safe pruning" to "recall-bounded approximate pruning."

### Gap 2: K_c as a Catalog Parameter

`K_c` must be a stored, versioned catalog statistic, not a formula-time constant:

```
catalog.maxsim_index_stats(
    corpus_size: N,
    centroid_count: K_c,             -- e.g. floor(sqrt(N * avg_doc_len))
    avg_doc_len: L_d_avg,
    centroid_recompute_epoch: ts,    -- staleness anchor, see Gap 3
    residual_bits: int               -- PLAID uses 1-2 bit compression
)
```

The optimizer reads `K_c` from this catalog entry at plan time rather than recomputing `sqrt(N × L_d)` inline. This matters because `K_c` is chosen once at index-build time and drifts out of the ideal `sqrt()` ratio as the corpus grows via inserts, which feeds directly into the staleness factor.

### Gap 3: Staleness Factor in the Cost Model

Define staleness `s ∈ [0, 1]` as the fraction of vectors not yet migrated to their Leiden-optimal page since the last stable partition:

```
s = writes_since_last_stable / total_vectors
    bounded by the lag counter from Q7
```

The traversal cost degrades by interpolating `N_intra / N_inter` toward the fully-random baseline as `s` grows:

```
C_traversal(s) = C_intra × N_intra × (1 - s) + C_inter × (N_inter + s × N_intra)     (6)
```

At `s=0` this reduces to the clean formula from before; at `s=1`, all intra-hops degrade to inter-cost, recovering the "plain HNSW, no partitioning" baseline (~7.89 μs/query from the earlier example). This gives the optimizer a continuous dial rather than a binary "stale/fresh" flag, and ECBB's bounding-sphere pruning should be discounted by `(1 - s)` in expected-value terms since stale pages have unreliable bounding spheres.

### Gap 4: Amortized Migration Cost Per Query

Per the migration cost `C_migration = O(D) + O(1) + O(N_page)`, amortize over queries using the migration rate `r` (migrations/sec) and query rate `λ` (queries/sec):

```
C_migration_amortized_per_query = (r / λ) × (D × t_AVX512_copy + t_CAS + N_page × t_ECBB_recompute)     (7)
```

This term gets **added** to `C_traversal(s)` in the total query cost — it's a background tax proportional to write pressure, not a per-query decision the optimizer can avoid. The key coupling: higher `r` reduces `s` (Gap 3) but increases this term, so the optimizer (or an autotuner) faces an explicit tradeoff surface between staleness cost and migration overhead, bounded by the worker's fractional-core CPU budget.

### Gap 5: Unifying Cost Model with Equivalence Laws

**The unification principle:** *equivalence laws are the generator, the cost model is the selector* — every law from the foundations doc that rewrites a plan node must have a corresponding cost-delta function, so plan enumeration becomes a search over the law-closure of the query, scored by the composed cost formulas (1)-(7).

| Equivalence law | Cost model hook |
|---|---|
| Predicate pushdown | Rescales `N` in formulas (3)-(5) by filter selectivity before vector cost is applied |
| FVS pre/post/in-filter choice | Chooses whether filter cost or `C_traversal` dominates the plan's critical path |
| ECBB page rejection | Discounts `N_inter` in (2)/(6) by the fraction of pages pruned, weighted by `(1 - s)` |
| ApproxMaxSim bound (corrected, Gap 1) | Substitutes (4)'s centroid stage for (3) only when the `δ_c`-corrected bound's expected recall exceeds a threshold |
| Layer co-location policy (Q4) | Changes `N_intra / N_inter` split ratio itself, i.e., feeds into (2) as a policy-dependent parameter, not a fixed 85/15 |
| Migration amortization (Gap 4) | Additive term (7) applied uniformly regardless of plan shape — it's a tax on the index, not on any specific plan |

This produces a genuine plan-search: laws generate the candidate set (join orders, filter placements, pruning-stage orderings), and the unified cost function `C_total = C_traversal(s) + C_migration_amortized + C_filter + C_MaxSim` ranks them.

### Gap 6: Matrix Subtyping for Inner Dimension Mismatch

**The gap:** What happens when `MATRIX(VECTOR(D_1), L_1)` is compared against `MATRIX(VECTOR(D_2), L_2)` with `D_1 ≠ D_2`? The subtyping relation from before only handled the variable-length row dimension (`L`), not the inner vector width.

**Corrected rule:** `MATRIX(VECTOR(D), L)` is *not* a subtype of `MATRIX(VECTOR(D'), L)` when `D ≠ D'` — inner dimension is **invariant**, not covariant, because `maxsim_score` requires exact dot-product alignment. This must be a **type error at bind time**, not a runtime failure: `maxsim_score(doc, query)` should only type-check if `doc.elem_type.dim == query.elem_type.dim` is a static equality, analogous to how Calcite rejects `ARRAY<INT>` unification with `ARRAY<VARCHAR>` without an explicit cast — except here there is no valid implicit cast between vector widths (no `CAST(VECTOR(128) AS VECTOR(256))` makes semantic sense), so the lattice needs an explicit **bottom type** for mismatched-dimension matrix comparisons rather than a coercion path.

### Gap 7: Layer Policy Threshold — Derived Formula

From Q4, define `f_0` = fraction of total HNSW traversal hops occurring at layer 0 (empirically dominant since layer 0 has by far the most nodes/edges). The hybrid policy's expected cost:

```
C_hybrid = f_0 × C_intra + (1 - f_0) × C_inter
```

versus per-layer's cost `C_per-layer = C_inter` at layer 0 (no co-location benefit there) and per-community's cost at upper layers `= C_inter` (hubs bridge communities regardless).

**Hybrid dominates per-layer when:**

```
f_0 > (C_inter - C_hybrid,upper) / (C_inter - C_intra)
```

Using the earlier numbers (`C_intra = 128ns`, `C_inter = 7616ns`), and given that in standard HNSW (M ≈ 16) the empirical layer-0 hop fraction is typically 85-95% of total traversal volume (since layer count `L ≈ log_M(N)` is small — for N=10M, M=16, `L ≈ 5.7`, and each upper layer has roughly `1/M` the nodes of the layer below), the hybrid policy dominates whenever `f_0` exceeds roughly **1-2%**, which is trivially satisfied in essentially every realistic configuration.

The earlier "80-90% threshold" was a loose empirical guess; the actual break-even point is far lower, meaning hybrid should almost always be preferred over per-layer. The real decision boundary is between **hybrid and per-community**, governed by how much upper-layer hop cost the hybrid policy avoids paying by *not* co-locating those sparse layers.

---

## Part 2: Worked Numerical Examples (Preserved)

The gap closures reference and refine the worked examples from the prior research response. Two key numbers to preserve:

| Number | Value | Source | Used in |
|---|---|---|---|
| Leiden-optimized HNSW traversal | 2.91 μs/query | Prior response, 85/15 intra/inter split | Phase 9 (ECQO) baseline |
| Plain HNSW (no locality) | 7.89 μs/query | Prior response, fully random | Phase 9 fallback when `s=1` |
| Per-hop intra/inter ratio | ~60x | Hennessy & Patterson, PLAID measurements | Layer policy threshold |
| Speedup at clean partitioning | ~2.7x | Prior response | Leiden quality metric |
| PLAID at 140M passages | 278.6 ms | Prior response, 1024 survivors | MaxSim phase baseline |
| PLAID full brute-force | ~31,000 sec | Prior response | MaxSim phase upper bound |
| Hybrid policy break-even | `f_0 > 1-2%` | Gap closure 7 | Layer policy default |

---

## Part 3: Critical Review of the Gap Closures

The seven gap closures are principled. Each one addresses the original gap with a derived formula or a design decision. But the closures under-specify several things, and one limitation deserves a louder flag than the original response gave it.

### What's Strong

1. The **δ_c correction** in Gap 1 is the standard approach for sound upper bounds with residual quantization. The Voronoi cell interpretation is correct: each centroid's correction is the max residual norm in its cell.
2. The **K_c as catalog parameter** in Gap 2 is the right design. Storing `centroid_count`, `avg_doc_len`, `centroid_recompute_epoch`, `residual_bits` in `maxsim_index_stats` decouples the cost formula from index-build choices and exposes them to the optimizer.
3. The **staleness factor as a continuous dial** in Gap 3 is the correct framing. The linear interpolation in formula (6) degrades gracefully from clean to fully-random as `s → 1`.
4. The **amortized migration cost** in Gap 4 is correct: `r/λ × cost_per_migration`, added uniformly to every plan.
5. The **"equivalence laws as generator, cost model as selector"** framing in Gap 5 is the keystone. The table mapping each law to a cost hook is exactly the bridge between the algebra and the cost model.
6. The **matrix subtyping correction** in Gap 6 is right: invariant inner dimension, type error at bind time, no implicit cast. The "no valid CAST between VECTOR widths" point is sharp.
7. The **layer policy break-even at 1-2%** in Gap 7 is much sharper than the empirical 80-90%. The real decision boundary is correctly identified as hybrid-vs-per-community, not hybrid-vs-per-layer.

### What Still Needs to Land

1. **The δ_c correction has a storage cost.** Each centroid in each document needs its own δ_c (max residual norm within that centroid's Voronoi cell). For `K_c = 65536` centroids and `N = 140M` documents, that's `140M × 65536 × 4 bytes = 36 TB` if stored per-document-per-centroid. The realistic alternative: pre-compute a single δ_c per centroid (max residual norm across the entire corpus) and use it for all documents. This loses some tightness but reduces storage to `65536 × 4 = 256 KB`. The cost model should specify which variant.

2. **K_c is per-document or global, and the cost formula needs to know which.** PLAID's actual implementation uses a global codebook of centroids shared across all documents. The cost `K_c × L_q × D` assumes this global codebook. If the codebook is per-document, the cost is `N × K_c_per_doc × L_q × D` — vastly more. The catalog entry in Gap 2 should specify `centroid_scope: 'global' | 'per_document'` and the cost formula branches on it.

3. **The linear staleness interpolation in formula (6) is an approximation.** The actual degradation is likely super-linear once a critical mass of vectors are misplaced (HNSW graph connectivity degrades non-uniformly). The cost model should treat the linear formula as a starting point and specify a calibration mechanism: measure actual recall at synthetic `s` values, fit a curve, update the interpolation. The PLAID paper doesn't give this data, so it's an empirical question for the implementation.

4. **The amortized migration cost formula (7) assumes `r` and `λ` are known.** In practice, both are stochastic (Poisson-distributed query and write arrivals). The cost model should use expected values or, better, a distribution. The expected migration cost `E[r/λ × cost_per_migration]` is a first-order approximation; the variance matters for tail-latency planning.

5. **The equivalence law table in Gap 5 is incomplete.** The foundations doc has more laws than the six listed:
   - **Selection idempotence** (with the similarity-monoid caveat from the foundations doc — `σ_p σ_p = σ_p` is **unsound in the similarity monoid**)
   - **Projection pushdown**
   - **Join commutativity** (`R ⋈ S = S ⋈ R` for inner joins)
   - **Aggregation pushdown**
   - **DISTINCT elimination**
   - **The θ_GLS-based FVS pushdown** (the GLS metric is still undefined in the literature)
   - **The K-NN join reorder** (`topk(k, R ⋈_E S, d) ≈ R ⋈_E topk(k × E[fanout], S, d)`)
   
   The table needs cost-delta functions for all ~12 equivalence laws in the catalog, not just six.

6. **The matrix subtyping correction is right, but the type lattice needs an explicit unification failure.** When `MATRIX(VECTOR(128), L)` meets `MATRIX(VECTOR(256), L')`, the type checker has to report a specific error ("inner VECTOR dimension mismatch: 128 vs 256") and suggest a fix (none exists in this case, since the dimensions are semantically incompatible). The lattice needs a `MISMATCH` or `TYPE_ERROR` node, or an explicit unification failure exception type.

7. **The "hybrid almost always wins" conclusion depends on workload.** The 1-2% break-even for hybrid-vs-per-layer is correct for typical HNSW parameters. But the *real* decision — hybrid vs per-community — depends on upper-layer hop frequency. If the workload is dominated by long-range queries (high ef, deep traversal), per-community may win because upper-layer hubs span communities. The cost model should specify: when `f_upper > τ_upper` (some threshold), prefer per-community; otherwise prefer hybrid. The threshold depends on Leiden quality (modularity) and the HNSW degree distribution.

### One Limitation That Deserves a Louder Flag

8. **PLAID's guarantee is empirical, not proven.** Gap 1 in the closures correctly notes this: "PLAID's actual guarantee is empirical (calibrated `tcs` threshold, not a proven bound)." This is a serious limitation. If the cost model assumes provable bounds but the underlying pruning is only empirically sound, the optimizer's reasoning is unsound at the *guarantee* level. The cost model should explicitly model this as:

   ```
   recall = empirical_recall(PLAID_params) ± empirical_uncertainty
   ```

   The uncertainty enters the cost via the accuracy target in `FETCH APPROX FIRST k ROWS ONLY WITH TARGET ACCURACY n PERCENT`. The optimizer needs to compare `expected_recall ≥ n` against the **lower bound** of the empirical distribution, not the mean. This makes the cost model stochastic, not deterministic.

   The current gap closures don't propagate this uncertainty through the cost model. If the recall distribution has a long left tail, the safety margin (the difference between expected recall and the target) has to be larger than the optimizer's current formulation would suggest.

---

## Part 4: What This Enables for the Build

- **Phase 8 (FVS):** The cost model is now concrete enough to implement the FVS strategy selection. The Leiden partitioning provides a new dimension for the optimizer to reason about. **Gated on the staleness factor (Gap 3) and the amortized migration cost (Gap 4) being integrated into the cost function.**
- **Phase 9 (ECQO):** The Leiden partitioning is a cost-model input. The ECBB pruning is a new optimization rule. Both are concrete enough to implement. **Gated on the dimension-aware page layout and the ECBB shape decision (from the prior review).**
- **New multi-vector phase:** The MaxSim cost model is now concrete enough to plan, with the `δ_c` correction and the catalog-based `K_c` parameter. **Gated on the per-document vs global codebook specification (point 2 above) and the empirical-recall uncertainty (point 8 above).**
- **Type system extension:** The matrix type extension is concrete enough to transcribe, with the corrected subtyping rule and the unification failure type. **Gated on adding the `MISMATCH` node to the lattice (point 6 above).**

---

## Part 5: What's Still Missing

| Item | Severity | Type |
|---|---|---|
| Complete cost-delta functions for all 12+ equivalence laws (point 5) | **Blocking** | Architectural gap |
| Plan search algorithm specification (Cascades vs Volcano vs hybrid) | High | Algorithm choice |
| Cost of optimization (planning time) in the model | High | Cost model gap |
| Empirical-vs-probabilistic accuracy in the cost model (point 8) | **Blocking** | Stochastic extension |
| Per-document vs global codebook specification (point 2) | High | Catalog design |
| δ_c storage variant (per-doc vs global, point 1) | Medium | Storage tradeoff |
| Non-linear staleness calibration (point 3) | Medium | Empirical calibration |
| Stochastic r and λ in the amortized migration cost (point 4) | Medium | Distribution modeling |
| `f_upper` threshold for hybrid-vs-per-community (point 7) | Medium | Empirical question |
| θ_GLS metric computation (still undefined in literature) | **Blocking** | Research gap |
| Layer policy threshold derivation for hybrid-vs-per-community | Medium | Design gap |
| HNSW layer structure in the MVCC slab (from prior reviews) | Blocking | Specification gap |
| xmin/xmax commit protocol (from prior reviews) | Blocking | Specification gap |

---

## Part 6: Next Research Target

The biggest unaddressed gaps are the **complete equivalence law table** (point 5) and the **empirical-recall uncertainty** in the cost model (point 8). The next research target is the **unified optimizer specification**, which should produce a document that specifies:

1. **The complete cost-delta function table** for all 12+ equivalence laws from the foundations doc, not just the six listed. Each law gets a cost-delta function that the optimizer composes into the total plan cost.

2. **The plan search algorithm** (Cascades-style memoization vs Volcano-style top-down vs hybrid) with a specification of when each is appropriate. The plan space is large; the search algorithm itself has a cost (planning time), and the optimizer should pick the search algorithm based on query complexity.

3. **The cost of optimization** in the model. Planning time is a real cost — a 10ms plan generation on a 1ms query is a 10x overhead. The cost model should include the optimization cost and the optimizer should pick plans that are "good enough" without exhaustive search.

4. **The empirical-vs-probabilistic accuracy distinction.** The cost model becomes stochastic, not deterministic. The optimizer compares `expected_recall ≥ n` against the **lower bound** of the empirical recall distribution. This requires a model of the recall distribution per plan, derived from the equivalence law catalog and the index parameters.

5. **Worked examples for the representative hybrid queries** from the foundations doc. Apply the unified optimizer to the worked example (compromised hosts connecting to servers via HTTPS with 90% accuracy target) and to the MaxSim hybrid query. Show the plan enumeration, the cost ranking, and the chosen plan. End-to-end trace.

6. **Validation against published benchmarks.** Compare the cost model's predictions against:
   - PLAID's published 92-352 ms latency at 140M passages
   - HNSW-only benchmarks on ANN-Benchmarks datasets
   - LDBC SNB for graph + relational workloads
   - Col-Bandit's published numbers for MaxSim with bandit pruning
   
   The cost model should be within 2x of published numbers; if not, the model needs calibration.

7. **The θ_GLS computation** (still undefined in the literature). This is the blocking research item for the FVS equivalence law. Without it, the FVS pushdown law has no computable cost-delta.

This is the bridge research that ties the algebra (foundations doc), the storage (CoW Slab, Leiden, MaxSim), and the cost model (the two response docs) into a single optimizer framework. Without it, each piece is correct in isolation but the optimizer can't reason across them. With it, the build can start Phase 8 (FVS strategy selection) with a cost model that knows about Leiden partitioning, multi-vector queries, and the algebraic equivalence laws.

---

## Part 7: Summary

This doc preserves the seven gap closures from the prior critical review and adds a second-pass critical review. The closures are principled: the `δ_c` correction is correct, the K_c catalog design is right, the staleness factor is a continuous dial, the amortized migration cost is sound, the unification principle ("laws as generator, cost model as selector") is the keystone, the matrix subtyping correction is right, and the layer policy break-even at 1-2% is much sharper than the empirical 80-90%.

The second-pass review identifies eight things that still need to land: the δ_c storage cost, the per-document-vs-global codebook specification, the non-linear staleness calibration, the stochastic `r` and `λ`, the complete equivalence law table (12+ laws, not 6), the unification failure type in the lattice, the `f_upper` threshold for hybrid-vs-per-community, and the empirical-recall uncertainty in the cost model. The empirical-recall limitation deserves the loudest flag — it changes the cost model from deterministic to stochastic and propagates through every FVS plan choice.

The build can start on Phase 8 (FVS) and Phase 9 (ECQO) with the cost model from these gap closures, gated on the staleness factor and the amortized migration cost. The multi-vector phase can start with the MaxSim cost model, gated on the per-doc-vs-global codebook specification and the empirical-recall uncertainty. The unified optimizer specification is the next research target, with the complete equivalence law table and the stochastic cost model as the central deliverables.
