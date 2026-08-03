# Optimizer Cost Model: Second-Pass Gap Closures and Critical Review

**Date:** 2026-08-01
**Status:** Reference document — second-pass gap closures preserved with critical analysis
**Preread:**
- `optimizer-cost-model-research-response-08-01-2026.md` — the prior research response
- `optimizer-cost-model-gap-closures-08-01-2026.md` — the first round of gap closures
- `foundations-of-unified-query-algebra-08-01-2026.md` — the algebra, equivalence laws, type system
- `mvcc-hmgi-late-interaction-research-review-08-01-2026.md` — the MVCC, Leiden, MaxSim review

---

## What This Document Is

This doc preserves the second round of gap closures, addressing the eight points raised in the prior critical review of the optimizer cost model gap closures. The closures include the `δ_c` storage tradeoff, the centroid scope flag, non-linear staleness calibration, stochastic migration cost, the complete 13-law equivalence table, explicit type unification failure, the modularity-aware layer policy threshold, and the lower confidence bound on recall.

The doc also adds a third-pass critical review. The hybrid-vs-per-community threshold formula has a conceptual error that needs to land before the cost model is correct, and several other items are flagged as gaps.

The gap closures are the *answer*. The critical review is the *verification*.

---

## Part 1: The Eight Gap Closures

The response addressed each of the eight points from the prior critical review with the following corrections.

### Gap 1: δ_c Storage — Global-Per-Centroid Variant

**Specification:** Store one δ_c per centroid globally, not per (document, centroid) pair:

```
δ_c = max_{v ∈ corpus, assign(v)=c} ||v - c||
storage = K_c × 4 bytes
```

At `K_c = 65536` this is 256 KB — trivially cacheable, resident in L2/L3 during query execution, versus 36 TB for the per-document variant. The cost model must document the tightness tradeoff explicitly: the global `δ_c` is the *max over the entire corpus*, so it's a looser (more conservative) bound than a per-document `δ_c` would be, meaning centroid-stage recall is higher (fewer false negatives) but the pruning is less aggressive (more survivors pass through to full evaluation) — this is a correctness-preserving choice that trades some of formula (4)'s speedup for feasible storage. Formula (4) should be annotated: `δ_c` is corpus-global unless the catalog explicitly states otherwise.

### Gap 2: Centroid Scope in the Catalog

```
catalog.maxsim_index_stats(
    centroid_scope: 'global' | 'per_document',
    centroid_count: K_c,          -- meaning depends on scope
    ...
)
```

Cost formula branches:

```
C_centroid = {
    K_c × L_q × D                  if scope = 'global'
    N × K_c_doc × L_q × D          if scope = 'per_document'
}
```

PLAID uses global scope exclusively, so the earlier worked example (`K_c = 42426`, 52 ms centroid stage) is only valid under `centroid_scope = 'global'`. If an implementation deviates to per-document centroids, the optimizer must detect the catalog flag and reject the PLAID-style plan shape entirely, since the per-document cost term is `N`-scaled and destroys the entire pruning benefit — it would be worse than brute force once you add the centroid-scan overhead on top.

### Gap 3: Non-Linear Staleness Calibration

Formula (6)'s linear interpolation is a placeholder. Replace with an explicit calibration hook:

```
C_traversal(s) = C_intra × N_intra × (1 - g(s)) + C_inter × (N_inter + g(s) × N_intra)
```

where `g(s)` is a **fitted, empirically-calibrated** monotonic function (`g(0) = 0`, `g(1) = 1`) rather than assumed linear (`g(s) = s`). The calibration procedure: inject synthetic write load at controlled rates, measure actual recall@k and latency at sampled `s` values, fit `g` (a sigmoid or power-law is the natural candidate since graph connectivity degradation is typically threshold-like — small `s` has negligible effect until a critical mass of misplaced vectors breaks enough shortest paths). This is explicitly an empirical, implementation-specific curve since neither PLAID nor the Leiden paper measures this quantity — the catalog should store the fitted `g` parameters per index instance and refit periodically as workload characteristics shift.

### Gap 4: Stochastic Migration Cost

Model `r` and `λ` as independent Poisson processes. The amortized cost becomes a ratio of two Poisson-derived random variables; using expected values only captures the first moment:

```
E[C_migration_amortized] ≈ E[r] / E[λ] × C_per_migration
```

But for tail-latency planning, the relevant quantity is the queueing delay distribution, not just the mean. Treating the migration worker as an M/M/1-style queue, the **Pollaczek–Khinchine** result is directly applicable here: waiting time grows with the *variance* of service time, not just the mean utilization `ρ = r/λ`. Two systems with identical mean migration rate can have very different tail latency if migration batch sizes or write bursts are variable. Practical consequence: the cost model should track `Var(r)`, `Var(λ)` alongside the means, and expose a p99 migration-lag estimate (not just expected lag) to the staleness detector in #3 — bursty write traffic will spike `s` transiently even when average migration rate looks adequate.

### Gap 5: Complete Equivalence Law Table

Extending the earlier six-law table to the full 13-law catalog:

| Equivalence law | Cost-delta hook | Caveat |
|---|---|---|
| Predicate pushdown | Rescales `N` before vector cost | — |
| FVS pre/post/in-filter | Chooses dominant cost term | — |
| ECBB page rejection | Discounts `N_inter`, weighted by `(1 - s)` | Recall risk under staleness |
| ApproxMaxSim bound | Substitutes (4) for (3) | Only sound with `δ_c` correction (Gap 1) |
| Layer co-location policy | Sets intra/inter split ratio | Workload-dependent (see #7) |
| Migration amortization | Additive tax, now stochastic (Gap 4) | Applies uniformly across plans |
| Selection idempotence | `σ_p σ_p → σ_p` collapses redundant filter nodes | **Unsound** in the similarity monoid — repeated similarity-threshold selection is not idempotent since floating-point/approximate scoring can vary run to run; must be disabled for similarity predicates specifically |
| Projection pushdown | Reduces materialized column width before join/sort | Straightforward, no vector interaction |
| Join commutativity | `R ⋈ S = S ⋈ R` reorders build/probe side | Cost delta = swap of hash-build cost estimate |
| Aggregation pushdown | Push GROUP BY below join when grouping columns are foreign-key-preserved | No vector interaction |
| DISTINCT elimination | Remove redundant DISTINCT when uniqueness is already guaranteed by a key constraint | No vector interaction |
| θ_GLS-based FVS pushdown | Push filter below a graph-local-similarity threshold node | **Metric still undefined in the literature** — cannot assign a cost delta until `θ_GLS` has a formal definition; flag as an open research item, not implementable yet |
| k-NN join reorder | `topk(k, R ⋈_E S, d) ≈ R ⋈_E topk(k × E[fanout], S, d)` | Approximate equivalence — introduces its own recall risk proportional to fanout estimation error, compounding with staleness (Gap 3) and PLAID's empirical bound (Gap 8) |

Two of these — θ_GLS and selection idempotence-under-similarity — are not safe to treat as unconditional rewrites; they need to be flagged in the catalog as **conditional or unsound-by-default** laws requiring explicit guard conditions.

### Gap 6: Explicit Type Unification Failure

Add a bottom-type / error node to the lattice rather than allowing silent coercion attempts:

```
unify(MATRIX(VECTOR(D1), L), MATRIX(VECTOR(D2), L')) =
    if D1 == D2: MATRIX(VECTOR(D1), unify_dim(L, L'))
    else: raise TypeUnificationError(
        "inner VECTOR dimension mismatch: {D1} vs {D2}",
        suggestion = None  -- no valid cast exists
    )
```

This mirrors how Calcite raises a validation error on incompatible `ARRAY` element types rather than silently null-coercing, but goes further by explicitly encoding "no suggested fix" — unlike a `VARCHAR(10)` vs `VARCHAR(20)` mismatch (which Calcite can widen), a vector-dimension mismatch reflects genuinely incompatible embedding spaces, so the error message should say so rather than implying a coercion path exists.

### Gap 7: Hybrid vs Per-Community Threshold

Define `f_upper` as the fraction of total traversal hops occurring above the bottom 2 layers, and `Q` as the Leiden partition's modularity score. The decision rule:

```
prefer per-community if f_upper > τ_upper,
    τ_upper = (C_inter - C_hybrid,upper) / ((C_inter - C_intra) × Q)
```

The `Q` term in the denominator captures that higher-modularity partitions (Traag et al.'s well-connected-communities guarantee) make per-community upper-layer traversal *less* costly than in a low-modularity partition, since even hub-crossing hops are more likely to land in a "nearby" community. So `τ_upper` rises with `Q` — better partitions tolerate more long-range-hop-heavy workloads before per-community's community-locality benefit outweighs hybrid's layer-0 efficiency. This makes the policy choice workload- *and* partition-quality-dependent, not a fixed constant as the earlier "1-2% threshold" implied — that number was only the hybrid-vs-per-layer boundary, not the (harder) hybrid-vs-per-community boundary.

### Gap 8: Modeling PLAID's Empirical (Not Proven) Guarantee

The cost/recall model must be explicitly stochastic:

```
recall(PLAID_params) = μ_hat_recall ± σ_hat_recall,
    measured empirically per (K_c, n_docs, t_cs)
```

For `FETCH APPROX FIRST k ROWS ONLY WITH TARGET ACCURACY n PERCENT`, the optimizer must satisfy:

```
μ_hat_recall - z · σ_hat_recall ≥ n / 100
```

using a **lower confidence bound**, not the mean — this is exactly analogous to Oracle's `DBMS_VECTOR.INDEX_ACCURACY_QUERY` mechanism, which empirically reports achievable accuracy for a given target rather than assuming a closed-form guarantee. Any plan whose empirical recall distribution's lower bound falls short of the target accuracy must be rejected or escalated to a more conservative plan (larger `n_docs`, smaller `t_cs`, or fallback to exact brute-force MaxSim). This is the correct way to encode "PLAID's guarantee is empirical, not proven" directly into the optimizer's decision procedure, rather than treating recall as a deterministic property of the plan shape.

---

## Part 2: Worked Numerical Examples (Preserved)

The gap closures reference and refine the worked examples from the prior research response. Key numbers to preserve:

| Number | Value | Source | Used in |
|---|---|---|---|
| Leiden-optimized HNSW traversal | 2.91 μs/query | Prior response, 85/15 intra/inter split | Phase 9 (ECQO) baseline |
| Plain HNSW (no locality) | 7.89 μs/query | Prior response, fully random | Phase 9 fallback when `s=1` |
| Per-hop intra/inter ratio | ~60x | Hennessy & Patterson, PLAID measurements | Layer policy threshold |
| Speedup at clean partitioning | ~2.7x | Prior response | Leiden quality metric |
| PLAID at 140M passages | 278.6 ms | Prior response, 1024 survivors | MaxSim phase baseline |
| PLAID full brute-force | ~31,000 sec | Prior response | MaxSim phase upper bound |
| Hybrid-vs-per-layer break-even | `f_0 > 1-2%` | First-pass gap closure | Layer policy default (per-layer) |
| δ_c global storage | 256 KB | Second-pass closure | Catalog design |
| δ_c per-document storage | 36 TB | Infeasible | Rejected |
| Pollaczek–Khinchine | M/G/1 with general service time | Second-pass closure | Migration cost tail latency |

---

## Part 3: Critical Review of the Second-Pass Gap Closures

The eight gap closures are substantive. Several are correctness fixes, not just refinements. But the hybrid-vs-per-community threshold formula has a conceptual error that needs to land before the cost model is correct, and there are a few other items to flag.

### What's Strong

1. The **δ_c storage decision** in Gap 1 is the right tradeoff. The 256 KB global-per-centroid variant is correctness-preserving; the 36 TB per-document variant is infeasible. The annotation to formula (4) is the right discipline.
2. The **centroid scope flag** in Gap 2 prevents silent cost miscalculations. PLAID's global scope is the only tractable option at scale.
3. The **`g(s)` calibration** in Gap 3 is the correct framing. Sigmoid or power-law are the natural candidates. The threshold-like degradation insight (small `s` has negligible effect until a critical mass) is right.
4. The **Pollaczek–Khinchine insight** in Gap 4 is correct: variance matters for p99, not just mean. The M/G/1 framing with general service time is the right model.
5. The **complete 13-law table** in Gap 5 is the right level of detail. Three critical flags are correctly identified:
   - Selection idempotence is **unsound in the similarity monoid** (the floating-point/approximate scoring variance is sharper than the prior foundations doc said)
   - θ_GLS is **still undefined in the literature**
   - k-NN join reorder has **compounding recall risk** with PLAID's empirical bound and staleness
6. The **`TypeUnificationError` with `suggestion=None`** in Gap 6 is sharp. "No valid cast exists" is the right framing.
7. The **`Q` (modularity) insight** in Gap 7 — better partitions tolerate more long-range hops — is a real contribution to the threshold formula. The PMC Leiden citation is relevant.
8. The **lower confidence bound `μ - z·σ ≥ n/100`** in Gap 8 is the correct stochastic formulation. The Oracle `DBMS_VECTOR.INDEX_ACCURACY_QUERY` analogy is apt.

### What Needs to Land

1. **The hybrid-vs-per-community threshold formula in Gap 7 has a conceptual error.** The numerator is `C_inter - C_hybrid,upper`. Under the hybrid policy, upper layers are *not* co-located, so `C_hybrid,upper = C_inter` and the numerator is zero. That makes `τ_upper = 0` for any `Q > 0`, meaning per-community always wins, which is wrong.

   The conceptual issue: under the *per-community* policy, upper-layer hubs are not co-located either — hubs by definition span communities. The co-location benefit is *only* at layer 0-1, where most nodes live. So per-community and hybrid have the *same* upper-layer cost (`C_inter`). The cost difference is only at layer 0-1, where per-community gives `C_intra` and hybrid also gives `C_intra` (both co-locate the dense layer). At first glance they're equal.

   The actual difference: per-community *may* co-locate some upper-layer hubs if those hubs happen to belong to Leiden communities with their frequent neighbors. The Leiden quality `Q` captures this: a high-Q partition has well-connected communities, so upper-layer hubs are more likely to have intra-community neighbors. But this is a *probabilistic* benefit, not a deterministic one. The threshold formula needs to express this as: hybrid wins unless the upper-layer hop is *likely* to land in the same Leiden community as the source. The `Q` term should be in a probability, not a cost delta.

2. **The Pollaczek–Khinchine formula needs the second moment, not just variance.** For M/G/1, the mean waiting time is `E[W] = λ × E[S²] / (2 × (1 - ρ))` where `E[S²] = Var(S) + (E[S])²`. The response says "variance of service time" but the correct quantity is the second moment. This is a notational fix; the implementation needs the second moment, not just the variance.

3. **The `z` value in the lower confidence bound is unspecified.** The response writes `μ - z·σ ≥ n/100` but doesn't specify `z`. This is a configuration parameter (1.645 for 90% confidence, 1.96 for 95%, 2.576 for 99%). The cost model should expose it as `accuracy_confidence_level: float ∈ (0, 1)` and compute `z` from the inverse normal CDF.

4. **The `g(s)` calibration procedure is high-level, not actionable.** The response describes "inject synthetic write load at controlled rates, measure actual recall@k and latency at sampled s values, fit g." The implementation needs:
   - The synthetic write load generator (what's representative? uniform random? skewed to specific Leiden communities?)
   - The recall@k measurement (against what ground truth? pre-staleness query results?)
   - The fitting procedure (sigmoid: `g(s) = 1 / (1 + exp(-k(s - s_0)))`? what are `k` and `s_0`? power-law: `g(s) = s^α`? what's `α`?)
   - The refit cadence (after N writes? after a time interval? after a Leiden re-partition?)

5. **The complete provenance of recall error sources needs a joint distribution, not three independent ones.** The k-NN join reorder's fanout approximation, PLAID's empirical pruning uncertainty, and staleness-induced HNSW degradation all *compound*. The cost model needs to compute the *joint* recall distribution, not track three per-source recalls and assume independence. If they're correlated (which they likely are — staleness degrades both HNSW connectivity and PLAID's centroid validity), the joint distribution has a wider left tail than the product of marginals.

6. **The k-NN join reorder's fanout estimation is single-valued but the actual fanout is power-law.** Real graph fanout is highly non-uniform (most nodes have low degree, hubs have very high degree). The `E[fanout]` is the mean; the variance is large. The k-NN join reorder law is only sound when fanout is uniform or the top-k ranking is preserved through the join. With power-law fanout, the law is unsound for the high-fanout tail. The optimizer needs to detect when a query touches high-fanout nodes and either reject the reorder or apply a fanout-aware correction.

7. **The "selection idempotence in the similarity monoid" caveat is sharper than the prior foundations doc and is worth pulling back into the foundations.** The response notes that "floating-point/approximate scoring can vary run to run" — this is HNSW's non-determinism (different runs can produce different results due to insertion order, neighbor selection ties, etc.). The same query twice can produce different top-k results, so `σ_p σ_p` is not idempotent under non-deterministic scoring. The foundations doc should be updated to note this.

8. **The cost model should specify what to do when the lower confidence bound falls below the target accuracy.** The response says "Any plan whose empirical recall distribution's lower bound falls short of the target accuracy must be rejected or escalated to a more conservative plan." This is right, but the cost model needs to specify *which* conservative plan:
   - Larger `n_docs` (more survivors from PLAID)
   - Smaller `t_cs` (less aggressive centroid pruning)
   - Fallback to exact brute-force MaxSim
   - Combination of the above

   The optimizer's plan choice is to pick the conservative escalation that satisfies the accuracy target at minimum cost. This is a standard cost-based optimization, but the cost model needs to know the cost-delta of each escalation option.

### One Limitation That Deserves Continued Attention

9. **PLAID's guarantee is empirical, not proven, and the cost model needs to track this across the entire plan space.** Gap 8 in the closures addresses this for the `FETCH APPROX` accuracy target. But the cost model should also propagate the empirical-recall uncertainty into:
   - The FVS pre/post/in-filter strategy choice (each strategy has a different recall distribution)
   - The k-NN join reorder's fanout approximation (compounds with PLAID's uncertainty)
   - The Leiden partitioning staleness (degrades HNSW connectivity, which degrades PLAID's centroid validity)

   The right framing: the cost model tracks a per-plan recall distribution, and the optimizer compares the lower confidence bound against the target. The plan with the lowest cost *whose lower confidence bound meets the target* wins.

---

## Part 4: What This Enables for the Build

- **Phase 8 (FVS):** The cost model is now concrete enough to implement the FVS strategy selection. The Leiden partitioning provides a new dimension for the optimizer to reason about. **Gated on the staleness factor (Gap 3), the amortized migration cost (Gap 4), and the lower confidence bound (Gap 8) being integrated into the cost function.**
- **Phase 9 (ECQO):** The Leiden partitioning is a cost-model input. The ECBB pruning is a new optimization rule. Both are concrete enough to implement. **Gated on the dimension-aware page layout and the ECBB shape decision (from prior reviews).**
- **New multi-vector phase:** The MaxSim cost model is now concrete enough to plan, with the `δ_c` correction, the catalog-based `K_c` parameter, and the centroid scope flag. **Gated on the joint recall distribution (point 5 above) and the conservative escalation options (point 8 above).**
- **Type system extension:** The matrix type extension is concrete enough to transcribe, with the corrected subtyping rule and the `TypeUnificationError` exception. **Gated on adding the `MISMATCH` node to the lattice (from prior review).**
- **Complete equivalence law table:** The 13-law table is concrete enough to transcribe into the optimizer's rule set. **Gated on the `θ_GLS` computation (still undefined in literature) and the power-law fanout detection (point 6 above).**

---

## Part 5: What's Still Missing

| Item | Severity | Type |
|---|---|---|
| Hybrid-vs-per-community threshold formula (point 1) | **Blocking** | Conceptual error |
| Joint recall distribution across error sources (point 5) | **Blocking** | Stochastic model gap |
| `g(s)` calibration procedure specifics (point 4) | High | Implementation gap |
| P-K formula second moment, not variance (point 2) | Medium | Notation fix |
| `z` value specification (point 3) | Medium | Configuration gap |
| Power-law fanout detection in k-NN join (point 6) | High | Algorithm gap |
| HNSW non-determinism in foundations doc (point 7) | Low | Doc update |
| Conservative escalation options in cost model (point 8) | Medium | Plan choice gap |
| Recall uncertainty propagation through FVS, k-NN, Leiden (point 9) | High | Design gap |
| θ_GLS metric computation | **Blocking** | Research gap |
| HNSW layer structure in the MVCC slab (from prior reviews) | Blocking | Specification gap |
| xmin/xmax commit protocol (from prior reviews) | Blocking | Specification gap |

---

## Part 6: Next Research Target

The biggest unaddressed gaps are the **corrected hybrid-vs-per-community threshold formula** (point 1), the **joint recall distribution** (point 5), and the **`θ_GLS` computation** (still undefined in literature). The next research target is the **unified optimizer specification** that ties everything together.

The unified optimizer spec should produce a document that specifies:

1. **The hybrid-vs-per-community threshold formula, corrected** for the conceptual error in point 1. The right framing is probabilistic: hybrid wins unless the upper-layer hop is likely to land in the same Leiden community as the source. The `Q` term should be in a probability, not a cost delta.

2. **The joint recall distribution** across the three error sources (PLAID empirical, k-NN join reorder, staleness), with a closed-form expression for the lower confidence bound. If the sources are correlated (likely), the joint distribution has a wider left tail than the product of marginals.

3. **The `g(s)` calibration procedure** with specific fit functions (sigmoid or power-law), specific load generators, specific ground-truth sources, and specific refit cadence.

4. **The P-K formula** with the second moment `E[S²] = Var(S) + (E[S])²`, not just the variance.

5. **The `z` value configuration** as `accuracy_confidence_level: float ∈ (0, 1)` in the cost model, with `z` computed from the inverse normal CDF.

6. **The power-law fanout detection algorithm** for the k-NN join reorder, with a threshold on `Var(fanout) / E[fanout]` that triggers either rejection of the reorder or a fanout-aware correction.

7. **The conservative escalation options** with their cost-delta functions:
   - Larger `n_docs` (PLAID cost increase)
   - Smaller `t_cs` (PLAID recall increase)
   - Fallback to exact brute-force MaxSim (full matrix cost)
   - Combination of the above

8. **The recall uncertainty propagation** through FVS pre/post/in-filter strategy choice, k-NN join reorder, and Leiden partitioning staleness. The right framing: per-plan recall distribution, lower confidence bound against the target.

9. **End-to-end worked examples** for the representative hybrid queries from the foundations doc, applying the unified cost function. Show plan enumeration, cost ranking, recall distributions, and the chosen plan.

10. **Validation against published benchmarks** to within 2x of published numbers (PLAID, ANN-Benchmarks, LDBC SNB, Col-Bandit).

11. **The `θ_GLS` computation.** This is the blocking research item for the FVS equivalence law. Without it, the FVS pushdown law has no computable cost-delta. The research target should include a candidate implementation: a learned predictor over the predicate and the vector distribution, a sampling-based estimator, or an analytical formula based on the predicate's spatial concentration.

This is the bridge research that ties the algebra (foundations doc), the storage (CoW Slab, Leiden, MaxSim), and the cost model (the three response docs) into a single optimizer framework. Without it, each piece is correct in isolation but the optimizer can't reason across them. With it, the build can start Phase 8 (FVS strategy selection) with a cost model that knows about Leiden partitioning, multi-vector queries, and the algebraic equivalence laws.

---

## Part 7: Summary

This doc preserves the eight second-pass gap closures and adds a third-pass critical review. The closures are substantive: the `δ_c` global-per-centroid decision (256 KB vs 36 TB), the centroid scope flag, the `g(s)` non-linear calibration, the Pollaczek–Khinchine extension for tail latency, the complete 13-law equivalence table with conditional/unsound flags, the `TypeUnificationError` with `suggestion=None`, the modularity-aware layer policy threshold, and the lower confidence bound on recall.

The third-pass review identifies the **hybrid-vs-per-community threshold formula conceptual error** (the `C_hybrid,upper` term makes the numerator zero) and the **missing joint recall distribution** across the three error sources. The corrected framing is probabilistic for the threshold and a closed-form lower confidence bound for the recall. Several smaller items — the P-K second moment, the `z` value, the `g(s)` fit procedure, the power-law fanout detection, the HNSW non-determinism, and the conservative escalation options — round out the gaps.

The build can start on Phase 8 (FVS) and Phase 9 (ECQO) with the cost model from these gap closures, gated on the staleness factor, the amortized migration cost, and the lower confidence bound. The multi-vector phase can start with the MaxSim cost model, gated on the joint recall distribution and the conservative escalation options. The unified optimizer specification is the next research target, with the corrected threshold formula, the joint recall distribution, and the `θ_GLS` computation as the central deliverables.
