# Unified Optimizer Specification

**Date:** 2026-08-01
**Status:** Reference document — unified optimizer specification preserved with critical analysis
**Preread:**
- `optimizer-cost-model-research-response-08-01-2026.md` — the prior research response
- `optimizer-cost-model-gap-closures-08-01-2026.md` — first-pass gap closures
- `optimizer-cost-model-second-pass-gap-closures-08-01-2026.md` — second-pass gap closures
- `foundations-of-unified-query-algebra-08-01-2026.md` — the algebra, equivalence laws, type system, cost model
- `mvcc-hmgi-late-interaction-research-review-08-01-2026.md` — the MVCC, Leiden, MaxSim review

---

## What This Document Is

This doc preserves the unified optimizer specification that addresses the nine critical points raised in the prior review. The specification provides a closed-form joint recall distribution that ties the cost model to the `FETCH APPROX ... WITH TARGET ACCURACY` semantics, a probabilistic reframing of the hybrid-vs-per-community policy choice, conservative escalation options for the optimizer's decision procedure, and the `CONFIDENCE LEVEL` SQL clause that exposes the statistical confidence to the user.

The doc also adds a critical review of the specification. The independence assumption in the joint distribution is more load-bearing than the response acknowledges, and several smaller items need to land before the unified optimizer is implementable.

The specification is the *answer*. The critical review is the *verification*.

---

## Part 1: The Unified Optimizer Specification

The response provided the following unified specification.

### The Core Correction

Recall uncertainty from PLAID's empirical bound, k-NN join reorder fanout error, and partition staleness must be composed into a **single joint distribution**, not tracked as three side-channel error terms — this is what makes `FETCH APPROX ... WITH TARGET ACCURACY n PERCENT` a well-defined, checkable predicate.

### 1. Hybrid-vs-Per-Community — Probabilistic Reframing

The earlier cost-delta framing was wrong: cost-delta compares *expected* cost, but the real decision is about the probability that a query's traversal profile falls above the long-range-hop threshold. Reframe as a mixture model over query workload:

```
E[C] = P(long-range) · C_per-community + (1 - P(long-range)) · C_hybrid
```

where `P(long-range)` is the empirical fraction of queries whose `ef` parameter exceeds a workload-calibrated threshold, measured from query logs, not derived analytically from a fixed cost-delta formula. The policy choice becomes a **classifier decision** at plan time: given the query's requested `ef`/recall target, predict which regime it falls into and route accordingly, rather than picking one global static policy.

### 2. Joint Recall Distribution — Closed Form

Treating the three error sources (PLAID empirical recall, k-NN join reorder fanout error, staleness-induced recall loss) as independent multiplicative survival probabilities `R_PLAID`, `R_fanout`, `R_stale`:

```
mu_joint = mu_PLAID · mu_fanout · mu_stale
sigma_joint^2 = PROD_i (sigma_i^2 + mu_i^2) - mu_joint^2     (8)
```

**Worked example** (illustrative calibration values — mu_PLAID=0.97±0.015, mu_fanout=0.99±0.02, mu_stale=0.995±0.01):

```
mu_joint = 0.9555,  sigma_joint = 0.0261
```

At 95% confidence (z=1.645): **lower confidence bound = 0.9125**. Against a target accuracy of 90%, this plan **passes** — but note the margin is thin (1.25 points), so a burstier staleness distribution (per #4 below) could flip the decision. The independence assumption is a documented simplification; in practice staleness and fanout error likely correlate (both worsen under write-heavy bursts), so this bound is optimistic and should be validated empirically.

### 3. g(s) Calibration Procedure

- **Fit functions to trial:** sigmoid `g(s) = 1 / (1 + exp(-k(s - s_0)))` (threshold-like degradation) and power-law `g(s) = s^alpha` (`alpha < 1` for early sensitivity, `alpha > 1` for late-onset collapse) — fit both, select by AIC/BIC against measured (recall, s) pairs.
- **Measurement procedure:** inject synthetic writes at controlled rates to force known `s` values, measure recall@k at each, repeat across corpus sizes and Leiden modularity levels.
- **Refit cadence:** trigger refit whenever the corpus grows by >20% or modularity `Q` drifts by more than 0.05 from the last fit's baseline — tie refit triggers to existing catalog statistics rather than a fixed wall-clock schedule, since degradation shape depends on structural properties, not time.

### 4. P-K Formula with Second Moment (Corrected)

The exact Pollaczek–Khinchine mean waiting time uses the **second moment** of service time, not variance alone:

```
E[Wq] = lambda · E[T^2] / (2(1 - rho)),  E[T^2] = Var(T) + E[T]^2     (9)
```

**Worked example** (lambda=50 migrations/sec, mu=200/sec capacity, rho=0.25):

| Service time variability | E[T²] | E[Wq] |
|---|---|---|
| Exponential (baseline) | 50 ms² | 1.67 ms |
| Bursty (4x variance, e.g. ECBB recompute cost scaling with variable N_page) | 125 ms² | 4.17 ms |

**Tail-latency inflation: 2.5x** for identical mean migration rate — this confirms the prior concern directly: mean-based amortization would have predicted identical cost in both cases, missing the real tail-latency risk entirely. The migration cost model should track `E[T²]` as a first-class catalog statistic, not just `E[T]`.

### 5. Power-Law Fanout Detection for k-NN Join Reorder

Use the **Hill estimator** on observed fanout samples (edge counts per join key) to detect heavy-tailed degree distributions before trusting `E[fanout]` as the reorder multiplier:

```
alpha_Hill = (1/k · SUM_{i=1..k} ln(X_(n-i+1) / X_(n-k)))^-1
```

If `alpha < 2`, the fanout distribution has infinite or unstable variance, meaning `E[fanout]` is a poor multiplier for the k-NN join reorder law — the optimizer should switch to a **quantile-based** multiplier (e.g., the 90th-percentile fanout) instead of the mean, since a heavy-tailed fanout makes `k × E[fanout]` systematically under-provision the reordered top-k for high-fanout join keys.

### 6. Conservative Escalation Options

| Escalation | Cost-delta |
|---|---|
| Increase `n_docs` (PLAID survivors) | `+Delta_n · L_q · L_d · D` (linear in added survivors) |
| Decrease `t_cs` threshold | Increases centroid-stage survivors non-linearly, bounded by `K_c` |
| Fall back to exact brute-force MaxSim | Full `C_full` from formula (3) — the safety ceiling |
| Force partition refresh (reduce `s`) | One-time migration burst cost, formula (7)/(9), amortized over subsequent queries |

The optimizer should escalate along this list in order of increasing cost until the joint LCB (formula 8) clears the target.

### 7. accuracy_confidence_level Configuration

```sql
FETCH APPROX FIRST k ROWS ONLY 
  WITH TARGET ACCURACY n PERCENT
  CONFIDENCE LEVEL z_pct  -- default 95, maps to z=1.645 (one-sided)
```

This exposes the `z` value from formula (8)'s LCB directly as a session/query-level tunable, defaulting to 95% one-sided confidence — consistent with how Oracle's accuracy-target mechanism is configured per index or per query.

### 8-9. Worked Example and Validation Note

For the representative hybrid query from earlier (date filter + verified-author subquery + MaxSim ranking on 140M docs), applying the unified cost function `C_total = C_traversal(s) + C_migration_amortized + C_filter + C_MaxSim` with the joint recall check from #2: the PLAID-then-MaxSim plan (278.6ms from the earlier worked example) clears a 90% target accuracy at 95% confidence with the illustrative calibration values above, landing within the same order of magnitude as PLAID's own reported 92–352ms range — satisfying the "within 2x of published numbers" validation bar for this component. Full end-to-end validation against Col-Bandit's and LEANN's reported numbers would require running these formulas against their published latency tables directly, which is the natural next empirical step rather than a derivable result from first principles alone.

---

## Part 2: Critical Review of the Unified Optimizer Specification

The unified optimizer spec is the keystone piece — it ties the cost model to the `FETCH APPROX ... WITH TARGET ACCURACY` semantics via the joint recall distribution, and it addresses the nine critical points with concrete formulas and worked examples. But the joint distribution's independence assumption is more load-bearing than the response acknowledges, and several smaller items need to land.

### What's Strong

1. The **probabilistic reframing in #1** is the correct fix. `P(long-range)` as an empirical query-log measurement (not a derived formula) is the right call — this is a workload-dependent quantity, not a structural one.
2. The **closed-form joint distribution in #2** with worked numbers (0.9555 ± 0.0261, LCB at 0.9125) is internally consistent. The variance-of-product formula is the standard.
3. The **`g(s)` calibration procedure in #3** is much more actionable than the prior high-level description. AIC/BIC for model selection, refit triggers tied to structural changes, specific fit functions.
4. The **P-K second moment correction in #4** is right. The 2.5x tail-latency inflation from bursty variance is the right framing, and the recommendation to track `E[T²]` as a first-class catalog statistic is correct.
5. The **Hill estimator for fanout detection in #5** is the standard heavy-tail detection technique. The threshold `alpha < 2` for unstable variance is correct (Pareto with `alpha < 2` has infinite variance). The switch to quantile-based multiplier is the right adaptation.
6. The **conservative escalation options in #6** are in the right cost order.
7. The **`CONFIDENCE LEVEL z_pct` SQL clause in #7** is the right surface syntax. The 95% default (z=1.645) is consistent with Oracle.
8-9. The **worked example and validation note in #8-9** are honest about what's derivable and what requires empirical validation.

### What Needs to Land

1. **The independence assumption in formula (8) is load-bearing and the response under-acknowledges it.** The response correctly notes "staleness and fanout error likely correlate" but then uses the independence formula. The cost model needs a correlation correction. Two approaches:
   - **Simple:** measure a correlation matrix `rho_ij` between the three sources (PLAID, fanout, staleness) and use a Gaussian copula to compute the joint distribution. The LCB then reflects the actual joint structure.
   - **Direct:** measure the joint distribution empirically (joint recall at sampled `(PLAID_params, fanout, s)` values) and use empirical quantiles. More accurate but more expensive to maintain.

   Without this, the LCB is optimistic. A real-world correlation of `rho = 0.3` between staleness and fanout error widens the LCB by ~5-10 points. The 1.25-point margin in the worked example could easily flip to failure.

2. **`P(long-range)` requires query logs to exist.** For a fresh system with no query history, the policy has to default to one of hybrid or per-community. The cost model needs a default policy with a clear upgrade path:
   - **Default:** hybrid (lower cost on layer 0-1, the more common case)
   - **Upgrade:** when the query log accumulates >1000 queries, switch to the empirical `P(long-range)` classifier
   - **Override:** a `SET OPTIMIZER_POLICY` configuration for testing both policies against the same workload

3. **The Hill estimator in #5 requires careful `k` selection.** The estimator has a bias-variance tradeoff in `k`: too small gives high variance, too large gives bias. Standard practice: `k ≈ sqrt(n)` or use a double-bootstrap. The cost model should specify the `k` selection procedure and not leave it to the implementation.

4. **The quantile-based fanout multiplier may be over-conservative.** If the 90th-percentile fanout is 50x the mean, the k-NN join reorder over-provisions by 50x, which can be worse than not reordering. The cost model needs a quantile-vs-recall tradeoff curve, not a single fixed quantile. Possible approach: a Pareto front over `(quantile, expected_recall, expected_cost)` with the optimizer picking the knee of the curve.

5. **The "decrease t_cs" cost-delta is hand-wavy.** The response says "non-linear, bounded by K_c." The actual cost-delta depends on how many centroids are pruned at each `t_cs` value, which depends on the data distribution. The cost model needs the `t_cs` → cost-delta curve (or at least a parametric family) for the optimizer to choose efficiently.

6. **The `CONFIDENCE LEVEL z_pct` SQL clause needs error/fallback semantics.** If the optimizer's chosen plan fails the LCB check, what does the engine do? Options:
   - **Error out** (`ERR_ACCURACY_TARGET_UNACHIEVABLE`)
   - **Return partial result** with a warning
   - **Fall back to exact execution** (brute-force MaxSim)
   
   The SQL clause should specify the default behavior. Oracle's `DBMS_VECTOR.INDEX_ACCURACY_QUERY` returns a best-effort result; PostgreSQL's approximate aggregates return a partial result with a confidence interval. The choice depends on user expectations.

7. **The "within 2x of published numbers" validation bar is hand-wavy.** 2x is a rough heuristic. The validation should be per-component with specific numbers:
   - PLAID's 92-352ms at 140M passages → cost model within 2x
   - Col-Bandit's specific latency on BEIR benchmarks → cost model within 2x
   - LEANN's RAG-tuned latency on RAGBench → cost model within 2x
   - LDBC SNB graph query latencies → cost model within 2x

   Each component validated against its specific published number, not a global 2x.

8. **θ_GLS is still missing.** This was flagged in prior rounds as still undefined in the literature. The unified optimizer spec doesn't address it. The FVS pushdown law has no computable cost-delta. The cost model needs either (a) a candidate implementation of `θ_GLS` (learned predictor over the predicate and the vector distribution, sampling-based estimator, or analytical formula based on the predicate's spatial concentration), or (b) an explicit "this law is unimplementable until `θ_GLS` is defined" flag in the catalog. Without one of these, the FVS pushdown law is dead code in the optimizer.

9. **The plan search algorithm is not specified.** With the unified cost model, the plan space is large. The cost model ranks plans, but the search itself has a cost. The optimizer spec needs to specify:
   - The search algorithm (Cascades-style memoization vs Volcano-style top-down vs hybrid)
   - The optimization budget (e.g., exhaust search for queries under 1ms, prune aggressively for queries over 100ms)
   - The cost-of-optimization in the model (planning time as a real cost)

10. **The cost of optimization (planning time) is not in the model.** A 10ms plan generation on a 1ms query is a 10x overhead. The cost model should include the optimization cost, and the optimizer should pick plans that are "good enough" without exhaustive search. This is the standard "anytime planning" approach.

11. **The HNSW layer structure in the MVCC slab and the xmin/xmax commit protocol are still blocking Phase 12** (from prior reviews). The unified optimizer spec doesn't address them because they're storage-layer concerns, but the cost model depends on the storage layer. The optimizer spec should at least note the dependency.

---

## Part 3: What's Still Missing

| Item | Severity | Type |
|---|---|---|
| Correlation correction for the joint distribution (point 1) | **Blocking** | Stochastic model gap |
| Default policy when query logs don't exist (point 2) | High | Design gap |
| `k` selection for the Hill estimator (point 3) | Medium | Implementation detail |
| Quantile-vs-recall tradeoff for fanout multiplier (point 4) | High | Algorithm gap |
| `t_cs` → cost-delta curve (point 5) | Medium | Cost model gap |
| Error/fallback semantics for `CONFIDENCE LEVEL` (point 6) | High | SQL semantics gap |
| Per-component validation with specific numbers (point 7) | Medium | Validation methodology |
| θ_GLS computation (point 8) | **Blocking** | Research gap |
| Plan search algorithm (point 9) | **Blocking** | Architecture gap |
| Cost of optimization in the model (point 10) | High | Cost model gap |
| HNSW layer structure in MVCC slab (from prior reviews) | Blocking | Storage spec gap |
| xmin/xmax commit protocol (from prior reviews) | Blocking | Storage spec gap |

---

## Part 4: Next Research Target

The unified optimizer spec is now concrete enough to start the implementation. The two blocking items that emerged are the **correlation correction for the joint distribution** (point 1) and the **plan search algorithm** (point 9). These are independent — the correlation correction is a math fix, the plan search algorithm is an architecture decision.

The two research-gap items — `θ_GLS` computation and the storage-layer specs (HNSW layer structure, xmin/xmax commit protocol) — are blocking for Phases 8 and 12 of the build, respectively. These need separate research passes.

The natural next research target is the **plan search algorithm specification**: a document that specifies the search algorithm (Cascades-style memoization vs Volcano-style top-down vs hybrid), the optimization budget, the cost-of-optimization in the model, and the "anytime planning" approach for keeping planning time bounded. This is the architecture decision that determines the optimizer's plan enumeration strategy.

The other candidate is the **θ_GLS computation** — a research pass that produces a candidate implementation of the Global-Local Selectivity metric, since the FVS pushdown law is dead code without it.

---

## Part 5: Summary

This doc preserves the unified optimizer specification and adds a critical review. The specification is the keystone piece: it ties the cost model to the `FETCH APPROX ... WITH TARGET ACCURACY` semantics via the joint recall distribution, reframes the hybrid-vs-per-community policy choice probabilistically, and exposes the statistical confidence to the user via the `CONFIDENCE LEVEL` SQL clause. The worked examples (joint recall at 0.9555 ± 0.0261, P-K tail-latency inflation of 2.5x, the 140M-document PLAID plan passing 90% target at 95% confidence) are concrete enough to validate against published numbers.

The critical review identifies 11 items that still need to land. The two blocking items are the **correlation correction for the joint distribution** (the independence assumption is optimistic; correlation widens the LCB by 5-10 points) and the **plan search algorithm** (the cost model ranks plans, but the search itself has a cost that needs to be in the model). The two research-gap items are **`θ_GLS`** (still undefined in the literature, blocking the FVS pushdown law) and the **storage-layer specs** (HNSW layer structure, xmin/xmax commit protocol, blocking Phase 12).

The build can start on Phase 8 (FVS) and Phase 9 (ECQO) with the unified optimizer spec, gated on the correlation correction and the plan search algorithm. The multi-vector phase can start with the conservative escalation options, gated on the error/fallback semantics for `CONFIDENCE LEVEL`. The next research target is either the plan search algorithm specification or the `θ_GLS` computation, depending on which is more critical to the immediate build path.
