# Execution Engine Research Review: iFVS, Factorized Processing, and ECQO

**Date:** 2026-08-01
**Status:** Critical review of incoming research chunk
**Preread:** the two prior docs in `docs/sql-research/` and the incoming research text on iFVS, factorized processing, and ECQO

---

## Context

The incoming research chunk covers three frontier problems for a Go-native, off-heap, zero-alloc hybrid database executor:

1. **iFVS** — Instance-Optimized Filtered Vector Search with dynamic PQ codebook generation
2. **Factorized Processing** — f-representation memory layout for graph-join-heavy queries
3. **ECQO** — Exact Cardinality Query Optimization across relational, graph, and vector predicates

The math is largely settled in the source literature. The research chunk is the engineering translation into Go with off-heap arenas. This review flags the specific technical issues, gaps, and overclaims before the design is committed to.

---

## iFVS Findings

### Numerical stability: `softplus` overflow

**Claim:** `w_f = softplus(w_z) = ln(1 + exp(w_z))`

**Issue:** For `w_z` > ~88 (any large positive), `exp(w_z)` overflows float32. The formula is the textbook definition, not the numerically stable form. Bugs of this class show up in production six months in, not at unit-test time.

**Recommendation:** Use the stable form: `max(w_z, 0) + log(1 + exp(-|w_z|))`. Softplus should never be implemented as the naive formula in production code.

### Per-query arena scope: 128KB scratchpad is significant

**Claim:** "A query-scoped arena. An arena is a contiguous block of memory requested directly via system calls (e.g., mmap) or allocated once at startup."

**Issue:** 128KB per query × 1000 concurrent queries = 128MB just for the iFVS codebooks. The arena reset semantics (`currentOffset = 0` at query end) only works if the arena is **per-query**, not global. The research chunk is silent on this. A global shared scratchpad would have queries trampling each other's offsets.

**Recommendation:** Explicit per-query arena allocation, returned to a pool at query end. Document the per-query memory budget. With 128KB iFVS + 512 bytes weights + scratchpad overhead, expect ~150KB per concurrent query.

### Hash function claim: `unsafe.Pointer` is misleading

**Claim:** "xxHash or MurmurHash3, which are easily implementable in pure Go using unsafe.Pointer over byte slices."

**Issue:** xxHash and MurmurHash3 are pure Go without `unsafe`. The `unsafe.Pointer` is only needed for the *arena offsets* into the codebook, not for hashing byte slices. Two different concerns conflated.

**Recommendation:** Cite the hash library used (`github.com/zeebo/xxh3` is the Go-native one) without `unsafe.Pointer`. Use `unsafe.Pointer` only where it's actually needed — the arena offset reads.

### Missing: training procedure for `W` and the MLP

**Claim:** "pre-learned parameter weight matrix `W ∈ R^{M×r×K×d}`" and a "lightweight MLP mapping" that produces `w_z`.

**Issue:** Both `W` and the MLP weights are trained. The research references them but never specifies:
- What training data is used (queries with known relevant/irrelevant filters?)
- How often the matrix needs retraining as the data distribution shifts
- How the filter fingerprint is encoded as input to the MLP
- What the loss function is (recall? Q-error?)

This is the part that takes the most research time in practice. The iFVS paper (2024) has the training procedure; the chunk doesn't reproduce it.

**Recommendation:** Before any iFVS code lands, document the training pipeline. Without it, `W` is undefined and the filter-aware scoring is a constant matrix that doesn't adapt.

---

## Factorized Processing Findings

### Memory savings ≠ time savings

**Claim:** "By avoiding the instantiation of the flat representation, memory bandwidth saturation is averted, and CPU caches remain highly effective."

**Issue:** The factorized form saves *memory* (16 bytes of metadata instead of duplicating payloads 500 times) but the *logical cardinality* of a 500 × 1000 join is still 500,000 — the consumer still has to traverse that many logical tuples. The downstream operators pay the same logical cost. Factorized processing compresses *representation*, not *work*.

**Recommendation:** Update the framing. The savings is in *intermediate materialization*, not in *execution*. The factorized representation is a memory/bandwidth optimization that does not change asymptotic time complexity.

### Hash bucket size: 64 bytes is too small for the claimed layout

**Claim:** "The hash table itself is an off-heap array of 64-byte cache-aligned buckets. Each bucket contains a spinlock (for parallel builds), a count, and an array of uint64 handles pointing back to the factorized blocks."

**Issue:** Spinlock (4 bytes, padded to 8) + count (4 bytes) + 8 × uint64 handles (64 bytes) = 76 bytes. Doesn't fit in 64 bytes. The realistic options are:
- Open addressing with 4 entries per bucket (44 bytes, fits)
- Chained overflow with the chain pointer outside the cache line (saves cache misses on bucket hits, pays them on overflow)
- 8 entries per bucket at 72+ bytes (cache-line aligned but two cache lines, defeating the purpose)

**Recommendation:** Pick one of the three and specify it. Open addressing is the natural fit for a 64-byte cache line.

### "Zero-materialization merge" is misleading

**Claim:** "The new group appends the uint64 column base offsets of the matched build tuples directly to the probe group's metadata."

**Issue:** This is still a write per match — 16 bytes of new FactorizedGroup metadata for each matched tuple pair. The savings is the *avoided payload materialization* (no string copies, no struct copies), not the avoided write. The metadata write is mandatory.

**Recommendation:** Frame as "zero-payload-materialization merge" or "zero-copy payload merge." The metadata is constructed, the payloads are referenced.

### Missing: factorized state under writes

**Issue:** Factorized representations are read-only by default. When a node is inserted, deleted, or updated, the f-representation for any active query referencing that node is invalidated. The research doesn't address:
- How a write invalidates in-flight factorized states
- Whether the factorized state is rebuilt per-query (likely yes, since it's query-scoped) or maintained across queries (much harder)
- How writes interact with the iFVS codebook regeneration (a write changes the base PQ codebook, which changes the iFVS perturbation)

**Recommendation:** Document the write path explicitly. The simplest model: factorized states are per-query, built fresh, no caching. Writes invalidate nothing because nothing is cached.

---

## ECQO Findings

### Sample size formula assumes i.i.d., but HNSW probe is biased

**Claim:** `N = ⌈(E²Z²p(1-p))⁻¹⌉` for sample size, with bias correction `C_bias` to compensate for non-uniform density.

**Issue:** The formula is the standard proportion estimator under i.i.d. sampling. HNSW BFS from the entry point is **not i.i.d.** — it's biased toward hub nodes (the HNSW highway). The `C_bias` correction is mentioned but never defined. Without a real bias model, the estimate is systematically optimistic on selective predicates (the probe sees more valid candidates than the dataset average because hubs are more likely to satisfy any predicate).

**Recommendation:** Either derive `C_bias` from the HNSW graph structure (degree distribution, hub-vs-spoke density), or use a different probe strategy that doesn't start from the highway. A random-walk probe with rejection sampling is unbiased but slower; a stratified probe stratified by HNSW layer is a compromise.

### Selectivity thresholds (0.05, 0.01) are empirical, not principled

**Claim:** "Optimal Trigger Condition" rows with `σ_rel ≪ 0.05`, `σ_vec ≪ 0.01`, etc.

**Issue:** These are empirical heuristics, not theorems. The cost formula is a model; the thresholds are calibration points. Different hardware, different data distributions, different index implementations will shift them. The research presents them as if they're principled.

**Recommendation:** Mark them as starting points that the optimizer calibrates against measured query latency. Implement a self-tuning mechanism: record (estimated selectivity, actual selectivity, query latency) per execution and adjust the thresholds online.

### "Repelled via an exclusion distance penalty" is not an algorithm

**Claim:** "If the scalar predicate is violated, the node is repelled via an exclusion distance penalty. This penalty dynamically reshapes the vector distribution, effectively pushing non-target vectors away from the query."

**Issue:** This is a phrase, not an algorithm. ACORN, NaviX, and JAG each specify a concrete mechanism. "Exclusion distance penalty" doesn't specify:
- What the penalty value is
- Whether it modifies the HNSW neighbor list or the distance ranking
- How it interacts with the small-world connectivity property (this is the core of the in-filtering research problem)
- What happens to graph connectivity when many nodes are penalized

**Recommendation:** Pick one of ACORN / NaviX / JAG and write the algorithm down. The implementation will not be derivable from "exclusion distance penalty" alone. ACORN's denser neighborhoods + traversal heuristics is the most cited; start there.

### Missing: cost model under MVCC

**Issue:** ECQO models the cost of a single-query execution. Under MVCC, queries have a snapshot view of the database. The cost model should account for:
- How the snapshot view affects selectivity estimation (the histogram and edge counts are at the snapshot timestamp, not the latest)
- How the factorized state interacts with the snapshot
- Whether the iFVS codebook is snapshot-scoped or live

**Recommendation:** Scope ECQO v1 to read-committed or snapshot-isolation under a read-only query. Defer the MVCC integration to a later phase.

---

## What This Chunk Enables for the Build

- **iFVS:** Concrete enough to write the Go code for the codebook generation, the per-query arena, and the ADC lookup table layout. Gated on the training procedure documentation.
- **Factorized Processing:** Concrete enough to write the `FactorizedGroup` struct, the bit-packed uint64 handle, and the FactorizedHashJoin operator. Gated on the bucket layout decision (open addressing vs chained).
- **ECQO:** Concrete enough to write the optimizer rules and the localized HNSW probe. Gated on the bias model for the probe and the threshold calibration mechanism.

---

## What This Chunk Is Missing

1. **The iFVS training procedure.** Without this, `W` and the MLP weights are undefined. **Blocking.**
2. **The factorized state under writes.** How do writes interact with active factorized states? Most likely: factorized state is per-query, no caching, writes don't invalidate anything. Document this explicitly.
3. **Concurrent query execution.** The hash bucket spinlock is mentioned but the broader concurrency model isn't. Per-query factorized state suggests each query has its own scratchpad, but the underlying HNSW graph is shared. How do reads of the shared HNSW graph interact with writes? Read-copy-update? Single-writer?
4. **MVCC integration.** The cost model assumes a single database state. Under MVCC, the state is snapshot-scoped. The research doesn't address this.
5. **The cost model thresholds are not derived.** The 0.05 / 0.01 numbers are starting points, not principles. The implementation should treat them as configurable calibration parameters, not constants.
6. **The actual in-filtering algorithm.** Pick ACORN, NaviX, or JAG. "Exclusion distance penalty" is a placeholder.

---

## How This Fits with the Prior Research

The mathematical foundations research prompt (in `mathematical-foundations-research-prompt-08-01-2026.md`) covers Q1–Q8: the algebra, semantics, equivalence laws, type system, cost model, approximation semantics, worked example, and open questions. That prompt is the *what* — what the engine has to be true about.

This research chunk is the *how* — specific algorithms, memory layouts, and Go-friendly data structures for three of the eight questions (Q5 cost model, Q6 approximation, plus the factorized processing that wasn't explicitly in the foundations prompt but should be added to Q1 as part of the algebra).

The two together are the full picture. The foundations tell you the engine has to support factorized processing; this chunk tells you how the factorized processing works in practice. The foundations tell you the cost model has to handle vector selectivity; this chunk tells you ECQO is one way to do it, with the specific HNSW-probe mechanism.

**The next missing chunk** is the **update path**: how factorized states, iFVS codebooks, and ECQO cost models behave when data is being mutated, not just read. The current chunk opens that gap (factorized state is read-only, iFVS assumes a static base codebook, ECQO models a single query). The update-path research should cover:
- How writes invalidate active queries
- How the base PQ codebook is updated when vectors are inserted/deleted
- How graph edge counts and histograms are maintained incrementally
- How MVCC interacts with the factorized execution model

That's the next research target.

---

## Summary of Open Items

| Item | Severity | Type |
|---|---|---|
| iFVS `W` and MLP training procedure | Blocking | Specification gap |
| Per-query arena pool | Blocking | Implementation detail |
| Hash bucket layout (64-byte alignment) | Blocking | Implementation decision |
| In-filtering algorithm (ACORN / NaviX / JAG) | Blocking | Algorithm choice |
| Factorized state under writes | High | Design gap |
| HNSW probe bias model | High | Math gap |
| Threshold calibration mechanism | Medium | Tuning infrastructure |
| MVCC interaction | Medium | Deferred to later phase |
| Softplus numerical stability | Low | Implementation bug |
| `unsafe.Pointer` framing in hash claim | Low | Documentation cleanup |
| Memory vs time framing for factorized | Low | Documentation cleanup |
| `C_bias` definition | High | Math gap |
