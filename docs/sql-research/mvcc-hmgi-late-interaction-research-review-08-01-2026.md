# Research Review: MVCC, HMGI, and Multi-Vector Late Interaction

**Date:** 2026-08-01
**Status:** Critical review of incoming research chunk
**Preread:** all prior docs in `docs/sql-research/`, especially `foundations-of-unified-query-algebra-08-01-2026.md` and `execution-engine-research-review-08-01-2026.md`

---

## Context

The incoming research chunk covers three frontier problems for the unified hybrid database:

1. **Lock-Free MVCC for HNSW and Factorized Graphs** — Copy-on-Write Slab Graph with uint64 bit-packed metadata + Epoch-Based Reclamation
2. **Graph-Partitioned Vector Memory Layouts (HMGI)** — Leiden algorithm + 4KB page co-location + ECBB pruning
3. **Multi-Vector (Late Interaction) SQL Execution** — ColBERT MaxSim as nested monoid comprehensions (Max ⊕ Sum) with PLAID-style pruning

The chunk is the most concrete of the research so far — most of the designs are specific enough to implement. But it has several real technical errors and missing specifications that must land before the implementation.

---

## 1. MVCC / CoW Slab Findings

### 1.1 The tombstone bit is in the wrong place

**Claim:** The uint64 routing slot is bit-packed as (1 bit tombstone, 15 bits degree, 48 bits offset).

**Issue:** The whole point of the Bridge Protocol — the central insight of the chunk — is that tombstoned nodes *remain in the graph for routing*. If the tombstone flag is in the routing slot, the search algorithm reads the bit, sees the tombstone, and skips the node. That *is* the navigational dead-end problem the protocol is supposed to solve.

**Recommendation:** Move the tombstone flag into the *slab* itself, alongside xmin/xmax. The routing slot should be (degree, offset) — purely for navigation. The atomic CAS operates on the offset only. Visibility is decided at the slab level, not at the routing-slot level.

### 1.2 The 15-bit degree field needs scoping

**Claim:** "15 bits — represents the number of active edges in the currently referenced slab."

**Issue:** The spec is silent on what "active edges" means. If it's the total edges for a node (summed across all HNSW layers), the field can overflow for dense graphs (M=64 neighbors × 16 layers = 1,024 edges, which fits, but tighter configurations fail). If it's per-layer, the field is more than enough.

**Recommendation:** State explicitly: degree is the number of active edges in the *currently referenced slab*, which holds the neighbor list for *one layer* of one node. Per-layer, 15 bits (32,767) is more than enough. If a slab holds multiple layers, the degree field needs to expand or a separate field per layer is needed.

### 1.3 Missing: xmin/xmax commit protocol

**Claim:** "xmin Transaction ID, xmax Transaction ID" stored in the slab.

**Issue:** MVCC visibility requires "xmin is committed and less than the snapshot ID." But where is the commit log? The slab stores xmin and xmax; the chunk doesn't say how a reader determines that xmin is committed. The visibility check `xmin < snapshot` is undefined for uncommitted transactions.

**Recommendation:** Specify the commit protocol. Standard options:
- A **commit log** (append-only sequence of committed transaction IDs) maintained by the transaction manager
- A **hybrid logical clock (HLC)** for snapshot IDs, with commit timestamps stored in the slab
- A **global transaction table** with status (active, committed, aborted)

The HLC approach is the cleanest for distributed MVCC. The commit log approach is simpler for single-node.

### 1.4 The T_grace enforcement problem

**Claim:** `M_limbo ≤ R × T_grace × S_slab`. "By proactively bounding T_grace (for instance, by forcefully aborting runaway analytical queries that exceed a predefined time-to-live threshold), the engine mathematically guarantees sublinear growth."

**Issue:** The bound assumes the system *enforces* T_grace. If a query exceeds the grace period, three options:
- (a) Extend the grace period (limbo grows, possibly unbounded)
- (b) Abort the query (but a long-running analytical query may not respect the abort signal — it could still hold references)
- (c) Block the writer (defeats the lock-free design)

The chunk says "forcefully aborting runaway queries" but doesn't specify what happens when the abort doesn't take. Under adversarial load, the limbo list can grow unboundedly.

**Recommendation:** Specify the policy:
- Queries that exceed T_grace are forcibly rolled back at the next safe point (a goroutine cancellation point, not in a tight loop)
- The limbo list has a hard upper bound; if the bound is hit, the writer blocks (rare but bounded)
- T_grace is a configurable parameter, not a constant

### 1.5 The HNSW layer structure in the slab is underspecified

**Issue:** The slab holds the neighbor list for *one layer*, but HNSW has multiple layers per node. The chunk doesn't say:
- How many slabs per node (one per layer? or one with all layers?)
- How layer transitions are represented in the slab
- How a CAS operation that affects multiple layers is implemented (do all layers CAS at once, or one at a time?)

**Recommendation:** One slab per node, holding the neighbor lists for *all* layers, with a layer index per neighbor. The slab format:
```
[ xmin | xmax | vector_offset | layer_count | (layer_index, neighbor_count, neighbor_offset)* ]
```
The CAS operates on the whole slab. Layer-local changes write a new slab with the modified layer. The degree field becomes `sum(neighbor_count across layers)`.

### 1.6 The "100% lock-free" claim is overclaim

**Claim:** "100% lock-free, zero-GC Go code."

**Issue:** EBR is *wait-free for readers* (a single atomic load acquires the snapshot). But the global epoch advance requires all active threads to reach the safe point. Under contention, the epoch advance can starve. Lock-free has a specific meaning (at least one thread makes progress in bounded steps); EBR doesn't satisfy this for the global epoch. The limbo list can grow unboundedly if no thread is reaching the safe point.

**Recommendation:** Frame as "wait-free reads, lock-free writes, EBR for memory reclamation with bounded limbo." The right reference for the formal model is the original EBR paper (Fraser, Harris) and the 2GEBR / NEBR / IBR extensions that handle the starvation case.

---

## 2. Leiden / HMGI Findings

### 2.1 The 7-nodes-per-page math is dimension-specific

**Claim:** "Capacity_page = ⌊4096/576⌋ = 7 vectors per page."

**Issue:** The 576 bytes per node assumes 128-dim float32 (512 bytes) + 64-byte header. For modern embedding sizes:
- 384-dim float32: 1536 bytes per vector → 2 nodes per page
- 768-dim float32: 3072 bytes per vector → 1 node per page (and 1KB wasted)
- 1536-dim float32: 6144 bytes per vector → doesn't fit at all

The chunk's math is precise for 128-dim (BERT-base) but doesn't generalize. The implementation needs a per-dimension page layout, or a multi-page node layout for high-dimensional vectors.

**Recommendation:** Define a per-dimension page layout in the catalog. For D > 1024, use a multi-page node (the slab spans 2 or more 4KB pages). The ECBB also scales: a D-dim bounding sphere is D × 8 bytes for center + 8 bytes for radius, way more than 64 bytes for D > 4.

### 2.2 The ECBB shape should match the distance metric

**Claim:** "Exact Cardinality Bounding Box (ECBB) — A compressed Minimum Bounding Sphere (MBS) representing the spatial boundaries of the vectors contained within the page."

**Issue:** A sphere is the right shape for *cosine distance* (vectors on the unit hypersphere, so a spherical cap is the natural bound). For *L2 distance*, a hyperrectangle is tighter (less volume to bound the same points). For *inner product*, neither is great; an L2-ball in the L2-normalized space is acceptable.

The chunk picks one shape (sphere) for all three metrics. The implementation needs a metric-aware bounding shape.

**Recommendation:** Choose the bounding shape per distance metric:
- Cosine: spherical cap on the unit hypersphere
- L2: axis-aligned hyperrectangle
- Inner product: L2-ball after normalization (effectively the cosine case)

Store the bounding shape in the page header (16-64 bytes depending on dimension and shape).

### 2.3 The Leiden migration threshold τ is mentioned but not derived

**Claim:** "ΔQ > τ where τ is a predefined migration threshold configured to prevent hysteresis and thrashing."

**Issue:** The standard problem in incremental clustering: without a specific threshold formula, the system can oscillate as a node migrates between pages that both claim it.

**Recommendation:** Derive τ from the Leiden paper's recommendation or from a measured distribution. Standard:
- `τ = 1 / (2 × |E|)` (from the Leiden paper's stability analysis)
- Or a percentile of the historical ΔQ distribution (e.g., migrate only when ΔQ is in the top 10% of recent changes)

Also implement:
- **Hysteresis:** a node that just migrated to page B cannot migrate again for N events
- **Dampening:** require N consecutive decisions to migrate, not just one

### 2.4 The Leiden partitioning vs HNSW layer tradeoff is unquantified

**Issue:** If vectors are physically co-located by Leiden community, the HNSW traversal (which jumps across communities to find distant vectors) pays the cross-page cost on every long-range hop. The community co-location helps *local* traversal but hurts *long-range* traversal.

The chunk doesn't say:
- At what HNSW layer does the cost of cross-community traversal dominate the benefit of intra-community locality?
- Should the page layout be per-layer (co-locate within the bottom 2 HNSW layers; leave upper layers un-co-located)?
- Or per-community (always co-locate by community, regardless of layer)?

**Recommendation:** Empirically measure the tradeoff with the LDBC SNB benchmark or equivalent. A reasonable starting policy: co-locate within the bottom 2 HNSW layers (which contain ~95% of nodes), leave upper layers un-co-located (the hubs naturally span communities). Refine based on measured query latency.

### 2.5 Missing: Leiden partitioning interaction with the WAL

**Issue:** Leiden community assignments are derived state, not source-of-truth state. If the assignments are persisted in the catalog, the WAL has to record the changes. If the assignments are derived (not persisted), they can be recomputed from the WAL, but the cost is the full Leiden re-run.

**Recommendation:** Persist community assignments in the catalog, with WAL logging on migration. Migration is a single CAS on the slab pointer, but the WAL has to record (node_id, old_page, new_page) for crash recovery. On WAL replay, the migrations are reapplied in order.

### 2.6 Missing: the cost of incremental Leiden in the cost model

**Issue:** The chunk says background workers do the migration, but doesn't say how much CPU the workers consume, or whether they can block queries under sustained churn.

**Recommendation:** Model the migration cost:
- Per migration: O(D) bytes for the AVX-512 copy + O(1) for the CAS + O(N_page) for the ECBB recompute
- Total migrations per second: bounded by the migration rate (a configuration parameter)
- Worker CPU budget: a fraction of one core (e.g., 25% of one core dedicated to migration)

If the migration queue exceeds a threshold, the optimizer should know (a flag in the catalog) and prefer plans that don't depend on the Leiden partitioning for the queries in flight.

---

## 3. Multi-Vector / MaxSim Findings

### 3.1 The ApproxMaxSim upper bound is wrong without residual-sign correction

**Claim:** "Because the global codebook centroids are mathematically clustered to represent regional spatial centers, this approximation provides a sound upper bound for the true relevance score."

**Issue:** This is only true if the residual vectors are non-negative. PLAID-style residual quantization uses signed 4-bit values (or 8-bit). The residual can push the true dot product *below* the centroid approximation.

**Recommendation:** Use the corrected upper bound: `centroid_dot + max_residual_contribution`, where `max_residual_contribution` is the L1 norm of the residual multiplied by the L1 norm of the query token. For 4-bit residuals, `max_residual_value × L1_norm(query) / sqrt(D)` is a sound upper bound.

Without this correction, the pruning is unsound and the recall guarantee is broken.

### 3.2 The MaxSim SQL syntax is missing

**Claim:** "By representing late interaction as a monoid comprehension, the engine's query optimizer can systematically apply algebraic rewrite rules."

**Issue:** The algebra can express MaxSim, but the *SQL surface syntax* is non-trivial:
- Document tokens are a matrix, not a scalar: `doc.tokens : MATRIX(VECTOR(D))` or similar
- Query tokens are a matrix: `:query_tokens : MATRIX(VECTOR(D))`
- The MaxSim operator has to be a SQL-callable function: `maxsim_score(doc.tokens, :query_tokens) → FLOAT`
- The ApproxMaxSim pruning has to be a query transformation rule (not user-visible)

The type system has to handle matrix types, which the prior foundations doc didn't include.

**Recommendation:** Add matrix types to the type lattice. Type rules for `maxsim_score`:
```
Γ ⊢ doc : MATRIX(VECTOR(D), L_doc)
Γ ⊢ query : MATRIX(VECTOR(D), L_query)
─────────────────────────────────────
Γ ⊢ maxsim_score(doc, query) : FLOAT
```

The query syntax: `SELECT id, maxsim_score(tokens, :query) AS score FROM docs ORDER BY score DESC LIMIT k`. The ApproxMaxSim pruning is a query rewrite rule applied during optimization.

### 3.3 The "tens of milliseconds" latency claim is corpus-dependent

**Claim:** "The engine can execute complex multi-vector queries with latencies in the tens of milliseconds."

**Issue:** Tens of ms is achievable on small corpora (10K-100K documents with PLAID pruning). On million-document corpora, the latency is hundreds of ms to seconds. The chunk doesn't specify the corpus size.

**Recommendation:** State the corpus size with the latency claim. Target: tens of ms on 100K-document corpora, hundreds of ms on 1M-document corpora, seconds on 10M+. Compare against PLAID's published numbers for honesty.

### 3.4 Missing: the matrix type system

**Issue:** The type lattice in the prior foundations doc (in `foundations-of-unified-query-algebra-08-01-2026.md`) has scalar types, VECTOR(N), NODE, EDGE, PATH. It doesn't have matrix types. Multi-vector queries need `MATRIX(VECTOR(D), L)` where L is the number of tokens (variable per document).

**Recommendation:** Extend the type lattice:
- `MATRIX(element_type, dim_1, dim_2, ...)` for fixed-shape matrices
- `MATRIX(element_type, dim)` for one-variable matrices (e.g., variable-length token sequences)
- The MaxSim operator consumes two `MATRIX(VECTOR(D), L)` and produces a FLOAT

The variable-length matrix case requires the type checker to handle dimension polymorphism (L is a value-level variable, not a type-level constant).

### 3.5 The "100% lock-free, zero-GC Go code" claim is overclaim (repeat)

**Claim:** Same as 1.6.

**Issue:** Same as 1.6. The MaxSim computation does have SIMD-heavy inner loops, but the SIMD setup (loading centroids, unpacking residuals via VPSHUFB) has Go-side orchestration that can allocate. The "100% lock-free" is overclaim.

**Recommendation:** Same as 1.6. Frame as "no heap allocations on the hot path, bounded stack growth, no GC involvement in the data path."

---

## 4. What's Strong in This Chunk

- The **CoW Slab Graph** is the right design. Decoupling routing from visibility is the correct answer to HNSW MVCC.
- The **Bridge Protocol** (tombstoned nodes remain routable) is exactly the ACORN/NaviX/JAG insight, formalized in the MVCC context. The chunk has the right intuition even if the bit-packing is wrong.
- The **EBR math** is correct in principle. The limbo bound `M_limbo ≤ R × T_grace × S_slab` is the right formulation.
- The **Leiden algorithm choice** over Louvain is correct — Leiden guarantees connected communities, which matters for physical co-location.
- The **ΔQ modularity formula** is the standard Newman formulation, applied per-node.
- The **Max monoid (⊕) over Sum monoid (⊗)** decomposition of MaxSim is mathematically clean. The associativity checks out.
- The **ECBB pruning** is a real optimization. Pruning entire pages from the search space is bounded-cost.
- The **ApproxMaxSim pruning** is the right technique (PLAID, Col-Bandit). The residual-sign issue is fixable.

---

## 5. What's Still Missing

| Item | Severity | Type |
|---|---|---|
| HNSW layer structure in the MVCC slab (point 1.5) | Blocking | Specification gap |
| xmin/xmax commit protocol (point 1.3) | Blocking | Specification gap |
| T_grace enforcement under adversarial load (point 1.4) | High | Design gap |
| Leiden migration threshold derivation (point 2.3) | High | Math gap |
| Dimension-aware page layout (point 2.1) | High | Implementation gap |
| ECBB shape per distance metric (point 2.2) | Medium | Design choice |
| Leiden partitioning vs HNSW layer tradeoff (point 2.4) | Medium | Empirical question |
| Leiden partitioning interaction with WAL (point 2.5) | Medium | Design gap |
| ApproxMaxSim residual-sign correction (point 3.1) | Blocking | Math correction |
| MaxSim SQL syntax and matrix type system (points 3.2, 3.4) | High | Type system extension |
| Cost of incremental Leiden in the cost model (point 2.6) | Medium | Cost model gap |
| Corpus-size caveat for the latency claim (point 3.3) | Low | Documentation cleanup |
| "100% lock-free" framing (points 1.6, 3.5) | Low | Framing cleanup |
| Tombstone bit location (point 1.1) | Blocking | Implementation error |
| 15-bit degree field scoping (point 1.2) | Medium | Specification cleanup |

---

## 6. Synthesis: How This Fits with the Prior Research

The mathematical foundations doc (`foundations-of-unified-query-algebra-08-01-2026.md`) covers the algebra, type system, equivalence laws, cost model. The execution engine review (`execution-engine-research-review-08-01-2026.md`) covers iFVS, factorized processing, and ECQO. This chunk extends both with the storage layer (MVCC slabs, Leiden partitioning) and the multi-vector algebra (MaxSim).

The bridges between the layers:

- **The CoW Slab is the storage realization of the HNSW index** referenced in the algebra. The algebra doesn't care how HNSW is stored, but the cost model needs the slab's CAS-based concurrency.
- **The Leiden partitioning is a cost-model input.** The ECBBs let the optimizer prune the search space; the partitioning changes the cost formula (intra-community is cheap, inter-community is expensive).
- **The MaxSim algebra is a new monoid composition** in the algebra. The Max monoid over the Sum monoid is a specific structure that the cost model has to handle.

The next research chunk should bridge these layers — specifically, the **query optimizer's interaction with the Leiden partitioning and the multi-vector (MaxSim) query path**. Without this bridge, the Leiden partitioning and the MaxSim algebra are ungrounded: they exist in isolation, not in the optimizer's cost model or plan enumeration.

The other missing chunks (update operations, expressive power, SQL constraints) are also still open but lower priority for the build. The optimizer-bridge is the next research target.

---

## 7. What This Chunk Enables for the Build

- **Phase 12 (ACID/MVCC):** The CoW Slab design is concrete enough to implement. The bit-packing needs the tombstone move (point 1.1), the layer structure needs to be specified (point 1.5), and the commit protocol needs to be defined (point 1.3). Gated on these three.
- **Phase 9 (ECQO):** The Leiden partitioning adds a new cost-model input. The ECBB pruning is a new optimization rule. Gated on the dimension-aware page layout (point 2.1) and the ECBB shape decision (point 2.2).
- **New phase (multi-vector):** The MaxSim algebra is implementable as a new monoid composition. Gated on the matrix type system extension (point 3.4) and the ApproxMaxSim residual-sign correction (point 3.1).
