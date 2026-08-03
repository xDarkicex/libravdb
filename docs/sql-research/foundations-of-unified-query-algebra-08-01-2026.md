# Foundations of the Unified Query Algebra

**Date:** 2026-08-01
**Status:** Foundational reference document
**Preread:**
- `unified-sql-engine-vision-08-01-2026.md`
- `unified-query-superset-architecture-08-01-2026.md`
- `execution-engine-research-review-08-01-2026.md`

---

## What This Document Is

This is the **foundational reference** for the unified query algebra that the SQL/PGQ + vector + graph engine sits on. It synthesizes the formal mathematical treatment from the research literature (monoid comprehensions, graph path semantics, row polymorphism, approximation semantics) into a form that:

1. The parser type checker can be transcribed from
2. The optimizer rule set can be derived from
3. The executor can be verified against
4. Future contributors can read to understand what the engine has to be true about

It is **not** a research review (that is `execution-engine-research-review-08-01-2026.md`). It is **not** a research prompt (that was the deleted `mathematical-foundations-research-prompt-08-01-2026.md`). It is the **operational foundation** — what the build lands on.

The companion engineering doc, when it exists, will translate each formal concept into a Go-side implementation. This document is the *what*; the engineering doc is the *how*.

---

## 1. The Unified Algebra: Monoid Comprehensions

The algebra that uniformly captures relational, graph, and vector operations is the **monoid comprehension calculus** (Fegaras & Maier 1995, Buneman 1996, building on Wadler 1990).

### 1.1 The Calculus

A monoid is `(M, ⊕, Z)` — a type `M`, an associative merge operator `⊕`, and an identity `Z`. A monoid comprehension takes the form:

```
⊕ { e | q_1, q_2, ..., q_n }
```

where `⊕` is the monoid merge, `e` is the head expression (the projection), and the `q_i` are qualifiers — generators `v ← C` (drawing elements from a collection) or predicates (filters).

This is the canonical form for relational operations, generalized to any monoid.

### 1.2 The Four Monoids

The unified engine uses four monoids, each suited to a different operation class:

| Monoid | Domain | Merge (⊕) | Identity (Z) | Commutative | Idempotent | Use |
|---|---|---|---|---|---|---|
| **Set** | `P(T)` (powerset) | Union `∪` | `∅` | Yes | Yes | Distinct relational queries, graph node sets |
| **Bag** | `B(T)` (multiset) | Multiset union `⊎` | `∅` | Yes | **No** | Standard SQL multiset semantics |
| **List** | `[T]` (sequence) | Concatenation `∘` | `[]` | No | No | Graph path traversals (ordered sequences) |
| **Similarity** | Ordered lists of length ≤ k | Sort-Truncate `⊕_{d,q}^k` | `[]` | No | **No** | Vector top-k nearest neighbor searches |

### 1.3 The Bounded Similarity Monoid

The novel extension for vector top-k operations. Formally:

```
M_topk(d, q) = (L_≤k(T), ⊕_{d,q}^k, [])
```

Where:
- `L_≤k(T)` is the set of ordered lists of tuples, with maximum length `k`
- The merge operator: `L_1 ⊕_{d,q}^k L_2 = Π_≤k(sort_{d(x.v, q)}(L_1 ∘ L_2))`
  - Concatenate, sort by ascending distance to query vector `q`, truncate to top-k
- Empty list `[]` is the identity

**Properties:**
- **Associative** (the fundamental monoid requirement) ✓
- **Not commutative** in the strict list-order sense (but the unordered set converges) ✓
- **Not idempotent** — this is a critical consequence:

  > `(A, A) ⊕_{d,q}^k ≠ A`. Feeding the same elements twice doubles them before truncation, which can push valid elements out of the top-k. This means **the standard relational optimization `σ_p σ_p = σ_p` is unsound in the similarity monoid**. Duplicate elimination before top-k is unsafe.

### 1.4 Implications for Implementation

- **Every operator in the engine must specify which monoid it operates in.** The optimizer must be monoid-aware: set-monoid optimizations are different from list-monoid optimizations are different from similarity-monoid optimizations.
- **The similarity monoid's non-idempotence rules out a class of standard relational rewrites.** Specifically, projection pushdown and duplicate elimination must be restricted to not interfere with the top-k ranking.
- **The List monoid is the right monoid for graph path traversals.** Path order matters; the engine preserves it through the algebra.

---

## 2. Formal Semantics

### 2.1 Property Graph

A property graph is a tuple `G = (V, E, ρ, L, P)`:
- `V` — set of vertices
- `E` — set of directed edges
- `ρ: E → V × V` — assigns source and target vertices
- `L` — assigns sets of labels to elements (vertices and edges)
- `P` — assigns key-value properties

### 2.2 `GRAPH_TABLE` Semantics

`GRAPH_TABLE(G MATCH p COLUMNS c)` evaluates to a **multiset of relational tuples**.

- `[[p]]_G` — evaluation function maps path pattern `p` to a set of bindings (homomorphisms from pattern graph to data graph)
- A path `π` in `G` is an alternating sequence `v_0, e_1, v_1, ..., e_n, v_n`
- `COLUMNS c` acts as a relational projection `π_c` over the bindings, collapsing topology into a standard relational schema

### 2.3 Path Restrictors as Set-Theoretic Restrictions

Restrictors are **set-theoretic restrictions on the path set**, applied *before* the `COLUMNS` projection. The engine implements them as filters on the path enumeration.

| Restrictor | Mathematical Constraint | Determinism |
|---|---|---|
| `WALK` | No restriction. Bounded by finite length quantifier `{1,n}` for termination. | **Non-deterministic** (multiple paths match) |
| `TRAIL` | `∀ i ≠ j, e_i ≠ e_j` (no edge repeated) | **Non-deterministic** |
| `SIMPLE` | `∀ i ≠ j ∈ {0,...,n-1}, v_i ≠ v_j` (no vertex repeated, except possibly endpoints) | **Non-deterministic** |
| `ACYCLIC` | `∀ i ≠ j, v_i ≠ v_j` (no vertex repeated at all) | **Non-deterministic** |
| `ANY SHORTEST` | Choice function: select one path from all paths between `(v_0, v_n)` minimizing `|π|` | **Non-deterministic** (choice) |
| `ALL SHORTEST` | All paths between `(v_0, v_n)` whose length equals the minimum possible | **Deterministic** (all returned) |
| `SHORTEST k GROUP` | Partition by `(v_0, v_n)`, sort groups by length, return first k groups | **Deterministic** (all in groups returned) |

**Important correction from the source research:** the "Deterministic" label in the source research is wrong. Determinism here means *the result set is fully determined* (ALL SHORTEST, SHORTEST k GROUP). Non-determinism means *a subset or choice is made* (WALK, TRAIL, SIMPLE, ACYCLIC, ANY SHORTEST). The engine must implement ANY SHORTEST with a **choice function** — by default, lex-smallest sequence of edge/vertex IDs — to ensure reproducible query results.

### 2.4 Tie-Breaking for Top-K

When `ORDER BY distance LIMIT k` produces ties (multiple vectors at exactly the same distance), the total ordering is completed with a secondary lexicographic order on a unique record identifier:

```
d(x_i, q) ≺ d(x_j, q) ⟺ d(x_i, q) < d(x_j, q) ∨ (d(x_i, q) = d(x_j, q) ∧ id_i < id_j)
```

This guarantees deterministic pagination and reproducible results.

### 2.5 Approximation Semantics: Recall@k

`FETCH APPROX FIRST k ROWS ONLY WITH TARGET ACCURACY n PERCENT` shifts the semantics from exact sets to probabilistic databases. The formal constraint:

```
E[ |S_approx ∩ S_exact| / k ] ≥ n / 100
```

This is **expected Recall@k**, not precision, F1, or distance-error. The optimizer treats it as a probabilistic constraint: minimize expected execution cost `C` subject to `P(Recall ≥ α) ≥ 1 - δ`.

---

## 3. The Type System

### 3.1 The Lattice

The unified type system covers scalar types, vector types, and graph types:

```
TOP: any
  ├── Scalar: INT | BIGINT | FLOAT | DOUBLE | TEXT | BOOLEAN | TIMESTAMP | DATE | ...
  ├── Composite: ROW(...) | ARRAY(T) | MAP(K, V)
  ├── VECTOR(N) — N is a type-level parameter
  └── Graph: NODE(ρ) | EDGE(ρ) | PATH
        where ρ is a row variable (unknown remainder of property map)

BOTTOM: NULL
```

### 3.2 VECTOR(N) as a Parameterized Type

`VECTOR(N)` is a type constructor with `N ∈ ℕ` as a type-level parameter. The subtyping relation is **invariant with respect to N**: `VECTOR(128)` is not a subtype of `VECTOR(256)`, and vice versa. The metric operators (`<->`, `<#>`, `<=>`) require dimensional matching:

```
Γ ⊢ e_1 : VECTOR(N)    Γ ⊢ e_2 : VECTOR(N)
─────────────────────────────────────────────
       Γ ⊢ e_1 <-> e_2 : FLOAT
```

**Implementation question (deferred to the engineering doc):** is `N` a *dependent type* (value-level integer carried in the type), a *refinement type* (with N as a refinement requiring constraint solving), or a *runtime check* (with the type system treating all vectors as `VECTOR(?)`)? Each has different implementation cost and different guarantees. Recommendation: start with runtime check, evolve to refinement type as the type checker matures.

### 3.3 Graph Types via Row Polymorphism

Graph elements have open, heterogeneous property sets. Modeled via **row polymorphism** (Wand 1991):

```
Γ ⊢ node : { id: ID, labels: Set⟨Label⟩, p_1: τ_1, ..., p_n: τ_n | ρ }
```

Where `ρ` is a row variable — the statically unknown remainder of the property map. Subtyping is governed by:
- **Width subtyping** — a record with more properties is a subtype of one with fewer (Person with email is a subtype of Person without)
- **Depth subtyping** — subtyping within the property types themselves

### 3.4 Type Rules for Unified Operators

| Operator / Expression | Type Rule | Description |
|---|---|---|
| Similarity (`<->`) | `Γ ⊢ e_1 : VECTOR(N), Γ ⊢ e_2 : VECTOR(N) ⊢ Γ ⊢ e_1 <-> e_2 : FLOAT` | Dimension matching required |
| Graph node extraction | `Γ ⊢ G : GRAPH, Γ ⊢ p : Label ⊢ Γ ⊢ MATCH (n:p) : { labels ⊇ {p} | ρ }` | Row-polymorphic node type |
| Path restrictor | `Γ ⊢ π : WALK, is_injective(vertices(π)) ⊢ Γ ⊢ π : ACYCLIC` | Refinement of path type |
| `GRAPH_TABLE` output | `Γ ⊢ G : GRAPH, Γ ⊢ P : Pattern, Γ ⊢ c : Projection ⊢ Γ ⊢ GRAPH_TABLE(G, P, c) : Relation(τ_c)` | Converts topology to multiset relation |

### 3.5 Implications for Implementation

- The type checker implements the full lattice with the four monoid classes as type qualifiers on collections.
- The VECTOR(N) implementation choice (dependent vs refinement vs runtime) is deferred but should be tracked in the engineering doc.
- The row polymorphism machinery has to handle property lookup at runtime (the row variable `ρ` is dynamically extended as properties are accessed).

---

## 4. Equivalence Law Catalog

The optimizer's rewrite rules derive from these equivalence laws. Each law has a **condition of validity** that the optimizer checks before applying the transformation.

### 4.1 Top-K Pushdown Through Selection (FVS Core Law)

The fundamental law for Filtered Vector Search:

```
topk(k, σ_p(R), d) ≈ σ_p(topk(k + c(sel(p), θ_GLS), R, d))
```

| Aspect | Detail |
|---|---|
| **LHS** | Filter first, then top-k. Pre-filtering. |
| **RHS** | Top-k with oversampling, then filter. Post-filtering. |
| **Condition** | Valid probabilistically. Requires monotonic similarity distribution relative to `p`. The oversampling factor `c` is a function of `sel(p)` (predicate selectivity) and `θ_GLS` (Global-Local Selectivity metric). |
| **Failure mode** | If `θ_GLS` indicates the predicate is spatially concentrated (low global selectivity, high local selectivity), oversampling fails and the optimizer must switch to pre-filtering. |

**`θ_GLS` (Global-Local Selectivity) metric** is the correlation between global predicate selectivity and local vector neighborhood selectivity. Defined in the FVS literature; **must be made computable** before this law is implementable. This is the blocking research item.

### 4.2 K-NN Join Reorder

```
topk(k, R ⋈_E S, d) ≈ R ⋈_E topk(k × E[fanout], S, d)
```

| Aspect | Detail |
|---|---|
| **LHS** | Join, then top-k. |
| **RHS** | Top-k on the inner side, then join. |
| **Condition** | The join is 1:N (vector on the N side), and the join predicate **preserves rank monotonicity** (rank-join monotonicity — a better rank on the inner side corresponds to a better rank on the joined result). |
| **Adjustment** | The k limit is scaled by the expected fanout degree. |

**Rank-join monotonicity** is the formal property from Ilyas et al. The simple phrasing "the join predicate does not prune the minimal elements" is approximately right but imprecise.

### 4.3 Selection Pushdown Into `GRAPH_TABLE`

```
σ_p(GRAPH_TABLE(G, P)) = GRAPH_TABLE(σ_p(G), P)
```

| Aspect | Detail |
|---|---|
| **Condition** | `p` evaluates exclusively on properties of a specific node/edge, AND `p` is post-restrictor-evaluable (its evaluation cannot affect which paths the path algebra produces). |
| **Failure mode** | If `p` filters on a node used by the path restrictor (e.g., TRAIL bound to a specific node, ACYCLIC pruning through a filtered node), pushing `p` can shrink the path set and change the match. |

### 4.4 Standard Relational Laws (Carry Over)

The following relational equivalences hold in the set and bag monoids:
- `σ_p σ_q(R) = σ_q σ_p(R)` — selection commutativity
- `σ_p(σ_p(R)) = σ_p(R)` — selection idempotence
- `π_X(π_Y(R)) = π_X(R)` when `X ⊆ Y` — projection idempotence

**Critical caveat:** selection idempotence (`σ_p σ_p = σ_p`) is **unsound in the similarity monoid**. Filtering twice with the same predicate *can change the top-k* if the predicate has tied values at the top-k boundary. The optimizer must restrict the idempotence rule to set/bag monoids only.

### 4.5 Implementation Guidance

- The optimizer rule set is organized by monoid. Each rule has a guard that checks (a) the monoid context, (b) the validity conditions from the equivalence law, (c) the cost improvement.
- The FVS rule (4.1) and the k-NN join rule (4.2) are the *most valuable* rules for hybrid queries. They drive the FVS strategy selection.
- The graph predicate pushdown (4.3) drives the early-filter optimization in graph joins.

---

## 5. The Unified Cost Model

### 5.1 Operator Cost Formulas

| Operator | Expected Cost | Selectivity / Cardinality Dependencies |
|---|---|---|
| Relational Filter `σ_p` | `C_cpu × N` | 1024-bin equi-depth histogram |
| HNSW Probe `topk` | `C_dist × ef × log(N) × D` | ECQO adaptive sampling, `θ_GLS` |
| Graph Reachability | `C_edge × E[deg(v)]^hops` | Node degree distribution, label selectivity |
| Hash Join `⋈` | `C_hash × N_outer + C_probe × N_inner` | Join selectivity, uniformity assumptions |

**HNSW cost correction:** the source research used `O(log(N_filtered) × M × D)` which conflates two parameters. The correct formulation is `O(ef × log N × D)` where:
- `ef` is the search-time expansion factor (controls recall/latency trade-off)
- `log N` is the HNSW layer-traversal cost
- `D` is the per-node distance computation cost
- `M` (max-neighbors per layer) is a construction-time parameter, not a per-query cost

### 5.2 Graph Traversal Cost by Restrictor

- `WALK`, `TRAIL`, `SIMPLE`, `ACYCLIC` with length bound `{1, k}`: `O(|E|^k)` worst case, exponential
- `ALL SHORTEST`: `O(|V| + |E|)` via BFS
- `ANY SHORTEST`: `O(|V| + |E|)` via BFS + tie-breaking
- Unbounded `TRAIL`: NP-hard (Hamiltonian path), constrained in practice via DFS with edge marking

### 5.3 ECQO: Unified Cardinality Estimation

The cost model needs accurate selectivity estimation across all three paradigms. The **Exact Cardinality Query Optimization (ECQO)** approach (Exqutor, 2024) uses:

- **Vector selectivity (σ_vec):** localized ANN probe within the HNSW index at planning time. The probe is biased toward hub nodes; a bias correction factor `C_bias` is required (currently undefined in the literature — see open questions).
- **Relational selectivity (σ_rel):** 1024-bin equi-depth histogram, O(log 1024) = O(10) lookup.
- **Graph selectivity (σ_graph):** degree distribution statistics, label selectivity, edge count.

These combine into a unified cost model that ranks candidate plans.

### 5.4 Selectivity Thresholds Are Calibrated, Not Constant

The source research lists "optimal trigger conditions" with specific thresholds (e.g., `σ_rel ≪ 0.05`, `σ_vec ≪ 0.01`). These are **empirical starting points, not theorems.** The implementation must:

- Treat the thresholds as **configurable calibration parameters**
- Implement a self-tuning mechanism: record (estimated selectivity, actual selectivity, query latency) per execution
- Adjust the thresholds online based on observed error

### 5.5 Approximation Cost vs Accuracy

For each approximate operator, the cost-accuracy curve:

- HNSW `ef` parameter: roughly exponential cost increase for linear recall gain
- IVF `nprobe` parameter: similar shape
- Graph traversal fanout: linear cost increase for higher recall

The optimization problem: minimize cost `C` subject to `P(Recall ≥ α) ≥ 1 - δ`. This is a stochastic programming problem; the optimizer's job is to allocate the accuracy budget across operators.

---

## 6. Error Composition (Independent vs Correlated)

When a query pipeline concatenates approximate operators, the end-to-end accuracy depends on error independence:

| Case | Composition Rule | Example |
|---|---|---|
| **Independent errors** | Multiplicative: `recall_combined = recall_1 × recall_2 × ...` | 95%-accurate HNSW + 95%-accurate Bloom filter on graph edges → 0.9025 end-to-end |
| **Correlated errors (uniform filter)** | Additive: filter on a predicate uniform across the vector space preserves the base recall | 95% HNSW + 100% deterministic filter → 95% end-to-end |
| **Correlated errors (concentrated filter)** | **Catastrophic** — recall can drop below 20% via the **navigational dead-ends** phenomenon | 95% HNSW + filter isolating a sparse sub-manifold → recall plummets |

The third case is the *in-filtering research problem* (ACORN, NaviX, JAG). When the filter concentrates the vector distribution, the HNSW graph loses small-world connectivity and the search algorithm gets trapped in local minima.

### Implications for the Optimizer

- The optimizer must estimate the correlation structure, not just the per-operator accuracies.
- A "concentration test" on the filter (how spatially concentrated the surviving vectors are) flags correlated-error cases.
- The FVS strategy choice (pre-filter / post-filter / in-filter) is driven by the correlation structure:
  - **Independent / uniform filter:** post-filtering with oversampling is fine
  - **Mildly correlated:** in-filtering with ACORN-style denser neighborhoods
  - **Highly correlated (concentrated):** pre-filtering (filter first, then HNSW on the surviving set), accepting the linear scan cost on the predicate

---

## 7. Worked Example (Reference)

This is the worked example from the research, transcribed as the reference for end-to-end correctness.

### The Query

```sql
SELECT a.name, b.content, distance(b.embedding, :query) AS dist
FROM GRAPH_TABLE (
    network_graph
    MATCH (a IS host) -[e IS connection]->{1,3} (b IS server)
    WHERE a.status = 'COMPROMISED' AND e.port = 443
    COLUMNS (a.name AS name, b.content AS content, b.embedding AS embedding)
) AS gt
ORDER BY distance(gt.embedding, :query) ASC
FETCH APPROX FIRST 10 ROWS ONLY WITH TARGET ACCURACY 90 PERCENT;
```

### Algebraic Form

```
⊕_{d, q}^{10} { π_{a.name, b.content, dist}(b)
              | a ← V, b ← V, e ← E,
                labels(a, host) ∧ labels(b, server) ∧
                a.status = 'COMP' ∧ e.port = 443 ∧
                π ∈ WALK_{1,3}(a, b) }
```

The `⊕_{d, q}^{10}` is the bounded similarity monoid merge for k=10 with the L2 (or chosen) distance metric to query vector `q`.

### Two Candidate Plan Shapes

**Plan 1: Graph-First, Post-Filter (Top-K After)**

1. B-tree lookup on `a.status = 'COMPROMISED'`
2. BFS from filtered hosts up to 3 hops
3. Filter edges on `e.port = 443`
4. Collect `b.embedding` vectors
5. Brute-force distance computation: `C_dist × N_valid × D`
6. Top-10 by distance

**Plan 2: Vector-First, Pre-Filter**

1. HNSW probe for top-k candidates by distance to `q` (with oversampling: k × E[fanout])
2. Reverse graph traversal: for each candidate `b`, check if a path of length 1–3 exists from a `COMPROMISED` host via a `port = 443` edge

### Applicable Equivalence Law

`topk(k, ⋈_E (σ_p(A), B)) ≈ ⋈_E (σ_p(A), topk(k × E[fanout], B))` (4.2)

Plan 2 is valid only if the rank-join monotonicity condition holds: a better rank on `B` corresponds to a better rank on the joined result.

### Denotational Semantics

The WALK restrictor at length `{1,3}` produces a multiset of valid path sequences. The COLUMNS projection collapses these to relational tuples. The ORDER BY sorts by distance with secondary tie-break by `b.id`. The FETCH APPROX constrains the result to a subset `S_approx` where `E[|S_approx ∩ S_exact| / 10] ≥ 0.90`.

### Type Judgment

```
Γ ⊢ a : { status: TEXT, name: TEXT | ρ_a }
Γ ⊢ b : { content: TEXT, embedding: VECTOR(N) | ρ_b }
Γ ⊢ e : { port: INT | ρ_e }
Γ ⊢ GRAPH_TABLE(...) : { name: TEXT, content: TEXT, dist: FLOAT }
```

### Cost Comparison

| Plan | Cost Formula | When It Wins |
|---|---|---|
| **Plan 1 (Graph-first)** | `C_btree × 1 + C_bfs × |V_host| × 0.01 + C_dist × N_valid × D` | When `N_valid` (filtered survivors) is small (low hundreds) |
| **Plan 2 (Vector-first)** | `C_hnsw × 10 × E[fanout] × ef × log N × D + C_reverse_graph × 10 × E[fanout] × log \|V\|` | When `N_valid` is large (thousands+) and the graph traversal is expensive |

**With 500 valid survivors and 384-dim vectors, D=384:**
- Plan 1: `~ 500 × 384 = 192,000` distance operations, plus the BFS
- Plan 2: `~ 10 × 5 × ef × log N × 384` HNSW operations, plus `50` reverse graph traversals

At small `N_valid`, Plan 1 wins on simplicity (no HNSW random I/O, pure sequential distance). At large `N_valid`, Plan 2 wins because the HNSW is faster than brute force at high `N`.

**This crossover is the ECQO optimization target.** The optimizer's job is to compute the crossover for the current data distribution and pick the plan below it.

---

## 8. Mapping to the Build Plan

Each formal concept maps to a specific phase of the 12-phase build order from `unified-query-superset-architecture-08-01-2026.md`. This table is the bridge between the math and the code.

| Build Phase | Formal Foundation Used | What to Implement |
|---|---|---|
| **1. Extract `xDarkicex/lexer`** | — | SWAR core extraction. Pure mechanical work. |
| **2. Catalog** | §3 Type System | Tables, columns, indexes, vector dimensions, property graph definitions, type lattice |
| **3. B-tree / ART** | §5 Cost Model (B-tree row) | Implementation; no formal work needed yet |
| **4. SQL/PGQ Lexer + Parser** | §1.2 Four Monoids, §3 Type System | Token kinds, AST nodes, type checker for scalars + VECTOR(N) + graph types |
| **5. Relational Plan + Executor** | §1.2 Set/Bag Monoids, §4 Standard Relational Laws, §5 Cost Model | Plan nodes for `Scan`, `Filter`, `Project`, `Join`, `Aggregate`. Executor for set/bag monoids. |
| **6. pgvector Operators + `VectorKNN`** | §1.3 Bounded Similarity Monoid, §2.4 Tie-Breaking, §2.5 Approximation Semantics | Vector distance operators, HNSW-backed top-k, tie-breaking by record ID, `FETCH APPROX` Recall@k constraint |
| **7. GPM in Parser + `GraphExpand`** | §2.1-2.3 Property Graph, Path Restrictors | Path algebra, restrictor implementations, `GRAPH_TABLE` executor, choice function for ANY SHORTEST |
| **8. FVS Strategy Selection** | §4.1 Top-K Pushdown, §6 Error Composition | Optimizer rules for pre/post/in-filter selection. Requires `θ_GLS` implementation. |
| **9. ECQO Cardinality Estimation** | §5.3 ECQO, §5.4 Threshold Calibration | Localized ANN probe for σ_vec, histogram for σ_rel, degree distribution for σ_graph, self-tuning thresholds |
| **10. Factorized Processing** | §1.2 List Monoid, §1.3 Similarity Monoid (f-representation) | FactorizedGroup, f-representation memory layout, factorized hash join |
| **11. iFVS: Filter-Aware PQ** | §4.1 `θ_GLS` metric, plus the prior iFVS research | Per-query arena, dynamic codebook generation, weight training pipeline |
| **12. ACID/MVCC** | *(deferred research)* | Not yet in this document. See open questions §9. |

### Critical Path

The build cannot start without resolving the following open questions (see §9):
- `θ_GLS` metric computation (gates Phase 8)
- ANY SHORTEST choice function (gates Phase 7)
- VECTOR(N) implementation choice (gates Phase 4 type checker)

The algebra, type system, equivalence laws, and cost model are otherwise concrete enough to start Phases 2, 4, and 5 immediately.

---

## 9. Open Questions and Deferred Research

These are the research items that block specific build phases.

### Blocking

1. **`θ_GLS` (Global-Local Selectivity) metric computation** (gates Phase 8)
   - The FVS law `topk(k, σ_p(R), d) ≈ σ_p(topk(k + c(sel(p), θ_GLS), R, d))` requires a computable `θ_GLS`.
   - The literature defines it informally as "the correlation between global and local selectivity" but does not give a closed form.
   - **Required:** a closed-form computation over the predicate and the vector distribution. Candidates: a learned predictor, a sampling-based estimator, an analytical formula based on the predicate's spatial concentration.

2. **ANY SHORTEST choice function** (gates Phase 7)
   - ANY SHORTEST is non-deterministic; the implementation must make a choice.
   - Standard: lex-smallest sequence of edge/vertex IDs.
   - **Required:** explicit specification of the choice function for reproducible query results.

3. **VECTOR(N) type implementation choice** (gates Phase 4)
   - Three options: dependent type, refinement type, runtime check.
   - **Required:** decision based on the type checker's complexity budget and the SWAR lexer's constraints.

### High Priority (not strictly blocking)

4. **ECQO bias correction (`C_bias`)** (gates Phase 9)
   - The localized ANN probe is biased toward HNSW hub nodes.
   - The bias correction factor is named but never defined.
   - **Required:** a bias model (degree-based, layer-based, or stratified sampling).

5. **Closed-form cost comparison for plan selection** (gates Phase 8)
   - The worked example asserts Plan 1 vs Plan 2 but doesn't derive the crossover.
   - **Required:** explicit cost formulas with the crossover as a function of the cardinalities.

6. **Expressive power of the unified algebra** (theoretical)
   - Standard relational algebra = first-order logic. Standard graph reachability requires transitive closure (Datalog).
   - Adding top-k similarity operations pushes into metric space logic.
   - The complexity class of the unified algebra is an open problem.
   - **Required:** separate research pass if the theoretical boundary matters for the implementation.

### Deferred to Future Research

7. **Update operations against `GRAPH_TABLE` views**
   - View update translation is NP-hard in general; with vector properties and HNSW topology, the translation is non-deterministic.
   - The current algebra is read-only.
   - **Required:** separate research pass for the update path. Ties to the open question in the prior review (factorized state under writes).

8. **MVCC and snapshot semantics**
   - The algebra assumes a single database state. Under MVCC, the state is snapshot-scoped.
   - The cost model, the factorized state, and the iFVS codebook all need snapshot semantics.
   - **Required:** separate research pass for MVCC integration.

9. **SQL constraints on vector spaces**
   - `UNIQUE(embedding)` lacks topological utility due to floating-point drift.
   - A formal definition of spatial constraints (e.g., `∀x,y ∈ R, d(x,y) > ε`) requires novel validation algorithms.
   - **Required:** separate research pass if/when constraints are added to the type system.

10. **Conjunctive RPQ with vector predicates**
    - The current algebra extends RPQ with vector operations at the top level, but the interaction between RPQ path algebra and vector top-k inside the path is not formalized.
    - **Required:** separate research pass for in-path vector predicates.

---

## 10. Errata from the Research Chunk

Specific corrections to the source research that must be reflected in the implementation:

1. **Determinism labels (Path Restrictors table).** WALK, TRAIL, SIMPLE, ACYCLIC, ANY SHORTEST are non-deterministic. ALL SHORTEST, SHORTEST k GROUP are deterministic. The source has these swapped.

2. **Idempotence in the similarity monoid.** `σ_p σ_p = σ_p` is unsound in the similarity monoid. Duplicate filtering can change the top-k. Optimizer must restrict idempotence to set/bag monoids.

3. **HNSW cost formulation.** Use `O(ef × log N × D)`, not `O(log N × M × D)`. `ef` is the search-time parameter; `M` is the construction-time parameter and not in the per-query cost.

4. **Selection pushdown condition for GRAPH_TABLE.** The condition is "p is post-restrictor-evaluable" — `p`'s evaluation cannot affect the path set. The source's "monotonic with respect to the path restrictor" is approximately right but imprecise.

5. **K-NN join condition.** The precise property is **rank-join monotonicity** (Ilyas et al.), not "the join predicate does not prune the minimal elements."

6. **ANY SHORTEST choice function.** The implementation must specify how the choice is made. Default: lex-smallest sequence of edge/vertex IDs.

7. **Softplus numerical stability (from prior chunk).** The iFVS softplus formula `ln(1 + exp(w_z))` overflows for `w_z > 88`. Use the stable form `max(w_z, 0) + log(1 + exp(-|w_z|))`.

---

## 11. Recommended Reading Order for Contributors

For someone new to the project, the recommended order for understanding the foundations:

1. **§1.1-1.2 (Monoid Calculus + Four Monoids)** — the algebra
2. **§3.1-3.2 (Type Lattice + VECTOR(N))** — the type system
3. **§2.1-2.3 (GRAPH_TABLE + Path Restrictors)** — the graph semantics
4. **§4.1 (Top-K Pushdown / FVS Law)** — the most important equivalence law
5. **§6 (Error Composition)** — the in-filtering problem
6. **§5 (Cost Model)** — the optimizer's objective function
7. **§7 (Worked Example)** — the end-to-end reference
8. **§8 (Mapping to Build)** — the bridge to the code

Then read `unified-query-superset-architecture-08-01-2026.md` for the build order, and the engineering research (when it exists) for the Go-side implementation.

---

## 12. Summary

This document establishes the mathematical foundation for the unified query algebra. The key takeaways:

- **The algebra is monoid comprehensions** with four monoid types (Set, Bag, List, Similarity).
- **The type system** uses parameterized types for vectors and row polymorphism for graph types.
- **The equivalence laws** drive the optimizer. The FVS law (oversampling based on `θ_GLS`) is the most important.
- **The cost model** is unified across operators. ECQO is the cardinality estimation strategy.
- **The error composition** is multiplicative for independent errors, additive for uniform filters, and catastrophic for concentrated filters (navigational dead-ends).
- **Three open questions block the build:** `θ_GLS` computation, ANY SHORTEST choice function, VECTOR(N) implementation choice.

The foundation is concrete enough to start Phases 2, 4, and 5 of the build. Phases 7, 8, and 9 are blocked on the open questions.
