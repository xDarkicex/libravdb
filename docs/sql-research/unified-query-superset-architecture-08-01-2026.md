# Unified Query Superset: SQL/PGQ + pgvector + Graph in a Single Embedded Engine

**Date:** 2026-08-01
**Status:** Architecture research / pre-design synthesis
**Author:** Synthesized from internal discussion and external research

---

## Context

This document captures the architectural framing for adding a unified query layer to libraVDB: a single SQL/PGQ + pgvector + graph dialect that compiles relational, topological, and semantic queries into one factorized plan tree. The product is not "an SQL engine for a vector database" — it is an embedded, ACID-transactional hybrid database with a SWAR-driven lexer that no competitor can match on parse latency.

The work has not started. This is the conceptual frame the build will sit inside.

---

## The Starting Insight: SWAR Lexer Tech Is Reusable

pupate's SWAR byte classifier — 8 bytes per cycle via bitwise ops on `uint64`, zero-alloc, no per-byte branches — is language-agnostic. It currently lives at `pupate/internal/lexer/swar.go` and exposes:

- `FindByte(src, start, end, b) uint32`
- `FindByteNot(src, start, end, b) uint32`
- `SkipWS(src, pos, end) uint32`
- `FindAnyByte6(src, start, end, sig [6]byte) uint32`
- `FindAnyByte8(src, start, end, sig [8]byte) uint32`

None of it is markdown-specific. The markdown layer is in `pupate/internal/lexer/lexer.go` and the parser, not in the SWAR core.

The work to extract it to a shared `xDarkicex/lexer` module is mechanically trivial: `cp`, package rename, new `go.mod`, update import paths in pupate. ~10 minutes. This is the front door of the unified engine, and it ships with a property no competitor can match: predictable, zero-alloc parse latency in the nanosecond range.

---

## The Category: Unified Query Paradigm

The database industry is converging on a **syntactic sugar superset** pattern: a unified query grammar that bridges relational, graph, and vector paradigms within a single AST. Standards are settled. The field has matured enough that the playbook is visible. Multiple vendors have shipped or are shipping implementations.

### Field Map

| Player | Shape | Weakness |
|---|---|---|
| **Oracle Database 23ai** | Converged monolith. SQL/PGQ + VECTOR_DISTANCE + native graph | Not embedded. Heavyweight. Oracle. |
| **Neo4j** | Native graph + Cypher, partial SQL extensions | JVM. Separate server. GQL and SQL not unified. |
| **Kuzu** | Embedded graph + vector extension, DuckDB-style | No ACID. Cypher, not SQL/PGQ. No first-class SQL surface. |
| **Lance-Graph** | DataFusion bridge over Lance columnar format | No transactions. File-format bound. |
| **Google Spanner Graph** | SQL/PGQ + GQL dual support | Google-only. |
| **Kinetica** | Vector-as-edge dynamic graph inference | Cloud-scale only. |
| **DuckDB** | Columnar relational analytics | No first-class graph or vector. |

**libraVDB's unclaimed slot:** *the embedded hybrid with ACID, graph, vector, and SQL/PGQ in a single Go binary.* No competitor has all five. Some have four. None have all four plus the single-binary embedded shape.

---

## The Standards (Adopt, Don't Invent)

### SQL/PGQ (ISO/IEC 9075-16:2023)

The chosen dialect. Compositional with SQL. Same GPM syntax as GQL.

- `GRAPH_TABLE(...)` operator in the standard `FROM` clause
- Graph Pattern Matching (GPM) language: `(a) -[e]-> (b)` ASCII-art edge patterns
- `IS` bindings: `MATCH (a IS Person) -[e IS Knows]-> (b IS Person)`
- `COLUMNS (a.name, b.age)` projection that flattens graph results into tabular form
- Path restrictors and selectors:
  - `WALK` — arbitrary traversal, nodes and edges repeatable
  - `TRAIL` — no edge repetition
  - `SIMPLE` — no node repetition
  - `ACYCLIC` — no cycles
  - `ANY SHORTEST` — one shortest path per partition
  - `ALL SHORTEST` — all shortest paths per partition
  - `SHORTEST k GROUP` — first k groups by endpoint partition
- Path quantifiers: `->{1,3}` (1–3 hops), `->+` (1+), `->*` (0+)
- Flattening: `ONE ROW PER [MATCH | VERTEX | STEP]`

### GQL (ISO/IEC 39075:2024)

Standalone graph language with the same GPM. **Not chosen.** GQL doesn't compose with the rest of SQL — it's its own language. The user requirement is a SQL superset, so PGQ is the only choice that satisfies it.

### pgvector operators

- `<->` L2 / Euclidean distance
- `<#>` negative inner product
- `<=>` cosine distance (`1 - cos_sim`)
- `HAMMING_DISTANCE` for binary quantization
- Function form `VECTOR_DISTANCE(a, b, METRIC)` for users who don't memorize operators

### Approximation control

The one piece of approximate control exposed at the language level:

```sql
FETCH APPROX FIRST k ROWS ONLY WITH TARGET ACCURACY n PERCENT
```

(Oracle 23ai syntax.) Lets the user tell the engine "I accept approximation, here's the accuracy budget." Everything else — pre-filter, post-filter, in-filter selection — is an optimizer decision, not a language feature.

---

## The Hard Problems (Named, with Literature)

The research names the hard problems so the literature is searchable. These are the engineering targets, not the standards themselves.

### 1. Filtered Vector Search (FVS)

When an exact SQL predicate or a graph pattern must be combined with vector similarity, three execution strategies exist:

- **Pre-filtering (filter-first).** Run the exact predicate or graph traversal first to reduce the candidate set, then run vector search on the reduced set. Best when filter selectivity is very high (>99% eliminated). Disastrous when selectivity is low because the linear scan cost dominates.
- **Post-filtering (ANN-first).** Run HNSW or IVF to top-k nearest neighbors, then apply the exact predicate. Prone to poor recall if the filter is restrictive — the top-k vector results may all fail the predicate.
- **In-filtering (single-stage).** Evaluate exact predicates dynamically during HNSW neighbor expansion. Balanced tradeoff but hardest to implement. Risks breaking graph connectivity.

The optimizer must pick the strategy at plan time based on selectivity estimates. **Exqutor** introduces Exact Cardinality Query Optimization (ECQO) for this.

### 2. HNSW Graph Integrity Under Filtering

Standard HNSW in-filtering removes nodes that fail the predicate from the graph. This breaks the small-world connectivity property and traps the search in local minima, destroying recall.

Researched solutions:

- **ACORN** — Denser vertex neighborhoods + traversal heuristics to restore connectivity. Memory and construction overhead.
- **NaviX** — Neighbor expansion strategies for filtered search.
- **JAG (Joint Attribute and Graph)** — Indexes jointly on vector similarity and attribute proximity to prevent navigational dead-ends.

### 3. Instance-Optimized Filtered Vector Search (iFVS)

Standard Product Quantization uses a static, filter-agnostic codebook. When combined with relational filters, precision penalties compound and recall drops.

iFVS generates a filter-aware PQ codebook dynamically based on both the query vector and the relational predicate. The SQL predicate is encoded as a filter-aware weight vector that re-weights vector dimensions by relevance. Each hybrid query gets its own codebook, perturbed from a base codebook using a shared memory bank and learned adjustment directions. Maintains compact PQ codes while vastly improving QPS and recall across selectivity bins.

### 4. Cardinality Estimation Across Three Paradigms

Traditional histograms (1024-bin range, heavy hitters) work for relational but fail for high-dimensional vectors and non-linear graph topologies.

ECQO approach:

- Localized ANN probe to estimate vector predicate selectivity with high accuracy
- Adaptive sampling for hybrid queries — sample size grows until desired confidence is reached
- Histograms for relational, edge counts for graph
- Synthesize into a single cost model

### 5. Factorized Processing (Graph Join Blowup)

At 3+ hop graph joins, the Cartesian product is intractable. Kuzu's **f-representation** maintains intermediate results in compressed "unflat" group states instead of materializing tuples. A group is either "flat" (single tuple) or "unflat" (compressed list of values). Critical for multi-hop queries where flat representation blows up memory.

### 6. Lowering Graph Plans Into Relational Algebra

The **Lance-Graph** pattern: parse Cypher-like queries, build a graph-shaped intermediate plan (start nodes → traverse → filter → project), then systematically lower into Apache DataFusion's logical plan of standard relational operators. Proves that graph syntax is expressible in standard relational operators without a separate execution engine.

### 7. HMGI (Hybrid Multimodal Graph Inference)

For GraphRAG and agentic AI workloads. Decoupled architectures bounce between a vector index and a graph database, transferring large node ID lists over the network. HMGI runs ANNS concurrently with graph traversal, anchored to entities and relationships in the graph. Achieves 3x QPS over decoupled baselines. Modality-aware partitioning of vector embeddings alongside graph topologies optimizes the index structure.

---

## libraVDB's Existing Foundation

These pieces are already in place in the codebase:

| Package | What it provides | Role in unified engine |
|---|---|---|
| `internal/storage/wal/` | Write-ahead log | Prerequisite for ACID/MVCC |
| `internal/storage/{fsdurability,singlefile,wal}/` | Page and segment layer | Table storage fits here |
| `internal/filter/` | Typed expression AST + parser (`parser.go`, `equality.go`, `range.go`, `logical.go`, `containment.go`) | **Seed of the SQL WHERE clause evaluator** |
| `internal/record/` | Row format (`record.go`, `delta.go`, `generation.go`) | The row format SQL operates on |
| `internal/index/flat/` | Flat vector index | Vector scan path |
| `internal/index/hnsw/` | HNSW index | The `VectorKNN` physical operator's backing |
| `internal/index/ivfpq/` | IVF + Product Quantization | The compression layer iFVS will modify |
| `libravdb.Graph` | Graph store | The `GraphExpand` physical operator's backing |
| `libravdb.Filter` | Metadata filter | Compiles into plan-level `Filter` nodes |

### What's Missing

- **B-tree index** — for relational WHERE / JOIN / ORDER BY (non-vector)
- **Catalog** — table/column/index definitions, vector dimensions, type metadata
- **SQL/PGQ parser** — the front door
- **Plan tree + optimizer** — the brain
- **Factorized executor** — the muscle
- **ACID / MVCC transaction layer** — over the existing WAL

The parser is small. The optimizer and the executor are where the engineering effort concentrates.

---

## Architecture Decision: Factorized Native Executor

Two patterns from the research:

### Option A — DataFusion bridge (Lance-Graph pattern)

Parse the unified grammar into a single AST. Lower graph + vector into relational algebra nodes (`Scan`, `Filter`, `HashJoin`, `Project`). Add specialized physical operators for `VectorKNN` and `GraphExpand` that plug into a relational pipeline. Reuse or build a small relational executor.

**Pros:** No separate execution engine. Reuses columnar technology.
**Cons:** Factorized processing for multi-hop graph joins is awkward to retrofit. The relational executor has to grow graph + vector operators anyway.

### Option B — Factorized native executor (Kuzu pattern)

Build a factorized query processor from day one. Graph joins compress to f-representations; the Cartesian product never materializes. Vector scans are columnar over the row store. SQL/PGQ is one more input to the factorized plan.

**Pros:** Graph joins don't blow up at 3+ hops. Clean separation of physical operators. Single binary, no DataFusion dependency.
**Cons:** You write the executor. ~10K lines per Kuzu's experience.

### Decision

**Option B.** libraVDB is already a Go binary with its own storage, HNSW, IVFPQ, and graph store. The DataFusion bridge is the right call if you don't have a storage engine. We do. Factorized processing is a *day-one* property, not a retrofit. A separate Rust/Arrow dependency in a Go binary is also a poor fit.

---

## Build Order

12 phases. Each is shippable as a coherent increment.

| # | Phase | Reference impl | Notes |
|---|---|---|---|
| 1 | Extract `xDarkicex/lexer` from `pupate/internal/lexer/swar.go` | — | Mechanical: `cp` + import path update. ~10 minutes. |
| 2 | Catalog (`internal/catalog/`) | Kuzu's catalog | Tables, columns, vector dims, indexes, edge counts, histograms |
| 3 | B-tree index (`internal/index/btree/`) | Kuzu's btree | `Seek`, `Range`, `Insert`. Same interface contract as HNSW/IVFPQ |
| 4 | SQL/PGQ lexer + parser (relational + PGQ + pgvector tokens, no execution yet) | — | Single AST. Validates against in-memory schema during development |
| 5 | Relational plan + executor (no graph, no vector) | — | First end-to-end SQL query works |
| 6 | pgvector operators in expressions, `VectorKNN` physical op | pgvector | `<->`, `<#>`, `<=>`, `VECTOR_DISTANCE` |
| 7 | GPM in parser, `GraphExpand` physical op | Kuzu, Spanner Graph | Path restrictors, `ONE ROW PER MATCH/VERTEX/STEP` |
| 8 | FVS strategy selection: pre-filter / post-filter / in-filter | ECQO | Optimizer-driven, cardinality-aware |
| 9 | ECQO-style cardinality estimation | Exqutor | Localized ANN probe + relational histograms + graph edge counts |
| 10 | Factorized processing for graph joins | Kuzu f-representation | Critical for 3+ hop queries |
| 11 | iFVS: filter-aware PQ codebooks | iFVS paper | Modifies `internal/index/ivfpq/` |
| 12 | ACID / MVCC over the existing WAL | — | Last, depends on the WAL being correct |

Phases 1–5 are the foundation and yield "SQL over libraVDB." Phases 6–7 yield "graph + vector in the same plan." Phases 8–10 are "FVS done right." Phases 11–12 are the research frontiers where the implementation choices are still being made by everyone.

---

## The SWAR Tech's Specific Role

The SWAR lexer is not the product. But it's the part nobody else can copy.

| Competitive property | Oracle 23ai | Neo4j | Kuzu | Lance-Graph | **libraVDB (proposed)** |
|---|---|---|---|---|---|
| SQL/PGQ | ✓ | partial | ✗ | partial | **✓ (planned)** |
| Vector operators | ✓ | ✗ | extension | ✓ (Lance format) | **✓ (planned)** |
| Graph | ✓ (GRAPH_TABLE) | ✓ (native) | ✓ (native) | ✓ (over Lance) | **✓ (existing)** |
| ACID | ✓ | recent | ✗ | ✗ | **✓ (WAL in place)** |
| Embedded / in-process | ✗ | ✗ | ✓ | partial | **✓ (existing)** |
| Single binary | partial | ✗ (JVM) | ✓ (C++) | ✗ (Rust + DataFusion) | **✓ (Go)** |
| Zero-alloc parse | ✗ (JVM) | ✗ (JVM) | ✗ (allocates) | ✗ | **✓ (SWAR)** |
| Predictable p99 parse | ✗ | ✗ | ✗ | ✗ | **✓ (SWAR)** |

The bottom two rows are the SWAR wedge. They don't make the engine faster end-to-end — Oracle 23ai will out-perform on full query latency because it has decades of optimization work behind it. They make the engine's *parse cost* a product feature rather than a budget line. At 10K QPS on mixed workloads, the absence of GC pauses and the predictable parse cost become visible in p99 telemetry. None of the competitors can claim that.

The LDBC SNB numbers from the research (Kuzu q1=2.3ms, q4=1.0ms) are full-query latencies. The parse step is a small fraction. SWAR's advantage is in the small fraction, and in the variance of that fraction. That's a real but narrow wedge.

The big wedge is the unclaimed combination: embedded + ACID + graph + vector + SQL/PGQ in a single Go binary.

---

## What This Research Says We Should NOT Do

- **Don't invent syntax.** SQL/PGQ is the standard. Path restrictors, edge patterns, `IS` bindings, `COLUMNS` projection are all defined. Inventing syntax is a tax paid forever.
- **Don't expose filter strategy in the language.** Pre-filter / post-filter / in-filter is an optimizer decision. The single exception is `FETCH APPROX ... WITH TARGET ACCURACY`, which gives the user an accuracy budget.
- **Don't skip factorized processing.** Without it, 3-hop graph joins materialize the Cartesian product and exhaust memory.
- **Don't skip the WAL or the catalog.** Both are prerequisites for the optimizer to be correct. The optimizer needs cardinality estimates. Cardinality estimates need the catalog. The catalog needs the WAL to persist.
- **Don't try to be Oracle 23ai.** Oracle is the converged monolith. libraVDB's wedge is the embedded, single-binary, Go-native combination with zero-alloc parse. Different battlefield.
- **Don't separate the ASTs.** Graph, vector, and relational must be first-class nodes in one AST. Separate parsers for each paradigm force a multi-pass system that can't optimize across boundaries.
- **Don't expose the indexes directly.** Users write `WHERE x = 1`, not `USING BTREE(x)`. Index selection is the optimizer's job. The exception: `CREATE INDEX` is a DDL surface, but `USING` clauses should default to the optimizer's choice and only be overridable for benchmarking.

---

## Open Questions

These need answers before the catalog design:

1. **Vector dimensionality at table-creation time or per-row?** Per-row is more flexible but blows up storage. Per-column is simpler. pgvector does per-column. We should do per-column.
2. **SQL/PGQ path restrictors — full set or subset?** All seven restrictors is a lot. The minimum useful set is `WALK`, `TRAIL`, `SIMPLE`, `ANY SHORTEST`. Others can land incrementally.
3. **One ROW PER flattening — required for v1?** Oracle 23ai's `ONE ROW PER MATCH/VERTEX/STEP` is the difference between "you can do 3-hop queries" and "you can flatten the results into a useful form." Required for v1.
4. **Graph as separate model or view over tables?** SQL/PGQ uses view-like property graphs over relational tables. This is the standard and it composes with transactions naturally. Adopt it.
5. **What dialect quirks to accept?** pgvector is the de-facto vector standard. Cypher is the de-facto graph syntax. But we chose PGQ over Cypher. The user-facing ergonomics question is: do we also accept Cypher-style `MATCH` patterns in addition to PGQ-style `GRAPH_TABLE` patterns? One parser or two? Recommend: one parser, PGQ only. If a future product wants Cypher compatibility, it's a translation layer, not a parallel parser.
6. **Transaction isolation level?** Read-committed is the SQL default. Snapshot isolation is what makes MVCC pay off. Pick snapshot isolation from day one; it's strictly more useful than read-committed.

---

## Next Steps (If Proceeding)

1. **Catalog design.** Schema representation in code. How tables reference columns, indexes, and graph property definitions. How vector dimensions are stored. How edge counts are maintained. This is gating for the parser, the optimizer, and the B-tree.
2. **B-tree interface.** The `internal/index/btree/` package needs to match the contract of the existing vector indexes (`Seek`, `Range`, `Insert`). Get this right and the SQL planner has a uniform index abstraction.
3. **SWAR extraction.** `cp` `pupate/internal/lexer/swar.go` to `xDarkicex/lexer/swar.go`, rename package, write `go.mod`, update pupate's import paths. ~10 minutes. Not gating for the design work but the first concrete artifact.

The next concrete deliverable is the catalog design document. The B-tree interface follows from it. The parser follows from both. The optimizer follows from the parser, the catalog, and the B-tree. The executor follows from the optimizer and the factorized processing design.

---

## What This Document Is

A pre-design research synthesis. It does not commit to specific code structure, file paths, or implementation choices beyond what's already present. It frames the problem, names the standards and the research, identifies the engineering targets with their literature, and orders the work.

It is not a marketing document. It is not a tutorial. It is the reference frame the build sits inside.
