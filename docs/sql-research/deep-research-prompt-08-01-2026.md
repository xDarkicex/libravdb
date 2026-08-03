# Deep Research Prompt: Gating Decisions for the Unified SQL/PGQ + Vector + Graph Engine

**Date:** 2026-08-01
**Status:** Research prompt — for next research pass
**Preread:**
- `unified-sql-engine-vision-08-01-2026.md` (existing vision doc)
- `unified-query-superset-architecture-08-01-2026.md` (architecture synthesis)

---

## What This Document Is

The two existing docs in this directory establish the *what* (unified SQL/PGQ + pgvector + graph) and the *how* at a high level (SWAR lexer, factorized executor, in-filtering). They contain several **unverified claims** and **gating decisions** that need targeted research before implementation begins.

This prompt identifies the research questions, ranks them by impact, and frames each as a specific answerable question with sources to consult and criteria for an actionable answer.

The goal is not to survey the literature. It is to resolve the decisions that block the next concrete deliverable (catalog design, then parser, then B-tree/ART).

---

## Tier 1 — Gating Decisions (Block the Build)

These must be answered before any code lands. Wrong answers here invalidate downstream work.

---

### Q1.1 — ART vs B-tree for the Relational Index

**Claim in vision doc:** "Evolve `nanite`'s `[256]*RadixNode` tree into a full Adaptive Radix Tree (ART) with `Node4`, `Node16`, `Node48`, `Node256` using `uint64` offsets into `xDarkicex/memory` arenas."

**Why this needs research:**
- B-trees dominate production databases (PostgreSQL, MySQL/InnoDB, SQLite, DuckDB sort-of) because they are cache-friendly for sequential range scans and I/O-aligned for disk-based storage.
- ART (Leis et al., 2013) is provably faster for in-memory point lookups and small range scans, with adaptive node sizes that compress keys well.
- The workload is *not* purely point lookups. `WHERE price BETWEEN 10 AND 100`, `ORDER BY timestamp LIMIT 10`, `GROUP BY category` are range/sort operations that B-trees do well and ART does worse.
- "Off-heap uint64 offsets" changes the cost model. Pointer-chasing ART (cache misses on every node) vs offset-chasing ART (single base register, predictable stride) have very different cache profiles.

**Specific questions to answer:**

1. What is the measured performance of ART vs B-tree on a workload that mixes (a) point lookups by primary key, (b) range scans over 1–10% of rows, (c) ORDER BY with LIMIT, (d) GROUP BY on a high-cardinality column? Cite Hyrise's published numbers, Umbra's, HyPer's, or any other in-memory database that has compared them on equivalent hardware.

2. What is the ART implementation cost in Go? The original ART paper assumes 8-byte pointer chasing. The Go adaptation has to use `unsafe.Pointer` arithmetic and avoid GC. How does the `uint64` offset trick work with the existing `xDarkicex/memory` arena allocator? Is there a published Go ART implementation (e.g., `github.com/znly/stringsutil` has a radix tree, but is it ART?)?

3. What is the disk persistence story for ART? B-trees serialize naturally to pages. ART's adaptive node sizes make serialization a custom problem. Does the WAL suffice, or is a separate on-disk format required?

4. **The `nanite` claim:** the vision doc references "the `[256]*RadixNode` tree architecture found in the `nanite` HTTP router." Is that actually a radix tree, or is it a trie? Does it support variable-length keys? Can it be evolved into ART, or is it a fundamentally different structure? (Requires reading the `nanite` source code in `../xDarkicex/nanite/`.)

**Actionable answer format:**
- A recommendation: ART, B-tree, or hybrid (ART for indexes on small-cardinality columns, B-tree for everything else).
- A reference implementation to follow.
- A clear statement of the workload mix the recommendation is optimized for.

**Sources to consult:**
- Leis, Boncz, Kemper: "The Adaptive Radix Tree: ARTful Indexing for Main-Memory Databases" (2013)
- Hyrise (Hasso Plattner): ART implementation in C++
- Umbra: B-tree implementation in C++
- HyPer: ART in C++
- DuckDB's ART adaptation
- `github.com/znly/stringsutil` and similar Go radix trees
- The `nanite` source at `../xDarkicex/nanite/`

---

### Q1.2 — pgwire Implementation Strategy and Scope

**Claim in vision doc:** "Build the `pgwire` TCP listener from scratch with zero-copy buffers into `xDarkicex/memory` arenas, passing `unsafe.Slice` structures directly to the SWAR lexer."

**Why this needs research:**
- pgwire is the PostgreSQL Frontend/Backend Protocol v3. It is a large, sprawling protocol with many optional features. Building it from scratch is a multi-month commitment.
- Zero-copy into arena memory has a real interaction with PostgreSQL's protocol model. Postgres servers typically format response rows into text/binary wire format in-place. If we want zero-copy, every type encoder has to be aware of the arena.
- "ORMs remain entirely unaware" is the goal, but ORMs vary in how much of the protocol they use. psql, pgx, JDBC, psycopg, sqlalchemy, GORM, Bun, sqlx all have different expectations.
- Auth is part of pgwire. Skipping auth is reasonable for embedded use but breaks the "speaks pgwire" claim for any non-embedded deployment.

**Specific questions to answer:**

1. What is the **minimum viable pgwire surface** for ORM compatibility? Specifically:
   - Simple Query protocol (one-shot text queries) vs Extended Query protocol (prepared statements, parameter binding, describe)
   - Authentication: SASL, MD5, SCRAM-SHA-256, none?
   - ErrorResponse format (Postgres has a specific 5-field error code: severity, code, message, detail, hint)
   - DataRow format: text format vs binary format
   - RowDescription with OID mapping for types (oid 25 = TEXT, 23 = INT4, etc.)
   - Portal/bind/execute lifecycle
   - COPY (almost certainly skip)
   - LISTEN/NOTIFY (skip)
   - Cursors (skip?)
   - Transactions (BEGIN/COMMIT/ROLLBACK as SQL, not protocol)

2. What is the **zero-copy story** for pgwire? Specifically:
   - Can response bytes be written directly from arena offsets, or does the protocol require framing (length prefix, type tags) that forces at least one copy?
   - How does CockroachDB handle zero-copy in their pgwire (they don't — they marshal into `[]byte`)?
   - How does Vitess's `vttablet` handle it (they don't — they buffer)?

3. What are existing **Go pgwire implementations** we can study or fork?
   - `github.com/jackc/pgx` (uses libpq, not pure Go)
   - `github.com/cockroachdb/cockroach`'s SQL listener
   - `github.com/cockroachdb/pebble`'s test infrastructure
   - `github.com/erikgrinaker/pgwire` (pure Go wire format library)
   - `github.com/electric-sql/pgwire` (ElectricSQL)
   - `github.com/cloudquery/pgwire`
   - `github.com/kiwifarm/bass` (Bass interpreter with pgwire)
   - Any others that have done a from-scratch implementation

4. What **type OIDs** does the engine need to advertise? The vision doc doesn't address this. pgvector uses custom OIDs (typically 3900, 3901, 3902 for FLOAT[], FLOAT8, etc.). How do we register those in the wire protocol without colliding with the standard OID space?

**Actionable answer format:**
- A scope statement: which pgwire features are in v1, deferred, or skipped
- A reference implementation to study
- An answer to "can zero-copy be maintained end-to-end through pgwire, or is one copy at the protocol framing layer inevitable"
- A type OID table for the v1 type system

**Sources to consult:**
- PostgreSQL Frontend/Backend Protocol v3 specification (postgresql.org/docs/current/protocol.html)
- pgvector's OID assignments
- CockroachDB's SQL listener source
- The Go libraries listed above

---

### Q1.3 — In-Filtering HNSW: Solving the Navigational Dead-Ends Problem

**Claim in vision doc:** "LibraVDB naturally supports the academic holy grail: In-Filtering (Single-Stage Execution). Because LibraVDB owns the HNSW graph topology and the metadata schemas in the same binary, the engine can inject relational SQL predicates and property graph constraints directly into the HNSW neighbor expansion loop."

**Why this needs research:**
- This claim is **too strong** as stated. The navigational dead-ends problem is real and not solved by "owning both schemas in the same binary."
- The ACORN paper (2024) explicitly shows that naive in-filtering on HNSW destroys recall. ACORN's fix is denser neighborhoods + traversal heuristics, which costs memory and construction time.
- NaviX (2024) shows that neighbor expansion strategies matter.
- JAG (2024) shows that pre-built joint indexes are necessary at low selectivity.
- The vision doc's claim that in-filtering is "naturally supported" because of the unified binary is wrong. Owning both schemas lets you *implement* in-filtering cheaply (no cross-process boundary), but it doesn't solve the algorithmic problem.

**Specific questions to answer:**

1. What is the **recall degradation curve** for naive in-filtering HNSW as filter selectivity drops from 100% (no filter) to 0.1% (highly selective filter)? Cite ACORN, NaviX, JAG benchmarks.

2. Is **ACORN's approach** implementable inside the existing `libravdb/internal/index/hnsw/` package without rebuilding HNSW from scratch? Specifically:
   - Does the current HNSW implementation expose the neighbor expansion phase as a hook?
   - Can ACORN's denser neighborhoods be added by modifying the construction algorithm, or does it require a new graph type?
   - What is the memory overhead per node for ACORN-style denser neighborhoods?

3. Is **JAG's joint indexing** feasible? JAG builds the index jointly on vector similarity and attribute proximity. Does the existing HNSW construction algorithm admit a joint cost function, or does it have to be replaced?

4. What is the **minimum viable in-filtering implementation** that doesn't destroy recall? Specifically:
   - Use HNSW with post-filtering as the baseline.
   - Add ACORN-style denser neighborhoods as a v2 enhancement.
   - Add JAG as a v3 enhancement if needed.
   - The v1 strategy should be: pre-filtering when filter selectivity > 99%, post-filtering otherwise, in-filtering only when both fail to meet latency budget.

5. The vision doc conflates "we can do in-filtering" with "in-filtering is the right strategy always." The research consensus is that **no single strategy is always right** — the optimizer must choose. The question is: at what selectivity does each strategy win, on real workloads?

**Actionable answer format:**
- A corrected claim that in-filtering is one of three strategies, not the answer
- A reference implementation (ACORN, NaviX, JAG) and a path to integrate it into the existing HNSW package
- A selectivity-vs-strategy decision table
- A v1 implementation plan: pre-filter / post-filter / in-filter with explicit cost model

**Sources to consult:**
- ACORN paper (2024): https://arxiv.org/abs/2403.04871
- NaviX paper
- JAG paper: https://arxiv.org/abs/2404.05544
- iFVS paper: https://arxiv.org/abs/2403.11735
- The existing `libravdb/internal/index/hnsw/` source

---

### Q1.4 — Factorized Processing: What Does the Storage Layer Actually Need?

**Claim in vision doc:** "LibraVDB's v1.2.0 Graph Layer already utilizes 16-byte fixed edges and EdgeTable 4KB pages with inline-first-8 layout. By mapping the SoA AST directly to this off-heap structure, LibraVDB avoids materializing intermediate strings and redundant tuple data, mimicking the exact factorized execution required by state-of-the-art academic implementations."

**Why this needs research:**
- Factorized processing is not just about edge layout. It is a *query processor* property: the engine keeps intermediate results in a compressed "unflat" representation across operators, only flattening at the final output.
- Kuzu's f-representation is the canonical implementation. It works at the *operator* level (HashJoin produces factorized output for the next operator), not just at the storage level.
- The claim "mimicking the exact factorized execution" is unsubstantiated. Page layout is one piece; the operator-level factorization is the actual implementation.
- The "SoA AST directly to off-heap structure" claim needs scrutiny. SoA AST means all node kinds in separate slices, indexed by node ID. Mapping this to a 16-byte fixed edge layout is plausible for graph nodes, but the relational side has variable-length fields (TEXT, VECTOR(N)).

**Specific questions to answer:**

1. What is **factorized query processing** in detail, and what does it require at each layer of the database?
   - The operator layer: how `HashJoin`, `Filter`, `Project`, `Aggregate` produce and consume factorized representations
   - The data structure layer: f-representations as arrays of pointers-to-arrays, vs other schemes
   - The storage layer: how f-representations interact with page-based storage
   - The wire layer: how f-representations are flattened only at the boundary

2. What is the **minimum operator set** that has to be factorized for the 3-hop graph join to stay bounded?
   - `GraphExpand` (variable-length path traversal)
   - `HashJoin` (combining graph results with relational scans)
   - `Filter` (predicate pushdown into the factorized representation)
   - Does `Project` need to be factorized? (Probably not — it doesn't increase cardinality)

3. Is the **16-byte fixed edge layout** compatible with factorized processing, or does it need modification?
   - 16-byte fixed edges can be factorized trivially (each edge is a unit)
   - The 4KB EdgeTable pages are natural unit-of-I/O
   - The inline-first-8 layout may or may not be friendly to operator-level factorization (needs verification by reading the existing code)

4. How does **factorized processing interact with off-heap storage**?
   - Factorized representations are typically pointer-based (pointers to vectors of pointers)
   - Off-heap requires uint64 offsets
   - The translation is mechanical but invasive

5. What is the **expected memory profile** of factorized vs flat execution for a 3-hop query over a graph with 1M nodes per hop?
   - Flat: 10^18 tuples materialized (intractable)
   - Factorized: ~3 × N vector groups, each holding the matched node IDs from one hop

**Actionable answer format:**
- A clear statement of what factorized processing requires at each layer
- A gap analysis against the existing v1.2.0 graph layer
- A list of operators that need factorized implementations
- An integration plan with off-heap storage

**Sources to consult:**
- Kuzu's f-representation paper and source: https://github.com/kuzudb/kuzu
- "Factorized Query Processing" (Zukowski et al., VLDB 2007, the original)
- "A Sample of Completion Time" (Boncz et al., various)
- The existing `libravdb/internal/graph/` source
- The existing `libravdb/internal/record/` source

---

## Tier 2 — Significant Engineering Decisions

These are not gating but shape the implementation significantly.

---

### Q2.1 — SoA AST: Prior Art, Emission Patterns, Visitor Patterns

**Claim in vision doc:** "The parsed SQL/Graph AST nodes will be laid out in contiguous, typed slices."

**Why this needs research:**
- SoA for ASTs is unusual. Most compilers/parsers use tree-of-pointers (AoS) because ASTs are inherently tree-structured.
- SoA works for ASTs only if the structure is reduced to a "nodes table + parent/child indices" representation.
- This is similar to a "columnar AST" or "relational AST." Apache Calcite's `RelNode` tree is a similar shape but uses Java objects.
- Visitor patterns over SoA data structures are not standard. Need a pattern.

**Specific questions to answer:**

1. Has anyone built a SoA AST in production? Specifically:
   - LLVM IR is essentially SoA (each basic block has instructions in a vector, with successor/predecessor indices)
   - Cranelift's IR is SoA
   - These are not SQL ASTs but they prove the pattern is viable

2. What is the **emission pattern** from a SWAR lexer into a SoA AST?
   - The lexer produces tokens. The parser consumes tokens and produces AST nodes.
   - With SoA, "producing a node" means appending to multiple slices (kind, start, end, child indices, etc.)
   - The parser's stack can be SoA too (a parse stack as a slice)

3. How do **tree traversals** work on SoA data?
   - Pre-order traversal: iterate slice by node ID, push/pop parent IDs
   - Visitor pattern: each visitor is a function that takes a node ID and reads from the slices
   - No recursion, no pointer chasing, cache-friendly

4. How does the **planner consume** the SoA AST?
   - The plan tree is itself SoA?
   - Or does the planner copy from AST slices into plan slices?
   - If both are SoA, the transformation is a series of indexed slice operations

5. What is the **GC pressure** of SoA AST?
   - The slices are large and append-only
   - With pre-allocated capacity, no allocations on the hot path
   - With the `xDarkicex/memory` arena, the slices live in arena memory and the GC never sees them

**Actionable answer format:**
- A reference implementation pattern (LLVM IR, Cranelift IR, or similar)
- An emission pattern for SWAR → SoA AST
- A visitor pattern for SoA AST
- A prototype plan tree in SoA

**Sources to consult:**
- LLVM IR Programmer's Manual
- Cranelift IR documentation
- Calcite's `RelNode` Java implementation
- "Columnar AST" papers (if any)

---

### Q2.2 — "C/Assembly Pushdown" — What Does This Mean?

**Claim in vision doc:** "The SQL Query Planner natively parallelizes the workload across shards at the C/Assembly layer."

**Why this needs research:**
- This sentence is ambiguous. "C/Assembly layer" could mean:
  - (a) SIMD/AVX-512 acceleration of distance functions and bitmap operations
  - (b) Hand-written C for hot operators, called via cgo
  - (c) Hand-written Go assembly stubs for inner loops
  - (d) Custom ABI / calling convention for in-process sharding
  - (e) Something else entirely
- In Go, each of these has different tradeoffs:
  - (a) SIMD: pure Go with `unsafe` slice arithmetic, no cgo cost
  - (b) cgo: GC interaction issues, FFI overhead, build complexity
  - (c) Go assembly: platform-specific, hard to maintain, but no FFI overhead
  - (d) Custom ABI: not really possible in Go's runtime
- The architecture doc I wrote mentioned "SIMD instructions for evaluating the vector distance threshold against the properties of the traversed nodes" — that's option (a). Need to confirm this is what's meant.

**Specific questions to answer:**

1. What is the **specific workload** that benefits from C/Assembly? The candidates are:
   - Vector distance computation (L2, cosine, inner product) over arrays of 384–4096 floats
   - Bitmap operations for filter conjunction/disjunction
   - Hash computation for joins
   - Radix-tree traversal (if ART is chosen)
   - String comparison
   - HNSW graph traversal

2. For each candidate, is **pure Go + SIMD** (via `golang.org/x/sys/cpu` feature detection + `unsafe.Pointer` arithmetic) sufficient, or is cgo necessary? Cite existing Go SIMD libraries:
   - `github.com/zeebo/xxh3` (hash SIMD)
   - `github.com/klauspost/cpuid` (CPU feature detection)
   - `github.com/minio/simdjson-go` (SIMD JSON parser)
   - `github.com/golang/go` itself (use of SIMD in `crypto/sha256`, `math/bits`, etc.)

3. For the candidates where **cgo is needed**, what is the cgo overhead per call, and how does it interact with the zero-allocation requirement? The standard answer is: cgo calls prevent goroutine scheduling, which means a cgo-heavy hot path is a single-threaded bottleneck.

4. What is the **Go assembly** story for the hot path? The Go runtime uses assembly stubs for `runtime.memmove`, `runtime.memclr`, etc. Adding custom assembly for, e.g., vector distance, is possible but rarely done outside the standard library.

**Actionable answer format:**
- A clear interpretation of what "C/Assembly pushdown" means
- A list of hot-path operators that benefit
- A per-operator decision: pure Go SIMD, cgo, Go assembly, or "no acceleration"
- A benchmarking plan

**Sources to consult:**
- Go assembly documentation
- Go SIMD libraries listed above
- The Go runtime's assembly stubs in `runtime/`
- Existing pure-Go SIMD implementations of distance functions

---

### Q2.3 — ACID/MVCC Over the Existing WAL

**Implicit claim:** The existing `libravdb/internal/storage/wal/` is sufficient for ACID/MVCC.

**Why this needs research:**
- The architecture doc listed MVCC as Phase 12. The vision doc doesn't address it. The WAL is the prerequisite, but MVCC is a *separate* engineering effort.
- MVCC requires:
  - Versioning: each row has a creation transaction ID and a deletion transaction ID
  - Snapshot isolation: a transaction sees a consistent view of the database as of its start time
  - Garbage collection: old versions are cleaned up when no transaction can see them
  - Index maintenance: B-trees/ARTs/HNSW all have to handle version visibility
- Off-heap storage makes MVCC harder: the version chain pointers have to be uint64 offsets, not Go pointers.
- The existing WAL records byte-level changes. MVCC requires logical changes (row-level) to support snapshot isolation efficiently.

**Specific questions to answer:**

1. What is the **MVCC implementation strategy** that fits the existing WAL?
   - Append-only storage with version chains (CouchDB-style)
   - Time-travel storage with snapshot pages (SAP HANA-style)
   - Delta storage with base + delta pages (PostgreSQL-style)
   - Log-structured merge-tree with snapshots (RocksDB-style)

2. How do **secondary indexes (B-tree/ART, HNSW) handle MVCC visibility**?
   - PostgreSQL: index entries point to heap tuples; visibility check at read time
   - CockroachDB: index entries store the version timestamp
   - LMDB: read-only MVCC, single-writer
   - Each has different tradeoffs

3. What is the **transaction isolation level** to ship with v1?
   - Read uncommitted: useless, skip
   - Read committed: PostgreSQL default, simple to implement, but read-skew anomalies
   - Snapshot isolation: prevents most anomalies, requires MVCC
   - Serializable: requires SSI or 2PL, significant overhead
   - The vision doc mentions `LAG()`/`LEAD()` over `PARTITION BY session_id` for `libravdbd` — those require snapshot isolation at minimum

4. How does **MVCC interact with off-heap storage**?
   - Version chains: each row has a `next_version_offset` uint64
   - GC: arena-level page reclamation, or per-row reclamation
   - The arena allocator's lifecycle has to match the MVCC's lifecycle

5. What is the **concurrency model** for SQL transactions?
   - One writer at a time (LMDB-style) — simple, high latency under contention
   - Optimistic concurrency control (OCC) — high throughput under low contention
   - Multi-version concurrency control (MVCC) with snapshot isolation — best of both, complex

**Actionable answer format:**
- An MVCC strategy compatible with the existing WAL
- An index-visibility strategy for B-tree/ART and HNSW
- A transaction isolation level decision
- An off-heap MVCC implementation plan

**Sources to consult:**
- "Concurrency Control and Recovery in Database Systems" (Bernstein, Hadzilacos, Goodman)
- PostgreSQL's MVCC implementation
- CockroachDB's MVCC
- FoundationDB's MVCC (record-layer)
- The existing `libravdb/internal/storage/wal/` source

---

### Q2.4 — Type System for SQL + Vector + Graph

**Claim in vision doc:** (no explicit claim)

**Why this needs research:**
- A SQL engine needs a type system. The unified engine needs a type system that covers scalars, vectors, and graph nodes/edges.
- The catalog must store type definitions. The parser must type-check expressions. The executor must dispatch on type.
- VECTOR(N) has fixed dimensionality at table-creation time, not per-row. This needs policy.
- Graph nodes/edges are user-defined types (UDTs) in SQL/PGQ, with properties that are typed scalars.
- The `VECTOR_DISTANCE` function has overloads by metric (L2, cosine, inner product, hamming). Type checking must resolve the overload at parse time or plan time.

**Specific questions to answer:**

1. What is the **type lattice** for the unified engine?
   - Scalars: INT, BIGINT, FLOAT, DOUBLE, TEXT, BOOLEAN, TIMESTAMP, etc.
   - Vector: VECTOR(N) with implicit FLOAT32 elements, fixed N
   - Graph: NODE, EDGE — what are these at the type level? Are they UDTs?
   - Composite: ROW, ARRAY, MAP
   - NULL: bottom of the lattice

2. What is the **type checking strategy**?
   - Static: all types resolved at parse time (PostgreSQL-style, errors at parse)
   - Dynamic: types resolved at plan time (better for UDFs and JSON, but more runtime errors)
   - Hybrid: scalar types static, UDTs dynamic

3. What is the **type coercion policy**?
   - Implicit INT → BIGINT: yes
   - Implicit TEXT → INT: no
   - Implicit VECTOR(384) → VECTOR(512): no (must be explicit)
   - Implicit distance metric: no (L2 ≠ cosine)

4. How are **graph types** represented in the catalog?
   - SQL/PGQ uses `CREATE PROPERTY GRAPH` with VERTEX TABLES and EDGE TABLES
   - The vertex/edge types are aliases for underlying relational tables
   - Properties on nodes/edges are columns on those tables

5. What is the **type representation in the wire protocol** (pgwire)?
   - Each type has a PostgreSQL OID
   - Custom types (VECTOR, NODE, EDGE) need custom OIDs
   - The type OID table is part of the catalog

**Actionable answer format:**
- A type lattice diagram
- A type checking strategy
- A type OID table for pgwire
- A catalog schema for type definitions

**Sources to consult:**
- PostgreSQL's type system documentation
- pgvector's OID assignments
- SQL/PGQ's `CREATE PROPERTY GRAPH` type semantics
- Apache Calcite's type system

---

## Tier 3 — Polish and Integration

These shape the user-facing surface and integration with existing code.

---

### Q3.1 — Vector Dimensionality Policy

The architecture doc listed this as an open question: per-column or per-row?

**Research questions:**
1. What do pgvector, Milvus, Qdrant, Weaviate, LanceDB, Chroma do?
2. What are the storage implications of each?
3. What is the right choice for a unified SQL engine? (Likely per-column, but confirm.)

---

### Q3.2 — Path Restrictor Subset for v1

The architecture doc suggested: `WALK`, `TRAIL`, `SIMPLE`, `ANY SHORTEST`.

**Research questions:**
1. What restrictors do real Cypher/PGQ workloads use? (Cite the Cypher Query Log or equivalent.)
2. What is the implementation cost of each?
3. Which ones are essential for the libravdbd use case (multi-hop episodic context)?

---

### Q3.3 — libravdbd Integration: Which Imperative Patterns Are SQL-Expressible?

**Claim in vision doc:** "libravdbd" achieves 94.88% on LongMemEval by imperatively orchestrating Extended Tail tracking, Hierarchical Session Routing, and Parallel Multi-Session retrieval.

**Research questions:**
1. Read `libravdbd`'s actual orchestration code. What does the imperative flow look like?
2. Which patterns can be expressed in SQL/PGQ + window functions? Which can't?
3. Is `LAG()`/`LEAD()` over `PARTITION BY session_id` actually the right primitive for "fetch semantic hit + surrounding temporal context"? Or is a different SQL construct needed (e.g., `MATCH ... {1,k}` with temporal predicates)?
4. What's left in Go after the SQL subsumption? Transaction coordination? Result formatting? Streaming?

---

### Q3.4 — ORM Compatibility Surface for pgwire

**Research questions:**
1. What features of pgwire do the top 10 ORMs (GORM, Bun, sqlx, pgx, JDBC, psycopg, sqlalchemy, Diesel, TypeORM, Prisma) actually use?
2. What's the minimum pgwire implementation that supports 80% of common ORM workflows?
3. Which ORM features are fundamentally impossible without a full PostgreSQL backend (e.g., `LISTEN/NOTIFY`, advisory locks)?

---

## What This Research Does NOT Cover (Out of Scope)

To keep the research focused, the following are explicitly **not** part of this prompt:

- **Catalog design in detail.** This is the next concrete deliverable, not a research question. It follows from Q1.1, Q1.2, Q2.4.
- **SQL/PGQ grammar transcription.** The grammar is settled (ISO/IEC 9075-16:2023). Transcribing it is mechanical.
- **API design for the public SQL surface.** This is a product decision, not a research question.
- **Marketing, positioning, naming.** Not research.
- **Competitive analysis beyond what's needed for the wedge.** The architecture doc already covers this.

---

## Suggested Research Order

The questions have dependencies. Recommended order:

1. **Q1.4 (Factorized processing)** — answers whether the existing graph layout is sufficient or needs modification
2. **Q1.1 (ART vs B-tree)** — answers the relational index decision
3. **Q1.3 (In-filtering HNSW)** — answers the vector filter strategy
4. **Q1.2 (pgwire)** — answers the wire protocol scope
5. **Q2.4 (Type system)** — depends on Q1.2 (type OIDs)
6. **Q2.1 (SoA AST)** — independent, can run in parallel with the above
7. **Q2.2 (C/Assembly pushdown)** — independent, can run in parallel
8. **Q2.3 (MVCC)** — depends on Q1.1 (which index), Q2.4 (type system)
9. **Q3.1–Q3.4** — polish, run last

Tier 1 questions block the catalog design. The catalog design is the next deliverable. So the priority is to resolve all four Tier 1 questions before writing the catalog design doc.

---

## Format of Research Output

Each research question should produce a research note in `docs/sql-research/` with the naming pattern `research-<topic>-<mm-dd-yyyy>.md`. Each note should:

1. State the question
2. Summarize the findings
3. Cite specific sources (papers, code, documentation)
4. End with a recommendation
5. Flag any open sub-questions that need follow-up

A synthesis doc will follow once all four Tier 1 questions are resolved, integrating the recommendations into a coherent design before the catalog work begins.
