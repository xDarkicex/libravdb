# LibraVDB Architecture

This document describes the overall architecture, component design, data flow,
and system interactions within LibraVDB based on the current implementation.

## System Overview

LibraVDB is an **embedded vector database library** for Go with a layered
architecture that separates concerns and enables modularity. It stores all data
in a single portable file (`*.libravdb`) and provides HNSW-based approximate
nearest-neighbor search with optional quantization.

```
┌──────────────────────────────────────────────────────────────────┐
│                      Application                                 │
├──────────────────────────────────────────────────────────────────┤
│                    Public API (libravdb)                         │
│  Database │ Collection │ Tx │ QueryBuilder │ Streaming │ Batch   │
├──────────────────────────────────────────────────────────────────┤
│                   Internal Packages                              │
│  index/    filter/    graph/    quant/    memory/    obs/        │
│  ├─ hnsw/  ├─ equality/         ├─ product/ ├─ manager/         │
│  ├─ flat/  ├─ range/            └─ scalar/  ├─ cache/           │
│  └─ ivfpq/ ├─ containment/                  └─ mmap/            │
│            └─ logical/                                           │
├──────────────────────────────────────────────────────────────────┤
│                    Storage Layer                                  │
│  storage/  ──  singlefile/  ──  wal/                             │
│  (interfaces)  (LSM engine)   (write-ahead log)                  │
├──────────────────────────────────────────────────────────────────┤
│                   Operating System                               │
│        File I/O  │  Mmap  │  CPU Caches  │  Memory               │
└──────────────────────────────────────────────────────────────────┘
```

## Package Map

| Package | Import Path | Lines | Role |
|---------|------------|-------|------|
| `libravdb` | root | ~24K | Public API: Database, Collection, Tx, QueryBuilder, Streaming |
| `internal/index` | `.../internal/index` | 655 | Index interfaces and registry |
| `internal/index/hnsw` | `.../internal/index/hnsw` | ~6K | HNSW graph-based ANN index |
| `internal/index/flat` | `.../internal/index/flat` | ~1.5K | Brute-force exact search |
| `internal/index/ivfpq` | `.../internal/index/ivfpq` | ~5K | IVF-PQ with product quantization |
| `internal/graph` | `.../internal/graph` | ~4K | Property graph with WAL, segments, BFS |
| `internal/storage` | `.../internal/storage` | 113 | Storage interfaces |
| `internal/storage/singlefile` | `.../internal/storage/singlefile` | ~6K | Single-file LSM engine |
| `internal/storage/wal` | `.../internal/storage/wal` | 378 | Write-ahead log |
| `internal/quant` | `.../internal/quant` | ~4.5K | Product & scalar quantization |
| `internal/filter` | `.../internal/filter` | ~3K | Metadata filter DSL |
| `internal/memory` | `.../internal/memory` | ~4.3K | Mmap-backed memory management |
| `internal/obs` | `.../internal/obs` | 551 | Circuit breaker, health, Prometheus metrics |
| `internal/util` | `.../internal/util` | ~1K | Distance metrics, encoding, heap |

## Component Details

### 1. Database (`libravdb.Database`)

The top-level container. Owns the storage engine, collection registry, metrics,
health checker, and a scratch memory pool.

```go
type Database struct {
    storage       storage.Engine              // singlefile.Engine
    collections   map[string]*Collection      // collection registry
    bridge        *indexPersistenceBridge     // index snapshot provider
    metrics       *obs.Metrics                // Prometheus metrics
    health        *obs.HealthChecker           // health monitoring
    config        *Config                     // database config
    scratchPool   *sync.Pool                  // *memory.Arena pool
    logger        Logger                      // optional logger
    mu            sync.RWMutex
    closed        bool
}
```

**Key behaviors:**
- `Open()` auto-migrates v1 databases to the current format.
- Collections are lazily discovered from storage on reopen.
- The scratch pool provides allocation-free temporary memory for hot paths.

### 2. Collection (`libravdb.Collection`)

Manages vectors, their metadata, the search index, and storage handles.

```go
type Collection struct {
    name          string
    config        *CollectionConfig
    db            *Database
    index         index.Index                // HNSW, Flat, or IVF-PQ
    storage       storage.Collection         // singlefile collection handle
    shards        []shard                    // for sharded mode
    hooks         struct {
        onInsert  []InsertHook
        onDelete  []DeleteHook
    }
    mu            sync.RWMutex
    closed        bool
}
```

**Index creation flow:**
1. On collection creation, `createIndexForCollection` selects the index type
   based on configuration or auto-selection thresholds.
2. If an index snapshot is available (from persistence), it is deserialized.
3. If no snapshot exists, the index is built from storage records via
   `buildIndexForEntries`.
4. Auto-index thresholds default to HNSW at 10K vectors, IVF-PQ at 1M vectors.

### 3. Index Layer

Three index implementations share a common interface:

```go
type Index interface {
    Insert(ctx context.Context, entry *VectorEntry) error
    InsertBatch(ctx context.Context, entries []*VectorEntry) error
    Search(ctx context.Context, vector []float32, k int) ([]*SearchResult, error)
    Delete(ctx context.Context, id string) error
    Size() int
    MemoryUsage() int64
    Close() error
}
```

**HNSW** (`internal/index/hnsw`): The primary index. Multi-layer navigable
small-world graph with configurable M, EfConstruction, EfSearch. Supports
quantization, memory-mapped vector stores, and three vector store backends
(in-memory, slabby, mmap). Tombstone-based deletion.

**Flat** (`internal/index/flat`): Brute-force exact search. Good for small
collections (<10K vectors) where 100% recall matters.

**IVF-PQ** (`internal/index/ivfpq`): K-means clustering + product quantization.
Best for large collections (>1M vectors) where memory efficiency is critical.

### 4. Storage Layer

The storage layer is a **single-file LSM engine** (`internal/storage/singlefile`):

- **Page-based** with a 4096-byte page size.
- **File header** at page 0 with magic `"LIBRAVDB"`, format version, WAL
  boundaries, and active metapage pointer.
- **Dual metapages** (A/B) alternating on each checkpoint for crash-safe root
  switching.
- **B+tree catalog** keyed by collection name for collection discovery.
- **Per-collection B+trees** for record storage (by ID), ID index, and metadata
  index.
- **WAL** with frame-based entries, transaction bracketing (`TX_BEGIN` /
  `TX_COMMIT`), and committed-transaction replay on recovery.
- **Copy-on-write checkpointing** with page reclamation via a freelist.

See [Single-File Storage Spec](single-file-storage-spec.md) for the full binary
format specification.

### 5. Graph Layer (`internal/graph`)

An optional property graph for managing edges and relationships between vectors.
Used for graph-enhanced HNSW search where pre-computed edge sets filter
candidates.

**Components:**
- `store.go` — main graph store with edge table, index, and WAL
- `segments.go` — immutable edge segments
- `compaction.go` — segment compaction
- `bfs.go` — BFS traversal with off-heap bitsets
- `wal.go` — graph-level write-ahead log

### 6. Filter Layer (`internal/filter`)

A metadata filtering DSL supporting:

- **Equality**: `Eq(field, value)`, `NotEq(field, value)`
- **Range**: `Gt`, `Lt`, `Gte`, `Lte`, `Between`
- **Containment**: `ContainsAny`, `ContainsAll`, `Contains`
- **Logical**: `And`, `Or`, `Not` with nested grouping

Filters are applied post-search to refine results. Indexed fields (configured
via `WithIndexedFields`) enable pre-filtering at the storage level.

### 7. Quantization (`internal/quant`)

Two quantization strategies:

- **Product Quantization (PQ)**: Splits vectors into subvectors, quantizes each
  subspace independently using codebooks. 4–32x memory reduction.
- **Scalar Quantization (SQ)**: Quantizes each dimension independently. 2–8x
  memory reduction. Simpler and faster than PQ but less efficient.

Both require a training phase on a sample of the dataset.

## Data Flow

### Insert Path

```
Application
    │  col.Insert(ctx, id, vector, metadata)
    ▼
Collection.Insert()
    │  validation: dimension check, non-empty ID
    ▼
Storage: record serialized → WAL append (RECORD_PUT)
    │
    ▼
Index: vector added to HNSW graph (or Flat/IVFPQ)
    │  quantization applied if configured
    ▼
Memory: vector stored in VectorStore (memory/slabby/mmap)
    │
    ▼
Hooks: InsertHook invoked (if registered)
```

### Search Path

```
Application
    │  col.Search(ctx, queryVec, k)
    ▼
Collection.Search()
    │
    ▼
Index.Search()
    │  HNSW: greedy layer-by-layer traversal
    │  Flat: brute-force distance computation
    │  IVF-PQ: cluster probing + PQ distance
    ▼
Filter Application (if query builder used)
    │  metadata filters applied to candidates
    ▼
Result Assembly
    │  IDs, scores, metadata, vectors populated
    ▼
SearchResults returned (ordered by descending score)
```

### Transaction Commit Path

```
Application
    │  tx.Commit(ctx)
    ▼
commitTx()
    │
    ├─► 1. Acquire collection write locks
    ├─► 2. Validate CAS preconditions
    ├─► 3. Assign ordinals to new records
    ├─► 4. Build storage.TxOperation list
    ├─► 5. Commit to WAL (TX_BEGIN ... TX_COMMIT)
    ├─► 6. Apply mutations to in-memory indexes
    ├─► 7. Invoke InsertHook / DeleteHook
    └─► 8. Release locks
```

## Memory Architecture

### Vector Storage Backends

| Backend | Config Option | Description |
|---------|-------------|-------------|
| In-Memory | `WithRawVectorStoreMemory()` | Default. `[]float32` slices in Go heap. |
| Slabby | `WithRawVectorStoreSlabby(n)` | Fixed-size slab allocator. One slab per vector (dim × 4 bytes). |
| Mmap | (via memory.MemoryConfig) | OS-managed virtual memory for large datasets. |

### Memory Manager (`internal/memory`)

- Tracks memory usage across indexes, caches, and vector stores.
- Supports configurable memory limits with pressure levels (normal, warning,
  high, critical).
- Triggers GC, cache eviction, and degradation at threshold crossings.
- Platform-specific mmap support (Unix via `syscall.Mmap`, Windows via
  `syscall.MapViewOfFile`).

### Scratch Pool

A `sync.Pool` of 1MB `*memory.Arena` instances used by batch and streaming
operations to avoid heap allocations in hot paths. Arenas are reset and
returned to the pool after each use.

## Observability

### Prometheus Metrics (`internal/obs`)

- Counters: vector inserts, search queries, errors by code
- Histograms: search latency, insert latency
- Gauges: memory usage, collection count, active connections

### Health Checks

The `obs.HealthChecker` runs configurable health checks on:
- Database connectivity (storage file accessible)
- Memory usage (below threshold)
- Index integrity (no corruption detected)
- Storage space (sufficient free space)

### Circuit Breaker

The `obs.CircuitBreaker` protects against cascading failures:
- **CLOSED**: Normal operation, requests pass through.
- **OPEN**: Failure threshold exceeded, requests fail fast.
- **HALF_OPEN**: Testing recovery, limited requests allowed.

## Concurrency Model

See [Concurrency Design](concurrency.md) for the full model. Key points:

- Database-level RWMutex for collection registry.
- Per-collection RWMutex for index and storage operations.
- Stripe locking (64 stripes by FNV-32a hash) for per-ID serialization.
- Write controller bounds concurrent write parallelism.
- Worker pool for batch/streaming operations.
- Backpressure in streaming to prevent OOM.
- Single-writer commit via WAL serialization.

## Error Handling

See the [API Reference](../api-reference.md#error-handling) for sentinel errors
and structured error types. Key patterns:

- `VectorDBError` with machine-readable codes, severity levels, and recovery actions.
- `ErrorRecoveryManager` with pluggable `RecoveryStrategy` per error code.
- `GracefulDegradationManager` with 5 degradation levels.
- `AutomaticRecoveryOrchestrator` for cross-component recovery coordination.
- `BatchError` with per-item error tracking and retry logic.

## Design Decisions

### Why a single file?

A single `*.libravdb` file is:
- Portable — move/copy as one artifact.
- Self-contained — no sidecar files needed.
- Simple to backup and restore.
- Versioned from day one for forward compatibility.

### Why LSM-tree?

- Write-heavy workloads (vector insertion) benefit from append-only writes.
- Background compaction maintains read performance.
- Copy-on-write enables crash-safe checkpointing without blocking readers.

### Why HNSW as the primary index?

- Logarithmic search complexity (O(log N)).
- Excellent recall (>95%) with proper tuning.
- Incremental updates without full rebuild.
- Memory overhead is manageable (~50% over raw vectors).

### Why typed atomics?

The `sync/atomic` package (Go 1.19+) provides type-safe `Int64`, `Bool`,
`Pointer` types that are harder to misuse than raw `atomic.AddInt64`. Used for
lock-free counters, flags, and single-pointer swaps.
