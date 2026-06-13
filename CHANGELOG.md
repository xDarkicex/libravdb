# Changelog

All releases follow [Go module versioning](https://go.dev/doc/modules/version-numbers). The Go module path is `github.com/xDarkicex/libravdb` (major version 1). Human release numbers are mapped to Go module versions below.

## [v2.1.1] / Go v1.2.1 — 2026-06-13

11 files changed, 251 insertions, 74 deletions across 3 commits.

### Graph API

- **Complete `Graph` interface** — `BeginTxn`, `AddEdge`, `RemoveEdge`, `DropNodeEdges`, `Neighbors`, `Degree`, `InboundNeighbors`, `InboundDegree`, `NeighborsAny`, `ForEachEdge`, `BFS`, `GetBitset`/`PutBitset`, `GetFrontierBuf`/`PutFrontierBuf`, `Stats`, `Close` all exposed publicly.
- **Public type aliases** — `Edge`, `KindSet`, `Bitset`, `FrontierBuf`, `VisitAction`, `GraphStats`, `Txn` re-exported from `internal/graph` so consumers don't import internal packages.
- **`GraphConfig`** — direct struct (not pointer alias) for configuring memory budget at construction.

### Documentation

- **v2.1.0 changelog** — comprehensive entry covering all 8 phases of the graph layer.
- **README refresh** — repositioned as hybrid vector-graph database with updated architecture diagram, graph layer feature section, API usage example, and project structure.

### Housekeeping

- **gofmt pass** — whitespace and import ordering cleanup across `internal/storage`, `internal/util`, `internal/storage/wal`.

## [v2.1.0] / Go v1.2.0 — 2026-06-13

71 files changed, 5,135 insertions, 231 deletions across 8 commits.

### Graph Layer (`internal/graph/`)

New subsystem providing directed, typed edge relationships between vectors with zero heap allocations on the hot path.

- **Edge primitive** — 16-byte fixed struct (`Target uint64`, `Weight float32`, `Stamp uint32`/`Kind uint8` packed). Allocated from `ShardedFreeList` (SlotSize=80, 3 edges per slot, 64 shards). Kind encoded in Stamp[31:24], 24-bit timestamp in Stamp[23:0] for MVCC.
- **EdgeTable pages** — 4KB pages with inline-first-8 layout. First 8 edges stored directly in the header; overflow chaining activates for hub nodes (>8 edges). Per-page MVCC generation counter with per-page mutex for writes.
- **KindSet** — 256-bit bitset (`[4]uint64`) providing branch-free edge kind filtering in ~1 CPU cycle.
- **Operations** — `AddEdge`, `RemoveEdge`, `DropNodeEdges`, `Neighbors`, `NeighborsAny` (kind-filtered), `Degree` — all O(1). Lock-free reads via Hyaline SMR, writes via per-page mutex with 64-way sharding.
- **BFS traversal** — `BFS(start, maxDepth, visit, bitset, frontier)` with caller-managed off-heap buffers. Visited-node dedup via Bitset, bounded depth enforcement, early termination on callback return. Target: ~30 CPU cycles per edge.
- **Reverse index** — symmetric edge storage (forward A→B, reverse B→A) enabling O(degree) node deletion without full graph scan.
- **Forward/reverse atomicity** — compensating rollback guarantees index symmetry on partial failures.

### Durability

- **WAL integration** — 4 new operation codes: `OpEdgeAdd` (0x40), `OpEdgeRemove` (0x41), `OpNodeEdgeDrop` (0x42), `OpTxnCommit` (0x4F). Fixed-width binary records with CRC32 checksums, exact-width deserialization, and atomic transaction commit/abort semantics.
- **Transaction atomicity** — corrupted transactions marked and skipped entirely during replay. `lastFlushedGen` guarded against regression.
- **Checkpoint coordinator** — background goroutine polling vector and edge subsystem generation counters every 100ms. Advances checkpoint when both confirm flush.
- **Segment persistence** — `FlushToSegment`/`LoadFromSegment` with zero-copy mmap I/O via `memory.MmapFileReadOnly`. 32-byte binary header (version, node/edge count, CRC32) with embedded manifest pointers. Directory fsync after atomic rename for crash durability. SGMT 4-byte magic footer.
- **Segment compaction** — `CompactSegment` performs CRC re-validation, opportunistic V1→V2 layout migration, defragmentation, and recomputes counts from consumed data. Self-verifying: asserts full payload consumption.
- **KindManifest** — `DBManifest` with `MinReaderVersion` enforcement and `KindManifest` map (uint8→string). Embedded in segment between header and edge records — no sidecar files, no split-brain hazard.

### Public API (`libravdb/`)

- **Graph interface** — `AddEdge`, `RemoveEdge`, `DropNodeEdges`, `Neighbors`, `NeighborsAny`, `Degree`, `InboundNeighbors`, `InboundDegree`, `ForEachEdge`, `BFS`, `GetBitset`/`PutBitset`, `GetFrontierBuf`/`PutFrontierBuf`, `Stats`.
- **GraphConfig** — configurable memory budget (Edge slots, page slots, Bitset/Frontier pool sizes, Arena pages).
- **Insert hooks** — `InsertHook`/`DeleteHook` callbacks (max 4 each) fired before WAL append. Receive `GraphTx` interface exposing `AddEdge`/`RemoveEdge`. Nil hook rejection, error propagation aborts transaction.
- **Graph-filtered search** — `QueryBuilder.WithGraphFilter(GraphFilter)` propagates bitset to all index types. `GraphFilter` interface with single `Test(uint64) bool` method.
- **Edge, KindSet, Bitset** types exported alongside Graph interface.

### Index Integration

- All three index types (Flat, HNSW, IVF-PQ) accept `GraphFilter` in their `Search` signatures.
- **Flat**: filter check on loop index `i` — no struct change, no serialization impact.
- **HNSW**: filter check via `candidate.Ordinal` during traversal — filtered nodes traversed but excluded from results.
- **IVF-PQ**: filter check via `entry.Ordinal` — early skip before distance computation.

### Concurrency & Resilience

- **Exponential backoff with jitter** — write-path retry on per-page mutex contention, randomized jitter prevents thundering-herd wakeups.
- **Memory exhaustion** — `runtime.GC()` + retry before giving up on pool allocation failures.
- **WAL replay resilience** — corrupted records skipped with warnings, not panics.
- **Checkpoint coordinator idempotency** — `sync.Once`-guarded Start/Stop.
- **BFS depth enforcement** — depth guard before visit callback, no enqueue at maxDepth.
- **ForEachEdge global early-stop** — returning false from callback terminates full iteration.
- **FrontierBuf overflow detection** — `Push` returns bool, BFS checks result, never silently drops reachable nodes.
- **DropNodeEdges error propagation** — cleanup loop errors returned, not discarded.

### Testing

- **18 gopter property tests** (100 iterations each) covering all correctness properties: KindSet filtering, inline-first-8 layout, overflow activation, generation monotonicity, Neighbors completeness, Degree consistency, NeighborsAny filtering, BFS reachability, BFS early termination, forward/reverse symmetry, DropNodeEdges completeness, WAL CRC32 integrity, transaction atomicity, replay state reconstruction, replay reverse index reconstruction, metric counter correctness.
- **Stress tests**: hub nodes with 100K+ edges, concurrent readers/writers on shared pages, memory exhaustion.
- **Fuzz tests**: WAL record deserialization with random byte inputs, CRC32 corruption detection.
- **Benchmarks**: EdgeTable page reads, BFS inner loop, AddEdge, Neighbors, Filter bitset check.
- **CI**: GitHub Actions workflow targeting linux/amd64 + darwin/arm64 on PRs to main.

### Dependency Changes

- `github.com/xDarkicex/memory` — v1.0.2 → v1.0.3 (adds `MmapFileReadOnly`, `Munmap` for zero-copy segment I/O)

## [v2.0.0] / Go v1.1.0 — 2026-06-12

151 files changed, 7,763 insertions, 3,897 deletions across 59 commits.

### Breaking Changes

- **`libravdb.New` renamed to `libravdb.Open`** — all callers must update. The new name reflects that `Open` may create or open an existing database file.
- **Context propagation** — `Stats(ctx)`, `GetMemoryUsage(ctx)`, and `TriggerGC(ctx)` now require `context.Context`.
- **Binary format v2** — all index magic numbers changed (HNSW `0x484E5357` → `"LIBRAHNS"`, Flat → `"LIBRAFLT"`, IVFPQ → `"LIBRAIVF"`), WAL encoding switched from JSON to binary, codec version bumped to 2, Creator field embedded in file header (`"libravdb/2.0.0 xDarkicex"`). v1 `.libravdb` files are not directly readable by v2.

### Migration (v1 → v2)

- **Auto-migration on `Open()`** — v1 files are detected by `formatVersion == 1`, transparently migrated to v2 format, and the original is backed up as `.v1.bak`. No daemon code changes needed beyond the `New`→`Open` rename.
- **`Migrate(ctx, path)` API** — standalone migration for preflight or batch scripting. Crash-safe three-step atomic rename (migrating → staged → backup → path) with startup recovery for interrupted migrations.
- **`ErrV1FormatMigrationRequired`** sentinel for programmatic detection.

### New APIs

- **`Vacuum(ctx)`** — reclaim disk space without blocking concurrent operations. Three-phase WAL fast-forward design holds locks for ~1ms (snapshot) + ~5ms (catch-up + rename).
- **`Backup(ctx, destPath)`** — point-in-time snapshot to a new path under brief lock.
- **`Drop(ctx)`** — completely destroy the database file and close the engine.
- **SystemHealthMonitor** — wired into the Database lifecycle with circuit breaker, health checks, and Prometheus metrics.

### Performance

- **Off-heap vector storage** via `github.com/xDarkicex/memory` ShardedFreeList: Flat index vectors, HNSW node links, arena-backed search/insert/deserialization scratch buffers. Eliminates GC pressure on the hot path.
- **HNSW**: precomputed D×D distance matrices, bulk-read vectors (single `io.ReadFull` vs `binary.Read` per float32), arena-backed `readNodes` deserialization.
- **IVFPQ**: matmul-style cluster assignment with centroid norm precomputation, bounded max-heap for probe cluster selection (replaces full sort), `PrepareQuery` for concurrent query caching, arena-backed search path.
- **Flat**: scratchPool arena for search path, cached metadata size on insert to avoid O(N) `MemoryUsage` scans.
- **Algorithmic fixes**: k-means++ running-min, dot-product argmin in cluster assignment, O(N²) → O(N) delete fallback, bubble sort → `sort.Slice` in multiple hot paths.

### Concurrency

- **Shard mutation locks** — `updateSharded`, `upsertSharded`, and sharded `Delete` now use `Lock()` instead of `RLock()` to prevent concurrent mutation races.
- **Write admission gate** — bounded per-collection write concurrency with context-aware queueing.
- **Lock contention** — `sync.RWMutex` with lock mutation IDs for parallel shard inserts, mutex released before CPU-bound search computation in IVFPQ.
- **Goroutine lifecycle** — `goleak` integration, `errgroup` supervision, buffered error channels, context cancellation through flush and streaming paths.

### Bug Fixes

- Arena-backed string cloning in HNSW persistence (dangling pointer into recycled arena).
- Off-heap SFL for node link arrays in readNodes.
- `arena.Reset()` per-node to prevent exhaustion on large files.
- Vector deallocation on Flat delete, insertLocked, and BatchInsert.
- Nil guard for IVFPQ `CompressedVectors` with lazy initialization.
- `MaxVectorSize` cap before `make()` in binary decoder.
- `clear()` on arena-backed slices after reset in IVFPQ BatchInsert.
- `WriteByte` returns `error` for `io.ByteWriter` compliance.
- HNSW size==1 delete SFL link slot leak.
- Flat search result cloning to prevent caller mutation of internal state.
- Deep clone metadata maps to prevent nested reference corruption.
- Bounds and nil guards on `h.nodes[neighbor.ID]` in HNSW connection functions.
- `NaN` returned from `CosineDistance` for zero vectors.
- Path traversal rejection in `MemoryMapManager.CreateMapping`.
- `atomic.Bool` for closed/adaptiveMode flags.
- ArenaSlice error propagation in IVFPQ merge and batch chunk processing.

### Binary Format Details

| Component | v1 (old) | v2 (new) |
|---|---|---|
| File header | 72 bytes, checksum at 72:76 | 96 bytes with 24-byte Creator, checksum at 96:100 |
| HNSW magic | `0x484E5357` (uint32) | `"LIBRAHNS"` (8 bytes) |
| Flat encoding | JSON | `"LIBRAFLT"` binary with `FlatFormatVersion=1` |
| IVFPQ encoding | JSON | `"LIBRAIVF"` binary with `ivfpqFormatVersion=2` |
| WAL frames | JSON | Binary tagged-value encoding (codec v2) |
| Snapshot | JSON | Binary tagged-value encoding (codec v2) |
| `SFLMetadataOverhead` | implicit 48 (magic number) | `SFLMetadataOverhead` constant in `hnsw/format.go` |

### Removed

- `internal/storage/lsm/` — removed in favor of singlefile engine.
- `internal/storage/segments/` — removed alongside LSM.

### Dependency Changes

- `github.com/xDarkicex/memory` — v1.0.0 → v1.0.2
- `go.uber.org/goleak` v1.3.0 — promoted from indirect to direct
- `github.com/xDarkicex/slabby` — removed from direct dependencies
