# Changelog

## [2.0.0] — 2026-06-12

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
