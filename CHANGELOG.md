# Changelog

## [Unreleased]

### Added
- **IVF-PQ index persistence**: binary serialization format (v2) with CRC32 integrity checks, saving coarse centroids, PQ codebooks, and per-cluster inverted lists (ordinal + compressed codes). `SerializeToBytes` / `DeserializeFromBytes` + `SaveToDisk` / `LoadFromDisk` API.
- **IVF-PQ two-phase deserialization**: `DeserializeFromBytes` restores centroids and codebooks without retraining; `PopulateEntriesFromStorage` wires storage records into pre-serialized clusters by ordinal, avoiding centroid-distance recomputation entirely. Cluster membership is authoritative from the persisted inverted lists.
- **IVF-PQ quantizer reconstitution**: when the storage engine bridge creates an index without quantization config during reopen, `DeserializeFromBytes` now reconstitutes the quantizer from the payload's stored codebook metadata rather than rejecting the chunk — preventing silent fallback to full rebuild.
- **IVF-PQ restart benchmarks** at 1K, 10K, and 50K covering both valid-persisted and corrupt-chunk (forced rebuild) paths.
- **File compaction with WAL truncation**: `Compact()` public API rewrites the database from snapshot only, discarding all accumulated WAL history. Auto-compaction triggers in `checkpointLocked` when file size exceeds 2× the snapshot-only minimum.
- **`compactionErrors` counter** on Engine with `CompactionErrors()` accessor for observability into silent compaction failures.
- **Tiered off-heap max-heap**: power-of-2 `ShardedFreeList` pools (k=1–16, 17–128, 129–1024, 1025–4096) eliminate `runtime.mallocgc` and `runtime.growslice` from the inner distance-scan loop. `heapSlot` RAII wrapper binds each slot to its originating pool for safe deallocation.
- **k-cap DoS prevention**: all three index types (`flat`, `ivfpq`, `hnsw`) reject `k > 4096` at the `Search` entry point.
- **IVFPQ k-way merge**: parallel `collectCandidates` results merged via min-heap of size W (workers), reducing merge complexity from O(W·k log k) to O(W·k log W).

### Fixed
- **IVF-PQ persisted reopen performed full rebuild**: `DeserializeFromBytes` rejected valid chunks when the bridge-created index lacked quantization config, causing `loadIndexes` to fall back to `rebuildCollectionIndexFromRecords`. Both benchmark paths were identical (1.0x gap at all scales). Fixed by reconstituting the quantizer from stored payload metadata.
- **Memory manager data race**: removed dead `lastUsage` field (written under RLock, never read), converted `lastPressureLevel` to `atomic.Int32`.
- **Collection index swap race**: `Search` now snapshots `c.index` and shard indexes under `c.mu.RLock()` before releasing.
- **Engine read method races**: all 7 read methods (`Get`, `GetByOrdinal`, `GetIDByOrdinal`, `Exists`, `Count`, `MemoryUsage`, `NextOrdinal`) hold `e.mu.RLock()` across the full data access window.
- **MESI false sharing**: `groupCommitWindow`, `groupCommitMaxWindow`, and `groupCommitStepWindow` padded to 64-byte cache line boundaries.
- **Iterate starvation prevention**: `Collection.Iterate` collects IDs under RLock, processes in 10K-record chunks re-acquiring RLock per chunk, preventing `flushBatch` starvation at scale.
- **`ordinalToID` slot leak**: `applyRecordDelete` now clears the ordinal slot, and startup reconstruction skips deleted records so the slot stays released across restarts.
- **`shards` range copies mutex**: fixed `for i := range c.shards` to avoid copying `sync.Mutex`.

### Changed
- **`hnsw-performance-optimizations.md`**: rewritten to reflect actual tiered off-heap architecture, verified benchmarks, concurrency hardening, k-way merge, and k-cap protection.
- **IVF-PQ `PopulateEntriesFromStorage` allocation pre-sizing**: `ordinalToEntry` map, per-cluster `Entries` slices, and per-cluster `CompressedVectors` maps are now pre-sized using entry counts from the deserialized inverted lists, eliminating incremental rehash and slice-growth allocations during persisted reopen. Empty clusters are skipped. Net savings: ~1.7% B/op and ~0.2% allocs/op at 50K; dominant allocations (>90%) are structural (vector data clones and entry struct wrapping), inherent to the two-layer architecture and not reducible without refactoring storage→index data ownership.

### Benchmark Results (Apple M2)
- **IVF-PQ persisted reopen** (no retraining, direct cluster placement by ordinal):
  - 1K vectors (16 clusters, PQ 16×8): 1.81ms, 2.75MB, 12,442 allocs — **24.6x faster** than rebuild (44.5ms)
  - 10K vectors (64 clusters, PQ 16×8): 16.5ms, 22.6MB, 85,101 allocs — **278x faster** than rebuild (4.58s)
  - 50K vectors (128 clusters, PQ 16×8): 97.4ms, 110.4MB, 406,214 allocs — **893x faster** than rebuild (86.94s)
- IVF-PQ persisted reopen allocation profile (50K): dominant sources are vector data clones (~25.6MB) and entry struct wrapping (~4MB) — both structural. Remaining transient allocations: deserMeta entry structs (~2.4MB) and map bucket overhead (~2MB), freed at end of population.
- Flat search (1M entries, k=10): 192 µs/op, 6352 B/op, 32 allocs/op — allocations are from result extraction only, inner scan loop is zero-alloc.
- HNSW search (25K vectors, 32-dim): 25 ms/op, 3,866 qps.
