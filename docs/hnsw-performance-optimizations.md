# Index Performance Architecture

## Tiered Off-Heap Memory: Search Hot Path

The flat (brute-force) and IVFPQ search paths use a tiered power-of-2 inline max-heap backed by `memory.ShardedFreeList` off-heap pools. This eliminates `runtime.mallocgc` and `runtime.growslice` from the inner distance-scan loop — the CPU profile shows 81.8% of time in `CosineDistance_func` with zero allocation overhead.

### Heap Architecture

```
k=1–16    → 320 B  slot  (16 × 16 B elements + 64 B metadata)
k=17–128  → 2,112 B slot  (128 × 16 B + 64 B)
k=129–1024 → 16,448 B slot (1024 × 16 B + 64 B)
k=1025–4096 → 65,600 B slot (4096 × 16 B + 64 B)
k>4096    → rejected (ErrExceedsMaxK)
```

Each tier uses a pre-allocated 16 MB `ShardedFreeList` with Hyaline SMR. The RAII `heapSlot` struct captures the originating pool at allocation time, so `free()` routes to the correct tier by construction — no runtime lookup, no mismatched `Deallocate` error.

### Complexity

- **Previous**: unbounded `append` + `sort.Slice` → O(N log N) time, O(N) heap
- **Current**: bounded max-heap streaming → O(N log k) time, O(k) off-heap
- **At 1M scanned entries (k=10)**: ~6× asymptotic improvement

### Benchmark (Apple M2, -benchmem)

```
BenchmarkFlatSearch-8    6276    192891 ns/op    6352 B/op    32 allocs/op
```

The 32 allocs/op are entirely from result extraction (`cloneFloat32`, `cloneMetadata`), not the inner scan loop. CPU profile confirms `runtime.growslice` is absent and `runtime.mallocgcSmallScanNoHeader` accounts for 0.3% of samples.

## Concurrency Hardening

### Data Race Elimination (all index types)

All 7 engine read methods (`Get`, `GetByOrdinal`, `GetIDByOrdinal`, `Exists`, `Count`, `MemoryUsage`, `NextOrdinal`) hold `e.mu.RLock()` across the full data access window. Previously, the `persisted()` helper released the lock immediately, allowing a concurrent `flushBatch` to mutate state while the reader was mid-access.

### Iterate Starvation Prevention

`Collection.Iterate` collects record IDs under a single RLock (fast — map key iteration only), then processes in 10K-record chunks. Each chunk re-acquires RLock briefly to clone entries, then calls the user callback outside the lock. This bounds the worst-case lock hold to O(10K) regardless of collection cardinality, preventing `flushBatch` (which requires the exclusive `e.mu.Lock()`) from being starved by a multi-million-record iteration.

### Group Commit Atomic Padding

`groupCommitWindow`, `groupCommitMaxWindow`, and `groupCommitStepWindow` are each padded to 64 bytes (one Apple M2 cache line) to prevent MESI false sharing between ingestion threads (Store/Add) and the flusher loop (Load).

### Collection Index Swap Safety

`Collection.Search` snapshots `c.index` and `c.shards[i].index` under `c.mu.RLock()` before releasing the lock, preventing a race with concurrent transaction commit index swaps.

## IVFPQ Parallel Reduce: k-way Merge

The parallel `collectCandidates` distributes probed clusters across workers, each producing a pre-sorted (ascending distance) top-k result array. The coordinator merges these via a min-heap of size W (workers, typically ≤ GOMAXPROCS):

- **Before**: element-by-element re-heap into a size-k max-heap → O(W·k log k)
- **After**: k-way merge over W pre-sorted arrays → O(W·k log W)

## HNSW-Specific Optimizations

### Search Path

- Slice-based visited tracking replaces map-based for better cache locality
- `EfSearch` bounded by `max(EfSearch, k)` to guarantee k results
- k validated: `k ≤ 0` returns error, `k > 4096` returns `ErrExceedsMaxK`

### Insertion

- `BatchInsert` with chunked processing (100 vectors per chunk) and context cancellation
- Pre-allocated node slice growth to avoid repeated reallocations
- Pre-allocated link capacities based on HNSW parameters

### Neighbor Selection

- Limited diversity checks to 3 closest candidates instead of exhaustive O(n²)
- 80% distance threshold for redundancy detection
- Pre-sorted candidates by distance

## k-Cap DoS Prevention

All three index types (`flat`, `ivfpq`, `hnsw`) reject `k > 4096` at the `Search` entry point. Without this cap, `k=5,000,000` would trigger the Go heap fallback in `acquireHeapSlot` → `make([]heapElement, 5000000)` → ~80 MB heap allocation per query, bypassing the off-heap pool and the PID memory controller entirely.

## Memory Manager

`getCurrentUsage()` calculates heap usage via `runtime.ReadMemStats` plus registered cache and memory-mapped component sizes. The monitor loop checks memory pressure at `MonitorInterval`, evicts from registered caches, and enables memory mapping on `MemoryMappable` components ranked by size (largest first). `lastPressureLevel` uses `atomic.Int32` to avoid races between concurrent monitor ticks.

## Verified Benchmarks (Apple M2, 2026-05-20)

### Search Latency by Collection Size

| Size | Flat (default_10k) | HNSW (lowered_1k) |
|------|-------------------|-------------------|
| 500 | 0.05ms | 0.04ms |
| 1,000 | 0.08ms | 0.06ms |
| 2,000 | 0.11ms | 0.04ms |
| 5,000 | 0.25ms | 0.26ms |
| 10,000 | 0.39ms (HNSW) | 0.38ms (HNSW) |

### Index Build Cost

| Size | Flat | HNSW | Ratio |
|------|------|------|-------|
| 500 | 6ms | 55ms | 9× |
| 1,000 | 7ms | 156ms | 22× |
| 2,000 | 7ms | 446ms | 64× |
| 5,000 | 11ms | 2,149ms | 193× |

### IVFPQ vs HNSW (25,000 vectors, 32-dim)

| Index | Insert | Search | Memory | Throughput |
|-------|--------|--------|--------|------------|
| IVF-PQ Default | 2,464ms | 1,099ms | 12.7 MB | 91 qps |
| IVF-PQ HighSpeed | 1,025ms | 2,101ms | 12.7 MB | 48 qps |
| HNSW | 28,388ms | 25ms | 18.9 MB | 3,866 qps |

### Validation

| Metric | Target | Result |
|--------|--------|--------|
| Save throughput | >10 MB/s | 50.1 MB/s |
| Load throughput | >15 MB/s | 102.3 MB/s |
| Search impact during save | <5% | −1.2% |
| Corruption detection | 100% | 100% (3/3) |
| Atomic operations | 100% | 100% (10/10) |
