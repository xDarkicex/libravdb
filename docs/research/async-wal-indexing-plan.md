# Async WAL and Indexing Plan

## Scope correction

LibraVDB currently has two distinct WAL implementations:

- `internal/storage/wal` is used by the separate graph store and its edge
  transactions.
- `internal/storage/singlefile` is the active persistence engine opened by
  `libravdb.Open` and is the durability boundary for vector records.

The generic graph WAL is not the correct foundation for asynchronous HNSW
construction. HNSW is a derived index over durable vector records; persisting
its edge mutations would couple recovery to one graph implementation and
substantially amplify writes.

## Existing active WAL

The single-file engine already has:

- File, chunk, and WAL-frame magic/version fields.
- Castagnoli CRC32 checksums on chunks and WAL frame payloads.
- Monotonic LSNs, transaction IDs, previous-LSN links, and begin/commit frames.
- Streaming WAL replay with incomplete transactions discarded.
- Snapshot/checkpoint metadata containing `LastAppliedLSN`.
- Hybrid buffering: 256 entries, a 10 ms periodic flush, and an adaptive
  foreground group-commit window from 1 to 5 ms.

The older `internal/storage/wal.Read` materializes entries, but the active
single-file replay does not. It reads and applies framed transactions while
walking the file.

## Correctness gaps

### WAL fsync was disabled (resolved)

The active engine now defaults to synchronous durability. Every flush appends
all member transactions, issues one shared `file.Sync()`, and only then publishes
record state and acknowledges writers. `DurabilityUnsafeNoSync` remains an
explicit benchmark-only upper bound.

### Index snapshot and storage LSN are not coordinated

`putRecordsInlocked` appends and applies storage records, then can call
`checkpointLocked` before `Collection.Insert` performs `index.Insert`. The
checkpoint serializes the index but records the storage engine's latest LSN.
The persisted index can therefore claim an LSN that it has not indexed.

Recovery loads that index snapshot before replaying post-checkpoint WAL records.
Replayed records update storage state, but there is no corresponding incremental
application to the cached index. A crash after a checkpoint can consequently
restore a stale index even though storage records recover correctly.

### Group commit does not yet provide one sync per group

`flushBatch` can write several merged collection transactions. Re-enabling
`Sync` inside every `appendTransactionLocked` would issue one sync per merged
transaction rather than one sync for the entire group. The append and durable
sync phases need to be separated so all group members share one `file.Sync()`.

### Async visibility semantics are undefined

Returning after WAL durability but before HNSW construction makes `Get`
immediately consistent while ANN search is temporarily eventual. The API must
either document that contract or search a bounded pending-vector overlay to
provide read-your-write behavior.

## Decisions

### Transaction schema

Keep record-level WAL transactions. A vector insert is a transaction containing
the vector/metadata record and ordinal. Do not log HNSW edges or backlink diffs.

The HNSW index owns neither durability nor recovery. On recovery it is restored
from an index snapshot and durable record deltas, or rebuilt from durable
records when the snapshot cannot be proven current.

### Group-commit policy

Use the existing hybrid policy as the initial baseline:

- Wake immediately when 256 entries are pending.
- Flush low-volume traffic after 10 ms.
- Coalesce foreground durability waiters for 1 ms, growing toward 5 ms under
  contention.
- Add a byte ceiling so large vectors cannot create an oversized group even
  when the entry count is small.
- Append every transaction in the group, issue exactly one `file.Sync()`, then
  acknowledge all waiters.

The cadence must remain configurable and benchmarked as durability latency,
not silently compiled out.

## Implementation order

1. Restore group durability: split append from sync, issue one sync per flush
   group, fail the engine on ambiguous sync/write errors, and test torn writes,
   incomplete commits, sync failures, and reopen recovery.
2. Establish an index-applied LSN. Persist it with each index snapshot (or use a
   conservative global minimum), and never publish a checkpoint claiming more
   index progress than has completed.
3. Make recovery correct before making it fast. Initially rebuild any index
   whose applied LSN trails recovered storage. Then add incremental replay of
   record puts/deletes after the snapshot LSN.
4. Add a bounded asynchronous index queue keyed by durable LSN. Preserve per-ID
   mutation order, expose queue depth and lag, and apply backpressure rather
   than allowing unbounded RAM growth.
5. Define search visibility. Either provide explicit durable-but-eventual
   inserts or merge a bounded exact scan of pending vectors into ANN results.
6. Make checkpointing wait for or record the indexer's applied-LSN barrier.
   Runtime topology repair remains an index maintenance activity and never
   participates in WAL recovery.
7. Benchmark three separate numbers: accepted writes/s, durably committed
   writes/s, and graph-ready inserts/s, with p50/p99/p99.9 acknowledgement and
   index-lag latency.

## Non-goals for this cycle

- Persisting HNSW edge-level mutations.
- Using runtime repair as crash recovery.
- Claiming durability from `Write` or buffered flush without `Sync`.
- Hiding graph construction behind an unbounded queue.

## First optimization pass

Implemented and validated:

- Synchronous durability is now the default public contract.
- `DurabilityUnsafeNoSync` is an explicit benchmark-only option.
- All transactions collected by one flush share exactly one `file.Sync()`.
- Buffered record state is published only after that group sync succeeds.
- Sync failures are returned to foreground writers and do not make the record
  visible in the live storage map.
- The contiguous WAL transaction image is built in a reusable 64-byte-aligned,
  mmap-backed arena. Oversized transactions use an exact-size temporary mmap
  instead of a Go-heap fallback.
- Recovery rebuilds a derived index whenever committed WAL replay touches its
  collection after the checkpoint. This is deliberately conservative until
  index-delta replay and persisted index-applied LSNs are implemented.
- Unclaimed recovery-cache indexes are closed after collection loading,
  including physical shard indexes that the parent collection loader does not
  consume.

Measured on Apple M2 with 768d float32 records:

| Workload | Result | Group occupancy |
|---|---:|---:|
| Durable, 8 pending writers | ~1.44k writes/s | 7.99 entries/transaction |
| Durable, 32 pending writers | 5.44k-5.61k writes/s | 31.88-31.93 entries/transaction |
| Unsafe no-sync, 8 writers | ~6.3k-6.5k writes/s | 8.00 entries/transaction |
| Durable 256-entry batches | ~49k vectors/s | 256 entries/transaction |

Moving the transaction image off heap removed one allocation per single insert
and approximately 98 KB of Go-heap allocation per 256-entry batch. Two further
allocation-reduction experiments were rejected:

- Shared ticket/channel completion reduced allocations but regressed durable
  throughput by roughly 11% due to wakeup lock contention.
- Admission-time slice coalescing reduced allocation count but increased bytes
  and regressed durable throughput by roughly 12-15%.

The durable WAL therefore exceeds the current ~3.5k graph-ready HNSW rate when
the group committer has enough pending work. The asynchronous index queue should
provide that occupancy while remaining bounded and exposing its durable-LSN to
index-applied-LSN lag.

## Bounded asynchronous HNSW indexing

Implemented as an opt-in database mode through `WithAsyncIndexing(depth,
workers)`. The first production scope is deliberately narrow: non-sharded HNSW
inserts. Other index types and sharded collections keep synchronous behavior.
Updates, upserts, and deletes take an async mutation barrier and drain prior
insert work before changing the index.

Unless explicitly overridden, async mode raises foreground write admission to
32 and uses the async queue depth as the writer-wait bound. Explicit
`WithMaxConcurrentWrites` and `WithMaxWriteQueueDepth` values remain authoritative;
the WAL group target is capped by the configured writer concurrency.

The queue has these properties:

- Each task is a fixed 16-byte `{durableLSN, ordinal}` record in a 64-byte-
  aligned mmap arena. It does not retain IDs, metadata, or 768d vector payloads.
- Workers resolve the canonical storage-owned vector and ID by ordinal, then
  run ordinary HNSW insertion.
- Capacity is reserved before WAL admission and held through index application.
  A batch larger than queue capacity fails before reaching the WAL; a full queue
  applies backpressure instead of allocating or growing.
- Once a transaction enters a WAL group, cancellation cannot abandon it before
  the definitive sync result. A durable record therefore always receives an
  index task.
- `IndexingStats` exposes durable LSN, conservative applied LSN, LSN lag,
  pending tasks, reservations, capacity, and failure state. The applied
  watermark advances whenever the queue fully drains, so it never claims work
  is indexed early.
- `FlushIndex(ctx)` is the explicit graph-readiness barrier. Search remains
  eventually consistent between durable acknowledgement and that barrier.
- Async collections are omitted from index checkpoint chunks. Recovery rebuilds
  every live collection absent from a valid chunk, preventing a checkpoint from
  publishing a graph behind durable records.

### Group target sweep

Apple M2, 768d float32, HNSW `M=16`, `efConstruction=100`, four index workers,
32 foreground writers, synchronous durability:

| Policy | Durable acknowledgements/s | Graph-ready/s | Entries/WAL transaction | Decision |
|---|---:|---:|---:|---|
| Existing adaptive window | 3.22k-3.29k | 2.73k-2.78k | 16.7-17.4 | Too-small groups |
| Target 32, max 10 ms | 2.47k-2.50k | 2.47k-2.49k | ~29.0 | Rejected: latency ceiling stalls admission |
| Target 24, max 5 ms | 3.90k-3.99k | 3.41k-3.47k | 26.0-26.5 | Good |
| Target 28, max 5 ms | 4.00k-4.40k | 3.33k-3.61k | 28.4-28.7 | Retained |

The bounded queue absorbs the short durable/index rate difference, then forces
foreground throughput to converge on graph construction once full. This is the
intended contract: lower durable acknowledgement latency and near-32 WAL groups
without hiding an unbounded graph backlog.

## Lock-free async queue pass

The original async queue used a mutex-protected ring plus capacity, worker,
change, close, and failure channels. It was replaced by a bounded MPMC ring with:

- one 64-byte off-heap slot per task;
- a per-slot generation/sequence counter;
- cache-line-separated atomic producer and consumer positions;
- atomic outstanding reservations acquired before WAL admission;
- atomic failure publication and durable/applied watermarks.

The worker notification channel remains only as a parking hint. Queue
correctness, capacity, enqueue, dequeue, backpressure, and flush observation do
not depend on it. A pure polling version was rejected because idle/reserved WAL
periods stole CPU from HNSW and produced severe throughput variance.

Same-machine A/B (`067cf9e` mutex ring versus sequence-counter ring), Apple M2,
768d, four index workers:

| Queue | Durable acknowledgements/s | Graph-ready/s | Allocations |
|---|---:|---:|---:|
| Checkpoint mutex ring | 3.04k-3.70k | 1.74k-2.26k | 12/op |
| Off-heap MPMC ring | 3.33k-3.75k | 2.61k-2.74k | 12/op |

The queue change improves index-worker delivery under the same machine load but
does not remove the remaining allocations. Those are primarily ID formatting,
single-ID mutation bookkeeping, WAL request/completion objects, and write
admission coordination.

## Lock-free WAL admission experiment

A bounded MPSC sequence-counter ring was tested between concurrent writers and
the existing single WAL file owner. Three reservation variants were measured:
capacity CAS, increment/rollback reservation, and canonical one-XADD admission
with per-slot sequence publication. All variants passed focused durability and
race tests, and the final variant removed one allocation plus roughly 180-200
bytes per unsafe write.

It was rejected because it made the complete WAL path slower. On the same Apple
M2, the existing short mutex admission path sustained approximately 3.9k-4.5k
durable writes/s with 32 pending writers and 5.7k-5.8k unsafe writes/s with
eight writers. The MPSC path sustained approximately 3.7k-4.1k durable writes/s
and 5.4k-5.6k unsafe writes/s: a 4-10% unsafe regression and as much as a 15-20%
durable regression in earlier variants.

The current admission mutex is short and effectively uncontended; wrapping the
same heap-backed request slices and completion channel in sequence counters only
added atomic cache-line traffic. A future MPSC attempt must change the request
representation itself (fixed/off-heap encoded requests and atomic completion),
not merely place the existing request objects inside a lock-free ring.

## Immutable adjacency CAS experiment

The HNSW fixed adjacency arrays were replaced experimentally with immutable
off-heap blocks carrying count and heuristic state in the same publication.
Writers allocated and copied a replacement block, atomically CAS-published its
pointer, and retired the prior generation through the existing Hyaline SMR.
Search scratch slots entered both adjacency allocators' Hyaline domains for the
duration of traversal. Append, overflow pruning, repair, deletion, persistence,
and backlink paths were converted, and focused correctness and race tests
passed.

The design was rejected on throughput. On the 5k normalized 768d fixture at
`M=16`, `efConstruction=200`, immutable publication sustained 956-968 inserts/s;
the exact `f1b60dc` in-place checkpoint subsequently sustained 1,125 inserts/s
on the same machine, a roughly 14-15% advantage despite thermal variance. The
common in-place append is one ID store plus count publication; immutable
publication adds an off-heap allocation, full adjacency copy, CAS, and
retirement to every edge. The experiment was removed in full. Immutable CAS
remains appropriate for rare large structural replacement, not for HNSW's
per-edge insertion path.

## Atomic per-key mutation state

The 64 `sync.Mutex` mutation stripes and `lockMutationIDs` map/slice/sort/closure
were replaced by a lazily allocated 4096-slot off-heap atomic state table. Each
cache-line-isolated slot serializes one key hash; rare hash-slot collisions
conservatively serialize unrelated keys but cannot admit conflicting writes.
Single-key guards are stack values. Batch mutation uses try-all/release/retry,
so overlapping batches cannot deadlock and require no sorted heap workspace.

Paired Apple M2 runs of `BenchmarkCollectionAsyncHNSWInsert` at 5,000 writes:

| Mutation coordination | Accepted writes/s | Graph-ready/s | Bytes/op | Allocs/op |
|---|---:|---:|---:|---:|
| `f1b60dc` mutex stripes | 3.73k-4.39k | 1.92k-2.28k | 1,195-1,212 | 12 |
| Off-heap atomic state | 3.74k-4.54k | 2.02k-2.15k | 1,111-1,130 | 10 |

The graph-ready spread remains dominated by HNSW scheduling, but accepted
throughput is neutral-to-positive and the two-allocation reduction is stable.
Focused collision, overlapping-batch, race, and zero-allocation guard tests
pass, so the atomic state table is retained.

## Off-heap WAL request completion

The per-write buffered `chan walFlushResult` was replaced by a fixed 64-byte
request record in a 4096-slot, 64-byte-aligned mmap arena. Foreground and file
owner exchange an 8-byte `{slot, generation}` handle. Durable LSN, state, entry
count, and generation are atomically published in the record; error boxes are
allocated only when a WAL group fails.

A first implementation used one shared condition variable. Standalone WAL was
faster, but condition broadcast woke every foreground writer and reduced async
HNSW graph-ready throughput to 1.38k-1.46k/s. It was rejected. The retained
design lazily creates one reusable buffered parking channel per concurrently
used request slot. The channel is a scheduling hint tied to the fixed record,
not a per-write allocation; atomic state remains authoritative. One explicit
`runtime.Gosched` after completion preserves CPU fairness for index workers.

Paired Apple M2 WAL benchmark (`bfb0616` versus fixed requests, 3,000 writes):

| Path | Checkpoint | Fixed request | Allocation change |
|---|---:|---:|---:|
| Durable, 32 writers | 234-239 us/op | 183-191 us/op | 7 to 5 allocs/op |
| Unsafe, 8 writers | 166-177 us/op | 160-164 us/op | 9 to 7 allocs/op |
| Durable group occupancy | 30.0-31.3 | 31.9 | none |

At the integrated async HNSW layer, accepted throughput reaches 6.1k-6.4k/s
with 31.5-31.9 entries/WAL transaction, while allocations fall from 10 to 8
per insert and bytes fall from roughly 1,100 to 958-969. Repeated graph-ready
results average within roughly 2-3% of the checkpoint despite expected
concurrent-topology variance. The fixed request path is retained.
