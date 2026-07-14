# DiskANN SoA Candidate Queue Experiment

Date: 2026-07-11

## Design

LibraVDB's HNSW traversal now defaults to a DiskANN-style bounded sorted beam:

- `uint32` node IDs and expansion state are stored together, with the high ID
  bit marking an expanded candidate.
- `float32` distances are stored in a separate contiguous array.
- Both arrays are allocated from the existing 64-byte-aligned off-heap search
  arena.
- A cursor identifies the closest unexpanded candidate.
- Sorted insertion uses a binary-plus-linear hybrid lower-bound search and
  in-place `copy`/memmove.
- Capacity is fixed at `ef`; no Go heap growth is allowed.

The high-bit encoding is safe because the segmented node registry has a hard
capacity of `2^28` ordinals. A compile-time assertion ties that registry limit
to the queue's `2^31` expansion bit, and insertion rejects ordinals outside the
registry capacity.

This queue replaces both the result max-heap and working min-heap in SoA mode.
The prior four-ary heap remains available as a benchmark comparator.

## Correctness Gate

On the 5,000-vector, 128-dimensional shootout:

| Mode | Build throughput | Recall@10 | Level-0 links |
|---|---:|---:|---:|
| Four-ary heap | 4,538 inserts/s | 1.0000 | 337,439 |
| SoA beam | 4,550 inserts/s | 1.0000 | 337,439 |

The identical link count verifies that the SoA construction path did not buy
throughput by producing a sparser graph.

## 768d Construction

Configuration: 5,000 normalized random vectors, dimension 768, `M=36`,
`efConstruction=200`, `efSearch=200`, graph-only preloaded vectors, Apple M2,
`GOMAXPROCS=1`.

| Mode | Graph inserts/s, three runs | Mean | Recall@10 | Level-0 links |
|---|---|---:|---:|---:|
| Four-ary heap | 588.4, 585.0, 589.3 | 587.6 | 1.000 | 840,725 |
| SoA beam | 596.5, 598.8, 596.5 | 597.3 | 1.000 | 840,725 |

SoA improved graph construction throughput by approximately 1.65% while
producing the same edge count and recall.

## 768d Search

An alternating-order, 1,000-query benchmark removed systematic cache-order
bias. Across three runs, heap-to-SoA mean-latency speedups were:

- 1.022x
- 1.018x
- 1.034x

SoA improved p50, p95, and p99 in every paired run. Absolute latency varied
substantially with host thermal and scheduler state, so the alternating-order
ratios are the useful result. Recall@10 was 1.000 for every mode and run.

## Hybrid Lower Bound

The original lower bound used binary search throughout. A hybrid implementation
now narrows the interval with binary search and linearly scans the final 16
contiguous entries when the queue contains at least 96 candidates.

Isolated Apple M2 timings at `GOMAXPROCS=1`:

| Queue size | Binary | Hybrid | Hybrid improvement |
|---:|---:|---:|---:|
| 32 | 8.18 ns | 10.00 ns | -22.2% |
| 64 | 10.36 ns | 10.87 ns | -4.9% |
| 100 | 12.19 ns | 10.38 ns | 14.8% |
| 200 | 16.64 ns | 11.58 ns | 30.4% |
| 512 | 27.34 ns | 14.47 ns | 47.1% |

The 96-entry crossover keeps the faster binary path for small queues. In the
full 5,000-vector, 768d interleaved search benchmark, binary-to-hybrid mean
latency speedups were `1.010x`, `1.006x`, and `1.009x`, with recall@10 fixed at
`1.000`. P50 and p95 improved in all three runs. P99 improved twice and
regressed once, so the tail result remains noise-sensitive.

## Rejected NEON Tail

A 16-entry ARM64 NEON tail was implemented and tested rather than inferred. It
used `FCMGE` on four lanes at a time after the same binary narrowing, then moved
the comparison mask back to general-purpose registers to locate the first lane.
It preserved exact ordering but lost to the cache-hot scalar tail:

| Queue size | Scalar hybrid median | NEON-tail median | Regression |
|---:|---:|---:|---:|
| 100 | 26.78 ns | 33.22 ns | 24.0% |
| 200 | 28.86 ns | 35.74 ns | 23.8% |
| 512 | 35.23 ns | 42.15 ns | 19.6% |

The assembly call and NEON-to-GPR mask extraction cost more than the remaining
scalar comparisons. The kernel was removed; revisiting SIMD here only makes
sense if admission is moved into a larger assembly operation that amortizes the
boundary and keeps masks in vector registers.

## Rejected Batch Movement Experiments

An exact four-candidate merge was tested against four sequential insertions. It
sorted surviving candidates and rewrote the bounded queue once while preserving
expanded bits and the cursor. At `ef=200`, rewriting all queue entries was much
more expensive than the runtime's targeted overlapping copies:

- Two accepted candidates: roughly 7x slower.
- Four accepted candidates: roughly 2.2x slower.
- Zero or one accepted candidate: also slower.

A narrower hybrid manually moved paired ID/distance entries only when the tail
contained at most four entries, using runtime `copy` for larger tails. The
primitive itself won for one to four entries, but eight full 768d paired runs
produced speedup ratios of `1.016`, `1.011`, `0.994`, `0.974`, `1.003`, `1.001`,
`1.012`, and `0.998`. Aggregate improvement was approximately 0.09%, which is
noise. Both experiments were removed; the production queue retains the two
targeted runtime copies.

## One-Pass Eight-Vector NEON Kernel

The post-SoA CPU profile moved the primary bottleneck into
`L2Distance8PtrNEON` at 57.5% flat CPU. The original eight-vector kernel ran
two independent four-vector passes, loading the query twice and maintaining
four accumulators per candidate.

The production ARM64 path now uses a specialized one-pass kernel when the
dimension is divisible by 16:

- All eight candidate pointers advance in one loop.
- The query's 16-float block is loaded once per iteration.
- Each candidate uses two independent accumulators.
- Candidates are processed in pairs, leaving four independent FMLAs between
  writes to the same accumulator.
- Dimensions not divisible by 16 retain the original exact fallback kernel.

Seven paired 768d kernel runs showed speedups from `1.135x` to `1.156x`.
Five full 5,000-vector interleaved search runs showed mean-latency speedups of
`1.102x`, `1.099x`, `1.122x`, `1.105x`, and `1.097x`. P50, p95, and p99
improved in every run, and recall@10 remained `1.000` throughout.

The benchmark initially reported `58 B/op` and two allocations per search.
Those came from reflection-based `sort.Slice` in the benchmark result helper,
not HNSW traversal. Replacing it with typed `slices.SortFunc` reduced the final
768d traversal result to `0 B/op` and `0 allocs/op`.

Using two rather than four accumulators changes floating-point summation order.
The largest observed 768d distance delta was approximately `0.0034` on a
distance near `4960.8`. Construction remained deterministic, but the level-0
link count changed from `840,725` to `840,707` (18 links out of more than
840,000). Recall@10 remained `1.000` in all construction runs.

After applying the same aligned kernel to diversity pruning and heuristic
candidate expansion, three alternating-order build runs improved graph
construction by `1.082x`, `1.097x`, and `1.086x`. Each run retained recall@10
of `1.000` and produced the same `840,707` level-0 links.
