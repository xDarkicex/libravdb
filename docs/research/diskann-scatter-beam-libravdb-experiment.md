# DiskANN Scatter-Beam Experiment in LibraVDB

Date: 2026-07-11

## Source Behavior

The implementation was checked against Microsoft DiskANN's current
`diskann/src/graph/index.rs::search_internal` implementation, not only the
2019 paper.

DiskANN separates:

- `L`: the number of best candidates retained by the search queue.
- `W`: the number of closest unvisited candidates expanded in one round.

For each round it marks up to `W` candidates visited, expands and deduplicates
the union of their neighborhoods, computes distances for that union, and then
inserts the results into its sorted candidate queue. The paper recommends
`W=2,4,8` for SSD search and warns that `W>=16` wastes I/O and compute.

DiskANN's main benefit is I/O amortization: several SSD neighborhood reads are
issued per round. Its candidate queue is also materially different from
LibraVDB's two-heap HNSW queue: DiskANN uses a sorted structure-of-arrays queue
with visited flags and a cursor, including SIMD lower-bound scans.

## LibraVDB Prototype

The prototype used exact squared-L2 distances and existing ARM64 NEON kernels.
No compact or approximate distance was introduced. Neighbor IDs were
deduplicated through the existing visited marks before one bulk scoring pass.
All temporary IDs, pointers, and visited state remained off-heap; the expansion
wave was a fixed stack array.

Fixture:

- Apple M2, ARM64
- 5,000 normalized random vectors
- 768 dimensions
- `M=36`
- `efConstruction=200`
- `efSearch=200`
- recall@10 measured against exact brute force

## Results

All tested widths retained `1.000 recall@10`.

### Search

An initial 100-query sweep falsely favored `W=4` because width 1 encountered a
scheduler-scale p99 outlier. A 1,000-query run showed `W=4` and `W=8` were clear
regressions. Repeated `W=1` versus `W=2` runs were then affected by thermal
ordering, so a paired benchmark alternated execution order for every query.

Three paired 1,000-query runs:

| Run | W=1 mean | W=2 mean | W=1 p50 | W=2 p50 | W=1 p99 | W=2 p99 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.312 ms | 1.289 ms | 1.225 ms | 1.210 ms | 2.608 ms | 2.405 ms |
| 2 | 0.975 ms | 0.968 ms | 0.954 ms | 0.944 ms | 1.218 ms | 1.284 ms |
| 3 | 1.276 ms | 1.353 ms | 1.137 ms | 1.122 ms | 3.338 ms | 3.891 ms |

`W=2` improved p50 by roughly 1% but produced a slightly worse aggregate mean
and worse p99 in two of three paired runs. That is below the threshold for a
production scheduler change.

### Construction

The first width sweep suggested `W=4` improved construction by 5.5%, but paired
repeats reversed that result:

- `W=1`: 694.6, 641.3, and 669.1 inserts/s
- `W=4`: 639.6, 629.4, and 638.5 inserts/s

The repeated averages are approximately 668 inserts/s for `W=1` and 636
inserts/s for `W=4`, making scatter construction about 5% slower.

## Decision

Do not integrate DiskANN scatter beam into in-memory LibraVDB HNSW.

The method successfully amortizes high-latency SSD reads in DiskANN. In
LibraVDB, adjacency and vectors are already in off-heap memory and scored by
8-wide exact NEON kernels. Expanding extra frontier nodes increases distance
work and heap admission enough to erase the batching benefit.

Potentially transferable DiskANN work remains its SIMD structure-of-arrays
candidate queue and bulk provider interface, but either must independently
beat LibraVDB's current d-ary heaps before integration.
