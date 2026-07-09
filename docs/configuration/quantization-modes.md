# Quantization Modes for Production

This page explains when to use raw vectors, Product Quantization, Scalar Quantization, and Finite Scalar Quantization in LibraVDB.

## Production Default

Use raw vectors with SIMD distance and exact final ranking unless memory pressure requires quantization.

```go
collection, err := db.CreateCollection(ctx, "vectors",
    libravdb.WithDimension(768),
    libravdb.WithMetric(libravdb.CosineDistance),
    libravdb.WithHNSW(32, 200, 100),
    libravdb.WithRawVectorStoreSlabby(4096),
)
```

This is the safest default because the search path ranks candidates with the same distance function used by the API result contract. It avoids quantization distortion during result ordering.

## Safety Rule

Do not use quantized distance as the final result order.

Quantized distances are useful for traversal and memory reduction, but they are approximate. Production quantized search should:

1. Traverse with raw SIMD, PQ, SQ, or FSQ distances.
2. Over-fetch candidates with a conservative `EfSearch`/internal beam.
3. Rerank the final candidate set against raw vectors.
4. Return exact raw-distance order.

LibraVDB keeps raw vectors available for final reranking when quantization is enabled. Do not disable raw-vector storage for production recall-sensitive collections.

## Mode Selection

| Mode | Use When | Avoid When |
|---|---|---|
| Raw HNSW | Recall quality is the priority; memory fits; general production default | Dataset is too large for available memory |
| Product Quantization | Read-heavy, memory-sensitive collections with many candidates scored per query | Heavy online ingestion or frequent cold queries |
| FSQ | Write-heavy, fast-build, codebook-free mode; cold-query workloads; avoiding k-means training | Per-query candidate scoring dominates and PQ LUT warmup is amortized |
| Scalar Quantization | Simple low-complexity compression baseline | You need the best search throughput at large candidate counts |

## Product Quantization

Product Quantization uses trained codebooks and query-specific lookup tables. It has expensive training and compression compared with FSQ, but after `PrepareQuery` builds the lookup table, per-candidate scoring is very fast.

Use PQ for large, read-heavy collections.

```go
collection, err := db.CreateCollection(ctx, "read_heavy",
    libravdb.WithDimension(768),
    libravdb.WithMetric(libravdb.CosineDistance),
    libravdb.WithHNSW(32, 400, 200),
    libravdb.WithProductQuantization(
        8,    // codebooks
        8,    // bits per code
        0.10, // training ratio
    ),
)
```

Recommended starting points:

| Goal | Codebooks | Bits | Train Ratio |
|---|---:|---:|---:|
| Balanced | 8 | 8 | 0.10 |
| More accuracy | 16 | 8 | 0.10-0.20 |
| Smaller codes | 8 | 4-6 | 0.10 |

PQ is usually the better search quantizer once each query scores enough candidates to amortize lookup-table preparation.

## Finite Scalar Quantization

Finite Scalar Quantization is codebook-free. It uses per-channel min/max normalization, bounds values, rounds to finite levels, and packs integer codes. There is no k-means training and no query lookup table.

Use FSQ for write-heavy or high-churn collections where build speed matters.

```go
collection, err := db.CreateCollection(ctx, "write_heavy",
    libravdb.WithDimension(768),
    libravdb.WithMetric(libravdb.CosineDistance),
    libravdb.WithHNSW(32, 200, 200),
    libravdb.WithFSQQuantization(
        6,    // default bits when explicit levels are omitted
        0.10, // training ratio for per-channel ranges
        8, 8, 8, 6, 5, // optional repeating FSQ level cycle
    ),
)
```

Use explicit levels when you want a structured codebook-free representation. Omit the levels for uniform `2^bits` levels per dimension:

```go
libravdb.WithFSQQuantization(6, 0.10)
```

FSQ is a good first quantizer for online ingestion because it avoids PQ's centroid training and codebook lookup work.

## Scalar Quantization

Scalar Quantization is the simplest compression mode. It linearly maps each dimension into fixed-width integer values using trained min/max ranges.

```go
collection, err := db.CreateCollection(ctx, "simple_compression",
    libravdb.WithDimension(768),
    libravdb.WithHNSW(32, 200, 100),
    libravdb.WithScalarQuantization(
        8,    // bits per dimension
        0.10, // training ratio
    ),
)
```

Use SQ as a simple baseline or when predictable behavior matters more than maximum throughput.

## Benchmark Shape

Short local benchmark sample on Apple M2, `D=128`, 200 ms benches:

| Operation | PQ 8x8 | SQ 8-bit | FSQ 6-bit/levels |
|---|---:|---:|---:|
| Train | ~79 ms | ~125 us | ~131 us |
| Compress | ~11.2 us | ~2.6 us | ~4.0 us |
| PrepareQuery | ~10 us | ~1.6 ns | ~1.7 ns |
| DistanceToQuery | ~141 ns | ~2.25 us | ~1.25 us |

Interpretation:

- PQ pays more up front, then scores candidates very quickly.
- FSQ has almost no query warmup and much faster compression than PQ.
- FSQ is not automatically faster than PQ for read-heavy HNSW search, because HNSW scores many candidates per query.
- Exact rerank remains required for all quantized modes.

Run local benchmarks before choosing:

```sh
go test -run '^$' -bench 'BenchmarkQuantizer(Train|Compress|PrepareQuery|DistanceToQuery)$' ./internal/quant
```

## Recommended Defaults

For most production deployments:

```go
libravdb.WithHNSW(32, 200, 100)
// no quantization
```

For memory-sensitive read-heavy deployments:

```go
libravdb.WithHNSW(32, 400, 200)
libravdb.WithProductQuantization(8, 8, 0.10)
```

For write-heavy or frequently rebuilt deployments:

```go
libravdb.WithHNSW(32, 200, 200)
libravdb.WithFSQQuantization(6, 0.10, 8, 8, 8, 6, 5)
```

For recall-sensitive systems, validate with brute-force ground truth before lowering `EfSearch`, `EfConstruction`, code size, or training ratio.
