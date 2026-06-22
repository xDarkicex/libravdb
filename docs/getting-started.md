# Getting Started with LibraVDB

This guide walks you through installing LibraVDB, creating your first database
and collection, inserting vectors, running searches, and configuring for
production.

## Installation

```bash
go get github.com/xDarkicex/libravdb
```

Requires Go 1.25+. No CGo, no system dependencies beyond the Go toolchain.

## Quick Start

### 1. Create a Database

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/xDarkicex/libravdb/libravdb"
)

func main() {
    db, err := libravdb.Open(
        libravdb.WithStoragePath("./my_data"),
    )
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()
}
```

### 2. Create a Collection

```go
col, err := db.CreateCollection(context.Background(), "embeddings",
    libravdb.WithDimension(768),
    libravdb.WithMetric(libravdb.CosineDistance),
    libravdb.WithHNSW(32, 200, 100),
)
if err != nil {
    log.Fatal(err)
}
```

### 3. Insert Vectors

```go
// Single insert
err = col.Insert(ctx, "doc-1", embedding, map[string]interface{}{
    "title": "Introduction to Vector Search",
    "score": 0.95,
})

// Batch insert
entries := []libravdb.VectorEntry{
    {ID: "vec-1", Vector: vec1},
    {ID: "vec-2", Vector: vec2},
}
err = col.InsertBatch(ctx, entries)
```

### 4. Search

```go
results, err := col.Search(ctx, queryEmbedding, 10)
if err != nil {
    log.Fatal(err)
}
for _, r := range results.Results {
    fmt.Printf("ID: %s  Score: %.4f\n", r.ID, r.Score)
}
```

## Choosing a Distance Metric

| Metric | Go Constant | Best For |
|--------|------------|----------|
| Cosine | `libravdb.CosineDistance` | Text embeddings, normalized vectors |
| Euclidean (L2) | `libravdb.L2Distance` | Image embeddings, geometric data |
| Inner Product | `libravdb.InnerProduct` | Custom similarity, non-normalized vectors |

```go
// Cosine distance for text embeddings (most common)
libravdb.WithMetric(libravdb.CosineDistance)

// L2 distance for image embeddings
libravdb.WithMetric(libravdb.L2Distance)
```

## Choosing an Index

```go
// HNSW — best for most use cases (10K–10M vectors)
libravdb.WithHNSW(32, 200, 100)

// Flat — exact results for small collections (<10K)
libravdb.WithFlat()

// IVF-PQ — memory-efficient for large collections (>1M)
libravdb.WithIVFPQ(1024, 64)

// Auto — let the library choose
libravdb.WithAutoIndexSelection(true)
```

## Metadata Filtering

Define a schema for type-safe filtering:

```go
schema := libravdb.MetadataSchema{
    "category": libravdb.StringField,
    "score":    libravdb.FloatField,
    "tags":     libravdb.StringArrayField,
}

col, err := db.CreateCollection(ctx, "docs",
    libravdb.WithDimension(768),
    libravdb.WithMetadataSchema(schema),
    libravdb.WithIndexedFields("category", "score"),
)
```

Query with filters:

```go
results, err := col.Query(ctx).
    WithVector(queryVec).
    Eq("category", "technology").
    Gt("score", 0.8).
    Limit(10).
    Execute()
```

## High-Throughput Ingestion

For large datasets, use the streaming API:

```go
opts := libravdb.DefaultStreamingOptions()
opts.ChunkSize = 2000
opts.MaxConcurrency = 8
opts.ProgressCallback = func(stats *libravdb.StreamingStats) {
    fmt.Printf("\r%.0f vectors/sec", stats.ItemsPerSecond)
}

stream := col.NewStreamingBatchInsert(opts)
stream.Start()

for _, entry := range largeDataset {
    stream.Send(&libravdb.VectorEntry{
        ID:     entry.ID,
        Vector: entry.Vector,
    })
}

stream.Close()
stream.Wait()
```

## Transactions

For atomic cross-collection mutations:

```go
err := db.WithTx(ctx, func(tx libravdb.Tx) error {
    if err := tx.Insert(ctx, "users", "u1", userVec, userMeta); err != nil {
        return err // rollback
    }
    if err := tx.Insert(ctx, "profiles", "p1", profileVec, nil); err != nil {
        return err // rollback
    }
    return nil // commit
})
```

## Memory Management

```go
col, err := db.CreateCollection(ctx, "large",
    libravdb.WithDimension(768),
    libravdb.WithMemoryLimit(8 * 1024 * 1024 * 1024), // 8 GB
    libravdb.WithMemoryMapping(true),                  // mmap for overflow
    libravdb.WithCachePolicy(libravdb.LRUCache),
)
```

Monitor memory:

```go
stats := col.Stats()
if stats.MemoryStats != nil {
    fmt.Printf("Memory: %d MB / %d MB (pressure: %s)\n",
        stats.MemoryStats.Total/1024/1024,
        stats.MemoryStats.Limit/1024/1024,
        stats.MemoryStats.PressureLevel,
    )
}
```

## Production Checklist

- [ ] Set `WithStoragePath` to an absolute path on fast storage (SSD).
- [ ] Enable `WithMetrics(true)` for Prometheus monitoring.
- [ ] Set `WithMemoryLimit` based on available system RAM (≤75%).
- [ ] Use `WithMemoryMapping(true)` for datasets larger than RAM.
- [ ] Define `WithMetadataSchema` for type-safe metadata.
- [ ] Index frequently-filtered metadata fields.
- [ ] Tune HNSW parameters (`M`, `EfConstruction`, `EfSearch`) for your recall/latency tradeoff.
- [ ] Use batch/streaming APIs for bulk ingestion, not individual inserts.
- [ ] Set `WithLogger` to capture index rebuild timing.
- [ ] Close the database gracefully on shutdown (`defer db.Close()`).

## Next Steps

- [API Reference](api-reference.md) — Complete public API documentation
- [Configuration Guide](configuration/configuration.md) — All configuration options
- [Performance Tuning](configuration/performance-tuning.md) — Optimization strategies
- [Collections](concepts/collections.md) — Collection lifecycle and management
- [Indexing](concepts/indexing.md) — Index algorithm selection and tuning
- [Design Documents](design/README.md) — Internal architecture and algorithms
