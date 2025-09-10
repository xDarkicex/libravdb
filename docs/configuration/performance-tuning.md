# Performance Tuning Guide

This guide provides comprehensive strategies for optimizing LibraVDB performance across different use cases and scales.

## Performance Overview

LibraVDB performance depends on several factors:
- **Index Type**: HNSW vs IVF-PQ vs Flat
- **Memory Management**: Limits, caching, memory mapping
- **Quantization**: Product vs Scalar quantization
- **Hardware**: CPU, RAM, storage type
- **Data Characteristics**: Dimension, distribution, size

## Benchmarking Your Setup

Before optimizing, establish baseline performance:

```go
func benchmarkCollection(collection *libravdb.Collection) {
    // Insertion benchmark
    start := time.Now()
    for i := 0; i < 10000; i++ {
        vector := generateRandomVector(768)
        collection.Insert(ctx, fmt.Sprintf("vec_%d", i), vector, nil)
    }
    insertTime := time.Since(start)
    fmt.Printf("Insertion: %.2f vectors/sec\n", 10000.0/insertTime.Seconds())

    // Search benchmark
    queryVector := generateRandomVector(768)
    start = time.Now()
    for i := 0; i < 1000; i++ {
        collection.Search(ctx, queryVector, 10)
    }
    searchTime := time.Since(start)
    fmt.Printf("Search: %.2f queries/sec\n", 1000.0/searchTime.Seconds())
}
```

## Index Optimization

### HNSW Parameter Tuning

The HNSW algorithm has three key parameters that affect performance:

#### M (Max Connections)
Controls the connectivity of the graph:

```go
// Fast insertion, lower recall
libravdb.WithHNSW(16, 200, 50)

// Balanced (recommended for most cases)
libravdb.WithHNSW(32, 200, 50)

// High recall, slower insertion
libravdb.WithHNSW(64, 200, 50)
```

**Guidelines:**
- M=16: 2-3x faster insertion, 5-10% lower recall
- M=32: Balanced performance (recommended)
- M=64: 2x slower insertion, 2-5% higher recall

#### EfConstruction (Build-time Search Width)
Controls index quality during construction:

```go
// Fast building, lower quality
libravdb.WithHNSW(32, 100, 50)

// Balanced
libravdb.WithHNSW(32, 200, 50)

// High quality, slower building
libravdb.WithHNSW(32, 400, 50)
```

**Guidelines:**
- 100-200: Fast building for development/testing
- 200-400: Production balance
- 400-800: Maximum quality for critical applications

#### EfSearch (Query-time Search Width)
Adjustable at query time for speed/accuracy tradeoff:

```go
// Fast queries, lower recall
collection.Query(ctx).WithVector(vector).WithEfSearch(50).Execute()

// Balanced
collection.Query(ctx).WithVector(vector).WithEfSearch(100).Execute()

// High recall, slower queries
collection.Query(ctx).WithVector(vector).WithEfSearch(200).Execute()
```

### Index Type Selection

#### Automatic Selection
Let LibraVDB choose based on collection size:

```go
collection, err := db.CreateCollection(ctx, "adaptive",
    libravdb.WithAutoIndexSelection(true),
    libravdb.WithDimension(768),
)
```

#### Manual Selection Guidelines

**Flat Index** - Use when:
- Collection size < 10,000 vectors
- Exact results required
- Memory is abundant
- Query latency is not critical

```go
libravdb.WithFlat()
```

**HNSW Index** - Use when:
- Collection size: 10K - 1M vectors
- Need balance of speed and accuracy
- Memory is moderate
- Sub-millisecond queries required

```go
libravdb.WithHNSW(32, 200, 100)
```

**IVF-PQ Index** - Use when:
- Collection size > 1M vectors
- Memory is constrained
- Can tolerate slightly lower accuracy
- Batch queries are common

```go
// IVF-PQ is automatically selected with auto-selection for large collections
```

## Memory Optimization

### Memory Limits and Monitoring

Set appropriate memory limits based on available system resources:

```go
// Conservative: 25% of system RAM
collection, err := db.CreateCollection(ctx, "conservative",
    libravdb.WithMemoryLimit(systemRAM / 4),
)

// Aggressive: 75% of system RAM
collection, err := db.CreateCollection(ctx, "aggressive",
    libravdb.WithMemoryLimit(systemRAM * 3 / 4),
)
```

Monitor memory usage:

```go
stats := collection.Stats()
if stats.MemoryStats != nil {
    utilization := float64(stats.MemoryStats.Total) / float64(stats.MemoryStats.Limit)
    if utilization > 0.8 {
        fmt.Println("High memory utilization, consider optimization")
    }
}
```

### Memory Mapping

Enable memory mapping for large collections:

```go
collection, err := db.CreateCollection(ctx, "large_collection",
    libravdb.WithMemoryMapping(true),
    libravdb.WithMemoryLimit(32*1024*1024*1024), // 32GB
)
```

**Benefits:**
- Handle datasets larger than RAM
- Automatic OS-level caching
- Reduced memory pressure

**Considerations:**
- Requires fast storage (SSD recommended)
- Initial access may be slower
- OS manages memory more efficiently

### Cache Optimization

Choose the right cache policy:

```go
// LRU: Good for general workloads
libravdb.WithCachePolicy(libravdb.LRUCache)

// LFU: Good for stable access patterns
libravdb.WithCachePolicy(libravdb.LFUCache)

// FIFO: Good for streaming/sequential access
libravdb.WithCachePolicy(libravdb.FIFOCache)
```

## Quantization for Memory Efficiency

### Product Quantization

Reduces memory usage by 4-32x with minimal accuracy loss:

```go
// Conservative: 8 codebooks, 8 bits (8x compression)
libravdb.WithProductQuantization(8, 8, 0.1)

// Balanced: 16 codebooks, 8 bits (16x compression)
libravdb.WithProductQuantization(16, 8, 0.1)

// Aggressive: 32 codebooks, 4 bits (32x compression)
libravdb.WithProductQuantization(32, 4, 0.1)
```

**Performance Impact:**
- Memory: 8-32x reduction
- Search Speed: 10-20% slower
- Accuracy: 1-5% lower recall

### Scalar Quantization

Simpler quantization with good performance:

```go
// 8-bit quantization (4x compression)
libravdb.WithScalarQuantization(8, 0.1)

// 4-bit quantization (8x compression)
libravdb.WithScalarQuantization(4, 0.1)
```

**When to Use:**
- Simpler than Product Quantization
- Good for uniform data distributions
- Less memory savings than PQ but faster

## Batch Processing Optimization

### Streaming Insertions

For high-throughput insertions:

```go
opts := &libravdb.StreamingOptions{
    BufferSize:     50000,           // Large buffer
    ChunkSize:      5000,            // Large chunks
    MaxConcurrency: runtime.NumCPU(), // Use all CPUs
    Timeout:        5 * time.Minute,
}

stream := collection.NewStreamingBatchInsert(opts)
```

### Batch Size Tuning

Optimize batch sizes based on your data:

```go
// Small vectors (<100 dimensions)
libravdb.WithBatchChunkSize(10000)

// Medium vectors (100-1000 dimensions)
libravdb.WithBatchChunkSize(5000)

// Large vectors (>1000 dimensions)
libravdb.WithBatchChunkSize(1000)
```

## Hardware Optimization

### CPU Optimization

LibraVDB benefits from multiple CPU cores:

```go
// Use all available CPUs for batch operations
libravdb.WithBatchConcurrency(runtime.NumCPU())

// For systems with many cores, limit to avoid contention
maxConcurrency := min(runtime.NumCPU(), 16)
libravdb.WithBatchConcurrency(maxConcurrency)
```

### Storage Optimization

**SSD vs HDD:**
- SSD: 10-100x faster for random access
- Required for memory mapping performance
- Recommended for all production deployments

**Storage Configuration:**
```go
// For SSD storage
db, err := libravdb.New(
    libravdb.WithStoragePath("/fast/ssd/path"),
)

// Enable memory mapping with SSD
collection, err := db.CreateCollection(ctx, "ssd_optimized",
    libravdb.WithMemoryMapping(true),
)
```

### Memory Configuration

**System Memory Guidelines:**
- Minimum: 2x your largest collection size
- Recommended: 4x your largest collection size
- With quantization: 1.5x uncompressed size

```go
// Calculate memory needs
vectorCount := 1000000
dimension := 768
bytesPerVector := dimension * 4 // float32
uncompressedSize := vectorCount * bytesPerVector

// With 8x quantization
quantizedSize := uncompressedSize / 8
recommendedRAM := quantizedSize * 2 // 2x for overhead
```

## Query Optimization

### Search Parameter Tuning

Adjust search parameters based on requirements:

```go
// Fast queries (lower accuracy)
results, err := collection.Query(ctx).
    WithVector(queryVector).
    WithEfSearch(50).
    Limit(10).
    Execute()

// Balanced
results, err := collection.Query(ctx).
    WithVector(queryVector).
    WithEfSearch(100).
    Limit(10).
    Execute()

// High accuracy (slower)
results, err := collection.Query(ctx).
    WithVector(queryVector).
    WithEfSearch(200).
    Limit(10).
    Execute()
```

### Filter Optimization

Optimize filters for better performance:

```go
// Index frequently filtered fields
collection, err := db.CreateCollection(ctx, "filtered",
    libravdb.WithMetadataSchema(schema),
    libravdb.WithIndexedFields("category", "priority"), // Index these fields
)

// Use selective filters first
results, err := collection.Query(ctx).
    WithVector(queryVector).
    Eq("category", "documents").     // Selective filter first
    Gt("score", 0.8).               // Less selective filter second
    Execute()
```

## Monitoring and Profiling

### Performance Metrics

Monitor key performance indicators:

```go
// Collection-level metrics
stats := collection.Stats()
fmt.Printf("Vectors: %d\n", stats.VectorCount)
fmt.Printf("Memory: %d MB\n", stats.MemoryUsage/1024/1024)
fmt.Printf("Index Type: %s\n", stats.IndexType)

// Memory pressure monitoring
if stats.MemoryStats != nil {
    pressure := float64(stats.MemoryStats.Total) / float64(stats.MemoryStats.Limit)
    fmt.Printf("Memory Pressure: %.1f%%\n", pressure*100)
}
```

### Database-level Monitoring

```go
// Global statistics
dbStats := db.Stats()
fmt.Printf("Collections: %d\n", dbStats.CollectionCount)
fmt.Printf("Total Memory: %d MB\n", dbStats.MemoryUsage/1024/1024)

// Health monitoring
health, err := db.Health(ctx)
if err == nil {
    fmt.Printf("Database Health: %s\n", health.Status)
}
```

## Performance Profiles by Use Case

### Real-time Search Applications

```go
collection, err := db.CreateCollection(ctx, "realtime",
    libravdb.WithDimension(768),
    libravdb.WithHNSW(32, 200, 50),     // Fast queries
    libravdb.WithMemoryLimit(8*GB),      // Keep in memory
    // No quantization for maximum speed
)
```

### Large-scale Batch Processing

```go
collection, err := db.CreateCollection(ctx, "batch_processing",
    libravdb.WithDimension(768),
    libravdb.WithAutoIndexSelection(true), // Adapt to size
    libravdb.WithMemoryMapping(true),      // Handle large datasets
    libravdb.WithProductQuantization(16, 8, 0.1), // Memory efficiency
    libravdb.WithBatchChunkSize(10000),    // Large batches
    libravdb.WithBatchConcurrency(16),     // High concurrency
)
```

### Memory-constrained Environments

```go
collection, err := db.CreateCollection(ctx, "memory_constrained",
    libravdb.WithDimension(768),
    libravdb.WithHNSW(16, 100, 50),       // Smaller index
    libravdb.WithMemoryLimit(2*GB),        // Strict limit
    libravdb.WithMemoryMapping(true),      // Use disk when needed
    libravdb.WithProductQuantization(32, 4, 0.1), // Aggressive compression
)
```

### High-accuracy Applications

```go
collection, err := db.CreateCollection(ctx, "high_accuracy",
    libravdb.WithDimension(768),
    libravdb.WithHNSW(64, 800, 200),      // High-quality index
    libravdb.WithMemoryLimit(32*GB),       // Generous memory
    // No quantization for exact vectors
)
```

## Troubleshooting Performance Issues

### Slow Insertions
1. Reduce EfConstruction parameter
2. Use smaller M parameter
3. Increase batch sizes
4. Enable streaming insertions

### Slow Queries
1. Reduce EfSearch parameter
2. Enable quantization
3. Add memory mapping
4. Optimize filters

### High Memory Usage
1. Enable quantization
2. Set memory limits
3. Use memory mapping
4. Reduce cache sizes

### Poor Accuracy
1. Increase EfSearch parameter
2. Increase M parameter
3. Disable quantization
4. Use Flat index for small collections