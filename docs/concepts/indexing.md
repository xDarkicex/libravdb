# Indexing Algorithms

LibraVDB supports multiple indexing algorithms, each optimized for different use cases and performance characteristics. Understanding these algorithms helps you choose the right approach for your specific needs.

## Overview

LibraVDB provides three main indexing algorithms:

1. **HNSW** - Hierarchical Navigable Small World (recommended for most use cases)
2. **IVF-PQ** - Inverted File with Product Quantization (for large-scale, memory-constrained scenarios)
3. **Flat** - Brute-force exact search (for small collections or when exact results are required)

## HNSW (Hierarchical Navigable Small World)

HNSW is a graph-based algorithm that provides excellent performance for most vector search applications.

### How HNSW Works

HNSW builds a multi-layer graph where:
- Each vector is a node in the graph
- Nodes are connected to their nearest neighbors
- Higher layers have fewer nodes but longer connections
- Search navigates from top layer down to find approximate nearest neighbors

### Configuration

```go
collection, err := db.CreateCollection(ctx, "hnsw_collection",
    libravdb.WithDimension(768),
    libravdb.WithHNSW(32, 200, 50),
    // Parameters: M, EfConstruction, EfSearch
)
```

### Parameters

#### M (Max Connections)
Controls the maximum number of connections each node can have:

- **Low (16)**: Faster insertion, lower memory usage, slightly lower recall
- **Medium (32)**: Balanced performance (recommended)
- **High (64)**: Slower insertion, higher memory usage, better recall

```go
// Fast insertion, good for real-time applications
libravdb.WithHNSW(16, 200, 50)

// Balanced (recommended for most cases)
libravdb.WithHNSW(32, 200, 50)

// High accuracy, good for offline processing
libravdb.WithHNSW(64, 200, 50)
```

#### EfConstruction (Build-time Search Width)
Controls the search width during index construction:

- **Low (100-200)**: Faster building, lower index quality
- **Medium (200-400)**: Balanced (recommended)
- **High (400-800)**: Slower building, higher index quality

```go
// Fast building for development
libravdb.WithHNSW(32, 100, 50)

// Production balance
libravdb.WithHNSW(32, 200, 50)

// High quality for critical applications
libravdb.WithHNSW(32, 400, 50)
```

#### EfSearch (Query-time Search Width)
Controls the search width during queries (adjustable at runtime):

```go
// Fast queries, lower recall
results, err := collection.Query(ctx).
    WithVector(queryVector).
    WithEfSearch(50).
    Execute()

// Balanced
results, err := collection.Query(ctx).
    WithVector(queryVector).
    WithEfSearch(100).
    Execute()

// High recall, slower queries
results, err := collection.Query(ctx).
    WithVector(queryVector).
    WithEfSearch(200).
    Execute()
```

### Performance Characteristics

**Strengths:**
- Excellent query performance (sub-millisecond for millions of vectors)
- Good recall with proper parameter tuning
- Scales well with collection size
- Supports incremental updates

**Considerations:**
- Memory usage grows with collection size
- Parameter tuning affects performance significantly
- Build time increases with EfConstruction

### Use Cases

- **Real-time search applications**
- **Medium to large collections (10K - 10M vectors)**
- **When query latency is critical**
- **General-purpose vector search**

## IVF-PQ (Inverted File with Product Quantization)

IVF-PQ combines clustering with quantization for memory-efficient search at scale.

### How IVF-PQ Works

1. **Clustering**: Vectors are clustered into groups (cells)
2. **Quantization**: Vectors are compressed using Product Quantization
3. **Search**: Only relevant clusters are searched, using quantized vectors

### Configuration

```go
// IVF-PQ is automatically configured when using auto-selection
collection, err := db.CreateCollection(ctx, "large_collection",
    libravdb.WithAutoIndexSelection(true),
    libravdb.WithDimension(768),
)

// Or configure manually through quantization
collection, err := db.CreateCollection(ctx, "ivfpq_collection",
    libravdb.WithDimension(768),
    libravdb.WithProductQuantization(8, 8, 0.1),
)
```

### Parameters

#### NClusters (Number of Clusters)
Controls how many clusters to create:

- **Formula**: Typically `sqrt(N)` where N is the number of vectors
- **Range**: 100-10,000 clusters
- **Trade-off**: More clusters = better accuracy, higher memory usage

#### NProbes (Search Probes)
Controls how many clusters to search:

- **Low (1-5)**: Fast search, lower recall
- **Medium (10-20)**: Balanced
- **High (50-100)**: Slower search, higher recall

### Performance Characteristics

**Strengths:**
- Very memory efficient (8-32x compression)
- Scales to billions of vectors
- Good for batch processing
- Handles large datasets that don't fit in memory

**Considerations:**
- Slightly lower accuracy than HNSW
- Requires training phase
- Less suitable for real-time updates
- Query latency higher than HNSW

### Use Cases

- **Very large collections (>10M vectors)**
- **Memory-constrained environments**
- **Batch processing workloads**
- **When storage cost is a primary concern**

## Flat Index

Flat index performs exact brute-force search by comparing the query vector against all stored vectors.

### Configuration

```go
collection, err := db.CreateCollection(ctx, "exact_collection",
    libravdb.WithDimension(768),
    libravdb.WithFlat(),
)
```

### Performance Characteristics

**Strengths:**
- 100% recall (exact results)
- Simple and reliable
- No parameter tuning required
- Good for small collections

**Considerations:**
- Linear time complexity O(n)
- Memory usage grows linearly with collection size
- Becomes slow with large collections

### Use Cases

- **Small collections (<10K vectors)**
- **When exact results are required**
- **Baseline accuracy measurements**
- **Development and testing**

## Automatic Index Selection

LibraVDB can automatically choose the best index type based on collection size:

```go
collection, err := db.CreateCollection(ctx, "adaptive",
    libravdb.WithAutoIndexSelection(true),
    libravdb.WithDimension(768),
)
```

### Selection Logic

- **<10K vectors**: Flat index (exact search)
- **10K-1M vectors**: HNSW index (balanced performance)
- **>1M vectors**: IVF-PQ index (memory efficiency)

### Benefits

- Optimal performance at any scale
- No manual parameter tuning required
- Automatic transitions as collection grows
- Good default for most applications

## Index Comparison

| Algorithm | Memory Usage | Query Speed | Accuracy | Build Time | Use Case |
|-----------|--------------|-------------|----------|------------|----------|
| Flat | High | Slow (large) | 100% | Fast | Small collections |
| HNSW | Medium | Fast | 95-99% | Medium | General purpose |
| IVF-PQ | Low | Medium | 90-95% | Slow | Large scale |

## Performance Tuning by Algorithm

### HNSW Tuning

```go
// For real-time applications (prioritize speed)
libravdb.WithHNSW(16, 100, 50)

// For high accuracy (prioritize recall)
libravdb.WithHNSW(64, 400, 100)

// For balanced performance (recommended)
libravdb.WithHNSW(32, 200, 75)
```

### IVF-PQ Tuning

```go
// Memory-constrained (aggressive compression)
libravdb.WithProductQuantization(32, 4, 0.05)

// Balanced (recommended)
libravdb.WithProductQuantization(16, 8, 0.1)

// Accuracy-focused (less compression)
libravdb.WithProductQuantization(8, 8, 0.2)
```

## Choosing the Right Index

### Decision Matrix

**Choose Flat when:**
- Collection size < 10,000 vectors
- Exact results are required
- Memory is abundant
- Simplicity is preferred

**Choose HNSW when:**
- Collection size: 10K - 10M vectors
- Query latency is critical
- Good balance of speed and accuracy needed
- Real-time updates are required

**Choose IVF-PQ when:**
- Collection size > 1M vectors
- Memory is constrained
- Batch processing is acceptable
- Storage cost is a concern

**Choose Auto-selection when:**
- Collection size varies or is unknown
- Want optimal performance without tuning
- Building a general-purpose application
- Prefer simplicity over control

### Performance Guidelines

#### Memory Requirements

```go
// Estimate memory usage
vectorCount := 1000000
dimension := 768
bytesPerVector := dimension * 4 // float32

// Flat index
flatMemory := vectorCount * bytesPerVector

// HNSW index (approximate)
hnswMemory := flatMemory * 1.5 // 50% overhead for graph

// IVF-PQ index (with 8x compression)
ivfpqMemory := flatMemory / 8
```

#### Query Performance

```go
// Typical query times (1M vectors, 768 dimensions)
// Flat: 10-100ms
// HNSW: 0.1-1ms
// IVF-PQ: 1-10ms
```

## Advanced Index Features

### Quantization Integration

All index types support quantization for memory efficiency:

```go
// HNSW with quantization
collection, err := db.CreateCollection(ctx, "hnsw_quantized",
    libravdb.WithHNSW(32, 200, 50),
    libravdb.WithProductQuantization(8, 8, 0.1),
)

// Flat with quantization
collection, err := db.CreateCollection(ctx, "flat_quantized",
    libravdb.WithFlat(),
    libravdb.WithScalarQuantization(8, 0.1),
)
```

### Memory Mapping

Large indexes can use memory mapping:

```go
collection, err := db.CreateCollection(ctx, "memory_mapped",
    libravdb.WithHNSW(32, 200, 50),
    libravdb.WithMemoryMapping(true),
)
```

### Index Persistence

Indexes are automatically persisted and can be saved/loaded:

```go
// Save index to disk
err = collection.SaveIndex(ctx, "/path/to/index")

// Load index from disk
err = collection.LoadIndex(ctx, "/path/to/index")

// Get index metadata
metadata := collection.GetIndexMetadata()
```

## Best Practices

### 1. Start with Auto-selection
Begin with automatic index selection and optimize based on actual performance:

```go
collection, err := db.CreateCollection(ctx, "initial",
    libravdb.WithAutoIndexSelection(true),
)
```

### 2. Benchmark with Your Data
Test different configurations with your actual data and query patterns:

```go
func benchmarkIndex(collection *libravdb.Collection, queries [][]float32) {
    start := time.Now()
    for _, query := range queries {
        collection.Search(ctx, query, 10)
    }
    fmt.Printf("Average query time: %v\n", time.Since(start)/time.Duration(len(queries)))
}
```

### 3. Monitor Performance
Track index performance over time:

```go
stats := collection.Stats()
fmt.Printf("Index type: %s\n", stats.IndexType)
fmt.Printf("Memory usage: %d MB\n", stats.MemoryUsage/1024/1024)
fmt.Printf("Vector count: %d\n", stats.VectorCount)
```

### 4. Optimize Based on Usage
Adjust parameters based on your specific requirements:

- **Latency-critical**: Lower EfSearch, smaller M
- **Accuracy-critical**: Higher EfSearch, larger M
- **Memory-constrained**: Enable quantization, use IVF-PQ
- **Large-scale**: Use auto-selection, enable memory mapping