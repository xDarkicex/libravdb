# Collections

Collections are the fundamental organizational unit in LibraVDB. Each collection stores vectors of the same dimensionality and provides a consistent interface for insertion, search, and management operations.

## What is a Collection?

A collection is a named container that holds:
- **Vectors**: Fixed-dimension float32 arrays
- **Metadata**: Optional key-value pairs for each vector
- **Index**: Optimized data structure for fast similarity search
- **Configuration**: Settings that define behavior and performance characteristics

## Collection Lifecycle

### 1. Creation

```go
collection, err := db.CreateCollection(
    context.Background(),
    "my_collection",
    libravdb.WithDimension(768),
    libravdb.WithMetric(libravdb.CosineDistance),
)
```

### 2. Population

```go
// Insert individual vectors
err = collection.Insert(ctx, "doc1", vector, metadata)

// Batch insertions
stream := collection.NewStreamingBatchInsert(opts)
for _, entry := range entries {
    stream.Send(entry)
}
```

### 3. Querying

```go
// Simple search
results, err := collection.Search(ctx, queryVector, 10)

// Advanced filtering
results, err := collection.Query(ctx).
    WithVector(queryVector).
    Eq("category", "documents").
    Limit(10).
    Execute()
```

### 4. Management

```go
// Get statistics
stats := collection.Stats()

// Optimize performance
err = collection.OptimizeCollection(ctx, options)

// Close when done
err = collection.Close()
```

## Collection Configuration

### Basic Configuration

```go
collection, err := db.CreateCollection(ctx, "basic",
    libravdb.WithDimension(768),                    // Vector size
    libravdb.WithMetric(libravdb.CosineDistance),   // Distance function
)
```

### Index Configuration

Collections support multiple indexing algorithms:

#### HNSW (Hierarchical Navigable Small World)
Best for most use cases:

```go
libravdb.WithHNSW(32, 200, 50)
// M: max connections per node
// EfConstruction: build-time search width
// EfSearch: query-time search width
```

#### Flat Index
Exact search for small collections:

```go
libravdb.WithFlat()
```

#### Automatic Selection
Let LibraVDB choose the optimal index:

```go
libravdb.WithAutoIndexSelection(true)
```

### Memory Management

```go
collection, err := db.CreateCollection(ctx, "memory_managed",
    libravdb.WithMemoryLimit(2*1024*1024*1024), // 2GB limit
    libravdb.WithMemoryMapping(true),            // Enable memory mapping
    libravdb.WithCachePolicy(libravdb.LRUCache), // Cache eviction policy
)
```

### Quantization

Reduce memory usage with quantization:

```go
// Product Quantization
libravdb.WithProductQuantization(8, 8, 0.1)

// Scalar Quantization
libravdb.WithScalarQuantization(8, 0.1)
```

## Distance Metrics

Choose the appropriate distance metric for your data:

### Cosine Distance
Best for normalized embeddings (most common):

```go
libravdb.WithMetric(libravdb.CosineDistance)
```

**Use when:**
- Working with text embeddings
- Vectors are normalized or should be treated as directions
- Magnitude is less important than direction

### L2 (Euclidean) Distance
Standard geometric distance:

```go
libravdb.WithMetric(libravdb.L2Distance)
```

**Use when:**
- Working with image embeddings
- Magnitude matters
- Natural geometric interpretation needed

### Inner Product
Dot product similarity:

```go
libravdb.WithMetric(libravdb.InnerProduct)
```

**Use when:**
- Working with non-normalized embeddings
- Want to favor larger magnitude vectors
- Implementing custom similarity functions

## Metadata and Filtering

### Schema Definition

Define metadata structure for validation and optimization:

```go
schema := libravdb.MetadataSchema{
    "title":     libravdb.StringField,
    "category":  libravdb.StringField,
    "score":     libravdb.FloatField,
    "tags":      libravdb.StringArrayField,
    "created":   libravdb.TimeField,
    "published": libravdb.BoolField,
}

collection, err := db.CreateCollection(ctx, "structured",
    libravdb.WithMetadataSchema(schema),
    libravdb.WithIndexedFields("category", "score"), // Index for fast filtering
)
```

### Supported Field Types

- `StringField`: Text values
- `IntField`: Integer numbers
- `FloatField`: Floating-point numbers
- `BoolField`: Boolean values
- `TimeField`: Timestamps
- `StringArrayField`: Array of strings
- `IntArrayField`: Array of integers
- `FloatArrayField`: Array of floats

### Filtering Examples

```go
// Simple equality
results, err := collection.Query(ctx).
    WithVector(queryVector).
    Eq("category", "documents").
    Execute()

// Range filtering
results, err := collection.Query(ctx).
    WithVector(queryVector).
    Between("score", 0.8, 1.0).
    Execute()

// Complex logical operations
results, err := collection.Query(ctx).
    WithVector(queryVector).
    And().
        Eq("category", "documents").
        Or().
            Gt("score", 0.9).
            ContainsAny("tags", []interface{}{"important", "urgent"}).
        End().
    End().
    Execute()
```

## Collection Statistics

Monitor collection health and performance:

```go
stats := collection.Stats()

fmt.Printf("Collection: %s\n", stats.Name)
fmt.Printf("Vectors: %d\n", stats.VectorCount)
fmt.Printf("Dimension: %d\n", stats.Dimension)
fmt.Printf("Index Type: %s\n", stats.IndexType)
fmt.Printf("Memory Usage: %d MB\n", stats.MemoryUsage/1024/1024)

// Memory statistics (if memory management is enabled)
if stats.MemoryStats != nil {
    fmt.Printf("Memory Limit: %d MB\n", stats.MemoryStats.Limit/1024/1024)
    fmt.Printf("Memory Available: %d MB\n", stats.MemoryStats.Available/1024/1024)
    fmt.Printf("Pressure Level: %s\n", stats.MemoryStats.PressureLevel)
}

// Optimization status
if stats.OptimizationStatus != nil {
    fmt.Printf("Can Optimize: %v\n", stats.OptimizationStatus.CanOptimize)
    fmt.Printf("Last Optimization: %v\n", stats.OptimizationStatus.LastOptimization)
}
```

## Collection Optimization

### Automatic Optimization

```go
// Basic optimization
err = collection.OptimizeCollection(ctx, nil)

// Custom optimization options
options := &libravdb.OptimizationOptions{
    RebuildIndex:       true,
    OptimizeMemory:     true,
    CompactStorage:     true,
    UpdateQuantization: false,
}
err = collection.OptimizeCollection(ctx, options)
```

### Memory Optimization

```go
// Set memory limit
err = collection.SetMemoryLimit(4 * 1024 * 1024 * 1024) // 4GB

// Get current memory usage
usage, err := collection.GetMemoryUsage()
fmt.Printf("Total: %d MB\n", usage.Total/1024/1024)
fmt.Printf("Index: %d MB\n", usage.Indices/1024/1024)
fmt.Printf("Cache: %d MB\n", usage.Caches/1024/1024)

// Trigger garbage collection
err = collection.TriggerGC()
```

### Index Optimization

Collections can automatically switch index types based on size:

```go
// Enable automatic index selection
collection, err := db.CreateCollection(ctx, "adaptive",
    libravdb.WithAutoIndexSelection(true),
)

// The collection will automatically use:
// - Flat index for <10K vectors
// - HNSW index for 10K-1M vectors
// - IVF-PQ index for >1M vectors
```

## Best Practices

### 1. Choose Appropriate Dimensions
- Use the same dimension as your embedding model
- Common dimensions: 128, 256, 384, 512, 768, 1024, 1536
- Higher dimensions = more memory usage and slower operations

### 2. Select the Right Distance Metric
- Cosine distance for most text embeddings
- L2 distance for image embeddings
- Inner product for custom similarity functions

### 3. Configure Memory Appropriately
- Set memory limits based on available system resources
- Enable memory mapping for large collections
- Use quantization to reduce memory usage

### 4. Design Metadata Schema
- Define schema for validation and optimization
- Index frequently filtered fields
- Keep metadata lightweight

### 5. Monitor Performance
- Check collection statistics regularly
- Monitor memory usage and pressure
- Optimize when performance degrades

### 6. Use Appropriate Index Types
- Start with HNSW for most use cases
- Use Flat for small collections or exact search
- Enable auto-selection for varying collection sizes

### 7. Batch Operations
- Use streaming for large insertions
- Configure appropriate batch sizes
- Monitor insertion rates and adjust concurrency

## Common Patterns

### Document Search Collection

```go
collection, err := db.CreateCollection(ctx, "documents",
    libravdb.WithDimension(768),                    // Common text embedding size
    libravdb.WithMetric(libravdb.CosineDistance),   // Good for text
    libravdb.WithHNSW(32, 200, 100),               // Balanced performance
    libravdb.WithMetadataSchema(libravdb.MetadataSchema{
        "title":    libravdb.StringField,
        "content":  libravdb.StringField,
        "category": libravdb.StringField,
        "tags":     libravdb.StringArrayField,
    }),
    libravdb.WithIndexedFields("category"),         // Fast category filtering
)
```

### Image Similarity Collection

```go
collection, err := db.CreateCollection(ctx, "images",
    libravdb.WithDimension(512),                    // Common image embedding size
    libravdb.WithMetric(libravdb.L2Distance),       // Good for images
    libravdb.WithMemoryMapping(true),               // Handle large image datasets
    libravdb.WithProductQuantization(8, 8, 0.1),   // Reduce memory usage
)
```

### High-throughput Collection

```go
collection, err := db.CreateCollection(ctx, "high_throughput",
    libravdb.WithDimension(256),
    libravdb.WithAutoIndexSelection(true),          // Adapt to size
    libravdb.WithBatchChunkSize(5000),              // Large batches
    libravdb.WithBatchConcurrency(16),              // High concurrency
    libravdb.WithMemoryLimit(16*1024*1024*1024),    // 16GB limit
)
```