# Configuration Guide

This guide covers all configuration options available in LibraVDB for optimal performance and resource usage.

## Database Configuration

### Storage Configuration

```go
// Basic storage setup
db, err := libravdb.New(
    libravdb.WithStoragePath("./vector_data"),
)

// Production storage setup with monitoring
db, err := libravdb.New(
    libravdb.WithStoragePath("/var/lib/libravdb"),
    libravdb.WithMetrics(true),
    libravdb.WithTracing(true),
    libravdb.WithMaxCollections(1000),
)
```

#### Storage Path Considerations
- Use SSD storage for better performance
- Ensure sufficient disk space (vectors + indexes + metadata)
- Consider backup and replication strategies
- Use absolute paths in production

### Observability Configuration

```go
// Enable comprehensive monitoring
db, err := libravdb.New(
    libravdb.WithMetrics(true),    // Prometheus metrics
    libravdb.WithTracing(true),    // Distributed tracing
)
```

## Collection Configuration

### Basic Collection Setup

```go
collection, err := db.CreateCollection(ctx, "my_collection",
    libravdb.WithDimension(768),                    // Vector dimension
    libravdb.WithMetric(libravdb.CosineDistance),   // Distance metric
)
```

### Index Configuration

#### HNSW Index (Recommended)

```go
collection, err := db.CreateCollection(ctx, "hnsw_collection",
    libravdb.WithDimension(768),
    libravdb.WithHNSW(32, 200, 50),
    // Parameters:
    // M=32: Max connections per node (16-64 typical)
    // EfConstruction=200: Build-time search width (100-800)
    // EfSearch=50: Query-time search width (50-200)
)
```

**HNSW Parameter Guidelines:**
- **M (Max Connections)**:
  - 16: Fast insertion, lower recall
  - 32: Balanced (recommended)
  - 64: Slower insertion, higher recall
- **EfConstruction**:
  - 100-200: Fast building, lower recall
  - 200-400: Balanced (recommended)
  - 400-800: Slower building, higher recall
- **EfSearch**:
  - Adjust at query time for speed/accuracy tradeoff
  - Higher values = better recall, slower queries

#### Flat Index (Exact Search)

```go
collection, err := db.CreateCollection(ctx, "exact_collection",
    libravdb.WithDimension(768),
    libravdb.WithFlat(),
)
```

Use for:
- Small collections (<10K vectors)
- When exact results are required
- Baseline accuracy measurements

#### Automatic Index Selection

```go
collection, err := db.CreateCollection(ctx, "adaptive_collection",
    libravdb.WithDimension(768),
    libravdb.WithAutoIndexSelection(true),
)
```

LibraVDB automatically chooses:
- **Flat**: <10K vectors
- **HNSW**: 10K-1M vectors  
- **IVF-PQ**: >1M vectors

### Memory Management Configuration

#### Basic Memory Limits

```go
collection, err := db.CreateCollection(ctx, "memory_limited",
    libravdb.WithDimension(768),
    libravdb.WithMemoryLimit(2*1024*1024*1024), // 2GB limit
)
```

#### Advanced Memory Configuration

```go
memConfig := &memory.MemoryConfig{
    MaxMemory:       4 * 1024 * 1024 * 1024, // 4GB
    MonitorInterval: 30 * time.Second,        // Check every 30s
    GCThreshold:     0.8,                     // Trigger GC at 80%
    MMapThreshold:   1024 * 1024 * 1024,      // MMap files >1GB
    EnableMMap:      true,
}

collection, err := db.CreateCollection(ctx, "advanced_memory",
    libravdb.WithDimension(768),
    libravdb.WithMemoryConfig(memConfig),
    libravdb.WithMemoryMapping(true),
)
```

#### Cache Policies

```go
// LRU Cache (default)
libravdb.WithCachePolicy(libravdb.LRUCache)

// LFU Cache (for stable access patterns)
libravdb.WithCachePolicy(libravdb.LFUCache)

// FIFO Cache (for streaming workloads)
libravdb.WithCachePolicy(libravdb.FIFOCache)
```

### Quantization Configuration

#### Product Quantization

```go
collection, err := db.CreateCollection(ctx, "pq_collection",
    libravdb.WithDimension(768),
    libravdb.WithProductQuantization(
        8,    // codebooks (typically 8-16)
        8,    // bits per code (4-8)
        0.1,  // training ratio (5-20%)
    ),
)
```

**Product Quantization Guidelines:**
- **Codebooks**: More = better accuracy, more memory
- **Bits**: 4-8 bits typical, 8 bits recommended
- **Training Ratio**: 10% usually sufficient

#### Scalar Quantization

```go
collection, err := db.CreateCollection(ctx, "sq_collection",
    libravdb.WithDimension(768),
    libravdb.WithScalarQuantization(
        8,    // bits (4, 8, or 16)
        0.1,  // training ratio
    ),
)
```

#### Custom Quantization

```go
quantConfig := &quant.QuantizationConfig{
    Type:       quant.ProductQuantization,
    Codebooks:  16,
    Bits:       8,
    TrainRatio: 0.15,
    CacheSize:  10000,
}

collection, err := db.CreateCollection(ctx, "custom_quant",
    libravdb.WithQuantization(quantConfig),
)
```

### Metadata and Filtering Configuration

#### Metadata Schema

```go
schema := libravdb.MetadataSchema{
    "title":    libravdb.StringField,
    "category": libravdb.StringField,
    "score":    libravdb.FloatField,
    "tags":     libravdb.StringArrayField,
    "created":  libravdb.TimeField,
}

collection, err := db.CreateCollection(ctx, "structured_collection",
    libravdb.WithDimension(768),
    libravdb.WithMetadataSchema(schema),
    libravdb.WithIndexedFields("category", "score"), // Index for fast filtering
)
```

#### Field Types
- `StringField`: Text values
- `IntField`: Integer numbers
- `FloatField`: Floating-point numbers
- `BoolField`: Boolean values
- `TimeField`: Timestamps
- `StringArrayField`: Array of strings
- `IntArrayField`: Array of integers
- `FloatArrayField`: Array of floats

### Batch Processing Configuration

#### Basic Batch Configuration

```go
batchConfig := libravdb.BatchConfig{
    ChunkSize:       1000,              // Process 1000 items per chunk
    MaxConcurrency:  8,                 // Use 8 worker goroutines
    FailFast:        false,             // Continue on errors
    TimeoutPerChunk: 30 * time.Second,  // 30s timeout per chunk
}

collection, err := db.CreateCollection(ctx, "batch_collection",
    libravdb.WithBatchConfig(batchConfig),
)
```

#### Streaming Configuration

```go
streamOpts := &libravdb.StreamingOptions{
    BufferSize:            10000,        // Internal buffer size
    ChunkSize:             1000,         // Batch size
    MaxConcurrency:        8,            // Worker threads
    Timeout:               60 * time.Second,
    EnableBackpressure:    true,         // Handle slow consumers
    BackpressureThreshold: 0.8,          // Trigger at 80% buffer full
}
```

## Performance Tuning

### Memory Optimization

```go
// For large collections with memory constraints
collection, err := db.CreateCollection(ctx, "optimized",
    libravdb.WithDimension(768),
    libravdb.WithMemoryLimit(8*1024*1024*1024), // 8GB limit
    libravdb.WithMemoryMapping(true),            // Use memory mapping
    libravdb.WithProductQuantization(8, 8, 0.1), // Reduce memory footprint
    libravdb.WithCachePolicy(libravdb.LRUCache), // Efficient caching
)
```

### High-Throughput Configuration

```go
// For high insertion rates
collection, err := db.CreateCollection(ctx, "high_throughput",
    libravdb.WithDimension(768),
    libravdb.WithHNSW(16, 100, 50),     // Faster building
    libravdb.WithBatchChunkSize(5000),   // Large batches
    libravdb.WithBatchConcurrency(16),   // More workers
)
```

### High-Accuracy Configuration

```go
// For maximum search accuracy
collection, err := db.CreateCollection(ctx, "high_accuracy",
    libravdb.WithDimension(768),
    libravdb.WithHNSW(64, 800, 200),    // High-quality index
    // No quantization for exact vectors
)
```

## Environment-Specific Configurations

### Development Environment

```go
db, err := libravdb.New(
    libravdb.WithStoragePath("./dev_data"),
    libravdb.WithMetrics(false),         // Disable metrics
    libravdb.WithMaxCollections(10),     // Limit collections
)

collection, err := db.CreateCollection(ctx, "dev_collection",
    libravdb.WithDimension(128),         // Smaller dimension
    libravdb.WithFlat(),                 // Simple index
)
```

### Production Environment

```go
db, err := libravdb.New(
    libravdb.WithStoragePath("/var/lib/libravdb"),
    libravdb.WithMetrics(true),          // Enable monitoring
    libravdb.WithTracing(true),          // Enable tracing
    libravdb.WithMaxCollections(1000),   // Production scale
)

collection, err := db.CreateCollection(ctx, "prod_collection",
    libravdb.WithDimension(768),
    libravdb.WithHNSW(32, 400, 100),     // Balanced performance
    libravdb.WithMemoryLimit(16*1024*1024*1024), // 16GB limit
    libravdb.WithMemoryMapping(true),     // Use memory mapping
    libravdb.WithProductQuantization(8, 8, 0.1), // Memory efficiency
)
```

### High-Scale Environment

```go
// For millions of vectors
collection, err := db.CreateCollection(ctx, "large_scale",
    libravdb.WithDimension(768),
    libravdb.WithAutoIndexSelection(true), // Automatic optimization
    libravdb.WithMemoryLimit(64*1024*1024*1024), // 64GB limit
    libravdb.WithMemoryMapping(true),
    libravdb.WithProductQuantization(16, 8, 0.05), // Aggressive compression
    libravdb.WithBatchChunkSize(10000),
    libravdb.WithBatchConcurrency(32),
)
```

## Configuration Best Practices

### 1. Start Simple
Begin with basic configuration and optimize based on actual usage patterns.

### 2. Monitor Performance
Use metrics to understand bottlenecks and optimize accordingly.

### 3. Test Different Configurations
Benchmark different settings with your actual data and query patterns.

### 4. Consider Memory vs. Accuracy Tradeoffs
- More memory = better performance and accuracy
- Quantization = less memory but slightly lower accuracy
- Memory mapping = handle larger datasets with limited RAM

### 5. Batch Operations
Use streaming for large datasets to maintain consistent performance.

### 6. Index Selection
- Use Flat for <10K vectors or when exact results are required
- Use HNSW for most production workloads
- Use auto-selection for varying collection sizes

### 7. Memory Management
- Set appropriate memory limits based on available system resources
- Enable memory mapping for large collections
- Monitor memory usage and adjust limits as needed