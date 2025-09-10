# API Reference

Complete API documentation for LibraVDB.

## Database

### Creating a Database

```go
func New(opts ...Option) (*Database, error)
```

Creates a new database instance with the specified options.

**Options:**
- `WithStoragePath(path string)` - Set storage directory
- `WithMetrics(enabled bool)` - Enable/disable metrics collection
- `WithTracing(enabled bool)` - Enable/disable distributed tracing
- `WithMaxCollections(max int)` - Set maximum number of collections

**Example:**
```go
db, err := libravdb.New(
    libravdb.WithStoragePath("./data"),
    libravdb.WithMetrics(true),
    libravdb.WithMaxCollections(100),
)
```

### Database Methods

#### CreateCollection
```go
func (db *Database) CreateCollection(ctx context.Context, name string, opts ...CollectionOption) (*Collection, error)
```

Creates a new collection with the specified configuration.

#### GetCollection
```go
func (db *Database) GetCollection(name string) (*Collection, error)
```

Retrieves an existing collection by name.

#### ListCollections
```go
func (db *Database) ListCollections() []string
```

Returns the names of all collections in the database.

#### Health
```go
func (db *Database) Health(ctx context.Context) (*obs.HealthStatus, error)
```

Returns the current health status of the database.

#### Stats
```go
func (db *Database) Stats() *DatabaseStats
```

Returns comprehensive database statistics.

#### Close
```go
func (db *Database) Close() error
```

Gracefully shuts down the database and all collections.

## Collection

### Collection Options

#### Basic Configuration
```go
WithDimension(dim int)                    // Set vector dimension
WithMetric(metric DistanceMetric)         // Set distance metric
```

#### Index Configuration
```go
WithHNSW(m, efConstruction, efSearch int) // Configure HNSW index
WithFlat()                                // Use flat (exact) index
WithAutoIndexSelection(enabled bool)      // Enable automatic index selection
```

#### Memory Management
```go
WithMemoryLimit(bytes int64)              // Set memory limit
WithMemoryMapping(enabled bool)           // Enable memory mapping
WithCachePolicy(policy CachePolicy)      // Set cache eviction policy
```

#### Quantization
```go
WithQuantization(config *quant.QuantizationConfig)     // Custom quantization
WithProductQuantization(codebooks, bits int, trainRatio float64) // Product quantization
WithScalarQuantization(bits int, trainRatio float64)   // Scalar quantization
```

#### Metadata and Filtering
```go
WithMetadataSchema(schema MetadataSchema) // Define metadata schema
WithIndexedFields(fields ...string)      // Index specific fields for filtering
```

#### Batch Processing
```go
WithBatchConfig(config BatchConfig)      // Configure batch operations
WithBatchChunkSize(size int)            // Set batch chunk size
WithBatchConcurrency(concurrency int)   // Set batch concurrency
```

### Collection Methods

#### Insert
```go
func (c *Collection) Insert(ctx context.Context, id string, vector []float32, metadata map[string]interface{}) error
```

Inserts or updates a vector in the collection.

**Parameters:**
- `id` - Unique identifier for the vector
- `vector` - Float32 array matching collection dimension
- `metadata` - Optional key-value pairs for filtering

#### Search
```go
func (c *Collection) Search(ctx context.Context, vector []float32, k int) (*SearchResults, error)
```

Performs vector similarity search.

**Parameters:**
- `vector` - Query vector
- `k` - Number of results to return

**Returns:**
- `SearchResults` containing matched vectors with scores

#### Query
```go
func (c *Collection) Query(ctx context.Context) *QueryBuilder
```

Returns a query builder for advanced filtering and search.

#### Stats
```go
func (c *Collection) Stats() *CollectionStats
```

Returns collection statistics including memory usage and optimization status.

#### Close
```go
func (c *Collection) Close() error
```

Closes the collection and releases resources.

## Query Builder

The QueryBuilder provides a fluent interface for complex queries with filtering.

### Basic Query Methods

```go
func (qb *QueryBuilder) WithVector(vector []float32) *QueryBuilder
func (qb *QueryBuilder) Limit(limit int) *QueryBuilder
func (qb *QueryBuilder) WithThreshold(threshold float32) *QueryBuilder
func (qb *QueryBuilder) Execute() (*SearchResults, error)
```

### Filtering Methods

#### Equality Filters
```go
func (qb *QueryBuilder) Eq(field string, value interface{}) *QueryBuilder
func (qb *QueryBuilder) NotEq(field string, value interface{}) *QueryBuilder
```

#### Range Filters
```go
func (qb *QueryBuilder) Gt(field string, value interface{}) *QueryBuilder
func (qb *QueryBuilder) Gte(field string, value interface{}) *QueryBuilder
func (qb *QueryBuilder) Lt(field string, value interface{}) *QueryBuilder
func (qb *QueryBuilder) Lte(field string, value interface{}) *QueryBuilder
func (qb *QueryBuilder) Between(field string, min, max interface{}) *QueryBuilder
```

#### Containment Filters
```go
func (qb *QueryBuilder) Contains(field string, value interface{}) *QueryBuilder
func (qb *QueryBuilder) ContainsAny(field string, values []interface{}) *QueryBuilder
func (qb *QueryBuilder) ContainsAll(field string, values []interface{}) *QueryBuilder
```

#### Logical Operators
```go
func (qb *QueryBuilder) And() *QueryBuilder
func (qb *QueryBuilder) Or() *QueryBuilder
func (qb *QueryBuilder) Not() *QueryBuilder
func (qb *QueryBuilder) End() *QueryBuilder
```

### Query Examples

#### Simple Equality Filter
```go
results, err := collection.Query(ctx).
    WithVector(queryVector).
    Eq("category", "documents").
    Limit(10).
    Execute()
```

#### Range Filter
```go
results, err := collection.Query(ctx).
    WithVector(queryVector).
    Between("score", 0.8, 1.0).
    Limit(10).
    Execute()
```

#### Complex Logical Filter
```go
results, err := collection.Query(ctx).
    WithVector(queryVector).
    And().
        Eq("category", "documents").
        Or().
            Gt("priority", 5).
            ContainsAny("tags", []interface{}{"urgent", "important"}).
        End().
    End().
    Limit(10).
    Execute()
```

## Streaming Operations

### StreamingBatchInsert

For high-throughput batch insertions:

```go
func (c *Collection) NewStreamingBatchInsert(opts *StreamingOptions) *StreamingBatchInsert
```

#### StreamingOptions
```go
type StreamingOptions struct {
    BufferSize              int           // Internal buffer size
    ChunkSize               int           // Batch processing chunk size
    MaxConcurrency          int           // Maximum concurrent workers
    Timeout                 time.Duration // Operation timeout
    EnableBackpressure      bool          // Enable backpressure handling
    BackpressureThreshold   float64       // Buffer utilization threshold
    ProgressCallback        func(*StreamingStats) // Progress reporting
    ErrorCallback           func(error, *VectorEntry) // Error handling
}
```

#### Usage Example
```go
opts := libravdb.DefaultStreamingOptions()
opts.ChunkSize = 1000
opts.MaxConcurrency = 8

stream := collection.NewStreamingBatchInsert(opts)
err := stream.Start()
if err != nil {
    log.Fatal(err)
}
defer stream.Close()

// Send vectors
for _, entry := range largeDataset {
    err := stream.Send(entry)
    if err != nil {
        log.Printf("Failed to send entry: %v", err)
    }
}

// Get statistics
stats := stream.Stats()
fmt.Printf("Processed %d entries\n", stats.TotalProcessed)
```

## Data Types

### VectorEntry
```go
type VectorEntry struct {
    ID       string                 `json:"id"`
    Vector   []float32              `json:"vector"`
    Metadata map[string]interface{} `json:"metadata,omitempty"`
}
```

### SearchResult
```go
type SearchResult struct {
    ID       string                 `json:"id"`
    Score    float32                `json:"score"`
    Vector   []float32              `json:"vector,omitempty"`
    Metadata map[string]interface{} `json:"metadata,omitempty"`
}
```

### SearchResults
```go
type SearchResults struct {
    Results []*SearchResult `json:"results"`
    Took    time.Duration   `json:"took"`
    Total   int             `json:"total"`
}
```

### CollectionStats
```go
type CollectionStats struct {
    Name                 string                  `json:"name"`
    VectorCount          int                     `json:"vector_count"`
    Dimension            int                     `json:"dimension"`
    IndexType            string                  `json:"index_type"`
    MemoryUsage          int64                   `json:"memory_usage"`
    MemoryStats          *CollectionMemoryStats  `json:"memory_stats,omitempty"`
    OptimizationStatus   *OptimizationStatus     `json:"optimization_status,omitempty"`
    HasQuantization      bool                    `json:"has_quantization"`
    HasMemoryLimit       bool                    `json:"has_memory_limit"`
    MemoryMappingEnabled bool                    `json:"memory_mapping_enabled"`
}
```

## Constants

### Distance Metrics
```go
const (
    L2Distance     DistanceMetric = iota // Euclidean distance
    InnerProduct                         // Inner product (dot product)
    CosineDistance                       // Cosine distance
)
```

### Index Types
```go
const (
    HNSW  IndexType = iota // Hierarchical Navigable Small World
    IVFPQ                  // Inverted File with Product Quantization
    Flat                   // Brute-force exact search
)
```

### Cache Policies
```go
const (
    LRUCache  CachePolicy = iota // Least Recently Used
    LFUCache                     // Least Frequently Used
    FIFOCache                    // First In, First Out
)
```

## Error Handling

LibraVDB provides structured error information:

```go
type Error struct {
    Code      ErrorCode `json:"code"`
    Message   string    `json:"message"`
    Component string    `json:"component"`
    Context   *ErrorContext `json:"context,omitempty"`
}
```

### Common Error Codes
- `ErrInvalidDimension` - Vector dimension mismatch
- `ErrCollectionNotFound` - Collection doesn't exist
- `ErrDatabaseClosed` - Operation on closed database
- `ErrMemoryLimitExceeded` - Memory limit reached
- `ErrBackpressureActive` - Streaming backpressure triggered

### Error Handling Example
```go
if err != nil {
    if libravdbErr, ok := err.(*libravdb.Error); ok {
        switch libravdbErr.Code {
        case libravdb.ErrInvalidDimension:
            // Handle dimension mismatch
        case libravdb.ErrMemoryLimitExceeded:
            // Handle memory pressure
        default:
            // Handle other errors
        }
    }
}
```