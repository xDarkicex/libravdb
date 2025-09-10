# LibraVDB API Design

This document outlines the design principles, patterns, and architecture of LibraVDB's public API.

## Design Principles

### 1. Simplicity First
The API prioritizes ease of use while maintaining power and flexibility:
- Sensible defaults for common use cases
- Progressive disclosure of complexity
- Clear, self-documenting method names

### 2. Go Idioms
Following Go best practices and conventions:
- Context-aware operations
- Error handling with detailed error types
- Interface-based design for extensibility
- Minimal dependencies

### 3. Performance by Default
API design considers performance implications:
- Efficient memory usage patterns
- Batch operations for high throughput
- Streaming interfaces for large datasets
- Zero-copy operations where possible

### 4. Type Safety
Strong typing to prevent runtime errors:
- Compile-time configuration validation
- Typed metadata schemas
- Generic interfaces where appropriate

## API Layers

### Layer 1: Database Management
```go
type Database interface {
    CreateCollection(ctx context.Context, name string, opts ...CollectionOption) (*Collection, error)
    GetCollection(name string) (*Collection, error)
    ListCollections() []string
    Health(ctx context.Context) (*HealthStatus, error)
    Stats() *DatabaseStats
    Close() error
}
```

**Design Rationale:**
- Database acts as the root container and factory
- Collections are created through the database for consistency
- Health and stats provide observability
- Context support for cancellation and timeouts

### Layer 2: Collection Operations
```go
type Collection interface {
    Insert(ctx context.Context, id string, vector []float32, metadata map[string]interface{}) error
    Search(ctx context.Context, vector []float32, k int) (*SearchResults, error)
    Query(ctx context.Context) *QueryBuilder
    Stats() *CollectionStats
    Close() error
}
```

**Design Rationale:**
- Simple Insert/Search for basic operations
- Query builder for complex filtering
- Metadata as flexible map[string]interface{}
- Stats for monitoring and optimization

### Layer 3: Query Building
```go
type QueryBuilder interface {
    WithVector(vector []float32) *QueryBuilder
    Eq(field string, value interface{}) *QueryBuilder
    And() *QueryBuilder
    Or() *QueryBuilder
    Limit(limit int) *QueryBuilder
    Execute() (*SearchResults, error)
}
```

**Design Rationale:**
- Fluent interface for readability
- Chainable methods for complex queries
- Type-safe field operations
- Deferred execution for optimization

## Configuration Design

### Functional Options Pattern
LibraVDB uses the functional options pattern for configuration:

```go
type Option func(*Config) error
type CollectionOption func(*CollectionConfig) error

// Usage
db, err := libravdb.New(
    WithStoragePath("./data"),
    WithMetrics(true),
)

collection, err := db.CreateCollection(ctx, "vectors",
    WithDimension(768),
    WithHNSW(32, 200, 50),
    WithQuantization(quantConfig),
)
```

**Benefits:**
- Backward compatibility when adding new options
- Self-documenting configuration
- Compile-time validation
- Optional parameters with defaults

### Configuration Validation
All configuration is validated at creation time:

```go
func (c *CollectionConfig) validate() error {
    if c.Dimension <= 0 {
        return fmt.Errorf("dimension must be positive, got %d", c.Dimension)
    }
    // ... more validation
    return nil
}
```

## Error Handling Design

### Structured Errors
LibraVDB provides structured error information:

```go
type Error struct {
    Code      ErrorCode `json:"code"`
    Message   string    `json:"message"`
    Component string    `json:"component"`
    Context   *ErrorContext `json:"context,omitempty"`
}

type ErrorCode string

const (
    ErrInvalidDimension     ErrorCode = "INVALID_DIMENSION"
    ErrCollectionNotFound   ErrorCode = "COLLECTION_NOT_FOUND"
    ErrMemoryLimitExceeded  ErrorCode = "MEMORY_LIMIT_EXCEEDED"
    // ... more error codes
)
```

**Design Rationale:**
- Machine-readable error codes
- Human-readable messages
- Component identification for debugging
- Additional context for complex errors

### Error Recovery
Built-in error recovery mechanisms:

```go
type ErrorRecoveryManager interface {
    RegisterStrategy(code ErrorCode, strategy RecoveryStrategy)
    AttemptRecovery(ctx context.Context, err *Error) error
}
```

## Streaming API Design

### High-Throughput Insertions
For large-scale data ingestion:

```go
type StreamingBatchInsert interface {
    Start() error
    Send(entry *VectorEntry) error
    Stats() *StreamingStats
    Close() error
}

// Usage
stream := collection.NewStreamingBatchInsert(opts)
stream.Start()
for _, entry := range largeDataset {
    stream.Send(entry)
}
```

**Design Features:**
- Backpressure handling
- Progress callbacks
- Error callbacks
- Configurable batching

### Reader Interface
For flexible data sources:

```go
type StreamingReader interface {
    Read() (*VectorEntry, error)
    Close() error
}

// Usage
reader := NewChannelStreamingReader(dataChan)
stream, err := collection.StreamFromReader(reader, opts)
```

## Memory Management API

### Explicit Memory Control
```go
type Collection interface {
    SetMemoryLimit(bytes int64) error
    GetMemoryUsage() (*MemoryUsage, error)
    TriggerGC() error
    EnableMemoryMapping(path string) error
}
```

**Design Rationale:**
- Explicit control over memory usage
- Observability into memory consumption
- Manual GC triggering for optimization
- Memory mapping for large datasets

## Observability Design

### Metrics Integration
Built-in Prometheus metrics:

```go
type Metrics struct {
    VectorInserts   prometheus.Counter
    SearchQueries   prometheus.Counter
    SearchLatency   prometheus.Histogram
    MemoryUsage     prometheus.Gauge
}
```

### Health Checks
Comprehensive health monitoring:

```go
type HealthStatus struct {
    Status     string            `json:"status"`
    Components map[string]string `json:"components"`
    Timestamp  time.Time         `json:"timestamp"`
}
```

## Extensibility Design

### Plugin Architecture
Interface-based design allows for extensions:

```go
type Index interface {
    Insert(ctx context.Context, entry *VectorEntry) error
    Search(ctx context.Context, vector []float32, k int) ([]*SearchResult, error)
    Size() int
    Close() error
}

type Quantizer interface {
    Train(ctx context.Context, vectors [][]float32) error
    Encode(vector []float32) ([]byte, error)
    Decode(data []byte) ([]float32, error)
}
```

### Registry Pattern
For algorithm registration:

```go
type IndexRegistry interface {
    Register(name string, factory IndexFactory)
    Create(name string, config interface{}) (Index, error)
    List() []string
}
```

## API Evolution Strategy

### Versioning
- Semantic versioning for releases
- Backward compatibility within major versions
- Deprecation warnings before breaking changes

### Feature Flags
```go
type Config struct {
    ExperimentalFeatures map[string]bool
}
```

### Interface Segregation
Small, focused interfaces for easier testing and mocking:

```go
type Searcher interface {
    Search(ctx context.Context, vector []float32, k int) (*SearchResults, error)
}

type Inserter interface {
    Insert(ctx context.Context, id string, vector []float32, metadata map[string]interface{}) error
}
```

## Performance Considerations

### Zero-Copy Operations
Where possible, avoid unnecessary data copying:

```go
// Efficient: reuses input slice
func (c *Collection) SearchWithVector(vector []float32) (*SearchResults, error)

// Inefficient: would copy vector
func (c *Collection) SearchWithVectorCopy(vector []float32) (*SearchResults, error)
```

### Batch Operations
Encourage batching for better performance:

```go
type BatchInserter interface {
    InsertBatch(ctx context.Context, entries []*VectorEntry) error
}
```

### Lazy Initialization
Defer expensive operations until needed:

```go
type Collection struct {
    index Index // initialized on first use
    // ...
}
```

## Testing Strategy

### Interface Testing
All public interfaces have comprehensive test suites:

```go
func TestCollectionInterface(t *testing.T) {
    collection := createTestCollection(t)
    
    // Test all interface methods
    testInsert(t, collection)
    testSearch(t, collection)
    testQuery(t, collection)
}
```

### Mock Implementations
Interfaces allow for easy mocking:

```go
type MockIndex struct {
    insertFunc func(ctx context.Context, entry *VectorEntry) error
    searchFunc func(ctx context.Context, vector []float32, k int) ([]*SearchResult, error)
}
```

### Integration Tests
End-to-end testing of API workflows:

```go
func TestFullWorkflow(t *testing.T) {
    db := createTestDatabase(t)
    collection := createTestCollection(t, db)
    
    // Test complete workflow
    insertTestData(t, collection)
    searchAndValidate(t, collection)
    optimizeAndValidate(t, collection)
}
```

## Documentation Strategy

### Code Examples
Every public method includes usage examples:

```go
// Insert adds a vector to the collection.
//
// Example:
//   vector := []float32{0.1, 0.2, 0.3}
//   metadata := map[string]interface{}{"category": "test"}
//   err := collection.Insert(ctx, "doc1", vector, metadata)
func (c *Collection) Insert(ctx context.Context, id string, vector []float32, metadata map[string]interface{}) error
```

### API Stability Guarantees
Clear documentation of what's stable vs experimental:

```go
// Stable: This API is stable and will not change in backward-incompatible ways
func (c *Collection) Insert(ctx context.Context, id string, vector []float32, metadata map[string]interface{}) error

// Experimental: This API is experimental and may change
func (c *Collection) ExperimentalFeature() error
```

This API design provides a solid foundation for LibraVDB that balances simplicity, performance, and extensibility while following Go best practices.