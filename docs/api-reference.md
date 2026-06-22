# LibraVDB API Reference

Complete API reference for the `libravdb` package. Every public type, function,
method, constant, and variable is documented with signatures, parameter
descriptions, return values, and usage examples.

## Table of Contents

- [Database Lifecycle](#database-lifecycle)
- [Collection Operations](#collection-operations)
- [Query Builder](#query-builder)
- [Transactions](#transactions)
- [Batch Operations](#batch-operations)
- [Streaming Operations](#streaming-operations)
- [Graph Layer](#graph-layer)
- [Configuration Options](#configuration-options)
- [Data Types](#data-types)
- [Error Handling](#error-handling)
- [Health & Observability](#health--observability)

---

## Database Lifecycle

### func Open

```go
func Open(opts ...Option) (*Database, error)
```

Opens a database at the configured path, creating the storage file if it does
not exist. If the existing file is in v1 format, `Open` automatically migrates
it to the current format before completing.

Returns `*Database` and nil on success. Returns an error wrapping
`storage.ErrV1FormatMigrationRequired` if automatic migration fails.

**Example:**

```go
db, err := libravdb.Open(
    libravdb.WithStoragePath("./vector_data"),
    libravdb.WithMetrics(true),
)
if err != nil {
    log.Fatal(err)
}
defer db.Close()
```

### func (db *Database) CreateCollection

```go
func (db *Database) CreateCollection(ctx context.Context, name string, opts ...CollectionOption) (*Collection, error)
```

Creates a named collection with the specified configuration options. The
collection is persisted to storage immediately.

**Parameters:**
- `ctx` — context for cancellation and deadlines
- `name` — unique collection name within the database
- `opts` — zero or more `CollectionOption` values configuring dimension, index, memory, etc.

**Errors:**
- `ErrCollectionExists` — a collection with `name` already exists
- `ErrTooManyCollections` — the configured `MaxCollections` limit has been reached
- Dimension or option validation errors

**Example:**

```go
col, err := db.CreateCollection(ctx, "embeddings",
    libravdb.WithDimension(768),
    libravdb.WithMetric(libravdb.CosineDistance),
    libravdb.WithHNSW(32, 200, 100),
)
```

### func (db *Database) GetCollection

```go
func (db *Database) GetCollection(name string) (*Collection, error)
```

Retrieves an existing collection by name. The collection must have been
previously created and persisted — either in the current session or discovered
during reopen.

Returns `ErrCollectionNotFound` if no collection with `name` exists.

### func (db *Database) ListCollections

```go
func (db *Database) ListCollections() []string
```

Returns the names of all persisted collections known to the database. Includes
collections discovered during reopen, before any explicit `GetCollection` calls.
The returned slice is a snapshot; concurrent creation or deletion does not affect it.

### func (db *Database) ListCollectionsWithContext

```go
func (db *Database) ListCollectionsWithContext(ctx context.Context) ([]string, error)
```

Like `ListCollections`, but surfaces storage discovery errors to callers that
need strict reopen/lifecycle validation.

### func (db *Database) DeleteCollection

```go
func (db *Database) DeleteCollection(ctx context.Context, name string) error
```

Durably deletes a collection and all its persisted data. The deletion is
logical-first (the collection becomes invisible immediately) with physical
reclamation deferred to the next compaction/checkpoint.

### func (db *Database) DeleteCollections

```go
func (db *Database) DeleteCollections(ctx context.Context, names []string) error
```

Deletes multiple collections by exact name. Each deletion is independent; an
error deleting one does not prevent the others from being deleted.

### func (db *Database) BeginTx

```go
func (db *Database) BeginTx(ctx context.Context) (Tx, error)
```

Begins a multi-operation transaction. See the [Transactions](#transactions)
section for the full `Tx` interface and commit protocol.

### func (db *Database) WithTx

```go
func (db *Database) WithTx(ctx context.Context, fn func(tx Tx) error) error
```

Executes `fn` within a transaction. If `fn` returns nil, the transaction is
committed. If `fn` returns an error (or panics), the transaction is rolled back.

```go
err := db.WithTx(ctx, func(tx libravdb.Tx) error {
    if err := tx.Insert(ctx, "col", "id1", vec, meta); err != nil {
        return err
    }
    return tx.Insert(ctx, "col", "id2", vec2, nil)
})
```

### func (db *Database) Health

```go
func (db *Database) Health(ctx context.Context) (*obs.HealthStatus, error)
```

Returns the current health status of the database and all registered components.

### func (db *Database) Stats

```go
func (db *Database) Stats() *DatabaseStats
```

Returns a snapshot of database-wide statistics including per-collection stats,
total memory usage, collection count, and uptime.

### func (db *Database) Close

```go
func (db *Database) Close() error
```

Gracefully shuts down the database. All collections are closed, indexes are
finalized, the storage engine flushes pending writes, and the file is closed.
After `Close`, all operations return `ErrDatabaseClosed`.

---

## Collection Operations

### func (c *Collection) Insert

```go
func (c *Collection) Insert(ctx context.Context, id string, vector []float32, metadata map[string]interface{}) error
```

Inserts or overwrites a vector identified by `id`. If an entry with the same
`id` already exists, it is replaced.

**Parameters:**
- `id` — unique identifier for the vector (non-empty)
- `vector` — `[]float32` whose length must match the collection dimension
- `metadata` — optional key-value pairs for filtering; may be nil

**Errors:**
- `ErrInvalidDimension` / `ErrDimensionMismatch` — vector length does not match
- `ErrDatabaseClosed` — database has been shut down
- `ErrCollectionClosed` — collection has been closed

```go
err := col.Insert(ctx, "doc-42", embedding, map[string]interface{}{
    "title": "Example Document",
    "score": 0.95,
})
```

### func (c *Collection) Get

```go
func (c *Collection) Get(ctx context.Context, id string) (Record, error)
```

Retrieves a single record by ID. Returns a zero-value `Record` and a
not-found error if the ID does not exist.

### func (c *Collection) Update

```go
func (c *Collection) Update(ctx context.Context, id string, vector []float32, metadata map[string]interface{}) error
```

Updates an existing vector. Unlike `Insert`, this returns a not-found error
if the ID does not exist. Vector and metadata may be nil to leave those fields
unchanged.

### func (c *Collection) Upsert

```go
func (c *Collection) Upsert(ctx context.Context, id string, vector []float32, metadata map[string]interface{}) error
```

Inserts the vector if `id` does not exist, or updates it if it does. The vector
parameter is required (non-nil) for upsert.

### func (c *Collection) UpdateIfVersion

```go
func (c *Collection) UpdateIfVersion(ctx context.Context, id string, vector []float32, metadata map[string]interface{}, expectedVersion uint64) error
```

Compare-and-swap update. The update only succeeds if the record's current
version equals `expectedVersion`. Returns `*VersionConflictError` on mismatch.

### func (c *Collection) Delete

```go
func (c *Collection) Delete(ctx context.Context, id string) error
```

Logically deletes the vector identified by `id`. The deletion is durable once
the call returns without error. Physical reclamation occurs during compaction.

### func (c *Collection) DeleteIfVersion

```go
func (c *Collection) DeleteIfVersion(ctx context.Context, id string, expectedVersion uint64) error
```

Compare-and-swap delete. Only deletes if the record version matches
`expectedVersion`. Returns `*VersionConflictError` on mismatch.

### func (c *Collection) InsertBatch

```go
func (c *Collection) InsertBatch(ctx context.Context, entries []VectorEntry) error
```

Inserts multiple vectors through a stable public batch API. Each entry must
have a non-empty ID and a vector matching the collection dimension. The
operation is atomic at the storage level.

### func (c *Collection) DeleteBatch

```go
func (c *Collection) DeleteBatch(ctx context.Context, ids []string) error
```

Deletes multiple vectors by ID. IDs that do not exist are silently skipped.

### func (c *Collection) Search

```go
func (c *Collection) Search(ctx context.Context, vector []float32, k int) (*SearchResults, error)
```

Performs a k-nearest-neighbor vector similarity search.

**Parameters:**
- `vector` — query vector; must match the collection dimension
- `k` — number of results to return (must be positive)

**Returns** `*SearchResults` where `Results` is ordered by descending relevance
score (higher is always better). Results include IDs, scores, metadata, and
optionally vectors.

### func (c *Collection) Query

```go
func (c *Collection) Query(ctx context.Context) *QueryBuilder
```

Returns a `*QueryBuilder` for constructing filtered, parameterized searches.
See the [Query Builder](#query-builder) section.

### func (c *Collection) Count

```go
func (c *Collection) Count(ctx context.Context) (int, error)
```

Returns the exact number of live (non-deleted) records in the collection.

### func (c *Collection) Iterate

```go
func (c *Collection) Iterate(ctx context.Context, fn func(Record) error) error
```

Iterates all persisted records, calling `fn` for each. Iteration stops if `fn`
returns an error. The record's `Version` and `Ordinal` fields are populated.

```go
err := col.Iterate(ctx, func(r libravdb.Record) error {
    fmt.Printf("ID=%s Version=%d\n", r.ID, r.Version)
    return nil
})
```

### func (c *Collection) ListAll

```go
func (c *Collection) ListAll(ctx context.Context) ([]Record, error)
```

Returns all persisted records as a slice. For large collections, prefer
`Iterate` to avoid materializing everything in memory.

### func (c *Collection) ListByMetadata

```go
func (c *Collection) ListByMetadata(ctx context.Context, field string, value interface{}) ([]Record, error)
```

Returns records whose metadata `field` exactly matches `value`. This is a
convenience wrapper around the query builder's equality filter.

### func (c *Collection) Stats

```go
func (c *Collection) Stats() *CollectionStats
```

Returns a snapshot of collection statistics including vector count, dimension,
index type, memory usage, memory pressure, optimization status, and raw vector
store metrics.

### func (c *Collection) Dimension

```go
func (c *Collection) Dimension() int
```

Returns the configured vector dimension for this collection.

### func (c *Collection) Close

```go
func (c *Collection) Close() error
```

Closes the collection, releasing its index, storage handles, and memory
resources. After close, all operations return `ErrCollectionClosed`.

---

## Query Builder

The `QueryBuilder` provides a fluent interface for constructing filtered vector
searches and metadata-only listings.

### Core Query Methods

#### func (qb *QueryBuilder) WithVector

```go
func (qb *QueryBuilder) WithVector(vector []float32) *QueryBuilder
```

Sets the query vector. A defensive copy is made internally.

#### func (qb *QueryBuilder) Limit

```go
func (qb *QueryBuilder) Limit(limit int) *QueryBuilder
```

Sets the maximum number of results to return.

#### func (qb *QueryBuilder) WithThreshold

```go
func (qb *QueryBuilder) WithThreshold(threshold float32) *QueryBuilder
```

Excludes results with a relevance score below `threshold`.

#### func (qb *QueryBuilder) WithEfSearch

```go
func (qb *QueryBuilder) WithEfSearch(ef int) *QueryBuilder
```

Overrides the collection-level `EfSearch` parameter for this query. Larger
values improve recall at the cost of latency.

#### func (qb *QueryBuilder) WithFilter

```go
func (qb *QueryBuilder) WithFilter(f filter.Filter) *QueryBuilder
```

Adds an arbitrary `filter.Filter` from the `internal/filter` package.

#### func (qb *QueryBuilder) WithGraphFilter

```go
func (qb *QueryBuilder) WithGraphFilter(gf GraphFilter) *QueryBuilder
```

Attaches a graph-based filter that tests candidates against a pre-computed
bitset from the graph layer.

#### func (qb *QueryBuilder) Execute

```go
func (qb *QueryBuilder) Execute() (*SearchResults, error)
```

Executes the query and returns ranked search results. Requires `WithVector` to
have been called.

#### func (qb *QueryBuilder) List

```go
func (qb *QueryBuilder) List() ([]Record, error)
```

Executes a metadata-only listing (no vector search). Returns records matching
the configured filters.

### Equality Filters

```go
func (qb *QueryBuilder) Eq(field string, value interface{}) *QueryBuilder
func (qb *QueryBuilder) NotEq(field string, value interface{}) *QueryBuilder
```

Equality and inequality filtering on metadata fields.

### Range Filters

```go
func (qb *QueryBuilder) Gt(field string, value interface{}) *QueryBuilder
func (qb *QueryBuilder) Gte(field string, value interface{}) *QueryBuilder
func (qb *QueryBuilder) Lt(field string, value interface{}) *QueryBuilder
func (qb *QueryBuilder) Lte(field string, value interface{}) *QueryBuilder
func (qb *QueryBuilder) Between(field string, min, max interface{}) *QueryBuilder
```

Range comparisons on numeric metadata fields. `Gte` and `Lte` accept the bound
as the second argument. `Between` accepts min and max inclusive.

### Containment Filters

```go
func (qb *QueryBuilder) Contains(field string, value interface{}) *QueryBuilder
func (qb *QueryBuilder) ContainsAny(field string, values []interface{}) *QueryBuilder
func (qb *QueryBuilder) ContainsAll(field string, values []interface{}) *QueryBuilder
```

Array containment checks. `Contains` is shorthand for `ContainsAny` with a
single value. `ContainsAll` requires all specified values to be present.

### Logical Grouping (FilterChain)

```go
func (qb *QueryBuilder) And() *FilterChain
func (qb *QueryBuilder) Or() *FilterChain
func (qb *QueryBuilder) Not() *FilterChain
```

Begin a logical group. The returned `*FilterChain` supports the same filter
methods (`Eq`, `Gt`, `Between`, `ContainsAny`, etc.) plus nested `And`/`Or`/`Not`
calls. Call `End()` on the chain to return to the parent builder.

#### func (fc *FilterChain) End

```go
func (fc *FilterChain) End() *QueryBuilder
```

Closes the current logical group and returns to the parent `*QueryBuilder`.

### Query Examples

**Simple vector search:**

```go
results, err := col.Query(ctx).
    WithVector(queryVec).
    Limit(10).
    Execute()
```

**Metadata-only listing:**

```go
records, err := col.Query(ctx).
    Eq("sessionId", "s1").
    Limit(100).
    List()
```

**Filtered vector search:**

```go
results, err := col.Query(ctx).
    WithVector(queryVec).
    Eq("category", "documents").
    Between("score", 0.8, 1.0).
    Limit(10).
    Execute()
```

**Complex logical filter:**

```go
results, err := col.Query(ctx).
    WithVector(queryVec).
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

---

## Transactions

### type Tx

```go
type Tx interface {
    Insert(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}) error
    InsertOwned(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}) error
    Update(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}) error
    UpdateOwned(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}) error
    Upsert(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}) error
    UpdateIfVersion(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}, expectedVersion uint64) error
    Delete(ctx context.Context, collection, id string) error
    DeleteIfVersion(ctx context.Context, collection, id string, expectedVersion uint64) error
    DeleteBatch(ctx context.Context, collection string, ids []string) error
    ListByMetadata(ctx context.Context, collection, field string, value interface{}) ([]Record, error)
    Commit(ctx context.Context) error
    Rollback(ctx context.Context) error
}
```

Cross-collection transactional mutations. All mutations are staged in memory and
applied atomically at `Commit`. The `*Owned` variants take ownership of
the caller's `vector` slice (avoiding a copy); the caller must not reuse the
slice after the call.

**Lifecycle:**

1. `db.BeginTx(ctx)` or `db.WithTx(ctx, fn)` starts a transaction.
2. Stage mutations via `Insert`, `Update`, `Delete`, etc.
3. Call `Commit` to atomically apply all staged mutations.
4. Call `Rollback` (or let the transaction be garbage-collected) to discard.

### type VersionConflictError

```go
type VersionConflictError struct {
    ID              string
    ExpectedVersion uint64
    ActualVersion   uint64
}
```

Returned by `UpdateIfVersion` and `DeleteIfVersion` when the record version
does not match the expected version. Use `errors.Is(err, ErrVersionConflict)`
to detect.

### Transaction Example

```go
err := db.WithTx(ctx, func(tx libravdb.Tx) error {
    if err := tx.Insert(ctx, "users", "u1", userVec, userMeta); err != nil {
        return err
    }
    if err := tx.Insert(ctx, "docs", "d1", docVec, docMeta); err != nil {
        return err
    }
    return nil // commit
})
if err != nil {
    log.Printf("transaction failed: %v", err)
}
```

---

## Batch Operations

The batch API provides fine-grained control over bulk operations with chunking,
concurrency, retries, callbacks, and optional rollback.

### type BatchOperation

```go
type BatchOperation interface {
    Execute(ctx context.Context) (*BatchResult, error)
    Size() int
    EstimateMemory() int64
}
```

### Creating Batch Operations

```go
func (c *Collection) NewBatchInsert(entries []*VectorEntry, opts ...*BatchOptions) *BatchInsert
func (c *Collection) NewBatchUpdate(updates []*VectorUpdate, opts ...*BatchOptions) *BatchUpdate
func (c *Collection) NewBatchDelete(ids []string, opts ...*BatchOptions) *BatchDelete
```

### type BatchOptions

```go
type BatchOptions struct {
    ProgressCallback func(completed, total int)
    DetailedProgress func(progress *BatchProgress)
    ErrorCallback    func(item *BatchItemResult, err error)
    ChunkSize        int
    MaxConcurrency   int
    Timeout          time.Duration
    MaxRetries       int
    RetryDelay       time.Duration
    FailFast         bool
    EnableRollback   bool
}
```

`DefaultBatchOptions()` returns sensible defaults: `ChunkSize=1000`,
`MaxConcurrency=4`, `FailFast=false`, `Timeout=5m`, `MaxRetries=3`,
`RetryDelay=100ms`.

### type BatchResult

```go
type BatchResult struct {
    Items            []*BatchItemResult
    Errors           map[int]error
    Successful       int
    Failed           int
    Duration         time.Duration
    RollbackRequired bool
    RollbackError    error
}
```

### type BatchProgress

```go
type BatchProgress struct {
    Completed    int
    Total        int
    Successful   int
    Failed       int
    CurrentChunk int
    TotalChunks  int
    ElapsedTime  time.Duration
    EstimatedETA time.Duration
    ItemsPerSec  float64
    LastError    error
}
```

### type VectorUpdate

```go
type VectorUpdate struct {
    ID       string
    Vector   []float32
    Metadata map[string]interface{}
    Upsert   bool
}
```

When `Upsert` is true, the update creates the record if it does not exist. When
false and the vector is nil, metadata-only update is performed.

### Batch Example

```go
entries := make([]*libravdb.VectorEntry, 0, 50000)
for i := 0; i < 50000; i++ {
    entries = append(entries, &libravdb.VectorEntry{
        ID:     fmt.Sprintf("vec_%d", i),
        Vector: randomVector(768),
    })
}

opts := libravdb.DefaultBatchOptions()
opts.ChunkSize = 2000
opts.MaxConcurrency = 8
opts.ProgressCallback = func(completed, total int) {
    fmt.Printf("\rProgress: %d/%d (%.1f%%)", completed, total,
        float64(completed)/float64(total)*100)
}

batch := col.NewBatchInsert(entries, opts)
result, err := batch.Execute(ctx)
if err != nil {
    log.Fatalf("batch insert failed: %v", err)
}
fmt.Printf("\nDone: %d successful, %d failed, took %v\n",
    result.Successful, result.Failed, result.Duration)
```

---

## Streaming Operations

Streaming operations provide backpressure-aware, high-throughput ingestion for
very large datasets.

### Creating Streaming Operations

```go
func (c *Collection) NewStreamingBatchInsert(opts ...*StreamingOptions) *StreamingBatchInsert
func (c *Collection) NewStreamingBatchUpdate(opts ...*StreamingOptions) *StreamingBatchUpdate
func (c *Collection) NewStreamingBatchDelete(opts ...*StreamingOptions) *StreamingBatchDelete
func (c *Collection) StreamFromReader(reader StreamingReader, opts ...*StreamingOptions) (*StreamingBatchInsert, error)
```

### type StreamingOptions

```go
type StreamingOptions struct {
    BufferSize            int
    ChunkSize             int
    MaxConcurrency        int
    FlushInterval         time.Duration
    MaxMemoryUsage        int64
    BackpressureThreshold float64
    Timeout               time.Duration
    EnableBackpressure    bool
    ProgressCallback      func(stats *StreamingStats)
    ErrorCallback         func(err error, entry *VectorEntry)
    CompletionCallback    func(finalStats *StreamingStats)
}
```

`DefaultStreamingOptions()` returns: `BufferSize=10000`, `ChunkSize=1000`,
`MaxConcurrency=4`, `FlushInterval=5s`, `MaxMemoryUsage=1GB`,
`BackpressureThreshold=0.8`, `Timeout=30m`, `EnableBackpressure=true`.

### Streaming Lifecycle

```go
stream := col.NewStreamingBatchInsert(opts)

// 1. Start background workers
if err := stream.Start(); err != nil {
    log.Fatal(err)
}

// 2. Send entries (blocks if backpressure is active)
for _, entry := range largeDataset {
    if err := stream.Send(entry); err != nil {
        log.Printf("send failed: %v", err)
    }
}

// 3. Close the input channel and wait for workers to finish
if err := stream.Close(); err != nil {
    log.Fatal(err)
}

// 4. Wait for all workers to drain
if err := stream.Wait(); err != nil {
    log.Fatal(err)
}

// 5. Read final statistics
stats := stream.Stats()
fmt.Printf("Processed %d entries in %v (%.0f/sec)\n",
    stats.TotalProcessed, stats.ElapsedTime, stats.ItemsPerSecond)
```

### type StreamingStats

```go
type StreamingStats struct {
    Status             string
    TotalReceived      int64
    TotalProcessed     int64
    TotalSuccessful    int64
    TotalFailed        int64
    SuccessRate        float64
    ErrorRate          float64
    ItemsPerSecond     float64
    ElapsedTime        time.Duration
    EstimatedETA       time.Duration
    BufferUtilization  float64
    CurrentMemoryUsage int64
    ActiveWorkers      int
    QueuedItems        int
    BackpressureActive bool
    LastError          error
    ErrorsByType       map[string]int64
}
```

### Streaming Methods

```go
func (s *StreamingBatchInsert) Start() error
func (s *StreamingBatchInsert) Send(entry *VectorEntry) error
func (s *StreamingBatchInsert) SendBatch(entries []*VectorEntry) error
func (s *StreamingBatchInsert) Results() <-chan *StreamingResult
func (s *StreamingBatchInsert) Errors() <-chan error
func (s *StreamingBatchInsert) Stats() *StreamingStats
func (s *StreamingBatchInsert) Close() error
func (s *StreamingBatchInsert) Wait() error
```

`Send` blocks when the input buffer is full and backpressure is active. If the
buffer is full and backpressure is disabled, it returns `ErrBufferFull`.
`SendBatch` atomically submits multiple entries.

### type StreamingReader

```go
type StreamingReader interface {
    Read() (*VectorEntry, error)
    Close() error
}
```

Used with `StreamFromReader` to ingest from custom data sources.

```go
func NewChannelStreamingReader(ch <-chan *VectorEntry) *ChannelStreamingReader
```

---

## Graph Layer

The graph layer provides a property graph for managing edges and relationships
between vectors. It is an optional feature enabled via `WithGraph`.

### type Graph

```go
type Graph interface {
    BeginTxn() *Txn
    AddEdge(txn *Txn, src, tgt uint64, weight float32, kind uint8) error
    RemoveEdge(txn *Txn, src, tgt uint64, kind uint8) error
    DropNodeEdges(txn *Txn, nodeID uint64) error
    Neighbors(nodeID uint64) ([]Edge, error)
    Degree(nodeID uint64) (int, error)
    InboundNeighbors(nodeID uint64) ([]Edge, error)
    InboundDegree(nodeID uint64) (int, error)
    NeighborsAny(nodeID uint64, kindSet KindSet) ([]Edge, error)
    ForEachEdge(fn func(src, tgt uint64, edge Edge) bool)
    BFS(start uint64, maxDepth int, visit VisitAction, bitset *Bitset, frontier *FrontierBuf) error
    GetBitset() (*Bitset, error)
    PutBitset(b *Bitset)
    GetFrontierBuf() (*FrontierBuf, error)
    PutFrontierBuf(f *FrontierBuf)
    Stats() GraphStats
    Close() error
}
```

### func NewGraph

```go
func NewGraph(config GraphConfig) (Graph, error)
```

Creates a new standalone graph instance.

### type GraphConfig

```go
type GraphConfig struct {
    EdgeSlots        int
    EdgeSlotSize     int
    EdgeShards       int
    PageSlots        int
    PageShards       int
    BitsetPoolSize   int
    FrontierPoolSize int
    ArenaPages       int
}
```

Zero-value fields use sensible defaults from the internal graph package.

### Hooks

```go
type InsertHook func(txn GraphTx, id uint64, vector []float32, metadata map[string]interface{}) error
type DeleteHook func(txn GraphTx, id uint64) error
```

Hooks are invoked during transaction commits, after the WAL write but before the
transaction is considered durable. `InsertHook` is called for each inserted
vector; `DeleteHook` for each deleted vector.

### type GraphTx

```go
type GraphTx interface {
    AddEdge(src, tgt uint64, weight float32, kind uint8) error
    RemoveEdge(src, tgt uint64, kind uint8) error
}
```

The graph transaction handle passed to hooks. Edges added/removed during a hook
are committed atomically with the vector mutation.

### type GraphFilter

```go
type GraphFilter interface {
    Test(idx uint64) bool
}
```

Used with `QueryBuilder.WithGraphFilter` to filter search candidates against a
pre-computed graph bitset.

---

## Configuration Options

### Database Options (`Option`)

All database options are functions of type `func(*Config) error`.

| Option | Signature | Description |
|--------|-----------|-------------|
| `WithStoragePath` | `(path string) Option` | Sets the storage directory/file path. Default: `"./data"`. |
| `WithMetrics` | `(enabled bool) Option` | Enables Prometheus metrics collection. Default: `true`. |
| `WithTracing` | `(enabled bool) Option` | Enables distributed tracing spans. Default: `false`. |
| `WithMaxCollections` | `(max int) Option` | Maximum number of collections. Default: `100`. |
| `WithMaxConcurrentWrites` | `(max int) Option` | Bounds collection write execution parallelism. Default: `runtime.NumCPU()`. |
| `WithMaxWriteQueueDepth` | `(depth int) Option` | Bounds queued writers waiting for admission. Default: `32`. |
| `WithLogger` | `(logger Logger) Option` | Sets a logger for timing instrumentation during index rebuilds. |

### Collection Options (`CollectionOption`)

All collection options are functions of type `func(*CollectionConfig) error`.

**Dimension & Metric:**

| Option | Signature | Description |
|--------|-----------|-------------|
| `WithDimension` | `(dim int) CollectionOption` | Vector dimension (required). |
| `WithMetric` | `(metric DistanceMetric) CollectionOption` | Distance metric. |

**Index Configuration:**

| Option | Signature | Description |
|--------|-----------|-------------|
| `WithHNSW` | `(m, efConstruction, efSearch int) CollectionOption` | HNSW index parameters. |
| `WithFlat` | `() CollectionOption` | Brute-force exact search. |
| `WithIVFPQ` | `(nClusters, nProbes int) CollectionOption` | IVF-PQ index. |
| `WithAutoIndexSelection` | `(enabled bool) CollectionOption` | Automatic index selection by size. |
| `WithAutoIndexThresholds` | `(hnswThreshold, ivfpqThreshold int) CollectionOption` | Custom auto-index thresholds. |
| `WithSharding` | `(enabled bool) CollectionOption` | Enables sharding for HNSW/Flat indexes. |

**Vector Store:**

| Option | Signature | Description |
|--------|-----------|-------------|
| `WithRawVectorStoreMemory` | `() CollectionOption` | In-memory raw vector store (default). |
| `WithRawVectorStoreSlabby` | `(segmentCapacity int) CollectionOption` | Slabby-backed fixed-size vector store. |

**Persistence:**

| Option | Signature | Description |
|--------|-----------|-------------|
| `WithIndexPersistence` | `(enabled bool) CollectionOption` | Enable automatic index saves. |
| `WithPersistencePath` | `(path string) CollectionOption` | Path for automatic index saves. |
| `WithSaveInterval` | `(interval time.Duration) CollectionOption` | Interval between auto-saves. Default: `5m`. |

**Quantization:**

| Option | Signature | Description |
|--------|-----------|-------------|
| `WithQuantization` | `(config *quant.QuantizationConfig) CollectionOption` | Custom quantization config. |
| `WithProductQuantization` | `(codebooks, bits int, trainRatio float64) CollectionOption` | Product quantization. |
| `WithScalarQuantization` | `(bits int, trainRatio float64) CollectionOption` | Scalar quantization. |

**Memory Management:**

| Option | Signature | Description |
|--------|-----------|-------------|
| `WithMemoryLimit` | `(bytes int64) CollectionOption` | Maximum memory usage. |
| `WithMemoryMapping` | `(enabled bool) CollectionOption` | Enable OS memory mapping. |
| `WithCachePolicy` | `(policy CachePolicy) CollectionOption` | Cache eviction policy. |
| `WithMemoryConfig` | `(config *memory.MemoryConfig) CollectionOption` | Advanced memory config. |

**Metadata & Filtering:**

| Option | Signature | Description |
|--------|-----------|-------------|
| `WithMetadataSchema` | `(schema MetadataSchema) CollectionOption` | Define metadata field types. |
| `WithIndexedFields` | `(fields ...string) CollectionOption` | Index fields for fast filtering. |

**Batch Processing:**

| Option | Signature | Description |
|--------|-----------|-------------|
| `WithBatchConfig` | `(config BatchConfig) CollectionOption` | Full batch configuration. |
| `WithBatchChunkSize` | `(size int) CollectionOption` | Chunk size for batches. |
| `WithBatchConcurrency` | `(concurrency int) CollectionOption` | Max concurrency for batches. |

**Graph:**

| Option | Signature | Description |
|--------|-----------|-------------|
| `WithGraph` | `(g Graph) CollectionOption` | Attaches a graph instance to the collection. |

---

## Data Types

### type VectorEntry

```go
type VectorEntry struct {
    ID       string                 `json:"id"`
    Vector   []float32              `json:"vector"`
    Metadata map[string]interface{} `json:"metadata,omitempty"`
}
```

Input type for insert operations.

### type Record

```go
type Record struct {
    ID       string                 `json:"id"`
    Vector   []float32              `json:"vector"`
    Metadata map[string]interface{} `json:"metadata,omitempty"`
    Version  uint64                 `json:"version"`
    Ordinal  uint32                 `json:"ordinal"`
}
```

Output type for iteration and list operations. `Version` is a monotonically
increasing counter incremented on each update. `Ordinal` is the storage-level
slot identifier.

### type SearchResult

```go
type SearchResult struct {
    ID       string                 `json:"id"`
    Score    float32                `json:"score"`
    Vector   []float32              `json:"vector,omitempty"`
    Metadata map[string]interface{} `json:"metadata,omitempty"`
    Version  uint64                 `json:"version"`
}
```

A single search result. `Score` is a consumer-facing relevance score where
higher is always better. For cosine collections, this uses cosine similarity
semantics. Other metrics expose a normalized monotone relevance score.

### type SearchResults

```go
type SearchResults struct {
    Results []*SearchResult `json:"results"`
    Took    time.Duration   `json:"took"`
    Total   int             `json:"total"`
}
```

Complete search response. `Results` is ordered by descending score.

### type CollectionStats

```go
type CollectionStats struct {
    Name                 string
    VectorCount          int
    LiveRecordCount      int
    OrdinalUtilization   float64
    NextOrdinal          uint32
    Dimension            int
    IndexType            string
    MemoryUsage          int64
    MemoryStats          *CollectionMemoryStats
    RawVectorStoreStats  *RawVectorStoreStats
    OptimizationStatus   *OptimizationStatus
    HasQuantization      bool
    HasMemoryLimit       bool
    MemoryMappingEnabled bool
}
```

### type CollectionMemoryStats

```go
type CollectionMemoryStats struct {
    Total         int64
    Storage       int64
    Index         int64
    Cache         int64
    Quantized     int64
    MemoryMapped  int64
    Limit         int64
    Available     int64
    PressureLevel string // "normal", "warning", "high", "critical"
    Timestamp     time.Time
}
```

### type RawVectorStoreStats

```go
type RawVectorStoreStats struct {
    Backend             string
    VectorCount         int
    Dimension           int
    BytesPerVector      int
    MemoryUsage         int64
    ReservedBytes       int64
    ReservedDataBytes   int64
    ReservedMetaBytes   int64
    ReservedGuardBytes  int64
    LiveBytes           int64
    FreeBytes           int64
    CapacityUtilization float64
}
```

### type DatabaseStats

```go
type DatabaseStats struct {
    Collections     map[string]*CollectionStats
    CollectionCount int
    MemoryUsage     int64
    Uptime          time.Duration
}
```

### type GlobalMemoryUsage

```go
type GlobalMemoryUsage struct {
    Collections       map[string]*CollectionMemoryStats
    TotalMemory       int64
    TotalIndex        int64
    TotalCache        int64
    TotalQuantized    int64
    TotalMemoryMapped int64
    Timestamp         time.Time
}
```

### type OptimizationOptions

```go
type OptimizationOptions struct {
    RebuildIndex         bool
    OptimizeMemory       bool
    CompactStorage       bool
    UpdateQuantization   bool
    ForceIndexTypeSwitch bool
}
```

### type OptimizationStatus

```go
type OptimizationStatus struct {
    CanOptimize              bool
    InProgress               bool
    LastOptimization         time.Time
    RecommendedOptimizations []string
}
```

### type BatchConfig

```go
type BatchConfig struct {
    ChunkSize        int
    MaxConcurrency   int
    FailFast         bool
    TimeoutPerChunk  time.Duration
}
```

---

## Constants

### type DistanceMetric

```go
type DistanceMetric int

const (
    L2Distance     DistanceMetric = iota // Euclidean (L2) distance
    InnerProduct                         // Dot product similarity
    CosineDistance                       // Cosine distance
)
```

### type IndexType

```go
type IndexType int

const (
    HNSW  IndexType = iota // Hierarchical Navigable Small World
    IVFPQ                  // Inverted File with Product Quantization
    Flat                   // Brute-force exact search
)
```

### type CachePolicy

```go
type CachePolicy int

const (
    LRUCache  CachePolicy = iota // Least Recently Used
    LFUCache                     // Least Frequently Used
    FIFOCache                    // First In, First Out
)
```

### type FieldType

```go
type FieldType int

const (
    StringField      FieldType = iota // string
    IntField                          // int
    FloatField                        // float64
    BoolField                         // bool
    TimeField                         // time.Time
    StringArrayField                  // []string
    IntArrayField                     // []int
    FloatArrayField                   // []float64
)
```

### type MetadataSchema

```go
type MetadataSchema map[string]FieldType
```

Defines the expected types of metadata fields. Call `schema.Validate()` to
check for empty field names or invalid field type values.

---

## Error Handling

### Sentinel Errors

```go
// Lifecycle
var ErrDatabaseClosed     = errors.New("database is closed")
var ErrCollectionClosed   = errors.New("collection is closed")
var ErrCollectionExists   = errors.New("collection already exists")
var ErrTooManyCollections = errors.New("maximum number of collections exceeded")
var ErrCollectionNotFound = errors.New("collection not found")

// Validation
var ErrInvalidDimension   = errors.New("invalid vector dimension")
var ErrDimensionMismatch  = errors.New("collection dimension does not match")
var ErrInvalidK           = errors.New("k must be positive")
var ErrEmptyIndex         = errors.New("index is empty")

// Streaming
var ErrBackpressureActive = errors.New("backpressure is active, cannot send more data")
var ErrStreamingStopped   = errors.New("streaming operation has been stopped")
var ErrStreamingTimeout   = errors.New("streaming operation timed out")
var ErrBufferFull         = errors.New("streaming buffer is full")
var ErrWriteQueueFull     = errors.New("write queue is full")

// Memory
var ErrMemoryLimitExceeded    = errors.New("memory limit exceeded")
var ErrMemoryPressureCritical = errors.New("critical memory pressure detected")
var ErrMemoryMappingFailed    = errors.New("memory mapping operation failed")

// Quantization
var ErrQuantizationNotTrained     = errors.New("quantizer not trained")
var ErrQuantizationTrainingFailed = errors.New("quantization training failed")
var ErrQuantizationCorrupted      = errors.New("quantization data corrupted")

// Index
var ErrIndexCorrupted       = errors.New("index data corrupted")
var ErrIndexTypeMismatch    = errors.New("index type mismatch")
var ErrIndexRebuildRequired = errors.New("index rebuild required")

// Filter
var ErrFilterInvalid         = errors.New("invalid filter expression")
var ErrFilterExecutionFailed = errors.New("filter execution failed")
```

### type VectorDBError

```go
type VectorDBError struct {
    Code           ErrorCode
    Message        string
    Details        any
    Retryable      bool
    Severity       ErrorSeverity
    RecoveryAction RecoveryAction
    Context        *ErrorContext
    Cause          error
    Timestamp      time.Time
    RetryCount     int
    MaxRetries     int
}
```

Structured error with machine-readable codes, severity levels, recovery hints,
and chaining via `Unwrap()`.

### type ErrorCode

```go
type ErrorCode int

const (
    ErrCodeUnknown             ErrorCode = iota
    ErrCodeInvalidVector
    ErrCodeIndexCorrupted
    ErrCodeStorageFailure
    ErrCodeMemoryExhausted
    ErrCodeTimeout
    ErrCodeRateLimited
    ErrCodeMemoryPressure
    ErrCodeMemoryMappingFailure
    ErrCodeCacheFailure
    ErrCodeQuantizationFailure
    ErrCodeQuantizationCorruption
    ErrCodeQuantizationTraining
    ErrCodeBatchFailure
    ErrCodeBatchTimeout
    ErrCodeBatchSizeLimit
    ErrCodeIndexFailure
    ErrCodeIndexRebuild
    ErrCodeFilterFailure
    ErrCodeFilterInvalid
)
```

### type ErrorSeverity

```go
type ErrorSeverity int

const (
    SeverityInfo     ErrorSeverity = iota
    SeverityWarning
    SeverityError
    SeverityCritical
    SeverityFatal
)
```

### type RecoveryAction

```go
type RecoveryAction int

const (
    RecoveryNone                 RecoveryAction = iota
    RecoveryRetry
    RecoveryFallback
    RecoveryGracefulDegradation
    RecoveryRestart
    RecoveryRebuild
)
```

### Creating Structured Errors

```go
func NewVectorDBError(code ErrorCode, message string, retryable bool) *VectorDBError
func NewVectorDBErrorWithContext(code ErrorCode, message string, retryable bool, component, operation string) *VectorDBError
```

Builder methods on `*VectorDBError`:

```go
func (e *VectorDBError) WithCause(cause error) *VectorDBError
func (e *VectorDBError) WithSeverity(severity ErrorSeverity) *VectorDBError
func (e *VectorDBError) WithRecoveryAction(action RecoveryAction) *VectorDBError
func (e *VectorDBError) WithMetadata(key string, value interface{}) *VectorDBError
func (e *VectorDBError) WithRequestID(requestID string) *VectorDBError
```

### Error Handling Example

```go
_, err := col.Search(ctx, queryVec, 10)
if err != nil {
    var vdbErr *libravdb.VectorDBError
    if errors.As(err, &vdbErr) {
        switch vdbErr.Code {
        case libravdb.ErrCodeMemoryExhausted:
            // trigger memory reclamation
        case libravdb.ErrCodeIndexCorrupted:
            // rebuild index
        }
        if vdbErr.IsRetryable() {
            // retry with backoff
        }
    } else if errors.Is(err, libravdb.ErrCollectionClosed) {
        // collection was closed, re-open or fail
    }
}
```

### type BatchError

```go
type BatchError struct {
    Code        BatchErrorCode
    Operation   string
    Message     string
    BatchSize   int
    Cause       error
    Retryable   bool
    Recoverable bool
    ItemErrors  map[int]error
    Metadata    map[string]interface{}
    Duration    time.Duration
    Timestamp   time.Time
}
```

Batch-specific error with per-item error tracking.

```go
func NewBatchError(code BatchErrorCode, operation, message string, batchSize int) *BatchError
```

---

## Health & Observability

### type HealthStatus

```go
type HealthStatus struct {
    Overall    HealthLevel
    Components map[string]HealthLevel
    Details    map[string]any
    Timestamp  time.Time
}
```

### type HealthLevel

```go
type HealthLevel int

const (
    HealthUnknown   HealthLevel = iota
    HealthHealthy
    HealthDegraded
    HealthUnhealthy
    HealthCritical
)
```

### type HealthCheck

```go
type HealthCheck func(ctx context.Context) (HealthLevel, error)
```

### type SystemHealthMonitor

```go
type SystemHealthMonitor interface {
    GetHealthStatus() HealthStatus
    RegisterHealthCheck(name string, check HealthCheck) error
    UnregisterHealthCheck(name string) error
    Start(ctx context.Context) error
    Stop() error
}
```

### type CircuitBreaker

```go
type CircuitBreaker interface {
    Execute(ctx context.Context, fn func() error) error
    State() string // "CLOSED", "OPEN", "HALF_OPEN"
    Reset()
}
```

### type GracefulDegradationManager

```go
type GracefulDegradationManager interface {
    HandleMemoryPressure(ctx context.Context, pressureLevel int) error
    HandleQuantizationFailure(ctx context.Context, fallbackEnabled bool) error
    HandleIndexCorruption(ctx context.Context, rebuildEnabled bool) error
    GetDegradationLevel() int
    SetDegradationLevel(level int) error
}
```

### Recovery Orchestration

```go
func NewErrorRecoveryManager() *ErrorRecoveryManager
func NewAutomaticRecoveryOrchestrator(erm *ErrorRecoveryManager) *AutomaticRecoveryOrchestrator
```

The `ErrorRecoveryManager` coordinates pluggable recovery strategies per error
code. The `AutomaticRecoveryOrchestrator` provides cross-component recovery
coordination with attempt tracking and statistics.

```go
type RecoveryStats struct {
    TotalAttempts      int
    SuccessfulAttempts int
    SuccessRate        float64
    AverageDuration    time.Duration
    ByErrorCode        map[ErrorCode]int
    ByComponent        map[string]int
}
```

---

## Logger Interface

```go
type Logger interface {
    Printf(format string, v ...interface{})
}
```

Compatible with the standard library's `log.Printf` signature. Set via
`WithLogger`. When set, the database emits timing instrumentation for index
rebuilds during transaction commits.

---

## Migration

```go
func Migrate(ctx context.Context, path string) error
```

Migrates a v1-format database at `path` to the current format. Returns an error
if the path does not contain a valid v1 database or migration fails.
`Open` handles this automatically; call `Migrate` directly only for pre-flight
migration or offline maintenance.
