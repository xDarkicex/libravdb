# Batch Operations Design

This document describes the design and implementation of batch operations in
LibraVDB — chunked processing, concurrency, retries, rollback, and error
recovery.

## Overview

Batch operations provide fine-grained control over bulk vector mutations. They
split work into chunks, process chunks concurrently through a bounded worker
pool, support configurable retry logic, and can roll back partial success.

## Architecture

```
Application
    │  NewBatchInsert(entries, opts)
    ▼
┌─────────────────────────────┐
│         BatchInsert          │
│  entries []*VectorEntry     │
│  options *BatchOptions      │
│  progressTracker            │
│  insertedIDs []string       │ ← for rollback
└──────────┬──────────────────┘
           │ Execute(ctx)
           ▼
    ┌──────────────┐
    │  Chunk loop   │
    │  chunkSize N  │
    └──┬───────────┘
       │
       ├──► processChunk() ──► tryProcessChunkFast()  ← arena-based fast path
       │                              │
       │                              ├─ success → merge results
       │                              └─ fallback → processChunkIndividually()
       │
       ├──► Progress callbacks
       ├──► FailFast check
       └──► Context cancellation check
```

## Three Batch Types

### BatchInsert

```go
type BatchInsert struct {
    collection      *Collection
    options         *BatchOptions
    progressTracker *progressTracker
    entries         []*VectorEntry
    insertedIDs     []string   // for rollback
}
```

Processes `[]*VectorEntry` through the collection's insert path.

### BatchUpdate

```go
type BatchUpdate struct {
    collection       *Collection
    options          *BatchOptions
    progressTracker  *progressTracker
    updates          []*VectorUpdate
    modifiedIDs      []string
    originalEntries  []*VectorEntry  // for rollback
}
```

Each `VectorUpdate` specifies an ID, optional new vector, optional new
metadata, and an `Upsert` flag. For non-upsert updates, the original entry
is preserved for potential rollback.

### BatchDelete

```go
type BatchDelete struct {
    collection      *Collection
    options         *BatchOptions
    progressTracker *progressTracker
    ids             []string
    deletedEntries  []*VectorEntry  // for rollback
}
```

Deletes entries by ID. Before deletion, the original entry is captured for
potential rollback re-insertion.

## Chunk Processing

### Fast Path (Arena-Based)

When an entire chunk can be processed in one call, the batch uses an
arena-backed fast path:

```go
func (b *BatchInsert) tryProcessChunkFast(ctx context.Context, chunk []*VectorEntry, startIndex int) (*chunkResult, bool, error)
```

1. Acquire a `*memory.Arena` from the database scratch pool.
2. Allocate `[]index.VectorEntry` and `[]*index.VectorEntry` slices from the
   arena (zero heap allocations).
3. Validate all entries in the chunk.
4. Call `collection.insertBatch(ctx, indexEntries)` — a single batch index
   insert.
5. Return results; arena is reset and returned to the pool.

If validation fails or the arena allocation fails, the function signals
"not handled" and the caller falls back to the individual path.

### Individual Path

```go
func (b *BatchInsert) processChunkIndividually(ctx context.Context, chunk []*VectorEntry, startIndex int, chunkIdx int) (*chunkResult, error)
```

Processes each entry one at a time with per-item retries. Used when the
fast path cannot handle the chunk (e.g., validation errors, mixed shard
distribution).

### Concurrent Execution

When `MaxConcurrency > 1` and there are multiple chunks, the batch uses
a worker pool for parallel chunk processing:

```go
func (b *BatchInsert) executeConcurrent(ctx context.Context, result *BatchResult, startTime time.Time, chunkSize, totalChunks int, workerConcurrency int) (*BatchResult, error)
```

Chunks are submitted to a `workerPool`. Results arrive on a channel and
are merged in order. `FailFast` cancellation is handled via context
cancellation.

## Retry Logic

Each individual item is processed with configurable retry:

```go
func (b *BatchInsert) processItemWithRetries(ctx context.Context, entry *VectorEntry, itemResult *BatchItemResult) (bool, error)
```

- Retries up to `MaxRetries` times (default: 3).
- Non-retryable errors (dimension mismatch, empty ID, duplicates) skip retry.
- Configurable `RetryDelay` between attempts (default: 100ms).
- Context cancellation is checked before each retry.
- Each attempt increments `itemResult.Retries`.

## Rollback

When `EnableRollback` is true, failed batch operations can reverse
successfully-processed items:

- **BatchInsert**: Deletes all successfully inserted IDs.
- **BatchUpdate**: Restores original entries (vector + metadata) for modified IDs.
- **BatchDelete**: Re-inserts original entries for deleted IDs.

Rollback is triggered on:
- A chunk processing error.
- `FailFast` encountering the first failure.
- Context cancellation.

Rollback runs in a best-effort background context (not the cancelled parent
context) to ensure cleanup proceeds.

## Error Categorization

Errors are categorized into standard codes for structured reporting:

```go
const (
    BatchErrorValidation   = "VALIDATION_ERROR"    // dimension mismatch, empty ID
    BatchErrorDuplicate    = "DUPLICATE_ERROR"     // duplicate ID
    BatchErrorNotFound     = "NOT_FOUND_ERROR"     // ID not found for update/delete
    BatchErrorTimeout      = "TIMEOUT_ERROR"       // context deadline exceeded
    BatchErrorMemory       = "MEMORY_ERROR"        // memory limit reached
    BatchErrorInternal     = "INTERNAL_ERROR"       // unexpected internal error
    BatchErrorCancellation = "CANCELLATION_ERROR"   // context cancelled
)
```

## Progress Tracking

The `progressTracker` provides real-time progress:

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

Two callback styles are supported:
- `ProgressCallback(completed, total int)` — simple count-based.
- `DetailedProgress(progress *BatchProgress)` — full progress struct with ETA.

## Batch Recovery Manager

For programmatic recovery from batch failures:

```go
type BatchRecoveryManager struct {
    maxRetryAttempts int
    retryBackoff     time.Duration
}
```

Recovery strategies by error type:
- **Size exceeded** → auto-split into optimal chunks.
- **Timeout** → reduce chunk size, retry with extended timeout.
- **Partial failure** → collect failed indices, retry only failed items.
- **Memory exhaustion** → trigger GC, reduce concurrency, retry.
- **Concurrency limit** → reduce worker count, retry sequentially.

## Batch Operation Tracker

The `BatchOperationTracker` maintains a registry of in-flight and completed
batch operations:

```go
type BatchOperationTracker struct {
    operations map[string]*BatchOperationStatus
    mu         sync.RWMutex
}

type BatchOperationStatus struct {
    ID           string
    Operation    string
    BatchSize    int
    Processed    int
    Failed       int
    Status       string  // "running", "completed", "failed"
    StartTime    time.Time
    EndTime      time.Time
    Error        *BatchError
}
```

`CleanupCompletedOperations(maxAge)` periodically removes completed operations
older than `maxAge` to prevent unbounded memory growth.

## Usage Recommendations

1. **Tune chunk size** based on vector dimension. Larger dimensions (768+)
   should use smaller chunks (500–2000).
2. **Set MaxConcurrency** to `runtime.NumCPU()` for CPU-bound work, lower for
   I/O-bound work.
3. **Enable rollback** only when partial failure is unacceptable — it adds
   overhead from capturing original entries.
4. **Use the fast path** by ensuring entries are valid before submission; the
   arena-based path is 2–3x faster than individual processing.
5. **Monitor progress** via `DetailedProgress` for long-running batches.
