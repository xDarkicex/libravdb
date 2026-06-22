# Concurrency Design

This document describes the concurrency model, locking strategy, and goroutine
lifecycle management in LibraVDB.

## Design Principles

1. **Share memory by communicating.** Transfer ownership of data via channels
   instead of protecting shared state with locks wherever possible.
2. **Context is mandatory.** Every blocking operation — I/O, channel send/receive,
   DB query — must accept and respect `context.Context`.
3. **Goroutines are cheap, not free.** Unbounded goroutine creation causes OOM.
   Cap concurrency explicitly via worker pools and semaphores.
4. **Channels have direction.** Declare `chan<- T` and `<-chan T` in function
   signatures to make data flow explicit.
5. **Only the sender closes a channel.** Closing from the receiver is a race.
   The producer owns the channel lifecycle.

## Locking Hierarchy

```
Database.mu (RWMutex)
    │
    ├── collection registration/deregistration
    │
    └── Collection.mu (RWMutex)
        │
        ├── index operations (insert, search, delete)
        ├── storage I/O
        ├── config reads
        │
        └── Shard-level locks (per-shard mutex for sharded collections)
```

### Lock Ordering Rules

1. **Database lock before collection lock.** Never acquire a collection lock
   while holding another collection's lock.
2. **Read locks are shared.** Multiple readers can search concurrently.
3. **Write locks are exclusive.** An insert blocks other inserts on the same
   collection, but does not block searches on other collections.
4. **Stripe locks for per-ID serialization.** The collection uses FNV-32a-based
   striping to serialize mutations on the same ID without locking the entire
   collection.

### Stripe Locking

```go
const stripeCount = 64

func (c *Collection) mutationStripe(id string) *sync.Mutex {
    h := fnv32a(id)
    return &c.stripes[h%stripeCount]
}
```

When processing batch inserts, the collection acquires stripe locks for all
affected IDs in hash order to prevent deadlock:

```go
func (c *Collection) lockMutationIDs(ids []string) func() {
    // Sort stripes by index to guarantee lock ordering
    // Acquire each unique stripe once
    // Return a single unlock function
}
```

## Write Controller

The database employs a write controller to bound concurrent write operations:

```go
type writeController struct {
    sem     chan struct{}  // buffered channel as semaphore
    max     int
}
```

- `acquire(ctx)` — blocks until a write slot is available or ctx is cancelled.
  Returns a release function.
- `release()` — returns a write slot to the pool.
- `maxParallelism()` — returns the configured maximum.

The controller is configured via `WithMaxConcurrentWrites` (default: `runtime.NumCPU()`)
and `WithMaxWriteQueueDepth` (default: `32`). When the queue depth is reached,
`acquire` returns `ErrWriteQueueFull`.

## Worker Pool

Batch operations use a bounded worker pool:

```go
type workerPool struct {
    jobs    chan func() error
    errs    chan error
    workers int
    wg      sync.WaitGroup
}
```

- `newWorkerPool(workers int)` — creates a pool with `workers` goroutines.
- `submit(job func() error)` — enqueues a job; blocks if the channel is full.
- `wait(ctx)` — waits for all workers to finish; returns the first error.
- `close()` — closes the job channel after all submissions are done.

Workers exit when the job channel is closed.

## Backpressure

The streaming API implements backpressure to prevent OOM when producers are
faster than consumers:

```go
type BackpressureController struct {
    enabled           bool
    threshold         float64  // 0.0–1.0
    maxMemoryUsage    int64
    currentBufferSize int32
    active            int32
}
```

`ShouldApplyBackpressure()` returns `true` when buffer utilization exceeds
the threshold (default `0.8` / 80%). The streaming `Send` method blocks when
backpressure is active, propagating the slowdown upstream.

For lossy workloads (metrics, telemetry), an explicit drop policy is available:

```go
func emitOrDrop(ch chan<- Metric, m Metric) (dropped bool) {
    select {
    case ch <- m:
        return false
    default:
        return true
    }
}
```

## Scratch Pool

The database maintains a `sync.Pool` of `*memory.Arena` instances for
allocation-free temporary memory in hot paths:

```go
scratchPool: &sync.Pool{
    New: func() interface{} {
        arena, _ := memory.NewArena(1024 * 1024)
        return arena
    },
}
```

Batch operations acquire an arena, use it for temporary slices and structs,
then reset and return it to the pool:

```go
arena := db.scratchPool.Get().(*memory.Arena)
defer func() {
    arena.Reset()
    db.scratchPool.Put(arena)
}()
```

## Goroutine Lifecycle

### Errgroup for Supervised Goroutines

The codebase uses `golang.org/x/sync/errgroup` for goroutine groups where any
failure should cancel the entire group:

```go
g, ctx := errgroup.WithContext(ctx)
for i := 0; i < workers; i++ {
    g.Go(func() error {
        // work
    })
}
return g.Wait()
```

### Exit Conditions

Every goroutine must have a documented, tested exit condition:

1. **Context cancellation** — `<-ctx.Done()` is checked in all select loops.
2. **Channel close** — `for range ch` exits when the channel is closed.
3. **Done signal** — `<-doneChan` provides an explicit stop signal.

### Leak Detection

Tests use `go.uber.org/goleak` to verify no goroutines outlive the test:

```go
func TestMain(m *testing.M) {
    goleak.VerifyTestMain(m)
}
```

## Concurrent Read/Write Semantics

| Operation | Reads during Insert | Inserts during Search | Multiple Readers |
|-----------|--------------------|----------------------|--------------------|
| Search | Consistent snapshot | Blocked (write lock) | Concurrent (RLock) |
| Insert | Blocked (write lock) | Serialized (write lock) | N/A |
| Get | Sees committed state | Sees pre-insert state | Concurrent |
| Iterate | Consistent snapshot | May or may not see in-flight insert | Concurrent |
| Count | Consistent snapshot | May or may not include in-flight insert | Concurrent |

## Transaction Concurrency

1. **Staging is parallel.** Multiple transactions can stage mutations
   concurrently (each has its own buffer).
2. **Commit is serialized.** Only one transaction commits at a time. The WAL
   serves as the serialization point.
3. **Readers never block on staging.** In-memory indexes are not modified until
   commit.
4. **Readers briefly coordinate at commit.** The metapage switch is atomic;
   readers see either the old root or the new root, never a partial commit.

## Sharded Collections

When sharding is enabled (`WithSharding(true)`), the collection is split into
4 internal shards. Each shard has its own index, storage handle, and lock:

```go
type shard struct {
    index   index.Index
    storage storage.Collection
    mu      sync.RWMutex
}
```

Sharding allows parallel writes to different shards. The shard is selected by
hashing the record ID:

```go
func shardForID(id string) int {
    h := fnv.New32a()
    h.Write([]byte(id))
    return int(h.Sum32()) % shardCount
}
```

Searches query all shards and merge results.

## Atomic State

For lock-free counters and flags, the codebase uses typed atomics from
`sync/atomic` (Go 1.19+):

```go
// Streaming state flags
started int32  // atomic: 0=not started, 1=started
stopped int32  // atomic: 0=running, 1=stopped

// Usage
atomic.CompareAndSwapInt32(&s.started, 0, 1)
atomic.LoadInt32(&s.stopped)
```

## Performance Considerations

1. **Cache-line padding.** Where false sharing is a concern, structs include
   padding fields to ensure hot fields occupy separate cache lines.
2. **RWMutex over Mutex.** Read-heavy paths use `sync.RWMutex` so concurrent
   searches do not contend.
3. **Channel buffer sizing.** Channels are buffered to the expected steady-state
   capacity to avoid unnecessary goroutine parking.
4. **Batch commits.** The storage engine supports group commit for write
   throughput under load.
