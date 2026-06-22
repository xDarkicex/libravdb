# Transaction Design

This document describes the transaction system in LibraVDB — the commit
protocol, compare-and-swap semantics, hook integration, and recovery guarantees.

## Overview

LibraVDB transactions provide atomic, cross-collection mutation with
serializable isolation at the WAL commit boundary. The design follows a
**staged in-memory mutation** model: mutations are accumulated in a transaction
buffer, validated, and atomically committed to the WAL. If the commit succeeds,
all mutations are durable. If it fails (or `Rollback` is called), none are
applied.

## Architecture

```
Application
    │
    ▼
BeginTx / WithTx
    │
    ▼
┌─────────────────────────────────┐
│         transaction             │
│                                 │
│  ops []txMutation               │  ← staged mutations (in-memory)
│  mu  sync.Mutex                 │
│                                 │
│  Insert() ─────────────────►    │
│  Update() ─────────────────►    │
│  Delete() ─────────────────►    │
│  Upsert() ─────────────────►    │
│  UpdateIfVersion() ────────►    │
│  DeleteIfVersion() ────────►    │
│  DeleteBatch() ────────────►    │
│  ListByMetadata() ─────────►    │
│                                 │
│  Commit() ───► commitTx()       │
│  Rollback() ─► discard ops      │
└─────────────────────────────────┘
    │
    ▼  commitTx()
┌─────────────────────────────────┐
│  1. acquire collection locks    │
│  2. validate CAS preconditions  │
│  3. assign ordinals             │
│  4. build storage operations    │
│  5. commit to WAL               │
│  6. apply to in-memory index    │
│  7. release locks               │
└─────────────────────────────────┘
```

## The Tx Interface

```go
type Tx interface {
    // Mutations
    Insert(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}) error
    InsertOwned(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}) error
    Update(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}) error
    UpdateOwned(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}) error
    Upsert(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}) error
    UpdateIfVersion(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}, expectedVersion uint64) error
    Delete(ctx context.Context, collection, id string) error
    DeleteIfVersion(ctx context.Context, collection, id string, expectedVersion uint64) error
    DeleteBatch(ctx context.Context, collection string, ids []string) error

    // Reads (within transaction context)
    ListByMetadata(ctx context.Context, collection, field string, value interface{}) ([]Record, error)

    // Lifecycle
    Commit(ctx context.Context) error
    Rollback(ctx context.Context) error
}
```

All mutation methods are staging operations — they validate the input and queue
the mutation. No mutation is visible to other readers until `Commit` succeeds.

## Mutation Types

Each staged mutation is represented internally as:

```go
type txMutationKind uint8

const (
    txInsert         txMutationKind = iota
    txInsertOwned
    txUpdate
    txUpdateOwned
    txUpsert
    txUpdateIfVersion
    txDelete
    txDeleteIfVersion
)

type txMutation struct {
    kind            txMutationKind
    collection      string
    id              string
    vector          []float32
    metadata        map[string]interface{}
    expectedVersion uint64
    hasVersion      bool
}
```

The `*Owned` variants (`InsertOwned`, `UpdateOwned`) take ownership of the
caller's `vector` slice — the caller must not reuse the slice after the call.
This avoids a defensive copy in performance-sensitive paths.

## Transaction Lifecycle

### 1. Begin

```go
tx, err := db.BeginTx(ctx)
```

Or use the convenience wrapper:

```go
err := db.WithTx(ctx, func(tx Tx) error {
    // stage mutations
    return nil // commit
})
```

`WithTx` automatically rolls back if `fn` returns an error or panics. It
commits if `fn` returns nil.

### 2. Stage

Mutations are validated immediately (dimension check, non-empty ID, collection
exists) but not applied. Validation errors are returned from the staging call.
The transaction buffer is protected by a mutex; staging is serialized within a
single transaction.

### 3. Commit

`Commit` executes the following protocol:

1. **Acquire collection locks** — all collections touched by the transaction are
   locked for writing.
2. **Validate CAS preconditions** — for `UpdateIfVersion` / `DeleteIfVersion`,
   the current record version is checked against `expectedVersion`. A mismatch
   returns `*VersionConflictError` and aborts the commit.
3. **Assign ordinals** — new records are allocated storage slots from the
   collection's ordinal counter.
4. **Build storage operations** — all mutations are converted to
   `storage.TxOperation` values.
5. **Commit to WAL** — the storage engine atomically writes all operations
   within a single WAL transaction (`TX_BEGIN` ... `TX_COMMIT`).
6. **Apply to indexes** — each collection's in-memory index is updated with the
   committed mutations.
7. **Invoke hooks** — `InsertHook` and `DeleteHook` callbacks are invoked with
   the committed mutations. Hook errors do not abort the commit (hooks are
   post-commit).
8. **Release locks** — collection locks are released.

### 4. Rollback

`Rollback` discards all staged mutations. The transaction buffer is cleared.
No locks are held during rollback (they are only acquired at commit time).

### 5. Garbage Collection

If a transaction is neither committed nor rolled back, it is discarded when the
`Tx` reference is garbage-collected. Staged mutations are lost. This is safe
because no mutations are visible until commit.

## Compare-and-Swap (CAS)

LibraVDB supports optimistic concurrency control via versioned records. Every
record has a `Version uint64` that is incremented on each update.

### UpdateIfVersion

```go
err := tx.UpdateIfVersion(ctx, "col", "doc-1", newVec, newMeta, 5)
```

Succeeds only if the record's current version equals `5`. Returns
`*VersionConflictError` on mismatch.

### DeleteIfVersion

```go
err := tx.DeleteIfVersion(ctx, "col", "doc-1", 5)
```

Succeeds only if the record's current version equals `5`.

### VersionConflictError

```go
type VersionConflictError struct {
    ID              string
    ExpectedVersion uint64
    ActualVersion   uint64
}
```

```go
var ErrVersionConflict = errors.New("version conflict")
```

Use `errors.Is(err, ErrVersionConflict)` to detect version conflicts:

```go
if errors.Is(err, libravdb.ErrVersionConflict) {
    var vc *libravdb.VersionConflictError
    errors.As(err, &vc)
    log.Printf("version conflict on %s: expected %d, got %d",
        vc.ID, vc.ExpectedVersion, vc.ActualVersion)
}
```

## Hook Integration

Hooks enable graph-layer side effects during vector mutations. They are
registered on the collection and invoked during transaction commit.

### InsertHook

```go
type InsertHook func(txn GraphTx, id uint64, vector []float32, metadata map[string]interface{}) error
```

Called for each inserted vector after the WAL commit. The `id uint64` parameter
is the storage-level ordinal (not the string ID). The hook receives a
`GraphTx` for adding/removing edges.

### DeleteHook

```go
type DeleteHook func(txn GraphTx, id uint64) error
```

Called for each deleted vector after the WAL commit.

### GraphTx

```go
type GraphTx interface {
    AddEdge(src, tgt uint64, weight float32, kind uint8) error
    RemoveEdge(src, tgt uint64, kind uint8) error
}
```

Edges added/removed within hooks are committed atomically with the vector
mutation. The graph transaction scope is the same as the database transaction.

### Hook Execution Order

1. All storage mutations are committed to the WAL.
2. Indexes are updated.
3. `InsertHook` is called for each inserted vector, in insertion order.
4. `DeleteHook` is called for each deleted vector, in deletion order.

Hook errors are logged but do not cause the transaction to roll back. The
vector mutations are already durable by the time hooks execute.

## Concurrency Model

### Single-Writer Commit

Only one transaction can commit at a time. The WAL provides a serialization
point: all mutations in a transaction are written as a single contiguous WAL
segment bracketed by `TX_BEGIN` and `TX_COMMIT` frames.

### Read Isolation

Readers see a consistent snapshot: either all mutations from a committed
transaction are visible, or none are. This is achieved through the metapage
switch mechanism — the durable root is published atomically.

### Staging Isolation

Multiple transactions can stage mutations concurrently (each has its own
buffer). Only `Commit` serializes. This means two concurrent transactions can
stage conflicting mutations (e.g., both delete the same record); the first to
commit succeeds, the second may encounter a not-found condition during CAS
validation.

## WAL Frame Types

The WAL records the following frame types for transactions:

| Frame Type | Purpose |
|---|---|
| `TX_BEGIN` | Marks the start of a transaction. |
| `RECORD_PUT` | Insert or update a record. |
| `RECORD_DELETE` | Delete a record. |
| `COLLECTION_CREATE` | Create a new collection. |
| `COLLECTION_DELETE` | Delete a collection. |
| `COLLECTION_UPDATE_META` | Update collection metadata. |
| `INDEX_SNAPSHOT_INSTALL` | Install a pre-built index snapshot. |
| `TX_COMMIT` | Marks the successful end of a transaction. All preceding frames become durable. |
| `TX_ABORT` | Marks an aborted transaction. Frames are ignored during recovery. |

## Recovery

During startup, the recovery algorithm:

1. Loads the active metapage to determine the last checkpoint LSN.
2. Scans the WAL from `last_applied_lsn + 1`.
3. Groups frames by `txid`.
4. Replays only committed transactions (`TX_COMMIT` present) in LSN order.
5. Skips uncommitted or aborted transactions (`TX_ABORT` or no `TX_COMMIT`).

This guarantees that after recovery, the database state reflects all and only
committed transactions.

## Batch Operations vs Transactions

Batch operations (`BatchInsert`, `BatchUpdate`, `BatchDelete`) are not
transactional in the same way — they process items in chunks, and each chunk
may be committed independently. For atomic multi-record mutation, use `Tx`.

However, batch operations *within* a single collection do provide chunk-level
atomicity and optional rollback (`EnableRollback: true`), which reverses
successfully-processed items if a later chunk fails.

## Best Practices

1. **Keep transactions short.** Long-running transactions hold collection locks
   at commit time and block other writers.
2. **Use CAS for safe updates.** `UpdateIfVersion` prevents lost-update
   scenarios when multiple writers may modify the same record.
3. **Stage reads early.** `ListByMetadata` within a transaction sees the
   committed state, not the staged mutations.
4. **Prefer `WithTx` over manual `BeginTx`/`Commit`/`Rollback`.** It ensures
   proper cleanup even on panic.
5. **Use `*Owned` variants in hot paths.** When the caller can transfer
   ownership of the vector slice, `InsertOwned` / `UpdateOwned` avoids an
   unnecessary allocation.
