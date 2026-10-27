# Committed change feed

The native Go API exposes compact, committed record and graph mutation references.
It does not run consumers, infer domain meaning, retain duplicate payloads, or
provide replication/synchronization between databases.

```go
cursor, err := db.ChangeCheckpoint(ctx) // save alongside your initial baseline
if err != nil { return err }
// Later, possibly after closing and reopening the database:
page, err := db.ReadChanges(ctx, cursor, 200)
if err != nil { return err }
// Process parallel columns in page.Rows. Persist page.Next after processing.
// Continue while page.More; use the returned cursor unchanged.
```

`ReadChanges(ctx, "", limit)` begins at the available feed baseline, reported as
`StartLSN`. It does not assert that all history since database creation is present.
A database upgraded from an older snapshot starts at that snapshot's committed
position; legacy WAL commits without feed lineage markers advance that boundary.
Consumers needing a complete project state must first establish a baseline using
normal reads. For a race-free bootstrap, checkpoint **before** collecting that
baseline and apply subsequent changes idempotently (or use an explicitly pinned
snapshot for the baseline).

## Cursors and ordering

Cursors are opaque, versioned encodings of the database file identity, feed
baseline, committed LSN, mutation offset and a random per-commit lineage marker.
They survive checkpointing, compaction, and reopen. `errors.Is(err,
libravdb.ErrChangeCursor)` identifies malformed, foreign, unavailable, or divergent
positions. Restoring a prefix invalidates cursors that refer to its discarded
suffix. A physical clone preserving the same prefix accepts that prefix's cursor;
it is not a separate logical identity until its histories diverge. Cursors are
coordination tokens, not authentication credentials.

Rows are ordered by committed transaction LSN, then mutation position in the WAL.
A `record_put` is an upsert, not a distinction between insert and update. The
columns include `BeforeLSNs` (state before the transaction), `CommitLSNs` (state
after the transaction), `MutationLSNs` (individual WAL references), `Positions`
(zero-based) and `TransactionSizes`. Metadata and vectors can be read separately
at the committed state; those historical reads remain subject to temporal retention.
Several puts to a record within one transaction need not have separately queryable
intermediate snapshots. This feed is a mutation-reference stream, not a payload
replication protocol.

`record_put`/`record_delete` use collection and record ID. `edge_add`/`edge_remove`
use collection, graph source/target IDs and numeric edge kind. `node_drop` refers
to a graph node and removal of its incident edges, rather than enumerating every
removed edge. `vertex_label` uses `Sources` for node ID and `RecordIDs` for label;
`edge_kind` uses `RecordIDs` for registry name and `EdgeKinds` for numeric kind.
Collection create/delete/config/stats changes use collection names. Allocation
reservations are internal: they advance the watermark but produce no rows.

## Bounded pagination and atomicity

`limit` must be 1..10000. A page examines at most that many transaction entries and
returns at most that many mutations. A large transaction can span pages. Every
row names its transaction, position and transaction size. Buffer a transaction
until its final row before exposing it to an atomic downstream projection.
`TransactionComplete` indicates whether the page cursor ends at a transaction
boundary; a page can contain several complete transactions followed by one partial
transaction. `More` means more of the captured committed prefix remains, including
possibly internal transactions with no visible rows. An empty page with `More`
set must still be followed. `Watermark` is the checkpoint captured with that page;
new writes can extend the feed on a later call.

Only committed frames are published. Rolled-back and failed pre-commit writes
produce no rows. An ambiguous sync failure is not published by the failing handle;
recovery can subsequently find a complete committed WAL transaction and expose it.
Reads refresh remote writers before answering, without producing a new commit.
Returned slices belong to the caller; strings are immutable Go values.

## Persistence and cost

Compact references are derived from existing WAL frames before pooled payloads are
released and published after the existing sync succeeds. They are stored in snapshot
codec v9 and reconstructed from committed post-snapshot WAL on recovery. This adds
16 bytes to each new transaction's commit payload for its lineage marker, plus
compact references to checkpoints. Older binaries cannot read v9 snapshots;
downgrading or mixing old and new writers is unsupported.

References currently have indefinite retention and grow with committed history.
Temporal payload compaction does not remove them. The feed is held in memory and
included in checkpoints, so this first version targets embedded workspace histories;
it is not an out-of-core change archive. Polling seeks by binary search over commit
positions and reads only the bounded page; it does not scan live records or their
metadata. Benchmarks are recorded in `docs/benchmarks/change-feed.txt`.
