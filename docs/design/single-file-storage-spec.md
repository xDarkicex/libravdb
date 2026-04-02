# Single-File Storage Spec

Public contract: one directly-written file, `*.libravdb`.

Design goals:
- single movable artifact
- fast append/write path
- restart-safe collection discovery
- exact recovery semantics
- storage-versioned from day one
- no dependency on external sidecar files

## 1. File Layout

File is page-based.

Defaults:
- page size: `4096` bytes
- all multibyte integers: little-endian
- all offsets: `uint64`
- all page IDs: `uint64`
- all checksums: `crc32c`

High-level regions:
1. file header page `0`
2. metapage A `1`
3. metapage B `2`
4. WAL region starting at page `3`
5. data/index/catalog pages after WAL growth boundary

The WAL can grow forward; background checkpoint/compaction rewrites stable pages and advances the durable root.

## 2. Page Layout

Every page begins with a fixed header:

```text
struct PageHeader {
    uint32 magic;
    uint16 version;
    uint16 page_type;
    uint64 page_id;
    uint64 lsn;
    uint32 payload_len;
    uint32 checksum;
}
```

Page types:
- `META`
- `CATALOG_INTERNAL`
- `CATALOG_LEAF`
- `COLLECTION_INTERNAL`
- `COLLECTION_LEAF`
- `VECTOR_BLOB`
- `INDEX_BLOB`
- `FREELIST`
- `WAL_FRAME`
- `OVERFLOW`

Rules:
- pages are immutable once checkpointed, except metapages and WAL append area
- large values spill to overflow/blob chains
- logical trees use copy-on-write page replacement

## 3. File Header

Page `0` is fixed-size header.

```text
struct FileHeader {
    byte[8] magic = "LIBRAVDB";
    uint16 format_version;
    uint16 page_size;
    uint32 feature_flags;
    uint64 file_id;
    uint64 created_unix_nano;
    uint64 last_checkpoint_lsn;
    uint64 active_meta_page;
    uint64 wal_start_page;
    uint64 wal_head_page;
    uint64 wal_tail_page;
    uint32 checksum;
}
```

Purpose:
- identify file
- locate active metapage
- locate WAL boundaries
- support crash-safe root switching

## 4. Metapages

Two metapages, A and B, alternating on each successful checkpoint.

```text
struct MetaPage {
    uint64 meta_epoch;
    uint64 root_catalog_page;
    uint64 root_freelist_page;
    uint64 last_applied_lsn;
    uint64 page_count;
    uint64 collection_count;
    uint32 checksum;
}
```

Recovery picks the valid metapage with highest `meta_epoch`.

## 5. Catalog Records

Catalog is a B+tree keyed by collection name.

Key:
- `collection_name` as UTF-8 bytes

Value:

```text
struct CatalogRecord {
    uint64 collection_id;
    uint64 config_version;
    uint64 created_lsn;
    uint64 updated_lsn;
    uint64 root_data_page;
    uint64 root_id_index_page;
    uint64 root_meta_index_page;
    uint64 root_vector_index_blob;
    uint64 live_count;
    uint64 tombstone_count;
    uint32 state_flags;
    uint32 metadata_len;
    byte[metadata_len] encoded_collection_metadata;
}
```

Deleted collections:
- not physically removed immediately
- state flips to `deleted`
- hidden from normal catalog scans after delete commit
- reclaimed at compaction

## 6. Collection Metadata Encoding

Use versioned binary encoding, not JSON, for core metadata.

```text
struct CollectionMetadataV1 {
    uint16 schema_version;
    uint16 distance_metric;
    uint16 index_type;
    uint16 reserved;
    uint32 dimension;
    uint32 hnsw_m;
    uint32 hnsw_ef_construction;
    uint32 hnsw_ef_search;
    float64 hnsw_ml;
    uint16 raw_vector_store;
    uint16 compression_type;
    uint32 indexed_field_count;
}
```

Field descriptors:

```text
struct IndexedFieldDescriptor {
    uint16 field_type;
    uint16 flags;
    uint16 name_len;
    byte[name_len] name;
}
```

Principles:
- binary for engine-critical config
- version every structure
- optional opaque extension block allowed later

## 7. Record Storage

Each collection stores records in a collection-local B+tree keyed by record ID.

Record value:

```text
struct RecordValue {
    uint64 record_version;
    uint64 created_lsn;
    uint64 updated_lsn;
    uint32 flags;
    uint32 vector_dim;
    uint32 vector_inline_bytes;
    uint32 meta_inline_bytes;
    byte[vector_inline_bytes] vector_data;
    byte[meta_inline_bytes] metadata_data;
    uint64 vector_overflow_page;
    uint64 meta_overflow_page;
}
```

Metadata encoding:
- use `msgpack` or `cbor`
- deterministic encoding preferred
- preserve exact scalar types where possible

## 8. WAL Record Format

WAL is append-only and frame-based.

Frame header:

```text
struct WalFrameHeader {
    uint32 magic;
    uint16 version;
    uint16 record_type;
    uint64 lsn;
    uint64 txid;
    uint64 prev_lsn;
    uint32 payload_len;
    uint32 checksum;
}
```

Record types:
- `TX_BEGIN`
- `TX_COMMIT`
- `TX_ABORT`
- `COLLECTION_CREATE`
- `COLLECTION_DELETE`
- `COLLECTION_UPDATE_META`
- `RECORD_PUT`
- `RECORD_DELETE`
- `INDEX_SNAPSHOT_INSTALL`
- `CHECKPOINT_BEGIN`
- `CHECKPOINT_END`

Payload examples:

`COLLECTION_CREATE`

```text
collection_name
collection_id
encoded_collection_metadata
```

`RECORD_PUT`

```text
collection_id
record_id
record_version
vector bytes or blob ref
metadata bytes or blob ref
```

`RECORD_DELETE`

```text
collection_id
record_id
delete_version
```

Atomicity:
- durability boundary is `TX_COMMIT`
- uncommitted txids are ignored during recovery

## 9. Recovery Algorithm

Startup algorithm:
1. read file header
2. read metapages A and B
3. choose highest valid `meta_epoch`
4. load `root_catalog_page`, freelist root, `last_applied_lsn`
5. scan WAL from `last_applied_lsn + 1`
6. group frames by `txid`
7. replay only committed transactions in LSN order
8. rebuild in-memory handles from catalog
9. mark dirty state if WAL replay occurred
10. optionally trigger checkpoint if replay volume exceeds threshold

Rules:
- metapage is source of stable roots
- WAL is source of post-checkpoint truth
- collection discovery always comes from catalog after replay
- indexes can be replay-maintained for small updates, or installed from `INDEX_SNAPSHOT_INSTALL` and lazily rebuilt if stale

## 10. Deletion Semantics

Record deletion:
- logical delete first
- WAL appends `RECORD_DELETE`
- record removed from primary ID tree on checkpointed apply
- secondary metadata indexes updated transactionally
- vector index entry removed or tombstoned
- live count decremented

Collection deletion:
- WAL appends `COLLECTION_DELETE`
- catalog state changes to deleted at commit
- collection disappears from `ListCollections()` immediately after replay/open
- physical pages reclaimed only after compaction/checkpoint proves no active root references remain

Guarantee:
- delete is durable at commit
- reclaim is asynchronous

## 11. Compaction Strategy

Use checkpoint + page reclamation, not ad hoc file rewrite on every delete.

Trigger conditions:
- WAL bytes exceed threshold
- tombstone ratio exceeds threshold
- free-page fragmentation exceeds threshold
- explicit optimize call

Compaction steps:
1. freeze new checkpoint epoch
2. copy live catalog/data/index pages reachable from current logical roots into new clean pages
3. build new freelist from unreachable pages
4. write `CHECKPOINT_BEGIN`
5. flush new pages
6. write new metapage with updated roots and `last_applied_lsn`
7. write `CHECKPOINT_END`
8. advance WAL tail / truncate reclaimable WAL region

This is copy-on-write checkpointing:
- readers always have a stable root
- crash before metapage switch leaves old root valid
- crash after metapage switch leaves new root valid

## 12. Concurrency Model

Single-process embedded DB, multi-goroutine safe.

Locks:
- database-level WAL append mutex
- catalog RW lock
- per-collection tree/index locks
- checkpoint mutex

Write model:
- writers serialize at WAL commit boundary
- within a transaction, collection-local mutation work can be parallelized
- root publication is atomic via metapage switch

Read model:
- readers use snapshot from active metapage + committed in-memory overlay
- readers never block on checkpoint page copying
- readers may briefly coordinate with writer for in-memory index visibility

Recommended transaction model:
- one writer transaction committing at a time
- many concurrent readers
- optional batched writer group commit for throughput

This is simpler and safer than trying to support true multi-writer page mutation immediately.

## 13. Index Persistence

Do not treat HNSW/IVFPQ as purely in-memory adjuncts.

Persist index state as internal blobs:
- collection catalog points to `root_vector_index_blob`
- index snapshot install is transactional
- small mutations may be logged incrementally
- large rebuilds write fresh snapshot blobs and atomically swap pointer

Rule:
- record tree is canonical truth
- vector index is accelerative but must be recoverable
- if index blob is corrupt or stale, rebuild from canonical records

## 14. Portability Guarantees

A `.libravdb` file should be:
- self-contained
- endian-fixed
- versioned
- checksummed
- movable by copy/clone as one file

No sidecars required for correctness.

## 15. Minimal First Implementation

If you want the shortest path to a correct first single-file engine, build in this order:
1. file header + metapages
2. WAL with committed tx replay
3. catalog B+tree
4. collection record B+tree
5. collection discovery from catalog
6. record insert/delete/update
7. metadata scan/query
8. checkpoint + metapage switching
9. free-page reclamation
10. vector index snapshot persistence

That gets correctness and portability first, then performance refinement.
