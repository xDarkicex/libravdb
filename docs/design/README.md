# LibraVDB Design Documents

This directory contains detailed technical design documents for LibraVDB's
internal architecture, algorithms, and subsystems.

## Documents

| Document | Description |
|----------|-------------|
| [architecture.md](architecture.md) | Overall system architecture, component map, data flow, memory layout |
| [api.md](api.md) | Public API design principles, patterns, and evolution strategy |
| [transactions.md](transactions.md) | Transaction system: commit protocol, CAS, hooks, recovery |
| [concurrency.md](concurrency.md) | Concurrency model: locking hierarchy, worker pools, backpressure |
| [storage.md](storage.md) | LSM-tree storage layer: MemTable, SSTables, compaction, caching |
| [single-file-storage-spec.md](single-file-storage-spec.md) | Binary format specification for the `*.libravdb` file |
| [hnsw.md](hnsw.md) | HNSW algorithm design, parameters, and performance characteristics |
| [vector-store.md](vector-store.md) | VectorStore abstraction: slabby integration, memory mapping plan |
| [batch-operations.md](batch-operations.md) | Batch operations: chunking, concurrency, retries, rollback |

## Related Documents

- [API Reference](../api-reference.md) — Complete public API documentation
- [Getting Started](../getting-started.md) — Quickstart guide
- [Configuration Guide](../configuration/configuration.md) — All configuration options
- [Performance Tuning](../configuration/performance-tuning.md) — Optimization strategies
- [Advanced Error Handling](../errors/advanced_error_handling.md) — Error recovery system
- [Collections](../concepts/collections.md) — Collection lifecycle and management
- [Indexing](../concepts/indexing.md) — Index algorithm selection and tuning
- [Segment Format v1](../schema/segment_v1.md) — SSTable binary format
