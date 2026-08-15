# LibraVDB Python SDK

The LibraVDB Python SDK provides a native, high-performance interface to the LibraVDB engine. It offers 100% API parity with the Go library, allowing you to build blazingly fast local agentic workflows, RAG pipelines, and vector applications entirely in Python.

## Unified SQL Engine

This SDK natively integrates with the LibraVDB Unified SQL Engine. You can execute expressive queries seamlessly across multiple paradigms:
- **Relational SQL**: Standard ANSI SQL data manipulation.
- **Vector SQL**: Order by `VECTOR_DISTANCE` and perform similarity matching.
- **Graph SQL**: Perform Cypher-like graph traversals using `JOIN MATCH (src)-[:EDGE]->(tgt)`.
- **Temporal SQL**: Query historical database snapshots using `AS OF TIMESTAMP`.

## Installation

*(Coming soon to PyPI)*

To build and use it locally:
1. Run `./build.sh` in the `sdk/cgo/` directory to compile the C-Shared library.
2. Add the `sdk/python` directory to your `PYTHONPATH`.

## Quick Start

```python
from libravdb import LibraVDB
from libravdb.filters import Filter

# Open or create a local single-file database
db = LibraVDB("./my_database")

# Create a collection with vector dimension 3
col = db.create_collection("docs", 3)

# Insert a vector
col.insert("doc1", [1.0, 2.0, 3.0], metadata={"category": "ai", "active": True})

# Query with Graph Filters
filter = Filter.eq("category", "ai") & Filter.eq("active", True)
results = col.search([1.0, 2.0, 3.0], k=10, filter=filter)

print(results)
# [{'id': 'doc1', 'metadata': {'category': 'ai', 'active': True}}]

db.close()
```

## Advanced Features & Explanations

The Python SDK is designed for maximum performance, overcoming traditional FFI (Foreign Function Interface) bottlenecks via flattening and JSON Serialization. 

### High-Performance Batching
When inserting thousands of vectors, Python handles the flattening of matrices under the hood. This ensures `insert_batch` executes with almost zero memory overhead and crosses the C-bridge exactly once.

```python
ids = ["vec_1", "vec_2"]
vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
metadata = [{"type": "text"}, {"type": "image"}]

col.insert_batch(ids, vectors, metadata)
```

### Optimistic Concurrency Control (OCC)
In multi-threaded or distributed systems, prevent race conditions by specifying an `expected_version`. If another thread has modified the vector since you last retrieved it, LibraVDB will reject the update.

```python
try:
    col.update_if_version("vec_1", [0.9, 0.9, 0.9], expected_version=1)
except RuntimeError as e:
    print("Version conflict! Vector was modified by someone else.")
```

### Memory Management & Memory Mapping
LibraVDB runs completely off-heap. You can strictly constrain how much RAM the database or specific collections are allowed to consume.

```python
# Limit the global database to 1GB of RAM
db.set_memory_limit(1024 * 1024 * 1024)

# Or map a collection directly to disk for zero-copy reads (useful for datasets larger than RAM)
col.enable_memory_mapping("./my_database/docs_mmap.bin")
```

### Database Lifecycle
You can safely manage the database files via the SDK:
- `db.backup("/path/to/clone")` - Safely clones the graph and LSM tree.
- `db.vacuum()` - Reclaims unused disk space from deleted vectors.
- `db.drop()` - Completely destroys the database and cleans up files.
