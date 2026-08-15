# LibraVDB TypeScript SDK

The LibraVDB TypeScript SDK provides a native, high-performance Node.js interface to the LibraVDB engine using [Koffi](https://koffi.dev/) for blazing-fast C Foreign Function Interface (FFI) bindings. It achieves 100% API parity with the Go library and Python SDK.

## Unified SQL Engine

This SDK natively integrates with the LibraVDB Unified SQL Engine. You can execute expressive queries seamlessly across multiple paradigms:
- **Relational SQL**: Standard ANSI SQL data manipulation.
- **Vector SQL**: Order by `VECTOR_DISTANCE` and perform similarity matching.
- **Graph SQL**: Perform Cypher-like graph traversals using `JOIN MATCH (src)-[:EDGE]->(tgt)`.
- **Temporal SQL**: Query historical database snapshots using `AS OF TIMESTAMP`.

## Installation

*(Coming soon to NPM)*

To build and use it locally:
1. Run `./build.sh` in the `sdk/cgo/` directory to compile the C-Shared library.
2. Run `npm install` inside the `sdk/typescript` folder.
3. Build the TypeScript package: `npm run build`.

## Quick Start

```typescript
import { LibraVDB, Filter } from 'libravdb-ts';

// Open or create a local single-file database
const db = new LibraVDB("./my_database");

// Create a collection with vector dimension 3
const col = db.createCollection("docs", 3);

// Insert a vector
col.insert("doc1", [1.0, 2.0, 3.0], { category: "ai", active: true });

// Query with Graph Filters
const filter = Filter.and(Filter.eq("category", "ai"), Filter.eq("active", true));
const results = col.search([1.0, 2.0, 3.0], 10, filter);

console.log(results);
// [{ id: 'doc1', metadata: { category: 'ai', active: true } }]

db.close();
```

## High-Performance Batching
When inserting thousands of vectors, Koffi allows us to map a massive 1D `Float32Array` directly to the C pointer. The SDK handles this flattening automatically for you, crossing the C-bridge exactly once per batch.

```typescript
const ids = ["vec_1", "vec_2"];
const vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
const metadata = [{ type: "text" }, { type: "image" }];

col.insertBatch(ids, vectors, metadata);
```

## Optimistic Concurrency Control (OCC)
For multi-threaded or distributed environments, prevent race conditions by specifying an `expectedVersion`.

```typescript
try {
    col.updateIfVersion("vec_1", [0.9, 0.9, 0.9], 1);
} catch (e) {
    console.error("Version conflict! Vector was modified by someone else.");
}
```
