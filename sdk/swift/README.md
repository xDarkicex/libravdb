# LibraVDB Swift SDK

A seamless native Swift wrapper for LibraVDB, leveraging zero-overhead C-interoperability and SPM (Swift Package Manager).

## Unified SQL Engine

This SDK natively integrates with the LibraVDB Unified SQL Engine. You can execute expressive queries seamlessly across multiple paradigms:
- **Relational SQL**: Standard ANSI SQL data manipulation.
- **Vector SQL**: Order by `VECTOR_DISTANCE` and perform similarity matching.
- **Graph SQL**: Perform Cypher-like graph traversals using `JOIN MATCH (src)-[:EDGE]->(tgt)`.
- **Temporal SQL**: Query historical database snapshots using `AS OF TIMESTAMP`.

## Installation

Add the library to your `Package.swift` dependencies:

```swift
dependencies: [
    .package(url: "https://github.com/xDarkicex/libravdb.git", from: "0.1.0")
]
```

**Note:** Since LibraVDB uses a Go C-shared library, you must provide `libravdb.dylib` / `libravdb.so` in your dynamic linker path or package directory.

## Usage

```swift
import LibraVDB

// 1. Initialize Database
let db = try Database(path: "./my_db")

// 2. Create or Get a Collection
let col = try db.createCollection(name: "items", dimension: 3)

// 3. Insert Vectors
try col.insert(id: "doc1", vector: [0.1, 0.2, 0.3], metadata: ["tag": "A"])

// 4. Batch Insertion (Zero-Copy)
try col.insertBatch(
    ids: ["doc2", "doc3"],
    vectors: [
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ],
    metadata: [
        ["tag": "B"],
        ["tag": "C"]
    ]
)

// 5. Query with AST Filtering
let filter = Filter.eq("tag", "A")
let results = try col.search(vector: [0.1, 0.2, 0.3], k: 10, filter: filter)
print(results)
```

## Advanced Filtering

LibraVDB Swift SDK supports complex filter trees cleanly using Swift Enums/Structs:

```swift
let filter = Filter.and(
    Filter.eq("status", "active"),
    Filter.in("category", ["electronics", "home"])
)
```
