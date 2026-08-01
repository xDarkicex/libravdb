# LibraVDB Swift SDK

A seamless native Swift wrapper for LibraVDB, leveraging zero-overhead C-interoperability and SPM (Swift Package Manager).

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
