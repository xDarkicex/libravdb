# LibraVDB Odin SDK

A native, high-performance Odin wrapper for LibraVDB, built with zero-overhead `foreign` C interoperability.

## Unified SQL Engine

This SDK natively integrates with the LibraVDB Unified SQL Engine. You can execute expressive queries seamlessly across multiple paradigms:
- **Relational SQL**: Standard ANSI SQL data manipulation.
- **Vector SQL**: Order by `VECTOR_DISTANCE` and perform similarity matching.
- **Graph SQL**: Perform Cypher-like graph traversals using `JOIN MATCH (src)-[:EDGE]->(tgt)`.
- **Temporal SQL**: Query historical database snapshots using `AS OF TIMESTAMP`.

## Installation

Add the `libravdb` package to your project. Since LibraVDB uses a Go C-shared library, ensure `libravdb.dylib` (macOS), `libravdb.so` (Linux), or `libravdb.dll` (Windows) is available in your dynamic linker path or next to your executable.

## Usage

```odin
package main

import "core:fmt"
import "libravdb"

main :: proc() {
    // 1. Initialize Database
    db, err := libravdb.open_db("./my_db")
    if err != .None {
        fmt.println("Failed to open DB")
        return
    }
    defer libravdb.close_db(&db)

    // 2. Create or Get a Collection
    col, col_err := libravdb.create_collection(&db, "items", 3)
    
    // 3. Insert Vectors
    vec1 := []f32{0.1, 0.2, 0.3}
    meta1 := `{"tag": "A"}`
    libravdb.insert(&col, "doc1", vec1, meta1)

    // 4. Batch Insertion (Zero-Copy)
    ids := []string{"doc2", "doc3"}
    vecs := [][]f32{
        {0.4, 0.5, 0.6},
        {0.7, 0.8, 0.9},
    }
    metas := []string{`{"tag": "B"}`, `{"tag": "C"}`}
    libravdb.insert_batch(&col, ids, vecs, metas)

    // 5. Query with AST Filtering
    filter := libravdb.eq_str("tag", "A")
    filter_json := libravdb.to_json(filter)
    
    results, search_err := libravdb.search(&col, vec1, 10, filter_json)
    fmt.println(results)
}
```

## Advanced Filtering

LibraVDB Odin SDK supports complex filter trees cleanly using nested unions and structs:

```odin
import "libravdb"

filter := libravdb.and_filters([]libravdb.Filter{
    libravdb.eq_str("status", "active"),
    libravdb.in_str("category", []string{"electronics", "home"}),
})

json_str := libravdb.to_json(filter)
```
