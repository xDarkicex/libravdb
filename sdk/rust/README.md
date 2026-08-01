# LibraVDB Rust SDK

The official Rust SDK for LibraVDB provides an extremely fast, zero-allocation interface to the core Go engine via a `bindgen` generated CGO bridge. It achieves 100% API parity with the Python, Node, and Ruby SDKs, specifically designed for high-performance agent harnesses and AI sidecars.

## Architecture

This SDK utilizes `bindgen` to generate unsafe Rust bindings against the `libravdb` CGO library. These raw C-bindings are wrapped in a 100% safe, idiomatic Rust API (`LibraVDB` and `Collection`), returning `Result<T, LibraError>` types and managing memory allocations and freeing across the FFI boundary automatically.

## Installation

*(Coming soon to Crates.io)*

To build and use locally:
1. Compile the CGO library by running `./build.sh` inside `sdk/cgo/`.
2. Add this package to your `Cargo.toml`:
   ```toml
   [dependencies]
   libravdb = { path = "path/to/sdk/rust" }
   ```

## Quick Start

```rust
use libravdb::{LibraVDB, Filter};
use serde_json::json;

fn main() -> Result<(), libravdb::LibraError> {
    // Open or create a database
    let db = LibraVDB::new("./my_database")?;

    // Create a collection
    let col = db.create_collection("docs", 3)?;

    // Insert a vector
    col.insert("doc1", &[1.0, 2.0, 3.0], Some(json!({"category": "ai"})))?;

    // Search with AST Filters
    let filter = Filter::eq("category", "ai");
    let results = col.search(&[1.0, 2.0, 3.0], 10, Some(&filter))?;
    
    println!("Results: {:?}", results);

    Ok(())
}
```

## High-Performance Batching
Instead of crossing the FFI boundary thousands of times, the Rust SDK automatically flattens `Vec<Vec<f32>>` into a single, contiguous 1D array of C memory in a single allocation block.

```rust
let ids = vec!["vec1".to_string(), "vec2".to_string()];
let vectors = vec![
    vec![0.1, 0.2, 0.3],
    vec![0.4, 0.5, 0.6]
];
let meta = vec![json!({"id": 1}), json!({"id": 2})];

col.insert_batch(&ids, &vectors, Some(&meta))?;
```

## Optimistic Concurrency Control (OCC)
In high-throughput Rust architectures (e.g. Tokio/Actix setups), prevent race conditions on vector mutations by specifying an expected version constraint.

```rust
if let Err(e) = col.update_if_version("vec1", &[0.9, 0.9, 0.9], 1, None) {
    println!("Transaction rejected: {}", e.0);
}
```
