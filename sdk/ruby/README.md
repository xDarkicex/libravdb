# LibraVDB Ruby SDK

The LibraVDB Ruby SDK provides a native, high-performance interface to the LibraVDB engine using the `ffi` gem. It achieves 100% API parity with the Go library, Python SDK, and TypeScript SDK.

## Unified SQL Engine

This SDK natively integrates with the LibraVDB Unified SQL Engine. You can execute expressive queries seamlessly across multiple paradigms:
- **Relational SQL**: Standard ANSI SQL data manipulation.
- **Vector SQL**: Order by `VECTOR_DISTANCE` and perform similarity matching.
- **Graph SQL**: Perform Cypher-like graph traversals using `JOIN MATCH (src)-[:EDGE]->(tgt)`.
- **Temporal SQL**: Query historical database snapshots using `AS OF TIMESTAMP`.

## Installation

*(Coming soon to RubyGems)*

To build and use it locally:
1. Run `./build.sh` in the `sdk/cgo/` directory to compile the C-Shared library.
2. Add this gem to your `Gemfile`:
   ```ruby
   gem 'libravdb', path: 'path/to/libravdb/sdk/ruby'
   ```
3. Run `bundle install`.

## Quick Start

```ruby
require 'libravdb'

# Open or create a local single-file database
db = LibraVDB::Client.new("./my_database")

# Create a collection with vector dimension 3
col = db.create_collection("docs", 3)

# Insert a vector
col.insert("doc1", [1.0, 2.0, 3.0], { category: "ai", active: true })

# Query with Graph Filters
filter = LibraVDB::Filter.and(
  LibraVDB::Filter.eq("category", "ai"), 
  LibraVDB::Filter.eq("active", true)
)
results = col.search([1.0, 2.0, 3.0], 10, filter)

puts results
# [{"id"=>"doc1", "metadata"=>{"category"=>"ai", "active"=>true}}]

db.close
```

## High-Performance Batching
When inserting thousands of vectors, the SDK automatically flattens your Ruby arrays and writes them directly into an `FFI::MemoryPointer` block of C memory. This means the C-bridge is only crossed exactly once per batch, eliminating standard Ruby object allocation overhead.

```ruby
ids = ["vec_1", "vec_2"]
vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
metadata = [{ type: "text" }, { type: "image" }]

col.insert_batch(ids, vectors, metadata)
```

## Optimistic Concurrency Control (OCC)
In multi-threaded architectures (like Puma or Sidekiq), prevent race conditions by specifying an `expected_version`.

```ruby
begin
  col.update_if_version("vec_1", [0.9, 0.9, 0.9], 1)
rescue => e
  puts "Version conflict! Vector was modified by someone else."
end
```
