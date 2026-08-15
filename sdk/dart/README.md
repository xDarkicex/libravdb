# LibraVDB Dart SDK

The official Dart SDK provides a high-performance, idiomatic Dart wrapper around the native LibraVDB CGO engine. Leveraging **`dart:ffi`**, it delivers zero-allocation vector batching and seamless native bindings for Dart servers and Flutter applications.

It guarantees 100% API parity with the Python, Node, Ruby, Rust, C++, and Java SDKs.

## Unified SQL Engine

This SDK natively integrates with the LibraVDB Unified SQL Engine. You can execute expressive queries seamlessly across multiple paradigms:
- **Relational SQL**: Standard ANSI SQL data manipulation.
- **Vector SQL**: Order by `VECTOR_DISTANCE` and perform similarity matching.
- **Graph SQL**: Perform Cypher-like graph traversals using `JOIN MATCH (src)-[:EDGE]->(tgt)`.
- **Temporal SQL**: Query historical database snapshots using `AS OF TIMESTAMP`.

## Installation

Add the following to your `pubspec.yaml`:

```yaml
dependencies:
  libravdb:
    path: path/to/libravdb/sdk/dart
```

Ensure the `libravdb` native shared library is built and available on your system's library path (`LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`), or alongside your executable.

## Quick Start

```dart
import 'package:libravdb/libravdb.dart';

void main() {
  // 1. Initialize the Database
  final db = LibraVDB('./my_database');

  try {
    // 2. Create Collection (dimension 3)
    final col = db.createCollection('docs', 3);

    // 3. Insert Vector with Metadata
    col.insert('doc1', [1.0, 2.0, 3.0], metadata: {'category': 'ai', 'active': true});

    // 4. Query with JSON AST Filters
    final filter = Filter.eq('category', 'ai');
    final results = col.search([1.0, 2.0, 3.0], 10, filter: filter);

    print('Results: $results');
  } finally {
    // 5. Close the database
    db.close();
  }
}
```

## High-Performance Batching
When inserting thousands of vectors, the SDK automatically flattens `List<List<double>>` into a single, contiguous 1D block of native C-memory via `malloc`. This prevents garbage collection overhead and executes the batch across the FFI boundary in a single `O(1)` function call.

```dart
final ids = ['vec_1', 'vec_2'];
final vectors = [
  [0.1, 0.2, 0.3],
  [0.4, 0.5, 0.6]
];
final metadata = [
  {'type': 'text'},
  {'type': 'image'}
];

col.insertBatch(ids, vectors, metadata: metadata);
```

## Memory Safety Architecture
Because Dart's Garbage Collector operates entirely independently of C memory, we must carefully manage FFI memory mapping.
1. **Pointers from CGO**: Any string returned by the Go engine is captured as a `Pointer<Utf8>`. It is safely parsed into a Dart `String`, and the exact pointer is instantly sent back to the Go engine via `FreeString(ptr)` to prevent leaks.
2. **Batch Allocations**: The `insertBatch` function temporarily uses Dart's `malloc` arena to structure the contiguous arrays for Go. After the FFI call returns, all temporary `malloc` allocations are immediately `calloc.free()`'d, ensuring the Dart GC and C-heap remain perfectly stable even under massive loads.
