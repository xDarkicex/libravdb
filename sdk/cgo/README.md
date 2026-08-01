# LibraVDB C-Shared Bridge (CGO)

This directory contains the CGO (C-Go) bridge for LibraVDB. It compiles the native Go engine into a C-compatible shared library (`.so`, `.dylib`, or `.dll`), exposing the core vector database capabilities to any language that supports C Foreign Function Interfaces (FFI), such as Python, Node.js (via N-API or ffi-napi), Rust, and Java (via JNI/JNA).

## Architecture

To avoid the tremendous overhead of crossing the FFI boundary thousands of times for large operations, this bridge relies heavily on:
1. **1D Array Flattening**: Instead of passing arrays of structs (like `[]VectorEntry`), FFI bridges pack matrices into flat 1D `float*` arrays. This allows languages like Python to perform a single native memory copy rather than thousands of individual allocations.
2. **JSON Serialization**: Complex structures (like Graph Filters, Stats, and Health checks) are serialized into JSON strings before crossing the boundary, making them easy to parse natively in the host language.
3. **Handle-Based Memory Management**: The Go runtime holds references to the `Database` and `Collection` objects in a concurrent `map[int]interface{}`. It returns an integer "handle" to the host language. The host language passes this integer back to Go to execute operations, ensuring that the Go Garbage Collector doesn't prematurely clean up the database engine while Python/Node is using it.

## Building

To build the shared library for your platform, run the included build script:

```bash
./build.sh
```

This will produce `libravdb.so` (Linux), `libravdb.dylib` (macOS), or `libravdb.dll` (Windows), along with a `libravdb.h` C header file.

## Exported API

The bridge exports functions covering 100% of the public LibraVDB API, categorized into:
- **Database Lifecycle**: `OpenDB`, `CloseDB`, `Backup`, `Vacuum`, `DropDatabase`
- **Global Memory & Health**: `SetGlobalMemoryLimit`, `GetGlobalMemoryUsage`, `TriggerGlobalGC`, `Ping`, `GetDatabaseHealth`, `GetDatabaseStats`
- **Collection Management**: `CreateCollection`, `GetCollection`, `ListCollections`, `DeleteCollection`, `OptimizeCollection`
- **Vector CRUD Operations**: `InsertVector`, `UpsertVector`, `UpdateVector`, `DeleteVector`, `GetVector`
- **Batch Operations**: `InsertBatch`, `DeleteBatch`
- **Optimistic Concurrency Control (OCC)**: `UpdateVectorIfVersion`, `DeleteVectorIfVersion`
- **Query & Scan**: `QueryVector` (with JSON AST filtering), `ScanCollection`
- **Advanced Collection Storage**: `EnableMemoryMapping`, `DisableMemoryMapping`, `SaveIndex`, `LoadIndex`
