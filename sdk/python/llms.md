# LibraVDB Python SDK: LLM System Instructions

This document is intended to provide LLMs and agentic coders with the necessary context to generate, debug, and understand the LibraVDB Python SDK.

## Core Architecture

LibraVDB is a high-performance vector database written in Go. The Python SDK does NOT communicate via HTTP/gRPC. Instead, it is a direct FFI (Foreign Function Interface) wrapper around the compiled Go engine (`libravdb.so/dylib/dll`). 

**Important Architectural Principles to Remember:**
1. **The C-Shared Bridge:** The SDK operates by loading a C-Shared library compiled from Go using `ctypes`. The interface definitions live in `core.py`. The high-level Python API lives in `client.py`.
2. **Handle-Based Concurrency:** Go does not export struct pointers to Python because the Go Garbage Collector would panic. Instead, `client.py` holds an `int` handle (`_handle`). This integer maps to a `sync.RWMutex` protected map inside Go (`collections` and `databases`).
3. **1D Flattening for Speed:** To avoid `ctypes` overhead when passing 2D arrays (like in `insert_batch`), Python flattens all `List[List[float]]` into a single 1D `(ctypes.c_float * N)` array. The CGO bridge un-flattens it on the other side.
4. **JSON for Complex Types:** The CGO bridge parses and returns complex data (like filters, statistics, health checks, and metadata) entirely via JSON strings. `core.py` handles the conversion using `_to_c_string` and `_from_c_string`.

## API Surface

The Python SDK provides 100% parity with the Go engine. If you need to write scripts using the SDK, here are the key methods available:

### `LibraVDB(path: str)`
- `.create_collection(name, dim)`
- `.get_collection(name, dim)`
- `.list_collections()`
- `.delete_collection(name)`
- `.vacuum()`: Reclaims disk space.
- `.backup(dest_path)`: Copies the DB safely.
- `.drop()`: Destroys the DB.
- `.ping()`, `.health()`, `.stats()`
- `.set_memory_limit(bytes)`, `.memory_usage()`, `.trigger_gc()`

### `Collection`
- `.insert(id, vector, metadata)`, `.upsert()`, `.update()`, `.delete()`
- `.insert_batch(ids, vectors, metadata)`: High performance batching.
- `.delete_batch(ids)`
- `.get(id)` -> returns `{"id": "...", "metadata": {...}}`
- `.count()` -> returns total vectors
- `.update_if_version(id, vector, expected_version, metadata)`: OCC updating.
- `.delete_if_version(id, expected_version)`: OCC deletion.
- `.search(vector, k, filter)`: Vector similarity search.
- `.scan(offset, limit)`: Paginate without similarity search.
- `.save_index(path)`, `.load_index(path)`
- `.enable_memory_mapping(path)`, `.disable_memory_mapping()`

### `Filter` (AST Builder)
Used to construct queries safely to be passed to `.search()`.
```python
from libravdb.filters import Filter
f = Filter.and_(Filter.eq("key", "value"), Filter.gt("age", 18))
```

## Debugging Guide

If you encounter an error passing data to the Go backend, double check `core.py`.
- Ensure all `c_char_p` returns are explicitly freed using `_lib.FreeString()` within `_from_c_string`, otherwise memory leaks will occur.
- Ensure that memory mappings and batch inserts use precisely the expected dimensions. A mismatch in the flattened 1D array size relative to the collection's defined `dim` will cause a segfault or index out of bounds panic in Go.
