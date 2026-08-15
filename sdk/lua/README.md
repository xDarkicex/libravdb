# LibraVDB Lua SDK

The official Lua SDK for LibraVDB, built entirely on top of **LuaJIT FFI**.

This SDK provides a highly performant, object-oriented API that binds directly to the underlying LibraVDB C-Shared library (`libravdb.dylib` / `libravdb.so`) without any C middleware or compilation steps required.

## Features
- **Zero-compilation CGO FFI**: Leverages `ffi.cdef[[...]]` to map the Go engine natively into Lua using LuaJIT.
- **Embedded JSON**: Includes a lightweight, pure-Lua JSON encoder/decoder (`json.lua`), allowing you to pass native Lua tables for vectors and metadata matching the ergonomics of the Python/Node SDKs.
- **Unified SQL Engine**: Full support for Relational, Vector, Graph, and Temporal unified SQL queries.
- **Object-Oriented**: Cleanly namespaced `LibraVDB` and `Collection` classes using Lua metatables (`__index`).

## Installation

Ensure you have **LuaJIT** installed (this SDK requires LuaJIT's `ffi` library):
```bash
luajit -v
```

Include `libravdb.lua` and `json.lua` in your project path, and ensure the LibraVDB shared library (`../cgo/libravdb.dylib` or `.so`) is accessible.

## Quickstart

```lua
local libra = require("libravdb")
local LibraVDB = libra.LibraVDB

-- Open the database
local db = LibraVDB.new("path/to/database")

-- Create a collection
local collection = db:create_collection("my_collection", 3)

-- Insert vectors (tables are automatically serialized to JSON)
collection:insert("vec1", {1.0, 0.5, 0.0}, {category = "A"})

-- Unified SQL Query
local result = db:query("SELECT id FROM my_collection ORDER BY VECTOR_DISTANCE(embedding, '{\"vec\":[1.0,0.5,0.0]}') ASC LIMIT 1")

-- Clean up
db:close()
```

## Testing
Run the comprehensive integration test suite to verify Relational, Vector, Graph, and Temporal functionality:

```bash
luajit test_libravdb.lua
```
