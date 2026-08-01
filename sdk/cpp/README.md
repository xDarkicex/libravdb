# LibraVDB C++ SDK

The official C++ SDK provides a modern, fast, and type-safe wrapper around the native LibraVDB CGO engine. Leveraging C++17 and `nlohmann::json`, it delivers zero-allocation vector batching and complete memory safety across the FFI boundary, achieving 100% API parity with the Python, Node, Ruby, and Rust SDKs.

## Installation

This SDK uses CMake. It automatically downloads the header-only `nlohmann/json` library during configuration.

1. Ensure the CGO shared library is built by running `./build.sh` in the `sdk/cgo/` directory.
2. In your C++ project, link against the `libravdb_cpp` library and `libravdb` shared library.

```cmake
add_subdirectory(path/to/libravdb/sdk/cpp libravdb)
target_link_libraries(my_app PRIVATE libravdb_cpp)
```

## Quick Start

```cpp
#include <iostream>
#include <libravdb.hpp>

using namespace libravdb;

int main() {
    try {
        // Open or create a local single-file database
        LibraVDB db("./my_database");

        // Create a collection with vector dimension 3
        Collection col = db.create_collection("docs", 3);

        // Insert a vector with JSON metadata
        col.insert("doc1", {1.0f, 2.0f, 3.0f}, json{{"category", "ai"}});

        // Query with AST Filters
        Filter filter = Filter::eq("category", "ai");
        json results = col.search({1.0f, 2.0f, 3.0f}, 10, filter);

        std::cout << "Results: " << results.dump(4) << std::endl;

    } catch (const LibraException& e) {
        std::cerr << "Database error: " << e.what() << std::endl;
    }
    return 0;
}
```

## High-Performance Batching
When inserting thousands of vectors, the SDK automatically flattens `std::vector<std::vector<float>>` into a single, contiguous 1D block of C++ memory, crossing the C-bridge exactly once per batch to prevent allocation overhead.

```cpp
std::vector<std::string> ids = {"vec_1", "vec_2"};
std::vector<std::vector<float>> vectors = {
    {0.1f, 0.2f, 0.3f},
    {0.4f, 0.5f, 0.6f}
};
std::vector<json> metadata = {
    {{"type", "text"}},
    {{"type", "image"}}
};

col.insert_batch(ids, vectors, metadata);
```

## Optimistic Concurrency Control (OCC)
In multi-threaded C++ architectures, prevent race conditions by specifying an expected version constraint.

```cpp
try {
    col.update_if_version("vec_1", {0.9f, 0.9f, 0.9f}, 1);
} catch (const LibraException& e) {
    std::cout << "Version conflict! Vector was modified by another thread." << std::endl;
}
```
