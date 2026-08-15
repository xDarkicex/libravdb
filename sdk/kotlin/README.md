# LibraVDB Kotlin Native SDK

The official Kotlin SDK provides a high-performance, idiomatic Kotlin wrapper around the native LibraVDB CGO engine. Unlike the Java SDK (which runs on the JVM), this SDK utilizes **Kotlin Multiplatform (Native)** and `cinterop` to compile directly to machine code, bypassing JNI entirely.

It guarantees 100% API parity with the Python, Node, Ruby, Rust, C++, Java, and Dart SDKs.

## Unified SQL Engine

This SDK natively integrates with the LibraVDB Unified SQL Engine. You can execute expressive queries seamlessly across multiple paradigms:
- **Relational SQL**: Standard ANSI SQL data manipulation.
- **Vector SQL**: Order by `VECTOR_DISTANCE` and perform similarity matching.
- **Graph SQL**: Perform Cypher-like graph traversals using `JOIN MATCH (src)-[:EDGE]->(tgt)`.
- **Temporal SQL**: Query historical database snapshots using `AS OF TIMESTAMP`.

## Installation

This SDK is built using Gradle and the Kotlin Multiplatform plugin.

1. Ensure the `libravdb` native shared library is built in `sdk/cgo`.
2. Add this project as a dependency or publish it to your local Maven repository.

## Quick Start

```kotlin
import io.libravdb.*

fun main() {
    // 1. Initialize the Database
    val db = LibraVDB("./my_database")

    try {
        // 2. Create Collection (dimension 3)
        val col = db.createCollection("docs", 3)

        // 3. Insert Vector with Metadata
        val meta = buildJsonObject {
            put("category", "ai")
            put("active", true)
        }
        col.insert("doc1", floatArrayOf(1.0f, 2.0f, 3.0f), metadata = meta)

        // 4. Query with JSON AST Filters
        val filter = Filter.eq("category", "ai")
        val results = col.search(floatArrayOf(1.0f, 2.0f, 3.0f), 10, filter = filter)

        println("Results: $results")
    } finally {
        // 5. Close the database
        db.close()
    }
}
```

## Kotlin Native Memory Management (`memScoped`)
Because Kotlin Native compiles to C-like memory architectures, all string buffers and vector batch arrays passed to the CGO engine are managed via Kotlin's `memScoped` arena.
- **Batching**: Vectors (`List<FloatArray>`) are dynamically flattened into a single, contiguous C-array using `allocArray<FloatVar>()`. This guarantees `O(1)` memory continuity over the FFI boundary.
- **Auto-Cleanup**: Once the CGO engine executes the batch, the `memScoped` arena instantly cleans up the allocations, ensuring zero memory leaks and deterministic execution without relying on a Garbage Collector.
