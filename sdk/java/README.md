# LibraVDB Java SDK

The official Java SDK provides a high-performance, pure Java wrapper around the native LibraVDB CGO engine. Leveraging **JNA (Java Native Access)** and **Jackson**, it delivers zero-allocation vector batching and seamless JSON serialization without requiring complex C/C++ JNI builds.

It guarantees 100% API parity with the Python, Node, Ruby, Rust, and C++ SDKs.

## Installation

This SDK is built with Maven. The native `libravdb` shared library must be available in your system's library path (`LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH`), or explicitly provided to JNA via `-Djna.library.path=/path/to/libravdb/sdk/cgo`.

Add the following to your `pom.xml`:
```xml
<dependency>
    <groupId>io.libravdb</groupId>
    <artifactId>libravdb-java</artifactId>
    <version>1.0.0</version>
</dependency>
```

## Quick Start

```java
import io.libravdb.LibraVDB;
import io.libravdb.Collection;
import io.libravdb.Filter;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.Optional;

public class Main {
    public static void main(String[] args) throws Exception {
        ObjectMapper mapper = new ObjectMapper();

        // 1. Initialize the Database (AutoCloseable)
        try (LibraVDB db = new LibraVDB("./my_database")) {
            
            // 2. Create Collection (dimension 3)
            Collection col = db.createCollection("docs", 3);

            // 3. Insert Vector with Metadata
            JsonNode metadata = mapper.readTree("{\"category\":\"ai\", \"active\":true}");
            col.insert("doc1", new float[]{1.0f, 2.0f, 3.0f}, Optional.of(metadata));

            // 4. Query with JSON AST Filters
            Filter filter = Filter.eq("category", "ai");
            JsonNode results = col.search(new float[]{1.0f, 2.0f, 3.0f}, 10, Optional.of(filter));

            System.out.println("Results: " + results.toPrettyString());
        }
    }
}
```

## High-Performance Batching
When inserting thousands of vectors, the SDK automatically flattens `List<float[]>` into a single, contiguous 1D array, passing it to the C-bridge exactly once per batch to prevent FFI allocation overhead.

```java
List<String> ids = List.of("vec_1", "vec_2");
List<float[]> vectors = List.of(
    new float[]{0.1f, 0.2f, 0.3f},
    new float[]{0.4f, 0.5f, 0.6f}
);
List<JsonNode> metadata = List.of(
    mapper.readTree("{\"type\":\"text\"}"),
    mapper.readTree("{\"type\":\"image\"}")
);

col.insertBatch(ids, vectors, Optional.of(metadata));
```

## Memory Safety
The Java SDK completely hides all C-pointers from the developer. Internally, the JNA `LibraVDBLibrary` strictly maps all returned C-strings to `Pointer` instances, safely extracts the Java `String`, and immediately calls `FreeString(ptr)` on the native Go engine to prevent any memory leaks across the boundary.
