# LibraVDB C# / .NET SDK

A high-performance C#/.NET native wrapper for LibraVDB, providing zero-allocation vector batching using `unsafe` blocks and zero-overhead CGO P/Invoke bindings.

## Unified SQL Engine

This SDK natively integrates with the LibraVDB Unified SQL Engine. You can execute expressive queries seamlessly across multiple paradigms:
- **Relational SQL**: Standard ANSI SQL data manipulation.
- **Vector SQL**: Order by `VECTOR_DISTANCE` and perform similarity matching.
- **Graph SQL**: Perform Cypher-like graph traversals using `JOIN MATCH (src)-[:EDGE]->(tgt)`.
- **Temporal SQL**: Query historical database snapshots using `AS OF TIMESTAMP`.

## Installation

Add the library to your project:
```bash
dotnet add reference path/to/LibraVDB.csproj
```

**Note:** Ensure `libravdb.dylib` / `libravdb.so` / `libravdb.dll` is in your runtime library path.

## Usage

```csharp
using System;
using LibraVDB;

class Program
{
    static void Main()
    {
        // 1. Initialize Database
        using var db = new Database("./my_db");
        
        // 2. Create or Get a Collection
        var col = db.CreateCollection("items", 3);
        
        // 3. Insert Vectors
        col.Insert("doc1", new float[] { 0.1f, 0.2f, 0.3f }, new { tag = "A" });
        
        // 4. Batch Insertion (Zero-Allocation via Unsafe Pointers)
        string[] ids = { "doc2", "doc3" };
        float[][] vectors = {
            new float[] { 0.4f, 0.5f, 0.6f },
            new float[] { 0.7f, 0.8f, 0.9f }
        };
        object[] metas = { new { tag = "B" }, new { tag = "C" } };
        col.InsertBatch(ids, vectors, metas);
        
        // 5. Query with AST Filtering
        var filter = Filter.Eq("tag", "A");
        string results = col.Search(new float[] { 0.1f, 0.2f, 0.3f }, k: 10, filter);
        Console.WriteLine(results);
    }
}
```

## Advanced Filtering

LibraVDB C# SDK supports complex filter trees:
```csharp
var filter = Filter.And(
    Filter.Eq("status", "active"),
    Filter.In("category", new[] { "electronics", "home" })
);
```
