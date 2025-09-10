# Getting Started with LibraVDB

This guide will help you get up and running with LibraVDB quickly.

## Installation

Add LibraVDB to your Go project:

```bash
go get github.com/xDarkicex/libravdb
```

## Basic Concepts

### Database
The `Database` is the top-level container that manages multiple collections and provides global configuration.

### Collection
A `Collection` is a named group of vectors with the same dimensionality and configuration. Each collection has its own index and storage.

### Vector Entry
A vector entry consists of:
- **ID**: Unique identifier (string)
- **Vector**: Float32 array of fixed dimension
- **Metadata**: Optional key-value pairs for filtering

## Your First Vector Database

### 1. Create a Database

```go
package main

import (
    "context"
    "log"
    
    "github.com/xDarkicex/libravdb/libravdb"
)

func main() {
    // Create database with custom storage path
    db, err := libravdb.New(
        libravdb.WithStoragePath("./my_vector_db"),
        libravdb.WithMetrics(true),
    )
    if err != nil {
        log.Fatal("Failed to create database:", err)
    }
    defer db.Close()
}
```

### 2. Create a Collection

```go
// Create a collection for 768-dimensional vectors (common for text embeddings)
collection, err := db.CreateCollection(
    context.Background(),
    "documents",
    libravdb.WithDimension(768),
    libravdb.WithMetric(libravdb.CosineDistance),
    libravdb.WithHNSW(32, 200, 50), // M=32, EfConstruction=200, EfSearch=50
)
if err != nil {
    log.Fatal("Failed to create collection:", err)
}
```

### 3. Insert Vectors

```go
// Insert a single vector
vector := make([]float32, 768)
// ... populate vector with your embedding data ...

metadata := map[string]interface{}{
    "title":    "My Document",
    "category": "research",
    "author":   "John Doe",
    "tags":     []string{"ai", "machine-learning"},
}

err = collection.Insert(context.Background(), "doc_1", vector, metadata)
if err != nil {
    log.Fatal("Failed to insert vector:", err)
}
```

### 4. Search for Similar Vectors

```go
// Search for the 10 most similar vectors
queryVector := make([]float32, 768)
// ... populate with your query embedding ...

results, err := collection.Search(context.Background(), queryVector, 10)
if err != nil {
    log.Fatal("Search failed:", err)
}

fmt.Printf("Found %d results in %v\n", len(results.Results), results.Took)
for i, result := range results.Results {
    fmt.Printf("%d. ID: %s, Score: %.3f, Title: %s\n", 
        i+1, result.ID, result.Score, result.Metadata["title"])
}
```

## Distance Metrics

Choose the appropriate distance metric for your use case:

```go
// Cosine Distance (recommended for normalized embeddings)
libravdb.WithMetric(libravdb.CosineDistance)

// L2 (Euclidean) Distance
libravdb.WithMetric(libravdb.L2Distance)

// Inner Product (for embeddings that aren't normalized)
libravdb.WithMetric(libravdb.InnerProduct)
```

## Index Types

LibraVDB supports multiple indexing algorithms:

### HNSW (Hierarchical Navigable Small World)
Best for most use cases - good balance of speed and accuracy:

```go
libravdb.WithHNSW(32, 200, 50)
// M: max connections per node (16-64 typical)
// EfConstruction: search width during construction (100-800)
// EfSearch: search width during queries (50-200)
```

### Flat Index
Exact search, good for small collections (<10K vectors):

```go
libravdb.WithFlat()
```

### Auto Selection
Let LibraVDB choose the best index based on collection size:

```go
libravdb.WithAutoIndexSelection(true)
```

## Error Handling

LibraVDB provides detailed error information:

```go
if err != nil {
    if libravdbErr, ok := err.(*libravdb.Error); ok {
        fmt.Printf("Error Code: %s\n", libravdbErr.Code)
        fmt.Printf("Message: %s\n", libravdbErr.Message)
        fmt.Printf("Component: %s\n", libravdbErr.Component)
    }
    return err
}
```

## Complete Example

Here's a complete working example:

```go
package main

import (
    "context"
    "fmt"
    "log"
    "math/rand"
    
    "github.com/xDarkicex/libravdb/libravdb"
)

func main() {
    // Create database
    db, err := libravdb.New(libravdb.WithStoragePath("./example_db"))
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()

    // Create collection
    collection, err := db.CreateCollection(
        context.Background(),
        "example_vectors",
        libravdb.WithDimension(128),
        libravdb.WithMetric(libravdb.CosineDistance),
    )
    if err != nil {
        log.Fatal(err)
    }

    // Insert some example vectors
    for i := 0; i < 1000; i++ {
        vector := make([]float32, 128)
        for j := range vector {
            vector[j] = rand.Float32()
        }
        
        metadata := map[string]interface{}{
            "index":    i,
            "category": fmt.Sprintf("cat_%d", i%5),
        }
        
        err := collection.Insert(context.Background(), fmt.Sprintf("vec_%d", i), vector, metadata)
        if err != nil {
            log.Printf("Failed to insert vector %d: %v", i, err)
        }
    }

    // Search for similar vectors
    queryVector := make([]float32, 128)
    for i := range queryVector {
        queryVector[i] = rand.Float32()
    }

    results, err := collection.Search(context.Background(), queryVector, 5)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Search completed in %v\n", results.Took)
    for i, result := range results.Results {
        fmt.Printf("%d. ID: %s, Score: %.3f, Category: %s\n",
            i+1, result.ID, result.Score, result.Metadata["category"])
    }

    // Get collection statistics
    stats := collection.Stats()
    fmt.Printf("\nCollection Stats:\n")
    fmt.Printf("- Vectors: %d\n", stats.VectorCount)
    fmt.Printf("- Memory Usage: %d bytes\n", stats.MemoryUsage)
    fmt.Printf("- Index Type: %s\n", stats.IndexType)
}
```

## Next Steps

- Learn about [Advanced Filtering](concepts/filtering.md)
- Explore [Performance Tuning](performance-tuning.md)
- Check out [Configuration Options](configuration.md)
- See [Real-world Examples](examples/)