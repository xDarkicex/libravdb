# Basic Usage Examples

This document provides practical examples of common LibraVDB usage patterns.

## Simple Vector Search

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
    db, err := libravdb.New(libravdb.WithStoragePath("./simple_example"))
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()

    // Create collection
    collection, err := db.CreateCollection(
        context.Background(),
        "simple_vectors",
        libravdb.WithDimension(128),
        libravdb.WithMetric(libravdb.CosineDistance),
    )
    if err != nil {
        log.Fatal(err)
    }

    // Insert some vectors
    for i := 0; i < 100; i++ {
        vector := make([]float32, 128)
        for j := range vector {
            vector[j] = rand.Float32()
        }
        
        err := collection.Insert(context.Background(), fmt.Sprintf("vec_%d", i), vector, nil)
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

    fmt.Printf("Found %d similar vectors:\n", len(results.Results))
    for i, result := range results.Results {
        fmt.Printf("%d. ID: %s, Score: %.3f\n", i+1, result.ID, result.Score)
    }
}
```

## Document Search with Metadata

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
    db, err := libravdb.New(libravdb.WithStoragePath("./document_search"))
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()

    // Create collection with metadata schema
    schema := libravdb.MetadataSchema{
        "title":    libravdb.StringField,
        "category": libravdb.StringField,
        "author":   libravdb.StringField,
        "tags":     libravdb.StringArrayField,
        "score":    libravdb.FloatField,
    }

    collection, err := db.CreateCollection(
        context.Background(),
        "documents",
        libravdb.WithDimension(768), // Common for text embeddings
        libravdb.WithMetric(libravdb.CosineDistance),
        libravdb.WithMetadataSchema(schema),
        libravdb.WithIndexedFields("category", "author"), // Index for fast filtering
    )
    if err != nil {
        log.Fatal(err)
    }

    // Insert documents with metadata
    documents := []struct {
        id       string
        title    string
        category string
        author   string
        tags     []string
        score    float64
    }{
        {"doc1", "Introduction to AI", "technology", "Alice", []string{"ai", "intro"}, 4.5},
        {"doc2", "Machine Learning Basics", "technology", "Bob", []string{"ml", "basics"}, 4.2},
        {"doc3", "Cooking with Python", "programming", "Charlie", []string{"python", "tutorial"}, 4.8},
        {"doc4", "Advanced Algorithms", "technology", "Alice", []string{"algorithms", "advanced"}, 4.7},
        {"doc5", "Web Development Guide", "programming", "Diana", []string{"web", "guide"}, 4.3},
    }

    for _, doc := range documents {
        // Generate random vector (in real use, this would be from an embedding model)
        vector := make([]float32, 768)
        for i := range vector {
            vector[i] = rand.Float32()
        }

        metadata := map[string]interface{}{
            "title":    doc.title,
            "category": doc.category,
            "author":   doc.author,
            "tags":     doc.tags,
            "score":    doc.score,
        }

        err := collection.Insert(context.Background(), doc.id, vector, metadata)
        if err != nil {
            log.Printf("Failed to insert document %s: %v", doc.id, err)
        }
    }

    // Search with filtering
    queryVector := make([]float32, 768)
    for i := range queryVector {
        queryVector[i] = rand.Float32()
    }

    fmt.Println("=== Search Examples ===\n")

    // 1. Simple search
    fmt.Println("1. Simple search (top 3):")
    results, err := collection.Search(context.Background(), queryVector, 3)
    if err != nil {
        log.Fatal(err)
    }
    printDocumentResults(results)

    // 2. Filter by category
    fmt.Println("2. Technology documents only:")
    results, err = collection.Query(context.Background()).
        WithVector(queryVector).
        Eq("category", "technology").
        Limit(5).
        Execute()
    if err != nil {
        log.Fatal(err)
    }
    printDocumentResults(results)

    // 3. Filter by author and score
    fmt.Println("3. Alice's documents with score > 4.5:")
    results, err = collection.Query(context.Background()).
        WithVector(queryVector).
        And().
            Eq("author", "Alice").
            Gt("score", 4.5).
        End().
        Limit(5).
        Execute()
    if err != nil {
        log.Fatal(err)
    }
    printDocumentResults(results)

    // 4. Complex filtering
    fmt.Println("4. Technology OR programming with high scores:")
    results, err = collection.Query(context.Background()).
        WithVector(queryVector).
        And().
            Or().
                Eq("category", "technology").
                Eq("category", "programming").
            End().
            Gte("score", 4.5).
        End().
        Limit(5).
        Execute()
    if err != nil {
        log.Fatal(err)
    }
    printDocumentResults(results)
}

func printDocumentResults(results *libravdb.SearchResults) {
    fmt.Printf("Found %d results (took %v):\n", len(results.Results), results.Took)
    for i, result := range results.Results {
        fmt.Printf("  %d. %s - %s by %s (score: %.1f, similarity: %.3f)\n",
            i+1,
            result.Metadata["title"],
            result.Metadata["category"],
            result.Metadata["author"],
            result.Metadata["score"],
            result.Score)
    }
    fmt.Println()
}
```

## High-Performance Batch Processing

```go
package main

import (
    "context"
    "fmt"
    "log"
    "math/rand"
    "runtime"
    "time"

    "github.com/xDarkicex/libravdb/libravdb"
)

func main() {
    db, err := libravdb.New(libravdb.WithStoragePath("./batch_example"))
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()

    // Create collection optimized for batch processing
    collection, err := db.CreateCollection(
        context.Background(),
        "batch_vectors",
        libravdb.WithDimension(256),
        libravdb.WithMetric(libravdb.CosineDistance),
        libravdb.WithBatchChunkSize(5000),              // Large chunks
        libravdb.WithBatchConcurrency(runtime.NumCPU()), // Use all CPUs
        libravdb.WithMemoryLimit(4*1024*1024*1024),     // 4GB limit
    )
    if err != nil {
        log.Fatal(err)
    }

    // Configure streaming options
    opts := &libravdb.StreamingOptions{
        BufferSize:     20000,
        ChunkSize:      5000,
        MaxConcurrency: runtime.NumCPU(),
        Timeout:        5 * time.Minute,
        ProgressCallback: func(stats *libravdb.StreamingStats) {
            fmt.Printf("Progress: %d/%d (%.1f%%), %.0f/sec\n",
                stats.TotalProcessed,
                stats.TotalReceived,
                float64(stats.TotalProcessed)/float64(stats.TotalReceived)*100,
                stats.ItemsPerSecond)
        },
    }

    // Create streaming batch insert
    stream := collection.NewStreamingBatchInsert(opts)
    err = stream.Start()
    if err != nil {
        log.Fatal(err)
    }
    defer stream.Close()

    // Generate and stream large dataset
    fmt.Println("Streaming 100,000 vectors...")
    start := time.Now()

    for i := 0; i < 100000; i++ {
        vector := make([]float32, 256)
        for j := range vector {
            vector[j] = rand.Float32()
        }

        entry := &libravdb.VectorEntry{
            ID:     fmt.Sprintf("batch_vec_%d", i),
            Vector: vector,
            Metadata: map[string]interface{}{
                "batch":     i / 10000,
                "timestamp": time.Now().Unix(),
                "category":  fmt.Sprintf("cat_%d", i%10),
            },
        }

        err := stream.Send(entry)
        if err != nil {
            log.Printf("Failed to send entry %d: %v", i, err)
        }

        // Print progress every 10,000 entries
        if (i+1)%10000 == 0 {
            fmt.Printf("Sent %d entries...\n", i+1)
        }
    }

    // Wait for processing to complete
    time.Sleep(2 * time.Second)

    // Get final statistics
    stats := stream.Stats()
    elapsed := time.Since(start)

    fmt.Printf("\n=== Batch Processing Results ===\n")
    fmt.Printf("Total time: %v\n", elapsed)
    fmt.Printf("Entries sent: %d\n", stats.TotalReceived)
    fmt.Printf("Entries processed: %d\n", stats.TotalProcessed)
    fmt.Printf("Successful: %d\n", stats.TotalSuccessful)
    fmt.Printf("Failed: %d\n", stats.TotalFailed)
    fmt.Printf("Average rate: %.0f entries/sec\n", float64(stats.TotalProcessed)/elapsed.Seconds())

    // Test search performance
    fmt.Printf("\n=== Search Performance Test ===\n")
    queryVector := make([]float32, 256)
    for i := range queryVector {
        queryVector[i] = rand.Float32()
    }

    searchStart := time.Now()
    results, err := collection.Search(context.Background(), queryVector, 10)
    searchTime := time.Since(searchStart)

    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Search time: %v\n", searchTime)
    fmt.Printf("Results found: %d\n", len(results.Results))
    fmt.Printf("Collection stats:\n")
    
    collectionStats := collection.Stats()
    fmt.Printf("  - Vectors: %d\n", collectionStats.VectorCount)
    fmt.Printf("  - Memory usage: %d MB\n", collectionStats.MemoryUsage/1024/1024)
    fmt.Printf("  - Index type: %s\n", collectionStats.IndexType)
}
```

## Memory-Optimized Large Collection

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
    db, err := libravdb.New(libravdb.WithStoragePath("./memory_optimized"))
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()

    // Create memory-optimized collection
    collection, err := db.CreateCollection(
        context.Background(),
        "large_collection",
        libravdb.WithDimension(768),
        libravdb.WithMetric(libravdb.CosineDistance),
        libravdb.WithAutoIndexSelection(true),           // Adapt index to size
        libravdb.WithMemoryLimit(2*1024*1024*1024),      // 2GB limit
        libravdb.WithMemoryMapping(true),                // Use memory mapping
        libravdb.WithProductQuantization(8, 8, 0.1),     // 8x compression
        libravdb.WithCachePolicy(libravdb.LRUCache),     // Efficient caching
    )
    if err != nil {
        log.Fatal(err)
    }

    // Insert vectors with memory monitoring
    fmt.Println("Inserting vectors with memory monitoring...")
    
    for i := 0; i < 50000; i++ {
        vector := make([]float32, 768)
        for j := range vector {
            vector[j] = rand.Float32()
        }

        metadata := map[string]interface{}{
            "index":    i,
            "category": fmt.Sprintf("cat_%d", i%20),
        }

        err := collection.Insert(context.Background(), fmt.Sprintf("vec_%d", i), vector, metadata)
        if err != nil {
            log.Printf("Failed to insert vector %d: %v", i, err)
        }

        // Monitor memory usage every 5000 insertions
        if (i+1)%5000 == 0 {
            stats := collection.Stats()
            if stats.MemoryStats != nil {
                usage := float64(stats.MemoryStats.Total) / float64(stats.MemoryStats.Limit) * 100
                fmt.Printf("Inserted %d vectors, memory usage: %.1f%% (%d MB / %d MB)\n",
                    i+1, usage,
                    stats.MemoryStats.Total/1024/1024,
                    stats.MemoryStats.Limit/1024/1024)
            }
        }
    }

    // Get detailed memory statistics
    fmt.Printf("\n=== Memory Usage Breakdown ===\n")
    usage, err := collection.GetMemoryUsage()
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Total memory: %d MB\n", usage.Total/1024/1024)
    fmt.Printf("Index memory: %d MB\n", usage.Indices/1024/1024)
    fmt.Printf("Cache memory: %d MB\n", usage.Caches/1024/1024)
    fmt.Printf("Quantized memory: %d MB\n", usage.Quantized/1024/1024)
    fmt.Printf("Memory mapped: %d MB\n", usage.MemoryMapped/1024/1024)
    fmt.Printf("Available: %d MB\n", usage.Available/1024/1024)

    // Test search performance
    fmt.Printf("\n=== Performance Test ===\n")
    queryVector := make([]float32, 768)
    for i := range queryVector {
        queryVector[i] = rand.Float32()
    }

    // Test different search parameters
    searchParams := []struct {
        name     string
        efSearch int
        k        int
    }{
        {"Fast", 50, 10},
        {"Balanced", 100, 10},
        {"Accurate", 200, 10},
    }

    for _, param := range searchParams {
        results, err := collection.Query(context.Background()).
            WithVector(queryVector).
            WithEfSearch(param.efSearch).
            Limit(param.k).
            Execute()
        
        if err != nil {
            log.Printf("Search failed: %v", err)
            continue
        }

        fmt.Printf("%s search (ef=%d): %v, %d results\n",
            param.name, param.efSearch, results.Took, len(results.Results))
    }

    // Trigger optimization
    fmt.Printf("\n=== Collection Optimization ===\n")
    optimizationOptions := &libravdb.OptimizationOptions{
        RebuildIndex:   false, // Don't rebuild, just optimize memory
        OptimizeMemory: true,
        CompactStorage: true,
    }

    err = collection.OptimizeCollection(context.Background(), optimizationOptions)
    if err != nil {
        log.Printf("Optimization failed: %v", err)
    } else {
        fmt.Println("Collection optimized successfully")
        
        // Check memory usage after optimization
        usage, _ = collection.GetMemoryUsage()
        fmt.Printf("Memory after optimization: %d MB\n", usage.Total/1024/1024)
    }
}
```

## Real-time Recommendation System

```go
package main

import (
    "context"
    "fmt"
    "log"
    "math/rand"
    "time"

    "github.com/xDarkicex/libravdb/libravdb"
)

func main() {
    db, err := libravdb.New(libravdb.WithStoragePath("./recommendations"))
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()

    // Create collection optimized for real-time queries
    collection, err := db.CreateCollection(
        context.Background(),
        "products",
        libravdb.WithDimension(128), // Smaller dimension for speed
        libravdb.WithMetric(libravdb.CosineDistance),
        libravdb.WithHNSW(32, 200, 50), // Balanced HNSW parameters
        libravdb.WithMemoryLimit(1*1024*1024*1024), // 1GB limit
    )
    if err != nil {
        log.Fatal(err)
    }

    // Simulate product catalog
    categories := []string{"electronics", "books", "clothing", "home", "sports"}
    brands := []string{"BrandA", "BrandB", "BrandC", "BrandD", "BrandE"}

    fmt.Println("Loading product catalog...")
    for i := 0; i < 10000; i++ {
        vector := make([]float32, 128)
        for j := range vector {
            vector[j] = rand.Float32()
        }

        metadata := map[string]interface{}{
            "name":     fmt.Sprintf("Product %d", i),
            "category": categories[i%len(categories)],
            "brand":    brands[i%len(brands)],
            "price":    rand.Float64()*1000 + 10, // $10-$1010
            "rating":   rand.Float64()*2 + 3,     // 3.0-5.0 stars
            "in_stock": rand.Float64() > 0.1,     // 90% in stock
        }

        err := collection.Insert(context.Background(), fmt.Sprintf("product_%d", i), vector, metadata)
        if err != nil {
            log.Printf("Failed to insert product %d: %v", i, err)
        }
    }

    fmt.Printf("Loaded %d products\n", collection.Stats().VectorCount)

    // Simulate real-time recommendation requests
    fmt.Println("\n=== Real-time Recommendation Simulation ===")

    userPreferences := []struct {
        name       string
        category   string
        maxPrice   float64
        minRating  float64
    }{
        {"Tech Enthusiast", "electronics", 500, 4.0},
        {"Book Lover", "books", 50, 3.5},
        {"Fashion Forward", "clothing", 200, 4.2},
        {"Home Decorator", "home", 300, 3.8},
        {"Sports Fan", "sports", 150, 4.0},
    }

    for _, user := range userPreferences {
        fmt.Printf("\nRecommendations for %s:\n", user.name)

        // Generate user preference vector (in real app, this would be learned)
        userVector := make([]float32, 128)
        for i := range userVector {
            userVector[i] = rand.Float32()
        }

        start := time.Now()

        // Get personalized recommendations with filtering
        results, err := collection.Query(context.Background()).
            WithVector(userVector).
            And().
                Eq("category", user.category).
                Lte("price", user.maxPrice).
                Gte("rating", user.minRating).
                Eq("in_stock", true).
            End().
            Limit(5).
            Execute()

        queryTime := time.Since(start)

        if err != nil {
            log.Printf("Recommendation query failed: %v", err)
            continue
        }

        fmt.Printf("  Query time: %v\n", queryTime)
        fmt.Printf("  Found %d recommendations:\n", len(results.Results))

        for i, result := range results.Results {
            fmt.Printf("    %d. %s - %s ($%.2f, %.1f★, similarity: %.3f)\n",
                i+1,
                result.Metadata["name"],
                result.Metadata["brand"],
                result.Metadata["price"],
                result.Metadata["rating"],
                result.Score)
        }
    }

    // Performance benchmark
    fmt.Println("\n=== Performance Benchmark ===")
    
    numQueries := 1000
    queryVector := make([]float32, 128)
    for i := range queryVector {
        queryVector[i] = rand.Float32()
    }

    start := time.Now()
    for i := 0; i < numQueries; i++ {
        _, err := collection.Search(context.Background(), queryVector, 10)
        if err != nil {
            log.Printf("Benchmark query %d failed: %v", i, err)
        }
    }
    totalTime := time.Since(start)

    fmt.Printf("Executed %d queries in %v\n", numQueries, totalTime)
    fmt.Printf("Average query time: %v\n", totalTime/time.Duration(numQueries))
    fmt.Printf("Queries per second: %.0f\n", float64(numQueries)/totalTime.Seconds())

    // Collection statistics
    fmt.Println("\n=== Collection Statistics ===")
    stats := collection.Stats()
    fmt.Printf("Vector count: %d\n", stats.VectorCount)
    fmt.Printf("Memory usage: %d MB\n", stats.MemoryUsage/1024/1024)
    fmt.Printf("Index type: %s\n", stats.IndexType)
    fmt.Printf("Has quantization: %v\n", stats.HasQuantization)
}
```

These examples demonstrate:

1. **Simple Vector Search** - Basic setup and search operations
2. **Document Search** - Metadata schemas and complex filtering
3. **Batch Processing** - High-throughput streaming insertions
4. **Memory Optimization** - Large collections with memory constraints
5. **Real-time Recommendations** - Low-latency queries with filtering

Each example includes error handling, performance monitoring, and best practices for production use.