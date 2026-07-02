# LibraVDB

<div align="center">

[![Go Version](https://img.shields.io/badge/go-1.25+-blue.svg)](https://golang.org/doc/devel/release.html)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Go Report Card](https://goreportcard.com/badge/github.com/xDarkicex/libravdb)](https://goreportcard.com/report/github.com/xDarkicex/libravdb)
[![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen.svg)](https://github.com/xDarkicex/libravdb/actions)
[![Security Scan](https://img.shields.io/badge/security-govulncheck-brightgreen.svg)](https://github.com/xDarkicex/libravdb/actions)
[![Go Reference](https://pkg.go.dev/badge/github.com/xDarkicex/libravdb.svg)](https://pkg.go.dev/github.com/xDarkicex/libravdb)

**High-Performance Hybrid Vector-Graph Database for Go**

*Vector similarity search meets directed relationship modeling — zero-allocation graph traversal with WAL durability*

[**Quick Start**](#-quick-start) •
[**Documentation**](#-documentation) •
[**Examples**](#-usage-examples) •
[**Contributing**](#-contributing)

</div>

---

## Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Architecture](#-architecture)
- [Documentation](#-documentation)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Advanced Features](#-advanced-features)
- [Roadmap](#-roadmap)
- [Use Cases](#-use-cases)
- [Development](#-development)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [Community](#-community)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## 🎯 Overview

LibraVDB is a high-performance hybrid vector-graph database library for Go applications. It provides similarity search, metadata-aware retrieval, directed edge relationship modeling, graph traversal, batch and streaming ingestion, and persistent single-file storage with HNSW, IVF-PQ, and Flat indexing.

### Why LibraVDB?

- **🚀 Performance First**: Optimized for high-throughput insertions and sub-millisecond search latency
- **🔧 Go Native**: Designed specifically for Go with idiomatic APIs and zero external dependencies
- **📈 Durable by Default**: Single-file binary persistence with WAL-backed durability and crash recovery
- **🧠 Memory Efficient**: Off-heap memory management via `github.com/xDarkicex/memory` — zero GC pressure on hot paths
- **🔀 Hybrid Vector-Graph**: Directed typed edges, BFS traversal, reverse-index lookups, and graph-filtered similarity search
- **🔍 Feature Rich**: Complex filtering, streaming operations, and automatic index optimization
- **📊 Observable**: Built-in metrics, health checks, and performance monitoring
- **🛡️ Safer Writes**: Bounded write admission for concurrent, batch, and streaming writers

## ✨ Key Features

### Core Capabilities
- **Multiple Index Types**: HNSW, IVF-PQ, and Flat algorithms with automatic selection
- **Advanced Quantization**: Product and Scalar quantization for memory optimization
- **Rich Metadata Filtering**: Complex AND/OR/NOT operations with type-safe schemas
- **Streaming Operations**: High-throughput batch processing with backpressure control
- **Memory Management**: Configurable limits, memory mapping, and automatic optimization
- **Graph Layer** (`v1.2.0`): Directed typed edges between vectors with zero heap allocations. 16-byte fixed Edge (Target, Weight, Stamp/Kind), EdgeTable 4KB pages with inline-first-8 layout, lock-free reads via Hyaline SMR, BFS traversal with caller-managed off-heap buffers, reverse index for O(degree) node deletion, WAL durability (4 new op types, CRC32 checksums), segment persistence with zero-copy mmap I/O, graph-filtered similarity search across all index types, insert/delete hooks for automatic edge maintenance
- **Persistent Storage**: Single-file binary storage with WAL-backed durability
- **Storage-Owned HNSW**: Canonical vectors and metadata live in storage; HNSW owns graph topology plus optional compressed artifacts

## 📦 Persistence Model

LibraVDB persists databases as a single `.libravdb` file.

- Importing the package does not create files.
- A database file is created or opened when you call `libravdb.Open(...)`.
- If no path is provided, the default path resolves to `./data.libravdb`.
- `WithStoragePath(...)` should point to a database file such as `./mydb.libravdb`.
- The `.libravdb` file is the portable unit you can move or copy after closing the database.

For HNSW-backed collections:
- canonical raw vectors and metadata are stored once in canonical storage
- HNSW uses internal ordinals and provider-backed vector access
- HNSW nodes do not own raw vectors or metadata
- optional compressed vectors remain index-owned derived data

## 🛡️ Write Concurrency Safety

LibraVDB now includes a Phase 1 write-admission layer intended to make local and plugin-style usage safer.

- direct writes, batch writes, and streaming writes are admitted through a bounded per-collection write gate
- queued writers are bounded instead of piling up indefinitely
- waiting writers respect context cancellation
- batch and streaming worker counts are clamped to collection write parallelism

This improves safety under bursty or subagent-style write traffic, but it is not yet the full adaptive scheduler. If you expect very heavy write concurrency, keep batch and streaming concurrency conservative and prefer one coordinated writer path per collection.

## 🚀 Quick Start

### Installation

```bash
go get github.com/xDarkicex/libravdb
```

### Basic Example

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/xDarkicex/libravdb/libravdb"
)

func main() {
    // Create a single-file database
    db, err := libravdb.Open(
        libravdb.WithStoragePath("./vector_data.libravdb"),
        libravdb.WithMetrics(true),
    )
    if err != nil {
        log.Fatal("Failed to create database:", err)
    }
    defer db.Close()

    // Create collection with automatic optimization
    collection, err := db.CreateCollection(
        context.Background(),
        "documents",
        libravdb.WithDimension(768),                    // OpenAI embedding size
        libravdb.WithMetric(libravdb.CosineDistance),   // Best for text embeddings
        libravdb.WithAutoIndexSelection(true),          // Automatic optimization
        libravdb.WithMemoryLimit(2*1024*1024*1024),     // 2GB memory limit
    )
    if err != nil {
        log.Fatal("Failed to create collection:", err)
    }

    // Insert vectors with metadata
    documents := []struct {
        id       string
        vector   []float32
        metadata map[string]interface{}
    }{
        {
            id:     "doc1",
            vector: generateEmbedding("Machine learning fundamentals"),
            metadata: map[string]interface{}{
                "title":    "ML Fundamentals",
                "category": "education",
                "tags":     []string{"ml", "ai", "tutorial"},
                "score":    4.8,
            },
        },
        // ... more documents
    }

    for _, doc := range documents {
        err := collection.Insert(context.Background(), doc.id, doc.vector, doc.metadata)
        if err != nil {
            log.Printf("Failed to insert %s: %v", doc.id, err)
        }
    }

    // Perform similarity search with filtering
    queryVector := generateEmbedding("artificial intelligence tutorial")
    
    results, err := collection.Query(context.Background()).
        WithVector(queryVector).
        And().
            Eq("category", "education").
            Gte("score", 4.0).
            ContainsAny("tags", []interface{}{"ai", "ml"}).
        End().
        Limit(10).
        Execute()
    
    if err != nil {
        log.Fatal("Search failed:", err)
    }

    // Display results
    fmt.Printf("Found %d relevant documents:\n", len(results.Results))
    for i, result := range results.Results {
        fmt.Printf("%d. %s (similarity: %.3f)\n", 
            i+1, result.Metadata["title"], result.Score)
    }
}

func generateEmbedding(text string) []float32 {
    // In practice, use OpenAI, Cohere, or other embedding APIs
    // This is just a placeholder
    embedding := make([]float32, 768)
    // ... generate actual embedding
    return embedding
}
```

## 💡 Usage Examples

### Document Search System

```go
// Create collection optimized for text search
collection, err := db.CreateCollection(ctx, "documents",
    libravdb.WithDimension(1536),                       // OpenAI text-embedding-3-large
    libravdb.WithMetric(libravdb.CosineDistance),
    libravdb.WithHNSW(32, 200, 100),                   // High accuracy settings
    libravdb.WithMetadataSchema(libravdb.MetadataSchema{
        "title":     libravdb.StringField,
        "content":   libravdb.StringField,
        "category":  libravdb.StringField,
        "tags":      libravdb.StringArrayField,
        "published": libravdb.TimeField,
        "score":     libravdb.FloatField,
    }),
    libravdb.WithIndexedFields("category", "published"), // Fast filtering
)
```

### High-Throughput Batch Processing

```go
// Configure for controlled throughput
opts := &libravdb.StreamingOptions{
    BufferSize:     50000,
    ChunkSize:      5000,
    MaxConcurrency: 2,
    Timeout:        5 * time.Minute,
    ProgressCallback: func(stats *libravdb.StreamingStats) {
        fmt.Printf("Processed: %d/%d (%.1f%%), Rate: %.0f/sec\n",
            stats.TotalProcessed, stats.TotalReceived,
            float64(stats.TotalProcessed)/float64(stats.TotalReceived)*100,
            stats.ItemsPerSecond)
    },
}

stream := collection.NewStreamingBatchInsert(opts)
stream.Start()

// Process large numbers of vectors without unbounded writer fan-out
for _, entry := range millionVectorDataset {
    stream.Send(entry)
}

stats := stream.Stats()
fmt.Printf("Final: %d processed, %d successful, %d failed\n",
    stats.TotalProcessed, stats.TotalSuccessful, stats.TotalFailed)
```

### Memory-Optimized Large Scale

```go
// Configure for large datasets with limited memory
collection, err := db.CreateCollection(ctx, "large_scale",
    libravdb.WithDimension(768),
    libravdb.WithAutoIndexSelection(true),              // Automatic optimization
    libravdb.WithMemoryLimit(8*1024*1024*1024),         // 8GB limit
    libravdb.WithMemoryMapping(true),                   // Use disk for overflow
    libravdb.WithProductQuantization(16, 8, 0.05),      // 16x compression
    libravdb.WithCachePolicy(libravdb.LRUCache),
)
```

### Real-Time Recommendation Engine

```go
// Optimized for low-latency queries
collection, err := db.CreateCollection(ctx, "recommendations",
    libravdb.WithDimension(256),                        // Smaller for speed
    libravdb.WithHNSW(16, 100, 50),                    // Fast search settings
    libravdb.WithMemoryLimit(4*1024*1024*1024),         // Keep in memory
)

// Real-time recommendation query
recommendations, err := collection.Query(ctx).
    WithVector(userPreferenceVector).
    And().
        Eq("category", userCategory).
        Gte("rating", 4.0).
        NotEq("user_id", currentUserID).
    End().
    WithThreshold(0.7).                                 // Minimum similarity
    Limit(20).
    Execute()
```

### Graph Layer

```go
// Create a graph for relationship modeling between vectors
graph, err := libravdb.NewGraph(libravdb.GraphConfig{
    EdgeSlots: 1000000,     // 1M edges
    PageSlots: 65536,       // 64K pages
})
if err != nil {
    log.Fatal(err)
}

// Attach the graph to a new collection during creation
collection, err := db.CreateCollection(ctx, "nodes",
    libravdb.WithDimension(768),
    libravdb.WithGraph(graph),
)

// Or attach it to an existing collection later
collection.SetGraph(graph)

// Register an insert hook to automatically create edges based on vector metadata
collection.RegisterInsertHook(func(txn libravdb.GraphTx, id uint64, vector []float32, metadata map[string]interface{}) error {
    if parentID, ok := metadata["parent_id"].(uint64); ok {
        return txn.AddEdge(id, parentID, 1.0, 2) // kind=2: "child_of"
    }
    return nil
})

// Begin a transaction to perform manual mutations
txn := graph.BeginTxn()
graph.AddEdge(txn, 1, 2, 0.95, 1)   // kind=1: "similar_to"
graph.AddEdge(txn, 2, 3, 0.87, 2)   // kind=2: "derived_from"

// Query outgoing edges for a node
edges, _ := graph.Neighbors(1)
for _, e := range edges {
    fmt.Printf("1 -> %d (kind=%d, weight=%.3f)\n", e.Target, e.Kind, e.Weight)
}

// Filter by edge kind with KindSet
kindSet := libravdb.NewKindSet(1) // only "similar_to" edges
similar, _ := graph.NeighborsAny(1, kindSet)

// Degree and reverse lookups
count, _ := graph.Degree(1)
inbound, _ := graph.InboundNeighbors(1) // "what points to node 1?"

// Graph-filtered vector search
bitset, _ := graph.GetBitset()
frontier, _ := graph.GetFrontierBuf()
defer graph.PutBitset(bitset)
defer graph.PutFrontierBuf(frontier)

// BFS from node 1, collecting all reachable nodes into the bitset
graph.BFS(1, 3, func(nodeID uint64, depth int) bool {
    return true
}, bitset, frontier)

// Search only among nodes reachable from node 1
results, _ := collection.Query(ctx).
    WithVector(queryVector).
    WithGraphFilter(bitset).
    Limit(10).
    Execute()

// Iterate all edges
graph.ForEachEdge(func(src, tgt uint64, edge libravdb.Edge) bool {
    fmt.Printf("%d -> %d\n", src, tgt)
    return true
})

// Remove edges
graph.RemoveEdge(txn, 1, 2, 1) // remove kind=1 edge from 1->2
graph.DropNodeEdges(txn, 1)    // remove all edges for node 1
```

More examples available in [docs/examples/](docs/examples/).

## 🏗️ Architecture

LibraVDB employs a layered architecture designed for performance, scalability, and maintainability:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Application Layer                              │
├─────────────────────────────────────────────────────────────────────────┤
│                           LibraVDB API                                 │
│  Database Mgmt │ Collection Ops │ Query Builder │ Stream │ Graph API   │
├─────────────────────────────────────────────────────────────────────────┤
│                          Processing Layer                               │
│   Index Layer  │  Filter Layer  │  Memory Mgmt  │  Graph Layer │ Obs.  │
├─────────────────────────────────────────────────────────────────────────┤
│                         Algorithm Layer                                 │
│  HNSW │ IVF-PQ │ Flat │ BFS Traversal │ Quantization │ Cache │ Monitor  │
├─────────────────────────────────────────────────────────────────────────┤
│                          Storage Layer                                 │
│  Single-File Engine │ Canonical Records │ WAL / Snapshot │ Segments    │
├─────────────────────────────────────────────────────────────────────────┤
│                        Operating System                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Components

- **Database Layer**: Collection management, global configuration, health monitoring
- **Collection Layer**: Vector operations, metadata management, index coordination, write admission, hook execution
- **Index Layer**: HNSW, IVF-PQ, and Flat algorithms with automatic selection and graph-filtered search
- **Graph Layer**: Directed typed edges, EdgeTable pages, BFS traversal, reverse index, WAL durability, segment persistence
- **Storage Layer**: Single-file canonical storage with WAL-backed durability and reopen/rebuild support
- **Memory Layer**: Off-heap memory management via `github.com/xDarkicex/memory` with Hyaline SMR, ShardedFreeList, Arena
- **Observability Layer**: Metrics, tracing, health checks, and performance monitoring

Detailed architecture documentation: [docs/design/architecture.md](docs/design/architecture.md)

## 📚 Documentation

### Getting Started
- [**Installation & Setup**](docs/getting-started.md) - Complete setup guide with examples
- [**API Reference**](docs/api-reference.md) - Comprehensive API documentation
- [**Configuration Guide**](docs/configuration/configuration.md) - Advanced configuration options
- [**Performance Tuning**](docs/configuration/performance-tuning.md) - Optimization strategies

### Core Concepts
- [**Collections**](docs/concepts/collections.md) - Understanding vector collections and lifecycle
- [**Indexing Algorithms**](docs/concepts/indexing.md) - HNSW, IVF-PQ, and Flat indexes explained
- [**Memory Management**](docs/concepts/memory-management.md) - Memory optimization strategies
- [**Filtering**](docs/concepts/filtering.md) - Advanced metadata filtering capabilities

### Advanced Topics
- [**Architecture Design**](docs/design/architecture.md) - System architecture and component design
- [**HNSW Implementation**](docs/design/hnsw.md) - Detailed HNSW algorithm implementation
- [**Storage Design**](docs/design/storage.md) - Single-file storage architecture
- [**API Design**](docs/design/api.md) - API design principles and patterns

### Examples & Tutorials
- [**Basic Usage**](docs/examples/basic_usage.md) - Fundamental operations and patterns
- [**Advanced Error Handling**](docs/errors/advanced_error_handling.md) - Error handling strategies
- [**Schema Specifications**](docs/schema/) - Binary format specifications

## 📦 Installation

### Requirements

- **Go 1.25+**: LibraVDB requires Go 1.25 or later
- **Memory**: Minimum 1GB RAM (4GB+ recommended for production)
- **Storage**: SSD recommended for optimal performance
- **CPU**: Multi-core processor recommended for search and controlled batch ingestion

### Install via Go Modules

```bash
go get github.com/xDarkicex/libravdb
```

### Development Installation

```bash
# Clone repository
git clone https://github.com/xDarkicex/libravdb.git
cd libravdb

# Setup development environment
./scripts/setup.sh

# Verify installation
go build ./...
go test ./...
```

### Docker Support

```dockerfile
FROM golang:1.25-alpine AS builder
WORKDIR /app
COPY . .
RUN go build -o libravdb-app ./examples/

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /app/libravdb-app .
CMD ["./libravdb-app"]
```

## ⚙️ Configuration

LibraVDB provides extensive configuration options for optimal performance:

### Database Configuration

```go
db, err := libravdb.Open(
    libravdb.WithStoragePath("/var/lib/libravdb/data.libravdb"), // Production database file
    libravdb.WithMetrics(true),                        // Enable Prometheus metrics
    libravdb.WithTracing(true),                        // Enable distributed tracing
    libravdb.WithMaxCollections(1000),                 // Maximum collections
    libravdb.WithMaxConcurrentWrites(2),               // Conservative write parallelism
    libravdb.WithMaxWriteQueueDepth(32),               // Bound queued writers
)
```

### Collection Configuration

```go
collection, err := db.CreateCollection(ctx, "vectors",
    // Basic configuration
    libravdb.WithDimension(768),
    libravdb.WithMetric(libravdb.CosineDistance),
    
    // Index configuration
    libravdb.WithHNSW(32, 400, 100),                   // High accuracy HNSW
    libravdb.WithAutoIndexSelection(true),             // Automatic optimization
    
    // Memory management
    libravdb.WithMemoryLimit(16*1024*1024*1024),       // 16GB limit
    libravdb.WithMemoryMapping(true),                  // Enable memory mapping
    libravdb.WithCachePolicy(libravdb.LRUCache),
    
    // Quantization
    libravdb.WithProductQuantization(8, 8, 0.1),       // 8x compression
    
    // Metadata and filtering
    libravdb.WithMetadataSchema(schema),
    libravdb.WithIndexedFields("category", "timestamp"),
    
)
```

### Environment-Specific Configurations

**Development**:
```go
libravdb.WithStoragePath("./dev_data.libravdb")
libravdb.WithMetrics(false)
libravdb.WithMemoryLimit(1*1024*1024*1024) // 1GB
```

**Production**:
```go
libravdb.WithStoragePath("/var/lib/libravdb/data.libravdb")
libravdb.WithMetrics(true)
libravdb.WithTracing(true)
libravdb.WithMaxConcurrentWrites(2)
libravdb.WithMaxWriteQueueDepth(64)
libravdb.WithMemoryLimit(32*1024*1024*1024) // 32GB
```

**High-Scale**:
```go
libravdb.WithStoragePath("/var/lib/libravdb/data.libravdb")
libravdb.WithAutoIndexSelection(true)
libravdb.WithMemoryMapping(true)
libravdb.WithProductQuantization(16, 8, 0.05)
libravdb.WithMaxConcurrentWrites(2)
libravdb.WithMaxWriteQueueDepth(128)
```

Complete configuration guide: [docs/configuration/configuration.md](docs/configuration/configuration.md)

## 🔧 Advanced Features

### Automatic Index Optimization

```go
// LibraVDB automatically selects the best index type based on collection size
collection, err := db.CreateCollection(ctx, "adaptive",
    libravdb.WithAutoIndexSelection(true),
    libravdb.WithDimension(768),
)

// Manual optimization
err = collection.OptimizeCollection(ctx, &libravdb.OptimizationOptions{
    RebuildIndex:       true,
    OptimizeMemory:     true,
    UpdateQuantization: true,
})
```

### Advanced Memory Management

```go
// Set global memory limits
err = db.SetGlobalMemoryLimit(64 * 1024 * 1024 * 1024) // 64GB

// Monitor memory usage
usage, err := db.GetGlobalMemoryUsage()
fmt.Printf("Total memory: %d MB, Collections: %d\n", 
    usage.TotalMemory/1024/1024, len(usage.Collections))

// Trigger garbage collection
err = db.TriggerGlobalGC()
```

### Complex Query Operations

```go
// Build complex queries with multiple conditions
results, err := collection.Query(ctx).
    WithVector(queryVector).
    And().
        Or().
            Eq("category", "technology").
            Eq("category", "science").
        End().
        Between("published_date", startDate, endDate).
        Not().
            ContainsAny("tags", []interface{}{"deprecated", "archived"}).
        End().
        Gte("rating", 4.0).
    End().
    WithThreshold(0.8).
    Limit(50).
    Execute()
```

### Database Management

```go
// Safely drop a collection and reclaim its disk space immediately
err = db.DropCollection(ctx, "temporary_data")

// Compact the database to reclaim space from deleted records and collections
err = db.Vacuum(ctx)

// Create a safe point-in-time copy of the database file
err = db.Backup(ctx, "./backup.libravdb")
```

### Migration from v1

LibraVDB v2 introduces a major storage format upgrade. When you call `libravdb.Open()` on a v1 database file, it will automatically stream the data into a new v2 database file and rename the old file to `.v1.bak`. 

If you prefer to perform migrations manually or via script, use `Migrate`:
```go
err := libravdb.Migrate(ctx, "./data.libravdb")
```

### Lifecycle And Export

```go
collections := db.ListCollections()

records, err := collection.Query(ctx).
    Eq("sessionId", "s1").
    Limit(100).
    List()
if err != nil {
    log.Fatal(err)
}

count, err := collection.Count(ctx)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("collection has %d records\n", count)

if err := db.DeleteCollection(ctx, "session:old"); err != nil {
    log.Fatal(err)
}

// Reclaim disk space without blocking concurrent operations
if err := db.Vacuum(ctx); err != nil {
    log.Fatal(err)
}

// Create a safe, point-in-time snapshot to a new path
if err := db.Backup(ctx, "./backup.libravdb"); err != nil {
    log.Fatal(err)
}

// Completely destroy the database file and close the engine
if err := db.Drop(ctx); err != nil {
    log.Fatal(err)
}
```

### Performance Monitoring

```go
// Get detailed collection statistics
stats := collection.Stats()
fmt.Printf("Collection: %s\n", stats.Name)
fmt.Printf("Vectors: %d, Memory: %d MB\n", 
    stats.VectorCount, stats.MemoryUsage/1024/1024)
fmt.Printf("Index: %s, Quantized: %v\n", 
    stats.IndexType, stats.HasQuantization)

// Monitor optimization status
if stats.OptimizationStatus.CanOptimize {
    fmt.Println("Collection can be optimized")
}
```

## 🗺️ Roadmap

### Configurable Local Sharding

Collections already support internal write-lane sharding via `WithSharding(true)`, which splits a collection into parallel shards with independent storage and indexes. The next step is making this user-configurable:

- `WithShardCount(n)` — user-controlled shard count instead of the current hardcoded constant
- `ShardStats` — per-shard observability (vector count, memory usage, disk size) so consumers can detect hot shards
- Expose shard-level index rebuilds for targeted maintenance

### Dynamic Resharding

Adding or removing shards without dropping and recreating the collection:

- **Split**: grow from N → 2N shards, redistributing vectors by a consistent hash ring
- **Merge**: shrink back from 2N → N shards, combining adjacent shard pairs
- Hash-ring-based routing so the owning shard is deterministic and stable across resizes

### Consumer-Side Distributed Sharding

The library stays an embedded single-file engine. Distributed sharding is implemented at the consumer/daemon layer:

- Each `.libravdb` file is a shard — the library needs no network awareness
- The consumer owns hash-based write routing, scatter/gather search with oversampling, and ranked result merging
- Shard sync (WAL shipping, snapshot transfer) lives in the daemon, not the library

This is the "wrap it, don't rewrite it" model — SQLite didn't grow a network stack, people built rqlite around it.

---

## 🎯 Use Cases

### Semantic Search & RAG Applications

```go
// Optimized for text embeddings and semantic search
collection, err := db.CreateCollection(ctx, "knowledge_base",
    libravdb.WithDimension(1536),                       // OpenAI text-embedding-3-large
    libravdb.WithMetric(libravdb.CosineDistance),
    libravdb.WithHNSW(32, 200, 100),
    libravdb.WithMetadataSchema(libravdb.MetadataSchema{
        "document_id": libravdb.StringField,
        "chunk_id":    libravdb.StringField,
        "content":     libravdb.StringField,
        "source":      libravdb.StringField,
        "timestamp":   libravdb.TimeField,
    }),
)
```

### Recommendation Systems

```go
// User and item embeddings for collaborative filtering
userCollection, err := db.CreateCollection(ctx, "users",
    libravdb.WithDimension(128),
    libravdb.WithHNSW(16, 100, 50),                    // Fast recommendations
    libravdb.WithMemoryLimit(2*1024*1024*1024),
)

itemCollection, err := db.CreateCollection(ctx, "items",
    libravdb.WithDimension(128),
    libravdb.WithAutoIndexSelection(true),
    libravdb.WithProductQuantization(8, 8, 0.1),       // Memory efficient
)
```

### Image & Video Search

```go
// Visual similarity search with high-dimensional embeddings
imageCollection, err := db.CreateCollection(ctx, "images",
    libravdb.WithDimension(2048),                       // ResNet/CLIP embeddings
    libravdb.WithMetric(libravdb.L2Distance),           // Good for visual features
    libravdb.WithMemoryMapping(true),                   // Handle large datasets
    libravdb.WithIVFPQ(1000, 20),                      // Memory efficient for large scale
)
```

### Anomaly Detection

```go
// Detect outliers in high-dimensional data
anomalyCollection, err := db.CreateCollection(ctx, "system_metrics",
    libravdb.WithDimension(50),                         // System metrics
    libravdb.WithFlat(),                                // Exact search for anomalies
    libravdb.WithMemoryLimit(1*1024*1024*1024),
)
```

## 🛠️ Development

### Prerequisites

- **Go 1.25+**: Latest Go version for optimal performance
- **Git**: Version control
- **Make** (optional): For convenience commands

### Development Setup

```bash
# Clone and setup
git clone https://github.com/xDarkicex/libravdb.git
cd libravdb

# One-time development environment setup
./scripts/setup.sh

# Verify setup
go build ./...
go test ./...
```

### Development Workflow

```bash
# Format and lint code
./scripts/lint.sh

# Run comprehensive tests
./scripts/test.sh

# Run benchmarks
go test -bench=. -benchmem ./benchmark/

# Generate documentation
go doc -all ./libravdb
```

### Project Structure

```
libravdb/
├── libravdb/          # Main library package (Graph, Hooks, Database, Collection)
├── internal/          # Internal packages
│   ├── graph/         # Graph layer: edges, EdgeTable, BFS, WAL, segments, compaction
│   ├── index/         # Indexing algorithms (HNSW, IVF-PQ, Flat)
│   ├── storage/       # Storage layer (single-file engine, WAL)
│   ├── filter/        # Query filtering
│   ├── quant/         # Quantization
│   ├── obs/           # Observability
│   └── util/          # Utilities
├── examples/          # Usage examples
├── tests/             # Integration tests
├── benchmark/         # Performance benchmarks
├── docs/              # Documentation
└── scripts/           # Development scripts
```

### Code Quality Standards

- **Test Coverage**: >85% for new code
- **Benchmarks**: Required for performance-critical changes
- **Documentation**: GoDoc for all public APIs
- **Linting**: golangci-lint with strict settings
- **Formatting**: gofmt and goimports

## 🧪 Testing

### Test Categories

```bash
# Unit tests
go test ./libravdb -v

# Integration tests
go test -tags=integration ./tests -v

# Benchmark tests
go test -bench=. -benchmem ./...

# Race condition detection
go test -race ./...

# Memory leak detection
go test -memprofile=mem.prof ./...
```

### Comprehensive Test Suite

```bash
# Run all tests with coverage
./scripts/test.sh

# Generate coverage report
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out -o coverage.html

# Performance validation
./benchmark/run_benchmarks.sh
```

### Test Results

Current test coverage and performance metrics:

```
Package Coverage:
- libravdb:     92.3%
- internal/*:   88.7%
- Overall:      89.5%

Benchmark Results:
- Insert:       150K ops/sec
- Search:       12K qps (p95: 0.8ms)
- Memory:       2.1GB for 1M vectors
```

## 🤝 Contributing

We welcome contributions from the community! LibraVDB thrives on collaboration and diverse perspectives.

### How to Contribute

1. **Read our guides**:
   - [Contributing Guidelines](CONTRIBUTING.md)
   - [Code of Conduct](CODE_OF_CONDUCT.md)

2. **Start with issues**:
   - Check [existing issues](https://github.com/xDarkicex/libravdb/issues)
   - Look for `good first issue` labels
   - Discuss your approach before implementing

3. **Development process**:
   - Fork the repository
   - Create a feature branch
   - Write tests for new functionality
   - Ensure all tests pass
   - Submit a pull request

### Contribution Areas

- **🐛 Bug Reports**: Help us identify and fix issues
- **✨ Feature Requests**: Suggest new capabilities
- **📚 Documentation**: Improve guides and examples
- **🚀 Performance**: Optimize algorithms and data structures
- **🧪 Testing**: Expand test coverage and scenarios
- **🔧 Tools**: Improve development and deployment tools

### Recognition

Contributors are recognized in:
- [CONTRIBUTORS.md](CONTRIBUTORS.md) - Hall of fame
- Release notes for significant contributions
- GitHub contributor statistics

## 🌟 Community

### Communication Channels

- **GitHub Issues**: [Bug reports and feature requests](https://github.com/xDarkicex/libravdb/issues)
- **GitHub Discussions**: [Community discussions and Q&A](https://github.com/xDarkicex/libravdb/discussions)
- **Documentation**: [Comprehensive guides and API reference](docs/)

### Getting Help

1. **Check Documentation**: Start with our comprehensive docs
2. **Search Issues**: Look for existing solutions
3. **Ask Questions**: Use GitHub Discussions for help
4. **Report Bugs**: Create detailed issue reports

### Community Guidelines

We are committed to providing a welcoming and inclusive environment. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) and help us maintain a positive community.

## 📄 License

**LibraVDB is licensed under the Apache License 2.0**

```
Copyright 2024 LibraVDB Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### Why Apache 2.0?

- **Permissive**: Allows commercial and private use
- **Patent Protection**: Includes explicit patent license grants
- **Enterprise Friendly**: Widely accepted in corporate environments
- **Community Standard**: Used by major open source projects

### Third-Party Acknowledgments

LibraVDB incorporates research and techniques from various academic papers and open source projects. See the [NOTICE](NOTICE) file for detailed attributions and acknowledgments.

## 🙏 Acknowledgments

LibraVDB builds upon decades of research and development in vector databases and similarity search:

### Research Foundations
- **HNSW Algorithm**: Based on research by Yu. A. Malkov and D. A. Yashunin
- **Single-File Storage Design**: Informed by WAL, snapshot, and durable embedded database design patterns
- **Product Quantization**: Based on work by Hervé Jégou, Matthijs Douze, and Cordelia Schmid
- **Vector Database Concepts**: Building on research from Facebook AI, Google Research, and academic institutions

### Open Source Community
- **Go Community**: For excellent tooling, libraries, and best practices
- **Vector Database Ecosystem**: Learning from projects like Faiss, Annoy, and Hnswlib
- **Contributors**: Everyone who has contributed code, documentation, and feedback

### Special Thanks
- Early adopters and beta testers who provided valuable feedback
- Academic researchers whose work made this project possible
- The broader machine learning and information retrieval communities

---

<div align="center">

**LibraVDB** - *Empowering Go applications with high-performance vector search capabilities*

[**Get Started**](docs/getting-started.md) • [**API Docs**](docs/api-reference.md) • [**Examples**](docs/examples/) • [**Contributing**](CONTRIBUTING.md)

Made with ❤️ by the LibraVDB community

</div>
