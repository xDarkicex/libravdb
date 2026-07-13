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
- [Measured Performance](#-measured-performance)
- [Durability and Recovery](#-durability-and-recovery)
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

## 📊 Measured Performance

These are local Apple M2 measurements with Go 1.25 and 768-dimensional
`float32` vectors. HNSW construction numbers marked graph-ready preload vector
storage outside the timed region so graph topology cost is not confused with
the required owned-vector copy. Results are workload and hardware dependent;
use the commands below on deployment hardware.

### HNSW Construction And Search

The primary scale fixture contains 50,000 deduplicated LongMemEval messages and
100 held-out queries embedded with Nomic Embed Text v1.5 Q8 GGUF at 768
dimensions. Exact brute-force squared-L2 top-10 results provide ground truth.
Fixture SHA-256:
`a176d920c50b0d8e635c522e1452b4f282c0c99b9b223089c66a9ae86bf4a243`.

| Configuration | Graph-ready inserts/s | Recall@10, ef=200 | Higher-ef / repair outcome |
|---|---:|---:|---|
| `M=36`, serial | 628 | 1.000 | Exact at ef=200 and ef=300 |
| `M=36`, four workers | 1,706-1,929 | 0.996-1.000 | Schedule-dependent; some misses persisted through ef=300 |
| `M=24`, four workers | 2,602 | 0.998 | 0.999 from ef=216 through ef=300 |
| `M=16`, four workers | 2,946-3,408 | 0.996-0.998 | Up to 0.999 at ef=300 across repeated builds |
| `M=16`, serial | 1,122 | 0.999 | 1.000 at ef=300 |
| `M=36`, four workers plus synchronous repair | 1,246 | 1.000 | Repaired 47,445 of 50,000 nodes |

At `M=16` and `efSearch=200`, search measured approximately 0.33-0.39 ms
p50 and 0.66-0.78 ms p99. The exact serial `M=36` control measured p50
0.634 ms and p99 1.240 ms at ef=200.

Concurrent topology quality is schedule-dependent. Increasing `efSearch`
repairs shallow misses, but some concurrent builds retain topology misses
through ef=300. Blanket repair restores exact recall but touches almost the
entire graph and gives back most of the construction throughput.

The separate 5k normalized-random fixture remains an adversarial isotropic
topology test. On that fixture the ARM64 path produced 873.5 graph inserts/s at
`M=36`, recall@10=1.000, and 0 B/op with 0 allocs/op in HNSW traversal. It is
not presented as the production semantic workload.

Run the checked-in 5k parameter and concurrency benchmarks:

```bash
go test ./internal/index/hnsw \
  -run '^$' \
  -bench 'BenchmarkHNSWNomic768(BuildParam|Concurrent)' \
  -benchtime=1x \
  -count=1 \
  -benchmem
```

The 50k semantic benchmark needs the external fixture described in
[`docs/research/semantic-scale-validation.md`](docs/research/semantic-scale-validation.md):

```bash
LIBRAVDB_SEMANTIC_FIXTURE=/tmp/nomic-longmemeval-50k-gguf-q8.semantic.f32 \
go test ./internal/index/hnsw \
  -run '^$' \
  -bench '^BenchmarkHNSWSemanticScale$' \
  -benchtime=1x \
  -count=1 \
  -v
```

### Durable WAL And Asynchronous HNSW

The active WAL is the WAL inside `internal/storage/singlefile`, not the
standalone graph-store WAL package. These measurements include the canonical
768d vector write and synchronous durability acknowledgement, but exclude HNSW
construction unless stated otherwise.

| Workload | Measured result | WAL group occupancy |
|---|---:|---:|
| Durable, 8 pending writers | approximately 1.44k writes/s | 7.99 entries/transaction |
| Durable, 32 pending writers | 5.44k-5.61k writes/s | 31.88-31.93 entries/transaction |
| Durable 256-entry batches | approximately 49k vectors/s | 256 entries/transaction |
| Async HNSW, retained 28-entry/5 ms policy | 4.00k-4.40k durable acknowledgements/s; 3.33k-3.61k graph-ready/s | 28.4-28.7 entries/transaction |

Later off-heap WAL request and reusable descriptor work reduced the integrated
async HNSW path to approximately 537-548 B/op and 5 allocations/op while
retaining roughly 6.3k accepted writes/s when the bounded index queue has
capacity. Accepted throughput is not graph-ready throughput: once the queue is
full, backpressure forces the two rates to converge.

Reproduce the WAL and integrated asynchronous-index measurements:

```bash
go test ./internal/storage/singlefile \
  -run '^$' \
  -bench '^BenchmarkWALInsertConcurrent$' \
  -benchtime=3000x \
  -count=3 \
  -benchmem

go test ./libravdb \
  -run '^$' \
  -bench '^BenchmarkCollectionAsyncHNSWInsert$' \
  -benchtime=5000x \
  -count=3 \
  -benchmem
```

Restart benchmarks cover persisted-index loading and forced rebuild fallback:

```bash
go test ./libravdb \
  -run '^$' \
  -bench '^BenchmarkRestart.*(Persisted|Rebuild)$' \
  -benchtime=1x \
  -benchmem
```

Detailed methodology and rejected experiments are recorded in
[`docs/research/semantic-scale-validation.md`](docs/research/semantic-scale-validation.md),
[`docs/research/diskann-soa-candidate-queue-experiment.md`](docs/research/diskann-soa-candidate-queue-experiment.md),
and [`docs/research/async-wal-indexing-plan.md`](docs/research/async-wal-indexing-plan.md).

### Throughput Boundaries And The 20-30k Target

LibraVDB reports three different write rates. They are not interchangeable:

| Rate | Meaning |
|---|---|
| Admission rate | The bounded frontend reserved capacity for a write. This is internal flow-control progress, not a user acknowledgement. |
| Durable acknowledgements/s | `Insert` returned after the canonical vector record survived the WAL synchronization barrier. The async benchmark reports this as `accepted_writes/s`; it is durable but may not be graph-visible yet. |
| Graph-ready inserts/s | HNSW construction completed and the record is visible to ANN traversal. |

The durable WAL is not the current 768d throughput ceiling. It has measured
approximately 49k vectors/s for durable 256-record batches. The current
single-graph ceiling is exact FP32 HNSW construction, which reaches roughly
2.9k-3.4k graph-ready inserts/s on the four-worker 50k semantic fixture at
`M=16`.

Approximately 85% of sampled construction CPU is inside exact SIMD distance
kernels. At 3.5k graph-ready inserts/s, reaching 20k requires a 5.7x total
speedup and reaching 30k requires 8.6x. With 15% of time outside distance
kernels, eliminating distance calculation entirely would still cap the current
control flow near 23k/s. The 20-30k graph-ready target therefore requires less
vector work and less per-insert orchestration; another isolated queue or heap
optimization is insufficient.

The planned path is deliberately separated from shipped benchmark claims:

1. **Daemon-side durable microbatching:** coalesce approximately 128-256
   records per `InsertBatch` call so gRPC ingestion can use the WAL's measured
   durable batch capacity.
2. **Exact-bounded int8 construction codes:** keep canonical FP32 vectors, use
   compact int8 codes plus conservative distance-error intervals for candidate
   expansion, and load FP32 only when candidate bounds overlap. This remains a
   research target until fallback rate, topology, recall, persistence, and
   cross-platform SIMD results pass the semantic benchmark.
3. **Epoch-batched HNSW mutation:** evaluate a bounded batch against a stable
   graph epoch, group backlinks by target, prune each affected neighborhood
   once, and publish the completed adjacency changes. This attacks repeated
   overflow pruning and random mutation traffic without changing the durable
   record model.
4. **Configurable independent shards:** construct separate HNSW graphs without
   shared adjacency mutation, search shards concurrently, and merge top-k
   results. Aggregate graph-ready throughput can scale with cores and memory
   channels, at the cost of query fan-out that must remain inside the p99
   latency budget.

The quantized construction bound follows from the triangle inequality. For a
vector `x`, reconstruction `x_hat`, and reconstruction error
`epsilon_x = ||x - x_hat||`:

```text
abs(||q - x|| - ||q_hat - x_hat||) <= epsilon_q + epsilon_x

lower = max(0, ||q_hat - x_hat|| - epsilon_q - epsilon_x)
upper =        ||q_hat - x_hat|| + epsilon_q + epsilon_x
```

Non-overlapping intervals can be ordered without reading the 3,072-byte FP32
payload. Overlapping intervals fall back to exact FP32 distance, preserving the
existing construction decision. Int8 is the first target because it reduces
vector bandwidth by 4x while retaining materially tighter bounds than binary
Hamming codes.

The release does **not** claim 20-30k graph-ready inserts/s today. It does claim
that durable batched ingestion is already in that throughput class and records
the concrete architectural work required to move graph readiness into the same
class.

## 🛡️ Durability And Recovery

Synchronous WAL durability is the default public contract:

- Transactions are framed with magic/version fields, monotonic LSNs, transaction
  IDs, previous-LSN links, and Castagnoli CRC32 checksums.
- A WAL group is appended, synced once with `File.Sync`, and only then published
  to the live record map and acknowledged to writers.
- A write or sync failure is returned to every affected writer; failed durable
  records are not made visible.
- Recovery streams committed transactions in LSN order, discards incomplete
  transactions, and tolerates a truncated final WAL frame.
- Dual metapages and checksummed snapshot/index chunks permit fallback from a
  torn or corrupt newest checkpoint.
- Index snapshots persist their exact applied LSN. HNSW and Flat recovery apply
  bounded post-snapshot insert/update/delete deltas; IVF-PQ falls back to a full
  rebuild when retraining is required.
- Vacuum, migration swaps, backup creation, and file removal sync the containing
  directory on POSIX. Windows replacements use `MOVEFILE_WRITE_THROUGH`.

Do not use `cp` on a live `.libravdb` file. Use `Database.Backup` to create a
point-in-time copy while writes continue:

```go
if err := db.Backup(ctx, "./backup.libravdb"); err != nil {
    log.Fatal(err)
}
```

Async HNSW indexing changes search visibility, not storage durability. A
successful insert is durable before it is necessarily graph-visible. Use
`IndexingStats` to observe the durable/applied LSN gap and `FlushIndex` when a
caller requires a graph-readiness barrier:

```go
stats := collection.IndexingStats()
fmt.Printf("durable=%d applied=%d lag=%d pending=%d\n",
    stats.DurableLSN, stats.AppliedLSN, stats.LSNLag, stats.Pending)

if err := collection.FlushIndex(ctx); err != nil {
    log.Fatal(err)
}
```

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

### Cross-Compilation

LibraVDB is a Go library, so applications compile it into their own binary.
Set `GOOS` and `GOARCH` when building the consuming application; users do not
need a separate LibraVDB runtime installed.

```bash
# Linux x86-64
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
  go build -trimpath -o app-linux-amd64 ./cmd/app

# Linux ARM64
CGO_ENABLED=0 GOOS=linux GOARCH=arm64 \
  go build -trimpath -o app-linux-arm64 ./cmd/app

# macOS Apple Silicon
CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 \
  go build -trimpath -o app-darwin-arm64 ./cmd/app

# Windows x86-64
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 \
  go build -trimpath -o app-windows-amd64.exe ./cmd/app
```

Validated full-library build targets are Linux amd64, Linux arm64, macOS arm64,
and Windows amd64. ARM64 uses the checked-in NEON kernels.
AMD64 selects generated AVX2/FMA kernels at runtime when supported and retains
the generic fallback for older CPUs.

Generated amd64 assembly is committed to the repository, so library consumers
do not need Avo. Contributors changing SIMD generation must regenerate and
verify it:

```bash
go generate ./internal/util/simd
git diff --exit-code -- \
  internal/util/simd/distance_amd64.s \
  internal/util/simd/stub_amd64.go
```

Cross-compile the package itself during development with:

```bash
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build ./libravdb
CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build ./libravdb
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build ./libravdb
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

### Lifecycle, Streaming Export, And Import

`Database.Iterate` exposes every persisted record without knowing whether the
consumer writes a file, sends gRPC messages, pipes stdout, or forwards records
to another database. The callback provides backpressure: LibraVDB does not
advance until it returns.

```go
err := db.Iterate(ctx, func(collectionName string, record libravdb.Record) error {
    // destination belongs to the application or daemon. It can encode, send,
    // pipe, or write the record in any format.
    return destination.WriteRecord(collectionName, record)
})
if err != nil {
    log.Fatal(err)
}
```

Use `Collection.Iterate` when exporting only one collection. `Collection.Config`
returns a defensive copy of the portable collection configuration; the
process-local `Graph` attachment is intentionally omitted.

```go
config := collection.Config()

err := collection.Iterate(ctx, func(record libravdb.Record) error {
    return destination.WriteRecord(collectionName, record)
})
```

Iteration is bounded internally and does not materialize all records. Record
ordering is an implementation detail and must not be used as an export schema.
For a logically consistent export while the live database is changing, create
a point-in-time backup and iterate the backup:

```go
const snapshotPath = "./export-snapshot.libravdb"
if err := db.Backup(ctx, snapshotPath); err != nil {
    log.Fatal(err)
}

snapshot, err := libravdb.Open(libravdb.WithStoragePath(snapshotPath))
if err != nil {
    log.Fatal(err)
}
defer snapshot.Close()

if err := snapshot.Iterate(ctx, destination.WriteRecord); err != nil {
    log.Fatal(err)
}
```

LibraVDB does not impose an archive or wire-format importer. A consumer decodes
its own input and feeds records into the existing bounded batch API. Imports
should normally target an empty collection; `InsertBatch` assigns new internal
ordinals and record versions while rebuilding the selected index.

```go
const importBatchSize = 1000
batch := make([]libravdb.VectorEntry, 0, importBatchSize)

for source.Next() {
    item := source.Record()
    batch = append(batch, libravdb.VectorEntry{
        ID:       item.ID,
        Vector:   item.Vector,
        Metadata: item.Metadata,
    })

    if len(batch) == importBatchSize {
        if err := collection.InsertBatch(ctx, batch); err != nil {
            log.Fatal(err)
        }
        batch = batch[:0]
    }
}
if err := source.Err(); err != nil {
    log.Fatal(err)
}
if len(batch) > 0 {
    if err := collection.InsertBatch(ctx, batch); err != nil {
        log.Fatal(err)
    }
}
```

For an exact physical copy that preserves storage versions, ordinals, and
checkpoint state, use `Backup` instead of logical export/import.

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

Performance results are reported with their dataset, dimensions, durability
mode, recall, concurrency, and hardware in [Measured Performance](#-measured-performance).
The project does not publish a context-free insert or QPS headline because WAL
acknowledgement, graph readiness, vector ownership, recall target, and index
configuration measure different work.

CI runs unit, integration, race, graph, HNSW, and SIMD tests across the
configured Linux amd64 and macOS arm64 matrix. The release validation described
under Cross-Compilation additionally builds Linux arm64 and Windows amd64.

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
