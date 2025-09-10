# HNSW (Hierarchical Navigable Small World) Design

This document details the design and implementation of the HNSW algorithm in LibraVDB.

## Algorithm Overview

HNSW is a graph-based approximate nearest neighbor search algorithm that builds a multi-layer graph structure where each layer contains a subset of the data points, with higher layers having exponentially fewer points but longer-range connections.

### Core Concepts

1. **Multi-layer Graph**: Each vector exists in layer 0, with probability of existing in higher layers decreasing exponentially
2. **Navigable Small World**: Each layer forms a navigable small world graph with logarithmic search complexity
3. **Greedy Search**: Search proceeds greedily from top layer to bottom, maintaining a dynamic candidate list

## Mathematical Foundation

### Layer Assignment

The layer for each new element is chosen randomly according to:
```
level = floor(-ln(unif(0,1)) * ml)
```

Where:
- `ml` is the level generation factor (typically 1/ln(2))
- `unif(0,1)` is a uniform random number between 0 and 1

### Connection Strategy

Each node maintains connections to its M closest neighbors at each layer, where:
- M is the maximum number of connections per node
- Higher layers may have fewer connections due to sparsity
- Connections are bidirectional

## Implementation Architecture

### Core Data Structures

```go
type Index struct {
    config               *HNSWConfig
    nodes                []*Node
    entryPoint           *Node
    levelGenerator       *rand.Rand
    distance             DistanceFunc
    idToIndex            map[string]uint32
    entryPointCandidates []uint32
    quantizer            quant.Quantizer
    quantizationTrained  bool
    trainingVectors      [][]float32
    mu                   sync.RWMutex
}

type Node struct {
    ID       string
    Vector   []float32
    Level    int
    Links    [][]uint32  // Links for each level
    Metadata map[string]interface{}
}
```

### Configuration Parameters

```go
type HNSWConfig struct {
    Dimension      int
    M              int     // Max connections per node
    EfConstruction int     // Size of dynamic candidate list during construction
    EfSearch       int     // Size of dynamic candidate list during search
    ML             float64 // Level generation factor
    Metric         DistanceMetric
    RandomSeed     int64
    Quantization   *quant.QuantizationConfig
}
```

## Algorithm Implementation

### 1. Insertion Algorithm

The insertion process involves:
1. Generate level for new node
2. Create node structure
3. Handle quantization if enabled
4. Find entry point and insert into graph
5. Update connections at each level

### 2. Search Algorithm

The search process:
1. Start from entry point at highest layer
2. Search each layer greedily
3. Maintain dynamic candidate list
4. Return k closest candidates

### 3. Layer Search Implementation

Each layer search maintains:
- Visited set to avoid cycles
- Candidate heap for exploration
- Result heap for best candidates
- Dynamic pruning based on distance

## Optimization Strategies

### 1. Memory Layout Optimization

- Node packing for cache efficiency
- Link compression for memory savings
- Aligned data structures for SIMD operations

### 2. Distance Calculation Optimization

- SIMD vectorized operations
- Distance caching for frequent computations
- Optimized distance functions per metric type

### 3. Parallel Search

- Concurrent layer search across multiple entry points
- Worker pool for batch operations
- Lock-free data structures where possible

### 4. Dynamic Optimization

- Adaptive EfSearch based on performance
- Connection pruning for memory management
- Automatic quantization triggers

## Integration with Other Components

### 1. Quantization Integration

HNSW integrates with quantization by:
- Training quantizers on insertion data
- Storing both original and quantized vectors
- Using quantized vectors for distance calculations
- Falling back to original vectors when needed

### 2. Memory Management Integration

Memory management features:
- Memory pressure monitoring
- Automatic optimization triggers
- Memory mapping for large graphs
- Garbage collection coordination

### 3. Persistence Integration

Persistence capabilities:
- Graph serialization to disk
- Metadata preservation
- Incremental updates
- Recovery from corruption

## Performance Characteristics

### Time Complexity

- **Insertion**: O(log N * M * EfConstruction)
- **Search**: O(log N * EfSearch)
- **Memory**: O(N * M * log N)

### Space Complexity

**Without Quantization**:
```
Memory = N * (D * 4 + M * log(N) * 4 + overhead)
```

**With Quantization**:
```
Memory = N * (compressed_size + M * log(N) * 4 + overhead)
```

### Performance Tuning Guidelines

**For High Throughput Insertion**:
- Lower M (16-32)
- Lower EfConstruction (100-200)
- Enable quantization
- Use batch insertions

**For High Accuracy Search**:
- Higher M (32-64)
- Higher EfConstruction (200-800)
- Higher EfSearch (100-500)
- Disable quantization

**For Memory Efficiency**:
- Enable quantization
- Lower M
- Use memory mapping
- Implement connection pruning

## Testing and Validation

### Unit Tests
- Single insertion/search operations
- Parameter validation
- Error handling
- Edge cases

### Integration Tests
- Quantization integration
- Memory management
- Persistence operations
- Concurrent access

### Benchmark Tests
- Insertion throughput
- Search latency
- Memory usage
- Scalability limits

This HNSW implementation provides a robust, high-performance foundation for approximate nearest neighbor search in LibraVDB.