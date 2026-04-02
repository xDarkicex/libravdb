# Vector Store Redesign for HNSW

This document proposes a storage-layout redesign for LibraVDB's HNSW index so
the implementation can move beyond micro-optimizations and address the
structural memory and insertion costs that remain after the recent scratch,
sorting, and WAL improvements.

The primary goal is to preserve current HNSW correctness while changing how
vector payloads are owned and accessed.

## Why This Exists

The current HNSW implementation stores vectors directly on each node:

```go
type Node struct {
    ID               string
    Index            uint32
    Vector           []float32
    CompressedVector []byte
    Level            int
    Links            [][]uint32
    Metadata         map[string]interface{}
}
```

This layout is simple, but it makes every insertion pay for:

- a per-node `[]float32` allocation
- a per-node backing array allocation for the vector payload
- scattered heap ownership for vectors, which hurts locality and makes memory
  mapping harder to reason about

Recent profiling confirmed that a meaningful part of the remaining insertion
cost is now structural retained-state allocation, not temporary search
machinery.

## Design Goals

The redesign must:

- preserve HNSW graph semantics and search correctness
- preserve deterministic fixed-seed behavior
- preserve quantized-distance behavior
- make vector ownership explicit instead of embedding raw payloads in each node
- allow collection-level storage strategies instead of one storage model for all
  collections
- support future memory mapping and persistence improvements cleanly

## Non-Goals

This redesign does not attempt to:

- change the HNSW graph algorithm itself
- replace metadata storage
- replace link storage in the first phase
- make all memory lock-free

## Core Idea

Introduce a collection-owned `VectorStore` abstraction and make HNSW nodes refer
to vectors by handle/offset instead of owning `[]float32` directly.

New conceptual shape:

```go
type Node struct {
    ID         string
    Index      uint32
    Level      int
    Links      [][]uint32
    Metadata   map[string]interface{}
    VectorRef  VectorRef
}
```

The vector payload moves behind a store:

```go
type VectorRef struct {
    Kind   VectorEncoding
    Slot   uint32
    Length uint32
}

type VectorStore interface {
    PutRaw(vec []float32) (VectorRef, error)
    PutCompressed(buf []byte) (VectorRef, error)

    GetRaw(ref VectorRef) ([]float32, error)
    GetCompressed(ref VectorRef) ([]byte, error)

    Delete(ref VectorRef) error
    Reset() error

    MemoryUsage() int64
}
```

## Recommended Storage Model

### Phase 1: Fixed-Dimension Raw Vector Store

For unquantized collections, vectors have a fixed dimension per collection. That
means each vector has a fixed byte size:

```text
raw_vector_bytes = dimension * 4
```

That makes raw-vector storage a good fit for a fixed-size allocator.

Recommended implementation:

- one vector store per collection/index
- slab/page size = `dimension * 4`
- node stores a slot/handle instead of `[]float32`
- query-time search uses a temporary query slice as today
- node-to-node or query-to-node distance resolves vectors from the store

This keeps the HNSW graph structure unchanged while moving vector ownership out
of node objects.

### Phase 2: Quantized Vector Store

For quantized collections, compressed vectors are also byte-oriented and fit the
same abstraction even better:

- compressed payloads remain opaque byte slices
- store can use fixed slabs when compressed size is fixed
- or size-class stores when compressed payload size varies

### Phase 3: Memory-Mappable Vector Store

Once vector ownership is centralized, memory mapping becomes much cleaner:

- raw vector pages can be persisted contiguously
- quantized payloads can be persisted contiguously
- node records only need references
- rebuilding from disk becomes pointer/offset reconstruction rather than
  scattered slice rebuilding

## Why `slabby` Fits

`slabby` is a good fit if used behind `VectorStore`, not directly throughout
HNSW internals.

Good fit:

- fixed-dimension raw vector storage
- fixed-size compressed payload storage
- scratch work buffers

Bad fit:

- variable-size metadata maps
- mixed node object graphs
- APIs that want arbitrary long-lived Go-owned slices everywhere

The right use is:

- create one `slabby` allocator per collection/size-class
- store vectors in slabs
- keep only a stable handle/slot in the node
- decode or reinterpret bytes only at the `VectorStore` boundary

## Algorithmic Invariants

The redesign must preserve these invariants:

### 1. Stable Node Identity

HNSW links reference node indices, not vector memory addresses.
Changing vector ownership must not change:

- `Node.Index`
- `h.nodes[]` index semantics
- link encoding

### 2. Stable Vector Reads

Distance functions must observe the same vector values they do today.

For raw vectors:

```text
distance(query, node) == distance(query, vector_store.GetRaw(node.VectorRef))
```

For quantized vectors:

```text
quantizer.Distance(nodeA, nodeB)
quantizer.DistanceToQuery(node, query)
```

must still use the same compressed payload bytes as before.

### 3. No Borrowed Scratch Escapes

Any temporary conversion from slab-backed bytes to `[]float32` must remain
internal to the store or the immediate call site. HNSW search should not retain
temporary borrowed views in long-lived node state.

### 4. Persistence Must Be Explicit

Persisted HNSW format currently writes vectors per node. After the redesign,
persistence must deliberately choose one of:

- node records inline vector payloads reconstructed from the store
- a separate vector-data section plus node refs

The second option is preferred long term, but the first can be used as a
compatibility bridge during migration.

## Proposed Interface Surface

### Node storage contract

```go
type VectorEncoding uint8

const (
    VectorEncodingRaw VectorEncoding = iota
    VectorEncodingCompressed
)

type VectorRef struct {
    Kind  VectorEncoding
    Slot  uint32
    Bytes uint32
}
```

### HNSW-side vector resolution

```go
func (h *Index) getNodeVector(node *Node) ([]float32, error)
func (h *Index) getNodeCompressed(node *Node) ([]byte, error)
```

These become store-backed instead of field-backed.

## Concrete `slabby` Integration Shape

Recommended first backend:

```go
type SlabbyRawVectorStore struct {
    dim       int
    bytesPer  int
    alloc     *slabby.HandleAllocator
}
```

Storage rule:

- allocator slab size = `dim * 4`
- one handle per stored raw vector
- `VectorRef.Slot` maps to slab handle ID or an internal slot table

Two practical implementation strategies:

### Strategy A: Handle Table

- node stores a compact store slot
- store owns a table mapping slot -> slabby handle metadata
- good if handles/generations must stay hidden from HNSW

### Strategy B: Direct slab ID reference

- node stores slab ID directly
- store validates access and ownership
- simpler and faster, but exposes more allocator shape into persistence logic

Strategy A is cleaner for long-term maintainability.

## Risk Areas

### 1. `[]byte` to `[]float32` views

If raw vectors are stored in slab bytes, conversion must respect:

- alignment
- endianness
- lifetime

The safest first version is:

- copy bytes into a temporary `[]float32` on read for raw mode

That gives correctness but not the full performance win.

The higher-performance version is:

- enforce alignment in the store
- use an internal unsafe view helper
- never let the borrowed view escape the immediate distance call

### 2. Quantizer interactions

Current quantized flow expects `CompressedVector []byte` on the node. Moving
that behind a store means:

- quantized distance helpers must accept store-backed payload lookup
- persistence and memory mapping must serialize compressed payloads correctly

### 3. Delete/reuse semantics

If slots are reused, references must not silently alias new vectors.

That means either:

- generation-counted refs
- or monotonic slot allocation until compaction/rebuild

For correctness and simplicity, the first phase should prefer monotonic slot
allocation with optional free-list reuse later.

## Migration Plan

### Phase 0: Abstraction only

- add `VectorRef`
- add `VectorStore` interface
- keep existing node fields as compatibility fields
- route `getNodeVector` / quantized access through helper methods

### Phase 1: Raw vector store backend

- implement `InMemoryRawVectorStore`
- optional second backend: `SlabbyRawVectorStore`
- change new-node creation to store raw vectors through the store
- stop storing `node.Vector` directly for raw mode

### Phase 2: Quantized vector store backend

- move compressed payload ownership into the store too
- change quantized lookup paths accordingly

### Phase 3: Persistence format update

- add vector-store-aware persistence sections
- preserve backward-compatible load path if needed

## Recommended First Implementation

Build the abstraction in this order:

1. Introduce `VectorRef` and `VectorStore`
2. Implement a simple `InMemoryRawVectorStore`
3. Switch raw HNSW node storage to `VectorRef`
4. Keep quantized payloads on-node temporarily
5. Validate correctness and benchmarks
6. Add `SlabbyRawVectorStore`
7. Compare:
   - insertion allocs/op
   - bytes/op
   - p95 search latency
   - memory-mapped behavior

This minimizes risk while creating the seam needed to use `slabby` correctly.

## Decision

We should use `slabby` if we first create a `VectorStore` abstraction.

We should not wire `slabby` directly into the current `node.Vector []float32`
shape.

The right architectural path is:

- `VectorStore` abstraction first
- simple in-memory backend first
- `slabby` backend second
- persistence update after the storage seam is proven

This gives LibraVDB a realistic path toward the README-level performance and
memory claims without coupling HNSW internals directly to one allocator API.
