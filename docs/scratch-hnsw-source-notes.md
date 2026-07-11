# Scratch Notes: HNSW Source Research

This file is a working scratchpad for source-level observations from other
systems while tuning libraVDB HNSW throughput, recall, and p99 latency.

## Current libraVDB Context

- The HNSW fast path has already proven that the low-M insert path can be very
  fast, but high-M construction is currently catastrophic for throughput.
- M=16 is fast but under-connected on the 768d stress fixture.
- M=32 gets close to exact recall but still misses a small tail.
- M=36/M=40 can recover exact recall, but insertion throughput collapses.
- The normalized-dot rewrite was tested and lost to the existing squared-L2
  SIMD path. The theory was fine, but the measured implementation was worse.
- Local `A + A^T + A^2` repair did not recover recall in the current code. The
  measured result means local closure alone is not the missing topology source.
- The next credible wins must reduce scalar orchestration, pointer chasing,
  pruning cost, and heap/admission churn without simply raising M.

## Elasticsearch Findings

Source checkout:

`/private/tmp/libravdb-research-elasticsearch`

### 1. Bulk Sparse SIMD Scoring Is A First-Class Path

Elasticsearch exposes native vector scoring modes for:

- single vector score
- contiguous bulk score
- offset bulk score
- sparse-address bulk score

The important one for HNSW traversal is sparse-address bulk scoring. HNSW gives
random node IDs and random vector addresses; ES handles this by passing an array
of raw vector addresses to a native scorer rather than bouncing back through the
host language every few vectors.

Relevant source:

- `libs/native/src/main/java/org/elasticsearch/nativeaccess/VectorSimilarityFunctions.java`
- `Operation.BULK_SPARSE` documents an array of 8-byte native vector addresses,
  one query vector, count, and output score buffer.
- Native functions include:
  - `dotProductF32BulkSparse`
  - `squareDistanceF32BulkSparse`
  - `dotProductI8BulkSparse`
  - `squareDistanceI8BulkSparse`
  - `dotProductI4BulkSparse`
  - BBQ bulk sparse variants

ARM64 float32 implementation:

- `libs/simdvec/native/src/vec/c/aarch64/vec_f32_2.cpp`
- The float32 bulk scorer processes 8 vectors per bulk block.
- It uses separate accumulators per vector and writes scores to a result array.

Implication for libraVDB:

- Our per-4 `L2Distance4NEON -> Go admission -> L2Distance4NEON` shape is too
  chatty.
- We need a bulk sparse NEON API that accepts `[]uintptr` vector addresses,
  a query pointer, dimension, count, and `[]float32` scores.
- First target should be 8 vectors per assembly call because ES uses 8 on
  ARM64/SVE and it maps naturally to our current 4-wide path.

### 2. Prefetch Ring + Delayed Bulk Scoring

Elasticsearch direct rescore has a ring buffer:

- `PREFETCH_BUFFER_SIZE = 100`
- `BULK_SCORE_SIZE = 32`

Relevant source:

- `server/src/main/java/org/elasticsearch/search/vectors/RescoreKnnVectorQuery.java`

Flow:

1. Iterate candidate docs.
2. If an index input slice exists, call `input.prefetch(ord * vectorByteSize,
   vectorByteSize)`.
3. Append doc/scorer to a ring.
4. When the ring fills, score entries in bulk.
5. Reuse the bulk scorer for the same scorer.

Implication for libraVDB:

- Our scalar `vec[0]` touch is a demand-load, not a true prefetch pipeline.
- HNSW search should gather neighbor IDs/vector pointers into a ring, issue
  PRFM or equivalent prefetch, then score in larger batches.
- We should separate gather from scoring enough to create real lead distance.

### 3. Reservoir Top-K Instead Of Heap Churn

Elasticsearch has `BulkNeighborQueue`.

Relevant source:

- `server/src/main/java/org/elasticsearch/index/codec/vectors/cluster/BulkNeighborQueue.java`

Important details:

- For tiny `k <= 10`, it uses a `LongHeap`.
- For larger k, it uses `ReservoirTopK`.
- `ReservoirTopK` capacity is `2 * maxSize`.
- It keeps a threshold score.
- It does batch-level fast reject when the batch best score cannot beat the
  threshold.
- It inserts candidates with a branchless-ish pattern:

```java
values[size] = encoded;
int acceptedDelta = encoded > threshold ? 1 : 0;
size += acceptedDelta;
accepted += acceptedDelta;
```

- When capacity fills, it compacts/selects back down instead of performing
  heap maintenance on every candidate.

Implication for libraVDB:

- For `ef > 10`, a binary heap is likely the wrong hot admission structure.
- We should implement a reservoir top-K for search/result admission:
  - packed `uint64` distance/id or score/id
  - capacity `2 * ef`
  - threshold maintained by periodic select/partition
  - batch-level reject from the best score in a SIMD block
  - sorted output only at the boundary, not during traversal

### 4. Bulk Diversity Pruning

Elasticsearch HNSW utility code bulk-scores candidate pools and bulk-scores
kept neighbors during diversity pruning.

Relevant source:

- `server/src/main/java/org/elasticsearch/index/codec/vectors/HnswUtils.java`

Observed flow:

- For a node, copy candidate IDs into an array.
- `scorer.bulkScore(candidateNodes, candidateScores, numValid)`.
- Sort by score.
- During prune, set scorer to candidate and bulk-score all kept nodes into a
  scratch score array.
- Accept candidate only if it survives the diversity check.

Implication for libraVDB:

- Our high-M collapse is dominated by pruning/link repair math.
- The pruning path needs the same bulk sparse scorer as search.
- Instead of pairwise scalar candidate-vs-kept scoring, score kept nodes in
  blocks and reject as soon as any block occludes.
- This is more important for high-M construction than an 8-wide query search
  kernel alone.

### 5. Quantized Candidate Generation + Exact Rescore

Elasticsearch defaults to compressed candidate generation for high dimensions.

Relevant source:

- `server/src/main/java/org/elasticsearch/index/mapper/vectors/DenseVectorFieldMapper.java`

Important constants:

- `DEFAULT_OVERSAMPLE = 3.0`
- `OVERSAMPLE_LIMIT = 10_000`
- `BBQ_DIMS_DEFAULT_THRESHOLD = 384`

Observed behavior:

- For dims above the BBQ threshold, ES can default to BBQ HNSW with rescore.
- Query-time logic computes `adjustedK = ceil(k * oversample)`.
- It ensures `numCands >= adjustedK`.
- It runs approximate/quantized candidate generation, then exact reranks back to
  the requested k.

Implication for libraVDB:

- ES is not betting on perfect high-dimensional HNSW topology alone.
- It is using a coarse approximate stage plus exact rescore.
- For libraVDB, this can be either:
  - Matryoshka prefix coarse search plus full float32 rerank, or
  - quantized coarse graph plus full float32 rerank.
- This should be treated as a product feature, not a hack, but it should not
  replace fixing the baseline HNSW construction cost.

### 6. Adaptive Early Termination

Elasticsearch has an adaptive saturation collector.

Relevant source:

- `server/src/main/java/org/elasticsearch/search/vectors/AdaptiveHnswQueueSaturationCollector.java`

Observed behavior:

- Tracks newly collected neighbors per candidate.
- Computes a smoothed discovery rate.
- Maintains mean/stddev with Welford variance.
- Computes adaptive saturation threshold and patience.
- Early exits when frontier discovery stays saturated.

Implication for libraVDB:

- Fixed ef burns work on searches where the frontier has stopped improving.
- We should add telemetry first:
  - candidates expanded
  - candidates accepted
  - best/worst threshold movement
  - discovery rate per expansion
- Then implement adaptive stop only after we can prove it preserves recall.

## FalkorDB Findings

Source checkout:

`/private/tmp/libravdb-research-falkordb`

### 1. FalkorDB Vector Search Delegates To RediSearch/VecSim

Relevant source:

- `src/procedures/proc_vector_query.c`
- `src/index/index_vector_create.c`
- `src/index/index.c`
- `src/index/index_field.h`

Observed behavior:

- Vector KNN procedure creates a RediSearch results iterator:
  `RediSearch_GetResultsIterator(root, idx)`.
- FalkorDB iterates result IDs and loads graph entities.
- Vector index creation parses:
  - `dimension`
  - `similarityFunction`
  - `M`
  - `efConstruction`
  - `efRuntime`
- It then calls `RediSearch_VectorFieldSetHNSWParams`.

Defaults:

- `M = 16`
- `efConstruction = 200`
- `efRuntime = 10`

Implication for libraVDB:

- FalkorDB does not contain the HNSW wizardry. RediSearch/VecSim does.
- Comparing against FalkorDB vector performance means we are effectively
  comparing against RediSearch/VecSim for vector search.
- Next source review should be RediSearch and its VecSim dependency.

### 2. GraphBLAS Is For Graph Algebra, Not Vector HNSW

FalkorDB uses GraphBLAS for graph query execution and algebraic path expansion.

Relevant source:

- `src/graph/delta_matrix`
- `src/arithmetic/algebraic_expression`
- `deps/GraphBLAS`

Important pattern:

- A `Delta_Matrix` has:
  - synced base matrix
  - `delta_plus`
  - `delta_minus`
  - optional transpose
- Matrix multiply evaluates:

```text
A * (B.base + B.delta_plus) masked by B.delta_minus
```

This is visible in `src/graph/delta_matrix/delta_mxm.c`.

Implication for libraVDB:

- The useful GraphBLAS idea is not a local `A^2` repair pass.
- The useful idea is separating stable graph state from mutation deltas:
  - base adjacency in compact contiguous arrays
  - append-only edge deltas for recent inserts
  - deletion/tombstone deltas for pruned edges
  - background compaction into cache-friendly adjacency
- That may help pointer chasing and lock pressure, but it is a larger storage
  architecture change than a pruning tweak.

## What We Are Probably Missing

The source evidence points to these missing pieces:

1. Bulk sparse distance scoring over raw vector addresses.
2. Prefetch ring with real lead distance before scoring.
3. Reservoir top-K/bulk admission instead of per-candidate heap churn.
4. Bulk diversity pruning during construction and backlink repair.
5. Stable adjacency + delta adjacency model to reduce synchronous mutation
   pressure and pointer chasing.
6. Optional compressed/coarse candidate generation plus exact rerank.
7. Adaptive early termination based on measured frontier saturation.

## Proposed Next Implementation Order

1. Implement `BulkSparseL2NEON`.
   - Inputs: query pointer, vector pointer array, dimension, count, score buffer.
   - First version: 8 vector pointers per assembly block.
   - Reuse existing 64-byte alignment assumptions.

2. Replace search scoring loop with gather/prefetch/bulk-score/bulk-admit.
   - Gather neighbor IDs and vector pointers into scratch.
   - PRFM or otherwise prefetch vector addresses.
   - Score batches of 16 or 32 through bulk sparse NEON.
   - Admit scores through a reservoir collector.

3. Implement reservoir top-K for HNSW admission.
   - Keep heap for tiny k only if measured faster.
   - For ef-sized queues, use `2 * ef` reservoir and compact on capacity.

4. Apply bulk sparse scoring to diversity pruning.
   - Candidate-vs-kept pruning should score kept neighbors in blocks.
   - Stop early when any block proves occlusion.

5. Revisit topology after scalar overhead is removed.
   - Re-test M=24/M=32 with the same recall harness.
   - Do not raise M as the first answer.

6. Apply the RediSearch/VecSim lessons where they fit.
   - The core VecSim HNSW loop is not radically more advanced than ours.
   - The production wins are around tiering, layout, visited tags, prefetch,
     hybrid query policy, and quantized SIMD.

## RediSearch / VecSim Findings

Source checkout:

`/private/tmp/libravdb-research-redisearch`

VecSim dependency:

`/private/tmp/libravdb-research-redisearch/deps/VectorSimilarity`

### 1. RediSearch Uses VecSim As The Vector Engine

Relevant source:

- `src/vector_index.c`
- `src/document.c`
- `src/iterators/hybrid_reader.c`
- `deps/VectorSimilarity/src/VecSim`

Observed behavior:

- RediSearch wraps VecSim indexes and calls `VecSimIndex_AddVector` during
  document indexing.
- Standard KNN queries call `VecSimIndex_TopKQuery`.
- Filtered/hybrid queries can use:
  - normal HNSW top-k search
  - VecSim batch iterators
  - ad-hoc brute force over the filtered subset
- The hybrid reader can switch policy based on estimated filter selectivity and
  vector index size.

Implication for libraVDB:

- FalkorDB vector performance is effectively RediSearch/VecSim performance.
- RediSearch is not relying on GraphBLAS for vector search.
- The interesting production behavior is the routing between ANN and brute-force
  exact scoring when filters make brute force cheaper.

### 2. Tiered HNSW Is A Real Async Write Path

Relevant source:

- `src/vector_index.c`
- `deps/VectorSimilarity/src/VecSim/algorithms/hnsw/hnsw_tiered.h`

Observed behavior:

- RediSearch initializes tiered params with:
  - `primaryIndexParams`
  - a worker thread pool job queue
  - `flatBufferLimit`
  - a submit callback
- VecSim tiered HNSW insert flow:
  1. Add the vector to a flat frontend buffer when the buffer has capacity.
  2. Create an `HNSWInsertJob`.
  3. Submit the job to the background queue.
  4. The background job copies the vector out of the flat buffer.
  5. It inserts into the real HNSW index outside the flat-buffer lock.
  6. It removes the vector/job from the frontend buffer after indexing.
- If the flat buffer is full or write mode is in-place, it inserts directly into
  HNSW.

Implication for libraVDB:

- This is the clearest RediSearch answer to "HNSW is slower than ingestion."
- The production design is not "make synchronous HNSW inserts free"; it is:
  - absorb writes into a cheap flat frontend
  - search both frontend and HNSW when needed
  - asynchronously drain into HNSW
- This is very relevant once WAL and daemon write latency become the wall.
- For the current HNSW-only work, it argues for keeping live insert cheap and
  pushing heavy topology repair/compaction off the request path.

### 3. VecSim Defaults Are Conservative HNSW, Not Magic

Relevant source:

- `deps/VectorSimilarity/src/VecSim/vec_sim_common.h`
- `deps/VectorSimilarity/src/VecSim/algorithms/hnsw/hnsw.h`

Defaults:

- `DEFAULT_BLOCK_SIZE = 1024`
- `HNSW_DEFAULT_M = 16`
- `HNSW_DEFAULT_EF_C = 200`
- `HNSW_DEFAULT_EF_RT = 10`
- `HNSW_DEFAULT_EPSILON = 0.01`

Observed behavior:

- `M0 = 2 * M` for the base layer.
- `efConstruction = max(user_efConstruction, M)`.
- `efRuntime` is the default search beam, but callers can override at query
  time.
- VecSim also exposes an SVS/Vamana path separately from HNSW.

SVS/Vamana defaults worth noting:

- `alpha_l2 = 1.2`
- `alpha_ip = 0.95`
- `graph_max_degree = 32`
- `construction_window_size = 200`
- `search_window_size = 10`
- `use_search_history = true`

Implication for libraVDB:

- VecSim's default HNSW is not proving that M=16 solves every high-dimensional
  case.
- Their Vamana path explicitly uses alpha and max degree 32, which matches our
  conclusion that topology quality needs alpha/degree/window tuning.
- If we want alpha-style topology improvements, Vamana-style construction may
  be the more honest target than forcing all of it into HNSW's existing prune.

### 4. Graph Storage Uses Blocks And Inline Adjacency

Relevant source:

- `deps/VectorSimilarity/src/VecSim/containers/data_blocks_container.h`
- `deps/VectorSimilarity/src/VecSim/containers/data_block.h`
- `deps/VectorSimilarity/src/VecSim/algorithms/hnsw/graph_data.h`

Observed vector storage:

- Vectors are stored in fixed-size blocks.
- `getElement(id)` maps:

```text
block = blocks[id / block_size]
slot  = id % block_size
ptr   = block.data + slot * element_size
```

- This is the same broad class as our segmented/off-heap storage.

Observed graph storage:

- `ElementGraphData` stores:
  - `toplevel`
  - `neighborsGuard`
  - upper-level data pointer
  - inline `level0`
- `ElementLevelData` stores:
  - incoming unidirectional edge pointer vector
  - explicit `numLinks`
  - flexible array member `links[]`
- Links are contiguous IDs, not per-edge allocations.

Implication for libraVDB:

- Our segmented off-heap vector store is directionally correct.
- The specific thing to copy is explicit per-level link counts and inline or
  contiguous link arrays.
- Sentinel scanning should not exist in hot traversal.
- Direct vector pointer access from node metadata remains important, because
  repeated registry lookup is avoidable pointer chasing.

### 5. Visited State Uses Tags, Not Clearing

Relevant source:

- `deps/VectorSimilarity/src/VecSim/algorithms/hnsw/visited_nodes_handler.cpp`

Observed behavior:

- Each visited array entry stores a tag.
- A search gets a fresh tag by incrementing `cur_tag`.
- Mark visited by writing the current tag.
- Test visited by comparing stored tag with current tag.
- The whole array is only reset on tag overflow.
- Handlers are pooled for concurrent searches.

Implication for libraVDB:

- If our visited path clears arrays, bitsets, or maps per search, that is wasted
  memory bandwidth.
- Use generation tags for dense IDs:
  - `[]uint32 visitedTags`
  - `uint32 currentTag`
  - reset only on wrap
- This is simple, exact, and SIMD-friendly because it turns visited into dense
  integer loads/stores.

### 6. Search Traversal Is Conventional But Prefetches

Relevant source:

- `deps/VectorSimilarity/src/VecSim/algorithms/hnsw/hnsw.h`
- `deps/VectorSimilarity/src/VecSim/algorithms/hnsw/hnsw_batch_iterator.h`

Observed behavior:

- Search uses hnswlib-style candidate and result heaps.
- Per candidate:
  - lock/read links
  - get explicit `numLinks`
  - prefetch first visited tag and vector data
  - for each neighbor, prefetch the next neighbor's tag and vector data
  - skip visited and in-process nodes
  - compute distance
  - push into candidate/result heaps when it passes threshold
- Batch iterator uses the same style: scalar distances with prefetch, not a
  sparse bulk scorer like Elasticsearch.

Implication for libraVDB:

- VecSim is not beating scalar orchestration by radically changing HNSW search.
- It does validate two practical changes:
  - prefetch tags and vector data ahead of use
  - avoid clearing visited state
- Elasticsearch remains the stronger source for bulk sparse scoring and
  reservoir admission.

### 7. In-Process Flags Keep Searches Away From Half-Built Nodes

Relevant source:

- `deps/VectorSimilarity/src/VecSim/algorithms/hnsw/hnsw.h`

Observed behavior:

- Element metadata has flags:
  - `DELETE_MARK`
  - `IN_PROCESS`
- New elements start as `IN_PROCESS`.
- Search skips in-process nodes.
- Insert unmarks `IN_PROCESS` only after graph insertion completes.
- Flags are updated with atomic bit operations.

Implication for libraVDB:

- This is the conservative alternative to exposing in-flight nodes during
  construction.
- For maximum recall under concurrent construction, in-flight snapshots are
  still attractive.
- For correctness and avoiding torn reads, an explicit `IN_PROCESS` flag is
  still useful even if we later expose in-flight candidates to insertion-only
  searches.

### 8. Backlinks Use Ordered Locks And Unidirectional Edge Tracking

Relevant source:

- `deps/VectorSimilarity/src/VecSim/algorithms/hnsw/hnsw.h`
- `deps/VectorSimilarity/src/VecSim/algorithms/hnsw/graph_data.h`

Observed behavior:

- `mutuallyConnectNewElement` connects a new node to selected neighbors.
- When locking two nodes, it locks in node-ID order to prevent deadlock.
- If a neighbor is full, `revisitNeighborConnections` prunes the neighbor's
  connections.
- The code may temporarily release locks around pruning work to avoid holding
  multiple node locks across expensive logic.
- It tracks incoming unidirectional edges when a reciprocal edge is not retained.

Implication for libraVDB:

- Ordered node locking is a proven baseline if we keep locks in the mutation
  path.
- Incoming unidirectional edges are worth serious consideration. They preserve
  reachability information that strict bidirectional capacity can otherwise
  discard.
- A delta-adjacency model could represent these unidirectional or pending edges
  cheaply before compaction.

### 9. Pruning Is Standard HNSW Heuristic, Scalar Pairwise

Relevant source:

- `deps/VectorSimilarity/src/VecSim/algorithms/hnsw/hnsw.h`

Observed behavior:

- `getNeighborsByHeuristic2_internal`:
  1. Sort candidates by distance to the query/node.
  2. Cache vectors for selected candidates.
  3. For each candidate, compute distance to each already selected neighbor.
  4. Reject if an already selected neighbor is closer to the candidate than the
     query/node is.
- This is standard HNSW diversity pruning.
- It is not bulk SIMD.

Implication for libraVDB:

- VecSim does not solve the high-M construction collapse in HNSW prune itself.
- Elasticsearch's bulk pruning pattern is still the better lead for our
  construction bottleneck.
- Our own high-M work should focus on:
  - fewer prune events
  - bulk candidate-vs-kept scoring
  - deferred backlink pruning
  - alpha/Vamana-style reachability if we can validate it

### 10. Repair Jobs Exist, But For Deletes

Relevant source:

- `deps/VectorSimilarity/src/VecSim/algorithms/hnsw/hnsw_tiered.h`
- `deps/VectorSimilarity/src/VecSim/algorithms/hnsw/hnsw.h`

Observed behavior:

- Tiered index has `HNSWRepairJob`.
- `repairNodeConnections(node_id, level)`:
  - finds deleted neighbors
  - collects current non-deleted neighbors
  - adds neighbors of deleted neighbors as candidates
  - prunes back to max degree when needed
  - mutually updates affected nodes
- The implementation uses bitmap/set style scratch structures.

Implication for libraVDB:

- VecSim repair is a deletion-repair mechanism, not a general graph-quality
  optimizer.
- Our failed `A + A^T + A^2` local repair result is consistent with this:
  neighbor-of-neighbor repair helps when a deleted edge created a hole, but it
  does not magically repair under-connected high-dimensional topology.

### 11. Hybrid Query Policy Can Avoid HNSW When Filters Are Tight

Relevant source:

- `src/iterators/hybrid_reader.c`
- `deps/VectorSimilarity/src/VecSim/algorithms/hnsw/hnsw.h`

Observed behavior:

- RediSearch hybrid iterator computes vector results in batches and joins them
  with another filter iterator.
- It can switch from batch vector search to ad-hoc brute force.
- VecSim contains a hard-coded decision tree in `preferAdHocSearch`.
- Features include:
  - index size
  - vector dimension
  - M
  - filtered subset ratio
  - k

Implication for libraVDB:

- A production vector DB should not always use HNSW.
- If a metadata filter leaves a small candidate subset, exact SIMD brute force
  is often faster and more stable than graph traversal.
- This is an important future p99 reducer for filtered workloads.

### 12. SIMD Kernels Confirm Float32 L2 Is Not The Weak Link

Relevant source:

- `deps/VectorSimilarity/src/VecSim/spaces/L2/L2_NEON_FP32.h`
- `deps/VectorSimilarity/src/VecSim/spaces/IP/IP_NEON_FP32.h`
- `deps/VectorSimilarity/src/VecSim/spaces/L2/L2_NEON_DOTPROD_UINT8.h`
- `deps/VectorSimilarity/src/VecSim/spaces/IP/IP_NEON_DOTPROD_SQ8_SQ8.h`

Observed behavior:

- Float32 L2 NEON uses 4 accumulators over 16 floats per loop:
  - `vld1q_f32`
  - `vsubq_f32`
  - `vmlaq_f32`
  - horizontal add
- Float32 IP NEON uses the same broad structure.
- Uint8 L2 uses ARM dot-product instructions and processes many more elements
  per loop.
- SQ8 IP stores metadata after quantized bytes:
  - min
  - delta
  - sum
- SQ8 IP uses a formula that reconstructs the affine quantized dot product from
  integer dot products and the stored metadata.

Implication for libraVDB:

- Our measured result that squared-L2 NEON beat the normalized-dot rewrite is
  credible. VecSim's float32 kernels are not obviously more exotic.
- Quantized uint8/SQ8 is where VecSim gets a different hardware profile:
  smaller memory footprint, dot-product instructions, and more elements per
  load.
- For the immediate HNSW target, bulk float32 sparse scoring is lower risk than
  introducing quantization.
- For larger-than-RAM or extreme memory-bandwidth pressure, SQ8/uint8 with
  exact rerank is the credible next feature.

## RediSearch / VecSim Takeaway

VecSim HNSW is not magic. Its core HNSW search and prune logic is mostly
conventional:

- hnswlib-style heaps
- standard diversity pruning
- immediate backlink revisiting
- scalar per-neighbor distance calls

The production engineering around it is the important part:

1. Tiered async writes: flat frontend buffer plus background HNSW insert jobs.
2. Stable block storage for vectors and graph metadata.
3. Inline contiguous adjacency with explicit link counts.
4. Visited generation tags instead of per-search clearing.
5. In-process flags for correctness around concurrent inserts.
6. Ordered per-node locks and unidirectional incoming edge tracking.
7. Prefetch during scalar graph traversal.
8. Hybrid query policy that switches to brute force for small filtered subsets.
9. Quantized SIMD kernels for memory-bound regimes.
10. Separate SVS/Vamana path for alpha-style graph construction.

For libraVDB, this suggests:

- Do not expect RediSearch HNSW prune/search code to solve high-M collapse
  directly.
- Copy the layout and write-path lessons.
- Use Elasticsearch's bulk sparse scorer and reservoir admission ideas for the
  current hot loop.
- Keep Vamana/SVS alpha construction as a serious branch if HNSW topology cannot
  hit the recall/throughput target at M=24 or M=32.

## Concrete Next Targets After This Source Pass

1. Add explicit visited generation tags if we are still clearing visited state.
2. Ensure every level has explicit link counts; remove sentinel scans.
3. Add direct vector pointers to node metadata for traversal/prune hot paths.
4. Implement `BulkSparseL2NEON` for 8-vector ARM64 scoring.
5. Convert search gather to:
   - gather IDs/pointers
   - prefetch tags/vectors
   - bulk score
   - bulk admit
6. Replace ef-sized result heap admission with a reservoir collector.
7. Apply bulk sparse scoring to diversity pruning.
8. Revisit deferred backlink pruning or delta adjacency for high-M construction.
9. Treat tiered flat-buffer + async HNSW insert as the future write-path answer
   once the synchronous HNSW core is healthier.
10. Re-open RediSearch/VecSim later specifically for:
    - SQ8 metadata layout
    - SVS/Vamana alpha graph construction
    - tiered query merge behavior between flat frontend and HNSW backend
