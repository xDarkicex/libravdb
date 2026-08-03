Hybrid Multi-Modal Search Execution: A Mathematical and Empirical Analysis of Execution Orders
Executive Summary and Verdict
The architecture of modern hybrid database engines requires the seamless integration of relational algebra, property-graph traversal, and high-dimensional vector similarity search. In distributed systems, query execution order is historically dictated by the prohibitive serialization and network costs associated with passing intermediate identifiers between disparate subsystems. However, within a unified, single-process engine operating under a single memory space, these serialization costs are eliminated. This fundamentally isolates the execution order decision to pure algorithmic complexity, memory locality, and mathematical probability.

The central claim under investigation posits that an optimizer should evaluate relational and graph predicates first to generate a small candidate set, computing vector similarity exclusively on this pruned subset to guarantee zero recall loss. Rigorous mathematical modeling, hardware cost analysis, and empirical evidence from production systems demonstrate that this claim conflates the absolute accuracy of exact distance calculation with the structural fragility of approximate graph traversal.

The definitive verdict is that there is no mathematically superior, fixed execution order for hybrid multi-modal search. Instead, filtered Approximate Nearest Neighbor (ANN) search operates as a highly sensitive phase-transition system. The optimal execution strategy is governed by predicate selectivity, the dimensionality of the vector space, graph degree, and hardware cache mechanics.   

The empirical truths governing these systems are as follows:
First, the pre-filtering (prune-first) strategy combined with exact distance calculation guarantees absolute recall. However, its computational viability is strictly bounded to regimes of high selectivity—typically when the predicates filter out more than 95% of the corpus—where sequential Single Instruction, Multiple Data (SIMD) execution outpaces the random memory access penalties of graph pointer-chasing.
Second, applying the pre-filtering strategy to an unmodified approximate graph index (such as Hierarchical Navigable Small World, or HNSW) induces a catastrophic failure known as graph shattering. Below a mathematically defined selectivity threshold, the small-world navigability of the graph is destroyed, rendering the search void and causing severe recall degradation.
Third, post-filtering (rank-first) with statistical oversampling is mathematically optimal for relaxed filters characterized by low selectivity. However, this strategy experiences a violent recall cliff governed by order statistics when the selectivity exceeds the oversampling ratio, a failure mode further amplified by vector quantization techniques that compress distance distributions.   

Consequently, a single-pass execution plan is mathematically suboptimal. The superior architecture is an adaptive, cost-based dispatcher that dynamically transitions between Post-Filtering with Oversampling, Traversal-Time Filtering (In-Filtering) utilizing adaptive beam expansion, and Exact Pre-Filtering, driven by real-time catalog statistics and selectivity estimation.

Formalizing the Trade-Off: The Phase Transition of Filtered Vector Search
To rigorously evaluate the superiority of execution orders, the execution landscape must be formalized not as a continuous spectrum, but as a phase-transition system. Given a query vector q, a combined attribute predicate P (representing the union of relational WHERE constraints and graph MATCH reachability), and a corpus of N vectors, the objective is to retrieve the k vectors nearest to q among the subset X 
P

 ={x∈X:P(x)}. The selectivity of the predicate is defined as s=∣X 
P

 ∣/N, where s∈[0,1].

Recent analytical models of filtered ANN strategy selection demonstrate that the optimal execution strategy changes abruptly at specific critical boundaries. Plan regret—defined as the recall lost by selecting a suboptimal execution plan compared to an oracle strategy—spikes exponentially within the critical regions surrounding these phase boundaries.   

The Mathematics of Post-Filtering and Oversampling
In a rank-first (post-filtering) execution strategy, the database engine queries the unfiltered ANN index to retrieve a broad candidate set of size K (where K>k). The predicate P is subsequently applied to this retrieved set, and the top k vectors that satisfy the condition are returned to the user. The oversampling factor is formally defined as α=K/k.

Assuming that the predicate P is statistically independent of the geometric distribution of the vectors in the latent semantic space, the probability that any retrieved vector satisfies the predicate is exactly s. Consequently, the number of valid vectors E retrieved within the unfiltered top-K set follows a binomial distribution, expressed as E∼Binomial(K,s).   

For the post-filtering strategy to successfully yield k results without suffering recall loss, the condition E≥k must be met. The probabilistic recall guarantee, representing the likelihood of success, is derived from the cumulative distribution function of the binomial distribution:

P(E≥k)=1− 
i=0
∑
k−1

 ( 
i
K

 )s 
i
 (1−s) 
K−i
 
Setting the oversampling parameter to the naive expected value of K=k/s yields an expected number of valid points equal to k. However, due to inherent statistical variance, retrieving exactly k/s candidates provides only a roughly 50% probability of fully satisfying the required k results. To achieve a stringent 99% probabilistic recall guarantee, the oversampling factor K must be elevated to incorporate a confidence interval derived from the distribution's variance, Ks(1−s).

The Post-Filter Cliff and Order Statistics
The mathematical failure of the post-filtering strategy manifests at a distinct phase boundary located at s≈k/K. Above this critical selectivity threshold—for instance, when a filter retains 50% of the corpus (s=0.5)—post-filtering is computationally inexpensive and consistently yields near-perfect recall. The engine effortlessly oversamples by a small factor, filters the results, and returns the top k.   

However, as the selectivity s drops below the k/K threshold, the probability of successfully locating k valid items collapses precipitously. This phase boundary acts as a steep local cliff. If the query optimizer overestimates the true selectivity of a predicate within this critical region, it will select a post-filtering plan that fails to retrieve enough candidates, resulting in massive plan regret.   

Failure Modes of Statistical Amplification
The binomial guarantee underlying post-filtering is mathematically sound only under assumptions of statistical independence and uniform distance distributions. In production environments, this guarantee provably fails under specific adversarial conditions.

The most prominent failure mode is the presence of correlated filters, also known as adversarial selectivity. If the predicate P is negatively correlated with the query's vector similarity, the valid vectors reside far from the query vector q in the high-dimensional space. For example, if a user issues a semantic search for "luxury automated vehicles" but applies a strict relational filter for price < 5000, the vectors representing budget vehicles will be geometrically distant from the query vector. In such scenarios, the target vectors may not exist in the top K, or even the top 100K, of the unfiltered space. Amplification provably fails because the local selectivity s within the query's immediate geometric neighborhood effectively drops to zero, rendering global statistical models useless.

A secondary failure mode involves score distribution clustering, a phenomenon exacerbated by the curse of dimensionality. In highly dimensional latent spaces, distance metrics frequently exhibit magnitude heterogeneity, resulting in tightly clustered similarity scores where thousands of vectors may be nearly equidistant from the query. When distances are this condensed, retrieving a massive candidate set K to satisfy a low selectivity s introduces severe topological noise. The top-K set becomes flooded with near-ties that fail the relational predicate, exhausting the computational retrieval budget long before k valid matches are discovered.   

Filtered ANN Graph Search: The In-Filter Regime
To circumvent the order-statistics cliff inherent in post-filtering, systems often attempt to push the relational and graph predicates directly into the graph traversal phase. In this in-filtering, or traversal-time filtering strategy, the engine navigates the HNSW or DiskANN graph while evaluating the predicate P at each node—often via an in-memory bitmask generated by the relational sub-engine—placing only valid nodes into the priority queue for subsequent expansion.

Graph Shattering and Site Percolation
The architectural claim that pre-filtering guarantees zero recall loss is catastrophically false when applied to an unmodified approximate graph index. Graph-based ANN indexes rely fundamentally on the small-world property, defined as the ability to greedily route a query from a random entry point to a target neighborhood via short, highly connected paths.   

When a strict relational bitmask—for instance, one with a selectivity of s=0.02—is statically overlaid onto the graph during query time, the traversal engine is mathematically forced to ignore 98% of the available routing nodes. This aggressive masking induces a physical phase transition known as site percolation.   

Independent mathematical models from statistical mechanics place the critical percolation threshold for an ANN graph at approximately s 
c

 ≈0.83/M, where M represents the graph's maximum degree or connectivity parameter. If the selectivity s drops below this critical threshold s 
c

 , the graph physically shatters into isolated, disconnected subgraphs. The greedy search algorithm inevitably encounters navigational dead ends, trapping the traversal in local minima because the highly connected bridge nodes required to cross the latent space have been masked out by the filter.   

Under these conditions, the search halts prematurely. Engineers maintaining Apache Lucene and Elasticsearch have extensively documented this phenomenon, noting that filtering out valid navigational nodes forces the graph exploration to terminate abruptly, leading to unacceptably poor recall. Consequently, naive pre-filtering on a sparse graph breaks connectivity and actively prevents the discovery of valid nearest neighbors.   

Selectivity (s)	Graph Condition	Traversal Consequence	Optimal Strategy
s>0.8	Fully Connected	Efficient routing, minimal overhead	Post-Filter (Rank-First)
0.1<s<0.8	Weakly Connected	Longer pathing, increased distance computes	In-Filter with Adaptive Expansion
s<0.83/M	Shattered (Percolation)	Navigational dead ends, near-zero recall	Exact Scan (Prune-First)
Advanced Solutions: Traversal and Structural Adaptations
To resolve the graph shattering problem, contemporary academic literature and production engines alter either the traversal algorithm at query time or the graph structure during index construction.

Traversal adaptations, most notably the ACORN (Performant and Predicate-Agnostic Search) algorithm, modify the standard HNSW search logic to allow the engine to route through invalid nodes without adding them to the final result set. By evaluating predicates dynamically during the walk, ACORN monitors the density of the subgraph. If it detects a high concentration of filtered-out nodes, it dynamically expands its neighborhood exploration bounds. Modern implementations, such as the ACORN-1 variant recently integrated into Apache Lucene and Vespa, selectively explore three-hop neighborhoods if the standard two-hop neighborhood yields an insufficient number of valid candidates. While this mechanism successfully maintains navigability and restores recall, evaluating the predicate at every node and loading invalid vectors into memory introduces significant branch-prediction penalties and instruction-cache overhead, elevating query latency.   

Structural adaptations attempt to solve the problem during the index build phase. Filtered-DiskANN introduces algorithms such as StitchedVamana and FilteredVamana. These approaches utilize a specific FilteredRobustPrune mechanism during graph construction to guarantee that sufficient edge connectivity exists exclusively among nodes that share identical categorical labels. StitchedVamana conceptually constructs distinct, dense subgraphs for different predicate tags and stitches them together into a unified index. While this structural approach delivers excellent recall and query-time performance for highly frequent, single-dimensional tags, it faces severe limitations in the context of a unified SQL engine. StitchedVamana incurs massive memory overhead and build-time latency when confronted with arbitrary, high-cardinality, or complex compound relational predicates, making it largely unviable for dynamic multi-join queries.   

Similarly, approaches like the Unified Dominance Graph (UDG) attempt to solve this for specific interval filters by mapping object and query endpoints into a normalized two-dimensional dominance space, effectively compressing query-state-specific proximity graphs. Yet, these structural adaptations remain tightly coupled to predefined predicates and struggle against ad-hoc analytical filtering.   

The Exact-Search Regime: Hardware Constraints and Cost Models
When the combined relational and graph sub-engines evaluate a query and generate an exceedingly restrictive candidate set—for example, a selectivity of s<0.05—the mathematically superior execution strategy shifts abruptly away from approximate graph traversal entirely. Scanning this minute candidate set and computing exact distance metrics yields absolute zero recall loss by mathematical definition.

Within a zero-serialization, single-process engine, the crossover threshold between an Exact Pre-Filtered Scan and an Approximate Graph Walk is governed not by algorithmic complexity, but by the physical mechanics of the underlying hardware—specifically, CPU cache hierarchies and SIMD register pressure.

Memory Bandwidth and SIMD Execution
Modern vector similarity search is heavily reliant on Single Instruction, Multiple Data (SIMD) instruction sets. An AVX-512 capable processor can pack sixteen 32-bit floating-point numbers into a single 512-bit register, executing two fused multiply-add (FMA) operations per clock cycle. At a sustained processing rate, a 4GHz CPU is capable of executing a 1024-dimensional vector dot product in roughly 32 clock cycles, equating to 8 nanoseconds per vector.   

The architectural bottleneck in this operation is never the arithmetic logic unit; it is the memory bus. Streaming one million candidate vectors, representing approximately 4 gigabytes of continuous data, through the CPU is strictly bottlenecked by the rate at which the L1 and L2 caches can be fed from main dynamic random-access memory (DRAM).   

If the pre-filtered candidate set is materialized into a contiguous memory block or structured using a Structure of Arrays (SoA) layout, an exact distance scan exhibits perfect spatial locality. The hardware prefetcher successfully anticipates the memory access patterns, effectively hiding the DRAM latency and allowing the SIMD registers to operate near peak theoretical throughput.

Conversely, traversing an HNSW graph is fundamentally an exercise in pointer-chasing. Fetching a neighbor's vector requires dereferencing an arbitrary, unpredictable memory address, resulting in an immediate L1 data cache miss. The CPU pipeline stalls while waiting for the required 64-byte cache line to be retrieved from main memory, wasting hundreds of clock cycles per hop.

Formulating the Hardware Cost Model
The decision to execute an exact scan versus a graph traversal can be mathematically formalized. Let ∣C∣=s⋅N represent the size of the pre-filtered candidate set. Let d equal the dimensionality of the vectors. Let c 
seq

  represent the amortized cost of a sequential SIMD vector fetch, and c 
rand

  represent the heavily penalized cost of a random memory fetch resulting from a cache miss.

The total computational cost of an Exact Scan is expressed as:

C 
exact

 =∣C∣⋅d⋅c 
seq

 
The expected computational cost of an In-Filtered ANN graph traversal is expressed as:

C 
ANN

 =V⋅d⋅c 
rand

 +C 
routing

 
Where V represents the total number of nodes visited during the traversal, and C 
routing

  represents the heuristic overhead of priority queue management and bitmask evaluation.

Because the penalty for a random memory fetch is vastly greater than a sequential fetch (c 
rand

 ≫c 
seq

 , frequently by two orders of magnitude), there exists a mathematically defined critical threshold ∣C 
crit

 ∣ where the cost of computing exact distances on the entire pruned subset becomes strictly less than the cost of navigating the approximate graph. If a query's graph MATCH and relational WHERE clauses successfully reduce the candidate set below this threshold, exact computation is unequivocally faster and structurally guarantees 100% recall. Production systems across the industry routinely identify this crossover threshold to be between 1% and 5% of the total vector corpus.   

Score Distribution and Quantization Effects
The mathematical purity of execution order analysis is heavily disrupted by the introduction of vector compression techniques. To accommodate billion-scale datasets within available RAM, database engines universally deploy quantization methods such as Product Quantization (PQ), Scalar Quantization (SQ), or Binary Quantization (such as Elasticsearch's DiskBBQ or Milvus's RaBitQ).   

Quantization is inherently lossy. Techniques like Product Quantization partition the high-dimensional vector space into independent sub-spaces, replacing continuous floating-point values with discrete centroid identifiers. This process introduces unavoidable quantization error, which severely squashes the distance distribution. Vectors that were geometrically distinct and resolvable in full-precision space are forced to share identical approximate distances, creating massive blocks of near-ties.   

When similarity scores are tightly clustered due to this magnitude heterogeneity, the Rank-First (Post-Filter) strategy suffers catastrophic degradation. If the top 1,000 candidates retrieved from the ANN index possess near-identical distance scores because of PQ clustering, the ranking within that top 1,000 is effectively arbitrary. If the relational post-filter is subsequently applied to this randomized cluster, it blindly drops vectors, decimating recall because the true semantic nearest neighbors were randomly pushed out of the retrieved set by artificial quantization ties.   

To combat this, search engines must deploy a two-phase Reranking architecture. Systems automatically oversample the quantized graph to retrieve a massively inflated candidate pool, apply the relational bitmask, and then rescore the surviving vectors by loading their full-precision representations from a secondary data store. In the context of a unified, single-process engine, the presence of quantization makes the Exact Scan on a pruned subset significantly more appealing. Executing an exact scan entirely sidesteps the quantization error inherent in the approximate index, providing superior ranking accuracy while eliminating the latency of the rescoring phase.   

Graph-Constrained Vector Search and Knowledge Graphs
The execution of vector search constrained by a property-graph traversal introduces unique topological variables. When evaluating a query containing a Breadth-First Search (BFS) pattern match—such as MATCH (s)-[:DEPENDS_ON*1..3]->(doc:Manual)—the fundamental question arises: Does a BFS traversal naturally produce a restrictive prune step that validates a prune-first execution order?

The validity of this approach depends entirely on the graph's topology and the degree of manifold alignment between the relational graph and the latent vector space.

If the graph exhibits low density, the BFS traversal acts as a highly selective, aggressive filter. For example, if a 3-hop traversal yields only 200 documents out of a 10-million document corpus, the selectivity is extremely high. Applying an exact vector distance computation on these 200 retrieved vectors is computationally trivial and highly cache-efficient. In this regime, executing the graph traversal first and pruning the vector space is unequivocally correct.

Conversely, if the graph exhibits high density—characterized by the presence of super-nodes connected to vast swathes of the network—the BFS traversal will output a massive candidate set. Pre-filtering this massive set via an exact scan would trigger a full sequential table scan, inflating query latency to unacceptably high levels. In this high-density regime, it is mathematically superior to defer the graph traversal until after a relaxed ANN vector search has been completed, utilizing the vector similarity metric as the primary pruning mechanism to narrow the graph expansion.

Furthermore, property-graph traversals naturally output clustered subsets. Nodes connected by short paths in a knowledge graph frequently share semantic similarity, meaning their corresponding embeddings are clustered tightly in the latent space. If the target candidates are geometrically clustered, applying a bitmask to the HNSW graph (In-Filtering) is mathematically safer than applying a uniform random filter. The site percolation threshold (s 
c

 ) is significantly lower for clustered distributions because removing nodes removes localized chunks of the vector space, rather than uniformly shattering the interstitial bridge nodes required for navigation.

State of the Art in Production Systems
An analysis of industry-leading vector databases confirms that no mature system relies on a single, statically planned execution order. Instead, they implement dynamic, threshold-driven routing mechanisms to navigate the phase transitions of hybrid search.

Database Engine	Primary Filtering Strategy	Fallback / Alternative Strategies	Implementation Details
Vespa	Cost-Based Adaptive	Exact Scan, Post-Filtering	
Compares estimated hit-ratio against approximate-threshold and post-filter-threshold to route queries dynamically.

Elasticsearch	Post-Filter (Oversampled)	ACORN In-Filtering, Exact Scan	
Uses num_candidates to oversample; triggers ACORN-1 logic if filter removes >40% of vectors.

pgvector	In-Filter (Iterative)	Exact Scan, HNSW Graph Search	
Introduced hnsw.iterative_scan to prevent post-filter cliffs by dynamically resuming graph walks.

Qdrant	In-Filter (Payload Index)	Exact Scan	
Uses filter cardinality estimation; falls back to full exact scan if cardinality is below a ~10KB threshold.

Milvus	In-Filter (BitsetView)	RaBitQ Quantized Search	
Executes via Knowhere engine using bitmasks; optimizes memory footprint via 1-bit quantization.

Weaviate	In-Filter (Allow-list)	Flat (Exact) Search	
Automatically switches to brute-force exact search when the HNSW traversal becomes too restrictive.

  
Vespa
Vespa maintains the most rigorous engineering architecture regarding hybrid execution. It operates a three-tier adaptive model governed by explicit filter selectivity estimation. When a filter matches fewer than 5% of documents (governed by the approximate-threshold), Vespa completely bypasses the HNSW graph and computes exact distances over the filtered subset. For moderate filters, it employs an ACORN-1 logic (filter-first-exploration), checking filters during the graph walk and utilizing an adaptive beam search (exploration-slack) to expand into three-hop neighborhoods to maintain recall. For highly relaxed filters, it defaults to post-filtering, scaling the user's targetHits by an expected hit ratio to oversample effectively.   

Elasticsearch and Apache Lucene
Elasticsearch relies on a hybrid of Post-Filtering and In-Filtering. For standard queries, the num_candidates parameter dictates the graph oversampling rate. To prevent the graph traversal from terminating prematurely in restricted spaces, Lucene dynamically branches its search. If a filter removes more than 40% of the vector space, the engine initiates ACORN-1 logic to evaluate filters dynamically during exploration. Crucially, if the number of matching documents in a specific segment is determined to be less than the num_candidates threshold, Elasticsearch actively abandons the HNSW graph and forces a brute-force exact scan on the filtered subset.   

pgvector (PostgreSQL)
Operating within a traditional relational optimizer, pgvector historically suffered from severe filtered query degradation. Early versions applied WHERE clauses exclusively after the HNSW scan, subjecting queries directly to the post-filter cliff. To rectify this, version 0.8.0 introduced hnsw.iterative_scan and hnsw.max_scan_tuples. This architecture transforms a naive post-filter into a dynamic, iterative pipeline. If the HNSW graph yields vectors that fail the relational filter, the database forces the index to resume scanning deeper into the graph until the requested limit is fulfilled. While this mitigates recall loss, it vastly inflates CPU overhead and latency when dealing with highly selective filters.   

Milvus and Qdrant
Milvus, utilizing its Knowhere execution engine, implements a BitsetView mechanism for soft-deletions and metadata filtering. This operates as a mid-traversal in-filter, though highly restrictive bitmasks can still induce exhaustive traversal times. Milvus heavily relies on aggressive quantization (RaBitQ) to reduce memory overhead, compressing indices to a fraction of their size while relying on tiered storage for exact rescoring. Qdrant directly integrates payload indexing with the HNSW graph, estimating filter cardinality prior to execution. If the estimated cardinality drops below a specific threshold, Qdrant abandons the graph structure entirely and executes a full exact scan over the payload index.   

A Recommended Decision Rule for a Unified Optimizer
The architectural premise that a unified, single-process engine should commit statically to a single execution order is demonstrably flawed. The zero-serialization environment removes the historical network latency penalties of multi-phase planning, rendering an Adaptive, Cost-Based Dispatcher the only mathematically sound solution.

Leveraging in-memory catalog statistics, graph traversal cardinalities, and vector space metrics, the query optimizer must implement a dynamic, three-phase decision rule. This rule is parameterized by the estimated selectivity σ of the combined relational and graph predicates, the requested retrieval limit k, and the total corpus size N.

Phase 1: Selectivity Estimation
The optimizer first evaluates the relational WHERE clauses using standard database histograms and HyperLogLog structures to calculate σ 
rel

 . Concurrently, it estimates the BFS reachability cardinality of the graph MATCH clauses to yield σ 
graph

 . These probabilities are mathematically combined to generate the global predicate selectivity σ∈[0,1] and the estimated absolute candidate set size ∣C∣=σ⋅N.

Phase 2: Dispatch Execution Thresholds
The optimizer evaluates the estimated candidate size ∣C∣ and the selectivity σ against two critical algorithmic thresholds, T 
exact

  and T 
post

 :

Rule A: Exact Prune-First Scan (The Dense SIMD Regime)
This rule is triggered when ∣C∣<T 
exact

 . The threshold T 
exact

  represents the hardware crossover point where sequential SIMD execution outperforms random memory cache misses. It is formally defined as T 
exact

 = 
d⋅(c 
rand

 −c 
seq

 )
C 
routing

 

 , which empirically translates to a candidate set size of roughly 1% to 5% of the total vector corpus.
Under this rule, the engine fully evaluates the graph BFS and relational predicates to materialize a bitmask or an array of internal identifiers. It then iterates sequentially through the dense tensor store, computing exact vector distances. This execution path guarantees 100% mathematical recall, incurs zero graph traversal overhead, and strictly minimizes L1 data cache thrashing.   

Rule B: Post-Filter with Dynamic Amplification (The Rank-First Regime)
This rule is triggered when σ>T 
post

 . The threshold T 
post

  designates the region above the order-statistics phase boundary where the oversampling factor remains computationally trivial and statistically safe. Empirically, this applies when σ>0.8.
Under this rule, the engine executes an unfiltered HNSW traversal targeting an oversampled candidate pool of K=k/σ, augmented by a variance buffer. The combined graph and relational bitmask is then applied to the retrieved results. This execution path guarantees near-perfect recall bounded by binomial probability, while maximizing graph traversal speed by eliminating branch-prediction misses during the search algorithm.

Rule C: Filtered ANN (The In-Filter Regime)
This rule is triggered when the query falls into the intermediate zone: T 
exact

 ≤σ≤T 
post

 .
Under this rule, the engine executes an HNSW graph traversal, querying the pre-evaluated relational and graph bitmask at each traversal hop. To prevent the graph from suffering site percolation shattering, the engine must implement an ACORN-style adaptive beam expansion. If a local geometric neighborhood yields zero valid nodes due to heavy filtering, the engine dynamically increases the exploration degree or falls back to an iterative deep-scan mechanism to guarantee traversal continuity. This path maintains high recall, prevents the exhaustive latency associated with exact scans on massive subsets, and successfully avoids the statistical recall cliff of post-filtering.   

Gaps in the Literature
While deep academic research exists regarding vector search execution, several critical gaps remain glaringly silent within the literature, particularly concerning unified multimodal database engines:

First, current cost models universally assume that relational attributes and property-graph topologies are statistically independent of the latent vector embeddings. The literature lacks robust, low-latency mathematical estimators capable of detecting cross-modal correlation—specifically, whether a property-graph BFS cluster perfectly aligns with a latent semantic vector cluster. If perfect manifold alignment exists, the site percolation thresholds discussed previously shift dramatically, rendering standard conservative fallback heuristics highly inefficient.

Second, existing research focuses heavily on executing filters either strictly before or concurrently during a vector search. There is a marked absence of formal research investigating the dynamic interleaving of execution: for example, initiating a partial HNSW descent to locate a broad semantic neighborhood, pausing the vector search to execute the graph BFS exclusively within that localized semantic subspace, and subsequently completing the exact vector ranking.

Third, while the negative impact of Product Quantization on post-filtering is well documented, formal mathematical models detailing how aggressive 1-bit or binary quantization (such as Milvus's RaBitQ or Elasticsearch's DiskBBQ) alters the exact hardware crossover point T 
exact

  remain missing. As high-dimensional vectors are compressed from 4 kilobytes down to 128 bytes, the cache-miss penalty dynamics shift significantly, which fundamentally alters SIMD bandwidth constraints and likely pushes the optimal Exact-Scan threshold considerably higher than current models suggest.   

By leveraging the zero-serialization capabilities of a unified engine, system architects must not commit to a static execution order. Implementing dynamic thresholds based on the physics of SIMD cache mechanics and the mathematics of order statistics will guarantee optimal execution across the vast spectrum of hybrid multimodal query topologies.

