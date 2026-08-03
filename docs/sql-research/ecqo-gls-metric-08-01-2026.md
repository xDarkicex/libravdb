# Formal Specification of the Global-Local Selectivity ($\theta_{GLS}$) Metric

**Date:** August 1, 2026
**Context:** Mathematical definition and implementation blueprint for $\theta_{GLS}$, the core metric enabling Exact Cardinality Query Optimization (ECQO) to perform Filtered Vector Search (FVS) pushdowns safely.

---

## 1. The Architectural Context of $\theta_{GLS}$
In hybrid query optimization, the equivalence law for Filtered Vector Search asserts:
$$ top\text{-}k(k, \sigma_p(R), d) \approx \sigma_p(top\text{-}k(k + c(sel(p), \theta_{GLS}), R, d)) $$

The oversampling cost-delta function $c()$ calculates the surplus candidate vectors needed to guarantee $k$ valid hits after the relational predicate $\sigma_p$ is evaluated. Without a computable $\theta_{GLS}$, the optimizer risks severe recall collapse (underestimation) or massive latency spikes (overestimation).

## 2. Mathematical Definition
$\theta_{GLS}$ quantifies spatial autocorrelation—the degree to which a predicate's satisfaction correlates with the underlying high-dimensional vector geometry. It is derived from the **Local Moran's I coefficient**:
$$ \theta_{GLS}(p) = \frac{N}{\sum_{i=1}^N \sum_{j=1}^N w_{ij}} \frac{\sum_{i=1}^N \sum_{j=1}^N w_{ij}(x_i - \sigma_g)(x_j - \sigma_g)}{\sum_{i=1}^N (x_i - \sigma_g)^2} $$
*   $\theta_{GLS} \approx 0$: The predicate is spatially random (independent of vector topology).
*   $\theta_{GLS} \to 1$: The predicate is highly clustered, requiring drastic oversampling modification.

## 3. Computable Implementation: Histogram-Based Region Variance
To meet the stringent sub-millisecond budget of ECQO, $\theta_{GLS}$ must be computed in under $10\mu s$. 
The chosen implementation is a coarse-grained macroscopic approximation utilizing spatial partitioning (e.g., $C=1024$ Voronoi cells or Leiden pages).

For a predicate $p$, the system maintains a localized selectivity array: $[\sigma_{l,1}, \sigma_{l,2}, \dots, \sigma_{l,C}]$.
The optimizer fetches global selectivity ($\sigma_g$) and computes the normalized spatial variance:
$$ \theta_{GLS}(p) \approx \frac{\frac{1}{C}\sum_{c=1}^C (\sigma_{l,c} - \sigma_g)^2}{\sigma_g(1-\sigma_g)} $$

### ECQO Budget Allocation ($1000\mu s$ Total)
1.  AST Parsing: $150\mu s$
2.  Relational Histograms ($\sigma_g$): $50\mu s$
3.  Localized ANN Probe: $600-700\mu s$
4.  Plan Enumeration: $100\mu s$
5.  **$\theta_{GLS}$ Computation:** $\le 10\mu s$ (AVX-512 scans the 1024-float array in 3-8 $\mu s$).

## 4. Catalog Representation and Lifecycle
The vector catalog maintains an in-memory hash table: `Map[predicate_hash, Array[float32]]`.
Tracking the top $10,000$ frequent predicates requires a negligible $40$ MB of RAM. 
To avoid WAL bottlenecks during high-velocity ingestion, regional statistics are aggregated in thread-local buffers and flushed asynchronously via an Exponential Moving Average (EMA).

## 5. Bounding the Worst-Case Execution
To prevent infinite loops during massive uncertainty, the cost-delta function gracefully degrades to a theoretical maximum oversampling limit for perfect uniformity:
$$ \lim_{\theta_{GLS} \to 0} c(sel(p), \theta_{GLS}) = \frac{k}{sel(p)} - k $$
As $\theta_{GLS} \to 1$, oversampling scales to a hardware-specific constant relative to cluster diameter.

## 6. Interaction with iFVS
$\theta_{GLS}$ and iFVS compose synergistically:
*   $\theta_{GLS}$ detects strong spatial clustering (dense sub-manifolds prone to PQ distance collapse).
*   The optimizer dynamically injects the **iFVS perturbation operator**, shifting codebook centroids toward the cluster.
*   This millisecond overhead vastly reduces the requisite oversampling factor $c()$, minimizing total traversal latency.
