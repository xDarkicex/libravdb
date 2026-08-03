# Foundations of a Unified Algebra for Relational, Graph, and Vector Query Processing

**Date:** August 1, 2026
**Context:** Comprehensive preservation and synthesis of formal algebraic foundations, type systems, equivalence laws, and cost modeling required for executing hybrid queries in LibraVDB. This document serves as the absolute mathematical reference for the query optimizer and execution engine.

---

## 1. The Unified Algebra and the Monadic Framework

To construct a single execution engine capable of processing relational data, topological graph patterns, and semantic vector similarity without federated bridging, the mathematical foundation must uniformly capture all three paradigms. The **monoid comprehension calculus** provides this extensible algebraic scaffolding, treating operations over multiple collection types uniformly as monoid homomorphisms.

A monoid is an algebraic structure $(M, \oplus, Z)$ consisting of a type $M$, an associative merge operation $\oplus$, and an identity (zero) element $Z$. Monoid comprehensions take the general form:
$$ \oplus \{e \mid q_1, q_2, \ldots, q_n\} $$
where $\oplus$ is the accumulator, $e$ is the head expression (projection), and $q_i$ are qualifiers (generators or boolean predicates).

### The Four Core Monoids of the Unified Engine

| Monoid Type | Domain | Merge Operator ($\oplus$) | Identity ($Z$) | Commutative | Idempotent | Application in LibraVDB |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Set** | $\mathcal{P}(T)$ | Union ($\cup$) | $\emptyset$ | Yes | Yes | Relational distinct queries, graph node sets. |
| **Bag** | $\mathcal{B}(T)$ | Multiset Union ($\biguplus$) | $\emptyset$ | Yes | No | Standard relational multiset operations (SQL `SELECT`). |
| **List** | $[T]$ | Concatenation ($\circ$) | $[]$ | No | No | Graph path traversals (ordered vertex/edge sequences). |
| **Similarity** | Ordered Lists $\le k$ | Sort-Truncate ($\oplus_{d,q}^{k}$) | $[]$ | No | No | Vector top-$k$ nearest neighbor searches. |

### The Bounded Similarity Monoid $\mathcal{M}_{top_k}(d, q)$
To natively integrate vector searches, we define the bounded similarity monoid parameterized by limit $k$, distance metric $d$ (e.g., L2 norm or Cosine), and query vector $q$. The merge operator concatenates two lists, sorts them in ascending order of distance, and truncates to the top $k$:
$$ L_1 \oplus_{d,q}^{k} L_2 = \Pi_{\le k}(\text{sort}_{d(x.v, q)}(L_1 \circ L_2)) $$
This operation is strictly associative, meaning $(A \oplus_{d,q}^{k} B) \oplus_{d,q}^{k} C = A \oplus_{d,q}^{k} (B \oplus_{d,q}^{k} C)$. Because it adheres to the monadic framework, vector top-$k$ comprehensions compose seamlessly with relational predicates and graph edge generators.

---

## 2. Formal Semantics of Unified Queries

Defining exact denotational semantics ensures operators like `GRAPH_TABLE` map to precise mathematical objects, allowing the query optimizer to prove transformation correctness.

### Denotational Semantics of GRAPH_TABLE
The `GRAPH_TABLE(G MATCH p COLUMNS c)` operator constructs a tabular view from a property graph $G = (V, E, \rho, L, P)$. The evaluation function $[[\cdot]]_G$ maps a path pattern $p$ to a set of bindings (homomorphisms). 

Path restrictors translate to strict set-theoretic restrictions on the path $\pi = (v_0, e_1, v_1, \ldots, e_n, v_n)$:
*   **WALK:** Unrestricted vertex-edge sequences. Must be bounded by a finite length quantifier to guarantee termination. (Deterministic).
*   **TRAIL:** The set of paths where no edge appears more than once ($\forall i \neq j, e_i \neq e_j$). (Deterministic).
*   **SIMPLE:** The set of paths where no vertex appears more than once, except possibly endpoints ($\forall i \neq j \in \{0,\ldots,n-1\}, v_i \neq v_j$). (Deterministic).
*   **ACYCLIC:** No vertex appears more than once at all ($\forall i \neq j, v_i \neq v_j$). (Deterministic).
*   **ANY SHORTEST:** A choice function selecting exactly one path minimizing $|\pi|$. (Non-Deterministic).
*   **ALL SHORTEST:** The set of all paths matching the minimum possible path length. (Deterministic).

The `COLUMNS` clause acts as a relational projection $\pi_c$ over the bindings, translating topological scope into standard relational schema.

### Approximation Semantics and Probabilistic Bounds
The operator `FETCH APPROX FIRST k ROWS ONLY WITH TARGET ACCURACY n PERCENT` shifts query evaluation from deterministic sets to **Probabilistic Databases**.
If $S_{exact}$ is the true top-$k$ set computed via exhaustive scan, the approximate operator returns $S_{approx}$ of size $k$. The target accuracy $n\%$ defines the expected Recall@k constraint:
$$ E\left[\frac{|S_{approx} \cap S_{exact}|}{k}\right] \ge \frac{n}{100} $$
Query evaluation becomes a constrained optimization problem: minimize estimated execution cost $C$ subject to $P(\text{Recall} \ge \alpha) \ge 1 - \delta$.

---

## 3. Equivalence Laws for the Unified Algebra

Equivalence laws justify the logical plan transformations performed by the optimizer (e.g., Cascades/Volcano framework). Without them, Filtered Vector Search (FVS) cannot be algebraically proven correct.

### Top-K Pushdown and Global-Local Selectivity (GLS)
Exact pushdown $top\text{-}k(k, \sigma_p(R), d) \equiv \sigma_p(top\text{-}k(k, R, d))$ is invalid unless all top-$k$ elements satisfy predicate $p$. Instead, the optimizer relies on a **parameterized oversampling equivalence**:
$$ top\text{-}k(k, \sigma_p(R), d) \approx \sigma_p(top\text{-}k(k + c(sel(p), \theta_{GLS}), R, d)) $$
Where $\theta_{GLS}$ is the spatial correlation between the predicate and the local vector neighborhood. If spatially independent, oversampling scales as $c \propto k / sel(p)$.

### Unified Equivalence Catalog
| Left-Hand Side (LHS) | Right-Hand Side (RHS) | Condition of Validity |
| :--- | :--- | :--- |
| $top\text{-}k(k, \sigma_p(R), d)$ | $\sigma_p(top\text{-}k(k+c, R, d))$ | Valid probabilistically ($c = f(sel(p), \theta_{GLS})$). Requires monotonic similarity distribution relative to $p$. |
| $top\text{-}k(k, R \bowtie_E S, d)$ | $R \bowtie_E top\text{-}k(k \times E[\text{fanout}], S, d)$ | Valid if join is 1:N and join predicate does not prune minimal elements. |
| $\sigma_p(\text{GRAPH\_TABLE}(G, P))$ | $\text{GRAPH\_TABLE}(\sigma_p(G), P)$ | Valid if $p$ evaluates exclusively on properties of a node/edge and is monotonic to the path restrictor (e.g., ACYCLIC). |
| $R \bowtie_{k\text{-}NN} S$ | $top\text{-}k(k, S) \bowtie_{Hash} R$ | Valid if rank monotonicity is preserved (Rank-Join principle). |

---

## 4. The Type System and Row Polymorphism

To support unified operations, the type system must form a cohesive lattice integrating standard scalars, fixed vectors, and schema-flexible graph elements.

### The Vector Type Constructor
`VECTOR(N)` is a parameterized type constructor, ensuring metric operations are type-safe. Subtyping is invariant with respect to $N$: $\dim(v_1) = \dim(v_2)$.
$$ \frac{\Gamma \vdash e_1 : \text{VECTOR}(N) \quad \Gamma \vdash e_2 : \text{VECTOR}(N)}{\Gamma \vdash e_1 \leftrightarrow e_2 : \text{FLOAT}} $$

### Graph Row Polymorphism
Graph databases employ open schemas. To model this statically within a SQL engine, LibraVDB utilizes **Wand's Row Polymorphism** (1989/1991).
A graph node is an extensible record type, evaluated using a row variable $\rho$ representing the statically unknown remainder of the property map:
$$ \Gamma \vdash node : \{id: \text{ID}, labels: \text{Set}\langle\text{Label}\rangle, p_1: \tau_1, \ldots, p_n: \tau_n \mid \rho\} $$
When matching a node `(n:Person {name: 'Alice'})`, the inference engine deduces $n : \{name: \text{TEXT} \mid \rho\}$. This guarantees type safety for `COLUMNS` projections while preserving open-world flexibility.

---

## 5. The Unified Cost Model (ECQO)

The cost-based optimizer maps logical expressions to physical operators using Exact Cardinality Query Optimization (ECQO).

### Operator Cost Formulas
*   **B-tree / ART:** Point lookups cost $O(\log N)$. Range scans evaluate to $O(K + \log N)$.
*   **HNSW Traversal:** Expected cost of approximate k-NN is $O(\log(N_{filtered}) \times M \times D)$.
*   **Graph Traversal:** Unconstrained worst-case is $O(|V| + |E|)$. `TRAIL` bounding is NP-hard unbounded, requiring active edge-marking depth-first search.

### Error Composition and Navigational Dead-Ends
When concatenating approximate operators, accuracy must compose:
*   **Independent Errors:** A 95%-accurate HNSW index combined with a 95%-accurate Bloom filter yields $0.95 \times 0.95 \approx 0.9025$ recall.
*   **Correlated Errors:** Filtering a 95%-accurate HNSW index with a 100%-accurate relational predicate theoretically preserves 95% only if the predicate evaluates uniformly. If the filter isolates a sparse sub-manifold, graph connectivity degrades (navigational dead-ends), dropping recall drastically.
The optimizer utilizes local selectivity thresholds to adaptively switch routing heuristics (e.g., blind routing vs. directed routing) at runtime to satisfy $P(\text{Recall} \ge \alpha) \ge 1 - \delta$.

---

## 6. Worked Example: End-to-End Formal Treatment

**The Query:** Identify compromised hosts connecting to servers via HTTPS and retrieve the 10 servers most semantically similar to a query vector, bound by a 90% accuracy constraint.

```sql
SELECT a.name, b.content, distance(b.embedding, :query) AS dist
FROM GRAPH_TABLE (
    network_graph
    MATCH (a IS host) -[e IS connection]->{1,3} (b IS server)
    WHERE a.status = 'COMPROMISED' AND e.port = 443
    COLUMNS (a.name AS name, b.content AS content, b.embedding AS embedding)
) AS gt
ORDER BY distance(gt.embedding, :query) ASC
FETCH APPROX FIRST 10 ROWS ONLY WITH TARGET ACCURACY 90 PERCENT;
```

### (a) Algebraic Form
Nested monoid comprehension utilizing $\mathcal{M}_{top_{10}}(d, q)$ over a graph `WALK` length 1-3:
$$ \oplus_{d,q}^{10} \{\pi_{a.name, b.content, dist}(b) \mid a \leftarrow V, b \leftarrow V, e \leftarrow E, \text{labels}(a, \text{host}) \land \text{labels}(b, \text{server}) \land a.status = \text{'COMP'} \land e.port = 443 \land \pi \in \text{WALK}_1^3(a,b)\} $$

### (b) Type Judgment (Row Polymorphism)
$$ \Gamma \vdash a : \{status: \text{TEXT}, name: \text{TEXT} \mid \rho_a\} $$
$$ \Gamma \vdash b : \{content: \text{TEXT}, embedding: \text{VECTOR}(N) \mid \rho_b\} $$
$$ \Gamma \vdash e : \{port: \text{INT} \mid \rho_e\} $$

### (c) Cost Model Synthesis
*   Selectivity of `status = 'COMPROMISED'` is estimated at **0.01** via a 1024-bin histogram.
*   Graph fanout estimates predict an average of **5** reachable servers per compromised host.
*   Expected cardinality of valid servers = $0.01 \times |V_{host}| \times 5$.

### (d) Optimal Plan Shape Selection
Because the expected cardinality is sufficiently low (e.g., 500 rows), ECQO selects a **Graph-first (Pre-filter)** plan. It utilizes the ART/B-tree to locate compromised hosts, traverses the graph (filtering by port 443), collects the valid servers, and computes exact vector distances via brute-force AVX2 SIMD. It entirely bypasses the HNSW vector index, guaranteeing 100% accuracy and completely avoiding the massive navigational overhead and random I/O associated with traversing an HNSW graph on a highly sparse subset.

---

## 7. Open Questions and Future Theoretical Frontiers

The formal unification surfaces mathematical frontiers where consensus has not been reached:
1.  **Expressive Power:** While relational algebra is equivalent to first-order logic, the addition of top-$k$ similarity operations over high-dimensional manifolds pushes the algebra into metric space logic. The formal complexity class remains an open problem.
2.  **View Updates:** Translating `INSERT`/`UPDATE` operations against a virtual `GRAPH_TABLE` view containing vector properties introduces non-determinism that current monoid calculus cannot fully express.
3.  **Spatial Constraints:** Enforcing constraints like `UNIQUE(embedding)` lacks topological utility due to floating-point drift. Defining spatial dispersion constraints (e.g., $\forall x, y \in R, d(x,y) > \epsilon$) requires novel mathematical validation algorithms upon insertion.
