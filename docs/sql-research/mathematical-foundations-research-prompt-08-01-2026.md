# Mathematical Foundations Research Prompt: The Unified SQL + Graph + Vector Algebra

**Date:** 2026-08-01
**Status:** Foundational research — for the next research pass
**Preread:**
- `unified-sql-engine-vision-08-01-2026.md`
- `unified-query-superset-architecture-08-01-2026.md`
- `deep-research-prompt-08-01-2026.md` (engineering research; orthogonal to this document)

---

## What This Document Is

The previous research prompt asked engineering questions: which Go library, which AST layout, which protocol framing. Those are downstream of the *mathematics* of the unified engine. This prompt asks the foundational questions:

- What is the unified algebra?
- What are the equivalence laws that let an optimizer rewrite a unified query?
- What is the formal semantics of a `GRAPH_PATTERN ... ORDER BY <-> ... LIMIT 10` expression?
- What is the type system that covers scalars, vectors, and graph entities uniformly?
- What is the cost model that lets the optimizer pick between graph-first, vector-first, or join-first plans?

These are questions for the academic database theory literature. The answers exist, in pieces, across the relational, graph, and vector research communities. The research pass is to *synthesize* them into a coherent foundation that the implementation can sit on.

Without this foundation, every "unified query" is a special case. With it, every unified query is a compositional expression in a single algebra.

---

## Q1 — The Unified Algebra

**The question:** What is the algebra that uniformly captures relational operations, graph path operations, and vector operations, with compositional semantics?

**Why it matters:** The algebra is the optimizer's universe. Every equivalence law, every rewrite rule, every cost model lives in the algebra. If the algebra is incomplete (e.g., vector operations are UDFs not first-class), the optimizer can only reason about the relational part, and the "unified" engine degenerates into a relational engine with vector functions bolted on.

**Specific questions to answer:**

1. **What is the monad-comprehension foundation that unifies SQL, XPath, and graph pattern matching?** Wadler's 1990s work on monad comprehensions shows that SQL `SELECT ... FROM ... WHERE ...`, XPath `for ... in ... where ... return`, and graph pattern matching can all be expressed as comprehensions over a "bulk" monad (list/set/multiset). What is the exact monad for each, and what is the denotational semantics of comprehension composition?

2. **How do vector operations fit into this framework?** A vector column is a tuple component. The distance function is a derived measure. Top-k by distance is a non-boolean "selection." Can top-k be expressed as a comprehension, or does it require a new monad (the "ordered monad" or "similarity monad")?

3. **What is the algebra of path expressions over labeled property graphs?** This is the standard "regular path query" (RPQ) algebra, extended to conjunctive RPQ (CRPQ), 2RPQ (two-way), and ECRPQ (extended with negation). How does this algebra interact with the relational algebra? Specifically:
   - Is `GRAPH_TABLE(MATCH (a) -[e]-> (b) WHERE ...) AS g` equivalent to a relational expression over the underlying vertex and edge tables? (This is the "graph-as-view" claim of SQL/PGQ.)
   - What is the precise translation from GRAPH_TABLE to relational algebra?
   - What path restrictors (WALK, TRAIL, SIMPLE, ACYCLIC, ANY SHORTEST) are definable as algebraic restrictions on the path set?

4. **What is the algebra of k-nearest-neighbor queries?** Is top-k a first-class operator (`topk(k, R, f)`), or is it a derived operator from `ORDER BY ... LIMIT`? The choice has implications for:
   - Equivalence laws (pushdown through top-k)
   - Compositionality with joins (k-NN join)
   - Compositionality with selections (filtering + top-k)

5. **What is the unified algebra for `ORDER BY distance LIMIT k` combined with `WHERE` filters and `GRAPH_PATTERN`?** The denotational question: given a database state Δ, what is the result relation? The result is a finite sequence of tuples (ordered). What is the formal definition of the result as a function of Δ, the query, and the k?

**Actionable answer format:**
- A formal definition of the unified algebra (operators, types, equations)
- A monad-comprehension formulation that includes all three paradigms
- A precise translation from SQL/PGQ + pgvector syntax to the algebra
- A proof of compositionality: every operator in the algebra composes with every other

**Sources to consult:**
- Wadler, "Comprehending Monads" (1990)
- Wadler, "The Essence of Functional Programming" (1992) — monad comprehensions
- Buneman, "Comprehensions" (1996) — bulk types and comprehensions
- Fegaras, Maier, "Optimizing Object Queries Using an Effective Calculus" (1995) — monoid comprehensions
- Libkin, "Elements of Finite Model Theory" (2004) — finite semantics for queries
- Angles, Gutierrez, "Foundations of Modern Graph Pattern Matching" (2018 textbook) — the canonical graph algebra reference
- "The Relational Model with Relations" (Rel, 2018+) — categorical relational algebra
- Apache Calcite's algebraic specification — practical reference for what a unified algebra looks like
- "GQL: A Query Language for Property Graphs" (ISO/IEC 39075) — formal semantics appendices
- "SQL/PGQ" (ISO/IEC 9075-16) — formal semantics appendices

---

## Q2 — Equivalence Laws for the Unified Algebra

**The question:** What are the algebraic identities that let an optimizer rewrite a unified query? Specifically, when is pre-filtering equivalent to post-filtering? When can a graph traversal be reordered with a vector search? When can a vector distance predicate be pushed through a join?

**Why it matters:** The optimizer's power is bounded by the equivalence laws it knows. If the algebra has no equivalences between vector and relational operators, the optimizer cannot reason across the boundary. The FVS strategies (pre-filter, post-filter, in-filter) are *equivalent* in some regimes and *inequivalent* in others. The equivalence law is the formalization of "in this regime."

**Specific questions to answer:**

1. **What are the equivalence laws for vector pre-filtering vs post-filtering?**
   - For a filter predicate σ_p and a vector relation R, when is `topk(k, σ_p(R), d) ≡ σ_p(topk(k + c, R, d))`? (Pre-filter is equivalent to over-fetch + post-filter, for some c.)
   - What is the value of c as a function of filter selectivity, vector distribution, and the distance function?
   - Under what conditions is the equivalence exact, and when is it approximate (with the user-specified accuracy target)?

2. **What are the equivalence laws for graph traversal reorderings?**
   - When can a relational selection σ_p be pushed through a graph traversal? (Always, when the predicate is on a node/edge variable, never when it depends on path-level properties.)
   - When can a graph traversal be reordered with a relational join? (When the join doesn't change the graph topology.)
   - When can a `GRAPH_TABLE` be flattened into relational operators? (Always, given the view-of-tables semantics, but at what cost?)

3. **What are the equivalence laws for k-NN joins?**
   - The k-NN join: for each tuple in R, find its k nearest neighbors in S by distance. Equivalent formulations:
     - Nested-loop with top-k
     - Block nested-loop with top-k
     - Hash-based partition + per-partition top-k
     - HNSW on S, probe for each R
   - What are the equivalence laws? When is one formulation cheaper than another?

4. **What are the equivalence laws for top-k with limit offset?**
   - `LIMIT k OFFSET m` vs `LIMIT k + m` then drop first m — when are these equivalent?
   - With ties, the equivalence breaks (the tie-breaking order matters)
   - How does this interact with `FETCH APPROX FIRST k ROWS ONLY WITH TARGET ACCURACY n PERCENT`?

5. **What are the filter commutativity rules for combined predicates?**
   - σ_p(σ_q(R)) ≡ σ_q(σ_p(R)) — always true
   - topk(k, σ_p(R), d) ≡ σ_p(topk(k, R, d)) — never true (filter can change the top-k)
   - topk(k, σ_p(R), d) ≡ topk(k, σ_p(topk(k + c, R, d)), d) — true for c ≥ expected filtered-out count

**Actionable answer format:**
- A table of equivalence laws for the unified algebra
- A characterization of when each law is exact vs approximate
- For approximate laws, the accuracy bound as a function of the input distribution
- A worked example: the FVS strategy selection, formalized as an equivalence-law application

**Sources to consult:**
- Aho, Sagiv, Ullman, "Equivalences Among Relational Expressions" (1979) — the original
- Abiteboul, Hull, Vianu, "Foundations of Databases" (1995) — Chapter on query equivalence
- Graefe, "The Cascades Framework for Query Optimization" (1995)
- "Cost-Based Query Optimization" (Graefe, McKenna) — equivalence-law application
- "The k-NN Join" (Böhm, Kriegel, 2001)
- "An Optimality Proof for k-NN Query Optimization" (Friedman, et al.)
- "Top-k Query Processing" (Ilyas, Beskales, Soliman) — equivalence laws for top-k
- Exqutor paper (2024) — cost-based optimization for vector predicates

---

## Q3 — The Formal Semantics of a Unified Query

**The question:** What is the denotational semantics of a query that mixes relational, graph, and vector operations? Given a database state, what is the result?

**Why it matters:** The semantics defines what queries *mean*. The optimizer's rewrites are equivalence laws because the two sides of a rewrite denote the same function from database states to results. The implementation has to produce results that match the denotational definition. Without the formal semantics, every feature is a special case.

**Specific questions to answer:**

1. **What is the formal semantics of a path expression with a path restrictor?** The standard semantics:
   - `[e MATCH p1 (a) ->{1,3} (b)]` is the set of all paths from `a` to `b` of length 1–3 that match pattern p1
   - The restrictor WALK allows any path
   - TRAIL forbids edge repetition
   - SIMPLE forbids node repetition
   - ACYCLIC forbids cycles
   - ANY SHORTEST returns exactly one shortest path per (a, b) pair
   - ALL SHORTEST returns all shortest paths
   - SHORTEST k GROUP partitions by (a, b) and returns k groups
   - What is the precise denotational definition of each?

2. **What is the formal semantics of `ORDER BY vector_distance LIMIT k` in the presence of ties?**
   - With distance ties, the order is not total
   - SQL specifies "the result is implementation-defined" in the presence of ties (most engines)
   - Is the result a sequence (ordered) or a bag with a tie-breaking rule?
   - What is the effect of `OFFSET m` on ties?
   - What is the formal semantics of `FETCH APPROX ... WITH TARGET ACCURACY`? (What does 90% accuracy mean formally?)

3. **What is the formal semantics of a graph pattern with a vector predicate on a node variable?**
   - `MATCH (a) -[e]-> (b) WHERE b.embedding <-> query < 0.5`
   - Is the predicate applied to the graph's node set before or after pattern matching?
   - Does the predicate change the graph (e.g., remove nodes) or just filter the matches?
   - What is the result type — paths, nodes, edges, or all three?

4. **What is the formal semantics of `GRAPH_TABLE` as a subquery in a `FROM` clause?**
   - `FROM GRAPH_TABLE(g MATCH (a) -[e]-> (b) COLUMNS (a.x, b.y)) AS gt`
   - The result is a relation with columns `x` and `y`
   - What is the type of each column?
   - How is the relation deduplicated?
   - What happens when the pattern matches no paths?

5. **What is the formal semantics of aggregation over a graph pattern?**
   - `SELECT count(*), avg(b.embedding <-> query) FROM GRAPH_TABLE(...)`
   - Is the count over paths, nodes, or edges?
   - Is the average over the matched instances of the distance expression, or over the result relation?
   - What is the formal definition of aggregation semantics over a pattern?

6. **What is the formal semantics of approximate execution (`FETCH APPROX ... WITH TARGET ACCURACY n PERCENT`)?**
   - Is the accuracy measured per-operator or end-to-end?
   - How do per-operator accuracies compose? (If HNSW is 95% accurate and the post-filter is 100% accurate, what's the end-to-end accuracy? 95%? 95% × 100%? 100%? Depends on how the operators compose.)
   - What is the probabilistic semantics? (Expected accuracy, with-probability accuracy, worst-case accuracy?)

**Actionable answer format:**
- A denotational semantics for each of the above constructs
- A worked example: the formal semantics of a representative unified query
- A proof of well-definedness: every well-typed query has a unique result
- A statement of how approximation composes across operators

**Sources to consult:**
- The SQL/PGQ standard (ISO/IEC 9075-16:2023), formal semantics appendices
- The GQL standard (ISO/IEC 39075:2024), formal semantics appendices
- Angles, Gutierrez textbook, Chapters on semantics
- "Semantics of Graph Query Languages" (Angles, 2022)
- "Approximate Query Processing" (Chaudhuri, et al.) — accuracy semantics
- "Probabilistic Databases" (Suciu, Olteanu, Koch) — formal semantics under uncertainty
- Wadler, Fegaras monad comprehension papers

---

## Q4 — The Type System

**The question:** What is the type system that covers scalars, vectors (with fixed dimensionality), and graph entities (nodes, edges, paths) uniformly?

**Why it matters:** Type checking at parse time catches errors before the optimizer. Type-directed dispatch at execution time lets the engine select the right operator implementation. Without a unified type system, vector operations are UDFs, graph operations are syntax sugar for joins, and "unified" is a lie at the implementation level.

**Specific questions to answer:**

1. **What is the type lattice?**
   - Scalars: INT, BIGINT, FLOAT, DOUBLE, TEXT, BOOLEAN, TIMESTAMP, DATE, etc.
   - Vectors: VECTOR(N) where N is a positive integer (the dimensionality, fixed at table creation)
   - Graph: NODE, EDGE, PATH — what are these types formally?
   - Composite: ROW, ARRAY, MAP
   - NULL: bottom
   - The lattice has a subtyping relation (VECTOR(384) is NOT a subtype of VECTOR(512))
   - What is the subtyping relation for graph types? (NODE with property x is a subtype of NODE without property x?)

2. **What is the type checking strategy?**
   - Static: all types resolved at parse time. Errors at parse. The cost is in the type checker, not the runtime.
   - Dynamic: types resolved at plan time. Errors at execution. More flexible, less safe.
   - Hybrid: scalar types static, UDTs dynamic. Compromise.

3. **What are the type rules for the unified operators?**
   - `<->` takes two VECTOR(N) of the same N (or coercible) and returns a scalar
   - `MATCH (a IS Person) -[e IS Knows]-> (b IS Person)` requires that the `Person` label is bound to a node type and `Knows` to an edge type
   - What is the formal type judgment for each operator?

4. **What is the polymorphism story?**
   - Parametric polymorphism: `VECTOR` as a type constructor `VECTOR(N)`, with N as a type parameter
   - Row polymorphism: graph types as rows of (label, properties)
   - Subtype polymorphism: `Person` is a subtype of `Node` (if labels are subtypes)
   - How do these interact?

5. **What is the type representation in the catalog?**
   - Each table has a set of columns with type definitions
   - Each VECTOR(N) column has a fixed N
   - Each property graph has vertex tables, edge tables, and property definitions
   - What is the storage representation of the catalog in the type system itself?

**Actionable answer format:**
- A type lattice diagram
- A formal type judgment for each unified operator
- A type checking algorithm
- A catalog type representation

**Sources to consult:**
- "Types and Programming Languages" (Pierce) — the standard type theory reference
- "The Relational Model with Relations" (Rel) — categorical types for relations
- "Simply Typed Lambda Calculus" — foundations
- "Row Polymorphism" (Wand, 1991) — for graph types
- PostgreSQL's type system documentation
- Apache Calcite's type system
- "pgvector: Type System" — the precedent for VECTOR(N) in a SQL type system

---

## Q5 — The Unified Cost Model

**The question:** What is the cost model that lets the optimizer pick a plan across relational, graph, and vector operations?

**Why it matters:** The cost model is the optimizer's objective function. Every plan is scored. The plan with the lowest cost wins. If the cost model can't reason about vector selectivity, the optimizer always picks the relational-first plan, and the FVS strategies are inaccessible. If it can't reason about graph fanout, the optimizer always picks the join-first plan, and the graph-first strategies are inaccessible.

**Specific questions to answer:**

1. **What is the cost of a vector operation?**
   - HNSW: O(log N × M) where M is the number of layers traversed and the average neighbors per layer
   - Brute force: O(N × d) where d is the dimensionality
   - IVF-PQ: O(N / sqrt(k) × d) approximately, depending on the IVF clustering
   - How does this cost scale with filter selectivity (for in-filtering)?

2. **What is the cost of a graph operation?**
   - BFS/DFS traversal: O(V + E) worst case
   - With filter selectivity: O(σ × (V + E)) where σ is the selectivity
   - With restrictors: WALK is O(V + E), TRAIL is exponential in the worst case, SIMPLE is also exponential
   - What is the realistic cost on real graph distributions (power-law, etc.)?

3. **What is the cost of a relational operation?**
   - Selection σ_p(R): O(|R|) worst case, O(|R| × p) on average with selectivity p
   - Join R ⋈ S: O(|R| × |S|) nested loop, O(|R| + |S|) hash, O(|R| log |R| + |S| log |S|) sort-merge
   - With index: O(|R| × log |S|) for indexed nested loop
   - What is the cost with a B-tree / ART index?

4. **How do these costs compose in a unified plan?**
   - A plan is a tree of operators. Each operator has a cost. The total cost is the sum of operator costs.
   - The challenge: a vector operation and a graph operation can have *correlated* cost (e.g., a vector search that is constrained to a subgraph has lower cost than a full vector search).
   - The cost model has to capture these correlations.

5. **What is the selectivity estimation for vector predicates?**
   - Traditional histograms fail for high-dimensional vectors
   - The Exqutor approach: localized ANN probe to estimate selectivity
   - The iFVS approach: filter-aware PQ codebook that encodes filter relevance into the cost
   - How do these compose? Can a single selectivity estimate drive a unified plan choice?

6. **What is the selectivity estimation for graph patterns?**
   - Path selectivity: what fraction of paths of length L match a given pattern?
   - Label selectivity: what fraction of nodes have a given label?
   - Property selectivity: what fraction of nodes satisfy a property predicate?
   - The challenge: path selectivity depends on the graph structure, not just the schema

**Actionable answer format:**
- A cost formula for each operator
- A cost composition rule for operator trees
- A selectivity estimation strategy for each predicate type
- A worked example: the cost of the representative unified query under three different plan shapes

**Sources to consult:**
- Graefe, McKenna, "The Volcano Optimizer Generator" (1993) — the canonical cost model
- "Cost-Based Query Optimization" (Graefe) — the canonical reference
- "Cardinality Estimation" (Leis, et al., 2015) — for relational
- Exqutor paper (2024) — for vector selectivity
- "Graph Cardinality Estimation" (Park, et al.) — for graph selectivity
- "iFVS" (2024) — filter-aware vector cost model
- "FoundationDB's Cost Model" — for transaction cost

---

## Q6 — The Approximation Semantics

**The question:** What does `FETCH APPROX FIRST k ROWS ONLY WITH TARGET ACCURACY n PERCENT` formally mean, and how do per-operator accuracies compose?

**Why it matters:** Approximate execution is the only way to bound the latency of vector search and large graph traversals. The user expresses an accuracy budget. The engine decomposes that budget across operators. If the semantics is wrong, the engine either over-allocates work (paying for accuracy it doesn't need) or under-allocates (returning results that miss the accuracy target).

**Specific questions to answer:**

1. **What is the formal definition of "accuracy" for a top-k vector query?**
   - Recall: fraction of the true top-k in the result
   - Precision: fraction of the result that is in the true top-k
   - F1: harmonic mean
   - Distance-error: sum of (true_distance - reported_distance) over the result
   - Which one does the user mean by "accuracy"?

2. **How do per-operator accuracies compose?**
   - If HNSW has recall 95% and the post-filter is 100% accurate, what's the end-to-end recall? (95% — filtering doesn't add errors.)
   - If the post-filter is 99% accurate and HNSW is 95% accurate, what's the end-to-end recall? (Not just multiplication — the errors may be independent or correlated.)
   - What is the formal model of accuracy composition?

3. **What is the formal semantics of `WITH TARGET ACCURACY n PERCENT`?**
   - Expected accuracy ≥ n?
   - With-probability accuracy ≥ n?
   - Worst-case accuracy ≥ n?
   - Each has different cost implications.

4. **How does the optimizer allocate an accuracy budget across operators?**
   - Given a total budget of 90% and a plan with two approximate operators, how to split?
   - Optimal allocation: marginal cost of accuracy per operator
   - What is the right optimization problem? Lagrangian? Cost minimization subject to accuracy constraint?

5. **What is the cost of meeting an accuracy target?**
   - HNSW `ef` parameter controls recall vs latency: roughly exponential
   - IVF `nprobe` parameter controls recall vs latency
   - Graph traversal fanout: more pre-computed paths = higher recall
   - What is the cost-accuracy curve for each operator?

**Actionable answer format:**
- A formal definition of accuracy for top-k vector queries
- A composition rule for per-operator accuracies
- An optimization problem: minimize cost subject to accuracy constraint
- A worked example: allocating a 90% accuracy budget across an HNSW search and a graph traversal

**Sources to consult:**
- "Approximate Query Processing" (Chaudhuri, et al.) — the canonical reference
- "Probabilistic Databases" (Suciu, Olteanu, Koch) — formal semantics
- "Quality-Aware Query Optimization" (the academic literature on accuracy-constrained optimization)
- HNSW parameter tuning literature
- IVF parameter tuning literature

---

## Q7 — Worked Example: The Formal Treatment of a Representative Unified Query

**The question:** Take a representative unified query and produce its formal treatment — algebra, semantics, equivalence laws, type rules, cost model.

**The query:**
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

**Specific questions to answer:**

1. **What is the algebraic form of this query?**
   - Decompose into: graph pattern (with restrictor), relational filter, projection, vector distance, top-k with accuracy target
   - What is the denotational definition of each step?

2. **What are the equivalence laws that apply?**
   - Can the filter `a.status = 'COMPROMISED'` be pushed into the graph pattern? (Yes, it's a node predicate.)
   - Can the filter `e.port = 443` be pushed into the graph pattern? (Yes, it's an edge predicate.)
   - Can the top-k be applied before or after the graph pattern? (This is the FVS question for graph + vector.)
   - Is the result equivalent to: filter nodes by status, traverse from those nodes, filter edges by port, project, top-k by distance?

3. **What is the cost model?**
   - Cost of `a.status = 'COMPROMISED'` filter: depends on selectivity (let's say 1%)
   - Cost of graph traversal from filtered nodes: 1% of nodes × 3-hop fanout
   - Cost of `e.port = 443` filter: depends on edge selectivity
   - Cost of top-k by distance: HNSW with ef chosen to meet 90% accuracy
   - Total cost: sum, with selectivity-driven multipliers

4. **What is the optimal plan shape?**
   - Pre-filter status, then graph, then filter port, then top-k?
   - Or graph-first, then filter?
   - Or top-k first, then filter?
   - The optimizer's job: enumerate plans, score by cost, pick the lowest.

5. **What is the type judgment?**
   - `a.name`: TEXT (from a.name column)
   - `b.content`: TEXT (from b.content column)
   - `b.embedding`: VECTOR(384) (assumed)
   - `distance(b.embedding, :query)`: scalar distance, type-checked against the metric

**Actionable answer format:**
- A complete formal treatment of this query
- The algebra, semantics, equivalence laws, type rules, and cost model applied
- A worked optimizer trace: enumerate plans, score, pick the best

---

## Q8 — The Open Research Questions

After answering Q1–Q7, the following open questions are likely to remain:

1. **What is the "right" unified algebra?** The monad-comprehension approach is promising but not proven for vector operations. Is there an alternative (e.g., categorical query languages, dependent types)?

2. **What is the "right" approximation semantics?** Expected accuracy? With-probability? Worst-case? Each has different cost and different guarantees. The literature doesn't converge.

3. **How does the unified algebra interact with update operations?** `INSERT INTO nodes VALUES (...)`, `DELETE FROM edges WHERE ...`, `UPDATE nodes SET embedding = ... WHERE ...`. Are updates part of the algebra, or is the algebra read-only?

4. **What is the role of constraints in the unified engine?** `PRIMARY KEY`, `FOREIGN KEY`, `UNIQUE`, `CHECK`. Constraints are integral to SQL. How do they interact with vector and graph operations? (E.g., does `UNIQUE(embedding)` make sense?)

5. **What is the right expressive power?** SQL/PGQ has the expressive power of relational algebra + graph patterns. Adding vector operations might increase the expressive power beyond relational-complete. Is that desirable?

---

## What This Document Is Not

- **Not an engineering prompt.** The previous research prompt (`deep-research-prompt-08-01-2026.md`) covers engineering.
- **Not a survey.** Each question targets a specific foundation, not a literature overview.
- **Not a design doc.** The answers to these questions will inform the design, but this document doesn't propose a design.

---

## Sources to Prioritize

The research questions are heavy on database theory. The canonical sources:

**Foundational theory:**
- Abiteboul, Hull, Vianu, "Foundations of Databases" (1995) — the standard reference
- Libkin, "Elements of Finite Model Theory" (2004) — finite query semantics
- "The Relational Model with Relations" — categorical relational algebra

**Monad comprehensions and bulk types:**
- Wadler, "Comprehending Monads" (1990)
- Wadler, "The Essence of Functional Programming" (1992)
- Fegaras, Maier, "Optimizing Object Queries Using an Effective Calculus" (1995)
- Buneman, "Comprehensions" (1996)

**Graph pattern matching:**
- Angles, Gutierrez, "Foundations of Modern Graph Pattern Matching" (2018)
- "GQL: A Query Language for Property Graphs" (ISO/IEC 39075) — formal semantics
- "SQL/PGQ" (ISO/IEC 9075-16) — formal semantics

**Top-k and similarity:**
- Ilyas, Beskales, Soliman, "Top-k Query Processing" (2008)
- Böhm, Kriegel, "The k-NN Join" (2001)

**Query optimization:**
- Graefe, "The Cascades Framework for Query Optimization" (1995)
- Graefe, McKenna, "The Volcano Optimizer Generator" (1993)
- Leis, et al., "Cardinality Estimation" (2015)

**Approximation and probability:**
- Chaudhuri, et al., "Approximate Query Processing"
- Suciu, Olteanu, Koch, "Probabilistic Databases" (2011)
- Exqutor paper (2024) — vector selectivity
- ACORN, NaviX, JAG, iFVS papers (2024) — vector filter strategies

**Type theory:**
- Pierce, "Types and Programming Languages" (2002)
- Wand, "Type Inference for Record Concatenation and Multiple Inheritance" (1991) — row polymorphism

---

## Format of Research Output

Each question should produce a research note in `docs/sql-research/` with the naming pattern `foundations-<topic>-<mm-dd-yyyy>.md`. Each note should:

1. State the question
2. Define terms formally
3. Cite specific sources (papers, standards, textbook chapters)
4. Provide a formal answer (definitions, theorems, proofs where appropriate)
5. End with a recommendation and any open sub-questions

After Q1–Q7 are resolved, a synthesis document integrates them into a unified formal foundation. That foundation is what the engineering work in the previous research prompt sits on.
