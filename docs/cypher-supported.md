# Cypher and Graph Query Support

LibraVDB provides a focused, production-oriented Cypher-style graph surface
inside its unified SQL engine. It supports the graph query forms needed for
connected-data retrieval, recommendation workloads, semantic graph search,
and Graphiti-style entity and relationship management.

This document is the authoritative reference for LibraVDB's Cypher-style
syntax. It describes the implemented surface; it is not a claim of complete
openCypher, GQL, or Neo4j compatibility. SQL and PostgreSQL compatibility are
described separately in [Supported SQL](sql-supported.md).

## Execution model

Graph queries execute over graph-backed LibraVDB collections. A collection
record has a durable graph-node identity, and graph relationships connect
those node identities. The same storage engine owns:

- record metadata and vectors;
- graph adjacency, reverse adjacency, edge types, weights, and properties;
- WAL durability and recovery;
- epoch transactions and snapshot visibility;
- native SQL and PostgreSQL-wire execution.

Cypher-style syntax is therefore another query surface over the unified
engine, not a separate graph database or an application-side translation
layer. Graph traversal and relational/vector expressions can be combined in a
single execution path where the form is supported.

## Access paths and clients

| Access path | Example entry point | Parameters | Result behavior |
| --- | --- | --- | --- |
| Native Go | `Database.Query` | none | `SearchResults` |
| Native Go | `Database.QueryWithParams` | named `$name` or `@name` | typed `SearchResults` |
| Epoch transaction | `EpochTx.Query` | named parameters | transaction-visible results |
| PostgreSQL wire | `pgwire.NewServer` | positional `$1`, `$2`, … | PostgreSQL rows and types |
| Go clients | `pgx`, `database/sql` | driver-native positional parameters | standard `Rows`/`Scan` behavior |
| GORM | PostgreSQL dialector with `Raw`/`Scan` | driver-native positional parameters | ORM-managed row projections |
| Other PostgreSQL clients | psycopg, asyncpg, SQLAlchemy, Django | driver-native positional parameters | supported pgwire result contract |

Native Go parameter names do not include the `$` or `@` prefix in the
`QueryParams` map:

```go
rows, err := db.QueryWithParams(ctx, `
    MATCH (source:Person)-[:FOLLOWS]->(target:Person)
    RETURN source.id AS source_id, target.id AS target_id
    LIMIT $limit`, libravdb.QueryParams{
    "limit": 10,
})
```

Over pgwire, use the normal PostgreSQL positional form:

```go
rows, err := sqlDB.QueryContext(ctx, `
    MATCH (source:Person)-[:FOLLOWS]->(target:Person)
    RETURN source.id AS source_id, target.id AS target_id
    LIMIT $1`, 10)
```

Parameters are bound values, not interpolated SQL text. Vector parameters are
accepted as typed vector values, and scalar parameters can be used in vertex
property maps, predicates, projections, `WITH` expressions, and
`LIMIT`/`OFFSET` clauses.

## Graph schema and bootstrap

### Graph-backed collections

SQL can create a graph-backed collection and register relationship kinds:

```sql
CREATE GRAPH TABLE people (
    id   TEXT PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE EDGE TYPE FOLLOWS;

INSERT INTO people (id, name) VALUES ('alice', 'Alice');
INSERT INTO people (id, name) VALUES ('bob', 'Bob');
INSERT INTO GRAPH_EDGES VALUES ('alice', 'FOLLOWS', 'bob');
```

`CREATE GRAPH TABLE` creates a collection whose records also have graph-node
identities. `CREATE EDGE TYPE` creates the durable logical name-to-kind
mapping used by typed patterns. `GRAPH_NODES` is a read-only virtual relation;
it is not a table into which application rows should be inserted.

Vertex labels are explicit graph metadata; they are not inferred from a SQL
table name or from an arbitrary metadata field. Register them through the
native graph API or `EpochTx.RegisterVertexLabel`. A `MERGE` pattern containing
a label also assigns that label to the matched or newly created vertex:

```sql
MERGE (person:Person {uuid: $uuid})
SET person.name = $name;
```

After labels have been registered, label predicates such as `(person:Person)`
are available to native `MATCH`, `JOIN MATCH`, and `GRAPH_TABLE` queries.

The equivalent native Go setup uses `CreateCollection(..., WithGraph(...))`
and the graph registration APIs. Both setup paths produce the same graph
storage and query behavior.

### Directed and undirected relationships

Edge types are directed by default:

```sql
CREATE EDGE TYPE FOLLOWS;
```

An undirected type is traversable from either endpoint:

```sql
CREATE EDGE TYPE KNOWS UNDIRECTED;
INSERT INTO GRAPH_EDGES VALUES ('alice', 'KNOWS', 'bob');
```

The undirected relationship is stored once with canonical endpoints. Reverse
adjacency supplies the opposite traversal; LibraVDB does not duplicate the
edge, WAL record, history entry, or property block. `-` in a pattern requests
undirected traversal for that pattern, while `->` and `<-` request directed
traversal.

## Query entry points

LibraVDB supports three graph-query forms. They share the same pattern
grammar and graph executor, but their surrounding clauses differ.

### Native top-level `MATCH`

Native Cypher-style statements start with `MATCH` and end with `RETURN`:

```sql
MATCH (source:Person)-[:FOLLOWS]->(target:Person)
WHERE source.id = $origin_id
RETURN source.id AS source_id,
       target.id AS target_id
ORDER BY target_id
LIMIT $limit;
```

The graph-backed collection is resolved from the graph pattern and its labels,
property schema, and relationship type. This form is intended for graph-first
queries where the pattern is the primary source relation.

### SQL `JOIN MATCH` and `OPTIONAL MATCH`

SQL can use an ordinary table or collection as the source relation:

```sql
SELECT source.id AS source_id, target.id AS target_id
FROM people AS source
JOIN MATCH (source)-[:FOLLOWS]->(target)
WHERE source.id = $origin_id;
```

`OPTIONAL MATCH` is supported in this SQL form and has left-outer semantics:
source rows without a matching graph path remain in the result with NULL
terminal values.

```sql
SELECT source.id AS source_id, target.id AS target_id
FROM people AS source
OPTIONAL MATCH (source)-[:FOLLOWS]->(target)
ORDER BY source_id;
```

### `GRAPH_TABLE`

`GRAPH_TABLE` provides a graph-first relation inside a normal SQL statement:

```sql
SELECT source.id, target.id
FROM GRAPH_TABLE(
    people MATCH (source)-[:FOLLOWS]->(target)
)
ORDER BY source.id, target.id;
```

Use `JOIN MATCH` when the relational table is the driving source. Use
`GRAPH_TABLE` when the graph pattern is the driving relation and should be
composed with ordinary SQL projections, filters, grouping, or joins.

## Pattern grammar

### Vertices

| Form | Meaning |
| --- | --- |
| `(n)` | anonymous vertex bound to `n` |
| `()` | anonymous vertex without a binding |
| `(n:Person)` | vertex `n` with the `Person` label |
| `(n:Person:Active)` | vertex with all listed labels; labels are conjunctive |
| `(n {uuid: $id})` | vertex property predicate |
| `(n:Person {uuid: $id})` | label and property predicate |

Vertex property maps use scalar expressions and bound parameters:

```sql
MATCH (person:Person {uuid: $person_id})
RETURN person.id, person.name;
```

Labels are graph vertex labels. They are filters over graph metadata and are
not ordinary SQL column names. A vertex must satisfy every label listed in a
multi-label pattern.

### Relationships

| Form | Meaning |
| --- | --- |
| `-[r]->` | directed relationship bound to `r` |
| `-[:FOLLOWS]->` | directed relationship of a registered type |
| `-[r:FOLLOWS]->` | typed relationship bound to `r` |
| `-[]->` | anonymous relationship of any type |
| `<-[:FOLLOWS]-` | reverse directed traversal |
| `-[:KNOWS]-` | undirected traversal |

Relationship aliases expose the edge property and stable endpoint fields
described below. Relationship types are registered names, not arbitrary
runtime strings; an unknown type is rejected rather than silently matching
zero rows.

### Variable-length paths

The following forms are supported:

```sql
-- Bounded range, one through three hops.
MATCH (a)-[:FOLLOWS*1..3]->(b)
RETURN a.id, b.id;

-- One or more hops.
MATCH (a)-[:FOLLOWS+]->(b)
RETURN a.id, b.id;

-- Zero or more hops.
MATCH (a)-[:FOLLOWS*]->(b)
RETURN a.id, b.id;

-- Arrow quantifier spelling.
MATCH (a)-[:FOLLOWS]->{1,3}(b)
RETURN a.id, b.id;
```

Bounds are applied during traversal. This limits graph expansion work rather
than filtering an unbounded result after materialization.

### Edge predicates

Edge predicates can be attached to a bracketed relationship:

```sql
MATCH (a)-[r:RELATES {weight > 0.5}]->(b)
RETURN a.id, b.id, r.weight;
```

The edge-local `WHERE` form is equivalent:

```sql
MATCH (a)-[r:RELATES WHERE r.weight > $minimum_weight]->(b)
RETURN a.id, b.id;
```

Anonymous directed relationships can carry a property predicate directly:

```sql
MATCH (a)-{weight > 0.5}->(b)
RETURN a.id, b.id;
```

Predicates support the scalar comparison and boolean expressions available to
the unified SQL expression evaluator. Edge properties are JSON-compatible
values stored with the edge and are evaluated against the visible graph
version.

## Projections and result shapes

### Vertex and edge projections

Native Cypher projections use the bound aliases:

```sql
MATCH (a:Person)-[r:FOLLOWS]->(b:Person)
RETURN a.id AS source_id,
       b.id AS target_id,
       r.type AS edge_type,
       r.weight AS edge_weight;
```

For SQL `JOIN MATCH`, LibraVDB exposes stable virtual columns for single-hop
rows:

| Column | Type | Definition |
| --- | --- | --- |
| `source_id` | `TEXT` | source vertex record ID |
| `target_id` | `TEXT` | target vertex record ID |
| `edge_type` | `TEXT` | registered logical relationship name |
| `edge_weight` | `REAL` | durable relationship weight |

These columns are derived from the graph match and are not stored metadata.
Applications should project them instead of parsing the legacy composite result
ID (`source|target`). Stable endpoint and edge columns are available through
native SQL and pgwire.

### Path variables and `GraphPath`

A path can be bound before the pattern:

```sql
MATCH p = (a)-[:FOLLOWS]->(b)-[:FOLLOWS]->(c)
RETURN p;
```

Native Go returns a `libravdb.GraphPath` value containing ordered node IDs,
edge types, and edge weights. Through pgwire, the path is serialized as a
JSON-compatible value with the corresponding `nodes`, `edge_types`, and
`edge_weights` fields.

### `shortestPath`

Bounded shortest-path expressions are supported:

```sql
MATCH shortestPath((a)-[:FOLLOWS*1..3]->(b))
RETURN a.id AS source_id, b.id AS target_id;
```

The path can also be named and returned:

```sql
MATCH p = shortestPath((a)-[*1..3]->(b))
RETURN p;
```

The bounded pattern controls the traversal domain. An unbounded shortest-path
request should be expressed with an explicit operational bound.

### Pattern comprehensions

Pattern comprehensions produce an array from matching terminal values:

```sql
MATCH (person)
RETURN [(person)-[:FOLLOWS]->(friend) | friend.id] AS friend_ids;
```

The pattern may include a predicate before the projection expression:

```sql
MATCH (person)
RETURN [
    (person)-[:FOLLOWS]->(friend)
    WHERE friend.active = true | friend.id
] AS active_friend_ids;
```

Through pgwire, the result is returned as a JSON-compatible array value.

## `WITH` pipelines

Native Cypher `WITH` is a projection boundary. It replaces the current binding
scope with the projected values and allows subsequent filtering, ordering,
limiting, aggregation, or another graph match.

Supported features include:

- expression projections with `AS` aliases;
- `WITH *` pass-through;
- `DISTINCT`;
- scalar predicates after `WITH`;
- `ORDER BY`, `SKIP`, and parameterized `LIMIT`;
- grouped aggregates such as `count(n)`;
- multiple chained `WITH` clauses;
- `MATCH` after a `WITH` boundary;
- vector similarity expressions evaluated over the bound vertex vector.

Example:

```sql
MATCH (n:Person {group_id: $group_id})
WITH n,
     array_cosine_similarity(n.vector, $query_vector) AS score
WHERE score >= $minimum_score
RETURN n.id AS person_id, score
ORDER BY score DESC
LIMIT $limit;
```

`SIMILARITY(vector, query_vector)`,
`ARRAY_COSINE_SIMILARITY(vector, query_vector)`, and
`VECTOR_DISTANCE(vector, query_vector)` use the existing vector execution
path. `ARRAY_COSINE_SIMILARITY` is the Graphiti/Kuzu-compatible spelling of
the max-similarity operation.

Aggregation and chained matching are also supported:

```sql
MATCH (n:Person)
WITH n.group_id AS group_id, count(n) AS member_count
WHERE member_count > $minimum_members
RETURN group_id, member_count
ORDER BY member_count DESC;
```

```sql
MATCH (n:Person)
WITH n.id AS origin_id
MATCH (origin)-[:FOLLOWS]->(friend)
WHERE origin.id = origin_id
RETURN origin_id, friend.id AS friend_id;
```

`WITH` here is a Cypher pipeline clause. A SQL CTE such as `WITH recent AS
(SELECT ...)` is a separate SQL feature and is documented in
[Supported SQL](sql-supported.md).

## Graph mutation statements

### `MERGE`

`MERGE` is a top-level graph upsert statement. It finds or creates the vertices
and relationship described by the pattern and is idempotent for an existing
matching graph pattern:

```sql
MERGE (person:Person {uuid: $uuid})
SET person.name = $name;
```

Single-hop relationship creation and relationship properties are supported:

```sql
MERGE (a:Person {uuid: $from_uuid})
      -[r:KNOWS {weight: $weight}]->
      (b:Person {uuid: $to_uuid})
SET r.fact = $fact;
```

Conditional assignment blocks are supported:

```sql
MERGE (person:Person {uuid: $uuid})
ON CREATE SET person.created_at = $created_at
ON MATCH  SET person.updated_at = $updated_at;
```

Plain `SET` applies to both create and match outcomes. If both a plain `SET`
and a conditional block assign the same field, evaluation order is:

1. plain `SET`;
2. `ON CREATE SET` when the pattern was created;
3. `ON MATCH SET` when the pattern matched.

Vertex and relationship aliases may be assignment targets. The entire merge,
including record metadata, graph nodes, and relationships, is committed as one
mutation path.

### `DELETE` and `DETACH DELETE`

Native graph deletion starts with a `MATCH` pattern:

```sql
MATCH (a)-[r:KNOWS]->(b)
DELETE r;
```

Deleting a vertex without `DETACH` fails if it still has incident edges. Use
`DETACH DELETE` to remove the vertex and all incident relationships atomically:

```sql
MATCH (person:Person {uuid: $uuid})
DETACH DELETE person;
```

Relationship deletion leaves both endpoint records intact. A failed delete
does not partially mutate the graph or record state. The current native
Cypher delete surface accepts one matched graph pattern per statement; ordinary
relational `DELETE FROM ...` remains available for SQL tables.

## Common-neighbor and semijoin queries

Common-neighbor traversal is available through SQL `JOIN MATCH` and can be
materialized as an ordinary relational semijoin:

```sql
SELECT p.id, p.metadata
FROM people AS p
WHERE p.id IN (
    SELECT src.id
    FROM people AS src
    JOIN MATCH (src)-[]->(shared)
    JOIN MATCH (origin)-[]->(shared)
    WHERE origin.id = $origin_id
      AND src.id <> $origin_id
)
ORDER BY p.id;
```

The specialized plan traverses the graph once per side, deduplicates candidate
IDs, and probes the outer relation. It preserves ordinary predicates,
ordering, limits, and the active epoch or historical snapshot.

When recommendation or audit workflows require the evidence instead of only
candidate IDs, use `GRAPH_SEMIJOIN`:

```sql
SELECT candidate_id, evidence_id, edge_type, shared_count
FROM GRAPH_SEMIJOIN(
    'people', $origin_id, 'FOLLOWS',
    $source_expansion_limit,
    $origin_expansion_limit,
    $candidate_limit
)
ORDER BY candidate_id, evidence_id;
```

The result contains the candidate, the shared evidence node, the logical edge
type, and the count of shared neighbors. Expansion and candidate limits bound
the graph work and result size.

## Temporal and transactional behavior

Graph queries executed inside an epoch transaction see the transaction's
record and graph overlay. Record and graph mutations performed through the
same supported transaction path are published together after durable commit.

Historical graph reads are expressed through the SQL/PGQ relation form, where
the source relation carries the snapshot clause:

```sql
SELECT source.id, target.id
FROM people AS OF LSN $snapshot_lsn AS source
JOIN MATCH (source)-[:FOLLOWS]->(target);
```

`AS OF TIMESTAMP` is also supported. The graph traversal, record predicates,
vector projections, aggregates, CTEs, and semijoins in the statement use the
same resolved snapshot when the query shape supports historical execution.
Exact LSN reads are preferred when several independent reads must observe one
commit boundary.

## Explain and observability

`EXPLAIN ANALYZE` is supported for graph queries. Plain `EXPLAIN` is not a
supported execution form; use `EXPLAIN ANALYZE` when query-local measurements
are required.

```sql
EXPLAIN ANALYZE
SELECT DISTINCT src.id
FROM people AS src
JOIN MATCH (src)-[]->(shared)
JOIN MATCH (origin)-[]->(shared)
WHERE origin.id = $origin_id
  AND src.id <> $origin_id;
```

The stable JSON result contains:

| Field | Meaning |
| --- | --- |
| `strategy` | graph execution strategy, such as `graph_join_match` |
| `anchor` | starting graph alias when available |
| `actual_rows` | rows returned by the analyzed query |
| `graph_expansions` | graph neighbor expansions performed |
| `predicate_rejections` | rows rejected by graph/relational predicates |
| `index_hits` | index probes recorded during execution |
| `execution_time_ns` | query execution duration in nanoseconds |
| `plan_reused` | whether a cached plan was reused |

Estimated cardinalities are intentionally not reported until the graph planner
has a real cardinality estimator.

## Compatibility boundaries

The following boundaries are deliberate:

- The implementation is a focused Cypher-style surface, not complete
  openCypher or GQL.
- Native top-level graph statements start with `MATCH` and require `RETURN`,
  `DELETE`, or `DETACH DELETE`; `MERGE` is the separate top-level upsert form.
- `OPTIONAL MATCH` is supported in the SQL `FROM ... OPTIONAL MATCH` form;
  standalone top-level `OPTIONAL MATCH` is not a separate entry statement.
- General arbitrary clause reordering, such as `MATCH ... MERGE ...`, is not
  implied by standalone `MATCH` and standalone `MERGE` support.
- SQL `DELETE FROM` and SQL `INSERT INTO GRAPH_EDGES` remain the SQL mutation
  forms; native graph deletion uses `MATCH ... DELETE` or `DETACH DELETE`.
- Relationship type names are registered graph kinds and are not arbitrary
  parameter values.
- Path traversal should be bounded for predictable work. Unbounded `*` and
  `+` forms are accepted where listed, but their expansion cost is determined
  by the graph and the active limits.
- Result ordering is guaranteed only when an explicit `ORDER BY` is supplied.

## Verification coverage

The supported surface is regression-tested through the native Go SQL suite and
the PostgreSQL-wire compatibility harness. Coverage includes:

- typed and multi-label vertices;
- directed and undirected relationships;
- edge predicates and anonymous edge property syntax;
- variable-length and shortest paths;
- path variables and pattern comprehensions;
- `OPTIONAL MATCH` through SQL/PGQ;
- idempotent `MERGE`, universal and conditional `SET`;
- relationship and `DETACH DELETE` behavior;
- `WITH` projection, aggregation, filtering, ordering, and chaining;
- vector similarity inside graph pipelines;
- stable endpoint/edge projections;
- common-neighbor semijoins and evidence rows;
- query-local `EXPLAIN ANALYZE` metrics;
- native Go, pgx/database/sql, GORM, and PostgreSQL client flows.

For implementation-level examples, see the graph and pgwire regression tests
under `libravdb/*graph*test.go` and `internal/pgwire/graph_ddl_test.go`.
