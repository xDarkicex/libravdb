# Supported SQL

LibraVDB provides a SQL surface for relational data, vector search, graph
traversal, and temporal snapshots. The dialect is intentionally focused: it
uses familiar PostgreSQL syntax where that syntax maps cleanly to LibraVDB's
storage and execution model, and adds first-class SQL operators for vectors,
graphs, and time.

This page is the public SQL compatibility reference. A statement is listed as
supported only when it is implemented by the executor and covered by the
repository's native SQL tests or the external PostgreSQL-wire compatibility
matrix. PostgreSQL compatibility outside the surface described here should not
be assumed.

## SQL surfaces

The same SQL execution model is available through two public paths.

| Surface | Typical entry point | Parameter convention | Best suited for |
| --- | --- | --- | --- |
| Native Go SQL | `Database.Query`, `Database.QueryWithParams` | Named `$name` or `@name` parameters | Embedded applications and low-latency in-process queries |
| Native epoch SQL | `EpochTx.Query`, `SQLSession` | Named parameters; transaction-local state | Atomic record and graph changes |
| PostgreSQL wire | `pgwire.NewServer` / `pgwire.Serve` | PostgreSQL positional `$1`, `$2`, … | Existing PostgreSQL drivers and applications |

Native parameters are supplied through `libravdb.QueryParams`. The parameter
map key does not include the `$` or `@` prefix:

```go
rows, err := db.QueryWithParams(ctx, `
    SELECT id, VECTOR_DISTANCE(embedding, $query_vector) AS distance
    FROM documents
    ORDER BY distance
    LIMIT $limit`, libravdb.QueryParams{
    "query_vector": []float32{0.1, 0.2, 0.3},
    "limit":        10,
})
```

Over PostgreSQL wire, use the driver's normal positional parameters:

```sql
SELECT id, VECTOR_DISTANCE(embedding, $1) AS distance
FROM documents
ORDER BY distance
LIMIT $2;
```

Parameters remain typed values. LibraVDB does not implement parameter binding
by interpolating values into SQL text.

### Parameterized JSONB and upserts

Decoded JSON objects may be passed directly as a bound parameter. Use an
explicit `::json` or `::jsonb` cast at the SQL boundary so the parameter has
the document type expected by the statement:

```go
profile := map[string]interface{}{
    "name": "Ada Lovelace",
    "roles": []interface{}{"admin", "owner"},
    "settings": map[string]interface{}{"alerts": true},
}

_, err := db.QueryWithParams(ctx, `
    INSERT INTO people (id, metadata, vector)
    VALUES ($1, $2::jsonb, $3)
    ON CONFLICT (id) DO UPDATE SET
        metadata = EXCLUDED.metadata,
        vector = EXCLUDED.vector`, libravdb.QueryParams{
    "1": "person-1",
    "2": profile,
    "3": []float32{0.1, 0.2, 0.3},
})
```

The same SQL and parameter contract is available through every SDK that uses
the shared `DatabaseQueryWithParams` binding. Python dictionaries, Rust JSON
values, Java maps, and equivalent structured values are serialized by the SDK
binding and decoded once at the common Go boundary. JSON documents remain
JSON values through `INSERT`, `EXCLUDED`, `ON CONFLICT DO UPDATE`,
`RETURNING`, WAL recovery, and reopen. Top-level numeric arrays retain their
existing vector meaning; pass a top-level JSON array as JSON text (for
example, a JSON string or `json.RawMessage`) together with `::jsonb`.

`ON CONFLICT DO NOTHING`, conflict predicates, and expressions over
`EXCLUDED` values use the same typed path. This is the recommended way for an
SDK application to replace a native collection upsert when its record has
nested JSON metadata.

## Client compatibility

The following client paths are part of the supported wire surface for the SQL
features documented here.

| Client or framework | Transport | Supported scope |
| --- | --- | --- |
| Go `database/sql` with pgx | pgwire | Startup, simple and extended queries, prepared statements, DDL, DML, transactions, typed results |
| Direct `pgx` | pgwire | Extended protocol, binary parameters/results, portals, `RETURNING`, vector and JSON values |
| GORM PostgreSQL dialector | pgwire through pgx | Catalog discovery, `AutoMigrate`, typed CRUD, qualified projections, `Raw`, and `VECTOR_AVG` |
| psycopg 3 and psycopg2 | pgwire | Driver startup, parameter binding, DDL/DML, JSONB, catalog access, pipelines, cursors, and savepoints as applicable |
| asyncpg | pgwire | Prepared-statement reuse, batching, portal suspension, pool reset, JSONB, catalog access, and DML |
| SQLAlchemy 2 | pgwire through psycopg | Reflection, ORM relationships, eager loading, PostgreSQL upserts, JSONB expressions, and bulk DML |
| Alembic | pgwire through SQLAlchemy | Catalog comparison and tested migration upgrade/downgrade operations |
| Django PostgreSQL backend | pgwire through psycopg | Migrations, JSONField CRUD/filtering, identity primary keys, and `inspectdb` catalog introspection |

This matrix demonstrates compatibility for the listed operations. It is not a
claim that LibraVDB implements every PostgreSQL extension, system catalog, or
driver-specific feature.

The tested protocol behaviors include repeated prepared execution after schema
changes, batched `executemany`/pipeline writes, suspended portals for bounded
cursor fetches, connection-pool reset, nested savepoints, binary JSONB/vector
values, and typed `Describe` metadata. SQLAlchemy relationship reflection and
eager loading, Alembic migration comparison and execution, GORM schema
discovery/`AutoMigrate`, and Django migration/`inspectdb` flows are supported
for the catalog and SQL operations listed in this page.

## Data definition language

### Tables and column types

`CREATE TABLE` supports ordinary relational tables and tables backed by
LibraVDB collections. The following types are supported in SQL table
definitions and metadata schemas:

| Type family | Examples | Notes |
| --- | --- | --- |
| Text | `TEXT`, `VARCHAR(n)`, `CHAR`, `STRING` | `VARCHAR(n)` length metadata is retained for catalog clients |
| Boolean | `BOOLEAN`, `BOOL` | SQL `TRUE`, `FALSE`, and `NULL` semantics are preserved |
| Integers | `SMALLINT`, `INTEGER`, `INT`, `BIGINT`, serial variants | Integer values retain their PostgreSQL-compatible wire types |
| Floating point | `FLOAT`, `DOUBLE PRECISION` | Scientific notation is accepted and validated |
| Time | `TIMESTAMP`, `TIMESTAMPTZ`, `DATE`, `TIME` | Used by temporal predicates and ordinary metadata columns |
| Identifiers | `UUID` | UUID values are validated and exposed with PostgreSQL UUID metadata |
| Documents | `JSON`, `JSONB` | JSON values are validated, canonicalized, and persisted |
| Vectors | `VECTOR(n)` | The dimension is required and must be positive |

Example:

```sql
CREATE TABLE documents (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    category    TEXT DEFAULT 'general',
    score       FLOAT,
    embedding   VECTOR(384),
    payload     JSONB,
    created_at  TIMESTAMPTZ
);
```

`CREATE TABLE` also supports multiline statements, quoted identifiers, inline
and table-level constraints, named constraints, and the identity/serial forms
used by common PostgreSQL ORMs.

Column defaults use supported literal values and are applied on insert before
`NOT NULL`, `CHECK`, foreign-key, and uniqueness validation. Arbitrary
expression defaults are not part of the current contract.

### Constraints

The supported constraint forms are:

| Constraint | Supported behavior |
| --- | --- |
| `PRIMARY KEY` | Single-column and composite keys; primary-key uniqueness and identity mapping |
| `UNIQUE` | Column and table constraints; composite unique keys |
| `NOT NULL` | Enforced on direct writes, transactions, and epoch overlays |
| `DEFAULT` | Applied before constraint validation |
| `CHECK` | Boolean expressions with SQL three-valued logic |
| `FOREIGN KEY` | Single-column and composite references |
| Referential actions | `CASCADE`, `RESTRICT`, `NO ACTION`, `SET NULL`, and `SET DEFAULT` where the source column definition permits the action |

```sql
CREATE TABLE authors (
    id       TEXT PRIMARY KEY,
    email    TEXT NOT NULL,
    CONSTRAINT authors_email_unique UNIQUE (email),
    name     TEXT NOT NULL
);

CREATE TABLE documents (
    id          TEXT PRIMARY KEY,
    author_id   TEXT NOT NULL,
    score       FLOAT DEFAULT 0,
    CONSTRAINT documents_author_fk
        FOREIGN KEY (author_id) REFERENCES authors(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CONSTRAINT score_range CHECK (score >= 0 AND score <= 1)
);
```

Constraint metadata is reflected through the virtual PostgreSQL catalog. The
constraint itself is enforced against the authoritative row and transaction
state; catalog metadata is not a separate user-data store.

### Indexes and schema changes

The following DDL is supported:

```sql
CREATE INDEX documents_category_idx ON documents (category);
CREATE UNIQUE INDEX authors_email_idx ON authors (email);

ALTER TABLE documents ADD COLUMN language TEXT DEFAULT 'en';
ALTER TABLE documents DROP COLUMN language;

DROP INDEX documents_category_idx;
DROP TABLE documents;
```

Indexes and column definitions survive close/reopen. Dropping a primary-key or
constrained column is rejected rather than silently invalidating dependent
data. Adding a vector column with `ALTER TABLE` is not supported; create the
table with its vector schema or create a new collection.

JSON path indexes are supported for one extracted path:

```sql
CREATE INDEX documents_active_idx
ON documents (payload#>>'{profile,active}');

SELECT id
FROM documents
WHERE payload#>>'{profile,active}' = 'true';
```

The index definition is durable. Its candidate postings are derived from the
authoritative records and can be rebuilt after reopen; the executor performs a
final JSON predicate check for correctness.

## Data manipulation language

### INSERT

Single-row, multi-row, and parameterized inserts are supported:

```sql
INSERT INTO authors (id, email, name)
VALUES ('a1', 'ada@example.com', 'Ada');

INSERT INTO authors (id, email, name)
VALUES
    ('a2', 'grace@example.com', 'Grace'),
    ('a3', 'alan@example.com', 'Alan');

INSERT INTO documents (id, author_id, title, embedding)
VALUES ($1, $2, $3, $4);
```

String literals may contain commas and escaped quotes in multi-row values. JSON
and JSONB literals are parsed as one value even when they contain nested commas.

### UPDATE

Updates support predicates, arithmetic, casts, `CASE`, `NULLIF`, bit shifts,
concatenation, and boolean expressions:

```sql
UPDATE documents
SET title = title || ' (reviewed)',
    score = CASE WHEN score < 1 THEN score + 0.1 ELSE score END
WHERE id = 'd1';

UPDATE task_flags
SET completed = NOT completed
WHERE id = 'd2';
```

Unary boolean expressions are available in `SET` expressions as well as
predicates.

JSONB mutation functions are also valid in `UPDATE SET` expressions. The
mutation is evaluated against each visible row and committed through the same
atomic transaction path as other updates; a parameter may supply the JSON
replacement value:

```sql
UPDATE people
SET metadata = jsonb_set(metadata, '{career}', '[]'::jsonb, true)
WHERE jsonb_typeof(metadata->'career') = 'string';

UPDATE people
SET metadata = jsonb_set(metadata, '{skills}', $1::jsonb, true)
WHERE id = $2;
```

`jsonb_set`, `json_set`, `jsonb_insert`, and `json_insert` preserve the JSON
document shape through native Go SQL, epoch transactions, pgwire, WAL
recovery, and reopen. JSON-aware predicates in the `WHERE` clause use the
same decoded document evaluator as JSON projections.

### DELETE

```sql
DELETE FROM documents
WHERE category = 'temporary';
```

Deletes enforce referential actions and update graph identity state when a row
is graph-backed.

### `RETURNING`

`RETURNING` is supported for `INSERT`, `UPDATE`, `DELETE`, and upsert forms.
Explicit projections preserve the requested order; `RETURNING *` uses the
table's deterministic schema order.

```sql
INSERT INTO documents (id, title)
VALUES ('d1', 'First document')
RETURNING id, title;

UPDATE documents
SET title = 'Updated document'
WHERE id = 'd1'
RETURNING *;

DELETE FROM documents
WHERE id = 'd1'
RETURNING id, title;
```

### Upserts and conflict actions

Supported conflict targets include a column list and a named unique
constraint. Both `DO UPDATE` and `DO NOTHING` are supported, including
`EXCLUDED`, expressions, casts, `CASE`, `WHERE`, and `RETURNING`.

```sql
INSERT INTO authors (id, email, name)
VALUES ('a1', 'ada@example.com', 'Ada Lovelace')
ON CONFLICT ON CONSTRAINT authors_email_unique DO UPDATE
SET name = EXCLUDED.name
RETURNING id, name;

INSERT INTO authors (id, email, name)
VALUES ('a1', 'ada@example.com', 'ignored')
ON CONFLICT (id) DO NOTHING;

INSERT INTO authors (id, email, name)
VALUES ('a2', 'ada@example.com', 'Ada Byron')
ON CONFLICT ON CONSTRAINT authors_email_unique DO UPDATE
SET name = EXCLUDED.name
WHERE authors.name <> EXCLUDED.name;
```

The named constraint example uses unique-constraint metadata that PostgreSQL
clients expect. Foreign-key constraints are not conflict targets.

### `INSERT ... SELECT`

```sql
INSERT INTO document_archive (id, title)
SELECT id, title
FROM documents
WHERE category = 'archived';
```

The source query is evaluated before the target writes are applied.

### SQL prepared statements

SQL-level prepared statements are connection-local and are separate from the
wire protocol's `Parse`/`Bind`/`Execute` messages. They are supported by the
native SQL session and pgwire paths:

```sql
PREPARE bump AS
    INSERT INTO prepared_rows (id, value)
    VALUES ($1, $2)
    ON CONFLICT (id) DO UPDATE
    SET value = prepared_rows.value + EXCLUDED.value;

EXECUTE bump('p1', 4);
DEALLOCATE bump;
```

`DEALLOCATE ALL` clears SQL-level prepared statements and connection-local
portals. Extended-protocol prepared statements support typed parameters,
statement reuse, `Describe`, and portal execution independently of this SQL
syntax.

### Physical plan reuse and SQL metrics

The database maintains a bounded cache of compiled physical plans for ordinary
relational `SELECT` statements whose plan contains no request-bound parameter
values or virtual execution state. Cache entries are keyed by SQL text and the
catalog generation used during binding. Creating or altering a table publishes
a new immutable catalog generation, which makes older entries ineligible
immediately; the old plan is never used against the new schema.

Parameterized statements remain fully supported. Common scalar `WHERE`
parameters are rebound into cached predicate slots, so repeated prepared reads
reuse the physical plan without carrying stale values between executions.
More complex parameterized virtual, aggregate, graph, and vector shapes still
bind and optimize per execution until they have equivalent explicit slots.

Native applications can read cumulative, concurrency-safe SQL counters:

```go
stats := db.SQLStats()
fmt.Println(stats.Queries, stats.PlanCacheHits, stats.LastExecutionNanos)
db.ResetSQLStats()
```

The same snapshot is available to SQL and pgwire clients as a JSONB scalar:

```sql
SELECT LIBRAVDB_SQL_STATS();
```

The JSON object includes `queries`, `errors`, `plan_cache_hits`,
`plan_cache_misses`, `total_execution_nanos`, `last_execution_nanos`,
`rows_returned`, `rows_examined`, `graph_expansions`, and `index_hits`.
Counters are process-local observability state; they are not persisted in WAL
or included in temporal snapshots. The external PostgreSQL compatibility
harness verifies scalar parameter-slot reuse through native `QueryWithParams`
and pgx-backed `database/sql`, catalog-generation invalidation after
`ALTER TABLE`, and JSONB stats retrieval over pgwire.

### `COPY` over PostgreSQL wire

The pgwire server supports `COPY FROM STDIN` and `COPY TO STDOUT` for supported
tables, including text and CSV forms, CSV headers, explicit columns, and
transaction/epoch staging. This is a wire-protocol capability; the native
in-process SQL API uses `INSERT` or the collection batch APIs instead.

```sql
COPY documents (id, title) FROM STDIN WITH (FORMAT csv, HEADER true);
COPY documents (id, title) TO STDOUT WITH (FORMAT csv, HEADER true);
```

## SELECT

### Projection, aliases, ordering, and pagination

```sql
SELECT d.id,
       d.title AS document_title,
       d.category
FROM documents AS d
WHERE d.category = 'graph'
ORDER BY d.title ASC, d.id ASC
OFFSET 10
LIMIT 20;
```

Supported query features include qualified identifiers, aliases, `SELECT *`,
scalar expressions, deterministic ordering, `OFFSET`, and `LIMIT`. `ORDER BY`
may refer to a projected alias.

### Predicates and scalar expressions

The relational predicate surface includes:

| Category | Supported forms |
| --- | --- |
| Comparisons | `=`, `<>`, `!=`, `<`, `<=`, `>`, `>=` |
| Boolean logic | `AND`, `OR`, `NOT` with SQL NULL behavior |
| Membership | `IN`, `NOT IN` where supported by the expression context |
| Ranges | Inclusive `BETWEEN` and `NOT BETWEEN` forms used by the SQL executor |
| Null tests | `IS NULL`, `IS NOT NULL` |
| Text matching | `LIKE`, `ILIKE` |
| Expressions | Arithmetic, concatenation `||`, bit shifts `<<` and `>>`, casts, `CASE`, `NULLIF`, `NOW()` |

```sql
SELECT id
FROM documents
WHERE category IN ('graph', 'vector')
  AND score BETWEEN 0.2 AND 0.8
  AND title ILIKE '%search%'
  AND payload IS NOT NULL
ORDER BY id;
```

The same predicates and pagination clauses accept typed parameters:

```sql
SELECT id
FROM documents
WHERE category IN ($category_a, $category_b)
  AND score BETWEEN $minimum_score AND $maximum_score
ORDER BY id
OFFSET $skip
LIMIT $take;
```

`LIKE` is case-sensitive. `ILIKE` uses the supported case-folded comparison.
SQL `NULL` is distinct from empty strings, JSON literal `null`, and zero
values.

### Joins

Relational inner, left, right, full, cross, and chained joins are supported:

```sql
SELECT d.id, a.name AS author_name
FROM documents AS d
JOIN authors AS a ON a.id = d.author_id
WHERE a.name = 'Ada'
ORDER BY d.id;

SELECT a.id, d.title
FROM authors AS a
LEFT JOIN documents AS d ON d.author_id = a.id
ORDER BY a.id, d.id;
```

Unmatched sides of outer joins are returned as SQL `NULL`. Derived relations
may be used as join sources where their correlation and join direction are
supported. Correlated `RIGHT JOIN` and `FULL JOIN` against a derived relation
are rejected because their unmatched-side semantics require a stable
right-side relation; use `INNER`, `LEFT`, or `CROSS JOIN` for those correlated
forms.

### DISTINCT and set operations

```sql
SELECT DISTINCT category
FROM documents
ORDER BY category;

SELECT id FROM documents WHERE category = 'graph'
UNION ALL
SELECT id FROM documents WHERE score > 0.9;
```

`UNION`, `UNION ALL`, `INTERSECT`, `INTERSECT ALL`, `EXCEPT`, and `EXCEPT ALL`
are supported. Distinct operations compare the complete projected row.

### Aggregation

The standard scalar aggregates are `COUNT`, `SUM`, `AVG`, `MIN`, and `MAX`.
Grouped aggregation preserves aliases and supports `HAVING`, aggregate ordering,
and expressions over aggregate results:

```sql
SELECT category,
       COUNT(*) AS document_count,
       AVG(score) AS average_score
FROM documents
GROUP BY category
HAVING COUNT(*) > 1
ORDER BY document_count DESC;
```

Aggregate arguments may be parameters, and aggregate results may participate
in arithmetic expressions:

```sql
SELECT namespace,
       SUM(alpha) / SUM(alpha + beta) AS beta_mean,
       MIN($admission_threshold) AS admission_threshold
FROM context_transitions
GROUP BY namespace;
```

The collection aggregates `ARRAY_AGG` and `STRING_AGG` are also supported.
`ARRAY_AGG` preserves SQL NULL elements; `STRING_AGG` skips NULL inputs and
returns SQL `NULL` for an empty input.

```sql
SELECT category,
       ARRAY_AGG(title) AS titles,
       STRING_AGG(title, ' | ') AS title_list
FROM documents
GROUP BY category
ORDER BY category;
```

`VECTOR_AVG` computes a component-wise centroid for non-NULL vectors and
returns a PostgreSQL-compatible float array result:

```sql
SELECT VECTOR_AVG(embedding) AS centroid
FROM documents
WHERE category = 'graph';
```

Ordered-set aggregates `PERCENTILE_CONT`, `PERCENTILE_DISC`, and `MODE` are
supported with `WITHIN GROUP (ORDER BY ...)` for scalar and grouped queries.

```sql
SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY score) AS p50,
       PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY score) AS median_value,
       MODE() WITHIN GROUP (ORDER BY category) AS most_common_category
FROM documents;
```

### Window functions

The following ranking and offset window functions are supported:

`ROW_NUMBER`, `RANK`, `DENSE_RANK`, `PERCENT_RANK`, `CUME_DIST`, `NTILE`,
`LAG`, and `LEAD`.

Aggregate windows support `COUNT`, `SUM`, `AVG`, `MIN`, and `MAX`. Windows may
use multiple `PARTITION BY` and `ORDER BY` expressions, `ASC`/`DESC`, explicit
`NULLS FIRST`/`NULLS LAST`, named `WINDOW` definitions, and supported `ROWS` or
`RANGE` frames. `RANGE` offsets are supported for one numeric ordering
expression; multi-key or nonnumeric offset frames are rejected explicitly.

```sql
SELECT id,
       category,
       score,
       ROW_NUMBER() OVER (
           PARTITION BY category
           ORDER BY score DESC, id
       ) AS category_rank,
       LAG(score) OVER (
           PARTITION BY category
           ORDER BY score
       ) AS previous_score
FROM documents
ORDER BY id;
```

Window expressions may also be evaluated over grouped aggregate results:

```sql
SELECT category,
       COUNT(*) AS document_count,
       ROW_NUMBER() OVER (
           ORDER BY COUNT(*) DESC, category ASC
       ) AS category_rank
FROM documents
GROUP BY category
ORDER BY category;
```

Ordered-set aggregates are not currently ordinary window functions; use them
as grouped or scalar aggregates.

## Common table expressions and subqueries

### Non-recursive CTEs

CTEs are query-local relations. They do not create collections, write catalog
metadata, or generate user-data WAL records.

```sql
WITH recent AS (
    SELECT id, author_id
    FROM documents
    WHERE created_at >= $cutoff
), selected_authors AS (
    SELECT a.id, a.name
    FROM authors AS a
    JOIN recent AS r ON r.author_id = a.id
)
SELECT id, name
FROM selected_authors
ORDER BY id;
```

CTEs are resolved in declaration order. A temporal relation and a bounded
input limit may appear inside a CTE:

```sql
WITH bounded AS (
    SELECT namespace, alpha, beta
    FROM context_transitions AS OF TIMESTAMP $end
    WHERE namespace = $namespace
    ORDER BY id
    LIMIT $input_limit
)
SELECT namespace,
       SUM(alpha) / SUM(alpha + beta) AS beta_mean
FROM bounded
GROUP BY namespace
LIMIT $output_limit;
```

The inner `LIMIT` is applied while materializing the CTE input, before the
outer aggregation.

### Recursive CTEs

Bounded recursive CTE evaluation is supported:

```sql
WITH RECURSIVE tree AS (
    SELECT id, parent_id
    FROM tree_nodes
    WHERE id = 'root'
  UNION ALL
    SELECT child.id, child.parent_id
    FROM tree_nodes AS child
    JOIN tree ON child.parent_id = tree.id
)
SELECT id
FROM tree
ORDER BY id;
```

Evaluation stops when an iteration produces no rows and is protected by the
session recursion limit. The default and maximum recursion limits are finite;
unbounded recursive workloads are not supported.

### Membership, existence, scalar, and derived subqueries

```sql
SELECT d.id
FROM documents AS d
WHERE d.author_id IN (
    SELECT a.id FROM authors AS a WHERE a.name = 'Ada'
);

SELECT d.id,
       (SELECT a.name
        FROM authors AS a
        WHERE a.id = d.author_id
        LIMIT 1) AS author_name
FROM documents AS d;

SELECT r.id, r.category
FROM (
    SELECT id, category
    FROM documents
    WHERE score > 0.8
) AS r
ORDER BY r.id;
```

Correlated predicates retain their outer alias scope. Scalar subqueries return
SQL `NULL` when no row matches; `COUNT` over an empty input returns zero while
the other scalar aggregates return SQL `NULL`.

## Vector search

### Distance functions and operators

`VECTOR_DISTANCE` and `SIMILARITY` can be used in projections, predicates,
ordering, joins, temporal queries, and combined scoring expressions.

The pgvector-style operators are:

| Operator | Meaning | Ordering convention |
| --- | --- | --- |
| `<->` | Squared L2 distance | Smaller is closer |
| `<#>` | Negative inner product | Smaller is a better inner product |
| `<=>` | Cosine distance | Smaller is closer |

```sql
SELECT id,
       VECTOR_DISTANCE(embedding, $query_vector) AS distance
FROM documents
ORDER BY distance ASC, id ASC
LIMIT 10;

SELECT id, embedding <=> $query_vector AS cosine_distance
FROM documents
ORDER BY cosine_distance
LIMIT 10;
```

Vector literals and typed vector parameters are supported. The query vector
must have the same dimension as the column. Vector scores can participate in
arithmetic expressions and aliases used by `ORDER BY`.

### Unified vector and lexical ranking

`FTS_RANK` and `RRF` allow vector and lexical signals to be combined in one
projection:

```sql
SELECT id,
       RRF(
           VECTOR_DISTANCE(embedding, $query_vector),
           FTS_RANK(content, $text_query)
       ) AS relevance
FROM documents
ORDER BY relevance DESC
LIMIT 10;
```

`GRAPH_CENTRALITY` may be included as a third ranking signal for graph-backed
collections.

## Multimodal query composition

Relational predicates can select graph anchors before traversal, and the
result can be ranked by vector distance, lexical relevance, or graph
centrality in one query:

```sql
SELECT d.title,
       VECTOR_DISTANCE(d.embedding, $query_vector) AS distance
FROM documents AS d
JOIN authors AS a ON a.id = d.author_id
JOIN MATCH (d)-[:RELATES]->(target)
WHERE a.name = $author
ORDER BY distance, d.id
LIMIT 10;
```

The same query can use a unified reciprocal-rank score:

```sql
SELECT d.id,
       RRF(
           VECTOR_DISTANCE(d.embedding, $query_vector),
           FTS_RANK(d.content, $text_query),
           GRAPH_CENTRALITY(d)
       ) AS unified_relevance
FROM documents AS d
WHERE MATCH (d)<-[:CITES]-(reference)
ORDER BY unified_relevance DESC
LIMIT 10;
```

Temporal forms of these queries use the same resolved snapshot for the
relational row, vector value, and graph adjacency.

## Full-text search

The supported full-text subset includes `to_tsvector`, `to_tsquery`,
`plainto_tsquery`, `phraseto_tsquery`, `websearch_to_tsquery`, the `@@`
operator, and `ts_rank`/`FTS_RANK` scoring.

```sql
SELECT id,
       ts_rank(
           to_tsvector('english', content),
           plainto_tsquery('english', $query)
       ) AS rank
FROM documents
WHERE to_tsvector('english', content) @@
      plainto_tsquery('english', $query)
ORDER BY rank DESC;
```

The implementation provides deterministic scan-time token and position
scoring. A PostgreSQL-compatible persistent GIN or BM25 index is not implied
by using these functions.

## JSON and JSONB

JSON and JSONB values are validated at write time, canonicalized, and retained
through transactions, WAL recovery, close/reopen, and temporal views.

### Extraction and predicates

```sql
SELECT payload->>'name' AS name,
       payload->'profile'->>'active' AS active
FROM documents
WHERE id = 'd1';

SELECT id
FROM documents
WHERE payload @> '{"roles":["admin"]}'::jsonb;

SELECT id
FROM documents
WHERE payload ? 'profile';

SELECT id
FROM documents
WHERE payload ?| ARRAY['profile', 'missing'];
```

The supported operators include `->`, `->>`, `#>`, `#>>`, `@>`, `<@`, `?`,
`?|`, `?&`, `@?`, `@@`, `#-`, and JSONB concatenation `||`. Array indexing and
typed parameter operands are supported:

```sql
SELECT payload->'roles'->>0 AS first_role
FROM documents
WHERE payload#>>$path = $value;
```

JSONPath supports root/member paths, array wildcards and indices, recursive
descent, strict/lax prefixes, scalar comparisons, and filter expressions such
as `? (@.active == true)`. `@?` preserves error-suppressing existence
behavior; strict `@@` evaluation reports missing-path errors.

### JSON functions

The supported function families include:

- `jsonb_array_elements`, `jsonb_array_elements_text`
- `jsonb_each`, `jsonb_each_text`, `jsonb_object_keys`
- `jsonb_to_record`, `jsonb_to_recordset`
- `jsonb_populate_record`, `jsonb_populate_recordset`
- `jsonb_array_length`, `jsonb_typeof`
- `jsonb_set`, `jsonb_insert`
- `jsonb_build_object`, `jsonb_build_array`, `to_jsonb`

`jsonb_set` and `jsonb_insert` may be used in both `SELECT` expressions and
`UPDATE SET`; the JSON mutation forms are evaluated atomically with the row
update. `json_set` and `json_insert` are accepted aliases for the same
operations.

Set-returning JSON functions can be used as query sources, including lateral
expansion of a row's JSON array:

```sql
SELECT role
FROM documents AS d
CROSS JOIN jsonb_array_elements_text(d.payload->'roles') AS role;
```

JSONB containment and key predicates may use LibraVDB's rebuildable derived
postings for candidate reduction:

```sql
SELECT id
FROM documents
WHERE payload @> '{"profile":{"active":true}}';
```

The JSON evaluator remains the final correctness check. The postings are
reconstructed from authoritative row data after reopen or schema changes and
are visible to ordinary current-state queries; epoch overlays use their
visible relation image. This is not a claim of a PostgreSQL-compatible on-disk
GIN posting-file format.

JSON literal `null` is distinct from SQL `NULL`. A JSON field containing
`null` remains discoverable by JSON operators and is returned as JSON text,
while a nullable SQL column containing SQL `NULL` is returned as a database
NULL to native and wire clients.

## Graph SQL

Graph queries operate over graph-backed collections. Relationship kinds are
registered graph labels, and graph identity is mapped to collection record
IDs. `MATCH` is available as a `WHERE` predicate, a `JOIN MATCH` relation, and
inside `GRAPH_TABLE`.

### Graph schema and bootstrap

SQL can create the graph-backed collection and durable relationship registry;
no native Go bootstrap call is required:

```sql
CREATE GRAPH TABLE users (
    id   TEXT PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE EDGE TYPE FOLLOWS;

INSERT INTO users (id, name) VALUES ('alice', 'Alice');
INSERT INTO users (id, name) VALUES ('bob', 'Bob');
INSERT INTO GRAPH_EDGES VALUES ('alice', 'FOLLOWS', 'bob');
```

`CREATE GRAPH TABLE` uses the same collection, record, graph-node, WAL, epoch,
and reopen machinery as the native API. Each inserted record receives a durable
graph identity; `GRAPH_EDGES` then resolves the logical record IDs to those
nodes. `CREATE EDGE TYPE` assigns an internal numeric kind and persists the
name-to-kind mapping, so typed `MATCH` continues to work after a process
restart. `CREATE TABLE ... REFERENCES GRAPH_NODES` declares a relational
foreign key; it does not by itself attach a graph layer. `GRAPH_NODES` remains a
read-only virtual relation.

Relationship kinds are directed by default. Declare a kind `UNDIRECTED` when
the relationship should be traversable from either endpoint:

```sql
CREATE EDGE TYPE KNOWS UNDIRECTED;
INSERT INTO GRAPH_EDGES VALUES ('alice', 'KNOWS', 'bob');

-- Both queries return the other endpoint.
SELECT target.id
FROM users AS source
JOIN MATCH (source)-[:KNOWS]->(target)
WHERE source.id = 'alice';

SELECT target.id
FROM users AS source
JOIN MATCH (source)-[:KNOWS]->(target)
WHERE source.id = 'bob';
```

An undirected relationship is stored once with one canonical source and target;
the reverse adjacency index supplies the opposite traversal. This avoids
duplicate edge rows, WAL records, history versions, or property blocks. The
ordinary `-[:KNOWS]-` pattern remains available when a query should traverse
both directions for the selected pattern. `DELETE FROM GRAPH_EDGES` accepts
either endpoint order for an undirected kind and removes the one canonical
edge.

### Traversal

```sql
SELECT source.id, target.id
FROM documents AS source
JOIN MATCH (source)-[:RELATES]->(target)
ORDER BY source.id, target.id;
```

Traversal supports direction, relationship-kind filters, vertex labels, and
bounded variable-length paths:

```sql
SELECT target.id
FROM documents AS source
JOIN MATCH (source)-[:RELATES*1..3]->(target)
WHERE source.category = 'graph';

SELECT source.id
FROM documents AS source
WHERE MATCH (source)<-[:CITES]-(reference:Document);
```

The equivalent `GRAPH_TABLE` form is supported for graph-first query shapes:

```sql
SELECT id
FROM GRAPH_TABLE(documents MATCH (source)-[:RELATES]->(target))
ORDER BY id;
```

The bounded path's lower and upper hop limits are enforced during traversal.
The executor does not expose a path object or shortest-path result expression;
use a bounded traversal and project the terminal nodes when that is sufficient.

### Stable graph row projections

`JOIN MATCH` exposes deterministic virtual columns for applications that need a
stable row shape over native SQL or pgwire. The columns are derived from the
matched graph row and are not stored as vertex metadata:

| Column | Type | Meaning |
| --- | --- | --- |
| `source_id` | `TEXT` | ID of the source vertex in the match pattern |
| `target_id` | `TEXT` | ID of the terminal vertex in the match pattern |
| `edge_type` | `TEXT` | Registered relationship kind name |
| `edge_weight` | `REAL` | Durable edge weight |

For a single-hop match, all four fields are available directly. `edge_type`
and `edge_weight` may also be projected through the edge variable using
`r.type`/`r.kind` and `r.weight`:

```sql
SELECT source_id,
       target_id,
       edge_type,
       edge_weight
FROM people AS src
JOIN MATCH (src)-[r:FOLLOWS]->(tgt)
WHERE src.id = $origin_id;

SELECT src.id AS source_id,
       tgt.id AS target_id,
       r.type AS edge_type,
       r.weight AS edge_weight
FROM people AS src
JOIN MATCH (src)-[r:FOLLOWS]->(tgt)
WHERE src.id = $origin_id;
```

The result columns are emitted in the requested projection order with stable
PostgreSQL-compatible types. Existing graph result IDs may retain the legacy
`source|target` composite form for compatibility; consumers should use these
projected columns instead of parsing that internal row ID. Edge projections
are defined for single-hop rows so one returned row corresponds to one stored
edge; bounded multi-hop traversal should project its vertex aliases.

### Common-neighbor traversal

Two `JOIN MATCH` stages may use the same terminal alias to express a common
neighbor without issuing one query per neighbor. The second graph anchor is
resolved from the same graph-backed collection and is constrained by ordinary
SQL predicates:

```sql
SELECT DISTINCT src.id, src.metadata
FROM people AS src
JOIN MATCH (src)-[]->(shared)
JOIN MATCH (origin)-[]->(shared)
WHERE origin.id = $origin_id
  AND src.id <> $origin_id
ORDER BY src.id;
```

LibraVDB materializes the origin-side terminal set, intersects it with each
source-side terminal set, and then applies the remaining predicates and
projection. The repeated `shared` alias is an equality join on the graph
endpoint; it is not two independent traversals whose results are combined as
a Cartesian product. `DISTINCT` is recommended when multiple origin rows or
multiple matching paths can produce the same projected source row.

### Edge properties

Edges may carry arbitrary JSON-compatible fields in addition to their durable
target, weight, and registered kind. The properties are stored with the
node-owned edge/page representation and follow the existing WAL, epoch,
reverse-edge, compaction, and reopen machinery.

Insert an edge with properties through the virtual `GRAPH_EDGES` relation:

```sql
INSERT INTO GRAPH_EDGES
    (source, type, target, properties)
VALUES
    ('alice', 'ROUTES_TO', 'bob',
     '{"cost":4.2,"confidence":0.98}');
```

Filter properties during traversal rather than after materializing all edges:

```sql
SELECT source.id, target.id
FROM users AS source
JOIN MATCH (source)-[
    edge:ROUTES_TO WHERE edge.cost <= $max_cost
        AND edge.confidence >= $min_confidence
]->(target);
```

The compact property block is also supported:

```sql
SELECT source.id
FROM users AS source
WHERE MATCH (source)-[
    edge:ROUTES_TO {
        cost < 5.0, confidence >= 0.9
    }
]->(target);
```

Property comparisons support equality and inequality operators, numeric and
text values, parameters, and boolean `AND`/`OR` combinations. Missing fields
and JSON `null` follow SQL NULL comparison behavior. The property envelope is
not a sidecar store. The registered relationship kind is available as
`edge.type`/`edge.kind`, and the durable numeric edge weight is available as
`edge.weight`; kind fields support equality and inequality comparisons.
Comma-separated entries in the compact property block are conjoined as `AND`.

The traditional edge fields remain available:

```sql
SELECT source.id
FROM users AS source
WHERE MATCH (source)-[
    edge:FOLLOWS WHERE edge.weight > 0.8
]->(target);
```

### Graph mutations

`GRAPH_EDGES` is a virtual mutation relation backed by the normal graph
transaction and WAL machinery:

```sql
INSERT INTO GRAPH_EDGES VALUES ('alice', 'FOLLOWS', 'bob');

DELETE FROM GRAPH_EDGES
WHERE source = 'alice'
  AND type = 'FOLLOWS'
  AND target = 'bob';
```

The three-column form uses the default edge weight. Parameters are supported
for source, type, and target; the property column accepts a JSON object value.
Graph-edge inserts and deletes participate in epoch transactions and
savepoints.

### Graph relations and algorithms

The virtual `GRAPH_NODES` relation exposes durable graph identity:

```sql
SELECT id, collection, record_id
FROM GRAPH_NODES
WHERE collection = 'users';
```

Relational tables may reference `GRAPH_NODES` and participate in the same
atomic cascade path as ordinary foreign keys. Numeric graph IDs and logical
text/UUID record IDs are supported by the graph identity resolver:

```sql
CREATE TABLE graph_refs (
    id       TEXT PRIMARY KEY,
    graph_id BIGINT REFERENCES GRAPH_NODES(id) ON DELETE CASCADE
);

CREATE TABLE text_graph_refs (
    id TEXT PRIMARY KEY REFERENCES GRAPH_NODES(id) ON DELETE CASCADE
);
```

Deleting the parent record removes dependent rows and the corresponding graph
identity atomically. The same behavior is visible through native SQL, epoch
overlays, and pgwire.

`GRAPH_CENTRALITY` can be used in a score expression:

```sql
SELECT d.id,
       (1.0 - VECTOR_DISTANCE(d.embedding, $query_vector))
       * GRAPH_CENTRALITY(d) AS score
FROM documents AS d
WHERE MATCH (d)<-[:CITES]-(reference)
ORDER BY score DESC
LIMIT 10;
```

`COMPUTE LEIDEN` is supported as a statement and as a CTE relation:

```sql
BEGIN EPOCH TRANSACTION;

COMPUTE LEIDEN FROM MATCH
    (seed:Document)-[:RELATES*1..3]->(target)
OPTIONS (resolution = 1.0, iterations = 2, max_vertices = 10000);

ROLLBACK;
```

The standalone form runs in an epoch SQL session and returns the typed Leiden
relation, including node and community identifiers plus diagnostics. The CTE
form can be joined to relational rows:

```sql
WITH communities AS (
    COMPUTE LEIDEN FROM MATCH
        (seed:Document)-[:RELATES*1..3]->(target)
    OPTIONS (resolution = 1.0, iterations = 2)
)
SELECT d.id, c.community_id
FROM documents AS d
JOIN communities AS c ON c.node_id = d.graph_id;
```

Leiden options include resolution, iteration/pass limits, and a vertex bound;
conflicting options are rejected with a validation error.

## Graph and `SELECT` subqueries

Graph joins can be nested in ordinary SQL subqueries, including membership
subqueries:

```sql
SELECT id, email, name
FROM users
WHERE id IN (
    SELECT source.id
    FROM users AS source
    JOIN MATCH (source)-[:FOLLOWS]->(target)
    WHERE target.name ILIKE '%ca%'
)
ORDER BY name;
```

The graph subquery is evaluated as a query-local relation. It does not create a
temporary collection or alter graph state.

## Temporal SQL

### Historical snapshots

`AS OF TIMESTAMP` evaluates a relation against a retained historical snapshot.
The same snapshot is used for record data, vector values, and graph adjacency
when those sources are combined:

```sql
SELECT d.title,
       d.embedding <-> $query_vector AS distance
FROM documents AS OF TIMESTAMP $snapshot d
WHERE MATCH (d)-[:RELATES]->(target)
ORDER BY distance
LIMIT 10;
```

The timestamp may be a literal or a typed parameter. An `AS OF` source may be
used inside a bounded CTE as shown earlier. Historical queries are subject to
the configured retention window; a valid but expired snapshot returns a
retention-expired error rather than silently returning current data.

`AS OF TIMESTAMP` is not allowed inside an active epoch transaction. Use a
normal database query for historical reads and an epoch transaction for staged
current-state work.

### Exact commit LSN tokens

LibraVDB exposes the storage engine's exact durable commit LSN as the snapshot
token. This is the same value used by the native snapshot APIs; there is no
second timestamp- or application-generated token.

| Surface | Access | Result |
| --- | --- | --- |
| Native Go | `db.LatestCommitLSN(ctx)` | `uint64` exact latest commit LSN |
| Native Go snapshot | `db.SnapshotAtLSN(ctx, lsn)` | Pinned `TemporalSnapshot` at that exact commit |
| Native SDKs | `latest_commit_lsn()` / `latestCommitLSN()` | Available in every official SDK; exact numeric type where safe, otherwise the decimal LSN string |
| SQL | `SELECT LIBRAVDB_LATEST_COMMIT_LSN()` | One `BIGINT` value |
| PostgreSQL wire | Startup `ParameterStatus` key `libravdb_latest_commit_lsn` | Exact LSN observed when the connection started |

The dedicated SDK getter is available with the following idiomatic spelling
and exact return representation:

| SDK | Method | Return type |
| --- | --- | --- |
| Python | `latest_commit_lsn()` | `int` |
| Rust | `latest_commit_lsn()` | `u64` |
| TypeScript | `latestCommitLSN()` | `bigint` |
| C++ | `latest_commit_lsn()` | `uint64_t` |
| C# | `LatestCommitLSN()` | `ulong` |
| Dart | `latestCommitLSN()` | decimal `String` |
| Java | `latestCommitLSN()` | `BigInteger` |
| Kotlin Native | `latestCommitLSN()` | `ULong` |
| Lua | `latest_commit_lsn()` | decimal string |
| Perl | `latest_commit_lsn()` | decimal string |
| PHP | `latestCommitLSN()` | decimal `string` |
| R | `latest_commit_lsn()` | decimal character string |
| Ruby | `latest_commit_lsn()` | `Integer` |
| Swift | `latestCommitLSN()` | `UInt64` |
| Odin | `latest_commit_lsn()` | decimal string |

All wrappers call the same `DatabaseLatestCommitLSN` C ABI export. The string
forms preserve values above the signed 64-bit range without rounding.

```go
lsn, err := db.LatestCommitLSN(ctx)
if err != nil {
    return err
}
snapshot, err := db.SnapshotAtLSN(ctx, lsn)
if err != nil {
    return err
}
defer snapshot.Close()
```

The SQL function reads the live latest commit position and is therefore useful
when a long-lived pgwire connection needs to refresh its token:

```sql
SELECT LIBRAVDB_LATEST_COMMIT_LSN() AS snapshot_lsn;
```

The token is an exact commit position, not a wall-clock timestamp. It can be
used with `SnapshotAtLSN` for native historical reads and compared with commit
receipts. Retention still applies: after temporal compaction, an expired LSN
returns the normal retention-expired error.

### Version ranges

Retained versions are exposed through the `VERSIONS OF` virtual relation:

```sql
SELECT id, version, title, version_start, version_end
FROM VERSIONS OF documents
BETWEEN TIMESTAMP $start AND TIMESTAMP $end
ORDER BY version;
```

Version rows include record metadata and version information. The current
version has a SQL NULL `version_end`.

## Transactions and session state

### Epoch transactions

Epoch transactions provide atomic record, vector, and graph mutations. The
transaction-local view includes staged writes, while other sessions continue
to see the committed state until commit.

`BEGIN` and `START TRANSACTION` are accepted as PostgreSQL-compatible aliases
for the same session transaction branch as `BEGIN EPOCH TRANSACTION`.

```sql
BEGIN EPOCH TRANSACTION;

INSERT INTO documents (id, title, category)
VALUES ('draft-1', 'Draft', 'temporary');

INSERT INTO GRAPH_EDGES VALUES ('draft-1', 'RELATES', 'd1');

SELECT id FROM documents WHERE id = 'draft-1';
COMMIT;
```

Rollback discards both record and graph changes:

```sql
BEGIN EPOCH TRANSACTION;
INSERT INTO GRAPH_EDGES VALUES ('draft-1', 'RELATES', 'd2');
ROLLBACK;
```

Standard savepoint commands are supported within an epoch transaction:

```sql
BEGIN EPOCH TRANSACTION;
INSERT INTO documents (id, title) VALUES ('keep', 'keep');
SAVEPOINT branch;
INSERT INTO documents (id, title) VALUES ('discard', 'discard');
ROLLBACK TO SAVEPOINT branch;
RELEASE SAVEPOINT branch;
COMMIT;
```

Session state is connection-local. The pgwire server supports the compatibility
settings commonly issued by PostgreSQL clients, including `statement_timeout`,
`max_recursion_depth`, `timezone`, `search_path`, `client_encoding`,
`standard_conforming_strings`, `datestyle`, `intervalstyle`, and
`extra_float_digits`.

```sql
SET statement_timeout = '5s';
SET max_recursion_depth = 100;
RESET statement_timeout;
RESET max_recursion_depth;
RESET ALL;
DISCARD ALL;
```

`statement_timeout` can shorten the server safety timeout. `max_recursion_depth`
limits recursive CTE evaluation. `SET LOCAL` is not supported until
transaction-local setting restoration is implemented.

## PostgreSQL session compatibility

The pgwire session exposes the startup values and compatibility queries issued
by PostgreSQL drivers and ORM dialects:

```sql
SHOW server_version;
SHOW client_encoding;
SHOW integer_datetimes;
SHOW standard_conforming_strings;
SHOW timezone;

SELECT current_schema();
SELECT current_database();
SELECT version();
SELECT current_setting('jit') AS current_jit,
       set_config('jit', 'off', false) AS new_jit;
SELECT set_config('TimeZone', 'UTC', false);
```

`set_config` is connection-local and returns the applied value. It does not
write a row, catalog entry, WAL record, or epoch mutation. The `local` form
(`set_config(..., true)`) is rejected until transaction-local setting
restoration is implemented.

## PostgreSQL catalog compatibility

The pgwire layer exposes live virtual catalog projections used by drivers and
ORMs. The following relations and views are supported for the documented
reflection paths:

| Catalog surface | Purpose |
| --- | --- |
| `pg_catalog.pg_namespace` | Schemas and namespace lookup |
| `pg_catalog.pg_class` | Relations and relation kinds |
| `pg_catalog.pg_attribute` | Column names, order, and nullability |
| `pg_catalog.pg_type` | PostgreSQL type names, OIDs, arrays, JSON/JSONB, UUID, and vectors |
| `pg_catalog.pg_constraint` | Primary, unique, foreign-key, and check-constraint reflection |
| `pg_catalog.pg_index` | Index and primary-key reflection |
| `pg_catalog.pg_attrdef` | Default-expression reflection |
| `pg_catalog.pg_proc` | Function lookup required by compatible clients |
| `pg_catalog.pg_range` | Range/type startup probes |
| `pg_catalog.pg_collation`, `pg_catalog.pg_description` | ORM comment and collation reflection projections |
| `pg_catalog.pg_indexes` | Index view used by ORM inspection |
| `information_schema` relations | Table, column, constraint, and schema inspection |

Catalog rows are derived from live collection and SQL metadata. They are not a
second storage engine and do not require separate user-data WAL or epoch state.
Catalog definitions, collection metadata schemas, and indexed-field
declarations are persisted and restored with the collection configuration.

The type projection includes the standard OIDs used by the supported clients,
including `bool` (16), `int8` (20), `text` (25), `json` (114), `jsonb` (3802),
`uuid` (2950), and PostgreSQL float-array results such as `_float4` (1021).
Result descriptions use these types for ordinary columns, expressions,
aggregates, empty result sets, and prepared statements.

## PostgreSQL wire protocol

The public package exposes the server without requiring callers to import an
internal package:

```go
import "github.com/xDarkicex/libravdb/pgwire"

server := pgwire.NewServer(db, pgwire.ServerConfig{
    Addr: "127.0.0.1:5432",
})
if err := server.Serve(ctx); err != nil {
    log.Fatal(err)
}
```

The documented wire path includes:

- Startup and authentication negotiation, including SCRAM where configured.
- Simple and extended query protocol messages.
- Parse, bind, describe, execute, sync, prepared statements, named portals,
  portal suspension, and statement reuse.
- Positional `$1` parameters are the PostgreSQL wire convention; the
  compatibility layer also recognizes named `$name` and `@name` markers and
  maps them to the protocol's encounter-order parameter slots.
- Text and supported binary parameter/result encodings.
- PostgreSQL NULL encoding and typed result metadata.
- Transactions, epoch aliases, savepoints, and connection-local settings.
- `COPY FROM STDIN` and `COPY TO STDOUT` for supported tables.

Driver compatibility depends on the SQL and type features used by the client.
The wire server should not be treated as a complete PostgreSQL server
implementation.

## Literal and identifier syntax

The lexer and parser support SQL comments, quoted identifiers, escaped string
literals, scientific numeric notation, and the operators used by the SQL
surface:

```sql
-- Line comments and /* block comments */ are ignored.
INSERT INTO "order" ("key", "value", "select")
VALUES ('a''b', 1.0e+2, E'line\\ntext');

SELECT "select", value << 1 AS shifted
FROM "order"
WHERE "key" <> 'missing';
```

`1e0`, `1.0e+0`, and equivalent uppercase forms are parsed as numeric
literals. Malformed scientific notation such as `1e` is rejected rather than
being split into an integer and an identifier. Reserved words can be used as
identifiers when quoted.

## Result and parameter semantics

The native and pgwire paths preserve the type and nullability information
needed by Go drivers and ORMs:

| Result value | Wire/native behavior |
| --- | --- |
| SQL `NULL` | Encoded as a PostgreSQL NULL, distinct from an empty string or zero |
| `JSON` / `JSONB` | Text JSON results and PostgreSQL JSONB binary results are supported |
| Vector values | Float-array-compatible results; vector distances are numeric |
| `GRAPH_NODES.id` | Typed `BIGINT`/`int8`, including empty-result descriptions |
| `ARRAY_AGG` | PostgreSQL text-array-compatible result with nullable elements |
| `VECTOR_AVG` | PostgreSQL float-array-compatible centroid |

Parameters are typed without SQL text substitution. They are supported in
predicates, arithmetic, casts, aggregate arguments, `CASE`, `IN`, `BETWEEN`,
`OFFSET`, `LIMIT`, JSON path/key operands, vector operators, graph edge
predicates, temporal bounds, and DML values. Uninferable or invalid parameter
types return an error during binding or description.

## Supported casts and functions

The general expression evaluator supports the following commonly used casts:

```sql
SELECT amount::float,
       id::text,
       flag::boolean,
       payload::jsonb,
       embedding::vector,
       '550e8400-e29b-41d4-a716-446655440000'::uuid
FROM documents;
```

The supported scalar function families include:

| Family | Functions |
| --- | --- |
| Time and null handling | `NOW`, `NULLIF` |
| Vector | `VECTOR_DISTANCE`, `SIMILARITY`, `VECTOR_AVG` |
| Graph | `GRAPH_CENTRALITY` |
| Full text | `to_tsvector`, `to_tsquery`, `plainto_tsquery`, `phraseto_tsquery`, `websearch_to_tsquery`, `ts_rank`, `FTS_RANK` |
| Ranking | `RRF` |
| JSON/JSONB | See [JSON and JSONB](#json-and-jsonb) |
| Aggregation | `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, `ARRAY_AGG`, `STRING_AGG` |

Unknown functions, casts, operators, or type forms fail with an execution
error. They are not silently accepted as no-ops.

## Deliberate boundaries

The following areas are outside the current supported SQL contract:

- Complete PostgreSQL compatibility or arbitrary PostgreSQL extensions.
- Arbitrary system catalog queries beyond the documented reflection
  projections.
- General-purpose stored procedures, triggers, user-defined functions, or
  PL/pgSQL.
- `SET LOCAL` and transaction-local session-setting restoration.
- Arbitrary unbounded recursion.
- Path-valued results, path reconstruction, and an explicit shortest-path
  operator. Use bounded `MATCH` traversal when terminal nodes are sufficient.
- Graph pattern syntax that has not been listed or exercised, including
  arbitrary Cypher grammar.
- SQL mutation syntax for edge properties other than the documented JSON
  `GRAPH_EDGES` property column.
- Persistent PostgreSQL GIN/BM25 index formats. JSON containment postings and
  full-text scoring are implemented by LibraVDB's own storage/execution paths.
- PostgreSQL extension types and operators that are not listed above.

Parsing alone does not establish support. Applications should rely on the
examples and feature boundaries in this page when selecting SQL for a beta or
production workload.
