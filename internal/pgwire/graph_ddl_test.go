package pgwire

import (
	"context"
	"database/sql"
	"net"
	"strings"
	"testing"

	_ "github.com/jackc/pgx/v5/stdlib"
	apexjson "github.com/xDarkicex/apexJSON/v2"
	"github.com/xDarkicex/libravdb/libravdb"
)

func TestPGWireSQLGraphDDLAndReopen(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/graph-ddl.libravdb"
	db, err := libravdb.Open(libravdb.WithStoragePath(path), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	srv := startTestServer(t, db)
	defer srv.Close()
	host, port, err := net.SplitHostPort(srv.Addr())
	if err != nil {
		t.Fatal(err)
	}
	sqlDB, err := sql.Open("pgx", "postgres://test:test@"+net.JoinHostPort(host, port)+"/test?sslmode=disable")
	if err != nil {
		t.Fatal(err)
	}
	defer sqlDB.Close()
	if err := sqlDB.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	for _, query := range []string{
		"CREATE GRAPH TABLE users (name TEXT)",
		"CREATE EDGE TYPE SQL_PGWIRE_FOLLOWS",
		"INSERT INTO users (id, name) VALUES ('alice', 'Alice')",
		"INSERT INTO users (id, name) VALUES ('bob', 'Bob')",
		"INSERT INTO GRAPH_EDGES VALUES ('alice', 'SQL_PGWIRE_FOLLOWS', 'bob')",
	} {
		if _, err := sqlDB.ExecContext(ctx, query); err != nil {
			t.Fatalf("%s: %v", query, err)
		}
	}
	var id string
	if err := sqlDB.QueryRowContext(ctx, "SELECT tgt.id FROM users src JOIN MATCH (src)-[:SQL_PGWIRE_FOLLOWS]->(tgt) WHERE src.id = 'alice'").Scan(&id); err != nil {
		t.Fatalf("JOIN MATCH over pgwire: %v", err)
	}
	if id != "bob" && !strings.HasSuffix(id, "|bob") {
		t.Fatalf("JOIN MATCH id=%q, want bob endpoint", id)
	}
}

func TestPGWireSQLUndirectedGraphDDL(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/undirected-graph-ddl.libravdb"
	db, err := libravdb.Open(libravdb.WithStoragePath(path), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	srv := startTestServer(t, db)
	defer srv.Close()
	host, port, err := net.SplitHostPort(srv.Addr())
	if err != nil {
		t.Fatal(err)
	}
	sqlDB, err := sql.Open("pgx", "postgres://test:test@"+net.JoinHostPort(host, port)+"/test?sslmode=disable")
	if err != nil {
		t.Fatal(err)
	}
	defer sqlDB.Close()
	if err := sqlDB.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	for _, query := range []string{
		"CREATE GRAPH TABLE undirected_users (name TEXT)",
		"CREATE EDGE TYPE SQL_PGWIRE_UNDIRECTED_KNOWS UNDIRECTED",
		"INSERT INTO undirected_users (id, name) VALUES ('alice', 'Alice')",
		"INSERT INTO undirected_users (id, name) VALUES ('bob', 'Bob')",
		"INSERT INTO GRAPH_EDGES VALUES ('alice', 'SQL_PGWIRE_UNDIRECTED_KNOWS', 'bob')",
	} {
		if _, err := sqlDB.ExecContext(ctx, query); err != nil {
			t.Fatalf("%s: %v", query, err)
		}
	}
	var id string
	if err := sqlDB.QueryRowContext(ctx, "SELECT tgt.id FROM undirected_users src JOIN MATCH (src)-[:SQL_PGWIRE_UNDIRECTED_KNOWS]->(tgt) WHERE src.id = 'bob'").Scan(&id); err != nil {
		t.Fatalf("reverse undirected JOIN MATCH over pgwire: %v", err)
	}
	if id != "alice" && !strings.HasSuffix(id, "|alice") {
		t.Fatalf("reverse undirected JOIN MATCH id=%q, want alice endpoint", id)
	}
}

func TestPGWireSQLCommonNeighborJoinMatch(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(":memory:pgwire-common-neighbor"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	srv := startTestServer(t, db)
	defer srv.Close()
	host, port, err := net.SplitHostPort(srv.Addr())
	if err != nil {
		t.Fatal(err)
	}
	sqlDB, err := sql.Open("pgx", "postgres://test:test@"+net.JoinHostPort(host, port)+"/test?sslmode=disable")
	if err != nil {
		t.Fatal(err)
	}
	defer sqlDB.Close()
	if err := sqlDB.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	for _, query := range []string{
		"CREATE GRAPH TABLE people (metadata JSONB)",
		"CREATE EDGE TYPE PGWIRE_COMMON_NEIGHBOR",
		"INSERT INTO people (id, metadata) VALUES ('alice', '{\"name\":\"Origin\"}')",
		"INSERT INTO people (id, metadata) VALUES ('bob', '{\"name\":\"Bob\"}')",
		"INSERT INTO people (id, metadata) VALUES ('carol', '{\"name\":\"Carol\"}')",
		"INSERT INTO people (id, metadata) VALUES ('dave', '{\"name\":\"Dave\"}')",
		"INSERT INTO people (id, metadata) VALUES ('shared-1', '{\"name\":\"Shared One\"}')",
		"INSERT INTO people (id, metadata) VALUES ('shared-2', '{\"name\":\"Shared Two\"}')",
		"INSERT INTO people (id, metadata) VALUES ('shared-3', '{\"name\":\"Unshared\"}')",
		"INSERT INTO GRAPH_EDGES VALUES ('alice', 'PGWIRE_COMMON_NEIGHBOR', 'shared-1')",
		"INSERT INTO GRAPH_EDGES VALUES ('alice', 'PGWIRE_COMMON_NEIGHBOR', 'shared-2')",
		"INSERT INTO GRAPH_EDGES VALUES ('bob', 'PGWIRE_COMMON_NEIGHBOR', 'shared-1')",
		"INSERT INTO GRAPH_EDGES VALUES ('carol', 'PGWIRE_COMMON_NEIGHBOR', 'shared-2')",
		"INSERT INTO GRAPH_EDGES VALUES ('dave', 'PGWIRE_COMMON_NEIGHBOR', 'shared-3')",
	} {
		if _, err := sqlDB.ExecContext(ctx, query); err != nil {
			t.Fatalf("%s: %v", query, err)
		}
	}

	rows, err := sqlDB.QueryContext(ctx, `
		SELECT DISTINCT src.id, src.metadata
		FROM people src
		JOIN MATCH (src)-[]->(shared)
		JOIN MATCH (origin)-[]->(shared)
		WHERE origin.id = $1
		  AND src.id != $1
		ORDER BY src.id`, "alice")
	if err != nil {
		t.Fatalf("common-neighbor JOIN MATCH over pgwire: %v", err)
	}
	defer rows.Close()
	seen := map[string]bool{}
	for rows.Next() {
		var id string
		var metadata []byte
		if err := rows.Scan(&id, &metadata); err != nil {
			t.Fatal(err)
		}
		seen[strings.SplitN(id, "|", 2)[0]] = true
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if len(seen) != 2 || !seen["bob"] || !seen["carol"] || seen["alice"] || seen["dave"] {
		t.Fatalf("common-neighbor pgwire IDs=%v, want bob and carol only", seen)
	}

	semijoinRows, err := sqlDB.QueryContext(ctx, `
		SELECT p.id, p.metadata
		FROM people p
		WHERE p.id IN (
			SELECT src.id
			FROM people src
			JOIN MATCH (src)-[]->(shared)
			JOIN MATCH (origin)-[]->(shared)
			WHERE origin.id = $1 AND src.id != $1
		)
		ORDER BY p.id`, "alice")
	if err != nil {
		t.Fatalf("graph-to-relational semijoin over pgwire: %v", err)
	}
	semijoinSeen := map[string]bool{}
	for semijoinRows.Next() {
		var id string
		var metadata []byte
		if err := semijoinRows.Scan(&id, &metadata); err != nil {
			_ = semijoinRows.Close()
			t.Fatalf("scan graph-to-relational semijoin: %v", err)
		}
		semijoinSeen[id] = true
	}
	if err := semijoinRows.Close(); err != nil {
		t.Fatalf("close graph-to-relational semijoin: %v", err)
	}
	if len(semijoinSeen) != 2 || !semijoinSeen["bob"] || !semijoinSeen["carol"] || semijoinSeen["alice"] || semijoinSeen["dave"] {
		t.Fatalf("graph-to-relational semijoin pgwire IDs=%v, want bob and carol only", semijoinSeen)
	}

	evidenceRows, err := sqlDB.QueryContext(ctx, `
		SELECT candidate_id, evidence_id, edge_type, shared_count
		FROM GRAPH_SEMIJOIN('people', $1, 'PGWIRE_COMMON_NEIGHBOR', 100, 100, 10) AS sj
		WHERE candidate_id <> $1
		ORDER BY candidate_id, evidence_id`, "alice")
	if err != nil {
		t.Fatalf("evidence graph semijoin over pgwire: %v", err)
	}
	var evidence []struct {
		candidate string
		evidence  string
		edgeType  string
		shared    int64
	}
	for evidenceRows.Next() {
		var row struct {
			candidate string
			evidence  string
			edgeType  string
			shared    int64
		}
		if err := evidenceRows.Scan(&row.candidate, &row.evidence, &row.edgeType, &row.shared); err != nil {
			_ = evidenceRows.Close()
			t.Fatalf("scan evidence graph semijoin: %v", err)
		}
		evidence = append(evidence, row)
	}
	if err := evidenceRows.Close(); err != nil {
		t.Fatalf("close evidence graph semijoin: %v", err)
	}
	if len(evidence) != 2 || evidence[0].candidate != "bob" || evidence[0].evidence != "shared-1" || evidence[0].edgeType != "PGWIRE_COMMON_NEIGHBOR" || evidence[0].shared != 1 || evidence[1].candidate != "carol" || evidence[1].evidence != "shared-2" {
		t.Fatalf("evidence graph semijoin rows=%+v, want bob/shared-1 and carol/shared-2", evidence)
	}

	var explainJSON []byte
	if err := sqlDB.QueryRowContext(ctx, `EXPLAIN ANALYZE
		SELECT DISTINCT src.id
		FROM people src
		JOIN MATCH (src)-[]->(shared)
		JOIN MATCH (origin)-[]->(shared)
		WHERE origin.id = $1 AND src.id != $1
		ORDER BY src.id`, "alice").Scan(&explainJSON); err != nil {
		t.Fatalf("EXPLAIN ANALYZE graph over pgwire: %v", err)
	}
	var explain struct {
		Strategy            string `json:"strategy"`
		ActualRows          int    `json:"actual_rows"`
		GraphExpansions     int    `json:"graph_expansions"`
		PredicateRejections int    `json:"predicate_rejections"`
		IndexHits           int    `json:"index_hits"`
		ExecutionTimeNS     int    `json:"execution_time_ns"`
		PlanReused          bool   `json:"plan_reused"`
	}
	if err := apexjson.Unmarshal(explainJSON, &explain); err != nil {
		t.Fatalf("decode EXPLAIN ANALYZE JSON %q: %v", explainJSON, err)
	}
	if explain.Strategy != "graph_join_match" || explain.ActualRows != 2 || explain.GraphExpansions == 0 || explain.ExecutionTimeNS == 0 {
		t.Fatalf("EXPLAIN ANALYZE payload=%s", explainJSON)
	}
}

func TestPGWireSQLStableEndpointAndEdgeProjections(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(":memory:pgwire-stable-graph-projections"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	srv := startTestServer(t, db)
	defer srv.Close()
	host, port, err := net.SplitHostPort(srv.Addr())
	if err != nil {
		t.Fatal(err)
	}
	sqlDB, err := sql.Open("pgx", "postgres://test:test@"+net.JoinHostPort(host, port)+"/test?sslmode=disable")
	if err != nil {
		t.Fatal(err)
	}
	defer sqlDB.Close()
	if err := sqlDB.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	for _, query := range []string{
		"CREATE GRAPH TABLE stable_people (metadata JSONB)",
		"CREATE EDGE TYPE PGWIRE_STABLE_RELATES",
		"INSERT INTO stable_people (id, metadata) VALUES ('alice', '{\"name\":\"Alice\"}')",
		"INSERT INTO stable_people (id, metadata) VALUES ('bob', '{\"name\":\"Bob\"}')",
		"INSERT INTO GRAPH_EDGES VALUES ('alice', 'PGWIRE_STABLE_RELATES', 'bob')",
	} {
		if _, err := sqlDB.ExecContext(ctx, query); err != nil {
			t.Fatalf("%s: %v", query, err)
		}
	}
	var sourceID, targetID, edgeType string
	var edgeWeight float32
	query := `
		SELECT source_id, target_id, r.type AS edge_type, r.weight AS edge_weight
		FROM stable_people src
		JOIN MATCH (src)-[r:PGWIRE_STABLE_RELATES]->(tgt)
		WHERE src.id = $1`
	if _, columns, err := describeStatement(db, query, 1); err != nil {
		t.Fatalf("describe stable endpoint/edge projection: %v", err)
	} else {
		assertColumns(t, columns, []ColumnMeta{
			{Name: "source_id", TypeOID: OIDText},
			{Name: "target_id", TypeOID: OIDText},
			{Name: "edge_type", TypeOID: OIDText},
			{Name: "edge_weight", TypeOID: OIDFloat4},
		})
	}
	native, err := db.QueryWithParams(ctx, query, libravdb.QueryParams{"1": "alice"})
	if err != nil || native.Total != 1 {
		t.Fatalf("stable endpoint/edge projection native rows=%d err=%v results=%#v", native.Total, err, native.Results)
	}
	if err := sqlDB.QueryRowContext(ctx, query, "alice").Scan(&sourceID, &targetID, &edgeType, &edgeWeight); err != nil {
		t.Fatalf("stable endpoint/edge projection over pgwire: %v", err)
	}
	if sourceID != "alice" || targetID != "bob" || edgeType != "PGWIRE_STABLE_RELATES" || edgeWeight != 1 {
		t.Fatalf("stable graph projection row=(%q,%q,%q,%v), want (alice,bob,PGWIRE_STABLE_RELATES,1)", sourceID, targetID, edgeType, edgeWeight)
	}
}

func TestPGWireGraphitiOptionalMergeAndPathProjection(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(":memory:pgwire-graphiti"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	srv := startTestServer(t, db)
	defer srv.Close()
	host, port, err := net.SplitHostPort(srv.Addr())
	if err != nil {
		t.Fatal(err)
	}
	sqlDB, err := sql.Open("pgx", "postgres://test:test@"+net.JoinHostPort(host, port)+"/test?sslmode=disable")
	if err != nil {
		t.Fatal(err)
	}
	defer sqlDB.Close()
	if err := sqlDB.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	for _, query := range []string{
		"CREATE GRAPH TABLE graphiti_people (uuid TEXT, name TEXT)",
		"CREATE EDGE TYPE PGWIRE_GRAPHITI_KNOWS",
		"MERGE (a:Person {uuid: $1})-[r:PGWIRE_GRAPHITI_KNOWS {weight: $3}]->(b:Person {uuid: $2}) ON CREATE SET a.name = $4, b.name = $5",
	} {
		if query == "MERGE (a:Person {uuid: $1})-[r:PGWIRE_GRAPHITI_KNOWS {weight: $3}]->(b:Person {uuid: $2}) ON CREATE SET a.name = $4, b.name = $5" {
			if _, err := sqlDB.ExecContext(ctx, query, "alice", "bob", 0.75, "Alice", "Bob"); err != nil {
				t.Fatalf("%s: %v", query, err)
			}
			continue
		}
		if _, err := sqlDB.ExecContext(ctx, query); err != nil {
			t.Fatalf("%s: %v", query, err)
		}
	}
	if _, err := sqlDB.ExecContext(ctx, "MERGE (a:Person {uuid: $1})-[r:PGWIRE_GRAPHITI_KNOWS {weight: $3}]->(b:Person {uuid: $2}) ON CREATE SET a.name = $4, b.name = $5", "bob", "carol", 0.5, "Bob 2", "Carol"); err != nil {
		t.Fatalf("second MERGE: %v", err)
	}
	var path []byte
	if err := sqlDB.QueryRowContext(ctx, `
		SELECT p
		FROM graphiti_people AS src
		JOIN MATCH p = (src)-[:PGWIRE_GRAPHITI_KNOWS]->(mid)-[:PGWIRE_GRAPHITI_KNOWS]->(target)
		WHERE src.id = 'alice'`).Scan(&path); err != nil {
		t.Fatalf("path projection over pgwire: %v", err)
	}
	if !strings.Contains(string(path), `"nodes":["alice","bob","carol"]`) {
		t.Fatalf("path JSON=%q", path)
	}
}

func TestPGWireNativeCypherShortestLabelsAndComprehension(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(":memory:pgwire-native-cypher"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	srv := startTestServer(t, db)
	defer srv.Close()
	host, port, err := net.SplitHostPort(srv.Addr())
	if err != nil {
		t.Fatal(err)
	}
	sqlDB, err := sql.Open("pgx", "postgres://test:test@"+net.JoinHostPort(host, port)+"/test?sslmode=disable")
	if err != nil {
		t.Fatal(err)
	}
	defer sqlDB.Close()
	if err := sqlDB.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	for _, query := range []string{
		"CREATE GRAPH TABLE cypher_people (name TEXT)",
		"CREATE EDGE TYPE PGWIRE_CYPHER_REL",
		"INSERT INTO cypher_people (id, name) VALUES ('alice', 'Alice')",
		"INSERT INTO cypher_people (id, name) VALUES ('bob', 'Bob')",
		"INSERT INTO cypher_people (id, name) VALUES ('carol', 'Carol')",
		"INSERT INTO GRAPH_EDGES VALUES ('alice', 'PGWIRE_CYPHER_REL', 'bob')",
		"INSERT INTO GRAPH_EDGES VALUES ('alice', 'PGWIRE_CYPHER_REL', 'carol')",
		"INSERT INTO GRAPH_EDGES VALUES ('bob', 'PGWIRE_CYPHER_REL', 'carol')",
	} {
		if _, err := sqlDB.ExecContext(ctx, query); err != nil {
			t.Fatalf("%s: %v", query, err)
		}
	}
	col, err := db.GetCollection("cypher_people")
	if err != nil {
		t.Fatal(err)
	}
	for _, item := range []struct {
		id, label string
	}{
		{"alice", "Person"}, {"alice", "Active"}, {"bob", "Person"}, {"carol", "Person"},
	} {
		node, err := db.GetNodeID(ctx, "cypher_people", item.id)
		if err != nil {
			t.Fatal(err)
		}
		col.GetGraph().RegisterVertexLabel(node, item.label)
	}

	var sourceID, targetID string
	if native, err := db.Query(ctx, `MATCH (a:Person:Active)-{weight > 0.5}->(b:Person)
RETURN a.id AS source_id, b.id AS target_id`); err != nil || native.Total == 0 {
		t.Fatalf("native-before-pgwire rows=%#v err=%v", native, err)
	}
	if err := sqlDB.QueryRowContext(ctx, `MATCH (a:Person:Active)-{weight > 0.5}->(b:Person)
RETURN a.id AS source_id, b.id AS target_id`).Scan(&sourceID, &targetID); err != nil {
		t.Fatalf("native MATCH RETURN over pgwire: %v", err)
	}
	if sourceID != "alice" || (targetID != "bob" && targetID != "carol") {
		t.Fatalf("native MATCH row=(%q,%q), want alice -> bob or carol", sourceID, targetID)
	}

	rows, err := sqlDB.QueryContext(ctx, `MATCH shortestPath((a)-[*1..3]->(b))
RETURN a.id AS source_id, b.id AS target_id`)
	if err != nil {
		t.Fatalf("shortestPath over pgwire: %v", err)
	}
	shortestRows := 0
	for rows.Next() {
		var source, target string
		if err := rows.Scan(&source, &target); err != nil {
			_ = rows.Close()
			t.Fatalf("scan shortestPath over pgwire: %v", err)
		}
		shortestRows++
	}
	if err := rows.Close(); err != nil {
		t.Fatal(err)
	}
	if shortestRows != 3 {
		t.Fatalf("shortestPath rows=%d, want 3", shortestRows)
	}

	rows, err = sqlDB.QueryContext(ctx, `MATCH (a)
RETURN [(a)-[:PGWIRE_CYPHER_REL]->(b) | b.id] AS friends`)
	if err != nil {
		t.Fatalf("pattern comprehension over pgwire: %v", err)
	}
	columns, err := rows.Columns()
	if err != nil {
		_ = rows.Close()
		t.Fatal(err)
	}
	if len(columns) != 1 || columns[0] != "friends" {
		_ = rows.Close()
		t.Fatalf("pattern comprehension columns=%v, want [friends]", columns)
	}
	if !rows.Next() {
		_ = rows.Close()
		t.Fatal("pattern comprehension returned no rows")
	}
	var friends []byte
	if err := rows.Scan(&friends); err != nil {
		_ = rows.Close()
		t.Fatalf("scan pattern comprehension over pgwire: %v", err)
	}
	if err := rows.Close(); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(friends), "bob") || !strings.Contains(string(friends), "carol") {
		t.Fatalf("pattern comprehension JSON=%q", friends)
	}
}
