package pgwire

import (
	"context"
	"database/sql"
	"net"
	"strings"
	"testing"

	_ "github.com/jackc/pgx/v5/stdlib"
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
}
