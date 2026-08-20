package libravdb

import (
	"context"
	"strings"
	"testing"
)

func TestSQLGraphDDLBootstrapsAndReopensUnifiedGraph(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/graph-ddl.libravdb"

	db, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, "CREATE GRAPH TABLE users (name TEXT)"); err != nil {
		t.Fatalf("CREATE GRAPH TABLE: %v", err)
	}
	if _, err := db.Query(ctx, "CREATE EDGE TYPE SQL_DDL_FOLLOWS"); err != nil {
		t.Fatalf("CREATE EDGE TYPE: %v", err)
	}
	for _, query := range []string{
		"INSERT INTO users (id, name) VALUES ('alice', 'Alice')",
		"INSERT INTO users (id, name) VALUES ('bob', 'Bob')",
		"INSERT INTO GRAPH_EDGES VALUES ('alice', 'SQL_DDL_FOLLOWS', 'bob')",
	} {
		if _, err := db.Query(ctx, query); err != nil {
			t.Fatalf("%s: %v", query, err)
		}
	}
	rows, err := db.Query(ctx, "SELECT tgt.id FROM users src JOIN MATCH (src)-[:SQL_DDL_FOLLOWS]->(tgt) WHERE src.id = 'alice'")
	if err != nil {
		t.Fatalf("JOIN MATCH before reopen: %v", err)
	}
	if len(rows.Results) != 1 || rows.Results[0].Metadata["id"] != "bob" {
		t.Fatalf("JOIN MATCH before reopen = %#v, want bob", rows.Results)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	db, err = Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer db.Close()
	users, err := db.GetCollection("users")
	if err != nil {
		t.Fatalf("reopened users: %v", err)
	}
	if users.GetGraph() == nil {
		t.Fatal("reopened graph table has no graph attachment")
	}
	rows, err = db.Query(ctx, "SELECT tgt.id FROM users src JOIN MATCH (src)-[:SQL_DDL_FOLLOWS]->(tgt) WHERE src.id = 'alice'")
	if err != nil {
		t.Fatalf("JOIN MATCH after reopen: %v", err)
	}
	if len(rows.Results) != 1 || rows.Results[0].Metadata["id"] != "bob" {
		t.Fatalf("JOIN MATCH after reopen = %#v, want bob", rows.Results)
	}
}

func TestSQLGraphVectorMergeSurvivesCatalogUpdatesAndReopen(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/graph-vector-catalog.libravdb"

	db, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	for _, query := range []string{
		"CREATE GRAPH TABLE Entity (uuid STRING PRIMARY KEY, name STRING, name_embedding VECTOR(4))",
		"CREATE GRAPH TABLE Episodic (uuid STRING PRIMARY KEY, name STRING, content STRING)",
		"CREATE GRAPH TABLE Community (uuid STRING PRIMARY KEY, name STRING, summary STRING)",
		"CREATE GRAPH TABLE Saga (uuid STRING PRIMARY KEY, name STRING, group_id STRING)",
		"CREATE GRAPH TABLE RelatesToNode_ (uuid STRING PRIMARY KEY, name STRING, fact STRING)",
		"CREATE EDGE TYPE SQL_GRAPH_VECTOR_MENTIONS",
		"CREATE EDGE TYPE SQL_GRAPH_VECTOR_RELATES_TO",
	} {
		if _, err := db.Query(ctx, query); err != nil {
			t.Fatalf("%s: %v", query, err)
		}
	}

	vector := []float32{0.01, 0.02, 0.03, 0.04}
	if _, err := db.QueryWithParams(ctx,
		"MERGE (n:Entity {uuid: $uuid}) SET n.name = $name, n.name_embedding = $embedding",
		QueryParams{"uuid": "alice", "name": "Alice", "embedding": vector}); err != nil {
		t.Fatalf("MERGE after graph-table bootstrap: %v", err)
	}
	entity, err := db.GetCollection("Entity")
	if err != nil {
		t.Fatal(err)
	}
	record, err := entity.Get(ctx, "alice")
	if err != nil {
		t.Fatal(err)
	}
	if len(record.Vector) != len(vector) {
		t.Fatalf("stored vector length=%d, want %d", len(record.Vector), len(vector))
	}
	for i := range vector {
		if record.Vector[i] != vector[i] {
			t.Fatalf("stored vector=%v, want %v", record.Vector, vector)
		}
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	db, err = Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer db.Close()
	entity, err = db.GetCollection("Entity")
	if err != nil {
		t.Fatal(err)
	}
	vector = []float32{0.04, 0.03, 0.02, 0.01}
	if _, err := db.QueryWithParams(ctx,
		"MERGE (n:Entity {uuid: $uuid}) SET n.name_embedding = $embedding",
		QueryParams{"uuid": "bob", "embedding": vector}); err != nil {
		t.Fatalf("MERGE after reopen: %v", err)
	}
	record, err = entity.Get(ctx, "bob")
	if err != nil {
		t.Fatal(err)
	}
	if len(record.Vector) != len(vector) {
		t.Fatalf("reopened stored vector length=%d, want %d", len(record.Vector), len(vector))
	}
	for i := range vector {
		if record.Vector[i] != vector[i] {
			t.Fatalf("reopened stored vector=%v, want %v", record.Vector, vector)
		}
	}
}

func TestSQLUndirectedEdgeTypeTraversesBothEndpointsAndSurvivesReopen(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/undirected-graph-ddl.libravdb"

	db, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	for _, query := range []string{
		"CREATE GRAPH TABLE undirected_users (name TEXT)",
		"CREATE EDGE TYPE SQL_DDL_UNDIRECTED_KNOWS UNDIRECTED",
		"INSERT INTO undirected_users (id, name) VALUES ('alice', 'Alice')",
		"INSERT INTO undirected_users (id, name) VALUES ('bob', 'Bob')",
		"INSERT INTO GRAPH_EDGES VALUES ('alice', 'SQL_DDL_UNDIRECTED_KNOWS', 'bob')",
	} {
		if _, err := db.Query(ctx, query); err != nil {
			t.Fatalf("%s: %v", query, err)
		}
	}
	rows, err := db.Query(ctx, "SELECT tgt.id FROM undirected_users src JOIN MATCH (src)-[:SQL_DDL_UNDIRECTED_KNOWS]->(tgt) WHERE src.id = 'bob'")
	if err != nil {
		t.Fatalf("reverse undirected MATCH before reopen: %v", err)
	}
	if len(rows.Results) != 1 || !strings.HasSuffix(rows.Results[0].ID, "alice") {
		t.Fatalf("reverse undirected MATCH before reopen = %#v, want alice", rows.Results)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	db, err = Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer db.Close()
	users, err := db.GetCollection("undirected_users")
	if err != nil {
		t.Fatalf("reopened undirected_users: %v", err)
	}
	rows, err = db.Query(ctx, "SELECT tgt.id FROM undirected_users src JOIN MATCH (src)-[:SQL_DDL_UNDIRECTED_KNOWS]->(tgt) WHERE src.id = 'bob'")
	if err != nil {
		t.Fatalf("reverse undirected MATCH after reopen: %v", err)
	}
	if len(rows.Results) != 1 || !strings.HasSuffix(rows.Results[0].ID, "alice") {
		t.Fatalf("reverse undirected MATCH after reopen = %#v, want alice", rows.Results)
	}
	if _, err := db.Query(ctx, "DELETE FROM GRAPH_EDGES WHERE source = 'bob' AND type = 'SQL_DDL_UNDIRECTED_KNOWS' AND target = 'alice'"); err != nil {
		t.Fatalf("reverse undirected GRAPH_EDGES delete: %v", err)
	}
	rows, err = db.Query(ctx, "SELECT tgt.id FROM undirected_users src JOIN MATCH (src)-[:SQL_DDL_UNDIRECTED_KNOWS]->(tgt) WHERE src.id = 'bob'")
	if err != nil {
		t.Fatalf("undirected MATCH after delete: %v", err)
	}
	aliceNode, err := db.GetNodeID(ctx, "undirected_users", "alice")
	if err != nil {
		t.Fatalf("resolve alice after delete: %v", err)
	}
	bobNode, err := db.GetNodeID(ctx, "undirected_users", "bob")
	if err != nil {
		t.Fatalf("resolve bob after delete: %v", err)
	}
	aliceEdges, err := users.GetGraph().Neighbors(aliceNode)
	if err != nil {
		t.Fatalf("read alice neighbors after delete: %v", err)
	}
	bobEdges, err := users.GetGraph().Neighbors(bobNode)
	if err != nil {
		t.Fatalf("read bob neighbors after delete: %v", err)
	}
	if len(aliceEdges) != 0 || len(bobEdges) != 0 {
		t.Fatalf("undirected graph state after delete: alice=%#v bob=%#v, want no edges", aliceEdges, bobEdges)
	}
	if len(rows.Results) != 0 {
		t.Fatalf("undirected MATCH after delete = %#v, want no rows", rows.Results)
	}
}

func TestSQLGraphTablesShareDefaultNamespaceAcrossReopen(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/shared-graph-namespace.libravdb"

	db, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	for _, table := range []string{"Entity", "Episodic", "Community", "Saga", "RelatesToNode_"} {
		if _, err := db.Query(ctx, "CREATE GRAPH TABLE "+table+" (uuid STRING PRIMARY KEY, name TEXT)"); err != nil {
			t.Fatalf("create graph table %s: %v", table, err)
		}
	}
	if _, err := db.Query(ctx, "CREATE EDGE TYPE SHARED_NAMESPACE_REL"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.QueryWithParams(ctx,
		"MERGE (n:Entity {uuid: $uuid}) SET n.name = $name",
		QueryParams{"uuid": "entity-1", "name": "Entity 1"}); err != nil {
		t.Fatalf("MERGE after multi-table bootstrap: %v", err)
	}
	if _, err := db.QueryWithParams(ctx,
		"MERGE (n:Episodic {uuid: $uuid}) SET n.name = $name",
		QueryParams{"uuid": "episode-1", "name": "Episode 1"}); err != nil {
		t.Fatalf("second-table MERGE: %v", err)
	}

	entity, err := db.GetCollection("Entity")
	if err != nil {
		t.Fatal(err)
	}
	episodic, err := db.GetCollection("Episodic")
	if err != nil {
		t.Fatal(err)
	}
	entityGraph, ok := entity.GetGraph().(*collectionGraph)
	if !ok {
		t.Fatalf("Entity graph = %T, want collection-bound default graph", entity.GetGraph())
	}
	episodicGraph, ok := episodic.GetGraph().(*collectionGraph)
	if !ok || entityGraph.Graph != episodicGraph.Graph {
		t.Fatal("SQL graph tables do not share the same default graph runtime")
	}
	entityNode, err := db.GetNodeID(ctx, "Entity", "entity-1")
	if err != nil {
		t.Fatal(err)
	}
	episodicNode, err := db.GetNodeID(ctx, "Episodic", "episode-1")
	if err != nil {
		t.Fatal(err)
	}
	txn := entity.GetGraph().BeginTxn()
	if err := txn.AddEdge(entityNode, episodicNode, 1, ResolveEdgeKind("SHARED_NAMESPACE_REL")); err != nil {
		t.Fatal(err)
	}
	if err := txn.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	if stats := entity.GetGraph().Stats(); stats.EdgesAdded == 0 {
		t.Fatalf("cross-table edge was not applied before reopen: %#v", stats)
	}
	neighbors, err := episodic.GetGraph().Neighbors(entityNode)
	if err != nil || len(neighbors) != 1 || neighbors[0].Target != episodicNode {
		t.Fatalf("cross-table edge before reopen = %#v, err=%v", neighbors, err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	db, err = Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer db.Close()
	entity, err = db.GetCollection("Entity")
	if err != nil {
		t.Fatal(err)
	}
	episodic, err = db.GetCollection("Episodic")
	if err != nil {
		t.Fatal(err)
	}
	entityGraph, ok = entity.GetGraph().(*collectionGraph)
	if !ok {
		t.Fatalf("reopened Entity graph = %T, want collection-bound default graph", entity.GetGraph())
	}
	episodicGraph, ok = episodic.GetGraph().(*collectionGraph)
	if !ok || entityGraph.Graph != episodicGraph.Graph {
		t.Fatal("reopened SQL graph tables do not share the default graph runtime")
	}
	neighbors, err = entity.GetGraph().Neighbors(entityNode)
	if err != nil || len(neighbors) != 1 || neighbors[0].Target != episodicNode {
		count := 0
		entity.GetGraph().ForEachEdge(func(src, tgt uint64, edge Edge) bool {
			count++
			return true
		})
		t.Fatalf("cross-table edge after reopen = %#v, err=%v, total_edges=%d stats=%#v", neighbors, err, count, entity.GetGraph().Stats())
	}
	entity.GetGraph().RegisterVertexLabel(entityNode, "Person")
	rows, err := db.Query(ctx, `MATCH (n:Person) WITH n.uuid AS u WHERE u <> 'missing' RETURN u`)
	if err != nil || rows.Total != 1 || rows.Results[0].Metadata["u"] != "entity-1" {
		t.Fatalf("shared-graph Cypher WITH rows=%#v err=%v", rows, err)
	}
}

func TestSQLGraphSharedNamespaceCypherWithUsesLabeledOwner(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:shared-cypher-with"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	for _, table := range []string{"sql_graph_ddl_probe", "sql_graph_undirected_probe", "sql_common_probe", "sql_stable_probe"} {
		if _, err := db.Query(ctx, "CREATE GRAPH TABLE "+table+" (id TEXT PRIMARY KEY, name TEXT)"); err != nil {
			t.Fatalf("create %s: %v", table, err)
		}
	}
	if _, err := db.Query(ctx, "CREATE GRAPH TABLE sql_graphiti_probe (id TEXT PRIMARY KEY, uuid TEXT, name TEXT)"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, "CREATE EDGE TYPE SHARED_CYPHER_WITH_REL"); err != nil {
		t.Fatal(err)
	}
	merge := "MERGE (a:Person {uuid: $1})-[r:SHARED_CYPHER_WITH_REL {weight: $3}]->(b:Person {uuid: $2}) ON CREATE SET a.name = $4, b.name = $5"
	if _, err := db.QueryWithParams(ctx, merge, QueryParams{"1": "alice", "2": "bob", "3": 0.75, "4": "Alice", "5": "Bob"}); err != nil {
		t.Fatal(err)
	}
	col, err := db.GetCollection("sql_graphiti_probe")
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"alice", "bob"} {
		node, nodeErr := db.GetNodeID(ctx, col.name, id)
		if nodeErr != nil {
			t.Fatal(nodeErr)
		}
		col.GetGraph().RegisterVertexLabel(node, "Person")
	}
	rows, err := db.Query(ctx, `MATCH (n:Person) WITH n.uuid AS u WHERE u <> 'missing' RETURN u ORDER BY u`)
	if err != nil || rows.Total != 2 || rows.Results[0].Metadata["u"] != "alice" || rows.Results[1].Metadata["u"] != "bob" {
		t.Fatalf("shared-graph Cypher WITH rows=%#v err=%v", rows, err)
	}
	pattern, err := db.Query(ctx, `MATCH (a) RETURN [(a)-[:SHARED_CYPHER_WITH_REL]->(b) | b.id] AS friends`)
	if err != nil || pattern.Total != 2 {
		t.Fatalf("shared-graph pattern rows=%#v err=%v", pattern, err)
	}
}
