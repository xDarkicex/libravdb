package libravdb

import (
	"context"
	"testing"
)

func TestSQLGraphToRelationalSemijoin(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql-graph-semijoin"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	gr, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer gr.Close()
	if !RegisterEdgeKind("SQL_SEMIJOIN_FOLLOWS", 253) && ResolveEdgeKind("SQL_SEMIJOIN_FOLLOWS") != 253 {
		t.Fatal("register SQL_SEMIJOIN_FOLLOWS")
	}
	people, err := db.CreateCollection(ctx, "people", WithMetadataOnly(), WithGraph(gr), WithMetadataSchema(MetadataSchema{
		"metadata": JSONBField,
	}))
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"alice", "bob", "carol", "dave", "shared-1", "shared-2", "shared-3"} {
		if err := people.Insert(ctx, id, nil, map[string]interface{}{"metadata": map[string]interface{}{"id": id}}); err != nil {
			t.Fatalf("insert %s: %v", id, err)
		}
	}
	node := func(id string) uint64 { return mustNodeID(t, db, ctx, "people", id) }
	txn := gr.BeginTxn()
	for _, edge := range [][2]string{
		{"alice", "shared-1"},
		{"alice", "shared-2"},
		{"bob", "shared-1"},
		{"carol", "shared-2"},
		{"dave", "shared-3"},
	} {
		if err := txn.AddEdge(node(edge[0]), node(edge[1]), 1, 253); err != nil {
			t.Fatal(err)
		}
	}
	if err := txn.Commit(ctx); err != nil {
		t.Fatal(err)
	}

	db.ResetSQLStats()
	semijoinSQL := `
		SELECT p.id, p.metadata
		FROM people p
		WHERE p.id IN (
			SELECT src.id
			FROM people src
			JOIN MATCH (src)-[]->(shared)
			JOIN MATCH (origin)-[]->(shared)
			WHERE origin.id = $1 AND src.id != $1
		)
		ORDER BY p.id`
	rows, err := db.QueryWithParams(ctx, semijoinSQL, QueryParams{"1": "alice"})
	if err != nil {
		t.Fatalf("graph semijoin: %v", err)
	}
	if rows.Total != 2 {
		t.Fatalf("graph semijoin rows=%d, want 2: %#v", rows.Total, rows)
	}
	if rows.Results[0].Metadata["id"] != "bob" || rows.Results[1].Metadata["id"] != "carol" {
		t.Fatalf("graph semijoin results=%#v, want bob and carol", rows.Results)
	}
	otherOrigin, err := db.QueryWithParams(ctx, semijoinSQL, QueryParams{"1": "bob"})
	if err != nil || otherOrigin.Total != 1 || otherOrigin.Results[0].Metadata["id"] != "alice" {
		t.Fatalf("reused graph semijoin rows=%+v err=%v, want alice", otherOrigin, err)
	}
	if stats := db.SQLStats(); stats.PlanCacheMisses == 0 || stats.PlanCacheHits == 0 {
		t.Fatalf("graph semijoin plan cache stats=%+v, want a miss followed by a hit", stats)
	}

	evidence, err := db.QueryWithParams(ctx, `
		SELECT candidate_id, evidence_id, edge_type, shared_count
		FROM GRAPH_SEMIJOIN('people', $1, 'SQL_SEMIJOIN_FOLLOWS') AS sj
		WHERE candidate_id <> $1
		ORDER BY candidate_id, evidence_id`, QueryParams{"1": "alice"})
	if err != nil {
		t.Fatalf("evidence graph semijoin: %v", err)
	}
	if evidence.Total != 2 || len(evidence.Columns) != 4 {
		t.Fatalf("evidence graph semijoin shape=%#v", evidence)
	}
	if evidence.Results[0].Metadata["candidate_id"] != "bob" || evidence.Results[0].Metadata["evidence_id"] != "shared-1" || evidence.Results[0].Metadata["edge_type"] != "SQL_SEMIJOIN_FOLLOWS" || evidence.Results[0].Metadata["shared_count"] != int64(1) {
		t.Fatalf("evidence graph semijoin rows=%#v", evidence.Results)
	}
}

func TestSQLGraphToRelationalSemijoinASOFLSN(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql-graph-semijoin-lsn"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	gr, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer gr.Close()
	if !RegisterEdgeKind("SQL_SEMIJOIN_LSN", 254) && ResolveEdgeKind("SQL_SEMIJOIN_LSN") != 254 {
		t.Fatal("register SQL_SEMIJOIN_LSN")
	}
	people, err := db.CreateCollection(ctx, "people", WithMetadataOnly(), WithGraph(gr), WithMetadataSchema(MetadataSchema{"metadata": JSONBField}))
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"alice", "bob", "carol", "shared-1", "shared-2"} {
		if err := people.Insert(ctx, id, nil, map[string]interface{}{"metadata": map[string]interface{}{"id": id}}); err != nil {
			t.Fatalf("insert %s: %v", id, err)
		}
	}
	node := func(id string) uint64 { return mustNodeID(t, db, ctx, "people", id) }
	first := gr.BeginTxn()
	if err := first.AddEdge(node("alice"), node("shared-1"), 1, 254); err != nil {
		t.Fatal(err)
	}
	if err := first.AddEdge(node("bob"), node("shared-1"), 1, 254); err != nil {
		t.Fatal(err)
	}
	if err := first.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	baseLSN, err := db.LatestCommitLSN(ctx)
	if err != nil {
		t.Fatal(err)
	}
	second := gr.BeginTxn()
	if err := second.AddEdge(node("alice"), node("shared-2"), 1, 254); err != nil {
		t.Fatal(err)
	}
	if err := second.AddEdge(node("carol"), node("shared-2"), 1, 254); err != nil {
		t.Fatal(err)
	}
	if err := second.Commit(ctx); err != nil {
		t.Fatal(err)
	}

	rows, err := db.QueryWithParams(ctx, `
		SELECT p.id
		FROM people AS OF LSN $snapshot p
		WHERE p.id IN (
			SELECT src.id
			FROM people src
			JOIN MATCH (src)-[]->(shared)
			JOIN MATCH (origin)-[]->(shared)
			WHERE origin.id = $origin AND src.id != $origin
		)
		ORDER BY p.id`, QueryParams{"snapshot": baseLSN, "origin": "alice"})
	if err != nil {
		t.Fatalf("graph semijoin AS OF LSN: %v", err)
	}
	if rows.Total != 1 || rows.Results[0].Metadata["id"] != "bob" {
		t.Fatalf("graph semijoin AS OF LSN rows=%#v, want bob only", rows.Results)
	}
}
