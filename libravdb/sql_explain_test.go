package libravdb

import (
	"context"
	"testing"
)

func TestSQLExplainAnalyzeGraphJoin(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(t.TempDir()+"/explain.libravdb"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	graph, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer graph.Close()
	if !RegisterEdgeKind("EXPLAIN_FOLLOWS", 252) {
		t.Fatal("register EXPLAIN_FOLLOWS")
	}
	people, err := db.CreateCollection(ctx, "people", WithMetadataOnly(), WithGraph(graph), WithMetadataSchema(MetadataSchema{
		"name": StringField,
	}))
	if err != nil {
		t.Fatal(err)
	}
	for _, person := range []struct{ id, name string }{
		{"alice", "Alice"}, {"bob", "Bob"}, {"carol", "Carol"}, {"shared", "Shared"},
	} {
		if err := people.Insert(ctx, person.id, nil, map[string]interface{}{"name": person.name}); err != nil {
			t.Fatalf("insert %s: %v", person.id, err)
		}
	}
	node := func(id string) uint64 {
		value, err := db.GetNodeID(ctx, "people", id)
		if err != nil {
			t.Fatalf("GetNodeID(%s): %v", id, err)
		}
		return value
	}
	txn := graph.BeginTxn()
	for _, edge := range [][2]string{{"alice", "shared"}, {"bob", "shared"}, {"carol", "shared"}} {
		if err := txn.AddEdge(node(edge[0]), node(edge[1]), 1, 252); err != nil {
			t.Fatal(err)
		}
	}
	if err := txn.Commit(ctx); err != nil {
		t.Fatal(err)
	}

	db.ResetSQLStats()
	rows, err := db.QueryWithParams(ctx, `EXPLAIN ANALYZE
		SELECT DISTINCT src.id
		FROM people src
		JOIN MATCH (src)-[]->(shared)
		JOIN MATCH (origin)-[]->(shared)
		WHERE origin.id = $1 AND src.id != $1
		ORDER BY src.id`, QueryParams{"1": "alice"})
	if err != nil {
		t.Fatalf("EXPLAIN ANALYZE: %v", err)
	}
	if rows.Total != 1 || len(rows.Results) != 1 || len(rows.Columns) != 1 || rows.Columns[0] != SQLExplainColumn {
		t.Fatalf("explain shape: total=%d results=%d columns=%v", rows.Total, len(rows.Results), rows.Columns)
	}
	plan, ok := rows.Results[0].Metadata[SQLExplainColumn].(SQLExplainPlan)
	if !ok {
		t.Fatalf("explain value type=%T value=%#v", rows.Results[0].Metadata[SQLExplainColumn], rows.Results[0].Metadata[SQLExplainColumn])
	}
	if plan.Strategy != "graph_join_match" || plan.Anchor != "src" {
		t.Fatalf("explain graph shape=%#v", plan)
	}
	if plan.ActualRows != 2 {
		t.Fatalf("actual rows=%d, want 2", plan.ActualRows)
	}
	if plan.GraphExpansions == 0 {
		t.Fatalf("graph expansions=%d, want non-zero", plan.GraphExpansions)
	}
	if plan.ExecutionTimeNanos == 0 {
		t.Fatalf("execution time=%d, want non-zero", plan.ExecutionTimeNanos)
	}
	if plan.PlanReused {
		t.Fatalf("first graph explain unexpectedly reused a plan: %#v", plan)
	}
	stats := db.SQLStats()
	if stats.Queries != 1 || stats.RowsReturned != 2 || stats.GraphExpansions != plan.GraphExpansions {
		t.Fatalf("query-local stats leaked incorrectly: stats=%#v plan=%#v", stats, plan)
	}
}
