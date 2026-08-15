package libravdb

import (
	"context"
	"testing"

	"github.com/xDarkicex/libravdb/internal/graph"
)

func TestSQLEdgeWeightPredicatePushdown(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:edge_weight_predicate"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	g, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	col, err := db.CreateCollection(ctx, "documents", WithMetadataOnly(), WithGraph(g))
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"source-high", "source-low", "target-high", "target-low"} {
		if err := col.Insert(ctx, id, nil, nil); err != nil {
			t.Fatalf("insert %s: %v", id, err)
		}
	}

	node := func(id string) uint64 {
		n, err := db.GetNodeID(ctx, "documents", id)
		if err != nil {
			t.Fatalf("node %s: %v", id, err)
		}
		return n
	}
	const edgeKind = uint8(93)
	if !graph.RegisterEdgeKind("RELATES_WEIGHTED", edgeKind) {
		t.Fatal("edge kind registration failed")
	}
	txn := g.BeginTxn()
	if err := g.AddEdgeWithProperties(txn, node("source-high"), node("target-high"), 0.9, edgeKind, map[string]interface{}{"cost": 0.9, "confidence": 0.98}); err != nil {
		t.Fatal(err)
	}
	if err := g.AddEdgeWithProperties(txn, node("source-low"), node("target-low"), 0.2, edgeKind, map[string]interface{}{"cost": 0.2, "confidence": 0.55}); err != nil {
		t.Fatal(err)
	}
	if err := txn.Commit(ctx); err != nil {
		t.Fatal(err)
	}

	rows, err := db.Query(ctx, "SELECT target.id FROM documents s JOIN MATCH (s)-[r:RELATES_WEIGHTED WHERE r.weight > 0.8]->(target)")
	if err != nil {
		t.Fatalf("literal edge predicate: %v", err)
	}
	if rows.Total != 1 || rows.Results[0].ID != "source-high|target-high" {
		for _, row := range rows.Results {
			t.Logf("literal result id=%q metadata=%v", row.ID, row.Metadata)
		}
		t.Fatalf("literal edge predicate rows=%d, want target-high only", rows.Total)
	}
	for _, tc := range []struct {
		expr string
		want string
	}{
		{expr: "r.weight = 0.9", want: "source-high"},
		{expr: "r.weight != 0.9", want: "source-low"},
		{expr: "r.weight <> 0.9", want: "source-low"},
		{expr: "r.weight < 0.5", want: "source-low"},
		{expr: "r.weight <= 0.2", want: "source-low"},
		{expr: "r.weight > 0.8", want: "source-high"},
		{expr: "r.weight >= 0.9", want: "source-high"},
	} {
		rows, err := db.Query(ctx, "SELECT id FROM documents s WHERE MATCH (s)-[r:RELATES_WEIGHTED WHERE "+tc.expr+"]->(target)")
		if err != nil {
			t.Fatalf("edge predicate %s: %v", tc.expr, err)
		}
		if rows.Total != 1 || rows.Results[0].ID != tc.want {
			t.Fatalf("edge predicate %s rows=%v, want %s", tc.expr, rows.Results, tc.want)
		}
	}
	propertyRows, err := db.Query(ctx, "SELECT id FROM documents s WHERE MATCH (s)-[r:RELATES_WEIGHTED {weight > 0.8, type: 'RELATES_WEIGHTED'}]->(target)")
	if err != nil {
		t.Fatalf("edge property block: %v", err)
	}
	if propertyRows.Total != 1 || propertyRows.Results[0].ID != "source-high" {
		t.Fatalf("edge property block rows=%v, want source-high", propertyRows.Results)
	}
	propertyRows, err = db.Query(ctx, "SELECT id FROM documents s WHERE MATCH (s)-[r:RELATES_WEIGHTED {weight > 0.8 OR weight < 0.3}]->(target)")
	if err != nil {
		t.Fatalf("edge property OR block: %v", err)
	}
	if propertyRows.Total != 2 {
		t.Fatalf("edge property OR block rows=%v, want both sources", propertyRows.Results)
	}
	rows, err = db.Query(ctx, "SELECT id FROM GRAPH_TABLE(documents MATCH (s)-[r:RELATES_WEIGHTED WHERE r.weight > 0.8]->(target))")
	if err != nil {
		t.Fatalf("GRAPH_TABLE edge predicate: %v", err)
	}
	if rows.Total != 1 || rows.Results[0].ID != "target-high" {
		t.Fatalf("GRAPH_TABLE edge predicate rows=%v, want target-high only", rows.Results)
	}

	rows, err = db.QueryWithParams(ctx, "SELECT id FROM documents s WHERE MATCH (s)-[r:RELATES_WEIGHTED WHERE r.weight >= $threshold]->(target)", QueryParams{"threshold": float32(0.9)})
	if err != nil {
		t.Fatalf("parameter edge predicate: %v", err)
	}
	if rows.Total != 1 || rows.Results[0].ID != "source-high" {
		t.Fatalf("parameter edge predicate rows=%v, want source-high only", rows.Results)
	}
	rows, err = db.Query(ctx, "SELECT id FROM documents s WHERE MATCH (s)-[r:RELATES_WEIGHTED WHERE r.cost > 0.8]->(target)")
	if err != nil {
		t.Fatalf("arbitrary edge property predicate: %v", err)
	}
	if rows.Total != 1 || rows.Results[0].ID != "source-high" {
		t.Fatalf("arbitrary edge property rows=%v, want source-high", rows.Results)
	}

	epoch, err := db.BeginEpochTx(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer epoch.Rollback(ctx)
	rows, err = epoch.Query(ctx, "SELECT id FROM documents s WHERE MATCH (s)-[r:RELATES_WEIGHTED WHERE r.weight > 0.8]->(target)", nil)
	if err != nil {
		t.Fatalf("epoch edge predicate: %v", err)
	}
	if rows.Total != 1 || rows.Results[0].ID != "source-high" {
		t.Fatalf("epoch edge predicate rows=%v, want source-high only", rows.Results)
	}
	rows, err = epoch.Query(ctx, "SELECT id FROM documents s WHERE MATCH (s)-[r:RELATES_WEIGHTED WHERE r.cost >= $threshold]->(target)", QueryParams{"threshold": float32(0.8)})
	if err != nil {
		t.Fatalf("epoch arbitrary edge property predicate: %v", err)
	}
	if rows.Total != 1 || rows.Results[0].ID != "source-high" {
		t.Fatalf("epoch arbitrary edge property rows=%v, want source-high", rows.Results)
	}
}
