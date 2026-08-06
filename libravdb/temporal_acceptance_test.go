package libravdb

import (
	"context"
	"fmt"
	"testing"
	"time"

	internalgraph "github.com/xDarkicex/libravdb/internal/graph"
)

func snapshotAfterGraphCommit(t *testing.T, db *Database, g Graph, source, target uint64) *TemporalSnapshot {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		snap, err := db.SnapshotAt(context.Background(), time.Now().UTC())
		if err == nil {
			edges, edgeErr := g.NeighborsAtLSN(source, snap.LSN)
			if edgeErr == nil {
				for _, edge := range edges {
					if edge.Target == target {
						return snap
					}
				}
			}
			snap.Close()
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("graph edge did not become durably visible before deadline")
	return nil
}

// TestTemporalAcceptance_FullPipeline executes the public SQL acceptance
// shape. It proves that a single temporal snapshot applies to MATCH, terminal
// predicates, historical vectors, ordering, and LIMIT; it deliberately does
// not rely on a hand-built physical plan.
func TestTemporalAcceptance_FullPipeline(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(t.TempDir() + "/acceptance.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(ctx)

	gr, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	defer gr.Close()
	col, err := db.CreateCollection(ctx, "customers", WithDimension(3), WithMetric(L2Distance), WithGraph(gr),
		WithMetadataSchema(MetadataSchema{"name": StringField, "category": StringField}))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	// The current graph API uses uint8 kinds. Register the SQL-visible name
	// used in this query before writing kind 1 edges.
	_ = internalgraph.RegisterEdgeKind("PURCHASED", 7)

	if err := col.Insert(ctx, "C1", []float32{1, 0, 0}, map[string]interface{}{"name": "Alice", "category": "customer"}); err != nil {
		t.Fatalf("insert customer: %v", err)
	}
	if err := col.Insert(ctx, "P1", []float32{0, 0, 1}, map[string]interface{}{"name": "AuthModule", "category": "Security"}); err != nil {
		t.Fatalf("insert product: %v", err)
	}
	c1Node, err := db.GetNodeID(ctx, "customers", "C1")
	if err != nil {
		t.Fatal(err)
	}
	p1Node, err := db.GetNodeID(ctx, "customers", "P1")
	if err != nil {
		t.Fatal(err)
	}
	gr.RegisterVertexLabel(p1Node, "Product")
	txn := gr.BeginTxn()
	if err := txn.AddEdge(c1Node, p1Node, 1, 7); err != nil {
		t.Fatal(err)
	}
	if err := txn.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	snap1 := snapshotAfterGraphCommit(t, db, gr, c1Node, p1Node)
	t1 := snap1.Timestamp
	snap1.Close()

	// Change the customer vector after T1. L2([1,0,0], [3,0,0]) is 4.
	if err := col.Update(ctx, "C1", []float32{3, 0, 0}, map[string]interface{}{"name": "Alice", "category": "customer"}); err != nil {
		t.Fatalf("update customer: %v", err)
	}
	snap2, err := db.SnapshotAt(ctx, time.Now().UTC())
	if err != nil {
		t.Fatal(err)
	}
	t2 := snap2.Timestamp
	snap2.Close()

	// Remove the qualifying relationship after T2.
	txn = gr.BeginTxn()
	if err := txn.RemoveEdge(c1Node, p1Node, 7); err != nil {
		t.Fatal(err)
	}
	if err := txn.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	// A removal has no live edge to poll; wait until its temporal view excludes it.
	var t3 time.Time
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		snap, snapshotErr := db.SnapshotAt(ctx, time.Now().UTC())
		if snapshotErr == nil {
			edges, edgeErr := gr.NeighborsAtLSN(c1Node, snap.LSN)
			if edgeErr == nil && len(edges) == 0 {
				t3 = snap.Timestamp
				snap.Close()
				break
			}
			snap.Close()
		}
		time.Sleep(5 * time.Millisecond)
	}
	if t3.IsZero() {
		t.Fatal("graph edge removal did not become durably visible before deadline")
	}

	queryAt := func(ts time.Time) (*SearchResults, error) {
		sql := fmt.Sprintf("SELECT c.name, VECTOR_DISTANCE(c.embedding, $prompt_vec) AS similarity "+
			"FROM customers c AS OF TIMESTAMP '%s' "+
			"WHERE MATCH (c)-[:PURCHASED]->(p:Product) AND p.category = 'Security' "+
			"ORDER BY similarity ASC LIMIT 5", ts.UTC().Format(time.RFC3339Nano))
		return db.QueryWithParams(ctx, sql, QueryParams{"prompt_vec": []float32{1, 0, 0}})
	}

	assertOne := func(label string, ts time.Time, wantScore float32) {
		t.Helper()
		results, queryErr := queryAt(ts)
		if queryErr != nil {
			t.Fatalf("%s query: %v", label, queryErr)
		}
		if results.Total != 1 || len(results.Results) != 1 {
			t.Fatalf("%s rows=%d, want exactly C1", label, results.Total)
		}
		got := results.Results[0]
		if got.ID != "C1" || got.Metadata["name"] != "Alice" {
			t.Fatalf("%s result=%+v, want C1/Alice", label, got)
		}
		if got.Score != wantScore {
			t.Fatalf("%s score=%v, want %v", label, got.Score, wantScore)
		}
	}

	assertOne("T1", t1, 0)
	assertOne("T2", t2, 4)
	results, err := queryAt(t3)
	if err != nil {
		t.Fatalf("T3 query: %v", err)
	}
	if results.Total != 0 {
		t.Fatalf("T3 rows=%d, want 0 after PURCHASED removal", results.Total)
	}
}
