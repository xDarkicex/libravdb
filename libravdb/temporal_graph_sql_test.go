package libravdb

import (
	"context"
	"fmt"
	"testing"
	"time"

	internalgraph "github.com/xDarkicex/libravdb/internal/graph"
)

// TestTemporalSQL_GraphOnly verifies that a graph predicate in an ordinary
// temporal SELECT is executed against the historical topology, without a
// vector projection or relational JOIN. The endpoint label and predicate are
// evaluated at the same snapshot as the edge visibility.
func TestTemporalSQL_GraphOnly(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(t.TempDir() + "/graph-only.libravdb"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Drop(ctx)

	g, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	col, err := db.CreateCollection(ctx, "nodes", WithMetadataOnly(), WithGraph(g),
		WithMetadataSchema(MetadataSchema{"category": StringField}))
	if err != nil {
		t.Fatal(err)
	}
	const edgeKind uint8 = 11
	if !internalgraph.RegisterEdgeKind("GRAPH_ONLY_LINK", edgeKind) && internalgraph.ResolveEdgeKind("GRAPH_ONLY_LINK") != edgeKind {
		t.Fatalf("edge kind registration conflict")
	}
	if err := col.Insert(ctx, "source", nil, map[string]interface{}{"category": "Source"}); err != nil {
		t.Fatal(err)
	}
	if err := col.Insert(ctx, "target", nil, map[string]interface{}{"category": "Security"}); err != nil {
		t.Fatal(err)
	}
	source, err := db.GetNodeID(ctx, "nodes", "source")
	if err != nil {
		t.Fatal(err)
	}
	target, err := db.GetNodeID(ctx, "nodes", "target")
	if err != nil {
		t.Fatal(err)
	}
	g.RegisterVertexLabel(target, "Target")
	txn := g.BeginTxn()
	if err := txn.AddEdge(source, target, 1, edgeKind); err != nil {
		t.Fatal(err)
	}
	if err := txn.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	snap1 := snapshotAfterGraphCommit(t, db, g, source, target)
	t1 := snap1.Timestamp
	snap1.Close()

	queryAt := func(ts time.Time) (*SearchResults, error) {
		sql := fmt.Sprintf("SELECT s.category FROM nodes s AS OF TIMESTAMP '%s' "+
			"WHERE MATCH (s)-[r:GRAPH_ONLY_LINK WHERE r.weight > 0.5]->(p:Target) AND p.category = 'Security' LIMIT 5",
			ts.UTC().Format(time.RFC3339Nano))
		return db.Query(ctx, sql)
	}
	rows, err := queryAt(t1)
	if err != nil {
		t.Fatalf("historical graph query: %v", err)
	}
	if rows.Total != 1 || len(rows.Results) != 1 || rows.Results[0].ID != "source" {
		t.Fatalf("at insertion snapshot got %#v, want source", rows.Results)
	}

	txn = g.BeginTxn()
	if err := txn.RemoveEdge(source, target, edgeKind); err != nil {
		t.Fatal(err)
	}
	if err := txn.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	var t2 time.Time
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		snap, snapshotErr := db.SnapshotAt(ctx, time.Now().UTC())
		if snapshotErr == nil {
			edges, edgeErr := g.NeighborsAtLSN(source, snap.LSN)
			if edgeErr == nil && len(edges) == 0 {
				t2 = snap.Timestamp
				snap.Close()
				break
			}
			snap.Close()
		}
		time.Sleep(5 * time.Millisecond)
	}
	if t2.IsZero() {
		t.Fatal("edge removal did not become visible")
	}
	rows, err = queryAt(t2)
	if err != nil {
		t.Fatalf("post-removal graph query: %v", err)
	}
	if rows.Total != 0 {
		t.Fatalf("post-removal rows=%d, want 0", rows.Total)
	}
}
