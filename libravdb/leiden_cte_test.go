package libravdb

import (
	"context"
	"reflect"
	"testing"
	"time"

	"github.com/xDarkicex/lexer/parser"
	"github.com/xDarkicex/libravdb/internal/graph"
)

// =============================================================================
// Test harness
// =============================================================================

type cteHarness struct {
	db      *Database
	gr      Graph
	col     *Collection
	colName string
}

func newCTEHarness(t *testing.T, colName string) *cteHarness {
	t.Helper()
	db, err := Open(WithStoragePath(t.TempDir() + "/leiden_cte.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { db.Drop(context.Background()) })

	gr, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	t.Cleanup(func() { gr.Close() })

	col, err := db.CreateCollection(context.Background(), colName, WithDimension(3), WithGraph(gr))
	if err != nil {
		t.Fatalf("CreateCollection %q: %v", colName, err)
	}

	return &cteHarness{db: db, gr: gr, col: col, colName: colName}
}

func (h *cteHarness) insertLabeled(id, label string) uint64 {
	h.col.Insert(context.Background(), id, []float32{1, 0, 0}, nil)
	nid, _ := h.db.GetNodeID(context.Background(), h.colName, id)
	h.gr.RegisterVertexLabel(nid, label)
	return nid
}

// =============================================================================
// Test 1: Target binding
// =============================================================================

func TestLeidenCTE_TargetBinding(t *testing.T) {
	h := newCTEHarness(t, "nodes")
	graph.RegisterEdgeKind("LINK", 10)
	h.insertLabeled("s1", "seeds")
	h.insertLabeled("s2", "seeds")

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	sql := `WITH local_clusters AS (
    COMPUTE LEIDEN FROM MATCH
    (s:seeds)-[:LINK*1..2]->(target)
)
SELECT d.title, c.community_id
FROM nodes d
JOIN local_clusters c ON d.node_id = c.node_id`

	src := []byte(sql)
	var doc parser.QueryDoc
	if err := parser.Parse(src, &doc); err != nil {
		t.Fatalf("Parse: %v", err)
	}

	bound, err := epoch.BindLeidenCTE(context.Background(), src, &doc, 0)
	if err != nil {
		t.Fatalf("BindLeidenCTE: %v", err)
	}

	if bound.Name != "local_clusters" {
		t.Errorf("Name: want local_clusters, got %q", bound.Name)
	}
	if bound.Collection != "nodes" {
		t.Errorf("Collection: want nodes, got %q", bound.Collection)
	}
	if bound.JoinAlias != "c" {
		t.Errorf("JoinAlias: want c, got %q", bound.JoinAlias)
	}
	if bound.Plan == nil {
		t.Fatal("Plan must not be nil")
	}
	if bound.Plan.Spec.Collection != "nodes" {
		t.Errorf("Spec.Collection: want nodes, got %q", bound.Plan.Spec.Collection)
	}
	if len(bound.Plan.Spec.SeedNodeIDs) != 2 {
		t.Fatalf("SeedNodeIDs: want 2 seeds, got %d", len(bound.Plan.Spec.SeedNodeIDs))
	}
	if len(bound.Plan.Spec.EdgeKinds) != 1 || bound.Plan.Spec.EdgeKinds[0] != 10 {
		t.Errorf("EdgeKinds: want [10], got %v", bound.Plan.Spec.EdgeKinds)
	}
	if bound.Plan.Spec.Direction != LeidenMatchOutbound {
		t.Error("Direction: want outbound")
	}
	if bound.Plan.Spec.MinHops != 1 || bound.Plan.Spec.MaxHops != 2 {
		t.Errorf("Hops: want [1,2], got [%d,%d]", bound.Plan.Spec.MinHops, bound.Plan.Spec.MaxHops)
	}

	t.Logf("bound CTE %q on collection %q, alias %q, %d seeds",
		bound.Name, bound.Collection, bound.JoinAlias, len(bound.Plan.Spec.SeedNodeIDs))
	t.Log("✅ target CTE binding")
}

// =============================================================================
// Test 2: Outer collection is authoritative
// =============================================================================

func TestLeidenCTE_OuterCollectionAuthoritative(t *testing.T) {
	h := newCTEHarness(t, "documents")
	graph.RegisterEdgeKind("LINK", 10)

	// Also create a second collection with the same seed label.
	gr2, _ := NewGraph(GraphConfig{})
	defer gr2.Close()
	db2 := h.db
	col2, _ := db2.CreateCollection(context.Background(), "other", WithDimension(3), WithGraph(gr2))
	col2.Insert(context.Background(), "x", []float32{1, 0, 0}, nil)
	xNID, _ := db2.GetNodeID(context.Background(), "other", "x")
	gr2.RegisterVertexLabel(xNID, "seeds")

	// Insert seeds into the primary collection too.
	h.insertLabeled("s1", "seeds")

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	sql := `WITH local_clusters AS (
    COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK*1..2]->(target)
)
SELECT d.title, c.community_id
FROM documents d
JOIN local_clusters c ON d.node_id = c.node_id`

	src := []byte(sql)
	var doc parser.QueryDoc
	parser.Parse(src, &doc)

	bound, err := epoch.BindLeidenCTE(context.Background(), src, &doc, 0)
	if err != nil {
		t.Fatalf("BindLeidenCTE: %v", err)
	}

	if bound.Collection != "documents" {
		t.Fatalf("Collection: want documents (authoritative FROM), got %q", bound.Collection)
	}

	t.Log("✅ outer collection is authoritative")
}

// =============================================================================
// Test 3: Collection mismatch
// =============================================================================

func TestLeidenCTE_CollectionMismatch(t *testing.T) {
	h := newCTEHarness(t, "nodes")
	graph.RegisterEdgeKind("LINK", 10)
	h.insertLabeled("s", "seeds")

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	// Create a second collection so it's a valid name.
	h.db.CreateCollection(context.Background(), "other", WithDimension(3), WithGraph(h.gr))

	sql := `WITH local_clusters AS (
    COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK]->(target)
)
SELECT d.title, c.community_id
FROM nodes d
JOIN local_clusters c ON d.node_id = c.node_id`

	src := []byte(sql)
	var doc parser.QueryDoc
	parser.Parse(src, &doc)

	// Lower the logical plan manually and set a conflicting collection.
	cte := doc.CTEs[doc.SelectStmts[0].CTEsStart]
	plan, _ := LowerComputeLeidenPlan(src, &doc, int(cte.Body.ID))
	plan.Collection = "other"
	// Re-parse fresh since LowerComputeLeidenPlan mutates doc state (IDs shift).
	var doc2 parser.QueryDoc
	parser.Parse(src, &doc2)
	// Manually set the conflict on the plan that BindLeidenCTE will lower internally.
	// We do this by binding manually with the conflicting plan.
	_, err := epoch.BindLeidenMatchPlan(context.Background(), plan, "nodes")
	if err == nil {
		t.Fatal("expected error for collection mismatch")
	}
	t.Logf("collection mismatch rejected: %v", err)

	t.Log("✅ collection mismatch rejected")
}

// =============================================================================
// Test 4: Missing outer FROM
// =============================================================================

func TestLeidenCTE_MissingOuterFrom(t *testing.T) {
	h := newCTEHarness(t, "nodes")
	graph.RegisterEdgeKind("LINK", 10)
	h.insertLabeled("s", "seeds")

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	// CTE SELECT with GRAPH_TABLE instead of table expression.
	sql := `WITH local_clusters AS (
    COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK]->(target)
)
SELECT c.community_id
FROM GRAPH_TABLE(nodes MATCH (a)-[:LINK]->(b))`

	src := []byte(sql)
	var doc parser.QueryDoc
	if err := parser.Parse(src, &doc); err != nil {
		t.Fatalf("Parse: %v", err)
	}

	_, err := epoch.BindLeidenCTE(context.Background(), src, &doc, 0)
	if err == nil {
		t.Fatal("expected error for graph-table FROM")
	}
	t.Logf("graph-table FROM rejected: %v", err)

	t.Log("✅ missing outer FROM rejected")
}

// =============================================================================
// Test 5: Missing JOIN reference
// =============================================================================

func TestLeidenCTE_MissingJoinReference(t *testing.T) {
	h := newCTEHarness(t, "nodes")
	graph.RegisterEdgeKind("LINK", 10)
	h.insertLabeled("s", "seeds")

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	// CTE defined but not joined.
	sql := `WITH local_clusters AS (
    COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK]->(target)
)
SELECT c.community_id
FROM nodes c`

	src := []byte(sql)
	var doc parser.QueryDoc
	parser.Parse(src, &doc)

	_, err := epoch.BindLeidenCTE(context.Background(), src, &doc, 0)
	if err == nil {
		t.Fatal("expected error for missing JOIN reference")
	}
	t.Logf("missing JOIN: %v", err)

	t.Log("✅ missing JOIN reference rejected")
}

// =============================================================================
// Test 6: Join alias preservation
// =============================================================================

func TestLeidenCTE_JoinAliasPreservation(t *testing.T) {
	h := newCTEHarness(t, "nodes")
	graph.RegisterEdgeKind("LINK", 10)
	h.insertLabeled("s", "seeds")

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	tests := []struct {
		joinClause string
		wantAlias  string
	}{
		{
			"JOIN local_clusters c ON d.node_id = c.node_id",
			"c",
		},
		{
			"JOIN local_clusters clusters ON d.node_id = clusters.node_id",
			"clusters",
		},
	}

	for _, tt := range tests {
		t.Run(tt.wantAlias, func(t *testing.T) {
			sql := `WITH local_clusters AS (
    COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK]->(target)
)
SELECT d.title, c.community_id
FROM nodes d
` + tt.joinClause

			src := []byte(sql)
			var doc parser.QueryDoc
			if err := parser.Parse(src, &doc); err != nil {
				t.Fatalf("Parse: %v", err)
			}

			bound, err := epoch.BindLeidenCTE(context.Background(), src, &doc, 0)
			if err != nil {
				t.Fatalf("BindLeidenCTE: %v", err)
			}
			if bound.JoinAlias != tt.wantAlias {
				t.Errorf("JoinAlias: want %q, got %q", tt.wantAlias, bound.JoinAlias)
			}
		})
	}

	t.Log("✅ join alias preservation")
}

// =============================================================================
// Test 7: Snapshot isolation
// =============================================================================

func TestLeidenCTE_SnapshotIsolation(t *testing.T) {
	h := newCTEHarness(t, "nodes")
	graph.RegisterEdgeKind("LINK", 10)
	h.insertLabeled("pre", "seeds")

	t0 := time.Now().UTC()
	time.Sleep(10 * time.Millisecond)

	postNID := h.insertLabeled("post", "seeds")
	_ = postNID

	epoch, err := h.db.BeginEpochTxAt(context.Background(), t0)
	if err != nil {
		t.Fatalf("BeginEpochTxAt: %v", err)
	}
	defer epoch.Rollback(context.Background())

	sql := `WITH local_clusters AS (
    COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK*1..1]->(target)
)
SELECT d.title, c.community_id
FROM nodes d
JOIN local_clusters c ON d.node_id = c.node_id`

	src := []byte(sql)
	var doc parser.QueryDoc
	parser.Parse(src, &doc)

	bound, err := epoch.BindLeidenCTE(context.Background(), src, &doc, 0)
	if err != nil {
		t.Fatalf("BindLeidenCTE: %v", err)
	}

	// Post-t0 seed must not be in bound seeds.
	for _, nid := range bound.Plan.Spec.SeedNodeIDs {
		if nid == postNID {
			t.Fatal("post-t0 seed must not appear in snapshot-isolated bind")
		}
	}

	t.Logf("snapshot isolation: %d seeds (post-t0 excluded)", len(bound.Plan.Spec.SeedNodeIDs))
	t.Log("✅ snapshot isolation")
}

// =============================================================================
// Test 8: Closed/cancelled epoch
// =============================================================================

func TestLeidenCTE_ClosedAndCancelled(t *testing.T) {
	h := newCTEHarness(t, "nodes")
	graph.RegisterEdgeKind("LINK", 10)
	h.insertLabeled("s", "seeds")

	src := []byte(`WITH c AS (COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK]->(target)) SELECT 1 FROM nodes d JOIN c x ON d.node_id = x.node_id`)
	var doc parser.QueryDoc
	parser.Parse(src, &doc)

	// Closed epoch.
	epoch, _ := h.db.BeginEpochTx(context.Background())
	epoch.Rollback(context.Background())
	_, err := epoch.BindLeidenCTE(context.Background(), src, &doc, 0)
	if err == nil {
		t.Fatal("expected error for closed epoch")
	}
	t.Logf("closed epoch: %v", err)

	// Cancelled context.
	epoch2, _ := h.db.BeginEpochTx(context.Background())
	defer epoch2.Rollback(context.Background())
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = epoch2.BindLeidenCTE(ctx, src, &doc, 0)
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
	t.Logf("cancelled context: %v", err)

	t.Log("✅ closed/cancelled epoch")
}

// =============================================================================
// Test 9: No mutation
// =============================================================================

func TestLeidenCTE_NoMutation(t *testing.T) {
	h := newCTEHarness(t, "nodes")
	graph.RegisterEdgeKind("LINK", 10)
	h.insertLabeled("s", "seeds")

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	genBefore := epoch.generation
	recordsBefore, _ := epoch.ListRecords(context.Background(), "nodes")

	src := []byte(`WITH c AS (COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK]->(target)) SELECT 1 FROM nodes d JOIN c x ON d.node_id = x.node_id`)
	var doc parser.QueryDoc
	parser.Parse(src, &doc)

	for i := 0; i < 5; i++ {
		_, err := epoch.BindLeidenCTE(context.Background(), src, &doc, i%1) // always index 0
		if err != nil {
			t.Fatalf("bind %d: %v", i, err)
		}
	}

	if epoch.generation != genBefore {
		t.Errorf("generation changed: %d → %d", genBefore, epoch.generation)
	}
	recordsAfter, _ := epoch.ListRecords(context.Background(), "nodes")
	if len(recordsAfter) != len(recordsBefore) {
		t.Errorf("record count changed: %d → %d", len(recordsBefore), len(recordsAfter))
	}

	t.Log("✅ no mutation across 5 binds")
}

// =============================================================================
// Test 10: Determinism
// =============================================================================

func TestLeidenCTE_Determinism(t *testing.T) {
	h := newCTEHarness(t, "nodes")
	graph.RegisterEdgeKind("LINK", 10)
	h.insertLabeled("s1", "seeds")
	h.insertLabeled("s2", "seeds")

	epoch, _ := h.db.BeginEpochTx(context.Background())
	defer epoch.Rollback(context.Background())

	sql := `WITH local_clusters AS (
    COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK*1..2]->(target)
)
SELECT d.title, c.community_id
FROM nodes d
JOIN local_clusters c ON d.node_id = c.node_id`

	var first *BoundLeidenCTE
	for i := 0; i < 10; i++ {
		src := []byte(sql)
		var doc parser.QueryDoc
		parser.Parse(src, &doc)
		bound, err := epoch.BindLeidenCTE(context.Background(), src, &doc, 0)
		if err != nil {
			t.Fatalf("bind %d: %v", i, err)
		}
		if i == 0 {
			first = bound
			continue
		}
		if bound.Name != first.Name {
			t.Fatalf("call %d: Name differs", i)
		}
		if bound.Collection != first.Collection {
			t.Fatalf("call %d: Collection differs", i)
		}
		if bound.JoinAlias != first.JoinAlias {
			t.Fatalf("call %d: JoinAlias differs", i)
		}
		if !reflect.DeepEqual(bound.Plan.Spec.SeedNodeIDs, first.Plan.Spec.SeedNodeIDs) {
			t.Fatalf("call %d: SeedNodeIDs differ", i)
		}
		if !reflect.DeepEqual(bound.Plan.Spec.EdgeKinds, first.Plan.Spec.EdgeKinds) {
			t.Fatalf("call %d: EdgeKinds differ", i)
		}
	}

	t.Log("✅ determinism across 10 calls")
}
