package libravdb

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/graph"
)

// snapshotAtLatest returns a stable timestamp that can be fed back through
// the public AS OF TIMESTAMP syntax. SnapshotAt also pins the history while a
// test is making its assertions, which keeps these tests independent of any
// retention policy a caller may configure.
func snapshotAtLatest(t *testing.T, db *Database) *TemporalSnapshot {
	t.Helper()
	snap, err := db.SnapshotAt(context.Background(), time.Now().UTC().Add(time.Second))
	if err != nil {
		t.Fatalf("SnapshotAt(latest): %v", err)
	}
	return snap
}

func resultIDs(results *SearchResults) map[string]bool {
	ids := make(map[string]bool)
	if results == nil {
		return ids
	}
	for _, result := range results.Results {
		if result != nil {
			ids[result.ID] = true
		}
	}
	return ids
}

func waitForTemporalGraphState(t *testing.T, db *Database, g Graph, source, target uint64, wantEdge bool) *TemporalSnapshot {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		snap, err := db.SnapshotAt(context.Background(), time.Now().UTC().Add(time.Second))
		if err == nil {
			edges, edgeErr := g.NeighborsAtLSN(source, snap.LSN)
			if edgeErr == nil {
				present := false
				for _, edge := range edges {
					if edge.Target == target {
						present = true
						break
					}
				}
				if present == wantEdge {
					return snap
				}
			}
			snap.Close()
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("graph state did not become present=%v before deadline", wantEdge)
	return nil
}

// TestTemporalAcceptanceMatrix_HistoricalRecordsAndVectorRanking covers the
// record MVCC boundary and the exact historical vector path together. The
// record inserted after snap1 must not affect either the visible rows or the
// ranking at snap1.
func TestTemporalAcceptanceMatrix_HistoricalRecordsAndVectorRanking(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(t.TempDir() + "/records-and-vectors.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(ctx)

	col, err := db.CreateCollection(ctx, "docs", WithDimension(3), WithMetric(L2Distance))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	if err := col.Insert(ctx, "old", []float32{1, 0, 0}, map[string]interface{}{"version": "v1"}); err != nil {
		t.Fatalf("insert old: %v", err)
	}
	if err := col.Insert(ctx, "anchor", []float32{2, 0, 0}, nil); err != nil {
		t.Fatalf("insert anchor: %v", err)
	}
	snap1 := snapshotAtLatest(t, db)
	defer snap1.Close()

	// Both mutations occur after snap1: one changes an existing version and
	// one creates a record that must be absent from the historical view.
	if err := col.Update(ctx, "old", []float32{10, 0, 0}, map[string]interface{}{"version": "v2"}); err != nil {
		t.Fatalf("update old: %v", err)
	}
	if err := col.Insert(ctx, "future", []float32{0, 0, 0}, nil); err != nil {
		t.Fatalf("insert future: %v", err)
	}

	historical, err := col.GetAtLSN(ctx, "old", snap1.LSN)
	if err != nil {
		t.Fatalf("GetAtLSN(old): %v", err)
	}
	if historical == nil || historical.Vector[0] != 1 || historical.Metadata["version"] != "v1" {
		t.Fatalf("historical old=%+v, want v1/[1,0,0]", historical)
	}
	future, err := col.GetAtLSN(ctx, "future", snap1.LSN)
	if err != nil {
		t.Fatalf("GetAtLSN(future): %v", err)
	}
	if future != nil {
		t.Fatalf("post-snapshot record leaked into historical read: %+v", future)
	}

	var visible []string
	if err := col.ListVisibleAtLSN(ctx, snap1.LSN, func(record *Record) bool {
		visible = append(visible, record.ID)
		return true
	}); err != nil {
		t.Fatalf("ListVisibleAtLSN: %v", err)
	}
	visibleIDs := make(map[string]bool, len(visible))
	for _, id := range visible {
		visibleIDs[id] = true
	}
	if len(visible) != 2 || !visibleIDs["old"] || !visibleIDs["anchor"] {
		t.Fatalf("historical visible IDs=%v, want old and anchor", visible)
	}
	if visibleIDs["future"] {
		t.Fatalf("future record present in historical IDs=%v", visible)
	}

	sql := fmt.Sprintf("SELECT id, VECTOR_DISTANCE(embedding, '[0,0,0]') AS distance "+
		"FROM docs AS OF TIMESTAMP '%s' ORDER BY distance ASC LIMIT 10", snap1.Timestamp.Format(time.RFC3339Nano))
	results, err := db.Query(ctx, sql)
	if err != nil {
		t.Fatalf("historical vector query: %v", err)
	}
	if results.Total != 2 || len(results.Results) != 2 {
		t.Fatalf("historical vector rows=%d, want 2: %+v", results.Total, results.Results)
	}
	if results.Results[0].ID != "old" || results.Results[1].ID != "anchor" {
		t.Fatalf("historical vector order=%q,%q, want old,anchor", results.Results[0].ID, results.Results[1].ID)
	}
	if got := results.Results[0].Score; got != float32(1) {
		t.Fatalf("historical old distance=%v, want 1; result=%+v columns=%v", got, results.Results[0], results.Columns)
	}
}

// TestTemporalAcceptanceMatrix_HistoricalGraphVisibility verifies both the
// graph's LSN reader and SQL MATCH use the same historical topology.
func TestTemporalAcceptanceMatrix_HistoricalGraphVisibility(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(t.TempDir() + "/graph.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(ctx)

	g, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	defer g.Close()
	col, err := db.CreateCollection(ctx, "nodes", WithMetadataOnly(), WithGraph(g),
		WithMetadataSchema(MetadataSchema{"kind": StringField}))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	const edgeKind uint8 = 207
	const edgeName = "TEMPORAL_MATRIX_LINK"
	if !graph.RegisterEdgeKind(edgeName, edgeKind) && graph.ResolveEdgeKind(edgeName) != edgeKind {
		t.Fatalf("edge kind %q already has a different value", edgeName)
	}
	if err := col.Insert(ctx, "source", nil, map[string]interface{}{"kind": "source"}); err != nil {
		t.Fatalf("insert source: %v", err)
	}
	if err := col.Insert(ctx, "target", nil, map[string]interface{}{"kind": "target"}); err != nil {
		t.Fatalf("insert target: %v", err)
	}
	source, err := db.GetNodeID(ctx, "nodes", "source")
	if err != nil {
		t.Fatalf("source node: %v", err)
	}
	target, err := db.GetNodeID(ctx, "nodes", "target")
	if err != nil {
		t.Fatalf("target node: %v", err)
	}
	g.RegisterVertexLabel(target, "matrix_target")
	txn := g.BeginTxn()
	if err := txn.AddEdge(source, target, 1, edgeKind); err != nil {
		t.Fatalf("add edge: %v", err)
	}
	if err := txn.Commit(ctx); err != nil {
		t.Fatalf("commit edge: %v", err)
	}
	snap1 := waitForTemporalGraphState(t, db, g, source, target, true)
	defer snap1.Close()

	edges, err := g.NeighborsAtLSN(source, snap1.LSN)
	if err != nil || len(edges) != 1 || edges[0].Target != target {
		t.Fatalf("historical neighbors=%+v, err=%v", edges, err)
	}
	queryAt := func(ts time.Time) (*SearchResults, error) {
		return db.Query(ctx, fmt.Sprintf("SELECT s.kind FROM nodes s AS OF TIMESTAMP '%s' "+
			"WHERE MATCH (s)-[:%s]->(p:matrix_target) LIMIT 10", ts.Format(time.RFC3339Nano), edgeName))
	}
	rows, err := queryAt(snap1.Timestamp)
	if err != nil {
		t.Fatalf("historical graph SQL: %v", err)
	}
	if rows.Total != 1 || rows.Results[0].ID != "source" {
		t.Fatalf("historical graph SQL rows=%+v, want source", rows.Results)
	}

	txn = g.BeginTxn()
	if err := txn.RemoveEdge(source, target, edgeKind); err != nil {
		t.Fatalf("remove edge: %v", err)
	}
	if err := txn.Commit(ctx); err != nil {
		t.Fatalf("commit edge removal: %v", err)
	}
	snap2 := waitForTemporalGraphState(t, db, g, source, target, false)
	defer snap2.Close()
	oldEdges, err := g.NeighborsAtLSN(source, snap1.LSN)
	if err != nil || len(oldEdges) != 1 {
		t.Fatalf("old graph after removal=%+v, err=%v", oldEdges, err)
	}
	newRows, err := queryAt(snap2.Timestamp)
	if err != nil {
		t.Fatalf("post-removal graph SQL: %v", err)
	}
	if newRows.Total != 0 {
		t.Fatalf("post-removal historical graph rows=%d, want 0", newRows.Total)
	}
}

// TestTemporalAcceptanceMatrix_HistoricalEpochOverlay exercises staged
// records and graph operations over a pinned historical base. Writes made
// after the base snapshot remain invisible unless explicitly staged in the
// epoch.
func TestTemporalAcceptanceMatrix_HistoricalEpochOverlay(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(t.TempDir() + "/historical-epoch.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(ctx)

	g, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	defer g.Close()
	col, err := db.CreateCollection(ctx, "docs", WithDimension(3), WithGraph(g))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	if err := col.Insert(ctx, "base", []float32{1, 0, 0}, nil); err != nil {
		t.Fatalf("insert base: %v", err)
	}
	baseNode, err := db.GetNodeID(ctx, "docs", "base")
	if err != nil {
		t.Fatalf("base node: %v", err)
	}
	if err := col.Insert(ctx, "target", []float32{2, 0, 0}, nil); err != nil {
		t.Fatalf("insert target: %v", err)
	}
	targetNode, err := db.GetNodeID(ctx, "docs", "target")
	if err != nil {
		t.Fatalf("target node: %v", err)
	}
	const edgeKind uint8 = 208
	const edgeName = "TEMPORAL_MATRIX_EPOCH_LINK"
	if !graph.RegisterEdgeKind(edgeName, edgeKind) && graph.ResolveEdgeKind(edgeName) != edgeKind {
		t.Fatalf("edge kind %q already has a different value", edgeName)
	}
	txn := g.BeginTxn()
	if err := txn.AddEdge(baseNode, targetNode, 1, edgeKind); err != nil {
		t.Fatalf("add base edge: %v", err)
	}
	if err := txn.Commit(ctx); err != nil {
		t.Fatalf("commit base edge: %v", err)
	}
	baseSnap := waitForTemporalGraphState(t, db, g, baseNode, targetNode, true)
	baseTime := baseSnap.Timestamp
	baseLSN := baseSnap.LSN
	baseSnap.Close()

	if err := col.Insert(ctx, "post-snapshot", []float32{0, 0, 1}, nil); err != nil {
		t.Fatalf("insert post-snapshot: %v", err)
	}
	epoch, err := db.BeginEpochTxAt(ctx, baseTime)
	if err != nil {
		t.Fatalf("BeginEpochTxAt: %v", err)
	}
	defer func() { _ = epoch.Rollback(ctx) }()
	if epoch.SnapshotLSN() != baseLSN {
		t.Fatalf("epoch LSN=%d, want base LSN=%d", epoch.SnapshotLSN(), baseLSN)
	}

	baseRecords, err := epoch.ListRecords(ctx, "docs")
	if err != nil {
		t.Fatalf("historical ListRecords: %v", err)
	}
	baseIDs := recordsToIDs(baseRecords)
	if baseIDs["post-snapshot"] {
		t.Fatalf("post-snapshot record leaked into epoch: %v", baseIDs)
	}
	if !baseIDs["base"] || !baseIDs["target"] {
		t.Fatalf("epoch base records=%v, want base and target", baseIDs)
	}

	if err := epoch.Update(ctx, "docs", "base", []float32{3, 0, 0}, nil); err != nil {
		t.Fatalf("stage historical update: %v", err)
	}
	if err := epoch.Insert(ctx, "docs", "staged", []float32{0, 1, 0}, nil); err != nil {
		t.Fatalf("stage insert: %v", err)
	}
	stagedNode, err := epoch.LookupNodeID(ctx, "docs", "staged")
	if err != nil {
		t.Fatalf("staged node: %v", err)
	}
	gtx, err := epoch.GraphTxn("docs")
	if err != nil {
		t.Fatalf("GraphTxn: %v", err)
	}
	if err := gtx.AddEdge(baseNode, stagedNode, 1, edgeKind); err != nil {
		t.Fatalf("stage graph edge: %v", err)
	}
	neighbors, err := gtx.NeighborsOverlay(baseNode)
	if err != nil {
		t.Fatalf("NeighborsOverlay: %v", err)
	}
	if len(neighbors) != 2 {
		t.Fatalf("historical epoch neighbors=%+v, want base and staged edges", neighbors)
	}

	finalRecords, err := epoch.ListRecords(ctx, "docs")
	if err != nil {
		t.Fatalf("staged ListRecords: %v", err)
	}
	byID := make(map[string]Record, len(finalRecords))
	for _, record := range finalRecords {
		byID[record.ID] = record
	}
	if len(byID) != 3 || byID["base"].Vector[0] != 3 || byID["staged"].Vector[1] != 1 || byID["post-snapshot"].ID != "" {
		t.Fatalf("staged historical records=%+v", byID)
	}
}

func recordsToIDs(records []Record) map[string]bool {
	ids := make(map[string]bool, len(records))
	for _, record := range records {
		ids[record.ID] = true
	}
	return ids
}

// TestTemporalAcceptanceMatrix_HistoricalSavepointRollback ensures a branch
// can be discarded without changing the pinned historical base or admitting
// records committed after that base.
func TestTemporalAcceptanceMatrix_HistoricalSavepointRollback(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(t.TempDir() + "/historical-savepoint.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(ctx)
	col, err := db.CreateCollection(ctx, "docs", WithDimension(3))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	if err := col.Insert(ctx, "base", []float32{1, 0, 0}, nil); err != nil {
		t.Fatalf("insert base: %v", err)
	}
	snap := snapshotAtLatest(t, db)
	baseTime := snap.Timestamp
	snap.Close()
	if err := col.Insert(ctx, "post-snapshot", []float32{0, 0, 1}, nil); err != nil {
		t.Fatalf("insert post-snapshot: %v", err)
	}

	epoch, err := db.BeginEpochTxAt(ctx, baseTime)
	if err != nil {
		t.Fatalf("BeginEpochTxAt: %v", err)
	}
	defer func() { _ = epoch.Rollback(ctx) }()
	if err := epoch.Savepoint("before-branch"); err != nil {
		t.Fatalf("Savepoint: %v", err)
	}
	if err := epoch.Update(ctx, "docs", "base", []float32{9, 0, 0}, nil); err != nil {
		t.Fatalf("branch update: %v", err)
	}
	if err := epoch.Insert(ctx, "docs", "branch", []float32{0, 1, 0}, nil); err != nil {
		t.Fatalf("branch insert: %v", err)
	}
	branch, err := epoch.ListRecords(ctx, "docs")
	if err != nil {
		t.Fatalf("branch ListRecords: %v", err)
	}
	branchIDs := recordsToIDs(branch)
	if branchIDs["post-snapshot"] {
		t.Fatalf("post-snapshot record visible in branch: %v", branchIDs)
	}
	if !branchIDs["branch"] {
		t.Fatalf("staged branch record missing: %v", branchIDs)
	}

	if err := epoch.RollbackTo("before-branch"); err != nil {
		t.Fatalf("RollbackTo: %v", err)
	}
	restored, err := epoch.ListRecords(ctx, "docs")
	if err != nil {
		t.Fatalf("restored ListRecords: %v", err)
	}
	restoredIDs := recordsToIDs(restored)
	if len(restoredIDs) != 1 || !restoredIDs["base"] || restoredIDs["branch"] || restoredIDs["post-snapshot"] {
		t.Fatalf("after savepoint rollback IDs=%v, want only base", restoredIDs)
	}
	if restored[0].Vector[0] != 1 {
		t.Fatalf("after savepoint rollback base=%v, want original vector", restored[0].Vector)
	}
}

// TestTemporalAcceptanceMatrix_ConcurrentSessions verifies that two epoch
// sessions have independent read-your-writes overlays and remain pinned to
// their own starting snapshot even after the other session commits.
func TestTemporalAcceptanceMatrix_ConcurrentSessions(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(t.TempDir() + "/concurrent-sessions.libravdb"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Drop(ctx)
	col, err := db.CreateCollection(ctx, "docs", WithDimension(3))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	if err := col.Insert(ctx, "base", []float32{1, 0, 0}, nil); err != nil {
		t.Fatalf("insert base: %v", err)
	}
	s1, err := db.NewSQLSession(ctx)
	if err != nil {
		t.Fatalf("NewSQLSession s1: %v", err)
	}
	defer s1.Close()
	s2, err := db.NewSQLSession(ctx)
	if err != nil {
		t.Fatalf("NewSQLSession s2: %v", err)
	}
	defer s2.Close()
	if err := s1.Exec("BEGIN EPOCH TRANSACTION"); err != nil {
		t.Fatalf("s1 begin: %v", err)
	}
	if err := s2.Exec("BEGIN EPOCH TRANSACTION"); err != nil {
		t.Fatalf("s2 begin: %v", err)
	}

	errCh := make(chan error, 2)
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		errCh <- s1.Exec("INSERT INTO docs (id, embedding) VALUES ('s1-only', '[2,0,0]')")
	}()
	go func() {
		defer wg.Done()
		errCh <- s2.Exec("INSERT INTO docs (id, embedding) VALUES ('s2-only', '[3,0,0]')")
	}()
	wg.Wait()
	close(errCh)
	for err := range errCh {
		if err != nil {
			t.Fatalf("concurrent staged insert: %v", err)
		}
	}

	checkSession := func(label string, session *SQLSession, own, other string) {
		t.Helper()
		results, queryErr := session.Query("SELECT id FROM docs")
		if queryErr != nil {
			t.Fatalf("%s query: %v", label, queryErr)
		}
		ids := resultIDs(results)
		if !ids["base"] || !ids[own] || ids[other] {
			t.Fatalf("%s IDs=%v, want base+%s and not %s", label, ids, own, other)
		}
	}
	checkSession("s1", s1, "s1-only", "s2-only")
	checkSession("s2", s2, "s2-only", "s1-only")

	if err := s1.Exec("COMMIT"); err != nil {
		t.Fatalf("s1 commit: %v", err)
	}
	// s2's epoch began before s1 committed, so s1-only remains invisible to it.
	checkSession("s2 after s1 commit", s2, "s2-only", "s1-only")
	if err := s2.Exec("COMMIT"); err != nil {
		t.Fatalf("s2 commit: %v", err)
	}
	live, err := db.Query(ctx, "SELECT id FROM docs")
	if err != nil {
		t.Fatalf("live query: %v", err)
	}
	ids := resultIDs(live)
	if !ids["base"] || !ids["s1-only"] || !ids["s2-only"] {
		t.Fatalf("live IDs=%v, want all committed rows", ids)
	}
}
