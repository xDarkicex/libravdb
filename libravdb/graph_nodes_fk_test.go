package libravdb

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

func TestGraphNodesVirtualRelationAndReadOnly(t *testing.T) {
	ctx := context.Background()
	db := openTempDB(t, "graph_nodes_virtual")
	defer db.Close()

	g, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	col, err := db.CreateCollection(ctx, "docs", WithMetadataOnly(), WithGraph(g))
	if err != nil {
		t.Fatal(err)
	}
	if err := col.Insert(ctx, "doc-a", nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := col.Insert(ctx, "doc-b", nil, nil); err != nil {
		t.Fatal(err)
	}

	rows, err := db.Query(ctx, "SELECT id, collection, record_id FROM GRAPH_NODES")
	if err != nil {
		t.Fatalf("SELECT GRAPH_NODES: %v", err)
	}
	if len(rows.Results) != 2 || rows.Columns == nil || len(rows.Columns) != 3 {
		t.Fatalf("GRAPH_NODES rows/columns = %d/%v, want 2/3", len(rows.Results), rows.Columns)
	}
	if rows.Results[0].Metadata["collection"] != "docs" || rows.Results[0].Metadata["record_id"] == nil {
		t.Fatalf("unexpected GRAPH_NODES row: %#v", rows.Results[0].Metadata)
	}
	first, _ := db.GetNodeID(ctx, "docs", "doc-a")
	second, _ := db.GetNodeID(ctx, "docs", "doc-b")
	if first == 0 || second == 0 || first >= second {
		t.Fatalf("GRAPH_NODES ordering/IDs = %d, %d", first, second)
	}

	for _, sql := range []string{
		"INSERT INTO GRAPH_NODES (id, collection, record_id) VALUES (999, 'x', 'y')",
		"UPDATE GRAPH_NODES SET collection = 'x' WHERE id = 1",
		"DELETE FROM GRAPH_NODES WHERE id = 1",
	} {
		if _, err := db.Query(ctx, sql); err == nil || !strings.Contains(strings.ToLower(err.Error()), "read-only") {
			t.Fatalf("%s: expected read-only error, got %v", sql, err)
		}
	}
}

func TestGraphNodesForeignKeyDeleteCascadeIsAtomic(t *testing.T) {
	ctx := context.Background()
	db := openTempDB(t, "graph_nodes_fk_cascade")
	defer db.Close()

	g, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	col, err := db.CreateCollection(ctx, "docs", WithMetadataOnly(), WithGraph(g))
	if err != nil {
		t.Fatal(err)
	}
	if err := col.Insert(ctx, "doc-a", nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := col.Insert(ctx, "doc-b", nil, nil); err != nil {
		t.Fatal(err)
	}
	src, _ := db.GetNodeID(ctx, "docs", "doc-a")
	tgt, _ := db.GetNodeID(ctx, "docs", "doc-b")
	gtx := g.BeginTxn()
	if err := g.AddEdge(gtx, src, tgt, 1, 1); err != nil {
		t.Fatal(err)
	}
	if err := gtx.Commit(ctx); err != nil {
		t.Fatal(err)
	}

	if _, err := db.Query(ctx, "CREATE TABLE refs (id TEXT PRIMARY KEY, graph_id BIGINT REFERENCES GRAPH_NODES(id) ON DELETE CASCADE)"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, fmt.Sprintf("INSERT INTO refs (id, graph_id) VALUES ('r1', %d)", src)); err != nil {
		t.Fatalf("valid GRAPH_NODES FK insert: %v", err)
	}
	refsCol, _ := db.GetCollection("refs")
	if _, err := db.Query(ctx, "INSERT INTO refs (id, graph_id) VALUES ('bad', 999999)"); err == nil {
		t.Fatal("invalid GRAPH_NODES FK insert succeeded")
	}

	deleteResult, deleteErr := db.Query(ctx, "DELETE FROM docs WHERE id = 'doc-a'")
	if deleteErr != nil {
		t.Fatalf("parent delete cascade: %v", deleteErr)
	}
	if deleteResult == nil {
		t.Fatal("parent delete returned nil result")
	}
	rows, err := db.Query(ctx, "SELECT id FROM refs WHERE id = 'r1'")
	if err != nil {
		t.Fatal(err)
	}
	if len(rows.Results) != 0 {
		t.Fatalf("cascaded child is still query-visible: %d rows", len(rows.Results))
	}
	if _, err := refsCol.Get(ctx, "r1"); err == nil {
		t.Fatal("cascaded child still exists")
	}
	if got, err := g.Neighbors(src); err != nil {
		t.Fatal(err)
	} else if len(got) != 0 {
		t.Fatalf("graph edge survived GRAPH_NODES cascade: %#v", got)
	}
	if got, err := g.InboundNeighbors(tgt); err != nil {
		t.Fatal(err)
	} else if len(got) != 0 {
		t.Fatalf("inbound graph edge survived GRAPH_NODES cascade: %#v", got)
	}
	if _, _, err := db.ResolveNodeID(ctx, src); err == nil {
		t.Fatal("deleted graph node still resolves")
	}
}

func TestGraphNodesEpochOverlayAndSavepointRollback(t *testing.T) {
	ctx := context.Background()
	db := openTempDB(t, "graph_nodes_epoch_overlay")
	defer db.Close()

	g, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	col, err := db.CreateCollection(ctx, "docs", WithMetadataOnly(), WithGraph(g))
	if err != nil {
		t.Fatal(err)
	}
	if err := col.Insert(ctx, "base", nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := col.Insert(ctx, "target", nil, nil); err != nil {
		t.Fatal(err)
	}
	baseID, _ := db.GetNodeID(ctx, "docs", "base")
	targetID, _ := db.GetNodeID(ctx, "docs", "target")
	baseGraphTxn := g.BeginTxn()
	if err := baseGraphTxn.AddEdge(baseID, targetID, 1, 1); err != nil {
		t.Fatal(err)
	}
	if err := baseGraphTxn.Commit(ctx); err != nil {
		t.Fatal(err)
	}

	epoch, err := db.BeginEpochTx(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if err := epoch.Savepoint("before_branch"); err != nil {
		t.Fatal(err)
	}
	if err := epoch.Insert(ctx, "docs", "staged", nil, nil); err != nil {
		t.Fatal(err)
	}
	stagedID, err := epoch.LookupNodeID(ctx, "docs", "staged")
	if err != nil || stagedID == 0 {
		t.Fatalf("staged graph node ID = %d, %v", stagedID, err)
	}

	rows, err := epoch.Query(ctx, "SELECT id, collection, record_id FROM GRAPH_NODES", QueryParams{})
	if err != nil {
		t.Fatalf("epoch GRAPH_NODES query: %v", err)
	}
	if len(rows.Results) != 3 {
		t.Fatalf("epoch GRAPH_NODES rows = %d, want 3", len(rows.Results))
	}
	if err := epoch.Delete(ctx, "docs", "base"); err != nil {
		t.Fatal(err)
	}
	if _, err := epoch.LookupNodeID(ctx, "docs", "base"); err == nil {
		t.Fatal("epoch-deleted graph node still resolves")
	}
	if _, _, err := epoch.ResolveNodeID(ctx, dbNodeID(t, db, "docs", "base")); err == nil {
		t.Fatal("epoch-deleted durable graph node still resolves")
	}
	rows, err = epoch.Query(ctx, "SELECT id, collection, record_id FROM GRAPH_NODES", QueryParams{})
	if err != nil {
		t.Fatalf("epoch GRAPH_NODES after delete: %v", err)
	}
	if len(rows.Results) != 2 {
		t.Fatalf("epoch GRAPH_NODES after delete = %#v, want staged+target", rows.Results)
	}

	if err := epoch.RollbackTo("before_branch"); err != nil {
		t.Fatal(err)
	}
	if _, err := epoch.LookupNodeID(ctx, "docs", "staged"); err == nil {
		t.Fatal("rolled-back provisional graph node still resolves")
	}
	rows, err = epoch.Query(ctx, "SELECT id, collection, record_id FROM GRAPH_NODES", QueryParams{})
	if err != nil {
		t.Fatalf("epoch GRAPH_NODES after rollback: %v", err)
	}
	if len(rows.Results) != 2 {
		t.Fatalf("epoch GRAPH_NODES after rollback = %#v, want base+target", rows.Results)
	}
	if err := epoch.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	live, err := db.Query(ctx, "SELECT id, collection, record_id FROM GRAPH_NODES")
	if err != nil {
		t.Fatal(err)
	}
	if len(live.Results) != 2 {
		t.Fatalf("live GRAPH_NODES after epoch rollback = %#v", live.Results)
	}
	if got, err := g.Neighbors(baseID); err != nil {
		t.Fatal(err)
	} else if len(got) != 1 || got[0].Target != targetID {
		t.Fatalf("savepoint rollback leaked graph-node drop: %#v", got)
	}
}

func TestGraphNodesEpochForeignKeyAcceptsProvisionalNode(t *testing.T) {
	ctx := context.Background()
	db := openTempDB(t, "graph_nodes_epoch_fk")
	defer db.Close()

	g, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	if _, err := db.CreateCollection(ctx, "docs", WithMetadataOnly(), WithGraph(g)); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, "CREATE TABLE refs (id TEXT PRIMARY KEY, graph_id BIGINT REFERENCES GRAPH_NODES(id) ON DELETE CASCADE)"); err != nil {
		t.Fatal(err)
	}
	epoch, err := db.BeginEpochTx(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if err := epoch.Insert(ctx, "docs", "staged", nil, nil); err != nil {
		t.Fatal(err)
	}
	nodeID, err := epoch.LookupNodeID(ctx, "docs", "staged")
	if err != nil {
		t.Fatal(err)
	}
	if err := epoch.Insert(ctx, "refs", "r1", nil, map[string]interface{}{
		"graph_id": fmt.Sprintf("%d", nodeID),
	}); err != nil {
		t.Fatalf("FK to provisional GRAPH_NODES node rejected: %v", err)
	}
	if err := epoch.Delete(ctx, "docs", "staged"); err != nil {
		t.Fatalf("delete of staged GRAPH_NODES parent: %v", err)
	}
	refs, err := epoch.Query(ctx, "SELECT id FROM refs WHERE id = 'r1'", QueryParams{})
	if err != nil {
		t.Fatal(err)
	}
	if len(refs.Results) != 0 {
		t.Fatal("epoch GRAPH_NODES cascade left child of staged parent")
	}
	nodes, err := epoch.Query(ctx, "SELECT id FROM GRAPH_NODES", QueryParams{})
	if err != nil {
		t.Fatal(err)
	}
	if len(nodes.Results) != 0 {
		t.Fatal("epoch delete left staged GRAPH_NODES parent")
	}
	if err := epoch.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	rows, err := db.Query(ctx, "SELECT id FROM refs WHERE id = 'r1'")
	if err != nil {
		t.Fatal(err)
	}
	if len(rows.Results) != 0 {
		t.Fatal("rolled-back provisional GRAPH_NODES FK child became durable")
	}
}

func TestGraphNodesDirectCollectionDeleteCascadesAtomically(t *testing.T) {
	ctx := context.Background()
	db := openTempDB(t, "graph_nodes_direct_delete")
	defer db.Close()

	g, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	col, err := db.CreateCollection(ctx, "docs", WithMetadataOnly(), WithGraph(g))
	if err != nil {
		t.Fatal(err)
	}
	if err := col.Insert(ctx, "parent", nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := col.Insert(ctx, "target", nil, nil); err != nil {
		t.Fatal(err)
	}
	src, _ := db.GetNodeID(ctx, "docs", "parent")
	tgt, _ := db.GetNodeID(ctx, "docs", "target")
	gtx := g.BeginTxn()
	if err := gtx.AddEdge(src, tgt, 1, 1); err != nil {
		t.Fatal(err)
	}
	if err := gtx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, "CREATE TABLE refs (id TEXT PRIMARY KEY, graph_id BIGINT REFERENCES GRAPH_NODES(id) ON DELETE CASCADE)"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, fmt.Sprintf("INSERT INTO refs (id, graph_id) VALUES ('r1', %d)", src)); err != nil {
		t.Fatal(err)
	}
	if err := col.Delete(ctx, "parent"); err != nil {
		t.Fatalf("direct graph collection delete: %v", err)
	}
	rows, err := db.Query(ctx, "SELECT id FROM refs WHERE id = 'r1'")
	if err != nil {
		t.Fatal(err)
	}
	if len(rows.Results) != 0 {
		t.Fatal("direct delete left GRAPH_NODES child")
	}
	if got, err := g.Neighbors(src); err != nil {
		t.Fatal(err)
	} else if len(got) != 0 {
		t.Fatalf("direct delete left outbound graph edges: %#v", got)
	}
	if _, _, err := db.ResolveNodeID(ctx, src); err == nil {
		t.Fatal("direct delete left durable GRAPH_NODES mapping")
	}
}

func TestGraphNodesLogicalTextAndUUIDForeignKeysCascade(t *testing.T) {
	ctx := context.Background()
	db := openTempDB(t, "graph_nodes_logical_fk")
	defer db.Close()

	docsGraph, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer docsGraph.Close()
	docs, err := db.CreateCollection(ctx, "docs", WithMetadataOnly(), WithGraph(docsGraph))
	if err != nil {
		t.Fatal(err)
	}
	if err := docs.Insert(ctx, "doc-text", nil, nil); err != nil {
		t.Fatal(err)
	}

	uuidGraph, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer uuidGraph.Close()
	uuidNodes, err := db.CreateCollection(ctx, "uuid_nodes", WithMetadataOnly(), WithGraph(uuidGraph))
	if err != nil {
		t.Fatal(err)
	}
	uuidID := "550e8400-e29b-41d4-a716-446655440000"
	if err := uuidNodes.Insert(ctx, uuidID, nil, nil); err != nil {
		t.Fatal(err)
	}

	if _, err := db.Query(ctx, "CREATE TABLE text_graph_refs (id TEXT PRIMARY KEY REFERENCES GRAPH_NODES(id) ON DELETE CASCADE)"); err != nil {
		t.Fatalf("TEXT GRAPH_NODES FK DDL: %v", err)
	}
	if _, err := db.Query(ctx, "CREATE TABLE uuid_graph_refs (id UUID PRIMARY KEY REFERENCES GRAPH_NODES(id) ON DELETE CASCADE)"); err != nil {
		t.Fatalf("UUID GRAPH_NODES FK DDL: %v", err)
	}
	if _, err := db.Query(ctx, "INSERT INTO text_graph_refs (id) VALUES ('doc-text')"); err != nil {
		t.Fatalf("TEXT logical GRAPH_NODES FK insert: %v", err)
	}
	if _, err := db.Query(ctx, fmt.Sprintf("INSERT INTO uuid_graph_refs (id) VALUES ('%s')", uuidID)); err != nil {
		t.Fatalf("UUID logical GRAPH_NODES FK insert: %v", err)
	}
	if _, err := db.Query(ctx, "INSERT INTO text_graph_refs (id) VALUES ('missing')"); err == nil {
		t.Fatal("missing logical GRAPH_NODES record was accepted")
	}

	if err := docs.Delete(ctx, "doc-text"); err != nil {
		t.Fatalf("TEXT parent delete cascade: %v", err)
	}
	textRefs, err := db.Query(ctx, "SELECT id FROM text_graph_refs")
	if err != nil {
		t.Fatal(err)
	}
	if len(textRefs.Results) != 0 {
		t.Fatalf("TEXT logical child survived cascade: %#v", textRefs.Results)
	}
	if err := uuidNodes.Delete(ctx, uuidID); err != nil {
		t.Fatalf("UUID parent delete cascade: %v", err)
	}
	uuidRefs, err := db.Query(ctx, "SELECT id FROM uuid_graph_refs")
	if err != nil {
		t.Fatal(err)
	}
	if len(uuidRefs.Results) != 0 {
		t.Fatalf("UUID logical child survived cascade: %#v", uuidRefs.Results)
	}
}

func dbNodeID(t *testing.T, db *Database, collection, id string) uint64 {
	t.Helper()
	nodeID, err := db.GetNodeID(context.Background(), collection, id)
	if err != nil {
		t.Fatal(err)
	}
	return nodeID
}
