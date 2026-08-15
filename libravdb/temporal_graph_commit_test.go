package libravdb

import (
	"context"
	"testing"
)

func TestCombinedEpochRecordAndGraphShareCommitLSN(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(t.TempDir() + "/combined-graph-temporal.libravdb"))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer db.Drop(ctx)

	graphStore, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("new graph: %v", err)
	}
	defer graphStore.Close()
	collection, err := db.CreateCollection(ctx, "docs", WithDimension(1), WithGraph(graphStore))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}
	if err := collection.Insert(ctx, "source", []float32{1}, nil); err != nil {
		t.Fatalf("insert source: %v", err)
	}
	if err := collection.Insert(ctx, "target", []float32{2}, nil); err != nil {
		t.Fatalf("insert target: %v", err)
	}
	source, err := db.GetNodeID(ctx, "docs", "source")
	if err != nil {
		t.Fatalf("source node: %v", err)
	}
	target, err := db.GetNodeID(ctx, "docs", "target")
	if err != nil {
		t.Fatalf("target node: %v", err)
	}

	initialGraph := graphStore.BeginTxn()
	if err := initialGraph.AddEdge(source, target, 1, 10); err != nil {
		t.Fatalf("initial graph edge: %v", err)
	}
	if err := initialGraph.Commit(ctx); err != nil {
		t.Fatalf("initial graph commit: %v", err)
	}
	baseLSN, err := db.LatestCommitLSN(ctx)
	if err != nil {
		t.Fatalf("base LSN: %v", err)
	}

	// The record update and new graph edge share one terminal WAL commit.
	addEpoch, err := db.BeginEpochTx(ctx)
	if err != nil {
		t.Fatalf("begin add epoch: %v", err)
	}
	if err := addEpoch.Update(ctx, "docs", "source", []float32{3}, nil); err != nil {
		t.Fatalf("stage source update: %v", err)
	}
	if err := addEpoch.AddGraphEdge("docs", source, target, 2, 11); err != nil {
		t.Fatalf("stage graph add: %v", err)
	}
	if err := addEpoch.Commit(ctx); err != nil {
		t.Fatalf("commit graph add epoch: %v", err)
	}
	addLSN, err := db.LatestCommitLSN(ctx)
	if err != nil {
		t.Fatalf("add LSN: %v", err)
	}
	if addLSN <= baseLSN {
		t.Fatalf("add LSN %d is not after base LSN %d", addLSN, baseLSN)
	}

	assertVectorAtLSN(t, collection, "source", baseLSN, 1)
	assertVectorAtLSN(t, collection, "source", addLSN, 3)
	assertEdgeKindAtLSN(t, graphStore, source, baseLSN, 11, false)
	assertEdgeKindAtLSN(t, graphStore, source, addLSN, 11, true)

	// Removing the original edge and updating the record must likewise share
	// one LSN. The later node drop exercises the delete/drop publication path.
	removeEpoch, err := db.BeginEpochTx(ctx)
	if err != nil {
		t.Fatalf("begin remove epoch: %v", err)
	}
	if err := removeEpoch.Update(ctx, "docs", "source", []float32{4}, nil); err != nil {
		t.Fatalf("stage second source update: %v", err)
	}
	if err := removeEpoch.RemoveGraphEdge("docs", source, target, 10); err != nil {
		t.Fatalf("stage graph remove: %v", err)
	}
	if err := removeEpoch.Commit(ctx); err != nil {
		t.Fatalf("commit graph remove epoch: %v", err)
	}
	removeLSN, err := db.LatestCommitLSN(ctx)
	if err != nil {
		t.Fatalf("remove LSN: %v", err)
	}
	assertVectorAtLSN(t, collection, "source", removeLSN, 4)
	assertEdgeKindAtLSN(t, graphStore, source, addLSN, 10, true)
	assertEdgeKindAtLSN(t, graphStore, source, removeLSN, 10, false)
	assertEdgeKindAtLSN(t, graphStore, source, removeLSN, 11, true)

	deleteEpoch, err := db.BeginEpochTx(ctx)
	if err != nil {
		t.Fatalf("begin delete epoch: %v", err)
	}
	if err := deleteEpoch.Delete(ctx, "docs", "source"); err != nil {
		t.Fatalf("stage source delete: %v", err)
	}
	if err := deleteEpoch.DropGraphNodeEdges("docs", source); err != nil {
		t.Fatalf("stage node drop: %v", err)
	}
	if err := deleteEpoch.Commit(ctx); err != nil {
		t.Fatalf("commit delete epoch: %v", err)
	}
	deleteLSN, err := db.LatestCommitLSN(ctx)
	if err != nil {
		t.Fatalf("delete LSN: %v", err)
	}
	deleted, err := collection.GetAtLSN(ctx, "source", removeLSN)
	if err != nil || deleted == nil {
		t.Fatalf("source at pre-delete LSN = %+v, err=%v", deleted, err)
	}
	deleted, err = collection.GetAtLSN(ctx, "source", deleteLSN)
	if err != nil {
		t.Fatalf("source at delete LSN: %v", err)
	}
	if deleted != nil {
		t.Fatalf("source still visible at delete LSN %d: %+v", deleteLSN, deleted)
	}
	assertEdgeKindAtLSN(t, graphStore, source, removeLSN, 11, true)
	assertEdgeKindAtLSN(t, graphStore, source, deleteLSN, 11, false)
}

func assertVectorAtLSN(t *testing.T, collection *Collection, id string, lsn uint64, want float32) {
	t.Helper()
	record, err := collection.GetAtLSN(context.Background(), id, lsn)
	if err != nil {
		t.Fatalf("GetAtLSN(%d): %v", lsn, err)
	}
	if record == nil || len(record.Vector) != 1 || record.Vector[0] != want {
		t.Fatalf("record %s at LSN %d = %+v, want vector [%v]", id, lsn, record, want)
	}
}

func assertEdgeKindAtLSN(t *testing.T, graphStore Graph, source, lsn uint64, kind uint8, want bool) {
	t.Helper()
	edges, err := graphStore.NeighborsAtLSN(source, lsn)
	if err != nil {
		t.Fatalf("NeighborsAtLSN(%d): %v", lsn, err)
	}
	found := false
	for _, edge := range edges {
		if edge.GetKind() == kind {
			found = true
			break
		}
	}
	if found != want {
		t.Fatalf("edge kind %d at LSN %d found=%v, want %v; edges=%+v", kind, lsn, found, want, edges)
	}
}
