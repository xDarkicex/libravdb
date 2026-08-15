package libravdb

import (
	"context"
	"errors"
	"path/filepath"
	"sync/atomic"
	"testing"
)

func TestCommitReceiptIndividualGroupedAndLatestLSN(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "commit-receipts.libravdb")
	db, err := Open(WithStoragePath(path))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, "docs", WithDimension(1))
	if err != nil {
		t.Fatal(err)
	}

	initialLatest, err := db.LatestCommitLSN(ctx)
	if err != nil || initialLatest == 0 {
		t.Fatalf("latest after collection creation = %d, err=%v", initialLatest, err)
	}

	individual, err := db.BeginTxWithReceipt(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if err := individual.Insert(ctx, "docs", "one", []float32{1}, map[string]interface{}{"n": 1}); err != nil {
		t.Fatal(err)
	}
	receipt1, err := individual.CommitWithReceipt(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if receipt1.CommitLSN <= initialLatest {
		t.Fatalf("individual receipt LSN = %d, want > %d", receipt1.CommitLSN, initialLatest)
	}
	assertSnapshotLSN(t, db, receipt1.CommitLSN, collection, "one", 1)

	receipt2, err := db.WithTxReceipt(ctx, func(tx ReceiptTx) error {
		if err := tx.Insert(ctx, "docs", "two", []float32{2}, map[string]interface{}{"n": 2}); err != nil {
			return err
		}
		return tx.Insert(ctx, "docs", "three", []float32{3}, map[string]interface{}{"n": 3})
	})
	if err != nil {
		t.Fatal(err)
	}
	if receipt2.CommitLSN <= receipt1.CommitLSN {
		t.Fatalf("grouped receipt LSN = %d, want > %d", receipt2.CommitLSN, receipt1.CommitLSN)
	}
	assertSnapshotLSN(t, db, receipt2.CommitLSN, collection, "three", 3)

	latest, err := db.LatestCommitLSN(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if latest != receipt2.CommitLSN {
		t.Fatalf("latest LSN = %d, want grouped receipt %d", latest, receipt2.CommitLSN)
	}

	if err := db.WithTx(ctx, func(tx Tx) error {
		return tx.Update(ctx, "docs", "one", []float32{10}, map[string]interface{}{"n": 10})
	}); err != nil {
		t.Fatal(err)
	}
	legacyLatest, err := db.LatestCommitLSN(ctx)
	if err != nil || legacyLatest <= receipt2.CommitLSN {
		t.Fatalf("legacy Commit latest LSN = %d, err=%v, want > %d", legacyLatest, err, receipt2.CommitLSN)
	}
}

func TestCommitReceiptGraphHookAndSnapshotAtLSNAfterReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "commit-receipts-graph.libravdb")
	db, err := Open(WithStoragePath(path))
	if err != nil {
		t.Fatal(err)
	}

	graphStore, err := NewGraph(GraphConfig{})
	if err != nil {
		db.Close()
		t.Fatal(err)
	}
	collection, err := db.CreateCollection(ctx, "docs", WithDimension(1), WithGraph(graphStore))
	if err != nil {
		graphStore.Close()
		db.Close()
		t.Fatal(err)
	}
	if err := collection.Insert(ctx, "from", []float32{1}, nil); err != nil {
		t.Fatal(err)
	}
	if err := collection.Insert(ctx, "to", []float32{2}, nil); err != nil {
		t.Fatal(err)
	}
	fromNode, err := db.GetNodeID(ctx, "docs", "from")
	if err != nil {
		t.Fatal(err)
	}
	toNode, err := db.GetNodeID(ctx, "docs", "to")
	if err != nil {
		t.Fatal(err)
	}

	edgeTx := graphStore.BeginTxn()
	if err := edgeTx.AddEdge(fromNode, toNode, 1, 7); err != nil {
		t.Fatal(err)
	}
	if err := edgeTx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	edgeLSN, err := db.LatestCommitLSN(ctx)
	if err != nil {
		t.Fatal(err)
	}

	var hookCalled atomic.Bool
	if err := collection.RegisterDeleteHook(func(txn GraphTx, id uint64) error {
		if id != fromNode {
			return errors.New("unexpected delete-hook node")
		}
		hookCalled.Store(true)
		// Keep the hook mutation observable after the source node is dropped.
		return txn.AddEdge(toNode, toNode, 1, 8)
	}); err != nil {
		t.Fatal(err)
	}

	deleteTx, err := db.BeginTxWithReceipt(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if err := deleteTx.Delete(ctx, "docs", "from"); err != nil {
		t.Fatal(err)
	}
	deleteReceipt, err := deleteTx.CommitWithReceipt(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if deleteReceipt.CommitLSN <= edgeLSN {
		t.Fatalf("graph-hook receipt LSN = %d, want > edge LSN %d", deleteReceipt.CommitLSN, edgeLSN)
	}
	if !hookCalled.Load() {
		t.Fatal("delete hook was not invoked")
	}

	snapshot, err := db.SnapshotAtLSN(ctx, deleteReceipt.CommitLSN)
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.LSN != deleteReceipt.CommitLSN {
		t.Fatalf("snapshot LSN = %d, want receipt %d", snapshot.LSN, deleteReceipt.CommitLSN)
	}
	snapshot.Close()
	edges, err := graphStore.NeighborsAtLSN(toNode, deleteReceipt.CommitLSN)
	if err != nil {
		t.Fatal(err)
	}
	var hookEdge bool
	for _, edge := range edges {
		if edge.Target == toNode && edge.GetKind() == 8 {
			hookEdge = true
		}
	}
	if !hookEdge {
		t.Fatalf("graph edges at delete receipt = %+v, want hook edge", edges)
	}
	baseEdges, err := graphStore.NeighborsAtLSN(toNode, edgeLSN)
	if err != nil {
		t.Fatal(err)
	}
	for _, edge := range baseEdges {
		if edge.Target == toNode && edge.GetKind() == 8 {
			t.Fatalf("future graph-hook edge visible at base LSN %d: %+v", edgeLSN, baseEdges)
		}
	}

	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	reopened, err := Open(WithStoragePath(path))
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	latest, err := reopened.LatestCommitLSN(ctx)
	if err != nil || latest != deleteReceipt.CommitLSN {
		t.Fatalf("reopened latest LSN = %d, err=%v, want %d", latest, err, deleteReceipt.CommitLSN)
	}
	reopenedSnapshot, err := reopened.SnapshotAtLSN(ctx, deleteReceipt.CommitLSN)
	if err != nil {
		t.Fatal(err)
	}
	if reopenedSnapshot.LSN != deleteReceipt.CommitLSN {
		t.Fatalf("reopened snapshot LSN = %d, want %d", reopenedSnapshot.LSN, deleteReceipt.CommitLSN)
	}
	reopenedSnapshot.Close()
}

func TestCommitReceiptCancellationClosedAndUnknownLSNErrors(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(filepath.Join(t.TempDir(), "commit-receipts-errors.libravdb")))
	if err != nil {
		t.Fatal(err)
	}
	collection, err := db.CreateCollection(ctx, "docs", WithDimension(1))
	if err != nil {
		t.Fatal(err)
	}

	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := db.BeginTxWithReceipt(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("BeginTxWithReceipt canceled error = %v, want context.Canceled", err)
	}

	tx, err := db.BeginTxWithReceipt(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if err := tx.Insert(ctx, "docs", "x", []float32{1}, nil); err != nil {
		t.Fatal(err)
	}
	if err := tx.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := tx.CommitWithReceipt(ctx); !errors.Is(err, ErrTxClosed) {
		t.Fatalf("CommitWithReceipt after rollback = %v, want ErrTxClosed", err)
	}

	latest, err := db.LatestCommitLSN(ctx)
	if err != nil || latest == 0 {
		t.Fatalf("latest after collection creation = %d, err=%v", latest, err)
	}
	if _, err := db.SnapshotAtLSN(ctx, latest+1_000_000); err == nil {
		t.Fatal("SnapshotAtLSN unknown LSN succeeded")
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := db.LatestCommitLSN(ctx); !errors.Is(err, ErrDatabaseClosed) {
		t.Fatalf("LatestCommitLSN after close = %v, want ErrDatabaseClosed", err)
	}
	_ = collection
}

func assertSnapshotLSN(t *testing.T, db *Database, lsn uint64, collection *Collection, id string, want float32) {
	t.Helper()
	snapshot, err := db.SnapshotAtLSN(context.Background(), lsn)
	if err != nil {
		t.Fatalf("SnapshotAtLSN(%d): %v", lsn, err)
	}
	defer snapshot.Close()
	if snapshot.LSN != lsn {
		t.Fatalf("snapshot LSN = %d, want %d", snapshot.LSN, lsn)
	}
	record, err := collection.GetAtLSN(context.Background(), id, lsn)
	if err != nil {
		t.Fatalf("GetAtLSN(%d): %v", lsn, err)
	}
	if record == nil || len(record.Vector) != 1 || record.Vector[0] != want {
		t.Fatalf("record at LSN %d = %+v, want vector [%v]", lsn, record, want)
	}
}
