package singlefile

import (
	"context"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/index"
	"github.com/xDarkicex/libravdb/internal/storage"
)

func TestCommitCatalogCheckpointRecoveryKeepsLatestCommit(t *testing.T) {
	for _, compact := range []bool{false, true} {
		name := "checkpoint"
		if compact {
			name = "compaction"
		}
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "catalog.db")
			e := coordinationEngine(t, path)
			if err := e.createCollection("rows", storage.CollectionConfig{Dimension: 1}); err != nil {
				t.Fatal(err)
			}
			col, _ := e.GetCollection("rows")
			ctx := context.Background()
			entry := &index.VectorEntry{ID: "task", Vector: []float32{1}, Metadata: map[string]interface{}{"status": "created"}}
			if err := col.Insert(ctx, entry); err != nil {
				t.Fatal(err)
			}
			// Compaction retains this older graph transaction for topology recovery,
			// while the later record update is represented only by the snapshot.
			if err := e.AppendGraphLabel(ctx, entry.GraphNodeID, "task", nil); err != nil {
				t.Fatal(err)
			}
			receipt, err := e.CommitTxDurable(ctx, []storage.TxOperation{{Type: storage.TxOperationPut, Collection: "rows", ID: "task", Ordinal: entry.Ordinal, Vector: []float32{2}, Metadata: map[string]interface{}{"status": "completed"}}})
			if err != nil {
				t.Fatal(err)
			}
			want := receipt.CommitLSN
			wantCatalog := append([]commitEntry(nil), e.commitCatalog...)
			e.mu.Lock()
			err = e.checkpointLocked()
			e.mu.Unlock()
			if err != nil {
				t.Fatal(err)
			}
			if compact {
				if err := e.Compact(); err != nil {
					t.Fatal(err)
				}
			}
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
			recovered := coordinationEngine(t, path)
			got, err := recovered.LatestCommitLSN()
			if err != nil {
				t.Fatal(err)
			}
			if got != want {
				t.Errorf("recovered commit catalog ends at %d, want last committed LSN %d (state lastLSN=%d)", got, want, recovered.lastLSN.Load())
			}
			record, err := recovered.GetRecordAtLSN("rows", "task", got)
			if err != nil {
				t.Fatal(err)
			}
			if record == nil || record.Metadata["status"] != "completed" {
				t.Errorf("pinning recovered catalog LSN %d returned stale task: %+v", got, record)
			}
			if !reflect.DeepEqual(recovered.commitCatalog, wantCatalog) {
				t.Errorf("recovered catalog=%v, want %v", recovered.commitCatalog, wantCatalog)
			}
			resolved, _, err := recovered.ResolveTimestamp(time.Unix(0, wantCatalog[len(wantCatalog)-1].Timestamp))
			if err != nil || resolved != want {
				t.Errorf("latest timestamp resolved to %d, want %d: %v", resolved, want, err)
			}
			for i := 1; i < len(recovered.commitCatalog); i++ {
				if recovered.commitCatalog[i-1].LSN >= recovered.commitCatalog[i].LSN {
					t.Fatalf("commit catalog is duplicated or out of order: %v", recovered.commitCatalog)
				}
			}
		})
	}
}

func TestRemoteRefreshPublishesCommitCatalog(t *testing.T) {
	path := filepath.Join(t.TempDir(), "refresh-catalog.db")
	a := coordinationEngine(t, path)
	if err := a.createCollection("rows", storage.CollectionConfig{Dimension: 1}); err != nil {
		t.Fatal(err)
	}
	b := coordinationEngine(t, path)
	col, _ := b.GetCollection("rows")
	if err := col.Insert(context.Background(), &index.VectorEntry{ID: "remote", Vector: []float32{1}}); err != nil {
		t.Fatal(err)
	}
	want, err := b.LatestCommitLSN()
	if err != nil {
		t.Fatal(err)
	}
	got, err := a.LatestCommitLSN()
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("refreshed catalog LSN=%d, want %d (state lastLSN=%d)", got, want, a.lastLSN.Load())
	}
}

func TestCommitCatalogRecoveryRepairsPreviouslyUnorderedSnapshot(t *testing.T) {
	path := filepath.Join(t.TempDir(), "legacy-catalog.db")
	e := coordinationEngine(t, path)
	if err := e.createCollection("rows", storage.CollectionConfig{Dimension: 1}); err != nil {
		t.Fatal(err)
	}
	col, _ := e.GetCollection("rows")
	for _, id := range []string{"one", "two"} {
		if err := col.Insert(context.Background(), &index.VectorEntry{ID: id, Vector: []float32{1}}); err != nil {
			t.Fatal(err)
		}
	}
	want := append([]commitEntry(nil), e.commitCatalog...)
	// Simulate a checkpoint written after older WAL entries had been appended
	// to an already-complete snapshot catalog by the previous recovery code.
	e.mu.Lock()
	e.commitCatalog = append([]commitEntry{want[len(want)-1]}, want...)
	err := e.checkpointLocked()
	e.mu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Compact(); err != nil {
		t.Fatal(err)
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	for range 2 {
		recovered := coordinationEngine(t, path)
		if !reflect.DeepEqual(recovered.commitCatalog, want) {
			t.Fatalf("catalog was not repaired: got %v, want %v", recovered.commitCatalog, want)
		}
		if err := recovered.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestCommitCatalogReplayDoesNotResurrectPrunedEntries(t *testing.T) {
	e := &Engine{oldestRetainedLSN: 10, commitCatalog: []commitEntry{{LSN: 3, Timestamp: 1}, {LSN: 12, Timestamp: 4}}}
	e.recordCommitLocked(6, 2)
	e.recordCommitLocked(9, 3)
	e.recordCommitLocked(12, 4)
	e.recordCommitLocked(15, 5)
	want := []commitEntry{{LSN: 3, Timestamp: 1}, {LSN: 12, Timestamp: 4}, {LSN: 15, Timestamp: 5}}
	if !reflect.DeepEqual(e.commitCatalog, want) {
		t.Fatalf("replay changed retention boundary or duplicated commits: %v", e.commitCatalog)
	}
}
