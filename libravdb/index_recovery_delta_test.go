package libravdb

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/xDarkicex/libravdb/internal/storage/singlefile"
)

func TestHNSWRecoveryReplaysPostCheckpointDeltas(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "hnsw-delta-source.libravdb")
	copyPath := filepath.Join(dir, "hnsw-delta-copy.libravdb")
	db, err := Open(WithStoragePath(sourcePath))
	if err != nil {
		t.Fatalf("open source: %v", err)
	}
	collection, err := db.CreateCollection(
		context.Background(),
		"vectors",
		WithDimension(4),
		WithMetric(L2Distance),
		WithHNSW(8, 64, 64),
	)
	if err != nil {
		_ = db.Close()
		t.Fatalf("create collection: %v", err)
	}
	for _, entry := range []struct {
		id     string
		vector []float32
	}{
		{id: "update", vector: []float32{1, 0, 0, 0}},
		{id: "delete", vector: []float32{0, 1, 0, 0}},
		{id: "keep", vector: []float32{0, 0, 1, 0}},
	} {
		if err := collection.Insert(context.Background(), entry.id, entry.vector, nil); err != nil {
			_ = db.Close()
			t.Fatalf("insert checkpoint entry %s: %v", entry.id, err)
		}
	}
	compactor, ok := db.storage.(interface{ Compact() error })
	if !ok {
		_ = db.Close()
		t.Fatal("single-file storage does not expose Compact")
	}
	if err := compactor.Compact(); err != nil {
		_ = db.Close()
		t.Fatalf("compact checkpoint: %v", err)
	}

	updatedVector := []float32{0, 0, 0, 1}
	if err := collection.Update(context.Background(), "update", updatedVector, nil); err != nil {
		_ = db.Close()
		t.Fatalf("update after checkpoint: %v", err)
	}
	if err := collection.Insert(context.Background(), "new", []float32{0.5, 0.5, 0, 0}, nil); err != nil {
		_ = db.Close()
		t.Fatalf("insert after checkpoint: %v", err)
	}
	if err := collection.Delete(context.Background(), "delete"); err != nil {
		_ = db.Close()
		t.Fatalf("delete after checkpoint: %v", err)
	}
	copyFile(t, sourcePath, copyPath)
	if err := db.Close(); err != nil {
		t.Fatalf("close source: %v", err)
	}

	recovered, err := Open(WithStoragePath(copyPath))
	if err != nil {
		t.Fatalf("open live copy: %v", err)
	}
	defer recovered.Close()
	recoveredCollection, err := recovered.GetCollection("vectors")
	if err != nil {
		t.Fatalf("get recovered collection: %v", err)
	}
	if got := recoveredCollection.index.Size(); got != 3 {
		t.Fatalf("recovered index size = %d, want 3", got)
	}
	if _, err := recoveredCollection.Get(context.Background(), "delete"); err == nil {
		t.Fatal("deleted record survived recovery")
	}
	entry, err := recoveredCollection.Get(context.Background(), "update")
	if err != nil {
		t.Fatalf("get updated record: %v", err)
	}
	for i := range updatedVector {
		if entry.Vector[i] != updatedVector[i] {
			t.Fatalf("updated vector = %v, want %v", entry.Vector, updatedVector)
		}
	}
	results, err := recoveredCollection.Search(context.Background(), updatedVector, 1)
	if err != nil {
		t.Fatalf("search recovered index: %v", err)
	}
	if len(results.Results) != 1 || results.Results[0].ID != "update" {
		t.Fatalf("recovered search results = %+v, want update", results)
	}

	engine, ok := recovered.storage.(*singlefile.Engine)
	if !ok {
		t.Fatal("recovered storage is not single-file engine")
	}
	stats := engine.RecoveryStats()
	if stats.RebuiltIndexes != 1 || stats.ReplayedIndexPuts != 2 || stats.ReplayedIndexDeletes != 1 {
		t.Fatalf("recovery stats = %+v, want one base rebuild and 2/1 deltas", stats)
	}
}
