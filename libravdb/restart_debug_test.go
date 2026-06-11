package libravdb

import (
	"context"
	"fmt"
	"math/rand"
	"testing"
	"time"
)

func TestRestartDebugHNSW(t *testing.T) {
	path := t.TempDir() + "/debug.libravdb"
	ctx := context.Background()

	t.Log("step 1: create db with HNSW, 1000 vectors")
	db, err := Open(WithStoragePath(path))
	if err != nil {
		t.Fatal(err)
	}
	col, err := db.CreateCollection(ctx, "test", WithDimension(128), WithHNSW(16, 100, 50), WithMetric(CosineDistance))
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(42))
	batch := make([]VectorEntry, 0, 500)
	for i := 0; i < 1000; i++ {
		vec := make([]float32, 128)
		for j := range vec {
			vec[j] = rng.Float32()
		}
		batch = append(batch, VectorEntry{ID: fmt.Sprintf("v%d", i), Vector: vec})
		if len(batch) == 500 || i == 999 {
			if err := col.InsertBatch(ctx, batch); err != nil {
				t.Fatal(err)
			}
			batch = batch[:0]
		}
	}
	start := time.Now()
	t.Logf("step 2: compact start")
	if err := db.storage.(interface{ Compact() error }).Compact(); err != nil {
		t.Fatal(err)
	}
	t.Logf("step 2: compact done in %v", time.Since(start))
	t.Log("step 3: close")
	db.Close()

	t.Log("step 4: reopen")
	start = time.Now()
	db2, err := Open(WithStoragePath(path))
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("step 4: reopen done in %v", time.Since(start))
	db2.Close()
	fmt.Println("HNSW RESTART OK")
}
