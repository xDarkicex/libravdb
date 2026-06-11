package libravdb

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"
)

// closureSize runs a single persisted+rebuild measurement for a given config.
func closureSize(tb testing.TB, cfg restartBenchConfig) (persisted, rebuild time.Duration) {
	ctx := context.Background()
	path := tb.TempDir() + "/closure.libravdb"

	_ = createAndCompact(tb, ctx, path, cfg)

	t0 := time.Now()
	db2, err := Open(WithStoragePath(path))
	if err != nil {
		tb.Fatal(err)
	}
	persisted = time.Since(t0)
	db2.Close()

	rebuildPath := path + ".rebuild"
	copyFile(tb, path, rebuildPath)
	corruptIndexChunk(tb, rebuildPath)

	t0 = time.Now()
	db3, err := Open(WithStoragePath(rebuildPath))
	if err != nil {
		tb.Fatal(err)
	}
	rebuild = time.Since(t0)
	db3.Close()
	os.Remove(rebuildPath)
	return
}

func TestClosure1K(t *testing.T) {
	p, r := closureSize(t, restartBenchConfig{dim: 128, count: 1000, indexType: HNSW})
	fmt.Printf("CLOSURE|HNSW|1000|persisted=%v|rebuild=%v\n", p, r)
	pf, rf := closureSize(t, restartBenchConfig{dim: 128, count: 1000, indexType: Flat})
	fmt.Printf("CLOSURE|Flat|1000|persisted=%v|rebuild=%v\n", pf, rf)
}

func TestClosure10K(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping 10K in short mode")
	}
	p, r := closureSize(t, restartBenchConfig{dim: 128, count: 10000, indexType: HNSW})
	fmt.Printf("CLOSURE|HNSW|10000|persisted=%v|rebuild=%v\n", p, r)
	pf, rf := closureSize(t, restartBenchConfig{dim: 128, count: 10000, indexType: Flat})
	fmt.Printf("CLOSURE|Flat|10000|persisted=%v|rebuild=%v\n", pf, rf)
}

func TestClosure50K(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping 50K in short mode")
	}
	p, r := closureSize(t, restartBenchConfig{dim: 128, count: 50000, indexType: HNSW})
	fmt.Printf("CLOSURE|HNSW|50000|persisted=%v|rebuild=%v\n", p, r)
	pf, rf := closureSize(t, restartBenchConfig{dim: 128, count: 50000, indexType: Flat})
	fmt.Printf("CLOSURE|Flat|50000|persisted=%v|rebuild=%v\n", pf, rf)
}
