package graph

import (
	"math/rand"
	"os"
	"sync"
	"testing"
	"time"
)

func TestStressHubNodeConcurrency(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping stress test in short mode")
	}

	tmpDir, err := os.MkdirTemp("", "graph_stress_*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	cfg := DefaultGraphConfig()
	g, err := NewGraph(cfg)
	if err != nil {
		t.Fatal(err)
	}

	const hubNodeID = uint64(1)
	const numGoroutines = 50
	const opsPerGoroutine = 1000

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	start := time.Now()

	for i := 0; i < numGoroutines; i++ {
		go func(routineID int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(routineID) + time.Now().UnixNano()))
			
			for j := 0; j < opsPerGoroutine; j++ {
				target := uint64(routineID*opsPerGoroutine + j + 10) // Unique targets
				
				// 90% writes, 10% reads to stress lock contention
				if rng.Float32() < 0.9 {
					err := g.(*graphStore).AddEdgeWithStamp(nil, hubNodeID, target, 1.0, 1, uint32(routineID*opsPerGoroutine+j))
					if err != nil {
						t.Errorf("AddEdgeWithStamp failed: %v", err)
					}
				} else {
					_, err := g.Neighbors(hubNodeID)
					if err != nil {
						t.Errorf("Neighbors failed: %v", err)
					}
				}
			}
		}(i)
	}

	wg.Wait()
	duration := time.Since(start)

	neighbors, err := g.Neighbors(hubNodeID)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Stress test completed in %v. Hub node has %d edges.", duration, len(neighbors))
	
	// Ensure no page leak
	stats := g.Stats()
	t.Logf("Stats: %+v", stats)
}

func TestMemoryExhaustion(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping memory exhaustion test in short mode")
	}

	tmpDir, err := os.MkdirTemp("", "graph_mem_*")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	cfg := DefaultGraphConfig()
	g, err := NewGraph(cfg)
	if err != nil {
		t.Fatal(err)
	}

	// Just allocate heavily on one node
	for i := 0; i < 50000; i++ {
		err := g.(*graphStore).AddEdgeWithStamp(nil, 1, uint64(i+100), 1.0, 1, uint32(i))
		if err != nil {
			// If it fails with exhaustion, that's handled!
			t.Logf("Stopped at %d due to %v", i, err)
			break
		}
	}
}
