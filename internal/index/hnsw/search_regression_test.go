package hnsw

import (
	"context"
	"fmt"
	"sort"
	"testing"

	"github.com/xDarkicex/libravdb/internal/util"
)

// TestSearchScratchZeroAllocs verifies that search scratch buffers
// are allocated from the off-heap arena, not from the Go heap.
func TestSearchScratchZeroAllocs(t *testing.T) {
	dim := 16
	cfg := &Config{
		Dimension:      dim,
		M:              8,
		EfConstruction: 32,
		EfSearch:       16,
		ML:             1.0,
		Metric:         util.L2Distance,
		RandomSeed:     42,
	}
	idx, err := NewHNSW(cfg)
	if err != nil {
		t.Fatalf("NewHNSW: %v", err)
	}
	defer idx.Close()

	ctx := context.Background()
	for i := range 50 {
		vec := make([]float32, dim)
		vec[0] = float32(i)
		entry := &VectorEntry{ID: fmt.Sprintf("vec_%d", i), Vector: vec}
		if err := idx.Insert(ctx, entry); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}

	// acqureSearchScratch + releaseSearchScratch should not allocate
	// from the Go heap: visitedMarks, maxHeapBuf, and minHeapBuf are
	// backed by the per-scratch memory.Arena.
	allocs := testing.AllocsPerRun(100, func() {
		scratch := idx.acquireSearchScratch()
		idx.releaseSearchScratch(scratch)
	})
	if allocs != 0 {
		t.Errorf("acquire/release scratch: want 0 allocs, got %.0f", allocs)
	}
}

// TestSearchScratchResultsCorrect verifies that search results are correct
// (ascending distance order) when using the arena-backed scratch buffers.
func TestSearchScratchResultsCorrect(t *testing.T) {
	dim := 16
	cfg := &Config{
		Dimension:      dim,
		M:              8,
		EfConstruction: 32,
		EfSearch:       16,
		ML:             1.0,
		Metric:         util.L2Distance,
		RandomSeed:     42,
	}
	idx, err := NewHNSW(cfg)
	if err != nil {
		t.Fatalf("NewHNSW: %v", err)
	}
	defer idx.Close()

	ctx := context.Background()
	for i := range 100 {
		vec := make([]float32, dim)
		vec[0] = float32(i)
		entry := &VectorEntry{ID: fmt.Sprintf("vec_%d", i), Vector: vec}
		if err := idx.Insert(ctx, entry); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}

	query := make([]float32, dim)
	query[0] = 50.0
	k := 10

	results, err := idx.Search(ctx, query, k)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}

	if len(results) != k {
		t.Fatalf("got %d results, want %d", len(results), k)
	}

	// Verify results are in ascending score order (lower = closer).
	for i := 1; i < len(results); i++ {
		if results[i-1].Score > results[i].Score {
			t.Errorf("results not sorted: [%d].Score=%.4f > [%d].Score=%.4f",
				i-1, results[i-1].Score, i, results[i].Score)
		}
	}

	// Verify results match brute-force ordering. Equal-distance results
	// are order-independent — HNSW and brute force may sort ties differently.
	type pair struct {
		id   string
		dist float32
	}
	var brute []pair
	for i := range 100 {
		vec := make([]float32, dim)
		vec[0] = float32(i)
		d := util.L2Distance_func(query, vec)
		brute = append(brute, pair{id: fmt.Sprintf("vec_%d", i), dist: d})
	}
	sort.Slice(brute, func(i, j int) bool { return brute[i].dist < brute[j].dist })

	// Build a set of expected IDs for each unique distance band.
	byDist := make(map[float32]map[string]bool)
	for _, p := range brute[:k] {
		if byDist[p.dist] == nil {
			byDist[p.dist] = make(map[string]bool)
		}
		byDist[p.dist][p.id] = true
	}
	for _, r := range results {
		ids, ok := byDist[r.Score]
		if !ok {
			t.Errorf("result ID=%s score=%.4f not in top-%d (no entry at this distance)",
				r.ID, r.Score, k)
			continue
		}
		if !ids[r.ID] {
			t.Errorf("result ID=%s score=%.4f not found among expected IDs at this distance", r.ID, r.Score)
		}
		delete(ids, r.ID)
	}
}
