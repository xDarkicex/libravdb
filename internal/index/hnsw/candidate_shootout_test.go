package hnsw

import (
	"context"
	"fmt"
	"sort"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/util"
)

// TestCandidateStructureShootout compares heap vs unsorted on both
// insertion throughput and search recall, side by side.
func TestCandidateStructureShootout(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}

	const (
		dim     = 128
		numVecs = 5000
		k       = 10
		M       = 16
		efCons  = 100
	)

	// Generate deterministic vectors.
	rng := NewPCG(42)
	vectors := make([][]float32, numVecs)
	for i := range numVecs {
		v := make([]float32, dim)
		for j := range dim {
			v[j] = float32(rng.Float64()*2 - 1)
		}
		vectors[i] = v
	}

	// Generate query vectors from a different seed.
	rng2 := NewPCG(99)
	queries := make([][]float32, 20)
	for i := range 20 {
		q := make([]float32, dim)
		for j := range dim {
			q[j] = float32(rng2.Float64()*2 - 1)
		}
		queries[i] = q
	}

	// Brute-force ground truth for recall measurement.
	type pair struct {
		id   int
		dist float32
	}
	groundTruth := make([][]int, len(queries))
	for qi, q := range queries {
		all := make([]pair, numVecs)
		for i, v := range vectors {
			all[i] = pair{id: i, dist: util.L2Distance_func(q, v)}
		}
		sort.Slice(all, func(i, j int) bool { return all[i].dist < all[j].dist })
		top := make([]int, k)
		for i := range k {
			top[i] = all[i].id
		}
		groundTruth[qi] = top
	}

	modes := []struct {
		name string
		mode candidateMode
	}{
		{name: "heap", mode: candidateModeHeap},
		{name: "unsorted", mode: candidateModeUnsorted},
		{name: "reservoir", mode: candidateModeReservoir},
		{name: "soa", mode: candidateModeSOA},
	}

	for _, tc := range modes {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &Config{
				Dimension:      dim,
				M:              M,
				EfConstruction: efCons,
				EfSearch:       50,
				ML:             1.0,
				Metric:         util.L2Distance,
				RandomSeed:     42,
			}
			idx, err := NewHNSW(cfg)
			if err != nil {
				t.Fatalf("NewHNSW: %v", err)
			}
			defer idx.Close()
			idx.candidateMode.Store(uint32(tc.mode))

			ctx := context.Background()

			// Measure insertion throughput.
			start := time.Now()
			for i, vec := range vectors {
				entry := &VectorEntry{
					ID:     fmt.Sprintf("v_%06d", i),
					Vector: vec,
				}
				if err := idx.Insert(ctx, entry); err != nil {
					t.Fatalf("Insert %d: %v", i, err)
				}
			}
			insertDur := time.Since(start)
			opsPerSec := float64(numVecs) / insertDur.Seconds()
			totalLinks := 0
			for i := 0; i < idx.nodes.Len(); i++ {
				if node := idx.nodes.Get(uint32(i)); node != nil {
					totalLinks += int(node.LinkCounts[0] + node.BacklinkCounts[0])
				}
			}

			// Measure search recall.
			var totalRecall float64
			minRecall := 1.0
			for qi, q := range queries {
				results, err := idx.Search(ctx, q, k, nil)
				if err != nil {
					t.Fatalf("Search: %v", err)
				}

				truthSet := make(map[int]bool, k)
				for _, id := range groundTruth[qi] {
					truthSet[id] = true
				}

				hits := 0
				for _, r := range results {
					if truthSet[int(r.Ordinal)] {
						hits++
					}
				}
				recall := float64(hits) / float64(k)
				totalRecall += recall
				if recall < minRecall {
					minRecall = recall
				}
			}
			avgRecall := totalRecall / float64(len(queries))

			t.Logf("throughput=%.0f ops/s | avg_recall=%.4f | min_recall=%.4f | level0_links=%d",
				opsPerSec, avgRecall, minRecall, totalLinks)
		})
	}
}

func TestReservoirTopKMatchesSortedExact(t *testing.T) {
	const (
		k     = 32
		count = 257
	)

	rng := NewPCG(123)
	candidates := make([]util.Candidate, count)
	for i := range candidates {
		// Force some ties so ID tie-breaking is covered too.
		distance := float32(rng.Uint32()%4096) / 17
		candidates[i] = util.Candidate{ID: uint32(count - i), Distance: distance}
	}

	expected := append([]util.Candidate(nil), candidates...)
	sort.Slice(expected, func(i, j int) bool {
		return compareCandidateValues(expected[i], expected[j]) < 0
	})
	expected = expected[:k]

	buf := make([]util.Candidate, 0, k*2)
	reservoir := reservoirTopK{items: buf, maxSize: k}
	working := candidateMinHeap{items: make([]util.Candidate, 0, count)}
	for _, candidate := range candidates {
		admitCandidateReservoir(&reservoir, &working, candidate.ID, candidate.Distance)
	}
	got := append([]util.Candidate(nil), reservoir.Items()...)
	sort.Slice(got, func(i, j int) bool {
		return compareCandidateValues(got[i], got[j]) < 0
	})

	if len(got) != len(expected) {
		t.Fatalf("got %d candidates, expected %d", len(got), len(expected))
	}
	for i := range expected {
		if got[i] != expected[i] {
			t.Fatalf("candidate %d mismatch: got %+v expected %+v", i, got[i], expected[i])
		}
	}
}

// PCG is a minimal permuted-congruential generator for deterministic random data.
type PCG struct {
	state uint64
}

func NewPCG(seed uint64) *PCG {
	p := &PCG{state: seed + 1442695040888963407}
	p.Uint32()
	return p
}

func (p *PCG) Uint32() uint32 {
	old := p.state
	p.state = old*6364136223846793005 + 1442695040888963407
	xorshifted := uint32(((old >> 18) ^ old) >> 27)
	rot := uint32(old >> 59)
	return (xorshifted >> rot) | (xorshifted << ((-rot) & 31))
}

func (p *PCG) Float64() float64 {
	return float64(p.Uint32()) / float64(1<<32)
}
