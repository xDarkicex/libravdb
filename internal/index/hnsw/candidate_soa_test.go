package hnsw

import (
	"context"
	"math"
	"strconv"
	"testing"

	"github.com/xDarkicex/libravdb/internal/util"
)

func TestSOACandidateQueueSortedCursorAndCapacity(t *testing.T) {
	ids := make([]uint32, 4)
	distances := make([]float32, 4)
	queue := newSOACandidateQueue(ids, distances, 4)

	for _, candidate := range []util.Candidate{
		{ID: 30, Distance: 3},
		{ID: 10, Distance: 1},
		{ID: 40, Distance: 4},
		{ID: 20, Distance: 2},
	} {
		if !queue.Insert(candidate) {
			t.Fatalf("failed to insert %+v", candidate)
		}
	}

	first, ok := queue.PopClosestUnexpanded()
	if !ok || first.ID != 10 || first.Distance != 1 {
		t.Fatalf("first pop = %+v, %v", first, ok)
	}
	second, ok := queue.PopClosestUnexpanded()
	if !ok || second.ID != 20 || second.Distance != 2 {
		t.Fatalf("second pop = %+v, %v", second, ok)
	}

	// Insertion before the cursor must make the new candidate immediately
	// expandable while retaining visited state on shifted entries.
	if !queue.Insert(util.Candidate{ID: 5, Distance: 0.5}) {
		t.Fatal("failed to insert candidate before cursor")
	}
	third, ok := queue.PopClosestUnexpanded()
	if !ok || third.ID != 5 || third.Distance != 0.5 {
		t.Fatalf("third pop = %+v, %v", third, ok)
	}
	fourth, ok := queue.PopClosestUnexpanded()
	if !ok || fourth.ID != 30 || fourth.Distance != 3 {
		t.Fatalf("fourth pop = %+v, %v", fourth, ok)
	}

	if queue.Insert(util.Candidate{ID: 99, Distance: 99}) {
		t.Fatal("full queue accepted a candidate worse than its boundary")
	}
	got := queue.AppendCandidates(make([]util.Candidate, 0, 4))
	want := []util.Candidate{
		{ID: 5, Distance: 0.5},
		{ID: 10, Distance: 1},
		{ID: 20, Distance: 2},
		{ID: 30, Distance: 3},
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("candidate %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}

func TestSOACandidateModeBuildsConnectedGraph(t *testing.T) {
	index, err := NewHNSW(&Config{
		Dimension: 8, M: 4, EfConstruction: 16, EfSearch: 16,
		ML: 1, Metric: util.L2Distance, RandomSeed: 42,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer index.Close()
	index.candidateMode.Store(uint32(candidateModeSOA))
	ctx := context.Background()
	var query []float32
	for i := 0; i < 32; i++ {
		vector := make([]float32, 8)
		for j := range vector {
			vector[j] = float32((i+1)*(j+3)%17) / 17
		}
		if err := index.Insert(ctx, &VectorEntry{Vector: vector}); err != nil {
			t.Fatal(err)
		}
		if i == 7 {
			query = vector
		}
	}

	totalLinks := 0
	for i := 0; i < index.nodes.Len(); i++ {
		node := index.nodes.Get(uint32(i))
		if node != nil {
			totalLinks += int(node.LinkCounts[0] + node.BacklinkCounts[0])
		}
	}
	if totalLinks < index.nodes.Len()*4 {
		t.Fatalf("SoA construction produced a sparse graph: %d links for %d nodes", totalLinks, index.nodes.Len())
	}
	scratch := index.acquireSearchScratchWithEF(16)
	candidates, err := index.searchLevelScratchValues(ctx, query, index.getEntryPoint(), 16, 0, scratch, nil, nil)
	index.releaseSearchScratch(scratch)
	if err != nil {
		t.Fatal(err)
	}
	if len(candidates) != 16 {
		t.Fatalf("SoA traversal returned %d candidates, want 16", len(candidates))
	}
}

func TestSOACandidateQueueTieBreaksByID(t *testing.T) {
	queue := newSOACandidateQueue(make([]uint32, 3), make([]float32, 3), 3)
	queue.Insert(util.Candidate{ID: 9, Distance: 1})
	queue.Insert(util.Candidate{ID: 3, Distance: 1})
	queue.Insert(util.Candidate{ID: 6, Distance: 1})

	got := queue.AppendCandidates(make([]util.Candidate, 0, 3))
	for i, wantID := range []uint32{3, 6, 9} {
		if got[i].ID != wantID {
			t.Fatalf("candidate %d ID = %d, want %d", i, got[i].ID, wantID)
		}
	}
}

func TestSOACandidateQueueBoundaryAndNaN(t *testing.T) {
	queue := newSOACandidateQueue(make([]uint32, 8), make([]float32, 8), 8)
	for i := 0; i < 8; i++ {
		if !queue.Insert(util.Candidate{ID: uint32(i), Distance: float32(i)}) {
			t.Fatalf("insert %d failed", i)
		}
	}
	if queue.Insert(util.Candidate{ID: 100, Distance: 8}) {
		t.Fatal("queue accepted candidate at the rejected full boundary")
	}
	if !queue.Insert(util.Candidate{ID: 101, Distance: 6.5}) {
		t.Fatal("queue rejected candidate inside the full boundary")
	}
	if queue.Worst().Distance != 6.5 {
		t.Fatalf("worst distance = %v, want 6.5", queue.Worst().Distance)
	}
	if queue.Insert(util.Candidate{ID: 102, Distance: float32(math.NaN())}) {
		t.Fatal("queue accepted NaN distance")
	}
}

func TestSOACandidateQueueCursorStress(t *testing.T) {
	const capacity = 32
	queue := newSOACandidateQueue(make([]uint32, capacity), make([]float32, capacity), capacity)
	rng := NewPCG(918273)
	for i := 0; i < 10_000; i++ {
		queue.Insert(util.Candidate{
			ID:       uint32(i),
			Distance: float32(rng.Uint32()%4096) / 17,
		})
		if i%3 == 0 {
			queue.PopClosestUnexpanded()
		}
		if queue.size > capacity || queue.cursor > queue.size {
			t.Fatalf("iteration %d invalid size/cursor: %d/%d", i, queue.size, queue.cursor)
		}
		for j := 1; j < queue.size; j++ {
			left := util.Candidate{ID: queue.ids[j-1] & soaCandidateIDMask, Distance: queue.distances[j-1]}
			right := util.Candidate{ID: queue.ids[j] & soaCandidateIDMask, Distance: queue.distances[j]}
			if candidateBetter(right, left) {
				t.Fatalf("iteration %d queue is unsorted at %d: %+v then %+v", i, j, left, right)
			}
		}
		for j := 0; j < queue.cursor; j++ {
			if queue.ids[j]&soaCandidateVisited == 0 {
				t.Fatalf("iteration %d unexpanded candidate before cursor at %d", i, j)
			}
		}
	}
}

func TestSOACandidateQueueZeroAllocations(t *testing.T) {
	ids := make([]uint32, 200)
	distances := make([]float32, 200)
	allocs := testing.AllocsPerRun(1000, func() {
		queue := newSOACandidateQueue(ids, distances, 200)
		for i := 0; i < 200; i++ {
			queue.Insert(util.Candidate{ID: uint32(i), Distance: float32(200 - i)})
		}
		for queue.HasUnexpanded() {
			queue.PopClosestUnexpanded()
		}
	})
	if allocs != 0 {
		t.Fatalf("SoA queue allocated %v times per run", allocs)
	}
}

func TestSOALowerBoundHybridMatchesBinary(t *testing.T) {
	for _, size := range []int{1, 15, 16, 17, 63, 96, 100, 200, 512} {
		ids := make([]uint32, size)
		distances := make([]float32, size)
		for i := 0; i < size; i++ {
			ids[i] = uint32(i)
			distances[i] = float32(i / 3) // Include equal-distance ID runs.
		}
		queue := newSOACandidateQueue(ids, distances, size)
		queue.size = size
		rng := NewPCG(uint64(size * 31))
		for i := 0; i < 10_000; i++ {
			candidate := util.Candidate{
				ID:       rng.Uint32() % uint32(max(size, 1)),
				Distance: float32(rng.Uint32() % uint32(max(size/3+2, 2))),
			}
			binary := queue.lowerBoundBinary(candidate)
			hybrid := queue.lowerBoundHybrid(candidate)
			if hybrid != binary {
				t.Fatalf("size=%d candidate=%+v hybrid=%d binary=%d", size, candidate, hybrid, binary)
			}
		}
	}
}

func TestHNSWRejectsOrdinalBeyondRegistryCapacity(t *testing.T) {
	index, err := NewHNSW(&Config{
		Dimension: 4, M: 4, EfConstruction: 8, EfSearch: 8,
		ML: 1, Metric: util.L2Distance, RandomSeed: 42,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer index.Close()
	index.nextOrdinal.Store(maxNodeCapacity)
	err = index.Insert(context.Background(), &VectorEntry{Vector: []float32{1, 2, 3, 4}})
	if err == nil {
		t.Fatal("insert beyond registry capacity succeeded")
	}
}

var lowerBoundBenchmarkSink int

func BenchmarkSOALowerBoundShape(b *testing.B) {
	for _, size := range []int{32, 64, 100, 200, 512} {
		ids := make([]uint32, size)
		distances := make([]float32, size)
		for i := range distances {
			ids[i] = uint32(i)
			distances[i] = float32(i) * 2
		}
		queue := newSOACandidateQueue(ids, distances, size)
		queue.size = size
		targets := make([]util.Candidate, 1024)
		rng := NewPCG(uint64(size))
		for i := range targets {
			position := int(rng.Uint32() % uint32(size))
			targets[i] = util.Candidate{ID: uint32(position), Distance: float32(position*2) + 1}
		}

		b.Run("binary/"+strconv.Itoa(size), func(b *testing.B) {
			var result int
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				result += queue.lowerBoundBinary(targets[i&1023])
			}
			lowerBoundBenchmarkSink = result
		})
		b.Run("hybrid_scalar/"+strconv.Itoa(size), func(b *testing.B) {
			var result int
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				result += queue.lowerBoundHybrid(targets[i&1023])
			}
			lowerBoundBenchmarkSink = result
		})
	}
}
