package hnsw

import (
	"context"
	"math/rand"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/util"
)

const (
	benchDim            = 128
	benchBuildSize      = 5000
	benchSearchQueries  = 100
	benchSearchK        = 10
	benchVectorPoolSize = 8192
)

func benchmarkHNSWConfig() Config {
	return Config{
		Dimension:      benchDim,
		M:              16,
		EfConstruction: 100,
		EfSearch:       50,
		ML:             1.0,
		Metric:         util.L2Distance,
		RandomSeed:     42,
	}
}

func benchmarkVectors(n int, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	vectors := make([][]float32, n)
	for i := range vectors {
		vec := make([]float32, benchDim)
		for j := range vec {
			vec[j] = rng.Float32()
		}
		vectors[i] = vec
	}
	return vectors
}

func benchmarkIDs(n int) []string {
	ids := make([]string, n)
	for i := range ids {
		ids[i] = "vec_" + strconv.FormatUint(uint64(i), 10)
	}
	return ids
}

func buildBenchmarkIndex(b testing.TB, vectors [][]float32, ids []string) *Index {
	b.Helper()

	config := benchmarkHNSWConfig()
	index, err := NewHNSW(&config)
	if err != nil {
		b.Fatalf("failed to create HNSW index: %v", err)
	}

	ctx := context.Background()
	for i, vec := range vectors {
		entry := VectorEntry{Vector: vec}
		if ids != nil {
			entry.ID = ids[i]
		}
		if err := index.Insert(ctx, &entry); err != nil {
			index.Close()
			b.Fatalf("insert %d failed: %v", i, err)
		}
	}

	return index
}

func bruteForceTruth(vectors, queries [][]float32, k int) [][]int {
	type pair struct {
		id   int
		dist float32
	}

	truth := make([][]int, len(queries))
	for qi, q := range queries {
		all := make([]pair, len(vectors))
		for i, v := range vectors {
			all[i] = pair{id: i, dist: util.L2Distance_func(q, v)}
		}
		sort.Slice(all, func(i, j int) bool { return all[i].dist < all[j].dist })

		top := make([]int, k)
		for i := range top {
			top[i] = all[i].id
		}
		truth[qi] = top
	}
	return truth
}

func recallAtK(results []*SearchResult, truth []int, k int) float64 {
	if len(results) == 0 || len(truth) == 0 || k == 0 {
		return 0
	}
	truthSet := make(map[uint32]struct{}, len(truth))
	for _, id := range truth {
		truthSet[uint32(id)] = struct{}{}
	}
	hits := 0
	limit := min(k, len(results))
	for i := 0; i < limit; i++ {
		if _, ok := truthSet[results[i].Ordinal]; ok {
			hits++
		}
	}
	return float64(hits) / float64(k)
}

func recallResultsAtK(results []*SearchResult, truthSet map[uint32]struct{}, k int) float64 {
	if len(results) == 0 || len(truthSet) == 0 || k == 0 {
		return 0
	}
	hits := 0
	limit := min(k, len(results))
	for i := 0; i < limit; i++ {
		if _, ok := truthSet[results[i].Ordinal]; ok {
			hits++
		}
	}
	return float64(hits) / float64(k)
}

func benchmarkTruthSets(truth [][]int) []map[uint32]struct{} {
	sets := make([]map[uint32]struct{}, len(truth))
	for i, ids := range truth {
		set := make(map[uint32]struct{}, len(ids))
		for _, id := range ids {
			set[uint32(id)] = struct{}{}
		}
		sets[i] = set
	}
	return sets
}

func recallOrdinalsAtK(ordinals []uint32, truthSet map[uint32]struct{}, k int) float64 {
	if len(ordinals) == 0 || len(truthSet) == 0 || k == 0 {
		return 0
	}
	hits := 0
	limit := min(k, len(ordinals))
	for i := 0; i < limit; i++ {
		if _, ok := truthSet[ordinals[i]]; ok {
			hits++
		}
	}
	return float64(hits) / float64(k)
}

func percentileDuration(samples []int64, pct float64) time.Duration {
	if len(samples) == 0 {
		return 0
	}
	idx := int(float64(len(samples)-1) * pct)
	if idx < 0 {
		idx = 0
	}
	if idx >= len(samples) {
		idx = len(samples) - 1
	}
	return time.Duration(samples[idx])
}

func BenchmarkHNSWLockFreeThroughput(b *testing.B) {
	config := benchmarkHNSWConfig()

	index, err := NewHNSW(&config)
	if err != nil {
		b.Fatalf("failed to create HNSW index: %v", err)
	}
	defer index.Close()

	// Pre-generate a reusable vector pool to remove RNG overhead without
	// allocating b.N vectors. Each benchmark operation still gets a unique ID.
	vectors := benchmarkVectors(benchVectorPoolSize, 42)

	b.ResetTimer()
	b.ReportAllocs()

	var counter uint32
	var firstErr error
	var errOnce sync.Once
	ctx := context.Background()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			ordinal := atomic.AddUint32(&counter, 1) - 1
			entry := VectorEntry{
				ID:     "vec_" + strconv.FormatUint(uint64(ordinal), 10),
				Vector: vectors[int(ordinal)&(benchVectorPoolSize-1)],
			}
			if err := index.Insert(ctx, &entry); err != nil {
				errOnce.Do(func() { firstErr = err })
				return
			}
		}
	})
	if firstErr != nil {
		b.Fatalf("insert failed: %v", firstErr)
	}
	b.ReportMetric(float64(index.size.Load()), "nodes")
}

func BenchmarkHNSWBuildFixedSize(b *testing.B) {
	vectors := benchmarkVectors(benchBuildSize, 42)
	ids := benchmarkIDs(benchBuildSize)

	for _, tc := range []struct {
		name string
		ids  []string
	}{
		{name: "external_ids"},
		{name: "ordinal_only"},
	} {
		tc := tc
		if tc.name == "external_ids" {
			tc.ids = ids
		}

		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			var totalInserts uint64

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				config := benchmarkHNSWConfig()
				index, err := NewHNSW(&config)
				if err != nil {
					b.Fatalf("failed to create HNSW index: %v", err)
				}
				ctx := context.Background()
				b.StartTimer()

				for j, vec := range vectors {
					entry := VectorEntry{Vector: vec}
					if tc.ids != nil {
						entry.ID = tc.ids[j]
					}
					if err := index.Insert(ctx, &entry); err != nil {
						b.Fatalf("insert %d failed: %v", j, err)
					}
				}
				totalInserts += uint64(len(vectors))

				b.StopTimer()
				index.Close()
			}
			elapsed := b.Elapsed()
			if elapsed > 0 {
				b.ReportMetric(float64(totalInserts)/elapsed.Seconds(), "insert/s")
			}
			b.ReportMetric(float64(benchBuildSize), "nodes/build")
		})
	}
}

func searchExplicitEFOrdinals(ctx context.Context, index *Index, query []float32, k int, ef int, ordinals []uint32) ([]uint32, int, error) {
	ordinals = ordinals[:0]

	var queryState any
	if index.quantizer != nil {
		queryState = index.quantizer.PrepareQuery(query)
	}

	ep := index.getEntryPoint()
	for level := index.getMaxLevel(); level > 0; level-- {
		candidate, err := index.greedySearchLevel(ctx, query, ep, level, queryState)
		if err != nil {
			return ordinals, 0, err
		}
		if candidate != nil {
			ep = index.nodes.Get(candidate.ID)
		}
	}

	scratch := index.acquireSearchScratchWithEF(ef)
	candidates, err := index.searchLevelScratchValues(ctx, query, ep, ef, 0, scratch, queryState, nil)
	candidateCount := len(candidates)
	if err == nil && candidateCount > 0 {
		sort.Slice(candidates, func(i, j int) bool {
			return compareCandidateValues(candidates[i], candidates[j]) < 0
		})
		limit := min(k, candidateCount)
		for _, candidate := range candidates[:limit] {
			node := index.nodes.Get(candidate.ID)
			if node != nil {
				ordinals = append(ordinals, node.Ordinal)
			}
		}
	}
	index.releaseSearchScratch(scratch)
	return ordinals, candidateCount, err
}

func BenchmarkHNSWParallelInsertIDModes(b *testing.B) {
	vectors := benchmarkVectors(benchVectorPoolSize, 42)

	for _, tc := range []struct {
		name    string
		withIDs bool
	}{
		{name: "external_ids", withIDs: true},
		{name: "ordinal_only", withIDs: false},
	} {
		tc := tc
		b.Run(tc.name, func(b *testing.B) {
			config := benchmarkHNSWConfig()
			index, err := NewHNSW(&config)
			if err != nil {
				b.Fatalf("failed to create HNSW index: %v", err)
			}
			defer index.Close()

			b.ReportAllocs()
			b.ResetTimer()

			var counter uint32
			var firstErr error
			var errOnce sync.Once
			ctx := context.Background()
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					ordinal := atomic.AddUint32(&counter, 1) - 1
					entry := VectorEntry{
						Vector: vectors[int(ordinal)&(benchVectorPoolSize-1)],
					}
					if tc.withIDs {
						entry.ID = "vec_" + strconv.FormatUint(uint64(ordinal), 10)
					}
					if err := index.Insert(ctx, &entry); err != nil {
						errOnce.Do(func() { firstErr = err })
						return
					}
				}
			})
			if firstErr != nil {
				b.Fatalf("insert failed: %v", firstErr)
			}
			b.ReportMetric(float64(index.size.Load()), "nodes")
		})
	}
}

func BenchmarkHNSWSearchRecallLatency(b *testing.B) {
	vectors := benchmarkVectors(benchBuildSize, 42)
	queries := benchmarkVectors(benchSearchQueries, 99)
	truth := bruteForceTruth(vectors, queries, benchSearchK)
	truthSets := benchmarkTruthSets(truth)
	index := buildBenchmarkIndex(b, vectors, benchmarkIDs(len(vectors)))
	defer index.Close()

	ctx := context.Background()
	latencies := make([]int64, b.N)
	var totalRecall float64

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		qi := i % len(queries)
		start := time.Now()
		results, err := index.Search(ctx, queries[qi], benchSearchK, nil)
		latencies[i] = time.Since(start).Nanoseconds()
		if err != nil {
			b.Fatalf("search failed: %v", err)
		}
		totalRecall += recallResultsAtK(results, truthSets[qi], benchSearchK)
	}
	b.StopTimer()

	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	if b.N > 0 {
		b.ReportMetric(totalRecall/float64(b.N), "recall@10")
		b.ReportMetric(float64(percentileDuration(latencies, 0.50).Nanoseconds()), "p50-ns")
		b.ReportMetric(float64(percentileDuration(latencies, 0.95).Nanoseconds()), "p95-ns")
		b.ReportMetric(float64(percentileDuration(latencies, 0.99).Nanoseconds()), "p99-ns")
	}
}

func BenchmarkHNSWEfSweepRecallLatency(b *testing.B) {
	vectors := benchmarkVectors(benchBuildSize, 42)
	queries := benchmarkVectors(benchSearchQueries, 99)
	truth := bruteForceTruth(vectors, queries, benchSearchK)
	truthSets := benchmarkTruthSets(truth)
	index := buildBenchmarkIndex(b, vectors, benchmarkIDs(len(vectors)))
	defer index.Close()

	ctx := context.Background()
	for _, ef := range []int{50, 100, 150, 200, 300, 400} {
		ef := ef
		b.Run("ef_"+strconv.Itoa(ef), func(b *testing.B) {
			latencies := make([]int64, b.N)
			ordinalBufs := make([][]uint32, len(queries))
			for i := range ordinalBufs {
				ordinalBufs[i] = make([]uint32, 0, benchSearchK)
			}

			var totalCandidates uint64
			var totalRecall float64

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				qi := i % len(queries)
				start := time.Now()
				ordinals, candidateCount, err := searchExplicitEFOrdinals(ctx, index, queries[qi], benchSearchK, ef, ordinalBufs[qi])
				latencies[i] = time.Since(start).Nanoseconds()
				if err != nil {
					b.Fatalf("explicit ef search failed: %v", err)
				}
				ordinalBufs[qi] = ordinals
				totalCandidates += uint64(candidateCount)
				totalRecall += recallOrdinalsAtK(ordinals, truthSets[qi], benchSearchK)
			}
			b.StopTimer()

			sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
			if b.N > 0 {
				b.ReportMetric(float64(ef), "ef")
				b.ReportMetric(totalRecall/float64(b.N), "recall@10")
				b.ReportMetric(float64(totalCandidates)/float64(b.N), "candidates/op")
				b.ReportMetric(float64(percentileDuration(latencies, 0.50).Nanoseconds()), "p50-ns")
				b.ReportMetric(float64(percentileDuration(latencies, 0.95).Nanoseconds()), "p95-ns")
				b.ReportMetric(float64(percentileDuration(latencies, 0.99).Nanoseconds()), "p99-ns")
			}
		})
	}
}

func BenchmarkHNSWSearchTraversalOnly(b *testing.B) {
	vectors := benchmarkVectors(benchBuildSize, 42)
	queries := benchmarkVectors(benchSearchQueries, 99)
	index := buildBenchmarkIndex(b, vectors, benchmarkIDs(len(vectors)))
	defer index.Close()

	ctx := context.Background()
	ef := max(index.config.EfSearch, benchSearchK, index.config.EfConstruction*3)
	latencies := make([]int64, b.N)
	var totalCandidates uint64

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		start := time.Now()
		query := queries[i%len(queries)]
		var queryState any
		if index.quantizer != nil {
			queryState = index.quantizer.PrepareQuery(query)
		}

		ep := index.getEntryPoint()
		for level := index.getMaxLevel(); level > 0; level-- {
			candidate, err := index.greedySearchLevel(ctx, query, ep, level, queryState)
			if err != nil {
				b.Fatalf("greedy search failed: %v", err)
			}
			if candidate != nil {
				ep = index.nodes.Get(candidate.ID)
			}
		}

		scratch := index.acquireSearchScratchWithEF(ef)
		candidates, err := index.searchLevelScratchValues(ctx, query, ep, ef, 0, scratch, queryState, nil)
		index.releaseSearchScratch(scratch)
		if err != nil {
			b.Fatalf("level search failed: %v", err)
		}
		totalCandidates += uint64(len(candidates))
		latencies[i] = time.Since(start).Nanoseconds()
	}
	b.StopTimer()

	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	if b.N > 0 {
		b.ReportMetric(float64(totalCandidates)/float64(b.N), "candidates/op")
		b.ReportMetric(float64(percentileDuration(latencies, 0.50).Nanoseconds()), "p50-ns")
		b.ReportMetric(float64(percentileDuration(latencies, 0.95).Nanoseconds()), "p95-ns")
		b.ReportMetric(float64(percentileDuration(latencies, 0.99).Nanoseconds()), "p99-ns")
	}
}
