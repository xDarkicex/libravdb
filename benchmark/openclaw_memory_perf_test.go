package benchmark

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"testing"
	"time"

	internalhnsw "github.com/xDarkicex/libravdb/internal/index/hnsw"
	"github.com/xDarkicex/libravdb/internal/util"
	"github.com/xDarkicex/libravdb/libravdb"
)

const (
	openClawDimension = 384
	openClawTopK      = 12
)

type openClawWorkload struct {
	name      string
	indexType string
	rawStore  string
	vectors   int
	queries   int
}

var openClawWorkloads = []openClawWorkload{
	{name: "session_hnsw_1k_memory", indexType: "hnsw", rawStore: "memory", vectors: 1000, queries: 200},
	{name: "session_hnsw_1k_slabby", indexType: "hnsw", rawStore: "slabby", vectors: 1000, queries: 200},
	{name: "durable_hnsw_2500_memory", indexType: "hnsw", rawStore: "memory", vectors: 2500, queries: 250},
	{name: "durable_hnsw_2500_slabby", indexType: "hnsw", rawStore: "slabby", vectors: 2500, queries: 250},
	{name: "durable_flat_10k", indexType: "flat", vectors: 10000, queries: 100},
}

func TestOpenClawMemoryProfile(t *testing.T) {
	if os.Getenv("LIBRAVDB_RUN_OPENCLAW_PERF") != "1" {
		t.Skip("set LIBRAVDB_RUN_OPENCLAW_PERF=1 to run the OpenClaw performance profile")
	}

	for _, workload := range openClawWorkloads {
		t.Run(workload.name, func(t *testing.T) {
			ctx := context.Background()
			db, collection, entries, queries := createOpenClawCollection(t, workload)
			defer db.Close()

			start := time.Now()
			maxConcurrency := openClawBatchConcurrency()
			batch := collection.NewBatchInsert(entries, &libravdb.BatchOptions{
				ChunkSize:      min(1000, len(entries)),
				MaxConcurrency: maxConcurrency,
			})
			result, err := batch.Execute(ctx)
			insertDuration := time.Since(start)
			if err != nil {
				t.Fatalf("batch insert failed: %v", err)
			}
			if result.Failed != 0 {
				t.Fatalf("expected 0 failed inserts, got %d", result.Failed)
			}

			insertThroughput := float64(len(entries)) / insertDuration.Seconds()

			latencies := make([]time.Duration, 0, len(queries))
			searchStart := time.Now()
			for _, query := range queries {
				s0 := time.Now()
				searchResults, searchErr := collection.Search(ctx, query, openClawTopK)
				latencies = append(latencies, time.Since(s0))
				if searchErr != nil {
					t.Fatalf("search failed: %v", searchErr)
				}
				if len(searchResults.Results) == 0 {
					t.Fatal("expected non-empty search results")
				}
			}
			searchDuration := time.Since(searchStart)
			searchQPS := float64(len(queries)) / searchDuration.Seconds()

			memUsage, memErr := collection.GetMemoryUsage()
			if memErr != nil {
				t.Fatalf("failed to get memory usage: %v", memErr)
			}
			rawProfile := collection.DebugRawVectorStoreProfile()
			if workload.indexType == "hnsw" && rawProfile != nil {
				t.Fatalf("expected provider-backed HNSW collection to have no raw vector store profile, got %v", rawProfile)
			}

			t.Logf(
				"PROFILE workload=%s index=%s vectors=%d queries=%d insert_secs=%.3f insert_ops_sec=%.1f search_qps=%.1f search_avg_ms=%.3f search_p50_ms=%.3f search_p95_ms=%.3f search_max_ms=%.3f memory_mb=%.2f",
				workload.name,
				workload.indexType,
				workload.vectors,
				len(queries),
				insertDuration.Seconds(),
				insertThroughput,
				searchQPS,
				durationAvg(latencies).Seconds()*1000,
				percentileDuration(latencies, 0.50).Seconds()*1000,
				percentileDuration(latencies, 0.95).Seconds()*1000,
				percentileDuration(latencies, 1.0).Seconds()*1000,
				float64(memUsage.Total)/(1024*1024),
			)
			t.Logf("BATCH workload=%s max_concurrency=%d", workload.name, maxConcurrency)
			if rawProfile != nil {
				t.Logf(
					"RAW_STORE workload=%s backend=%v vector_count=%v dimension=%v bytes_per_vector=%v memory_usage=%v reserved_bytes=%v reserved_data_bytes=%v reserved_meta_bytes=%v reserved_guard_bytes=%v live_bytes=%v free_bytes=%v capacity_utilization=%.4f peak_note=end_equals_peak_for_monotonic_insert_workload",
					workload.name,
					rawProfile["backend"],
					rawProfile["vector_count"],
					rawProfile["dimension"],
					rawProfile["bytes_per_vector"],
					rawProfile["memory_usage"],
					rawProfile["reserved_bytes"],
					rawProfile["reserved_data_bytes"],
					rawProfile["reserved_meta_bytes"],
					rawProfile["reserved_guard_bytes"],
					rawProfile["live_bytes"],
					rawProfile["free_bytes"],
					rawProfile["capacity_utilization"].(float64),
				)
			}
		})
	}
}

func TestHNSW2500ProviderSearchRegression(t *testing.T) {
	if os.Getenv("LIBRAVDB_RUN_PERF_ACCEPTANCE") != "1" {
		t.Skip("set LIBRAVDB_RUN_PERF_ACCEPTANCE=1 to run HNSW search regression acceptance")
	}

	ctx := context.Background()
	workload := openClawWorkload{name: "durable_hnsw_2500_memory", indexType: "hnsw", rawStore: "memory", vectors: 2500, queries: 250}
	entries := generateOpenClawEntries(workload.vectors, openClawDimension, 42)
	queries := generateOpenClawQueries(entries, workload.queries)

	db, collection := newOpenClawDBAndCollection(t, workload)
	defer db.Close()
	batch := collection.NewBatchInsert(entries, &libravdb.BatchOptions{
		ChunkSize:      min(1000, len(entries)),
		MaxConcurrency: openClawBatchConcurrency(),
	})
	if _, err := batch.Execute(ctx); err != nil {
		t.Fatalf("provider-backed batch insert failed: %v", err)
	}

	standalone, err := internalhnsw.NewHNSW(&internalhnsw.Config{
		Dimension:      openClawDimension,
		M:              16,
		EfConstruction: 200,
		EfSearch:       64,
		ML:             1.0 / math.Log(2.0),
		Metric:         util.CosineDistance,
		RawVectorStore: internalhnsw.RawVectorStoreMemory,
	})
	if err != nil {
		t.Fatalf("new standalone hnsw: %v", err)
	}
	defer standalone.Close()
	for _, entry := range entries {
		if err := standalone.Insert(ctx, &internalhnsw.VectorEntry{ID: entry.ID, Vector: entry.Vector, Metadata: entry.Metadata}); err != nil {
			t.Fatalf("standalone insert failed: %v", err)
		}
	}

	measureCollection := func() time.Duration {
		start := time.Now()
		for _, query := range queries {
			results, err := collection.Search(ctx, query, openClawTopK)
			if err != nil {
				t.Fatalf("provider-backed search failed: %v", err)
			}
			if len(results.Results) == 0 {
				t.Fatal("expected provider-backed search results")
			}
		}
		return time.Since(start)
	}
	measureStandalone := func() time.Duration {
		start := time.Now()
		for _, query := range queries {
			results, err := standalone.Search(ctx, query, openClawTopK)
			if err != nil {
				t.Fatalf("standalone search failed: %v", err)
			}
			if len(results) == 0 {
				t.Fatal("expected standalone search results")
			}
		}
		return time.Since(start)
	}

	providerDuration := measureCollection()
	standaloneDuration := measureStandalone()
	ratio := float64(providerDuration) / float64(standaloneDuration)
	t.Logf("provider-backed duration=%s standalone duration=%s ratio=%.3f", providerDuration, standaloneDuration, ratio)
	if ratio > 3.0 {
		t.Fatalf("provider-backed HNSW search regression too large: ratio=%.3f", ratio)
	}
}

func BenchmarkOpenClawMemoryBatchInsert(b *testing.B) {
	workloads := []openClawWorkload{
		{name: "hnsw_1k_memory", indexType: "hnsw", rawStore: "memory", vectors: 1000},
		{name: "hnsw_1k_slabby", indexType: "hnsw", rawStore: "slabby", vectors: 1000},
		{name: "hnsw_2500_memory", indexType: "hnsw", rawStore: "memory", vectors: 2500},
		{name: "hnsw_2500_slabby", indexType: "hnsw", rawStore: "slabby", vectors: 2500},
	}

	for _, workload := range workloads {
		b.Run(workload.name, func(b *testing.B) {
			entries := generateOpenClawEntries(workload.vectors, openClawDimension, 42)
			b.ReportAllocs()
			b.SetBytes(int64(workload.vectors * openClawDimension * 4))

			for i := 0; i < b.N; i++ {
				ctx := context.Background()
				db, collection := newOpenClawDBAndCollection(b, workload)
				maxConcurrency := openClawBatchConcurrency()
				batch := collection.NewBatchInsert(entries, &libravdb.BatchOptions{
					ChunkSize:      min(1000, len(entries)),
					MaxConcurrency: maxConcurrency,
				})

				b.StartTimer()
				result, err := batch.Execute(ctx)
				b.StopTimer()

				if err != nil {
					db.Close()
					b.Fatalf("batch insert failed: %v", err)
				}
				if result.Failed != 0 {
					db.Close()
					b.Fatalf("expected 0 failed inserts, got %d", result.Failed)
				}

				db.Close()
			}
		})
	}
}

func BenchmarkOpenClawMemorySearch(b *testing.B) {
	workloads := []openClawWorkload{
		{name: "hnsw_1k_memory", indexType: "hnsw", rawStore: "memory", vectors: 1000, queries: 200},
		{name: "hnsw_1k_slabby", indexType: "hnsw", rawStore: "slabby", vectors: 1000, queries: 200},
		{name: "hnsw_2500_memory", indexType: "hnsw", rawStore: "memory", vectors: 2500, queries: 250},
		{name: "hnsw_2500_slabby", indexType: "hnsw", rawStore: "slabby", vectors: 2500, queries: 250},
		{name: "flat_10k", indexType: "flat", vectors: 10000, queries: 100},
	}

	for _, workload := range workloads {
		b.Run(workload.name, func(b *testing.B) {
			ctx := context.Background()
			db, collection, _, queries := createOpenClawCollection(b, workload)
			defer db.Close()

			maxConcurrency := openClawBatchConcurrency()
			batch := collection.NewBatchInsert(generateOpenClawEntries(workload.vectors, openClawDimension, 42), &libravdb.BatchOptions{
				ChunkSize:      min(1000, workload.vectors),
				MaxConcurrency: maxConcurrency,
			})
			if _, err := batch.Execute(ctx); err != nil {
				b.Fatalf("batch insert failed: %v", err)
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				query := queries[i%len(queries)]
				results, err := collection.Search(ctx, query, openClawTopK)
				if err != nil {
					b.Fatalf("search failed: %v", err)
				}
				if len(results.Results) == 0 {
					b.Fatal("expected non-empty search results")
				}
			}
		})
	}
}

func openClawBatchConcurrency() int {
	if raw := os.Getenv("LIBRAVDB_OPENCLAW_MAX_CONCURRENCY"); raw != "" {
		if value, err := strconv.Atoi(raw); err == nil && value > 0 {
			return value
		}
	}

	procs := runtime.GOMAXPROCS(0)
	if procs < 1 {
		return 1
	}
	if procs > 8 {
		return 8
	}
	return procs
}

func TestOpenClawBatchConcurrency(t *testing.T) {
	t.Setenv("LIBRAVDB_OPENCLAW_MAX_CONCURRENCY", "3")
	if got := openClawBatchConcurrency(); got != 3 {
		t.Fatalf("expected env override concurrency 3, got %d", got)
	}

	t.Setenv("LIBRAVDB_OPENCLAW_MAX_CONCURRENCY", "invalid")
	if got := openClawBatchConcurrency(); got < 1 {
		t.Fatalf("expected fallback concurrency >= 1, got %d", got)
	}
}

func createOpenClawCollection(tb testing.TB, workload openClawWorkload) (*libravdb.Database, *libravdb.Collection, []*libravdb.VectorEntry, [][]float32) {
	tb.Helper()

	db, collection := newOpenClawDBAndCollection(tb, workload)
	entries := generateOpenClawEntries(workload.vectors, openClawDimension, 42)
	queries := generateOpenClawQueries(entries, workload.queries)
	return db, collection, entries, queries
}

func newOpenClawDBAndCollection(tb testing.TB, workload openClawWorkload) (*libravdb.Database, *libravdb.Collection) {
	tb.Helper()

	db, err := libravdb.New(libravdb.WithStoragePath(filepath.Join(tb.TempDir(), "openclaw.libravdb")), libravdb.WithMetrics(false))
	if err != nil {
		tb.Fatalf("failed to create database: %v", err)
	}

	opts := []libravdb.CollectionOption{
		libravdb.WithDimension(openClawDimension),
		libravdb.WithMetric(libravdb.CosineDistance),
	}
	switch workload.indexType {
	case "flat":
		opts = append(opts, libravdb.WithFlat())
	default:
		opts = append(opts, libravdb.WithHNSW(16, 200, 64))
		switch workload.rawStore {
		case "slabby":
			opts = append(opts, libravdb.WithRawVectorStoreSlabby(4096))
		default:
			opts = append(opts, libravdb.WithRawVectorStoreMemory())
		}
	}

	collection, err := db.CreateCollection(context.Background(), "openclaw_memory", opts...)
	if err != nil {
		db.Close()
		tb.Fatalf("failed to create collection: %v", err)
	}

	return db, collection
}

func generateOpenClawEntries(count, dimension int, seed int64) []*libravdb.VectorEntry {
	rng := rand.New(rand.NewSource(seed))
	entries := make([]*libravdb.VectorEntry, count)
	vectorBacking := make([]float32, count*dimension)
	for i := 0; i < count; i++ {
		offset := i * dimension
		vector := vectorBacking[offset : offset+dimension : offset+dimension]
		var norm float64
		for j := range vector {
			v := rng.Float64()*2 - 1
			norm += v * v
			vector[j] = float32(v)
		}
		norm = math.Sqrt(norm)
		if norm == 0 {
			norm = 1
		}
		for j := range vector {
			vector[j] = float32(float64(vector[j]) / norm)
		}

		entries[i] = &libravdb.VectorEntry{
			ID:     fmt.Sprintf("mem_%d", i),
			Vector: vector,
			Metadata: map[string]interface{}{
				"scope":       []string{"session", "user", "global"}[i%3],
				"role":        []string{"user", "assistant"}[i%2],
				"token_count": 32 + (i % 96),
				"kind":        []string{"fact", "preference", "summary", "artifact"}[i%4],
			},
		}
	}
	return entries
}

func generateOpenClawQueries(entries []*libravdb.VectorEntry, count int) [][]float32 {
	if count <= 0 {
		return [][]float32{entries[0].Vector}
	}

	queries := make([][]float32, count)
	for i := 0; i < count; i++ {
		src := entries[(i*17)%len(entries)].Vector
		query := make([]float32, len(src))
		copy(query, src)
		queries[i] = query
	}
	return queries
}

func percentileDuration(values []time.Duration, pct float64) time.Duration {
	if len(values) == 0 {
		return 0
	}

	sorted := make([]time.Duration, len(values))
	copy(sorted, values)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	if pct <= 0 {
		return sorted[0]
	}
	if pct >= 1 {
		return sorted[len(sorted)-1]
	}

	idx := int(math.Ceil(float64(len(sorted))*pct)) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func durationAvg(values []time.Duration) time.Duration {
	if len(values) == 0 {
		return 0
	}
	var total time.Duration
	for _, v := range values {
		total += v
	}
	return total / time.Duration(len(values))
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
