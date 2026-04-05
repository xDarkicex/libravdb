// Benchmark comparing auto-index threshold candidates for Flat vs HNSW decision.
// This is a library-general diagnostic - measures query latency at various collection
// sizes to determine the practical crossover point where HNSW becomes better than Flat.
//
// Run with: go test -v -run TestAutoIndexThresholdBenchmark ./benchmark/ -bench=.
package benchmark

import (
	"context"
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/libravdb"
)

// Threshold candidates to benchmark.
var thresholdCandidates = []struct {
	name string
	hnsw  int
	ivfpq int
}{
	{"default_10k", 10000, 1000000},
	{"lowered_1k", 1000, 100000},
	{"lowered_2k", 2000, 100000},
	{"lowered_4k", 4096, 100000},
}

// Collection sizes to test.
var sizeSamples = []int{500, 1000, 2000, 5000, 10000}

const (
	benchDim       = 32
	benchSearchK   = 8
	benchRuns      = 5
	benchWarmup    = 1
	benchSeed int64 = 42
)

// genVector creates a pseudo-random normalized vector.
func genVector(rng *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	var norm float32
	for i := 0; i < dim; i++ {
		v[i] = rng.Float32()*2 - 1
		norm += v[i] * v[i]
	}
	norm = float32(1) / norm
	for i := 0; i < dim; i++ {
		v[i] *= norm
	}
	return v
}

// measureThreshold creates a collection at given size with explicit index type
// based on threshold, inserts vectors, and measures search latency.
func measureThreshold(tb testing.TB, size int, threshold int) ([]float64, string, error) {
	ctx := context.Background()
	tmpDir := tb.TempDir()
	dbPath := tmpDir + "/bench.libravdb"

	db, err := libravdb.New(
		libravdb.WithStoragePath(dbPath),
		libravdb.WithMetrics(false),
	)
	if err != nil {
		return nil, "", fmt.Errorf("failed to create database: %w", err)
	}
	defer db.Close()

	// Determine index type based on threshold
	idxType := libravdb.Flat
	if size >= threshold {
		idxType = libravdb.HNSW
	}

	col, err := db.CreateCollection(ctx, "probe",
		libravdb.WithDimension(benchDim),
		func(c *libravdb.CollectionConfig) error {
			c.IndexType = idxType
			return nil
		},
	)
	if err != nil {
		return nil, "", fmt.Errorf("failed to create collection: %w", err)
	}

	// Insert vectors
	rng := rand.New(rand.NewSource(benchSeed))
	entries := make([]libravdb.VectorEntry, size)
	for i := 0; i < size; i++ {
		vec := genVector(rng, benchDim)
		entries[i] = libravdb.VectorEntry{
			ID:       fmt.Sprintf("vec_%d", i),
			Vector:   vec,
			Metadata: map[string]interface{}{"index": i},
		}
	}

	if err := col.InsertBatch(ctx, entries); err != nil {
		return nil, "", fmt.Errorf("batch insert failed: %w", err)
	}

	// Generate query vector
	queryVec := genVector(rng, benchDim)

	// Warm-up
	_, _ = col.Search(ctx, queryVec, benchSearchK)

	// Measured runs
	var latencies []float64
	for i := 0; i < benchRuns; i++ {
		start := time.Now()
		results, err := col.Search(ctx, queryVec, benchSearchK)
		elapsed := time.Since(start)
		if err != nil {
			return nil, "", fmt.Errorf("search failed: %w", err)
		}
		if len(results.Results) == 0 {
			return nil, "", fmt.Errorf("no search results")
		}
		latencies = append(latencies, float64(elapsed.Microseconds())/1000.0)
	}

	return latencies, idxType.String(), nil
}

// avg computes arithmetic mean.
func avg(latencies []float64) float64 {
	if len(latencies) == 0 {
		return 0
	}
	var sum float64
	for _, l := range latencies {
		sum += l
	}
	return sum / float64(len(latencies))
}

// sprintLatencies returns a compact string representation.
func sprintLatencies(latencies []float64) string {
	if len(latencies) == 0 {
		return "[]"
	}
	s := "["
	for i, l := range latencies {
		if i > 0 {
			s += " "
		}
		s += fmt.Sprintf("%.1f", l)
	}
	s += "]"
	return s
}

// TestAutoIndexThresholdBenchmark compares query latency across collection sizes
// and threshold candidates to determine the practical crossover point.
func TestAutoIndexThresholdBenchmark(t *testing.T) {
	t.Log("=== Auto-Index Threshold Benchmark ===")
	t.Logf("Dimension: %d, Search k: %d, Runs: %d warmup + %d measured",
		benchDim, benchSearchK, benchWarmup, benchRuns)
	t.Logf("Threshold candidates: %d", len(thresholdCandidates))
	t.Logf("Collection sizes: %v", sizeSamples)
	t.Log("")

	type probeResult struct {
		indexType string
		mean      float64
		raw       []float64
	}
	results := make(map[string]map[int]probeResult)

	for _, tc := range thresholdCandidates {
		t.Logf("--- Testing threshold: %s (HNSW @ %d) ---", tc.name, tc.hnsw)
		results[tc.name] = make(map[int]probeResult)

		for _, size := range sizeSamples {
			// Determine expected index type for reporting
			expectedType := "Flat"
			if size >= tc.hnsw {
				expectedType = "HNSW"
			}

			latencies, idxType, err := measureThreshold(t, size, tc.hnsw)
			if err != nil {
				t.Errorf("size=%d threshold=%s: %v", size, tc.name, err)
				continue
			}

			mean := avg(latencies)
			results[tc.name][size] = probeResult{indexType: idxType, mean: mean, raw: latencies}

			t.Logf("  size=%-6d threshold=%s expected=%s got=%s mean=%.2fms runs=%s",
				size, tc.name, expectedType, idxType, mean, sprintLatencies(latencies))
		}
	}

	// Summary table
	t.Log("")
	t.Log("=== Summary: Mean latency (ms) by size and threshold ===")
	header := "size\t"
	for _, tc := range thresholdCandidates {
		header += fmt.Sprintf("%s\t", tc.name)
	}
	t.Log(header)
	t.Log("-------------------------------------------------------------------")
	for _, size := range sizeSamples {
		row := fmt.Sprintf("%d\t", size)
		for _, tc := range thresholdCandidates {
			if r, ok := results[tc.name][size]; ok {
				row += fmt.Sprintf("%.2f(%s)\t", r.mean, r.indexType)
			} else {
				row += "N/A\t"
			}
		}
		t.Log(row)
	}

	// Crossover analysis: compare lowered thresholds against default
	t.Log("")
	t.Log("=== Crossover Analysis ===")
	for _, tc := range thresholdCandidates {
		if tc.name == "default_10k" {
			continue
		}
		t.Logf("Comparing %s vs default_10k:", tc.name)
		for _, size := range sizeSamples {
			defaultResult, defaultOK := results["default_10k"][size]
			loweredResult, loweredOK := results[tc.name][size]

			if defaultOK && loweredOK && defaultResult.mean > 0 && loweredResult.mean > 0 {
				ratio := loweredResult.mean / defaultResult.mean
				if ratio > 1.0 {
					t.Logf("  size=%d: default=%.2fms vs %s=%.2fms (default %.0f%% faster)",
						size, defaultResult.mean, tc.name, loweredResult.mean, (ratio-1)*100)
				} else if ratio < 1.0 {
					t.Logf("  size=%d: default=%.2fms vs %s=%.2fms (%s %.0f%% faster)",
						size, defaultResult.mean, tc.name, loweredResult.mean, tc.name, (1/ratio-1)*100)
				} else {
					t.Logf("  size=%d: both ~%.2fms (no meaningful difference)", size, defaultResult.mean)
				}
			}
		}
	}
}
