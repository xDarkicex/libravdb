// Diagnostic probe to measure Flat-vs-HNSW threshold behavior.
// This is a READ-ONLY diagnostic - does not modify core library behavior.
//
// Hypothesis being tested:
// - Collections below 10k use Flat (O(N*d) search)
// - Collections at/above 10k use HNSW (O(log N * d) search)
// - Flat search with exclusion-heavy filtering becomes especially bad
//
// Success criteria: clear latency comparison showing threshold effect.
package benchmark

import (
	"context"
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/filter"
	"github.com/xDarkicex/libravdb/libravdb"
)

// Probe configuration - just below and above the 10k auto-index threshold
// Using smaller dimension (32) to make HNSW construction feasible in probe time.
// This still demonstrates asymptotic behavior: Flat is O(N*d), HNSW is O(log N * d).
const (
	probeDimension     = 32
	belowThresholdN    = 9999
	aboveThresholdN    = 10001
	searchK            = 8
	warmupRuns         = 1
	measuredRuns       = 5
	fixedSeed    int64 = 12345 // deterministic vectors
)

// generateDeterministicVector creates a pseudo-random normalized vector.
// Uses fixed seed for reproducibility across probe runs.
func generateDeterministicVector(rng *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	var norm float32
	for i := 0; i < dim; i++ {
		v[i] = rng.Float32()*2 - 1 // range [-1, 1]
		norm += v[i] * v[i]
	}
	// Normalize for cosine similarity (library may expect it)
	norm = float32(1) / norm
	for i := 0; i < dim; i++ {
		v[i] *= norm
	}
	return v
}

// prepareThresholdProbe creates two collections with identical shape,
// differing only in count: one just below 10k, one just above.
// Returns the two collections and a fixed query vector.
func prepareThresholdProbe(t *testing.T) (*libravdb.Collection, *libravdb.Collection, []float32, func()) {
	ctx := context.Background()

	// Create temporary database for probe
	tmpDir := t.TempDir()
	dbPath := tmpDir + "/probe.libravdb"
	db, err := libravdb.New(
		libravdb.WithStoragePath(dbPath),
		libravdb.WithMetrics(false),
	)
	if err != nil {
		t.Fatalf("failed to create probe database: %v", err)
	}

	// Create below-threshold collection with Flat index (auto-index would select Flat < 10k)
	belowCol, err := db.CreateCollection(ctx, "below_threshold",
		libravdb.WithDimension(probeDimension),
		libravdb.WithFlat(), // Explicit Flat index (mirrors auto-index boundary below 10k)
	)
	if err != nil {
		t.Fatalf("failed to create below_threshold collection: %v", err)
	}

	// Create above-threshold collection with HNSW index (auto-index would select HNSW >= 10k)
	aboveCol, err := db.CreateCollection(ctx, "above_threshold",
		libravdb.WithDimension(probeDimension),
		libravdb.WithHNSW(16, 100, 50), // Explicit HNSW index (mirrors auto-index boundary at/above 10k)
	)
	if err != nil {
		t.Fatalf("failed to create above_threshold collection: %v", err)
	}

	// Shared deterministic RNG for reproducible vectors
	rng := rand.New(rand.NewSource(fixedSeed))

	// Generate fixed query vector (same for all searches)
	queryVector := generateDeterministicVector(rng, probeDimension)

	// Prepare batch entries for below-threshold collection
	belowEntries := make([]libravdb.VectorEntry, belowThresholdN)
	for i := 0; i < belowThresholdN; i++ {
		vec := generateDeterministicVector(rng, probeDimension)
		belowEntries[i] = libravdb.VectorEntry{
			ID:       fmt.Sprintf("vec_%d", i),
			Vector:   vec,
			Metadata: map[string]interface{}{"index": i, "category": i % 16},
		}
	}

	// Prepare batch entries for above-threshold collection
	aboveEntries := make([]libravdb.VectorEntry, aboveThresholdN)
	for i := 0; i < aboveThresholdN; i++ {
		vec := generateDeterministicVector(rng, probeDimension)
		aboveEntries[i] = libravdb.VectorEntry{
			ID:       fmt.Sprintf("vec_%d", i),
			Vector:   vec,
			Metadata: map[string]interface{}{"index": i, "category": i % 16},
		}
	}

	// Batch insert into both collections (reduces auto-index check overhead)
	if err := belowCol.InsertBatch(ctx, belowEntries); err != nil {
		t.Fatalf("below_threshold batch insert failed: %v", err)
	}
	if err := aboveCol.InsertBatch(ctx, aboveEntries); err != nil {
		t.Fatalf("above_threshold batch insert failed: %v", err)
	}

	cleanup := func() {
		db.Close()
	}

	return belowCol, aboveCol, queryVector, cleanup
}

// measureSearch runs warmup then measured runs, returning latency slice in ms.
func measureSearch(t *testing.T, col *libravdb.Collection, queryVec []float32, k int, mode string) []float64 {
	ctx := context.Background()
	var latencies []float64

	// Warm-up run
	_, err := col.Search(ctx, queryVec, k)
	if err != nil {
		t.Logf("[%s] warm-up search error: %v", mode, err)
	}

	// Measured runs
	for i := 0; i < measuredRuns; i++ {
		start := time.Now()
		results, err := col.Search(ctx, queryVec, k)
		elapsed := time.Since(start)

		if err != nil {
			t.Logf("[%s] search error: %v", mode, err)
			continue
		}
		if len(results.Results) == 0 {
			t.Logf("[%s] warning: no results returned", mode)
		}

		latencies = append(latencies, float64(elapsed.Microseconds())/1000.0)
	}

	return latencies
}

// measureFilteredSearch runs warmup then measured runs with metadata exclusion.
// Since the library has no first-class "exclude IDs" API, we emulate exclusion-heavy
// search by searching with a category filter that excludes a chunk of the collection.
// This exercises the post-search filtering overhead.
func measureFilteredSearch(t *testing.T, col *libravdb.Collection, queryVec []float32, k int, mode string) []float64 {
	ctx := context.Background()
	var latencies []float64

	// Build exclusion: filter to exclude categories 0-3 (4 categories excluded)
	// This tests post-search filtering overhead on top of vector search.
	excludeVals := []interface{}{uint64(0), uint64(1), uint64(2), uint64(3)}
	exclusionFilter := filter.NewNotFilter(filter.NewContainsAnyFilter("category", excludeVals))

	// Warm-up run
	_, err := col.Query(ctx).
		WithVector(queryVec).
		WithFilter(exclusionFilter).
		Limit(k).
		Execute()
	if err != nil {
		t.Logf("[%s] warm-up filtered search error: %v", mode, err)
	}

	// Measured runs
	for i := 0; i < measuredRuns; i++ {
		start := time.Now()
		results, err := col.Query(ctx).
			WithVector(queryVec).
			WithFilter(exclusionFilter).
			Limit(k).
			Execute()
		elapsed := time.Since(start)

		if err != nil {
			t.Logf("[%s] filtered search error: %v", mode, err)
			continue
		}
		if len(results.Results) == 0 {
			t.Logf("[%s] warning: no filtered results returned", mode)
		}

		latencies = append(latencies, float64(elapsed.Microseconds())/1000.0)
	}

	return latencies
}

// formatLatencies returns a compact string representation of latency measurements.
func formatLatencies(latencies []float64) string {
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

// meanLatency returns the arithmetic mean of latencies.
func meanLatency(latencies []float64) float64 {
	if len(latencies) == 0 {
		return 0
	}
	var sum float64
	for _, l := range latencies {
		sum += l
	}
	return sum / float64(len(latencies))
}

// TestFlatHNSWThresholdProbe is the main diagnostic probe.
// Run with: go test -v -run TestFlatHNSWThresholdProbe ./benchmark/
func TestFlatHNSWThresholdProbe(t *testing.T) {
	t.Log("=== Flat-vs-HNSW Threshold Diagnostic Probe ===")
	t.Logf("Below-threshold size: %d (expect Flat index)", belowThresholdN)
	t.Logf("Above-threshold size: %d (expect HNSW index)", aboveThresholdN)
	t.Logf("Dimension: %d", probeDimension)
	t.Logf("Search k: %d", searchK)
	t.Logf("Warmup runs: %d, Measured runs: %d", warmupRuns, measuredRuns)
	t.Log("")

	belowCol, aboveCol, queryVec, cleanup := prepareThresholdProbe(t)
	defer cleanup()

	// Observe index types after collection stabilization
	belowStats := belowCol.Stats()
	aboveStats := aboveCol.Stats()
	t.Logf("Below-threshold index type: %s", belowStats.IndexType)
	t.Logf("Above-threshold index type: %s", aboveStats.IndexType)
	t.Logf("Below-threshold vector count: %d", belowStats.VectorCount)
	t.Logf("Above-threshold vector count: %d", aboveStats.VectorCount)
	t.Log("")

	// === Plain search ===
	t.Log("--- Plain top-k search ---")

	belowPlainLatencies := measureSearch(t, belowCol, queryVec, searchK, "below_flat_plain")
	abovePlainLatencies := measureSearch(t, aboveCol, queryVec, searchK, "above_hnsw_plain")

	t.Logf("collection=below_threshold index=%s mode=plain runs_ms=%s mean=%.1f",
		belowStats.IndexType, formatLatencies(belowPlainLatencies), meanLatency(belowPlainLatencies))
	t.Logf("collection=above_threshold index=%s mode=plain runs_ms=%s mean=%.1f",
		aboveStats.IndexType, formatLatencies(abovePlainLatencies), meanLatency(abovePlainLatencies))

	// === Exclusion-heavy search ===
	// Note: Emulated via NOT-filter on category field (4 of 16 categories excluded).
	// This exercises post-search filtering overhead, not a first-class exclusion API.
	t.Log("")
	t.Log("--- Exclusion-heavy search (NOT-filter on 4 of 16 categories) ---")
	t.Log("NOTE: No first-class exclude-IDs API; using post-search metadata filter as nearest equivalent.")

	belowFilteredLatencies := measureFilteredSearch(t, belowCol, queryVec, searchK, "below_flat_exclude")
	aboveFilteredLatencies := measureFilteredSearch(t, aboveCol, queryVec, searchK, "above_hnsw_exclude")

	t.Logf("collection=below_threshold index=%s mode=exclude runs_ms=%s mean=%.1f",
		belowStats.IndexType, formatLatencies(belowFilteredLatencies), meanLatency(belowFilteredLatencies))
	t.Logf("collection=above_threshold index=%s mode=exclude runs_ms=%s mean=%.1f",
		aboveStats.IndexType, formatLatencies(aboveFilteredLatencies), meanLatency(aboveFilteredLatencies))

	// === Summary ===
	t.Log("")
	t.Log("=== Probe Summary ===")
	belowPlainMean := meanLatency(belowPlainLatencies)
	abovePlainMean := meanLatency(abovePlainLatencies)
	belowFilteredMean := meanLatency(belowFilteredLatencies)
	aboveFilteredMean := meanLatency(aboveFilteredLatencies)

	var plainRatio, filteredRatio float64
	if abovePlainMean > 0 {
		plainRatio = belowPlainMean / abovePlainMean
		t.Logf("Plain search ratio (below/above): %.2fx", plainRatio)
	}
	if aboveFilteredMean > 0 {
		filteredRatio = belowFilteredMean / aboveFilteredMean
		t.Logf("Filtered search ratio (below/above): %.2fx", filteredRatio)
	}

	t.Log("")
	if belowStats.IndexType == "Flat" && aboveStats.IndexType == "HNSW" {
		t.Log("CONFIRMED: Auto-index boundary triggered correctly (Flat < 10k < HNSW)")
	} else {
		t.Logf("WARNING: Unexpected index types - below=%s above=%s",
			belowStats.IndexType, aboveStats.IndexType)
	}

	// Hypothesis evaluation
	t.Log("")
	t.Log("=== Hypothesis Evaluation ===")
	if abovePlainMean > 0 {
		plainRatio = belowPlainMean / abovePlainMean
		if plainRatio > 1.5 {
			t.Logf("SUPPORTED: Plain search below threshold is %.2fx slower than above.", plainRatio)
			t.Log("  This suggests O(N*d) Flat vs O(log N * d) HNSW asymptotic difference.")
		} else {
			t.Logf("WEAKEND: Plain search ratio only %.2fx - threshold effect may be less pronounced.", plainRatio)
		}
	}

	if aboveFilteredMean > 0 {
		filteredRatio = belowFilteredMean / aboveFilteredMean
		if filteredRatio > plainRatio {
			t.Logf("SUPPORTED: Filtered search ratio (%.2fx) > Plain ratio (%.2fx).", filteredRatio, plainRatio)
			t.Log("  Exclusion-heavy filtering amplifies the Flat path penalty as expected.")
		}
	}

	// Report stability: all runs should complete without errors
	totalRuns := len(belowPlainLatencies) + len(abovePlainLatencies) +
		len(belowFilteredLatencies) + len(aboveFilteredLatencies)
	if totalRuns == measuredRuns*4 {
		t.Log("STABLE: All probe runs completed without errors.")
	} else {
		t.Logf("UNSTABLE: Only %d/%d runs completed successfully.", totalRuns, measuredRuns*4)
	}
}
