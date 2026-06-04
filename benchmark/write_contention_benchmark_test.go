// Benchmark measuring write-path throughput under concurrent load.
// Validates Issue #2 Slice 1: ordinal pre-reservation + payload encoding hoist.
//
// Run with: go test -v -run TestWriteContentionBenchmark ./benchmark/
package benchmark

import (
	"context"
	"fmt"
	"math/rand"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/libravdb"
)

type contentionResult struct {
	name            string
	workers         int
	totalOps        int
	duration        time.Duration
	opsPerSec       float64
	p50Latency      time.Duration
	p95Latency      time.Duration
	p99Latency      time.Duration
	avgLatency      time.Duration
}

func (r contentionResult) String() string {
	return fmt.Sprintf(
		`%s | workers=%2d | ops=%6d | %8s | %10.0f ops/s | p50=%8s p95=%8s p99=%8s avg=%8s`,
		r.name, r.workers, r.totalOps, r.duration.Round(time.Millisecond),
		r.opsPerSec,
		r.p50Latency.Round(time.Microsecond),
		r.p95Latency.Round(time.Microsecond),
		r.p99Latency.Round(time.Microsecond),
		r.avgLatency.Round(time.Microsecond),
	)
}

// TestWriteContentionBenchmark measures sharded InsertBatch and concurrent direct
// Insert throughput at varying worker counts.
func TestWriteContentionBenchmark(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping contention benchmark in short mode")
	}

	const (
		dim            = 128
		totalVectors   = 20000
		batchSize      = 100
		vectorsPerBatch = totalVectors / batchSize // 200 entries per batch
	)

	ctx := context.Background()

	// ---- Sharded InsertBatch benchmarks ----
	for _, workers := range []int{1, 4, 8, 16} {
		result := benchShardedInsertBatch(t, ctx, dim, totalVectors, batchSize, workers)
		t.Log(result)
	}

	// ---- Concurrent Direct Insert benchmarks ----
	for _, workers := range []int{4, 8} {
		result := benchConcurrentDirectInsert(t, ctx, dim, totalVectors, workers)
		t.Log(result)
	}
}

func benchShardedInsertBatch(tb testing.TB, ctx context.Context, dim, totalVectors, batchSize, workers int) contentionResult {
	tmpDir := tb.TempDir()
	dbPath := tmpDir + "/sharded_batch_bench.libravdb"

	db, err := libravdb.New(
		libravdb.WithStoragePath(dbPath),
		libravdb.WithMetrics(false),
	)
	if err != nil {
		tb.Fatalf("create db: %v", err)
	}
	defer db.Close()

	coll, err := db.CreateCollection(ctx, "bench",
		libravdb.WithDimension(dim),
		libravdb.WithHNSW(16, 100, 50),
	)
	if err != nil {
		tb.Fatalf("create collection: %v", err)
	}

	rng := rand.New(rand.NewSource(42))
	vectors := make([]libravdb.VectorEntry, totalVectors)
	for i := 0; i < totalVectors; i++ {
		vectors[i] = libravdb.VectorEntry{
			ID:     fmt.Sprintf("vec_%06d", i),
			Vector: genBenchVector(rng, dim),
		}
	}

	batches := make([][]libravdb.VectorEntry, 0, totalVectors/batchSize)
	for i := 0; i < totalVectors; i += batchSize {
		end := i + batchSize
		if end > totalVectors {
			end = totalVectors
		}
		batches = append(batches, vectors[i:end])
	}

	var wg sync.WaitGroup
	workCh := make(chan []libravdb.VectorEntry, len(batches))
	for _, b := range batches {
		workCh <- b
	}
	close(workCh)

	latencies := make([]time.Duration, 0, len(batches))
	var latMu sync.Mutex
	var firstErr error
	var errOnce sync.Once

	start := time.Now()
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for batch := range workCh {
				batchStart := time.Now()
				if err := coll.InsertBatch(ctx, batch); err != nil {
					errOnce.Do(func() { firstErr = err })
					return
				}
				elapsed := time.Since(batchStart)
				latMu.Lock()
				latencies = append(latencies, elapsed)
				latMu.Unlock()
			}
		}()
	}
	wg.Wait()
	duration := time.Since(start)

	if firstErr != nil {
		tb.Fatalf("InsertBatch error: %v", firstErr)
	}

	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	return contentionResult{
		name:       "InsertBatch(sharded)",
		workers:    workers,
		totalOps:   totalVectors,
		duration:   duration,
		opsPerSec:  float64(totalVectors) / duration.Seconds(),
		p50Latency: latencies[len(latencies)*50/100],
		p95Latency: latencies[len(latencies)*95/100],
		p99Latency: latencies[len(latencies)*99/100],
		avgLatency: avgLatency(latencies),
	}
}

func benchConcurrentDirectInsert(tb testing.TB, ctx context.Context, dim, totalVectors, workers int) contentionResult {
	tmpDir := tb.TempDir()
	dbPath := tmpDir + "/concurrent_insert_bench.libravdb"

	db, err := libravdb.New(
		libravdb.WithStoragePath(dbPath),
		libravdb.WithMetrics(false),
	)
	if err != nil {
		tb.Fatalf("create db: %v", err)
	}
	defer db.Close()

	coll, err := db.CreateCollection(ctx, "bench",
		libravdb.WithDimension(dim),
		libravdb.WithFlat(),
	)
	if err != nil {
		tb.Fatalf("create collection: %v", err)
	}

	rng := rand.New(rand.NewSource(42))
	vectors := make([]libravdb.VectorEntry, totalVectors)
	for i := 0; i < totalVectors; i++ {
		vectors[i] = libravdb.VectorEntry{
			ID:     fmt.Sprintf("vec_%06d", i),
			Vector: genBenchVector(rng, dim),
		}
	}

	var wg sync.WaitGroup
	workCh := make(chan libravdb.VectorEntry, totalVectors)
	for _, v := range vectors {
		workCh <- v
	}
	close(workCh)

	latencies := make([]time.Duration, 0, totalVectors)
	var latMu sync.Mutex
	var firstErr error
	var errOnce sync.Once

	start := time.Now()
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for vec := range workCh {
				insStart := time.Now()
				if err := coll.Insert(ctx, vec.ID, vec.Vector, vec.Metadata); err != nil {
					errOnce.Do(func() { firstErr = err })
					return
				}
				elapsed := time.Since(insStart)
				latMu.Lock()
				latencies = append(latencies, elapsed)
				latMu.Unlock()
			}
		}()
	}
	wg.Wait()
	duration := time.Since(start)

	if firstErr != nil {
		tb.Fatalf("Insert error: %v", firstErr)
	}

	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	return contentionResult{
		name:       "Insert(direct)",
		workers:    workers,
		totalOps:   totalVectors,
		duration:   duration,
		opsPerSec:  float64(totalVectors) / duration.Seconds(),
		p50Latency: latencies[len(latencies)*50/100],
		p95Latency: latencies[len(latencies)*95/100],
		p99Latency: latencies[len(latencies)*99/100],
		avgLatency: avgLatency(latencies),
	}
}

func avgLatency(latencies []time.Duration) time.Duration {
	if len(latencies) == 0 {
		return 0
	}
	var sum time.Duration
	for _, l := range latencies {
		sum += l
	}
	return sum / time.Duration(len(latencies))
}
