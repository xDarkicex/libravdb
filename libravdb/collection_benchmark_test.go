package libravdb

import (
	"context"
	"fmt"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/storage"
)

func BenchmarkCollectionAsyncHNSWInsert(b *testing.B) {
	ctx := context.Background()
	db, err := Open(
		WithStoragePath(testDBPathBench(b)),
		WithDurability(DurabilitySynchronous),
		WithAsyncIndexing(4096, 4),
		WithMaxConcurrentWrites(32),
		WithMaxWriteQueueDepth(4096),
	)
	if err != nil {
		b.Fatalf("open: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(
		ctx,
		"async_hnsw",
		WithDimension(768),
		WithMetric(L2Distance),
		WithHNSW(16, 100, 200),
		WithIDMapCapacity(max(b.N+16, 4096)),
	)
	if err != nil {
		b.Fatalf("create collection: %v", err)
	}

	const fixtureCount = 256
	vectors := make([][]float32, fixtureCount)
	for i := range vectors {
		vectors[i] = benchVector(768, i+1)
	}
	var before storage.WriteStats
	if provider, ok := db.storage.(storage.WriteStatsProvider); ok {
		before = provider.WriteStats()
	}

	var nextID atomic.Uint64
	b.SetParallelism(4)
	b.SetBytes(768 * 4)
	b.ReportAllocs()
	b.ResetTimer()
	acceptedStart := time.Now()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			id := nextID.Add(1)
			if err := collection.Insert(ctx, strconv.FormatUint(id, 10), vectors[id%fixtureCount], nil); err != nil {
				b.Errorf("insert: %v", err)
				return
			}
		}
	})
	acceptedElapsed := time.Since(acceptedStart)
	statsAtAck := collection.IndexingStats()
	b.StopTimer()
	if err := collection.FlushIndex(ctx); err != nil {
		b.Fatalf("flush index: %v", err)
	}
	graphReadyElapsed := time.Since(acceptedStart)

	if acceptedElapsed > 0 {
		b.ReportMetric(float64(b.N)/acceptedElapsed.Seconds(), "accepted_writes/s")
	}
	if graphReadyElapsed > 0 {
		b.ReportMetric(float64(b.N)/graphReadyElapsed.Seconds(), "graph_ready/s")
	}
	stats := collection.IndexingStats()
	b.ReportMetric(float64(stats.LSNLag), "lsn_lag")
	b.ReportMetric(float64(statsAtAck.Pending), "pending_at_ack")
	b.ReportMetric(float64(statsAtAck.LSNLag), "lsn_lag_at_ack")
	if provider, ok := db.storage.(storage.WriteStatsProvider); ok {
		after := provider.WriteStats()
		transactions := after.WALTransactions - before.WALTransactions
		entries := after.BufferedVectorEntries - before.BufferedVectorEntries
		if transactions > 0 {
			b.ReportMetric(float64(entries)/float64(transactions), "entries/wal_tx")
		}
	}
}

func BenchmarkCollectionInsert(b *testing.B) {
	benchmarkCollectionInsert(b, false, DurabilitySynchronous)
}

func BenchmarkCollectionInsertUnsafeNoSync(b *testing.B) {
	benchmarkCollectionInsert(b, false, DurabilityUnsafeNoSync)
}

func BenchmarkCollectionInsertSteadyState(b *testing.B) {
	benchmarkCollectionInsertSteadyStateBatch(b, DurabilitySynchronous)
}

func BenchmarkCollectionInsertSteadyStateUnsafeNoSync(b *testing.B) {
	benchmarkCollectionInsertSteadyStateBatch(b, DurabilityUnsafeNoSync)
}

func benchmarkCollectionInsert(b *testing.B, logDelta bool, durability DurabilityMode) {
	benchmarks := []struct {
		name string
		opts []CollectionOption
	}{
		{
			name: "Flat",
			opts: []CollectionOption{WithFlat()},
		},
		{
			name: "HNSW",
			opts: []CollectionOption{WithHNSW(16, 100, 50)},
		},
		{
			name: "IVFPQ",
			opts: []CollectionOption{WithIVFPQ(8, 4)},
		},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			ctx := context.Background()
			db, err := Open(WithStoragePath(testDBPathBench(b)), WithDurability(durability))
			if err != nil {
				b.Fatalf("new db: %v", err)
			}
			defer db.Close()

			opts := append([]CollectionOption{
				WithDimension(64),
				WithMetric(CosineDistance),
			}, bm.opts...)

			collection, err := db.CreateCollection(ctx, "bench_insert", opts...)
			if err != nil {
				b.Fatalf("create collection: %v", err)
			}

			if bm.name == "IVFPQ" {
				seed := make([]VectorEntry, 8)
				for i := range seed {
					seed[i] = VectorEntry{
						ID:     fmt.Sprintf("seed_%d", i),
						Vector: benchVector(64, i),
					}
				}
				if err := collection.InsertBatch(ctx, seed); err != nil {
					b.Fatalf("seed insert batch: %v", err)
				}
			}

			var statsBefore storage.WriteStats
			var hasStats bool
			if statsProvider, ok := db.storage.(storage.WriteStatsProvider); ok {
				statsBefore = statsProvider.WriteStats()
				hasStats = true
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := collection.Insert(ctx, fmt.Sprintf("vec_%d", i), benchVector(64, i+1000), nil); err != nil {
					b.Fatalf("insert: %v", err)
				}
			}
			if hasStats {
				statsAfter := db.storage.(storage.WriteStatsProvider).WriteStats()
				if logDelta {
					b.Logf("steady-state write stats delta: wal_tx=%d wal_bytes=%d batch_flushes=%d batched_entries=%d checkpoints=%d",
						statsAfter.WALTransactions-statsBefore.WALTransactions,
						statsAfter.WALBytes-statsBefore.WALBytes,
						statsAfter.BatchFlushes-statsBefore.BatchFlushes,
						statsAfter.BufferedVectorEntries-statsBefore.BufferedVectorEntries,
						statsAfter.Checkpoints-statsBefore.Checkpoints)
				} else {
					b.Logf("write stats: wal_tx=%d wal_bytes=%d batch_flushes=%d batched_entries=%d checkpoints=%d",
						statsAfter.WALTransactions, statsAfter.WALBytes, statsAfter.BatchFlushes, statsAfter.BufferedVectorEntries, statsAfter.Checkpoints)
				}
			}
		})
	}
}

func benchmarkCollectionInsertSteadyStateBatch(b *testing.B, durability DurabilityMode) {
	const batchSize = 256

	benchmarks := []struct {
		name string
		opts []CollectionOption
	}{
		{
			name: "Flat",
			opts: []CollectionOption{WithFlat()},
		},
		{
			name: "HNSW",
			opts: []CollectionOption{WithHNSW(16, 100, 50)},
		},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			ctx := context.Background()
			db, err := Open(WithStoragePath(testDBPathBench(b)), WithDurability(durability))
			if err != nil {
				b.Fatalf("new db: %v", err)
			}
			defer db.Close()

			opts := append([]CollectionOption{
				WithDimension(64),
				WithMetric(CosineDistance),
			}, bm.opts...)

			collection, err := db.CreateCollection(ctx, "bench_insert", opts...)
			if err != nil {
				b.Fatalf("create collection: %v", err)
			}

			var statsBefore storage.WriteStats
			var hasStats bool
			if statsProvider, ok := db.storage.(storage.WriteStatsProvider); ok {
				statsBefore = statsProvider.WriteStats()
				hasStats = true
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				entries := make([]VectorEntry, batchSize)
				for j := 0; j < batchSize; j++ {
					idx := i*batchSize + j
					entries[j] = VectorEntry{
						ID:     fmt.Sprintf("vec_%d", idx),
						Vector: benchVector(64, idx+1000),
					}
				}
				if err := collection.InsertBatch(ctx, entries); err != nil {
					b.Fatalf("insert batch: %v", err)
				}
			}
			if hasStats {
				statsAfter := db.storage.(storage.WriteStatsProvider).WriteStats()
				b.Logf("steady-state batch write stats delta: wal_tx=%d wal_bytes=%d batch_flushes=%d batched_entries=%d checkpoints=%d",
					statsAfter.WALTransactions-statsBefore.WALTransactions,
					statsAfter.WALBytes-statsBefore.WALBytes,
					statsAfter.BatchFlushes-statsBefore.BatchFlushes,
					statsAfter.BufferedVectorEntries-statsBefore.BufferedVectorEntries,
					statsAfter.Checkpoints-statsBefore.Checkpoints)
			}
		})
	}
}

func BenchmarkCollectionSearch(b *testing.B) {
	benchmarks := []struct {
		name       string
		opts       []CollectionOption
		entryCount int
	}{
		{
			name:       "Flat",
			opts:       []CollectionOption{WithFlat()},
			entryCount: 4096,
		},
		{
			name:       "FlatSharded",
			opts:       []CollectionOption{WithFlat(), WithSharding(true)},
			entryCount: 4096,
		},
		{
			name:       "HNSW",
			opts:       []CollectionOption{WithHNSW(16, 100, 50)},
			entryCount: 4096,
		},
		{
			name:       "HNSWSharded",
			opts:       []CollectionOption{WithHNSW(16, 100, 50), WithSharding(true)},
			entryCount: 4096,
		},
		{
			name:       "IVFPQ",
			opts:       []CollectionOption{WithIVFPQ(8, 4)},
			entryCount: 4096,
		},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			ctx := context.Background()
			db, err := Open(WithStoragePath(testDBPathBench(b)))
			if err != nil {
				b.Fatalf("new db: %v", err)
			}
			defer db.Close()

			opts := append([]CollectionOption{
				WithDimension(64),
				WithMetric(CosineDistance),
			}, bm.opts...)

			collection, err := db.CreateCollection(ctx, "bench_search", opts...)
			if err != nil {
				b.Fatalf("create collection: %v", err)
			}

			entries := make([]VectorEntry, bm.entryCount)
			for i := range entries {
				entries[i] = VectorEntry{
					ID:     fmt.Sprintf("vec_%d", i),
					Vector: benchVector(64, i),
				}
			}
			if err := collection.InsertBatch(ctx, entries); err != nil {
				b.Fatalf("insert batch: %v", err)
			}

			query := benchVector(64, 13)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := collection.Search(ctx, query, 10); err != nil {
					b.Fatalf("search: %v", err)
				}
			}
		})
	}
}

func benchVector(dim, seed int) []float32 {
	vector := make([]float32, dim)
	base := float32((seed%17)+1) / 17
	for i := range vector {
		vector[i] = base + float32((seed+i)%7)/10
	}
	return vector
}

func testDBPathBench(b *testing.B) string {
	b.Helper()
	return b.TempDir() + "/bench.libravdb"
}
