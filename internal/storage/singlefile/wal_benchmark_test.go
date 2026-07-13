package singlefile

import (
	"context"
	"path/filepath"
	"strconv"
	"sync/atomic"
	"testing"

	"github.com/xDarkicex/libravdb/internal/index"
	"github.com/xDarkicex/libravdb/internal/storage"
)

func BenchmarkWALInsertConcurrent(b *testing.B) {
	for _, tc := range []struct {
		name        string
		sync        bool
		parallelism int
	}{
		{name: "durable_writers_8", sync: true, parallelism: 1},
		{name: "durable_writers_32", sync: true, parallelism: 4},
		{name: "unsafe_no_sync_writers_8", sync: false, parallelism: 1},
	} {
		b.Run(tc.name, func(b *testing.B) {
			path := filepath.Join(b.TempDir(), "wal_insert.libravdb")
			engineIface, err := New(path, WithWALSync(tc.sync))
			if err != nil {
				b.Fatalf("new engine: %v", err)
			}
			engine := engineIface.(*Engine)
			defer engine.Close()

			collection, err := engine.CreateCollection("vectors", &storage.CollectionConfig{
				Dimension:      768,
				Metric:         2,
				IndexType:      0,
				RawVectorStore: "memory",
				RawStoreCap:    b.N + 16,
			})
			if err != nil {
				b.Fatalf("create collection: %v", err)
			}

			vector := make([]float32, 768)
			for i := range vector {
				vector[i] = float32(i&31) / 31
			}

			var nextID atomic.Uint64
			before := engine.WriteStats()
			b.SetBytes(int64(len(vector) * 4))
			b.ReportAllocs()
			b.SetParallelism(tc.parallelism)
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					id := nextID.Add(1)
					if err := collection.Insert(context.Background(), &index.VectorEntry{
						ID:     strconv.FormatUint(id, 10),
						Vector: vector,
					}); err != nil {
						b.Errorf("insert: %v", err)
						return
					}
				}
			})
			b.StopTimer()
			after := engine.WriteStats()
			transactions := after.WALTransactions - before.WALTransactions
			flushes := after.BatchFlushes - before.BatchFlushes
			entries := after.BufferedVectorEntries - before.BufferedVectorEntries
			if transactions > 0 {
				b.ReportMetric(float64(entries)/float64(transactions), "entries/wal_tx")
			}
			b.ReportMetric(float64(transactions), "wal_tx")
			b.ReportMetric(float64(flushes), "flushes")
		})
	}
}
