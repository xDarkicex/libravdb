package libravdb

import (
	"context"
	"fmt"
	"testing"
)

func BenchmarkCollectionInsert(b *testing.B) {
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
			db, err := New(WithStoragePath(testDBPathBench(b)))
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

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := collection.Insert(ctx, fmt.Sprintf("vec_%d", i), benchVector(64, i+1000), nil); err != nil {
					b.Fatalf("insert: %v", err)
				}
			}
		})
	}
}

func BenchmarkCollectionSearch(b *testing.B) {
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
			db, err := New(WithStoragePath(testDBPathBench(b)))
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

			entries := make([]VectorEntry, 256)
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
