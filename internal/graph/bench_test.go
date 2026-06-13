package graph

import (
	"os"
	"testing"
)

func BenchmarkAddEdge(b *testing.B) {
	tmpDir, err := os.MkdirTemp("", "graph_bench_*")
	if err != nil {
		b.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	cfg := DefaultGraphConfig()
	// cfg.DataPath = tmpDir // GraphConfig doesn't have DataPath, graph is off-heap only
	g, err := NewGraph(cfg)
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := g.(*graphStore).AddEdgeWithStamp(nil, uint64(i), uint64(i+1), 1.0, 1, uint32(i)); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNeighbors(b *testing.B) {
	tmpDir, err := os.MkdirTemp("", "graph_bench_*")
	if err != nil {
		b.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	cfg := DefaultGraphConfig()
	g, err := NewGraph(cfg)
	if err != nil {
		b.Fatal(err)
	}

	// Add some edges to node 1
	for i := 0; i < 1000; i++ {
		if err := g.(*graphStore).AddEdgeWithStamp(nil, 1, uint64(10+i), 1.0, 1, uint32(i)); err != nil {
			b.Fatal(err)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = g.Neighbors(1)
	}
}

func BenchmarkEdgeTableIndex(b *testing.B) {
	idx := NewEdgeTableIndex(1024)

	// Pre-populate
	for i := 0; i < 10000; i++ {
		idx.Insert(uint64(i), uint32(i+100))
	}

	b.ResetTimer()
	b.Run("Lookup", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = idx.Lookup(uint64(i % 10000))
		}
	})

	b.Run("Insert", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			idx.Insert(uint64(i+10000), uint32(i+20000))
		}
	})
}
