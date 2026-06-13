package graph

import (
	"context"
	"path/filepath"
	"testing"
)

func TestCompactSegment(t *testing.T) {
	dir := t.TempDir()
	segPath := filepath.Join(dir, "original.seg")
	compactPath := filepath.Join(dir, "compacted.seg")

	// 1. Create a segment
	gi, err := NewGraph(DefaultGraphConfig())
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	g := gi.(*graphStore)

	txn := &Txn{ID: 1}
	if err := g.AddEdge(txn, 1, 2, 0.5, 1); err != nil {
		t.Fatal(err)
	}
	if err := g.AddEdge(txn, 1, 3, 0.8, 2); err != nil {
		t.Fatal(err)
	}
	if err := txn.Commit(context.Background()); err != nil {
		t.Fatal(err)
	}

	// set manifest to version 1
	g.manifest.MinReaderVersion = 1
	if err := g.manifest.RegisterKind(1, "test_kind_1"); err != nil {
		t.Fatal(err)
	}
	if err := g.manifest.RegisterKind(2, "test_kind_2"); err != nil {
		t.Fatal(err)
	}

	if err := g.FlushToSegment(segPath); err != nil {
		t.Fatalf("FlushToSegment: %v", err)
	}
	g.Close()

	// 2. Compact the segment
	if err := CompactSegment(segPath, compactPath); err != nil {
		t.Fatalf("CompactSegment: %v", err)
	}

	// 3. Load the compacted segment
	gi2, err := NewGraph(DefaultGraphConfig())
	if err != nil {
		t.Fatalf("NewGraph g2: %v", err)
	}
	g2 := gi2.(*graphStore)
	defer g2.Close()

	if err := g2.LoadFromSegment(compactPath, nil); err != nil {
		t.Fatalf("LoadFromSegment: %v", err)
	}

	// Verify data
	edges, err := g2.Neighbors(1)
	if err != nil {
		t.Fatal(err)
	}
	if len(edges) != 2 {
		t.Fatalf("expected 2 edges, got %d", len(edges))
	}

	// Verify migration happened
	if g2.manifest.MinReaderVersion != 2 {
		t.Fatalf("expected MinReaderVersion 2 after compaction, got %d", g2.manifest.MinReaderVersion)
	}
	if len(g2.manifest.KindManifest) != 2 {
		t.Fatalf("expected 2 kinds, got %d", len(g2.manifest.KindManifest))
	}
}
