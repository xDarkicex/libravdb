package graph

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestSegmentFlushAndLoadBasic(t *testing.T) {
	dir := t.TempDir()
	segPath := filepath.Join(dir, "test.seg")

	gi, err := NewGraph(DefaultGraphConfig())
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	g := gi.(*graphStore)
	defer g.Close()

	// Add some edges
	txn := &Txn{ID: 1}
	if err := g.AddEdge(txn, 1, 2, 0.5, 1); err != nil {
		t.Fatal(err)
	}
	if err := g.AddEdge(txn, 1, 3, 0.8, 1); err != nil {
		t.Fatal(err)
	}
	if err := txn.Commit(context.Background()); err != nil {
		t.Fatal(err)
	}

	if err := g.FlushToSegment(segPath); err != nil {
		t.Fatalf("FlushToSegment: %v", err)
	}

	// Load into a new graph
	gi2, err := NewGraph(DefaultGraphConfig())
	if err != nil {
		t.Fatalf("NewGraph g2: %v", err)
	}
	g2 := gi2.(*graphStore)
	defer g2.Close()

	if err := g2.LoadFromSegment(segPath, nil); err != nil {
		t.Fatalf("LoadFromSegment: %v", err)
	}

	edges, err := g2.Neighbors(1)
	if err != nil {
		t.Fatal(err)
	}
	if len(edges) != 2 {
		t.Fatalf("expected 2 edges, got %d", len(edges))
	}

	// check reverse index
	rEdges, err := g2.neighborsFromTable(2, g2.reverse.locator, g2.reverse.pool, g2.cfg.PageShards)
	if err != nil {
		t.Fatal(err)
	}
	if len(rEdges) != 1 || rEdges[0].Target != 1 {
		t.Fatalf("reverse edge missing or wrong: %v", rEdges)
	}
}

func TestSegmentCorruption(t *testing.T) {
	dir := t.TempDir()
	segPath := filepath.Join(dir, "corrupt.seg")

	gi, err := NewGraph(DefaultGraphConfig())
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	g := gi.(*graphStore)
	defer g.Close()

	txn := &Txn{ID: 1}
	if err := g.AddEdge(txn, 10, 20, 0.5, 1); err != nil {
		t.Fatal(err)
	}
	if err := txn.Commit(context.Background()); err != nil {
		t.Fatal(err)
	}

	if err := g.FlushToSegment(segPath); err != nil {
		t.Fatalf("FlushToSegment: %v", err)
	}

	// Corrupt the file
	data, err := os.ReadFile(segPath)
	if err != nil {
		t.Fatal(err)
	}

	if len(data) <= SegmentHeaderSize+5 {
		t.Fatalf("segment too small to corrupt reliably: len %d", len(data))
	}
	data[SegmentHeaderSize+5] ^= 0xFF

	if err := os.WriteFile(segPath, data, 0644); err != nil {
		t.Fatal(err)
	}

	gi2, err := NewGraph(DefaultGraphConfig())
	if err != nil {
		t.Fatalf("NewGraph g2: %v", err)
	}
	g2 := gi2.(*graphStore)
	defer g2.Close()

	err = g2.LoadFromSegment(segPath, nil)
	if err == nil {
		t.Fatal("expected CRC error on corrupted segment")
	}
}
