package libravdb

import (
	"errors"
	"testing"

	"github.com/xDarkicex/libravdb/internal/graph"
)

type mockGraph struct {
	Graph
}

func TestWithGraph(t *testing.T) {
	mock := &mockGraph{}
	cfg := &CollectionConfig{}

	opt := WithGraph(mock)
	err := opt(cfg)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if cfg.Graph != mock {
		t.Errorf("expected graph to be set in config")
	}
}

func TestSetGraph(t *testing.T) {
	mock := &mockGraph{}
	c := &Collection{}

	c.SetGraph(mock)

	if c.graph != mock {
		t.Errorf("expected graph to be set on collection")
	}
}

func graphWithEdge(t *testing.T) Graph {
	t.Helper()
	g, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	txn := g.BeginTxn()
	if txn == nil {
		t.Fatal("BeginTxn returned nil")
	}
	if err := g.AddEdge(txn, 11, 22, 1, 7); err != nil {
		t.Fatalf("AddEdge: %v", err)
	}
	if err := txn.ApplyInMemory(); err != nil {
		t.Fatalf("ApplyInMemory: %v", err)
	}
	return g
}

func TestSetGraphWithErrorCopiesLiveTopology(t *testing.T) {
	source := graphWithEdge(t)
	target, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph target: %v", err)
	}
	defer target.Close()
	c := &Collection{}
	c.SetGraph(source)

	if err := c.SetGraphWithError(target); err != nil {
		t.Fatalf("SetGraphWithError: %v", err)
	}
	neighbors, err := target.Neighbors(11)
	if err != nil {
		t.Fatalf("target Neighbors: %v", err)
	}
	if len(neighbors) != 1 || neighbors[0].Target != 22 || neighbors[0].GetKind() != 7 {
		t.Fatalf("copied topology = %#v", neighbors)
	}
	if _, err := source.Neighbors(11); !errors.Is(err, graph.ErrGraphClosed) {
		t.Fatalf("source Neighbors error = %v, want ErrGraphClosed", err)
	}
}

func TestSetGraphWithErrorClosedSourceIsControlled(t *testing.T) {
	source := graphWithEdge(t)
	if err := source.Close(); err != nil {
		t.Fatalf("source Close: %v", err)
	}
	target, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph target: %v", err)
	}
	defer target.Close()
	c := &Collection{}
	c.SetGraph(source)

	if err := c.SetGraphWithError(target); !errors.Is(err, graph.ErrGraphClosed) {
		t.Fatalf("SetGraphWithError error = %v, want ErrGraphClosed", err)
	}
	if c.GetGraph() != target {
		t.Fatal("replacement graph was not attached after closed-source error")
	}
	neighbors, err := target.Neighbors(11)
	if err != nil {
		t.Fatalf("target Neighbors: %v", err)
	}
	if len(neighbors) != 0 {
		t.Fatalf("target unexpectedly received topology from closed source: %#v", neighbors)
	}
	_ = target.Close()
}

func TestSetGraphWithErrorEmptySource(t *testing.T) {
	source, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph source: %v", err)
	}
	target, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph target: %v", err)
	}
	defer target.Close()
	c := &Collection{}
	c.SetGraph(source)

	if err := c.SetGraphWithError(target); err != nil {
		t.Fatalf("SetGraphWithError empty source: %v", err)
	}
	neighbors, err := target.Neighbors(11)
	if err != nil {
		t.Fatalf("target Neighbors: %v", err)
	}
	if len(neighbors) != 0 {
		t.Fatalf("empty source copied unexpected edges: %#v", neighbors)
	}
}

func TestSetGraphWithErrorConcurrentReaders(t *testing.T) {
	c := &Collection{}
	c.SetGraph(graphWithEdge(t))

	const replacements = 12
	const readers = 4
	readErrs := make(chan error, readers)
	stop := make(chan struct{})
	for i := 0; i < readers; i++ {
		go func() {
			for {
				select {
				case <-stop:
					readErrs <- nil
					return
				default:
				}
				g := c.GetGraph()
				if g == nil {
					readErrs <- errors.New("graph detached during replacement")
					return
				}
				if _, err := g.Neighbors(11); err != nil && !errors.Is(err, graph.ErrGraphClosed) {
					readErrs <- err
					return
				}
			}
		}()
	}

	for i := 0; i < replacements; i++ {
		target, err := NewGraph(GraphConfig{})
		if err != nil {
			close(stop)
			t.Fatalf("NewGraph replacement: %v", err)
		}
		if err := c.SetGraphWithError(target); err != nil {
			close(stop)
			t.Fatalf("replacement %d: %v", i, err)
		}
	}
	close(stop)
	for i := 0; i < readers; i++ {
		if err := <-readErrs; err != nil {
			t.Fatalf("concurrent reader: %v", err)
		}
	}
	if g := c.GetGraph(); g != nil {
		_ = g.Close()
	}
}
