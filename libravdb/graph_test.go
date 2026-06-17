package libravdb

import (
	"testing"
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
