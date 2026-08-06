package graph

import (
	"testing"
	"time"

	"github.com/leanovate/gopter"
	"github.com/leanovate/gopter/gen"
	"github.com/leanovate/gopter/prop"
)

func TestMetrics_PageRankMaintenanceStatus(t *testing.T) {
	store, err := NewGraph(testConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	initial := store.Stats()
	if initial.MutationGeneration != 0 || initial.PageRankAvailable || initial.PageRankStale {
		t.Fatalf("initial maintenance status: %+v", initial)
	}
	txn := store.BeginTxn()
	if err := txn.AddEdge(1, 2, 1, 1); err != nil {
		t.Fatal(err)
	}
	if err := txn.Commit(nil); err != nil {
		t.Fatal(err)
	}
	afterMutation := store.Stats()
	if afterMutation.MutationGeneration != 1 {
		t.Fatalf("mutation generation=%d, want 1", afterMutation.MutationGeneration)
	}
	store.RecordPageRankPublication(42, 15*time.Millisecond)
	published := store.Stats()
	if !published.PageRankAvailable || published.PageRankStale || published.LastPageRankGeneration != 1 || published.LastPageRankLSN != 42 || published.PageRankDuration != 15*time.Millisecond {
		t.Fatalf("published maintenance status: %+v", published)
	}
	txn = store.BeginTxn()
	if err := txn.AddEdge(3, 2, 1, 1); err != nil {
		t.Fatal(err)
	}
	if err := txn.Commit(nil); err != nil {
		t.Fatal(err)
	}
	stale := store.Stats()
	if !stale.PageRankAvailable || !stale.PageRankStale || stale.MutationGeneration != 2 {
		t.Fatalf("stale maintenance status: %+v", stale)
	}
}

func TestTxnRollbackDiscardsStagedEdges(t *testing.T) {
	store, err := NewGraph(DefaultGraphConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	txn := store.BeginTxn()
	if err := txn.AddEdge(1, 2, 1, 1); err != nil {
		t.Fatal(err)
	}
	if err := txn.Rollback(); err != nil {
		t.Fatal(err)
	}
	if got, err := store.Neighbors(1); err != nil {
		t.Fatal(err)
	} else if len(got) != 0 {
		t.Fatalf("rollback published %d edges", len(got))
	}
	if err := txn.AddEdge(2, 3, 1, 1); err == nil {
		t.Fatal("expected closed transaction error")
	}
}

func TestTxnNeighborsOverlaySeesStagedEdges(t *testing.T) {
	store, err := NewGraph(DefaultGraphConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	txn := store.BeginTxn()
	if err := txn.AddEdge(10, 20, 1, 3); err != nil {
		t.Fatal(err)
	}
	got, err := txn.NeighborsOverlay(10)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Target != 20 || got[0].GetKind() != 3 {
		t.Fatalf("unexpected overlay neighbors: %#v", got)
	}
	if live, err := store.Neighbors(10); err != nil {
		t.Fatal(err)
	} else if len(live) != 0 {
		t.Fatalf("overlay leaked into live graph: %#v", live)
	}
	_ = txn.Rollback()
}

func TestMetrics_Correctness(t *testing.T) {
	// Property 18: Metric Counter Correctness
	properties := gopter.NewProperties(nil)

	properties.Property("Each operation increments its counter exactly", prop.ForAll(
		func(edges uint16) bool {
			cfg := testConfig()
			store, err := NewGraph(cfg)
			if err != nil {
				return false
			}
			defer store.Close()

			txn := &Txn{ID: 1}
			for i := uint16(0); i < edges; i++ {
				store.AddEdge(txn, 1, uint64(i+2), 1.0, 0)
			}

			stats := store.Stats()

			if stats.EdgesAdded != uint64(edges) {
				return false
			}

			if edges > 0 && stats.PagesAllocated == 0 {
				return false
			}

			return true
		},
		gen.UInt16Range(1, 300),
	))

	properties.TestingRun(t)
}
