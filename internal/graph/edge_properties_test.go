package graph

import (
	"context"
	"testing"
)

func TestEdgePropertiesFollowNodeOwnedPagesAndTraversal(t *testing.T) {
	gi, err := NewGraph(DefaultGraphConfig())
	if err != nil {
		t.Fatal(err)
	}
	g := gi.(*graphStore)
	defer g.Close()

	txn := g.BeginTxn()
	if err := txn.AddEdgeWithProperties(1, 2, 1, 7, map[string]interface{}{
		"cost":       4.2,
		"confidence": 0.98,
		"kind":       "strong",
	}); err != nil {
		t.Fatal(err)
	}
	if err := txn.Commit(context.Background()); err != nil {
		t.Fatal(err)
	}

	views, err := g.NeighborsWithProperties(1)
	if err != nil {
		t.Fatal(err)
	}
	if len(views) != 1 {
		t.Fatalf("expected one edge, got %d", len(views))
	}
	jsonBytes, err := EdgePropertyJSON(views[0].Properties)
	if err != nil {
		t.Fatal(err)
	}
	if string(jsonBytes) != `{"confidence":0.98,"cost":4.2,"kind":"strong"}` {
		t.Fatalf("unexpected canonical properties: %s", jsonBytes)
	}

	predicate := EdgePredicate{Nodes: []EdgePredicateNode{
		{Op: EdgePredicateComparison, Property: EdgePropertyArbitrary, Name: "cost", Compare: WeightLess, Value: EdgePropertyValue{Kind: EdgePropertyNumber, Number: 5}},
	}, Root: 0}
	if !predicate.MatchesWithProperties(views[0].Edge, views[0].Properties) {
		t.Fatal("arbitrary numeric property predicate did not match")
	}

	bitset, err := g.GetBitset()
	if err != nil {
		t.Fatal(err)
	}
	defer g.PutBitset(bitset)
	frontier, err := g.GetFrontierBuf()
	if err != nil {
		t.Fatal(err)
	}
	defer g.PutFrontierBuf(frontier)
	if err := g.BFSPattern(1, []EdgePlan{{Dir: 1, Min: 1, Max: 1, Predicate: predicate}}, 1, func(nodeID uint64, band int, step int) bool {
		return true
	}, bitset, frontier); err != nil {
		t.Fatal(err)
	}
}
