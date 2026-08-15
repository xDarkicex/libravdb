package graph

import "testing"

func TestBFSPatternWeightFilterPushdown(t *testing.T) {
	store, err := NewGraph(testConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	txn := &Txn{ID: 1}
	if err := store.AddEdge(txn, 1, 2, 0.2, 7); err != nil {
		t.Fatal(err)
	}
	if err := store.AddEdge(txn, 1, 3, 0.9, 7); err != nil {
		t.Fatal(err)
	}
	bitset, err := store.GetBitset()
	if err != nil {
		t.Fatal(err)
	}
	defer store.PutBitset(bitset)
	frontier, err := store.GetFrontierBuf()
	if err != nil {
		t.Fatal(err)
	}
	defer store.PutFrontierBuf(frontier)

	var visited []uint64
	err = store.BFSPattern(1, []EdgePlan{{
		Dir: 1, Min: 1, Max: 1,
		KindSet: NewKindSet(7),
		Weight:  WeightFilter{Enabled: true, Op: WeightGreater, Value: 0.8},
	}}, 1, func(nodeID uint64, band, step int) bool {
		if step > 0 {
			visited = append(visited, nodeID)
		}
		return true
	}, bitset, frontier)
	if err != nil {
		t.Fatal(err)
	}
	if len(visited) != 1 || visited[0] != 3 {
		t.Fatalf("visited=%v, want only node 3", visited)
	}
}
