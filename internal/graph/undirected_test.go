package graph

import "testing"

func TestUndirectedEdgeUsesOneStoredEdgeAndTraversesEitherWay(t *testing.T) {
	store, err := NewGraph(testConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	const kind uint8 = 247
	store.SetEdgeKindDirection(kind, true)
	txn := store.BeginTxn()
	if err := txn.AddEdge(1, 2, 1, kind); err != nil {
		t.Fatal(err)
	}
	if err := txn.Commit(nil); err != nil {
		t.Fatal(err)
	}

	for _, check := range []struct {
		name string
		node uint64
		want uint64
	}{
		{name: "outbound endpoint", node: 1, want: 2},
		{name: "reverse endpoint", node: 2, want: 1},
	} {
		edges, err := store.Neighbors(check.node)
		if err != nil {
			t.Fatalf("%s: %v", check.name, err)
		}
		if len(edges) != 1 || edges[0].Target != check.want || edges[0].GetKind() != kind {
			t.Fatalf("%s: %#v, want one edge to %d", check.name, edges, check.want)
		}
	}

	count := 0
	store.ForEachEdge(func(src, tgt uint64, edge Edge) bool {
		count++
		if src != 1 || tgt != 2 || edge.GetKind() != kind {
			t.Errorf("stored edge=(%d,%d,%d), want canonical (1,2,%d)", src, tgt, edge.GetKind(), kind)
		}
		return true
	})
	if count != 1 {
		t.Fatalf("stored edge count=%d, want 1", count)
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
	seen := make(map[uint64]bool)
	if err := store.BFSPattern(2, []EdgePlan{{Dir: 1, KindSet: NewKindSet(kind), Min: 1, Max: 1}}, 1, func(node uint64, _, _ int) bool {
		seen[node] = true
		return true
	}, bitset, frontier); err != nil {
		t.Fatal(err)
	}
	if !seen[1] {
		t.Fatalf("directed arrow from reverse endpoint did not traverse undirected edge: %#v", seen)
	}

	removeTxn := store.BeginTxn()
	if err := removeTxn.RemoveEdge(2, 1, kind); err != nil {
		t.Fatalf("reverse-order remove: %v", err)
	}
	if err := removeTxn.Commit(nil); err != nil {
		t.Fatal(err)
	}
	edges, err := store.Neighbors(1)
	if err != nil {
		t.Fatal(err)
	}
	if len(edges) != 0 {
		t.Fatalf("neighbors after reverse-order remove=%#v, want empty", edges)
	}
}
