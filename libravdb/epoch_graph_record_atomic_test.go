package libravdb

import (
	"context"
	"testing"
)

func TestEpochRecordAndGraphReplacementIsAtomicAndCleansStaleEdges(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:epoch-graph-record-atomic"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	graph, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer graph.Close()
	const follows uint8 = 231
	const friends uint8 = 232
	if !RegisterEdgeKind("EPOCH_ATOMIC_FOLLOWS", follows) || !RegisterEdgeKind("EPOCH_ATOMIC_FRIENDS", friends) {
		t.Fatal("register edge kinds")
	}
	people, err := db.CreateCollection(ctx, "epoch_people", WithDimension(1), WithGraph(graph), WithMetadataSchema(MetadataSchema{
		"name": StringField,
	}))
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"alice", "bob", "carol"} {
		if err := people.Insert(ctx, id, []float32{0}, map[string]interface{}{"name": id}); err != nil {
			t.Fatalf("insert %s: %v", id, err)
		}
	}
	node := func(id string) uint64 {
		value, nodeErr := db.GetNodeID(ctx, "epoch_people", id)
		if nodeErr != nil {
			t.Fatalf("GetNodeID(%s): %v", id, nodeErr)
		}
		return value
	}
	initial := graph.BeginTxn()
	if err := initial.AddEdge(node("alice"), node("bob"), 1, follows); err != nil {
		t.Fatal(err)
	}
	if err := initial.AddEdge(node("alice"), node("carol"), 1, friends); err != nil {
		t.Fatal(err)
	}
	if err := initial.Commit(ctx); err != nil {
		t.Fatal(err)
	}

	epoch, err := db.BeginEpochTx(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if err := epoch.Update(ctx, "epoch_people", "alice", []float32{0}, map[string]interface{}{"name": "Alice Updated"}); err != nil {
		t.Fatal(err)
	}
	if err := epoch.ReplaceGraphEdgesByID(ctx, "epoch_people", "alice", "EPOCH_ATOMIC_FOLLOWS", []GraphEdgeMutation{
		{TargetID: "carol", EdgeType: "EPOCH_ATOMIC_FOLLOWS", Weight: 2},
	}); err != nil {
		t.Fatal(err)
	}
	if err := epoch.Commit(ctx); err != nil {
		t.Fatal(err)
	}

	updated, err := people.Get(ctx, "alice")
	if err != nil || updated.Metadata["name"] != "Alice Updated" {
		t.Fatalf("record update=%#v err=%v", updated, err)
	}
	edges, err := graph.Neighbors(node("alice"))
	if err != nil {
		t.Fatal(err)
	}
	if len(edges) != 2 {
		t.Fatalf("edge count=%d, want replaced follows plus preserved friend: %+v", len(edges), edges)
	}
	followsTargets := map[uint64]bool{}
	for _, edge := range edges {
		if edge.GetKind() == follows {
			followsTargets[edge.Target] = true
		}
	}
	if followsTargets[node("bob")] || !followsTargets[node("carol")] {
		t.Fatalf("stale/replaced follows edges=%v", followsTargets)
	}

	rollback, err := db.BeginEpochTx(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if err := rollback.ReplaceGraphEdgesByID(ctx, "epoch_people", "alice", "EPOCH_ATOMIC_FOLLOWS", nil); err != nil {
		t.Fatal(err)
	}
	if err := rollback.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	edges, err = graph.Neighbors(node("alice"))
	if err != nil || len(edges) != 2 {
		t.Fatalf("rollback changed graph edges=%+v err=%v", edges, err)
	}

	deleteEpoch, err := db.BeginEpochTx(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if err := deleteEpoch.Delete(ctx, "epoch_people", "carol"); err != nil {
		t.Fatal(err)
	}
	if err := deleteEpoch.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	edges, err = graph.Neighbors(node("alice"))
	if err != nil {
		t.Fatal(err)
	}
	for _, edge := range edges {
		if edge.Target == node("carol") {
			t.Fatalf("deleted node edge survived: %+v", edges)
		}
	}
}
