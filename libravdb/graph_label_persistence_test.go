package libravdb

import (
	"context"
	"testing"
)

func TestGraphVertexLabelPersistsAcrossReopen(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/labels.libravdb"
	db, err := Open(WithStoragePath(path))
	if err != nil {
		t.Fatal(err)
	}
	g, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	col, err := db.CreateCollection(ctx, "docs", WithDimension(2), WithGraph(g))
	if err != nil {
		t.Fatal(err)
	}
	if err := col.Insert(ctx, "d1", []float32{1, 0}, nil); err != nil {
		t.Fatal(err)
	}
	nodeID, err := db.GetNodeID(ctx, "docs", "d1")
	if err != nil {
		t.Fatal(err)
	}
	g.RegisterVertexLabel(nodeID, "Manual")
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	db, err = Open(WithStoragePath(path))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Drop(ctx)
	col, err = db.GetCollection("docs")
	if err != nil {
		t.Fatal(err)
	}
	recovered, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	col.SetGraph(recovered)
	nodeID, err = db.GetNodeID(ctx, "docs", "d1")
	if err != nil {
		t.Fatal(err)
	}
	for _, got := range recovered.GetLabelNodes("Manual") {
		if got == nodeID {
			return
		}
	}
	t.Fatalf("label Manual for node %d was not restored", nodeID)
}
