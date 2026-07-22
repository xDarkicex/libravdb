package index

import (
	"context"
	"testing"

	"github.com/xDarkicex/libravdb/internal/util"
)

func TestNewFlatUsesGenerationCore(t *testing.T) {
	ctx := context.Background()
	idx, err := NewFlat(&FlatConfig{Dimension: 2, Metric: util.L2Distance})
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	wrapper, ok := idx.(*flatWrapper)
	if !ok || wrapper.core == nil {
		t.Fatalf("NewFlat did not return a core-backed wrapper: %T", idx)
	}
	if err := idx.BatchInsert(ctx, []*VectorEntry{
		{ID: "left", Ordinal: 1, Version: 1, Vector: []float32{1, 0}},
		{ID: "right", Ordinal: 2, Version: 1, Vector: []float32{0, 1}},
	}); err != nil {
		t.Fatal(err)
	}
	if got := idx.Size(); got != 2 {
		t.Fatalf("size after batch = %d, want 2", got)
	}
	results, err := idx.Search(ctx, []float32{1, 0}, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 || results[0].ID != "left" || results[0].Ordinal != 1 {
		t.Fatalf("unexpected results: %#v", results)
	}
	if err := idx.Insert(ctx, &VectorEntry{ID: "left", Ordinal: 1, Version: 2, Vector: []float32{0, 2}}); err != nil {
		t.Fatal(err)
	}
	results, err = idx.Search(ctx, []float32{0, 2}, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 || results[0].ID != "left" || results[0].Version != 2 {
		t.Fatalf("update was not published through core: %#v", results)
	}
	if err := idx.Delete(ctx, "left"); err != nil {
		t.Fatal(err)
	}
	if got := idx.Size(); got != 1 {
		t.Fatalf("size after delete = %d, want 1", got)
	}

	snapshot, err := idx.SerializeToBytes()
	if err != nil {
		t.Fatal(err)
	}
	restored, err := NewFlat(&FlatConfig{Dimension: 2, Metric: util.L2Distance})
	if err != nil {
		t.Fatal(err)
	}
	defer restored.Close()
	if err := restored.DeserializeFromBytes(ctx, snapshot); err != nil {
		t.Fatal(err)
	}
	restoredResults, err := restored.Search(ctx, []float32{0, 1}, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(restoredResults) != 1 || restoredResults[0].ID != "right" || restoredResults[0].Ordinal != 2 {
		t.Fatalf("restored core results = %#v", restoredResults)
	}
}
