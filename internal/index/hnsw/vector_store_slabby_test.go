package hnsw

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"

	"github.com/xDarkicex/libravdb/internal/util"
)

func TestSlabbyRawVectorStoreRoundTrip(t *testing.T) {
	store, err := NewSlabbyRawVectorStore(4, 2)
	if err != nil {
		t.Fatalf("failed to create slabby raw vector store: %v", err)
	}
	defer store.Close()

	vectors := [][]float32{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
		{9, 10, 11, 12},
		{13, 14, 15, 16},
		{17, 18, 19, 20},
	}

	refs := make([]VectorRef, 0, len(vectors))
	for i, vec := range vectors {
		ref, err := store.Put(vec)
		if err != nil {
			t.Fatalf("put vector %d failed: %v", i, err)
		}
		refs = append(refs, ref)

		got, err := store.Get(ref)
		if err != nil {
			t.Fatalf("get vector %d failed: %v", i, err)
		}
		for j := range vec {
			if got[j] != vec[j] {
				t.Fatalf("vector %d mismatch at dim %d: got %f want %f", i, j, got[j], vec[j])
			}
		}
	}

	if len(store.allocators) < 3 {
		t.Fatalf("expected slabby store to grow across segments, got %d allocator(s)", len(store.allocators))
	}

	if err := store.Delete(refs[1]); err != nil {
		t.Fatalf("delete failed: %v", err)
	}
	if _, err := store.Get(refs[1]); err == nil {
		t.Fatal("expected deleted slabby vector lookup to fail")
	}

	if err := store.Reset(); err != nil {
		t.Fatalf("reset failed: %v", err)
	}
	if len(store.allocators) != 0 || len(store.slots) != 0 {
		t.Fatalf("expected reset to clear slabby store, got %d allocators and %d slots", len(store.allocators), len(store.slots))
	}
}

func TestHNSWSlabbyRawVectorStoreSaveLoad(t *testing.T) {
	config := &Config{
		Dimension:      16,
		M:              8,
		EfConstruction: 32,
		EfSearch:       16,
		ML:             1.0,
		Metric:         util.L2Distance,
		RandomSeed:     42,
		RawVectorStore: RawVectorStoreSlabby,
		RawStoreCap:    4,
	}

	index, err := NewHNSW(config)
	if err != nil {
		t.Fatalf("failed to create slabby-backed HNSW index: %v", err)
	}
	defer index.Close()

	vectors := generateTestVectors(32, 16)
	for i, vec := range vectors {
		if err := index.Insert(context.Background(), &VectorEntry{
			ID:       fmt.Sprintf("vec_%d", i),
			Vector:   vec,
			Metadata: map[string]interface{}{"index": i},
		}); err != nil {
			t.Fatalf("insert vector %d failed: %v", i, err)
		}
	}

	results, err := index.Search(context.Background(), vectors[0], 5)
	if err != nil {
		t.Fatalf("pre-save search failed: %v", err)
	}
	if len(results) == 0 || results[0].ID != "vec_0" {
		t.Fatalf("expected vec_0 as nearest result before save, got %#v", results)
	}

	path := filepath.Join(t.TempDir(), "slabby-index.hnsw")
	if err := index.SaveToDisk(context.Background(), path); err != nil {
		t.Fatalf("save failed: %v", err)
	}

	loaded, err := NewHNSW(config)
	if err != nil {
		t.Fatalf("failed to create load target index: %v", err)
	}
	defer loaded.Close()

	if err := loaded.LoadFromDisk(context.Background(), path); err != nil {
		t.Fatalf("load failed: %v", err)
	}

	if _, err := loaded.getNodeVector(loaded.nodes[0]); err != nil {
		t.Fatalf("failed to access loaded slabby-backed node vector: %v", err)
	}

	loadedResults, err := loaded.Search(context.Background(), vectors[0], 5)
	if err != nil {
		t.Fatalf("post-load search failed: %v", err)
	}
	if len(loadedResults) == 0 || loadedResults[0].ID != "vec_0" {
		t.Fatalf("expected vec_0 as nearest result after load, got %#v", loadedResults)
	}
}
