package btree

import (
	"context"
	"testing"
)

func TestDeleteAfterDeserializeFindsRestoredKey(t *testing.T) {
	ctx := context.Background()
	original, err := New(DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer original.Close()
	for _, id := range []string{"todo-1", "todo-2", "todo-3"} {
		if err := original.Insert(ctx, []byte(id), []byte("value")); err != nil {
			t.Fatalf("insert %s: %v", id, err)
		}
	}
	serialized, err := original.SerializeToBytes()
	if err != nil {
		t.Fatal(err)
	}

	restored, err := New(DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer restored.Close()
	if err := restored.DeserializeFromBytes(ctx, serialized); err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"todo-1", "todo-2", "todo-3"} {
		if _, err := restored.Search(ctx, []byte(id)); err != nil {
			t.Fatalf("search restored %s: %v", id, err)
		}
	}
	if err := restored.Delete(ctx, []byte("todo-3")); err != nil {
		t.Fatalf("delete restored todo-3: %v", err)
	}
	if _, err := restored.Search(ctx, []byte("todo-3")); err != errKeyNotFound {
		t.Fatalf("deleted restored todo-3 search error = %v, want errKeyNotFound", err)
	}
	if err := restored.Insert(ctx, []byte("todo-3"), []byte("replacement")); err != nil {
		t.Fatalf("reinsert restored todo-3: %v", err)
	}
}
