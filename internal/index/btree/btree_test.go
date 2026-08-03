package btree

import (
	"context"
	"fmt"
	"testing"
)

func TestBTree_InsertAndSearch(t *testing.T) {
	tree, err := New(DefaultConfig())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer tree.Close()

	ctx := context.Background()

	// Insert keys
	pairs := map[string]string{
		"apple":  "red",
		"banana": "yellow",
		"cherry": "red",
		"date":   "brown",
		"elder":  "purple",
	}
	for k, v := range pairs {
		if err := tree.Insert(ctx, []byte(k), []byte(v)); err != nil {
			t.Fatalf("Insert(%q): %v", k, err)
		}
	}

	// Search and verify
	for k, want := range pairs {
		got, err := tree.Search(ctx, []byte(k))
		if err != nil {
			t.Errorf("Search(%q): %v", k, err)
			continue
		}
		if string(got) != want {
			t.Errorf("Search(%q) = %q, want %q", k, got, want)
		}
	}

	// Search for non-existent key
	_, err = tree.Search(ctx, []byte("zebra"))
	if err != errKeyNotFound {
		t.Errorf("Search(zebra) error = %v, want errKeyNotFound", err)
	}
}

func TestBTree_InsertMany(t *testing.T) {
	tree, err := New(DefaultConfig())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer tree.Close()

	ctx := context.Background()
	N := 1000

	for i := 0; i < N; i++ {
		key := []byte(fmt.Sprintf("key-%05d", i))
		val := []byte(fmt.Sprintf("value-%05d", i))
		if err := tree.Insert(ctx, key, val); err != nil {
			t.Fatalf("Insert(%q) at %d: %v", key, i, err)
		}
	}

	if tree.Len() != N {
		t.Errorf("Len() = %d, want %d", tree.Len(), N)
	}

	// Verify all keys
	for i := 0; i < N; i++ {
		key := []byte(fmt.Sprintf("key-%05d", i))
		want := []byte(fmt.Sprintf("value-%05d", i))
		got, err := tree.Search(ctx, key)
		if err != nil {
			t.Errorf("Search(%q): %v", key, err)
			continue
		}
		if string(got) != string(want) {
			t.Errorf("Search(%q) = %q, want %q", key, got, want)
		}
	}
}

func TestBTree_SplitRoot(t *testing.T) {
	tree, err := New(DefaultConfig())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer tree.Close()

	ctx := context.Background()

	// Insert enough keys to force root split.
	// Each leaf page holds ~200 keys (4032 - 32 = 4000 bytes / ~20 bytes per kv).
	// 500 keys should trigger at least one split.
	for i := 0; i < 500; i++ {
		key := []byte(fmt.Sprintf("k-%04d", i))
		val := []byte(fmt.Sprintf("v-%04d", i))
		if err := tree.Insert(ctx, key, val); err != nil {
			t.Fatalf("Insert(%q) at %d: %v", key, i, err)
		}
	}

	// Verify all keys still findable
	for i := 0; i < 500; i++ {
		key := []byte(fmt.Sprintf("k-%04d", i))
		_, err := tree.Search(ctx, key)
		if err != nil {
			t.Errorf("Search(%q) after split: %v", key, err)
		}
	}
}

func TestBTree_BatchInsert(t *testing.T) {
	tree, _ := New(DefaultConfig())
	defer tree.Close()
	ctx := context.Background()

	N := 50000
	pairs := make([]KVPair, N)
	for i := 0; i < N; i++ {
		pairs[i] = KVPair{
			Key:   []byte(fmt.Sprintf("key-%08d", i)),
			Value: []byte(fmt.Sprintf("val-%08d", i)),
		}
	}

	if err := tree.BatchInsert(ctx, pairs); err != nil {
		t.Fatalf("BatchInsert: %v", err)
	}

	if tree.Len() != N {
		t.Fatalf("Len() = %d, want %d", tree.Len(), N)
	}

	// Spot-check at intervals
	for i := 0; i < N; i += 5000 {
		key := []byte(fmt.Sprintf("key-%08d", i))
		val, err := tree.Search(ctx, key)
		if err != nil {
			t.Errorf("Search(%q): %v", key, err)
			continue
		}
		want := fmt.Sprintf("val-%08d", i)
		if string(val) != want {
			t.Errorf("Search(%q) = %q, want %q", key, val, want)
		}
	}

	// Cursor forward
	c := tree.SeekFirst()
	count := 0
	prev := ""
	for c.Valid() {
		key := string(c.Key())
		if count > 0 && key <= prev {
			t.Fatalf("order violation at %d: %q <= %q", count, key, prev)
		}
		prev = key
		count++
		c.Next()
	}
	if count != N {
		t.Errorf("cursor forward: %d, want %d", count, N)
	}

}

func TestMerge_AfterDelete(t *testing.T) {
	tree, _ := New(DefaultConfig())
	defer tree.Close()
	ctx := context.Background()

	// Insert 300 keys to create a multi-page tree
	for i := 0; i < 300; i++ {
		tree.Insert(ctx, []byte(fmt.Sprintf("k-%04d", i)), []byte(fmt.Sprintf("v-%04d", i)))
	}
	initialPages := len(tree.pageReg.snapshotIDs())
	t.Logf("After insert: %d keys, %d pages", tree.Len(), initialPages)

	// Delete most keys
	for i := 0; i < 250; i++ {
		if err := tree.Delete(ctx, []byte(fmt.Sprintf("k-%04d", i))); err != nil {
			t.Logf("Delete k-%04d: %v", i, err)
		}
	}

	afterPages := len(tree.pageReg.snapshotIDs())
	t.Logf("After delete: %d keys, %d pages", tree.Len(), afterPages)

	// Remaining keys should still be findable
	for i := 250; i < 300; i++ {
		key := []byte(fmt.Sprintf("k-%04d", i))
		_, err := tree.Search(ctx, key)
		if err != nil {
			t.Errorf("Search(%q): %v", key, err)
		}
	}
}

func TestMerge_CollapseRoot(t *testing.T) {
	tree, _ := New(DefaultConfig())
	defer tree.Close()
	ctx := context.Background()

	// Insert enough to split root
	for i := 0; i < 300; i++ {
		tree.Insert(ctx, []byte(fmt.Sprintf("k-%04d", i)), []byte(fmt.Sprintf("v-%04d", i)))
	}
	rootID := tree.rootID.Load()
	t.Logf("Root before delete: leaf=%v count=%d",
		tree.pageReg.get(rootID).IsLeaf(), tree.pageReg.get(rootID).Header.Count)

	// Delete enough to collapse back to leaf
	for i := 0; i < 290; i++ {
		tree.Delete(ctx, []byte(fmt.Sprintf("k-%04d", i)))
	}

	rootID = tree.rootID.Load()
	t.Logf("Root after delete: leaf=%v count=%d",
		tree.pageReg.get(rootID).IsLeaf(), tree.pageReg.get(rootID).Header.Count)

	// Remaining keys should be findable
	for i := 290; i < 300; i++ {
		key := []byte(fmt.Sprintf("k-%04d", i))
		_, err := tree.Search(ctx, key)
		if err != nil {
			t.Errorf("Search(%q): %v", key, err)
		}
	}
}
