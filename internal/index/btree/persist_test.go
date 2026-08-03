package btree

import (
	"context"
	"fmt"
	"os"
	"testing"
)

func TestPersist_RoundTrip(t *testing.T) {
	tree, _ := New(DefaultConfig())
	defer tree.Close()
	ctx := context.Background()

	N := 500
	for i := 0; i < N; i++ {
		tree.Insert(ctx, []byte(fmt.Sprintf("key-%05d", i)), []byte(fmt.Sprintf("val-%05d", i)))
	}

	data, err := tree.SerializeToBytes()
	if err != nil {
		t.Fatalf("SerializeToBytes: %v", err)
	}

	// Create a fresh tree and deserialize
	tree2, _ := New(DefaultConfig())
	defer tree2.Close()

	if err := tree2.DeserializeFromBytes(ctx, data); err != nil {
		t.Fatalf("DeserializeFromBytes: %v", err)
	}

	if tree2.Len() != N {
		t.Fatalf("Len() = %d, want %d", tree2.Len(), N)
	}

	// Verify all keys found
	for i := 0; i < N; i++ {
		key := []byte(fmt.Sprintf("key-%05d", i))
		val, err := tree2.Search(ctx, key)
		if err != nil {
			t.Errorf("Search(%q): %v", key, err)
			continue
		}
		want := fmt.Sprintf("val-%05d", i)
		if string(val) != want {
			t.Errorf("Search(%q) = %q, want %q", key, val, want)
		}
	}

	// Cursor iteration
	c := tree2.SeekFirst()
	count := 0
	for c.Valid() {
		count++
		c.Next()
	}
	if count != N {
		t.Errorf("cursor iterated %d keys, want %d", count, N)
	}
}

func TestPersist_Checksum(t *testing.T) {
	tree, _ := New(DefaultConfig())
	defer tree.Close()
	ctx := context.Background()

	tree.Insert(ctx, []byte("hello"), []byte("world"))

	data, _ := tree.SerializeToBytes()

	// Corrupt the last byte
	corrupted := make([]byte, len(data))
	copy(corrupted, data)
	corrupted[len(corrupted)-2] ^= 0xFF

	tree2, _ := New(DefaultConfig())
	defer tree2.Close()

	err := tree2.DeserializeFromBytes(ctx, corrupted)
	if err == nil {
		t.Fatal("expected checksum error on corrupted data")
	}
	t.Logf("corruption detected: %v", err)
}

func TestPersist_SaveLoadDisk(t *testing.T) {
	tree, _ := New(DefaultConfig())
	defer tree.Close()
	ctx := context.Background()

	for i := 0; i < 100; i++ {
		tree.Insert(ctx, []byte(fmt.Sprintf("k-%02d", i)), []byte(fmt.Sprintf("v-%02d", i)))
	}

	path := t.TempDir() + "/btree.snap"
	if err := tree.SaveToDisk(ctx, path); err != nil {
		t.Fatalf("SaveToDisk: %v", err)
	}

	// Verify file exists
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("file not created: %v", err)
	}

	tree2, _ := New(DefaultConfig())
	defer tree2.Close()

	if err := tree2.LoadFromDisk(ctx, path); err != nil {
		t.Fatalf("LoadFromDisk: %v", err)
	}

	if tree2.Len() != 100 {
		t.Fatalf("Len() = %d, want 100", tree2.Len())
	}
}
