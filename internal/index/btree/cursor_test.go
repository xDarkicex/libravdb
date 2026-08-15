package btree

import (
	"context"
	"fmt"
	"math/rand"
	"strings"
	"testing"
)

func TestCursor_SeekNext(t *testing.T) {
	tree, _ := New(DefaultConfig())
	defer tree.Close()
	ctx := context.Background()

	keys := []string{"b", "d", "f", "h", "j"}
	for _, k := range keys {
		tree.Insert(ctx, []byte(k), []byte("v-"+k))
	}

	c := tree.Seek([]byte("c"))
	if !c.Valid() {
		t.Fatal("expected valid cursor at 'd'")
	}
	if string(c.Key()) != "d" {
		t.Fatalf("Seek(c) = %q, want 'd'", c.Key())
	}

	// Next should be 'f'
	c.Next()
	if string(c.Key()) != "f" {
		t.Fatalf("Next = %q, want 'f'", c.Key())
	}

	// Next → Next → end
	c.Next() // h
	c.Next() // j
	if string(c.Key()) != "j" {
		t.Fatalf("Next×2 = %q, want 'j'", c.Key())
	}
	if c.Next() {
		t.Fatal("expected end after 'j'")
	}
}

func TestCursor_SeekFirst(t *testing.T) {
	tree, _ := New(DefaultConfig())
	defer tree.Close()
	ctx := context.Background()

	for i := 0; i < 10; i++ {
		tree.Insert(ctx, []byte(fmt.Sprintf("k-%02d", i)), []byte(fmt.Sprintf("v-%02d", i)))
	}

	c := tree.SeekFirst()
	if !c.Valid() {
		t.Fatal("expected valid cursor")
	}
	if string(c.Key()) != "k-00" {
		t.Fatalf("first key = %q", c.Key())
	}

	count := 1
	for c.Next() {
		count++
	}
	if count != 10 {
		t.Fatalf("iterated %d keys, want 10", count)
	}
}

func TestCursor_IterateAll(t *testing.T) {
	tree, _ := New(DefaultConfig())
	defer tree.Close()
	ctx := context.Background()

	N := 500
	for i := 0; i < N; i++ {
		tree.Insert(ctx, []byte(fmt.Sprintf("key-%05d", i)), []byte(fmt.Sprintf("val-%05d", i)))
	}

	c := tree.SeekFirst()
	count := 0
	var prev []byte
	for c.Valid() {
		key := cloneBytes(c.Key())
		if prev != nil && string(key) <= string(prev) {
			t.Fatalf("key order violation at %d: %q <= %q", count, key, prev)
		}
		prev = key
		count++
		c.Next()
	}
	if count != N {
		t.Fatalf("iterated %d keys, want %d", count, N)
	}
}

func TestCursor_IterateAllVariableLengthKeys(t *testing.T) {
	tree, err := New(DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer tree.Close()

	const want = 500
	keys := make([]string, 0, want)
	for i := 0; i < want; i++ {
		keys = append(keys, fmt.Sprintf("city-%d-%s", i, strings.Repeat("x", i%17)))
	}
	rand.New(rand.NewSource(42)).Shuffle(len(keys), func(i, j int) { keys[i], keys[j] = keys[j], keys[i] })
	for _, key := range keys {
		if err := tree.Insert(context.Background(), []byte(key), []byte("value")); err != nil {
			t.Fatalf("insert %q: %v", key, err)
		}
	}

	cursor := tree.SeekFirst()
	count := 0
	var previous string
	for cursor.Valid() {
		key := string(cursor.Key())
		if previous != "" && key <= previous {
			t.Fatalf("key order violation at %d: %q <= %q", count, key, previous)
		}
		previous = key
		count++
		cursor.Next()
	}
	if count != want {
		t.Fatalf("iterated %d keys, want %d", count, want)
	}
}

func TestCursor_SeekEmpty(t *testing.T) {
	tree, _ := New(DefaultConfig())
	defer tree.Close()

	c := tree.Seek([]byte("anything"))
	if c.Valid() {
		t.Fatal("expected invalid cursor on empty tree")
	}
}

func TestCursor_Prev(t *testing.T) {
	tree, _ := New(DefaultConfig())
	defer tree.Close()
	ctx := context.Background()

	keys := []string{"a", "c", "e", "g", "i"}
	for _, k := range keys {
		tree.Insert(ctx, []byte(k), []byte("v-"+k))
	}

	// SeekLast → "i"
	c := tree.SeekLast()
	if !c.Valid() {
		t.Fatal("expected valid cursor at 'i'")
	}
	if string(c.Key()) != "i" {
		t.Fatalf("SeekLast = %q, want 'i'", c.Key())
	}

	// Prev → "g"
	c.Prev()
	if string(c.Key()) != "g" {
		t.Fatalf("Prev = %q, want 'g'", c.Key())
	}

	// Prev×3 → "a"
	c.Prev() // e
	c.Prev() // c
	c.Prev() // a
	if string(c.Key()) != "a" {
		t.Fatalf("Prev×3 = %q, want 'a'", c.Key())
	}

	// Prev past start → invalid
	if c.Prev() {
		t.Fatal("expected end at start")
	}
}

func TestCursor_BackAndForth(t *testing.T) {
	tree, _ := New(DefaultConfig())
	defer tree.Close()
	ctx := context.Background()

	N := 200
	for i := 0; i < N; i++ {
		tree.Insert(ctx, []byte(fmt.Sprintf("key-%03d", i)), []byte(fmt.Sprintf("val-%03d", i)))
	}

	// Forward all the way
	c := tree.SeekFirst()
	forward := make([]string, 0, N)
	for c.Valid() {
		forward = append(forward, string(c.Key()))
		c.Next()
	}
	if len(forward) != N {
		t.Fatalf("forward count = %d, want %d", len(forward), N)
	}

	// Backward all the way
	c = tree.SeekLast()
	backward := make([]string, 0, N)
	for c.Valid() {
		backward = append(backward, string(c.Key()))
		c.Prev()
	}
	if len(backward) != N {
		t.Fatalf("backward count = %d, want %d", len(backward), N)
	}

	// Backward should be reverse of forward
	for i := 0; i < N; i++ {
		if forward[i] != backward[N-1-i] {
			t.Fatalf("forward[%d]=%q != backward[%d]=%q", i, forward[i], N-1-i, backward[N-1-i])
		}
	}
}
