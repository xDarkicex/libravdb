package hnsw

import (
	"testing"
	"unsafe"
)

func TestInMemoryRawVectorStorePutDoesNotAllocate(t *testing.T) {
	store := NewInMemoryRawVectorStoreWithCapacity(4, 256)
	defer store.Close()

	vector := [4]float32{1, 2, 3, 4}
	allocs := testing.AllocsPerRun(100, func() {
		if _, err := store.Put(vector[:]); err != nil {
			t.Fatalf("Put: %v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("off-heap vector Put created %.0f Go heap allocations", allocs)
	}
}

func TestInMemoryRawVectorStoreReusesReleasedPointer(t *testing.T) {
	store := NewInMemoryRawVectorStoreWithCapacity(4, 256)
	defer store.Close()

	first, err := store.Put([]float32{1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	firstVector, err := store.Get(first)
	if err != nil {
		t.Fatal(err)
	}
	firstPtr := unsafe.Pointer(&firstVector[0])
	if err := store.release(first); err != nil {
		t.Fatal(err)
	}

	second, err := store.Put([]float32{5, 6, 7, 8})
	if err != nil {
		t.Fatal(err)
	}
	if second.Slot != first.Slot {
		t.Fatalf("released logical slot was not reused: got %d want %d", second.Slot, first.Slot)
	}
	secondVector, err := store.Get(second)
	if err != nil {
		t.Fatal(err)
	}
	if secondPtr := unsafe.Pointer(&secondVector[0]); secondPtr != firstPtr {
		t.Fatalf("released vector pointer was not reused: got %p want %p", secondPtr, firstPtr)
	}
}
