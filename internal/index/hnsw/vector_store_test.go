package hnsw

import "testing"

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
