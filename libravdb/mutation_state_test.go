package libravdb

import (
	"strconv"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/index"
)

func TestMutationStateSerializesSameKey(t *testing.T) {
	collection := &Collection{}
	first := collection.lockMutationID("same-key")
	acquired := make(chan mutationGuard, 1)
	go func() {
		acquired <- collection.lockMutationID("same-key")
	}()

	select {
	case guard := <-acquired:
		guard.unlock()
		t.Fatal("same-key mutation was admitted concurrently")
	case <-time.After(20 * time.Millisecond):
	}

	first.unlock()
	select {
	case guard := <-acquired:
		guard.unlock()
	case <-time.After(time.Second):
		t.Fatal("waiting same-key mutation was not admitted")
	}
	collection.mutationState.Swap(nil).close()
}

func TestMutationStateBatchOverlapDoesNotDeadlock(t *testing.T) {
	collection := &Collection{}
	left := []*index.VectorEntry{{ID: "a"}, {ID: "b"}}
	right := []*index.VectorEntry{{ID: "b"}, {ID: "a"}}
	start := make(chan struct{})
	done := make(chan struct{}, 2)
	for _, entries := range [][]*index.VectorEntry{left, right} {
		entries := entries
		go func() {
			<-start
			guard := collection.lockMutationEntries(entries)
			guard.unlock()
			done <- struct{}{}
		}()
	}
	close(start)
	for range 2 {
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Fatal("overlapping mutation batches deadlocked")
		}
	}
	collection.mutationState.Swap(nil).close()
}

func TestMutationStateConservativelySerializesHashSlotCollision(t *testing.T) {
	collection := &Collection{}
	firstID := "collision-0"
	firstSlot := mutationSlotFor(mutationToken(firstID))
	secondID := ""
	for i := 1; i < 100_000; i++ {
		candidate := "collision-" + strconv.Itoa(i)
		if mutationSlotFor(mutationToken(candidate)) == firstSlot {
			secondID = candidate
			break
		}
	}
	if secondID == "" {
		t.Fatal("failed to find mutation-slot collision")
	}

	first := collection.lockMutationID(firstID)
	acquired := make(chan mutationGuard, 1)
	go func() {
		acquired <- collection.lockMutationID(secondID)
	}()
	select {
	case guard := <-acquired:
		guard.unlock()
		t.Fatal("colliding mutation slot was admitted concurrently")
	case <-time.After(20 * time.Millisecond):
	}
	first.unlock()
	guard := <-acquired
	guard.unlock()
	collection.mutationState.Swap(nil).close()
}

func TestMutationStateSingleKeyHasNoSteadyStateAllocations(t *testing.T) {
	collection := &Collection{}
	warmup := collection.lockMutationID("allocation-test")
	warmup.unlock()
	allocations := testing.AllocsPerRun(1000, func() {
		guard := collection.lockMutationID("allocation-test")
		guard.unlock()
	})
	if allocations != 0 {
		t.Fatalf("single-key mutation state allocated %.2f objects/op", allocations)
	}
	collection.mutationState.Swap(nil).close()
}
