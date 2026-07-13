package singlefile

import (
	"errors"
	"sync"
	"testing"
	"unsafe"

	"github.com/xDarkicex/libravdb/internal/storage"
)

func TestWALRequestRecordIsOneCacheLine(t *testing.T) {
	if size := unsafe.Sizeof(walRequestRecord{}); size != 64 {
		t.Fatalf("WAL request record size = %d, want 64", size)
	}
}

func TestWALRequestPoolCompletionAndReuse(t *testing.T) {
	pool, err := newWALRequestPool()
	if err != nil {
		t.Fatal(err)
	}
	defer pool.close()

	first := pool.acquire(3)
	pool.complete(first, storage.DurableRange{FirstLSN: 40, CommitLSN: 42}, nil)
	durable, err := pool.waitFor(first)
	if err != nil || durable.FirstLSN != 40 || durable.CommitLSN != 42 {
		t.Fatalf("first completion = (%+v, %v), want ({40 42}, nil)", durable, err)
	}

	second := pool.acquire(1)
	if second.index != first.index {
		t.Fatalf("request slot was not immediately reusable: first=%d second=%d", first.index, second.index)
	}
	if second.generation == first.generation {
		t.Fatal("reused request slot retained its generation")
	}
	pool.complete(second, storage.DurableRange{FirstLSN: 82, CommitLSN: 84}, nil)
	durable, err = pool.waitFor(second)
	if err != nil || durable.FirstLSN != 82 || durable.CommitLSN != 84 {
		t.Fatalf("second completion = (%+v, %v), want ({82 84}, nil)", durable, err)
	}
}

func TestWALRequestPoolPreservesCompletionError(t *testing.T) {
	pool, err := newWALRequestPool()
	if err != nil {
		t.Fatal(err)
	}
	defer pool.close()

	want := errors.New("sync failed")
	request := pool.acquire(1)
	pool.complete(request, storage.DurableRange{}, want)
	_, got := pool.waitFor(request)
	if !errors.Is(got, want) {
		t.Fatalf("completion error = %v, want %v", got, want)
	}
}

func TestWALRequestPoolWakesGroupCompletion(t *testing.T) {
	pool, err := newWALRequestPool()
	if err != nil {
		t.Fatal(err)
	}
	defer pool.close()

	const count = 32
	requests := make([]walRequestHandle, count)
	for i := range requests {
		requests[i] = pool.acquire(1)
	}

	var wg sync.WaitGroup
	errs := make(chan error, count)
	for _, request := range requests {
		request := request
		wg.Add(1)
		go func() {
			defer wg.Done()
			durable, err := pool.waitFor(request)
			if err != nil {
				errs <- err
				return
			}
			if durable.CommitLSN != 99 {
				errs <- errors.New("wrong durable LSN")
			}
		}()
	}
	for _, request := range requests {
		pool.complete(request, storage.DurableRange{FirstLSN: 97, CommitLSN: 99}, nil)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
}

func TestWALRequestPoolSteadyStateDoesNotAllocate(t *testing.T) {
	pool, err := newWALRequestPool()
	if err != nil {
		t.Fatal(err)
	}
	defer pool.close()

	allocations := testing.AllocsPerRun(1000, func() {
		request := pool.acquire(1)
		pool.complete(request, storage.DurableRange{FirstLSN: 1, CommitLSN: 1}, nil)
		_, _ = pool.waitFor(request)
	})
	if allocations != 0 {
		t.Fatalf("WAL request completion allocated %.2f objects/op", allocations)
	}
}
