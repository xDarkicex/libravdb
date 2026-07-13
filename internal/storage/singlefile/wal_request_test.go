package singlefile

import (
	"errors"
	"sync"
	"testing"
	"unsafe"
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
	pool.complete(first, 42, nil)
	lsn, err := pool.waitFor(first)
	if err != nil || lsn != 42 {
		t.Fatalf("first completion = (%d, %v), want (42, nil)", lsn, err)
	}

	second := pool.acquire(1)
	if second.index != first.index {
		t.Fatalf("request slot was not immediately reusable: first=%d second=%d", first.index, second.index)
	}
	if second.generation == first.generation {
		t.Fatal("reused request slot retained its generation")
	}
	pool.complete(second, 84, nil)
	lsn, err = pool.waitFor(second)
	if err != nil || lsn != 84 {
		t.Fatalf("second completion = (%d, %v), want (84, nil)", lsn, err)
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
	pool.complete(request, 0, want)
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
			lsn, err := pool.waitFor(request)
			if err != nil {
				errs <- err
				return
			}
			if lsn != 99 {
				errs <- errors.New("wrong durable LSN")
			}
		}()
	}
	for _, request := range requests {
		pool.complete(request, 99, nil)
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
		pool.complete(request, 1, nil)
		_, _ = pool.waitFor(request)
	})
	if allocations != 0 {
		t.Fatalf("WAL request completion allocated %.2f objects/op", allocations)
	}
}
