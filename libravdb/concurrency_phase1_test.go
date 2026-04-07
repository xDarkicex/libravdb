package libravdb

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"
)

func TestCollectionWriteQueueFull(t *testing.T) {
	db, err := New(
		WithStoragePath(testDBPath(t)),
		WithMaxConcurrentWrites(1),
		WithMaxWriteQueueDepth(1),
	)
	if err != nil {
		t.Fatalf("new db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	collection, err := db.CreateCollection(context.Background(), "queue_full", WithDimension(3))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	release, err := collection.acquireWrite(context.Background())
	if err != nil {
		t.Fatalf("acquire permit: %v", err)
	}

	queuedDone := make(chan error, 1)
	go func() {
		queuedDone <- collection.Insert(context.Background(), "queued", []float32{1, 2, 3}, nil)
	}()

	deadline := time.Now().Add(250 * time.Millisecond)
	for {
		collection.writes.mu.Lock()
		waiting := collection.writes.waiting
		collection.writes.mu.Unlock()
		if waiting >= 1 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("timed out waiting for queued writer")
		}
		time.Sleep(time.Millisecond)
	}

	err = collection.Insert(context.Background(), "overflow", []float32{4, 5, 6}, nil)
	if !errors.Is(err, ErrWriteQueueFull) {
		t.Fatalf("expected ErrWriteQueueFull, got %v", err)
	}

	release()

	select {
	case err := <-queuedDone:
		if err != nil {
			t.Fatalf("queued insert failed: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for queued insert")
	}
}

func TestCollectionWriteWaitRespectsContextCancellation(t *testing.T) {
	db, err := New(
		WithStoragePath(testDBPath(t)),
		WithMaxConcurrentWrites(1),
		WithMaxWriteQueueDepth(2),
	)
	if err != nil {
		t.Fatalf("new db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	collection, err := db.CreateCollection(context.Background(), "queue_cancel", WithDimension(3))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	release, err := collection.acquireWrite(context.Background())
	if err != nil {
		t.Fatalf("acquire permit: %v", err)
	}
	defer release()

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	err = collection.Insert(ctx, "timed_out", []float32{1, 2, 3}, nil)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected context deadline exceeded, got %v", err)
	}
}

func TestNonShardedConcurrentSingleInserts(t *testing.T) {
	db, err := New(
		WithStoragePath(testDBPath(t)),
		WithMaxConcurrentWrites(16),
		WithMaxWriteQueueDepth(16),
	)
	if err != nil {
		t.Fatalf("new db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	collection, err := db.CreateCollection(context.Background(), "queue_nonsharded", WithDimension(3))
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	const inserts = 16
	start := make(chan struct{})
	errCh := make(chan error, inserts)
	var wg sync.WaitGroup

	for i := 0; i < inserts; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			vec := []float32{float32(i), float32(i + 1), float32(i + 2)}
			if err := collection.Insert(context.Background(), fmt.Sprintf("vec_%d", i), vec, nil); err != nil {
				errCh <- err
			}
		}(i)
	}

	close(start)
	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Fatalf("concurrent insert failed: %v", err)
	}

	count, err := collection.Count(context.Background())
	if err != nil {
		t.Fatalf("count collection: %v", err)
	}
	if count != inserts {
		t.Fatalf("expected %d inserted vectors, got %d", inserts, count)
	}
}
