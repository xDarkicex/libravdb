package libravdb

import (
	"context"
	"errors"
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
