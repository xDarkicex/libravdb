package libravdb

import (
	"context"
	"sync"
)

// writeController provides Phase 1 write admission control for a collection.
// It bounds in-flight mutation work and the number of waiting writers.
type writeController struct {
	permits       chan struct{}
	maxConcurrent int
	maxQueueDepth int

	mu      sync.Mutex
	waiting int
}

func newWriteController(maxConcurrent, maxQueueDepth int) *writeController {
	if maxConcurrent <= 0 {
		maxConcurrent = 1
	}
	if maxQueueDepth < 0 {
		maxQueueDepth = 0
	}
	return &writeController{
		permits:       make(chan struct{}, maxConcurrent),
		maxConcurrent: maxConcurrent,
		maxQueueDepth: maxQueueDepth,
	}
}

func (w *writeController) acquire(ctx context.Context) (func(), error) {
	if w == nil {
		return func() {}, nil
	}

	select {
	case w.permits <- struct{}{}:
		return w.release, nil
	default:
	}

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	w.mu.Lock()
	if w.waiting >= w.maxQueueDepth {
		w.mu.Unlock()
		return nil, ErrWriteQueueFull
	}
	w.waiting++
	w.mu.Unlock()

	acquired := false
	defer func() {
		if !acquired {
			w.mu.Lock()
			w.waiting--
			w.mu.Unlock()
		}
	}()

	select {
	case w.permits <- struct{}{}:
		acquired = true
		w.mu.Lock()
		w.waiting--
		w.mu.Unlock()
		return w.release, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (w *writeController) release() {
	if w == nil {
		return
	}
	select {
	case <-w.permits:
	default:
	}
}

func (w *writeController) maxParallelism() int {
	if w == nil || w.maxConcurrent <= 0 {
		return 1
	}
	return w.maxConcurrent
}
