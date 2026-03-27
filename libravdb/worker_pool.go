package libravdb

import (
	"context"
	"sync"
)

// workerPool manages a pool of workers for concurrent batch processing
type workerPool struct {
	workers int
	jobs    chan func() error
	wg      sync.WaitGroup
	closed  bool
	mu      sync.Mutex
}

// newWorkerPool creates a new worker pool with the specified number of workers
func newWorkerPool(workers int) *workerPool {
	if workers <= 0 {
		workers = 1
	}

	pool := &workerPool{
		workers: workers,
		jobs:    make(chan func() error, workers*2), // Buffer to prevent blocking
	}

	// Start workers
	for i := 0; i < workers; i++ {
		pool.wg.Add(1)
		go pool.worker()
	}

	return pool
}

// worker is the worker goroutine that processes jobs
func (p *workerPool) worker() {
	defer p.wg.Done()

	for job := range p.jobs {
		job() // Ignore errors for now, handle them in the job itself
	}
}

// submit submits a job to the worker pool
func (p *workerPool) submit(job func() error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return
	}

	p.jobs <- job
}

// wait waits for all submitted jobs to complete
func (p *workerPool) wait(ctx context.Context) error {
	// Close the jobs channel to signal workers to stop
	p.mu.Lock()
	if !p.closed {
		p.closed = true
		close(p.jobs)
	}
	p.mu.Unlock()

	// Wait for all workers to finish
	done := make(chan struct{})
	go func() {
		p.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// close shuts down the worker pool
func (p *workerPool) close() {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return
	}

	p.closed = true
	close(p.jobs)
	p.wg.Wait()
}
