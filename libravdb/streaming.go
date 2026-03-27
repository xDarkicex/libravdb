package libravdb

import (
	"context"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"time"
)

// StreamingBatchInsert provides a streaming interface for large dataset ingestion
// with memory-efficient processing and backpressure handling
type StreamingBatchInsert struct {
	collection   *Collection
	options      *StreamingOptions
	inputChan    chan *VectorEntry
	resultChan   chan *StreamingResult
	errorChan    chan error
	doneChan     chan struct{}
	ctx          context.Context
	cancel       context.CancelFunc
	wg           sync.WaitGroup
	stats        *StreamingStats
	backpressure *BackpressureController
	started      int32
	stopped      int32
}

// StreamingOptions configures streaming batch operations
type StreamingOptions struct {
	// BufferSize is the size of the input buffer channel
	BufferSize int `json:"buffer_size"`

	// ChunkSize is the number of items to process in each batch
	ChunkSize int `json:"chunk_size"`

	// MaxConcurrency is the maximum number of concurrent workers
	MaxConcurrency int `json:"max_concurrency"`

	// FlushInterval is how often to flush pending items
	FlushInterval time.Duration `json:"flush_interval"`

	// MaxMemoryUsage is the maximum memory usage before applying backpressure
	MaxMemoryUsage int64 `json:"max_memory_usage"`

	// BackpressureThreshold is the buffer fill percentage that triggers backpressure
	BackpressureThreshold float64 `json:"backpressure_threshold"`

	// Timeout is the overall timeout for the streaming operation
	Timeout time.Duration `json:"timeout"`

	// EnableBackpressure determines if backpressure handling is enabled
	EnableBackpressure bool `json:"enable_backpressure"`

	// ProgressCallback is called periodically with progress updates
	ProgressCallback func(stats *StreamingStats) `json:"-"`

	// ErrorCallback is called when errors occur during processing
	ErrorCallback func(err error, entry *VectorEntry) `json:"-"`

	// CompletionCallback is called when streaming completes
	CompletionCallback func(finalStats *StreamingStats) `json:"-"`
}

// StreamingResult represents the result of processing a batch of items
type StreamingResult struct {
	BatchID    int                `json:"batch_id"`
	Processed  int                `json:"processed"`
	Successful int                `json:"successful"`
	Failed     int                `json:"failed"`
	Errors     []error            `json:"errors,omitempty"`
	Items      []*BatchItemResult `json:"items,omitempty"`
	Duration   time.Duration      `json:"duration"`
	Timestamp  time.Time          `json:"timestamp"`
}

// StreamingStats provides real-time statistics for streaming operations
type StreamingStats struct {
	// Counters
	TotalReceived   int64 `json:"total_received"`
	TotalProcessed  int64 `json:"total_processed"`
	TotalSuccessful int64 `json:"total_successful"`
	TotalFailed     int64 `json:"total_failed"`

	// Rates
	ItemsPerSecond float64 `json:"items_per_second"`
	SuccessRate    float64 `json:"success_rate"`
	ErrorRate      float64 `json:"error_rate"`

	// Memory and backpressure
	CurrentMemoryUsage int64   `json:"current_memory_usage"`
	BufferUtilization  float64 `json:"buffer_utilization"`
	BackpressureActive bool    `json:"backpressure_active"`

	// Timing
	StartTime    time.Time     `json:"start_time"`
	LastUpdate   time.Time     `json:"last_update"`
	ElapsedTime  time.Duration `json:"elapsed_time"`
	EstimatedETA time.Duration `json:"estimated_eta,omitempty"`

	// Current state
	ActiveWorkers int    `json:"active_workers"`
	QueuedItems   int    `json:"queued_items"`
	Status        string `json:"status"`

	// Errors
	LastError    error            `json:"last_error,omitempty"`
	ErrorsByType map[string]int64 `json:"errors_by_type,omitempty"`
	RecentErrors []error          `json:"recent_errors,omitempty"`

	mutex sync.RWMutex
}

// BackpressureController manages backpressure for streaming operations
type BackpressureController struct {
	enabled           bool
	threshold         float64
	maxMemoryUsage    int64
	currentMemory     int64
	bufferSize        int
	currentBufferSize int32
	active            int32
	mutex             sync.RWMutex
}

// DefaultStreamingOptions returns sensible defaults for streaming operations
func DefaultStreamingOptions() *StreamingOptions {
	return &StreamingOptions{
		BufferSize:            10000,
		ChunkSize:             1000,
		MaxConcurrency:        4,
		FlushInterval:         5 * time.Second,
		MaxMemoryUsage:        1024 * 1024 * 1024, // 1GB
		BackpressureThreshold: 0.8,                // 80%
		Timeout:               30 * time.Minute,
		EnableBackpressure:    true,
	}
}

// NewStreamingBatchInsert creates a new streaming batch insert operation
func (c *Collection) NewStreamingBatchInsert(opts ...*StreamingOptions) *StreamingBatchInsert {
	options := DefaultStreamingOptions()
	if len(opts) > 0 && opts[0] != nil {
		options = opts[0]
	}

	ctx, cancel := context.WithCancel(context.Background())
	if options.Timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, options.Timeout)
	}

	stats := &StreamingStats{
		StartTime:    time.Now(),
		Status:       "initialized",
		ErrorsByType: make(map[string]int64),
		RecentErrors: make([]error, 0, 10),
	}

	backpressure := &BackpressureController{
		enabled:        options.EnableBackpressure,
		threshold:      options.BackpressureThreshold,
		maxMemoryUsage: options.MaxMemoryUsage,
		bufferSize:     options.BufferSize,
	}

	return &StreamingBatchInsert{
		collection:   c,
		options:      options,
		inputChan:    make(chan *VectorEntry, options.BufferSize),
		resultChan:   make(chan *StreamingResult, 100),
		errorChan:    make(chan error, 100),
		doneChan:     make(chan struct{}),
		ctx:          ctx,
		cancel:       cancel,
		stats:        stats,
		backpressure: backpressure,
	}
}

// Start begins the streaming operation
func (s *StreamingBatchInsert) Start() error {
	if !atomic.CompareAndSwapInt32(&s.started, 0, 1) {
		return fmt.Errorf("streaming operation already started")
	}

	s.stats.mutex.Lock()
	s.stats.Status = "running"
	s.stats.StartTime = time.Now()
	s.stats.mutex.Unlock()

	// Start worker goroutines
	for i := 0; i < s.options.MaxConcurrency; i++ {
		s.wg.Add(1)
		go s.worker(i)
	}

	// Start statistics updater
	s.wg.Add(1)
	go s.statsUpdater()

	// Start flush timer
	s.wg.Add(1)
	go s.flushTimer()

	return nil
}

// Send adds a vector entry to the streaming pipeline
// Returns an error if backpressure is active or the operation is stopped
func (s *StreamingBatchInsert) Send(entry *VectorEntry) error {
	if atomic.LoadInt32(&s.stopped) == 1 {
		return fmt.Errorf("streaming operation is stopped")
	}

	// Respect timeout/cancellation before attempting a buffered send.
	select {
	case <-s.ctx.Done():
		return s.ctx.Err()
	default:
	}

	// Check backpressure
	if s.backpressure.ShouldApplyBackpressure() {
		return ErrBackpressureActive
	}

	// Update buffer utilization
	atomic.AddInt32(&s.backpressure.currentBufferSize, 1)

	select {
	case s.inputChan <- entry:
		atomic.AddInt64(&s.stats.TotalReceived, 1)
		return nil
	case <-s.ctx.Done():
		atomic.AddInt32(&s.backpressure.currentBufferSize, -1)
		return s.ctx.Err()
	default:
		atomic.AddInt32(&s.backpressure.currentBufferSize, -1)
		return ErrBackpressureActive
	}
}

// SendBatch sends multiple entries efficiently
func (s *StreamingBatchInsert) SendBatch(entries []*VectorEntry) error {
	for _, entry := range entries {
		if err := s.Send(entry); err != nil {
			return fmt.Errorf("failed to send entry %s: %w", entry.ID, err)
		}
	}
	return nil
}

// Results returns a channel for receiving batch processing results
func (s *StreamingBatchInsert) Results() <-chan *StreamingResult {
	return s.resultChan
}

// Errors returns a channel for receiving processing errors
func (s *StreamingBatchInsert) Errors() <-chan error {
	return s.errorChan
}

// Stats returns current streaming statistics
func (s *StreamingBatchInsert) Stats() *StreamingStats {
	s.stats.mutex.RLock()
	defer s.stats.mutex.RUnlock()

	// Create a copy to avoid race conditions
	statsCopy := *s.stats
	statsCopy.ErrorsByType = make(map[string]int64)
	for k, v := range s.stats.ErrorsByType {
		statsCopy.ErrorsByType[k] = v
	}

	statsCopy.RecentErrors = make([]error, len(s.stats.RecentErrors))
	copy(statsCopy.RecentErrors, s.stats.RecentErrors)

	// Refresh live queue/backpressure state so callers do not depend on the
	// periodic stats ticker for correctness-sensitive reads.
	currentBuffer := atomic.LoadInt32(&s.backpressure.currentBufferSize)
	statsCopy.BufferUtilization = float64(currentBuffer) / float64(s.options.BufferSize)
	statsCopy.QueuedItems = int(currentBuffer)
	statsCopy.BackpressureActive = s.backpressure.IsActive()

	return &statsCopy
}

// Close gracefully shuts down the streaming operation
func (s *StreamingBatchInsert) Close() error {
	if !atomic.CompareAndSwapInt32(&s.stopped, 0, 1) {
		return nil // Already stopped
	}

	s.stats.mutex.Lock()
	s.stats.Status = "stopping"
	s.stats.mutex.Unlock()

	// Cancel context first to signal all goroutines to stop
	s.cancel()

	// Close input channel to signal workers to finish
	close(s.inputChan)

	// Wait for all workers to complete with timeout
	done := make(chan struct{})
	go func() {
		s.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		// All workers finished normally
	case <-time.After(5 * time.Second):
		// Timeout waiting for workers - continue anyway
	}

	// Close channels
	close(s.resultChan)
	close(s.errorChan)
	close(s.doneChan)

	s.stats.mutex.Lock()
	s.stats.Status = "stopped"
	s.stats.LastUpdate = time.Now()
	s.stats.ElapsedTime = time.Since(s.stats.StartTime)
	s.stats.mutex.Unlock()

	// Call completion callback
	if s.options.CompletionCallback != nil {
		s.options.CompletionCallback(s.Stats())
	}

	return nil
}

// Wait waits for the streaming operation to complete
func (s *StreamingBatchInsert) Wait() error {
	select {
	case <-s.doneChan:
		return nil
	case <-s.ctx.Done():
		return s.ctx.Err()
	}
}

// worker processes entries from the input channel
func (s *StreamingBatchInsert) worker(workerID int) {
	defer s.wg.Done()

	batch := make([]*VectorEntry, 0, s.options.ChunkSize)
	batchID := 0

	for {
		select {
		case entry, ok := <-s.inputChan:
			if !ok {
				// Channel closed, process final batch if any
				if len(batch) > 0 {
					s.processBatch(batch, batchID, workerID)
				}
				return
			}

			batch = append(batch, entry)
			atomic.AddInt32(&s.backpressure.currentBufferSize, -1)

			// Process batch when it reaches chunk size
			if len(batch) >= s.options.ChunkSize {
				s.processBatch(batch, batchID, workerID)
				batch = batch[:0] // Reset batch
				batchID++
				continue
			}

			// Flush partial batches when the shared queue drains so low-volume
			// streams still make forward progress under multi-worker scheduling.
			if len(batch) > 0 && len(s.inputChan) == 0 {
				s.processBatch(batch, batchID, workerID)
				batch = batch[:0]
				batchID++
			}

		case <-s.ctx.Done():
			return
		}
	}
}

// processBatch processes a batch of entries
func (s *StreamingBatchInsert) processBatch(batch []*VectorEntry, batchID, workerID int) {
	startTime := time.Now()

	// Update active workers count
	s.stats.mutex.Lock()
	s.stats.ActiveWorkers++
	s.stats.mutex.Unlock()

	defer func() {
		s.stats.mutex.Lock()
		s.stats.ActiveWorkers--
		s.stats.mutex.Unlock()
	}()

	// Create batch operation
	batchOpts := DefaultBatchOptions()
	batchOpts.ChunkSize = len(batch)
	batchOpts.MaxConcurrency = 1 // Single worker per batch
	batchOpts.FailFast = false

	batchInsert := s.collection.NewBatchInsert(batch, batchOpts)

	// Execute batch
	result, err := batchInsert.Execute(s.ctx)

	duration := time.Since(startTime)
	processed := len(batch)

	// Update statistics
	atomic.AddInt64(&s.stats.TotalProcessed, int64(processed))

	streamingResult := &StreamingResult{
		BatchID:   batchID,
		Processed: processed,
		Duration:  duration,
		Timestamp: time.Now(),
	}

	if err != nil {
		// Handle batch error
		s.handleError(err, nil)
		streamingResult.Failed = processed
		streamingResult.Errors = []error{err}
		atomic.AddInt64(&s.stats.TotalFailed, int64(processed))
	} else if result != nil {
		streamingResult.Successful = result.Successful
		streamingResult.Failed = result.Failed
		streamingResult.Items = result.Items

		atomic.AddInt64(&s.stats.TotalSuccessful, int64(result.Successful))
		atomic.AddInt64(&s.stats.TotalFailed, int64(result.Failed))

		// Handle individual item errors
		for _, item := range result.Items {
			if item.Error != nil {
				s.handleError(item.Error, nil)
			}
		}
	}

	// Send result
	select {
	case s.resultChan <- streamingResult:
	case <-s.ctx.Done():
		return
	}
}

// flushTimer periodically flushes pending items
func (s *StreamingBatchInsert) flushTimer() {
	defer s.wg.Done()

	ticker := time.NewTicker(s.options.FlushInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// Flush logic would go here if we had partial batches to flush
			// For now, we rely on workers to process complete batches
		case <-s.ctx.Done():
			return
		}
	}
}

// statsUpdater periodically updates statistics and calls progress callback
func (s *StreamingBatchInsert) statsUpdater() {
	defer s.wg.Done()

	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			s.updateStats()
			if s.options.ProgressCallback != nil {
				s.options.ProgressCallback(s.Stats())
			}
		case <-s.ctx.Done():
			return
		}
	}
}

// updateStats updates internal statistics
func (s *StreamingBatchInsert) updateStats() {
	s.stats.mutex.Lock()
	defer s.stats.mutex.Unlock()

	now := time.Now()
	s.stats.LastUpdate = now
	s.stats.ElapsedTime = now.Sub(s.stats.StartTime)

	// Calculate rates
	if s.stats.ElapsedTime > 0 {
		s.stats.ItemsPerSecond = float64(s.stats.TotalProcessed) / s.stats.ElapsedTime.Seconds()
	}

	total := s.stats.TotalSuccessful + s.stats.TotalFailed
	if total > 0 {
		s.stats.SuccessRate = float64(s.stats.TotalSuccessful) / float64(total)
		s.stats.ErrorRate = float64(s.stats.TotalFailed) / float64(total)
	}

	// Update buffer utilization
	currentBuffer := atomic.LoadInt32(&s.backpressure.currentBufferSize)
	s.stats.BufferUtilization = float64(currentBuffer) / float64(s.options.BufferSize)
	s.stats.QueuedItems = int(currentBuffer)

	// Update backpressure status
	s.stats.BackpressureActive = s.backpressure.IsActive()
}

// handleError handles errors during processing
func (s *StreamingBatchInsert) handleError(err error, entry *VectorEntry) {
	if err == nil {
		return
	}

	s.stats.mutex.Lock()
	s.stats.LastError = err

	// Categorize error
	errorType := categorizeStreamingError(err)
	s.stats.ErrorsByType[errorType]++

	// Add to recent errors (keep only last 10)
	s.stats.RecentErrors = append(s.stats.RecentErrors, err)
	if len(s.stats.RecentErrors) > 10 {
		s.stats.RecentErrors = s.stats.RecentErrors[1:]
	}
	s.stats.mutex.Unlock()

	// Call error callback
	if s.options.ErrorCallback != nil {
		s.options.ErrorCallback(err, entry)
	}

	// Send error to error channel (non-blocking)
	select {
	case s.errorChan <- err:
	default:
		// Error channel is full, drop the error
	}
}

// ShouldApplyBackpressure determines if backpressure should be applied
func (bp *BackpressureController) ShouldApplyBackpressure() bool {
	if !bp.enabled {
		return false
	}

	bp.mutex.RLock()
	defer bp.mutex.RUnlock()

	// Check buffer utilization
	currentBuffer := atomic.LoadInt32(&bp.currentBufferSize)
	bufferUtilization := float64(currentBuffer) / float64(bp.bufferSize)

	// Check memory usage
	memoryPressure := bp.currentMemory > bp.maxMemoryUsage

	shouldApply := bufferUtilization > bp.threshold || memoryPressure

	if shouldApply {
		atomic.StoreInt32(&bp.active, 1)
	} else {
		atomic.StoreInt32(&bp.active, 0)
	}

	return shouldApply
}

// IsActive returns whether backpressure is currently active
func (bp *BackpressureController) IsActive() bool {
	return atomic.LoadInt32(&bp.active) == 1
}

// UpdateMemoryUsage updates the current memory usage for backpressure calculation
func (bp *BackpressureController) UpdateMemoryUsage(usage int64) {
	bp.mutex.Lock()
	bp.currentMemory = usage
	bp.mutex.Unlock()
}

// categorizeStreamingError categorizes errors for statistics
func categorizeStreamingError(err error) string {
	if err == nil {
		return "none"
	}

	errStr := err.Error()
	switch {
	case err == context.DeadlineExceeded:
		return "timeout"
	case err == context.Canceled:
		return "cancelled"
	case err == ErrBackpressureActive:
		return "backpressure"
	case contains(errStr, "dimension"):
		return "validation"
	case contains(errStr, "memory"):
		return "memory"
	case contains(errStr, "duplicate"):
		return "duplicate"
	default:
		return "internal"
	}
}

// StreamingReader provides an interface for reading from various data sources
type StreamingReader interface {
	// Read reads the next vector entry from the source
	Read() (*VectorEntry, error)

	// Close closes the reader and releases resources
	Close() error
}

// ChannelStreamingReader reads from a channel of VectorEntry
type ChannelStreamingReader struct {
	ch     <-chan *VectorEntry
	closed bool
}

// NewChannelStreamingReader creates a new channel-based streaming reader
func NewChannelStreamingReader(ch <-chan *VectorEntry) *ChannelStreamingReader {
	return &ChannelStreamingReader{ch: ch}
}

// Read reads the next entry from the channel
func (r *ChannelStreamingReader) Read() (*VectorEntry, error) {
	if r.closed {
		return nil, io.EOF
	}

	entry, ok := <-r.ch
	if !ok {
		r.closed = true
		return nil, io.EOF
	}

	return entry, nil
}

// Close closes the reader
func (r *ChannelStreamingReader) Close() error {
	r.closed = true
	return nil
}

// StreamFromReader creates a streaming batch insert from a StreamingReader
func (c *Collection) StreamFromReader(reader StreamingReader, opts ...*StreamingOptions) (*StreamingBatchInsert, error) {
	stream := c.NewStreamingBatchInsert(opts...)

	// Start the streaming operation
	if err := stream.Start(); err != nil {
		return nil, fmt.Errorf("failed to start streaming: %w", err)
	}

	// Start a goroutine to read from the reader and send to the stream
	go func() {
		defer stream.Close()
		defer reader.Close()

		for {
			entry, err := reader.Read()
			if err == io.EOF {
				break
			}
			if err != nil {
				stream.handleError(fmt.Errorf("reader error: %w", err), nil)
				break
			}

			if err := stream.Send(entry); err != nil {
				stream.handleError(fmt.Errorf("send error: %w", err), entry)
				break
			}
		}
	}()

	return stream, nil
}

// StreamingBatchUpdate provides a streaming interface for large-scale update operations
type StreamingBatchUpdate struct {
	collection   *Collection
	options      *StreamingOptions
	inputChan    chan *VectorUpdate
	resultChan   chan *StreamingResult
	errorChan    chan error
	doneChan     chan struct{}
	ctx          context.Context
	cancel       context.CancelFunc
	wg           sync.WaitGroup
	stats        *StreamingStats
	backpressure *BackpressureController
	started      int32
	stopped      int32
}

// StreamingBatchDelete provides a streaming interface for large-scale delete operations
type StreamingBatchDelete struct {
	collection   *Collection
	options      *StreamingOptions
	inputChan    chan string // Channel for IDs to delete
	resultChan   chan *StreamingResult
	errorChan    chan error
	doneChan     chan struct{}
	ctx          context.Context
	cancel       context.CancelFunc
	wg           sync.WaitGroup
	stats        *StreamingStats
	backpressure *BackpressureController
	started      int32
	stopped      int32
}

// NewStreamingBatchUpdate creates a new streaming batch update operation
func (c *Collection) NewStreamingBatchUpdate(opts ...*StreamingOptions) *StreamingBatchUpdate {
	options := DefaultStreamingOptions()
	if len(opts) > 0 && opts[0] != nil {
		options = opts[0]
	}

	ctx, cancel := context.WithCancel(context.Background())
	if options.Timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, options.Timeout)
	}

	stats := &StreamingStats{
		StartTime:    time.Now(),
		Status:       "initialized",
		ErrorsByType: make(map[string]int64),
		RecentErrors: make([]error, 0, 10),
	}

	backpressure := &BackpressureController{
		enabled:        options.EnableBackpressure,
		threshold:      options.BackpressureThreshold,
		maxMemoryUsage: options.MaxMemoryUsage,
		bufferSize:     options.BufferSize,
	}

	return &StreamingBatchUpdate{
		collection:   c,
		options:      options,
		inputChan:    make(chan *VectorUpdate, options.BufferSize),
		resultChan:   make(chan *StreamingResult, 100),
		errorChan:    make(chan error, 100),
		doneChan:     make(chan struct{}),
		ctx:          ctx,
		cancel:       cancel,
		stats:        stats,
		backpressure: backpressure,
	}
}

// NewStreamingBatchDelete creates a new streaming batch delete operation
func (c *Collection) NewStreamingBatchDelete(opts ...*StreamingOptions) *StreamingBatchDelete {
	options := DefaultStreamingOptions()
	if len(opts) > 0 && opts[0] != nil {
		options = opts[0]
	}

	ctx, cancel := context.WithCancel(context.Background())
	if options.Timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, options.Timeout)
	}

	stats := &StreamingStats{
		StartTime:    time.Now(),
		Status:       "initialized",
		ErrorsByType: make(map[string]int64),
		RecentErrors: make([]error, 0, 10),
	}

	backpressure := &BackpressureController{
		enabled:        options.EnableBackpressure,
		threshold:      options.BackpressureThreshold,
		maxMemoryUsage: options.MaxMemoryUsage,
		bufferSize:     options.BufferSize,
	}

	return &StreamingBatchDelete{
		collection:   c,
		options:      options,
		inputChan:    make(chan string, options.BufferSize),
		resultChan:   make(chan *StreamingResult, 100),
		errorChan:    make(chan error, 100),
		doneChan:     make(chan struct{}),
		ctx:          ctx,
		cancel:       cancel,
		stats:        stats,
		backpressure: backpressure,
	}
}

// Start begins the streaming update operation
func (s *StreamingBatchUpdate) Start() error {
	if !atomic.CompareAndSwapInt32(&s.started, 0, 1) {
		return fmt.Errorf("streaming operation already started")
	}

	s.stats.mutex.Lock()
	s.stats.Status = "running"
	s.stats.StartTime = time.Now()
	s.stats.mutex.Unlock()

	// Start worker goroutines
	for i := 0; i < s.options.MaxConcurrency; i++ {
		s.wg.Add(1)
		go s.updateWorker(i)
	}

	// Start statistics updater
	s.wg.Add(1)
	go s.updateStatsUpdater()

	return nil
}

// Start begins the streaming delete operation
func (s *StreamingBatchDelete) Start() error {
	if !atomic.CompareAndSwapInt32(&s.started, 0, 1) {
		return fmt.Errorf("streaming operation already started")
	}

	s.stats.mutex.Lock()
	s.stats.Status = "running"
	s.stats.StartTime = time.Now()
	s.stats.mutex.Unlock()

	// Start worker goroutines
	for i := 0; i < s.options.MaxConcurrency; i++ {
		s.wg.Add(1)
		go s.deleteWorker(i)
	}

	// Start statistics updater
	s.wg.Add(1)
	go s.deleteStatsUpdater()

	return nil
}

// Send adds an update to the streaming pipeline
func (s *StreamingBatchUpdate) Send(update *VectorUpdate) error {
	if atomic.LoadInt32(&s.stopped) == 1 {
		return fmt.Errorf("streaming operation has been stopped")
	}

	// Apply backpressure if enabled
	if s.backpressure.enabled {
		if s.backpressure.ShouldApplyBackpressure() {
			atomic.StoreInt32(&s.backpressure.active, 1)
			// Block until backpressure is relieved
			for s.backpressure.ShouldApplyBackpressure() {
				select {
				case <-s.ctx.Done():
					return s.ctx.Err()
				case <-time.After(10 * time.Millisecond):
					// Continue checking
				}
			}
			atomic.StoreInt32(&s.backpressure.active, 0)
		}
	}

	select {
	case s.inputChan <- update:
		atomic.AddInt64(&s.stats.TotalReceived, 1)
		atomic.AddInt32(&s.backpressure.currentBufferSize, 1)
		return nil
	case <-s.ctx.Done():
		return s.ctx.Err()
	}
}

// Send adds an ID to delete to the streaming pipeline
func (s *StreamingBatchDelete) Send(id string) error {
	if atomic.LoadInt32(&s.stopped) == 1 {
		return fmt.Errorf("streaming operation has been stopped")
	}

	// Apply backpressure if enabled
	if s.backpressure.enabled {
		if s.backpressure.ShouldApplyBackpressure() {
			atomic.StoreInt32(&s.backpressure.active, 1)
			// Block until backpressure is relieved
			for s.backpressure.ShouldApplyBackpressure() {
				select {
				case <-s.ctx.Done():
					return s.ctx.Err()
				case <-time.After(10 * time.Millisecond):
					// Continue checking
				}
			}
			atomic.StoreInt32(&s.backpressure.active, 0)
		}
	}

	select {
	case s.inputChan <- id:
		atomic.AddInt64(&s.stats.TotalReceived, 1)
		atomic.AddInt32(&s.backpressure.currentBufferSize, 1)
		return nil
	case <-s.ctx.Done():
		return s.ctx.Err()
	}
}

// updateWorker processes updates in batches
func (s *StreamingBatchUpdate) updateWorker(workerID int) {
	defer s.wg.Done()

	batch := make([]*VectorUpdate, 0, s.options.ChunkSize)
	ticker := time.NewTicker(s.options.FlushInterval)
	defer ticker.Stop()

	batchID := 0

	for {
		select {
		case update, ok := <-s.inputChan:
			if !ok {
				// Channel closed, process final batch
				if len(batch) > 0 {
					s.processUpdateBatch(batch, batchID, workerID)
				}
				return
			}

			batch = append(batch, update)
			atomic.AddInt32(&s.backpressure.currentBufferSize, -1)

			// Process batch when full
			if len(batch) >= s.options.ChunkSize {
				s.processUpdateBatch(batch, batchID, workerID)
				batch = batch[:0] // Reset batch
				batchID++
			}

		case <-ticker.C:
			// Flush pending items
			if len(batch) > 0 {
				s.processUpdateBatch(batch, batchID, workerID)
				batch = batch[:0] // Reset batch
				batchID++
			}

		case <-s.ctx.Done():
			return
		}
	}
}

// deleteWorker processes deletes in batches
func (s *StreamingBatchDelete) deleteWorker(workerID int) {
	defer s.wg.Done()

	batch := make([]string, 0, s.options.ChunkSize)
	ticker := time.NewTicker(s.options.FlushInterval)
	defer ticker.Stop()

	batchID := 0

	for {
		select {
		case id, ok := <-s.inputChan:
			if !ok {
				// Channel closed, process final batch
				if len(batch) > 0 {
					s.processDeleteBatch(batch, batchID, workerID)
				}
				return
			}

			batch = append(batch, id)
			atomic.AddInt32(&s.backpressure.currentBufferSize, -1)

			// Process batch when full
			if len(batch) >= s.options.ChunkSize {
				s.processDeleteBatch(batch, batchID, workerID)
				batch = batch[:0] // Reset batch
				batchID++
			}

		case <-ticker.C:
			// Flush pending items
			if len(batch) > 0 {
				s.processDeleteBatch(batch, batchID, workerID)
				batch = batch[:0] // Reset batch
				batchID++
			}

		case <-s.ctx.Done():
			return
		}
	}
}

// processUpdateBatch processes a batch of updates
func (s *StreamingBatchUpdate) processUpdateBatch(batch []*VectorUpdate, batchID, workerID int) {
	startTime := time.Now()

	batchUpdate := s.collection.NewBatchUpdate(batch)
	result, err := batchUpdate.Execute(s.ctx)

	duration := time.Since(startTime)

	streamingResult := &StreamingResult{
		BatchID:    batchID,
		Processed:  len(batch),
		Successful: result.Successful,
		Failed:     result.Failed,
		Duration:   duration,
		Timestamp:  time.Now(),
	}

	if err != nil {
		streamingResult.Errors = []error{err}
		s.handleUpdateError(err, nil)
	}

	// Update statistics
	atomic.AddInt64(&s.stats.TotalProcessed, int64(len(batch)))
	atomic.AddInt64(&s.stats.TotalSuccessful, int64(result.Successful))
	atomic.AddInt64(&s.stats.TotalFailed, int64(result.Failed))

	// Send result
	select {
	case s.resultChan <- streamingResult:
	case <-s.ctx.Done():
	}
}

// processDeleteBatch processes a batch of deletes
func (s *StreamingBatchDelete) processDeleteBatch(batch []string, batchID, workerID int) {
	startTime := time.Now()

	batchDelete := s.collection.NewBatchDelete(batch)
	result, err := batchDelete.Execute(s.ctx)

	duration := time.Since(startTime)

	streamingResult := &StreamingResult{
		BatchID:    batchID,
		Processed:  len(batch),
		Successful: result.Successful,
		Failed:     result.Failed,
		Duration:   duration,
		Timestamp:  time.Now(),
	}

	if err != nil {
		streamingResult.Errors = []error{err}
		s.handleDeleteError(err, "")
	}

	// Update statistics
	atomic.AddInt64(&s.stats.TotalProcessed, int64(len(batch)))
	atomic.AddInt64(&s.stats.TotalSuccessful, int64(result.Successful))
	atomic.AddInt64(&s.stats.TotalFailed, int64(result.Failed))

	// Send result
	select {
	case s.resultChan <- streamingResult:
	case <-s.ctx.Done():
	}
}

// handleUpdateError handles errors during update processing
func (s *StreamingBatchUpdate) handleUpdateError(err error, update *VectorUpdate) {
	s.stats.mutex.Lock()
	defer s.stats.mutex.Unlock()

	s.stats.LastError = err
	if len(s.stats.RecentErrors) >= 10 {
		s.stats.RecentErrors = s.stats.RecentErrors[1:]
	}
	s.stats.RecentErrors = append(s.stats.RecentErrors, err)

	// Categorize error
	errorType := "unknown"
	if err != nil {
		errStr := err.Error()
		switch {
		case contains(errStr, "dimension"):
			errorType = "validation"
		case contains(errStr, "not found"):
			errorType = "not_found"
		case contains(errStr, "timeout"):
			errorType = "timeout"
		default:
			errorType = "internal"
		}
	}
	s.stats.ErrorsByType[errorType]++

	// Call error callback if provided
	if s.options.ErrorCallback != nil {
		var entry *VectorEntry
		if update != nil {
			entry = &VectorEntry{
				ID:       update.ID,
				Vector:   update.Vector,
				Metadata: update.Metadata,
			}
		}
		s.options.ErrorCallback(err, entry)
	}

	// Send error to error channel
	select {
	case s.errorChan <- err:
	default:
		// Error channel full, drop error
	}
}

// handleDeleteError handles errors during delete processing
func (s *StreamingBatchDelete) handleDeleteError(err error, id string) {
	s.stats.mutex.Lock()
	defer s.stats.mutex.Unlock()

	s.stats.LastError = err
	if len(s.stats.RecentErrors) >= 10 {
		s.stats.RecentErrors = s.stats.RecentErrors[1:]
	}
	s.stats.RecentErrors = append(s.stats.RecentErrors, err)

	// Categorize error
	errorType := "unknown"
	if err != nil {
		errStr := err.Error()
		switch {
		case contains(errStr, "not found"):
			errorType = "not_found"
		case contains(errStr, "timeout"):
			errorType = "timeout"
		default:
			errorType = "internal"
		}
	}
	s.stats.ErrorsByType[errorType]++

	// Call error callback if provided
	if s.options.ErrorCallback != nil {
		var entry *VectorEntry
		if id != "" {
			entry = &VectorEntry{ID: id}
		}
		s.options.ErrorCallback(err, entry)
	}

	// Send error to error channel
	select {
	case s.errorChan <- err:
	default:
		// Error channel full, drop error
	}
}

// updateStatsUpdater periodically updates statistics for update operations
func (s *StreamingBatchUpdate) updateStatsUpdater() {
	defer s.wg.Done()

	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			s.updateStats()
		case <-s.ctx.Done():
			return
		}
	}
}

// deleteStatsUpdater periodically updates statistics for delete operations
func (s *StreamingBatchDelete) deleteStatsUpdater() {
	defer s.wg.Done()

	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			s.updateStats()
		case <-s.ctx.Done():
			return
		}
	}
}

// updateStats updates the streaming statistics
func (s *StreamingBatchUpdate) updateStats() {
	s.stats.mutex.Lock()
	defer s.stats.mutex.Unlock()

	now := time.Now()
	s.stats.LastUpdate = now
	s.stats.ElapsedTime = now.Sub(s.stats.StartTime)

	// Calculate rates
	if s.stats.ElapsedTime > 0 {
		s.stats.ItemsPerSecond = float64(s.stats.TotalProcessed) / s.stats.ElapsedTime.Seconds()
	}

	if s.stats.TotalProcessed > 0 {
		s.stats.SuccessRate = float64(s.stats.TotalSuccessful) / float64(s.stats.TotalProcessed)
		s.stats.ErrorRate = float64(s.stats.TotalFailed) / float64(s.stats.TotalProcessed)
	}

	// Buffer utilization
	currentBuffer := atomic.LoadInt32(&s.backpressure.currentBufferSize)
	s.stats.BufferUtilization = float64(currentBuffer) / float64(s.backpressure.bufferSize)
	s.stats.QueuedItems = int(currentBuffer)
	s.stats.BackpressureActive = atomic.LoadInt32(&s.backpressure.active) == 1

	// Call progress callback if provided
	if s.options.ProgressCallback != nil {
		s.options.ProgressCallback(s.stats)
	}
}

// updateStats updates the streaming statistics for delete operations
func (s *StreamingBatchDelete) updateStats() {
	s.stats.mutex.Lock()
	defer s.stats.mutex.Unlock()

	now := time.Now()
	s.stats.LastUpdate = now
	s.stats.ElapsedTime = now.Sub(s.stats.StartTime)

	// Calculate rates
	if s.stats.ElapsedTime > 0 {
		s.stats.ItemsPerSecond = float64(s.stats.TotalProcessed) / s.stats.ElapsedTime.Seconds()
	}

	if s.stats.TotalProcessed > 0 {
		s.stats.SuccessRate = float64(s.stats.TotalSuccessful) / float64(s.stats.TotalProcessed)
		s.stats.ErrorRate = float64(s.stats.TotalFailed) / float64(s.stats.TotalProcessed)
	}

	// Buffer utilization
	currentBuffer := atomic.LoadInt32(&s.backpressure.currentBufferSize)
	s.stats.BufferUtilization = float64(currentBuffer) / float64(s.backpressure.bufferSize)
	s.stats.QueuedItems = int(currentBuffer)
	s.stats.BackpressureActive = atomic.LoadInt32(&s.backpressure.active) == 1

	// Call progress callback if provided
	if s.options.ProgressCallback != nil {
		s.options.ProgressCallback(s.stats)
	}
}

// Results returns the result channel for streaming updates
func (s *StreamingBatchUpdate) Results() <-chan *StreamingResult {
	return s.resultChan
}

// Errors returns the error channel for streaming updates
func (s *StreamingBatchUpdate) Errors() <-chan error {
	return s.errorChan
}

// Stats returns current statistics for streaming updates
func (s *StreamingBatchUpdate) Stats() *StreamingStats {
	s.stats.mutex.RLock()
	defer s.stats.mutex.RUnlock()

	// Create a copy to avoid race conditions
	statsCopy := *s.stats
	statsCopy.ErrorsByType = make(map[string]int64)
	for k, v := range s.stats.ErrorsByType {
		statsCopy.ErrorsByType[k] = v
	}

	recentErrors := make([]error, len(s.stats.RecentErrors))
	copy(recentErrors, s.stats.RecentErrors)
	statsCopy.RecentErrors = recentErrors

	return &statsCopy
}

// Results returns the result channel for streaming deletes
func (s *StreamingBatchDelete) Results() <-chan *StreamingResult {
	return s.resultChan
}

// Errors returns the error channel for streaming deletes
func (s *StreamingBatchDelete) Errors() <-chan error {
	return s.errorChan
}

// Stats returns current statistics for streaming deletes
func (s *StreamingBatchDelete) Stats() *StreamingStats {
	s.stats.mutex.RLock()
	defer s.stats.mutex.RUnlock()

	// Create a copy to avoid race conditions
	statsCopy := *s.stats
	statsCopy.ErrorsByType = make(map[string]int64)
	for k, v := range s.stats.ErrorsByType {
		statsCopy.ErrorsByType[k] = v
	}

	recentErrors := make([]error, len(s.stats.RecentErrors))
	copy(recentErrors, s.stats.RecentErrors)
	statsCopy.RecentErrors = recentErrors

	return &statsCopy
}

// Close gracefully shuts down the streaming update operation
func (s *StreamingBatchUpdate) Close() error {
	if !atomic.CompareAndSwapInt32(&s.stopped, 0, 1) {
		return nil // Already stopped
	}

	s.stats.mutex.Lock()
	s.stats.Status = "stopping"
	s.stats.mutex.Unlock()

	// Close input channel to signal workers to finish
	close(s.inputChan)

	// Cancel context
	s.cancel()

	// Wait for all workers to finish
	s.wg.Wait()

	// Close result and error channels
	close(s.resultChan)
	close(s.errorChan)
	close(s.doneChan)

	s.stats.mutex.Lock()
	s.stats.Status = "stopped"
	s.stats.mutex.Unlock()

	// Call completion callback if provided
	if s.options.CompletionCallback != nil {
		s.options.CompletionCallback(s.stats)
	}

	return nil
}

// Close gracefully shuts down the streaming delete operation
func (s *StreamingBatchDelete) Close() error {
	if !atomic.CompareAndSwapInt32(&s.stopped, 0, 1) {
		return nil // Already stopped
	}

	s.stats.mutex.Lock()
	s.stats.Status = "stopping"
	s.stats.mutex.Unlock()

	// Close input channel to signal workers to finish
	close(s.inputChan)

	// Cancel context
	s.cancel()

	// Wait for all workers to finish
	s.wg.Wait()

	// Close result and error channels
	close(s.resultChan)
	close(s.errorChan)
	close(s.doneChan)

	s.stats.mutex.Lock()
	s.stats.Status = "stopped"
	s.stats.mutex.Unlock()

	// Call completion callback if provided
	if s.options.CompletionCallback != nil {
		s.options.CompletionCallback(s.stats)
	}

	return nil
}

// Wait blocks until the streaming update operation completes
func (s *StreamingBatchUpdate) Wait() error {
	select {
	case <-s.doneChan:
		return nil
	case <-s.ctx.Done():
		return s.ctx.Err()
	}
}

// Wait blocks until the streaming delete operation completes
func (s *StreamingBatchDelete) Wait() error {
	select {
	case <-s.doneChan:
		return nil
	case <-s.ctx.Done():
		return s.ctx.Err()
	}
}
