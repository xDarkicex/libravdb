package libravdb

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Helper function to generate unique collection names for tests
func uniqueCollectionName(prefix string) string {
	return fmt.Sprintf("%s_%d", prefix, time.Now().UnixNano())
}

func TestStreamingBatchInsert_Basic(t *testing.T) {
	db, err := New(WithStoragePath(":memory:"))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(context.Background(), uniqueCollectionName("test_streaming_basic"), WithDimension(3), WithMetric(CosineDistance))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Create streaming insert with small buffer for testing
	opts := DefaultStreamingOptions()
	opts.BufferSize = 10
	opts.ChunkSize = 5
	opts.MaxConcurrency = 2

	stream := collection.NewStreamingBatchInsert(opts)

	// Start streaming
	if err := stream.Start(); err != nil {
		t.Fatalf("Failed to start streaming: %v", err)
	}
	defer stream.Close()

	// Send test data
	testEntries := []*VectorEntry{
		{ID: "1", Vector: []float32{1.0, 2.0, 3.0}},
		{ID: "2", Vector: []float32{4.0, 5.0, 6.0}},
		{ID: "3", Vector: []float32{7.0, 8.0, 9.0}},
		{ID: "4", Vector: []float32{10.0, 11.0, 12.0}},
		{ID: "5", Vector: []float32{13.0, 14.0, 15.0}},
	}

	// Send entries
	for _, entry := range testEntries {
		if err := stream.Send(entry); err != nil {
			t.Errorf("Failed to send entry %s: %v", entry.ID, err)
		}
	}

	// Wait a bit for processing
	time.Sleep(100 * time.Millisecond)

	// Check statistics
	stats := stream.Stats()
	if stats.TotalReceived != int64(len(testEntries)) {
		t.Errorf("Expected %d received, got %d", len(testEntries), stats.TotalReceived)
	}

	// Close and wait for completion
	if err := stream.Close(); err != nil {
		t.Errorf("Failed to close stream: %v", err)
	}
}

func TestStreamingBatchInsert_Backpressure(t *testing.T) {
	db, err := New(WithStoragePath(":memory:"))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(context.Background(), uniqueCollectionName("test_streaming_backpressure"), WithDimension(3), WithMetric(CosineDistance))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Create streaming insert with very small buffer to trigger backpressure
	opts := DefaultStreamingOptions()
	opts.BufferSize = 2
	opts.ChunkSize = 10              // Larger than buffer to prevent immediate processing
	opts.BackpressureThreshold = 0.5 // 50% threshold
	opts.EnableBackpressure = true

	stream := collection.NewStreamingBatchInsert(opts)

	if err := stream.Start(); err != nil {
		t.Fatalf("Failed to start streaming: %v", err)
	}
	defer stream.Close()

	// Fill buffer to trigger backpressure
	entry1 := &VectorEntry{ID: "1", Vector: []float32{1.0, 2.0, 3.0}}
	entry2 := &VectorEntry{ID: "2", Vector: []float32{4.0, 5.0, 6.0}}
	entry3 := &VectorEntry{ID: "3", Vector: []float32{7.0, 8.0, 9.0}}

	// First two should succeed
	if err := stream.Send(entry1); err != nil {
		t.Errorf("First send should succeed: %v", err)
	}
	if err := stream.Send(entry2); err != nil {
		t.Errorf("Second send should succeed: %v", err)
	}

	// Third should trigger backpressure
	err = stream.Send(entry3)
	if err != ErrBackpressureActive {
		t.Errorf("Expected backpressure error, got: %v", err)
	}

	// Check backpressure status
	stats := stream.Stats()
	if !stats.BackpressureActive {
		t.Error("Backpressure should be active")
	}
}

func TestStreamingBatchInsert_ContextCancellation(t *testing.T) {
	db, err := New(WithStoragePath(":memory:"))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(context.Background(), uniqueCollectionName("test_streaming_cancellation"), WithDimension(3), WithMetric(CosineDistance))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Create streaming insert with timeout
	opts := DefaultStreamingOptions()
	opts.Timeout = 100 * time.Millisecond
	opts.BufferSize = 10
	opts.ChunkSize = 5

	stream := collection.NewStreamingBatchInsert(opts)

	if err := stream.Start(); err != nil {
		t.Fatalf("Failed to start streaming: %v", err)
	}
	defer stream.Close()

	// Wait for timeout
	err = stream.Wait()
	if err != context.DeadlineExceeded {
		t.Errorf("Expected timeout error, got: %v", err)
	}

	// Verify stream is stopped
	entry := &VectorEntry{ID: "1", Vector: []float32{1.0, 2.0, 3.0}}
	err = stream.Send(entry)
	if err == nil {
		t.Error("Send should fail after timeout")
	}
}

func TestStreamingBatchInsert_ProgressCallback(t *testing.T) {
	db, err := New(WithStoragePath(":memory:"))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(context.Background(), uniqueCollectionName("test_streaming_progress"), WithDimension(3), WithMetric(CosineDistance))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	var progressCallCount int32
	var lastStats *StreamingStats

	opts := DefaultStreamingOptions()
	opts.BufferSize = 10
	opts.ChunkSize = 3
	opts.ProgressCallback = func(stats *StreamingStats) {
		atomic.AddInt32(&progressCallCount, 1)
		lastStats = stats
	}

	stream := collection.NewStreamingBatchInsert(opts)

	if err := stream.Start(); err != nil {
		t.Fatalf("Failed to start streaming: %v", err)
	}
	defer stream.Close()

	// Send some entries
	for i := 0; i < 6; i++ {
		entry := &VectorEntry{
			ID:     fmt.Sprintf("entry_%d", i),
			Vector: []float32{float32(i), float32(i + 1), float32(i + 2)},
		}
		if err := stream.Send(entry); err != nil {
			t.Errorf("Failed to send entry %d: %v", i, err)
		}
	}

	// Wait for processing and progress updates
	time.Sleep(2 * time.Second)

	// Check that progress callback was called
	callCount := atomic.LoadInt32(&progressCallCount)
	if callCount == 0 {
		t.Error("Progress callback should have been called")
	}

	if lastStats == nil {
		t.Error("Should have received progress stats")
	} else {
		if lastStats.TotalReceived != 6 {
			t.Errorf("Expected 6 received, got %d", lastStats.TotalReceived)
		}
	}
}

func TestStreamingBatchInsert_ErrorHandling(t *testing.T) {
	db, err := New(WithStoragePath(":memory:"))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(context.Background(), uniqueCollectionName("test_streaming_errors"), WithDimension(3), WithMetric(CosineDistance))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	var errorCallCount int32
	var lastError error

	opts := DefaultStreamingOptions()
	opts.BufferSize = 10
	opts.ChunkSize = 2
	opts.ErrorCallback = func(err error, entry *VectorEntry) {
		atomic.AddInt32(&errorCallCount, 1)
		lastError = err
	}

	stream := collection.NewStreamingBatchInsert(opts)

	if err := stream.Start(); err != nil {
		t.Fatalf("Failed to start streaming: %v", err)
	}
	defer stream.Close()

	// Send valid entry
	validEntry := &VectorEntry{ID: "valid", Vector: []float32{1.0, 2.0, 3.0}}
	if err := stream.Send(validEntry); err != nil {
		t.Errorf("Failed to send valid entry: %v", err)
	}

	// Send invalid entry (wrong dimension)
	invalidEntry := &VectorEntry{ID: "invalid", Vector: []float32{1.0, 2.0}} // Wrong dimension
	if err := stream.Send(invalidEntry); err != nil {
		t.Errorf("Failed to send invalid entry: %v", err)
	}

	// Wait for processing
	time.Sleep(200 * time.Millisecond)

	// Check error handling
	callCount := atomic.LoadInt32(&errorCallCount)
	if callCount == 0 {
		t.Error("Error callback should have been called for invalid entry")
	}

	if lastError == nil {
		t.Error("Should have received an error")
	}

	// Check error statistics
	stats := stream.Stats()
	if len(stats.ErrorsByType) == 0 {
		t.Error("Should have error statistics")
	}
}

func TestStreamingBatchInsert_ConcurrentSend(t *testing.T) {
	db, err := New(WithStoragePath(":memory:"))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(context.Background(), uniqueCollectionName("test_streaming_concurrent"), WithDimension(3), WithMetric(CosineDistance))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	opts := DefaultStreamingOptions()
	opts.BufferSize = 100
	opts.ChunkSize = 10
	opts.MaxConcurrency = 4

	stream := collection.NewStreamingBatchInsert(opts)

	if err := stream.Start(); err != nil {
		t.Fatalf("Failed to start streaming: %v", err)
	}
	defer stream.Close()

	// Send entries concurrently from multiple goroutines
	const numGoroutines = 5
	const entriesPerGoroutine = 20
	var wg sync.WaitGroup

	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		go func(goroutineID int) {
			defer wg.Done()
			for i := 0; i < entriesPerGoroutine; i++ {
				entry := &VectorEntry{
					ID:     fmt.Sprintf("g%d_entry_%d", goroutineID, i),
					Vector: []float32{float32(goroutineID), float32(i), float32(goroutineID + i)},
				}
				if err := stream.Send(entry); err != nil {
					t.Errorf("Goroutine %d failed to send entry %d: %v", goroutineID, i, err)
				}
			}
		}(g)
	}

	wg.Wait()

	// Wait for processing
	time.Sleep(500 * time.Millisecond)

	// Check statistics
	stats := stream.Stats()
	expectedTotal := int64(numGoroutines * entriesPerGoroutine)
	if stats.TotalReceived != expectedTotal {
		t.Errorf("Expected %d received, got %d", expectedTotal, stats.TotalReceived)
	}
}

func TestStreamingBatchInsert_SendBatch(t *testing.T) {
	db, err := New(WithStoragePath(":memory:"))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(context.Background(), uniqueCollectionName("test_streaming_sendbatch"), WithDimension(3), WithMetric(CosineDistance))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	opts := DefaultStreamingOptions()
	opts.BufferSize = 20
	opts.ChunkSize = 5

	stream := collection.NewStreamingBatchInsert(opts)

	if err := stream.Start(); err != nil {
		t.Fatalf("Failed to start streaming: %v", err)
	}
	defer stream.Close()

	// Create batch of entries
	entries := make([]*VectorEntry, 10)
	for i := 0; i < 10; i++ {
		entries[i] = &VectorEntry{
			ID:     fmt.Sprintf("batch_entry_%d", i),
			Vector: []float32{float32(i), float32(i + 1), float32(i + 2)},
		}
	}

	// Send batch
	if err := stream.SendBatch(entries); err != nil {
		t.Errorf("Failed to send batch: %v", err)
	}

	// Wait for processing
	time.Sleep(200 * time.Millisecond)

	// Check statistics
	stats := stream.Stats()
	if stats.TotalReceived != 10 {
		t.Errorf("Expected 10 received, got %d", stats.TotalReceived)
	}
}

func TestChannelStreamingReader(t *testing.T) {
	// Create a channel with test data
	ch := make(chan *VectorEntry, 5)
	testEntries := []*VectorEntry{
		{ID: "1", Vector: []float32{1.0, 2.0, 3.0}},
		{ID: "2", Vector: []float32{4.0, 5.0, 6.0}},
		{ID: "3", Vector: []float32{7.0, 8.0, 9.0}},
	}

	// Send test data
	for _, entry := range testEntries {
		ch <- entry
	}
	close(ch)

	// Create reader
	reader := NewChannelStreamingReader(ch)
	defer reader.Close()

	// Read all entries
	var readEntries []*VectorEntry
	for {
		entry, err := reader.Read()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			t.Errorf("Unexpected error reading: %v", err)
		}
		readEntries = append(readEntries, entry)
	}

	// Verify entries
	if len(readEntries) != len(testEntries) {
		t.Errorf("Expected %d entries, got %d", len(testEntries), len(readEntries))
	}

	for i, entry := range readEntries {
		if entry.ID != testEntries[i].ID {
			t.Errorf("Entry %d ID mismatch: expected %s, got %s", i, testEntries[i].ID, entry.ID)
		}
	}
}

func TestStreamFromReader(t *testing.T) {
	db, err := New(WithStoragePath(":memory:"))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(context.Background(), uniqueCollectionName("test_streaming_reader"), WithDimension(3), WithMetric(CosineDistance))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Create a channel with test data
	ch := make(chan *VectorEntry, 10)
	testEntries := []*VectorEntry{
		{ID: "1", Vector: []float32{1.0, 2.0, 3.0}},
		{ID: "2", Vector: []float32{4.0, 5.0, 6.0}},
		{ID: "3", Vector: []float32{7.0, 8.0, 9.0}},
		{ID: "4", Vector: []float32{10.0, 11.0, 12.0}},
		{ID: "5", Vector: []float32{13.0, 14.0, 15.0}},
	}

	// Send test data in a goroutine
	go func() {
		for _, entry := range testEntries {
			ch <- entry
		}
		close(ch)
	}()

	// Create reader and stream
	reader := NewChannelStreamingReader(ch)
	opts := DefaultStreamingOptions()
	opts.BufferSize = 10
	opts.ChunkSize = 3

	stream, err := collection.StreamFromReader(reader, opts)
	if err != nil {
		t.Fatalf("Failed to create stream from reader: %v", err)
	}
	defer stream.Close()

	// Wait for processing
	time.Sleep(500 * time.Millisecond)

	// Check statistics
	stats := stream.Stats()
	if stats.TotalReceived != int64(len(testEntries)) {
		t.Errorf("Expected %d received, got %d", len(testEntries), stats.TotalReceived)
	}
}

func TestStreamingStats_ThreadSafety(t *testing.T) {
	db, err := New(WithStoragePath(":memory:"))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(context.Background(), uniqueCollectionName("test_streaming_threadsafety"), WithDimension(3), WithMetric(CosineDistance))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	opts := DefaultStreamingOptions()
	opts.BufferSize = 100
	opts.ChunkSize = 10

	stream := collection.NewStreamingBatchInsert(opts)

	if err := stream.Start(); err != nil {
		t.Fatalf("Failed to start streaming: %v", err)
	}
	defer stream.Close()

	// Concurrently read stats while sending data
	var wg sync.WaitGroup
	const numReaders = 5
	const numSenders = 3
	const entriesPerSender = 20

	// Start stats readers
	for i := 0; i < numReaders; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				stats := stream.Stats()
				_ = stats // Use stats to avoid unused variable
				time.Sleep(10 * time.Millisecond)
			}
		}()
	}

	// Start data senders
	for i := 0; i < numSenders; i++ {
		wg.Add(1)
		go func(senderID int) {
			defer wg.Done()
			for j := 0; j < entriesPerSender; j++ {
				entry := &VectorEntry{
					ID:     fmt.Sprintf("sender_%d_entry_%d", senderID, j),
					Vector: []float32{float32(senderID), float32(j), float32(senderID + j)},
				}
				if err := stream.Send(entry); err != nil {
					// Ignore backpressure errors in this test
					if err != ErrBackpressureActive {
						t.Errorf("Sender %d failed to send entry %d: %v", senderID, j, err)
					}
				}
				time.Sleep(5 * time.Millisecond)
			}
		}(i)
	}

	wg.Wait()

	// Final stats check
	finalStats := stream.Stats()
	if finalStats.TotalReceived == 0 {
		t.Error("Should have received some entries")
	}
}

func TestBackpressureController(t *testing.T) {
	bp := &BackpressureController{
		enabled:        true,
		threshold:      0.7,
		maxMemoryUsage: 1000,
		bufferSize:     10,
	}

	// Initially no backpressure
	if bp.ShouldApplyBackpressure() {
		t.Error("Should not apply backpressure initially")
	}

	// Simulate buffer filling up
	atomic.StoreInt32(&bp.currentBufferSize, 8) // 80% of buffer

	if !bp.ShouldApplyBackpressure() {
		t.Error("Should apply backpressure when buffer is 80% full")
	}

	// Test memory pressure
	atomic.StoreInt32(&bp.currentBufferSize, 5) // Back to 50%
	bp.UpdateMemoryUsage(1500)                  // Exceed memory limit

	if !bp.ShouldApplyBackpressure() {
		t.Error("Should apply backpressure when memory usage exceeds limit")
	}

	// Test disabled backpressure
	bp.enabled = false
	if bp.ShouldApplyBackpressure() {
		t.Error("Should not apply backpressure when disabled")
	}
}

// Benchmark tests for streaming performance
func BenchmarkStreamingBatchInsert_Send(b *testing.B) {
	db, err := New(WithStoragePath(":memory:"))
	if err != nil {
		b.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(context.Background(), "test_streaming_bench_send", WithDimension(128), WithMetric(CosineDistance))
	if err != nil {
		b.Fatalf("Failed to create collection: %v", err)
	}

	opts := DefaultStreamingOptions()
	opts.BufferSize = 10000
	opts.ChunkSize = 1000
	opts.EnableBackpressure = false // Disable for benchmark

	stream := collection.NewStreamingBatchInsert(opts)
	if err := stream.Start(); err != nil {
		b.Fatalf("Failed to start streaming: %v", err)
	}
	defer stream.Close()

	// Create test entry
	vector := make([]float32, 128)
	for i := range vector {
		vector[i] = float32(i)
	}
	entry := &VectorEntry{ID: "test", Vector: vector}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			stream.Send(entry)
		}
	})
}

func BenchmarkStreamingBatchInsert_SendBatch(b *testing.B) {
	db, err := New(WithStoragePath(":memory:"))
	if err != nil {
		b.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(context.Background(), "test_streaming_bench_batch", WithDimension(128), WithMetric(CosineDistance))
	if err != nil {
		b.Fatalf("Failed to create collection: %v", err)
	}

	opts := DefaultStreamingOptions()
	opts.BufferSize = 10000
	opts.ChunkSize = 1000
	opts.EnableBackpressure = false

	stream := collection.NewStreamingBatchInsert(opts)
	if err := stream.Start(); err != nil {
		b.Fatalf("Failed to start streaming: %v", err)
	}
	defer stream.Close()

	// Create batch of test entries
	batchSize := 100
	entries := make([]*VectorEntry, batchSize)
	for i := 0; i < batchSize; i++ {
		vector := make([]float32, 128)
		for j := range vector {
			vector[j] = float32(j + i)
		}
		entries[i] = &VectorEntry{
			ID:     fmt.Sprintf("test_%d", i),
			Vector: vector,
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stream.SendBatch(entries)
	}
}
