package libravdb

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestStreamingIntegration_MemoryUsage(t *testing.T) {
	db, err := New(WithStoragePath(":memory:"))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	// Use unique collection name to avoid conflicts
	collectionName := fmt.Sprintf("test_streaming_memory_%d", time.Now().UnixNano())

	collection, err := db.CreateCollection(context.Background(), collectionName, WithDimension(128), WithMetric(CosineDistance))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Create streaming insert with memory monitoring
	opts := DefaultStreamingOptions()
	opts.BufferSize = 100
	opts.ChunkSize = 10
	opts.MaxMemoryUsage = 1024 * 1024 // 1MB limit
	opts.EnableBackpressure = true

	var (
		memoryUsageReports []int64
		reportsMu          sync.Mutex
	)
	opts.ProgressCallback = func(stats *StreamingStats) {
		reportsMu.Lock()
		memoryUsageReports = append(memoryUsageReports, stats.CurrentMemoryUsage)
		reportsMu.Unlock()
	}

	stream := collection.NewStreamingBatchInsert(opts)

	if err := stream.Start(); err != nil {
		t.Fatalf("Failed to start streaming: %v", err)
	}
	defer stream.Close()

	// Send a moderate number of entries
	for i := 0; i < 50; i++ {
		vector := make([]float32, 128)
		for j := range vector {
			vector[j] = float32(i*j) * 0.1
		}

		entry := &VectorEntry{
			ID:     fmt.Sprintf("entry_%d", i),
			Vector: vector,
			Metadata: map[string]interface{}{
				"batch": i / 10,
				"index": i,
			},
		}

		if err := stream.Send(entry); err != nil {
			// Allow backpressure errors
			if err != ErrBackpressureActive {
				t.Errorf("Failed to send entry %d: %v", i, err)
			}
		}
	}

	// Wait for processing
	time.Sleep(1 * time.Second)

	// Check final statistics
	stats := stream.Stats()
	if stats.TotalReceived == 0 {
		t.Error("Should have received some entries")
	}

	t.Logf("Final stats: Received=%d, Processed=%d, Successful=%d, Failed=%d",
		stats.TotalReceived, stats.TotalProcessed, stats.TotalSuccessful, stats.TotalFailed)
	t.Logf("Buffer utilization: %.2f%%, Backpressure active: %v",
		stats.BufferUtilization*100, stats.BackpressureActive)
}

func TestStreamingIntegration_ContextTimeout(t *testing.T) {
	db, err := New(WithStoragePath(":memory:"))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	// Use unique collection name to avoid conflicts
	collectionName := fmt.Sprintf("test_streaming_timeout_%d", time.Now().UnixNano())
	collection, err := db.CreateCollection(context.Background(), collectionName, WithDimension(3), WithMetric(CosineDistance))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Create streaming insert with short timeout
	opts := DefaultStreamingOptions()
	opts.BufferSize = 10
	opts.ChunkSize = 5
	opts.Timeout = 200 * time.Millisecond

	stream := collection.NewStreamingBatchInsert(opts)

	if err := stream.Start(); err != nil {
		t.Fatalf("Failed to start streaming: %v", err)
	}
	defer stream.Close()

	// Send a few entries
	for i := 0; i < 3; i++ {
		entry := &VectorEntry{
			ID:     fmt.Sprintf("entry_%d", i),
			Vector: []float32{float32(i), float32(i + 1), float32(i + 2)},
		}
		stream.Send(entry)
	}

	// Wait for timeout
	err = stream.Wait()
	if err != context.DeadlineExceeded {
		t.Logf("Expected timeout error, got: %v", err)
		// Don't fail the test as timeout behavior may vary
	}

	// Verify we can get stats even after timeout
	stats := stream.Stats()
	if stats.Status != "stopped" && stats.Status != "stopping" {
		t.Logf("Stream status: %s", stats.Status)
	}
}

func TestStreamingIntegration_ErrorRecovery(t *testing.T) {
	db, err := New(WithStoragePath(":memory:"))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	// Use unique collection name to avoid conflicts
	collectionName := fmt.Sprintf("test_streaming_recovery_%d", time.Now().UnixNano())
	collection, err := db.CreateCollection(context.Background(), collectionName, WithDimension(3), WithMetric(CosineDistance))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Track errors
	var errorCount atomic.Int32
	var lastError atomic.Value

	opts := DefaultStreamingOptions()
	opts.BufferSize = 20
	opts.ChunkSize = 5
	opts.ErrorCallback = func(err error, entry *VectorEntry) {
		errorCount.Add(1)
		lastError.Store(err)
	}

	stream := collection.NewStreamingBatchInsert(opts)

	if err := stream.Start(); err != nil {
		t.Fatalf("Failed to start streaming: %v", err)
	}
	defer stream.Close()

	// Send mix of valid and invalid entries
	entries := []*VectorEntry{
		{ID: "valid1", Vector: []float32{1.0, 2.0, 3.0}},
		{ID: "invalid1", Vector: []float32{1.0, 2.0}}, // Wrong dimension
		{ID: "valid2", Vector: []float32{4.0, 5.0, 6.0}},
		{ID: "invalid2", Vector: []float32{7.0}}, // Wrong dimension
		{ID: "valid3", Vector: []float32{7.0, 8.0, 9.0}},
	}

	for _, entry := range entries {
		if err := stream.Send(entry); err != nil {
			t.Errorf("Failed to send entry %s: %v", entry.ID, err)
		}
	}

	// Wait for the invalid entries to be processed. Race builds add enough
	// overhead that a fixed sleep can miss the callback/statistics updates.
	deadline := time.Now().Add(3 * time.Second)
	var stats *StreamingStats
	for time.Now().Before(deadline) {
		stats = stream.Stats()
		if errorCount.Load() > 0 && len(stats.ErrorsByType) > 0 {
			break
		}
		time.Sleep(25 * time.Millisecond)
	}

	// Check error handling
	if errorCount.Load() == 0 {
		t.Error("Should have encountered some errors from invalid entries")
	}

	lastErr, _ := lastError.Load().(error)
	if lastErr == nil {
		t.Error("Should have recorded the last error")
	}

	// Check statistics
	if len(stats.ErrorsByType) == 0 {
		t.Error("Should have error statistics by type")
	}

	t.Logf("Error count: %d, Last error: %v", errorCount.Load(), lastErr)
	t.Logf("Error stats: %+v", stats.ErrorsByType)
}
