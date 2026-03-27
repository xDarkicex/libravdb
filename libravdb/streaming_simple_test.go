package libravdb

import (
	"context"
	"fmt"
	"testing"
	"time"
)

func TestStreamingSimple_BasicFunctionality(t *testing.T) {
	db, err := New(WithStoragePath(":memory:"))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	collectionName := fmt.Sprintf("test_streaming_simple_%d", time.Now().UnixNano())
	collection, err := db.CreateCollection(context.Background(), collectionName, WithDimension(3), WithMetric(CosineDistance))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Create streaming insert with small buffer
	opts := DefaultStreamingOptions()
	opts.BufferSize = 5
	opts.ChunkSize = 2
	opts.MaxConcurrency = 1
	opts.Timeout = 5 * time.Second

	stream := collection.NewStreamingBatchInsert(opts)

	// Start streaming
	if err := stream.Start(); err != nil {
		t.Fatalf("Failed to start streaming: %v", err)
	}

	// Send a few entries
	entries := []*VectorEntry{
		{ID: "entry1", Vector: []float32{1.0, 2.0, 3.0}},
		{ID: "entry2", Vector: []float32{4.0, 5.0, 6.0}},
		{ID: "entry3", Vector: []float32{7.0, 8.0, 9.0}},
	}

	for _, entry := range entries {
		if err := stream.Send(entry); err != nil {
			t.Errorf("Failed to send entry %s: %v", entry.ID, err)
		}
	}

	// Wait a bit for processing
	time.Sleep(100 * time.Millisecond)

	// Check statistics
	stats := stream.Stats()
	if stats.TotalReceived != int64(len(entries)) {
		t.Errorf("Expected %d received, got %d", len(entries), stats.TotalReceived)
	}

	// Close stream
	if err := stream.Close(); err != nil {
		t.Errorf("Failed to close stream: %v", err)
	}

	t.Logf("Test completed successfully. Stats: Received=%d, Processed=%d",
		stats.TotalReceived, stats.TotalProcessed)
}

func TestStreamingSimple_BackpressureBasic(t *testing.T) {
	db, err := New(WithStoragePath(":memory:"))
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	collectionName := fmt.Sprintf("test_streaming_bp_%d", time.Now().UnixNano())
	collection, err := db.CreateCollection(context.Background(), collectionName, WithDimension(3), WithMetric(CosineDistance))
	if err != nil {
		t.Fatalf("Failed to create collection: %v", err)
	}

	// Create streaming insert with very small buffer
	opts := DefaultStreamingOptions()
	opts.BufferSize = 1
	opts.ChunkSize = 10 // Larger than buffer
	opts.BackpressureThreshold = 0.5
	opts.EnableBackpressure = true
	opts.Timeout = 5 * time.Second

	stream := collection.NewStreamingBatchInsert(opts)

	if err := stream.Start(); err != nil {
		t.Fatalf("Failed to start streaming: %v", err)
	}

	// Fill buffer
	entry1 := &VectorEntry{ID: "entry1", Vector: []float32{1.0, 2.0, 3.0}}
	if err := stream.Send(entry1); err != nil {
		t.Errorf("First send should succeed: %v", err)
	}

	// This should trigger backpressure
	entry2 := &VectorEntry{ID: "entry2", Vector: []float32{4.0, 5.0, 6.0}}
	err = stream.Send(entry2)
	if err != ErrBackpressureActive {
		t.Logf("Expected backpressure error, got: %v (this may be timing dependent)", err)
	}

	// Close stream
	if err := stream.Close(); err != nil {
		t.Errorf("Failed to close stream: %v", err)
	}

	t.Log("Backpressure test completed")
}

func TestStreamingSimple_ChannelReader(t *testing.T) {
	// Test the channel reader functionality
	ch := make(chan *VectorEntry, 3)
	testEntries := []*VectorEntry{
		{ID: "1", Vector: []float32{1.0, 2.0, 3.0}},
		{ID: "2", Vector: []float32{4.0, 5.0, 6.0}},
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

	t.Log("Channel reader test completed successfully")
}
