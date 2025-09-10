//go:build example
// +build example

package main

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/xDarkicex/libravdb/libravdb"
)

func main() {
	// Create database
	db, err := libravdb.New(libravdb.WithStoragePath("./streaming_example_data"))
	if err != nil {
		log.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	// Create collection
	collection, err := db.CreateCollection(
		context.Background(),
		"streaming_example",
		libravdb.WithDimension(128),
		libravdb.WithMetric(libravdb.CosineDistance),
	)
	if err != nil {
		log.Fatalf("Failed to create collection: %v", err)
	}

	// Example 1: Basic streaming with progress tracking
	fmt.Println("=== Example 1: Basic Streaming ===")
	basicStreamingExample(collection)

	// Example 2: Streaming with backpressure handling
	fmt.Println("\n=== Example 2: Backpressure Handling ===")
	backpressureExample(collection)

	// Example 3: Streaming from a channel
	fmt.Println("\n=== Example 3: Channel-based Streaming ===")
	channelStreamingExample(collection)

	// Example 4: Error handling and recovery
	fmt.Println("\n=== Example 4: Error Handling ===")
	errorHandlingExample(collection)

	fmt.Println("\nAll streaming examples completed successfully!")
}

func basicStreamingExample(collection *libravdb.Collection) {
	// Configure streaming options
	opts := libravdb.DefaultStreamingOptions()
	opts.BufferSize = 1000
	opts.ChunkSize = 100
	opts.MaxConcurrency = 4
	opts.Timeout = 30 * time.Second

	// Add progress callback
	opts.ProgressCallback = func(stats *libravdb.StreamingStats) {
		fmt.Printf("Progress: %d/%d processed (%.1f%%), %.1f items/sec\n",
			stats.TotalProcessed, stats.TotalReceived,
			float64(stats.TotalProcessed)/float64(stats.TotalReceived)*100,
			stats.ItemsPerSecond)
	}

	// Create streaming insert
	stream := collection.NewStreamingBatchInsert(opts)

	// Start streaming
	if err := stream.Start(); err != nil {
		log.Fatalf("Failed to start streaming: %v", err)
	}
	defer stream.Close()

	// Generate and send test data
	fmt.Println("Sending 500 vectors...")
	for i := 0; i < 500; i++ {
		vector := generateRandomVector(128)
		entry := &libravdb.VectorEntry{
			ID:     fmt.Sprintf("basic_entry_%d", i),
			Vector: vector,
			Metadata: map[string]interface{}{
				"batch":     i / 100,
				"timestamp": time.Now().Unix(),
				"category":  fmt.Sprintf("cat_%d", i%5),
			},
		}

		if err := stream.Send(entry); err != nil {
			fmt.Printf("Failed to send entry %d: %v\n", i, err)
		}
	}

	// Wait for processing to complete
	time.Sleep(2 * time.Second)

	// Get final statistics
	stats := stream.Stats()
	fmt.Printf("Final stats: Received=%d, Processed=%d, Successful=%d, Failed=%d\n",
		stats.TotalReceived, stats.TotalProcessed, stats.TotalSuccessful, stats.TotalFailed)
}

func backpressureExample(collection *libravdb.Collection) {
	// Configure with small buffer to demonstrate backpressure
	opts := libravdb.DefaultStreamingOptions()
	opts.BufferSize = 10
	opts.ChunkSize = 50 // Larger than buffer
	opts.BackpressureThreshold = 0.8
	opts.EnableBackpressure = true
	opts.Timeout = 30 * time.Second

	// Track backpressure events
	backpressureCount := 0
	opts.ProgressCallback = func(stats *libravdb.StreamingStats) {
		if stats.BackpressureActive {
			backpressureCount++
			fmt.Printf("Backpressure active! Buffer utilization: %.1f%%\n",
				stats.BufferUtilization*100)
		}
	}

	stream := collection.NewStreamingBatchInsert(opts)

	if err := stream.Start(); err != nil {
		log.Fatalf("Failed to start streaming: %v", err)
	}
	defer stream.Close()

	// Send data rapidly to trigger backpressure
	fmt.Println("Sending data rapidly to trigger backpressure...")
	successCount := 0
	backpressureErrors := 0

	for i := 0; i < 100; i++ {
		vector := generateRandomVector(128)
		entry := &libravdb.VectorEntry{
			ID:     fmt.Sprintf("bp_entry_%d", i),
			Vector: vector,
		}

		err := stream.Send(entry)
		if err == libravdb.ErrBackpressureActive {
			backpressureErrors++
			// Wait a bit and retry
			time.Sleep(10 * time.Millisecond)
			continue
		} else if err != nil {
			fmt.Printf("Unexpected error: %v\n", err)
		} else {
			successCount++
		}
	}

	fmt.Printf("Sent %d entries successfully, encountered %d backpressure events\n",
		successCount, backpressureErrors)
}

func channelStreamingExample(collection *libravdb.Collection) {
	// Create a channel for data
	dataChan := make(chan *libravdb.VectorEntry, 100)

	// Start a goroutine to generate data
	go func() {
		defer close(dataChan)
		for i := 0; i < 200; i++ {
			vector := generateRandomVector(128)
			entry := &libravdb.VectorEntry{
				ID:     fmt.Sprintf("channel_entry_%d", i),
				Vector: vector,
				Metadata: map[string]interface{}{
					"source": "channel",
					"index":  i,
				},
			}
			dataChan <- entry
			time.Sleep(5 * time.Millisecond) // Simulate data generation delay
		}
	}()

	// Create streaming reader from channel
	reader := libravdb.NewChannelStreamingReader(dataChan)

	// Configure streaming options
	opts := libravdb.DefaultStreamingOptions()
	opts.BufferSize = 50
	opts.ChunkSize = 25

	// Create stream from reader
	stream, err := collection.StreamFromReader(reader, opts)
	if err != nil {
		log.Fatalf("Failed to create stream from reader: %v", err)
	}
	defer stream.Close()

	// Monitor results
	go func() {
		for result := range stream.Results() {
			fmt.Printf("Batch %d: %d processed, %d successful, %d failed\n",
				result.BatchID, result.Processed, result.Successful, result.Failed)
		}
	}()

	// Wait for completion
	fmt.Println("Processing data from channel...")
	time.Sleep(3 * time.Second)

	stats := stream.Stats()
	fmt.Printf("Channel streaming completed: %d received, %d processed\n",
		stats.TotalReceived, stats.TotalProcessed)
}

func errorHandlingExample(collection *libravdb.Collection) {
	opts := libravdb.DefaultStreamingOptions()
	opts.BufferSize = 20
	opts.ChunkSize = 5

	// Track errors
	errorCount := 0
	opts.ErrorCallback = func(err error, entry *libravdb.VectorEntry) {
		errorCount++
		fmt.Printf("Error processing entry %s: %v\n", entry.ID, err)
	}

	stream := collection.NewStreamingBatchInsert(opts)

	if err := stream.Start(); err != nil {
		log.Fatalf("Failed to start streaming: %v", err)
	}
	defer stream.Close()

	// Send mix of valid and invalid entries
	fmt.Println("Sending mix of valid and invalid entries...")
	for i := 0; i < 20; i++ {
		var vector []float32
		var id string

		if i%4 == 0 {
			// Invalid entry - wrong dimension
			vector = generateRandomVector(64) // Wrong dimension
			id = fmt.Sprintf("invalid_entry_%d", i)
		} else {
			// Valid entry
			vector = generateRandomVector(128)
			id = fmt.Sprintf("valid_entry_%d", i)
		}

		entry := &libravdb.VectorEntry{
			ID:     id,
			Vector: vector,
		}

		if err := stream.Send(entry); err != nil {
			fmt.Printf("Failed to send entry %s: %v\n", id, err)
		}
	}

	// Wait for processing
	time.Sleep(1 * time.Second)

	stats := stream.Stats()
	fmt.Printf("Error handling results: %d errors detected, %d successful, %d failed\n",
		errorCount, stats.TotalSuccessful, stats.TotalFailed)

	// Print error statistics
	fmt.Println("Error breakdown by type:")
	for errorType, count := range stats.ErrorsByType {
		fmt.Printf("  %s: %d\n", errorType, count)
	}
}

func generateRandomVector(dimension int) []float32 {
	vector := make([]float32, dimension)
	for i := range vector {
		vector[i] = rand.Float32()*2 - 1 // Random values between -1 and 1
	}
	return vector
}
