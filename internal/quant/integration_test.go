package quant

import (
	"context"
	"testing"
)

func TestProductQuantizerIntegration(t *testing.T) {
	// Test that Product Quantizer can be created through the global registry
	config := DefaultConfig(ProductQuantization)

	// Verify the type is supported
	if !IsSupported(ProductQuantization) {
		t.Fatal("ProductQuantization should be supported by global registry")
	}

	// Create quantizer through registry
	quantizer, err := Create(config)
	if err != nil {
		t.Fatalf("Failed to create quantizer through registry: %v", err)
	}

	if quantizer == nil {
		t.Fatal("Created quantizer should not be nil")
	}

	// Verify it's the correct type
	pq, ok := quantizer.(*ProductQuantizer)
	if !ok {
		t.Fatal("Created quantizer should be a ProductQuantizer")
	}

	// Test basic functionality
	if pq.IsTrained() {
		t.Error("New quantizer should not be trained")
	}

	// Generate test data and train
	vectors := generateRandomVectors(100, 16) // 16 dimensions, divisible by 4 codebooks
	ctx := context.Background()

	err = pq.Train(ctx, vectors)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	if !pq.IsTrained() {
		t.Error("Quantizer should be trained after training")
	}

	// Test compression/decompression
	testVector := generateRandomVector(16)
	compressed, err := pq.Compress(testVector)
	if err != nil {
		t.Fatalf("Compression failed: %v", err)
	}

	decompressed, err := pq.Decompress(compressed)
	if err != nil {
		t.Fatalf("Decompression failed: %v", err)
	}

	if len(decompressed) != len(testVector) {
		t.Errorf("Decompressed vector has wrong dimension: got %d, want %d",
			len(decompressed), len(testVector))
	}

	// Test compression ratio
	ratio := pq.CompressionRatio()
	if ratio <= 1.0 {
		t.Errorf("Compression ratio should be > 1.0, got %f", ratio)
	}

	// Test memory usage
	usage := pq.MemoryUsage()
	if usage <= 0 {
		t.Errorf("Memory usage should be positive, got %d", usage)
	}
}

func TestQuantizerFactoryRegistration(t *testing.T) {
	// Test that all expected quantizer types are registered
	supportedTypes := SupportedTypes()

	found := false
	for _, qType := range supportedTypes {
		if qType == ProductQuantization {
			found = true
			break
		}
	}

	if !found {
		t.Error("ProductQuantization should be in supported types")
	}

	// Test factory retrieval
	registry := NewRegistry()
	factory := NewProductQuantizerFactory()

	err := registry.Register(ProductQuantization, factory)
	if err != nil {
		t.Fatalf("Failed to register factory: %v", err)
	}

	retrievedFactory, err := registry.GetFactory(ProductQuantization)
	if err != nil {
		t.Fatalf("Failed to get factory: %v", err)
	}

	if retrievedFactory.Name() != factory.Name() {
		t.Errorf("Retrieved factory name mismatch: got %s, want %s",
			retrievedFactory.Name(), factory.Name())
	}
}
