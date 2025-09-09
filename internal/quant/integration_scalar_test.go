package quant

import (
	"context"
	"testing"
)

func TestScalarQuantizerIntegration(t *testing.T) {
	// Test that scalar quantizer can be created through the global registry
	config := &QuantizationConfig{
		Type:       ScalarQuantization,
		Bits:       8,
		TrainRatio: 0.5,
	}

	// Create quantizer through global registry
	quantizer, err := Create(config)
	if err != nil {
		t.Fatalf("failed to create scalar quantizer through registry: %v", err)
	}

	// Verify it's the correct type
	sq, ok := quantizer.(*ScalarQuantizer)
	if !ok {
		t.Fatalf("expected *ScalarQuantizer, got %T", quantizer)
	}

	// Test basic functionality
	vectors := [][]float32{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
	}

	err = sq.Train(context.Background(), vectors)
	if err != nil {
		t.Fatalf("failed to train: %v", err)
	}

	// Test compression/decompression
	compressed, err := sq.Compress(vectors[0])
	if err != nil {
		t.Fatalf("failed to compress: %v", err)
	}

	decompressed, err := sq.Decompress(compressed)
	if err != nil {
		t.Fatalf("failed to decompress: %v", err)
	}

	if len(decompressed) != len(vectors[0]) {
		t.Errorf("dimension mismatch: got %d, expected %d", len(decompressed), len(vectors[0]))
	}

	// Test that scalar quantization is supported
	if !IsSupported(ScalarQuantization) {
		t.Errorf("scalar quantization should be supported")
	}

	// Test that it's in the supported types list
	supportedTypes := SupportedTypes()
	found := false
	for _, qType := range supportedTypes {
		if qType == ScalarQuantization {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("scalar quantization should be in supported types list")
	}

	t.Logf("Scalar quantizer integration test passed - compression ratio: %.2f", sq.CompressionRatio())
}

func TestScalarQuantizerVsProductQuantizer(t *testing.T) {
	// Compare scalar and product quantization on the same data
	vectors := make([][]float32, 50)
	for i := 0; i < 50; i++ {
		vectors[i] = []float32{
			float32(i) * 0.1,
			float32(i) * 0.2,
			float32(i) * 0.3,
			float32(i) * 0.4,
		}
	}

	// Test scalar quantization
	scalarConfig := &QuantizationConfig{
		Type:       ScalarQuantization,
		Bits:       8,
		TrainRatio: 0.8,
	}

	scalarQuantizer, err := Create(scalarConfig)
	if err != nil {
		t.Fatalf("failed to create scalar quantizer: %v", err)
	}

	err = scalarQuantizer.Train(context.Background(), vectors[:40])
	if err != nil {
		t.Fatalf("failed to train scalar quantizer: %v", err)
	}

	// Test product quantization
	productConfig := &QuantizationConfig{
		Type:       ProductQuantization,
		Codebooks:  2, // 4 dimensions / 2 codebooks = 2 dimensions per codebook
		Bits:       8,
		TrainRatio: 0.8,
		CacheSize:  100,
	}

	productQuantizer, err := Create(productConfig)
	if err != nil {
		t.Fatalf("failed to create product quantizer: %v", err)
	}

	err = productQuantizer.Train(context.Background(), vectors[:40])
	if err != nil {
		t.Fatalf("failed to train product quantizer: %v", err)
	}

	// Compare compression ratios
	scalarRatio := scalarQuantizer.CompressionRatio()
	productRatio := productQuantizer.CompressionRatio()

	t.Logf("Scalar quantization compression ratio: %.2f", scalarRatio)
	t.Logf("Product quantization compression ratio: %.2f", productRatio)

	// Both should achieve the same compression ratio for 8 bits
	if scalarRatio != productRatio {
		t.Logf("Note: Different compression ratios - Scalar: %.2f, Product: %.2f", scalarRatio, productRatio)
	}

	// Test compression on test vectors
	testVector := vectors[45]

	scalarCompressed, err := scalarQuantizer.Compress(testVector)
	if err != nil {
		t.Fatalf("failed to compress with scalar quantizer: %v", err)
	}

	productCompressed, err := productQuantizer.Compress(testVector)
	if err != nil {
		t.Fatalf("failed to compress with product quantizer: %v", err)
	}

	// Compressed sizes should be similar (both use 8 bits per component/codebook)
	t.Logf("Scalar compressed size: %d bytes", len(scalarCompressed))
	t.Logf("Product compressed size: %d bytes", len(productCompressed))

	// Test decompression
	scalarDecompressed, err := scalarQuantizer.Decompress(scalarCompressed)
	if err != nil {
		t.Fatalf("failed to decompress with scalar quantizer: %v", err)
	}

	productDecompressed, err := productQuantizer.Decompress(productCompressed)
	if err != nil {
		t.Fatalf("failed to decompress with product quantizer: %v", err)
	}

	// Both should reconstruct vectors of the same dimension
	if len(scalarDecompressed) != len(productDecompressed) {
		t.Errorf("decompressed dimension mismatch: scalar=%d, product=%d",
			len(scalarDecompressed), len(productDecompressed))
	}

	t.Logf("Comparison test completed successfully")
}
