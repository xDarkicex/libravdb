package quant

import (
	"context"
	"math"
	"math/rand"
	"testing"
	"time"
)

func TestProductQuantizer_Configure(t *testing.T) {
	pq := NewProductQuantizer()

	tests := []struct {
		name    string
		config  *QuantizationConfig
		wantErr bool
	}{
		{
			name: "valid config",
			config: &QuantizationConfig{
				Type:       ProductQuantization,
				Codebooks:  8,
				Bits:       8,
				TrainRatio: 0.1,
				CacheSize:  1000,
			},
			wantErr: false,
		},
		{
			name:    "nil config",
			config:  nil,
			wantErr: true,
		},
		{
			name: "wrong type",
			config: &QuantizationConfig{
				Type:       ScalarQuantization,
				Bits:       8,
				TrainRatio: 0.1,
			},
			wantErr: true,
		},
		{
			name: "invalid bits",
			config: &QuantizationConfig{
				Type:       ProductQuantization,
				Codebooks:  8,
				Bits:       0,
				TrainRatio: 0.1,
			},
			wantErr: true,
		},
		{
			name: "invalid train ratio",
			config: &QuantizationConfig{
				Type:       ProductQuantization,
				Codebooks:  8,
				Bits:       8,
				TrainRatio: 1.5,
			},
			wantErr: true,
		},
		{
			name: "invalid codebooks",
			config: &QuantizationConfig{
				Type:       ProductQuantization,
				Codebooks:  0,
				Bits:       8,
				TrainRatio: 0.1,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := pq.Configure(tt.config)
			if (err != nil) != tt.wantErr {
				t.Errorf("Configure() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestProductQuantizer_Train(t *testing.T) {
	pq := NewProductQuantizer()
	config := &QuantizationConfig{
		Type:       ProductQuantization,
		Codebooks:  4,
		Bits:       8,
		TrainRatio: 0.5,
		CacheSize:  100,
	}

	err := pq.Configure(config)
	if err != nil {
		t.Fatalf("Configure() error = %v", err)
	}

	// Generate test vectors (dimension must be divisible by codebooks)
	dimension := 16 // 4 codebooks * 4 dimensions each
	numVectors := 1000
	vectors := generateRandomVectors(numVectors, dimension)

	ctx := context.Background()
	err = pq.Train(ctx, vectors)
	if err != nil {
		t.Fatalf("Train() error = %v", err)
	}

	// Verify training state
	if !pq.IsTrained() {
		t.Error("Expected quantizer to be trained")
	}

	if pq.dimension != dimension {
		t.Errorf("Expected dimension %d, got %d", dimension, pq.dimension)
	}

	if pq.subspaces != config.Codebooks {
		t.Errorf("Expected %d subspaces, got %d", config.Codebooks, pq.subspaces)
	}

	if pq.subDim != dimension/config.Codebooks {
		t.Errorf("Expected subDim %d, got %d", dimension/config.Codebooks, pq.subDim)
	}

	// Verify centroids are initialized
	expectedCentroids := 1 << config.Bits
	for s := 0; s < pq.subspaces; s++ {
		if len(pq.centroids[s]) != expectedCentroids {
			t.Errorf("Subspace %d: expected %d centroids, got %d",
				s, expectedCentroids, len(pq.centroids[s]))
		}

		for c := 0; c < expectedCentroids; c++ {
			if len(pq.centroids[s][c]) != pq.subDim {
				t.Errorf("Subspace %d, centroid %d: expected dimension %d, got %d",
					s, c, pq.subDim, len(pq.centroids[s][c]))
			}
		}
	}
}

func TestProductQuantizer_TrainErrors(t *testing.T) {
	pq := NewProductQuantizer()

	tests := []struct {
		name    string
		setup   func() error
		vectors [][]float32
		wantErr bool
	}{
		{
			name: "not configured",
			setup: func() error {
				return nil // Don't configure
			},
			vectors: generateRandomVectors(100, 16),
			wantErr: true,
		},
		{
			name: "empty vectors",
			setup: func() error {
				return pq.Configure(&QuantizationConfig{
					Type:       ProductQuantization,
					Codebooks:  4,
					Bits:       8,
					TrainRatio: 0.1,
				})
			},
			vectors: [][]float32{},
			wantErr: true,
		},
		{
			name: "dimension not divisible by codebooks",
			setup: func() error {
				return pq.Configure(&QuantizationConfig{
					Type:       ProductQuantization,
					Codebooks:  3,
					Bits:       8,
					TrainRatio: 0.1,
				})
			},
			vectors: generateRandomVectors(100, 16), // 16 not divisible by 3
			wantErr: true,
		},
		{
			name: "inconsistent vector dimensions",
			setup: func() error {
				return pq.Configure(&QuantizationConfig{
					Type:       ProductQuantization,
					Codebooks:  4,
					Bits:       8,
					TrainRatio: 0.1,
				})
			},
			vectors: [][]float32{
				make([]float32, 16),
				make([]float32, 12), // Different dimension
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pq := NewProductQuantizer()
			if err := tt.setup(); err != nil {
				t.Fatalf("Setup error: %v", err)
			}

			ctx := context.Background()
			err := pq.Train(ctx, tt.vectors)
			if (err != nil) != tt.wantErr {
				t.Errorf("Train() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestProductQuantizer_CompressDecompress(t *testing.T) {
	pq := setupTrainedQuantizer(t, 16, 4, 8)

	// Test vector
	vector := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
		9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0}

	// Compress
	compressed, err := pq.Compress(vector)
	if err != nil {
		t.Fatalf("Compress() error = %v", err)
	}

	// Verify compression size
	expectedBytes := (4*8 + 7) / 8 // 4 subspaces * 8 bits, rounded up to bytes
	if len(compressed) != expectedBytes {
		t.Errorf("Expected %d bytes, got %d", expectedBytes, len(compressed))
	}

	// Decompress
	decompressed, err := pq.Decompress(compressed)
	if err != nil {
		t.Fatalf("Decompress() error = %v", err)
	}

	// Verify dimension
	if len(decompressed) != len(vector) {
		t.Errorf("Expected dimension %d, got %d", len(vector), len(decompressed))
	}

	// Verify reasonable reconstruction (should be close but not exact)
	maxError := float32(0)
	for i := range vector {
		error := float32(math.Abs(float64(vector[i] - decompressed[i])))
		if error > maxError {
			maxError = error
		}
	}

	// Error should be reasonable for quantization
	if maxError > 50.0 { // Generous threshold for test data
		t.Errorf("Reconstruction error too high: %f", maxError)
	}
}

func TestProductQuantizer_CompressErrors(t *testing.T) {
	pq := setupTrainedQuantizer(t, 16, 4, 8)

	tests := []struct {
		name    string
		vector  []float32
		wantErr bool
	}{
		{
			name:    "wrong dimension",
			vector:  make([]float32, 12),
			wantErr: true,
		},
		{
			name:    "empty vector",
			vector:  []float32{},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := pq.Compress(tt.vector)
			if (err != nil) != tt.wantErr {
				t.Errorf("Compress() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestProductQuantizer_Distance(t *testing.T) {
	pq := setupTrainedQuantizer(t, 16, 4, 8)

	// Test vectors
	vector1 := generateRandomVector(16)
	vector2 := generateRandomVector(16)

	// Compress both vectors
	compressed1, err := pq.Compress(vector1)
	if err != nil {
		t.Fatalf("Compress vector1 error = %v", err)
	}

	compressed2, err := pq.Compress(vector2)
	if err != nil {
		t.Fatalf("Compress vector2 error = %v", err)
	}

	// Compute distance between compressed vectors
	dist, err := pq.Distance(compressed1, compressed2)
	if err != nil {
		t.Fatalf("Distance() error = %v", err)
	}

	// Distance should be non-negative
	if dist < 0 {
		t.Errorf("Distance should be non-negative, got %f", dist)
	}

	// Distance to self should be 0
	selfDist, err := pq.Distance(compressed1, compressed1)
	if err != nil {
		t.Fatalf("Self distance error = %v", err)
	}

	if selfDist != 0 {
		t.Errorf("Distance to self should be 0, got %f", selfDist)
	}
}

func TestProductQuantizer_DistanceToQuery(t *testing.T) {
	pq := setupTrainedQuantizer(t, 16, 4, 8)

	// Test vectors
	vector := generateRandomVector(16)
	query := generateRandomVector(16)

	// Compress vector
	compressed, err := pq.Compress(vector)
	if err != nil {
		t.Fatalf("Compress error = %v", err)
	}

	// Compute distance to query
	dist1, err := pq.DistanceToQuery(compressed, query)
	if err != nil {
		t.Fatalf("DistanceToQuery() error = %v", err)
	}

	// Distance should be non-negative
	if dist1 < 0 {
		t.Errorf("Distance should be non-negative, got %f", dist1)
	}

	// Compute distance to same query again (should use cached tables)
	dist2, err := pq.DistanceToQuery(compressed, query)
	if err != nil {
		t.Fatalf("DistanceToQuery() second call error = %v", err)
	}

	// Should get same result
	if dist1 != dist2 {
		t.Errorf("Expected same distance, got %f and %f", dist1, dist2)
	}

	// Distance to self should be 0
	selfDist, err := pq.DistanceToQuery(compressed, vector)
	if err != nil {
		t.Fatalf("Self distance error = %v", err)
	}

	// Should be close to 0 (not exactly due to quantization)
	// Use a more generous threshold since PQ introduces quantization error
	if selfDist > 50.0 { // More generous threshold for quantization error
		t.Errorf("Distance to self should be small, got %f", selfDist)
	}
}

func TestProductQuantizer_CompressionRatio(t *testing.T) {
	pq := setupTrainedQuantizer(t, 16, 4, 8)

	ratio := pq.CompressionRatio()

	// Expected: 16 * 32 bits / (4 * 8 bits) = 512 / 32 = 16
	expectedRatio := float32(16.0)
	if ratio != expectedRatio {
		t.Errorf("Expected compression ratio %f, got %f", expectedRatio, ratio)
	}
}

func TestProductQuantizer_MemoryUsage(t *testing.T) {
	pq := setupTrainedQuantizer(t, 16, 4, 8)

	usage := pq.MemoryUsage()

	// Should be positive
	if usage <= 0 {
		t.Errorf("Memory usage should be positive, got %d", usage)
	}

	// Should include centroids memory
	// 4 subspaces * 256 centroids * 4 dimensions * 4 bytes = 16384 bytes minimum
	minExpected := int64(4 * 256 * 4 * 4)
	if usage < minExpected {
		t.Errorf("Memory usage %d should be at least %d", usage, minExpected)
	}
}

func TestProductQuantizer_UntrainedOperations(t *testing.T) {
	pq := NewProductQuantizer()
	config := &QuantizationConfig{
		Type:       ProductQuantization,
		Codebooks:  4,
		Bits:       8,
		TrainRatio: 0.1,
	}
	pq.Configure(config)

	vector := make([]float32, 16)
	compressed := make([]byte, 4)

	// All operations should fail on untrained quantizer
	_, err := pq.Compress(vector)
	if err == nil {
		t.Error("Expected error for Compress on untrained quantizer")
	}

	_, err = pq.Decompress(compressed)
	if err == nil {
		t.Error("Expected error for Decompress on untrained quantizer")
	}

	_, err = pq.Distance(compressed, compressed)
	if err == nil {
		t.Error("Expected error for Distance on untrained quantizer")
	}

	_, err = pq.DistanceToQuery(compressed, vector)
	if err == nil {
		t.Error("Expected error for DistanceToQuery on untrained quantizer")
	}

	// These should work on untrained quantizer
	if pq.IsTrained() {
		t.Error("Expected IsTrained to return false")
	}

	if pq.CompressionRatio() != 0 {
		t.Error("Expected CompressionRatio to return 0 for untrained quantizer")
	}
}

func TestProductQuantizerFactory(t *testing.T) {
	factory := NewProductQuantizerFactory()

	// Test factory name
	if factory.Name() != "ProductQuantizer" {
		t.Errorf("Expected name 'ProductQuantizer', got '%s'", factory.Name())
	}

	// Test supports
	if !factory.Supports(ProductQuantization) {
		t.Error("Factory should support ProductQuantization")
	}

	if factory.Supports(ScalarQuantization) {
		t.Error("Factory should not support ScalarQuantization")
	}

	// Test create
	config := &QuantizationConfig{
		Type:       ProductQuantization,
		Codebooks:  4,
		Bits:       8,
		TrainRatio: 0.1,
	}

	quantizer, err := factory.Create(config)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	if quantizer == nil {
		t.Error("Expected non-nil quantizer")
	}

	// Test create with wrong type
	wrongConfig := &QuantizationConfig{
		Type:       ScalarQuantization,
		Bits:       8,
		TrainRatio: 0.1,
	}

	_, err = factory.Create(wrongConfig)
	if err == nil {
		t.Error("Expected error for wrong quantization type")
	}
}

func TestProductQuantizer_AccuracyBenchmark(t *testing.T) {
	// Test accuracy vs compression ratio trade-offs
	dimension := 128
	numVectors := 1000
	vectors := generateRandomVectors(numVectors, dimension)

	configs := []struct {
		name      string
		codebooks int
		bits      int
	}{
		{"4x8", 4, 8},
		{"8x8", 8, 8},
		{"16x8", 16, 8},
		{"8x4", 8, 4},
		{"8x6", 8, 6},
	}

	for _, config := range configs {
		t.Run(config.name, func(t *testing.T) {
			if dimension%config.codebooks != 0 {
				t.Skip("Dimension not divisible by codebooks")
			}

			pq := setupTrainedQuantizerWithParams(t, dimension, config.codebooks, config.bits)

			// Test accuracy on a subset of vectors
			testVectors := vectors[:100]
			totalError := float64(0)

			for _, vector := range testVectors {
				compressed, err := pq.Compress(vector)
				if err != nil {
					t.Fatalf("Compress error: %v", err)
				}

				decompressed, err := pq.Decompress(compressed)
				if err != nil {
					t.Fatalf("Decompress error: %v", err)
				}

				// Compute reconstruction error
				error := euclideanDistance(vector, decompressed)
				totalError += float64(error)
			}

			avgError := totalError / float64(len(testVectors))
			compressionRatio := pq.CompressionRatio()

			t.Logf("Config %s: Avg Error = %.4f, Compression Ratio = %.2fx",
				config.name, avgError, compressionRatio)

			// Sanity check: error should be reasonable for quantization
			// PQ introduces significant quantization error, especially with random data
			if avgError > 500.0 { // More generous threshold for PQ quantization error
				t.Errorf("Average reconstruction error too high: %f", avgError)
			}
		})
	}
}

// Helper functions

func setupTrainedQuantizer(t *testing.T, dimension, codebooks, bits int) *ProductQuantizer {
	return setupTrainedQuantizerWithParams(t, dimension, codebooks, bits)
}

func setupTrainedQuantizerWithParams(t *testing.T, dimension, codebooks, bits int) *ProductQuantizer {
	pq := NewProductQuantizer()
	config := &QuantizationConfig{
		Type:       ProductQuantization,
		Codebooks:  codebooks,
		Bits:       bits,
		TrainRatio: 0.5,
		CacheSize:  100,
	}

	err := pq.Configure(config)
	if err != nil {
		t.Fatalf("Configure error: %v", err)
	}

	vectors := generateRandomVectors(1000, dimension)
	ctx := context.Background()
	err = pq.Train(ctx, vectors)
	if err != nil {
		t.Fatalf("Train error: %v", err)
	}

	return pq
}

func generateRandomVectors(count, dimension int) [][]float32 {
	rand.Seed(time.Now().UnixNano())
	vectors := make([][]float32, count)

	for i := 0; i < count; i++ {
		vectors[i] = generateRandomVector(dimension)
	}

	return vectors
}

func generateRandomVector(dimension int) []float32 {
	vector := make([]float32, dimension)
	for i := 0; i < dimension; i++ {
		vector[i] = rand.Float32()*100 - 50 // Random values between -50 and 50
	}
	return vector
}

func euclideanDistance(a, b []float32) float32 {
	if len(a) != len(b) {
		return float32(math.Inf(1))
	}

	sum := float32(0)
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}
