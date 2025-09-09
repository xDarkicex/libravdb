package quant

import (
	"context"
	"testing"
)

func BenchmarkProductQuantizer_Train(b *testing.B) {
	pq := NewProductQuantizer()
	config := &QuantizationConfig{
		Type:       ProductQuantization,
		Codebooks:  8,
		Bits:       8,
		TrainRatio: 0.1,
		CacheSize:  1000,
	}
	pq.Configure(config)

	vectors := generateRandomVectors(10000, 128) // Large dataset
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pq.Train(ctx, vectors)
	}
}

func BenchmarkProductQuantizer_Compress(b *testing.B) {
	pq := setupTrainedQuantizerForBench(128, 8, 8)
	vector := generateRandomVector(128)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := pq.Compress(vector)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkProductQuantizer_Decompress(b *testing.B) {
	pq := setupTrainedQuantizerForBench(128, 8, 8)
	vector := generateRandomVector(128)
	compressed, _ := pq.Compress(vector)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := pq.Decompress(compressed)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkProductQuantizer_Distance(b *testing.B) {
	pq := setupTrainedQuantizerForBench(128, 8, 8)
	vector1 := generateRandomVector(128)
	vector2 := generateRandomVector(128)
	compressed1, _ := pq.Compress(vector1)
	compressed2, _ := pq.Compress(vector2)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := pq.Distance(compressed1, compressed2)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkProductQuantizer_DistanceToQuery(b *testing.B) {
	pq := setupTrainedQuantizerForBench(128, 8, 8)
	vector := generateRandomVector(128)
	query := generateRandomVector(128)
	compressed, _ := pq.Compress(vector)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := pq.DistanceToQuery(compressed, query)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkProductQuantizer_CompressionRatios(b *testing.B) {
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

	dimension := 128
	vectors := generateRandomVectors(1000, dimension)

	for _, config := range configs {
		if dimension%config.codebooks != 0 {
			continue
		}

		b.Run(config.name, func(b *testing.B) {
			pq := NewProductQuantizer()
			cfg := &QuantizationConfig{
				Type:       ProductQuantization,
				Codebooks:  config.codebooks,
				Bits:       config.bits,
				TrainRatio: 0.1,
			}
			pq.Configure(cfg)

			ctx := context.Background()
			pq.Train(ctx, vectors)

			testVector := generateRandomVector(dimension)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				compressed, err := pq.Compress(testVector)
				if err != nil {
					b.Fatal(err)
				}
				_, err = pq.Decompress(compressed)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// Helper function for benchmarks
func setupTrainedQuantizerForBench(dimension, codebooks, bits int) *ProductQuantizer {
	pq := NewProductQuantizer()
	config := &QuantizationConfig{
		Type:       ProductQuantization,
		Codebooks:  codebooks,
		Bits:       bits,
		TrainRatio: 0.1,
		CacheSize:  100,
	}

	pq.Configure(config)
	vectors := generateRandomVectors(1000, dimension)
	ctx := context.Background()
	err := pq.Train(ctx, vectors)
	if err != nil {
		panic(err)
	}

	return pq
}
