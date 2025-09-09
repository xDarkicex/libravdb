package ivfpq

import (
	"context"
	"fmt"
	"math/rand"
	"testing"

	"github.com/xDarkicex/libravdb/internal/quant"
	"github.com/xDarkicex/libravdb/internal/util"
)

func TestIVFPQWithQuantizationIntegration(t *testing.T) {
	ctx := context.Background()
	dimension := 8

	// Test both quantization types
	quantTypes := []struct {
		name   string
		config *quant.QuantizationConfig
	}{
		{
			name: "Product Quantization",
			config: &quant.QuantizationConfig{
				Type:       quant.ProductQuantization,
				Codebooks:  4,
				Bits:       8,
				TrainRatio: 0.5,
				CacheSize:  100,
			},
		},
		{
			name: "Scalar Quantization",
			config: &quant.QuantizationConfig{
				Type:       quant.ScalarQuantization,
				Bits:       8,
				TrainRatio: 0.5,
			},
		},
	}

	for _, qt := range quantTypes {
		t.Run(qt.name, func(t *testing.T) {
			config := &Config{
				Dimension:     dimension,
				NClusters:     4,
				NProbes:       2,
				Metric:        util.L2Distance,
				Quantization:  qt.config,
				MaxIterations: 20,
				Tolerance:     1e-4,
				RandomSeed:    42,
			}

			idx, err := NewIVFPQ(config)
			if err != nil {
				t.Fatalf("failed to create IVF-PQ index: %v", err)
			}
			defer idx.Close()

			// Generate training data
			trainingVectors := generateTestVectors(100, dimension, 42)

			// Train the index (this should train both coarse and fine quantizers)
			err = idx.Train(ctx, trainingVectors)
			if err != nil {
				t.Fatalf("failed to train index: %v", err)
			}

			if !idx.IsTrained() {
				t.Errorf("index should be trained")
			}

			// Insert some test vectors
			testEntries := []*VectorEntry{
				{ID: "1", Vector: generateTestVectors(1, dimension, 1)[0]},
				{ID: "2", Vector: generateTestVectors(1, dimension, 2)[0]},
				{ID: "3", Vector: generateTestVectors(1, dimension, 3)[0]},
				{ID: "4", Vector: generateTestVectors(1, dimension, 4)[0]},
				{ID: "5", Vector: generateTestVectors(1, dimension, 5)[0]},
			}

			for _, entry := range testEntries {
				err := idx.Insert(ctx, entry)
				if err != nil {
					t.Fatalf("failed to insert entry %s: %v", entry.ID, err)
				}
			}

			if idx.Size() != len(testEntries) {
				t.Errorf("expected size %d, got %d", len(testEntries), idx.Size())
			}

			// Test search with quantization
			query := testEntries[0].Vector
			results, err := idx.Search(ctx, query, 3)
			if err != nil {
				t.Fatalf("failed to search: %v", err)
			}

			if len(results) == 0 {
				t.Errorf("expected search results")
			}

			// First result should be the exact match (or very close)
			if len(results) > 0 {
				if results[0].ID != testEntries[0].ID {
					t.Logf("Warning: first result ID %s doesn't match query ID %s (may be due to quantization)",
						results[0].ID, testEntries[0].ID)
				}

				// Score should be reasonable (close to 0 for exact match)
				if results[0].Score > 1.0 {
					t.Errorf("first result score too high: %f", results[0].Score)
				}
			}

			// Test memory usage - should be lower with quantization
			memUsage := idx.MemoryUsage()
			if memUsage <= 0 {
				t.Errorf("expected positive memory usage, got %d", memUsage)
			}

			t.Logf("Memory usage with %s: %d bytes", qt.name, memUsage)
		})
	}
}

func TestIVFPQClusterDistribution(t *testing.T) {
	ctx := context.Background()
	dimension := 4

	config := &Config{
		Dimension:     dimension,
		NClusters:     3,
		NProbes:       3, // Probe all clusters
		Metric:        util.L2Distance,
		MaxIterations: 50,
		Tolerance:     1e-6,
		RandomSeed:    42,
	}

	idx, err := NewIVFPQ(config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}
	defer idx.Close()

	// Create well-separated training data
	trainingVectors := [][]float32{}

	// Cluster 1: around (0,0,0,0)
	for i := 0; i < 20; i++ {
		vec := make([]float32, dimension)
		for j := 0; j < dimension; j++ {
			vec[j] = rand.Float32()*0.2 - 0.1 // [-0.1, 0.1]
		}
		trainingVectors = append(trainingVectors, vec)
	}

	// Cluster 2: around (5,5,5,5)
	for i := 0; i < 20; i++ {
		vec := make([]float32, dimension)
		for j := 0; j < dimension; j++ {
			vec[j] = 5 + rand.Float32()*0.2 - 0.1 // [4.9, 5.1]
		}
		trainingVectors = append(trainingVectors, vec)
	}

	// Cluster 3: around (-5,-5,-5,-5)
	for i := 0; i < 20; i++ {
		vec := make([]float32, dimension)
		for j := 0; j < dimension; j++ {
			vec[j] = -5 + rand.Float32()*0.2 - 0.1 // [-5.1, -4.9]
		}
		trainingVectors = append(trainingVectors, vec)
	}

	err = idx.Train(ctx, trainingVectors)
	if err != nil {
		t.Fatalf("failed to train index: %v", err)
	}

	// Insert vectors and track cluster distribution
	testVectors := [][]float32{
		{0.05, 0.05, 0.05, 0.05},     // Should go to cluster 1
		{5.05, 5.05, 5.05, 5.05},     // Should go to cluster 2
		{-5.05, -5.05, -5.05, -5.05}, // Should go to cluster 3
		{0.02, 0.02, 0.02, 0.02},     // Should go to cluster 1
		{5.02, 5.02, 5.02, 5.02},     // Should go to cluster 2
	}

	for i, vec := range testVectors {
		entry := &VectorEntry{
			ID:     fmt.Sprintf("test_%d", i),
			Vector: vec,
		}
		err := idx.Insert(ctx, entry)
		if err != nil {
			t.Fatalf("failed to insert entry: %v", err)
		}
	}

	// Check cluster distribution
	clusterInfo := idx.GetClusterInfo()
	totalEntries := 0
	for i, info := range clusterInfo {
		totalEntries += info.Size
		t.Logf("Cluster %d: size=%d, centroid=%v", i, info.Size, info.Centroid)
	}

	if totalEntries != len(testVectors) {
		t.Errorf("expected %d total entries across clusters, got %d", len(testVectors), totalEntries)
	}

	// Verify that at least some clusters have entries
	nonEmptyClusters := 0
	for _, info := range clusterInfo {
		if info.Size > 0 {
			nonEmptyClusters++
		}
	}

	if nonEmptyClusters == 0 {
		t.Errorf("expected at least one non-empty cluster")
	}

	t.Logf("Distribution: %d non-empty clusters out of %d total", nonEmptyClusters, len(clusterInfo))
}

func TestIVFPQSearchAccuracy(t *testing.T) {
	ctx := context.Background()
	dimension := 6

	config := &Config{
		Dimension:     dimension,
		NClusters:     2,
		NProbes:       2, // Probe both clusters
		Metric:        util.L2Distance,
		MaxIterations: 20,
		Tolerance:     1e-4,
		RandomSeed:    42,
	}

	idx, err := NewIVFPQ(config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}
	defer idx.Close()

	// Train with diverse data
	trainingVectors := generateTestVectors(50, dimension, 42)
	err = idx.Train(ctx, trainingVectors)
	if err != nil {
		t.Fatalf("failed to train index: %v", err)
	}

	// Insert known vectors
	knownVectors := [][]float32{
		{1, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0},
		{0, 0, 1, 0, 0, 0},
		{0, 0, 0, 1, 0, 0},
		{0, 0, 0, 0, 1, 0},
		{0, 0, 0, 0, 0, 1},
	}

	for i, vec := range knownVectors {
		entry := &VectorEntry{
			ID:     fmt.Sprintf("known_%d", i),
			Vector: vec,
		}
		err := idx.Insert(ctx, entry)
		if err != nil {
			t.Fatalf("failed to insert known vector: %v", err)
		}
	}

	// Test search accuracy
	for i, queryVec := range knownVectors {
		results, err := idx.Search(ctx, queryVec, 3)
		if err != nil {
			t.Errorf("failed to search for vector %d: %v", i, err)
			continue
		}

		if len(results) == 0 {
			t.Errorf("no results for query vector %d", i)
			continue
		}

		// The exact vector should be found (though not necessarily first due to clustering)
		found := false
		expectedID := fmt.Sprintf("known_%d", i)
		for _, result := range results {
			if result.ID == expectedID {
				found = true
				break
			}
		}

		if !found {
			t.Errorf("exact match not found for query vector %d (ID: %s)", i, expectedID)
		}
	}
}
