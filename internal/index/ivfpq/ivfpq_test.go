package ivfpq

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/xDarkicex/libravdb/internal/quant"
	"github.com/xDarkicex/libravdb/internal/util"
)

func TestNewIVFPQ(t *testing.T) {
	tests := []struct {
		name        string
		config      *Config
		expectError bool
	}{
		{
			name:        "nil config",
			config:      nil,
			expectError: true,
		},
		{
			name: "invalid dimension",
			config: &Config{
				Dimension: 0,
				NClusters: 10,
				NProbes:   2,
			},
			expectError: true,
		},
		{
			name: "invalid clusters",
			config: &Config{
				Dimension: 128,
				NClusters: 0,
				NProbes:   2,
			},
			expectError: true,
		},
		{
			name: "invalid probes",
			config: &Config{
				Dimension: 128,
				NClusters: 10,
				NProbes:   15, // More than clusters
			},
			expectError: true,
		},
		{
			name: "valid config",
			config: &Config{
				Dimension:     128,
				NClusters:     10,
				NProbes:       2,
				Metric:        util.L2Distance,
				MaxIterations: 100,
				Tolerance:     1e-4,
				RandomSeed:    42,
			},
			expectError: false,
		},
		{
			name: "valid config with quantization",
			config: &Config{
				Dimension:     128,
				NClusters:     10,
				NProbes:       2,
				Metric:        util.L2Distance,
				Quantization:  quant.DefaultConfig(quant.ProductQuantization),
				MaxIterations: 100,
				Tolerance:     1e-4,
				RandomSeed:    42,
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			idx, err := NewIVFPQ(tt.config)

			if tt.expectError {
				if err == nil {
					t.Errorf("expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			if idx == nil {
				t.Errorf("expected non-nil index")
				return
			}

			// Verify initial state
			if idx.Size() != 0 {
				t.Errorf("expected size 0, got %d", idx.Size())
			}

			if idx.IsTrained() {
				t.Errorf("expected untrained index")
			}
		})
	}
}

func TestDefaultConfig(t *testing.T) {
	tests := []struct {
		dimension int
		expected  int // Expected number of clusters
	}{
		{dimension: 64, expected: 64}, // Min clusters
		{dimension: 128, expected: 128},
		{dimension: 512, expected: 512},
		{dimension: 8192, expected: 4096}, // Max clusters
	}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			config := DefaultConfig(tt.dimension)

			if config.Dimension != tt.dimension {
				t.Errorf("expected dimension %d, got %d", tt.dimension, config.Dimension)
			}

			if config.NClusters != tt.expected {
				t.Errorf("expected %d clusters, got %d", tt.expected, config.NClusters)
			}

			if config.NProbes <= 0 || config.NProbes > config.NClusters {
				t.Errorf("invalid number of probes: %d", config.NProbes)
			}
		})
	}
}

func TestTraining(t *testing.T) {
	ctx := context.Background()

	// Create test data
	dimension := 4
	nVectors := 100
	vectors := generateTestVectors(nVectors, dimension, 42)

	config := &Config{
		Dimension:     dimension,
		NClusters:     4,
		NProbes:       2,
		Metric:        util.L2Distance,
		MaxIterations: 10,
		Tolerance:     1e-4,
		RandomSeed:    42,
	}

	idx, err := NewIVFPQ(config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}

	// Test training
	err = idx.Train(ctx, vectors)
	if err != nil {
		t.Fatalf("failed to train index: %v", err)
	}

	if !idx.IsTrained() {
		t.Errorf("expected trained index")
	}

	// Verify centroids are initialized
	clusterInfo := idx.GetClusterInfo()
	if len(clusterInfo) != config.NClusters {
		t.Errorf("expected %d clusters, got %d", config.NClusters, len(clusterInfo))
	}

	for i, info := range clusterInfo {
		if len(info.Centroid) != dimension {
			t.Errorf("cluster %d centroid has wrong dimension: %d", i, len(info.Centroid))
		}

		// Check that centroid is not zero vector
		nonZero := false
		for _, val := range info.Centroid {
			if val != 0 {
				nonZero = true
				break
			}
		}
		if !nonZero {
			t.Errorf("cluster %d centroid is zero vector", i)
		}
	}
}

func TestTrainingErrors(t *testing.T) {
	ctx := context.Background()
	dimension := 4

	config := &Config{
		Dimension:     dimension,
		NClusters:     4,
		NProbes:       2,
		Metric:        util.L2Distance,
		MaxIterations: 10,
		Tolerance:     1e-4,
		RandomSeed:    42,
	}

	idx, err := NewIVFPQ(config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}

	tests := []struct {
		name    string
		vectors [][]float32
	}{
		{
			name:    "empty vectors",
			vectors: [][]float32{},
		},
		{
			name: "too few vectors",
			vectors: [][]float32{
				{1, 2, 3, 4},
				{5, 6, 7, 8},
			}, // Only 2 vectors for 4 clusters
		},
		{
			name: "wrong dimension",
			vectors: [][]float32{
				{1, 2, 3},    // Wrong dimension
				{4, 5, 6, 7}, // Correct dimension
				{8, 9, 10, 11},
				{12, 13, 14, 15},
				{16, 17, 18, 19},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := idx.Train(ctx, tt.vectors)
			if err == nil {
				t.Errorf("expected error but got none")
			}
		})
	}
}

func TestInsertAndSearch(t *testing.T) {
	ctx := context.Background()
	dimension := 4

	// Create and train index
	config := &Config{
		Dimension:     dimension,
		NClusters:     2,
		NProbes:       2,
		Metric:        util.L2Distance,
		MaxIterations: 10,
		Tolerance:     1e-4,
		RandomSeed:    42,
	}

	idx, err := NewIVFPQ(config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}

	// Generate training data
	trainingVectors := generateTestVectors(20, dimension, 42)
	err = idx.Train(ctx, trainingVectors)
	if err != nil {
		t.Fatalf("failed to train index: %v", err)
	}

	// Insert test entries
	entries := []*VectorEntry{
		{ID: "1", Vector: []float32{1, 0, 0, 0}, Metadata: map[string]interface{}{"label": "a"}},
		{ID: "2", Vector: []float32{0, 1, 0, 0}, Metadata: map[string]interface{}{"label": "b"}},
		{ID: "3", Vector: []float32{0, 0, 1, 0}, Metadata: map[string]interface{}{"label": "c"}},
		{ID: "4", Vector: []float32{0, 0, 0, 1}, Metadata: map[string]interface{}{"label": "d"}},
	}

	for _, entry := range entries {
		err := idx.Insert(ctx, entry)
		if err != nil {
			t.Fatalf("failed to insert entry %s: %v", entry.ID, err)
		}
	}

	if idx.Size() != len(entries) {
		t.Errorf("expected size %d, got %d", len(entries), idx.Size())
	}

	// Test search
	query := []float32{1, 0, 0, 0}
	results, err := idx.Search(ctx, query, 2)
	if err != nil {
		t.Fatalf("failed to search: %v", err)
	}

	if len(results) == 0 {
		t.Errorf("expected search results")
	}

	// First result should be closest to query
	if len(results) > 0 && results[0].ID != "1" {
		t.Errorf("expected first result to be ID '1', got '%s'", results[0].ID)
	}
}

func TestInsertErrors(t *testing.T) {
	ctx := context.Background()
	dimension := 4

	config := &Config{
		Dimension:     dimension,
		NClusters:     2,
		NProbes:       1,
		Metric:        util.L2Distance,
		MaxIterations: 10,
		Tolerance:     1e-4,
		RandomSeed:    42,
	}

	// idx is not used since we create fresh indices in each test
	// idx, err := NewIVFPQ(config)
	// if err != nil {
	//	t.Fatalf("failed to create index: %v", err)
	// }

	tests := []struct {
		name  string
		entry *VectorEntry
		train bool
	}{
		{
			name:  "nil entry",
			entry: nil,
			train: true,
		},
		{
			name: "wrong dimension",
			entry: &VectorEntry{
				ID:     "1",
				Vector: []float32{1, 2, 3}, // Wrong dimension
			},
			train: true,
		},
		{
			name: "untrained index",
			entry: &VectorEntry{
				ID:     "1",
				Vector: []float32{1, 2, 3, 4},
			},
			train: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create fresh index for each test
			freshIdx, err := NewIVFPQ(config)
			if err != nil {
				t.Fatalf("failed to create index: %v", err)
			}

			if tt.train {
				trainingVectors := generateTestVectors(10, dimension, 42)
				err := freshIdx.Train(ctx, trainingVectors)
				if err != nil {
					t.Fatalf("failed to train index: %v", err)
				}
			}

			err = freshIdx.Insert(ctx, tt.entry)
			if err == nil {
				t.Errorf("expected error but got none")
			}
		})
	}
}

func TestDelete(t *testing.T) {
	ctx := context.Background()
	dimension := 4

	// Create and train index
	config := &Config{
		Dimension:     dimension,
		NClusters:     2,
		NProbes:       2,
		Metric:        util.L2Distance,
		MaxIterations: 10,
		Tolerance:     1e-4,
		RandomSeed:    42,
	}

	idx, err := NewIVFPQ(config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}

	// Train index
	trainingVectors := generateTestVectors(20, dimension, 42)
	err = idx.Train(ctx, trainingVectors)
	if err != nil {
		t.Fatalf("failed to train index: %v", err)
	}

	// Insert entries
	entries := []*VectorEntry{
		{ID: "1", Vector: []float32{1, 0, 0, 0}},
		{ID: "2", Vector: []float32{0, 1, 0, 0}},
		{ID: "3", Vector: []float32{0, 0, 1, 0}},
	}

	for _, entry := range entries {
		err := idx.Insert(ctx, entry)
		if err != nil {
			t.Fatalf("failed to insert entry: %v", err)
		}
	}

	initialSize := idx.Size()

	// Delete existing entry
	err = idx.Delete(ctx, "2")
	if err != nil {
		t.Errorf("failed to delete entry: %v", err)
	}

	if idx.Size() != initialSize-1 {
		t.Errorf("expected size %d after deletion, got %d", initialSize-1, idx.Size())
	}

	// Try to delete non-existent entry
	err = idx.Delete(ctx, "nonexistent")
	if err == nil {
		t.Errorf("expected error when deleting non-existent entry")
	}

	// Try to delete empty ID
	err = idx.Delete(ctx, "")
	if err == nil {
		t.Errorf("expected error when deleting empty ID")
	}
}

func TestSearchErrors(t *testing.T) {
	ctx := context.Background()
	dimension := 4

	config := &Config{
		Dimension:     dimension,
		NClusters:     2,
		NProbes:       1,
		Metric:        util.L2Distance,
		MaxIterations: 10,
		Tolerance:     1e-4,
		RandomSeed:    42,
	}

	// idx is not used since we create fresh indices in each test
	// idx, err := NewIVFPQ(config)
	// if err != nil {
	//	t.Fatalf("failed to create index: %v", err)
	// }

	tests := []struct {
		name  string
		query []float32
		k     int
		train bool
	}{
		{
			name:  "wrong dimension",
			query: []float32{1, 2, 3}, // Wrong dimension
			k:     1,
			train: true,
		},
		{
			name:  "invalid k",
			query: []float32{1, 2, 3, 4},
			k:     0,
			train: true,
		},
		{
			name:  "untrained index",
			query: []float32{1, 2, 3, 4},
			k:     1,
			train: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create fresh index for each test
			freshIdx, err := NewIVFPQ(config)
			if err != nil {
				t.Fatalf("failed to create index: %v", err)
			}

			if tt.train {
				trainingVectors := generateTestVectors(10, dimension, 42)
				err := freshIdx.Train(ctx, trainingVectors)
				if err != nil {
					t.Fatalf("failed to train index: %v", err)
				}
			}

			_, err = freshIdx.Search(ctx, tt.query, tt.k)
			if err == nil {
				t.Errorf("expected error but got none")
			}
		})
	}
}

func TestClusterAssignment(t *testing.T) {
	ctx := context.Background()
	dimension := 2

	// Create index with known centroids
	config := &Config{
		Dimension:     dimension,
		NClusters:     2,
		NProbes:       1,
		Metric:        util.L2Distance,
		MaxIterations: 10,
		Tolerance:     1e-4,
		RandomSeed:    42,
	}

	idx, err := NewIVFPQ(config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}

	// Create training vectors that will form clear clusters
	trainingVectors := [][]float32{
		{0, 0}, {0.1, 0.1}, {-0.1, -0.1}, // Cluster around origin
		{10, 10}, {10.1, 10.1}, {9.9, 9.9}, // Cluster around (10,10)
	}

	err = idx.Train(ctx, trainingVectors)
	if err != nil {
		t.Fatalf("failed to train index: %v", err)
	}

	// Test assignment accuracy
	testCases := []struct {
		vector   []float32
		expected int // Expected cluster (we don't know exact assignment, but test consistency)
	}{
		{[]float32{0, 0}, -1},     // Near first cluster
		{[]float32{10, 10}, -1},   // Near second cluster
		{[]float32{0.5, 0.5}, -1}, // Should be closer to first cluster
		{[]float32{9.5, 9.5}, -1}, // Should be closer to second cluster
	}

	for i, tc := range testCases {
		clusterID, err := idx.assignToCluster(tc.vector)
		if err != nil {
			t.Errorf("case %d: failed to assign cluster: %v", i, err)
		}

		if clusterID < 0 || clusterID >= config.NClusters {
			t.Errorf("case %d: invalid cluster ID %d", i, clusterID)
		}
	}

	// Test that similar vectors get assigned to same cluster
	cluster1, _ := idx.assignToCluster([]float32{0, 0})
	cluster2, _ := idx.assignToCluster([]float32{0.1, 0.1})

	if cluster1 != cluster2 {
		t.Errorf("similar vectors assigned to different clusters: %d vs %d", cluster1, cluster2)
	}
}

func TestClusterCreationAccuracy(t *testing.T) {
	ctx := context.Background()
	dimension := 3

	config := &Config{
		Dimension:     dimension,
		NClusters:     3,
		NProbes:       1,
		Metric:        util.L2Distance,
		MaxIterations: 50,
		Tolerance:     1e-6,
		RandomSeed:    42,
	}

	idx, err := NewIVFPQ(config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}

	// Create well-separated clusters in 3D space
	trainingVectors := [][]float32{
		// Cluster 1: around (0,0,0)
		{0, 0, 0}, {0.1, 0, 0}, {0, 0.1, 0}, {0, 0, 0.1}, {-0.1, 0, 0}, {0, -0.1, 0}, {0, 0, -0.1},
		// Cluster 2: around (5,5,5)
		{5, 5, 5}, {5.1, 5, 5}, {5, 5.1, 5}, {5, 5, 5.1}, {4.9, 5, 5}, {5, 4.9, 5}, {5, 5, 4.9},
		// Cluster 3: around (-5,-5,-5)
		{-5, -5, -5}, {-5.1, -5, -5}, {-5, -5.1, -5}, {-5, -5, -5.1}, {-4.9, -5, -5}, {-5, -4.9, -5}, {-5, -5, -4.9},
	}

	err = idx.Train(ctx, trainingVectors)
	if err != nil {
		t.Fatalf("failed to train index: %v", err)
	}

	// Verify cluster centroids are reasonable
	clusterInfo := idx.GetClusterInfo()
	if len(clusterInfo) != config.NClusters {
		t.Fatalf("expected %d clusters, got %d", config.NClusters, len(clusterInfo))
	}

	// Test assignment accuracy for vectors close to expected cluster centers
	testCases := []struct {
		name   string
		vector []float32
	}{
		{"near origin", []float32{0.05, 0.05, 0.05}},
		{"near (5,5,5)", []float32{5.05, 5.05, 5.05}},
		{"near (-5,-5,-5)", []float32{-5.05, -5.05, -5.05}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			clusterID, err := idx.assignToCluster(tc.vector)
			if err != nil {
				t.Errorf("failed to assign cluster: %v", err)
			}

			if clusterID < 0 || clusterID >= config.NClusters {
				t.Errorf("invalid cluster ID %d", clusterID)
			}

			// Verify the assigned cluster centroid is actually closest
			assignedCentroid := clusterInfo[clusterID].Centroid
			assignedDistance := util.L2Distance_func(tc.vector, assignedCentroid)

			for i, info := range clusterInfo {
				if i == clusterID {
					continue
				}
				otherDistance := util.L2Distance_func(tc.vector, info.Centroid)
				if otherDistance < assignedDistance {
					t.Errorf("vector %v assigned to cluster %d (distance %.4f) but cluster %d is closer (distance %.4f)",
						tc.vector, clusterID, assignedDistance, i, otherDistance)
				}
			}
		})
	}
}

func TestProbeClusterAccuracy(t *testing.T) {
	ctx := context.Background()
	dimension := 2

	config := &Config{
		Dimension:     dimension,
		NClusters:     4,
		NProbes:       2, // Probe 2 closest clusters
		Metric:        util.L2Distance,
		MaxIterations: 20,
		Tolerance:     1e-4,
		RandomSeed:    42,
	}

	idx, err := NewIVFPQ(config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}

	// Create training vectors in 4 corners of a square
	trainingVectors := [][]float32{
		// Corner 1: (0,0)
		{0, 0}, {0.1, 0}, {0, 0.1}, {0.1, 0.1},
		// Corner 2: (10,0)
		{10, 0}, {10.1, 0}, {10, 0.1}, {10.1, 0.1},
		// Corner 3: (0,10)
		{0, 10}, {0.1, 10}, {0, 10.1}, {0.1, 10.1},
		// Corner 4: (10,10)
		{10, 10}, {10.1, 10}, {10, 10.1}, {10.1, 10.1},
	}

	err = idx.Train(ctx, trainingVectors)
	if err != nil {
		t.Fatalf("failed to train index: %v", err)
	}

	// Test probe cluster selection
	testCases := []struct {
		name  string
		query []float32
	}{
		{"center query", []float32{5, 5}},
		{"corner query", []float32{0.5, 0.5}},
		{"edge query", []float32{5, 0.5}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			probeClusters, err := idx.findProbeClusters(tc.query)
			if err != nil {
				t.Errorf("failed to find probe clusters: %v", err)
			}

			if len(probeClusters) != config.NProbes {
				t.Errorf("expected %d probe clusters, got %d", config.NProbes, len(probeClusters))
			}

			// Verify probe clusters are unique
			seen := make(map[int]bool)
			for _, clusterID := range probeClusters {
				if seen[clusterID] {
					t.Errorf("duplicate cluster ID %d in probe list", clusterID)
				}
				seen[clusterID] = true

				if clusterID < 0 || clusterID >= config.NClusters {
					t.Errorf("invalid cluster ID %d", clusterID)
				}
			}

			// Verify probe clusters are actually the closest ones
			clusterInfo := idx.GetClusterInfo()
			distances := make([]float32, len(clusterInfo))
			for i, info := range clusterInfo {
				distances[i] = util.L2Distance_func(tc.query, info.Centroid)
			}

			// Check that probed clusters are among the closest
			for _, probeID := range probeClusters {
				probeDistance := distances[probeID]
				closerCount := 0
				for _, distance := range distances {
					if distance < probeDistance {
						closerCount++
					}
				}
				if closerCount >= config.NProbes {
					t.Errorf("probe cluster %d (distance %.4f) is not among the %d closest clusters",
						probeID, probeDistance, config.NProbes)
				}
			}
		})
	}
}

func TestMemoryUsage(t *testing.T) {
	ctx := context.Background()
	dimension := 4

	config := &Config{
		Dimension:     dimension,
		NClusters:     2,
		NProbes:       1,
		Metric:        util.L2Distance,
		MaxIterations: 10,
		Tolerance:     1e-4,
		RandomSeed:    42,
	}

	idx, err := NewIVFPQ(config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}

	// Memory usage should be positive even for empty index
	initialUsage := idx.MemoryUsage()
	if initialUsage <= 0 {
		t.Errorf("expected positive memory usage, got %d", initialUsage)
	}

	// Train index
	trainingVectors := generateTestVectors(20, dimension, 42)
	err = idx.Train(ctx, trainingVectors)
	if err != nil {
		t.Fatalf("failed to train index: %v", err)
	}

	// Insert some entries
	for i := 0; i < 10; i++ {
		entry := &VectorEntry{
			ID:     fmt.Sprintf("entry_%d", i),
			Vector: generateTestVectors(1, dimension, int64(i))[0],
		}
		err := idx.Insert(ctx, entry)
		if err != nil {
			t.Fatalf("failed to insert entry: %v", err)
		}
	}

	// Memory usage should increase after insertions
	finalUsage := idx.MemoryUsage()
	if finalUsage <= initialUsage {
		t.Errorf("expected memory usage to increase after insertions: %d -> %d", initialUsage, finalUsage)
	}
}

func TestClose(t *testing.T) {
	ctx := context.Background()
	dimension := 4

	config := &Config{
		Dimension:     dimension,
		NClusters:     2,
		NProbes:       1,
		Metric:        util.L2Distance,
		MaxIterations: 10,
		Tolerance:     1e-4,
		RandomSeed:    42,
	}

	idx, err := NewIVFPQ(config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}

	// Train and populate index
	trainingVectors := generateTestVectors(20, dimension, 42)
	err = idx.Train(ctx, trainingVectors)
	if err != nil {
		t.Fatalf("failed to train index: %v", err)
	}

	entry := &VectorEntry{
		ID:     "test",
		Vector: []float32{1, 2, 3, 4},
	}
	err = idx.Insert(ctx, entry)
	if err != nil {
		t.Fatalf("failed to insert entry: %v", err)
	}

	// Close index
	err = idx.Close()
	if err != nil {
		t.Errorf("failed to close index: %v", err)
	}

	// Verify index is cleaned up
	if idx.Size() != 0 {
		t.Errorf("expected size 0 after close, got %d", idx.Size())
	}

	if idx.IsTrained() {
		t.Errorf("expected untrained state after close")
	}
}

func TestKMeansConvergence(t *testing.T) {
	ctx := context.Background()
	dimension := 2

	config := &Config{
		Dimension:     dimension,
		NClusters:     3,
		NProbes:       1,
		Metric:        util.L2Distance,
		MaxIterations: 100,
		Tolerance:     1e-6,
		RandomSeed:    42,
	}

	idx, err := NewIVFPQ(config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}

	// Create three distinct clusters of training data
	trainingVectors := [][]float32{}

	// Cluster 1: around (0, 0)
	for i := 0; i < 20; i++ {
		x := rand.Float32()*0.5 - 0.25 // [-0.25, 0.25]
		y := rand.Float32()*0.5 - 0.25
		trainingVectors = append(trainingVectors, []float32{x, y})
	}

	// Cluster 2: around (5, 0)
	for i := 0; i < 20; i++ {
		x := 5 + rand.Float32()*0.5 - 0.25 // [4.75, 5.25]
		y := rand.Float32()*0.5 - 0.25
		trainingVectors = append(trainingVectors, []float32{x, y})
	}

	// Cluster 3: around (2.5, 4)
	for i := 0; i < 20; i++ {
		x := 2.5 + rand.Float32()*0.5 - 0.25 // [2.25, 2.75]
		y := 4 + rand.Float32()*0.5 - 0.25   // [3.75, 4.25]
		trainingVectors = append(trainingVectors, []float32{x, y})
	}

	err = idx.Train(ctx, trainingVectors)
	if err != nil {
		t.Fatalf("failed to train index: %v", err)
	}

	// Verify cluster centroids are reasonable
	clusterInfo := idx.GetClusterInfo()
	if len(clusterInfo) != config.NClusters {
		t.Fatalf("expected %d clusters, got %d", config.NClusters, len(clusterInfo))
	}

	// Expected cluster centers (approximately)
	expectedCenters := [][]float32{
		{0, 0},   // Cluster around origin
		{5, 0},   // Cluster around (5,0)
		{2.5, 4}, // Cluster around (2.5,4)
	}

	// For each expected center, find the closest actual centroid
	for i, expected := range expectedCenters {
		minDistance := float32(math.Inf(1))
		closestCluster := -1

		for j, info := range clusterInfo {
			distance := util.L2Distance_func(expected, info.Centroid)
			if distance < minDistance {
				minDistance = distance
				closestCluster = j
			}
		}

		if closestCluster == -1 {
			t.Errorf("no cluster found for expected center %d: %v", i, expected)
			continue
		}

		// Verify the closest centroid is reasonably close to expected
		if minDistance > 1.0 { // Allow some tolerance
			t.Errorf("cluster centroid %v is too far from expected center %v (distance: %.4f)",
				clusterInfo[closestCluster].Centroid, expected, minDistance)
		}

		t.Logf("Expected center %v matched to cluster %d centroid %v (distance: %.4f)",
			expected, closestCluster, clusterInfo[closestCluster].Centroid, minDistance)
	}

	// Verify all clusters have reasonable sizes
	totalSize := 0
	for i, info := range clusterInfo {
		if info.Size < 0 {
			t.Errorf("cluster %d has negative size: %d", i, info.Size)
		}
		totalSize += info.Size
		t.Logf("Cluster %d: size=%d, centroid=%v", i, info.Size, info.Centroid)
	}

	// Note: totalSize will be 0 here because we haven't inserted the training vectors
	// into the index, only used them for training. This is expected behavior.
}

// generateTestVectors creates test vectors for testing
func generateTestVectors(count, dimension int, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	vectors := make([][]float32, count)

	for i := 0; i < count; i++ {
		vector := make([]float32, dimension)
		for j := 0; j < dimension; j++ {
			vector[j] = rng.Float32()*2 - 1 // Random values between -1 and 1
		}
		vectors[i] = vector
	}

	return vectors
}
