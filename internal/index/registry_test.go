package index

import (
	"context"
	"testing"

	"github.com/xDarkicex/libravdb/internal/quant"
	"github.com/xDarkicex/libravdb/internal/util"
)

func TestIndexFactory_CreateIndex(t *testing.T) {
	factory := NewIndexFactory()

	tests := []struct {
		name        string
		indexType   IndexType
		config      interface{}
		expectError bool
	}{
		{
			name:      "valid HNSW config",
			indexType: IndexTypeHNSW,
			config: &HNSWConfig{
				Dimension:      128,
				M:              16,
				EfConstruction: 200,
				EfSearch:       50,
				ML:             1.0 / 2.303,
				Metric:         util.L2Distance,
			},
			expectError: false,
		},
		{
			name:      "valid IVF-PQ config",
			indexType: IndexTypeIVFPQ,
			config: &IVFPQConfig{
				Dimension:     128,
				NClusters:     64,
				NProbes:       8,
				Metric:        util.L2Distance,
				MaxIterations: 100,
				Tolerance:     1e-4,
				RandomSeed:    42,
			},
			expectError: false,
		},
		{
			name:      "valid IVF-PQ config with quantization",
			indexType: IndexTypeIVFPQ,
			config: &IVFPQConfig{
				Dimension:     128,
				NClusters:     64,
				NProbes:       8,
				Metric:        util.L2Distance,
				Quantization:  quant.DefaultConfig(quant.ProductQuantization),
				MaxIterations: 100,
				Tolerance:     1e-4,
				RandomSeed:    42,
			},
			expectError: false,
		},
		{
			name:        "invalid config type for HNSW",
			indexType:   IndexTypeHNSW,
			config:      &IVFPQConfig{}, // Wrong config type
			expectError: true,
		},
		{
			name:        "invalid config type for IVF-PQ",
			indexType:   IndexTypeIVFPQ,
			config:      &HNSWConfig{}, // Wrong config type
			expectError: true,
		},
		{
			name:      "valid Flat config",
			indexType: IndexTypeFlat,
			config: &FlatConfig{
				Dimension: 128,
				Metric:    util.L2Distance,
			},
			expectError: false,
		},
		{
			name:        "invalid config type for Flat",
			indexType:   IndexTypeFlat,
			config:      &HNSWConfig{}, // Wrong config type
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			index, err := factory.CreateIndex(tt.indexType, tt.config)

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

			if index == nil {
				t.Errorf("expected non-nil index")
				return
			}

			// Verify index is properly initialized
			if index.Size() != 0 {
				t.Errorf("expected empty index, got size %d", index.Size())
			}

			// Clean up
			err = index.Close()
			if err != nil {
				t.Errorf("failed to close index: %v", err)
			}
		})
	}
}

func TestIndexFactory_SupportedIndexTypes(t *testing.T) {
	factory := NewIndexFactory()
	supported := factory.SupportedIndexTypes()

	expectedTypes := []IndexType{IndexTypeHNSW, IndexTypeIVFPQ, IndexTypeFlat}

	if len(supported) != len(expectedTypes) {
		t.Errorf("expected %d supported types, got %d", len(expectedTypes), len(supported))
	}

	for _, expected := range expectedTypes {
		found := false
		for _, actual := range supported {
			if actual == expected {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected index type %v not found in supported types", expected)
		}
	}
}

func TestIVFPQIntegration(t *testing.T) {
	ctx := context.Background()

	// Create IVF-PQ index through factory
	factory := NewIndexFactory()
	config := &IVFPQConfig{
		Dimension:     4,
		NClusters:     2,
		NProbes:       2,
		Metric:        util.L2Distance,
		MaxIterations: 10,
		Tolerance:     1e-4,
		RandomSeed:    42,
	}

	index, err := factory.CreateIndex(IndexTypeIVFPQ, config)
	if err != nil {
		t.Fatalf("failed to create IVF-PQ index: %v", err)
	}
	defer index.Close()

	// Test that we can't insert before training
	entry := &VectorEntry{
		ID:     "test1",
		Vector: []float32{1, 0, 0, 0},
	}

	err = index.Insert(ctx, entry)
	if err == nil {
		t.Errorf("expected error when inserting into untrained index")
	}

	// This test verifies the integration works but doesn't train the index
	// since that would require access to the underlying IVF-PQ implementation
	// which is abstracted away by the interface
}

func TestDefaultIndexFactory(t *testing.T) {
	// Verify the default factory is available
	if DefaultIndexFactory == nil {
		t.Errorf("DefaultIndexFactory should not be nil")
	}

	// Test that it works
	config := &HNSWConfig{
		Dimension:      64,
		M:              16,
		EfConstruction: 200,
		EfSearch:       50,
		ML:             1.0 / 2.303,
		Metric:         util.L2Distance,
	}

	index, err := DefaultIndexFactory.CreateIndex(IndexTypeHNSW, config)
	if err != nil {
		t.Errorf("DefaultIndexFactory failed to create index: %v", err)
	}

	if index != nil {
		index.Close()
	}
}
