package flat

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/xDarkicex/libravdb/internal/util"
)

func TestNewFlat(t *testing.T) {
	tests := []struct {
		name      string
		config    *Config
		expectErr bool
	}{
		{
			name: "valid config",
			config: &Config{
				Dimension: 128,
				Metric:    util.CosineDistance,
			},
			expectErr: false,
		},
		{
			name: "zero dimension",
			config: &Config{
				Dimension: 0,
				Metric:    util.CosineDistance,
			},
			expectErr: true,
		},
		{
			name: "negative dimension",
			config: &Config{
				Dimension: -1,
				Metric:    util.CosineDistance,
			},
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			idx, err := NewFlat(tt.config)
			if tt.expectErr {
				if err == nil {
					t.Error("expected error but got none")
				}
				return
			}
			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}
			if idx == nil {
				t.Error("expected index but got nil")
			}
		})
	}
}

func TestFlatInsert(t *testing.T) {
	config := &Config{
		Dimension: 3,
		Metric:    util.CosineDistance,
	}
	idx, err := NewFlat(config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}
	defer idx.Close()

	ctx := context.Background()

	// Test normal insertion
	entry := &VectorEntry{
		ID:     "test1",
		Vector: []float32{1.0, 2.0, 3.0},
		Metadata: map[string]interface{}{
			"category": "test",
		},
	}

	err = idx.Insert(ctx, entry)
	if err != nil {
		t.Errorf("failed to insert vector: %v", err)
	}

	if idx.Size() != 1 {
		t.Errorf("expected size 1, got %d", idx.Size())
	}

	// Test dimension mismatch
	badEntry := &VectorEntry{
		ID:     "test2",
		Vector: []float32{1.0, 2.0}, // Wrong dimension
	}

	err = idx.Insert(ctx, badEntry)
	if err == nil {
		t.Error("expected error for dimension mismatch")
	}

	// Test update existing entry
	updatedEntry := &VectorEntry{
		ID:     "test1",
		Vector: []float32{4.0, 5.0, 6.0},
		Metadata: map[string]interface{}{
			"category": "updated",
		},
	}

	err = idx.Insert(ctx, updatedEntry)
	if err != nil {
		t.Errorf("failed to update vector: %v", err)
	}

	if idx.Size() != 1 {
		t.Errorf("expected size 1 after update, got %d", idx.Size())
	}
}

func TestFlatSearch(t *testing.T) {
	config := &Config{
		Dimension: 3,
		Metric:    util.CosineDistance,
	}
	idx, err := NewFlat(config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}
	defer idx.Close()

	ctx := context.Background()

	// Insert test vectors
	vectors := []*VectorEntry{
		{ID: "v1", Vector: []float32{1.0, 0.0, 0.0}},
		{ID: "v2", Vector: []float32{0.0, 1.0, 0.0}},
		{ID: "v3", Vector: []float32{0.0, 0.0, 1.0}},
		{ID: "v4", Vector: []float32{1.0, 1.0, 0.0}},
	}

	for _, v := range vectors {
		if err := idx.Insert(ctx, v); err != nil {
			t.Fatalf("failed to insert vector %s: %v", v.ID, err)
		}
	}

	// Test search
	query := []float32{1.0, 0.0, 0.0}
	results, err := idx.Search(ctx, query, 2)
	if err != nil {
		t.Errorf("search failed: %v", err)
	}

	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}

	// First result should be v1 (exact match)
	if results[0].ID != "v1" {
		t.Errorf("expected first result to be v1, got %s", results[0].ID)
	}

	// Test search with k=0
	results, err = idx.Search(ctx, query, 0)
	if err != nil {
		t.Errorf("search with k=0 failed: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected 0 results for k=0, got %d", len(results))
	}

	// Test search with dimension mismatch
	badQuery := []float32{1.0, 0.0}
	_, err = idx.Search(ctx, badQuery, 1)
	if err == nil {
		t.Error("expected error for dimension mismatch in search")
	}
}

func TestFlatSearchAccuracy(t *testing.T) {
	config := &Config{
		Dimension: 2,
		Metric:    util.L2Distance,
	}
	idx, err := NewFlat(config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}
	defer idx.Close()

	ctx := context.Background()

	// Insert vectors in a known pattern
	vectors := []*VectorEntry{
		{ID: "origin", Vector: []float32{0.0, 0.0}},
		{ID: "x1", Vector: []float32{1.0, 0.0}},
		{ID: "x2", Vector: []float32{2.0, 0.0}},
		{ID: "y1", Vector: []float32{0.0, 1.0}},
		{ID: "diagonal", Vector: []float32{1.0, 1.0}},
	}

	for _, v := range vectors {
		if err := idx.Insert(ctx, v); err != nil {
			t.Fatalf("failed to insert vector %s: %v", v.ID, err)
		}
	}

	// Search from origin - should get exact distances
	query := []float32{0.0, 0.0}
	results, err := idx.Search(ctx, query, 5)
	if err != nil {
		t.Errorf("search failed: %v", err)
	}

	// Verify exact search results
	expectedOrder := []string{"origin", "x1", "y1", "diagonal", "x2"}
	expectedDistances := []float32{0.0, 1.0, 1.0, float32(math.Sqrt(2)), 2.0}

	for i, expected := range expectedOrder {
		if i >= len(results) {
			t.Errorf("missing result at index %d", i)
			continue
		}
		if results[i].ID != expected {
			t.Errorf("result %d: expected %s, got %s", i, expected, results[i].ID)
		}
		if math.Abs(float64(results[i].Score-expectedDistances[i])) > 1e-6 {
			t.Errorf("result %d: expected distance %f, got %f", i, expectedDistances[i], results[i].Score)
		}
	}
}

func TestFlatDelete(t *testing.T) {
	config := &Config{
		Dimension: 3,
		Metric:    util.CosineDistance,
	}
	idx, err := NewFlat(config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}
	defer idx.Close()

	ctx := context.Background()

	// Insert test vectors
	vectors := []*VectorEntry{
		{ID: "v1", Vector: []float32{1.0, 0.0, 0.0}},
		{ID: "v2", Vector: []float32{0.0, 1.0, 0.0}},
		{ID: "v3", Vector: []float32{0.0, 0.0, 1.0}},
	}

	for _, v := range vectors {
		if err := idx.Insert(ctx, v); err != nil {
			t.Fatalf("failed to insert vector %s: %v", v.ID, err)
		}
	}

	if idx.Size() != 3 {
		t.Errorf("expected size 3, got %d", idx.Size())
	}

	// Delete middle vector
	err = idx.Delete(ctx, "v2")
	if err != nil {
		t.Errorf("failed to delete vector: %v", err)
	}

	if idx.Size() != 2 {
		t.Errorf("expected size 2 after delete, got %d", idx.Size())
	}

	// Verify v2 is gone
	results, err := idx.Search(ctx, []float32{0.0, 1.0, 0.0}, 3)
	if err != nil {
		t.Errorf("search failed: %v", err)
	}

	for _, result := range results {
		if result.ID == "v2" {
			t.Error("deleted vector v2 still found in search results")
		}
	}

	// Test delete non-existent vector
	err = idx.Delete(ctx, "nonexistent")
	if err == nil {
		t.Error("expected error when deleting non-existent vector")
	}
}

func TestFlatMemoryUsage(t *testing.T) {
	config := &Config{
		Dimension: 100,
		Metric:    util.CosineDistance,
	}
	idx, err := NewFlat(config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}
	defer idx.Close()

	ctx := context.Background()

	// Initial memory usage should be minimal
	initialUsage := idx.MemoryUsage()
	if initialUsage < 0 {
		t.Error("memory usage should be non-negative")
	}

	// Insert vectors and check memory growth
	for i := 0; i < 100; i++ {
		vector := make([]float32, 100)
		for j := range vector {
			vector[j] = float32(i + j)
		}
		entry := &VectorEntry{
			ID:     fmt.Sprintf("v%d", i),
			Vector: vector,
			Metadata: map[string]interface{}{
				"index": i,
			},
		}
		if err := idx.Insert(ctx, entry); err != nil {
			t.Fatalf("failed to insert vector: %v", err)
		}
	}

	finalUsage := idx.MemoryUsage()
	if finalUsage <= initialUsage {
		t.Error("memory usage should increase after inserting vectors")
	}
}

func TestFlatPersistence(t *testing.T) {
	config := &Config{
		Dimension: 3,
		Metric:    util.CosineDistance,
	}
	idx, err := NewFlat(config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}

	ctx := context.Background()

	// Insert test data
	vectors := []*VectorEntry{
		{ID: "v1", Vector: []float32{1.0, 0.0, 0.0}, Metadata: map[string]interface{}{"type": "test"}},
		{ID: "v2", Vector: []float32{0.0, 1.0, 0.0}, Metadata: map[string]interface{}{"type": "test"}},
	}

	for _, v := range vectors {
		if err := idx.Insert(ctx, v); err != nil {
			t.Fatalf("failed to insert vector: %v", err)
		}
	}

	// Save to disk
	tempDir := t.TempDir()
	savePath := filepath.Join(tempDir, "flat_index.json")

	err = idx.SaveToDisk(ctx, savePath)
	if err != nil {
		t.Errorf("failed to save index: %v", err)
	}

	// Verify file exists
	if _, err := os.Stat(savePath); os.IsNotExist(err) {
		t.Error("saved file does not exist")
	}

	idx.Close()

	// Create new index and load from disk
	idx2, err := NewFlat(config)
	if err != nil {
		t.Fatalf("failed to create second index: %v", err)
	}
	defer idx2.Close()

	err = idx2.LoadFromDisk(ctx, savePath)
	if err != nil {
		t.Errorf("failed to load index: %v", err)
	}

	// Verify data was loaded correctly
	if idx2.Size() != 2 {
		t.Errorf("expected size 2 after loading, got %d", idx2.Size())
	}

	// Test search on loaded index
	results, err := idx2.Search(ctx, []float32{1.0, 0.0, 0.0}, 1)
	if err != nil {
		t.Errorf("search on loaded index failed: %v", err)
	}

	if len(results) != 1 || results[0].ID != "v1" {
		t.Error("loaded index search results incorrect")
	}
}

func TestFlatDistanceMetrics(t *testing.T) {
	metrics := []util.DistanceMetric{
		util.CosineDistance,
		util.L2Distance,
		util.InnerProduct,
	}

	for i, metric := range metrics {
		t.Run(fmt.Sprintf("metric_%d", i), func(t *testing.T) {
			config := &Config{
				Dimension: 3,
				Metric:    metric,
			}
			idx, err := NewFlat(config)
			if err != nil {
				t.Fatalf("failed to create index with metric %v: %v", metric, err)
			}
			defer idx.Close()

			ctx := context.Background()

			// Insert test vectors
			vectors := []*VectorEntry{
				{ID: "v1", Vector: []float32{1.0, 0.0, 0.0}},
				{ID: "v2", Vector: []float32{0.0, 1.0, 0.0}},
			}

			for _, v := range vectors {
				if err := idx.Insert(ctx, v); err != nil {
					t.Fatalf("failed to insert vector: %v", err)
				}
			}

			// Test search works with this metric
			results, err := idx.Search(ctx, []float32{1.0, 0.0, 0.0}, 2)
			if err != nil {
				t.Errorf("search failed with metric %v: %v", metric, err)
			}

			if len(results) != 2 {
				t.Errorf("expected 2 results with metric %v, got %d", metric, len(results))
			}
		})
	}
}

func TestFlatEmptyIndex(t *testing.T) {
	config := &Config{
		Dimension: 3,
		Metric:    util.CosineDistance,
	}
	idx, err := NewFlat(config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}
	defer idx.Close()

	ctx := context.Background()

	// Search on empty index
	results, err := idx.Search(ctx, []float32{1.0, 0.0, 0.0}, 5)
	if err != nil {
		t.Errorf("search on empty index failed: %v", err)
	}

	if len(results) != 0 {
		t.Errorf("expected 0 results on empty index, got %d", len(results))
	}

	if idx.Size() != 0 {
		t.Errorf("expected size 0 for empty index, got %d", idx.Size())
	}
}

func BenchmarkFlatInsert(b *testing.B) {
	config := &Config{
		Dimension: 128,
		Metric:    util.CosineDistance,
	}
	idx, err := NewFlat(config)
	if err != nil {
		b.Fatalf("failed to create index: %v", err)
	}
	defer idx.Close()

	ctx := context.Background()
	vector := make([]float32, 128)
	for i := range vector {
		vector[i] = float32(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		entry := &VectorEntry{
			ID:     fmt.Sprintf("v%d", i),
			Vector: vector,
		}
		idx.Insert(ctx, entry)
	}
}

func BenchmarkFlatSearch(b *testing.B) {
	config := &Config{
		Dimension: 128,
		Metric:    util.CosineDistance,
	}
	idx, err := NewFlat(config)
	if err != nil {
		b.Fatalf("failed to create index: %v", err)
	}
	defer idx.Close()

	ctx := context.Background()

	// Insert 1000 vectors
	for i := 0; i < 1000; i++ {
		vector := make([]float32, 128)
		for j := range vector {
			vector[j] = float32(i + j)
		}
		entry := &VectorEntry{
			ID:     fmt.Sprintf("v%d", i),
			Vector: vector,
		}
		idx.Insert(ctx, entry)
	}

	query := make([]float32, 128)
	for i := range query {
		query[i] = float32(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(ctx, query, 10)
	}
}
