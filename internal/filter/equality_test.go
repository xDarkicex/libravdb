package filter

import (
	"context"
	"testing"
)

func TestEqualityFilter_Apply(t *testing.T) {
	ctx := context.Background()

	entries := []*VectorEntry{
		{ID: "1", Metadata: map[string]interface{}{"category": "electronics", "price": 100}},
		{ID: "2", Metadata: map[string]interface{}{"category": "books", "price": 20}},
		{ID: "3", Metadata: map[string]interface{}{"category": "electronics", "price": 200}},
		{ID: "4", Metadata: map[string]interface{}{"category": "clothing", "active": true}},
		{ID: "5", Metadata: nil}, // No metadata
		{ID: "6", Metadata: map[string]interface{}{"other": "value"}}, // Missing field
	}

	tests := []struct {
		name     string
		filter   *EqualityFilter
		expected []string // Expected IDs
	}{
		{
			name:     "string equality match",
			filter:   NewEqualityFilter("category", "electronics"),
			expected: []string{"1", "3"},
		},
		{
			name:     "numeric equality match",
			filter:   NewEqualityFilter("price", 100),
			expected: []string{"1"},
		},
		{
			name:     "boolean equality match",
			filter:   NewEqualityFilter("active", true),
			expected: []string{"4"},
		},
		{
			name:     "no matches",
			filter:   NewEqualityFilter("category", "nonexistent"),
			expected: []string{},
		},
		{
			name:     "missing field",
			filter:   NewEqualityFilter("missing", "value"),
			expected: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := tt.filter.Apply(ctx, entries)
			if err != nil {
				t.Fatalf("Apply() error = %v", err)
			}

			if len(result) != len(tt.expected) {
				t.Errorf("Apply() returned %d results, expected %d", len(result), len(tt.expected))
			}

			resultIDs := make(map[string]bool)
			for _, entry := range result {
				resultIDs[entry.ID] = true
			}

			for _, expectedID := range tt.expected {
				if !resultIDs[expectedID] {
					t.Errorf("Apply() missing expected ID %s", expectedID)
				}
			}
		})
	}
}

func TestEqualityFilter_NumericTypeConversion(t *testing.T) {
	ctx := context.Background()

	entries := []*VectorEntry{
		{ID: "1", Metadata: map[string]interface{}{"value": int(42)}},
		{ID: "2", Metadata: map[string]interface{}{"value": int32(42)}},
		{ID: "3", Metadata: map[string]interface{}{"value": int64(42)}},
		{ID: "4", Metadata: map[string]interface{}{"value": float32(42.0)}},
		{ID: "5", Metadata: map[string]interface{}{"value": float64(42.0)}},
		{ID: "6", Metadata: map[string]interface{}{"value": 43}},
	}

	filter := NewEqualityFilter("value", 42)
	result, err := filter.Apply(ctx, entries)
	if err != nil {
		t.Fatalf("Apply() error = %v", err)
	}

	// Should match all entries with value 42, regardless of numeric type
	expectedCount := 5
	if len(result) != expectedCount {
		t.Errorf("Apply() returned %d results, expected %d", len(result), expectedCount)
	}
}

func TestEqualityFilter_Validate(t *testing.T) {
	tests := []struct {
		name      string
		filter    *EqualityFilter
		wantError bool
	}{
		{
			name:      "valid filter",
			filter:    NewEqualityFilter("field", "value"),
			wantError: false,
		},
		{
			name:      "empty field name",
			filter:    NewEqualityFilter("", "value"),
			wantError: true,
		},
		{
			name:      "nil value",
			filter:    NewEqualityFilter("field", nil),
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.filter.Validate()
			if (err != nil) != tt.wantError {
				t.Errorf("Validate() error = %v, wantError %v", err, tt.wantError)
			}
		})
	}
}

func TestEqualityFilter_EstimateSelectivity(t *testing.T) {
	filter := NewEqualityFilter("field", "value")
	selectivity := filter.EstimateSelectivity()

	if selectivity <= 0 || selectivity > 1 {
		t.Errorf("EstimateSelectivity() = %f, want value between 0 and 1", selectivity)
	}
}

func TestEqualityFilter_String(t *testing.T) {
	filter := NewEqualityFilter("category", "electronics")
	str := filter.String()

	expected := "category == electronics"
	if str != expected {
		t.Errorf("String() = %s, want %s", str, expected)
	}
}
