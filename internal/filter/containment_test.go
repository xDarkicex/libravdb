package filter

import (
	"context"
	"testing"
)

func TestContainmentFilter_Apply(t *testing.T) {
	ctx := context.Background()

	entries := []*VectorEntry{
		{ID: "1", Metadata: map[string]interface{}{"tags": []string{"red", "blue", "green"}}},
		{ID: "2", Metadata: map[string]interface{}{"tags": []string{"red", "yellow"}}},
		{ID: "3", Metadata: map[string]interface{}{"tags": []string{"blue", "green", "purple"}}},
		{ID: "4", Metadata: map[string]interface{}{"tags": "red"}},                                    // Single value, not array
		{ID: "5", Metadata: map[string]interface{}{"categories": []string{"electronics", "gadgets"}}}, // Different field
		{ID: "6", Metadata: nil}, // No metadata
	}

	tests := []struct {
		name     string
		filter   *ContainmentFilter
		expected []string
	}{
		{
			name:     "contains any - multiple matches",
			filter:   NewContainsAnyFilter("tags", []interface{}{"red", "purple"}),
			expected: []string{"1", "2", "3", "4"},
		},
		{
			name:     "contains any - single match",
			filter:   NewContainsAnyFilter("tags", []interface{}{"yellow"}),
			expected: []string{"2"},
		},
		{
			name:     "contains all - match",
			filter:   NewContainsAllFilter("tags", []interface{}{"red", "blue"}),
			expected: []string{"1"},
		},
		{
			name:     "contains all - no match",
			filter:   NewContainsAllFilter("tags", []interface{}{"red", "purple"}),
			expected: []string{},
		},
		{
			name:     "exact match",
			filter:   NewExactMatchFilter("tags", []interface{}{"red", "yellow"}),
			expected: []string{"2"},
		},
		{
			name:     "exact match - different order",
			filter:   NewExactMatchFilter("tags", []interface{}{"yellow", "red"}),
			expected: []string{"2"},
		},
		{
			name:     "no matches",
			filter:   NewContainsAnyFilter("tags", []interface{}{"nonexistent"}),
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

func TestContainmentFilter_NumericArrays(t *testing.T) {
	ctx := context.Background()

	entries := []*VectorEntry{
		{ID: "1", Metadata: map[string]interface{}{"numbers": []int{1, 2, 3}}},
		{ID: "2", Metadata: map[string]interface{}{"numbers": []float64{1.0, 2.5, 3.0}}},
		{ID: "3", Metadata: map[string]interface{}{"numbers": []interface{}{1, 2.0, 3}}},
	}

	filter := NewContainsAnyFilter("numbers", []interface{}{2, 2.5})
	result, err := filter.Apply(ctx, entries)
	if err != nil {
		t.Fatalf("Apply() error = %v", err)
	}

	// Should match all entries due to numeric type conversion
	expected := []string{"1", "2", "3"}
	if len(result) != len(expected) {
		t.Errorf("Apply() returned %d results, expected %d", len(result), len(expected))
	}
}

func TestContainmentFilter_SingleValueAsArray(t *testing.T) {
	ctx := context.Background()

	entries := []*VectorEntry{
		{ID: "1", Metadata: map[string]interface{}{"category": "electronics"}},
		{ID: "2", Metadata: map[string]interface{}{"category": "books"}},
	}

	filter := NewContainsAnyFilter("category", []interface{}{"electronics", "clothing"})
	result, err := filter.Apply(ctx, entries)
	if err != nil {
		t.Fatalf("Apply() error = %v", err)
	}

	// Should match entry 1 even though category is not an array
	expected := []string{"1"}
	if len(result) != len(expected) {
		t.Errorf("Apply() returned %d results, expected %d", len(result), len(expected))
	}
}

func TestContainmentFilter_Validate(t *testing.T) {
	tests := []struct {
		name      string
		filter    *ContainmentFilter
		wantError bool
	}{
		{
			name:      "valid filter",
			filter:    NewContainsAnyFilter("field", []interface{}{"value1", "value2"}),
			wantError: false,
		},
		{
			name:      "empty field name",
			filter:    NewContainsAnyFilter("", []interface{}{"value"}),
			wantError: true,
		},
		{
			name:      "empty values list",
			filter:    NewContainsAnyFilter("field", []interface{}{}),
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

func TestContainmentFilter_EstimateSelectivity(t *testing.T) {
	tests := []struct {
		name     string
		mode     ContainmentMode
		expected float64
	}{
		{
			name:     "contains any",
			mode:     ContainsAny,
			expected: 0.4,
		},
		{
			name:     "contains all",
			mode:     ContainsAll,
			expected: 0.2,
		},
		{
			name:     "exact match",
			mode:     ExactMatch,
			expected: 0.1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filter := &ContainmentFilter{
				Field:  "field",
				Values: []interface{}{"value"},
				Mode:   tt.mode,
			}
			selectivity := filter.EstimateSelectivity()
			if selectivity != tt.expected {
				t.Errorf("EstimateSelectivity() = %f, want %f", selectivity, tt.expected)
			}
		})
	}
}

func TestContainmentFilter_String(t *testing.T) {
	tests := []struct {
		name     string
		mode     ContainmentMode
		expected string
	}{
		{
			name:     "contains any",
			mode:     ContainsAny,
			expected: "tags CONTAINS ANY [red blue]",
		},
		{
			name:     "contains all",
			mode:     ContainsAll,
			expected: "tags CONTAINS ALL [red blue]",
		},
		{
			name:     "exact match",
			mode:     ExactMatch,
			expected: "tags EXACTLY [red blue]",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filter := &ContainmentFilter{
				Field:  "tags",
				Values: []interface{}{"red", "blue"},
				Mode:   tt.mode,
			}
			str := filter.String()
			if str != tt.expected {
				t.Errorf("String() = %s, want %s", str, tt.expected)
			}
		})
	}
}
