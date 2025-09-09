package filter

import (
	"context"
	"testing"
)

func TestLogicalFilter_Apply(t *testing.T) {
	ctx := context.Background()

	entries := []*VectorEntry{
		{ID: "1", Metadata: map[string]interface{}{"category": "electronics", "price": 100}},
		{ID: "2", Metadata: map[string]interface{}{"category": "electronics", "price": 200}},
		{ID: "3", Metadata: map[string]interface{}{"category": "books", "price": 50}},
		{ID: "4", Metadata: map[string]interface{}{"category": "books", "price": 150}},
		{ID: "5", Metadata: map[string]interface{}{"category": "clothing", "price": 75}},
	}

	// Create base filters
	electronicsFilter := NewEqualityFilter("category", "electronics")
	booksFilter := NewEqualityFilter("category", "books")
	expensiveFilter := NewGreaterThanFilter("price", 100)

	tests := []struct {
		name     string
		filter   *LogicalFilter
		expected []string
	}{
		{
			name:     "AND filter - electronics AND expensive",
			filter:   NewAndFilter(electronicsFilter, expensiveFilter),
			expected: []string{"1", "2"}, // Both entries 1 (price=100) and 2 (price=200) are >= 100
		},
		{
			name:     "OR filter - electronics OR books",
			filter:   NewOrFilter(electronicsFilter, booksFilter),
			expected: []string{"1", "2", "3", "4"},
		},
		{
			name:     "NOT filter - NOT electronics",
			filter:   NewNotFilter(electronicsFilter),
			expected: []string{"3", "4", "5"},
		},
		{
			name:     "complex AND with multiple filters",
			filter:   NewAndFilter(booksFilter, expensiveFilter),
			expected: []string{"4"},
		},
		{
			name:     "AND with no matches",
			filter:   NewAndFilter(electronicsFilter, NewEqualityFilter("category", "books")),
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

func TestLogicalFilter_NestedLogical(t *testing.T) {
	ctx := context.Background()

	entries := []*VectorEntry{
		{ID: "1", Metadata: map[string]interface{}{"category": "electronics", "price": 100, "brand": "apple"}},
		{ID: "2", Metadata: map[string]interface{}{"category": "electronics", "price": 200, "brand": "samsung"}},
		{ID: "3", Metadata: map[string]interface{}{"category": "books", "price": 50, "brand": "penguin"}},
		{ID: "4", Metadata: map[string]interface{}{"category": "books", "price": 150, "brand": "oreilly"}},
	}

	// (electronics AND apple) OR (books AND expensive)
	electronicsAndApple := NewAndFilter(
		NewEqualityFilter("category", "electronics"),
		NewEqualityFilter("brand", "apple"),
	)
	booksAndExpensive := NewAndFilter(
		NewEqualityFilter("category", "books"),
		NewGreaterThanFilter("price", 100),
	)
	complexFilter := NewOrFilter(electronicsAndApple, booksAndExpensive)

	result, err := complexFilter.Apply(ctx, entries)
	if err != nil {
		t.Fatalf("Apply() error = %v", err)
	}

	expected := []string{"1", "4"}
	if len(result) != len(expected) {
		t.Errorf("Apply() returned %d results, expected %d", len(result), len(expected))
	}

	resultIDs := make(map[string]bool)
	for _, entry := range result {
		resultIDs[entry.ID] = true
	}

	for _, expectedID := range expected {
		if !resultIDs[expectedID] {
			t.Errorf("Apply() missing expected ID %s", expectedID)
		}
	}
}

func TestLogicalFilter_OrDeduplication(t *testing.T) {
	ctx := context.Background()

	entries := []*VectorEntry{
		{ID: "1", Metadata: map[string]interface{}{"category": "electronics", "price": 100}},
		{ID: "2", Metadata: map[string]interface{}{"category": "books", "price": 200}},
	}

	// Create filters that both match entry 1
	filter1 := NewEqualityFilter("category", "electronics")
	filter2 := NewEqualityFilter("price", 100)
	orFilter := NewOrFilter(filter1, filter2)

	result, err := orFilter.Apply(ctx, entries)
	if err != nil {
		t.Fatalf("Apply() error = %v", err)
	}

	// Should only return entry 1 once, even though both filters match it
	if len(result) != 1 {
		t.Errorf("Apply() returned %d results, expected 1 (deduplication failed)", len(result))
	}

	if result[0].ID != "1" {
		t.Errorf("Apply() returned ID %s, expected 1", result[0].ID)
	}
}

func TestLogicalFilter_Validate(t *testing.T) {
	validFilter := NewEqualityFilter("field", "value")
	invalidFilter := NewEqualityFilter("", "value") // Invalid: empty field

	tests := []struct {
		name      string
		filter    *LogicalFilter
		wantError bool
	}{
		{
			name:      "valid AND filter",
			filter:    NewAndFilter(validFilter),
			wantError: false,
		},
		{
			name:      "valid OR filter",
			filter:    NewOrFilter(validFilter),
			wantError: false,
		},
		{
			name:      "valid NOT filter",
			filter:    NewNotFilter(validFilter),
			wantError: false,
		},
		{
			name:      "empty filters list",
			filter:    &LogicalFilter{Operator: AndOperator, Filters: []Filter{}},
			wantError: true,
		},
		{
			name:      "NOT with multiple filters",
			filter:    &LogicalFilter{Operator: NotOperator, Filters: []Filter{validFilter, validFilter}},
			wantError: true,
		},
		{
			name:      "invalid child filter",
			filter:    NewAndFilter(invalidFilter),
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

func TestLogicalFilter_EstimateSelectivity(t *testing.T) {
	// Create filters with known selectivities
	filter1 := NewEqualityFilter("field1", "value1") // 0.1 selectivity
	filter2 := NewEqualityFilter("field2", "value2") // 0.1 selectivity

	tests := []struct {
		name     string
		filter   *LogicalFilter
		expected float64
	}{
		{
			name:     "AND selectivity",
			filter:   NewAndFilter(filter1, filter2),
			expected: 0.01, // 0.1 * 0.1
		},
		{
			name:     "NOT selectivity",
			filter:   NewNotFilter(filter1),
			expected: 0.9, // 1.0 - 0.1
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			selectivity := tt.filter.EstimateSelectivity()
			// Use approximate comparison for floating point values
			if abs(selectivity-tt.expected) > 0.0001 {
				t.Errorf("EstimateSelectivity() = %f, want %f", selectivity, tt.expected)
			}
		})
	}
}

func TestLogicalFilter_String(t *testing.T) {
	filter1 := NewEqualityFilter("category", "electronics")
	filter2 := NewEqualityFilter("price", 100)

	tests := []struct {
		name     string
		filter   *LogicalFilter
		expected string
	}{
		{
			name:     "AND filter",
			filter:   NewAndFilter(filter1, filter2),
			expected: "(category == electronics) AND (price == 100)",
		},
		{
			name:     "OR filter",
			filter:   NewOrFilter(filter1, filter2),
			expected: "(category == electronics) OR (price == 100)",
		},
		{
			name:     "NOT filter",
			filter:   NewNotFilter(filter1),
			expected: "NOT (category == electronics)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			str := tt.filter.String()
			if str != tt.expected {
				t.Errorf("String() = %s, want %s", str, tt.expected)
			}
		})
	}
}

// Helper function for floating point comparison
func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
