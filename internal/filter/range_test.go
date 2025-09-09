package filter

import (
	"context"
	"testing"
	"time"
)

func TestRangeFilter_Apply(t *testing.T) {
	ctx := context.Background()

	entries := []*VectorEntry{
		{ID: "1", Metadata: map[string]interface{}{"price": 50}},
		{ID: "2", Metadata: map[string]interface{}{"price": 100}},
		{ID: "3", Metadata: map[string]interface{}{"price": 150}},
		{ID: "4", Metadata: map[string]interface{}{"price": 200}},
		{ID: "5", Metadata: map[string]interface{}{"name": "test"}}, // Different field
		{ID: "6", Metadata: nil},                                    // No metadata
	}

	tests := []struct {
		name     string
		filter   *RangeFilter
		expected []string
	}{
		{
			name:     "range with both bounds",
			filter:   NewBetweenFilter("price", 100, 150),
			expected: []string{"2", "3"},
		},
		{
			name:     "greater than filter",
			filter:   NewGreaterThanFilter("price", 100),
			expected: []string{"2", "3", "4"},
		},
		{
			name:     "less than filter",
			filter:   NewLessThanFilter("price", 150),
			expected: []string{"1", "2", "3"},
		},
		{
			name:     "no matches",
			filter:   NewBetweenFilter("price", 300, 400),
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

func TestRangeFilter_StringComparison(t *testing.T) {
	ctx := context.Background()

	entries := []*VectorEntry{
		{ID: "1", Metadata: map[string]interface{}{"name": "apple"}},
		{ID: "2", Metadata: map[string]interface{}{"name": "banana"}},
		{ID: "3", Metadata: map[string]interface{}{"name": "cherry"}},
		{ID: "4", Metadata: map[string]interface{}{"name": "date"}},
	}

	filter := NewBetweenFilter("name", "banana", "cherry")
	result, err := filter.Apply(ctx, entries)
	if err != nil {
		t.Fatalf("Apply() error = %v", err)
	}

	expected := []string{"2", "3"}
	if len(result) != len(expected) {
		t.Errorf("Apply() returned %d results, expected %d", len(result), len(expected))
	}
}

func TestRangeFilter_TimeComparison(t *testing.T) {
	ctx := context.Background()

	baseTime := time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC)
	entries := []*VectorEntry{
		{ID: "1", Metadata: map[string]interface{}{"created": baseTime}},
		{ID: "2", Metadata: map[string]interface{}{"created": baseTime.Add(24 * time.Hour)}},
		{ID: "3", Metadata: map[string]interface{}{"created": baseTime.Add(48 * time.Hour)}},
		{ID: "4", Metadata: map[string]interface{}{"created": baseTime.Add(72 * time.Hour)}},
	}

	filter := NewBetweenFilter("created", baseTime.Add(12*time.Hour), baseTime.Add(60*time.Hour))
	result, err := filter.Apply(ctx, entries)
	if err != nil {
		t.Fatalf("Apply() error = %v", err)
	}

	expected := []string{"2", "3"}
	if len(result) != len(expected) {
		t.Errorf("Apply() returned %d results, expected %d", len(result), len(expected))
	}
}

func TestRangeFilter_NumericTypeConversion(t *testing.T) {
	ctx := context.Background()

	entries := []*VectorEntry{
		{ID: "1", Metadata: map[string]interface{}{"value": int(50)}},
		{ID: "2", Metadata: map[string]interface{}{"value": float32(75.5)}},
		{ID: "3", Metadata: map[string]interface{}{"value": float64(100.0)}},
		{ID: "4", Metadata: map[string]interface{}{"value": int64(125)}},
	}

	filter := NewBetweenFilter("value", 60, 110)
	result, err := filter.Apply(ctx, entries)
	if err != nil {
		t.Fatalf("Apply() error = %v", err)
	}

	expected := []string{"2", "3"}
	if len(result) != len(expected) {
		t.Errorf("Apply() returned %d results, expected %d", len(result), len(expected))
	}
}

func TestRangeFilter_Validate(t *testing.T) {
	tests := []struct {
		name      string
		filter    *RangeFilter
		wantError bool
	}{
		{
			name:      "valid range filter",
			filter:    NewBetweenFilter("field", 10, 20),
			wantError: false,
		},
		{
			name:      "valid greater than filter",
			filter:    NewGreaterThanFilter("field", 10),
			wantError: false,
		},
		{
			name:      "valid less than filter",
			filter:    NewLessThanFilter("field", 20),
			wantError: false,
		},
		{
			name:      "empty field name",
			filter:    NewBetweenFilter("", 10, 20),
			wantError: true,
		},
		{
			name:      "no bounds specified",
			filter:    NewRangeFilter("field", nil, nil),
			wantError: true,
		},
		{
			name:      "min greater than max",
			filter:    NewBetweenFilter("field", 20, 10),
			wantError: true,
		},
		{
			name:      "incomparable types",
			filter:    NewBetweenFilter("field", "string", 10),
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

func TestRangeFilter_EstimateSelectivity(t *testing.T) {
	tests := []struct {
		name     string
		filter   *RangeFilter
		expected float64
	}{
		{
			name:     "both bounds",
			filter:   NewBetweenFilter("field", 10, 20),
			expected: 0.3,
		},
		{
			name:     "single bound",
			filter:   NewGreaterThanFilter("field", 10),
			expected: 0.5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			selectivity := tt.filter.EstimateSelectivity()
			if selectivity != tt.expected {
				t.Errorf("EstimateSelectivity() = %f, want %f", selectivity, tt.expected)
			}
		})
	}
}

func TestRangeFilter_String(t *testing.T) {
	tests := []struct {
		name     string
		filter   *RangeFilter
		expected string
	}{
		{
			name:     "both bounds",
			filter:   NewBetweenFilter("price", 10, 20),
			expected: "price BETWEEN 10 AND 20",
		},
		{
			name:     "greater than",
			filter:   NewGreaterThanFilter("price", 10),
			expected: "price >= 10",
		},
		{
			name:     "less than",
			filter:   NewLessThanFilter("price", 20),
			expected: "price <= 20",
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
