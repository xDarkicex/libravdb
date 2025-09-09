package filter

import (
	"context"
	"testing"
	"time"
)

// TestFilterIntegration tests the complete filtering system with complex scenarios
func TestFilterIntegration(t *testing.T) {
	ctx := context.Background()

	// Create test data with various field types
	baseTime := time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC)
	entries := []*VectorEntry{
		{
			ID: "1",
			Metadata: map[string]interface{}{
				"category":   "electronics",
				"price":      299.99,
				"brand":      "apple",
				"tags":       []string{"smartphone", "premium", "5g"},
				"rating":     4.5,
				"in_stock":   true,
				"created_at": baseTime,
				"colors":     []string{"black", "white"},
			},
		},
		{
			ID: "2",
			Metadata: map[string]interface{}{
				"category":   "electronics",
				"price":      199.99,
				"brand":      "samsung",
				"tags":       []string{"smartphone", "budget", "4g"},
				"rating":     4.2,
				"in_stock":   true,
				"created_at": baseTime.Add(24 * time.Hour),
				"colors":     []string{"blue", "red"},
			},
		},
		{
			ID: "3",
			Metadata: map[string]interface{}{
				"category":   "books",
				"price":      29.99,
				"brand":      "penguin",
				"tags":       []string{"fiction", "bestseller"},
				"rating":     4.8,
				"in_stock":   false,
				"created_at": baseTime.Add(48 * time.Hour),
				"colors":     []string{"multicolor"},
			},
		},
		{
			ID: "4",
			Metadata: map[string]interface{}{
				"category":   "electronics",
				"price":      599.99,
				"brand":      "apple",
				"tags":       []string{"laptop", "premium", "m1"},
				"rating":     4.9,
				"in_stock":   true,
				"created_at": baseTime.Add(72 * time.Hour),
				"colors":     []string{"silver", "space_gray"},
			},
		},
		{
			ID: "5",
			Metadata: map[string]interface{}{
				"category":   "clothing",
				"price":      79.99,
				"brand":      "nike",
				"tags":       []string{"shoes", "running", "breathable"},
				"rating":     4.3,
				"in_stock":   true,
				"created_at": baseTime.Add(96 * time.Hour),
				"colors":     []string{"black", "white", "red"},
			},
		},
	}

	t.Run("complex AND filter", func(t *testing.T) {
		// Find electronics that are premium and in stock
		filter := NewAndFilter(
			NewEqualityFilter("category", "electronics"),
			NewContainsAnyFilter("tags", []interface{}{"premium"}),
			NewEqualityFilter("in_stock", true),
		)

		result, err := filter.Apply(ctx, entries)
		if err != nil {
			t.Fatalf("Apply() error = %v", err)
		}

		expected := []string{"1", "4"} // Apple products
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
	})

	t.Run("complex OR with nested AND", func(t *testing.T) {
		// Find (expensive electronics) OR (highly rated books)
		expensiveElectronics := NewAndFilter(
			NewEqualityFilter("category", "electronics"),
			NewGreaterThanFilter("price", 500),
		)

		highlyRatedBooks := NewAndFilter(
			NewEqualityFilter("category", "books"),
			NewGreaterThanFilter("rating", 4.5),
		)

		filter := NewOrFilter(expensiveElectronics, highlyRatedBooks)

		result, err := filter.Apply(ctx, entries)
		if err != nil {
			t.Fatalf("Apply() error = %v", err)
		}

		expected := []string{"3", "4"} // Book and expensive laptop
		if len(result) != len(expected) {
			t.Errorf("Apply() returned %d results, expected %d", len(result), len(expected))
		}
	})

	t.Run("range filter with time", func(t *testing.T) {
		// Find items created in the first 3 days
		filter := NewBetweenFilter("created_at", baseTime, baseTime.Add(72*time.Hour))

		result, err := filter.Apply(ctx, entries)
		if err != nil {
			t.Fatalf("Apply() error = %v", err)
		}

		expected := []string{"1", "2", "3", "4"}
		if len(result) != len(expected) {
			t.Errorf("Apply() returned %d results, expected %d", len(result), len(expected))
		}
	})

	t.Run("containment filter with arrays", func(t *testing.T) {
		// Find items that have both black and white colors
		filter := NewContainsAllFilter("colors", []interface{}{"black", "white"})

		result, err := filter.Apply(ctx, entries)
		if err != nil {
			t.Fatalf("Apply() error = %v", err)
		}

		expected := []string{"1", "5"} // iPhone and Nike shoes
		if len(result) != len(expected) {
			t.Errorf("Apply() returned %d results, expected %d", len(result), len(expected))
		}
	})

	t.Run("NOT filter", func(t *testing.T) {
		// Find items that are NOT electronics
		filter := NewNotFilter(NewEqualityFilter("category", "electronics"))

		result, err := filter.Apply(ctx, entries)
		if err != nil {
			t.Fatalf("Apply() error = %v", err)
		}

		expected := []string{"3", "5"} // Book and shoes
		if len(result) != len(expected) {
			t.Errorf("Apply() returned %d results, expected %d", len(result), len(expected))
		}
	})

	t.Run("highly complex nested filter", func(t *testing.T) {
		// Find: (Apple products OR Samsung products) AND (in stock) AND (price < 400) AND NOT (books)
		appleOrSamsung := NewOrFilter(
			NewEqualityFilter("brand", "apple"),
			NewEqualityFilter("brand", "samsung"),
		)

		inStockAndAffordable := NewAndFilter(
			NewEqualityFilter("in_stock", true),
			NewLessThanFilter("price", 400),
		)

		notBooks := NewNotFilter(NewEqualityFilter("category", "books"))

		complexFilter := NewAndFilter(appleOrSamsung, inStockAndAffordable, notBooks)

		result, err := complexFilter.Apply(ctx, entries)
		if err != nil {
			t.Fatalf("Apply() error = %v", err)
		}

		expected := []string{"1", "2"} // iPhone and Samsung phone
		if len(result) != len(expected) {
			t.Errorf("Apply() returned %d results, expected %d", len(result), len(expected))
		}
	})
}

// TestFilterWithParser tests the integration of filters with the parser
func TestFilterWithParser(t *testing.T) {
	ctx := context.Background()

	schema := map[string]FieldType{
		"category":   StringField,
		"price":      FloatField,
		"brand":      StringField,
		"tags":       StringArrayField,
		"rating":     FloatField,
		"in_stock":   BoolField,
		"created_at": TimeField,
	}

	parser := NewFilterParser(schema)

	entries := []*VectorEntry{
		{ID: "1", Metadata: map[string]interface{}{"category": "electronics", "price": 299.99, "brand": "apple"}},
		{ID: "2", Metadata: map[string]interface{}{"category": "books", "price": 29.99, "brand": "penguin"}},
	}

	t.Run("create and apply equality filter", func(t *testing.T) {
		filter, err := parser.CreateEqualityFilter("category", "electronics")
		if err != nil {
			t.Fatalf("CreateEqualityFilter() error = %v", err)
		}

		result, err := filter.Apply(ctx, entries)
		if err != nil {
			t.Fatalf("Apply() error = %v", err)
		}

		if len(result) != 1 || result[0].ID != "1" {
			t.Errorf("Apply() returned unexpected results")
		}
	})

	t.Run("create and apply range filter", func(t *testing.T) {
		filter, err := parser.CreateRangeFilter("price", "25.00", "100.00")
		if err != nil {
			t.Fatalf("CreateRangeFilter() error = %v", err)
		}

		result, err := filter.Apply(ctx, entries)
		if err != nil {
			t.Fatalf("Apply() error = %v", err)
		}

		if len(result) != 1 || result[0].ID != "2" {
			t.Errorf("Apply() returned unexpected results")
		}
	})
}

// TestFilterEdgeCases tests various edge cases and error conditions
func TestFilterEdgeCases(t *testing.T) {
	ctx := context.Background()

	entries := []*VectorEntry{
		{ID: "1", Metadata: map[string]interface{}{"field": nil}},
		{ID: "2", Metadata: map[string]interface{}{"field": ""}},
		{ID: "3", Metadata: map[string]interface{}{"field": 0}},
		{ID: "4", Metadata: map[string]interface{}{"field": false}},
		{ID: "5", Metadata: map[string]interface{}{}}, // Empty metadata
		{ID: "6", Metadata: nil},                      // Nil metadata
	}

	t.Run("equality filter with nil values", func(t *testing.T) {
		filter := NewEqualityFilter("field", nil)

		// Should fail validation
		err := filter.Validate()
		if err == nil {
			t.Error("Validate() should fail for nil value")
		}
	})

	t.Run("equality filter with empty string", func(t *testing.T) {
		filter := NewEqualityFilter("field", "")

		result, err := filter.Apply(ctx, entries)
		if err != nil {
			t.Fatalf("Apply() error = %v", err)
		}

		if len(result) != 1 || result[0].ID != "2" {
			t.Errorf("Apply() should match empty string")
		}
	})

	t.Run("range filter with zero values", func(t *testing.T) {
		// Create entries with only numeric values for this test
		numericEntries := []*VectorEntry{
			{ID: "1", Metadata: map[string]interface{}{"value": -5}},
			{ID: "2", Metadata: map[string]interface{}{"value": 0}},
			{ID: "3", Metadata: map[string]interface{}{"value": 5}},
			{ID: "4", Metadata: map[string]interface{}{"value": 10}},
		}

		filter := NewBetweenFilter("value", -1, 1)

		result, err := filter.Apply(ctx, numericEntries)
		if err != nil {
			t.Fatalf("Apply() error = %v", err)
		}

		if len(result) != 1 || result[0].ID != "2" {
			t.Errorf("Apply() should match zero value, got %d results", len(result))
			for _, r := range result {
				t.Logf("Result ID: %s, value: %v", r.ID, r.Metadata["value"])
			}
		}
	})

	t.Run("containment filter with empty arrays", func(t *testing.T) {
		entriesWithArrays := []*VectorEntry{
			{ID: "1", Metadata: map[string]interface{}{"tags": []string{}}},
			{ID: "2", Metadata: map[string]interface{}{"tags": []string{"tag1"}}},
		}

		filter := NewContainsAnyFilter("tags", []interface{}{"tag1"})

		result, err := filter.Apply(ctx, entriesWithArrays)
		if err != nil {
			t.Fatalf("Apply() error = %v", err)
		}

		if len(result) != 1 || result[0].ID != "2" {
			t.Errorf("Apply() should not match empty array")
		}
	})

	t.Run("logical filter with empty results", func(t *testing.T) {
		// Create filters that match nothing
		filter1 := NewEqualityFilter("nonexistent", "value")
		filter2 := NewEqualityFilter("another_nonexistent", "value")

		andFilter := NewAndFilter(filter1, filter2)
		orFilter := NewOrFilter(filter1, filter2)

		andResult, err := andFilter.Apply(ctx, entries)
		if err != nil {
			t.Fatalf("AND Apply() error = %v", err)
		}

		orResult, err := orFilter.Apply(ctx, entries)
		if err != nil {
			t.Fatalf("OR Apply() error = %v", err)
		}

		if len(andResult) != 0 {
			t.Errorf("AND filter should return empty results")
		}

		if len(orResult) != 0 {
			t.Errorf("OR filter should return empty results")
		}
	})
}
