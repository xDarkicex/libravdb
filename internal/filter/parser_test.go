package filter

import (
	"testing"
	"time"
)

func TestFilterParser_ParseValue(t *testing.T) {
	schema := map[string]FieldType{
		"name":    StringField,
		"age":     IntField,
		"price":   FloatField,
		"active":  BoolField,
		"created": TimeField,
		"tags":    StringArrayField,
		"numbers": IntArrayField,
		"scores":  FloatArrayField,
	}

	parser := NewFilterParser(schema)

	tests := []struct {
		name      string
		field     string
		value     string
		expected  interface{}
		wantError bool
	}{
		{
			name:      "string field",
			field:     "name",
			value:     "test",
			expected:  "test",
			wantError: false,
		},
		{
			name:      "integer field",
			field:     "age",
			value:     "25",
			expected:  int64(25),
			wantError: false,
		},
		{
			name:      "float field",
			field:     "price",
			value:     "19.99",
			expected:  19.99,
			wantError: false,
		},
		{
			name:      "boolean field - true",
			field:     "active",
			value:     "true",
			expected:  true,
			wantError: false,
		},
		{
			name:      "boolean field - false",
			field:     "active",
			value:     "false",
			expected:  false,
			wantError: false,
		},
		{
			name:      "time field - RFC3339",
			field:     "created",
			value:     "2023-01-01T12:00:00Z",
			expected:  time.Date(2023, 1, 1, 12, 0, 0, 0, time.UTC),
			wantError: false,
		},
		{
			name:      "time field - date only",
			field:     "created",
			value:     "2023-01-01",
			expected:  time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC),
			wantError: false,
		},
		{
			name:      "string array field",
			field:     "tags",
			value:     "red,blue,green",
			expected:  []string{"red", "blue", "green"},
			wantError: false,
		},
		{
			name:      "integer array field",
			field:     "numbers",
			value:     "1,2,3",
			expected:  []int64{1, 2, 3},
			wantError: false,
		},
		{
			name:      "float array field",
			field:     "scores",
			value:     "1.5,2.7,3.9",
			expected:  []float64{1.5, 2.7, 3.9},
			wantError: false,
		},
		{
			name:      "invalid integer",
			field:     "age",
			value:     "not_a_number",
			wantError: true,
		},
		{
			name:      "invalid float",
			field:     "price",
			value:     "not_a_float",
			wantError: true,
		},
		{
			name:      "invalid boolean",
			field:     "active",
			value:     "maybe",
			wantError: true,
		},
		{
			name:      "field not in schema",
			field:     "unknown",
			value:     "value",
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := parser.ParseValue(tt.field, tt.value)
			if (err != nil) != tt.wantError {
				t.Errorf("ParseValue() error = %v, wantError %v", err, tt.wantError)
				return
			}

			if !tt.wantError {
				// For time comparisons, we need special handling
				if expectedTime, ok := tt.expected.(time.Time); ok {
					if resultTime, ok := result.(time.Time); ok {
						if !expectedTime.Equal(resultTime) {
							t.Errorf("ParseValue() = %v, want %v", result, tt.expected)
						}
					} else {
						t.Errorf("ParseValue() result is not time.Time: %T", result)
					}
				} else {
					// For arrays, we need deep comparison
					if !deepEqual(result, tt.expected) {
						t.Errorf("ParseValue() = %v, want %v", result, tt.expected)
					}
				}
			}
		})
	}
}

func TestFilterParser_NoSchema(t *testing.T) {
	parser := NewFilterParser(nil)

	tests := []struct {
		name     string
		value    string
		expected interface{}
	}{
		{
			name:     "infer boolean",
			value:    "true",
			expected: true,
		},
		{
			name:     "infer integer",
			value:    "42",
			expected: int64(42),
		},
		{
			name:     "infer float",
			value:    "3.14",
			expected: 3.14,
		},
		{
			name:     "infer string",
			value:    "hello",
			expected: "hello",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := parser.ParseValue("field", tt.value)
			if err != nil {
				t.Errorf("ParseValue() error = %v", err)
				return
			}

			if !deepEqual(result, tt.expected) {
				t.Errorf("ParseValue() = %v (%T), want %v (%T)", result, result, tt.expected, tt.expected)
			}
		})
	}
}

func TestFilterParser_ValidateField(t *testing.T) {
	schema := map[string]FieldType{
		"existing": StringField,
	}

	parser := NewFilterParser(schema)

	tests := []struct {
		name      string
		field     string
		wantError bool
	}{
		{
			name:      "existing field",
			field:     "existing",
			wantError: false,
		},
		{
			name:      "non-existing field",
			field:     "nonexistent",
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := parser.ValidateField(tt.field)
			if (err != nil) != tt.wantError {
				t.Errorf("ValidateField() error = %v, wantError %v", err, tt.wantError)
			}
		})
	}
}

func TestFilterParser_ValidateFieldType(t *testing.T) {
	schema := map[string]FieldType{
		"name":  StringField,
		"age":   IntField,
		"price": FloatField,
	}

	parser := NewFilterParser(schema)

	tests := []struct {
		name          string
		field         string
		expectedTypes []FieldType
		wantError     bool
	}{
		{
			name:          "valid string field",
			field:         "name",
			expectedTypes: []FieldType{StringField},
			wantError:     false,
		},
		{
			name:          "valid numeric field for range",
			field:         "age",
			expectedTypes: []FieldType{IntField, FloatField},
			wantError:     false,
		},
		{
			name:          "invalid field type",
			field:         "name",
			expectedTypes: []FieldType{IntField},
			wantError:     true,
		},
		{
			name:          "non-existing field",
			field:         "nonexistent",
			expectedTypes: []FieldType{StringField},
			wantError:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := parser.ValidateFieldType(tt.field, tt.expectedTypes...)
			if (err != nil) != tt.wantError {
				t.Errorf("ValidateFieldType() error = %v, wantError %v", err, tt.wantError)
			}
		})
	}
}

func TestFilterParser_CreateFilters(t *testing.T) {
	schema := map[string]FieldType{
		"name":  StringField,
		"age":   IntField,
		"price": FloatField,
		"tags":  StringArrayField,
	}

	parser := NewFilterParser(schema)

	t.Run("create equality filter", func(t *testing.T) {
		filter, err := parser.CreateEqualityFilter("name", "test")
		if err != nil {
			t.Errorf("CreateEqualityFilter() error = %v", err)
			return
		}

		if filter.Field != "name" || filter.Value != "test" {
			t.Errorf("CreateEqualityFilter() = %+v, want field=name, value=test", filter)
		}
	})

	t.Run("create range filter", func(t *testing.T) {
		filter, err := parser.CreateRangeFilter("age", "18", "65")
		if err != nil {
			t.Errorf("CreateRangeFilter() error = %v", err)
			return
		}

		if filter.Field != "age" || filter.Min != int64(18) || filter.Max != int64(65) {
			t.Errorf("CreateRangeFilter() = %+v, want field=age, min=18, max=65", filter)
		}
	})

	t.Run("create containment filter", func(t *testing.T) {
		filter, err := parser.CreateContainmentFilter("tags", []string{"red", "blue"}, ContainsAny)
		if err != nil {
			t.Errorf("CreateContainmentFilter() error = %v", err)
			return
		}

		if filter.Field != "tags" || len(filter.Values) != 2 || filter.Mode != ContainsAny {
			t.Errorf("CreateContainmentFilter() = %+v, want field=tags, 2 values, ContainsAny mode", filter)
		}
	})
}

// Helper function for deep equality comparison
func deepEqual(a, b interface{}) bool {
	// Handle slice comparisons
	switch aVal := a.(type) {
	case []string:
		if bVal, ok := b.([]string); ok {
			if len(aVal) != len(bVal) {
				return false
			}
			for i, v := range aVal {
				if v != bVal[i] {
					return false
				}
			}
			return true
		}
	case []int64:
		if bVal, ok := b.([]int64); ok {
			if len(aVal) != len(bVal) {
				return false
			}
			for i, v := range aVal {
				if v != bVal[i] {
					return false
				}
			}
			return true
		}
	case []float64:
		if bVal, ok := b.([]float64); ok {
			if len(aVal) != len(bVal) {
				return false
			}
			for i, v := range aVal {
				if v != bVal[i] {
					return false
				}
			}
			return true
		}
	}

	return a == b
}
