package filter

import (
	"context"
	"fmt"
	"reflect"
)

// EqualityFilter implements exact equality matching for metadata fields
type EqualityFilter struct {
	Field string
	Value interface{}
}

// NewEqualityFilter creates a new equality filter
func NewEqualityFilter(field string, value interface{}) *EqualityFilter {
	return &EqualityFilter{
		Field: field,
		Value: value,
	}
}

// Apply filters entries that have the exact field value
func (f *EqualityFilter) Apply(ctx context.Context, entries []*VectorEntry) ([]*VectorEntry, error) {
	if err := f.Validate(); err != nil {
		return nil, err
	}

	var result []*VectorEntry
	for _, entry := range entries {
		if entry.Metadata == nil {
			continue
		}

		fieldValue, exists := entry.Metadata[f.Field]
		if !exists {
			continue
		}

		if f.valuesEqual(fieldValue, f.Value) {
			result = append(result, entry)
		}
	}

	return result, nil
}

// Validate checks if the filter configuration is valid
func (f *EqualityFilter) Validate() error {
	if f.Field == "" {
		return NewFilterError("equality", f.Field, "field name cannot be empty")
	}
	if f.Value == nil {
		return NewFilterError("equality", f.Field, "value cannot be nil")
	}
	return nil
}

// EstimateSelectivity returns selectivity estimate (conservative 0.1 for equality)
func (f *EqualityFilter) EstimateSelectivity() float64 {
	return 0.1 // Conservative estimate: 10% of entries match
}

// String returns a string representation of the filter
func (f *EqualityFilter) String() string {
	return fmt.Sprintf("%s == %v", f.Field, f.Value)
}

// valuesEqual compares two values for equality, handling type conversions
func (f *EqualityFilter) valuesEqual(a, b interface{}) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}

	// Direct equality check first
	if reflect.DeepEqual(a, b) {
		return true
	}

	// Handle numeric type conversions
	return f.numericEqual(a, b) || f.stringEqual(a, b)
}

// numericEqual handles numeric type conversions for equality
func (f *EqualityFilter) numericEqual(a, b interface{}) bool {
	aVal, aOk := f.toFloat64(a)
	bVal, bOk := f.toFloat64(b)

	if aOk && bOk {
		return aVal == bVal
	}
	return false
}

// stringEqual handles string comparisons
func (f *EqualityFilter) stringEqual(a, b interface{}) bool {
	aStr, aOk := a.(string)
	bStr, bOk := b.(string)

	if aOk && bOk {
		return aStr == bStr
	}
	return false
}

// toFloat64 converts various numeric types to float64
func (f *EqualityFilter) toFloat64(v interface{}) (float64, bool) {
	switch val := v.(type) {
	case int:
		return float64(val), true
	case int8:
		return float64(val), true
	case int16:
		return float64(val), true
	case int32:
		return float64(val), true
	case int64:
		return float64(val), true
	case uint:
		return float64(val), true
	case uint8:
		return float64(val), true
	case uint16:
		return float64(val), true
	case uint32:
		return float64(val), true
	case uint64:
		return float64(val), true
	case float32:
		return float64(val), true
	case float64:
		return val, true
	default:
		return 0, false
	}
}
