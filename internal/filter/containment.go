package filter

import (
	"context"
	"fmt"
	"reflect"
)

// ContainmentFilter implements array containment filtering for multi-valued fields
type ContainmentFilter struct {
	Field  string
	Values []interface{}
	Mode   ContainmentMode
}

// ContainmentMode defines how containment matching works
type ContainmentMode int

const (
	// ContainsAny matches if the field contains any of the specified values
	ContainsAny ContainmentMode = iota
	// ContainsAll matches if the field contains all of the specified values
	ContainsAll
	// ExactMatch matches if the field exactly matches the specified values (same elements, any order)
	ExactMatch
)

// NewContainsAnyFilter creates a filter that matches if the field contains any of the values
func NewContainsAnyFilter(field string, values []interface{}) *ContainmentFilter {
	return &ContainmentFilter{
		Field:  field,
		Values: values,
		Mode:   ContainsAny,
	}
}

// NewContainsAllFilter creates a filter that matches if the field contains all of the values
func NewContainsAllFilter(field string, values []interface{}) *ContainmentFilter {
	return &ContainmentFilter{
		Field:  field,
		Values: values,
		Mode:   ContainsAll,
	}
}

// NewExactMatchFilter creates a filter that matches if the field exactly matches the values
func NewExactMatchFilter(field string, values []interface{}) *ContainmentFilter {
	return &ContainmentFilter{
		Field:  field,
		Values: values,
		Mode:   ExactMatch,
	}
}

// Apply filters entries based on array containment rules
func (f *ContainmentFilter) Apply(ctx context.Context, entries []*VectorEntry) ([]*VectorEntry, error) {
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

		if f.matchesContainment(fieldValue) {
			result = append(result, entry)
		}
	}

	return result, nil
}

// Validate checks if the filter configuration is valid
func (f *ContainmentFilter) Validate() error {
	if f.Field == "" {
		return NewFilterError("containment", f.Field, "field name cannot be empty")
	}

	if len(f.Values) == 0 {
		return NewFilterError("containment", f.Field, "values list cannot be empty")
	}

	return nil
}

// EstimateSelectivity returns selectivity estimate based on containment mode
func (f *ContainmentFilter) EstimateSelectivity() float64 {
	switch f.Mode {
	case ContainsAny:
		// More permissive, higher selectivity
		return 0.4
	case ContainsAll:
		// More restrictive, lower selectivity
		return 0.2
	case ExactMatch:
		// Most restrictive, lowest selectivity
		return 0.1
	default:
		return 0.3
	}
}

// String returns a string representation of the filter
func (f *ContainmentFilter) String() string {
	switch f.Mode {
	case ContainsAny:
		return fmt.Sprintf("%s CONTAINS ANY %v", f.Field, f.Values)
	case ContainsAll:
		return fmt.Sprintf("%s CONTAINS ALL %v", f.Field, f.Values)
	case ExactMatch:
		return fmt.Sprintf("%s EXACTLY %v", f.Field, f.Values)
	default:
		return fmt.Sprintf("%s CONTAINS %v", f.Field, f.Values)
	}
}

// matchesContainment checks if a field value matches the containment criteria
func (f *ContainmentFilter) matchesContainment(fieldValue interface{}) bool {
	// Convert field value to slice
	fieldSlice := f.toSlice(fieldValue)
	if fieldSlice == nil {
		// If field is not an array, treat it as a single-element array
		fieldSlice = []interface{}{fieldValue}
	}

	switch f.Mode {
	case ContainsAny:
		return f.containsAny(fieldSlice, f.Values)
	case ContainsAll:
		return f.containsAll(fieldSlice, f.Values)
	case ExactMatch:
		return f.exactMatch(fieldSlice, f.Values)
	default:
		return false
	}
}

// containsAny checks if fieldSlice contains any of the target values
func (f *ContainmentFilter) containsAny(fieldSlice, targetValues []interface{}) bool {
	for _, target := range targetValues {
		for _, field := range fieldSlice {
			if f.valuesEqual(field, target) {
				return true
			}
		}
	}
	return false
}

// containsAll checks if fieldSlice contains all of the target values
func (f *ContainmentFilter) containsAll(fieldSlice, targetValues []interface{}) bool {
	for _, target := range targetValues {
		found := false
		for _, field := range fieldSlice {
			if f.valuesEqual(field, target) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

// exactMatch checks if fieldSlice exactly matches targetValues (same elements, any order)
func (f *ContainmentFilter) exactMatch(fieldSlice, targetValues []interface{}) bool {
	if len(fieldSlice) != len(targetValues) {
		return false
	}

	// Check that every element in fieldSlice exists in targetValues
	for _, field := range fieldSlice {
		found := false
		for _, target := range targetValues {
			if f.valuesEqual(field, target) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	// Check that every element in targetValues exists in fieldSlice
	for _, target := range targetValues {
		found := false
		for _, field := range fieldSlice {
			if f.valuesEqual(field, target) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	return true
}

// toSlice converts various array types to []interface{}
func (f *ContainmentFilter) toSlice(v interface{}) []interface{} {
	if v == nil {
		return nil
	}

	rv := reflect.ValueOf(v)
	if rv.Kind() != reflect.Slice && rv.Kind() != reflect.Array {
		return nil
	}

	result := make([]interface{}, rv.Len())
	for i := 0; i < rv.Len(); i++ {
		result[i] = rv.Index(i).Interface()
	}
	return result
}

// valuesEqual compares two values for equality, handling type conversions
func (f *ContainmentFilter) valuesEqual(a, b interface{}) bool {
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
func (f *ContainmentFilter) numericEqual(a, b interface{}) bool {
	aVal, aOk := f.toFloat64(a)
	bVal, bOk := f.toFloat64(b)

	if aOk && bOk {
		return aVal == bVal
	}
	return false
}

// stringEqual handles string comparisons
func (f *ContainmentFilter) stringEqual(a, b interface{}) bool {
	aStr, aOk := a.(string)
	bStr, bOk := b.(string)

	if aOk && bOk {
		return aStr == bStr
	}
	return false
}

// toFloat64 converts various numeric types to float64
func (f *ContainmentFilter) toFloat64(v interface{}) (float64, bool) {
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
