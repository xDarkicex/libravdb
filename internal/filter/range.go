package filter

import (
	"context"
	"fmt"
	"time"
)

// RangeFilter implements range-based filtering for numeric and date fields
type RangeFilter struct {
	Field string
	Min   interface{} // nil means no lower bound
	Max   interface{} // nil means no upper bound
}

// NewRangeFilter creates a new range filter
func NewRangeFilter(field string, min, max interface{}) *RangeFilter {
	return &RangeFilter{
		Field: field,
		Min:   min,
		Max:   max,
	}
}

// NewGreaterThanFilter creates a filter for values greater than the specified value
func NewGreaterThanFilter(field string, value interface{}) *RangeFilter {
	return &RangeFilter{
		Field: field,
		Min:   value,
		Max:   nil,
	}
}

// NewLessThanFilter creates a filter for values less than the specified value
func NewLessThanFilter(field string, value interface{}) *RangeFilter {
	return &RangeFilter{
		Field: field,
		Min:   nil,
		Max:   value,
	}
}

// NewBetweenFilter creates a filter for values between min and max (inclusive)
func NewBetweenFilter(field string, min, max interface{}) *RangeFilter {
	return &RangeFilter{
		Field: field,
		Min:   min,
		Max:   max,
	}
}

// Apply filters entries that fall within the specified range
func (f *RangeFilter) Apply(ctx context.Context, entries []*VectorEntry) ([]*VectorEntry, error) {
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

		if f.valueInRange(fieldValue) {
			result = append(result, entry)
		}
	}

	return result, nil
}

// Validate checks if the filter configuration is valid
func (f *RangeFilter) Validate() error {
	if f.Field == "" {
		return NewFilterError("range", f.Field, "field name cannot be empty")
	}

	if f.Min == nil && f.Max == nil {
		return NewFilterError("range", f.Field, "at least one bound (min or max) must be specified")
	}

	// Validate that min and max are comparable types
	if f.Min != nil && f.Max != nil {
		if !f.areComparable(f.Min, f.Max) {
			return NewFilterError("range", f.Field, "min and max values must be of comparable types")
		}

		// Check that min <= max
		if f.compareValues(f.Min, f.Max) > 0 {
			return NewFilterError("range", f.Field, "min value must be less than or equal to max value")
		}
	}

	return nil
}

// EstimateSelectivity returns selectivity estimate based on range bounds
func (f *RangeFilter) EstimateSelectivity() float64 {
	if f.Min != nil && f.Max != nil {
		return 0.3 // Both bounds: moderate selectivity
	}
	return 0.5 // Single bound: lower selectivity
}

// String returns a string representation of the filter
func (f *RangeFilter) String() string {
	if f.Min != nil && f.Max != nil {
		return fmt.Sprintf("%s BETWEEN %v AND %v", f.Field, f.Min, f.Max)
	} else if f.Min != nil {
		return fmt.Sprintf("%s >= %v", f.Field, f.Min)
	} else {
		return fmt.Sprintf("%s <= %v", f.Field, f.Max)
	}
}

// valueInRange checks if a value falls within the filter's range
func (f *RangeFilter) valueInRange(value interface{}) bool {
	if f.Min != nil {
		if f.compareValues(value, f.Min) < 0 {
			return false
		}
	}

	if f.Max != nil {
		if f.compareValues(value, f.Max) > 0 {
			return false
		}
	}

	return true
}

// compareValues compares two values, returning -1, 0, or 1
func (f *RangeFilter) compareValues(a, b interface{}) int {
	// Handle numeric comparisons
	if aNum, aOk := f.toFloat64(a); aOk {
		if bNum, bOk := f.toFloat64(b); bOk {
			if aNum < bNum {
				return -1
			} else if aNum > bNum {
				return 1
			}
			return 0
		}
	}

	// Handle string comparisons
	if aStr, aOk := a.(string); aOk {
		if bStr, bOk := b.(string); bOk {
			if aStr < bStr {
				return -1
			} else if aStr > bStr {
				return 1
			}
			return 0
		}
	}

	// Handle time comparisons
	if aTime, aOk := f.toTime(a); aOk {
		if bTime, bOk := f.toTime(b); bOk {
			if aTime.Before(bTime) {
				return -1
			} else if aTime.After(bTime) {
				return 1
			}
			return 0
		}
	}

	// If types don't match or aren't comparable, consider them equal
	return 0
}

// areComparable checks if two values can be compared
func (f *RangeFilter) areComparable(a, b interface{}) bool {
	// Both numeric
	if _, aOk := f.toFloat64(a); aOk {
		if _, bOk := f.toFloat64(b); bOk {
			return true
		}
	}

	// Both strings
	if _, aOk := a.(string); aOk {
		if _, bOk := b.(string); bOk {
			return true
		}
	}

	// Both times
	if _, aOk := f.toTime(a); aOk {
		if _, bOk := f.toTime(b); bOk {
			return true
		}
	}

	return false
}

// toFloat64 converts various numeric types to float64
func (f *RangeFilter) toFloat64(v interface{}) (float64, bool) {
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

// toTime converts various time representations to time.Time
func (f *RangeFilter) toTime(v interface{}) (time.Time, bool) {
	switch val := v.(type) {
	case time.Time:
		return val, true
	case string:
		// Try parsing common time formats
		formats := []string{
			time.RFC3339,
			time.RFC3339Nano,
			"2006-01-02T15:04:05",
			"2006-01-02 15:04:05",
			"2006-01-02",
		}
		for _, format := range formats {
			if t, err := time.Parse(format, val); err == nil {
				return t, true
			}
		}
	case int64:
		// Unix timestamp
		return time.Unix(val, 0), true
	}
	return time.Time{}, false
}
