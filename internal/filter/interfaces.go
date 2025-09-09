package filter

import (
	"context"
	"fmt"
)

// Filter represents a metadata filter that can be applied to vector entries
type Filter interface {
	// Apply filters the given entries and returns matching ones
	Apply(ctx context.Context, entries []*VectorEntry) ([]*VectorEntry, error)

	// Validate checks if the filter is valid
	Validate() error

	// EstimateSelectivity returns an estimate of how selective this filter is (0.0 to 1.0)
	EstimateSelectivity() float64

	// String returns a string representation of the filter
	String() string
}

// VectorEntry represents a vector with metadata (matches libravdb.VectorEntry)
type VectorEntry struct {
	ID       string                 `json:"id"`
	Vector   []float32              `json:"vector"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// FilterType represents the type of filter
type FilterType int

const (
	EqualityFilterType FilterType = iota
	RangeFilterType
	ContainmentFilterType
	LogicalFilterType
)

// LogicalOperator represents logical operations for combining filters
type LogicalOperator int

const (
	AndOperator LogicalOperator = iota
	OrOperator
	NotOperator
)

// FieldType represents the type of a metadata field
type FieldType int

const (
	StringField FieldType = iota
	IntField
	FloatField
	BoolField
	TimeField
	StringArrayField
	IntArrayField
	FloatArrayField
)

// FilterError represents errors that occur during filter operations
type FilterError struct {
	Type    string
	Field   string
	Message string
}

func (e *FilterError) Error() string {
	if e.Field != "" {
		return fmt.Sprintf("filter error on field '%s': %s", e.Field, e.Message)
	}
	return fmt.Sprintf("filter error: %s", e.Message)
}

// NewFilterError creates a new filter error
func NewFilterError(filterType, field, message string) *FilterError {
	return &FilterError{
		Type:    filterType,
		Field:   field,
		Message: message,
	}
}
