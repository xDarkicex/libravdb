package filter

import (
	"context"
	"fmt"
	"strings"
)

// LogicalFilter implements logical operations (AND, OR, NOT) on other filters
type LogicalFilter struct {
	Operator LogicalOperator
	Filters  []Filter
}

// NewAndFilter creates a filter that requires all child filters to match
func NewAndFilter(filters ...Filter) *LogicalFilter {
	return &LogicalFilter{
		Operator: AndOperator,
		Filters:  filters,
	}
}

// NewOrFilter creates a filter that requires any child filter to match
func NewOrFilter(filters ...Filter) *LogicalFilter {
	return &LogicalFilter{
		Operator: OrOperator,
		Filters:  filters,
	}
}

// NewNotFilter creates a filter that negates the result of the child filter
func NewNotFilter(filter Filter) *LogicalFilter {
	return &LogicalFilter{
		Operator: NotOperator,
		Filters:  []Filter{filter},
	}
}

// Apply applies the logical operation to the child filters
func (f *LogicalFilter) Apply(ctx context.Context, entries []*VectorEntry) ([]*VectorEntry, error) {
	if err := f.Validate(); err != nil {
		return nil, err
	}

	switch f.Operator {
	case AndOperator:
		return f.applyAnd(ctx, entries)
	case OrOperator:
		return f.applyOr(ctx, entries)
	case NotOperator:
		return f.applyNot(ctx, entries)
	default:
		return nil, NewFilterError("logical", "", fmt.Sprintf("unsupported logical operator: %v", f.Operator))
	}
}

// Validate checks if the filter configuration is valid
func (f *LogicalFilter) Validate() error {
	if len(f.Filters) == 0 {
		return NewFilterError("logical", "", "logical filter must have at least one child filter")
	}

	if f.Operator == NotOperator && len(f.Filters) != 1 {
		return NewFilterError("logical", "", "NOT filter must have exactly one child filter")
	}

	// Validate all child filters
	for i, childFilter := range f.Filters {
		if err := childFilter.Validate(); err != nil {
			return NewFilterError("logical", "", fmt.Sprintf("child filter %d validation failed: %v", i, err))
		}
	}

	return nil
}

// EstimateSelectivity estimates selectivity based on child filter selectivities
func (f *LogicalFilter) EstimateSelectivity() float64 {
	if len(f.Filters) == 0 {
		return 1.0
	}

	switch f.Operator {
	case AndOperator:
		// AND: multiply selectivities (more restrictive)
		selectivity := 1.0
		for _, filter := range f.Filters {
			selectivity *= filter.EstimateSelectivity()
		}
		return selectivity
	case OrOperator:
		// OR: use complement multiplication (less restrictive)
		complement := 1.0
		for _, filter := range f.Filters {
			complement *= (1.0 - filter.EstimateSelectivity())
		}
		return 1.0 - complement
	case NotOperator:
		// NOT: complement of child selectivity
		return 1.0 - f.Filters[0].EstimateSelectivity()
	default:
		return 0.5
	}
}

// String returns a string representation of the filter
func (f *LogicalFilter) String() string {
	if len(f.Filters) == 0 {
		return "EMPTY"
	}

	switch f.Operator {
	case AndOperator:
		var parts []string
		for _, filter := range f.Filters {
			parts = append(parts, fmt.Sprintf("(%s)", filter.String()))
		}
		return strings.Join(parts, " AND ")
	case OrOperator:
		var parts []string
		for _, filter := range f.Filters {
			parts = append(parts, fmt.Sprintf("(%s)", filter.String()))
		}
		return strings.Join(parts, " OR ")
	case NotOperator:
		return fmt.Sprintf("NOT (%s)", f.Filters[0].String())
	default:
		return "UNKNOWN"
	}
}

// applyAnd applies AND logic - all filters must match
func (f *LogicalFilter) applyAnd(ctx context.Context, entries []*VectorEntry) ([]*VectorEntry, error) {
	result := entries

	// Apply each filter sequentially, passing results to the next
	for _, filter := range f.Filters {
		var err error
		result, err = filter.Apply(ctx, result)
		if err != nil {
			return nil, err
		}

		// Short-circuit if no entries remain
		if len(result) == 0 {
			break
		}
	}

	return result, nil
}

// applyOr applies OR logic - any filter can match
func (f *LogicalFilter) applyOr(ctx context.Context, entries []*VectorEntry) ([]*VectorEntry, error) {
	var allResults []*VectorEntry
	seenIDs := make(map[string]bool)

	// Apply each filter to the original entries and collect unique results
	for _, filter := range f.Filters {
		results, err := filter.Apply(ctx, entries)
		if err != nil {
			return nil, err
		}

		// Add unique results
		for _, entry := range results {
			if !seenIDs[entry.ID] {
				allResults = append(allResults, entry)
				seenIDs[entry.ID] = true
			}
		}
	}

	return allResults, nil
}

// applyNot applies NOT logic - negate the child filter result
func (f *LogicalFilter) applyNot(ctx context.Context, entries []*VectorEntry) ([]*VectorEntry, error) {
	// Apply the child filter
	matchedResults, err := f.Filters[0].Apply(ctx, entries)
	if err != nil {
		return nil, err
	}

	// Create a set of matched IDs for quick lookup
	matchedIDs := make(map[string]bool)
	for _, entry := range matchedResults {
		matchedIDs[entry.ID] = true
	}

	// Return entries that were NOT matched
	var result []*VectorEntry
	for _, entry := range entries {
		if !matchedIDs[entry.ID] {
			result = append(result, entry)
		}
	}

	return result, nil
}
