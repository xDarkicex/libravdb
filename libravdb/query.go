package libravdb

import (
	"context"
	"fmt"
)

// QueryBuilder provides a fluent interface for building vector queries
type QueryBuilder struct {
	ctx        context.Context
	collection *Collection
	vector     []float32
	filters    []Filter
	limit      int
	threshold  float32
	efSearch   int // Override collection default
}

// Filter represents a metadata filter condition
type Filter struct {
	Field    string
	Operator FilterOperator
	Value    interface{}
}

// FilterOperator defines filter comparison operations
type FilterOperator int

const (
	Equal FilterOperator = iota
	NotEqual
	GreaterThan
	LessThan
	In
	NotIn
)

// WithVector sets the query vector
func (qb *QueryBuilder) WithVector(vector []float32) *QueryBuilder {
	qb.vector = make([]float32, len(vector))
	copy(qb.vector, vector)
	return qb
}

// WithFilter adds a metadata filter
func (qb *QueryBuilder) WithFilter(field string, op FilterOperator, value interface{}) *QueryBuilder {
	qb.filters = append(qb.filters, Filter{
		Field:    field,
		Operator: op,
		Value:    value,
	})
	return qb
}

// Eq adds an equality filter (convenience method)
func (qb *QueryBuilder) Eq(field string, value interface{}) *QueryBuilder {
	return qb.WithFilter(field, Equal, value)
}

// In adds an "in" filter (convenience method)
func (qb *QueryBuilder) In(field string, values interface{}) *QueryBuilder {
	return qb.WithFilter(field, In, values)
}

// Limit sets the maximum number of results to return
func (qb *QueryBuilder) Limit(k int) *QueryBuilder {
	qb.limit = k
	return qb
}

// WithThreshold sets a minimum similarity threshold
func (qb *QueryBuilder) WithThreshold(threshold float32) *QueryBuilder {
	qb.threshold = threshold
	return qb
}

// WithEfSearch overrides the collection's default efSearch parameter
func (qb *QueryBuilder) WithEfSearch(efSearch int) *QueryBuilder {
	qb.efSearch = efSearch
	return qb
}

// Execute runs the query and returns results
func (qb *QueryBuilder) Execute() (*SearchResults, error) {
	if qb.vector == nil {
		return nil, fmt.Errorf("query vector is required")
	}

	if qb.limit <= 0 {
		return nil, fmt.Errorf("limit must be positive, got %d", qb.limit)
	}

	// For Phase 1, we only support vector search without filters
	if len(qb.filters) > 0 {
		return nil, fmt.Errorf("metadata filters not yet implemented")
	}

	// TODO: Apply efSearch override to index
	// TODO: Apply threshold filtering to results

	result, err := qb.collection.Search(qb.ctx, qb.vector, qb.limit)
	if err != nil {
		return nil, err
	}

	return &SearchResults{
		Results: result.Results,
		Took:    result.Took,
		Total:   len(result.Results),
	}, nil
}
