package libravdb

import (
	"context"
	"fmt"

	"github.com/xDarkicex/libravdb/internal/filter"
)

// QueryBuilder provides a fluent interface for building vector queries
type QueryBuilder struct {
	ctx        context.Context
	collection *Collection
	vector     []float32
	filters    []filter.Filter
	limit      int
	threshold  float32
	efSearch   int // Override collection default
}

// Filter represents a metadata filter condition (deprecated, use filter package)
type Filter struct {
	Field    string
	Operator FilterOperator
	Value    interface{}
}

// FilterOperator defines filter comparison operations (deprecated, use filter package)
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

// WithFilter adds a metadata filter using the new filter system
func (qb *QueryBuilder) WithFilter(f filter.Filter) *QueryBuilder {
	qb.filters = append(qb.filters, f)
	return qb
}

// Eq adds an equality filter (convenience method)
func (qb *QueryBuilder) Eq(field string, value interface{}) *QueryBuilder {
	return qb.WithFilter(filter.NewEqualityFilter(field, value))
}

// Gt adds a greater than filter (convenience method)
func (qb *QueryBuilder) Gt(field string, value interface{}) *QueryBuilder {
	return qb.WithFilter(filter.NewGreaterThanFilter(field, value))
}

// Lt adds a less than filter (convenience method)
func (qb *QueryBuilder) Lt(field string, value interface{}) *QueryBuilder {
	return qb.WithFilter(filter.NewLessThanFilter(field, value))
}

// Between adds a range filter (convenience method)
func (qb *QueryBuilder) Between(field string, min, max interface{}) *QueryBuilder {
	return qb.WithFilter(filter.NewBetweenFilter(field, min, max))
}

// ContainsAny adds a containment filter for any values (convenience method)
func (qb *QueryBuilder) ContainsAny(field string, values []interface{}) *QueryBuilder {
	return qb.WithFilter(filter.NewContainsAnyFilter(field, values))
}

// ContainsAll adds a containment filter for all values (convenience method)
func (qb *QueryBuilder) ContainsAll(field string, values []interface{}) *QueryBuilder {
	return qb.WithFilter(filter.NewContainsAllFilter(field, values))
}

// And creates a new QueryBuilder with AND logic for combining filters
func (qb *QueryBuilder) And() *FilterChain {
	return &FilterChain{
		queryBuilder: qb,
		operator:     filter.AndOperator,
		filters:      []filter.Filter{},
	}
}

// Or creates a new QueryBuilder with OR logic for combining filters
func (qb *QueryBuilder) Or() *FilterChain {
	return &FilterChain{
		queryBuilder: qb,
		operator:     filter.OrOperator,
		filters:      []filter.Filter{},
	}
}

// Not creates a NOT filter wrapper around the provided filter
func (qb *QueryBuilder) Not(f filter.Filter) *QueryBuilder {
	return qb.WithFilter(filter.NewNotFilter(f))
}

// NotEq adds a not-equal filter (convenience method)
func (qb *QueryBuilder) NotEq(field string, value interface{}) *QueryBuilder {
	return qb.Not(filter.NewEqualityFilter(field, value))
}

// FilterChain provides a fluent interface for chaining filters with logical operators
type FilterChain struct {
	queryBuilder *QueryBuilder
	operator     filter.LogicalOperator
	filters      []filter.Filter
}

// Eq adds an equality filter to the chain
func (fc *FilterChain) Eq(field string, value interface{}) *FilterChain {
	fc.filters = append(fc.filters, filter.NewEqualityFilter(field, value))
	return fc
}

// Gt adds a greater than filter to the chain
func (fc *FilterChain) Gt(field string, value interface{}) *FilterChain {
	fc.filters = append(fc.filters, filter.NewGreaterThanFilter(field, value))
	return fc
}

// Lt adds a less than filter to the chain
func (fc *FilterChain) Lt(field string, value interface{}) *FilterChain {
	fc.filters = append(fc.filters, filter.NewLessThanFilter(field, value))
	return fc
}

// Between adds a range filter to the chain
func (fc *FilterChain) Between(field string, min, max interface{}) *FilterChain {
	fc.filters = append(fc.filters, filter.NewBetweenFilter(field, min, max))
	return fc
}

// ContainsAny adds a containment filter for any values to the chain
func (fc *FilterChain) ContainsAny(field string, values []interface{}) *FilterChain {
	fc.filters = append(fc.filters, filter.NewContainsAnyFilter(field, values))
	return fc
}

// ContainsAll adds a containment filter for all values to the chain
func (fc *FilterChain) ContainsAll(field string, values []interface{}) *FilterChain {
	fc.filters = append(fc.filters, filter.NewContainsAllFilter(field, values))
	return fc
}

// Filter adds a custom filter to the chain
func (fc *FilterChain) Filter(f filter.Filter) *FilterChain {
	fc.filters = append(fc.filters, f)
	return fc
}

// Not adds a NOT filter wrapper around the provided filter to the chain
func (fc *FilterChain) Not(f filter.Filter) *FilterChain {
	fc.filters = append(fc.filters, filter.NewNotFilter(f))
	return fc
}

// NotEq adds a not-equal filter to the chain
func (fc *FilterChain) NotEq(field string, value interface{}) *FilterChain {
	return fc.Not(filter.NewEqualityFilter(field, value))
}

// End completes the filter chain and returns the QueryBuilder
func (fc *FilterChain) End() *QueryBuilder {
	if len(fc.filters) == 0 {
		return fc.queryBuilder
	}

	var combinedFilter filter.Filter
	if len(fc.filters) == 1 {
		combinedFilter = fc.filters[0]
	} else {
		switch fc.operator {
		case filter.AndOperator:
			combinedFilter = filter.NewAndFilter(fc.filters...)
		case filter.OrOperator:
			combinedFilter = filter.NewOrFilter(fc.filters...)
		default:
			combinedFilter = filter.NewAndFilter(fc.filters...)
		}
	}

	return fc.queryBuilder.WithFilter(combinedFilter)
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

	// Optimize filters before execution
	optimizedFilters := qb.optimizeFilters()

	// Get initial search results from vector index
	result, err := qb.collection.Search(qb.ctx, qb.vector, qb.getSearchLimit())
	if err != nil {
		return nil, err
	}

	// Apply metadata filters if present
	if len(optimizedFilters) > 0 {
		filteredResults, err := qb.applyFilters(result.Results, optimizedFilters)
		if err != nil {
			return nil, fmt.Errorf("failed to apply filters: %w", err)
		}
		result.Results = filteredResults
	}

	// Apply threshold filtering
	if qb.threshold > 0 {
		result.Results = qb.applyThreshold(result.Results)
	}

	// Limit final results
	if len(result.Results) > qb.limit {
		result.Results = result.Results[:qb.limit]
	}

	return &SearchResults{
		Results: result.Results,
		Took:    result.Took,
		Total:   len(result.Results),
	}, nil
}

// optimizeFilters optimizes the filter execution order based on selectivity
func (qb *QueryBuilder) optimizeFilters() []filter.Filter {
	if len(qb.filters) == 0 {
		return nil
	}

	// Create a copy of filters for optimization
	optimized := make([]filter.Filter, len(qb.filters))
	copy(optimized, qb.filters)

	// Sort filters by selectivity (most selective first)
	// This is a simple optimization - more sophisticated cost-based optimization could be added
	for i := 0; i < len(optimized)-1; i++ {
		for j := i + 1; j < len(optimized); j++ {
			if optimized[i].EstimateSelectivity() > optimized[j].EstimateSelectivity() {
				optimized[i], optimized[j] = optimized[j], optimized[i]
			}
		}
	}

	return optimized
}

// applyFilters applies metadata filters to search results
func (qb *QueryBuilder) applyFilters(results []*SearchResult, filters []filter.Filter) ([]*SearchResult, error) {
	// Convert libravdb.SearchResult to filter.VectorEntry
	filterEntries := make([]*filter.VectorEntry, len(results))
	for i, result := range results {
		filterEntries[i] = &filter.VectorEntry{
			ID:       result.ID,
			Vector:   result.Vector,
			Metadata: result.Metadata,
		}
	}

	// Apply each filter sequentially
	for _, f := range filters {
		var err error
		filterEntries, err = f.Apply(qb.ctx, filterEntries)
		if err != nil {
			return nil, err
		}

		// Short-circuit if no results remain
		if len(filterEntries) == 0 {
			break
		}
	}

	// Convert back to libravdb.SearchResult, preserving scores
	// Create a map for quick lookup of original scores
	scoreMap := make(map[string]float32)
	for _, result := range results {
		scoreMap[result.ID] = result.Score
	}

	filteredResults := make([]*SearchResult, len(filterEntries))
	for i, entry := range filterEntries {
		filteredResults[i] = &SearchResult{
			ID:       entry.ID,
			Score:    scoreMap[entry.ID], // Preserve original score
			Vector:   entry.Vector,
			Metadata: entry.Metadata,
		}
	}

	return filteredResults, nil
}

// applyThreshold filters results based on similarity threshold
func (qb *QueryBuilder) applyThreshold(results []*SearchResult) []*SearchResult {
	var filtered []*SearchResult
	for _, result := range results {
		if result.Score >= qb.threshold {
			filtered = append(filtered, result)
		}
	}
	return filtered
}

// getSearchLimit calculates the search limit to use for the initial vector search
// This accounts for potential filtering that might reduce the result set
func (qb *QueryBuilder) getSearchLimit() int {
	if len(qb.filters) == 0 {
		return qb.limit
	}

	// Estimate the selectivity of all filters combined
	combinedSelectivity := 1.0
	for _, f := range qb.filters {
		combinedSelectivity *= f.EstimateSelectivity()
	}

	// Increase search limit to account for filtering
	// Use a minimum multiplier to ensure we get enough candidates
	multiplier := 1.0 / combinedSelectivity
	if multiplier < 2.0 {
		multiplier = 2.0
	}
	if multiplier > 10.0 {
		multiplier = 10.0
	}

	return int(float64(qb.limit) * multiplier)
}
