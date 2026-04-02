package libravdb

import (
	"context"
	"fmt"
	"time"

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

// Gte adds a greater than or equal filter (convenience method).
func (qb *QueryBuilder) Gte(field string, value interface{}) *QueryBuilder {
	return qb.WithFilter(filter.NewRangeFilter(field, value, nil))
}

// Lt adds a less than filter (convenience method)
func (qb *QueryBuilder) Lt(field string, value interface{}) *QueryBuilder {
	return qb.WithFilter(filter.NewLessThanFilter(field, value))
}

// Lte adds a less than or equal filter (convenience method).
func (qb *QueryBuilder) Lte(field string, value interface{}) *QueryBuilder {
	return qb.WithFilter(filter.NewRangeFilter(field, nil, value))
}

// Between adds a range filter (convenience method)
func (qb *QueryBuilder) Between(field string, min, max interface{}) *QueryBuilder {
	return qb.WithFilter(filter.NewBetweenFilter(field, min, max))
}

// Contains adds a containment filter for a single value (convenience method).
func (qb *QueryBuilder) Contains(field string, value interface{}) *QueryBuilder {
	return qb.WithFilter(filter.NewContainsAnyFilter(field, []interface{}{value}))
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
	return newFilterChain(qb, nil, filter.AndOperator, false)
}

// Or creates a new QueryBuilder with OR logic for combining filters
func (qb *QueryBuilder) Or() *FilterChain {
	return newFilterChain(qb, nil, filter.OrOperator, false)
}

// Not creates a new QueryBuilder chain with NOT logic for combining filters.
func (qb *QueryBuilder) Not() *FilterChain {
	return newFilterChain(qb, nil, filter.AndOperator, true)
}

// NotEq adds a not-equal filter (convenience method)
func (qb *QueryBuilder) NotEq(field string, value interface{}) *QueryBuilder {
	return qb.WithFilter(filter.NewNotFilter(filter.NewEqualityFilter(field, value)))
}

// FilterChain provides a fluent interface for chaining filters with logical operators
type FilterChain struct {
	queryBuilder *QueryBuilder
	parentChain  *FilterChain
	operator     filter.LogicalOperator
	filters      []filter.Filter
	negated      bool
	applied      bool
}

func newFilterChain(qb *QueryBuilder, parent *FilterChain, operator filter.LogicalOperator, negated bool) *FilterChain {
	return &FilterChain{
		queryBuilder: qb,
		parentChain:  parent,
		operator:     operator,
		filters:      []filter.Filter{},
		negated:      negated,
	}
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

// Gte adds a greater than or equal filter to the chain.
func (fc *FilterChain) Gte(field string, value interface{}) *FilterChain {
	fc.filters = append(fc.filters, filter.NewRangeFilter(field, value, nil))
	return fc
}

// Lt adds a less than filter to the chain
func (fc *FilterChain) Lt(field string, value interface{}) *FilterChain {
	fc.filters = append(fc.filters, filter.NewLessThanFilter(field, value))
	return fc
}

// Lte adds a less than or equal filter to the chain.
func (fc *FilterChain) Lte(field string, value interface{}) *FilterChain {
	fc.filters = append(fc.filters, filter.NewRangeFilter(field, nil, value))
	return fc
}

// Between adds a range filter to the chain
func (fc *FilterChain) Between(field string, min, max interface{}) *FilterChain {
	fc.filters = append(fc.filters, filter.NewBetweenFilter(field, min, max))
	return fc
}

// Contains adds a containment filter for a single value to the chain.
func (fc *FilterChain) Contains(field string, value interface{}) *FilterChain {
	fc.filters = append(fc.filters, filter.NewContainsAnyFilter(field, []interface{}{value}))
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

// And creates a nested AND filter chain.
func (fc *FilterChain) And() *FilterChain {
	return newFilterChain(fc.queryBuilder, fc, filter.AndOperator, false)
}

// Or creates a nested OR filter chain.
func (fc *FilterChain) Or() *FilterChain {
	return newFilterChain(fc.queryBuilder, fc, filter.OrOperator, false)
}

// Filter adds a custom filter to the chain
func (fc *FilterChain) Filter(f filter.Filter) *FilterChain {
	fc.filters = append(fc.filters, f)
	return fc
}

// Not creates a nested NOT filter chain.
func (fc *FilterChain) Not() *FilterChain {
	return newFilterChain(fc.queryBuilder, fc, filter.AndOperator, true)
}

// NotEq adds a not-equal filter to the chain
func (fc *FilterChain) NotEq(field string, value interface{}) *FilterChain {
	fc.filters = append(fc.filters, filter.NewNotFilter(filter.NewEqualityFilter(field, value)))
	return fc
}

// End completes the filter chain and returns the parent chain when nested.
// Root chains remain chainable through delegated QueryBuilder methods.
func (fc *FilterChain) End() *FilterChain {
	fc.finalize()
	if fc.parentChain != nil {
		return fc.parentChain
	}
	return fc
}

func (fc *FilterChain) finalize() *QueryBuilder {
	if fc.applied {
		return fc.queryBuilder
	}

	combinedFilter := fc.combinedFilter()
	fc.applied = true
	if combinedFilter == nil {
		return fc.queryBuilder
	}

	if fc.parentChain != nil {
		fc.parentChain.filters = append(fc.parentChain.filters, combinedFilter)
		return fc.parentChain.queryBuilder
	}

	return fc.queryBuilder.WithFilter(combinedFilter)
}

func (fc *FilterChain) combinedFilter() filter.Filter {
	if len(fc.filters) == 0 {
		return nil
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

	if fc.negated {
		return filter.NewNotFilter(combinedFilter)
	}

	return combinedFilter
}

// Limit finalizes the current chain and forwards to QueryBuilder.
func (fc *FilterChain) Limit(k int) *QueryBuilder {
	return fc.finalize().Limit(k)
}

// WithThreshold finalizes the current chain and forwards to QueryBuilder.
func (fc *FilterChain) WithThreshold(threshold float32) *QueryBuilder {
	return fc.finalize().WithThreshold(threshold)
}

// WithEfSearch finalizes the current chain and forwards to QueryBuilder.
func (fc *FilterChain) WithEfSearch(efSearch int) *QueryBuilder {
	return fc.finalize().WithEfSearch(efSearch)
}

// Execute finalizes the current chain and forwards to QueryBuilder.
func (fc *FilterChain) Execute() (*SearchResults, error) {
	return fc.finalize().Execute()
}

// List finalizes the current chain and forwards to QueryBuilder.
func (fc *FilterChain) List() ([]Record, error) {
	return fc.finalize().List()
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

// List executes a metadata-only or vector-backed query and returns stable record rows.
// When no vector is provided, it scans the collection and applies metadata filters and limit.
func (qb *QueryBuilder) List() ([]Record, error) {
	start := time.Now()

	var records []Record
	if qb.vector == nil {
		all, err := qb.collection.ListAll(qb.ctx)
		if err != nil {
			return nil, err
		}
		records = all
	} else {
		if qb.limit <= 0 {
			return nil, fmt.Errorf("limit must be positive, got %d", qb.limit)
		}

		results, err := qb.collection.Search(qb.ctx, qb.vector, qb.getSearchLimit())
		if err != nil {
			return nil, err
		}
		records = recordsFromSearchResults(results.Results)
	}

	filtered, err := qb.applyFiltersToRecords(records, qb.optimizeFilters())
	if err != nil {
		return nil, fmt.Errorf("failed to apply filters: %w", err)
	}

	if qb.limit > 0 && len(filtered) > qb.limit {
		filtered = filtered[:qb.limit]
	}

	_ = start
	return filtered, nil
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

func (qb *QueryBuilder) applyFiltersToRecords(records []Record, filters []filter.Filter) ([]Record, error) {
	if len(filters) == 0 {
		copied := make([]Record, 0, len(records))
		for _, record := range records {
			copied = append(copied, Record{
				ID:       record.ID,
				Vector:   cloneVector(record.Vector),
				Metadata: cloneMetadata(record.Metadata),
			})
		}
		return copied, nil
	}

	filteredEntries, err := qb.applyFilterEntries(filterEntriesFromRecords(records), filters)
	if err != nil {
		return nil, err
	}

	filteredRecords := make([]Record, 0, len(filteredEntries))
	for _, entry := range filteredEntries {
		filteredRecords = append(filteredRecords, Record{
			ID:       entry.ID,
			Vector:   cloneVector(entry.Vector),
			Metadata: cloneMetadata(entry.Metadata),
		})
	}

	return filteredRecords, nil
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

func (qb *QueryBuilder) applyFilterEntries(entries []*filter.VectorEntry, filters []filter.Filter) ([]*filter.VectorEntry, error) {
	filterEntries := entries
	for _, f := range filters {
		var err error
		filterEntries, err = f.Apply(qb.ctx, filterEntries)
		if err != nil {
			return nil, err
		}
		if len(filterEntries) == 0 {
			break
		}
	}
	return filterEntries, nil
}

func recordsFromSearchResults(results []*SearchResult) []Record {
	records := make([]Record, 0, len(results))
	for _, result := range results {
		records = append(records, Record{
			ID:       result.ID,
			Vector:   cloneVector(result.Vector),
			Metadata: cloneMetadata(result.Metadata),
		})
	}
	return records
}
