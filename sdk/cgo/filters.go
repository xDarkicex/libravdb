package main

import (
	"encoding/json"
	"fmt"

	"github.com/xDarkicex/libravdb/internal/filter"
)

type ASTNode struct {
	Type    string        `json:"type"`
	Field   string        `json:"field,omitempty"`
	Value   interface{}   `json:"value,omitempty"`
	Values  []interface{} `json:"values,omitempty"`
	Filters []ASTNode     `json:"filters,omitempty"`
	Filter  *ASTNode      `json:"filter,omitempty"` // for NOT filter
}

func parseFilterJSON(jsonStr string) (filter.Filter, error) {
	if jsonStr == "" || jsonStr == "{}" {
		return nil, nil
	}

	var root ASTNode
	if err := json.Unmarshal([]byte(jsonStr), &root); err != nil {
		return nil, fmt.Errorf("failed to parse filter json: %v", err)
	}

	return buildFilter(root)
}

func buildFilter(node ASTNode) (filter.Filter, error) {
	switch node.Type {
	case "eq":
		return filter.NewEqualityFilter(node.Field, node.Value), nil
	case "gt":
		return filter.NewGreaterThanFilter(node.Field, node.Value), nil
	case "lt":
		return filter.NewLessThanFilter(node.Field, node.Value), nil
	case "between":
		// Assume Value is [min, max]
		arr, ok := node.Value.([]interface{})
		if !ok || len(arr) != 2 {
			return nil, fmt.Errorf("between filter requires an array of 2 values")
		}
		return filter.NewBetweenFilter(node.Field, arr[0], arr[1]), nil
	case "contains_any":
		return filter.NewContainsAnyFilter(node.Field, node.Values), nil
	case "contains_all":
		return filter.NewContainsAllFilter(node.Field, node.Values), nil
	case "exact_match":
		return filter.NewExactMatchFilter(node.Field, node.Values), nil
	case "and":
		var subFilters []filter.Filter
		for _, sub := range node.Filters {
			f, err := buildFilter(sub)
			if err != nil {
				return nil, err
			}
			subFilters = append(subFilters, f)
		}
		return filter.NewAndFilter(subFilters...), nil
	case "or":
		var subFilters []filter.Filter
		for _, sub := range node.Filters {
			f, err := buildFilter(sub)
			if err != nil {
				return nil, err
			}
			subFilters = append(subFilters, f)
		}
		return filter.NewOrFilter(subFilters...), nil
	case "not":
		if node.Filter == nil {
			return nil, fmt.Errorf("not filter missing child filter")
		}
		f, err := buildFilter(*node.Filter)
		if err != nil {
			return nil, err
		}
		return filter.NewNotFilter(f), nil
	default:
		return nil, fmt.Errorf("unknown filter type: %s", node.Type)
	}
}
