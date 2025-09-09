package filter

import (
	"fmt"
	"strconv"
	"strings"
	"time"
)

// FilterParser provides utilities for parsing and validating filter expressions
type FilterParser struct {
	schema map[string]FieldType
}

// NewFilterParser creates a new filter parser with optional schema validation
func NewFilterParser(schema map[string]FieldType) *FilterParser {
	return &FilterParser{
		schema: schema,
	}
}

// ParseValue parses a string value into the appropriate type based on field schema
func (p *FilterParser) ParseValue(field string, value string) (interface{}, error) {
	if p.schema == nil {
		// No schema - try to infer type
		return p.inferType(value), nil
	}

	fieldType, exists := p.schema[field]
	if !exists {
		return nil, NewFilterError("parser", field, "field not found in schema")
	}

	return p.parseTypedValue(value, fieldType)
}

// ParseValues parses multiple string values for array fields
func (p *FilterParser) ParseValues(field string, values []string) ([]interface{}, error) {
	var result []interface{}

	for _, value := range values {
		parsed, err := p.ParseValue(field, value)
		if err != nil {
			return nil, err
		}
		result = append(result, parsed)
	}

	return result, nil
}

// ValidateField checks if a field exists in the schema
func (p *FilterParser) ValidateField(field string) error {
	if p.schema == nil {
		return nil // No schema validation
	}

	if _, exists := p.schema[field]; !exists {
		return NewFilterError("parser", field, "field not found in schema")
	}

	return nil
}

// ValidateFieldType checks if a field is of the expected type
func (p *FilterParser) ValidateFieldType(field string, expectedTypes ...FieldType) error {
	if p.schema == nil {
		return nil // No schema validation
	}

	fieldType, exists := p.schema[field]
	if !exists {
		return NewFilterError("parser", field, "field not found in schema")
	}

	for _, expected := range expectedTypes {
		if fieldType == expected {
			return nil
		}
	}

	return NewFilterError("parser", field, fmt.Sprintf("field type %v not compatible with expected types %v", fieldType, expectedTypes))
}

// GetFieldType returns the type of a field from the schema
func (p *FilterParser) GetFieldType(field string) (FieldType, bool) {
	if p.schema == nil {
		return StringField, false
	}

	fieldType, exists := p.schema[field]
	return fieldType, exists
}

// parseTypedValue parses a string value according to the specified field type
func (p *FilterParser) parseTypedValue(value string, fieldType FieldType) (interface{}, error) {
	switch fieldType {
	case StringField:
		return value, nil

	case IntField:
		intVal, err := strconv.ParseInt(value, 10, 64)
		if err != nil {
			return nil, NewFilterError("parser", "", fmt.Sprintf("invalid integer value: %s", value))
		}
		return intVal, nil

	case FloatField:
		floatVal, err := strconv.ParseFloat(value, 64)
		if err != nil {
			return nil, NewFilterError("parser", "", fmt.Sprintf("invalid float value: %s", value))
		}
		return floatVal, nil

	case BoolField:
		boolVal, err := strconv.ParseBool(value)
		if err != nil {
			return nil, NewFilterError("parser", "", fmt.Sprintf("invalid boolean value: %s", value))
		}
		return boolVal, nil

	case TimeField:
		return p.parseTimeValue(value)

	case StringArrayField:
		// For array fields, split by comma if not already split
		if strings.Contains(value, ",") {
			return strings.Split(value, ","), nil
		}
		return []string{value}, nil

	case IntArrayField:
		parts := strings.Split(value, ",")
		var result []int64
		for _, part := range parts {
			intVal, err := strconv.ParseInt(strings.TrimSpace(part), 10, 64)
			if err != nil {
				return nil, NewFilterError("parser", "", fmt.Sprintf("invalid integer in array: %s", part))
			}
			result = append(result, intVal)
		}
		return result, nil

	case FloatArrayField:
		parts := strings.Split(value, ",")
		var result []float64
		for _, part := range parts {
			floatVal, err := strconv.ParseFloat(strings.TrimSpace(part), 64)
			if err != nil {
				return nil, NewFilterError("parser", "", fmt.Sprintf("invalid float in array: %s", part))
			}
			result = append(result, floatVal)
		}
		return result, nil

	default:
		return nil, NewFilterError("parser", "", fmt.Sprintf("unsupported field type: %v", fieldType))
	}
}

// parseTimeValue attempts to parse a time value using common formats
func (p *FilterParser) parseTimeValue(value string) (time.Time, error) {
	formats := []string{
		time.RFC3339,
		time.RFC3339Nano,
		"2006-01-02T15:04:05",
		"2006-01-02 15:04:05",
		"2006-01-02",
		"15:04:05",
	}

	for _, format := range formats {
		if t, err := time.Parse(format, value); err == nil {
			return t, nil
		}
	}

	// Try parsing as Unix timestamp
	if timestamp, err := strconv.ParseInt(value, 10, 64); err == nil {
		return time.Unix(timestamp, 0), nil
	}

	return time.Time{}, NewFilterError("parser", "", fmt.Sprintf("unable to parse time value: %s", value))
}

// inferType attempts to infer the type of a string value
func (p *FilterParser) inferType(value string) interface{} {
	// Try boolean
	if boolVal, err := strconv.ParseBool(value); err == nil {
		return boolVal
	}

	// Try integer
	if intVal, err := strconv.ParseInt(value, 10, 64); err == nil {
		return intVal
	}

	// Try float
	if floatVal, err := strconv.ParseFloat(value, 64); err == nil {
		return floatVal
	}

	// Try time
	if timeVal, err := p.parseTimeValue(value); err == nil {
		return timeVal
	}

	// Default to string
	return value
}

// CreateEqualityFilter creates an equality filter with proper type parsing
func (p *FilterParser) CreateEqualityFilter(field, value string) (*EqualityFilter, error) {
	if err := p.ValidateField(field); err != nil {
		return nil, err
	}

	parsedValue, err := p.ParseValue(field, value)
	if err != nil {
		return nil, err
	}

	return NewEqualityFilter(field, parsedValue), nil
}

// CreateRangeFilter creates a range filter with proper type parsing
func (p *FilterParser) CreateRangeFilter(field, minValue, maxValue string) (*RangeFilter, error) {
	if err := p.ValidateField(field); err != nil {
		return nil, err
	}

	// Validate field type is suitable for range operations
	if err := p.ValidateFieldType(field, IntField, FloatField, TimeField, StringField); err != nil {
		return nil, err
	}

	var min, max interface{}
	var err error

	if minValue != "" {
		min, err = p.ParseValue(field, minValue)
		if err != nil {
			return nil, err
		}
	}

	if maxValue != "" {
		max, err = p.ParseValue(field, maxValue)
		if err != nil {
			return nil, err
		}
	}

	return NewRangeFilter(field, min, max), nil
}

// CreateContainmentFilter creates a containment filter with proper type parsing
func (p *FilterParser) CreateContainmentFilter(field string, values []string, mode ContainmentMode) (*ContainmentFilter, error) {
	if err := p.ValidateField(field); err != nil {
		return nil, err
	}

	parsedValues, err := p.ParseValues(field, values)
	if err != nil {
		return nil, err
	}

	return &ContainmentFilter{
		Field:  field,
		Values: parsedValues,
		Mode:   mode,
	}, nil
}
