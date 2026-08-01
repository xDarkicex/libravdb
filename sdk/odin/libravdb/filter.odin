package libravdb

import "core:encoding/json"
import "core:fmt"

FilterValue :: union {
    string,
    f64,
    i64,
    bool,
}

Filter :: struct {
    type:    string `json:"type"`,
    field:   string `json:"field,omitempty"`,
    value:   FilterValue `json:"value,omitempty"`,
    values:  []FilterValue `json:"values,omitempty"`,
    filters: []Filter `json:"filters,omitempty"`,
}

// eq creates an equality filter.
eq_str :: proc(field: string, value: string) -> Filter {
    return Filter{type = "eq", field = field, value = value}
}

eq_int :: proc(field: string, value: i64) -> Filter {
    return Filter{type = "eq", field = field, value = value}
}

eq_float :: proc(field: string, value: f64) -> Filter {
    return Filter{type = "eq", field = field, value = value}
}

// contains_any creates an IN filter.
in_str :: proc(field: string, values: []string) -> Filter {
    vals := make([]FilterValue, len(values))
    for v, i in values { vals[i] = v }
    return Filter{type = "contains_any", field = field, values = vals}
}

// logical operators
and_filters :: proc(filters: []Filter) -> Filter {
    return Filter{type = "and", filters = filters}
}

or_filters :: proc(filters: []Filter) -> Filter {
    return Filter{type = "or", filters = filters}
}

// to_json stringifies the filter.
to_json :: proc(f: Filter) -> string {
    data, err := json.marshal(f)
    if err != nil {
        return "{}"
    }
    return string(data)
}
