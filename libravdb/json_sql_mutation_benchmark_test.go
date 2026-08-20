package libravdb

import (
	"strconv"
	"testing"
)

// These benchmarks keep the former tree mutation algorithm as a test-only
// baseline. They make the performance tradeoff of the native ApexJSON path
// visible instead of allowing an allocation reduction to hide a latency
// regression.
const benchmarkJSONDocument = `{"profile":{"name":"Alice","career":"engineer"},"items":[1,3,5],"meta":{"active":true}}`

var benchmarkJSONMutationSink interface{}

func benchmarkTreeSet(node interface{}, path []string, replacement interface{}, createMissing bool) (interface{}, bool) {
	if len(path) == 0 {
		return replacement, true
	}
	switch current := node.(type) {
	case map[string]interface{}:
		out := make(map[string]interface{}, len(current))
		for key, value := range current {
			out[key] = value
		}
		key := path[0]
		child, exists := current[key]
		if len(path) == 1 {
			if !exists && !createMissing {
				return node, false
			}
			out[key] = replacement
			return out, true
		}
		if !exists {
			if !createMissing {
				return node, false
			}
			child = make(map[string]interface{})
		}
		updated, changed := benchmarkTreeSet(child, path[1:], replacement, createMissing)
		if !changed {
			return node, false
		}
		out[key] = updated
		return out, true
	case []interface{}:
		index, err := strconv.Atoi(path[0])
		if err != nil || index < 0 {
			return node, false
		}
		out := append([]interface{}(nil), current...)
		if index >= len(out) {
			if !createMissing || index != len(out) {
				return node, false
			}
			if len(path) == 1 {
				return append(out, replacement), true
			}
			out = append(out, make(map[string]interface{}))
		}
		if len(path) == 1 {
			out[index] = replacement
			return out, true
		}
		updated, changed := benchmarkTreeSet(out[index], path[1:], replacement, createMissing)
		if !changed {
			return node, false
		}
		out[index] = updated
		return out, true
	default:
		return node, false
	}
}

func benchmarkTreeInsert(node interface{}, path []string, replacement interface{}, insertAfter bool) (interface{}, bool) {
	if len(path) == 0 {
		return node, false
	}
	switch current := node.(type) {
	case map[string]interface{}:
		key := path[0]
		if len(path) == 1 {
			if _, exists := current[key]; exists {
				return node, false
			}
			out := make(map[string]interface{}, len(current)+1)
			for k, value := range current {
				out[k] = value
			}
			out[key] = replacement
			return out, true
		}
		child, exists := current[key]
		if !exists {
			return node, false
		}
		updated, changed := benchmarkTreeInsert(child, path[1:], replacement, insertAfter)
		if !changed {
			return node, false
		}
		out := make(map[string]interface{}, len(current))
		for k, value := range current {
			out[k] = value
		}
		out[key] = updated
		return out, true
	case []interface{}:
		index, err := strconv.Atoi(path[0])
		if err != nil {
			return node, false
		}
		if len(path) == 1 {
			if index < 0 {
				index = len(current) + index
			}
			if index < 0 {
				index = 0
			}
			if index > len(current) {
				index = len(current)
			}
			if insertAfter && index < len(current) {
				index++
			}
			out := make([]interface{}, 0, len(current)+1)
			out = append(out, current[:index]...)
			out = append(out, replacement)
			out = append(out, current[index:]...)
			return out, true
		}
		if index < 0 {
			index = len(current) + index
		}
		if index < 0 || index >= len(current) {
			return node, false
		}
		updated, changed := benchmarkTreeInsert(current[index], path[1:], replacement, insertAfter)
		if !changed {
			return node, false
		}
		out := append([]interface{}(nil), current...)
		out[index] = updated
		return out, true
	default:
		return node, false
	}
}

func BenchmarkJSONBSetNative(b *testing.B) {
	for i := 0; i < b.N; i++ {
		value, err := jsonbSet(benchmarkJSONDocument, "{profile,career}", `[]`, true)
		if err != nil {
			b.Fatal(err)
		}
		benchmarkJSONMutationSink = value
	}
}

func BenchmarkJSONBSetTreeBaseline(b *testing.B) {
	for i := 0; i < b.N; i++ {
		root, ok := decodeJSONValue(benchmarkJSONDocument)
		if !ok {
			b.Fatal("decode document")
		}
		replacement, ok := decodeJSONValue(`[]`)
		if !ok {
			b.Fatal("decode replacement")
		}
		value, changed := benchmarkTreeSet(root, []string{"profile", "career"}, replacement, true)
		if !changed {
			b.Fatal("tree mutation did not change document")
		}
		benchmarkJSONMutationSink = value
	}
}

func BenchmarkJSONBInsertNative(b *testing.B) {
	for i := 0; i < b.N; i++ {
		value, err := jsonbInsert(benchmarkJSONDocument, "{items,1}", `2`, false)
		if err != nil {
			b.Fatal(err)
		}
		benchmarkJSONMutationSink = value
	}
}

func BenchmarkJSONBInsertTreeBaseline(b *testing.B) {
	for i := 0; i < b.N; i++ {
		root, ok := decodeJSONValue(benchmarkJSONDocument)
		if !ok {
			b.Fatal("decode document")
		}
		replacement, ok := decodeJSONValue(`2`)
		if !ok {
			b.Fatal("decode replacement")
		}
		value, changed := benchmarkTreeInsert(root, []string{"items", "1"}, replacement, false)
		if !changed {
			b.Fatal("tree mutation did not change document")
		}
		benchmarkJSONMutationSink = value
	}
}
