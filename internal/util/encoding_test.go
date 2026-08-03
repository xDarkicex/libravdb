package util

import (
	"bytes"
	"testing"
)

func TestWriteValue_RoundTrip(t *testing.T) {
	tests := []struct {
		name  string
		value interface{}
	}{
		{"nil", nil},
		{"bool_true", true},
		{"bool_false", false},
		{"string_empty", ""},
		{"string_ascii", "hello"},
		{"string_unicode", "héllo wörld"},
		{"int_zero", int(0)},
		{"int_pos", int(42)},
		{"int_neg", int(-1)},
		{"int64", int64(1 << 40)},
		{"uint64", uint64(1 << 40)},
		{"float32", float32(3.14)},
		{"float64", float64(2.718281828)},
		{"string_slice_empty", []string{}},
		{"string_slice", []string{"a", "b", "c"}},
		{"interface_slice", []interface{}{"a", int64(1), float64(2.0)}},
		{"map_empty", map[string]interface{}{}},
		{"map_small", map[string]interface{}{"key": "value", "count": int64(5)}},
		{"map_nested", map[string]interface{}{
			"outer": map[string]interface{}{"inner": "val"},
		}},
		// New: []byte
		{"bytes_empty", []byte{}},
		{"bytes_small", []byte("hello world")},
		{"bytes_binary", []byte{0x00, 0x01, 0x02, 0xFF, 0xFE}},
		{"bytes_large", make([]byte, 4096)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			enc := AcquireBinaryEncoder(4096)
			defer ReleaseBinaryEncoder(enc)

			if err := enc.WriteValue(tt.value); err != nil {
				t.Fatalf("WriteValue(%T): %v", tt.value, err)
			}

			dec := &BinaryDecoder{Data: enc.Buf}
			got, err := dec.ReadValue()
			if err != nil {
				t.Fatalf("ReadValue: %v", err)
			}

			if !valuesEqual(tt.value, got) {
				t.Errorf("round-trip mismatch: %T\n  wrote: %v\n   read: %v", tt.value, tt.value, got)
			}
		})
	}
}

func TestWriteMetadata_RoundTripBytes(t *testing.T) {
	// Specific test for []byte in metadata maps — the new path.
	meta := map[string]interface{}{
		"title":   "My Document",
		"content": []byte{0xDE, 0xAD, 0xBE, 0xEF},
		"empty":   []byte{},
	}

	enc := AcquireBinaryEncoder(4096)
	defer ReleaseBinaryEncoder(enc)

	if err := enc.WriteMetadata(meta); err != nil {
		t.Fatalf("WriteMetadata: %v", err)
	}

	dec := &BinaryDecoder{Data: enc.Buf}
	got, err := dec.ReadMetadata()
	if err != nil {
		t.Fatalf("ReadMetadata: %v", err)
	}

	if got["title"] != "My Document" {
		t.Errorf("title = %v, want %q", got["title"], "My Document")
	}
	if !bytes.Equal(got["content"].([]byte), []byte{0xDE, 0xAD, 0xBE, 0xEF}) {
		t.Errorf("content = %v, want DEADBEEF", got["content"])
	}
	if !bytes.Equal(got["empty"].([]byte), []byte{}) {
		t.Errorf("empty = %v, want empty slice", got["empty"])
	}
}

func TestEstimateValueSize_Bytes(t *testing.T) {
	tests := []struct {
		value []byte
		want  int
	}{
		{nil, 1 + 4 + 0},     // type byte + length prefix + 0 bytes
		{[]byte{}, 1 + 4 + 0}, // same as nil
		{[]byte("hello"), 1 + 4 + 5},
		{make([]byte, 1024), 1 + 4 + 1024},
	}

	for _, tt := range tests {
		var v interface{} = tt.value
		got := EstimateValueSize(v)
		if got != tt.want {
			t.Errorf("EstimateValueSize([]byte len=%d) = %d, want %d", len(tt.value), got, tt.want)
		}
	}
}

func valuesEqual(a, b interface{}) bool {
	if a == nil && b == nil {
		return true
	}
	switch va := a.(type) {
	case []byte:
		vb, ok := b.([]byte)
		return ok && bytes.Equal(va, vb)
	case []string:
		vb, ok := b.([]string)
		if !ok || len(va) != len(vb) {
			return false
		}
		for i := range va {
			if va[i] != vb[i] {
				return false
			}
		}
		return true
	case []interface{}:
		vb, ok := b.([]interface{})
		if !ok || len(va) != len(vb) {
			return false
		}
		for i := range va {
			if !valuesEqual(va[i], vb[i]) {
				return false
			}
		}
		return true
	case map[string]interface{}:
		vb, ok := b.(map[string]interface{})
		if !ok || len(va) != len(vb) {
			return false
		}
		for k, av := range va {
			if !valuesEqual(av, vb[k]) {
				return false
			}
		}
		return true
	default:
		return a == b || (isFloatEq(a, b))
	}
}

func isFloatEq(a, b interface{}) bool {
	af, aok := toFloat64(a)
	bf, bok := toFloat64(b)
	return aok && bok && af == bf
}

func toFloat64(v interface{}) (float64, bool) {
	switch t := v.(type) {
	case float32:
		return float64(t), true
	case float64:
		return t, true
	}
	return 0, false
}
