package util

import (
	"testing"

	apexjson "github.com/xDarkicex/apexJSON/v2"
)

// TestJSONNullMarshalsAsNull checks that the JSON-null sentinel encodes as
// the JSON null literal, wherever it sits in a document.
//
// This is what lets a document be encoded as it stands. The alternative was
// copying the tree first to replace every sentinel with a nil, which cost an
// allocation per level of every document that carried a null.
func TestJSONNullMarshalsAsNull(t *testing.T) {
	cases := []struct {
		name string
		in   interface{}
		want string
	}{
		{"bare", JSONNull{}, "null"},
		{"in-object", map[string]interface{}{"a": JSONNull{}}, `{"a":null}`},
		{"in-array", []interface{}{JSONNull{}, 1}, `[null,1]`},
		{"nested", map[string]interface{}{"a": map[string]interface{}{"b": JSONNull{}}}, `{"a":{"b":null}}`},
		{"beside-sql-null", map[string]interface{}{"a": nil, "b": JSONNull{}}, `{"a":null,"b":null}`},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			got, err := apexjson.Marshal(testCase.in)
			if err != nil {
				t.Fatalf("Marshal: %v", err)
			}
			if string(got) != testCase.want {
				t.Errorf("got %s, want %s", got, testCase.want)
			}
		})
	}
}

// TestJSONNullKeepsItsBinaryTag checks that the binary encoding still
// distinguishes the sentinel from a nil.
//
// The sentinel exists to tell SQL NULL from the JSON literal null, and the
// binary form is what carries that distinction across a wire that is not
// JSON. The JSON encoding cannot: it has one null, and the sentinel is it.
func TestJSONNullKeepsItsBinaryTag(t *testing.T) {
	enc := &BinaryEncoder{}
	if err := enc.WriteValue(JSONNull{}); err != nil {
		t.Fatalf("WriteValue: %v", err)
	}
	if len(enc.Buf) == 0 {
		t.Fatal("nothing was written")
	}
	if enc.Buf[0] != valueTypeJSONNull {
		t.Errorf("tag = %#x, want %#x", enc.Buf[0], valueTypeJSONNull)
	}

	nilEnc := &BinaryEncoder{}
	if err := nilEnc.WriteValue(nil); err != nil {
		t.Fatalf("WriteValue: %v", err)
	}
	if nilEnc.Buf[0] == valueTypeJSONNull {
		t.Error("a nil and the sentinel encoded as the same thing")
	}
}
