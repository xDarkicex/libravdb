package optimizer

import (
	"encoding/json"
	"testing"

	"github.com/xDarkicex/lexer"
)

func TestParameterSetResolvesProtocolAliasesWithoutRewrite(t *testing.T) {
	params := NewParameterSet(map[string]interface{}{
		"$1":        int64(42),
		"@QueryVec": []float32{1, 2, 3},
	})
	src := []byte("$1 @queryvec @QUERYVEC")

	got, ok := params.Lookup(src, 0, 2)
	if !ok || got.Kind != ScalarInt || got.Int != 42 {
		t.Fatalf("$1 lookup = %#v, %v", got, ok)
	}
	got, ok = params.Lookup(src, 3, 12)
	if !ok || got.Kind != ScalarVector || len(got.Vector) != 3 {
		t.Fatalf("@queryvec lookup = %#v, %v", got, ok)
	}
	got, ok = params.Lookup(src, 13, uint32(len(src)))
	if !ok || got.Kind != ScalarVector || got.Vector[2] != 3 {
		t.Fatalf("@QUERYVEC lookup = %#v, %v", got, ok)
	}
}

func TestScalarNullIsNotTextNULL(t *testing.T) {
	value := ScalarFromInterface(nil)
	if !value.IsNull() {
		t.Fatalf("nil parameter kind = %v, want ScalarNull", value.Kind)
	}
	if got := value.Bytes(); got != nil {
		t.Fatalf("NULL bytes = %q, want nil", got)
	}
	if value := ScalarFromLiteralBytes([]byte("NULL")); !value.IsNull() {
		t.Fatalf("NULL literal kind = %v, want ScalarNull", value.Kind)
	}
}

func TestScalarFromInterfacePreservesJSONAndVectors(t *testing.T) {
	jsonValue := ScalarFromInterface(map[string]interface{}{
		"name":   "Ada",
		"nested": map[string]interface{}{"ok": true},
		"items":  []interface{}{1, "two"},
	})
	if jsonValue.Kind != ScalarJSON {
		t.Fatalf("JSON parameter kind = %v, want ScalarJSON", jsonValue.Kind)
	}
	var decoded map[string]interface{}
	if err := json.Unmarshal(jsonValue.Bytes(), &decoded); err != nil {
		t.Fatalf("JSON parameter bytes are invalid: %v", err)
	}
	if decoded["name"] != "Ada" {
		t.Fatalf("decoded JSON = %#v", decoded)
	}

	vector := ScalarFromInterface([]float32{1, 2, 3})
	if vector.Kind != ScalarVector || len(vector.Vector) != 3 {
		t.Fatalf("vector parameter changed kind=%v value=%#v", vector.Kind, vector)
	}
}

func TestMatchesOperatorSupportsAllScalarComparisons(t *testing.T) {
	cases := []struct {
		name string
		cmp  int
		op   lexer.Kind
		want bool
	}{
		{"eq", 0, lexer.KindEquals, true},
		{"neq", 1, lexer.KindNotEqual, true},
		{"neq equal", 0, lexer.KindNotEqual, false},
		{"gt", 1, lexer.KindGreaterThan, true},
		{"gte equal", 0, lexer.KindGreaterEqual, true},
		{"gte less", -1, lexer.KindGreaterEqual, false},
		{"lt", -1, lexer.KindLessThan, true},
		{"lte equal", 0, lexer.KindLessEqual, true},
		{"lte greater", 1, lexer.KindLessEqual, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := MatchesOperator(tc.cmp, uint8(tc.op)); got != tc.want {
				t.Fatalf("MatchesOperator(%d, %v) = %v, want %v", tc.cmp, tc.op, got, tc.want)
			}
		})
	}
}
