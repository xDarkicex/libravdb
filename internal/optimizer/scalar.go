package optimizer

import (
	"fmt"
	"math"
	"strconv"
	"time"

	apexjson "github.com/xDarkicex/apexJSON/v2"
	"github.com/xDarkicex/lexer"
)

// ScalarKind is the semantic type carried by a SQL predicate value. The
// optimizer must not use SQL source text as its runtime value representation.
type ScalarKind uint8

const (
	ScalarInvalid ScalarKind = iota
	ScalarNull
	ScalarString
	ScalarInt
	ScalarFloat
	ScalarBool
	ScalarVector
	ScalarBytes
	ScalarTimestamp
	// ScalarJSON carries a JSON document supplied as a bound parameter. It is
	// distinct from ScalarString so callers can pass decoded maps without
	// first serializing them into SQL text.
	ScalarJSON
)

// ScalarValue is the typed value used by optimizer predicates. Bytes is kept
// only for opaque values and legacy plan construction; SQL NULL is represented
// by Kind == ScalarNull and never by a byte slice containing "NULL".
type ScalarValue struct {
	Kind      ScalarKind
	BytesData []byte
	Int       int64
	Float     float64
	Bool      bool
	Vector    []float32
	Time      time.Time
}

// NamedScalar is a named parameter entry. Name is stored without the '@'
// marker and is compared ASCII-case-insensitively without allocating.
type NamedScalar struct {
	Name  []byte
	Value ScalarValue
}

// ParameterSet is the execution-time parameter representation. Positional
// entries are indexed by PostgreSQL ordinal ($1 is Positional[0]); named
// entries are looked up by @name. The source query is never rewritten.
type ParameterSet struct {
	Positional []ScalarValue
	Named      []NamedScalar
}

// NewParameterSet converts the compatibility map used by the public Go API
// into the native typed representation. This conversion is deliberately at
// the API boundary, never inside the optimizer hot path.
func NewParameterSet(params map[string]interface{}) *ParameterSet {
	if len(params) == 0 {
		return nil
	}
	p := &ParameterSet{}
	for key, value := range params {
		if len(key) > 0 && key[0] == '$' {
			if ordinal, ok := parseOrdinalBytes([]byte(key[1:])); ok && ordinal > 0 {
				if len(p.Positional) < ordinal {
					p.Positional = append(p.Positional, make([]ScalarValue, ordinal-len(p.Positional))...)
				}
				p.Positional[ordinal-1] = ScalarFromInterface(value)
				continue
			}
		}
		if len(key) > 0 && (key[0] == '@' || key[0] == '$') {
			key = key[1:]
		}
		if ordinal, ok := parseOrdinalBytes([]byte(key)); ok && ordinal > 0 {
			if len(p.Positional) < ordinal {
				p.Positional = append(p.Positional, make([]ScalarValue, ordinal-len(p.Positional))...)
			}
			p.Positional[ordinal-1] = ScalarFromInterface(value)
			continue
		}
		p.Named = append(p.Named, NamedScalar{Name: append([]byte(nil), key...), Value: ScalarFromInterface(value)})
	}
	return p
}

// Lookup resolves a parser identifier by its source offsets. It recognizes
// only the parameter marker and ordinal/name bytes; no strings or reflection
// are used on the execution path.
func (p *ParameterSet) Lookup(src []byte, start, end uint32) (ScalarValue, bool) {
	if p == nil || start >= uint32(len(src)) || end > uint32(len(src)) || start >= end {
		return ScalarValue{}, false
	}
	marker := src[start]
	body := src[start+1 : end]
	if marker == '$' {
		ordinal, ok := parseOrdinalBytes(body)
		if ok && ordinal > 0 && ordinal <= len(p.Positional) {
			value := p.Positional[ordinal-1]
			if value.Kind != ScalarInvalid {
				return value, true
			}
		}
	}
	if marker != '@' && marker != '$' {
		return ScalarValue{}, false
	}
	for i := range p.Named {
		if asciiEqualFold(p.Named[i].Name, body) {
			return p.Named[i].Value, true
		}
	}
	return ScalarValue{}, false
}

func NullValue() ScalarValue { return ScalarValue{Kind: ScalarNull} }

func StringValue(v string) ScalarValue {
	return ScalarValue{Kind: ScalarString, BytesData: []byte(v)}
}

func BytesValue(v []byte) ScalarValue {
	return ScalarValue{Kind: ScalarString, BytesData: append([]byte(nil), v...)}
}

func IntValue(v int64) ScalarValue { return ScalarValue{Kind: ScalarInt, Int: v} }

func FloatValue(v float64) ScalarValue { return ScalarValue{Kind: ScalarFloat, Float: v} }

func BoolValue(v bool) ScalarValue { return ScalarValue{Kind: ScalarBool, Bool: v} }

func VectorValue(v []float32) ScalarValue {
	return ScalarValue{Kind: ScalarVector, Vector: append([]float32(nil), v...)}
}

func JSONValue(v []byte) ScalarValue {
	return ScalarValue{Kind: ScalarJSON, BytesData: append([]byte(nil), v...)}
}

// ScalarFromInterface converts a native query parameter or metadata value to
// a typed scalar without going through SQL text.
func ScalarFromInterface(v interface{}) ScalarValue {
	if v == nil {
		return NullValue()
	}
	switch x := v.(type) {
	case ScalarValue:
		return x
	case string:
		return StringValue(x)
	case []byte:
		return ScalarValue{Kind: ScalarBytes, BytesData: append([]byte(nil), x...)}
	case int:
		return IntValue(int64(x))
	case int8:
		return IntValue(int64(x))
	case int16:
		return IntValue(int64(x))
	case int32:
		return IntValue(int64(x))
	case int64:
		return IntValue(x)
	case uint:
		return IntValue(int64(x))
	case uint8:
		return IntValue(int64(x))
	case uint16:
		return IntValue(int64(x))
	case uint32:
		return IntValue(int64(x))
	case uint64:
		if x <= math.MaxInt64 {
			return IntValue(int64(x))
		}
	case float32:
		return FloatValue(float64(x))
	case float64:
		return FloatValue(x)
	case bool:
		return BoolValue(x)
	case []float32:
		return VectorValue(x)
	case time.Time:
		return ScalarValue{Kind: ScalarTimestamp, Time: x}
	case apexjson.RawMessage:
		return JSONValue(x)
	case interface {
		String() string
		Float64() (float64, error)
		Int64() (int64, error)
	}:
		return JSONValue([]byte(x.String()))
	case map[string]interface{}, map[string]string, []interface{}, []string, []bool,
		[]int, []int8, []int16, []int32, []int64, []uint, []uint16,
		[]uint32, []uint64, []float64:
		// SDKs deserialize nested JSON objects into map[string]interface{} and
		// arrays into []interface{}. Keep the document opaque at the optimizer
		// boundary; JSON/JSONB casts and the collection validator decode it into
		// LibraVDB's canonical JSON tree later.
		if encoded, err := apexjson.Marshal(x); err == nil {
			return JSONValue(encoded)
		}
	}
	return ScalarValue{}
}

// ScalarFromLiteralBytes converts an already-tokenized SQL literal into a
// typed value. The lexer/parser has already removed quoting; this function does
// not parse SQL syntax or mutate source text.
func ScalarFromLiteralBytes(raw []byte) ScalarValue {
	if asciiEqualFold(raw, []byte("NULL")) {
		return NullValue()
	}
	if asciiEqualFold(raw, []byte("TRUE")) {
		return BoolValue(true)
	}
	if asciiEqualFold(raw, []byte("FALSE")) {
		return BoolValue(false)
	}
	if hasFloatMarker(raw) {
		if v, err := strconv.ParseFloat(string(raw), 64); err == nil {
			return FloatValue(v)
		}
	}
	if v, ok := parseSignedIntBytes(raw); ok {
		return IntValue(v)
	}
	return BytesValue(raw)
}

func (v ScalarValue) IsNull() bool { return v.Kind == ScalarNull }

// Bytes returns the canonical textual form needed only at storage/index
// boundaries that still use byte keys. It is not used for semantic comparison.
func (v ScalarValue) Bytes() []byte {
	switch v.Kind {
	case ScalarNull, ScalarInvalid:
		return nil
	case ScalarString, ScalarJSON:
		return append([]byte(nil), v.BytesData...)
	case ScalarInt:
		return []byte(strconv.FormatInt(v.Int, 10))
	case ScalarFloat:
		return []byte(strconv.FormatFloat(v.Float, 'f', -1, 64))
	case ScalarBool:
		return []byte(strconv.FormatBool(v.Bool))
	case ScalarBytes:
		return append([]byte(nil), v.BytesData...)
	case ScalarTimestamp:
		return []byte(v.Time.UTC().Format(time.RFC3339Nano))
	case ScalarVector:
		out := make([]byte, 0, len(v.Vector)*8+2)
		out = append(out, '[')
		for i, value := range v.Vector {
			if i > 0 {
				out = append(out, ',')
			}
			out = strconv.AppendFloat(out, float64(value), 'f', -1, 32)
		}
		return append(out, ']')
	default:
		return nil
	}
}

// CompareScalar compares an actual record value with a typed predicate value.
// The bool result reports SQL NULL on the actual side. NULL comparisons are
// UNKNOWN and must not match ordinary predicates.
func CompareScalar(actual interface{}, expected ScalarValue) (cmp int, actualNull bool, err error) {
	if actual == nil {
		return 0, true, nil
	}
	if expected.Kind == ScalarNull || expected.Kind == ScalarInvalid {
		return 0, false, nil
	}
	if expected.Kind == ScalarString || expected.Kind == ScalarBytes || expected.Kind == ScalarJSON {
		right := expected.BytesData
		return compareValueBytes(actual, right), false, nil
	}
	if expected.Kind == ScalarInt {
		left, err := numericAsFloat(actual)
		if err != nil {
			return 0, false, err
		}
		if left < float64(expected.Int) {
			return -1, false, nil
		}
		if left > float64(expected.Int) {
			return 1, false, nil
		}
		return 0, false, nil
	}
	if expected.Kind == ScalarFloat {
		left, err := numericAsFloat(actual)
		if err != nil {
			return 0, false, err
		}
		if left < expected.Float {
			return -1, false, nil
		}
		if left > expected.Float {
			return 1, false, nil
		}
		return 0, false, nil
	}
	if expected.Kind == ScalarBool {
		left, ok := actual.(bool)
		if !ok {
			parsed, parseOK := boolValue(actual)
			if !parseOK {
				return 0, false, fmt.Errorf("%T is not boolean", actual)
			}
			left = parsed
		}
		if left == expected.Bool {
			return 0, false, nil
		}
		if !left && expected.Bool {
			return -1, false, nil
		}
		return 1, false, nil
	}
	if expected.Kind == ScalarTimestamp {
		if left, ok := actual.(time.Time); ok {
			if left.Before(expected.Time) {
				return -1, false, nil
			}
			if left.After(expected.Time) {
				return 1, false, nil
			}
			return 0, false, nil
		}
		return compareValueBytes(actual, []byte(expected.Time.UTC().Format(time.RFC3339Nano))), false, nil
	}
	return 0, false, fmt.Errorf("unsupported scalar comparison kind %d", expected.Kind)
}

func asciiEqualFold(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		x, y := a[i], b[i]
		if x >= 'A' && x <= 'Z' {
			x += 'a' - 'A'
		}
		if y >= 'A' && y <= 'Z' {
			y += 'a' - 'A'
		}
		if x != y {
			return false
		}
	}
	return true
}

func compareBytes(a, b []byte) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] < b[i] {
			return -1
		}
		if a[i] > b[i] {
			return 1
		}
	}
	if len(a) < len(b) {
		return -1
	}
	if len(a) > len(b) {
		return 1
	}
	return 0
}

func compareValueBytes(actual interface{}, expected []byte) int {
	switch x := actual.(type) {
	case string:
		return compareStringBytes(x, expected)
	case []byte:
		return compareBytes(x, expected)
	case int:
		return compareBytes(strconv.AppendInt(nil, int64(x), 10), expected)
	case int64:
		return compareBytes(strconv.AppendInt(nil, x, 10), expected)
	case uint64:
		return compareBytes(strconv.AppendUint(nil, x, 10), expected)
	case float64:
		return compareBytes(strconv.AppendFloat(nil, x, 'f', -1, 64), expected)
	case float32:
		return compareBytes(strconv.AppendFloat(nil, float64(x), 'f', -1, 32), expected)
	case bool:
		if x {
			return compareBytes([]byte("true"), expected)
		}
		return compareBytes([]byte("false"), expected)
	default:
		return 0
	}
}

func compareStringBytes(a string, b []byte) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] < b[i] {
			return -1
		}
		if a[i] > b[i] {
			return 1
		}
	}
	if len(a) < len(b) {
		return -1
	}
	if len(a) > len(b) {
		return 1
	}
	return 0
}

func boolValue(v interface{}) (bool, bool) {
	switch x := v.(type) {
	case bool:
		return x, true
	case string:
		if asciiEqualFold([]byte(x), []byte("true")) {
			return true, true
		}
		if asciiEqualFold([]byte(x), []byte("false")) {
			return false, true
		}
	case []byte:
		if asciiEqualFold(x, []byte("true")) {
			return true, true
		}
		if asciiEqualFold(x, []byte("false")) {
			return false, true
		}
	}
	return false, false
}

func numericAsFloat(v interface{}) (float64, error) {
	switch x := v.(type) {
	case int:
		return float64(x), nil
	case int8:
		return float64(x), nil
	case int16:
		return float64(x), nil
	case int32:
		return float64(x), nil
	case int64:
		return float64(x), nil
	case uint:
		return float64(x), nil
	case uint8:
		return float64(x), nil
	case uint16:
		return float64(x), nil
	case uint32:
		return float64(x), nil
	case uint64:
		return float64(x), nil
	case float32:
		return float64(x), nil
	case float64:
		return x, nil
	case string:
		return strconv.ParseFloat(x, 64)
	case []byte:
		f, err := strconv.ParseFloat(string(x), 64)
		return f, err
	default:
		return 0, fmt.Errorf("%T is not numeric", v)
	}
}

func hasFloatMarker(raw []byte) bool {
	for _, c := range raw {
		if c == '.' || c == 'e' || c == 'E' {
			return true
		}
	}
	return false
}

func parseOrdinalBytes(raw []byte) (int, bool) {
	if len(raw) == 0 {
		return 0, false
	}
	n := 0
	for _, c := range raw {
		if c < '0' || c > '9' {
			return 0, false
		}
		n = n*10 + int(c-'0')
		if n < 0 {
			return 0, false
		}
	}
	return n, true
}

func parseSignedIntBytes(raw []byte) (int64, bool) {
	if len(raw) == 0 {
		return 0, false
	}
	neg := false
	start := 0
	if raw[0] == '-' || raw[0] == '+' {
		neg = raw[0] == '-'
		start = 1
	}
	if start == len(raw) {
		return 0, false
	}
	var n uint64
	for _, c := range raw[start:] {
		if c < '0' || c > '9' {
			return 0, false
		}
		n = n*10 + uint64(c-'0')
	}
	if neg {
		if n > uint64(math.MaxInt64)+1 {
			return 0, false
		}
		if n == uint64(math.MaxInt64)+1 {
			return math.MinInt64, true
		}
		return -int64(n), true
	}
	if n > uint64(math.MaxInt64) {
		return 0, false
	}
	return int64(n), true
}

// MatchesOperator applies the SQL comparison operators used by the optimizer.
func MatchesOperator(cmp int, op uint8) bool {
	switch op {
	case uint8(lexer.KindEquals):
		return cmp == 0
	case uint8(lexer.KindGreaterThan):
		return cmp > 0
	case uint8(lexer.KindLessThan):
		return cmp < 0
	case uint8(lexer.KindGreaterEqual):
		return cmp >= 0
	case uint8(lexer.KindLessEqual):
		return cmp <= 0
	case uint8(lexer.KindNotEqual):
		return cmp != 0
	default:
		return false
	}
}
