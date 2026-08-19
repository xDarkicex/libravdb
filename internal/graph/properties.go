package graph

import (
	"bytes"
	"fmt"
	"math"
	"strconv"

	apexjson "github.com/xDarkicex/apexJSON/v2"
)

// EdgePropertyEncodingVersion identifies the on-edge property envelope. The
// envelope is intentionally small: [version byte][canonical JSON object]. A
// future version can change the value encoding without changing WAL/page
// plumbing.
const EdgePropertyEncodingVersion byte = 1

// EdgePropertyValueKind is the JSON scalar type used by edge predicates.
type EdgePropertyValueKind uint8

const (
	EdgePropertyNull EdgePropertyValueKind = iota
	EdgePropertyNumber
	EdgePropertyString
	EdgePropertyBool
)

// EdgePropertyValue is the typed comparison value carried in a planned edge
// predicate. Missing properties and JSON null both evaluate as SQL NULL and
// therefore fail ordinary comparisons.
type EdgePropertyValue struct {
	Kind   EdgePropertyValueKind
	Number float64
	String string
	Bool   bool
}

// NormalizeEdgeProperties validates and canonicalizes a JSON object for
// durable edge storage. A nil or empty value means the edge has no arbitrary
// properties. Top-level arrays/scalars are rejected so r.field is unambiguous.
func NormalizeEdgeProperties(raw []byte) ([]byte, error) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return nil, nil
	}
	dec, err := apexjson.NewDecoder()
	if err != nil {
		return nil, fmt.Errorf("edge properties must be a JSON object: %w", err)
	}
	defer dec.Free()
	if err := dec.Parse(raw); err != nil {
		return nil, fmt.Errorf("edge properties must be a JSON object: %w", err)
	}
	root := dec.Root()
	if root.Type() == apexjson.TypeNull {
		return nil, nil
	}
	if root.Type() != apexjson.TypeObject {
		return nil, fmt.Errorf("edge properties must be a JSON object")
	}
	object, ok := edgePropertyNative(root)
	if !ok {
		return nil, fmt.Errorf("edge properties contain an unsupported value")
	}
	canonical, err := apexjson.Marshal(object)
	if err != nil {
		return nil, fmt.Errorf("encode edge properties: %w", err)
	}
	out := make([]byte, 1+len(canonical))
	out[0] = EdgePropertyEncodingVersion
	copy(out[1:], canonical)
	return out, nil
}

// EncodeEdgeProperties converts a Go property map into the durable envelope.
func EncodeEdgeProperties(properties map[string]interface{}) ([]byte, error) {
	if len(properties) == 0 {
		return nil, nil
	}
	raw, err := apexjson.Marshal(properties)
	if err != nil {
		return nil, err
	}
	return NormalizeEdgeProperties(raw)
}

func decodeEdgePropertyObject(raw []byte) (map[string]interface{}, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	if raw[0] != EdgePropertyEncodingVersion {
		return nil, fmt.Errorf("unsupported edge property encoding version %d", raw[0])
	}
	dec, err := apexjson.NewDecoder()
	if err != nil {
		return nil, err
	}
	defer dec.Free()
	if err := dec.Parse(raw[1:]); err != nil {
		return nil, err
	}
	object, ok := edgePropertyNative(dec.Root())
	if !ok {
		return nil, fmt.Errorf("edge properties must be a JSON object")
	}
	return object.(map[string]interface{}), nil
}

// EdgePropertyJSON returns the canonical JSON object without the internal
// envelope. It returns a copy suitable for application/API use.
func EdgePropertyJSON(raw []byte) ([]byte, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	if raw[0] != EdgePropertyEncodingVersion {
		return nil, fmt.Errorf("unsupported edge property encoding version %d", raw[0])
	}
	return append([]byte(nil), raw[1:]...), nil
}

func edgePropertyValue(value interface{}) (EdgePropertyValue, bool) {
	if number, ok := value.(interface {
		String() string
		Float64() (float64, error)
		Int64() (int64, error)
	}); ok {
		f, err := number.Float64()
		if err != nil || math.IsNaN(f) || math.IsInf(f, 0) {
			return EdgePropertyValue{}, false
		}
		return EdgePropertyValue{Kind: EdgePropertyNumber, Number: f}, true
	}
	switch v := value.(type) {
	case nil:
		return EdgePropertyValue{Kind: EdgePropertyNull}, true
	case float64:
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return EdgePropertyValue{}, false
		}
		return EdgePropertyValue{Kind: EdgePropertyNumber, Number: v}, true
	case string:
		return EdgePropertyValue{Kind: EdgePropertyString, String: v}, true
	case bool:
		return EdgePropertyValue{Kind: EdgePropertyBool, Bool: v}, true
	default:
		// Objects and arrays are valid stored JSON values, but scalar
		// comparisons against them are deliberately not part of this first
		// predicate implementation.
		return EdgePropertyValue{}, false
	}
}

func findEdgeProperty(raw []byte, name string) (EdgePropertyValue, bool) {
	if len(raw) == 0 || raw[0] != EdgePropertyEncodingVersion {
		return EdgePropertyValue{}, false
	}
	dec, err := apexjson.NewDecoder()
	if err != nil {
		return EdgePropertyValue{}, false
	}
	defer dec.Free()
	if err := dec.Parse(raw[1:]); err != nil {
		return EdgePropertyValue{}, false
	}
	value := dec.Get(name)
	return edgePropertyJSONValue(value)
}

func edgePropertyJSONValue(value apexjson.Value) (EdgePropertyValue, bool) {
	switch value.Type() {
	case apexjson.TypeNull:
		return EdgePropertyValue{Kind: EdgePropertyNull}, true
	case apexjson.TypeNumber:
		f, err := strconv.ParseFloat(string(value.Bytes()), 64)
		if err != nil || math.IsNaN(f) || math.IsInf(f, 0) {
			return EdgePropertyValue{}, false
		}
		return EdgePropertyValue{Kind: EdgePropertyNumber, Number: f}, true
	case apexjson.TypeString:
		return EdgePropertyValue{Kind: EdgePropertyString, String: value.Str()}, true
	case apexjson.TypeBool:
		return EdgePropertyValue{Kind: EdgePropertyBool, Bool: value.Bool()}, true
	default:
		return EdgePropertyValue{}, false
	}
}

func edgePropertyNative(value apexjson.Value) (interface{}, bool) {
	switch value.Type() {
	case apexjson.TypeNull:
		return nil, true
	case apexjson.TypeBool:
		return value.Bool(), true
	case apexjson.TypeString:
		return value.Str(), true
	case apexjson.TypeNumber:
		text := value.Bytes()
		if bytes.IndexAny(text, ".eE") >= 0 {
			f, err := strconv.ParseFloat(string(text), 64)
			if err != nil || math.IsNaN(f) || math.IsInf(f, 0) {
				return nil, false
			}
			return f, true
		}
		if i, err := strconv.ParseInt(string(text), 10, 64); err == nil {
			return i, true
		}
		if u, err := strconv.ParseUint(string(text), 10, 64); err == nil {
			return u, true
		}
		return nil, false
	case apexjson.TypeArray:
		out := make([]interface{}, 0, 4)
		it := value.ArrayIter()
		for it.Next() {
			item, ok := edgePropertyNative(it.Value())
			if !ok {
				return nil, false
			}
			out = append(out, item)
		}
		return out, true
	case apexjson.TypeObject:
		out := make(map[string]interface{}, 8)
		it := value.ObjectIter()
		for it.Next() {
			item, ok := edgePropertyNative(it.Value())
			if !ok {
				return nil, false
			}
			out[it.Key()] = item
		}
		return out, true
	default:
		return nil, false
	}
}
