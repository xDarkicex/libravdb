package graph

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
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
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var object map[string]interface{}
	if err := dec.Decode(&object); err != nil {
		return nil, fmt.Errorf("edge properties must be a JSON object: %w", err)
	}
	if object == nil {
		return nil, nil
	}
	var extra interface{}
	if err := dec.Decode(&extra); err == nil {
		return nil, fmt.Errorf("edge properties contain trailing JSON")
	}
	canonical, err := json.Marshal(object)
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
	raw, err := json.Marshal(properties)
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
	dec := json.NewDecoder(bytes.NewReader(raw[1:]))
	dec.UseNumber()
	var object map[string]interface{}
	if err := dec.Decode(&object); err != nil {
		return nil, err
	}
	return object, nil
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
	switch v := value.(type) {
	case nil:
		return EdgePropertyValue{Kind: EdgePropertyNull}, true
	case json.Number:
		f, err := v.Float64()
		if err != nil || math.IsNaN(f) || math.IsInf(f, 0) {
			return EdgePropertyValue{}, false
		}
		return EdgePropertyValue{Kind: EdgePropertyNumber, Number: f}, true
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
	object, err := decodeEdgePropertyObject(raw)
	if err != nil || object == nil {
		return EdgePropertyValue{}, false
	}
	value, ok := object[name]
	if !ok {
		return EdgePropertyValue{}, false
	}
	return edgePropertyValue(value)
}
