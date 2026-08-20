package libravdb

import (
	"bytes"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

	apexjson "github.com/xDarkicex/apexJSON/v2"
	"github.com/xDarkicex/lexer"
	"github.com/xDarkicex/libravdb/internal/util"
)

// sqlLazyJSONValue keeps a validated JSON document as its original byte span.
// Read-only SQL operators parse that span into ApexJSON's off-heap tape and
// return scalar values or another borrowed span. Mutation and public-result
// boundaries call decodeJSONValue/materializeSQLJSONValue and obtain the
// existing owned Go representation.
//
// Keeping this wrapper private is intentional: callers must not retain a
// query-local borrowed JSON value after the SQL result has been materialized.
type sqlLazyJSONValue struct {
	raw     []byte
	decoder *apexjson.Decoder
	value   apexjson.Value
}

func lazyJSONDocument(raw []byte) (sqlLazyJSONValue, bool) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 {
		return sqlLazyJSONValue{}, false
	}
	dec, err := getSQLJSONDecoder()
	if err != nil {
		return sqlLazyJSONValue{}, false
	}
	defer putSQLJSONDecoder(dec)
	if err := dec.Parse(raw); err != nil {
		return sqlLazyJSONValue{}, false
	}
	return sqlLazyJSONValue{raw: raw}, true
}

func jsonDocumentRaw(value interface{}) ([]byte, bool) {
	switch v := value.(type) {
	case sqlLazyJSONValue:
		return v.raw, len(v.raw) != 0
	case apexjson.RawMessage:
		return []byte(v), len(v) != 0
	default:
		return nil, false
	}
}

func jsonCastSourceRaw(value interface{}) ([]byte, bool) {
	if raw, ok := jsonDocumentRaw(value); ok {
		return raw, true
	}
	switch v := value.(type) {
	case string:
		return []byte(v), len(v) != 0
	case []byte:
		return v, len(v) != 0
	default:
		return nil, false
	}
}

func decodeJSONReadValue(value interface{}) (interface{}, bool) {
	// These values already are owned JSON trees. Read-only operators must not
	// clone them merely to inspect them; mutation paths continue to use
	// decodeJSONValue, which deliberately makes an owned canonical copy.
	switch v := value.(type) {
	case sqlLazyJSONValue:
		// A retained ApexJSON tape is already parsed. Materialize directly from
		// its root when a read-only operator needs a Go tree; reparsing raw would
		// throw away the query-local tape and make containment pay twice.
		if v.value.Exists() {
			return canonicalJSONValue(v.value)
		}
		return decodeJSONValue(v)
	case apexjson.RawMessage:
		return decodeJSONValue(v)
	case nil, util.JSONNull, bool,
		float64, float32, int, int8, int16, int32, int64,
		uint, uint8, uint16, uint32, uint64,
		map[string]interface{}, []interface{}:
		return value, true
	default:
		return decodeJSONValue(value)
	}
}

// JSON operators are evaluated by the query-local relation path. That keeps
// extraction and containment semantics correct for metadata values without
// introducing a second physical index format; a JSON index can be added later
// behind the same operator contract.
func isJSONOperator(operator uint8) bool {
	switch lexer.Kind(operator) {
	case lexer.KindArrowRight, lexer.KindJSONExtract, lexer.KindJSONExtractText,
		lexer.KindJSONPath, lexer.KindJSONPathText,
		lexer.KindJSONContains, lexer.KindJSONContainedBy, lexer.KindJSONExists,
		lexer.KindJSONAny, lexer.KindJSONAll, lexer.KindJSONDelete:
		return true
	default:
		return false
	}
}

func isJSONExtractionOperator(operator uint8) bool {
	switch lexer.Kind(operator) {
	case lexer.KindArrowRight, lexer.KindJSONExtract, lexer.KindJSONExtractText:
		return true
	case lexer.KindJSONPath, lexer.KindJSONPathText:
		return true
	default:
		return false
	}
}

func isJSONTextExtractionOperator(operator uint8) bool {
	kind := lexer.Kind(operator)
	return kind == lexer.KindJSONExtractText || kind == lexer.KindJSONPathText
}

func isJSONContainmentOperator(operator uint8) bool {
	switch lexer.Kind(operator) {
	case lexer.KindJSONContains, lexer.KindJSONContainedBy:
		return true
	default:
		return false
	}
}

func isJSONKeyExistenceOperator(operator uint8) bool {
	return lexer.Kind(operator) == lexer.KindJSONExists
}

func isJSONKeySetOperator(operator uint8) bool {
	kind := lexer.Kind(operator)
	return kind == lexer.KindJSONAny || kind == lexer.KindJSONAll
}

func isJSONPathPredicateOperator(operator uint8) bool {
	kind := lexer.Kind(operator)
	return kind == lexer.KindJSONPathExists || kind == lexer.KindFTSMatch
}

func decodeJSONValue(value interface{}) (interface{}, bool) {
	if text, ok := jsonNumberText(value); ok {
		return canonicalJSONNumber(text)
	}
	switch v := value.(type) {
	case nil:
		return nil, true
	case util.JSONNull:
		return v, true
	case sqlLazyJSONValue:
		if v.value.Exists() {
			// Mutation/output paths still receive an owned tree from
			// canonicalJSONValue, but do not throw away a live query-local tape
			// and parse the same document a second time.
			return canonicalJSONValue(v.value)
		}
		return decodeJSONText(v.raw)
	case apexjson.RawMessage:
		return decodeJSONText([]byte(v))
	case map[string]interface{}, []interface{}, bool, float64, float32, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		return canonicalJSONNode(v)
	case string:
		return decodeJSONText([]byte(v))
	case []byte:
		return decodeJSONText(v)
	default:
		encoded, err := apexjson.Marshal(v)
		if err != nil {
			return nil, false
		}
		return decodeJSONText(encoded)
	}
}

func decodeJSONText(data []byte) (interface{}, bool) {
	data = bytes.TrimSpace(data)
	if len(data) == 0 {
		return nil, false
	}
	decoder, err := getSQLJSONDecoder()
	if err != nil {
		return nil, false
	}
	defer putSQLJSONDecoder(decoder)
	if err := decoder.Parse(data); err != nil {
		return nil, false
	}
	return canonicalJSONValue(decoder.Root())
}

func parseSQLLazyJSON(raw []byte) (*apexjson.Decoder, apexjson.Value, error) {
	decoder, err := getSQLJSONDecoder()
	if err != nil {
		return nil, apexjson.Value{}, err
	}
	if err := decoder.Parse(raw); err != nil {
		putSQLJSONDecoder(decoder)
		return nil, apexjson.Value{}, err
	}
	return decoder, decoder.Root(), nil
}

func sqlLazyJSONValueFromApex(value apexjson.Value, textResult bool) (interface{}, bool, error) {
	switch value.Type() {
	case apexjson.TypeNull:
		if textResult {
			return "null", true, nil
		}
		return util.JSONNull{}, true, nil
	case apexjson.TypeBool:
		if textResult {
			if value.Bool() {
				return "true", true, nil
			}
			return "false", true, nil
		}
		return value.Bool(), true, nil
	case apexjson.TypeNumber:
		raw := value.Bytes()
		if textResult {
			return string(raw), true, nil
		}
		if bytes.IndexAny(raw, ".eE") >= 0 {
			parsed, err := strconv.ParseFloat(string(raw), 64)
			if err != nil || math.IsNaN(parsed) || math.IsInf(parsed, 0) {
				return nil, false, fmt.Errorf("invalid JSON number %q", raw)
			}
			return parsed, true, nil
		}
		if parsed, err := strconv.ParseInt(string(raw), 10, 64); err == nil {
			return parsed, true, nil
		}
		if parsed, err := strconv.ParseUint(string(raw), 10, 64); err == nil {
			return parsed, true, nil
		}
		return nil, false, fmt.Errorf("invalid JSON number %q", raw)
	case apexjson.TypeString:
		return value.Str(), true, nil
	case apexjson.TypeObject, apexjson.TypeArray:
		raw := value.Bytes()
		if textResult {
			return string(raw), true, nil
		}
		return sqlLazyJSONValue{raw: raw}, true, nil
	default:
		return nil, false, fmt.Errorf("invalid JSON value")
	}
}

func sqlLazyJSONValueFromApexWithDecoder(value apexjson.Value, textResult bool, decoder *apexjson.Decoder) (interface{}, bool, error) {
	result, ok, err := sqlLazyJSONValueFromApex(value, textResult)
	if err != nil || !ok || decoder == nil {
		return result, ok, err
	}
	if lazy, isLazy := result.(sqlLazyJSONValue); isLazy {
		lazy.decoder = decoder
		lazy.value = value
		return lazy, true, nil
	}
	return result, true, nil
}

func sqlLazyJSONValueAtPath(root apexjson.Value, segments []string) (apexjson.Value, bool) {
	value := root
	for _, segment := range segments {
		if !value.Exists() {
			return apexjson.Value{}, false
		}
		switch value.Type() {
		case apexjson.TypeObject:
			var found bool
			it := value.ObjectIter()
			for it.Next() {
				if it.Key() == segment {
					value = it.Value()
					found = true
					break
				}
			}
			if !found {
				return apexjson.Value{}, false
			}
		case apexjson.TypeArray:
			index, err := strconv.Atoi(segment)
			if err != nil || index < 0 {
				return apexjson.Value{}, false
			}
			it := value.ArrayIter()
			found := false
			for it.Next() {
				if it.Index() == index {
					value = it.Value()
					found = true
					break
				}
			}
			if !found {
				return apexjson.Value{}, false
			}
		default:
			return apexjson.Value{}, false
		}
	}
	return value, value.Exists()
}

func jsonExtractRaw(document []byte, key interface{}, textResult bool) (interface{}, bool, error) {
	keyText, ok := jsonKeyValue(key)
	if !ok {
		return nil, false, fmt.Errorf("JSON extraction key must be text or an integer")
	}
	decoder, root, err := parseSQLLazyJSON(document)
	if err != nil {
		return nil, false, fmt.Errorf("invalid JSON document: %w", err)
	}
	defer putSQLJSONDecoder(decoder)
	if root.Type() == apexjson.TypeNull {
		return nil, true, nil
	}
	value := decoder.Get(keyText)
	if !value.Exists() {
		return nil, true, nil
	}
	return sqlLazyJSONValueFromApex(value, textResult)
}

func jsonPathRaw(document []byte, path interface{}, textResult bool) (interface{}, bool, error) {
	segments, ok := jsonPathSegments(path)
	if !ok {
		return nil, false, fmt.Errorf("JSON path must be a text array literal such as '{a,b}'")
	}
	decoder, root, err := parseSQLLazyJSON(document)
	if err != nil {
		return nil, false, fmt.Errorf("invalid JSON document: %w", err)
	}
	defer putSQLJSONDecoder(decoder)
	if root.Type() == apexjson.TypeNull {
		return nil, true, nil
	}
	value := decoder.GetPath(segments)
	if !value.Exists() {
		return nil, true, nil
	}
	return sqlLazyJSONValueFromApex(value, textResult)
}

// canonicalJSONValue converts apexJSON's lazy tape into the owned tree used
// by JSONB mutation and comparison paths. Parsing and navigation stay on the
// native off-heap tape; materialization happens only where the SQL value
// contract requires an independently owned Go value.
func canonicalJSONValue(value apexjson.Value) (interface{}, bool) {
	switch value.Type() {
	case apexjson.TypeNull:
		return util.JSONNull{}, true
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
			item, ok := canonicalJSONValue(it.Value())
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
			item, ok := canonicalJSONValue(it.Value())
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

func canonicalJSONNumber(text string) (interface{}, bool) {
	if strings.ContainsAny(text, ".eE") {
		f, err := strconv.ParseFloat(text, 64)
		if err != nil || math.IsNaN(f) || math.IsInf(f, 0) {
			return nil, false
		}
		return f, true
	}
	if i, err := strconv.ParseInt(text, 10, 64); err == nil {
		return i, true
	}
	if u, err := strconv.ParseUint(text, 10, 64); err == nil {
		return u, true
	}
	return nil, false
}

func jsonNumberText(value interface{}) (string, bool) {
	n, ok := value.(interface {
		String() string
		Float64() (float64, error)
		Int64() (int64, error)
	})
	if !ok {
		return "", false
	}
	return n.String(), true
}

// canonicalJSONNode converts a decoded JSON tree into the representation used
// by metadata/WAL rows. Objects and arrays are always newly allocated, so a
// caller cannot mutate a staged or committed JSON value through an input map.
// Integral numbers are kept as int64/uint64 when possible; fractional and
// exponent-form numbers use float64. This also gives JSONB comparisons stable
// numeric semantics for 1 and 1.0.
func canonicalJSONNode(value interface{}) (interface{}, bool) {
	switch v := value.(type) {
	case util.JSONNull:
		return v, true
	case sqlLazyJSONValue:
		decoded, ok := decodeJSONValue(v)
		if !ok {
			return nil, false
		}
		return canonicalJSONNode(decoded)
	case apexjson.RawMessage:
		decoded, ok := decodeJSONValue(v)
		if !ok {
			return nil, false
		}
		return canonicalJSONNode(decoded)
	case nil:
		// A nil encountered inside a decoded JSON tree is the JSON literal
		// null. SQL NULL is kept distinct by decodeJSONValue's top-level nil
		// case and never enters this canonicalization path.
		return util.JSONNull{}, true
	case bool, string:
		return v, true
	case float32:
		f := float64(v)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return nil, false
		}
		if f == math.Trunc(f) && f >= -9223372036854775808 && f < 9223372036854775808 {
			return int64(f), true
		}
		return f, true
	case float64:
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, false
		}
		if v == math.Trunc(v) && v >= -9223372036854775808 && v < 9223372036854775808 {
			return int64(v), true
		}
		return v, true
	case int:
		return int64(v), true
	case int8:
		return int64(v), true
	case int16:
		return int64(v), true
	case int32:
		return int64(v), true
	case int64:
		return v, true
	case uint:
		return uint64(v), true
	case uint8:
		return uint64(v), true
	case uint16:
		return uint64(v), true
	case uint32:
		return uint64(v), true
	case uint64:
		return v, true
	case []string:
		out := make([]interface{}, len(v))
		for i := range v {
			out[i] = v[i]
		}
		return out, true
	case []interface{}:
		out := make([]interface{}, len(v))
		for i := range v {
			item, ok := canonicalJSONNode(v[i])
			if !ok {
				return nil, false
			}
			out[i] = item
		}
		return out, true
	case map[string]interface{}:
		out := make(map[string]interface{}, len(v))
		for key, item := range v {
			canonical, ok := canonicalJSONNode(item)
			if !ok {
				return nil, false
			}
			out[key] = canonical
		}
		return out, true
	default:
		return nil, false
	}
}

func encodeJSONValue(value interface{}) (string, error) {
	encoded, err := apexjson.Marshal(jsonWireValue(value))
	if err != nil {
		return "", err
	}
	return string(encoded), nil
}

// jsonWireValue converts the internal JSON-null sentinel back to the JSON
// encoder's nil representation, recursively. It deliberately leaves SQL nil
// untouched; callers use this only when serializing a JSON document.
func jsonWireValue(value interface{}) interface{} {
	switch v := value.(type) {
	case util.JSONNull:
		return nil
	case sqlLazyJSONValue:
		if decoded, ok := decodeJSONValue(v); ok {
			return jsonWireValue(decoded)
		}
		return nil
	case apexjson.RawMessage:
		return v
	case []interface{}:
		out := make([]interface{}, len(v))
		for i := range v {
			out[i] = jsonWireValue(v[i])
		}
		return out
	case map[string]interface{}:
		out := make(map[string]interface{}, len(v))
		for key, item := range v {
			out[key] = jsonWireValue(item)
		}
		return out
	default:
		return value
	}
}

func materializeSQLJSONValue(value interface{}) interface{} {
	switch v := value.(type) {
	case sqlLazyJSONValue:
		if v.value.Exists() {
			if decoded, ok := canonicalJSONValue(v.value); ok {
				return decoded
			}
		}
		if decoded, ok := decodeJSONValue(v); ok {
			return decoded
		}
		return nil
	case apexjson.RawMessage:
		if decoded, ok := decodeJSONValue(v); ok {
			return decoded
		}
		return nil
	case []interface{}:
		return v
	case map[string]interface{}:
		return v
	default:
		return value
	}
}

func jsonKeyValue(value interface{}) (string, bool) {
	switch v := value.(type) {
	case string:
		return v, true
	case []byte:
		return string(v), true
	case float64:
		return strconv.FormatInt(int64(v), 10), v == float64(int64(v))
	case float32:
		return strconv.FormatInt(int64(v), 10), v == float32(int64(v))
	case int:
		return strconv.Itoa(v), true
	case int64:
		return strconv.FormatInt(v, 10), true
	case uint64:
		return strconv.FormatUint(v, 10), true
	default:
		return "", false
	}
}

func jsonExtract(document, key interface{}, textResult bool) (interface{}, bool, error) {
	if lazy, ok := document.(sqlLazyJSONValue); ok && lazy.decoder != nil && lazy.value.Exists() {
		keyText, keyOK := jsonKeyValue(key)
		if !keyOK {
			return nil, false, fmt.Errorf("JSON extraction key must be text or an integer")
		}
		value, found := sqlLazyJSONValueAtPath(lazy.value, []string{keyText})
		if !found {
			return nil, true, nil
		}
		return sqlLazyJSONValueFromApexWithDecoder(value, textResult, lazy.decoder)
	}
	if raw, rawOK := jsonDocumentRaw(document); rawOK {
		return jsonExtractRaw(raw, key, textResult)
	}
	root, ok := decodeJSONReadValue(document)
	if !ok {
		return nil, false, fmt.Errorf("invalid JSON document")
	}
	keyValue, keyOK := jsonKeyValue(key)
	if !keyOK {
		return nil, false, fmt.Errorf("JSON extraction key must be text or an integer")
	}
	var extracted interface{}
	switch container := root.(type) {
	case map[string]interface{}:
		extracted, ok = container[keyValue]
	case []interface{}:
		index, err := strconv.Atoi(keyValue)
		if err != nil || index < 0 || index >= len(container) {
			return nil, true, nil
		}
		extracted, ok = container[index], true
	default:
		return nil, true, nil
	}
	if !ok {
		return nil, true, nil
	}
	if textResult {
		if scalar, ok := extracted.(string); ok {
			return scalar, true, nil
		}
		encoded, err := encodeJSONValue(extracted)
		return encoded, err == nil, err
	}
	return extracted, true, nil
}

func jsonPath(document, path interface{}, textResult bool) (interface{}, bool, error) {
	if lazy, ok := document.(sqlLazyJSONValue); ok && lazy.decoder != nil && lazy.value.Exists() {
		segments, segmentsOK := jsonPathSegments(path)
		if !segmentsOK {
			return nil, false, fmt.Errorf("JSON path must be a text array literal such as '{a,b}'")
		}
		value, found := sqlLazyJSONValueAtPath(lazy.value, segments)
		if !found {
			return nil, true, nil
		}
		return sqlLazyJSONValueFromApexWithDecoder(value, textResult, lazy.decoder)
	}
	if raw, rawOK := jsonDocumentRaw(document); rawOK {
		return jsonPathRaw(raw, path, textResult)
	}
	root, ok := decodeJSONReadValue(document)
	if !ok {
		return nil, false, fmt.Errorf("invalid JSON document")
	}
	segments, ok := jsonPathSegments(path)
	if !ok {
		return nil, false, fmt.Errorf("JSON path must be a text array literal such as '{a,b}'")
	}
	value := root
	for _, segment := range segments {
		var found bool
		switch container := value.(type) {
		case map[string]interface{}:
			value, found = container[segment]
		case []interface{}:
			index, err := strconv.Atoi(segment)
			if err != nil || index < 0 || index >= len(container) {
				return nil, true, nil
			}
			value, found = container[index], true
		default:
			return nil, true, nil
		}
		if !found {
			return nil, true, nil
		}
	}
	if textResult {
		if scalar, ok := value.(string); ok {
			return scalar, true, nil
		}
		encoded, err := encodeJSONValue(value)
		return encoded, err == nil, err
	}
	return value, true, nil
}

func jsonPathSegments(path interface{}) ([]string, bool) {
	text, ok := jsonKeyValue(path)
	if !ok {
		return nil, false
	}
	text = strings.TrimSpace(text)
	if len(text) < 2 || text[0] != '{' || text[len(text)-1] != '}' {
		return nil, false
	}
	text = text[1 : len(text)-1]
	if text == "" {
		return []string{}, true
	}
	segments := make([]string, 0, 4)
	start := 0
	for i := 0; i <= len(text); i++ {
		if i != len(text) && text[i] != ',' {
			continue
		}
		segment := strings.TrimSpace(text[start:i])
		if len(segment) >= 2 && segment[0] == '"' && segment[len(segment)-1] == '"' {
			segment = segment[1 : len(segment)-1]
		}
		segment = strings.ReplaceAll(segment, `\,`, ",")
		segments = append(segments, segment)
		start = i + 1
	}
	return segments, true
}

func jsonExists(document, key interface{}) (bool, error) {
	if raw, rawOK := jsonDocumentRaw(document); rawOK {
		keyText, keyOK := jsonKeyValue(key)
		if !keyOK {
			return false, fmt.Errorf("JSON existence key must be text")
		}
		decoder, root, err := parseSQLLazyJSON(raw)
		if err != nil {
			return false, fmt.Errorf("invalid JSON document: %w", err)
		}
		defer putSQLJSONDecoder(decoder)
		if root.Type() != apexjson.TypeObject {
			return false, nil
		}
		return decoder.Get(keyText).Exists(), nil
	}
	root, ok := decodeJSONReadValue(document)
	if !ok {
		return false, fmt.Errorf("invalid JSON document")
	}
	keyText, ok := jsonKeyValue(key)
	if !ok {
		return false, fmt.Errorf("JSON existence key must be text")
	}
	switch container := root.(type) {
	case map[string]interface{}:
		_, exists := container[keyText]
		return exists, nil
	case []interface{}:
		for _, candidate := range container {
			if text, ok := candidate.(string); ok && text == keyText {
				return true, nil
			}
		}
	}
	return false, nil
}

// jsonKeyList accepts PostgreSQL's text-array literal form ({a,b}) as well as
// native []string/[]interface{} values supplied through the Go or pgwire
// parameter APIs. Quoted array elements and escaped commas are handled by the
// same path-segment tokenizer used by #>/#>>.
func jsonKeyList(value interface{}) ([]string, bool) {
	if list, ok := value.([]string); ok {
		return append([]string(nil), list...), true
	}
	if list, ok := value.([]interface{}); ok {
		out := make([]string, 0, len(list))
		for _, item := range list {
			key, ok := jsonKeyValue(item)
			if !ok {
				return nil, false
			}
			out = append(out, key)
		}
		return out, true
	}
	text, ok := jsonKeyValue(value)
	if !ok {
		return nil, false
	}
	text = strings.TrimSpace(text)
	if len(text) < 2 || text[0] != '{' || text[len(text)-1] != '}' {
		return nil, false
	}
	segments, ok := jsonPathSegments(text)
	return segments, ok
}

func parseJSONArrayConstructor(value string) ([]interface{}, bool) {
	text := strings.TrimSpace(value)
	if len(text) < len("array[]") || !strings.EqualFold(text[:5], "array") || text[5] != '[' || text[len(text)-1] != ']' {
		return nil, false
	}
	inner := text[6 : len(text)-1]
	if strings.TrimSpace(inner) == "" {
		return []interface{}{}, true
	}
	items := make([]interface{}, 0, 4)
	start := 0
	inString := false
	for i := 0; i <= len(inner); i++ {
		if i < len(inner) && inner[i] == '\'' {
			if inString && i+1 < len(inner) && inner[i+1] == '\'' {
				i++
				continue
			}
			inString = !inString
		}
		if i != len(inner) && (inner[i] != ',' || inString) {
			continue
		}
		item := strings.TrimSpace(inner[start:i])
		if len(item) < 2 || item[0] != '\'' || item[len(item)-1] != '\'' {
			return nil, false
		}
		item = strings.ReplaceAll(item[1:len(item)-1], "''", "'")
		items = append(items, item)
		start = i + 1
	}
	return items, true
}

func jsonAnyOrAll(document, keys interface{}, requireAll bool) (bool, error) {
	if raw, rawOK := jsonDocumentRaw(document); rawOK {
		keyList, keyOK := jsonKeyList(keys)
		if !keyOK || len(keyList) == 0 {
			return false, fmt.Errorf("JSON key-set operand must be a non-empty text array")
		}
		decoder, root, err := parseSQLLazyJSON(raw)
		if err != nil {
			return false, fmt.Errorf("invalid JSON document: %w", err)
		}
		defer putSQLJSONDecoder(decoder)
		for _, key := range keyList {
			exists := false
			if root.Type() == apexjson.TypeObject {
				exists = decoder.Get(key).Exists()
			} else if root.Type() == apexjson.TypeArray {
				it := root.ArrayIter()
				for it.Next() {
					item := it.Value()
					if item.Type() == apexjson.TypeString && item.Str() == key {
						exists = true
						break
					}
				}
			}
			if exists != requireAll {
				return !requireAll, nil
			}
		}
		return requireAll, nil
	}
	root, ok := decodeJSONReadValue(document)
	if !ok {
		return false, fmt.Errorf("invalid JSON document")
	}
	keyList, ok := jsonKeyList(keys)
	if !ok || len(keyList) == 0 {
		return false, fmt.Errorf("JSON key-set operand must be a non-empty text array")
	}
	for _, key := range keyList {
		exists := false
		switch container := root.(type) {
		case map[string]interface{}:
			_, exists = container[key]
		case []interface{}:
			for _, item := range container {
				if text, ok := item.(string); ok && text == key {
					exists = true
					break
				}
			}
		}
		if exists != requireAll {
			return !requireAll, nil
		}
	}
	return requireAll, nil
}

// jsonMutationRaw converts the SQL JSON value contract into one complete raw
// JSON value without building a Go JSON tree. Strings and byte slices retain
// the existing SQL behavior: they are JSON text and must therefore contain a
// valid JSON value. Structured values are encoded directly by ApexJSON.
func jsonMutationRaw(value interface{}) ([]byte, error) {
	if raw, ok := jsonDocumentRaw(value); ok {
		raw = bytes.TrimSpace(raw)
		if len(raw) != 0 {
			return raw, nil
		}
	}
	switch v := value.(type) {
	case string:
		raw := bytes.TrimSpace([]byte(v))
		if len(raw) != 0 {
			return raw, nil
		}
	case []byte:
		raw := bytes.TrimSpace(v)
		if len(raw) != 0 {
			return raw, nil
		}
	case util.JSONNull:
		return []byte("null"), nil
	}
	encoded, err := apexjson.Marshal(jsonWireValue(value))
	if err != nil {
		return nil, err
	}
	return encoded, nil
}

// nativeJSONMutation applies a variable-size JSON mutation through
// ApexJSON's off-heap candidate/commit path. The returned RawMessage owns a
// copy of the committed span because the pooled decoder is reset before this
// function returns.
func nativeJSONMutation(document, path, replacement interface{}, mutate func(*apexjson.Decoder, []string, []byte) (bool, error)) (interface{}, error) {
	documentRaw, err := jsonMutationRaw(document)
	if err != nil {
		return nil, fmt.Errorf("invalid JSON document: %w", err)
	}
	segments, ok := jsonPathSegments(path)
	if !ok || len(segments) == 0 {
		return nil, fmt.Errorf("JSON mutation path must be a non-empty text array")
	}
	replacementRaw, err := jsonMutationRaw(replacement)
	if err != nil {
		return nil, fmt.Errorf("invalid JSON replacement value: %w", err)
	}
	decoder, err := getSQLJSONDecoder()
	if err != nil {
		return nil, err
	}
	defer putSQLJSONDecoder(decoder)
	if err := decoder.Parse(documentRaw); err != nil {
		return nil, fmt.Errorf("invalid JSON document: %w", err)
	}
	if _, err := mutate(decoder, segments, replacementRaw); err != nil {
		return nil, err
	}
	committed := decoder.Root().RawJSON()
	if len(committed) == 0 {
		return nil, fmt.Errorf("invalid JSON mutation result")
	}
	owned := make(apexjson.RawMessage, len(committed))
	copy(owned, committed)
	return owned, nil
}

func jsonbSet(document, path, replacement interface{}, createMissing bool) (interface{}, error) {
	return nativeJSONMutation(document, path, replacement, func(decoder *apexjson.Decoder, segments []string, raw []byte) (bool, error) {
		return decoder.SetPathRaw(segments, raw, createMissing)
	})
}

func jsonbInsert(document, path, replacement interface{}, insertAfter bool) (interface{}, error) {
	return nativeJSONMutation(document, path, replacement, func(decoder *apexjson.Decoder, segments []string, raw []byte) (bool, error) {
		return decoder.InsertPathRaw(segments, raw, insertAfter)
	})
}

func deleteJSONPath(node interface{}, segments []string) (interface{}, bool) {
	if len(segments) == 0 {
		return node, false
	}
	switch current := node.(type) {
	case map[string]interface{}:
		key := segments[0]
		if len(segments) == 1 {
			if _, exists := current[key]; !exists {
				return node, false
			}
			out := make(map[string]interface{}, len(current)-1)
			for k, value := range current {
				if k != key {
					out[k] = value
				}
			}
			return out, true
		}
		child, exists := current[key]
		if !exists {
			return node, false
		}
		updated, changed := deleteJSONPath(child, segments[1:])
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
		index, err := strconv.Atoi(segments[0])
		if err != nil {
			return node, false
		}
		if index < 0 {
			index = len(current) + index
		}
		if index < 0 || index >= len(current) {
			return node, false
		}
		if len(segments) == 1 {
			out := make([]interface{}, 0, len(current)-1)
			out = append(out, current[:index]...)
			out = append(out, current[index+1:]...)
			return out, true
		}
		updated, changed := deleteJSONPath(current[index], segments[1:])
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

func jsonbDelete(document, path interface{}) (interface{}, error) {
	root, ok := decodeJSONValue(document)
	if !ok {
		return nil, fmt.Errorf("invalid JSON document")
	}
	segments, ok := jsonPathSegments(path)
	if !ok || len(segments) == 0 {
		return nil, fmt.Errorf("JSON delete path must be a non-empty text array")
	}
	updated, _ := deleteJSONPath(root, segments)
	return updated, nil
}

func jsonbConcat(left, right interface{}) (interface{}, error) {
	// Concatenation produces a new JSON value. Keep the ownership boundary
	// used by mutation/output paths so the result cannot alias a stored tree.
	lhs, ok := decodeJSONValue(left)
	if !ok {
		return nil, fmt.Errorf("invalid left JSON value")
	}
	rhs, ok := decodeJSONValue(right)
	if !ok {
		return nil, fmt.Errorf("invalid right JSON value")
	}
	if lm, ok := lhs.(map[string]interface{}); ok {
		if rm, ok := rhs.(map[string]interface{}); ok {
			out := make(map[string]interface{}, len(lm)+len(rm))
			for key, value := range lm {
				out[key] = value
			}
			for key, value := range rm {
				out[key] = value
			}
			return out, nil
		}
	}
	leftArray, leftIsArray := lhs.([]interface{})
	rightArray, rightIsArray := rhs.([]interface{})
	if !leftIsArray {
		leftArray = []interface{}{lhs}
	}
	if !rightIsArray {
		rightArray = []interface{}{rhs}
	}
	out := make([]interface{}, 0, len(leftArray)+len(rightArray))
	out = append(out, leftArray...)
	out = append(out, rightArray...)
	return out, nil
}

func jsonConstructorValue(value interface{}) (interface{}, bool) {
	// SQL string arguments to json_build_* are JSON strings, not JSON text to
	// parse. Explicit ::json/jsonb casts already arrive as decoded trees.
	if lazy, ok := value.(sqlLazyJSONValue); ok {
		return decodeJSONValue(lazy)
	}
	if raw, ok := value.(apexjson.RawMessage); ok {
		return decodeJSONValue(raw)
	}
	if text, ok := value.(string); ok {
		return text, true
	}
	if data, ok := value.([]byte); ok {
		return string(data), true
	}
	return canonicalJSONNode(value)
}

func jsonBuildArray(args []interface{}) (interface{}, error) {
	out := make([]interface{}, len(args))
	for i, arg := range args {
		value, ok := jsonConstructorValue(arg)
		if !ok {
			return nil, fmt.Errorf("invalid JSON constructor value at argument %d", i+1)
		}
		out[i] = value
	}
	return out, nil
}

func jsonBuildObject(args []interface{}) (interface{}, error) {
	if len(args)%2 != 0 {
		return nil, fmt.Errorf("json_build_object requires an even number of arguments")
	}
	out := make(map[string]interface{}, len(args)/2)
	for i := 0; i < len(args); i += 2 {
		key, ok := jsonKeyValue(args[i])
		if !ok || key == "" {
			return nil, fmt.Errorf("json_build_object keys must be non-empty text")
		}
		value, ok := jsonConstructorValue(args[i+1])
		if !ok {
			return nil, fmt.Errorf("invalid JSON constructor value for key %q", key)
		}
		out[key] = value
	}
	return out, nil
}

func jsonbTypeof(value interface{}) (string, bool) {
	if lazy, lazyOK := value.(sqlLazyJSONValue); lazyOK && lazy.value.Exists() {
		switch lazy.value.Type() {
		case apexjson.TypeNull:
			return "null", true
		case apexjson.TypeObject:
			return "object", true
		case apexjson.TypeArray:
			return "array", true
		case apexjson.TypeString:
			return "string", true
		case apexjson.TypeBool:
			return "boolean", true
		case apexjson.TypeNumber:
			return "number", true
		}
	}
	if raw, rawOK := jsonDocumentRaw(value); rawOK {
		decoder, root, err := parseSQLLazyJSON(raw)
		if err != nil {
			return "", false
		}
		defer putSQLJSONDecoder(decoder)
		switch root.Type() {
		case apexjson.TypeNull:
			return "null", true
		case apexjson.TypeObject:
			return "object", true
		case apexjson.TypeArray:
			return "array", true
		case apexjson.TypeString:
			return "string", true
		case apexjson.TypeBool:
			return "boolean", true
		case apexjson.TypeNumber:
			return "number", true
		default:
			return "", false
		}
	}
	node, ok := decodeJSONReadValue(value)
	// JSON extraction returns a native Go string for a JSON string member.
	// It is already decoded JSON at that point, so do not require it to be
	// re-encoded as a JSON text literal before reporting its JSONB type.
	if !ok {
		if _, isString := value.(string); isString {
			return "string", true
		}
	}
	if !ok {
		return "", false
	}
	switch node.(type) {
	case nil, util.JSONNull:
		return "null", true
	case map[string]interface{}:
		return "object", true
	case []interface{}:
		return "array", true
	case string:
		return "string", true
	case bool:
		return "boolean", true
	default:
		return "number", true
	}
}

func evaluateJSONFunction(name string, args []interface{}) (interface{}, bool, error) {
	switch {
	case strings.EqualFold(name, "jsonb_set") || strings.EqualFold(name, "json_set"):
		if len(args) < 3 || len(args) > 4 {
			return nil, false, fmt.Errorf("%s expects document, path, replacement, and optional create_missing", name)
		}
		if args[0] == nil || args[1] == nil || args[2] == nil {
			return nil, true, nil
		}
		createMissing := true
		if len(args) == 4 {
			flag, ok := jsonFunctionBool(args[3])
			if !ok {
				return nil, false, fmt.Errorf("%s create_missing must be boolean", name)
			}
			createMissing = flag
		}
		value, err := jsonbSet(args[0], args[1], args[2], createMissing)
		return value, true, err
	case strings.EqualFold(name, "jsonb_insert") || strings.EqualFold(name, "json_insert"):
		if len(args) < 3 || len(args) > 4 {
			return nil, false, fmt.Errorf("%s expects document, path, replacement, and optional insert_after", name)
		}
		if args[0] == nil || args[1] == nil || args[2] == nil {
			return nil, true, nil
		}
		insertAfter := false
		if len(args) == 4 {
			flag, ok := jsonFunctionBool(args[3])
			if !ok {
				return nil, false, fmt.Errorf("%s insert_after must be boolean", name)
			}
			insertAfter = flag
		}
		value, err := jsonbInsert(args[0], args[1], args[2], insertAfter)
		return value, true, err
	case strings.EqualFold(name, "jsonb_build_array") || strings.EqualFold(name, "json_build_array"):
		value, err := jsonBuildArray(args)
		return value, true, err
	case strings.EqualFold(name, "jsonb_build_object") || strings.EqualFold(name, "json_build_object"):
		value, err := jsonBuildObject(args)
		return value, true, err
	case strings.EqualFold(name, "to_jsonb") || strings.EqualFold(name, "to_json"):
		if len(args) != 1 {
			return nil, false, fmt.Errorf("%s expects one argument", name)
		}
		if args[0] == nil {
			return nil, true, nil
		}
		value, ok := jsonConstructorValue(args[0])
		if !ok {
			return nil, false, fmt.Errorf("%s received an unsupported value", name)
		}
		return value, true, nil
	case strings.EqualFold(name, "jsonb_populate_record") || strings.EqualFold(name, "json_populate_record"):
		if len(args) != 2 {
			return nil, false, fmt.Errorf("%s expects a base object and JSON object", name)
		}
		base, ok := decodeJSONValue(args[0])
		if !ok || base == nil {
			base = map[string]interface{}{}
		}
		baseObject, ok := base.(map[string]interface{})
		if !ok {
			return nil, false, fmt.Errorf("%s base value must be an object", name)
		}
		patch, ok := decodeJSONValue(args[1])
		if !ok {
			return nil, false, fmt.Errorf("%s JSON value is invalid", name)
		}
		patchObject, ok := patch.(map[string]interface{})
		if !ok {
			return nil, false, fmt.Errorf("%s JSON value must be an object", name)
		}
		out := make(map[string]interface{}, len(baseObject)+len(patchObject))
		for key, item := range baseObject {
			out[key] = item
		}
		for key, item := range patchObject {
			out[key] = item
		}
		return out, true, nil
	case strings.EqualFold(name, "jsonb_array_length"):
		if len(args) != 1 {
			return nil, false, fmt.Errorf("jsonb_array_length expects one argument")
		}
		if args[0] == nil {
			return nil, true, nil
		}
		if raw, rawOK := jsonDocumentRaw(args[0]); rawOK {
			decoder, root, err := parseSQLLazyJSON(raw)
			if err != nil {
				return nil, false, fmt.Errorf("invalid JSON document: %w", err)
			}
			defer putSQLJSONDecoder(decoder)
			if root.Type() != apexjson.TypeArray {
				return nil, false, fmt.Errorf("jsonb_array_length requires an array")
			}
			return int64(countApexJSONArray(root)), true, nil
		}
		node, ok := decodeJSONReadValue(args[0])
		if !ok {
			return nil, false, fmt.Errorf("invalid JSON document")
		}
		array, ok := node.([]interface{})
		if !ok {
			return nil, false, fmt.Errorf("jsonb_array_length requires an array")
		}
		return int64(len(array)), true, nil
	case strings.EqualFold(name, "jsonb_typeof") || strings.EqualFold(name, "json_typeof"):
		if len(args) != 1 {
			return nil, false, fmt.Errorf("%s expects one argument", name)
		}
		if args[0] == nil {
			return nil, true, nil
		}
		value, ok := jsonbTypeof(args[0])
		return value, ok, nil
	default:
		return nil, false, nil
	}
}

func jsonFunctionBool(value interface{}) (bool, bool) {
	switch v := value.(type) {
	case bool:
		return v, true
	case string:
		parsed, err := strconv.ParseBool(strings.TrimSpace(v))
		return parsed, err == nil
	default:
		return false, false
	}
}

// evaluateJSONArrayExpansion evaluates PostgreSQL's JSON set-returning array
// functions for a FROM table-function source. The returned slice owns its
// decoded values; row materialization adds another metadata copy before it is
// exposed to callers.
func evaluateJSONArrayExpansion(name string, args []interface{}) ([]interface{}, bool, error) {
	textMode := strings.EqualFold(name, "json_array_elements_text") || strings.EqualFold(name, "jsonb_array_elements_text")
	objectKeys := strings.EqualFold(name, "json_object_keys") || strings.EqualFold(name, "jsonb_object_keys")
	each := strings.EqualFold(name, "json_each") || strings.EqualFold(name, "jsonb_each") || strings.EqualFold(name, "json_each_text") || strings.EqualFold(name, "jsonb_each_text")
	record := strings.EqualFold(name, "json_to_record") || strings.EqualFold(name, "jsonb_to_record")
	recordset := strings.EqualFold(name, "json_to_recordset") || strings.EqualFold(name, "jsonb_to_recordset")
	populateRecord := strings.EqualFold(name, "json_populate_record") || strings.EqualFold(name, "jsonb_populate_record")
	populateRecordset := strings.EqualFold(name, "json_populate_recordset") || strings.EqualFold(name, "jsonb_populate_recordset")
	if !strings.EqualFold(name, "json_array_elements") && !strings.EqualFold(name, "jsonb_array_elements") && !textMode && !objectKeys && !each {
		if !record && !recordset && !populateRecord && !populateRecordset {
			return nil, false, nil
		}
	}
	if (record || recordset) && len(args) != 1 {
		return nil, true, fmt.Errorf("%s expects one JSON argument", name)
	}
	if populateRecord && len(args) != 2 {
		return nil, true, fmt.Errorf("%s expects a base object and JSON object", name)
	}
	if populateRecordset && len(args) != 2 {
		return nil, true, fmt.Errorf("%s expects a base object and JSON array", name)
	}
	if !record && !recordset && !populateRecord && !populateRecordset && len(args) != 1 {
		return nil, true, fmt.Errorf("%s expects one JSON array argument", name)
	}
	if record || recordset || populateRecord || populateRecordset {
		var source interface{}
		var ok bool
		if populateRecord || populateRecordset {
			source, ok = decodeJSONValue(args[1])
		} else {
			source, ok = decodeJSONValue(args[0])
		}
		if !ok {
			return nil, true, fmt.Errorf("invalid JSON document")
		}
		if record {
			object, isObject := source.(map[string]interface{})
			if !isObject {
				return nil, true, fmt.Errorf("%s requires a JSON object", name)
			}
			return []interface{}{object}, true, nil
		}
		if populateRecord {
			object, isObject := source.(map[string]interface{})
			if !isObject {
				return nil, true, fmt.Errorf("%s JSON value must be an object", name)
			}
			decoded, baseOK := decodeJSONValue(args[0])
			if !baseOK || decoded == nil {
				return nil, true, fmt.Errorf("%s base value is invalid", name)
			}
			base, isObject := decoded.(map[string]interface{})
			if !isObject {
				return nil, true, fmt.Errorf("%s base value must be an object", name)
			}
			merged := make(map[string]interface{}, len(base)+len(object))
			for key, value := range base {
				merged[key] = value
			}
			for key, value := range object {
				merged[key] = value
			}
			return []interface{}{merged}, true, nil
		}
		array, isArray := source.([]interface{})
		if !isArray {
			return nil, true, fmt.Errorf("%s requires a JSON array", name)
		}
		base := map[string]interface{}{}
		if populateRecordset {
			decoded, baseOK := decodeJSONValue(args[0])
			if !baseOK || decoded == nil {
				return nil, true, fmt.Errorf("%s base value is invalid", name)
			}
			var isObject bool
			base, isObject = decoded.(map[string]interface{})
			if !isObject {
				return nil, true, fmt.Errorf("%s base value must be an object", name)
			}
		}
		out := make([]interface{}, 0, len(array))
		for _, item := range array {
			object, isObject := item.(map[string]interface{})
			if !isObject {
				return nil, true, fmt.Errorf("%s array elements must be objects", name)
			}
			merged := make(map[string]interface{}, len(base)+len(object))
			for key, value := range base {
				merged[key] = value
			}
			for key, value := range object {
				merged[key] = value
			}
			out = append(out, merged)
		}
		return out, true, nil
	}
	if lazy, lazyOK := args[0].(sqlLazyJSONValue); lazyOK && lazy.value.Exists() {
		items, err := evaluateJSONArrayExpansionApex(name, lazy.value, textMode, objectKeys, each, lazy.decoder)
		if err != nil {
			return nil, true, err
		}
		return items, true, nil
	}
	if raw, rawOK := jsonDocumentRaw(args[0]); rawOK {
		items, err := evaluateJSONArrayExpansionRaw(name, raw, textMode, objectKeys, each)
		if err != nil {
			return nil, true, err
		}
		return items, true, nil
	}
	node, ok := decodeJSONReadValue(args[0])
	if !ok {
		return nil, true, fmt.Errorf("invalid JSON document")
	}
	if objectKeys || each {
		object, ok := node.(map[string]interface{})
		if !ok {
			return nil, true, fmt.Errorf("%s requires an object", name)
		}
		keys := make([]string, 0, len(object))
		for key := range object {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		out := make([]interface{}, 0, len(keys))
		for _, key := range keys {
			if objectKeys {
				out = append(out, key)
				continue
			}
			item := object[key]
			if strings.HasSuffix(strings.ToLower(name), "_text") {
				if text, ok := item.(string); ok {
					item = text
				} else if encoded, err := encodeJSONValue(item); err == nil {
					item = encoded
				}
			}
			out = append(out, map[string]interface{}{"key": key, "value": item})
		}
		return out, true, nil
	}
	array, ok := node.([]interface{})
	if !ok {
		return nil, true, fmt.Errorf("%s requires an array", name)
	}
	out := make([]interface{}, len(array))
	for i, value := range array {
		if textMode {
			if text, ok := value.(string); ok {
				out[i] = text
			} else {
				encoded, err := encodeJSONValue(value)
				if err != nil {
					return nil, true, err
				}
				out[i] = encoded
			}
		} else {
			out[i] = value
		}
	}
	return out, true, nil
}

func evaluateJSONArrayExpansionApex(name string, root apexjson.Value, textMode, objectKeys, each bool, decoder *apexjson.Decoder) ([]interface{}, error) {
	if objectKeys || each {
		if root.Type() != apexjson.TypeObject {
			return nil, fmt.Errorf("%s requires an object", name)
		}
		keys := make([]string, 0, 8)
		it := root.ObjectIter()
		for it.Next() {
			keys = append(keys, it.Key())
		}
		sort.Strings(keys)
		out := make([]interface{}, 0, len(keys))
		for _, key := range keys {
			value, exists := jsonApexObjectField(root, key)
			if !exists {
				continue
			}
			if objectKeys {
				out = append(out, key)
				continue
			}
			item, ok, err := sqlLazyJSONValueFromApexWithDecoder(value, strings.HasSuffix(strings.ToLower(name), "_text"), decoder)
			if err != nil || !ok {
				return nil, err
			}
			item = materializeSQLJSONValue(item)
			out = append(out, map[string]interface{}{"key": key, "value": item})
		}
		return out, nil
	}
	if root.Type() != apexjson.TypeArray {
		return nil, fmt.Errorf("%s requires an array", name)
	}
	out := make([]interface{}, 0, 8)
	it := root.ArrayIter()
	for it.Next() {
		item, ok, err := sqlLazyJSONValueFromApexWithDecoder(it.Value(), textMode, decoder)
		if err != nil || !ok {
			return nil, err
		}
		out = append(out, item)
	}
	return out, nil
}

func evaluateJSONArrayExpansionRaw(name string, raw []byte, textMode, objectKeys, each bool) ([]interface{}, error) {
	decoder, root, err := parseSQLLazyJSON(raw)
	if err != nil {
		return nil, fmt.Errorf("invalid JSON document: %w", err)
	}
	defer putSQLJSONDecoder(decoder)
	if objectKeys || each {
		if root.Type() != apexjson.TypeObject {
			return nil, fmt.Errorf("%s requires an object", name)
		}
		keys := make([]string, 0, 8)
		it := root.ObjectIter()
		for it.Next() {
			keys = append(keys, it.Key())
		}
		sort.Strings(keys)
		out := make([]interface{}, 0, len(keys))
		for _, key := range keys {
			value, exists := jsonApexObjectField(root, key)
			if !exists {
				continue
			}
			if objectKeys {
				out = append(out, key)
				continue
			}
			item, ok, err := sqlLazyJSONValueFromApex(value, strings.HasSuffix(strings.ToLower(name), "_text"))
			if err != nil || !ok {
				return nil, err
			}
			item = materializeSQLJSONValue(item)
			out = append(out, map[string]interface{}{"key": key, "value": item})
		}
		return out, nil
	}
	if root.Type() != apexjson.TypeArray {
		return nil, fmt.Errorf("%s requires an array", name)
	}
	out := make([]interface{}, 0, 8)
	it := root.ArrayIter()
	for it.Next() {
		item, ok, err := sqlLazyJSONValueFromApex(it.Value(), textMode)
		if err != nil || !ok {
			return nil, err
		}
		out = append(out, item)
	}
	return out, nil
}

func countApexJSONArray(value apexjson.Value) int {
	count := 0
	it := value.ArrayIter()
	for it.Next() {
		count++
	}
	return count
}

func jsonContains(left, right interface{}) (bool, error) {
	if leftLazy, leftOK := left.(sqlLazyJSONValue); leftOK && leftLazy.value.Exists() {
		if rightLazy, rightOK := right.(sqlLazyJSONValue); rightOK && rightLazy.value.Exists() {
			return jsonContainsApexValue(leftLazy.value, rightLazy.value), nil
		}
	}
	if leftRaw, leftOK := jsonDocumentRaw(left); leftOK {
		if rightRaw, rightOK := jsonDocumentRaw(right); rightOK {
			return jsonContainsRaw(leftRaw, rightRaw)
		}
	}
	lhs, ok := decodeJSONReadValue(left)
	if !ok {
		return false, fmt.Errorf("invalid left JSON value")
	}
	// A stored JSON/JSONB value is already an owned tree. Do not clone it for
	// a read-only containment predicate. Mutation paths intentionally keep the
	// cloning behavior in decodeJSONValue.
	if _, structured := right.(map[string]interface{}); structured {
		return jsonContainsValue(lhs, right), nil
	}
	if _, structured := right.([]interface{}); structured {
		return jsonContainsValue(lhs, right), nil
	}
	rhs, ok := decodeJSONReadValue(right)
	if !ok {
		return false, fmt.Errorf("invalid right JSON value")
	}
	return jsonContainsValue(lhs, rhs), nil
}

func jsonContainsRaw(leftRaw, rightRaw []byte) (bool, error) {
	leftDecoder, left, err := parseSQLLazyJSON(leftRaw)
	if err != nil {
		return false, fmt.Errorf("invalid left JSON value: %w", err)
	}
	defer putSQLJSONDecoder(leftDecoder)
	rightDecoder, right, err := parseSQLLazyJSON(rightRaw)
	if err != nil {
		return false, fmt.Errorf("invalid right JSON value: %w", err)
	}
	defer putSQLJSONDecoder(rightDecoder)
	return jsonContainsApexValue(left, right), nil
}

func jsonApexObjectField(object apexjson.Value, key string) (apexjson.Value, bool) {
	it := object.ObjectIter()
	for it.Next() {
		if it.Key() == key {
			return it.Value(), true
		}
	}
	return apexjson.Value{}, false
}

func jsonContainsApexValue(left, right apexjson.Value) bool {
	if !left.Exists() || !right.Exists() {
		return false
	}
	if right.Type() == apexjson.TypeObject {
		if left.Type() != apexjson.TypeObject {
			return false
		}
		it := right.ObjectIter()
		for it.Next() {
			key := it.Key()
			wanted := it.Value()
			candidate, exists := jsonApexObjectField(left, key)
			if !exists || !jsonContainsApexValue(candidate, wanted) {
				return false
			}
		}
		return true
	}
	if right.Type() == apexjson.TypeArray {
		if left.Type() != apexjson.TypeArray {
			return false
		}
		for wantedIter := right.ArrayIter(); wantedIter.Next(); {
			wanted := wantedIter.Value()
			found := false
			for candidateIter := left.ArrayIter(); candidateIter.Next(); {
				if jsonContainsApexValue(candidateIter.Value(), wanted) {
					found = true
					break
				}
			}
			if !found {
				return false
			}
		}
		return true
	}
	if left.Type() != right.Type() {
		return false
	}
	switch right.Type() {
	case apexjson.TypeNull:
		return true
	case apexjson.TypeBool:
		return left.Bool() == right.Bool()
	case apexjson.TypeString:
		return left.Str() == right.Str()
	case apexjson.TypeNumber:
		leftRaw, rightRaw := left.Bytes(), right.Bytes()
		if bytes.Equal(leftRaw, rightRaw) {
			return true
		}
		return left.Float() == right.Float()
	default:
		return false
	}
}

func jsonContainsValue(left, right interface{}) bool {
	// SQL NULL is not a JSON value and never satisfies containment. JSON null
	// is represented by util.JSONNull and reaches the scalar comparison path.
	if left == nil || right == nil {
		return left == nil && right == nil
	}
	switch expected := right.(type) {
	case map[string]interface{}:
		actual, ok := left.(map[string]interface{})
		if !ok {
			return false
		}
		for key, value := range expected {
			candidate, exists := actual[key]
			if !exists || !jsonContainsValue(candidate, value) {
				return false
			}
		}
		return true
	case []interface{}:
		actual, ok := left.([]interface{})
		if !ok {
			return false
		}
		for _, wanted := range expected {
			found := false
			for _, candidate := range actual {
				if jsonContainsValue(candidate, wanted) {
					found = true
					break
				}
			}
			if !found {
				return false
			}
		}
		return true
	default:
		return jsonScalarEqual(left, right)
	}
}

func jsonScalarEqual(left, right interface{}) bool {
	leftCanonical, leftOK := canonicalJSONNode(left)
	rightCanonical, rightOK := canonicalJSONNode(right)
	if !leftOK || !rightOK {
		return strings.TrimSpace(fmt.Sprint(left)) == strings.TrimSpace(fmt.Sprint(right))
	}
	leftJSON, leftErr := encodeJSONValue(leftCanonical)
	rightJSON, rightErr := encodeJSONValue(rightCanonical)
	if leftErr != nil || rightErr != nil {
		return strings.TrimSpace(fmt.Sprint(left)) == strings.TrimSpace(fmt.Sprint(right))
	}
	return leftJSON == rightJSON
}

func evaluateJSONBinary(operator uint8, left, right interface{}) (interface{}, bool, error) {
	// PostgreSQL propagates SQL NULL through JSON operators. JSON literal null
	// is the non-nil util.JSONNull sentinel and therefore remains queryable.
	if left == nil || right == nil {
		return nil, true, nil
	}
	if lexer.Kind(operator) == lexer.KindJSONDelete {
		value, err := jsonbDelete(left, right)
		return value, true, err
	}
	if lexer.Kind(operator) == lexer.KindConcat {
		// JSONB concatenation is selected when either operand is a JSON tree.
		// Plain text concatenation remains ordinary SQL concatenation.
		if _, leftJSON := decodeJSONValue(left); leftJSON {
			if _, rightJSON := decodeJSONValue(right); rightJSON {
				value, err := jsonbConcat(left, right)
				return value, true, err
			}
		}
		return recordMetaToString(left) + recordMetaToString(right), true, nil
	}
	if isJSONPathPredicateOperator(operator) {
		// @@ is shared with PostgreSQL full-text search.  A JSONPath literal is
		// unambiguously identified by its '$' root; all other @@ operands keep
		// their existing FTS semantics and are evaluated by the FTS path.
		path, ok := jsonKeyValue(right)
		if !ok {
			return nil, false, nil
		}
		trimmedPath := strings.TrimSpace(path)
		if strings.HasPrefix(strings.ToLower(trimmedPath), "strict ") || strings.HasPrefix(strings.ToLower(trimmedPath), "lax ") {
			trimmedPath = strings.TrimSpace(trimmedPath[4:])
		}
		if !strings.HasPrefix(trimmedPath, "$") {
			return nil, false, nil
		}
		matched, err := jsonPathPredicate(left, path, lexer.Kind(operator) == lexer.KindJSONPathExists)
		return matched, true, err
	}
	if isJSONExtractionOperator(operator) {
		if lexer.Kind(operator) == lexer.KindJSONPath || lexer.Kind(operator) == lexer.KindJSONPathText {
			return jsonPath(left, right, isJSONTextExtractionOperator(operator))
		}
		return jsonExtract(left, right, isJSONTextExtractionOperator(operator))
	}
	if isJSONKeyExistenceOperator(operator) {
		exists, err := jsonExists(left, right)
		return exists, true, err
	}
	if isJSONKeySetOperator(operator) {
		all := lexer.Kind(operator) == lexer.KindJSONAll
		exists, err := jsonAnyOrAll(left, right, all)
		return exists, true, err
	}
	contains, err := jsonContains(left, right)
	if err != nil {
		return nil, false, err
	}
	if lexer.Kind(operator) == lexer.KindJSONContainedBy {
		// `<@` is the inverse containment relation.
		contains, err = jsonContains(right, left)
	}
	return contains, true, err
}

// jsonPathPredicate implements the JSONPath predicate subset used by the SQL
// executor.  It intentionally evaluates against the decoded JSON tree and
// never mutates it.  Supported forms are the PostgreSQL-compatible root/member
// and array selectors ($, .key, ['key'], [n], [*]), optional filter predicates
// (`? (@ <op> literal)`), and scalar comparisons (`$.score > 0.5`).  Boolean
// literals and &&/|| combinations are accepted inside filters.  Unsupported
// path syntax is rejected with an explicit error rather than silently widening
// a result set.
func jsonPathPredicate(document interface{}, expression string, existence bool) (bool, error) {
	root, ok := decodeJSONValue(document)
	if !ok {
		return false, fmt.Errorf("invalid JSON document")
	}
	expression = strings.TrimSpace(expression)
	strict := false
	switch {
	case strings.HasPrefix(strings.ToLower(expression), "strict "):
		strict = true
		expression = strings.TrimSpace(expression[len("strict "):])
	case strings.HasPrefix(strings.ToLower(expression), "lax "):
		expression = strings.TrimSpace(expression[len("lax "):])
	}
	if expression == "" || expression[0] != '$' {
		return false, fmt.Errorf("JSONPath must start with '$'")
	}
	pathExpr, filterExpr := splitJSONPathFilter(expression)
	values, err := jsonPathQueryMode(root, pathExpr, strict)
	if err != nil {
		return false, err
	}
	if filterExpr != "" {
		filtered := values[:0]
		for _, value := range values {
			matched, err := evalJSONPathFilter(value, filterExpr)
			if err != nil {
				return false, err
			}
			if matched {
				filtered = append(filtered, value)
			}
		}
		values = filtered
	}
	if existence {
		return len(values) != 0, nil
	}
	if len(values) == 0 {
		return false, nil
	}
	if boolean, ok := values[0].(bool); ok {
		return boolean, nil
	}
	return true, nil
}

func splitJSONPathFilter(expression string) (string, string) {
	depth := 0
	quote := byte(0)
	for i := 0; i < len(expression); i++ {
		ch := expression[i]
		if quote != 0 {
			if ch == quote && (i == 0 || expression[i-1] != '\\') {
				quote = 0
			}
			continue
		}
		switch ch {
		case '\'', '"':
			quote = ch
		case '[', '(':
			depth++
		case ']', ')':
			if depth > 0 {
				depth--
			}
		case '?':
			if depth == 0 {
				return strings.TrimSpace(expression[:i]), strings.TrimSpace(expression[i+1:])
			}
		}
	}
	// PostgreSQL also permits a boolean JSONPath expression without an
	// explicit filter block, for example `$.score > 0.5`.
	for _, op := range []string{"==", "!=", ">=", "<=", ">", "<"} {
		if i := strings.Index(expression[1:], op); i >= 0 {
			i++
			return strings.TrimSpace(expression[:i]), "(@ " + op + " " + strings.TrimSpace(expression[i+len(op):]) + ")"
		}
	}
	return strings.TrimSpace(expression), ""
}

func jsonPathQuery(root interface{}, expression string) ([]interface{}, error) {
	return jsonPathQueryMode(root, expression, false)
}

func jsonPathQueryMode(root interface{}, expression string, strict bool) ([]interface{}, error) {
	if expression == "$" {
		return []interface{}{root}, nil
	}
	if len(expression) == 0 || expression[0] != '$' {
		return nil, fmt.Errorf("JSONPath must start with '$'")
	}
	values := []interface{}{root}
	for i := 1; i < len(expression); {
		switch expression[i] {
		case '.':
			i++
			// PostgreSQL JSONPath supports both .**.member and the common
			// $..member spelling for recursive descent.
			recursive := false
			if i < len(expression) && expression[i] == '.' {
				recursive = true
				i++
			} else if i+1 < len(expression) && expression[i] == '*' && expression[i+1] == '*' {
				recursive = true
				i += 2
				if i < len(expression) && expression[i] == '.' {
					i++
				}
			}
			start := i
			for i < len(expression) && (isJSONPathIdent(expression[i]) || expression[i] == '*') {
				i++
			}
			// Function selectors are useful inside filters, e.g.
			// $.items[*] ? (@.score.type() == "number").
			if i < len(expression) && expression[i] == '(' {
				end := strings.IndexByte(expression[i:], ')')
				if end < 0 {
					return nil, fmt.Errorf("unterminated JSONPath function")
				}
				i += end + 1
			}
			if start == i {
				return nil, fmt.Errorf("JSONPath member name expected")
			}
			member := expression[start:i]
			if strings.HasSuffix(member, "()") {
				name := strings.TrimSuffix(member, "()")
				switch strings.ToLower(name) {
				case "type", "size", "length":
				default:
					return nil, fmt.Errorf("unsupported JSONPath function %q", name)
				}
				values = jsonPathFunction(values, name)
			} else if recursive {
				values = jsonPathRecursiveMember(values, member)
			} else {
				values = jsonPathMember(values, member)
			}
			if strict && len(values) == 0 {
				return nil, fmt.Errorf("strict JSONPath step %q matched no value", member)
			}
		case '[':
			end := i + 1
			quote := byte(0)
			for end < len(expression) {
				if quote != 0 {
					if expression[end] == quote && expression[end-1] != '\\' {
						quote = 0
					}
				} else if expression[end] == '\'' || expression[end] == '"' {
					quote = expression[end]
				} else if expression[end] == ']' {
					break
				}
				end++
			}
			if end >= len(expression) {
				return nil, fmt.Errorf("unterminated JSONPath bracket")
			}
			selector := strings.TrimSpace(expression[i+1 : end])
			var err error
			values, err = jsonPathBracket(values, selector)
			if err != nil {
				if strict {
					return nil, err
				}
				values = nil
			}
			if strict && len(values) == 0 {
				return nil, fmt.Errorf("strict JSONPath selector %q matched no value", selector)
			}
			i = end + 1
		default:
			return nil, fmt.Errorf("unsupported JSONPath syntax near %q", expression[i:])
		}
	}
	return values, nil
}

func isJSONPathIdent(ch byte) bool {
	return ch >= 'a' && ch <= 'z' || ch >= 'A' && ch <= 'Z' || ch >= '0' && ch <= '9' || ch == '_' || ch == '-'
}

func jsonPathMember(values []interface{}, member string) []interface{} {
	out := make([]interface{}, 0, len(values))
	for _, value := range values {
		object, ok := value.(map[string]interface{})
		if !ok {
			continue
		}
		if member == "*" {
			for _, child := range object {
				out = append(out, child)
			}
			continue
		}
		if child, ok := object[member]; ok {
			out = append(out, child)
		}
	}
	return out
}

func jsonPathRecursiveMember(values []interface{}, member string) []interface{} {
	out := make([]interface{}, 0)
	var visit func(interface{})
	visit = func(value interface{}) {
		switch container := value.(type) {
		case map[string]interface{}:
			for key, child := range container {
				if member == "*" || key == member {
					out = append(out, child)
				}
				visit(child)
			}
		case []interface{}:
			for _, child := range container {
				if member == "*" {
					out = append(out, child)
				}
				visit(child)
			}
		}
	}
	for _, value := range values {
		visit(value)
	}
	return out
}

func jsonPathFunction(values []interface{}, name string) []interface{} {
	out := make([]interface{}, 0, len(values))
	for _, value := range values {
		switch strings.ToLower(name) {
		case "type":
			typeName, ok := jsonbTypeof(value)
			if ok {
				out = append(out, typeName)
			}
		case "size", "length":
			switch container := value.(type) {
			case []interface{}:
				out = append(out, int64(len(container)))
			case map[string]interface{}:
				out = append(out, int64(len(container)))
			}
		default:
			// Unsupported JSONPath functions are deliberately rejected by the
			// caller rather than treated as a member name.
		}
	}
	return out
}

func jsonPathBracket(values []interface{}, selector string) ([]interface{}, error) {
	if selector == "*" {
		out := make([]interface{}, 0)
		for _, value := range values {
			switch child := value.(type) {
			case []interface{}:
				out = append(out, child...)
			case map[string]interface{}:
				for _, item := range child {
					out = append(out, item)
				}
			}
		}
		return out, nil
	}
	if len(selector) >= 2 && (selector[0] == '\'' || selector[0] == '"') && selector[len(selector)-1] == selector[0] {
		key := selector[1 : len(selector)-1]
		return jsonPathMember(values, key), nil
	}
	if to := strings.Index(strings.ToLower(selector), " to "); to > 0 {
		start, startErr := strconv.Atoi(strings.TrimSpace(selector[:to]))
		end, endErr := strconv.Atoi(strings.TrimSpace(selector[to+4:]))
		if startErr != nil || endErr != nil {
			return nil, fmt.Errorf("JSONPath range bounds must be integers")
		}
		out := make([]interface{}, 0)
		for _, value := range values {
			array, ok := value.([]interface{})
			if !ok {
				continue
			}
			s, e := start, end
			if s < 0 {
				s = len(array) + s
			}
			if e < 0 {
				e = len(array) + e
			}
			if s < 0 {
				s = 0
			}
			if e >= len(array) {
				e = len(array) - 1
			}
			if s <= e && s < len(array) {
				out = append(out, array[s:e+1]...)
			}
		}
		return out, nil
	}
	index, err := strconv.Atoi(selector)
	if err != nil {
		return nil, fmt.Errorf("JSONPath array index must be an integer")
	}
	out := make([]interface{}, 0, len(values))
	for _, value := range values {
		array, ok := value.([]interface{})
		if index < 0 && ok {
			index = len(array) + index
		}
		if ok && index >= 0 && index < len(array) {
			out = append(out, array[index])
		}
	}
	return out, nil
}

func evalJSONPathFilter(value interface{}, filter string) (bool, error) {
	filter = strings.TrimSpace(filter)
	if strings.HasPrefix(filter, "(") && strings.HasSuffix(filter, ")") {
		filter = strings.TrimSpace(filter[1 : len(filter)-1])
	}
	if idx := strings.Index(filter, "||"); idx >= 0 {
		left, err := evalJSONPathFilter(value, filter[:idx])
		if err != nil {
			return false, err
		}
		right, err := evalJSONPathFilter(value, filter[idx+2:])
		return left || right, err
	}
	if idx := strings.Index(filter, "&&"); idx >= 0 {
		left, err := evalJSONPathFilter(value, filter[:idx])
		if err != nil {
			return false, err
		}
		right, err := evalJSONPathFilter(value, filter[idx+2:])
		return left && right, err
	}
	if strings.HasPrefix(strings.ToLower(filter), "exists(") && strings.HasSuffix(filter, ")") {
		argument := strings.TrimSpace(filter[len("exists(") : len(filter)-1])
		if !strings.HasPrefix(argument, "@") {
			return false, fmt.Errorf("JSONPath exists() expects an @ path")
		}
		pathValues, err := jsonPathQuery(value, "$"+argument[1:])
		return len(pathValues) > 0, err
	}
	if strings.HasPrefix(filter, "@.") && !strings.ContainsAny(filter, "=<>!") {
		pathValues, err := jsonPathQuery(value, "$"+filter[1:])
		if err != nil {
			return false, err
		}
		return len(pathValues) > 0, nil
	}
	for _, op := range []string{"==", "!=", ">=", "<=", ">", "<"} {
		if idx := strings.Index(filter, op); idx >= 0 {
			left := strings.TrimSpace(filter[:idx])
			right := strings.TrimSpace(filter[idx+len(op):])
			var actual = value
			if strings.HasPrefix(left, "@.") {
				pathValues, pathErr := jsonPathQuery(value, "$"+left[1:])
				if pathErr != nil || len(pathValues) == 0 {
					return false, pathErr
				}
				actual = pathValues[0]
			} else if left != "@" {
				return false, fmt.Errorf("JSONPath filter left operand must be '@' or '@.field'")
			}
			wanted, ok := parseJSONPathLiteral(right)
			if !ok {
				return false, fmt.Errorf("invalid JSONPath filter literal")
			}
			cmp := compareJSONPathValues(actual, wanted)
			switch op {
			case "==":
				return cmp == 0, nil
			case "!=":
				return cmp != 0, nil
			case ">":
				return cmp > 0, nil
			case "<":
				return cmp < 0, nil
			case ">=":
				return cmp >= 0, nil
			case "<=":
				return cmp <= 0, nil
			}
		}
	}
	if filter == "@" {
		return value != nil, nil
	}
	return false, fmt.Errorf("unsupported JSONPath filter %q", filter)
}

func parseJSONPathLiteral(text string) (interface{}, bool) {
	text = strings.TrimSpace(text)
	if strings.EqualFold(text, "true") {
		return true, true
	}
	if strings.EqualFold(text, "false") {
		return false, true
	}
	if strings.EqualFold(text, "null") {
		return util.JSONNull{}, true
	}
	if len(text) >= 2 && ((text[0] == '\'' && text[len(text)-1] == '\'') || (text[0] == '"' && text[len(text)-1] == '"')) {
		if text[0] == '\'' {
			return strings.ReplaceAll(text[1:len(text)-1], "''", "'"), true
		}
		return strings.ReplaceAll(text[1:len(text)-1], `\"`, `"`), true
	}
	if n, err := strconv.ParseFloat(text, 64); err == nil {
		return n, true
	}
	return nil, false
}

func compareJSONPathValues(left, right interface{}) int {
	if jsonScalarEqual(left, right) {
		return 0
	}
	lf, lok := jsonPathNumber(left)
	rf, rok := jsonPathNumber(right)
	if lok && rok {
		if lf < rf {
			return -1
		}
		if lf > rf {
			return 1
		}
	}
	ls, lok := left.(string)
	rs, rok := right.(string)
	if lok && rok {
		if ls < rs {
			return -1
		}
		if ls > rs {
			return 1
		}
	}
	return 1
}

func jsonPathNumber(value interface{}) (float64, bool) {
	switch v := value.(type) {
	case int:
		return float64(v), true
	case int64:
		return float64(v), true
	case uint64:
		return float64(v), true
	case float64:
		return v, true
	case float32:
		return float64(v), true
	default:
		return 0, false
	}
}
