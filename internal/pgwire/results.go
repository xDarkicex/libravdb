package pgwire

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
	"time"

	"github.com/xDarkicex/libravdb/internal/util"
	"github.com/xDarkicex/libravdb/libravdb"
)

// ColumnMeta describes a result set column for RowDescription encoding.
type ColumnMeta struct {
	Name    string
	TypeOID uint32
}

// sendRowDescription sends a RowDescription ('T') message describing the result columns.
// Each column uses the appropriate PostgreSQL type OID instead of hardcoded TEXT.
func sendRowDescription(w io.Writer, columns []ColumnMeta) error {
	return sendRowDescriptionWithFormats(w, columns, nil)
}

// sendRowDescriptionWithFormats emits the negotiated result format for every
// column. An empty format list means text for all columns; one code applies to
// every column; otherwise one code is required per column.
func sendRowDescriptionWithFormats(w io.Writer, columns []ColumnMeta, formats []int16) error {
	if len(columns) == 0 {
		return WriteMessage(w, msgNoData, nil)
	}
	if len(formats) != 0 && len(formats) != 1 && len(formats) != len(columns) {
		return fmt.Errorf("result format count %d does not match %d columns", len(formats), len(columns))
	}

	size := 2
	for _, col := range columns {
		size += len(col.Name) + 1 + 4 + 2 + 4 + 2 + 4 + 2 // null-terminated name + 18 bytes metadata
	}
	buf := make([]byte, 0, size)

	buf = append(buf, 0, 0)
	binary.BigEndian.PutUint16(buf[len(buf)-2:], uint16(len(columns)))

	for i, col := range columns {
		buf = append(buf, col.Name...)
		buf = append(buf, 0)

		// Table OID (0)
		buf = append(buf, 0, 0, 0, 0)
		// Column attribute number (0)
		buf = append(buf, 0, 0)
		// Data type OID
		binary.BigEndian.PutUint32(buf[len(buf):len(buf)+4], col.TypeOID)
		buf = buf[:len(buf)+4]
		// Data type size (-1 = variable)
		binary.BigEndian.PutUint16(buf[len(buf):len(buf)+2], uint16(pgTypeSize(col.TypeOID)))
		buf = buf[:len(buf)+2]
		// Type modifier (-1)
		binary.BigEndian.PutUint32(buf[len(buf):len(buf)+4], 0xFFFFFFFF)
		buf = buf[:len(buf)+4]
		format := resultFormatAt(formats, i)
		buf = append(buf, byte(format>>8), byte(format))
	}

	return WriteMessage(w, msgRowDescription, buf)
}

// sendDataRow sends a DataRow ('D') message with the given column values as text.
// Each element is a *string: nil means SQL NULL (encoded as length -1),
// a pointer to "" is an empty string (encoded as length 0), and a pointer
// to a non-empty string is encoded with its length and bytes.
func sendDataRow(w io.Writer, values []*string) error {
	// numColumns (int16) + per-column: len (int32) + value bytes
	// NULL columns contribute only the 4-byte length (-1).
	size := 2
	for _, v := range values {
		size += 4
		if v != nil {
			size += len(*v)
		}
	}
	buf := make([]byte, 0, size)

	// Column count (int16).
	off := len(buf)
	buf = buf[:off+2]
	binary.BigEndian.PutUint16(buf[off:], uint16(len(values)))

	for _, v := range values {
		off = len(buf)
		buf = buf[:off+4]
		if v == nil {
			// SQL NULL: length -1 (0xFFFFFFFF as uint32), no data bytes follow.
			const nullLen = uint32(0xFFFFFFFF)
			binary.BigEndian.PutUint32(buf[off:], nullLen)
		} else {
			// Non-NULL: length N followed by N data bytes.
			binary.BigEndian.PutUint32(buf[off:], uint32(len(*v)))
			buf = append(buf, *v...)
		}
	}

	return WriteMessage(w, msgDataRow, buf)
}

// sendDataRowWithFormats encodes each result using the negotiated text or
// binary format. NULL is represented by a -1 length in either format.
func sendDataRowWithFormats(w io.Writer, result *libravdb.SearchResult, columns []ColumnMeta, formats []int16) error {
	values := make([][]byte, len(columns))
	nulls := make([]bool, len(columns))
	size := 2
	for i, col := range columns {
		value, present := resultValue(result, col.Name)
		if !present || value == nil {
			nulls[i] = true
			size += 4
			continue
		}
		encoded, err := encodeResultValue(value, col.TypeOID, resultFormatAt(formats, i))
		if err != nil {
			return fmt.Errorf("encode column %q: %w", col.Name, err)
		}
		values[i] = encoded
		size += 4 + len(encoded)
	}

	buf := make([]byte, 2, size)
	binary.BigEndian.PutUint16(buf, uint16(len(columns)))
	for i, value := range values {
		if nulls[i] {
			buf = append(buf, 0xff, 0xff, 0xff, 0xff)
			continue
		}
		var length [4]byte
		binary.BigEndian.PutUint32(length[:], uint32(len(value)))
		buf = append(buf, length[:]...)
		buf = append(buf, value...)
	}
	return WriteMessage(w, msgDataRow, buf)
}

func resultFormatAt(formats []int16, index int) int16 {
	if len(formats) == 0 {
		return 0
	}
	if len(formats) == 1 {
		return formats[0]
	}
	if index < len(formats) {
		return formats[index]
	}
	return 0
}

func resultValue(result *libravdb.SearchResult, name string) (interface{}, bool) {
	if result == nil {
		return nil, false
	}
	switch name {
	case "id", "ID":
		return result.ID, true
	case "score", "SCORE":
		// A collection may legitimately declare metadata named `score`.
		// Prefer that relational column when it is present; SearchResult.Score
		// is only the fallback relevance score for projections that do not
		// expose a stored score field.
		if result.Metadata != nil {
			if value, ok := result.Metadata[name]; ok {
				return value, true
			}
			if value, ok := result.Metadata["score"]; ok {
				return value, true
			}
		}
		return result.Score, true
	case "version", "VERSION":
		if result.Metadata != nil {
			if value, ok := result.Metadata[name]; ok {
				return value, true
			}
			if value, ok := result.Metadata["version"]; ok {
				return value, true
			}
		}
		return result.Version, true
	case "ordinal", "ORDINAL":
		return result.Ordinal, true
	default:
		if result.Metadata == nil {
			return nil, false
		}
		value, ok := result.Metadata[name]
		return value, ok
	}
}

func encodeResultValue(value interface{}, oid uint32, format int16) ([]byte, error) {
	if format == 0 {
		if oid == OIDInt2Array || oid == OIDTextArray || oid == OIDInt4Array || oid == OIDInt8Array || oid == OIDFloat4Array || oid == OIDFloat8Array || oid == OIDBoolArray || oid == OIDOIDArray {
			return []byte(encodeTextArray(value)), nil
		}
		return []byte(metadataValueToString(value)), nil
	}
	if format != 1 {
		return nil, fmt.Errorf("unsupported result format %d", format)
	}
	switch oid {
	case OIDBool:
		b, err := resultBool(value)
		if err != nil {
			return nil, err
		}
		if b {
			return []byte{1}, nil
		}
		return []byte{0}, nil
	case OIDInt2:
		n, err := resultInt64(value)
		if err != nil || n < math.MinInt16 || n > math.MaxInt16 {
			return nil, fmt.Errorf("value is not a valid int2")
		}
		var out [2]byte
		binary.BigEndian.PutUint16(out[:], uint16(int16(n)))
		return out[:], nil
	case OIDInt4:
		n, err := resultInt64(value)
		if err != nil || n < math.MinInt32 || n > math.MaxInt32 {
			return nil, fmt.Errorf("value is not a valid int4")
		}
		var out [4]byte
		binary.BigEndian.PutUint32(out[:], uint32(int32(n)))
		return out[:], nil
	case OIDInt8, OIDOID:
		n, err := resultInt64(value)
		if err != nil {
			return nil, fmt.Errorf("value is not a valid int8")
		}
		if oid == OIDOID && (n < 0 || n > math.MaxUint32) {
			return nil, fmt.Errorf("value is not a valid oid")
		}
		if oid == OIDOID {
			var out [4]byte
			binary.BigEndian.PutUint32(out[:], uint32(n))
			return out[:], nil
		}
		var out [8]byte
		binary.BigEndian.PutUint64(out[:], uint64(n))
		return out[:], nil
	case OIDFloat4:
		f, err := resultFloat64(value)
		if err != nil {
			return nil, err
		}
		var out [4]byte
		binary.BigEndian.PutUint32(out[:], math.Float32bits(float32(f)))
		return out[:], nil
	case OIDFloat8:
		f, err := resultFloat64(value)
		if err != nil {
			return nil, err
		}
		var out [8]byte
		binary.BigEndian.PutUint64(out[:], math.Float64bits(f))
		return out[:], nil
	case OIDFloat4Array, OIDFloat8Array:
		return encodeFloatArray(value, oid)
	case OIDInt2Array, OIDTextArray, OIDInt4Array, OIDInt8Array, OIDBoolArray, OIDOIDArray:
		return encodeBinaryArray(value, oid)
	case OIDTimestamp, OIDTimestamptz:
		return encodeTimestamp(value)
	case OIDDate:
		return encodeDate(value)
	case OIDJSON, OIDJSONB:
		text := metadataValueToString(value)
		if oid == OIDJSONB {
			return append([]byte{1}, []byte(text)...), nil
		}
		return []byte(text), nil
	case OIDUUID:
		text := metadataValueToString(value)
		if len(text) != 36 {
			return nil, fmt.Errorf("value is not a valid uuid")
		}
		raw := make([]byte, 0, 16)
		for i := 0; i < len(text); i++ {
			if text[i] == '-' {
				continue
			}
			if i+1 >= len(text) {
				return nil, fmt.Errorf("value is not a valid uuid")
			}
			hi, ok := hexNibble(text[i])
			lo, ok2 := hexNibble(text[i+1])
			if !ok || !ok2 {
				return nil, fmt.Errorf("value is not a valid uuid")
			}
			raw = append(raw, hi<<4|lo)
			i++
		}
		if len(raw) != 16 {
			return nil, fmt.Errorf("value is not a valid uuid")
		}
		return raw, nil
	default:
		return []byte(metadataValueToString(value)), nil
	}
}

func hexNibble(c byte) (byte, bool) {
	switch {
	case c >= '0' && c <= '9':
		return c - '0', true
	case c >= 'a' && c <= 'f':
		return c - 'a' + 10, true
	case c >= 'A' && c <= 'F':
		return c - 'A' + 10, true
	default:
		return 0, false
	}
}

func resultInt64(value interface{}) (int64, error) {
	switch v := value.(type) {
	case int:
		return int64(v), nil
	case int8:
		return int64(v), nil
	case int16:
		return int64(v), nil
	case int32:
		return int64(v), nil
	case int64:
		return v, nil
	case uint:
		if uint64(v) > math.MaxInt64 {
			return 0, fmt.Errorf("integer overflow")
		}
		return int64(v), nil
	case uint8:
		return int64(v), nil
	case uint16:
		return int64(v), nil
	case uint32:
		return int64(v), nil
	case uint64:
		if v > math.MaxInt64 {
			return 0, fmt.Errorf("integer overflow")
		}
		return int64(v), nil
	case float64:
		return int64(v), nil
	case float32:
		return int64(v), nil
	case string:
		return strconv.ParseInt(v, 10, 64)
	case []byte:
		return strconv.ParseInt(string(v), 10, 64)
	default:
		return strconv.ParseInt(metadataValueToString(value), 10, 64)
	}
}

func resultFloat64(value interface{}) (float64, error) {
	switch v := value.(type) {
	case float64:
		return v, nil
	case float32:
		return float64(v), nil
	case int:
		return float64(v), nil
	case int64:
		return float64(v), nil
	case uint64:
		return float64(v), nil
	case string:
		return strconv.ParseFloat(v, 64)
	case []byte:
		return strconv.ParseFloat(string(v), 64)
	default:
		return strconv.ParseFloat(metadataValueToString(value), 64)
	}
}

func resultBool(value interface{}) (bool, error) {
	switch v := value.(type) {
	case bool:
		return v, nil
	case string:
		return strconv.ParseBool(v)
	case []byte:
		return strconv.ParseBool(string(v))
	default:
		return strconv.ParseBool(metadataValueToString(value))
	}
}

func encodeFloatArray(value interface{}, oid uint32) ([]byte, error) {
	values := make([]float64, 0)
	switch v := value.(type) {
	case []float32:
		values = make([]float64, len(v))
		for i := range v {
			values[i] = float64(v[i])
		}
	case []float64:
		values = append(values, v...)
	case string:
		parsed := parseVectorParam(v)
		if parsed == nil {
			return nil, fmt.Errorf("invalid float array")
		}
		values = make([]float64, len(parsed))
		for i := range parsed {
			values[i] = float64(parsed[i])
		}
	default:
		return nil, fmt.Errorf("value is not a float array")
	}
	elemOID := uint32(OIDFloat4)
	elemWidth := 4
	if oid == OIDFloat8Array {
		elemOID = OIDFloat8
		elemWidth = 8
	}
	buf := make([]byte, 0, 20+len(values)*(8+elemWidth))
	var header [20]byte
	binary.BigEndian.PutUint32(header[0:4], 1) // ndim
	binary.BigEndian.PutUint32(header[4:8], 0) // no NULL elements
	binary.BigEndian.PutUint32(header[8:12], elemOID)
	binary.BigEndian.PutUint32(header[12:16], uint32(len(values)))
	binary.BigEndian.PutUint32(header[16:20], 1) // lower bound
	buf = append(buf, header[:]...)
	for _, value := range values {
		if elemWidth == 4 {
			var raw [4]byte
			binary.BigEndian.PutUint32(raw[:], math.Float32bits(float32(value)))
			var n [4]byte
			binary.BigEndian.PutUint32(n[:], 4)
			buf = append(buf, n[:]...)
			buf = append(buf, raw[:]...)
		} else {
			var raw [8]byte
			binary.BigEndian.PutUint64(raw[:], math.Float64bits(value))
			var n [4]byte
			binary.BigEndian.PutUint32(n[:], 8)
			buf = append(buf, n[:]...)
			buf = append(buf, raw[:]...)
		}
	}
	return buf, nil
}

func encodeBinaryArray(value interface{}, oid uint32) ([]byte, error) {
	items := make([]interface{}, 0)
	switch v := value.(type) {
	case []interface{}:
		items = v
	case []string:
		for _, item := range v {
			items = append(items, item)
		}
	case []int:
		for _, item := range v {
			items = append(items, item)
		}
	case []int32:
		for _, item := range v {
			items = append(items, item)
		}
	case []int64:
		for _, item := range v {
			items = append(items, item)
		}
	case []bool:
		for _, item := range v {
			items = append(items, item)
		}
	default:
		return nil, fmt.Errorf("value is not a supported array")
	}
	var elemOID uint32 = OIDText
	switch oid {
	case OIDInt2Array:
		elemOID = OIDInt2
	case OIDInt4Array:
		elemOID = OIDInt4
	case OIDInt8Array:
		elemOID = OIDInt8
	case OIDBoolArray:
		elemOID = OIDBool
	case OIDOIDArray:
		elemOID = OIDOID
	}
	buf := make([]byte, 0, 20+len(items)*16)
	var header [20]byte
	binary.BigEndian.PutUint32(header[0:4], 1)
	containsNull := uint32(0)
	for _, item := range items {
		if item == nil {
			containsNull = 1
			break
		}
	}
	binary.BigEndian.PutUint32(header[4:8], containsNull)
	binary.BigEndian.PutUint32(header[8:12], elemOID)
	binary.BigEndian.PutUint32(header[12:16], uint32(len(items)))
	binary.BigEndian.PutUint32(header[16:20], 1)
	buf = append(buf, header[:]...)
	for _, item := range items {
		if item == nil {
			buf = append(buf, 0xff, 0xff, 0xff, 0xff)
			continue
		}
		var raw []byte
		var err error
		switch elemOID {
		case OIDInt2:
			n, e := resultInt64(item)
			err = e
			if err == nil {
				var b [2]byte
				binary.BigEndian.PutUint16(b[:], uint16(int16(n)))
				raw = b[:]
			}
		case OIDInt4:
			n, e := resultInt64(item)
			err = e
			if err == nil {
				var b [4]byte
				binary.BigEndian.PutUint32(b[:], uint32(int32(n)))
				raw = b[:]
			}
		case OIDInt8:
			n, e := resultInt64(item)
			err = e
			if err == nil {
				var b [8]byte
				binary.BigEndian.PutUint64(b[:], uint64(n))
				raw = b[:]
			}
		case OIDBool:
			v, e := resultBool(item)
			err = e
			if err == nil {
				if v {
					raw = []byte{1}
				} else {
					raw = []byte{0}
				}
			}
		default:
			raw = []byte(metadataValueToString(item))
		}
		if err != nil {
			return nil, err
		}
		var n [4]byte
		binary.BigEndian.PutUint32(n[:], uint32(len(raw)))
		buf = append(buf, n[:]...)
		buf = append(buf, raw...)
	}
	return buf, nil
}

func encodeTimestamp(value interface{}) ([]byte, error) {
	instant, err := resultTime(value)
	if err != nil {
		return nil, err
	}
	micros := instant.UnixMicro() - time.Date(2000, 1, 1, 0, 0, 0, 0, time.UTC).UnixMicro()
	var out [8]byte
	binary.BigEndian.PutUint64(out[:], uint64(micros))
	return out[:], nil
}

func encodeDate(value interface{}) ([]byte, error) {
	instant, err := resultTime(value)
	if err != nil {
		return nil, err
	}
	days := int32(instant.UTC().Sub(time.Date(2000, 1, 1, 0, 0, 0, 0, time.UTC)) / (24 * time.Hour))
	var out [4]byte
	binary.BigEndian.PutUint32(out[:], uint32(days))
	return out[:], nil
}

func resultTime(value interface{}) (time.Time, error) {
	switch v := value.(type) {
	case time.Time:
		return v, nil
	case string:
		for _, layout := range []string{
			time.RFC3339Nano,
			"2006-01-02 15:04:05.999999999Z07:00",
			"2006-01-02 15:04:05.999999999",
			"2006-01-02 15:04:05",
		} {
			if parsed, err := time.Parse(layout, v); err == nil {
				return parsed.UTC(), nil
			}
		}
		return time.Parse("2006-01-02", v)
	case []byte:
		return resultTime(string(v))
	default:
		return time.Time{}, fmt.Errorf("value is not a timestamp/date")
	}
}

// sendCommandComplete sends a CommandComplete ('C') message.
func sendCommandComplete(w io.Writer, tag string) error {
	return WriteMessage(w, msgCommandComplete, append([]byte(tag), 0))
}

// sendReadyForQuery sends ReadyForQuery ('Z') with the given transaction status.
func sendReadyForQuery(w io.Writer, status byte) error {
	return WriteMessage(w, msgReadyForQuery, []byte{status})
}

// sendResults encodes SearchResults into pgwire DataRow messages.
func sendResults(w io.Writer, results *libravdb.SearchResults, columns []ColumnMeta) error {
	return sendResultsWithFormats(w, results, columns, nil)
}

func sendResultsWithFormats(w io.Writer, results *libravdb.SearchResults, columns []ColumnMeta, formats []int16) error {
	if err := sendRowDescriptionWithFormats(w, columns, formats); err != nil {
		return err
	}

	// DataRows
	for _, r := range results.Results {
		if len(formats) == 0 {
			vals := buildResultRow(r, columns)
			if err := sendDataRow(w, vals); err != nil {
				return err
			}
			continue
		}
		if err := sendDataRowWithFormats(w, r, columns, formats); err != nil {
			return err
		}
	}

	return nil
}

// buildResultRow constructs a row of string values from a SearchResult.
// Each element is a *string: nil means SQL NULL. Built-in columns (id, score,
// version, ordinal) are always non-NULL. Metadata columns map nil or missing
// values to SQL NULL, preserving the distinction between NULL and empty string.
func buildResultRow(r *libravdb.SearchResult, columns []ColumnMeta) []*string {
	vals := make([]*string, len(columns))
	for i, col := range columns {
		switch col.Name {
		case "id", "ID":
			s := r.ID
			vals[i] = &s
		case "score", "SCORE":
			value := interface{}(r.Score)
			builtInScore := true
			if r.Metadata != nil {
				if metadataValue, ok := r.Metadata[col.Name]; ok {
					value = metadataValue
					builtInScore = false
				} else if metadataValue, ok := r.Metadata["score"]; ok {
					value = metadataValue
					builtInScore = false
				}
			}
			// Keep the historical text representation for the built-in
			// relevance score. Metadata-backed score columns retain their
			// natural SQL text representation.
			var s string
			if builtInScore {
				s = fmt.Sprintf("%f", r.Score)
			} else {
				s = metadataValueToString(value)
			}
			vals[i] = &s
		case "version", "VERSION":
			value := interface{}(r.Version)
			if r.Metadata != nil {
				if metadataValue, ok := r.Metadata[col.Name]; ok {
					value = metadataValue
				} else if metadataValue, ok := r.Metadata["version"]; ok {
					value = metadataValue
				}
			}
			s := metadataValueToString(value)
			vals[i] = &s
		case "ordinal", "ORDINAL":
			value := interface{}(r.Ordinal)
			if r.Metadata != nil {
				if metadataValue, ok := r.Metadata[col.Name]; ok {
					value = metadataValue
				} else if metadataValue, ok := r.Metadata["ordinal"]; ok {
					value = metadataValue
				}
			}
			s := metadataValueToString(value)
			vals[i] = &s
		default:
			// Projected column: pull from record metadata (SQL SELECT path).
			// Nil or missing metadata values are SQL NULL, not empty string.
			if r.Metadata != nil {
				if v, ok := r.Metadata[col.Name]; ok {
					if v == nil {
						// Explicit nil in metadata → SQL NULL.
						vals[i] = nil
					} else {
						s := metadataValueToText(v, col.TypeOID)
						vals[i] = &s
					}
					continue
				}
			}
			// Key not present or Metadata is nil → SQL NULL.
			vals[i] = nil
		}
	}
	return vals
}

// metadataValueToString renders a metadata value for the pgwire text format.
func metadataValueToString(v interface{}) string {
	switch t := v.(type) {
	case util.JSONNull:
		return "null"
	case string:
		return t
	case []byte:
		return string(t)
	case int:
		return strconv.Itoa(t)
	case int64:
		return strconv.FormatInt(t, 10)
	case uint64:
		return strconv.FormatUint(t, 10)
	case int32:
		return strconv.FormatInt(int64(t), 10)
	case float64:
		return strconv.FormatFloat(t, 'f', -1, 64)
	case float32:
		return strconv.FormatFloat(float64(t), 'f', -1, 32)
	case bool:
		return strconv.FormatBool(t)
	case map[string]interface{}, []interface{}, map[string]string:
		if encoded, err := json.Marshal(jsonWireValue(t)); err == nil {
			return string(encoded)
		}
		return fmt.Sprintf("%v", t)
	default:
		return fmt.Sprintf("%v", t)
	}
}

func metadataValueToText(v interface{}, oid uint32) string {
	if oid == OIDInt2Array || oid == OIDTextArray || oid == OIDInt4Array || oid == OIDInt8Array || oid == OIDFloat4Array || oid == OIDFloat8Array || oid == OIDBoolArray || oid == OIDOIDArray {
		return encodeTextArray(v)
	}
	return metadataValueToString(v)
}

func encodeTextArray(value interface{}) string {
	values := make([]interface{}, 0)
	switch v := value.(type) {
	case []interface{}:
		values = v
	case []string:
		for _, item := range v {
			values = append(values, item)
		}
	case []int:
		for _, item := range v {
			values = append(values, item)
		}
	case []int32:
		for _, item := range v {
			values = append(values, item)
		}
	case []int64:
		for _, item := range v {
			values = append(values, item)
		}
	case []float32:
		for _, item := range v {
			values = append(values, item)
		}
	case []float64:
		for _, item := range v {
			values = append(values, item)
		}
	case []bool:
		for _, item := range v {
			values = append(values, item)
		}
	default:
		return metadataValueToString(value)
	}
	var out strings.Builder
	out.WriteByte('{')
	for i, item := range values {
		if i > 0 {
			out.WriteByte(',')
		}
		if item == nil {
			out.WriteString("NULL")
			continue
		}
		text := metadataValueToString(item)
		needsQuote := text == "" || strings.ContainsAny(text, `{},"\\ `)
		if needsQuote {
			out.WriteByte('"')
			for _, ch := range text {
				if ch == '"' || ch == '\\' {
					out.WriteByte('\\')
				}
				out.WriteRune(ch)
			}
			out.WriteByte('"')
		} else {
			out.WriteString(text)
		}
	}
	out.WriteByte('}')
	return out.String()
}

func jsonWireValue(value interface{}) interface{} {
	switch v := value.(type) {
	case util.JSONNull:
		return nil
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
