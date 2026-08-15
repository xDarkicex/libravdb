package pgwire

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"testing"

	"github.com/xDarkicex/libravdb/libravdb"
)

// =============================================================================
// Unit: sendDataRow NULL encoding
// =============================================================================

func TestSendDataRow_NullEncodedAsLengthMinusOne(t *testing.T) {
	var buf bytes.Buffer

	str := "hello"
	values := []*string{nil, &str}
	if err := sendDataRow(&buf, values); err != nil {
		t.Fatalf("sendDataRow: %v", err)
	}

	// Decode the DataRow message.
	msgType, payload, err := ReadMessage(&buf)
	if err != nil {
		t.Fatalf("ReadMessage: %v", err)
	}
	if msgType != msgDataRow {
		t.Fatalf("expected DataRow ('D'), got '%c'", msgType)
	}

	// Decode raw wire format.
	nulls, decoded, err := decodeDataRowNullable(payload)
	if err != nil {
		t.Fatalf("decodeDataRowNullable: %v", err)
	}

	if len(nulls) != 2 {
		t.Fatalf("expected 2 columns, got %d", len(nulls))
	}

	// Column 0: NULL.
	if !nulls[0] {
		t.Error("column 0: expected NULL (length -1)")
	}
	// Column 1: non-NULL "hello".
	if nulls[1] {
		t.Error("column 1: expected non-NULL")
	}
	if decoded[1] == nil || *decoded[1] != "hello" {
		t.Errorf("column 1: want %q, got %v", "hello", decoded[1])
	}
}

func TestSendDataRow_EmptyStringDistinctFromNull(t *testing.T) {
	var buf bytes.Buffer

	emptyStr := ""
	str := "data"
	// nil = NULL, pointer to "" = empty string — must be distinguishable.
	values := []*string{nil, &emptyStr, &str}
	if err := sendDataRow(&buf, values); err != nil {
		t.Fatalf("sendDataRow: %v", err)
	}

	msgType, payload, err := ReadMessage(&buf)
	if err != nil {
		t.Fatalf("ReadMessage: %v", err)
	}
	if msgType != msgDataRow {
		t.Fatalf("expected 'D', got '%c'", msgType)
	}

	nulls, decoded, err := decodeDataRowNullable(payload)
	if err != nil {
		t.Fatalf("decodeDataRowNullable: %v", err)
	}
	if len(nulls) != 3 {
		t.Fatalf("expected 3 columns, got %d", len(nulls))
	}

	// Column 0: NULL.
	if !nulls[0] {
		t.Error("column 0: expected NULL")
	}
	// Column 1: empty string (NOT null).
	if nulls[1] {
		t.Error("column 1: expected non-NULL (empty string), got NULL")
	}
	if decoded[1] == nil || *decoded[1] != "" {
		t.Errorf("column 1: want empty string, got %v", decoded[1])
	}
	// Column 2: "data".
	if nulls[2] {
		t.Error("column 2: expected non-NULL")
	}
	if decoded[2] == nil || *decoded[2] != "data" {
		t.Errorf("column 2: want %q, got %v", "data", decoded[2])
	}
}

func TestSendDataRow_AllNull(t *testing.T) {
	var buf bytes.Buffer

	values := []*string{nil, nil, nil}
	if err := sendDataRow(&buf, values); err != nil {
		t.Fatalf("sendDataRow: %v", err)
	}

	msgType, payload, err := ReadMessage(&buf)
	if err != nil {
		t.Fatalf("ReadMessage: %v", err)
	}
	if msgType != msgDataRow {
		t.Fatalf("expected 'D', got '%c'", msgType)
	}

	nulls, _, err := decodeDataRowNullable(payload)
	if err != nil {
		t.Fatalf("decodeDataRowNullable: %v", err)
	}

	for i, isNull := range nulls {
		if !isNull {
			t.Errorf("column %d: expected NULL", i)
		}
	}
}

func TestSendDataRow_AllNonNull(t *testing.T) {
	var buf bytes.Buffer

	a, b := "alpha", "beta"
	values := []*string{&a, &b}
	if err := sendDataRow(&buf, values); err != nil {
		t.Fatalf("sendDataRow: %v", err)
	}

	_, decoded, err := decodeDataRowNullable(readDataRowMessage(t, &buf))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	if len(decoded) != 2 {
		t.Fatalf("expected 2 columns, got %d", len(decoded))
	}
	if decoded[0] == nil || *decoded[0] != "alpha" {
		t.Errorf("column 0: want alpha, got %v", decoded[0])
	}
	if decoded[1] == nil || *decoded[1] != "beta" {
		t.Errorf("column 1: want beta, got %v", decoded[1])
	}
}

// =============================================================================
// Unit: buildResultRow NULL semantics
// =============================================================================

func TestBuildResultRow_BuiltinColumnsNeverNull(t *testing.T) {
	r := &libravdb.SearchResult{
		ID:      "rec-1",
		Score:   0.95,
		Version: 3,
		Ordinal: 7,
	}

	columns := []ColumnMeta{
		{Name: "id", TypeOID: OIDText},
		{Name: "score", TypeOID: OIDFloat8},
		{Name: "version", TypeOID: OIDFloat8},
		{Name: "ordinal", TypeOID: OIDInt8},
	}

	row := buildResultRow(r, columns)

	if len(row) != 4 {
		t.Fatalf("expected 4 columns, got %d", len(row))
	}

	for i, col := range columns {
		if row[i] == nil {
			t.Errorf("built-in column %q: should never be NULL, got nil", col.Name)
		}
	}

	if *row[0] != "rec-1" {
		t.Errorf("id: want rec-1, got %q", *row[0])
	}
	if *row[1] != "0.950000" {
		t.Errorf("score: want 0.950000, got %q", *row[1])
	}
}

func TestBuildResultRow_MetadataNilValueIsNull(t *testing.T) {
	// Metadata key exists but value is nil → SQL NULL.
	r := &libravdb.SearchResult{
		ID: "rec-1",
		Metadata: map[string]interface{}{
			"name": nil,
		},
	}

	columns := []ColumnMeta{
		{Name: "id", TypeOID: OIDText},
		{Name: "name", TypeOID: OIDText},
	}

	row := buildResultRow(r, columns)

	if row[0] == nil {
		t.Error("id: should not be NULL")
	}
	if row[1] != nil {
		t.Errorf("name: explicit nil in metadata should be NULL, got %q", *row[1])
	}
}

func TestBuildResultRow_MetadataMissingKeyIsNull(t *testing.T) {
	// Metadata exists but key is absent → SQL NULL.
	r := &libravdb.SearchResult{
		ID:       "rec-1",
		Metadata: map[string]interface{}{},
	}

	columns := []ColumnMeta{
		{Name: "id", TypeOID: OIDText},
		{Name: "email", TypeOID: OIDText}, // not in metadata
	}

	row := buildResultRow(r, columns)

	if row[0] == nil {
		t.Error("id: should not be NULL")
	}
	if row[1] != nil {
		t.Errorf("email: missing key should be NULL, got %q", *row[1])
	}
}

func TestBuildResultRow_NilMetadataIsNull(t *testing.T) {
	// Metadata map itself is nil → all projected columns are NULL.
	r := &libravdb.SearchResult{
		ID:       "rec-1",
		Metadata: nil,
	}

	columns := []ColumnMeta{
		{Name: "id", TypeOID: OIDText},
		{Name: "name", TypeOID: OIDText},
		{Name: "age", TypeOID: OIDInt8},
	}

	row := buildResultRow(r, columns)

	if row[0] == nil {
		t.Error("id: built-in should not be NULL")
	}
	if row[1] != nil {
		t.Errorf("name: nil metadata should produce NULL, got %q", *row[1])
	}
	if row[2] != nil {
		t.Errorf("age: nil metadata should produce NULL, got %q", *row[2])
	}
}

func TestBuildResultRow_EmptyStringNotConvertedToNull(t *testing.T) {
	// Empty string in metadata must stay empty string, not become NULL.
	r := &libravdb.SearchResult{
		ID: "rec-1",
		Metadata: map[string]interface{}{
			"bio": "",
		},
	}

	columns := []ColumnMeta{
		{Name: "id", TypeOID: OIDText},
		{Name: "bio", TypeOID: OIDText},
	}

	row := buildResultRow(r, columns)

	if row[1] == nil {
		t.Error("bio: empty string metadata should not be NULL")
	} else if *row[1] != "" {
		t.Errorf("bio: want empty string, got %q", *row[1])
	}
}

func TestBuildResultRow_NumericMetadataTypes(t *testing.T) {
	r := &libravdb.SearchResult{
		ID: "rec-1",
		Metadata: map[string]interface{}{
			"count":   int64(42),
			"balance": float64(99.95),
			"active":  true,
			"ratio":   float32(0.5),
		},
	}

	columns := []ColumnMeta{
		{Name: "id", TypeOID: OIDText},
		{Name: "count", TypeOID: OIDInt8},
		{Name: "balance", TypeOID: OIDFloat8},
		{Name: "active", TypeOID: OIDBool},
		{Name: "ratio", TypeOID: OIDFloat4},
		{Name: "missing_num", TypeOID: OIDInt8},
	}

	row := buildResultRow(r, columns)

	if row[1] == nil || *row[1] != "42" {
		t.Errorf("count: want 42, got %v", row[1])
	}
	if row[2] == nil || *row[2] != "99.95" {
		t.Errorf("balance: want 99.95, got %v", row[2])
	}
	if row[3] == nil || *row[3] != "true" {
		t.Errorf("active: want true, got %v", row[3])
	}
	if row[4] == nil || *row[4] != "0.5" {
		t.Errorf("ratio: want 0.5, got %v", row[4])
	}
	// Missing numeric column → NULL.
	if row[5] != nil {
		t.Errorf("missing_num: absent metadata should be NULL, got %q", *row[5])
	}
}

func TestBuildResultRow_VectorColumn(t *testing.T) {
	r := &libravdb.SearchResult{
		ID: "vec-1",
		Metadata: map[string]interface{}{
			"embedding": []byte{0x00, 0x00, 0x80, 0x3F}, // 1.0 in float32 LE
		},
	}

	columns := []ColumnMeta{
		{Name: "id", TypeOID: OIDText},
		{Name: "embedding", TypeOID: OIDFloat4Array},
	}

	row := buildResultRow(r, columns)

	if row[0] == nil || *row[0] != "vec-1" {
		t.Errorf("id: want vec-1, got %v", row[0])
	}
	if row[1] == nil {
		t.Error("embedding: should not be NULL")
	}
}

func TestBuildResultRow_MixedNullAndValues(t *testing.T) {
	r := &libravdb.SearchResult{
		ID:      "mix-1",
		Score:   0.88,
		Version: 1,
		Metadata: map[string]interface{}{
			"title":       "Hello",
			"description": nil, // explicit NULL
			"status":      "",  // empty string
			// "category" key intentionally absent
		},
	}

	columns := []ColumnMeta{
		{Name: "id", TypeOID: OIDText},
		{Name: "score", TypeOID: OIDFloat8},
		{Name: "title", TypeOID: OIDText},
		{Name: "description", TypeOID: OIDText},
		{Name: "status", TypeOID: OIDText},
		{Name: "category", TypeOID: OIDText},
	}

	row := buildResultRow(r, columns)

	// id: non-null.
	if row[0] == nil || *row[0] != "mix-1" {
		t.Errorf("id: want mix-1, got %v", row[0])
	}
	// score: non-null.
	if row[1] == nil {
		t.Error("score: should not be NULL")
	}
	// title: non-null.
	if row[2] == nil || *row[2] != "Hello" {
		t.Errorf("title: want Hello, got %v", row[2])
	}
	// description: NULL (explicit nil).
	if row[3] != nil {
		t.Errorf("description: explicit nil should be NULL, got %q", *row[3])
	}
	// status: empty string (NOT null).
	if row[4] == nil {
		t.Error("status: empty string should not be NULL")
	} else if *row[4] != "" {
		t.Errorf("status: want empty string, got %q", *row[4])
	}
	// category: NULL (absent key).
	if row[5] != nil {
		t.Errorf("category: absent key should be NULL, got %q", *row[5])
	}
}

// =============================================================================
// Unit: sendResults wire-format round-trip
// =============================================================================

func TestSendResults_RoundTripPreservesNullSemantics(t *testing.T) {
	results := &libravdb.SearchResults{
		Results: []*libravdb.SearchResult{
			{
				ID:      "r1",
				Score:   0.9,
				Version: 1,
				Metadata: map[string]interface{}{
					"name":  "Alice",
					"email": nil, // NULL
					"bio":   "",  // empty string
				},
			},
		},
		Columns:     []string{"id", "score", "name", "email", "bio"},
		ColumnTypes: []uint16{0, 0, 0, 0, 0},
		Total:       1,
	}

	columns := []ColumnMeta{
		{Name: "id", TypeOID: OIDText},
		{Name: "score", TypeOID: OIDFloat8},
		{Name: "name", TypeOID: OIDText},
		{Name: "email", TypeOID: OIDText},
		{Name: "bio", TypeOID: OIDText},
	}

	var buf bytes.Buffer
	if err := sendResults(&buf, results, columns); err != nil {
		t.Fatalf("sendResults: %v", err)
	}

	// Read RowDescription.
	msgType, _, err := ReadMessage(&buf)
	if err != nil {
		t.Fatalf("ReadMessage RowDescription: %v", err)
	}
	if msgType != msgRowDescription {
		t.Fatalf("expected 'T', got '%c'", msgType)
	}

	// Read DataRow.
	msgType, payload, err := ReadMessage(&buf)
	if err != nil {
		t.Fatalf("ReadMessage DataRow: %v", err)
	}
	if msgType != msgDataRow {
		t.Fatalf("expected 'D', got '%c'", msgType)
	}

	nulls, decoded, err := decodeDataRowNullable(payload)
	if err != nil {
		t.Fatalf("decodeDataRowNullable: %v", err)
	}

	if len(nulls) != 5 {
		t.Fatalf("expected 5 columns, got %d", len(nulls))
	}

	// id: non-NULL.
	if nulls[0] {
		t.Error("id: expected non-NULL")
	}
	if decoded[0] == nil || *decoded[0] != "r1" {
		t.Errorf("id: want r1, got %v", decoded[0])
	}
	// score: non-NULL.
	if nulls[1] {
		t.Error("score: expected non-NULL")
	}
	// name: non-NULL.
	if nulls[2] {
		t.Error("name: expected non-NULL")
	}
	if decoded[2] == nil || *decoded[2] != "Alice" {
		t.Errorf("name: want Alice, got %v", decoded[2])
	}
	// email: NULL.
	if !nulls[3] {
		t.Error("email: expected NULL")
	}
	// bio: empty string (NOT null).
	if nulls[4] {
		t.Error("bio: expected non-NULL (empty string)")
	}
	if decoded[4] == nil || *decoded[4] != "" {
		t.Errorf("bio: want empty string, got %v", decoded[4])
	}
}

// =============================================================================
// Text-format: nullable text, numeric, vector, and graph columns
// =============================================================================

func TestNullEncoding_TextFormat_NullableTextColumns(t *testing.T) {
	// Verify text NULL encoding end-to-end through sendDataRow wire format.
	var buf bytes.Buffer

	name := "Alice"
	values := []*string{&name, nil} // name + NULL email
	if err := sendDataRow(&buf, values); err != nil {
		t.Fatalf("sendDataRow: %v", err)
	}

	nulls, decoded, err := decodeDataRowNullable(readDataRowMessage(t, &buf))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	if nulls[0] {
		t.Error("name: expected non-NULL text")
	}
	if !nulls[1] {
		t.Error("email: expected NULL text")
	}
	_ = decoded
}

func TestNullEncoding_TextFormat_NullableNumericColumns(t *testing.T) {
	var buf bytes.Buffer

	age := "42"
	values := []*string{nil, &age} // NULL salary + age
	if err := sendDataRow(&buf, values); err != nil {
		t.Fatalf("sendDataRow: %v", err)
	}

	nulls, decoded, err := decodeDataRowNullable(readDataRowMessage(t, &buf))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	if !nulls[0] {
		t.Error("salary: expected NULL numeric")
	}
	if nulls[1] {
		t.Error("age: expected non-NULL numeric")
	}
	if decoded[1] == nil || *decoded[1] != "42" {
		t.Errorf("age: want 42, got %v", decoded[1])
	}
}

func TestNullEncoding_TextFormat_NullableVectorColumns(t *testing.T) {
	// Vector columns in text format are opaque strings; NULL is still NULL.
	var buf bytes.Buffer

	vec := "[1.0, 2.0, 3.0]"
	values := []*string{&vec, nil}
	if err := sendDataRow(&buf, values); err != nil {
		t.Fatalf("sendDataRow: %v", err)
	}

	nulls, decoded, err := decodeDataRowNullable(readDataRowMessage(t, &buf))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	if nulls[0] {
		t.Error("vector: expected non-NULL")
	}
	if decoded[0] == nil || *decoded[0] != vec {
		t.Errorf("vector: want %q, got %v", vec, decoded[0])
	}
	if !nulls[1] {
		t.Error("vector2: expected NULL")
	}
}

func TestNullEncoding_TextFormat_NullableGraphColumns(t *testing.T) {
	// Graph result columns (node_id, community_id, etc.) in text format.
	var buf bytes.Buffer

	nodeID := "42"
	communityID := "7"
	truncated := "false"
	values := []*string{&nodeID, &communityID, nil, &truncated}
	if err := sendDataRow(&buf, values); err != nil {
		t.Fatalf("sendDataRow: %v", err)
	}

	nulls, decoded, err := decodeDataRowNullable(readDataRowMessage(t, &buf))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	// node_id: non-NULL.
	if nulls[0] {
		t.Error("node_id: expected non-NULL")
	}
	if decoded[0] == nil || *decoded[0] != "42" {
		t.Errorf("node_id: want 42, got %v", decoded[0])
	}
	// community_id: non-NULL.
	if nulls[1] {
		t.Error("community_id: expected non-NULL")
	}
	// record_id: NULL.
	if !nulls[2] {
		t.Error("record_id: expected NULL")
	}
	// truncated: non-NULL.
	if nulls[3] {
		t.Error("truncated: expected non-NULL")
	}
	if decoded[3] == nil || *decoded[3] != "false" {
		t.Errorf("truncated: want false, got %v", decoded[3])
	}
}

// =============================================================================
// Integration: metadata NULL values survive round-trip through SearchResult
// =============================================================================

func TestNullEncoding_AllNullProjectedColumns(t *testing.T) {
	// When all metadata fields are nil or absent, all projected columns are NULL.
	r := &libravdb.SearchResult{
		ID:       "r-null-all",
		Score:    0.5,
		Version:  1,
		Metadata: nil,
	}

	columns := []ColumnMeta{
		{Name: "id", TypeOID: OIDText},
		{Name: "score", TypeOID: OIDFloat8},
		{Name: "col_a", TypeOID: OIDText},
		{Name: "col_b", TypeOID: OIDInt8},
		{Name: "col_c", TypeOID: OIDBool},
	}

	row := buildResultRow(r, columns)

	// Built-in columns always non-NULL.
	if row[0] == nil {
		t.Error("id: built-in should not be NULL")
	}
	if row[1] == nil {
		t.Error("score: built-in should not be NULL")
	}
	// All projected columns from nil metadata → NULL.
	for i := 2; i < len(columns); i++ {
		if row[i] != nil {
			t.Errorf("projected column %q with nil metadata: expected NULL, got %q",
				columns[i].Name, *row[i])
		}
	}
}

// =============================================================================
// Helpers: NULL-aware DataRow decoding
// =============================================================================

// decodeDataRowNullable decodes a DataRow payload into a parallel pair:
//   - nulls[i] is true when column i is SQL NULL (length -1).
//   - decoded[i] is a pointer to the text value, or nil when the column is NULL.
func decodeDataRowNullable(payload []byte) ([]bool, []*string, error) {
	if len(payload) < 2 {
		return nil, nil, fmt.Errorf("DataRow too short: %d bytes", len(payload))
	}
	n := int(binary.BigEndian.Uint16(payload[:2]))
	nulls := make([]bool, n)
	decoded := make([]*string, n)
	off := 2

	for i := 0; i < n; i++ {
		if off+4 > len(payload) {
			return nil, nil, fmt.Errorf("DataRow truncated at column %d", i)
		}
		// Read as int32: -1 (0xFFFFFFFF) means SQL NULL.
		colLen := int32(binary.BigEndian.Uint32(payload[off:]))
		off += 4

		if colLen == -1 {
			nulls[i] = true
			decoded[i] = nil
		} else {
			nulls[i] = false
			if off+int(colLen) > len(payload) {
				return nil, nil, fmt.Errorf("DataRow column %d value truncated", i)
			}
			s := string(payload[off : off+int(colLen)])
			decoded[i] = &s
			off += int(colLen)
		}
	}
	return nulls, decoded, nil
}

// readDataRowMessage reads a single DataRow message from the reader and returns
// its payload. Fails the test if the message type is not 'D'.
func readDataRowMessage(t *testing.T, r io.Reader) []byte {
	t.Helper()
	msgType, payload, err := ReadMessage(r)
	if err != nil {
		t.Fatalf("ReadMessage: %v", err)
	}
	if msgType != msgDataRow {
		t.Fatalf("expected DataRow ('D'), got '%c'", msgType)
	}
	return payload
}

// =============================================================================
// Dependency check: all callsites compile with new signatures
// =============================================================================

func TestSendQueryResult_HandlesNilSearchResults(t *testing.T) {
	// sendQueryResult with nil results should send EmptyQuery and ReadyForQuery.
	// This is a compile-time + smoke check that the new []*string signatures
	// don't break the calling chain.
	var buf bytes.Buffer
	if err := sendQueryResult(&buf, nil, nil); err != nil {
		t.Fatalf("sendQueryResult(nil): %v", err)
	}

	// Should receive EmptyQuery ('I').
	msgType, _, err := ReadMessage(&buf)
	if err != nil {
		t.Fatalf("ReadMessage: %v", err)
	}
	if msgType != msgEmptyQuery {
		t.Fatalf("expected EmptyQuery ('I'), got '%c'", msgType)
	}

	// Then ReadyForQuery ('Z').
	msgType, _, err = ReadMessage(&buf)
	if err != nil {
		t.Fatalf("ReadMessage Z: %v", err)
	}
	if msgType != msgReadyForQuery {
		t.Fatalf("expected ReadyForQuery ('Z'), got '%c'", msgType)
	}
}

func TestSendQueryResult_HandlesEmptyResults(t *testing.T) {
	results := &libravdb.SearchResults{
		Results: nil,
		Total:   0,
	}
	var buf bytes.Buffer
	if err := sendQueryResult(&buf, results, nil); err != nil {
		t.Fatalf("sendQueryResult(empty): %v", err)
	}

	// Should get EmptyQuery.
	msgType, _, err := ReadMessage(&buf)
	if err != nil {
		t.Fatalf("ReadMessage: %v", err)
	}
	if msgType != msgEmptyQuery {
		t.Fatalf("expected EmptyQuery ('I'), got '%c'", msgType)
	}
}

func TestSendExtendedQueryResult_HandlesNilSearchResults(t *testing.T) {
	var buf bytes.Buffer
	if err := sendExtendedQueryResult(&buf, nil, nil); err != nil {
		t.Fatalf("sendExtendedQueryResult(nil): %v", err)
	}

	msgType, _, err := ReadMessage(&buf)
	if err != nil {
		t.Fatalf("ReadMessage: %v", err)
	}
	if msgType != msgEmptyQuery {
		t.Fatalf("expected EmptyQuery ('I'), got '%c'", msgType)
	}
}

// =============================================================================
// Regression: columnOIDFor handles nil metadata in rows
// =============================================================================

func TestColumnOIDFor_HandlesNilSearchResult(t *testing.T) {
	// columnOIDFor iterates results; nil rows must be skipped.
	results := &libravdb.SearchResults{
		Results: []*libravdb.SearchResult{nil, {Metadata: map[string]interface{}{"x": "42"}}},
	}

	oid := columnOIDFor(results, "x")
	if oid != OIDInt8 {
		t.Errorf("x value 42: expected OIDInt8 (%d), got %d", OIDInt8, oid)
	}
}

func TestColumnOIDFor_HandlesNilMetadata(t *testing.T) {
	results := &libravdb.SearchResults{
		Results: []*libravdb.SearchResult{
			{ID: "a", Metadata: nil},
			{ID: "b", Metadata: map[string]interface{}{"y": int64(99)}},
		},
	}

	oid := columnOIDFor(results, "y")
	if oid != OIDInt8 {
		t.Errorf("y value int64(99): expected OIDInt8 (%d), got %d", OIDInt8, oid)
	}
}
