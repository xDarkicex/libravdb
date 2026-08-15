package pgwire

import (
	"bytes"
	"context"
	"net"
	"testing"

	"github.com/xDarkicex/libravdb/libravdb"
)

// =============================================================================
// COPY detection
// =============================================================================

func TestIsCopy_DetectsFromStdin(t *testing.T) {
	tests := []struct {
		query string
		want  bool
	}{
		{"COPY users FROM STDIN", true},
		{"COPY users (id, name) FROM STDIN", true},
		{"COPY users FROM STDIN WITH (FORMAT csv)", true},
		{"COPY users TO STDOUT", true},
		{"COPY users (id) TO STDOUT", true},
		{"SELECT * FROM users", false},
		{"COPY", false},
		{"", false},
	}
	for _, tt := range tests {
		if got := isCopy(tt.query); got != tt.want {
			t.Errorf("isCopy(%q) = %v, want %v", tt.query, got, tt.want)
		}
	}
}

func TestIsCopyToStdout(t *testing.T) {
	if !isCopyToStdout("COPY users TO STDOUT") {
		t.Error("COPY TO STDOUT not detected")
	}
	if isCopyToStdout("COPY users FROM STDIN") {
		t.Error("COPY FROM STDIN misdetected as TO STDOUT")
	}
}

// =============================================================================
// Query parsing
// =============================================================================

func TestParseCopyOptions_TableAndColumns(t *testing.T) {
	opts := parseCopyOptions("COPY users (id, name, email) FROM STDIN")
	if opts.table != "users" {
		t.Errorf("table: want users, got %q", opts.table)
	}
	if len(opts.columns) != 3 {
		t.Fatalf("columns: want 3, got %d", len(opts.columns))
	}
	if opts.columns[0] != "id" || opts.columns[1] != "name" || opts.columns[2] != "email" {
		t.Errorf("columns: got %v", opts.columns)
	}
}

func TestParseCopyOptions_ToStdout(t *testing.T) {
	opts := parseCopyOptions("COPY items TO STDOUT")
	if opts.table != "items" {
		t.Errorf("table: want items, got %q", opts.table)
	}
}

func TestParseCopyOptions_Defaults(t *testing.T) {
	opts := parseCopyOptions("COPY t FROM STDIN")
	if opts.format != copyFormatText {
		t.Errorf("format: want text, got %q", opts.format)
	}
	if opts.delimiter != '\t' {
		t.Errorf("delimiter: want tab, got %q", opts.delimiter)
	}
	if opts.nullStr != "\\N" {
		t.Errorf("null: want \\N, got %q", opts.nullStr)
	}
}

func TestParseCopyOptions_CSVFormat(t *testing.T) {
	opts := parseCopyOptions("COPY t FROM STDIN WITH (FORMAT csv)")
	if opts.format != copyFormatCSV {
		t.Errorf("format: want csv, got %q", opts.format)
	}
	if opts.delimiter != ',' {
		t.Errorf("delimiter: want comma, got %q", opts.delimiter)
	}
	if opts.nullStr != "" {
		t.Errorf("csv null: want empty, got %q", opts.nullStr)
	}
}

func TestParseCopyOptions_CSVWithHeader(t *testing.T) {
	opts := parseCopyOptions("COPY t FROM STDIN WITH (FORMAT csv, HEADER true)")
	if !opts.header {
		t.Error("header should be true")
	}
}

func TestParseCopyOptions_CustomNull(t *testing.T) {
	opts := parseCopyOptions("COPY t FROM STDIN WITH (NULL '@@')")
	if opts.nullStr != "@@" {
		t.Errorf("null: want @@, got %q", opts.nullStr)
	}
}

// =============================================================================
// Text format row parsing
// =============================================================================

func TestParseTextRow_BasicFields(t *testing.T) {
	opts := defaultCopyOptions()
	row := parseTextRow([]byte("alice\t30\tNYC\n"), opts)

	if len(row) != 3 {
		t.Fatalf("expected 3 fields, got %d", len(row))
	}
	if row[0] == nil || *row[0] != "alice" {
		t.Errorf("field 0: want alice, got %v", row[0])
	}
	if row[1] == nil || *row[1] != "30" {
		t.Errorf("field 1: want 30, got %v", row[1])
	}
	if row[2] == nil || *row[2] != "NYC" {
		t.Errorf("field 2: want NYC, got %v", row[2])
	}
}

func TestParseTextRow_NullMarker(t *testing.T) {
	opts := defaultCopyOptions()
	row := parseTextRow([]byte("alice\t\\N\tNYC\n"), opts)

	if len(row) != 3 {
		t.Fatalf("expected 3 fields, got %d", len(row))
	}
	// \N must be NULL (nil pointer), not the literal string "\N".
	if row[1] != nil {
		t.Errorf("field 1 (\\N): expected NULL (nil), got %q", *row[1])
	}
}

func TestParseTextRow_EmptyString(t *testing.T) {
	// Consecutive tabs produce an empty string, not NULL.
	opts := defaultCopyOptions()
	row := parseTextRow([]byte("alice\t\tNYC\n"), opts)

	if len(row) != 3 {
		t.Fatalf("expected 3 fields, got %d", len(row))
	}
	if row[1] == nil {
		t.Error("field 1 (empty): expected empty string, got NULL")
	} else if *row[1] != "" {
		t.Errorf("field 1: want empty string, got %q", *row[1])
	}
}

func TestParseTextRow_EmptyStringVsNull(t *testing.T) {
	// Tab\t\N\ttab → empty string vs NULL must be distinguishable.
	opts := defaultCopyOptions()
	row := parseTextRow([]byte("alice\t\t\\N\n"), opts)

	if len(row) != 3 {
		t.Fatalf("expected 3 fields, got %d", len(row))
	}
	// Field 1: empty string (not NULL).
	if row[1] == nil {
		t.Error("field 1: empty string should not be NULL")
	} else if *row[1] != "" {
		t.Errorf("field 1: want empty, got %q", *row[1])
	}
	// Field 2: NULL.
	if row[2] != nil {
		t.Errorf("field 2 (\\N): expected NULL, got %q", *row[2])
	}
}

func TestParseTextRow_TrailingNewline(t *testing.T) {
	opts := defaultCopyOptions()
	row := parseTextRow([]byte("hello\tworld\r\n"), opts)
	if len(row) != 2 {
		t.Fatalf("expected 2 fields, got %d", len(row))
	}
}

func TestParseTextRow_EndOfData(t *testing.T) {
	opts := defaultCopyOptions()
	row := parseTextRow([]byte("\\.\n"), opts)
	if len(row) != 0 {
		t.Errorf("\\. should return empty row, got %d fields", len(row))
	}
}

func TestParseTextRow_EmptyPayload(t *testing.T) {
	opts := defaultCopyOptions()
	row := parseTextRow(nil, opts)
	if len(row) != 0 {
		t.Errorf("nil payload should return empty row, got %d", len(row))
	}
}

func TestParseTextRow_LeadingTrailingTabs(t *testing.T) {
	// Leading empty field → empty string.
	opts := defaultCopyOptions()
	row := parseTextRow([]byte("\talice\t\n"), opts)

	if len(row) != 3 {
		t.Fatalf("expected 3 fields, got %d", len(row))
	}
	if row[0] == nil {
		t.Error("field 0: leading tab should be empty string, not NULL")
	} else if *row[0] != "" {
		t.Errorf("field 0: want empty, got %q", *row[0])
	}
	if row[1] == nil || *row[1] != "alice" {
		t.Errorf("field 1: want alice, got %v", row[1])
	}
	if row[2] == nil {
		t.Error("field 2: trailing tab should be empty string, not NULL")
	} else if *row[2] != "" {
		t.Errorf("field 2: want empty, got %q", *row[2])
	}
}

func TestParseTextRow_AllNulls(t *testing.T) {
	opts := defaultCopyOptions()
	row := parseTextRow([]byte("\\N\t\\N\t\\N\n"), opts)
	if len(row) != 3 {
		t.Fatalf("expected 3 fields, got %d", len(row))
	}
	for i, v := range row {
		if v != nil {
			t.Errorf("field %d: expected NULL, got %q", i, *v)
		}
	}
}

// =============================================================================
// CSV format row parsing
// =============================================================================

func TestParseCSVRow_BasicFields(t *testing.T) {
	opts := copyOptions{format: copyFormatCSV, delimiter: ',', nullStr: "", quote: '"', escape: '"'}
	row := parseCSVRow([]byte("alice,30,NYC\n"), opts)

	if len(row) != 3 {
		t.Fatalf("expected 3 fields, got %d", len(row))
	}
	if row[0] == nil || *row[0] != "alice" {
		t.Errorf("field 0: want alice, got %v", row[0])
	}
	if row[1] == nil || *row[1] != "30" {
		t.Errorf("field 1: want 30, got %v", row[1])
	}
	if row[2] == nil || *row[2] != "NYC" {
		t.Errorf("field 2: want NYC, got %v", row[2])
	}
}

func TestParseCSVRow_NullMarker(t *testing.T) {
	// Default CSV NULL is unquoted empty string (consecutive commas).
	opts := copyOptions{format: copyFormatCSV, delimiter: ',', nullStr: "", quote: '"', escape: '"'}
	row := parseCSVRow([]byte("alice,,NYC\n"), opts)

	if len(row) != 3 {
		t.Fatalf("expected 3 fields, got %d", len(row))
	}
	if row[1] != nil {
		t.Errorf("field 1 (empty unquoted): expected NULL, got %q", *row[1])
	}
}

func TestParseCSVRow_QuotedEmptyString(t *testing.T) {
	// Quoted empty string "" is NOT NULL.
	opts := copyOptions{format: copyFormatCSV, delimiter: ',', nullStr: "", quote: '"', escape: '"'}
	row := parseCSVRow([]byte("alice,\"\",NYC\n"), opts)

	if len(row) != 3 {
		t.Fatalf("expected 3 fields, got %d", len(row))
	}
	if row[1] == nil {
		t.Error("field 1 (quoted empty): expected empty string, got NULL")
	} else if *row[1] != "" {
		t.Errorf("field 1: want empty, got %q", *row[1])
	}
}

func TestParseCSVRow_QuotedFieldWithComma(t *testing.T) {
	opts := copyOptions{format: copyFormatCSV, delimiter: ',', nullStr: "", quote: '"', escape: '"'}
	row := parseCSVRow([]byte("\"New York, NY\",30\n"), opts)

	if len(row) != 2 {
		t.Fatalf("expected 2 fields, got %d", len(row))
	}
	if row[0] == nil || *row[0] != "New York, NY" {
		t.Errorf("field 0: want 'New York, NY', got %v", row[0])
	}
}

func TestParseCSVRow_DoubledQuoteEscape(t *testing.T) {
	// PostgreSQL CSV uses doubled quotes for escaping: "say ""hello""" → say "hello"
	opts := copyOptions{format: copyFormatCSV, delimiter: ',', nullStr: "", quote: '"', escape: '"'}
	row := parseCSVRow([]byte("\"say \"\"hello\"\"\",30\n"), opts)

	if len(row) != 2 {
		t.Fatalf("expected 2 fields, got %d", len(row))
	}
	if row[0] == nil || *row[0] != "say \"hello\"" {
		t.Errorf("field 0: want 'say \"hello\"', got %q", *row[0])
	}
}

// =============================================================================
// Entry building
// =============================================================================

func TestBuildEntryFromRow_NullMetadata(t *testing.T) {
	id := "rec-1"
	name := "Alice"
	row := []*string{&id, &name, nil} // id, name, NULL email
	columns := []string{"id", "name", "email"}

	entry := buildEntryFromRow(row, columns)

	if entry.ID != "rec-1" {
		t.Errorf("id: want rec-1, got %q", entry.ID)
	}
	if entry.Metadata["name"] != "Alice" {
		t.Errorf("name: want Alice, got %v", entry.Metadata["name"])
	}
	// Email should be nil in metadata (SQL NULL).
	if v, ok := entry.Metadata["email"]; !ok {
		t.Error("email key missing from metadata")
	} else if v != nil {
		t.Errorf("email: expected nil, got %v", v)
	}
}

func TestBuildEntryFromRow_EmptyStringInMetadata(t *testing.T) {
	id := "rec-1"
	bio := ""
	row := []*string{&id, &bio}
	columns := []string{"id", "bio"}

	entry := buildEntryFromRow(row, columns)

	if entry.Metadata["bio"] != "" {
		t.Errorf("bio: want empty string, got %v", entry.Metadata["bio"])
	}
}

func TestBuildEntryFromRow_NoColumns(t *testing.T) {
	id := "rec-1"
	vec := "[1.0,2.0]"
	row := []*string{&id, &vec}
	entry := buildEntryFromRow(row, nil)

	if entry.ID != "rec-1" {
		t.Errorf("id: want rec-1, got %q", entry.ID)
	}
	if len(entry.Vector) != 2 {
		t.Errorf("vector: want 2 elements, got %d", len(entry.Vector))
	}
}

// =============================================================================
// COPY TO STDOUT row formatting
// =============================================================================

func TestFormatCopyRow_TextFormat(t *testing.T) {
	opts := defaultCopyOptions()
	rec := libravdb.Record{
		ID:      "r1",
		Version: 1,
		Metadata: map[string]interface{}{
			"name":  "Alice",
			"email": nil,
			"bio":   "",
		},
	}

	data := formatCopyRow(rec, []string{"id", "name", "email", "bio"}, opts)
	expected := "r1\tAlice\t\\N\t\n"
	if string(data) != expected {
		t.Errorf("text format row:\n  got  %q\n  want %q", string(data), expected)
	}
}

func TestFormatCopyRow_CSVFormat(t *testing.T) {
	opts := copyOptions{
		format:    copyFormatCSV,
		delimiter: ',',
		nullStr:   "",
		quote:     '"',
	}
	rec := libravdb.Record{
		ID: "r1",
		Metadata: map[string]interface{}{
			"name":  "Alice",
			"email": nil,
			"bio":   "hello, world",
		},
	}

	data := formatCopyRow(rec, []string{"id", "name", "email", "bio"}, opts)
	expected := "r1,Alice,,\"hello, world\"\n"
	if string(data) != expected {
		t.Errorf("csv format row:\n  got  %q\n  want %q", string(data), expected)
	}
}

func TestFormatCopyRow_VectorColumn(t *testing.T) {
	opts := defaultCopyOptions()
	rec := libravdb.Record{
		ID:     "v1",
		Vector: []float32{1.0, 2.0, 3.5},
	}

	data := formatCopyRow(rec, []string{"id", "vector"}, opts)
	expected := "v1\t[1,2,3.5]\n"
	if string(data) != expected {
		t.Errorf("vector format:\n  got  %q\n  want %q", string(data), expected)
	}
}

func TestFormatCopyHeader_CSV(t *testing.T) {
	opts := copyOptions{format: copyFormatCSV, delimiter: ',', quote: '"'}
	data := formatCopyHeader([]string{"id", "name"}, opts)
	expected := "id,name\n"
	if string(data) != expected {
		t.Errorf("csv header:\n  got  %q\n  want %q", string(data), expected)
	}
}

// =============================================================================
// Integration: COPY ... FROM STDIN wire-protocol test
// =============================================================================

func TestCopyFromStdin_InsertsRecords(t *testing.T) {
	db := openTestDB(t)
	defer db.Close()

	ctx := context.Background()
	_, err := db.CreateCollection(ctx, "cp_test", libravdb.WithMetadataOnly())
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	srv := startTestServer(t, db)
	defer srv.Close()

	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	// Send COPY ... FROM STDIN query.
	sendSimpleQuery(conn, "COPY cp_test (id, name, age) FROM STDIN")

	// Expect CopyInResponse ('G').
	msgType, _, err := ReadMessage(conn)
	if err != nil {
		t.Fatalf("expected CopyInResponse: %v", err)
	}
	if msgType != msgCopyInResponse {
		t.Fatalf("expected CopyInResponse ('G'), got '%c'", msgType)
	}

	// Send data rows.
	sendCopyDataMsg(conn, "r1\tAlice\t30\n")
	sendCopyDataMsg(conn, "r2\tBob\t\\N\n") // NULL age
	sendCopyDataMsg(conn, "r3\t\t25\n")     // empty name

	// Send CopyDone.
	WriteMessage(conn, msgCopyDone, nil)

	// Expect CommandComplete + ReadyForQuery.
	msgType, _, err = ReadMessage(conn)
	if err != nil {
		t.Fatalf("expected CommandComplete: %v", err)
	}
	if msgType != msgCommandComplete {
		t.Fatalf("expected CommandComplete ('C'), got '%c'", msgType)
	}

	msgType, _, err = ReadMessage(conn)
	if err != nil {
		t.Fatalf("expected ReadyForQuery: %v", err)
	}
	if msgType != msgReadyForQuery {
		t.Fatalf("expected ReadyForQuery ('Z'), got '%c'", msgType)
	}

	// Verify rows were inserted.
	col, _ := db.GetCollection("cp_test")

	rec, err := col.Get(ctx, "r1")
	if err != nil {
		t.Fatalf("Get r1: %v", err)
	}
	if rec.Metadata["name"] != "Alice" || rec.Metadata["age"] != "30" {
		t.Errorf("r1 metadata: %v", rec.Metadata)
	}

	rec, err = col.Get(ctx, "r2")
	if err != nil {
		t.Fatalf("Get r2: %v", err)
	}
	if v, ok := rec.Metadata["age"]; !ok || v != nil {
		t.Errorf("r2 age: expect nil (SQL NULL), got %v", v)
	}

	rec, err = col.Get(ctx, "r3")
	if err != nil {
		t.Fatalf("Get r3: %v", err)
	}
	if rec.Metadata["name"] != "" {
		t.Errorf("r3 name: want empty string, got %q", rec.Metadata["name"])
	}
}

func TestCopyFromStdin_CopyFailAborts(t *testing.T) {
	db := openTestDB(t)
	defer db.Close()

	ctx := context.Background()
	db.CreateCollection(ctx, "cp_fail", libravdb.WithMetadataOnly())

	srv := startTestServer(t, db)
	defer srv.Close()

	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	// Start COPY.
	sendSimpleQuery(conn, "COPY cp_fail (id) FROM STDIN")

	// Expect CopyInResponse.
	msgType, _, err := ReadMessage(conn)
	if err != nil {
		t.Fatalf("expected CopyInResponse: %v", err)
	}
	if msgType != msgCopyInResponse {
		t.Fatalf("expected 'G', got '%c'", msgType)
	}

	// Send CopyFail — client aborts the COPY.
	WriteMessage(conn, msgCopyFail, nil)

	// Expect ReadyForQuery directly (no CommandComplete).
	msgType, _, err = ReadMessage(conn)
	if err != nil {
		t.Fatalf("expected ReadyForQuery after CopyFail: %v", err)
	}
	if msgType != msgReadyForQuery {
		t.Fatalf("expected ReadyForQuery ('Z'), got '%c'", msgType)
	}

	t.Log("CopyFail correctly returned ReadyForQuery")
}

func TestCopyFromStdin_NonexistentTable(t *testing.T) {
	db := openTestDB(t)
	defer db.Close()

	srv := startTestServer(t, db)
	defer srv.Close()

	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	sendSimpleQuery(conn, "COPY nonexistent FROM STDIN")

	// Expect CopyInResponse first.
	msgType, _, err := ReadMessage(conn)
	if err != nil {
		t.Fatalf("expected CopyInResponse: %v", err)
	}
	if msgType != msgCopyInResponse {
		t.Fatalf("expected 'G', got '%c'", msgType)
	}

	// Send CopyDone immediately.
	WriteMessage(conn, msgCopyDone, nil)

	// Expect ErrorResponse (table doesn't exist).
	msgType, _, err = ReadMessage(conn)
	if err != nil {
		t.Fatalf("expected ErrorResponse: %v", err)
	}
	if msgType != msgErrorResponse {
		t.Fatalf("expected ErrorResponse ('E'), got '%c'", msgType)
	}

	t.Log("Nonexistent table correctly returned error")
}

// =============================================================================
// Integration: COPY ... TO STDOUT wire-protocol test
// =============================================================================

func TestCopyToStdout_SendsRecords(t *testing.T) {
	db := openTestDB(t)
	defer db.Close()

	ctx := context.Background()
	_, err := db.CreateCollection(ctx, "cp_out", libravdb.WithMetadataOnly())
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	col, _ := db.GetCollection("cp_out")
	col.Insert(ctx, "a", nil, map[string]interface{}{"name": "Alice"})
	col.Insert(ctx, "b", nil, map[string]interface{}{"name": "Bob"})

	srv := startTestServer(t, db)
	defer srv.Close()

	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	// Send COPY ... TO STDOUT.
	sendSimpleQuery(conn, "COPY cp_out TO STDOUT")

	// Expect CopyOutResponse ('H').
	msgType, _, err := ReadMessage(conn)
	if err != nil {
		t.Fatalf("expected CopyOutResponse: %v", err)
	}
	if msgType != msgCopyOutResponse {
		t.Fatalf("expected CopyOutResponse ('H'), got '%c'", msgType)
	}

	// Read CopyData rows.
	rowCount := 0
	for {
		msgType, payload, err := ReadMessage(conn)
		if err != nil {
			t.Fatalf("reading COPY data: %v", err)
		}
		if msgType == msgCopyDone {
			break
		}
		if msgType == msgCopyData {
			// Parse the text row.
			row := parseTextRow(payload, defaultCopyOptions())
			if len(row) >= 2 {
				rowCount++
				t.Logf("COPY OUT row %d: id=%v name=%v", rowCount, row[0], row[1])
			}
		} else {
			t.Fatalf("unexpected message '%c' during COPY OUT", msgType)
		}
	}

	if rowCount < 2 {
		t.Errorf("expected at least 2 rows, got %d", rowCount)
	}

	// CommandComplete + ReadyForQuery.
	consumeUntilReady(t, conn)

	t.Logf("COPY TO STDOUT returned %d rows", rowCount)
}

func TestCopyToStdout_CSVWithHeader(t *testing.T) {
	db := openTestDB(t)
	defer db.Close()

	ctx := context.Background()
	db.CreateCollection(ctx, "cp_csv", libravdb.WithMetadataOnly())
	col, _ := db.GetCollection("cp_csv")
	col.Insert(ctx, "x", nil, map[string]interface{}{"val": "42"})

	srv := startTestServer(t, db)
	defer srv.Close()

	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	sendSimpleQuery(conn, "COPY cp_csv TO STDOUT WITH (FORMAT csv, HEADER true)")

	// CopyOutResponse.
	msgType, _, _ := ReadMessage(conn)
	if msgType != msgCopyOutResponse {
		t.Fatalf("expected 'H', got '%c'", msgType)
	}

	// First CopyData should be the header row.
	msgType, payload, _ := ReadMessage(conn)
	if msgType != msgCopyData {
		t.Fatalf("expected CopyData (header), got '%c'", msgType)
	}
	headerLine := string(bytes.TrimRight(payload, "\r\n"))
	t.Logf("CSV header: %q", headerLine)
	if headerLine == "" {
		t.Error("CSV header is empty")
	}

	// Read until CopyDone.
	rows := 0
	for {
		msgType, _, err := ReadMessage(conn)
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		if msgType == msgCopyDone {
			break
		}
		if msgType == msgCopyData {
			rows++
		}
	}
	if rows < 1 {
		t.Error("expected at least 1 data row after header")
	}
	consumeUntilReady(t, conn)
	t.Logf("CSV COPY OUT: header + %d data rows", rows)
}

// =============================================================================
// Integration: epoch transaction routing
// =============================================================================

func TestCopyFromStdin_InsideEpochTransaction(t *testing.T) {
	db := openTestDB(t)
	defer db.Close()

	ctx := context.Background()
	db.CreateCollection(ctx, "cp_epoch", libravdb.WithMetadataOnly())

	srv := startTestServer(t, db)
	defer srv.Close()

	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	// BEGIN EPOCH.
	sendSimpleQuery(conn, "BEGIN EPOCH TRANSACTION")
	consumeUntilReady(t, conn) // Expect ReadyForQuery('T')

	// COPY inside epoch.
	sendSimpleQuery(conn, "COPY cp_epoch (id, name) FROM STDIN")

	msgType, _, _ := ReadMessage(conn) // CopyInResponse
	if msgType != msgCopyInResponse {
		t.Fatalf("expected 'G', got '%c'", msgType)
	}

	sendCopyDataMsg(conn, "e1\tEpochRow\n")
	WriteMessage(conn, msgCopyDone, nil)

	// CommandComplete + ReadyForQuery('T') — still in transaction.
	_, _, _ = ReadMessage(conn)       // CommandComplete
	msgType, _, _ = ReadMessage(conn) // ReadyForQuery
	if msgType != msgReadyForQuery {
		t.Fatalf("expected 'Z', got '%c'", msgType)
	}

	// Before commit, the row should NOT be visible.
	col, _ := db.GetCollection("cp_epoch")
	_, err := col.Get(ctx, "e1")
	if err == nil {
		t.Error("row should NOT be visible before commit")
	}

	// COMMIT.
	sendSimpleQuery(conn, "COMMIT")
	consumeUntilReady(t, conn) // ReadyForQuery('I')

	// After commit, the row SHOULD be visible.
	rec, err := col.Get(ctx, "e1")
	if err != nil {
		t.Fatalf("row should be visible after commit: %v", err)
	}
	if rec.Metadata["name"] != "EpochRow" {
		t.Errorf("name: want EpochRow, got %v", rec.Metadata["name"])
	}

	t.Log("COPY inside epoch transaction: staged and committed correctly")
}

// =============================================================================
// CSV COPY FROM STDIN integration
// =============================================================================

func TestCopyFromStdin_CSVFormat(t *testing.T) {
	db := openTestDB(t)
	defer db.Close()

	ctx := context.Background()
	db.CreateCollection(ctx, "cp_csv_in", libravdb.WithMetadataOnly())

	srv := startTestServer(t, db)
	defer srv.Close()

	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	sendSimpleQuery(conn, "COPY cp_csv_in (id, name, email) FROM STDIN WITH (FORMAT csv)")

	msgType, _, _ := ReadMessage(conn) // CopyInResponse
	if msgType != msgCopyInResponse {
		t.Fatalf("expected 'G', got '%c'", msgType)
	}

	// Send CSV data: id, name, email (with NULL email)
	sendCopyDataMsg(conn, "c1,Alice,alice@example.com\n")
	sendCopyDataMsg(conn, "c2,Bob,\n")                 // NULL email (unquoted empty)
	sendCopyDataMsg(conn, "\"c3\",\"Smith, John\",\n") // quoted name with comma

	WriteMessage(conn, msgCopyDone, nil)
	consumeUntilReady(t, conn)

	col, _ := db.GetCollection("cp_csv_in")

	// Verify c1.
	rec, _ := col.Get(ctx, "c1")
	if rec.Metadata["email"] != "alice@example.com" {
		t.Errorf("c1 email: %v", rec.Metadata["email"])
	}

	// Verify c2 — email is NULL.
	rec, _ = col.Get(ctx, "c2")
	if v, ok := rec.Metadata["email"]; !ok || v != nil {
		t.Errorf("c2 email: want nil, got %v", v)
	}

	// Verify c3 — quoted name with comma.
	rec, _ = col.Get(ctx, "c3")
	if rec.Metadata["name"] != "Smith, John" {
		t.Errorf("c3 name: want 'Smith, John', got %q", rec.Metadata["name"])
	}
}

// =============================================================================
// Helper: send raw CopyData message
// =============================================================================

func sendCopyDataMsg(conn net.Conn, data string) {
	WriteMessage(conn, msgCopyData, []byte(data))
}
