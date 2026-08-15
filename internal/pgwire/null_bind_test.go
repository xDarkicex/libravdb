package pgwire

import (
	"encoding/binary"
	"net"
	"testing"

	"github.com/xDarkicex/libravdb/libravdb"
)

func openNullableParameterDB(t *testing.T, name string) *libravdb.Database {
	t.Helper()
	db, err := libravdb.Open(
		libravdb.WithStoragePath(":memory:pgwire_null_"+name),
		libravdb.WithMetrics(false),
	)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if _, err := db.Query(t.Context(), `
		CREATE TABLE nullable_wire (
			id TEXT PRIMARY KEY,
			note TEXT,
			required TEXT NOT NULL,
			defaulted TEXT NOT NULL DEFAULT 'default'
		)`); err != nil {
		db.Close()
		t.Fatalf("CREATE TABLE: %v", err)
	}
	if _, err := db.Query(t.Context(),
		"INSERT INTO nullable_wire (id, note, required) VALUES ('seed', 'before', 'ok')"); err != nil {
		db.Close()
		t.Fatalf("seed insert: %v", err)
	}
	return db
}

func sendBindValues(t *testing.T, conn net.Conn, portal, stmt string, formats []int16, values [][]byte) {
	t.Helper()
	buf := make([]byte, 0, 64)
	buf = append(buf, portal...)
	buf = append(buf, 0)
	buf = append(buf, stmt...)
	buf = append(buf, 0)

	countOffset := len(buf)
	buf = append(buf, 0, 0)
	binary.BigEndian.PutUint16(buf[countOffset:], uint16(len(formats)))
	for _, format := range formats {
		if format != 0 && format != 1 {
			t.Fatalf("unsupported test format %d", format)
		}
		buf = append(buf, byte(format>>8), byte(format))
	}

	buf = append(buf, byte(len(values)>>8), byte(len(values)))
	for _, value := range values {
		off := len(buf)
		buf = append(buf, 0, 0, 0, 0)
		if value == nil {
			binary.BigEndian.PutUint32(buf[off:], ^uint32(0))
			continue
		}
		binary.BigEndian.PutUint32(buf[off:], uint32(len(value)))
		buf = append(buf, value...)
	}
	// All result columns use text format.
	buf = append(buf, 0, 0)
	if err := WriteMessage(conn, msgBind, buf); err != nil {
		t.Fatalf("Bind: %v", err)
	}
}

func readNullableReadyStatus(t *testing.T, conn net.Conn) byte {
	t.Helper()
	for {
		msgType, payload, err := ReadMessage(conn)
		if err != nil {
			t.Fatalf("ReadyForQuery: %v", err)
		}
		if msgType == msgReadyForQuery {
			if len(payload) != 1 {
				t.Fatalf("ReadyForQuery payload length=%d", len(payload))
			}
			return payload[0]
		}
	}
}

func readNullableSimpleRows(t *testing.T, conn net.Conn) [][]*string {
	t.Helper()
	var rows [][]*string
	for {
		msgType, payload, err := ReadMessage(conn)
		if err != nil {
			t.Fatalf("simple result: %v", err)
		}
		switch msgType {
		case msgDataRow:
			_, values, err := decodeDataRowNullable(payload)
			if err != nil {
				t.Fatalf("decode simple DataRow: %v", err)
			}
			rows = append(rows, values)
		case msgReadyForQuery:
			return rows
		case msgErrorResponse:
			t.Fatalf("unexpected simple ErrorResponse")
		}
	}
}

func runNullableExtended(t *testing.T, conn net.Conn, stmtName, query string, formats []int16, values [][]byte) ([][]*string, byte) {
	t.Helper()
	sendParse(t, conn, stmtName, query, nil)
	assertMessageType(t, conn, msgParseComplete, "ParseComplete")
	sendBindValues(t, conn, "", stmtName, formats, values)
	assertMessageType(t, conn, msgBindComplete, "BindComplete")
	sendExecute(t, conn, "", 0)

	var rows [][]*string
	for {
		msgType, payload, err := ReadMessage(conn)
		if err != nil {
			t.Fatalf("extended result: %v", err)
		}
		switch msgType {
		case msgRowDescription, msgCommandComplete, msgEmptyQuery:
			if msgType == msgCommandComplete || msgType == msgEmptyQuery {
				goto sync
			}
		case msgDataRow:
			_, values, err := decodeDataRowNullable(payload)
			if err != nil {
				t.Fatalf("decode extended DataRow: %v", err)
			}
			rows = append(rows, values)
		case msgErrorResponse:
			t.Fatalf("unexpected extended ErrorResponse for %s", query)
		}
	}

sync:
	sendSync(t, conn)
	return rows, readNullableReadyStatus(t, conn)
}

func runNullableExtendedExpectError(t *testing.T, conn net.Conn, stmtName, query string, formats []int16, values [][]byte) byte {
	t.Helper()
	sendParse(t, conn, stmtName, query, nil)
	assertMessageType(t, conn, msgParseComplete, "ParseComplete")
	sendBindValues(t, conn, "", stmtName, formats, values)
	assertMessageType(t, conn, msgBindComplete, "BindComplete")
	sendExecute(t, conn, "", 0)
	assertMessageType(t, conn, msgErrorResponse, "ErrorResponse")
	sendSync(t, conn)
	return readNullableReadyStatus(t, conn)
}

func TestPgwireBindNullEmptyAndLiteralStringRemainDistinct(t *testing.T) {
	db := openNullableParameterDB(t, "distinct")
	defer db.Close()
	srv := startTestServer(t, db)
	defer srv.Close()
	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	insert := "INSERT INTO nullable_wire (id, note, required) VALUES ($1, $2, $3)"
	runNullableExtended(t, conn, "insert-null", insert, nil,
		[][]byte{[]byte("null-row"), nil, []byte("ok")})
	// A binary-format NULL has the same native SQL meaning as a text NULL.
	runNullableExtended(t, conn, "insert-binary-null", insert, []int16{1, 0, 0},
		[][]byte{[]byte("binary-null-row"), nil, []byte("ok")})
	runNullableExtended(t, conn, "insert-empty", insert, nil,
		[][]byte{[]byte("empty-row"), []byte{}, []byte("ok")})
	runNullableExtended(t, conn, "insert-literal-null", insert, nil,
		[][]byte{[]byte("literal-null-row"), []byte("NULL"), []byte("ok")})

	col, err := db.GetCollection("nullable_wire")
	if err != nil {
		t.Fatalf("GetCollection: %v", err)
	}
	checks := []struct {
		id       string
		wantNull bool
		want     string
	}{
		{"null-row", true, ""},
		{"binary-null-row", true, ""},
		{"empty-row", false, ""},
		{"literal-null-row", false, "NULL"},
	}
	for _, check := range checks {
		record, err := col.Get(t.Context(), check.id)
		if err != nil {
			t.Fatalf("Get %s: %v", check.id, err)
		}
		value, ok := record.Metadata["note"]
		if check.wantNull {
			if !ok || value != nil {
				t.Errorf("%s note=%#v, want native NULL", check.id, value)
			}
		} else if !ok || value != check.want {
			t.Errorf("%s note=%#v, want %q", check.id, value, check.want)
		}
	}

	rows, status := runNullableExtended(t, conn, "select-null", "SELECT id, note FROM nullable_wire WHERE note IS NULL", nil, nil)
	if status != 'I' || len(rows) != 2 {
		t.Fatalf("IS NULL rows=%d status=%q, want 2/I", len(rows), status)
	}
	for _, row := range rows {
		if len(row) != 2 || row[1] != nil {
			t.Fatalf("IS NULL row=%v, want a NULL note", row)
		}
	}
	rows, status = runNullableExtended(t, conn, "select-not-null", "SELECT id, note FROM nullable_wire WHERE note IS NOT NULL", nil, nil)
	if status != 'I' || len(rows) != 3 {
		t.Fatalf("IS NOT NULL rows=%d status=%q, want 3/I", len(rows), status)
	}
	rows, status = runNullableExtended(t, conn, "select-null-equals", "SELECT id FROM nullable_wire WHERE note = $1", nil, [][]byte{nil})
	if status != 'I' || len(rows) != 0 {
		t.Fatalf("NULL equality rows=%d status=%q, want 0/I", len(rows), status)
	}

	rows, status = runNullableExtended(t, conn, "select-empty", "SELECT id, note FROM nullable_wire WHERE note = $1", nil, [][]byte{[]byte{}})
	if status != 'I' || len(rows) != 1 || rows[0][1] == nil || *rows[0][1] != "" {
		t.Fatalf("empty equality rows=%v status=%q, want one empty-string row", rows, status)
	}
	rows, status = runNullableExtended(t, conn, "select-literal", "SELECT id, note FROM nullable_wire WHERE note = $1", nil, [][]byte{[]byte("NULL")})
	if status != 'I' || len(rows) != 1 || rows[0][1] == nil || *rows[0][1] != "NULL" {
		t.Fatalf("literal equality rows=%v status=%q, want one literal row", rows, status)
	}
}

func TestPgwireNullableInsertUpdateConstraintsAndInjection(t *testing.T) {
	db := openNullableParameterDB(t, "constraints")
	defer db.Close()
	srv := startTestServer(t, db)
	defer srv.Close()
	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	insert := "INSERT INTO nullable_wire (id, note, required) VALUES ($1, $2, $3)"
	if status := runNullableExtendedExpectError(t, conn, "not-null", insert, nil,
		[][]byte{[]byte("bad-required"), []byte("note"), nil}); status != 'I' {
		t.Fatalf("NOT NULL error status=%q, want I", status)
	}
	if _, err := db.GetCollection("nullable_wire"); err != nil {
		t.Fatalf("collection after rejected insert: %v", err)
	}

	// Empty strings satisfy NOT NULL, and omitted defaulted columns receive
	// their configured default before validation.
	runNullableExtended(t, conn, "empty-required", insert, nil,
		[][]byte{[]byte("empty-required"), []byte{}, []byte{}})
	runNullableExtended(t, conn, "defaulted", "INSERT INTO nullable_wire (id, note, required) VALUES ($1, $2, $3)", nil,
		[][]byte{[]byte("defaulted"), nil, []byte("ok")})
	defaulted, err := db.GetCollection("nullable_wire")
	if err != nil {
		t.Fatalf("GetCollection for defaulted row: %v", err)
	}
	defaultedRecord, err := defaulted.Get(t.Context(), "defaulted")
	if err != nil {
		t.Fatalf("Get defaulted: %v", err)
	}
	if value := defaultedRecord.Metadata["defaulted"]; value != "default" {
		t.Fatalf("defaulted column=%#v, want %q", value, "default")
	}
	if status := runNullableExtendedExpectError(t, conn, "missing-required", "INSERT INTO nullable_wire (id, note) VALUES ($1, $2)", nil,
		[][]byte{[]byte("missing-required"), nil}); status != 'I' {
		t.Fatalf("missing required status=%q, want I", status)
	}

	if _, err := db.Query(t.Context(), "CREATE TABLE checked_wire (id TEXT PRIMARY KEY, note TEXT, CHECK (note IS NOT NULL))"); err != nil {
		t.Fatalf("CREATE CHECK table: %v", err)
	}
	if status := runNullableExtendedExpectError(t, conn, "check-null", "INSERT INTO checked_wire (id, note) VALUES ($1, $2)", nil,
		[][]byte{[]byte("check-null"), nil}); status != 'I' {
		t.Fatalf("CHECK NULL status=%q, want I", status)
	}

	if _, err := db.Query(t.Context(), "CREATE TABLE parent_wire (id TEXT PRIMARY KEY)"); err != nil {
		t.Fatalf("CREATE parent table: %v", err)
	}
	if _, err := db.Query(t.Context(), "INSERT INTO parent_wire (id) VALUES ('parent')"); err != nil {
		t.Fatalf("parent insert: %v", err)
	}
	if _, err := db.Query(t.Context(), "CREATE TABLE child_wire (id TEXT PRIMARY KEY, parent_id TEXT, FOREIGN KEY (parent_id) REFERENCES parent_wire (id))"); err != nil {
		t.Fatalf("CREATE child table: %v", err)
	}
	runNullableExtended(t, conn, "fk-null", "INSERT INTO child_wire (id, parent_id) VALUES ($1, $2)", nil,
		[][]byte{[]byte("child-null"), nil})
	if _, err := db.GetCollection("child_wire"); err != nil {
		t.Fatalf("child collection after nullable FK insert: %v", err)
	}

	if _, err := db.Query(t.Context(), "CREATE TABLE vector_wire (id TEXT PRIMARY KEY, embedding VECTOR(3))"); err != nil {
		t.Fatalf("CREATE vector table: %v", err)
	}
	if status := runNullableExtendedExpectError(t, conn, "vector-null", "INSERT INTO vector_wire (id, embedding) VALUES ($1, $2)", nil,
		[][]byte{[]byte("vector-null"), nil}); status != 'I' {
		t.Fatalf("vector NULL status=%q, want I", status)
	}

	runNullableExtended(t, conn, "update-null", "UPDATE nullable_wire SET note = $1 WHERE id = $2", nil,
		[][]byte{nil, []byte("seed")})
	seed, err := db.GetCollection("nullable_wire")
	if err != nil {
		t.Fatalf("GetCollection: %v", err)
	}
	seedRecord, err := seed.Get(t.Context(), "seed")
	if err != nil {
		t.Fatalf("Get seed: %v", err)
	}
	if value, ok := seedRecord.Metadata["note"]; !ok || value != nil {
		t.Fatalf("updated seed note=%#v, want native NULL", value)
	}

	injection := "x'); INSERT INTO nullable_wire (id, required) VALUES ('injected', 'bad'); --"
	runNullableExtended(t, conn, "injection", insert, nil,
		[][]byte{[]byte("injection"), []byte(injection), []byte("ok")})
	injected, err := seed.Get(t.Context(), "injection")
	if err != nil || injected.Metadata["note"] != injection {
		t.Fatalf("injection text was not preserved: record=%#v err=%v", injected, err)
	}
	if _, err := seed.Get(t.Context(), "injected"); err == nil {
		t.Fatal("bound injection text altered the SQL statement")
	}
}

func TestPgwireEpochNullableMutationAndSavepointRollback(t *testing.T) {
	db := openNullableParameterDB(t, "epoch")
	defer db.Close()
	srv := startTestServer(t, db)
	defer srv.Close()
	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	execSimpleTransactionTest(t, conn, "BEGIN", 'T')
	insert := "INSERT INTO nullable_wire (id, note, required) VALUES ($1, $2, $3)"
	_, status := runNullableExtended(t, conn, "epoch-insert", insert, nil,
		[][]byte{[]byte("epoch-bound-null"), nil, []byte("ok")})
	if status != 'T' {
		t.Fatalf("epoch insert status=%q, want T", status)
	}
	_, status = runNullableExtended(t, conn, "epoch-insert-before", insert, nil,
		[][]byte{[]byte("epoch-null"), []byte("before"), []byte("ok")})
	if status != 'T' {
		t.Fatalf("epoch second insert status=%q, want T", status)
	}

	execSimpleTransactionTest(t, conn, "SAVEPOINT null_branch", 'T')
	_, status = runNullableExtended(t, conn, "epoch-update-null", "UPDATE nullable_wire SET note = $1 WHERE id = $2", nil,
		[][]byte{nil, []byte("epoch-null")})
	if status != 'T' {
		t.Fatalf("epoch update status=%q, want T", status)
	}
	execSimpleTransactionTest(t, conn, "ROLLBACK TO null_branch", 'T')
	execSimpleTransactionTest(t, conn, "RELEASE SAVEPOINT null_branch", 'T')
	execSimpleTransactionTest(t, conn, "COMMIT", 'I')

	col, err := db.GetCollection("nullable_wire")
	if err != nil {
		t.Fatalf("GetCollection: %v", err)
	}
	record, err := col.Get(t.Context(), "epoch-null")
	if err != nil {
		t.Fatalf("Get epoch-null: %v", err)
	}
	if value, ok := record.Metadata["note"]; !ok || value != "before" {
		t.Fatalf("savepoint rollback retained NULL mutation: note=%#v", value)
	}
	boundNull, err := col.Get(t.Context(), "epoch-bound-null")
	if err != nil {
		t.Fatalf("Get epoch-bound-null: %v", err)
	}
	if value, ok := boundNull.Metadata["note"]; !ok || value != nil {
		t.Fatalf("epoch NULL insert note=%#v, want native NULL", value)
	}

	// The NULL insert itself was not discarded by the savepoint and remains
	// distinguishable after commit.
	// Update a committed row through a second epoch to verify native NULL in
	// an epoch UPDATE as well.
	execSimpleTransactionTest(t, conn, "BEGIN EPOCH", 'T')
	runNullableExtended(t, conn, "epoch-update-committed", "UPDATE nullable_wire SET note = $1 WHERE id = $2", nil,
		[][]byte{nil, []byte("seed")})
	execSimpleTransactionTest(t, conn, "COMMIT", 'I')
	seed, err := col.Get(t.Context(), "seed")
	if err != nil {
		t.Fatalf("Get updated seed: %v", err)
	}
	if value, ok := seed.Metadata["note"]; !ok || value != nil {
		t.Fatalf("epoch NULL update note=%#v, want native NULL", value)
	}
}

func TestPgwireSimpleAndExtendedNullResultEncoding(t *testing.T) {
	db := openNullableParameterDB(t, "results")
	defer db.Close()
	srv := startTestServer(t, db)
	defer srv.Close()
	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	sendSimpleQuery(conn, "SELECT id, note FROM nullable_wire WHERE id = 'seed'")
	simpleRows := readNullableSimpleRows(t, conn)
	if len(simpleRows) != 1 || len(simpleRows[0]) != 2 || simpleRows[0][1] == nil || *simpleRows[0][1] != "before" {
		t.Fatalf("simple result rows=%v", simpleRows)
	}

	// Make a NULL-valued row and verify the wire representation in both paths.
	insert := "INSERT INTO nullable_wire (id, note, required) VALUES ($1, $2, $3)"
	runNullableExtended(t, conn, "insert-result-null", insert, nil,
		[][]byte{[]byte("result-null"), nil, []byte("ok")})
	sendSimpleQuery(conn, "SELECT id, note FROM nullable_wire WHERE id = 'result-null'")
	simpleRows = readNullableSimpleRows(t, conn)
	if len(simpleRows) != 1 || simpleRows[0][1] != nil {
		t.Fatalf("simple NULL result rows=%v", simpleRows)
	}

	extendedRows, status := runNullableExtended(t, conn, "select-result-null", "SELECT id, note FROM nullable_wire WHERE id = $1", nil,
		[][]byte{[]byte("result-null")})
	if status != 'I' || len(extendedRows) != 1 || extendedRows[0][1] != nil {
		t.Fatalf("extended NULL result rows=%v status=%q", extendedRows, status)
	}
}
