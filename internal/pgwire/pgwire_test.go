package pgwire

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"net"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/graph"
	"github.com/xDarkicex/libravdb/libravdb"
)

func TestServerStartupAndQuery(t *testing.T) {
	// 1. Create an in-memory database with a collection
	db, err := libravdb.Open(
		libravdb.WithStoragePath(":memory:pgwire_test"),
		libravdb.WithMetrics(false),
	)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	ctx := context.Background()
	_, err = db.CreateCollection(ctx, "items", libravdb.WithMetadataOnly())
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	// Insert some rows
	col, _ := db.GetCollection("items")
	col.Insert(ctx, "a", nil, nil)
	col.Insert(ctx, "b", nil, nil)

	// 2. Start pgwire server on a random port
	srv := NewServer(db, ServerConfig{Addr: "127.0.0.1:0"})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	errCh := make(chan error, 1)
	go func() {
		errCh <- srv.Serve(ctx)
	}()

	// Wait for the server to be listening
	var addr string
	for i := 0; i < 50; i++ {
		addr = srv.Addr()
		if addr != "" {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if addr == "" {
		t.Fatal("server did not start listening")
	}

	// 3. Connect as a PostgreSQL client
	conn, err := net.DialTimeout("tcp", addr, 2*time.Second)
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer conn.Close()

	// 4. Send startup packet (protocol v3, user=test, database=test)
	if err := sendStartupPacket(conn, "test", "test"); err != nil {
		t.Fatalf("sendStartupPacket: %v", err)
	}

	// 5. Read server responses: AuthOK, ParameterStatus x4, BackendKeyData, ReadyForQuery
	for i := 0; i < 7; i++ {
		msgType, _, err := ReadMessage(conn)
		if err != nil {
			t.Fatalf("expected server message %d: %v", i, err)
		}
		t.Logf("Received server message: %c", msgType)
	}

	// 6. Send a simple query (this will fail at bind time since catalog is empty,
	//    but we should get an error response in pgwire format)
	query := "SELECT 1\x00"
	if err := WriteMessage(conn, msgQuery, []byte(query)); err != nil {
		t.Fatalf("WriteMessage query: %v", err)
	}

	// 7. Read the response — could be error or result
	for i := 0; i < 10; i++ {
		msgType, payload, err := ReadMessage(conn)
		if err != nil {
			t.Fatalf("reading query response %d: %v", i, err)
		}
		t.Logf("Query response: %c (len=%d)", msgType, len(payload))

		switch msgType {
		case msgErrorResponse:
			t.Logf("Server error (expected with empty catalog): %q", string(payload))
			goto done
		case msgReadyForQuery:
			goto done
		}
	}
done:
	cancel()
	srv.Close()
	<-errCh
}

func TestServerSSLNegotiation(t *testing.T) {
	db, err := libravdb.Open(
		libravdb.WithStoragePath(":memory:ssl_test"),
		libravdb.WithMetrics(false),
	)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	srv := NewServer(db, ServerConfig{Addr: "127.0.0.1:0"})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	errCh := make(chan error, 1)
	go func() { errCh <- srv.Serve(ctx) }()

	var addr string
	for i := 0; i < 50; i++ {
		addr = srv.Addr()
		if addr != "" {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if addr == "" {
		t.Fatal("server did not start listening")
	}

	conn, err := net.DialTimeout("tcp", addr, 2*time.Second)
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer conn.Close()

	// Send SSLRequest
	var buf [8]byte
	binary.BigEndian.PutUint32(buf[0:4], 8)                      // length
	binary.BigEndian.PutUint32(buf[4:8], uint32(sslRequestCode)) // SSL request code
	if _, err := conn.Write(buf[:]); err != nil {
		t.Fatalf("Write SSLRequest: %v", err)
	}

	// Read SSL decline byte ('N')
	var sslResp [1]byte
	if _, err := io.ReadFull(conn, sslResp[:]); err != nil {
		t.Fatalf("Read SSL response: %v", err)
	}
	if sslResp[0] != 'N' {
		t.Errorf("expected SSL decline 'N', got %c", sslResp[0])
	}
	t.Log("SSL correctly declined with 'N'")

	// Now send real startup packet (after SSL decline, client retries without SSL)
	if err := sendStartupPacket(conn, "test", "test"); err != nil {
		t.Fatalf("sendStartupPacket after SSL: %v", err)
	}

	// Read server responses
	for i := 0; i < 6; i++ {
		msgType, _, err := ReadMessage(conn)
		if err != nil {
			t.Fatalf("expected server message %d after SSL: %v", i, err)
		}
		t.Logf("After SSL: %c", msgType)
	}

	cancel()
	srv.Close()
	<-errCh
}

func TestExtendedQueryProtocol(t *testing.T) {
	db, err := libravdb.Open(
		libravdb.WithStoragePath(":memory:ext_test"),
		libravdb.WithMetrics(false),
	)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	srv := NewServer(db, ServerConfig{Addr: "127.0.0.1:0"})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	errCh := make(chan error, 1)
	go func() { errCh <- srv.Serve(ctx) }()

	var addr string
	for i := 0; i < 50; i++ {
		addr = srv.Addr()
		if addr != "" {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if addr == "" {
		t.Fatal("server did not start listening")
	}

	conn, err := net.DialTimeout("tcp", addr, 2*time.Second)
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer conn.Close()

	// Startup handshake
	if err := sendStartupPacket(conn, "test", "test"); err != nil {
		t.Fatalf("sendStartupPacket: %v", err)
	}
	// Drain startup responses (AuthOK, ParameterStatus x4, BackendKeyData, ReadyForQuery)
	for i := 0; i < 7; i++ {
		if _, _, err := ReadMessage(conn); err != nil {
			t.Fatalf("startup message %d: %v", i, err)
		}
	}

	// Extended query: Parse → Bind → Describe → Execute → Sync
	// Parse: stmt name + query
	parsePayload := append(append([]byte("stmt1\x00"), []byte("SELECT 1\x00")...), 0, 0) // 0 param types
	if err := WriteMessage(conn, msgParse, parsePayload); err != nil {
		t.Fatalf("Write Parse: %v", err)
	}

	// Expect ParseComplete
	msgType, _, err := ReadMessage(conn)
	if err != nil {
		t.Fatalf("Read ParseComplete: %v", err)
	}
	if msgType != msgParseComplete {
		t.Fatalf("expected ParseComplete (1), got %c", msgType)
	}
	t.Log("Parse → ParseComplete ✓")

	// Bind: portal name + stmt name + 0 params
	bindPayload := append(append([]byte("\x00"), []byte("stmt1\x00")...), 0, 0, 0, 0) // 0 formats, 0 params, 0 result formats
	if err := WriteMessage(conn, msgBind, bindPayload); err != nil {
		t.Fatalf("Write Bind: %v", err)
	}

	msgType, _, err = ReadMessage(conn)
	if err != nil {
		t.Fatalf("Read BindComplete: %v", err)
	}
	if msgType != msgBindComplete {
		t.Fatalf("expected BindComplete (2), got %c", msgType)
	}
	t.Log("Bind → BindComplete ✓")

	// Describe portal — a bound portal reports the statement's RowDescription.
	descPayload := []byte{'P', 0} // describe portal ""
	if err := WriteMessage(conn, msgDescribe, descPayload); err != nil {
		t.Fatalf("Write Describe: %v", err)
	}

	msgType, payload, err := ReadMessage(conn)
	if err != nil {
		t.Fatalf("Read Describe response: %v", err)
	}
	if msgType != msgRowDescription {
		t.Fatalf("expected RowDescription (T) after Describe(portal), got %c", msgType)
	}
	cols := decodeRowDescription(t, payload)
	t.Logf("Describe → RowDescription with %d columns: %+v", len(cols), cols)

	// Execute portal
	execPayload := append([]byte{0}, 0, 0, 0, 0) // portal "" + maxRows=0
	if err := WriteMessage(conn, msgExecute, execPayload); err != nil {
		t.Fatalf("Write Execute: %v", err)
	}

	// Should get ErrorResponse (SELECT 1 has no FROM clause, will fail in optimizer)
	// or potentially results if catalog + collection exists
	for i := 0; i < 5; i++ {
		msgType, _, err = ReadMessage(conn)
		if err != nil {
			t.Fatalf("Execute response %d: %v", i, err)
		}
		t.Logf("Execute response: %c", msgType)
		if msgType == msgErrorResponse || msgType == msgReadyForQuery {
			break
		}
	}
	t.Log("Extended query protocol flow: Parse→Bind→Describe→Execute ✓")

	cancel()
	srv.Close()
	<-errCh
}

// sendStartupPacket sends a PostgreSQL v3 startup packet.
func sendStartupPacket(conn net.Conn, user, database string) error {
	// Build startup payload: protocol version (int32) + key=value pairs
	var payload []byte

	// Protocol version 3.0
	payload = binary.BigEndian.AppendUint32(payload, uint32(protocolVersion))

	// "user" parameter
	payload = append(payload, "user"...)
	payload = append(payload, 0)
	payload = append(payload, user...)
	payload = append(payload, 0)

	// "database" parameter
	payload = append(payload, "database"...)
	payload = append(payload, 0)
	payload = append(payload, database...)
	payload = append(payload, 0)

	// Terminating null
	payload = append(payload, 0)

	// Length prefix (includes self)
	length := 4 + len(payload)
	var lenBuf [4]byte
	binary.BigEndian.PutUint32(lenBuf[:], uint32(length))

	if _, err := conn.Write(lenBuf[:]); err != nil {
		return err
	}
	if _, err := conn.Write(payload); err != nil {
		return err
	}
	return nil
}

// =============================================================================
// COMPUTE LEIDEN pgwire tests
// =============================================================================

func TestComputeLeiden_SimpleQuery_RowDescription(t *testing.T) {
	db := openTestDB(t)
	defer db.Close()

	gr, col := createLeidenTestGraph(t, db)

	srv := startTestServer(t, db)
	defer srv.Close()

	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	// BEGIN EPOCH via simple query.
	sendSimpleQuery(conn, "BEGIN EPOCH TRANSACTION")
	// Consume response: CommandComplete, ReadyForQuery.
	consumeUntilReady(t, conn)

	// COMPUTE LEIDEN simple query.
	sendSimpleQuery(conn, "COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK*1..2]->(target)")
	_ = gr
	_ = col

	// Read RowDescription.
	msgType, payload, err := ReadMessage(conn)
	if err != nil {
		t.Fatalf("expected RowDescription: %v", err)
	}
	if msgType != 'T' {
		t.Fatalf("expected RowDescription ('T'), got '%c'", msgType)
	}

	// Decode RowDescription.
	cols := decodeRowDescription(t, payload)
	if len(cols) != 7 {
		t.Fatalf("expected 7 columns, got %d: %v", len(cols), cols)
	}

	// Verify exact column names and OIDs in order.
	expected := []struct {
		name string
		oid  uint32
	}{
		{"node_id", OIDInt8},
		{"community_id", OIDInt8},
		{"collection", OIDText},
		{"record_id", OIDText},
		{"truncated", OIDBool},
		{"scope", OIDText},
		{"modularity", OIDFloat8},
	}
	for i, exp := range expected {
		if cols[i].Name != exp.name {
			t.Errorf("column[%d] name: want %q, got %q", i, exp.name, cols[i].Name)
		}
		if cols[i].TypeOID != exp.oid {
			t.Errorf("column[%d] %q OID: want %d, got %d", i, exp.name, exp.oid, cols[i].TypeOID)
		}
	}

	// Read DataRows (stops at CommandComplete).
	rows := readDataRowsUntilComplete(t, conn)
	t.Logf("COMPUTE LEIDEN returned %d rows", len(rows))

	if len(rows) == 0 {
		t.Fatal("expected at least one DataRow")
	}

	// Verify row values.
	for _, row := range rows {
		if len(row) != 7 {
			t.Fatalf("expected 7 values per row, got %d", len(row))
		}
		// node_id: should parse as integer.
		if _, err := parseUint(row[0]); err != nil {
			t.Errorf("node_id %q: %v", row[0], err)
		}
		// community_id: should parse as integer.
		if _, err := parseUint(row[1]); err != nil {
			t.Errorf("community_id %q: %v", row[1], err)
		}
		// record_id: should be non-empty.
		if row[3] == "" {
			t.Error("record_id is empty")
		}
		// truncated: true or false.
		if row[4] != "true" && row[4] != "false" {
			t.Errorf("truncated: want true/false, got %q", row[4])
		}
		// modularity: should parse as float.
		if _, err := parseFloat(row[6]); err != nil {
			t.Errorf("modularity %q: %v", row[6], err)
		}
	}

	// Read ReadyForQuery.
	consumeReadyForQuery(t, conn)

	// Rollback.
	sendSimpleQuery(conn, "ROLLBACK")
	consumeUntilReady(t, conn)

	t.Log("✅ pgwire simple query RowDescription + DataRows")
}

func TestComputeLeiden_SimpleQuery_EmptyResult(t *testing.T) {
	db := openTestDB(t)
	defer db.Close()

	gr, _ := createLeidenTestGraph(t, db)

	srv := startTestServer(t, db)
	defer srv.Close()

	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	sendSimpleQuery(conn, "BEGIN EPOCH TRANSACTION")
	consumeUntilReady(t, conn)

	// Use MinHops=2 with no edges at depth 2 → empty result.
	sendSimpleQuery(conn, "COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK*2..2]->(target)")
	_ = gr

	// Read RowDescription — must still have all 7 columns.
	msgType, payload, err := ReadMessage(conn)
	if err != nil {
		t.Fatalf("expected RowDescription: %v", err)
	}
	if msgType != 'T' {
		t.Fatalf("expected RowDescription ('T'), got '%c'", msgType)
	}
	cols := decodeRowDescription(t, payload)
	if len(cols) != 7 {
		t.Fatalf("empty result: expected 7 columns, got %d", len(cols))
	}
	// Verify column[0] is still node_id.
	if cols[0].Name != "node_id" {
		t.Errorf("empty result column[0]: want node_id, got %q", cols[0].Name)
	}

	// Read DataRows — should be zero (stops at CommandComplete).
	rows := readDataRowsUntilComplete(t, conn)
	if len(rows) > 0 {
		t.Fatalf("empty result: expected 0 DataRows, got %d", len(rows))
	}

	consumeReadyForQuery(t, conn)
	sendSimpleQuery(conn, "ROLLBACK")
	consumeUntilReady(t, conn)

	t.Log("✅ pgwire empty result retains 7-column RowDescription")
}

func TestComputeLeiden_ExtendedQuery(t *testing.T) {
	db := openTestDB(t)
	defer db.Close()

	_, _ = createLeidenTestGraph(t, db)

	srv := startTestServer(t, db)
	defer srv.Close()

	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	// BEGIN EPOCH.
	sendSimpleQuery(conn, "BEGIN EPOCH TRANSACTION")
	consumeUntilReady(t, conn)

	// Parse.
	sendParse(t, conn, "", "COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK*1..2]->(target)", nil)
	assertMessageType(t, conn, '1', "ParseComplete")

	// Bind unnamed portal.
	sendBind(t, conn, "", "", nil, nil, 0)
	assertMessageType(t, conn, '2', "BindComplete")

	// Describe portal — a bound portal reports the statement's RowDescription.
	sendDescribe(t, conn, 'P', "")
	msgType, payload, err := ReadMessage(conn)
	if err != nil {
		t.Fatalf("expected RowDescription after Describe(portal): %v", err)
	}
	if msgType != 'T' {
		t.Fatalf("extended: expected RowDescription ('T') after Describe(portal), got '%c'", msgType)
	}
	cols := decodeRowDescription(t, payload)
	if len(cols) != 7 {
		t.Fatalf("extended: Describe(portal) expected 7 columns, got %d", len(cols))
	}
	if cols[0].Name != "node_id" || cols[0].TypeOID != OIDInt8 {
		t.Errorf("extended Describe col[0]: want node_id/OIDInt8, got %q/%d", cols[0].Name, cols[0].TypeOID)
	}

	// Execute follows a portal Describe. The RowDescription was already sent;
	// PostgreSQL emits DataRow/CommandComplete here without a duplicate T.
	sendExecute(t, conn, "", 0)

	rows := readDataRowsUntilComplete(t, conn)
	if len(rows) == 0 {
		t.Fatal("extended: expected DataRows")
	}

	// Sync.
	sendSync(t, conn)
	assertMessageType(t, conn, 'Z', "ReadyForQuery")

	// Named prepared statement and portal.
	sendParse(t, conn, "leiden_ps", "COMPUTE LEIDEN FROM MATCH (s:seeds)-[:LINK*1..2]->(target)", nil)
	assertMessageType(t, conn, '1', "ParseComplete")

	sendBind(t, conn, "leiden_portal", "leiden_ps", nil, nil, 0)
	assertMessageType(t, conn, '2', "BindComplete")

	sendExecute(t, conn, "leiden_portal", 0)
	// Extended protocol: Execute produces T + D* + C.
	// readDataRowsUntilComplete consumes 'T' (RowDescription) inline.
	rows2 := readDataRowsUntilComplete(t, conn)
	if len(rows2) == 0 {
		t.Fatal("named portal: expected DataRows")
	}

	sendSync(t, conn)
	assertMessageType(t, conn, 'Z', "ReadyForQuery")

	// Rollback.
	sendSimpleQuery(conn, "ROLLBACK")
	consumeUntilReady(t, conn)

	t.Log("✅ pgwire extended query protocol")
}

func TestComputeLeiden_OrdinarySQL_Regression(t *testing.T) {
	db := openTestDB(t)
	defer db.Close()

	ctx := context.Background()
	db.CreateCollection(ctx, "items", libravdb.WithMetadataOnly())
	col, _ := db.GetCollection("items")
	col.Insert(ctx, "a", nil, nil)
	col.Insert(ctx, "b", nil, nil)

	srv := startTestServer(t, db)
	defer srv.Close()

	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	// Ordinary SELECT — default columns are id + score.
	sendSimpleQuery(conn, "SELECT id FROM items")
	msgType, payload, err := ReadMessage(conn)
	if err != nil {
		t.Fatalf("SELECT: %v", err)
	}
	if msgType != 'T' {
		t.Fatalf("SELECT: expected RowDescription, got '%c'", msgType)
	}
	cols := decodeRowDescription(t, payload)
	// Default: id (text) + score (float8). With explicit projection "id",
	// the Columns list may be just ["id"] → 1 column is valid when
	// the optimizer populates Projections.
	if len(cols) == 0 {
		t.Fatal("SELECT: expected at least 1 column")
	}
	t.Logf("ordinary SELECT: %d columns: %+v", len(cols), cols)
	consumeUntilReady(t, conn)

	t.Log("✅ ordinary SQL regression: columns preserved")
}

// =============================================================================
// Test helpers
// =============================================================================

func openTestDB(t *testing.T) *libravdb.Database {
	t.Helper()
	db, err := libravdb.Open(
		libravdb.WithStoragePath(":memory:pgwire_leiden_test"),
		libravdb.WithMetrics(false),
	)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	return db
}

func createLeidenTestGraph(t *testing.T, db *libravdb.Database) (libravdb.Graph, *libravdb.Collection) {
	t.Helper()
	graph.RegisterEdgeKind("LINK", 10)

	gr, err := libravdb.NewGraph(libravdb.GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}

	col, err := db.CreateCollection(context.Background(), "nodes", libravdb.WithDimension(3), libravdb.WithGraph(gr))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	// Insert records and label seeds.
	col.Insert(context.Background(), "s1", []float32{1, 0, 0}, nil)
	col.Insert(context.Background(), "s2", []float32{1, 1, 0}, nil)
	col.Insert(context.Background(), "t1", []float32{0, 1, 0}, nil)
	col.Insert(context.Background(), "t2", []float32{0, 0, 1}, nil)

	s1, _ := db.GetNodeID(context.Background(), "nodes", "s1")
	s2, _ := db.GetNodeID(context.Background(), "nodes", "s2")
	t1, _ := db.GetNodeID(context.Background(), "nodes", "t1")
	t2, _ := db.GetNodeID(context.Background(), "nodes", "t2")

	gr.RegisterVertexLabel(s1, "seeds")
	gr.RegisterVertexLabel(s2, "seeds")

	txn := gr.BeginTxn()
	txn.AddEdge(s1, t1, 1.0, 10)
	txn.AddEdge(s2, t2, 1.0, 10)
	txn.Commit(context.Background())

	return gr, col
}

func startTestServer(t *testing.T, db *libravdb.Database) *Server {
	t.Helper()
	srv := NewServer(db, ServerConfig{Addr: "127.0.0.1:0"})
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	errCh := make(chan error, 1)
	go func() {
		errCh <- srv.Serve(ctx)
	}()

	for i := 0; i < 500; i++ {
		select {
		case err := <-errCh:
			t.Fatalf("server did not start: %v", err)
		default:
		}
		if srv.Addr() != "" {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if srv.Addr() == "" {
		select {
		case err := <-errCh:
			t.Fatalf("server did not start: %v", err)
		default:
			t.Fatal("server did not start")
		}
	}
	return srv
}

func dialTestServer(t *testing.T, srv *Server) net.Conn {
	t.Helper()
	conn, err := net.DialTimeout("tcp", srv.Addr(), 2*time.Second)
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	return conn
}

func doTestStartup(t *testing.T, conn net.Conn) {
	t.Helper()
	if err := sendStartupPacket(conn, "test", "test"); err != nil {
		t.Fatalf("startup: %v", err)
	}
	// AuthOK + 4×ParameterStatus + BackendKeyData + ReadyForQuery = 7 messages.
	for i := 0; i < 7; i++ {
		msgType, _, err := ReadMessage(conn)
		if err != nil {
			t.Fatalf("startup msg %d: %v", i, err)
		}
		_ = msgType
	}
}

func sendSimpleQuery(conn net.Conn, sql string) {
	WriteMessage(conn, 'Q', []byte(sql))
}

func consumeUntilReady(t *testing.T, conn net.Conn) {
	t.Helper()
	for {
		msgType, _, err := ReadMessage(conn)
		if err != nil {
			t.Fatalf("consumeUntilReady: %v", err)
		}
		if msgType == 'Z' {
			return
		}
	}
}

func consumeReadyForQuery(t *testing.T, conn net.Conn) {
	t.Helper()
	msgType, _, err := ReadMessage(conn)
	if err != nil {
		t.Fatalf("consumeReadyForQuery: %v", err)
	}
	if msgType != 'Z' {
		t.Fatalf("expected ReadyForQuery ('Z'), got '%c'", msgType)
	}
}

func assertMessageType(t *testing.T, conn net.Conn, expected byte, label string) {
	t.Helper()
	msgType, _, err := ReadMessage(conn)
	if err != nil {
		t.Fatalf("expected %s: %v", label, err)
	}
	if msgType != expected {
		t.Fatalf("expected %s ('%c'), got '%c'", label, expected, msgType)
	}
}

func decodeRowDescription(t *testing.T, payload []byte) []ColumnMeta {
	t.Helper()
	if len(payload) < 2 {
		t.Fatal("RowDescription too short")
	}
	n := int(binary.BigEndian.Uint16(payload[:2]))
	cols := make([]ColumnMeta, n)
	off := 2
	for i := 0; i < n; i++ {
		// Null-terminated name.
		nameEnd := off
		for nameEnd < len(payload) && payload[nameEnd] != 0 {
			nameEnd++
		}
		if nameEnd >= len(payload) {
			t.Fatal("unterminated column name in RowDescription")
		}
		cols[i].Name = string(payload[off:nameEnd])
		off = nameEnd + 1

		// Skip table OID (4) + attr num (2).
		off += 6

		// Type OID (4).
		cols[i].TypeOID = binary.BigEndian.Uint32(payload[off:])
		off += 4

		// Skip type size (2) + type modifier (4) + format code (2).
		off += 8
	}
	return cols
}

func readDataRowsUntilComplete(t *testing.T, conn net.Conn) [][]string {
	t.Helper()
	var rows [][]string
	for {
		msgType, payload, err := ReadMessage(conn)
		if err != nil {
			t.Fatalf("readDataRowsUntilComplete: %v", err)
		}
		switch msgType {
		case 'D':
			row := decodeDataRow(t, payload)
			rows = append(rows, row)
		case 'C':
			// CommandComplete — end of DataRows.
			return rows
		case 'T':
			// RowDescription — consume and skip (extended protocol sends
			// it inline after Execute, before DataRows).
		default:
			t.Fatalf("unexpected message '%c' while reading DataRows", msgType)
		}
	}
}

func decodeDataRow(t *testing.T, payload []byte) []string {
	t.Helper()
	if len(payload) < 2 {
		t.Fatal("DataRow too short")
	}
	n := int(binary.BigEndian.Uint16(payload[:2]))
	vals := make([]string, n)
	off := 2
	for i := 0; i < n; i++ {
		if off+4 > len(payload) {
			t.Fatal("DataRow truncated")
		}
		// Read as int32: -1 (0xFFFFFFFF) means SQL NULL.
		colLen := int32(binary.BigEndian.Uint32(payload[off:]))
		off += 4
		if colLen == -1 {
			vals[i] = "" // NULL — represented as empty string in this legacy helper
		} else {
			cl := int(colLen)
			if off+cl > len(payload) {
				t.Fatal("DataRow column truncated")
			}
			vals[i] = string(payload[off : off+cl])
			off += cl
		}
	}
	return vals
}

func parseUint(s string) (uint64, error) {
	var n uint64
	for _, c := range s {
		if c < '0' || c > '9' {
			return 0, fmt.Errorf("not a uint: %q", s)
		}
		n = n*10 + uint64(c-'0')
	}
	return n, nil
}

func parseFloat(s string) (float64, error) {
	var f float64
	_, err := fmt.Sscanf(s, "%f", &f)
	return f, err
}

// Extended query protocol helpers.

func sendParse(t *testing.T, conn net.Conn, stmtName, query string, paramOIDs []uint32) {
	t.Helper()
	var buf []byte
	// Statement name (null-terminated).
	buf = append(buf, stmtName...)
	buf = append(buf, 0)
	// Query string (null-terminated).
	buf = append(buf, query...)
	buf = append(buf, 0)
	// Number of param OIDs.
	if paramOIDs == nil {
		buf = append(buf, 0, 0)
	} else {
		panic("param OIDs not yet supported")
	}
	WriteMessage(conn, 'P', buf)
}

func sendBind(t *testing.T, conn net.Conn, portal, stmtName string, paramFormats, resultFormats []uint16, numParams uint16) {
	t.Helper()
	var buf []byte
	buf = append(buf, portal...)
	buf = append(buf, 0)
	buf = append(buf, stmtName...)
	buf = append(buf, 0)
	// Number of param format codes.
	off := len(buf)
	buf = append(buf, 0, 0)
	binary.BigEndian.PutUint16(buf[off:], numParams)
	// Param format codes (0=text for each).
	for i := uint16(0); i < numParams; i++ {
		buf = append(buf, 0, 0)
	}
	// Number of param values (0).
	buf = append(buf, 0, 0)
	// Number of result format codes (0 = all text).
	buf = append(buf, 0, 0)
	WriteMessage(conn, 'B', buf)
	_ = paramFormats
	_ = resultFormats
}

func sendDescribe(t *testing.T, conn net.Conn, kind byte, name string) {
	t.Helper()
	var buf []byte
	buf = append(buf, kind)
	buf = append(buf, name...)
	buf = append(buf, 0)
	WriteMessage(conn, 'D', buf)
}

func sendExecute(t *testing.T, conn net.Conn, portal string, maxRows uint32) {
	t.Helper()
	var buf []byte
	buf = append(buf, portal...)
	buf = append(buf, 0)
	off := len(buf)
	buf = append(buf, 0, 0, 0, 0)
	binary.BigEndian.PutUint32(buf[off:], maxRows)
	WriteMessage(conn, 'E', buf)
}

func sendSync(t *testing.T, conn net.Conn) {
	t.Helper()
	WriteMessage(conn, 'S', nil)
}
