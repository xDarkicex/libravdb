package pgwire

import (
	"context"
	"encoding/binary"
	"io"
	"net"
	"testing"
	"time"

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

	// 5. Read server responses: AuthOK, ParameterStatus x3, BackendKeyData, ReadyForQuery
	for i := 0; i < 6; i++ {
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
	// Drain startup responses (AuthOK, ParameterStatus x3, BackendKeyData, ReadyForQuery)
	for i := 0; i < 6; i++ {
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

	// Describe portal
	descPayload := []byte{'P', 0} // describe portal ""
	if err := WriteMessage(conn, msgDescribe, descPayload); err != nil {
		t.Fatalf("Write Describe: %v", err)
	}

	msgType, _, err = ReadMessage(conn)
	if err != nil {
		t.Fatalf("Read Describe response: %v", err)
	}
	if msgType != msgNoData {
		t.Fatalf("expected NoData (n), got %c", msgType)
	}
	t.Log("Describe → NoData ✓")

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
