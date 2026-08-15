package pgwire

import (
	"bytes"
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/libravdb"
)

type temporalWireFixture struct {
	db      *libravdb.Database
	oldTime time.Time
	oldID   string
}

func newTemporalWireFixture(t *testing.T) temporalWireFixture {
	t.Helper()
	db, err := libravdb.Open(
		libravdb.WithStoragePath(t.TempDir()+"/temporal-wire.libravdb"),
		libravdb.WithMetrics(false),
	)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	col, err := db.CreateCollection(context.Background(), "docs", libravdb.WithDimension(3))
	if err != nil {
		db.Close()
		t.Fatalf("CreateCollection: %v", err)
	}
	if err := col.Insert(context.Background(), "old", []float32{1, 0, 0}, nil); err != nil {
		db.Close()
		t.Fatalf("insert old: %v", err)
	}
	snap, err := db.SnapshotAt(context.Background(), time.Now().UTC().Add(time.Second))
	if err != nil {
		db.Close()
		t.Fatalf("SnapshotAt: %v", err)
	}
	oldTime := snap.Timestamp
	snap.Close()
	if err := col.Insert(context.Background(), "future", []float32{0, 0, 0}, nil); err != nil {
		db.Close()
		t.Fatalf("insert future: %v", err)
	}
	return temporalWireFixture{db: db, oldTime: oldTime, oldID: "old"}
}

func temporalWireQuery(ts time.Time) string {
	return fmt.Sprintf("SELECT id FROM docs AS OF TIMESTAMP '%s' ORDER BY id LIMIT 10", ts.Format(time.RFC3339Nano))
}

func readReadyStatus(t *testing.T, conn interface{ Read([]byte) (int, error) }) byte {
	t.Helper()
	for {
		msgType, payload, err := ReadMessage(conn)
		if err != nil {
			t.Fatalf("read ReadyForQuery: %v", err)
		}
		if msgType == msgReadyForQuery {
			if len(payload) != 1 {
				t.Fatalf("ReadyForQuery payload length=%d, want 1", len(payload))
			}
			return payload[0]
		}
	}
}

func TestTemporalAcceptance_PgwireSimpleProtocol(t *testing.T) {
	fixture := newTemporalWireFixture(t)
	defer fixture.db.Close()

	srv := startTestServer(t, fixture.db)
	defer srv.Close()
	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	sendSimpleQuery(conn, temporalWireQuery(fixture.oldTime))
	msgType, payload, err := ReadMessage(conn)
	if err != nil {
		t.Fatalf("simple temporal RowDescription: %v", err)
	}
	if msgType != msgRowDescription {
		t.Fatalf("simple temporal first message=%q, want RowDescription", msgType)
	}
	cols := decodeRowDescription(t, payload)
	if len(cols) != 1 || cols[0].Name != "id" {
		t.Fatalf("simple temporal columns=%+v, want id", cols)
	}
	rows := readDataRowsUntilComplete(t, conn)
	if len(rows) != 1 || len(rows[0]) != 1 || rows[0][0] != fixture.oldID {
		t.Fatalf("simple temporal rows=%v, want [[%s]]", rows, fixture.oldID)
	}
	status := readReadyStatus(t, conn)
	if status != 'I' {
		t.Fatalf("simple temporal ReadyForQuery status=%q, want idle", status)
	}
}

func TestTemporalAcceptance_PgwireExtendedProtocol(t *testing.T) {
	fixture := newTemporalWireFixture(t)
	defer fixture.db.Close()

	srv := startTestServer(t, fixture.db)
	defer srv.Close()
	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	sendParse(t, conn, "historical", temporalWireQuery(fixture.oldTime), nil)
	assertMessageType(t, conn, msgParseComplete, "temporal ParseComplete")
	sendBind(t, conn, "historical-portal", "historical", nil, nil, 0)
	assertMessageType(t, conn, msgBindComplete, "temporal BindComplete")
	sendExecute(t, conn, "historical-portal", 0)
	rows := readDataRowsUntilComplete(t, conn)
	if len(rows) != 1 || len(rows[0]) != 1 || rows[0][0] != fixture.oldID {
		t.Fatalf("extended temporal rows=%v, want [[%s]]", rows, fixture.oldID)
	}
	// Extended protocol emits ReadyForQuery only after Sync.
	sendSync(t, conn)
	status := readReadyStatus(t, conn)
	if status != 'I' {
		t.Fatalf("extended temporal ReadyForQuery status=%q, want idle", status)
	}
}

func TestTemporalAcceptance_PgwireVersionsRange(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(t.TempDir()+"/versions-wire.libravdb"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	col, err := db.CreateCollection(ctx, "docs", libravdb.WithDimension(3), libravdb.WithMetadataSchema(libravdb.MetadataSchema{"title": libravdb.StringField}))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	if err := col.Insert(ctx, "d1", []float32{1, 0, 0}, map[string]interface{}{"title": "first"}); err != nil {
		t.Fatalf("insert: %v", err)
	}
	startSnap, err := db.SnapshotAt(ctx, time.Now().UTC().Add(time.Second))
	if err != nil {
		t.Fatalf("start snapshot: %v", err)
	}
	start := startSnap.Timestamp
	startSnap.Close()
	if err := col.Update(ctx, "d1", []float32{0, 1, 0}, map[string]interface{}{"title": "second"}); err != nil {
		t.Fatalf("update: %v", err)
	}
	endSnap, err := db.SnapshotAt(ctx, time.Now().UTC().Add(time.Second))
	if err != nil {
		t.Fatalf("end snapshot: %v", err)
	}
	end := endSnap.Timestamp
	endSnap.Close()

	srv := startTestServer(t, db)
	defer srv.Close()
	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)
	query := fmt.Sprintf("SELECT id, version, title FROM VERSIONS OF docs BETWEEN TIMESTAMP '%s' AND TIMESTAMP '%s' ORDER BY version", start.Format(time.RFC3339Nano), end.Format(time.RFC3339Nano))
	sendSimpleQuery(conn, query)
	msgType, payload, err := ReadMessage(conn)
	if err != nil {
		t.Fatalf("versions RowDescription: %v", err)
	}
	if msgType != msgRowDescription {
		t.Fatalf("versions first message=%q, want RowDescription", msgType)
	}
	cols := decodeRowDescription(t, payload)
	if len(cols) != 3 || cols[0].Name != "id" || cols[1].Name != "version" || cols[2].Name != "title" {
		t.Fatalf("versions columns=%+v", cols)
	}
	rows := readDataRowsUntilComplete(t, conn)
	if len(rows) != 2 || rows[0][0] != "d1" || rows[0][1] != "1" || rows[0][2] != "first" || rows[1][1] != "2" || rows[1][2] != "second" {
		t.Fatalf("versions rows=%v", rows)
	}
	if status := readReadyStatus(t, conn); status != 'I' {
		t.Fatalf("ReadyForQuery status=%q, want idle", status)
	}
}

// TestTemporalAcceptance_PgwireReadyForQueryHistoricalEpoch pins a wire
// connection's epoch at an older timestamp directly. pgwire does not expose
// a BEGIN EPOCH AT TIMESTAMP command, but the connection state still must
// preserve the in-transaction ReadyForQuery status for such an epoch.
func TestTemporalAcceptance_PgwireReadyForQueryHistoricalEpoch(t *testing.T) {
	fixture := newTemporalWireFixture(t)
	defer fixture.db.Close()

	epoch, err := fixture.db.BeginEpochTxAt(context.Background(), fixture.oldTime)
	if err != nil {
		t.Fatalf("BeginEpochTxAt: %v", err)
	}
	state := newConnState()
	state.epoch = epoch
	defer state.rollbackEpoch()

	var simple bytes.Buffer
	if err := handleQuery(&simple, fixture.db, state, "SELECT id FROM docs"); err != nil {
		t.Fatalf("historical simple query: %v", err)
	}
	status := readReadyStatus(t, &simple)
	if status != 'T' {
		t.Errorf("historical simple ReadyForQuery status=%q, want in-transaction", status)
	}

	// The extended path sends the state-bearing ReadyForQuery on Sync.
	state.rollbackEpoch()
	epoch, err = fixture.db.BeginEpochTxAt(context.Background(), fixture.oldTime)
	if err != nil {
		t.Fatalf("BeginEpochTxAt extended: %v", err)
	}
	state.epoch = epoch
	var extended bytes.Buffer
	stmtPayload := append([]byte("historical\x00"), []byte("SELECT id FROM docs\x00")...)
	stmtPayload = append(stmtPayload, 0, 0)
	if err := handleParse(&extended, state, stmtPayload); err != nil {
		t.Fatalf("historical extended parse: %v", err)
	}
	if _, _, err := ReadMessage(&extended); err != nil {
		t.Fatalf("read ParseComplete: %v", err)
	}
	bindPayload := []byte("portal\x00historical\x00\x00\x00\x00\x00")
	if err := handleBind(&extended, state, bindPayload); err != nil {
		t.Fatalf("historical extended bind: %v", err)
	}
	if _, _, err := ReadMessage(&extended); err != nil {
		t.Fatalf("read BindComplete: %v", err)
	}
	executePayload := []byte("portal\x00\x00\x00\x00\x00")
	if err := handleExecute(&extended, fixture.db, state, executePayload); err != nil {
		t.Fatalf("historical extended execute: %v", err)
	}
	// Drain RowDescription/DataRows/CommandComplete before Sync.
	for {
		msgType, _, readErr := ReadMessage(&extended)
		if readErr != nil {
			t.Fatalf("drain extended historical result: %v", readErr)
		}
		if msgType == msgCommandComplete {
			break
		}
	}
	if err := handleSync(&extended, state); err != nil {
		t.Fatalf("historical extended sync: %v", err)
	}
	status = readReadyStatus(t, &extended)
	if status != 'T' {
		t.Fatalf("historical extended ReadyForQuery status=%q, want in-transaction", status)
	}
}
