package pgwire

import (
	"bytes"
	"context"
	"fmt"
	"testing"

	"github.com/xDarkicex/libravdb/libravdb"
)

func TestLatestCommitLSNFunctionAndStartupMetadata(t *testing.T) {
	db, err := libravdb.Open(
		libravdb.WithStoragePath(":memory:pgwire_snapshot_lsn"),
		libravdb.WithMetrics(false),
	)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	if _, err := db.Query(context.Background(), `CREATE TABLE docs (id TEXT PRIMARY KEY)`); err != nil {
		t.Fatalf("CREATE TABLE: %v", err)
	}

	want, err := db.LatestCommitLSN(context.Background())
	if err != nil {
		t.Fatalf("LatestCommitLSN: %v", err)
	}
	results, columns, handled := interceptSystemQuery(`SELECT LIBRAVDB_LATEST_COMMIT_LSN()`, db)
	if !handled || len(columns) != 1 || columns[0].Name != "libravdb_latest_commit_lsn" || columns[0].TypeOID != OIDInt8 {
		t.Fatalf("SQL function was not exposed as int8: handled=%v columns=%#v", handled, columns)
	}
	value, ok := results.Results[0].Metadata["libravdb_latest_commit_lsn"]
	if !ok || value != fmt.Sprint(want) {
		t.Fatalf("SQL function value=%#v, want %d", value, want)
	}

	srv := startTestServer(t, db)
	defer srv.Close()
	conn := dialTestServer(t, srv)
	defer conn.Close()
	if err := sendStartupPacket(conn, "test", "test"); err != nil {
		t.Fatalf("startup: %v", err)
	}
	var startupLSN string
	for i := 0; i < 8; i++ {
		msgType, payload, err := ReadMessage(conn)
		if err != nil {
			t.Fatalf("startup message %d: %v", i, err)
		}
		if msgType != msgParameterStatus {
			continue
		}
		keyEnd := bytes.IndexByte(payload, 0)
		if keyEnd < 0 || keyEnd+1 >= len(payload) {
			continue
		}
		valueEnd := bytes.IndexByte(payload[keyEnd+1:], 0)
		if valueEnd < 0 {
			continue
		}
		key := string(payload[:keyEnd])
		if key == "libravdb_latest_commit_lsn" {
			startupLSN = string(payload[keyEnd+1 : keyEnd+1+valueEnd])
		}
	}
	if startupLSN != fmt.Sprint(want) {
		t.Fatalf("startup LSN=%q, want %d", startupLSN, want)
	}
}
