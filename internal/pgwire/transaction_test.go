package pgwire

import (
	"context"
	"net"
	"testing"

	"github.com/xDarkicex/lexer/parser"
)

func execSimpleTransactionTest(t *testing.T, conn net.Conn, query string, wantStatus byte) {
	t.Helper()
	sendSimpleQuery(conn, query)
	if got := readReadyStatus(t, conn); got != wantStatus {
		t.Fatalf("%s ReadyForQuery status=%q, want %q", query, got, wantStatus)
	}
}

func execExtendedCommandTest(t *testing.T, conn net.Conn, stmtName, query string, wantStatus byte) {
	t.Helper()
	sendParse(t, conn, stmtName, query, nil)
	assertMessageType(t, conn, msgParseComplete, "ParseComplete")
	sendBind(t, conn, "", stmtName, nil, nil, 0)
	assertMessageType(t, conn, msgBindComplete, "BindComplete")
	sendExecute(t, conn, "", 0)
	assertMessageType(t, conn, msgCommandComplete, "CommandComplete")
	sendSync(t, conn)
	if got := readReadyStatus(t, conn); got != wantStatus {
		t.Fatalf("extended %s ReadyForQuery status=%q, want %q", query, got, wantStatus)
	}
}

func TestPgwireTransactionBeginInsertRollback(t *testing.T) {
	db := openDescribeTestDB(t, "tx_begin_insert_rollback")
	defer db.Close()

	srv := startTestServer(t, db)
	defer srv.Close()
	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	execSimpleTransactionTest(t, conn, "BEGIN", 'T')
	execSimpleTransactionTest(t, conn, "INSERT INTO docs (id, embedding) VALUES ('rolled-back', '[1,0,0]')", 'T')

	col, err := db.GetCollection("docs")
	if err != nil {
		t.Fatalf("GetCollection: %v", err)
	}
	if _, err := col.Get(t.Context(), "rolled-back"); err == nil {
		t.Fatal("epoch insert became visible before rollback")
	}

	execSimpleTransactionTest(t, conn, "ROLLBACK", 'I')
	if _, err := col.Get(t.Context(), "rolled-back"); err == nil {
		t.Fatal("rolled-back insert is visible after rollback")
	}
}

func TestPgwireTransactionFailedStateAndRollback(t *testing.T) {
	db := openDescribeTestDB(t, "tx_failed_state")
	defer db.Close()

	srv := startTestServer(t, db)
	defer srv.Close()
	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	execSimpleTransactionTest(t, conn, "BEGIN", 'T')
	execSimpleTransactionTest(t, conn, "THIS IS NOT VALID SQL", 'E')
	execSimpleTransactionTest(t, conn, "SELECT id FROM docs", 'E')
	execSimpleTransactionTest(t, conn, "ROLLBACK", 'I')
}

func TestPgwireTransactionStartAndEpochAlias(t *testing.T) {
	t.Run("start transaction", func(t *testing.T) {
		db := openDescribeTestDB(t, "tx_start")
		defer db.Close()
		srv := startTestServer(t, db)
		defer srv.Close()
		conn := dialTestServer(t, srv)
		defer conn.Close()
		doTestStartup(t, conn)

		execSimpleTransactionTest(t, conn, "START TRANSACTION", 'T')
		execSimpleTransactionTest(t, conn, "COMMIT", 'I')
	})

	t.Run("explicit epoch alias", func(t *testing.T) {
		db := openDescribeTestDB(t, "tx_epoch_alias")
		defer db.Close()
		srv := startTestServer(t, db)
		defer srv.Close()
		conn := dialTestServer(t, srv)
		defer conn.Close()
		doTestStartup(t, conn)

		execSimpleTransactionTest(t, conn, "BEGIN EPOCH TRANSACTION", 'T')
		execSimpleTransactionTest(t, conn, "ROLLBACK", 'I')
	})
}

func TestPgwireTransactionSavepointRollbackToRelease(t *testing.T) {
	db := openDescribeTestDB(t, "tx_savepoint")
	defer db.Close()

	srv := startTestServer(t, db)
	defer srv.Close()
	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	execSimpleTransactionTest(t, conn, "BEGIN", 'T')
	execSimpleTransactionTest(t, conn, "INSERT INTO docs (id, embedding) VALUES ('before-sp', '[1,0,0]')", 'T')
	execSimpleTransactionTest(t, conn, "SAVEPOINT branch", 'T')
	execSimpleTransactionTest(t, conn, "INSERT INTO docs (id, embedding) VALUES ('after-sp', '[0,1,0]')", 'T')
	execSimpleTransactionTest(t, conn, "ROLLBACK TO branch", 'T')
	execSimpleTransactionTest(t, conn, "RELEASE SAVEPOINT branch", 'T')
	execSimpleTransactionTest(t, conn, "COMMIT", 'I')

	col, err := db.GetCollection("docs")
	if err != nil {
		t.Fatalf("GetCollection: %v", err)
	}
	if _, err := col.Get(t.Context(), "before-sp"); err != nil {
		t.Fatalf("row before savepoint missing after commit: %v", err)
	}
	if _, err := col.Get(t.Context(), "after-sp"); err == nil {
		t.Fatal("row after savepoint survived ROLLBACK TO")
	}
}

func TestPgwireExtendedTransactionStateMatchesSimple(t *testing.T) {
	db := openDescribeTestDB(t, "tx_extended")
	defer db.Close()

	srv := startTestServer(t, db)
	defer srv.Close()
	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	execExtendedCommandTest(t, conn, "begin", "BEGIN", 'T')

	// A statement error puts the connection in the failed state. The server
	// must consume Sync before accepting the cleanup command.
	sendParse(t, conn, "bad", "THIS IS NOT VALID SQL", nil)
	assertMessageType(t, conn, msgParseComplete, "ParseComplete")
	sendBind(t, conn, "", "bad", nil, nil, 0)
	assertMessageType(t, conn, msgBindComplete, "BindComplete")
	sendExecute(t, conn, "", 0)
	assertMessageType(t, conn, msgErrorResponse, "ErrorResponse")
	sendSync(t, conn)
	if got := readReadyStatus(t, conn); got != 'E' {
		t.Fatalf("failed extended transaction status=%q, want E", got)
	}

	// Normal statements are rejected while failed, but ROLLBACK is accepted.
	sendParse(t, conn, "rejected", "SELECT id FROM docs", nil)
	assertMessageType(t, conn, msgErrorResponse, "rejected ErrorResponse")
	sendSync(t, conn)
	if got := readReadyStatus(t, conn); got != 'E' {
		t.Fatalf("rejected extended transaction status=%q, want E", got)
	}
	execExtendedCommandTest(t, conn, "rollback", "ROLLBACK", 'I')
}

func TestPgwireExtendedEpochAlias(t *testing.T) {
	db := openDescribeTestDB(t, "tx_extended_epoch_alias")
	defer db.Close()

	srv := startTestServer(t, db)
	defer srv.Close()
	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	execExtendedCommandTest(t, conn, "epoch", "BEGIN EPOCH TRANSACTION", 'T')
	execExtendedCommandTest(t, conn, "rollback", "ROLLBACK", 'I')
}

func TestPgwireRollbackClosedEpochIsIdempotent(t *testing.T) {
	db := openDescribeTestDB(t, "tx_closed_epoch_cleanup")
	defer db.Close()

	epoch, err := db.BeginEpochTx(context.Background())
	if err != nil {
		t.Fatalf("BeginEpochTx: %v", err)
	}
	if err := epoch.Commit(context.Background()); err != nil {
		t.Fatalf("Commit epoch: %v", err)
	}

	state := newConnState()
	state.markTransactionStarted(epoch)
	stmt := parser.TransactionStmt{Kind: parser.TransactionRollback}
	tag, err := applyTransactionCommand(context.Background(), db, state, stmt)
	if err != nil {
		t.Fatalf("ROLLBACK closed epoch: %v", err)
	}
	if tag != "ROLLBACK" || state.txStatus() != transactionIdle || state.epoch != nil {
		t.Fatalf("closed epoch cleanup tag=%q status=%q epoch=%p", tag, state.txStatus(), state.epoch)
	}
}
