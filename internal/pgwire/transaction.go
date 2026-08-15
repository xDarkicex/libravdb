package pgwire

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/xDarkicex/lexer/parser"
	"github.com/xDarkicex/libravdb/libravdb"
)

var errCurrentTransactionAborted = fmt.Errorf("current transaction is aborted, commands ignored until end of transaction block")

func (s *connState) txStatus() transactionState {
	if s == nil {
		return transactionIdle
	}
	if s.transactionState == transactionFailed {
		return transactionFailed
	}
	if s.epoch != nil || s.transactionState == transactionInProgress {
		return transactionInProgress
	}
	return transactionIdle
}

func (s *connState) markTransactionStarted(epoch *libravdb.EpochTx) {
	s.epoch = epoch
	s.transactionState = transactionInProgress
}

func (s *connState) markTransactionFailed() {
	if s.txStatus() != transactionIdle {
		s.transactionState = transactionFailed
	}
}

func (s *connState) clearTransaction() {
	s.epoch = nil
	s.transactionState = transactionIdle
}

func (s *connState) readyStatus() byte {
	switch s.txStatus() {
	case transactionInProgress:
		return 'T'
	case transactionFailed:
		return 'E'
	default:
		return 'I'
	}
}

// parsePgwireTransactionControl parses only a single transaction statement.
// All protocol paths call this helper, so BEGIN variants and savepoints cannot
// diverge between simple and extended execution.
func parsePgwireTransactionControl(sql string) (parser.TransactionStmt, bool, error) {
	sql = strings.TrimSpace(strings.TrimRight(sql, ";"))
	var doc parser.QueryDoc
	if err := parser.Parse([]byte(sql), &doc); err != nil {
		return parser.TransactionStmt{}, false, nil
	}
	if len(doc.TransactionStmts) != 1 || len(doc.Nodes) != 0 ||
		len(doc.InsertStmts) != 0 || len(doc.UpdateStmts) != 0 || len(doc.DeleteStmts) != 0 ||
		len(doc.CreateTableStmts) != 0 || len(doc.DropTableStmts) != 0 ||
		len(doc.CreateIndexStmts) != 0 || len(doc.DropIndexStmts) != 0 || len(doc.AlterTableStmts) != 0 ||
		len(doc.ComputeLeidenStmts) != 0 {
		return parser.TransactionStmt{}, false, nil
	}
	return doc.TransactionStmts[0], true, nil
}

func isTransactionCleanupKind(kind parser.TransactionKind) bool {
	switch kind {
	case parser.TransactionCommit, parser.TransactionRollback, parser.TransactionRollbackToSavepoint:
		return true
	default:
		return false
	}
}

func transactionQueryAllowedAfterFailure(sql string) bool {
	stmt, ok, _ := parsePgwireTransactionControl(sql)
	return ok && isTransactionCleanupKind(stmt.Kind)
}

// applyTransactionCommand applies one parsed transaction statement and
// returns its PostgreSQL command tag. A failed transaction retains its epoch
// until ROLLBACK (or COMMIT, which performs the PostgreSQL-compatible cleanup).
func applyTransactionCommand(ctx context.Context, db *libravdb.Database, state *connState, stmt parser.TransactionStmt) (string, error) {
	if state == nil {
		return "", fmt.Errorf("connection transaction state is nil")
	}

	switch stmt.Kind {
	case parser.TransactionBegin, parser.TransactionBeginEpoch:
		if state.txStatus() == transactionFailed {
			return "", errCurrentTransactionAborted
		}
		if state.txStatus() != transactionIdle {
			return "", fmt.Errorf("a transaction is already in progress")
		}
		epoch, err := db.BeginEpochTx(ctx)
		if err != nil {
			return "", fmt.Errorf("BEGIN: %w", err)
		}
		state.markTransactionStarted(epoch)
		return "BEGIN", nil

	case parser.TransactionCommit:
		if state.epoch == nil {
			if state.txStatus() == transactionFailed {
				state.clearTransaction()
				return "ROLLBACK", nil
			}
			return "COMMIT", nil
		}
		if state.txStatus() == transactionFailed {
			if err := state.epoch.Rollback(ctx); err != nil {
				if errors.Is(err, libravdb.ErrEpochClosed) {
					// The epoch may have been closed by connection cleanup while
					// the wire state was still draining. ROLLBACK is a protocol
					// cleanup command and is idempotent once the engine has already
					// discarded the branch.
					state.clearTransaction()
					return "ROLLBACK", nil
				}
				return "", fmt.Errorf("ROLLBACK: %w", err)
			}
			state.clearTransaction()
			return "ROLLBACK", nil
		}
		if err := state.epoch.Commit(ctx); err != nil {
			state.markTransactionFailed()
			return "", fmt.Errorf("COMMIT: %w", err)
		}
		state.clearTransaction()
		return "COMMIT", nil

	case parser.TransactionRollback:
		if state.epoch == nil {
			state.clearTransaction()
			return "ROLLBACK", nil
		}
		if err := state.epoch.Rollback(ctx); err != nil {
			if errors.Is(err, libravdb.ErrEpochClosed) {
				// A closed epoch has already discarded its staged branch. Treat
				// the wire cleanup as successful and clear the stale connection
				// pointer so pool reset can continue normally.
				state.clearTransaction()
				return "ROLLBACK", nil
			}
			state.markTransactionFailed()
			return "", fmt.Errorf("ROLLBACK: %w", err)
		}
		state.clearTransaction()
		return "ROLLBACK", nil

	case parser.TransactionSavepoint:
		if state.epoch == nil || state.txStatus() == transactionIdle {
			return "", fmt.Errorf("savepoint is only valid inside a transaction")
		}
		if state.txStatus() == transactionFailed {
			return "", errCurrentTransactionAborted
		}
		if err := state.epoch.Savepoint(stmt.SavepointName); err != nil {
			state.markTransactionFailed()
			return "", err
		}
		return "SAVEPOINT", nil

	case parser.TransactionRollbackToSavepoint:
		if state.epoch == nil || state.txStatus() == transactionIdle {
			return "", fmt.Errorf("savepoint is only valid inside a transaction")
		}
		if err := state.epoch.RollbackTo(stmt.SavepointName); err != nil {
			state.markTransactionFailed()
			return "", err
		}
		state.transactionState = transactionInProgress
		return "ROLLBACK", nil

	case parser.TransactionReleaseSavepoint:
		if state.epoch == nil || state.txStatus() == transactionIdle {
			return "", fmt.Errorf("savepoint is only valid inside a transaction")
		}
		if state.txStatus() == transactionFailed {
			return "", errCurrentTransactionAborted
		}
		if err := state.epoch.ReleaseSavepoint(stmt.SavepointName); err != nil {
			state.markTransactionFailed()
			return "", err
		}
		return "RELEASE", nil
	default:
		return "", fmt.Errorf("unsupported transaction command %d", stmt.Kind)
	}
}

func handleSimpleTransaction(w io.Writer, db *libravdb.Database, state *connState, stmt parser.TransactionStmt) error {
	ctx, cancel := state.statementContext(context.Background())
	defer cancel()

	tag, err := applyTransactionCommand(ctx, db, state, stmt)
	if err != nil {
		return sendSimpleError(w, state, err)
	}
	if err := sendCommandComplete(w, tag); err != nil {
		return err
	}
	return sendReadyForQuery(w, state.readyStatus())
}

func sendSimpleError(w io.Writer, state *connState, err error) error {
	if state != nil && state.txStatus() != transactionIdle {
		state.markTransactionFailed()
	}
	var sendErr error
	if errors.Is(err, context.DeadlineExceeded) {
		sendErr = sendErrorWithCode(w, "ERROR", "57014", "canceling statement due to statement timeout")
	} else if errors.Is(err, context.Canceled) {
		sendErr = sendErrorWithCode(w, "ERROR", "57014", "canceling statement due to user request")
	} else {
		sendErr = sendError(w, "ERROR", err)
	}
	if sendErr != nil {
		return sendErr
	}
	return sendReadyForQuery(w, state.readyStatus())
}

func sendExtendedExecutionError(w io.Writer, state *connState, err error) error {
	if state != nil && state.txStatus() != transactionIdle {
		state.markTransactionFailed()
		// In an explicit transaction PostgreSQL requires Sync before any
		// further extended-protocol messages after an error. Autocommit
		// statements are independent transactions; keeping the connection in
		// the discard-until-Sync state there breaks database/sql statement
		// caches that pipeline recovery/deallocation after a failed command.
		state.extendedSyncRequired = true
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return sendErrorWithCode(w, "ERROR", "57014", "canceling statement due to statement timeout")
	}
	if errors.Is(err, context.Canceled) {
		return sendErrorWithCode(w, "ERROR", "57014", "canceling statement due to user request")
	}
	return sendError(w, "ERROR", err)
}
