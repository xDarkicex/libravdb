package pgwire

import (
	"context"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/xDarkicex/lexer/parser"
	"github.com/xDarkicex/libravdb/libravdb"
)

// handleQuery processes a Simple Query ('Q') message.
func handleQuery(rw io.ReadWriter, db *libravdb.Database, state *connState, query string) error {
	// ── Transaction control: intercept before system functions ──
	trimmed := strings.TrimSpace(strings.TrimRight(query, ";"))
	upper := strings.ToUpper(trimmed)

	if strings.HasPrefix(upper, "BEGIN EPOCH") {
		if state.epoch != nil {
			return sendError(rw, "ERROR", fmt.Errorf("a transaction is already in progress"))
		}
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		epoch, err := db.BeginEpochTx(ctx)
		if err != nil {
			return sendError(rw, "ERROR", fmt.Errorf("BEGIN EPOCH TRANSACTION: %w", err))
		}
		state.epoch = epoch
		if err := sendCommandComplete(rw, "BEGIN"); err != nil {
			return err
		}
		return sendReadyForQuery(rw, 'T') // 'T' = in transaction block
	}

	// Savepoint controls are parsed by the shared SQL parser and applied to
	// this connection's epoch. They must be handled before the normal query
	// path because savepoints are session state, not data statements.
	if stmt, ok, err := parsePgwireTransactionControl(query); ok && isPgwireSavepointKind(stmt.Kind) {
		if err != nil {
			return sendError(rw, "ERROR", err)
		}
		if state.epoch == nil {
			return sendError(rw, "ERROR", fmt.Errorf("savepoint is only valid inside an epoch transaction"))
		}
		switch stmt.Kind {
		case parser.TransactionSavepoint:
			err = state.epoch.Savepoint(stmt.SavepointName)
		case parser.TransactionRollbackToSavepoint:
			err = state.epoch.RollbackTo(stmt.SavepointName)
		case parser.TransactionReleaseSavepoint:
			err = state.epoch.ReleaseSavepoint(stmt.SavepointName)
		}
		if err != nil {
			return sendError(rw, "ERROR", err)
		}
		if err := sendCommandComplete(rw, transactionCommandTag(stmt.Kind)); err != nil {
			return err
		}
		return sendReadyForQuery(rw, 'T')
	}

	if upper == "COMMIT" || upper == "COMMIT TRANSACTION" {
		if state.epoch == nil {
			return sendError(rw, "ERROR", fmt.Errorf("no transaction in progress"))
		}
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if err := state.epoch.Commit(ctx); err != nil {
			state.epoch = nil
			return sendError(rw, "ERROR", fmt.Errorf("COMMIT: %w", err))
		}
		state.epoch = nil
		if err := sendCommandComplete(rw, "COMMIT"); err != nil {
			return err
		}
		return sendReadyForQuery(rw, 'I') // 'I' = idle
	}

	if upper == "ROLLBACK" || upper == "ROLLBACK TRANSACTION" {
		if state.epoch == nil {
			return sendError(rw, "ERROR", fmt.Errorf("no transaction in progress"))
		}
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if err := state.epoch.Rollback(ctx); err != nil {
			return sendError(rw, "ERROR", fmt.Errorf("ROLLBACK: %w", err))
		}
		state.epoch = nil
		if err := sendCommandComplete(rw, "ROLLBACK"); err != nil {
			return err
		}
		return sendReadyForQuery(rw, 'I')
	}

	// Check for system function / pg_catalog interception before normal query path
	if results, columns, handled := interceptSystemQuery(query, db); handled {
		return sendQueryResult(rw, results, columns)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// If an epoch is active, route through it for read-your-writes.
	if state.epoch != nil {
		results, err := state.epoch.Query(ctx, query, nil)
		if err != nil {
			if serr := sendError(rw, "ERROR", err); serr != nil {
				return serr
			}
			return sendReadyForQuery(rw, 'T')
		}
		return sendQueryResult(rw, results, inferColumns(results))
	}

	results, err := db.Query(ctx, query)
	if err != nil {
		if serr := sendError(rw, "ERROR", err); serr != nil {
			return serr
		}
		// Simple Query protocol: an ErrorResponse must still be followed by
		// ReadyForQuery to close the cycle. Without it, clients block forever.
		return sendReadyForQuery(rw, 'I')
	}

	return sendQueryResult(rw, results, inferColumns(results))
}

// parsePgwireTransactionControl recognizes savepoint statements using the
// shared lexer/parser. Normal SQL parse failures are returned as unhandled so
// the existing query path can produce its normal error response.
func parsePgwireTransactionControl(sql string) (parser.TransactionStmt, bool, error) {
	var doc parser.QueryDoc
	if err := parser.Parse([]byte(sql), &doc); err != nil {
		return parser.TransactionStmt{}, false, nil
	}
	if len(doc.TransactionStmts) != 1 {
		return parser.TransactionStmt{}, false, nil
	}
	stmt := doc.TransactionStmts[0]
	switch stmt.Kind {
	case parser.TransactionBeginEpoch, parser.TransactionCommit, parser.TransactionRollback,
		parser.TransactionSavepoint, parser.TransactionRollbackToSavepoint, parser.TransactionReleaseSavepoint:
		return stmt, true, nil
	default:
		return parser.TransactionStmt{}, false, nil
	}
}

func transactionCommandTag(kind parser.TransactionKind) string {
	switch kind {
	case parser.TransactionBeginEpoch:
		return "BEGIN"
	case parser.TransactionCommit:
		return "COMMIT"
	case parser.TransactionRollback:
		return "ROLLBACK"
	case parser.TransactionSavepoint:
		return "SAVEPOINT"
	case parser.TransactionRollbackToSavepoint:
		return "ROLLBACK"
	case parser.TransactionReleaseSavepoint:
		return "RELEASE"
	default:
		return "TRANSACTION"
	}
}

func isPgwireSavepointKind(kind parser.TransactionKind) bool {
	switch kind {
	case parser.TransactionSavepoint, parser.TransactionRollbackToSavepoint, parser.TransactionReleaseSavepoint:
		return true
	default:
		return false
	}
}

// sendQueryResult sends a complete query result: RowDescription, DataRows, CommandComplete, ReadyForQuery.
func sendQueryResult(rw io.Writer, results *libravdb.SearchResults, columns []ColumnMeta) error {
	if results == nil || results.Results == nil {
		if err := WriteMessage(rw, msgEmptyQuery, nil); err != nil {
			return err
		}
		return sendReadyForQuery(rw, 'I')
	}

	if err := sendResults(rw, results, columns); err != nil {
		return fmt.Errorf("sending results: %w", err)
	}

	var tag string
	if results.Total > 0 || len(results.Results) > 0 {
		tag = fmt.Sprintf("SELECT %d", len(results.Results))
	} else {
		tag = "SELECT 0"
	}
	if err := sendCommandComplete(rw, tag); err != nil {
		return err
	}

	return sendReadyForQuery(rw, 'I')
}

// sendExtendedQueryResult sends an extended-protocol query result:
// RowDescription, DataRows, CommandComplete. It does NOT send ReadyForQuery
// — in the extended protocol, ReadyForQuery is sent only by handleSync.
// Sending it here would leave a stale Z in the client's buffer, desyncing
// the next message cycle (pgx reads the stale Z and breaks out of its
// readloop before ParseComplete/ParameterDescription arrive).
func sendExtendedQueryResult(rw io.Writer, results *libravdb.SearchResults, columns []ColumnMeta) error {
	if results == nil || results.Results == nil {
		return WriteMessage(rw, msgEmptyQuery, nil)
	}

	if err := sendResults(rw, results, columns); err != nil {
		return fmt.Errorf("sending results: %w", err)
	}

	var tag string
	if results.Total > 0 || len(results.Results) > 0 {
		tag = fmt.Sprintf("SELECT %d", len(results.Results))
	} else {
		tag = "SELECT 0"
	}
	return sendCommandComplete(rw, tag)
}

// inferColumns returns the column list for a result set.
// When the SQL executor populated Columns (projected SELECT list), those are
// used verbatim. Otherwise fall back to the default id/score shape.
func inferColumns(results *libravdb.SearchResults) []ColumnMeta {
	if results != nil && len(results.Columns) > 0 {
		cols := make([]ColumnMeta, 0, len(results.Columns))
		for _, name := range results.Columns {
			cols = append(cols, ColumnMeta{Name: name, TypeOID: columnOIDFor(results, name)})
		}
		return cols
	}
	if results == nil || len(results.Results) == 0 {
		return []ColumnMeta{{Name: "id", TypeOID: OIDText}, {Name: "score", TypeOID: OIDFloat8}}
	}
	// Default: id (text), score (float8) columns
	return []ColumnMeta{{Name: "id", TypeOID: OIDText}, {Name: "score", TypeOID: OIDFloat8}}
}

// columnOIDFor picks a PostgreSQL type OID for a projected column based on
// the first non-nil value seen across rows (metadata values are stored as
// strings by the SQL INSERT path, so numeric-looking values map to numerics).
//
// COMPUTE LEIDEN column names are mapped explicitly so that empty results
// and typed metadata values always produce correct RowDescription OIDs.
func columnOIDFor(results *libravdb.SearchResults, name string) uint32 {
	// Standard built-in columns.
	switch name {
	case "id", "ID":
		return OIDText
	case "score", "SCORE", "version", "VERSION":
		return OIDFloat8
	case "ordinal", "ORDINAL":
		return OIDInt8

	// COMPUTE LEIDEN result columns — explicit OID mapping so empty
	// results and non-string metadata values always resolve correctly.
	case "node_id":
		return OIDInt8
	case "community_id":
		return OIDInt8
	case "collection":
		return OIDText
	case "record_id":
		return OIDText
	case "truncated":
		return OIDBool
	case "scope":
		return OIDText
	case "modularity":
		return OIDFloat8
	}

	for _, r := range results.Results {
		if r == nil || r.Metadata == nil {
			continue
		}
		v, ok := r.Metadata[name]
		if !ok || v == nil {
			continue
		}
		switch s := v.(type) {
		case string:
			if isIntString(s) {
				return OIDInt8
			}
			if isFloatString(s) {
				return OIDFloat8
			}
			return OIDText
		case int, int64, uint64, int32:
			return OIDInt8
		case float64, float32:
			return OIDFloat8
		case bool:
			return OIDBool
		default:
			return OIDText
		}
	}
	return OIDText
}

func isIntString(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		if i == 0 && (c == '-' || c == '+') {
			continue
		}
		if c < '0' || c > '9' {
			return false
		}
	}
	return true
}

func isFloatString(s string) bool {
	if s == "" {
		return false
	}
	hasDot := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if i == 0 && (c == '-' || c == '+') {
			continue
		}
		if c == '.' {
			hasDot = true
			continue
		}
		if c < '0' || c > '9' {
			return false
		}
	}
	return hasDot
}
