package pgwire

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/xDarkicex/libravdb/libravdb"
)

// handleQuery processes a Simple Query ('Q') message.
func handleQuery(rw io.ReadWriter, db *libravdb.Database, state *connState, query string) error {
	if handled, err := handleSQLPrepareExecute(rw, db, state, query); handled {
		return err
	}
	if handled, err := handleConnectionReset(rw, state, query); handled {
		return err
	}
	if handled, err := handleServerCursorSimple(rw, state, query); handled {
		return err
	}
	trimmed := strings.TrimSpace(strings.TrimRight(query, ";"))
	if trimmed == "" {
		if err := WriteMessage(rw, msgEmptyQuery, nil); err != nil {
			return err
		}
		return sendReadyForQuery(rw, state.readyStatus())
	}
	if handled, err := handleSQLDeallocate(rw, state, trimmed); handled {
		return err
	}
	if stmt, ok, _ := parsePgwireTransactionControl(trimmed); ok {
		if state.txStatus() == transactionFailed && !isTransactionCleanupKind(stmt.Kind) {
			return sendSimpleError(rw, state, errCurrentTransactionAborted)
		}
		return handleSimpleTransaction(rw, db, state, stmt)
	}

	if state.txStatus() == transactionFailed {
		return sendSimpleError(rw, state, errCurrentTransactionAborted)
	}
	if handled, commandTag, err := applySessionSettingSQL(state, query); handled {
		if err != nil {
			return sendSimpleError(rw, state, err)
		}
		if err := sendCommandComplete(rw, commandTag); err != nil {
			return err
		}
		return sendReadyForQuery(rw, state.readyStatus())
	}
	if results, columns, handled, err := handleAsyncpgJITQuery(query, &state.config, nil); handled {
		if err != nil {
			return sendSimpleError(rw, state, err)
		}
		return sendQueryResultWithStatus(rw, results, columns, state.readyStatus())
	}
	if results, columns, handled, err := handleSetConfigFunction(query, &state.config, nil); handled {
		if err != nil {
			return sendSimpleError(rw, state, err)
		}
		return sendQueryResultWithStatus(rw, results, columns, state.readyStatus())
	}

	// Rewrite pg_catalog. schema prefix so the parser can resolve system tables
	// (pg_class, pg_attribute, pg_type, pg_namespace) as bare identifiers.
	query = rewritePgCatalogQuery(query)

	// Check for system function interception before normal query path.
	if results, columns, handled := interceptSystemQuery(query, db); handled {
		return sendQueryResultWithStatus(rw, results, columns, state.readyStatus())
	}

	ctx, cancel := state.statementContext(context.Background())
	defer cancel()

	// If an epoch is active, route through it for read-your-writes.
	if state.epoch != nil {
		results, err := state.epoch.QueryWithSessionConfig(ctx, query, nil, &state.config)
		if err != nil {
			return sendSimpleError(rw, state, err)
		}
		return sendQueryResultWithStatus(rw, results, inferColumns(results), state.readyStatus())
	}

	results, err := db.QueryWithSessionConfig(ctx, query, nil, &state.config)
	if err != nil {
		return sendSimpleError(rw, state, err)
	}

	// Simple-query clients (notably psycopg2 and asyncpg for DDL) require a
	// CommandComplete for a non-empty statement that produces no row stream.
	// EmptyQueryResponse is reserved for an actually empty query string.
	if results == nil || results.Results == nil {
		if isRowProducingSQL(query) {
			if _, columns, describeErr := describeStatement(db, query, 0); describeErr == nil && len(columns) > 0 {
				if err := sendRowDescription(rw, columns); err != nil {
					return err
				}
			}
		}
		tag := commandTagForSQL(query, resultTotal(results))
		if err := sendCommandComplete(rw, tag); err != nil {
			return err
		}
		return sendReadyForQuery(rw, state.readyStatus())
	}

	return sendQueryResultWithStatus(rw, results, inferColumns(results), state.readyStatus())
}

// handleConnectionReset accepts the standard multi-statement reset batch used
// by asyncpg pools. These commands are connection state cleanup, not user SQL
// relations: they must produce one normal simple-query response stream and a
// single final ReadyForQuery after the batch.
func handleConnectionReset(rw io.ReadWriter, state *connState, query string) (bool, error) {
	parts := strings.Split(query, ";")
	commands := make([]string, 0, len(parts))
	for _, part := range parts {
		command := strings.TrimSpace(part)
		if command != "" {
			commands = append(commands, command)
		}
	}
	if len(commands) == 0 {
		return false, nil
	}
	for _, command := range commands {
		upper := strings.ToUpper(command)
		switch upper {
		case "SELECT PG_ADVISORY_UNLOCK_ALL()":
			results := &libravdb.SearchResults{
				Results: []*libravdb.SearchResult{{ID: "true", Score: 1, Metadata: map[string]interface{}{"pg_advisory_unlock_all": true}}},
				Total:   1,
			}
			if err := sendResults(rw, results, []ColumnMeta{{Name: "pg_advisory_unlock_all", TypeOID: OIDBool}}); err != nil {
				return true, err
			}
			if err := sendCommandComplete(rw, "SELECT 1"); err != nil {
				return true, err
			}
		case "CLOSE ALL":
			if err := sendCommandComplete(rw, "CLOSE CURSOR"); err != nil {
				return true, err
			}
		case "UNLISTEN *":
			if err := sendCommandComplete(rw, "UNLISTEN"); err != nil {
				return true, err
			}
		case "RESET ALL", "DISCARD ALL":
			if state != nil {
				state.config = libravdb.DefaultSessionConfig()
			}
			if err := sendCommandComplete(rw, "RESET"); err != nil {
				return true, err
			}
		default:
			return false, nil
		}
	}
	return true, sendReadyForQuery(rw, state.readyStatus())
}

func resultTotal(results *libravdb.SearchResults) int {
	if results == nil {
		return 0
	}
	return results.Total
}

// sendQueryResult sends a complete query result: RowDescription, DataRows, CommandComplete, ReadyForQuery.
func sendQueryResult(rw io.Writer, results *libravdb.SearchResults, columns []ColumnMeta) error {
	return sendQueryResultWithStatus(rw, results, columns, 'I')
}

// sendQueryResultWithStatus is the simple-protocol result path with an
// explicit transaction status. Simple Query must report that an active epoch
// remains open after a successful read; the default wrapper preserves the
// historical idle behavior for callers outside a transaction.
func sendQueryResultWithStatus(rw io.Writer, results *libravdb.SearchResults, columns []ColumnMeta, status byte) error {
	if results == nil || results.Results == nil {
		if err := WriteMessage(rw, msgEmptyQuery, nil); err != nil {
			return err
		}
		return sendReadyForQuery(rw, status)
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

	return sendReadyForQuery(rw, status)
}

// sendExtendedQueryResult sends an extended-protocol query result:
// RowDescription, DataRows, CommandComplete. It does NOT send ReadyForQuery
// — in the extended protocol, ReadyForQuery is sent only by handleSync.
// Sending it here would leave a stale Z in the client's buffer, desyncing
// the next message cycle (pgx reads the stale Z and breaks out of its
// readloop before ParseComplete/ParameterDescription arrive).
func sendExtendedQueryResult(rw io.Writer, results *libravdb.SearchResults, columns []ColumnMeta) error {
	return sendExtendedQueryResultWithFormats(rw, results, columns, nil)
}

func sendExtendedQueryResultWithFormats(rw io.Writer, results *libravdb.SearchResults, columns []ColumnMeta, formats []int16) error {
	if results == nil || results.Results == nil {
		return WriteMessage(rw, msgEmptyQuery, nil)
	}

	if err := sendResultsWithFormats(rw, results, columns, formats); err != nil {
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
		for i, name := range results.Columns {
			oid := uint32(0)
			if i < len(results.ColumnTypes) && results.ColumnTypes[i] != 0 {
				oid = catalogTypeToOID(results.ColumnTypes[i])
			}
			if oid == 0 {
				oid = columnOIDFor(results, name)
			}
			cols = append(cols, ColumnMeta{Name: name, TypeOID: oid})
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
		// Ordinary record IDs are textual, but GRAPH_NODES materializes its
		// durable numeric identity in metadata. Inspect it when no catalog
		// type metadata was provided.
		for _, r := range results.Results {
			if r == nil || r.Metadata == nil {
				continue
			}
			value, ok := r.Metadata[name]
			if !ok {
				value = r.Metadata["id"]
			}
			switch value.(type) {
			case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
				return OIDInt8
			}
		}
		return OIDText
	case "score", "SCORE":
		return OIDFloat8
	case "version", "VERSION", "begin_lsn", "BEGIN_LSN", "end_lsn", "END_LSN":
		return OIDInt8
	case "version_start", "VERSION_START", "version_end", "VERSION_END":
		return OIDTimestamptz
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
		case []string:
			return OIDTextArray
		case []int:
			return OIDInt8Array
		case []int64:
			return OIDInt8Array
		case []int32:
			return OIDInt4Array
		case []float32:
			return OIDFloat4Array
		case []float64:
			return OIDFloat8Array
		case []bool:
			return OIDBoolArray
		case []interface{}:
			return arrayElementOID(s)
		case map[string]interface{}, map[string]string:
			return OIDJSONB
		default:
			return OIDText
		}
	}
	return OIDText
}

func arrayElementOID(values []interface{}) uint32 {
	for _, value := range values {
		if value == nil {
			continue
		}
		switch value.(type) {
		case string:
			return OIDTextArray
		case int, int8, int16, int64, uint, uint8, uint16, uint32, uint64:
			return OIDInt8Array
		case int32:
			return OIDInt4Array
		case float32:
			return OIDFloat4Array
		case float64:
			return OIDFloat8Array
		case bool:
			return OIDBoolArray
		}
	}
	return OIDTextArray
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
