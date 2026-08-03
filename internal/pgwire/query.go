package pgwire

import (
	"context"
	"fmt"
	"io"
	"time"

	"github.com/xDarkicex/libravdb/libravdb"
)

// handleQuery processes a Simple Query ('Q') message.
func handleQuery(rw io.ReadWriter, db *libravdb.Database, query string) error {
	// Check for system function / pg_catalog interception before normal query path
	if results, columns, handled := interceptSystemQuery(query, db); handled {
		return sendQueryResult(rw, results, columns)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

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
func columnOIDFor(results *libravdb.SearchResults, name string) uint32 {
	if name == "id" || name == "ID" {
		return OIDText
	}
	if name == "score" || name == "SCORE" {
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
