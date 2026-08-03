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

// inferColumns returns sensible column names from search results.
func inferColumns(results *libravdb.SearchResults) []ColumnMeta {
	if len(results.Results) == 0 {
		return []ColumnMeta{{Name: "id", TypeOID: OIDText}, {Name: "score", TypeOID: OIDFloat8}}
	}
	// Default: id (text), score (float8) columns
	return []ColumnMeta{{Name: "id", TypeOID: OIDText}, {Name: "score", TypeOID: OIDFloat8}}
}
