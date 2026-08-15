package pgwire

import (
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/xDarkicex/libravdb/libravdb"
)

// serverCursor is the small connection-local state needed by DECLARE/FETCH.
// The rows still come from the normal executor result; this only retains the
// result and read position between FETCH commands.
type serverCursor struct {
	Name    string
	Query   string
	Results *libravdb.SearchResults
	Columns []ColumnMeta
	Row     int
}

func parseServerCursorDeclare(query string) (name, selectQuery string, ok bool) {
	trimmed := strings.TrimSpace(strings.TrimSuffix(query, ";"))
	fields := strings.Fields(trimmed)
	if len(fields) < 5 || !strings.EqualFold(fields[0], "DECLARE") {
		return "", "", false
	}
	forIndex := -1
	for i := 2; i < len(fields); i++ {
		if strings.EqualFold(fields[i], "FOR") {
			forIndex = i
			break
		}
	}
	if forIndex < 0 || forIndex+1 >= len(fields) {
		return "", "", false
	}
	name = strings.Trim(fields[1], `"`)
	if name == "" {
		return "", "", false
	}
	// Locate the FOR token in the original SQL so quoted literals and the
	// underlying SELECT remain byte-for-byte intact for parameter binding.
	upper := strings.ToUpper(trimmed)
	forPos := strings.Index(upper, " FOR ")
	if forPos < 0 {
		return "", "", false
	}
	selectQuery = strings.TrimSpace(trimmed[forPos+5:])
	if selectQuery == "" {
		return "", "", false
	}
	return name, selectQuery, true
}

func parseServerCursorName(token string) string {
	return strings.Trim(strings.TrimSpace(token), `"`)
}

func handleServerCursorSimple(rw io.ReadWriter, state *connState, query string) (bool, error) {
	trimmed := strings.TrimSpace(strings.TrimSuffix(query, ";"))
	fields := strings.Fields(trimmed)
	if len(fields) == 0 {
		return false, nil
	}
	if strings.EqualFold(fields[0], "FETCH") {
		if len(fields) < 5 || !strings.EqualFold(fields[1], "FORWARD") || !strings.EqualFold(fields[3], "FROM") {
			return false, nil
		}
		limit := -1
		if !strings.EqualFold(fields[2], "ALL") {
			parsed, err := strconv.Atoi(fields[2])
			if err != nil || parsed < 0 {
				return true, sendSimpleError(rw, state, fmt.Errorf("invalid FETCH row count %q", fields[2]))
			}
			limit = parsed
		}
		name := parseServerCursorName(fields[4])
		cursor, ok := state.serverCursors[name]
		if !ok {
			return true, sendSimpleError(rw, state, fmt.Errorf("cursor %q does not exist", name))
		}
		if cursor.Results == nil {
			cursor.Results = &libravdb.SearchResults{Results: []*libravdb.SearchResult{}}
		}
		start := cursor.Row
		if start < 0 {
			start = 0
		}
		end := len(cursor.Results.Results)
		if limit >= 0 && start+limit < end {
			end = start + limit
		}
		rows := &libravdb.SearchResults{
			Results: cursor.Results.Results[start:end],
			Total:   end - start,
			Columns: columnNames(cursor.Columns),
		}
		if err := sendResults(rw, rows, cursor.Columns); err != nil {
			return true, err
		}
		cursor.Row = end
		if err := sendCommandComplete(rw, fmt.Sprintf("FETCH %d", end-start)); err != nil {
			return true, err
		}
		return true, sendReadyForQuery(rw, state.readyStatus())
	}
	if strings.EqualFold(fields[0], "CLOSE") && len(fields) == 2 {
		name := parseServerCursorName(fields[1])
		if _, ok := state.serverCursors[name]; !ok {
			return true, sendSimpleError(rw, state, fmt.Errorf("cursor %q does not exist", name))
		}
		delete(state.serverCursors, name)
		if err := sendCommandComplete(rw, "CLOSE CURSOR"); err != nil {
			return true, err
		}
		return true, sendReadyForQuery(rw, state.readyStatus())
	}
	return false, nil
}
