package pgwire

import (
	"context"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/xDarkicex/libravdb/libravdb"
)

// isCopyIn checks if a query starts a COPY ... FROM STDIN operation.
func isCopyIn(query string) bool {
	upper := strings.ToUpper(strings.TrimSpace(query))
	return strings.HasPrefix(upper, "COPY ") && strings.Contains(upper, "FROM STDIN")
}

// handleCopyIn manages the COPY ... FROM STDIN protocol flow:
//
//	Server: CopyInResponse
//	Client: CopyData (multiple) → rows accumulated
//	Client: CopyDone (or CopyFail)
//	Server: CommandComplete + ReadyForQuery
func handleCopyIn(rw io.ReadWriter, arena *connArena, db *libravdb.Database, query string) error {
	// Determine target table and columns from the COPY query
	tableName, columns := parseCopyQuery(query)
	if tableName == "" {
		return sendError(rw, "ERROR", fmt.Errorf("COPY: could not determine target table from %q", query))
	}

	// Send CopyInResponse: overall format=text, 0 columns (all text)
	if err := sendCopyInResponse(rw); err != nil {
		return err
	}

	// Accumulate rows from CopyData messages
	var rows [][]string
	for {
		arena.reset()
		msgType, payload, err := readMessageArena(rw, arena)
		if err != nil {
			return fmt.Errorf("COPY read: %w", err)
		}

		switch msgType {
		case msgCopyData:
			row := parseCopyData(payload)
			if len(row) > 0 {
				rows = append(rows, row)
			}

		case msgCopyDone:
			goto process

		case msgCopyFail:
			// Client aborted — send ReadyForQuery
			return sendReadyForQuery(rw, 'I')

		default:
			return sendError(rw, "ERROR", fmt.Errorf("unexpected message %c during COPY", msgType))
		}
	}

process:
	// TODO: batch insert rows into the collection
	// For now, insert rows one at a time via the collection API
	col, err := db.GetCollection(tableName)
	if err != nil {
		return sendError(rw, "ERROR", fmt.Errorf("COPY target table %q: %w", tableName, err))
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	inserted := 0
	for _, row := range rows {
		entry := buildVectorEntry(row, columns)
		if entry.ID == "" {
			continue
		}
		if err := col.Insert(ctx, entry.ID, entry.Vector, entry.Metadata); err != nil {
			return sendError(rw, "ERROR", fmt.Errorf("COPY row %d: %w", inserted, err))
		}
		inserted++
	}

	// CommandComplete
	tag := fmt.Sprintf("COPY %d", inserted)
	if err := sendCommandComplete(rw, tag); err != nil {
		return err
	}
	return sendReadyForQuery(rw, 'I')
}

// parseCopyQuery extracts the table name and optional column list from a COPY query.
// Examples:
//
//	COPY users FROM STDIN
//	COPY users (id, name) FROM STDIN
func parseCopyQuery(query string) (table string, columns []string) {
	// Simple parsing: split by whitespace, find table after COPY
	upper := strings.ToUpper(query)
	parts := strings.Fields(upper)
	for i, p := range parts {
		if p == "COPY" && i+1 < len(parts) {
			table = parts[i+1]
			// Check for column list
			if strings.Contains(table, "(") {
				table = strings.TrimSuffix(strings.SplitN(table, "(", 2)[0], " ")
			}
			break
		}
	}

	// Extract column list if present
	colStart := strings.Index(query, "(")
	fromIdx := strings.Index(upper, "FROM")
	if colStart >= 0 && fromIdx > colStart {
		colStr := query[colStart+1 : fromIdx]
		colStr = strings.TrimRight(colStr, " )")
		for _, c := range strings.Split(colStr, ",") {
			columns = append(columns, strings.TrimSpace(c))
		}
	}

	return table, columns
}

// parseCopyData parses a single row from CopyData payload.
// Text format: tab-separated values, \N for NULL.
func parseCopyData(payload []byte) []string {
	if len(payload) == 0 {
		return nil
	}
	line := strings.TrimRight(string(payload), "\r\n")
	if line == "" || line == "\\." {
		return nil
	}
	return strings.Split(line, "\t")
}

// buildVectorEntry converts a parsed COPY row into a VectorEntry.
// Vectors must be pre-encoded as JSON arrays (e.g., "[0.1, 0.2, 0.3]").
func buildVectorEntry(row []string, columns []string) libravdb.VectorEntry {
	entry := libravdb.VectorEntry{
		Metadata: make(map[string]interface{}),
	}
	for i, val := range row {
		// Handle \N (NULL marker in COPY text format)
		if val == "\\N" {
			continue
		}
		if i < len(columns) {
			colName := strings.ToLower(columns[i])
			switch colName {
			case "id":
				entry.ID = val
			case "vector", "vec", "embedding":
				entry.Vector = parseVectorLiteralStr(val)
			default:
				entry.Metadata[colName] = val
			}
		} else if i == 0 {
			entry.ID = val
		} else if i == 1 {
			entry.Vector = parseVectorLiteralStr(val)
		}
	}
	return entry
}

// parseVectorLiteralStr parses a JSON-style float array string like "[0.1, 0.2, 0.3]".
func parseVectorLiteralStr(s string) []float32 {
	s = strings.TrimSpace(s)
	if len(s) >= 2 && s[0] == '[' && s[len(s)-1] == ']' {
		s = s[1 : len(s)-1]
	}
	parts := strings.Split(s, ",")
	if len(parts) == 0 {
		return nil
	}
	floats := make([]float32, len(parts))
	for i, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		var f float32
		if _, err := fmt.Sscanf(p, "%f", &f); err != nil {
			return nil
		}
		floats[i] = f
	}
	return floats
}

// sendCopyInResponse sends a CopyInResponse message.
func sendCopyInResponse(w io.Writer) error {
	// Format: int8 overallFormat (0=text), int16 numColumns, int16[numColumns] columnFormats
	// 0 columns = all text format
	var buf [3]byte
	// overallFormat = 0 (text)
	// numColumns = 0 (all text, no per-column format)
	return WriteMessage(w, msgCopyInResponse, buf[:])
}
