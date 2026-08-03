package pgwire

import (
	"encoding/binary"
	"fmt"
	"io"

	"github.com/xDarkicex/libravdb/libravdb"
)

// ColumnMeta describes a result set column for RowDescription encoding.
type ColumnMeta struct {
	Name    string
	TypeOID uint32
}

// sendRowDescription sends a RowDescription ('T') message describing the result columns.
// Each column uses the appropriate PostgreSQL type OID instead of hardcoded TEXT.
func sendRowDescription(w io.Writer, columns []ColumnMeta) error {
	if len(columns) == 0 {
		return WriteMessage(w, msgNoData, nil)
	}

	size := 2
	for _, col := range columns {
		size += len(col.Name) + 1 + 4 + 2 + 4 + 2 + 4 + 2 // null-terminated name + 18 bytes metadata
	}
	buf := make([]byte, 0, size)

	buf = append(buf, 0, 0)
	binary.BigEndian.PutUint16(buf[len(buf)-2:], uint16(len(columns)))

	for _, col := range columns {
		buf = append(buf, col.Name...)
		buf = append(buf, 0)

		// Table OID (0)
		buf = append(buf, 0, 0, 0, 0)
		// Column attribute number (0)
		buf = append(buf, 0, 0)
		// Data type OID
		binary.BigEndian.PutUint32(buf[len(buf):len(buf)+4], col.TypeOID)
		buf = buf[:len(buf)+4]
		// Data type size (-1 = variable)
		binary.BigEndian.PutUint16(buf[len(buf):len(buf)+2], 0xFFFF)
		buf = buf[:len(buf)+2]
		// Type modifier (-1)
		binary.BigEndian.PutUint32(buf[len(buf):len(buf)+4], 0xFFFFFFFF)
		buf = buf[:len(buf)+4]
		// Format code (0 = text)
		buf = append(buf, 0, 0)
	}

	return WriteMessage(w, msgRowDescription, buf)
}

// sendDataRow sends a DataRow ('D') message with the given column values as text.
func sendDataRow(w io.Writer, values []string) error {
	// numColumns (int16) + per-column: len (int32) + value bytes
	size := 2
	for _, v := range values {
		size += 4 + len(v)
	}
	buf := make([]byte, 0, size)

	binary.BigEndian.PutUint16(buf[:2], uint16(len(values)))
	buf = buf[:2]

	for _, v := range values {
		// Column value length
		binary.BigEndian.PutUint32(buf[len(buf):len(buf)+4], uint32(len(v)))
		buf = buf[:len(buf)+4]
		// Column value
		buf = append(buf, v...)
	}

	return WriteMessage(w, msgDataRow, buf)
}

// sendCommandComplete sends a CommandComplete ('C') message.
func sendCommandComplete(w io.Writer, tag string) error {
	return WriteMessage(w, msgCommandComplete, append([]byte(tag), 0))
}

// sendReadyForQuery sends ReadyForQuery ('Z') with the given transaction status.
func sendReadyForQuery(w io.Writer, status byte) error {
	return WriteMessage(w, msgReadyForQuery, []byte{status})
}

// sendResults encodes SearchResults into pgwire DataRow messages.
func sendResults(w io.Writer, results *libravdb.SearchResults, columns []ColumnMeta) error {
	if err := sendRowDescription(w, columns); err != nil {
		return err
	}

	// DataRows
	for _, r := range results.Results {
		vals := buildResultRow(r, columns)
		if err := sendDataRow(w, vals); err != nil {
			return err
		}
	}

	return nil
}

// buildResultRow constructs a row of string values from a SearchResult.
func buildResultRow(r *libravdb.SearchResult, columns []ColumnMeta) []string {
	vals := make([]string, len(columns))
	for i, col := range columns {
		switch col.Name {
		case "id", "ID":
			vals[i] = r.ID
		case "score", "SCORE":
			vals[i] = fmt.Sprintf("%f", r.Score)
		case "version", "VERSION":
			vals[i] = fmt.Sprintf("%d", r.Version)
		case "ordinal", "ORDINAL":
			vals[i] = fmt.Sprintf("%d", r.Ordinal)
		default:
			if len(columns) == 1 && col.Name == columns[0].Name {
				vals[i] = r.ID
			}
		}
	}
	return vals
}
