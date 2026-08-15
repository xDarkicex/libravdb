package pgwire

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/xDarkicex/libravdb/libravdb"
)

// ── COPY format constants ────────────────────────────────────────────────────

const (
	copyFormatText = "text"
	copyFormatCSV  = "csv"
	copyNullMarker = "\\N" // PostgreSQL default NULL marker in text format
)

// copyOptions holds the parsed options for a COPY command.
type copyOptions struct {
	table     string
	columns   []string
	format    string // copyFormatText or copyFormatCSV
	delimiter byte   // default: '\t' for text, ',' for CSV
	nullStr   string // default: "\\N" for text, "" for CSV
	header    bool   // first row is column names (CSV only)
	quote     byte   // default: '"' (CSV only)
	escape    byte   // default: '"' (CSV only, same as quote)
}

// defaultCopyOptions returns text-format defaults.
func defaultCopyOptions() copyOptions {
	return copyOptions{
		format:    copyFormatText,
		delimiter: '\t',
		nullStr:   copyNullMarker,
		quote:     '"',
		escape:    '"',
	}
}

// ── COPY detection ───────────────────────────────────────────────────────────

// isCopy checks if a query starts a COPY operation (FROM STDIN or TO STDOUT).
func isCopy(query string) bool {
	upper := strings.ToUpper(strings.TrimSpace(query))
	if !strings.HasPrefix(upper, "COPY ") {
		return false
	}
	return strings.Contains(upper, "FROM STDIN") || strings.Contains(upper, "TO STDOUT")
}

// isCopyToStdout checks if a query is COPY ... TO STDOUT.
func isCopyToStdout(query string) bool {
	upper := strings.ToUpper(strings.TrimSpace(query))
	return strings.HasPrefix(upper, "COPY ") && strings.Contains(upper, "TO STDOUT")
}

// ── COPY dispatcher ──────────────────────────────────────────────────────────

// handleCopy dispatches to handleCopyIn or handleCopyOut based on the query.
func handleCopy(rw io.ReadWriter, arena *connArena, db *libravdb.Database, state *connState, query string) error {
	if isCopyToStdout(query) {
		return handleCopyOut(rw, db, state, query)
	}
	return handleCopyIn(rw, arena, db, state, query)
}

// ── Query parser ─────────────────────────────────────────────────────────────

// parseCopyOptions extracts table name, column list, and format options from a
// COPY query. Handles both FROM STDIN and TO STDOUT variants.
//
// Examples:
//
//	COPY users FROM STDIN
//	COPY users (id, name) FROM STDIN WITH (FORMAT csv, HEADER true)
//	COPY users TO STDOUT
func parseCopyOptions(query string) copyOptions {
	opts := defaultCopyOptions()

	// Split into parts, preserving parenthesized content.
	upper := strings.ToUpper(query)

	// Extract table name: first word after COPY.
	parts := strings.Fields(query)
	for i, p := range parts {
		if strings.EqualFold(p, "COPY") && i+1 < len(parts) {
			name := parts[i+1]
			// Strip parenthesized column list from table name.
			if idx := strings.Index(name, "("); idx >= 0 {
				name = name[:idx]
			}
			opts.table = name
			break
		}
	}

	// Extract column list if present.
	colStart := strings.Index(query, "(")
	fromIdx := strings.Index(upper, " FROM ")
	toIdx := strings.Index(upper, " TO ")
	endIdx := fromIdx
	if endIdx < 0 {
		endIdx = toIdx
	}
	if colStart >= 0 && endIdx > colStart {
		colStr := query[colStart+1 : endIdx]
		colStr = strings.TrimRight(colStr, " )")
		for _, c := range strings.Split(colStr, ",") {
			opts.columns = append(opts.columns, strings.TrimSpace(c))
		}
	}

	// Extract WITH options.
	withIdx := strings.Index(upper, " WITH (")
	if withIdx < 0 {
		withIdx = strings.Index(upper, " WITH(")
	}
	if withIdx >= 0 {
		optsStr := query[withIdx:]
		optsStr = strings.TrimSpace(optsStr)
		optsStr = strings.TrimPrefix(optsStr, "WITH")
		optsStr = strings.TrimPrefix(optsStr, "with")
		optsStr = strings.TrimSpace(optsStr)
		optsStr = strings.TrimPrefix(optsStr, "(")
		optsStr = strings.TrimSuffix(optsStr, ")")
		optsStr = strings.TrimSpace(optsStr)

		applyCopyOptions(&opts, optsStr)
	}

	return opts
}

// applyCopyOptions parses WITH (...) option key=value pairs into opts.
func applyCopyOptions(opts *copyOptions, raw string) {
	// Manual parsing: split by comma but respect quotes.
	parts := splitOptions(raw)
	for _, part := range parts {
		kv := strings.SplitN(strings.TrimSpace(part), " ", 2)
		if len(kv) < 2 {
			continue
		}
		key := strings.ToLower(strings.TrimSpace(kv[0]))
		val := strings.TrimSpace(kv[1])
		// Strip surrounding quotes.
		val = strings.Trim(val, "'\"")

		switch key {
		case "format":
			val = strings.ToLower(val)
			if val == copyFormatCSV {
				opts.format = copyFormatCSV
				opts.delimiter = ','
				opts.nullStr = ""
			}
		case "delimiter":
			if len(val) > 0 {
				opts.delimiter = val[0]
			}
		case "null":
			opts.nullStr = val
		case "header":
			opts.header = strings.ToLower(val) == "true"
		case "quote":
			if len(val) > 0 {
				opts.quote = val[0]
			}
		case "escape":
			if len(val) > 0 {
				opts.escape = val[0]
			}
		}
	}
}

// splitOptions splits a WITH options string by comma, respecting quotes.
func splitOptions(raw string) []string {
	var parts []string
	var buf bytes.Buffer
	inQuote := false
	var quoteChar byte

	for i := 0; i < len(raw); i++ {
		c := raw[i]
		if inQuote {
			buf.WriteByte(c)
			if c == quoteChar {
				inQuote = false
			}
		} else if c == '\'' || c == '"' {
			inQuote = true
			quoteChar = c
			buf.WriteByte(c)
		} else if c == ',' {
			parts = append(parts, buf.String())
			buf.Reset()
		} else {
			buf.WriteByte(c)
		}
	}
	if buf.Len() > 0 {
		parts = append(parts, buf.String())
	}
	return parts
}

// ── COPY ... FROM STDIN ──────────────────────────────────────────────────────

// handleCopyIn manages the COPY ... FROM STDIN protocol flow:
//
//	Server: CopyInResponse
//	Client: CopyData (multiple) → rows accumulated
//	Client: CopyDone (or CopyFail)
//	Server: CommandComplete + ReadyForQuery
func handleCopyIn(rw io.ReadWriter, arena *connArena, db *libravdb.Database, state *connState, query string) error {
	opts := parseCopyOptions(query)
	if opts.table == "" {
		return sendError(rw, "ERROR", fmt.Errorf("COPY: could not determine target table from %q", query))
	}

	// Send CopyInResponse.
	if err := sendCopyInResponse(rw, opts); err != nil {
		return err
	}

	// Accumulate rows from CopyData messages.
	var rows [][]*string
	for {
		arena.reset()
		msgType, payload, err := readMessageArena(rw, arena)
		if err != nil {
			return fmt.Errorf("COPY read: %w", err)
		}

		switch msgType {
		case msgCopyData:
			row := parseCopyRow(payload, opts)
			if len(row) > 0 {
				rows = append(rows, row)
			}

		case msgCopyDone:
			goto process

		case msgCopyFail:
			// Client aborted — send ReadyForQuery.
			if state.txStatus() != transactionIdle {
				state.markTransactionFailed()
			}
			return sendReadyForQuery(rw, state.readyStatus())

		default:
			return sendError(rw, "ERROR", fmt.Errorf("unexpected message %c during COPY", msgType))
		}
	}

process:
	// Skip header row if present (CSV with HEADER).
	if opts.header && len(rows) > 0 {
		rows = rows[1:]
	}

	// Validate target collection exists.
	col, err := db.GetCollection(opts.table)
	if err != nil {
		return sendError(rw, "ERROR", fmt.Errorf("COPY target table %q: %w", opts.table, err))
	}

	ctx, cancel := state.statementContext(context.Background())
	defer cancel()

	inserted := 0
	for _, row := range rows {
		entry := buildEntryFromRow(row, opts.columns)
		if entry.ID == "" {
			continue
		}

		// Route through epoch transaction if active, matching the executor path.
		if state != nil && state.epoch != nil {
			if err := state.epoch.Insert(ctx, opts.table, entry.ID, entry.Vector, entry.Metadata); err != nil {
				return sendError(rw, "ERROR", fmt.Errorf("COPY row %d: %w", inserted, err))
			}
		} else {
			if err := col.Insert(ctx, entry.ID, entry.Vector, entry.Metadata); err != nil {
				return sendError(rw, "ERROR", fmt.Errorf("COPY row %d: %w", inserted, err))
			}
		}
		inserted++
	}

	// CommandComplete.
	tag := fmt.Sprintf("COPY %d", inserted)
	if err := sendCommandComplete(rw, tag); err != nil {
		return err
	}
	return sendReadyForQuery(rw, state.readyStatus())
}

// ── Row parsing ──────────────────────────────────────────────────────────────

// parseCopyRow parses a single row from CopyData payload according to format.
// Returns []*string where nil means SQL NULL and non-nil is the column value.
func parseCopyRow(payload []byte, opts copyOptions) []*string {
	switch opts.format {
	case copyFormatCSV:
		return parseCSVRow(payload, opts)
	default:
		return parseTextRow(payload, opts)
	}
}

// parseTextRow parses a row in PostgreSQL COPY text format.
// Fields are separated by the delimiter (default tab).
// \N represents NULL. Backslash-escaped characters are unescaped.
func parseTextRow(payload []byte, opts copyOptions) []*string {
	if len(payload) == 0 {
		return nil
	}
	// Strip trailing newline characters.
	line := bytes.TrimRight(payload, "\r\n")
	if len(line) == 0 || bytes.Equal(line, []byte("\\.")) {
		return nil
	}

	var fields []*string
	var field bytes.Buffer
	nullMarker := []byte(opts.nullStr)

	for i := 0; i < len(line); i++ {
		c := line[i]

		if c == '\\' && i+1 < len(line) {
			// Backslash escape.
			next := line[i+1]
			switch next {
			case 'N':
				// \N is the NULL marker. It only counts as NULL if it
				// appears as a standalone field value, not embedded.
				// We accumulate into the field buffer and check after
				// the delimiter.
				field.WriteByte('\\')
				field.WriteByte('N')
				i++
			case '\\':
				field.WriteByte('\\')
				i++
			case 't':
				field.WriteByte('\t')
				i++
			case 'n':
				field.WriteByte('\n')
				i++
			case 'r':
				field.WriteByte('\r')
				i++
			default:
				// Unknown escape: pass through literally.
				field.WriteByte('\\')
				field.WriteByte(next)
				i++
			}
		} else if c == opts.delimiter {
			// End of field.
			fields = append(fields, resolveTextField(field.Bytes(), nullMarker))
			field.Reset()
		} else {
			field.WriteByte(c)
		}
	}
	// Last field.
	fields = append(fields, resolveTextField(field.Bytes(), nullMarker))

	return fields
}

// resolveTextField converts the raw bytes of a text-format field into a value.
// If the raw bytes exactly match the NULL marker (default \N), returns nil.
// Empty field bytes return a pointer to "".
func resolveTextField(raw, nullMarker []byte) *string {
	if bytes.Equal(raw, nullMarker) {
		return nil // SQL NULL
	}
	s := string(raw)
	return &s
}

// parseCSVRow parses a row in PostgreSQL COPY CSV format.
// Fields are separated by the delimiter (default comma).
// Quoted fields may contain delimiters, newlines, and escaped quotes.
// An unquoted empty string is NULL. A quoted empty field ("") is an empty string.
func parseCSVRow(payload []byte, opts copyOptions) []*string {
	if len(payload) == 0 {
		return nil
	}
	line := bytes.TrimRight(payload, "\r\n")
	if len(line) == 0 || bytes.Equal(line, []byte("\\.")) {
		return nil
	}

	var fields []*string
	var field bytes.Buffer
	inQuote := false
	wasQuoted := false // tracks whether the current field was enclosed in quotes

	for i := 0; i < len(line); i++ {
		c := line[i]

		if inQuote {
			if c == opts.quote {
				// Check for doubled quote (escape).
				if i+1 < len(line) && line[i+1] == opts.quote {
					field.WriteByte(opts.quote)
					i++ // skip the doubled quote
				} else {
					inQuote = false
				}
			} else if c == opts.escape && i+1 < len(line) {
				// Escape character: next char is literal.
				i++
				field.WriteByte(line[i])
			} else {
				field.WriteByte(c)
			}
		} else {
			if c == opts.quote && field.Len() == 0 {
				// Start of quoted field.
				inQuote = true
				wasQuoted = true
			} else if c == opts.delimiter {
				// End of field.
				fields = append(fields, resolveCSVFieldQuoted(field.Bytes(), opts, wasQuoted))
				field.Reset()
				wasQuoted = false
			} else {
				field.WriteByte(c)
			}
		}
	}
	// Last field.
	fields = append(fields, resolveCSVFieldQuoted(field.Bytes(), opts, wasQuoted))

	return fields
}

// resolveCSVFieldQuoted converts the raw bytes of a CSV field into a value.
// wasQuoted indicates the field was enclosed in quotes; a quoted empty field
// is an empty string, while an unquoted empty field is NULL.
func resolveCSVFieldQuoted(raw []byte, opts copyOptions, wasQuoted bool) *string {
	if wasQuoted {
		// Quoted field: even if empty, it's a non-NULL empty string.
		s := string(raw)
		return &s
	}
	if opts.nullStr == "" && len(raw) == 0 {
		return nil // SQL NULL (unquoted empty field in CSV)
	}
	if bytes.Equal(raw, []byte(opts.nullStr)) {
		return nil
	}
	s := string(raw)
	return &s
}

// ── Entry building ───────────────────────────────────────────────────────────

// buildEntryFromRow converts a parsed COPY row into a VectorEntry.
// columns is the explicit column list from the COPY command (may be empty).
func buildEntryFromRow(row []*string, columns []string) libravdb.VectorEntry {
	entry := libravdb.VectorEntry{
		Metadata: make(map[string]interface{}),
	}

	for i, val := range row {
		if val == nil {
			// NULL value — if it maps to a known column, set metadata to nil.
			if i < len(columns) {
				colName := strings.ToLower(columns[i])
				if colName == "id" || colName == "vector" || colName == "vec" || colName == "embedding" {
					continue // id and vector cannot be NULL; skip
				}
				entry.Metadata[colName] = nil
			}
			continue
		}

		// Non-NULL value.
		s := *val
		if i < len(columns) {
			colName := strings.ToLower(columns[i])
			switch colName {
			case "id":
				entry.ID = s
			case "vector", "vec", "embedding":
				entry.Vector = parseVectorLiteralStr(s)
			default:
				entry.Metadata[colName] = s
			}
		} else if i == 0 {
			entry.ID = s
		} else if i == 1 {
			entry.Vector = parseVectorLiteralStr(s)
		}
	}
	return entry
}

// ── COPY ... TO STDOUT ───────────────────────────────────────────────────────

// handleCopyOut manages the COPY ... TO STDOUT protocol flow:
//
//	Server: CopyOutResponse
//	Server: CopyData (multiple) → rows sent to client
//	Server: CopyDone
//	Server: CommandComplete + ReadyForQuery
func handleCopyOut(rw io.Writer, db *libravdb.Database, state *connState, query string) error {
	opts := parseCopyOptions(query)
	if opts.table == "" {
		return sendError(rw, "ERROR", fmt.Errorf("COPY TO STDOUT: could not determine table from %q", query))
	}

	col, err := db.GetCollection(opts.table)
	if err != nil {
		return sendError(rw, "ERROR", fmt.Errorf("COPY target table %q: %w", opts.table, err))
	}

	ctx, cancel := state.statementContext(context.Background())
	defer cancel()

	records, err := col.ListAll(ctx)
	if err != nil {
		return sendError(rw, "ERROR", fmt.Errorf("COPY TO STDOUT reading %q: %w", opts.table, err))
	}

	// Send CopyOutResponse.
	if err := sendCopyOutResponse(rw, opts); err != nil {
		return err
	}

	// Build column list: use explicit columns if provided, otherwise derive from records.
	outColumns := opts.columns
	if len(outColumns) == 0 {
		outColumns = inferColumnsForCopyOut(records)
	}

	// Header row (CSV only).
	if opts.header && opts.format == copyFormatCSV {
		headerData := formatCopyHeader(outColumns, opts)
		if err := sendCopyData(rw, headerData); err != nil {
			return err
		}
	}

	// Send each record as a CopyData row.
	rowCount := 0
	for _, rec := range records {
		data := formatCopyRow(rec, outColumns, opts)
		if err := sendCopyData(rw, data); err != nil {
			return err
		}
		rowCount++
	}

	// CopyDone.
	if err := sendCopyDone(rw); err != nil {
		return err
	}

	// CommandComplete.
	tag := fmt.Sprintf("COPY %d", rowCount)
	if err := sendCommandComplete(rw, tag); err != nil {
		return err
	}
	return sendReadyForQuery(rw, state.readyStatus())
}

// inferColumnsForCopyOut builds a column list from the first record's metadata.
func inferColumnsForCopyOut(records []libravdb.Record) []string {
	columns := []string{"id"}
	if len(records) > 0 && len(records[0].Vector) > 0 {
		columns = append(columns, "vector")
	}
	// Collect metadata keys from all records to handle sparse metadata.
	seen := make(map[string]struct{})
	for _, rec := range records {
		for k := range rec.Metadata {
			if _, ok := seen[k]; !ok {
				seen[k] = struct{}{}
				columns = append(columns, k)
			}
		}
	}
	return columns
}

// formatCopyHeader formats a header row for CSV mode.
func formatCopyHeader(columns []string, opts copyOptions) []byte {
	fields := make([]string, len(columns))
	for i, col := range columns {
		fields[i] = csvQuoteField(col, opts)
	}
	return []byte(strings.Join(fields, string(opts.delimiter)) + "\n")
}

// formatCopyRow formats a Record as a COPY text or CSV row.
// NULL metadata values are represented per the format's NULL marker.
func formatCopyRow(rec libravdb.Record, columns []string, opts copyOptions) []byte {
	var fields []string

	for _, col := range columns {
		val := fieldValue(rec, col)
		fields = append(fields, formatCopyField(val, opts))
	}

	return []byte(strings.Join(fields, string(opts.delimiter)) + "\n")
}

// fieldValue retrieves a column value from a Record.
func fieldValue(rec libravdb.Record, colName string) *string {
	switch strings.ToLower(colName) {
	case "id":
		s := rec.ID
		return &s
	case "vector", "vec", "embedding":
		s := formatVectorLiteral(rec.Vector)
		return &s
	case "version":
		s := fmt.Sprintf("%d", rec.Version)
		return &s
	case "ordinal":
		s := fmt.Sprintf("%d", rec.Ordinal)
		return &s
	default:
		if rec.Metadata != nil {
			if v, ok := rec.Metadata[colName]; ok {
				if v == nil {
					return nil // SQL NULL
				}
				s := metadataValueToString(v)
				return &s
			}
		}
		return nil // Not present → NULL
	}
}

// formatCopyField formats a single field value for COPY TO STDOUT.
// nil → NULL marker, non-nil → text or CSV representation.
func formatCopyField(val *string, opts copyOptions) string {
	if val == nil {
		if opts.format == copyFormatCSV && opts.nullStr == "" {
			return "" // CSV NULL is unquoted empty
		}
		return opts.nullStr
	}

	if opts.format == copyFormatCSV {
		return csvQuoteField(*val, opts)
	}

	// Text format: escape backslashes and NULL-marker-like prefixes.
	return escapeTextField(*val, opts)
}

// csvQuoteField quotes a CSV field if it contains special characters.
func csvQuoteField(s string, opts copyOptions) string {
	needsQuote := strings.ContainsAny(s, string([]byte{opts.delimiter, '"', '\n', '\r'}))
	if !needsQuote {
		return s
	}
	// Double any existing quotes.
	escaped := strings.ReplaceAll(s, "\"", "\"\"")
	return "\"" + escaped + "\""
}

// escapeTextField escapes a value for COPY text format.
// Backslashes are doubled; the NULL marker is produced only for nil.
func escapeTextField(s string, opts copyOptions) string {
	s = strings.ReplaceAll(s, "\\", "\\\\")
	return s
}

// formatVectorLiteral formats a float32 slice as a JSON-style array string.
func formatVectorLiteral(vec []float32) string {
	if len(vec) == 0 {
		return "[]"
	}
	parts := make([]string, len(vec))
	for i, v := range vec {
		parts[i] = fmt.Sprintf("%g", v)
	}
	return "[" + strings.Join(parts, ",") + "]"
}

// ── Wire protocol helpers ────────────────────────────────────────────────────

// sendCopyInResponse sends a CopyInResponse message.
func sendCopyInResponse(w io.Writer, opts copyOptions) error {
	// Format byte: 0 = text, 1 = binary.
	var formatByte byte
	if opts.format == copyFormatCSV {
		formatByte = 1 // signal CSV via binary flag (non-standard but informative)
	}
	// overallFormat (1 byte) + numColumns (2 bytes) + per-column formats (2 bytes each)
	buf := []byte{formatByte, 0, 0} // 0 columns = all default format
	return WriteMessage(w, msgCopyInResponse, buf)
}

// sendCopyOutResponse sends a CopyOutResponse message.
func sendCopyOutResponse(w io.Writer, opts copyOptions) error {
	var formatByte byte
	if opts.format == copyFormatCSV {
		formatByte = 1
	}
	buf := []byte{formatByte, 0, 0}
	return WriteMessage(w, msgCopyOutResponse, buf)
}

// sendCopyData sends a CopyData message containing one or more rows.
func sendCopyData(w io.Writer, data []byte) error {
	return WriteMessage(w, msgCopyData, data)
}

// sendCopyDone sends a CopyDone message (server → client).
func sendCopyDone(w io.Writer) error {
	return WriteMessage(w, msgCopyDone, nil)
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
