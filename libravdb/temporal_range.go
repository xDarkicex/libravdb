package libravdb

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/xDarkicex/lexer/parser"
	"github.com/xDarkicex/libravdb/internal/optimizer"
	"github.com/xDarkicex/libravdb/internal/storage"
)

// selectHasTemporalRange reports whether the query's root relation is a
// VERSIONS OF source. Such sources are query-local virtual relations and must
// bypass catalog/physical-plan binding just like derived tables and CTEs.
func selectHasTemporalRange(doc *parser.QueryDoc, stmt *parser.SelectStmt) bool {
	if doc == nil || stmt == nil || stmt.FromTable.Kind != parser.NodeKindTableExpr {
		return false
	}
	ref := stmt.FromTable
	return ref.ID >= 0 && int(ref.ID) < len(doc.TableExprs) && doc.TableExprs[ref.ID].TemporalRange
}

func parseTemporalRangeTime(src []byte, start, end uint32, params *optimizer.ParameterSet) (time.Time, error) {
	if start >= end || end > uint32(len(src)) {
		return time.Time{}, fmt.Errorf("empty temporal range bound")
	}
	text := string(src[start:end])
	if len(text) > 1 && (text[0] == '$' || text[0] == '@') {
		if params == nil {
			return time.Time{}, fmt.Errorf("temporal range parameter %q is not bound", text)
		}
		value, ok := params.Lookup(src, start, end)
		if !ok {
			return time.Time{}, fmt.Errorf("temporal range parameter %q is not bound", text)
		}
		switch value.Kind {
		case optimizer.ScalarString, optimizer.ScalarBytes:
			text = string(value.BytesData)
		case optimizer.ScalarTimestamp:
			return value.Time.UTC(), nil
		default:
			return time.Time{}, fmt.Errorf("temporal range parameter %q must be text or timestamp", text)
		}
	}
	if len(text) >= 2 && text[0] == '\'' && text[len(text)-1] == '\'' {
		text = text[1 : len(text)-1]
	}
	text = strings.TrimSpace(text)
	layouts := [...]string{
		time.RFC3339Nano,
		"2006-01-02 15:04:05.999999999",
		"2006-01-02 15:04:05",
	}
	for _, layout := range layouts {
		if parsed, err := time.Parse(layout, text); err == nil {
			return parsed.UTC(), nil
		}
	}
	return time.Time{}, fmt.Errorf("invalid temporal range timestamp %q", text)
}

func (db *Database) temporalRangeRows(ctx context.Context, src []byte, doc *parser.QueryDoc, table *parser.TableExpr, params *optimizer.ParameterSet) ([]virtualSQLRow, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if table == nil || !table.TemporalRange {
		return nil, fmt.Errorf("invalid temporal range source")
	}
	start, err := parseTemporalRangeTime(src, table.RangeStartStart, table.RangeStartEnd, params)
	if err != nil {
		return nil, fmt.Errorf("temporal range start: %w", err)
	}
	end, err := parseTemporalRangeTime(src, table.RangeEndStart, table.RangeEndEnd, params)
	if err != nil {
		return nil, fmt.Errorf("temporal range end: %w", err)
	}
	if end.Before(start) {
		return nil, fmt.Errorf("temporal range end precedes start")
	}
	startSnap, err := db.SnapshotAt(ctx, start)
	if err != nil {
		return nil, fmt.Errorf("resolve temporal range start: %w", err)
	}
	defer startSnap.Close()
	endSnap, err := db.SnapshotAt(ctx, end)
	if err != nil {
		return nil, fmt.Errorf("resolve temporal range end: %w", err)
	}
	defer endSnap.Close()
	reader, ok := db.storage.(storage.TemporalRangeReader)
	if !ok {
		return nil, fmt.Errorf("storage engine does not support temporal version ranges")
	}
	collection := sourceSpan(src, table.Start, table.End)
	alias := sourceSpan(src, table.Alias, table.AliasEnd)
	if alias == "" {
		alias = collection
	}
	rows := make([]virtualSQLRow, 0)
	err = reader.ListVersionsBetween(collection, startSnap.LSN, endSnap.LSN, func(version *storage.TemporalVersion) bool {
		if err := ctx.Err(); err != nil {
			return false
		}
		values := cloneMetadata(version.Metadata)
		if values == nil {
			values = make(map[string]interface{})
		}
		values["id"] = version.ID
		values["version"] = version.Version
		values["ordinal"] = version.Ordinal
		values["begin_lsn"] = version.BeginLSN
		values["end_lsn"] = version.EndLSN
		values["version_start"] = version.BeginTime
		if version.EndTime.IsZero() {
			values["version_end"] = nil
		} else {
			values["version_end"] = version.EndTime
		}
		row := virtualSQLRow{ID: version.ID, Values: values}
		qualifyVirtualRow(&row, alias)
		rows = append(rows, row)
		return true
	})
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return rows, nil
}
