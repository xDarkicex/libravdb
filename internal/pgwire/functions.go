package pgwire

import (
	"fmt"
	"strings"

	"github.com/xDarkicex/libravdb/libravdb"
)

// interceptSystemQuery checks if a SQL query matches a known system function or
// pg_catalog introspection query. If it does, it returns synthetic results directly
// without going through the full parse/bind/optimize/execute pipeline.
//
// This is critical for ORM compatibility: tools like Prisma, Hibernate, and
// ActiveRecord introspection-query pg_catalog tables before any user query.
func interceptSystemQuery(query string, db *libravdb.Database) (*libravdb.SearchResults, []ColumnMeta, bool) {
	trimmed := strings.TrimSpace(strings.TrimRight(query, ";"))
	upper := strings.ToUpper(trimmed)

	// System functions: SELECT version(), SELECT current_database(), etc.
	if results, columns, ok := handleSystemFunction(upper); ok {
		return results, columns, true
	}

	// pg_catalog introspection
	if strings.Contains(upper, "PG_CATALOG") || strings.Contains(upper, "PG_CLASS") ||
		strings.Contains(upper, "PG_ATTRIBUTE") || strings.Contains(upper, "PG_TYPE") ||
		strings.Contains(upper, "PG_NAMESPACE") || strings.Contains(upper, "INFORMATION_SCHEMA") {
		return handlePgCatalog(upper, db)
	}

	// SET commands (clients send these; ignore gracefully)
	if strings.HasPrefix(upper, "SET ") {
		return &libravdb.SearchResults{}, nil, true
	}

	// BEGIN / COMMIT / ROLLBACK (ignore — no transaction support yet)
	if upper == "BEGIN" || upper == "COMMIT" || upper == "ROLLBACK" {
		return &libravdb.SearchResults{}, nil, true
	}

	return nil, nil, false
}

// handleSystemFunction intercepts common system function calls.
func handleSystemFunction(sql string) (*libravdb.SearchResults, []ColumnMeta, bool) {
	var result string
	var columns []ColumnMeta

	switch {
	case containsFunc(sql, "VERSION()"):
		result = "libraVDB/0.1"
		columns = []ColumnMeta{{Name: "version", TypeOID: OIDText}}

	case containsFunc(sql, "CURRENT_DATABASE()"):
		result = "libravdb"
		columns = []ColumnMeta{{Name: "current_database", TypeOID: OIDName}}

	case containsFunc(sql, "CURRENT_SCHEMA()"), containsFunc(sql, "CURRENT_SCHEMAS"):
		result = "public"
		columns = []ColumnMeta{{Name: "current_schema", TypeOID: OIDName}}

	case containsFunc(sql, "PG_TYPEOF"):
		result = "25"
		columns = []ColumnMeta{{Name: "pg_typeof", TypeOID: OIDText}}

	case containsFunc(sql, "NOW()"):
		result = "2025-01-01 00:00:00"
		columns = []ColumnMeta{{Name: "now", TypeOID: OIDTimestamptz}}

	default:
		return nil, nil, false
	}

	return &libravdb.SearchResults{
		Results: []*libravdb.SearchResult{{ID: result, Score: 1.0}},
		Total:   1,
	}, columns, true
}

// handlePgCatalog returns synthetic results for pg_catalog/system table introspection.
func handlePgCatalog(sql string, db *libravdb.Database) (*libravdb.SearchResults, []ColumnMeta, bool) {
	// Minimal pg_class response — returns an empty result set with correct columns.
	// ORMs introspect this to discover tables. With catalog auto-registration,
	// tables are discoverable; for now return empty to signal "no tables" without error.
	switch {
	case strings.Contains(sql, "PG_CLASS"):
		return &libravdb.SearchResults{}, []ColumnMeta{
			{Name: "oid", TypeOID: OIDInt4},
			{Name: "relname", TypeOID: OIDName},
			{Name: "relnamespace", TypeOID: OIDInt4},
			{Name: "relkind", TypeOID: OIDBPChar},
			{Name: "reltuples", TypeOID: OIDFloat4},
		}, true

	case strings.Contains(sql, "PG_ATTRIBUTE"):
		return &libravdb.SearchResults{}, []ColumnMeta{
			{Name: "attrelid", TypeOID: OIDInt4},
			{Name: "attname", TypeOID: OIDName},
			{Name: "atttypid", TypeOID: OIDInt4},
			{Name: "attnum", TypeOID: OIDInt2},
			{Name: "attnotnull", TypeOID: OIDBool},
		}, true

	case strings.Contains(sql, "PG_TYPE"):
		return &libravdb.SearchResults{}, []ColumnMeta{
			{Name: "oid", TypeOID: OIDInt4},
			{Name: "typname", TypeOID: OIDName},
			{Name: "typlen", TypeOID: OIDInt2},
		}, true

	case strings.Contains(sql, "PG_NAMESPACE"):
		return &libravdb.SearchResults{
			Results: []*libravdb.SearchResult{
				{ID: "public", Score: 1.0},
				{ID: "pg_catalog", Score: 1.0},
				{ID: "information_schema", Score: 1.0},
			},
			Total: 3,
		}, []ColumnMeta{
			{Name: "nspname", TypeOID: OIDName},
			{Name: "nspowner", TypeOID: OIDInt4},
		}, true

	case strings.Contains(sql, "INFORMATION_SCHEMA.TABLES") ||
		strings.Contains(sql, "information_schema") && strings.Contains(sql, "table_name"):
		// Return empty table list — ORMs will try user tables next
		return &libravdb.SearchResults{}, []ColumnMeta{
			{Name: "table_name", TypeOID: OIDName},
			{Name: "table_schema", TypeOID: OIDName},
		}, true

	default:
		return &libravdb.SearchResults{}, nil, true
	}
}

// containsFunc checks if a function call appears in the SQL (case-insensitive).
func containsFunc(sql, fn string) bool {
	// Strip whitespace for matching
	clean := strings.Map(func(r rune) rune {
		if r == ' ' || r == '\t' || r == '\n' {
			return -1
		}
		return r
	}, sql)
	return strings.Contains(strings.ToUpper(clean), fn)
}

// Ensure fmt is used (for future expansion).
var _ = fmt.Sprintf
