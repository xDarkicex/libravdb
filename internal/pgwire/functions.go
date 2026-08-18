package pgwire

import (
	"context"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/xDarkicex/libravdb/internal/catalog"
	"github.com/xDarkicex/libravdb/internal/optimizer"
	"github.com/xDarkicex/libravdb/libravdb"
)

// rewritePgCatalogQuery strips the pg_catalog. schema prefix from table
// references so the parser can resolve system tables (pg_class, pg_attribute,
// pg_type, pg_namespace) as bare identifiers. The SQL engine materializes
// these tables through the executor's system table path.
//
// The parser does not support schema-qualified table names (pg_catalog.X),
// so stripping the prefix here allows ORMs that use the prefix — Prisma,
// SQLAlchemy — to introspect the database through the full parse/bind/
// optimize/execute pipeline with WHERE, JOIN, and projection support.
func rewritePgCatalogQuery(query string) string {
	// Only strip the schema prefix (pg_catalog.), not bare occurrences of
	// the string (e.g. 'pg_catalog' as a string literal in WHERE clauses).
	return strings.ReplaceAll(query, "pg_catalog.", "")
}

// interceptSystemQuery handles SQL queries that need a short-circuit response
// without going through the parse/bind/optimize/execute pipeline.
//
// System function calls (SELECT version(), etc.) and
// information_schema queries are intercepted directly. pg_catalog queries
// are NOT intercepted — the caller must first strip the pg_catalog. prefix
// via rewritePgCatalogQuery, then the SQL engine handles them via the
// executor's system table materialization path.
func interceptSystemQuery(query string, db *libravdb.Database) (*libravdb.SearchResults, []ColumnMeta, bool) {
	return interceptSystemQueryWithParams(query, db, nil)
}

// interceptSystemQueryWithParams is the catalog compatibility boundary used
// by both simple and extended protocol execution. ORM metadata queries are
// deliberately answered from the live catalog/collection configuration here;
// they must not be routed through ordinary user-table scans or require
// persisted pg_catalog collections.
func interceptSystemQueryWithParams(query string, db *libravdb.Database, params *optimizer.ParameterSet) (*libravdb.SearchResults, []ColumnMeta, bool) {
	trimmed := strings.TrimSpace(strings.TrimRight(query, ";"))
	upper := strings.ToUpper(trimmed)

	if results, columns, ok := handleShowSetting(upper); ok {
		return results, columns, true
	}

	// System functions: SELECT version(), SELECT current_database(), etc.
	if results, columns, ok := handleSystemFunction(upper, db); ok {
		return results, columns, true
	}
	// asyncpg resolves codecs for explicitly typed parameters with a recursive
	// pg_type/pg_namespace/pg_range CTE. This is a catalog lookup, not an
	// application CTE; answer the requested live type rows directly so custom
	// types such as VECTOR can be bound through the normal extended protocol.
	if isAsyncpgTypeInfoQuery(upper) {
		return handleAsyncpgTypeInfoQuery(params)
	}
	// Alembic reads its connection-local migration version table before
	// autogenerate. The table is tool metadata rather than an application
	// relation, so expose its standard empty version projection only while a
	// fresh database has not created it. Once created, the normal live-table
	// path must answer version reads so relative upgrade/downgrade targets work.
	if strings.Contains(upper, "FROM ALEMBIC_VERSION") && strings.Contains(upper, "VERSION_NUM") {
		if db != nil {
			if collection, err := db.GetCollection("alembic_version"); err == nil {
				// Keep the migration table on the ordinary live collection path,
				// but normalize this one stable projection so SQLAlchemy/Alembic
				// always receives a row-capable extended-protocol description.
				records, listErr := collection.ListAll(context.Background())
				if listErr == nil {
					rows := make([]*libravdb.SearchResult, 0, len(records))
					for _, record := range records {
						rows = append(rows, &libravdb.SearchResult{ID: record.ID, Metadata: record.Metadata, Vector: record.Vector, Version: record.Version, Ordinal: record.Ordinal})
					}
					columns := []ColumnMeta{{Name: "version_num", TypeOID: OIDText}}
					return &libravdb.SearchResults{Results: rows, Total: len(rows), Columns: []string{"version_num"}}, columns, true
				}
				return nil, nil, false
			}
		}
		columns := []ColumnMeta{{Name: "version_num", TypeOID: OIDText}}
		return &libravdb.SearchResults{
			Results:     []*libravdb.SearchResult{},
			Total:       0,
			Columns:     columnNames(columns),
			ColumnTypes: columnOIDs(columns),
		}, columns, true
	}
	// Django's PostgreSQL introspector asks for a six-column field map using
	// pg_attribute/pg_type/pg_attrdef/pg_collation plus col_description(). It
	// is a live schema projection, so resolve it from the same collection
	// configuration used by native binding and SQLAlchemy.
	if strings.Contains(upper, "PG_ATTRIBUTE") &&
		strings.Contains(upper, "PG_ATTRDEF") &&
		strings.Contains(upper, "PG_COLLATION") &&
		strings.Contains(upper, "COL_DESCRIPTION") &&
		strings.Contains(upper, "PG_TYPE") &&
		strings.Contains(upper, "PG_CLASS") {
		return handleDjangoTableDescriptionQuery(db, trimmed, params)
	}
	// Django reconstructs foreign keys with one-based conkey/confkey array
	// subscripts. Answer this exact projection from durable FK metadata; this
	// also avoids routing synthetic pg_constraint joins through user-table
	// execution.
	if strings.Contains(upper, "PG_CONSTRAINT") &&
		strings.Contains(upper, "CONKEY[1]") &&
		strings.Contains(upper, "CONFKEY[1]") {
		return handleDjangoRelationsQuery(db, trimmed, params)
	}
	// Django's first get_constraints query expands conkey into a column-name
	// array and asks for the constraint kind. Return the live primary-key and
	// FK definitions in that exact five-column shape.
	if strings.Contains(upper, "FROM PG_CONSTRAINT AS C") &&
		strings.Contains(upper, "ARRAY(") &&
		strings.Contains(upper, "CONKEY") &&
		strings.Contains(upper, "CONTYPE") {
		return handleDjangoConstraintQuery(db, trimmed, params)
	}
	// The second Django constraint query expands pg_index/indkey with
	// ordinality. Index definitions are not currently exposed as a complete
	// PostgreSQL pg_index relation, so return the correctly typed empty result
	// rather than allowing the synthetic join to fail during inspectdb.
	if strings.Contains(upper, "UNNEST(I.INDKEY") && strings.Contains(upper, "PG_INDEX") {
		return handleDjangoIndexQuery()
	}
	// SQLAlchemy reflects live columns with a pg_class/pg_attribute join and
	// scalar pg_description/default subqueries. Resolve that stable projection
	// from the same collection metadata used by GORM.
	if strings.Contains(upper, "PG_ATTRIBUTE") && strings.Contains(upper, "PG_CLASS") && strings.Contains(upper, "PG_DESCRIPTION") {
		return handleSQLAlchemyColumnsQuery(db, trimmed)
	}
	// SQLAlchemy's get_multi_indexes query also contains pg_constraint,
	// pg_index, and ARRAY_AGG, but its row shape is unrelated to the
	// primary-key reflection query below. Match its indrelid/indoption
	// projection first so the dialect can address every expected key.
	if strings.Contains(upper, "PG_INDEX") &&
		strings.Contains(upper, "INDRELID") &&
		strings.Contains(upper, "INDOPTION") &&
		strings.Contains(upper, "ELEMENTS") {
		return handlePgIndexReflectionQuery()
	}
	// SQLAlchemy asks for domains and enums during dialect initialization. The
	// engine has no user-defined PostgreSQL domains/enums, so typed empty
	// relations are the correct catalog response.
	if strings.Contains(upper, "PG_TYPE_IS_VISIBLE") && strings.Contains(upper, "PG_TYPE") {
		if strings.Contains(upper, "FORMAT_TYPE") {
			return handlePgDomainQuery()
		}
		if strings.Contains(upper, "PG_ENUM") {
			return handlePgEnumQuery()
		}
	}
	// SQLAlchemy's primary/unique reflection query joins pg_constraint and
	// pg_index through derived tables. Return the live primary-key shape before
	// the general parser sees the unsupported derived relation.
	if strings.Contains(upper, "PG_CONSTRAINT") && strings.Contains(upper, "PG_INDEX") && strings.Contains(upper, "ARRAY_AGG") {
		return handlePgConstraintReflectionQuery(db, trimmed, params)
	}
	// SQLAlchemy's foreign-key reflection query joins pg_class, pg_constraint,
	// pg_description, and an aliased pg_namespace, then expects the exact
	// five-column result of PostgreSQL's dialect query. Resolve it from the
	// collection configuration, which is the same durable FK metadata used by
	// enforcement and DDL replay.
	constraintType := catalogConstraintType(params)
	if strings.Contains(upper, "PG_GET_CONSTRAINTDEF") &&
		strings.Contains(upper, "PG_CONSTRAINT") &&
		strings.Contains(upper, "PG_NAMESPACE") &&
		(constraintType == "f" || strings.Contains(upper, "CONTYPE = 'F'") || strings.Contains(upper, "CLS_REF")) &&
		strings.Contains(upper, "FROM PG_CLASS") {
		return handlePgForeignKeyReflectionQuery(db)
	}
	if strings.Contains(upper, "PG_GET_CONSTRAINTDEF") &&
		strings.Contains(upper, "PG_CONSTRAINT") &&
		!strings.Contains(upper, "PG_INDEX") &&
		!strings.Contains(upper, "CLS_REF") &&
		(constraintType == "c" || constraintType == "" || strings.Contains(upper, "CONTYPE = 'C'")) {
		return handlePgCheckConstraintReflectionQuery(db)
	}
	// SQLAlchemy's table-comment reflection selects relname together with the
	// nullable pg_description value. The engine has no table-comment catalog
	// entries, but must still return both columns for every live relation.
	if strings.HasPrefix(strings.TrimSpace(upper), "SELECT PG_CLASS.RELNAME, PG_DESCRIPTION.DESCRIPTION") &&
		strings.Contains(upper, "PG_DESCRIPTION") &&
		!strings.Contains(upper, "PG_CONSTRAINT") {
		return handlePgTableCommentQuery(db)
	}
	// pg_description queries often contain information_schema subqueries in
	// their predicates. Match the projected system relation before the broader
	// information_schema compatibility branch so callers receive one
	// `description` column rather than the 12-column information_schema shape.
	if strings.Contains(upper, "FROM PG_DESCRIPTION") && !strings.Contains(upper, "PG_ATTRIBUTE") {
		return handlePgDescriptionQuery()
	}
	// SQLAlchemy's table-list query joins pg_class to pg_namespace. The
	// optimizer's ordinary join path intentionally does not join synthetic
	// system relations, so answer this metadata projection from the live
	// collection catalog instead.
	if strings.Contains(upper, "PG_CLASS") && strings.Contains(upper, "PG_NAMESPACE") && !strings.Contains(upper, "PG_ATTRIBUTE") && !strings.Contains(upper, "INFORMATION_SCHEMA") {
		return handlePgClassNamespaceQuery(db, trimmed)
	}

	// information_schema — not supported, return empty results so ORMs
	// fall back to pg_catalog or user-table introspection.
	if strings.Contains(upper, "INFORMATION_SCHEMA") {
		return handleInformationSchemaWithParams(trimmed, db, params)
	}
	// psycopg's optional hstore lookup asks for a full pg_type row including
	// array metadata and regtype casts. hstore is not a libraVDB type, so the
	// PostgreSQL-compatible answer is an empty typed relation. Intercept this
	// lookup before the deliberately smaller native pg_type materialization.
	if strings.Contains(upper, "FROM PG_TYPE") && strings.Contains(upper, "TYPARRAY") {
		return handlePgTypeInfoQuery(trimmed, params)
	}

	// GORM's ColumnTypes query joins pg_attribute to pg_class and
	// pg_namespace. The regular optimizer can materialize each system table
	// individually, but not yet as a join; answer this stable metadata shape
	// directly so the ORM sees the same schema it would receive from PostgreSQL.
	if strings.Contains(upper, "PG_ATTRIBUTE") &&
		(strings.Contains(upper, "PG_CLASS") || strings.Contains(upper, "PG_NAMESPACE")) {
		return handlePgAttributeQuery(trimmed, db, params)
	}
	if strings.Contains(upper, "PG_INDEXES") {
		return handlePgIndexesQuery(trimmed, db, params)
	}

	return nil, nil, false
}

func isAsyncpgTypeInfoQuery(query string) bool {
	upper := strings.ToUpper(query)
	return strings.Contains(upper, "WITH RECURSIVE TYPEINFO_TREE") &&
		strings.Contains(upper, "PG_TYPE") && strings.Contains(upper, "TYPELEM")
}

func handleShowSetting(sql string) (*libravdb.SearchResults, []ColumnMeta, bool) {
	if !strings.HasPrefix(strings.TrimSpace(sql), "SHOW ") {
		return nil, nil, false
	}
	name := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(sql), "SHOW "))
	values := map[string]string{
		"APPLICATION_NAME":            "",
		"BYTEA_OUTPUT":                "hex",
		"CLIENT_ENCODING":             "UTF8",
		"DATESTYLE":                   "ISO, MDY",
		"EXTRA_FLOAT_DIGITS":          "3",
		"INTEGER_DATETIMES":           "on",
		"INTERVALSTYLE":               "postgres",
		"SEARCH_PATH":                 `"$user", public`,
		"SERVER_VERSION":              "16.0",
		"SERVER_VERSION_NUM":          "160000",
		"STATEMENT_TIMEOUT":           "0",
		"STANDARD_CONFORMING_STRINGS": "on",
		"TRANSACTION ISOLATION LEVEL": "read committed",
		"TIME ZONE":                   "UTC",
		"TIMEZONE":                    "UTC",
	}
	value, ok := values[name]
	if !ok {
		return nil, nil, false
	}
	columns := []ColumnMeta{{Name: strings.ToLower(name), TypeOID: OIDText}}
	return catalogRows(columns, map[string]interface{}{columns[0].Name: value}), columns, true
}

// handleSystemFunction intercepts common system function calls.
func handleSystemFunction(sql string, db *libravdb.Database) (*libravdb.SearchResults, []ColumnMeta, bool) {
	trimmed := strings.TrimSpace(sql)
	upper := strings.ToUpper(trimmed)
	// These are compatibility shims for standalone system-function queries.
	// Never intercept a DML statement merely because an expression inside it
	// contains NOW(), VERSION(), or another function name.
	if !strings.HasPrefix(upper, "SELECT ") && upper != "SELECT" {
		return nil, nil, false
	}
	var result string
	var resultValue interface{}
	var columns []ColumnMeta

	switch {
	case standaloneSystemFunction(sql, "VERSION()"):
		result = "PostgreSQL 16.0 (libraVDB/0.1)"
		columns = []ColumnMeta{{Name: "version", TypeOID: OIDText}}

	case standaloneSystemFunction(sql, "CURRENT_DATABASE()"):
		result = "libravdb"
		columns = []ColumnMeta{{Name: "current_database", TypeOID: OIDName}}

	case standaloneSystemFunction(sql, "CURRENT_SCHEMA()"), standaloneSystemFunction(sql, "CURRENT_SCHEMAS"):
		result = "public"
		columns = []ColumnMeta{{Name: "current_schema", TypeOID: OIDName}}

	case standaloneSystemFunction(sql, "PG_TYPEOF"):
		result = "25"
		columns = []ColumnMeta{{Name: "pg_typeof", TypeOID: OIDText}}

	case standaloneSystemFunction(sql, "NOW()"):
		result = time.Now().UTC().Format(time.RFC3339Nano)
		columns = []ColumnMeta{{Name: "now", TypeOID: OIDTimestamptz}}

	case standaloneSystemFunction(sql, "LIBRAVDB_LATEST_COMMIT_LSN()"):
		// Keep the wire value in the existing int8 column contract. The
		// connection's database is always present on the network path; nil is
		// retained for in-process startup/function tests and represents an empty
		// state rather than inventing another token source.
		var lsn uint64
		if db != nil {
			if current, err := db.LatestCommitLSN(context.Background()); err == nil {
				lsn = current
			}
		}
		result = strconv.FormatUint(lsn, 10)
		resultValue = result
		columns = []ColumnMeta{{Name: "libravdb_latest_commit_lsn", TypeOID: OIDInt8}}

	case standaloneSystemFunction(sql, "LIBRAVDB_SQL_STATS()"):
		// Keep the metrics payload structured at the database boundary. The
		// JSONB result encoder serializes the public SQLQueryStats snapshot for
		// native pgwire clients without exposing internal atomics.
		result = "1"
		if db != nil {
			resultValue = db.SQLStats()
		} else {
			resultValue = libravdb.SQLQueryStats{}
		}
		columns = []ColumnMeta{{Name: "libravdb_sql_stats", TypeOID: OIDJSONB}}

	default:
		return nil, nil, false
	}

	metadata := map[string]interface{}{}
	if len(columns) > 0 {
		// System-function rows must carry the named value as metadata. The
		// pgwire encoder uses column names to project DataRow fields; leaving
		// metadata nil silently encoded a SQL NULL even though the function
		// returned a value, which breaks database/sql/GORM scans.
		if resultValue != nil {
			metadata[columns[0].Name] = resultValue
		} else {
			metadata[columns[0].Name] = result
		}
	}
	return &libravdb.SearchResults{
		Results: []*libravdb.SearchResult{{ID: result, Score: 1.0, Metadata: metadata}},
		Total:   1,
	}, columns, true
}

// handleInformationSchema returns empty results for information_schema queries.
// information_schema is not supported; ORMs that encounter an empty result set
// typically fall back to pg_catalog introspection, which is fully supported.
func handleInformationSchema(sql string) (*libravdb.SearchResults, []ColumnMeta, bool) {
	return handleInformationSchemaWithParams(sql, nil, nil)
}

func handleInformationSchemaWithParams(sql string, db *libravdb.Database, params *optimizer.ParameterSet) (*libravdb.SearchResults, []ColumnMeta, bool) {
	upper := strings.ToUpper(sql)
	switch {
	case strings.Contains(upper, "INFORMATION_SCHEMA.TABLES"):
		columns := []ColumnMeta{{Name: "count", TypeOID: OIDInt8}}
		name := catalogTargetTable(sql, params)
		count := int64(0)
		if db != nil && name != "" {
			if _, err := db.GetCollection(name); err == nil {
				count = 1
			}
		}
		return catalogRows(columns, map[string]interface{}{"count": count}), columns, true

	case strings.Contains(upper, "INFORMATION_SCHEMA.COLUMNS"):
		columns := []ColumnMeta{
			{Name: "column_name", TypeOID: OIDName},
			{Name: "is_nullable", TypeOID: OIDBool},
			{Name: "udt_name", TypeOID: OIDName},
			{Name: "character_maximum_length", TypeOID: OIDInt8},
			{Name: "numeric_precision", TypeOID: OIDInt8},
			{Name: "numeric_precision_radix", TypeOID: OIDInt8},
			{Name: "numeric_scale", TypeOID: OIDInt8},
			{Name: "datetime_precision", TypeOID: OIDInt8},
			{Name: "8 * typlen", TypeOID: OIDInt8},
			{Name: "column_default", TypeOID: OIDText},
			{Name: "description", TypeOID: OIDText},
			{Name: "identity_increment", TypeOID: OIDText},
		}
		name := catalogTargetTable(sql, params)
		rows := make([]*libravdb.SearchResult, 0)
		if db != nil {
			if col, err := db.GetCollection(name); err == nil {
				cfg := col.Config()
				for _, field := range collectionCatalogFields(cfg, col.Dimension()) {
					row := map[string]interface{}{
						"column_name":              field.name,
						"is_nullable":              !field.notNull,
						"udt_name":                 field.udt,
						"character_maximum_length": nil,
						"numeric_precision":        nil,
						"numeric_precision_radix":  nil,
						"numeric_scale":            nil,
						"datetime_precision":       nil,
						"8 * typlen":               int64(field.typlen),
						"column_default":           nil,
						"description":              nil,
						"identity_increment":       nil,
					}
					rows = append(rows, &libravdb.SearchResult{ID: field.name, Score: 1, Metadata: row})
				}
			}
		}
		return &libravdb.SearchResults{Results: rows, Total: len(rows), Columns: columnNames(columns), ColumnTypes: columnOIDs(columns)}, columns, true

	case strings.Contains(upper, "INFORMATION_SCHEMA.TABLE_CONSTRAINTS"):
		// GORM asks this relation both for UNIQUE constraints and for the
		// primary/unique column mapping. Return the declared primary key when
		// available; an empty result is correct for a table with no match.
		if strings.Contains(upper, "SELECT CONSTRAINT_NAME") && !strings.Contains(upper, "CCU.COLUMN_NAME") {
			columns := []ColumnMeta{{Name: "constraint_name", TypeOID: OIDName}}
			return catalogConstraintRows(sql, db, params, columns), columns, true
		}
		columns := []ColumnMeta{
			{Name: "column_name", TypeOID: OIDName},
			{Name: "constraint_name", TypeOID: OIDName},
			{Name: "constraint_type", TypeOID: OIDName},
		}
		return catalogConstraintRows(sql, db, params, columns), columns, true
	default:
		return &libravdb.SearchResults{}, nil, true
	}
}

type catalogField struct {
	name    string
	udt     string
	typlen  int
	notNull bool
}

func collectionCatalogFields(cfg libravdb.CollectionConfig, dimension int) []catalogField {
	fields := make([]catalogField, 0, len(cfg.MetadataSchema)+2)
	fields = append(fields, catalogField{name: "id", udt: "text", typlen: -1, notNull: true})
	names := make([]string, 0, len(cfg.MetadataSchema))
	for name := range cfg.MetadataSchema {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		// Every collection has one physical id column. SQL DDL may also retain
		// that declared key in MetadataSchema; expose it only once to catalog
		// introspection (matching SELECT * and Describe behavior).
		if strings.EqualFold(name, "id") {
			continue
		}
		field := cfg.MetadataSchema[name]
		udt, size := catalogFieldType(field)
		flags := cfg.ColumnConstraints[name]
		fields = append(fields, catalogField{name: name, udt: udt, typlen: size, notNull: flags&1 != 0})
	}
	if dimension > 0 {
		fields = append(fields, catalogField{name: "embedding", udt: "_float4", typlen: -1, notNull: true})
	}
	return fields
}

func catalogFieldType(field libravdb.FieldType) (string, int) {
	switch field {
	case libravdb.IntField:
		return "int4", 4
	case libravdb.BigIntField:
		return "int8", 8
	case libravdb.FloatField:
		return "float8", 8
	case libravdb.BoolField:
		return "bool", 1
	case libravdb.TimeField:
		return "timestamptz", 8
	case libravdb.JSONField:
		return "json", -1
	case libravdb.JSONBField:
		return "jsonb", -1
	case libravdb.StringArrayField:
		return "_text", -1
	case libravdb.IntArrayField:
		return "_int4", -1
	case libravdb.FloatArrayField:
		return "_float4", -1
	default:
		return "text", -1
	}
}

func handlePgAttributeQuery(sql string, db *libravdb.Database, params *optimizer.ParameterSet) (*libravdb.SearchResults, []ColumnMeta, bool) {
	columns := []ColumnMeta{{Name: "column_name", TypeOID: OIDName}, {Name: "data_type", TypeOID: OIDText}}
	name := catalogTargetTable(sql, params)
	rows := make([]*libravdb.SearchResult, 0)
	if db != nil {
		if col, err := db.GetCollection(name); err == nil {
			for _, field := range collectionCatalogFields(col.Config(), col.Dimension()) {
				rows = append(rows, &libravdb.SearchResult{ID: field.name, Score: 1, Metadata: map[string]interface{}{
					"column_name": field.name,
					"data_type":   field.udt,
				}})
			}
		}
	}
	return &libravdb.SearchResults{Results: rows, Total: len(rows), Columns: columnNames(columns), ColumnTypes: columnOIDs(columns)}, columns, true
}

// handlePgIndexesQuery supplies the small pg_indexes surface used by ORMs to
// check whether a named unique/composite index already exists during
// AutoMigrate.  The index itself remains part of the libraVDB catalog; this is
// only a typed virtual catalog view.
func handlePgIndexesQuery(sql string, db *libravdb.Database, params *optimizer.ParameterSet) (*libravdb.SearchResults, []ColumnMeta, bool) {
	tableName := catalogTargetTable(sql, params)
	indexName := catalogPredicateValue(sql, "INDEXNAME")
	found := false
	if db != nil && tableName != "" {
		if col, err := db.GetCollection(tableName); err == nil {
			cfg := col.Config()
			if indexName == tableName+"_pkey" || strings.EqualFold(indexName, "PRIMARY") {
				found = true
			}
			for name := range cfg.NamedUniqueConstraints {
				if strings.EqualFold(name, indexName) {
					found = true
					break
				}
			}
			for _, index := range cfg.SQLIndexes {
				if strings.EqualFold(index.Name, indexName) {
					found = true
					break
				}
			}
		}
	}
	if strings.Contains(strings.ToUpper(sql), "COUNT(") {
		columns := []ColumnMeta{{Name: "count", TypeOID: OIDInt8}}
		count := int64(0)
		if found {
			count = 1
		}
		return catalogRows(columns, map[string]interface{}{"count": count}), columns, true
	}
	columns := []ColumnMeta{{Name: "indexname", TypeOID: OIDName}, {Name: "indexdef", TypeOID: OIDText}}
	if !found {
		return &libravdb.SearchResults{Columns: columnNames(columns), ColumnTypes: columnOIDs(columns)}, columns, true
	}
	definition := "CREATE INDEX " + indexName + " ON " + tableName
	return &libravdb.SearchResults{Results: []*libravdb.SearchResult{{ID: indexName, Score: 1, Metadata: map[string]interface{}{"indexname": indexName, "indexdef": definition}}}, Total: 1, Columns: columnNames(columns), ColumnTypes: columnOIDs(columns)}, columns, true
}

func handlePgIndexReflectionQuery() (*libravdb.SearchResults, []ColumnMeta, bool) {
	columns := []ColumnMeta{
		{Name: "indrelid", TypeOID: OIDOID},
		{Name: "relname", TypeOID: OIDName},
		{Name: "indisunique", TypeOID: OIDBool},
		{Name: "has_constraint", TypeOID: OIDBool},
		{Name: "indoption", TypeOID: OIDInt2Array},
		{Name: "reloptions", TypeOID: OIDTextArray},
		{Name: "amname", TypeOID: OIDName},
		{Name: "filter_definition", TypeOID: OIDText},
		{Name: "indnkeyatts", TypeOID: OIDInt2},
		{Name: "indnullsnotdistinct", TypeOID: OIDBool},
		{Name: "elements", TypeOID: OIDTextArray},
		{Name: "elements_is_expr", TypeOID: OIDBoolArray},
		{Name: "elements_opclass", TypeOID: OIDTextArray},
		{Name: "elements_opdefault", TypeOID: OIDBoolArray},
	}
	return &libravdb.SearchResults{Results: []*libravdb.SearchResult{}, Total: 0, Columns: columnNames(columns), ColumnTypes: columnOIDs(columns)}, columns, true
}

func handlePgDescriptionQuery() (*libravdb.SearchResults, []ColumnMeta, bool) {
	columns := []ColumnMeta{{Name: "description", TypeOID: OIDText}}
	return &libravdb.SearchResults{Columns: columnNames(columns), ColumnTypes: columnOIDs(columns)}, columns, true
}

func handlePgTableCommentQuery(db *libravdb.Database) (*libravdb.SearchResults, []ColumnMeta, bool) {
	columns := []ColumnMeta{
		{Name: "relname", TypeOID: OIDName},
		{Name: "description", TypeOID: OIDText},
	}
	rows := make([]*libravdb.SearchResult, 0)
	if db != nil {
		for _, tableName := range db.ListCollections() {
			rows = append(rows, &libravdb.SearchResult{
				ID:    tableName,
				Score: 1,
				Metadata: map[string]interface{}{
					"relname":     tableName,
					"description": nil,
				},
			})
		}
	}
	return &libravdb.SearchResults{Results: rows, Total: len(rows), Columns: columnNames(columns), ColumnTypes: columnOIDs(columns)}, columns, true
}

func handlePgCheckConstraintReflectionQuery(db *libravdb.Database) (*libravdb.SearchResults, []ColumnMeta, bool) {
	columns := []ColumnMeta{
		{Name: "relname", TypeOID: OIDName},
		{Name: "conname", TypeOID: OIDName},
		{Name: "condef", TypeOID: OIDText},
		{Name: "description", TypeOID: OIDText},
	}
	rows := make([]*libravdb.SearchResult, 0)
	if db == nil {
		return &libravdb.SearchResults{Results: rows, Total: 0, Columns: columnNames(columns), ColumnTypes: columnOIDs(columns)}, columns, true
	}
	for _, tableName := range db.ListCollections() {
		col, err := db.GetCollection(tableName)
		if err != nil {
			continue
		}
		cfg := col.Config()
		if len(cfg.CheckConstraints) == 0 {
			rows = append(rows, &libravdb.SearchResult{
				ID: tableName, Score: 1,
				Metadata: map[string]interface{}{
					"relname": tableName, "conname": nil, "condef": nil, "description": nil,
				},
			})
			continue
		}
		for _, check := range cfg.CheckConstraints {
			name := check.Name
			if name == "" {
				name = tableName + "_check"
			}
			definition := "CHECK (" + check.Expression + ")"
			rows = append(rows, &libravdb.SearchResult{
				ID: tableName, Score: 1,
				Metadata: map[string]interface{}{
					"relname": tableName, "conname": name, "condef": definition, "description": nil,
				},
			})
		}
	}
	return &libravdb.SearchResults{Results: rows, Total: len(rows), Columns: columnNames(columns), ColumnTypes: columnOIDs(columns)}, columns, true
}

func handlePgTypeInfoQuery(sql string, params *optimizer.ParameterSet) (*libravdb.SearchResults, []ColumnMeta, bool) {
	columns := []ColumnMeta{
		{Name: "name", TypeOID: OIDName},
		{Name: "oid", TypeOID: OIDOID},
		{Name: "array_oid", TypeOID: OIDOID},
		{Name: "regtype", TypeOID: OIDText},
		{Name: "delimiter", TypeOID: OIDChar},
	}
	rows := make([]*libravdb.SearchResult, 0, 1)
	name := catalogTypeLookupName(params)
	var oid, arrayOID int64
	switch strings.ToLower(name) {
	case "vector":
		oid, arrayOID = 16384, 16385
	case "bit":
		oid, arrayOID = 1560, 1561
	}
	if oid != 0 {
		rows = append(rows, &libravdb.SearchResult{
			ID: name, Score: 1,
			Metadata: map[string]interface{}{
				"name": name, "oid": oid, "array_oid": arrayOID,
				"regtype": name, "delimiter": ",",
			},
		})
	}
	return &libravdb.SearchResults{Results: rows, Total: len(rows), Columns: columnNames(columns), ColumnTypes: columnOIDs(columns)}, columns, true
}

func handleAsyncpgTypeInfoQuery(params *optimizer.ParameterSet) (*libravdb.SearchResults, []ColumnMeta, bool) {
	columns := []ColumnMeta{
		{Name: "oid", TypeOID: OIDOID},
		{Name: "ns", TypeOID: OIDName},
		{Name: "name", TypeOID: OIDName},
		{Name: "kind", TypeOID: OIDChar},
		{Name: "basetype", TypeOID: OIDOID},
		{Name: "elemtype", TypeOID: OIDOID},
		{Name: "elemdelim", TypeOID: OIDChar},
		{Name: "range_subtype", TypeOID: OIDOID},
		{Name: "attrtypoids", TypeOID: OIDOIDArray},
		{Name: "attrnames", TypeOID: OIDTextArray},
		{Name: "depth", TypeOID: OIDInt4},
		{Name: "basetype_name", TypeOID: OIDText},
		{Name: "elemtype_name", TypeOID: OIDText},
		{Name: "range_subtype_name", TypeOID: OIDText},
	}
	oids := asyncpgRequestedTypeOIDs(params)
	if len(oids) == 0 {
		oids = []uint32{OIDText, OIDInt8, OIDVector}
	}
	rows := make([]*libravdb.SearchResult, 0, len(oids))
	seen := make(map[uint32]struct{}, len(oids))
	// The real recursive query returns the array element/base type rows too.
	// asyncpg needs those rows to construct a derived codec; returning only
	// the requested array row leaves its element codec unresolved and causes
	// it to recursively issue the same introspection query.
	for next := 0; next < len(oids); next++ {
		oid := oids[next]
		if _, ok := seen[oid]; ok {
			continue
		}
		seen[oid] = struct{}{}
		name, namespace, elemType, delimiter, ok := asyncpgTypeInfo(oid)
		if !ok {
			continue
		}
		if elemOID, ok := asyncpgOIDValue(elemType); ok {
			if _, alreadySeen := seen[elemOID]; !alreadySeen {
				oids = append(oids, elemOID)
			}
		}
		elemTypeName := interface{}(nil)
		if elemOID, ok := asyncpgOIDValue(elemType); ok {
			elemTypeName = PGTypeName(elemOID)
		}
		metadata := map[string]interface{}{
			"oid": oid, "ns": namespace, "name": name, "kind": "b",
			"basetype": nil, "elemtype": elemType, "elemdelim": delimiter,
			"range_subtype": nil, "attrtypoids": nil, "attrnames": nil,
			"depth": int32(0), "basetype_name": nil, "elemtype_name": elemTypeName,
			"range_subtype_name": nil,
		}
		rows = append(rows, &libravdb.SearchResult{ID: name, Score: 1, Metadata: metadata})
	}
	return &libravdb.SearchResults{Results: rows, Total: len(rows), Columns: columnNames(columns), ColumnTypes: columnOIDs(columns)}, columns, true
}

func asyncpgOIDValue(value interface{}) (uint32, bool) {
	switch v := value.(type) {
	case uint32:
		return v, true
	case uint16:
		return uint32(v), true
	case uint64:
		if v <= uint64(^uint32(0)) {
			return uint32(v), true
		}
	case int:
		if v >= 0 && uint64(v) <= uint64(^uint32(0)) {
			return uint32(v), true
		}
	case int32:
		if v >= 0 {
			return uint32(v), true
		}
	case int64:
		if v >= 0 && uint64(v) <= uint64(^uint32(0)) {
			return uint32(v), true
		}
	}
	return 0, false
}

func asyncpgRequestedTypeOIDs(params *optimizer.ParameterSet) []uint32 {
	if params == nil || len(params.Positional) == 0 {
		return nil
	}
	raw := params.Positional[0].Bytes()
	oids := make([]uint32, 0, 4)
	var value uint64
	reading := false
	for _, b := range raw {
		if b >= '0' && b <= '9' {
			reading = true
			value = value*10 + uint64(b-'0')
			continue
		}
		if reading {
			if value <= uint64(^uint32(0)) {
				oids = append(oids, uint32(value))
			}
			value = 0
			reading = false
		}
	}
	if reading && value <= uint64(^uint32(0)) {
		oids = append(oids, uint32(value))
	}
	return oids
}

func asyncpgTypeInfo(oid uint32) (name, namespace string, elemType interface{}, delimiter interface{}, ok bool) {
	switch oid {
	case OIDBool:
		return "bool", "pg_catalog", nil, nil, true
	case OIDInt4:
		return "int4", "pg_catalog", nil, nil, true
	case OIDInt8:
		return "int8", "pg_catalog", nil, nil, true
	case OIDOID:
		return "oid", "pg_catalog", nil, nil, true
	case OIDChar:
		return "char", "pg_catalog", nil, nil, true
	case OIDName:
		return "name", "pg_catalog", nil, nil, true
	case OIDFloat4:
		return "float4", "pg_catalog", nil, nil, true
	case OIDFloat8:
		return "float8", "pg_catalog", nil, nil, true
	case OIDText:
		return "text", "pg_catalog", nil, nil, true
	case OIDJSON:
		return "json", "pg_catalog", nil, nil, true
	case OIDJSONB:
		return "jsonb", "pg_catalog", nil, nil, true
	case OIDUUID:
		return "uuid", "pg_catalog", nil, nil, true
	case OIDTimestamp:
		return "timestamp", "pg_catalog", nil, nil, true
	case OIDTimestamptz:
		return "timestamptz", "pg_catalog", nil, nil, true
	case OIDVector:
		return "vector", "public", nil, nil, true
	case OIDTextArray:
		return "_text", "pg_catalog", OIDText, ",", true
	case OIDInt4Array:
		return "_int4", "pg_catalog", OIDInt4, ",", true
	case OIDInt8Array:
		return "_int8", "pg_catalog", OIDInt8, ",", true
	case OIDFloat4Array:
		return "_float4", "pg_catalog", OIDFloat4, ",", true
	case OIDFloat8Array:
		return "_float8", "pg_catalog", OIDFloat8, ",", true
	case OIDOIDArray:
		return "_oid", "pg_catalog", OIDOID, ",", true
	default:
		return "", "", nil, nil, false
	}
}

func handlePgClassNamespaceQuery(db *libravdb.Database, sql string) (*libravdb.SearchResults, []ColumnMeta, bool) {
	upper := strings.ToUpper(sql)
	projection := upper
	if start := strings.Index(projection, "SELECT "); start >= 0 {
		projection = projection[start+len("SELECT "):]
	}
	if end := strings.Index(projection, "FROM PG_CLASS"); end >= 0 {
		projection = projection[:end]
	} else if end := strings.Index(projection, "FROM\nPG_CLASS"); end >= 0 {
		projection = projection[:end]
	}
	var columns []ColumnMeta
	// Django's get_table_list() asks for exactly (relname, relkind,
	// obj_description). Keep this shape distinct from SQLAlchemy's simpler
	// pg_class/pg_namespace projections, which may request oid and relname.
	if strings.Contains(projection, "OBJ_DESCRIPTION(") {
		columns = []ColumnMeta{
			{Name: "relname", TypeOID: OIDName},
			{Name: "type", TypeOID: OIDChar},
			{Name: "comment", TypeOID: OIDText},
		}
	} else {
		if strings.Contains(projection, "OID") {
			columns = append(columns, ColumnMeta{Name: "oid", TypeOID: OIDOID})
		}
		if strings.Contains(projection, "RELNAME") {
			columns = append(columns, ColumnMeta{Name: "relname", TypeOID: OIDName})
		}
	}
	if len(columns) == 0 {
		columns = []ColumnMeta{{Name: "relname", TypeOID: OIDName}}
	}
	names := []string{}
	if db != nil {
		names = db.ListCollections()
	}
	rows := make([]*libravdb.SearchResult, 0, len(names))
	for i, name := range names {
		rows = append(rows, &libravdb.SearchResult{
			ID: name, Score: 1,
			Metadata: map[string]interface{}{
				"oid":            int64(100 + i),
				"relname":        name,
				"type":           "t",
				"comment":        nil,
				"relnamespace":   int64(2200),
				"relkind":        "r",
				"relpersistence": "p",
			},
		})
	}
	return &libravdb.SearchResults{Results: rows, Total: len(rows), Columns: columnNames(columns), ColumnTypes: columnOIDs(columns)}, columns, true
}

func handlePgDomainQuery() (*libravdb.SearchResults, []ColumnMeta, bool) {
	columns := []ColumnMeta{
		{Name: "name", TypeOID: OIDName},
		{Name: "attype", TypeOID: OIDText},
		{Name: "nullable", TypeOID: OIDBool},
		{Name: "default", TypeOID: OIDText},
		{Name: "visible", TypeOID: OIDBool},
		{Name: "schema", TypeOID: OIDName},
		{Name: "condefs", TypeOID: OIDTextArray},
		{Name: "connames", TypeOID: OIDTextArray},
		{Name: "collname", TypeOID: OIDName},
	}
	return &libravdb.SearchResults{Results: []*libravdb.SearchResult{}, Total: 0, Columns: columnNames(columns), ColumnTypes: columnOIDs(columns)}, columns, true
}

func handlePgEnumQuery() (*libravdb.SearchResults, []ColumnMeta, bool) {
	columns := []ColumnMeta{
		{Name: "name", TypeOID: OIDName},
		{Name: "visible", TypeOID: OIDBool},
		{Name: "schema", TypeOID: OIDName},
		{Name: "labels", TypeOID: OIDTextArray},
	}
	return &libravdb.SearchResults{Results: []*libravdb.SearchResult{}, Total: 0, Columns: columnNames(columns), ColumnTypes: columnOIDs(columns)}, columns, true
}

func handleSQLAlchemyColumnsQuery(db *libravdb.Database, sql string) (*libravdb.SearchResults, []ColumnMeta, bool) {
	columns := []ColumnMeta{
		{Name: "name", TypeOID: OIDName},
		{Name: "format_type", TypeOID: OIDText},
		{Name: "default", TypeOID: OIDText},
		{Name: "not_null", TypeOID: OIDBool},
		{Name: "table_name", TypeOID: OIDName},
		{Name: "comment", TypeOID: OIDText},
		{Name: "generated", TypeOID: OIDChar},
		{Name: "identity_options", TypeOID: OIDJSON},
		{Name: "collation", TypeOID: OIDName},
	}
	rows := make([]*libravdb.SearchResult, 0)
	if db != nil {
		for _, tableName := range db.ListCollections() {
			col, err := db.GetCollection(tableName)
			if err != nil {
				continue
			}
			for _, field := range collectionCatalogFields(col.Config(), col.Dimension()) {
				formatType := sqlAlchemyFormatType(field.udt)
				rows = append(rows, &libravdb.SearchResult{
					ID: field.name, Score: 1,
					Metadata: map[string]interface{}{
						"name":             field.name,
						"format_type":      formatType,
						"default":          nil,
						"not_null":         field.notNull,
						"table_name":       tableName,
						"comment":          nil,
						"generated":        "",
						"identity_options": nil,
						"collation":        nil,
					},
				})
			}
		}
	}
	return &libravdb.SearchResults{Results: rows, Total: len(rows), Columns: columnNames(columns), ColumnTypes: columnOIDs(columns)}, columns, true
}

func handleDjangoTableDescriptionQuery(db *libravdb.Database, sql string, params *optimizer.ParameterSet) (*libravdb.SearchResults, []ColumnMeta, bool) {
	columns := []ColumnMeta{
		{Name: "column_name", TypeOID: OIDName},
		{Name: "is_nullable", TypeOID: OIDBool},
		{Name: "column_default", TypeOID: OIDText},
		{Name: "collation", TypeOID: OIDText},
		{Name: "is_autofield", TypeOID: OIDBool},
		{Name: "column_comment", TypeOID: OIDText},
	}
	rows := make([]*libravdb.SearchResult, 0)
	name := catalogTargetTable(sql, params)
	if db != nil {
		if col, err := db.GetCollection(name); err == nil {
			cfg := col.Config()
			for _, field := range collectionCatalogFields(cfg, col.Dimension()) {
				var defaultValue interface{}
				if value, ok := cfg.ColumnDefaults[field.name]; ok {
					defaultValue = value
				}
				flags := cfg.ColumnConstraints[field.name]
				rows = append(rows, &libravdb.SearchResult{
					ID: field.name, Score: 1,
					Metadata: map[string]interface{}{
						"column_name":    field.name,
						"is_nullable":    !field.notNull,
						"column_default": defaultValue,
						"collation":      nil,
						"is_autofield":   flags&catalog.ColFlagHasDefault != 0 && strings.EqualFold(field.name, "id"),
						"column_comment": nil,
					},
				})
			}
		}
	}
	return &libravdb.SearchResults{Results: rows, Total: len(rows), Columns: columnNames(columns), ColumnTypes: columnOIDs(columns)}, columns, true
}

func handleDjangoRelationsQuery(db *libravdb.Database, sql string, params *optimizer.ParameterSet) (*libravdb.SearchResults, []ColumnMeta, bool) {
	columns := []ColumnMeta{
		{Name: "source_column", TypeOID: OIDName},
		{Name: "target_table", TypeOID: OIDName},
		{Name: "target_column", TypeOID: OIDName},
	}
	rows := make([]*libravdb.SearchResult, 0)
	name := catalogTargetTable(sql, params)
	if db != nil {
		if col, err := db.GetCollection(name); err == nil {
			for _, foreignKey := range col.Config().ForeignKeys {
				if !strings.EqualFold(foreignKey.SourceTable, name) {
					continue
				}
				rows = append(rows, &libravdb.SearchResult{
					ID: foreignKey.SourceColumn, Score: 1,
					Metadata: map[string]interface{}{
						"source_column": foreignKey.SourceColumn,
						"target_table":  foreignKey.TargetTable,
						"target_column": foreignKey.TargetColumn,
					},
				})
			}
		}
	}
	return &libravdb.SearchResults{Results: rows, Total: len(rows), Columns: columnNames(columns), ColumnTypes: columnOIDs(columns)}, columns, true
}

func handleDjangoConstraintQuery(db *libravdb.Database, sql string, params *optimizer.ParameterSet) (*libravdb.SearchResults, []ColumnMeta, bool) {
	columns := []ColumnMeta{
		{Name: "conname", TypeOID: OIDName},
		{Name: "columns", TypeOID: OIDTextArray},
		{Name: "contype", TypeOID: OIDChar},
		{Name: "used_cols", TypeOID: OIDText},
		{Name: "reloptions", TypeOID: OIDTextArray},
	}
	rows := make([]*libravdb.SearchResult, 0)
	name := catalogTargetTable(sql, params)
	if db != nil {
		if col, err := db.GetCollection(name); err == nil {
			cfg := col.Config()
			primaryKey := "id"
			if len(cfg.PrimaryKeyColumns) == 1 {
				primaryKey = cfg.PrimaryKeyColumns[0]
			}
			rows = append(rows, &libravdb.SearchResult{
				ID: name + "_pkey", Score: 1,
				Metadata: map[string]interface{}{
					"conname":    name + "_pkey",
					"columns":    []string{primaryKey},
					"contype":    "p",
					"used_cols":  nil,
					"reloptions": nil,
				},
			})
		}
	}
	return &libravdb.SearchResults{Results: rows, Total: len(rows), Columns: columnNames(columns), ColumnTypes: columnOIDs(columns)}, columns, true
}

func handleDjangoIndexQuery() (*libravdb.SearchResults, []ColumnMeta, bool) {
	columns := []ColumnMeta{
		{Name: "indexname", TypeOID: OIDName},
		{Name: "columns", TypeOID: OIDTextArray},
		{Name: "indisunique", TypeOID: OIDBool},
		{Name: "indisprimary", TypeOID: OIDBool},
		{Name: "orders", TypeOID: OIDTextArray},
		{Name: "type", TypeOID: OIDName},
		{Name: "exprdef", TypeOID: OIDText},
		{Name: "options", TypeOID: OIDTextArray},
	}
	return &libravdb.SearchResults{Results: []*libravdb.SearchResult{}, Total: 0, Columns: columnNames(columns), ColumnTypes: columnOIDs(columns)}, columns, true
}

func sqlAlchemyFormatType(udt string) string {
	switch strings.ToLower(udt) {
	case "int2":
		return "SMALLINT"
	case "int4":
		return "INTEGER"
	case "int8":
		return "BIGINT"
	case "float4":
		return "REAL"
	case "float8":
		return "DOUBLE PRECISION"
	case "_float4":
		return "REAL[]"
	case "_float8":
		return "DOUBLE PRECISION[]"
	case "_int4":
		return "INTEGER[]"
	case "_text":
		return "TEXT[]"
	case "json":
		return "JSON"
	case "jsonb":
		return "JSONB"
	case "uuid":
		return "UUID"
	case "varchar":
		return "VARCHAR"
	case "bool":
		return "BOOLEAN"
	default:
		return strings.ToUpper(udt)
	}
}

func handlePgConstraintReflectionQuery(db *libravdb.Database, sql string, params *optimizer.ParameterSet) (*libravdb.SearchResults, []ColumnMeta, bool) {
	columns := []ColumnMeta{
		{Name: "conrelid", TypeOID: OIDOID},
		{Name: "cols", TypeOID: OIDTextArray},
		{Name: "conname", TypeOID: OIDName},
		{Name: "description", TypeOID: OIDText},
		{Name: "indnkeyatts", TypeOID: OIDInt2},
		{Name: "indnullsnotdistinct", TypeOID: OIDBool},
	}
	rows := make([]*libravdb.SearchResult, 0)
	if catalogHasTextParam(params, "u") {
		return &libravdb.SearchResults{Results: rows, Total: 0, Columns: columnNames(columns), ColumnTypes: columnOIDs(columns)}, columns, true
	}
	if db != nil {
		for i, tableName := range db.ListCollections() {
			rows = append(rows, &libravdb.SearchResult{
				ID: tableName, Score: 1,
				Metadata: map[string]interface{}{
					"conrelid":            int64(100 + i),
					"cols":                []string{"id"},
					"conname":             tableName + "_pkey",
					"description":         nil,
					"indnkeyatts":         int64(1),
					"indnullsnotdistinct": false,
				},
			})
		}
	}
	return &libravdb.SearchResults{Results: rows, Total: len(rows), Columns: columnNames(columns), ColumnTypes: columnOIDs(columns)}, columns, true
}

// handlePgForeignKeyReflectionQuery supplies the projection emitted by
// SQLAlchemy's PostgreSQL get_multi_foreign_keys implementation.  The
// condef text deliberately follows pg_get_constraintdef's parseable shape so
// SQLAlchemy can retain its normal FK parsing and Alembic comparison logic.
func handlePgForeignKeyReflectionQuery(db *libravdb.Database) (*libravdb.SearchResults, []ColumnMeta, bool) {
	columns := []ColumnMeta{
		{Name: "relname", TypeOID: OIDName},
		{Name: "conname", TypeOID: OIDName},
		{Name: "condef", TypeOID: OIDText},
		{Name: "nspname", TypeOID: OIDName},
		{Name: "description", TypeOID: OIDText},
	}
	rows := make([]*libravdb.SearchResult, 0)
	if db == nil {
		return &libravdb.SearchResults{Results: rows, Total: 0, Columns: columnNames(columns), ColumnTypes: columnOIDs(columns)}, columns, true
	}

	for _, tableName := range db.ListCollections() {
		col, err := db.GetCollection(tableName)
		if err != nil {
			continue
		}
		cfg := col.Config()
		if len(cfg.ForeignKeys) == 0 {
			// PostgreSQL returns one NULL constraint row for a table with no
			// foreign keys. SQLAlchemy uses it to populate an empty list.
			rows = append(rows, pgForeignKeyReflectionRow(tableName, nil, "", "", ""))
			continue
		}

		// Composite constraints are stored as one ForeignKeyInfo per column.
		// Group adjacent entries with the same logical constraint before
		// rebuilding the PostgreSQL constraint definition.
		type fkGroup struct {
			name, targetTable, targetSchema string
			sourceColumns, targetColumns    []string
			onDelete, onUpdate              uint8
		}
		groups := make([]fkGroup, 0, len(cfg.ForeignKeys))
		groupIndex := make(map[string]int, len(cfg.ForeignKeys))
		for _, fk := range cfg.ForeignKeys {
			name := fk.Name
			if name == "" {
				name = tableName + "_" + fk.SourceColumn + "_fkey"
			}
			key := name + "\x00" + fk.TargetTable
			idx, ok := groupIndex[key]
			if !ok {
				idx = len(groups)
				groupIndex[key] = idx
				groups = append(groups, fkGroup{
					name: name, targetTable: fk.TargetTable, targetSchema: "public",
					onDelete: fk.OnDelete, onUpdate: fk.OnUpdate,
				})
			}
			groups[idx].sourceColumns = append(groups[idx].sourceColumns, fk.SourceColumn)
			groups[idx].targetColumns = append(groups[idx].targetColumns, fk.TargetColumn)
		}
		for _, group := range groups {
			condef := "FOREIGN KEY (" + strings.Join(group.sourceColumns, ", ") + ") REFERENCES " +
				group.targetTable + "(" + strings.Join(group.targetColumns, ", ") + ")"
			condef += foreignKeyActionClause("DELETE", group.onDelete)
			condef += foreignKeyActionClause("UPDATE", group.onUpdate)
			rows = append(rows, pgForeignKeyReflectionRow(tableName, group.name, condef, group.targetSchema, ""))
		}
	}

	return &libravdb.SearchResults{Results: rows, Total: len(rows), Columns: columnNames(columns), ColumnTypes: columnOIDs(columns)}, columns, true
}

func pgForeignKeyReflectionRow(tableName string, conname interface{}, condef, schema, description string) *libravdb.SearchResult {
	return &libravdb.SearchResult{
		ID:    tableName,
		Score: 1,
		Metadata: map[string]interface{}{
			"relname":     tableName,
			"conname":     conname,
			"condef":      nullableCatalogText(condef),
			"nspname":     nullableCatalogText(schema),
			"description": nullableCatalogText(description),
		},
	}
}

func nullableCatalogText(value string) interface{} {
	if value == "" {
		return nil
	}
	return value
}

func foreignKeyActionClause(kind string, action uint8) string {
	var value string
	switch action {
	case catalog.OnDeleteCascade:
		value = "CASCADE"
	case catalog.OnDeleteRestrict:
		value = "RESTRICT"
	case catalog.OnDeleteSetNull:
		value = "SET NULL"
	case catalog.OnDeleteSetDefault:
		value = "SET DEFAULT"
	default:
		return ""
	}
	return " ON " + kind + " " + value
}

func catalogPredicateValue(sql, marker string) string {
	upper := strings.ToUpper(sql)
	if at := strings.Index(upper, marker); at >= 0 {
		if quote := strings.IndexByte(sql[at:], '\''); quote >= 0 {
			start := at + quote + 1
			if end := strings.IndexByte(sql[start:], '\''); end >= 0 {
				return sql[start : start+end]
			}
		}
	}
	return ""
}

func catalogConstraintRows(sql string, db *libravdb.Database, params *optimizer.ParameterSet, columns []ColumnMeta) *libravdb.SearchResults {
	name := catalogTargetTable(sql, params)
	if db == nil {
		return &libravdb.SearchResults{Results: nil, Total: 0, Columns: columnNames(columns), ColumnTypes: columnOIDs(columns)}
	}
	col, err := db.GetCollection(name)
	if err != nil {
		return &libravdb.SearchResults{Results: nil, Total: 0, Columns: columnNames(columns), ColumnTypes: columnOIDs(columns)}
	}
	cfg := col.Config()
	pk := ""
	if len(cfg.PrimaryKeyColumns) > 0 {
		pk = cfg.PrimaryKeyColumns[0]
	} else {
		pk = "id"
	}
	if strings.Contains(strings.ToUpper(sql), "CONSTRAINT_TYPE = 'UNIQUE'") {
		return &libravdb.SearchResults{Results: nil, Total: 0, Columns: columnNames(columns), ColumnTypes: columnOIDs(columns)}
	}
	if len(columns) == 1 {
		return catalogRows(columns, map[string]interface{}{"constraint_name": name + "_pkey"})
	}
	return catalogRows(columns, map[string]interface{}{"column_name": pk, "constraint_name": name + "_pkey", "constraint_type": "PRIMARY KEY"})
}

func catalogTargetTable(sql string, params *optimizer.ParameterSet) string {
	if params != nil {
		for i := len(params.Positional) - 1; i >= 0; i-- {
			v := params.Positional[i]
			if v.Kind != optimizer.ScalarString && v.Kind != optimizer.ScalarBytes {
				continue
			}
			candidate := string(v.BytesData)
			upper := strings.ToUpper(candidate)
			if candidate != "" && upper != "BASE TABLE" && upper != "UNIQUE" && upper != "PRIMARY KEY" && upper != "CURRENT_SCHEMA" {
				return candidate
			}
		}
	}
	upper := strings.ToUpper(sql)
	for _, marker := range []string{"TABLE_NAME", "TABLENAME", "RELNAME"} {
		for searchAt := 0; searchAt < len(upper); {
			relativeAt := strings.Index(upper[searchAt:], marker)
			if relativeAt < 0 {
				break
			}
			at := searchAt + relativeAt
			searchAt = at + len(marker)
			if (at > 0 && isSQLIdentifierByte(upper[at-1])) ||
				(at+len(marker) < len(upper) && isSQLIdentifierByte(upper[at+len(marker)])) {
				continue
			}
			tail := sql[at+len(marker):]
			if equals := strings.IndexByte(tail, '='); equals >= 0 {
				if quote := strings.IndexByte(tail[equals+1:], '\''); quote >= 0 {
					start := at + len(marker) + equals + quote + 2
					if end := strings.IndexByte(sql[start:], '\''); end >= 0 {
						return sql[start : start+end]
					}
				}
			}
		}
	}
	return ""
}

func isSQLIdentifierByte(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9') || b == '_'
}

func catalogConstraintType(params *optimizer.ParameterSet) string {
	if params == nil || len(params.Positional) < 2 {
		return ""
	}
	value := params.Positional[1]
	if value.Kind != optimizer.ScalarString && value.Kind != optimizer.ScalarBytes {
		return ""
	}
	return strings.ToLower(string(value.BytesData))
}

func catalogTypeLookupName(params *optimizer.ParameterSet) string {
	if params == nil {
		return ""
	}
	for _, value := range params.Positional {
		if value.Kind == optimizer.ScalarString || value.Kind == optimizer.ScalarBytes {
			return string(value.BytesData)
		}
	}
	for _, named := range params.Named {
		value := named.Value
		if value.Kind == optimizer.ScalarString || value.Kind == optimizer.ScalarBytes {
			return string(value.BytesData)
		}
	}
	return ""
}

func catalogHasTextParam(params *optimizer.ParameterSet, want string) bool {
	if params == nil {
		return false
	}
	for _, value := range params.Positional {
		if (value.Kind == optimizer.ScalarString || value.Kind == optimizer.ScalarBytes) && strings.EqualFold(string(value.BytesData), want) {
			return true
		}
	}
	for _, named := range params.Named {
		value := named.Value
		if (value.Kind == optimizer.ScalarString || value.Kind == optimizer.ScalarBytes) && strings.EqualFold(string(value.BytesData), want) {
			return true
		}
	}
	return false
}

func catalogRows(columns []ColumnMeta, metadata map[string]interface{}) *libravdb.SearchResults {
	return &libravdb.SearchResults{Results: []*libravdb.SearchResult{{ID: "", Score: 1, Metadata: metadata}}, Total: 1, Columns: columnNames(columns), ColumnTypes: columnOIDs(columns)}
}

func columnNames(columns []ColumnMeta) []string {
	names := make([]string, len(columns))
	for i := range columns {
		names[i] = columns[i].Name
	}
	return names
}

func columnOIDs(columns []ColumnMeta) []uint16 {
	oids := make([]uint16, len(columns))
	for i := range columns {
		oids[i] = uint16(columns[i].TypeOID)
	}
	return oids
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

func standaloneSystemFunction(sql, fn string) bool {
	clean := strings.Map(func(r rune) rune {
		if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
			return -1
		}
		return r
	}, sql)
	return strings.EqualFold(clean, "SELECT"+fn)
}
