package pgwire

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/xDarkicex/lexer"
	"github.com/xDarkicex/lexer/parser"
	"github.com/xDarkicex/libravdb/internal/catalog"
	"github.com/xDarkicex/libravdb/libravdb"
)

// describeStatement statically determines the parameter type OIDs and the
// result RowDescription for a prepared SQL statement, without executing it.
//
// The extended-protocol Describe (Statement and Portal) must report what a
// statement returns before it runs. The engine has no plan-only mode, so the
// shape is derived from the parse tree plus catalog metadata. Column names
// mirror the optimizer's projection lowering exactly; identifier columns
// resolve their type from the catalog so a RowDescription is correct even
// before any rows exist.
//
// The result intentionally matches what Execute will report: the executor
// sends its own RowDescription, and clients decode rows against that one, so
// a Describe that disagrees with Execute would desynchronize buffers.
func describeStatement(db *libravdb.Database, query string, paramCount int) ([]uint32, []ColumnMeta, error) {
	trimmed := strings.TrimSpace(strings.TrimRight(query, ";"))
	// Keep Describe in lockstep with both simple and extended execution: the
	// parser accepts system tables by their bare names, while pgwire clients
	// commonly qualify them with pg_catalog.
	trimmed = rewritePgCatalogQuery(trimmed)

	// asyncpg temporarily disables JIT while resolving a newly discovered
	// type, then restores it with set_config('jit', $1, false). The execution
	// path handles this as a session query, so Describe must report the
	// restore value as text as well; leaving it unspecified makes asyncpg ask
	// the recursive type lookup for OID 0 and retry forever.
	if isAsyncpgJITQuery(trimmed) {
		paramOIDs := make([]uint32, paramCount)
		if len(paramOIDs) > 0 {
			paramOIDs[0] = OIDText
		}
		return paramOIDs, []ColumnMeta{{Name: "cur", TypeOID: OIDText}, {Name: "new", TypeOID: OIDText}}, nil
	}

	// System functions and pg_catalog introspection produce synthetic results.
	// Checked before parsing because the parser has no grammar for VERSION(),
	// CURRENT_DATABASE(), and friends.
	if _, columns, handled := interceptSystemQuery(trimmed, db); handled {
		paramOIDs := make([]uint32, paramCount)
		// asyncpg sends its recursive type lookup with a binary oid[] bind.
		// This query is intercepted before normal expression inference, so its
		// parameter type must be supplied explicitly or the binary Bind value
		// is treated as an unknown byte slice and asyncpg recursively retries
		// the same introspection query.
		if isAsyncpgTypeInfoQuery(trimmed) && len(paramOIDs) > 0 {
			paramOIDs[0] = OIDOIDArray
		}
		return paramOIDs, columns, nil
	}

	src := []byte(trimmed)
	doc := &parser.QueryDoc{}
	if err := parser.Parse(src, doc); err != nil {
		return nil, nil, fmt.Errorf("parse error: %w", err)
	}

	cat := db.Catalog()
	if cat == nil {
		return nil, nil, fmt.Errorf("catalog not initialized")
	}
	if doc.Explain {
		if !doc.ExplainAnalyze {
			return nil, nil, fmt.Errorf("EXPLAIN without ANALYZE is not supported; use EXPLAIN ANALYZE")
		}
		if !describeHasGraphJoin(doc) {
			return nil, nil, fmt.Errorf("EXPLAIN ANALYZE currently supports graph queries only")
		}
		markParams(doc, src)
		binder := catalog.NewBinder(cat, src)
		if err := binder.Bind(doc); err != nil {
			return nil, nil, fmt.Errorf("bind error: %w", err)
		}
		paramOIDs := inferParamOIDs(doc, src, cat, nil, paramCount)
		return paramOIDs, []ColumnMeta{{Name: libravdb.SQLExplainColumn, TypeOID: OIDJSONB}}, nil
	}
	if dmlColumns, dml := describeDMLReturning(db, doc, src); dml {
		return inferParamOIDs(doc, src, cat, nil, paramCount), dmlColumns, nil
	}

	// Transaction controls and DML/DDL statements produce no result rows, so
	// their Describe is NoData.
	root, ok := topLevelStatement(doc)
	if !ok {
		// Non-returning DML still needs its parameter OIDs. Strict clients such
		// as asyncpg use them to select codecs before Bind; returning all zeros
		// makes the client issue a recursive pg_type introspection query for
		// ordinary INSERT/UPDATE/DELETE statements.
		return inferParamOIDs(doc, src, cat, nil, paramCount), nil, nil
	}

	var columns []ColumnMeta
	var byOID map[uint32]*catalog.TableDef // table scope for column type resolution
	if root.Kind == parser.NodeKindComputeLeidenStmt {
		columns = leidenColumns()
	} else {
		stmt := &doc.SelectStmts[root.ID]
		if starColumns, ok := describeCollectionStar(db, doc, src, stmt); ok {
			columns = starColumns
		}

		// Parameters are native query values, so they never resolve against the
		// catalog. Pre-mark them so the binder skips
		// them. $N is already handled by the binder, but @name is not; marking
		// both uniformly avoids a spurious "identifier not found".
		markParams(doc, src)

		if len(columns) > 0 {
			// SELECT * still needs a normal bind/scope pass so parameters in
			// predicates inherit the catalog type of the compared column.
			binder := catalog.NewBinder(cat, src)
			if err := binder.Bind(doc); err != nil {
				return nil, nil, fmt.Errorf("bind error: %w", err)
			}
			byOID = buildScope(cat, src, stmt, doc)
			// Bare SELECT * has already been expanded from the collection schema.
			// This keeps Describe Statement/Portal byte-for-byte aligned with the
			// execution RowDescription used by database/sql and GORM.
		} else if stmt.CTEsCount > 0 || describeSelectHasDerivedRelation(doc, stmt) || describeSelectHasWindow(doc, stmt) || describeSelectHasTemporalRange(doc, stmt) || len(doc.SubqueryExprs) > 0 {
			// A Leiden CTE names a virtual relation (the CTE) that is absent
			// from the catalog; derived and correlated subqueries likewise have
			// query-local scope. Describe the projection list leniently instead.
			columns = describeProjectionsLenient(db, doc, src, stmt)
		} else {
			binder := catalog.NewBinder(cat, src)
			if err := binder.Bind(doc); err != nil {
				// A query the engine cannot bind cannot execute either; surface
				// the failure now, matching PostgreSQL's describe-time errors.
				return nil, nil, fmt.Errorf("bind error: %w", err)
			}
			byOID = buildScope(cat, src, stmt, doc)
			columns = describeSelect(doc, src, cat, stmt, byOID)
		}
	}

	paramOIDs := inferParamOIDs(doc, src, cat, byOID, paramCount)
	return paramOIDs, columns, nil
}

// describeCollectionStar expands a bare SELECT * against the same collection
// schema used by the executor.  A previous default id/score description was
// correct for search results but wrong for relational SELECT *: GORM (and
// pgx) use the Describe result-format count when decoding Execute rows, so a
// two-column description followed by a three-column execution result causes a
// protocol reset.  Return false for joins, derived tables, and unknown/system
// relations, which continue through the existing projection describer.
func describeCollectionStar(db *libravdb.Database, doc *parser.QueryDoc, src []byte, stmt *parser.SelectStmt) ([]ColumnMeta, bool) {
	if db == nil || doc == nil || stmt == nil || stmt.FromTable.Kind != parser.NodeKindTableExpr || stmt.FromTable.ID < 0 || int(stmt.FromTable.ID) >= len(doc.TableExprs) || stmt.ProjectionsCount != 1 {
		return nil, false
	}
	proj := doc.Projections[stmt.ProjectionsStart]
	if !proj.Star {
		return nil, false
	}
	table := doc.TableExprs[stmt.FromTable.ID]
	if table.IsDerived || table.Start >= table.End || table.End > uint32(len(src)) {
		return nil, false
	}
	col, err := db.GetCollection(string(src[table.Start:table.End]))
	if err != nil {
		return nil, false
	}
	cfg := col.Config()
	names := make([]string, 0, len(cfg.MetadataSchema)+1)
	columns := make([]ColumnMeta, 0, len(cfg.MetadataSchema)+1)
	var idOID uint32 = OIDText
	for name, field := range cfg.MetadataSchema {
		if strings.EqualFold(name, "id") {
			idOID = collectionFieldOID(field)
			break
		}
	}
	columns = append(columns, ColumnMeta{Name: "id", TypeOID: idOID})
	for name := range cfg.MetadataSchema {
		if strings.EqualFold(name, "id") {
			continue
		}
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		columns = append(columns, ColumnMeta{Name: name, TypeOID: collectionFieldOID(cfg.MetadataSchema[name])})
	}
	return columns, true
}

func collectionFieldOID(field libravdb.FieldType) uint32 {
	switch field {
	case libravdb.IntField:
		return OIDInt4
	case libravdb.BigIntField:
		return OIDInt8
	case libravdb.FloatField:
		return OIDFloat8
	case libravdb.BoolField:
		return OIDBool
	case libravdb.TimeField:
		return OIDTimestamptz
	case libravdb.JSONField:
		return OIDJSON
	case libravdb.JSONBField:
		return OIDJSONB
	case libravdb.StringArrayField:
		return OIDTextArray
	case libravdb.IntArrayField:
		return OIDInt4Array
	case libravdb.FloatArrayField:
		return OIDFloat4Array
	default:
		return OIDText
	}
}

func describeDMLReturning(db *libravdb.Database, doc *parser.QueryDoc, src []byte) ([]ColumnMeta, bool) {
	var refs []parser.NodeRef
	var star bool
	var tableStart, tableEnd uint32
	switch {
	case len(doc.InsertStmts) > 0:
		refs, star = doc.InsertStmts[0].Returning, doc.InsertStmts[0].ReturningStar
		tableStart, tableEnd = doc.InsertStmts[0].TableStart, doc.InsertStmts[0].TableEnd
	case len(doc.UpdateStmts) > 0:
		refs, star = doc.UpdateStmts[0].Returning, doc.UpdateStmts[0].ReturningStar
		tableStart, tableEnd = doc.UpdateStmts[0].TableStart, doc.UpdateStmts[0].TableEnd
	case len(doc.DeleteStmts) > 0:
		refs, star = doc.DeleteStmts[0].Returning, doc.DeleteStmts[0].ReturningStar
		tableStart, tableEnd = doc.DeleteStmts[0].TableStart, doc.DeleteStmts[0].TableEnd
	default:
		return nil, false
	}
	if !star && len(refs) == 0 {
		return nil, false
	}
	if star {
		columns := []ColumnMeta{{Name: "id", TypeOID: OIDText}}
		if db != nil && tableEnd > tableStart && tableEnd <= uint32(len(src)) {
			if col, err := db.GetCollection(string(src[tableStart:tableEnd])); err == nil {
				cfg := col.Config()
				names := make([]string, 0, len(cfg.MetadataSchema)+1)
				for name := range cfg.MetadataSchema {
					if strings.EqualFold(name, "id") {
						continue
					}
					names = append(names, name)
				}
				if col.Dimension() > 0 {
					names = append(names, "embedding")
				}
				sort.Strings(names)
				for _, name := range names {
					columns = append(columns, ColumnMeta{Name: name, TypeOID: OIDText})
				}
			}
		}
		return columns, true
	}
	columns := make([]ColumnMeta, 0, len(refs))
	for _, ref := range refs {
		if ref.Kind != parser.NodeKindIdentifier || ref.ID < 0 || int(ref.ID) >= len(doc.Identifiers) {
			return nil, false
		}
		id := doc.Identifiers[ref.ID]
		if id.End <= id.Start || id.End > uint32(len(src)) {
			return nil, false
		}
		name := string(src[id.Start:id.End])
		if dot := strings.LastIndexByte(name, '.'); dot >= 0 {
			name = name[dot+1:]
		}
		oid := uint32(OIDText)
		if strings.EqualFold(name, "id") {
			oid = OIDText
		}
		columns = append(columns, ColumnMeta{Name: name, TypeOID: oid})
	}
	return columns, true
}

// topLevelStatement returns the first row-producing statement root in the
// document. The parser appends statement roots to doc.Nodes before the match
// path elements they contain, so scanning is required rather than indexing
// position zero.
func topLevelStatement(doc *parser.QueryDoc) (parser.NodeRef, bool) {
	for i := range doc.Nodes {
		switch doc.Nodes[i].Kind {
		case parser.NodeKindSelectStmt, parser.NodeKindComputeLeidenStmt:
			return doc.Nodes[i], true
		}
	}
	return parser.NodeRef{}, false
}

// markParams marks $N/@name parameter identifiers as resolved so the binder
// skips catalog resolution for them.
func markParams(doc *parser.QueryDoc, src []byte) {
	for i := range doc.Identifiers {
		id := &doc.Identifiers[i]
		if id.ResolvedKind == parser.ResolvedKindUnknown {
			c := src[id.Start]
			if c == '$' || c == '@' {
				id.ResolvedKind = parser.ResolvedKindColumn
			}
		}
	}
}

// describeSelect builds the RowDescription columns for a SELECT statement by
// walking its projection list. The logic mirrors the optimizer's projection
// lowering so the column count and names agree with Execute.
func describeSelect(doc *parser.QueryDoc, src []byte, cat *catalog.Catalog, stmt *parser.SelectStmt, byOID map[uint32]*catalog.TableDef) []ColumnMeta {
	// Aggregate-only queries use the compact aggregate description below. A
	// grouped aggregate must walk the projection list instead: execution emits
	// each GROUP BY column followed by the aggregate column(s).
	if hasAggregate(stmt, doc) && !describeSelectHasCollectionAggregate(doc, src, stmt) && !selectHasAggregateProjection(doc, stmt) {
		return aggregateColumns(doc, src, cat, stmt, byOID)
	}

	cols := make([]ColumnMeta, 0, stmt.ProjectionsCount)
	for i := int32(0); i < stmt.ProjectionsCount; i++ {
		proj := &doc.Projections[stmt.ProjectionsStart+i]
		if proj.Star {
			// SELECT * contributes no columns to the optimizer's projection
			// list. A bare SELECT * therefore reports the default id/score
			// shape (the executor's empty-projection fallback); a mixed
			// "SELECT *, col" reports only the explicit columns.
			continue
		}
		switch proj.Expr.Kind {
		case parser.NodeKindIdentifier:
			id := &doc.Identifiers[proj.Expr.ID]
			name := string(src[id.Start:id.End])
			if proj.AliasEnd > proj.Alias {
				name = string(src[proj.Alias:proj.AliasEnd])
			}
			oid := oidForGraphProjection(doc, src, id)
			if oid == 0 {
				oid = oidForColumn(cat, byOID, id, src)
			}
			cols = append(cols, ColumnMeta{Name: name, TypeOID: oid})
		case parser.NodeKindVectorFunc:
			vf := &doc.VectorFuncs[proj.Expr.ID]
			name := "vector_distance"
			if vf.IsMaxSim {
				name = "similarity"
			}
			if proj.AliasEnd > proj.Alias {
				name = string(src[proj.Alias:proj.AliasEnd])
			}
			cols = append(cols, ColumnMeta{Name: name, TypeOID: OIDFloat8})
		case parser.NodeKindBinaryExpr:
			name := "json_expression"
			if proj.AliasEnd > proj.Alias {
				name = string(src[proj.Alias:proj.AliasEnd])
			}
			oid := jsonBinaryExprOID(doc, src, cat, byOID, proj.Expr)
			if isVectorOperatorExpr(doc, proj.Expr) {
				if name == "json_expression" {
					name = "vector_distance"
				}
				oid = OIDFloat8
			}
			cols = append(cols, ColumnMeta{Name: name, TypeOID: oid})
		case parser.NodeKindFunctionExpr:
			if proj.Expr.ID < 0 || int(proj.Expr.ID) >= len(doc.FunctionExprs) {
				continue
			}
			fn := &doc.FunctionExprs[proj.Expr.ID]
			functionName := string(src[fn.NameStart:fn.NameEnd])
			name := functionName
			if proj.AliasEnd > proj.Alias {
				name = string(src[proj.Alias:proj.AliasEnd])
			}
			var oid uint32 = OIDText
			if strings.EqualFold(functionName, "NOW") {
				oid = OIDTimestamptz
			} else if strings.EqualFold(functionName, "NULLIF") && fn.ArgsCount > 0 && fn.ArgsStart >= 0 && fn.ArgsStart < int32(len(doc.FunctionArgs)) {
				oid = scalarExprOID(doc, src, cat, byOID, doc.FunctionArgs[fn.ArgsStart])
			} else if jsonOID := jsonFunctionOID(functionName); jsonOID != 0 {
				oid = jsonOID
			} else if strings.EqualFold(functionName, "array_agg") {
				oid = collectionAggregateOID(doc, src, cat, byOID, fn, false)
			} else if strings.EqualFold(functionName, "string_agg") {
				oid = OIDText
			} else if strings.EqualFold(functionName, "RRF") || strings.EqualFold(functionName, "FTS_RANK") || strings.EqualFold(functionName, "ts_rank") || strings.EqualFold(functionName, "ts_rank_cd") {
				oid = OIDFloat8
			}
			cols = append(cols, ColumnMeta{Name: name, TypeOID: oid})
		case parser.NodeKindCaseExpr, parser.NodeKindCastExpr:
			name := "case"
			if proj.Expr.Kind == parser.NodeKindCastExpr {
				name = "cast"
			}
			if proj.AliasEnd > proj.Alias {
				name = string(src[proj.Alias:proj.AliasEnd])
			}
			cols = append(cols, ColumnMeta{Name: name, TypeOID: scalarExprOID(doc, src, cat, byOID, proj.Expr)})
		case parser.NodeKindAggregateExpr:
			// Grouped aggregates retain every projected column. Preserve an
			// explicit SQL alias because execution uses it in SearchResults.Columns.
			ae := &doc.AggregateExprs[proj.Expr.ID]
			column := aggregateColumnMeta(doc, src, cat, byOID, ae)
			if proj.AliasEnd > proj.Alias {
				column.Name = string(src[proj.Alias:proj.AliasEnd])
			}
			cols = append(cols, column)
		default:
			// Literal / expression projections are dropped from the optimizer's
			// projection list, so they contribute nothing to the RowDescription.
		}
	}
	// No selectable projection (e.g. SELECT 1): the executor reports the
	// default id/score shape.
	if len(cols) == 0 {
		return defaultDescribeColumns()
	}
	return cols
}

func oidForGraphProjection(doc *parser.QueryDoc, src []byte, id *parser.Identifier) uint32 {
	if doc == nil || id == nil || id.ColumnOID != 0 || !describeHasGraphJoin(doc) {
		return 0
	}
	field := strings.ToLower(string(src[id.Start:id.End]))
	switch field {
	case "source_id", "target_id", "edge_type":
		return OIDText
	case "edge_weight":
		return OIDFloat4
	}
	if id.QualEnd <= id.QualStart {
		return 0
	}
	qualifier := string(src[id.QualStart:id.QualEnd])
	if !describeHasGraphEdgeAlias(doc, src, qualifier) {
		return 0
	}
	switch field {
	case "type", "kind", "edge_type":
		return OIDText
	case "weight", "edge_weight":
		return OIDFloat4
	default:
		return 0
	}
}

func describeHasGraphEdgeAlias(doc *parser.QueryDoc, src []byte, qualifier string) bool {
	if doc == nil || qualifier == "" {
		return false
	}
	for i := range doc.SelectStmts {
		for j := range doc.SelectStmts[i].Joins {
			match := doc.SelectStmts[i].Joins[j].MatchPath
			if match.Kind != parser.NodeKindMatchPath || match.ID < 0 || int(match.ID) >= len(doc.MatchPaths) {
				continue
			}
			path := doc.MatchPaths[match.ID]
			for n := int32(0); n < path.PathNodesCount; n++ {
				ref := doc.Nodes[path.PathNodesStart+n]
				if ref.Kind != parser.NodeKindEdge || ref.ID < 0 || int(ref.ID) >= len(doc.Edges) {
					continue
				}
				edge := doc.Edges[ref.ID]
				if edge.AliasEnd > edge.Alias && strings.EqualFold(qualifier, string(src[edge.Alias:edge.AliasEnd])) {
					return true
				}
			}
		}
	}
	return false
}

func describeHasGraphJoin(doc *parser.QueryDoc) bool {
	if doc == nil {
		return false
	}
	for i := range doc.SelectStmts {
		for j := range doc.SelectStmts[i].Joins {
			if doc.SelectStmts[i].Joins[j].MatchPath.Kind == parser.NodeKindMatchPath {
				return true
			}
		}
	}
	return false
}

func selectHasAggregateProjection(doc *parser.QueryDoc, stmt *parser.SelectStmt) bool {
	if doc == nil || stmt == nil {
		return false
	}
	for i := int32(0); i < stmt.ProjectionsCount; i++ {
		if doc.Projections[stmt.ProjectionsStart+i].Expr.Kind == parser.NodeKindAggregateExpr {
			return true
		}
	}
	return false
}

func scalarExprOID(doc *parser.QueryDoc, src []byte, cat *catalog.Catalog, byOID map[uint32]*catalog.TableDef, ref parser.NodeRef) uint32 {
	if doc == nil {
		return OIDText
	}
	switch ref.Kind {
	case parser.NodeKindCastExpr:
		if ref.ID < 0 || int(ref.ID) >= len(doc.CastExprs) {
			return OIDText
		}
		name := strings.ToLower(strings.TrimSpace(string(src[doc.CastExprs[ref.ID].TypeStart:doc.CastExprs[ref.ID].TypeEnd])))
		switch name {
		case "json":
			return OIDJSON
		case "jsonb":
			return OIDJSONB
		case "vector":
			// An explicit ::vector cast is a PostgreSQL extension type, not a
			// float4[] parameter.  Advertising the real vector OID lets clients
			// such as asyncpg use their text fallback codec for string literals;
			// vector operators without an explicit cast retain the established
			// float4[] inference below for Go []float32 callers.
			return OIDVector
		case "uuid":
			return OIDUUID
		case "bigint":
			return OIDInt8
		case "int", "int2", "int4", "integer", "smallint":
			return OIDInt4
		case "float", "float4", "float8", "real", "double", "double precision", "numeric", "decimal":
			if name == "float4" || name == "real" {
				return OIDFloat4
			}
			return OIDFloat8
		case "bool", "boolean":
			return OIDBool
		default:
			return OIDText
		}
	case parser.NodeKindCaseExpr:
		if ref.ID >= 0 && int(ref.ID) < len(doc.CaseExprs) {
			ce := doc.CaseExprs[ref.ID]
			for i := int32(0); i < ce.WhensCount; i++ {
				if oid := scalarExprOID(doc, src, cat, byOID, doc.CaseWhens[ce.WhensStart+i].Value); oid != OIDText {
					return oid
				}
			}
			if ce.HasElse {
				return scalarExprOID(doc, src, cat, byOID, ce.Else)
			}
		}
		return OIDText
	case parser.NodeKindBinaryExpr:
		if ref.ID >= 0 && int(ref.ID) < len(doc.BinaryExprs) {
			op := lexer.Kind(doc.BinaryExprs[ref.ID].Operator)
			switch op {
			case lexer.KindEquals, lexer.KindNotEqual, lexer.KindGreaterThan, lexer.KindLessThan, lexer.KindGreaterEqual, lexer.KindLessEqual, lexer.KindAnd, lexer.KindOr:
				return OIDBool
			case lexer.KindPlus, lexer.KindDash, lexer.KindAsterisk, lexer.KindSlash, lexer.KindPercent, lexer.KindShiftLeft, lexer.KindShiftRight:
				return OIDInt8
			case lexer.KindConcat:
				return OIDText
			}
		}
	case parser.NodeKindNumber:
		if ref.ID >= 0 && int(ref.ID) < len(doc.Numbers) {
			text := string(src[doc.Numbers[ref.ID].Start:doc.Numbers[ref.ID].End])
			if strings.ContainsAny(text, ".eE") {
				return OIDFloat8
			}
		}
		return OIDInt8
	case parser.NodeKindString:
		return OIDText
	case parser.NodeKindIdentifier:
		if ref.ID >= 0 && int(ref.ID) < len(doc.Identifiers) {
			id := &doc.Identifiers[ref.ID]
			return oidForColumn(cat, byOID, id, src)
		}
	}
	return OIDText
}

// hasAggregate reports whether the statement is routed to the aggregate
// executor, matching the optimizer's hasAggregate computation.
func hasAggregate(stmt *parser.SelectStmt, doc *parser.QueryDoc) bool {
	if len(stmt.GroupBy) > 0 || stmt.HavingExpr.Kind != parser.NodeKindUnknown {
		return true
	}
	for i := int32(0); i < stmt.ProjectionsCount; i++ {
		if doc.Projections[stmt.ProjectionsStart+i].Expr.Kind == parser.NodeKindAggregateExpr {
			return true
		}
	}
	return false
}

// describeSelectHasCollectionAggregate identifies the generic function forms
// handled by the virtual relation executor. They are ordinary FunctionExpr
// nodes rather than parser AggregateExpr nodes, so grouped queries must not be
// described as the single-column legacy aggregate shape.
func describeSelectHasCollectionAggregate(doc *parser.QueryDoc, src []byte, stmt *parser.SelectStmt) bool {
	for i := int32(0); i < stmt.ProjectionsCount; i++ {
		ref := doc.Projections[stmt.ProjectionsStart+i].Expr
		if ref.Kind != parser.NodeKindFunctionExpr || ref.ID < 0 || int(ref.ID) >= len(doc.FunctionExprs) {
			continue
		}
		fn := &doc.FunctionExprs[ref.ID]
		if fn.HasWindow || fn.NameStart >= fn.NameEnd || fn.NameEnd > uint32(len(src)) {
			continue
		}
		name := src[fn.NameStart:fn.NameEnd]
		if asciiEqualFold(name, "array_agg") || asciiEqualFold(name, "string_agg") {
			return true
		}
	}
	return false
}

func asciiEqualFold(a []byte, b string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		x, y := a[i], b[i]
		if x >= 'A' && x <= 'Z' {
			x += 'a' - 'A'
		}
		if y >= 'A' && y <= 'Z' {
			y += 'a' - 'A'
		}
		if x != y {
			return false
		}
	}
	return true
}

// aggregateColumns returns the single column executeAggregate reports for a
// query: the aggregate function name and a type derived from the function and,
// for MIN/MAX, the underlying column's catalog type.
func aggregateColumns(doc *parser.QueryDoc, src []byte, cat *catalog.Catalog, stmt *parser.SelectStmt, byOID map[uint32]*catalog.TableDef) []ColumnMeta {
	var aggFunc parser.AggregateFunc = parser.AggCount
	var aggRef parser.NodeRef
	for i := int32(0); i < stmt.ProjectionsCount; i++ {
		proj := &doc.Projections[stmt.ProjectionsStart+i]
		if proj.Expr.Kind == parser.NodeKindAggregateExpr {
			ae := &doc.AggregateExprs[proj.Expr.ID]
			aggFunc = ae.Func
			aggRef = ae.Expr
			if ae.OrderedSet {
				aggRef = ae.OrderExpr
			}
			break
		}
	}
	name := aggregateName(aggFunc)
	var oid uint32 = OIDInt8
	switch aggFunc {
	case parser.AggCount:
		oid = OIDInt8
	case parser.AggSum, parser.AggAvg:
		oid = OIDFloat8
	case parser.AggVectorAvg:
		oid = OIDFloat4Array
	case parser.AggMin, parser.AggMax:
		if aggRef.Kind == parser.NodeKindIdentifier {
			id := &doc.Identifiers[aggRef.ID]
			if t := columnType(cat, byOID, id, src); t != 0 {
				oid = catalogTypeToOID(t)
			} else {
				oid = OIDText
			}
		} else {
			oid = OIDText
		}
	case parser.AggPercentileCont:
		oid = OIDFloat8
	case parser.AggPercentileDisc, parser.AggMode:
		if aggRef.Kind == parser.NodeKindIdentifier {
			id := &doc.Identifiers[aggRef.ID]
			if t := columnType(cat, byOID, id, src); t != 0 {
				oid = catalogTypeToOID(t)
			} else {
				oid = OIDText
			}
		} else {
			oid = OIDText
		}
	}
	return []ColumnMeta{{Name: name, TypeOID: oid}}
}

// aggregateColumnMeta builds the ColumnMeta for one aggregate projection.
// Used by the defensive branch in describeSelect and by aggregateColumns.
func aggregateColumnMeta(doc *parser.QueryDoc, src []byte, cat *catalog.Catalog, byOID map[uint32]*catalog.TableDef, ae *parser.AggregateExpr) ColumnMeta {
	name := aggregateName(ae.Func)
	var oid uint32 = OIDInt8
	switch ae.Func {
	case parser.AggCount:
		oid = OIDInt8
	case parser.AggSum, parser.AggAvg:
		oid = OIDFloat8
	case parser.AggVectorAvg:
		oid = OIDFloat4Array
	case parser.AggMin, parser.AggMax:
		if ae.Expr.Kind == parser.NodeKindIdentifier {
			id := &doc.Identifiers[ae.Expr.ID]
			if t := columnType(cat, byOID, id, src); t != 0 {
				oid = catalogTypeToOID(t)
			} else {
				oid = OIDText
			}
		} else {
			oid = OIDText
		}
	case parser.AggPercentileCont:
		oid = OIDFloat8
	case parser.AggPercentileDisc, parser.AggMode:
		if ae.OrderExpr.Kind == parser.NodeKindIdentifier {
			id := &doc.Identifiers[ae.OrderExpr.ID]
			if t := columnType(cat, byOID, id, src); t != 0 {
				oid = catalogTypeToOID(t)
			} else {
				oid = OIDText
			}
		} else {
			oid = OIDText
		}
	}
	return ColumnMeta{Name: name, TypeOID: oid}
}

// aggregateName mirrors aggregateColumnName in the executor.
func aggregateName(f parser.AggregateFunc) string {
	switch f {
	case parser.AggCount:
		return "count"
	case parser.AggSum:
		return "sum"
	case parser.AggAvg:
		return "avg"
	case parser.AggMin:
		return "min"
	case parser.AggMax:
		return "max"
	case parser.AggPercentileCont:
		return "percentile_cont"
	case parser.AggPercentileDisc:
		return "percentile_disc"
	case parser.AggMode:
		return "mode"
	case parser.AggVectorAvg:
		return "vector_avg"
	default:
		return "count"
	}
}

func collectionAggregateOID(doc *parser.QueryDoc, src []byte, cat *catalog.Catalog, byOID map[uint32]*catalog.TableDef, fn *parser.FunctionExpr, lenient bool) uint32 {
	if fn == nil || fn.ArgsCount == 0 || fn.ArgsStart < 0 || int(fn.ArgsStart) >= len(doc.FunctionArgs) {
		return OIDTextArray
	}
	arg := doc.FunctionArgs[fn.ArgsStart]
	if lenient || cat == nil || arg.Kind != parser.NodeKindIdentifier || arg.ID < 0 || int(arg.ID) >= len(doc.Identifiers) {
		return OIDTextArray
	}
	id := &doc.Identifiers[arg.ID]
	switch columnType(cat, byOID, id, src) {
	case catalog.TypeInt:
		return OIDInt4Array
	case catalog.TypeBigInt:
		return OIDInt8Array
	case catalog.TypeFloat4:
		return OIDFloat4Array
	case catalog.TypeFloat:
		return OIDFloat8Array
	case catalog.TypeBool:
		return OIDBoolArray
	default:
		return OIDTextArray
	}
}

// describeProjectionsLenient describes the outer projection list of a Leiden
// CTE query without catalog binding. The CTE name is a virtual relation, so
// column types cannot be resolved from the catalog; names still match what
// execution produces and types fall back to name-based inference.
func describeProjectionsLenient(db *libravdb.Database, doc *parser.QueryDoc, src []byte, stmt *parser.SelectStmt) []ColumnMeta {
	// Keep grouped projections intact. The aggregate-only shortcut would
	// advertise just COUNT/SUM/etc. even when execution also emits the GROUP BY
	// columns (especially for HAVING subqueries routed through the virtual
	// evaluator).
	if hasAggregate(stmt, doc) && len(stmt.GroupBy) == 0 && !describeSelectHasWindow(doc, stmt) {
		var aggFunc parser.AggregateFunc = parser.AggCount
		for i := int32(0); i < stmt.ProjectionsCount; i++ {
			proj := &doc.Projections[stmt.ProjectionsStart+i]
			if proj.Expr.Kind == parser.NodeKindAggregateExpr {
				aggFunc = doc.AggregateExprs[proj.Expr.ID].Func
				break
			}
		}
		oid := uint32(OIDText)
		if aggFunc == parser.AggVectorAvg {
			oid = OIDFloat4Array
		}
		return []ColumnMeta{{Name: aggregateName(aggFunc), TypeOID: oid}}
	}
	cols := make([]ColumnMeta, 0, stmt.ProjectionsCount)
	for i := int32(0); i < stmt.ProjectionsCount; i++ {
		proj := &doc.Projections[stmt.ProjectionsStart+i]
		if proj.Star {
			// Window queries execute through the virtual relation path, where star
			// expands to the source metadata names before window columns. Mirror
			// that deterministic sorted expansion for extended-protocol Describe.
			if db != nil && stmt.FromTable.Kind == parser.NodeKindTableExpr && stmt.FromTable.ID >= 0 && int(stmt.FromTable.ID) < len(doc.TableExprs) {
				table := doc.TableExprs[stmt.FromTable.ID]
				if col, err := db.GetCollection(string(src[table.Start:table.End])); err == nil {
					names := make([]string, 0, len(col.Config().MetadataSchema)+1)
					names = append(names, "id")
					for name := range col.Config().MetadataSchema {
						names = append(names, name)
					}
					if table.TemporalRange {
						names = append(names, "version", "ordinal", "begin_lsn", "end_lsn", "version_start", "version_end")
					}
					sort.Strings(names)
					for _, name := range names {
						oid := windowStarColumnOID(col, name)
						if table.TemporalRange {
							oid = temporalRangeColumnOID(name)
						}
						cols = append(cols, ColumnMeta{Name: name, TypeOID: oid})
					}
				}
			}
			continue
		}
		var name string
		if proj.AliasEnd > proj.Alias {
			name = string(src[proj.Alias:proj.AliasEnd])
		}
		switch proj.Expr.Kind {
		case parser.NodeKindIdentifier:
			id := &doc.Identifiers[proj.Expr.ID]
			if name == "" {
				name = string(src[id.Start:id.End])
			}
			oid := oidForName(name)
			if describeSelectHasTemporalRange(doc, stmt) {
				oid = temporalRangeColumnOID(name)
			}
			cols = append(cols, ColumnMeta{Name: name, TypeOID: oid})
		case parser.NodeKindVectorFunc:
			vf := &doc.VectorFuncs[proj.Expr.ID]
			if name == "" {
				if vf.IsMaxSim {
					name = "similarity"
				} else {
					name = "vector_distance"
				}
			}
			cols = append(cols, ColumnMeta{Name: name, TypeOID: OIDFloat8})
		case parser.NodeKindBinaryExpr:
			if name == "" {
				name = "json_expression"
			}
			oid := jsonBinaryExprOID(doc, src, nil, nil, proj.Expr)
			if isVectorOperatorExpr(doc, proj.Expr) {
				if name == "json_expression" {
					name = "vector_distance"
				}
				oid = OIDFloat8
			}
			cols = append(cols, ColumnMeta{Name: name, TypeOID: oid})
		case parser.NodeKindSubqueryExpr:
			if name == "" {
				name = "scalar_subquery"
			}
			cols = append(cols, ColumnMeta{Name: name, TypeOID: OIDText})
		case parser.NodeKindAggregateExpr:
			if proj.Expr.ID < 0 || int(proj.Expr.ID) >= len(doc.AggregateExprs) {
				continue
			}
			ae := &doc.AggregateExprs[proj.Expr.ID]
			if name == "" {
				name = aggregateName(ae.Func)
			}
			var oid uint32 = OIDFloat8
			if ae.Func == parser.AggCount {
				oid = OIDInt8
			} else if ae.Func == parser.AggVectorAvg {
				oid = OIDFloat4Array
			}
			cols = append(cols, ColumnMeta{Name: name, TypeOID: oid})
		case parser.NodeKindFunctionExpr:
			if proj.Expr.ID < 0 || int(proj.Expr.ID) >= len(doc.FunctionExprs) {
				continue
			}
			fn := &doc.FunctionExprs[proj.Expr.ID]
			functionName := string(src[fn.NameStart:fn.NameEnd])
			if name == "" {
				name = functionName
			}
			var oid uint32 = OIDText
			if jsonOID := jsonFunctionOID(functionName); jsonOID != 0 {
				oid = jsonOID
			} else if strings.EqualFold(functionName, "array_agg") {
				oid = collectionAggregateOID(doc, src, nil, nil, fn, true)
			} else if strings.EqualFold(functionName, "string_agg") {
				oid = OIDText
			} else if strings.EqualFold(functionName, "RRF") || strings.EqualFold(functionName, "FTS_RANK") || strings.EqualFold(functionName, "ts_rank") || strings.EqualFold(functionName, "ts_rank_cd") {
				oid = OIDFloat8
			}
			if fn.HasWindow {
				lower := strings.ToLower(functionName)
				switch lower {
				case "row_number", "rank", "dense_rank", "ntile":
					oid = OIDInt8
				case "percent_rank", "cume_dist":
					oid = OIDFloat8
				case "lag", "lead":
					if fn.ArgsCount > 0 {
						oid = exprOID(doc, src, nil, nil, doc.FunctionArgs[fn.ArgsStart])
						if oid == 0 {
							oid = OIDText
						}
					}
				}
			}
			cols = append(cols, ColumnMeta{Name: name, TypeOID: oid})
		default:
			// Literal projections are dropped, matching the optimizer.
		}
	}
	if len(cols) == 0 {
		return defaultDescribeColumns()
	}
	return cols
}

func describeSelectHasDerivedRelation(doc *parser.QueryDoc, stmt *parser.SelectStmt) bool {
	if stmt == nil {
		return false
	}
	if stmt.FromTable.Kind == parser.NodeKindTableExpr && stmt.FromTable.ID >= 0 && int(stmt.FromTable.ID) < len(doc.TableExprs) && doc.TableExprs[stmt.FromTable.ID].IsDerived {
		return true
	}
	for i := range stmt.Joins {
		if stmt.Joins[i].Derived.Kind == parser.NodeKindTableExpr {
			return true
		}
	}
	return false
}

func describeSelectHasTemporalRange(doc *parser.QueryDoc, stmt *parser.SelectStmt) bool {
	if doc == nil || stmt == nil || stmt.FromTable.Kind != parser.NodeKindTableExpr || stmt.FromTable.ID < 0 || int(stmt.FromTable.ID) >= len(doc.TableExprs) {
		return false
	}
	return doc.TableExprs[stmt.FromTable.ID].TemporalRange
}

func temporalRangeColumnOID(name string) uint32 {
	switch strings.ToLower(name) {
	case "version", "begin_lsn", "end_lsn", "ordinal":
		return OIDInt8
	case "version_start", "version_end":
		return OIDTimestamptz
	default:
		return oidForName(name)
	}
}

func describeSelectHasWindow(doc *parser.QueryDoc, stmt *parser.SelectStmt) bool {
	if stmt == nil {
		return false
	}
	for i := int32(0); i < stmt.ProjectionsCount; i++ {
		projection := doc.Projections[stmt.ProjectionsStart+i]
		if projection.Expr.Kind == parser.NodeKindFunctionExpr && projection.Expr.ID >= 0 && int(projection.Expr.ID) < len(doc.FunctionExprs) && doc.FunctionExprs[projection.Expr.ID].HasWindow {
			return true
		}
		if projection.Expr.Kind == parser.NodeKindAggregateExpr && projection.Expr.ID >= 0 && int(projection.Expr.ID) < len(doc.AggregateExprs) && doc.AggregateExprs[projection.Expr.ID].HasWindow {
			return true
		}
	}
	return false
}

// buildScope builds the map of table OID → TableDef for a SELECT's FROM and
// JOIN tables, mirroring the binder's scope construction so identifier
// projections resolve their column types.
func buildScope(cat *catalog.Catalog, src []byte, stmt *parser.SelectStmt, doc *parser.QueryDoc) map[uint32]*catalog.TableDef {
	byOID := make(map[uint32]*catalog.TableDef)
	add := func(name string) {
		if sysDef, ok := catalog.ResolveSystemTable(name); ok {
			byOID[sysDef.OID] = sysDef
			return
		}
		if t, err := cat.GetTable(catalog.HashIdentifier(name)); err == nil {
			byOID[t.OID] = t
		}
	}
	if stmt.FromTable.Kind == parser.NodeKindTableExpr {
		te := &doc.TableExprs[stmt.FromTable.ID]
		add(string(src[te.Start:te.End]))
	} else if stmt.FromTable.Kind == parser.NodeKindGraphTable {
		gt := &doc.GraphTables[stmt.FromTable.ID]
		add(string(src[gt.TableStart:gt.TableEnd]))
	}
	for i := range stmt.Joins {
		jc := &stmt.Joins[i]
		if jc.MatchPath.Kind == parser.NodeKindMatchPath {
			continue // graph join — no catalog table
		}
		if jc.TableEnd > jc.TableStart {
			add(string(src[jc.TableStart:jc.TableEnd]))
		}
	}
	return byOID
}

// oidForColumn resolves an identifier projection's PostgreSQL type OID from
// the catalog, falling back to text when the column cannot be resolved.
func oidForColumn(cat *catalog.Catalog, byOID map[uint32]*catalog.TableDef, id *parser.Identifier, src []byte) uint32 {
	if t := columnType(cat, byOID, id, src); t != 0 {
		return catalogTypeToOID(t)
	}
	return OIDText
}

// columnType returns the catalog type code for a bound column identifier, or
// 0 when it is not a resolvable scalar column.
func columnType(cat *catalog.Catalog, byOID map[uint32]*catalog.TableDef, id *parser.Identifier, src []byte) uint16 {
	if id.ResolvedKind != parser.ResolvedKindColumn {
		return 0
	}
	colHash := catalog.HashIdentifier(string(src[id.Start:id.End]))
	if catalog.IsSystemTableOID(id.TableOID) {
		col, err := catalog.ResolveSystemColumn(id.TableOID, colHash)
		if err != nil {
			return 0
		}
		return col.Type
	}
	t, ok := byOID[id.TableOID]
	if !ok {
		return 0
	}
	col, err := cat.GetColumn(t, colHash)
	if err != nil {
		return 0
	}
	return col.Type
}

// jsonBinaryExprOID describes the scalar type produced by PostgreSQL's JSON
// operators. Extraction with -> returns a JSON value, ->> returns text, and
// containment is boolean. The catalog lookup is intentionally used only for
// extraction's left operand so JSON (rather than JSONB) columns retain their
// declared wire type.
func jsonBinaryExprOID(doc *parser.QueryDoc, src []byte, cat *catalog.Catalog, byOID map[uint32]*catalog.TableDef, ref parser.NodeRef) uint32 {
	if ref.Kind != parser.NodeKindBinaryExpr || ref.ID < 0 || int(ref.ID) >= len(doc.BinaryExprs) {
		return OIDText
	}
	be := &doc.BinaryExprs[ref.ID]
	switch lexer.Kind(be.Operator) {
	case lexer.KindJSONContains, lexer.KindJSONContainedBy, lexer.KindJSONExists, lexer.KindJSONAny, lexer.KindJSONAll, lexer.KindJSONPathExists:
		return OIDBool
	case lexer.KindJSONDelete:
		return OIDJSONB
	case lexer.KindFTSMatch:
		if be.Right.Kind == parser.NodeKindString && be.Right.ID >= 0 && int(be.Right.ID) < len(doc.Strings) {
			lit := doc.Strings[be.Right.ID]
			if lit.Start < uint32(len(src)) && lit.End <= uint32(len(src)) && lit.End > lit.Start+1 && isJSONPathLiteralText(string(src[lit.Start+1:lit.End-1])) {
				return OIDBool
			}
		}
		return OIDText
	case lexer.KindJSONExtractText, lexer.KindJSONPathText:
		return OIDText
	case lexer.KindArrowRight, lexer.KindJSONExtract, lexer.KindJSONPath:
		if be.Left.Kind == parser.NodeKindIdentifier && be.Left.ID >= 0 && int(be.Left.ID) < len(doc.Identifiers) && cat != nil && byOID != nil {
			id := &doc.Identifiers[be.Left.ID]
			if typ := columnType(cat, byOID, id, src); typ == catalog.TypeJSON {
				return OIDJSON
			}
		}
		return OIDJSONB
	default:
		return OIDText
	}
}

func isJSONPathLiteralText(text string) bool {
	text = strings.TrimSpace(text)
	if strings.HasPrefix(strings.ToLower(text), "strict ") || strings.HasPrefix(strings.ToLower(text), "lax ") {
		text = strings.TrimSpace(text[4:])
	}
	return strings.HasPrefix(text, "$")
}

func jsonFunctionOID(name string) uint32 {
	switch strings.ToLower(name) {
	case "json_set", "jsonb_set", "json_insert", "jsonb_insert",
		"json_build_array", "jsonb_build_array", "json_build_object", "jsonb_build_object",
		"json_populate_record", "jsonb_populate_record", "to_json", "to_jsonb":
		return OIDJSONB
	case "json_typeof", "jsonb_typeof":
		return OIDText
	case "jsonb_array_length":
		return OIDInt8
	default:
		return 0
	}
}

// defaultDescribeColumns is the shape the executor reports when a statement
// has no explicit projection list: id (text) + score (float8).
func defaultDescribeColumns() []ColumnMeta {
	return []ColumnMeta{{Name: "id", TypeOID: OIDText}, {Name: "score", TypeOID: OIDFloat8}}
}

// leidenColumns is the fixed result shape of a standalone COMPUTE LEIDEN.
func leidenColumns() []ColumnMeta {
	return []ColumnMeta{
		{Name: "node_id", TypeOID: OIDInt8},
		{Name: "community_id", TypeOID: OIDInt8},
		{Name: "collection", TypeOID: OIDText},
		{Name: "record_id", TypeOID: OIDText},
		{Name: "truncated", TypeOID: OIDBool},
		{Name: "scope", TypeOID: OIDText},
		{Name: "modularity", TypeOID: OIDFloat8},
	}
}

// oidForName maps a column name to its type OID when no catalog is available,
// mirroring the static name defaults in columnOIDFor.
func oidForName(name string) uint32 {
	switch name {
	case "id", "ID":
		return OIDText
	case "score", "SCORE", "version", "VERSION":
		return OIDFloat8
	case "ordinal", "ORDINAL":
		return OIDInt8
	case "node_id", "community_id":
		return OIDInt8
	case "collection", "record_id", "scope":
		return OIDText
	case "truncated":
		return OIDBool
	case "modularity":
		return OIDFloat8
	default:
		return OIDText
	}
}

// windowStarColumnOID mirrors the executor's metadata schema for a lenient
// Describe of SELECT * in a window query. Name-only inference is insufficient
// here: a BIGINT metadata field must be described as int8, not the float8
// fallback used by oidForName for generic score-like names.
func windowStarColumnOID(col *libravdb.Collection, name string) uint32 {
	if col == nil {
		return oidForName(name)
	}
	if strings.EqualFold(name, "id") {
		return OIDText
	}
	if strings.EqualFold(name, "embedding") && col.Dimension() > 0 {
		return OIDFloat4Array
	}
	cfg := col.Config()
	for schemaName, fieldType := range cfg.MetadataSchema {
		if !strings.EqualFold(schemaName, name) {
			continue
		}
		switch fieldType {
		case libravdb.IntField:
			return OIDInt4
		case libravdb.BigIntField:
			return OIDInt8
		case libravdb.FloatField:
			return OIDFloat8
		case libravdb.BoolField:
			return OIDBool
		case libravdb.TimeField:
			return OIDTimestamp
		case libravdb.StringArrayField:
			return OIDTextArray
		case libravdb.IntArrayField:
			return OIDInt4Array
		case libravdb.FloatArrayField:
			return OIDFloat8Array
		case libravdb.JSONField:
			return OIDJSON
		case libravdb.JSONBField:
			return OIDJSONB
		default:
			return OIDText
		}
	}
	return oidForName(name)
}

// inferParamOIDs derives each parameter's type OID from its usage context:
// comparison with a column, IN/BETWEEN lists, or a vector-function operand.
// Parameters whose type cannot be inferred stay 0 (unspecified), which
// PostgreSQL clients treat as "send as text".
func inferParamOIDs(doc *parser.QueryDoc, src []byte, cat *catalog.Catalog, byOID map[uint32]*catalog.TableDef, paramCount int) []uint32 {
	oids := make([]uint32, paramCount)
	if paramCount == 0 {
		return oids
	}

	// Map each parameter identifier to its bound ordinal. Positional
	// parameters occupy [0, max($N)); unique named parameters follow them.
	// This is the same canonical ordering used by buildQueryParams.
	ordinal := make(map[int32]int, paramCount)
	info := analyzeParamsBytes(src)
	namedOrdinals := make(map[string]int, len(info.namedOrder))
	for i, name := range info.namedOrder {
		namedOrdinals[strings.ToLower(name)] = info.numPositional + i
	}
	for i := range doc.Identifiers {
		id := &doc.Identifiers[i]
		if int(id.Start) >= len(src) || int(id.End) > len(src) {
			continue
		}
		c := src[id.Start]
		if c != '$' && c != '@' {
			continue
		}
		text := string(src[id.Start:id.End])
		if c == '$' {
			if n, err := strconv.Atoi(text[1:]); err == nil && n >= 1 {
				ordinal[id.ID] = n - 1
			} else if n, ok := namedOrdinals[strings.ToLower(text[1:])]; ok {
				ordinal[id.ID] = n
			}
		} else {
			if n, ok := namedOrdinals[strings.ToLower(text[1:])]; ok {
				ordinal[id.ID] = n
			}
		}
	}

	setOID := func(idID int32, oid uint32) {
		if oid == 0 {
			return
		}
		idx, ok := ordinal[idID]
		if !ok || idx < 0 || idx >= len(oids) {
			return
		}
		if oids[idx] == 0 {
			oids[idx] = oid
		}
	}

	// INSERT parameters inherit the declared type of their target columns even
	// when they are not part of a comparison expression. This is important for
	// strict clients such as asyncpg: an unspecified parameter OID makes the
	// client issue a recursive pg_type introspection query before every bind.
	// Use the live catalog schema so this follows the same table/column typing
	// used by execution and pgwire RowDescription.
	if cat != nil {
		for i := range doc.InsertStmts {
			stmt := &doc.InsertStmts[i]
			if stmt.TableEnd <= stmt.TableStart || stmt.TableEnd > uint32(len(src)) {
				continue
			}
			table, err := cat.GetTable(catalog.HashIdentifier(string(src[stmt.TableStart:stmt.TableEnd])))
			if err != nil {
				continue
			}
			allColumns := cat.AllColumns(table)
			if len(allColumns) == 0 || len(stmt.Values) == 0 {
				continue
			}
			columnTypes := make([]uint32, len(allColumns))
			for j := range allColumns {
				columnTypes[j] = catalog.ColumnTypeToPGOID(allColumns[j].Type)
			}
			if len(stmt.Columns) > 0 {
				columnTypes = make([]uint32, len(stmt.Columns))
				for j, ref := range stmt.Columns {
					if ref.Kind != parser.NodeKindIdentifier || ref.ID < 0 || int(ref.ID) >= len(doc.Identifiers) {
						continue
					}
					id := &doc.Identifiers[ref.ID]
					if col, err := cat.GetColumn(table, catalog.HashIdentifier(string(src[id.Start:id.End]))); err == nil {
						columnTypes[j] = catalog.ColumnTypeToPGOID(col.Type)
					} else if (strings.EqualFold(string(src[id.Start:id.End]), "embedding") ||
						strings.EqualFold(string(src[id.Start:id.End]), "vector") ||
						strings.EqualFold(string(src[id.Start:id.End]), "vec")) &&
						catHasPrimaryVector(cat) {
						// The physical vector is stored in the collection's existing
						// vector index rather than the metadata-column block. Treat
						// the established embedding/vector aliases as its native
						// float-array bind type for strict clients such as asyncpg.
						columnTypes[j] = OIDFloat4Array
					}
				}
			}
			for j, ref := range stmt.Values {
				if ref.Kind != parser.NodeKindIdentifier || ref.ID < 0 || int(ref.ID) >= len(doc.Identifiers) {
					continue
				}
				id := &doc.Identifiers[ref.ID]
				if id.Start >= uint32(len(src)) || id.End > uint32(len(src)) {
					continue
				}
				if src[id.Start] != '$' && src[id.Start] != '@' {
					continue
				}
				if castOID := explicitParameterCastOID(src, id.End); castOID != 0 {
					setOID(id.ID, castOID)
					continue
				}
				if len(columnTypes) > 0 {
					setOID(id.ID, columnTypes[j%len(columnTypes)])
				}
			}
		}
	}
	setTemporalBoundOID := func(start, end uint32) {
		if start >= end || end > uint32(len(src)) {
			return
		}
		for i := range doc.Identifiers {
			id := &doc.Identifiers[i]
			if id.Start == start && id.End == end {
				setOID(id.ID, OIDTimestamptz)
			}
		}
		text := string(src[start:end])
		if len(text) < 2 || (text[0] != '$' && text[0] != '@') {
			return
		}
		idx := -1
		if text[0] == '$' {
			if n, err := strconv.Atoi(text[1:]); err == nil && n > 0 {
				idx = n - 1
			} else {
				if n, ok := namedOrdinals[strings.ToLower(text[1:])]; ok {
					idx = n
				}
			}
		} else {
			if n, ok := namedOrdinals[strings.ToLower(text[1:])]; ok {
				idx = n
			}
		}
		if idx >= 0 && idx < len(oids) && oids[idx] == 0 {
			oids[idx] = OIDTimestamptz
		}
	}
	setLSNBoundOID := func(start, end uint32) {
		if start >= end || end > uint32(len(src)) {
			return
		}
		for i := range doc.Identifiers {
			id := &doc.Identifiers[i]
			if id.Start == start && id.End == end {
				setOID(id.ID, OIDInt8)
			}
		}
		text := string(src[start:end])
		if len(text) < 2 || (text[0] != '$' && text[0] != '@') {
			return
		}
		idx := -1
		if text[0] == '$' {
			if n, err := strconv.Atoi(text[1:]); err == nil && n > 0 {
				idx = n - 1
			} else if n, ok := namedOrdinals[strings.ToLower(text[1:])]; ok {
				idx = n
			}
		} else if n, ok := namedOrdinals[strings.ToLower(text[1:])]; ok {
			idx = n
		}
		if idx >= 0 && idx < len(oids) && oids[idx] == 0 {
			oids[idx] = OIDInt8
		}
	}
	for i := range doc.TableExprs {
		table := &doc.TableExprs[i]
		if table.TemporalRange {
			setTemporalBoundOID(table.RangeStartStart, table.RangeStartEnd)
			setTemporalBoundOID(table.RangeEndStart, table.RangeEndEnd)
		}
		if table.TemporalLSN {
			setLSNBoundOID(table.LSNStart, table.LSNEnd)
		}
	}
	var isParam func(parser.NodeRef) (int32, bool)
	isParam = func(ref parser.NodeRef) (int32, bool) {
		if ref.Kind == parser.NodeKindCastExpr {
			if ref.ID < 0 || int(ref.ID) >= len(doc.CastExprs) {
				return 0, false
			}
			return isParam(doc.CastExprs[ref.ID].Expr)
		}
		if ref.Kind != parser.NodeKindIdentifier || ref.ID < 0 || int(ref.ID) >= len(doc.Identifiers) {
			return 0, false
		}
		id := &doc.Identifiers[ref.ID]
		if id.Start >= uint32(len(src)) {
			return 0, false
		}
		c := src[id.Start]
		if c == '$' || c == '@' {
			return id.ID, true
		}
		return 0, false
	}
	paramContextOID := func(ref parser.NodeRef, fallback uint32) uint32 {
		if ref.Kind == parser.NodeKindCastExpr {
			if ref.ID >= 0 && int(ref.ID) < len(doc.CastExprs) {
				if castOID := scalarExprOID(doc, src, cat, byOID, ref); castOID != 0 {
					return castOID
				}
				ref = doc.CastExprs[ref.ID].Expr
			}
		}
		if ref.Kind != parser.NodeKindIdentifier || ref.ID < 0 || int(ref.ID) >= len(doc.Identifiers) {
			return fallback
		}
		id := &doc.Identifiers[ref.ID]
		if id.End <= uint32(len(src)) {
			if castOID := explicitParameterCastOID(src, id.End); castOID != 0 {
				return castOID
			}
		}
		return fallback
	}

	// Edge-local graph predicates compare the physical edge weight, which is
	// a float32 value. Walk AND-composed property blocks before the generic
	// comparison pass; the latter cannot resolve r.weight as a catalog column
	// and would otherwise conservatively mark the parameter as text.
	var inferEdgePredicateParams func(parser.NodeRef)
	inferEdgePredicateParams = func(ref parser.NodeRef) {
		if ref.Kind != parser.NodeKindBinaryExpr || ref.ID < 0 || int(ref.ID) >= len(doc.BinaryExprs) {
			return
		}
		be := &doc.BinaryExprs[ref.ID]
		if be.Operator == uint8(lexer.KindAnd) || be.Operator == uint8(lexer.KindOr) {
			inferEdgePredicateParams(be.Left)
			inferEdgePredicateParams(be.Right)
			return
		}
		if be.Left.Kind != parser.NodeKindIdentifier || be.Left.ID < 0 || int(be.Left.ID) >= len(doc.Identifiers) {
			return
		}
		left := &doc.Identifiers[be.Left.ID]
		if left.Start >= uint32(len(src)) || left.End > uint32(len(src)) {
			return
		}
		propertyName := string(src[left.Start:left.End])
		if pid, ok := isParam(be.Right); ok {
			if strings.EqualFold(propertyName, "type") || strings.EqualFold(propertyName, "kind") {
				setOID(pid, OIDText)
			} else {
				// Arbitrary JSON edge properties are dynamically typed at
				// storage time. Numeric comparisons use float8 on the wire,
				// matching the native numeric predicate representation.
				setOID(pid, OIDFloat8)
			}
		}
	}
	for i := range doc.Edges {
		edge := &doc.Edges[i]
		inferEdgePredicateParams(edge.Predicate)
	}

	// Comparison with a column (or literal) infers the parameter's type.
	for i := range doc.BinaryExprs {
		be := &doc.BinaryExprs[i]
		if isVectorOperatorExpr(doc, parser.NodeRef{Kind: parser.NodeKindBinaryExpr, ID: int32(i)}) {
			if pid, ok := isParam(be.Right); ok {
				setOID(pid, paramContextOID(be.Right, OIDFloat4Array))
			}
			if pid, ok := isParam(be.Left); ok {
				setOID(pid, paramContextOID(be.Left, OIDFloat4Array))
			}
			continue
		}
		if be.Operator == uint8(lexer.KindJSONContains) || be.Operator == uint8(lexer.KindJSONContainedBy) || be.Operator == uint8(lexer.KindJSONExists) || be.Operator == uint8(lexer.KindJSONAny) || be.Operator == uint8(lexer.KindJSONAll) || be.Operator == uint8(lexer.KindJSONPathExists) {
			if be.Operator == uint8(lexer.KindJSONExists) || be.Operator == uint8(lexer.KindJSONAny) || be.Operator == uint8(lexer.KindJSONAll) || be.Operator == uint8(lexer.KindJSONPathExists) {
				if pid, ok := isParam(be.Right); ok {
					setOID(pid, OIDText)
				}
				continue
			}
			if pid, ok := isParam(be.Left); ok {
				setOID(pid, OIDJSONB)
			}
			if pid, ok := isParam(be.Right); ok {
				setOID(pid, OIDJSONB)
			}
			continue
		}
		if be.Operator == uint8(lexer.KindArrowRight) || be.Operator == uint8(lexer.KindJSONExtract) || be.Operator == uint8(lexer.KindJSONExtractText) || be.Operator == uint8(lexer.KindJSONPath) || be.Operator == uint8(lexer.KindJSONPathText) {
			if pid, ok := isParam(be.Left); ok {
				setOID(pid, OIDJSONB)
			}
			if pid, ok := isParam(be.Right); ok {
				setOID(pid, OIDText)
			}
			continue
		}
		if pid, ok := isParam(be.Left); ok {
			setOID(pid, exprOID(doc, src, cat, byOID, be.Right))
		} else if pid, ok := isParam(be.Right); ok {
			setOID(pid, exprOID(doc, src, cat, byOID, be.Left))
		}
	}

	// IN (…) and BETWEEN … AND … inherit the compared expression's type.
	for i := range doc.InExprs {
		in := &doc.InExprs[i]
		t := exprOID(doc, src, cat, byOID, in.Expr)
		if t == 0 {
			continue
		}
		for j := int32(0); j < in.ListCount; j++ {
			if pid, ok := isParam(doc.Nodes[in.ListStart+j]); ok {
				setOID(pid, t)
			}
		}
	}
	for i := range doc.BetweenExprs {
		bt := &doc.BetweenExprs[i]
		t := exprOID(doc, src, cat, byOID, bt.Expr)
		if t == 0 {
			continue
		}
		if pid, ok := isParam(bt.Lower); ok {
			setOID(pid, t)
		}
		if pid, ok := isParam(bt.Upper); ok {
			setOID(pid, t)
		}
	}

	// Vector-function operands use the one-dimensional float-array wire type.
	// Text input such as '[1,0,0]' is decoded into []float32 at Bind time.
	for i := range doc.VectorFuncs {
		vf := &doc.VectorFuncs[i]
		if pid, ok := isParam(vf.VectorA); ok {
			setOID(pid, paramContextOID(vf.VectorA, OIDFloat4Array))
		}
		if pid, ok := isParam(vf.VectorB); ok {
			setOID(pid, paramContextOID(vf.VectorB, OIDFloat4Array))
		}
	}
	// FTS_RANK's query operand is textual. RRF itself is a score-producing
	// wrapper, so its nested vector/text operands are inferred by these
	// component walks rather than by treating the wrapper as a generic text
	// function.
	for i := range doc.FunctionExprs {
		fn := &doc.FunctionExprs[i]
		if fn.ArgsCount == 0 || fn.NameStart >= uint32(len(src)) || fn.NameEnd > uint32(len(src)) {
			continue
		}
		name := string(src[fn.NameStart:fn.NameEnd])
		if !strings.EqualFold(name, "FTS_RANK") && !strings.EqualFold(name, "to_tsvector") && !strings.EqualFold(name, "to_tsquery") && !strings.EqualFold(name, "plainto_tsquery") && !strings.EqualFold(name, "phraseto_tsquery") && !strings.EqualFold(name, "websearch_to_tsquery") && !strings.EqualFold(name, "ts_rank") && !strings.EqualFold(name, "ts_rank_cd") {
			continue
		}
		if fn.ArgsStart < 0 || fn.ArgsStart+fn.ArgsCount > int32(len(doc.FunctionArgs)) {
			continue
		}
		// Text/query arguments are the final argument for constructors. Rank
		// functions optionally take a normalization integer after the query,
		// so their query is always argument two.
		queryIndex := fn.ArgsCount - 1
		if (strings.EqualFold(name, "ts_rank") || strings.EqualFold(name, "ts_rank_cd")) && fn.ArgsCount >= 2 {
			queryIndex = 1
		}
		if pid, ok := isParam(doc.FunctionArgs[fn.ArgsStart+queryIndex]); ok {
			setOID(pid, OIDText)
		}
	}

	return oids
}

// explicitParameterCastOID recognizes the compact PostgreSQL cast spelling
// immediately following a parameter. The parser intentionally keeps INSERT
// value parameters as identifier nodes, so the cast is still represented in
// the source span rather than as a CastExpr node in that AST position.
func explicitParameterCastOID(src []byte, end uint32) uint32 {
	i := int(end)
	for i < len(src) {
		switch src[i] {
		case ' ', '\t', '\r', '\n':
			i++
		default:
			goto castStart
		}
	}
	return 0

castStart:
	if i+2 > len(src) || src[i] != ':' || src[i+1] != ':' {
		return 0
	}
	i += 2
	start := i
	for i < len(src) {
		c := src[i]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' {
			i++
			continue
		}
		break
	}
	if start == i {
		return 0
	}
	switch strings.ToLower(string(src[start:i])) {
	case "vector":
		return OIDVector
	case "json":
		return OIDJSON
	case "jsonb":
		return OIDJSONB
	case "uuid":
		return OIDUUID
	case "bigint", "int8":
		return OIDInt8
	case "int", "int4", "integer":
		return OIDInt4
	case "smallint", "int2":
		return OIDInt2
	case "real", "float4":
		return OIDFloat4
	case "double", "float8":
		return OIDFloat8
	case "bool", "boolean":
		return OIDBool
	case "text", "varchar":
		return OIDText
	default:
		return 0
	}
}

func catHasPrimaryVector(cat *catalog.Catalog) bool {
	if cat == nil {
		return false
	}
	_, err := cat.GetVectorIndex(catalog.HashIdentifier("vector"))
	return err == nil
}

// exprOID returns the type OID an expression evaluates to, or 0 when the type
// is not inferable. Used both for projection typing and parameter inference.
func exprOID(doc *parser.QueryDoc, src []byte, cat *catalog.Catalog, byOID map[uint32]*catalog.TableDef, ref parser.NodeRef) uint32 {
	switch ref.Kind {
	case parser.NodeKindIdentifier:
		id := &doc.Identifiers[ref.ID]
		c := src[id.Start]
		if c == '$' || c == '@' {
			return 0 // parameter compared to another parameter: ambiguous
		}
		if id.ResolvedKind != parser.ResolvedKindColumn {
			return 0 // vector/graph operands carry no scalar SQL type
		}
		if t := columnType(cat, byOID, id, src); t != 0 {
			return catalogTypeToOID(t)
		}
		return OIDText
	case parser.NodeKindVectorFunc:
		return OIDFloat8
	case parser.NodeKindAggregateExpr:
		return OIDInt8
	case parser.NodeKindNumber:
		num := &doc.Numbers[ref.ID]
		if strings.ContainsAny(string(src[num.Start:num.End]), ".eE") {
			return OIDFloat8
		}
		return OIDInt8
	case parser.NodeKindString:
		return OIDText
	case parser.NodeKindBinaryExpr:
		if isVectorOperatorExpr(doc, ref) {
			return OIDFloat8
		}
		return 0
	default:
		return 0
	}
}

func isVectorOperatorExpr(doc *parser.QueryDoc, ref parser.NodeRef) bool {
	if doc == nil || ref.Kind != parser.NodeKindBinaryExpr || ref.ID < 0 || int(ref.ID) >= len(doc.BinaryExprs) {
		return false
	}
	switch lexer.Kind(doc.BinaryExprs[ref.ID].Operator) {
	case lexer.KindL2Dist, lexer.KindIPDist, lexer.KindCosineDist:
		return true
	default:
		return false
	}
}
