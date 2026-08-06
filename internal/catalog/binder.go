package catalog

import (
	"fmt"

	"github.com/xDarkicex/lexer/parser"
)

// Binder performs the OID resolution pass over the SoA QueryDoc.
type Binder struct {
	catalog *Catalog
	src     []byte // The original SQL source byte slice
}

func NewBinder(cat *Catalog, src []byte) *Binder {
	return &Binder{
		catalog: cat,
		src:     src,
	}
}

// Bind modifies the doc in-place, filling in TableOID and ColumnOID for Identifiers.
// It returns an error if any identifier fails to resolve against the catalog.
func (b *Binder) Bind(doc *parser.QueryDoc) error {
	var scope []*TableDef
	// aliasScope maps qualifier name → TableDef for qualified identifiers
	// (s.owner_id). Both the raw table name and any FROM alias resolve here.
	aliasScope := make(map[uint64]*TableDef)

	// 1. Resolve tables in FROM clauses and build scope stack.
	for i := 0; i < len(doc.TableExprs); i++ {
		t := &doc.TableExprs[i]
		hash := hashIdentifier(b.src, t.Start, t.End)
		def, err := b.catalog.GetTable(hash)
		if err != nil {
			// System table fallback: pg_class, etc. are not in the catalog binary
			// but are resolved via the hardcoded system table registry.
			name := string(b.src[t.Start:t.End])
			if sysDef, ok := ResolveSystemTable(name); ok {
				t.TableOID = sysDef.OID
				scope = append(scope, sysDef)
				aliasScope[hash] = sysDef
				if t.AliasEnd > t.Alias {
					aliasScope[hashIdentifier(b.src, t.Alias, t.AliasEnd)] = sysDef
				}
				continue
			}
			return fmt.Errorf("table '%s' not found", name)
		}
		t.TableOID = def.OID
		scope = append(scope, def)
		aliasScope[hash] = def
		if t.AliasEnd > t.Alias {
			aliasScope[hashIdentifier(b.src, t.Alias, t.AliasEnd)] = def
		}
	}

	for i := 0; i < len(doc.GraphTables); i++ {
		gt := &doc.GraphTables[i]
		hash := hashIdentifier(b.src, gt.TableStart, gt.TableEnd)
		def, err := b.catalog.GetTable(hash)
		if err != nil {
			name := string(b.src[gt.TableStart:gt.TableEnd])
			if sysDef, ok := ResolveSystemTable(name); ok {
				gt.TableOID = sysDef.OID
				scope = append(scope, sysDef)
				aliasScope[hash] = sysDef
				continue
			}
			return fmt.Errorf("graph table '%s' not found", name)
		}
		gt.TableOID = def.OID
		scope = append(scope, def)
		aliasScope[hash] = def
	}

	// 1a. Relational JOIN tables are part of the same SELECT scope.  Keeping
	// them out of scope made qualified references such as t.sla_status fail to
	// bind even though the parser had retained the JOIN alias.
	for i := 0; i < len(doc.SelectStmts); i++ {
		stmt := &doc.SelectStmts[i]
		for j := range stmt.Joins {
			jc := &stmt.Joins[j]
			if jc.MatchPath.Kind == parser.NodeKindMatchPath || jc.TableEnd <= jc.TableStart {
				continue
			}
			hash := hashIdentifier(b.src, jc.TableStart, jc.TableEnd)
			def, err := b.catalog.GetTable(hash)
			if err != nil {
				return fmt.Errorf("join table '%s' not found", string(b.src[jc.TableStart:jc.TableEnd]))
			}
			scope = append(scope, def)
			aliasScope[hash] = def
			if jc.AliasEnd > jc.Alias {
				aliasScope[hashIdentifier(b.src, jc.Alias, jc.AliasEnd)] = def
			}
		}
	}

	// 1b. Bind JOIN MATCH graph joins: the anchor vertex (first vertex in the
	// match path) must be the FROM alias, e.g. FROM services s JOIN MATCH
	// (s)-[:DEPENDS_ON*1..3]->(api). Vertex aliases beyond the anchor are graph
	// node aliases and do not resolve against the catalog.
	for i := 0; i < len(doc.SelectStmts); i++ {
		stmt := &doc.SelectStmts[i]
		for j := range stmt.Joins {
			jc := &stmt.Joins[j]
			if jc.MatchPath.Kind != parser.NodeKindMatchPath {
				continue
			}
			mp := &doc.MatchPaths[jc.MatchPath.ID]
			if mp.PathNodesCount == 0 {
				continue
			}
			firstRef := doc.Nodes[mp.PathNodesStart]
			if firstRef.Kind != parser.NodeKindVertex {
				continue
			}
			v := &doc.Vertexes[firstRef.ID]
			if v.AliasEnd <= v.Alias {
				return fmt.Errorf("JOIN MATCH anchor vertex must have an alias matching a FROM alias")
			}
			ah := hashIdentifier(b.src, v.Alias, v.AliasEnd)
			anchorDef, ok := aliasScope[ah]
			if !ok {
				return fmt.Errorf("JOIN MATCH anchor vertex '%s' does not match any FROM alias", string(b.src[v.Alias:v.AliasEnd]))
			}
			// Graph vertex aliases name records in the graph relation. Registering
			// them in the SQL scope lets projections such as doc.title and vector
			// expressions such as doc.embedding bind to that relation. A future
			// multi-relation graph catalog can replace this mapping per vertex.
			for n := int32(0); n < mp.PathNodesCount; n++ {
				ref := doc.Nodes[mp.PathNodesStart+n]
				if ref.Kind != parser.NodeKindVertex {
					continue
				}
				vertex := &doc.Vertexes[ref.ID]
				if vertex.AliasEnd > vertex.Alias {
					aliasScope[hashIdentifier(b.src, vertex.Alias, vertex.AliasEnd)] = anchorDef
				}
			}
		}
	}

	// WHERE MATCH uses the same vertex-alias scope as JOIN MATCH. Without
	// this pass, predicates such as p.category in
	// WHERE MATCH (c)-[:PURCHASED]->(p:Product) AND p.category = 'Security'
	// either fail binding or (worse) become detached scalar predicates.
	for i := 0; i < len(doc.SelectStmts); i++ {
		stmt := &doc.SelectStmts[i]
		var bindWhereMatch func(parser.NodeRef) error
		bindWhereMatch = func(ref parser.NodeRef) error {
			switch ref.Kind {
			case parser.NodeKindBinaryExpr:
				be := &doc.BinaryExprs[ref.ID]
				if err := bindWhereMatch(be.Left); err != nil {
					return err
				}
				return bindWhereMatch(be.Right)
			case parser.NodeKindMatchPath:
				mp := &doc.MatchPaths[ref.ID]
				if mp.PathNodesCount == 0 {
					return nil
				}
				first := doc.Nodes[mp.PathNodesStart]
				if first.Kind != parser.NodeKindVertex {
					return fmt.Errorf("WHERE MATCH must start with a vertex alias")
				}
				anchor := &doc.Vertexes[first.ID]
				if anchor.AliasEnd <= anchor.Alias {
					return fmt.Errorf("WHERE MATCH anchor vertex must have an alias matching a FROM alias")
				}
				anchorDef, ok := aliasScope[hashIdentifier(b.src, anchor.Alias, anchor.AliasEnd)]
				if !ok {
					return fmt.Errorf("WHERE MATCH anchor vertex '%s' does not match any FROM alias", string(b.src[anchor.Alias:anchor.AliasEnd]))
				}
				for n := int32(0); n < mp.PathNodesCount; n++ {
					vertexRef := doc.Nodes[mp.PathNodesStart+n]
					if vertexRef.Kind != parser.NodeKindVertex {
						continue
					}
					vertex := &doc.Vertexes[vertexRef.ID]
					if vertex.AliasEnd > vertex.Alias {
						aliasScope[hashIdentifier(b.src, vertex.Alias, vertex.AliasEnd)] = anchorDef
					}
				}
			}
			return nil
		}
		if err := bindWhereMatch(stmt.WhereExpr); err != nil {
			return err
		}
	}

	// 1b. Resolve CRUD statement tables.
	for i := 0; i < len(doc.InsertStmts); i++ {
		stmt := &doc.InsertStmts[i]
		hash := hashIdentifier(b.src, stmt.TableStart, stmt.TableEnd)
		def, err := b.catalog.GetTable(hash)
		if err != nil {
			name := string(b.src[stmt.TableStart:stmt.TableEnd])
			if sysDef, ok := ResolveSystemTable(name); ok {
				// Allow system tables to bind (executor will reject unsupported ops)
				scope = append(scope, sysDef)
				continue
			}
			return fmt.Errorf("table '%s' not found", name)
		}
		scope = append(scope, def)
	}
	for i := 0; i < len(doc.UpdateStmts); i++ {
		stmt := &doc.UpdateStmts[i]
		hash := hashIdentifier(b.src, stmt.TableStart, stmt.TableEnd)
		def, err := b.catalog.GetTable(hash)
		if err != nil {
			name := string(b.src[stmt.TableStart:stmt.TableEnd])
			if sysDef, ok := ResolveSystemTable(name); ok {
				scope = append(scope, sysDef)
				continue
			}
			return fmt.Errorf("table '%s' not found", name)
		}
		scope = append(scope, def)
	}
	for i := 0; i < len(doc.DeleteStmts); i++ {
		stmt := &doc.DeleteStmts[i]
		hash := hashIdentifier(b.src, stmt.TableStart, stmt.TableEnd)
		def, err := b.catalog.GetTable(hash)
		if err != nil {
			name := string(b.src[stmt.TableStart:stmt.TableEnd])
			if sysDef, ok := ResolveSystemTable(name); ok {
				scope = append(scope, sysDef)
				continue
			}
			return fmt.Errorf("table '%s' not found", name)
		}
		scope = append(scope, def)
	}

	// 2. Resolve identifiers (columns, vectors, graphs) deterministically using scope.
	// First collect SELECT projection aliases so ORDER BY can reference them.
	// Postgres semantics: ORDER BY may use a select-list alias as a column name.
	aliasSet := make(map[uint64]struct{})
	for i := 0; i < len(doc.SelectStmts); i++ {
		stmt := &doc.SelectStmts[i]
		for j := int32(0); j < stmt.ProjectionsCount; j++ {
			proj := &doc.Projections[stmt.ProjectionsStart+j]
			if proj.AliasEnd > proj.Alias {
				aliasSet[hashIdentifier(b.src, proj.Alias, proj.AliasEnd)] = struct{}{}
			}
		}
	}
	for i := 0; i < len(doc.Identifiers); i++ {
		id := &doc.Identifiers[i]
		hash := hashIdentifier(b.src, id.Start, id.End)

		// ORDER BY alias: resolve against the SELECT list before catalog lookup.
		if _, isAlias := aliasSet[hash]; isAlias {
			id.ResolvedKind = parser.ResolvedKindColumn
			continue
		}

		resolved := false

		// Qualified identifier: s.owner_id resolves only against the qualifier's
		// table (FROM alias or table name). No cross-table fallback.
		if id.QualStart != 0 {
			qhash := hashIdentifier(b.src, id.QualStart, id.QualEnd)
			tDef, ok := aliasScope[qhash]
			if !ok {
				return fmt.Errorf("unknown qualifier '%s'", string(b.src[id.QualStart:id.QualEnd]))
			}
			var col *ColumnDef
			var colErr error
			if IsSystemTableOID(tDef.OID) {
				col, colErr = ResolveSystemColumn(tDef.OID, hash)
			} else {
				col, colErr = b.catalog.GetColumn(tDef, hash)
			}
			if colErr != nil {
				colName := string(b.src[id.Start:id.End])
				if colName == "embedding" || colName == "vector" {
					id.TableOID = tDef.OID
					id.ResolvedKind = parser.ResolvedKindVector
					resolved = true
				} else {
					return fmt.Errorf("column '%s' not found in table '%s'", colName, string(b.src[id.QualStart:id.QualEnd]))
				}
			} else {
				id.TableOID = tDef.OID
				id.ColumnOID = col.OID
				id.ResolvedKind = parser.ResolvedKindColumn
				resolved = true
			}
		} else {
			// Unqualified: check scope tables for columns deterministically.
			// System tables use a hardcoded column registry instead of the catalog binary.
			for _, tDef := range scope {
				var col *ColumnDef
				var colErr error
				if IsSystemTableOID(tDef.OID) {
					col, colErr = ResolveSystemColumn(tDef.OID, hash)
				} else {
					col, colErr = b.catalog.GetColumn(tDef, hash)
				}
				if colErr == nil {
					id.TableOID = tDef.OID
					id.ColumnOID = col.OID
					id.ResolvedKind = parser.ResolvedKindColumn
					resolved = true
					break
				}
			}
		}

		if !resolved {
			// Check if it's a Vector Index
			vec, err := b.catalog.GetVectorIndex(hash)
			if err == nil {
				id.TableOID = vec.OID
				id.ResolvedKind = parser.ResolvedKindVector
				resolved = true
				continue
			}

			// Check if it's a Graph Label
			graph, err := b.catalog.GetGraphLabel(hash)
			if err == nil {
				id.TableOID = graph.OID
				id.ResolvedKind = parser.ResolvedKindGraph
				resolved = true
				continue
			}

			// Unqualified embedding/vector resolve to the collection primary vector.
			if !resolved {
				colName := string(b.src[id.Start:id.End])
				// FROM alias (e.g. doc in GRAPH_CENTRALITY(doc))
				if _, ok := aliasScope[hash]; ok {
					for _, tDef := range scope {
						if tDef.OID <= 99 {
							continue
						}
						id.TableOID = tDef.OID
						id.ResolvedKind = parser.ResolvedKindGraph
						resolved = true
						break
					}
				} else if colName == "embedding" || colName == "vector" {
					for _, tDef := range scope {
						id.TableOID = tDef.OID
						id.ResolvedKind = parser.ResolvedKindVector
						resolved = true
						break
					}
				}
			}
			if resolved {
				continue
			}
			// $param references are resolved at optimization time.
			if b.src[id.Start] == '$' {
				id.ResolvedKind = parser.ResolvedKindColumn // treated as expression operand
				resolved = true
				continue
			}
			return fmt.Errorf("identifier '%s' not found in scope or catalog", string(b.src[id.Start:id.End]))
		}
	}

	return nil
}

// hashIdentifier computes a case-insensitive FNV-1a hash directly from the source slice.
// This ensures 0 allocations on the bind path.
func hashIdentifier(src []byte, start, end uint32) uint64 {
	var hash uint64 = 14695981039346656037
	for i := start; i < end; i++ {
		c := src[i]
		// Convert ASCII uppercase to lowercase
		if c >= 'A' && c <= 'Z' {
			c += 32
		}
		hash ^= uint64(c)
		hash *= 1099511628211
	}
	return hash
}
