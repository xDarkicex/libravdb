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
				continue
			}
			return fmt.Errorf("table '%s' not found", name)
		}
		t.TableOID = def.OID
		scope = append(scope, def)
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
				continue
			}
			return fmt.Errorf("graph table '%s' not found", name)
		}
		gt.TableOID = def.OID
		scope = append(scope, def)
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

		// Check scope tables for columns deterministically.
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
