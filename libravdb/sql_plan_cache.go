package libravdb

import (
	"strings"
	"sync"

	"github.com/xDarkicex/lexer"
	"github.com/xDarkicex/lexer/parser"
	"github.com/xDarkicex/libravdb/internal/catalog"
	"github.com/xDarkicex/libravdb/internal/optimizer"
)

// sqlPlanCache is deliberately small and bounded. Catalogs are immutable
// snapshots, so a plan is valid only for the catalog generation and catalog
// object against which it was bound.
type sqlPlanCache struct {
	mu         sync.RWMutex
	maxEntries int
	entries    map[string]sqlPlanCacheEntry
}

type sqlPlanCacheEntry struct {
	catalog        *catalog.Catalog
	generation     uint64
	plan           *optimizer.PhysicalPlan
	parameterSlots []sqlPlanParameterSlot
}

type sqlPlanParameterSlot struct {
	predicateIndex int
	start          uint32
	end            uint32
}

func newSQLPlanCache(maxEntries int) *sqlPlanCache {
	if maxEntries < 1 {
		maxEntries = 1
	}
	return &sqlPlanCache{
		maxEntries: maxEntries,
		entries:    make(map[string]sqlPlanCacheEntry, maxEntries),
	}
}

func (c *sqlPlanCache) get(key string, generation uint64, cat *catalog.Catalog, params *optimizer.ParameterSet, src []byte) (*optimizer.PhysicalPlan, bool) {
	if c == nil {
		return nil, false
	}
	c.mu.RLock()
	entry, ok := c.entries[key]
	c.mu.RUnlock()
	if !ok || entry.generation != generation || entry.catalog != cat || entry.plan == nil {
		if ok {
			c.mu.Lock()
			current, stillPresent := c.entries[key]
			if stillPresent && (current.generation != generation || current.catalog != cat) {
				delete(c.entries, key)
			}
			c.mu.Unlock()
		}
		return nil, false
	}
	clone := cloneCachedSQLPlan(entry.plan)
	for _, slot := range entry.parameterSlots {
		if params == nil {
			return nil, false
		}
		value, found := params.Lookup(src, slot.start, slot.end)
		if !found || slot.predicateIndex < 0 || slot.predicateIndex >= len(clone.Predicates) {
			return nil, false
		}
		predicate := &clone.Predicates[slot.predicateIndex]
		predicate.TypedValue = value
		predicate.ValueIsNull = value.IsNull()
		predicate.Value = value.Bytes()
	}
	return clone, true
}

func (c *sqlPlanCache) put(key string, generation uint64, cat *catalog.Catalog, plan *optimizer.PhysicalPlan, parameterSlots []sqlPlanParameterSlot) {
	if c == nil || plan == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, exists := c.entries[key]; !exists && len(c.entries) >= c.maxEntries {
		// The cache is an optimization, not a correctness boundary. Evict one
		// arbitrary old entry instead of adding LRU bookkeeping to the query hot
		// path; generation checks still provide precise schema invalidation.
		for oldKey := range c.entries {
			delete(c.entries, oldKey)
			break
		}
	}
	c.entries[key] = sqlPlanCacheEntry{
		catalog:        cat,
		generation:     generation,
		parameterSlots: append([]sqlPlanParameterSlot(nil), parameterSlots...),
		// The caller executes its freshly optimized plan immediately. Keep a
		// separate copy in the cache so execution-time fields cannot race with
		// a concurrent cache hit.
		plan: cloneCachedSQLPlan(plan),
	}
}

func (c *sqlPlanCache) reset() {
	if c == nil {
		return
	}
	c.mu.Lock()
	c.entries = make(map[string]sqlPlanCacheEntry, c.maxEntries)
	c.mu.Unlock()
}

// cloneCachedSQLPlan copies the mutable plan slices used by ordinary
// relational execution. The eligibility check below excludes graph, vector,
// aggregate, DML, temporal, and virtual-expression plans, keeping this clone
// deliberately narrow and auditable.
func cloneCachedSQLPlan(plan *optimizer.PhysicalPlan) *optimizer.PhysicalPlan {
	clone := *plan
	clone.QueryVector = append([]float32(nil), plan.QueryVector...)
	clone.Predicates = append([]optimizer.RelationalPredicate(nil), plan.Predicates...)
	clone.PredicateAlternatives = make(optimizer.PredicateAlternatives, len(plan.PredicateAlternatives))
	for i := range plan.PredicateAlternatives {
		clone.PredicateAlternatives[i] = append([]optimizer.RelationalPredicate(nil), plan.PredicateAlternatives[i]...)
	}
	clone.Projections = append([]string(nil), plan.Projections...)
	clone.ProjectionRefs = append([]optimizer.ProjectionRef(nil), plan.ProjectionRefs...)
	return &clone
}

func normalizeSQLPlanKey(sql string) string {
	trimmed := strings.TrimSpace(sql)
	for strings.HasSuffix(trimmed, ";") {
		trimmed = strings.TrimSpace(strings.TrimSuffix(trimmed, ";"))
	}
	return trimmed
}

// sqlPlanCacheEligible is intentionally conservative. A physical plan stores
// resolved values and execution state; only the scalar predicate parameter
// slots recognized below are rebound on a cache hit. Other parameterized or
// virtual plans continue through the normal optimizer path.
func sqlPlanCacheEligible(src []byte, doc *parser.QueryDoc) (bool, []sqlPlanParameterSlot) {
	if doc == nil || len(doc.SelectStmts) != 1 || len(doc.InsertStmts) != 0 ||
		len(doc.UpdateStmts) != 0 || len(doc.DeleteStmts) != 0 ||
		len(doc.CreateTableStmts) != 0 || len(doc.CreateEdgeTypeStmts) != 0 ||
		len(doc.DropTableStmts) != 0 || len(doc.CreateIndexStmts) != 0 ||
		len(doc.DropIndexStmts) != 0 || len(doc.AlterTableStmts) != 0 ||
		len(doc.InsertGraphEdgeStmts) != 0 || len(doc.SubqueryExprs) != 0 {
		return false, nil
	}
	stmt := &doc.SelectStmts[0]
	if stmt.FromTable.Kind != parser.NodeKindTableExpr || len(stmt.Joins) != 0 ||
		stmt.CTEsCount != 0 || len(stmt.GroupBy) != 0 ||
		stmt.HavingExpr.Kind != parser.NodeKindUnknown || stmt.UnionNext.Kind != parser.NodeKindUnknown ||
		stmt.SetOp != parser.SetOpNone || selectHasTemporalSnapshot(doc, stmt) {
		return false, nil
	}
	if stmt.LimitExpr.Kind != parser.NodeKindUnknown || stmt.OffsetExpr.Kind != parser.NodeKindUnknown {
		return false, nil
	}
	// These routes intentionally execute against virtual rows or post-process
	// expressions and therefore do not produce a reusable ordinary plan.
	if virtualSelectHasJSON(src, doc, stmt) || virtualSelectHasWindow(doc, stmt) ||
		virtualSelectHasCollectionAggregate(src, doc, stmt) ||
		virtualSelectHasOrderedSetAggregate(doc, stmt) ||
		virtualSelectHasParameterizedAggregate(src, doc, stmt) ||
		virtualSelectHasNestedAggregateProjection(doc, stmt) ||
		virtualSelectHasScalarExpressions(src, doc, stmt) || selectHasDerivedRelation(doc, stmt) {
		return false, nil
	}
	for i := int32(0); i < stmt.ProjectionsCount; i++ {
		projection := doc.Projections[int(stmt.ProjectionsStart+i)]
		if projection.Star {
			continue
		}
		if projection.Expr.Kind != parser.NodeKindIdentifier {
			return false, nil
		}
	}
	parameterSlots, ok := sqlPlanParameterSlots(src, doc, stmt)
	if !ok {
		return false, nil
	}
	return true, parameterSlots
}

// sqlPlanParameterSlots recognizes the scalar predicate shape for which a
// physical plan can be rebound without rerunning the full optimizer. It is
// intentionally limited to AND-connected binary predicates with an ordinary
// column on the left and a scalar literal or parameter on the right.
func sqlPlanParameterSlots(src []byte, doc *parser.QueryDoc, stmt *parser.SelectStmt) ([]sqlPlanParameterSlot, bool) {
	if stmt.WhereExpr.Kind == parser.NodeKindUnknown {
		return nil, true
	}
	slots := make([]sqlPlanParameterSlot, 0, 2)
	predicateIndex := 0
	var walk func(parser.NodeRef) bool
	walk = func(node parser.NodeRef) bool {
		switch node.Kind {
		case parser.NodeKindBinaryExpr:
			if node.ID < 0 || int(node.ID) >= len(doc.BinaryExprs) {
				return false
			}
			be := doc.BinaryExprs[node.ID]
			if be.Operator == uint8(lexer.KindAnd) {
				return walk(be.Left) && walk(be.Right)
			}
			if be.Left.Kind != parser.NodeKindIdentifier {
				return false
			}
			switch be.Right.Kind {
			case parser.NodeKindNumber, parser.NodeKindString:
				// Literal scalar; no rebind slot is needed.
			case parser.NodeKindIdentifier:
				if be.Right.ID < 0 || int(be.Right.ID) >= len(doc.Identifiers) {
					return false
				}
				id := doc.Identifiers[be.Right.ID]
				if id.Start >= uint32(len(src)) || id.End > uint32(len(src)) {
					return false
				}
				if src[id.Start] == '$' || src[id.Start] == '@' {
					slots = append(slots, sqlPlanParameterSlot{predicateIndex: predicateIndex, start: id.Start, end: id.End})
				} else if id.ResolvedKind != parser.ResolvedKindLiteral {
					// A column-to-column predicate needs a different physical
					// representation and is not this cache's scalar shape.
					return false
				}
			default:
				return false
			}
			predicateIndex++
			return true
		default:
			return false
		}
	}
	if !walk(stmt.WhereExpr) {
		return nil, false
	}
	return slots, true
}
