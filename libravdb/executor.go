package libravdb

import (
	"context"
	"errors"
	"fmt"
	"strings"

	btree "github.com/xDarkicex/libravdb/internal/index/btree"
	"github.com/xDarkicex/libravdb/internal/optimizer"
)

// Executor dispatches physical plans to concrete execution paths.
type Executor struct {
	db *Database
}

func newExecutor(db *Database) *Executor {
	return &Executor{db: db}
}

// Execute routes a physical plan to the appropriate execution engine.
func (e *Executor) Execute(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	switch plan.Kind {
	case optimizer.QueryKindKNN:
		return e.executeKNN(ctx, plan)
	case optimizer.QueryKindGraph:
		return e.executeGraph(ctx, plan)
	case optimizer.QueryKindRelational:
		return e.executeRelational(ctx, plan)
	case optimizer.QueryKindInsert:
		return e.executeInsert(ctx, plan)
	case optimizer.QueryKindUpdate:
		return e.executeUpdate(ctx, plan)
	case optimizer.QueryKindDelete:
		return e.executeDelete(ctx, plan)
	case optimizer.QueryKindJoin:
		return e.executeJoin(ctx, plan)
	case optimizer.QueryKindAggregate:
		return e.executeAggregate(ctx, plan)
	case optimizer.QueryKindDDL:
		return e.executeDDL(ctx, plan)
	default:
		// MaxSim and other future kinds fall through here
		return nil, fmt.Errorf("unknown query kind %d", plan.Kind)
	}
}

// executeKNN is the zero-change fast path for vector similarity search.
// It preserves the existing QueryBuilder fluent API path byte-for-byte.
func (e *Executor) executeKNN(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	col, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, fmt.Errorf("could not get collection %q: %w", plan.CollectionName, err)
	}

	if col.Dimension() == 0 {
		return nil, fmt.Errorf("collection %q is metadata-only; vector search not available", plan.CollectionName)
	}

	qb := col.Query(ctx)

	if plan.HasVectorSearch {
		qb.WithVector(plan.QueryVector)
		if plan.Similarity > 0 {
			qb.WithThreshold(plan.Similarity)
		}
	}

	if plan.Limit >= 0 {
		qb.Limit(plan.Limit)
	}

	results, err := qb.Execute()
	if err != nil {
		return nil, err
	}

	// Hybrid: apply relational predicates as post-filter on vector results
	if plan.HasRelationalQuery && len(plan.Predicates) > 0 && len(results.Results) > 0 {
		results = filterByPredicates(results, plan.Predicates)
	}

	return results, nil
}

// filterByPredicates applies relational predicates as a post-filter on search results.
func filterByPredicates(results *SearchResults, predicates []optimizer.RelationalPredicate) *SearchResults {
	filtered := make([]*SearchResult, 0, len(results.Results))
	for _, r := range results.Results {
		match := true
		for _, pred := range predicates {
			if !predicateMatches(r, pred) {
				match = false
				break
			}
		}
		if match {
			filtered = append(filtered, r)
		}
	}
	results.Results = filtered
	results.Total = len(filtered)
	return results
}

func predicateMatches(r *SearchResult, pred optimizer.RelationalPredicate) bool {
	switch pred.Operator {
	case 12: // KindEquals
		return r.ID == string(pred.Value)
	case 13: // KindGreaterThan
		return r.ID > string(pred.Value)
	case 14: // KindLessThan
		return r.ID < string(pred.Value)
	}
	return true // unknown operator → include
}

// executeGraph performs direction-aware graph traversal using BFSPattern.
// Seeds are selected by a three-way priority cascade:
//  1. Explicit seed (WHERE a.id = N) — validated via ResolveNodeID
//  2. Vector-anchored (WHERE SIMILARITY(...) + GRAPH_TABLE) — using SearchWithGraphFilter
//  3. Label-scan — NOT YET SUPPORTED (returns error)
func (e *Executor) executeGraph(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	var seeds []uint64

	// Priority 1: explicit seed (WHERE a.id = N)
	if plan.HasExplicitSeed {
		_, _, err := e.db.ResolveNodeID(ctx, plan.ExplicitSeedID)
		if err != nil {
			return nil, fmt.Errorf("explicit graph seed %d: %w", plan.ExplicitSeedID, err)
		}
		seeds = append(seeds, plan.ExplicitSeedID)
	}

	// Priority 2: vector-anchored traversal
	if len(seeds) == 0 && plan.HasVectorAnchor {
		col, err := e.db.GetCollection(plan.CollectionName)
		if err != nil {
			return nil, err
		}
		results, err := col.SearchWithGraphFilter(ctx, plan.GraphAnchorVector, plan.Limit, nil)
		if err != nil {
			return nil, fmt.Errorf("vector-anchored seed search: %w", err)
		}
		for _, r := range results.Results {
			nodeID, err := e.db.GetNodeID(ctx, plan.CollectionName, r.ID)
			if err != nil {
				continue
			}
			seeds = append(seeds, nodeID)
		}
	}

	// Priority 3: label-scan — not yet supported
	if len(seeds) == 0 {
		return nil, errors.New(
			"graph query requires either WHERE a.id = N (explicit seed) " +
				"or a vector predicate (vector-anchored traversal); " +
				"label-scan seeding not supported yet")
	}

	col, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
	}
	// Guardrail: metadata-only collections can't use vector-anchored traversal
	if plan.HasVectorAnchor && col.Dimension() == 0 {
		return nil, fmt.Errorf("collection %q is metadata-only; vector-anchored traversal not available — use WHERE a.id = N to anchor graph traversal", plan.CollectionName)
	}
	g := col.GetGraph()
	if g == nil {
		return nil, fmt.Errorf("collection %q has no graph", plan.CollectionName)
	}

	// Acquire pooled off-heap buffers
	bitset, err := g.GetBitset()
	if err != nil {
		return nil, err
	}
	defer g.PutBitset(bitset)

	frontier, err := g.GetFrontierBuf()
	if err != nil {
		return nil, err
	}
	defer g.PutFrontierBuf(frontier)

	// Convert optimizer.GraphEdgePlan to graph.EdgePlan
	edges := make([]EdgePlan, len(plan.GraphEdges))
	totalMinDepth := 0
	for i, gep := range plan.GraphEdges {
		max := int(gep.QuantMax)
		if max == 0 {
			if gep.QuantMin == 0 {
				max = 1 // default: exactly 1 hop
			} else {
				max = 1 << 20 // ->+ unbounded
			}
		}
		edges[i] = EdgePlan{Dir: gep.Direction, Min: int(gep.QuantMin), Max: max}
		totalMinDepth += int(gep.QuantMin)
	}

	// BFS from each seed, tracking min depth per node
	seen := make(map[uint64]int) // nodeID → min depth reached
	firstEdgeHasZeroMin := len(plan.GraphEdges) > 0 && plan.GraphEdges[0].QuantMin == 0

	for _, seed := range seeds {
		if firstEdgeHasZeroMin {
			if _, exists := seen[seed]; !exists {
				seen[seed] = 0
			}
		}

		if plan.Limit > 0 && len(seen) >= plan.Limit {
			break
		}

		if err := g.BFSPattern(seed, edges, plan.MaxHops, func(nodeID uint64, depth int) bool {
			if existing, ok := seen[nodeID]; !ok || depth < existing {
				seen[nodeID] = depth
			}
			return plan.Limit <= 0 || len(seen) < plan.Limit
		}, bitset, frontier); err != nil {
			return nil, err
		}

		bitset.Clear()
		frontier.Clear()
	}

	// Filter by cumulative minimum depth: nodes must satisfy all edge QuantMin requirements
	for nodeID, depth := range seen {
		if depth < totalMinDepth && !(depth == 0 && firstEdgeHasZeroMin) {
			delete(seen, nodeID)
		}
	}

	// Project GraphNodeIDs to SearchResults via ResolveNodeID
	results := &SearchResults{}
	for nodeID := range seen {
		_, recID, err := e.db.ResolveNodeID(ctx, nodeID)
		if err != nil {
			continue
		}
		results.Results = append(results.Results, &SearchResult{
			ID:    recID,
			Score: 1.0,
		})
		if plan.Limit > 0 && len(results.Results) >= plan.Limit {
			break
		}
	}
	results.Total = len(results.Results)
	return results, nil
}

// executeRelational handles exact-match, range, and full-scan queries against a B-tree index.
func (e *Executor) executeRelational(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	col, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
	}
	idx := col.GetIndex()
	if idx == nil {
		return nil, fmt.Errorf("collection %q has no index", plan.CollectionName)
	}

	tree, ok := idx.(interface{ Tree() *btree.BTree })
	if !ok {
		return nil, fmt.Errorf("collection %q index does not support Tree() access", plan.CollectionName)
	}

	// If there's an exact-match predicate, use B-tree Search directly
	if len(plan.Predicates) == 1 && plan.Predicates[0].Operator == 12 { // KindEquals
		pred := plan.Predicates[0]
		val, err := tree.Tree().Search(ctx, pred.Value)
		if err == nil {
			ord, ver, _ := btree.DecodeValue(val)
			return &SearchResults{
				Results: []*SearchResult{{ID: string(pred.Value), Version: uint64(ver), Score: 1.0, Ordinal: ord}},
				Total:   1,
			}, nil
		}
		return &SearchResults{}, nil
	}

	// Range scan or full scan via cursor
	var c *btree.Cursor
	if plan.IsDesc {
		c = tree.Tree().SeekLast()
	} else {
		c = tree.Tree().SeekFirst()
	}

	// Build predicate matchers for range queries
	hasRangeFilter := false
	var rangeStart, rangeEnd []byte
	rangeExclusive := false
	for _, pred := range plan.Predicates {
		switch pred.Operator {
		case 13: // >
			rangeStart = pred.Value
			rangeExclusive = true
			hasRangeFilter = true
		case 14: // <
			rangeEnd = pred.Value
			hasRangeFilter = true
		}
	}
	_ = rangeExclusive

	var results []*SearchResult
	advance := c.Next
	if plan.IsDesc {
		advance = c.Prev
	}

	for c.Valid() {
		key := string(c.Key())

		// Apply range filter
		if hasRangeFilter {
			if rangeStart != nil {
				cmp := key < string(rangeStart)
				if rangeExclusive {
					cmp = key <= string(rangeStart)
				}
				if cmp {
					advance()
					continue
				}
			}
			if rangeEnd != nil && key >= string(rangeEnd) {
				break
			}
		}

		ord, ver, _ := btree.DecodeValue(c.Value())
		results = append(results, &SearchResult{
			ID:      key,
			Version: uint64(ver),
			Ordinal: ord,
			Score:   1.0,
		})

		if plan.Limit > 0 && len(results) >= plan.Limit {
			break
		}
		advance()
	}

	return &SearchResults{Results: results, Total: len(results)}, nil
}

// executeInsert handles INSERT INTO via col.InsertBatch.
func (e *Executor) executeInsert(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	col, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
	}

	// Guardrail: metadata-only collections reject vector columns
	if col.Dimension() == 0 {
		for _, c := range plan.InsertColumns {
			if c == "vector" || c == "vec" || c == "embedding" {
				return nil, fmt.Errorf("collection %q is metadata-only; vector columns not accepted", plan.CollectionName)
			}
		}
	}

	colCount := len(plan.InsertColumns)
	if colCount == 0 {
		colCount = 1 // default single column
	}

	// Group flat values into rows
	var entries []VectorEntry
	for i := 0; i < len(plan.InsertValues); i += colCount {
		var id string
		var vec []float32
		meta := make(map[string]interface{})
		for j := 0; j < colCount && i+j < len(plan.InsertValues); j++ {
			val := string(plan.InsertValues[i+j])
			if colCount > 0 && j < len(plan.InsertColumns) {
				colName := plan.InsertColumns[j]
				if colName == "id" || colName == "ID" {
					id = val
				} else if colName == "vector" || colName == "vec" || colName == "embedding" {
					vec = parseVectorLiteral(val)
					if vec == nil && val != "" {
						return nil, fmt.Errorf("invalid vector literal for column %q: %q", colName, val)
					}
				} else {
					meta[colName] = val
				}
			} else if j == 0 {
				id = val
			}
		}
		if id == "" {
			return nil, fmt.Errorf("INSERT requires an 'id' column")
		}
		entries = append(entries, VectorEntry{ID: id, Vector: vec, Metadata: meta})
	}

	if len(entries) == 0 {
		return &SearchResults{}, nil
	}
	if err := col.InsertBatch(ctx, entries); err != nil {
		return nil, err
	}
	return &SearchResults{Total: len(entries)}, nil
}

// executeAggregate scans a collection and computes COUNT/SUM/AVG/MIN/MAX.
func (e *Executor) executeAggregate(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	col, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
	}
	idx := col.GetIndex()
	if idx == nil {
		return nil, fmt.Errorf("collection %q has no index", plan.CollectionName)
	}
	tree, ok := idx.(interface{ Tree() *btree.BTree })
	if !ok {
		return nil, fmt.Errorf("collection %q index does not support Tree() access", plan.CollectionName)
	}

	// Full scan via cursor
	c := tree.Tree().SeekFirst()
	var count int64
	var sum float64
	var minVal, maxVal string
	hasMinMax := false

	for c.Valid() {
		key := string(c.Key())
		count++

		if plan.AggregateFunc != 0 { // not just COUNT(*)
			if !hasMinMax {
				minVal = key
				maxVal = key
				hasMinMax = true
			}
			if key < minVal {
				minVal = key
			}
			if key > maxVal {
				maxVal = key
			}
			// SUM/AVG: try numeric parse
			var f float64
			if _, err := fmt.Sscanf(key, "%f", &f); err == nil {
				sum += f
			}
		}
		c.Next()
	}

	var resultValue string
	switch plan.AggregateFunc {
	case 0: // COUNT
		resultValue = fmt.Sprintf("%d", count)
	case 1: // SUM
		resultValue = fmt.Sprintf("%f", sum)
	case 2: // AVG
		if count > 0 {
			resultValue = fmt.Sprintf("%f", sum/float64(count))
		} else {
			resultValue = "0"
		}
	case 3: // MIN
		resultValue = minVal
	case 4: // MAX
		resultValue = maxVal
	}

	return &SearchResults{
		Results: []*SearchResult{{ID: resultValue, Score: 1.0}},
		Total:   1,
	}, nil
}

// sqlTypeToFieldType maps SQL column types to metadata FieldTypes for schema
// registration. Returns ok=false for types without a metadata equivalent.
func sqlTypeToFieldType(sqlType string) (FieldType, bool) {
	switch strings.ToUpper(strings.TrimSpace(sqlType)) {
	case "INT", "INTEGER", "BIGINT", "SMALLINT", "SERIAL":
		return IntField, true
	case "TEXT", "VARCHAR", "CHAR", "STRING":
		return StringField, true
	case "FLOAT", "REAL", "DOUBLE", "DOUBLE PRECISION", "DECIMAL", "NUMERIC":
		return FloatField, true
	case "BOOL", "BOOLEAN":
		return BoolField, true
	case "TIMESTAMP", "TIME", "DATE":
		return TimeField, true
	default:
		return StringField, false
	}
}

// executeDDL handles CREATE TABLE, DROP TABLE, CREATE INDEX.
func (e *Executor) executeDDL(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	switch plan.DDLKind {
	case 0: // CREATE TABLE
		opts := []CollectionOption{WithMetadataOnly()}
		var schema MetadataSchema
		for _, col := range plan.DDLColumns {
			if col.Type == "VECTOR" || col.Type == "vector" || col.Type == "FLOAT[]" || col.Type == "float[]" {
				// Vector column present — switch to vector mode; dimension must be specified
				// For now, treat any column named "vector" as needing dimension 3
				opts = []CollectionOption{WithDimension(3)}
				continue
			}
			if schema == nil {
				schema = make(MetadataSchema)
			}
			if ft, ok := sqlTypeToFieldType(col.Type); ok {
				schema[col.Name] = ft
			}
		}
		if len(schema) > 0 {
			opts = append(opts, WithMetadataSchema(schema))
		}
		_, err := e.db.CreateCollection(ctx, plan.DDLTableName, opts...)
		if err != nil {
			return nil, err
		}
		return &SearchResults{}, nil

	case 1: // DROP TABLE
		if err := e.db.DeleteCollection(ctx, plan.DDLTableName); err != nil {
			return nil, err
		}
		return &SearchResults{}, nil

	case 2: // CREATE INDEX
		// Index creation is handled transparently by the collection
		if _, err := e.db.GetCollection(plan.DDLTableName); err != nil {
			if plan.DDLIfExists {
				return &SearchResults{}, nil
			}
			return nil, fmt.Errorf("CREATE INDEX: table %q not found", plan.DDLTableName)
		}
		return &SearchResults{}, nil

	case 3: // DROP INDEX
		// Index management is internal — no-op for now
		return &SearchResults{}, nil

	case 4: // ALTER TABLE ADD COLUMN
		// Currently a no-op: column metadata is stored in the catalog
		// Future: validate table exists, propagate column to catalog
		if _, err := e.db.GetCollection(plan.DDLTableName); err != nil {
			return nil, fmt.Errorf("ALTER TABLE: table %q not found", plan.DDLTableName)
		}
		return &SearchResults{}, nil

	default:
		return nil, fmt.Errorf("unknown DDL kind %d", plan.DDLKind)
	}
}

func parseVectorLiteral(s string) []float32 {
	if len(s) >= 2 && s[0] == '[' && s[len(s)-1] == ']' {
		s = s[1 : len(s)-1]
	}
	parts := splitComma(s)
	if len(parts) == 0 {
		return nil
	}
	floats := make([]float32, len(parts))
	for i, part := range parts {
		if _, err := fmt.Sscanf(part, "%f", &floats[i]); err != nil {
			return nil // garbage — caller should reject
		}
	}
	return floats
}

func splitComma(s string) []string {
	var parts []string
	start := 0
	for i := 0; i <= len(s); i++ {
		if i == len(s) || s[i] == ',' {
			p := s[start:i]
			for len(p) > 0 && p[0] == ' ' { p = p[1:] }
			for len(p) > 0 && p[len(p)-1] == ' ' { p = p[:len(p)-1] }
			if len(p) > 0 {
				parts = append(parts, p)
			}
			start = i + 1
		}
	}
	return parts
}

// executeUpdate handles UPDATE ... SET ... WHERE via SELECT-then-write.
func (e *Executor) executeUpdate(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	if len(plan.Predicates) == 0 {
		return nil, fmt.Errorf("UPDATE requires a WHERE clause")
	}
	// Phase 1: resolve matching IDs via relational execution
	resolvePlan := &optimizer.PhysicalPlan{
		Kind:           optimizer.QueryKindRelational,
		CollectionName: plan.CollectionName,
		Predicates:     plan.Predicates,
		HasRelationalQuery: len(plan.Predicates) > 0,
	}
	results, err := e.executeRelational(ctx, resolvePlan)
	if err != nil {
		return nil, fmt.Errorf("UPDATE resolve phase: %w", err)
	}
	if len(results.Results) == 0 {
		return results, nil
	}

	// Phase 2: all-or-nothing write via transaction
	tx, err := e.db.BeginTx(ctx)
	if err != nil {
		return nil, err
	}
	ids := make([]string, len(results.Results))
	for i, r := range results.Results {
		ids[i] = r.ID
		meta := make(map[string]interface{})
		for j, col := range plan.SetColumns {
			if j < len(plan.SetValues) {
				meta[col] = string(plan.SetValues[j])
			}
		}
		if err := tx.Update(ctx, plan.CollectionName, r.ID, nil, meta); err != nil {
			tx.Rollback(ctx)
			return nil, fmt.Errorf("UPDATE row %q: %w", r.ID, err)
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return nil, err
	}
	return &SearchResults{Results: results.Results, Total: len(ids)}, nil
}

// executeDelete handles DELETE FROM ... WHERE via SELECT-then-write.
func (e *Executor) executeDelete(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	if len(plan.Predicates) == 0 {
		return nil, fmt.Errorf("DELETE requires a WHERE clause")
	}
	resolvePlan := &optimizer.PhysicalPlan{
		Kind:           optimizer.QueryKindRelational,
		CollectionName: plan.CollectionName,
		Predicates:     plan.Predicates,
		HasRelationalQuery: len(plan.Predicates) > 0,
	}
	results, err := e.executeRelational(ctx, resolvePlan)
	if err != nil {
		return nil, fmt.Errorf("DELETE resolve phase: %w", err)
	}
	if len(results.Results) == 0 {
		return results, nil
	}

	ids := make([]string, len(results.Results))
	for i, r := range results.Results {
		ids[i] = r.ID
	}

	tx, err := e.db.BeginTx(ctx)
	if err != nil {
		return nil, err
	}
	if err := tx.DeleteBatch(ctx, plan.CollectionName, ids); err != nil {
		tx.Rollback(ctx)
		return nil, err
	}
	if err := tx.Commit(ctx); err != nil {
		return nil, err
	}
	return &SearchResults{Results: results.Results, Total: len(ids)}, nil
}

// executeJoin performs a merge join over two B-tree-indexed collections.
// Both cursors advance in lockstep — O(N+M) with zero extra structures.
// Supports INNER (default), LEFT, and CROSS join types.
func (e *Executor) executeJoin(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	if len(plan.Joins) == 0 {
		return e.executeRelational(ctx, plan)
	}

	leftCol, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
	}
	leftTree, ok := leftCol.GetIndex().(interface{ Tree() *btree.BTree })
	if !ok {
		return nil, fmt.Errorf("JOIN left collection %q does not support Tree() access", plan.CollectionName)
	}

	var results []*SearchResult
	for _, join := range plan.Joins {
		rightCol, err := e.db.GetCollection(join.CollectionName)
		if err != nil {
			return nil, err
		}
		rightTree, ok := rightCol.GetIndex().(interface{ Tree() *btree.BTree })
		if !ok {
			return nil, fmt.Errorf("JOIN right collection %q does not support Tree() access", join.CollectionName)
		}

		isLeftJoin := join.JoinType == 1 // parser.JoinLeft

		left := leftTree.Tree().SeekFirst()
		right := rightTree.Tree().SeekFirst()

		for left.Valid() {
			leftKey := string(left.Key())

			if !right.Valid() {
				// Right exhausted — for LEFT JOIN, emit remaining left rows
				if isLeftJoin {
					results = append(results, &SearchResult{
						ID:    leftKey + "|",
						Score: 1.0,
					})
				}
				left.Next()
				continue
			}

			rightKey := string(right.Key())

			if leftKey < rightKey {
				if isLeftJoin {
					// LEFT JOIN: emit left row with empty right side
					results = append(results, &SearchResult{
						ID:    leftKey + "|",
						Score: 1.0,
					})
				}
				left.Next()
			} else if leftKey > rightKey {
				right.Next()
			} else {
				// Match — collect all right matches for this left key
				for right.Valid() && string(right.Key()) == leftKey {
					results = append(results, &SearchResult{
						ID:    leftKey + "|" + string(right.Key()),
						Score: 1.0,
					})
					right.Next()
				}
				left.Next()
			}
			if plan.Limit > 0 && len(results) >= plan.Limit {
				break
			}
		}
		if plan.Limit > 0 && len(results) >= plan.Limit {
			break
		}
	}
	return &SearchResults{Results: results, Total: len(results)}, nil
}
