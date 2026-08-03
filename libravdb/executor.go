package libravdb

import (
	"context"
	"errors"
	"fmt"

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
