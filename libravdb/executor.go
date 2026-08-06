package libravdb

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/xDarkicex/libravdb/internal/catalog"
	"github.com/xDarkicex/libravdb/internal/graph"
	btree "github.com/xDarkicex/libravdb/internal/index/btree"
	"github.com/xDarkicex/libravdb/internal/optimizer"
	"github.com/xDarkicex/libravdb/internal/util"
)

// Executor dispatches physical plans to concrete execution paths.
type Executor struct {
	db *Database
}

func recordsVisibleInContext(ctx context.Context, col *Collection) ([]Record, error) {
	if epoch := epochFromContext(ctx); epoch != nil {
		return epoch.ListRecords(ctx, col.name)
	}
	return col.ListAll(ctx)
}

func newExecutor(db *Database) *Executor {
	return &Executor{db: db}
}

// ExecuteAtLSN executes a physical plan against the historical state at
// snapshotLSN. The plan's SnapshotLSN field is set automatically. All reads
// use temporal APIs (GetAtLSN, NeighborsAtLSN, exact vector scoring). Live
// HNSW is never used for historical queries.
func (e *Executor) ExecuteAtLSN(ctx context.Context, plan *optimizer.PhysicalPlan, snapshotLSN uint64) (*SearchResults, error) {
	if snapshotLSN == 0 {
		return nil, fmt.Errorf("snapshot LSN must be non-zero for temporal execution")
	}
	plan.SnapshotLSN = snapshotLSN
	return e.executeTemporal(ctx, plan)
}

// executeTemporal routes a plan with SnapshotLSN != 0 to the appropriate
// temporal execution path.
func (e *Executor) executeTemporal(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	switch {
	case plan.Kind == optimizer.QueryKindMultiModal:
		return e.executeMultiModalAtLSN(ctx, plan)
	case plan.Kind == optimizer.QueryKindGraph:
		return e.executeGraphAtLSN(ctx, plan)
	case plan.Kind == optimizer.QueryKindRelational:
		return e.executeRelationalAtLSN(ctx, plan)
	case plan.Kind == optimizer.QueryKindVectorProjection:
		return e.executeVectorProjectionAtLSN(ctx, plan)
	default:
		return nil, fmt.Errorf("temporal execution not supported for query kind %d", plan.Kind)
	}
}

// executeGraphAtLSN executes GRAPH_TABLE queries against the historical graph
// snapshot. It deliberately discovers the complete bounded path before LIMIT
// is applied, so LIMIT cannot hide a qualifying historical match. For a
// WHERE MATCH relation the source rows are returned (the graph predicate is
// existential); GRAPH_TABLE retains its endpoint-materialization behavior.
func (e *Executor) executeGraphAtLSN(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	col, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
	}
	g := col.GetGraph()
	if g == nil {
		return nil, fmt.Errorf("collection %q has no graph", plan.CollectionName)
	}
	var seeds []uint64
	var seedPredicates []optimizer.RelationalPredicate
	var terminalPredicates = plan.Predicates
	var terminalLabels map[uint64]struct{}
	matchJoin := len(plan.GraphJoins) > 0
	if matchJoin {
		join := plan.GraphJoins[0]
		terminalPredicates = join.TerminalPredicates
		for _, predicate := range plan.Predicates {
			if predicate.Alias == "" || predicate.Alias == join.LeftAlias {
				seedPredicates = append(seedPredicates, predicate)
			}
		}
		if join.TerminalLabel != "" {
			terminalLabels = make(map[uint64]struct{})
			for _, nodeID := range g.GetLabelNodes(join.TerminalLabel) {
				terminalLabels[nodeID] = struct{}{}
			}
		}
	}
	if plan.HasExplicitSeed {
		seeds = append(seeds, plan.ExplicitSeedID)
	} else if plan.SeedLabel != "" {
		seeds = append(seeds, g.GetLabelNodes(plan.SeedLabel)...)
	} else {
		// A relational FROM ... WHERE MATCH query names its start vertex by
		// alias, not by an explicit ID or label. In that shape the complete
		// visible collection is the seed relation; the bounded BFS below then
		// retains only source rows with a qualifying historical endpoint.
		if err := col.ListVisibleAtLSN(ctx, plan.SnapshotLSN, func(rec *Record) bool {
			if len(seedPredicates) > 0 && !recordMatchesPredicatesSnapshot(rec, seedPredicates) {
				return true
			}
			nodeID, nodeErr := e.db.GetNodeID(ctx, plan.CollectionName, rec.ID)
			if nodeErr == nil {
				seeds = append(seeds, nodeID)
			}
			return true
		}); err != nil {
			return nil, err
		}
	}
	if len(plan.GraphEdges) == 0 {
		return &SearchResults{Columns: plan.Projections}, nil
	}
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
	edges := make([]EdgePlan, len(plan.GraphEdges))
	for i, gep := range plan.GraphEdges {
		if gep.EdgeType != "" && gep.EdgeKind == 0 {
			return nil, fmt.Errorf("graph edge kind %q is not registered", gep.EdgeType)
		}
		edges[i] = graphEdgePlanForTraversal(gep)
	}
	candidates := make(map[string]struct{})
	tn, ok := g.(interface {
		NeighborsAtLSN(uint64, uint64) ([]Edge, error)
	})
	if !ok {
		return nil, fmt.Errorf("collection %q graph does not support temporal traversal", col.name)
	}
	for _, seed := range seeds {
		anchorID := ""
		if matchJoin {
			if _, resolvedID, resolveErr := e.db.ResolveNodeID(ctx, seed); resolveErr == nil {
				anchorID = resolvedID
			}
		}
		if err := e.temporalBFSPattern(ctx, tn, seed, anchorID, matchJoin, edges, plan.MaxHops, plan.SnapshotLSN, terminalPredicates, terminalLabels, candidates, bitset, frontier); err != nil {
			return nil, err
		}
		bitset.Clear()
		frontier.Clear()
	}
	ids := make([]string, 0, len(candidates))
	for id := range candidates {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	if plan.Limit > 0 && len(ids) > plan.Limit {
		ids = ids[:plan.Limit]
	}
	results := &SearchResults{Columns: plan.Projections, Results: make([]*SearchResult, 0, len(ids))}
	for _, id := range ids {
		rec, rerr := col.GetAtLSN(ctx, id, plan.SnapshotLSN)
		if rerr != nil || rec == nil {
			continue
		}
		results.Results = append(results.Results, &SearchResult{ID: id, Score: 1, Metadata: rec.Metadata})
	}
	results.Total = len(results.Results)
	return results, nil
}

// Execute routes a physical plan to the appropriate execution engine.
// Temporal queries (AS OF TIMESTAMP) are resolved to an LSN and routed
// through the temporal execution path.
func (e *Executor) Execute(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	// Resolve temporal snapshot before any data access.
	var temporalHandle *TemporalSnapshot
	if !plan.SnapshotTimestamp.IsZero() && plan.SnapshotLSN == 0 {
		snap, err := e.db.SnapshotAt(ctx, plan.SnapshotTimestamp)
		if err != nil {
			return nil, fmt.Errorf("AS OF TIMESTAMP %s: %w",
				plan.SnapshotTimestamp.Format(time.RFC3339), err)
		}
		defer snap.Close()
		temporalHandle = snap
		plan.SnapshotLSN = snap.LSN
	}
	if plan.SnapshotLSN != 0 {
		// Hold the pin for the duration of temporal execution.
		results, err := e.ExecuteAtLSN(ctx, plan, plan.SnapshotLSN)
		if temporalHandle != nil {
			temporalHandle.Close()
		}
		return results, err
	}

	// System tables (pg_class, etc.) are materialized in memory rather than
	// looked up as collections. The binder assigns reserved OIDs 1-99 to them.
	if catalog.IsSystemTableOID(plan.CollectionOID) {
		return e.executeSystemTable(ctx, plan)
	}

	// A composed relational + graph + vector plan owns all of its clauses. It
	// must run before generic hybrid routing, which only understands a single
	// collection's scalar/graph constraints.
	if plan.Kind == optimizer.QueryKindMultiModal {
		if plan.HasScoreExpr && plan.HasGraphCentrality {
			return e.executeLiveScored(ctx, plan)
		}
		return e.executeMultiModal(ctx, plan)
	}

	// Hybrid queries (vector + predicates/graph) route through the adaptive
	// cost-based dispatcher. Pure vector, pure relational, and pure graph
	// queries keep their existing fast paths.
	if isHybridQuery(plan) {
		return e.executeHybrid(ctx, plan)
	}

	switch plan.Kind {
	case optimizer.QueryKindKNN:
		return e.executeKNN(ctx, plan)
	case optimizer.QueryKindVectorProjection:
		return e.executeVectorProjection(ctx, plan)
	case optimizer.QueryKindGraph:
		return e.executeGraph(ctx, plan)
	case optimizer.QueryKindRelational:
		return e.executeRelational(ctx, plan)
	case optimizer.QueryKindInsert:
		return e.executeInsert(ctx, plan)
	case optimizer.QueryKindInsertGraphEdge:
		return e.executeInsertGraphEdge(ctx, plan)
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

// executeMultiModal composes relational anchor selection, MATCH traversal,
// and vector top-k. The intermediate representation is record IDs, never
// user-facing joined rows: anchors select BFS seeds, terminal graph vertices
// become a bitmap, and the existing filtered-ANN path ranks only those
// terminals.
func (e *Executor) executeMultiModal(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	if len(plan.Joins) == 0 || len(plan.GraphJoins) == 0 || len(plan.QueryVector) == 0 {
		return nil, fmt.Errorf("multimodal query requires relational JOIN, JOIN MATCH, and vector top-k")
	}
	col, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
	}
	anchors, err := e.multiModalAnchors(ctx, col, plan.Joins)
	if err != nil {
		return nil, err
	}
	if len(anchors) == 0 {
		return &SearchResults{Columns: plan.Projections}, nil
	}
	candidates, err := e.multiModalGraphCandidates(ctx, col, plan.GraphJoins, anchors)
	if err != nil {
		return nil, err
	}
	if len(candidates) == 0 {
		return &SearchResults{Columns: plan.Projections}, nil
	}

	matchCount := len(candidates)

	// Exact fallback scores the authoritative IDs directly. In particular, it
	// avoids treating a shard-local ordinal as globally unique.
	if matchCount <= exactCandidateCap/10 && plan.RecallContract != optimizer.RecallExact {
		return e.executeMultiModalExact(ctx, col, plan, candidates)
	}

	// Build ordinal bitmap directly from pre-computed candidate IDs — no
	// collection enumeration. Sharded collections receive one local bitmap per
	// HNSW shard through GraphFilter.ForShard.
	bitmap, err := e.buildOrdinalBitmapFromIDs(ctx, col, candidates)
	if err != nil {
		return nil, err
	}
	defer bitmap.release()

	k := plan.Limit
	if k <= 0 {
		k = 10
	}
	sigma := bitmap.selectivity
	searchEf := e.calibratedBinomialStart(plan.CollectionName, k, sigma)

	qb := col.Query(ctx)
	qb.WithVector(plan.QueryVector)
	qb.WithGraphFilter(bitmap)
	if searchEf > 0 {
		qb.WithEfSearch(searchEf)
	}
	qb.Limit(k)

	results, err := qb.Execute()
	if err != nil {
		return nil, err
	}
	return e.buildSelectResults(ctx, col, results.Results, plan), nil
}

// executeMultiModalAtLSN is the temporal variant of executeMultiModal. It uses
// executeLiveScored handles multimodal queries with compound scoring expressions
// in the live (non-temporal) path. Graph candidates are generated, centrality is
// computed per-candidate, and the scored expression is evaluated.
func (e *Executor) executeLiveScored(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	col, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
	}
	anchors, err := e.multiModalAnchors(ctx, col, plan.Joins)
	if err != nil {
		return nil, err
	}
	if len(anchors) == 0 {
		return &SearchResults{Columns: plan.Projections}, nil
	}
	candidates, err := e.multiModalGraphCandidates(ctx, col, plan.GraphJoins, anchors)
	if err != nil {
		return nil, err
	}
	if len(candidates) == 0 {
		return &SearchResults{Columns: plan.Projections}, nil
	}
	centralityMap := make(map[string]float64, len(candidates))
	for id := range candidates {
		if nodeID, err := e.db.GetNodeID(ctx, col.name, id); err == nil {
			centralityMap[id] = col.graph.GraphCentrality(nodeID)
		}
	}
	distFn, _ := util.GetDistanceFunc(util.DistanceMetric(col.config.Metric))
	expr := buildScoreExpr(plan, distFn, plan.QueryVector)
	return e.executeScoredMultiModalWithCentrality(ctx, col, plan, candidates, expr, centralityMap, 0, plan.IsDesc)
}

// ListVisibleAtLSN for anchors, temporal graph traversal for candidates, and
// exact vector scoring from historical record versions. Live HNSW is never used.
func (e *Executor) executeMultiModalAtLSN(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	snapshotLSN := plan.SnapshotLSN
	col, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
	}

	// 1. Relational anchors at snapshot LSN.
	anchors, err := e.multiModalAnchorsAtLSN(ctx, col, plan.Joins, snapshotLSN)
	if err != nil {
		return nil, err
	}
	if len(anchors) == 0 {
		return &SearchResults{Columns: plan.Projections}, nil
	}
	// 2. Graph traversal at snapshot LSN.
	candidates, err := e.multiModalGraphCandidatesAtLSN(ctx, col, plan, anchors, snapshotLSN)
	if err != nil {
		return nil, err
	}
	if len(candidates) == 0 {
		return &SearchResults{Columns: plan.Projections}, nil
	}

	// 3. Scoring: use expression-based scoring when HasScoreExpr is set,
	// otherwise use exact vector scoring from historical record versions.
	if plan.HasScoreExpr && plan.HasGraphCentrality {
		centralityMap := make(map[string]float64, len(candidates))
		for id := range candidates {
			if nodeID, err := e.db.GetNodeID(ctx, col.name, id); err == nil {
				// Temporal: use CentralityAtLSN for historical snapshots.
				if snapshotLSN != 0 {
					if g, ok := col.graph.(interface{ CentralityAtLSN(uint64, uint64) float64 }); ok {
						centralityMap[id] = g.CentralityAtLSN(nodeID, snapshotLSN)
						continue
					}
				}
				centralityMap[id] = col.graph.GraphCentrality(nodeID)
			}
		}
		distFn, _ := util.GetDistanceFunc(util.DistanceMetric(col.config.Metric))
		expr := buildScoreExpr(plan, distFn, plan.QueryVector)
		return e.executeScoredMultiModalWithCentrality(ctx, col, plan, candidates, expr, centralityMap, snapshotLSN, plan.IsDesc)
	}
	return e.scoreCandidatesAtLSN(ctx, col, plan, candidates, snapshotLSN)
}

// multiModalAnchorsAtLSN applies relational joins using only records visible
// at the snapshot LSN.
func (e *Executor) multiModalAnchorsAtLSN(ctx context.Context, left *Collection, joins []optimizer.JoinPlan, snapshotLSN uint64) ([]string, error) {
	// Collect left records visible at snapshot.
	leftIDs := make(map[string]struct{})
	if err := left.ListVisibleAtLSN(ctx, snapshotLSN, func(r *Record) bool {
		leftIDs[r.ID] = struct{}{}
		return true
	}); err != nil {
		return nil, err
	}

	for _, join := range joins {
		if join.LeftColumn == "" || join.RightColumn == "" {
			return nil, fmt.Errorf("JOIN requires an equality condition")
		}
		right, err := e.db.GetCollection(join.CollectionName)
		if err != nil {
			return nil, err
		}
		rightKeys := make(map[string]struct{})
		if err := right.ListVisibleAtLSN(ctx, snapshotLSN, func(r *Record) bool {
			if !recordMatchesPredicatesSnapshot(r, join.RightPredicates) {
				return true
			}
			if key, ok := multiModalRecordColumn(*r, join.RightColumn); ok {
				rightKeys[key] = struct{}{}
			}
			return true
		}); err != nil {
			return nil, err
		}
		// Intersect with left — only records visible at snapshot count.
		for id := range leftIDs {
			rec, err := left.GetAtLSN(ctx, id, snapshotLSN)
			if err != nil || rec == nil {
				delete(leftIDs, id)
				continue
			}
			key, ok := multiModalRecordColumn(*rec, join.LeftColumn)
			if !ok {
				delete(leftIDs, id)
				continue
			}
			if _, ok := rightKeys[key]; !ok {
				delete(leftIDs, id)
			}
		}
	}

	anchors := make([]string, 0, len(leftIDs))
	for id := range leftIDs {
		anchors = append(anchors, id)
	}
	return anchors, nil
}

// multiModalGraphCandidatesAtLSN traverses the graph at the snapshot LSN,
// using only edges visible at that LSN.
func (e *Executor) multiModalGraphCandidatesAtLSN(ctx context.Context, col *Collection, plan *optimizer.PhysicalPlan, anchors []string, snapshotLSN uint64) (map[string]struct{}, error) {
	g := col.GetGraph()
	if g == nil {
		return nil, fmt.Errorf("collection %q has no graph", col.name)
	}
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

	// Temporal neighbor lookup: use NeighborsAtLSN if available.
	type temporalNeighbor interface {
		NeighborsAtLSN(nodeID uint64, snapshotLSN uint64) ([]Edge, error)
	}

	candidates := make(map[string]struct{})
	for _, join := range plan.GraphJoins {
		edges := make([]EdgePlan, len(join.GraphEdges))
		for i, gep := range join.GraphEdges {
			edges[i] = graphEdgePlanForTraversal(gep)
			if gep.EdgeType != "" && gep.EdgeKind == 0 {
				return nil, fmt.Errorf("graph edge kind %q is not registered", gep.EdgeType)
			}
			if gep.EdgeKind != 0 {
				edges[i].KindSet.Set(gep.EdgeKind)
			}
		}
		if len(edges) == 0 {
			continue
		}
		terminalLabels := map[uint64]struct{}(nil)
		if join.TerminalLabel != "" {
			terminalLabels = make(map[uint64]struct{})
			for _, nodeID := range g.GetLabelNodes(join.TerminalLabel) {
				terminalLabels[nodeID] = struct{}{}
			}
		}
		scoreAnchor := join.PredicateMatch
		if !scoreAnchor && len(plan.VectorFuncProjections) > 0 {
			scoreAnchor = plan.VectorFuncProjections[0].SourceAlias != "" &&
				plan.VectorFuncProjections[0].SourceAlias == join.LeftAlias
		}
		for _, anchorID := range anchors {
			nodeID, err := e.db.GetNodeID(ctx, col.name, anchorID)
			if err != nil {
				continue
			}
			// Temporal BFS: filter edges by snapshot LSN visibility.
			if tn, ok := g.(temporalNeighbor); ok {
				if err := e.temporalBFSPattern(ctx, tn, nodeID, anchorID, scoreAnchor, edges, join.MaxHops, snapshotLSN, join.TerminalPredicates, terminalLabels, candidates, bitset, frontier); err != nil {
					return nil, err
				}
			} else {
				return nil, fmt.Errorf("collection %q graph does not support temporal traversal", col.name)
			}
			bitset.Clear()
			frontier.Clear()
		}
	}
	return candidates, nil
}

// graphEdgePlanForTraversal converts the parser's compact quantifier encoding
// into traversal semantics. (0, 0) is an unquantified SQL edge, meaning
// exactly one hop—not a zero-hop match. Keeping this conversion in one place
// prevents every graph execution path from silently treating MATCH (a)->(b)
// as though (a) itself could satisfy the terminal pattern.
func graphEdgePlanForTraversal(gep optimizer.GraphEdgePlan) EdgePlan {
	minHops := int(gep.QuantMin)
	maxHops := int(gep.QuantMax)
	if maxHops == 0 {
		if minHops == 0 {
			minHops, maxHops = 1, 1
		} else {
			maxHops = 1 << 20
		}
	}
	ep := EdgePlan{Dir: gep.Direction, Min: minHops, Max: maxHops}
	if gep.EdgeKind != 0 {
		ep.KindSet.Set(gep.EdgeKind)
	}
	return ep
}

// temporalBFSPattern runs BFS using NeighborsAtLSN for temporal edge visibility.
func (e *Executor) temporalBFSPattern(ctx context.Context, g interface {
	NeighborsAtLSN(nodeID uint64, snapshotLSN uint64) ([]Edge, error)
}, start uint64, anchorID string, scoreAnchor bool, edges []EdgePlan, maxDepth int, snapshotLSN uint64, terminalPredicates []optimizer.RelationalPredicate, terminalLabels map[uint64]struct{}, candidates map[string]struct{}, bitset *Bitset, frontier *FrontierBuf) error {
	bitset.Clear()
	frontier.Clear()

	// Seed the frontier with start node, band 0.
	visitedKey := func(nodeID uint64, band int) uint64 {
		return nodeID*uint64(len(edges)) + uint64(band)
	}
	bitset.Set(visitedKey(start, 0))
	frontier.Push(start, 0, 0)

	for !frontier.Empty() {
		nodeID, band, step := frontier.Pop()
		if band >= len(edges) {
			continue
		}
		edgePlan := edges[band]

		// Check if we've reached the final band with sufficient hops.
		if band == len(edges)-1 && step >= edgePlan.Min && step <= edgePlan.Max {
			labelMatches := true
			if terminalLabels != nil {
				_, labelMatches = terminalLabels[nodeID]
			}
			if labelMatches {
				if colName, recordID, err := e.db.ResolveNodeID(ctx, nodeID); err == nil {
					if col, cerr := e.db.GetCollection(colName); cerr == nil {
						if rec, rerr := col.GetAtLSN(ctx, recordID, snapshotLSN); rerr == nil && rec != nil {
							if recordMatchesPredicatesSnapshot(rec, terminalPredicates) {
								if scoreAnchor {
									candidates[anchorID] = struct{}{}
								} else {
									candidates[recordID] = struct{}{}
								}
							}
						}
					}
				}
			}
			if len(edges) == 1 && step >= edgePlan.Max {
				continue
			}
		}

		// Expand within current band if under max hops.
		if step < edgePlan.Max {
			neighbors, err := g.NeighborsAtLSN(nodeID, snapshotLSN)
			if err != nil {
				return err
			}
			for _, neighbor := range neighbors {
				// Filter by edge kind if specified.
				if !edgePlan.KindSet.Has(neighbor.GetKind()) {
					continue
				}
				key := visitedKey(neighbor.Target, band)
				if !bitset.Test(key) {
					bitset.Set(key)
					frontier.Push(neighbor.Target, band, step+1)
				}
			}
		}

		// Transition to next band if min hops satisfied.
		if step >= edgePlan.Min && band+1 < len(edges) && step <= edgePlan.Max {
			nextKey := visitedKey(nodeID, band+1)
			if !bitset.Test(nextKey) {
				bitset.Set(nextKey)
				frontier.Push(nodeID, band+1, 0)
			}
		}
	}
	return nil
}

// scoreCandidatesAtLSN scores each candidate using its historical vector at
// snapshotLSN, then returns top-k by vector distance.
func (e *Executor) scoreCandidatesAtLSN(ctx context.Context, col *Collection, plan *optimizer.PhysicalPlan, candidates map[string]struct{}, snapshotLSN uint64) (*SearchResults, error) {
	k := plan.Limit
	if k <= 0 {
		k = 10
	}
	type result struct {
		id    string
		score float32
	}
	results := make([]result, 0, len(candidates))
	for id := range candidates {
		rec, err := col.GetAtLSN(ctx, id, snapshotLSN)
		if err != nil || rec == nil || len(rec.Vector) == 0 {
			continue
		}
		score := computeVectorScore(col, optimizer.VectorFuncProjection{
			IsDistance:  true,
			QueryVector: plan.QueryVector,
		}, rec.Vector)
		results = append(results, result{id: id, score: score})
	}
	sort.Slice(results, func(i, j int) bool { return results[i].score < results[j].score })
	if len(results) > k {
		results = results[:k]
	}
	out := &SearchResults{Results: make([]*SearchResult, len(results)), Total: len(results), Columns: plan.Projections}
	for i, r := range results {
		rec, _ := col.GetAtLSN(ctx, r.id, snapshotLSN)
		sr := &SearchResult{ID: r.id, Score: r.score}
		if rec != nil {
			sr.Metadata = rec.Metadata
		}
		out.Results[i] = sr
	}
	return out, nil
}

// executeRelationalAtLSN handles simple relational reads at a snapshot LSN.
// executeVectorProjectionAtLSN scores all visible records at snapshotLSN
// using the plan's vector function projections and returns top-k.
func (e *Executor) executeVectorProjectionAtLSN(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	col, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
	}
	if len(plan.QueryVector) == 0 {
		return nil, fmt.Errorf("no query vector for temporal vector projection")
	}
	k := plan.Limit
	if k <= 0 {
		k = 10
	}
	type scored struct {
		rec   *Record
		score float32
	}
	var results []scored
	col.ListVisibleAtLSN(ctx, plan.SnapshotLSN, func(r *Record) bool {
		if len(r.Vector) == 0 || len(r.Vector) != len(plan.QueryVector) {
			return true
		}
		if plan.HasRelationalQuery && len(plan.Predicates) > 0 &&
			!recordMatchesPredicatesSnapshot(r, plan.Predicates) {
			return true
		}
		s := computeVectorScore(col, optimizer.VectorFuncProjection{
			IsDistance: true, QueryVector: plan.QueryVector,
		}, r.Vector)
		results = append(results, scored{rec: r, score: s})
		return true
	})
	// Sort by score ascending (distance).
	for i := 1; i < len(results); i++ {
		for j := i; j > 0 && results[j].score < results[j-1].score; j-- {
			results[j], results[j-1] = results[j-1], results[j]
		}
	}
	if len(results) > k {
		results = results[:k]
	}
	out := &SearchResults{Results: make([]*SearchResult, len(results)), Total: len(results)}
	for i, s := range results {
		out.Results[i] = &SearchResult{
			ID: s.rec.ID, Score: s.score, Metadata: s.rec.Metadata,
		}
	}
	out.Columns = plan.Projections
	return out, nil
}

func (e *Executor) executeRelationalAtLSN(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	col, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
	}
	var results []*SearchResult
	if err := col.ListVisibleAtLSN(ctx, plan.SnapshotLSN, func(r *Record) bool {
		if plan.HasRelationalQuery && len(plan.Predicates) > 0 && !recordMatchesPredicatesSnapshot(r, plan.Predicates) {
			return true
		}
		results = append(results, &SearchResult{ID: r.ID, Score: 1.0, Metadata: r.Metadata})
		return plan.Limit <= 0 || len(results) < plan.Limit
	}); err != nil {
		return nil, err
	}
	return &SearchResults{Results: results, Total: len(results), Columns: plan.Projections}, nil
}

func recordMatchesPredicatesSnapshot(r *Record, predicates []optimizer.RelationalPredicate) bool {
	for _, pred := range predicates {
		if !recordMatchesPredicateSnapshot(r, pred) {
			return false
		}
	}
	return true
}

func recordMatchesPredicateSnapshot(r *Record, pred optimizer.RelationalPredicate) bool {
	if pred.Column == "id" || pred.Column == "ID" {
		return compareColumn(r.ID, string(pred.Value), pred.Operator)
	}
	if r.Metadata == nil {
		return false
	}
	v, ok := r.Metadata[pred.Column]
	if !ok {
		return false
	}
	return compareColumn(recordMetaToString(v), string(pred.Value), pred.Operator)
}

func (e *Executor) executeMultiModalExact(ctx context.Context, col *Collection, plan *optimizer.PhysicalPlan, candidateIDs map[string]struct{}) (*SearchResults, error) {
	records := make([]Record, 0, len(candidateIDs))
	for id := range candidateIDs {
		record, err := col.Get(ctx, id)
		if err != nil {
			if errors.Is(err, ErrRecordNotFound) || isNotFoundError(err) {
				continue
			}
			return nil, err
		}
		records = append(records, record)
	}
	results := scoreAndSelectTopK(col, records, plan.QueryVector, plan.Limit)
	return e.buildSelectResults(ctx, col, results.Results, plan), nil
}

// multiModalAnchors applies every relational join before graph traversal.
// Right-side literal ON predicates are pushed into the right input, then the
// surviving right join keys are intersected with the left records' keys.
func (e *Executor) multiModalAnchors(ctx context.Context, left *Collection, joins []optimizer.JoinPlan) ([]string, error) {
	leftRecords, err := recordsVisibleInContext(ctx, left)
	if err != nil {
		return nil, err
	}
	allowed := make(map[string]struct{}, len(leftRecords))
	for _, record := range leftRecords {
		allowed[record.ID] = struct{}{}
	}
	for _, join := range joins {
		if join.LeftColumn == "" || join.RightColumn == "" {
			return nil, fmt.Errorf("JOIN %q requires an equality condition between left and right columns", join.CollectionName)
		}
		right, err := e.db.GetCollection(join.CollectionName)
		if err != nil {
			return nil, err
		}
		rightRecords, err := recordsVisibleInContext(ctx, right)
		if err != nil {
			return nil, err
		}
		rightKeys := make(map[string]struct{}, len(rightRecords))
		for _, record := range rightRecords {
			if !recordMatchesPredicates(record, join.RightPredicates) {
				continue
			}
			if key, ok := multiModalRecordColumn(record, join.RightColumn); ok {
				rightKeys[key] = struct{}{}
			}
		}
		for _, record := range leftRecords {
			if _, ok := allowed[record.ID]; !ok {
				continue
			}
			key, ok := multiModalRecordColumn(record, join.LeftColumn)
			if !ok {
				delete(allowed, record.ID)
				continue
			}
			if _, ok := rightKeys[key]; !ok {
				delete(allowed, record.ID)
			}
		}
	}
	anchors := make([]string, 0, len(allowed))
	for _, record := range leftRecords {
		if _, ok := allowed[record.ID]; ok {
			anchors = append(anchors, record.ID)
		}
	}
	return anchors, nil
}

func multiModalRecordColumn(record Record, column string) (string, bool) {
	if column == "id" || column == "ID" {
		return record.ID, true
	}
	value, ok := record.Metadata[column]
	if !ok {
		return "", false
	}
	return recordMetaToString(value), true
}

// multiModalGraphCandidates traverses complete paths from relationally valid
// anchors. Only vertices in the final MATCH band are admitted, so intermediate
// graph nodes can never leak into document vector ranking.
func (e *Executor) multiModalGraphCandidates(ctx context.Context, col *Collection, joins []optimizer.GraphJoinPlan, anchors []string) (map[string]struct{}, error) {
	if epoch := epochFromContext(ctx); epoch != nil {
		if tx, err := epoch.GraphTxn(col.name); err == nil {
			return e.multiModalGraphCandidatesEpoch(ctx, col, tx, joins, anchors)
		}
	}
	g := col.GetGraph()
	if g == nil {
		return nil, fmt.Errorf("collection %q has no graph", col.name)
	}
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

	candidates := make(map[string]struct{})
	for _, join := range joins {
		edges := make([]EdgePlan, len(join.GraphEdges))
		for i, gep := range join.GraphEdges {
			edges[i] = graphEdgePlanForTraversal(gep)
		}
		if len(edges) == 0 {
			continue
		}
		returnsSource := join.PredicateMatch
		for _, anchorID := range anchors {
			nodeID, err := e.db.GetNodeID(ctx, col.name, anchorID)
			if err != nil {
				continue
			}
			matched := false
			if err := g.BFSPattern(nodeID, edges, join.MaxHops, func(vertexID uint64, band int, step int) bool {
				if band == len(edges)-1 && !(vertexID == nodeID && step == 0 && edges[0].Min > 0) {
					if returnsSource {
						matched = true
					} else if _, recordID, resolveErr := e.db.ResolveNodeID(ctx, vertexID); resolveErr == nil {
						candidates[recordID] = struct{}{}
					}
				}
				return true
			}, bitset, frontier); err != nil {
				return nil, err
			}
			if matched {
				candidates[anchorID] = struct{}{}
			}
			bitset.Clear()
			frontier.Clear()
		}
	}
	return candidates, nil
}

// multiModalGraphCandidatesEpoch traverses the live graph plus staged edge
// overlays. It intentionally uses ordinary Go queues for correctness first;
// the pooled zero-allocation traversal is a later optimization once epoch
// semantics are stable.
func (e *Executor) multiModalGraphCandidatesEpoch(ctx context.Context, col *Collection, tx interface {
	NeighborsOverlay(uint64) ([]Edge, error)
	InboundNeighborsOverlay(uint64) ([]Edge, error)
}, joins []optimizer.GraphJoinPlan, anchors []string) (map[string]struct{}, error) {
	candidates := make(map[string]struct{})
	for _, join := range joins {
		edges := make([]EdgePlan, len(join.GraphEdges))
		for i, gep := range join.GraphEdges {
			edges[i] = graphEdgePlanForTraversal(gep)
			if gep.EdgeKind != 0 {
				edges[i].KindSet.Set(gep.EdgeKind)
			}
		}
		for _, anchorID := range anchors {
			start, err := e.lookupNodeIDInContext(ctx, col.name, anchorID)
			if err != nil {
				continue
			}
			matched := false
			type state struct {
				node       uint64
				band, step int
			}
			queue := []state{{node: start, band: 0, step: 0}}
			seen := make(map[[3]uint64]struct{})
			for len(queue) > 0 {
				cur := queue[0]
				queue = queue[1:]
				key := [3]uint64{cur.node, uint64(cur.band), uint64(cur.step)}
				if _, ok := seen[key]; ok {
					continue
				}
				seen[key] = struct{}{}
				if cur.band >= len(edges) {
					continue
				}
				band := edges[cur.band]
				if cur.step >= band.Min && cur.band == len(edges)-1 {
					matched = true
				}
				if cur.step >= band.Min && cur.band+1 < len(edges) {
					queue = append(queue, state{node: cur.node, band: cur.band + 1})
				}
				if cur.step >= band.Max {
					continue
				}
				var neighbors []Edge
				if band.Dir < 0 {
					neighbors, err = tx.InboundNeighborsOverlay(cur.node)
				} else {
					neighbors, err = tx.NeighborsOverlay(cur.node)
				}
				if err != nil {
					return nil, err
				}
				for _, edge := range neighbors {
					if band.KindSet != (KindSet{}) && !band.KindSet.Has(edge.GetKind()) {
						continue
					}
					queue = append(queue, state{node: edge.Target, band: cur.band, step: cur.step + 1})
				}
			}
			if matched {
				candidates[anchorID] = struct{}{}
			}
		}
	}
	return candidates, nil
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

	// Inside an epoch, fall back to exact scoring over the merged
	// committed+staged view. The live HNSW does not include staged inserts.
	if epoch := epochFromContext(ctx); epoch != nil {
		return e.executeVectorProjection(ctx, plan)
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

// executeVectorProjection runs a full vector scan for SELECT queries whose
// projection list contains SIMILARITY()/VECTOR_DISTANCE(). Every record's
// stored vector is scored against each vector-func projection's query vector
// via the SIMD-backed util distance functions, then ORDER BY is applied.
func (e *Executor) executeVectorProjection(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	col, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, fmt.Errorf("could not get collection %q: %w", plan.CollectionName, err)
	}
	if col.Dimension() == 0 {
		return nil, fmt.Errorf("collection %q is metadata-only; vector search not available", plan.CollectionName)
	}
	records, err := recordsVisibleInContext(ctx, col)
	if err != nil {
		return nil, err
	}
	out := &SearchResults{}
	for _, rec := range records {
		if len(rec.Vector) == 0 {
			continue
		}
		sr := &SearchResult{ID: rec.ID, Score: 1.0}
		// Compute each vector-func projection's score into metadata.
		if len(plan.VectorFuncProjections) > 0 {
			sr.Metadata = make(map[string]interface{}, len(plan.VectorFuncProjections)+1)
			for _, vfp := range plan.VectorFuncProjections {
				if len(vfp.QueryVector) == 0 || len(vfp.QueryVector) != len(rec.Vector) {
					continue
				}
				sr.Metadata[vfp.Name] = computeVectorScore(col, vfp, rec.Vector)
			}
		}
		out.Results = append(out.Results, sr)
	}
	out.Total = len(out.Results)
	out.Columns = plan.Projections
	if plan.OrderBy != "" {
		e.applyOrderBy(out, plan)
	}
	if plan.Limit > 0 && len(out.Results) > plan.Limit {
		out.Results = out.Results[:plan.Limit]
		out.Total = len(out.Results)
	}
	return out, nil
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
	colName := pred.Column
	// The record ID is addressable as a column too.
	if colName == "id" || colName == "ID" {
		return compareColumn(r.ID, string(pred.Value), pred.Operator)
	}
	if r.Metadata == nil {
		return false
	}
	v, ok := r.Metadata[colName]
	if !ok {
		return false
	}
	var s string
	switch t := v.(type) {
	case string:
		s = t
	case []byte:
		s = string(t)
	case int:
		s = fmt.Sprintf("%d", t)
	case int64:
		s = fmt.Sprintf("%d", t)
	case uint64:
		s = fmt.Sprintf("%d", t)
	case float64:
		s = strconv.FormatFloat(t, 'f', -1, 64)
	case float32:
		s = strconv.FormatFloat(float64(t), 'f', -1, 32)
	case bool:
		s = fmt.Sprintf("%t", t)
	default:
		s = fmt.Sprintf("%v", t)
	}
	return compareColumn(s, string(pred.Value), pred.Operator)
}

// compareColumn compares a column value with a literal, coercing both sides
// to numbers when both parse as numbers so "10" > "9" is numeric, not lexical.
func compareColumn(colVal, lit string, op uint8) bool {
	if cf, cok := strconv.ParseFloat(colVal, 64); cok == nil {
		if lf, lok := strconv.ParseFloat(lit, 64); lok == nil {
			switch op {
			case 12: // KindEquals
				return cf == lf
			case 13: // KindGreaterThan
				return cf > lf
			case 14: // KindLessThan
				return cf < lf
			}
		}
	}
	switch op {
	case 12: // KindEquals
		return colVal == lit
	case 13: // KindGreaterThan
		return colVal > lit
	case 14: // KindLessThan
		return colVal < lit
	}
	return true // unknown operator → include
}

// executeGraphEpoch performs BFS graph traversal through the epoch overlay.
// Each neighbor expansion uses NeighborsOverlay/InboundNeighborsOverlay so
// staged edges are visible and concurrent commits at higher LSNs are invisible.
func (e *Executor) executeGraphEpoch(ctx context.Context, plan *optimizer.PhysicalPlan, gtx *graph.Txn, col *Collection) (*SearchResults, error) {
	if len(plan.GraphEdges) == 0 {
		return &SearchResults{}, nil
	}
	// Build edge plans from the optimizer's graph edge descriptors.
	type bfsEdge struct {
		dir  int8
		kind uint8
		qmin uint16
		qmax uint16
	}
	var edges []bfsEdge
	for _, gep := range plan.GraphEdges {
		edges = append(edges, bfsEdge{
			dir:  gep.Direction,
			kind: gep.EdgeKind,
			qmin: gep.QuantMin,
			qmax: gep.QuantMax,
		})
	}
	if len(edges) == 0 {
		return &SearchResults{}, nil
	}

	// Collect seeds (same priority cascade as executeGraph).
	var seeds []uint64
	if plan.HasExplicitSeed {
		seeds = append(seeds, plan.ExplicitSeedID)
	}
	if plan.SeedLabel != "" {
		g := col.GetGraph()
		if g != nil {
			seeds = append(seeds, g.GetLabelNodes(plan.SeedLabel)...)
		}
	}
	if len(seeds) == 0 {
		records, err := recordsVisibleInContext(ctx, col)
		if err != nil {
			return nil, err
		}
		for _, rec := range records {
			nid, err := e.lookupNodeIDInContext(ctx, col.name, rec.ID)
			if err == nil {
				seeds = append(seeds, nid)
			}
		}
	}
	if len(seeds) == 0 {
		return &SearchResults{}, nil
	}

	returnsSource := len(plan.GraphJoins) > 0 && plan.GraphJoins[0].PredicateMatch
	seen := make(map[uint64]bool)
	seedMatched := make(map[uint64]bool)

	for _, seed := range seeds {
		type bfsNode struct {
			nodeID uint64
			band   int
			step   int
		}
		queue := []bfsNode{{nodeID: seed, band: 0, step: 0}}
		type visitKey struct {
			nodeID uint64
			band   int
			step   int
		}
		visited := make(map[visitKey]bool)

		for len(queue) > 0 {
			cur := queue[0]
			queue = queue[1:]

			key := visitKey{cur.nodeID, cur.band, cur.step}
			if visited[key] {
				continue
			}
			visited[key] = true

			if cur.band >= len(edges) {
				seen[cur.nodeID] = true
				continue
			}

			band := edges[cur.band]
			var neighbors []Edge
			var err error
			if band.dir < 0 {
				neighbors, err = gtx.InboundNeighborsOverlay(cur.nodeID)
			} else {
				neighbors, err = gtx.NeighborsOverlay(cur.nodeID)
			}
			if err != nil {
				continue
			}

			for _, n := range neighbors {
				if band.kind != 0 && n.GetKind() != band.kind {
					continue
				}
				nextStep := cur.step + 1
				nextBand := cur.band
				if cur.band < len(edges)-1 && nextStep >= int(band.qmax) && band.qmax > 0 && band.qmax != 0xFFFF {
					nextBand = cur.band + 1
					nextStep = 0
				}
				queue = append(queue, bfsNode{nodeID: n.Target, band: nextBand, step: nextStep})
			}

			if returnsSource && cur.step >= int(band.qmin) {
				seedMatched[seed] = true
			}
		}
	}

	candidates := seen
	if returnsSource {
		candidates = seedMatched
	}

	// Build results (shared with the non-epoch path below).
	return e.buildGraphResultsFromCandidates(ctx, plan, candidates, col)
}

// resolveNodeIDInContext resolves a graph node ID to a (collection, recordID) pair.
// Inside an epoch, provisional node IDs are resolved from the epoch's local mapping.
func (e *Executor) resolveNodeIDInContext(ctx context.Context, nodeID uint64) (string, string, error) {
	if epoch := epochFromContext(ctx); epoch != nil {
		return epoch.ResolveNodeID(ctx, nodeID)
	}
	return e.db.ResolveNodeID(ctx, nodeID)
}

// lookupNodeIDInContext resolves a (collection, recordID) pair to a graph node ID,
// using the epoch's provisional mapping when available.
func (e *Executor) lookupNodeIDInContext(ctx context.Context, collection, id string) (uint64, error) {
	if epoch := epochFromContext(ctx); epoch != nil {
		return epoch.LookupNodeID(ctx, collection, id)
	}
	return e.db.GetNodeID(ctx, collection, id)
}

// neighborsInContext returns the outbound neighbors for a node, using the
// epoch overlay when present and falling back to live graph otherwise.
func (e *Executor) neighborsInContext(ctx context.Context, gtx *graph.Txn, nodeID uint64) ([]graph.Edge, error) {
	if gtx != nil {
		return gtx.NeighborsOverlay(nodeID)
	}
	if epoch := epochFromContext(ctx); epoch != nil {
		// Try to get the graph txn for the first epoch collection.
		for _, colName := range epoch.graphNames() {
			if txn, err := epoch.GraphTxn(colName); err == nil {
				return txn.NeighborsOverlay(nodeID)
			}
		}
	}
	// Fallback: find any graph-enabled collection.
	col := e.db.firstGraphCollection()
	if col != nil {
		g := col.GetGraph()
		if g != nil {
			return g.Neighbors(nodeID)
		}
	}
	return nil, nil
}

// inboundNeighborsInContext returns the inbound neighbors for a node,
// using the epoch overlay when present and falling back to live graph.
func (e *Executor) inboundNeighborsInContext(ctx context.Context, gtx *graph.Txn, nodeID uint64) ([]graph.Edge, error) {
	if gtx != nil {
		return gtx.InboundNeighborsOverlay(nodeID)
	}
	if epoch := epochFromContext(ctx); epoch != nil {
		for _, colName := range epoch.graphNames() {
			if txn, err := epoch.GraphTxn(colName); err == nil {
				return txn.InboundNeighborsOverlay(nodeID)
			}
		}
	}
	col := e.db.firstGraphCollection()
	if col != nil {
		g := col.GetGraph()
		if g != nil {
			return g.InboundNeighbors(nodeID)
		}
	}
	return nil, nil
}

// graphNames returns the collection names for which graph transactions exist.
func (e *EpochTx) graphNames() []string {
	e.mu.Lock()
	defer e.mu.Unlock()
	names := make([]string, 0, len(e.graphs))
	for n := range e.graphs {
		names = append(names, n)
	}
	return names
}

// buildGraphResultsFromCandidates resolves candidate node IDs to record IDs,
// optionally scores by vector distance, sorts, and applies LIMIT.
func (e *Executor) buildGraphResultsFromCandidates(ctx context.Context, plan *optimizer.PhysicalPlan, candidates map[uint64]bool, col *Collection) (*SearchResults, error) {
	// Vector-scored graph results.
	if plan.HasVectorSearch && len(plan.QueryVector) > 0 && len(candidates) > 0 {
		records, err := recordsVisibleInContext(ctx, col)
		if err != nil {
			return nil, err
		}
		recordMap := make(map[string]Record, len(records))
		for _, rec := range records {
			recordMap[rec.ID] = rec
		}
		type recScore struct {
			id    string
			score float32
		}
		var scoredList []recScore
		for nodeID := range candidates {
			_, recID, err := e.resolveNodeIDInContext(ctx, nodeID)
			if err != nil {
				continue
			}
			rec, ok := recordMap[recID]
			if !ok || len(rec.Vector) == 0 {
				continue
			}
			var dist float32
			switch col.config.Metric {
			case L2Distance:
				dist = util.L2Distance_func(plan.QueryVector, rec.Vector)
			case InnerProduct:
				dist = util.InnerProduct_func(plan.QueryVector, rec.Vector)
			case CosineDistance:
				dist = util.CosineDistance_func(plan.QueryVector, rec.Vector)
			default:
				dist = util.L2Distance_func(plan.QueryVector, rec.Vector)
			}
			scoredList = append(scoredList, recScore{id: recID, score: dist})
		}
		sort.Slice(scoredList, func(i, j int) bool {
			if plan.IsDesc {
				return scoredList[i].score > scoredList[j].score
			}
			return scoredList[i].score < scoredList[j].score
		})
		if plan.Limit > 0 && len(scoredList) > plan.Limit {
			scoredList = scoredList[:plan.Limit]
		}
		results := &SearchResults{}
		for _, s := range scoredList {
			results.Results = append(results.Results, &SearchResult{ID: s.id, Score: s.score})
		}
		results.Total = len(results.Results)
		return results, nil
	}

	// Non-scored: just resolve node IDs to record IDs.
	results := &SearchResults{}
	for nodeID := range candidates {
		_, recID, err := e.resolveNodeIDInContext(ctx, nodeID)
		if err != nil {
			continue
		}
		results.Results = append(results.Results, &SearchResult{ID: recID, Score: 1.0})
		if plan.Limit > 0 && len(results.Results) >= plan.Limit {
			break
		}
	}
	results.Total = len(results.Results)
	return results, nil
}

// executeGraph performs direction-aware graph traversal using BFSPattern.
// It always completes the bounded MATCH traversal before applying plan.Limit
// to output rows. This preserves a complete candidate set for any subsequent
// vector-ranking composition; traversal order is never a ranking order.
// Seeds are selected by a three-way priority cascade:
//  1. Explicit seed (WHERE a.id = N) — validated via ResolveNodeID
//  2. Vector-anchored (WHERE SIMILARITY(...) + GRAPH_TABLE) — using SearchWithGraphFilter
//  3. Label-scan — NOT YET SUPPORTED (returns error)
func (e *Executor) executeGraph(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	// Standalone SELECT * FROM MATCH has no explicit collection name.
	// Resolve to the first graph-enabled collection at execution time.
	if plan.CollectionName == "" {
		col := e.db.firstGraphCollection()
		if col == nil {
			return nil, fmt.Errorf("no graph-enabled collection found for implicit MATCH source")
		}
		plan.CollectionName = col.name
	}

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

	// Priority 3: label-scan seeding
	if len(seeds) == 0 && plan.SeedLabel != "" {
		col, err := e.db.GetCollection(plan.CollectionName)
		if err != nil {
			return nil, fmt.Errorf("label-scan seed: %w", err)
		}
		g := col.GetGraph()
		if g != nil {
			seeds = g.GetLabelNodes(plan.SeedLabel)
		}
	}
	// Priority 4: source-row seeding — iterate all visible records
	// and use their graph node IDs as seeds. Supports WHERE MATCH
	// without explicit seed, anchor, or label.
	if len(seeds) == 0 {
		col, err := e.db.GetCollection(plan.CollectionName)
		if err != nil {
			return nil, err
		}
		records, err := recordsVisibleInContext(ctx, col)
		if err != nil {
			return nil, fmt.Errorf("source-row seed: %w", err)
		}
		// Partition predicates: terminal-alias predicates validate
		// reached vertices during BFS; source predicates filter seeds.
		var terminalAlias string
		var sourcePreds []optimizer.RelationalPredicate
		if len(plan.GraphJoins) > 0 {
			terminalAlias = plan.GraphJoins[0].TerminalAlias
		}
		if plan.HasRelationalQuery {
			for _, p := range plan.Predicates {
				if p.Alias == "" || p.Alias != terminalAlias {
					sourcePreds = append(sourcePreds, p)
				}
			}
		}
		for _, rec := range records {
			if len(sourcePreds) > 0 && !recordMatchesPredicates(rec, sourcePreds) {
				continue
			}
			nodeID, err := e.db.GetNodeID(ctx, plan.CollectionName, rec.ID)
			if err != nil {
				continue
			}
			seeds = append(seeds, nodeID)
		}
	}
	if len(seeds) == 0 {
		return nil, errors.New(
			"graph query requires either WHERE a.id = N (explicit seed), " +
				"a vector predicate (vector-anchored traversal), " +
				"a labeled start vertex (label-scan seeding), " +
				"or seeded from visible source rows")
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

	// Inside an epoch, route graph traversal through the overlay so
	// staged edges are visible and concurrent commits are invisible.
	if epoch := epochFromContext(ctx); epoch != nil {
		gtx, err := epoch.GraphTxn(plan.CollectionName)
		if err != nil {
			return nil, fmt.Errorf("epoch graph txn: %w", err)
		}
		return e.executeGraphEpoch(ctx, plan, gtx, col)
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
		edges[i] = graphEdgePlanForTraversal(gep)
		totalMinDepth += edges[i].Min
	}

	// Determine whether to emit source rows (WHERE MATCH) or terminal
	// vertices (GRAPH_TABLE / explicit seeds).
	returnsSource := false
	if len(plan.GraphJoins) > 0 && plan.GraphJoins[0].PredicateMatch {
		returnsSource = true
	}

	// BFS from each seed, tracking visited nodes (band-stateful traversal).
	seen := make(map[uint64]bool)
	seedMatched := make(map[uint64]bool)

	// Prepare terminal label set and predicates for validation.
	var terminalLabelNodes map[uint64]bool
	var terminalPredicates []optimizer.RelationalPredicate
	if returnsSource && len(plan.GraphJoins) > 0 {
		join := plan.GraphJoins[0]
		if join.TerminalLabel != "" {
			terminalLabelNodes = make(map[uint64]bool)
			for _, nid := range g.GetLabelNodes(join.TerminalLabel) {
				terminalLabelNodes[nid] = true
			}
		}
		terminalPredicates = join.TerminalPredicates
	}

	for _, seed := range seeds {
		matched := false
		if err := g.BFSPattern(seed, edges, plan.MaxHops, func(nodeID uint64, band int, step int) bool {
			if returnsSource {
				if band == len(edges)-1 && !(nodeID == seed && step == 0 && edges[0].Min > 0) {
					// Validate terminal label if required.
					if terminalLabelNodes != nil && !terminalLabelNodes[nodeID] {
						return true
					}
					// Validate terminal predicates if required.
					if len(terminalPredicates) > 0 {
						colName, recID, err := e.resolveNodeIDInContext(ctx, nodeID)
						if err != nil {
							return true
						}
						col, cerr := e.db.GetCollection(colName)
						if cerr != nil {
							return true
						}
						rec, gerr := col.Get(ctx, recID)
						if gerr != nil {
							return true
						}
						if !recordMatchesPredicates(rec, terminalPredicates) {
							return true
						}
					}
					matched = true
				}
			} else {
				// Exclude the initial seed (band=0, step=0) — it
				// has not traversed any edge and should not appear
				// in traversal results. Band-transition nodes at
				// step=0 are valid traversal endpoints.
				if step > 0 || band > 0 {
					seen[nodeID] = true
				}
			}
			return true
		}, bitset, frontier); err != nil {
			return nil, err
		}
		if returnsSource && matched {
			seedMatched[seed] = true
		}

		bitset.Clear()
		frontier.Clear()
	}

	// Collect candidate node IDs for projection.
	candidates := seen
	if returnsSource {
		candidates = seedMatched
	}

	// Vector-scored graph results: when ORDER BY VECTOR_DISTANCE is present,
	// score each MATCH candidate by vector distance, sort, and apply LIMIT.
	if plan.HasVectorSearch && len(plan.QueryVector) > 0 && len(candidates) > 0 {
		col, err := e.db.GetCollection(plan.CollectionName)
		if err != nil {
			return nil, err
		}
		records, err := recordsVisibleInContext(ctx, col)
		if err != nil {
			return nil, err
		}
		recordMap := make(map[string]Record, len(records))
		for _, rec := range records {
			recordMap[rec.ID] = rec
		}

		type recScore struct {
			id    string
			score float32
		}
		var scoredList []recScore
		for nodeID := range candidates {
			_, recID, err := e.resolveNodeIDInContext(ctx, nodeID)
			if err != nil {
				continue
			}
			rec, ok := recordMap[recID]
			if !ok || len(rec.Vector) == 0 {
				continue
			}
			var dist float32
			switch col.config.Metric {
			case L2Distance:
				dist = util.L2Distance_func(plan.QueryVector, rec.Vector)
			case InnerProduct:
				dist = util.InnerProduct_func(plan.QueryVector, rec.Vector)
			case CosineDistance:
				dist = util.CosineDistance_func(plan.QueryVector, rec.Vector)
			default:
				dist = util.L2Distance_func(plan.QueryVector, rec.Vector)
			}
			scoredList = append(scoredList, recScore{id: recID, score: dist})
		}
		sort.Slice(scoredList, func(i, j int) bool {
			if plan.IsDesc {
				return scoredList[i].score > scoredList[j].score
			}
			return scoredList[i].score < scoredList[j].score
		})
		if plan.Limit > 0 && len(scoredList) > plan.Limit {
			scoredList = scoredList[:plan.Limit]
		}

		results := &SearchResults{}
		for _, s := range scoredList {
			results.Results = append(results.Results, &SearchResult{
				ID: s.id, Score: s.score,
			})
		}
		results.Total = len(results.Results)
		return results, nil
	}

	// Project GraphNodeIDs to SearchResults via ResolveNodeID.
	// For WHERE MATCH (PredicateMatch), emit source row IDs rather
	// than terminal vertices.
	results := &SearchResults{}
	if returnsSource {
		for nodeID := range seedMatched {
			_, recID, err := e.resolveNodeIDInContext(ctx, nodeID)
			if err != nil {
				continue
			}
			results.Results = append(results.Results, &SearchResult{
				ID: recID, Score: 1.0,
			})
			if plan.Limit > 0 && len(results.Results) >= plan.Limit {
				break
			}
		}
	} else {
		for nodeID := range seen {
			_, recID, err := e.resolveNodeIDInContext(ctx, nodeID)
			if err != nil {
				continue
			}
			results.Results = append(results.Results, &SearchResult{
				ID: recID, Score: 1.0,
			})
			if plan.Limit > 0 && len(results.Results) >= plan.Limit {
				break
			}
		}
	}
	results.Total = len(results.Results)
	return results, nil
}

// executeRelational handles exact-match, range, and full-scan queries against a B-tree index.
// When an epoch is active, routes through the merged committed+staged record view instead
// of the live B-tree, which does not include staged inserts.
func (e *Executor) executeRelational(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	col, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
	}

	// Inside an epoch, use the merged record overlay (committed + staged).
	if epoch := epochFromContext(ctx); epoch != nil {
		return e.executeRelationalFullScan(ctx, col, plan)
	}

	idx := col.GetIndex()
	if idx == nil {
		return nil, fmt.Errorf("collection %q has no index", plan.CollectionName)
	}

	tree, ok := idx.(interface{ Tree() *btree.BTree })
	if !ok {
		return e.executeRelationalFullScan(ctx, col, plan)
	}

	// If there's an exact-match predicate, use B-tree Search directly
	if len(plan.Predicates) == 1 && plan.Predicates[0].Operator == 12 &&
		strings.EqualFold(plan.Predicates[0].Column, "id") { // KindEquals on physical key
		pred := plan.Predicates[0]
		val, err := tree.Tree().Search(ctx, pred.Value)
		if err == nil {
			ord, ver, _ := btree.DecodeValue(val)
			return e.buildSelectResult(ctx, col, &SearchResult{ID: string(pred.Value), Version: uint64(ver), Score: 1.0, Ordinal: ord}, plan), nil
		}
		return &SearchResults{}, nil
	}
	for _, pred := range plan.Predicates {
		if !strings.EqualFold(pred.Column, "id") {
			// The B-tree is keyed by physical record ID, not metadata. Use
			// the predicate-aware full scan for metadata conditions.
			return e.executeRelationalFullScan(ctx, col, plan)
		}
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

	return e.buildSelectResults(ctx, col, results, plan), nil
}

// buildSelectResult enriches a single search result with the record's metadata
// projected to the plan's column list.
func (e *Executor) buildSelectResult(ctx context.Context, col *Collection, sr *SearchResult, plan *optimizer.PhysicalPlan) *SearchResults {
	results := &SearchResults{}
	if sr == nil {
		return results
	}
	sr = e.attachMetadata(ctx, col, sr, plan)
	results.Results = []*SearchResult{sr}
	results.Total = 1
	results.Columns = plan.Projections
	return results
}

// buildSelectResults enriches a batch of search results with record metadata
// projected to the plan's column list, then applies ORDER BY if requested.
func (e *Executor) buildSelectResults(ctx context.Context, col *Collection, results []*SearchResult, plan *optimizer.PhysicalPlan) *SearchResults {
	out := &SearchResults{}
	if len(results) == 0 {
		out.Columns = plan.Projections
		return out
	}
	for _, sr := range results {
		out.Results = append(out.Results, e.attachMetadata(ctx, col, sr, plan))
	}
	out.Total = len(out.Results)
	out.Columns = plan.Projections
	if plan.OrderBy != "" {
		e.applyOrderBy(out, plan)
	}
	return out
}

// applyOrderBy sorts results by the ORDER BY column's value in the result
// metadata. Numeric-looking values sort numerically; everything else sorts
// lexically. IsDesc reverses the order.
func (e *Executor) applyOrderBy(out *SearchResults, plan *optimizer.PhysicalPlan) {
	colName := plan.OrderBy
	less := func(a, b *SearchResult) bool {
		av, aok := a.Metadata[colName]
		bv, bok := b.Metadata[colName]
		if !aok && !bok {
			return a.ID < b.ID
		}
		if !aok {
			return true
		}
		if !bok {
			return false
		}
		// Numeric comparison when both parse as float64.
		af, aIsNum := toFloat(av)
		bf, bIsNum := toFloat(bv)
		if aIsNum && bIsNum {
			if af != bf {
				return af < bf
			}
			// Tie: break deterministically by ID.
			return a.ID < b.ID
		}
		as, bs := fmt.Sprint(av), fmt.Sprint(bv)
		if as != bs {
			return as < bs
		}
		return a.ID < b.ID
	}
	if plan.IsDesc {
		// Reverse via sort.Slice with inverted comparator.
		sort.Slice(out.Results, func(i, j int) bool {
			return less(out.Results[j], out.Results[i])
		})
		return
	}
	sort.Slice(out.Results, func(i, j int) bool {
		return less(out.Results[i], out.Results[j])
	})
}

// toFloat attempts to convert a metadata value to float64 for numeric ordering.
func toFloat(v interface{}) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case string:
		f, err := strconv.ParseFloat(strings.TrimSpace(n), 64)
		if err != nil {
			return 0, false
		}
		return f, true
	default:
		return 0, false
	}
}

// attachMetadata loads the full record for a result ID and projects its
// metadata down to the plan's column list. The "id" column is always
// available from the result itself. Vector-func projections
// (SIMILARITY/VECTOR_DISTANCE) are computed from the record's stored
// vector against the plan's query vector.
func (e *Executor) attachMetadata(ctx context.Context, col *Collection, sr *SearchResult, plan *optimizer.PhysicalPlan) *SearchResult {
	rec, err := col.Get(ctx, sr.ID)
	if err != nil || rec.Metadata == nil {
		return sr
	}
	// Vector-func projections need the record's stored vector. Compute them
	// before projecting so they land in the output metadata.
	if len(plan.VectorFuncProjections) > 0 && len(rec.Vector) > 0 {
		for _, vfp := range plan.VectorFuncProjections {
			if len(vfp.QueryVector) == 0 || len(vfp.QueryVector) != len(rec.Vector) {
				continue
			}
			score := computeVectorScore(col, vfp, rec.Vector)
			if sr.Metadata == nil {
				sr.Metadata = make(map[string]interface{}, len(plan.VectorFuncProjections))
			}
			sr.Metadata[vfp.Name] = score
		}
	}
	if len(plan.Projections) == 0 {
		// All columns: expose every metadata field.
		sr.Metadata = rec.Metadata
		return sr
	}
	// Projected columns: keep only what was selected, in order.
	proj := make(map[string]interface{}, len(plan.Projections))
	for _, colName := range plan.Projections {
		if colName == "id" || colName == "ID" {
			proj[colName] = sr.ID
			continue
		}
		if value, ok := sr.Metadata[colName]; ok {
			proj[colName] = value
			continue
		}
		if v, ok := rec.Metadata[colName]; ok {
			proj[colName] = v
		}
	}
	sr.Metadata = proj
	return sr
}

// computeVectorScore computes the SIMILARITY or VECTOR_DISTANCE score for a
// single record vector against a vector-func projection's query vector,
// using the collection's configured distance metric. It dispatches through
// the same SIMD-backed util functions the index uses, so it inherits the
// AVX2 assembly on amd64 and NEON on arm64.
func computeVectorScore(col *Collection, vfp optimizer.VectorFuncProjection, recVector []float32) float32 {
	var score float32
	switch col.config.Metric {
	case L2Distance:
		score = util.L2Distance_func(vfp.QueryVector, recVector)
	case InnerProduct:
		score = util.InnerProduct_func(vfp.QueryVector, recVector)
	case CosineDistance:
		score = util.CosineDistance_func(vfp.QueryVector, recVector)
	default:
		score = util.CosineDistance_func(vfp.QueryVector, recVector)
	}
	if vfp.IsDistance {
		return score
	}
	// SIMILARITY = 1 - distance (cosine distance is 1 - cosine sim).
	return 1 - score
}

// compositePrimaryKeyColumns resolves the configured primary-key columns. A
// freshly-created collection carries the ordered declaration in its config;
// after reopen, recover the same set from catalog ColumnDef flags. The
// composite key encoder below canonicalizes by column name, so catalog column
// order remains safe even when declaration order was not persisted separately.
func (e *Executor) compositePrimaryKeyColumns(collection string, available []string) []string {
	if col, err := e.db.GetCollection(collection); err == nil {
		if cfg := col.Config(); len(cfg.PrimaryKeyColumns) > 0 {
			return append([]string(nil), cfg.PrimaryKeyColumns...)
		}
	}
	e.db.mu.RLock()
	cat := e.db.catalog
	e.db.mu.RUnlock()
	if cat == nil {
		return nil
	}
	hashes, err := cat.PrimaryKeyColumnHashes(catalog.HashIdentifier(collection))
	if err != nil || len(hashes) == 0 {
		return nil
	}
	result := make([]string, 0, len(hashes))
	for _, hash := range hashes {
		for _, name := range available {
			if catalog.HashIdentifier(name) == hash {
				result = append(result, name)
				break
			}
		}
	}
	if len(result) != len(hashes) {
		return nil
	}
	return result
}

func (e *Executor) isPrimaryKeyColumn(collection, column string) bool {
	if col, err := e.db.GetCollection(collection); err == nil {
		for _, name := range col.Config().PrimaryKeyColumns {
			if strings.EqualFold(name, column) {
				return true
			}
		}
	}
	e.db.mu.RLock()
	cat := e.db.catalog
	e.db.mu.RUnlock()
	if cat == nil {
		return false
	}
	hashes, err := cat.PrimaryKeyColumnHashes(catalog.HashIdentifier(collection))
	if err != nil {
		return false
	}
	columnHash := catalog.HashIdentifier(column)
	for _, hash := range hashes {
		if hash == columnHash {
			return true
		}
	}
	return false
}

// encodeCompositePrimaryKey creates a collision-free physical record key.
// Components are sorted by case-insensitive column name so the key remains
// stable after reopen even though the compact catalog stores column flags but
// not a separate declaration-order array.
func encodeCompositePrimaryKey(columns []string, values map[string]string) (string, error) {
	type component struct{ name, value string }
	parts := make([]component, 0, len(columns))
	for _, column := range columns {
		value, ok := values[strings.ToLower(column)]
		if !ok || value == "" {
			return "", fmt.Errorf("missing value for PRIMARY KEY column %q", column)
		}
		parts = append(parts, component{name: strings.ToLower(column), value: value})
	}
	sort.Slice(parts, func(i, j int) bool { return parts[i].name < parts[j].name })
	var b strings.Builder
	b.WriteString("__pk:")
	for _, part := range parts {
		b.WriteString(strconv.Itoa(len(part.name)))
		b.WriteByte(':')
		b.WriteString(part.name)
		b.WriteString(strconv.Itoa(len(part.value)))
		b.WriteByte(':')
		b.WriteString(part.value)
		b.WriteByte('|')
	}
	return b.String(), nil
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
	pkColumns := e.compositePrimaryKeyColumns(plan.CollectionName, plan.InsertColumns)
	// Preserve the longstanding physical-id behavior for a single-column
	// PRIMARY KEY on id. Composite keys (and single non-id keys) use the
	// canonical derived key path below.
	if len(pkColumns) == 1 && strings.EqualFold(pkColumns[0], "id") {
		pkColumns = nil
	}

	// Group flat values into rows
	var entries []VectorEntry
	for i := 0; i < len(plan.InsertValues); i += colCount {
		var id string
		var vec []float32
		meta := make(map[string]interface{})
		rowValues := make(map[string]string, colCount)
		for j := 0; j < colCount && i+j < len(plan.InsertValues); j++ {
			val := string(plan.InsertValues[i+j])
			if colCount > 0 && j < len(plan.InsertColumns) {
				colName := plan.InsertColumns[j]
				rowValues[strings.ToLower(colName)] = val
				if strings.EqualFold(colName, "id") {
					id = val
				} else if strings.EqualFold(colName, "vector") || strings.EqualFold(colName, "vec") || strings.EqualFold(colName, "embedding") {
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
		if len(pkColumns) > 0 {
			if _, suppliedID := rowValues["id"]; suppliedID {
				isPKID := false
				for _, pkColumn := range pkColumns {
					if strings.EqualFold(pkColumn, "id") {
						isPKID = true
						break
					}
				}
				if !isPKID {
					return nil, fmt.Errorf("do not supply physical id when using composite PRIMARY KEY; provide the declared key columns")
				}
			}
			var keyErr error
			id, keyErr = encodeCompositePrimaryKey(pkColumns, rowValues)
			if keyErr != nil {
				return nil, keyErr
			}
		} else if id == "" {
			return nil, fmt.Errorf("INSERT requires an 'id' column")
		}
		entries = append(entries, VectorEntry{ID: id, Vector: vec, Metadata: meta})
	}

	if len(entries) == 0 {
		return &SearchResults{}, nil
	}

	// If inside an epoch, stage through the epoch's record transaction.
	if epoch := epochFromContext(ctx); epoch != nil {
		for _, entry := range entries {
			if err := epoch.Insert(ctx, plan.CollectionName, entry.ID, entry.Vector, entry.Metadata); err != nil {
				return nil, fmt.Errorf("epoch insert %q: %w", entry.ID, err)
			}
		}
		return &SearchResults{Total: len(entries)}, nil
	}

	if err := col.InsertBatch(ctx, entries); err != nil {
		return nil, err
	}
	return &SearchResults{Total: len(entries)}, nil
}

// executeInsertGraphEdge handles INSERT INTO GRAPH_EDGES VALUES (src, kind, tgt).
// When called within an epoch transaction, edges are staged into the epoch's
// graph transaction. Otherwise, a direct graph transaction is created, committed,
// and the edge is immediately published.
func (e *Executor) executeInsertGraphEdge(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	if len(plan.InsertValues) < 3 {
		return nil, fmt.Errorf("INSERT INTO GRAPH_EDGES requires 3 values (src, kind, tgt)")
	}

	srcID := string(plan.InsertValues[0])
	kindName := string(plan.InsertValues[1])
	tgtID := string(plan.InsertValues[2])

	kind := graph.ResolveEdgeKind(kindName)
	if kind == 0 && kindName != "" {
		return nil, fmt.Errorf("unknown edge kind %q", kindName)
	}

	// Find the first collection that has a graph for node ID resolution.
	col := e.db.firstGraphCollection()
	if col == nil {
		return nil, fmt.Errorf("no collection with a graph found for edge insert")
	}

	// Resolve record IDs to graph node IDs.
	// Inside an epoch, check provisional IDs for staged records first.
	var srcNode, tgtNode uint64
	var err error
	if epoch := epochFromContext(ctx); epoch != nil {
		srcNode, err = epoch.LookupNodeID(ctx, col.name, srcID)
		if err != nil {
			return nil, fmt.Errorf("resolving source node %q: %w", srcID, err)
		}
		tgtNode, err = epoch.LookupNodeID(ctx, col.name, tgtID)
		if err != nil {
			return nil, fmt.Errorf("resolving target node %q: %w", tgtID, err)
		}
	} else {
		srcNode, err = col.LookupNodeID(ctx, srcID)
		if err != nil {
			return nil, fmt.Errorf("resolving source node %q: %w", srcID, err)
		}
		tgtNode, err = col.LookupNodeID(ctx, tgtID)
		if err != nil {
			return nil, fmt.Errorf("resolving target node %q: %w", tgtID, err)
		}
	}

	g := col.GetGraph()
	if g == nil {
		return nil, fmt.Errorf("collection %q has no graph", col.name)
	}

	// If we are inside an epoch, stage the edge via EpochTx.AddGraphEdge.
	// This increments generation and routes through the ordered operation log.
	// Direct gtx.AddEdge() bypasses generation accounting and ordered logging.
	if epoch := epochFromContext(ctx); epoch != nil {
		if err := epoch.AddGraphEdge(col.name, srcNode, tgtNode, 1.0, kind); err != nil {
			return nil, fmt.Errorf("staging graph edge: %w", err)
		}
		return &SearchResults{Total: 1}, nil
	}

	// Direct path: stage via txn and commit immediately.
	txn := g.BeginTxn()
	if err := txn.AddEdge(srcNode, tgtNode, 1.0, kind); err != nil {
		txn.Rollback()
		return nil, fmt.Errorf("staging graph edge: %w", err)
	}
	if err := txn.Commit(ctx); err != nil {
		return nil, fmt.Errorf("committing graph edge: %w", err)
	}
	return &SearchResults{Total: 1}, nil
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

	colName := aggregateColumnName(plan.AggregateFunc)
	metaValue := aggregateMetaValue(plan.AggregateFunc, count, sum, minVal, maxVal, resultValue)
	return &SearchResults{
		Results: []*SearchResult{{
			ID:       resultValue,
			Score:    1.0,
			Metadata: map[string]interface{}{colName: metaValue},
		}},
		Total:   1,
		Columns: []string{colName},
	}, nil
}

// sqlTypeToFieldType maps SQL column types to metadata FieldTypes for schema
// registration. Returns ok=false for types without a metadata equivalent.
func sqlTypeToFieldType(sqlType string) (FieldType, bool) {
	switch strings.ToUpper(strings.TrimSpace(sqlType)) {
	case "INT", "INTEGER", "BIGINT", "SMALLINT", "SERIAL":
		return IntField, true
	case "TEXT", "VARCHAR", "CHAR", "STRING", "UUID":
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
		var vectorCount int
		primaryKeyColumns := append([]string(nil), plan.DDLPrimaryKeyColumns...)
		columnConstraints := map[string]uint16{
			"id": catalog.ColFlagPrimaryKey | catalog.ColFlagNotNull,
		}
		for _, col := range plan.DDLColumns {
			if col.VectorDimension > 0 {
				vectorCount++
				if vectorCount > 1 {
					return nil, fmt.Errorf(
						"multiple VECTOR columns in table %q; only one vector column per collection is supported",
						plan.DDLTableName)
				}
				opts = []CollectionOption{WithDimension(int(col.VectorDimension))}
				continue
			}
			// Reject bare VECTOR without a dimension.
			if col.Type == "VECTOR" || col.Type == "vector" {
				return nil, fmt.Errorf(
					"VECTOR column %q requires a dimension, e.g. VECTOR(768)", col.Name)
			}
			// Collect PRIMARY KEY columns for key derivation at insert time.
			// Column-level PRIMARY KEY is allowed on any column; the internal
			// record key is derived from the declared PK at insert time.
			if col.Flags&catalog.ColFlagPrimaryKey != 0 && col.Name != "id" {
				plan.DDLPrimaryKeyColumns = append(plan.DDLPrimaryKeyColumns, col.Name)
			}
			// Collect column constraints for catalog persistence.
			if col.Flags != 0 {
				columnConstraints[col.Name] = col.Flags
			}
			if schema == nil {
				schema = make(MetadataSchema)
			}
			if ft, ok := sqlTypeToFieldType(col.Type); ok {
				schema[col.Name] = ft
			}
		}
		if len(primaryKeyColumns) > 0 {
			seenPK := make(map[string]struct{}, len(primaryKeyColumns))
			for _, pkName := range primaryKeyColumns {
				if _, duplicate := seenPK[strings.ToLower(pkName)]; duplicate {
					return nil, fmt.Errorf("duplicate column %q in PRIMARY KEY", pkName)
				}
				seenPK[strings.ToLower(pkName)] = struct{}{}
				found := false
				for _, col := range plan.DDLColumns {
					if strings.EqualFold(col.Name, pkName) {
						if col.VectorDimension > 0 || strings.EqualFold(col.Type, "VECTOR") {
							return nil, fmt.Errorf("VECTOR column %q cannot be part of PRIMARY KEY", pkName)
						}
						columnConstraints[col.Name] |= catalog.ColFlagPrimaryKey | catalog.ColFlagNotNull
						found = true
						break
					}
				}
				if !found {
					return nil, fmt.Errorf("PRIMARY KEY column %q does not exist in table %q", pkName, plan.DDLTableName)
				}
			}
			opts = append(opts, WithPrimaryKeyColumns(primaryKeyColumns...))
		}
		if len(schema) > 0 {
			opts = append(opts, WithMetadataSchema(schema))
		}
		if len(columnConstraints) > 0 {
			opts = append(opts, WithColumnConstraints(columnConstraints))
		}
		if len(plan.DDLForeignKeys) > 0 {
			// DDL-time FK validation: verify referenced table and column
			// exist in the catalog before accepting the constraint.
			e.db.mu.RLock()
			cat := e.db.catalog
			e.db.mu.RUnlock()
			for _, pfk := range plan.DDLForeignKeys {
				if len(pfk.SourceColumns) == 0 || len(pfk.SourceColumns) != len(pfk.TargetColumns) {
					return nil, fmt.Errorf("foreign key %q must have the same non-zero number of source and target columns", pfk.Name)
				}
				// Allow self-referencing FKs — the target table is
				// being created in this same DDL statement.
				if strings.EqualFold(pfk.TargetTable, plan.DDLTableName) {
					continue
				}
				tgtHash := catalog.HashIdentifier(pfk.TargetTable)
				tgtTable, err := cat.GetTable(tgtHash)
				if err != nil {
					// System tables (GRAPH_NODES) are not in the catalog binary.
					if sysDef, ok := catalog.ResolveSystemTable(pfk.TargetTable); ok {
						if _, colErr := catalog.ResolveSystemColumn(sysDef.OID,
							catalog.HashIdentifier(pfk.TargetColumns[0])); colErr != nil {
							return nil, fmt.Errorf(
								"foreign key %q: column %q does not exist in system table %q",
								pfk.Name, pfk.TargetColumns[0], pfk.TargetTable)
						}
						continue
					}
					return nil, fmt.Errorf(
						"foreign key %q references table %q which does not exist",
						pfk.Name, pfk.TargetTable)
				}
				for _, targetColumn := range pfk.TargetColumns {
					colHash := catalog.HashIdentifier(targetColumn)
					if _, err := cat.GetColumn(tgtTable, colHash); err != nil {
						return nil, fmt.Errorf(
							"foreign key %q: column %q does not exist in referenced table %q",
							pfk.Name, targetColumn, pfk.TargetTable)
					}
				}
			}
			fks := make([]catalog.ForeignKeyInfo, 0, len(plan.DDLForeignKeys))
			for fkIndex, pfk := range plan.DDLForeignKeys {
				constraintName := pfk.Name
				if constraintName == "" {
					// Unnamed constraints still need a stable logical group key;
					// otherwise all empty names collide in the catalog.
					constraintName = fmt.Sprintf("__fk_%s_%d", plan.DDLTableName, fkIndex)
				}
				n := len(pfk.SourceColumns)
				if len(pfk.TargetColumns) < n {
					n = len(pfk.TargetColumns)
				}
				for i := 0; i < n; i++ {
					fks = append(fks, catalog.ForeignKeyInfo{
						Name:         constraintName,
						SourceTable:  plan.DDLTableName,
						SourceColumn: pfk.SourceColumns[i],
						TargetTable:  pfk.TargetTable,
						TargetColumn: pfk.TargetColumns[i],
						OnDelete:     pfk.OnDelete,
						OnUpdate:     pfk.OnUpdate,
					})
				}
			}
			opts = append(opts, WithForeignKeys(fks))
		}
		_, err := e.db.CreateCollection(ctx, plan.DDLTableName, opts...)
		if err != nil {
			return nil, err
		}
		return &SearchResults{}, nil

	case 1: // DROP TABLE
		tableHash := catalog.HashIdentifier(plan.DDLTableName)
		e.db.mu.RLock()
		fks := e.db.catalog.ForeignKeysToTable(tableHash)
		collections := e.db.collections
		e.db.mu.RUnlock()
		if len(fks) > 0 {
			// Build a list of referencing tables for the error.
			refs := make(map[string]bool, len(fks))
			for _, fk := range fks {
				for name := range collections {
					if catalog.HashIdentifier(name) == fk.SourceTableHash {
						refs[name] = true
					}
				}
			}
			refNames := make([]string, 0, len(refs))
			for n := range refs {
				refNames = append(refNames, n)
			}
			return nil, fmt.Errorf(
				"cannot drop table %q: foreign key constraints in %v reference it",
				plan.DDLTableName, refNames)
		}
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
			for len(p) > 0 && p[0] == ' ' {
				p = p[1:]
			}
			for len(p) > 0 && p[len(p)-1] == ' ' {
				p = p[:len(p)-1]
			}
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
		Kind:               optimizer.QueryKindRelational,
		CollectionName:     plan.CollectionName,
		Predicates:         plan.Predicates,
		HasRelationalQuery: len(plan.Predicates) > 0,
	}
	results, err := e.executeRelational(ctx, resolvePlan)
	if err != nil {
		return nil, fmt.Errorf("UPDATE resolve phase: %w", err)
	}
	if len(results.Results) == 0 {
		return results, nil
	}

	// Phase 2: all-or-nothing write via epoch or direct transaction.
	if epoch := epochFromContext(ctx); epoch != nil {
		ids := make([]string, len(results.Results))
		for i, r := range results.Results {
			ids[i] = r.ID
			meta := make(map[string]interface{})
			for j, col := range plan.SetColumns {
				if j < len(plan.SetValues) {
					meta[col] = string(plan.SetValues[j])
				}
			}
			newID, keyChanged, err := e.updatedPrimaryKeyID(ctx, plan.CollectionName, r.ID, r.Metadata, meta)
			if err != nil {
				return nil, fmt.Errorf("UPDATE row %q: %w", r.ID, err)
			}
			if keyChanged {
				err = epoch.Rename(ctx, plan.CollectionName, r.ID, newID, nil, mergeMetadata(r.Metadata, meta))
			} else {
				err = epoch.Update(ctx, plan.CollectionName, r.ID, nil, meta)
			}
			if err != nil {
				return nil, fmt.Errorf("UPDATE row %q: %w", r.ID, err)
			}
		}
		return &SearchResults{Results: results.Results, Total: len(ids)}, nil
	}

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
		newID, keyChanged, err := e.updatedPrimaryKeyID(ctx, plan.CollectionName, r.ID, r.Metadata, meta)
		if err != nil {
			_ = tx.Rollback(ctx)
			return nil, fmt.Errorf("UPDATE row %q: %w", r.ID, err)
		}
		if keyChanged {
			err = tx.Rename(ctx, plan.CollectionName, r.ID, newID, nil, mergeMetadata(r.Metadata, meta))
		} else {
			err = tx.Update(ctx, plan.CollectionName, r.ID, nil, meta)
		}
		if err != nil {
			return nil, fmt.Errorf("UPDATE row %q: %w", r.ID, err)
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return nil, err
	}
	return &SearchResults{Results: results.Results, Total: len(ids)}, nil
}

func mergeMetadata(base, updates map[string]interface{}) map[string]interface{} {
	out := cloneMetadata(base)
	for k, v := range updates {
		if out == nil {
			out = make(map[string]interface{})
		}
		out[k] = v
	}
	return out
}

// updatedPrimaryKeyID derives the physical key after an UPDATE and reports
// whether a declared SQL primary-key component changed.
func (e *Executor) updatedPrimaryKeyID(ctx context.Context, collection, oldID string, oldMetadata, updates map[string]interface{}) (string, bool, error) {
	available := make([]string, 0, len(updates)+1)
	if col, err := e.db.GetCollection(collection); err == nil {
		for name := range col.Config().MetadataSchema {
			available = append(available, name)
		}
	}
	for name := range updates {
		available = append(available, name)
	}
	available = append(available, "id")
	pkColumns := e.compositePrimaryKeyColumns(collection, available)
	if len(pkColumns) == 0 {
		return oldID, false, nil
	}
	merged := mergeMetadata(oldMetadata, updates)
	values := make(map[string]string, len(merged)+1)
	for k, v := range merged {
		values[strings.ToLower(k)] = recordMetaToString(v)
	}
	values["id"] = oldID
	for _, column := range pkColumns {
		if strings.EqualFold(column, "id") {
			if value, ok := values["id"]; ok {
				for name, update := range updates {
					if strings.EqualFold(name, "id") {
						value = recordMetaToString(update)
						break
					}
				}
				values["id"] = value
			}
		}
	}
	newID := oldID
	if len(pkColumns) == 1 && strings.EqualFold(pkColumns[0], "id") {
		newID = values["id"]
	} else {
		var err error
		newID, err = encodeCompositePrimaryKey(pkColumns, values)
		if err != nil {
			return oldID, false, err
		}
	}
	return newID, newID != oldID, nil
}

// executeDelete handles DELETE FROM ... WHERE via SELECT-then-write.
func (e *Executor) executeDelete(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	if len(plan.Predicates) == 0 {
		return nil, fmt.Errorf("DELETE requires a WHERE clause")
	}
	resolvePlan := &optimizer.PhysicalPlan{
		Kind:               optimizer.QueryKindRelational,
		CollectionName:     plan.CollectionName,
		Predicates:         plan.Predicates,
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

	// If inside an epoch, stage deletes through the epoch.
	if epoch := epochFromContext(ctx); epoch != nil {
		for _, id := range ids {
			if err := epoch.Delete(ctx, plan.CollectionName, id); err != nil {
				return nil, fmt.Errorf("DELETE %q: %w", id, err)
			}
		}
		return &SearchResults{Results: results.Results, Total: len(ids)}, nil
	}

	// Use the collection's constraint-aware delete path for autocommit SQL.
	// Preflight every row before mutating any row so RESTRICT failures cannot
	// leave a partially deleted statement; CASCADE actions are then applied by
	// Collection.Delete in the same statement path.
	col, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
	}
	for _, id := range ids {
		if _, err := col.checkDeleteFKReferences(ctx, id); err != nil {
			return nil, err
		}
	}
	for _, id := range ids {
		if err := col.Delete(ctx, id); err != nil {
			return nil, err
		}
	}
	return &SearchResults{Results: results.Results, Total: len(ids)}, nil
}

// executeJoin performs a merge join over two B-tree-indexed collections.
// Both cursors advance in lockstep — O(N+M) with zero extra structures.
// Supports INNER (default), LEFT, and CROSS join types.
// executeRelationalFullScan handles relational queries when the index doesn't
// support B-tree access (HNSW, Flat, IVFPQ). Iterates all records via ListAll.
func (e *Executor) executeRelationalFullScan(ctx context.Context, col *Collection, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	records, err := recordsVisibleInContext(ctx, col)
	if err != nil {
		return nil, err
	}
	var results []*SearchResult
	for _, rec := range records {
		if !recordMatchesPredicates(rec, plan.Predicates) {
			continue
		}
		results = append(results, &SearchResult{ID: rec.ID, Score: 1.0, Metadata: rec.Metadata, Ordinal: rec.Ordinal})
		if plan.Limit > 0 && len(results) >= plan.Limit {
			break
		}
	}
	return &SearchResults{Results: results, Total: len(results), Columns: plan.Projections}, nil
}

func (e *Executor) executeJoin(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	// Graph joins (JOIN MATCH) take precedence: every row of the left
	// collection seeds a BFS traversal over the match-path edges.
	if len(plan.GraphJoins) > 0 {
		return e.executeGraphJoin(ctx, plan)
	}
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

// executeGraphJoin implements JOIN MATCH: for each row of the left (FROM)
// collection, resolve the row's key to a graph node and run a BFS over the
// match-path edges. Each reached vertex emits a joined row (leftKey|vertexID).
// LEFT JOIN emits left rows even when no vertex is reached.
func (e *Executor) executeGraphJoin(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	// Epoch guard: route JOIN MATCH through epoch-aware path when inside an epoch.
	if epoch := epochFromContext(ctx); epoch != nil {
		return e.executeGraphJoinEpoch(ctx, plan, epoch)
	}

	leftCol, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
	}
	leftTree, ok := leftCol.GetIndex().(interface{ Tree() *btree.BTree })
	if !ok {
		return nil, fmt.Errorf("JOIN MATCH left collection %q does not support Tree() access", plan.CollectionName)
	}
	g := leftCol.GetGraph()
	if g == nil {
		return nil, fmt.Errorf("JOIN MATCH left collection %q has no graph", plan.CollectionName)
	}

	// Acquire pooled off-heap buffers (reused across all rows and joins)
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

	var results []*SearchResult
	for _, gjp := range plan.GraphJoins {
		isLeftJoin := gjp.JoinType == 1 // parser.JoinLeft

		// Convert optimizer.GraphEdgePlan to graph.EdgePlan
		edges := make([]EdgePlan, len(gjp.GraphEdges))
		totalMinDepth := 0
		for i, gep := range gjp.GraphEdges {
			edges[i] = graphEdgePlanForTraversal(gep)
			totalMinDepth += edges[i].Min
		}

		c := leftTree.Tree().SeekFirst()
		for c.Valid() {
			leftKey := string(c.Key())

			// Resolve this row's key to a graph node (the anchor).
			nodeID, err := e.db.GetNodeID(ctx, plan.CollectionName, leftKey)
			if err != nil {
				// Row is not a graph node — no traversal possible.
				if isLeftJoin {
					results = append(results, &SearchResult{ID: leftKey + "|", Score: 1.0})
				}
				c.Next()
				continue
			}

			seedID := nodeID
			seen := make(map[uint64]bool) // nodeID → reached via traversal

			if err := g.BFSPattern(nodeID, edges, gjp.MaxHops, func(vid uint64, band int, step int) bool {
				// Include the seed only if the first band allows zero-hop
				// matches (Min == 0 for ->*).  Otherwise exclude the seed
				// initialization visit — it must be reached via expansion
				// or band transition to count.
				if vid == seedID && band == 0 && step == 0 {
					if edges[0].Min == 0 {
						seen[vid] = true
					}
				} else {
					seen[vid] = true
				}
				return plan.Limit <= 0 || len(results) < plan.Limit
			}, bitset, frontier); err != nil {
				return nil, err
			}

			bitset.Clear()
			frontier.Clear()

			// Emit joined rows: leftKey|vertexRecID, filtering by min depth.
			emitted := false
			for vid := range seen {
				_, recID, err := e.db.ResolveNodeID(ctx, vid)
				if err != nil {
					continue
				}
				results = append(results, &SearchResult{ID: leftKey + "|" + recID, Score: 1.0})
				emitted = true
				if plan.Limit > 0 && len(results) >= plan.Limit {
					break
				}
			}
			if !emitted && isLeftJoin {
				results = append(results, &SearchResult{ID: leftKey + "|", Score: 1.0})
			}

			c.Next()
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

// aggregateColumnName returns the output column name for an aggregate function.
// executeGraphJoinEpoch runs JOIN MATCH inside an epoch using the epoch overlay.
// Relational left-side rows come from RecordsEpoch; graph traversal uses
// NeighborsOverlay/InboundNeighborsOverlay; terminal resolution uses epoch-aware helpers.
func (e *Executor) executeGraphJoinEpoch(ctx context.Context, plan *optimizer.PhysicalPlan, epoch *EpochTx) (*SearchResults, error) {
	leftCol, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
	}
	gtx, err := epoch.GraphTxn(plan.CollectionName)
	if err != nil {
		return nil, fmt.Errorf("epoch graph txn for JOIN MATCH: %w", err)
	}

	// Get left-side rows from epoch-visible records.
	leftRecords, err := recordsVisibleInContext(ctx, leftCol)
	if err != nil {
		return nil, err
	}
	if len(leftRecords) == 0 {
		return &SearchResults{}, nil
	}

	// Build edge plans.
	graphEdges := make([]graph.EdgePlan, len(plan.GraphEdges))
	for i, gep := range plan.GraphEdges {
		minHops := int(gep.QuantMin)
		maxHops := int(gep.QuantMax)
		if maxHops == 0 {
			if gep.QuantMin == 0 {
				minHops = 1
				maxHops = 1
			} else {
				maxHops = 1 << 20
			}
		}
		graphEdges[i] = graph.EdgePlan{Dir: gep.Direction, Min: minHops, Max: maxHops}
		if gep.EdgeKind != 0 {
			graphEdges[i].KindSet.Set(gep.EdgeKind)
		}
	}

	lastBand := len(graphEdges) - 1
	var results []*SearchResult

	for _, leftRec := range leftRecords {
		// Resolve left record to a graph seed node.
		seed, err := e.lookupNodeIDInContext(ctx, plan.CollectionName, leftRec.ID)
		if err != nil {
			continue
		}

		// BFS over epoch overlay.
		type bfsState struct {
			nid        uint64
			band, step int
		}
		queue := []bfsState{{nid: seed, band: 0, step: 0}}
		visited := make(map[uint64]bool)
		visited[seed] = true

		for len(queue) > 0 {
			cur := queue[0]
			queue = queue[1:]

			if cur.step >= graphEdges[cur.band].Min && cur.band == lastBand {
				// Resolve terminal node to record ID.
				_, terminalID, rerr := e.resolveNodeIDInContext(ctx, cur.nid)
				if rerr == nil {
					// Get terminal record data from epoch view.
					records, _ := recordsVisibleInContext(ctx, leftCol)
					for _, r := range records {
						if r.ID == terminalID {
							results = append(results, &SearchResult{
								ID: terminalID, Score: 1.0,
								Metadata: cloneMetadata(r.Metadata),
							})
							break
						}
					}
				}
			}

			if cur.step >= graphEdges[cur.band].Max || cur.band >= len(graphEdges) {
				continue
			}

			advanceBand := cur.step >= graphEdges[cur.band].Min && cur.band < lastBand
			useInbound := cur.band < len(graphEdges) && graphEdges[cur.band].Dir == -1

			var neighbors []graph.Edge
			if useInbound {
				neighbors, _ = gtx.InboundNeighborsOverlay(cur.nid)
			} else {
				neighbors, _ = gtx.NeighborsOverlay(cur.nid)
			}

			for _, nb := range neighbors {
				kindSetZero := graph.KindSet{}
				if cur.band < len(graphEdges) && graphEdges[cur.band].KindSet != kindSetZero && !graphEdges[cur.band].KindSet.Has(nb.GetKind()) {
					continue
				}
				if visited[nb.Target] {
					continue
				}
				visited[nb.Target] = true
				nextBand := cur.band
				nextStep := cur.step + 1
				if advanceBand && cur.step >= graphEdges[cur.band].Max-1 {
					nextBand = cur.band + 1
					nextStep = 0
				}
				queue = append(queue, bfsState{nid: nb.Target, band: nextBand, step: nextStep})
			}

			if plan.Limit > 0 && len(results) >= plan.Limit {
				break
			}
		}
	}

	return &SearchResults{Results: results, Total: len(results)}, nil
}

func aggregateColumnName(funcType uint8) string {
	switch funcType {
	case 0:
		return "count"
	case 1:
		return "sum"
	case 2:
		return "avg"
	case 3:
		return "min"
	case 4:
		return "max"
	default:
		return "count"
	}
}

// aggregateMetaValue returns the typed aggregate result for Metadata encoding.
func aggregateMetaValue(funcType uint8, count int64, sum float64, minVal, maxVal, resultValue string) interface{} {
	switch funcType {
	case 0: // COUNT
		return count
	case 1: // SUM
		return sum
	case 2: // AVG
		if count > 0 {
			return sum / float64(count)
		}
		return float64(0)
	case 3: // MIN
		return minVal
	case 4: // MAX
		return maxVal
	}
	return count
}

// executeSystemTable handles queries against system tables (pg_class, etc.).
func (e *Executor) executeSystemTable(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	rows, err := e.materializeSystemTableRows(ctx, plan.CollectionName)
	if err != nil {
		return nil, err
	}
	if len(plan.Predicates) > 0 {
		wrapped := &SearchResults{Results: rows}
		rows = filterByPredicates(wrapped, plan.Predicates).Results
	}
	switch plan.Kind {
	case optimizer.QueryKindAggregate:
		return e.computeSystemAggregate(rows, plan), nil
	case optimizer.QueryKindRelational:
		if len(plan.Projections) > 0 {
			for _, r := range rows {
				proj := make(map[string]interface{}, len(plan.Projections))
				for _, colName := range plan.Projections {
					if v, ok := r.Metadata[colName]; ok {
						proj[colName] = v
					}
				}
				r.Metadata = proj
			}
		}
		if plan.Limit > 0 && plan.Limit < len(rows) {
			rows = rows[:plan.Limit]
		}
		return &SearchResults{Results: rows, Total: len(rows), Columns: plan.Projections}, nil
	default:
		return nil, fmt.Errorf("query kind %d not supported on system table %q", plan.Kind, plan.CollectionName)
	}
}

// materializeSystemTableRows builds in-memory rows for a system table.
func (e *Executor) materializeSystemTableRows(ctx context.Context, tableName string) ([]*SearchResult, error) {
	switch strings.ToLower(tableName) {
	case "pg_class":
		return e.materializePgClass(ctx)
	case "graph_nodes":
		return e.materializeGraphNodes(ctx)
	default:
		return nil, fmt.Errorf("unsupported system table: %s", tableName)
	}
}

// materializePgClass returns one row per real user collection.
func (e *Executor) materializePgClass(ctx context.Context) ([]*SearchResult, error) {
	names, err := e.db.ListCollectionsWithContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("pg_class: listing collections: %w", err)
	}
	rows := make([]*SearchResult, 0, len(names))
	for i, name := range names {
		var rowCount int64
		if col, colErr := e.db.GetCollection(name); colErr == nil {
			rowCount = int64(col.Stats(ctx).LiveRecordCount)
		}
		rows = append(rows, &SearchResult{
			ID:    name,
			Score: 1.0,
			Metadata: map[string]interface{}{
				"oid":          int64(100 + i),
				"relname":      name,
				"relnamespace": int64(0),
				"relkind":      "r",
				"reltuples":    float64(rowCount),
			},
		})
	}
	return rows, nil
}

// materializeGraphNodes returns one row per graph node across all graph-enabled
// collections, iterating the reverse directory's off-heap HashMap.
func (e *Executor) materializeGraphNodes(ctx context.Context) ([]*SearchResult, error) {
	names, err := e.db.ListCollectionsWithContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("GRAPH_NODES: listing collections: %w", err)
	}

	var rows []*SearchResult
	for _, name := range names {
		col, colErr := e.db.GetCollection(name)
		if colErr != nil {
			continue
		}
		g := col.GetGraph()
		if g == nil {
			continue
		}

		// Iterate records in this graph-enabled collection.
		if err := col.Iterate(ctx, func(rec Record) error {
			nodeID, nerr := e.db.GetNodeID(ctx, name, rec.ID)
			if nerr != nil || nodeID == 0 {
				return nil
			}
			rows = append(rows, &SearchResult{
				ID:    fmt.Sprintf("%d", nodeID),
				Score: 1.0,
				Metadata: map[string]interface{}{
					"id":         int64(nodeID),
					"collection": name,
					"record_id":  rec.ID,
				},
			})
			return nil
		}); err != nil {
			return nil, fmt.Errorf("GRAPH_NODES: iterating %s: %w", name, err)
		}
	}
	return rows, nil
}

// computeSystemAggregate computes an aggregate over in-memory system table rows.
func (e *Executor) computeSystemAggregate(rows []*SearchResult, plan *optimizer.PhysicalPlan) *SearchResults {
	colName := aggregateColumnName(plan.AggregateFunc)
	count := int64(len(rows))
	var resultValue interface{} = count
	if plan.AggregateFunc != 0 {
		var sum float64
		var minVal, maxVal string
		hasMinMax := false
		validCount := int64(0)
		for _, r := range rows {
			if r.Metadata == nil {
				continue
			}
			v, ok := r.Metadata[plan.AggregateColumn]
			if !ok {
				continue
			}
			validCount++
			strVal := fmt.Sprintf("%v", v)
			if !hasMinMax {
				minVal = strVal
				maxVal = strVal
				hasMinMax = true
			}
			if strVal < minVal {
				minVal = strVal
			}
			if strVal > maxVal {
				maxVal = strVal
			}
			var f float64
			if _, err := fmt.Sscanf(strVal, "%f", &f); err == nil {
				sum += f
			}
		}
		switch plan.AggregateFunc {
		case 1:
			resultValue = sum
		case 2:
			if validCount > 0 {
				resultValue = sum / float64(validCount)
			} else {
				resultValue = float64(0)
			}
		case 3:
			resultValue = minVal
		case 4:
			resultValue = maxVal
		}
	}
	return &SearchResults{
		Results: []*SearchResult{{ID: fmt.Sprintf("%v", resultValue), Score: 1.0, Metadata: map[string]interface{}{colName: resultValue}}},
		Total:   1,
		Columns: []string{colName},
	}
}
