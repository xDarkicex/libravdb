package libravdb

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/xDarkicex/lexer"
	"github.com/xDarkicex/lexer/parser"
	"github.com/xDarkicex/libravdb/internal/catalog"
	"github.com/xDarkicex/libravdb/internal/graph"
	btree "github.com/xDarkicex/libravdb/internal/index/btree"
	"github.com/xDarkicex/libravdb/internal/optimizer"
	"github.com/xDarkicex/libravdb/internal/storage"
	"github.com/xDarkicex/libravdb/internal/util"
)

// Executor dispatches physical plans to concrete execution paths.
type Executor struct {
	db *Database
}

func recordsVisibleInContext(ctx context.Context, col *Collection) ([]Record, error) {
	var (
		records []Record
		err     error
	)
	if epoch := epochFromContext(ctx); epoch != nil {
		records, err = epoch.ListRecords(ctx, col.name)
	} else if tx := transactionFromContext(ctx); tx != nil {
		records, err = tx.visibleRecords(ctx, col.name)
	} else {
		records, err = col.ListAll(ctx)
	}
	if err == nil {
		trackSQLRowsExamined(ctx, len(records))
	}
	return records, err
}

// forEachVisibleRecord keeps the ordinary live aggregate path streaming. The
// epoch and transaction paths still use their merged record snapshots because
// those overlays are not represented by the collection's committed iterator.
func forEachVisibleRecord(ctx context.Context, col *Collection, fn func(Record) error) error {
	if epochFromContext(ctx) != nil || transactionFromContext(ctx) != nil {
		records, err := recordsVisibleInContext(ctx, col)
		if err != nil {
			return err
		}
		for _, record := range records {
			if err := fn(record); err != nil {
				return err
			}
		}
		return nil
	}

	var examined int
	err := col.Iterate(ctx, func(record Record) error {
		examined++
		if err := ctx.Err(); err != nil {
			return err
		}
		return fn(record)
	})
	trackSQLRowsExamined(ctx, examined)
	return err
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
		if plan.HasRRF {
			return e.executeRRF(ctx, plan, plan.SnapshotLSN)
		}
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

func hasReturning(plan *optimizer.PhysicalPlan) bool {
	return plan != nil && (plan.ReturningStar || len(plan.ReturningColumns) > 0)
}

// materializeReturning projects DML rows into the same SearchResults shape
// used by SELECT. It owns fresh row/metadata maps so callers and pgwire
// adapters cannot mutate staged or persisted state through a RETURNING result.
func materializeReturning(plan *optimizer.PhysicalPlan, rows []*SearchResult) *SearchResults {
	if !hasReturning(plan) {
		return &SearchResults{Total: len(rows)}
	}
	columns := append([]string(nil), plan.ReturningColumns...)
	if plan.ReturningStar {
		// RETURNING * always includes the physical record key first. Keep the
		// key in the projected shape even when the row metadata also exposes an
		// id field through a schema declaration.
		if len(columns) == 0 {
			columns = append(columns, "id")
		}
		seen := map[string]struct{}{"id": {}}
		for _, row := range rows {
			if row == nil {
				continue
			}
			for key := range row.Metadata {
				canonical := strings.ToLower(key)
				if canonical == "id" {
					continue
				}
				if _, ok := seen[canonical]; !ok {
					seen[canonical] = struct{}{}
					columns = append(columns, key)
				}
			}
			if len(row.Vector) > 0 {
				if _, ok := seen["embedding"]; !ok {
					seen["embedding"] = struct{}{}
					columns = append(columns, "embedding")
				}
			}
		}
		if len(columns) > 1 {
			sort.Strings(columns[1:])
		}
	}
	out := &SearchResults{Columns: columns, Results: make([]*SearchResult, 0, len(rows))}
	for _, row := range rows {
		if row == nil {
			continue
		}
		projected := &SearchResult{ID: row.ID, Vector: cloneVector(row.Vector), Version: row.Version, Score: row.Score, Ordinal: row.Ordinal, Metadata: make(map[string]interface{}, len(columns))}
		for _, column := range columns {
			if strings.EqualFold(column, "id") {
				continue
			}
			if strings.EqualFold(column, "embedding") && len(row.Vector) > 0 {
				projected.Metadata[column] = formatConflictVector(row.Vector)
				continue
			}
			value, ok := metadataColumnValue(row.Metadata, column)
			if ok {
				projected.Metadata[column] = value
			} else {
				projected.Metadata[column] = nil
			}
		}
		out.Results = append(out.Results, projected)
	}
	out.Total = len(out.Results)
	return out
}

func metadataColumnValue(metadata map[string]interface{}, column string) (interface{}, bool) {
	for key, value := range metadata {
		if strings.EqualFold(key, column) {
			return value, true
		}
	}
	return nil, false
}

func searchRowsFromEntries(entries []VectorEntry) []*SearchResult {
	rows := make([]*SearchResult, 0, len(entries))
	for _, entry := range entries {
		rows = append(rows, &SearchResult{ID: entry.ID, Vector: cloneVector(entry.Vector), Score: 1, Metadata: cloneMetadata(entry.Metadata)})
	}
	return rows
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
		NeighborsAtLSNWithProperties(uint64, uint64) ([]graph.EdgeView, error)
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
	if len(plan.UnionQueries) > 0 {
		return e.executeSetOperation(ctx, plan)
	}
	// Direct pgvector operators are lowered with an explicit metric and must
	// not fall through to the collection-configured metric paths. This route
	// also owns projection materialization for `op AS distance` expressions.
	if plan.HasVectorOperator {
		return e.executeVectorOperatorSQL(ctx, plan)
	}

	// System tables (pg_class, etc.) are materialized in memory rather than
	// looked up as collections. The binder assigns reserved OIDs 1-99 to them.
	if catalog.IsSystemTableOID(plan.CollectionOID) || isSystemTableName(plan.CollectionName) {
		switch plan.Kind {
		case optimizer.QueryKindInsert, optimizer.QueryKindUpdate, optimizer.QueryKindDelete,
			optimizer.QueryKindInsertGraphEdge, optimizer.QueryKindDDL:
			return nil, fmt.Errorf("system table %q is read-only", plan.CollectionName)
		}
		return e.executeSystemTable(ctx, plan)
	}

	// A composed relational + graph + vector plan owns all of its clauses. It
	// must run before generic hybrid routing, which only understands a single
	// collection's scalar/graph constraints.
	if plan.Kind == optimizer.QueryKindMultiModal {
		if plan.HasRRF {
			return e.executeRRF(ctx, plan, 0)
		}
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
		if plan.GraphEdgeDelete {
			return e.executeDeleteGraphEdges(ctx, plan)
		}
		return e.executeDelete(ctx, plan)
	case optimizer.QueryKindJoin:
		return e.executeJoin(ctx, plan)
	case optimizer.QueryKindAggregate:
		if len(plan.GraphJoins) > 0 {
			return e.executeGraphJoinAggregate(ctx, plan)
		}
		return e.executeAggregate(ctx, plan)
	case optimizer.QueryKindDDL:
		return e.executeDDL(ctx, plan)
	default:
		// MaxSim and other future kinds fall through here
		return nil, fmt.Errorf("unknown query kind %d", plan.Kind)
	}
}

func (e *Executor) executeSetOperation(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	if len(plan.UnionQueries) != 2 {
		return nil, fmt.Errorf("set operation requires exactly two branches")
	}
	left, err := e.db.Query(ctx, plan.UnionQueries[0])
	if err != nil {
		return nil, fmt.Errorf("set-operation left branch: %w", err)
	}
	right, err := e.db.Query(ctx, plan.UnionQueries[1])
	if err != nil {
		return nil, fmt.Errorf("set-operation right branch: %w", err)
	}
	if left == nil || right == nil {
		return nil, fmt.Errorf("set operation returned a nil branch")
	}
	if len(left.Columns) != len(right.Columns) {
		return nil, fmt.Errorf("set operation column count mismatch: left=%d right=%d", len(left.Columns), len(right.Columns))
	}
	result := &SearchResults{Columns: append([]string(nil), left.Columns...), ColumnTypes: append([]uint16(nil), left.ColumnTypes...)}
	rightKeys := make(map[string]int, len(right.Results))
	for _, row := range right.Results {
		rightKeys[setResultKey(row, right.Columns)]++
	}

	switch plan.SetOp {
	case uint8(parser.SetOpUnion):
		if plan.SetOpAll {
			for _, row := range left.Results {
				result.Results = append(result.Results, cloneSetResult(row))
			}
			for _, row := range right.Results {
				result.Results = append(result.Results, cloneSetResult(row))
			}
		} else {
			seen := make(map[string]struct{}, len(left.Results)+len(right.Results))
			appendDistinctSetResults(result, left.Results, left.Columns, seen)
			appendDistinctSetResults(result, right.Results, right.Columns, seen)
		}
	case uint8(parser.SetOpIntersect):
		if plan.SetOpAll {
			for _, row := range left.Results {
				key := setResultKey(row, left.Columns)
				if rightKeys[key] == 0 {
					continue
				}
				rightKeys[key]--
				result.Results = append(result.Results, cloneSetResult(row))
			}
		} else {
			seen := make(map[string]struct{}, len(left.Results))
			for _, row := range left.Results {
				key := setResultKey(row, left.Columns)
				if _, exists := seen[key]; exists || rightKeys[key] == 0 {
					continue
				}
				seen[key] = struct{}{}
				result.Results = append(result.Results, cloneSetResult(row))
			}
		}
	case uint8(parser.SetOpExcept):
		if plan.SetOpAll {
			for _, row := range left.Results {
				key := setResultKey(row, left.Columns)
				if rightKeys[key] > 0 {
					rightKeys[key]--
					continue
				}
				result.Results = append(result.Results, cloneSetResult(row))
			}
		} else {
			seen := make(map[string]struct{}, len(left.Results))
			for _, row := range left.Results {
				key := setResultKey(row, left.Columns)
				if _, exists := seen[key]; exists || rightKeys[key] > 0 {
					continue
				}
				seen[key] = struct{}{}
				result.Results = append(result.Results, cloneSetResult(row))
			}
		}
	default:
		return nil, fmt.Errorf("unsupported set operation %d", plan.SetOp)
	}
	result.Total = len(result.Results)
	return result, nil
}

func appendDistinctSetResults(dst *SearchResults, rows []*SearchResult, columns []string, seen map[string]struct{}) {
	for _, row := range rows {
		key := setResultKey(row, columns)
		if _, exists := seen[key]; exists {
			continue
		}
		seen[key] = struct{}{}
		dst.Results = append(dst.Results, cloneSetResult(row))
	}
}

func setResultKey(row *SearchResult, columns []string) string {
	if row == nil {
		return "<nil-row>"
	}
	var b strings.Builder
	for _, column := range columns {
		var value interface{}
		if strings.EqualFold(column, "id") {
			value = row.ID
		} else if row.Metadata != nil {
			value = row.Metadata[column]
		}
		encoded := fmt.Sprintf("%T:%v", value, value)
		b.WriteString(strconv.Itoa(len(encoded)))
		b.WriteByte(':')
		b.WriteString(encoded)
		b.WriteByte('|')
	}
	return b.String()
}

func cloneSetResult(row *SearchResult) *SearchResult {
	if row == nil {
		return nil
	}
	metadata := cloneMetadata(row.Metadata)
	return &SearchResult{ID: row.ID, Score: row.Score, Metadata: metadata}
}

// executeMultiModal composes relational anchor selection, MATCH traversal,
// and vector top-k. The intermediate representation is record IDs, never
// user-facing joined rows: anchors select BFS seeds, terminal graph vertices
// become a bitmap, and the existing filtered-ANN path ranks only those
// terminals.
func (e *Executor) executeMultiModal(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	if len(plan.GraphJoins) == 0 || len(plan.QueryVector) == 0 {
		return nil, fmt.Errorf("multimodal query requires graph MATCH and vector top-k")
	}
	col, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
	}
	var anchors []string
	if len(plan.Joins) > 0 {
		anchors, err = e.multiModalAnchors(ctx, col, plan.Joins)
		if err != nil {
			return nil, err
		}
	} else {
		// A graph+vector query may use the collection itself as its relational
		// source (`FROM documents d WHERE MATCH ...`) without an additional
		// scalar JOIN. Seed that form from the epoch/live-visible records.
		records, listErr := recordsVisibleInContext(ctx, col)
		if listErr != nil {
			return nil, listErr
		}
		anchors = make([]string, 0, len(records))
		// Qualified WHERE predicates on the MATCH anchor (for example
		// `WHERE s.id = 'Hypothesis_A'`) must be applied before traversal.
		// Terminal predicates belong to the graph path and are evaluated by the
		// traversal implementation instead.
		sourcePredicates := plan.Predicates
		if len(plan.GraphJoins) > 0 {
			sourcePredicates = nil
			leftAlias := plan.GraphJoins[0].LeftAlias
			for _, predicate := range plan.Predicates {
				if predicate.Alias == "" || strings.EqualFold(predicate.Alias, leftAlias) {
					sourcePredicates = append(sourcePredicates, predicate)
				}
			}
		}
		for _, rec := range records {
			if len(sourcePredicates) > 0 && !recordMatchesPredicates(rec, sourcePredicates) {
				continue
			}
			anchors = append(anchors, rec.ID)
		}
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
	// Epoch/transaction overlays are not indexed in the live HNSW. Always
	// score the authoritative visible records directly for private contexts;
	// otherwise a staged terminal can be present in the graph overlay but be
	// silently omitted by the live ANN bitmap path.
	privateContext := epochFromContext(ctx) != nil || transactionFromContext(ctx) != nil
	if privateContext || (matchCount <= exactCandidateCap/10 && plan.RecallContract != optimizer.RecallExact) {
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
		NeighborsAtLSNWithProperties(nodeID uint64, snapshotLSN uint64) ([]graph.EdgeView, error)
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
		if !scoreAnchor && len(plan.VectorOperatorProjections) > 0 {
			scoreAnchor = plan.VectorOperatorProjections[0].SourceAlias != "" &&
				plan.VectorOperatorProjections[0].SourceAlias == join.LeftAlias
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
	ep := EdgePlan{Dir: gep.Direction, Min: minHops, Max: maxHops, Weight: gep.Weight, Predicate: gep.Predicate}
	if gep.EdgeKind != 0 {
		ep.KindSet.Set(gep.EdgeKind)
	}
	return ep
}

// temporalBFSPattern runs BFS using NeighborsAtLSN for temporal edge visibility.
func (e *Executor) temporalBFSPattern(ctx context.Context, g interface {
	NeighborsAtLSN(nodeID uint64, snapshotLSN uint64) ([]Edge, error)
	NeighborsAtLSNWithProperties(nodeID uint64, snapshotLSN uint64) ([]graph.EdgeView, error)
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
			neighbors, err := g.NeighborsAtLSNWithProperties(nodeID, snapshotLSN)
			if err != nil {
				return err
			}
			for _, view := range neighbors {
				if !edgePlan.MatchesWithProperties(view.Edge, view.Properties) {
					continue
				}
				key := visitedKey(view.Edge.Target, band)
				if !bitset.Test(key) {
					bitset.Set(key)
					frontier.Push(view.Edge.Target, band, step+1)
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
	scoreProjection := optimizer.VectorFuncProjection{
		IsDistance:  true,
		QueryVector: plan.QueryVector,
	}
	if len(plan.VectorFuncProjections) > 0 {
		scoreProjection = plan.VectorFuncProjections[0]
	}
	for id := range candidates {
		rec, err := col.GetAtLSN(ctx, id, snapshotLSN)
		if err != nil || rec == nil || len(rec.Vector) == 0 {
			continue
		}
		var score float32
		if plan.HasVectorOperator {
			operatorScore, ok := vectorOperatorScore(plan.VectorOperator, plan.QueryVector, rec.Vector)
			if !ok {
				continue
			}
			score = operatorScore
		} else {
			score = computeVectorScore(col, scoreProjection, rec.Vector)
		}
		results = append(results, result{id: id, score: score})
	}
	sort.SliceStable(results, func(i, j int) bool {
		if results[i].score == results[j].score {
			return results[i].id < results[j].id
		}
		if plan.IsDesc {
			return results[i].score > results[j].score
		}
		return results[i].score < results[j].score
	})
	start := plan.Offset
	if start < 0 {
		start = 0
	}
	if start >= len(results) {
		results = nil
	} else if start > 0 {
		results = results[start:]
	}
	out := &SearchResults{Results: make([]*SearchResult, 0, len(results)), Columns: plan.Projections}
	for _, r := range results {
		rec, _ := col.GetAtLSN(ctx, r.id, snapshotLSN)
		sr := &SearchResult{ID: r.id, Score: r.score}
		if rec != nil {
			sr.Metadata = cloneMetadata(rec.Metadata)
			for _, vfp := range plan.VectorFuncProjections {
				if len(vfp.QueryVector) == len(rec.Vector) && len(rec.Vector) > 0 {
					if sr.Metadata == nil {
						sr.Metadata = make(map[string]interface{}, len(plan.VectorFuncProjections))
					}
					sr.Metadata[vfp.Name] = computeVectorScore(col, vfp, rec.Vector)
				}
			}
			for _, vop := range plan.VectorOperatorProjections {
				if score, ok := vectorOperatorScore(vop.Operator, vop.QueryVector, rec.Vector); ok {
					if sr.Metadata == nil {
						sr.Metadata = make(map[string]interface{}, len(plan.VectorOperatorProjections))
					}
					sr.Metadata[vop.Name] = score
				}
			}
		}
		out.Results = append(out.Results, sr)
	}
	if plan.OrderBy != "" {
		e.applyOrderBy(out, plan)
	}
	if len(out.Results) > k {
		out.Results = out.Results[:k]
	}
	out.Total = len(out.Results)
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
		if plan.HasRelationalQuery && planHasPredicates(plan) &&
			!planMatchesSnapshotRecord(plan, r) {
			return true
		}
		var s float32
		if plan.HasVectorOperator {
			operatorScore, ok := vectorOperatorScore(plan.VectorOperator, plan.QueryVector, r.Vector)
			if !ok {
				return true
			}
			s = operatorScore
		} else {
			s = computeVectorScore(col, optimizer.VectorFuncProjection{
				IsDistance: true, QueryVector: plan.QueryVector,
			}, r.Vector)
		}
		results = append(results, scored{rec: r, score: s})
		return true
	})
	// Sort by the SQL operator's distance semantics. pgvector operators are
	// distance expressions, so ascending is the default; DESC reverses it.
	sort.SliceStable(results, func(i, j int) bool {
		if results[i].score == results[j].score {
			return results[i].rec.ID < results[j].rec.ID
		}
		if plan.IsDesc {
			return results[i].score > results[j].score
		}
		return results[i].score < results[j].score
	})
	start := plan.Offset
	if start < 0 {
		start = 0
	}
	if start >= len(results) {
		results = nil
	} else if start > 0 {
		results = results[start:]
	}
	if len(results) > k {
		results = results[:k]
	}
	out := &SearchResults{Results: make([]*SearchResult, len(results)), Total: len(results)}
	for i, s := range results {
		// Temporal vector projections must materialize the same selected
		// columns as the live vector path. Historically this path only exposed
		// the record metadata and stored the distance in SearchResult.Score,
		// which made a projected alias such as `distance` arrive over pgwire as
		// SQL NULL. Build the projected row from the historical record and
		// compute every VECTOR_DISTANCE/SIMILARITY alias against its snapshot
		// vector before returning it.
		metadata := make(map[string]interface{}, len(plan.Projections)+len(plan.VectorFuncProjections))
		for _, colName := range plan.Projections {
			if colName == "id" || colName == "ID" {
				metadata[colName] = s.rec.ID
				continue
			}
			if value, ok := s.rec.Metadata[colName]; ok {
				metadata[colName] = value
			}
		}
		for _, vfp := range plan.VectorFuncProjections {
			if len(vfp.QueryVector) == len(s.rec.Vector) && len(s.rec.Vector) > 0 {
				metadata[vfp.Name] = computeVectorScore(col, vfp, s.rec.Vector)
			}
		}
		for _, vop := range plan.VectorOperatorProjections {
			if score, ok := vectorOperatorScore(vop.Operator, vop.QueryVector, s.rec.Vector); ok {
				metadata[vop.Name] = score
			}
		}
		out.Results[i] = &SearchResult{ID: s.rec.ID, Score: s.score, Metadata: metadata}
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
		if plan.HasRelationalQuery && planHasPredicates(plan) && !planMatchesSnapshotRecord(plan, r) {
			return true
		}
		if !recordMatchesFTSPredicates(*r, plan.FTSPredicates) {
			return true
		}
		results = append(results, &SearchResult{ID: r.ID, Score: 1.0, Metadata: r.Metadata})
		return plan.Limit <= 0 || len(results) < plan.Limit
	}); err != nil {
		return nil, err
	}
	columns := plan.Projections
	if len(columns) == 0 {
		columns = collectionColumns(col)
	}
	return &SearchResults{
		Results:     results,
		Total:       len(results),
		Columns:     columns,
		ColumnTypes: collectionColumnTypes(col, columns),
	}, nil
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
		if pred.NullTest == optimizer.NullTestIsNull {
			return false
		}
		if pred.NullTest == optimizer.NullTestNotNull {
			return true
		}
		if pred.ValueIsNull {
			return false
		}
		return scalarPredicateMatches(r.ID, pred)
	}
	var v interface{}
	ok := false
	if r.Metadata != nil {
		v, ok = r.Metadata[pred.Column]
	}
	isNull := !ok || v == nil
	if pred.NullTest == optimizer.NullTestIsNull {
		return isNull
	}
	if pred.NullTest == optimizer.NullTestNotNull {
		return !isNull
	}
	if isNull || pred.ValueIsNull {
		return false
	}
	return scalarPredicateMatches(v, pred)
}

func (e *Executor) executeMultiModalExact(ctx context.Context, col *Collection, plan *optimizer.PhysicalPlan, candidateIDs map[string]struct{}) (*SearchResults, error) {
	records := make([]Record, 0, len(candidateIDs))
	if epochFromContext(ctx) != nil || transactionFromContext(ctx) != nil {
		visible, err := recordsVisibleInContext(ctx, col)
		if err != nil {
			return nil, err
		}
		for _, record := range visible {
			if _, ok := candidateIDs[record.ID]; ok {
				records = append(records, record)
			}
		}
	} else {
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
				trackSQLGraphExpansion(ctx, 1)
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
					if join.PredicateMatch {
						matched = true
					} else if _, recordID, resolveErr := e.resolveNodeIDInContext(ctx, cur.node); resolveErr == nil {
						candidates[recordID] = struct{}{}
					}
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
					if !band.Matches(edge) {
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
	if plan.HasRelationalQuery && planHasPredicates(plan) && len(results.Results) > 0 {
		if len(plan.PredicateAlternatives) > 0 {
			filtered := results.Results[:0]
			for _, result := range results.Results {
				if searchResultMatchesPlan(plan, result) {
					filtered = append(filtered, result)
				}
			}
			results.Results = filtered
			results.Total = len(filtered)
		} else {
			results = filterByPredicates(results, plan.Predicates)
		}
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
		if pred.NullTest == optimizer.NullTestIsNull {
			return false
		}
		if pred.NullTest == optimizer.NullTestNotNull {
			return true
		}
		if pred.ValueIsNull {
			return false
		}
		return scalarPredicateMatches(r.ID, pred)
	}
	var v interface{}
	ok := false
	if r.Metadata != nil {
		v, ok = r.Metadata[colName]
	}
	isNull := !ok || v == nil
	if pred.NullTest == optimizer.NullTestIsNull {
		return isNull
	}
	if pred.NullTest == optimizer.NullTestNotNull {
		return !isNull
	}
	if isNull || pred.ValueIsNull {
		return false
	}
	return scalarPredicateMatches(v, pred)
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
		dir       int8
		kind      uint8
		qmin      uint16
		qmax      uint16
		weight    graph.WeightFilter
		predicate graph.EdgePredicate
	}
	var edges []bfsEdge
	for _, gep := range plan.GraphEdges {
		qmin, qmax := gep.QuantMin, gep.QuantMax
		if qmax == 0 {
			if qmin == 0 {
				qmin, qmax = 1, 1
			} else {
				qmax = parser.QuantUnbounded
			}
		}
		edges = append(edges, bfsEdge{
			dir:       gep.Direction,
			kind:      gep.EdgeKind,
			qmin:      qmin,
			qmax:      qmax,
			weight:    gep.Weight,
			predicate: gep.Predicate,
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

			key := visitKey(cur)
			if visited[key] {
				continue
			}
			visited[key] = true

			if cur.band >= len(edges) {
				seen[cur.nodeID] = true
				continue
			}

			band := edges[cur.band]
			var neighbors []graph.EdgeView
			var err error
			if band.dir < 0 {
				neighbors, err = gtx.InboundNeighborsOverlayWithProperties(cur.nodeID)
			} else {
				neighbors, err = gtx.NeighborsOverlayWithProperties(cur.nodeID)
			}
			if err != nil {
				continue
			}

			for _, view := range neighbors {
				n := view.Edge
				if band.kind != 0 && n.GetKind() != band.kind {
					continue
				}
				if !band.weight.Matches(n.Weight) {
					continue
				}
				if !band.predicate.MatchesWithProperties(n, view.Properties) {
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
			trackSQLGraphExpansion(ctx, 1)
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
	if len(plan.FTSPredicates) > 0 {
		return e.executeRelationalFullScan(ctx, col, plan)
	}
	if len(plan.PredicateAlternatives) > 0 {
		return e.executeRelationalFullScan(ctx, col, plan)
	}

	idx := col.GetIndex()
	if idx == nil {
		return nil, fmt.Errorf("collection %q has no index", plan.CollectionName)
	}

	tree, ok := idx.(interface{ Tree() *btree.BTree })
	if !ok || tree == nil {
		return e.executeRelationalFullScan(ctx, col, plan)
	}
	for _, pred := range plan.Predicates {
		if pred.ValueIsNull {
			return &SearchResults{}, nil
		}
		if pred.NullTest != optimizer.NullTestNone {
			return e.executeRelationalFullScan(ctx, col, plan)
		}
	}

	// Exact primary-key equality and IN predicates can use direct B-tree
	// probes. IN is represented as one equality predicate with InValues; do
	// not collapse it to PredicateValue(), which would silently return only
	// the first requested ID.
	if len(plan.Predicates) == 1 && plan.Predicates[0].Operator == 12 &&
		strings.EqualFold(plan.Predicates[0].Column, "id") {
		pred := plan.Predicates[0]
		values := pred.InValues
		if len(values) == 0 {
			values = []optimizer.ScalarValue{pred.PredicateValue()}
		}
		if pred.Not {
			return e.executeRelationalFullScan(ctx, col, plan)
		}
		results := make([]*SearchResult, 0, len(values))
		seen := make(map[string]struct{}, len(values))
		for _, value := range values {
			if value.IsNull() {
				continue
			}
			key := value.Bytes()
			if _, exists := seen[string(key)]; exists {
				continue
			}
			seen[string(key)] = struct{}{}
			trackSQLIndexHit(ctx, 1)
			trackSQLRowsExamined(ctx, 1)
			val, err := tree.Tree().Search(ctx, key)
			if err != nil {
				continue
			}
			ord, ver, _ := btree.DecodeValue(val)
			results = append(results, &SearchResult{ID: string(key), Version: uint64(ver), Score: 1.0, Ordinal: ord})
		}
		return e.buildSelectResults(ctx, col, results, plan), nil
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
			rangeStart = pred.PredicateValue().Bytes()
			rangeExclusive = true
			hasRangeFilter = true
		case 14: // <
			rangeEnd = pred.PredicateValue().Bytes()
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

		advance()
	}

	return e.buildSelectResults(ctx, col, results, plan), nil
}

// vectorOperatorMetric maps PostgreSQL/pgvector operator semantics to the
// distance implementation. <#> deliberately returns negative inner product,
// matching pgvector's contract where ascending order means highest inner
// product.
func vectorOperatorMetric(op uint8) (DistanceMetric, bool) {
	switch lexer.Kind(op) {
	case lexer.KindL2Dist:
		return L2Distance, true
	case lexer.KindIPDist:
		return InnerProduct, true
	case lexer.KindCosineDist:
		return CosineDistance, true
	default:
		return L2Distance, false
	}
}

func vectorOperatorScore(op uint8, query, vector []float32) (float32, bool) {
	metric, ok := vectorOperatorMetric(op)
	if !ok || len(query) == 0 || len(query) != len(vector) || len(vector) == 0 {
		return 0, false
	}
	switch metric {
	case L2Distance:
		return util.L2Distance_func(query, vector), true
	case InnerProduct:
		// InnerProduct_func already returns the negative dot product, matching
		// pgvector's <#> ascending-distance contract.
		return util.InnerProduct_func(query, vector), true
	case CosineDistance:
		return util.CosineDistance_func(query, vector), true
	default:
		return 0, false
	}
}

// executeVectorOperatorSQL evaluates simple SQL vector-operator queries over
// the effective visible relation. Matching collection metrics use the
// existing ANN top-k search to preserve the fast path; a metric mismatch,
// predicates, epoch overlay, or missing index falls back to an exact scan so
// operator semantics remain correct.
func (e *Executor) executeVectorOperatorSQL(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	if plan == nil || plan.CollectionName == "" {
		return nil, fmt.Errorf("vector operator query requires a collection")
	}
	if len(plan.Joins) > 0 || len(plan.GraphJoins) > 0 || plan.HasGraphTraversal {
		return nil, fmt.Errorf("vector operators with graph or relational joins are not supported in this query shape")
	}
	col, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
	}
	metric, ok := vectorOperatorMetric(plan.VectorOperator)
	if !ok {
		return nil, fmt.Errorf("unsupported vector distance operator %d", plan.VectorOperator)
	}
	if len(plan.QueryVector) == 0 {
		return nil, fmt.Errorf("vector operator query requires a non-empty query vector")
	}

	// Use the existing index only when its metric exactly matches the SQL
	// operator. The returned IDs are rescored with the operator itself before
	// projection, so SearchResult.Score remains the SQL distance rather than
	// the engine's normalized public relevance score.
	var indexedIDs []string
	canUseANN := plan.HasVectorOperatorOrder && len(plan.Predicates) == 0 && len(plan.PredicateAlternatives) == 0 &&
		epochFromContext(ctx) == nil && transactionFromContext(ctx) == nil &&
		plan.Limit > 0 && col.GetIndex() != nil && col.Config().Metric == metric
	if canUseANN {
		k := plan.Limit + plan.Offset
		if k > 0 {
			ann, annErr := col.Query(ctx).WithVector(plan.QueryVector).Limit(k).Execute()
			if annErr == nil {
				indexedIDs = make([]string, 0, len(ann.Results))
				for _, row := range ann.Results {
					if row != nil {
						indexedIDs = append(indexedIDs, row.ID)
					}
				}
			}
		}
	}

	var records []Record
	if len(indexedIDs) > 0 {
		records = make([]Record, 0, len(indexedIDs))
		for _, id := range indexedIDs {
			record, getErr := col.Get(ctx, id)
			if getErr == nil {
				records = append(records, record)
			}
		}
	} else {
		records, err = recordsVisibleInContext(ctx, col)
		if err != nil {
			return nil, err
		}
	}

	type scoredRecord struct {
		record Record
		score  float32
	}
	scored := make([]scoredRecord, 0, len(records))
	for _, record := range records {
		if planHasPredicates(plan) && !planMatchesRecord(plan, record) {
			continue
		}
		score, scoreOK := vectorOperatorScore(plan.VectorOperator, plan.QueryVector, record.Vector)
		if !scoreOK {
			continue
		}
		scored = append(scored, scoredRecord{record: record, score: score})
	}
	if plan.HasVectorOperatorOrder {
		sort.SliceStable(scored, func(i, j int) bool {
			if scored[i].score == scored[j].score {
				return scored[i].record.ID < scored[j].record.ID
			}
			if plan.IsDesc {
				return scored[i].score > scored[j].score
			}
			return scored[i].score < scored[j].score
		})
	}

	rows := make([]*SearchResult, 0, len(scored))
	for _, item := range scored {
		rows = append(rows, &SearchResult{
			ID:       item.record.ID,
			Score:    item.score,
			Vector:   cloneVector(item.record.Vector),
			Metadata: cloneMetadata(item.record.Metadata),
		})
	}
	return e.buildSelectResults(ctx, col, rows, plan), nil
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
	if len(results.Columns) == 0 {
		results.Columns = collectionColumns(col)
	}
	results.ColumnTypes = collectionColumnTypes(col, results.Columns)
	return results
}

// buildSelectResults enriches a batch of search results with record metadata
// projected to the plan's column list, then applies ORDER BY if requested.
func (e *Executor) buildSelectResults(ctx context.Context, col *Collection, results []*SearchResult, plan *optimizer.PhysicalPlan) *SearchResults {
	out := &SearchResults{}
	columns := plan.Projections
	if len(columns) == 0 {
		// A bare SELECT * must expose the collection's actual relational
		// columns over pgwire. The legacy empty projection shape (id/score)
		// loses metadata fields and causes database/sql ORMs such as GORM to
		// scan every non-key field as NULL/default values.
		columns = collectionColumns(col)
	}
	if len(results) == 0 {
		out.Columns = columns
		return out
	}
	for _, sr := range results {
		out.Results = append(out.Results, e.attachMetadata(ctx, col, sr, plan))
	}
	if plan.Distinct {
		out.Results = distinctSearchResults(out.Results, plan.Projections)
	}
	out.Total = len(out.Results)
	out.Columns = columns
	out.ColumnTypes = collectionColumnTypes(col, columns)
	if plan.OrderBy != "" {
		e.applyOrderBy(out, plan)
	}
	if plan.Offset > 0 {
		if plan.Offset >= len(out.Results) {
			out.Results = nil
		} else {
			out.Results = out.Results[plan.Offset:]
		}
	}
	if plan.Limit > 0 && len(out.Results) > plan.Limit {
		out.Results = out.Results[:plan.Limit]
	}
	out.Total = len(out.Results)
	return out
}

func distinctSearchResults(rows []*SearchResult, columns []string) []*SearchResult {
	seen := make(map[string]struct{}, len(rows))
	out := make([]*SearchResult, 0, len(rows))
	for _, row := range rows {
		var key strings.Builder
		if len(columns) == 0 {
			key.WriteString(row.ID)
		} else {
			for _, column := range columns {
				var value interface{}
				if strings.EqualFold(column, "id") {
					value = row.ID
				} else if row.Metadata != nil {
					value = row.Metadata[column]
				}
				key.WriteString(fmt.Sprintf("%T:%v\x00", value, value))
			}
		}
		if _, ok := seen[key.String()]; ok {
			continue
		}
		seen[key.String()] = struct{}{}
		out = append(out, row)
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
	if err != nil {
		// Epoch and transaction overlays contain records that are intentionally
		// absent from the live collection. Projection materialization must read
		// that effective view so staged terminal vectors and ids are returned.
		if visible, visibleErr := recordsVisibleInContext(ctx, col); visibleErr == nil {
			for i := range visible {
				if visible[i].ID == sr.ID {
					rec = visible[i]
					err = nil
					break
				}
			}
		}
	}
	if err != nil {
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
	if len(plan.VectorOperatorProjections) > 0 && len(rec.Vector) > 0 {
		for _, vop := range plan.VectorOperatorProjections {
			if score, ok := vectorOperatorScore(vop.Operator, vop.QueryVector, rec.Vector); ok {
				if sr.Metadata == nil {
					sr.Metadata = make(map[string]interface{}, len(plan.VectorOperatorProjections))
				}
				sr.Metadata[vop.Name] = score
			}
		}
	}
	if len(plan.FTSRankProjections) > 0 {
		if sr.Metadata == nil {
			sr.Metadata = make(map[string]interface{}, len(plan.FTSRankProjections))
		}
		for _, fts := range plan.FTSRankProjections {
			if value, ok := recordMetadataValue(rec.Metadata, fts.TextColumn); ok && value != nil {
				sr.Metadata[fts.Name] = ftsRankText(recordMetaToString(value), fts.TextQuery, "plain")
			} else {
				sr.Metadata[fts.Name] = float64(0)
			}
		}
	}
	if len(plan.FTSProjections) > 0 {
		if sr.Metadata == nil {
			sr.Metadata = make(map[string]interface{}, len(plan.FTSProjections))
		}
		for _, fts := range plan.FTSProjections {
			text := fts.Query
			if fts.Column != "" {
				if value, ok := recordMetadataValue(rec.Metadata, fts.Column); ok && value != nil {
					text = recordMetaToString(value)
				} else {
					text = ""
				}
			}
			switch fts.Kind {
			case optimizer.FTSProjectionVector:
				sr.Metadata[fts.Name] = ftsVectorStringConfig(text, fts.Config)
			case optimizer.FTSProjectionQuery:
				sr.Metadata[fts.Name] = ftsQueryStringConfig(text, fts.QueryMode, fts.Config)
			case optimizer.FTSProjectionRank:
				sr.Metadata[fts.Name] = ftsRankTextConfigNorm(text, fts.Query, fts.QueryMode, fts.Config, fts.Normalization)
			}
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
		sourceName := colName
		for _, ref := range plan.ProjectionRefs {
			if strings.EqualFold(ref.OutputName, colName) {
				sourceName = ref.SourceName
				break
			}
		}
		// The vector is stored on the record itself, not in the metadata map.
		// Vector-operator searches used to project `embedding` by looking only
		// in metadata, which silently emitted SQL NULL for a valid stored vector.
		// Keep the physical vector in the established wire representation used by
		// INSERT ... RETURNING and ordinary vector projections.
		if (strings.EqualFold(sourceName, "embedding") || strings.EqualFold(sourceName, "vector") || strings.EqualFold(sourceName, "vec")) && len(rec.Vector) > 0 {
			proj[colName] = formatConflictVector(rec.Vector)
			continue
		}
		if value, ok := recordMetadataValue(sr.Metadata, sourceName); ok {
			proj[colName] = value
			continue
		}
		if v, ok := recordMetadataValue(rec.Metadata, sourceName); ok {
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

// autoIncrementID reports whether the physical id column is a PostgreSQL
// serial column. The HasDefault flag is also persisted in the catalog, so the
// behavior survives a close/reopen just like the rest of the SQL schema.
func autoIncrementID(col *Collection) bool {
	if col == nil || col.config == nil {
		return false
	}
	fieldType, ok := col.config.MetadataSchema["id"]
	if !ok || (fieldType != IntField && fieldType != BigIntField) {
		return false
	}
	return col.config.ColumnConstraints["id"]&catalog.ColFlagHasDefault != 0
}

// nextAutoIncrementID allocates a PostgreSQL-compatible positive integer ID.
// The first allocation derives its starting point from persisted rows; later
// allocations use the in-memory counter and remain unique across concurrent
// connections. Sequence gaps after a failed write are intentional, matching
// PostgreSQL sequence behavior.
func (e *Executor) nextAutoIncrementID(ctx context.Context, table string, col *Collection) (string, error) {
	e.db.autoIncrementMu.Lock()
	defer e.db.autoIncrementMu.Unlock()

	key := strings.ToLower(table)
	next, initialized := e.db.autoIncrementNext[key]
	if !initialized {
		next = 1
		records, err := col.ListAll(ctx)
		if err != nil {
			return "", err
		}
		for _, record := range records {
			value, err := strconv.ParseUint(record.ID, 10, 64)
			if err == nil && value >= next {
				next = value + 1
			}
		}
	}
	e.db.autoIncrementNext[key] = next + 1
	return strconv.FormatUint(next, 10), nil
}

func (e *Executor) observeAutoIncrementID(table, id string) {
	value, err := strconv.ParseUint(id, 10, 64)
	if err != nil {
		return
	}
	e.db.autoIncrementMu.Lock()
	defer e.db.autoIncrementMu.Unlock()
	next := value + 1
	key := strings.ToLower(table)
	if current, ok := e.db.autoIncrementNext[key]; !ok || next > current {
		e.db.autoIncrementNext[key] = next
	}
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
	if catalog.IsSystemTableOID(plan.CollectionOID) || isSystemTableName(plan.CollectionName) {
		return nil, fmt.Errorf("system table %q is read-only", plan.CollectionName)
	}
	col, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
	}
	autoID := autoIncrementID(col)

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

	if plan.InsertSelectSQL != "" {
		selected, err := e.db.Query(ctx, plan.InsertSelectSQL)
		if err != nil {
			return nil, fmt.Errorf("INSERT ... SELECT source: %w", err)
		}
		entries, err := entriesFromSelectResults(selected, plan)
		if err != nil {
			return nil, err
		}
		if len(entries) == 0 {
			return materializeReturning(plan, nil), nil
		}
		if plan.InsertConflictAction != 0 {
			return e.executeInsertOnConflict(ctx, plan, col, entries)
		}
		if epoch := epochFromContext(ctx); epoch != nil {
			for _, entry := range entries {
				if err := epoch.Insert(ctx, plan.CollectionName, entry.ID, entry.Vector, entry.Metadata); err != nil {
					return nil, fmt.Errorf("epoch insert %q: %w", entry.ID, err)
				}
			}
			return materializeReturning(plan, searchRowsFromEntries(entries)), nil
		}
		if err := col.InsertBatch(ctx, entries); err != nil {
			return nil, err
		}
		return materializeReturning(plan, searchRowsFromEntries(entries)), nil
	}

	// Group flat VALUES values into rows.
	var entries []VectorEntry
	for i := 0; i < len(plan.InsertValues); i += colCount {
		var id string
		var vec []float32
		meta := make(map[string]interface{})
		rowValues := make(map[string]string, colCount)
		for j := 0; j < colCount && i+j < len(plan.InsertValues); j++ {
			valueIndex := i + j
			isNull := valueIndex < len(plan.InsertValueNull) && plan.InsertValueNull[valueIndex]
			val := string(plan.InsertValues[valueIndex])
			if colCount > 0 && j < len(plan.InsertColumns) {
				colName := plan.InsertColumns[j]
				if isNull {
					if strings.EqualFold(colName, "id") {
						if autoID {
							continue
						}
						return nil, fmt.Errorf("INSERT column %q is NOT NULL", colName)
					}
					if strings.EqualFold(colName, "vector") || strings.EqualFold(colName, "vec") || strings.EqualFold(colName, "embedding") {
						return nil, fmt.Errorf("INSERT column %q is NOT NULL", colName)
					}
					meta[colName] = nil
					continue
				}
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
			if !autoID {
				return nil, fmt.Errorf("INSERT requires an 'id' column")
			}
			id, err = e.nextAutoIncrementID(ctx, plan.CollectionName, col)
			if err != nil {
				return nil, fmt.Errorf("allocate id: %w", err)
			}
		} else if autoID {
			e.observeAutoIncrementID(plan.CollectionName, id)
		}
		entries = append(entries, VectorEntry{ID: id, Vector: vec, Metadata: meta})
	}

	if len(entries) == 0 {
		return materializeReturning(plan, nil), nil
	}

	if plan.InsertConflictAction != 0 {
		return e.executeInsertOnConflict(ctx, plan, col, entries)
	}

	// If inside an epoch, stage through the epoch's record transaction.
	if epoch := epochFromContext(ctx); epoch != nil {
		for _, entry := range entries {
			if err := epoch.Insert(ctx, plan.CollectionName, entry.ID, entry.Vector, entry.Metadata); err != nil {
				return nil, fmt.Errorf("epoch insert %q: %w", entry.ID, err)
			}
		}
		return materializeReturning(plan, searchRowsFromEntries(entries)), nil
	}

	if err := col.InsertBatch(ctx, entries); err != nil {
		return nil, err
	}
	return materializeReturning(plan, searchRowsFromEntries(entries)), nil
}

func entriesFromSelectResults(results *SearchResults, plan *optimizer.PhysicalPlan) ([]VectorEntry, error) {
	if results == nil || len(results.Results) == 0 {
		return nil, nil
	}
	sourceColumns := results.Columns
	if len(sourceColumns) == 0 {
		sourceColumns = plan.InsertSelectColumns
	}
	if len(plan.InsertColumns) == 0 {
		return nil, fmt.Errorf("INSERT ... SELECT requires a target column list")
	}
	entries := make([]VectorEntry, 0, len(results.Results))
	for _, row := range results.Results {
		if row == nil {
			continue
		}
		meta := make(map[string]interface{}, len(plan.InsertColumns))
		id := ""
		var vector []float32
		for i, target := range plan.InsertColumns {
			if i >= len(sourceColumns) {
				return nil, fmt.Errorf("INSERT ... SELECT returned %d columns, need %d", len(sourceColumns), len(plan.InsertColumns))
			}
			source := sourceColumns[i]
			var value interface{}
			if strings.EqualFold(source, "id") {
				value = row.ID
			} else if row.Metadata != nil {
				value = row.Metadata[source]
				if value == nil {
					for key, candidate := range row.Metadata {
						if strings.EqualFold(key, source) {
							value = candidate
							break
						}
					}
				}
			}
			if strings.EqualFold(target, "id") {
				if value != nil {
					id = recordMetaToString(value)
				}
				continue
			}
			if strings.EqualFold(target, "vector") || strings.EqualFold(target, "vec") || strings.EqualFold(target, "embedding") {
				if value != nil {
					vector = parseVectorLiteral(recordMetaToString(value))
				}
				continue
			}
			meta[target] = value
		}
		if id == "" {
			return nil, fmt.Errorf("INSERT ... SELECT must project an id column")
		}
		entries = append(entries, VectorEntry{ID: id, Vector: vector, Metadata: meta})
	}
	return entries, nil
}

// executeInsertOnConflict implements the supported SQL upsert contract. The
// conflict target must be the physical id, a declared primary-key tuple, or a
// declared single-column UNIQUE key. DO NOTHING skips the conflicting row;
// DO UPDATE applies only the listed assignments and preserves unspecified
// columns. All rows share one existing transaction/epoch boundary.
func (e *Executor) executeInsertOnConflict(ctx context.Context, plan *optimizer.PhysicalPlan, col *Collection, entries []VectorEntry) (*SearchResults, error) {
	target := append([]string(nil), plan.InsertConflictColumns...)
	if plan.InsertConflictConstraint != "" {
		var ok bool
		target, ok = namedConflictConstraintColumns(col, plan.InsertConflictConstraint)
		if !ok {
			return nil, fmt.Errorf("ON CONFLICT constraint %q does not exist on table %q", plan.InsertConflictConstraint, plan.CollectionName)
		}
	}
	if len(target) == 0 {
		target = []string{"id"}
	}
	if plan.InsertConflictConstraint == "" && !e.validConflictTarget(plan.CollectionName, target) {
		return nil, fmt.Errorf("ON CONFLICT target (%s) is not a PRIMARY KEY or UNIQUE key", strings.Join(target, ", "))
	}

	visible, err := recordsVisibleInContext(ctx, col)
	if err != nil {
		return nil, err
	}
	working := make(map[string]Record, len(visible)+len(entries))
	for _, record := range visible {
		working[record.ID] = record
	}

	epoch := epochFromContext(ctx)
	var tx Tx
	if epoch == nil {
		tx, err = e.db.BeginTx(ctx)
		if err != nil {
			return nil, err
		}
		defer func() {
			if tx != nil {
				_ = tx.Rollback(ctx)
			}
		}()
	}

	affected := 0
	returnRows := make([]*SearchResult, 0, len(entries))
	for _, proposed := range entries {
		conflictID := ""
		for _, record := range working {
			if sameConflictKey(record, proposed, target) {
				conflictID = record.ID
				break
			}
		}

		if conflictID == "" {
			if epoch != nil {
				if err := epoch.Upsert(ctx, plan.CollectionName, proposed.ID, proposed.Vector, proposed.Metadata); err != nil {
					return nil, fmt.Errorf("upsert %q: %w", proposed.ID, err)
				}
			} else if err := tx.Upsert(ctx, plan.CollectionName, proposed.ID, proposed.Vector, proposed.Metadata); err != nil {
				return nil, fmt.Errorf("upsert %q: %w", proposed.ID, err)
			}
			working[proposed.ID] = Record{ID: proposed.ID, Vector: cloneVector(proposed.Vector), Metadata: cloneMetadata(proposed.Metadata)}
			affected++
			returnRows = append(returnRows, &SearchResult{ID: proposed.ID, Vector: cloneVector(proposed.Vector), Score: 1, Metadata: cloneMetadata(proposed.Metadata)})
			continue
		}

		if plan.InsertConflictAction == 1 { // DO NOTHING
			continue
		}
		current := working[conflictID]
		if plan.InsertConflictHasWhere {
			condition, isNull, err := evalConflictExpr(plan, plan.InsertConflictWhereRoot, current, proposed)
			if err != nil {
				return nil, fmt.Errorf("evaluate ON CONFLICT WHERE: %w", err)
			}
			if isNull || !strings.EqualFold(condition, "true") {
				continue
			}
		}
		vector, metadata, err := applyConflictAssignments(current, proposed, plan)
		if err != nil {
			return nil, err
		}
		if epoch != nil {
			if err := epoch.Update(ctx, plan.CollectionName, conflictID, vector, metadata); err != nil {
				return nil, fmt.Errorf("ON CONFLICT DO UPDATE %q: %w", conflictID, err)
			}
		} else if err := tx.Update(ctx, plan.CollectionName, conflictID, vector, metadata); err != nil {
			return nil, fmt.Errorf("ON CONFLICT DO UPDATE %q: %w", conflictID, err)
		}
		updated := current
		if vector != nil {
			updated.Vector = cloneVector(vector)
		}
		updated.Metadata = cloneMetadata(current.Metadata)
		if updated.Metadata == nil {
			updated.Metadata = make(map[string]interface{})
		}
		for key, value := range metadata {
			updated.Metadata[key] = value
		}
		working[conflictID] = updated
		affected++
		returnRows = append(returnRows, &SearchResult{ID: updated.ID, Vector: cloneVector(updated.Vector), Score: 1, Metadata: cloneMetadata(updated.Metadata)})
	}

	if tx != nil {
		if err := tx.Commit(ctx); err != nil {
			return nil, err
		}
		tx = nil
	}
	if hasReturning(plan) {
		return materializeReturning(plan, returnRows), nil
	}
	return &SearchResults{Total: affected}, nil
}

func namedConflictConstraintColumns(col *Collection, name string) ([]string, bool) {
	if col == nil {
		return nil, false
	}
	cfg := col.Config()
	for constraint, columns := range cfg.NamedUniqueConstraints {
		if strings.EqualFold(constraint, name) && len(columns) > 0 {
			return append([]string(nil), columns...), true
		}
	}
	// A named table-level primary key is also a legal conflict constraint.
	// Primary-key names are currently retained only when supplied through the
	// runtime named-constraint option; the physical PK remains the fallback.
	return nil, false
}

func (e *Executor) validConflictTarget(collection string, target []string) bool {
	if len(target) == 1 && strings.EqualFold(target[0], "id") {
		return true
	}
	col, err := e.db.GetCollection(collection)
	if err != nil {
		return false
	}
	cfg := col.Config()
	for _, columns := range cfg.NamedUniqueConstraints {
		if sameColumnSet(columns, target) {
			return true
		}
	}
	if len(cfg.PrimaryKeyColumns) == len(target) && len(target) > 0 {
		matched := make(map[string]struct{}, len(target))
		for _, name := range cfg.PrimaryKeyColumns {
			matched[strings.ToLower(name)] = struct{}{}
		}
		for _, name := range target {
			if _, ok := matched[strings.ToLower(name)]; !ok {
				matched = nil
				break
			}
		}
		if matched != nil {
			return true
		}
	}
	if len(target) != 1 {
		return false
	}
	for name, flags := range cfg.ColumnConstraints {
		if strings.EqualFold(name, target[0]) && flags&catalog.ColFlagUnique != 0 {
			return true
		}
	}
	e.db.mu.RLock()
	cat := e.db.catalog
	e.db.mu.RUnlock()
	if cat != nil {
		if hashes, err := cat.PrimaryKeyColumnHashes(catalog.HashIdentifier(collection)); err == nil && len(hashes) == len(target) && len(hashes) > 0 {
			matched := make(map[uint64]struct{}, len(target))
			for _, hash := range hashes {
				matched[hash] = struct{}{}
			}
			for _, name := range target {
				if _, ok := matched[catalog.HashIdentifier(name)]; !ok {
					matched = nil
					break
				}
			}
			if matched != nil {
				return true
			}
		}
		if table, err := cat.GetTable(catalog.HashIdentifier(collection)); err == nil {
			if column, err := cat.GetColumn(table, catalog.HashIdentifier(target[0])); err == nil {
				return column.Flags&catalog.ColFlagUnique != 0 || column.Flags&catalog.ColFlagPrimaryKey != 0
			}
		}
	}
	return false
}

func sameConflictKey(record Record, proposed VectorEntry, target []string) bool {
	for _, column := range target {
		left, leftOK := conflictColumnValue(record.ID, record.Metadata, column)
		right, rightOK := conflictColumnValue(proposed.ID, proposed.Metadata, column)
		// SQL NULL never conflicts with another NULL in a UNIQUE key.
		// Empty strings are valid SQL values and must participate in UNIQUE/PK
		// conflict detection. Only an absent/nil value represents SQL NULL here;
		// SQL NULL never conflicts with another NULL in a UNIQUE key.
		if !leftOK || !rightOK || left != right {
			return false
		}
	}
	return true
}

func conflictColumnValue(id string, metadata map[string]interface{}, column string) (string, bool) {
	if strings.EqualFold(column, "id") {
		return id, id != ""
	}
	for name, value := range metadata {
		if strings.EqualFold(name, column) {
			if value == nil {
				return "", false
			}
			switch value.(type) {
			case map[string]interface{}, map[string]string, []interface{}, []string, []bool,
				[]int, []int8, []int16, []int32, []int64, []uint, []uint8, []uint16,
				[]uint32, []uint64, []float32, []float64:
				if encoded, err := json.Marshal(value); err == nil {
					return string(encoded), true
				}
			}
			return recordMetaToString(value), true
		}
	}
	return "", false
}

func sameColumnSet(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !strings.EqualFold(a[i], b[i]) {
			return false
		}
	}
	return true
}

func namedUniqueKey(id string, metadata map[string]interface{}, columns []string) (string, bool) {
	var b strings.Builder
	for _, column := range columns {
		value, ok := conflictColumnValue(id, metadata, column)
		if !ok {
			return "", false
		}
		b.WriteString(value)
		b.WriteByte(0)
	}
	return b.String(), true
}

func applyConflictAssignments(current Record, proposed VectorEntry, plan *optimizer.PhysicalPlan) ([]float32, map[string]interface{}, error) {
	metadata := make(map[string]interface{})
	for key, value := range current.Metadata {
		metadata[key] = value
	}
	var vector []float32
	for i, column := range plan.InsertConflictSetColumns {
		if strings.EqualFold(column, "id") {
			return nil, nil, fmt.Errorf("ON CONFLICT DO UPDATE cannot modify primary key column %q", column)
		}
		isVector := strings.EqualFold(column, "vector") || strings.EqualFold(column, "vec") || strings.EqualFold(column, "embedding")
		if i < len(plan.InsertConflictExprRoots) {
			value, isNull, err := evalConflictExpr(plan, plan.InsertConflictExprRoots[i], current, proposed)
			if err != nil {
				return nil, nil, fmt.Errorf("evaluate ON CONFLICT assignment %q: %w", column, err)
			}
			if isNull {
				if isVector {
					return nil, nil, fmt.Errorf("ON CONFLICT DO UPDATE cannot set vector column %q to NULL", column)
				}
				metadata[column] = nil
				continue
			}
			if isVector {
				vector = parseVectorLiteral(value)
				if vector == nil {
					return nil, nil, fmt.Errorf("invalid vector expression for column %q", column)
				}
			} else {
				metadata[column] = value
			}
			continue
		}
		excluded := ""
		if i < len(plan.InsertConflictSetExcluded) {
			excluded = plan.InsertConflictSetExcluded[i]
		}
		if excluded != "" {
			if isVector {
				vector = cloneVector(proposed.Vector)
				continue
			}
			value, ok := conflictColumnValue(proposed.ID, proposed.Metadata, excluded)
			if !ok {
				metadata[column] = nil
			} else {
				metadata[column] = value
			}
			continue
		}
		isNull := i < len(plan.InsertConflictSetValueNull) && plan.InsertConflictSetValueNull[i]
		if isVector {
			if isNull {
				return nil, nil, fmt.Errorf("ON CONFLICT DO UPDATE cannot set vector column %q to NULL", column)
			}
			value := ""
			if i < len(plan.InsertConflictSetValues) {
				value = string(plan.InsertConflictSetValues[i])
			}
			vector = parseVectorLiteral(value)
			if vector == nil {
				return nil, nil, fmt.Errorf("invalid vector literal in ON CONFLICT assignment for %q", column)
			}
			continue
		}
		if isNull {
			metadata[column] = nil
		} else if i < len(plan.InsertConflictSetValues) {
			metadata[column] = string(plan.InsertConflictSetValues[i])
		}
	}
	return vector, metadata, nil
}

// evalConflictExpr evaluates a lowered ON CONFLICT assignment against the
// current conflicting row and the proposed EXCLUDED row. SQL NULL propagates
// through arithmetic expressions; empty strings remain ordinary values.
func evalConflictExpr(plan *optimizer.PhysicalPlan, root int32, current Record, proposed VectorEntry) (string, bool, error) {
	if plan == nil {
		return "", false, fmt.Errorf("nil physical plan")
	}
	exprs := plan.InsertConflictExprs
	if root < 0 || int(root) >= len(exprs) {
		return "", false, fmt.Errorf("invalid expression root %d", root)
	}
	expr := exprs[root]
	switch expr.Kind {
	case optimizer.ConflictExprLiteral:
		return string(expr.Literal), expr.IsNull, nil
	case optimizer.ConflictExprColumn:
		value, ok := conflictColumnValue(current.ID, current.Metadata, expr.Column)
		return value, !ok, nil
	case optimizer.ConflictExprExcludedColumn:
		value, ok := conflictColumnValue(proposed.ID, proposed.Metadata, expr.Column)
		if strings.EqualFold(expr.Column, "embedding") || strings.EqualFold(expr.Column, "vector") || strings.EqualFold(expr.Column, "vec") {
			if len(proposed.Vector) == 0 {
				return "", true, nil
			}
			return formatConflictVector(proposed.Vector), false, nil
		}
		return value, !ok, nil
	case optimizer.ConflictExprBinary:
		left, leftNull, err := evalConflictExpr(plan, expr.Left, current, proposed)
		if err != nil {
			return "", false, err
		}
		right, rightNull, err := evalConflictExpr(plan, expr.Right, current, proposed)
		if err != nil {
			return "", false, err
		}
		if leftNull || rightNull {
			return "", true, nil
		}
		return evalConflictBinary(left, right, expr.Operator)
	case optimizer.ConflictExprUnary:
		value, isNull, err := evalConflictExpr(plan, expr.Left, current, proposed)
		if err != nil {
			return "", false, err
		}
		// SQL uses three-valued logic: NOT NULL is still NULL.
		if isNull {
			return "", true, nil
		}
		if lexer.Kind(expr.Operator) != lexer.KindNot {
			return "", false, fmt.Errorf("unsupported unary operator %d", expr.Operator)
		}
		parsed, err := strconv.ParseBool(strings.TrimSpace(value))
		if err != nil {
			return "", false, fmt.Errorf("NOT requires a boolean operand, got %q", value)
		}
		return strconv.FormatBool(!parsed), false, nil
	case optimizer.ConflictExprCase:
		start := expr.CaseWhenStart
		end := start + expr.CaseWhenCount
		if start < 0 || end < start || int(end) > len(plan.InsertConflictCases) {
			return "", false, fmt.Errorf("invalid CASE branch range")
		}
		for _, branch := range plan.InsertConflictCases[start:end] {
			condition, isNull, err := evalConflictExpr(plan, branch.Condition, current, proposed)
			if err != nil {
				return "", false, err
			}
			if !isNull && strings.EqualFold(condition, "true") {
				return evalConflictExpr(plan, branch.Value, current, proposed)
			}
		}
		if expr.CaseElse >= 0 {
			return evalConflictExpr(plan, expr.CaseElse, current, proposed)
		}
		return "", true, nil
	case optimizer.ConflictExprFunction:
		if strings.EqualFold(expr.Function, "NOW") {
			if expr.Left >= 0 || expr.Right >= 0 {
				return "", false, fmt.Errorf("NOW() does not accept arguments")
			}
			return time.Now().UTC().Format(time.RFC3339Nano), false, nil
		}
		if !strings.EqualFold(expr.Function, "NULLIF") {
			return "", false, fmt.Errorf("unsupported function %q", expr.Function)
		}
		left, leftNull, err := evalConflictExpr(plan, expr.Left, current, proposed)
		if err != nil {
			return "", false, err
		}
		right, rightNull, err := evalConflictExpr(plan, expr.Right, current, proposed)
		if err != nil {
			return "", false, err
		}
		if leftNull {
			return "", true, nil
		}
		if !rightNull && left == right {
			return "", true, nil
		}
		return left, false, nil
	case optimizer.ConflictExprCast:
		value, isNull, err := evalConflictExpr(plan, expr.Left, current, proposed)
		if err != nil || isNull {
			return value, isNull, err
		}
		typ := strings.ToLower(strings.TrimSpace(expr.Type))
		switch typ {
		case "text", "varchar", "character varying", "char", "string", "json", "jsonb":
			return value, false, nil
		case "uuid":
			if !validVirtualUUID(strings.TrimSpace(value)) {
				return "", false, fmt.Errorf("cannot cast %q to uuid", value)
			}
			return value, false, nil
		case "int", "int2", "int4", "integer", "smallint", "bigint":
			parsed, err := strconv.ParseInt(strings.TrimSpace(value), 10, 64)
			if err != nil {
				return "", false, fmt.Errorf("cannot cast %q to %s: %w", value, expr.Type, err)
			}
			return strconv.FormatInt(parsed, 10), false, nil
		case "bool", "boolean":
			parsed, err := strconv.ParseBool(strings.TrimSpace(value))
			if err != nil {
				return "", false, fmt.Errorf("cannot cast %q to %s: %w", value, expr.Type, err)
			}
			return strconv.FormatBool(parsed), false, nil
		case "vector":
			if len(parseVectorLiteral(strings.TrimSpace(value))) == 0 {
				return "", false, fmt.Errorf("cannot cast %q to vector", value)
			}
			return value, false, nil
		case "timestamp", "timestamptz", "date":
			if _, err := time.Parse(time.RFC3339Nano, strings.TrimSpace(value)); err != nil {
				return "", false, fmt.Errorf("cannot cast %q to %s: %w", value, expr.Type, err)
			}
			return value, false, nil
		case "float", "float4", "float8", "double", "double precision", "real", "numeric", "decimal":
			parsed, err := strconv.ParseFloat(value, 64)
			if err != nil {
				return "", false, fmt.Errorf("cannot cast %q to %s: %w", value, expr.Type, err)
			}
			return strconv.FormatFloat(parsed, 'f', -1, 64), false, nil
		default:
			return "", false, fmt.Errorf("unsupported cast target type %q", typ)
		}
	case optimizer.ConflictExprJSONFunction:
		value, isNull, err := evalConflictExprValue(plan, root, current, proposed)
		if err != nil || isNull {
			return "", isNull, err
		}
		encoded, err := encodeJSONValue(value)
		if err != nil {
			return "", false, err
		}
		return encoded, false, nil
	default:
		return "", false, fmt.Errorf("unsupported ON CONFLICT expression kind %d", expr.Kind)
	}
}

// evalConflictExprValue preserves structured JSON values for UPDATE SET
// expressions. The legacy conflict evaluator intentionally returns strings
// for scalar arithmetic and casts; JSON mutation functions need the decoded
// tree so Collection validation can store it atomically as JSONB metadata.
func evalConflictExprValue(plan *optimizer.PhysicalPlan, root int32, current Record, proposed VectorEntry) (interface{}, bool, error) {
	if plan == nil || root < 0 || int(root) >= len(plan.InsertConflictExprs) {
		return nil, false, fmt.Errorf("invalid JSON expression root %d", root)
	}
	expr := plan.InsertConflictExprs[root]
	if expr.Kind != optimizer.ConflictExprJSONFunction {
		value, isNull, err := evalConflictExpr(plan, root, current, proposed)
		if err != nil || isNull {
			return nil, isNull, err
		}
		return value, false, nil
	}

	children := [...]int32{expr.Left, expr.Right, expr.Third, expr.Fourth}
	args := make([]interface{}, 0, len(children))
	for _, child := range children {
		if child < 0 {
			break
		}
		value, isNull, err := evalConflictExprValue(plan, child, current, proposed)
		if err != nil {
			return nil, false, err
		}
		if isNull {
			return nil, true, nil
		}
		args = append(args, value)
	}
	value, handled, err := evaluateJSONFunction(expr.Function, args)
	if err != nil {
		return nil, false, err
	}
	if !handled {
		return nil, false, fmt.Errorf("unsupported JSON mutation function %q", expr.Function)
	}
	if value == nil {
		return nil, true, nil
	}
	return value, false, nil
}

func evalConflictBinary(left, right string, operator uint8) (string, bool, error) {
	switch lexer.Kind(operator) {
	case lexer.KindEquals, lexer.KindGreaterThan, lexer.KindLessThan:
		leftFloat, leftErr := strconv.ParseFloat(left, 64)
		rightFloat, rightErr := strconv.ParseFloat(right, 64)
		if leftErr == nil && rightErr == nil {
			switch lexer.Kind(operator) {
			case lexer.KindEquals:
				return strconv.FormatBool(leftFloat == rightFloat), false, nil
			case lexer.KindGreaterThan:
				return strconv.FormatBool(leftFloat > rightFloat), false, nil
			case lexer.KindLessThan:
				return strconv.FormatBool(leftFloat < rightFloat), false, nil
			}
		}
		switch lexer.Kind(operator) {
		case lexer.KindEquals:
			return strconv.FormatBool(left == right), false, nil
		case lexer.KindGreaterThan:
			return strconv.FormatBool(left > right), false, nil
		case lexer.KindLessThan:
			return strconv.FormatBool(left < right), false, nil
		}
	case lexer.KindAnd, lexer.KindOr:
		l := strings.EqualFold(left, "true")
		r := strings.EqualFold(right, "true")
		if lexer.Kind(operator) == lexer.KindAnd {
			return strconv.FormatBool(l && r), false, nil
		}
		return strconv.FormatBool(l || r), false, nil
	case lexer.KindShiftLeft, lexer.KindShiftRight:
		leftInt, leftErr := strconv.ParseInt(left, 10, 64)
		rightInt, rightErr := strconv.ParseInt(right, 10, 64)
		if leftErr != nil || rightErr != nil || rightInt < 0 || rightInt >= 64 {
			return "", false, fmt.Errorf("shift requires integer operands and a shift count in [0,63]")
		}
		var value int64
		if lexer.Kind(operator) == lexer.KindShiftLeft {
			value = leftInt << uint(rightInt)
		} else {
			value = leftInt >> uint(rightInt)
		}
		return strconv.FormatInt(value, 10), false, nil
	}
	if lexer.Kind(operator) == lexer.KindConcat {
		return left + right, false, nil
	}
	leftInt, leftIntErr := strconv.ParseInt(left, 10, 64)
	rightInt, rightIntErr := strconv.ParseInt(right, 10, 64)
	if leftIntErr == nil && rightIntErr == nil {
		var value int64
		switch lexer.Kind(operator) {
		case lexer.KindPlus:
			value = leftInt + rightInt
		case lexer.KindDash:
			value = leftInt - rightInt
		case lexer.KindAsterisk:
			value = leftInt * rightInt
		case lexer.KindSlash:
			if rightInt == 0 {
				return "", false, fmt.Errorf("division by zero")
			}
			value = leftInt / rightInt
		case lexer.KindPercent:
			if rightInt == 0 {
				return "", false, fmt.Errorf("modulo by zero")
			}
			value = leftInt % rightInt
		default:
			return "", false, fmt.Errorf("operator %d is not supported in ON CONFLICT arithmetic", operator)
		}
		return strconv.FormatInt(value, 10), false, nil
	}
	leftFloat, leftErr := strconv.ParseFloat(left, 64)
	rightFloat, rightErr := strconv.ParseFloat(right, 64)
	if leftErr != nil || rightErr != nil {
		return "", false, fmt.Errorf("operator %d requires numeric operands, got %q and %q", operator, left, right)
	}
	var value float64
	switch lexer.Kind(operator) {
	case lexer.KindPlus:
		value = leftFloat + rightFloat
	case lexer.KindDash:
		value = leftFloat - rightFloat
	case lexer.KindAsterisk:
		value = leftFloat * rightFloat
	case lexer.KindSlash:
		if rightFloat == 0 {
			return "", false, fmt.Errorf("division by zero")
		}
		value = leftFloat / rightFloat
	default:
		return "", false, fmt.Errorf("operator %d is not supported in ON CONFLICT arithmetic", operator)
	}
	return strconv.FormatFloat(value, 'f', -1, 64), false, nil
}

func formatConflictVector(vector []float32) string {
	var b strings.Builder
	b.WriteByte('[')
	for i, value := range vector {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.FormatFloat(float64(value), 'f', -1, 32))
	}
	b.WriteByte(']')
	return b.String()
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
	var properties []byte
	if len(plan.InsertValues) > 3 {
		var err error
		properties, err = graph.NormalizeEdgeProperties(plan.InsertValues[3])
		if err != nil {
			return nil, fmt.Errorf("invalid GRAPH_EDGES properties: %w", err)
		}
	}

	kind := graph.ResolveEdgeKind(kindName)
	if kind == 0 && kindName != "" {
		return nil, fmt.Errorf("unknown edge kind %q", kindName)
	}

	// GRAPH_EDGES does not carry a collection column, so resolve both record
	// IDs against the same graph-backed collection. The old first-collection
	// lookup made SQL graph bootstrap fail as soon as a database contained more
	// than one graph collection: CREATE GRAPH TABLE could create the nodes, but
	// GRAPH_EDGES would search an unrelated graph first.
	col, srcNode, tgtNode, err := e.resolveGraphEdgeEndpoints(ctx, srcID, tgtID)
	if err != nil {
		return nil, err
	}

	g := col.GetGraph()
	if g == nil {
		return nil, fmt.Errorf("collection %q has no graph", col.name)
	}

	// If we are inside an epoch, stage the edge via EpochTx.AddGraphEdge.
	// This increments generation and routes through the ordered operation log.
	// Direct gtx.AddEdge() bypasses generation accounting and ordered logging.
	if epoch := epochFromContext(ctx); epoch != nil {
		if err := epoch.AddGraphEdgeWithPropertiesJSON(col.name, srcNode, tgtNode, 1.0, kind, properties); err != nil {
			return nil, fmt.Errorf("staging graph edge: %w", err)
		}
		return &SearchResults{Total: 1}, nil
	}

	// Direct path: stage via txn and commit immediately.
	txn := g.BeginTxn()
	if err := txn.AddEdgeWithPropertiesJSON(srcNode, tgtNode, 1.0, kind, properties); err != nil {
		txn.Rollback()
		return nil, fmt.Errorf("staging graph edge: %w", err)
	}
	if err := txn.Commit(ctx); err != nil {
		return nil, fmt.Errorf("committing graph edge: %w", err)
	}
	return &SearchResults{Total: 1}, nil
}

// resolveGraphEdgeEndpoints resolves both endpoint record IDs within one
// graph-backed collection. Record IDs are database-scoped graph identities,
// but GRAPH_EDGES intentionally keeps its SQL shape compact and therefore
// omits a collection column. Stable collection ordering preserves the
// existing deterministic behavior when duplicate record IDs exist.
func (e *Executor) resolveGraphEdgeEndpoints(ctx context.Context, srcID, tgtID string) (*Collection, uint64, uint64, error) {
	names := e.db.graphCollectionNames("")
	if len(names) == 0 {
		return nil, 0, 0, fmt.Errorf("no collection with a graph found for edge insert")
	}

	var srcErr, tgtErr error
	for _, name := range names {
		col, err := e.db.GetCollection(name)
		if err != nil || col.GetGraph() == nil {
			continue
		}

		var srcNode, tgtNode uint64
		if epoch := epochFromContext(ctx); epoch != nil {
			srcNode, srcErr = epoch.LookupNodeID(ctx, name, srcID)
			tgtNode, tgtErr = epoch.LookupNodeID(ctx, name, tgtID)
		} else {
			srcNode, srcErr = col.LookupNodeID(ctx, srcID)
			tgtNode, tgtErr = col.LookupNodeID(ctx, tgtID)
		}
		if srcErr == nil && tgtErr == nil && srcNode != 0 && tgtNode != 0 {
			return col, srcNode, tgtNode, nil
		}
	}

	if srcErr != nil {
		return nil, 0, 0, fmt.Errorf("resolving source node %q: %w", srcID, srcErr)
	}
	if tgtErr != nil {
		return nil, 0, 0, fmt.Errorf("resolving target node %q: %w", tgtID, tgtErr)
	}
	return nil, 0, 0, fmt.Errorf("resolving graph edge endpoints %q and %q failed", srcID, tgtID)
}

// executeAggregate scans a collection and computes COUNT/SUM/AVG/MIN/MAX.
func (e *Executor) executeAggregate(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	col, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
	}
	if plan.AggregateFunc == uint8(parser.AggVectorAvg) {
		records, err := recordsVisibleInContext(ctx, col)
		if err != nil {
			return nil, err
		}
		return e.executeVectorAverageAggregate(records, plan)
	}
	// COUNT(*) over the live committed collection is maintained by storage as
	// LiveCount. Use it when there is no filter, grouping, or DISTINCT modifier
	// instead of cloning every record merely to count it.
	if plan.AggregateFunc == 0 && plan.AggregateColumn == "" && !plan.AggregateDistinct &&
		len(plan.Predicates) == 0 && len(plan.PredicateAlternatives) == 0 && len(plan.FTSPredicates) == 0 &&
		len(plan.GroupByColumns) == 0 && !plan.HavingAggregate && plan.HavingExpr == "" &&
		epochFromContext(ctx) == nil && transactionFromContext(ctx) == nil {
		count, err := col.Count(ctx)
		if err != nil {
			return nil, err
		}
		resultValue := strconv.Itoa(count)
		colName := plan.AggregateAlias
		if colName == "" {
			colName = aggregateColumnName(plan.AggregateFunc)
		}
		return &SearchResults{
			Results: []*SearchResult{{
				ID:       resultValue,
				Score:    1.0,
				Metadata: map[string]interface{}{colName: int64(count)},
			}},
			Total:   1,
			Columns: []string{colName},
		}, nil
	}
	// GROUP BY is executed as a real partitioned aggregate. The previous
	// implementation populated GroupByColumns during planning but collapsed
	// every group into one global count, which silently dropped the grouped
	// columns from the wire result.
	if len(plan.GroupByColumns) > 0 {
		type aggregateGroup struct {
			keyValues []string
			singleKey string
			count     int64
			sum       float64
			minVal    string
			maxVal    string
			hasMinMax bool
		}
		groups := make(map[string]*aggregateGroup)
		if err := forEachVisibleRecord(ctx, col, func(record Record) error {
			if !planMatchesRecord(plan, record) {
				return nil
			}
			var key string
			var keyValues []string
			if len(plan.GroupByColumns) == 1 {
				value, ok := joinRecordValue(record, plan.GroupByColumns[0])
				if !ok {
					value = ""
				}
				key = value
			} else {
				keyValues = make([]string, len(plan.GroupByColumns))
				for i, column := range plan.GroupByColumns {
					value, ok := joinRecordValue(record, column)
					if !ok {
						value = ""
					}
					keyValues[i] = value
				}
				key = strings.Join(keyValues, "\x00")
			}
			group := groups[key]
			if group == nil {
				group = &aggregateGroup{}
				if len(plan.GroupByColumns) == 1 {
					group.singleKey = key
				} else {
					// keyValues is newly allocated for this input row and is no
					// longer used after the group is created, so transfer ownership
					// instead of copying it a second time.
					group.keyValues = keyValues
				}
				groups[key] = group
			}
			group.count++

			value := record.ID
			if plan.AggregateColumn != "" {
				if aggregateValue, ok := joinRecordValue(record, plan.AggregateColumn); ok {
					value = aggregateValue
				} else {
					return nil
				}
			}
			if plan.AggregateFunc != 0 {
				if !group.hasMinMax {
					group.minVal, group.maxVal, group.hasMinMax = value, value, true
				}
				if value < group.minVal {
					group.minVal = value
				}
				if value > group.maxVal {
					group.maxVal = value
				}
				if parsed, parseErr := strconv.ParseFloat(value, 64); parseErr == nil {
					group.sum += parsed
				}
			}
			return nil
		}); err != nil {
			return nil, err
		}

		aggregateName := plan.AggregateAlias
		if aggregateName == "" {
			aggregateName = aggregateColumnName(plan.AggregateFunc)
		}
		columns := append([]string(nil), plan.GroupByColumns...)
		columns = append(columns, aggregateName)
		out := &SearchResults{Columns: columns}
		for _, group := range groups {
			resultValue := aggregateResultValue(plan.AggregateFunc, group.count, group.sum, group.minVal, group.maxVal)
			if plan.HavingAggregate && !aggregateHavingMatches(
				aggregateMetaValue(plan.HavingAggregateFunc, group.count, group.sum, group.minVal, group.maxVal, resultValue),
				plan.HavingOp, plan.HavingValue) {
				continue
			}
			metadata := make(map[string]interface{}, len(columns))
			for i, column := range plan.GroupByColumns {
				if len(plan.GroupByColumns) == 1 {
					metadata[column] = group.singleKey
				} else {
					metadata[column] = group.keyValues[i]
				}
			}
			metadata[aggregateName] = aggregateMetaValue(plan.AggregateFunc, group.count, group.sum, group.minVal, group.maxVal, resultValue)
			id := ""
			if len(group.keyValues) > 0 {
				id = group.keyValues[0]
			} else if len(plan.GroupByColumns) == 1 {
				id = group.singleKey
			}
			out.Results = append(out.Results, &SearchResult{ID: id, Score: 1.0, Metadata: metadata})
		}
		if plan.OrderBy != "" {
			e.applyOrderBy(out, plan)
		} else {
			sort.SliceStable(out.Results, func(i, j int) bool { return out.Results[i].ID < out.Results[j].ID })
		}
		if plan.Limit > 0 && len(out.Results) > plan.Limit {
			out.Results = out.Results[:plan.Limit]
		}
		out.Total = len(out.Results)
		return out, nil
	}
	var count int64
	var sum float64
	var minVal, maxVal string
	hasMinMax := false
	if err := forEachVisibleRecord(ctx, col, func(record Record) error {
		if !planMatchesRecord(plan, record) {
			return nil
		}
		value := record.ID
		if plan.AggregateColumn != "" {
			var ok bool
			value, ok = joinRecordValue(record, plan.AggregateColumn)
			if !ok {
				return nil
			}
		}
		count++
		if plan.AggregateFunc != 0 {
			if !hasMinMax {
				minVal, maxVal, hasMinMax = value, value, true
			}
			if value < minVal {
				minVal = value
			}
			if value > maxVal {
				maxVal = value
			}
			if f, parseErr := strconv.ParseFloat(value, 64); parseErr == nil {
				sum += f
			}
		}
		return nil
	}); err != nil {
		return nil, err
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

	colName := plan.AggregateAlias
	if colName == "" {
		colName = aggregateColumnName(plan.AggregateFunc)
	}
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

// executeGraphJoinAggregate groups the terminal rows produced by JOIN MATCH.
// Graph predicates must be evaluated while the aliases still identify their
// graph vertices; applying the aggregate plan directly to the base collection
// would incorrectly evaluate `tgt.id <> $1` against `me.id`.
func (e *Executor) executeGraphJoinAggregate(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	joinPlan := *plan
	joinPlan.Kind = optimizer.QueryKindJoin
	joinPlan.GroupByColumns = nil
	joinPlan.HavingAggregate = false
	joinPlan.HavingExpr = ""
	joinPlan.Limit = -1
	joinPlan.Offset = 0
	joinPlan.Distinct = false

	joined, err := e.executeGraphJoin(ctx, &joinPlan)
	if err != nil {
		return nil, err
	}

	type aggregateGroup struct {
		keyValues []string
		count     int64
		sum       float64
		minVal    string
		maxVal    string
		hasMinMax bool
	}
	groups := make(map[string]*aggregateGroup)
	for _, row := range joined.Results {
		if row == nil {
			continue
		}
		keyValues := make([]string, len(plan.GroupByColumns))
		for i, column := range plan.GroupByColumns {
			value, ok := graphJoinResultValue(row, column)
			if !ok {
				value = ""
			}
			keyValues[i] = value
		}
		key := strings.Join(keyValues, "\x00")
		group := groups[key]
		if group == nil {
			group = &aggregateGroup{keyValues: append([]string(nil), keyValues...)}
			groups[key] = group
		}
		group.count++

		value := row.ID
		if plan.AggregateColumn != "" {
			if aggregateValue, ok := graphJoinResultValue(row, plan.AggregateColumn); ok {
				value = aggregateValue
			} else {
				continue
			}
		}
		if plan.AggregateFunc != 0 {
			if !group.hasMinMax {
				group.minVal, group.maxVal, group.hasMinMax = value, value, true
			}
			if value < group.minVal {
				group.minVal = value
			}
			if value > group.maxVal {
				group.maxVal = value
			}
			if parsed, parseErr := strconv.ParseFloat(value, 64); parseErr == nil {
				group.sum += parsed
			}
		}
	}

	aggregateName := plan.AggregateAlias
	if aggregateName == "" {
		aggregateName = aggregateColumnName(plan.AggregateFunc)
	}
	columns := append([]string(nil), plan.GroupByColumns...)
	columns = append(columns, aggregateName)
	out := &SearchResults{Columns: columns}
	for _, group := range groups {
		resultValue := aggregateResultValue(plan.AggregateFunc, group.count, group.sum, group.minVal, group.maxVal)
		if plan.HavingAggregate && !aggregateHavingMatches(
			aggregateMetaValue(plan.HavingAggregateFunc, group.count, group.sum, group.minVal, group.maxVal, resultValue),
			plan.HavingOp, plan.HavingValue) {
			continue
		}
		metadata := make(map[string]interface{}, len(columns))
		for i, column := range plan.GroupByColumns {
			metadata[column] = group.keyValues[i]
		}
		metadata[aggregateName] = aggregateMetaValue(plan.AggregateFunc, group.count, group.sum, group.minVal, group.maxVal, resultValue)
		id := ""
		if len(group.keyValues) > 0 {
			id = group.keyValues[0]
		}
		out.Results = append(out.Results, &SearchResult{ID: id, Score: 1.0, Metadata: metadata})
	}
	if plan.OrderBy != "" {
		e.applyOrderBy(out, plan)
	} else {
		sort.SliceStable(out.Results, func(i, j int) bool { return out.Results[i].ID < out.Results[j].ID })
	}
	if plan.Limit > 0 && len(out.Results) > plan.Limit {
		out.Results = out.Results[:plan.Limit]
	}
	out.Total = len(out.Results)
	return out, nil
}

func graphJoinResultValue(row *SearchResult, column string) (string, bool) {
	if row == nil {
		return "", false
	}
	if strings.EqualFold(column, "id") {
		if value, ok := row.Metadata["id"]; ok && value != nil {
			return recordMetaToString(value), true
		}
		return row.ID, row.ID != ""
	}
	if value, ok := row.Metadata[column]; ok && value != nil {
		return recordMetaToString(value), true
	}
	for key, value := range row.Metadata {
		if strings.EqualFold(key, column) && value != nil {
			return recordMetaToString(value), true
		}
	}
	return "", false
}

// executeVectorAverageAggregate computes a component-wise mean over the
// collection's persisted vector column. It stays in the normal aggregate
// executor so predicates, GROUP BY, ordering, limits, epochs, and historical
// record visibility all use the same machinery as scalar aggregates.
func (e *Executor) executeVectorAverageAggregate(records []Record, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	if plan.AggregateColumn == "" || !strings.EqualFold(plan.AggregateColumn, "embedding") {
		return nil, fmt.Errorf("VECTOR_AVG only supports the vector column embedding")
	}

	average := func(sum []float64, count int64) interface{} {
		if count == 0 {
			return nil
		}
		out := make([]float32, len(sum))
		for i, value := range sum {
			out[i] = float32(value / float64(count))
		}
		return out
	}

	add := func(sum []float64, vector []float32) ([]float64, error) {
		if len(vector) == 0 {
			return sum, nil
		}
		if len(sum) == 0 {
			sum = make([]float64, len(vector))
		} else if len(sum) != len(vector) {
			return nil, fmt.Errorf("VECTOR_AVG dimension mismatch: got %d, want %d", len(vector), len(sum))
		}
		for i, value := range vector {
			sum[i] += float64(value)
		}
		return sum, nil
	}

	aggregateName := plan.AggregateAlias
	if aggregateName == "" {
		aggregateName = aggregateColumnName(plan.AggregateFunc)
	}
	if len(plan.GroupByColumns) == 0 {
		var sum []float64
		var count int64
		for _, record := range records {
			if !planMatchesRecord(plan, record) {
				continue
			}
			var addErr error
			sum, addErr = add(sum, record.Vector)
			if addErr != nil {
				return nil, addErr
			}
			if len(record.Vector) > 0 {
				count++
			}
		}
		return &SearchResults{
			Results: []*SearchResult{{
				Score:    1,
				Metadata: map[string]interface{}{aggregateName: average(sum, count)},
			}},
			Total:       1,
			Columns:     []string{aggregateName},
			ColumnTypes: []uint16{catalog.TypeVector},
		}, nil
	}

	type vectorAggregateGroup struct {
		keyValues []string
		sum       []float64
		count     int64
	}
	groups := make(map[string]*vectorAggregateGroup)
	for _, record := range records {
		if !planMatchesRecord(plan, record) {
			continue
		}
		keyValues := make([]string, len(plan.GroupByColumns))
		for i, column := range plan.GroupByColumns {
			value, ok := joinRecordValue(record, column)
			if !ok {
				value = ""
			}
			keyValues[i] = value
		}
		key := strings.Join(keyValues, "\x00")
		group := groups[key]
		if group == nil {
			group = &vectorAggregateGroup{keyValues: append([]string(nil), keyValues...)}
			groups[key] = group
		}
		var addErr error
		group.sum, addErr = add(group.sum, record.Vector)
		if addErr != nil {
			return nil, addErr
		}
		if len(record.Vector) > 0 {
			group.count++
		}
	}

	columns := append([]string(nil), plan.GroupByColumns...)
	columns = append(columns, aggregateName)
	out := &SearchResults{
		Columns:     columns,
		ColumnTypes: make([]uint16, len(columns)),
	}
	out.ColumnTypes[len(out.ColumnTypes)-1] = catalog.TypeVector
	for _, group := range groups {
		metadata := make(map[string]interface{}, len(columns))
		for i, column := range plan.GroupByColumns {
			metadata[column] = group.keyValues[i]
		}
		metadata[aggregateName] = average(group.sum, group.count)
		id := ""
		if len(group.keyValues) > 0 {
			id = group.keyValues[0]
		}
		out.Results = append(out.Results, &SearchResult{ID: id, Score: 1, Metadata: metadata})
	}
	if plan.OrderBy != "" {
		e.applyOrderBy(out, plan)
	} else {
		sort.SliceStable(out.Results, func(i, j int) bool { return out.Results[i].ID < out.Results[j].ID })
	}
	if plan.Limit > 0 && len(out.Results) > plan.Limit {
		out.Results = out.Results[:plan.Limit]
	}
	out.Total = len(out.Results)
	return out, nil
}

// aggregateHavingMatches evaluates the scalar comparison in a HAVING clause
// against a typed aggregate result. It deliberately reuses the same typed
// comparison path as ordinary WHERE predicates, so COUNT(*) > 1 does not
// fall back to parsing or comparing formatted SQL strings.
func aggregateHavingMatches(actual interface{}, op uint8, raw []byte) bool {
	expected := optimizer.ScalarFromLiteralBytes(raw)
	cmp, actualNull, err := optimizer.CompareScalar(actual, expected)
	if err != nil || actualNull || expected.IsNull() {
		return false
	}
	return optimizer.MatchesOperator(cmp, op)
}

// sqlTypeToFieldType maps SQL column types to metadata FieldTypes for schema
// registration. Returns ok=false for types without a metadata equivalent.
func sqlTypeToFieldType(sqlType string) (FieldType, bool) {
	switch sqlBaseTypeName(sqlType) {
	case "BIGINT", "BIGSERIAL":
		return BigIntField, true
	case "INT", "INTEGER", "SMALLINT", "SERIAL", "SMALLSERIAL":
		return IntField, true
	case "TEXT", "VARCHAR", "CHAR", "STRING", "UUID":
		return StringField, true
	case "FLOAT", "REAL", "DOUBLE", "DOUBLE PRECISION", "DECIMAL", "NUMERIC":
		return FloatField, true
	case "BOOL", "BOOLEAN":
		return BoolField, true
	case "TIMESTAMP", "TIMESTAMPTZ", "TIMESTAMP WITH TIME ZONE", "TIME", "DATE":
		return TimeField, true
	case "JSON":
		return JSONField, true
	case "JSONB":
		return JSONBField, true
	default:
		return StringField, false
	}
}

func sqlBaseTypeName(sqlType string) string {
	typeName := strings.ToUpper(strings.TrimSpace(sqlType))
	if paramStart := strings.IndexByte(typeName, '('); paramStart >= 0 {
		typeName = strings.TrimSpace(typeName[:paramStart])
	}
	return typeName
}

func graphNodesFKSourceTypeSupported(sqlType string) bool {
	switch sqlBaseTypeName(sqlType) {
	case "BIGINT", "INT8", "UINT64", "TEXT", "VARCHAR", "CHAR", "STRING", "UUID":
		return true
	default:
		return false
	}
}

func catalogTypeToFieldType(typ uint16) FieldType {
	switch typ {
	case catalog.TypeJSON:
		return JSONField
	case catalog.TypeJSONB:
		return JSONBField
	default:
		return StringField
	}
}

// executeDDL handles CREATE TABLE, DROP TABLE, CREATE INDEX.
func (e *Executor) executeDDL(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	switch plan.DDLKind {
	case 0: // CREATE TABLE
		opts := []CollectionOption{WithMetadataOnly()}
		var graphLayer Graph
		if plan.DDLGraph {
			var err error
			graphLayer, err = NewGraph(GraphConfig{})
			if err != nil {
				return nil, fmt.Errorf("create graph table %q: %w", plan.DDLTableName, err)
			}
		}
		var schema MetadataSchema
		var vectorCount int
		primaryKeyColumns := append([]string(nil), plan.DDLPrimaryKeyColumns...)
		columnConstraints := map[string]uint16{
			"id": catalog.ColFlagPrimaryKey | catalog.ColFlagNotNull,
		}
		for _, col := range plan.DDLColumns {
			if col.VectorDimension > 0 {
				// The parser stores VECTOR(n) as uint32, while collection
				// configuration uses int. Reject only a platform conversion
				// overflow here; there is no arbitrary 65,535 model ceiling.
				if uint64(col.VectorDimension) > uint64(^uint(0)>>1) {
					return nil, fmt.Errorf("VECTOR dimension %d exceeds platform integer capacity", col.VectorDimension)
				}
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
			if sqlBaseTypeName(col.Type) == "VECTOR" {
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
			if strings.EqualFold(strings.TrimSpace(col.Type), "SERIAL") ||
				strings.EqualFold(strings.TrimSpace(col.Type), "BIGSERIAL") ||
				strings.EqualFold(strings.TrimSpace(col.Type), "SMALLSERIAL") {
				// PostgreSQL serial types have an implicit sequence-backed
				// default. Preserve that fact in the existing catalog flag set.
				col.Flags |= catalog.ColFlagHasDefault
			}
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
			if plan.DDLPrimaryKeyConstraint != "" {
				opts = append(opts, WithNamedUniqueConstraint(plan.DDLPrimaryKeyConstraint, primaryKeyColumns...))
			}
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
				// GRAPH_NODES.id is a durable physical graph identity, but the
				// engine also accepts a logical record identity (TEXT/UUID) and
				// resolves it through the owning graph collection's existing
				// forward map. No second identity store is needed.
				if strings.EqualFold(pfk.TargetTable, "GRAPH_NODES") {
					for _, sourceColumn := range pfk.SourceColumns {
						found := false
						for _, ddlCol := range plan.DDLColumns {
							if !strings.EqualFold(ddlCol.Name, sourceColumn) {
								continue
							}
							found = true
							typeName := strings.ToUpper(ddlCol.Type)
							if !graphNodesFKSourceTypeSupported(typeName) {
								return nil, fmt.Errorf("foreign key %q to GRAPH_NODES.id requires BIGINT/UINT64 or TEXT/UUID source column %q", pfk.Name, sourceColumn)
							}
						}
						if !found {
							return nil, fmt.Errorf("foreign key %q source column %q does not exist in table %q", pfk.Name, sourceColumn, plan.DDLTableName)
						}
					}
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

			// DDL-time FK action validation: SET NULL / SET DEFAULT preconditions.
			for _, pfk := range plan.DDLForeignKeys {
				for _, srcCol := range pfk.SourceColumns {
					colFlags := columnConstraints[srcCol]
					switch {
					case pfk.OnDelete == catalog.OnDeleteSetNull || pfk.OnUpdate == catalog.OnDeleteSetNull:
						if colFlags&catalog.ColFlagNotNull != 0 {
							return nil, fmt.Errorf(
								"foreign key %q: ON DELETE/UPDATE SET NULL requires source column %q to allow NULL",
								pfk.Name, srcCol)
						}
					case pfk.OnDelete == catalog.OnDeleteSetDefault || pfk.OnUpdate == catalog.OnDeleteSetDefault:
						if plan.DDLColumnDefaults == nil {
							return nil, fmt.Errorf(
								"foreign key %q: ON DELETE/UPDATE SET DEFAULT requires source column %q to have a DEFAULT value",
								pfk.Name, srcCol)
						}
						if _, hasDefault := plan.DDLColumnDefaults[srcCol]; !hasDefault {
							return nil, fmt.Errorf(
								"foreign key %q: ON DELETE/UPDATE SET DEFAULT requires source column %q to have a DEFAULT value",
								pfk.Name, srcCol)
						}
					}
				}
			}
		}

		// Store CHECK constraints.
		if len(plan.DDLCheckConstraints) > 0 {
			opts = append(opts, WithCheckConstraints(plan.DDLCheckConstraints))
		}

		// Store column DEFAULTs and mark HasDefault flag.
		if len(plan.DDLColumnDefaults) > 0 {
			opts = append(opts, WithColumnDefaults(plan.DDLColumnDefaults))
			for colName := range plan.DDLColumnDefaults {
				if flags, ok := columnConstraints[colName]; ok {
					columnConstraints[colName] = flags | catalog.ColFlagHasDefault
				} else {
					columnConstraints[colName] = catalog.ColFlagHasDefault
				}
			}
		}
		if graphLayer != nil {
			opts = append(opts, WithGraph(graphLayer))
		}

		_, err := e.db.CreateCollection(ctx, plan.DDLTableName, opts...)
		if err != nil {
			if graphLayer != nil {
				_ = graphLayer.Close()
			}
			return nil, err
		}
		return &SearchResults{}, nil

	case 5: // CREATE EDGE TYPE
		if err := e.db.createSQLEdgeKind(plan.DDLEdgeTypeName, plan.DDLEdgeTypeUndirected, plan.DDLEdgeTypeDirectionSet); err != nil {
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
		col, err := e.db.GetCollection(plan.DDLTableName)
		if err != nil {
			if plan.DDLIfExists {
				return &SearchResults{}, nil
			}
			return nil, fmt.Errorf("CREATE INDEX: table %q not found", plan.DDLTableName)
		}
		if plan.DDLUnique {
			columns := append([]string(nil), plan.DDLIndexColumns...)
			if len(columns) == 0 && plan.DDLColName != "" {
				columns = []string{plan.DDLColName}
			}
			if len(columns) == 0 || plan.DDLIndexName == "" {
				return nil, fmt.Errorf("CREATE UNIQUE INDEX requires a name and at least one column")
			}
			cfg := col.Config()
			if cfg.NamedUniqueConstraints == nil {
				cfg.NamedUniqueConstraints = make(map[string][]string)
			}
			for existing, cols := range cfg.NamedUniqueConstraints {
				if strings.EqualFold(existing, plan.DDLIndexName) {
					if !sameColumnSet(cols, columns) {
						return nil, fmt.Errorf("unique constraint %q already exists with different columns", plan.DDLIndexName)
					}
					return &SearchResults{}, nil
				}
			}
			visible, err := recordsVisibleInContext(ctx, col)
			if err != nil {
				return nil, err
			}
			seen := make(map[string]string, len(visible))
			for _, record := range visible {
				key, ok := namedUniqueKey(record.ID, record.Metadata, columns)
				if !ok {
					continue
				}
				if prior, exists := seen[key]; exists && prior != record.ID {
					return nil, fmt.Errorf("UNIQUE constraint %q violated by rows %q and %q", plan.DDLIndexName, prior, record.ID)
				}
				seen[key] = record.ID
			}
			col.mu.Lock()
			if col.config.NamedUniqueConstraints == nil {
				col.config.NamedUniqueConstraints = make(map[string][]string)
			}
			col.config.NamedUniqueConstraints[plan.DDLIndexName] = columns
			col.mu.Unlock()
		} else if plan.DDLJSONPath != "" {
			columns := append([]string(nil), plan.DDLIndexColumns...)
			if len(columns) != 1 || plan.DDLIndexName == "" {
				return nil, fmt.Errorf("JSON index requires one column and a name")
			}
			if _, ok := jsonPathSegments(plan.DDLJSONPath); !ok {
				return nil, fmt.Errorf("invalid JSON index path %q", plan.DDLJSONPath)
			}
			cfg := col.Config()
			fieldType, exists := cfg.MetadataSchema[columns[0]]
			if !exists {
				// After reopen, relational schema lives in the catalog rather than
				// the physical collection config. Resolve the column type there.
				table, tableErr := e.db.catalog.GetTable(catalog.HashIdentifier(plan.DDLTableName))
				if tableErr == nil {
					column, colErr := e.db.catalog.GetColumn(table, catalog.HashIdentifier(columns[0]))
					if colErr == nil {
						exists = true
						fieldType = catalogTypeToFieldType(column.Type)
					}
				}
			}
			if !exists || (fieldType != JSONField && fieldType != JSONBField) {
				return nil, fmt.Errorf("JSON index column %q must be JSON or JSONB", columns[0])
			}
			for _, existing := range cfg.JSONIndexes {
				if strings.EqualFold(existing.Name, plan.DDLIndexName) {
					if !strings.EqualFold(existing.Column, columns[0]) || existing.Path != plan.DDLJSONPath || existing.TextResult != plan.DDLJSONText {
						return nil, fmt.Errorf("JSON index %q already exists with different definition", plan.DDLIndexName)
					}
					return &SearchResults{}, nil
				}
			}
			col.mu.Lock()
			col.config.JSONIndexes = append(col.config.JSONIndexes, JSONIndexDefinition{
				Name: plan.DDLIndexName, Column: columns[0], Path: plan.DDLJSONPath, TextResult: plan.DDLJSONText,
			})
			col.mu.Unlock()
			updatedCfg := col.Config()
			e.db.registerCollectionInCatalog(plan.DDLTableName, &updatedCfg)
		}
		return &SearchResults{}, nil

	case 3: // DROP INDEX
		// JSON expression indexes are catalog-backed definitions. Remove the
		// declaration and let the next catalog generation rebuild without it;
		// vector/legacy indexes retain their historical no-op behavior here.
		if plan.DDLIndexName != "" {
			e.db.mu.RLock()
			collections := make(map[string]*Collection, len(e.db.collections))
			for name, collection := range e.db.collections {
				collections[name] = collection
			}
			e.db.mu.RUnlock()
			for collectionName, col := range collections {
				cfg := col.Config()
				filtered := cfg.JSONIndexes[:0]
				removed := false
				for _, index := range cfg.JSONIndexes {
					if strings.EqualFold(index.Name, plan.DDLIndexName) {
						removed = true
						continue
					}
					filtered = append(filtered, index)
				}
				if removed {
					cfg.JSONIndexes = append([]JSONIndexDefinition(nil), filtered...)
					col.mu.Lock()
					col.config.JSONIndexes = append([]JSONIndexDefinition(nil), filtered...)
					col.jsonIndex = nil
					col.jsonIndexBuiltAt = 0
					col.jsonContainmentIndex = nil
					col.jsonContainmentBuiltAt = 0
					col.mu.Unlock()
					e.db.registerCollectionInCatalog(collectionName, &cfg)
				}
			}
		}
		return &SearchResults{}, nil

	case 4: // ALTER TABLE ADD/DROP COLUMN
		col, err := e.db.GetCollection(plan.DDLTableName)
		if err != nil {
			return nil, fmt.Errorf("ALTER TABLE: table %q not found", plan.DDLTableName)
		}
		if plan.DDLDropColumn {
			name := strings.TrimSpace(plan.DDLDropColumnName)
			if name == "" {
				return nil, fmt.Errorf("ALTER TABLE: DROP COLUMN requires a column name")
			}
			if strings.EqualFold(name, "id") {
				return nil, fmt.Errorf("ALTER TABLE: cannot drop the primary key column %q", name)
			}

			col.mu.Lock()
			if col.config == nil || col.config.MetadataSchema == nil {
				col.mu.Unlock()
				return nil, fmt.Errorf("ALTER TABLE: column %q does not exist", name)
			}
			actualName := ""
			for existing := range col.config.MetadataSchema {
				if strings.EqualFold(existing, name) {
					actualName = existing
					break
				}
			}
			if actualName == "" {
				col.mu.Unlock()
				return nil, fmt.Errorf("ALTER TABLE: column %q does not exist", name)
			}
			if flags := col.config.ColumnConstraints[actualName]; flags&(catalog.ColFlagPrimaryKey|catalog.ColFlagUnique) != 0 {
				col.mu.Unlock()
				return nil, fmt.Errorf("ALTER TABLE: cannot drop constrained column %q", actualName)
			}
			delete(col.config.MetadataSchema, actualName)
			delete(col.config.ColumnConstraints, actualName)
			delete(col.config.ColumnDefaults, actualName)
			for constraintName, columns := range col.config.NamedUniqueConstraints {
				for _, columnName := range columns {
					if strings.EqualFold(columnName, actualName) {
						delete(col.config.NamedUniqueConstraints, constraintName)
						break
					}
				}
			}
			foreignKeys := col.config.ForeignKeys[:0]
			for _, foreignKey := range col.config.ForeignKeys {
				if !strings.EqualFold(foreignKey.SourceColumn, actualName) {
					foreignKeys = append(foreignKeys, foreignKey)
				}
			}
			col.config.ForeignKeys = append([]catalog.ForeignKeyInfo(nil), foreignKeys...)
			primaryKeyColumns := col.config.PrimaryKeyColumns[:0]
			for _, columnName := range col.config.PrimaryKeyColumns {
				if !strings.EqualFold(columnName, actualName) {
					primaryKeyColumns = append(primaryKeyColumns, columnName)
				}
			}
			col.config.PrimaryKeyColumns = append([]string(nil), primaryKeyColumns...)
			col.mu.Unlock()

			updatedCfg := col.Config()
			e.db.registerCollectionInCatalog(plan.DDLTableName, &updatedCfg)
			return &SearchResults{}, nil
		}
		// Some PostgreSQL clients (including GORM's AutoMigrate) emit an
		// ALTER COLUMN ... SET DEFAULT reconciliation statement after reading
		// an existing schema. The current optimizer represents that metadata
		// reconciliation as an ALTER plan with no AddColumn payload. The
		// default is already persisted on the collection in this case, so the
		// operation is safely idempotent.
		if len(plan.DDLColumns) == 0 {
			_ = col
			return &SearchResults{}, nil
		}
		if len(plan.DDLColumns) != 1 {
			return nil, fmt.Errorf("ALTER TABLE: ADD COLUMN requires exactly one column")
		}
		add := plan.DDLColumns[0]
		name := strings.TrimSpace(add.Name)
		if name == "" {
			// ALTER COLUMN ... SET DEFAULT is currently represented by the
			// parser as an ALTER plan without an AddColumn payload.
			return &SearchResults{}, nil
		}
		if strings.EqualFold(name, "id") {
			return nil, fmt.Errorf("ALTER TABLE: column %q already exists", name)
		}
		if add.VectorDimension > 0 || sqlBaseTypeName(add.Type) == "VECTOR" {
			return nil, fmt.Errorf("ALTER TABLE: adding VECTOR columns is not supported")
		}
		fieldType, ok := sqlTypeToFieldType(add.Type)
		if !ok {
			return nil, fmt.Errorf("ALTER TABLE: unsupported type %q for column %q", add.Type, name)
		}

		col.mu.Lock()
		if col.config == nil {
			col.mu.Unlock()
			return nil, fmt.Errorf("ALTER TABLE: table %q has no configuration", plan.DDLTableName)
		}
		for existing := range col.config.MetadataSchema {
			if strings.EqualFold(existing, name) {
				col.mu.Unlock()
				return nil, fmt.Errorf("ALTER TABLE: column %q already exists", name)
			}
		}
		if col.config.MetadataSchema == nil {
			col.config.MetadataSchema = make(MetadataSchema)
		}
		col.config.MetadataSchema[name] = fieldType
		if add.Flags != 0 {
			if col.config.ColumnConstraints == nil {
				col.config.ColumnConstraints = make(map[string]uint16)
			}
			col.config.ColumnConstraints[name] = add.Flags
		}
		col.mu.Unlock()
		updatedCfg := col.Config()

		// The catalog is the durable SQL schema source. Republish it after the
		// in-memory collection mutation so binders, pgwire Describe, and future
		// reopen operations observe the new column.
		e.db.registerCollectionInCatalog(plan.DDLTableName, &updatedCfg)
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
	if catalog.IsSystemTableOID(plan.CollectionOID) || isSystemTableName(plan.CollectionName) {
		return nil, fmt.Errorf("system table %q is read-only", plan.CollectionName)
	}
	if !planHasPredicates(plan) {
		return nil, fmt.Errorf("UPDATE requires a WHERE clause")
	}
	// Phase 1: resolve matching IDs via relational execution
	resolvePlan := &optimizer.PhysicalPlan{
		Kind:                  optimizer.QueryKindRelational,
		CollectionName:        plan.CollectionName,
		Predicates:            plan.Predicates,
		PredicateAlternatives: plan.PredicateAlternatives,
		HasRelationalQuery:    planHasPredicates(plan),
	}
	results, err := e.executeRelational(ctx, resolvePlan)
	if err != nil {
		return nil, fmt.Errorf("UPDATE resolve phase: %w", err)
	}
	return e.executeUpdateRows(ctx, plan, results)
}

// executeUpdateRows applies an UPDATE to an already-resolved row set. The
// virtual SQL path uses this entry point for predicates such as
// jsonb_typeof(payload->'career') = 'string' that require expression-aware
// row evaluation instead of the column-predicate fast path.
func (e *Executor) executeUpdateRows(ctx context.Context, plan *optimizer.PhysicalPlan, results *SearchResults) (*SearchResults, error) {
	if len(results.Results) == 0 {
		if hasReturning(plan) {
			return materializeReturning(plan, nil), nil
		}
		return results, nil
	}

	// Phase 2a: evaluate every assignment before staging or writing any row.
	// This is important for JSONB mutations: a malformed path/replacement on a
	// later row must not leave an earlier row updated when the caller is using
	// an EpochTx (the direct path also has a storage transaction below).
	type preparedUpdate struct {
		row        *SearchResult
		metadata   map[string]interface{}
		newID      string
		keyChanged bool
		returnMeta map[string]interface{}
	}
	prepared := make([]preparedUpdate, len(results.Results))
	for i, r := range results.Results {
		meta, err := e.evaluateUpdateMetadata(plan, r)
		if err != nil {
			return nil, fmt.Errorf("UPDATE row %q: %w", r.ID, err)
		}
		newID, keyChanged, err := e.updatedPrimaryKeyID(ctx, plan.CollectionName, r.ID, r.Metadata, meta)
		if err != nil {
			return nil, fmt.Errorf("UPDATE row %q: %w", r.ID, err)
		}
		prepared[i] = preparedUpdate{
			row:        r,
			metadata:   meta,
			newID:      newID,
			keyChanged: keyChanged,
			returnMeta: mergeMetadata(r.Metadata, meta),
		}
	}

	// Phase 2b: all-or-nothing write via epoch or direct transaction.
	if epoch := epochFromContext(ctx); epoch != nil {
		ids := make([]string, len(prepared))
		returnRows := make([]*SearchResult, 0, len(prepared))
		for i, item := range prepared {
			r := item.row
			ids[i] = r.ID
			var err error
			returnID := r.ID
			if item.keyChanged {
				err = epoch.Rename(ctx, plan.CollectionName, r.ID, item.newID, nil, item.returnMeta)
				returnID = item.newID
			} else {
				err = epoch.Update(ctx, plan.CollectionName, r.ID, nil, item.metadata)
			}
			if err != nil {
				return nil, fmt.Errorf("UPDATE row %q: %w", r.ID, err)
			}
			returnRows = append(returnRows, &SearchResult{ID: returnID, Vector: cloneVector(r.Vector), Score: r.Score, Metadata: item.returnMeta})
		}
		if hasReturning(plan) {
			return materializeReturning(plan, returnRows), nil
		}
		return &SearchResults{Results: results.Results, Total: len(ids)}, nil
	}

	tx, err := e.db.BeginTx(ctx)
	if err != nil {
		return nil, err
	}
	ids := make([]string, len(prepared))
	returnRows := make([]*SearchResult, 0, len(prepared))
	for i, item := range prepared {
		r := item.row
		ids[i] = r.ID
		returnID := r.ID
		if item.keyChanged {
			err = tx.Rename(ctx, plan.CollectionName, r.ID, item.newID, nil, item.returnMeta)
			returnID = item.newID
		} else {
			err = tx.Update(ctx, plan.CollectionName, r.ID, nil, item.metadata)
		}
		if err != nil {
			_ = tx.Rollback(ctx)
			return nil, fmt.Errorf("UPDATE row %q: %w", r.ID, err)
		}
		returnRows = append(returnRows, &SearchResult{ID: returnID, Vector: cloneVector(r.Vector), Score: r.Score, Metadata: item.returnMeta})
	}
	if err := tx.Commit(ctx); err != nil {
		return nil, err
	}
	if hasReturning(plan) {
		return materializeReturning(plan, returnRows), nil
	}
	return &SearchResults{Results: results.Results, Total: len(ids)}, nil
}

// evaluateUpdateMetadata evaluates each UPDATE assignment against the row
// being modified. Expression trees share the same exact evaluator as
// ON CONFLICT assignments, so arithmetic, shifts, CASE, NULLIF, NOW(), and
// casts retain their SQL semantics instead of being stored as raw source text.
func (e *Executor) evaluateUpdateMetadata(plan *optimizer.PhysicalPlan, row *SearchResult) (map[string]interface{}, error) {
	meta := make(map[string]interface{}, len(plan.SetColumns))
	if row == nil {
		return meta, nil
	}
	current := Record{ID: row.ID, Metadata: row.Metadata}
	exprPlan := *plan
	exprPlan.InsertConflictExprs = plan.SetExprs
	exprPlan.InsertConflictCases = plan.SetExprCases
	for j, col := range plan.SetColumns {
		if j < len(plan.SetExprRoots) && plan.SetExprRoots[j] >= 0 {
			value, isNull, err := evalConflictExprValue(&exprPlan, plan.SetExprRoots[j], current, VectorEntry{ID: row.ID, Metadata: row.Metadata})
			if err != nil {
				return nil, fmt.Errorf("evaluate assignment %q: %w", col, err)
			}
			if isNull {
				meta[col] = nil
			} else {
				meta[col] = value
			}
			continue
		}
		if j < len(plan.SetValues) {
			if j < len(plan.SetValueNull) && plan.SetValueNull[j] {
				meta[col] = nil
			} else {
				meta[col] = string(plan.SetValues[j])
			}
		}
	}
	return meta, nil
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
	var newID string
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
	if catalog.IsSystemTableOID(plan.CollectionOID) {
		return nil, fmt.Errorf("system table %q is read-only", plan.CollectionName)
	}
	if !planHasPredicates(plan) {
		return nil, fmt.Errorf("DELETE requires a WHERE clause")
	}
	resolvePlan := &optimizer.PhysicalPlan{
		Kind:                  optimizer.QueryKindRelational,
		CollectionName:        plan.CollectionName,
		Predicates:            plan.Predicates,
		PredicateAlternatives: plan.PredicateAlternatives,
		HasRelationalQuery:    planHasPredicates(plan),
	}
	results, err := e.executeRelational(ctx, resolvePlan)
	if err != nil {
		return nil, fmt.Errorf("DELETE resolve phase: %w", err)
	}
	if len(results.Results) == 0 {
		if hasReturning(plan) {
			return materializeReturning(plan, nil), nil
		}
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
		if hasReturning(plan) {
			return materializeReturning(plan, results.Results), nil
		}
		return &SearchResults{Results: results.Results, Total: len(ids)}, nil
	}

	// Route autocommit SQL deletes through the transaction layer. This keeps
	// parent records, FK cascades, and graph-node edge drops in one WAL commit.
	tx, err := e.db.BeginTx(ctx)
	if err != nil {
		return nil, err
	}
	for _, id := range ids {
		if err := tx.Delete(ctx, plan.CollectionName, id); err != nil {
			_ = tx.Rollback(ctx)
			return nil, err
		}
	}
	if err := tx.Commit(ctx); err != nil {
		return nil, err
	}
	if hasReturning(plan) {
		return materializeReturning(plan, results.Results), nil
	}
	return &SearchResults{Results: results.Results, Total: len(ids)}, nil
}

// executeDeleteGraphEdges implements DELETE FROM GRAPH_EDGES through the
// existing graph transaction/WAL/epoch machinery. GRAPH_EDGES is a virtual
// relation, so its rows are discovered from the owning graph and mapped back
// to logical record IDs through the durable node directory.
func (e *Executor) executeDeleteGraphEdges(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	if !planHasPredicates(plan) {
		return nil, fmt.Errorf("DELETE FROM GRAPH_EDGES requires a WHERE clause")
	}
	if hasReturning(plan) {
		return nil, fmt.Errorf("DELETE FROM GRAPH_EDGES does not support RETURNING")
	}
	col := e.db.firstGraphCollection()
	if col == nil || col.GetGraph() == nil {
		return nil, fmt.Errorf("no collection with a graph found for edge delete")
	}
	g := col.GetGraph()
	edgeKindNames := e.graphEdgeKindNames()
	type edgeDelete struct {
		src  uint64
		tgt  uint64
		kind uint8
	}
	matches := make([]edgeDelete, 0)
	g.ForEachEdge(func(src, tgt uint64, edge graph.Edge) bool {
		srcCollection, srcID, srcErr := e.db.ResolveNodeID(ctx, src)
		tgtCollection, tgtID, tgtErr := e.db.ResolveNodeID(ctx, tgt)
		if srcErr != nil || tgtErr != nil || srcCollection != col.name || tgtCollection != col.name {
			return true
		}
		matchesPredicates := graphEdgeMatchesPredicates(srcID, tgtID, edge, plan.Predicates, edgeKindNames)
		if len(plan.PredicateAlternatives) > 0 {
			matchesPredicates = false
			for _, clause := range plan.PredicateAlternatives {
				if graphEdgeMatchesPredicates(srcID, tgtID, edge, clause, edgeKindNames) {
					matchesPredicates = true
					break
				}
			}
		}
		if !matchesPredicates && col.GetGraph().IsEdgeKindUndirected(edge.GetKind()) {
			matchesPredicates = graphEdgeMatchesPredicates(tgtID, srcID, edge, plan.Predicates, edgeKindNames)
			if len(plan.PredicateAlternatives) > 0 {
				matchesPredicates = false
				for _, clause := range plan.PredicateAlternatives {
					if graphEdgeMatchesPredicates(tgtID, srcID, edge, clause, edgeKindNames) {
						matchesPredicates = true
						break
					}
				}
			}
		}
		if matchesPredicates {
			matches = append(matches, edgeDelete{src: src, tgt: tgt, kind: edge.GetKind()})
		}
		return true
	})

	if len(matches) == 0 {
		return &SearchResults{}, nil
	}
	if epoch := epochFromContext(ctx); epoch != nil {
		for _, match := range matches {
			if err := epoch.RemoveGraphEdge(col.name, match.src, match.tgt, match.kind); err != nil {
				return nil, fmt.Errorf("staging GRAPH_EDGES delete: %w", err)
			}
		}
		return &SearchResults{Total: len(matches)}, nil
	}

	txn := g.BeginTxn()
	for _, match := range matches {
		if err := txn.RemoveEdge(match.src, match.tgt, match.kind); err != nil {
			_ = txn.Rollback()
			return nil, fmt.Errorf("staging GRAPH_EDGES delete: %w", err)
		}
	}
	if err := txn.Commit(ctx); err != nil {
		return nil, fmt.Errorf("committing GRAPH_EDGES delete: %w", err)
	}
	return &SearchResults{Total: len(matches)}, nil
}

func (e *Executor) graphEdgeKindNames() map[uint8][]string {
	names := make(map[uint8][]string)
	if definitions, ok := e.db.storage.(storage.EdgeKindDefinitionStore); ok {
		if rows, err := definitions.ListEdgeKindDefinitions(); err == nil {
			for name, definition := range rows {
				names[definition.Kind] = append(names[definition.Kind], name)
			}
			return names
		}
	}
	if kinds, ok := e.db.storage.(storage.EdgeKindStore); ok {
		if rows, err := kinds.ListEdgeKinds(); err == nil {
			for name, kind := range rows {
				names[kind] = append(names[kind], name)
			}
		}
	}
	return names
}

func graphEdgeMatchesPredicates(sourceID, targetID string, edge graph.Edge, predicates []optimizer.RelationalPredicate, edgeKindNames map[uint8][]string) bool {
	for _, predicate := range predicates {
		if strings.EqualFold(predicate.Column, "type") || strings.EqualFold(predicate.Column, "kind") || strings.EqualFold(predicate.Column, "edge_kind") {
			if names := edgeKindNames[edge.GetKind()]; len(names) > 0 {
				matched := false
				for _, name := range names {
					if scalarPredicateMatches(name, predicate) {
						matched = true
						break
					}
				}
				if !matched {
					return false
				}
				continue
			}
		}
		var actual interface{}
		switch {
		case strings.EqualFold(predicate.Column, "source"), strings.EqualFold(predicate.Column, "src"):
			actual = sourceID
		case strings.EqualFold(predicate.Column, "target"), strings.EqualFold(predicate.Column, "tgt"):
			actual = targetID
		case strings.EqualFold(predicate.Column, "type"), strings.EqualFold(predicate.Column, "kind"), strings.EqualFold(predicate.Column, "edge_kind"):
			if name := graph.EdgeKindName(edge.GetKind()); name != "" {
				actual = name
			} else {
				actual = int64(edge.GetKind())
			}
		case strings.EqualFold(predicate.Column, "weight"):
			actual = edge.Weight
		default:
			return false
		}
		if predicate.NullTest == optimizer.NullTestIsNull || predicate.NullTest == optimizer.NullTestNotNull {
			if predicate.NullTest == optimizer.NullTestIsNull {
				return false
			}
			continue
		}
		if !scalarPredicateMatches(actual, predicate) {
			return false
		}
	}
	return true
}

// executeJoin evaluates relational joins over the epoch-visible row views.
// It deliberately uses the same row path for B-tree, vector, and metadata
// collections, and preserves SQL outer-join null padding. Graph joins are
// handled by executeGraphJoin before this path.
// executeRelationalFullScan handles relational queries when the index doesn't
// support B-tree access (HNSW, Flat, IVFPQ). Iterates all records via ListAll.
func (e *Executor) executeRelationalFullScan(ctx context.Context, col *Collection, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	records, err := recordsVisibleInContext(ctx, col)
	if err != nil {
		return nil, err
	}
	var results []*SearchResult
	for _, rec := range records {
		if !planMatchesRecord(plan, rec) {
			continue
		}
		if !recordMatchesFTSPredicates(rec, plan.FTSPredicates) {
			continue
		}
		results = append(results, &SearchResult{ID: rec.ID, Score: 1.0, Metadata: rec.Metadata, Ordinal: rec.Ordinal})
	}
	return e.buildSelectResults(ctx, col, results, plan), nil
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
	leftRecords, err := recordsVisibleInContext(ctx, leftCol)
	if err != nil {
		return nil, err
	}
	leftAlias := plan.CollectionName
	if len(plan.Joins) > 0 && plan.Joins[0].LeftAlias != "" {
		leftAlias = plan.Joins[0].LeftAlias
	}
	rows := make([]sqlJoinRow, 0, len(leftRecords))
	for i := range leftRecords {
		record := &leftRecords[i]
		rows = append(rows, sqlJoinRow{Sources: map[string]*Record{leftAlias: record}, Schemas: map[string][]string{leftAlias: collectionColumns(leftCol)}, BaseAlias: leftAlias})
	}

	for _, join := range plan.Joins {
		rightCol, err := e.db.GetCollection(join.CollectionName)
		if err != nil {
			return nil, err
		}
		rightRecords, err := recordsVisibleInContext(ctx, rightCol)
		if err != nil {
			return nil, err
		}
		rows = applyRelationalJoin(rows, rightRecords, collectionColumns(rightCol), join)
	}

	results := make([]*SearchResult, 0, len(rows))
	for _, row := range rows {
		if len(plan.PredicateAlternatives) > 0 {
			if !joinedRowMatchesAlternatives(row, plan.PredicateAlternatives, leftAlias) {
				continue
			}
		} else if !joinedRowMatchesPredicates(row, plan.Predicates, leftAlias) {
			continue
		}
		result := row.searchResult()
		for _, ref := range plan.ProjectionRefs {
			if value, ok := recordMetadataValue(result.Metadata, ref.SourceName); ok {
				result.Metadata[ref.OutputName] = value
			}
		}
		results = append(results, result)
	}
	out := &SearchResults{Results: results, Total: len(results), Columns: plan.Projections}
	if plan.Distinct {
		out.Results = distinctSearchResults(out.Results, plan.Projections)
	}
	if plan.OrderBy != "" {
		e.applyOrderBy(out, plan)
	}
	if plan.Offset > 0 {
		if plan.Offset >= len(out.Results) {
			out.Results = nil
		} else {
			out.Results = out.Results[plan.Offset:]
		}
	}
	if plan.Limit > 0 && len(out.Results) > plan.Limit {
		out.Results = out.Results[:plan.Limit]
	}
	out.Total = len(out.Results)
	return out, nil
}

// sqlJoinRow keeps each relation separate while a multi-join is built. A nil
// source is a SQL NULL-padded side of an outer join; its schema is retained so
// projected columns remain present in the row description.
type sqlJoinRow struct {
	Sources   map[string]*Record
	Schemas   map[string][]string
	BaseAlias string
}

func (r sqlJoinRow) clone() sqlJoinRow {
	out := sqlJoinRow{Sources: make(map[string]*Record, len(r.Sources)+1), Schemas: make(map[string][]string, len(r.Schemas)+1), BaseAlias: r.BaseAlias}
	for alias, record := range r.Sources {
		out.Sources[alias] = record
	}
	for alias, columns := range r.Schemas {
		out.Schemas[alias] = append([]string(nil), columns...)
	}
	return out
}

func (r sqlJoinRow) searchResult() *SearchResult {
	metadata := make(map[string]interface{})
	resultID := ""
	if base := r.Sources[r.BaseAlias]; base != nil {
		resultID = base.ID
	}
	aliases := make([]string, 0, len(r.Sources))
	for alias := range r.Sources {
		aliases = append(aliases, alias)
	}
	sort.Strings(aliases)
	// Always materialize the base relation first. Go map iteration order must
	// never decide which relation wins the unqualified `id`/metadata slot or
	// how ORDER BY id behaves on a joined result.
	if len(aliases) > 0 && r.BaseAlias != "" {
		for i, alias := range aliases {
			if alias == r.BaseAlias {
				aliases[0], aliases[i] = aliases[i], aliases[0]
				break
			}
		}
	}
	for _, alias := range aliases {
		record := r.Sources[alias]
		for _, column := range r.Schemas[alias] {
			if _, exists := metadata[column]; !exists {
				metadata[column] = nil
			}
		}
		if record == nil {
			continue
		}
		if resultID == "" {
			resultID = record.ID
		}
		if _, exists := metadata["id"]; !exists || metadata["id"] == nil {
			metadata["id"] = record.ID
		}
		for key, value := range record.Metadata {
			if _, exists := metadata[key]; !exists || metadata[key] == nil {
				metadata[key] = value
			}
		}
	}
	return &SearchResult{ID: resultID, Score: 1, Metadata: metadata}
}

func collectionColumns(col *Collection) []string {
	if col == nil {
		return nil
	}
	cfg := col.Config()
	columns := make([]string, 0, len(cfg.MetadataSchema)+1)
	// The physical record id is exposed exactly once. CREATE TABLE plans may
	// also retain an `id` entry in MetadataSchema (it is the declared primary
	// key), so skip that duplicate when constructing SELECT * / pgwire columns.
	columns = append(columns, "id")
	for name := range cfg.MetadataSchema {
		if strings.EqualFold(name, "id") {
			continue
		}
		columns = append(columns, name)
	}
	sort.Strings(columns[1:])
	return columns
}

// collectionColumnTypes returns the catalog types for a relational result in
// the same order as collectionColumns/PhysicalPlan.Projections. SQL metadata
// is stored in its native durable representation (often a string), so wire
// adapters must use the declared catalog type instead of guessing from row
// values. This is especially important for BOOLEAN, TIMESTAMP, and JSON.
func collectionColumnTypes(col *Collection, columns []string) []uint16 {
	if col == nil || len(columns) == 0 {
		return nil
	}
	types := make([]uint16, len(columns))
	cfg := col.Config()
	var table *catalog.TableDef
	if col.db != nil && col.db.catalog != nil {
		table, _ = col.db.catalog.GetTable(catalog.HashIdentifier(col.name))
	}
	for i, name := range columns {
		if table != nil {
			if column, err := col.db.catalog.GetColumn(table, catalog.HashIdentifier(name)); err == nil {
				types[i] = column.Type
				continue
			}
		}
		if strings.EqualFold(name, "id") {
			types[i] = catalog.TypeString
			continue
		}
		for field, fieldType := range cfg.MetadataSchema {
			if strings.EqualFold(field, name) {
				types[i] = metadataFieldTypeToCatalogType(fieldType)
				break
			}
		}
	}
	return types
}

func metadataFieldTypeToCatalogType(fieldType FieldType) uint16 {
	switch fieldType {
	case IntField:
		return catalog.TypeInt
	case FloatField:
		return catalog.TypeFloat
	case BoolField:
		return catalog.TypeBool
	case TimeField:
		return catalog.TypeTimestamp
	case JSONField:
		return catalog.TypeJSON
	case JSONBField:
		return catalog.TypeJSONB
	case BigIntField:
		return catalog.TypeBigInt
	default:
		return catalog.TypeString
	}
}

func applyRelationalJoin(left []sqlJoinRow, right []Record, rightColumns []string, join optimizer.JoinPlan) []sqlJoinRow {
	rightAlias := join.RightAlias
	if rightAlias == "" {
		rightAlias = join.CollectionName
	}
	next := make([]sqlJoinRow, 0)
	rightMatched := make([]bool, len(right))
	for _, leftRow := range left {
		matched := false
		for j := range right {
			rightRow := right[j]
			if !joinRowsMatch(leftRow, rightRow, join) {
				continue
			}
			matched = true
			rightMatched[j] = true
			joined := leftRow.clone()
			joined.Sources[rightAlias] = &rightRow
			joined.Schemas[rightAlias] = rightColumns
			next = append(next, joined)
		}
		if !matched && (join.JoinType == uint8(parser.JoinLeft) || join.JoinType == uint8(parser.JoinFull)) {
			joined := leftRow.clone()
			joined.Sources[rightAlias] = nil
			joined.Schemas[rightAlias] = rightColumns
			next = append(next, joined)
		}
	}
	if join.JoinType == uint8(parser.JoinRight) || join.JoinType == uint8(parser.JoinFull) {
		for j := range right {
			if rightMatched[j] {
				continue
			}
			joined := sqlJoinRow{Sources: make(map[string]*Record), Schemas: make(map[string][]string), BaseAlias: ""}
			// Preserve all prior aliases as NULL for an unmatched RIGHT/FULL
			// row. The first existing row gives us the complete alias set.
			if len(left) > 0 {
				joined.BaseAlias = left[0].BaseAlias
				for alias, columns := range left[0].Schemas {
					joined.Sources[alias] = nil
					joined.Schemas[alias] = append([]string(nil), columns...)
				}
			}
			joined.Sources[rightAlias] = &right[j]
			joined.Schemas[rightAlias] = rightColumns
			next = append(next, joined)
		}
	}
	return next
}

func joinRowsMatch(left sqlJoinRow, right Record, join optimizer.JoinPlan) bool {
	if join.JoinType == uint8(parser.JoinCross) {
		return true
	}
	if len(join.RightPredicates) > 0 && !recordMatchesPredicates(right, join.RightPredicates) {
		return false
	}
	if join.LeftColumn == "" || join.RightColumn == "" {
		return false
	}
	leftRecord := left.Sources[join.LeftAlias]
	if leftRecord == nil {
		return false
	}
	leftValue, leftOK := joinRecordValue(*leftRecord, join.LeftColumn)
	rightValue, rightOK := joinRecordValue(right, join.RightColumn)
	return leftOK && rightOK && leftValue == rightValue
}

func joinedRowMatchesPredicates(row sqlJoinRow, predicates []optimizer.RelationalPredicate, defaultAlias string) bool {
	for _, predicate := range predicates {
		var result *SearchResult
		if predicate.Alias != "" {
			record := row.Sources[predicate.Alias]
			if record == nil {
				result = &SearchResult{Metadata: map[string]interface{}{}}
			} else {
				result = &SearchResult{ID: record.ID, Metadata: record.Metadata}
			}
		} else {
			result = row.searchResult()
		}
		if !predicateMatches(result, predicate) {
			return false
		}
	}
	return true
}

func joinRecordValue(record Record, column string) (string, bool) {
	if strings.EqualFold(column, "id") {
		return record.ID, record.ID != ""
	}
	for name, value := range record.Metadata {
		if strings.EqualFold(name, column) {
			if value == nil {
				return "", false
			}
			return recordMetaToString(value), true
		}
	}
	return "", false
}

func joinResultMetadata(left, right Record) map[string]interface{} {
	metadata := cloneMetadata(left.Metadata)
	if metadata == nil {
		metadata = make(map[string]interface{})
	}
	for key, value := range right.Metadata {
		metadata[key] = value
	}
	return metadata
}

// executeGraphJoin implements JOIN MATCH: for each row of the left (FROM)
// collection, resolve the row's key to a graph node and run a BFS over the
// match-path edges. Each reached vertex emits a joined row (leftKey|vertexID).
// LEFT JOIN emits left rows even when no vertex is reached.
func (e *Executor) executeGraphJoin(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	if graphPlanHasEdgeProjection(plan) && !graphPlanSupportsSingleHopEdgeProjection(plan) {
		return nil, fmt.Errorf("edge projections require a single-hop JOIN MATCH")
	}
	// Epoch guard: route JOIN MATCH through epoch-aware path when inside an epoch.
	if epoch := epochFromContext(ctx); epoch != nil {
		if len(plan.GraphProjections) > 0 && graphPlanSupportsSingleHopEdgeProjection(plan) {
			return e.executeProjectedGraphJoinEpoch(ctx, plan, epoch)
		}
		if graphJoinsFormCommonNeighbor(plan.GraphJoins) {
			return e.executeCommonNeighborGraphJoinEpoch(ctx, plan, epoch)
		}
		if len(plan.GraphJoins) > 1 && graphJoinsFormChain(plan.GraphJoins) {
			return e.executeChainedGraphJoinEpoch(ctx, plan, epoch)
		}
		return e.executeGraphJoinEpoch(ctx, plan, epoch)
	}
	if len(plan.GraphProjections) > 0 && graphPlanSupportsSingleHopEdgeProjection(plan) {
		return e.executeProjectedGraphJoin(ctx, plan)
	}
	if graphJoinsFormCommonNeighbor(plan.GraphJoins) {
		return e.executeCommonNeighborGraphJoin(ctx, plan)
	}
	if len(plan.GraphJoins) > 1 && graphJoinsFormChain(plan.GraphJoins) {
		return e.executeChainedGraphJoin(ctx, plan)
	}

	leftCol, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
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
	leftRecords, err := recordsVisibleInContext(ctx, leftCol)
	if err != nil {
		return nil, err
	}
	recordsByID := make(map[string]Record, len(leftRecords))
	for i := range leftRecords {
		recordsByID[leftRecords[i].ID] = leftRecords[i]
	}

	var results []*SearchResult
	for _, gjp := range plan.GraphJoins {
		isLeftJoin := gjp.JoinType == 1 // parser.JoinLeft
		sourcePredicates := graphJoinSourcePredicates(plan.Predicates, gjp, plan.CollectionName)

		// Convert optimizer.GraphEdgePlan to graph.EdgePlan
		// (the graph join owns the edge plans; plan.GraphEdges is populated
		// only for WHERE MATCH/standalone graph plans).
		edges := make([]EdgePlan, len(gjp.GraphEdges))
		totalMinDepth := 0
		for i, gep := range gjp.GraphEdges {
			edges[i] = graphEdgePlanForTraversal(gep)
			totalMinDepth += edges[i].Min
		}

		for _, leftRecord := range leftRecords {
			leftKey := leftRecord.ID
			if len(sourcePredicates) > 0 && !recordMatchesPredicates(leftRecord, sourcePredicates) {
				continue
			}

			// Resolve this row's key to a graph node (the anchor).
			nodeID, err := e.db.GetNodeID(ctx, plan.CollectionName, leftKey)
			if err != nil {
				// Row is not a graph node — no traversal possible.
				if isLeftJoin {
					resultID := leftKey + "|"
					results = append(results, &SearchResult{
						ID: resultID, Score: 1.0,
						Metadata: graphJoinProjectionMetadata(leftRecord, nil, resultID, gjp, plan),
					})
				}
				continue
			}

			seedID := nodeID
			seen := make(map[uint64]bool) // nodeID → reached via traversal

			if err := g.BFSPattern(nodeID, edges, gjp.MaxHops, func(vid uint64, band int, step int) bool {
				trackSQLGraphExpansion(ctx, 1)
				// BFSPattern visits intermediate states as well as final
				// matches. Keep valid intermediate-band rows (the existing
				// JOIN MATCH contract exposes them), but exclude states below
				// the current band's minimum hop count. Without this check a
				// single-band *3..5 pattern incorrectly emitted 0, 1, and 2
				// hop rows as matches.
				if step < edges[band].Min {
					return plan.Limit <= 0 || len(results) < plan.Limit
				}
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
				terminal, terminalOK := recordsByID[recID]
				if !terminalOK {
					continue
				}
				if len(gjp.TerminalPredicates) > 0 && !recordMatchesPredicates(terminal, gjp.TerminalPredicates) {
					continue
				}
				if len(plan.PredicateAlternatives) > 0 && !graphJoinMatchesAlternatives(plan, map[string]Record{
					gjp.LeftAlias:     leftRecord,
					gjp.TerminalAlias: terminal,
				}, gjp.LeftAlias) {
					continue
				}
				resultID := leftKey + "|" + recID
				results = append(results, &SearchResult{
					ID: resultID, Score: 1.0,
					Metadata: graphJoinProjectionMetadata(leftRecord, &terminal, resultID, gjp, plan),
				})
				emitted = true
				if plan.Limit > 0 && len(results) >= plan.Limit {
					break
				}
			}
			if !emitted && isLeftJoin {
				resultID := leftKey + "|"
				results = append(results, &SearchResult{
					ID: resultID, Score: 1.0,
					Metadata: graphJoinProjectionMetadata(leftRecord, nil, resultID, gjp, plan),
				})
			}

			if plan.Limit > 0 && len(results) >= plan.Limit {
				break
			}
		}
		if plan.Limit > 0 && len(results) >= plan.Limit {
			break
		}
	}
	out := &SearchResults{Results: results, Total: len(results), Columns: plan.Projections, ColumnTypes: graphProjectionColumnTypes(plan)}
	if plan.Distinct {
		out.Results = distinctSearchResults(out.Results, plan.Projections)
	}
	if plan.Offset > 0 {
		if plan.Offset >= len(out.Results) {
			out.Results = nil
		} else {
			out.Results = out.Results[plan.Offset:]
		}
	}
	if plan.Limit > 0 && len(out.Results) > plan.Limit {
		out.Results = out.Results[:plan.Limit]
	}
	out.Total = len(out.Results)
	return out, nil
}

// graphJoinsFormChain reports whether each JOIN MATCH consumes the terminal
// vertex of the previous one. A chained query is a relational pipeline:
//
//	FROM users me
//	JOIN MATCH (me)-[:FOLLOWS]->(mid)
//	JOIN MATCH (mid)-[:FOLLOWS]->(tgt)
//
// It must not be evaluated as two independent traversals from the FROM rows.
func graphJoinsFormChain(joins []optimizer.GraphJoinPlan) bool {
	if len(joins) < 2 {
		return false
	}
	for i := 1; i < len(joins); i++ {
		if joins[i-1].TerminalAlias == "" || joins[i].LeftAlias == "" ||
			!strings.EqualFold(joins[i-1].TerminalAlias, joins[i].LeftAlias) {
			return false
		}
	}
	return true
}

// graphJoinsFormCommonNeighbor identifies the two-stage graph join used for
// common-neighbor recommendations:
//
//	FROM people src
//	JOIN MATCH (src)-[]->(shared)
//	JOIN MATCH (origin)-[]->(shared)
//
// This is deliberately separate from graphJoinsFormChain. The second stage
// starts at an independent graph anchor and joins on the repeated terminal
// alias; treating it as a chain would traverse from the first shared node and
// produce the wrong semantics.
func graphJoinsFormCommonNeighbor(joins []optimizer.GraphJoinPlan) bool {
	if len(joins) != 2 {
		return false
	}
	first, second := joins[0], joins[1]
	if first.JoinType != 0 || second.JoinType != 0 {
		return false
	}
	if first.LeftAlias == "" || second.LeftAlias == "" ||
		first.TerminalAlias == "" || second.TerminalAlias == "" {
		return false
	}
	return !strings.EqualFold(first.LeftAlias, second.LeftAlias) &&
		strings.EqualFold(first.TerminalAlias, second.TerminalAlias) &&
		strings.EqualFold(first.LeftCollection, second.LeftCollection)
}

// executeCommonNeighborGraphJoin evaluates two independent one-hop (or
// bounded-path) traversals as a relational intersection. The origin-side
// terminal set is materialized once, then each source-side terminal is tested
// against that set. This preserves the repeated `shared` alias semantics and
// avoids the old N+1 application traversal pattern.
func (e *Executor) executeCommonNeighborGraphJoin(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	leftCol, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
	}
	g := leftCol.GetGraph()
	if g == nil {
		return nil, fmt.Errorf("JOIN MATCH left collection %q has no graph", plan.CollectionName)
	}
	records, err := recordsVisibleInContext(ctx, leftCol)
	if err != nil {
		return nil, err
	}
	if len(records) == 0 {
		return materializeChainedGraphJoinRows(nil, plan), nil
	}
	recordsByID := make(map[string]Record, len(records))
	for _, record := range records {
		recordsByID[record.ID] = record
	}

	first, second := plan.GraphJoins[0], plan.GraphJoins[1]
	firstSourcePredicates := graphJoinSourcePredicates(plan.Predicates, first, plan.CollectionName)
	originSourcePredicates := graphJoinSourcePredicates(plan.Predicates, second, plan.CollectionName)
	terminalPredicates := graphJoinTerminalPredicates(plan.Predicates, first)
	if len(terminalPredicates) == 0 {
		terminalPredicates = graphJoinTerminalPredicates(plan.Predicates, second)
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

	originTerminals := make(map[uint64][]Record)
	originEdges := graphJoinEdges(second)
	for _, origin := range records {
		if len(originSourcePredicates) > 0 && !recordMatchesPredicates(origin, originSourcePredicates) {
			continue
		}
		originNode, lookupErr := e.db.GetNodeID(ctx, plan.CollectionName, origin.ID)
		if lookupErr != nil {
			continue
		}
		terminalIDs, traverseErr := collectGraphJoinTerminals(g, originNode, originEdges, second.MaxHops, second.TerminalLabel, bitset, frontier)
		if traverseErr != nil {
			return nil, traverseErr
		}
		for _, terminalNode := range terminalIDs {
			originTerminals[terminalNode] = append(originTerminals[terminalNode], origin)
		}
	}
	if len(originTerminals) == 0 {
		return materializeChainedGraphJoinRows(nil, plan), nil
	}

	rows := make([]chainedGraphJoinRow, 0)
	sourceEdges := graphJoinEdges(first)
	for _, source := range records {
		if len(firstSourcePredicates) > 0 && !recordMatchesPredicates(source, firstSourcePredicates) {
			continue
		}
		sourceNode, lookupErr := e.db.GetNodeID(ctx, plan.CollectionName, source.ID)
		if lookupErr != nil {
			continue
		}
		terminalIDs, traverseErr := collectGraphJoinTerminals(g, sourceNode, sourceEdges, first.MaxHops, first.TerminalLabel, bitset, frontier)
		if traverseErr != nil {
			return nil, traverseErr
		}
		for _, terminalNode := range terminalIDs {
			origins := originTerminals[terminalNode]
			if len(origins) == 0 {
				continue
			}
			_, terminalID, resolveErr := e.db.ResolveNodeID(ctx, terminalNode)
			if resolveErr != nil {
				continue
			}
			terminal, ok := recordsByID[terminalID]
			if !ok || (len(terminalPredicates) > 0 && !recordMatchesPredicates(terminal, terminalPredicates)) {
				continue
			}
			for _, origin := range origins {
				aliases := map[string]Record{
					first.LeftAlias:     source,
					second.LeftAlias:    origin,
					first.TerminalAlias: terminal,
				}
				if len(plan.PredicateAlternatives) > 0 && !graphJoinMatchesAlternatives(plan, aliases, first.LeftAlias) {
					continue
				}
				rows = append(rows, chainedGraphJoinRow{base: source, aliases: aliases})
			}
		}
	}
	return materializeChainedGraphJoinRows(rows, plan), nil
}

func graphJoinEdges(join optimizer.GraphJoinPlan) []EdgePlan {
	edges := make([]EdgePlan, len(join.GraphEdges))
	for i, edge := range join.GraphEdges {
		edges[i] = graphEdgePlanForTraversal(edge)
	}
	return edges
}

func collectGraphJoinTerminals(g Graph, start uint64, edges []EdgePlan, maxHops int, label string, bitset *graph.Bitset, frontier *graph.FrontierBuf) ([]uint64, error) {
	if len(edges) == 0 {
		return nil, nil
	}
	allowed := map[uint64]struct{}(nil)
	if label != "" {
		allowed = make(map[uint64]struct{})
		for _, nodeID := range g.GetLabelNodes(label) {
			allowed[nodeID] = struct{}{}
		}
	}
	terminals := make([]uint64, 0)
	seen := make(map[uint64]struct{})
	err := g.BFSPattern(start, edges, maxHops, func(nodeID uint64, band int, step int) bool {
		if band != len(edges)-1 || step < edges[band].Min ||
			(nodeID == start && band == 0 && step == 0 && edges[0].Min > 0) {
			return true
		}
		if len(allowed) > 0 {
			if _, ok := allowed[nodeID]; !ok {
				return true
			}
		}
		if _, ok := seen[nodeID]; !ok {
			seen[nodeID] = struct{}{}
			terminals = append(terminals, nodeID)
		}
		return true
	}, bitset, frontier)
	bitset.Clear()
	frontier.Clear()
	return terminals, err
}

func (e *Executor) executeCommonNeighborGraphJoinEpoch(ctx context.Context, plan *optimizer.PhysicalPlan, epoch *EpochTx) (*SearchResults, error) {
	leftCol, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
	}
	g := leftCol.GetGraph()
	if g == nil {
		return nil, fmt.Errorf("JOIN MATCH left collection %q has no graph", plan.CollectionName)
	}
	gtx, err := epoch.GraphTxn(plan.CollectionName)
	if err != nil {
		return nil, fmt.Errorf("epoch graph txn for common-neighbor JOIN MATCH: %w", err)
	}
	records, err := recordsVisibleInContext(ctx, leftCol)
	if err != nil {
		return nil, err
	}
	if len(records) == 0 {
		return materializeChainedGraphJoinRows(nil, plan), nil
	}
	recordsByID := make(map[string]Record, len(records))
	for _, record := range records {
		recordsByID[record.ID] = record
	}

	first, second := plan.GraphJoins[0], plan.GraphJoins[1]
	firstSourcePredicates := graphJoinSourcePredicates(plan.Predicates, first, plan.CollectionName)
	originSourcePredicates := graphJoinSourcePredicates(plan.Predicates, second, plan.CollectionName)
	terminalPredicates := graphJoinTerminalPredicates(plan.Predicates, first)
	if len(terminalPredicates) == 0 {
		terminalPredicates = graphJoinTerminalPredicates(plan.Predicates, second)
	}

	originTerminals := make(map[uint64][]Record)
	for _, origin := range records {
		if len(originSourcePredicates) > 0 && !recordMatchesPredicates(origin, originSourcePredicates) {
			continue
		}
		originNode, lookupErr := e.lookupNodeIDInContext(ctx, plan.CollectionName, origin.ID)
		if lookupErr != nil {
			continue
		}
		terminalIDs, traverseErr := collectEpochGraphJoinTerminals(gtx, g, originNode, graphJoinEdges(second), second.MaxHops, second.TerminalLabel)
		if traverseErr != nil {
			return nil, traverseErr
		}
		for _, terminalNode := range terminalIDs {
			originTerminals[terminalNode] = append(originTerminals[terminalNode], origin)
		}
	}
	if len(originTerminals) == 0 {
		return materializeChainedGraphJoinRows(nil, plan), nil
	}

	rows := make([]chainedGraphJoinRow, 0)
	for _, source := range records {
		if len(firstSourcePredicates) > 0 && !recordMatchesPredicates(source, firstSourcePredicates) {
			continue
		}
		sourceNode, lookupErr := e.lookupNodeIDInContext(ctx, plan.CollectionName, source.ID)
		if lookupErr != nil {
			continue
		}
		terminalIDs, traverseErr := collectEpochGraphJoinTerminals(gtx, g, sourceNode, graphJoinEdges(first), first.MaxHops, first.TerminalLabel)
		if traverseErr != nil {
			return nil, traverseErr
		}
		for _, terminalNode := range terminalIDs {
			origins := originTerminals[terminalNode]
			if len(origins) == 0 {
				continue
			}
			collection, terminalID, resolveErr := e.resolveNodeIDInContext(ctx, terminalNode)
			if resolveErr != nil || collection != plan.CollectionName {
				continue
			}
			terminal, ok := recordsByID[terminalID]
			if !ok || (len(terminalPredicates) > 0 && !recordMatchesPredicates(terminal, terminalPredicates)) {
				continue
			}
			for _, origin := range origins {
				aliases := map[string]Record{
					first.LeftAlias:     source,
					second.LeftAlias:    origin,
					first.TerminalAlias: terminal,
				}
				if len(plan.PredicateAlternatives) > 0 && !graphJoinMatchesAlternatives(plan, aliases, first.LeftAlias) {
					continue
				}
				rows = append(rows, chainedGraphJoinRow{base: source, aliases: aliases})
			}
		}
	}
	return materializeChainedGraphJoinRows(rows, plan), nil
}

func collectEpochGraphJoinTerminals(gtx *graph.Txn, g Graph, start uint64, edges []EdgePlan, maxHops int, label string) ([]uint64, error) {
	if gtx == nil || len(edges) == 0 {
		return nil, nil
	}
	allowed := map[uint64]struct{}(nil)
	if label != "" {
		allowed = make(map[uint64]struct{})
		for _, nodeID := range g.GetLabelNodes(label) {
			allowed[nodeID] = struct{}{}
		}
	}
	type state struct {
		node       uint64
		band, step int
	}
	queue := []state{{node: start, band: 0, step: 0}}
	visited := make(map[[3]uint64]struct{})
	terminals := make([]uint64, 0)
	seenTerminals := make(map[uint64]struct{})
	lastBand := len(edges) - 1
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		if current.band < 0 || current.band >= len(edges) {
			continue
		}
		key := [3]uint64{current.node, uint64(current.band), uint64(current.step)}
		if _, seen := visited[key]; seen {
			continue
		}
		visited[key] = struct{}{}
		band := edges[current.band]
		if current.band == lastBand && current.step >= band.Min &&
			!(current.node == start && current.band == 0 && current.step == 0 && band.Min > 0) {
			if len(allowed) == 0 {
				if _, seen := seenTerminals[current.node]; !seen {
					seenTerminals[current.node] = struct{}{}
					terminals = append(terminals, current.node)
				}
			} else if _, ok := allowed[current.node]; ok {
				if _, seen := seenTerminals[current.node]; !seen {
					seenTerminals[current.node] = struct{}{}
					terminals = append(terminals, current.node)
				}
			}
		}
		if current.step >= band.Min && current.band+1 < len(edges) {
			queue = append(queue, state{node: current.node, band: current.band + 1})
		}
		if current.step >= band.Max {
			continue
		}
		var neighbors []Edge
		if band.Dir < 0 {
			neighbors, _ = gtx.InboundNeighborsOverlay(current.node)
		} else if band.Dir > 0 {
			neighbors, _ = gtx.NeighborsOverlay(current.node)
		} else {
			outbound, outErr := gtx.NeighborsOverlay(current.node)
			if outErr != nil {
				return nil, outErr
			}
			inbound, inErr := gtx.InboundNeighborsOverlay(current.node)
			if inErr != nil {
				return nil, inErr
			}
			neighbors = append(outbound, inbound...)
		}
		for _, edge := range neighbors {
			if !band.Matches(edge) {
				continue
			}
			nextBand := current.band
			nextStep := current.step + 1
			if current.band < lastBand && current.step >= band.Min && current.step >= band.Max-1 {
				nextBand++
				nextStep = 0
			}
			queue = append(queue, state{node: edge.Target, band: nextBand, step: nextStep})
		}
	}
	return terminals, nil
}

type chainedGraphJoinRow struct {
	base    Record
	aliases map[string]Record
}

// executeChainedGraphJoin evaluates JOIN MATCH clauses as a sequence of
// relational graph joins. Each stage starts at the prior stage's terminal
// alias, so terminal predicates such as `tgt.id <> $1` are evaluated against
// the actual final vertex rather than the original source row.
func (e *Executor) executeChainedGraphJoin(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	leftCol, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
	}
	g := leftCol.GetGraph()
	if g == nil {
		return nil, fmt.Errorf("JOIN MATCH left collection %q has no graph", plan.CollectionName)
	}
	leftRecords, err := recordsVisibleInContext(ctx, leftCol)
	if err != nil {
		return nil, err
	}

	rows := make([]chainedGraphJoinRow, 0, len(leftRecords))
	first := plan.GraphJoins[0]
	firstSourcePredicates := graphJoinSourcePredicates(plan.Predicates, first, plan.CollectionName)
	for _, record := range leftRecords {
		if len(firstSourcePredicates) > 0 && !recordMatchesPredicates(record, firstSourcePredicates) {
			continue
		}
		rows = append(rows, chainedGraphJoinRow{
			base:    record,
			aliases: map[string]Record{first.LeftAlias: record},
		})
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

	for _, join := range plan.GraphJoins {
		edges := make([]EdgePlan, len(join.GraphEdges))
		for i, gep := range join.GraphEdges {
			edges[i] = graphEdgePlanForTraversal(gep)
		}
		if len(edges) == 0 {
			return &SearchResults{}, nil
		}
		terminalPredicates := graphJoinTerminalPredicates(plan.Predicates, join)
		if len(terminalPredicates) == 0 {
			terminalPredicates = join.TerminalPredicates
		}
		terminalLabels := map[uint64]struct{}(nil)
		if join.TerminalLabel != "" {
			terminalLabels = make(map[uint64]struct{})
			for _, nodeID := range g.GetLabelNodes(join.TerminalLabel) {
				terminalLabels[nodeID] = struct{}{}
			}
		}

		nextRows := make([]chainedGraphJoinRow, 0, len(rows))
		for _, row := range rows {
			anchor, ok := row.aliases[join.LeftAlias]
			if !ok {
				continue
			}
			stageSourcePredicates := graphJoinSourcePredicates(plan.Predicates, join, plan.CollectionName)
			if len(stageSourcePredicates) > 0 && !recordMatchesPredicates(anchor, stageSourcePredicates) {
				continue
			}
			anchorID, lookupErr := e.db.GetNodeID(ctx, plan.CollectionName, anchor.ID)
			if lookupErr != nil {
				if join.JoinType == 1 { // LEFT JOIN MATCH
					nextRows = append(nextRows, row)
				}
				continue
			}

			terminalIDs := make([]uint64, 0)
			seenTerminals := make(map[uint64]struct{})
			if err := g.BFSPattern(anchorID, edges, join.MaxHops, func(nodeID uint64, band int, step int) bool {
				trackSQLGraphExpansion(ctx, 1)
				if band != len(edges)-1 || step < edges[band].Min ||
					(nodeID == anchorID && band == 0 && step == 0 && edges[0].Min > 0) {
					return true
				}
				if len(terminalLabels) > 0 {
					if _, ok := terminalLabels[nodeID]; !ok {
						return true
					}
				}
				if _, seen := seenTerminals[nodeID]; !seen {
					seenTerminals[nodeID] = struct{}{}
					terminalIDs = append(terminalIDs, nodeID)
				}
				return true
			}, bitset, frontier); err != nil {
				return nil, err
			}
			bitset.Clear()
			frontier.Clear()

			emitted := false
			for _, terminalNodeID := range terminalIDs {
				collection, terminalID, resolveErr := e.db.ResolveNodeID(ctx, terminalNodeID)
				if resolveErr != nil || collection != plan.CollectionName {
					continue
				}
				terminal, getErr := leftCol.Get(ctx, terminalID)
				if getErr != nil || (len(terminalPredicates) > 0 && !recordMatchesPredicates(terminal, terminalPredicates)) {
					continue
				}
				aliases := make(map[string]Record, len(row.aliases)+1)
				for alias, record := range row.aliases {
					aliases[alias] = record
				}
				aliases[join.TerminalAlias] = terminal
				nextRows = append(nextRows, chainedGraphJoinRow{base: row.base, aliases: aliases})
				emitted = true
			}
			if !emitted && join.JoinType == 1 { // LEFT JOIN MATCH
				nextRows = append(nextRows, row)
			}
		}
		rows = nextRows
		if len(rows) == 0 {
			return &SearchResults{}, nil
		}
	}

	return materializeChainedGraphJoinRows(rows, plan), nil
}

// executeChainedGraphJoinEpoch is the epoch-overlay counterpart of
// executeChainedGraphJoin. It keeps the same alias pipeline while traversing
// the transaction's staged/live graph view rather than the live graph alone.
func (e *Executor) executeChainedGraphJoinEpoch(ctx context.Context, plan *optimizer.PhysicalPlan, epoch *EpochTx) (*SearchResults, error) {
	leftCol, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
	}
	gtx, err := epoch.GraphTxn(plan.CollectionName)
	if err != nil {
		return nil, fmt.Errorf("epoch graph txn for chained JOIN MATCH: %w", err)
	}
	leftRecords, err := recordsVisibleInContext(ctx, leftCol)
	if err != nil {
		return nil, err
	}
	recordsByID := make(map[string]Record, len(leftRecords))
	for _, record := range leftRecords {
		recordsByID[record.ID] = record
	}

	rows := make([]chainedGraphJoinRow, 0, len(leftRecords))
	first := plan.GraphJoins[0]
	firstSourcePredicates := graphJoinSourcePredicates(plan.Predicates, first, plan.CollectionName)
	for _, record := range leftRecords {
		if len(firstSourcePredicates) > 0 && !recordMatchesPredicates(record, firstSourcePredicates) {
			continue
		}
		rows = append(rows, chainedGraphJoinRow{base: record, aliases: map[string]Record{first.LeftAlias: record}})
	}

	for _, join := range plan.GraphJoins {
		edges := make([]EdgePlan, len(join.GraphEdges))
		for i, gep := range join.GraphEdges {
			edges[i] = graphEdgePlanForTraversal(gep)
		}
		if len(edges) == 0 {
			return &SearchResults{}, nil
		}
		terminalPredicates := graphJoinTerminalPredicates(plan.Predicates, join)
		if len(terminalPredicates) == 0 {
			terminalPredicates = join.TerminalPredicates
		}
		terminalLabels := map[uint64]struct{}(nil)
		if join.TerminalLabel != "" {
			terminalLabels = make(map[uint64]struct{})
			for _, nodeID := range leftCol.GetGraph().GetLabelNodes(join.TerminalLabel) {
				terminalLabels[nodeID] = struct{}{}
			}
		}

		nextRows := make([]chainedGraphJoinRow, 0, len(rows))
		for _, row := range rows {
			anchor, ok := row.aliases[join.LeftAlias]
			if !ok {
				continue
			}
			stageSourcePredicates := graphJoinSourcePredicates(plan.Predicates, join, plan.CollectionName)
			if len(stageSourcePredicates) > 0 && !recordMatchesPredicates(anchor, stageSourcePredicates) {
				continue
			}
			anchorID, lookupErr := e.lookupNodeIDInContext(ctx, plan.CollectionName, anchor.ID)
			if lookupErr != nil {
				if join.JoinType == 1 {
					nextRows = append(nextRows, row)
				}
				continue
			}

			type state struct {
				node       uint64
				band, step int
			}
			queue := []state{{node: anchorID, band: 0, step: 0}}
			visited := make(map[[3]uint64]struct{})
			terminalIDs := make([]uint64, 0)
			seenTerminals := make(map[uint64]struct{})
			lastBand := len(edges) - 1
			for len(queue) > 0 {
				current := queue[0]
				queue = queue[1:]
				if current.band < 0 || current.band >= len(edges) {
					continue
				}
				key := [3]uint64{current.node, uint64(current.band), uint64(current.step)}
				if _, seen := visited[key]; seen {
					continue
				}
				visited[key] = struct{}{}
				band := edges[current.band]
				if current.band == lastBand && current.step >= band.Min &&
					!(current.node == anchorID && current.band == 0 && current.step == 0 && band.Min > 0) {
					if len(terminalLabels) == 0 {
						if _, seen := seenTerminals[current.node]; !seen {
							seenTerminals[current.node] = struct{}{}
							terminalIDs = append(terminalIDs, current.node)
						}
					} else if _, ok := terminalLabels[current.node]; ok {
						if _, seen := seenTerminals[current.node]; !seen {
							seenTerminals[current.node] = struct{}{}
							terminalIDs = append(terminalIDs, current.node)
						}
					}
				}
				if current.step >= band.Max {
					continue
				}

				var neighbors []graph.Edge
				if band.Dir < 0 {
					neighbors, _ = gtx.InboundNeighborsOverlay(current.node)
				} else {
					neighbors, _ = gtx.NeighborsOverlay(current.node)
				}
				for _, neighbor := range neighbors {
					if !band.Matches(neighbor) {
						continue
					}
					nextBand := current.band
					nextStep := current.step + 1
					if current.band < lastBand && current.step >= band.Min && current.step >= band.Max-1 {
						nextBand++
						nextStep = 0
					}
					queue = append(queue, state{node: neighbor.Target, band: nextBand, step: nextStep})
				}
			}

			emitted := false
			for _, terminalNodeID := range terminalIDs {
				collection, terminalID, resolveErr := e.resolveNodeIDInContext(ctx, terminalNodeID)
				if resolveErr != nil || collection != plan.CollectionName {
					continue
				}
				terminal, ok := recordsByID[terminalID]
				if !ok || (len(terminalPredicates) > 0 && !recordMatchesPredicates(terminal, terminalPredicates)) {
					continue
				}
				aliases := make(map[string]Record, len(row.aliases)+1)
				for alias, record := range row.aliases {
					aliases[alias] = record
				}
				aliases[join.TerminalAlias] = terminal
				nextRows = append(nextRows, chainedGraphJoinRow{base: row.base, aliases: aliases})
				emitted = true
			}
			if !emitted && join.JoinType == 1 {
				nextRows = append(nextRows, row)
			}
		}
		rows = nextRows
		if len(rows) == 0 {
			return &SearchResults{}, nil
		}
	}
	return materializeChainedGraphJoinRows(rows, plan), nil
}

func materializeChainedGraphJoinRows(rows []chainedGraphJoinRow, plan *optimizer.PhysicalPlan) *SearchResults {
	finalJoin := plan.GraphJoins[len(plan.GraphJoins)-1]
	out := &SearchResults{Columns: plan.Projections, ColumnTypes: graphProjectionColumnTypes(plan)}
	for _, row := range rows {
		if len(plan.PredicateAlternatives) > 0 && !graphJoinMatchesAlternatives(plan, row.aliases, plan.GraphJoins[0].LeftAlias) {
			continue
		}
		terminal, ok := row.aliases[finalJoin.TerminalAlias]
		resultID := row.base.ID + "|"
		if ok {
			resultID += terminal.ID
		}
		out.Results = append(out.Results, &SearchResult{ID: resultID, Score: 1.0, Metadata: chainedGraphJoinProjectionMetadata(row.aliases, resultID, plan)})
	}
	if plan.Distinct {
		out.Results = distinctSearchResults(out.Results, plan.Projections)
	}
	if plan.Offset > 0 {
		if plan.Offset >= len(out.Results) {
			out.Results = nil
		} else {
			out.Results = out.Results[plan.Offset:]
		}
	}
	if plan.Limit > 0 && len(out.Results) > plan.Limit {
		out.Results = out.Results[:plan.Limit]
	}
	out.Total = len(out.Results)
	return out
}

func graphJoinTerminalPredicates(predicates []optimizer.RelationalPredicate, join optimizer.GraphJoinPlan) []optimizer.RelationalPredicate {
	if join.TerminalAlias == "" {
		return nil
	}
	result := make([]optimizer.RelationalPredicate, 0)
	for _, predicate := range predicates {
		if predicate.Alias != "" && strings.EqualFold(predicate.Alias, join.TerminalAlias) {
			result = append(result, predicate)
		}
	}
	return result
}

func chainedGraphJoinProjectionMetadata(aliases map[string]Record, resultID string, plan *optimizer.PhysicalPlan) map[string]interface{} {
	metadata := make(map[string]interface{}, len(plan.ProjectionRefs)+len(plan.GraphProjections))
	if len(plan.ProjectionRefs) == 0 && len(plan.GraphProjections) == 0 {
		metadata["id"] = resultID
		return metadata
	}
	for _, ref := range plan.ProjectionRefs {
		var record Record
		var ok bool
		if ref.SourceAlias == "" {
			if record, ok = aliases[plan.CollectionName]; !ok {
				for _, candidate := range aliases {
					record, ok = candidate, true
					break
				}
			}
		} else {
			for alias, candidate := range aliases {
				if strings.EqualFold(alias, ref.SourceAlias) {
					record, ok = candidate, true
					break
				}
			}
		}
		if !ok {
			metadata[ref.OutputName] = nil
			continue
		}
		if strings.EqualFold(ref.SourceName, "id") {
			metadata[ref.OutputName] = record.ID
		} else if record.Metadata != nil {
			metadata[ref.OutputName] = record.Metadata[ref.SourceName]
		} else {
			metadata[ref.OutputName] = nil
		}
	}
	var source, target *Record
	if first, ok := aliases[plan.GraphJoins[0].LeftAlias]; ok {
		source = &first
	}
	finalJoin := plan.GraphJoins[len(plan.GraphJoins)-1]
	if final, ok := aliases[finalJoin.TerminalAlias]; ok {
		target = &final
	}
	applyGraphProjectionMetadata(metadata, plan.GraphProjections, source, target, nil, "")
	return metadata
}

func graphJoinSourcePredicates(predicates []optimizer.RelationalPredicate, join optimizer.GraphJoinPlan, collection string) []optimizer.RelationalPredicate {
	result := make([]optimizer.RelationalPredicate, 0, len(predicates))
	for _, predicate := range predicates {
		if predicate.Alias == "" || strings.EqualFold(predicate.Alias, join.LeftAlias) || strings.EqualFold(predicate.Alias, collection) {
			result = append(result, predicate)
		}
	}
	return result
}

func graphJoinProjectionMetadata(left Record, terminal *Record, resultID string, join optimizer.GraphJoinPlan, plan *optimizer.PhysicalPlan) map[string]interface{} {
	metadata := make(map[string]interface{}, len(plan.ProjectionRefs)+len(plan.GraphProjections))
	if len(plan.ProjectionRefs) == 0 && len(plan.GraphProjections) == 0 {
		metadata["id"] = resultID
		return metadata
	}
	for _, ref := range plan.ProjectionRefs {
		var record *Record
		if ref.SourceAlias != "" && strings.EqualFold(ref.SourceAlias, join.TerminalAlias) {
			record = terminal
		} else if ref.SourceAlias == "" || strings.EqualFold(ref.SourceAlias, join.LeftAlias) {
			record = &left
		}
		if record == nil {
			metadata[ref.OutputName] = nil
			continue
		}
		if strings.EqualFold(ref.SourceName, "id") {
			metadata[ref.OutputName] = record.ID
		} else if record.Metadata != nil {
			metadata[ref.OutputName] = record.Metadata[ref.SourceName]
		} else {
			metadata[ref.OutputName] = nil
		}
	}
	applyGraphProjectionMetadata(metadata, plan.GraphProjections, &left, terminal, nil, "")
	return metadata
}

func applyGraphProjectionMetadata(metadata map[string]interface{}, projections []optimizer.GraphProjection, source, target *Record, edge *graph.Edge, fallbackType string) {
	for _, projection := range projections {
		switch projection.Kind {
		case optimizer.GraphProjectionSourceID:
			if source != nil {
				metadata[projection.OutputName] = source.ID
			} else {
				metadata[projection.OutputName] = nil
			}
		case optimizer.GraphProjectionTargetID:
			if target != nil {
				metadata[projection.OutputName] = target.ID
			} else {
				metadata[projection.OutputName] = nil
			}
		case optimizer.GraphProjectionEdgeType:
			typeName := fallbackType
			if typeName == "" && edge != nil {
				typeName = graph.EdgeKindName(edge.GetKind())
			}
			if typeName == "" {
				metadata[projection.OutputName] = nil
			} else {
				metadata[projection.OutputName] = typeName
			}
		case optimizer.GraphProjectionEdgeWeight:
			if edge != nil {
				metadata[projection.OutputName] = edge.Weight
			} else {
				metadata[projection.OutputName] = nil
			}
		}
	}
}

func graphProjectionColumnTypes(plan *optimizer.PhysicalPlan) []uint16 {
	if plan == nil || len(plan.Projections) == 0 {
		return nil
	}
	types := make([]uint16, len(plan.Projections))
	for i, column := range plan.Projections {
		for _, projection := range plan.GraphProjections {
			if !strings.EqualFold(column, projection.OutputName) {
				continue
			}
			switch projection.Kind {
			case optimizer.GraphProjectionEdgeWeight:
				types[i] = catalog.TypeFloat4
			default:
				types[i] = catalog.TypeString
			}
			break
		}
	}
	return types
}

func graphPlanHasEdgeProjection(plan *optimizer.PhysicalPlan) bool {
	if plan == nil {
		return false
	}
	for _, projection := range plan.GraphProjections {
		if projection.Kind == optimizer.GraphProjectionEdgeType || projection.Kind == optimizer.GraphProjectionEdgeWeight {
			return true
		}
	}
	return false
}

func graphPlanSupportsSingleHopEdgeProjection(plan *optimizer.PhysicalPlan) bool {
	if plan == nil || len(plan.GraphJoins) != 1 || len(plan.GraphJoins[0].GraphEdges) != 1 {
		return false
	}
	edge := graphEdgePlanForTraversal(plan.GraphJoins[0].GraphEdges[0])
	return edge.Min == 1 && edge.Max == 1
}

func graphPatternNeighbors(g Graph, nodeID uint64, direction int8) ([]graph.EdgeView, error) {
	if direction > 0 {
		return g.NeighborsWithProperties(nodeID)
	}
	if direction < 0 {
		return g.InboundNeighborsWithProperties(nodeID)
	}
	outbound, err := g.NeighborsWithProperties(nodeID)
	if err != nil {
		return nil, err
	}
	inbound, err := g.InboundNeighborsWithProperties(nodeID)
	if err != nil {
		return nil, err
	}
	seen := make(map[string]struct{}, len(outbound)+len(inbound))
	result := make([]graph.EdgeView, 0, len(outbound)+len(inbound))
	for _, view := range append(outbound, inbound...) {
		key := fmt.Sprintf("%d/%d/%g/%x", view.Edge.Target, view.Edge.GetKind(), view.Edge.Weight, view.Properties)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		result = append(result, view)
	}
	return result, nil
}

func (e *Executor) executeProjectedGraphJoin(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	leftCol, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
	}
	g := leftCol.GetGraph()
	if g == nil {
		return nil, fmt.Errorf("JOIN MATCH left collection %q has no graph", plan.CollectionName)
	}
	if !graphPlanSupportsSingleHopEdgeProjection(plan) {
		return nil, fmt.Errorf("edge projections require a single-hop JOIN MATCH")
	}
	records, err := recordsVisibleInContext(ctx, leftCol)
	if err != nil {
		return nil, err
	}
	recordsByID := make(map[string]Record, len(records))
	for _, record := range records {
		recordsByID[record.ID] = record
	}
	join := plan.GraphJoins[0]
	edgePlan := graphEdgePlanForTraversal(join.GraphEdges[0])
	sourcePredicates := graphJoinSourcePredicates(plan.Predicates, join, plan.CollectionName)
	terminalPredicates := graphJoinTerminalPredicates(plan.Predicates, join)
	allowedLabels := map[uint64]struct{}(nil)
	if join.TerminalLabel != "" {
		allowedLabels = make(map[uint64]struct{})
		for _, nodeID := range g.GetLabelNodes(join.TerminalLabel) {
			allowedLabels[nodeID] = struct{}{}
		}
	}
	results := make([]*SearchResult, 0)
	for _, source := range records {
		if len(sourcePredicates) > 0 && !recordMatchesPredicates(source, sourcePredicates) {
			continue
		}
		sourceNode, lookupErr := e.db.GetNodeID(ctx, plan.CollectionName, source.ID)
		if lookupErr != nil {
			continue
		}
		neighbors, neighborErr := graphPatternNeighbors(g, sourceNode, edgePlan.Dir)
		if neighborErr != nil {
			return nil, neighborErr
		}
		for _, view := range neighbors {
			if !edgePlan.MatchesWithProperties(view.Edge, view.Properties) {
				continue
			}
			collection, targetID, resolveErr := e.db.ResolveNodeID(ctx, view.Edge.Target)
			if resolveErr != nil || collection != plan.CollectionName {
				continue
			}
			target, ok := recordsByID[targetID]
			if !ok || (len(allowedLabels) > 0 && !hasGraphLabel(allowedLabels, view.Edge.Target)) ||
				(len(terminalPredicates) > 0 && !recordMatchesPredicates(target, terminalPredicates)) {
				continue
			}
			aliases := map[string]Record{join.LeftAlias: source, join.TerminalAlias: target}
			if len(plan.PredicateAlternatives) > 0 && !graphJoinMatchesAlternatives(plan, aliases, join.LeftAlias) {
				continue
			}
			resultID := source.ID + "|" + target.ID
			metadata := graphJoinProjectionMetadata(source, &target, resultID, join, plan)
			applyGraphProjectionMetadata(metadata, plan.GraphProjections, &source, &target, &view.Edge, join.GraphEdges[0].EdgeType)
			results = append(results, &SearchResult{ID: resultID, Score: 1, Metadata: metadata})
		}
	}
	out := &SearchResults{Results: results, Total: len(results), Columns: plan.Projections, ColumnTypes: graphProjectionColumnTypes(plan)}
	if plan.Distinct {
		out.Results = distinctSearchResults(out.Results, plan.Projections)
	}
	if plan.Offset > 0 {
		if plan.Offset >= len(out.Results) {
			out.Results = nil
		} else {
			out.Results = out.Results[plan.Offset:]
		}
	}
	if plan.Limit > 0 && len(out.Results) > plan.Limit {
		out.Results = out.Results[:plan.Limit]
	}
	out.Total = len(out.Results)
	return out, nil
}

func hasGraphLabel(labels map[uint64]struct{}, nodeID uint64) bool {
	if len(labels) == 0 {
		return true
	}
	_, ok := labels[nodeID]
	return ok
}

func (e *Executor) executeProjectedGraphJoinEpoch(ctx context.Context, plan *optimizer.PhysicalPlan, epoch *EpochTx) (*SearchResults, error) {
	leftCol, err := e.db.GetCollection(plan.CollectionName)
	if err != nil {
		return nil, err
	}
	g := leftCol.GetGraph()
	if g == nil {
		return nil, fmt.Errorf("JOIN MATCH left collection %q has no graph", plan.CollectionName)
	}
	if !graphPlanSupportsSingleHopEdgeProjection(plan) {
		return nil, fmt.Errorf("edge projections require a single-hop JOIN MATCH")
	}
	gtx, err := epoch.GraphTxn(plan.CollectionName)
	if err != nil {
		return nil, fmt.Errorf("epoch graph txn for projected JOIN MATCH: %w", err)
	}
	records, err := recordsVisibleInContext(ctx, leftCol)
	if err != nil {
		return nil, err
	}
	recordsByID := make(map[string]Record, len(records))
	for _, record := range records {
		recordsByID[record.ID] = record
	}
	join := plan.GraphJoins[0]
	edgePlan := graphEdgePlanForTraversal(join.GraphEdges[0])
	sourcePredicates := graphJoinSourcePredicates(plan.Predicates, join, plan.CollectionName)
	terminalPredicates := graphJoinTerminalPredicates(plan.Predicates, join)
	allowedLabels := map[uint64]struct{}(nil)
	if join.TerminalLabel != "" {
		allowedLabels = make(map[uint64]struct{})
		for _, nodeID := range g.GetLabelNodes(join.TerminalLabel) {
			allowedLabels[nodeID] = struct{}{}
		}
	}
	results := make([]*SearchResult, 0)
	for _, source := range records {
		if len(sourcePredicates) > 0 && !recordMatchesPredicates(source, sourcePredicates) {
			continue
		}
		sourceNode, lookupErr := e.lookupNodeIDInContext(ctx, plan.CollectionName, source.ID)
		if lookupErr != nil {
			continue
		}
		var neighbors []graph.EdgeView
		if edgePlan.Dir > 0 {
			neighbors, err = gtx.NeighborsOverlayWithProperties(sourceNode)
		} else if edgePlan.Dir < 0 {
			neighbors, err = gtx.InboundNeighborsOverlayWithProperties(sourceNode)
		} else {
			neighbors, err = graphPatternNeighborsEpoch(gtx, sourceNode)
		}
		if err != nil {
			return nil, err
		}
		for _, view := range neighbors {
			if !edgePlan.MatchesWithProperties(view.Edge, view.Properties) {
				continue
			}
			collection, targetID, resolveErr := e.resolveNodeIDInContext(ctx, view.Edge.Target)
			if resolveErr != nil || collection != plan.CollectionName {
				continue
			}
			target, ok := recordsByID[targetID]
			if !ok || !hasGraphLabel(allowedLabels, view.Edge.Target) ||
				(len(terminalPredicates) > 0 && !recordMatchesPredicates(target, terminalPredicates)) {
				continue
			}
			aliases := map[string]Record{join.LeftAlias: source, join.TerminalAlias: target}
			if len(plan.PredicateAlternatives) > 0 && !graphJoinMatchesAlternatives(plan, aliases, join.LeftAlias) {
				continue
			}
			resultID := source.ID + "|" + target.ID
			metadata := graphJoinProjectionMetadata(source, &target, resultID, join, plan)
			applyGraphProjectionMetadata(metadata, plan.GraphProjections, &source, &target, &view.Edge, join.GraphEdges[0].EdgeType)
			results = append(results, &SearchResult{ID: resultID, Score: 1, Metadata: metadata})
		}
	}
	out := &SearchResults{Results: results, Total: len(results), Columns: plan.Projections, ColumnTypes: graphProjectionColumnTypes(plan)}
	if plan.Distinct {
		out.Results = distinctSearchResults(out.Results, plan.Projections)
	}
	if plan.Offset > 0 {
		if plan.Offset >= len(out.Results) {
			out.Results = nil
		} else {
			out.Results = out.Results[plan.Offset:]
		}
	}
	if plan.Limit > 0 && len(out.Results) > plan.Limit {
		out.Results = out.Results[:plan.Limit]
	}
	out.Total = len(out.Results)
	return out, nil
}

func graphPatternNeighborsEpoch(txn *graph.Txn, nodeID uint64) ([]graph.EdgeView, error) {
	outbound, err := txn.NeighborsOverlayWithProperties(nodeID)
	if err != nil {
		return nil, err
	}
	inbound, err := txn.InboundNeighborsOverlayWithProperties(nodeID)
	if err != nil {
		return nil, err
	}
	seen := make(map[string]struct{}, len(outbound)+len(inbound))
	result := make([]graph.EdgeView, 0, len(outbound)+len(inbound))
	for _, view := range append(outbound, inbound...) {
		key := fmt.Sprintf("%d/%d/%g/%x", view.Edge.Target, view.Edge.GetKind(), view.Edge.Weight, view.Properties)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		result = append(result, view)
	}
	return result, nil
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

	// Build edge plans. JOIN MATCH stores its edges in GraphJoins; the
	// top-level GraphEdges field is used by WHERE MATCH/standalone graph
	// plans. Accept both shapes so an epoch JOIN MATCH cannot index an empty
	// edge slice (and so staged records use the same path as live records).
	optimizerEdges := plan.GraphEdges
	if len(optimizerEdges) == 0 && len(plan.GraphJoins) > 0 {
		optimizerEdges = plan.GraphJoins[0].GraphEdges
	}
	if len(optimizerEdges) == 0 {
		return &SearchResults{}, nil
	}
	graphEdges := make([]graph.EdgePlan, len(optimizerEdges))
	for i, gep := range optimizerEdges {
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
		graphEdges[i] = graph.EdgePlan{Dir: gep.Direction, Min: minHops, Max: maxHops, Weight: gep.Weight, Predicate: gep.Predicate}
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
			if cur.band < 0 || cur.band >= len(graphEdges) {
				continue
			}

			if cur.step >= graphEdges[cur.band].Min && cur.band == lastBand {
				// Resolve terminal node to record ID.
				_, terminalID, rerr := e.resolveNodeIDInContext(ctx, cur.nid)
				if rerr == nil {
					// Get terminal record data from epoch view.
					records, _ := recordsVisibleInContext(ctx, leftCol)
					for _, r := range records {
						if r.ID == terminalID {
							aliases := map[string]Record{}
							if len(plan.GraphJoins) > 0 {
								join := plan.GraphJoins[0]
								aliases[join.LeftAlias] = leftRec
								aliases[join.TerminalAlias] = r
							}
							if len(plan.PredicateAlternatives) > 0 && !graphJoinMatchesAlternatives(plan, aliases, plan.CollectionName) {
								break
							}
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
				if cur.band >= len(graphEdges) || !graphEdges[cur.band].Matches(nb) {
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
	case 5:
		return "percentile_cont"
	case 6:
		return "percentile_disc"
	case 7:
		return "mode"
	case 8:
		return "vector_avg"
	default:
		return "count"
	}
}

func aggregateResultValue(funcType uint8, count int64, sum float64, minVal, maxVal string) string {
	switch funcType {
	case 0:
		return strconv.FormatInt(count, 10)
	case 1:
		return fmt.Sprintf("%f", sum)
	case 2:
		if count > 0 {
			return fmt.Sprintf("%f", sum/float64(count))
		}
		return "0"
	case 3:
		return minVal
	case 4:
		return maxVal
	default:
		return strconv.FormatInt(count, 10)
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

func isSystemTableName(name string) bool {
	_, ok := catalog.ResolveSystemTable(name)
	return ok
}

// executeSystemTable handles queries against system tables (pg_class, etc.).
func (e *Executor) executeSystemTable(ctx context.Context, plan *optimizer.PhysicalPlan) (*SearchResults, error) {
	rows, err := e.materializeSystemTableRows(ctx, plan.CollectionName)
	if err != nil {
		return nil, err
	}
	if planHasPredicates(plan) {
		if len(plan.PredicateAlternatives) > 0 {
			filtered := rows[:0]
			for _, row := range rows {
				if searchResultMatchesPlan(plan, row) {
					filtered = append(filtered, row)
				}
			}
			rows = filtered
		} else {
			wrapped := &SearchResults{Results: rows}
			rows = filterByPredicates(wrapped, plan.Predicates).Results
		}
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
		// Preserve authoritative system-table types even for empty results,
		// where row-value inference cannot distinguish GRAPH_NODES.id from an
		// ordinary textual record id.
		var columnTypes []uint16
		if sysDef, ok := catalog.ResolveSystemTable(plan.CollectionName); ok && len(plan.Projections) > 0 {
			columnTypes = make([]uint16, len(plan.Projections))
			for i, name := range plan.Projections {
				if col, err := catalog.ResolveSystemColumn(sysDef.OID, catalog.HashIdentifier(name)); err == nil {
					columnTypes[i] = col.Type
				}
			}
		}
		return &SearchResults{Results: rows, Total: len(rows), Columns: plan.Projections, ColumnTypes: columnTypes}, nil
	default:
		return nil, fmt.Errorf("query kind %d not supported on system table %q", plan.Kind, plan.CollectionName)
	}
}

// materializeSystemTableRows builds in-memory rows for a system table.
func (e *Executor) materializeSystemTableRows(ctx context.Context, tableName string) ([]*SearchResult, error) {
	switch strings.ToLower(tableName) {
	case "pg_class":
		return e.materializePgClass(ctx)
	case "pg_attribute":
		return e.materializePgAttribute(ctx)
	case "pg_type":
		return e.materializePgType(ctx)
	case "pg_namespace":
		return e.materializePgNamespace(ctx)
	case "pg_range", "pg_proc", "pg_constraint", "pg_index", "pg_attrdef":
		return []*SearchResult{}, nil
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

		// Use the same visibility and identity overlay as every other epoch
		// executor path. Iterate would read live storage and would therefore
		// leak post-snapshot records or omit staged inserts.
		records, err := recordsVisibleInContext(ctx, col)
		if err != nil {
			return nil, fmt.Errorf("GRAPH_NODES: listing %s: %w", name, err)
		}
		for _, rec := range records {
			nodeID, nerr := e.lookupNodeIDInContext(ctx, name, rec.ID)
			if nerr != nil || nodeID == 0 {
				continue
			}
			rows = append(rows, &SearchResult{
				ID:    fmt.Sprintf("%d", nodeID),
				Score: 1.0,
				Metadata: map[string]interface{}{
					"id":         nodeID,
					"collection": name,
					"record_id":  rec.ID,
				},
			})
		}
	}
	sort.Slice(rows, func(i, j int) bool {
		left, _ := strconv.ParseUint(rows[i].ID, 10, 64)
		right, _ := strconv.ParseUint(rows[j].ID, 10, 64)
		return left < right
	})
	return rows, nil
}

// computeSystemAggregate computes an aggregate over in-memory system table rows.
// materializePgAttribute returns one row per column across all user tables.
func (e *Executor) materializePgAttribute(ctx context.Context) ([]*SearchResult, error) {
	e.db.mu.RLock()
	cat := e.db.catalog
	e.db.mu.RUnlock()

	if cat == nil {
		return nil, nil
	}

	names, err := e.db.ListCollectionsWithContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("pg_attribute: listing collections: %w", err)
	}

	var rows []*SearchResult
	for _, name := range names {
		tableHash := catalog.HashIdentifier(name)
		table, tableErr := cat.GetTable(tableHash)
		if tableErr != nil {
			continue
		}
		columns := cat.AllColumns(table)
		for i, col := range columns {
			attnum := int64(i + 1)
			notNull := false
			if col.Flags&catalog.ColFlagNotNull != 0 {
				notNull = true
			}
			rows = append(rows, &SearchResult{
				ID:    fmt.Sprintf("%d.%d", table.OID, attnum),
				Score: 1.0,
				Metadata: map[string]interface{}{
					"attrelid":   int64(table.OID),
					"attname":    col.Name,
					"atttypid":   int64(catalog.ColumnTypeToPGOID(col.Type)),
					"attnum":     attnum,
					"attnotnull": notNull,
				},
			})
		}
	}
	return rows, nil
}

// materializePgType returns rows for known PostgreSQL type mappings.
func (e *Executor) materializePgType(ctx context.Context) ([]*SearchResult, error) {
	type pgTypeRow struct {
		OID     int64
		TypName string
		TypLen  int64
	}
	types := []pgTypeRow{
		{23, "int4", 4},
		{20, "int8", 8},
		{21, "int2", 2},
		{700, "float4", 4},
		{701, "float8", 8},
		{25, "text", -1},
		{1043, "varchar", -1},
		{1042, "bpchar", -1},
		{19, "name", 64},
		{16, "bool", 1},
		{1114, "timestamp", 8},
		{1184, "timestamptz", 8},
		{1082, "date", 4},
		{1021, "_float4", -1},
		{1022, "_float8", -1},
		{1009, "_text", -1},
		{1007, "_int4", -1},
		{114, "json", -1},
		{3802, "jsonb", -1},
		{2950, "uuid", 16},
		{2951, "_uuid", -1},
	}
	rows := make([]*SearchResult, len(types))
	for i, t := range types {
		rows[i] = &SearchResult{
			ID:    t.TypName,
			Score: 1.0,
			Metadata: map[string]interface{}{
				"oid":     t.OID,
				"typname": t.TypName,
				"typlen":  t.TypLen,
			},
		}
	}
	return rows, nil
}

// materializePgNamespace returns the three standard PostgreSQL namespaces.
func (e *Executor) materializePgNamespace(ctx context.Context) ([]*SearchResult, error) {
	return []*SearchResult{
		{
			ID:    "pg_catalog",
			Score: 1.0,
			Metadata: map[string]interface{}{
				"oid":      int64(11),
				"nspname":  "pg_catalog",
				"nspowner": int64(10),
			},
		},
		{
			ID:    "public",
			Score: 1.0,
			Metadata: map[string]interface{}{
				"oid":      int64(2200),
				"nspname":  "public",
				"nspowner": int64(10),
			},
		},
		{
			ID:    "information_schema",
			Score: 1.0,
			Metadata: map[string]interface{}{
				"oid":      int64(13371),
				"nspname":  "information_schema",
				"nspowner": int64(10),
			},
		},
	}, nil
}

func (e *Executor) computeSystemAggregate(rows []*SearchResult, plan *optimizer.PhysicalPlan) *SearchResults {
	colName := aggregateColumnName(plan.AggregateFunc)
	var resultType uint16 = catalog.TypeBigInt // COUNT returns bigint.
	switch plan.AggregateFunc {
	case 1, 2: // SUM, AVG return double precision in the SQL executor.
		resultType = catalog.TypeFloat
	case 3, 4: // MIN, MAX preserve the source system-column type.
		if sysDef, ok := catalog.ResolveSystemTable(plan.CollectionName); ok {
			if col, err := catalog.ResolveSystemColumn(sysDef.OID, catalog.HashIdentifier(plan.AggregateColumn)); err == nil {
				resultType = col.Type
			}
		}
	}
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
		Results:     []*SearchResult{{ID: fmt.Sprintf("%v", resultValue), Score: 1.0, Metadata: map[string]interface{}{colName: resultValue}}},
		Total:       1,
		Columns:     []string{colName},
		ColumnTypes: []uint16{resultType},
	}
}
