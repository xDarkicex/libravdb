package libravdb

import (
	"context"
	"fmt"
	"sort"

	"github.com/xDarkicex/libravdb/internal/graph"
)

// =============================================================================
// Public API types
// =============================================================================

// LeidenMatchDirection controls traversal direction from seeds.
type LeidenMatchDirection uint8

const (
	LeidenMatchOutbound LeidenMatchDirection = iota
	LeidenMatchInbound
)

// LeidenMatchSpec describes a graph MATCH request over epoch-visible topology.
// It is the Go-native representation consumed by a future SQL MATCH operator.
type LeidenMatchSpec struct {
	Collection  string
	SeedNodeIDs []uint64
	EdgeKinds   []uint8

	MinHops int
	MaxHops int

	Direction LeidenMatchDirection
}

// LeidenMatchResult pairs the full Leiden community detection result with the
// deterministic set of target node IDs reached in the MATCH traversal.
type LeidenMatchResult struct {
	Collection     string
	LeidenResult   *EpochLeidenResult
	MatchedNodeIDs []uint64
}

// =============================================================================
// ComputeLeidenFromMatch
// =============================================================================

// ComputeLeidenFromMatch performs an epoch-visible graph traversal from seed
// nodes respecting edge-kind filtering, direction, and inclusive hop interval,
// then runs Leiden community detection over the bounded local graph.
//
// MatchedNodeIDs contains only nodes whose shortest-path length satisfies
// MinHops ≤ depth ≤ MaxHops. Seeds are included only when MinHops == 0.
// Traversal expands while depth < MaxHops through the epoch overlay (pinned
// snapshot + staged ops).
func (e *EpochTx) ComputeLeidenFromMatch(
	ctx context.Context,
	spec LeidenMatchSpec,
	opts EpochLeidenOptions,
) (*LeidenMatchResult, error) {
	if err := spec.validate(e); err != nil {
		return nil, err
	}

	// Deduplicate and sort seed IDs for determinism.
	seeds := dedupAndSortNodeIDs(spec.SeedNodeIDs)

	// Acquire the collection-specific epoch graph transaction.
	gtx, err := e.GraphTxn(spec.Collection)
	if err != nil {
		return nil, fmt.Errorf("epoch graph txn for %q: %w", spec.Collection, err)
	}

	// Build edge-kind set for O(1) filtering.
	kindSet := makeEdgeKindSet(spec.EdgeKinds)

	// BFS: depth[0] = seeds, expand while depth < spec.MaxHops.
	type bfsEntry struct {
		nodeID uint64
		depth  int
	}
	queue := make([]bfsEntry, 0, len(seeds))
	depth := make(map[uint64]int, len(seeds))

	for _, seed := range seeds {
		depth[seed] = 0
		queue = append(queue, bfsEntry{seed, 0})
	}

	// Standard BFS — front pointer avoids slice shifting.
	front := 0
	for front < len(queue) {
		cur := queue[front]
		front++

		if cur.depth >= spec.MaxHops {
			continue
		}

		// Expand neighbors.
		var neighbors []graph.Edge
		var nerr error
		if spec.Direction == LeidenMatchInbound {
			neighbors, nerr = gtx.InboundNeighborsOverlay(cur.nodeID)
		} else {
			neighbors, nerr = gtx.NeighborsOverlay(cur.nodeID)
		}
		if nerr != nil {
			continue
		}

		// Filter by edge kind and sort by (target, kind) for determinism.
		filtered := filterEdgesByKind(neighbors, kindSet)
		sortEdgesByTargetThenKind(filtered)

		for _, nb := range filtered {
			if _, visited := depth[nb.Target]; visited {
				continue // shortest depth already set
			}
			depth[nb.Target] = cur.depth + 1
			queue = append(queue, bfsEntry{nb.Target, cur.depth + 1})
		}
	}

	// Collect matched nodes: MinHops ≤ depth ≤ MaxHops.
	matched := make([]uint64, 0)
	for nodeID, d := range depth {
		if d >= spec.MinHops && d <= spec.MaxHops {
			matched = append(matched, nodeID)
		}
	}
	sort.Slice(matched, func(i, j int) bool { return matched[i] < matched[j] })

	// Collect all nodes in the local graph (seeds + all reached nodes) for Leiden.
	allNodes := make(map[uint64]bool, len(depth))
	for nodeID := range depth {
		allNodes[nodeID] = true
	}
	localSeeds := make([]uint64, 0, len(allNodes))
	for n := range allNodes {
		localSeeds = append(localSeeds, n)
	}
	sort.Slice(localSeeds, func(i, j int) bool { return localSeeds[i] < localSeeds[j] })

	if len(localSeeds) == 0 {
		return &LeidenMatchResult{
			Collection: spec.Collection,
			LeidenResult: &EpochLeidenResult{
				Communities: []EpochCommunity{},
			},
			MatchedNodeIDs: matched,
		}, nil
	}

	// Run Leiden over the bounded local graph rooted at the seed closure.
	// Use opts.EdgeKinds for Leiden construction unless explicitly empty,
	// in which case inherit from spec.
	leidenOpts := opts
	if len(leidenOpts.EdgeKinds) == 0 && len(spec.EdgeKinds) > 0 {
		leidenOpts.EdgeKinds = append([]uint8(nil), spec.EdgeKinds...)
	}
	leidenOpts.Seeds = localSeeds
	if leidenOpts.ExpansionHops <= 0 || leidenOpts.ExpansionHops > spec.MaxHops {
		leidenOpts.ExpansionHops = spec.MaxHops
	}

	result, err := e.ComputeLeiden(ctx, leidenOpts)
	if err != nil {
		return nil, fmt.Errorf("ComputeLeiden over match graph: %w", err)
	}

	return &LeidenMatchResult{
		Collection:     spec.Collection,
		LeidenResult:   result,
		MatchedNodeIDs: matched,
	}, nil
}

// =============================================================================
// Validation
// =============================================================================

func (spec LeidenMatchSpec) validate(e *EpochTx) error {
	if spec.Collection == "" {
		return fmt.Errorf("LeidenMatchSpec.Collection is required")
	}
	if len(spec.SeedNodeIDs) == 0 {
		return fmt.Errorf("LeidenMatchSpec.SeedNodeIDs is required")
	}
	if spec.MinHops < 0 {
		return fmt.Errorf("LeidenMatchSpec.MinHops must be >= 0, got %d", spec.MinHops)
	}
	if spec.MaxHops < 0 {
		return fmt.Errorf("LeidenMatchSpec.MaxHops must be >= 0, got %d", spec.MaxHops)
	}
	if spec.MinHops > spec.MaxHops {
		return fmt.Errorf("LeidenMatchSpec.MinHops (%d) > MaxHops (%d)", spec.MinHops, spec.MaxHops)
	}

	// Verify collection exists and has a graph.
	col, err := e.db.GetCollection(spec.Collection)
	if err != nil {
		return fmt.Errorf("collection %q: %w", spec.Collection, err)
	}
	if col.GetGraph() == nil {
		return fmt.Errorf("collection %q has no graph", spec.Collection)
	}

	return nil
}

// =============================================================================
// Internal helpers
// =============================================================================

// dedupAndSortNodeIDs returns a sorted, deduplicated copy of ids.
func dedupAndSortNodeIDs(ids []uint64) []uint64 {
	if len(ids) == 0 {
		return nil
	}
	seen := make(map[uint64]bool, len(ids))
	out := make([]uint64, 0, len(ids))
	for _, id := range ids {
		if !seen[id] {
			seen[id] = true
			out = append(out, id)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// makeEdgeKindSet returns nil if kinds is empty (match all), otherwise a set.
func makeEdgeKindSet(kinds []uint8) map[uint8]bool {
	if len(kinds) == 0 {
		return nil
	}
	m := make(map[uint8]bool, len(kinds))
	for _, k := range kinds {
		m[k] = true
	}
	return m
}

// filterEdgesByKind returns edges whose kind is in kindSet. If kindSet is nil,
// all edges pass through.
func filterEdgesByKind(edges []graph.Edge, kindSet map[uint8]bool) []graph.Edge {
	if kindSet == nil {
		return edges
	}
	out := make([]graph.Edge, 0, len(edges))
	for _, e := range edges {
		if kindSet[e.GetKind()] {
			out = append(out, e)
		}
	}
	return out
}

// sortEdgesByTargetThenKind sorts in-place by target node ID, then edge kind.
func sortEdgesByTargetThenKind(edges []graph.Edge) {
	sort.Slice(edges, func(i, j int) bool {
		if edges[i].Target != edges[j].Target {
			return edges[i].Target < edges[j].Target
		}
		return edges[i].GetKind() < edges[j].GetKind()
	})
}
