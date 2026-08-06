package libravdb

import (
	"context"
	"fmt"
	"math"
	"sort"

	"github.com/xDarkicex/libravdb/internal/graph"
)

// =============================================================================
// Public API types
// =============================================================================

type EpochLeidenScope string

const (
	EpochLeidenScopeCompleteLocalGraph EpochLeidenScope = "complete_local_graph"
	EpochLeidenScopeBudgetTruncated    EpochLeidenScope = "budget_truncated"
)

type EpochLeidenOptions struct {
	Seeds                []uint64
	EdgeKinds            []uint8
	MaxVertices          int
	MaxEdges             int
	ExpansionHops        int
	Resolution           float64
	MaxLevels            int
	MaxLocalMovingPasses int
	Epsilon              float64
}

func (o *EpochLeidenOptions) validate() error {
	if o.Resolution <= 0 {
		return fmt.Errorf("Resolution must be > 0, got %v", o.Resolution)
	}
	if o.MaxVertices < 0 {
		return fmt.Errorf("MaxVertices must be >= 0, got %d", o.MaxVertices)
	}
	if o.MaxEdges < 0 {
		return fmt.Errorf("MaxEdges must be >= 0, got %d", o.MaxEdges)
	}
	if o.MaxLevels < 0 {
		return fmt.Errorf("MaxLevels must be >= 0, got %d", o.MaxLevels)
	}
	return nil
}

func (o *EpochLeidenOptions) defaults() {
	if o.ExpansionHops <= 0 {
		o.ExpansionHops = 2
	}
	if o.Resolution <= 0 {
		o.Resolution = 1.0
	}
	if o.MaxLevels <= 0 {
		o.MaxLevels = 10
	}
	if o.MaxLocalMovingPasses <= 0 {
		o.MaxLocalMovingPasses = 10
	}
	if o.Epsilon <= 0 {
		o.Epsilon = 1e-12
	}
}

type EpochCommunity struct {
	ID      uint64
	Members []uint64
}

type EpochLeidenResult struct {
	Communities       []EpochCommunity
	Modularity        float64
	InitialModularity float64
	Levels            int
	Moves             int
	Vertices          int
	Edges             int
	Truncated         bool
	Approximate       bool
	Scope             EpochLeidenScope
	SelfLoops         int
}

// LeidenAssignment is a single row in the flat (NodeID, CommunityID) relation
// derived from an EpochLeidenResult. It is the stable Go-native representation
// consumed by future SQL relation adapters.
type LeidenAssignment struct {
	NodeID      uint64
	CommunityID uint64
}

// Assignments flattens the community result into one row per original graph
// node. NodeID is always the original durable graph node ID — never an
// internal array index or aggregate supernode ID. CommunityID matches the
// public EpochCommunity.ID.
//
// Rows are sorted by ascending NodeID, then ascending CommunityID as a
// defensive tie-break. The returned slice is a newly allocated defensive copy;
// mutating it does not affect r.Communities.
//
// A nil receiver returns nil. An empty result returns an empty (non-nil) slice.
// ComputeLeiden guarantees each original node appears in exactly one community,
// so the method does not need to deduplicate.
func (r *EpochLeidenResult) Assignments() []LeidenAssignment {
	if r == nil {
		return nil
	}
	total := 0
	for _, c := range r.Communities {
		total += len(c.Members)
	}
	if total == 0 {
		return []LeidenAssignment{}
	}
	out := make([]LeidenAssignment, 0, total)
	for _, c := range r.Communities {
		for _, member := range c.Members {
			out = append(out, LeidenAssignment{
				NodeID:      member,
				CommunityID: c.ID,
			})
		}
	}
	// Sort by ascending NodeID, then ascending CommunityID.
	sort.Slice(out, func(i, j int) bool {
		if out[i].NodeID != out[j].NodeID {
			return out[i].NodeID < out[j].NodeID
		}
		return out[i].CommunityID < out[j].CommunityID
	})
	return out
}

// =============================================================================
// Internal graph representation
// =============================================================================

type leidenGraph struct {
	nodes           []uint64
	nodeIdx         map[uint64]int
	adj             [][]leidenNeighbor // symmetric, non-self edges only
	selfLoopWeight  []float64          // physical self-loop weight (once)
	degrees         []float64          // k_i = Σ_{j} A_ij, includes 2*selfLoopWeight[i]
	totalM          float64            // m = Σ undirected non-loop weights + Σ self-loop weights
	nodeToComm      []uint32
	originalMembers [][]uint64 // for base graph: [{origID}]; propagated through aggregation
}

type leidenNeighbor struct {
	to     int
	weight float64
}

// addNode adds a vertex with its original graph node ID. If the node already
// exists, the id is appended to its originalMembers.
func (lg *leidenGraph) addNode(id uint64) {
	if idx, ok := lg.nodeIdx[id]; ok {
		lg.originalMembers[idx] = append(lg.originalMembers[idx], id)
		return
	}
	lg.nodeIdx[id] = len(lg.nodes)
	lg.nodes = append(lg.nodes, id)
	lg.adj = append(lg.adj, nil)
	lg.degrees = append(lg.degrees, 0)
	lg.selfLoopWeight = append(lg.selfLoopWeight, 0)
	lg.originalMembers = append(lg.originalMembers, []uint64{id})
}

// addUndirectedEdge adds a non-loop undirected edge of weight w. u != v required.
func (lg *leidenGraph) addUndirectedEdge(u, v uint64, weight float64) {
	if u == v {
		lg.addSelfLoop(u, weight)
		return
	}
	if weight <= 0 {
		return
	}
	lg.addNode(u)
	lg.addNode(v)
	ui, vi := lg.nodeIdx[u], lg.nodeIdx[v]
	for i := range lg.adj[ui] {
		if lg.adj[ui][i].to == vi {
			lg.adj[ui][i].weight += weight
			for j := range lg.adj[vi] {
				if lg.adj[vi][j].to == ui {
					lg.adj[vi][j].weight += weight
					break
				}
			}
			lg.degrees[ui] += weight
			lg.degrees[vi] += weight
			lg.totalM += weight
			return
		}
	}
	lg.adj[ui] = append(lg.adj[ui], leidenNeighbor{to: vi, weight: weight})
	lg.adj[vi] = append(lg.adj[vi], leidenNeighbor{to: ui, weight: weight})
	lg.degrees[ui] += weight
	lg.degrees[vi] += weight
	lg.totalM += weight
}

// addSelfLoop adds a physical self-loop of weight w.
// Undirected convention: A_uu = 2w, so degree[u] += 2*w, totalM += w.
func (lg *leidenGraph) addSelfLoop(u uint64, weight float64) {
	if weight <= 0 {
		return
	}
	lg.addNode(u)
	idx := lg.nodeIdx[u]
	lg.selfLoopWeight[idx] += weight
	lg.degrees[idx] += 2 * weight
	lg.totalM += weight
}

func (lg *leidenGraph) totalSelfLoopCount() int {
	c := 0
	for _, w := range lg.selfLoopWeight {
		if w > 0 {
			c++
		}
	}
	return c
}

// =============================================================================
// ComputeLeiden — public entry point
// =============================================================================

func (e *EpochTx) ComputeLeiden(ctx context.Context, opts EpochLeidenOptions) (*EpochLeidenResult, error) {
	opts.defaults()
	if err := opts.validate(); err != nil {
		return nil, err
	}
	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return nil, ErrEpochClosed
	}
	e.mu.Unlock()

	seeds := opts.Seeds
	if len(seeds) == 0 {
		seeds = e.collectDeltaNodes()
	}
	if len(seeds) == 0 {
		return nil, fmt.Errorf("no seeds provided and no delta nodes in epoch")
	}

	lg, truncated, err := e.buildLeidenGraph(ctx, seeds, opts)
	if err != nil {
		return nil, err
	}
	if len(lg.nodes) == 0 {
		return nil, fmt.Errorf("induced subgraph is empty")
	}

	// Save original graph stats for final reporting.
	origVertices := len(lg.nodes)
	origEdges := int(lg.totalM)
	origSelfLoops := lg.totalSelfLoopCount()

	n := len(lg.nodes)
	lg.nodeToComm = make([]uint32, n)
	for i := range lg.nodeToComm {
		lg.nodeToComm[i] = uint32(i)
	}

	twoM := 2.0 * lg.totalM
	if twoM < 1e-15 {
		twoM = 1.0
	}
	initialQ := computeModularity(lg, opts.Resolution, twoM)
	totalMoves := 0
	levels := 0

	// Multi-level Leiden.
	current := lg
	for level := 0; level < opts.MaxLevels; level++ {
		passMoves := 0
		for pass := 0; pass < opts.MaxLocalMovingPasses; pass++ {
			mv := localMovingPhase(current, opts.Resolution, twoM, opts.Epsilon)
			passMoves += mv
			if mv == 0 {
				break
			}
		}
		totalMoves += passMoves
		constrainedRefinement(current, opts.Resolution, twoM, opts.Epsilon)
		levels++

		// Aggregate if compression is possible.
		commSet := make(map[uint32]bool)
		for _, c := range current.nodeToComm {
			commSet[c] = true
		}
		if len(commSet) == 1 || len(commSet) == len(current.nodes) {
			break
		}
		agg := aggregateGraph(current)
		if len(agg.nodes) >= len(current.nodes) {
			break
		}
		current = agg
		n = len(current.nodes)
		twoM = 2.0 * current.totalM
		if twoM < 1e-15 {
			twoM = 1.0
		}
	}
	if levels == 0 {
		levels = 1
	}

	// Project final partition back to original node IDs.
	communities := projectToOriginalIDs(current)

	// Compute final modularity on the original base graph with the projected partition.
	finalMod := computeProjectedModularity(lg, communities, opts.Resolution)

	scope := EpochLeidenScopeCompleteLocalGraph
	if truncated {
		scope = EpochLeidenScopeBudgetTruncated
	}

	return &EpochLeidenResult{
		Communities:       communities,
		Modularity:        finalMod,
		InitialModularity: initialQ,
		Levels:            levels,
		Moves:             totalMoves,
		Vertices:          origVertices,
		Edges:             origEdges,
		Truncated:         truncated,
		Approximate:       false,
		Scope:             scope,
		SelfLoops:         origSelfLoops,
	}, nil
}

// projectToOriginalIDs builds EpochCommunity values from the (possibly aggregated)
// graph's originalMembers. Community ID = minimum original node ID in the community.
func projectToOriginalIDs(lg *leidenGraph) []EpochCommunity {
	commToMembers := make(map[uint32]map[uint64]bool)
	for i, c := range lg.nodeToComm {
		if commToMembers[c] == nil {
			commToMembers[c] = make(map[uint64]bool)
		}
		for _, origID := range lg.originalMembers[i] {
			commToMembers[c][origID] = true
		}
	}
	var result []EpochCommunity
	for c, memberSet := range commToMembers {
		members := make([]uint64, 0, len(memberSet))
		for id := range memberSet {
			members = append(members, id)
		}
		sort.Slice(members, func(i, j int) bool { return members[i] < members[j] })
		// Community ID = minimum original node ID.
		cid := members[0]
		_ = c // original community label is not needed
		result = append(result, EpochCommunity{ID: cid, Members: members})
	}
	sort.Slice(result, func(i, j int) bool { return result[i].ID < result[j].ID })
	return result
}

// computeProjectedModularity evaluates modularity on the base graph `base`
// using the final projected partition. It builds a map from original node ID
// to community ID from `communities`, then applies the community-sum formula
// over the base graph's degrees and adjacency.
func computeProjectedModularity(base *leidenGraph, communities []EpochCommunity, gamma float64) float64 {
	m := base.totalM
	if m < 1e-15 {
		return 0
	}
	origToCommIdx := make(map[uint64]int, len(base.nodes))
	for ci, comm := range communities {
		for _, id := range comm.Members {
			origToCommIdx[id] = ci
		}
	}
	baseComm := make([]uint32, len(base.nodes))
	for i, origID := range base.nodes {
		if ci, ok := origToCommIdx[origID]; ok {
			baseComm[i] = uint32(ci)
		} else {
			baseComm[i] = uint32(i)
		}
	}

	var ic = make(map[uint32]float64)
	var kc = make(map[uint32]float64)
	for i := 0; i < len(base.nodes); i++ {
		c := baseComm[i]
		kc[c] += base.degrees[i]
		for _, nb := range base.adj[i] {
			if baseComm[nb.to] == c {
				ic[c] += nb.weight
			}
		}
		if base.selfLoopWeight[i] > 0 {
			ic[c] += 2 * base.selfLoopWeight[i]
		}
	}
	var q float64
	for c := range kc {
		iUndirected := ic[c] / 2.0
		q += iUndirected/m - gamma*kc[c]*kc[c]/(4.0*m*m)
	}
	return q
}

// =============================================================================
// collectDeltaNodes
// =============================================================================

func (e *EpochTx) collectDeltaNodes() []uint64 {
	seen := make(map[uint64]bool)
	for _, gtx := range e.graphs {
		adds, removes, drops := gtx.StagedOps()
		for _, op := range adds {
			seen[op.Src] = true
			seen[op.Tgt] = true
		}
		for _, op := range removes {
			seen[op.Src] = true
			seen[op.Tgt] = true
		}
		for _, drop := range drops {
			seen[drop.NodeID] = true
		}
	}
	nodes := make([]uint64, 0, len(seen))
	for n := range seen {
		nodes = append(nodes, n)
	}
	sort.Slice(nodes, func(i, j int) bool { return nodes[i] < nodes[j] })
	return nodes
}

// =============================================================================
// buildLeidenGraph
// =============================================================================

func (e *EpochTx) buildLeidenGraph(ctx context.Context, seeds []uint64, opts EpochLeidenOptions) (*leidenGraph, bool, error) {
	lg := &leidenGraph{nodeIdx: make(map[uint64]int)}
	truncated := false

	type bfsEntry struct {
		node uint64
		hop  int
	}
	queue := make([]bfsEntry, 0, len(seeds))
	visited := make(map[uint64]bool)

	for _, seed := range seeds {
		if !visited[seed] {
			visited[seed] = true
			queue = append(queue, bfsEntry{seed, 0})
			lg.addNode(seed)
		}
	}

	for len(queue) > 0 {
		if ctx.Err() != nil {
			return nil, false, ctx.Err()
		}
		entry := queue[0]
		queue = queue[1:]
		if entry.hop >= opts.ExpansionHops {
			continue
		}
		for _, nb := range e.getEpochNeighbors(entry.node, opts.EdgeKinds) {
			nbID := nb.Target
			w := float64(nb.Weight)
			if w <= 0 {
				w = 1.0
			}
			if nbID == entry.node {
				lg.addSelfLoop(entry.node, w)
			} else {
				lg.addUndirectedEdge(entry.node, nbID, w)
			}
			if !visited[nbID] {
				if opts.MaxVertices > 0 && len(lg.nodes) >= opts.MaxVertices {
					truncated = true
					continue
				}
				if opts.MaxEdges > 0 && int(lg.totalM) >= opts.MaxEdges {
					truncated = true
					continue
				}
				visited[nbID] = true
				lg.addNode(nbID)
				queue = append(queue, bfsEntry{nbID, entry.hop + 1})
			}
		}
	}
	return lg, truncated, nil
}

func (e *EpochTx) getEpochNeighbors(nodeID uint64, edgeKinds []uint8) []graph.Edge {
	kindSet := makeKindSet(edgeKinds)
	for _, gtx := range e.graphs {
		edges, err := gtx.NeighborsOverlay(nodeID)
		if err == nil && len(edges) > 0 {
			return filterByKinds(edges, kindSet)
		}
	}
	for _, gtx := range e.graphs {
		edges, _ := gtx.NeighborsOverlay(nodeID)
		return filterByKinds(edges, kindSet)
	}
	return nil
}

func makeKindSet(kinds []uint8) map[uint8]bool {
	if len(kinds) == 0 {
		return nil
	}
	m := make(map[uint8]bool, len(kinds))
	for _, k := range kinds {
		m[k] = true
	}
	return m
}

func filterByKinds(edges []graph.Edge, kindSet map[uint8]bool) []graph.Edge {
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

// =============================================================================
// leidenMoveState — incremental community statistics for O(E) local moving
// =============================================================================

type leidenMoveState struct {
	communityDegree map[uint32]float64 // K_C = sum of degrees of nodes in C
}

func newLeidenMoveState(lg *leidenGraph) *leidenMoveState {
	s := &leidenMoveState{communityDegree: make(map[uint32]float64, len(lg.nodes))}
	for i := 0; i < len(lg.nodes); i++ {
		s.communityDegree[lg.nodeToComm[i]] += lg.degrees[i]
	}
	return s
}

func (s *leidenMoveState) neighborWeights(lg *leidenGraph, i int) map[uint32]float64 {
	w := make(map[uint32]float64)
	for _, nb := range lg.adj[i] {
		c := lg.nodeToComm[nb.to]
		w[c] += nb.weight
	}
	return w
}

func deltaMoveIncremental(lg *leidenGraph, i int, srcComm, dstComm uint32, kiSrc, kiDst float64, state *leidenMoveState, gamma float64) float64 {
	m := lg.totalM
	if m < 1e-15 {
		return 0
	}
	if srcComm == dstComm {
		return 0
	}
	ki := lg.degrees[i]
	kSrc := state.communityDegree[srcComm]
	kDst := state.communityDegree[dstComm]
	term1 := (kiDst - kiSrc) / m
	term2 := gamma * ki * (kDst - kSrc + ki) / (2.0 * m * m)
	return term1 - term2
}

func (s *leidenMoveState) applyMove(lg *leidenGraph, i int, srcComm, dstComm uint32) {
	ki := lg.degrees[i]
	s.communityDegree[srcComm] -= ki
	s.communityDegree[dstComm] += ki
	lg.nodeToComm[i] = dstComm
}

func computeCommunityDegreesFromScratch(lg *leidenGraph) map[uint32]float64 {
	out := make(map[uint32]float64)
	for i := 0; i < len(lg.nodes); i++ {
		out[lg.nodeToComm[i]] += lg.degrees[i]
	}
	return out
}

// deltaMove is retained for backward compatibility with tests.
// It delegates to deltaMoveIncremental with a temporary state.
func deltaMove(lg *leidenGraph, i int, dstComm uint32, gamma, twoM float64) float64 {
	m := lg.totalM
	if m < 1e-15 {
		return 0
	}
	srcComm := lg.nodeToComm[i]
	if srcComm == dstComm {
		return 0
	}
	state := newLeidenMoveState(lg)
	neighborW := state.neighborWeights(lg, i)
	return deltaMoveIncremental(lg, i, srcComm, dstComm, neighborW[srcComm], neighborW[dstComm], state, gamma)
}

// =============================================================================
// localMovingPhase
// =============================================================================

func localMovingPhase(lg *leidenGraph, gamma, twoM, epsilon float64) int {
	n := len(lg.nodes)
	if n == 0 {
		return 0
	}
	state := newLeidenMoveState(lg)
	moves := 0
	for pass := 0; pass < 1; pass++ {
		passMoves := 0
		for i := 0; i < n; i++ {
			oldComm := lg.nodeToComm[i]
			neighborW := state.neighborWeights(lg, i)
			bestComm := oldComm
			bestDQ := 0.0
			candidates := make(map[uint32]bool)
			candidates[oldComm] = true
			candidates[uint32(i)] = true
			for c := range neighborW {
				candidates[c] = true
			}
			for c := range candidates {
				if c == oldComm {
					continue
				}
				kiSrc := neighborW[oldComm]
				kiDst := neighborW[c]
				dq := deltaMoveIncremental(lg, i, oldComm, c, kiSrc, kiDst, state, gamma)
				if dq > bestDQ && dq > epsilon {
					bestDQ = dq
					bestComm = c
				} else if math.Abs(dq-bestDQ) <= epsilon && dq > epsilon && c < bestComm {
					bestComm = c
				}
			}
			if bestComm != oldComm {
				state.applyMove(lg, i, oldComm, bestComm)
				passMoves++
			}
		}
		moves += passMoves
		if passMoves == 0 {
			break
		}
	}
	return moves
}

// refineCommunities
// =============================================================================

func refineCommunities(lg *leidenGraph) {
	n := len(lg.nodes)
	visited := make([]bool, n)
	commNodes := make(map[uint32][]int)
	for i := 0; i < n; i++ {
		c := lg.nodeToComm[i]
		commNodes[c] = append(commNodes[c], i)
	}
	maxCID := uint32(0)
	for _, c := range lg.nodeToComm {
		if c >= maxCID {
			maxCID = c + 1
		}
	}
	newCID := maxCID

	for _, members := range commNodes {
		for _, start := range members {
			if visited[start] {
				continue
			}
			parentComm := lg.nodeToComm[start]
			queue := []int{start}
			visited[start] = true
			for len(queue) > 0 {
				curr := queue[0]
				queue = queue[1:]
				lg.nodeToComm[curr] = newCID
				for _, nb := range lg.adj[curr] {
					if !visited[nb.to] && lg.nodeToComm[nb.to] == parentComm {
						visited[nb.to] = true
						queue = append(queue, nb.to)
					}
				}
			}
			newCID++
		}
	}
}

// =============================================================================
// constrainedRefinement — parent-constrained, exact-modularity, connectivity-safe
// =============================================================================

func constrainedRefinement(lg *leidenGraph, gamma, twoM, epsilon float64) {
	n := len(lg.nodes)
	if n == 0 {
		return
	}
	coarseParent := make([]uint32, n)
	copy(coarseParent, lg.nodeToComm)
	for i := 0; i < n; i++ {
		lg.nodeToComm[i] = uint32(i)
	}
	state := newLeidenMoveState(lg)
	for i := 0; i < n; i++ {
		pi := coarseParent[i]
		oldComm := lg.nodeToComm[i]
		neighborW := state.neighborWeights(lg, i)
		bestComm := oldComm
		bestDQ := 0.0
		candidates := make(map[uint32]bool)
		candidates[oldComm] = true
		candidates[uint32(i)] = true
		for c := range neighborW {
			candidates[c] = true
		}
		for c := range candidates {
			if c == oldComm {
				continue
			}
			// Check parent constraint.
			cParent := uint32(0)
			found := false
			for j := 0; j < n && !found; j++ {
				if lg.nodeToComm[j] == c {
					cParent = coarseParent[j]
					found = true
				}
			}
			if cParent != pi {
				continue
			}
			kiSrc := neighborW[oldComm]
			kiDst := neighborW[c]
			dq := deltaMoveIncremental(lg, i, oldComm, c, kiSrc, kiDst, state, gamma)
			if dq > bestDQ && dq > epsilon {
				bestDQ = dq
				bestComm = c
			} else if math.Abs(dq-bestDQ) <= epsilon && dq > epsilon && c < bestComm {
				bestComm = c
			}
		}
		if bestComm != oldComm {
			state.applyMove(lg, i, oldComm, bestComm)
		}
	}
	refineConnectivity(lg, coarseParent)
}

func refineConnectivity(lg *leidenGraph, coarseParent []uint32) {
	n := len(lg.nodes)
	visited := make([]bool, n)
	type groupKey struct{ parent, refined uint32 }
	groups := make(map[groupKey][]int)
	for i := 0; i < n; i++ {
		key := groupKey{coarseParent[i], lg.nodeToComm[i]}
		groups[key] = append(groups[key], i)
	}
	maxCID := uint32(0)
	for _, c := range lg.nodeToComm {
		if c >= maxCID {
			maxCID = c + 1
		}
	}
	newCID := maxCID
	for _, members := range groups {
		for _, start := range members {
			if visited[start] {
				continue
			}
			parentRefined := lg.nodeToComm[start]
			queue := []int{start}
			visited[start] = true
			for len(queue) > 0 {
				curr := queue[0]
				queue = queue[1:]
				lg.nodeToComm[curr] = newCID
				for _, nb := range lg.adj[curr] {
					if !visited[nb.to] && lg.nodeToComm[nb.to] == parentRefined {
						visited[nb.to] = true
						queue = append(queue, nb.to)
					}
				}
			}
			newCID++
		}
	}
}

// =============================================================================
// aggregateGraph — modularity-preserving supernode construction
// =============================================================================

func aggregateGraph(lg *leidenGraph) *leidenGraph {
	// Collect community → child indices.
	commChildren := make(map[uint32][]int)
	for i, c := range lg.nodeToComm {
		commChildren[c] = append(commChildren[c], i)
	}
	commList := make([]uint32, 0, len(commChildren))
	for c := range commChildren {
		commList = append(commList, c)
	}
	sort.Slice(commList, func(i, j int) bool { return commList[i] < commList[j] })

	// Create supernode graph. Supernode ID = minimum original node ID in the community.
	agg := &leidenGraph{nodeIdx: make(map[uint64]int)}
	commToSuperID := make(map[uint32]uint64, len(commList))

	for _, c := range commList {
		children := commChildren[c]
		// Collect all original IDs from children.
		allOrig := make(map[uint64]bool)
		for _, child := range children {
			for _, origID := range lg.originalMembers[child] {
				allOrig[origID] = true
			}
		}
		var minID uint64 = ^uint64(0)
		for id := range allOrig {
			if id < minID {
				minID = id
			}
		}
		commToSuperID[c] = minID
		agg.addNode(minID)
		// Store ALL original IDs as members.
		idx := agg.nodeIdx[minID]
		agg.originalMembers[idx] = agg.originalMembers[idx][:0]
		for id := range allOrig {
			agg.originalMembers[idx] = append(agg.originalMembers[idx], id)
		}
		sort.Slice(agg.originalMembers[idx], func(i, j int) bool {
			return agg.originalMembers[idx][i] < agg.originalMembers[idx][j]
		})
	}

	// Process each physical edge exactly once (i < j for non-loop edges).
	// Cross-community edges use seenPairs to deduplicate; internal edges
	// always add to self-loop weight regardless.
	seenPairs := make(map[[2]uint64]bool)
	for i := 0; i < len(lg.nodes); i++ {
		ci := lg.nodeToComm[i]
		su := commToSuperID[ci]
		for _, nb := range lg.adj[i] {
			j := nb.to
			if i >= j {
				continue
			}
			cj := lg.nodeToComm[j]
			sv := commToSuperID[cj]

			if ci == cj {
				// Internal community edge → supernode self-loop.
				agg.addSelfLoop(su, nb.weight)
			} else {
				key := [2]uint64{su, sv}
				if su > sv {
					key = [2]uint64{sv, su}
				}
				if seenPairs[key] {
					continue
				}
				seenPairs[key] = true
				agg.addUndirectedEdge(su, sv, nb.weight)
			}
		}
		// Physical self-loops → supernode self-loops.
		if lg.selfLoopWeight[i] > 0 {
			agg.addSelfLoop(su, lg.selfLoopWeight[i])
		}
	}

	// Initialize nodeToComm on the aggregated graph (each supernode in its own community).
	agg.nodeToComm = make([]uint32, len(agg.nodes))
	for i := range agg.nodeToComm {
		agg.nodeToComm[i] = uint32(i)
	}
	return agg
}

// resetPartition re-initializes nodeToComm so each node is its own community.

// =============================================================================
// computeModularity
// =============================================================================

func computeModularity(lg *leidenGraph, gamma, twoM float64) float64 {
	if twoM < 1e-15 {
		return 0
	}
	m := lg.totalM
	var ic = make(map[uint32]float64)
	var kc = make(map[uint32]float64)

	for i := 0; i < len(lg.nodes); i++ {
		c := lg.nodeToComm[i]
		kc[c] += lg.degrees[i]
		for _, nb := range lg.adj[i] {
			if lg.nodeToComm[nb.to] == c {
				ic[c] += nb.weight
			}
		}
		// Self-loop internal weight.
		if lg.selfLoopWeight[i] > 0 {
			ic[c] += 2 * lg.selfLoopWeight[i]
		}
	}

	var q float64
	for c := range kc {
		iUndirected := ic[c] / 2.0
		q += iUndirected/m - gamma*(kc[c]*kc[c])/(4.0*m*m)
	}
	return q
}

// =============================================================================
// referenceModularity (O(V²))
// =============================================================================

func referenceModularity(lg *leidenGraph, gamma, twoM float64) float64 {
	if twoM < 1e-15 {
		return 0
	}
	n := len(lg.nodes)
	var q float64
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if lg.nodeToComm[i] != lg.nodeToComm[j] {
				continue
			}
			aij := 0.0
			for _, nb := range lg.adj[i] {
				if nb.to == j {
					aij = nb.weight
					break
				}
			}
			// Self-loop contribution: if i==j, A_ii = 2*selfLoopWeight.
			if i == j {
				aij = 2 * lg.selfLoopWeight[i]
			}
			q += aij - gamma*(lg.degrees[i]*lg.degrees[j]/twoM)
		}
	}
	return q / twoM
}

const _ = math.MaxFloat64
