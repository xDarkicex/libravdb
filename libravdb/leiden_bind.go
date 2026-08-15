package libravdb

import (
	"context"
	"fmt"
	"sort"

	"github.com/xDarkicex/libravdb/internal/graph"
)

// =============================================================================
// Bound plan type
// =============================================================================

// BoundLeidenMatchPlan is a fully-resolved executable Leiden match request.
// All string identifiers have been resolved to numeric graph IDs.
type BoundLeidenMatchPlan struct {
	Spec    LeidenMatchSpec
	Options EpochLeidenOptions
}

// =============================================================================
// BindLeidenMatchPlan
// =============================================================================

// BindLeidenMatchPlan resolves a logical LeidenMatchPlan against the active
// epoch state. Seed labels are resolved to visible graph node IDs, edge kind
// names are converted to numeric codes, and options are validated and populated.
//
// The returned plan is a defensive copy; mutating it does not affect the input.
// No Leiden execution or graph traversal occurs during binding.
func (e *EpochTx) BindLeidenMatchPlan(
	ctx context.Context,
	plan *LeidenMatchPlan,
	collection string,
) (*BoundLeidenMatchPlan, error) {
	if e == nil {
		return nil, fmt.Errorf("EpochTx must not be nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if plan == nil {
		return nil, fmt.Errorf("LeidenMatchPlan must not be nil")
	}

	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return nil, ErrEpochClosed
	}
	e.mu.Unlock()

	// ── Collection resolution ──
	coll, err := resolveCollection(e, plan.Collection, collection)
	if err != nil {
		return nil, err
	}

	col, err := e.db.GetCollection(coll)
	if err != nil {
		return nil, fmt.Errorf("collection %q: %w", coll, err)
	}
	g := col.GetGraph()
	if g == nil {
		return nil, fmt.Errorf("collection %q has no graph", coll)
	}

	// ── Terminal-label policy ──
	if plan.TerminalLabel != "" {
		return nil, fmt.Errorf("terminal label %q is not yet supported in COMPUTE LEIDEN binding", plan.TerminalLabel)
	}

	// ── Seed binding ──
	seedIDs, err := bindSeeds(ctx, e, g, coll, plan)
	if err != nil {
		return nil, err
	}

	// ── Edge kind binding ──
	edgeKinds, err := bindEdgeKind(plan.EdgeKind)
	if err != nil {
		return nil, err
	}

	// ── Direction and hop validation ──
	if err := validateDirection(plan.Direction); err != nil {
		return nil, err
	}
	if err := validateHops(plan.MinHops, plan.MaxHops); err != nil {
		return nil, err
	}

	// ── Populate options ──
	opts := plan.Options // value copy
	opts.Seeds = make([]uint64, len(seedIDs))
	copy(opts.Seeds, seedIDs)

	if len(edgeKinds) > 0 && len(opts.EdgeKinds) == 0 {
		opts.EdgeKinds = make([]uint8, len(edgeKinds))
		copy(opts.EdgeKinds, edgeKinds)
	}
	if opts.ExpansionHops <= 0 {
		opts.ExpansionHops = plan.MaxHops
	}

	// ── Build spec ──
	var specEdgeKinds []uint8
	if edgeKinds != nil {
		specEdgeKinds = make([]uint8, len(edgeKinds))
		copy(specEdgeKinds, edgeKinds)
	}
	specSeeds := make([]uint64, len(seedIDs))
	copy(specSeeds, seedIDs)

	spec := LeidenMatchSpec{
		Collection:  coll,
		SeedNodeIDs: specSeeds,
		EdgeKinds:   specEdgeKinds,
		MinHops:     plan.MinHops,
		MaxHops:     plan.MaxHops,
		Direction:   plan.Direction,
	}

	return &BoundLeidenMatchPlan{
		Spec:    spec,
		Options: opts,
	}, nil
}

// =============================================================================
// Collection resolution
// =============================================================================

func resolveCollection(e *EpochTx, planCol, explicitCol string) (string, error) {
	if explicitCol != "" {
		if planCol != "" && planCol != explicitCol {
			return "", fmt.Errorf("collection mismatch: plan specifies %q, explicit argument specifies %q", planCol, explicitCol)
		}
		return explicitCol, nil
	}
	if planCol != "" {
		return planCol, nil
	}
	return "", fmt.Errorf("collection is required: populate plan.Collection or pass a collection argument")
}

// =============================================================================
// Seed binding
// =============================================================================

func bindSeeds(ctx context.Context, e *EpochTx, g Graph, collection string, plan *LeidenMatchPlan) ([]uint64, error) {
	if plan.SeedLabel == "" {
		return nil, fmt.Errorf("seed label is required; explicit seed-node binding is not yet supported")
	}

	candidates := g.GetLabelNodes(plan.SeedLabel)
	if len(candidates) == 0 {
		return nil, fmt.Errorf("seed label %q has no registered graph nodes", plan.SeedLabel)
	}

	// Deduplicate and sort.
	seen := make(map[uint64]bool, len(candidates))
	deduped := make([]uint64, 0, len(candidates))
	for _, nid := range candidates {
		if !seen[nid] {
			seen[nid] = true
			deduped = append(deduped, nid)
		}
	}
	sort.Slice(deduped, func(i, j int) bool { return deduped[i] < deduped[j] })

	// Build epoch-visible record set for collection filtering.
	records, err := e.ListRecords(ctx, collection)
	if err != nil {
		return nil, fmt.Errorf("epoch ListRecords for seed binding: %w", err)
	}
	visible := make(map[string]bool, len(records))
	for _, rec := range records {
		visible[rec.ID] = true
	}

	// Filter: resolve each candidate, keep only those in the right collection
	// with an epoch-visible record.
	visibleSeeds := make([]uint64, 0, len(deduped))
	for _, nid := range deduped {
		resolvedCol, recordID, err := e.ResolveNodeID(ctx, nid)
		if err != nil {
			continue
		}
		if resolvedCol != collection {
			continue
		}
		if !visible[recordID] {
			continue
		}
		visibleSeeds = append(visibleSeeds, nid)
	}

	if len(visibleSeeds) == 0 {
		return nil, fmt.Errorf("no visible seed nodes for label %q in collection %q", plan.SeedLabel, collection)
	}

	return visibleSeeds, nil
}

// =============================================================================
// Edge kind binding
// =============================================================================

func bindEdgeKind(edgeKind string) ([]uint8, error) {
	if edgeKind == "" {
		return nil, nil // nil filter: all edge kinds
	}

	resolved := graph.ResolveEdgeKind(edgeKind)
	if resolved == 0 {
		return nil, fmt.Errorf("unknown edge kind %q", edgeKind)
	}
	return []uint8{resolved}, nil
}

// =============================================================================
// Validation helpers
// =============================================================================

// resolveLeidenCollection finds the collection whose graph contains nodes
// with the given label. Returns an error if zero or more than one collection
// has matching labeled nodes. Used by the standalone COMPUTE LEIDEN SQL
// statement which does not specify a collection in the grammar.
func (db *Database) resolveLeidenCollection(ctx context.Context, seedLabel string) (string, error) {
	if seedLabel == "" {
		return "", fmt.Errorf("seed label is required to resolve collection")
	}

	collections, err := db.ListCollectionsWithContext(ctx)
	if err != nil {
		return "", fmt.Errorf("list collections: %w", err)
	}

	var candidates []string
	for _, name := range collections {
		col, err := db.GetCollection(name)
		if err != nil {
			continue
		}
		g := col.GetGraph()
		if g == nil {
			continue
		}
		nodes := g.GetLabelNodes(seedLabel)
		if len(nodes) > 0 {
			candidates = append(candidates, name)
		}
	}

	switch len(candidates) {
	case 0:
		return "", fmt.Errorf("no collection with graph labeled %q", seedLabel)
	case 1:
		return candidates[0], nil
	default:
		return "", fmt.Errorf("seed label %q matches multiple collections: %v", seedLabel, candidates)
	}
}

func validateDirection(d LeidenMatchDirection) error {
	switch d {
	case LeidenMatchOutbound, LeidenMatchInbound:
		return nil
	default:
		return fmt.Errorf("unsupported direction %d", d)
	}
}

func validateHops(min, max int) error {
	if min < 0 {
		return fmt.Errorf("MinHops must be >= 0, got %d", min)
	}
	if max <= 0 {
		return fmt.Errorf("MaxHops must be > 0, got %d", max)
	}
	if min > max {
		return fmt.Errorf("MinHops (%d) must not exceed MaxHops (%d)", min, max)
	}
	return nil
}
