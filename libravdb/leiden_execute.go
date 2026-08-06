package libravdb

import (
	"context"
	"fmt"
)

// =============================================================================
// Execution result type
// =============================================================================

// LeidenExecutionResult pairs the raw Leiden match result with its
// materialized relation. Both fields are non-nil on success.
type LeidenExecutionResult struct {
	MatchResult *LeidenMatchResult
	Relation    *LeidenRelation
}

// =============================================================================
// ExecuteBoundLeidenMatchPlan
// =============================================================================

// ExecuteBoundLeidenMatchPlan runs the full execution pipeline for a bound
// Leiden plan: compute communities over the epoch-visible MATCH closure, then
// materialize the relation rows for MATCH-qualified target nodes.
//
// The pipeline is exactly:
//
//	validate → ComputeLeidenFromMatch → MaterializeLeidenRelation → return
//
// No Leiden traversal or modularity logic is duplicated. No relation
// materialization is duplicated. Execution is synchronous and deterministic.
func (e *EpochTx) ExecuteBoundLeidenMatchPlan(
	ctx context.Context,
	bound *BoundLeidenMatchPlan,
) (*LeidenExecutionResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if bound == nil {
		return nil, fmt.Errorf("execute Leiden plan: BoundLeidenMatchPlan must not be nil")
	}

	if err := validateBoundPlan(e, bound); err != nil {
		return nil, fmt.Errorf("execute Leiden plan: %w", err)
	}

	// Phase 1: Compute Leiden communities over the epoch-visible MATCH closure.
	matchResult, err := e.ComputeLeidenFromMatch(ctx, LeidenMatchSpec{
		Collection:  bound.Spec.Collection,
		SeedNodeIDs: bound.Spec.SeedNodeIDs,
		EdgeKinds:   bound.Spec.EdgeKinds,
		MinHops:     bound.Spec.MinHops,
		MaxHops:     bound.Spec.MaxHops,
		Direction:   bound.Spec.Direction,
	}, bound.Options)
	if err != nil {
		return nil, fmt.Errorf("compute Leiden from match: %w", err)
	}

	// Phase 2: Materialize relation rows for MATCH-qualified target nodes.
	relation, err := e.MaterializeLeidenRelation(ctx, matchResult)
	if err != nil {
		return nil, fmt.Errorf("materialize Leiden relation: %w", err)
	}

	return &LeidenExecutionResult{
		MatchResult: matchResult,
		Relation:    relation,
	}, nil
}

// validateBoundPlan performs pre-execution checks.
func validateBoundPlan(e *EpochTx, bound *BoundLeidenMatchPlan) error {
	e.mu.Lock()
	closed := e.closed
	e.mu.Unlock()
	if closed {
		return ErrEpochClosed
	}

	if bound.Spec.Collection == "" {
		return fmt.Errorf("Spec.Collection must not be empty")
	}
	if len(bound.Spec.SeedNodeIDs) == 0 {
		return fmt.Errorf("Spec.SeedNodeIDs must not be empty")
	}
	if bound.Spec.MinHops < 0 {
		return fmt.Errorf("Spec.MinHops must be >= 0, got %d", bound.Spec.MinHops)
	}
	if bound.Spec.MaxHops <= 0 {
		return fmt.Errorf("Spec.MaxHops must be > 0, got %d", bound.Spec.MaxHops)
	}
	if bound.Spec.MinHops > bound.Spec.MaxHops {
		return fmt.Errorf("Spec.MinHops (%d) must not exceed MaxHops (%d)", bound.Spec.MinHops, bound.Spec.MaxHops)
	}
	if bound.Spec.Direction != LeidenMatchOutbound && bound.Spec.Direction != LeidenMatchInbound {
		return fmt.Errorf("unsupported direction %d", bound.Spec.Direction)
	}
	return nil
}
