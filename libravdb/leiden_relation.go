package libravdb

import (
	"context"
	"fmt"
	"sort"
)

// =============================================================================
// Public API types
// =============================================================================

// LeidenRelationRow is a single tuple in the materialized local_clusters
// relation. It is the stable Go-native row representation consumed by a
// future SQL virtual-table or CTE operator.
type LeidenRelationRow struct {
	NodeID      uint64
	CommunityID uint64
	Collection  string
	RecordID    string
}

// LeidenRelation is the complete materialized relation derived from a
// LeidenMatchResult. It contains rows only for MATCH-qualified target nodes
// (not every node in the Leiden local graph), along with diagnostic fields
// propagated from the Leiden computation.
type LeidenRelation struct {
	Rows       []LeidenRelationRow
	Truncated  bool
	Scope      EpochLeidenScope
	Modularity float64
}

// =============================================================================
// MaterializeLeidenRelation
// =============================================================================

// MaterializeLeidenRelation converts a LeidenMatchResult into a stable
// relational row set. Only nodes present in matchResult.MatchedNodeIDs are
// emitted — intermediate BFS nodes that are not MATCH targets are excluded.
//
// Each row's community assignment is resolved from the Leiden result's
// Assignments(). Node-to-record resolution uses epoch-aware methods so that
// staged records and provisional mappings are visible, and snapshot-isolated
// committed state is respected. Rows are sorted deterministically by NodeID,
// then CommunityID, then RecordID.
func (e *EpochTx) MaterializeLeidenRelation(
	ctx context.Context,
	matchResult *LeidenMatchResult,
) (*LeidenRelation, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if matchResult == nil {
		return nil, fmt.Errorf("LeidenMatchResult must not be nil")
	}
	if matchResult.LeidenResult == nil {
		return nil, fmt.Errorf("LeidenMatchResult.LeidenResult must not be nil")
	}
	if matchResult.Collection == "" {
		return nil, fmt.Errorf("LeidenMatchResult.Collection must not be empty")
	}

	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return nil, ErrEpochClosed
	}
	e.mu.Unlock()

	// Validate collection exists before proceeding (ListRecords will also
	// validate internally, but we want a clear error before building maps).
	if _, err := e.db.GetCollection(matchResult.Collection); err != nil {
		return nil, fmt.Errorf("collection %q: %w", matchResult.Collection, err)
	}

	// Build community assignment index: nodeID → communityID.
	assignments := matchResult.LeidenResult.Assignments()
	nodeToCommunity := make(map[uint64]uint64, len(assignments))
	for _, a := range assignments {
		if prev, exists := nodeToCommunity[a.NodeID]; exists {
			return nil, fmt.Errorf(
				"duplicate community assignment for node %d: communities %d and %d",
				a.NodeID, prev, a.CommunityID,
			)
		}
		nodeToCommunity[a.NodeID] = a.CommunityID
	}

	// Build epoch-visible record set for this collection.
	records, err := e.ListRecords(ctx, matchResult.Collection)
	if err != nil {
		return nil, fmt.Errorf("epoch ListRecords for %q: %w", matchResult.Collection, err)
	}
	recordSet := make(map[string]bool, len(records))
	for _, rec := range records {
		recordSet[rec.ID] = true
	}

	// Materialize rows for each matched node.
	// Deduplicate matched node IDs first.
	seen := make(map[uint64]bool, len(matchResult.MatchedNodeIDs))
	rows := make([]LeidenRelationRow, 0, len(matchResult.MatchedNodeIDs))

	for _, nodeID := range matchResult.MatchedNodeIDs {
		if seen[nodeID] {
			continue
		}
		seen[nodeID] = true

		// Resolve node to (collection, recordID) via epoch-aware path.
		resolvedCol, recordID, err := e.ResolveNodeID(ctx, nodeID)
		if err != nil {
			return nil, fmt.Errorf("resolve node %d: %w", nodeID, err)
		}

		// Collection identity guard: cross-collection nodes are not allowed.
		if resolvedCol != matchResult.Collection {
			return nil, fmt.Errorf(
				"node %d resolves to collection %q, expected %q",
				nodeID, resolvedCol, matchResult.Collection,
			)
		}

		// Verify the record is visible in the epoch view.
		// Intermediate BFS nodes without a record are silently excluded
		// because they have no row to emit in the relation. Only nodes
		// with an epoch-visible record participate.
		if !recordSet[recordID] {
			continue
		}

		// Look up community assignment.
		communityID, ok := nodeToCommunity[nodeID]
		if !ok {
			return nil, fmt.Errorf(
				"no community assignment for matched node %d (record %q)",
				nodeID, recordID,
			)
		}

		rows = append(rows, LeidenRelationRow{
			NodeID:      nodeID,
			CommunityID: communityID,
			Collection:  matchResult.Collection,
			RecordID:    recordID,
		})
	}

	// Deterministic sort: NodeID ASC, CommunityID ASC, RecordID ASC.
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].NodeID != rows[j].NodeID {
			return rows[i].NodeID < rows[j].NodeID
		}
		if rows[i].CommunityID != rows[j].CommunityID {
			return rows[i].CommunityID < rows[j].CommunityID
		}
		return rows[i].RecordID < rows[j].RecordID
	})

	return &LeidenRelation{
		Rows:       rows,
		Truncated:  matchResult.LeidenResult.Truncated,
		Scope:      matchResult.LeidenResult.Scope,
		Modularity: matchResult.LeidenResult.Modularity,
	}, nil
}
