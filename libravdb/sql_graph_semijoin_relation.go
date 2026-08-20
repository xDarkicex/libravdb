package libravdb

import (
	"context"
	"fmt"
	"sort"
	"strings"

	graphpkg "github.com/xDarkicex/libravdb/internal/graph"
)

// virtualGraphSemijoinRelationRows implements the explicit evidence relation
// used by GRAPH_SEMIJOIN(...). It deliberately shares the graph's ordinary
// neighbor API and record visibility path with MATCH/JOIN MATCH; no duplicate
// graph or storage representation is created.
//
// Arguments are:
//
//	collection, origin_id [, edge_type [, source_expansion_limit [, origin_expansion_limit [, candidate_limit]]]]
func (db *Database) virtualGraphSemijoinRelationRows(ctx context.Context, args []interface{}) ([]virtualSQLRow, error) {
	if len(args) < 2 || len(args) > 6 {
		return nil, fmt.Errorf("GRAPH_SEMIJOIN requires 2 to 6 arguments")
	}
	collection := strings.TrimSpace(recordMetaToString(args[0]))
	originID := recordMetaToString(args[1])
	if collection == "" || originID == "" {
		return nil, fmt.Errorf("GRAPH_SEMIJOIN collection and origin_id must be non-empty")
	}
	col, err := db.GetCollection(collection)
	if err != nil {
		return nil, err
	}
	g := col.GetGraph()
	if g == nil {
		return nil, fmt.Errorf("GRAPH_SEMIJOIN collection %q has no graph", collection)
	}

	edgeKind := uint8(0)
	edgeType := ""
	if len(args) >= 3 && args[2] != nil {
		edgeType = recordMetaToString(args[2])
		if edgeType != "" {
			edgeKind = ResolveEdgeKind(edgeType)
			if edgeKind == 0 {
				return nil, fmt.Errorf("unknown edge kind %q", edgeType)
			}
		}
	}
	limit := func(index int) (int, error) {
		if len(args) <= index || args[index] == nil {
			return 0, nil
		}
		value, ok := toInt64(args[index])
		if !ok || value < 0 {
			return 0, fmt.Errorf("GRAPH_SEMIJOIN limit %d must be a non-negative integer", index-2)
		}
		return int(value), nil
	}
	sourceLimit, err := limit(3)
	if err != nil {
		return nil, err
	}
	originLimit, err := limit(4)
	if err != nil {
		return nil, err
	}
	candidateLimit, err := limit(5)
	if err != nil {
		return nil, err
	}

	originNode, err := db.GetNodeID(ctx, collection, originID)
	if err != nil {
		return nil, fmt.Errorf("GRAPH_SEMIJOIN origin %q: %w", originID, err)
	}
	originNeighbors, err := graphPatternNeighbors(g, originNode, 1)
	if err != nil {
		return nil, err
	}
	originShared := make(map[uint64]struct{}, len(originNeighbors))
	originSeen := 0
	for _, view := range originNeighbors {
		if edgeKind != 0 && view.Edge.GetKind() != edgeKind {
			continue
		}
		if originLimit > 0 && originSeen >= originLimit {
			break
		}
		originSeen++
		originShared[view.Edge.Target] = struct{}{}
	}
	trackSQLGraphExpansion(ctx, originSeen)
	if len(originShared) == 0 {
		return nil, nil
	}

	records, err := recordsVisibleInContext(ctx, col)
	if err != nil {
		return nil, err
	}
	type evidenceRow struct {
		candidate string
		evidence  string
		typeName  string
		kind      uint8
	}
	rowsByKey := make(map[string]evidenceRow)
	sharedByCandidate := make(map[string]map[uint64]struct{})
	for _, record := range records {
		nodeID, lookupErr := db.GetNodeID(ctx, collection, record.ID)
		if lookupErr != nil {
			continue
		}
		neighbors, neighborErr := graphPatternNeighbors(g, nodeID, 1)
		if neighborErr != nil {
			return nil, neighborErr
		}
		expanded := 0
		for _, view := range neighbors {
			if edgeKind != 0 && view.Edge.GetKind() != edgeKind {
				continue
			}
			if sourceLimit > 0 && expanded >= sourceLimit {
				break
			}
			expanded++
			if _, shared := originShared[view.Edge.Target]; !shared {
				continue
			}
			_, evidenceID, resolveErr := db.ResolveNodeID(ctx, view.Edge.Target)
			if resolveErr != nil {
				continue
			}
			// A process can legitimately have multiple SQL edge names mapped
			// to the same numeric kind (the graph store persists the kind, while
			// SQL preserves the requested name).  When the relation was filtered
			// by an explicit edge type, return that logical name instead of the
			// registry's first canonical alias.
			name := edgeType
			if name == "" {
				name = graphpkg.EdgeKindName(view.Edge.GetKind())
			}
			candidateShared := sharedByCandidate[record.ID]
			if candidateShared == nil {
				candidateShared = make(map[uint64]struct{})
				sharedByCandidate[record.ID] = candidateShared
			}
			candidateShared[view.Edge.Target] = struct{}{}
			key := fmt.Sprintf("%s\x00%s\x00%d", record.ID, evidenceID, view.Edge.GetKind())
			rowsByKey[key] = evidenceRow{candidate: record.ID, evidence: evidenceID, typeName: name, kind: view.Edge.GetKind()}
		}
		trackSQLGraphExpansion(ctx, expanded)
	}

	rows := make([]evidenceRow, 0, len(rowsByKey))
	for _, row := range rowsByKey {
		rows = append(rows, row)
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].candidate != rows[j].candidate {
			return rows[i].candidate < rows[j].candidate
		}
		if rows[i].evidence != rows[j].evidence {
			return rows[i].evidence < rows[j].evidence
		}
		return rows[i].kind < rows[j].kind
	})
	if candidateLimit > 0 {
		seenCandidates := 0
		lastCandidate := ""
		cut := len(rows)
		for i, row := range rows {
			if row.candidate != lastCandidate {
				lastCandidate = row.candidate
				seenCandidates++
				if seenCandidates > candidateLimit {
					cut = i
					break
				}
			}
		}
		rows = rows[:cut]
	}

	out := make([]virtualSQLRow, 0, len(rows))
	for _, row := range rows {
		sharedCount := len(sharedByCandidate[row.candidate])
		out = append(out, virtualSQLRow{ID: row.candidate, Values: map[string]interface{}{
			"candidate_id": row.candidate,
			"evidence_id":  row.evidence,
			"edge_type":    row.typeName,
			"shared_count": int64(sharedCount),
		}})
	}
	return out, nil
}
