package libravdb

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/xDarkicex/lexer/parser"
)

// =============================================================================
// Logical plan type
// =============================================================================

// LeidenMatchPlan is the validated logical plan produced by lowering a parsed
// COMPUTE LEIDEN statement. It carries string identifiers and hop bounds
// extracted directly from source offsets. No database access, node-ID
// resolution, or Leiden execution occurs during lowering.
//
// Collection remains empty at this stage; a later SQL binder populates it from
// the surrounding relation/CTE context.
type LeidenMatchPlan struct {
	Collection string

	SeedAlias     string
	SeedLabel     string
	TerminalAlias string
	TerminalLabel string

	EdgeKind string // "" means all edge kinds

	Direction LeidenMatchDirection
	MinHops   int
	MaxHops   int

	Options EpochLeidenOptions
}

// =============================================================================
// LowerComputeLeidenPlan
// =============================================================================

// LowerComputeLeidenPlan converts a parsed ComputeLeidenStmt into a validated
// LeidenMatchPlan. It extracts all identifiers from source offsets and enforces
// the supported logical shape: exactly two vertices, exactly one edge, finite
// hop bounds, and outbound or inbound direction.
//
// The returned plan is a newly allocated defensive copy; it does not alias
// parser-owned slices or the source buffer.
func LowerComputeLeidenPlan(
	src []byte,
	doc *parser.QueryDoc,
	stmtIndex int,
) (*LeidenMatchPlan, error) {
	if doc == nil {
		return nil, fmt.Errorf("QueryDoc must not be nil")
	}
	if stmtIndex < 0 || stmtIndex >= len(doc.ComputeLeidenStmts) {
		return nil, fmt.Errorf("stmtIndex %d out of range [0, %d)", stmtIndex, len(doc.ComputeLeidenStmts))
	}

	stmt := doc.ComputeLeidenStmts[stmtIndex]
	if stmt.MatchPath.Kind != parser.NodeKindMatchPath {
		return nil, fmt.Errorf("ComputeLeidenStmt has invalid MatchPath kind %v", stmt.MatchPath.Kind)
	}
	mp := doc.MatchPaths[stmt.MatchPath.ID]

	// Extract path nodes.
	nodes := doc.Nodes[mp.PathNodesStart : mp.PathNodesStart+mp.PathNodesCount]
	vertices := make([]parser.Vertex, 0, 2)
	edges := make([]parser.Edge, 0, 1)

	for _, ref := range nodes {
		switch ref.Kind {
		case parser.NodeKindVertex:
			vertices = append(vertices, doc.Vertexes[ref.ID])
		case parser.NodeKindEdge:
			edges = append(edges, doc.Edges[ref.ID])
		default:
			return nil, fmt.Errorf("unexpected node kind %v in MATCH path", ref.Kind)
		}
	}

	// Validate structural shape.
	switch {
	case len(vertices) == 0:
		return nil, fmt.Errorf("MATCH path must contain at least one vertex")
	case len(vertices) < 2:
		return nil, fmt.Errorf("MATCH path requires at least two vertices, got %d", len(vertices))
	case len(vertices) > 2:
		return nil, fmt.Errorf("COMPUTE LEIDEN supports exactly two vertices in MATCH path, got %d", len(vertices))
	case len(edges) == 0:
		return nil, fmt.Errorf("MATCH path must contain at least one edge")
	case len(edges) > 1:
		return nil, fmt.Errorf("COMPUTE LEIDEN supports exactly one edge in MATCH path, got %d", len(edges))
	}

	plan := &LeidenMatchPlan{}

	// ── Seed vertex (first) ──
	seedV := vertices[0]
	if seedV.Alias != 0 {
		plan.SeedAlias = string(src[seedV.Alias:seedV.AliasEnd])
	} else {
		return nil, fmt.Errorf("seed vertex must have an alias")
	}
	if seedV.LabelStart != seedV.LabelEnd {
		plan.SeedLabel = string(src[seedV.LabelStart:seedV.LabelEnd])
	}

	// ── Terminal vertex (second) ──
	termV := vertices[1]
	if termV.Alias != 0 {
		plan.TerminalAlias = string(src[termV.Alias:termV.AliasEnd])
	} else {
		return nil, fmt.Errorf("terminal vertex must have an alias")
	}
	if termV.LabelStart != termV.LabelEnd {
		plan.TerminalLabel = string(src[termV.LabelStart:termV.LabelEnd])
	}

	// ── Edge ──
	e := edges[0]

	// Direction.
	switch e.Direction {
	case 1:
		plan.Direction = LeidenMatchOutbound
	case -1:
		plan.Direction = LeidenMatchInbound
	case 0:
		return nil, fmt.Errorf("undirected edges are not supported in COMPUTE LEIDEN")
	default:
		return nil, fmt.Errorf("unsupported edge direction %d", e.Direction)
	}

	// Edge kind.
	if e.TypeStart != e.TypeEnd {
		plan.EdgeKind = string(src[e.TypeStart:e.TypeEnd])
	}

	// Quantifier.
	plan.MinHops, plan.MaxHops = resolveHopBounds(e.QuantMin, e.QuantMax)
	if plan.MinHops < 0 || plan.MaxHops < 0 {
		return nil, fmt.Errorf("hop bounds must be non-negative, got min=%d max=%d", plan.MinHops, plan.MaxHops)
	}
	if plan.MinHops > plan.MaxHops {
		return nil, fmt.Errorf("min_hops (%d) must not exceed max_hops (%d)", plan.MinHops, plan.MaxHops)
	}

	// ── Options ──
	if err := lowerLeidenOptions(src, doc, stmt, plan); err != nil {
		return nil, err
	}

	return plan, nil
}

// resolveHopBounds converts parser quantifier values into a finite inclusive
// hop interval. Unquantified edges become [1,1] (exactly one hop). Unbounded
// maxima are rejected.
func resolveHopBounds(qmin, qmax uint16) (int, int) {
	if qmin == 0 && qmax == 0 {
		return 1, 1 // unquantified single hop
	}
	min := int(qmin)
	max := int(qmax)
	if qmax == parser.QuantUnbounded {
		// Reject unbounded traversal.
		return -1, -1
	}
	return min, max
}

// lowerLeidenOptions populates plan.Options from the OPTIONS clause.
func lowerLeidenOptions(src []byte, doc *parser.QueryDoc, stmt parser.ComputeLeidenStmt, plan *LeidenMatchPlan) error {
	opts := doc.LeidenOptions[stmt.OptionsStart : stmt.OptionsStart+stmt.OptionsCount]
	seen := make(map[string]bool, len(opts))

	for _, opt := range opts {
		name := string(src[opt.NameStart:opt.NameEnd])
		nameLower := strings.ToLower(name)

		if seen[nameLower] {
			return fmt.Errorf("duplicate option %q", name)
		}
		seen[nameLower] = true

		valueStr := nodeRefSource(src, doc, opt.Value)

		switch opt.Kind {
		case parser.LeidenOptionResolution:
			v, err := parseFloat64(valueStr)
			if err != nil {
				return fmt.Errorf("option resolution: invalid numeric value %q: %w", valueStr, err)
			}
			if v <= 0 {
				return fmt.Errorf("resolution must be > 0, got %v", v)
			}
			plan.Options.Resolution = v

		case parser.LeidenOptionIterations:
			if seen["max_local_moving_passes"] {
				return fmt.Errorf("iterations conflicts with max_local_moving_passes")
			}
			v, err := parsePositiveInt(valueStr)
			if err != nil {
				return fmt.Errorf("option iterations: invalid value %q: %w", valueStr, err)
			}
			plan.Options.MaxLocalMovingPasses = v

		case parser.LeidenOptionMaxLevels:
			v, err := parsePositiveInt(valueStr)
			if err != nil {
				return fmt.Errorf("option max_levels: invalid value %q: %w", valueStr, err)
			}
			plan.Options.MaxLevels = v

		case parser.LeidenOptionMaxLocalMovingPasses:
			if seen["iterations"] {
				return fmt.Errorf("max_local_moving_passes conflicts with iterations")
			}
			v, err := parsePositiveInt(valueStr)
			if err != nil {
				return fmt.Errorf("option max_local_moving_passes: invalid value %q: %w", valueStr, err)
			}
			plan.Options.MaxLocalMovingPasses = v

		case parser.LeidenOptionMinHops:
			v, err := parseNonNegativeInt(valueStr)
			if err != nil {
				return fmt.Errorf("option min_hops: invalid value %q: %w", valueStr, err)
			}
			plan.MinHops = v

		case parser.LeidenOptionMaxHops:
			v, err := parseNonNegativeInt(valueStr)
			if err != nil {
				return fmt.Errorf("option max_hops: invalid value %q: %w", valueStr, err)
			}
			plan.MaxHops = v

		case parser.LeidenOptionMaxVertices:
			v, err := parseNonNegativeInt(valueStr)
			if err != nil {
				return fmt.Errorf("option max_vertices: invalid value %q: %w", valueStr, err)
			}
			plan.Options.MaxVertices = v

		case parser.LeidenOptionMaxEdges:
			v, err := parseNonNegativeInt(valueStr)
			if err != nil {
				return fmt.Errorf("option max_edges: invalid value %q: %w", valueStr, err)
			}
			plan.Options.MaxEdges = v

		case parser.LeidenOptionEdgeKind:
			// Edge kind may be an unquoted identifier or a quoted string.
			plan.EdgeKind = stripOptionalQuotes(valueStr)

		case parser.LeidenOptionDirection:
			switch strings.ToLower(valueStr) {
			case "outbound":
				plan.Direction = LeidenMatchOutbound
			case "inbound":
				plan.Direction = LeidenMatchInbound
			default:
				return fmt.Errorf("option direction: unsupported value %q (want outbound or inbound)", valueStr)
			}
		}
	}

	// Cross-option validation after parsing all options.
	if plan.MinHops > plan.MaxHops {
		return fmt.Errorf("min_hops (%d) must not exceed max_hops (%d)", plan.MinHops, plan.MaxHops)
	}
	if plan.MaxHops <= 0 {
		return fmt.Errorf("max_hops must be > 0, got %d", plan.MaxHops)
	}

	return nil
}

// =============================================================================
// Numeric parsing helpers
// =============================================================================

// parseFloat64 parses a float64 from a source literal string.
func parseFloat64(s string) (float64, error) {
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, err
	}
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, fmt.Errorf("invalid float value %q", s)
	}
	return v, nil
}

// parsePositiveInt parses a strictly positive int from a source literal.
func parsePositiveInt(s string) (int, error) {
	v, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return 0, err
	}
	if v <= 0 {
		return 0, fmt.Errorf("must be > 0, got %d", v)
	}
	return int(v), nil
}

// parseNonNegativeInt parses a non-negative int from a source literal.
func parseNonNegativeInt(s string) (int, error) {
	v, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("expected integer, got %q", s)
	}
	if v < 0 {
		return 0, fmt.Errorf("must be >= 0, got %d", v)
	}
	return int(v), nil
}

// stripOptionalQuotes removes surrounding single quotes from a string value.
func stripOptionalQuotes(s string) string {
	if len(s) >= 2 && s[0] == '\'' && s[len(s)-1] == '\'' {
		return s[1 : len(s)-1]
	}
	return s
}

// nodeRefSource extracts the source text for a NodeRef's underlying value.
// The NodeRef.Kind determines which slice to index into for the offsets.
func nodeRefSource(src []byte, doc *parser.QueryDoc, ref parser.NodeRef) string {
	switch ref.Kind {
	case parser.NodeKindNumber:
		n := doc.Numbers[ref.ID]
		return string(src[n.Start:n.End])
	case parser.NodeKindIdentifier:
		id := doc.Identifiers[ref.ID]
		return string(src[id.Start:id.End])
	case parser.NodeKindString:
		s := doc.Strings[ref.ID]
		return string(src[s.Start:s.End])
	default:
		return ""
	}
}
