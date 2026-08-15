package libravdb

import (
	"reflect"
	"testing"

	"github.com/xDarkicex/lexer/parser"
)

// lower is a test helper that parses src and lowers the first ComputeLeidenStmt.
func lower(t *testing.T, src string) *LeidenMatchPlan {
	t.Helper()
	var doc parser.QueryDoc
	if err := parser.Parse([]byte(src), &doc); err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if len(doc.ComputeLeidenStmts) == 0 {
		t.Fatalf("no ComputeLeidenStmts in parsed doc")
	}
	plan, err := LowerComputeLeidenPlan([]byte(src), &doc, 0)
	if err != nil {
		t.Fatalf("LowerComputeLeidenPlan: %v", err)
	}
	return plan
}

// lowerErr asserts that lowering fails.
func lowerErr(t *testing.T, src string) {
	t.Helper()
	var doc parser.QueryDoc
	if err := parser.Parse([]byte(src), &doc); err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if len(doc.ComputeLeidenStmts) == 0 {
		t.Fatalf("no ComputeLeidenStmts in parsed doc")
	}
	_, err := LowerComputeLeidenPlan([]byte(src), &doc, 0)
	if err == nil {
		t.Fatal("expected lowering error, got nil")
	}
	t.Logf("correctly rejected: %v", err)
}

// =============================================================================
// Test 1: Basic outbound plan
// =============================================================================

func TestLeidenPlan_BasicOutbound(t *testing.T) {
	plan := lower(t, `COMPUTE LEIDEN FROM MATCH (s:seeds)-[:CONNECTED_TO*1..3]->(target)
		OPTIONS (resolution = 1.0, iterations = 2)`)

	assertStr(t, plan.SeedAlias, "s", "SeedAlias")
	assertStr(t, plan.SeedLabel, "seeds", "SeedLabel")
	assertStr(t, plan.TerminalAlias, "target", "TerminalAlias")
	assertStr(t, plan.TerminalLabel, "", "TerminalLabel")
	assertStr(t, plan.EdgeKind, "CONNECTED_TO", "EdgeKind")

	if plan.Direction != LeidenMatchOutbound {
		t.Errorf("Direction: want LeidenMatchOutbound, got %d", plan.Direction)
	}
	if plan.MinHops != 1 {
		t.Errorf("MinHops: want 1, got %d", plan.MinHops)
	}
	if plan.MaxHops != 3 {
		t.Errorf("MaxHops: want 3, got %d", plan.MaxHops)
	}
	if plan.Options.Resolution != 1.0 {
		t.Errorf("Resolution: want 1.0, got %v", plan.Options.Resolution)
	}
	if plan.Options.MaxLocalMovingPasses != 2 {
		t.Errorf("MaxLocalMovingPasses: want 2, got %d", plan.Options.MaxLocalMovingPasses)
	}
	if plan.Collection != "" {
		t.Errorf("Collection: want empty, got %q", plan.Collection)
	}
	t.Log("✅ basic outbound plan")
}

// =============================================================================
// Test 2: Unquantified edge → single hop
// =============================================================================

func TestLeidenPlan_UnquantifiedEdge(t *testing.T) {
	plan := lower(t, `COMPUTE LEIDEN FROM MATCH (s:seeds)-[:CONNECTED_TO]->(target)`)

	if plan.MinHops != 1 {
		t.Errorf("MinHops: want 1, got %d", plan.MinHops)
	}
	if plan.MaxHops != 1 {
		t.Errorf("MaxHops: want 1, got %d", plan.MaxHops)
	}
	t.Log("✅ unquantified edge → [1,1]")
}

// =============================================================================
// Test 3: Inbound path
// =============================================================================

func TestLeidenPlan_InboundPath(t *testing.T) {
	plan := lower(t, `COMPUTE LEIDEN FROM MATCH (s:seeds)<-[:CONNECTED_TO*2..4]-(target)`)

	if plan.Direction != LeidenMatchInbound {
		t.Errorf("Direction: want LeidenMatchInbound, got %d", plan.Direction)
	}
	if plan.MinHops != 2 || plan.MaxHops != 4 {
		t.Errorf("Hops: want [2,4], got [%d,%d]", plan.MinHops, plan.MaxHops)
	}
	t.Log("✅ inbound path with quantifier")
}

// =============================================================================
// Test 4: Untyped edge
// =============================================================================

func TestLeidenPlan_UntypedEdge(t *testing.T) {
	plan := lower(t, `COMPUTE LEIDEN FROM MATCH (s:seeds)-[*1..2]->(target)`)

	if plan.EdgeKind != "" {
		t.Errorf("EdgeKind: want empty, got %q", plan.EdgeKind)
	}
	t.Log("✅ untyped edge → EdgeKind=\"\"")
}

// =============================================================================
// Test 5: Option overrides
// =============================================================================

func TestLeidenPlan_OptionOverrides(t *testing.T) {
	plan := lower(t, `COMPUTE LEIDEN FROM MATCH (a)-[:LINK*1..3]->(b) OPTIONS (
		resolution = 0.5,
		max_levels = 4,
		max_local_moving_passes = 8,
		min_hops = 2,
		max_hops = 5,
		max_vertices = 100,
		max_edges = 200,
		direction = inbound
	)`)

	if plan.Options.Resolution != 0.5 {
		t.Errorf("Resolution: want 0.5, got %v", plan.Options.Resolution)
	}
	if plan.Options.MaxLevels != 4 {
		t.Errorf("MaxLevels: want 4, got %d", plan.Options.MaxLevels)
	}
	if plan.Options.MaxLocalMovingPasses != 8 {
		t.Errorf("MaxLocalMovingPasses: want 8, got %d", plan.Options.MaxLocalMovingPasses)
	}
	if plan.MinHops != 2 {
		t.Errorf("MinHops: want 2 (overridden), got %d", plan.MinHops)
	}
	if plan.MaxHops != 5 {
		t.Errorf("MaxHops: want 5 (overridden), got %d", plan.MaxHops)
	}
	if plan.Options.MaxVertices != 100 {
		t.Errorf("MaxVertices: want 100, got %d", plan.Options.MaxVertices)
	}
	if plan.Options.MaxEdges != 200 {
		t.Errorf("MaxEdges: want 200, got %d", plan.Options.MaxEdges)
	}
	if plan.Direction != LeidenMatchInbound {
		t.Errorf("Direction: want inbound, got %d", plan.Direction)
	}
	t.Log("✅ all option overrides mapped correctly")
}

// =============================================================================
// Test 6: Reject structural violations
// =============================================================================

func TestLeidenPlan_RejectTooManyVertices(t *testing.T) {
	lowerErr(t, `COMPUTE LEIDEN FROM MATCH (a)-[:E]->(b)-[:E]->(c)`)
}

func TestLeidenPlan_RejectTooManyEdges(t *testing.T) {
	// MATCH with two edges but parser only supports one — test structural validation.
	lowerErr(t, `COMPUTE LEIDEN FROM MATCH (a)-[:E]->(b)-[:E]->(c)`)
}

func TestLeidenPlan_RejectMissingAlias(t *testing.T) {
	// Vertex without alias: (:Label) — parser allows this, lowering rejects.
	lowerErr(t, `COMPUTE LEIDEN FROM MATCH (:seeds)-[:E]->(target)`)
}

func TestLeidenPlan_RejectMissingTerminalAlias(t *testing.T) {
	lowerErr(t, `COMPUTE LEIDEN FROM MATCH (s:seeds)-[:E]->(:Label)`)
}

func TestLeidenPlan_RejectUndirectedEdge(t *testing.T) {
	lowerErr(t, `COMPUTE LEIDEN FROM MATCH (a)-[:E]-(b)`)
}

func TestLeidenPlan_RejectUnboundedStar(t *testing.T) {
	lowerErr(t, `COMPUTE LEIDEN FROM MATCH (a)-[:E]->*(b)`)
}

func TestLeidenPlan_RejectUnboundedPlus(t *testing.T) {
	lowerErr(t, `COMPUTE LEIDEN FROM MATCH (a)-[:E]->+(b)`)
}

// =============================================================================
// Test 7: Reject invalid option values
// =============================================================================

func TestLeidenPlan_RejectFractionalIntOption(t *testing.T) {
	lowerErr(t, `COMPUTE LEIDEN FROM MATCH (a)-[:E]->(b) OPTIONS (max_levels = 1.5)`)
}

func TestLeidenPlan_RejectNegativeBudget(t *testing.T) {
	// -5 lexes as KindDash + KindNumber; the parser rejects this before
	// lowering sees it. Verify the parser-level error.
	var doc parser.QueryDoc
	err := parser.Parse([]byte(`COMPUTE LEIDEN FROM MATCH (a)-[:E]->(b) OPTIONS (max_vertices = -5)`), &doc)
	if err == nil {
		t.Fatal("expected parser error for negative literal")
	}
	t.Logf("parser rejected negative literal: %v", err)
}

func TestLeidenPlan_RejectNegativeBudget_LoweringGuard(t *testing.T) {
	// Use string value "-5" that parses as an identifier to reach the
	// lowering's numeric validation.
	var doc parser.QueryDoc
	src := []byte(`COMPUTE LEIDEN FROM MATCH (a)-[:E]->(b) OPTIONS (max_vertices = '-5')`)
	if err := parser.Parse(src, &doc); err != nil {
		t.Fatalf("Parse: %v", err)
	}
	_, err := LowerComputeLeidenPlan(src, &doc, 0)
	if err == nil {
		t.Fatal("expected lowering error for negative integer via string")
	}
	t.Logf("lowering rejected negative int: %v", err)
}

func TestLeidenPlan_RejectZeroResolution(t *testing.T) {
	lowerErr(t, `COMPUTE LEIDEN FROM MATCH (a)-[:E]->(b) OPTIONS (resolution = 0)`)
}

func TestLeidenPlan_RejectMinHopsGtMaxHops(t *testing.T) {
	lowerErr(t, `COMPUTE LEIDEN FROM MATCH (a)-[:E*5..3]->(b)`)
}

func TestLeidenPlan_RejectOptionMinHopsGtMaxHops(t *testing.T) {
	lowerErr(t, `COMPUTE LEIDEN FROM MATCH (a)-[:E]->(b) OPTIONS (min_hops = 5, max_hops = 3)`)
}

func TestLeidenPlan_RejectDuplicateOptions(t *testing.T) {
	// The parser already rejects duplicate option names. Verify that.
	var doc parser.QueryDoc
	err := parser.Parse([]byte(`COMPUTE LEIDEN FROM MATCH (a)-[:E]->(b) OPTIONS (resolution = 1.0, resolution = 2.0)`), &doc)
	if err == nil {
		t.Fatal("expected parser error for duplicate options")
	}
	t.Logf("parser rejected duplicate option: %v", err)
}

func TestLeidenPlan_RejectDuplicateOptions_LoweringGuard(t *testing.T) {
	// Build a QueryDoc with duplicate options by hand to test the
	// lowering's defensive duplicate check.
	doc := &parser.QueryDoc{
		MatchPaths: []parser.MatchPath{{
			ID:             0,
			PathNodesStart: 0,
			PathNodesCount: 3,
		}},
		Vertexes: []parser.Vertex{
			{ID: 0, Alias: 0, AliasEnd: 1, LabelStart: 0, LabelEnd: 0},
			{ID: 1, Alias: 2, AliasEnd: 3, LabelStart: 0, LabelEnd: 0},
		},
		Edges: []parser.Edge{
			{ID: 0, Direction: 1, TypeStart: 5, TypeEnd: 6, QuantMin: 0, QuantMax: 0},
		},
		Nodes: []parser.NodeRef{
			{Kind: parser.NodeKindVertex, ID: 0},
			{Kind: parser.NodeKindEdge, ID: 0},
			{Kind: parser.NodeKindVertex, ID: 1},
		},
		ComputeLeidenStmts: []parser.ComputeLeidenStmt{{
			ID:           0,
			MatchPath:    parser.NodeRef{Kind: parser.NodeKindMatchPath, ID: 0},
			OptionsStart: 0,
			OptionsCount: 2,
		}},
		LeidenOptions: []parser.LeidenOption{
			{Kind: parser.LeidenOptionResolution, NameStart: 0, NameEnd: 10, Value: parser.NodeRef{Kind: parser.NodeKindNumber, ID: 0}},
			{Kind: parser.LeidenOptionResolution, NameStart: 0, NameEnd: 10, Value: parser.NodeRef{Kind: parser.NodeKindNumber, ID: 1}},
		},
		Numbers: []parser.Number{
			{ID: 0, Start: 0, End: 3},
			{ID: 1, Start: 0, End: 3},
		},
	}
	src := []byte("resolution=1.0resolution=2.0aabbEE000111")
	_, err := LowerComputeLeidenPlan(src, doc, 0)
	if err == nil {
		t.Fatal("expected lowering error for duplicate options")
	}
	t.Logf("lowering rejected duplicate option: %v", err)
}

func TestLeidenPlan_RejectConflictingIterationsAndPasses(t *testing.T) {
	lowerErr(t, `COMPUTE LEIDEN FROM MATCH (a)-[:E]->(b) OPTIONS (iterations = 5, max_local_moving_passes = 3)`)
}

func TestLeidenPlan_RejectInvalidDirection(t *testing.T) {
	lowerErr(t, `COMPUTE LEIDEN FROM MATCH (a)-[:E]->(b) OPTIONS (direction = sideways)`)
}

func TestLeidenPlan_RejectInvalidNumericLiteral(t *testing.T) {
	lowerErr(t, `COMPUTE LEIDEN FROM MATCH (a)-[:E]->(b) OPTIONS (resolution = abc)`)
}

// =============================================================================
// Test 8: Reject API misuse
// =============================================================================

func TestLeidenPlan_RejectNilDoc(t *testing.T) {
	_, err := LowerComputeLeidenPlan([]byte(""), nil, 0)
	if err == nil {
		t.Fatal("expected error for nil QueryDoc")
	}
	t.Logf("nil doc: %v", err)
}

func TestLeidenPlan_RejectInvalidStmtIndex(t *testing.T) {
	var doc parser.QueryDoc
	_ = parser.Parse([]byte(`COMPUTE LEIDEN FROM MATCH (a)-[:E]->(b)`), &doc)

	_, err := LowerComputeLeidenPlan([]byte(""), &doc, -1)
	if err == nil {
		t.Fatal("expected error for stmtIndex -1")
	}
	t.Logf("stmtIndex -1: %v", err)

	_, err = LowerComputeLeidenPlan([]byte(""), &doc, 999)
	if err == nil {
		t.Fatal("expected error for stmtIndex 999")
	}
	t.Logf("stmtIndex 999: %v", err)
}

// =============================================================================
// Test 9: Defensive copy
// =============================================================================

func TestLeidenPlan_DefensiveCopy(t *testing.T) {
	src := `COMPUTE LEIDEN FROM MATCH (s:seeds)-[:CONNECTED_TO*1..3]->(target) OPTIONS (resolution = 1.0)`
	var doc parser.QueryDoc
	if err := parser.Parse([]byte(src), &doc); err != nil {
		t.Fatalf("Parse: %v", err)
	}

	plan, err := LowerComputeLeidenPlan([]byte(src), &doc, 0)
	if err != nil {
		t.Fatalf("first Lower: %v", err)
	}

	// Mutate the plan.
	plan.SeedAlias = "MUTATED"
	plan.EdgeKind = "MUTATED"
	plan.MinHops = 999
	plan.Options.Resolution = 99.9

	// Lower again from the same AST.
	plan2, err := LowerComputeLeidenPlan([]byte(src), &doc, 0)
	if err != nil {
		t.Fatalf("second Lower: %v", err)
	}

	if plan2.SeedAlias != "s" {
		t.Errorf("SeedAlias mutated: want 's', got %q", plan2.SeedAlias)
	}
	if plan2.EdgeKind != "CONNECTED_TO" {
		t.Errorf("EdgeKind mutated: want 'CONNECTED_TO', got %q", plan2.EdgeKind)
	}
	if plan2.MinHops != 1 {
		t.Errorf("MinHops mutated: want 1, got %d", plan2.MinHops)
	}
	if plan2.Options.Resolution != 1.0 {
		t.Errorf("Resolution mutated: want 1.0, got %v", plan2.Options.Resolution)
	}
	t.Log("✅ defensive copy: mutation does not affect source")
}

// =============================================================================
// Test 10: Determinism
// =============================================================================

func TestLeidenPlan_Determinism(t *testing.T) {
	src := `COMPUTE LEIDEN FROM MATCH (s:seeds)-[:CONNECTED_TO*2..4]->(target)
		OPTIONS (resolution = 0.5, iterations = 3, max_vertices = 100)`

	var first *LeidenMatchPlan
	for i := 0; i < 10; i++ {
		var doc parser.QueryDoc
		if err := parser.Parse([]byte(src), &doc); err != nil {
			t.Fatalf("Parse iter %d: %v", i, err)
		}
		plan, err := LowerComputeLeidenPlan([]byte(src), &doc, 0)
		if err != nil {
			t.Fatalf("Lower iter %d: %v", i, err)
		}
		if i == 0 {
			first = plan
			continue
		}
		if !reflect.DeepEqual(plan, first) {
			t.Fatalf("iter %d: plan differs from first call", i)
		}
	}
	t.Log("✅ determinism across 10 calls")
}

// =============================================================================
// Test 11: Edge kind via option
// =============================================================================

func TestLeidenPlan_EdgeKindViaOption(t *testing.T) {
	plan := lower(t, `COMPUTE LEIDEN FROM MATCH (a)-[*1..2]->(b) OPTIONS (edge_kind = 'LINK')`)

	if plan.EdgeKind != "LINK" {
		t.Errorf("EdgeKind: want 'LINK' (from option), got %q", plan.EdgeKind)
	}
	t.Log("✅ edge_kind option overrides edge type")
}

// =============================================================================
// Test 12: Unquantified edge in source, quantified via options
// =============================================================================

func TestLeidenPlan_HopBoundsViaOptions(t *testing.T) {
	plan := lower(t, `COMPUTE LEIDEN FROM MATCH (a)-[:E]->(b) OPTIONS (min_hops = 2, max_hops = 4)`)

	if plan.MinHops != 2 || plan.MaxHops != 4 {
		t.Errorf("Hops: want [2,4] from options, got [%d,%d]", plan.MinHops, plan.MaxHops)
	}
	t.Log("✅ hop bounds overridden via options")
}

// =============================================================================
// Helper
// =============================================================================

func assertStr(t *testing.T, got, want, field string) {
	t.Helper()
	if got != want {
		t.Errorf("%s: want %q, got %q", field, want, got)
	}
}
