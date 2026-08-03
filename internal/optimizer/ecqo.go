package optimizer

import (
	"errors"
	"fmt"
	"math"

	"github.com/xDarkicex/lexer/parser"
	"github.com/xDarkicex/libravdb/internal/catalog"
)

var (
	ErrUnsupportedQuery = errors.New("unsupported query shape")
)

// QueryKind classifies the query type for executor dispatch.
type QueryKind uint8

const (
	QueryKindKNN        QueryKind = iota // vector similarity search (default)
	QueryKindGraph                       // graph pattern matching via GRAPH_TABLE
	QueryKindRelational                  // relational exact-match / range scan
)

// RelationalPredicate is a single WHERE clause predicate extracted for relational execution.
type RelationalPredicate struct {
	Column   string // column name resolved from source bytes
	Operator uint8  // lexer.KindEquals, KindGreaterThan, KindLessThan, etc.
	Value    []byte // literal value from source bytes (number string or unquoted string)
}

// GraphEdgePlan is a single edge extracted from the MATCH path,
// carrying its direction and quantifier bounds for traversal.
type GraphEdgePlan struct {
	Direction int8   // -1=inbound, 0=undirected, 1=outbound (from parser.Edge.Direction)
	QuantMin  uint16 // minimum hops (0 for ->*)
	QuantMax  uint16 // maximum hops (0=default→1, QuantUnbounded for ->+/->*)
}

// PhysicalPlan represents the executable operations derived from the AST.
// It maps the abstract SQL constructs to LibraVDB's concrete capabilities.
type PhysicalPlan struct {
	// Collection identity
	CollectionOID  uint32
	CollectionName string // resolved from source bytes at plan time

	// Query classification
	Kind QueryKind

	// KNN / vector search
	HasVectorSearch bool
	QueryVector     []float32 // Extracted from AST
	VectorIndexOID  uint32
	Similarity      float32
	Limit           int

	// Graph traversal — populated when Kind == QueryKindGraph
	HasGraphTraversal bool
	GraphEdges        []GraphEdgePlan
	MaxHops           int

	// Seed selection
	HasExplicitSeed   bool
	ExplicitSeedID    uint64
	HasVectorAnchor   bool
	GraphAnchorVector []float32

	// Relational query fields — populated when Kind == QueryKindRelational
	HasRelationalQuery bool
	Predicates         []RelationalPredicate
	Projections        []string // SELECT column list (empty = all columns)
	OrderBy            string   // column name for ORDER BY (empty = none)
	IsDesc             bool     // ORDER BY DESC
}

// Optimizer is the Exact Cardinality Quantized Optimizer (ECQO).
type Optimizer struct {
	catalog *catalog.Catalog
}

func NewOptimizer(cat *catalog.Catalog) *Optimizer {
	return &Optimizer{catalog: cat}
}

// Optimize maps a bound AST to a PhysicalPlan.
func (o *Optimizer) Optimize(doc *parser.QueryDoc, src []byte) (*PhysicalPlan, error) {
	if len(doc.SelectStmts) == 0 {
		return nil, ErrUnsupportedQuery
	}

	stmt := &doc.SelectStmts[0]
	plan := &PhysicalPlan{
		Limit: -1, // No limit by default
	}

	// 1. Resolve FROM clause
	if stmt.FromTable.Kind == parser.NodeKindTableExpr {
		t := &doc.TableExprs[stmt.FromTable.ID]
		plan.CollectionOID = t.TableOID
		plan.CollectionName = string(src[t.Start:t.End])
	} else if stmt.FromTable.Kind == parser.NodeKindGraphTable {
		gt := &doc.GraphTables[stmt.FromTable.ID]
		plan.CollectionOID = gt.TableOID
		plan.CollectionName = string(src[gt.TableStart:gt.TableEnd])
		plan.Kind = QueryKindGraph
		plan.HasGraphTraversal = true

		// Parse MatchPath into GraphEdgePlans
		if gt.MatchPath.Kind == parser.NodeKindMatchPath {
			mp := &doc.MatchPaths[gt.MatchPath.ID]
			plan.MaxHops = 0
			for i := int32(0); i < mp.PathNodesCount; i++ {
				ref := doc.Nodes[mp.PathNodesStart+i]
				if ref.Kind != parser.NodeKindEdge {
					continue
				}
				e := &doc.Edges[ref.ID]
				gep := GraphEdgePlan{
					Direction: e.Direction,
					QuantMin:  e.QuantMin,
					QuantMax:  e.QuantMax,
				}
				plan.GraphEdges = append(plan.GraphEdges, gep)

				// Compute MaxHops: sum QuantMax across all edges
				max := int(e.QuantMax)
				if e.QuantMax == 0 {
					if e.QuantMin == 0 {
						max = 1 // default: exactly 1 hop
					} else {
						max = 1 << 20 // ->+ : unbounded
					}
				}
				plan.MaxHops += max
			}
			if plan.MaxHops == 0 {
				plan.MaxHops = 1
			}
		}
	} else {
		return nil, fmt.Errorf("unsupported FROM clause kind")
	}

	// 2. Map WHERE clause (Vector Search + Exact Filters)
	if stmt.WhereExpr.Kind == parser.NodeKindUnknown {
		// Full-scan: FROM table with no WHERE → cursor iteration
		if plan.Kind == QueryKindKNN {
			plan.Kind = QueryKindRelational
			plan.HasRelationalQuery = true
		}
	} else {
		whereNode := stmt.WhereExpr

		// Walk WHERE tree for relational predicates (kind-agnostic: extract first, decide later)
		o.extractRelationalPredicates(doc, src, plan, whereNode)

		// BETWEEN: x BETWEEN lower AND upper → >= lower, <= upper
		if whereNode.Kind == parser.NodeKindBetweenExpr {
			bw := &doc.BetweenExprs[whereNode.ID]
			if bw.Expr.Kind == parser.NodeKindIdentifier {
				id := &doc.Identifiers[bw.Expr.ID]
				if id.ResolvedKind == parser.ResolvedKindColumn {
					o.setRelationalKind(plan)
					col := string(src[id.Start:id.End])
					if bw.Lower.Kind == parser.NodeKindNumber {
						plan.Predicates = append(plan.Predicates, RelationalPredicate{
							Column: col, Operator: 13, // >=
							Value: src[doc.Numbers[bw.Lower.ID].Start:doc.Numbers[bw.Lower.ID].End],
						})
					}
					if bw.Upper.Kind == parser.NodeKindNumber {
						plan.Predicates = append(plan.Predicates, RelationalPredicate{
							Column: col, Operator: 14, // <=
							Value: src[doc.Numbers[bw.Upper.ID].Start:doc.Numbers[bw.Upper.ID].End],
						})
					}
				}
			}
		}

		// IN: x IN (1, 2, 3) → range [min, max] for dense ordered values
		if whereNode.Kind == parser.NodeKindInExpr {
			inNode := &doc.InExprs[whereNode.ID]
			if inNode.Expr.Kind == parser.NodeKindIdentifier {
				id := &doc.Identifiers[inNode.Expr.ID]
				if id.ResolvedKind == parser.ResolvedKindColumn {
					o.setRelationalKind(plan)
					col := string(src[id.Start:id.End])
					var vals [][]byte
					for i := int32(0); i < inNode.ListCount; i++ {
						item := doc.Nodes[inNode.ListStart+i]
						if item.Kind == parser.NodeKindNumber {
							vals = append(vals, src[doc.Numbers[item.ID].Start:doc.Numbers[item.ID].End])
						} else if item.Kind == parser.NodeKindString {
							vals = append(vals, src[doc.Strings[item.ID].Start+1:doc.Strings[item.ID].End-1])
						}
					}
					if len(vals) >= 2 {
						// Emit range [min, max]
						minV, maxV := vals[0], vals[0]
						for _, v := range vals[1:] {
							if string(v) < string(minV) { minV = v }
							if string(v) > string(maxV) { maxV = v }
						}
						plan.Predicates = append(plan.Predicates,
							RelationalPredicate{Column: col, Operator: 13, Value: minV},  // >= min
							RelationalPredicate{Column: col, Operator: 14, Value: maxV}) // <= max
					} else if len(vals) == 1 {
						plan.Predicates = append(plan.Predicates,
							RelationalPredicate{Column: col, Operator: 12, Value: vals[0]})
					}
				}
			}
		}

		// Check for vector-anchored graph query:
		// WHERE clause on a graph query that also has a VectorFunc
		if plan.Kind == QueryKindGraph {
			o.extractVectorAnchor(doc, src, stmt, plan, whereNode)
		}

		// Check for pgvector distance operators: vec <-> literal
		if whereNode.Kind == parser.NodeKindBinaryExpr {
			be := &doc.BinaryExprs[whereNode.ID]
			if be.Left.Kind == parser.NodeKindIdentifier {
				id := &doc.Identifiers[be.Left.ID]
				if id.ResolvedKind == parser.ResolvedKindVector || id.ResolvedKind == parser.ResolvedKindColumn {
					if isPgVectorOp(be.Operator) {
						plan.HasVectorSearch = true
						plan.VectorIndexOID = id.TableOID
						if be.Right.Kind == parser.NodeKindString {
							plan.QueryVector = parseVectorLiteral(doc, src, be.Right.ID)
						}
					}
				}
			}

			// Unwrap binary expr if it's SIMILARITY(...) > threshold
			// We assume the left side is the VectorFunc for now
			// e.g. SIMILARITY(...) > 0.5
			if be.Left.Kind == parser.NodeKindVectorFunc {
				whereNode = be.Left
				if be.Right.Kind == parser.NodeKindNumber {
					num := &doc.Numbers[be.Right.ID]
					threshStr := string(src[num.Start:num.End])
					var threshold float32
					fmt.Sscanf(threshStr, "%f", &threshold)
					plan.Similarity = threshold
				}
			}
		}

		if whereNode.Kind == parser.NodeKindVectorFunc {
			vf := &doc.VectorFuncs[whereNode.ID]

			// We expect VectorA to be the column/index, VectorB to be the literal
			if vf.VectorA.Kind == parser.NodeKindIdentifier {
				id := &doc.Identifiers[vf.VectorA.ID]
				if id.ResolvedKind == parser.ResolvedKindVector || id.ResolvedKind == parser.ResolvedKindColumn {
					plan.HasVectorSearch = true
					plan.VectorIndexOID = id.TableOID // For columns, this is the Table OID, which matches CollectionOID

					if vf.VectorB.Kind == parser.NodeKindString {
						sl := &doc.Strings[vf.VectorB.ID]
						// Strip quotes
						val := string(src[sl.Start+1 : sl.End-1])

						// Very basic float array parsing: [0.1, 0.2, 0.3]
						// Strip brackets
						if len(val) >= 2 && val[0] == '[' && val[len(val)-1] == ']' {
							val = val[1 : len(val)-1]
						}

						// Split by comma and parse
						var floats []float32

						// Basic tokenizer
						start := 0
						for i := 0; i <= len(val); i++ {
							if i == len(val) || val[i] == ',' {
								part := val[start:i]
								// trim spaces
								for len(part) > 0 && part[0] == ' ' { part = part[1:] }
								for len(part) > 0 && part[len(part)-1] == ' ' { part = part[:len(part)-1] }

								if len(part) > 0 {
									var f float32
									fmt.Sscanf(part, "%f", &f)
									floats = append(floats, f)
								}
								start = i + 1
							}
						}
						plan.QueryVector = floats
					}
				}
			}
		}
	}

	// 3. Map ORDER BY
	if stmt.OrderBy.Kind == parser.NodeKindIdentifier {
		id := &doc.Identifiers[stmt.OrderBy.ID]
		plan.OrderBy = string(src[id.Start:id.End])
		plan.IsDesc = stmt.IsDesc
	}

	// 4. Extract projection columns
	if stmt.ProjectionsCount > 0 {
		for i := int32(0); i < stmt.ProjectionsCount; i++ {
			proj := &doc.Projections[stmt.ProjectionsStart+i]
			if proj.Expr.Kind == parser.NodeKindIdentifier {
				id := &doc.Identifiers[proj.Expr.ID]
				plan.Projections = append(plan.Projections, string(src[id.Start:id.End]))
			}
		}
	}

	// 5. Map LIMIT
	if stmt.Limit >= 0 {
		num := &doc.Numbers[stmt.Limit]
		// Parse string to int
		limitStr := string(src[num.Start:num.End])
		var l int
		fmt.Sscanf(limitStr, "%d", &l)
		plan.Limit = l
	}

	return plan, nil
}

// extractVectorAnchor detects a vector predicate in the WHERE clause of a graph query
// and populates HasVectorAnchor / GraphAnchorVector on the plan.
func (o *Optimizer) extractVectorAnchor(doc *parser.QueryDoc, src []byte, stmt *parser.SelectStmt, plan *PhysicalPlan, whereNode parser.NodeRef) {
	// Walk the expression tree looking for a VectorFunc
	o.findVectorFunc(doc, src, whereNode, plan)
}

func (o *Optimizer) findVectorFunc(doc *parser.QueryDoc, src []byte, node parser.NodeRef, plan *PhysicalPlan) {
	if node.Kind == parser.NodeKindVectorFunc {
		vf := &doc.VectorFuncs[node.ID]
		if vf.VectorB.Kind == parser.NodeKindString {
			sl := &doc.Strings[vf.VectorB.ID]
			val := string(src[sl.Start+1 : sl.End-1])
			if len(val) >= 2 && val[0] == '[' && val[len(val)-1] == ']' {
				val = val[1 : len(val)-1]
			}
			var floats []float32
			start := 0
			for i := 0; i <= len(val); i++ {
				if i == len(val) || val[i] == ',' {
					part := val[start:i]
					for len(part) > 0 && part[0] == ' ' { part = part[1:] }
					for len(part) > 0 && part[len(part)-1] == ' ' { part = part[:len(part)-1] }
					if len(part) > 0 {
						var f float32
						fmt.Sscanf(part, "%f", &f)
						floats = append(floats, f)
					}
					start = i + 1
				}
			}
			if len(floats) > 0 {
				plan.HasVectorAnchor = true
				plan.GraphAnchorVector = floats
			}
		}
		return
	}
	if node.Kind == parser.NodeKindBinaryExpr {
		be := &doc.BinaryExprs[node.ID]
		o.findVectorFunc(doc, src, be.Left, plan)
		o.findVectorFunc(doc, src, be.Right, plan)
	}
}

// setRelationalKind marks the plan as relational if no vector/graph kind is set.
func (o *Optimizer) setRelationalKind(plan *PhysicalPlan) {
	plan.HasRelationalQuery = true
	if plan.Kind == QueryKindKNN && !plan.HasVectorSearch {
		plan.Kind = QueryKindRelational
	}
}

// extractRelationalPredicates recursively walks a WHERE expression tree,
// decomposing AND nodes and collecting leaf predicates (Identifier op Literal).
func (o *Optimizer) extractRelationalPredicates(doc *parser.QueryDoc, src []byte, plan *PhysicalPlan, node parser.NodeRef) {
	if node.Kind != parser.NodeKindBinaryExpr {
		return
	}
	be := &doc.BinaryExprs[node.ID]

	// AND decomposition: recurse into both sides
	if be.Operator == 25 { // lexer.KindAnd
		o.extractRelationalPredicates(doc, src, plan, be.Left)
		o.extractRelationalPredicates(doc, src, plan, be.Right)
		return
	}

	// Leaf predicate: Identifier op Literal
	if be.Left.Kind == parser.NodeKindIdentifier {
		id := &doc.Identifiers[be.Left.ID]
		if id.ResolvedKind == parser.ResolvedKindColumn {
			// Only override kind if not already set to Graph (FROM GRAPH_TABLE wins)
			// Hybrid: if vector search is already active, stay KNN (predicates applied as post-filter)
		if plan.Kind == QueryKindKNN && !plan.HasVectorSearch {
				plan.Kind = QueryKindRelational
			}
			plan.HasRelationalQuery = true
			pred := RelationalPredicate{
				Column:   string(src[id.Start:id.End]),
				Operator: be.Operator,
			}
			if be.Right.Kind == parser.NodeKindNumber {
				num := &doc.Numbers[be.Right.ID]
				pred.Value = src[num.Start:num.End]
			} else if be.Right.Kind == parser.NodeKindString {
				sl := &doc.Strings[be.Right.ID]
				pred.Value = src[sl.Start+1 : sl.End-1] // strip quotes
			}
			plan.Predicates = append(plan.Predicates, pred)
		}
	}
}

// isPgVectorOp returns true if the operator is a pgvector distance operator.
func isPgVectorOp(op uint8) bool {
	return op == 22 || op == 23 || op == 24 // KindL2Dist, KindIPDist, KindCosineDist
}

// parseVectorLiteral extracts a []float32 from a string literal in the AST.
func parseVectorLiteral(doc *parser.QueryDoc, src []byte, stringID int32) []float32 {
	sl := &doc.Strings[stringID]
	val := string(src[sl.Start+1 : sl.End-1]) // strip quotes
	if len(val) >= 2 && val[0] == '[' && val[len(val)-1] == ']' {
		val = val[1 : len(val)-1]
	}
	var floats []float32
	start := 0
	for i := 0; i <= len(val); i++ {
		if i == len(val) || val[i] == ',' {
			part := val[start:i]
			for len(part) > 0 && part[0] == ' ' {
				part = part[1:]
			}
			for len(part) > 0 && part[len(part)-1] == ' ' {
				part = part[:len(part)-1]
			}
			if len(part) > 0 {
				var f float32
				fmt.Sscanf(part, "%f", &f)
				floats = append(floats, f)
			}
			start = i + 1
		}
	}
	return floats
}

// EstimateMaxResidualBound calculates the maxResidualBound algebraically from the quantization step.
// For example, if using Scalar Quantization, the max distance per dimension is the step size.
// The max euclidean distance over `dim` dimensions is step * sqrt(dim).
func EstimateMaxResidualBound(qstep float32, dim int) float32 {
	return qstep * float32(math.Sqrt(float64(dim)))
}
