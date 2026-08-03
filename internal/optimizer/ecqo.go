package optimizer

import (
	"errors"
	"fmt"
	"math"

	"github.com/xDarkicex/lexer/parser"
	"github.com/xDarkicex/libravdb/internal/catalog"
	"github.com/xDarkicex/libravdb/internal/graph"
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
	QueryKindInsert                      // INSERT INTO
	QueryKindUpdate                      // UPDATE ... SET ... WHERE
	QueryKindDelete                      // DELETE FROM ... WHERE
	QueryKindJoin                        // SELECT ... JOIN ... ON
	QueryKindAggregate                    // SELECT COUNT/SUM/AVG/MIN/MAX ... GROUP BY/HAVING
	QueryKindDDL                         // CREATE TABLE, DROP TABLE, CREATE INDEX
	QueryKindVectorProjection            // SELECT with SIMILARITY()/VECTOR_DISTANCE() projections (full vector scan)
)

// RelationalPredicate is a single WHERE clause predicate extracted for relational execution.
type RelationalPredicate struct {
	Column   string // column name resolved from source bytes
	Operator uint8  // lexer.KindEquals, KindGreaterThan, KindLessThan, etc.
	Value    []byte // literal value from source bytes (number string or unquoted string)
}

// JoinPlan represents a single JOIN clause.
type JoinPlan struct {
	CollectionName string
	OnPredicates   []RelationalPredicate
	JoinType       uint8 // parser.JoinType value
}

// GraphJoinPlan represents a single JOIN MATCH graph join:
//   FROM services s JOIN MATCH (s)-[:DEPENDS_ON*1..3]->(api:Endpoint)
// The LeftAlias anchors the traversal: each row of the left collection is
// resolved to a graph node, and BFS is seeded from that node over GraphEdges.
type GraphJoinPlan struct {
	LeftAlias      string          // FROM alias anchoring the path (e.g. "s")
	LeftCollection string          // left (FROM) collection name
	GraphEdges     []GraphEdgePlan // edges extracted from the match path
	MaxHops        int             // sum of QuantMax across edges
	JoinType       uint8           // parser.JoinType value
}

// VectorFuncProjection is a SIMILARITY()/VECTOR_DISTANCE() entry in the
// SELECT list. Name is the projected column name (alias if given, else the
// function name); IsDistance distinguishes VECTOR_DISTANCE from SIMILARITY;
// QueryVector is the resolved query-side operand. The stored per-record
// vector comes from the collection at execution time.
type VectorFuncProjection struct {
	Name        string
	IsDistance  bool
	QueryVector []float32
}

// GraphEdgePlan is a single edge extracted from the MATCH path,
// carrying its direction, quantifier bounds, and optional type/kind for traversal.
type GraphEdgePlan struct {
	Direction int8   // -1=inbound, 0=undirected, 1=outbound (from parser.Edge.Direction)
	QuantMin  uint16 // minimum hops (0 for ->*)
	QuantMax  uint16 // maximum hops (0=default→1, QuantUnbounded for ->+/->*)
	EdgeType  string // edge type name from source (e.g., "KNOWS"); empty if not specified
	EdgeKind  uint8  // resolved kind number from registry; 0 if not specified/registered
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
	SeedLabel         string // vertex label for label-scan seeding (e.g., "Person")

	// Relational query fields — populated when Kind == QueryKindRelational
	HasRelationalQuery bool
	Predicates         []RelationalPredicate
	Projections        []string // SELECT column list (empty = all columns)
	OrderBy            string   // column name for ORDER BY (empty = none)
	IsDesc             bool     // ORDER BY DESC

	// Vector function projections — populated when a SELECT list contains
	// SIMILARITY(...) or VECTOR_DISTANCE(...). Each entry pairs the projected
	// column name with its resolved query vector. The projection list itself
	// lives in Projections; this slice carries the vector-func payload.
	VectorFuncProjections []VectorFuncProjection

	// JOIN fields — populated when Kind == QueryKindJoin
	Joins []JoinPlan

	// Graph JOIN fields — populated when a JOIN MATCH clause is present.
	// Left collection is CollectionName; each entry adds a graph traversal
	// seeded from every row of the left collection.
	GraphJoins []GraphJoinPlan

	// CRUD fields — populated for INSERT/UPDATE/DELETE
	InsertColumns []string // column names
	InsertValues  [][]byte // raw value bytes per column, flattened across rows
	SetColumns    []string // SET column names for UPDATE
	SetValues     [][]byte // SET value bytes for UPDATE

	// Aggregate fields — populated when Kind == QueryKindAggregate
	AggregateFunc     uint8    // parser.AggregateFunc value (AggCount, AggSum, etc.)
	AggregateColumn   string   // column name for the aggregate (empty for COUNT(*))
	AggregateDistinct bool     // DISTINCT modifier
	GroupByColumns    []string // GROUP BY column names
	HavingExpr        string   // HAVING expression column name
	HavingOp          uint8    // HAVING operator (e.g., KindGreaterThan)
	HavingValue       []byte   // HAVING literal value

	// DDL fields — populated when Kind == QueryKindDDL
	DDLKind       uint8  // 0=create table, 1=drop table, 2=create index, 3=drop index, 4=alter table
	DDLTableName  string
	DDLIndexName  string
	DDLColumns    []struct{ Name, Type string } // CREATE TABLE columns
	DDLColName    string                         // CREATE INDEX column
	DDLIfExists   bool                           // IF EXISTS modifier
	DDLUnique     bool                           // UNIQUE INDEX modifier

	// Recall contract for hybrid vector queries. Default is RecallExact.
	RecallContract uint8 // 0=Exact, 1=Bounded, 2=BestEffort
}

// RecallContract values for PhysicalPlan.RecallContract.
const (
	RecallExact      uint8 = 0 // complete candidate generation + exact scoring
	RecallBounded    uint8 = 1 // target recall with confidence/SLA
	RecallBestEffort uint8 = 2 // time-budgeted ANN, possible shortfall
)

// Optimizer is the Exact Cardinality Quantized Optimizer (ECQO).
type Optimizer struct {
	catalog *catalog.Catalog
}

func NewOptimizer(cat *catalog.Catalog) *Optimizer {
	return &Optimizer{catalog: cat}
}

// Optimize maps a bound AST to a PhysicalPlan.
func (o *Optimizer) Optimize(doc *parser.QueryDoc, src []byte) (*PhysicalPlan, error) {
	// DDL statements — dispatched directly
	if len(doc.CreateTableStmts) > 0 {
		return o.optimizeCreateTable(doc, src)
	}
	if len(doc.DropTableStmts) > 0 {
		return o.optimizeDropTable(doc, src)
	}
	if len(doc.CreateIndexStmts) > 0 {
		return o.optimizeCreateIndex(doc, src)
	}
	if len(doc.DropIndexStmts) > 0 {
		return o.optimizeDropIndex(doc, src)
	}
	if len(doc.AlterTableStmts) > 0 {
		return o.optimizeAlterTable(doc, src)
	}

	// CRUD statements — dispatched directly, no WHERE/FROM processing needed
	if len(doc.InsertStmts) > 0 {
		return o.optimizeInsert(doc, src)
	}
	if len(doc.UpdateStmts) > 0 {
		return o.optimizeUpdate(doc, src)
	}
	if len(doc.DeleteStmts) > 0 {
		return o.optimizeDelete(doc, src)
	}

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
			// Extract seed label from the first vertex in the match path.
			if mp.PathNodesCount > 0 {
				firstRef := doc.Nodes[mp.PathNodesStart]
				if firstRef.Kind == parser.NodeKindVertex {
					v := &doc.Vertexes[firstRef.ID]
					if v.LabelStart != v.LabelEnd {
						plan.SeedLabel = string(src[v.LabelStart:v.LabelEnd])
					}
				}
			}
			plan.GraphEdges, plan.MaxHops = o.extractMatchPath(doc, src, mp)
		}
	} else {
		return nil, fmt.Errorf("unsupported FROM clause kind")
	}

	// 2. Map WHERE clause (Vector Search + Exact Filters)
	if stmt.WhereExpr.Kind == parser.NodeKindUnknown {
		// Full-scan: FROM table with no WHERE → cursor iteration.
		// Vector-projection queries are classified after projection extraction.
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

	// 3. Map JOIN clauses
	for _, jc := range stmt.Joins {
		// Graph join: JOIN MATCH (s)-[e]->(b). The match path defines the
		// right side; the left side is the FROM collection (plan.CollectionName).
		if jc.MatchPath.Kind == parser.NodeKindMatchPath {
			mp := &doc.MatchPaths[jc.MatchPath.ID]
			gjp := GraphJoinPlan{
				LeftAlias:      plan.CollectionName,
				LeftCollection: plan.CollectionName,
				JoinType:       uint8(jc.Type),
			}
			// Anchor alias = first vertex alias in the path (e.g. "s" in
			// (s)-[:DEPENDS_ON*1..3]->(api)). The binder verified it matches a
			// FROM alias; at plan time the anchor is the left collection itself.
			if mp.PathNodesCount > 0 {
				firstRef := doc.Nodes[mp.PathNodesStart]
				if firstRef.Kind == parser.NodeKindVertex {
					v := &doc.Vertexes[firstRef.ID]
					if v.AliasEnd > v.Alias {
						gjp.LeftAlias = string(src[v.Alias:v.AliasEnd])
					}
				}
			}
			gjp.GraphEdges, gjp.MaxHops = o.extractMatchPath(doc, src, mp)
			plan.GraphJoins = append(plan.GraphJoins, gjp)
			plan.Kind = QueryKindJoin
			continue
		}

		jp := JoinPlan{
			CollectionName: string(src[jc.TableStart:jc.TableEnd]),
			JoinType:       uint8(jc.Type),
		}
		if jc.OnExpr.Kind != parser.NodeKindUnknown {
			// Extract ON predicates into a temporary plan
			tmp := &PhysicalPlan{Predicates: nil}
			o.extractRelationalPredicates(doc, src, tmp, jc.OnExpr)
			jp.OnPredicates = tmp.Predicates
		}
		plan.Joins = append(plan.Joins, jp)
		plan.Kind = QueryKindJoin
	}

	// 4. Map ORDER BY (now step 4 after JOIN)
	if stmt.OrderBy.Kind == parser.NodeKindIdentifier {
		id := &doc.Identifiers[stmt.OrderBy.ID]
		plan.OrderBy = string(src[id.Start:id.End])
		plan.IsDesc = stmt.IsDesc
	}

	// 4. Extract projection columns and detect aggregates
	hasAggregate := len(stmt.GroupBy) > 0 || stmt.HavingExpr.Kind != parser.NodeKindUnknown
	if stmt.ProjectionsCount > 0 {
		for i := int32(0); i < stmt.ProjectionsCount; i++ {
			proj := &doc.Projections[stmt.ProjectionsStart+i]
			if proj.Expr.Kind == parser.NodeKindIdentifier {
				id := &doc.Identifiers[proj.Expr.ID]
				plan.Projections = append(plan.Projections, string(src[id.Start:id.End]))
			} else if proj.Expr.Kind == parser.NodeKindAggregateExpr {
				hasAggregate = true
				ae := &doc.AggregateExprs[proj.Expr.ID]
				plan.AggregateFunc = uint8(ae.Func)
				plan.AggregateDistinct = ae.Distinct
				if ae.Expr.Kind == parser.NodeKindIdentifier {
					id := &doc.Identifiers[ae.Expr.ID]
					plan.AggregateColumn = string(src[id.Start:id.End])
				}
			} else if proj.Expr.Kind == parser.NodeKindVectorFunc {
				// SIMILARITY(col, vec) / VECTOR_DISTANCE(col, vec) in the SELECT list.
				vf := &doc.VectorFuncs[proj.Expr.ID]
				vfp := VectorFuncProjection{IsDistance: !vf.IsMaxSim}
				// Column name from the alias if present, else the function name.
				if proj.AliasEnd > proj.Alias {
					vfp.Name = string(src[proj.Alias:proj.AliasEnd])
				} else if vf.IsMaxSim {
					vfp.Name = "similarity"
				} else {
					vfp.Name = "vector_distance"
				}
				// Resolve the query vector from the second operand (string literal).
				if vf.VectorB.Kind == parser.NodeKindString {
					sl := &doc.Strings[vf.VectorB.ID]
					vfp.QueryVector = parseVectorLiteral(doc, src, sl.ID)
				}
				plan.VectorFuncProjections = append(plan.VectorFuncProjections, vfp)
				plan.Projections = append(plan.Projections, vfp.Name)
			}
		}
	}

	// Vector-func projections without a WHERE vector predicate are a full
	// vector projection scan, not a relational scan. This must be decided
	// AFTER projection extraction, since the kind was set at WHERE-mapping time.
	if len(plan.VectorFuncProjections) > 0 && plan.Kind == QueryKindRelational {
		plan.Kind = QueryKindVectorProjection
		plan.HasRelationalQuery = false
	}

	// GROUP BY columns
	for _, gb := range stmt.GroupBy {
		if gb.Kind == parser.NodeKindIdentifier {
			id := &doc.Identifiers[gb.ID]
			plan.GroupByColumns = append(plan.GroupByColumns, string(src[id.Start:id.End]))
		}
	}

	// HAVING clause
	if stmt.HavingExpr.Kind == parser.NodeKindBinaryExpr {
		be := &doc.BinaryExprs[stmt.HavingExpr.ID]
		if be.Left.Kind == parser.NodeKindIdentifier {
			id := &doc.Identifiers[be.Left.ID]
			plan.HavingExpr = string(src[id.Start:id.End])
			plan.HavingOp = be.Operator
			if be.Right.Kind == parser.NodeKindNumber {
				plan.HavingValue = src[doc.Numbers[be.Right.ID].Start:doc.Numbers[be.Right.ID].End]
			} else if be.Right.Kind == parser.NodeKindString {
				sl := &doc.Strings[be.Right.ID]
				plan.HavingValue = src[sl.Start+1 : sl.End-1]
			}
		}
	}

	if hasAggregate {
		plan.Kind = QueryKindAggregate
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
// extractMatchPath converts a MatchPath AST node into GraphEdgePlans and the
// cumulative MaxHops bound. Shared by GRAPH_TABLE FROM clauses and JOIN MATCH.
func (o *Optimizer) extractMatchPath(doc *parser.QueryDoc, src []byte, mp *parser.MatchPath) ([]GraphEdgePlan, int) {
	var edges []GraphEdgePlan
	maxHops := 0
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
		if e.TypeStart != e.TypeEnd {
			gep.EdgeType = string(src[e.TypeStart:e.TypeEnd])
			gep.EdgeKind = graph.ResolveEdgeKind(gep.EdgeType)
		}
		edges = append(edges, gep)

		// Compute MaxHops: sum QuantMax across all edges
		max := int(e.QuantMax)
		if e.QuantMax == 0 {
			if e.QuantMin == 0 {
				max = 1 // default: exactly 1 hop
			} else {
				max = 1 << 20 // ->+ : unbounded
			}
		}
		maxHops += max
	}
	if maxHops == 0 {
		maxHops = 1
	}
	return edges, maxHops
}

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

func (o *Optimizer) optimizeInsert(doc *parser.QueryDoc, src []byte) (*PhysicalPlan, error) {
	stmt := &doc.InsertStmts[0]
	plan := &PhysicalPlan{
		Kind:           QueryKindInsert,
		CollectionName: string(src[stmt.TableStart:stmt.TableEnd]),
	}
	for _, col := range stmt.Columns {
		id := &doc.Identifiers[col.ID]
		plan.InsertColumns = append(plan.InsertColumns, string(src[id.Start:id.End]))
	}
	for _, val := range stmt.Values {
		switch val.Kind {
		case parser.NodeKindString:
			sl := &doc.Strings[val.ID]
			plan.InsertValues = append(plan.InsertValues, src[sl.Start+1:sl.End-1])
		case parser.NodeKindNumber:
			num := &doc.Numbers[val.ID]
			plan.InsertValues = append(plan.InsertValues, src[num.Start:num.End])
		}
	}
	return plan, nil
}

func (o *Optimizer) optimizeUpdate(doc *parser.QueryDoc, src []byte) (*PhysicalPlan, error) {
	stmt := &doc.UpdateStmts[0]
	plan := &PhysicalPlan{
		Kind:           QueryKindUpdate,
		CollectionName: string(src[stmt.TableStart:stmt.TableEnd]),
	}
	for _, col := range stmt.SetColumns {
		id := &doc.Identifiers[col.ID]
		plan.SetColumns = append(plan.SetColumns, string(src[id.Start:id.End]))
	}
	for _, val := range stmt.SetValues {
		switch val.Kind {
		case parser.NodeKindString:
			sl := &doc.Strings[val.ID]
			plan.SetValues = append(plan.SetValues, src[sl.Start+1:sl.End-1])
		case parser.NodeKindNumber:
			num := &doc.Numbers[val.ID]
			plan.SetValues = append(plan.SetValues, src[num.Start:num.End])
		}
	}
	// Extract WHERE predicates for ID resolution
	if stmt.WhereExpr.Kind != parser.NodeKindUnknown {
		o.extractRelationalPredicates(doc, src, plan, stmt.WhereExpr)
	}
	return plan, nil
}

func (o *Optimizer) optimizeDelete(doc *parser.QueryDoc, src []byte) (*PhysicalPlan, error) {
	stmt := &doc.DeleteStmts[0]
	plan := &PhysicalPlan{
		Kind:           QueryKindDelete,
		CollectionName: string(src[stmt.TableStart:stmt.TableEnd]),
	}
	if stmt.WhereExpr.Kind != parser.NodeKindUnknown {
		o.extractRelationalPredicates(doc, src, plan, stmt.WhereExpr)
	}
	return plan, nil
}

func (o *Optimizer) optimizeCreateTable(doc *parser.QueryDoc, src []byte) (*PhysicalPlan, error) {
	stmt := &doc.CreateTableStmts[0]
	plan := &PhysicalPlan{
		Kind:         QueryKindDDL,
		DDLKind:      0,
		DDLTableName: string(src[stmt.TableStart:stmt.TableEnd]),
		CollectionName: string(src[stmt.TableStart:stmt.TableEnd]),
	}
	for _, col := range stmt.Columns {
		plan.DDLColumns = append(plan.DDLColumns, struct{ Name, Type string }{
			Name: string(src[col.NameStart:col.NameEnd]),
			Type: string(src[col.TypeStart:col.TypeEnd]),
		})
	}
	return plan, nil
}

func (o *Optimizer) optimizeDropTable(doc *parser.QueryDoc, src []byte) (*PhysicalPlan, error) {
	stmt := &doc.DropTableStmts[0]
	return &PhysicalPlan{
		Kind:         QueryKindDDL,
		DDLKind:      1,
		DDLTableName: string(src[stmt.TableStart:stmt.TableEnd]),
		CollectionName: string(src[stmt.TableStart:stmt.TableEnd]),
	}, nil
}

func (o *Optimizer) optimizeCreateIndex(doc *parser.QueryDoc, src []byte) (*PhysicalPlan, error) {
	stmt := &doc.CreateIndexStmts[0]
	return &PhysicalPlan{
		Kind:         QueryKindDDL,
		DDLKind:      2,
		DDLTableName: string(src[stmt.TableStart:stmt.TableEnd]),
		DDLIndexName: string(src[stmt.IndexStart:stmt.IndexEnd]),
		DDLColName:   string(src[stmt.ColStart:stmt.ColEnd]),
		DDLUnique:    stmt.Unique,
		CollectionName: string(src[stmt.TableStart:stmt.TableEnd]),
	}, nil
}

func (o *Optimizer) optimizeDropIndex(doc *parser.QueryDoc, src []byte) (*PhysicalPlan, error) {
	stmt := &doc.DropIndexStmts[0]
	return &PhysicalPlan{
		Kind:         QueryKindDDL,
		DDLKind:      3,
		DDLIndexName: string(src[stmt.IndexStart:stmt.IndexEnd]),
		DDLIfExists:  stmt.IfExists,
	}, nil
}

func (o *Optimizer) optimizeAlterTable(doc *parser.QueryDoc, src []byte) (*PhysicalPlan, error) {
	stmt := &doc.AlterTableStmts[0]
	return &PhysicalPlan{
		Kind:         QueryKindDDL,
		DDLKind:      4,
		DDLTableName: string(src[stmt.TableStart:stmt.TableEnd]),
		DDLColumns: []struct{ Name, Type string }{{
			Name: string(src[stmt.AddColumn.NameStart:stmt.AddColumn.NameEnd]),
			Type: string(src[stmt.AddColumn.TypeStart:stmt.AddColumn.TypeEnd]),
		}},
		CollectionName: string(src[stmt.TableStart:stmt.TableEnd]),
	}, nil
}

// EstimateMaxResidualBound calculates the maxResidualBound algebraically from the quantization step.
// For example, if using Scalar Quantization, the max distance per dimension is the step size.
// The max euclidean distance over `dim` dimensions is step * sqrt(dim).
func EstimateMaxResidualBound(qstep float32, dim int) float32 {
	return qstep * float32(math.Sqrt(float64(dim)))
}
