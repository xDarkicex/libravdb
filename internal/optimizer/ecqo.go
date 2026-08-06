package optimizer

import (
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/xDarkicex/lexer"
	"github.com/xDarkicex/lexer/parser"
	"github.com/xDarkicex/libravdb/internal/catalog"
	"github.com/xDarkicex/libravdb/internal/graph"
)

var (
	ErrUnsupportedQuery = errors.New("unsupported query shape")
)

// resolveVectorParam checks if an identifier starts with '$' or '@' and
// resolves it from the query params map. The returned slice is copied before it enters
// a physical plan, so callers cannot mutate an in-flight query by retaining a
// parameter slice.
func resolveVectorParam(src []byte, id *parser.Identifier, params map[string]interface{}) ([]float32, bool) {
	if len(src) > int(id.Start) && (src[id.Start] == '$' || src[id.Start] == '@') {
		name := string(src[id.Start+1 : id.End])
		if params != nil {
			if val, ok := params[name]; ok {
				if vec, ok := val.([]float32); ok {
					return append([]float32(nil), vec...), true
				}
			}
		}
	}
	return nil, false
}

func resolveVectorOperand(doc *parser.QueryDoc, src []byte, ref parser.NodeRef, params map[string]interface{}) ([]float32, error) {
	switch ref.Kind {
	case parser.NodeKindString:
		return parseVectorLiteral(doc, src, ref.ID), nil
	case parser.NodeKindIdentifier:
		id := &doc.Identifiers[ref.ID]
		if vec, ok := resolveVectorParam(src, id, params); ok {
			return vec, nil
		}
		if int(id.Start) < len(src) && (src[id.Start] == '$' || src[id.Start] == '@') {
			return nil, fmt.Errorf("vector parameter %q is missing or is not []float32", string(src[id.Start:id.End]))
		}
	}
	return nil, fmt.Errorf("vector query operand must be a vector literal or named vector parameter")
}

// QueryKind classifies the query type for executor dispatch.
type QueryKind uint8

const (
	QueryKindKNN              QueryKind = iota // vector similarity search (default)
	QueryKindGraph                             // graph pattern matching via GRAPH_TABLE
	QueryKindRelational                        // relational exact-match / range scan
	QueryKindInsert                            // INSERT INTO
	QueryKindUpdate                            // UPDATE ... SET ... WHERE
	QueryKindDelete                            // DELETE FROM ... WHERE
	QueryKindJoin                              // SELECT ... JOIN ... ON
	QueryKindAggregate                         // SELECT COUNT/SUM/AVG/MIN/MAX ... GROUP BY/HAVING
	QueryKindDDL                               // CREATE TABLE, DROP TABLE, CREATE INDEX
	QueryKindVectorProjection                  // SELECT with SIMILARITY()/VECTOR_DISTANCE() projections (full vector scan)
	QueryKindMultiModal                        // relational JOIN + JOIN MATCH + vector top-k
	QueryKindInsertGraphEdge                   // INSERT INTO GRAPH_EDGES VALUES (src, kind, tgt)
)

// RelationalPredicate is a single WHERE clause predicate extracted for relational execution.
type RelationalPredicate struct {
	Alias    string // source alias for qualified predicates; empty means unqualified
	Column   string // column name resolved from source bytes
	Operator uint8  // lexer.KindEquals, KindGreaterThan, KindLessThan, etc.
	Value    []byte // literal value from source bytes (number string or unquoted string)
}

// JoinPlan represents a single JOIN clause.
type JoinPlan struct {
	CollectionName  string
	LeftAlias       string
	RightAlias      string
	LeftColumn      string                // equality join key from the accumulated left relation
	RightColumn     string                // equality join key from CollectionName
	RightPredicates []RelationalPredicate // literal predicates in ON that apply to the right relation
	OnPredicates    []RelationalPredicate
	JoinType        uint8 // parser.JoinType value
}

// GraphJoinPlan represents a single JOIN MATCH graph join:
//
//	FROM services s JOIN MATCH (s)-[:DEPENDS_ON*1..3]->(api:Endpoint)
//
// The LeftAlias anchors the traversal: each row of the left collection is
// resolved to a graph node, and BFS is seeded from that node over GraphEdges.
type GraphJoinPlan struct {
	LeftAlias          string                // FROM alias anchoring the path (e.g. "s")
	LeftCollection     string                // left (FROM) collection name
	GraphEdges         []GraphEdgePlan       // edges extracted from the match path
	MaxHops            int                   // sum of QuantMax across edges
	JoinType           uint8                 // parser.JoinType value
	TerminalAlias      string                // final vertex alias, if present
	TerminalLabel      string                // final vertex label, if present
	TerminalPredicates []RelationalPredicate // WHERE predicates bound to final vertex
	// PredicateMatch is true when the graph path came from WHERE MATCH
	// (existential filter). Traversal emits source anchor IDs, not
	// terminal vertices. False for JOIN MATCH / GRAPH_TABLE.
	PredicateMatch bool
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
	SourceAlias string // qualified vector source, e.g. c in c.embedding
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

// DDLForeignKey is a parsed foreign key constraint lowered into the DDL plan.
// SourceColumns and TargetColumns are paired element-wise: column i in
// SourceColumns references column i in TargetColumns.
type DDLForeignKey struct {
	Name          string   // constraint name (empty if unnamed)
	SourceColumns []string // source (child) column names
	TargetTable   string   // target (parent) table name
	TargetColumns []string // target column names (paired with SourceColumns)
	OnDelete      uint8    // OnDeleteAction value
	OnUpdate      uint8    // OnDeleteAction value (reused enum)
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

	// SnapshotLSN carries a resolved temporal snapshot LSN for historical
	// queries. When non-zero, execution must use temporal read APIs.
	SnapshotLSN uint64
	// SnapshotTimestamp is the parsed AS OF TIMESTAMP value (UTC), resolved
	// to an LSN at execution time.
	SnapshotTimestamp time.Time

	// Relational query fields — populated when Kind == QueryKindRelational
	HasRelationalQuery bool
	Predicates         []RelationalPredicate
	Projections        []string // SELECT column list (empty = all columns)
	// Scoring expression — parser AST lowered to execution hints.
	HasScoreExpr       bool
	ScoreArithOp       uint8 // KindAsterisk(11), KindDash(18)
	HasGraphCentrality bool
	ScoreLiteralValue  float64

	OrderBy string // column name for ORDER BY (empty = none)
	IsDesc  bool   // ORDER BY DESC

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
	DDLKind      uint8 // 0=create table, 1=drop table, 2=create index, 3=drop index, 4=alter table
	DDLTableName string
	DDLIndexName string
	DDLColumns   []struct {
		Name            string
		Type            string
		VectorDimension uint32 // parsed dimension from VECTOR(n); 0 for non-vector columns
		Flags           uint16 // ColFlagNotNull, ColFlagPrimaryKey from parser
	} // CREATE TABLE columns
	DDLColName  string // CREATE INDEX column
	DDLIfExists bool   // IF EXISTS modifier
	DDLUnique   bool   // UNIQUE INDEX modifier

	// DDLForeignKeys carries parsed FK constraints for CREATE TABLE.
	DDLForeignKeys []DDLForeignKey
	// DDLPrimaryKeyColumns preserves the ordered table-level PRIMARY KEY
	// columns. A nil slice means no table-level composite key was declared.
	DDLPrimaryKeyColumns []string

	// DDLExternalKey is set when a CREATE TABLE FK references GRAPH_NODES,
	// signalling the optimizer that graph-aware join planning is applicable.
	DDLExternalKey bool

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
	src     []byte                 // set during OptimizeWithParams
	params  map[string]interface{} // set during OptimizeWithParams
}

func NewOptimizer(cat *catalog.Catalog) *Optimizer {
	return &Optimizer{catalog: cat}
}

// Optimize maps a bound AST to a PhysicalPlan.
func (o *Optimizer) Optimize(doc *parser.QueryDoc, src []byte) (*PhysicalPlan, error) {
	return o.OptimizeWithParams(doc, src, nil)
}

func (o *Optimizer) OptimizeWithParams(doc *parser.QueryDoc, src []byte, params map[string]interface{}) (*PhysicalPlan, error) {
	o.src = src
	o.params = params
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
	if len(doc.InsertGraphEdgeStmts) > 0 {
		return o.optimizeInsertGraphEdge(doc, src)
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
		if t.Temporal {
			ts := string(src[t.TimestampStart:t.TimestampEnd])
			if len(ts) >= 2 && ts[0] == '\'' && ts[len(ts)-1] == '\'' {
				ts = ts[1 : len(ts)-1]
			}
			// Accept RFC3339 or space-separated (treated as UTC).
			parsed, err := time.Parse(time.RFC3339, ts)
			if err != nil {
				// Try space-separated: "2026-07-01 14:00:00" → UTC
				parsed, err = time.Parse("2006-01-02 15:04:05", ts)
				if err != nil {
					return nil, fmt.Errorf("invalid AS OF TIMESTAMP %q: %w", ts, err)
				}
				parsed = parsed.UTC()
			}
			plan.SnapshotTimestamp = parsed.UTC()
		}
	} else if stmt.FromTable.Kind == parser.NodeKindGraphTable {
		gt := &doc.GraphTables[stmt.FromTable.ID]
		plan.CollectionOID = gt.TableOID
		plan.CollectionName = string(src[gt.TableStart:gt.TableEnd])
		// Standalone SELECT * FROM MATCH (no GRAPH_TABLE wrapper) produces
		// an empty collection name. The executor resolves it to the first
		// graph-enabled collection at execution time.
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

		// Walk WHERE tree for relational predicates.
		o.extractRelationalPredicates(doc, src, plan, whereNode)

		// WHERE MATCH: walk AND chains to find the MatchPath.
		o.extractWhereMatch(doc, src, plan, whereNode)
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
							if string(v) < string(minV) {
								minV = v
							}
							if string(v) > string(maxV) {
								maxV = v
							}
						}
						plan.Predicates = append(plan.Predicates,
							RelationalPredicate{Column: col, Operator: 13, Value: minV}, // >= min
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

					vec, err := resolveVectorOperand(doc, src, vf.VectorB, params)
					if err != nil {
						return nil, err
					}
					plan.QueryVector = vec
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

		leftAlias := plan.CollectionName
		if stmt.FromTable.Kind == parser.NodeKindTableExpr {
			from := &doc.TableExprs[stmt.FromTable.ID]
			if from.AliasEnd > from.Alias {
				leftAlias = string(src[from.Alias:from.AliasEnd])
			}
		}
		jp := JoinPlan{
			CollectionName: string(src[jc.TableStart:jc.TableEnd]),
			LeftAlias:      leftAlias,
			JoinType:       uint8(jc.Type),
		}
		if jc.AliasEnd > jc.Alias {
			jp.RightAlias = string(src[jc.Alias:jc.AliasEnd])
		} else {
			jp.RightAlias = jp.CollectionName
		}
		if jc.OnExpr.Kind != parser.NodeKindUnknown {
			o.extractJoinConditions(doc, src, &jp, jc.OnExpr)
		}
		plan.Joins = append(plan.Joins, jp)
		plan.Kind = QueryKindJoin
	}

	// 4. Map ORDER BY (now step 4 after JOIN)
	if stmt.OrderBy.Kind == parser.NodeKindVectorFunc {
		vf := &doc.VectorFuncs[stmt.OrderBy.ID]
		vec, err := resolveVectorOperand(doc, src, vf.VectorB, o.params)
		if err != nil {
			return nil, err
		}
		plan.QueryVector = vec
		plan.HasVectorSearch = true
		plan.OrderBy = "vector_distance"
		plan.IsDesc = stmt.IsDesc
	} else if stmt.OrderBy.Kind == parser.NodeKindIdentifier {
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
				if vf.VectorA.Kind == parser.NodeKindIdentifier {
					id := &doc.Identifiers[vf.VectorA.ID]
					if id.QualEnd > id.QualStart {
						vfp.SourceAlias = string(src[id.QualStart:id.QualEnd])
					}
				}
				// Column name from the alias if present, else the function name.
				if proj.AliasEnd > proj.Alias {
					vfp.Name = string(src[proj.Alias:proj.AliasEnd])
				} else if vf.IsMaxSim {
					vfp.Name = "similarity"
				} else {
					vfp.Name = "vector_distance"
				}
				vec, err := resolveVectorOperand(doc, src, vf.VectorB, params)
				if err != nil {
					return nil, err
				}
				vfp.QueryVector = vec
				plan.VectorFuncProjections = append(plan.VectorFuncProjections, vfp)
				plan.Projections = append(plan.Projections, vfp.Name)
			}
		}
	}

	// Lower scoring expression from projection tree.
	o.lowerScoreExprs(doc, stmt, plan)

	// Vector-func projections without a WHERE vector predicate are a full
	// vector projection scan, not a relational scan. This must be decided
	// AFTER projection extraction, since the kind was set at WHERE-mapping time.
	if len(plan.VectorFuncProjections) > 0 && plan.Kind == QueryKindRelational {
		plan.Kind = QueryKindVectorProjection
		plan.HasRelationalQuery = false
	}
	// A projected vector distance ordered with LIMIT is still a vector top-k
	// operation.  When relational JOIN and JOIN MATCH are present together,
	// retain every clause in one physical plan rather than allowing one legacy
	// executor path to discard the others.
	// Classify as multimodal when graph traversal (JOIN MATCH or WHERE MATCH)
	// and vector projection are both present. Relational JOIN is optional.
	hasGraph := len(plan.GraphJoins) > 0 || plan.HasGraphTraversal
	hasVector := len(plan.VectorFuncProjections) > 0 || plan.HasScoreExpr
	if hasGraph && hasVector {
		plan.Kind = QueryKindMultiModal
		plan.HasVectorSearch = true
		if len(plan.VectorFuncProjections) > 0 {
			plan.QueryVector = append([]float32(nil), plan.VectorFuncProjections[0].QueryVector...)
		}
	} else if plan.HasGraphTraversal {
		// WHERE MATCH without a vector projection is still a graph query. Keep
		// the lowered graph join as the source of traversal semantics so the
		// temporal executor can apply the same path, terminal-label, and
		// terminal-predicate rules as the multimodal route.
		plan.Kind = QueryKindGraph
		plan.HasGraphTraversal = true
		if len(plan.GraphEdges) == 0 && len(plan.GraphJoins) > 0 {
			plan.GraphEdges = append([]GraphEdgePlan(nil), plan.GraphJoins[0].GraphEdges...)
			plan.MaxHops = plan.GraphJoins[0].MaxHops
		}
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
			if len(floats) > 0 {
				plan.HasVectorAnchor = true
				plan.GraphAnchorVector = floats
				// A vector function nested inside a graph WHERE tree is both
				// an anchor source and a vector-search component. Marking only
				// the anchor caused vector+graph plans under AND to bypass the
				// hybrid dispatcher entirely.
				plan.HasVectorSearch = true
				plan.QueryVector = floats
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

// lowerScoreExprs walks projection expressions to detect compound score
// expressions like (1.0 - VECTOR_DISTANCE) * GRAPH_CENTRALITY and sets
// plan fields for the executor.
func (o *Optimizer) lowerScoreExprs(doc *parser.QueryDoc, stmt *parser.SelectStmt, plan *PhysicalPlan) {
	for i := int32(0); i < stmt.ProjectionsCount; i++ {
		proj := &doc.Projections[stmt.ProjectionsStart+i]
		if o.lowerScoreExpr(doc, proj.Expr, plan) {
			break
		}
	}
}

func (o *Optimizer) lowerScoreExpr(doc *parser.QueryDoc, node parser.NodeRef, plan *PhysicalPlan) bool {
	if node.Kind == parser.NodeKindBinaryExpr {
		be := &doc.BinaryExprs[node.ID]
		if be.Operator == 11 { // KindAsterisk = multiply
			plan.ScoreArithOp = 11
			plan.HasScoreExpr = true
			// Check left for subtraction, right for centrality.
			o.lowerScoreExpr(doc, be.Left, plan)
			o.lowerScoreExpr(doc, be.Right, plan)
			return true
		}
		if be.Operator == 18 { // KindDash = subtraction
			plan.ScoreArithOp = 18
			// Left operand is the literal (e.g. 1.0).
			if be.Left.Kind == parser.NodeKindNumber {
				num := &doc.Numbers[be.Left.ID]
				plan.ScoreLiteralValue = o.parseNumberLiteral(num)
			}
			o.lowerScoreExpr(doc, be.Right, plan)
			return false
		}
	}
	if node.Kind == parser.NodeKindGraphMetric {
		plan.HasGraphCentrality = true
		plan.HasScoreExpr = true
		return true
	}
	if node.Kind == parser.NodeKindVectorFunc {
		plan.HasScoreExpr = true
		vf := &doc.VectorFuncs[node.ID]
		if vec, err := resolveVectorOperand(doc, o.src, vf.VectorB, o.params); err == nil {
			plan.QueryVector = vec
			plan.HasVectorSearch = true
		}
		return true
	}
	return false
}

func (o *Optimizer) parseNumberLiteral(num *parser.Number) float64 {
	if num == nil || o.src == nil {
		return 1.0
	}
	var f float64
	fmt.Sscanf(string(o.src[num.Start:num.End]), "%f", &f)
	return f
}

// extractWhereMatch walks the WHERE expression tree for a MatchPath node and
// lowers it into the same GraphJoinPlan used by JOIN MATCH. Keeping the graph
// path as a join plan is important: execution must intersect it with vector
// ranking rather than treating MATCH as an advisory predicate.
func (o *Optimizer) extractWhereMatch(doc *parser.QueryDoc, src []byte, plan *PhysicalPlan, node parser.NodeRef) {
	switch node.Kind {
	case parser.NodeKindMatchPath:
		mp := &doc.MatchPaths[node.ID]
		plan.HasGraphTraversal = true
		plan.GraphEdges, plan.MaxHops = o.extractMatchPath(doc, src, mp)
		gjp := GraphJoinPlan{
			LeftAlias:      plan.CollectionName,
			LeftCollection: plan.CollectionName,
			GraphEdges:     plan.GraphEdges,
			MaxHops:        plan.MaxHops,
			PredicateMatch: true, // WHERE MATCH source-row semantics
		}
		var firstAlias string
		for i := int32(0); i < mp.PathNodesCount; i++ {
			ref := doc.Nodes[mp.PathNodesStart+i]
			if ref.Kind != parser.NodeKindVertex {
				continue
			}
			v := &doc.Vertexes[ref.ID]
			if v.AliasEnd > v.Alias {
				alias := string(src[v.Alias:v.AliasEnd])
				if firstAlias == "" {
					firstAlias = alias
					gjp.LeftAlias = alias
				}
				gjp.TerminalAlias = alias
			}
			if v.LabelEnd > v.LabelStart {
				gjp.TerminalLabel = string(src[v.LabelStart:v.LabelEnd])
			}
		}
		for _, predicate := range plan.Predicates {
			if predicate.Alias != "" && predicate.Alias == gjp.TerminalAlias {
				gjp.TerminalPredicates = append(gjp.TerminalPredicates, predicate)
			}
		}
		plan.GraphJoins = append(plan.GraphJoins, gjp)
	case parser.NodeKindBinaryExpr:
		be := &doc.BinaryExprs[node.ID]
		o.extractWhereMatch(doc, src, plan, be.Left)
		o.extractWhereMatch(doc, src, plan, be.Right)
	}
}

func (o *Optimizer) extractRelationalPredicates(doc *parser.QueryDoc, src []byte, plan *PhysicalPlan, node parser.NodeRef) {
	if node.Kind != parser.NodeKindBinaryExpr {
		return
	}
	be := &doc.BinaryExprs[node.ID]

	// AND decomposition: recurse into both sides
	if be.Operator == uint8(lexer.KindAnd) {
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
			if id.QualEnd > id.QualStart {
				pred.Alias = string(src[id.QualStart:id.QualEnd])
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

// extractJoinConditions preserves the two distinct parts of an ON clause:
// column-to-column equality is the join key, while column-to-literal clauses
// are predicates that can be pushed into the appropriate input.  The legacy
// RelationalPredicate representation alone cannot express s.owner_id = t.id.
func (o *Optimizer) extractJoinConditions(doc *parser.QueryDoc, src []byte, join *JoinPlan, node parser.NodeRef) {
	if node.Kind != parser.NodeKindBinaryExpr {
		return
	}
	be := &doc.BinaryExprs[node.ID]
	if be.Operator == uint8(lexer.KindAnd) {
		o.extractJoinConditions(doc, src, join, be.Left)
		o.extractJoinConditions(doc, src, join, be.Right)
		return
	}

	if be.Left.Kind != parser.NodeKindIdentifier {
		return
	}
	left := &doc.Identifiers[be.Left.ID]
	leftColumn := string(src[left.Start:left.End])
	leftQualifier := ""
	if left.QualEnd > left.QualStart {
		leftQualifier = string(src[left.QualStart:left.QualEnd])
	}

	if be.Operator == uint8(lexer.KindEquals) && be.Right.Kind == parser.NodeKindIdentifier {
		right := &doc.Identifiers[be.Right.ID]
		rightColumn := string(src[right.Start:right.End])
		rightQualifier := ""
		if right.QualEnd > right.QualStart {
			rightQualifier = string(src[right.QualStart:right.QualEnd])
		}
		switch {
		case leftQualifier == join.LeftAlias && rightQualifier == join.RightAlias:
			join.LeftColumn, join.RightColumn = leftColumn, rightColumn
		case leftQualifier == join.RightAlias && rightQualifier == join.LeftAlias:
			join.LeftColumn, join.RightColumn = rightColumn, leftColumn
		}
		return
	}

	predicate := RelationalPredicate{Column: leftColumn, Operator: be.Operator}
	switch be.Right.Kind {
	case parser.NodeKindNumber:
		n := &doc.Numbers[be.Right.ID]
		predicate.Value = src[n.Start:n.End]
	case parser.NodeKindString:
		s := &doc.Strings[be.Right.ID]
		predicate.Value = src[s.Start+1 : s.End-1]
	default:
		return
	}
	// The unqualified form is accepted as a right-side filter only when there
	// is no ambiguous left relation. Qualified t.sla_status is the normal form.
	if leftQualifier == join.RightAlias || leftQualifier == "" {
		join.RightPredicates = append(join.RightPredicates, predicate)
		join.OnPredicates = append(join.OnPredicates, predicate)
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

func (o *Optimizer) optimizeInsertGraphEdge(doc *parser.QueryDoc, src []byte) (*PhysicalPlan, error) {
	// Take the first statement; batch support can be added later.
	stmt := &doc.InsertGraphEdgeStmts[0]
	plan := &PhysicalPlan{
		Kind: QueryKindInsertGraphEdge,
	}
	plan.InsertValues = append(plan.InsertValues,
		src[stmt.SrcStart:stmt.SrcEnd],
		src[stmt.EdgeKindStart:stmt.EdgeKindEnd],
		src[stmt.TgtStart:stmt.TgtEnd],
	)
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
		Kind:           QueryKindDDL,
		DDLKind:        0,
		DDLTableName:   string(src[stmt.TableStart:stmt.TableEnd]),
		CollectionName: string(src[stmt.TableStart:stmt.TableEnd]),
	}
	for _, col := range stmt.Columns {
		plan.DDLColumns = append(plan.DDLColumns, struct {
			Name            string
			Type            string
			VectorDimension uint32
			Flags           uint16
		}{
			Name:            string(src[col.NameStart:col.NameEnd]),
			Type:            string(src[col.TypeStart:col.TypeEnd]),
			VectorDimension: col.TypeParam,
			Flags:           col.Flags,
		})
	}
	if stmt.PrimaryKey != nil {
		for _, ref := range stmt.PrimaryKey.Columns {
			plan.DDLPrimaryKeyColumns = append(plan.DDLPrimaryKeyColumns,
				string(src[ref.Start:ref.End]))
		}
	}
	// Lower FK constraints from AST to plan.
	for _, fk := range stmt.ForeignKeys {
		srcCols := make([]string, len(fk.SourceColumns))
		for i, c := range fk.SourceColumns {
			srcCols[i] = string(src[c.Start:c.End])
		}
		tgtCols := make([]string, len(fk.TargetColumns))
		for i, c := range fk.TargetColumns {
			tgtCols[i] = string(src[c.Start:c.End])
		}
		planFK := DDLForeignKey{
			SourceColumns: srcCols,
			TargetTable:   string(src[fk.TgtTableStart:fk.TgtTableEnd]),
			TargetColumns: tgtCols,
			OnDelete:      uint8(fk.OnDelete),
			OnUpdate:      uint8(fk.OnUpdate),
		}
		if fk.NameStart != 0 {
			planFK.Name = string(src[fk.NameStart:fk.NameEnd])
		}
		if planFK.TargetTable == "GRAPH_NODES" {
			plan.DDLExternalKey = true
		}
		plan.DDLForeignKeys = append(plan.DDLForeignKeys, planFK)
	}
	return plan, nil
}

func (o *Optimizer) optimizeDropTable(doc *parser.QueryDoc, src []byte) (*PhysicalPlan, error) {
	stmt := &doc.DropTableStmts[0]
	return &PhysicalPlan{
		Kind:           QueryKindDDL,
		DDLKind:        1,
		DDLTableName:   string(src[stmt.TableStart:stmt.TableEnd]),
		CollectionName: string(src[stmt.TableStart:stmt.TableEnd]),
	}, nil
}

func (o *Optimizer) optimizeCreateIndex(doc *parser.QueryDoc, src []byte) (*PhysicalPlan, error) {
	stmt := &doc.CreateIndexStmts[0]
	return &PhysicalPlan{
		Kind:           QueryKindDDL,
		DDLKind:        2,
		DDLTableName:   string(src[stmt.TableStart:stmt.TableEnd]),
		DDLIndexName:   string(src[stmt.IndexStart:stmt.IndexEnd]),
		DDLColName:     string(src[stmt.ColStart:stmt.ColEnd]),
		DDLUnique:      stmt.Unique,
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
		DDLColumns: []struct {
			Name            string
			Type            string
			VectorDimension uint32
			Flags           uint16
		}{{
			Name:            string(src[stmt.AddColumn.NameStart:stmt.AddColumn.NameEnd]),
			Type:            string(src[stmt.AddColumn.TypeStart:stmt.AddColumn.TypeEnd]),
			VectorDimension: stmt.AddColumn.TypeParam,
			Flags:           stmt.AddColumn.Flags,
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
