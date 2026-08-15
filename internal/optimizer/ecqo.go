package optimizer

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"

	"github.com/xDarkicex/lexer"
	"github.com/xDarkicex/lexer/parser"
	"github.com/xDarkicex/libravdb/internal/catalog"
	"github.com/xDarkicex/libravdb/internal/graph"
)

var (
	ErrUnsupportedQuery = errors.New("unsupported query shape")
)

// decodeSQLStringLiteral keeps the lexer/parser hot path span-based. Most
// literals return a direct source subslice; only literals containing SQL
// escapes need a decoded copy for downstream materialization.
func decodeSQLStringLiteral(src []byte, sl parser.StringLiteral) []byte {
	decode := lexer.DecodeStringLiteralInto
	if sl.Escape {
		decode = lexer.DecodeEscapeStringLiteralInto
	}
	if out, ok := decode(src, sl.Start, sl.End, nil); ok {
		return out
	}
	var scratch [256]byte
	if out, ok := decode(src, sl.Start, sl.End, scratch[:]); ok {
		return out
	}
	// A very large escaped literal cannot fit the stack scratch buffer. This
	// is an explicit materialization allocation, never a lexer/parser one.
	buf := make([]byte, int(sl.End-sl.Start))
	out, _ := decode(src, sl.Start, sl.End, buf)
	return out
}

// resolveVectorParam checks if an identifier starts with '$' or '@' and
// resolves it from the query params map. The returned slice is copied before it enters
// a physical plan, so callers cannot mutate an in-flight query by retaining a
// parameter slice.
func resolveVectorParam(src []byte, id *parser.Identifier, params *ParameterSet) ([]float32, bool) {
	if len(src) > int(id.Start) && (src[id.Start] == '$' || src[id.Start] == '@') {
		if value, ok := params.Lookup(src, id.Start, id.End); ok && value.Kind == ScalarVector {
			return append([]float32(nil), value.Vector...), true
		}
	}
	return nil, false
}

func resolveVectorOperand(doc *parser.QueryDoc, src []byte, ref parser.NodeRef, params *ParameterSet) ([]float32, error) {
	switch ref.Kind {
	case parser.NodeKindCastExpr:
		// PostgreSQL clients commonly bind pgvector parameters as
		// `$1::vector`. The cast is type information around the same parameter
		// node, not a different vector expression; unwrap it before resolving
		// the native []float32 value.
		if ref.ID < 0 || int(ref.ID) >= len(doc.CastExprs) {
			return nil, fmt.Errorf("vector cast expression is malformed")
		}
		return resolveVectorOperand(doc, src, doc.CastExprs[ref.ID].Expr, params)
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
	// TypedValue is the semantic predicate value. Runtime operators must use
	// this field rather than reparsing SQL source text.
	TypedValue ScalarValue
	// Value is retained only for compatibility with older plan constructors and
	// tests. Production optimizer output always sets TypedValue.
	Value []byte
	// ValueIsNull marks a present query parameter whose native value is SQL
	// NULL. Ordinary comparisons with NULL are UNKNOWN and match no rows.
	ValueIsNull bool
	// NullTest is parser.NullTestIsNull or parser.NullTestNotNull.
	NullTest  uint8
	Like      bool          // SQL LIKE pattern match
	ILike     bool          // SQL case-insensitive LIKE pattern match
	Inclusive bool          // inclusive bound used by BETWEEN
	Not       bool          // negated membership/range predicate
	InValues  []ScalarValue // exact membership values for IN/NOT IN
}

// PredicateAlternatives is a disjunctive normal form representation of a
// scalar WHERE expression. Each inner slice is an AND clause and the outer
// slice is OR-ed. The legacy Predicates field remains the fast path for pure
// conjunctions; alternatives are populated when a WHERE expression contains
// OR so execution cannot accidentally treat disjunctions as conjunctions (or
// drop them entirely).
type PredicateAlternatives [][]RelationalPredicate

// PredicateValue returns the canonical typed value, adapting legacy byte-only
// plan construction at an explicit compatibility boundary.
func (p RelationalPredicate) PredicateValue() ScalarValue {
	if p.TypedValue.Kind != ScalarInvalid {
		return p.TypedValue
	}
	if p.ValueIsNull {
		return NullValue()
	}
	return ScalarFromLiteralBytes(p.Value)
}

const (
	NullTestNone    uint8 = 0
	NullTestIsNull  uint8 = 1
	NullTestNotNull uint8 = 2
)

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

// VectorOperatorProjection describes a pgvector-style distance operator in a
// SELECT list, for example `embedding <-> $1 AS distance`. Operator semantics
// are kept separate from the collection's configured metric: SQL explicitly
// chooses L2, negative inner product, or cosine distance at the expression.
type VectorOperatorProjection struct {
	Name        string
	Operator    uint8
	QueryVector []float32
	SourceAlias string
}

// RRFComponent describes one independently-ranked signal in a reciprocal
// rank fusion expression. The executor evaluates all components over the
// same candidate set and fuses their deterministic ranks.
type RRFComponent struct {
	Kind        uint8
	Ascending   bool // lower-is-better, e.g. VECTOR_DISTANCE
	Vector      []float32
	TextColumn  string
	TextQuery   string
	SourceAlias string
}

// FTSRankProjection describes a standalone lexical score projection. RRF
// embeds the same scorer as a component, while this form allows clients to
// inspect or order by the lexical score directly.
type FTSRankProjection struct {
	Name        string
	TextColumn  string
	TextQuery   string
	SourceAlias string
}

type FTSProjectionKind uint8

const (
	FTSProjectionRank FTSProjectionKind = iota
	FTSProjectionVector
	FTSProjectionQuery
)

type FTSProjection struct {
	Name          string
	Kind          FTSProjectionKind
	Config        string
	Column        string
	SourceAlias   string
	Query         string
	QueryMode     string
	Normalization uint32
}

type FTSPredicate struct {
	Config      string
	Column      string
	SourceAlias string
	Text        string
	Query       string
	QueryMode   string
}

// ProjectionRef maps a projected output name (including an SQL alias) back
// to the source column used to materialize it.
type ProjectionRef struct {
	OutputName  string
	SourceName  string
	SourceAlias string
}

const (
	RRFComponentVectorDistance uint8 = iota
	RRFComponentFTSRank
	RRFComponentGraphCentrality
)

// GraphEdgePlan is a single edge extracted from the MATCH path,
// carrying its direction, quantifier bounds, and optional type/kind for traversal.
type GraphEdgePlan struct {
	Direction int8                // -1=inbound, 0=undirected, 1=outbound (from parser.Edge.Direction)
	QuantMin  uint16              // minimum hops (0 for ->*)
	QuantMax  uint16              // maximum hops (0=default→1, QuantUnbounded for ->+/->*)
	EdgeType  string              // edge type name from source (e.g., "KNOWS"); empty if not specified
	EdgeKind  uint8               // resolved kind number from registry; 0 if not specified/registered
	Weight    graph.WeightFilter  // optional edge-local weight predicate
	Predicate graph.EdgePredicate // optional full edge-property boolean predicate
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

// DDLCheckConstraint is a parsed CHECK constraint lowered into the DDL plan.
type DDLCheckConstraint struct {
	Name       string `json:"name,omitempty"`
	Expression string `json:"expression"`
	ColumnName string `json:"column_name,omitempty"`
}

// ConflictExpr is the lowered, allocation-stable expression tree used by
// INSERT ... ON CONFLICT DO UPDATE. It deliberately covers the scalar SQL
// expression subset that can be evaluated against the current row and the
// proposed EXCLUDED row without reparsing SQL at execution time.
type ConflictExpr struct {
	Kind          uint8
	Operator      uint8
	Left          int32
	Right         int32
	CaseWhenStart int32
	CaseWhenCount int32
	CaseElse      int32
	Column        string
	Function      string
	Type          string
	Literal       []byte
	IsNull        bool
}

type ConflictCase struct {
	Condition int32
	Value     int32
}

const (
	ConflictExprLiteral uint8 = iota
	ConflictExprColumn
	ConflictExprExcludedColumn
	ConflictExprBinary
	ConflictExprCase
	ConflictExprFunction
	ConflictExprCast
	ConflictExprUnary
)

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
	// HasVectorOperator marks a direct pgvector operator expression. The
	// operator is authoritative for scoring and may override the collection's
	// configured index metric.
	HasVectorOperator         bool
	HasVectorOperatorOrder    bool
	VectorOperator            uint8
	VectorOperatorProjections []VectorOperatorProjection
	Similarity                float32
	Limit                     int
	UnionQueries              []string
	SetOp                     uint8
	SetOpAll                  bool

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
	HasRelationalQuery    bool
	Predicates            []RelationalPredicate
	PredicateAlternatives PredicateAlternatives
	Projections           []string // SELECT column list (empty = all columns)
	ProjectionRefs        []ProjectionRef
	// Scoring expression — parser AST lowered to execution hints.
	HasScoreExpr       bool
	ScoreArithOp       uint8 // KindAsterisk(11), KindDash(18)
	HasGraphCentrality bool
	ScoreLiteralValue  float64
	ScoreAlias         string // projected alias for a compound score expression
	HasRRF             bool
	RRFAlias           string
	RRFK               float64
	RRFComponents      []RRFComponent
	FTSRankProjections []FTSRankProjection
	FTSProjections     []FTSProjection
	FTSPredicates      []FTSPredicate
	FTSError           string

	OrderBy  string // column name for ORDER BY (empty = none)
	IsDesc   bool   // ORDER BY DESC
	Distinct bool   // SELECT DISTINCT projection deduplication
	Offset   int    // rows to skip after projection/order, before LIMIT

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
	GraphEdgeDelete     bool     // DELETE FROM GRAPH_EDGES uses graph edge predicates
	InsertColumns       []string // column names
	InsertValues        [][]byte // raw value bytes per column, flattened across rows
	InsertSelectSQL     string
	InsertSelectColumns []string
	// InsertValueNull preserves explicit SQL NULL after string literals have
	// been unquoted. It is parallel to InsertValues.
	InsertValueNull []bool
	// ON CONFLICT lowering. Conflict action: 0 none, 1 DO NOTHING, 2 DO UPDATE.
	InsertConflictColumns      []string
	InsertConflictConstraint   string
	InsertConflictAction       uint8
	InsertConflictSetColumns   []string
	InsertConflictSetValues    [][]byte
	InsertConflictSetValueNull []bool
	InsertConflictSetExcluded  []string // non-empty entry means EXCLUDED.<column>
	InsertConflictExprs        []ConflictExpr
	InsertConflictExprRoots    []int32
	InsertConflictCases        []ConflictCase
	InsertConflictWhereRoot    int32
	InsertConflictHasWhere     bool
	// DML RETURNING projection. ReturningColumns preserves source order;
	// ReturningStar requests all visible row fields.
	ReturningColumns []string
	ReturningStar    bool
	SetColumns       []string // SET column names for UPDATE
	SetValues        [][]byte // SET value bytes for UPDATE
	// SetExprs/SetExprRoots preserve UPDATE assignment expressions (for
	// example value << 1 or CASE ... END) instead of silently reducing them to
	// a single literal during lowering.
	SetExprs     []ConflictExpr
	SetExprRoots []int32
	SetExprCases []ConflictCase
	// SetValueNull preserves explicit SQL NULL in UPDATE assignments.
	SetValueNull []bool

	// Aggregate fields — populated when Kind == QueryKindAggregate
	AggregateFunc         uint8    // parser.AggregateFunc value (AggCount, AggSum, etc.)
	AggregateColumn       string   // column name for the aggregate (empty for COUNT(*))
	AggregateAlias        string   // SELECT alias for the aggregate, when present
	AggregateDistinct     bool     // DISTINCT modifier
	GroupByColumns        []string // GROUP BY column names
	HavingExpr            string   // HAVING expression column name
	HavingOp              uint8    // HAVING operator (e.g., KindGreaterThan)
	HavingValue           []byte   // HAVING literal value
	HavingAggregate       bool     // HAVING compares an aggregate result
	HavingAggregateFunc   uint8    // parser.AggregateFunc for the HAVING aggregate
	HavingAggregateColumn string   // aggregate argument, empty for COUNT(*)

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
	DDLDropColumn     bool
	DDLDropColumnName string
	DDLColName        string // CREATE INDEX column
	DDLIfExists       bool   // IF EXISTS modifier
	DDLUnique         bool   // UNIQUE INDEX modifier
	DDLIndexColumns   []string
	// DDLJSONPath/DDLJSONText describe a narrow JSON expression index on one
	// JSON/JSONB column. The path is the PostgreSQL text-array literal without
	// surrounding SQL quotes.
	DDLJSONPath string
	DDLJSONText bool

	// DDLForeignKeys carries parsed FK constraints for CREATE TABLE.
	DDLForeignKeys []DDLForeignKey
	// DDLPrimaryKeyColumns preserves the ordered table-level PRIMARY KEY
	// columns. A nil slice means no table-level composite key was declared.
	DDLPrimaryKeyColumns    []string
	DDLPrimaryKeyConstraint string
	// DDLCheckConstraints carries parsed CHECK constraints for CREATE TABLE.
	DDLCheckConstraints []DDLCheckConstraint
	// DDLColumnDefaults maps column name → default value string for columns
	// declared with DEFAULT <literal>.
	DDLColumnDefaults map[string]string

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
	catalog     *catalog.Catalog
	src         []byte
	params      map[string]interface{} // compatibility-only DML boundary
	boundParams *ParameterSet
}

func NewOptimizer(cat *catalog.Catalog) *Optimizer {
	return &Optimizer{catalog: cat}
}

// Optimize maps a bound AST to a PhysicalPlan.
func (o *Optimizer) Optimize(doc *parser.QueryDoc, src []byte) (*PhysicalPlan, error) {
	return o.OptimizeWithParams(doc, src, nil)
}

func (o *Optimizer) OptimizeWithParams(doc *parser.QueryDoc, src []byte, params map[string]interface{}) (*PhysicalPlan, error) {
	return o.optimize(doc, src, params, NewParameterSet(params))
}

// OptimizeWithBoundParams is the native execution entry point. Callers that
// already decoded protocol values should use this method so predicates and
// vector operands remain typed all the way into planning.
func (o *Optimizer) OptimizeWithBoundParams(doc *parser.QueryDoc, src []byte, params *ParameterSet) (*PhysicalPlan, error) {
	return o.optimize(doc, src, nil, params)
}

// ExtractMatchPath lowers a parsed MATCH path using the same edge-kind,
// quantifier, and edge-property rules used by physical graph plans.  The
// virtual relational executor uses this when a JOIN MATCH appears inside a
// subquery; exposing the lowering keeps that path from growing a second,
// subtly different parser/lowering implementation.
func (o *Optimizer) ExtractMatchPath(doc *parser.QueryDoc, src []byte, mp *parser.MatchPath, params *ParameterSet) ([]GraphEdgePlan, int, error) {
	if o == nil {
		return nil, 0, fmt.Errorf("nil optimizer")
	}
	o.src = src
	o.boundParams = params
	return o.extractMatchPath(doc, src, mp)
}

func (o *Optimizer) optimize(doc *parser.QueryDoc, src []byte, legacyParams map[string]interface{}, params *ParameterSet) (*PhysicalPlan, error) {
	o.src = src
	o.params = legacyParams
	o.boundParams = params
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

	selectIndex := 0
	for i := len(doc.Nodes) - 1; i >= 0; i-- {
		if doc.Nodes[i].Kind == parser.NodeKindSelectStmt {
			selectIndex = int(doc.Nodes[i].ID)
			break
		}
	}
	if selectIndex < 0 || selectIndex >= len(doc.SelectStmts) {
		selectIndex = 0
	}
	stmt := &doc.SelectStmts[selectIndex]
	plan := &PhysicalPlan{
		Limit:    -1, // No limit by default
		Offset:   0,
		Distinct: stmt.Distinct,
	}
	if stmt.SetOp != parser.SetOpNone || stmt.UnionAll {
		if stmt.UnionNext.Kind != parser.NodeKindSelectStmt || stmt.UnionNext.ID < 0 || int(stmt.UnionNext.ID) >= len(doc.SelectStmts) {
			return nil, fmt.Errorf("invalid set-operation branch")
		}
		if stmt.UnionStart <= stmt.SourceStart || stmt.UnionStart > uint32(len(src)) {
			return nil, fmt.Errorf("invalid set-operation source span")
		}
		plan.UnionQueries = append(plan.UnionQueries, string(src[stmt.SourceStart:stmt.UnionStart]))
		branch := doc.SelectStmts[stmt.UnionNext.ID]
		if branch.SourceEnd <= branch.SourceStart || branch.SourceEnd > uint32(len(src)) {
			return nil, fmt.Errorf("invalid set-operation branch span")
		}
		plan.UnionQueries = append(plan.UnionQueries, string(src[branch.SourceStart:branch.SourceEnd]))
		plan.SetOp = uint8(stmt.SetOp)
		if plan.SetOp == 0 && stmt.UnionAll {
			plan.SetOp = uint8(parser.SetOpUnion)
		}
		plan.SetOpAll = stmt.SetOpAll || stmt.UnionAll
	}

	// 1. Resolve FROM clause
	if stmt.FromTable.Kind == parser.NodeKindTableExpr {
		t := &doc.TableExprs[stmt.FromTable.ID]
		plan.CollectionOID = t.TableOID
		plan.CollectionName = string(src[t.Start:t.End])
		if t.Temporal {
			ts := string(src[t.TimestampStart:t.TimestampEnd])
			// AS OF TIMESTAMP accepts a native bound $N/@name parameter in
			// addition to a quoted literal. Resolve it at the typed execution
			// boundary; never rewrite the SQL source into a quoted string.
			if len(ts) > 1 && (ts[0] == '$' || ts[0] == '@') {
				if value, found := o.boundParams.Lookup(src, t.TimestampStart, t.TimestampEnd); found {
					switch value.Kind {
					case ScalarString, ScalarBytes:
						ts = string(value.BytesData)
					case ScalarTimestamp:
						ts = value.Time.UTC().Format(time.RFC3339Nano)
					default:
						return nil, fmt.Errorf("AS OF TIMESTAMP parameter must be text or timestamp, got %d", value.Kind)
					}
				} else {
					return nil, fmt.Errorf("AS OF TIMESTAMP parameter %q is not bound", ts)
				}
			}
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
			var extractErr error
			plan.GraphEdges, plan.MaxHops, extractErr = o.extractMatchPath(doc, src, mp)
			if extractErr != nil {
				return nil, extractErr
			}
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
		if plan.FTSError != "" {
			return nil, fmt.Errorf("full-text predicate: %s", plan.FTSError)
		}

		// WHERE MATCH: walk AND chains to find the MatchPath.
		if err := o.extractWhereMatch(doc, src, plan, whereNode); err != nil {
			return nil, err
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

					vec, err := resolveVectorOperand(doc, src, vf.VectorB, o.boundParams)
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
			for i := int32(0); i < mp.PathNodesCount; i++ {
				ref := doc.Nodes[mp.PathNodesStart+i]
				if ref.Kind != parser.NodeKindVertex {
					continue
				}
				v := &doc.Vertexes[ref.ID]
				if v.AliasEnd > v.Alias {
					gjp.TerminalAlias = string(src[v.Alias:v.AliasEnd])
				}
				if v.LabelEnd > v.LabelStart {
					gjp.TerminalLabel = string(src[v.LabelStart:v.LabelEnd])
				}
			}
			var extractErr error
			gjp.GraphEdges, gjp.MaxHops, extractErr = o.extractMatchPath(doc, src, mp)
			if extractErr != nil {
				return nil, extractErr
			}
			for _, predicate := range plan.Predicates {
				if predicate.Alias != "" && strings.EqualFold(predicate.Alias, gjp.TerminalAlias) {
					gjp.TerminalPredicates = append(gjp.TerminalPredicates, predicate)
				}
			}
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
	if stmt.OrderBy.Kind == parser.NodeKindBinaryExpr && stmt.OrderBy.ID >= 0 && int(stmt.OrderBy.ID) < len(doc.BinaryExprs) &&
		isPgVectorOp(doc.BinaryExprs[stmt.OrderBy.ID].Operator) {
		info, err := o.lowerVectorOperator(doc, src, stmt.OrderBy)
		if err != nil {
			return nil, err
		}
		plan.HasVectorSearch = true
		plan.HasVectorOperator = true
		plan.HasVectorOperatorOrder = true
		plan.VectorOperator = info.Operator
		plan.VectorIndexOID = info.TableOID
		plan.QueryVector = info.QueryVector
		plan.IsDesc = stmt.IsDesc
		// The direct operator is already sorted by the vector executor. Do not
		// send its synthetic expression through the ordinary column sorter.
		plan.OrderBy = ""
	} else if stmt.OrderBy.Kind == parser.NodeKindVectorFunc {
		vf := &doc.VectorFuncs[stmt.OrderBy.ID]
		vec, err := resolveVectorOperand(doc, src, vf.VectorB, o.boundParams)
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
				outputName := string(src[id.Start:id.End])
				if proj.AliasEnd > proj.Alias {
					outputName = string(src[proj.Alias:proj.AliasEnd])
				}
				plan.Projections = append(plan.Projections, outputName)
				ref := ProjectionRef{OutputName: outputName, SourceName: string(src[id.Start:id.End])}
				if id.QualEnd > id.QualStart {
					ref.SourceAlias = string(src[id.QualStart:id.QualEnd])
				}
				plan.ProjectionRefs = append(plan.ProjectionRefs, ref)
			} else if proj.Expr.Kind == parser.NodeKindAggregateExpr {
				hasAggregate = true
				ae := &doc.AggregateExprs[proj.Expr.ID]
				plan.AggregateFunc = uint8(ae.Func)
				plan.AggregateDistinct = ae.Distinct
				if proj.AliasEnd > proj.Alias {
					plan.AggregateAlias = string(src[proj.Alias:proj.AliasEnd])
				}
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
				vec, err := resolveVectorOperand(doc, src, vf.VectorB, o.boundParams)
				if err != nil {
					return nil, err
				}
				vfp.QueryVector = vec
				plan.VectorFuncProjections = append(plan.VectorFuncProjections, vfp)
				plan.Projections = append(plan.Projections, vfp.Name)
			} else if proj.Expr.Kind == parser.NodeKindBinaryExpr && proj.Expr.ID >= 0 && int(proj.Expr.ID) < len(doc.BinaryExprs) &&
				isPgVectorOp(doc.BinaryExprs[proj.Expr.ID].Operator) {
				info, err := o.lowerVectorOperator(doc, src, proj.Expr)
				if err != nil {
					return nil, err
				}
				name := "vector_distance"
				if proj.AliasEnd > proj.Alias {
					name = string(src[proj.Alias:proj.AliasEnd])
				}
				info.Name = name
				plan.VectorOperatorProjections = append(plan.VectorOperatorProjections, VectorOperatorProjection{
					Name: info.Name, Operator: info.Operator, QueryVector: info.QueryVector, SourceAlias: info.SourceAlias,
				})
				plan.HasVectorSearch = true
				plan.HasVectorOperator = true
				if !plan.HasVectorOperatorOrder {
					plan.VectorOperator = info.Operator
					plan.VectorIndexOID = info.TableOID
					plan.QueryVector = append([]float32(nil), info.QueryVector...)
				}
				plan.Projections = append(plan.Projections, name)
			} else if proj.Expr.Kind == parser.NodeKindFunctionExpr {
				fn := &doc.FunctionExprs[proj.Expr.ID]
				if asciiEqualFold(src[fn.NameStart:fn.NameEnd], []byte("RRF")) {
					if err := o.lowerRRF(doc, src, proj.Expr, plan); err != nil {
						return nil, err
					}
					plan.HasScoreExpr = true
					plan.HasRRF = true
					name := "rrf"
					if proj.AliasEnd > proj.Alias {
						name = string(src[proj.Alias:proj.AliasEnd])
					}
					plan.RRFAlias = name
					plan.Projections = append(plan.Projections, name)
				} else if asciiEqualFold(src[fn.NameStart:fn.NameEnd], []byte("FTS_RANK")) {
					component, err := o.lowerFTSRank(doc, src, proj.Expr)
					if err != nil {
						return nil, err
					}
					name := "fts_rank"
					if proj.AliasEnd > proj.Alias {
						name = string(src[proj.Alias:proj.AliasEnd])
					}
					plan.FTSRankProjections = append(plan.FTSRankProjections, FTSRankProjection{
						Name: name, TextColumn: component.TextColumn, TextQuery: component.TextQuery, SourceAlias: component.SourceAlias,
					})
					plan.Projections = append(plan.Projections, name)
					plan.Kind = QueryKindRelational
					plan.HasRelationalQuery = true
				} else if isCoreFTSFunction(src[fn.NameStart:fn.NameEnd]) {
					fts, err := o.lowerFTSProjection(doc, src, proj.Expr)
					if err != nil {
						return nil, err
					}
					name := string(src[fn.NameStart:fn.NameEnd])
					if proj.AliasEnd > proj.Alias {
						name = string(src[proj.Alias:proj.AliasEnd])
					}
					fts.Name = name
					plan.FTSProjections = append(plan.FTSProjections, fts)
					plan.Projections = append(plan.Projections, name)
					plan.Kind = QueryKindRelational
					plan.HasRelationalQuery = true
				} else {
					return nil, fmt.Errorf("unsupported function projection %q", string(src[fn.NameStart:fn.NameEnd]))
				}
			} else if proj.AliasEnd > proj.Alias {
				// Compound score expressions (for example
				// `(1.0 - VECTOR_DISTANCE(...)) * GRAPH_CENTRALITY(...) AS
				// authoritative_relevance`) are represented by their SQL alias.
				// Keep that alias in RowDescription so protocol clients receive
				// the projected score instead of only the record column.
				alias := string(src[proj.Alias:proj.AliasEnd])
				plan.Projections = append(plan.Projections, alias)
				if proj.Expr.Kind == parser.NodeKindBinaryExpr {
					plan.ScoreAlias = alias
				}
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
		// Keep HasRelationalQuery set when a WHERE predicate was lowered. The
		// vector-projection executor is also responsible for applying those
		// predicates (including in the temporal path); clearing this bit turns
		// `... VECTOR_DISTANCE(...) ... WHERE id = ...` into an unfiltered
		// full scan and can return the wrong historical row.
	}
	// Direct pgvector operators use the same full historical scan machinery as
	// VECTOR_DISTANCE projections. Preserve the operator fields so temporal
	// execution scores with the SQL-selected metric instead of the collection
	// default. Graph/join plans retain their own classification.
	if plan.HasVectorOperator && plan.Kind == QueryKindRelational &&
		len(plan.Joins) == 0 && len(plan.GraphJoins) == 0 && !plan.HasGraphTraversal {
		plan.Kind = QueryKindVectorProjection
	}
	// A projected vector distance ordered with LIMIT is still a vector top-k
	// operation.  When relational JOIN and JOIN MATCH are present together,
	// retain every clause in one physical plan rather than allowing one legacy
	// executor path to discard the others.
	// Classify as multimodal when graph traversal (JOIN MATCH or WHERE MATCH)
	// and vector projection are both present. Relational JOIN is optional.
	if plan.HasRRF {
		// RRF is a candidate-set scorer even when its components do not include
		// a graph signal. Route it through the multimodal executor so it can
		// apply relational, graph, vector, and lexical components uniformly.
		plan.Kind = QueryKindMultiModal
		plan.HasRelationalQuery = true
	}
	hasGraph := len(plan.GraphJoins) > 0 || plan.HasGraphTraversal
	hasVector := len(plan.VectorFuncProjections) > 0 || plan.HasScoreExpr || plan.HasVectorOperator
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
		if be.Left.Kind == parser.NodeKindAggregateExpr {
			ae := &doc.AggregateExprs[be.Left.ID]
			plan.HavingAggregate = true
			plan.HavingAggregateFunc = uint8(ae.Func)
			if ae.Expr.Kind == parser.NodeKindIdentifier {
				id := &doc.Identifiers[ae.Expr.ID]
				plan.HavingAggregateColumn = string(src[id.Start:id.End])
			}
			plan.HavingOp = be.Operator
			if be.Right.Kind == parser.NodeKindNumber {
				num := &doc.Numbers[be.Right.ID]
				plan.HavingValue = src[num.Start:num.End]
			} else if be.Right.Kind == parser.NodeKindString {
				sl := &doc.Strings[be.Right.ID]
				plan.HavingValue = decodeSQLStringLiteral(src, *sl)
			}
		} else if be.Left.Kind == parser.NodeKindIdentifier {
			id := &doc.Identifiers[be.Left.ID]
			plan.HavingExpr = string(src[id.Start:id.End])
			plan.HavingOp = be.Operator
			if be.Right.Kind == parser.NodeKindNumber {
				plan.HavingValue = src[doc.Numbers[be.Right.ID].Start:doc.Numbers[be.Right.ID].End]
			} else if be.Right.Kind == parser.NodeKindString {
				sl := &doc.Strings[be.Right.ID]
				plan.HavingValue = decodeSQLStringLiteral(src, *sl)
			}
		}
	}

	if hasAggregate {
		plan.Kind = QueryKindAggregate
	}

	// 5. Map LIMIT/OFFSET. Literal clauses use the parser's Number arena;
	// parameterized clauses resolve through the native typed parameter set.
	if stmt.LimitExpr.Kind != parser.NodeKindUnknown {
		value, found := o.scalarForRef(doc, src, stmt.LimitExpr)
		if !found || value.IsNull() {
			return nil, fmt.Errorf("LIMIT parameter is missing or NULL")
		}
		limit, ok := scalarNonNegativeInt(value)
		if !ok {
			return nil, fmt.Errorf("LIMIT must be a non-negative integer")
		}
		plan.Limit = limit
	} else if stmt.Limit >= 0 {
		num := &doc.Numbers[stmt.Limit]
		// Parse string to int
		limitStr := string(src[num.Start:num.End])
		var l int
		fmt.Sscanf(limitStr, "%d", &l)
		plan.Limit = l
	}
	if stmt.OffsetExpr.Kind != parser.NodeKindUnknown {
		value, found := o.scalarForRef(doc, src, stmt.OffsetExpr)
		if !found || value.IsNull() {
			return nil, fmt.Errorf("OFFSET parameter is missing or NULL")
		}
		offset, ok := scalarNonNegativeInt(value)
		if !ok {
			return nil, fmt.Errorf("OFFSET must be a non-negative integer")
		}
		plan.Offset = offset
	} else if stmt.Offset >= 0 {
		num := &doc.Numbers[stmt.Offset]
		var offset int
		fmt.Sscanf(string(src[num.Start:num.End]), "%d", &offset)
		if offset < 0 {
			return nil, fmt.Errorf("OFFSET must not be negative")
		}
		plan.Offset = offset
	}

	return plan, nil
}

func isCoreFTSFunction(name []byte) bool {
	return asciiEqualFold(name, []byte("to_tsvector")) ||
		asciiEqualFold(name, []byte("to_tsquery")) ||
		asciiEqualFold(name, []byte("plainto_tsquery")) ||
		asciiEqualFold(name, []byte("phraseto_tsquery")) ||
		asciiEqualFold(name, []byte("websearch_to_tsquery")) ||
		asciiEqualFold(name, []byte("ts_rank")) ||
		asciiEqualFold(name, []byte("ts_rank_cd"))
}

func scalarNonNegativeInt(value ScalarValue) (int, bool) {
	var n int64
	switch value.Kind {
	case ScalarInt:
		n = value.Int
	case ScalarFloat:
		if value.Float != float64(int64(value.Float)) {
			return 0, false
		}
		n = int64(value.Float)
	case ScalarString, ScalarBytes:
		parsed, err := strconv.ParseInt(string(value.BytesData), 10, 64)
		if err != nil {
			return 0, false
		}
		n = parsed
	default:
		return 0, false
	}
	if n < 0 || uint64(n) > uint64(^uint(0)>>1) {
		return 0, false
	}
	return int(n), true
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
			val := string(decodeSQLStringLiteral(src, *sl))
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
func (o *Optimizer) extractMatchPath(doc *parser.QueryDoc, src []byte, mp *parser.MatchPath) ([]GraphEdgePlan, int, error) {
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
		if e.Predicate.Kind != parser.NodeKindUnknown {
			filter, predicateKind, predicate, err := o.lowerEdgePredicates(doc, src, e)
			if err != nil {
				return nil, 0, err
			}
			gep.Weight = filter
			gep.Predicate = predicate
			if predicateKind != 0 {
				if gep.EdgeKind != 0 && gep.EdgeKind != predicateKind {
					return nil, 0, fmt.Errorf("edge type predicate conflicts with edge type %q", gep.EdgeType)
				}
				gep.EdgeKind = predicateKind
			}
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
	return edges, maxHops, nil
}

// lowerEdgePredicates lowers the compact edge-property form into the fields
// already carried by graph.EdgePlan. The storage representation remains the
// node-owned Edge record: weight is Edge.Weight and type/kind is encoded in
// Edge.Stamp. Values are resolved during planning, so graph traversal never
// parses SQL or consults a parameter map.
func (o *Optimizer) lowerEdgePredicates(doc *parser.QueryDoc, src []byte, edge *parser.Edge) (graph.WeightFilter, uint8, graph.EdgePredicate, error) {
	if edgePredicateNeedsGeneral(doc, src, edge.Predicate) {
		predicate, err := o.lowerEdgePredicateTree(doc, src, edge)
		if err != nil {
			return graph.WeightFilter{}, 0, graph.EdgePredicate{}, err
		}
		return graph.WeightFilter{}, 0, predicate, nil
	}

	var weight graph.WeightFilter
	var kind uint8

	var leaves []parser.NodeRef
	var collect func(parser.NodeRef) error
	collect = func(ref parser.NodeRef) error {
		if ref.Kind != parser.NodeKindBinaryExpr || ref.ID < 0 || int(ref.ID) >= len(doc.BinaryExprs) {
			return fmt.Errorf("edge predicate must be a comparison")
		}
		be := &doc.BinaryExprs[ref.ID]
		if be.Operator == uint8(lexer.KindAnd) {
			if err := collect(be.Left); err != nil {
				return err
			}
			return collect(be.Right)
		}
		leaves = append(leaves, ref)
		return nil
	}
	if err := collect(edge.Predicate); err != nil {
		return graph.WeightFilter{}, 0, graph.EdgePredicate{}, err
	}

	for _, ref := range leaves {
		node, err := o.lowerEdgePredicateComparison(doc, src, edge, ref)
		if err != nil {
			return graph.WeightFilter{}, 0, graph.EdgePredicate{}, err
		}
		switch node.Property {
		case graph.EdgePropertyWeight:
			if weight.Enabled {
				return graph.WeightFilter{}, 0, graph.EdgePredicate{}, fmt.Errorf("multiple edge weight predicates are not supported")
			}
			weight = graph.WeightFilter{Enabled: true, Op: node.Compare, Value: node.Weight}

		case graph.EdgePropertyKind:
			if kind != 0 {
				return graph.WeightFilter{}, 0, graph.EdgePredicate{}, fmt.Errorf("multiple edge type predicates are not supported")
			}
			if node.Compare != graph.WeightEqual {
				return graph.WeightFilter{}, 0, graph.EdgePredicate{}, fmt.Errorf("edge type predicate requires equality")
			}
			kind = node.Kind

		default:
			return graph.WeightFilter{}, 0, graph.EdgePredicate{}, fmt.Errorf("unsupported edge property")
		}
	}
	return weight, kind, graph.EdgePredicate{}, nil
}

func edgePredicateNeedsGeneral(doc *parser.QueryDoc, src []byte, ref parser.NodeRef) bool {
	weightCount, kindCount := 0, 0
	var walk func(parser.NodeRef) bool
	walk = func(ref parser.NodeRef) bool {
		if ref.Kind != parser.NodeKindBinaryExpr || ref.ID < 0 || int(ref.ID) >= len(doc.BinaryExprs) {
			return false
		}
		be := &doc.BinaryExprs[ref.ID]
		if be.Operator == uint8(lexer.KindOr) {
			return true
		}
		if be.Operator == uint8(lexer.KindAnd) {
			return walk(be.Left) || walk(be.Right)
		}
		if be.Left.Kind != parser.NodeKindIdentifier || be.Left.ID < 0 || int(be.Left.ID) >= len(doc.Identifiers) {
			return false
		}
		id := &doc.Identifiers[be.Left.ID]
		name := string(src[id.Start:id.End])
		switch {
		case strings.EqualFold(name, "weight"):
			weightCount++
		case strings.EqualFold(name, "type") || strings.EqualFold(name, "kind"):
			kindCount++
			if lexer.Kind(be.Operator) != lexer.KindEquals {
				return true
			}
		default:
			return true
		}
		return weightCount > 1 || kindCount > 1
	}
	return walk(ref)
}

func (o *Optimizer) lowerEdgePredicateTree(doc *parser.QueryDoc, src []byte, edge *parser.Edge) (graph.EdgePredicate, error) {
	nodes := make([]graph.EdgePredicateNode, 0, 4)
	var lower func(parser.NodeRef) (int32, error)
	lower = func(ref parser.NodeRef) (int32, error) {
		if ref.Kind != parser.NodeKindBinaryExpr || ref.ID < 0 || int(ref.ID) >= len(doc.BinaryExprs) {
			return 0, fmt.Errorf("edge predicate must be a comparison")
		}
		be := &doc.BinaryExprs[ref.ID]
		if be.Operator == uint8(lexer.KindAnd) || be.Operator == uint8(lexer.KindOr) {
			left, err := lower(be.Left)
			if err != nil {
				return 0, err
			}
			right, err := lower(be.Right)
			if err != nil {
				return 0, err
			}
			op := graph.EdgePredicateAnd
			if be.Operator == uint8(lexer.KindOr) {
				op = graph.EdgePredicateOr
			}
			nodes = append(nodes, graph.EdgePredicateNode{Op: op, Left: left, Right: right})
			return int32(len(nodes) - 1), nil
		}
		node, err := o.lowerEdgePredicateComparison(doc, src, edge, ref)
		if err != nil {
			return 0, err
		}
		nodes = append(nodes, node)
		return int32(len(nodes) - 1), nil
	}
	root, err := lower(edge.Predicate)
	if err != nil {
		return graph.EdgePredicate{}, err
	}
	return graph.EdgePredicate{Nodes: nodes, Root: root}, nil
}

func (o *Optimizer) lowerEdgePredicateComparison(doc *parser.QueryDoc, src []byte, edge *parser.Edge, ref parser.NodeRef) (graph.EdgePredicateNode, error) {
	be := &doc.BinaryExprs[ref.ID]
	if be.Left.Kind != parser.NodeKindIdentifier || be.Left.ID < 0 || int(be.Left.ID) >= len(doc.Identifiers) {
		return graph.EdgePredicateNode{}, fmt.Errorf("edge predicate must reference an edge property")
	}
	left := &doc.Identifiers[be.Left.ID]
	if left.Start >= uint32(len(src)) || left.End > uint32(len(src)) {
		return graph.EdgePredicateNode{}, fmt.Errorf("invalid edge property span")
	}
	if left.QualEnd > left.QualStart {
		if edge.AliasEnd <= edge.Alias || !strings.EqualFold(string(src[left.QualStart:left.QualEnd]), string(src[edge.Alias:edge.AliasEnd])) {
			return graph.EdgePredicateNode{}, fmt.Errorf("edge predicate qualifier does not match edge alias")
		}
	}

	name := string(src[left.Start:left.End])
	switch {
	case strings.EqualFold(name, "weight"):
		value, err := o.edgeWeightValue(doc, src, be.Right)
		if err != nil {
			return graph.EdgePredicateNode{}, err
		}
		op, ok := edgeWeightOperator(lexer.Kind(be.Operator))
		if !ok {
			return graph.EdgePredicateNode{}, fmt.Errorf("unsupported edge weight comparison")
		}
		return graph.EdgePredicateNode{Op: graph.EdgePredicateComparison, Property: graph.EdgePropertyWeight, Compare: op, Weight: float32(value)}, nil

	case strings.EqualFold(name, "type") || strings.EqualFold(name, "kind"):
		var op graph.WeightOp
		switch lexer.Kind(be.Operator) {
		case lexer.KindEquals:
			op = graph.WeightEqual
		case lexer.KindNotEqual:
			op = graph.WeightNotEqual
		default:
			return graph.EdgePredicateNode{}, fmt.Errorf("edge type predicate requires equality or inequality")
		}
		kindName, err := o.edgeKindValue(doc, src, be.Right)
		if err != nil {
			return graph.EdgePredicateNode{}, err
		}
		kind := graph.ResolveEdgeKind(kindName)
		if kind == 0 {
			return graph.EdgePredicateNode{}, fmt.Errorf("unknown edge type %q", kindName)
		}
		return graph.EdgePredicateNode{Op: graph.EdgePredicateComparison, Property: graph.EdgePropertyKind, Compare: op, Kind: kind}, nil

	default:
		expected, err := o.edgePropertyValue(doc, src, be.Right)
		if err != nil {
			return graph.EdgePredicateNode{}, err
		}
		op, ok := edgeWeightOperator(lexer.Kind(be.Operator))
		if !ok {
			return graph.EdgePredicateNode{}, fmt.Errorf("unsupported edge property comparison")
		}
		return graph.EdgePredicateNode{
			Op: graph.EdgePredicateComparison, Property: graph.EdgePropertyArbitrary,
			Compare: op, Name: name, Value: expected,
		}, nil
	}
}

func (o *Optimizer) edgePropertyValue(doc *parser.QueryDoc, src []byte, ref parser.NodeRef) (graph.EdgePropertyValue, error) {
	switch ref.Kind {
	case parser.NodeKindNumber:
		if ref.ID < 0 || int(ref.ID) >= len(doc.Numbers) {
			return graph.EdgePropertyValue{}, fmt.Errorf("invalid edge property value")
		}
		n := doc.Numbers[ref.ID]
		value, err := strconv.ParseFloat(string(src[n.Start:n.End]), 64)
		if err != nil || math.IsNaN(value) || math.IsInf(value, 0) {
			return graph.EdgePropertyValue{}, fmt.Errorf("edge property comparison requires a finite numeric value")
		}
		return graph.EdgePropertyValue{Kind: graph.EdgePropertyNumber, Number: value}, nil
	case parser.NodeKindString:
		if ref.ID < 0 || int(ref.ID) >= len(doc.Strings) {
			return graph.EdgePropertyValue{}, fmt.Errorf("invalid edge property string value")
		}
		return graph.EdgePropertyValue{Kind: graph.EdgePropertyString, String: string(decodeSQLStringLiteral(src, doc.Strings[ref.ID]))}, nil
	case parser.NodeKindIdentifier:
		if ref.ID < 0 || int(ref.ID) >= len(doc.Identifiers) {
			return graph.EdgePropertyValue{}, fmt.Errorf("invalid edge property parameter")
		}
		id := doc.Identifiers[ref.ID]
		raw := src[id.Start:id.End]
		if len(raw) > 0 && (raw[0] == '$' || raw[0] == '@') {
			value, found := o.resolveParamScalar(doc, src, ref)
			if !found {
				return graph.EdgePropertyValue{}, fmt.Errorf("edge property parameter %q is missing", string(raw))
			}
			return edgePropertyValueFromScalar(value)
		}
		if strings.EqualFold(string(raw), "null") {
			return graph.EdgePropertyValue{Kind: graph.EdgePropertyNull}, nil
		}
		if strings.EqualFold(string(raw), "true") || strings.EqualFold(string(raw), "false") {
			return graph.EdgePropertyValue{Kind: graph.EdgePropertyBool, Bool: strings.EqualFold(string(raw), "true")}, nil
		}
	}
	return graph.EdgePropertyValue{}, fmt.Errorf("edge property comparison requires a scalar literal or bound parameter")
}

func edgePropertyValueFromScalar(value ScalarValue) (graph.EdgePropertyValue, error) {
	switch value.Kind {
	case ScalarNull:
		return graph.EdgePropertyValue{Kind: graph.EdgePropertyNull}, nil
	case ScalarInt:
		return graph.EdgePropertyValue{Kind: graph.EdgePropertyNumber, Number: float64(value.Int)}, nil
	case ScalarFloat:
		if math.IsNaN(value.Float) || math.IsInf(value.Float, 0) {
			return graph.EdgePropertyValue{}, fmt.Errorf("edge property parameter must be finite")
		}
		return graph.EdgePropertyValue{Kind: graph.EdgePropertyNumber, Number: value.Float}, nil
	case ScalarString, ScalarBytes:
		return graph.EdgePropertyValue{Kind: graph.EdgePropertyString, String: string(value.BytesData)}, nil
	case ScalarBool:
		return graph.EdgePropertyValue{Kind: graph.EdgePropertyBool, Bool: value.Bool}, nil
	default:
		return graph.EdgePropertyValue{}, fmt.Errorf("unsupported edge property parameter type")
	}
}

func (o *Optimizer) edgeKindValue(doc *parser.QueryDoc, src []byte, ref parser.NodeRef) (string, error) {
	if ref.Kind == parser.NodeKindString && ref.ID >= 0 && int(ref.ID) < len(doc.Strings) {
		return string(decodeSQLStringLiteral(src, doc.Strings[ref.ID])), nil
	}
	if ref.Kind == parser.NodeKindIdentifier && ref.ID >= 0 && int(ref.ID) < len(doc.Identifiers) {
		id := doc.Identifiers[ref.ID]
		if id.Start < uint32(len(src)) && id.End <= uint32(len(src)) {
			return string(src[id.Start:id.End]), nil
		}
	}
	return "", fmt.Errorf("edge type predicate requires a string or identifier value")
}

// lowerEdgeWeightFilter validates and binds the compact edge-property form
// supported by the graph engine: alias.weight <op> numeric value. The value is
// resolved during planning so graph traversal never parses SQL or consults a
// parameter map in its hot loop.
func (o *Optimizer) lowerEdgeWeightFilter(doc *parser.QueryDoc, src []byte, edge *parser.Edge) (graph.WeightFilter, error) {
	if edge == nil || edge.Predicate.Kind != parser.NodeKindBinaryExpr || edge.Predicate.ID < 0 || int(edge.Predicate.ID) >= len(doc.BinaryExprs) {
		return graph.WeightFilter{}, fmt.Errorf("edge predicate must be a comparison on weight")
	}
	be := &doc.BinaryExprs[edge.Predicate.ID]
	if be.Left.Kind != parser.NodeKindIdentifier || be.Left.ID < 0 || int(be.Left.ID) >= len(doc.Identifiers) {
		return graph.WeightFilter{}, fmt.Errorf("edge predicate must reference edge weight")
	}
	left := &doc.Identifiers[be.Left.ID]
	if left.Start >= uint32(len(src)) || left.End > uint32(len(src)) || !asciiEqualFold(src[left.Start:left.End], []byte("weight")) {
		return graph.WeightFilter{}, fmt.Errorf("unsupported edge property; only weight is supported")
	}
	if left.QualEnd > left.QualStart {
		if edge.AliasEnd <= edge.Alias || !asciiEqualFold(src[left.QualStart:left.QualEnd], src[edge.Alias:edge.AliasEnd]) {
			return graph.WeightFilter{}, fmt.Errorf("edge predicate qualifier does not match edge alias")
		}
	}
	value, err := o.edgeWeightValue(doc, src, be.Right)
	if err != nil {
		return graph.WeightFilter{}, err
	}
	op, ok := edgeWeightOperator(lexer.Kind(be.Operator))
	if !ok {
		return graph.WeightFilter{}, fmt.Errorf("unsupported edge weight comparison")
	}
	return graph.WeightFilter{Enabled: true, Op: op, Value: float32(value)}, nil
}

func (o *Optimizer) edgeWeightValue(doc *parser.QueryDoc, src []byte, ref parser.NodeRef) (float64, error) {
	if ref.Kind == parser.NodeKindNumber && ref.ID >= 0 && int(ref.ID) < len(doc.Numbers) {
		n := doc.Numbers[ref.ID]
		value, err := strconv.ParseFloat(string(src[n.Start:n.End]), 64)
		if err != nil || math.IsNaN(value) || math.IsInf(value, 0) {
			return 0, fmt.Errorf("edge weight predicate requires a finite numeric value")
		}
		return value, nil
	}
	if ref.Kind == parser.NodeKindIdentifier {
		value, found := o.resolveParamScalar(doc, src, ref)
		if found {
			switch value.Kind {
			case ScalarInt:
				return float64(value.Int), nil
			case ScalarFloat:
				if !math.IsNaN(value.Float) && !math.IsInf(value.Float, 0) {
					return value.Float, nil
				}
			}
		}
	}
	return 0, fmt.Errorf("edge weight predicate requires a numeric literal or bound parameter")
}

func edgeWeightOperator(kind lexer.Kind) (graph.WeightOp, bool) {
	switch kind {
	case lexer.KindEquals:
		return graph.WeightEqual, true
	case lexer.KindGreaterThan:
		return graph.WeightGreater, true
	case lexer.KindLessThan:
		return graph.WeightLess, true
	case lexer.KindGreaterEqual:
		return graph.WeightGreaterEqual, true
	case lexer.KindLessEqual:
		return graph.WeightLessEqual, true
	case lexer.KindNotEqual:
		return graph.WeightNotEqual, true
	default:
		return 0, false
	}
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

// lowerRRF validates and lowers RRF(signal, ...) into compact component
// descriptors. RRF deliberately accepts only database-owned scoring signals;
// arbitrary scalar expressions would make rank direction ambiguous and would
// turn a deterministic rank fusion operator into raw arithmetic.
func (o *Optimizer) lowerRRF(doc *parser.QueryDoc, src []byte, ref parser.NodeRef, plan *PhysicalPlan) error {
	if ref.Kind != parser.NodeKindFunctionExpr || ref.ID < 0 || int(ref.ID) >= len(doc.FunctionExprs) {
		return fmt.Errorf("RRF requires a function expression")
	}
	fn := &doc.FunctionExprs[ref.ID]
	if !asciiEqualFold(src[fn.NameStart:fn.NameEnd], []byte("RRF")) {
		return fmt.Errorf("expected RRF function")
	}
	if fn.HasWindow {
		return fmt.Errorf("RRF window usage is not supported")
	}
	if fn.ArgsCount < 2 || fn.ArgsStart < 0 || fn.ArgsStart+fn.ArgsCount > int32(len(doc.FunctionArgs)) {
		return fmt.Errorf("RRF requires at least two scoring signals")
	}
	plan.RRFK = 60
	plan.RRFComponents = make([]RRFComponent, 0, fn.ArgsCount)
	for i := int32(0); i < fn.ArgsCount; i++ {
		arg := doc.FunctionArgs[fn.ArgsStart+i]
		component, err := o.lowerRRFComponent(doc, src, arg)
		if err != nil {
			return fmt.Errorf("RRF signal %d: %w", i+1, err)
		}
		plan.RRFComponents = append(plan.RRFComponents, component)
		if component.Kind == RRFComponentVectorDistance {
			plan.QueryVector = append([]float32(nil), component.Vector...)
			plan.HasVectorSearch = true
		}
		if component.Kind == RRFComponentGraphCentrality {
			plan.HasGraphCentrality = true
		}
	}
	return nil
}

func (o *Optimizer) lowerRRFComponent(doc *parser.QueryDoc, src []byte, ref parser.NodeRef) (RRFComponent, error) {
	switch ref.Kind {
	case parser.NodeKindVectorFunc:
		if ref.ID < 0 || int(ref.ID) >= len(doc.VectorFuncs) {
			return RRFComponent{}, fmt.Errorf("vector signal is invalid")
		}
		vf := &doc.VectorFuncs[ref.ID]
		vector, err := resolveVectorOperand(doc, src, vf.VectorB, o.boundParams)
		if err != nil {
			return RRFComponent{}, err
		}
		if vf.VectorA.Kind != parser.NodeKindIdentifier {
			return RRFComponent{}, fmt.Errorf("vector signal requires a vector column")
		}
		component := RRFComponent{
			Kind:      RRFComponentVectorDistance,
			Ascending: !vf.IsMaxSim,
			Vector:    vector,
		}
		id := &doc.Identifiers[vf.VectorA.ID]
		if id.QualEnd > id.QualStart {
			component.SourceAlias = string(src[id.QualStart:id.QualEnd])
		}
		return component, nil
	case parser.NodeKindGraphMetric:
		if ref.ID < 0 || int(ref.ID) >= len(doc.GraphMetrics) {
			return RRFComponent{}, fmt.Errorf("graph signal is invalid")
		}
		gm := &doc.GraphMetrics[ref.ID]
		if gm.Kind != 0 {
			return RRFComponent{}, fmt.Errorf("unsupported graph metric kind %d", gm.Kind)
		}
		return RRFComponent{Kind: RRFComponentGraphCentrality}, nil
	case parser.NodeKindFunctionExpr:
		if ref.ID < 0 || int(ref.ID) >= len(doc.FunctionExprs) {
			return RRFComponent{}, fmt.Errorf("lexical signal is invalid")
		}
		component, err := o.lowerFTSRank(doc, src, ref)
		component.Kind = RRFComponentFTSRank
		return component, err
	default:
		return RRFComponent{}, fmt.Errorf("expected VECTOR_DISTANCE, FTS_RANK, or GRAPH_CENTRALITY")
	}
}

func (o *Optimizer) lowerFTSRank(doc *parser.QueryDoc, src []byte, ref parser.NodeRef) (RRFComponent, error) {
	if ref.Kind != parser.NodeKindFunctionExpr || ref.ID < 0 || int(ref.ID) >= len(doc.FunctionExprs) {
		return RRFComponent{}, fmt.Errorf("FTS_RANK requires a function expression")
	}
	fn := &doc.FunctionExprs[ref.ID]
	if !asciiEqualFold(src[fn.NameStart:fn.NameEnd], []byte("FTS_RANK")) {
		return RRFComponent{}, fmt.Errorf("expected FTS_RANK function")
	}
	if fn.HasWindow || fn.ArgsCount != 2 || fn.ArgsStart < 0 || fn.ArgsStart+fn.ArgsCount > int32(len(doc.FunctionArgs)) {
		return RRFComponent{}, fmt.Errorf("FTS_RANK requires text column and query arguments")
	}
	columnRef := doc.FunctionArgs[fn.ArgsStart]
	if columnRef.Kind != parser.NodeKindIdentifier || columnRef.ID < 0 || int(columnRef.ID) >= len(doc.Identifiers) {
		return RRFComponent{}, fmt.Errorf("FTS_RANK first argument must be a text column")
	}
	column := &doc.Identifiers[columnRef.ID]
	component := RRFComponent{TextColumn: string(src[column.Start:column.End])}
	if column.QualEnd > column.QualStart {
		component.SourceAlias = string(src[column.QualStart:column.QualEnd])
	}
	queryRef := doc.FunctionArgs[fn.ArgsStart+1]
	switch queryRef.Kind {
	case parser.NodeKindString:
		if queryRef.ID < 0 || int(queryRef.ID) >= len(doc.Strings) {
			return RRFComponent{}, fmt.Errorf("FTS_RANK query literal is invalid")
		}
		sl := doc.Strings[queryRef.ID]
		component.TextQuery = string(decodeSQLStringLiteral(src, sl))
	case parser.NodeKindIdentifier:
		value, found := o.resolveParamScalar(doc, src, queryRef)
		if !found || value.IsNull() || (value.Kind != ScalarString && value.Kind != ScalarBytes) {
			return RRFComponent{}, fmt.Errorf("FTS_RANK query must be a text literal or bound text parameter")
		}
		component.TextQuery = string(value.BytesData)
	default:
		return RRFComponent{}, fmt.Errorf("FTS_RANK query must be a text literal or bound text parameter")
	}
	if component.TextQuery == "" {
		return RRFComponent{}, fmt.Errorf("FTS_RANK query must not be empty")
	}
	return component, nil
}

type loweredFTSValue struct {
	kind        FTSProjectionKind
	config      string
	column      string
	sourceAlias string
	text        string
	queryMode   string
}

func (o *Optimizer) lowerFTSProjection(doc *parser.QueryDoc, src []byte, ref parser.NodeRef) (FTSProjection, error) {
	if ref.Kind != parser.NodeKindFunctionExpr || ref.ID < 0 || int(ref.ID) >= len(doc.FunctionExprs) {
		return FTSProjection{}, fmt.Errorf("invalid FTS function expression")
	}
	fn := &doc.FunctionExprs[ref.ID]
	name := string(src[fn.NameStart:fn.NameEnd])
	if fn.HasWindow {
		return FTSProjection{}, fmt.Errorf("FTS function %q cannot be used as a window", name)
	}
	switch {
	case asciiEqualFold(src[fn.NameStart:fn.NameEnd], []byte("to_tsvector")):
		value, err := o.lowerFTSVectorValue(doc, src, ref)
		return FTSProjection{Kind: FTSProjectionVector, Config: value.config, Column: value.column, SourceAlias: value.sourceAlias, Query: value.text}, err
	case asciiEqualFold(src[fn.NameStart:fn.NameEnd], []byte("to_tsquery")),
		asciiEqualFold(src[fn.NameStart:fn.NameEnd], []byte("plainto_tsquery")),
		asciiEqualFold(src[fn.NameStart:fn.NameEnd], []byte("phraseto_tsquery")),
		asciiEqualFold(src[fn.NameStart:fn.NameEnd], []byte("websearch_to_tsquery")):
		value, err := o.lowerFTSQueryValue(doc, src, ref)
		return FTSProjection{Kind: FTSProjectionQuery, Config: value.config, Query: value.text, QueryMode: value.queryMode}, err
	case asciiEqualFold(src[fn.NameStart:fn.NameEnd], []byte("ts_rank")),
		asciiEqualFold(src[fn.NameStart:fn.NameEnd], []byte("ts_rank_cd")):
		if (fn.ArgsCount != 2 && fn.ArgsCount != 3) || fn.ArgsStart < 0 || fn.ArgsStart+fn.ArgsCount > int32(len(doc.FunctionArgs)) {
			return FTSProjection{}, fmt.Errorf("%s requires a tsvector and tsquery", name)
		}
		vector, err := o.lowerFTSVectorValue(doc, src, doc.FunctionArgs[fn.ArgsStart])
		if err != nil {
			return FTSProjection{}, err
		}
		query, err := o.lowerFTSQueryValue(doc, src, doc.FunctionArgs[fn.ArgsStart+1])
		if err != nil {
			return FTSProjection{}, err
		}
		normalization := uint32(0)
		if fn.ArgsCount == 3 {
			ref := doc.FunctionArgs[fn.ArgsStart+2]
			if ref.Kind != parser.NodeKindNumber || ref.ID < 0 || int(ref.ID) >= len(doc.Numbers) {
				return FTSProjection{}, fmt.Errorf("%s normalization must be a numeric literal", name)
			}
			value, err := strconv.ParseUint(string(src[doc.Numbers[ref.ID].Start:doc.Numbers[ref.ID].End]), 10, 32)
			if err != nil || value > 63 {
				return FTSProjection{}, fmt.Errorf("%s normalization must be an integer between 0 and 63", name)
			}
			normalization = uint32(value)
		}
		return FTSProjection{Kind: FTSProjectionRank, Config: vector.config, Column: vector.column, SourceAlias: vector.sourceAlias, Query: query.text, QueryMode: query.queryMode, Normalization: normalization}, nil
	default:
		return FTSProjection{}, fmt.Errorf("unsupported FTS function %q", name)
	}
}

func (o *Optimizer) lowerFTSVectorValue(doc *parser.QueryDoc, src []byte, ref parser.NodeRef) (loweredFTSValue, error) {
	if ref.Kind == parser.NodeKindFunctionExpr {
		fn := &doc.FunctionExprs[ref.ID]
		if !asciiEqualFold(src[fn.NameStart:fn.NameEnd], []byte("to_tsvector")) {
			return loweredFTSValue{}, fmt.Errorf("expected to_tsvector expression")
		}
		if fn.ArgsCount != 1 && fn.ArgsCount != 2 || fn.ArgsStart < 0 || fn.ArgsStart+fn.ArgsCount > int32(len(doc.FunctionArgs)) {
			return loweredFTSValue{}, fmt.Errorf("to_tsvector expects text or config,text")
		}
		config := "simple"
		argIndex := int32(0)
		if fn.ArgsCount == 2 {
			var err error
			config, argIndex, err = o.lowerFTSConfig(doc, src, doc.FunctionArgs[fn.ArgsStart])
			if err != nil {
				return loweredFTSValue{}, err
			}
		}
		value, err := o.lowerFTSTextOperand(doc, src, doc.FunctionArgs[fn.ArgsStart+argIndex])
		if err != nil {
			return loweredFTSValue{}, err
		}
		value.kind = FTSProjectionVector
		value.config = config
		return value, nil
	}
	value, err := o.lowerFTSTextOperand(doc, src, ref)
	value.kind = FTSProjectionVector
	value.config = "simple"
	return value, err
}

func (o *Optimizer) lowerFTSQueryValue(doc *parser.QueryDoc, src []byte, ref parser.NodeRef) (loweredFTSValue, error) {
	if ref.Kind != parser.NodeKindFunctionExpr {
		return loweredFTSValue{}, fmt.Errorf("expected tsquery expression")
	}
	fn := &doc.FunctionExprs[ref.ID]
	name := string(src[fn.NameStart:fn.NameEnd])
	if fn.ArgsCount != 1 && fn.ArgsCount != 2 || fn.ArgsStart < 0 || fn.ArgsStart+fn.ArgsCount > int32(len(doc.FunctionArgs)) {
		return loweredFTSValue{}, fmt.Errorf("%s expects query or config,query", name)
	}
	config := "simple"
	argIndex := int32(0)
	if fn.ArgsCount == 2 {
		var err error
		config, argIndex, err = o.lowerFTSConfig(doc, src, doc.FunctionArgs[fn.ArgsStart])
		if err != nil {
			return loweredFTSValue{}, err
		}
	}
	queryRef := doc.FunctionArgs[fn.ArgsStart+argIndex]
	query, err := o.lowerFTSTextOperand(doc, src, queryRef)
	if err != nil {
		return loweredFTSValue{}, fmt.Errorf("%s query: %w", name, err)
	}
	query.kind = FTSProjectionQuery
	query.config = config
	switch {
	case asciiEqualFold(src[fn.NameStart:fn.NameEnd], []byte("plainto_tsquery")):
		query.queryMode = "plain"
	case asciiEqualFold(src[fn.NameStart:fn.NameEnd], []byte("phraseto_tsquery")):
		query.queryMode = "phrase"
	case asciiEqualFold(src[fn.NameStart:fn.NameEnd], []byte("websearch_to_tsquery")):
		query.queryMode = "web"
	default:
		query.queryMode = "raw"
	}
	return query, nil
}

func (o *Optimizer) lowerFTSConfig(doc *parser.QueryDoc, src []byte, ref parser.NodeRef) (string, int32, error) {
	var config string
	if ref.Kind == parser.NodeKindString && ref.ID >= 0 && int(ref.ID) < len(doc.Strings) {
		config = string(decodeSQLStringLiteral(src, doc.Strings[ref.ID]))
	} else if ref.Kind == parser.NodeKindIdentifier {
		if value, found := o.resolveParamScalar(doc, src, ref); found && !value.IsNull() && (value.Kind == ScalarString || value.Kind == ScalarBytes) {
			config = string(value.BytesData)
		} else {
			return "", 0, fmt.Errorf("FTS configuration must be a text literal or bound text parameter")
		}
	} else {
		return "", 0, fmt.Errorf("FTS configuration must be a text literal or bound text parameter")
	}
	config = strings.ToLower(strings.TrimSpace(config))
	if config == "pg_catalog.english" {
		config = "english"
	}
	if config != "simple" && config != "english" && config != "english_stem" {
		return "", 0, fmt.Errorf("unsupported FTS configuration %q", config)
	}
	return config, 1, nil
}

func (o *Optimizer) lowerFTSTextOperand(doc *parser.QueryDoc, src []byte, ref parser.NodeRef) (loweredFTSValue, error) {
	switch ref.Kind {
	case parser.NodeKindString:
		if ref.ID < 0 || int(ref.ID) >= len(doc.Strings) {
			return loweredFTSValue{}, fmt.Errorf("invalid text literal")
		}
		return loweredFTSValue{text: string(decodeSQLStringLiteral(src, doc.Strings[ref.ID]))}, nil
	case parser.NodeKindIdentifier:
		id := &doc.Identifiers[ref.ID]
		if value, found := o.resolveParamScalar(doc, src, ref); found {
			if value.IsNull() || (value.Kind != ScalarString && value.Kind != ScalarBytes) {
				return loweredFTSValue{}, fmt.Errorf("text operand must be textual")
			}
			return loweredFTSValue{text: string(value.BytesData)}, nil
		}
		value := loweredFTSValue{column: string(src[id.Start:id.End])}
		if id.QualEnd > id.QualStart {
			value.sourceAlias = string(src[id.QualStart:id.QualEnd])
		}
		return value, nil
	default:
		return loweredFTSValue{}, fmt.Errorf("text operand must be a column, literal, or text parameter")
	}
}

func (o *Optimizer) lowerFTSPredicate(doc *parser.QueryDoc, src []byte, left, right parser.NodeRef) (FTSPredicate, error) {
	vector, err := o.lowerFTSVectorValue(doc, src, left)
	if err != nil {
		return FTSPredicate{}, err
	}
	query, err := o.lowerFTSQueryValue(doc, src, right)
	if err != nil {
		return FTSPredicate{}, err
	}
	return FTSPredicate{Config: vector.config, Column: vector.column, SourceAlias: vector.sourceAlias, Text: vector.text, Query: query.text, QueryMode: query.queryMode}, nil
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
		if vec, err := resolveVectorOperand(doc, o.src, vf.VectorB, o.boundParams); err == nil {
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
func (o *Optimizer) extractWhereMatch(doc *parser.QueryDoc, src []byte, plan *PhysicalPlan, node parser.NodeRef) error {
	switch node.Kind {
	case parser.NodeKindMatchPath:
		mp := &doc.MatchPaths[node.ID]
		plan.HasGraphTraversal = true
		var err error
		plan.GraphEdges, plan.MaxHops, err = o.extractMatchPath(doc, src, mp)
		if err != nil {
			return err
		}
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
		if err := o.extractWhereMatch(doc, src, plan, be.Left); err != nil {
			return err
		}
		if err := o.extractWhereMatch(doc, src, plan, be.Right); err != nil {
			return err
		}
	}
	return nil
}

func (o *Optimizer) extractRelationalPredicates(doc *parser.QueryDoc, src []byte, plan *PhysicalPlan, node parser.NodeRef) {
	// The legacy predicate slice is an implicit AND. Lower any expression
	// containing OR into explicit disjunctive normal form so callers can retain
	// SQL boolean semantics without interpreting the leaves as an AND list.
	if predicateTreeContainsOr(doc, node) {
		if alternatives, ok := o.lowerPredicateAlternatives(doc, src, node); ok && len(alternatives) > 0 {
			plan.PredicateAlternatives = alternatives
			plan.HasRelationalQuery = true
			if plan.Kind == QueryKindKNN && !plan.HasVectorSearch {
				plan.Kind = QueryKindRelational
			}
			return
		}
	}
	switch node.Kind {
	case parser.NodeKindBinaryExpr:
		be := &doc.BinaryExprs[node.ID]
		// AND decomposition: recurse into both sides.
		if be.Operator == uint8(lexer.KindAnd) {
			o.extractRelationalPredicates(doc, src, plan, be.Left)
			o.extractRelationalPredicates(doc, src, plan, be.Right)
			return
		}
		if be.Operator == uint8(lexer.KindFTSMatch) {
			predicate, err := o.lowerFTSPredicate(doc, src, be.Left, be.Right)
			if err != nil {
				plan.FTSError = err.Error()
				return
			}
			plan.FTSPredicates = append(plan.FTSPredicates, predicate)
			plan.HasRelationalQuery = true
			if plan.Kind == QueryKindKNN && !plan.HasVectorSearch {
				plan.Kind = QueryKindRelational
			}
			return
		}
		if be.Left.Kind != parser.NodeKindIdentifier {
			return
		}
		o.appendScalarPredicate(doc, src, plan, be.Left, be.Operator, be.Right, be.NullTest)
	case parser.NodeKindBetweenExpr:
		bw := &doc.BetweenExprs[node.ID]
		if bw.Expr.Kind != parser.NodeKindIdentifier {
			return
		}
		id := &doc.Identifiers[bw.Expr.ID]
		if id.ResolvedKind != parser.ResolvedKindColumn {
			return
		}
		o.setRelationalKind(plan)
		lower, lowerOK := o.scalarForRef(doc, src, bw.Lower)
		upper, upperOK := o.scalarForRef(doc, src, bw.Upper)
		if !lowerOK || !upperOK {
			return
		}
		alias := ""
		if id.QualEnd > id.QualStart {
			alias = string(src[id.QualStart:id.QualEnd])
		}
		col := string(src[id.Start:id.End])
		plan.Predicates = append(plan.Predicates,
			RelationalPredicate{Alias: alias, Column: col, Operator: uint8(lexer.KindGreaterThan), TypedValue: lower, Inclusive: true, Not: bw.Not},
			RelationalPredicate{Alias: alias, Column: col, Operator: uint8(lexer.KindLessThan), TypedValue: upper, Inclusive: true, Not: bw.Not})
	case parser.NodeKindInExpr:
		in := &doc.InExprs[node.ID]
		if in.Expr.Kind != parser.NodeKindIdentifier {
			return
		}
		id := &doc.Identifiers[in.Expr.ID]
		if id.ResolvedKind != parser.ResolvedKindColumn {
			return
		}
		o.setRelationalKind(plan)
		values := make([]ScalarValue, 0, in.ListCount)
		for i := int32(0); i < in.ListCount; i++ {
			if value, ok := o.scalarForRef(doc, src, doc.Nodes[in.ListStart+i]); ok {
				values = append(values, value)
			}
		}
		if len(values) == 0 {
			return
		}
		alias := ""
		if id.QualEnd > id.QualStart {
			alias = string(src[id.QualStart:id.QualEnd])
		}
		plan.Predicates = append(plan.Predicates, RelationalPredicate{
			Alias: alias, Column: string(src[id.Start:id.End]), Operator: uint8(lexer.KindEquals), InValues: values, Not: in.Not,
		})
	}
}

func predicateTreeContainsOr(doc *parser.QueryDoc, node parser.NodeRef) bool {
	if node.Kind != parser.NodeKindBinaryExpr || node.ID < 0 || int(node.ID) >= len(doc.BinaryExprs) {
		return false
	}
	be := &doc.BinaryExprs[node.ID]
	if be.Operator == uint8(lexer.KindOr) {
		return true
	}
	return predicateTreeContainsOr(doc, be.Left) || predicateTreeContainsOr(doc, be.Right)
}

// lowerPredicateAlternatives lowers scalar boolean predicates into DNF. It
// intentionally returns false for MATCH/FTS/vector expressions; those have
// separate execution semantics and must not be silently folded into a scalar
// predicate tree.
func (o *Optimizer) lowerPredicateAlternatives(doc *parser.QueryDoc, src []byte, node parser.NodeRef) (PredicateAlternatives, bool) {
	if node.Kind == parser.NodeKindBinaryExpr && node.ID >= 0 && int(node.ID) < len(doc.BinaryExprs) {
		be := &doc.BinaryExprs[node.ID]
		switch be.Operator {
		case uint8(lexer.KindOr):
			left, leftOK := o.lowerPredicateAlternatives(doc, src, be.Left)
			right, rightOK := o.lowerPredicateAlternatives(doc, src, be.Right)
			if !leftOK || !rightOK {
				return nil, false
			}
			return append(left, right...), true
		case uint8(lexer.KindAnd):
			left, leftOK := o.lowerPredicateAlternatives(doc, src, be.Left)
			right, rightOK := o.lowerPredicateAlternatives(doc, src, be.Right)
			if !leftOK || !rightOK {
				return nil, false
			}
			combined := make(PredicateAlternatives, 0, len(left)*len(right))
			for _, leftClause := range left {
				for _, rightClause := range right {
					clause := make([]RelationalPredicate, 0, len(leftClause)+len(rightClause))
					clause = append(clause, leftClause...)
					clause = append(clause, rightClause...)
					combined = append(combined, clause)
				}
			}
			return combined, true
		}
	}

	// Reuse the existing scalar lowering for a leaf. A temporary plan lets
	// BETWEEN contribute its two comparisons while keeping the public plan's
	// legacy Predicates field untouched for OR expressions.
	tmp := &PhysicalPlan{}
	o.extractRelationalPredicates(doc, src, tmp, node)
	if len(tmp.PredicateAlternatives) > 0 || len(tmp.FTSPredicates) > 0 || len(tmp.Predicates) == 0 {
		return nil, false
	}
	return PredicateAlternatives{append([]RelationalPredicate(nil), tmp.Predicates...)}, true
}

func (o *Optimizer) appendScalarPredicate(doc *parser.QueryDoc, src []byte, plan *PhysicalPlan, left parser.NodeRef, op uint8, right parser.NodeRef, nullTest uint8) {
	id := &doc.Identifiers[left.ID]
	if id.ResolvedKind != parser.ResolvedKindColumn {
		return
	}
	if plan.Kind == QueryKindKNN && !plan.HasVectorSearch {
		plan.Kind = QueryKindRelational
	}
	plan.HasRelationalQuery = true
	pred := RelationalPredicate{Column: string(src[id.Start:id.End]), Operator: op, NullTest: nullTest}
	pred.Like = op == uint8(lexer.KindLike)
	pred.ILike = op == uint8(lexer.KindILike)
	if id.QualEnd > id.QualStart {
		pred.Alias = string(src[id.QualStart:id.QualEnd])
	}
	if nullTest == parser.NullTestNone {
		if value, ok := o.scalarForRef(doc, src, right); ok {
			pred.TypedValue = value
			pred.ValueIsNull = value.IsNull()
		}
	}
	plan.Predicates = append(plan.Predicates, pred)
}

func (o *Optimizer) scalarForRef(doc *parser.QueryDoc, src []byte, ref parser.NodeRef) (ScalarValue, bool) {
	switch ref.Kind {
	case parser.NodeKindNumber:
		n := &doc.Numbers[ref.ID]
		return ScalarFromLiteralBytes(src[n.Start:n.End]), true
	case parser.NodeKindString:
		s := &doc.Strings[ref.ID]
		return BytesValue(decodeSQLStringLiteral(src, *s)), true
	case parser.NodeKindIdentifier:
		if ref.ID >= 0 && int(ref.ID) < len(doc.Identifiers) {
			id := &doc.Identifiers[ref.ID]
			if id.ResolvedKind == parser.ResolvedKindLiteral {
				return ScalarFromLiteralBytes(src[id.Start:id.End]), true
			}
		}
		return o.resolveParamScalar(doc, src, ref)
	default:
		return ScalarValue{}, false
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
		// The left side may be the original FROM alias or any alias produced
		// by an earlier join in a multi-join chain. The right side must be the
		// relation introduced by this JOIN.
		switch {
		case leftQualifier != "" && strings.EqualFold(rightQualifier, join.RightAlias):
			join.LeftAlias = leftQualifier
			join.LeftColumn, join.RightColumn = leftColumn, rightColumn
		case rightQualifier != "" && strings.EqualFold(leftQualifier, join.RightAlias):
			join.LeftAlias = rightQualifier
			join.LeftColumn, join.RightColumn = rightColumn, leftColumn
		}
		return
	}

	predicate := RelationalPredicate{Column: leftColumn, Operator: be.Operator}
	switch be.Right.Kind {
	case parser.NodeKindNumber:
		n := &doc.Numbers[be.Right.ID]
		predicate.TypedValue = ScalarFromLiteralBytes(src[n.Start:n.End])
	case parser.NodeKindString:
		s := &doc.Strings[be.Right.ID]
		predicate.TypedValue = BytesValue(decodeSQLStringLiteral(src, *s))
	case parser.NodeKindIdentifier:
		if value, found := o.resolveParamScalar(doc, src, be.Right); found {
			predicate.TypedValue = value
			predicate.ValueIsNull = value.IsNull()
		}
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
	switch lexer.Kind(op) {
	case lexer.KindL2Dist, lexer.KindIPDist, lexer.KindCosineDist:
		return true
	default:
		return false
	}
}

type vectorOperatorInfo struct {
	Name        string
	Operator    uint8
	QueryVector []float32
	TableOID    uint32
	SourceAlias string
}

// lowerVectorOperator validates and resolves a direct pgvector distance
// expression. The right operand remains typed through the existing parameter
// set; SQL text is never rewritten into a quoted vector literal.
func (o *Optimizer) lowerVectorOperator(doc *parser.QueryDoc, src []byte, ref parser.NodeRef) (vectorOperatorInfo, error) {
	if ref.Kind != parser.NodeKindBinaryExpr || ref.ID < 0 || int(ref.ID) >= len(doc.BinaryExprs) {
		return vectorOperatorInfo{}, fmt.Errorf("vector operator expression is malformed")
	}
	be := &doc.BinaryExprs[ref.ID]
	if !isPgVectorOp(be.Operator) {
		return vectorOperatorInfo{}, fmt.Errorf("unsupported vector operator %d", be.Operator)
	}
	if be.Left.Kind != parser.NodeKindIdentifier || be.Left.ID < 0 || int(be.Left.ID) >= len(doc.Identifiers) {
		return vectorOperatorInfo{}, fmt.Errorf("vector operator left operand must be a vector column")
	}
	id := &doc.Identifiers[be.Left.ID]
	if id.ResolvedKind != parser.ResolvedKindVector && id.ResolvedKind != parser.ResolvedKindColumn {
		return vectorOperatorInfo{}, fmt.Errorf("vector operator left operand must resolve to a vector column")
	}
	vector, err := resolveVectorOperand(doc, src, be.Right, o.boundParams)
	if err != nil {
		return vectorOperatorInfo{}, err
	}
	alias := ""
	if id.QualEnd > id.QualStart {
		alias = string(src[id.QualStart:id.QualEnd])
	}
	return vectorOperatorInfo{
		Operator:    be.Operator,
		QueryVector: vector,
		TableOID:    id.TableOID,
		SourceAlias: alias,
	}, nil
}

// parseVectorLiteral extracts a []float32 from a string literal in the AST.
func parseVectorLiteral(doc *parser.QueryDoc, src []byte, stringID int32) []float32 {
	sl := &doc.Strings[stringID]
	val := string(decodeSQLStringLiteral(src, *sl))
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
	if stmt.HasSelect {
		if stmt.SelectStart >= stmt.SelectEnd || stmt.SelectEnd > uint32(len(src)) {
			return nil, fmt.Errorf("invalid INSERT ... SELECT source span")
		}
		plan.InsertSelectSQL = string(src[stmt.SelectStart:stmt.SelectEnd])
		if stmt.Select.Kind == parser.NodeKindSelectStmt && stmt.Select.ID >= 0 && int(stmt.Select.ID) < len(doc.SelectStmts) {
			selectStmt := doc.SelectStmts[stmt.Select.ID]
			for i := int32(0); i < selectStmt.ProjectionsCount; i++ {
				projection := doc.Projections[selectStmt.ProjectionsStart+i]
				if projection.Expr.Kind == parser.NodeKindIdentifier && projection.Expr.ID >= 0 && int(projection.Expr.ID) < len(doc.Identifiers) {
					id := doc.Identifiers[projection.Expr.ID]
					plan.InsertSelectColumns = append(plan.InsertSelectColumns, string(src[id.Start:id.End]))
				} else if projection.AliasEnd > projection.Alias {
					plan.InsertSelectColumns = append(plan.InsertSelectColumns, string(src[projection.Alias:projection.AliasEnd]))
				}
			}
		}
	}
	if !stmt.HasSelect {
		for _, val := range stmt.Values {
			literal, isNull, ok := o.lowerDMLLiteral(doc, src, val)
			if !ok {
				continue
			}
			plan.InsertValues = append(plan.InsertValues, literal)
			plan.InsertValueNull = append(plan.InsertValueNull, isNull)
		}
	}
	plan.InsertConflictAction = stmt.ConflictAction
	if stmt.ConflictConstraintEnd > stmt.ConflictConstraintStart {
		plan.InsertConflictConstraint = string(src[stmt.ConflictConstraintStart:stmt.ConflictConstraintEnd])
	}
	for _, ref := range stmt.ConflictColumns {
		if ref.Kind != parser.NodeKindIdentifier || ref.ID < 0 || int(ref.ID) >= len(doc.Identifiers) {
			return nil, fmt.Errorf("invalid ON CONFLICT target column")
		}
		id := &doc.Identifiers[ref.ID]
		plan.InsertConflictColumns = append(plan.InsertConflictColumns, string(src[id.Start:id.End]))
	}
	for _, assignment := range stmt.ConflictSet {
		if assignment.Column.Kind != parser.NodeKindIdentifier || assignment.Column.ID < 0 || int(assignment.Column.ID) >= len(doc.Identifiers) {
			return nil, fmt.Errorf("invalid ON CONFLICT assignment column")
		}
		column := &doc.Identifiers[assignment.Column.ID]
		plan.InsertConflictSetColumns = append(plan.InsertConflictSetColumns, string(src[column.Start:column.End]))
		root, err := o.lowerConflictExpr(doc, src, assignment.Value, &plan.InsertConflictExprs, &plan.InsertConflictCases)
		if err != nil {
			return nil, fmt.Errorf("invalid ON CONFLICT value for column %q: %w", string(src[column.Start:column.End]), err)
		}
		plan.InsertConflictExprRoots = append(plan.InsertConflictExprRoots, root)
	}
	if stmt.HasConflictWhere {
		root, err := o.lowerConflictExpr(doc, src, stmt.ConflictWhere, &plan.InsertConflictExprs, &plan.InsertConflictCases)
		if err != nil {
			return nil, fmt.Errorf("invalid ON CONFLICT WHERE: %w", err)
		}
		plan.InsertConflictWhereRoot = root
		plan.InsertConflictHasWhere = true
	}
	if err := lowerReturning(stmt.Returning, stmt.ReturningStar, doc, src, &plan.ReturningColumns, &plan.ReturningStar); err != nil {
		return nil, err
	}
	return plan, nil
}

func lowerReturning(refs []parser.NodeRef, star bool, doc *parser.QueryDoc, src []byte, columns *[]string, outStar *bool) error {
	if outStar != nil {
		*outStar = star
	}
	if columns == nil {
		return fmt.Errorf("RETURNING projection destination is nil")
	}
	for _, ref := range refs {
		if ref.Kind != parser.NodeKindIdentifier || ref.ID < 0 || int(ref.ID) >= len(doc.Identifiers) {
			return fmt.Errorf("RETURNING supports column identifiers only")
		}
		id := doc.Identifiers[ref.ID]
		if id.End <= id.Start || id.End > uint32(len(src)) {
			return fmt.Errorf("invalid RETURNING column span")
		}
		name := string(src[id.Start:id.End])
		if dot := strings.LastIndexByte(name, '.'); dot >= 0 {
			name = name[dot+1:]
		}
		*columns = append(*columns, name)
	}
	if star && len(*columns) != 0 {
		return fmt.Errorf("RETURNING cannot combine '*' with explicit columns")
	}
	return nil
}

func (o *Optimizer) lowerConflictExpr(doc *parser.QueryDoc, src []byte, ref parser.NodeRef, out *[]ConflictExpr, cases *[]ConflictCase) (int32, error) {
	if ref.ID < 0 {
		return -1, fmt.Errorf("invalid expression reference")
	}
	switch ref.Kind {
	case parser.NodeKindString:
		if int(ref.ID) >= len(doc.Strings) {
			return -1, fmt.Errorf("invalid string literal")
		}
		s := doc.Strings[ref.ID]
		root := int32(len(*out))
		*out = append(*out, ConflictExpr{Kind: ConflictExprLiteral, Literal: append([]byte(nil), decodeSQLStringLiteral(src, s)...)})
		return root, nil
	case parser.NodeKindNumber:
		if int(ref.ID) >= len(doc.Numbers) {
			return -1, fmt.Errorf("invalid numeric literal")
		}
		n := doc.Numbers[ref.ID]
		root := int32(len(*out))
		*out = append(*out, ConflictExpr{Kind: ConflictExprLiteral, Literal: append([]byte(nil), src[n.Start:n.End]...)})
		return root, nil
	case parser.NodeKindIdentifier:
		if int(ref.ID) >= len(doc.Identifiers) {
			return -1, fmt.Errorf("invalid identifier")
		}
		id := doc.Identifiers[ref.ID]
		if bytesEqualFold(src[id.Start:id.End], []byte("NULL")) {
			root := int32(len(*out))
			*out = append(*out, ConflictExpr{Kind: ConflictExprLiteral, IsNull: true})
			return root, nil
		}
		if id.ResolvedKind == parser.ResolvedKindLiteral {
			root := int32(len(*out))
			*out = append(*out, ConflictExpr{Kind: ConflictExprLiteral, Literal: append([]byte(nil), src[id.Start:id.End]...)})
			return root, nil
		}
		if id.ResolvedKind == parser.ResolvedKindExcluded {
			root := int32(len(*out))
			*out = append(*out, ConflictExpr{Kind: ConflictExprExcludedColumn, Column: string(src[id.Start:id.End])})
			return root, nil
		}
		if id.Start < uint32(len(src)) && (src[id.Start] == '$' || src[id.Start] == '@') {
			value, isNull, found := o.resolveParamValueState(doc, src, ref)
			if !found {
				return -1, fmt.Errorf("parameter %q is missing", string(src[id.Start:id.End]))
			}
			root := int32(len(*out))
			*out = append(*out, ConflictExpr{Kind: ConflictExprLiteral, Literal: append([]byte(nil), value...), IsNull: isNull})
			return root, nil
		}
		root := int32(len(*out))
		*out = append(*out, ConflictExpr{Kind: ConflictExprColumn, Column: string(src[id.Start:id.End])})
		return root, nil
	case parser.NodeKindBinaryExpr:
		if int(ref.ID) >= len(doc.BinaryExprs) {
			return -1, fmt.Errorf("invalid binary expression")
		}
		be := doc.BinaryExprs[ref.ID]
		left, err := o.lowerConflictExpr(doc, src, be.Left, out, cases)
		if err != nil {
			return -1, err
		}
		right, err := o.lowerConflictExpr(doc, src, be.Right, out, cases)
		if err != nil {
			return -1, err
		}
		root := int32(len(*out))
		*out = append(*out, ConflictExpr{Kind: ConflictExprBinary, Operator: be.Operator, Left: left, Right: right})
		return root, nil
	case parser.NodeKindUnaryExpr:
		if int(ref.ID) >= len(doc.UnaryExprs) {
			return -1, fmt.Errorf("invalid unary expression")
		}
		un := doc.UnaryExprs[ref.ID]
		if lexer.Kind(un.Operator) != lexer.KindNot {
			return -1, fmt.Errorf("unsupported unary operator %d", un.Operator)
		}
		child, err := o.lowerConflictExpr(doc, src, un.Expr, out, cases)
		if err != nil {
			return -1, err
		}
		root := int32(len(*out))
		*out = append(*out, ConflictExpr{Kind: ConflictExprUnary, Operator: un.Operator, Left: child})
		return root, nil
	case parser.NodeKindCaseExpr:
		if int(ref.ID) >= len(doc.CaseExprs) {
			return -1, fmt.Errorf("invalid CASE expression")
		}
		ce := doc.CaseExprs[ref.ID]
		expr := ConflictExpr{Kind: ConflictExprCase, CaseWhenStart: int32(len(*cases)), CaseWhenCount: ce.WhensCount, CaseElse: -1}
		for i := int32(0); i < ce.WhensCount; i++ {
			when := doc.CaseWhens[ce.WhensStart+i]
			condition, err := o.lowerConflictExpr(doc, src, when.Condition, out, cases)
			if err != nil {
				return -1, err
			}
			value, err := o.lowerConflictExpr(doc, src, when.Value, out, cases)
			if err != nil {
				return -1, err
			}
			*cases = append(*cases, ConflictCase{Condition: condition, Value: value})
		}
		if ce.HasElse {
			value, err := o.lowerConflictExpr(doc, src, ce.Else, out, cases)
			if err != nil {
				return -1, err
			}
			expr.CaseElse = value
		}
		root := int32(len(*out))
		*out = append(*out, expr)
		return root, nil
	case parser.NodeKindFunctionExpr:
		if int(ref.ID) >= len(doc.FunctionExprs) {
			return -1, fmt.Errorf("invalid function expression")
		}
		fn := doc.FunctionExprs[ref.ID]
		name := string(src[fn.NameStart:fn.NameEnd])
		if !strings.EqualFold(name, "NOW") && !strings.EqualFold(name, "NULLIF") {
			return -1, fmt.Errorf("unsupported ON CONFLICT function %q", name)
		}
		expr := ConflictExpr{Kind: ConflictExprFunction, Function: name, Left: -1, Right: -1}
		if fn.ArgsCount > 0 {
			if fn.ArgsCount != 2 || !strings.EqualFold(name, "NULLIF") {
				return -1, fmt.Errorf("function %q requires exactly two arguments", name)
			}
			left, err := o.lowerConflictExpr(doc, src, doc.FunctionArgs[fn.ArgsStart], out, cases)
			if err != nil {
				return -1, err
			}
			right, err := o.lowerConflictExpr(doc, src, doc.FunctionArgs[fn.ArgsStart+1], out, cases)
			if err != nil {
				return -1, err
			}
			expr.Left, expr.Right = left, right
		}
		root := int32(len(*out))
		*out = append(*out, expr)
		return root, nil
	case parser.NodeKindCastExpr:
		if int(ref.ID) >= len(doc.CastExprs) {
			return -1, fmt.Errorf("invalid cast expression")
		}
		cast := doc.CastExprs[ref.ID]
		child, err := o.lowerConflictExpr(doc, src, cast.Expr, out, cases)
		if err != nil {
			return -1, err
		}
		root := int32(len(*out))
		*out = append(*out, ConflictExpr{Kind: ConflictExprCast, Left: child, Type: string(src[cast.TypeStart:cast.TypeEnd])})
		return root, nil
	default:
		return -1, fmt.Errorf("expression kind %d is unsupported", ref.Kind)
	}
}

func bytesEqualFold(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		x, y := a[i], b[i]
		if x >= 'A' && x <= 'Z' {
			x += 'a' - 'A'
		}
		if y >= 'A' && y <= 'Z' {
			y += 'a' - 'A'
		}
		if x != y {
			return false
		}
	}
	return true
}

// lowerDMLLiteral preserves the distinction between SQL NULL and a quoted
// string containing the four letters "NULL". The latter is returned as
// ordinary bytes; the former is represented by a nil byte slice plus isNull.
func (o *Optimizer) lowerDMLLiteral(doc *parser.QueryDoc, src []byte, ref parser.NodeRef) ([]byte, bool, bool) {
	switch ref.Kind {
	case parser.NodeKindString:
		if ref.ID < 0 || int(ref.ID) >= len(doc.Strings) {
			return nil, false, false
		}
		sl := &doc.Strings[ref.ID]
		return decodeSQLStringLiteral(src, *sl), false, true
	case parser.NodeKindNumber:
		if ref.ID < 0 || int(ref.ID) >= len(doc.Numbers) {
			return nil, false, false
		}
		num := &doc.Numbers[ref.ID]
		return src[num.Start:num.End], false, true
	case parser.NodeKindIdentifier:
		if ref.ID < 0 || int(ref.ID) >= len(doc.Identifiers) {
			return nil, false, false
		}
		id := &doc.Identifiers[ref.ID]
		raw := src[id.Start:id.End]
		if len(raw) > 0 && (raw[0] == '$' || raw[0] == '@') {
			value, isNull, found := o.resolveParamValueState(doc, src, ref)
			if found {
				return value, isNull, true
			}
			return nil, false, false
		}
		if id.ResolvedKind != parser.ResolvedKindLiteral {
			return nil, false, false
		}
		if len(raw) == 4 && (raw[0] == 'N' || raw[0] == 'n') &&
			(raw[1] == 'U' || raw[1] == 'u') &&
			(raw[2] == 'L' || raw[2] == 'l') &&
			(raw[3] == 'L' || raw[3] == 'l') {
			return nil, true, true
		}
		return raw, false, true
	default:
		return nil, false, false
	}
}

func (o *Optimizer) optimizeInsertGraphEdge(doc *parser.QueryDoc, src []byte) (*PhysicalPlan, error) {
	// Take the first statement; batch support can be added later.
	stmt := &doc.InsertGraphEdgeStmts[0]
	plan := &PhysicalPlan{
		Kind: QueryKindInsertGraphEdge,
	}
	refs := []parser.NodeRef{stmt.SrcExpr, stmt.EdgeKindExpr, stmt.TgtExpr}
	legacy := [][2]uint32{
		{stmt.SrcStart, stmt.SrcEnd},
		{stmt.EdgeKindStart, stmt.EdgeKindEnd},
		{stmt.TgtStart, stmt.TgtEnd},
	}
	for i, ref := range refs {
		value, isNull, ok := o.lowerDMLLiteral(doc, src, ref)
		if !ok {
			if legacy[i][1] <= legacy[i][0] || legacy[i][1] > uint32(len(src)) {
				return nil, fmt.Errorf("GRAPH_EDGES value %d is missing or unsupported", i+1)
			}
			value = src[legacy[i][0]:legacy[i][1]]
		}
		if isNull {
			return nil, fmt.Errorf("GRAPH_EDGES value %d cannot be NULL", i+1)
		}
		plan.InsertValues = append(plan.InsertValues, value)
	}
	if stmt.HasProperties {
		value, isNull, ok := o.lowerDMLLiteral(doc, src, stmt.PropertiesExpr)
		if !ok {
			if stmt.PropertiesEnd <= stmt.PropertiesStart || stmt.PropertiesEnd > uint32(len(src)) {
				return nil, fmt.Errorf("GRAPH_EDGES properties are missing or unsupported")
			}
			value = src[stmt.PropertiesStart:stmt.PropertiesEnd]
		}
		if isNull {
			return nil, fmt.Errorf("GRAPH_EDGES properties cannot be NULL")
		}
		plan.InsertValues = append(plan.InsertValues, value)
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
		root, exprErr := o.lowerConflictExpr(doc, src, val, &plan.SetExprs, &plan.SetExprCases)
		if exprErr == nil {
			plan.SetExprRoots = append(plan.SetExprRoots, root)
		} else {
			// Keep the legacy literal path for callers using a value kind that
			// has no expression lowering yet; malformed expressions must still
			// be rejected rather than silently dropped.
			literal, isNull, ok := o.lowerDMLLiteral(doc, src, val)
			if !ok {
				return nil, fmt.Errorf("UPDATE SET expression: %w", exprErr)
			}
			plan.SetValues = append(plan.SetValues, literal)
			plan.SetValueNull = append(plan.SetValueNull, isNull)
			plan.SetExprRoots = append(plan.SetExprRoots, -1)
			continue
		}
		literal, isNull, ok := o.lowerDMLLiteral(doc, src, val)
		if ok {
			plan.SetValues = append(plan.SetValues, literal)
			plan.SetValueNull = append(plan.SetValueNull, isNull)
		}
	}
	// Extract WHERE predicates for ID resolution
	if stmt.WhereExpr.Kind != parser.NodeKindUnknown {
		o.extractRelationalPredicates(doc, src, plan, stmt.WhereExpr)
	}
	if err := lowerReturning(stmt.Returning, stmt.ReturningStar, doc, src, &plan.ReturningColumns, &plan.ReturningStar); err != nil {
		return nil, err
	}
	return plan, nil
}

func (o *Optimizer) optimizeDelete(doc *parser.QueryDoc, src []byte) (*PhysicalPlan, error) {
	stmt := &doc.DeleteStmts[0]
	plan := &PhysicalPlan{
		Kind:           QueryKindDelete,
		CollectionName: string(src[stmt.TableStart:stmt.TableEnd]),
	}
	plan.GraphEdgeDelete = strings.EqualFold(plan.CollectionName, "GRAPH_EDGES")
	if stmt.WhereExpr.Kind != parser.NodeKindUnknown {
		o.extractRelationalPredicates(doc, src, plan, stmt.WhereExpr)
	}
	if err := lowerReturning(stmt.Returning, stmt.ReturningStar, doc, src, &plan.ReturningColumns, &plan.ReturningStar); err != nil {
		return nil, err
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
		flags := col.Flags
		if col.HasIdentity {
			flags |= catalog.ColFlagHasDefault
		}
		vectorDimension := col.TypeParam
		typeName := strings.ToUpper(strings.TrimSpace(string(src[col.TypeStart:col.TypeEnd])))
		if typeEnd := strings.IndexByte(typeName, '('); typeEnd >= 0 {
			typeName = strings.TrimSpace(typeName[:typeEnd])
		}
		if typeName != "VECTOR" {
			// TypeParam is also populated for scalar declarations such as
			// VARCHAR(255). It is a vector dimension only for VECTOR(n).
			vectorDimension = 0
		}
		plan.DDLColumns = append(plan.DDLColumns, struct {
			Name            string
			Type            string
			VectorDimension uint32
			Flags           uint16
		}{
			Name:            string(src[col.NameStart:col.NameEnd]),
			Type:            string(src[col.TypeStart:col.TypeEnd]),
			VectorDimension: vectorDimension,
			Flags:           flags,
		})
	}
	if stmt.PrimaryKey != nil {
		if stmt.PrimaryKey.NameEnd > stmt.PrimaryKey.NameStart {
			plan.DDLPrimaryKeyConstraint = string(src[stmt.PrimaryKey.NameStart:stmt.PrimaryKey.NameEnd])
		}
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
	// Lower CHECK constraints from AST to plan.
	for _, chk := range stmt.CheckConstraints {
		ddlChk := DDLCheckConstraint{
			Expression: string(src[chk.ExprStart:chk.ExprEnd]),
			ColumnName: chk.ColumnName,
		}
		if chk.NameStart != 0 {
			ddlChk.Name = string(src[chk.NameStart:chk.NameEnd])
		}
		plan.DDLCheckConstraints = append(plan.DDLCheckConstraints, ddlChk)
	}
	// Lower DEFAULT expressions from AST to plan.
	for _, col := range stmt.Columns {
		if !col.HasDefault {
			continue
		}
		if plan.DDLColumnDefaults == nil {
			plan.DDLColumnDefaults = make(map[string]string)
		}
		colName := string(src[col.NameStart:col.NameEnd])
		plan.DDLColumnDefaults[colName] = extractDefaultValue(doc, src, col.DefaultExpr)
	}
	return plan, nil
}

// extractDefaultValue resolves a DEFAULT NodeRef to its literal string value
// by looking up the actual byte offsets in the doc's SoA slices.
func extractDefaultValue(doc *parser.QueryDoc, src []byte, ref parser.NodeRef) string {
	switch ref.Kind {
	case parser.NodeKindString:
		if ref.ID >= 0 && int(ref.ID) < len(doc.Strings) {
			sl := &doc.Strings[ref.ID]
			return string(decodeSQLStringLiteral(src, *sl))
		}
	case parser.NodeKindNumber:
		if ref.ID >= 0 && int(ref.ID) < len(doc.Numbers) {
			num := &doc.Numbers[ref.ID]
			return string(src[num.Start:num.End])
		}
	case parser.NodeKindIdentifier:
		if ref.ID >= 0 && int(ref.ID) < len(doc.Identifiers) {
			id := &doc.Identifiers[ref.ID]
			return string(src[id.Start:id.End]) // TRUE, FALSE, or NULL
		}
	}
	return ""
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
	plan := &PhysicalPlan{
		Kind:           QueryKindDDL,
		DDLKind:        2,
		DDLTableName:   string(src[stmt.TableStart:stmt.TableEnd]),
		DDLIndexName:   string(src[stmt.IndexStart:stmt.IndexEnd]),
		DDLColName:     string(src[stmt.ColStart:stmt.ColEnd]),
		DDLUnique:      stmt.Unique,
		CollectionName: string(src[stmt.TableStart:stmt.TableEnd]),
	}
	for _, column := range stmt.Columns {
		plan.DDLIndexColumns = append(plan.DDLIndexColumns, string(src[column.Start:column.End]))
	}
	if stmt.JSONPathEnd > stmt.JSONPathStart {
		path := string(src[stmt.JSONPathStart:stmt.JSONPathEnd])
		if len(path) >= 2 && ((path[0] == '\'' && path[len(path)-1] == '\'') || (path[0] == '"' && path[len(path)-1] == '"')) {
			path = path[1 : len(path)-1]
		}
		plan.DDLJSONPath = path
		plan.DDLJSONText = stmt.JSONPathOperator == uint8(lexer.KindJSONPathText)
	}
	if len(plan.DDLIndexColumns) == 0 && stmt.ColEnd > stmt.ColStart {
		plan.DDLIndexColumns = append(plan.DDLIndexColumns, string(src[stmt.ColStart:stmt.ColEnd]))
	}
	return plan, nil
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
	if stmt.DropColumn {
		return &PhysicalPlan{
			Kind:              QueryKindDDL,
			DDLKind:           4,
			DDLTableName:      string(src[stmt.TableStart:stmt.TableEnd]),
			DDLDropColumn:     true,
			DDLDropColumnName: string(src[stmt.DropColumnStart:stmt.DropColumnEnd]),
			CollectionName:    string(src[stmt.TableStart:stmt.TableEnd]),
		}, nil
	}
	vectorDimension := stmt.AddColumn.TypeParam
	typeName := strings.ToUpper(strings.TrimSpace(string(src[stmt.AddColumn.TypeStart:stmt.AddColumn.TypeEnd])))
	if typeEnd := strings.IndexByte(typeName, '('); typeEnd >= 0 {
		typeName = strings.TrimSpace(typeName[:typeEnd])
	}
	if typeName != "VECTOR" {
		vectorDimension = 0
	}
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
			VectorDimension: vectorDimension,
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

// resolveParamValueState preserves the distinction between an absent
// parameter and a present parameter whose native value is SQL NULL. The
// pgwire adapter uses the exact $N/@name key; the unprefixed fallback keeps
// the public QueryWithParams API compatible with its existing vector callers.
func (o *Optimizer) resolveParamValueState(doc *parser.QueryDoc, src []byte, ref parser.NodeRef) ([]byte, bool, bool) {
	if ref.Kind != parser.NodeKindIdentifier || ref.ID < 0 || int(ref.ID) >= len(doc.Identifiers) {
		return nil, false, false
	}
	id := &doc.Identifiers[ref.ID]
	if id.Start >= uint32(len(src)) || (src[id.Start] != '$' && src[id.Start] != '@') {
		return nil, false, false
	}
	if o.boundParams != nil {
		value, ok := o.boundParams.Lookup(src, id.Start, id.End)
		if ok {
			if value.IsNull() {
				return nil, true, true
			}
			return value.Bytes(), false, true
		}
	}
	// Compatibility-only fallback for direct Go callers that still provide a
	// map. Pgwire never reaches this branch.
	if o.params == nil {
		return nil, false, false
	}
	name := string(src[id.Start:id.End])
	base := name[1:]
	for _, key := range []string{name, base} {
		if val, ok := o.params[key]; ok {
			if val == nil {
				return nil, true, true
			}
			return valueToParamBytes(val), false, true
		}
	}
	return nil, false, false
}

// resolveParamScalar resolves a parameter without converting it through SQL
// source text. It is used by predicate planning; DML lowering retains the
// byte-oriented storage representation separately.
func (o *Optimizer) resolveParamScalar(doc *parser.QueryDoc, src []byte, ref parser.NodeRef) (ScalarValue, bool) {
	if ref.Kind != parser.NodeKindIdentifier || ref.ID < 0 || int(ref.ID) >= len(doc.Identifiers) {
		return ScalarValue{}, false
	}
	id := &doc.Identifiers[ref.ID]
	if id.Start >= uint32(len(src)) || id.End > uint32(len(src)) || (src[id.Start] != '$' && src[id.Start] != '@') {
		return ScalarValue{}, false
	}
	if o.boundParams != nil {
		if value, found := o.boundParams.Lookup(src, id.Start, id.End); found {
			return value, true
		}
	}
	if o.params == nil {
		return ScalarValue{}, false
	}
	// Compatibility-only map lookup. The native path above handles aliases
	// without allocating; this preserves existing Go QueryWithParams callers.
	name := string(src[id.Start:id.End])
	base := name[1:]
	for _, key := range []string{name, base} {
		if value, found := o.params[key]; found {
			return ScalarFromInterface(value), true
		}
	}
	return ScalarValue{}, false
}

// valueToParamBytes converts a typed QueryParams value into a []byte
// representation suitable for RelationalPredicate.Value.
func valueToParamBytes(v interface{}) []byte {
	return ScalarFromInterface(v).Bytes()
}
