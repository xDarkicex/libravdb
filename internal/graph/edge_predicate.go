package graph

// EdgePredicateOp identifies a boolean node in an edge-property predicate.
type EdgePredicateOp uint8

const (
	EdgePredicateComparison EdgePredicateOp = iota + 1
	EdgePredicateAnd
	EdgePredicateOr
)

// EdgePredicateProperty identifies one of the edge fields that is currently
// exposed to MATCH predicates. These are physical fields on Edge, not a
// second property store.
type EdgePredicateProperty uint8

const (
	EdgePropertyWeight EdgePredicateProperty = iota + 1
	EdgePropertyKind
	EdgePropertyArbitrary
)

// EdgePredicateNode is a compact, plan-time representation of a boolean edge
// predicate. Comparison values are resolved before traversal; the graph hot
// loop only reads the persisted Edge fields.
type EdgePredicateNode struct {
	Op       EdgePredicateOp
	Property EdgePredicateProperty
	Compare  WeightOp
	Weight   float32
	Kind     uint8
	Name     string
	Value    EdgePropertyValue
	Left     int32
	Right    int32
}

// EdgePredicate is an arena-backed boolean expression used when the legacy
// single KindSet/WeightFilter fast path cannot represent the predicate (for
// example, type = 'A' OR type = 'B'). A zero value accepts every edge.
type EdgePredicate struct {
	Nodes []EdgePredicateNode
	Root  int32
}

// Enabled reports whether the predicate contains a valid root node.
func (p EdgePredicate) Enabled() bool {
	return len(p.Nodes) > 0 && p.Root >= 0 && int(p.Root) < len(p.Nodes)
}

// Matches evaluates the predicate against a physical graph edge.
func (p EdgePredicate) Matches(edge Edge) bool {
	return p.MatchesWithProperties(edge, nil)
}

// MatchesWithProperties evaluates a predicate against the physical edge and
// its page-owned property envelope. Missing fields and JSON null follow SQL
// NULL comparison semantics and do not match.
func (p EdgePredicate) MatchesWithProperties(edge Edge, properties []byte) bool {
	if !p.Enabled() {
		return true
	}
	return p.matchesNode(p.Root, edge, properties)
}

func (p EdgePredicate) matchesNode(index int32, edge Edge, properties []byte) bool {
	if index < 0 || int(index) >= len(p.Nodes) {
		return false
	}
	n := p.Nodes[index]
	switch n.Op {
	case EdgePredicateComparison:
		switch n.Property {
		case EdgePropertyWeight:
			return matchWeight(n.Compare, edge.Weight, n.Weight)
		case EdgePropertyKind:
			return matchKind(n.Compare, edge.GetKind(), n.Kind)
		case EdgePropertyArbitrary:
			actual, ok := findEdgeProperty(properties, n.Name)
			return ok && matchPropertyValue(n.Compare, actual, n.Value)
		default:
			return false
		}
	case EdgePredicateAnd:
		return p.matchesNode(n.Left, edge, properties) && p.matchesNode(n.Right, edge, properties)
	case EdgePredicateOr:
		return p.matchesNode(n.Left, edge, properties) || p.matchesNode(n.Right, edge, properties)
	default:
		return false
	}
}

func matchPropertyValue(op WeightOp, actual, expected EdgePropertyValue) bool {
	if actual.Kind == EdgePropertyNull || expected.Kind == EdgePropertyNull || actual.Kind != expected.Kind {
		return false
	}
	switch actual.Kind {
	case EdgePropertyNumber:
		return matchFloat(op, actual.Number, expected.Number)
	case EdgePropertyString:
		return matchString(op, actual.String, expected.String)
	case EdgePropertyBool:
		switch op {
		case WeightEqual:
			return actual.Bool == expected.Bool
		case WeightNotEqual:
			return actual.Bool != expected.Bool
		default:
			return false
		}
	default:
		return false
	}
}

func matchFloat(op WeightOp, actual, expected float64) bool {
	switch op {
	case WeightEqual:
		return actual == expected
	case WeightNotEqual:
		return actual != expected
	case WeightLess:
		return actual < expected
	case WeightLessEqual:
		return actual <= expected
	case WeightGreater:
		return actual > expected
	case WeightGreaterEqual:
		return actual >= expected
	default:
		return false
	}
}

func matchString(op WeightOp, actual, expected string) bool {
	switch op {
	case WeightEqual:
		return actual == expected
	case WeightNotEqual:
		return actual != expected
	case WeightLess:
		return actual < expected
	case WeightLessEqual:
		return actual <= expected
	case WeightGreater:
		return actual > expected
	case WeightGreaterEqual:
		return actual >= expected
	default:
		return false
	}
}

func matchWeight(op WeightOp, actual, expected float32) bool {
	switch op {
	case WeightEqual:
		return actual == expected
	case WeightNotEqual:
		return actual != expected
	case WeightLess:
		return actual < expected
	case WeightLessEqual:
		return actual <= expected
	case WeightGreater:
		return actual > expected
	case WeightGreaterEqual:
		return actual >= expected
	default:
		return false
	}
}

func matchKind(op WeightOp, actual, expected uint8) bool {
	switch op {
	case WeightEqual:
		return actual == expected
	case WeightNotEqual:
		return actual != expected
	case WeightLess:
		return actual < expected
	case WeightLessEqual:
		return actual <= expected
	case WeightGreater:
		return actual > expected
	case WeightGreaterEqual:
		return actual >= expected
	default:
		return false
	}
}
