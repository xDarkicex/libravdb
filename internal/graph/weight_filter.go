package graph

// WeightOp is the semantic comparison used by an edge-local weight filter.
// It deliberately does not reuse lexer token values so the graph package is
// independent of the SQL parser.
type WeightOp uint8

const (
	WeightEqual WeightOp = iota + 1
	WeightNotEqual
	WeightLess
	WeightLessEqual
	WeightGreater
	WeightGreaterEqual
)

// WeightFilter is an optional predicate applied while graph pages are being
// enumerated. The zero value means that every edge weight is accepted.
type WeightFilter struct {
	Enabled bool
	Op      WeightOp
	Value   float32
}

func (f WeightFilter) Matches(weight float32) bool {
	if !f.Enabled {
		return true
	}
	switch f.Op {
	case WeightEqual:
		return weight == f.Value
	case WeightNotEqual:
		return weight != f.Value
	case WeightLess:
		return weight < f.Value
	case WeightLessEqual:
		return weight <= f.Value
	case WeightGreater:
		return weight > f.Value
	case WeightGreaterEqual:
		return weight >= f.Value
	default:
		return false
	}
}
