package pgwire

import "testing"

func TestNamedAtParameterProtocolHelpers(t *testing.T) {
	query := "SELECT VECTOR_DISTANCE(embedding, @query_vec) FROM documents"
	if got := countParams(query); got != 1 {
		t.Fatalf("countParams=%d, want 1", got)
	}
	info := analyzeParams(query)
	portal := &Portal{
		Stmt:   &PreparedStmt{numPositional: info.numPositional, namedOrder: info.namedOrder},
		Params: []ParamValue{{Value: []float32{1, 0, 0}}},
	}
	params := buildQueryParams(portal)
	vector, ok := params["query_vec"].([]float32)
	if !ok || len(vector) != 3 || vector[0] != 1 {
		t.Fatalf("native named parameter=%#v, want []float32{1, 0, 0}", params["query_vec"])
	}
}
