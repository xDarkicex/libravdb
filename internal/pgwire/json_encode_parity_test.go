package pgwire

import (
	"testing"

	apexjson "github.com/xDarkicex/apexJSON/v2"
	"github.com/xDarkicex/libravdb/internal/util"
	"github.com/xDarkicex/libravdb/libravdb"
)

// TestPGWire_MetadataValueToString_ByteParity checks that the pooled JSON
// encoder in metadataValueToString produces exactly the bytes the direct
// encoder produced.
//
// Every JSON-shaped value the wire layer can hand a client goes through
// metadataValueToString, and the string it returns is what the client sees.
// The pooled path has to be byte-for-byte the same as encoding the value
// directly, or a client's view of the data changes without the data changing.
func TestPGWire_MetadataValueToString_ByteParity(t *testing.T) {
	stats := libravdb.SQLQueryStats{
		Queries:             7,
		Errors:              1,
		PlanCacheHits:       3,
		PlanCacheMisses:     2,
		TotalExecutionNanos: 4242,
		LastExecutionNanos:  21,
		RowsReturned:        9,
		RowsExamined:        11,
		GraphExpansions:     4,
		IndexHits:           5,
	}
	plan := libravdb.SQLExplainPlan{
		Strategy:            "index scan",
		Anchor:              "n1",
		ActualRows:          3,
		GraphExpansions:     1,
		PredicateRejections: 2,
		IndexHits:           4,
		ExecutionTimeNanos:  99,
		PlanReused:          true,
	}
	path := libravdb.GraphPath{
		Nodes:       []string{"n1", "n2"},
		EdgeTypes:   []string{"FOLLOWS"},
		EdgeWeights: []float32{1.5},
	}

	testCases := []struct {
		name string
		in   interface{}
		// want is the value the old direct-encoder path produced.
		want interface{}
	}{
		{"map", map[string]interface{}{"a": 1, "b": "two", "c": true}, map[string]interface{}{"a": 1, "b": "two", "c": true}},
		{"map-nested", map[string]interface{}{"a": map[string]interface{}{"b": []interface{}{1, 2}}}, map[string]interface{}{"a": map[string]interface{}{"b": []interface{}{1, 2}}}},
		{"map-json-null", map[string]interface{}{"a": util.JSONNull{}}, map[string]interface{}{"a": nil}},
		{"slice", []interface{}{"a", 1, true, nil}, []interface{}{"a", 1, true, nil}},
		{"slice-json-null", []interface{}{util.JSONNull{}, "b"}, []interface{}{nil, "b"}},
		{"map-string", map[string]string{"a": "b"}, map[string]string{"a": "b"}},
		{"query-stats", stats, &stats},
		{"query-stats-pointer", &stats, &stats},
		{"explain-plan", plan, &plan},
		{"explain-plan-pointer", &plan, &plan},
		{"graph-path", path, &path},
		{"graph-path-pointer", &path, &path},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			expected, err := apexjson.Marshal(testCase.want)
			if err != nil {
				t.Fatalf("Marshal: %v", err)
			}
			got := metadataValueToString(testCase.in)
			if got != string(expected) {
				t.Errorf("got  %s\nwant %s", got, expected)
			}
		})
	}
}

// TestPGWire_MetadataValueToString_MapWireParity checks the map case
// separately, because it goes through jsonWireValue, which replaces the
// internal JSON-null sentinel with a real null.
func TestPGWire_MetadataValueToString_MapWireParity(t *testing.T) {
	in := map[string]interface{}{
		"null":   util.JSONNull{},
		"nested": map[string]interface{}{"null": util.JSONNull{}},
		"list":   []interface{}{util.JSONNull{}, 1},
	}
	// The sentinel encodes as the JSON null literal where it sits, so the
	// document is expected to encode as it stands.
	expected, err := apexjson.Marshal(in)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if got := metadataValueToString(in); got != string(expected) {
		t.Errorf("got  %s\nwant %s", got, expected)
	}
}
