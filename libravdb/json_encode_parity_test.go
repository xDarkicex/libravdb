package libravdb

import (
	"bytes"
	"context"
	"runtime"
	"testing"

	apexjson "github.com/xDarkicex/apexJSON/v2"
	"github.com/xDarkicex/libravdb/internal/graph"
	"github.com/xDarkicex/libravdb/internal/optimizer"
	"github.com/xDarkicex/libravdb/internal/util"
)

type customTestStruct struct {
	Name string `json:"name"`
	Age  int    `json:"age"`
}

func TestJSONEncode_ByteParity_EncodeJSONValue(t *testing.T) {
	testCases := []struct {
		name  string
		value interface{}
	}{
		{"int", 42},
		{"float", 3.14159},
		{"string", "hello world"},
		{"bool_true", true},
		{"bool_false", false},
		{"json_null", util.JSONNull{}},
		{"simple_map", map[string]interface{}{"a": 1, "b": "text"}},
		{"nested_map", map[string]interface{}{
			"id":     100,
			"obj":    map[string]interface{}{"sub": true, "null": util.JSONNull{}},
			"array":  []interface{}{"item1", 2, nil, util.JSONNull{}},
			"active": false,
		}},
		{"empty_map", map[string]interface{}{}},
		{"empty_slice", []interface{}{}},
		{"string_map", map[string]string{"foo": "bar", "key": "val"}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := encodeJSONValue(tc.value)
			if err != nil {
				t.Fatalf("encodeJSONValue failed: %v", err)
			}
			wantBytes, err := apexjson.Marshal(jsonWireValue(tc.value))
			if err != nil {
				t.Fatalf("apexjson.Marshal baseline failed: %v", err)
			}
			want := string(wantBytes)
			if got != want {
				t.Fatalf("byte mismatch for %s:\ngot:  %q\nwant: %q", tc.name, got, want)
			}
			if !bytes.Equal([]byte(got), wantBytes) {
				t.Fatalf("raw bytes mismatch for %s", tc.name)
			}
		})
	}
}

func TestJSONEncode_ByteParity_DecodeJSONValueDefaultBranch(t *testing.T) {
	custom := customTestStruct{Name: "Gopher", Age: 15}

	// Baseline decode using Marshal + decodeJSONText
	rawBaseline, err := apexjson.Marshal(custom)
	if err != nil {
		t.Fatalf("baseline Marshal: %v", err)
	}
	want, wantOK := decodeJSONText(rawBaseline)
	if !wantOK {
		t.Fatalf("baseline decode failed")
	}

	got, gotOK := decodeJSONValue(custom)
	if !gotOK {
		t.Fatalf("decodeJSONValue failed on custom struct")
	}

	gotJSON, err := apexjson.Marshal(got)
	if err != nil {
		t.Fatalf("marshal got: %v", err)
	}
	wantJSON, err := apexjson.Marshal(want)
	if err != nil {
		t.Fatalf("marshal want: %v", err)
	}
	if !bytes.Equal(gotJSON, wantJSON) {
		t.Fatalf("mismatch in decoded struct representation:\ngot:  %s\nwant: %s", string(gotJSON), string(wantJSON))
	}
}

func TestJSONEncode_ByteParity_ConflictColumnValue(t *testing.T) {
	testValues := []struct {
		name string
		val  interface{}
	}{
		{"map_interface", map[string]interface{}{"k": "v", "num": 10}},
		{"map_string", map[string]string{"k": "v"}},
		{"slice_interface", []interface{}{"a", 1, true}},
		{"slice_string", []string{"x", "y"}},
		{"slice_int", []int{1, 2, 3}},
		{"slice_float64", []float64{1.1, 2.2}},
	}

	for _, tc := range testValues {
		t.Run(tc.name, func(t *testing.T) {
			meta := map[string]interface{}{"col": tc.val}
			got, ok := conflictColumnValue("id1", meta, "col")
			if !ok {
				t.Fatalf("conflictColumnValue returned not ok")
			}
			wantBytes, err := apexjson.Marshal(tc.val)
			if err != nil {
				t.Fatalf("baseline Marshal failed: %v", err)
			}
			want := string(wantBytes)
			if got != want {
				t.Fatalf("conflictColumnValue mismatch for %s:\ngot:  %q\nwant: %q", tc.name, got, want)
			}
		})
	}
}

func TestJSONEncode_ByteParity_OptimizerScalarFromInterface(t *testing.T) {
	testValues := []struct {
		name string
		val  interface{}
	}{
		{"map", map[string]interface{}{"query": "term", "limit": 10}},
		{"map_string", map[string]string{"a": "b"}},
		{"slice", []interface{}{"alpha", 123}},
		{"slice_string", []string{"first", "second"}},
		{"slice_int", []int{10, 20}},
		{"slice_float64", []float64{3.14, 2.71}},
	}

	for _, tc := range testValues {
		t.Run(tc.name, func(t *testing.T) {
			scalar := optimizer.ScalarFromInterface(tc.val)
			if scalar.Kind != optimizer.ScalarJSON {
				t.Fatalf("expected ScalarJSON, got %v", scalar.Kind)
			}
			wantBytes, err := apexjson.Marshal(tc.val)
			if err != nil {
				t.Fatalf("baseline Marshal: %v", err)
			}
			if !bytes.Equal(scalar.BytesData, wantBytes) {
				t.Fatalf("ScalarFromInterface mismatch for %s:\ngot:  %q\nwant: %q", tc.name, string(scalar.BytesData), string(wantBytes))
			}
		})
	}
}

func TestJSONEncode_ByteParity_CastToText(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir()+"/json_parity"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	ctx := context.Background()
	if _, err := db.Query(ctx, "CREATE TABLE test_parity (id TEXT PRIMARY KEY, obj JSONB, arr JSONB)"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, `INSERT INTO test_parity (id, obj, arr) VALUES ('row1', '{"a":1,"b":"hello"}', '[10,20,30]')`); err != nil {
		t.Fatal(err)
	}

	res, err := db.Query(ctx, "SELECT obj::text AS obj, arr::text AS arr FROM test_parity WHERE id = 'row1'")
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(res.Results))
	}
	gotObj := res.Results[0].Metadata["obj"]
	gotArr := res.Results[0].Metadata["arr"]

	wantObjBytes, _ := apexjson.Marshal(map[string]interface{}{"a": int64(1), "b": "hello"})
	wantArrBytes, _ := apexjson.Marshal([]interface{}{int64(10), int64(20), int64(30)})

	if gotObj != string(wantObjBytes) {
		t.Fatalf("obj::text mismatch: got %q, want %q", gotObj, string(wantObjBytes))
	}
	if gotArr != string(wantArrBytes) {
		t.Fatalf("arr::text mismatch: got %q, want %q", gotArr, string(wantArrBytes))
	}
}

func TestJSONEncode_ByteParity_MergeEdgeProperties(t *testing.T) {
	db, err := Open(WithStoragePath(t.TempDir()+"/merge_parity"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	g, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()

	ctx := context.Background()
	col, err := db.CreateCollection(ctx, "merge_users", WithMetadataOnly(), WithGraph(g))
	if err != nil {
		t.Fatal(err)
	}
	_ = graph.RegisterEdgeKind("PARITY_REL", 232)

	if _, err := db.QueryWithParams(ctx, `MERGE (a:merge_users {id: $a, name: 'Alice'})-[r:PARITY_REL]->(b:merge_users {id: $b, name: 'Bob'}) SET r.fact = $fact, r.weight = $w`,
		QueryParams{"a": "u1", "b": "u2", "fact": "tested", "w": 42}); err != nil {
		t.Fatalf("MERGE: %v", err)
	}

	aliceNode, err := db.GetNodeID(ctx, "merge_users", "u1")
	if err != nil {
		t.Fatalf("u1 node not found: %v", err)
	}

	views, err := col.GetGraph().NeighborsWithProperties(aliceNode)
	if err != nil || len(views) != 1 {
		t.Fatalf("neighbors lookup views=%#v err=%v", views, err)
	}

	raw, err := graph.EdgePropertyJSON(views[0].Properties)
	if err != nil {
		t.Fatal(err)
	}

	var properties map[string]interface{}
	if err := apexjson.Unmarshal(raw, &properties); err != nil {
		t.Fatalf("unmarshal edge properties: %v", err)
	}

	if properties["fact"] != "tested" || properties["weight"] != float64(42) {
		t.Fatalf("unexpected edge properties: %#v", properties)
	}
}

// TestDecodeJSONTextAliasesItsInput characterizes the hazard the pooled
// encoder has to respect: the decoded value's strings point into the bytes
// they were decoded from, rather than into copies of them.
func TestDecodeJSONTextAliasesItsInput(t *testing.T) {
	buf := []byte(`{"name":"hello"}`)
	decoded, ok := decodeJSONText(buf)
	if !ok {
		t.Fatal("decodeJSONText failed")
	}
	m, ok := decoded.(map[string]interface{})
	if !ok {
		t.Fatalf("decoded %T", decoded)
	}
	got, _ := m["name"].(string)
	if got != "hello" {
		t.Fatalf("name = %q, want %q", got, "hello")
	}
	buf[9] = 'X'
	after, _ := m["name"].(string)
	if after == got {
		t.Log("the decoded string does not alias its input")
		return
	}
	t.Logf("the decoded string aliases its input: %q", after)
}

// TestDecodeJSONValueOutlivesTheEncoder is the regression test for the
// pooled encoder in decodeJSONValue.
//
// decodeJSONValue encodes a value the fast paths do not cover and decodes the
// text back. The decoded strings alias the bytes they were decoded from, so
// those bytes must outlive the value. If they come from a pooled buffer and
// the buffer goes back to the pool, the next user of the pool overwrites a
// string the caller still holds.
func TestDecodeJSONValueOutlivesTheEncoder(t *testing.T) {
	// One processor, so the pool this test churns is the pool the decode
	// used. With more, the decode and the test could land on different
	// per-processor pools and the test would prove nothing.
	previous := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(previous)

	type payload struct {
		Name string `json:"name"`
	}
	decoded, ok := decodeJSONValue(payload{Name: "hello"})
	if !ok {
		t.Fatal("decodeJSONValue failed")
	}
	m, ok := decoded.(map[string]interface{})
	if !ok {
		t.Fatalf("decoded %T", decoded)
	}
	if got, _ := m["name"].(string); got != "hello" {
		t.Fatalf("name = %q, want %q", got, "hello")
	}

	// Dirty every buffer the pool is holding, over and over, so whichever
	// one the decode used is overwritten. The estimate matches the one the
	// decode used, so the pool hands back the same buffer instead of
	// replacing it with a larger one.
	for i := 0; i < 128; i++ {
		enc := util.AcquireJSONEncoder(64)
		for j := 0; j < cap(enc.Buf); j++ {
			enc.Buf = append(enc.Buf, 'X')
		}
		util.ReleaseJSONEncoder(enc)
	}

	if got, _ := m["name"].(string); got != "hello" {
		t.Fatalf("the decoded string was overwritten by the pool: %q", got)
	}
}
