package libravdb

import (
	"context"
	"math"
	"testing"
)

// TestSQL_VectorDistanceProjection verifies SIMILARITY()/VECTOR_DISTANCE()
// work as SELECT projections with ORDER BY, computing real per-record scores
// through the SIMD-backed util distance functions.
func TestSQL_VectorDistanceProjection(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:vecproj_test"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	col, err := db.CreateCollection(ctx, "docs", WithDimension(4))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	// Insert records with 4-dim vectors. The engine's cosine path assumes
	// pre-normalized unit vectors (Nomic v1.5 emits normalized embeddings),
	// so normalize each vector in the fixture.
	vectors := map[string][]float32{
		"a": {1, 0, 0, 0},
		"b": {0.9, 0.1, 0, 0},
		"c": {0, 1, 0, 0},
		"d": {0, 0, 1, 0},
	}
	for id, vec := range vectors {
		if err := col.Insert(ctx, id, normalize(vec), nil); err != nil {
			t.Fatalf("insert %s: %v", id, err)
		}
	}

	// Query vector: closest to "a", then "b", then "c", then "d" (cosine).
	qv := []float32{1, 0, 0, 0}

	// VECTOR_DISTANCE projection + ORDER BY ASC (distance ascending = closest first).
	res, err := db.Query(ctx, "SELECT id, VECTOR_DISTANCE(vector, '[1,0,0,0]') AS d FROM docs ORDER BY d")
	if err != nil {
		t.Fatalf("Query vector distance failed: %v", err)
	}
	if len(res.Results) != 4 {
		t.Fatalf("expected 4 rows, got %d", len(res.Results))
	}
	// Verify order: a, b, c, d (ascending distance).
	expectedOrder := []string{"a", "b", "c", "d"}
	got := make([]string, 0, 4)
	for _, r := range res.Results {
		got = append(got, r.ID)
		// Verify the distance value is present and matches brute force.
		var d float64
		switch dv := r.Metadata["d"].(type) {
		case float64:
			d = dv
		case float32:
			d = float64(dv)
		default:
			t.Fatalf("row %s: d not numeric, got %T", r.ID, r.Metadata["d"])
		}
		rec, err := col.Get(ctx, r.ID)
		if err != nil {
			t.Fatalf("Get %s: %v", r.ID, err)
		}
		want := 1 - cosineSim(qv, rec.Vector)
		if math.Abs(float64(d)-float64(want)) > 1e-5 {
			t.Errorf("row %s: distance %f, want %f", r.ID, d, want)
		}
	}
	for i := range expectedOrder {
		if got[i] != expectedOrder[i] {
			t.Fatalf("order: got %v, want %v", got, expectedOrder)
		}
	}

	// ORDER BY DESC should reverse.
	resDesc, err := db.Query(ctx, "SELECT id, VECTOR_DISTANCE(vector, '[1,0,0,0]') AS d FROM docs ORDER BY d DESC")
	if err != nil {
		t.Fatalf("Query desc failed: %v", err)
	}
	gotDesc := make([]string, 0, 4)
	for _, r := range resDesc.Results {
		gotDesc = append(gotDesc, r.ID)
	}
	wantDesc := []string{"d", "c", "b", "a"}
	for i := range wantDesc {
		if gotDesc[i] != wantDesc[i] {
			t.Fatalf("desc order: got %v, want %v", gotDesc, wantDesc)
		}
	}

	// SIMILARITY projection (1 - distance) should be near 1 for the closest.
	resSim, err := db.Query(ctx, "SELECT id, SIMILARITY(vector, '[1,0,0,0]') AS sim FROM docs ORDER BY sim DESC")
	if err != nil {
		t.Fatalf("Query similarity failed: %v", err)
	}
	if len(resSim.Results) == 0 {
		t.Fatal("similarity query returned no rows")
	}
	if resSim.Results[0].ID != "a" {
		t.Errorf("similarity top: got %s, want a", resSim.Results[0].ID)
	}
	sim, ok := resSim.Results[0].Metadata["sim"].(float32)
	if !ok {
		t.Fatalf("sim not float32, got %T", resSim.Results[0].Metadata["sim"])
	}
	if math.Abs(float64(sim)-1.0) > 1e-5 {
		t.Errorf("sim of exact match: %f, want 1.0", sim)
	}

	// ARRAY_COSINE_SIMILARITY is a Kuzu/Graphiti-compatible spelling of the
	// existing max-similarity vector projection and must produce the same
	// values and ordering as SIMILARITY.
	resArray, err := db.Query(ctx, "SELECT id, array_cosine_similarity(vector, '[1,0,0,0]') AS sim FROM docs ORDER BY sim DESC")
	if err != nil {
		t.Fatalf("Query array_cosine_similarity failed: %v", err)
	}
	if len(resArray.Results) != len(resSim.Results) {
		t.Fatalf("array cosine row count: got %d, want %d", len(resArray.Results), len(resSim.Results))
	}
	for i := range resSim.Results {
		if resArray.Results[i].ID != resSim.Results[i].ID {
			t.Fatalf("array cosine order at %d: got %s, want %s", i, resArray.Results[i].ID, resSim.Results[i].ID)
		}
		want, ok := resSim.Results[i].Metadata["sim"].(float32)
		if !ok {
			t.Fatalf("similarity row %s not float32: %T", resSim.Results[i].ID, resSim.Results[i].Metadata["sim"])
		}
		got, ok := resArray.Results[i].Metadata["sim"].(float32)
		if !ok {
			t.Fatalf("array cosine row %s not float32: %T", resArray.Results[i].ID, resArray.Results[i].Metadata["sim"])
		}
		if math.Abs(float64(got-want)) > 1e-5 {
			t.Errorf("array cosine row %s: got %f, want %f", resArray.Results[i].ID, got, want)
		}
	}
}

func TestSQL_ArrayCosineSimilarityInCypherWith(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:array_cosine_cypher"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	g, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	defer g.Close()
	col, err := db.CreateCollection(ctx, "people", WithDimension(4), WithMetric(CosineDistance), WithGraph(g), WithMetadataSchema(MetadataSchema{
		"group_id": StringField,
	}))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	for _, row := range []struct {
		id string
		v  []float32
	}{
		{id: "alice", v: []float32{1, 0, 0, 0}},
		{id: "bob", v: []float32{0.9, 0.1, 0, 0}},
		{id: "carol", v: []float32{0, 1, 0, 0}},
	} {
		if err := col.Insert(ctx, row.id, normalize(row.v), map[string]interface{}{"group_id": "g"}); err != nil {
			t.Fatalf("Insert %s: %v", row.id, err)
		}
	}

	rows, err := db.QueryWithParams(ctx, `
		MATCH (n)
		WITH n, array_cosine_similarity(n.vector, $v) AS score
		WHERE score > $min
		RETURN n.id AS uuid, score
		ORDER BY score DESC
		LIMIT $limit`, QueryParams{
		"v":     []float32{1, 0, 0, 0},
		"min":   float32(0.5),
		"limit": int64(2),
	})
	if err != nil {
		t.Fatalf("Graphiti-shaped ARRAY_COSINE_SIMILARITY query: %v", err)
	}
	if rows.Total != 2 {
		t.Fatalf("rows=%d, want 2: %#v", rows.Total, rows.Results)
	}
	if rows.Results[0].Metadata["uuid"] != "alice" || rows.Results[1].Metadata["uuid"] != "bob" {
		t.Fatalf("ordered rows=%#v", rows.Results)
	}
}

func cosineSim(a, b []float32) float32 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i] * b[i])
		na += float64(a[i] * a[i])
		nb += float64(b[i] * b[i])
	}
	return float32(dot / (math.Sqrt(na) * math.Sqrt(nb)))
}

func normalize(v []float32) []float32 {
	var norm float64
	for _, x := range v {
		norm += float64(x) * float64(x)
	}
	norm = math.Sqrt(norm)
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(float64(x) / norm)
	}
	return out
}
