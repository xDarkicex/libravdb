package libravdb

import (
	"context"
	"testing"

	"github.com/xDarkicex/libravdb/internal/graph"
)

func TestSQL_RRFVectorAndFTSRank(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(t.TempDir()+"/rrf.libravdb"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	col, err := db.CreateCollection(ctx, "documents", WithDimension(3), WithMetadataSchema(MetadataSchema{
		"content": StringField,
	}))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	fixtures := []struct {
		id      string
		vector  []float32
		content string
	}{
		{"semantic", []float32{1, 0, 0}, "security incident response"},
		{"lexical", []float32{0.8, 0.2, 0}, "security security incident"},
		{"other", []float32{0, 1, 0}, "unrelated gardening notes"},
	}
	for _, fixture := range fixtures {
		if err := col.Insert(ctx, fixture.id, fixture.vector, map[string]interface{}{"content": fixture.content}); err != nil {
			t.Fatalf("Insert %s: %v", fixture.id, err)
		}
	}

	result, err := db.QueryWithParams(ctx,
		"SELECT id, RRF(VECTOR_DISTANCE(embedding, $query_vec), FTS_RANK(content, $text_query)) AS unified_relevance "+
			"FROM documents ORDER BY unified_relevance DESC LIMIT 3",
		QueryParams{
			"query_vec":  []float32{1, 0, 0},
			"text_query": "security incident",
		})
	if err != nil {
		t.Fatalf("RRF query: %v", err)
	}
	if result.Total != 3 {
		t.Fatalf("RRF rows: got %d, want 3", result.Total)
	}
	if result.Results[0].ID != "semantic" && result.Results[0].ID != "lexical" {
		t.Fatalf("RRF top row %q is neither signal leader", result.Results[0].ID)
	}
	for i, row := range result.Results {
		value, ok := row.Metadata["unified_relevance"].(float64)
		if !ok || value <= 0 {
			t.Fatalf("row %d RRF score: %#v", i, row.Metadata["unified_relevance"])
		}
	}

	// A nonmatching lexical row is not ranked into the FTS list and therefore
	// receives only the vector component's reciprocal-rank contribution.
	if result.Results[2].ID != "other" {
		t.Fatalf("expected nonmatching row last, got %q", result.Results[2].ID)
	}

	fts, err := db.QueryWithParams(ctx,
		"SELECT id, FTS_RANK(content, $text_query) AS lexical_score FROM documents ORDER BY lexical_score DESC LIMIT 3",
		QueryParams{"text_query": "security incident"})
	if err != nil {
		t.Fatalf("standalone FTS_RANK query: %v", err)
	}
	if fts.Total != 3 || fts.Results[0].ID != "lexical" || fts.Results[0].Metadata["lexical_score"].(float64) <= 0 {
		t.Fatalf("standalone FTS_RANK result: %#v", fts.Results)
	}
}

func TestSQL_RRFWithGraphCentralityAndMatch(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(t.TempDir()+"/rrf_graph.libravdb"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	gr, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	defer gr.Close()
	const edgeKind = uint8(197)
	if !graph.RegisterEdgeKind("RRF_CITES", edgeKind) {
		t.Log("RRF_CITES already registered")
	}
	col, err := db.CreateCollection(ctx, "graph_docs", WithDimension(3), WithGraph(gr), WithMetadataSchema(MetadataSchema{
		"content": StringField,
	}))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	for _, fixture := range []struct {
		id      string
		vector  []float32
		content string
	}{
		{"target", []float32{1, 0, 0}, "security incident response"},
		{"source_a", []float32{0, 1, 0}, "security report"},
		{"source_b", []float32{0, 0, 1}, "security report"},
	} {
		if err := col.Insert(ctx, fixture.id, fixture.vector, map[string]interface{}{"content": fixture.content}); err != nil {
			t.Fatalf("Insert %s: %v", fixture.id, err)
		}
	}
	target, _ := db.GetNodeID(ctx, "graph_docs", "target")
	sourceA, _ := db.GetNodeID(ctx, "graph_docs", "source_a")
	sourceB, _ := db.GetNodeID(ctx, "graph_docs", "source_b")
	gr.RegisterVertexLabel(target, "Document")
	gr.RegisterVertexLabel(sourceA, "Document")
	gr.RegisterVertexLabel(sourceB, "Document")
	txn := gr.BeginTxn()
	if err := txn.AddEdge(sourceA, target, 1, edgeKind); err != nil {
		t.Fatalf("AddEdge A: %v", err)
	}
	if err := txn.AddEdge(sourceB, target, 1, edgeKind); err != nil {
		t.Fatalf("AddEdge B: %v", err)
	}
	if err := txn.Commit(ctx); err != nil {
		t.Fatalf("Commit graph: %v", err)
	}

	result, err := db.QueryWithParams(ctx,
		"SELECT id, RRF(VECTOR_DISTANCE(embedding, $q), FTS_RANK(content, $text), GRAPH_CENTRALITY(d)) AS score "+
			"FROM graph_docs d WHERE MATCH (d)<-[:RRF_CITES]-(ref:Document) ORDER BY score DESC",
		QueryParams{"q": []float32{1, 0, 0}, "text": "security incident"})
	if err != nil {
		t.Fatalf("RRF graph query: %v", err)
	}
	if result.Total != 1 || result.Results[0].ID != "target" {
		t.Fatalf("RRF graph rows: %#v", result.Results)
	}
	if result.Results[0].Metadata["score"].(float64) <= 0 {
		t.Fatalf("RRF graph score is not positive: %#v", result.Results[0].Metadata["score"])
	}
}

func TestSQL_RRFRelationalJoinProjection(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(t.TempDir()+"/rrf_join.libravdb"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	if _, err := db.CreateCollection(ctx, "authors", WithMetadataOnly(), WithMetadataSchema(MetadataSchema{"name": StringField})); err != nil {
		t.Fatalf("Create authors: %v", err)
	}
	if _, err := db.CreateCollection(ctx, "documents", WithDimension(3), WithMetadataSchema(MetadataSchema{
		"author_id": StringField,
		"title":     StringField,
		"content":   StringField,
	})); err != nil {
		t.Fatalf("Create documents: %v", err)
	}
	authors, _ := db.GetCollection("authors")
	if err := authors.Insert(ctx, "a1", nil, map[string]interface{}{"name": "Ada"}); err != nil {
		t.Fatalf("Insert author: %v", err)
	}
	documents, _ := db.GetCollection("documents")
	if err := documents.Insert(ctx, "d1", []float32{1, 0, 0}, map[string]interface{}{
		"author_id": "a1",
		"title":     "Incident",
		"content":   "security incident response",
	}); err != nil {
		t.Fatalf("Insert document: %v", err)
	}

	result, err := db.QueryWithParams(ctx,
		"SELECT d.title, a.name AS author_name, RRF(VECTOR_DISTANCE(d.embedding, $q), FTS_RANK(d.content, $text)) AS unified_relevance "+
			"FROM documents d JOIN authors a ON d.author_id = a.id ORDER BY unified_relevance DESC",
		QueryParams{"q": []float32{1, 0, 0}, "text": "security incident"})
	if err != nil {
		t.Fatalf("RRF join query: %v", err)
	}
	if result.Total != 1 || result.Results[0].Metadata["title"] != "Incident" || result.Results[0].Metadata["author_name"] != "Ada" {
		t.Fatalf("RRF joined metadata: %#v", result.Results)
	}
	if result.Results[0].Metadata["unified_relevance"].(float64) <= 0 {
		t.Fatalf("RRF joined score: %#v", result.Results[0].Metadata["unified_relevance"])
	}
}
