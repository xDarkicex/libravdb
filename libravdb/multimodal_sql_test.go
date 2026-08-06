package libravdb

import (
	"context"
	"testing"

	"github.com/xDarkicex/libravdb/internal/graph"
)

func TestSQLMultiModalJoinGraphVector(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:multimodal_sql"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	g, err := NewGraph(GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	services, err := db.CreateCollection(ctx, "services", WithDimension(3), WithMetric(CosineDistance), WithGraph(g),
		WithMetadataSchema(MetadataSchema{"owner_id": StringField, "title": StringField}))
	if err != nil {
		t.Fatal(err)
	}
	teams, err := db.CreateCollection(ctx, "teams", WithMetadataOnly(), WithMetadataSchema(MetadataSchema{"sla_status": StringField}))
	if err != nil {
		t.Fatal(err)
	}

	for _, team := range []struct{ id, status string }{{"active", "active"}, {"inactive", "inactive"}} {
		if err := teams.Insert(ctx, team.id, nil, map[string]interface{}{"sla_status": team.status}); err != nil {
			t.Fatal(err)
		}
	}
	for _, record := range []struct {
		id, owner, title string
		vector           []float32
	}{
		{"svc-active", "active", "active service", []float32{0, 1, 0}},
		{"svc-inactive", "inactive", "inactive service", []float32{0, 1, 0}},
		{"api-active", "", "active api", []float32{0, 1, 0}},
		{"api-inactive", "", "inactive api", []float32{0, 1, 0}},
		{"doc-active", "", "active manual", []float32{0.98, 0.02, 0}},
		// This is nearer to the query but belongs to an inactive team. If
		// relational anchors are not applied before BFS, it incorrectly wins.
		{"doc-inactive", "", "inactive manual", []float32{1, 0, 0}},
	} {
		if err := services.Insert(ctx, record.id, record.vector, map[string]interface{}{"owner_id": record.owner, "title": record.title}); err != nil {
			t.Fatal(err)
		}
	}

	node := func(id string) uint64 {
		n, err := db.GetNodeID(ctx, "services", id)
		if err != nil {
			t.Fatalf("node %s: %v", id, err)
		}
		return n
	}
	if !graph.RegisterEdgeKind("DEPENDS_ON", 71) {
		t.Fatalf("RegisterEdgeKind DEPENDS_ON=71 failed: kind already claimed")
	}
	if !graph.RegisterEdgeKind("DOCUMENTED_BY", 72) {
		t.Fatalf("RegisterEdgeKind DOCUMENTED_BY=72 failed: kind already claimed")
	}
	for _, edge := range [][3]uint64{
		{node("svc-active"), node("api-active"), 71},
		{node("api-active"), node("doc-active"), 72},
		{node("svc-inactive"), node("api-inactive"), 71},
		{node("api-inactive"), node("doc-inactive"), 72},
	} {
		txn := g.BeginTxn()
		if err := g.AddEdge(txn, edge[0], edge[1], 1, uint8(edge[2])); err != nil {
			t.Fatal(err)
		}
		if err := txn.Commit(ctx); err != nil {
			t.Fatal(err)
		}
	}

	query := "SELECT doc.title, VECTOR_DISTANCE(vector, '[1,0,0]') AS semantic_relevance " +
		"FROM services s JOIN teams t ON s.owner_id = t.id AND t.sla_status = 'active' " +
		"JOIN MATCH (s)-[:DEPENDS_ON]->(api)-[:DOCUMENTED_BY]->(doc) " +
		"ORDER BY semantic_relevance ASC LIMIT 1"
	results, err := db.Query(ctx, query)
	if err != nil {
		t.Fatalf("multimodal SQL: %v", err)
	}
	if len(results.Results) != 1 {
		t.Fatalf("rows=%d, want 1", len(results.Results))
	}
	if results.Results[0].ID != "doc-active" {
		t.Fatalf("winner=%q, want doc-active; inactive graph path leaked into ranking", results.Results[0].ID)
	}
	if title := results.Results[0].Metadata["title"]; title != "active manual" {
		t.Fatalf("title=%v, want active manual", title)
	}
	if _, ok := results.Results[0].Metadata["semantic_relevance"]; !ok {
		t.Fatal("VECTOR_DISTANCE projection was not returned")
	}
}
