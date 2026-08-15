package libravdb

import (
	"context"
	"testing"
)

func TestSQL_PostgreSQLFTSCore(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(t.TempDir()+"/fts.libravdb"), WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	col, err := db.CreateCollection(ctx, "documents", WithMetadataOnly(), WithMetadataSchema(MetadataSchema{
		"content": StringField,
	}))
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	for _, row := range []struct{ id, content string }{
		{"d1", "security incident response"},
		{"d2", "security vector search"},
		{"d3", "gardening notes"},
	} {
		if err := col.Insert(ctx, row.id, nil, map[string]interface{}{"content": row.content}); err != nil {
			t.Fatalf("Insert %s: %v", row.id, err)
		}
	}

	result, err := db.QueryWithParams(ctx, `
		SELECT id,
		       to_tsvector('english', content) AS document_vector,
		       plainto_tsquery('english', $query) AS document_query,
		       ts_rank(to_tsvector('english', content), plainto_tsquery('english', $query)) AS rank
		FROM documents
		WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $query)
		ORDER BY rank DESC`, QueryParams{"query": "security incident"})
	if err != nil {
		t.Fatalf("core FTS query: %v", err)
	}
	if result.Total != 1 || result.Results[0].ID != "d1" {
		t.Fatalf("core FTS rows: %#v", result.Results)
	}
	row := result.Results[0].Metadata
	if row["document_vector"] == nil || row["document_query"] == nil {
		t.Fatalf("core FTS materialization: %#v", row)
	}
	if rank, ok := row["rank"].(float64); !ok || rank <= 0 {
		t.Fatalf("core FTS rank: %#v", row["rank"])
	}

	web, err := db.QueryWithParams(ctx,
		"SELECT id FROM documents WHERE to_tsvector(content) @@ websearch_to_tsquery($query) ORDER BY id",
		QueryParams{"query": "security OR gardening"})
	if err != nil {
		t.Fatalf("websearch FTS query: %v", err)
	}
	if web.Total != 3 {
		t.Fatalf("websearch rows: got %d, want 3", web.Total)
	}

	raw, err := db.Query(ctx,
		"SELECT id FROM documents WHERE to_tsvector(content) @@ to_tsquery('security & vector')")
	if err != nil {
		t.Fatalf("raw tsquery: %v", err)
	}
	if raw.Total != 1 || raw.Results[0].ID != "d2" {
		t.Fatalf("raw tsquery rows: %#v", raw.Results)
	}

	phrase, err := db.Query(ctx,
		"SELECT id FROM documents WHERE to_tsvector(content) @@ phraseto_tsquery('security incident') ORDER BY id")
	if err != nil {
		t.Fatalf("phrase FTS query: %v", err)
	}
	if phrase.Total != 1 || phrase.Results[0].ID != "d1" {
		t.Fatalf("phrase FTS rows: %#v", phrase.Results)
	}

	prefix, err := db.Query(ctx,
		"SELECT id FROM documents WHERE to_tsvector(content) @@ to_tsquery('secur:*') ORDER BY id")
	if err != nil {
		t.Fatalf("prefix FTS query: %v", err)
	}
	if prefix.Total != 2 || prefix.Results[0].ID != "d1" || prefix.Results[1].ID != "d2" {
		t.Fatalf("prefix FTS rows: %#v", prefix.Results)
	}

	not, err := db.Query(ctx,
		"SELECT id FROM documents WHERE to_tsvector(content) @@ websearch_to_tsquery('security -incident') ORDER BY id")
	if err != nil {
		t.Fatalf("negative websearch query: %v", err)
	}
	if not.Total != 1 || not.Results[0].ID != "d2" {
		t.Fatalf("negative websearch rows: %#v", not.Results)
	}

	english, err := db.Query(ctx,
		"SELECT id FROM documents WHERE to_tsvector('english', content) @@ plainto_tsquery('english', 'security incidents') ORDER BY id")
	if err != nil {
		t.Fatalf("english FTS query: %v", err)
	}
	if english.Total != 1 || english.Results[0].ID != "d1" {
		// The configured dictionary stems the plural query term to the
		// singular lexeme without broadening unrelated rows.
		t.Fatalf("english FTS rows: %#v", english.Results)
	}

	cover, err := db.Query(ctx,
		"SELECT ts_rank_cd(to_tsvector(content), plainto_tsquery('security incident')) AS rank FROM documents WHERE id = 'd1'")
	if err != nil {
		t.Fatalf("cover-density rank: %v", err)
	}
	if cover.Total != 1 || cover.Results[0].Metadata["rank"].(float64) <= 0 {
		t.Fatalf("cover-density rank rows: %#v", cover.Results)
	}
	normalized, err := db.Query(ctx,
		"SELECT ts_rank(to_tsvector(content), plainto_tsquery('security incident'), 2) AS rank FROM documents WHERE id = 'd1'")
	if err != nil {
		t.Fatalf("normalized rank: %v", err)
	}
	if normalized.Total != 1 || normalized.Results[0].Metadata["rank"].(float64) <= 0 {
		t.Fatalf("normalized rank rows: %#v", normalized.Results)
	}
}
