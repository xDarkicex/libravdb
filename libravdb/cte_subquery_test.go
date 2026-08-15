package libravdb

import (
	"context"
	"testing"
)

func TestSQLGenericCTEAndSubqueries(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:generic-cte-subqueries"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.CreateCollection(ctx, "authors", WithMetadataOnly(), WithMetadataSchema(MetadataSchema{"name": StringField})); err != nil {
		t.Fatal(err)
	}
	if _, err := db.CreateCollection(ctx, "documents", WithMetadataOnly(), WithMetadataSchema(MetadataSchema{
		"author_id": StringField,
		"category":  StringField,
	})); err != nil {
		t.Fatal(err)
	}
	for _, query := range []string{
		"INSERT INTO authors (id, name) VALUES ('a1', 'Ada'), ('a2', 'Grace')",
		"INSERT INTO documents (id, author_id, category) VALUES ('d1', 'a1', 'graph'), ('d2', 'a2', 'vector'), ('d3', 'missing', 'other')",
	} {
		if _, err := db.Query(ctx, query); err != nil {
			t.Fatal(query, err)
		}
	}

	cte, err := db.Query(ctx, `WITH recent AS (
		SELECT id FROM documents WHERE category = 'graph'
	)
	SELECT d.id, d.category
	FROM documents d
	JOIN recent r ON r.id = d.id
	ORDER BY d.id`)
	if err != nil {
		t.Fatal("generic CTE", err)
	}
	if cte.Total != 1 || cte.Results[0].ID != "d1" || cte.Results[0].Metadata["category"] != "graph" {
		t.Fatalf("generic CTE result=%#v", cte)
	}
	multiple, err := db.Query(ctx, `WITH recent AS (
		SELECT id, author_id FROM documents WHERE category = 'graph'
	), author_rows AS (
		SELECT a.id, a.name FROM authors a JOIN recent r ON a.id = r.author_id
	)
	SELECT id, name FROM author_rows ORDER BY id`)
	if err != nil {
		t.Fatalf("multiple CTEs: %v", err)
	}
	if multiple.Total != 1 || multiple.Results[0].ID != "a1" || multiple.Results[0].Metadata["name"] != "Ada" {
		t.Fatalf("multiple CTE result=%#v", multiple)
	}
	if _, err := db.CreateCollection(ctx, "tree_nodes", WithMetadataOnly(), WithMetadataSchema(MetadataSchema{
		"parent_id": StringField,
	})); err != nil {
		t.Fatal("create tree_nodes", err)
	}
	tree, _ := db.GetCollection("tree_nodes")
	for _, node := range []struct{ id, parent string }{{"root", ""}, {"child-a", "root"}, {"child-b", "root"}, {"grandchild", "child-a"}} {
		if err := tree.Insert(ctx, node.id, nil, map[string]interface{}{"parent_id": node.parent}); err != nil {
			t.Fatal("insert tree node", err)
		}
	}
	recursive, err := db.Query(ctx, `WITH RECURSIVE tree AS (
		SELECT id, parent_id FROM tree_nodes WHERE id = 'root'
		UNION ALL
		SELECT c.id, c.parent_id FROM tree_nodes c JOIN tree t ON c.parent_id = t.id
	)
	SELECT id FROM tree ORDER BY id`)
	if err != nil {
		t.Fatalf("recursive CTE: %v", err)
	}
	if recursive.Total != 4 {
		t.Fatalf("recursive CTE rows=%d result=%#v", recursive.Total, recursive)
	}
	session, err := db.NewSQLSession(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	if sessionCTE, err := session.Query(`WITH recent AS (SELECT id FROM documents WHERE category = 'graph') SELECT d.id FROM documents d JOIN recent r ON r.id = d.id`); err != nil || sessionCTE.Total != 1 {
		t.Fatalf("SQLSession generic CTE result=%#v err=%v", sessionCTE, err)
	}
	if sessionIN, err := session.Query("SELECT id FROM documents WHERE author_id IN (SELECT id FROM authors)"); err != nil || sessionIN.Total != 2 {
		t.Fatalf("SQLSession IN result=%#v err=%v", sessionIN, err)
	}

	in, err := db.Query(ctx, "SELECT id FROM documents WHERE author_id IN (SELECT id FROM authors) ORDER BY id")
	if err != nil {
		t.Fatal("IN subquery", err)
	}
	if in.Total != 2 || in.Results[0].ID != "d1" || in.Results[1].ID != "d2" {
		t.Fatalf("IN result=%#v", in)
	}

	exists, err := db.Query(ctx, "SELECT id FROM documents WHERE EXISTS (SELECT id FROM authors) ORDER BY id")
	if err != nil {
		t.Fatal("EXISTS subquery", err)
	}
	if exists.Total != 3 {
		t.Fatalf("EXISTS result=%#v", exists)
	}
}
