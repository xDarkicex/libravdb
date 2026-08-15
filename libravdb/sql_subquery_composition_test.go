package libravdb

import (
	"context"
	"testing"
)

func TestSQLSubqueryComposition(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:subquery-composition"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.CreateCollection(ctx, "authors", WithMetadataOnly(), WithMetadataSchema(MetadataSchema{"name": StringField, "score": IntField})); err != nil {
		t.Fatal(err)
	}
	if _, err := db.CreateCollection(ctx, "documents", WithMetadataOnly(), WithMetadataSchema(MetadataSchema{"author_id": StringField, "category": StringField})); err != nil {
		t.Fatal(err)
	}
	authors, _ := db.GetCollection("authors")
	documents, _ := db.GetCollection("documents")
	for _, row := range []struct {
		id    string
		name  string
		score int64
	}{{"a1", "Ada", 10}, {"a2", "Grace", 20}} {
		if err := authors.Insert(ctx, row.id, nil, map[string]interface{}{"name": row.name, "score": row.score}); err != nil {
			t.Fatal(err)
		}
	}
	for _, row := range []struct{ id, author, category string }{{"d1", "a1", "graph"}, {"d2", "a2", "vector"}, {"d3", "missing", "other"}} {
		if err := documents.Insert(ctx, row.id, nil, map[string]interface{}{"author_id": row.author, "category": row.category}); err != nil {
			t.Fatal(err)
		}
	}

	exists, err := db.Query(ctx, "SELECT d.id FROM documents d WHERE EXISTS (SELECT a.id FROM authors a WHERE a.id = d.author_id) ORDER BY d.id")
	if err != nil {
		t.Fatalf("correlated EXISTS: %v", err)
	}
	if exists.Total != 2 || exists.Results[0].ID != "d1" || exists.Results[1].ID != "d2" {
		t.Fatalf("correlated EXISTS result=%#v", exists)
	}

	in, err := db.Query(ctx, "SELECT d.id FROM documents d WHERE d.author_id IN (SELECT a.id FROM authors a WHERE a.name = 'Ada')")
	if err != nil {
		t.Fatalf("correlated IN: %v", err)
	}
	if in.Total != 1 || in.Results[0].ID != "d1" {
		t.Fatalf("correlated IN result=%#v", in)
	}

	scalar, err := db.Query(ctx, "SELECT d.id, (SELECT a.name FROM authors a WHERE a.id = d.author_id LIMIT 1) AS author_name FROM documents d ORDER BY d.id")
	if err != nil {
		t.Fatalf("scalar subquery: %v", err)
	}
	if scalar.Total != 3 || scalar.Results[0].Metadata["author_name"] != "Ada" || scalar.Results[2].Metadata["author_name"] != nil {
		t.Fatalf("scalar subquery result=%#v", scalar)
	}

	aggregateCount, err := db.Query(ctx, "SELECT d.id, (SELECT COUNT(*) FROM authors a WHERE a.id = d.author_id) AS author_count FROM documents d ORDER BY d.id")
	if err != nil {
		t.Fatalf("correlated aggregate COUNT: %v", err)
	}
	if aggregateCount.Total != 3 || aggregateCount.Results[0].Metadata["author_count"] != int64(1) || aggregateCount.Results[2].Metadata["author_count"] != int64(0) {
		t.Fatalf("correlated aggregate COUNT result=%#v", aggregateCount)
	}

	aggregateSum, err := db.Query(ctx, "SELECT d.id, (SELECT SUM(a.score) FROM authors a WHERE a.id = d.author_id) AS author_score FROM documents d ORDER BY d.id")
	if err != nil {
		t.Fatalf("correlated aggregate SUM: %v", err)
	}
	if aggregateSum.Total != 3 || aggregateSum.Results[0].Metadata["author_score"] != float64(10) || aggregateSum.Results[1].Metadata["author_score"] != float64(20) || aggregateSum.Results[2].Metadata["author_score"] != nil {
		t.Fatalf("correlated aggregate SUM result=%#v", aggregateSum)
	}

	aggregateAvg, err := db.Query(ctx, "SELECT (SELECT AVG(a.score) FROM authors a) AS average_score")
	if err != nil {
		t.Fatalf("uncorrelated aggregate AVG: %v", err)
	}
	if aggregateAvg.Total != 1 || aggregateAvg.Results[0].Metadata["average_score"] != float64(15) {
		t.Fatalf("uncorrelated aggregate AVG result=%#v", aggregateAvg)
	}
	aggregateExtrema, err := db.Query(ctx, "SELECT (SELECT MIN(a.score) FROM authors a) AS min_score, (SELECT MAX(a.score) FROM authors a) AS max_score")
	if err != nil {
		t.Fatalf("uncorrelated aggregate MIN/MAX: %v", err)
	}
	if aggregateExtrema.Total != 1 || aggregateExtrema.Results[0].Metadata["min_score"] != int64(10) || aggregateExtrema.Results[0].Metadata["max_score"] != int64(20) {
		t.Fatalf("uncorrelated aggregate MIN/MAX result=%#v", aggregateExtrema)
	}
	aggregateWhere, err := db.Query(ctx, "SELECT d.id FROM documents d WHERE d.author_id = (SELECT MIN(a.id) FROM authors a) ORDER BY d.id")
	if err != nil {
		t.Fatalf("aggregate scalar in WHERE: %v", err)
	}
	if aggregateWhere.Total != 1 || aggregateWhere.Results[0].ID != "d1" {
		ids := make([]string, 0, len(aggregateWhere.Results))
		for _, result := range aggregateWhere.Results {
			ids = append(ids, result.ID)
		}
		t.Fatalf("aggregate scalar in WHERE result=%v", ids)
	}
	numericAggregateWhere, err := db.Query(ctx, "SELECT a.id FROM authors a WHERE a.score > (SELECT AVG(x.score) FROM authors x) ORDER BY a.id")
	if err != nil {
		t.Fatalf("numeric aggregate scalar in WHERE: %v", err)
	}
	if numericAggregateWhere.Total != 1 || numericAggregateWhere.Results[0].ID != "a2" {
		t.Fatalf("numeric aggregate scalar in WHERE result=%#v", numericAggregateWhere)
	}
	aggregateHaving, err := db.Query(ctx, "SELECT d.author_id, COUNT(*) AS doc_count FROM documents d GROUP BY d.author_id HAVING COUNT(*) > (SELECT COUNT(*) FROM documents x WHERE x.id = 'does-not-exist') ORDER BY d.author_id")
	if err != nil {
		t.Fatalf("aggregate scalar in HAVING: %v", err)
	}
	if aggregateHaving.Total != 3 || aggregateHaving.Results[0].Metadata["doc_count"] != int64(1) {
		t.Fatalf("aggregate scalar in HAVING result=%#v", aggregateHaving)
	}
	if len(aggregateHaving.Columns) != 2 || aggregateHaving.Columns[0] != "author_id" || aggregateHaving.Columns[1] != "doc_count" {
		t.Fatalf("aggregate scalar in HAVING columns=%v", aggregateHaving.Columns)
	}

	derived, err := db.Query(ctx, "SELECT r.id, r.category FROM (SELECT id, category FROM documents WHERE category = 'graph') AS r ORDER BY r.id")
	if err != nil {
		t.Fatalf("derived table: %v", err)
	}
	if derived.Total != 1 || derived.Results[0].ID != "d1" || derived.Results[0].Metadata["category"] != "graph" {
		t.Fatalf("derived table result=%#v", derived)
	}

	joined, err := db.Query(ctx, "SELECT d.id, r.category FROM documents d JOIN (SELECT id, category FROM documents WHERE category = 'graph') r ON d.id = r.id")
	if err != nil {
		t.Fatalf("derived join: %v", err)
	}
	if joined.Total != 1 || joined.Results[0].ID != "d1" {
		t.Fatalf("derived join result=%#v", joined)
	}

	// A derived table may reference the current row, and that scope must
	// survive another derived SELECT nested inside it.  This is the
	// multi-level correlation shape that ordinary SQL clients generate when
	// composing query builders.
	nestedScalar, err := db.Query(ctx, "SELECT d.id, (SELECT y.name FROM (SELECT x.name FROM (SELECT a.name FROM authors a WHERE a.id = d.author_id) x) y LIMIT 1) AS nested_author FROM documents d ORDER BY d.id")
	if err != nil {
		t.Fatalf("deeply correlated scalar derived table: %v", err)
	}
	if nestedScalar.Total != 3 || nestedScalar.Results[0].Metadata["nested_author"] != "Ada" || nestedScalar.Results[1].Metadata["nested_author"] != "Grace" || nestedScalar.Results[2].Metadata["nested_author"] != nil {
		t.Fatalf("deeply correlated scalar derived table result=%#v", nestedScalar)
	}

	nestedJoin, err := db.Query(ctx, "SELECT d.id, x.name FROM documents d JOIN (SELECT y.name, y.id FROM (SELECT a.name, a.id FROM authors a WHERE a.id = d.author_id) y) x ON x.id = d.author_id ORDER BY d.id")
	if err != nil {
		t.Fatalf("deeply correlated derived join: %v", err)
	}
	if nestedJoin.Total != 2 || nestedJoin.Results[0].ID != "d1" || nestedJoin.Results[0].Metadata["name"] != "Ada" || nestedJoin.Results[1].ID != "d2" || nestedJoin.Results[1].Metadata["name"] != "Grace" {
		t.Fatalf("deeply correlated derived join result=%#v", nestedJoin)
	}
}
