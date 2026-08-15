package pgwire

import (
	"context"
	"database/sql"
	"net"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"
	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/xDarkicex/libravdb/libravdb"
)

func TestPGWireArrayAggAndStringAgg(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(t.TempDir()+"/collection-aggregates"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	col, err := db.CreateCollection(ctx, "aggregate_rows", libravdb.WithMetadataOnly(), libravdb.WithMetadataSchema(libravdb.MetadataSchema{
		"category": libravdb.StringField,
		"name":     libravdb.StringField,
	}))
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range []struct {
		id, category string
		name         interface{}
	}{
		{"a1", "a", "Ada"},
		{"a2", "a", "Grace"},
		{"a3", "a", nil},
		{"b1", "b", "Linus"},
	} {
		if err := col.Insert(ctx, row.id, nil, map[string]interface{}{"category": row.category, "name": row.name}); err != nil {
			t.Fatal(err)
		}
	}
	graph, err := libravdb.NewGraph(libravdb.GraphConfig{})
	if err != nil {
		t.Fatal(err)
	}
	defer graph.Close()
	graphVectorCol, err := db.CreateCollection(ctx, "graph_vector_avg_rows", libravdb.WithDimension(3), libravdb.WithGraph(graph), libravdb.WithMetadataSchema(libravdb.MetadataSchema{
		"category": libravdb.StringField,
	}))
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range []struct {
		id     string
		vector []float32
	}{
		{"g1", []float32{1, 2, 3}},
		{"g2", []float32{3, 4, 5}},
	} {
		if err := graphVectorCol.Insert(ctx, row.id, row.vector, map[string]interface{}{"category": "g"}); err != nil {
			t.Fatal(err)
		}
	}
	vectorCol, err := db.CreateCollection(ctx, "vector_avg_rows", libravdb.WithDimension(3))
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range []struct {
		id     string
		vector []float32
	}{
		{"v1", []float32{1, 2, 3}},
		{"v2", []float32{3, 4, 5}},
	} {
		if err := vectorCol.Insert(ctx, row.id, row.vector, nil); err != nil {
			t.Fatal(err)
		}
	}

	srv := startTestServer(t, db)
	defer srv.Close()
	host, port, err := net.SplitHostPort(srv.Addr())
	if err != nil {
		t.Fatal(err)
	}
	sqlDB, err := sql.Open("pgx", "postgres://test:test@"+net.JoinHostPort(host, port)+"/test?sslmode=disable")
	if err != nil {
		t.Fatal(err)
	}
	defer sqlDB.Close()
	if err := sqlDB.PingContext(ctx); err != nil {
		t.Fatal(err)
	}

	var names string
	var joined string
	if err := sqlDB.QueryRowContext(ctx, "SELECT ARRAY_AGG(name), STRING_AGG(name, '|') FROM aggregate_rows").Scan(&names, &joined); err != nil {
		t.Fatalf("pgwire aggregate scan: %v", err)
	}
	if names != "{Ada,Grace,NULL,Linus}" || joined != "Ada|Grace|Linus" {
		t.Fatalf("pgwire aggregates names=%#v joined=%q", names, joined)
	}

	var centroid pgtype.FlatArray[float32]
	centroidScanner := pgtype.NewMap().SQLScanner(&centroid)
	if err := sqlDB.QueryRowContext(ctx, "SELECT VECTOR_AVG(embedding) AS centroid FROM vector_avg_rows").Scan(centroidScanner); err != nil {
		t.Fatalf("pgwire VECTOR_AVG scan: %v", err)
	}
	if len(centroid) != 3 || centroid[0] != 2 || centroid[1] != 3 || centroid[2] != 4 {
		t.Fatalf("pgwire VECTOR_AVG centroid=%v", centroid)
	}

	directConn, err := pgx.Connect(ctx, "postgres://test:test@"+net.JoinHostPort(host, port)+"/test?sslmode=disable")
	if err != nil {
		t.Fatal(err)
	}
	defer directConn.Close(ctx)
	var graphCentroid pgtype.FlatArray[float32]
	if err := directConn.QueryRow(ctx, "SELECT VECTOR_AVG(embedding) AS centroid FROM graph_vector_avg_rows").Scan(&graphCentroid); err != nil {
		t.Fatalf("direct pgx graph VECTOR_AVG scan: %v", err)
	}
	if len(graphCentroid) != 3 || graphCentroid[0] != 2 || graphCentroid[1] != 3 || graphCentroid[2] != 4 {
		t.Fatalf("direct pgx graph VECTOR_AVG centroid=%v", graphCentroid)
	}

	rows, err := sqlDB.QueryContext(ctx, "SELECT category, ARRAY_AGG(name), STRING_AGG(name, ',') FROM aggregate_rows GROUP BY category ORDER BY category")
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	if !rows.Next() {
		t.Fatalf("missing first grouped aggregate: %v", rows.Err())
	}
	var category string
	var grouped pgtype.FlatArray[*string]
	arrayScanner := pgtype.NewMap().SQLScanner(&grouped)
	if err := rows.Scan(&category, arrayScanner, &joined); err != nil {
		t.Fatalf("grouped aggregate scan: %v", err)
	}
	if category != "a" || len(grouped) != 3 || grouped[0] == nil || *grouped[0] != "Ada" || grouped[1] == nil || *grouped[1] != "Grace" || grouped[2] != nil || joined != "Ada,Grace" {
		t.Fatalf("grouped aggregate category=%q values=%#v joined=%q", category, grouped, joined)
	}
	if !rows.Next() {
		t.Fatal("missing second grouped aggregate")
	}
	if err := rows.Scan(&category, arrayScanner, &joined); err != nil {
		t.Fatalf("second grouped aggregate scan: %v", err)
	}
	if category != "b" || len(grouped) != 1 || grouped[0] == nil || *grouped[0] != "Linus" || joined != "Linus" {
		t.Fatalf("second grouped aggregate category=%q values=%#v joined=%q", category, grouped, joined)
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}

	groupedRows, err := sqlDB.QueryContext(ctx, "SELECT category, COUNT(*) AS category_count FROM aggregate_rows GROUP BY category ORDER BY category")
	if err != nil {
		t.Fatal(err)
	}
	defer groupedRows.Close()
	if !groupedRows.Next() {
		t.Fatalf("missing first COUNT aggregate: %v", groupedRows.Err())
	}
	var count int64
	if err := groupedRows.Scan(&category, &count); err != nil {
		t.Fatalf("grouped COUNT scan: %v", err)
	}
	if category != "a" || count != 3 {
		t.Fatalf("grouped COUNT first row=(%q,%d)", category, count)
	}
	if !groupedRows.Next() {
		t.Fatalf("missing second COUNT aggregate: %v", groupedRows.Err())
	}
	if err := groupedRows.Scan(&category, &count); err != nil {
		t.Fatalf("second grouped COUNT scan: %v", err)
	}
	if category != "b" || count != 1 {
		t.Fatalf("grouped COUNT second row=(%q,%d)", category, count)
	}
	if groupedRows.Next() || groupedRows.Err() != nil {
		t.Fatalf("unexpected grouped COUNT rows: err=%v", groupedRows.Err())
	}
}
