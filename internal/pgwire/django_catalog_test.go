package pgwire

import (
	"context"
	"testing"

	"github.com/xDarkicex/libravdb/internal/optimizer"
	"github.com/xDarkicex/libravdb/libravdb"
)

func TestAsyncpgTypeInfoProjection(t *testing.T) {
	params := optimizer.NewParameterSet(map[string]interface{}{"$1": "{25,16384,20,1028}"})
	results, columns, handled := interceptSystemQueryWithParams(`WITH RECURSIVE typeinfo_tree AS (SELECT typelem FROM pg_catalog.pg_type WHERE oid = any($1::oid[])) SELECT * FROM typeinfo_tree`, nil, params)
	if !handled || len(columns) != 14 || results == nil || len(results.Results) != 5 {
		t.Fatalf("handled=%v columns=%#v results=%#v", handled, columns, results)
	}
	var sawOIDArray bool
	for _, result := range results.Results {
		if result.Metadata["oid"] == nil || result.Metadata["name"] == nil {
			t.Fatalf("incomplete type row=%#v", result.Metadata)
		}
		if result.Metadata["oid"] == uint32(OIDOIDArray) || result.Metadata["oid"] == OIDOIDArray {
			sawOIDArray = true
			if result.Metadata["elemtype_name"] != "oid" {
				t.Fatalf("oid[] elemtype_name=%#v, want oid", result.Metadata["elemtype_name"])
			}
		}
	}
	if !sawOIDArray {
		t.Fatalf("type projection did not include oid[]: %#v", results.Results)
	}
}

func TestCatalogTargetTableDjangoRelkindPredicate(t *testing.T) {
	sql := `SELECT a.attname
		FROM pg_attribute a
		JOIN pg_class c ON a.attrelid = c.oid
		WHERE c.relkind IN ('f', 'm', 'p', 'r', 'v')
		  AND c.relname = 'django_agent_memory'`
	if got := catalogTargetTable(sql, nil); got != "django_agent_memory" {
		t.Fatalf("catalogTargetTable=%q, want django_agent_memory", got)
	}
}

func TestDjangoTableDescriptionProjection(t *testing.T) {
	db, err := libravdb.Open(libravdb.WithStoragePath(":memory:django_catalog_projection"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	_, err = db.Query(context.Background(), `CREATE TABLE django_agent_memory (id TEXT PRIMARY KEY, session_id VARCHAR(255) NOT NULL, payload JSONB NOT NULL)`)
	if err != nil {
		t.Fatalf("CREATE TABLE: %v", err)
	}
	sql := `SELECT a.attname AS column_name,
        NOT (a.attnotnull OR (t.typtype = 'd' AND t.typnotnull)) AS is_nullable,
        pg_get_expr(ad.adbin, ad.adrelid) AS column_default,
        CASE WHEN collname = 'default' THEN NULL ELSE collname END AS collation,
        a.attidentity != '' AS is_autofield,
        col_description(a.attrelid, a.attnum) AS column_comment
        FROM pg_attribute a
        LEFT JOIN pg_attrdef ad ON a.attrelid = ad.adrelid AND a.attnum = ad.adnum
        LEFT JOIN pg_collation co ON a.attcollation = co.oid
        JOIN pg_type t ON a.atttypid = t.oid
        JOIN pg_class c ON a.attrelid = c.oid
        JOIN pg_namespace n ON c.relnamespace = n.oid
        WHERE c.relkind IN ('f', 'm', 'p', 'r', 'v')
          AND c.relname = 'django_agent_memory'
          AND n.nspname NOT IN ('pg_catalog', 'pg_toast')
		  AND pg_table_is_visible(c.oid)`
	results, columns, handled := interceptSystemQueryWithParams(sql, db, nil)
	if !handled || len(columns) != 6 || results == nil || len(results.Results) != 3 {
		t.Fatalf("handled=%v columns=%#v results=%#v", handled, columns, results)
	}
}
