package pgwire

import (
	"testing"

	"github.com/xDarkicex/lexer/parser"
)

func TestPgwireParameterAnalysisForDialectCatalogQuery(t *testing.T) {
	limitDoc := &parser.QueryDoc{}
	if err := parser.Parse([]byte(`SELECT * FROM "gorm_catalog_probe" LIMIT 5`), limitDoc); err != nil {
		t.Fatalf("quoted table LIMIT parse: %v", err)
	}
	paramLimitDoc := &parser.QueryDoc{}
	if err := parser.Parse([]byte(`SELECT * FROM "gorm_catalog_probe" LIMIT $1`), paramLimitDoc); err != nil {
		t.Fatalf("quoted table parameter LIMIT parse: %v", err)
	}
	quotedLimitDoc := &parser.QueryDoc{}
	if err := parser.Parse([]byte(`SELECT * FROM "gorm_catalog_probe" LIMIT '5'`), quotedLimitDoc); err != nil {
		t.Fatalf("quoted numeric LIMIT parse: %v", err)
	}
	qualifiedOrderDoc := &parser.QueryDoc{}
	if err := parser.Parse([]byte(`SELECT * FROM "gorm_catalog_probe" WHERE id = 'gorm-1' ORDER BY "gorm_catalog_probe"."id" LIMIT 1`), qualifiedOrderDoc); err != nil {
		t.Fatalf("quoted qualified ORDER BY parse: %v", err)
	}
	q := "SELECT c.column_name, c.is_nullable = 'YES', c.udt_name, c.character_maximum_length, c.numeric_precision, c.numeric_precision_radix, c.numeric_scale, c.datetime_precision, 8 * typlen, c.column_default, pd.description, c.identity_increment FROM information_schema.columns AS c JOIN pg_type AS pgt ON c.udt_name = pgt.typname LEFT JOIN pg_catalog.pg_description as pd ON pd.objsubid = c.ordinal_position AND pd.objoid = (SELECT oid FROM pg_catalog.pg_class WHERE relname = c.table_name AND relnamespace = (SELECT oid FROM pg_catalog.pg_namespace WHERE nspname = c.table_schema)) where table_catalog = $1 AND table_schema = CURRENT_SCHEMA() AND table_name = $2"
	info := analyzeParams(q)
	if info.total() != 2 {
		t.Fatalf("parameter analysis = %#v, want 2", info)
	}
}
