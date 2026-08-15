package libravdb

import (
	"context"
	"testing"

	"github.com/xDarkicex/libravdb/internal/util"
)

func TestSQLJSONOperators(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:json_sql"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	if _, err := db.Query(ctx, "CREATE TABLE json_docs (id TEXT PRIMARY KEY, payload JSONB)"); err != nil {
		t.Fatalf("create JSONB table: %v", err)
	}
	if _, err := db.Query(ctx, `INSERT INTO json_docs (id, payload) VALUES
		('d1', '{"name":"Ada","roles":["admin","editor"],"profile":{"active":true}}'),
		('d2', '{"name":"Grace","roles":["viewer"],"profile":{"active":false}}')`); err != nil {
		t.Fatalf("insert JSON documents: %v", err)
	}
	if _, err := db.Query(ctx, `CREATE INDEX json_docs_profile_active_idx
		ON json_docs (payload#>>'{profile,active}')`); err != nil {
		t.Fatalf("create JSON path index: %v", err)
	}
	indexed, err := db.Query(ctx, `SELECT id FROM json_docs
		WHERE payload#>>'{profile,active}' = 'true' ORDER BY id`)
	if err != nil {
		t.Fatalf("indexed JSON path predicate: %v", err)
	}
	if indexed.Total != 1 || indexed.Results[0].ID != "d1" {
		t.Fatalf("indexed JSON path result: %#v", indexed)
	}
	col, err := db.GetCollection("json_docs")
	if err != nil || col == nil || len(col.jsonIndex) == 0 {
		t.Fatalf("JSON inverted postings were not built: err=%v", err)
	}
	if _, err := db.Query(ctx, `DROP INDEX json_docs_profile_active_idx`); err != nil {
		t.Fatalf("drop JSON path index: %v", err)
	}
	if len(col.Config().JSONIndexes) != 0 {
		t.Fatalf("dropped JSON index remained in collection config: %#v", col.Config().JSONIndexes)
	}

	name, err := db.Query(ctx, `SELECT payload->>'name' AS name FROM json_docs WHERE id = 'd1'`)
	if err != nil {
		t.Fatalf("text extraction: %v", err)
	}
	if name.Total != 1 || name.Results[0].Metadata["name"] != "Ada" {
		t.Fatalf("text extraction result: %#v", name)
	}

	nested, err := db.Query(ctx, `SELECT payload->'profile'->>'active' AS active FROM json_docs WHERE id = 'd1'`)
	if err != nil {
		t.Fatalf("nested extraction: %v", err)
	}
	if nested.Total != 1 || nested.Results[0].Metadata["active"] != "true" {
		t.Fatalf("nested extraction result: %#v", nested)
	}

	pathValue, err := db.Query(ctx, `SELECT payload#>'{profile,active}' AS profile FROM json_docs WHERE id = 'd1'`)
	if err != nil {
		t.Fatalf("path extraction: %v", err)
	}
	if pathValue.Total != 1 || pathValue.Results[0].Metadata["profile"] != true {
		t.Fatalf("path extraction result: %#v", pathValue)
	}
	pathText, err := db.Query(ctx, `SELECT payload#>>'{profile,active}' AS active FROM json_docs WHERE id = 'd1'`)
	if err != nil {
		t.Fatalf("path text extraction: %v", err)
	}
	if pathText.Total != 1 || pathText.Results[0].Metadata["active"] != "true" {
		t.Fatalf("path text extraction result: %#v", pathText)
	}

	exists, err := db.Query(ctx, `SELECT id FROM json_docs WHERE payload ? 'name' ORDER BY id`)
	if err != nil {
		t.Fatalf("JSON key existence: %v", err)
	}
	if exists.Total != 2 || exists.Results[0].ID != "d1" || exists.Results[1].ID != "d2" {
		t.Fatalf("JSON key existence result: %#v", exists)
	}

	contains, err := db.Query(ctx, `SELECT id FROM json_docs WHERE payload @> '{"roles":["admin"]}' ORDER BY id`)
	if err != nil {
		t.Fatalf("containment: %v", err)
	}
	if contains.Total != 1 || contains.Results[0].ID != "d1" {
		t.Fatalf("containment result: %#v", contains)
	}
	if len(col.jsonContainmentIndex) == 0 || len(col.jsonContainmentIndex["payload"]) == 0 {
		t.Fatalf("JSON containment postings were not built: %#v", col.jsonContainmentIndex)
	}
	parameterized, err := db.QueryWithParams(ctx,
		`SELECT id FROM json_docs WHERE payload @> $needle ORDER BY id`,
		QueryParams{"needle": `{"roles":["admin"]}`})
	if err != nil {
		t.Fatalf("parameterized containment: %v", err)
	}
	if parameterized.Total != 1 || parameterized.Results[0].ID != "d1" {
		t.Fatalf("parameterized containment result: %#v", parameterized)
	}

	containedBy, err := db.Query(ctx, `SELECT id FROM json_docs WHERE '{"name":"Ada"}' <@ payload ORDER BY id`)
	if err != nil {
		t.Fatalf("contained-by: %v", err)
	}
	if containedBy.Total != 1 || containedBy.Results[0].ID != "d1" {
		t.Fatalf("contained-by result: %#v", containedBy)
	}
	anyKeys, err := db.Query(ctx, `SELECT id FROM json_docs WHERE payload ?| '{name,missing}' ORDER BY id`)
	if err != nil {
		t.Fatalf("JSON any-key existence: %v", err)
	}
	if anyKeys.Total != 2 {
		t.Fatalf("JSON any-key existence result: %#v", anyKeys)
	}
	allKeys, err := db.Query(ctx, `SELECT id FROM json_docs WHERE payload ?& '{name,profile}' ORDER BY id`)
	if err != nil {
		t.Fatalf("JSON all-key existence: %v", err)
	}
	if allKeys.Total != 2 {
		t.Fatalf("JSON all-key existence result: %#v", allKeys)
	}
	arrayAny, err := db.Query(ctx, `SELECT id FROM json_docs WHERE payload ?| ARRAY['name','missing'] ORDER BY id`)
	if err != nil {
		t.Fatalf("JSON ARRAY ?| existence: %v", err)
	}
	if arrayAny.Total != 2 {
		t.Fatalf("JSON ARRAY ?| result: %#v", arrayAny)
	}
	patched, err := db.Query(ctx, `SELECT jsonb_set(payload, '{profile,active}', 'false') AS patched
		FROM json_docs WHERE id = 'd1'`)
	if err != nil {
		t.Fatalf("jsonb_set: %v", err)
	}
	if patched.Total != 1 {
		t.Fatalf("jsonb_set result: %#v", patched)
	}
	patchedProfile, ok := patched.Results[0].Metadata["patched"].(map[string]interface{})
	if !ok || patchedProfile["profile"].(map[string]interface{})["active"] != false {
		t.Fatalf("jsonb_set value: %#v", patched.Results[0].Metadata["patched"])
	}
	arrayLength, err := db.Query(ctx, `SELECT jsonb_array_length(payload->'roles') AS n FROM json_docs WHERE id = 'd1'`)
	if err != nil {
		t.Fatalf("jsonb_array_length: %v", err)
	}
	if arrayLength.Total != 1 || arrayLength.Results[0].Metadata["n"] != int64(2) {
		t.Fatalf("jsonb_array_length result: %#v", arrayLength)
	}
	typeResult, err := db.Query(ctx, `SELECT jsonb_typeof(payload->'profile') AS kind FROM json_docs WHERE id = 'd1'`)
	if err != nil {
		t.Fatalf("jsonb_typeof: %v", err)
	}
	if typeResult.Total != 1 || typeResult.Results[0].Metadata["kind"] != "object" {
		t.Fatalf("jsonb_typeof result: %#v", typeResult)
	}

	if _, err := db.Query(ctx, `INSERT INTO json_docs (id, payload) VALUES ('bad', 'not-json')`); err == nil {
		t.Fatal("invalid JSON was accepted")
	}
}

func TestSQLJSONSurvivesReopen(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/json_reopen.libravdb"
	db, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, "CREATE TABLE json_reopen (id TEXT PRIMARY KEY, payload JSON)"); err != nil {
		t.Fatalf("create JSON table: %v", err)
	}
	if _, err := db.Query(ctx, `INSERT INTO json_reopen (id, payload) VALUES ('d1', '{"ok":true}')`); err != nil {
		t.Fatalf("insert JSON row: %v", err)
	}
	if _, err := db.Query(ctx, `CREATE INDEX json_reopen_ok_idx
		ON json_reopen (payload#>>'{ok}')`); err != nil {
		t.Fatalf("create JSON reopen index: %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	reopened, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	result, err := reopened.Query(ctx, `SELECT payload->>'ok' AS ok FROM json_reopen WHERE id = 'd1'`)
	if err != nil {
		t.Fatalf("query after reopen: %v", err)
	}
	if result.Total != 1 || result.Results[0].Metadata["ok"] != "true" {
		t.Fatalf("JSON after reopen: %#v", result)
	}
	indexed, err := reopened.Query(ctx, `SELECT id FROM json_reopen
		WHERE payload#>>'{ok}' = 'true'`)
	if err != nil {
		t.Fatalf("JSON index after reopen: %v", err)
	}
	if indexed.Total != 1 || indexed.Results[0].ID != "d1" {
		t.Fatalf("JSON index after reopen result: %#v", indexed)
	}
	reopenedCol, err := reopened.GetCollection("json_reopen")
	if err != nil || reopenedCol == nil || len(reopenedCol.jsonIndex) == 0 {
		t.Fatalf("JSON inverted postings were not rebuilt after reopen: err=%v", err)
	}
	if _, err := reopened.Query(ctx, `DROP INDEX json_reopen_ok_idx`); err != nil {
		t.Fatalf("drop reopened JSON index: %v", err)
	}
	if _, err := reopened.Query(ctx, `CREATE INDEX json_reopen_ok_idx
		ON json_reopen (payload#>>'{ok}')`); err != nil {
		t.Fatalf("recreate reopened JSON index: %v", err)
	}
	if _, err := reopened.Query(ctx, `SELECT id FROM json_reopen
		WHERE payload#>>'{ok}' = 'true'`); err != nil {
		t.Fatalf("query after JSON index recreate: %v", err)
	}
}

func TestJSONBCanonicalNumericComparison(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:json_canonical"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Query(ctx, "CREATE TABLE json_canonical (id TEXT PRIMARY KEY, payload JSONB)"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, `INSERT INTO json_canonical (id, payload) VALUES
		('integer', '{"n":1}'), ('decimal', '{"n":1.0,"object":{"b":2,"a":1}}')`); err != nil {
		t.Fatal(err)
	}
	result, err := db.Query(ctx, `SELECT id FROM json_canonical WHERE payload @> '{"n":1}' ORDER BY id`)
	if err != nil {
		t.Fatal(err)
	}
	if result.Total != 2 || result.Results[0].ID != "decimal" || result.Results[1].ID != "integer" {
		ids := make([]string, 0, len(result.Results))
		for _, row := range result.Results {
			ids = append(ids, row.ID)
		}
		t.Fatalf("canonical JSONB numeric comparison: ids=%v result=%#v", ids, result)
	}
	object, err := db.Query(ctx, `SELECT id FROM json_canonical WHERE payload @> '{"object":{"a":1}}'`)
	if err != nil {
		t.Fatal(err)
	}
	if object.Total != 1 || object.Results[0].ID != "decimal" {
		t.Fatalf("canonical JSONB object comparison: %#v", object)
	}
}

func TestSQLJSONPathPredicates(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:jsonpath"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Query(ctx, "CREATE TABLE jsonpath_docs (id TEXT PRIMARY KEY, payload JSONB)"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, `INSERT INTO jsonpath_docs (id, payload) VALUES
		('d1', '{"profile":{"active":true},"scores":[0,2,4]}'),
		('d2', '{"profile":{"active":false},"scores":[0,1]}')`); err != nil {
		t.Fatal(err)
	}
	active, err := db.Query(ctx, `SELECT id FROM jsonpath_docs WHERE payload @? '$.profile.active' ORDER BY id`)
	if err != nil {
		t.Fatalf("JSONPath existence: %v", err)
	}
	if active.Total != 2 {
		t.Fatalf("JSONPath existence result: %#v", active)
	}
	truth, err := db.Query(ctx, `SELECT id FROM jsonpath_docs WHERE payload @@ '$.profile.active' ORDER BY id`)
	if err != nil {
		t.Fatalf("JSONPath boolean predicate: %v", err)
	}
	if truth.Total != 1 || truth.Results[0].ID != "d1" {
		t.Fatalf("JSONPath boolean result: %#v", truth)
	}
	filtered, err := db.Query(ctx, `SELECT id FROM jsonpath_docs WHERE payload @? '$.scores[*] ? (@ > 1)' ORDER BY id`)
	if err != nil {
		t.Fatalf("JSONPath filter predicate: %v", err)
	}
	if filtered.Total != 1 || filtered.Results[0].ID != "d1" {
		t.Fatalf("JSONPath filter result: %#v", filtered)
	}
	scalar, err := db.Query(ctx, `SELECT id FROM jsonpath_docs WHERE payload @@ '$.profile.active == true'`)
	if err != nil || scalar.Total != 1 || scalar.Results[0].ID != "d1" {
		t.Fatalf("JSONPath scalar comparison: result=%#v err=%v", scalar, err)
	}
	nested, err := db.Query(ctx, `SELECT id FROM jsonpath_docs WHERE payload @? '$.profile ? (@.active == true)'`)
	if err != nil || nested.Total != 1 || nested.Results[0].ID != "d1" {
		t.Fatalf("JSONPath nested filter: result=%#v err=%v", nested, err)
	}
	recursive, err := db.Query(ctx, `SELECT id FROM jsonpath_docs WHERE payload @? '$.**.active' ORDER BY id`)
	if err != nil || recursive.Total != 2 {
		t.Fatalf("JSONPath recursive descent: result=%#v err=%v", recursive, err)
	}
	rangeResult, err := db.Query(ctx, `SELECT id FROM jsonpath_docs WHERE payload @? '$.scores[1 to 2] ? (@ > 1)'`)
	if err != nil || rangeResult.Total != 1 || rangeResult.Results[0].ID != "d1" {
		t.Fatalf("JSONPath range selector: result=%#v err=%v", rangeResult, err)
	}
	typeFilter, err := db.Query(ctx, `SELECT id FROM jsonpath_docs WHERE payload @? '$.profile ? (@.active.type() == "boolean")'`)
	if err != nil || typeFilter.Total != 2 {
		t.Fatalf("JSONPath type() filter: result=%#v err=%v", typeFilter, err)
	}
	strictMissing, err := db.Query(ctx, `SELECT id FROM jsonpath_docs WHERE payload @@ 'strict $.profile.missing'`)
	if err == nil || strictMissing != nil {
		t.Fatalf("strict JSONPath missing step: result=%#v err=%v", strictMissing, err)
	}
}

func TestSQLJSONArrayExpansion(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:json_expand"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	rows, err := db.Query(ctx, `SELECT elem FROM jsonb_array_elements('[1,2,3]'::jsonb) AS elem`)
	if err != nil {
		t.Fatalf("jsonb_array_elements: %v", err)
	}
	if rows.Total != 3 {
		t.Fatalf("jsonb_array_elements rows: %#v", rows)
	}
	if rows.Results[0].Metadata["elem"] != int64(1) || rows.Results[2].Metadata["elem"] != int64(3) {
		t.Fatalf("jsonb_array_elements values: %#v", rows)
	}
	textRows, err := db.Query(ctx, `SELECT item FROM jsonb_array_elements_text('["a","b"]'::jsonb) AS item`)
	if err != nil {
		t.Fatalf("jsonb_array_elements_text: %v", err)
	}
	if textRows.Total != 2 || textRows.Results[1].Metadata["item"] != "b" {
		t.Fatalf("jsonb_array_elements_text values: %#v", textRows)
	}
	objectRows, err := db.Query(ctx, `SELECT key, value FROM jsonb_each('{"b":2,"a":1}'::jsonb) AS e`)
	if err != nil || objectRows.Total != 2 || objectRows.Results[0].Metadata["key"] != "a" {
		t.Fatalf("jsonb_each values: result=%#v err=%v", objectRows, err)
	}
	if _, err := db.Query(ctx, `CREATE TABLE json_expand_rows (id TEXT PRIMARY KEY, payload JSONB)`); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, `INSERT INTO json_expand_rows (id, payload) VALUES ('d1', '[1,2]'::jsonb)`); err != nil {
		t.Fatal(err)
	}
	lateral, err := db.Query(ctx, `SELECT elem FROM json_expand_rows d CROSS JOIN jsonb_array_elements(d.payload) AS elem`)
	if err != nil || lateral.Total != 2 {
		t.Fatalf("lateral JSON expansion: result=%#v err=%v", lateral, err)
	}
}

func TestSQLJSONMutationConstructionAndRecords(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:json_mutation"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Query(ctx, "CREATE TABLE json_mutation (id TEXT PRIMARY KEY, payload JSONB)"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, `INSERT INTO json_mutation (id, payload) VALUES ('d1', '{"a":1,"arr":[1,3],"nested":{"drop":true}}')`); err != nil {
		t.Fatal(err)
	}

	inserted, err := db.Query(ctx, `SELECT jsonb_insert(payload, '{arr,1}', '2') AS value FROM json_mutation WHERE id = 'd1'`)
	if err != nil || inserted.Total != 1 {
		t.Fatalf("jsonb_insert: result=%#v err=%v", inserted, err)
	}
	arr := inserted.Results[0].Metadata["value"].(map[string]interface{})["arr"].([]interface{})
	if len(arr) != 3 || arr[1] != int64(2) {
		t.Fatalf("jsonb_insert value: %#v", inserted.Results[0].Metadata["value"])
	}

	deleted, err := db.Query(ctx, `SELECT payload #- '{nested,drop}' AS value FROM json_mutation WHERE id = 'd1'`)
	if err != nil || deleted.Total != 1 {
		t.Fatalf("JSON #-: result=%#v err=%v", deleted, err)
	}
	nested := deleted.Results[0].Metadata["value"].(map[string]interface{})["nested"].(map[string]interface{})
	if len(nested) != 0 {
		t.Fatalf("JSON #- value: %#v", deleted.Results[0].Metadata["value"])
	}

	merged, err := db.Query(ctx, `SELECT payload || '{"b":2}' AS value FROM json_mutation WHERE id = 'd1'`)
	if err != nil || merged.Total != 1 {
		t.Fatalf("JSON ||: result=%#v err=%v", merged, err)
	}
	if merged.Results[0].Metadata["value"].(map[string]interface{})["b"] != int64(2) {
		t.Fatalf("JSON || value: %#v", merged.Results[0].Metadata["value"])
	}

	built, err := db.Query(ctx, `SELECT jsonb_build_object('name', 'Ada', 'n', 2) AS object, jsonb_build_array('a', 2, true) AS array`)
	if err != nil || built.Total != 1 {
		t.Fatalf("JSON constructors: result=%#v err=%v", built, err)
	}
	object := built.Results[0].Metadata["object"].(map[string]interface{})
	if object["name"] != "Ada" || object["n"] != int64(2) {
		t.Fatalf("JSON object constructor: %#v", object)
	}
	encoded, err := db.Query(ctx, `SELECT to_jsonb('Ada') AS value`)
	if err != nil || encoded.Total != 1 || encoded.Results[0].Metadata["value"] != "Ada" {
		t.Fatalf("to_jsonb: result=%#v err=%v", encoded, err)
	}

	record, err := db.Query(ctx, `SELECT r.name, r.age FROM jsonb_to_record('{"name":"Ada","age":37}'::jsonb) AS r`)
	if err != nil || record.Total != 1 {
		t.Fatalf("jsonb_to_record: result=%#v err=%v", record, err)
	}
	if record.Results[0].Metadata["name"] != "Ada" || record.Results[0].Metadata["age"] != int64(37) {
		t.Fatalf("jsonb_to_record values: %#v", record.Results[0].Metadata)
	}

	recordset, err := db.Query(ctx, `SELECT r.name FROM jsonb_to_recordset('[{"name":"Ada"},{"name":"Grace"}]'::jsonb) AS r ORDER BY r.name`)
	if err != nil || recordset.Total != 2 || recordset.Results[0].Metadata["name"] != "Ada" {
		t.Fatalf("jsonb_to_recordset: result=%#v err=%v", recordset, err)
	}
	populated, err := db.Query(ctx, `SELECT r.name FROM jsonb_populate_recordset('{"name":"Unknown"}'::jsonb, '[{"name":"Ada"},{"name":"Grace"}]'::jsonb) AS r ORDER BY r.name`)
	if err != nil || populated.Total != 2 || populated.Results[0].Metadata["name"] != "Ada" {
		t.Fatalf("jsonb_populate_recordset: result=%#v err=%v", populated, err)
	}
}

func TestSQLJSONNullSemanticsAndPersistence(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/json_null"
	db, err := Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, "CREATE TABLE json_null (id TEXT PRIMARY KEY, payload JSONB, nullable JSONB)"); err != nil {
		db.Close()
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, `INSERT INTO json_null (id, payload, nullable)
		VALUES ('d1', '{"present":null,"nested":{"value":null}}', NULL)`); err != nil {
		db.Close()
		t.Fatalf("insert JSON/SQL nulls: %v", err)
	}
	result, err := db.Query(ctx, `SELECT
		jsonb_typeof(payload->'present') AS json_type,
		payload ? 'present' AS present,
		payload @? '$.present' AS path_present,
		payload->'present' AS extracted,
		nullable AS sql_value
		FROM json_null WHERE id = 'd1'`)
	if err != nil || result.Total != 1 {
		db.Close()
		t.Fatalf("query JSON/SQL nulls: result=%#v err=%v", result, err)
	}
	row := result.Results[0].Metadata
	if row["json_type"] != "null" || row["present"] != true || row["path_present"] != true {
		t.Fatalf("JSON null semantics: %#v", row)
	}
	if _, ok := row["extracted"].(util.JSONNull); !ok {
		t.Fatalf("JSON null extraction lost sentinel: %#v (%T)", row["extracted"], row["extracted"])
	}
	if value, exists := row["sql_value"]; !exists || value != nil {
		t.Fatalf("SQL NULL was not preserved: %#v", row)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	db, err = Open(WithStoragePath(path), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	reopened, err := db.Query(ctx, `SELECT jsonb_typeof(payload->'present') AS json_type,
		payload->'present' AS extracted, nullable AS sql_value FROM json_null WHERE id = 'd1'`)
	if err != nil || reopened.Total != 1 {
		t.Fatalf("reopen JSON/SQL nulls: result=%#v err=%v", reopened, err)
	}
	reopenedRow := reopened.Results[0].Metadata
	if reopenedRow["json_type"] != "null" {
		t.Fatalf("reopen JSON null type: %#v", reopenedRow)
	}
	if _, ok := reopenedRow["extracted"].(util.JSONNull); !ok {
		t.Fatalf("reopen JSON null sentinel lost: %#v (%T)", reopenedRow["extracted"], reopenedRow["extracted"])
	}
	if value, exists := reopenedRow["sql_value"]; !exists || value != nil {
		t.Fatalf("reopen SQL NULL was not preserved: %#v", reopenedRow)
	}
}
