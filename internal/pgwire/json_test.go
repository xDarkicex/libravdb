package pgwire

import (
	"context"
	"database/sql"
	"net"
	"testing"

	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/xDarkicex/libravdb/libravdb"
)

func seedJSONSQLDB(t *testing.T) *libravdb.Database {
	t.Helper()
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(t.TempDir()+"/json_pgwire"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if _, err := db.Query(ctx, "CREATE TABLE json_docs (id TEXT PRIMARY KEY, payload JSONB)"); err != nil {
		db.Close()
		t.Fatalf("CREATE TABLE: %v", err)
	}
	if _, err := db.Query(ctx, `INSERT INTO json_docs (id, payload) VALUES
		('d1', '{"name":"Ada","roles":["admin","editor"],"profile":{"active":true}}'),
		('d2', '{"name":"Grace","roles":["viewer"],"profile":{"active":false}}')`); err != nil {
		db.Close()
		t.Fatalf("INSERT: %v", err)
	}
	return db
}

func TestPostgreSQLJSON_SimpleQuery(t *testing.T) {
	db := seedJSONSQLDB(t)
	defer db.Close()

	srv := startTestServer(t, db)
	defer srv.Close()
	conn := dialTestServer(t, srv)
	defer conn.Close()
	doTestStartup(t, conn)

	sendSimpleQuery(conn, `SELECT payload->>'name' AS name, payload->'profile' AS profile
		FROM json_docs WHERE id = 'd1'`)
	msgType, payload, err := ReadMessage(conn)
	if err != nil {
		t.Fatalf("RowDescription: %v", err)
	}
	if msgType != msgRowDescription {
		t.Fatalf("first response: got %q, want RowDescription", msgType)
	}
	cols := decodeRowDescription(t, payload)
	if len(cols) != 2 || cols[0].Name != "name" || cols[0].TypeOID != OIDText || cols[1].Name != "profile" || cols[1].TypeOID != OIDJSONB {
		t.Fatalf("JSON RowDescription: %#v", cols)
	}
	rows := readDataRowsUntilComplete(t, conn)
	if len(rows) != 1 || rows[0][0] != "Ada" || rows[0][1] != `{"active":true}` {
		t.Fatalf("JSON DataRows: %#v", rows)
	}
	consumeReadyForQuery(t, conn)
}

func TestPGWireJSONBParameterizedContainment(t *testing.T) {
	ctx := context.Background()
	db := seedJSONSQLDB(t)
	defer db.Close()

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
	rows, err := sqlDB.QueryContext(ctx, `SELECT id FROM json_docs WHERE payload @> $1 ORDER BY id`, `{"roles":["admin"]}`)
	if err != nil {
		t.Fatalf("parameterized JSONB containment: %v", err)
	}
	defer rows.Close()
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			t.Fatal(err)
		}
		ids = append(ids, id)
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if len(ids) != 1 || ids[0] != "d1" {
		t.Fatalf("JSONB containment ids=%v", ids)
	}
}

func TestPGWireJSONPathAndKeyParameterized(t *testing.T) {
	ctx := context.Background()
	db := seedJSONSQLDB(t)
	defer db.Close()
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
	var active string
	var hasName bool
	if err := sqlDB.QueryRowContext(ctx,
		`SELECT payload#>>$1, payload ? $2 FROM json_docs WHERE id = 'd1'`,
		`{profile,active}`, "name").Scan(&active, &hasName); err != nil {
		t.Fatalf("parameterized JSON path/key query: %v", err)
	}
	if active != "true" || !hasName {
		t.Fatalf("JSON path/key values: active=%q hasName=%v", active, hasName)
	}
}

func TestPostgreSQLJSON_KeySetOperators(t *testing.T) {
	db := seedJSONSQLDB(t)
	defer db.Close()
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
	ctx := context.Background()
	if err := sqlDB.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	rows, err := sqlDB.QueryContext(ctx, `SELECT id FROM json_docs WHERE payload ?| '{name,missing}' ORDER BY id`)
	if err != nil {
		t.Fatalf("JSON ?| query: %v", err)
	}
	var anyIDs []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			t.Fatal(err)
		}
		anyIDs = append(anyIDs, id)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if len(anyIDs) != 2 || anyIDs[0] != "d1" || anyIDs[1] != "d2" {
		t.Fatalf("JSON ?| ids=%v", anyIDs)
	}
	var arrayAnyIDs []string
	rows, err = sqlDB.QueryContext(ctx, `SELECT id FROM json_docs WHERE payload ?| ARRAY['name','missing'] ORDER BY id`)
	if err != nil {
		t.Fatalf("JSON ARRAY ?| query: %v", err)
	}
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			t.Fatal(err)
		}
		arrayAnyIDs = append(arrayAnyIDs, id)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if len(arrayAnyIDs) != 2 || arrayAnyIDs[0] != "d1" || arrayAnyIDs[1] != "d2" {
		t.Fatalf("JSON ARRAY ?| ids=%v", arrayAnyIDs)
	}
	rows, err = sqlDB.QueryContext(ctx, `SELECT id FROM json_docs WHERE payload ?& '{name,profile}' ORDER BY id`)
	if err != nil {
		t.Fatalf("JSON ?& query: %v", err)
	}
	var allIDs []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			t.Fatal(err)
		}
		allIDs = append(allIDs, id)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if len(allIDs) != 2 || allIDs[0] != "d1" || allIDs[1] != "d2" {
		t.Fatalf("JSON ?& ids=%v", allIDs)
	}
}

func TestPostgreSQLJSONPathAndArrayExpansion(t *testing.T) {
	db := seedJSONSQLDB(t)
	defer db.Close()
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
	ctx := context.Background()
	if err := sqlDB.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	rows, err := sqlDB.QueryContext(ctx, `SELECT id FROM json_docs WHERE payload @? '$.profile.active' ORDER BY id`)
	if err != nil {
		t.Fatalf("JSONPath @? query: %v", err)
	}
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			t.Fatal(err)
		}
		ids = append(ids, id)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if len(ids) != 2 || ids[0] != "d1" || ids[1] != "d2" {
		t.Fatalf("JSONPath ids=%v", ids)
	}
	var active bool
	if err := sqlDB.QueryRowContext(ctx, `SELECT payload @@ '$.profile.active == true' FROM json_docs WHERE id = 'd1'`).Scan(&active); err != nil {
		t.Fatalf("JSONPath @@ query: %v", err)
	}
	if !active {
		t.Fatal("JSONPath @@ returned false for active document")
	}
	rows, err = sqlDB.QueryContext(ctx, `SELECT id FROM json_docs WHERE payload @? $1 ORDER BY id`, "$.profile.active")
	if err != nil {
		t.Fatalf("parameterized JSONPath @? query: %v", err)
	}
	var parameterizedIDs []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			t.Fatal(err)
		}
		parameterizedIDs = append(parameterizedIDs, id)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if len(parameterizedIDs) != 2 {
		t.Fatalf("parameterized JSONPath ids=%v", parameterizedIDs)
	}
	rows, err = sqlDB.QueryContext(ctx, `SELECT item FROM jsonb_array_elements_text('["a","b"]'::jsonb) AS item`)
	if err != nil {
		t.Fatalf("JSON array expansion query: %v", err)
	}
	var items []string
	for rows.Next() {
		var item string
		if err := rows.Scan(&item); err != nil {
			t.Fatal(err)
		}
		items = append(items, item)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if len(items) != 2 || items[0] != "a" || items[1] != "b" {
		t.Fatalf("JSON expansion items=%v", items)
	}
	rows, err = sqlDB.QueryContext(ctx, `SELECT key, value FROM jsonb_each('{"b":2,"a":1}'::jsonb) AS e`)
	if err != nil {
		t.Fatalf("JSON object expansion query: %v", err)
	}
	var keys []string
	for rows.Next() {
		var key, value string
		if err := rows.Scan(&key, &value); err != nil {
			t.Fatal(err)
		}
		keys = append(keys, key)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if len(keys) != 2 || keys[0] != "a" || keys[1] != "b" {
		t.Fatalf("JSON object expansion keys=%v", keys)
	}
	rows, err = sqlDB.QueryContext(ctx, `SELECT elem FROM json_docs d CROSS JOIN jsonb_array_elements(d.payload->'roles') AS elem`)
	if err != nil {
		t.Fatalf("lateral JSON expansion query: %v", err)
	}
	var roleCount int
	for rows.Next() {
		var role string
		if err := rows.Scan(&role); err != nil {
			t.Fatal(err)
		}
		roleCount++
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if roleCount != 3 {
		t.Fatalf("lateral JSON expansion count=%d", roleCount)
	}
	rows, err = sqlDB.QueryContext(ctx, `SELECT id FROM json_docs WHERE payload @? '$.**.active' ORDER BY id`)
	if err != nil {
		t.Fatalf("recursive JSONPath query: %v", err)
	}
	var recursiveIDs []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			t.Fatal(err)
		}
		recursiveIDs = append(recursiveIDs, id)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if len(recursiveIDs) != 2 || recursiveIDs[0] != "d1" || recursiveIDs[1] != "d2" {
		t.Fatalf("recursive JSONPath ids=%v", recursiveIDs)
	}
	if _, err := sqlDB.QueryContext(ctx, `SELECT id FROM json_docs WHERE payload @@ 'strict $.profile.missing'`); err == nil {
		t.Fatal("strict JSONPath missing step unexpectedly succeeded")
	}
}

func TestDescribeStatement_JSONTypes(t *testing.T) {
	db := seedJSONSQLDB(t)
	defer db.Close()

	params, cols, err := describeStatement(db, `SELECT payload->'profile' AS profile, payload->>'name' AS name
		FROM json_docs WHERE payload @> $1`, 1)
	if err != nil {
		t.Fatalf("describe JSON statement: %v", err)
	}
	if len(params) != 1 || params[0] != OIDJSONB {
		t.Fatalf("JSON parameter OIDs: got %v, want [%d]", params, OIDJSONB)
	}
	assertColumns(t, cols, []ColumnMeta{
		{Name: "profile", TypeOID: OIDJSONB},
		{Name: "name", TypeOID: OIDText},
	})
}

func TestJSONBResultBinaryEncoding(t *testing.T) {
	encoded, err := encodeResultValue(map[string]interface{}{"ok": true}, OIDJSONB, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(encoded) < 2 || encoded[0] != 1 || string(encoded[1:]) != `{"ok":true}` {
		t.Fatalf("JSONB binary value=%v", encoded)
	}
}

func TestPGWireJSON_DDLAndDML(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(t.TempDir()+"/json_pgwire_ddl"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
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
	if _, err := sqlDB.ExecContext(ctx, "CREATE TABLE json_wire (id TEXT PRIMARY KEY, payload JSONB)"); err != nil {
		t.Fatalf("pgwire CREATE TABLE: %v", err)
	}
	if _, err := sqlDB.ExecContext(ctx, `INSERT INTO json_wire (id, payload) VALUES ('d1', '{"kind":"vector"}')`); err != nil {
		t.Fatalf("pgwire INSERT: %v", err)
	}
	if _, err := sqlDB.ExecContext(ctx, `CREATE INDEX json_wire_kind_idx
		ON json_wire (payload#>>'{kind}')`); err != nil {
		t.Fatalf("pgwire CREATE JSON index: %v", err)
	}
	var kind string
	if err := sqlDB.QueryRowContext(ctx, `SELECT payload->>'kind' FROM json_wire WHERE id = 'd1'`).Scan(&kind); err != nil {
		t.Fatalf("pgwire JSON SELECT: %v", err)
	}
	if kind != "vector" {
		t.Fatalf("pgwire JSON value=%q", kind)
	}
	var indexedID string
	if err := sqlDB.QueryRowContext(ctx, `SELECT id FROM json_wire
		WHERE payload#>>'{kind}' = $1`, "vector").Scan(&indexedID); err != nil {
		t.Fatalf("pgwire indexed JSON predicate: %v", err)
	}
	if indexedID != "d1" {
		t.Fatalf("pgwire indexed JSON id=%q", indexedID)
	}
	var patched string
	if err := sqlDB.QueryRowContext(ctx,
		`SELECT jsonb_set(payload, '{kind}', '"document"') FROM json_wire WHERE id = 'd1'`).Scan(&patched); err != nil {
		t.Fatalf("pgwire jsonb_set: %v", err)
	}
	if patched != `{"kind":"document"}` {
		t.Fatalf("pgwire jsonb_set value=%q", patched)
	}
	if _, err := sqlDB.ExecContext(ctx, `CREATE TABLE jsonb_dml (id TEXT PRIMARY KEY, payload JSONB)`); err != nil {
		t.Fatalf("pgwire JSONB DML CREATE TABLE: %v", err)
	}
	if _, err := sqlDB.ExecContext(ctx, `INSERT INTO jsonb_dml (id, payload) VALUES
		('d1', '{"career":"engineer"}'), ('d2', '{"career":"scientist"}')`); err != nil {
		t.Fatalf("pgwire JSONB DML INSERT: %v", err)
	}
	if _, err := sqlDB.ExecContext(ctx, `UPDATE jsonb_dml
		SET payload = jsonb_set(payload, '{career}', '[]'::jsonb, true)
		WHERE jsonb_typeof(payload->'career') = 'string'`); err != nil {
		t.Fatalf("pgwire JSONB DML jsonb_set: %v", err)
	}
	if _, err := sqlDB.ExecContext(ctx, `UPDATE jsonb_dml
		SET payload = jsonb_set(payload, '{skills}', $1::jsonb, true)
		WHERE id = $2`, `["go","sql"]`, "d1"); err != nil {
		t.Fatalf("pgwire parameterized JSONB DML jsonb_set: %v", err)
	}
	var updatedPayload string
	if err := sqlDB.QueryRowContext(ctx, `SELECT payload FROM jsonb_dml WHERE id = 'd1'`).Scan(&updatedPayload); err != nil {
		t.Fatalf("pgwire JSONB DML SELECT: %v", err)
	}
	if updatedPayload != `{"career":[],"skills":["go","sql"]}` {
		t.Fatalf("pgwire JSONB DML payload=%q", updatedPayload)
	}
}

func TestPGWireJSONMutationAndRecordExpansion(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(t.TempDir()+"/json_pgwire_mutation"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
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
	nativeArray, err := db.Query(ctx, `SELECT jsonb_build_array('Ada',37,true)`)
	if err != nil || nativeArray.Total != 1 {
		t.Fatalf("native jsonb_build_array: result=%#v err=%v", nativeArray, err)
	}

	var deleted string
	if err := sqlDB.QueryRowContext(ctx, `SELECT '{"a":1,"nested":{"drop":true}}'::jsonb #- '{nested,drop}'`).Scan(&deleted); err != nil {
		t.Fatalf("pgwire JSON #-: %v", err)
	}
	if deleted != `{"a":1,"nested":{}}` {
		t.Fatalf("pgwire JSON #- value=%q", deleted)
	}

	var inserted string
	if err := sqlDB.QueryRowContext(ctx, `SELECT jsonb_insert('{"arr":[1,3]}'::jsonb, '{arr,1}', '2')`).Scan(&inserted); err != nil {
		t.Fatalf("pgwire jsonb_insert: %v", err)
	}
	if inserted != `{"arr":[1,2,3]}` {
		t.Fatalf("pgwire jsonb_insert value=%q", inserted)
	}

	var merged string
	if err := sqlDB.QueryRowContext(ctx, `SELECT '{"a":1}'::jsonb || '{"b":2}'::jsonb`).Scan(&merged); err != nil {
		t.Fatalf("pgwire JSON ||: %v", err)
	}
	if merged != `{"a":1,"b":2}` {
		t.Fatalf("pgwire JSON || value=%q", merged)
	}

	var built string
	if err := sqlDB.QueryRowContext(ctx, `SELECT jsonb_build_object('name','Ada','age',37)`).Scan(&built); err != nil {
		t.Fatalf("pgwire jsonb_build_object: %v", err)
	}
	if built != `{"age":37,"name":"Ada"}` && built != `{"name":"Ada","age":37}` {
		t.Fatalf("pgwire jsonb_build_object value=%q", built)
	}
	var builtArray string
	if err := sqlDB.QueryRowContext(ctx, `SELECT jsonb_build_array('Ada',37,true)`).Scan(&builtArray); err != nil {
		t.Fatalf("pgwire jsonb_build_array: %v", err)
	}
	if builtArray != `["Ada",37,true]` {
		t.Fatalf("pgwire jsonb_build_array value=%q", builtArray)
	}

	var name string
	var age int64
	if err := sqlDB.QueryRowContext(ctx, `SELECT r.name, r.age FROM jsonb_to_record('{"name":"Ada","age":37}'::jsonb) AS r`).Scan(&name, &age); err != nil {
		t.Fatalf("pgwire jsonb_to_record: %v", err)
	}
	if name != "Ada" || age != 37 {
		t.Fatalf("pgwire jsonb_to_record values name=%q age=%d", name, age)
	}

	rows, err := sqlDB.QueryContext(ctx, `SELECT r.name FROM jsonb_to_recordset('[{"name":"Ada"},{"name":"Grace"}]'::jsonb) AS r ORDER BY r.name`)
	if err != nil {
		t.Fatalf("pgwire jsonb_to_recordset: %v", err)
	}
	defer rows.Close()
	var names []string
	for rows.Next() {
		var value string
		if err := rows.Scan(&value); err != nil {
			t.Fatalf("pgwire jsonb_to_recordset scan: %v", err)
		}
		names = append(names, value)
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if len(names) != 2 || names[0] != "Ada" || names[1] != "Grace" {
		t.Fatalf("pgwire jsonb_to_recordset values=%v", names)
	}
	rows, err = sqlDB.QueryContext(ctx, `SELECT r.name FROM jsonb_populate_recordset('{"name":"Unknown"}'::jsonb, '[{"name":"Ada"},{"name":"Grace"}]'::jsonb) AS r ORDER BY r.name`)
	if err != nil {
		t.Fatalf("pgwire jsonb_populate_recordset: %v", err)
	}
	names = names[:0]
	for rows.Next() {
		var value string
		if err := rows.Scan(&value); err != nil {
			t.Fatalf("pgwire jsonb_populate_recordset scan: %v", err)
		}
		names = append(names, value)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if len(names) != 2 || names[0] != "Ada" || names[1] != "Grace" {
		t.Fatalf("pgwire jsonb_populate_recordset values=%v", names)
	}
}

func TestPGWireJSONNullSemantics(t *testing.T) {
	ctx := context.Background()
	db, err := libravdb.Open(libravdb.WithStoragePath(t.TempDir()+"/json_pgwire_null"), libravdb.WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
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
	if _, err := sqlDB.ExecContext(ctx, "CREATE TABLE json_null (id TEXT PRIMARY KEY, payload JSONB, nullable JSONB)"); err != nil {
		t.Fatalf("CREATE TABLE: %v", err)
	}
	if _, err := sqlDB.ExecContext(ctx, `INSERT INTO json_null (id, payload, nullable)
		VALUES ('d1', '{"present":null}', NULL)`); err != nil {
		t.Fatalf("INSERT: %v", err)
	}
	var jsonType, extracted string
	var sqlValue *string
	if err := sqlDB.QueryRowContext(ctx, `SELECT jsonb_typeof(payload->'present'),
		payload->'present', nullable FROM json_null WHERE id = 'd1'`).Scan(&jsonType, &extracted, &sqlValue); err != nil {
		t.Fatalf("JSON/SQL NULL query: %v", err)
	}
	if jsonType != "null" || extracted != "null" || sqlValue != nil {
		t.Fatalf("JSON/SQL NULL values: type=%q extracted=%q sql=%v", jsonType, extracted, sqlValue)
	}
}
