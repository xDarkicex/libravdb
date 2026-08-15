package libravdb

import (
	"context"
	"fmt"
	"testing"
)

func TestSQLUpsertOnConflictDoUpdateAndNothing(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql_upsert"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Query(ctx, "CREATE TABLE docs (id TEXT PRIMARY KEY, title TEXT, category TEXT)"); err != nil {
		t.Fatalf("create: %v", err)
	}
	if _, err := db.Query(ctx, "INSERT INTO docs (id, title, category) VALUES ('d1', 'old', 'a')"); err != nil {
		t.Fatalf("insert: %v", err)
	}
	if _, err := db.Query(ctx, "INSERT INTO docs (id, title, category) VALUES ('d1', 'new', 'b') ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title"); err != nil {
		t.Fatalf("do update: %v", err)
	}
	col, err := db.GetCollection("docs")
	if err != nil {
		t.Fatal(err)
	}
	record, err := col.Get(ctx, "d1")
	if err != nil {
		t.Fatal(err)
	}
	if record.Metadata["title"] != "new" || record.Metadata["category"] != "a" {
		t.Fatalf("DO UPDATE replaced wrong fields: %#v", record.Metadata)
	}
	if _, err := db.Query(ctx, "INSERT INTO docs (id, title, category) VALUES ('d1', 'ignored', 'ignored') ON CONFLICT (id) DO NOTHING"); err != nil {
		t.Fatalf("do nothing: %v", err)
	}
	record, err = col.Get(ctx, "d1")
	if err != nil {
		t.Fatal(err)
	}
	if record.Metadata["title"] != "new" || record.Metadata["category"] != "a" {
		t.Fatalf("DO NOTHING mutated row: %#v", record.Metadata)
	}
	if _, err := db.Query(ctx, "INSERT INTO docs (id, title, category) VALUES ('d2', 'inserted', 'c') ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title"); err != nil {
		t.Fatalf("non-conflicting upsert insert: %v", err)
	}
	if _, err := col.Get(ctx, "d2"); err != nil {
		t.Fatalf("new row missing after upsert: %v", err)
	}
}

func TestSQLUpsertEpochRollback(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql_upsert_epoch"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Query(ctx, "CREATE TABLE docs (id TEXT PRIMARY KEY, title TEXT)"); err != nil {
		t.Fatalf("create: %v", err)
	}
	if _, err := db.Query(ctx, "INSERT INTO docs (id, title) VALUES ('d1', 'old')"); err != nil {
		t.Fatalf("insert: %v", err)
	}
	session, err := db.NewSQLSession(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	if err := session.Exec("BEGIN EPOCH TRANSACTION"); err != nil {
		t.Fatal(err)
	}
	if err := session.Exec("INSERT INTO docs (id, title) VALUES ('d1', 'branch') ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title"); err != nil {
		t.Fatal(err)
	}
	if err := session.Exec("ROLLBACK"); err != nil {
		t.Fatal(err)
	}
	col, err := db.GetCollection("docs")
	if err != nil {
		t.Fatal(err)
	}
	record, err := col.Get(ctx, "d1")
	if err != nil {
		t.Fatal(err)
	}
	if record.Metadata["title"] != "old" {
		t.Fatalf("epoch rollback leaked upsert: %#v", record.Metadata)
	}
}

func TestSQLUpsertOnUniqueConflictTarget(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql_upsert_unique"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Query(ctx, "CREATE TABLE accounts (id TEXT PRIMARY KEY, email TEXT UNIQUE, name TEXT)"); err != nil {
		t.Fatalf("create: %v", err)
	}
	col, err := db.GetCollection("accounts")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, "INSERT INTO accounts (id, email, name) VALUES ('a1', 'ada@example.com', 'Ada')"); err != nil {
		t.Fatalf("insert: %v", err)
	}
	if _, err := db.Query(ctx, "INSERT INTO accounts (id, email, name) VALUES ('a0', '', 'Empty')"); err != nil {
		t.Fatalf("empty-string insert: %v", err)
	}
	if _, err := db.Query(ctx, "INSERT INTO accounts (id, email, name) VALUES ('a3', '', 'Ignored') ON CONFLICT (email) DO NOTHING"); err != nil {
		t.Fatalf("empty-string do nothing: %v", err)
	}
	if _, err := col.Get(ctx, "a3"); err == nil {
		t.Fatal("empty string was treated as NULL and failed to detect the UNIQUE conflict")
	}
	if _, err := db.Query(ctx, "INSERT INTO accounts (id, email, name) VALUES ('a2', 'ada@example.com', 'Ada Lovelace') ON CONFLICT (email) DO UPDATE SET name = EXCLUDED.name"); err != nil {
		t.Fatalf("unique-key upsert: %v", err)
	}
	record, err := col.Get(ctx, "a1")
	if err != nil {
		t.Fatal(err)
	}
	if record.Metadata["name"] != "Ada Lovelace" {
		t.Fatalf("unique-key upsert did not update existing row: %#v", record.Metadata)
	}
	if _, err := col.Get(ctx, "a2"); err == nil {
		t.Fatal("unique-key upsert created a second conflicting row")
	}
}

func TestSQLUpsertBasicExpressionAssignment(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql_upsert_expr"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Query(ctx, "CREATE TABLE counters (id TEXT PRIMARY KEY, title TEXT, counter TEXT)"); err != nil {
		t.Fatalf("create: %v", err)
	}
	if _, err := db.Query(ctx, "INSERT INTO counters (id, title, counter) VALUES ('c1', 'old', '1')"); err != nil {
		t.Fatalf("insert: %v", err)
	}
	if _, err := db.Query(ctx, "INSERT INTO counters (id, title, counter) VALUES ('c1', 'new', '2') ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title || '-updated', counter = counter + EXCLUDED.counter"); err != nil {
		t.Fatalf("expression upsert: %v", err)
	}
	col, err := db.GetCollection("counters")
	if err != nil {
		t.Fatal(err)
	}
	record, err := col.Get(ctx, "c1")
	if err != nil {
		t.Fatal(err)
	}
	if record.Metadata["title"] != "new-updated" || record.Metadata["counter"] != "3" {
		t.Fatalf("expression assignment produced %#v", record.Metadata)
	}
}

func TestSQLCounterUpsertExample(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(":memory:sql_counter_example"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Query(ctx, "CREATE TABLE counters (key TEXT PRIMARY KEY, value BIGINT NOT NULL DEFAULT 0)"); err != nil {
		t.Fatalf("create counters: %v", err)
	}
	if _, err := db.Query(ctx, "INSERT INTO counters (key) VALUES ('default_only')"); err != nil {
		t.Fatalf("default insert: %v", err)
	}
	defaultRecord, err := db.GetCollection("counters")
	if err != nil {
		t.Fatal(err)
	}
	defaultRow, err := defaultRecord.Get(ctx, "__pk:3:key12:default_only|")
	if err != nil || fmt.Sprint(defaultRow.Metadata["value"]) != "0" {
		t.Fatalf("default value=%#v err=%v, want 0", defaultRow.Metadata, err)
	}
	if _, err := db.Query(ctx, "INSERT INTO counters (key, value) VALUES ('page_views_home', 123) ON CONFLICT (key) DO UPDATE SET value = counters.value + EXCLUDED.value"); err != nil {
		t.Fatalf("first counter upsert: %v", err)
	}
	if _, err := db.Query(ctx, "INSERT INTO counters (key, value) VALUES ('page_views_home', 123) ON CONFLICT (key) DO UPDATE SET value = counters.value + EXCLUDED.value"); err != nil {
		t.Fatalf("second counter upsert: %v", err)
	}
	col, err := db.GetCollection("counters")
	if err != nil {
		t.Fatal(err)
	}
	record, err := col.Get(ctx, "__pk:3:key15:page_views_home|")
	if err != nil {
		t.Fatalf("counter record: %v", err)
	}
	if record.Metadata["value"] != "246" {
		t.Fatalf("counter value=%#v, want 246", record.Metadata["value"])
	}
}
