package libravdb

import (
	"context"
	"testing"
)

func TestCheckNullComparisonIsUnknownAndAccepted(t *testing.T) {
	ctx := context.Background()
	db := openTempDB(t, "check_null_semantics")
	defer db.Close()

	if _, err := db.Query(ctx, "CREATE TABLE nullable_checks (id TEXT PRIMARY KEY, score INTEGER, CHECK (score > 0), CHECK (score BETWEEN 1 AND 10))"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, "INSERT INTO nullable_checks (id, score) VALUES ('explicit-null', NULL)"); err != nil {
		t.Fatalf("NULL CHECK operand should be UNKNOWN/accepted, got: %v", err)
	}
	if _, err := db.Query(ctx, "INSERT INTO nullable_checks (id) VALUES ('missing-null')"); err != nil {
		t.Fatalf("missing nullable CHECK operand should be UNKNOWN/accepted, got: %v", err)
	}
}

func TestCheckBooleanExpressionsAndParentheses(t *testing.T) {
	ctx := context.Background()
	db := openTempDB(t, "check_boolean_expr")
	defer db.Close()

	if _, err := db.Query(ctx, "CREATE TABLE boolean_checks (id TEXT PRIMARY KEY, score INTEGER, CHECK ((score > 0 AND score < 10) OR score = 100), CHECK (NOT (score = 7)))"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Query(ctx, "INSERT INTO boolean_checks (id, score) VALUES ('ok', 5)"); err != nil {
		t.Fatalf("valid boolean CHECK row rejected: %v", err)
	}
	if _, err := db.Query(ctx, "INSERT INTO boolean_checks (id, score) VALUES ('special', 100)"); err != nil {
		t.Fatalf("valid OR branch rejected: %v", err)
	}
	if _, err := db.Query(ctx, "INSERT INTO boolean_checks (id, score) VALUES ('bad-range', 12)"); err == nil {
		t.Fatal("out-of-range boolean CHECK row was accepted")
	}
	if _, err := db.Query(ctx, "INSERT INTO boolean_checks (id, score) VALUES ('bad-not', 7)"); err == nil {
		t.Fatal("NOT CHECK row was accepted")
	}
}
