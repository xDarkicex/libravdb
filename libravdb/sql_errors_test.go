package libravdb

import (
	"context"
	"errors"
	"testing"
)

func TestSQLStructuredErrorsPreserveMessageAndClassification(t *testing.T) {
	db, err := Open(WithStoragePath(":memory:sql-errors"), WithMetrics(false))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	_, err = db.Query(context.Background(), "SELECT id FROM missing_sql_table")
	if err == nil {
		t.Fatal("missing table should fail")
	}
	var sqlErr *SQLError
	if !errors.As(err, &sqlErr) {
		t.Fatalf("error type=%T, want *SQLError", err)
	}
	if sqlErr.Code != SQLErrorUndefinedTable || sqlErr.SQLState != "42P01" || sqlErr.Message == "" {
		t.Fatalf("structured error=%+v", sqlErr)
	}
	if !errors.Is(err, &SQLError{Code: SQLErrorUndefinedTable}) {
		t.Fatalf("errors.Is did not preserve SQL error code: %v", err)
	}
}
