package pgwire

import (
	"bytes"
	"context"
	"testing"

	"github.com/xDarkicex/libravdb/libravdb"
)

func TestSQLStatsSystemFunctionUsesJSONBEnvelope(t *testing.T) {
	db, err := libravdb.Open(
		libravdb.WithStoragePath(":memory:pgwire_sql_stats"),
		libravdb.WithMetrics(false),
	)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()
	if _, err := db.Query(context.Background(), `CREATE TABLE stats_rows (id TEXT PRIMARY KEY)`); err != nil {
		t.Fatalf("CREATE TABLE: %v", err)
	}
	db.ResetSQLStats()
	if _, err := db.Query(context.Background(), `SELECT id FROM stats_rows`); err != nil {
		t.Fatalf("SELECT: %v", err)
	}

	results, columns, handled := interceptSystemQuery(`SELECT LIBRAVDB_SQL_STATS()`, db)
	if !handled || len(columns) != 1 || columns[0].Name != "libravdb_sql_stats" || columns[0].TypeOID != OIDJSONB {
		t.Fatalf("SQL stats interception: handled=%v columns=%#v", handled, columns)
	}
	value, ok := results.Results[0].Metadata["libravdb_sql_stats"]
	if !ok {
		t.Fatalf("SQL stats metadata=%#v", results.Results[0].Metadata)
	}
	if _, ok := value.(libravdb.SQLQueryStats); !ok {
		t.Fatalf("SQL stats value type=%T, want libravdb.SQLQueryStats", value)
	}
	encoded, err := encodeResultValue(value, OIDJSONB, 0)
	if err != nil {
		t.Fatalf("encode SQL stats JSONB: %v", err)
	}
	if !bytes.Contains(encoded, []byte(`"queries":1`)) {
		t.Fatalf("encoded SQL stats=%q, want JSONB payload with queries=1", encoded)
	}
}
