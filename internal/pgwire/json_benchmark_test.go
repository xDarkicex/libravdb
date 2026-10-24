package pgwire

import (
	"fmt"
	"testing"

	"github.com/xDarkicex/libravdb/libravdb"
)

func BenchmarkPGWireMetadataValueToString_JSON(b *testing.B) {
	doc := map[string]interface{}{
		"id":       "user-12345",
		"username": "ada_lovelace",
		"active":   true,
		"tags":     []interface{}{"admin", "developer", "researcher"},
		"profile": map[string]interface{}{
			"bio":       "First programmer",
			"login_cnt": 42,
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s := metadataValueToString(doc)
		if len(s) == 0 {
			b.Fatal("empty result")
		}
	}
}

func BenchmarkPGWireBuildResultRow_1000JSONRows(b *testing.B) {
	const numRows = 1000
	rows := make([]*libravdb.SearchResult, numRows)
	for i := 0; i < numRows; i++ {
		rows[i] = &libravdb.SearchResult{
			ID:    fmt.Sprintf("doc-%04d", i),
			Score: float32(i) * 0.1,
			Metadata: map[string]interface{}{
				"payload": map[string]interface{}{
					"index":  i,
					"active": i%2 == 0,
					"roles":  []interface{}{"viewer", "editor"},
				},
			},
		}
	}

	cols := []ColumnMeta{
		{Name: "id", TypeOID: OIDText},
		{Name: "payload", TypeOID: OIDJSONB},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, r := range rows {
			vals := buildResultRow(r, cols)
			if vals[1] == nil || len(*vals[1]) == 0 {
				b.Fatal("missing payload column value")
			}
		}
	}
}
