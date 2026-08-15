package pgwire

import (
	"bytes"
	"testing"

	"github.com/xDarkicex/libravdb/libravdb"
)

func TestExecutePortalRowsSuspendsAndResumes(t *testing.T) {
	portal := &Portal{
		Results: &libravdb.SearchResults{
			Results: []*libravdb.SearchResult{
				{ID: "a"}, {ID: "b"}, {ID: "c"},
			},
			Total: 3,
		},
		Columns: []ColumnMeta{{Name: "id", TypeOID: OIDText}},
	}

	var first bytes.Buffer
	if err := executePortalRows(&first, portal, 2); err != nil {
		t.Fatalf("first execute: %v", err)
	}
	if portal.RowIndex != 2 || portal.Complete {
		t.Fatalf("suspended state = index %d complete %v", portal.RowIndex, portal.Complete)
	}
	var kinds []byte
	for first.Len() > 0 {
		kind, _, err := ReadMessage(&first)
		if err != nil {
			t.Fatalf("read first response: %v", err)
		}
		kinds = append(kinds, kind)
	}
	want := []byte{msgRowDescription, msgDataRow, msgDataRow, msgPortalSuspended}
	if string(kinds) != string(want) {
		t.Fatalf("first response kinds = %q, want %q", kinds, want)
	}

	var second bytes.Buffer
	if err := executePortalRows(&second, portal, 0); err != nil {
		t.Fatalf("resume execute: %v", err)
	}
	if portal.RowIndex != 3 || !portal.Complete {
		t.Fatalf("completed state = index %d complete %v", portal.RowIndex, portal.Complete)
	}
	kinds = kinds[:0]
	for second.Len() > 0 {
		kind, _, err := ReadMessage(&second)
		if err != nil {
			t.Fatalf("read second response: %v", err)
		}
		kinds = append(kinds, kind)
	}
	want = []byte{msgDataRow, msgCommandComplete}
	if string(kinds) != string(want) {
		t.Fatalf("second response kinds = %q, want %q", kinds, want)
	}
}

func TestExecutePortalRowsEmptySelectStillDescribesRows(t *testing.T) {
	portal := &Portal{
		Stmt:    &PreparedStmt{Query: "SELECT version_num FROM alembic_version"},
		Results: &libravdb.SearchResults{Results: nil, Total: 0},
		Columns: []ColumnMeta{{Name: "version_num", TypeOID: OIDText}},
	}

	var output bytes.Buffer
	if err := executePortalRows(&output, portal, 0); err != nil {
		t.Fatalf("execute empty select: %v", err)
	}
	var kinds []byte
	for output.Len() > 0 {
		kind, _, err := ReadMessage(&output)
		if err != nil {
			t.Fatalf("read empty select response: %v", err)
		}
		kinds = append(kinds, kind)
	}
	want := []byte{msgRowDescription, msgCommandComplete}
	if string(kinds) != string(want) {
		t.Fatalf("empty select response kinds = %q, want %q", kinds, want)
	}
}
