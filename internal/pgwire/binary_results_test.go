package pgwire

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"github.com/xDarkicex/libravdb/libravdb"
)

func TestBinaryResultFormatsEncodeTypedRows(t *testing.T) {
	results := &libravdb.SearchResults{
		Results: []*libravdb.SearchResult{{
			ID:    "42",
			Score: 0.5,
			Metadata: map[string]interface{}{
				"active":    true,
				"embedding": []float32{1, -2.5},
				"note":      nil,
			},
		}},
		Total: 1,
	}
	columns := []ColumnMeta{
		{Name: "id", TypeOID: OIDInt8},
		{Name: "score", TypeOID: OIDFloat8},
		{Name: "active", TypeOID: OIDBool},
		{Name: "embedding", TypeOID: OIDFloat4Array},
		{Name: "note", TypeOID: OIDText},
	}
	var wire bytes.Buffer
	if err := sendResultsWithFormats(&wire, results, columns, []int16{1}); err != nil {
		t.Fatalf("send binary results: %v", err)
	}

	kind, description, err := ReadMessage(&wire)
	if err != nil || kind != msgRowDescription {
		t.Fatalf("RowDescription = %q, %v", kind, err)
	}
	if got := binary.BigEndian.Uint16(description[:2]); got != uint16(len(columns)) {
		t.Fatalf("column count = %d", got)
	}
	// Every column must advertise binary format when Bind supplied one code.
	off := 2
	for i := range columns {
		for off < len(description) && description[off] != 0 {
			off++
		}
		off++ // name terminator
		off += 4 + 2 + 4 + 2 + 4
		if got := binary.BigEndian.Uint16(description[off : off+2]); got != 1 {
			t.Fatalf("column %d format = %d, want binary", i, got)
		}
		off += 2
	}

	kind, row, err := ReadMessage(&wire)
	if err != nil || kind != msgDataRow {
		t.Fatalf("DataRow = %q, %v", kind, err)
	}
	if got := binary.BigEndian.Uint16(row[:2]); got != uint16(len(columns)) {
		t.Fatalf("row column count = %d", got)
	}
	off = 2
	readField := func() []byte {
		n := int32(binary.BigEndian.Uint32(row[off : off+4]))
		off += 4
		if n < 0 {
			return nil
		}
		value := row[off : off+int(n)]
		off += int(n)
		return value
	}
	if got := readField(); len(got) != 8 || int64(binary.BigEndian.Uint64(got)) != 42 {
		t.Fatalf("binary id = %v", got)
	}
	if got := readField(); len(got) != 8 || math.Float64frombits(binary.BigEndian.Uint64(got)) != 0.5 {
		t.Fatalf("binary score = %v", got)
	}
	if got := readField(); len(got) != 1 || got[0] != 1 {
		t.Fatalf("binary bool = %v", got)
	}
	array := readField()
	if len(array) < 20 || binary.BigEndian.Uint32(array[0:4]) != 1 || binary.BigEndian.Uint32(array[8:12]) != OIDFloat4 {
		t.Fatalf("binary vector array header = %v", array)
	}
	if got := readField(); got != nil {
		t.Fatalf("NULL field encoded as %v", got)
	}
}

func TestResultFormatCountValidation(t *testing.T) {
	columns := []ColumnMeta{{Name: "id", TypeOID: OIDText}, {Name: "score", TypeOID: OIDFloat8}}
	var wire bytes.Buffer
	err := sendRowDescriptionWithFormats(&wire, columns, []int16{1, 0, 1})
	if err == nil {
		t.Fatal("expected result-format count mismatch")
	}
}
