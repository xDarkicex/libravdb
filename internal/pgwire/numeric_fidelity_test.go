package pgwire

import (
	"encoding/binary"
	"testing"

	"github.com/xDarkicex/libravdb/internal/catalog"
	"github.com/xDarkicex/libravdb/libravdb"
)

func TestDecodeTextParametersWithoutSQLStringSubstitution(t *testing.T) {
	value, err := decodeParamValue([]byte("922337203685477580"), 0, OIDInt8)
	if err != nil || value != int64(922337203685477580) {
		t.Fatalf("int8 decode = %#v, %v", value, err)
	}
	value, err = decodeParamValue([]byte("TRUE"), 0, OIDBool)
	if err != nil || value != true {
		t.Fatalf("bool decode = %#v, %v", value, err)
	}
	value, err = decodeParamValue([]byte("[1.5, -2, 3e0]"), 0, OIDFloat4Array)
	vector, ok := value.([]float32)
	if err != nil || !ok || len(vector) != 3 || vector[1] != -2 {
		t.Fatalf("vector decode = %#v, %v", value, err)
	}
	value, err = decodeParamValue([]byte("[1.5, -2, 3e0]"), 0, OIDVector)
	vector, ok = value.([]float32)
	if err != nil || !ok || len(vector) != 3 || vector[1] != -2 {
		t.Fatalf("OIDVector decode = %#v, %v", value, err)
	}
	value, err = decodeParamValue(nil, 0, OIDText)
	if err != nil || value != nil {
		t.Fatalf("NULL decode = %#v, %v", value, err)
	}
}

func TestDecodeBinaryOIDArray(t *testing.T) {
	oids := []uint32{25, 16384, 20, OIDOIDArray}
	raw := make([]byte, 20+len(oids)*8)
	binary.BigEndian.PutUint32(raw[0:4], 1) // dimensions
	binary.BigEndian.PutUint32(raw[4:8], 0) // has null
	binary.BigEndian.PutUint32(raw[8:12], OIDOID)
	binary.BigEndian.PutUint32(raw[12:16], uint32(len(oids)))
	binary.BigEndian.PutUint32(raw[16:20], 1) // lower bound
	off := 20
	for _, oid := range oids {
		binary.BigEndian.PutUint32(raw[off:off+4], 4)
		binary.BigEndian.PutUint32(raw[off+4:off+8], oid)
		off += 8
	}
	value, err := decodeBinaryParam(raw, OIDOIDArray)
	if err != nil || value != "{25,16384,20,1028}" {
		t.Fatalf("oid[] decode = %#v, %v", value, err)
	}
}

func TestGraphNodesIDUsesInt8OID(t *testing.T) {
	results := &libravdb.SearchResults{
		Columns: []string{"id", "collection"},
		Results: []*libravdb.SearchResult{{
			ID: "9007199254740993",
			Metadata: map[string]interface{}{
				"id":         uint64(9007199254740993),
				"collection": "docs",
			},
		}},
	}
	cols := inferColumns(results)
	if cols[0].TypeOID != OIDInt8 {
		t.Fatalf("GRAPH_NODES.id OID = %d, want OIDInt8 (%d)", cols[0].TypeOID, OIDInt8)
	}
}

func TestGraphNodesEmptyResultUsesCatalogType(t *testing.T) {
	results := &libravdb.SearchResults{
		Columns:     []string{"id", "collection"},
		ColumnTypes: []uint16{catalog.TypeBigInt, catalog.TypeString},
	}
	cols := inferColumns(results)
	if cols[0].TypeOID != OIDInt8 {
		t.Fatalf("empty GRAPH_NODES.id OID = %d, want OIDInt8 (%d)", cols[0].TypeOID, OIDInt8)
	}
	if cols[1].TypeOID != OIDText {
		t.Fatalf("empty GRAPH_NODES.collection OID = %d, want OIDText (%d)", cols[1].TypeOID, OIDText)
	}
}

func TestOrdinaryIDRemainsText(t *testing.T) {
	results := &libravdb.SearchResults{
		Columns: []string{"id"},
		Results: []*libravdb.SearchResult{{ID: "user-1"}},
	}
	cols := inferColumns(results)
	if cols[0].TypeOID != OIDText {
		t.Fatalf("ordinary id OID = %d, want OIDText (%d)", cols[0].TypeOID, OIDText)
	}
}
