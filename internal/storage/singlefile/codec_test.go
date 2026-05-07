package singlefile

import "testing"

func TestMetadataCodecAcceptsAllIntegerWidths(t *testing.T) {
	metadata := map[string]interface{}{
		"int8":   int8(-8),
		"int16":  int16(-16),
		"int32":  int32(-32),
		"uint":   uint(64),
		"uint8":  uint8(8),
		"uint16": uint16(16),
		"uint32": uint32(32),
		"nested": map[string]interface{}{
			"signal": int32(4),
		},
		"slice": []interface{}{int32(7), uint16(9)},
	}

	enc := acquireBinaryEncoder(0)
	if err := enc.writeMetadata(metadata); err != nil {
		t.Fatalf("writeMetadata() error = %v", err)
	}
	encoded := enc.detachBytes()
	releaseBinaryEncoder(enc)

	got, err := (&binaryDecoder{data: encoded}).readMetadata()
	if err != nil {
		t.Fatalf("readMetadata() error = %v", err)
	}

	if got["int8"] != int64(-8) {
		t.Fatalf("int8 decoded as %#v, want int64(-8)", got["int8"])
	}
	if got["int16"] != int64(-16) {
		t.Fatalf("int16 decoded as %#v, want int64(-16)", got["int16"])
	}
	if got["int32"] != int64(-32) {
		t.Fatalf("int32 decoded as %#v, want int64(-32)", got["int32"])
	}
	if got["uint"] != uint64(64) {
		t.Fatalf("uint decoded as %#v, want uint64(64)", got["uint"])
	}
	if got["uint8"] != uint64(8) {
		t.Fatalf("uint8 decoded as %#v, want uint64(8)", got["uint8"])
	}
	if got["uint16"] != uint64(16) {
		t.Fatalf("uint16 decoded as %#v, want uint64(16)", got["uint16"])
	}
	if got["uint32"] != uint64(32) {
		t.Fatalf("uint32 decoded as %#v, want uint64(32)", got["uint32"])
	}

	nested := got["nested"].(map[string]interface{})
	if nested["signal"] != int64(4) {
		t.Fatalf("nested signal decoded as %#v, want int64(4)", nested["signal"])
	}

	slice := got["slice"].([]interface{})
	if slice[0] != int64(7) || slice[1] != uint64(9) {
		t.Fatalf("slice decoded as %#v, want [int64(7), uint64(9)]", slice)
	}
}
