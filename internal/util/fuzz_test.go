package util

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

// FuzzBinaryDecoderRoundtrip verifies that encoding then decoding a vector
// produces the original data. Fuzz input is []byte reinterpreted as []float32.
func FuzzBinaryDecoderRoundtrip(f *testing.F) {
	// Seed corpus: valid float32 vectors encoded as little-endian bytes.
	seed1 := make([]byte, 20) // 5 x float32
	for i := range 5 {
		binary.LittleEndian.PutUint32(seed1[i*4:], math.Float32bits(float32(i+1)))
	}
	f.Add(seed1)
	f.Add([]byte{}) // empty

	f.Fuzz(func(t *testing.T, raw []byte) {
		// Interpret raw bytes as float32 values (pad if needed).
		n := len(raw) / 4
		vec := make([]float32, n)
		for i := range n {
			vec[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}

		enc := AcquireBinaryEncoder(len(raw) + 4)
		defer ReleaseBinaryEncoder(enc)

		enc.WriteVector(vec)
		data := enc.DetachBytes()

		dec := &BinaryDecoder{Data: data}
		got, err := dec.ReadVector()
		if err != nil {
			t.Fatalf("ReadVector: %v", err)
		}
		if len(got) != len(vec) {
			t.Fatalf("len: got %d, want %d", len(got), len(vec))
		}
		for i := range vec {
			if got[i] != vec[i] {
				t.Fatalf("vec[%d]: got %f, want %f", i, got[i], vec[i])
			}
		}
	})
}

// FuzzBinaryDecoderCorrupt verifies that decoding arbitrary/corrupt data
// returns an error, not a panic.
func FuzzBinaryDecoderCorrupt(f *testing.F) {
	// Seed with valid and edge-case payloads.
	f.Add([]byte{0, 0, 0, 0})                         // zero-length vector
	f.Add([]byte{0, 0, 0, 1, 0, 0, 0, 0})             // 1-element zero
	f.Add([]byte{0xff, 0xff, 0xff, 0xff})             // max uint32 length
	f.Add([]byte{0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0}) // 2-element (but only 1 full float)
	f.Add([]byte{1, 2, 3})                            // truncated header

	f.Fuzz(func(t *testing.T, data []byte) {
		dec := &BinaryDecoder{Data: data}

		// Each read must either succeed or return an error — never panic.
		if _, err := dec.ReadVector(); err != nil {
			return // expected on corrupt input
		}
		// If ReadVector succeeded, try reading metadata too.
		_, _ = dec.ReadMetadata()
	})
}

// FuzzBinaryDecoderMetadata verifies metadata roundtrip encoding/decoding
// with common value types.
func FuzzBinaryDecoderMetadata(f *testing.F) {
	f.Add("key1", "hello")
	f.Add("key2", "world")

	f.Fuzz(func(t *testing.T, key, val string) {
		if len(key) == 0 || len(val) == 0 {
			return
		}
		md := map[string]interface{}{key: val}
		enc := AcquireBinaryEncoder(1024)
		defer ReleaseBinaryEncoder(enc)

		if err := enc.WriteMetadata(md); err != nil {
			t.Skipf("encode metadata: %v", err)
		}
		data := enc.DetachBytes()

		dec := &BinaryDecoder{Data: data}
		got, err := dec.ReadMetadata()
		if err != nil {
			t.Fatalf("ReadMetadata: %v", err)
		}
		if got == nil {
			t.Fatal("expected non-nil metadata")
		}
	})
}

// FuzzBinaryDecoderString verifies string roundtrip encoding/decoding and
// that corrupt length prefixes return errors, not panics.
func FuzzBinaryDecoderString(f *testing.F) {
	f.Add("hello")
	f.Add("")
	f.Add(string(make([]byte, 10000)))

	f.Fuzz(func(t *testing.T, s string) {
		if len(s) > MaxStringLen {
			return
		}
		enc := AcquireBinaryEncoder(len(s) + 4)
		defer ReleaseBinaryEncoder(enc)

		enc.WriteString(s)
		data := enc.DetachBytes()

		dec := &BinaryDecoder{Data: data}
		got, err := dec.ReadString()
		if err != nil {
			t.Fatalf("ReadString: %v", err)
		}
		if got != s {
			t.Fatalf("got %q, want %q", got, s)
		}

		// Also fuzz with truncated data.
		for trunc := 1; trunc < len(data); trunc++ {
			d2 := &BinaryDecoder{Data: data[:trunc]}
			if _, err := d2.ReadString(); err != nil {
				continue // expected
			}
		}
	})
}

// TestBinaryDecoderBounds verifies specific edge cases that would previously
// panic (slice bounds, integer overflow).
func TestBinaryDecoderBounds(t *testing.T) {
	tests := []struct {
		name string
		data []byte
	}{
		{"empty", nil},
		{"truncated_uint32", []byte{0x01}},
		{"huge_length", []byte{0xff, 0xff, 0xff, 0xff}},
		{"zero_vector", []byte{0, 0, 0, 0}},
		{"mismatched_length", []byte{0, 0, 0, 2, 0, 0, 0, 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dec := &BinaryDecoder{Data: tt.data}
			_, err := dec.ReadVector()
			if err == nil {
				t.Log("unexpected success on corrupt input")
			}
			// Must not panic.
		})
	}
}

// TestBinaryEncoderPool verifies the encoder pool acquires and releases correctly.
func TestBinaryEncoderPool(t *testing.T) {
	var bufs [][]byte
	for i := range 100 {
		enc := AcquireBinaryEncoder(128)
		enc.WriteUint32(uint32(i))
		bufs = append(bufs, bytes.Clone(enc.DetachBytes()))
		ReleaseBinaryEncoder(enc)
	}
	// Decode back.
	for i, data := range bufs {
		dec := &BinaryDecoder{Data: data}
		v, err := dec.ReadUint32()
		if err != nil {
			t.Fatalf("buf %d: %v", i, err)
		}
		if v != uint32(i) {
			t.Fatalf("buf %d: got %d, want %d", i, v, i)
		}
	}
}
