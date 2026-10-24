package util

import (
	"sync"
	"testing"
)

func TestJSONEncoder_PoolAndCapacity(t *testing.T) {
	// Acquire with default estimate (<= 0)
	enc1 := AcquireJSONEncoder(0)
	if enc1 == nil {
		t.Fatal("expected non-nil encoder")
	}
	if cap(enc1.Buf) < minPooledEncoderCapacity {
		t.Fatalf("expected cap >= %d, got %d", minPooledEncoderCapacity, cap(enc1.Buf))
	}
	enc1.Buf = append(enc1.Buf, []byte(`{"key":"value"}`)...)
	ReleaseJSONEncoder(enc1)

	// Acquire with custom estimate
	enc2 := AcquireJSONEncoder(1024)
	if cap(enc2.Buf) < 1024 {
		t.Fatalf("expected cap >= 1024, got %d", cap(enc2.Buf))
	}
	ReleaseJSONEncoder(enc2)

	// Release nil should be safe
	ReleaseJSONEncoder(nil)

	// Oversized buffer should be dropped (not pooled with excessive capacity)
	enc3 := AcquireJSONEncoder(maxPooledEncoderCapacity * 2)
	enc3.Buf = make([]byte, 0, maxPooledEncoderCapacity*2)
	ReleaseJSONEncoder(enc3)
	if enc3.Buf != nil {
		t.Fatalf("expected enc3.Buf to be nil after dropping oversized buffer")
	}
}

func TestJSONEncoder_ConcurrentPool(t *testing.T) {
	const goroutines = 16
	const iterations = 100

	var wg sync.WaitGroup
	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go func(id int) {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				enc := AcquireJSONEncoder(128)
				enc.Buf = append(enc.Buf, []byte(`{"id":123}`)...)
				if string(enc.Buf) != `{"id":123}` {
					t.Errorf("unexpected content: %s", string(enc.Buf))
				}
				ReleaseJSONEncoder(enc)
			}
		}(i)
	}
	wg.Wait()
}

func TestEstimateJSONCapacity(t *testing.T) {
	tests := []struct {
		name string
		val  interface{}
		want int
	}{
		{"nil", nil, 4},
		{"string", "hello", 12},
		{"bytes", []byte("world"), 12},
		{"map", map[string]interface{}{"a": 1, "b": 2}, 98},
		{"slice", []interface{}{1, 2, 3}, 92},
		{"string_map", map[string]string{"a": "b"}, 50},
		{"string_slice", []string{"a", "b"}, 62},
		{"default", 12345, 64},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := EstimateJSONCapacity(tt.val)
			if got != tt.want {
				t.Errorf("EstimateJSONCapacity(%v) = %d, want %d", tt.val, got, tt.want)
			}
		})
	}
}
