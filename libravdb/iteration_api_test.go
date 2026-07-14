package libravdb

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/memory"
	"github.com/xDarkicex/libravdb/internal/quant"
)

func TestDatabaseIterateStreamsAllCollections(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("open database: %v", err)
	}
	defer db.Close()

	for _, name := range []string{"beta", "alpha"} {
		collection, err := db.CreateCollection(ctx, name, WithDimension(2), WithFlat())
		if err != nil {
			t.Fatalf("create collection %q: %v", name, err)
		}
		for i := 0; i < 2; i++ {
			id := fmt.Sprintf("%s-%d", name, i)
			if err := collection.Insert(ctx, id, []float32{float32(i), 1}, map[string]interface{}{"collection": name}); err != nil {
				t.Fatalf("insert %q: %v", id, err)
			}
		}
	}

	var got []string
	err = db.Iterate(ctx, func(collection string, record Record) error {
		got = append(got, collection+"/"+record.ID)
		return nil
	})
	if err != nil {
		t.Fatalf("iterate database: %v", err)
	}

	want := []string{"alpha/alpha-0", "alpha/alpha-1", "beta/beta-0", "beta/beta-1"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("iteration order mismatch:\n got %v\nwant %v", got, want)
	}
}

func TestDatabaseIteratePropagatesCallbackAndContextErrors(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("open database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, "records", WithDimension(2), WithFlat())
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}
	if err := collection.Insert(ctx, "record", []float32{1, 2}, nil); err != nil {
		t.Fatalf("insert record: %v", err)
	}
	if err := collection.Insert(ctx, "record-2", []float32{2, 1}, nil); err != nil {
		t.Fatalf("insert second record: %v", err)
	}

	sentinel := errors.New("consumer stopped")
	if err := db.Iterate(ctx, func(string, Record) error { return sentinel }); !errors.Is(err, sentinel) {
		t.Fatalf("callback error = %v, want %v", err, sentinel)
	}
	if err := db.Iterate(ctx, nil); err == nil {
		t.Fatal("expected nil callback error")
	}

	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if err := db.Iterate(canceled, func(string, Record) error { return nil }); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled iteration error = %v, want %v", err, context.Canceled)
	}

	canceled, cancel = context.WithCancel(ctx)
	callbacks := 0
	err = db.Iterate(canceled, func(string, Record) error {
		callbacks++
		cancel()
		return nil
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-stream cancellation error = %v, want %v", err, context.Canceled)
	}
	if callbacks != 1 {
		t.Fatalf("received %d callbacks after mid-stream cancellation, want 1", callbacks)
	}
}

func TestCollectionIterateUsesSnapshotOrdinalFrontier(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("open database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, "records", WithDimension(2), WithFlat())
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}

	entries := make([]VectorEntry, 1100)
	for i := range entries {
		entries[i] = VectorEntry{ID: fmt.Sprintf("record-%04d", i), Vector: []float32{float32(i), 1}}
	}
	if err := collection.InsertBatch(ctx, entries); err != nil {
		t.Fatalf("insert batch: %v", err)
	}

	count := 0
	err = collection.Iterate(ctx, func(Record) error {
		count++
		if count == 1 {
			return collection.Insert(ctx, "late-record", []float32{1, 1}, nil)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("iterate collection: %v", err)
	}
	if count != len(entries) {
		t.Fatalf("iterated %d records, want %d; post-frontier insert leaked into stream", count, len(entries))
	}
}

func TestDatabaseIterateIncludesShardedRecords(t *testing.T) {
	ctx := context.Background()
	db, err := Open(WithStoragePath(testDBPath(t)))
	if err != nil {
		t.Fatalf("open database: %v", err)
	}
	defer db.Close()

	collection, err := db.CreateCollection(ctx, "sharded", WithDimension(2), WithFlat(), WithSharding(true))
	if err != nil {
		t.Fatalf("create sharded collection: %v", err)
	}

	const recordCount = 64
	entries := make([]VectorEntry, recordCount)
	for i := range entries {
		entries[i] = VectorEntry{ID: fmt.Sprintf("record-%02d", i), Vector: []float32{float32(i), 1}}
	}
	if err := collection.InsertBatch(ctx, entries); err != nil {
		t.Fatalf("insert sharded batch: %v", err)
	}

	seen := make(map[string]struct{}, recordCount)
	if err := db.Iterate(ctx, func(collectionName string, record Record) error {
		if collectionName != "sharded" {
			t.Fatalf("unexpected collection name %q", collectionName)
		}
		seen[record.ID] = struct{}{}
		return nil
	}); err != nil {
		t.Fatalf("iterate database: %v", err)
	}
	if len(seen) != recordCount {
		t.Fatalf("iterated %d unique sharded records, want %d", len(seen), recordCount)
	}
}

func TestCollectionConfigReturnsDefensiveCopy(t *testing.T) {
	collection := &Collection{config: &CollectionConfig{
		MetadataSchema: MetadataSchema{"source": StringField},
		IndexedFields:  []string{"source"},
		MemoryConfig: &memory.MemoryConfig{
			PressureThresholds: map[memory.MemoryPressureLevel]float64{memory.LowPressure: 0.7},
			MonitorInterval:    time.Second,
		},
		Quantization: &quant.QuantizationConfig{Levels: []int{8, 8}},
		Dimension:    2,
	}}

	config := collection.Config()
	config.MetadataSchema["source"] = IntField
	config.IndexedFields[0] = "changed"
	config.MemoryConfig.PressureThresholds[memory.LowPressure] = 0.1
	config.Quantization.Levels[0] = 2

	fresh := collection.Config()
	if fresh.MetadataSchema["source"] != StringField {
		t.Fatal("metadata schema mutation changed collection configuration")
	}
	if fresh.IndexedFields[0] != "source" {
		t.Fatal("indexed fields mutation changed collection configuration")
	}
	if fresh.MemoryConfig.PressureThresholds[memory.LowPressure] != 0.7 {
		t.Fatal("memory thresholds mutation changed collection configuration")
	}
	if fresh.Quantization.Levels[0] != 8 {
		t.Fatal("quantization levels mutation changed collection configuration")
	}
}
