package wal

import (
	"context"
	"path/filepath"
	"testing"
)

func TestAppendBatchPersistsAllEntries(t *testing.T) {
	w, err := New(filepath.Join(t.TempDir(), "test.wal"))
	if err != nil {
		t.Fatalf("failed to create WAL: %v", err)
	}
	defer w.Close()

	entries := []*Entry{
		{Operation: OpInsert, ID: "a", Vector: []float32{1, 2, 3}},
		{Operation: OpInsert, ID: "b", Vector: []float32{4, 5, 6}},
		{Operation: OpDelete, ID: "a"},
	}

	if err := w.AppendBatch(context.Background(), entries); err != nil {
		t.Fatalf("append batch failed: %v", err)
	}

	recovered, err := w.Read()
	if err != nil {
		t.Fatalf("read failed: %v", err)
	}

	if len(recovered) != len(entries) {
		t.Fatalf("expected %d recovered entries, got %d", len(entries), len(recovered))
	}

	for i, entry := range recovered {
		if entry.ID != entries[i].ID {
			t.Fatalf("entry %d id mismatch: got %q want %q", i, entry.ID, entries[i].ID)
		}
		if entry.Operation != entries[i].Operation {
			t.Fatalf("entry %d op mismatch: got %v want %v", i, entry.Operation, entries[i].Operation)
		}
	}
}
