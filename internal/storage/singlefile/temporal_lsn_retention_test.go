package singlefile

import (
	"context"
	"errors"
	"testing"

	"github.com/xDarkicex/libravdb/internal/storage"
)

func TestResolveLSNClassifiesCompactedAndAbsentLSNs(t *testing.T) {
	ctx := context.Background()
	engine, err := New(t.TempDir() + "/resolve-lsn-retention.libravdb")
	if err != nil {
		t.Fatalf("open engine: %v", err)
	}
	e := engine.(*Engine)
	defer e.Close()

	if _, err := e.CreateCollection("docs", &storage.CollectionConfig{
		Dimension: 1,
		Version:   2,
	}); err != nil {
		t.Fatalf("create collection: %v", err)
	}

	commitPut := func(id string, value float32) storage.CommitReceipt {
		t.Helper()
		prepared, err := e.PrepareTx(ctx, []storage.TxOperation{{
			Type:       storage.TxOperationPut,
			Collection: "docs",
			ID:         id,
			Vector:     []float32{value},
		}})
		if err != nil {
			t.Fatalf("prepare %s: %v", id, err)
		}
		receipt, err := e.CommitTxDurable(ctx, prepared)
		if err != nil {
			t.Fatalf("commit %s: %v", id, err)
		}
		if receipt.CommitLSN == 0 {
			t.Fatalf("commit %s returned zero LSN", id)
		}
		return receipt
	}

	first := commitPut("r1", 1)
	retained := commitPut("r2", 2)
	later := commitPut("r3", 3)
	if !(first.CommitLSN < retained.CommitLSN && retained.CommitLSN < later.CommitLSN) {
		t.Fatalf("commit LSNs are not ordered: first=%d retained=%d later=%d", first.CommitLSN, retained.CommitLSN, later.CommitLSN)
	}

	boundary, err := e.CompactTemporalHistory(retained.CommitLSN)
	if err != nil {
		t.Fatalf("compact temporal history: %v", err)
	}
	if boundary != retained.CommitLSN {
		t.Fatalf("retention boundary=%d, want %d", boundary, retained.CommitLSN)
	}

	if _, err := e.ResolveLSN(first.CommitLSN); !errors.Is(err, ErrRetentionExpired) {
		t.Fatalf("compacted known LSN error=%v, want ErrRetentionExpired", err)
	}
	if _, err := e.ResolveLSN(retained.CommitLSN); err != nil {
		t.Fatalf("retained boundary LSN should resolve: %v", err)
	}

	unknown := retained.CommitLSN + 1
	if unknown >= later.CommitLSN {
		unknown = later.CommitLSN - 1
	}
	if unknown <= retained.CommitLSN || unknown >= later.CommitLSN {
		t.Fatalf("could not choose an absent LSN in retained range: %d, retained=%d, later=%d", unknown, retained.CommitLSN, later.CommitLSN)
	}
	if _, err := e.ResolveLSN(unknown); !errors.Is(err, ErrUnknownLSN) {
		t.Fatalf("absent retained-range LSN error=%v, want ErrUnknownLSN", err)
	}
	if _, err := e.ResolveLSN(later.CommitLSN + 100); !errors.Is(err, ErrUnknownLSN) {
		t.Fatalf("future LSN error=%v, want ErrUnknownLSN", err)
	}
}

func TestRetainedHistoryStatsTrackCompactionAndReopen(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/retained-history-stats.libravdb"
	engine, err := New(path)
	if err != nil {
		t.Fatalf("open engine: %v", err)
	}
	e := engine.(*Engine)

	if _, err := e.CreateCollection("docs", &storage.CollectionConfig{Dimension: 1, Version: 2}); err != nil {
		t.Fatalf("create collection: %v", err)
	}
	commitPut := func(value float32) storage.CommitReceipt {
		t.Helper()
		prepared, err := e.PrepareTx(ctx, []storage.TxOperation{{
			Type: storage.TxOperationPut, Collection: "docs", ID: "r1", Vector: []float32{value},
		}})
		if err != nil {
			t.Fatalf("prepare value %v: %v", value, err)
		}
		receipt, err := e.CommitTxDurable(ctx, prepared)
		if err != nil {
			t.Fatalf("commit value %v: %v", value, err)
		}
		return receipt
	}

	commitPut(1)
	second := commitPut(2)
	versions, bytes, err := e.RetainedHistoryStats()
	if err != nil {
		t.Fatalf("history stats: %v", err)
	}
	if versions != 1 || bytes == 0 {
		t.Fatalf("history stats = versions=%d bytes=%d, want one non-empty archived version", versions, bytes)
	}

	if boundary, err := e.CompactTemporalHistory(second.CommitLSN); err != nil {
		t.Fatalf("compact temporal history: %v", err)
	} else if boundary != second.CommitLSN {
		t.Fatalf("retention boundary=%d, want %d", boundary, second.CommitLSN)
	}
	versions, bytes, err = e.RetainedHistoryStats()
	if err != nil {
		t.Fatalf("post-compaction history stats: %v", err)
	}
	if versions != 0 || bytes != 0 {
		t.Fatalf("post-compaction history stats = versions=%d bytes=%d, want zero", versions, bytes)
	}
	if err := e.Close(); err != nil {
		t.Fatalf("close engine: %v", err)
	}

	reopenedRaw, err := New(path)
	if err != nil {
		t.Fatalf("reopen engine: %v", err)
	}
	reopened := reopenedRaw.(*Engine)
	defer reopened.Close()
	versions, bytes, err = reopened.RetainedHistoryStats()
	if err != nil {
		t.Fatalf("reopened history stats: %v", err)
	}
	if versions != 0 || bytes != 0 {
		t.Fatalf("reopened history stats = versions=%d bytes=%d, want zero", versions, bytes)
	}
}
