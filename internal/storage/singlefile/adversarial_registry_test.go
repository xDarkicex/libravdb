package singlefile

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/xDarkicex/libravdb/internal/index"
	"github.com/xDarkicex/libravdb/internal/storage"
	"github.com/xDarkicex/libravdb/internal/util"
)

func tempDB(tb testing.TB) string {
	tb.Helper()
	dir := tb.TempDir()
	return dir + "/test.libravdb"
}

func openEngine(tb testing.TB, path string) (*Engine, func()) {
	tb.Helper()
	eng, err := New(path)
	if err != nil {
		tb.Fatalf("open engine: %v", err)
	}
	e, ok := eng.(*Engine)
	if !ok {
		tb.Fatalf("expected *singlefile.Engine")
	}
	return e, func() {
		eng.Close()
		os.Remove(path)
	}
}

// 1. Direct Paths produce nonzero IDs
func TestAdversarialGraphNodeID_InsertBatch(t *testing.T) {
	path := tempDB(t)
	e, cleanup := openEngine(t, path)
	defer cleanup()

	col, err := e.CreateCollection("colA", &storage.CollectionConfig{
		Dimension: 4, Metric: int(0), IndexType: int(2),
	})
	if err != nil {
		t.Fatalf("create col: %v", err)
	}

	ctx := context.Background()

	// Single Insert
	entA := &index.VectorEntry{ID: "r1", Vector: []float32{1, 0, 0, 0}}
	if err := col.Insert(ctx, entA); err != nil {
		t.Fatalf("insert: %v", err)
	}
	if entA.GraphNodeID == 0 {
		t.Errorf("Insert failed to assign non-zero GraphNodeID")
	}

	// Batch Insert
	entB := &index.VectorEntry{ID: "r2", Vector: []float32{0, 1, 0, 0}}
	if err := col.InsertBatch(ctx, []*index.VectorEntry{entB}); err != nil {
		t.Fatalf("insert batch: %v", err)
	}
	if entB.GraphNodeID == 0 {
		t.Errorf("InsertBatch failed to assign non-zero GraphNodeID")
	}
}

// 2. Update preserves ID
func TestAdversarialGraphNodeID_UpdatePreservesIdentity(t *testing.T) {
	path := tempDB(t)
	e, cleanup := openEngine(t, path)
	defer cleanup()

	col, _ := e.CreateCollection("col", &storage.CollectionConfig{Dimension: 4})
	ctx := context.Background()
	ent := &index.VectorEntry{ID: "r1", Vector: []float32{1, 0, 0, 0}}
	col.Insert(ctx, ent)
	firstID := ent.GraphNodeID

	entUpdate := &index.VectorEntry{ID: "r1", Vector: []float32{0, 1, 0, 0}}
	col.Insert(ctx, entUpdate) // acts as update because ID is same
	if entUpdate.GraphNodeID != firstID {
		t.Errorf("Update changed GraphNodeID: got %d, want %d", entUpdate.GraphNodeID, firstID)
	}
}

// 3. Delete/reinsert receives fresh identity
func TestAdversarialGraphNodeID_DeleteReinsertFreshIdentity(t *testing.T) {
	path := tempDB(t)
	e, cleanup := openEngine(t, path)
	defer cleanup()

	col, _ := e.CreateCollection("col", &storage.CollectionConfig{Dimension: 4})
	ctx := context.Background()
	ent := &index.VectorEntry{ID: "r1", Vector: []float32{1, 0, 0, 0}}
	col.Insert(ctx, ent)
	firstID := ent.GraphNodeID

	if err := col.Delete(ctx, "r1"); err != nil {
		t.Fatalf("delete: %v", err)
	}

	entReinsert := &index.VectorEntry{ID: "r1", Vector: []float32{0, 1, 0, 0}}
	col.Insert(ctx, entReinsert)
	if entReinsert.GraphNodeID == firstID {
		t.Errorf("Delete/Reinsert reused tombstoned ID %d", firstID)
	}
	if entReinsert.GraphNodeID == 0 {
		t.Errorf("Delete/Reinsert failed to assign non-zero ID")
	}
}

// 5. Aborted and failed commits do not publish an ID
func TestAdversarialGraphNodeID_AbortedCommitsNoGap(t *testing.T) {
	path := tempDB(t)
	e, cleanup := openEngine(t, path)
	defer cleanup()

	col, _ := e.CreateCollection("col", &storage.CollectionConfig{Dimension: 4})
	ctx := context.Background()

	// Start transaction and abort
	ops := []storage.TxOperation{
		{Collection: "col", ID: "tx1", Vector: []float32{1, 0, 0, 0}, Type: storage.TxOperationPut},
	}
	ops, err := e.PrepareTx(ctx, ops)
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}
	// We simulate abort by dropping the ops slice and not calling CommitTx.

	// Now insert a normal record.
	ent := &index.VectorEntry{ID: "r1", Vector: []float32{1, 0, 0, 0}}
	col.Insert(ctx, ent)

	// Since first ID was aborted, it should not have consumed a permanent ID if it wasn't committed.
	// We assume that the ID for r1 will be 1 (if the engine hasn't wasted one).
	if ent.GraphNodeID != 1 {
		t.Errorf("Aborted commit wasted a GraphNodeID gap. Got %d, want 1", ent.GraphNodeID)
	}
}

// 7. Duplicate-key operations inside one transaction
func TestAdversarialGraphNodeID_DuplicateKeyInTx(t *testing.T) {
	path := tempDB(t)
	e, cleanup := openEngine(t, path)
	defer cleanup()

	e.CreateCollection("col", &storage.CollectionConfig{Dimension: 4})
	ctx := context.Background()

	ops := []storage.TxOperation{
		{Collection: "col", ID: "dup", Vector: []float32{1, 0, 0, 0}, Type: storage.TxOperationPut},
		{Collection: "col", ID: "dup", Vector: []float32{0, 1, 0, 0}, Type: storage.TxOperationPut},
	}
	ops, _ = e.PrepareTx(ctx, ops)
	err := e.CommitTx(ctx, ops)
	if err != nil {
		t.Fatalf("commit: %v", err)
	}

	if ops[0].GraphNodeID != ops[1].GraphNodeID {
		t.Errorf("Duplicate keys in same tx must resolve to the same GraphNodeID! Got %d and %d", ops[0].GraphNodeID, ops[1].GraphNodeID)
	}
}

// 8. MaxUint64 overflow without wrapping to zero
func TestAdversarialGraphNodeID_MaxUint64Overflow(t *testing.T) {
	path := tempDB(t)
	e, cleanup := openEngine(t, path)
	defer cleanup()

	col, _ := e.CreateCollection("col", &storage.CollectionConfig{Dimension: 4})

	// Force the allocator state to MaxUint64
	e.mu.Lock()
	e.state.NextGraphNodeID = math.MaxUint64
	e.nextGraphNodeID.Store(math.MaxUint64)
	e.mu.Unlock()

	ctx := context.Background()
	ent := &index.VectorEntry{ID: "overflow", Vector: []float32{1, 0, 0, 0}}

	err := col.Insert(ctx, ent)
	if err == nil {
		t.Errorf("Expected error on MaxUint64 overflow, but insertion succeeded with ID %d", ent.GraphNodeID)
	}
}

// 12. Reverse API & Tombstone vs Unknown
func TestAdversarialGraphNodeID_ReverseAPI(t *testing.T) {
	path := tempDB(t)
	e, cleanup := openEngine(t, path)
	defer cleanup()

	col, _ := e.CreateCollection("col", &storage.CollectionConfig{Dimension: 4})
	ctx := context.Background()
	ent := &index.VectorEntry{ID: "r1", Vector: []float32{1, 0, 0, 0}}
	col.Insert(ctx, ent)
	id := ent.GraphNodeID

	if id == 0 {
		t.Fatalf("ID is 0")
	}

	// Unknown resolution
	_, _, err := e.ResolveNodeID(ctx, 99999)
	if err != storage.ErrUnknownGraphNodeID {
		t.Errorf("Expected ErrUnknownGraphNodeID for 99999, got %v", err)
	}

	// Live resolution
	colName, recID, err := e.ResolveNodeID(ctx, id)
	if err != nil {
		t.Errorf("Failed to resolve live ID: %v", err)
	}
	if colName != "col" || recID != "r1" {
		t.Errorf("Resolved to wrong record: %s %s", colName, recID)
	}

	// Tombstone resolution
	col.Delete(ctx, "r1")
	_, _, err = e.ResolveNodeID(ctx, id)
	if err != storage.ErrTombstonedGraphNodeID {
		t.Errorf("Expected ErrTombstonedGraphNodeID for deleted ID, got %v", err)
	}
}

// 11. Two collections local ordinal collision
func TestAdversarialGraphNodeID_CollidingOrdinals(t *testing.T) {
	path := tempDB(t)
	e, cleanup := openEngine(t, path)
	defer cleanup()

	colA, _ := e.CreateCollection("colA", &storage.CollectionConfig{Dimension: 4})
	colB, _ := e.CreateCollection("colB", &storage.CollectionConfig{Dimension: 4})

	ctx := context.Background()
	entA := &index.VectorEntry{ID: "rA", Vector: []float32{1, 0, 0, 0}}
	entB := &index.VectorEntry{ID: "rB", Vector: []float32{0, 1, 0, 0}}

	colA.Insert(ctx, entA)
	colB.Insert(ctx, entB)

	if entA.Ordinal != entB.Ordinal {
		t.Fatalf("Expected ordinals to collide (0 == 0). Got %d and %d", entA.Ordinal, entB.Ordinal)
	}

	if entA.GraphNodeID == entB.GraphNodeID {
		t.Errorf("GraphNodeIDs must not alias! Both got %d", entA.GraphNodeID)
	}
}

// 18. Concurrent Race
func TestAdversarialGraphNodeID_ConcurrentRace(t *testing.T) {
	path := tempDB(t)
	e, cleanup := openEngine(t, path)
	defer cleanup()

	col, _ := e.CreateCollection("col", &storage.CollectionConfig{Dimension: 4})
	ctx := context.Background()

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ent := &index.VectorEntry{ID: "r", Vector: []float32{1, 0, 0, 0}}
			// Same key concurrently inserted resolves to one committed identity
			_ = col.Insert(ctx, ent)
		}(i)
	}
	wg.Wait()

	// Check identity stability
	ent := &index.VectorEntry{ID: "r", Vector: []float32{1, 0, 0, 0}}
	col.Insert(ctx, ent) // Update

	if ent.GraphNodeID == 0 {
		t.Errorf("ID is 0 after concurrent inserts")
	}
}

// 22. V2 checkpoint failure prevents Ready: inject file-replace failure
// after V2 migration assigns IDs but before checkpoint commits.
func TestAdversarialGraphNodeID_V2CheckpointFailure(t *testing.T) {
	path := tempDB(t)

	// STEP 1: Create a valid V2 database using the same pattern as
	// TestAdversarialGraphNodeID_LegacyV2Migration.
	eng1, err := New(path)
	if err != nil {
		t.Fatalf("create engine: %v", err)
	}
	e1 := eng1.(*Engine)

	enc := util.AcquireBinaryEncoder(1024)
	enc.WriteByte(2)   // version = 2
	enc.WriteUint64(2) // NextCollectionID
	enc.WriteUint32(1) // count
	enc.WriteString("testcol")
	enc.WriteUint64(1) // ID
	config := storage.CollectionConfig{Dimension: 4}
	writeCollectionConfig(enc, config)
	enc.WriteUint64(0)
	enc.WriteUint64(0)
	enc.WriteBool(false)
	enc.WriteUint64(1) // LiveCount
	enc.WriteUint32(1) // NextOrdinal
	enc.WriteUint32(1) // RecordCount
	enc.WriteString("legacy_rec")
	enc.WriteUint64(1)
	enc.WriteUint64(0)
	enc.WriteUint64(0)
	enc.WriteBool(false)
	enc.WriteUint32(0) // Ordinal
	enc.WriteVector([]float32{1.0, 1.0, 1.0, 1.0})
	enc.WriteMetadata(map[string]interface{}{})
	payloadBytes := enc.DetachBytes()

	e1.mu.Lock()
	offset, err := e1.appendChunkLocked(chunkTypeSnapshot, payloadBytes)
	if err != nil {
		e1.mu.Unlock()
		t.Fatalf("append chunk: %v", err)
	}
	meta, err := e1.readMetaPage(e1.activeMetaPage)
	if err != nil {
		e1.mu.Unlock()
		t.Fatalf("read meta: %v", err)
	}
	meta.SnapshotOffset = offset
	meta.SnapshotLength = uint64(len(payloadBytes))
	buf := make([]byte, 4096)
	if err := writeFixedPage(e1.file, e1.activeMetaPage, encodeMeta(meta, buf)); err != nil {
		e1.mu.Unlock()
		t.Fatalf("write meta: %v", err)
	}
	e1.mu.Unlock()
	e1.Close()

	// STEP 2: Inject failure into the checkpoint write seam. The V2
	// migration checkpoint must issue a metapage write during open;
	// crash any checkpoint write that targets a metapage page (1 or 2).
	var checkpointWrites int32
	writeHook := func(offset int64, data []byte) (int, error) {
		page := offset / pageSize
		if page == 1 || page == 2 {
			atomic.AddInt32(&checkpointWrites, 1)
			return 0, os.ErrPermission
		}
		return len(data), nil
	}

	// STEP 3: Open — migration assigns IDs, checkpoint crashes on the
	// metapage write. New must return an error and must not reach
	// StatusReady, so no nondurable generated IDs are served.
	_, err = New(path, withCheckpointFaultHooks(writeHook, nil))
	t.Logf("checkpoint metapage writes attempted: %d", atomic.LoadInt32(&checkpointWrites))
	if atomic.LoadInt32(&checkpointWrites) == 0 {
		t.Errorf("checkpoint never issued a metapage write — V2 migration checkpoint did not execute")
	}
	if err == nil {
		t.Errorf("expected open failure when V2 migration checkpoint write fails")
	} else {
		t.Logf("correctly blocked open: %v", err)
	}

	// STEP 4: Verify the engine opens cleanly without the hook, and that
	// migration re-runs deterministically (the failed attempt must not
	// have durably committed).
	eng2, err := New(path)
	if err != nil {
		t.Fatalf("subsequent open failed: %v", err)
	}
	e2 := eng2.(*Engine)
	defer e2.Close()
	ctx := context.Background()
	gnid, err := e2.GetNodeID(ctx, "testcol", "legacy_rec")
	if err != nil {
		t.Fatalf("GetNodeID after clean open: %v", err)
	}
	if gnid == 0 {
		t.Errorf("legacy record still has GraphNodeID 0 after clean open")
	}
}

// 9. Legacy V2 Snapshot Migration
func TestAdversarialGraphNodeID_LegacyV2Migration(t *testing.T) {
	path := tempDB(t)

	// Create a valid V2 snapshot payload using the engine's util encoder
	enc := util.AcquireBinaryEncoder(1024)

	enc.WriteByte(2)   // version = 2
	enc.WriteUint64(2) // NextCollectionID
	enc.WriteUint32(1) // count

	// Collection "testcol"
	enc.WriteString("testcol")
	enc.WriteUint64(1) // ID

	config := storage.CollectionConfig{
		Dimension: 4,
	}
	// writeCollectionConfig is internal to package singlefile
	writeCollectionConfig(enc, config)

	enc.WriteUint64(0)   // CreatedLSN
	enc.WriteUint64(0)   // UpdatedLSN
	enc.WriteBool(false) // Deleted
	enc.WriteUint64(1)   // LiveCount
	enc.WriteUint32(1)   // NextOrdinal
	enc.WriteUint32(1)   // RecordCount

	// Record "legacy_rec"
	enc.WriteString("legacy_rec")
	enc.WriteUint64(1)   // Version
	enc.WriteUint64(0)   // CreatedLSN
	enc.WriteUint64(0)   // UpdatedLSN
	enc.WriteBool(false) // Deleted
	enc.WriteUint32(0)   // Ordinal
	// NO GraphNodeID! (since version = 2)
	enc.WriteVector([]float32{1.0, 1.0, 1.0, 1.0})
	enc.WriteMetadata(map[string]interface{}{})

	payloadBytes := enc.DetachBytes()

	// Open engine to get a valid empty container
	eng, err := New(path)
	if err != nil {
		t.Fatalf("create empty engine: %v", err)
	}
	e := eng.(*Engine)

	// We must write the V2 payload as a snapshot chunk and update metapage.
	// We can cheat by acquiring the engine lock, appending a snapshot chunk, and updating metapage.
	e.mu.Lock()
	offset, err := e.appendChunkLocked(chunkTypeSnapshot, payloadBytes)
	if err != nil {
		t.Fatalf("append v2 snapshot chunk: %v", err)
	}

	meta, err := e.readMetaPage(e.activeMetaPage)
	if err != nil {
		t.Fatalf("read meta: %v", err)
	}
	meta.SnapshotOffset = offset
	meta.SnapshotLength = uint64(len(payloadBytes))

	buf := make([]byte, 4096)
	err = writeFixedPage(e.file, e.activeMetaPage, encodeMeta(meta, buf))
	if err != nil {
		t.Fatalf("write meta: %v", err)
	}
	e.mu.Unlock()
	e.Close()

	if err != nil {
		t.Fatalf("write metapage: %v", err)
	}

	// Now reopen the engine. This will trigger recovery of the V2 snapshot.
	eng2, err := New(path)
	if err != nil {
		t.Fatalf("reopen engine with v2 snapshot: %v", err)
	}
	e2 := eng2.(*Engine)
	defer e2.Close()

	// Verify migration happened.
	ctx := context.Background()
	if errVal := e2.recoveryErr.Load(); errVal != nil {
		t.Logf("Recovery Error: %v", errVal)
	}

	gnid, err := e2.GetNodeID(ctx, "testcol", "legacy_rec")
	if err != nil {
		if col, ok := e2.state.Collections["testcol"]; ok {
			t.Logf("testcol exists. Records count: %d", len(col.Records))
			for id, rec := range col.Records {
				t.Logf("Record ID: %q, Ordinal: %d, GraphNodeID: %d", id, rec.Ordinal, rec.GraphNodeID)
			}
		} else {
			t.Logf("testcol does not exist in state!")
		}
		t.Fatalf("Failed to get node ID for migrated record: %v", err)
	}
	if gnid == 0 {
		t.Errorf("Migrated V2 record still has GraphNodeID 0")
	}

	// Test deterministic assignment across reopens.
	e2.Close()
	eng3, _ := New(path)
	e3 := eng3.(*Engine)
	defer e3.Close()
	gnid2, _ := e3.GetNodeID(ctx, "testcol", "legacy_rec")
	if gnid2 != gnid {
		t.Errorf("GraphNodeID changed after reopen! Expected %d, got %d", gnid, gnid2)
	}

	// Assert the resulting snapshot is rewritten using the current snapshot
	// codec. Legacy v2 input must never force a downgrade: the current encoder
	// carries newer persisted state (for example graph tombstones and the
	// temporal commit catalog) that must survive migration.
	meta3, err := e3.readMetaPage(e3.activeMetaPage)
	if err != nil {
		t.Fatalf("Failed to read metapage: %v", err)
	}
	snapshotChunk, err := e3.readChunkAt(meta3.SnapshotOffset)
	if err != nil {
		t.Fatalf("Failed to read snapshot chunk: %v", err)
	}
	if len(snapshotChunk) == 0 {
		t.Fatalf("Snapshot chunk is empty")
	}
	if snapshotChunk[0] != snapshotCodecVersion {
		t.Errorf("Expected migrated snapshot to be current snapshot codec v%d, got %d", snapshotCodecVersion, snapshotChunk[0])
	}
}

// 11. Sync failure after WAL append is an ambiguous commit. The engine must
// gate ALL further writes on the live instance — no retry, no candidate-ID
// reuse — until close and reopen. The failed insert must not leak an ID to
// the caller, and the refused retry must not receive one either.
func TestAdversarialGraphNodeID_SyncFailure(t *testing.T) {
	path := tempDB(t)
	e, cleanup := openEngine(t, path)
	defer cleanup()

	col, _ := e.CreateCollection("col", &storage.CollectionConfig{Dimension: 4})
	ctx := context.Background()

	// Install a test hook that fails the sync
	installTestFaultHooks(e, nil, func(*os.File) error {
		return os.ErrClosed // simulate failure
	})

	ent := &index.VectorEntry{ID: "r1", Vector: []float32{1, 0, 0, 0}}
	err := col.Insert(ctx, ent)
	if err == nil {
		t.Fatalf("expected sync error, got nil")
	}

	if ent.GraphNodeID != 0 {
		t.Errorf("failed insert leaked a GraphNodeID to the caller's entry: %d", ent.GraphNodeID)
	}

	// Clear the hook but do NOT close/reopen: the ambiguous transaction may
	// have reached the device. Every mutation path must now refuse writes.
	installTestFaultHooks(e, nil, nil)
	ent2 := &index.VectorEntry{ID: "r2", Vector: []float32{1, 0, 0, 0}}
	err = col.Insert(ctx, ent2)
	if err == nil {
		t.Fatalf("same-process retry after ambiguous sync must be refused")
	}
	if !errors.Is(err, errRecoveryRequired) {
		t.Errorf("expected errRecoveryRequired, got %v", err)
	}
	if ent2.GraphNodeID != 0 {
		t.Errorf("refused retry received a GraphNodeID: %d", ent2.GraphNodeID)
	}

	// Batch, transaction, delete, and collection mutations must refuse too.
	err = col.InsertBatch(ctx, []*index.VectorEntry{{ID: "r3", Vector: []float32{0, 0, 1, 0}}})
	if !errors.Is(err, errRecoveryRequired) {
		t.Errorf("InsertBatch after ambiguous sync: expected errRecoveryRequired, got %v", err)
	}
	err = e.CommitTx(ctx, []storage.TxOperation{
		{Collection: "col", ID: "r4", Vector: []float32{0, 0, 0, 1}, Type: storage.TxOperationPut},
	})
	if !errors.Is(err, errRecoveryRequired) {
		t.Errorf("CommitTx after ambiguous sync: expected errRecoveryRequired, got %v", err)
	}
	err = col.Delete(ctx, "r1")
	if !errors.Is(err, errRecoveryRequired) {
		t.Errorf("Delete after ambiguous sync: expected errRecoveryRequired, got %v", err)
	}
	_, err = e.CreateCollection("col2", &storage.CollectionConfig{Dimension: 4})
	if !errors.Is(err, errRecoveryRequired) {
		t.Errorf("CreateCollection after ambiguous sync: expected errRecoveryRequired, got %v", err)
	}
	err = e.DeleteCollection("col")
	if !errors.Is(err, errRecoveryRequired) {
		t.Errorf("DeleteCollection after ambiguous sync: expected errRecoveryRequired, got %v", err)
	}

	// Close and reopen. The ambiguous transaction either committed or did
	// not; recovery reconciles. Every replayed/new ID must resolve uniquely.
	e.Close()
	eng, err := New(path)
	if err != nil {
		t.Fatalf("reopen after ambiguous sync: %v", err)
	}
	e2 := eng.(*Engine)
	defer e2.Close()

	col2, err := e2.GetCollection("col")
	if err != nil {
		t.Fatalf("get collection after reopen: %v", err)
	}
	ent3 := &index.VectorEntry{ID: "r5", Vector: []float32{1, 0, 0, 0}}
	if err := col2.Insert(ctx, ent3); err != nil {
		t.Fatalf("insert after reopen: %v", err)
	}
	if ent3.GraphNodeID == 0 {
		t.Fatalf("post-reopen insert got zero ID")
	}

	// Collect every live record's GraphNodeID and prove uniqueness of
	// resolution: each ID resolves to exactly one (collection, record).
	seen := make(map[uint64]string)
	var dupErr error
	iterErr := col2.Iterate(ctx, func(entry *index.VectorEntry) error {
		if entry.GraphNodeID == 0 {
			return fmt.Errorf("record %s has zero GraphNodeID", entry.ID)
		}
		if prev, ok := seen[entry.GraphNodeID]; ok {
			dupErr = fmt.Errorf("GraphNodeID %d aliases records %s and %s", entry.GraphNodeID, prev, entry.ID)
			return dupErr
		}
		seen[entry.GraphNodeID] = entry.ID
		colName, recID, rerr := e2.ResolveNodeID(ctx, entry.GraphNodeID)
		if rerr != nil {
			return fmt.Errorf("resolve %d: %w", entry.GraphNodeID, rerr)
		}
		if colName != "col" || recID != entry.ID {
			return fmt.Errorf("ID %d resolves to %s/%s, want col/%s", entry.GraphNodeID, colName, recID, entry.ID)
		}
		return nil
	})
	if dupErr != nil {
		t.Errorf("%v", dupErr)
	}
	if iterErr != nil && dupErr == nil {
		t.Errorf("iterate: %v", iterErr)
	}
}

// 12. Write Failure injection ensures IDs do not leak
func TestAdversarialGraphNodeID_WriteFailure(t *testing.T) {
	path := tempDB(t)
	e, cleanup := openEngine(t, path)
	defer cleanup()

	col, _ := e.CreateCollection("col", &storage.CollectionConfig{Dimension: 4})
	ctx := context.Background()

	// Install a test hook that fails the write
	installTestFaultHooks(e, func(b []byte) (int, error) {
		return 0, os.ErrPermission // simulate failure
	}, nil)

	ent := &index.VectorEntry{ID: "r1", Vector: []float32{1, 0, 0, 0}}
	err := col.Insert(ctx, ent)
	if err == nil {
		t.Fatalf("expected write error, got nil")
	}

	if ent.GraphNodeID != 0 {
		t.Errorf("failed insert leaked a GraphNodeID to the caller's entry: %d", ent.GraphNodeID)
	}

	// Remove hook
	installTestFaultHooks(e, nil, nil)
	ent2 := &index.VectorEntry{ID: "r2", Vector: []float32{1, 0, 0, 0}}
	err2 := col.Insert(ctx, ent2)
	if err2 != nil {
		t.Errorf("ent2 insert failed: %v", err2)
	}

	if ent2.GraphNodeID != 1 {
		t.Errorf("GraphNodeID allocator leaked IDs during failed write! Expected 1, got %d", ent2.GraphNodeID)
	}
}

// 13. NUL Aliasing vulnerability test
func TestAdversarialGraphNodeID_NULAliasing(t *testing.T) {
	path := tempDB(t)
	e, cleanup := openEngine(t, path)
	defer cleanup()

	ctx := context.Background()
	e.CreateCollection("a\x00", &storage.CollectionConfig{Dimension: 4})
	e.CreateCollection("a", &storage.CollectionConfig{Dimension: 4})

	// Create alias:
	// "a\x00" + "b" == "a\x00b"
	// "a" + "\x00b" == "a\x00b"
	err := e.CommitTx(ctx, []storage.TxOperation{
		{Type: storage.TxOperationPut, Collection: "a\x00", ID: "b", Vector: []float32{1, 0, 0, 0}},
		{Type: storage.TxOperationPut, Collection: "a", ID: "\x00b", Vector: []float32{0, 1, 0, 0}},
	})
	if err != nil {
		t.Fatalf("commit failed: %v", err)
	}

	id1, _ := e.GetNodeID(ctx, "a\x00", "b")
	id2, _ := e.GetNodeID(ctx, "a", "\x00b")
	if id1 == id2 {
		t.Errorf("NUL aliasing vulnerability! Different records got the same GraphNodeID: %d", id1)
	}
}

// 14. Tombstone preserved across restart
func TestAdversarialGraphNodeID_TombstoneSurvivesRestart(t *testing.T) {
	path := tempDB(t)
	e, cleanup := openEngine(t, path)
	defer cleanup()

	col, _ := e.CreateCollection("col", &storage.CollectionConfig{Dimension: 4})
	ctx := context.Background()
	ent := &index.VectorEntry{ID: "r1", Vector: []float32{1, 0, 0, 0}}
	col.Insert(ctx, ent)
	id := ent.GraphNodeID

	col.Delete(ctx, "r1")
	// After delete, resolve should return tombstoned
	_, _, err := e.ResolveNodeID(ctx, id)
	if err != storage.ErrTombstonedGraphNodeID {
		t.Errorf("Expected ErrTombstonedGraphNodeID after delete, got %v", err)
	}

	// Close and reopen
	e.Close()
	eng, err := New(path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	e2 := eng.(*Engine)
	defer e2.Close()

	// After reopen, deleted ID should still be tombstoned, not unknown
	_, _, err = e2.ResolveNodeID(ctx, id)
	if err != storage.ErrTombstonedGraphNodeID {
		t.Errorf("Tombstone lost across restart: expected ErrTombstonedGraphNodeID, got %v", err)
	}
}

// 15. Short write fails without ID leak
func TestAdversarialGraphNodeID_ShortWrite(t *testing.T) {
	path := tempDB(t)
	e, cleanup := openEngine(t, path)
	defer cleanup()

	col, _ := e.CreateCollection("col", &storage.CollectionConfig{Dimension: 4})
	ctx := context.Background()

	// Inject a short-write hook (write 0 bytes, no error)
	installTestFaultHooks(e, func(b []byte) (int, error) {
		return 0, nil // short write: wrote 0 of N bytes
	}, nil)

	ent := &index.VectorEntry{ID: "r1", Vector: []float32{1, 0, 0, 0}}
	err := col.Insert(ctx, ent)
	if err == nil {
		t.Fatalf("expected short write error, got nil")
	}

	if ent.GraphNodeID != 0 {
		t.Errorf("short write leaked GraphNodeID %d", ent.GraphNodeID)
	}

	// A short write is also ambiguous: an unknown prefix may have reached
	// the file. The engine must refuse further writes on this instance.
	installTestFaultHooks(e, nil, nil)
	ent2 := &index.VectorEntry{ID: "r2", Vector: []float32{1, 0, 0, 0}}
	err = col.Insert(ctx, ent2)
	if !errors.Is(err, errRecoveryRequired) {
		t.Errorf("expected errRecoveryRequired after short write, got %v", err)
	}
	if ent2.GraphNodeID != 0 {
		t.Errorf("refused retry after short write received GraphNodeID %d", ent2.GraphNodeID)
	}

	// Reopen reconciles: whatever prefix reached disk is framed-checked
	// during replay. Fresh writes must succeed and resolve uniquely.
	e.Close()
	eng, err := New(path)
	if err != nil {
		t.Fatalf("reopen after short write: %v", err)
	}
	e2 := eng.(*Engine)
	defer e2.Close()
	col2, err := e2.GetCollection("col")
	if err != nil {
		t.Fatalf("get collection after reopen: %v", err)
	}
	ent3 := &index.VectorEntry{ID: "r3", Vector: []float32{1, 0, 0, 0}}
	if err := col2.Insert(ctx, ent3); err != nil {
		t.Fatalf("insert after reopen: %v", err)
	}
	if ent3.GraphNodeID == 0 {
		t.Errorf("post-reopen insert got zero ID")
	}
	colName, recID, rerr := e2.ResolveNodeID(ctx, ent3.GraphNodeID)
	if rerr != nil || colName != "col" || recID != "r3" {
		t.Errorf("ID %d resolves to %s/%s (%v), want col/r3", ent3.GraphNodeID, colName, recID, rerr)
	}
}

// 16. Sync failure + reopen: verify no duplicate ID
func TestAdversarialGraphNodeID_SyncFailureReopen(t *testing.T) {
	path := tempDB(t)
	e, cleanup := openEngine(t, path)
	defer cleanup()

	col, _ := e.CreateCollection("col", &storage.CollectionConfig{Dimension: 4})
	ctx := context.Background()

	// Insert one record successfully
	ent1 := &index.VectorEntry{ID: "r1", Vector: []float32{1, 0, 0, 0}}
	col.Insert(ctx, ent1)
	firstID := ent1.GraphNodeID

	// Insert a second record successfully so firstID is definitely below current max
	entA := &index.VectorEntry{ID: "rA", Vector: []float32{2, 0, 0, 0}}
	col.Insert(ctx, entA)
	if entA.GraphNodeID <= firstID {
		t.Fatalf("second insert should have higher ID")
	}

	// Inject sync failure for the next insert
	installTestFaultHooks(e, nil, func(*os.File) error {
		return os.ErrClosed
	})

	ent2 := &index.VectorEntry{ID: "r2", Vector: []float32{0, 1, 0, 0}}
	err := col.Insert(ctx, ent2)
	if err == nil {
		t.Fatalf("expected sync error, got nil")
	}
	if ent2.GraphNodeID != 0 {
		t.Errorf("failed insert leaked GraphNodeID to caller entry: %d", ent2.GraphNodeID)
	}

	// Remove hooks, close, and reopen
	installTestFaultHooks(e, nil, nil)
	e.Close()

	eng, err := New(path)
	if err != nil {
		t.Fatalf("reopen after sync failure: %v", err)
	}
	e2 := eng.(*Engine)
	defer e2.Close()

	// Verify r1 still resolves correctly
	colName, recID, err := e2.ResolveNodeID(ctx, firstID)
	if err != nil {
		t.Fatalf("resolve firstID after reopen: %v", err)
	}
	if colName != "col" || recID != "r1" {
		t.Errorf("resolved to %s/%s, want col/r1", colName, recID)
	}

	// Insert new records — must NOT produce conflicting GraphNodeIDs
	col2, _ := e2.CreateCollection("col2", &storage.CollectionConfig{Dimension: 4})
	ent3 := &index.VectorEntry{ID: "r3", Vector: []float32{1, 0, 0, 0}}
	col2.Insert(ctx, ent3)
	if ent3.GraphNodeID == 0 {
		t.Fatalf("new insert got zero ID")
	}
	// Resolve both IDs — they must map to distinct records
	_, _, err1 := e2.ResolveNodeID(ctx, firstID)
	_, _, err3 := e2.ResolveNodeID(ctx, ent3.GraphNodeID)
	if err1 != nil {
		t.Errorf("firstID resolution failed: %v", err1)
	}
	if err3 != nil {
		t.Errorf("newID resolution failed: %v", err3)
	}
}

// 17. Collection delete tombstones constituent GraphNodeIDs
func TestAdversarialGraphNodeID_CollectionDeleteTombstone(t *testing.T) {
	path := tempDB(t)
	e, cleanup := openEngine(t, path)
	defer cleanup()

	col, _ := e.CreateCollection("col", &storage.CollectionConfig{Dimension: 4})
	ctx := context.Background()

	ent1 := &index.VectorEntry{ID: "r1", Vector: []float32{1, 0, 0, 0}}
	ent2 := &index.VectorEntry{ID: "r2", Vector: []float32{0, 1, 0, 0}}
	col.Insert(ctx, ent1)
	col.Insert(ctx, ent2)
	id1 := ent1.GraphNodeID
	id2 := ent2.GraphNodeID

	// Live resolution before delete
	_, _, err := e.ResolveNodeID(ctx, id1)
	if err != nil {
		t.Fatalf("resolve live id1: %v", err)
	}

	// Delete the collection
	if err := e.DeleteCollection("col"); err != nil {
		t.Fatalf("delete collection: %v", err)
	}

	// Both IDs should now be tombstoned
	_, _, err = e.ResolveNodeID(ctx, id1)
	if err != storage.ErrTombstonedGraphNodeID {
		t.Errorf("Expected ErrTombstonedGraphNodeID for id1 after collection delete, got %v", err)
	}
	_, _, err = e.ResolveNodeID(ctx, id2)
	if err != storage.ErrTombstonedGraphNodeID {
		t.Errorf("Expected ErrTombstonedGraphNodeID for id2 after collection delete, got %v", err)
	}

	// After reopen, tombstones should persist
	e.Close()
	eng, _ := New(path)
	e2 := eng.(*Engine)
	defer e2.Close()
	_, _, err = e2.ResolveNodeID(ctx, id1)
	if err != storage.ErrTombstonedGraphNodeID {
		t.Errorf("Tombstone lost across restart after collection delete: got %v", err)
	}
}

// 18. Reverse directory pre-admission exhaustion. A small pool must cause
// ErrMemoryLimitExceeded before WAL append — no durable record, no visible
// mapping, no allocator advancement.
func TestAdversarialGraphNodeID_ReverseDirExhaustion(t *testing.T) {
	path := tempDB(t)
	// 1 MiB pool: enough for pool metadata plus thousands of 16-byte
	// entries, but exhausts before the loop bound. This is the supported
	// public admission bound (WithReverseDirectoryLimit).
	eng, err := New(path, WithReverseDirectoryLimit(1<<20))
	if err != nil {
		t.Fatalf("open engine: %v", err)
	}
	e := eng.(*Engine)
	defer func() {
		eng.Close()
		os.Remove(path)
	}()

	col, err := e.CreateCollection("col", &storage.CollectionConfig{Dimension: 4})
	if err != nil {
		t.Fatalf("create col: %v", err)
	}
	ctx := context.Background()

	// Batch inserts amortize fsync: each InsertBatch reserves all of its
	// reverse-directory slots in a single pre-admission call inside one
	// WAL transaction. Exceeding pool capacity must fail the whole batch
	// with ErrMemoryLimitExceeded before any WAL bytes are written.
	inserted := 0
	var exhausted bool
	for round := 0; round < 4000 && !exhausted; round++ {
		batch := make([]*index.VectorEntry, 64)
		for j := range batch {
			id := fmt.Sprintf("r%d", inserted+j)
			batch[j] = &index.VectorEntry{ID: id, Vector: []float32{float32(j % 4), 0, 0, 0}}
		}
		err := col.InsertBatch(ctx, batch)
		if err != nil {
			if errors.Is(err, storage.ErrMemoryLimitExceeded) {
				exhausted = true
				break
			}
			t.Fatalf("unexpected error on batch %d: %v", round, err)
		}
		inserted += len(batch)
	}
	if inserted == 0 {
		t.Fatalf("could not insert even a single record (pool too small for one entry)")
	}
	if !exhausted {
		t.Errorf("expected ErrMemoryLimitExceeded within %d inserts, but all succeeded", inserted)
	} else {
		t.Logf("exhausted after %d inserts with 1 MiB pool", inserted)
	}

	// After exhaustion, the allocator must not have advanced past what
	// was durably committed: a fresh engine with the default pool must
	// reopen and see exactly `inserted` live IDs, none aliased.
	eng.Close()
	eng2, err := New(path)
	if err != nil {
		t.Fatalf("reopen after exhaustion: %v", err)
	}
	e2 := eng2.(*Engine)
	defer e2.Close()
	for i := 0; i < inserted; i++ {
		id := fmt.Sprintf("r%d", i)
		gid, err := e2.GetNodeID(ctx, "col", id)
		if err != nil {
			t.Fatalf("reopen: record %s missing: %v", id, err)
		}
		colName, recID, err := e2.ResolveNodeID(ctx, gid)
		if err != nil || colName != "col" || recID != id {
			t.Errorf("reopen: ID %d resolves to %s/%s (%v), want col/%s", gid, colName, recID, err, id)
		}
	}
}

// 19. Concurrent resolve + close
func TestAdversarialGraphNodeID_ConcurrentResolveClose(t *testing.T) {
	path := tempDB(t)
	e, cleanup := openEngine(t, path)
	defer cleanup()

	col, _ := e.CreateCollection("col", &storage.CollectionConfig{Dimension: 4})
	ctx := context.Background()
	ent := &index.VectorEntry{ID: "r1", Vector: []float32{1, 0, 0, 0}}
	col.Insert(ctx, ent)
	id := ent.GraphNodeID

	var wg sync.WaitGroup
	// Concurrent resolves
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _, _ = e.ResolveNodeID(ctx, id)
		}()
	}
	// Concurrent close
	wg.Add(1)
	go func() {
		defer wg.Done()
		e.Close()
	}()
	// Should not panic or deadlock
	wg.Wait()
}

// 20. Delete + reinsert across reopen preserves new identity
func TestAdversarialGraphNodeID_DeleteReinsertReopen(t *testing.T) {
	path := tempDB(t)
	e, cleanup := openEngine(t, path)
	defer cleanup()

	col, _ := e.CreateCollection("col", &storage.CollectionConfig{Dimension: 4})
	ctx := context.Background()

	// Insert r1
	ent := &index.VectorEntry{ID: "r1", Vector: []float32{1, 0, 0, 0}}
	col.Insert(ctx, ent)
	firstID := ent.GraphNodeID

	// Delete r1
	col.Delete(ctx, "r1")

	// Reinsert r1 — should get a new ID
	ent2 := &index.VectorEntry{ID: "r1", Vector: []float32{0, 1, 0, 0}}
	col.Insert(ctx, ent2)
	secondID := ent2.GraphNodeID
	if secondID == firstID {
		t.Errorf("reinsert reused tombstoned ID %d", firstID)
	}

	// First ID should be tombstoned
	_, _, err := e.ResolveNodeID(ctx, firstID)
	if err != storage.ErrTombstonedGraphNodeID {
		t.Errorf("old ID not tombstoned after reinsert: %v", err)
	}

	// Second ID should resolve live
	_, _, err = e.ResolveNodeID(ctx, secondID)
	if err != nil {
		t.Errorf("new ID does not resolve: %v", err)
	}

	// Close and reopen
	e.Close()
	eng, _ := New(path)
	e2 := eng.(*Engine)
	defer e2.Close()

	// Verify old ID still tombstoned
	_, _, err = e2.ResolveNodeID(ctx, firstID)
	if err != storage.ErrTombstonedGraphNodeID {
		t.Errorf("tombstone lost across reopen after delete+reinsert: %v", err)
	}

	// Verify new ID resolves
	colName, recID, err := e2.ResolveNodeID(ctx, secondID)
	if err != nil {
		t.Errorf("new ID lost across reopen: %v", err)
	}
	if colName != "col" || recID != "r1" {
		t.Errorf("resolved to %s/%s, want col/r1", colName, recID)
	}
}

// 21. Ambiguous sync failure: WAL written but sync fails. On reopen,
// verify the incomplete transaction is discarded and IDs are not duplicated.
func TestAdversarialGraphNodeID_AmbiguousSync(t *testing.T) {
	path := tempDB(t)
	e, cleanup := openEngine(t, path)
	defer cleanup()

	col, _ := e.CreateCollection("col", &storage.CollectionConfig{Dimension: 4})
	ctx := context.Background()

	// Insert r1 — this is the baseline we know is committed
	ent1 := &index.VectorEntry{ID: "r1", Vector: []float32{1, 0, 0, 0}}
	col.Insert(ctx, ent1)
	baselineID := ent1.GraphNodeID

	// Inject sync failure. The next write goes to disk, but fsync fails.
	// The transaction is ambiguous: bytes may have reached the device.
	installTestFaultHooks(e, nil, func(*os.File) error {
		return os.ErrClosed
	})

	ent2 := &index.VectorEntry{ID: "r2", Vector: []float32{0, 1, 0, 0}}
	err := col.Insert(ctx, ent2)
	if err == nil {
		t.Fatalf("expected sync error")
	}
	if ent2.GraphNodeID != 0 {
		t.Errorf("failed insert leaked GraphNodeID %d to caller", ent2.GraphNodeID)
	}

	// Clear hooks and close
	installTestFaultHooks(e, nil, nil)
	e.Close()

	// Reopen — the ambiguous transaction must be reconciled
	eng, err := New(path)
	if err != nil {
		t.Fatalf("reopen after ambiguous sync: %v", err)
	}
	e2 := eng.(*Engine)
	defer e2.Close()

	// Baseline record must still exist with its original ID
	_, _, err = e2.ResolveNodeID(ctx, baselineID)
	if err != nil {
		t.Errorf("baseline ID lost after ambiguous sync: %v", err)
	}

	// Insert a new record — its ID must not alias the ambiguous transaction's
	// candidate ID (which may or may not have been durably written).
	col2, _ := e2.CreateCollection("col2", &storage.CollectionConfig{Dimension: 4})
	ent3 := &index.VectorEntry{ID: "r3", Vector: []float32{1, 0, 0, 0}}
	col2.Insert(ctx, ent3)
	if ent3.GraphNodeID == 0 {
		t.Fatalf("new insert got zero ID")
	}

	// Both IDs must resolve to distinct live records
	colA, recA, errA := e2.ResolveNodeID(ctx, baselineID)
	colB, recB, errB := e2.ResolveNodeID(ctx, ent3.GraphNodeID)
	t.Logf("baselineID=%d -> (%q,%q,%v); newID=%d -> (%q,%q,%v)",
		baselineID, colA, recA, errA, ent3.GraphNodeID, colB, recB, errB)
	if errA != nil || errB != nil {
		t.Errorf("resolution failed: baseline=%v, new=%v", errA, errB)
	}
	// They must not alias
	if colA == colB && recA == recB {
		t.Errorf("ambiguous sync produced aliased IDs: both resolve to %s/%s", colA, recA)
	}
}

// 23. Queued-batch gate race: a writer admitted just before an ambiguous
// sync failure must not be written by a later flush. The ownership hook parks
// A's flush after it detaches A's queue but before it acquires e.mu. B can
// therefore enter the replacement queue through the public Insert path. When
// A resumes and fails sync, the next flush must refuse B at the write-gate
// re-check without appending B to the WAL.
func TestAdversarialGraphNodeID_QueuedBatchGateRace(t *testing.T) {
	path := tempDB(t)

	// Hold the background flusher until A is queued.
	flusherGate := make(chan struct{})
	origStart := startBatchFlusher
	startBatchFlusher = func(e *Engine) {
		go func() {
			<-flusherGate
			e.batchFlusher()
		}()
	}
	defer func() { startBatchFlusher = origStart }()

	eng, err := New(path)
	if err != nil {
		t.Fatalf("open engine: %v", err)
	}
	e := eng.(*Engine)
	defer func() {
		eng.Close()
		os.Remove(path)
	}()

	col, err := e.CreateCollection("col", &storage.CollectionConfig{Dimension: 4})
	if err != nil {
		t.Fatalf("create col: %v", err)
	}
	ctx := context.Background()

	// Park the first flush after it owns A's queue and releases
	// batchBuffer.mu, but before it takes e.mu. This is the reachable race
	// window in which B can be admitted to the replacement queue.
	ownedA := make(chan struct{})
	releaseOwner := make(chan struct{})
	var ownerOnce sync.Once
	e.batchQueueOwnedHook = func() {
		ownerOnce.Do(func() {
			close(ownedA)
			<-releaseOwner
		})
	}

	// Once A resumes, its WAL append succeeds and its sync fails.
	var syncOnce int32
	installTestFaultHooks(e, nil, func(*os.File) error {
		if atomic.CompareAndSwapInt32(&syncOnce, 0, 1) {
			return os.ErrClosed
		}
		return nil
	})

	// Writer A: admitted while the flusher is gated.
	entA := &index.VectorEntry{ID: "a", Vector: []float32{1, 0, 0, 0}}
	errA := make(chan error, 1)
	go func() {
		errA <- col.Insert(ctx, entA)
	}()

	// Wait until A is actually queued in the batch buffer.
	waitFor(t, func() bool {
		e.batchBuffer.mu.Lock()
		defer e.batchBuffer.mu.Unlock()
		return len(e.batchBuffer.entries) > 0
	}, "writer A queued")

	// Release the flusher. It detaches A and parks at the ownership seam,
	// before taking e.mu or writing WAL.
	close(flusherGate)
	<-ownedA

	// Writer B passes the still-open admission gate and lands in the
	// replacement buffer while A's flush is parked before e.mu.
	entB := &index.VectorEntry{ID: "b", Vector: []float32{0, 1, 0, 0}}
	errB := make(chan error, 1)
	go func() {
		errB <- col.Insert(ctx, entB)
	}()
	waitFor(t, func() bool {
		e.batchBuffer.mu.Lock()
		defer e.batchBuffer.mu.Unlock()
		for i := range e.batchBuffer.entries {
			if e.batchBuffer.entries[i].entry == entB {
				return true
			}
		}
		return false
	}, "writer B queued in replacement buffer")

	// Resume A. Its sync fails and writesDisabled engages while B remains
	// queued. The flusher's next pass owns B and rejects it before WAL append.
	close(releaseOwner)

	// A must observe the sync error (its own ambiguous flush).
	if err := <-errA; err == nil {
		t.Fatalf("writer A: expected sync error, got nil")
	}

	// B must be refused with errRecoveryRequired by the flushBatch
	// re-check: queued work must not survive the gate transition.
	if err := <-errB; !errors.Is(err, errRecoveryRequired) {
		t.Fatalf("writer B: expected errRecoveryRequired, got %v", err)
	}
	if entB.GraphNodeID != 0 {
		t.Errorf("writer B entry received GraphNodeID %d after gate", entB.GraphNodeID)
	}

	// Prove B produced no WAL record: after close+reopen, B must be absent
	// while A (ambiguous but complete-framed) may or may not be present.
	installTestFaultHooks(e, nil, nil)
	e.Close()
	eng2, err := New(path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	e2 := eng2.(*Engine)
	defer e2.Close()

	col2, err := e2.GetCollection("col")
	if err != nil {
		t.Fatalf("get col after reopen: %v", err)
	}
	if exists, err := col2.Exists(ctx, "b"); err != nil || exists {
		t.Errorf("writer B record must be absent after reopen (exists=%v, err=%v)", exists, err)
	}

	// Every live record's ID resolves uniquely — no alias from the gate race.
	seen := make(map[uint64]string)
	iterErr := col2.Iterate(ctx, func(entry *index.VectorEntry) error {
		if entry.GraphNodeID == 0 {
			return fmt.Errorf("record %s has zero GraphNodeID", entry.ID)
		}
		if prev, ok := seen[entry.GraphNodeID]; ok {
			return fmt.Errorf("GraphNodeID %d aliases %s and %s", entry.GraphNodeID, prev, entry.ID)
		}
		seen[entry.GraphNodeID] = entry.ID
		colName, recID, rerr := e2.ResolveNodeID(ctx, entry.GraphNodeID)
		if rerr != nil {
			return fmt.Errorf("resolve %d: %w", entry.GraphNodeID, rerr)
		}
		if colName != "col" || recID != entry.ID {
			return fmt.Errorf("ID %d resolves to %s/%s, want col/%s", entry.GraphNodeID, colName, recID, entry.ID)
		}
		return nil
	})
	if iterErr != nil {
		t.Errorf("iterate after reopen: %v", iterErr)
	}
}

// waitFor polls cond until true or fails the test after a bounded number
// of attempts. Deterministic barrier for admission-before-flush interleavings.
func waitFor(t *testing.T, cond func() bool, what string) {
	t.Helper()
	for i := 0; i < 20000; i++ {
		if cond() {
			return
		}
		time.Sleep(50 * time.Microsecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}
