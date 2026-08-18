package libravdb

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/xDarkicex/libravdb/internal/graph"
	"github.com/xDarkicex/libravdb/internal/optimizer"
	"github.com/xDarkicex/libravdb/internal/storage"
)

var (
	ErrEpochClosed           = errors.New("epoch transaction is closed")
	ErrEpochSnapshotMismatch = errors.New("epoch snapshot mismatch: AS OF incompatible with pinned epoch snapshot")
	ErrSavepointExists       = errors.New("savepoint already exists")
	ErrSavepointNotFound     = errors.New("savepoint not found")
	ErrSavepointNotTop       = errors.New("savepoint is not the top-most savepoint")
	ErrSavepointOutsideEpoch = errors.New("savepoint is only valid inside an epoch transaction")
)

type epochContextKey struct{}

// EpochTx is the transaction context for an agent scratchpad. Record/vector
// mutations and graph mutations are staged until Commit. Rollback discards
// every staged mutation without appending WAL frames.
//
// Read-your-writes semantics: graph reads within the epoch are served from
// a snapshot pinned at epochLSN (the committed state visible at BeginEpochTx
// time). Staged edge operations (adds/removes) are merged on top. Concurrent
// commits from other sessions at higher LSNs are invisible.
//
// The epoch owns a TemporalSnapshot handle from begin until Commit/Rollback,
// preventing compaction from evicting the base state while the epoch is active.
type EpochTx struct {
	db       *Database
	record   Tx
	graphs   map[string]*graph.Txn
	epochLSN uint64 // snapshot LSN captured at BeginEpochTx; 0 = live reads
	mu       sync.Mutex
	closed   bool

	// snapshot is the pinned temporal snapshot handle. It is held from
	// BeginEpochTx/BeginEpochTxAt until Commit/Rollback to prevent
	// retention/compaction from invalidating the epoch's base state.
	snapshot *TemporalSnapshot

	// Provisional node IDs for staged records, keyed by (collection, recordID).
	// Staged inserts are not yet committed to storage, so they have no durable
	// node ID mapping. The epoch assigns temporary node IDs so staged records
	// can be referenced by INSERT INTO GRAPH_EDGES within the same epoch.
	provisionalNodes map[string]uint64 // "collection\x00recordID" → provisional nodeID
	nextProvisional  uint64            // counter, starts high to avoid live collisions

	// Savepoint stack for agent hypothesis branching. Each savepoint records
	// the position in the append-only mutation logs so rollback can truncate.
	savepoints []epochSavepoint

	// generation increments on every mutation, invalidating epoch-local caches.
	generation uint64
}

// epochSavepoint marks a position in the epoch's mutation logs.
type epochSavepoint struct {
	name             string
	recordLogLen     int               // len(recordTx.ops) at savepoint
	graphDropLen     int               // len(recordTx.graphDrops) at savepoint
	graphOpLen       map[string]int    // collection → len(txn.OrderedStagedOps()) at savepoint
	provisionalNodes map[string]uint64 // snapshot of provisionalNodes
	nextProvisional  uint64            // snapshot of nextProvisional
	generation       uint64
}

// SnapshotLSN returns the pinned snapshot LSN for this epoch. Zero means
// no snapshot was captured (e.g. storage does not support temporal resolution).
func (e *EpochTx) SnapshotLSN() uint64 { return e.epochLSN }

// releaseSnapshot releases the pinned temporal snapshot exactly once.
// Safe to call multiple times — subsequent calls are no-ops.
func (e *EpochTx) releaseSnapshot() {
	if e.snapshot != nil {
		e.snapshot.Close()
		e.snapshot = nil
	}
}

// BeginEpochTx starts an isolated agent scratchpad transaction pinned at the
// current committed state. Delegates to BeginEpochTxAt with time.Now().UTC().
func (db *Database) BeginEpochTx(ctx context.Context) (*EpochTx, error) {
	return db.BeginEpochTxAt(ctx, time.Now().UTC())
}

// BeginEpochTxAt starts an isolated agent scratchpad transaction pinned at the
// historical committed state as of the given timestamp. The epoch owns a
// TemporalSnapshot handle preventing compaction from evicting the base state
// while the epoch is active. If the requested timestamp predates retained
// history, ErrRetentionExpired is returned.
func (db *Database) BeginEpochTxAt(ctx context.Context, t time.Time) (*EpochTx, error) {
	tx, err := db.BeginTx(ctx)
	if err != nil {
		return nil, err
	}
	snap, err := db.SnapshotAt(ctx, t)
	if err != nil {
		tx.Rollback(ctx)
		return nil, fmt.Errorf("resolve snapshot at %v: %w", t, err)
	}
	return &EpochTx{
		db:               db,
		record:           tx,
		graphs:           make(map[string]*graph.Txn),
		epochLSN:         snap.LSN,
		snapshot:         snap,
		provisionalNodes: make(map[string]uint64),
		nextProvisional:  1 << 62, // high range to avoid collisions with live node IDs
	}, nil
}

// Context returns a child context carrying this epoch. SQL execution methods
// use it to select staged overlays without changing the normal query API.

// Savepoint creates a named savepoint at the current mutation state.
// Duplicate names within the same epoch are rejected.
func (e *EpochTx) Savepoint(name string) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return ErrEpochClosed
	}
	for _, sp := range e.savepoints {
		if sp.name == name {
			return fmt.Errorf("%w: %q", ErrSavepointExists, name)
		}
	}
	recordTx, ok := e.record.(*transaction)
	if !ok {
		return fmt.Errorf("record transaction is not a *transaction")
	}
	recordTx.mu.Lock()
	recordLogLen := len(recordTx.ops)
	graphDropLen := len(recordTx.graphDrops)
	recordTx.mu.Unlock()
	sp := epochSavepoint{
		name:             name,
		recordLogLen:     recordLogLen,
		graphDropLen:     graphDropLen,
		graphOpLen:       make(map[string]int, len(e.graphs)),
		provisionalNodes: make(map[string]uint64, len(e.provisionalNodes)),
		nextProvisional:  e.nextProvisional,
		generation:       e.generation,
	}
	for colName, gtx := range e.graphs {
		sp.graphOpLen[colName] = len(gtx.OrderedStagedOps())
	}
	for k, v := range e.provisionalNodes {
		sp.provisionalNodes[k] = v
	}
	e.savepoints = append(e.savepoints, sp)
	return nil
}

// RollbackTo restores the epoch state to the named savepoint.
// Younger savepoints are discarded. The savepoint remains active.
func (e *EpochTx) RollbackTo(name string) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return ErrEpochClosed
	}
	idx := -1
	for i, sp := range e.savepoints {
		if sp.name == name {
			idx = i
			break
		}
	}
	if idx < 0 {
		return fmt.Errorf("%w: %q", ErrSavepointNotFound, name)
	}
	sp := e.savepoints[idx]

	recordTx, ok := e.record.(*transaction)
	if !ok {
		return fmt.Errorf("record transaction is not a *transaction")
	}

	// Truncate record mutation log.
	recordTx.mu.Lock()
	if len(recordTx.ops) > sp.recordLogLen {
		recordTx.ops = recordTx.ops[:sp.recordLogLen]
	}
	if len(recordTx.graphDrops) > sp.graphDropLen {
		recordTx.graphDrops = recordTx.graphDrops[:sp.graphDropLen]
	}
	recordTx.mu.Unlock()

	// Rebuild graph transactions: replay surviving ordered operations
	// in original append order. Never replay grouped by kind — edge
	// add/remove sequences are semantically observable.
	for colName, gtx := range e.graphs {
		orderedOps := gtx.OrderedStagedOps()
		survivingLen := sp.graphOpLen[colName]
		if survivingLen > len(orderedOps) {
			survivingLen = len(orderedOps)
		}

		// If no operations survive and no graph transaction existed at
		// the savepoint, remove the graph transaction entirely.
		if survivingLen == 0 && sp.graphOpLen[colName] == 0 {
			delete(e.graphs, colName)
			continue
		}

		surviving := orderedOps[:survivingLen]

		col, cerr := e.db.GetCollection(colName)
		if cerr != nil {
			return cerr
		}
		g := col.GetGraph()
		if g == nil {
			continue
		}

		fresh := g.BeginTxn()
		fresh.SetEpochLSN(e.epochLSN)
		fresh.SetCollection(colName)

		for _, op := range surviving {
			switch op.Kind {
			case graph.StagedGraphEdgeAdd:
				if err := fresh.AddEdgeWithPropertiesJSON(op.Src, op.Tgt, op.Weight, op.EdgeKind, op.Properties); err != nil {
					return err
				}
			case graph.StagedGraphEdgeRemove:
				if err := fresh.RemoveEdge(op.Src, op.Tgt, op.EdgeKind); err != nil {
					return err
				}
			case graph.StagedGraphNodeDrop:
				if err := fresh.DropNodeEdges(op.NodeID); err != nil {
					return err
				}
			}
		}

		e.graphs[colName] = fresh
	}

	// Restore provisional nodes.
	// Clone to avoid aliasing the savepoint's map.
	e.provisionalNodes = make(map[string]uint64, len(sp.provisionalNodes))
	for k, v := range sp.provisionalNodes {
		e.provisionalNodes[k] = v
	}
	e.nextProvisional = sp.nextProvisional
	e.generation++ // invalidates caches

	// Discard younger savepoints.
	e.savepoints = e.savepoints[:idx+1]
	return nil
}

// ReleaseSavepoint removes the named savepoint without changing any mutations.
// Only the top-most savepoint may be released.
func (e *EpochTx) ReleaseSavepoint(name string) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return ErrEpochClosed
	}
	if len(e.savepoints) == 0 {
		return fmt.Errorf("%w: %q", ErrSavepointNotFound, name)
	}
	top := e.savepoints[len(e.savepoints)-1]
	if top.name != name {
		return fmt.Errorf("%w: %q (top is %q)", ErrSavepointNotTop, name, top.name)
	}
	e.savepoints = e.savepoints[:len(e.savepoints)-1]
	return nil
}

// numSavepoints returns the number of active savepoints.
func (e *EpochTx) numSavepoints() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.savepoints)
}

func (e *EpochTx) Context(ctx context.Context) context.Context {
	return context.WithValue(ctx, epochContextKey{}, e)
}

func epochFromContext(ctx context.Context) *EpochTx {
	if ctx == nil {
		return nil
	}
	e, _ := ctx.Value(epochContextKey{}).(*EpochTx)
	return e
}

// RecordTx returns the staged record/vector transaction.
func (e *EpochTx) RecordTx() (Tx, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, ErrEpochClosed
	}
	return e.record, nil
}

// Insert stages a vector record in the epoch.
func (e *EpochTx) Insert(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}) error {
	tx, err := e.RecordTx()
	if err != nil {
		return err
	}
	epochCtx := e.Context(ctx)
	if err := tx.Insert(epochCtx, collection, id, vector, metadata); err != nil {
		return err
	}
	// Assign a provisional node ID only after staging succeeds so a failed
	// insert cannot leave an endpoint visible in GRAPH_NODES or graph lookup.
	e.mu.Lock()
	key := collection + "\x00" + id
	if _, exists := e.provisionalNodes[key]; !exists {
		e.provisionalNodes[key] = e.nextProvisional
		e.nextProvisional++
	}
	e.generation++
	e.mu.Unlock()
	return nil
}

// LookupNodeID resolves a record ID to a graph node ID within the epoch.
// Staged inserts get provisional IDs; committed records resolve via storage.
func (e *EpochTx) LookupNodeID(ctx context.Context, collection, id string) (uint64, error) {
	e.mu.Lock()
	if nid, ok := e.provisionalNodes[collection+"\x00"+id]; ok {
		e.mu.Unlock()
		visible, err := e.recordVisible(ctx, collection, id)
		if err != nil {
			return 0, err
		}
		if !visible {
			return 0, fmt.Errorf("%w: %s", ErrRecordNotFound, id)
		}
		return nid, nil
	}
	e.mu.Unlock()
	visible, err := e.recordVisible(ctx, collection, id)
	if err != nil {
		return 0, err
	}
	if !visible {
		return 0, fmt.Errorf("%w: %s", ErrRecordNotFound, id)
	}
	return e.db.GetNodeID(ctx, collection, id)
}

// ResolveNodeID maps a graph node ID back to a (collection, recordID) pair.
// Provisional IDs are resolved from the epoch's local mapping; committed IDs
// are resolved via the database's live reverse directory.
func (e *EpochTx) ResolveNodeID(ctx context.Context, nodeID uint64) (string, string, error) {
	e.mu.Lock()
	for compositeKey, nid := range e.provisionalNodes {
		if nid == nodeID {
			e.mu.Unlock()
			// compositeKey is "collection\x00recordID"
			parts := splitCompositeKey(compositeKey)
			if len(parts) == 2 {
				visible, err := e.recordVisible(ctx, parts[0], parts[1])
				if err != nil {
					return "", "", err
				}
				if !visible {
					return "", "", fmt.Errorf("%w: %d", ErrRecordNotFound, nodeID)
				}
				return parts[0], parts[1], nil
			}
			return "", compositeKey, nil
		}
	}
	e.mu.Unlock()
	collection, id, err := e.db.ResolveNodeID(ctx, nodeID)
	if err != nil {
		return "", "", err
	}
	visible, err := e.recordVisible(ctx, collection, id)
	if err != nil {
		return "", "", err
	}
	if !visible {
		return "", "", fmt.Errorf("%w: %d", ErrRecordNotFound, nodeID)
	}
	return collection, id, nil
}

// Update stages a vector record update in the epoch.
func (e *EpochTx) Update(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}) error {
	tx, err := e.RecordTx()
	if err != nil {
		return err
	}
	epochCtx := e.Context(ctx)
	if err := tx.Update(epochCtx, collection, id, vector, metadata); err != nil {
		return err
	}
	e.mu.Lock()
	e.generation++
	e.mu.Unlock()
	return nil
}

// Upsert stages an insert-or-replace mutation in the epoch overlay. SQL
// ON CONFLICT execution uses this only after it has resolved a non-conflicting
// proposed row; conflicting DO UPDATE paths use Update so unspecified columns
// retain their existing values.
func (e *EpochTx) Upsert(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}) error {
	tx, err := e.RecordTx()
	if err != nil {
		return err
	}
	_, durableNodeErr := e.db.GetNodeID(ctx, collection, id)
	epochCtx := e.Context(ctx)
	if err := tx.Upsert(epochCtx, collection, id, vector, metadata); err != nil {
		return err
	}
	e.mu.Lock()
	key := collection + "\x00" + id
	if durableNodeErr != nil {
		if _, exists := e.provisionalNodes[key]; !exists {
			e.provisionalNodes[key] = e.nextProvisional
			e.nextProvisional++
		}
	}
	e.generation++
	e.mu.Unlock()
	return nil
}

// Rename stages a primary-key migration inside the epoch overlay.
func (e *EpochTx) Rename(ctx context.Context, collection, oldID, newID string, vector []float32, metadata map[string]interface{}) error {
	tx, err := e.RecordTx()
	if err != nil {
		return err
	}
	epochCtx := e.Context(ctx)
	if err := tx.Rename(epochCtx, collection, oldID, newID, vector, metadata); err != nil {
		return err
	}
	e.mu.Lock()
	oldKey := collection + "\x00" + oldID
	newKey := collection + "\x00" + newID
	if provisional, ok := e.provisionalNodes[oldKey]; ok {
		delete(e.provisionalNodes, oldKey)
		e.provisionalNodes[newKey] = provisional
	}
	e.mu.Unlock()
	e.generation++
	return nil
}

// Delete stages a vector record deletion in the epoch.
func (e *EpochTx) Delete(ctx context.Context, collection, id string) error {
	tx, err := e.RecordTx()
	if err != nil {
		return err
	}
	// A provisional node has no live reverse-directory entry, so its staged
	// graph edges must be dropped in the epoch overlay explicitly. Committed
	// nodes are handled by transaction.delete's durable graph-drop operation.
	e.mu.Lock()
	key := collection + "\x00" + id
	provisionalID, provisional := e.provisionalNodes[key]
	e.mu.Unlock()
	epochCtx := e.Context(ctx)
	if err := tx.Delete(epochCtx, collection, id); err != nil {
		return err
	}
	if provisional {
		if err := e.DropGraphNodeEdges(collection, provisionalID); err != nil {
			return err
		}
		e.mu.Lock()
		delete(e.provisionalNodes, key)
		e.mu.Unlock()
	}
	e.mu.Lock()
	e.generation++
	e.mu.Unlock()
	return nil
}

// recordVisible reports whether a record is visible in the current epoch
// branch without materializing the entire collection. It applies the pinned
// snapshot first, then the ordered record mutations with last-write-wins
// semantics. This is used to prevent stale GRAPH_NODES mappings from being
// returned after an epoch-local delete or snapshot exclusion.
func (e *EpochTx) recordVisible(ctx context.Context, collection, id string) (bool, error) {
	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return false, ErrEpochClosed
	}
	recordTx, ok := e.record.(*transaction)
	snapshotLSN := e.epochLSN
	e.mu.Unlock()

	col, err := e.db.GetCollection(collection)
	if err != nil {
		return false, err
	}
	visible := false
	if snapshotLSN > 0 {
		rec, err := col.GetAtLSN(ctx, id, snapshotLSN)
		if err != nil {
			return false, err
		}
		visible = rec != nil
	} else {
		_, err := col.Get(ctx, id)
		visible = err == nil
		if err != nil && !errors.Is(err, ErrRecordNotFound) {
			return false, err
		}
	}
	if !ok {
		return visible, nil
	}
	recordTx.mu.Lock()
	ops := append([]txMutation(nil), recordTx.ops...)
	recordTx.mu.Unlock()
	for _, op := range ops {
		if op.collection != collection {
			continue
		}
		switch op.kind {
		case txMutationDelete:
			if op.id == id {
				visible = false
			}
		case txMutationInsert, txMutationUpsert:
			if op.id == id {
				visible = true
			}
		case txMutationUpdate:
			// An update cannot create a record; retain the current state.
		case txMutationRename:
			if op.oldID == id {
				visible = false
			}
			if op.id == id {
				visible = true
			}
		}
	}
	return visible, nil
}

// Query executes a SQL statement against this epoch context. Graph traversal
// and multimodal candidate generation can therefore observe staged edges.
func (e *EpochTx) Query(ctx context.Context, sql string, params QueryParams) (*SearchResults, error) {
	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return nil, ErrEpochClosed
	}
	e.mu.Unlock()
	return e.db.queryWithContext(e.Context(ctx), sql, params)
}

// QueryWithSessionConfig executes SQL against the epoch snapshot with
// connection-local settings applied to query-local execution.
func (e *EpochTx) QueryWithSessionConfig(ctx context.Context, sql string, params QueryParams, config *SessionConfig) (*SearchResults, error) {
	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return nil, ErrEpochClosed
	}
	e.mu.Unlock()
	return e.db.queryWithBoundParamsAndConfig(e.Context(ctx), sql, optimizer.NewParameterSet(params), params, config)
}

// QueryWithBoundParams executes SQL with values already decoded into the
// native typed parameter set. It is used by pgwire extended execution so no
// SQL text substitution or map/string normalization is required.
func (e *EpochTx) QueryWithBoundParams(ctx context.Context, sql string, params *optimizer.ParameterSet) (*SearchResults, error) {
	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return nil, ErrEpochClosed
	}
	e.mu.Unlock()
	return e.db.queryWithBoundParams(e.Context(ctx), sql, params, nil)
}

// QueryWithBoundParamsAndSessionConfig preserves epoch visibility while
// applying the caller's connection-local execution settings.
func (e *EpochTx) QueryWithBoundParamsAndSessionConfig(ctx context.Context, sql string, params *optimizer.ParameterSet, config *SessionConfig) (*SearchResults, error) {
	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return nil, ErrEpochClosed
	}
	e.mu.Unlock()
	return e.db.queryWithBoundParamsAndConfig(e.Context(ctx), sql, params, nil, config)
}

// ListRecords returns the collection view with staged record mutations
// overlaid on the committed state visible at the epoch's pinned snapshot LSN.
// When epochLSN > 0, the base view uses ListVisibleAtLSN instead of live ListAll
// so that records committed after the epoch began are excluded.
func (e *EpochTx) ListRecords(ctx context.Context, collection string) ([]Record, error) {
	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return nil, ErrEpochClosed
	}
	recordTx, ok := e.record.(*transaction)
	snapshotLSN := e.epochLSN
	e.mu.Unlock()

	col, err := e.db.GetCollection(collection)
	if err != nil {
		return nil, err
	}

	// Base view: use snapshot (ListVisibleAtLSN) when pinned, live ListAll otherwise.
	var base []Record
	if snapshotLSN > 0 {
		err = col.ListVisibleAtLSN(ctx, snapshotLSN, func(rec *Record) bool {
			base = append(base, *rec)
			return true
		})
	} else {
		base, err = col.ListAll(ctx)
	}
	if err != nil {
		return nil, err
	}
	if !ok {
		return base, nil
	}
	recordTx.mu.Lock()
	ops := append([]txMutation(nil), recordTx.ops...)
	recordTx.mu.Unlock()
	byID := make(map[string]Record, len(base)+len(ops))
	order := make([]string, 0, len(base)+len(ops))
	for _, rec := range base {
		byID[rec.ID] = rec
		order = append(order, rec.ID)
	}
	for _, op := range ops {
		if op.collection != collection {
			continue
		}
		switch op.kind {
		case txMutationDelete:
			delete(byID, op.id)
		case txMutationInsert, txMutationUpdate, txMutationUpsert:
			if _, exists := byID[op.id]; !exists {
				order = append(order, op.id)
			}
			byID[op.id] = Record{ID: op.id, Vector: cloneVector(op.vector), Metadata: cloneMetadata(op.metadata)}
		case txMutationRename:
			delete(byID, op.oldID)
			if _, exists := byID[op.id]; !exists {
				order = append(order, op.id)
			}
			byID[op.id] = Record{ID: op.id, Vector: cloneVector(op.vector), Metadata: cloneMetadata(op.metadata)}
		}
	}
	out := make([]Record, 0, len(byID))
	for _, id := range order {
		if rec, exists := byID[id]; exists {
			out = append(out, rec)
		}
	}
	return out, nil
}

// GraphTxn returns the staged graph transaction for a collection.
func (e *EpochTx) GraphTxn(collection string) (*graph.Txn, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, ErrEpochClosed
	}
	if tx := e.graphs[collection]; tx != nil {
		return tx, nil
	}
	col, err := e.db.GetCollection(collection)
	if err != nil {
		return nil, err
	}
	g := col.GetGraph()
	if g == nil {
		return nil, errors.New("collection has no graph")
	}
	tx := g.BeginTxn()
	tx.SetEpochLSN(e.epochLSN) // pin read snapshot
	tx.SetCollection(collection)
	e.graphs[collection] = tx
	return tx, nil
}

// AddGraphEdge stages a directed edge add within the epoch. Increments generation.
func (e *EpochTx) AddGraphEdge(collection string, src, tgt uint64, weight float32, kind uint8) error {
	return e.AddGraphEdgeWithPropertiesJSON(collection, src, tgt, weight, kind, nil)
}

// GraphEdgeMutation describes one desired outgoing relationship using stable
// record IDs rather than internal graph node IDs.
type GraphEdgeMutation struct {
	TargetID string
	EdgeType string
	Weight   float32
}

// AddGraphEdgeByID stages a graph mutation using application record IDs. It
// resolves the IDs inside the epoch so staged inserts and ordinary committed
// records participate in the same atomic graph/record commit.
func (e *EpochTx) AddGraphEdgeByID(ctx context.Context, collection, sourceID, targetID, edgeType string, weight float32) error {
	kind := ResolveEdgeKind(edgeType)
	if kind == 0 {
		return fmt.Errorf("unknown edge kind %q", edgeType)
	}
	source, err := e.LookupNodeID(ctx, collection, sourceID)
	if err != nil {
		return err
	}
	target, err := e.LookupNodeID(ctx, collection, targetID)
	if err != nil {
		return err
	}
	return e.AddGraphEdge(collection, source, target, weight, kind)
}

// AddGraphEdgeWithPropertiesJSON stages an edge property envelope through the
// same ordered epoch operation log as ordinary graph edges.
func (e *EpochTx) AddGraphEdgeWithPropertiesJSON(collection string, src, tgt uint64, weight float32, kind uint8, properties []byte) error {
	gtx, err := e.GraphTxn(collection)
	if err != nil {
		return err
	}
	if err := gtx.AddEdgeWithPropertiesJSON(src, tgt, weight, kind, properties); err != nil {
		return err
	}
	e.generation++
	return nil
}

// RemoveGraphEdge stages a directed edge remove within the epoch. Increments generation.
func (e *EpochTx) RemoveGraphEdge(collection string, src, tgt uint64, kind uint8) error {
	gtx, err := e.GraphTxn(collection)
	if err != nil {
		return err
	}
	if err := gtx.RemoveEdge(src, tgt, kind); err != nil {
		return err
	}
	e.generation++
	return nil
}

// RemoveGraphEdgeByID stages an edge removal using stable record IDs.
func (e *EpochTx) RemoveGraphEdgeByID(ctx context.Context, collection, sourceID, targetID, edgeType string) error {
	kind := ResolveEdgeKind(edgeType)
	if kind == 0 {
		return fmt.Errorf("unknown edge kind %q", edgeType)
	}
	source, err := e.LookupNodeID(ctx, collection, sourceID)
	if err != nil {
		return err
	}
	target, err := e.LookupNodeID(ctx, collection, targetID)
	if err != nil {
		return err
	}
	return e.RemoveGraphEdge(collection, source, target, kind)
}

// ReplaceGraphEdgesByID atomically reconciles one source's outgoing edges of
// one type. Existing edges of that type are removed and the desired set is
// staged in their place; unrelated edge types remain untouched.
func (e *EpochTx) ReplaceGraphEdgesByID(ctx context.Context, collection, sourceID, edgeType string, desired []GraphEdgeMutation) error {
	kind := ResolveEdgeKind(edgeType)
	if kind == 0 {
		return fmt.Errorf("unknown edge kind %q", edgeType)
	}
	source, err := e.LookupNodeID(ctx, collection, sourceID)
	if err != nil {
		return err
	}
	gtx, err := e.GraphTxn(collection)
	if err != nil {
		return err
	}
	col, err := e.db.GetCollection(collection)
	if err != nil {
		return err
	}
	var existing []Edge
	if e.epochLSN != 0 {
		existing, err = col.GetGraph().NeighborsAtLSN(source, e.epochLSN)
	} else {
		existing, err = col.GetGraph().Neighbors(source)
	}
	if err != nil {
		return err
	}
	for _, edge := range existing {
		if edge.GetKind() == kind {
			if err := gtx.RemoveEdge(source, edge.Target, kind); err != nil {
				return err
			}
		}
	}
	for _, mutation := range desired {
		if mutation.EdgeType != "" && !strings.EqualFold(mutation.EdgeType, edgeType) {
			return fmt.Errorf("replacement edge type %q does not match %q", mutation.EdgeType, edgeType)
		}
		target, lookupErr := e.LookupNodeID(ctx, collection, mutation.TargetID)
		if lookupErr != nil {
			return lookupErr
		}
		if err := gtx.AddEdge(source, target, mutation.Weight, kind); err != nil {
			return err
		}
	}
	e.mu.Lock()
	e.generation++
	e.mu.Unlock()
	return nil
}

// DropGraphNodeEdges stages a node-edge drop within the epoch. Increments generation.
func (e *EpochTx) DropGraphNodeEdges(collection string, nodeID uint64) error {
	gtx, err := e.GraphTxn(collection)
	if err != nil {
		return err
	}
	if err := gtx.DropNodeEdges(nodeID); err != nil {
		return err
	}
	e.generation++
	return nil
}

// Rollback discards all staged records, vectors, and graph operations.
// The pinned snapshot is released.
func (e *EpochTx) Rollback(ctx context.Context) error {
	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return ErrEpochClosed
	}
	e.closed = true
	e.releaseSnapshot()
	graphs := make([]*graph.Txn, 0, len(e.graphs))
	for _, tx := range e.graphs {
		graphs = append(graphs, tx)
	}
	record := e.record
	e.mu.Unlock()
	for _, tx := range graphs {
		if err := tx.Rollback(); err != nil {
			return err
		}
	}
	return record.Rollback(ctx)
}

// Commit publishes staged records/vectors and graph mutations as one atomic
// WAL transaction. GraphNodeIDs are pre-assigned before frame construction,
// provisional edge endpoints are remapped to committed IDs, and everything
// is written through a single engine.CommitTx call.
func (e *EpochTx) Commit(ctx context.Context) error {
	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return ErrEpochClosed
	}
	e.closed = true
	graphs := make([]*graph.Txn, 0, len(e.graphs))
	for _, tx := range e.graphs {
		graphs = append(graphs, tx)
	}
	record := e.record
	recordTx, ok := record.(*transaction)
	if !ok {
		e.mu.Unlock()
		return fmt.Errorf("epoch record transaction is not a *transaction")
	}
	// Snapshot provisional nodes ("collection\x00recordID" → provisional nodeID).
	provisionalCopy := make(map[string]uint64, len(e.provisionalNodes))
	for k, v := range e.provisionalNodes {
		provisionalCopy[k] = v
	}
	recordTx.mu.Lock()
	recordGraphDrops := append([]txGraphDrop(nil), recordTx.graphDrops...)
	recordTx.mu.Unlock()
	e.mu.Unlock()

	reserveGraphOps := func(reservedIDs map[string]uint64) ([]storage.TxOperation, func(uint64) error) {
		// reservedIDs: "collection\x00recordID" → reserved GraphNodeID (pre-assigned by commitTxWithGraph).
		// Remap provisional edge endpoints to these committed IDs.
		if len(reservedIDs) > 0 && len(provisionalCopy) > 0 {
			// Build provisionalID → committedID mapping.
			remap := make(map[uint64]uint64)
			// provisionalCopy maps compositeKey → provisionalID.
			// reservedIDs maps compositeKey → committedID.
			for compositeKey, committedID := range reservedIDs {
				if provisionalID, ok := provisionalCopy[compositeKey]; ok {
					remap[provisionalID] = committedID
				}
			}
			for _, gtx := range graphs {
				gtx.RemapNodeIDs(remap)
			}
		}
		graphOps := e.buildGraphOps(graphs)
		for _, drop := range recordGraphDrops {
			graphOps = append(graphOps, storage.TxOperation{
				Type: storage.TxOperationGraphNodeDrop, Collection: drop.collection,
				EdgeSrc: drop.nodeID,
			})
		}
		return graphOps, func(commitLSN uint64) error {
			for _, gtx := range graphs {
				var err error
				if commitLSN != 0 {
					err = gtx.ApplyInMemoryAtLSN(commitLSN)
				} else {
					err = gtx.ApplyInMemory()
				}
				if err != nil {
					return err
				}
			}
			return e.db.applyGraphNodeDrops(recordGraphDrops, commitLSN)
		}
	}

	return e.db.commitTxWithGraph(ctx, recordTx.ops, reserveGraphOps, nil, nil)
}

// buildGraphOps converts staged graph Txns to TxOperations with collection identity.
func (e *EpochTx) buildGraphOps(graphs []*graph.Txn) []storage.TxOperation {
	var ops []storage.TxOperation
	for name, gtx := range e.graphs {
		adds, removes, drops := gtx.StagedOps()
		for _, add := range adds {
			ops = append(ops, storage.TxOperation{
				Type: storage.TxOperationGraphEdgeAdd, Collection: name,
				EdgeSrc: add.Src, EdgeTgt: add.Tgt, EdgeWeight: add.Weight, EdgeKind: add.Kind, EdgeProperties: append([]byte(nil), add.Properties...),
			})
		}
		for _, remove := range removes {
			ops = append(ops, storage.TxOperation{
				Type: storage.TxOperationGraphEdgeRemove, Collection: name,
				EdgeSrc: remove.Src, EdgeTgt: remove.Tgt, EdgeKind: remove.Kind,
			})
		}
		for _, drop := range drops {
			ops = append(ops, storage.TxOperation{
				Type: storage.TxOperationGraphNodeDrop, Collection: name,
				EdgeSrc: drop.NodeID,
			})
		}
	}
	return ops
}

// splitCompositeKey splits a "collection\x00recordID" key into its parts.
func splitCompositeKey(key string) []string {
	for i := 0; i < len(key); i++ {
		if key[i] == 0 {
			return []string{key[:i], key[i+1:]}
		}
	}
	return []string{key}
}

// buildEpochGraphOps converts staged graph operations to TxOperations.
