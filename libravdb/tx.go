package libravdb

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"sync"
	"time"
	"unsafe"

	"github.com/xDarkicex/libravdb/internal/catalog"
	"github.com/xDarkicex/libravdb/internal/graph"
	"github.com/xDarkicex/libravdb/internal/index"
	"github.com/xDarkicex/libravdb/internal/storage"
	"github.com/xDarkicex/memory"
)

// ctxKeySkipFKValidation is a context key that signals FK validation should be
// skipped. Used during cascade SET NULL / SET DEFAULT so the child row's FK
// check does not fail on a parent row that is being deleted in the same
// transaction.
type ctxKeySkipFKValidation struct{}

// transactionContextKey carries the private write transaction through schema
// validation. It never escapes the process and is used only to make staged
// parent rows visible to FK checks before WAL commit.
type transactionContextKey struct{}

func withTransactionContext(ctx context.Context, tx *transaction) context.Context {
	return context.WithValue(ctx, transactionContextKey{}, tx)
}

func transactionFromContext(ctx context.Context) *transaction {
	if ctx == nil {
		return nil
	}
	tx, _ := ctx.Value(transactionContextKey{}).(*transaction)
	return tx
}

var (
	ErrTxClosed             = errors.New("transaction is closed")
	ErrTxValidation         = errors.New("transaction validation failed")
	ErrTxCommitFailed       = errors.New("transaction commit failed")
	ErrTxRollbackFailed     = errors.New("transaction rollback failed")
	ErrTxReceiptUnsupported = errors.New("exact transaction commit receipts unsupported")
	ErrTxEngineUnsupported  = errors.New("storage engine does not support transactions")
	ErrTxConflict           = errors.New("transaction conflict")
	ErrRecordNotFound       = errors.New("record not found")
	ErrVersionConflict      = errors.New("version conflict")
	// ErrForeignKeyCycle is returned when a cascading delete would revisit a
	// row already on the current cascade stack.  Failing the transaction is
	// safer than recursing indefinitely or partially publishing a cycle.
	ErrForeignKeyCycle = errors.New("foreign-key cascade cycle detected")
)

type cascadeContextKey struct{}

// cascadeState is shared by every recursive delete in one transaction.  The
// active set detects cycles, while planned prevents the same row from being
// staged twice when multiple foreign keys point at it.
type cascadeState struct {
	active  map[string]struct{}
	planned map[string]struct{}
}

func cascadeStateFromContext(ctx context.Context) *cascadeState {
	if ctx == nil {
		return nil
	}
	state, _ := ctx.Value(cascadeContextKey{}).(*cascadeState)
	return state
}

func withCascadeState(ctx context.Context, state *cascadeState) context.Context {
	return context.WithValue(ctx, cascadeContextKey{}, state)
}

// VersionConflictError reports an optimistic concurrency failure.
type VersionConflictError struct {
	Collection      string
	ID              string
	ExpectedVersion uint64
	ActualVersion   uint64
}

func (e *VersionConflictError) Error() string {
	return fmt.Sprintf("version conflict for %s/%s: expected version %d, actual version %d", e.Collection, e.ID, e.ExpectedVersion, e.ActualVersion)
}

func (e *VersionConflictError) Is(target error) bool {
	return target == ErrVersionConflict
}

// Tx exposes an explicit transactional batch write API.
type Tx interface {
	Insert(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}) error
	Update(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}) error
	Delete(ctx context.Context, collection, id string) error
	UpdateIfVersion(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}, expectedVersion uint64) error
	DeleteIfVersion(ctx context.Context, collection, id string, expectedVersion uint64) error
	DeleteBatch(ctx context.Context, collection string, ids []string) error
	ListByMetadata(ctx context.Context, collection, field string, value interface{}) ([]Record, error)
	// InsertOwned is like Insert but takes ownership of vector and metadata slices/maps.
	// The caller must not read or write them after the call returns.
	InsertOwned(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}) error
	// UpdateOwned is like Update but takes ownership of vector and metadata slices/maps.
	// The caller must not read or write them after the call returns.
	UpdateOwned(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}) error
	Upsert(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}) error
	// Rename changes the physical record key while preserving the record
	// payload. It is used for updates to declared PRIMARY KEY columns.
	Rename(ctx context.Context, collection, oldID, newID string, vector []float32, metadata map[string]interface{}) error
	Commit(ctx context.Context) error
	Rollback(ctx context.Context) error
}

// CommitReceipt identifies the exact durable WAL transaction boundary for a
// successful commit. CommitLSN is zero only when no transaction was committed
// (for example, an empty transaction).
type CommitReceipt struct {
	CommitLSN uint64
}

// ReceiptTx is the optional receipt-capable extension returned by
// BeginTxWithReceipt. Tx remains unchanged so existing callers and alternate
// transaction implementations retain source compatibility.
type ReceiptTx interface {
	Tx
	CommitWithReceipt(ctx context.Context) (CommitReceipt, error)
}

type txMutationKind uint8

const (
	txMutationInsert txMutationKind = iota
	txMutationUpdate
	txMutationDelete
	txMutationUpsert
	txMutationRename
)

type txMutation struct {
	metadata           map[string]interface{}
	collection         string
	id                 string
	oldID              string
	graphNodeID        uint64
	vector             []float32
	expectedVersion    uint64
	kind               txMutationKind
	hasExpectedVersion bool
}

type transaction struct {
	db         *Database
	ops        []txMutation
	graphDrops []txGraphDrop
	graphHooks []txGraphHook
	mu         sync.Mutex
	closed     bool
	committed  bool
}

// txGraphDrop is published in the same storage transaction as its record
// delete. The graph WAL frame is durable before the in-memory graph is
// updated, matching EpochTx's combined commit contract.
type txGraphDrop struct {
	collection string
	nodeID     uint64
	graph      Graph
}

// txGraphHook retains deprecated delete-hook graph mutations until the record
// transaction is durably committed.  Hooks therefore remain source-compatible
// without reintroducing the old record-first/graph-second commit split.
type txGraphHook struct {
	collection string
	txn        *graph.Txn
}

func (db *Database) applyGraphNodeDrops(drops []txGraphDrop, commitLSN uint64) error {
	for _, drop := range drops {
		g := drop.graph
		if g == nil {
			continue
		}
		gtx := g.BeginTxn()
		if err := gtx.DropNodeEdges(drop.nodeID); err != nil {
			_ = gtx.Rollback()
			return err
		}
		var err error
		if commitLSN != 0 {
			err = gtx.ApplyInMemoryAtLSN(commitLSN)
		} else {
			err = gtx.ApplyInMemory()
		}
		if err != nil {
			_ = gtx.Rollback()
			return err
		}
	}
	return nil
}

// BeginTx starts a new write transaction.
func (db *Database) BeginTx(ctx context.Context) (Tx, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	db.mu.RLock()
	defer db.mu.RUnlock()
	if db.closed {
		return nil, ErrDatabaseClosed
	}

	if db.metrics != nil {
		db.metrics.TxBegins.Inc()
	}

	return &transaction{db: db}, nil
}

// BeginTxWithReceipt starts a transaction whose commit can return the exact
// durable WAL LSN. The transaction still exposes the complete legacy Tx API.
func (db *Database) BeginTxWithReceipt(ctx context.Context) (ReceiptTx, error) {
	tx, err := db.BeginTx(ctx)
	if err != nil {
		return nil, err
	}
	receiptTx, ok := tx.(ReceiptTx)
	if !ok {
		return nil, ErrTxReceiptUnsupported
	}
	return receiptTx, nil
}

// WithTx runs fn inside a write transaction and commits on success.
func (db *Database) WithTx(ctx context.Context, fn func(tx Tx) error) error {
	tx, err := db.BeginTx(ctx)
	if err != nil {
		return err
	}

	if err := fn(tx); err != nil {
		if rbErr := tx.Rollback(ctx); rbErr != nil {
			return fmt.Errorf("%w: callback error: %v, rollback error: %v", ErrTxRollbackFailed, err, rbErr)
		}
		return err
	}

	return tx.Commit(ctx)
}

// WithTxReceipt runs fn inside a write transaction and returns the exact
// durable commit LSN. Callback failures roll back and return no receipt.
func (db *Database) WithTxReceipt(ctx context.Context, fn func(tx ReceiptTx) error) (CommitReceipt, error) {
	var empty CommitReceipt
	tx, err := db.BeginTxWithReceipt(ctx)
	if err != nil {
		return empty, err
	}
	if err := fn(tx); err != nil {
		if rbErr := tx.Rollback(ctx); rbErr != nil {
			return empty, fmt.Errorf("%w: callback error: %v, rollback error: %v", ErrTxRollbackFailed, err, rbErr)
		}
		return empty, err
	}
	return tx.CommitWithReceipt(ctx)
}

func (tx *transaction) Insert(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}) error {
	ctx = withTransactionContext(ctx, tx)
	if err := tx.validateStage(ctx, collection, id, vector, true); err != nil {
		return err
	}
	coll, err := tx.db.GetCollection(collection)
	if err != nil {
		return err
	}
	metadata, err = tx.prepareInsertMetadata(coll, metadata)
	if err != nil {
		return err
	}
	if err := coll.validateForeignKeys(ctx, id, metadata); err != nil {
		return err
	}
	if err := coll.validateUniqueConstraints(ctx, id, metadata); err != nil {
		return err
	}
	return tx.append(txMutation{
		kind:       txMutationInsert,
		collection: collection,
		id:         id,
		vector:     cloneVector(vector),
		metadata:   cloneMetadata(metadata),
	})
}

func (tx *transaction) InsertOwned(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}) error {
	ctx = withTransactionContext(ctx, tx)
	if err := tx.validateStage(ctx, collection, id, vector, true); err != nil {
		return err
	}
	coll, err := tx.db.GetCollection(collection)
	if err != nil {
		return err
	}
	metadata, err = tx.prepareInsertMetadata(coll, metadata)
	if err != nil {
		return err
	}
	if err := coll.validateForeignKeys(ctx, id, metadata); err != nil {
		return err
	}
	if err := coll.validateUniqueConstraints(ctx, id, metadata); err != nil {
		return err
	}
	return tx.append(txMutation{
		kind:       txMutationInsert,
		collection: collection,
		id:         id,
		vector:     vector,
		metadata:   metadata,
	})
}

func (tx *transaction) Update(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}) error {
	return tx.update(ctx, collection, id, vector, metadata, 0, false)
}

func (tx *transaction) UpdateOwned(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}) error {
	return tx.updateOwned(ctx, collection, id, vector, metadata, 0, false)
}

func (tx *transaction) Upsert(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}) error {
	ctx = withTransactionContext(ctx, tx)
	if err := tx.validateStage(ctx, collection, id, vector, true); err != nil {
		return err
	}
	coll, err := tx.db.GetCollection(collection)
	if err != nil {
		return err
	}
	metadata, err = tx.prepareInsertMetadata(coll, metadata)
	if err != nil {
		return err
	}
	if err := coll.validateForeignKeys(ctx, id, metadata); err != nil {
		return err
	}
	if err := coll.validateUniqueConstraints(ctx, id, metadata); err != nil {
		return err
	}
	return tx.append(txMutation{
		kind:       txMutationUpsert,
		collection: collection,
		id:         id,
		vector:     cloneVector(vector),
		metadata:   cloneMetadata(metadata),
	})
}

// prepareInsertMetadata creates the transaction-owned metadata image and
// applies schema defaults and CHECK constraints before any FK/UNIQUE check or
// mutation-log append. The caller's map is never mutated.
func (tx *transaction) prepareInsertMetadata(coll *Collection, metadata map[string]interface{}) (map[string]interface{}, error) {
	prepared := cloneMetadata(metadata)
	if prepared == nil {
		prepared = make(map[string]interface{})
	}
	coll.applyDefaults(prepared)
	if err := coll.validateNotNullConstraints(prepared); err != nil {
		return nil, err
	}
	if err := coll.validateCheckConstraints(prepared); err != nil {
		return nil, err
	}
	return prepared, nil
}

// Rename stages a primary-key/physical-key migration. The commit state turns
// this into a delete of oldID plus a put of newID in one WAL transaction.
func (tx *transaction) Rename(ctx context.Context, collection, oldID, newID string, vector []float32, metadata map[string]interface{}) error {
	ctx = withTransactionContext(ctx, tx)
	if oldID == "" || newID == "" {
		return fmt.Errorf("%w: primary-key values cannot be empty", ErrTxValidation)
	}
	if oldID == newID {
		return tx.Update(ctx, collection, oldID, vector, metadata)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := tx.validateStage(ctx, collection, newID, vector, false); err != nil {
		return err
	}
	coll, err := tx.db.GetCollection(collection)
	if err != nil {
		return err
	}
	old, err := tx.currentRecord(ctx, collection, oldID)
	if err != nil {
		return err
	}
	if _, err := tx.currentRecord(ctx, collection, newID); err == nil {
		return fmt.Errorf("%w: record %s already exists in collection %s", ErrTxConflict, newID, collection)
	}
	if vector == nil {
		vector = old.Vector
	}
	renameMetadata := cloneMetadata(old.Metadata)
	if metadata != nil {
		for key, value := range metadata {
			if renameMetadata == nil {
				renameMetadata = make(map[string]interface{})
			}
			renameMetadata[key] = cloneMetadataValue(value)
		}
	}
	if err := coll.validateNotNullConstraints(renameMetadata); err != nil {
		return err
	}
	if err := coll.validateCheckConstraints(renameMetadata); err != nil {
		return err
	}
	if err := coll.validateForeignKeys(ctx, newID, renameMetadata); err != nil {
		return err
	}
	if err := coll.validateUniqueConstraints(ctx, newID, renameMetadata); err != nil {
		return err
	}
	cascades, err := coll.collectUpdateCascades(ctx, oldID, newID, old.Metadata, renameMetadata)
	if err != nil {
		return err
	}
	graphNodeID, _ := tx.db.GetNodeID(ctx, collection, oldID)
	if err := tx.append(txMutation{kind: txMutationRename, collection: collection, id: newID, oldID: oldID, graphNodeID: graphNodeID, vector: cloneVector(vector), metadata: cloneMetadata(renameMetadata)}); err != nil {
		return err
	}
	for _, cascade := range cascades {
		if err := tx.appendCascadeUpdate(cascade); err != nil {
			return err
		}
	}
	return nil
}

func (tx *transaction) currentRecord(ctx context.Context, collection, id string) (Record, error) {
	col, err := tx.db.GetCollection(collection)
	if err != nil {
		return Record{}, err
	}
	rec, err := col.Get(ctx, id)
	if err != nil {
		if !isNotFoundError(err) && !errors.Is(err, ErrRecordNotFound) {
			return Record{}, err
		}
		rec = Record{ID: id}
		// A missing base record is only valid if a later staged insert creates it.
		rec.Metadata = nil
		rec.Vector = nil
	}
	present := err == nil
	tx.mu.Lock()
	ops := append([]txMutation(nil), tx.ops...)
	tx.mu.Unlock()
	for _, op := range ops {
		if op.collection != collection {
			continue
		}
		switch op.kind {
		case txMutationDelete:
			if op.id == id {
				present = false
			}
		case txMutationInsert, txMutationUpsert:
			if op.id == id {
				rec = Record{ID: id, Vector: cloneVector(op.vector), Metadata: cloneMetadata(op.metadata)}
				present = true
			}
		case txMutationUpdate:
			if op.id == id && present {
				if op.vector != nil {
					rec.Vector = cloneVector(op.vector)
				}
				rec.Metadata = mergeMetadata(rec.Metadata, op.metadata)
			}
		case txMutationRename:
			if op.oldID == id {
				present = false
			}
			if op.id == id {
				rec = Record{ID: id, Vector: cloneVector(op.vector), Metadata: cloneMetadata(op.metadata)}
				present = true
			}
		}
	}
	if !present {
		return Record{}, fmt.Errorf("%w: vector with ID %s not found", ErrRecordNotFound, id)
	}
	return rec, nil
}

// visibleRecords reconstructs one collection's transaction-local relation:
// committed rows plus ordered staged mutations, with last-write-wins
// semantics. It is intentionally used by referential-integrity validation so
// a parent inserted earlier in the same transaction is a valid FK target and
// a parent deleted earlier is no longer one.
func (tx *transaction) visibleRecords(ctx context.Context, collection string) ([]Record, error) {
	col, err := tx.db.GetCollection(collection)
	if err != nil {
		return nil, err
	}
	base, err := col.ListAll(ctx)
	if err != nil {
		return nil, err
	}

	byID := make(map[string]Record, len(base))
	order := make([]string, 0, len(base))
	for _, rec := range base {
		byID[rec.ID] = rec
		order = append(order, rec.ID)
	}
	tx.mu.Lock()
	ops := append([]txMutation(nil), tx.ops...)
	tx.mu.Unlock()
	for _, op := range ops {
		if op.collection != collection {
			continue
		}
		switch op.kind {
		case txMutationDelete:
			delete(byID, op.id)
		case txMutationInsert, txMutationUpsert:
			if _, exists := byID[op.id]; !exists {
				order = append(order, op.id)
			}
			byID[op.id] = Record{ID: op.id, Vector: cloneVector(op.vector), Metadata: cloneMetadata(op.metadata)}
		case txMutationUpdate:
			current, exists := byID[op.id]
			if !exists {
				continue
			}
			if op.vector != nil {
				current.Vector = cloneVector(op.vector)
			}
			if op.metadata != nil {
				merged := cloneMetadata(current.Metadata)
				if merged == nil {
					merged = make(map[string]interface{}, len(op.metadata))
				}
				for key, value := range op.metadata {
					merged[key] = value
				}
				current.Metadata = merged
			}
			byID[op.id] = current
		case txMutationRename:
			delete(byID, op.oldID)
			if _, exists := byID[op.id]; !exists {
				order = append(order, op.id)
			}
			byID[op.id] = Record{ID: op.id, Vector: cloneVector(op.vector), Metadata: cloneMetadata(op.metadata)}
		}
	}

	visible := make([]Record, 0, len(byID))
	for _, id := range order {
		if rec, exists := byID[id]; exists {
			visible = append(visible, rec)
		}
	}
	return visible, nil
}

func (tx *transaction) UpdateIfVersion(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}, expectedVersion uint64) error {
	return tx.update(ctx, collection, id, vector, metadata, expectedVersion, true)
}

func (tx *transaction) update(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}, expectedVersion uint64, hasExpectedVersion bool) error {
	ctx = withTransactionContext(ctx, tx)
	if err := tx.validateStage(ctx, collection, id, vector, false); err != nil {
		return err
	}
	coll, err := tx.db.GetCollection(collection)
	if err != nil {
		return err
	}
	existing, err := tx.currentRecord(ctx, collection, id)
	if err != nil {
		return err
	}
	merged := cloneMetadata(existing.Metadata)
	for k, v := range metadata {
		if merged == nil {
			merged = make(map[string]interface{})
		}
		merged[k] = cloneMetadataValue(v)
	}
	if err := coll.validateNotNullConstraints(merged); err != nil {
		return err
	}
	if err := coll.validateCheckConstraints(merged); err != nil {
		return err
	}
	if err := coll.validateForeignKeys(ctx, id, merged); err != nil {
		return err
	}
	if err := coll.validateUniqueConstraints(ctx, id, merged); err != nil {
		return err
	}
	cascades, err := coll.collectUpdateCascades(ctx, id, id, existing.Metadata, merged)
	if err != nil {
		return err
	}
	preparedDelta := cloneMetadata(metadata)
	if err := coll.validateJSONFields(preparedDelta); err != nil {
		return err
	}
	if err := tx.append(txMutation{
		kind:               txMutationUpdate,
		collection:         collection,
		id:                 id,
		vector:             cloneVector(vector),
		metadata:           preparedDelta,
		hasExpectedVersion: hasExpectedVersion,
		expectedVersion:    expectedVersion,
	}); err != nil {
		return err
	}
	for _, cascade := range cascades {
		if err := tx.appendCascadeUpdate(cascade); err != nil {
			return err
		}
	}
	return nil
}

// appendCascadeUpdate converts one referential-action update into a single
// transaction mutation. In particular, a composite ON UPDATE CASCADE is
// appended as one metadata image so no intermediate mixed tuple can become
// visible to later validation or commit preparation.
func (tx *transaction) appendCascadeUpdate(cascade cascadeOp) error {
	updateMeta := make(map[string]interface{})
	switch cascade.action {
	case catalog.OnDeleteSetNull, catalog.OnDeleteSetDefault:
		for col, val := range cascade.columnValues {
			if val.Null {
				updateMeta[col] = nil
			} else {
				updateMeta[col] = parseDefaultLiteral(val.Value)
			}
		}
	case catalog.OnDeleteCascade:
		if cascade.updateCascade {
			for col, val := range cascade.columnValues {
				if val.Null {
					updateMeta[col] = nil
				} else {
					if cascade.action == catalog.OnDeleteSetDefault {
						updateMeta[col] = parseDefaultLiteral(val.Value)
					} else {
						updateMeta[col] = val.Value
					}
				}
			}
		} else if cascade.sourceCol != "" {
			updateMeta[cascade.sourceCol] = cascade.newFKValue
		}
	}
	if len(updateMeta) == 0 {
		return nil
	}
	return tx.append(txMutation{
		kind: txMutationUpdate, collection: cascade.collectionName,
		id: cascade.recordID, metadata: updateMeta,
	})
}

func (tx *transaction) updateOwned(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}, expectedVersion uint64, hasExpectedVersion bool) error {
	ctx = withTransactionContext(ctx, tx)
	if err := tx.validateStage(ctx, collection, id, vector, false); err != nil {
		return err
	}
	coll, err := tx.db.GetCollection(collection)
	if err != nil {
		return err
	}
	existing, err := tx.currentRecord(ctx, collection, id)
	if err != nil {
		return err
	}
	merged := cloneMetadata(existing.Metadata)
	for k, v := range metadata {
		if merged == nil {
			merged = make(map[string]interface{})
		}
		merged[k] = cloneMetadataValue(v)
	}
	if err := coll.validateNotNullConstraints(merged); err != nil {
		return err
	}
	if err := coll.validateCheckConstraints(merged); err != nil {
		return err
	}
	if err := coll.validateForeignKeys(ctx, id, merged); err != nil {
		return err
	}
	if err := coll.validateUniqueConstraints(ctx, id, merged); err != nil {
		return err
	}
	preparedDelta := cloneMetadata(metadata)
	if err := coll.validateJSONFields(preparedDelta); err != nil {
		return err
	}
	return tx.append(txMutation{
		kind:               txMutationUpdate,
		collection:         collection,
		id:                 id,
		vector:             vector,
		metadata:           preparedDelta,
		hasExpectedVersion: hasExpectedVersion,
		expectedVersion:    expectedVersion,
	})
}

func (tx *transaction) Delete(ctx context.Context, collection, id string) error {
	return tx.delete(ctx, collection, id, 0, false)
}

func (tx *transaction) DeleteIfVersion(ctx context.Context, collection, id string, expectedVersion uint64) error {
	return tx.delete(ctx, collection, id, expectedVersion, true)
}

func (tx *transaction) delete(ctx context.Context, collection, id string, expectedVersion uint64, hasExpectedVersion bool) error {
	ctx = withTransactionContext(ctx, tx)
	if err := ctx.Err(); err != nil {
		return err
	}
	if collection == "" {
		return fmt.Errorf("%w: collection name cannot be empty", ErrTxValidation)
	}
	if id == "" {
		return fmt.Errorf("%w: vector ID cannot be empty", ErrTxValidation)
	}
	coll, err := tx.db.GetCollection(collection)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrCollectionNotFound, err)
	}
	state := cascadeStateFromContext(ctx)
	if state == nil {
		state = &cascadeState{
			active:  make(map[string]struct{}),
			planned: make(map[string]struct{}),
		}
		ctx = withCascadeState(ctx, state)
	}
	key := collection + "\x00" + id
	if _, active := state.active[key]; active {
		return fmt.Errorf("%w: %s", ErrForeignKeyCycle, key)
	}
	if _, planned := state.planned[key]; planned {
		return nil
	}
	state.active[key] = struct{}{}
	state.planned[key] = struct{}{}
	defer delete(state.active, key)

	// Enforce RESTRICT before staging and translate CASCADE references into
	// ordinary transaction mutations so the entire delete remains rollbackable.
	cascades, err := coll.checkDeleteFKReferences(ctx, id)
	if err != nil {
		return err
	}
	for _, cascade := range cascades {
		switch cascade.action {
		case catalog.OnDeleteSetNull, catalog.OnDeleteSetDefault:
			// SET NULL / SET DEFAULT: update child's FK columns in the
			// same transaction instead of deleting the child row.
			// Skip FK validation during this cascade update — the parent
			// being deleted would fail the child's FK check.
			updateMeta := make(map[string]interface{}, len(cascade.columnValues))
			for col, val := range cascade.columnValues {
				if val.Null {
					updateMeta[col] = nil
				} else {
					updateMeta[col] = val.Value
				}
			}
			cascadeCtx := context.WithValue(ctx, ctxKeySkipFKValidation{}, true)
			if err := tx.Update(cascadeCtx, cascade.collectionName, cascade.recordID, nil, updateMeta); err != nil {
				return err
			}
		default:
			// CASCADE (and any unset action): recursively delete child row.
			if err := tx.delete(ctx, cascade.collectionName, cascade.recordID, 0, false); err != nil {
				return err
			}
		}
	}
	if err := tx.append(txMutation{
		kind:               txMutationDelete,
		collection:         collection,
		id:                 id,
		hasExpectedVersion: hasExpectedVersion,
		expectedVersion:    expectedVersion,
	}); err != nil {
		return err
	}
	// Every live record has a database-scoped GraphNodeID. Record the graph
	// tombstone beside the record delete so SQL autocommit and explicit Tx
	// paths publish both operations in one WAL transaction.
	if nodeID, nodeErr := tx.db.GetNodeID(ctx, collection, id); nodeErr == nil && nodeID != 0 {
		graphStore := coll.GetGraph()
		coll.mu.RLock()
		deleteHooks := append([]DeleteHook(nil), coll.deleteHooks...)
		coll.mu.RUnlock()
		// Deprecated delete hooks are staged into the same combined graph WAL
		// transaction. They no longer receive a live graph transaction that can
		// publish ahead of the record delete.
		if len(deleteHooks) > 0 {
			var hookTxn *graph.Txn
			if graphStore != nil {
				hookTxn = graphStore.BeginTxn()
				hookTxn.SetCollection(collection)
			}
			if hookTxn == nil {
				hookTxn = &graph.Txn{}
			}
			for _, hook := range deleteHooks {
				if hook == nil {
					continue
				}
				if err := hook(hookTxn, nodeID); err != nil {
					_ = hookTxn.Rollback()
					return fmt.Errorf("delete hook failed: %w", err)
				}
			}
			tx.mu.Lock()
			if !tx.closed {
				tx.graphHooks = append(tx.graphHooks, txGraphHook{collection: collection, txn: hookTxn})
			}
			tx.mu.Unlock()
		}
		tx.mu.Lock()
		if !tx.closed {
			tx.graphDrops = append(tx.graphDrops, txGraphDrop{collection: collection, nodeID: nodeID, graph: graphStore})
		}
		tx.mu.Unlock()
	}
	return nil
}

func (tx *transaction) DeleteBatch(ctx context.Context, collection string, ids []string) error {
	for _, id := range ids {
		if err := tx.Delete(ctx, collection, id); err != nil {
			return err
		}
	}
	return nil
}

func (tx *transaction) ListByMetadata(ctx context.Context, collection, field string, value interface{}) ([]Record, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if collection == "" {
		return nil, fmt.Errorf("%w: collection name cannot be empty", ErrTxValidation)
	}
	if field == "" {
		return nil, fmt.Errorf("%w: metadata field cannot be empty", ErrTxValidation)
	}

	tx.mu.Lock()
	if tx.closed {
		tx.mu.Unlock()
		return nil, ErrTxClosed
	}
	ops := append([]txMutation(nil), tx.ops...)
	tx.mu.Unlock()

	coll, err := tx.db.GetCollection(collection)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCollectionNotFound, err)
	}
	entries, err := coll.getAllVectors(ctx)
	if err != nil {
		return nil, err
	}

	working := make(map[string]*index.VectorEntry, len(entries))
	for _, entry := range entries {
		working[entry.ID] = entry
	}
	for _, op := range ops {
		if op.collection != collection {
			continue
		}
		switch op.kind {
		case txMutationInsert:
			working[op.id] = &index.VectorEntry{
				ID:       op.id,
				Vector:   op.vector,
				Metadata: op.metadata,
			}
		case txMutationUpsert:
			working[op.id] = &index.VectorEntry{
				ID:       op.id,
				Vector:   op.vector,
				Metadata: op.metadata,
			}
		case txMutationUpdate:
			current := working[op.id]
			if current == nil {
				continue
			}
			updated := cloneIndexEntry(current)
			if op.vector != nil {
				updated.Vector = cloneVector(op.vector)
			}
			if op.metadata != nil {
				merged := cloneMetadata(current.Metadata)
				if merged == nil {
					merged = make(map[string]interface{}, len(op.metadata))
				}
				for k, v := range op.metadata {
					merged[k] = v
				}
				updated.Metadata = merged
			}
			working[op.id] = updated
		case txMutationDelete:
			delete(working, op.id)
		}
	}

	ids := make([]string, 0, len(working))
	for id, entry := range working {
		if metadataEqual(entry.Metadata, field, value) {
			ids = append(ids, id)
		}
	}
	sort.Strings(ids)

	records := make([]Record, 0, len(ids))
	for _, id := range ids {
		entry := working[id]
		records = append(records, Record{
			ID:       entry.ID,
			Vector:   entry.Vector,   // Iterate/cloneEntry already cloned
			Metadata: entry.Metadata, // Iterate/cloneEntry already cloned
			Version:  entry.Version,
		})
	}
	return records, nil
}

func (tx *transaction) Commit(ctx context.Context) error {
	return tx.commitWithReceipt(ctx, nil)
}

// CommitWithReceipt publishes the transaction through the same atomic record
// and graph WAL path as Commit, returning the exact durable commit LSN.
func (tx *transaction) CommitWithReceipt(ctx context.Context) (CommitReceipt, error) {
	var receipt CommitReceipt
	err := tx.commitWithReceipt(ctx, &receipt)
	return receipt, err
}

func (tx *transaction) commitWithReceipt(ctx context.Context, receipt *CommitReceipt) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	tx.mu.Lock()
	if tx.closed {
		tx.mu.Unlock()
		return ErrTxClosed
	}
	ops := append([]txMutation(nil), tx.ops...)
	graphDrops := append([]txGraphDrop(nil), tx.graphDrops...)
	graphHooks := append([]txGraphHook(nil), tx.graphHooks...)
	tx.closed = true
	tx.mu.Unlock()

	start := time.Now()
	graphOps := make([]storage.TxOperation, 0, len(graphDrops)+len(graphHooks))
	for _, hook := range graphHooks {
		if hook.txn == nil {
			continue
		}
		adds, removes, drops := hook.txn.StagedOps()
		for _, add := range adds {
			graphOps = append(graphOps, storage.TxOperation{
				Type: storage.TxOperationGraphEdgeAdd, Collection: hook.collection,
				EdgeSrc: add.Src, EdgeTgt: add.Tgt, EdgeWeight: add.Weight, EdgeKind: add.Kind, EdgeProperties: append([]byte(nil), add.Properties...),
			})
		}
		for _, remove := range removes {
			graphOps = append(graphOps, storage.TxOperation{
				Type: storage.TxOperationGraphEdgeRemove, Collection: hook.collection,
				EdgeSrc: remove.Src, EdgeTgt: remove.Tgt, EdgeKind: remove.Kind,
			})
		}
		for _, drop := range drops {
			graphOps = append(graphOps, storage.TxOperation{
				Type: storage.TxOperationGraphNodeDrop, Collection: hook.collection,
				EdgeSrc: drop.NodeID,
			})
		}
	}
	for _, drop := range graphDrops {
		graphOps = append(graphOps, storage.TxOperation{
			Type:       storage.TxOperationGraphNodeDrop,
			Collection: drop.collection,
			EdgeSrc:    drop.nodeID,
		})
	}
	var graphApplyFn func(uint64) error
	if len(graphDrops) > 0 || len(graphHooks) > 0 {
		graphApplyFn = func(commitLSN uint64) error {
			for _, hook := range graphHooks {
				if hook.txn == nil {
					continue
				}
				var err error
				if commitLSN != 0 {
					err = hook.txn.ApplyInMemoryAtLSN(commitLSN)
				} else {
					err = hook.txn.ApplyInMemory()
				}
				if err != nil {
					return err
				}
			}
			return tx.db.applyGraphNodeDrops(graphDrops, commitLSN)
		}
	}
	if err := tx.db.commitTxWithGraphReceipt(ctx, ops, nil, graphApplyFn, graphOps, receipt); err != nil {
		tx.mu.Lock()
		tx.closed = false
		tx.mu.Unlock()
		if tx.db.metrics != nil {
			if errors.Is(err, ErrVersionConflict) {
				tx.db.metrics.CASConflicts.Inc()
				tx.db.metrics.CASAborts.Inc()
			}
			if errors.Is(err, ErrTxValidation) || errors.Is(err, ErrTxConflict) || errors.Is(err, ErrCollectionNotFound) || errors.Is(err, ErrRecordNotFound) {
				tx.db.metrics.TxConflicts.Inc()
			}
		}
		return fmt.Errorf("%w: %v", ErrTxCommitFailed, err)
	}

	tx.mu.Lock()
	tx.committed = true
	tx.mu.Unlock()

	if tx.db.metrics != nil {
		tx.db.metrics.TxCommits.Inc()
		tx.db.metrics.TxCommitOps.Observe(float64(len(ops)))
		tx.db.metrics.TxCommitLatency.Observe(time.Since(start).Seconds())
	}

	return nil
}

func (tx *transaction) Rollback(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	tx.mu.Lock()
	defer tx.mu.Unlock()
	if tx.closed {
		return ErrTxClosed
	}
	tx.closed = true
	tx.ops = nil
	tx.graphDrops = nil
	for _, hook := range tx.graphHooks {
		if hook.txn != nil {
			_ = hook.txn.Rollback()
		}
	}
	tx.graphHooks = nil

	if tx.db.metrics != nil {
		tx.db.metrics.TxRollbacks.Inc()
	}
	return nil
}

// StagedOps returns the accumulated record operations for combined atomic
// commit through the storage engine's transactional interface.
func (tx *transaction) StagedOps() []storage.TxOperation {
	tx.mu.Lock()
	ops := append([]txMutation(nil), tx.ops...)
	tx.mu.Unlock()

	out := make([]storage.TxOperation, 0, len(ops))
	for _, op := range ops {
		switch op.kind {
		case txMutationInsert, txMutationUpdate, txMutationUpsert:
			out = append(out, storage.TxOperation{
				Type:               storage.TxOperationPut,
				Collection:         op.collection,
				ID:                 op.id,
				Vector:             op.vector,
				Metadata:           op.metadata,
				HasExpectedVersion: op.hasExpectedVersion,
				ExpectedVersion:    op.expectedVersion,
			})
		case txMutationDelete:
			out = append(out, storage.TxOperation{
				Type:       storage.TxOperationDelete,
				Collection: op.collection,
				ID:         op.id,
			})
		case txMutationRename:
			out = append(out,
				storage.TxOperation{Type: storage.TxOperationDelete, Collection: op.collection, ID: op.oldID},
				storage.TxOperation{Type: storage.TxOperationPut, Collection: op.collection, ID: op.id, Vector: op.vector, Metadata: op.metadata, GraphNodeID: op.graphNodeID},
			)
		}
	}
	return out
}

func (tx *transaction) append(op txMutation) error {
	tx.mu.Lock()
	defer tx.mu.Unlock()
	if tx.closed {
		return ErrTxClosed
	}
	tx.ops = append(tx.ops, op)
	return nil
}

func (tx *transaction) validateStage(ctx context.Context, collection, id string, vector []float32, requireVector bool) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if collection == "" {
		return fmt.Errorf("%w: collection name cannot be empty", ErrTxValidation)
	}
	if id == "" {
		return fmt.Errorf("%w: vector ID cannot be empty", ErrTxValidation)
	}
	coll, err := tx.db.GetCollection(collection)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrCollectionNotFound, err)
	}
	if requireVector && len(vector) != coll.config.Dimension {
		return fmt.Errorf("%w: vector dimension %d does not match collection dimension %d", ErrTxValidation, len(vector), coll.config.Dimension)
	}
	if !requireVector && vector != nil && len(vector) != coll.config.Dimension {
		return fmt.Errorf("%w: vector dimension %d does not match collection dimension %d", ErrTxValidation, len(vector), coll.config.Dimension)
	}
	return nil
}

func (db *Database) commitTx(ctx context.Context, ops []txMutation) error {
	return db.commitTxWithGraph(ctx, ops, nil, nil, nil)
}

// commitTxWithGraph commits record operations through the standard flow,
// appends graph TxOperations to the same WAL transaction, and calls
// graphApplyFn to publish graph topology in-memory only after durable success.
//
// If reserveGraphOps is non-nil, it is called after PrepareTx assigns ordinals
// and before CommitTx. It receives the number of new records and returns graph
// TxOperations with provisional node IDs remapped to reserved committed IDs.
// This enables epoch transactions to pre-assign GraphNodeIDs before the
// combined WAL write, avoiding separate record-then-graph commits.
func (db *Database) commitTxWithGraph(ctx context.Context, ops []txMutation, reserveGraphOps func(reservedIDs map[string]uint64) ([]storage.TxOperation, func(uint64) error), graphApplyFn func(uint64) error, initialGraphOps []storage.TxOperation) error {
	return db.commitTxWithGraphReceipt(ctx, ops, reserveGraphOps, graphApplyFn, initialGraphOps, nil)
}

func (db *Database) commitTxWithGraphReceipt(ctx context.Context, ops []txMutation, reserveGraphOps func(reservedIDs map[string]uint64) ([]storage.TxOperation, func(uint64) error), graphApplyFn func(uint64) error, initialGraphOps []storage.TxOperation, receipt *CommitReceipt) error {
	var graphOps []storage.TxOperation
	if reserveGraphOps != nil {
		// graphOps will be built later, after PrepareTx.
	} else {
		graphOps = nil // passed directly (backward compat path)
	}

	if len(ops) == 0 && reserveGraphOps == nil && len(initialGraphOps) == 0 {
		return nil
	}

	engine, ok := db.storage.(storage.TransactionalEngine)
	if !ok {
		return ErrTxEngineUnsupported
	}

	ops, err := db.routeTxMutations(ops)
	if err != nil {
		return err
	}
	collections, names, err := db.txCollections(ops)
	if err != nil {
		return err
	}

	releases := make([]func(), 0, len(names))
	locked := make([]*Collection, 0, len(names))
	defer func() {
		for i := len(locked) - 1; i >= 0; i-- {
			locked[i].unlockForTransaction()
		}
		for i := len(releases) - 1; i >= 0; i-- {
			releases[i]()
		}
	}()

	for _, name := range names {
		release, err := collections[name].acquireWrite(ctx)
		if err != nil {
			return err
		}
		releases = append(releases, release)
	}
	for _, name := range names {
		collection := collections[name]
		collection.lockForTransaction()
		if collection.closed {
			collection.unlockForTransaction()
			return ErrCollectionClosed
		}
		locked = append(locked, collection)
	}

	state, err := buildTransactionState(ctx, collections, names, ops)
	if err != nil {
		return err
	}
	defer state.close()
	if err := state.apply(ctx, ops); err != nil {
		return err
	}
	if err := state.validateCAS(); err != nil {
		return err
	}

	preparedOps, err := engine.PrepareTx(ctx, state.storageOps())
	if err != nil {
		return err
	}
	state.applyPreparedOrdinals(preparedOps)

	// Build graph ops after record preparation. Reserve GraphNodeIDs for new
	// records, set them on prepared ops so CommitTx uses them, and pass the
	// reserved ID → recordID mapping to the callback for graph op remapping.
	var innerGraphApplyFn func(uint64) error
	if reserveGraphOps != nil {
		// Count new records that need GraphNodeID assignment.
		// Keyed by composite "collection\x00recordID" to avoid
		// collisions when two collections share a record ID.
		newRecords := make(map[string]int) // compositeKey → index in preparedOps
		for i, op := range preparedOps {
			if op.Type == storage.TxOperationPut && op.GraphNodeID == 0 {
				newRecords[op.Collection+"\x00"+op.ID] = i
			}
		}
		reservedIDs := make(map[string]uint64, len(newRecords)) // compositeKey → reserved nodeID
		if len(newRecords) > 0 {
			idBase, err := engine.ReserveGraphNodeIDs(ctx, len(newRecords))
			if err != nil {
				return fmt.Errorf("reserve graph node IDs: %w", err)
			}
			nextID := idBase
			for compositeKey, idx := range newRecords {
				preparedOps[idx].GraphNodeID = nextID
				reservedIDs[compositeKey] = nextID
				nextID++
			}
		}
		graphOps, innerGraphApplyFn = reserveGraphOps(reservedIDs)
	}

	preparedDeltas, deltaCollections, err := state.prepareIndexDeltas(ctx, names)
	if err != nil {
		return err
	}
	committedDeltas := false
	defer func() {
		if committedDeltas {
			return
		}
		for _, prepared := range preparedDeltas {
			_ = prepared.Abort()
		}
	}()

	rebuildNames := make([]string, 0, len(names))
	for _, name := range names {
		if !deltaCollections[name] {
			rebuildNames = append(rebuildNames, name)
		}
	}
	newIndexes, err := state.buildIndexes(ctx, rebuildNames, db.logger)
	if err != nil {
		closeIndexes(newIndexes)
		return err
	}
	defer closeIndexes(newIndexes)

	// Append graph ops to the same WAL transaction.
	combinedOps := preparedOps
	if len(graphOps) > 0 {
		combinedOps = append(append([]storage.TxOperation{}, preparedOps...), graphOps...)
	}
	if len(initialGraphOps) > 0 {
		combinedOps = append(append([]storage.TxOperation{}, combinedOps...), initialGraphOps...)
	}

	var commitLSN uint64
	needExactLSN := receipt != nil || innerGraphApplyFn != nil || graphApplyFn != nil
	if durableEngine, ok := engine.(storage.DurableTransactionalEngine); ok && needExactLSN {
		durableReceipt, err := durableEngine.CommitTxDurable(ctx, combinedOps)
		if err != nil {
			if receipt != nil && durableReceipt.CommitLSN != 0 {
				receipt.CommitLSN = durableReceipt.CommitLSN
			}
			return err
		}
		commitLSN = durableReceipt.CommitLSN
		if receipt != nil {
			receipt.CommitLSN = commitLSN
		}
	} else {
		if receipt != nil {
			return ErrTxReceiptUnsupported
		}
		if err := engine.CommitTx(ctx, combinedOps); err != nil {
			return err
		}
	}
	// Storage is durable at this point. Invalidate metadata posting lists even
	// if a later index-publication step reports an error, otherwise a committed
	// UPDATE/DELETE could remain visible through stale equality postings.
	for _, name := range names {
		collection := collections[name]
		if parentName, _, isShard := parseShardName(name); isShard {
			if parent, err := db.GetCollection(parentName); err == nil {
				collection = parent
			}
		}
		collection.markMetadataIndexDirty()
	}

	hasCAS := false
	for _, op := range preparedOps {
		if op.HasExpectedVersion {
			hasCAS = true
			break
		}
	}
	if hasCAS && db.metrics != nil {
		db.metrics.CASSuccesses.Inc()
	}
	for _, name := range names {
		prepared := preparedDeltas[name]
		if prepared == nil {
			continue
		}
		if err := prepared.Commit(); err != nil {
			return fmt.Errorf("publish prepared index delta for %s: %w", name, err)
		}
	}
	committedDeltas = true

	for _, name := range rebuildNames {
		collection := collections[name]
		oldIndex := collection.index
		collection.index = newIndexes[name]
		if collection.transactionShard != nil {
			collection.transactionShard.index = collection.index
		}
		delete(newIndexes, name)
		_ = oldIndex.Close()
	}

	// Publish graph topology in-memory after durable WAL success.
	if innerGraphApplyFn != nil {
		if err := innerGraphApplyFn(commitLSN); err != nil {
			return fmt.Errorf("apply graph mutations: %w", err)
		}
	}
	if graphApplyFn != nil {
		if err := graphApplyFn(commitLSN); err != nil {
			return fmt.Errorf("apply graph mutations: %w", err)
		}
	}

	return nil
}

func (c *Collection) lockForTransaction() {
	if c.transactionMu != nil {
		c.transactionMu.Lock()
		return
	}
	c.mu.Lock()
}

func (c *Collection) unlockForTransaction() {
	if c.transactionMu != nil {
		c.transactionMu.Unlock()
		return
	}
	c.mu.Unlock()
}

// routeTxMutations translates logical sharded collection names to the
// owning physical shard before constructing transaction state. Public
// callers continue to stage operations against the logical collection.
func (db *Database) routeTxMutations(ops []txMutation) ([]txMutation, error) {
	routed := append([]txMutation(nil), ops...)
	for i := range routed {
		collection, err := db.GetCollection(routed[i].collection)
		if err != nil {
			return nil, err
		}
		if len(collection.shards) == 0 {
			continue
		}
		if routed[i].kind == txMutationRename {
			oldShard := shardForID(routed[i].oldID)
			newShard := shardForID(routed[i].id)
			if oldShard != newShard {
				return nil, fmt.Errorf("%w: primary-key rename across shards is not supported", ErrTxValidation)
			}
		}
		shardIndex := shardForID(routed[i].id)
		if shardIndex < 0 || shardIndex >= len(collection.shards) {
			return nil, fmt.Errorf("%w: shard %d unavailable for collection %s", ErrTxValidation, shardIndex, routed[i].collection)
		}
		routed[i].collection = collection.shards[shardIndex].name
	}
	return routed, nil
}

// prepareIndexDeltas builds unpublished generation candidates for indexes that
// support copy-on-write transaction publication. It deliberately inspects only
// touched IDs; Flat never receives the historical collection as a batch.
func (s *txCommitState) prepareIndexDeltas(ctx context.Context, names []string) (map[string]index.PreparedMutation, map[string]bool, error) {
	prepared := make(map[string]index.PreparedMutation, len(names))
	deltaCollections := make(map[string]bool, len(names))
	for _, name := range names {
		state := s.collections[name]
		if state == nil {
			continue
		}
		deltaIndex, ok := state.collection.index.(index.DeltaIndex)
		if !ok {
			continue
		}
		deltaCollections[name] = true
		if state.flat != nil {
			state.flat.sortEntries()
			puts := make([]*index.VectorEntry, 0, len(state.flat.entries))
			deletes := make([]string, 0, len(state.flat.entries))
			for i := range state.flat.entries {
				entry := &state.flat.entries[i]
				switch {
				case entry.current != nil:
					puts = append(puts, entryForIndex(state.collection.config.Metric, entry.current))
				case entry.base != nil:
					deletes = append(deletes, entry.id)
				}
			}
			if len(puts) == 0 && len(deletes) == 0 {
				continue
			}
			candidate, err := deltaIndex.PrepareMutations(ctx, puts, deletes)
			if err != nil {
				for _, existing := range prepared {
					_ = existing.Abort()
				}
				return nil, nil, fmt.Errorf("prepare index delta for %s: %w", name, err)
			}
			prepared[name] = candidate
			continue
		}
		ids := make([]string, 0, len(state.touched))
		for id := range state.touched {
			ids = append(ids, id)
		}
		sort.Strings(ids)
		puts := make([]*index.VectorEntry, 0, len(ids))
		deletes := make([]string, 0, len(ids))
		for _, id := range ids {
			after := state.working[id]
			before := state.base[id]
			switch {
			case after != nil:
				puts = append(puts, entryForIndex(state.collection.config.Metric, after))
			case before != nil:
				deletes = append(deletes, id)
			}
		}
		if len(puts) == 0 && len(deletes) == 0 {
			continue
		}
		candidate, err := deltaIndex.PrepareMutations(ctx, puts, deletes)
		if err != nil {
			for _, existing := range prepared {
				_ = existing.Abort()
			}
			return nil, nil, fmt.Errorf("prepare index delta for %s: %w", name, err)
		}
		prepared[name] = candidate
	}
	return prepared, deltaCollections, nil
}

func (db *Database) txCollections(ops []txMutation) (map[string]*Collection, []string, error) {
	namesSet := make(map[string]struct{}, len(ops))
	for _, op := range ops {
		namesSet[op.collection] = struct{}{}
	}

	names := make([]string, 0, len(namesSet))
	for name := range namesSet {
		names = append(names, name)
	}
	sort.Strings(names)

	collections := make(map[string]*Collection, len(names))
	for _, name := range names {
		collection, err := db.GetCollection(name)
		if err != nil {
			parentName, shardIndex, isShard := parseShardName(name)
			if !isShard {
				return nil, nil, err
			}
			parent, parentErr := db.GetCollection(parentName)
			if parentErr != nil || shardIndex < 0 || shardIndex >= len(parent.shards) {
				return nil, nil, err
			}
			physical := &parent.shards[shardIndex]
			config := *parent.config
			config.Sharded = false
			collection = &Collection{
				name:             physical.name,
				config:           &config,
				storage:          physical.storage,
				index:            physical.index,
				writes:           parent.writes,
				graph:            parent.graph,
				transactionMu:    &physical.mu,
				transactionShard: physical,
			}
		}
		collections[name] = collection
	}

	return collections, names, nil
}

type txCollectionState struct {
	collection *Collection
	base       map[string]*index.VectorEntry
	working    map[string]*index.VectorEntry
	touched    map[string]struct{}
	expected   map[string]uint64
	casTouched map[string]struct{}
	flat       *flatTxCollectionState
}

type txCommitState struct {
	collections  map[string]*txCollectionState
	graphNodeIDs map[string]map[string]uint64
}

// flatTxSlot is deliberately pointer-free because it is allocated in a
// transaction-local mmap arena and indexed by memory.IDMap. The entry payload
// remains in flatTxCollectionState.entries so Go keeps vectors and metadata
// reachable while the storage transaction is prepared and committed.
type flatTxSlot struct {
	row uint32
}

// flatTxEntry is a bounded touched-record overlay. base is immutable and
// current evolves as staged operations for the same ID are applied in order.
// Unlike the historical base/working maps, its cardinality is the number of
// transaction IDs, never the collection size.
type flatTxEntry struct {
	base               *index.VectorEntry
	current            *index.VectorEntry
	id                 string
	slot               *flatTxSlot
	expectedVersion    uint64
	hasExpectedVersion bool
}

type flatTxCollectionState struct {
	lookup  *memory.TypedIDMap[flatTxSlot]
	arena   *memory.Arena
	slots   []flatTxSlot
	entries []flatTxEntry
}

func newFlatTxCollectionState(ops []txMutation, collection string) (*flatTxCollectionState, error) {
	var count, keyBytes uint64
	for _, op := range ops {
		if op.collection != collection {
			continue
		}
		count++
		keyBytes += uint64(len(op.id))
		if op.kind == txMutationRename {
			count++
			keyBytes += uint64(len(op.oldID))
		}
	}
	if count == 0 {
		return nil, fmt.Errorf("%w: empty flat transaction state for %s", ErrTxValidation, collection)
	}
	if keyBytes < 4096 {
		keyBytes = 4096
	}
	slotBytes := count * uint64(unsafe.Sizeof(flatTxSlot{}))
	if slotBytes < 4096 {
		slotBytes = 4096
	}
	arena, err := memory.NewArena(slotBytes, 64)
	if err != nil {
		return nil, fmt.Errorf("flat transaction arena: %w", err)
	}
	slots, err := memory.ArenaSlice[flatTxSlot](arena, int(count))
	if err != nil {
		_ = arena.Free()
		return nil, fmt.Errorf("flat transaction slots: %w", err)
	}
	lookup, err := memory.NewTypedIDMap[flatTxSlot](memory.IDMapConfig{
		Capacity:  count,
		KeyBytes:  keyBytes,
		Alignment: 128,
	})
	if err != nil {
		_ = arena.Free()
		return nil, fmt.Errorf("flat transaction ID map: %w", err)
	}
	return &flatTxCollectionState{
		lookup:  lookup,
		arena:   arena,
		slots:   slots,
		entries: make([]flatTxEntry, 0, count),
	}, nil
}

func (s *flatTxCollectionState) close() {
	if s == nil {
		return
	}
	if s.lookup != nil {
		_ = s.lookup.Free()
		s.lookup = nil
	}
	if s.arena != nil {
		_ = s.arena.Free()
		s.arena = nil
	}
	s.slots = nil
	s.entries = nil
}

func (s *flatTxCollectionState) entry(ctx context.Context, collection *Collection, id string) (*flatTxEntry, error) {
	if slot, ok := s.lookup.GetString(id); ok {
		return &s.entries[slot.row], nil
	}

	store := collection.storage
	if collection.shards != nil {
		store = collection.getShard(id).storage
	}
	var base *index.VectorEntry
	entry, err := store.Get(ctx, id)
	if err != nil {
		if !isNotFoundError(err) {
			return nil, fmt.Errorf("load transaction record %s: %w", id, err)
		}
	} else {
		base = entry
	}

	slotIndex := len(s.slots)
	s.slots = s.slots[:slotIndex+1]
	slot := &s.slots[slotIndex]
	*slot = flatTxSlot{row: uint32(len(s.entries))}
	s.entries = append(s.entries, flatTxEntry{
		id:      id,
		base:    base,
		current: base,
		slot:    slot,
	})
	if err := s.lookup.PutString(id, slot); err != nil {
		s.entries = s.entries[:len(s.entries)-1]
		s.slots = s.slots[:len(s.slots)-1]
		return nil, fmt.Errorf("index flat transaction record %s: %w", id, err)
	}
	return &s.entries[slot.row], nil
}

func (s *flatTxCollectionState) sortEntries() {
	sort.Slice(s.entries, func(i, j int) bool { return s.entries[i].id < s.entries[j].id })
	for i := range s.entries {
		s.entries[i].slot.row = uint32(i)
	}
}

func (s *flatTxCollectionState) apply(ctx context.Context, collection *Collection, op txMutation) error {
	if op.kind == txMutationRename {
		oldEntry, err := s.entry(ctx, collection, op.oldID)
		if err != nil {
			return err
		}
		if oldEntry.current == nil {
			return fmt.Errorf("%w: vector with ID %s not found", ErrTxValidation, op.oldID)
		}
		newEntry, err := s.entry(ctx, collection, op.id)
		if err != nil {
			return err
		}
		if newEntry.current != nil {
			return fmt.Errorf("%w: record %s already exists in collection %s", ErrTxConflict, op.id, op.collection)
		}
		replacement := &index.VectorEntry{ID: op.id, Vector: op.vector, Metadata: op.metadata, Ordinal: oldEntry.current.Ordinal}
		oldEntry.current = nil
		newEntry.current = replacement
		return nil
	}
	entry, err := s.entry(ctx, collection, op.id)
	if err != nil {
		return err
	}
	if op.hasExpectedVersion {
		if entry.hasExpectedVersion && entry.expectedVersion != op.expectedVersion {
			return fmt.Errorf("%w: conflicting expected versions for %s/%s", ErrTxConflict, op.collection, op.id)
		}
		entry.expectedVersion = op.expectedVersion
		entry.hasExpectedVersion = true
	}

	switch op.kind {
	case txMutationInsert:
		if entry.current != nil && entry.base != nil {
			return fmt.Errorf("%w: record %s already exists in collection %s", ErrTxConflict, op.id, op.collection)
		}
		replacement := &index.VectorEntry{ID: op.id, Vector: op.vector, Metadata: op.metadata}
		if entry.current != nil {
			replacement.Ordinal = entry.current.Ordinal
		}
		entry.current = replacement
	case txMutationUpsert:
		replacement := &index.VectorEntry{ID: op.id, Vector: op.vector, Metadata: op.metadata}
		if entry.current != nil {
			replacement.Ordinal = entry.current.Ordinal
		}
		entry.current = replacement
	case txMutationUpdate:
		if entry.current == nil {
			if op.hasExpectedVersion {
				return fmt.Errorf("%w: %s", ErrRecordNotFound, op.id)
			}
			return fmt.Errorf("%w: vector with ID %s not found", ErrTxValidation, op.id)
		}
		updated := cloneIndexEntry(entry.current)
		if op.vector != nil {
			updated.Vector = cloneVector(op.vector)
		}
		if op.metadata == nil {
			updated.Metadata = cloneMetadata(entry.current.Metadata)
		} else {
			merged := cloneMetadata(entry.current.Metadata)
			if merged == nil {
				merged = make(map[string]interface{}, len(op.metadata))
			}
			for k, v := range op.metadata {
				merged[k] = v
			}
			updated.Metadata = merged
		}
		entry.current = updated
	case txMutationDelete:
		if entry.current == nil {
			if op.hasExpectedVersion {
				return fmt.Errorf("%w: %s", ErrRecordNotFound, op.id)
			}
			return nil
		}
		entry.current = nil
	default:
		return fmt.Errorf("%w: unsupported mutation %d", ErrTxValidation, op.kind)
	}
	return nil
}

func buildTransactionState(ctx context.Context, collections map[string]*Collection, names []string, ops []txMutation) (*txCommitState, error) {
	state := &txCommitState{
		collections:  make(map[string]*txCollectionState, len(names)),
		graphNodeIDs: make(map[string]map[string]uint64),
	}
	fail := func(err error) (*txCommitState, error) {
		state.close()
		return nil, err
	}

	for _, name := range names {
		collection := collections[name]
		if _, isDelta := collection.index.(index.DeltaIndex); isDelta {
			flat, err := newFlatTxCollectionState(ops, name)
			if err != nil {
				return fail(err)
			}
			state.collections[name] = &txCollectionState{collection: collection, flat: flat}
			continue
		}
		entries, err := collection.getAllVectors(ctx)
		if err != nil {
			return fail(err)
		}

		base := make(map[string]*index.VectorEntry, len(entries))
		working := make(map[string]*index.VectorEntry, len(entries))
		for _, entry := range entries {
			base[entry.ID] = entry
			working[entry.ID] = entry
		}

		state.collections[name] = &txCollectionState{
			collection: collection,
			base:       base,
			working:    working,
			touched:    make(map[string]struct{}),
			expected:   make(map[string]uint64),
			casTouched: make(map[string]struct{}),
		}
	}

	return state, nil
}

func (s *txCommitState) close() {
	for _, state := range s.collections {
		state.flat.close()
	}
}

func (s *txCommitState) apply(ctx context.Context, ops []txMutation) error {
	for _, op := range ops {
		state := s.collections[op.collection]
		if state == nil {
			return fmt.Errorf("%w: collection %s not found", ErrTxValidation, op.collection)
		}
		if op.kind == txMutationRename && op.graphNodeID != 0 {
			if s.graphNodeIDs[op.collection] == nil {
				s.graphNodeIDs[op.collection] = make(map[string]uint64)
			}
			s.graphNodeIDs[op.collection][op.id] = op.graphNodeID
		}
		if state.flat != nil {
			if err := state.flat.apply(ctx, state.collection, op); err != nil {
				return err
			}
			continue
		}
		if op.kind == txMutationRename {
			old := state.working[op.oldID]
			if old == nil {
				return fmt.Errorf("%w: vector with ID %s not found", ErrTxValidation, op.oldID)
			}
			if _, exists := state.working[op.id]; exists {
				return fmt.Errorf("%w: record %s already exists in collection %s", ErrTxConflict, op.id, op.collection)
			}
			state.touched[op.oldID] = struct{}{}
			state.touched[op.id] = struct{}{}
			state.working[op.oldID] = nil
			state.working[op.id] = &index.VectorEntry{ID: op.id, Vector: op.vector, Metadata: op.metadata, Ordinal: old.Ordinal}
			continue
		}
		state.touched[op.id] = struct{}{}
		if op.hasExpectedVersion {
			if existing, ok := state.expected[op.id]; ok && existing != op.expectedVersion {
				return fmt.Errorf("%w: conflicting expected versions for %s/%s", ErrTxConflict, op.collection, op.id)
			}
			state.expected[op.id] = op.expectedVersion
			state.casTouched[op.id] = struct{}{}
		}

		switch op.kind {
		case txMutationInsert:
			current := state.working[op.id]
			base := state.base[op.id]
			if current != nil && base != nil {
				return fmt.Errorf("%w: record %s already exists in collection %s", ErrTxConflict, op.id, op.collection)
			}

			replacement := &index.VectorEntry{
				ID:       op.id,
				Vector:   op.vector,
				Metadata: op.metadata,
			}
			if current != nil {
				replacement.Ordinal = current.Ordinal
			}
			state.working[op.id] = replacement
		case txMutationUpsert:
			current := state.working[op.id]
			replacement := &index.VectorEntry{
				ID:       op.id,
				Vector:   op.vector,
				Metadata: op.metadata,
			}
			if current != nil {
				replacement.Ordinal = current.Ordinal
			}
			state.working[op.id] = replacement
		case txMutationUpdate:
			current := state.working[op.id]
			if current == nil {
				if op.hasExpectedVersion {
					return fmt.Errorf("%w: %s", ErrRecordNotFound, op.id)
				}
				return fmt.Errorf("%w: vector with ID %s not found", ErrTxValidation, op.id)
			}
			updated := cloneIndexEntry(current)
			if op.vector != nil {
				updated.Vector = cloneVector(op.vector)
			}
			if op.metadata == nil {
				updated.Metadata = cloneMetadata(current.Metadata)
			} else {
				merged := cloneMetadata(current.Metadata)
				if merged == nil {
					merged = make(map[string]interface{}, len(op.metadata))
				}
				for k, v := range op.metadata {
					merged[k] = v
				}
				updated.Metadata = merged
			}
			state.working[op.id] = updated
		case txMutationDelete:
			if _, exists := state.working[op.id]; !exists {
				if op.hasExpectedVersion {
					return fmt.Errorf("%w: %s", ErrRecordNotFound, op.id)
				}
				continue
			}
			delete(state.working, op.id)
		default:
			return fmt.Errorf("%w: unsupported mutation %d", ErrTxValidation, op.kind)
		}
	}

	return nil
}

func (s *txCommitState) storageOps() []storage.TxOperation {
	ops := make([]storage.TxOperation, 0)
	for collectionName, state := range s.collections {
		if state.flat != nil {
			state.flat.sortEntries()
			for i := range state.flat.entries {
				entry := &state.flat.entries[i]
				switch {
				case entry.current == nil && entry.base != nil:
					ops = append(ops, storage.TxOperation{
						Type:               storage.TxOperationDelete,
						Collection:         collectionName,
						ID:                 entry.id,
						ExpectedVersion:    entry.expectedVersion,
						HasExpectedVersion: entry.hasExpectedVersion,
					})
				case entry.current != nil:
					graphNodeID := uint64(0)
					if ids := s.graphNodeIDs[collectionName]; ids != nil {
						graphNodeID = ids[entry.id]
					}
					ops = append(ops, storage.TxOperation{
						Type:               storage.TxOperationPut,
						Collection:         collectionName,
						ID:                 entry.id,
						Ordinal:            entry.current.Ordinal,
						Vector:             entry.current.Vector,
						Metadata:           entry.current.Metadata,
						GraphNodeID:        graphNodeID,
						ExpectedVersion:    entry.expectedVersion,
						HasExpectedVersion: entry.hasExpectedVersion,
					})
				}
			}
			continue
		}
		ids := make([]string, 0, len(state.touched))
		for id := range state.touched {
			ids = append(ids, id)
		}
		sort.Strings(ids)

		for _, id := range ids {
			before := state.base[id]
			after := state.working[id]
			switch {
			case after == nil && before != nil:
				expectedVersion, hasExpectedVersion := state.expected[id]
				ops = append(ops, storage.TxOperation{
					Type:               storage.TxOperationDelete,
					Collection:         collectionName,
					ID:                 id,
					ExpectedVersion:    expectedVersion,
					HasExpectedVersion: hasExpectedVersion,
				})
			case after != nil:
				graphNodeID := uint64(0)
				if ids := s.graphNodeIDs[collectionName]; ids != nil {
					graphNodeID = ids[id]
				}
				expectedVersion, hasExpectedVersion := state.expected[id]
				ops = append(ops, storage.TxOperation{
					Type:               storage.TxOperationPut,
					Collection:         collectionName,
					ID:                 id,
					Ordinal:            after.Ordinal,
					Vector:             after.Vector,
					Metadata:           after.Metadata,
					GraphNodeID:        graphNodeID,
					ExpectedVersion:    expectedVersion,
					HasExpectedVersion: hasExpectedVersion,
				})
			}
		}
	}
	return ops
}

func (s *txCommitState) validateCAS() error {
	for collectionName, state := range s.collections {
		if state.flat != nil {
			for i := range state.flat.entries {
				entry := &state.flat.entries[i]
				if !entry.hasExpectedVersion {
					continue
				}
				if entry.base == nil {
					return fmt.Errorf("%w: %s", ErrRecordNotFound, entry.id)
				}
				if entry.base.Version != entry.expectedVersion {
					return &VersionConflictError{
						Collection:      collectionName,
						ID:              entry.id,
						ExpectedVersion: entry.expectedVersion,
						ActualVersion:   entry.base.Version,
					}
				}
			}
			continue
		}
		for id, expectedVersion := range state.expected {
			current := state.base[id]
			if current == nil {
				return fmt.Errorf("%w: %s", ErrRecordNotFound, id)
			}
			if current.Version != expectedVersion {
				return &VersionConflictError{
					Collection:      collectionName,
					ID:              id,
					ExpectedVersion: expectedVersion,
					ActualVersion:   current.Version,
				}
			}
		}
	}
	return nil
}

func (s *txCommitState) applyPreparedOrdinals(ops []storage.TxOperation) {
	for _, op := range ops {
		if op.Type != storage.TxOperationPut {
			continue
		}
		state := s.collections[op.Collection]
		if state == nil {
			continue
		}
		if state.flat != nil {
			slot, ok := state.flat.lookup.GetString(op.ID)
			if !ok {
				continue
			}
			entry := &state.flat.entries[slot.row]
			if entry.current == nil {
				continue
			}
			entry.current.Ordinal = op.Ordinal
			continue
		}
		entry := state.working[op.ID]
		if entry == nil {
			continue
		}
		entry.Ordinal = op.Ordinal
	}
}

func (s *txCommitState) buildIndexes(ctx context.Context, names []string, logger Logger) (map[string]index.Index, error) {
	start := time.Now()
	indexes := make(map[string]index.Index, len(names))
	totalRecords := 0
	for _, name := range names {
		colStart := time.Now()
		state := s.collections[name]
		entries := make([]*index.VectorEntry, 0, len(state.working))
		for _, entry := range state.working {
			if entry != nil {
				entries = append(entries, entry)
			}
		}
		sort.Slice(entries, func(i, j int) bool {
			return entries[i].ID < entries[j].ID
		})

		provider, _ := state.collection.storage.(interface {
			GetByOrdinal(uint32) ([]float32, error)
			Distance([]float32, uint32) (float32, error)
		})
		idx, err := buildIndexForEntries(ctx, state.collection.config, provider, entries)
		if err != nil {
			closeIndexes(indexes)
			return nil, err
		}
		indexes[name] = idx
		totalRecords += len(entries)
		if logger != nil {
			logger.Printf("libravdb: tx buildIndex collection=%s records=%d elapsed=%s", name, len(entries), time.Since(colStart).Round(time.Microsecond))
		}
	}
	if logger != nil {
		logger.Printf("libravdb: tx buildIndex summary collections=%d total_records=%d total_elapsed=%s", len(names), totalRecords, time.Since(start).Round(time.Microsecond))
	}
	return indexes, nil
}

func closeIndexes(indexes map[string]index.Index) {
	for _, idx := range indexes {
		if idx != nil {
			_ = idx.Close()
		}
	}
}

func cloneIndexEntry(entry *index.VectorEntry) *index.VectorEntry {
	if entry == nil {
		return nil
	}
	return &index.VectorEntry{
		ID:       entry.ID,
		Ordinal:  entry.Ordinal,
		Vector:   cloneVector(entry.Vector),
		Metadata: cloneMetadata(entry.Metadata),
		Version:  entry.Version,
	}
}

func metadataEqual(metadata map[string]interface{}, field string, value interface{}) bool {
	if metadata == nil {
		return value == nil
	}
	got, ok := metadata[field]
	if !ok {
		return value == nil
	}
	return reflect.DeepEqual(got, value)
}
