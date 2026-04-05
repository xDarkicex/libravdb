package libravdb

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/xDarkicex/libravdb/internal/index"
	"github.com/xDarkicex/libravdb/internal/storage"
)

var (
	ErrTxClosed            = errors.New("transaction is closed")
	ErrTxValidation        = errors.New("transaction validation failed")
	ErrTxCommitFailed      = errors.New("transaction commit failed")
	ErrTxRollbackFailed    = errors.New("transaction rollback failed")
	ErrTxEngineUnsupported = errors.New("storage engine does not support transactions")
	ErrTxConflict          = errors.New("transaction conflict")
)

// Tx exposes an explicit transactional batch write API.
type Tx interface {
	Insert(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}) error
	Update(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}) error
	Delete(ctx context.Context, collection, id string) error
	DeleteBatch(ctx context.Context, collection string, ids []string) error
	Commit(ctx context.Context) error
	Rollback(ctx context.Context) error
}

type txMutationKind uint8

const (
	txMutationInsert txMutationKind = iota
	txMutationUpdate
	txMutationDelete
)

type txMutation struct {
	kind       txMutationKind
	collection string
	id         string
	vector     []float32
	metadata   map[string]interface{}
}

type transaction struct {
	db        *Database
	mu        sync.Mutex
	ops       []txMutation
	closed    bool
	committed bool
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

func (tx *transaction) Insert(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}) error {
	if err := tx.validateStage(ctx, collection, id, vector, true); err != nil {
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

func (tx *transaction) Update(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}) error {
	if err := tx.validateStage(ctx, collection, id, vector, false); err != nil {
		return err
	}
	return tx.append(txMutation{
		kind:       txMutationUpdate,
		collection: collection,
		id:         id,
		vector:     cloneVector(vector),
		metadata:   cloneMetadata(metadata),
	})
}

func (tx *transaction) Delete(ctx context.Context, collection, id string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if collection == "" {
		return fmt.Errorf("%w: collection name cannot be empty", ErrTxValidation)
	}
	if id == "" {
		return fmt.Errorf("%w: vector ID cannot be empty", ErrTxValidation)
	}
	if _, err := tx.db.GetCollection(collection); err != nil {
		return fmt.Errorf("%w: %v", ErrCollectionNotFound, err)
	}
	return tx.append(txMutation{
		kind:       txMutationDelete,
		collection: collection,
		id:         id,
	})
}

func (tx *transaction) DeleteBatch(ctx context.Context, collection string, ids []string) error {
	for _, id := range ids {
		if err := tx.Delete(ctx, collection, id); err != nil {
			return err
		}
	}
	return nil
}

func (tx *transaction) Commit(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	tx.mu.Lock()
	if tx.closed {
		tx.mu.Unlock()
		return ErrTxClosed
	}
	ops := append([]txMutation(nil), tx.ops...)
	tx.closed = true
	tx.mu.Unlock()

	start := time.Now()
	if err := tx.db.commitTx(ctx, ops); err != nil {
		tx.mu.Lock()
		tx.closed = false
		tx.mu.Unlock()
		if tx.db.metrics != nil && (errors.Is(err, ErrTxValidation) || errors.Is(err, ErrTxConflict) || errors.Is(err, ErrCollectionNotFound)) {
			tx.db.metrics.TxConflicts.Inc()
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

	if tx.db.metrics != nil {
		tx.db.metrics.TxRollbacks.Inc()
	}
	return nil
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
	if len(ops) == 0 {
		return nil
	}

	engine, ok := db.storage.(storage.TransactionalEngine)
	if !ok {
		return ErrTxEngineUnsupported
	}

	collections, names, err := db.txCollections(ops)
	if err != nil {
		return err
	}

	releases := make([]func(), 0, len(names))
	locked := make([]*Collection, 0, len(names))
	defer func() {
		for i := len(locked) - 1; i >= 0; i-- {
			locked[i].mu.Unlock()
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
		collection.mu.Lock()
		if collection.closed {
			collection.mu.Unlock()
			locked = locked[:len(locked)]
			return ErrCollectionClosed
		}
		locked = append(locked, collection)
	}

	state, err := buildTransactionState(ctx, collections, names)
	if err != nil {
		return err
	}
	if err := state.apply(ops); err != nil {
		return err
	}

	preparedOps, err := engine.PrepareTx(ctx, state.storageOps())
	if err != nil {
		return err
	}
	state.applyPreparedOrdinals(preparedOps)

	newIndexes, err := state.buildIndexes(ctx, names)
	if err != nil {
		closeIndexes(newIndexes)
		return err
	}
	defer closeIndexes(newIndexes)

	if err := engine.CommitTx(ctx, preparedOps); err != nil {
		return err
	}

	for _, name := range names {
		collection := collections[name]
		oldIndex := collection.index
		collection.index = newIndexes[name]
		delete(newIndexes, name)
		_ = oldIndex.Close()
	}

	return nil
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
			return nil, nil, err
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
}

type txCommitState struct {
	collections map[string]*txCollectionState
}

func buildTransactionState(ctx context.Context, collections map[string]*Collection, names []string) (*txCommitState, error) {
	state := &txCommitState{
		collections: make(map[string]*txCollectionState, len(names)),
	}

	for _, name := range names {
		collection := collections[name]
		entries, err := collection.getAllVectors(ctx)
		if err != nil {
			return nil, err
		}

		base := make(map[string]*index.VectorEntry, len(entries))
		working := make(map[string]*index.VectorEntry, len(entries))
		for _, entry := range entries {
			base[entry.ID] = cloneIndexEntry(entry)
			working[entry.ID] = cloneIndexEntry(entry)
		}

		state.collections[name] = &txCollectionState{
			collection: collection,
			base:       base,
			working:    working,
			touched:    make(map[string]struct{}),
		}
	}

	return state, nil
}

func (s *txCommitState) apply(ops []txMutation) error {
	for _, op := range ops {
		state := s.collections[op.collection]
		if state == nil {
			return fmt.Errorf("%w: collection %s not found", ErrTxValidation, op.collection)
		}
		state.touched[op.id] = struct{}{}

		switch op.kind {
		case txMutationInsert:
			current := state.working[op.id]
			base := state.base[op.id]
			if current != nil && base != nil {
				return fmt.Errorf("%w: record %s already exists in collection %s", ErrTxConflict, op.id, op.collection)
			}

			replacement := &index.VectorEntry{
				ID:       op.id,
				Vector:   cloneVector(op.vector),
				Metadata: cloneMetadata(op.metadata),
			}
			if current != nil {
				replacement.Ordinal = current.Ordinal
			}
			state.working[op.id] = replacement
		case txMutationUpdate:
			current := state.working[op.id]
			if current == nil {
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
				if _, existed := state.base[op.id]; existed {
					continue
				}
				return fmt.Errorf("%w: vector with ID %s not found", ErrTxValidation, op.id)
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
				ops = append(ops, storage.TxOperation{
					Type:       storage.TxOperationDelete,
					Collection: collectionName,
					ID:         id,
				})
			case after != nil:
				ops = append(ops, storage.TxOperation{
					Type:       storage.TxOperationPut,
					Collection: collectionName,
					ID:         id,
					Ordinal:    after.Ordinal,
					Vector:     cloneVector(after.Vector),
					Metadata:   cloneMetadata(after.Metadata),
				})
			}
		}
	}
	return ops
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
		entry := state.working[op.ID]
		if entry == nil {
			continue
		}
		entry.Ordinal = op.Ordinal
	}
}

func (s *txCommitState) buildIndexes(ctx context.Context, names []string) (map[string]index.Index, error) {
	indexes := make(map[string]index.Index, len(names))
	for _, name := range names {
		state := s.collections[name]
		entries := make([]*index.VectorEntry, 0, len(state.working))
		for _, entry := range state.working {
			entries = append(entries, cloneIndexEntry(entry))
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
			return nil, err
		}
		indexes[name] = idx
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
	}
}
