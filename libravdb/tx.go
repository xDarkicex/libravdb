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

	"github.com/xDarkicex/libravdb/internal/index"
	"github.com/xDarkicex/libravdb/internal/storage"
	"github.com/xDarkicex/memory"
)

var (
	ErrTxClosed            = errors.New("transaction is closed")
	ErrTxValidation        = errors.New("transaction validation failed")
	ErrTxCommitFailed      = errors.New("transaction commit failed")
	ErrTxRollbackFailed    = errors.New("transaction rollback failed")
	ErrTxEngineUnsupported = errors.New("storage engine does not support transactions")
	ErrTxConflict          = errors.New("transaction conflict")
	ErrRecordNotFound      = errors.New("record not found")
	ErrVersionConflict     = errors.New("version conflict")
)

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
	Commit(ctx context.Context) error
	Rollback(ctx context.Context) error
}

type txMutationKind uint8

const (
	txMutationInsert txMutationKind = iota
	txMutationUpdate
	txMutationDelete
	txMutationUpsert
)

type txMutation struct {
	metadata           map[string]interface{}
	collection         string
	id                 string
	vector             []float32
	expectedVersion    uint64
	kind               txMutationKind
	hasExpectedVersion bool
}

type transaction struct {
	db        *Database
	ops       []txMutation
	mu        sync.Mutex
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

func (tx *transaction) InsertOwned(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}) error {
	if err := tx.validateStage(ctx, collection, id, vector, true); err != nil {
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
	if err := tx.validateStage(ctx, collection, id, vector, true); err != nil {
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

func (tx *transaction) UpdateIfVersion(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}, expectedVersion uint64) error {
	return tx.update(ctx, collection, id, vector, metadata, expectedVersion, true)
}

func (tx *transaction) update(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}, expectedVersion uint64, hasExpectedVersion bool) error {
	if err := tx.validateStage(ctx, collection, id, vector, false); err != nil {
		return err
	}
	return tx.append(txMutation{
		kind:               txMutationUpdate,
		collection:         collection,
		id:                 id,
		vector:             cloneVector(vector),
		metadata:           cloneMetadata(metadata),
		hasExpectedVersion: hasExpectedVersion,
		expectedVersion:    expectedVersion,
	})
}

func (tx *transaction) updateOwned(ctx context.Context, collection, id string, vector []float32, metadata map[string]interface{}, expectedVersion uint64, hasExpectedVersion bool) error {
	if err := tx.validateStage(ctx, collection, id, vector, false); err != nil {
		return err
	}
	return tx.append(txMutation{
		kind:               txMutationUpdate,
		collection:         collection,
		id:                 id,
		vector:             vector,
		metadata:           metadata,
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
		kind:               txMutationDelete,
		collection:         collection,
		id:                 id,
		hasExpectedVersion: hasExpectedVersion,
		expectedVersion:    expectedVersion,
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

	if err := engine.CommitTx(ctx, preparedOps); err != nil {
		return err
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
		delete(newIndexes, name)
		_ = oldIndex.Close()
	}

	return nil
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
	expected   map[string]uint64
	casTouched map[string]struct{}
	flat       *flatTxCollectionState
}

type txCommitState struct {
	collections map[string]*txCollectionState
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
		collections: make(map[string]*txCollectionState, len(names)),
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
		if state.flat != nil {
			if err := state.flat.apply(ctx, state.collection, op); err != nil {
				return err
			}
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
					ops = append(ops, storage.TxOperation{
						Type:               storage.TxOperationPut,
						Collection:         collectionName,
						ID:                 entry.id,
						Ordinal:            entry.current.Ordinal,
						Vector:             entry.current.Vector,
						Metadata:           entry.current.Metadata,
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
				expectedVersion, hasExpectedVersion := state.expected[id]
				ops = append(ops, storage.TxOperation{
					Type:               storage.TxOperationPut,
					Collection:         collectionName,
					ID:                 id,
					Ordinal:            after.Ordinal,
					Vector:             after.Vector,
					Metadata:           after.Metadata,
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
			entries = append(entries, entry)
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
