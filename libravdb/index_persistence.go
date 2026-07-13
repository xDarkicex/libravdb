package libravdb

import (
	"context"
	"fmt"
	"sync"

	"github.com/xDarkicex/libravdb/internal/index"
	"github.com/xDarkicex/libravdb/internal/storage"
)

// indexPersistenceBridge implements singlefile.IndexSnapshotProvider to bridge
// the storage engine's checkpoint/recovery with the libravdb index layer.
type indexPersistenceBridge struct {
	engine storage.Engine
	db     *Database
	cache  map[string]index.Index
	mu     sync.Mutex
}

func (b *indexPersistenceBridge) SetEngine(e storage.Engine) {
	b.engine = e
}

// takeCachedIndex returns and removes a cached index for the given collection,
// or nil if no index was deserialized/rebuilt during recovery.
func (b *indexPersistenceBridge) takeCachedIndex(name string) index.Index {
	b.mu.Lock()
	defer b.mu.Unlock()
	idx := b.cache[name]
	delete(b.cache, name)
	return idx
}

func (b *indexPersistenceBridge) closeCachedIndexes() {
	b.mu.Lock()
	for name, idx := range b.cache {
		if idx != nil {
			idx.Close()
		}
		delete(b.cache, name)
	}
	b.mu.Unlock()
}

// SerializeIndex returns serialized bytes for a collection's index.
// Returns (nil, nil) for empty collections (no index to persist).
// The engine skips nil entries in the index chunk; on recovery these
// collections are rebuilt from Records.
func (b *indexPersistenceBridge) SerializeIndex(collectionName string) ([]byte, error) {
	indexBytes, _, err := b.SerializeIndexAt(collectionName, 0)
	return indexBytes, err
}

// SerializeIndexAt captures an index together with the exact transaction
// frontier represented by that image. Async HNSW workers are paused only for
// the serialization window; WAL admission remains independent and bounded.
func (b *indexPersistenceBridge) SerializeIndexAt(collectionName string, checkpointLSN uint64) ([]byte, uint64, error) {
	b.mu.Lock()
	db := b.db
	b.mu.Unlock()
	if db == nil {
		return nil, checkpointLSN, nil
	}
	db.mu.RLock()
	col, ok := db.collections[collectionName]
	db.mu.RUnlock()
	if !ok || col == nil {
		return nil, checkpointLSN, nil
	}
	col.mu.RLock()
	defer col.mu.RUnlock()
	idx := col.index
	if idx == nil {
		return nil, checkpointLSN, nil
	}
	if col.asyncIndex != nil {
		q := col.asyncIndex
		q.applyGate.Lock()
		defer q.applyGate.Unlock()
		if err := q.failureValue(); err != nil {
			return nil, q.applied.Load(), err
		}
		appliedLSN := q.preciseAppliedLocked()
		if appliedLSN > checkpointLSN {
			appliedLSN = checkpointLSN
		}
		indexBytes, err := idx.SerializeToBytes()
		return indexBytes, appliedLSN, err
	}
	indexBytes, err := idx.SerializeToBytes()
	return indexBytes, checkpointLSN, err
}

// DeserializeIndex restores a collection's index from serialized bytes.
func (b *indexPersistenceBridge) DeserializeIndex(collectionName string, indexBytes []byte, config *storage.CollectionConfig) error {
	idx, err := b.createIndexFromEngineConfig(config)
	if err != nil {
		return fmt.Errorf("deserialize: create index for %s: %w", collectionName, err)
	}
	if err := idx.DeserializeFromBytes(context.Background(), indexBytes); err != nil {
		idx.Close()
		return fmt.Errorf("deserialize: load index for %s: %w", collectionName, err)
	}
	b.mu.Lock()
	b.cache[collectionName] = idx
	b.mu.Unlock()
	return nil
}

// RebuildIndex rebuilds a collection's index from storage records.
func (b *indexPersistenceBridge) RebuildIndex(collectionName string, config *storage.CollectionConfig) error {
	idx, err := b.createIndexFromEngineConfig(config)
	if err != nil {
		return fmt.Errorf("rebuild: create index for %s: %w", collectionName, err)
	}

	col, err := b.engine.GetCollection(collectionName)
	if err != nil {
		idx.Close()
		return fmt.Errorf("rebuild: get storage for %s: %w", collectionName, err)
	}

	// Build list of entries from storage records.
	var entries []*index.VectorEntry
	err = col.Iterate(context.Background(), func(entry *index.VectorEntry) error {
		entries = append(entries, entry)
		return nil
	})
	if err != nil {
		idx.Close()
		return fmt.Errorf("rebuild: iterate records for %s: %w", collectionName, err)
	}

	if len(entries) > 0 {
		metric := DistanceMetric(config.Metric)
		if err := prepareIndexForEntries(context.Background(), idx, metric, entries); err != nil {
			idx.Close()
			return fmt.Errorf("rebuild: prepare entries for %s: %w", collectionName, err)
		}
		if err := insertEntriesIntoIndex(context.Background(), idx, metric, entries); err != nil {
			idx.Close()
			return fmt.Errorf("rebuild: insert entries for %s: %w", collectionName, err)
		}
	}

	b.mu.Lock()
	if previous := b.cache[collectionName]; previous != nil {
		previous.Close()
	}
	b.cache[collectionName] = idx
	b.mu.Unlock()
	return nil
}

// DiscardIndex removes a checkpoint-restored index for a collection deleted by
// post-checkpoint WAL replay.
func (b *indexPersistenceBridge) DiscardIndex(collectionName string) {
	b.mu.Lock()
	if previous := b.cache[collectionName]; previous != nil {
		previous.Close()
		delete(b.cache, collectionName)
	}
	b.mu.Unlock()
}

// IndexTypeVersion returns the index type code and format version.
func (b *indexPersistenceBridge) IndexTypeVersion(collectionName string) (indexType uint8, indexVersion uint16) {
	b.mu.Lock()
	db := b.db
	b.mu.Unlock()
	if db == nil {
		return 0, 1
	}
	db.mu.RLock()
	col, ok := db.collections[collectionName]
	db.mu.RUnlock()
	if !ok || col == nil {
		return 0, 1
	}
	return uint8(col.config.IndexType), 1
}

// CanRestoreIndex keeps HNSW recovery on the record-rebuild path until its
// off-heap deserializer has independent lifetime and corruption validation.
func (b *indexPersistenceBridge) CanRestoreIndex(_ string, indexType uint8, _ uint16) bool {
	return IndexType(indexType) != HNSW
}

// SnapshotVectors copies node vectors from provider-backed indexes into local
// storage so that subsequent SerializeIndex calls do not re-enter the provider.
func (b *indexPersistenceBridge) SnapshotVectors(ctx context.Context) error {
	b.mu.Lock()
	db := b.db
	b.mu.Unlock()
	if db == nil {
		return nil
	}
	db.mu.RLock()
	defer db.mu.RUnlock()
	for name, col := range db.collections {
		col.mu.RLock()
		if col.asyncIndex != nil {
			col.mu.RUnlock()
			continue
		}
		idx := col.index
		col.mu.RUnlock()
		if idx == nil {
			continue
		}
		if snapshotter, ok := idx.(interface {
			SnapshotVectorsFromProvider(context.Context) error
		}); ok {
			if err := snapshotter.SnapshotVectorsFromProvider(ctx); err != nil {
				return fmt.Errorf("snapshot vectors for %s: %w", name, err)
			}
		}
	}
	return nil
}

// createIndexFromEngineConfig creates an empty index from a storage.CollectionConfig.
// The index is not backed by a storage provider (no GetByOrdinal/Distance) since
// deserialization and rebuild don't need storage access — they populate the index
// with all data up front and the index owns its vector store internally.
func (b *indexPersistenceBridge) createIndexFromEngineConfig(config *storage.CollectionConfig) (index.Index, error) {
	libraConfig := &CollectionConfig{
		Dimension:      config.Dimension,
		Metric:         DistanceMetric(config.Metric),
		IndexType:      IndexType(config.IndexType),
		M:              config.M,
		EfConstruction: config.EfConstruction,
		EfSearch:       config.EfSearch,
		ML:             config.ML,
		NClusters:      config.NClusters,
		NProbes:        config.NProbes,
		RawVectorStore: config.RawVectorStore,
		RawStoreCap:    config.RawStoreCap,
		IDMapCapacity:  config.IDMapCapacity,
	}
	return createIndexForCollection(libraConfig, nil)
}
