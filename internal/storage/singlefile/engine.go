package singlefile

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/xDarkicex/libravdb/internal/index"
	"github.com/xDarkicex/libravdb/internal/storage"
	"github.com/xDarkicex/libravdb/internal/storage/fsdurability"
	"github.com/xDarkicex/libravdb/internal/util"
	"github.com/xDarkicex/memory"
)

const (
	pageSize             = 4096
	formatVersion        = uint16(2) // On-disk file layout (header page, metapage, chunk framing)
	fileMagic            = "LIBRAVDB"
	fileCreator          = "libravdb/2.0.0 xDarkicex" // 24 bytes — embedded in file header
	headerMagic   uint32 = 0x4C564442
	metaMagic     uint32 = 0x4C56444D
	metaMagicV2   uint32 = 0x4C56444E // V2 metapage includes index persistence fields
	chunkMagic    uint32 = 0x4C564443

	chunkTypeSnapshot = uint16(1)
	chunkTypeWAL      = uint16(2)
	chunkTypeIndex    = uint16(3)
	indexBlockMagic   = uint32(0x4C564449) // "LVDI"
	indexBlockVersion = uint16(1)

	recordTypeTxBegin          = uint16(1)
	recordTypeTxCommit         = uint16(2)
	recordTypeTxAbort          = uint16(3)
	recordTypeCollectionCreate = uint16(10)
	recordTypeCollectionDelete = uint16(11)
	recordTypeRecordPut        = uint16(20)
	recordTypeRecordDelete     = uint16(21)
)

var castagnoli = crc32.MakeTable(crc32.Castagnoli)

var (
	replaceDatabaseFile = fsdurability.ReplaceFile
	syncDatabaseParent  = fsdurability.SyncParent
)

const (
	// Checkpoints are recovery accelerators, not the durability boundary.
	// Keeping them less frequent reduces sync pressure while WAL fsync still
	// preserves crash safety for acknowledged writes.
	checkpointThresholdBytes = 256 << 20
	checkpointThresholdOps   = 65536
	// Batch WAL buffer settings: vector entries are buffered and flushed together
	// to reduce fsync overhead. A batch is flushed when the buffered entry count
	// reaches batchSize or after batchFlushInterval elapses.
	batchSize          = 256                   // flush when buffer reaches this many vector entries
	batchFlushInterval = 10 * time.Millisecond // flush periodically if buffer not full
	// WAL transactions are serialized under Engine.mu, so their contiguous write
	// image can reuse one mmap-backed arena without synchronization or GC scans.
	walWriteArenaSize = 64 << 20
)

// groupCommitWindow is the short delay used to coalesce flushNow requests
// into a single durable WAL commit. Tests may temporarily override this.
// Each atomic is padded to 64 bytes to prevent false sharing between the
// ingestion threads (Store/Add) and the flusher loop (Load) on the same
// cache line.
var (
	groupCommitWindow     atomic.Int64
	_                     [7]uint64 // pad to 64-byte cache line boundary
	groupCommitMaxWindow  atomic.Int64
	_                     [7]uint64
	groupCommitStepWindow atomic.Int64
)

func init() {
	groupCommitWindow.Store(int64(1 * time.Millisecond))
	groupCommitMaxWindow.Store(int64(5 * time.Millisecond))
	groupCommitStepWindow.Store(int64(500 * time.Microsecond))
}

type fileHeader struct {
	Magic             [8]byte
	FormatVersion     uint16
	PageSize          uint16
	FeatureFlags      uint32
	FileID            uint64
	CreatedUnixNano   uint64
	Creator           [24]byte // "libravdb/2.0.0 xDarkicex"
	LastCheckpointLSN uint64
	ActiveMetaPage    uint64
	WALStartPage      uint64
	WALHeadPage       uint64
	WALTailPage       uint64
	Checksum          uint32
}

type metaPage struct {
	PageNumber      uint64
	Magic           uint32
	MetaEpoch       uint64
	RootCatalog     uint64
	RootFreelist    uint64
	LastAppliedLSN  uint64
	PageCount       uint64
	CollectionCount uint64
	SnapshotOffset  uint64
	SnapshotLength  uint64
	IndexOffset     uint64 // byte offset of index chunk (0 = no index persisted)
	IndexLength     uint64 // payload length of index chunk
	IndexChecksum   uint32 // CRC32 of entire indexBlock payload
	Checksum        uint32
}

type recoveryCandidate struct {
	meta  *metaPage
	state *persistedState
}

type chunkHeader struct {
	Magic      uint32
	Kind       uint16
	Version    uint16
	PayloadLen uint32
	Checksum   uint32
}

type walFrameHeader struct {
	Magic      uint32
	Version    uint16
	RecordType uint16
	LSN        uint64
	TxID       uint64
	PrevLSN    uint64
	PayloadLen uint32
	Checksum   uint32
}

type walRecord struct {
	PayloadEncoder *util.BinaryEncoder
	Payload        []byte
	Header         walFrameHeader
}

type persistedState struct {
	Collections      map[string]*persistedCollection `json:"collections"`
	NextCollectionID uint64                          `json:"next_collection_id"`
}

type persistedCollection struct {
	Records             map[string]*recordValue `json:"records"`
	vectorSFL           *memory.ShardedFreeList
	Config              storage.CollectionConfig `json:"config"`
	ordinalToID         []string
	vectorSlots         [][]byte
	ID                  uint64 `json:"id"`
	CreatedLSN          uint64 `json:"created_lsn"`
	UpdatedLSN          uint64 `json:"updated_lsn"`
	LiveCount           uint64 `json:"live_count"`
	NextOrdinal         uint32 `json:"next_ordinal"`
	reservedNextOrdinal atomic.Uint32
	Deleted             bool `json:"deleted"`
}

type recordValue struct {
	Metadata   map[string]interface{} `json:"metadata"`
	Vector     []float32              `json:"vector"`
	Version    uint64                 `json:"version"`
	CreatedLSN uint64                 `json:"created_lsn"`
	UpdatedLSN uint64                 `json:"updated_lsn"`
	Ordinal    uint32                 `json:"ordinal"`
	Deleted    bool                   `json:"deleted"`
}

type collectionCreatePayload struct {
	Name   string                   `json:"name"`
	Config storage.CollectionConfig `json:"config"`
}

type collectionDeletePayload struct {
	Name string `json:"name"`
}

type recordPutPayload struct {
	Metadata   map[string]interface{} `json:"metadata"`
	Collection string                 `json:"collection"`
	ID         string                 `json:"id"`
	Vector     []float32              `json:"vector"`
	Ordinal    uint32                 `json:"ordinal"`
}

type recordDeletePayload struct {
	Collection string `json:"collection"`
	ID         string `json:"id"`
}

// IndexSnapshotProvider is implemented by the layer above the engine to
// provide index serialization during checkpoint and deserialization during
// recovery. The engine itself has no knowledge of index internals.
type IndexSnapshotProvider interface {
	// SerializeIndex returns serialized bytes for a collection's index, or
	// (nil, nil) if the collection has no index yet (e.g. empty collection).
	SerializeIndex(collectionName string) ([]byte, error)
	// DeserializeIndex restores a collection's index from serialized bytes.
	// Config provides the collection's dimension, index type, and other
	// parameters needed to reconstruct the index.
	DeserializeIndex(collectionName string, indexBytes []byte, config *storage.CollectionConfig) error
	// RebuildIndex rebuilds a collection's index from scratch using the
	// engine's current Records. Called when no valid index checkpoint exists
	// or when the checkpoint is corrupt. Config provides the collection's
	// dimension, index type, and other parameters needed to create the index.
	RebuildIndex(collectionName string, config *storage.CollectionConfig) error
	// IndexTypeVersion returns the index type code (0=flat,1=hnsw,2=ivfpq)
	// and format version for verification on load.
	IndexTypeVersion(collectionName string) (indexType uint8, indexVersion uint16)
	// SnapshotVectors copies node vectors from provider-backed collections
	// into local storage so that SerializeIndex can proceed without calling
	// back into the provider (which would deadlock if the engine lock is held).
	// Called before e.mu.Lock in Compact and checkpoint paths.
	SnapshotVectors(ctx context.Context) error
}

// CoordinatedIndexSnapshotProvider attaches a WAL frontier to
// each serialized derived index. Recovery only trusts an index whose frontier
// covers that collection's state at the selected snapshot; lagging images are
// rebuilt once at that boundary before later WAL deltas are applied.
type CoordinatedIndexSnapshotProvider interface {
	SerializeIndexAt(collectionName string, checkpointLSN uint64) (indexBytes []byte, appliedLSN uint64, err error)
}

// IndexRestorePolicy lets a provider reject direct restoration of an otherwise
// compatible persisted index. Rejected indexes are rebuilt from durable records.
type IndexRestorePolicy interface {
	CanRestoreIndex(collectionName string, indexType uint8, indexVersion uint16) bool
}

// IncrementalIndexRecoveryProvider applies committed WAL mutations to an index
// that already represents the selected snapshot. Recovery invokes these methods
// in transaction and LSN order; returning an error aborts open before the index
// can become visible.
type IncrementalIndexRecoveryProvider interface {
	CanApplyIndexDeltas(collectionName string, config *storage.CollectionConfig) bool
	ApplyIndexPut(collectionName string, entry *index.VectorEntry, replace bool, previousOrdinal uint32, config *storage.CollectionConfig) error
	ApplyIndexDelete(collectionName, id string, ordinal uint32, config *storage.CollectionConfig) error
	DiscardIndex(collectionName string)
}

// indexBlockEntry is a single collection's serialized index within the index chunk.
type indexBlockEntry struct {
	name            string
	payload         []byte
	payloadChecksum uint32
	appliedLSN      uint64
	indexVersion    uint16
	indexType       uint8
	hasAppliedLSN   bool
}

// Engine is the single-file storage engine.
type Engine struct {
	recoveryErr   atomic.Value
	ctx           context.Context
	indexProvider IndexSnapshotProvider
	cancel        context.CancelFunc
	file          *os.File
	walWriteArena *memory.Arena
	walRequests   *walRequestPool
	state         *persistedState
	collections   map[string]*Collection
	path          string
	batchBuffer   struct {
		mu                 sync.Mutex
		flushMu            sync.Mutex
		entries            []batchEntry
		spareEntries       []batchEntry
		flusher            chan struct{}
		flushNow           []walRequestHandle
		spareFlushNow      []walRequestHandle
		flushSignalPending int32
		pendingWaiters     int32
	}
	dirtyOps             int
	compactionErrors     uint64
	lastTxID             uint64
	fileID               uint64
	dirtyBytes           uint64
	metaEpoch            uint64
	walTransactions      uint64
	walBytes             uint64
	batchFlushes         uint64
	batchedEntries       uint64
	checkpoints          uint64
	replayedTxs          uint64
	discardedTxs         uint64
	rebuiltIndexes       uint64
	replayedIndexPuts    uint64
	replayedIndexDeletes uint64
	lastLSN              uint64
	activeMetaPage       uint64
	mu                   sync.RWMutex
	status               atomic.Int32
	closed               atomic.Bool
	walSync              bool
	groupCommitTarget    int32
	groupCommitMaxDelay  time.Duration
	walSyncFn            func(*os.File) error // test hook; nil uses (*os.File).Sync
	dirty                bool                 // completion channels for foreground flushes
}

// batchEntry holds a buffered record pending WAL flush.
type batchEntry struct {
	collection string
	entry      *index.VectorEntry
	entries    []*index.VectorEntry
	firstLSN   uint64
	commitLSN  uint64
	walBytes   uint64
	encodedOne encodedPayload
	// encoded holds pre-encoded recordPut payloads, 1:1 with entries.
	// When set, flushBatch passes these to WAL framing to avoid
	// re-encoding under e.mu.Lock().
	encoded []encodedPayload
}

func (b *batchEntry) count() int {
	if b.entry != nil {
		return 1
	}
	return len(b.entries)
}

func (b *batchEntry) entryAt(i int) *index.VectorEntry {
	if b.entry != nil {
		return b.entry
	}
	return b.entries[i]
}

func (b *batchEntry) encodedAt(i int) encodedPayload {
	if b.entry != nil {
		return b.encodedOne
	}
	if i < len(b.encoded) {
		return b.encoded[i]
	}
	return encodedPayload{}
}

// startBatchFlusher is a hook for testing. It starts the background flusher goroutine.
// Defaults to launching the goroutine directly; can be overridden in tests.
var startBatchFlusher = func(e *Engine) {
	go e.batchFlusher()
}

// RecoveryStats exposes WAL and derived-index recovery outcomes.
type RecoveryStats struct {
	ReplayedTransactions  uint64
	DiscardedTransactions uint64
	RebuiltIndexes        uint64
	ReplayedIndexPuts     uint64
	ReplayedIndexDeletes  uint64
}

// Collection is a storage-backed collection view.
type Collection struct {
	engine *Engine
	name   string
	closed atomic.Bool
}

// Option is a functional option for New.
type Option func(*Engine) error

// WithIndexSnapshotProvider wires the index persistence bridge before recovery
// so persisted indexes can be deserialized during openExisting/loadIndexes.
func WithIndexSnapshotProvider(provider IndexSnapshotProvider) Option {
	return func(e *Engine) error {
		e.indexProvider = provider
		if es, ok := provider.(interface{ SetEngine(storage.Engine) }); ok {
			es.SetEngine(e)
		}
		return nil
	}
}

// WithWALSync controls whether foreground WAL commits wait for file.Sync.
// Production callers should keep this enabled. Disabling it is only suitable
// for benchmarks that intentionally measure the non-durable upper bound.
func WithWALSync(enabled bool) Option {
	return func(e *Engine) error {
		e.walSync = enabled
		return nil
	}
}

// WithWALGroupCommitTarget waits for up to maxDelay for target concurrent WAL
// transactions before syncing. It is intended for asynchronous index admission,
// where graph construction no longer needs to hold foreground writers open.
func WithWALGroupCommitTarget(target int, maxDelay time.Duration) Option {
	return func(e *Engine) error {
		if target <= 0 {
			return fmt.Errorf("WAL group commit target must be positive")
		}
		if maxDelay <= 0 {
			return fmt.Errorf("WAL group commit maximum delay must be positive")
		}
		e.groupCommitTarget = int32(target)
		e.groupCommitMaxDelay = maxDelay
		return nil
	}
}

// New opens or creates a single-file database.
func New(path string, opts ...Option) (storage.Engine, error) {
	resolved, err := resolveDatabasePath(path)
	if err != nil {
		return nil, err
	}

	file, err := os.OpenFile(resolved, os.O_RDWR|os.O_CREATE|oNoFollow, 0644)
	if err != nil {
		return nil, fmt.Errorf("open database file: %w", err)
	}

	// Check format version early before applying options or starting goroutines
	stat, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("stat database file: %w", err)
	}

	if stat.Size() > 0 {
		buf := make([]byte, 10)
		if _, err := file.ReadAt(buf, 0); err == nil {
			magic := string(buf[:8])
			version := binary.LittleEndian.Uint16(buf[8:10])
			if magic == fileMagic && version == 1 {
				file.Close()
				return nil, storage.ErrV1FormatMigrationRequired
			}
		}
	}

	engine := &Engine{
		path:        resolved,
		file:        file,
		state:       &persistedState{NextCollectionID: 1, Collections: make(map[string]*persistedCollection)},
		collections: make(map[string]*Collection),
		walSync:     true,
	}
	engine.ctx, engine.cancel = context.WithCancel(context.Background())

	// Apply options before recovery so provider is available to loadIndexes.
	for _, opt := range opts {
		if err := opt(engine); err != nil {
			file.Close()
			return nil, err
		}
	}

	// Initialize WAL batch buffer channels (flusher goroutine started after init succeeds)
	engine.batchBuffer.flusher = make(chan struct{})
	engine.batchBuffer.entries = make([]batchEntry, 0, batchSize)
	engine.batchBuffer.spareEntries = make([]batchEntry, 0, batchSize)
	engine.batchBuffer.flushNow = make([]walRequestHandle, 0, batchSize)
	engine.batchBuffer.spareFlushNow = make([]walRequestHandle, 0, batchSize)

	stat, err = file.Stat()
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("stat database file: %w", err)
	}
	engine.walRequests, err = newWALRequestPool()
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("initialize WAL request pool: %w", err)
	}

	if stat.Size() == 0 {
		if err := engine.initializeEmpty(); err != nil {
			_ = engine.walRequests.close()
			file.Close()
			return nil, err
		}
		if err := syncDatabaseParent(resolved); err != nil {
			_ = engine.walRequests.close()
			file.Close()
			return nil, fmt.Errorf("sync new database directory entry: %w", err)
		}
		// Start background flusher only after successful initialization
		startBatchFlusher(engine)
		engine.status.Store(int32(storage.StatusReady))
		engine.initReservedOrdinals()
		return engine, nil
	}

	if err := engine.openExisting(); err != nil {
		_ = engine.walRequests.close()
		file.Close()
		return nil, err
	}

	// Start background flusher only after successful open
	startBatchFlusher(engine)
	engine.initReservedOrdinals()
	return engine, nil
}

func resolveDatabasePath(path string) (string, error) {
	if path == "" {
		path = "./data.libravdb"
	}
	path = filepath.Clean(path)

	if strings.HasPrefix(path, ":memory:") {
		name := strings.TrimPrefix(path, ":memory:")
		if name == "" {
			name = "default"
		}
		safe := strings.NewReplacer(":", "-", "/", "-", "\\", "-").Replace(name)
		if safe == "" {
			safe = "default"
		}
		return filepath.Join(os.TempDir(), fmt.Sprintf("%s-%d.libravdb", safe, time.Now().UnixNano())), nil
	}

	if info, err := os.Stat(path); err == nil && info.IsDir() {
		return "", fmt.Errorf("storage path must be a .libravdb file, got directory %q", path)
	}

	if strings.HasSuffix(path, string(filepath.Separator)) {
		return "", fmt.Errorf("storage path must be a database file path, got directory-like path %q", path)
	}

	parent := filepath.Dir(path)
	if parent != "." && parent != "" {
		if err := os.MkdirAll(parent, 0755); err != nil {
			return "", fmt.Errorf("create parent directory: %w", err)
		}
	}

	return path, nil
}

func (e *Engine) initializeEmpty() error {
	e.fileID = uint64(time.Now().UnixNano())
	e.activeMetaPage = 1
	if err := e.writeInitialPages(); err != nil {
		return err
	}
	e.dirty = true
	return e.checkpointLocked()
}

func (e *Engine) openExisting() error {
	e.status.Store(int32(storage.StatusStarting))

	header, err := e.readHeader()
	if err != nil {
		e.fail(fmt.Errorf("read header: %w", err))
		return err
	}
	e.fileID = header.FileID

	chosen, err := e.selectRecoveryCandidate()
	if err != nil {
		e.fail(err)
		return err
	}
	e.metaEpoch = chosen.meta.MetaEpoch
	e.activeMetaPage = metaPageNumber(chosen.meta)
	e.lastLSN = chosen.meta.LastAppliedLSN
	e.state = chosen.state

	// Phase 1 was completed while selecting a complete recovery candidate. State
	// is only published after the newest complete snapshot has been decoded.
	e.status.Store(int32(storage.StatusRecoveringSnapshot))

	// Phase 2: load or rebuild indexes
	e.status.Store(int32(storage.StatusRecoveringIndexes))
	if err := e.loadIndexes(chosen.meta); err != nil {
		e.fail(fmt.Errorf("load indexes: %w", err))
		return err
	}

	// Phase 3: replay WAL
	e.status.Store(int32(storage.StatusReplayingWAL))
	if err := e.replayWAL(chosen.meta.LastAppliedLSN); err != nil {
		e.fail(fmt.Errorf("replay WAL: %w", err))
		return err
	}

	e.status.Store(int32(storage.StatusReady))
	return nil
}

// loadIndexes loads serialized indexes from the index chunk, or rebuilds
// them from Records if no valid index chunk exists.
func (e *Engine) loadIndexes(chosen *metaPage) error {
	if e.indexProvider == nil {
		return nil // no provider registered, indexes will be built by collection layer
	}

	// No index persisted — rebuild all from Records.
	if chosen.IndexOffset == 0 || chosen.IndexLength == 0 {
		return e.rebuildIndexesFromRecords()
	}

	// Read and validate the index chunk.
	indexBlock, err := e.readChunkAt(chosen.IndexOffset)
	if err != nil {
		// Index chunk unreadable — rebuild all.
		return e.rebuildIndexesFromRecords()
	}
	if uint64(len(indexBlock)) != chosen.IndexLength {
		return e.rebuildIndexesFromRecords()
	}
	if crc32.Checksum(indexBlock, castagnoli) != chosen.IndexChecksum {
		// Block-level checksum mismatch — rebuild all.
		return e.rebuildIndexesFromRecords()
	}

	entries, err := decodeIndexBlock(indexBlock)
	if err != nil {
		return e.rebuildIndexesFromRecords()
	}
	handled := make(map[string]struct{}, len(entries))

	// Per-collection validation and deserialization.
	for _, entry := range entries {
		collection := e.state.Collections[entry.name]
		if collection == nil {
			continue // stale collection, skip
		}
		if crc32.Checksum(entry.payload, castagnoli) != entry.payloadChecksum {
			// Per-collection checksum fail — rebuild this collection only.
			if err := e.rebuildCollectionIndexFromRecords(entry.name, collection); err != nil {
				return err
			}
			handled[entry.name] = struct{}{}
			continue
		}

		// Validate index type matches current collection config.
		if entry.indexType != uint8(collection.Config.IndexType) {
			if err := e.rebuildCollectionIndexFromRecords(entry.name, collection); err != nil {
				return err
			}
			handled[entry.name] = struct{}{}
			continue
		}

		// A derived index is directly recoverable when its transaction frontier
		// covers this collection's latest mutation. Unrelated collections may
		// advance the database checkpoint without invalidating this index.
		// Legacy blocks and genuinely lagging snapshots rebuild once at the
		// selected snapshot boundary before post-snapshot WAL delta replay.
		if !entry.hasAppliedLSN || entry.appliedLSN < collection.UpdatedLSN || entry.appliedLSN > chosen.LastAppliedLSN {
			if err := e.rebuildCollectionIndexFromRecords(entry.name, collection); err != nil {
				return err
			}
			handled[entry.name] = struct{}{}
			continue
		}

		// Validate index format version is supported.
		expectedType, expectedVersion := e.indexProvider.IndexTypeVersion(entry.name)
		if entry.indexType != expectedType || entry.indexVersion != expectedVersion {
			if err := e.rebuildCollectionIndexFromRecords(entry.name, collection); err != nil {
				return err
			}
			handled[entry.name] = struct{}{}
			continue
		}
		if policy, ok := e.indexProvider.(IndexRestorePolicy); ok &&
			!policy.CanRestoreIndex(entry.name, entry.indexType, entry.indexVersion) {
			if err := e.rebuildCollectionIndexFromRecords(entry.name, collection); err != nil {
				return err
			}
			handled[entry.name] = struct{}{}
			continue
		}

		if err := e.indexProvider.DeserializeIndex(entry.name, entry.payload, &collection.Config); err != nil {
			if err := e.rebuildCollectionIndexFromRecords(entry.name, collection); err != nil {
				return err
			}
			handled[entry.name] = struct{}{}
			continue
		}
		handled[entry.name] = struct{}{}
	}

	// A valid index block may intentionally omit derived indexes that were
	// behind durable records at checkpoint time. Rebuild every live collection
	// absent from the block so recovery never exposes a partial index.
	for name, collection := range e.state.Collections {
		if collection.Deleted {
			continue
		}
		if _, ok := handled[name]; ok {
			continue
		}
		if err := e.rebuildCollectionIndexFromRecords(name, collection); err != nil {
			return err
		}
	}

	return nil
}

// rebuildIndexesFromRecords rebuilds every collection's index from its Records map.
func (e *Engine) rebuildIndexesFromRecords() error {
	if e.indexProvider == nil {
		return nil
	}
	for name, collection := range e.state.Collections {
		if collection.Deleted {
			continue
		}
		if err := e.rebuildCollectionIndexFromRecords(name, collection); err != nil {
			return err
		}
	}
	return nil
}

// rebuildCollectionIndexFromRecords rebuilds a single collection's index from
// its Records map via the registered provider. This is faster than WAL replay
// because it iterates Records directly without JSON decoding or transaction
// bookkeeping overhead.
func (e *Engine) rebuildCollectionIndexFromRecords(name string, collection *persistedCollection) error {
	if e.indexProvider == nil {
		return nil
	}
	if err := e.indexProvider.RebuildIndex(name, &collection.Config); err != nil {
		return err
	}
	e.rebuiltIndexes++
	return nil
}

// fail transitions the engine to storage.StatusFailed and stores the error.
func (e *Engine) fail(err error) {
	e.recoveryErr.Store(err)
	e.status.Store(int32(storage.StatusFailed))
}

func metaPageNumber(meta *metaPage) uint64 {
	if meta == nil {
		return 1
	}
	return meta.PageNumber
}

func (e *Engine) writeInitialPages() error {
	header := &fileHeader{
		FormatVersion:   formatVersion,
		PageSize:        pageSize,
		FileID:          e.fileID,
		CreatedUnixNano: uint64(time.Now().UnixNano()),
		ActiveMetaPage:  1,
		WALStartPage:    3,
		WALHeadPage:     3,
		WALTailPage:     3,
	}
	copy(header.Magic[:], []byte(fileMagic))
	copy(header.Creator[:], []byte(fileCreator))
	if err := writeFixedPage(e.file, 0, encodeHeader(header, make([]byte, pageSize))); err != nil {
		return err
	}

	metaA := &metaPage{Magic: metaMagic, RootFreelist: 0}
	metaB := &metaPage{Magic: metaMagic, RootFreelist: math.MaxUint64}
	if err := writeFixedPage(e.file, 1, encodeMeta(metaA, make([]byte, pageSize))); err != nil {
		return err
	}
	if err := writeFixedPage(e.file, 2, encodeMeta(metaB, make([]byte, pageSize))); err != nil {
		return err
	}

	return e.file.Sync()
}

var pagePool = sync.Pool{
	New: func() interface{} {
		buf := make([]byte, pageSize)
		return &buf
	},
}

func writeFixedPage(file *os.File, page uint64, data []byte) error {
	if len(data) > pageSize {
		return fmt.Errorf("page payload too large: %d", len(data))
	}
	_, err := file.WriteAt(data, int64(page)*pageSize)
	return err
}

func encodeHeader(header *fileHeader, buf []byte) []byte {
	copy(buf[:8], header.Magic[:])
	binary.LittleEndian.PutUint16(buf[8:10], header.FormatVersion)
	binary.LittleEndian.PutUint16(buf[10:12], header.PageSize)
	binary.LittleEndian.PutUint32(buf[12:16], header.FeatureFlags)
	binary.LittleEndian.PutUint64(buf[16:24], header.FileID)
	binary.LittleEndian.PutUint64(buf[24:32], header.CreatedUnixNano)
	copy(buf[32:56], header.Creator[:])
	binary.LittleEndian.PutUint64(buf[56:64], header.LastCheckpointLSN)
	binary.LittleEndian.PutUint64(buf[64:72], header.ActiveMetaPage)
	binary.LittleEndian.PutUint64(buf[72:80], header.WALStartPage)
	binary.LittleEndian.PutUint64(buf[80:88], header.WALHeadPage)
	binary.LittleEndian.PutUint64(buf[88:96], header.WALTailPage)
	checksum := crc32.Checksum(buf[:96], castagnoli)
	binary.LittleEndian.PutUint32(buf[96:100], checksum)
	return buf
}

func encodeMeta(meta *metaPage, buf []byte) []byte {
	binary.LittleEndian.PutUint32(buf[0:4], metaMagicV2)
	binary.LittleEndian.PutUint64(buf[4:12], meta.MetaEpoch)
	binary.LittleEndian.PutUint64(buf[12:20], meta.RootCatalog)
	binary.LittleEndian.PutUint64(buf[20:28], meta.RootFreelist)
	binary.LittleEndian.PutUint64(buf[28:36], meta.LastAppliedLSN)
	binary.LittleEndian.PutUint64(buf[36:44], meta.PageCount)
	binary.LittleEndian.PutUint64(buf[44:52], meta.CollectionCount)
	binary.LittleEndian.PutUint64(buf[52:60], meta.SnapshotOffset)
	binary.LittleEndian.PutUint64(buf[60:68], meta.SnapshotLength)
	binary.LittleEndian.PutUint64(buf[68:76], meta.IndexOffset)
	binary.LittleEndian.PutUint64(buf[76:84], meta.IndexLength)
	binary.LittleEndian.PutUint32(buf[84:88], meta.IndexChecksum)
	checksum := crc32.Checksum(buf[:88], castagnoli)
	binary.LittleEndian.PutUint32(buf[88:92], checksum)
	return buf
}

func (e *Engine) readHeader() (*fileHeader, error) {
	bufPtr := pagePool.Get().(*[]byte)
	buf := *bufPtr
	defer pagePool.Put(bufPtr)
	if _, err := e.file.ReadAt(buf, 0); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}
	if string(buf[:8]) != fileMagic {
		return nil, fmt.Errorf("invalid database file magic")
	}
	version := binary.LittleEndian.Uint16(buf[8:10])
	header := &fileHeader{}
	copy(header.Magic[:], buf[:8])
	header.FormatVersion = version
	header.PageSize = binary.LittleEndian.Uint16(buf[10:12])
	header.FeatureFlags = binary.LittleEndian.Uint32(buf[12:16])
	header.FileID = binary.LittleEndian.Uint64(buf[16:24])
	header.CreatedUnixNano = binary.LittleEndian.Uint64(buf[24:32])

	if version >= 2 {
		// V2 header: 96-byte data with Creator field, checksum at 96:100.
		expected := crc32.Checksum(buf[:96], castagnoli)
		if got := binary.LittleEndian.Uint32(buf[96:100]); got != expected {
			return nil, fmt.Errorf("invalid header checksum")
		}
		copy(header.Creator[:], buf[32:56])
		header.LastCheckpointLSN = binary.LittleEndian.Uint64(buf[56:64])
		header.ActiveMetaPage = binary.LittleEndian.Uint64(buf[64:72])
		header.WALStartPage = binary.LittleEndian.Uint64(buf[72:80])
		header.WALHeadPage = binary.LittleEndian.Uint64(buf[80:88])
		header.WALTailPage = binary.LittleEndian.Uint64(buf[88:96])
		header.Checksum = binary.LittleEndian.Uint32(buf[96:100])
	} else {
		// V1 header: 72-byte data, checksum at 72:76.
		expected := crc32.Checksum(buf[:72], castagnoli)
		if got := binary.LittleEndian.Uint32(buf[72:76]); got != expected {
			return nil, fmt.Errorf("invalid header checksum")
		}
		header.LastCheckpointLSN = binary.LittleEndian.Uint64(buf[32:40])
		header.ActiveMetaPage = binary.LittleEndian.Uint64(buf[40:48])
		header.WALStartPage = binary.LittleEndian.Uint64(buf[48:56])
		header.WALHeadPage = binary.LittleEndian.Uint64(buf[56:64])
		header.WALTailPage = binary.LittleEndian.Uint64(buf[64:72])
		header.Checksum = binary.LittleEndian.Uint32(buf[72:76])
	}
	return header, nil
}

func (e *Engine) readMetaPage(page uint64) (*metaPage, error) {
	bufPtr := pagePool.Get().(*[]byte)
	buf := *bufPtr
	defer pagePool.Put(bufPtr)
	if _, err := e.file.ReadAt(buf, int64(page)*pageSize); err != nil {
		return nil, err
	}
	magic := binary.LittleEndian.Uint32(buf[0:4])

	meta := &metaPage{PageNumber: page, Magic: magic}
	meta.MetaEpoch = binary.LittleEndian.Uint64(buf[4:12])
	meta.RootCatalog = binary.LittleEndian.Uint64(buf[12:20])
	meta.RootFreelist = binary.LittleEndian.Uint64(buf[20:28])
	meta.LastAppliedLSN = binary.LittleEndian.Uint64(buf[28:36])
	meta.PageCount = binary.LittleEndian.Uint64(buf[36:44])
	meta.CollectionCount = binary.LittleEndian.Uint64(buf[44:52])
	meta.SnapshotOffset = binary.LittleEndian.Uint64(buf[52:60])
	meta.SnapshotLength = binary.LittleEndian.Uint64(buf[60:68])

	switch magic {
	case metaMagicV2:
		meta.IndexOffset = binary.LittleEndian.Uint64(buf[68:76])
		meta.IndexLength = binary.LittleEndian.Uint64(buf[76:84])
		meta.IndexChecksum = binary.LittleEndian.Uint32(buf[84:88])
		meta.Checksum = binary.LittleEndian.Uint32(buf[88:92])
		expected := crc32.Checksum(buf[:88], castagnoli)
		if meta.Checksum != expected {
			return nil, fmt.Errorf("invalid V2 metapage checksum")
		}
	case metaMagic:
		meta.Checksum = binary.LittleEndian.Uint32(buf[68:72])
		expected := crc32.Checksum(buf[:68], castagnoli)
		if meta.Checksum != expected {
			return nil, fmt.Errorf("invalid metapage checksum")
		}
		// IndexOffset==0 means "no index persisted" — correct for V1 files.
	default:
		return nil, fmt.Errorf("invalid metapage magic: 0x%X", magic)
	}
	return meta, nil
}

func (e *Engine) selectRecoveryCandidate() (*recoveryCandidate, error) {
	meta1, err1 := e.readMetaPage(1)
	meta2, err2 := e.readMetaPage(2)
	if err1 != nil && err2 != nil {
		return nil, fmt.Errorf("failed to read any valid metapage: meta 1: %v; meta 2: %v", err1, err2)
	}

	ordered := [2]*metaPage{}
	count := 0
	if err1 == nil {
		ordered[count] = meta1
		count++
	}
	if err2 == nil {
		ordered[count] = meta2
		count++
	}
	if count == 2 && ordered[1].MetaEpoch > ordered[0].MetaEpoch {
		ordered[0], ordered[1] = ordered[1], ordered[0]
	}

	var candidateErrors [2]error
	for i := 0; i < count; i++ {
		candidate, err := e.decodeRecoveryCandidate(ordered[i])
		if err == nil {
			return candidate, nil
		}
		candidateErrors[i] = err
	}
	if count == 1 {
		return nil, fmt.Errorf("failed to read complete checkpoint from metapage %d: %w", ordered[0].PageNumber, candidateErrors[0])
	}
	return nil, fmt.Errorf(
		"failed to read any complete checkpoint: metapage %d: %v; metapage %d: %v",
		ordered[0].PageNumber,
		candidateErrors[0],
		ordered[1].PageNumber,
		candidateErrors[1],
	)
}

func (e *Engine) decodeRecoveryCandidate(meta *metaPage) (*recoveryCandidate, error) {
	page := meta.PageNumber
	state := &persistedState{
		NextCollectionID: 1,
		Collections:      make(map[string]*persistedCollection),
	}
	if meta.SnapshotLength == 0 {
		if meta.SnapshotOffset != 0 {
			return nil, fmt.Errorf("metapage %d has snapshot offset %d with zero length", page, meta.SnapshotOffset)
		}
		return &recoveryCandidate{meta: meta, state: state}, nil
	}
	if meta.SnapshotOffset < 3*pageSize {
		return nil, fmt.Errorf("metapage %d snapshot offset %d overlaps fixed pages", page, meta.SnapshotOffset)
	}
	payload, err := e.readChunkAtKind(meta.SnapshotOffset, chunkTypeSnapshot)
	if err != nil {
		return nil, fmt.Errorf("metapage %d snapshot: %w", page, err)
	}
	if uint64(len(payload)) != meta.SnapshotLength {
		return nil, fmt.Errorf("metapage %d snapshot length %d does not match chunk length %d", page, meta.SnapshotLength, len(payload))
	}
	state, err = decodeStateBinary(payload)
	if err != nil {
		return nil, fmt.Errorf("metapage %d decode snapshot: %w", page, err)
	}
	return &recoveryCandidate{meta: meta, state: state}, nil
}

func (e *Engine) readChunkAt(offset uint64) ([]byte, error) {
	return e.readChunkAtKind(offset, 0)
}

func (e *Engine) readChunkAtKind(offset uint64, expectedKind uint16) ([]byte, error) {
	if offset > math.MaxInt64-16 {
		return nil, fmt.Errorf("chunk offset %d exceeds file address range", offset)
	}
	headerBuf := make([]byte, 16)
	if _, err := e.file.ReadAt(headerBuf, int64(offset)); err != nil {
		return nil, err
	}
	header := decodeChunkHeader(headerBuf)
	if header.Magic != chunkMagic {
		return nil, fmt.Errorf("invalid chunk magic at offset %d", offset)
	}
	if expectedKind != 0 && header.Kind != expectedKind {
		return nil, fmt.Errorf("chunk kind %d at offset %d, want %d", header.Kind, offset, expectedKind)
	}
	if header.Version == 0 || header.Version > formatVersion {
		return nil, fmt.Errorf("unsupported chunk version %d at offset %d", header.Version, offset)
	}
	if header.PayloadLen > maxChunkSize {
		return nil, fmt.Errorf("chunk size %d exceeds limit %d at offset %d", header.PayloadLen, maxChunkSize, offset)
	}
	chunkEnd := offset + 16 + uint64(header.PayloadLen)
	if chunkEnd < offset {
		return nil, fmt.Errorf("chunk at offset %d overflows file address range", offset)
	}
	stat, err := e.file.Stat()
	if err != nil {
		return nil, err
	}
	if stat.Size() < 0 || chunkEnd > uint64(stat.Size()) {
		return nil, fmt.Errorf("chunk at offset %d ends at %d beyond file size %d", offset, chunkEnd, stat.Size())
	}
	payload := make([]byte, header.PayloadLen)
	if _, err := e.file.ReadAt(payload, int64(offset)+16); err != nil {
		return nil, err
	}
	if crc32.Checksum(payload, castagnoli) != header.Checksum {
		return nil, fmt.Errorf("invalid chunk checksum at offset %d", offset)
	}
	return payload, nil
}

func decodeChunkHeader(buf []byte) chunkHeader {
	return chunkHeader{
		Magic:      binary.LittleEndian.Uint32(buf[0:4]),
		Kind:       binary.LittleEndian.Uint16(buf[4:6]),
		Version:    binary.LittleEndian.Uint16(buf[6:8]),
		PayloadLen: binary.LittleEndian.Uint32(buf[8:12]),
		Checksum:   binary.LittleEndian.Uint32(buf[12:16]),
	}
}

func (e *Engine) replayWAL(lastApplied uint64) error {
	stat, err := e.file.Stat()
	if err != nil {
		return err
	}
	offset := int64(3 * pageSize)
	fileSize := stat.Size()
	pending := make(map[uint64][]walRecord)
	touchedCollections := make(map[string]struct{})
	deltaProvider, _ := e.indexProvider.(IncrementalIndexRecoveryProvider)

	for fileSize >= 16 && offset >= 0 && offset <= fileSize-16 {
		headerBuf := make([]byte, 16)
		if _, err := e.file.ReadAt(headerBuf, offset); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return err
		}
		chunk := decodeChunkHeader(headerBuf)
		if chunk.Magic != chunkMagic {
			break
		}
		if chunk.PayloadLen > maxChunkSize {
			if chunk.Kind != chunkTypeWAL {
				break
			}
			return fmt.Errorf("chunk size %d exceeds limit %d during replay", chunk.PayloadLen, maxChunkSize)
		}
		chunkEnd := offset + 16 + int64(chunk.PayloadLen)
		if chunkEnd < offset || chunkEnd > fileSize {
			// A live copier or interrupted append may stop inside the final
			// chunk. No complete commit frame exists beyond this boundary.
			break
		}

		// Skip non-WAL chunks. Snapshot and index chunks are validated
		// by recoverSnapshot / loadIndexes before replayWAL runs;
		// re-validating them here is redundant and fatal on legitimate
		// corruption that those paths already handle gracefully.
		if chunk.Kind != chunkTypeWAL {
			offset += int64(16 + chunk.PayloadLen)
			continue
		}

		payload := make([]byte, chunk.PayloadLen)
		if _, err := e.file.ReadAt(payload, offset+16); err != nil {
			return err
		}
		if crc32.Checksum(payload, castagnoli) != chunk.Checksum {
			return fmt.Errorf("invalid chunk checksum during replay")
		}

		record, err := decodeWALRecord(payload)
		if err != nil {
			return err
		}
		if record.Header.LSN <= lastApplied {
			offset += int64(16 + chunk.PayloadLen)
			continue
		}
		switch record.Header.RecordType {
		case recordTypeTxBegin:
			pending[record.Header.TxID] = []walRecord{record}
		case recordTypeTxCommit:
			frames := append(pending[record.Header.TxID], record)
			if err := e.applyCommittedFrames(frames, touchedCollections, deltaProvider); err != nil {
				return err
			}
			e.replayedTxs++
			if record.Header.LSN > e.lastLSN {
				e.lastLSN = record.Header.LSN
			}
			if record.Header.TxID > e.lastTxID {
				e.lastTxID = record.Header.TxID
			}
			delete(pending, record.Header.TxID)
		case recordTypeTxAbort:
			e.discardedTxs++
			if record.Header.LSN > e.lastLSN {
				e.lastLSN = record.Header.LSN
			}
			if record.Header.TxID > e.lastTxID {
				e.lastTxID = record.Header.TxID
			}
			delete(pending, record.Header.TxID)
		default:
			pending[record.Header.TxID] = append(pending[record.Header.TxID], record)
		}
		offset += int64(16 + chunk.PayloadLen)
	}
	e.discardedTxs += uint64(len(pending))
	if e.indexProvider != nil {
		for name := range touchedCollections {
			collection := e.state.Collections[name]
			if collection == nil || collection.Deleted {
				if discarder, ok := e.indexProvider.(interface{ DiscardIndex(string) }); ok {
					discarder.DiscardIndex(name)
				}
				continue
			}
			if err := e.rebuildCollectionIndexFromRecords(name, collection); err != nil {
				return fmt.Errorf("rebuild replayed index %s: %w", name, err)
			}
		}
	}
	return nil
}

func decodeWALRecord(payload []byte) (walRecord, error) {
	if len(payload) < 40 {
		return walRecord{}, fmt.Errorf("wal payload too small")
	}
	header := walFrameHeader{
		Magic:      binary.LittleEndian.Uint32(payload[0:4]),
		Version:    binary.LittleEndian.Uint16(payload[4:6]),
		RecordType: binary.LittleEndian.Uint16(payload[6:8]),
		LSN:        binary.LittleEndian.Uint64(payload[8:16]),
		TxID:       binary.LittleEndian.Uint64(payload[16:24]),
		PrevLSN:    binary.LittleEndian.Uint64(payload[24:32]),
		PayloadLen: binary.LittleEndian.Uint32(payload[32:36]),
		Checksum:   binary.LittleEndian.Uint32(payload[36:40]),
	}
	body := payload[40:]
	if uint32(len(body)) != header.PayloadLen {
		return walRecord{}, fmt.Errorf("wal payload length mismatch")
	}
	if crc32.Checksum(body, castagnoli) != header.Checksum {
		return walRecord{}, fmt.Errorf("invalid wal frame checksum")
	}
	return walRecord{Header: header, Payload: body}, nil
}

func (e *Engine) applyCommittedFrames(
	frames []walRecord,
	touchedCollections map[string]struct{},
	deltaProvider IncrementalIndexRecoveryProvider,
) error {
	for _, record := range frames {
		switch record.Header.RecordType {
		case recordTypeCollectionCreate:
			payload, err := decodeCollectionCreatePayloadBinary(record.Payload)
			if err != nil {
				return err
			}
			previous := e.state.Collections[payload.Name]
			created := previous == nil || previous.Deleted
			e.applyCreateCollection(payload.Name, payload.Config, record.Header.LSN)
			if !created {
				continue
			}
			if deltaProvider == nil || !deltaProvider.CanApplyIndexDeltas(payload.Name, &payload.Config) {
				touchedCollections[payload.Name] = struct{}{}
				continue
			}
			deltaProvider.DiscardIndex(payload.Name)
			collection := e.state.Collections[payload.Name]
			if err := e.rebuildCollectionIndexFromRecords(payload.Name, collection); err != nil {
				return fmt.Errorf("initialize replayed index %s at LSN %d: %w", payload.Name, record.Header.LSN, err)
			}
		case recordTypeCollectionDelete:
			payload, err := decodeCollectionDeletePayloadBinary(record.Payload)
			if err != nil {
				return err
			}
			collection := e.state.Collections[payload.Name]
			deleted := collection != nil && !collection.Deleted
			var config *storage.CollectionConfig
			if collection != nil {
				config = &collection.Config
			}
			e.applyDeleteCollection(payload.Name, record.Header.LSN)
			if !deleted {
				continue
			}
			if deltaProvider == nil || !deltaProvider.CanApplyIndexDeltas(payload.Name, config) {
				touchedCollections[payload.Name] = struct{}{}
				continue
			}
			deltaProvider.DiscardIndex(payload.Name)
		case recordTypeRecordPut:
			payload, err := decodeRecordPutPayloadBinary(record.Payload)
			if err != nil {
				return err
			}
			collection := e.state.Collections[payload.Collection]
			if collection == nil || collection.Deleted {
				return fmt.Errorf("collection %s not found during index delta replay", payload.Collection)
			}
			previous := collection.Records[payload.ID]
			replace := previous != nil && !previous.Deleted
			var previousOrdinal uint32
			if replace {
				previousOrdinal = previous.Ordinal
			}
			if err := e.applyRecordPut(payload, record.Header.LSN); err != nil {
				return err
			}
			if deltaProvider == nil || !deltaProvider.CanApplyIndexDeltas(payload.Collection, &collection.Config) {
				touchedCollections[payload.Collection] = struct{}{}
				continue
			}
			current := collection.Records[payload.ID]
			entry := &index.VectorEntry{
				ID:       payload.ID,
				Vector:   current.Vector,
				Metadata: current.Metadata,
				Version:  current.Version,
				Ordinal:  current.Ordinal,
			}
			if err := deltaProvider.ApplyIndexPut(payload.Collection, entry, replace, previousOrdinal, &collection.Config); err != nil {
				return fmt.Errorf("replay index put %s/%s at LSN %d: %w", payload.Collection, payload.ID, record.Header.LSN, err)
			}
			e.replayedIndexPuts++
		case recordTypeRecordDelete:
			payload, err := decodeRecordDeletePayloadBinary(record.Payload)
			if err != nil {
				return err
			}
			collection := e.state.Collections[payload.Collection]
			var (
				ordinal uint32
				present bool
			)
			if collection != nil && !collection.Deleted {
				current := collection.Records[payload.ID]
				if current != nil && !current.Deleted {
					ordinal = current.Ordinal
					present = true
				}
			}
			e.applyRecordDelete(payload.Collection, payload.ID, record.Header.LSN)
			if !present {
				continue
			}
			if deltaProvider == nil || !deltaProvider.CanApplyIndexDeltas(payload.Collection, &collection.Config) {
				touchedCollections[payload.Collection] = struct{}{}
				continue
			}
			if err := deltaProvider.ApplyIndexDelete(payload.Collection, payload.ID, ordinal, &collection.Config); err != nil {
				return fmt.Errorf("replay index delete %s/%s at LSN %d: %w", payload.Collection, payload.ID, record.Header.LSN, err)
			}
			e.replayedIndexDeletes++
		}
	}
	return nil
}

func (e *Engine) applyCreateCollection(name string, config storage.CollectionConfig, lsn uint64) {
	if collection := e.state.Collections[name]; collection != nil && !collection.Deleted {
		return
	}
	e.state.Collections[name] = &persistedCollection{
		ID:         e.state.NextCollectionID,
		Config:     config,
		CreatedLSN: lsn,
		UpdatedLSN: lsn,
		Records:    make(map[string]*recordValue),
	}
	e.state.NextCollectionID++
}

func (e *Engine) applyDeleteCollection(name string, lsn uint64) {
	if collection := e.state.Collections[name]; collection != nil {
		collection.Deleted = true
		collection.UpdatedLSN = lsn
		// Free all off-heap vector slots. Individual deallocation is
		// unnecessary: Free() releases all mmap'd slabs at once.
		if collection.vectorSFL != nil {
			collection.vectorSFL.Free()
			collection.vectorSFL = nil
			collection.vectorSlots = nil
		}
	}
}

func (e *Engine) applyRecordPut(payload recordPutPayload, lsn uint64) error {
	return e.applyRecordPutFields(payload.Collection, payload.ID, payload.Ordinal, payload.Vector, payload.Metadata, lsn, false)
}

func (e *Engine) applyRecordPutOwned(payload recordPutPayload, lsn uint64, adopt bool) error {
	return e.applyRecordPutFields(payload.Collection, payload.ID, payload.Ordinal, payload.Vector, payload.Metadata, lsn, adopt)
}

// sflMetadataOverhead is the minimum reserved bytes at the start of each
// ShardedFreeList slot. The memory package's SFL metadata occupies offsets
// 0–43 (Hyaline chain at 0/8/16/24/32, structIdx+shardIdx at 40); rounded
// up to the nearest 8-byte boundary: 48.
const sflMetadataOverhead = 48

// initVectorSFL lazily initializes the ShardedFreeList for off-heap vector storage.
func (c *persistedCollection) initVectorSFL() error {
	if c.vectorSFL != nil {
		return nil
	}
	// Slot must hold the vector data plus the SFL internal metadata prefix.
	slotSize := uint64(sflMetadataOverhead + c.Config.Dimension*4)
	slotSize = (slotSize + 7) &^ 7 // 8-byte alignment
	sfl, err := memory.NewShardedFreeList(memory.FreeListConfig{
		SlotSize:  slotSize,
		SlabSize:  2 * 1024 * 1024,
		SlabCount: 16,
	}, 64, 8)
	if err != nil {
		return fmt.Errorf("init vector SFL: %w", err)
	}
	c.vectorSFL = sfl
	c.vectorSlots = make([][]byte, nextOrdinalCapacity(0, 16))
	return nil
}

// storeVectorOffHeap allocates an SFL slot, copies vector data into it, and
// returns a []float32 view of the off-heap slot. frees any previous slot at ordinal.
func (c *persistedCollection) storeVectorOffHeap(ordinal uint32, vector []float32) ([]float32, error) {
	if len(vector) != c.Config.Dimension {
		return nil, fmt.Errorf("vector dimension %d != collection dimension %d", len(vector), c.Config.Dimension)
	}
	if err := c.initVectorSFL(); err != nil {
		return nil, err
	}
	// Grow vectorSlots if needed.
	if int(ordinal) >= len(c.vectorSlots) {
		grown := make([][]byte, nextOrdinalCapacity(len(c.vectorSlots), int(ordinal)+1))
		copy(grown, c.vectorSlots)
		c.vectorSlots = grown
	}
	// Free previous slot if replacing.
	if existing := c.vectorSlots[ordinal]; existing != nil {
		c.vectorSFL.Deallocate(existing)
		c.vectorSlots[ordinal] = nil
	}
	slot, err := c.vectorSFL.Allocate()
	if err != nil {
		return nil, fmt.Errorf("allocate vector slot: %w", err)
	}
	// Copy vector bytes after the SFL metadata prefix.
	data := slot[sflMetadataOverhead:]
	copy(data, unsafe.Slice((*byte)(unsafe.Pointer(&vector[0])), len(vector)*4))
	c.vectorSlots[ordinal] = slot
	return unsafe.Slice((*float32)(unsafe.Pointer(&data[0])), c.Config.Dimension), nil
}

// freeVectorSlot returns a vector's off-heap slot to the SFL.
func (c *persistedCollection) freeVectorSlot(ordinal uint32) {
	if c.vectorSFL == nil || int(ordinal) >= len(c.vectorSlots) {
		return
	}
	if slot := c.vectorSlots[ordinal]; slot != nil {
		c.vectorSFL.Deallocate(slot)
		c.vectorSlots[ordinal] = nil
	}
}

func (e *Engine) applyRecordPutFields(collectionName, id string, ordinal uint32, vector []float32, metadata map[string]interface{}, lsn uint64, adopt bool) error {
	collection := e.state.Collections[collectionName]
	if collection == nil || collection.Deleted {
		return fmt.Errorf("collection %s not found during replay", collectionName)
	}
	current := collection.Records[id]
	if current == nil {
		current = &recordValue{
			Version:    1,
			CreatedLSN: lsn,
			Ordinal:    ordinal,
		}
		collection.Records[id] = current
		collection.LiveCount++
	} else if current.Deleted {
		current.Deleted = false
		collection.LiveCount++
	} else {
		current.Version++
	}
	if next := ordinal + 1; next > collection.NextOrdinal {
		collection.NextOrdinal = next
	}
	ensureOrdinalSlot(collection, current.Ordinal, id)
	// Store vector off-heap via ShardedFreeList.
	owned, err := collection.storeVectorOffHeap(current.Ordinal, vector)
	if err != nil {
		return fmt.Errorf("store vector off-heap: %w", err)
	}
	current.Vector = owned
	if adopt {
		current.Metadata = metadata
	} else {
		current.Metadata = cloneMetadata(metadata)
	}
	current.UpdatedLSN = lsn
	collection.UpdatedLSN = lsn
	return nil
}

func ensureOrdinalSlot(collection *persistedCollection, ordinal uint32, id string) {
	if int(ordinal) >= len(collection.ordinalToID) {
		grown := make([]string, nextOrdinalCapacity(len(collection.ordinalToID), int(ordinal)+1))
		copy(grown, collection.ordinalToID)
		collection.ordinalToID = grown
	}
	collection.ordinalToID[ordinal] = id
}

func ensureOrdinalCapacity(collection *persistedCollection, minSize int) {
	if collection == nil || minSize <= len(collection.ordinalToID) {
		return
	}
	grown := make([]string, nextOrdinalCapacity(len(collection.ordinalToID), minSize))
	copy(grown, collection.ordinalToID)
	collection.ordinalToID = grown
}

func nextOrdinalCapacity(currentLen, minSize int) int {
	if minSize <= currentLen {
		return currentLen
	}
	newSize := currentLen
	if newSize < 16 {
		newSize = 16
	}
	for newSize < minSize {
		if newSize < 1024 {
			newSize *= 2
		} else {
			newSize += newSize / 2
		}
	}
	return newSize
}

func ensureRecordCapacity(collection *persistedCollection, additional int) {
	if collection == nil || additional <= 0 {
		return
	}
	if collection.Records == nil {
		collection.Records = make(map[string]*recordValue, additional)
		return
	}
	if len(collection.Records) != 0 {
		return
	}
	collection.Records = make(map[string]*recordValue, additional)
}

func (e *Engine) applyRecordDelete(collectionName, id string, lsn uint64) {
	collection := e.state.Collections[collectionName]
	if collection == nil || collection.Deleted {
		return
	}
	current := collection.Records[id]
	if current == nil || current.Deleted {
		return
	}
	current.Deleted = true
	current.UpdatedLSN = lsn
	collection.freeVectorSlot(current.Ordinal)
	if collection.LiveCount > 0 {
		collection.LiveCount--
	}
	if int(current.Ordinal) < len(collection.ordinalToID) {
		collection.ordinalToID[current.Ordinal] = ""
	}
	collection.UpdatedLSN = lsn
}

// encodeIndexBlock serializes all collection indexes into a single binary blob.
// Format: magic, version, collectionCount | repeated
// { nameLen, name, indexType, indexVersion, appliedLSN, payloadLen, payload, payloadChecksum }.
func encodeIndexBlock(entries []indexBlockEntry) []byte {
	size := 12 // magic uint32 + version/reserved uint16 + collectionCount uint32
	for _, e := range entries {
		size += 2 + len(e.name)        // nameLen uint16 + name bytes
		size += 1 + 2 + 8              // indexType uint8 + indexVersion uint16 + appliedLSN uint64
		size += 4 + len(e.payload) + 4 // payloadLen uint32 + payload bytes + payloadChecksum uint32
	}
	buf := make([]byte, 0, size)
	buf = binary.LittleEndian.AppendUint32(buf, indexBlockMagic)
	buf = binary.LittleEndian.AppendUint16(buf, indexBlockVersion)
	buf = binary.LittleEndian.AppendUint16(buf, 0)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(entries)))
	for _, e := range entries {
		buf = binary.LittleEndian.AppendUint16(buf, uint16(len(e.name)))
		buf = append(buf, []byte(e.name)...)
		buf = append(buf, e.indexType)
		buf = binary.LittleEndian.AppendUint16(buf, e.indexVersion)
		buf = binary.LittleEndian.AppendUint64(buf, e.appliedLSN)
		buf = binary.LittleEndian.AppendUint32(buf, uint32(len(e.payload)))
		buf = append(buf, e.payload...)
		buf = binary.LittleEndian.AppendUint32(buf, e.payloadChecksum)
	}
	return buf
}

// decodeIndexBlock deserializes the index block. Returns entries for valid
// collections; entries with mismatched checksums are skipped (caller rebuilds).
func decodeIndexBlock(data []byte) ([]indexBlockEntry, error) {
	if len(data) < 4 {
		return nil, fmt.Errorf("index block too small")
	}
	versioned := binary.LittleEndian.Uint32(data) == indexBlockMagic
	count := binary.LittleEndian.Uint32(data)
	pos := 4
	if versioned {
		if len(data) < 12 {
			return nil, fmt.Errorf("versioned index block too small")
		}
		version := binary.LittleEndian.Uint16(data[4:6])
		if version != indexBlockVersion {
			return nil, fmt.Errorf("unsupported index block version %d", version)
		}
		count = binary.LittleEndian.Uint32(data[8:12])
		pos = 12
	}
	minEntrySize := 13 // empty name and payload in the legacy layout
	if versioned {
		minEntrySize += 8
	}
	if uint64(count) > uint64((len(data)-pos)/minEntrySize) {
		return nil, fmt.Errorf("index block entry count %d exceeds payload capacity", count)
	}
	entries := make([]indexBlockEntry, 0, count)
	for i := uint32(0); i < count; i++ {
		if pos+2 > len(data) {
			return nil, fmt.Errorf("truncated index block: expected entry %d name length at offset %d", i, pos)
		}
		nameLen := int(binary.LittleEndian.Uint16(data[pos:]))
		pos += 2
		if pos+nameLen > len(data) {
			return nil, fmt.Errorf("truncated index block: entry %d name extends past end at offset %d", i, pos)
		}
		name := string(data[pos : pos+nameLen])
		pos += nameLen
		headerSize := 7
		if versioned {
			headerSize += 8
		}
		if pos+headerSize > len(data) {
			return nil, fmt.Errorf("truncated index block: entry %d (%s) header cut at offset %d", i, name, pos)
		}
		indexType := data[pos]
		indexVersion := binary.LittleEndian.Uint16(data[pos+1:])
		pos += 3
		var appliedLSN uint64
		if versioned {
			appliedLSN = binary.LittleEndian.Uint64(data[pos:])
			pos += 8
		}
		payloadLen := int(binary.LittleEndian.Uint32(data[pos:]))
		pos += 4
		if pos+payloadLen+4 > len(data) {
			return nil, fmt.Errorf("truncated index entry for collection %s", name)
		}
		if payloadLen > maxIndexEntrySize {
			return nil, fmt.Errorf("index entry size %d exceeds limit %d", payloadLen, maxIndexEntrySize)
		}
		payload := make([]byte, payloadLen)
		copy(payload, data[pos:pos+payloadLen])
		pos += payloadLen
		payloadChecksum := binary.LittleEndian.Uint32(data[pos:])
		pos += 4
		entries = append(entries, indexBlockEntry{
			name:            name,
			indexType:       indexType,
			indexVersion:    indexVersion,
			appliedLSN:      appliedLSN,
			hasAppliedLSN:   versioned,
			payload:         payload,
			payloadChecksum: payloadChecksum,
		})
	}
	if pos != len(data) {
		return nil, fmt.Errorf("index block has %d trailing bytes", len(data)-pos)
	}
	return entries, nil
}

func (e *Engine) serializeIndexEntry(name string, checkpointLSN uint64) (indexBlockEntry, bool, error) {
	var (
		indexBytes []byte
		appliedLSN = checkpointLSN
		err        error
	)
	if provider, ok := e.indexProvider.(CoordinatedIndexSnapshotProvider); ok {
		indexBytes, appliedLSN, err = provider.SerializeIndexAt(name, checkpointLSN)
	} else {
		indexBytes, err = e.indexProvider.SerializeIndex(name)
	}
	if err != nil {
		return indexBlockEntry{}, false, err
	}
	if indexBytes == nil {
		return indexBlockEntry{}, false, nil
	}
	indexType, indexVersion := e.indexProvider.IndexTypeVersion(name)
	return indexBlockEntry{
		name:            name,
		indexType:       indexType,
		indexVersion:    indexVersion,
		appliedLSN:      appliedLSN,
		hasAppliedLSN:   true,
		payload:         indexBytes,
		payloadChecksum: crc32.Checksum(indexBytes, castagnoli),
	}, true, nil
}

func (e *Engine) checkpointLocked() error {
	if !e.dirty {
		return nil
	}
	// STEP 1: write snapshot chunk
	snapshot, err := encodeStateBinary(e.state)
	if err != nil {
		return fmt.Errorf("marshal snapshot: %w", err)
	}
	snapshotOffset, err := e.appendChunkLocked(chunkTypeSnapshot, snapshot)
	if err != nil {
		return err
	}

	// STEP 2: serialize and write index chunk (after snapshot, before metapage)
	var indexBlock []byte
	var indexChecksum uint32
	if e.indexProvider != nil {
		names := make([]string, 0, len(e.state.Collections))
		for name := range e.state.Collections {
			names = append(names, name)
		}
		sort.Strings(names)
		entries := make([]indexBlockEntry, 0, len(names))
		for _, name := range names {
			entry, present, err := e.serializeIndexEntry(name, e.lastLSN)
			if err != nil {
				return fmt.Errorf("serialize index for %s: %w", name, err)
			}
			if !present {
				continue // empty collection (no index to persist); rebuilt from Records on recovery
			}
			entries = append(entries, entry)
		}
		if len(entries) > 0 {
			indexBlock = encodeIndexBlock(entries)
			indexChecksum = crc32.Checksum(indexBlock, castagnoli)
		}
	}
	var indexOffset uint64
	var indexLength uint64
	if len(indexBlock) > 0 {
		indexOffset, err = e.appendChunkLocked(chunkTypeIndex, indexBlock)
		if err != nil {
			return err
		}
		indexLength = uint64(len(indexBlock))
	}

	// STEP 3: write metapage (now authoritative: snapshot + index)
	e.metaEpoch++
	nextMetaPage := uint64(1)
	rootFreelist := uint64(0)
	if e.activeMetaPage == 1 {
		nextMetaPage = 2
		rootFreelist = math.MaxUint64
	}

	stat, err := e.file.Stat()
	if err != nil {
		return err
	}
	pageCount := uint64((stat.Size() + pageSize - 1) / pageSize)

	meta := &metaPage{
		Magic:           metaMagicV2,
		MetaEpoch:       e.metaEpoch,
		RootCatalog:     3,
		RootFreelist:    rootFreelist,
		LastAppliedLSN:  e.lastLSN,
		PageCount:       pageCount,
		CollectionCount: uint64(e.visibleCollectionCountLocked()),
		SnapshotOffset:  snapshotOffset,
		SnapshotLength:  uint64(len(snapshot)),
		IndexOffset:     indexOffset,
		IndexLength:     indexLength,
		IndexChecksum:   indexChecksum,
	}
	bufPtr := pagePool.Get().(*[]byte)
	buf := *bufPtr
	defer pagePool.Put(bufPtr)

	for i := range buf {
		buf[i] = 0
	}
	// STEP 4: fsync (durable: snapshot + index + metapage)
	if err := writeFixedPage(e.file, nextMetaPage, encodeMeta(meta, buf)); err != nil {
		return err
	}
	if err := e.file.Sync(); err != nil {
		return err
	}

	// STEP 5: write header (publishes the new metapage as authoritative)
	header, err := e.readHeader()
	if err != nil {
		return err
	}
	header.LastCheckpointLSN = e.lastLSN
	header.ActiveMetaPage = nextMetaPage
	header.WALHeadPage = pageCount

	for i := range buf {
		buf[i] = 0
	}
	if err := writeFixedPage(e.file, 0, encodeHeader(header, buf)); err != nil {
		return err
	}
	if err := e.file.Sync(); err != nil {
		return err
	}

	e.activeMetaPage = nextMetaPage
	e.dirty = false
	e.dirtyBytes = 0
	e.dirtyOps = 0
	e.checkpoints++

	// Auto-compact when WAL bloat exceeds 2× the minimum file size.
	compactSize := int64(3*pageSize) + 16 + int64(len(snapshot)) + 16 + int64(len(indexBlock))
	if stat.Size() > compactSize*2 {
		if err := e.compactFile(); err != nil {
			log.Printf("singlefile: auto-compact failed: %v", err)
		}
	}
	return nil
}

func captureState(e *Engine) *persistedState {
	cloned := &persistedState{
		NextCollectionID: e.state.NextCollectionID,
		Collections:      make(map[string]*persistedCollection, len(e.state.Collections)),
	}
	for name, coll := range e.state.Collections {
		c := &persistedCollection{
			ID:          coll.ID,
			Config:      coll.Config,
			CreatedLSN:  coll.CreatedLSN,
			UpdatedLSN:  coll.UpdatedLSN,
			LiveCount:   coll.LiveCount,
			NextOrdinal: coll.NextOrdinal,
			Deleted:     coll.Deleted,
			Records:     make(map[string]*recordValue, len(coll.Records)),
		}
		for id, rec := range coll.Records {
			r := &recordValue{
				Vector:     append([]float32(nil), rec.Vector...),
				Version:    rec.Version,
				CreatedLSN: rec.CreatedLSN,
				UpdatedLSN: rec.UpdatedLSN,
				Ordinal:    rec.Ordinal,
				Deleted:    rec.Deleted,
			}
			if rec.Metadata != nil {
				r.Metadata = make(map[string]interface{}, len(rec.Metadata))
				for k, v := range rec.Metadata {
					r.Metadata[k] = v
				}
			}
			c.Records[id] = r
		}
		cloned.Collections[name] = c
	}
	return cloned
}

// Vacuum reclaims disk space by writing a compacted database to a temporary file
// and atomically replacing the active file. It minimizes lock contention by
// performing heavy serialization unlocked, then copying only the appended WAL
// deltas during a brief final lock.
func (e *Engine) Vacuum(ctx context.Context) error {
	// Phase 1: flush + snapshot (brief lock)
	e.mu.Lock()
	if e.closed.Load() {
		e.mu.Unlock()
		return fmt.Errorf("engine closed")
	}
	if err := e.checkpointLocked(); err != nil {
		e.mu.Unlock()
		return fmt.Errorf("vacuum pre-checkpoint: %w", err)
	}

	// Snapshot vectors for indexes safely
	if e.indexProvider != nil {
		if err := e.indexProvider.SnapshotVectors(ctx); err != nil {
			e.mu.Unlock()
			return fmt.Errorf("vacuum snapshot vectors: %w", err)
		}
	}

	stat, err := e.file.Stat()
	if err != nil {
		e.mu.Unlock()
		return fmt.Errorf("vacuum stat: %w", err)
	}
	phase1Size := stat.Size()
	snapshotState := captureState(e)
	origHeader, err := e.readHeader()
	if err != nil {
		e.mu.Unlock()
		return fmt.Errorf("vacuum read header: %w", err)
	}
	phase1LSN := e.lastLSN
	e.mu.Unlock()

	// Phase 2: Serialization and writing temp file (Unlocked)
	snapshotBytes, err := encodeStateBinary(snapshotState)
	if err != nil {
		return fmt.Errorf("vacuum encode state: %w", err)
	}
	if len(snapshotBytes) > maxChunkSize {
		return fmt.Errorf("vacuum snapshot size %d exceeds limit %d", len(snapshotBytes), maxChunkSize)
	}

	var indexBlock []byte
	var indexChecksum uint32
	if e.indexProvider != nil {
		names := make([]string, 0, len(snapshotState.Collections))
		for name := range snapshotState.Collections {
			names = append(names, name)
		}
		sort.Strings(names)
		entries := make([]indexBlockEntry, 0, len(names))
		for _, name := range names {
			entry, present, err := e.serializeIndexEntry(name, phase1LSN)
			if err != nil {
				return fmt.Errorf("vacuum serialize index %s: %w", name, err)
			}
			if !present {
				continue
			}
			entries = append(entries, entry)
		}
		if len(entries) > 0 {
			indexBlock = encodeIndexBlock(entries)
			indexChecksum = crc32.Checksum(indexBlock, castagnoli)
		}
	}

	tmpPath := e.path + ".vacuum"
	tmpFile, err := os.OpenFile(tmpPath, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return fmt.Errorf("vacuum create temp file: %w", err)
	}

	cleanup := true
	defer func() {
		if cleanup {
			tmpFile.Close()
			os.Remove(tmpPath)
		}
	}()

	const snapshotOffset = uint64(3 * pageSize)
	indexOffset := snapshotOffset + 16 + uint64(len(snapshotBytes))
	indexLength := uint64(len(indexBlock))
	totalSize := int64(indexOffset)
	if len(indexBlock) > 0 {
		totalSize += 16 + int64(indexLength)
	}
	pageCount := uint64((totalSize + pageSize - 1) / pageSize)

	fh := &fileHeader{
		FormatVersion:     origHeader.FormatVersion,
		PageSize:          origHeader.PageSize,
		FeatureFlags:      origHeader.FeatureFlags,
		FileID:            origHeader.FileID,
		CreatedUnixNano:   origHeader.CreatedUnixNano,
		Creator:           origHeader.Creator,
		LastCheckpointLSN: phase1LSN,
		ActiveMetaPage:    1,
		WALStartPage:      pageCount,
		WALHeadPage:       pageCount,
		WALTailPage:       pageCount,
	}
	copy(fh.Magic[:], origHeader.Magic[:])

	bufPtr := pagePool.Get().(*[]byte)
	buf := *bufPtr
	for i := range buf {
		buf[i] = 0
	}
	if err := writeFixedPage(tmpFile, 0, encodeHeader(fh, buf)); err != nil {
		pagePool.Put(bufPtr)
		return fmt.Errorf("vacuum write header: %w", err)
	}

	meta := &metaPage{
		Magic:           metaMagicV2,
		MetaEpoch:       1,
		RootCatalog:     3,
		RootFreelist:    0,
		LastAppliedLSN:  phase1LSN,
		PageCount:       pageCount,
		CollectionCount: uint64(len(snapshotState.Collections)),
		SnapshotOffset:  snapshotOffset,
		SnapshotLength:  uint64(len(snapshotBytes)),
		IndexOffset:     indexOffset,
		IndexLength:     indexLength,
		IndexChecksum:   indexChecksum,
	}
	for i := range buf {
		buf[i] = 0
	}
	if err := writeFixedPage(tmpFile, 1, encodeMeta(meta, buf)); err != nil {
		pagePool.Put(bufPtr)
		return fmt.Errorf("vacuum write meta A: %w", err)
	}

	meta.RootFreelist = math.MaxUint64
	for i := range buf {
		buf[i] = 0
	}
	if err := writeFixedPage(tmpFile, 2, encodeMeta(meta, buf)); err != nil {
		pagePool.Put(bufPtr)
		return fmt.Errorf("vacuum write meta B: %w", err)
	}
	pagePool.Put(bufPtr)

	var chunkHdr [16]byte
	binary.LittleEndian.PutUint32(chunkHdr[0:4], chunkMagic)
	binary.LittleEndian.PutUint16(chunkHdr[4:6], chunkTypeSnapshot)
	binary.LittleEndian.PutUint16(chunkHdr[6:8], formatVersion)
	binary.LittleEndian.PutUint32(chunkHdr[8:12], uint32(len(snapshotBytes)))
	binary.LittleEndian.PutUint32(chunkHdr[12:16], crc32.Checksum(snapshotBytes, castagnoli))
	if _, err := tmpFile.WriteAt(chunkHdr[:], int64(snapshotOffset)); err != nil {
		return fmt.Errorf("vacuum write chunk header: %w", err)
	}
	if _, err := tmpFile.WriteAt(snapshotBytes, int64(snapshotOffset)+16); err != nil {
		return fmt.Errorf("vacuum write snapshot payload: %w", err)
	}

	if len(indexBlock) > 0 {
		var idxHdr [16]byte
		binary.LittleEndian.PutUint32(idxHdr[0:4], chunkMagic)
		binary.LittleEndian.PutUint16(idxHdr[4:6], chunkTypeIndex)
		binary.LittleEndian.PutUint16(idxHdr[6:8], formatVersion)
		binary.LittleEndian.PutUint32(idxHdr[8:12], uint32(len(indexBlock)))
		binary.LittleEndian.PutUint32(idxHdr[12:16], indexChecksum)
		if _, err := tmpFile.WriteAt(idxHdr[:], int64(indexOffset)); err != nil {
			return fmt.Errorf("vacuum write index chunk header: %w", err)
		}
		if _, err := tmpFile.WriteAt(indexBlock, int64(indexOffset)+16); err != nil {
			return fmt.Errorf("vacuum write index payload: %w", err)
		}
	}

	// Phase 3: Catch-up & Swap (Brief Lock)
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed.Load() {
		return fmt.Errorf("engine closed during vacuum")
	}

	// Flush any in-flight writes
	if err := e.checkpointLocked(); err != nil {
		return fmt.Errorf("vacuum final checkpoint: %w", err)
	}

	stat, err = e.file.Stat()
	if err != nil {
		return fmt.Errorf("vacuum final stat: %w", err)
	}
	currentSize := stat.Size()

	if currentSize > phase1Size {
		// Copy WAL bytes that landed during Phase 2
		// Seek temp file to end (should be at totalSize)
		if _, err := tmpFile.Seek(0, io.SeekEnd); err != nil {
			return fmt.Errorf("vacuum seek temp file: %w", err)
		}

		// Create section reader for the delta
		deltaReader := io.NewSectionReader(e.file, phase1Size, currentSize-phase1Size)
		if _, err := io.Copy(tmpFile, deltaReader); err != nil {
			return fmt.Errorf("vacuum copy WAL deltas: %w", err)
		}
	}

	if err := tmpFile.Sync(); err != nil {
		return fmt.Errorf("vacuum final sync: %w", err)
	}
	if err := tmpFile.Close(); err != nil {
		return fmt.Errorf("vacuum close temp: %w", err)
	}

	if err := e.file.Close(); err != nil {
		// reopen guard
		if f, ferr := os.OpenFile(e.path, os.O_RDWR, 0644); ferr != nil {
			e.status.Store(int32(storage.StatusFailed))
			return fmt.Errorf("vacuum close original: %w; reopen: %w", err, ferr)
		} else {
			e.file = f
		}
		return fmt.Errorf("vacuum close original: %w", err)
	}

	e.file = nil // Safety before rename
	if err := replaceDatabaseFile(tmpPath, e.path); err != nil {
		if f, ferr := os.OpenFile(e.path, os.O_RDWR, 0644); ferr != nil {
			e.status.Store(int32(storage.StatusFailed))
			return fmt.Errorf("vacuum rename: %w; reopen: %w", err, ferr)
		} else {
			e.file = f
		}
		return fmt.Errorf("vacuum rename: %w", err)
	}

	f, err := os.OpenFile(e.path, os.O_RDWR, 0644)
	if err != nil {
		e.status.Store(int32(storage.StatusFailed))
		return fmt.Errorf("vacuum open new file: %w", err)
	}
	e.file = f
	e.dirty = false
	cleanup = false // Successfully swapped, defer won't remove it
	if err := syncDatabaseParent(e.path); err != nil {
		return fmt.Errorf("vacuum sync parent directory: %w", err)
	}
	return nil
}

// Backup creates a point-in-time copy of the active database to destPath.
// It uses the same non-blocking fast-forward logic as Vacuum.
func (e *Engine) Backup(ctx context.Context, destPath string) error {
	// Phase 1: flush + snapshot (brief lock)
	e.mu.Lock()
	if e.closed.Load() {
		e.mu.Unlock()
		return fmt.Errorf("engine closed")
	}
	if err := e.checkpointLocked(); err != nil {
		e.mu.Unlock()
		return fmt.Errorf("backup pre-checkpoint: %w", err)
	}

	if e.indexProvider != nil {
		if err := e.indexProvider.SnapshotVectors(ctx); err != nil {
			e.mu.Unlock()
			return fmt.Errorf("backup snapshot vectors: %w", err)
		}
	}

	stat, err := e.file.Stat()
	if err != nil {
		e.mu.Unlock()
		return fmt.Errorf("backup stat: %w", err)
	}
	phase1Size := stat.Size()
	snapshotState := captureState(e)
	origHeader, err := e.readHeader()
	if err != nil {
		e.mu.Unlock()
		return fmt.Errorf("backup read header: %w", err)
	}
	phase1LSN := e.lastLSN
	e.mu.Unlock()

	// Phase 2: Serialization and writing backup file (Unlocked)
	snapshotBytes, err := encodeStateBinary(snapshotState)
	if err != nil {
		return fmt.Errorf("backup encode state: %w", err)
	}
	if len(snapshotBytes) > maxChunkSize {
		return fmt.Errorf("backup snapshot size %d exceeds limit %d", len(snapshotBytes), maxChunkSize)
	}

	var indexBlock []byte
	var indexChecksum uint32
	if e.indexProvider != nil {
		names := make([]string, 0, len(snapshotState.Collections))
		for name := range snapshotState.Collections {
			names = append(names, name)
		}
		sort.Strings(names)
		entries := make([]indexBlockEntry, 0, len(names))
		for _, name := range names {
			entry, present, err := e.serializeIndexEntry(name, phase1LSN)
			if err != nil {
				return fmt.Errorf("backup serialize index %s: %w", name, err)
			}
			if !present {
				continue
			}
			entries = append(entries, entry)
		}
		if len(entries) > 0 {
			indexBlock = encodeIndexBlock(entries)
			indexChecksum = crc32.Checksum(indexBlock, castagnoli)
		}
	}

	destFile, err := os.OpenFile(destPath, os.O_RDWR|os.O_CREATE|os.O_EXCL, 0644)
	if err != nil {
		return fmt.Errorf("backup create file: %w", err)
	}

	cleanup := true
	defer func() {
		if cleanup {
			destFile.Close()
			os.Remove(destPath)
		}
	}()

	const snapshotOffset = uint64(3 * pageSize)
	indexOffset := snapshotOffset + 16 + uint64(len(snapshotBytes))
	indexLength := uint64(len(indexBlock))
	totalSize := int64(indexOffset)
	if len(indexBlock) > 0 {
		totalSize += 16 + int64(indexLength)
	}
	pageCount := uint64((totalSize + pageSize - 1) / pageSize)

	fh := &fileHeader{
		FormatVersion:     origHeader.FormatVersion,
		PageSize:          origHeader.PageSize,
		FeatureFlags:      origHeader.FeatureFlags,
		FileID:            origHeader.FileID,
		CreatedUnixNano:   origHeader.CreatedUnixNano,
		Creator:           origHeader.Creator,
		LastCheckpointLSN: phase1LSN,
		ActiveMetaPage:    1,
		WALStartPage:      pageCount,
		WALHeadPage:       pageCount,
		WALTailPage:       pageCount,
	}
	copy(fh.Magic[:], origHeader.Magic[:])

	bufPtr := pagePool.Get().(*[]byte)
	buf := *bufPtr
	for i := range buf {
		buf[i] = 0
	}
	if err := writeFixedPage(destFile, 0, encodeHeader(fh, buf)); err != nil {
		pagePool.Put(bufPtr)
		return fmt.Errorf("backup write header: %w", err)
	}

	meta := &metaPage{
		Magic:           metaMagicV2,
		MetaEpoch:       1,
		RootCatalog:     3,
		RootFreelist:    0,
		LastAppliedLSN:  phase1LSN,
		PageCount:       pageCount,
		CollectionCount: uint64(len(snapshotState.Collections)),
		SnapshotOffset:  snapshotOffset,
		SnapshotLength:  uint64(len(snapshotBytes)),
		IndexOffset:     indexOffset,
		IndexLength:     indexLength,
		IndexChecksum:   indexChecksum,
	}
	for i := range buf {
		buf[i] = 0
	}
	if err := writeFixedPage(destFile, 1, encodeMeta(meta, buf)); err != nil {
		pagePool.Put(bufPtr)
		return fmt.Errorf("backup write meta A: %w", err)
	}

	meta.RootFreelist = math.MaxUint64
	for i := range buf {
		buf[i] = 0
	}
	if err := writeFixedPage(destFile, 2, encodeMeta(meta, buf)); err != nil {
		pagePool.Put(bufPtr)
		return fmt.Errorf("backup write meta B: %w", err)
	}
	pagePool.Put(bufPtr)

	var chunkHdr [16]byte
	binary.LittleEndian.PutUint32(chunkHdr[0:4], chunkMagic)
	binary.LittleEndian.PutUint16(chunkHdr[4:6], chunkTypeSnapshot)
	binary.LittleEndian.PutUint16(chunkHdr[6:8], formatVersion)
	binary.LittleEndian.PutUint32(chunkHdr[8:12], uint32(len(snapshotBytes)))
	binary.LittleEndian.PutUint32(chunkHdr[12:16], crc32.Checksum(snapshotBytes, castagnoli))
	if _, err := destFile.WriteAt(chunkHdr[:], int64(snapshotOffset)); err != nil {
		return fmt.Errorf("backup write chunk header: %w", err)
	}
	if _, err := destFile.WriteAt(snapshotBytes, int64(snapshotOffset)+16); err != nil {
		return fmt.Errorf("backup write snapshot payload: %w", err)
	}

	if len(indexBlock) > 0 {
		var idxHdr [16]byte
		binary.LittleEndian.PutUint32(idxHdr[0:4], chunkMagic)
		binary.LittleEndian.PutUint16(idxHdr[4:6], chunkTypeIndex)
		binary.LittleEndian.PutUint16(idxHdr[6:8], formatVersion)
		binary.LittleEndian.PutUint32(idxHdr[8:12], uint32(len(indexBlock)))
		binary.LittleEndian.PutUint32(idxHdr[12:16], indexChecksum)
		if _, err := destFile.WriteAt(idxHdr[:], int64(indexOffset)); err != nil {
			return fmt.Errorf("backup write index chunk header: %w", err)
		}
		if _, err := destFile.WriteAt(indexBlock, int64(indexOffset)+16); err != nil {
			return fmt.Errorf("backup write index payload: %w", err)
		}
	}

	// Phase 3: Catch-up (Brief Lock)
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed.Load() {
		return fmt.Errorf("engine closed during backup")
	}

	if err := e.checkpointLocked(); err != nil {
		return fmt.Errorf("backup final checkpoint: %w", err)
	}

	stat, err = e.file.Stat()
	if err != nil {
		return fmt.Errorf("backup final stat: %w", err)
	}
	currentSize := stat.Size()

	if currentSize > phase1Size {
		if _, err := destFile.Seek(0, io.SeekEnd); err != nil {
			return fmt.Errorf("backup seek temp file: %w", err)
		}
		deltaReader := io.NewSectionReader(e.file, phase1Size, currentSize-phase1Size)
		if _, err := io.Copy(destFile, deltaReader); err != nil {
			return fmt.Errorf("backup copy WAL deltas: %w", err)
		}
	}

	if err := destFile.Sync(); err != nil {
		return fmt.Errorf("backup final sync: %w", err)
	}
	if err := destFile.Close(); err != nil {
		return fmt.Errorf("backup close dest: %w", err)
	}
	if err := syncDatabaseParent(destPath); err != nil {
		return fmt.Errorf("backup sync parent directory: %w", err)
	}

	cleanup = false
	return nil
}

// Drop destroys the storage engine by closing its file descriptor and removing it
// from the filesystem.
func (e *Engine) Drop(ctx context.Context) error {
	e.mu.Lock()
	if e.closed.Load() {
		e.mu.Unlock()
		return fmt.Errorf("database already closed")
	}
	e.mu.Unlock()

	if err := e.Close(); err != nil {
		return fmt.Errorf("drop close: %w", err)
	}

	if err := os.Remove(e.path); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("drop remove: %w", err)
	}
	if err := syncDatabaseParent(e.path); err != nil {
		return fmt.Errorf("drop sync parent directory: %w", err)
	}

	return nil
}

// compactFile rewrites the database into a fresh temp file containing only the
// current snapshot, then atomically renames it over the original. Caller must
// hold e.mu. The checkpoint that precedes this call must already be fsynced.
func (e *Engine) compactFile() error {
	err := e.compactFileLocked()
	if err != nil {
		e.compactionErrors++
	}
	return err
}

// compactFileLocked implements the actual compaction logic. The caller
// (compactFile) wraps it to increment compactionErrors on failure.
func (e *Engine) compactFileLocked() error {
	snapshot, err := encodeStateBinary(e.state)
	if err != nil {
		return fmt.Errorf("compact: encode state: %w", err)
	}
	if len(snapshot) > maxChunkSize {
		return fmt.Errorf("compact: snapshot size %d exceeds limit %d", len(snapshot), maxChunkSize)
	}

	// Preserve FileID and CreatedUnixNano from the live header.
	origHeader, err := e.readHeader()
	if err != nil {
		return fmt.Errorf("compact: read header: %w", err)
	}

	// Build index block (same as checkpointLocked).
	var indexBlock []byte
	var indexChecksum uint32
	if e.indexProvider != nil {
		names := make([]string, 0, len(e.state.Collections))
		for name := range e.state.Collections {
			names = append(names, name)
		}
		sort.Strings(names)
		entries := make([]indexBlockEntry, 0, len(names))
		for _, name := range names {
			entry, present, err := e.serializeIndexEntry(name, e.lastLSN)
			if err != nil {
				return fmt.Errorf("compact: serialize index for %s: %w", name, err)
			}
			if !present {
				continue
			}
			entries = append(entries, entry)
		}
		if len(entries) > 0 {
			indexBlock = encodeIndexBlock(entries)
			indexChecksum = crc32.Checksum(indexBlock, castagnoli)
		}
	}

	tmpPath := e.path + ".compact"
	tmpFile, err := os.OpenFile(tmpPath, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return fmt.Errorf("compact: create temp file: %w", err)
	}
	ok := false
	defer func() {
		if !ok {
			tmpFile.Close()
			os.Remove(tmpPath)
		}
	}()

	// ── Calculate layout ───────────────────────────────────────────────────
	const snapshotOffset = uint64(3 * pageSize) // 12288
	indexOffset := snapshotOffset + 16 + uint64(len(snapshot))
	indexLength := uint64(len(indexBlock))
	totalSize := int64(indexOffset)
	if len(indexBlock) > 0 {
		totalSize += 16 + int64(indexLength)
	}
	pageCount := uint64((totalSize + pageSize - 1) / pageSize)

	// ── Page 0: file header ────────────────────────────────────────────────
	fh := &fileHeader{
		FormatVersion:     origHeader.FormatVersion,
		PageSize:          origHeader.PageSize,
		FeatureFlags:      origHeader.FeatureFlags,
		FileID:            origHeader.FileID,
		CreatedUnixNano:   origHeader.CreatedUnixNano,
		Creator:           origHeader.Creator,
		LastCheckpointLSN: e.lastLSN,
		ActiveMetaPage:    1,
		WALStartPage:      pageCount, // no WAL in compacted file
		WALHeadPage:       pageCount,
		WALTailPage:       pageCount,
	}
	copy(fh.Magic[:], origHeader.Magic[:])

	bufPtr := pagePool.Get().(*[]byte)
	buf := *bufPtr
	defer pagePool.Put(bufPtr)
	for i := range buf {
		buf[i] = 0
	}

	if err := writeFixedPage(tmpFile, 0, encodeHeader(fh, buf)); err != nil {
		return fmt.Errorf("compact: write header: %w", err)
	}

	// ── Pages 1 & 2: dual metapages (V2 with index fields) ────────────────
	meta := &metaPage{
		Magic:           metaMagicV2,
		MetaEpoch:       1,
		RootCatalog:     3,
		RootFreelist:    0, // meta A marker
		LastAppliedLSN:  e.lastLSN,
		PageCount:       pageCount,
		CollectionCount: uint64(e.visibleCollectionCountLocked()),
		SnapshotOffset:  snapshotOffset,
		SnapshotLength:  uint64(len(snapshot)),
		IndexOffset:     indexOffset,
		IndexLength:     indexLength,
		IndexChecksum:   indexChecksum,
	}
	if len(indexBlock) > maxChunkSize {
		return fmt.Errorf("compact: index block size %d exceeds limit %d", len(indexBlock), maxChunkSize)
	}

	for i := range buf {
		buf[i] = 0
	}
	if err := writeFixedPage(tmpFile, 1, encodeMeta(meta, buf)); err != nil {
		return fmt.Errorf("compact: write meta A: %w", err)
	}
	meta.RootFreelist = math.MaxUint64 // meta B marker
	for i := range buf {
		buf[i] = 0
	}
	if err := writeFixedPage(tmpFile, 2, encodeMeta(meta, buf)); err != nil {
		return fmt.Errorf("compact: write meta B: %w", err)
	}

	// ── Page 3 (offset 12288): snapshot chunk ─────────────────────────────
	checksum := crc32.Checksum(snapshot, castagnoli)
	var chunkHdr [16]byte
	binary.LittleEndian.PutUint32(chunkHdr[0:4], chunkMagic)
	binary.LittleEndian.PutUint16(chunkHdr[4:6], chunkTypeSnapshot)
	binary.LittleEndian.PutUint16(chunkHdr[6:8], formatVersion)
	binary.LittleEndian.PutUint32(chunkHdr[8:12], uint32(len(snapshot)))
	binary.LittleEndian.PutUint32(chunkHdr[12:16], checksum)
	if _, err := tmpFile.WriteAt(chunkHdr[:], int64(snapshotOffset)); err != nil {
		return fmt.Errorf("compact: write chunk header: %w", err)
	}
	if _, err := tmpFile.WriteAt(snapshot, int64(snapshotOffset)+16); err != nil {
		return fmt.Errorf("compact: write snapshot payload: %w", err)
	}

	// ── Index chunk (if present) ──────────────────────────────────────────
	if len(indexBlock) > 0 {
		var idxHdr [16]byte
		binary.LittleEndian.PutUint32(idxHdr[0:4], chunkMagic)
		binary.LittleEndian.PutUint16(idxHdr[4:6], chunkTypeIndex)
		binary.LittleEndian.PutUint16(idxHdr[6:8], formatVersion)
		binary.LittleEndian.PutUint32(idxHdr[8:12], uint32(len(indexBlock)))
		binary.LittleEndian.PutUint32(idxHdr[12:16], indexChecksum)
		if _, err := tmpFile.WriteAt(idxHdr[:], int64(indexOffset)); err != nil {
			return fmt.Errorf("compact: write index chunk header: %w", err)
		}
		if _, err := tmpFile.WriteAt(indexBlock, int64(indexOffset)+16); err != nil {
			return fmt.Errorf("compact: write index block payload: %w", err)
		}
	}

	// ── Fsync before rename ───────────────────────────────────────────────
	if err := tmpFile.Sync(); err != nil {
		return fmt.Errorf("compact: sync: %w", err)
	}
	if err := tmpFile.Close(); err != nil {
		return fmt.Errorf("compact: close temp: %w", err)
	}

	// ── Atomic rename ─────────────────────────────────────────────────────
	// Close original before rename; if anything fails, reopen original.
	if err := e.file.Close(); err != nil {
		if f, ferr := os.OpenFile(e.path, os.O_RDWR, 0644); ferr != nil {
			e.status.Store(int32(storage.StatusFailed))
			return fmt.Errorf("compact: close original: %w; reopen: %w", err, ferr)
		} else {
			e.file = f
		}
		return fmt.Errorf("compact: close original: %w", err)
	}
	e.file = nil

	if err := replaceDatabaseFile(tmpPath, e.path); err != nil {
		if f, ferr := os.OpenFile(e.path, os.O_RDWR, 0644); ferr != nil {
			e.status.Store(int32(storage.StatusFailed))
			return fmt.Errorf("compact: rename: %w; reopen: %w", err, ferr)
		} else {
			e.file = f
		}
		return fmt.Errorf("compact: rename: %w", err)
	}

	newFile, err := os.OpenFile(e.path, os.O_RDWR, 0644)
	if err != nil {
		return fmt.Errorf("compact: reopen: %w", err)
	}
	e.file = newFile

	// ── Reset bookkeeping to match the compacted file ─────────────────────
	e.activeMetaPage = 1
	e.metaEpoch = 1
	e.dirty = false
	e.dirtyBytes = 0
	e.dirtyOps = 0
	e.checkpoints++

	ok = true
	if err := syncDatabaseParent(e.path); err != nil {
		return fmt.Errorf("compact: sync parent directory: %w", err)
	}
	return nil
}

// Compact explicitly rewrites the database file, discarding all WAL history
// subsumed by the current snapshot. Blocks until complete. Safe to call at any
// time; the caller does not need to hold any lock.
func (e *Engine) Compact() error {
	// Snapshot vectors from provider-backed indexes into local storage
	// before acquiring e.mu.Lock. Once the lock is held, provider calls
	// would deadlock (e.mu.Lock held; provider.GetByOrdinal needs e.mu.RLock).
	if e.indexProvider != nil {
		if err := e.indexProvider.SnapshotVectors(context.Background()); err != nil {
			return fmt.Errorf("compact: snapshot vectors: %w", err)
		}
	}

	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed.Load() {
		return fmt.Errorf("compact: database is closed")
	}
	return e.compactFile()
}

// CompactionErrors returns the count of compaction errors since engine startup.
func (e *Engine) CompactionErrors() uint64 {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.compactionErrors
}

// appendChunkLocked writes a chunk header + payload at the current file end.
// Caller must hold e.mu. Does NOT fsync — callers that require durability
// (e.g. checkpointLocked) must call e.file.Sync() after appending all chunks.
func (e *Engine) appendChunkLocked(kind uint16, payload []byte) (uint64, error) {
	return e.appendChunkHeaderPayloadLocked(kind, nil, payload)
}

func (e *Engine) appendChunkHeaderPayloadLocked(kind uint16, headerPart, payloadPart []byte) (uint64, error) {
	offset, err := e.file.Seek(0, io.SeekEnd)
	if err != nil {
		return 0, err
	}
	payloadLen := uint32(len(headerPart) + len(payloadPart))
	checksum := crc32.Update(0, castagnoli, headerPart)
	checksum = crc32.Update(checksum, castagnoli, payloadPart)
	var header [16]byte
	binary.LittleEndian.PutUint32(header[0:4], chunkMagic)
	binary.LittleEndian.PutUint16(header[4:6], kind)
	binary.LittleEndian.PutUint16(header[6:8], formatVersion)
	binary.LittleEndian.PutUint32(header[8:12], payloadLen)
	binary.LittleEndian.PutUint32(header[12:16], checksum)
	if _, err := e.file.Write(header[:]); err != nil {
		return 0, err
	}
	if len(headerPart) > 0 {
		if _, err := e.file.Write(headerPart); err != nil {
			return 0, err
		}
	}
	if len(payloadPart) > 0 {
		if _, err := e.file.Write(payloadPart); err != nil {
			return 0, err
		}
	}
	return uint64(offset), nil
}

func (e *Engine) appendTransactionLocked(records []walRecord) (uint64, error) {
	defer releaseWALFramePayloads(records)
	offset, err := e.file.Seek(0, io.SeekEnd)
	if err != nil {
		return 0, err
	}

	totalSize := 0
	for _, record := range records {
		totalSize += 16 + 40 + len(record.Payload)
	}

	buf, temporaryArena, err := e.allocateWALWriteBufferLocked(totalSize)
	if err != nil {
		return 0, err
	}
	if temporaryArena != nil {
		defer temporaryArena.Free()
	}
	var written uint64
	for _, record := range records {
		// Reserve space in buf for chunk header + frame header.
		start := len(buf)
		buf = buf[:start+16+40]

		// Write frame header directly into buf — no stack array escape.
		fh := buf[start+16 : start+16+40]
		binary.LittleEndian.PutUint32(fh[0:4], record.Header.Magic)
		binary.LittleEndian.PutUint16(fh[4:6], record.Header.Version)
		binary.LittleEndian.PutUint16(fh[6:8], record.Header.RecordType)
		binary.LittleEndian.PutUint64(fh[8:16], record.Header.LSN)
		binary.LittleEndian.PutUint64(fh[16:24], record.Header.TxID)
		binary.LittleEndian.PutUint64(fh[24:32], record.Header.PrevLSN)
		binary.LittleEndian.PutUint32(fh[32:36], record.Header.PayloadLen)
		binary.LittleEndian.PutUint32(fh[36:40], record.Header.Checksum)

		payloadLen := uint32(40 + len(record.Payload))
		checksum := crc32.Update(0, castagnoli, fh)
		checksum = crc32.Update(checksum, castagnoli, record.Payload)

		// Write chunk header directly into buf.
		ch := buf[start : start+16]
		binary.LittleEndian.PutUint32(ch[0:4], chunkMagic)
		binary.LittleEndian.PutUint16(ch[4:6], chunkTypeWAL)
		binary.LittleEndian.PutUint16(ch[6:8], formatVersion)
		binary.LittleEndian.PutUint32(ch[8:12], payloadLen)
		binary.LittleEndian.PutUint32(ch[12:16], checksum)

		buf = append(buf, record.Payload...)
		written += uint64(16 + 40 + len(record.Payload))
	}

	if _, err := e.file.Write(buf); err != nil {
		return written, err
	}
	e.walTransactions++
	e.walBytes += written
	_ = offset
	return written, nil
}

func releaseWALFramePayloads(records []walRecord) {
	for i := range records {
		if records[i].PayloadEncoder != nil {
			releaseDetachedPayload(records[i].Payload, records[i].PayloadEncoder)
			records[i].PayloadEncoder = nil
		}
	}
}

func (e *Engine) syncWALLocked() error {
	if !e.walSync {
		return nil
	}
	var err error
	if e.walSyncFn != nil {
		err = e.walSyncFn(e.file)
	} else {
		err = e.file.Sync()
	}
	if err != nil {
		return fmt.Errorf("sync WAL: %w", err)
	}
	return nil
}

// allocateWALWriteBufferLocked returns an off-heap buffer with exactly size
// bytes of append capacity. Normal transactions reuse one mmap-backed arena;
// unusually large transactions receive a temporary exact-size mapping rather
// than falling back to the Go heap. Caller must hold e.mu.
func (e *Engine) allocateWALWriteBufferLocked(size int) ([]byte, *memory.Arena, error) {
	if size <= 0 {
		return nil, nil, fmt.Errorf("invalid WAL write buffer size %d", size)
	}

	arena := e.walWriteArena
	if size <= walWriteArenaSize {
		if arena == nil {
			var err error
			arena, err = memory.NewArena(walWriteArenaSize, 64)
			if err != nil {
				return nil, nil, fmt.Errorf("allocate WAL write arena: %w", err)
			}
			e.walWriteArena = arena
		} else {
			arena.Reset()
		}
	} else {
		var err error
		arena, err = memory.NewArena(uint64(size), 64)
		if err != nil {
			return nil, nil, fmt.Errorf("allocate temporary WAL write arena (%d bytes): %w", size, err)
		}
	}

	ptr, err := arena.Alloc(uint64(size))
	if err != nil {
		if arena != e.walWriteArena {
			_ = arena.Free()
		}
		return nil, nil, fmt.Errorf("reserve WAL write buffer (%d bytes): %w", size, err)
	}
	buf := unsafe.Slice((*byte)(ptr), size)[:0:size]
	if arena == e.walWriteArena {
		return buf, nil, nil
	}
	return buf, arena, nil
}

func (e *Engine) nextLSNLocked() uint64 {
	e.lastLSN++
	return e.lastLSN
}

func (e *Engine) nextTxIDLocked() uint64 {
	e.lastTxID++
	return e.lastTxID
}

func newFrame(recordType uint16, lsn, txID, prevLSN uint64, payload encodedPayload) walRecord {
	return walRecord{
		Header: walFrameHeader{
			Magic:      chunkMagic,
			Version:    formatVersion,
			RecordType: recordType,
			LSN:        lsn,
			TxID:       txID,
			PrevLSN:    prevLSN,
			PayloadLen: uint32(len(payload.bytes)),
			Checksum:   crc32.Checksum(payload.bytes, castagnoli),
		},
		Payload:        payload.bytes,
		PayloadEncoder: payload.encoder,
	}
}

func releaseDetachedPayload(payload []byte, enc *util.BinaryEncoder) {
	if enc == nil {
		return
	}
	if payload != nil {
		enc.Buf = payload[:0]
	}
	util.ReleaseBinaryEncoder(enc)
}

func (e *Engine) createCollection(name string, config storage.CollectionConfig) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed.Load() {
		return fmt.Errorf("database is closed")
	}
	if collection := e.state.Collections[name]; collection != nil && !collection.Deleted {
		return fmt.Errorf("collection %s already exists", name)
	}

	payload, err := encodeCollectionCreatePayloadBinary(collectionCreatePayload{Name: name, Config: config})
	if err != nil {
		return err
	}
	txID := e.nextTxIDLocked()
	beginLSN := e.nextLSNLocked()
	opLSN := e.nextLSNLocked()
	commitLSN := e.nextLSNLocked()
	frames := []walRecord{
		newFrame(recordTypeTxBegin, beginLSN, txID, 0, emptyPayload()),
		newFrame(recordTypeCollectionCreate, opLSN, txID, beginLSN, payload),
		newFrame(recordTypeTxCommit, commitLSN, txID, opLSN, emptyPayload()),
	}
	written, err := e.appendTransactionLocked(frames)
	if err != nil {
		return err
	}
	if err := e.syncWALLocked(); err != nil {
		return err
	}
	e.applyCreateCollection(name, config, opLSN)
	e.collections[name] = &Collection{engine: e, name: name}
	e.markDirtyLocked(written, 1)
	return e.maybeCheckpointLocked()
}

// batchFlusher is a background goroutine that periodically flushes the WAL batch buffer.
// It wakes up every batchFlushInterval and flushes any accumulated entries.
func (e *Engine) batchFlusher() {
	ticker := time.NewTicker(batchFlushInterval)
	defer ticker.Stop()

	timer := time.NewTimer(batchFlushInterval)
	defer timer.Stop()

	for {
		select {
		case <-e.ctx.Done():
			return
		case <-ticker.C:
			_ = e.flushBatch()
		case <-e.batchBuffer.flusher:
			e.waitForGroupCommit()
			_ = e.flushBatch()
		case <-timer.C:
			_ = e.flushBatch()
		}
		// Reset timer for next iteration
		if !timer.Stop() {
			select {
			case <-timer.C:
			default:
			}
		}
		timer.Reset(batchFlushInterval)
	}
}

func (e *Engine) waitForGroupCommit() {
	target := e.groupCommitTarget
	if target <= 0 {
		delay := adaptiveGroupCommitWindow(atomic.LoadInt32(&e.batchBuffer.pendingWaiters))
		if delay > 0 {
			time.Sleep(delay)
		}
		return
	}

	deadline := time.Now().Add(e.groupCommitMaxDelay)
	step := time.Duration(groupCommitStepWindow.Load())
	if step <= 0 {
		step = 100 * time.Microsecond
	}
	for atomic.LoadInt32(&e.batchBuffer.pendingWaiters) < target {
		remaining := time.Until(deadline)
		if remaining <= 0 {
			return
		}
		if step > remaining {
			step = remaining
		}
		time.Sleep(step)
	}
}

func adaptiveGroupCommitWindow(waiters int32) time.Duration {
	window := time.Duration(groupCommitWindow.Load())
	if window <= 0 {
		return 0
	}
	if waiters <= 1 {
		return window
	}
	step := time.Duration(groupCommitStepWindow.Load())
	maxWindow := time.Duration(groupCommitMaxWindow.Load())
	delay := window + time.Duration(waiters-1)*step
	if delay > maxWindow {
		return maxWindow
	}
	return delay
}

func (e *Engine) requestBatchFlush() {
	if !atomic.CompareAndSwapInt32(&e.batchBuffer.flushSignalPending, 0, 1) {
		return
	}
	select {
	case e.batchBuffer.flusher <- struct{}{}:
	default:
	}
}

// flushBatch flushes all accumulated entries in the batch buffer to WAL.
// It acquires the engine mutex and writes all buffered entries as a single transaction.
// If the buffer is empty, this is a no-op.
// After flushing, it signals any waiting foreground flush completions.
// Returns the first error encountered during flush, or nil on success.
func (e *Engine) flushBatch() error {
	e.batchBuffer.flushMu.Lock()
	defer e.batchBuffer.flushMu.Unlock()

	e.batchBuffer.mu.Lock()
	if len(e.batchBuffer.entries) == 0 && len(e.batchBuffer.flushNow) == 0 {
		e.batchBuffer.mu.Unlock()
		atomic.StoreInt32(&e.batchBuffer.flushSignalPending, 0)
		return nil
	}
	// Take ownership of the buffer and reset
	entries := e.batchBuffer.entries
	e.batchBuffer.entries = e.batchBuffer.spareEntries[:0]
	e.batchBuffer.spareEntries = nil
	// Take ownership of pending flush completions
	pendingFlushes := e.batchBuffer.flushNow
	e.batchBuffer.flushNow = e.batchBuffer.spareFlushNow[:0]
	e.batchBuffer.spareFlushNow = nil
	e.batchBuffer.mu.Unlock()
	atomic.AddInt32(&e.batchBuffer.pendingWaiters, -int32(len(pendingFlushes)))
	defer func() {
		for i := range entries {
			releaseBatchEntryPayloads(&entries[i])
			entries[i] = batchEntry{}
		}
		clear(pendingFlushes)
		e.batchBuffer.mu.Lock()
		e.batchBuffer.spareEntries = entries[:0]
		e.batchBuffer.spareFlushNow = pendingFlushes[:0]
		e.batchBuffer.mu.Unlock()
	}()

	// Nothing to flush and no one waiting
	if len(entries) == 0 && len(pendingFlushes) == 0 {
		return nil
	}

	// Acquire engine lock for state modifications
	e.mu.Lock()
	defer e.mu.Unlock()

	// Signal all waiters with the result
	var firstErr error
	signalErr := func(err error) {
		if err != nil && firstErr == nil {
			firstErr = err
		}
		for i, request := range pendingFlushes {
			var durable storage.DurableRange
			if i < len(entries) {
				durable.FirstLSN = entries[i].firstLSN
				durable.CommitLSN = entries[i].commitLSN
			}
			e.walRequests.complete(request, durable, err)
		}
	}

	if e.closed.Load() {
		signalErr(fmt.Errorf("database is closed"))
		atomic.StoreInt32(&e.batchBuffer.flushSignalPending, 0)
		return fmt.Errorf("database is closed")
	}

	batchedEntries := 0
	for i := range entries {
		batchedEntries += entries[i].count()
	}

	// Write contiguous collection runs directly. This preserves grouping without
	// allocating merged descriptor and pointer slices for the common scalar path.
	for start := 0; start < len(entries); {
		end := start + 1
		for end < len(entries) && entries[end].collection == entries[start].collection {
			end++
		}
		written, firstLSN, commitLSN, err := e.appendBatchRunWALLocked(entries[start:end])
		if err != nil {
			// A write failure makes the on-disk prefix ambiguous. Do not retry
			// automatically and risk duplicating a transaction; recovery will
			// accept only complete framed commits.
			signalErr(err)
			atomic.StoreInt32(&e.batchBuffer.flushSignalPending, 0)
			return err
		}
		for i := start; i < end; i++ {
			entries[i].firstLSN = firstLSN
			entries[i].commitLSN = commitLSN
		}
		entries[start].walBytes = written
		start = end
	}
	if err := e.syncWALLocked(); err != nil {
		signalErr(err)
		atomic.StoreInt32(&e.batchBuffer.flushSignalPending, 0)
		return err
	}
	for start := 0; start < len(entries); {
		end := start + 1
		for end < len(entries) && entries[end].collection == entries[start].collection {
			end++
		}
		lsn := entries[start].firstLSN
		for i := start; i < end; i++ {
			for j := 0; j < entries[i].count(); j++ {
				entry := entries[i].entryAt(j)
				if err := e.applyRecordPutFields(entries[i].collection, entry.ID, entry.Ordinal, entry.Vector, entry.Metadata, lsn, true); err != nil {
					signalErr(err)
					atomic.StoreInt32(&e.batchBuffer.flushSignalPending, 0)
					return err
				}
				lsn++
			}
		}
		e.markDirtyLocked(entries[start].walBytes, int(lsn-entries[start].firstLSN))
		start = end
	}
	if err := e.maybeCheckpointLocked(); err != nil {
		signalErr(err)
		atomic.StoreInt32(&e.batchBuffer.flushSignalPending, 0)
		return err
	}

	e.batchFlushes++
	e.batchedEntries += uint64(batchedEntries)
	signalErr(nil)
	atomic.StoreInt32(&e.batchBuffer.flushSignalPending, 0)

	e.batchBuffer.mu.Lock()
	pendingAgain := len(e.batchBuffer.entries) > 0 || len(e.batchBuffer.flushNow) > 0
	e.batchBuffer.mu.Unlock()
	if pendingAgain {
		e.requestBatchFlush()
	}
	return nil
}

func (e *Engine) appendBatchRunWALLocked(batches []batchEntry) (uint64, uint64, uint64, error) {
	if len(batches) == 0 {
		return 0, 0, 0, nil
	}
	collection := e.state.Collections[batches[0].collection]
	if collection == nil || collection.Deleted {
		return 0, 0, 0, fmt.Errorf("collection %s not found", batches[0].collection)
	}

	entryCount := 0
	maxOrdinal := -1
	for i := range batches {
		entryCount += batches[i].count()
		for j := 0; j < batches[i].count(); j++ {
			entry := batches[i].entryAt(j)
			if entry != nil && int(entry.Ordinal) > maxOrdinal {
				maxOrdinal = int(entry.Ordinal)
			}
		}
	}
	if maxOrdinal >= 0 {
		ensureOrdinalCapacity(collection, maxOrdinal+1)
	}
	ensureRecordCapacity(collection, entryCount)

	txID := e.nextTxIDLocked()
	beginLSN := e.nextLSNLocked()
	frames := make([]walRecord, entryCount+2)
	frames[0] = newFrame(recordTypeTxBegin, beginLSN, txID, 0, emptyPayload())
	prevLSN := beginLSN
	frameIndex := 1
	for i := range batches {
		for j := 0; j < batches[i].count(); j++ {
			entry := batches[i].entryAt(j)
			encoded := batches[i].encodedAt(j)
			if encoded.encoder == nil {
				var err error
				encoded, err = encodeRecordPutPayloadBinary(recordPutPayload{
					Collection: batches[i].collection,
					ID:         entry.ID,
					Ordinal:    entry.Ordinal,
					Vector:     entry.Vector,
					Metadata:   entry.Metadata,
				})
				if err != nil {
					releaseWALFramePayloads(frames[:frameIndex])
					return 0, 0, 0, err
				}
			}
			lsn := e.nextLSNLocked()
			frames[frameIndex] = newFrame(recordTypeRecordPut, lsn, txID, prevLSN, encoded)
			if batches[i].entry != nil {
				batches[i].encodedOne = encodedPayload{}
			} else if j < len(batches[i].encoded) {
				batches[i].encoded[j] = encodedPayload{}
			}
			prevLSN = lsn
			frameIndex++
		}
	}
	commitLSN := e.nextLSNLocked()
	frames[frameIndex] = newFrame(recordTypeTxCommit, commitLSN, txID, prevLSN, emptyPayload())
	written, err := e.appendTransactionLocked(frames)
	for i := range batches {
		batches[i].encoded = nil
	}
	if err != nil {
		return written, 0, 0, err
	}
	return written, frames[1].Header.LSN, commitLSN, nil
}

func releaseBatchEntryPayloads(batch *batchEntry) {
	if batch.encodedOne.encoder != nil {
		releaseDetachedPayload(batch.encodedOne.bytes, batch.encodedOne.encoder)
		batch.encodedOne = encodedPayload{}
	}
	for i := range batch.encoded {
		if batch.encoded[i].encoder != nil {
			releaseDetachedPayload(batch.encoded[i].bytes, batch.encoded[i].encoder)
			batch.encoded[i] = encodedPayload{}
		}
	}
}

func (e *Engine) putRecords(ctx context.Context, name string, entries []*index.VectorEntry) (walRequestHandle, error) {
	if e.closed.Load() {
		return walRequestHandle{}, fmt.Errorf("database is closed")
	}

	// Pre-encode recordPut payloads before acquiring any locks.
	encoded := make([]encodedPayload, len(entries))
	for i, entry := range entries {
		if i%100 == 0 {
			select {
			case <-ctx.Done():
				for j := 0; j < i; j++ {
					releaseDetachedPayload(encoded[j].bytes, encoded[j].encoder)
				}
				return walRequestHandle{}, ctx.Err()
			default:
			}
		}
		payload, err := encodeRecordPutPayloadBinary(recordPutPayload{
			Collection: name,
			ID:         entry.ID,
			Ordinal:    entry.Ordinal,
			Vector:     entry.Vector,
			Metadata:   entry.Metadata,
		})
		if err != nil {
			for j := 0; j < i; j++ {
				releaseDetachedPayload(encoded[j].bytes, encoded[j].encoder)
			}
			return walRequestHandle{}, err
		}
		encoded[i] = payload
	}
	return e.admitBatch(batchEntry{collection: name, entries: entries, encoded: encoded}, len(entries))
}

func (e *Engine) putRecord(ctx context.Context, name string, entry *index.VectorEntry) (walRequestHandle, error) {
	if e.closed.Load() {
		return walRequestHandle{}, fmt.Errorf("database is closed")
	}
	select {
	case <-ctx.Done():
		return walRequestHandle{}, ctx.Err()
	default:
	}
	encoded, err := encodeRecordPutPayloadBinary(recordPutPayload{
		Collection: name,
		ID:         entry.ID,
		Ordinal:    entry.Ordinal,
		Vector:     entry.Vector,
		Metadata:   entry.Metadata,
	})
	if err != nil {
		return walRequestHandle{}, err
	}
	return e.admitBatch(batchEntry{collection: name, entry: entry, encodedOne: encoded}, 1)
}

func (e *Engine) admitBatch(batch batchEntry, entryCount int) (walRequestHandle, error) {
	request := e.walRequests.acquire(entryCount)
	e.batchBuffer.mu.Lock()
	if e.closed.Load() {
		e.batchBuffer.mu.Unlock()
		if batch.entry != nil {
			releaseDetachedPayload(batch.encodedOne.bytes, batch.encodedOne.encoder)
		} else {
			for j := range batch.encoded {
				releaseDetachedPayload(batch.encoded[j].bytes, batch.encoded[j].encoder)
			}
		}
		e.walRequests.cancel(request)
		return walRequestHandle{}, fmt.Errorf("database is closed")
	}
	bufferedEntries := 0
	for i := range e.batchBuffer.entries {
		bufferedEntries += e.batchBuffer.entries[i].count()
	}
	shouldFlush := bufferedEntries+entryCount >= batchSize
	e.batchBuffer.entries = append(e.batchBuffer.entries, batch)
	e.batchBuffer.flushNow = append(e.batchBuffer.flushNow, request)
	atomic.AddInt32(&e.batchBuffer.pendingWaiters, 1)
	e.batchBuffer.mu.Unlock()

	// Signal the flusher once for the current commit window and return immediately.
	e.requestBatchFlush()

	// If we've reached batch size, do a synchronous flush before returning.
	// This ensures durability for this caller's data.
	if shouldFlush {
		_ = e.flushBatch()
	}

	return request, nil
}

func (e *Engine) waitForWALFlush(ctx context.Context, request walRequestHandle) (storage.DurableRange, error) {
	_ = ctx
	// Once admitted to the WAL group, the transaction may commit even if the
	// caller's context is canceled. Wait for the definitive durable result so
	// callers never abandon follow-up index work for a committed record.
	return e.walRequests.waitFor(request)
}

func (e *Engine) deleteRecord(name, id string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed.Load() {
		return fmt.Errorf("database is closed")
	}
	collection := e.state.Collections[name]
	if collection == nil || collection.Deleted {
		return fmt.Errorf("collection %s not found", name)
	}
	record := collection.Records[id]
	if record == nil || record.Deleted {
		return fmt.Errorf("entry %s does not exist", id)
	}

	payload, err := encodeRecordDeletePayloadBinary(recordDeletePayload{Collection: name, ID: id})
	if err != nil {
		return err
	}
	txID := e.nextTxIDLocked()
	beginLSN := e.nextLSNLocked()
	opLSN := e.nextLSNLocked()
	commitLSN := e.nextLSNLocked()
	frames := []walRecord{
		newFrame(recordTypeTxBegin, beginLSN, txID, 0, emptyPayload()),
		newFrame(recordTypeRecordDelete, opLSN, txID, beginLSN, payload),
		newFrame(recordTypeTxCommit, commitLSN, txID, opLSN, emptyPayload()),
	}
	written, err := e.appendTransactionLocked(frames)
	if err != nil {
		return err
	}
	if err := e.syncWALLocked(); err != nil {
		return err
	}
	e.applyRecordDelete(name, id, opLSN)
	e.markDirtyLocked(written, 1)
	return e.maybeCheckpointLocked()
}

// PrepareTx validates transactional operations and assigns ordinals for new rows
// without mutating durable state.
func (e *Engine) PrepareTx(ctx context.Context, ops []storage.TxOperation) ([]storage.TxOperation, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.closed.Load() {
		return nil, fmt.Errorf("database is closed")
	}

	prepared := make([]storage.TxOperation, len(ops))
	nextOrdinals := make(map[string]uint32, len(ops))
	for i, op := range ops {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		collection := e.state.Collections[op.Collection]
		if collection == nil || collection.Deleted {
			return nil, fmt.Errorf("collection %s not found", op.Collection)
		}

		prepared[i] = storage.TxOperation{
			Type:               op.Type,
			Collection:         op.Collection,
			ID:                 op.ID,
			Ordinal:            op.Ordinal,
			Vector:             append([]float32(nil), op.Vector...),
			Metadata:           cloneMetadata(op.Metadata),
			ExpectedVersion:    op.ExpectedVersion,
			HasExpectedVersion: op.HasExpectedVersion,
		}

		if op.Type != storage.TxOperationPut {
			continue
		}

		if current := collection.Records[op.ID]; current != nil {
			prepared[i].Ordinal = current.Ordinal
			continue
		}

		nextOrdinal, ok := nextOrdinals[op.Collection]
		if !ok {
			nextOrdinal = collection.NextOrdinal
		}
		prepared[i].Ordinal = nextOrdinal
		nextOrdinals[op.Collection] = nextOrdinal + 1
	}

	return prepared, nil
}

// CommitTx durably appends a single atomic transaction spanning multiple collections.
func (e *Engine) CommitTx(ctx context.Context, ops []storage.TxOperation) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(ops) == 0 {
		return nil
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed.Load() {
		return fmt.Errorf("database is closed")
	}

	txID := e.nextTxIDLocked()
	beginLSN := e.nextLSNLocked()
	frames := make([]walRecord, len(ops)+2)
	frames[0] = newFrame(recordTypeTxBegin, beginLSN, txID, 0, emptyPayload())
	prevLSN := beginLSN

	for i, op := range ops {
		if err := ctx.Err(); err != nil {
			return err
		}

		collection := e.state.Collections[op.Collection]
		if collection == nil || collection.Deleted {
			return fmt.Errorf("collection %s not found", op.Collection)
		}

		lsn := e.nextLSNLocked()
		switch op.Type {
		case storage.TxOperationPut:
			payload, err := encodeRecordPutPayloadBinary(recordPutPayload{
				Collection: op.Collection,
				ID:         op.ID,
				Ordinal:    op.Ordinal,
				Vector:     op.Vector,
				Metadata:   op.Metadata,
			})
			if err != nil {
				return err
			}
			frames[i+1] = newFrame(recordTypeRecordPut, lsn, txID, prevLSN, payload)
		case storage.TxOperationDelete:
			payload, err := encodeRecordDeletePayloadBinary(recordDeletePayload{
				Collection: op.Collection,
				ID:         op.ID,
			})
			if err != nil {
				return err
			}
			frames[i+1] = newFrame(recordTypeRecordDelete, lsn, txID, prevLSN, payload)
		default:
			return fmt.Errorf("unsupported transaction operation type %d", op.Type)
		}
		prevLSN = lsn
	}

	commitLSN := e.nextLSNLocked()
	frames[len(frames)-1] = newFrame(recordTypeTxCommit, commitLSN, txID, prevLSN, emptyPayload())
	written, err := e.appendTransactionLocked(frames)
	if err != nil {
		return err
	}
	if err := e.syncWALLocked(); err != nil {
		return err
	}

	for i, op := range ops {
		recordLSN := frames[i+1].Header.LSN
		switch op.Type {
		case storage.TxOperationPut:
			if err := e.applyRecordPutFields(op.Collection, op.ID, op.Ordinal, op.Vector, op.Metadata, recordLSN, false); err != nil {
				return err
			}
		case storage.TxOperationDelete:
			e.applyRecordDelete(op.Collection, op.ID, recordLSN)
		}
	}

	e.markDirtyLocked(written, len(ops))
	return e.maybeCheckpointLocked()
}

// CreateCollection creates a new collection with persisted config.
func (e *Engine) CreateCollection(name string, config interface{}) (storage.Collection, error) {
	cfg, ok := config.(*storage.CollectionConfig)
	if !ok || cfg == nil {
		return nil, fmt.Errorf("collection config must be *storage.CollectionConfig")
	}
	if err := e.createCollection(name, *cfg); err != nil {
		return nil, err
	}
	return e.GetCollection(name)
}

// GetCollection retrieves an existing collection.
func (e *Engine) GetCollection(name string) (storage.Collection, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed.Load() {
		return nil, fmt.Errorf("database is closed")
	}
	persisted := e.state.Collections[name]
	if persisted == nil || persisted.Deleted {
		return nil, fmt.Errorf("collection %s not found", name)
	}
	if collection, ok := e.collections[name]; ok && !collection.closed.Load() {
		return collection, nil
	}
	collection := &Collection{engine: e, name: name}
	e.collections[name] = collection
	return collection, nil
}

// GetCollectionWithConfig retrieves an existing collection and its configuration.
func (e *Engine) GetCollectionWithConfig(name string) (storage.Collection, *storage.CollectionConfig, error) {
	e.mu.RLock()
	persisted := e.state.Collections[name]
	e.mu.RUnlock()
	if persisted == nil || persisted.Deleted {
		return nil, nil, fmt.Errorf("collection %s not found", name)
	}
	collection, err := e.GetCollection(name)
	if err != nil {
		return nil, nil, err
	}
	configCopy := persisted.Config
	return collection, &configCopy, nil
}

// ListCollections returns live persisted collections.
func (e *Engine) ListCollections() ([]string, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed.Load() {
		return nil, fmt.Errorf("database is closed")
	}
	names := make([]string, 0, len(e.state.Collections))
	for name, collection := range e.state.Collections {
		if collection != nil && !collection.Deleted {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	return names, nil
}

// DeleteCollection logically deletes a collection.
func (e *Engine) DeleteCollection(name string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed.Load() {
		return fmt.Errorf("database is closed")
	}
	collection := e.state.Collections[name]
	if collection == nil || collection.Deleted {
		return fmt.Errorf("collection %s not found", name)
	}
	payload, err := encodeCollectionDeletePayloadBinary(collectionDeletePayload{Name: name})
	if err != nil {
		return err
	}
	txID := e.nextTxIDLocked()
	beginLSN := e.nextLSNLocked()
	opLSN := e.nextLSNLocked()
	commitLSN := e.nextLSNLocked()
	frames := []walRecord{
		newFrame(recordTypeTxBegin, beginLSN, txID, 0, emptyPayload()),
		newFrame(recordTypeCollectionDelete, opLSN, txID, beginLSN, payload),
		newFrame(recordTypeTxCommit, commitLSN, txID, opLSN, emptyPayload()),
	}
	written, err := e.appendTransactionLocked(frames)
	if err != nil {
		return err
	}
	if err := e.syncWALLocked(); err != nil {
		return err
	}
	e.applyDeleteCollection(name, opLSN)
	if collectionObj := e.collections[name]; collectionObj != nil {
		collectionObj.closed.Store(true)
		delete(e.collections, name)
	}
	e.markDirtyLocked(written, 1)
	return e.maybeCheckpointLocked()
}

// Close checkpoints and closes the database file.
func (e *Engine) Close() error {
	// Cancel the engine context to stop the batch flusher.
	e.cancel()

	// Flush any remaining buffered entries before close.
	_ = e.flushBatch()

	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed.Load() {
		return nil
	}
	// Free all off-heap vector storage. Individual slot deallocation is
	// unnecessary: Free() releases all mmap'd slabs at once.
	for _, collection := range e.state.Collections {
		if collection.vectorSFL != nil {
			collection.vectorSFL.Free()
			collection.vectorSFL = nil
			collection.vectorSlots = nil
		}
	}
	if e.walWriteArena != nil {
		if err := e.walWriteArena.Free(); err != nil {
			return err
		}
		e.walWriteArena = nil
	}
	if e.walRequests != nil {
		if err := e.walRequests.close(); err != nil {
			return err
		}
		e.walRequests = nil
	}
	if e.dirty {
		if err := e.file.Sync(); err != nil {
			return err
		}
	}
	e.closed.Store(true)
	return e.file.Close()
}

func (e *Engine) markDirtyLocked(walBytes uint64, ops int) {
	e.dirty = true
	e.dirtyBytes += walBytes
	e.dirtyOps += ops
}

func (e *Engine) maybeCheckpointLocked() error {
	if e.dirtyBytes < checkpointThresholdBytes && e.dirtyOps < checkpointThresholdOps {
		return nil
	}
	return e.checkpointLocked()
}

// RecoveryStats returns WAL replay/discard counters observed during open.
func (e *Engine) RecoveryStats() RecoveryStats {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return RecoveryStats{
		ReplayedTransactions:  e.replayedTxs,
		DiscardedTransactions: e.discardedTxs,
		RebuiltIndexes:        e.rebuiltIndexes,
		ReplayedIndexPuts:     e.replayedIndexPuts,
		ReplayedIndexDeletes:  e.replayedIndexDeletes,
	}
}

// WriteStats returns coarse write-path counters for benchmarking and profiling.
func (e *Engine) WriteStats() storage.WriteStats {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return storage.WriteStats{
		WALTransactions:       e.walTransactions,
		WALBytes:              e.walBytes,
		BatchFlushes:          e.batchFlushes,
		BufferedVectorEntries: e.batchedEntries,
		Checkpoints:           e.checkpoints,
	}
}

func (e *Engine) visibleCollectionCountLocked() int {
	count := 0
	for _, collection := range e.state.Collections {
		if collection != nil && !collection.Deleted {
			count++
		}
	}
	return count
}

func cloneMetadata(metadata map[string]interface{}) map[string]interface{} {
	if metadata == nil {
		return nil
	}
	cloned := make(map[string]interface{}, len(metadata))
	for k, v := range metadata {
		cloned[k] = v
	}
	return cloned
}

func cloneEntry(record *recordValue) *index.VectorEntry {
	return &index.VectorEntry{
		ID:       "",
		Ordinal:  record.Ordinal,
		Vector:   append([]float32(nil), record.Vector...),
		Metadata: cloneMetadata(record.Metadata),
		Version:  record.Version,
	}
}

func (c *Collection) persisted() (*persistedCollection, error) {
	c.engine.mu.RLock()
	defer c.engine.mu.RUnlock()
	if c.closed.Load() || c.engine.closed.Load() {
		return nil, fmt.Errorf("collection %s is closed", c.name)
	}
	persisted := c.engine.state.Collections[c.name]
	if persisted == nil || persisted.Deleted {
		return nil, fmt.Errorf("collection %s not found", c.name)
	}
	return persisted, nil
}

func (c *Collection) AssignOrdinals(ctx context.Context, entries []*index.VectorEntry) error {
	_ = ctx
	c.engine.mu.Lock()
	defer c.engine.mu.Unlock()
	if c.closed.Load() || c.engine.closed.Load() {
		return fmt.Errorf("collection %s is closed", c.name)
	}
	persisted := c.engine.state.Collections[c.name]
	if persisted == nil || persisted.Deleted {
		return fmt.Errorf("collection %s not found", c.name)
	}
	for _, entry := range entries {
		if entry == nil {
			continue
		}
		if current := persisted.Records[entry.ID]; current != nil {
			entry.Ordinal = current.Ordinal
			continue
		}
		entry.Ordinal = persisted.NextOrdinal
		persisted.NextOrdinal++
	}
	return nil
}

// Insert persists a single vector entry.
// It ensures immediate durability by forcing a flush before returning.
func (c *Collection) Insert(ctx context.Context, entry *index.VectorEntry) error {
	_, err := c.InsertDurable(ctx, entry)
	return err
}

func (c *Collection) InsertDurable(ctx context.Context, entry *index.VectorEntry) (uint64, error) {
	durable, err := c.InsertDurableRange(ctx, entry)
	return durable.CommitLSN, err
}

func (c *Collection) InsertDurableRange(ctx context.Context, entry *index.VectorEntry) (storage.DurableRange, error) {
	_ = ctx
	if err := c.assignOrdinal(entry); err != nil {
		return storage.DurableRange{}, err
	}
	request, err := c.engine.putRecord(ctx, c.name, entry)
	if err != nil {
		return storage.DurableRange{}, err
	}
	return c.engine.waitForWALFlush(ctx, request)
}

func (c *Collection) assignOrdinal(entry *index.VectorEntry) error {
	c.engine.mu.RLock()
	defer c.engine.mu.RUnlock()
	if c.closed.Load() || c.engine.closed.Load() {
		return fmt.Errorf("collection %s is closed", c.name)
	}
	persisted := c.engine.state.Collections[c.name]
	if persisted == nil || persisted.Deleted {
		return fmt.Errorf("collection %s not found", c.name)
	}
	if entry == nil {
		return nil
	}
	if current := persisted.Records[entry.ID]; current != nil {
		entry.Ordinal = current.Ordinal
		return nil
	}
	n := persisted.reservedNextOrdinal.Add(1)
	entry.Ordinal = n - 1
	return nil
}

// InsertBatch persists multiple vector entries.
// It uses buffered batching for better throughput, but ensures data is flushed
// before returning so callers can immediately see the inserted data.
func (c *Collection) InsertBatch(ctx context.Context, entries []*index.VectorEntry) error {
	_, err := c.InsertBatchDurable(ctx, entries)
	return err
}

func (c *Collection) InsertBatchDurable(ctx context.Context, entries []*index.VectorEntry) (uint64, error) {
	durable, err := c.InsertBatchDurableRange(ctx, entries)
	return durable.CommitLSN, err
}

func (c *Collection) InsertBatchDurableRange(ctx context.Context, entries []*index.VectorEntry) (storage.DurableRange, error) {
	_ = ctx
	if err := c.assignOrdinals(entries); err != nil {
		return storage.DurableRange{}, err
	}
	request, err := c.engine.putRecords(ctx, c.name, entries)
	if err != nil {
		return storage.DurableRange{}, err
	}
	return c.engine.waitForWALFlush(ctx, request)
}

// DurableFrontier returns the latest transaction LSN represented by the
// collection's recovered storage view. The async derived-index tracker starts
// from this boundary after an index has been restored or rebuilt.
func (c *Collection) DurableFrontier() uint64 {
	c.engine.mu.RLock()
	defer c.engine.mu.RUnlock()
	return c.engine.lastLSN
}

// Get returns a persisted entry by ID.
func (c *Collection) Get(ctx context.Context, id string) (*index.VectorEntry, error) {
	_ = ctx
	c.engine.mu.RLock()
	defer c.engine.mu.RUnlock()
	if c.closed.Load() || c.engine.closed.Load() {
		return nil, fmt.Errorf("collection %s is closed", c.name)
	}
	persisted := c.engine.state.Collections[c.name]
	if persisted == nil || persisted.Deleted {
		return nil, fmt.Errorf("collection %s not found", c.name)
	}
	record := persisted.Records[id]
	if record == nil || record.Deleted {
		return nil, fmt.Errorf("entry %s not found", id)
	}
	entry := cloneEntry(record)
	entry.ID = id
	return entry, nil
}

func (c *Collection) Exists(ctx context.Context, id string) (bool, error) {
	_ = ctx
	c.engine.mu.RLock()
	defer c.engine.mu.RUnlock()
	if c.closed.Load() || c.engine.closed.Load() {
		return false, fmt.Errorf("collection %s is closed", c.name)
	}
	persisted := c.engine.state.Collections[c.name]
	if persisted == nil || persisted.Deleted {
		return false, fmt.Errorf("collection %s not found", c.name)
	}
	record := persisted.Records[id]
	return record != nil && !record.Deleted, nil
}

func (c *Collection) GetByOrdinal(ordinal uint32) ([]float32, error) {
	c.engine.mu.RLock()
	defer c.engine.mu.RUnlock()
	if c.closed.Load() || c.engine.closed.Load() {
		return nil, fmt.Errorf("collection %s is closed", c.name)
	}
	persisted := c.engine.state.Collections[c.name]
	if persisted == nil || persisted.Deleted {
		return nil, fmt.Errorf("collection %s not found", c.name)
	}
	if int(ordinal) >= len(persisted.ordinalToID) {
		return nil, fmt.Errorf("ordinal %d not found", ordinal)
	}
	id := persisted.ordinalToID[ordinal]
	if id == "" {
		return nil, fmt.Errorf("ordinal %d not found", ordinal)
	}
	record := persisted.Records[id]
	if record == nil || record.Deleted {
		return nil, fmt.Errorf("ordinal %d not found", ordinal)
	}
	return record.Vector, nil
}

func (c *Collection) Distance(query []float32, ordinal uint32) (float32, error) {
	vector, err := c.GetByOrdinal(ordinal)
	if err != nil {
		return 0, err
	}
	if len(query) != len(vector) {
		return 0, fmt.Errorf("query dimension %d does not match stored dimension %d", len(query), len(vector))
	}
	var sum float32
	for i := range query {
		diff := query[i] - vector[i]
		sum += diff * diff
	}
	return sum, nil
}

func (c *Collection) GetIDByOrdinal(ctx context.Context, ordinal uint32) (string, error) {
	_ = ctx
	c.engine.mu.RLock()
	defer c.engine.mu.RUnlock()
	if c.closed.Load() || c.engine.closed.Load() {
		return "", fmt.Errorf("collection %s is closed", c.name)
	}
	persisted := c.engine.state.Collections[c.name]
	if persisted == nil || persisted.Deleted {
		return "", fmt.Errorf("collection %s not found", c.name)
	}
	if int(ordinal) >= len(persisted.ordinalToID) {
		return "", fmt.Errorf("ordinal %d not found", ordinal)
	}
	id := persisted.ordinalToID[ordinal]
	if id == "" {
		return "", fmt.Errorf("ordinal %d not found", ordinal)
	}
	record := persisted.Records[id]
	if record == nil || record.Deleted {
		return "", fmt.Errorf("ordinal %d not found", ordinal)
	}
	return id, nil
}

func (c *Collection) MemoryUsage(ctx context.Context) (int64, error) {
	_ = ctx
	c.engine.mu.RLock()
	defer c.engine.mu.RUnlock()
	if c.closed.Load() || c.engine.closed.Load() {
		return 0, fmt.Errorf("collection %s is closed", c.name)
	}
	persisted := c.engine.state.Collections[c.name]
	if persisted == nil || persisted.Deleted {
		return 0, fmt.Errorf("collection %s not found", c.name)
	}
	var usage int64
	for id, record := range persisted.Records {
		if record == nil || record.Deleted {
			continue
		}
		usage += int64(len(id))
		usage += int64(len(record.Vector) * 4)
		for key, value := range record.Metadata {
			usage += int64(len(key))
			usage += util.EstimateMetadataValueSize(value)
		}
	}
	return usage, nil
}

// Delete marks a record deleted durably.
func (c *Collection) Delete(ctx context.Context, id string) error {
	_ = ctx
	return c.engine.deleteRecord(c.name, id)
}

// Iterate walks all live records.
func (c *Collection) Iterate(ctx context.Context, fn func(*index.VectorEntry) error) error {
	if fn == nil {
		return fmt.Errorf("iterate callback cannot be nil")
	}

	c.engine.mu.RLock()
	if c.closed.Load() || c.engine.closed.Load() {
		c.engine.mu.RUnlock()
		return fmt.Errorf("collection %s is closed", c.name)
	}
	persisted := c.engine.state.Collections[c.name]
	if persisted == nil || persisted.Deleted {
		c.engine.mu.RUnlock()
		return fmt.Errorf("collection %s not found", c.name)
	}

	// Snapshot the ordinal frontier. Inserts committed after iteration begins
	// receive ordinals at or beyond this boundary and are excluded.
	ordinalLimit := persisted.NextOrdinal
	c.engine.mu.RUnlock()

	// Process fixed ordinal windows. Each window re-acquires RLock only while
	// resolving and cloning records, then invokes callbacks without holding it.
	// Memory remains O(chunkSize), independent of collection cardinality.
	const chunkSize uint32 = 1024
	for start := uint32(0); start < ordinalLimit; {
		end := start + min(chunkSize, ordinalLimit-start)

		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		chunk := make([]*index.VectorEntry, 0, int(end-start))
		c.engine.mu.RLock()
		// Refetch persisted — a concurrent DeleteCollection may have replaced
		// the pointer since the initial RLock acquisition above.
		persisted := c.engine.state.Collections[c.name]
		if persisted == nil || persisted.Deleted {
			c.engine.mu.RUnlock()
			return fmt.Errorf("collection %s was deleted during iteration", c.name)
		}
		ordinalEnd := min(end, uint32(len(persisted.ordinalToID)))
		for ordinal := start; ordinal < ordinalEnd; ordinal++ {
			id := persisted.ordinalToID[ordinal]
			if id == "" {
				continue
			}
			record := persisted.Records[id]
			if record == nil || record.Deleted || record.Ordinal != ordinal {
				continue
			}
			entry := cloneEntry(record)
			entry.ID = id
			chunk = append(chunk, entry)
		}
		c.engine.mu.RUnlock()

		for _, entry := range chunk {
			if err := ctx.Err(); err != nil {
				return err
			}
			if err := fn(entry); err != nil {
				return err
			}
		}
		start = end
	}
	return nil
}

// Count returns the exact number of live records.
func (c *Collection) Count(ctx context.Context) (int, error) {
	_ = ctx
	c.engine.mu.RLock()
	defer c.engine.mu.RUnlock()
	if c.closed.Load() || c.engine.closed.Load() {
		return 0, fmt.Errorf("collection %s is closed", c.name)
	}
	persisted := c.engine.state.Collections[c.name]
	if persisted == nil || persisted.Deleted {
		return 0, fmt.Errorf("collection %s not found", c.name)
	}
	return int(persisted.LiveCount), nil
}

// initReservedOrdinals seeds each collection's atomic ordinal pre-reservation
// counter from the committed NextOrdinal. Called after recovery so the first
// insert doesn't collide with ordinals already assigned to committed records.
func (e *Engine) initReservedOrdinals() {
	for _, coll := range e.state.Collections {
		if coll != nil && !coll.Deleted {
			coll.reservedNextOrdinal.Store(coll.NextOrdinal)
		}
	}
}

// NextOrdinal returns the next ordinal that would be assigned to a new record.
func (c *Collection) NextOrdinal(ctx context.Context) (uint32, error) {
	_ = ctx
	c.engine.mu.RLock()
	defer c.engine.mu.RUnlock()
	if c.closed.Load() || c.engine.closed.Load() {
		return 0, fmt.Errorf("collection %s is closed", c.name)
	}
	persisted := c.engine.state.Collections[c.name]
	if persisted == nil || persisted.Deleted {
		return 0, fmt.Errorf("collection %s not found", c.name)
	}
	return persisted.NextOrdinal, nil
}

// Close releases the collection handle.
func (c *Collection) Close() error {
	c.closed.Store(true)
	return nil
}

func (c *Collection) assignOrdinals(entries []*index.VectorEntry) error {
	c.engine.mu.RLock()
	defer c.engine.mu.RUnlock()
	if c.closed.Load() || c.engine.closed.Load() {
		return fmt.Errorf("collection %s is closed", c.name)
	}
	persisted := c.engine.state.Collections[c.name]
	if persisted == nil || persisted.Deleted {
		return fmt.Errorf("collection %s not found", c.name)
	}
	for _, entry := range entries {
		if entry == nil {
			continue
		}
		if current := persisted.Records[entry.ID]; current != nil {
			entry.Ordinal = current.Ordinal
			continue
		}
		n := persisted.reservedNextOrdinal.Add(1)
		entry.Ordinal = n - 1
	}
	return nil
}
