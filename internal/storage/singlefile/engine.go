package singlefile

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/xDarkicex/libravdb/internal/index"
	"github.com/xDarkicex/libravdb/internal/storage"
)

const (
	pageSize             = 4096
	formatVersion        = uint16(1)
	fileMagic            = "LIBRAVDB"
	headerMagic   uint32 = 0x4C564442
	metaMagic     uint32 = 0x4C56444D
	metaMagicV2   uint32 = 0x4C56444E // V2 metapage includes index persistence fields
	chunkMagic    uint32 = 0x4C564443

	chunkTypeSnapshot = uint16(1)
	chunkTypeWAL      = uint16(2)
	chunkTypeIndex    = uint16(3)

	recordTypeTxBegin          = uint16(1)
	recordTypeTxCommit         = uint16(2)
	recordTypeTxAbort          = uint16(3)
	recordTypeCollectionCreate = uint16(10)
	recordTypeCollectionDelete = uint16(11)
	recordTypeRecordPut        = uint16(20)
	recordTypeRecordDelete     = uint16(21)
)

var castagnoli = crc32.MakeTable(crc32.Castagnoli)

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
	Header         walFrameHeader
	Payload        []byte
	payloadEncoder *binaryEncoder
}

type persistedState struct {
	NextCollectionID uint64                          `json:"next_collection_id"`
	Collections      map[string]*persistedCollection `json:"collections"`
}

type persistedCollection struct {
	ID          uint64                   `json:"id"`
	Config      storage.CollectionConfig `json:"config"`
	CreatedLSN  uint64                   `json:"created_lsn"`
	UpdatedLSN  uint64                   `json:"updated_lsn"`
	Deleted     bool                     `json:"deleted"`
	LiveCount   uint64                   `json:"live_count"`
	NextOrdinal uint32                   `json:"next_ordinal"`
	Records     map[string]*recordValue  `json:"records"`
	ordinalToID []string
}

type recordValue struct {
	Version    uint64                 `json:"version"`
	CreatedLSN uint64                 `json:"created_lsn"`
	UpdatedLSN uint64                 `json:"updated_lsn"`
	Deleted    bool                   `json:"deleted"`
	Ordinal    uint32                 `json:"ordinal"`
	Vector     []float32              `json:"vector"`
	Metadata   map[string]interface{} `json:"metadata"`
}

type collectionCreatePayload struct {
	Name   string                   `json:"name"`
	Config storage.CollectionConfig `json:"config"`
}

type collectionDeletePayload struct {
	Name string `json:"name"`
}

type recordPutPayload struct {
	Collection string                 `json:"collection"`
	ID         string                 `json:"id"`
	Ordinal    uint32                 `json:"ordinal"`
	Vector     []float32              `json:"vector"`
	Metadata   map[string]interface{} `json:"metadata"`
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

// indexBlockEntry is a single collection's serialized index within the index chunk.
type indexBlockEntry struct {
	name           string
	indexType      uint8
	indexVersion   uint16
	payload        []byte
	payloadChecksum uint32
}

// Engine is the single-file storage engine.
type Engine struct {
	mu              sync.RWMutex
	path            string
	file            *os.File
	state           *persistedState
	collections     map[string]*Collection
	fileID          uint64
	metaEpoch       uint64
	activeMetaPage  uint64
	lastLSN         uint64
	lastTxID        uint64
	dirty           bool
	dirtyBytes      uint64
	dirtyOps        int
	walTransactions uint64
	walBytes        uint64
	batchFlushes    uint64
	batchedEntries  uint64
	checkpoints      uint64
	replayedTxs      uint64
	discardedTxs     uint64
	compactionErrors uint64
	closed           bool
	status           atomic.Int32 // EngineStatus: Starting → Recovering* → Ready | Failed
	recoveryErr      atomic.Value // error, set when status transitions to Failed
	indexProvider    IndexSnapshotProvider

	// WAL batch buffer: accumulates entries across multiple putRecords calls
	// and flushes them together to reduce fsync overhead.
	batchBuffer struct {
		mu                 sync.Mutex
		entries            []batchEntry  // accumulated entries awaiting flush
		flusher            chan struct{} // signal to wake up flusher
		closed             bool
		flushNow           []chan error // completion channels for foreground flushes
		flushSignalPending int32
		pendingWaiters     int32
	}
}

// batchEntry holds a buffered record pending WAL flush.
type batchEntry struct {
	collection string
	entries    []*index.VectorEntry
}

// startBatchFlusher is a hook for testing. It starts the background flusher goroutine.
// Defaults to launching the goroutine directly; can be overridden in tests.
var startBatchFlusher = func(e *Engine) {
	go e.batchFlusher()
}

// RecoveryStats exposes WAL replay outcomes for debugging and tests.
type RecoveryStats struct {
	ReplayedTransactions  uint64
	DiscardedTransactions uint64
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

// New opens or creates a single-file database.
func New(path string, opts ...Option) (storage.Engine, error) {
	resolved, err := resolveDatabasePath(path)
	if err != nil {
		return nil, err
	}

	file, err := os.OpenFile(resolved, os.O_RDWR|os.O_CREATE, 0644)
	if err != nil {
		return nil, fmt.Errorf("open database file: %w", err)
	}

	engine := &Engine{
		path:        resolved,
		file:        file,
		state:       &persistedState{NextCollectionID: 1, Collections: make(map[string]*persistedCollection)},
		collections: make(map[string]*Collection),
	}

	// Apply options before recovery so provider is available to loadIndexes.
	for _, opt := range opts {
		if err := opt(engine); err != nil {
			file.Close()
			return nil, err
		}
	}

	// Initialize WAL batch buffer channels (flusher goroutine started after init succeeds)
	engine.batchBuffer.flusher = make(chan struct{})
	engine.batchBuffer.entries = nil

	stat, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("stat database file: %w", err)
	}

	if stat.Size() == 0 {
		if err := engine.initializeEmpty(); err != nil {
			file.Close()
			return nil, err
		}
		// Start background flusher only after successful initialization
		startBatchFlusher(engine)
		engine.status.Store(int32(storage.StatusReady))
		return engine, nil
	}

	if err := engine.openExisting(); err != nil {
		file.Close()
		return nil, err
	}

	// Start background flusher only after successful open
	startBatchFlusher(engine)
	return engine, nil
}

func resolveDatabasePath(path string) (string, error) {
	if path == "" {
		path = "./data.libravdb"
	}

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

	meta1, err1 := e.readMetaPage(1)
	meta2, err2 := e.readMetaPage(2)
	if err1 != nil && err2 != nil {
		e.fail(fmt.Errorf("failed to read any valid metapage: %v / %v", err1, err2))
		return fmt.Errorf("failed to read any valid metapage: %v / %v", err1, err2)
	}

	chosen := chooseMeta(meta1, err1, meta2, err2)
	e.metaEpoch = chosen.MetaEpoch
	e.activeMetaPage = metaPageNumber(chosen)
	e.lastLSN = chosen.LastAppliedLSN

	// Phase 1: load snapshot
	e.status.Store(int32(storage.StatusRecoveringSnapshot))
	if chosen.SnapshotLength > 0 {
		if err := e.loadSnapshot(chosen.SnapshotOffset, chosen.SnapshotLength); err != nil {
			e.fail(fmt.Errorf("load snapshot: %w", err))
			return err
		}
	}

	// Phase 2: load or rebuild indexes
	e.status.Store(int32(storage.StatusRecoveringIndexes))
	if err := e.loadIndexes(chosen); err != nil {
		e.fail(fmt.Errorf("load indexes: %w", err))
		return err
	}

	// Phase 3: replay WAL
	e.status.Store(int32(storage.StatusReplayingWAL))
	if err := e.replayWAL(chosen.LastAppliedLSN); err != nil {
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
			continue
		}

		// Validate index type matches current collection config.
		if entry.indexType != uint8(collection.Config.IndexType) {
			if err := e.rebuildCollectionIndexFromRecords(entry.name, collection); err != nil {
				return err
			}
			continue
		}

		// Validate index format version is supported.
		if entry.indexVersion != formatVersion {
			if err := e.rebuildCollectionIndexFromRecords(entry.name, collection); err != nil {
				return err
			}
			continue
		}

		if err := e.indexProvider.DeserializeIndex(entry.name, entry.payload, &collection.Config); err != nil {
			if err := e.rebuildCollectionIndexFromRecords(entry.name, collection); err != nil {
				return err
			}
			continue
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
	return e.indexProvider.RebuildIndex(name, &collection.Config)
}

// fail transitions the engine to storage.StatusFailed and stores the error.
func (e *Engine) fail(err error) {
	e.recoveryErr.Store(err)
	e.status.Store(int32(storage.StatusFailed))
}

func chooseMeta(meta1 *metaPage, err1 error, meta2 *metaPage, err2 error) *metaPage {
	if err1 == nil && (err2 != nil || meta1.MetaEpoch >= meta2.MetaEpoch) {
		return meta1
	}
	return meta2
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
	if err := writeFixedPage(e.file, 0, encodeHeader(header)); err != nil {
		return err
	}

	metaA := &metaPage{Magic: metaMagic, RootFreelist: 0}
	metaB := &metaPage{Magic: metaMagic, RootFreelist: math.MaxUint64}
	if err := writeFixedPage(e.file, 1, encodeMeta(metaA)); err != nil {
		return err
	}
	if err := writeFixedPage(e.file, 2, encodeMeta(metaB)); err != nil {
		return err
	}

	return e.file.Sync()
}

func writeFixedPage(file *os.File, page uint64, data []byte) error {
	if len(data) > pageSize {
		return fmt.Errorf("page payload too large: %d", len(data))
	}
	buf := make([]byte, pageSize)
	copy(buf, data)
	_, err := file.WriteAt(buf, int64(page)*pageSize)
	return err
}

func encodeHeader(header *fileHeader) []byte {
	buf := make([]byte, pageSize)
	copy(buf[:8], header.Magic[:])
	binary.LittleEndian.PutUint16(buf[8:10], header.FormatVersion)
	binary.LittleEndian.PutUint16(buf[10:12], header.PageSize)
	binary.LittleEndian.PutUint32(buf[12:16], header.FeatureFlags)
	binary.LittleEndian.PutUint64(buf[16:24], header.FileID)
	binary.LittleEndian.PutUint64(buf[24:32], header.CreatedUnixNano)
	binary.LittleEndian.PutUint64(buf[32:40], header.LastCheckpointLSN)
	binary.LittleEndian.PutUint64(buf[40:48], header.ActiveMetaPage)
	binary.LittleEndian.PutUint64(buf[48:56], header.WALStartPage)
	binary.LittleEndian.PutUint64(buf[56:64], header.WALHeadPage)
	binary.LittleEndian.PutUint64(buf[64:72], header.WALTailPage)
	checksum := crc32.Checksum(buf[:72], castagnoli)
	binary.LittleEndian.PutUint32(buf[72:76], checksum)
	return buf
}

func encodeMeta(meta *metaPage) []byte {
	buf := make([]byte, pageSize)
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
	buf := make([]byte, pageSize)
	if _, err := e.file.ReadAt(buf, 0); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}
	if string(buf[:8]) != fileMagic {
		return nil, fmt.Errorf("invalid database file magic")
	}
	expected := crc32.Checksum(buf[:72], castagnoli)
	if got := binary.LittleEndian.Uint32(buf[72:76]); got != expected {
		return nil, fmt.Errorf("invalid header checksum")
	}
	header := &fileHeader{}
	copy(header.Magic[:], buf[:8])
	header.FormatVersion = binary.LittleEndian.Uint16(buf[8:10])
	header.PageSize = binary.LittleEndian.Uint16(buf[10:12])
	header.FeatureFlags = binary.LittleEndian.Uint32(buf[12:16])
	header.FileID = binary.LittleEndian.Uint64(buf[16:24])
	header.CreatedUnixNano = binary.LittleEndian.Uint64(buf[24:32])
	header.LastCheckpointLSN = binary.LittleEndian.Uint64(buf[32:40])
	header.ActiveMetaPage = binary.LittleEndian.Uint64(buf[40:48])
	header.WALStartPage = binary.LittleEndian.Uint64(buf[48:56])
	header.WALHeadPage = binary.LittleEndian.Uint64(buf[56:64])
	header.WALTailPage = binary.LittleEndian.Uint64(buf[64:72])
	header.Checksum = binary.LittleEndian.Uint32(buf[72:76])
	return header, nil
}

func (e *Engine) readMetaPage(page uint64) (*metaPage, error) {
	buf := make([]byte, pageSize)
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

func (e *Engine) loadSnapshot(offset, length uint64) error {
	payload, err := e.readChunkAt(offset)
	if err != nil {
		return err
	}
	if uint64(len(payload)) != length {
		return fmt.Errorf("snapshot length mismatch")
	}
	state, err := decodeStateBinary(payload)
	if err != nil {
		return fmt.Errorf("decode snapshot: %w", err)
	}
	e.state = state
	return nil
}

func (e *Engine) readChunkAt(offset uint64) ([]byte, error) {
	headerBuf := make([]byte, 16)
	if _, err := e.file.ReadAt(headerBuf, int64(offset)); err != nil {
		return nil, err
	}
	header := decodeChunkHeader(headerBuf)
	if header.Magic != chunkMagic {
		return nil, fmt.Errorf("invalid chunk magic at offset %d", offset)
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
	pending := make(map[uint64][]walRecord)

	for offset+16 <= stat.Size() {
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
			if err := e.applyCommittedFrames(frames); err != nil {
				return err
			}
			e.replayedTxs++
			delete(pending, record.Header.TxID)
		case recordTypeTxAbort:
			e.discardedTxs++
			delete(pending, record.Header.TxID)
		default:
			pending[record.Header.TxID] = append(pending[record.Header.TxID], record)
		}
		if record.Header.LSN > e.lastLSN {
			e.lastLSN = record.Header.LSN
		}
		if record.Header.TxID > e.lastTxID {
			e.lastTxID = record.Header.TxID
		}
		offset += int64(16 + chunk.PayloadLen)
	}
	e.discardedTxs += uint64(len(pending))
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

func (e *Engine) applyCommittedFrames(frames []walRecord) error {
	for _, record := range frames {
		switch record.Header.RecordType {
		case recordTypeCollectionCreate:
			payload, err := decodeCollectionCreatePayloadBinary(record.Payload)
			if err != nil {
				return err
			}
			e.applyCreateCollection(payload.Name, payload.Config, record.Header.LSN)
		case recordTypeCollectionDelete:
			payload, err := decodeCollectionDeletePayloadBinary(record.Payload)
			if err != nil {
				return err
			}
			e.applyDeleteCollection(payload.Name, record.Header.LSN)
		case recordTypeRecordPut:
			payload, err := decodeRecordPutPayloadBinary(record.Payload)
			if err != nil {
				return err
			}
			if err := e.applyRecordPut(payload, record.Header.LSN); err != nil {
				return err
			}
		case recordTypeRecordDelete:
			payload, err := decodeRecordDeletePayloadBinary(record.Payload)
			if err != nil {
				return err
			}
			e.applyRecordDelete(payload.Collection, payload.ID, record.Header.LSN)
		}
	}
	return nil
}

func (e *Engine) applyCreateCollection(name string, config storage.CollectionConfig, lsn uint64) {
	if _, exists := e.state.Collections[name]; exists {
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
	}
}

func (e *Engine) applyRecordPut(payload recordPutPayload, lsn uint64) error {
	return e.applyRecordPutFields(payload.Collection, payload.ID, payload.Ordinal, payload.Vector, payload.Metadata, lsn, false)
}

func (e *Engine) applyRecordPutOwned(payload recordPutPayload, lsn uint64, adopt bool) error {
	return e.applyRecordPutFields(payload.Collection, payload.ID, payload.Ordinal, payload.Vector, payload.Metadata, lsn, adopt)
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
	if adopt {
		current.Vector = vector
		current.Metadata = metadata
	} else {
		current.Vector = append([]float32(nil), vector...)
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
	if collection.LiveCount > 0 {
		collection.LiveCount--
	}
	if int(current.Ordinal) < len(collection.ordinalToID) {
		collection.ordinalToID[current.Ordinal] = ""
	}
	collection.UpdatedLSN = lsn
}

// encodeIndexBlock serializes all collection indexes into a single binary blob.
// Format: collectionCount | repeated { nameLen, name, indexType, indexVersion, payloadLen, payload, payloadChecksum }
func encodeIndexBlock(entries []indexBlockEntry) []byte {
	size := 4 // collectionCount uint32
	for _, e := range entries {
		size += 2 + len(e.name) // nameLen uint16 + name bytes
		size += 1 + 2            // indexType uint8 + indexVersion uint16
		size += 4 + len(e.payload) + 4 // payloadLen uint32 + payload bytes + payloadChecksum uint32
	}
	buf := make([]byte, 0, size)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(entries)))
	for _, e := range entries {
		buf = binary.LittleEndian.AppendUint16(buf, uint16(len(e.name)))
		buf = append(buf, []byte(e.name)...)
		buf = append(buf, e.indexType)
		buf = binary.LittleEndian.AppendUint16(buf, e.indexVersion)
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
	count := binary.LittleEndian.Uint32(data)
	pos := 4
	entries := make([]indexBlockEntry, 0, count)
	for i := uint32(0); i < count; i++ {
		if pos+2 > len(data) {
			break
		}
		nameLen := int(binary.LittleEndian.Uint16(data[pos:]))
		pos += 2
		if pos+nameLen > len(data) {
			break
		}
		name := string(data[pos : pos+nameLen])
		pos += nameLen
		if pos+7 > len(data) {
			break
		}
		indexType := data[pos]
		indexVersion := binary.LittleEndian.Uint16(data[pos+1:])
		payloadLen := int(binary.LittleEndian.Uint32(data[pos+3:]))
		pos += 7
		if pos+payloadLen+4 > len(data) {
			break
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
			payload:         payload,
			payloadChecksum: payloadChecksum,
		})
	}
	return entries, nil
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
			indexBytes, err := e.indexProvider.SerializeIndex(name)
			if err != nil {
				return fmt.Errorf("serialize index for %s: %w", name, err)
			}
			if indexBytes == nil {
				continue // empty collection (no index to persist); rebuilt from Records on recovery
			}
			indexType, indexVersion := e.indexProvider.IndexTypeVersion(name)
			entries = append(entries, indexBlockEntry{
				name:            name,
				indexType:       indexType,
				indexVersion:    indexVersion,
				payload:         indexBytes,
				payloadChecksum: crc32.Checksum(indexBytes, castagnoli),
			})
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
	// STEP 4: fsync (durable: snapshot + index + metapage)
	if err := writeFixedPage(e.file, nextMetaPage, encodeMeta(meta)); err != nil {
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
	if err := writeFixedPage(e.file, 0, encodeHeader(header)); err != nil {
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
		_ = e.compactFile()
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
			indexBytes, err := e.indexProvider.SerializeIndex(name)
			if err != nil {
				return fmt.Errorf("compact: serialize index for %s: %w", name, err)
			}
			if indexBytes == nil {
				continue
			}
			indexType, indexVersion := e.indexProvider.IndexTypeVersion(name)
			entries = append(entries, indexBlockEntry{
				name:            name,
				indexType:       indexType,
				indexVersion:    indexVersion,
				payload:         indexBytes,
				payloadChecksum: crc32.Checksum(indexBytes, castagnoli),
			})
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
		LastCheckpointLSN: e.lastLSN,
		ActiveMetaPage:    1,
		WALStartPage:      pageCount, // no WAL in compacted file
		WALHeadPage:       pageCount,
		WALTailPage:       pageCount,
	}
	copy(fh.Magic[:], origHeader.Magic[:])
	if err := writeFixedPage(tmpFile, 0, encodeHeader(fh)); err != nil {
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
	if err := writeFixedPage(tmpFile, 1, encodeMeta(meta)); err != nil {
		return fmt.Errorf("compact: write meta A: %w", err)
	}
	meta.RootFreelist = math.MaxUint64 // meta B marker
	if err := writeFixedPage(tmpFile, 2, encodeMeta(meta)); err != nil {
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
		e.file, _ = os.OpenFile(e.path, os.O_RDWR, 0644)
		return fmt.Errorf("compact: close original: %w", err)
	}
	e.file = nil

	if err := os.Rename(tmpPath, e.path); err != nil {
		e.file, _ = os.OpenFile(e.path, os.O_RDWR, 0644)
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
	if e.closed {
		return fmt.Errorf("compact: database is closed")
	}
	return e.compactFile()
}

// CompactionErrors returns the count of compaction errors since engine startup.
func (e *Engine) CompactionErrors() uint64 {
	return e.compactionErrors
}

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
	defer func() {
		for i := range records {
			if records[i].payloadEncoder != nil {
				releaseDetachedPayload(records[i].Payload, records[i].payloadEncoder)
				records[i].payloadEncoder = nil
			}
		}
	}()
	offset, err := e.file.Seek(0, io.SeekEnd)
	if err != nil {
		return 0, err
	}

	totalSize := 0
	for _, record := range records {
		totalSize += 16 + 40 + len(record.Payload)
	}

	buf := make([]byte, 0, totalSize)
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
	if err := e.file.Sync(); err != nil {
		return written, err
	}
	e.walTransactions++
	e.walBytes += written
	_ = offset
	return written, nil
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
		payloadEncoder: payload.encoder,
	}
}

func releaseDetachedPayload(payload []byte, enc *binaryEncoder) {
	if enc == nil {
		return
	}
	if payload != nil {
		enc.buf = payload[:0]
	}
	releaseBinaryEncoder(enc)
}

func (e *Engine) createCollection(name string, config storage.CollectionConfig) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
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
		case <-ticker.C:
			_ = e.flushBatch()
		case <-e.batchBuffer.flusher:
			delay := adaptiveGroupCommitWindow(atomic.LoadInt32(&e.batchBuffer.pendingWaiters))
			if delay > 0 {
				time.Sleep(delay)
			}
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

		// Check if the engine is closed
		e.batchBuffer.mu.Lock()
		closed := e.batchBuffer.closed
		e.batchBuffer.mu.Unlock()
		if closed {
			return
		}
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
	e.batchBuffer.mu.Lock()
	if len(e.batchBuffer.entries) == 0 && len(e.batchBuffer.flushNow) == 0 {
		e.batchBuffer.mu.Unlock()
		atomic.StoreInt32(&e.batchBuffer.flushSignalPending, 0)
		return nil
	}
	// Take ownership of the buffer and reset
	entries := e.batchBuffer.entries
	e.batchBuffer.entries = nil
	// Take ownership of pending flush completions
	pendingFlushes := e.batchBuffer.flushNow
	e.batchBuffer.flushNow = nil
	e.batchBuffer.mu.Unlock()
	atomic.AddInt32(&e.batchBuffer.pendingWaiters, -int32(len(pendingFlushes)))

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
		for _, done := range pendingFlushes {
			done <- err
		}
	}

	if e.closed {
		// Signal any waiting flushes with error
		signalErr(fmt.Errorf("database is closed"))
		atomic.StoreInt32(&e.batchBuffer.flushSignalPending, 0)
		return fmt.Errorf("database is closed")
	}

	// Merge contiguous runs for the same collection so one flush can cover more
	// buffered work without changing the original ordering of distinct collections.
	merged := make([]batchEntry, 0, len(entries))
	for _, batch := range entries {
		if len(merged) > 0 && merged[len(merged)-1].collection == batch.collection {
			merged[len(merged)-1].entries = append(merged[len(merged)-1].entries, batch.entries...)
			continue
		}
		merged = append(merged, batchEntry{
			collection: batch.collection,
			entries:    batch.entries,
		})
	}

	batchedEntries := 0
	for _, batch := range merged {
		batchedEntries += len(batch.entries)
	}

	// Write all entries to WAL - inline the put logic to avoid lock issues
	for i, batch := range merged {
		if err := e.putRecordsInlocked(batch.collection, batch.entries); err != nil {
			// On error, re-queue only the failed suffix (merged[i:] onwards).
			// Do NOT re-queue merged[:i] because they were already committed.
			failedSuffix := merged[i:]
			e.batchBuffer.mu.Lock()
			e.batchBuffer.entries = append(failedSuffix, e.batchBuffer.entries...)
			e.batchBuffer.mu.Unlock()
			signalErr(err)
			atomic.StoreInt32(&e.batchBuffer.flushSignalPending, 0)
			e.requestBatchFlush()
			return err
		}
	}

	// Signal all waiting foreground flush completions with success
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

// putRecordsInlocked writes records to WAL and applies them to memory.
// Caller must hold e.mu. This is the internal batch-friendly variant.
func (e *Engine) putRecordsInlocked(name string, entries []*index.VectorEntry) error {
	collection := e.state.Collections[name]
	if collection == nil || collection.Deleted {
		return fmt.Errorf("collection %s not found", name)
	}
	maxOrdinal := -1
	for _, entry := range entries {
		if entry != nil && int(entry.Ordinal) > maxOrdinal {
			maxOrdinal = int(entry.Ordinal)
		}
	}
	if maxOrdinal >= 0 {
		ensureOrdinalCapacity(collection, maxOrdinal+1)
	}
	ensureRecordCapacity(collection, len(entries))

	txID := e.nextTxIDLocked()
	beginLSN := e.nextLSNLocked()
	frames := make([]walRecord, len(entries)+2)
	frames[0] = newFrame(recordTypeTxBegin, beginLSN, txID, 0, emptyPayload())
	prevLSN := frames[0].Header.LSN
	ownedVectors := make([][]float32, len(entries))
	totalVectorLen := 0
	for _, entry := range entries {
		totalVectorLen += len(entry.Vector)
	}
	vectorBacking := make([]float32, totalVectorLen)
	vectorOffset := 0
	for i, entry := range entries {
		vectorLen := len(entry.Vector)
		ownedVector := vectorBacking[vectorOffset : vectorOffset+vectorLen : vectorOffset+vectorLen]
		copy(ownedVector, entry.Vector)
		vectorOffset += vectorLen
		ownedVectors[i] = ownedVector
		payloadStruct := recordPutPayload{
			Collection: name,
			ID:         entry.ID,
			Ordinal:    entry.Ordinal,
			Vector:     ownedVector,
			Metadata:   entry.Metadata,
		}
		payload, err := encodeRecordPutPayloadBinary(payloadStruct)
		if err != nil {
			return err
		}
		lsn := e.nextLSNLocked()
		frames[i+1] = newFrame(recordTypeRecordPut, lsn, txID, prevLSN, payload)
		prevLSN = lsn
	}
	commitLSN := e.nextLSNLocked()
	frames[len(frames)-1] = newFrame(recordTypeTxCommit, commitLSN, txID, prevLSN, emptyPayload())
	written, err := e.appendTransactionLocked(frames)
	if err != nil {
		return err
	}
	for i, entry := range entries {
		if err := e.applyRecordPutFields(name, entry.ID, entry.Ordinal, ownedVectors[i], entry.Metadata, frames[i+1].Header.LSN, true); err != nil {
			return err
		}
	}
	e.markDirtyLocked(written, len(ownedVectors))
	if err := e.maybeCheckpointLocked(); err != nil {
		return err
	}
	return nil
}

func (e *Engine) putRecords(name string, entries []*index.VectorEntry) (bool, error) {
	if e.closed {
		return false, fmt.Errorf("database is closed")
	}

	// Add entries to batch buffer
	e.batchBuffer.mu.Lock()
	bufferedEntries := 0
	for _, batch := range e.batchBuffer.entries {
		bufferedEntries += len(batch.entries)
	}
	shouldFlush := bufferedEntries+len(entries) >= batchSize
	e.batchBuffer.entries = append(e.batchBuffer.entries, batchEntry{
		collection: name,
		entries:    entries,
	})
	e.batchBuffer.mu.Unlock()

	// Signal the flusher once for the current commit window and return immediately.
	e.requestBatchFlush()

	// If we've reached batch size, do a synchronous flush before returning.
	// This ensures durability for this caller's data.
	if shouldFlush {
		return true, e.flushBatch()
	}

	return false, nil
}

// flushNow forces an immediate flush of the batch buffer and waits for completion.
// This is used by single-record Insert to ensure immediate durability.
// Returns the error from the flush operation, or nil on success.
func (e *Engine) flushNow() error {
	done := make(chan error)
	e.batchBuffer.mu.Lock()
	// Only add to pending flushes if there's actually something to flush
	// or if we want to ensure any in-progress flush completes
	e.batchBuffer.flushNow = append(e.batchBuffer.flushNow, done)
	e.batchBuffer.mu.Unlock()
	atomic.AddInt32(&e.batchBuffer.pendingWaiters, 1)

	// Signal the flusher once for the current commit window.
	e.requestBatchFlush()

	// Wait for flush to complete and return the error
	return <-done
}

func (e *Engine) deleteRecord(name, id string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
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

	if e.closed {
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

	if e.closed {
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

	if e.closed {
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
	if e.closed {
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

	if e.closed {
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
	// Signal the batch flusher to stop and flush remaining entries.
	e.batchBuffer.mu.Lock()
	if !e.batchBuffer.closed {
		e.batchBuffer.closed = true
	}
	e.batchBuffer.mu.Unlock()

	// Wake the flusher so it checks the closed flag promptly.
	select {
	case e.batchBuffer.flusher <- struct{}{}:
	default:
	}

	// Flush any remaining buffered entries before close.
	_ = e.flushBatch()

	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	if e.dirty {
		if err := e.file.Sync(); err != nil {
			return err
		}
	}
	e.closed = true
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
	if c.closed.Load() || c.engine.closed {
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
	if c.closed.Load() || c.engine.closed {
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
	_ = ctx
	if err := c.assignOrdinals([]*index.VectorEntry{entry}); err != nil {
		return err
	}
	flushed, err := c.engine.putRecords(c.name, []*index.VectorEntry{entry})
	if err != nil {
		return err
	}
	if flushed {
		return nil
	}
	return c.engine.flushNow()
}

// InsertBatch persists multiple vector entries.
// It uses buffered batching for better throughput, but ensures data is flushed
// before returning so callers can immediately see the inserted data.
func (c *Collection) InsertBatch(ctx context.Context, entries []*index.VectorEntry) error {
	_ = ctx
	if err := c.assignOrdinals(entries); err != nil {
		return err
	}
	flushed, err := c.engine.putRecords(c.name, entries)
	if err != nil {
		return err
	}
	if flushed {
		return nil
	}
	return c.engine.flushNow()
}

// Get returns a persisted entry by ID.
func (c *Collection) Get(ctx context.Context, id string) (*index.VectorEntry, error) {
	_ = ctx
	c.engine.mu.RLock()
	defer c.engine.mu.RUnlock()
	if c.closed.Load() || c.engine.closed {
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
	if c.closed.Load() || c.engine.closed {
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
	if c.closed.Load() || c.engine.closed {
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
	if c.closed.Load() || c.engine.closed {
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
	if c.closed.Load() || c.engine.closed {
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
			usage += estimateMetadataValueSize(value)
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
	c.engine.mu.RLock()
	if c.closed.Load() || c.engine.closed {
		c.engine.mu.RUnlock()
		return fmt.Errorf("collection %s is closed", c.name)
	}
	persisted := c.engine.state.Collections[c.name]
	if persisted == nil || persisted.Deleted {
		c.engine.mu.RUnlock()
		return fmt.Errorf("collection %s not found", c.name)
	}

	// Collect live IDs under lock (fast — map key iteration only, no cloning).
	// We release the lock before sorting/cloning so flushBatch is never starved
	// by a multi-million-record Iterate.
	ids := make([]string, 0, len(persisted.Records))
	for id, record := range persisted.Records {
		if record != nil && !record.Deleted {
			ids = append(ids, id)
		}
	}
	c.engine.mu.RUnlock()

	sort.Strings(ids)

	// Process in chunks of 10K. Each chunk re-acquires RLock briefly to clone
	// entries, then calls the user callback outside the lock. This bounds the
	// worst-case lock hold to O(chunkSize) regardless of collection cardinality.
	const chunkSize = 10000
	for start := 0; start < len(ids); start += chunkSize {
		end := min(start+chunkSize, len(ids))

		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		chunk := make([]*index.VectorEntry, 0, end-start)
		c.engine.mu.RLock()
		for _, id := range ids[start:end] {
			record := persisted.Records[id]
			if record == nil || record.Deleted {
				continue
			}
			entry := cloneEntry(record)
			entry.ID = id
			chunk = append(chunk, entry)
		}
		c.engine.mu.RUnlock()

		for _, entry := range chunk {
			if err := fn(entry); err != nil {
				return err
			}
		}
	}
	return nil
}

// Count returns the exact number of live records.
func (c *Collection) Count(ctx context.Context) (int, error) {
	_ = ctx
	c.engine.mu.RLock()
	defer c.engine.mu.RUnlock()
	if c.closed.Load() || c.engine.closed {
		return 0, fmt.Errorf("collection %s is closed", c.name)
	}
	persisted := c.engine.state.Collections[c.name]
	if persisted == nil || persisted.Deleted {
		return 0, fmt.Errorf("collection %s not found", c.name)
	}
	return int(persisted.LiveCount), nil
}

// NextOrdinal returns the next ordinal that would be assigned to a new record.
func (c *Collection) NextOrdinal(ctx context.Context) (uint32, error) {
	_ = ctx
	c.engine.mu.RLock()
	defer c.engine.mu.RUnlock()
	if c.closed.Load() || c.engine.closed {
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
	c.engine.mu.Lock()
	defer c.engine.mu.Unlock()
	if c.closed.Load() || c.engine.closed {
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
