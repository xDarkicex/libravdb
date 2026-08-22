package libravdb

import (
	"container/heap"
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/xDarkicex/lexer"
	"github.com/xDarkicex/libravdb/internal/catalog"
	"github.com/xDarkicex/libravdb/internal/filter"
	"github.com/xDarkicex/libravdb/internal/graph"
	"github.com/xDarkicex/libravdb/internal/index"
	"github.com/xDarkicex/libravdb/internal/memory"
	"github.com/xDarkicex/libravdb/internal/obs"
	"github.com/xDarkicex/libravdb/internal/optimizer"
	"github.com/xDarkicex/libravdb/internal/quant"
	"github.com/xDarkicex/libravdb/internal/storage"
	"github.com/xDarkicex/libravdb/internal/util"
)

// Collection represents a named collection of vectors with a specific schema
type Collection struct {
	lastOptimization       time.Time
	index                  index.Index
	memoryManager          memory.MemoryManager
	storage                storage.Collection
	config                 *CollectionConfig
	writes                 *writeController
	metrics                *obs.Metrics
	db                     *Database
	name                   string
	shards                 []shard
	mutationState          atomic.Pointer[mutationStateTable]
	asyncMutation          sync.RWMutex
	mu                     sync.RWMutex
	transactionMu          *sync.RWMutex // physical-shard lock for transaction views
	transactionShard       *shard        // index publication target for transaction views
	closed                 bool
	optimizationInProgress bool
	graph                  Graph
	insertHooks            []InsertHook
	deleteHooks            []DeleteHook // deprecated; retained for source compatibility
	asyncIndex             *asyncIndexQueue
	metadataIndexMu        sync.Mutex
	metadataIndex          map[string]map[string][]string
	metadataIndexBuiltAt   uint64
	jsonIndex              map[string]map[string][]string
	jsonIndexBuiltAt       uint64
	// jsonContainmentIndex is a rebuildable, GIN-shaped posting map. It is
	// derived from committed row metadata; row WAL remains authoritative.
	jsonContainmentIndex     map[string]map[string][]string
	jsonContainmentBuiltAt   uint64
	metadataMutationEpoch    atomic.Uint64
	metadataLookupIndexed    atomic.Uint64
	metadataLookupFallback   atomic.Uint64
	metadataIndexRebuilds    atomic.Uint64
	metadataIndexRecords     atomic.Uint64
	metadataLookupCandidates atomic.Uint64
	costModel                *collectionCostModelState
}

// CollectionConfig holds collection-specific configuration
type CollectionConfig struct {
	MetadataSchema         MetadataSchema                 `json:"metadata_schema,omitempty"`
	ColumnConstraints      map[string]uint16              `json:"column_constraints,omitempty"` // column name -> ColFlag* bits
	NamedUniqueConstraints map[string][]string            `json:"named_unique_constraints,omitempty"`
	PrimaryKeyColumns      []string                       `json:"primary_key_columns,omitempty"`
	ForeignKeys            []catalog.ForeignKeyInfo       `json:"foreign_keys,omitempty"` // FK constraints
	CheckConstraints       []optimizer.DDLCheckConstraint `json:"check_constraints,omitempty"`
	ColumnDefaults         map[string]string              `json:"column_defaults,omitempty"` // column name -> default literal value
	MemoryConfig           *memory.MemoryConfig           `json:"memory_config,omitempty"`
	Quantization           *quant.QuantizationConfig      `json:"quantization,omitempty"`
	RawVectorStore         string                         `json:"raw_vector_store,omitempty"`
	SavePath               string                         `json:"save_path"`
	IndexedFields          []string                       `json:"indexed_fields,omitempty"`
	SQLIndexes             []SQLIndexDefinition           `json:"sql_indexes,omitempty"`
	SQLIndexedFields       []string                       `json:"sql_indexed_fields,omitempty"`
	JSONIndexes            []JSONIndexDefinition          `json:"json_indexes,omitempty"`
	BatchConfig            BatchConfig                    `json:"batch_config,omitempty"`
	AutoIndexThresholds    struct {
		HNSWThreshold  int `json:"hnsw_threshold,omitempty"`
		IVFPQThreshold int `json:"ivfpq_threshold,omitempty"`
	} `json:"auto_index_thresholds,omitempty"`
	MemoryLimit   int64          `json:"memory_limit,omitempty"`
	EfSearch      int            `json:"ef_search"`
	ML            float64        `json:"ml"`
	RawStoreCap   int            `json:"raw_store_cap,omitempty"`
	IDMapCapacity int            `json:"id_map_capacity,omitempty"`
	Metric        DistanceMetric `json:"metric"`
	SaveInterval  time.Duration  `json:"save_interval"`
	Graph         Graph          `json:"-"`
	// GraphNamespace is persisted graph ownership metadata. SQL-created graph
	// tables use the database-wide default namespace; an empty value retains
	// the explicit/native isolated-graph behavior.
	GraphNamespace     string      `json:"graph_namespace,omitempty"`
	NProbes            int         `json:"n_probes,omitempty"`
	NClusters          int         `json:"n_clusters,omitempty"`
	IndexType          IndexType   `json:"index_type"`
	Version            int         `json:"version"`
	Dimension          int         `json:"dimension"`
	CachePolicy        CachePolicy `json:"cache_policy,omitempty"`
	M                  int         `json:"m"`
	EfConstruction     int         `json:"ef_construction"`
	Sharded            bool        `json:"sharded,omitempty"`
	EnableMMapping     bool        `json:"enable_mmapping,omitempty"`
	AutoIndexSelection bool        `json:"auto_index_selection,omitempty"`
	AutoSave           bool        `json:"auto_save"`
}

const defaultGraphNamespace = "default"

// withGraphNamespace is intentionally internal. SQL DDL opts graph tables
// into the database-owned namespace without expanding the native collection
// API, where WithGraph continues to mean an explicitly isolated graph.
func withGraphNamespace(namespace string) CollectionOption {
	return func(c *CollectionConfig) error {
		c.GraphNamespace = namespace
		return nil
	}
}

// collectionGraph binds graph transactions to one collection while sharing
// the underlying graph store. A graph transaction carries its collection in
// WAL records, so shared topology remains recoverable without mutating a
// graph-global collection name on every query.
type collectionGraph struct {
	Graph
	collection string
}

func (g *collectionGraph) BeginTxn() *graph.Txn {
	if g == nil || g.Graph == nil {
		return nil
	}
	if binder, ok := g.Graph.(interface {
		BeginTxnFor(string) *graph.Txn
	}); ok {
		return binder.BeginTxnFor(g.collection)
	}
	txn := g.Graph.BeginTxn()
	if txn != nil {
		txn.SetCollection(g.collection)
	}
	return txn
}

func (g *collectionGraph) SetWALWriter(w storage.GraphWALWriter) {
	if setter, ok := g.Graph.(interface {
		SetWALWriter(storage.GraphWALWriter)
	}); ok {
		setter.SetWALWriter(w)
	}
}

// SetCollectionName is deliberately a no-op. The wrapper binds each
// transaction independently; changing the shared store's global name would
// race concurrent SQL sessions and misroute WAL frames.
func (g *collectionGraph) SetCollectionName(string) {}

// ForEachVertexLabel keeps the optional label-inspection seam visible through
// a collection-bound shared-graph view. Query planning uses it to distinguish
// an actually labeled namespace from a graph that only has schema-side label
// hints.
func (g *collectionGraph) ForEachVertexLabel(fn func(uint64, string) bool) {
	if labels, ok := g.Graph.(interface {
		ForEachVertexLabel(func(uint64, string) bool)
	}); ok {
		labels.ForEachVertexLabel(fn)
	}
}

func (g *collectionGraph) ReplayEdgeAdd(src, tgt uint64, weight float32, kind uint8, properties []byte, commitLSN uint64) error {
	target, ok := g.Graph.(storage.GraphRecoveryTarget)
	if !ok {
		return fmt.Errorf("graph does not support WAL recovery")
	}
	return target.ReplayEdgeAdd(src, tgt, weight, kind, properties, commitLSN)
}

func (g *collectionGraph) ReplayEdgeRemove(src, tgt uint64, kind uint8, commitLSN uint64) error {
	target, ok := g.Graph.(storage.GraphRecoveryTarget)
	if !ok {
		return fmt.Errorf("graph does not support WAL recovery")
	}
	return target.ReplayEdgeRemove(src, tgt, kind, commitLSN)
}

func (g *collectionGraph) ReplayNodeEdgeDrop(nodeID uint64, commitLSN uint64) error {
	target, ok := g.Graph.(storage.GraphRecoveryTarget)
	if !ok {
		return fmt.Errorf("graph does not support WAL recovery")
	}
	return target.ReplayNodeEdgeDrop(nodeID, commitLSN)
}

func (g *collectionGraph) ReplayVertexLabel(nodeID uint64, label string, commitLSN uint64) error {
	target, ok := g.Graph.(storage.GraphRecoveryTarget)
	if !ok {
		return fmt.Errorf("graph does not support WAL recovery")
	}
	return target.ReplayVertexLabel(nodeID, label, commitLSN)
}

// Shared graph namespaces are owned by Database, not by an individual
// collection. Collection.Close therefore must not close this wrapper's
// underlying graph; Database.Close closes the namespace exactly once.
func (g *collectionGraph) Close() error { return nil }

// GraphAvailable exposes the lifecycle state of the wrapped graph without
// adding lifecycle methods to the public Graph interface. Shared namespace
// wrappers otherwise hide the concrete graph's closed-state check.
func (g *collectionGraph) GraphAvailable() bool {
	if g == nil || g.Graph == nil {
		return false
	}
	if checker, ok := g.Graph.(interface{ GraphAvailable() bool }); ok {
		return checker.GraphAvailable()
	}
	return true
}

// SQLIndexDefinition describes a named ordinary SQL index declared with
// CREATE INDEX. It is persisted as a logical declaration; posting lists are
// still derived from the collection records.
type SQLIndexDefinition struct {
	Name    string   `json:"name"`
	Columns []string `json:"columns"`
	Unique  bool     `json:"unique,omitempty"`
}

// metadataSchemaToStorage converts the public field-type enum into the
// storage-neutral representation carried by the persisted collection config.
// Keeping this conversion at the package boundary avoids coupling the storage
// package to libravdb's public types.
func metadataSchemaToStorage(schema MetadataSchema) map[string]uint8 {
	if schema == nil {
		return nil
	}
	converted := make(map[string]uint8, len(schema))
	for field, fieldType := range schema {
		converted[field] = uint8(fieldType)
	}
	return converted
}

func metadataSchemaFromStorage(schema map[string]uint8) MetadataSchema {
	if schema == nil {
		return nil
	}
	converted := make(MetadataSchema, len(schema))
	for field, fieldType := range schema {
		converted[field] = FieldType(fieldType)
	}
	return converted
}

func cloneSQLIndexDefinitions(indexes []SQLIndexDefinition) []SQLIndexDefinition {
	if len(indexes) == 0 {
		return nil
	}
	cloned := make([]SQLIndexDefinition, len(indexes))
	for i, definition := range indexes {
		cloned[i] = definition
		cloned[i].Columns = append([]string(nil), definition.Columns...)
	}
	return cloned
}

func sqlIndexesToStorage(indexes []SQLIndexDefinition) []storage.SQLIndexDefinition {
	if len(indexes) == 0 {
		return nil
	}
	converted := make([]storage.SQLIndexDefinition, len(indexes))
	for i, definition := range indexes {
		converted[i] = storage.SQLIndexDefinition{
			Name:    definition.Name,
			Columns: append([]string(nil), definition.Columns...),
			Unique:  definition.Unique,
		}
	}
	return converted
}

func sqlIndexesFromStorage(indexes []storage.SQLIndexDefinition) []SQLIndexDefinition {
	if len(indexes) == 0 {
		return nil
	}
	converted := make([]SQLIndexDefinition, len(indexes))
	for i, definition := range indexes {
		converted[i] = SQLIndexDefinition{
			Name:    definition.Name,
			Columns: append([]string(nil), definition.Columns...),
			Unique:  definition.Unique,
		}
	}
	return converted
}

// JSONIndexDefinition is a durable expression-index declaration for one
// JSON/JSONB column. Path is a PostgreSQL text-array literal such as
// "{profile,active}". TextResult corresponds to #>>; false corresponds to
// #>. The inverted postings are derived from records and rebuilt after reopen.
type JSONIndexDefinition struct {
	Name       string `json:"name"`
	Column     string `json:"column"`
	Path       string `json:"path"`
	TextResult bool   `json:"text_result,omitempty"`
}

// DistanceMetric defines the distance function to use
type DistanceMetric int

const (
	L2Distance DistanceMetric = iota
	InnerProduct
	CosineDistance
)

type trainableIndex interface {
	Train(ctx context.Context, vectors [][]float32) error
	IsTrained() bool
}

func (c *Collection) Dimension() int {
	return c.config.Dimension
}

// Config returns a defensive copy of the collection configuration. The
// process-local Graph attachment is intentionally omitted from the copy.
func (c *Collection) Config() CollectionConfig {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.config == nil {
		return CollectionConfig{}
	}

	config := *c.config
	config.Graph = nil
	config.IndexedFields = append([]string(nil), c.config.IndexedFields...)
	config.SQLIndexes = cloneSQLIndexDefinitions(c.config.SQLIndexes)
	config.SQLIndexedFields = append([]string(nil), c.config.SQLIndexedFields...)
	config.JSONIndexes = append([]JSONIndexDefinition(nil), c.config.JSONIndexes...)
	config.PrimaryKeyColumns = append([]string(nil), c.config.PrimaryKeyColumns...)
	if c.config.NamedUniqueConstraints != nil {
		config.NamedUniqueConstraints = make(map[string][]string, len(c.config.NamedUniqueConstraints))
		for name, columns := range c.config.NamedUniqueConstraints {
			config.NamedUniqueConstraints[name] = append([]string(nil), columns...)
		}
	}

	if c.config.MetadataSchema != nil {
		config.MetadataSchema = make(MetadataSchema, len(c.config.MetadataSchema))
		for field, fieldType := range c.config.MetadataSchema {
			config.MetadataSchema[field] = fieldType
		}
	}

	if c.config.MemoryConfig != nil {
		memoryConfig := *c.config.MemoryConfig
		if c.config.MemoryConfig.PressureThresholds != nil {
			memoryConfig.PressureThresholds = make(map[memory.MemoryPressureLevel]float64, len(c.config.MemoryConfig.PressureThresholds))
			for level, threshold := range c.config.MemoryConfig.PressureThresholds {
				memoryConfig.PressureThresholds[level] = threshold
			}
		}
		config.MemoryConfig = &memoryConfig
	}

	if c.config.Quantization != nil {
		quantization := *c.config.Quantization
		quantization.Levels = append([]int(nil), c.config.Quantization.Levels...)
		config.Quantization = &quantization
	}

	return config
}

// SetGraph attaches a Graph interface to an existing collection. If the
// database's storage engine supports graph edge durability, the WAL writer
// is wired automatically so Txn.Commit() writes durable edge records.
//
// SetGraph preserves the historical no-error API. Call SetGraphWithError when
// a replacement graph must report that the previous graph was already closed
// or that topology could not be copied.
func (c *Collection) SetGraph(g Graph) {
	_ = c.SetGraphWithError(g)
}

// SetGraphWithError attaches g and copies topology from the previous graph
// only while that graph is known to be live. A closed source is not treated as
// an empty graph: the new graph is attached and wired, but the unavailable
// topology is reported to the caller and the old graph is not closed again.
func (c *Collection) SetGraphWithError(g Graph) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	previous := c.graph
	c.graph = g

	var attachErr error
	// Graph-enabled collections are now recreated automatically on reopen.
	// Preserve the historical public reattach workflow as well: callers that
	// provide a replacement graph receive the live topology already recovered
	// into the previous runtime graph. The copy is in-memory only and therefore
	// does not emit duplicate WAL frames.
	if previous != nil && previous != g && g != nil {
		if err := copyGraphTopology(previous, g); err != nil {
			attachErr = err
		} else if err := previous.Close(); err != nil {
			attachErr = fmt.Errorf("close replaced graph: %w", err)
		}
	}

	if c.db != nil && g != nil {
		if definitions, ok := c.db.storage.(storage.EdgeKindDefinitionStore); ok {
			if kinds, err := definitions.ListEdgeKindDefinitions(); err == nil {
				for _, definition := range kinds {
					g.SetEdgeKindDirection(definition.Kind, definition.Undirected)
				}
			}
		}
		// Wire WAL writer independently.
		if w, ok := g.(interface {
			SetWALWriter(w storage.GraphWALWriter)
		}); ok {
			if walWriter, ok := c.db.storage.(storage.GraphWALWriter); ok {
				w.SetWALWriter(walWriter)
			}
		}
		// Wire collection name independently.
		if w, ok := g.(interface{ SetCollectionName(name string) }); ok {
			w.SetCollectionName(c.name)
		}
		// Register as per-collection recovery target.
		if setter, ok := c.db.storage.(interface {
			SetGraphRecoveryTarget(collection string, target storage.GraphRecoveryTarget)
		}); ok {
			if target, ok := g.(storage.GraphRecoveryTarget); ok {
				setter.SetGraphRecoveryTarget(c.name, target)
			}
		}
	}
	return attachErr
}

func graphAvailable(g Graph) bool {
	if g == nil {
		return false
	}
	if checker, ok := g.(interface{ GraphAvailable() bool }); ok {
		return checker.GraphAvailable()
	}
	return true
}

func copyGraphTopology(source, target Graph) error {
	if !graphAvailable(source) || !graphAvailable(target) {
		return graph.ErrGraphClosed
	}
	txn := target.BeginTxn()
	if txn == nil {
		return graph.ErrGraphClosed
	}
	var copyErr error
	source.ForEachEdge(func(src, tgt uint64, edge Edge) bool {
		if err := target.AddEdge(txn, src, tgt, edge.Weight, edge.GetKind()); err != nil {
			copyErr = err
			return false
		}
		return true
	})
	if copyErr != nil {
		return fmt.Errorf("copy graph topology: %w", copyErr)
	}
	if err := txn.ApplyInMemory(); err != nil {
		return fmt.Errorf("copy graph topology: %w", err)
	}
	if labels, ok := source.(interface {
		ForEachVertexLabel(func(uint64, string) bool)
	}); ok {
		labels.ForEachVertexLabel(func(nodeID uint64, label string) bool {
			target.RegisterVertexLabel(nodeID, label)
			return true
		})
	}
	return nil
}

// GetIndex returns the collection's index, or nil if none is configured.
func (c *Collection) GetIndex() index.Index {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.index
}

// GetGraph returns the collection's graph layer, or nil if none is configured.
func (c *Collection) GetGraph() Graph {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.graph
}

// getOrdinal resolves an ID to its shard-local ordinal without hydrating a
// vector or metadata payload. Storage engines that do not provide the narrow
// lookup retain the compatible Collection.Get fallback.
func (c *Collection) getOrdinal(ctx context.Context, id string) (uint32, error) {
	type ordinalProvider interface {
		GetOrdinal(context.Context, string) (uint32, error)
	}
	if c.shards != nil {
		storage := c.getShard(id).storage
		if provider, ok := storage.(ordinalProvider); ok {
			return provider.GetOrdinal(ctx, id)
		}
	} else if provider, ok := c.storage.(ordinalProvider); ok {
		return provider.GetOrdinal(ctx, id)
	}
	record, err := c.Get(ctx, id)
	if err != nil {
		return 0, err
	}
	return record.Ordinal, nil
}

func trainingIndexState(idx index.Index) (trainableIndex, bool) {
	trainable, ok := idx.(trainableIndex)
	if !ok || trainable.IsTrained() {
		return nil, false
	}
	return trainable, true
}

func (c *Collection) ivfpqConfig() *index.IVFPQConfig {
	nClusters := c.config.NClusters
	if nClusters <= 0 {
		nClusters = 100
	}

	nProbes := c.config.NProbes
	if nProbes <= 0 {
		nProbes = 10
	}
	if nProbes > nClusters {
		nProbes = nClusters
	}

	return &index.IVFPQConfig{
		Dimension:     c.config.Dimension,
		NClusters:     nClusters,
		NProbes:       nProbes,
		Metric:        util.DistanceMetric(c.config.Metric),
		Quantization:  c.config.Quantization,
		MaxIterations: 100,
		Tolerance:     1e-4,
		RandomSeed:    42,
	}
}

func prepareIndexForEntries(ctx context.Context, idx index.Index, metric DistanceMetric, entries []*index.VectorEntry) error {
	trainable, ok := trainingIndexState(idx)
	if !ok {
		return nil
	}
	if len(entries) == 0 {
		return nil
	}

	indexEntries := entriesForIndex(metric, entries)
	vectors := make([][]float32, len(entries))
	for i, entry := range indexEntries {
		vectors[i] = entry.Vector
	}

	if err := trainable.Train(ctx, vectors); err != nil {
		return fmt.Errorf("failed to train index: %w", err)
	}
	return nil
}

func insertEntriesIntoIndex(ctx context.Context, idx index.Index, metric DistanceMetric, entries []*index.VectorEntry) error {
	if len(entries) == 0 {
		return nil
	}

	if err := idx.BatchInsert(ctx, entriesForIndex(metric, entries)); err != nil {
		return fmt.Errorf("failed to batch insert into index: %w", err)
	}
	return nil
}

func entryForIndex(metric DistanceMetric, entry *index.VectorEntry) *index.VectorEntry {
	if metric != CosineDistance || entry == nil || len(entry.Vector) == 0 {
		return entry
	}
	indexEntry := *entry
	indexEntry.Vector = vectorForIndex(metric, entry.Vector)
	return &indexEntry
}

func entriesForIndex(metric DistanceMetric, entries []*index.VectorEntry) []*index.VectorEntry {
	if metric != CosineDistance {
		return entries
	}
	indexEntries := make([]*index.VectorEntry, len(entries))
	for i, entry := range entries {
		indexEntries[i] = entryForIndex(metric, entry)
	}
	return indexEntries
}

func vectorForIndex(metric DistanceMetric, vector []float32) []float32 {
	if metric != CosineDistance || len(vector) == 0 {
		return vector
	}
	var normSq float64
	for _, v := range vector {
		normSq += float64(v) * float64(v)
	}
	if normSq == 0 {
		return append([]float32(nil), vector...)
	}
	norm := math.Sqrt(normSq)
	if math.Abs(norm-1) <= 1e-5 {
		return vector
	}
	normalized := make([]float32, len(vector))
	invNorm := float32(1 / norm)
	for i, v := range vector {
		normalized[i] = v * invNorm
	}
	return normalized
}

func createIndexForCollection(config *CollectionConfig, provider interface {
	GetByOrdinal(uint32) ([]float32, error)
	Distance([]float32, uint32) (float32, error)
}) (index.Index, error) {
	switch config.IndexType {
	case HNSW:
		return index.NewHNSW(&index.HNSWConfig{
			Dimension:      config.Dimension,
			M:              config.M,
			EfConstruction: config.EfConstruction,
			EfSearch:       config.EfSearch,
			ML:             config.ML,
			Metric:         util.DistanceMetric(config.Metric),
			Provider:       provider,
			RawVectorStore: config.RawVectorStore,
			RawStoreCap:    config.RawStoreCap,
			IDMapCapacity:  config.IDMapCapacity,
			Quantization:   config.Quantization,
		})
	case IVFPQ:
		temp := &Collection{config: config}
		return index.NewIVFPQ(temp.ivfpqConfig())
	case Flat:
		return index.NewFlat(&index.FlatConfig{
			Dimension:    config.Dimension,
			Metric:       util.DistanceMetric(config.Metric),
			Quantization: config.Quantization,
		})
	case BTree:
		return index.NewBTree(&index.BTreeConfig{
			PageSlots:  16384, // 64MB — grows with usage, Prealloc=false
			PageShards: 64,
		})
	default:
		return nil, fmt.Errorf("unsupported index type: %v", config.IndexType)
	}
}

func buildIndexForEntries(ctx context.Context, config *CollectionConfig, provider interface {
	GetByOrdinal(uint32) ([]float32, error)
	Distance([]float32, uint32) (float32, error)
}, entries []*index.VectorEntry) (index.Index, error) {
	idx, err := createIndexForCollection(config, provider)
	if err != nil {
		return nil, err
	}
	if err := prepareIndexForEntries(ctx, idx, config.Metric, entries); err != nil {
		idx.Close()
		return nil, err
	}
	if err := insertEntriesIntoIndex(ctx, idx, config.Metric, entries); err != nil {
		idx.Close()
		return nil, fmt.Errorf("failed to insert vectors into index: %w", err)
	}
	return idx, nil
}

// IndexType defines the index algorithm to use
type IndexType int

const (
	HNSW IndexType = iota
	IVFPQ
	Flat
	BTree
)

// DefaultAutoIndexThresholds defines the default thresholds for auto-index selection.
// These can be overridden via CollectionOption when creating a collection.
const (
	// DefaultHNSWThreshold is the default vector count at which HNSW is selected over Flat.
	// Collections with fewer vectors use Flat (exact search).
	// Collections at or above this count use HNSW (approximate search with better asymptotic performance).
	// The value 2000 balances query latency savings against HNSW build/update overhead.
	DefaultHNSWThreshold = 2000

	// DefaultIVFPQThreshold is the default vector count at which IVF-PQ is selected over HNSW.
	// Collections below this use HNSW for accuracy/speed balance.
	// Collections at or above this use IVF-PQ for memory efficiency at scale.
	DefaultIVFPQThreshold = 1000000
)

// selectOptimalIndexType chooses the best index type based on collection size.
// Uses the provided thresholds to determine the switching points.
func selectOptimalIndexType(vectorCount int, hnswThreshold, ivfpqThreshold int) IndexType {
	if vectorCount < hnswThreshold {
		// Small collections: use Flat for exact search and simplicity
		return Flat
	} else if vectorCount < ivfpqThreshold {
		// Medium collections: use HNSW for good balance of speed and accuracy
		return HNSW
	} else {
		// Large collections: use IVF-PQ for memory efficiency
		return IVFPQ
	}
}

// newCollection creates a new collection instance
func newCollection(ctx context.Context, name string, storageEngine storage.Engine, metrics *obs.Metrics, writes *writeController, opts ...CollectionOption) (*Collection, error) {
	config := &CollectionConfig{
		Dimension:      768, // Default for common embeddings
		Metric:         CosineDistance,
		IndexType:      HNSW,
		M:              ProductionHNSWM,
		EfConstruction: ProductionHNSWEfConstruction,
		EfSearch:       ProductionHNSWEfSearch,
		NClusters:      100,
		NProbes:        10,
		ML:             1.0 / math.Log(32.0),
		RawVectorStore: "slabby",
		RawStoreCap:    4096,
		// Default memory management settings
		MemoryLimit:    0, // No limit by default
		CachePolicy:    LRUCache,
		EnableMMapping: false, // Disabled by default
		// Default batch configuration
		BatchConfig: DefaultBatchConfig(),
	}

	// Apply options
	for _, opt := range opts {
		if err := opt(config); err != nil {
			return nil, fmt.Errorf("failed to apply collection option: %w", err)
		}
	}

	// Validate configuration
	if err := config.validate(); err != nil {
		return nil, fmt.Errorf("invalid collection config: %w", err)
	}

	if config.AutoIndexSelection {
		hnswThreshold := config.AutoIndexThresholds.HNSWThreshold
		if hnswThreshold == 0 {
			hnswThreshold = DefaultHNSWThreshold
		}
		ivfpqThreshold := config.AutoIndexThresholds.IVFPQThreshold
		if ivfpqThreshold == 0 {
			ivfpqThreshold = DefaultIVFPQThreshold
		}
		config.IndexType = selectOptimalIndexType(0, hnswThreshold, ivfpqThreshold)
	}

	// Sharded collections require explicit opt-in and have restrictions
	if config.Sharded {
		// AutoIndexSelection can switch to IVFPQ which is not supported for sharding
		if config.AutoIndexSelection {
			return nil, fmt.Errorf("sharding is not supported with AutoIndexSelection: automatic index selection can switch to IVFPQ which does not support sharding")
		}

		// Only HNSW and Flat support sharding
		if config.IndexType != HNSW && config.IndexType != Flat {
			return nil, fmt.Errorf("sharding is only supported for HNSW and Flat index types, got: %v", config.IndexType)
		}
	}

	// Convert to LSM config format
	engineConfig := &storage.CollectionConfig{
		Dimension:        config.Dimension,
		Metric:           int(config.Metric),
		IndexType:        int(config.IndexType),
		M:                config.M,
		EfConstruction:   config.EfConstruction,
		EfSearch:         config.EfSearch,
		NClusters:        config.NClusters,
		NProbes:          config.NProbes,
		ML:               config.ML,
		Version:          2,
		RawVectorStore:   config.RawVectorStore,
		RawStoreCap:      config.RawStoreCap,
		IDMapCapacity:    config.IDMapCapacity,
		MetadataSchema:   metadataSchemaToStorage(config.MetadataSchema),
		IndexedFields:    append([]string(nil), config.IndexedFields...),
		SQLIndexes:       sqlIndexesToStorage(config.SQLIndexes),
		SQLIndexedFields: append([]string(nil), config.SQLIndexedFields...),
		GraphEnabled:     config.Graph != nil,
		GraphNamespace:   config.GraphNamespace,
	}

	// Initialize memory manager if memory management is configured
	var memManager memory.MemoryManager
	if config.MemoryLimit > 0 || config.MemoryConfig != nil {
		memConfig := memory.DefaultMemoryConfig()
		if config.MemoryConfig != nil {
			memConfig = *config.MemoryConfig
		}
		if config.MemoryLimit > 0 {
			memConfig.MaxMemory = config.MemoryLimit
		}
		memConfig.EnableMMap = config.EnableMMapping

		memManager = memory.NewManager(memConfig)

		// Start memory monitoring
		if err := memManager.Start(ctx); err != nil {
			return nil, fmt.Errorf("failed to start memory manager: %w", err)
		}
	}

	// Create the collection
	c := &Collection{
		name:          name,
		config:        config,
		writes:        writes,
		metrics:       metrics,
		memoryManager: memManager,
		graph:         config.Graph,
		costModel:     newCollectionCostModelState(nil, 0),
	}

	// Initialize storage and index based on sharding mode
	if config.Sharded {
		// Sharded path: create multiple shard storage collections and indexes
		shardNames := shardStorageNames(name)
		if err := c.initShards(storageEngine, shardNames, engineConfig); err != nil {
			return nil, fmt.Errorf("failed to initialize shards: %w", err)
		}
	} else {
		// Non-sharded path: create single storage collection and index
		var err error
		c.storage, err = storageEngine.CreateCollection(name, engineConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create collection storage: %w", err)
		}

		provider, _ := c.storage.(interface {
			GetByOrdinal(uint32) ([]float32, error)
			Distance([]float32, uint32) (float32, error)
		})

		c.index, err = createIndexForCollection(config, provider)
		if err != nil {
			c.storage.Close()
			return nil, fmt.Errorf("failed to create index: %w", err)
		}

		// Register the index as a memory-mappable component if supported
		if memManager != nil {
			if mappable, ok := c.index.(memory.MemoryMappable); ok {
				if err := memManager.RegisterMemoryMappable(fmt.Sprintf("index_%s", name), mappable); err != nil {
					c.index.Close()
					c.storage.Close()
					return nil, fmt.Errorf("failed to register index for memory management: %w", err)
				}
			}
		}
	}

	return c, nil
}

// newCollectionFromStorage creates a collection instance from existing storage
func newCollectionFromStorage(ctx context.Context, name string, storageCollection storage.Collection, metrics *obs.Metrics, engineConfig *storage.CollectionConfig, writes *writeController, cachedIndex index.Index, graphOverride Graph) (*Collection, error) {
	graphLayer := graphOverride
	if engineConfig.GraphEnabled && graphLayer == nil {
		var err error
		graphLayer, err = NewGraph(GraphConfig{})
		if err != nil {
			return nil, fmt.Errorf("recreate graph for collection %q: %w", name, err)
		}
	}
	// Convert LSM config to libravdb config
	config := &CollectionConfig{
		Dimension:        engineConfig.Dimension,
		Metric:           DistanceMetric(engineConfig.Metric),
		IndexType:        IndexType(engineConfig.IndexType),
		M:                engineConfig.M,
		EfConstruction:   engineConfig.EfConstruction,
		EfSearch:         engineConfig.EfSearch,
		NClusters:        engineConfig.NClusters,
		NProbes:          engineConfig.NProbes,
		ML:               engineConfig.ML,
		Version:          engineConfig.Version,
		RawVectorStore:   engineConfig.RawVectorStore,
		RawStoreCap:      engineConfig.RawStoreCap,
		IDMapCapacity:    engineConfig.IDMapCapacity,
		MetadataSchema:   metadataSchemaFromStorage(engineConfig.MetadataSchema),
		IndexedFields:    append([]string(nil), engineConfig.IndexedFields...),
		SQLIndexes:       sqlIndexesFromStorage(engineConfig.SQLIndexes),
		SQLIndexedFields: append([]string(nil), engineConfig.SQLIndexedFields...),
		Graph:            graphLayer,
		GraphNamespace:   engineConfig.GraphNamespace,
	}
	if config.NClusters <= 0 {
		config.NClusters = 100
	}
	if config.NProbes <= 0 {
		config.NProbes = min(config.NClusters, 10)
	}

	// Use cached index if available (deserialized or rebuilt during recovery),
	// otherwise create a new one and rebuild from storage records.
	//
	// If the cached index fails validation (e.g. corrupted file, bit rot),
	// discard it and rebuild from raw storage — the rebuild is lossless since
	// storage holds the authoritative copy of every record.
	var idx index.Index
	var indexValid bool
	if cachedIndex != nil {
		idx = cachedIndex
		// Quick sanity check: a loaded index must report non-negative size.
		if idx.Size() >= 0 {
			indexValid = true
		}
	}
	if !indexValid {
		if cachedIndex != nil {
			_ = cachedIndex.Close()
			cachedIndex = nil
		}
		provider, _ := storageCollection.(interface {
			GetByOrdinal(uint32) ([]float32, error)
			Distance([]float32, uint32) (float32, error)
		})
		var err error
		idx, err = createIndexForCollection(config, provider)
		if err != nil {
			return nil, fmt.Errorf("failed to create index: %w", err)
		}
	}

	// Initialize memory manager if memory management is configured
	var memManager memory.MemoryManager
	if config.MemoryLimit > 0 || config.MemoryConfig != nil {
		memConfig := memory.DefaultMemoryConfig()
		if config.MemoryConfig != nil {
			memConfig = *config.MemoryConfig
		}
		if config.MemoryLimit > 0 {
			memConfig.MaxMemory = config.MemoryLimit
		}
		memConfig.EnableMMap = config.EnableMMapping

		memManager = memory.NewManager(memConfig)

		// Register the index as a memory-mappable component if supported
		if mappable, ok := idx.(memory.MemoryMappable); ok {
			if err := memManager.RegisterMemoryMappable(fmt.Sprintf("index_%s", name), mappable); err != nil {
				if !indexValid && cachedIndex == nil {
					idx.Close()
				}
				storageCollection.Close()
				return nil, fmt.Errorf("failed to register index for memory management: %w", err)
			}
		}

		// Start memory monitoring
		if err := memManager.Start(ctx); err != nil {
			if !indexValid && cachedIndex == nil {
				idx.Close()
			}
			storageCollection.Close()
			return nil, fmt.Errorf("failed to start memory manager: %w", err)
		}
	}

	collection := &Collection{
		name:          name,
		config:        config,
		index:         idx,
		storage:       storageCollection,
		writes:        writes,
		metrics:       metrics,
		memoryManager: memManager,
		graph:         config.Graph,
		costModel:     newCollectionCostModelState(engineConfig.CostModelStats, engineConfig.DataLSN),
	}

	// Rebuild index from storage data (skipped if cached index was used).
	if cachedIndex == nil {
		if err := collection.rebuildIndex(ctx); err != nil {
			return nil, fmt.Errorf("failed to rebuild index: %w", err)
		}
	} else if hasMeta, ok := idx.(interface{ HasDeserializedMeta() bool }); ok && hasMeta.HasDeserializedMeta() {
		// IVF-PQ two-phase deserialization: centroids and codebooks were
		// restored by DeserializeFromBytes, but cluster entries must be
		// populated from live storage records. Only entered when the
		// index has pending deserialized metadata (never for HNSW, Flat,
		// or rebuilt IVF-PQ where the bridge already fully populated).
		if populator, ok := idx.(interface {
			PopulateEntriesFromStorage(interface {
				IterateEntries(fn func(id string, ordinal uint32, vector []float32, metadata map[string]interface{}) error) error
			}) error
		}); ok {
			provider := storageEntryProvider{col: storageCollection}
			if err := populator.PopulateEntriesFromStorage(provider); err != nil {
				// IVF-PQ cluster entries failed to load. Close the partially-
				// deserialized index and rebuild from scratch. Storage is the
				// authoritative copy — the rebuild is lossless.
				_ = idx.Close()
				var rebuildErr error
				provider, _ := storageCollection.(interface {
					GetByOrdinal(uint32) ([]float32, error)
					Distance([]float32, uint32) (float32, error)
				})
				collection.index, rebuildErr = createIndexForCollection(config, provider)
				if rebuildErr != nil {
					return nil, fmt.Errorf("failed to create index for rebuild: %w", rebuildErr)
				}
				if rebuildErr := collection.rebuildIndex(ctx); rebuildErr != nil {
					return nil, fmt.Errorf("failed to rebuild index from storage: %w", rebuildErr)
				}
				return collection, nil
			}
		}
	}
	// All other cached-index cases (HNSW, Flat, rebuilt IVF-PQ):
	// the bridge already fully populated the index during recovery.

	return collection, nil
}

// storageEntryProvider adapts a storage.Collection to ivfpq.EntryProvider.
type storageEntryProvider struct {
	col storage.Collection
}

func (p storageEntryProvider) IterateEntries(fn func(id string, ordinal uint32, vector []float32, metadata map[string]interface{}) error) error {
	return p.col.Iterate(context.Background(), func(entry *index.VectorEntry) error {
		return fn(entry.ID, entry.Ordinal, entry.Vector, entry.Metadata)
	})
}

// newShardedCollectionFromStorage creates a sharded collection from existing shard storages
func newShardedCollectionFromStorage(ctx context.Context, name string, shardStorages []storage.Collection, engineConfig *storage.CollectionConfig, metrics *obs.Metrics, writes *writeController, graphOverride Graph) (*Collection, error) {
	graphLayer := graphOverride
	if engineConfig.GraphEnabled && graphLayer == nil {
		var err error
		graphLayer, err = NewGraph(GraphConfig{})
		if err != nil {
			return nil, fmt.Errorf("recreate graph for collection %q: %w", name, err)
		}
	}
	// Convert LSM config to libravdb config
	config := &CollectionConfig{
		Dimension:        engineConfig.Dimension,
		Metric:           DistanceMetric(engineConfig.Metric),
		IndexType:        IndexType(engineConfig.IndexType),
		M:                engineConfig.M,
		EfConstruction:   engineConfig.EfConstruction,
		EfSearch:         engineConfig.EfSearch,
		NClusters:        engineConfig.NClusters,
		NProbes:          engineConfig.NProbes,
		ML:               engineConfig.ML,
		Version:          engineConfig.Version,
		RawVectorStore:   engineConfig.RawVectorStore,
		RawStoreCap:      engineConfig.RawStoreCap,
		IDMapCapacity:    engineConfig.IDMapCapacity,
		MetadataSchema:   metadataSchemaFromStorage(engineConfig.MetadataSchema),
		IndexedFields:    append([]string(nil), engineConfig.IndexedFields...),
		SQLIndexes:       sqlIndexesFromStorage(engineConfig.SQLIndexes),
		SQLIndexedFields: append([]string(nil), engineConfig.SQLIndexedFields...),
		Graph:            graphLayer,
		GraphNamespace:   engineConfig.GraphNamespace,
		Sharded:          true, // Mark as sharded so lifecycle methods work correctly
	}
	if config.NClusters <= 0 {
		config.NClusters = 100
	}
	if config.NProbes <= 0 {
		config.NProbes = min(config.NClusters, 10)
	}

	// Create the collection and initialize shards
	c := &Collection{
		name:      name,
		config:    config,
		writes:    writes,
		metrics:   metrics,
		graph:     config.Graph,
		costModel: newCollectionCostModelState(engineConfig.CostModelStats, engineConfig.DataLSN),
	}

	// Initialize shards from loaded storages
	c.shards = make([]shard, shardCount)
	for i := 0; i < shardCount; i++ {
		provider, _ := shardStorages[i].(interface {
			GetByOrdinal(uint32) ([]float32, error)
			Distance([]float32, uint32) (float32, error)
		})

		idx, err := createIndexForCollection(config, provider)
		if err != nil {
			// Close already-opened shards and their indexes
			for j := 0; j < i; j++ {
				if c.shards[j].index != nil {
					c.shards[j].index.Close()
				}
				shardStorages[j].Close()
			}
			return nil, fmt.Errorf("failed to create shard %d index: %w", i, err)
		}

		c.shards[i] = shard{
			name:    shardStorageNames(name)[i],
			storage: shardStorages[i],
			index:   idx,
		}
	}

	// Rebuild each shard's index from its storage
	for i := range c.shards {
		if err := c.rebuildShardIndex(ctx, i); err != nil {
			return nil, fmt.Errorf("failed to rebuild shard %d index: %w", i, err)
		}
	}

	return c, nil
}

// rebuildShardIndex rebuilds a single shard's index from its storage
func (c *Collection) rebuildShardIndex(ctx context.Context, shardIdx int) error {
	shard := &c.shards[shardIdx]
	vectors, err := c.getAllVectorsFromShard(ctx, shardIdx)
	if err != nil {
		return err
	}
	if err := prepareIndexForEntries(ctx, shard.index, c.config.Metric, vectors); err != nil {
		return err
	}
	return insertEntriesIntoIndex(ctx, shard.index, c.config.Metric, vectors)
}

// getAllVectorsFromShard returns all vectors from a specific shard's storage
func (c *Collection) getAllVectorsFromShard(ctx context.Context, shardIdx int) ([]*index.VectorEntry, error) {
	var entries []*index.VectorEntry
	err := c.shards[shardIdx].storage.Iterate(ctx, func(entry *index.VectorEntry) error {
		entries = append(entries, entry)
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("failed to iterate shard storage: %w", err)
	}
	return entries, nil
}

// rebuildIndex rebuilds the index from storage data
func (c *Collection) rebuildIndex(ctx context.Context) error {
	vectors, err := c.getAllVectors(ctx)
	if err != nil {
		return err
	}
	if err := prepareIndexForEntries(ctx, c.index, c.config.Metric, vectors); err != nil {
		return err
	}
	return insertEntriesIntoIndex(ctx, c.index, c.config.Metric, vectors)
}

// Insert adds or updates a vector in the collection
func (c *Collection) Insert(ctx context.Context, id string, vector []float32, metadata map[string]interface{}) (err error) {
	defer func() {
		if err == nil {
			c.addToMetadataIndex(id, metadata)
			c.markMetadataIndexDirty()
		}
	}()
	// Preflight: validate dimension before acquiring write permit or mutex
	if len(vector) != c.config.Dimension {
		return fmt.Errorf("vector dimension %d does not match collection dimension %d",
			len(vector), c.config.Dimension)
	}

	// Preflight: apply column DEFAULTs for omitted columns. A nil metadata map
	// still needs to receive defaults, so materialize a private map first.
	metadata = c.metadataWithDefaults(metadata)

	// Preflight: enforce all declared NOT NULL columns after defaults.
	if err := c.validateNotNullConstraints(metadata); err != nil {
		return err
	}

	// Preflight: validate CHECK constraints.
	if err := c.validateCheckConstraints(metadata); err != nil {
		return err
	}

	// Preflight: validate foreign key constraints.
	if err := c.validateForeignKeys(ctx, id, metadata); err != nil {
		return err
	}

	// Preflight: validate UNIQUE constraints.
	if err := c.validateUniqueConstraints(ctx, id, metadata); err != nil {
		return err
	}

	// Stage entry before acquiring lock (no shared state accessed yet)
	storageEntry := &index.VectorEntry{
		ID:       id,
		Vector:   vector,
		Metadata: metadata,
	}

	release, err := c.acquireWrite(ctx)
	if err != nil {
		return err
	}
	defer release()

	c.mu.RLock()
	if c.closed {
		c.mu.RUnlock()
		return ErrCollectionClosed
	}

	// Non-sharded path: use single storage and index
	if c.shards == nil {
		// Keep the collection read lock until the mutation guard has released
		// its off-heap slot. Close takes c.mu exclusively before freeing that
		// arena, so defer registration order is a lifetime invariant here:
		// mutation.unlock must run before c.mu.RUnlock.
		defer c.mu.RUnlock()
		mutation := c.lockMutationID(id)
		defer mutation.unlock()

		if exists, err := c.storage.Exists(ctx, id); err != nil {
			return fmt.Errorf("failed to check existing vector: %w", err)
		} else if exists {
			return fmt.Errorf("failed to insert into index: node with ID '%s' already exists", id)
		}
		asyncReserved := false
		if c.asyncIndex != nil {
			c.asyncMutation.RLock()
			defer c.asyncMutation.RUnlock()
			if err := c.asyncIndex.reserve(ctx, 1); err != nil {
				return fmt.Errorf("failed to reserve asynchronous index capacity: %w", err)
			}
			asyncReserved = true
			defer func() {
				if asyncReserved {
					c.asyncIndex.cancelReservation(1)
				}
			}()
		}

		if err := c.storage.AssignOrdinals(ctx, []*index.VectorEntry{storageEntry}); err != nil {
			return err
		}

		type transactionStarter interface {
			BeginTxn() *graph.Txn
		}

		if len(c.insertHooks) > 0 {
			var txn *graph.Txn
			if c.graph != nil {
				if starter, ok := c.graph.(transactionStarter); ok {
					txn = starter.BeginTxn()
				}
			}
			if txn == nil {
				txn = &graph.Txn{}
			}

			for _, hook := range c.insertHooks {
				if err := hook(txn, uint64(storageEntry.Ordinal), vector, storageEntry.Metadata); err != nil {
					return fmt.Errorf("insert hook failed: %w", err)
				}
			}
			if c.graph != nil && txn.ID != 0 {
				if err := txn.Commit(ctx); err != nil {
					return fmt.Errorf("failed to commit graph transaction: %w", err)
				}
			}
		}

		if c.asyncIndex != nil {
			durable, err := c.asyncIndex.storage.InsertDurableRange(ctx, storageEntry)
			if err != nil {
				return fmt.Errorf("failed to write to storage: %w", err)
			}
			c.asyncIndex.commitOne(storageEntry, durable)
			asyncReserved = false
			if c.metrics != nil {
				c.metrics.VectorInserts.Inc()
			}
			return nil
		}

		if err := c.storage.Insert(ctx, storageEntry); err != nil {
			return fmt.Errorf("failed to write to storage: %w", err)
		}

		if err := c.index.Insert(ctx, entryForIndex(c.config.Metric, storageEntry)); err != nil {
			if delErr := c.storage.Delete(ctx, id); delErr != nil {
				return fmt.Errorf("failed to insert into index: %w (CRITICAL: rollback storage.Delete failed: %v)", err, delErr)
			}
			return fmt.Errorf("failed to insert into index: %w", err)
		}

		// Update metrics after unlock (Prometheus counters are concurrency-safe)
		if c.metrics != nil {
			c.metrics.VectorInserts.Inc()
		}
		return nil
	}

	// Sharded path: route to the correct shard for this ID
	shard := c.getShard(id)
	shard.mu.Lock()
	defer c.mu.RUnlock()
	defer shard.mu.Unlock()

	if exists, err := shard.storage.Exists(ctx, id); err != nil {
		return fmt.Errorf("failed to check existing vector: %w", err)
	} else if exists {
		return fmt.Errorf("failed to insert into index: node with ID '%s' already exists", id)
	}

	if err := shard.storage.AssignOrdinals(ctx, []*index.VectorEntry{storageEntry}); err != nil {
		return err
	}

	if len(c.insertHooks) > 0 {
		type transactionStarter interface {
			BeginTxn() *graph.Txn
		}
		var txn *graph.Txn
		if c.graph != nil {
			if starter, ok := c.graph.(transactionStarter); ok {
				txn = starter.BeginTxn()
			}
		}
		if txn == nil {
			txn = &graph.Txn{}
		}

		for _, hook := range c.insertHooks {
			if err := hook(txn, uint64(storageEntry.Ordinal), vector, storageEntry.Metadata); err != nil {
				return fmt.Errorf("insert hook failed: %w", err)
			}
		}
		if c.graph != nil && txn.ID != 0 {
			if err := txn.Commit(ctx); err != nil {
				return fmt.Errorf("failed to commit graph transaction: %w", err)
			}
		}
	}

	if err := shard.storage.Insert(ctx, storageEntry); err != nil {
		return fmt.Errorf("failed to write to storage: %w", err)
	}
	if err := shard.index.Insert(ctx, entryForIndex(c.config.Metric, storageEntry)); err != nil {
		if delErr := shard.storage.Delete(ctx, id); delErr != nil {
			return fmt.Errorf("failed to insert into index: %w (CRITICAL: rollback storage.Delete failed: %v)", err, delErr)
		}
		return fmt.Errorf("failed to insert into index: %w", err)
	}

	// Update metrics after unlock (Prometheus counters are concurrency-safe)
	if c.metrics != nil {
		c.metrics.VectorInserts.Inc()
	}

	return nil
}

func (c *Collection) insertBatch(ctx context.Context, entries []*index.VectorEntry) (err error) {
	defer func() {
		if err == nil && len(entries) > 0 {
			c.markMetadataIndexDirty()
		}
	}()
	// Preflight: reject nil/empty batch before acquiring write permit
	if len(entries) == 0 {
		return nil
	}

	// Preflight: check for nil entries
	for i, entry := range entries {
		if entry == nil {
			return fmt.Errorf("entry at index %d is nil", i)
		}
	}

	// Preflight: duplicate IDs within this batch (pure CPU, no shared state)
	seen := make(map[string]struct{}, len(entries))
	for _, entry := range entries {
		if _, ok := seen[entry.ID]; ok {
			return fmt.Errorf("failed to insert into index: node with ID '%s' already exists", entry.ID)
		}
		seen[entry.ID] = struct{}{}
	}

	// Preflight: vector dimension validation (pure CPU, no shared state)
	dimension := c.config.Dimension
	if dimension > 0 {
		for _, entry := range entries {
			if len(entry.Vector) != dimension {
				return fmt.Errorf("vector dimension %d does not match collection dimension %d",
					len(entry.Vector), dimension)
			}
		}
	}

	// Batch inserts use a lower-level storage path than Collection.Insert, so
	// repeat the schema-constraint preflight here. Without this check SQL
	// INSERT (which is intentionally batched) could bypass FK and UNIQUE
	// enforcement while direct Go inserts correctly rejected the same row.
	for _, entry := range entries {
		entry.Metadata = c.metadataWithDefaults(entry.Metadata)
		if err := c.validateNotNullConstraints(entry.Metadata); err != nil {
			return err
		}
		if err := c.validateCheckConstraints(entry.Metadata); err != nil {
			return err
		}
		if err := c.validateForeignKeys(ctx, entry.ID, entry.Metadata); err != nil {
			return err
		}
		if err := c.validateUniqueConstraints(ctx, entry.ID, entry.Metadata); err != nil {
			return err
		}
	}

	release, err := c.acquireWrite(ctx)
	if err != nil {
		return err
	}
	defer release()

	c.mu.RLock()
	closed := c.closed
	shards := c.shards
	// Snapshot backend handles before releasing the read lock so concurrent
	// Close or switchIndexType cannot swap them under us.
	storage := c.storage
	index := c.index
	if closed {
		c.mu.RUnlock()
		return ErrCollectionClosed
	}

	if shards != nil {
		defer c.mu.RUnlock()
		return c.insertBatchSharded(ctx, entries, shards)
	}
	// Non-sharded path (fallback - should not reach here for supported indexes).
	// Retain c.mu.RLock until the mutation guard is released: Close must not
	// free the off-heap mutation table while this batch owns slots in it.
	defer c.mu.RUnlock()
	mutation := c.lockMutationEntries(entries)
	defer mutation.unlock()
	// Check existence against persisted storage
	for _, entry := range entries {
		exists, err := storage.Exists(ctx, entry.ID)
		if err != nil {
			return fmt.Errorf("failed to check existing vector: %w", err)
		}
		if exists {
			return fmt.Errorf("failed to insert into index: node with ID '%s' already exists", entry.ID)
		}
	}

	if c.asyncIndex != nil {
		c.asyncMutation.RLock()
		defer c.asyncMutation.RUnlock()
		if err := c.asyncIndex.reserve(ctx, len(entries)); err != nil {
			return fmt.Errorf("failed to reserve asynchronous index capacity: %w", err)
		}
		durable, err := c.asyncIndex.storage.InsertBatchDurableRange(ctx, entries)
		if err != nil {
			c.asyncIndex.cancelReservation(len(entries))
			return fmt.Errorf("failed to write batch to storage: %w", err)
		}
		c.asyncIndex.commit(entries, durable)
		if c.metrics != nil {
			c.metrics.VectorInserts.Add(float64(len(entries)))
		}
		return nil
	}

	if err := storage.InsertBatch(ctx, entries); err != nil {
		return fmt.Errorf("failed to write batch to storage: %w", err)
	}
	if err := prepareIndexForEntries(ctx, index, c.config.Metric, entries); err != nil {
		var rollbackErrs []error
		for _, storedEntry := range entries {
			if delErr := storage.Delete(ctx, storedEntry.ID); delErr != nil {
				rollbackErrs = append(rollbackErrs, delErr)
			}
		}
		if len(rollbackErrs) > 0 {
			return fmt.Errorf("failed to prepare index: %w (CRITICAL: %d rollback failures, e.g., %v)", err, len(rollbackErrs), rollbackErrs[0])
		}
		return fmt.Errorf("failed to prepare index for batch insert: %w", err)
	}
	if err := insertEntriesIntoIndex(ctx, index, c.config.Metric, entries); err != nil {
		var rollbackErrs []error
		for _, storedEntry := range entries {
			if delErr := storage.Delete(ctx, storedEntry.ID); delErr != nil {
				rollbackErrs = append(rollbackErrs, delErr)
			}
		}
		if len(rollbackErrs) > 0 {
			return fmt.Errorf("failed to insert into index: %w (CRITICAL: %d rollback failures, e.g., %v)", err, len(rollbackErrs), rollbackErrs[0])
		}
		return fmt.Errorf("failed to insert into index: %w", err)
	}

	// Update metrics after unlock (Prometheus counters are concurrency-safe)
	if c.metrics != nil {
		c.metrics.VectorInserts.Add(float64(len(entries)))
	}

	return nil
}

func (c *Collection) insertBatchSharded(ctx context.Context, entries []*index.VectorEntry, shards []shard) error {
	// Group entries by shard
	shardGroups := groupEntriesByShard(entries)

	// Parallelize across shards with bounded concurrency
	var wg sync.WaitGroup
	errCh := make(chan error, len(shardGroups))
	maxConcurrency := 4
	sem := make(chan struct{}, maxConcurrency)

	for shardIdx, shardEntries := range shardGroups {
		if len(shardEntries) == 0 {
			continue
		}

		shardRef := &shards[shardIdx]
		wg.Add(1)
		go func(shardIdx int, s *shard, shardEntries []*index.VectorEntry) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			s.mu.Lock()
			defer s.mu.Unlock()

			// Check existence in this shard's storage
			for _, entry := range shardEntries {
				exists, err := s.storage.Exists(ctx, entry.ID)
				if err != nil {
					errCh <- fmt.Errorf("failed to check existing vector: %w", err)
					return
				}
				if exists {
					errCh <- fmt.Errorf("failed to insert into index: node with ID '%s' already exists", entry.ID)
					return
				}
			}

			if err := s.storage.InsertBatch(ctx, shardEntries); err != nil {
				errCh <- fmt.Errorf("failed to write batch to storage: %w", err)
				return
			}
			if err := prepareIndexForEntries(ctx, s.index, c.config.Metric, shardEntries); err != nil {
				var rollbackErrs []error
				for _, storedEntry := range shardEntries {
					if delErr := s.storage.Delete(ctx, storedEntry.ID); delErr != nil {
						rollbackErrs = append(rollbackErrs, delErr)
					}
				}
				if len(rollbackErrs) > 0 {
					errCh <- fmt.Errorf("failed to prepare index: %w (CRITICAL: %d rollback failures, e.g., %v)", err, len(rollbackErrs), rollbackErrs[0])
				} else {
					errCh <- fmt.Errorf("failed to prepare index for batch insert: %w", err)
				}
				return
			}
			if err := insertEntriesIntoIndex(ctx, s.index, c.config.Metric, shardEntries); err != nil {
				var rollbackErrs []error
				for _, storedEntry := range shardEntries {
					if delErr := s.storage.Delete(ctx, storedEntry.ID); delErr != nil {
						rollbackErrs = append(rollbackErrs, delErr)
					}
				}
				if len(rollbackErrs) > 0 {
					errCh <- fmt.Errorf("failed to insert into index: %w (CRITICAL: %d rollback failures, e.g., %v)", err, len(rollbackErrs), rollbackErrs[0])
				} else {
					errCh <- fmt.Errorf("failed to insert into index: %w", err)
				}
				return
			}
		}(shardIdx, shardRef, shardEntries)
	}

	wg.Wait()
	close(errCh)

	// Collect errors
	var errs []error
	for err := range errCh {
		errs = append(errs, err)
	}

	if len(errs) > 0 {
		return fmt.Errorf("shard batch insert errors: %v", errs)
	}

	return nil
}

func (c *Collection) rollbackBatchIndex(ctx context.Context, ids []string) {
	for i := len(ids) - 1; i >= 0; i-- {
		_ = c.index.Delete(ctx, ids[i])
	}
}

// Update modifies an existing vector in the collection
func (c *Collection) Update(ctx context.Context, id string, vector []float32, metadata map[string]interface{}) (err error) {
	// Resolve the complete post-update metadata first. Constraint checks must
	// see unchanged FK columns as well as columns supplied by this partial
	// UPDATE, and ON UPDATE actions need the old and new parent tuple.
	existing, getErr := c.Get(ctx, id)
	if getErr != nil {
		return getErr
	}
	oldMetadata := cloneMetadata(existing.Metadata)
	newMetadata := cloneMetadata(oldMetadata)
	for k, v := range metadata {
		if newMetadata == nil {
			newMetadata = make(map[string]interface{})
		}
		newMetadata[k] = cloneMetadataValue(v)
	}

	// Preflight: enforce NOT NULL constraints against the complete post-update
	// row before evaluating CHECK/FK/UNIQUE constraints.
	if err := c.validateNotNullConstraints(newMetadata); err != nil {
		return err
	}

	// Preflight: validate CHECK constraints against post-update row.
	if err := c.validateCheckConstraints(newMetadata); err != nil {
		return err
	}

	// Preflight: validate foreign key constraints with new values.
	if err := c.validateForeignKeys(ctx, id, newMetadata); err != nil {
		return err
	}

	// Preflight: validate UNIQUE constraints with new values.
	if err := c.validateUniqueConstraints(ctx, id, newMetadata); err != nil {
		return err
	}

	var updateCascades []cascadeOp
	updateCascades, err = c.collectUpdateCascades(ctx, id, id, oldMetadata, newMetadata)
	if err != nil {
		return err
	}
	defer func() {
		if err == nil {
			if oldMetadata != nil {
				c.removeFromMetadataIndex(id, oldMetadata)
			}
			c.addToMetadataIndex(id, newMetadata)
			c.markMetadataIndexDirty()
			for _, op := range updateCascades {
				c.executeCascadeMutation(ctx, op)
			}
		}
	}()

	// Fetch existing record early for ON UPDATE cascade check.
	release, err := c.acquireWrite(ctx)
	if err != nil {
		return err
	}
	defer release()

	unlockAsync, err := c.lockAsyncMutation(ctx)
	if err != nil {
		return fmt.Errorf("failed to flush asynchronous index before update: %w", err)
	}
	defer unlockAsync()

	c.mu.RLock()
	if c.closed {
		c.mu.RUnlock()
		return ErrCollectionClosed
	}

	// Validate input
	if id == "" {
		c.mu.RUnlock()
		return fmt.Errorf("vector ID cannot be empty")
	}

	if vector != nil && len(vector) != c.config.Dimension {
		c.mu.RUnlock()
		return fmt.Errorf("vector dimension %d does not match collection dimension %d",
			len(vector), c.config.Dimension)
	}

	// Non-sharded path
	if c.shards == nil {
		// The guard releases before the collection read lock (LIFO defers),
		// preventing Close from freeing its off-heap table early.
		defer c.mu.RUnlock()
		mutation := c.lockMutationID(id)
		defer mutation.unlock()
		return c.updateNonSharded(ctx, id, vector, metadata)
	}

	// Sharded path: route to the correct shard for this ID
	defer c.mu.RUnlock()
	return c.updateSharded(ctx, id, vector, metadata)
}

func (c *Collection) updateNonSharded(ctx context.Context, id string, vector []float32, metadata map[string]interface{}) error {
	existingEntry, err := c.storage.Get(ctx, id)
	if err != nil {
		return fmt.Errorf("vector with ID %s not found", id)
	}

	// Prepare the updated entry
	updatedEntry := &index.VectorEntry{
		ID:       id,
		Vector:   vector,
		Metadata: metadata,
	}

	// Use existing data for partial updates
	if existingEntry != nil {
		if vector == nil {
			updatedEntry.Vector = existingEntry.Vector
		}
		if metadata == nil {
			updatedEntry.Metadata = existingEntry.Metadata
		} else if existingEntry.Metadata != nil {
			// Merge metadata (new values override existing ones)
			mergedMetadata := make(map[string]interface{})
			for k, v := range existingEntry.Metadata {
				mergedMetadata[k] = v
			}
			for k, v := range metadata {
				mergedMetadata[k] = v
			}
			updatedEntry.Metadata = mergedMetadata
		}
	}
	updatedEntry.Ordinal = existingEntry.Ordinal
	if deleter, ok := c.index.(interface {
		DeleteByOrdinal(context.Context, uint32) error
	}); ok {
		if err := deleter.DeleteByOrdinal(ctx, updatedEntry.Ordinal); err != nil {
			return fmt.Errorf("failed to delete existing vector from index: %w", err)
		}
	} else if err := c.index.Delete(ctx, id); err != nil {
		return fmt.Errorf("failed to delete existing vector from index: %w", err)
	}

	if err := c.storage.Insert(ctx, updatedEntry); err != nil {
		return fmt.Errorf("failed to write update to storage: %w", err)
	}
	if err := c.index.Insert(ctx, entryForIndex(c.config.Metric, updatedEntry)); err != nil {
		return fmt.Errorf("failed to insert updated vector into index: %w", err)
	}

	// Update metrics
	if c.metrics != nil {
		c.metrics.VectorUpdates.Inc()
	}

	// Incrementally update metadata posting index.
	c.removeFromMetadataIndex(id, existingEntry.Metadata)
	c.addToMetadataIndex(id, updatedEntry.Metadata)

	return nil
}

// Upsert writes a record regardless of whether it exists, replacing if it does.
func (c *Collection) Upsert(ctx context.Context, id string, vector []float32, metadata map[string]interface{}) (err error) {
	defer func() {
		if err == nil {
			c.markMetadataIndexDirty()
		}
	}()
	if len(vector) != c.config.Dimension {
		return fmt.Errorf("vector dimension %d does not match collection dimension %d",
			len(vector), c.config.Dimension)
	}
	metadata = c.metadataWithDefaults(metadata)
	if err := c.validateNotNullConstraints(metadata); err != nil {
		return err
	}
	if err := c.validateCheckConstraints(metadata); err != nil {
		return err
	}
	if err := c.validateForeignKeys(ctx, id, metadata); err != nil {
		return err
	}
	if err := c.validateUniqueConstraints(ctx, id, metadata); err != nil {
		return err
	}
	release, err := c.acquireWrite(ctx)
	if err != nil {
		return err
	}
	defer release()

	unlockAsync, err := c.lockAsyncMutation(ctx)
	if err != nil {
		return fmt.Errorf("failed to flush asynchronous index before upsert: %w", err)
	}
	defer unlockAsync()

	c.mu.RLock()
	if c.closed {
		c.mu.RUnlock()
		return ErrCollectionClosed
	}

	if c.shards == nil {
		// Keep the collection alive through mutation-slot release.
		defer c.mu.RUnlock()
		mutation := c.lockMutationID(id)
		defer mutation.unlock()
		return c.upsertNonSharded(ctx, id, vector, metadata)
	}

	defer c.mu.RUnlock()
	return c.upsertSharded(ctx, id, vector, metadata)
}

func (c *Collection) upsertNonSharded(ctx context.Context, id string, vector []float32, metadata map[string]interface{}) error {
	exists, err := c.storage.Exists(ctx, id)
	if err != nil {
		return fmt.Errorf("failed to check existing vector: %w", err)
	}

	entry := &index.VectorEntry{
		ID:       id,
		Vector:   vector,
		Metadata: metadata,
	}

	if !exists {
		if err := c.storage.Insert(ctx, entry); err != nil {
			return fmt.Errorf("failed to write to storage: %w", err)
		}
		if err := c.index.Insert(ctx, entryForIndex(c.config.Metric, entry)); err != nil {
			if delErr := c.storage.Delete(ctx, id); delErr != nil {
				return fmt.Errorf("failed to insert into index: %w (CRITICAL: rollback storage.Delete failed: %v)", err, delErr)
			}
			return fmt.Errorf("failed to insert into index: %w", err)
		}
		if c.metrics != nil {
			c.metrics.VectorUpserts.Inc()
		}
		return nil
	}

	existingEntry, err := c.storage.Get(ctx, id)
	if err != nil {
		return fmt.Errorf("vector with ID %s not found", id)
	}
	entry.Ordinal = existingEntry.Ordinal

	if deleter, ok := c.index.(interface {
		DeleteByOrdinal(context.Context, uint32) error
	}); ok {
		if err := deleter.DeleteByOrdinal(ctx, entry.Ordinal); err != nil {
			return fmt.Errorf("failed to delete existing vector from index: %w", err)
		}
	} else if err := c.index.Delete(ctx, id); err != nil {
		return fmt.Errorf("failed to delete existing vector from index: %w", err)
	}

	if err := c.storage.Insert(ctx, entry); err != nil {
		return fmt.Errorf("failed to write to storage: %w", err)
	}
	if err := c.index.Insert(ctx, entryForIndex(c.config.Metric, entry)); err != nil {
		if rebuildErr := c.rebuildIndex(ctx); rebuildErr != nil {
			return fmt.Errorf("index insert failed: %w; rebuild after index insert also failed: %v", err, rebuildErr)
		}
		return fmt.Errorf("index insert failed, index rebuilt from storage: %w", err)
	}

	if c.metrics != nil {
		c.metrics.VectorUpserts.Inc()
	}

	return nil
}

func (c *Collection) updateSharded(ctx context.Context, id string, vector []float32, metadata map[string]interface{}) error {
	mutation := c.lockMutationID(id)
	defer mutation.unlock()

	shard := c.getShard(id)
	shard.mu.Lock()
	defer shard.mu.Unlock()

	existingEntry, err := shard.storage.Get(ctx, id)
	if err != nil {
		return fmt.Errorf("vector with ID %s not found", id)
	}

	// Prepare the updated entry
	updatedEntry := &index.VectorEntry{
		ID:       id,
		Vector:   vector,
		Metadata: metadata,
	}

	// Use existing data for partial updates
	if existingEntry != nil {
		if vector == nil {
			updatedEntry.Vector = existingEntry.Vector
		}
		if metadata == nil {
			updatedEntry.Metadata = existingEntry.Metadata
		} else if existingEntry.Metadata != nil {
			// Merge metadata (new values override existing ones)
			mergedMetadata := make(map[string]interface{})
			for k, v := range existingEntry.Metadata {
				mergedMetadata[k] = v
			}
			for k, v := range metadata {
				mergedMetadata[k] = v
			}
			updatedEntry.Metadata = mergedMetadata
		}
	}
	updatedEntry.Ordinal = existingEntry.Ordinal
	if deleter, ok := shard.index.(interface {
		DeleteByOrdinal(context.Context, uint32) error
	}); ok {
		if err := deleter.DeleteByOrdinal(ctx, updatedEntry.Ordinal); err != nil {
			return fmt.Errorf("failed to delete existing vector from index: %w", err)
		}
	} else if err := shard.index.Delete(ctx, id); err != nil {
		return fmt.Errorf("failed to delete existing vector from index: %w", err)
	}

	if err := shard.storage.Insert(ctx, updatedEntry); err != nil {
		return fmt.Errorf("failed to write update to storage: %w", err)
	}
	if err := shard.index.Insert(ctx, entryForIndex(c.config.Metric, updatedEntry)); err != nil {
		return fmt.Errorf("failed to insert updated vector into index: %w", err)
	}

	// Update metrics
	if c.metrics != nil {
		c.metrics.VectorUpdates.Inc()
	}

	return nil
}

func (c *Collection) upsertSharded(ctx context.Context, id string, vector []float32, metadata map[string]interface{}) error {
	mutation := c.lockMutationID(id)
	defer mutation.unlock()

	shard := c.getShard(id)
	shard.mu.Lock()
	defer shard.mu.Unlock()

	exists, err := shard.storage.Exists(ctx, id)
	if err != nil {
		return fmt.Errorf("failed to check existing vector: %w", err)
	}

	entry := &index.VectorEntry{
		ID:       id,
		Vector:   vector,
		Metadata: metadata,
	}

	if !exists {
		if err := shard.storage.Insert(ctx, entry); err != nil {
			return fmt.Errorf("failed to write to storage: %w", err)
		}
		if err := shard.index.Insert(ctx, entryForIndex(c.config.Metric, entry)); err != nil {
			if delErr := shard.storage.Delete(ctx, id); delErr != nil {
				return fmt.Errorf("failed to insert into index: %w (CRITICAL: rollback storage.Delete failed: %v)", err, delErr)
			}
			return fmt.Errorf("failed to insert into index: %w", err)
		}
		if c.metrics != nil {
			c.metrics.VectorUpserts.Inc()
		}
		return nil
	}

	existingEntry, err := shard.storage.Get(ctx, id)
	if err != nil {
		return fmt.Errorf("vector with ID %s not found", id)
	}
	entry.Ordinal = existingEntry.Ordinal

	if deleter, ok := shard.index.(interface {
		DeleteByOrdinal(context.Context, uint32) error
	}); ok {
		if err := deleter.DeleteByOrdinal(ctx, entry.Ordinal); err != nil {
			return fmt.Errorf("failed to delete existing vector from index: %w", err)
		}
	} else if err := shard.index.Delete(ctx, id); err != nil {
		return fmt.Errorf("failed to delete existing vector from index: %w", err)
	}

	if err := shard.storage.Insert(ctx, entry); err != nil {
		return fmt.Errorf("failed to write to storage: %w", err)
	}
	if err := shard.index.Insert(ctx, entryForIndex(c.config.Metric, entry)); err != nil {
		shardIdx := shardForID(id)
		if rebuildErr := c.rebuildShardIndex(ctx, shardIdx); rebuildErr != nil {
			return fmt.Errorf("index insert failed: %w; rebuild after index insert also failed: %v", err, rebuildErr)
		}
		return fmt.Errorf("index insert failed, index rebuilt from storage: %w", err)
	}

	if c.metrics != nil {
		c.metrics.VectorUpserts.Inc()
	}

	return nil
}

// Delete removes a vector from the collection
func (c *Collection) Delete(ctx context.Context, id string) (err error) {
	// All database-owned collections use the combined transaction path. This
	// keeps FK cascades, record tombstones, deprecated delete-hook graph ops,
	// and GRAPH_NODES edge drops in one durable WAL transaction regardless of
	// whether the caller uses SQL, Tx.Delete, or Collection.Delete directly.
	if c != nil && c.db != nil {
		return c.db.WithTx(ctx, func(tx Tx) error {
			return tx.Delete(ctx, c.name, id)
		})
	}

	var oldMetadata map[string]interface{}
	var cascadeDeletes []cascadeOp
	defer func() {
		if err == nil {
			c.removeFromMetadataIndex(id, oldMetadata)
			c.markMetadataIndexDirty()
			// Execute cascading deletes after the parent is removed.
			for _, op := range cascadeDeletes {
				c.executeCascadeMutation(ctx, op)
			}
		}
	}()

	// Preflight: check FK references (RESTRICT rejection + CASCADE collection).
	cascades, err := c.checkDeleteFKReferences(ctx, id)
	if err != nil {
		return err
	}
	cascadeDeletes = cascades

	release, err := c.acquireWrite(ctx)
	if err != nil {
		return err
	}
	defer release()

	unlockAsync, err := c.lockAsyncMutation(ctx)
	if err != nil {
		return fmt.Errorf("failed to flush asynchronous index before delete: %w", err)
	}
	defer unlockAsync()

	c.mu.RLock()
	if c.closed {
		c.mu.RUnlock()
		return ErrCollectionClosed
	}

	// Validate input
	if id == "" {
		c.mu.RUnlock()
		return fmt.Errorf("vector ID cannot be empty")
	}

	// Non-sharded path
	if c.shards == nil {
		// Keep the collection alive through mutation-slot release.
		defer c.mu.RUnlock()
		mutation := c.lockMutationID(id)
		defer mutation.unlock()

		entry, err := c.storage.Get(ctx, id)
		var hasEntry bool
		if err == nil {
			hasEntry = true
			oldMetadata = entry.Metadata
		} else if !isNotFoundError(err) {
			return fmt.Errorf("failed to get vector for deletion: %w", err)
		}

		var indexErr error
		if hasEntry {
			type transactionStarter interface {
				BeginTxn() *graph.Txn
			}

			if len(c.deleteHooks) > 0 {
				var txn *graph.Txn
				if c.graph != nil {
					if starter, ok := c.graph.(transactionStarter); ok {
						txn = starter.BeginTxn()
					}
				}
				if txn == nil {
					txn = &graph.Txn{}
				}

				for _, hook := range c.deleteHooks {
					if err := hook(txn, uint64(entry.Ordinal)); err != nil {
						return fmt.Errorf("delete hook failed: %w", err)
					}
				}
				if c.graph != nil && txn.ID != 0 {
					if err := txn.Commit(ctx); err != nil {
						return fmt.Errorf("failed to commit graph transaction: %w", err)
					}
				}
			}

			if err := c.storage.Delete(ctx, id); err != nil {
				if !isNotFoundError(err) {
					return fmt.Errorf("failed to write deletion to storage: %w", err)
				}
			}
			indexErr = deleteIndexEntry(ctx, c.index, id, entry.Ordinal)
		} else {
			indexErr = c.index.Delete(ctx, id)
		}

		if indexErr != nil {
			if !isNotFoundError(indexErr) {
				if rebuildErr := c.rebuildIndex(ctx); rebuildErr != nil {
					return fmt.Errorf("failed to delete vector from index: %w; rebuild after delete also failed: %v", indexErr, rebuildErr)
				}
			}
		}

		// Update metrics
		if c.metrics != nil {
			c.metrics.VectorDeletes.Inc()
		}
		return nil
	}

	// Sharded path: keep the collection read lock until the mutation guard is
	// released. The guard owns an off-heap mutation table that Close destroys.
	defer c.mu.RUnlock()
	mutation := c.lockMutationID(id)
	defer mutation.unlock()

	shardIdx := shardForID(id)
	shard := &c.shards[shardIdx]
	shard.mu.Lock()
	defer shard.mu.Unlock()

	entry, err := shard.storage.Get(ctx, id)
	var hasEntry bool
	if err == nil {
		hasEntry = true
	} else if !isNotFoundError(err) {
		return fmt.Errorf("failed to get vector for deletion from shard %d: %w", shardIdx, err)
	}

	var indexErr error
	if hasEntry {
		if len(c.deleteHooks) > 0 {
			type transactionStarter interface {
				BeginTxn() *graph.Txn
			}
			var txn *graph.Txn
			if c.graph != nil {
				if starter, ok := c.graph.(transactionStarter); ok {
					txn = starter.BeginTxn()
				}
			}
			if txn == nil {
				txn = &graph.Txn{}
			}

			for _, hook := range c.deleteHooks {
				if err := hook(txn, uint64(entry.Ordinal)); err != nil {
					return fmt.Errorf("delete hook failed: %w", err)
				}
			}
			if c.graph != nil && txn.ID != 0 {
				if err := txn.Commit(ctx); err != nil {
					return fmt.Errorf("failed to commit graph transaction: %w", err)
				}
			}
		}

		if err := shard.storage.Delete(ctx, id); err != nil {
			if !isNotFoundError(err) {
				return fmt.Errorf("failed to write deletion to storage shard %d: %w", shardIdx, err)
			}
		}
		indexErr = deleteIndexEntry(ctx, shard.index, id, entry.Ordinal)
	} else {
		indexErr = shard.index.Delete(ctx, id)
	}
	if indexErr != nil {
		if !isNotFoundError(indexErr) {
			if rebuildErr := c.rebuildShardIndex(ctx, shardIdx); rebuildErr != nil {
				return fmt.Errorf("failed to delete vector from shard index: %w; rebuild after delete also failed: %v", indexErr, rebuildErr)
			}
		}
	}

	// Update metrics
	if c.metrics != nil {
		c.metrics.VectorDeletes.Inc()
	}

	return nil
}

func deleteIndexEntry(ctx context.Context, idx index.Index, id string, ordinal uint32) error {
	if deleter, ok := idx.(interface {
		DeleteByOrdinal(context.Context, uint32) error
	}); ok {
		if err := deleter.DeleteByOrdinal(ctx, ordinal); err == nil || !isNotFoundError(err) {
			return err
		}
	}
	return idx.Delete(ctx, id)
}

func isNotFoundError(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, util.ErrNotFound) || errors.Is(err, util.ErrEmptyIndex) {
		return true
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "not found") ||
		strings.Contains(msg, "does not exist") ||
		strings.Contains(msg, "index is empty") ||
		strings.Contains(msg, "cannot delete from empty index")
}

// InsertBatch inserts multiple vectors using the public collection API.
func (c *Collection) InsertBatch(ctx context.Context, entries []VectorEntry) error {
	// Preflight: apply DEFAULTs, then validate NOT NULL, CHECK, FK, and UNIQUE.
	for i := range entries {
		entries[i].Metadata = c.metadataWithDefaults(entries[i].Metadata)
		if err := c.validateNotNullConstraints(entries[i].Metadata); err != nil {
			return err
		}
		if err := c.validateCheckConstraints(entries[i].Metadata); err != nil {
			return err
		}
		if err := c.validateForeignKeys(ctx, entries[i].ID, entries[i].Metadata); err != nil {
			return err
		}
		if err := c.validateUniqueConstraints(ctx, entries[i].ID, entries[i].Metadata); err != nil {
			return err
		}
	}
	indexEntries := make([]*index.VectorEntry, 0, len(entries))
	for _, entry := range entries {
		indexEntries = append(indexEntries, &index.VectorEntry{
			ID:       entry.ID,
			Vector:   entry.Vector,
			Metadata: entry.Metadata,
		})
	}
	return c.insertBatch(ctx, indexEntries)
}

// DeleteBatch deletes multiple vectors by ID.
func (c *Collection) DeleteBatch(ctx context.Context, ids []string) error {
	if c != nil && c.db != nil {
		return c.db.WithTx(ctx, func(tx Tx) error {
			return tx.DeleteBatch(ctx, c.name, ids)
		})
	}
	for _, id := range ids {
		if err := c.Delete(ctx, id); err != nil {
			return err
		}
	}
	return nil
}

// Get returns a persisted record by ID.
func (c *Collection) Get(ctx context.Context, id string) (Record, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return Record{}, ErrCollectionClosed
	}

	// Route to the correct shard for this ID
	if c.shards != nil {
		shard := c.getShard(id)
		entry, err := shard.storage.Get(ctx, id)
		if err != nil {
			return Record{}, fmt.Errorf("%w: %s", ErrRecordNotFound, id)
		}
		return recordFromIndexEntry(entry), nil
	}

	entry, err := c.storage.Get(ctx, id)
	if err != nil {
		return Record{}, fmt.Errorf("%w: %s", ErrRecordNotFound, id)
	}
	return recordFromIndexEntry(entry), nil
}

// UpdateIfVersion updates a record only if its current committed version matches expectedVersion.
func (c *Collection) UpdateIfVersion(ctx context.Context, id string, vector []float32, metadata map[string]interface{}, expectedVersion uint64) error {
	return c.withCAS(ctx, func(tx Tx) error {
		return tx.UpdateIfVersion(ctx, c.name, id, vector, metadata, expectedVersion)
	})
}

// DeleteIfVersion deletes a record only if its current committed version matches expectedVersion.
func (c *Collection) DeleteIfVersion(ctx context.Context, id string, expectedVersion uint64) error {
	return c.withCAS(ctx, func(tx Tx) error {
		return tx.DeleteIfVersion(ctx, c.name, id, expectedVersion)
	})
}

func (c *Collection) withCAS(ctx context.Context, fn func(tx Tx) error) error {
	if c == nil {
		return ErrCollectionClosed
	}
	if c.db == nil {
		return ErrTxEngineUnsupported
	}
	return c.db.WithTx(ctx, fn)
}

// Iterate walks all persisted records in the collection.
func (c *Collection) Iterate(ctx context.Context, fn func(Record) error) error {
	if fn == nil {
		return fmt.Errorf("iterate callback cannot be nil")
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return ErrCollectionClosed
	}

	// Sharded path: iterate over all shards
	if c.shards != nil {
		for i := range c.shards {
			err := c.shards[i].storage.Iterate(ctx, func(entry *index.VectorEntry) error {
				return fn(recordFromIndexEntry(entry))
			})
			if err != nil {
				return err
			}
		}
		return nil
	}

	return c.storage.Iterate(ctx, func(entry *index.VectorEntry) error {
		return fn(recordFromIndexEntry(entry))
	})
}

// ListAll returns all persisted records in the collection.
func (c *Collection) ListAll(ctx context.Context) ([]Record, error) {
	records := make([]Record, 0)
	if err := c.Iterate(ctx, func(record Record) error {
		records = append(records, record)
		return nil
	}); err != nil {
		return nil, err
	}
	return records, nil
}

// validateUniqueConstraints checks that values for columns with the UNIQUE
// flag don't already exist in other records. Uses unsafe.Pointer for direct
// catalog column access, matching the codebase convention for zero-copy reads.
// applyDefaults fills in missing metadata keys from the column DEFAULT
// declarations in the collection config. The metadata map is mutated in place.
func (c *Collection) applyDefaults(metadata map[string]interface{}) {
	if c.config == nil || metadata == nil {
		return
	}
	// Collect column name → default value pairs. Prefer CollectionConfig
	// (populated during CREATE), fall back to catalog (authoritative after reopen).
	defaults := make(map[string]string)
	for colName, val := range c.config.ColumnDefaults {
		defaults[colName] = val
	}
	if len(defaults) == 0 && c.db != nil && c.db.catalog != nil {
		tableHash := catalog.HashIdentifier(c.name)
		catDefaults := c.db.catalog.DefaultValuesForTable(tableHash)
		if len(catDefaults) > 0 {
			tbl, err := c.db.catalog.GetTable(tableHash)
			if err == nil {
				colSize := uint32(unsafe.Sizeof(catalog.ColumnDef{}))
				for i := uint32(0); i < tbl.ColumnsCount; i++ {
					col := (*catalog.ColumnDef)(unsafe.Pointer(&c.db.catalog.Data()[tbl.ColumnsOffset+i*colSize]))
					if val, ok := catDefaults[col.NameHash]; ok {
						name := c.db.catalog.ColumnName(tbl, col)
						if name != "" {
							defaults[name] = val
						}
					}
				}
			}
		}
	}
	for colName, defaultVal := range defaults {
		if _, exists := metadata[colName]; exists {
			continue
		}
		metadata[colName] = parseDefaultLiteral(defaultVal)
	}
}

// metadataWithDefaults returns metadata prepared for a write. It preserves
// the historical in-place mutation behavior for non-nil maps, while ensuring
// a nil map can receive DEFAULT values. This is used by every direct
// collection write path before NOT NULL/CHECK validation.
func (c *Collection) metadataWithDefaults(metadata map[string]interface{}) map[string]interface{} {
	// Writes always take ownership of a private, recursively cloned image.
	// This is especially important for JSON/JSONB maps and arrays, which are
	// otherwise mutable through the caller after an epoch or WAL append.
	metadata = cloneMetadata(metadata)
	if metadata == nil {
		metadata = make(map[string]interface{})
	}
	c.applyDefaults(metadata)
	return metadata
}

// validateNotNullConstraints enforces declared NOT NULL columns for the
// effective metadata image. The collection config is authoritative for a
// newly-created collection; the catalog fallback makes the same enforcement
// survive close/reopen, where physical storage config intentionally omits SQL
// schema metadata. A present empty string is a value and is therefore valid;
// only a missing key or a Go nil value represents SQL NULL here.
func (c *Collection) validateNotNullConstraints(metadata map[string]interface{}) error {
	if c == nil {
		return nil
	}
	required := make(map[uint64]string)
	if c.config != nil {
		for name, flags := range c.config.ColumnConstraints {
			if flags&catalog.ColFlagNotNull != 0 {
				required[catalog.HashIdentifier(name)] = name
			}
		}
	}

	// Catalog flags are needed after reopen and also cover schemas created by
	// older API paths that did not populate CollectionConfig constraints.
	if c.db != nil && c.db.catalog != nil {
		tableHash := catalog.HashIdentifier(c.name)
		if table, err := c.db.catalog.GetTable(tableHash); err == nil {
			data := c.db.catalog.Data()
			colSize := uint32(unsafe.Sizeof(catalog.ColumnDef{}))
			for i := uint32(0); i < table.ColumnsCount; i++ {
				offset := table.ColumnsOffset + i*colSize
				if int(offset+colSize) > len(data) {
					break
				}
				col := (*catalog.ColumnDef)(unsafe.Pointer(&data[offset]))
				if col.Flags&catalog.ColFlagNotNull == 0 {
					continue
				}
				name := c.db.catalog.ColumnName(table, col)
				if name != "" {
					required[col.NameHash] = name
				}
			}
		}
	}

	// The physical record key is supplied separately from metadata. Its
	// non-null/empty validation remains the existing ID validation contract.
	idHash := catalog.HashIdentifier("id")
	names := make([]string, 0, len(required))
	for hash, name := range required {
		if hash != idHash {
			names = append(names, name)
		}
	}
	sort.Slice(names, func(i, j int) bool {
		return strings.ToLower(names[i]) < strings.ToLower(names[j])
	})
	for _, name := range names {
		hash := catalog.HashIdentifier(name)
		found := false
		var value interface{}
		for key, candidate := range metadata {
			if catalog.HashIdentifier(key) == hash {
				found = true
				value = candidate
				break
			}
		}
		if !found || value == nil {
			return fmt.Errorf("NOT NULL constraint violation: column %q cannot be null", name)
		}
	}
	if err := c.validateJSONFields(metadata); err != nil {
		return err
	}
	return nil
}

// validateJSONFields enforces that values written to JSON/JSONB columns are
// valid JSON documents. SQL text literals are accepted when they contain a
// complete JSON document; direct Go callers may also provide decoded maps,
// slices, and scalar values. Missing values are left to the NOT NULL/default
// validation path and remain valid for nullable columns.
func (c *Collection) validateJSONFields(metadata map[string]interface{}) error {
	if c == nil || c.config == nil || len(c.config.MetadataSchema) == 0 {
		return nil
	}
	for name, fieldType := range c.config.MetadataSchema {
		if fieldType != JSONField && fieldType != JSONBField {
			continue
		}
		var value interface{}
		found := false
		for key, candidate := range metadata {
			if strings.EqualFold(key, name) {
				value = candidate
				found = true
				break
			}
		}
		if !found || value == nil {
			continue
		}
		canonical, ok := decodeJSONValue(value)
		if !ok {
			return fmt.Errorf("invalid JSON value for column %q", name)
		}
		// Replace the supplied value with the canonical, recursively owned
		// tree. JSON and JSONB therefore follow the same safe storage path;
		// JSONB comparisons additionally benefit from normalized numbers.
		for key := range metadata {
			if strings.EqualFold(key, name) {
				metadata[key] = canonical
				break
			}
		}
	}
	return nil
}

// parseDefaultLiteral converts a DEFAULT literal string to its Go value.
// Handles: NULL, TRUE, FALSE, quoted strings, and numbers.
func parseDefaultLiteral(s string) interface{} {
	switch s {
	case "NULL", "null":
		return nil
	case "TRUE", "true":
		return true
	case "FALSE", "false":
		return false
	}
	// Quoted string literal — strip surrounding quotes.
	if len(s) >= 2 && s[0] == '\'' && s[len(s)-1] == '\'' {
		return s[1 : len(s)-1]
	}
	// Number literal — try integer then float.
	if i, err := strconv.ParseInt(s, 10, 64); err == nil {
		return i
	}
	if f, err := strconv.ParseFloat(s, 64); err == nil {
		return f
	}
	// Fallback: return as string.
	return s
}

// validateCheckConstraints evaluates all CHECK constraints for the row.
// Returns an error naming the constraint that failed, or nil if all pass.
func (c *Collection) validateCheckConstraints(metadata map[string]interface{}) error {
	if c.config == nil {
		return nil
	}
	// Collect CHECK expressions: prefer CollectionConfig (populated during
	// CREATE), fall back to catalog (authoritative after reopen).
	var exprs []string
	for _, chk := range c.config.CheckConstraints {
		exprs = append(exprs, chk.Expression)
	}
	if len(exprs) == 0 && c.db != nil && c.db.catalog != nil {
		tableHash := catalog.HashIdentifier(c.name)
		for _, chk := range c.db.catalog.CheckConstraintsForTable(tableHash) {
			expr := c.db.catalog.CheckExpr(chk)
			if expr != "" {
				exprs = append(exprs, expr)
			}
		}
	}
	for _, expr := range exprs {
		result, err := evaluateCheckExpr(expr, metadata)
		if err != nil {
			return fmt.Errorf("CHECK constraint evaluation error: %w", err)
		}
		if !result {
			return fmt.Errorf("CHECK constraint %q failed", expr)
		}
	}
	return nil
}

// evaluateCheckExpr evaluates CHECK expressions with SQL three-valued logic.
// The evaluator supports comparisons, IS [NOT] NULL, BETWEEN, boolean
// AND/OR/NOT, and nested parentheses. Unsupported SQL functions/operators
// return an explicit error rather than being silently accepted.
func evaluateCheckExpr(expr string, metadata map[string]interface{}) (bool, error) {
	return evaluateCheckBooleanExpr(expr, metadata)
}

// evaluateCheckExprLegacy is retained as a reference implementation for
// debugging older persisted schemas; new writes use evaluateCheckBooleanExpr.
func evaluateCheckExprLegacy(expr string, metadata map[string]interface{}) (bool, error) {
	expr = strings.TrimSpace(expr)
	if expr == "" {
		return true, nil
	}

	// col IS NOT NULL
	if strings.HasSuffix(expr, " IS NOT NULL") {
		col := strings.TrimSpace(expr[:len(expr)-len(" IS NOT NULL")])
		val, ok := metadata[col]
		return ok && val != nil, nil
	}
	// col IS NULL
	if strings.HasSuffix(expr, " IS NULL") {
		col := strings.TrimSpace(expr[:len(expr)-len(" IS NULL")])
		val, ok := metadata[col]
		return !ok || val == nil, nil
	}

	// col BETWEEN low AND high
	if idx := strings.Index(strings.ToUpper(expr), " BETWEEN "); idx >= 0 {
		col := strings.TrimSpace(expr[:idx])
		rest := strings.TrimSpace(expr[idx+len(" BETWEEN "):])
		andIdx := strings.Index(strings.ToUpper(rest), " AND ")
		if andIdx < 0 {
			return false, fmt.Errorf("malformed BETWEEN: missing AND")
		}
		lowStr := strings.TrimSpace(rest[:andIdx])
		highStr := strings.TrimSpace(rest[andIdx+len(" AND "):])
		val, ok := metadata[col]
		if !ok || val == nil {
			// SQL CHECK constraints reject only FALSE. A NULL operand makes
			// the predicate UNKNOWN, which satisfies the constraint unless a
			// separate NOT NULL constraint rejects the row.
			return true, nil
		}
		low := parseDefaultLiteral(stripQuotes(lowStr))
		high := parseDefaultLiteral(stripQuotes(highStr))
		return compareVals(val, low) >= 0 && compareVals(val, high) <= 0, nil
	}

	// col OP literal or col1 OP col2 — find the operator
	ops := []string{"!=", "<=", ">=", "<>", "=", "<", ">"}
	for _, op := range ops {
		idx := strings.Index(expr, op)
		if idx < 0 {
			continue
		}
		col := strings.TrimSpace(expr[:idx])
		right := strings.TrimSpace(expr[idx+len(op):])
		val, ok := metadata[col]
		if !ok || val == nil {
			return true, nil // NULL operand → UNKNOWN; CHECK accepts UNKNOWN
		}
		// Try right side as a column reference first, then as a literal.
		var rightVal interface{}
		if rv, rok := metadata[right]; rok && rv != nil {
			rightVal = rv
		} else {
			rightVal = parseDefaultLiteral(stripQuotes(right))
		}
		if rightVal == nil {
			return true, nil // comparison with NULL is UNKNOWN
		}
		cmp := compareVals(val, rightVal)
		switch op {
		case "=":
			return cmp == 0, nil
		case "!=", "<>":
			return cmp != 0, nil
		case "<":
			return cmp < 0, nil
		case ">":
			return cmp > 0, nil
		case "<=":
			return cmp <= 0, nil
		case ">=":
			return cmp >= 0, nil
		}
	}
	return false, fmt.Errorf("unsupported CHECK expression: %q", expr)
}

// stripQuotes removes surrounding single quotes from a string literal.
func stripQuotes(s string) string {
	if len(s) >= 2 && s[0] == '\'' && s[len(s)-1] == '\'' {
		return s[1 : len(s)-1]
	}
	return s
}

// compareVals compares two metadata values for CHECK constraint evaluation.
// Returns -1, 0, 1 for less, equal, greater. Only numeric and string types are
// comparable; mismatched types are not equal.
func compareVals(a, b interface{}) int {
	if a == nil || b == nil {
		if a == b {
			return 0
		}
		return -1
	}
	// Try numeric comparison first.
	aF, aOk := toFloat(a)
	bF, bOk := toFloat(b)
	if aOk && bOk {
		switch {
		case aF < bF:
			return -1
		case aF > bF:
			return 1
		default:
			return 0
		}
	}
	// Fall back to string comparison.
	aStr := recordMetaToString(a)
	bStr := recordMetaToString(b)
	return strings.Compare(aStr, bStr)
}

func (c *Collection) validateUniqueConstraints(ctx context.Context, id string, metadata map[string]interface{}) error {
	if c == nil || metadata == nil {
		return nil
	}
	// Named UNIQUE indexes/constraints are maintained on the collection
	// configuration so ON CONFLICT ON CONSTRAINT and ordinary INSERT/UPDATE
	// paths share the same enforcement.
	c.mu.RLock()
	named := make(map[string][]string, len(c.config.NamedUniqueConstraints))
	for name, columns := range c.config.NamedUniqueConstraints {
		named[name] = append([]string(nil), columns...)
	}
	c.mu.RUnlock()
	if len(named) > 0 {
		records, err := recordsVisibleInContext(ctx, c)
		if err == nil {
			for name, columns := range named {
				key, ok := namedUniqueKey(id, metadata, columns)
				if !ok {
					continue // SQL NULL does not participate in UNIQUE conflicts.
				}
				for _, record := range records {
					if record.ID == id {
						continue
					}
					other, otherOK := namedUniqueKey(record.ID, record.Metadata, columns)
					if otherOK && other == key {
						return fmt.Errorf("UNIQUE constraint %q violation", name)
					}
				}
			}
		}
	}
	if c.db == nil || c.db.catalog == nil {
		return nil
	}
	tableHash := catalog.HashIdentifier(c.name)
	tbl, err := c.db.catalog.GetTable(tableHash)
	if err != nil {
		return nil
	}
	data := c.db.catalog.Data()
	colSize := uint32(unsafe.Sizeof(catalog.ColumnDef{}))
	for i := uint32(0); i < tbl.ColumnsCount; i++ {
		col := (*catalog.ColumnDef)(unsafe.Pointer(&data[tbl.ColumnsOffset+i*colSize]))
		if col.Flags&catalog.ColFlagUnique == 0 {
			continue
		}
		for key, val := range metadata {
			if catalog.HashIdentifier(key) != col.NameHash {
				continue
			}
			valStr := recordMetaToString(val)
			existing, err := c.ListByMetadata(ctx, key, val)
			if err != nil {
				continue
			}
			for _, rec := range existing {
				if rec.ID != id {
					return fmt.Errorf(
						"UNIQUE constraint violation: value %q already exists in column %q",
						valStr, key)
				}
			}
		}
	}
	return nil
}

func (c *Collection) markMetadataIndexDirty() {
	if c != nil && c.config != nil && (len(c.config.IndexedFields) > 0 || len(c.config.JSONIndexes) > 0 || len(c.config.MetadataSchema) > 0) {
		c.metadataMutationEpoch.Add(1)
	}
	if c != nil && c.costModel != nil {
		c.costModel.markDirty()
	}
}

func jsonContainmentToken(prefix, value string) string { return prefix + value }

func appendJSONContainmentTokens(postings map[string][]string, node interface{}, id string) {
	seen := make(map[string]struct{})
	appendJSONContainmentTokensSeen(postings, node, id, seen)
}

func appendJSONContainmentTokensSeen(postings map[string][]string, node interface{}, id string, seen map[string]struct{}) {
	add := func(token string) {
		if _, exists := seen[token]; exists {
			return
		}
		seen[token] = struct{}{}
		postings[token] = append(postings[token], id)
	}
	switch value := node.(type) {
	case map[string]interface{}:
		for key, child := range value {
			add(jsonContainmentToken("k:", key))
			appendJSONContainmentTokensSeen(postings, child, id, seen)
		}
	case []interface{}:
		for _, child := range value {
			appendJSONContainmentTokensSeen(postings, child, id, seen)
		}
	default:
		if encoded, err := encodeJSONValue(node); err == nil {
			add(jsonContainmentToken("v:", encoded))
		}
	}
}

func (c *Collection) lookupJSONContainment(ctx context.Context, column string, operator lexer.Kind, value interface{}) ([]Record, bool, error) {
	if c == nil || c.config == nil || c.db == nil {
		return nil, false, nil
	}
	if epochFromContext(ctx) != nil || transactionFromContext(ctx) != nil {
		return nil, false, nil
	}
	_, found := c.config.MetadataSchema[column]
	if !found {
		for name, field := range c.config.MetadataSchema {
			if strings.EqualFold(name, column) && (field == JSONField || field == JSONBField) {
				column = name
				found = true
				break
			}
		}
	}
	if !found || (c.config.MetadataSchema[column] != JSONField && c.config.MetadataSchema[column] != JSONBField) {
		return nil, false, nil
	}
	node := value
	if operator != lexer.KindJSONExists {
		var valid bool
		node, valid = decodeJSONReadValue(value)
		if !valid {
			return nil, true, nil
		}
	}
	tokens := make(map[string]struct{})
	if operator == lexer.KindJSONExists {
		if key, ok := jsonKeyValue(node); ok {
			tokens[jsonContainmentToken("k:", key)] = struct{}{}
		}
	} else {
		var scratch = make(map[string][]string)
		appendJSONContainmentTokens(scratch, node, "")
		for token := range scratch {
			tokens[token] = struct{}{}
		}
	}
	if len(tokens) == 0 {
		return nil, true, nil
	}
	c.metadataIndexMu.Lock()
	epoch := c.metadataMutationEpoch.Load()
	if c.jsonContainmentIndex == nil || c.jsonContainmentBuiltAt != epoch {
		if err := c.rebuildJSONIndexLocked(ctx, epoch); err != nil {
			c.metadataIndexMu.Unlock()
			return nil, true, err
		}
	}
	var ids []string
	first := true
	for token := range tokens {
		posting := c.jsonContainmentIndex[strings.ToLower(column)][token]
		if first {
			ids = append([]string(nil), posting...)
			first = false
			continue
		}
		ids = intersectStringIDs(ids, posting)
	}
	c.metadataIndexMu.Unlock()
	// ?| is an OR of key postings; the generic intersection above applies to
	// containment and ?&. Callers use this method only for those two forms.
	records := make([]Record, 0, len(ids))
	for _, id := range ids {
		record, err := c.Get(ctx, id)
		if err != nil {
			if isNotFoundError(err) || errors.Is(err, ErrRecordNotFound) {
				continue
			}
			return nil, true, err
		}
		records = append(records, record)
	}
	return records, true, nil
}

func intersectStringIDs(left, right []string) []string {
	set := make(map[string]struct{}, len(right))
	for _, id := range right {
		set[id] = struct{}{}
	}
	seen := make(map[string]struct{}, len(left))
	out := left[:0]
	for _, id := range left {
		if _, ok := set[id]; ok {
			if _, duplicate := seen[id]; duplicate {
				continue
			}
			seen[id] = struct{}{}
			out = append(out, id)
		}
	}
	return out
}

func jsonIndexIdentity(column, path string, textResult bool) string {
	return strings.ToLower(strings.TrimSpace(column)) + "\x00" + strings.TrimSpace(path) + "\x00" + strconv.FormatBool(textResult)
}

func jsonIndexPostingKey(value interface{}, textResult bool) (string, bool) {
	if value == nil {
		return "", false
	}
	if textResult {
		if text, ok := value.(string); ok {
			return "text:" + text, true
		}
		encoded, err := encodeJSONValue(value)
		if err != nil {
			return "", false
		}
		return "text:" + encoded, true
	}
	encoded, err := encodeJSONValue(value)
	if err != nil {
		return "", false
	}
	return "json:" + encoded, true
}

// lookupIndexedJSON returns candidates for an equality predicate over a
// configured JSON path. The final JSON expression evaluator remains
// authoritative; postings only narrow the candidate set.
func (c *Collection) lookupIndexedJSON(ctx context.Context, column, path string, textResult bool, value interface{}) ([]Record, bool, error) {
	if c == nil || c.config == nil {
		return nil, false, nil
	}
	identity := jsonIndexIdentity(column, path, textResult)
	configured := false
	for _, index := range c.config.JSONIndexes {
		if jsonIndexIdentity(index.Column, index.Path, index.TextResult) == identity {
			configured = true
			break
		}
	}
	if !configured {
		return nil, false, nil
	}
	posting, ok := jsonIndexPostingKey(value, textResult)
	if !ok {
		return nil, true, nil
	}
	c.metadataIndexMu.Lock()
	epoch := c.metadataMutationEpoch.Load()
	if c.jsonIndex == nil || c.jsonIndexBuiltAt != epoch {
		if err := c.rebuildJSONIndexLocked(ctx, epoch); err != nil {
			c.metadataIndexMu.Unlock()
			return nil, true, err
		}
	}
	ids := c.jsonIndex[identity][posting]
	copyIDs := append([]string(nil), ids...)
	c.metadataIndexMu.Unlock()
	records := make([]Record, 0, len(copyIDs))
	for _, id := range copyIDs {
		record, err := c.Get(ctx, id)
		if err != nil {
			if isNotFoundError(err) || errors.Is(err, ErrRecordNotFound) {
				continue
			}
			return nil, true, err
		}
		records = append(records, record)
	}
	return records, true, nil
}

// lookupVisibleJSONOverlay evaluates an indexed JSON path against the
// transaction-local relation image. The committed posting map cannot be used
// here: epoch inserts/updates/deletes and savepoint rollback are not present in
// that map. This overlay deliberately favors correctness; a future physical
// overlay can replace the scan without changing the SQL contract.
func (c *Collection) lookupVisibleJSONOverlay(ctx context.Context, column, path string, textResult bool, value interface{}) ([]Record, bool, error) {
	if c == nil || c.config == nil {
		return nil, false, nil
	}
	identity := jsonIndexIdentity(column, path, textResult)
	configured := false
	for _, index := range c.config.JSONIndexes {
		if jsonIndexIdentity(index.Column, index.Path, index.TextResult) == identity {
			configured = true
			break
		}
	}
	if !configured {
		return nil, false, nil
	}
	wanted, ok := jsonIndexPostingKey(value, textResult)
	if !ok {
		return nil, true, nil
	}
	records, err := recordsVisibleInContext(ctx, c)
	if err != nil {
		return nil, true, err
	}
	matched := make([]Record, 0, len(records))
	for _, record := range records {
		extracted, exists, extractErr := jsonPath(record.Metadata[column], path, textResult)
		if extractErr != nil {
			return nil, true, extractErr
		}
		if !exists {
			continue
		}
		posting, postingOK := jsonIndexPostingKey(extracted, textResult)
		if postingOK && posting == wanted {
			matched = append(matched, record)
		}
	}
	return matched, true, nil
}

func (c *Collection) rebuildJSONIndexLocked(ctx context.Context, epoch uint64) error {
	postings := make(map[string]map[string][]string, len(c.config.JSONIndexes))
	containment := make(map[string]map[string][]string)
	for column, fieldType := range c.config.MetadataSchema {
		if fieldType == JSONField || fieldType == JSONBField {
			containment[strings.ToLower(column)] = make(map[string][]string)
		}
	}
	for _, index := range c.config.JSONIndexes {
		postings[jsonIndexIdentity(index.Column, index.Path, index.TextResult)] = make(map[string][]string)
	}
	add := func(entry *index.VectorEntry) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		for _, definition := range c.config.JSONIndexes {
			identity := jsonIndexIdentity(definition.Column, definition.Path, definition.TextResult)
			value, ok, err := jsonPath(entry.Metadata[definition.Column], definition.Path, definition.TextResult)
			if err != nil {
				return fmt.Errorf("build JSON index %q: %w", definition.Name, err)
			}
			if !ok {
				continue
			}
			key, ok := jsonIndexPostingKey(value, definition.TextResult)
			if ok {
				postings[identity][key] = append(postings[identity][key], entry.ID)
			}
		}
		for column := range containment {
			// Index construction only reads the JSON tree. Avoid cloning stored
			// metadata here; mutation paths retain decodeJSONValue's ownership
			// boundary.
			node, ok := decodeJSONReadValue(entry.Metadata[column])
			if !ok {
				// Metadata keys preserve user casing; locate the field once if
				// the schema spelling differs from the canonical index key.
				for name, candidate := range entry.Metadata {
					if strings.EqualFold(name, column) {
						node, ok = decodeJSONReadValue(candidate)
						break
					}
				}
			}
			if ok {
				appendJSONContainmentTokens(containment[column], node, entry.ID)
			}
		}
		return nil
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.closed {
		return ErrCollectionClosed
	}
	if c.shards != nil {
		for i := range c.shards {
			if err := c.shards[i].storage.Iterate(ctx, add); err != nil {
				return fmt.Errorf("build JSON index for shard %d: %w", i, err)
			}
		}
	} else if err := c.storage.Iterate(ctx, add); err != nil {
		return fmt.Errorf("build JSON index: %w", err)
	}
	c.jsonIndex = postings
	c.jsonIndexBuiltAt = epoch
	c.jsonContainmentIndex = containment
	c.jsonContainmentBuiltAt = epoch
	return nil
}

// validateForeignKeys checks FK column values against parent tables.
func (c *Collection) validateForeignKeys(ctx context.Context, id string, metadata map[string]interface{}) error {
	// Skip FK validation during cascade SET NULL / SET DEFAULT — the parent
	// row is being deleted/updated in the same transaction.
	if ctx.Value(ctxKeySkipFKValidation{}) != nil {
		return nil
	}
	if c.db == nil || c.db.catalog == nil {
		return nil
	}
	tableHash := catalog.HashIdentifier(c.name)
	groups := c.db.catalog.ForeignKeyGroupsForTable(tableHash)
	if len(groups) == 0 {
		return nil
	}

	c.db.mu.RLock()
	nameByHash := make(map[uint64]string, len(c.db.collections))
	for name := range c.db.collections {
		nameByHash[catalog.HashIdentifier(name)] = name
	}
	c.db.mu.RUnlock()
	for _, group := range groups {
		// A composite FK is nullable as a unit: if any component is NULL,
		// the constraint is not checked unless that component is NOT NULL.
		sourceValues := make([]fkValue, len(group.Pairs))
		null := false
		for i, fk := range group.Pairs {
			sourceValues[i] = fkValueFromRecord(id, metadata, fk.SourceColHash)
			if sourceValues[i].Null {
				null = true
				if c.isFKColumnNotNull(fk) {
					return fmt.Errorf("NOT NULL constraint violation: column with foreign key must not be null")
				}
			}
		}
		if null {
			continue
		}
		targetName := nameByHash[group.TargetTableHash]
		if targetName == "" {
			// Check system tables (GRAPH_NODES, etc.)
			if len(group.Pairs) != 1 {
				return fmt.Errorf("foreign key violation: system table %q does not support composite references", targetName)
			}
			if err := c.checkFKTargetSystemTable(ctx, group.Pairs[0], sourceValues[0].Value); err != nil {
				return err
			}
			continue
		}

		parent, err := c.db.GetCollection(targetName)
		if err != nil {
			return fmt.Errorf("foreign key: target table %q not found", targetName)
		}

		if !c.parentHasTuple(ctx, parent, group.Pairs, sourceValues) {
			return fmt.Errorf("foreign key violation: referenced value does not exist in %s", targetName)
		}
	}
	return nil
}

// fkValue preserves SQL NULL separately from an ordinary string value. In
// particular, an empty string is a valid non-NULL FK value and must still be
// checked against the referenced table.
type fkValue struct {
	Value string
	Null  bool
}

func fkValueFromRecord(id string, metadata map[string]interface{}, hash uint64) fkValue {
	if hash == catalog.HashIdentifier("id") {
		return fkValue{Value: id}
	}
	return fkValueFromMeta(metadata, hash)
}

func recordValueByHash(r Record, hash uint64) fkValue {
	if hash == catalog.HashIdentifier("id") {
		return fkValue{Value: r.ID}
	}
	return fkValueFromMeta(r.Metadata, hash)
}

func (c *Collection) parentHasTuple(ctx context.Context, parent *Collection, pairs []*catalog.ForeignKeyDef, values []fkValue) bool {
	var (
		records []Record
		err     error
	)
	// Historical epochs take precedence over the live collection. A regular
	// transaction context then overlays its ordered staged mutations. This
	// keeps FK checks consistent with the relation visible to the writer.
	if epoch := epochFromContext(ctx); epoch != nil {
		records, err = epoch.ListRecords(ctx, parent.name)
	} else if tx := transactionFromContext(ctx); tx != nil {
		records, err = tx.visibleRecords(ctx, parent.name)
	} else {
		records, err = parent.ListAll(ctx)
	}
	if err != nil {
		return false
	}
	for _, record := range records {
		match := true
		for i, pair := range pairs {
			parentValue := recordValueByHash(record, pair.TargetColHash)
			if i >= len(values) || parentValue.Null || values[i].Null || parentValue.Value != values[i].Value {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}

// isFKColumnNotNull returns true if the FK's source column has a NOT NULL
// constraint in the catalog.
func (c *Collection) isFKColumnNotNull(fk *catalog.ForeignKeyDef) bool {
	if c.db == nil || c.db.catalog == nil {
		return false
	}
	tgt, err := c.db.catalog.GetTable(fk.SourceTableHash)
	if err != nil {
		return false
	}
	col, err := c.db.catalog.GetColumn(tgt, fk.SourceColHash)
	if err != nil {
		return false
	}
	return col.Flags&catalog.ColFlagNotNull != 0
}

// resolveGraphNodeFKValue accepts either the durable numeric GRAPH_NODES.id or
// a logical graph record ID (TEXT/UUID). Logical IDs are translated through
// the graph collection's existing forward map, including provisional epoch
// nodes; no parallel identity store is introduced for foreign keys.
func (c *Collection) resolveGraphNodeFKValue(ctx context.Context, value string) (uint64, error) {
	if c.db == nil {
		return 0, storage.ErrUnknownGraphNodeID
	}

	// Preserve the established numeric GRAPH_NODES.id contract first. This is
	// also required for epoch-provisional node IDs.
	if nodeID, err := strconv.ParseUint(value, 10, 64); err == nil && nodeID != 0 {
		var resolveErr error
		if epoch := epochFromContext(ctx); epoch != nil {
			_, _, resolveErr = epoch.ResolveNodeID(ctx, nodeID)
		} else {
			_, _, resolveErr = c.db.ResolveNodeID(ctx, nodeID)
		}
		if resolveErr == nil {
			return nodeID, nil
		}
	}

	// A text/UUID FK value names the graph record. Search the graph-backed
	// collections in stable order, preferring the FK-owning collection when it
	// is itself graph-backed.
	for _, collectionName := range c.db.graphCollectionNames(c.name) {
		var (
			nodeID uint64
			err    error
		)
		if epoch := epochFromContext(ctx); epoch != nil {
			nodeID, err = epoch.LookupNodeID(ctx, collectionName, value)
		} else {
			nodeID, err = c.db.GetNodeID(ctx, collectionName, value)
		}
		if err == nil && nodeID != 0 {
			return nodeID, nil
		}
	}
	return 0, storage.ErrUnknownGraphNodeID
}

// checkFKTargetSystemTable validates an FK reference to a system table
// (GRAPH_NODES, etc.) using the appropriate virtual lookup.
func (c *Collection) checkFKTargetSystemTable(ctx context.Context, fk *catalog.ForeignKeyDef, fkValue string) error {
	if fk.TargetTableHash != catalog.HashIdentifier("GRAPH_NODES") {
		return nil // unknown system table — skip (validated at DDL time)
	}
	_, err := c.resolveGraphNodeFKValue(ctx, fkValue)
	if err != nil {
		return fmt.Errorf(
			"foreign key violation: graph node or record %q does not exist", fkValue)
	}
	return nil
}

func (c *Collection) checkFKTargetByID(ctx context.Context, parent *Collection, fkValue, targetName string) error {
	idx := parent.GetIndex()
	if idx == nil {
		return fmt.Errorf("foreign key: parent table %q has no index", targetName)
	}
	if getter, ok := idx.(interface {
		Get(context.Context, string) (uint32, uint32, uint64, error)
	}); ok {
		if _, _, _, err := getter.Get(ctx, fkValue); err != nil {
			return fmt.Errorf(
				"foreign key violation: value %q does not exist in %s(id)",
				fkValue, targetName)
		}
	}
	return nil
}

func (c *Collection) checkFKTargetByMetadata(ctx context.Context, parent *Collection, fk *catalog.ForeignKeyDef, fkValue, targetName string) error {
	targetCol := "?"
	cfg := parent.Config()
	if cfg.MetadataSchema != nil {
		for name := range cfg.MetadataSchema {
			if catalog.HashIdentifier(name) == fk.TargetColHash {
				targetCol = name
				break
			}
		}
	}
	records, err := parent.ListByMetadata(ctx, targetCol, fkValue)
	if err != nil || len(records) == 0 {
		return fmt.Errorf(
			"foreign key violation: value %q does not exist in %s(%s)",
			fkValue, targetName, targetCol)
	}
	return nil
}

// cascadeOp describes a cascading mutation to execute after the parent
// record has been removed or updated.
type cascadeOp struct {
	collectionName string
	recordID       string
	newFKValue     string                        // for ON UPDATE CASCADE: the new FK column value
	sourceCol      string                        // for ON UPDATE CASCADE: the FK column name to update
	action         uint8                         // catalog.OnDelete* constant
	columnValues   map[string]cascadeColumnValue // for SET NULL / SET DEFAULT: col → value
	updateCascade  bool                          // columnValues represents one ON UPDATE tuple
}

// cascadeColumnValue keeps SET NULL distinct from a legitimate empty-string
// DEFAULT or FK value.
type cascadeColumnValue struct {
	Value string
	Null  bool
}

// collectUpdateCascades checks for ON UPDATE RESTRICT violations and
// collects ON UPDATE actions when a parent record's FK-referenced tuple
// changes. Composite constraints are evaluated and applied as one tuple;
// processing each ForeignKeyDef independently would allow a child row to be
// left with a mixed old/new composite key.
func (c *Collection) collectUpdateCascades(ctx context.Context, oldID, newID string, oldMetadata, newMetadata map[string]interface{}) ([]cascadeOp, error) {
	if c.db == nil || c.db.catalog == nil {
		return nil, nil
	}
	tableHash := catalog.HashIdentifier(c.name)
	groups := c.db.catalog.ForeignKeyGroupsToTable(tableHash)
	if len(groups) == 0 {
		return nil, nil
	}

	c.db.mu.RLock()
	nameByHash := make(map[uint64]string, len(c.db.collections))
	for name := range c.db.collections {
		nameByHash[catalog.HashIdentifier(name)] = name
	}
	c.db.mu.RUnlock()

	var cascades []cascadeOp
	for _, group := range groups {
		if group.OnUpdate != catalog.OnDeleteCascade &&
			group.OnUpdate != catalog.OnDeleteRestrict &&
			group.OnUpdate != catalog.OnDeleteSetNull &&
			group.OnUpdate != catalog.OnDeleteSetDefault &&
			group.OnUpdate != catalog.OnDeleteNoAction {
			continue
		}

		oldValues := make([]fkValue, len(group.Pairs))
		newValues := make([]fkValue, len(group.Pairs))
		changed := false
		for i, fk := range group.Pairs {
			oldValues[i] = fkValueFromRecord(oldID, oldMetadata, fk.TargetColHash)
			newValues[i] = fkValueFromRecord(newID, newMetadata, fk.TargetColHash)
			if oldValues[i].Null != newValues[i].Null || oldValues[i].Value != newValues[i].Value {
				changed = true
			}
		}
		if !changed {
			continue
		}

		childName := nameByHash[group.SourceTableHash]
		if childName == "" {
			continue
		}
		child, err := c.db.GetCollection(childName)
		if err != nil {
			continue
		}

		// Find child rows referencing the complete old tuple. Read through the
		// active epoch/transaction overlay so staged child rows participate in
		// the same referential action.
		childRecords, err := recordsVisibleInContext(ctx, child)
		if err != nil {
			return nil, err
		}
		var matchingIDs []string
		for _, record := range childRecords {
			match := true
			for i, fk := range group.Pairs {
				childValue := recordValueByHash(record, fk.SourceColHash)
				if childValue.Null || oldValues[i].Null || childValue.Value != oldValues[i].Value {
					match = false
					break
				}
			}
			if match {
				matchingIDs = append(matchingIDs, record.ID)
			}
		}

		switch group.OnUpdate {
		case catalog.OnDeleteCascade:
			for _, childID := range matchingIDs {
				values := make(map[string]cascadeColumnValue, len(group.Pairs))
				for i, fk := range group.Pairs {
					srcColName := resolveFKSourceCol(child, fk.SourceColHash)
					if srcColName == "" {
						return nil, fmt.Errorf("foreign key violation: ON UPDATE CASCADE cannot update source column hash %d in %s", fk.SourceColHash, childName)
					}
					values[srcColName] = cascadeColumnValue{Value: newValues[i].Value, Null: newValues[i].Null}
				}
				cascades = append(cascades, cascadeOp{
					collectionName: childName,
					recordID:       childID,
					columnValues:   values,
					updateCascade:  true,
					action:         catalog.OnDeleteCascade,
				})
			}
		case catalog.OnDeleteSetNull:
			for _, childID := range matchingIDs {
				colVals := make(map[string]cascadeColumnValue, len(group.Pairs))
				for _, fk := range group.Pairs {
					srcColName := resolveFKSourceCol(child, fk.SourceColHash)
					if srcColName == "" {
						return nil, fmt.Errorf("foreign key violation: ON UPDATE SET NULL cannot update source column hash %d in %s", fk.SourceColHash, childName)
					}
					colVals[srcColName] = cascadeColumnValue{Null: true}
				}
				cascades = append(cascades, cascadeOp{
					collectionName: childName,
					recordID:       childID,
					action:         catalog.OnDeleteSetNull,
					columnValues:   colVals,
				})
			}
		case catalog.OnDeleteSetDefault:
			for _, childID := range matchingIDs {
				colVals := make(map[string]cascadeColumnValue, len(group.Pairs))
				for _, fk := range group.Pairs {
					srcColName := resolveFKSourceCol(child, fk.SourceColHash)
					if srcColName == "" {
						return nil, fmt.Errorf("foreign key violation: ON UPDATE SET DEFAULT cannot update source column hash %d in %s", fk.SourceColHash, childName)
					}
					if childCfg := child.Config(); childCfg.ColumnDefaults != nil {
						if defVal, ok := childCfg.ColumnDefaults[srcColName]; ok {
							colVals[srcColName] = cascadeColumnValue{Value: defVal}
						}
					}
					if _, ok := colVals[srcColName]; !ok {
						return nil, fmt.Errorf(
							"foreign key violation: ON UPDATE SET DEFAULT requires DEFAULT on %s in %s",
							srcColName, childName)
					}
				}
				cascades = append(cascades, cascadeOp{
					collectionName: childName,
					recordID:       childID,
					action:         catalog.OnDeleteSetDefault,
					columnValues:   colVals,
				})
			}
		case catalog.OnDeleteRestrict, catalog.OnDeleteNoAction:
			// NO ACTION is deliberately immediate in this engine. DEFERRABLE
			// constraints and statement-end validation are not part of the
			// catalog contract, so it has the same enforcement point as RESTRICT.
			if len(matchingIDs) > 0 {
				return nil, fmt.Errorf(
					"foreign key violation: cannot update %s in %s because %d row(s) in %s reference it",
					oldID, c.name, len(matchingIDs), childName)
			}
		}
	}
	return cascades, nil
}

// fkValueFromMeta extracts a value from metadata while preserving SQL NULL.
// A missing key and an explicit nil are both NULL; an empty string is not.
func fkValueFromMeta(meta map[string]interface{}, colHash uint64) fkValue {
	if meta == nil {
		return fkValue{Null: true}
	}
	for k, v := range meta {
		if catalog.HashIdentifier(k) == colHash {
			if v == nil {
				return fkValue{Null: true}
			}
			return fkValue{Value: recordMetaToString(v)}
		}
	}
	return fkValue{Null: true}
}

// checkDeleteFKReferences validates that deleting the given record does not
// violate any RESTRICT foreign keys, and collects CASCADE targets.
func (c *Collection) checkDeleteFKReferences(ctx context.Context, id string) ([]cascadeOp, error) {
	if c.db == nil || c.db.catalog == nil {
		return nil, nil
	}
	tableHash := catalog.HashIdentifier(c.name)
	groups := c.db.catalog.ForeignKeyGroupsToTable(tableHash)
	// A record deletion also tombstones its database-scoped GraphNodeID. For
	// graph-enabled collections, include FKs targeting the virtual
	// GRAPH_NODES(id) relation so those children participate in the same
	// referential-action plan.
	if c.GetGraph() != nil && tableHash != catalog.HashIdentifier("GRAPH_NODES") {
		groups = append(groups, c.db.catalog.ForeignKeyGroupsToTable(catalog.HashIdentifier("GRAPH_NODES"))...)
	}
	if len(groups) == 0 {
		return nil, nil
	}
	var parentRecord Record
	var err error
	if epoch := epochFromContext(ctx); epoch != nil {
		records, listErr := epoch.ListRecords(ctx, c.name)
		if listErr != nil {
			return nil, listErr
		}
		found := false
		for _, record := range records {
			if record.ID == id {
				parentRecord = record
				found = true
				break
			}
		}
		if !found {
			return nil, nil
		}
	} else if tx := transactionFromContext(ctx); tx != nil {
		records, listErr := tx.visibleRecords(ctx, c.name)
		if listErr != nil {
			return nil, listErr
		}
		found := false
		for _, record := range records {
			if record.ID == id {
				parentRecord = record
				found = true
				break
			}
		}
		if !found {
			return nil, nil
		}
	} else {
		parentRecord, err = c.Get(ctx, id)
		if err != nil {
			return nil, nil
		}
	}

	c.db.mu.RLock()
	nameByHash := make(map[uint64]string, len(c.db.collections))
	for name := range c.db.collections {
		nameByHash[catalog.HashIdentifier(name)] = name
	}
	c.db.mu.RUnlock()

	var cascades []cascadeOp
	for _, group := range groups {
		childName := nameByHash[group.SourceTableHash]
		if childName == "" {
			continue // child table not loaded
		}

		child, err := c.db.GetCollection(childName)
		if err != nil {
			continue
		}

		parentValues := make([]fkValue, len(group.Pairs))
		if group.TargetTableHash == catalog.HashIdentifier("GRAPH_NODES") {
			// GRAPH_NODES.id is the durable numeric graph identity, not the
			// application record ID. Resolve the parent record's existing node
			// through the database-owned forward map before matching children.
			if len(group.Pairs) != 1 || group.Pairs[0].TargetColHash != catalog.HashIdentifier("id") {
				return nil, fmt.Errorf("foreign key violation: GRAPH_NODES references must target id")
			}
			var nodeID uint64
			var nodeErr error
			if epoch := epochFromContext(ctx); epoch != nil {
				nodeID, nodeErr = epoch.LookupNodeID(ctx, c.name, id)
			} else {
				nodeID, nodeErr = c.db.GetNodeID(ctx, c.name, id)
			}
			if nodeErr != nil || nodeID == 0 {
				return nil, fmt.Errorf("foreign key violation: graph node for %s.%s is unavailable", c.name, id)
			}
			parentValues[0] = fkValue{Value: strconv.FormatUint(nodeID, 10)}
		} else {
			for i, pair := range group.Pairs {
				parentValues[i] = recordValueByHash(parentRecord, pair.TargetColHash)
			}
		}
		var matchingIDs []string
		childRecords, listErr := recordsVisibleInContext(ctx, child)
		if listErr != nil {
			continue
		}
		for _, record := range childRecords {
			match := true
			null := false
			for i, pair := range group.Pairs {
				value := recordValueByHash(record, pair.SourceColHash)
				if value.Null {
					null = true
					break
				}
				if parentValues[i].Null {
					match = false
					break
				}
				if group.TargetTableHash == catalog.HashIdentifier("GRAPH_NODES") {
					nodeID, resolveErr := c.resolveGraphNodeFKValue(ctx, value.Value)
					if resolveErr != nil || strconv.FormatUint(nodeID, 10) != parentValues[i].Value {
						match = false
						break
					}
				} else if value.Value != parentValues[i].Value {
					match = false
					break
				}
			}
			if match && !null {
				matchingIDs = append(matchingIDs, record.ID)
			}
		}

		switch group.OnDelete {
		case catalog.OnDeleteCascade:
			for _, childID := range matchingIDs {
				cascades = append(cascades, cascadeOp{
					collectionName: childName,
					recordID:       childID,
					action:         catalog.OnDeleteCascade,
				})
			}
		case catalog.OnDeleteSetNull:
			for _, childID := range matchingIDs {
				colVals := make(map[string]cascadeColumnValue, len(group.Pairs))
				for _, pair := range group.Pairs {
					srcCol := resolveFKSourceCol(child, pair.SourceColHash)
					if srcCol != "" {
						colVals[srcCol] = cascadeColumnValue{Null: true}
					}
				}
				cascades = append(cascades, cascadeOp{
					collectionName: childName,
					recordID:       childID,
					action:         catalog.OnDeleteSetNull,
					columnValues:   colVals,
				})
			}
		case catalog.OnDeleteSetDefault:
			for _, childID := range matchingIDs {
				colVals := make(map[string]cascadeColumnValue, len(group.Pairs))
				for _, pair := range group.Pairs {
					srcCol := resolveFKSourceCol(child, pair.SourceColHash)
					if srcCol == "" {
						continue
					}
					if childCfg := child.Config(); childCfg.ColumnDefaults != nil {
						if defVal, ok := childCfg.ColumnDefaults[srcCol]; ok {
							colVals[srcCol] = cascadeColumnValue{Value: defVal}
						}
					}
				}
				if len(colVals) == 0 {
					return nil, fmt.Errorf(
						"foreign key violation: ON DELETE SET DEFAULT requires DEFAULT values on FK columns in %s",
						childName)
				}
				cascades = append(cascades, cascadeOp{
					collectionName: childName,
					recordID:       childID,
					action:         catalog.OnDeleteSetDefault,
					columnValues:   colVals,
				})
			}
		case catalog.OnDeleteRestrict, catalog.OnDeleteNoAction:
			// NO ACTION is deliberately immediate; deferred constraint timing is
			// not supported by the current catalog/transaction contract.
			if len(matchingIDs) > 0 {
				return nil, fmt.Errorf(
					"foreign key violation: cannot delete %s from %s because %d row(s) in %s reference it",
					id, c.name, len(matchingIDs), childName)
			}
		}
	}
	return cascades, nil
}

// resolveFKSourceCol resolves a source column hash in the child collection
// to its metadata key name.
func resolveFKSourceCol(child *Collection, colHash uint64) string {
	cfg := child.Config()
	if cfg.MetadataSchema != nil {
		for name := range cfg.MetadataSchema {
			if catalog.HashIdentifier(name) == colHash {
				return name
			}
		}
	}
	for _, field := range cfg.IndexedFields {
		if catalog.HashIdentifier(field) == colHash {
			return field
		}
	}
	return ""
}

// executeCascadeMutation executes a cascading delete or update on a child record.
func (c *Collection) executeCascadeMutation(ctx context.Context, op cascadeOp) {
	child, err := c.db.GetCollection(op.collectionName)
	if err != nil {
		return
	}
	switch {
	case op.action == catalog.OnDeleteSetNull || op.action == catalog.OnDeleteSetDefault:
		// SET NULL / SET DEFAULT: update child's FK columns.
		updateMeta := make(map[string]interface{}, len(op.columnValues))
		for col, val := range op.columnValues {
			if val.Null {
				updateMeta[col] = nil // SET NULL
			} else {
				updateMeta[col] = parseDefaultLiteral(val.Value) // SET DEFAULT
			}
		}
		_ = child.Update(ctx, op.recordID, nil, updateMeta)
	case op.updateCascade:
		// ON UPDATE CASCADE: apply the complete composite tuple in one
		// metadata update, never one component at a time.
		updateMeta := make(map[string]interface{}, len(op.columnValues))
		for col, val := range op.columnValues {
			if val.Null {
				updateMeta[col] = nil
			} else {
				updateMeta[col] = val.Value
			}
		}
		_ = child.Update(ctx, op.recordID, nil, updateMeta)
	case op.newFKValue != "":
		// Backward-compatible single-column ON UPDATE CASCADE.
		_ = child.Update(ctx, op.recordID, nil,
			map[string]interface{}{op.sourceCol: op.newFKValue})
	default:
		// ON DELETE CASCADE: remove the child.
		_ = child.Delete(ctx, op.recordID)
	}
}

// initMetadataIndexLocked ensures the metadata index maps are allocated.
func (c *Collection) initMetadataIndexLocked() {
	if c.metadataIndex != nil {
		return
	}
	c.metadataIndex = make(map[string]map[string][]string, len(c.config.IndexedFields))
	for _, field := range c.config.IndexedFields {
		c.metadataIndex[field] = make(map[string][]string)
	}
}

// addToMetadataIndex adds a record's indexed field values to the posting
// lists. Caller must have already successfully committed the record to
// storage. Safe for concurrent callers via metadataIndexMu.
func (c *Collection) addToMetadataIndex(id string, metadata map[string]interface{}) {
	if c.config == nil || len(c.config.IndexedFields) == 0 || metadata == nil {
		return
	}
	c.metadataIndexMu.Lock()
	defer c.metadataIndexMu.Unlock()
	c.initMetadataIndexLocked()
	for _, field := range c.config.IndexedFields {
		value, ok := metadata[field]
		if !ok {
			continue
		}
		fieldIdx := c.metadataIndex[field]
		for _, key := range metadataPostingKeys(value) {
			fieldIdx[key] = append(fieldIdx[key], id)
		}
	}
	epoch := c.metadataMutationEpoch.Load()
	c.metadataIndexBuiltAt = epoch
}

// removeFromMetadataIndex removes a record's indexed field values from the
// posting lists. Safe for concurrent callers via metadataIndexMu.
func (c *Collection) removeFromMetadataIndex(id string, metadata map[string]interface{}) {
	if c.metadataIndex == nil || c.config == nil || len(c.config.IndexedFields) == 0 || metadata == nil {
		return
	}
	c.metadataIndexMu.Lock()
	defer c.metadataIndexMu.Unlock()
	for _, field := range c.config.IndexedFields {
		value, ok := metadata[field]
		if !ok {
			continue
		}
		fieldIdx := c.metadataIndex[field]
		for _, key := range metadataPostingKeys(value) {
			ids := fieldIdx[key]
			for i, existing := range ids {
				if existing == id {
					fieldIdx[key] = append(ids[:i], ids[i+1:]...)
					break
				}
			}
		}
	}
	epoch := c.metadataMutationEpoch.Load()
	c.metadataIndexBuiltAt = epoch
}

func (c *Collection) hasIndexedMetadataField(field string) bool {
	if c == nil || c.config == nil {
		return false
	}
	for _, indexed := range c.config.IndexedFields {
		if indexed == field {
			return true
		}
	}
	return false
}

// lookupIndexedMetadata resolves one equality predicate through the
// collection's configured metadata posting lists. The lists are rebuilt
// lazily after a mutation, so steady-state filtered queries do not scan every
// vector or copy every vector payload just to construct an ordinal bitmap.
func (c *Collection) lookupIndexedMetadata(ctx context.Context, field string, value interface{}) ([]Record, bool, error) {
	if !c.hasIndexedMetadataField(field) {
		return nil, false, nil
	}
	c.metadataLookupIndexed.Add(1)

	c.metadataIndexMu.Lock()
	epoch := c.metadataMutationEpoch.Load()
	if c.metadataIndex == nil || c.metadataIndexBuiltAt != epoch {
		if err := c.rebuildMetadataIndexLocked(ctx, epoch); err != nil {
			c.metadataIndexMu.Unlock()
			return nil, true, err
		}
	}
	ids := make([]string, 0)
	seen := make(map[string]struct{})
	for _, key := range metadataPostingKeys(value) {
		for _, id := range c.metadataIndex[field][key] {
			if _, exists := seen[id]; exists {
				continue
			}
			seen[id] = struct{}{}
			ids = append(ids, id)
		}
	}
	c.metadataIndexMu.Unlock()

	c.metadataLookupCandidates.Add(uint64(len(ids)))
	records := make([]Record, 0, len(ids))
	for _, id := range ids {
		record, err := c.Get(ctx, id)
		if err != nil {
			// A concurrent delete can invalidate a posting after the snapshot.
			// Skipping it preserves correctness; the mutation epoch forces the
			// next lookup to rebuild the posting lists.
			if isNotFoundError(err) || errors.Is(err, ErrRecordNotFound) {
				continue
			}
			return nil, true, err
		}
		records = append(records, record)
	}
	return records, true, nil
}

func (c *Collection) rebuildMetadataIndexLocked(ctx context.Context, epoch uint64) error {
	c.metadataIndexRebuilds.Add(1)
	postings := make(map[string]map[string][]string, len(c.config.IndexedFields))
	for _, field := range c.config.IndexedFields {
		postings[field] = make(map[string][]string)
	}

	add := func(entry *index.VectorEntry) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		c.metadataIndexRecords.Add(1)
		for field, values := range postings {
			value, ok := entry.Metadata[field]
			if !ok {
				continue
			}
			for _, key := range metadataPostingKeys(value) {
				values[key] = append(values[key], entry.ID)
			}
		}
		return nil
	}

	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.closed {
		return ErrCollectionClosed
	}
	if c.shards != nil {
		for i := range c.shards {
			if err := c.shards[i].storage.Iterate(ctx, add); err != nil {
				return fmt.Errorf("build metadata index for shard %d: %w", i, err)
			}
		}
	} else if err := c.storage.Iterate(ctx, add); err != nil {
		return fmt.Errorf("build metadata index: %w", err)
	}

	c.metadataIndex = postings
	c.metadataIndexBuiltAt = epoch
	return nil
}

// metadataPostingKeys deliberately includes the SQL text-comparison key and,
// for numeric values, the cross-type key used by EqualityFilter. Candidate
// enumeration may be a superset; the authoritative predicate check removes
// false positives, while this prevents an index-induced false negative.
func metadataPostingKeys(value interface{}) []string {
	keys := []string{"text:" + recordMetaToString(value)}
	if numeric, ok := metadataNumericValue(value); ok {
		keys = append(keys, "number:"+strconv.FormatFloat(numeric, 'g', -1, 64))
	}
	return keys
}

func metadataNumericValue(value interface{}) (float64, bool) {
	switch typed := value.(type) {
	case int:
		return float64(typed), true
	case int8:
		return float64(typed), true
	case int16:
		return float64(typed), true
	case int32:
		return float64(typed), true
	case int64:
		return float64(typed), true
	case uint:
		return float64(typed), true
	case uint8:
		return float64(typed), true
	case uint16:
		return float64(typed), true
	case uint32:
		return float64(typed), true
	case uint64:
		return float64(typed), true
	case float32:
		return float64(typed), true
	case float64:
		return typed, true
	default:
		return 0, false
	}
}

// ListByMetadata returns records where the given metadata field equals the provided value.
func (c *Collection) ListByMetadata(ctx context.Context, field string, value interface{}) ([]Record, error) {
	records, indexed, err := c.lookupIndexedMetadata(ctx, field, value)
	if err != nil {
		return nil, err
	}
	if !indexed {
		c.metadataLookupFallback.Add(1)
		records, err = c.ListAll(ctx)
		if err != nil {
			return nil, err
		}
	}

	filtered, err := filter.NewEqualityFilter(field, value).Apply(ctx, filterEntriesFromRecords(records))
	if err != nil {
		return nil, err
	}

	result := make([]Record, 0, len(filtered))
	for _, entry := range filtered {
		result = append(result, Record{
			ID:       entry.ID,
			Vector:   cloneVector(entry.Vector),
			Metadata: cloneMetadata(entry.Metadata),
		})
	}
	return result, nil
}

// Count returns the exact number of live records in the collection.
func (c *Collection) Count(ctx context.Context) (int, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return 0, ErrCollectionClosed
	}

	// Sharded path: sum counts from all shards
	if c.shards != nil {
		total := 0
		for i := range c.shards {
			count, err := c.shards[i].storage.Count(ctx)
			if err != nil {
				return 0, fmt.Errorf("shard %d count: %w", i, err)
			}
			total += count
		}
		return total, nil
	}

	return c.storage.Count(ctx)
}

func (c *Collection) acquireWrite(ctx context.Context) (func(), error) {
	if c == nil {
		return func() {}, nil
	}
	if c.writes == nil {
		return func() {}, nil
	}
	return c.writes.acquire(ctx)
}

func (c *Collection) effectiveWriteConcurrency(requested int) int {
	if requested <= 0 {
		requested = 1
	}
	if c == nil || c.writes == nil {
		return requested
	}
	if limit := c.writes.maxParallelism(); limit > 0 && requested > limit {
		return limit
	}
	return requested
}

// Search finds the k most similar vectors to the query vector.
func (c *Collection) Search(ctx context.Context, vector []float32, k int) (*SearchResults, error) {
	return c.SearchWithGraphFilter(ctx, vector, k, nil)
}

// SearchWithGraphFilter finds the k most similar vectors, applying an optional graph bitset filter.
func (c *Collection) SearchWithGraphFilter(ctx context.Context, vector []float32, k int, filter GraphFilter) (*SearchResults, error) {
	return c.searchWithGraphFilterAndEf(ctx, vector, k, 0, filter)
}

// searchWithGraphFilterAndEf applies a per-query HNSW breadth when supported.
// Other index backends retain their normal Search behavior.
func (c *Collection) searchWithGraphFilterAndEf(ctx context.Context, vector []float32, k, ef int, filter GraphFilter) (*SearchResults, error) {

	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return nil, fmt.Errorf("collection is closed")
	}
	// Use local variables to avoid multiple pointer dereferences during inner loops
	idx := c.index
	shardIndexes := make([]index.Index, len(c.shards))
	for i := range c.shards {
		shardIndexes[i] = c.shards[i].index
	}

	var indexFilter index.GraphFilter
	if filter != nil {
		indexFilter = filter
	}

	// Validate input
	if len(vector) != c.config.Dimension {
		return nil, fmt.Errorf("query vector dimension %d does not match collection dimension %d",
			len(vector), c.config.Dimension)
	}

	if k <= 0 {
		return nil, fmt.Errorf("k must be positive, got %d", k)
	}

	// Start timing
	start := time.Now()
	indexVector := vectorForIndex(c.config.Metric, vector)

	// Search all shards in parallel and collect results
	type shardResult struct {
		err      error
		results  []*index.SearchResult
		shardIdx int // -1 for an unsharded collection
	}
	type indexedResult struct {
		result   *index.SearchResult
		shardIdx int
	}

	var resultsCh chan shardResult
	var wg sync.WaitGroup

	if c.shards != nil {
		// Sharded search: query all shards in parallel
		resultsCh = make(chan shardResult, len(c.shards))
		for i := range c.shards {
			wg.Add(1)
			go func(shardIdx int) {
				defer wg.Done()
				// Each shard only needs its local top-k; the parent merges all shard results.
				shardK := k
				shardFilter := indexFilter
				if factory, ok := filter.(interface{ ForShard(int) GraphFilter }); ok {
					shardFilter = factory.ForShard(shardIdx)
				}
				results, err := searchIndexWithEf(shardIndexes[shardIdx], ctx, indexVector, shardK, ef, shardFilter)
				resultsCh <- shardResult{results: results, err: err, shardIdx: shardIdx}
			}(i)
		}
	} else {
		// Non-sharded search
		resultsCh = make(chan shardResult, 1)
		wg.Add(1)
		go func() {
			defer wg.Done()
			results, err := searchIndexWithEf(idx, ctx, indexVector, k, ef, indexFilter)
			resultsCh <- shardResult{results: results, err: err, shardIdx: -1}
		}()
	}

	go func() {
		wg.Wait()
		close(resultsCh)
	}()

	// Collect all shard results
	var allResults []indexedResult
	for sr := range resultsCh {
		if sr.err != nil {
			// Handle empty index gracefully - just means no results from this shard
			if strings.Contains(sr.err.Error(), "index is empty") {
				continue
			}
			if c.metrics != nil {
				c.metrics.SearchErrors.Inc()
			}
			return nil, fmt.Errorf("shard search failed: %w", sr.err)
		}
		for _, result := range sr.results {
			allResults = append(allResults, indexedResult{result: result, shardIdx: sr.shardIdx})
		}
	}

	// Convert and merge results.
	// For sharded collections, parallelize the storage hydration step so the
	// merge phase does not become a sequential bottleneck.
	publicResults := make([]*SearchResult, len(allResults))
	distanceFunc, err := util.GetDistanceFunc(util.DistanceMetric(c.config.Metric))
	if err != nil {
		return nil, fmt.Errorf("resolve collection distance metric: %w", err)
	}
	hydrateResult := func(i int, indexed indexedResult) {
		r := indexed.result
		ordinalOnly := r.ID == ""
		result := &SearchResult{
			ID:      r.ID,
			Score:   r.Score,
			Version: r.Version,
		}
		if len(r.Vector) > 0 {
			result.Vector = r.Vector // index already cloned; we take ownership
		}
		if r.Metadata != nil {
			result.Metadata = r.Metadata // index already cloned; we take ownership
		}

		// Get full record from storage if needed.
		if result.Vector == nil || result.Metadata == nil || result.Version == 0 {
			var entry *index.VectorEntry
			var getErr error
			if c.shards != nil {
				shardStorage := c.shards[indexed.shardIdx].storage
				id := r.ID
				if id == "" {
					id, getErr = shardStorage.GetIDByOrdinal(ctx, r.Ordinal)
				}
				if getErr == nil {
					entry, getErr = shardStorage.Get(ctx, id)
				}
			} else {
				id := r.ID
				if id == "" {
					id, getErr = c.storage.GetIDByOrdinal(ctx, r.Ordinal)
				}
				if getErr == nil {
					entry, getErr = c.storage.Get(ctx, id)
				}
			}
			if getErr == nil {
				result.ID = entry.ID
				result.Version = entry.Version
				if result.Vector == nil {
					result.Vector = entry.Vector // cloneEntry already cloned
				}
				if result.Metadata == nil {
					result.Metadata = entry.Metadata // cloneEntry already cloned
				}
			} else if result.Metadata == nil {
				result.Metadata = map[string]interface{}{}
			}
		}
		// IVF-PQ retains only ordinal and PQ-code state. Once its candidate has
		// been hydrated from authoritative storage, compute the exact score used
		// by the public search contract rather than exposing a quantized distance.
		if ordinalOnly && len(result.Vector) > 0 {
			result.Score = distanceFunc(indexVector, vectorForIndex(c.config.Metric, result.Vector))
		}
		publicResults[i] = result
	}

	if c.shards != nil && len(allResults) > 0 {
		workerCount := len(c.shards)
		if workerCount > len(allResults) {
			workerCount = len(allResults)
		}
		if workerCount < 1 {
			workerCount = 1
		}

		var hydrateWG sync.WaitGroup
		sem := make(chan struct{}, workerCount)
		for i, result := range allResults {
			i, result := i, result
			hydrateWG.Add(1)
			go func() {
				defer hydrateWG.Done()
				sem <- struct{}{}
				defer func() { <-sem }()
				hydrateResult(i, result)
			}()
		}
		hydrateWG.Wait()
	} else {
		for i, result := range allResults {
			hydrateResult(i, result)
		}
	}

	normalizePublicSearchResults(c.config.Metric, publicResults)
	publicResults = selectTopKSearchResults(publicResults, k)

	// Update metrics
	if c.metrics != nil {
		c.metrics.SearchQueries.Inc()
		c.metrics.SearchLatency.Observe(time.Since(start).Seconds())
	}

	return &SearchResults{
		Results: publicResults,
		Took:    time.Since(start),
		Total:   len(publicResults),
	}, nil
}

func searchIndexWithEf(idx index.Index, ctx context.Context, query []float32, k, ef int, filter index.GraphFilter) ([]*index.SearchResult, error) {
	if ef > 0 {
		if searchable, ok := idx.(interface {
			SearchWithEf(context.Context, []float32, int, int, index.GraphFilter) ([]*index.SearchResult, error)
		}); ok {
			return searchable.SearchWithEf(ctx, query, k, ef, filter)
		}
	}
	return idx.Search(ctx, query, k, filter)
}

type searchResultMinHeap []*SearchResult

func (h searchResultMinHeap) Len() int { return len(h) }

func (h searchResultMinHeap) Less(i, j int) bool {
	return h[i].Score < h[j].Score
}

func (h searchResultMinHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *searchResultMinHeap) Push(x interface{}) {
	*h = append(*h, x.(*SearchResult))
}

func (h *searchResultMinHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}

func selectTopKSearchResults(results []*SearchResult, k int) []*SearchResult {
	if len(results) == 0 || k <= 0 {
		return nil
	}

	h := &searchResultMinHeap{}
	heap.Init(h)

	for _, r := range results {
		if h.Len() < k {
			heap.Push(h, r)
			continue
		}
		if r.Score <= (*h)[0].Score {
			continue
		}
		heap.Pop(h)
		heap.Push(h, r)
	}

	selected := make([]*SearchResult, h.Len())
	for i := len(selected) - 1; i >= 0; i-- {
		selected[i] = heap.Pop(h).(*SearchResult)
	}
	return selected
}

// Query returns a new query builder for this collection
func (c *Collection) Query(ctx context.Context) *QueryBuilder {
	return &QueryBuilder{
		ctx:        ctx,
		collection: c,
		limit:      10, // default
	}
}

// Stats returns collection statistics
func (c *Collection) Stats(ctx context.Context) *CollectionStats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var storageUsage int64
	var indexUsage int64
	var vectorCount int

	if c.shards != nil {
		// Aggregate stats from all shards
		for i := range c.shards {
			if su, err := c.shards[i].storage.MemoryUsage(ctx); err == nil {
				storageUsage += su
			}
			indexUsage += c.shards[i].index.MemoryUsage()
			vectorCount += c.shards[i].index.Size()
		}
	} else {
		storageUsage = c.storageMemoryUsageLocked(ctx)
		indexUsage = c.index.MemoryUsage()
		vectorCount = c.index.Size()
	}

	stats := &CollectionStats{
		Name:                 c.name,
		VectorCount:          vectorCount,
		Dimension:            c.config.Dimension,
		IndexType:            c.config.IndexType.String(),
		MemoryUsage:          storageUsage + indexUsage,
		HasQuantization:      c.config.Quantization != nil,
		HasMemoryLimit:       c.config.MemoryLimit > 0,
		MemoryMappingEnabled: c.config.EnableMMapping,
	}

	// Add enhanced memory statistics if memory manager is available
	if c.memoryManager != nil {
		usage := c.memoryManager.GetUsage()
		stats.MemoryStats = &CollectionMemoryStats{
			Total:         storageUsage + usage.Total,
			Storage:       storageUsage,
			Index:         usage.Indices,
			Cache:         usage.Caches,
			Quantized:     usage.Quantized,
			MemoryMapped:  usage.MemoryMapped,
			Limit:         usage.Limit,
			Available:     usage.Available,
			PressureLevel: calculatePressureLevel(storageUsage+usage.Total, usage.Limit),
			Timestamp:     usage.Timestamp,
		}
	} else {
		stats.MemoryStats = &CollectionMemoryStats{
			Total:         storageUsage + indexUsage,
			Storage:       storageUsage,
			Index:         indexUsage,
			Limit:         c.config.MemoryLimit,
			PressureLevel: calculatePressureLevel(storageUsage+indexUsage, c.config.MemoryLimit),
			Timestamp:     time.Now(),
		}
	}
	if rawProfile := c.DebugRawVectorStoreProfile(); rawProfile != nil {
		stats.RawVectorStoreStats = &RawVectorStoreStats{
			Backend:             rawProfile["backend"].(string),
			VectorCount:         rawProfile["vector_count"].(int),
			Dimension:           rawProfile["dimension"].(int),
			BytesPerVector:      rawProfile["bytes_per_vector"].(int),
			MemoryUsage:         rawProfile["memory_usage"].(int64),
			ReservedBytes:       rawProfile["reserved_bytes"].(int64),
			ReservedDataBytes:   rawProfile["reserved_data_bytes"].(int64),
			ReservedMetaBytes:   rawProfile["reserved_meta_bytes"].(int64),
			ReservedGuardBytes:  rawProfile["reserved_guard_bytes"].(int64),
			LiveBytes:           rawProfile["live_bytes"].(int64),
			FreeBytes:           rawProfile["free_bytes"].(int64),
			CapacityUtilization: rawProfile["capacity_utilization"].(float64),
		}
	}

	// Add optimization status
	stats.OptimizationStatus = &OptimizationStatus{
		InProgress:       c.optimizationInProgress,
		LastOptimization: c.lastOptimization,
		CanOptimize:      !c.closed && !c.optimizationInProgress,
	}

	// Ordinal statistics
	if c.shards != nil {
		var liveCount int
		var nextOrdinal uint32
		for i := range c.shards {
			if cnt, err := c.shards[i].storage.Count(ctx); err == nil {
				liveCount += cnt
			}
			if no, err := c.shards[i].storage.NextOrdinal(ctx); err == nil && no > nextOrdinal {
				nextOrdinal = no
			}
		}
		stats.LiveRecordCount = liveCount
		stats.NextOrdinal = nextOrdinal
		if nextOrdinal > 0 {
			stats.OrdinalUtilization = float64(liveCount) / float64(nextOrdinal)
		}
	} else {
		if cnt, err := c.storage.Count(ctx); err == nil {
			stats.LiveRecordCount = cnt
		}
		if no, err := c.storage.NextOrdinal(ctx); err == nil {
			stats.NextOrdinal = no
			if no > 0 {
				stats.OrdinalUtilization = float64(stats.LiveRecordCount) / float64(no)
			}
		}
	}

	return stats
}

// GetMemoryUsage returns current memory usage statistics for the collection
func (c *Collection) GetMemoryUsage(ctx context.Context) (*memory.MemoryUsage, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return nil, ErrCollectionClosed
	}

	var totalStorage int64
	var totalIndex int64

	if c.shards != nil {
		// Aggregate memory usage from all shards
		for i := range c.shards {
			if su, err := c.shards[i].storage.MemoryUsage(ctx); err == nil {
				totalStorage += su
			}
			totalIndex += c.shards[i].index.MemoryUsage()
		}
	} else {
		totalStorage = c.storageMemoryUsageLocked(ctx)
		totalIndex = c.index.MemoryUsage()
	}

	if c.memoryManager == nil {
		usage := &memory.MemoryUsage{
			Total:     totalStorage + totalIndex,
			Indices:   totalIndex,
			Caches:    totalStorage,
			Timestamp: time.Now(),
		}
		return usage, nil
	}

	usage := c.memoryManager.GetUsage()
	usage.Total += totalStorage
	usage.Caches += totalStorage
	usage.Indices = totalIndex
	return &usage, nil
}

func (c *Collection) storageMemoryUsageLocked(ctx context.Context) int64 {
	if c.storage == nil {
		return 0
	}
	usage, err := c.storage.MemoryUsage(ctx)
	if err != nil {
		return 0
	}
	return usage
}

// DebugRawVectorStoreProfile exposes backend-specific raw vector storage stats
// for profiling and benchmarking.
func (c *Collection) DebugRawVectorStoreProfile() map[string]any {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.shards != nil {
		// Sharded collections do not support single-profile debugging
		return nil
	}

	if profiler, ok := c.index.(interface{ RawVectorStoreProfile() map[string]any }); ok {
		return profiler.RawVectorStoreProfile()
	}
	return nil
}

// SetMemoryLimit updates the memory limit for the collection
func (c *Collection) SetMemoryLimit(bytes int64) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return ErrCollectionClosed
	}

	// Sharded collections do not yet support per-shard memory limit management
	if c.shards != nil {
		return fmt.Errorf("SetMemoryLimit is not supported for sharded collections")
	}

	// Update config
	c.config.MemoryLimit = bytes

	// Update memory manager if it exists
	if c.memoryManager != nil {
		return c.memoryManager.SetLimit(bytes)
	}

	// If no memory manager exists and limit is set, create one
	if bytes > 0 {
		memConfig := memory.DefaultMemoryConfig()
		if c.config.MemoryConfig != nil {
			memConfig = *c.config.MemoryConfig
		}
		memConfig.MaxMemory = bytes
		memConfig.EnableMMap = c.config.EnableMMapping

		memManager := memory.NewManager(memConfig)

		// Register the index as a memory-mappable component if supported
		if mappable, ok := c.index.(memory.MemoryMappable); ok {
			if err := memManager.RegisterMemoryMappable(fmt.Sprintf("index_%s", c.name), mappable); err != nil {
				return fmt.Errorf("failed to register index for memory management: %w", err)
			}
		}

		// Start memory monitoring
		if err := memManager.Start(context.Background()); err != nil {
			return fmt.Errorf("failed to start memory manager: %w", err)
		}

		c.memoryManager = memManager
	}

	return nil
}

// TriggerGC forces garbage collection for the collection
func (c *Collection) TriggerGC() error {
	c.mu.RLock()
	closed := c.closed
	memManager := c.memoryManager
	c.mu.RUnlock()

	if closed {
		return ErrCollectionClosed
	}

	// Sharded collections do not yet support per-shard memory manager GC
	if c.shards != nil {
		// Fallback to runtime GC if no memory manager or for sharded collections
		memory.ForceGC()
		return nil
	}

	if memManager != nil {
		return memManager.TriggerGC()
	}

	// Fallback to runtime GC if no memory manager
	memory.ForceGC()
	return nil
}

// OptimizeCollection performs collection optimization including index rebuilding and memory optimization
func (c *Collection) OptimizeCollection(ctx context.Context, options *OptimizationOptions) error {
	// Check initial state and set optimization in progress
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return ErrCollectionClosed
	}

	if c.optimizationInProgress {
		c.mu.Unlock()
		return fmt.Errorf("optimization already in progress")
	}

	// Sharded collections do not support OptimizeCollection
	if c.shards != nil {
		c.mu.Unlock()
		return fmt.Errorf("OptimizeCollection is not supported for sharded collections")
	}

	// Set default options if not provided
	if options == nil {
		options = &OptimizationOptions{
			RebuildIndex:       true,
			OptimizeMemory:     true,
			CompactStorage:     true,
			UpdateQuantization: false,
		}
	}

	c.optimizationInProgress = true
	memManager := c.memoryManager
	hasQuantization := c.config.Quantization != nil
	c.mu.Unlock()

	// Ensure we reset optimization status on exit
	defer func() {
		c.mu.Lock()
		c.optimizationInProgress = false
		c.lastOptimization = time.Now()
		c.mu.Unlock()
	}()

	// Step 1: Optimize memory if requested
	if options.OptimizeMemory && memManager != nil {
		if err := memManager.HandleMemoryLimitExceeded(); err != nil {
			return fmt.Errorf("memory optimization failed: %w", err)
		}
	}

	// Step 2: Rebuild index if requested
	if options.RebuildIndex {
		if err := c.rebuildIndexOptimized(ctx, options); err != nil {
			return fmt.Errorf("index rebuild failed: %w", err)
		}
	}

	// Step 3: Update quantization if requested
	if options.UpdateQuantization && hasQuantization {
		if err := c.updateQuantization(ctx); err != nil {
			return fmt.Errorf("quantization update failed: %w", err)
		}
	}

	// Step 4: Compact storage if requested
	if options.CompactStorage {
		// Note: This would require storage layer support for compaction
		// For now, we'll just trigger GC
		if err := c.TriggerGC(); err != nil {
			return fmt.Errorf("storage compaction failed: %w", err)
		}
	}

	return nil
}

// rebuildIndexOptimized rebuilds the index with optimization considerations
func (c *Collection) rebuildIndexOptimized(ctx context.Context, options *OptimizationOptions) error {
	c.mu.Lock()
	if c.shards != nil {
		c.mu.Unlock()
		return fmt.Errorf("rebuildIndexOptimized is not supported for sharded collections")
	}
	autoIndexSelection := c.config.AutoIndexSelection
	currentType := c.config.IndexType
	hnswThreshold := c.config.AutoIndexThresholds.HNSWThreshold
	if hnswThreshold == 0 {
		hnswThreshold = DefaultHNSWThreshold
	}
	ivfpqThreshold := c.config.AutoIndexThresholds.IVFPQThreshold
	if ivfpqThreshold == 0 {
		ivfpqThreshold = DefaultIVFPQThreshold
	}
	currentSize := c.index.Size()
	c.mu.Unlock()

	if autoIndexSelection {
		optimalType := selectOptimalIndexType(currentSize, hnswThreshold, ivfpqThreshold)
		if optimalType != currentType {
			return c.switchIndexType(ctx, optimalType)
		}
	}

	// For optimization, we don't need to rebuild if the index is already populated
	// and we're not switching types. This avoids the duplicate insertion issue.
	if currentSize > 0 {
		return nil
	}

	// Only rebuild if index is empty (e.g., after loading from storage)
	return c.rebuildIndex(ctx)
}

// updateQuantization retrains quantization parameters with current data
func (c *Collection) updateQuantization(ctx context.Context) error {
	if c.shards != nil {
		return fmt.Errorf("updateQuantization is not supported for sharded collections")
	}

	if c.config.Quantization == nil {
		return fmt.Errorf("no quantization configured")
	}

	// Get all vectors for retraining
	vectors, err := c.getAllVectors(ctx)
	if err != nil {
		return fmt.Errorf("failed to get vectors for quantization update: %w", err)
	}

	if len(vectors) == 0 {
		return nil // Nothing to retrain
	}

	// Extract vector data for training
	trainingVectors := make([][]float32, len(vectors))
	for i, entry := range vectors {
		trainingVectors[i] = entry.Vector
	}

	// Create new quantizer and train it
	quantizer, err := quant.Create(c.config.Quantization)
	if err != nil {
		return fmt.Errorf("failed to create quantizer: %w", err)
	}

	if err := quantizer.Train(ctx, trainingVectors); err != nil {
		return fmt.Errorf("failed to train quantizer: %w", err)
	}

	// Update the index with the new quantizer
	return c.rebuildIndex(ctx)
}

// GetOptimizationStatus returns the current optimization status
func (c *Collection) GetOptimizationStatus() *OptimizationStatus {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return &OptimizationStatus{
		InProgress:       c.optimizationInProgress,
		LastOptimization: c.lastOptimization,
		CanOptimize:      !c.closed && !c.optimizationInProgress,
	}
}

// EnableMemoryMapping enables memory mapping for the collection's index
func (c *Collection) EnableMemoryMapping(path string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return ErrCollectionClosed
	}

	// Sharded collections do not support memory mapping
	if c.shards != nil {
		return fmt.Errorf("EnableMemoryMapping is not supported for sharded collections")
	}

	// Update config
	c.config.EnableMMapping = true

	// Enable memory mapping on the index if supported
	if mappable, ok := c.index.(memory.MemoryMappable); ok {
		if mappable.CanMemoryMap() {
			return mappable.EnableMemoryMapping(path)
		}
	}

	return fmt.Errorf("index does not support memory mapping")
}

// DisableMemoryMapping disables memory mapping for the collection's index
func (c *Collection) DisableMemoryMapping() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return ErrCollectionClosed
	}

	// Sharded collections do not support memory mapping
	if c.shards != nil {
		return fmt.Errorf("DisableMemoryMapping is not supported for sharded collections")
	}

	// Update config
	c.config.EnableMMapping = false

	// Disable memory mapping on the index if supported
	if mappable, ok := c.index.(memory.MemoryMappable); ok {
		if mappable.IsMemoryMapped() {
			return mappable.DisableMemoryMapping()
		}
	}

	return nil
}

// Close shuts down the collection
func (c *Collection) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil
	}

	var errors []error
	if c.asyncIndex != nil {
		if err := c.asyncIndex.close(); err != nil {
			errors = append(errors, fmt.Errorf("asynchronous index close: %w", err))
		}
	}

	// Stop memory manager if it exists
	if c.memoryManager != nil {
		if err := c.memoryManager.Stop(); err != nil {
			errors = append(errors, err)
		}
	}

	// Close shards if sharded collection
	if c.shards != nil {
		for i := range c.shards {
			c.shards[i].mu.Lock()
			if c.shards[i].index != nil {
				if err := c.shards[i].index.Close(); err != nil {
					errors = append(errors, fmt.Errorf("shard %d index close: %w", i, err))
				}
			}
			if c.shards[i].storage != nil {
				if err := c.shards[i].storage.Close(); err != nil {
					errors = append(errors, fmt.Errorf("shard %d storage close: %w", i, err))
				}
			}
			c.shards[i].mu.Unlock()
		}
	} else {
		// Non-sharded collection
		if c.index != nil {
			if err := c.index.Close(); err != nil {
				errors = append(errors, err)
			}
		}
		if c.storage != nil {
			if err := c.storage.Close(); err != nil {
				errors = append(errors, err)
			}
		}
	}
	if mutationState := c.mutationState.Swap(nil); mutationState != nil {
		mutationState.close()
	}

	// Close attached graph so its memory pools release background goroutines.
	if c.graph != nil {
		if err := c.graph.Close(); err != nil {
			errors = append(errors, fmt.Errorf("graph close: %w", err))
		}
		c.graph = nil
	}

	c.closed = true

	if len(errors) > 0 {
		return fmt.Errorf("errors during collection shutdown: %v", errors)
	}

	return nil
}

// SaveIndex persists the collection's index to disk
func (c *Collection) SaveIndex(ctx context.Context, path string) error {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return fmt.Errorf("collection is closed")
	}

	// Sharded collections do not support SaveIndex
	if c.shards != nil {
		return fmt.Errorf("SaveIndex is not supported for sharded collections")
	}

	return c.index.SaveToDisk(ctx, path)
}

// LoadIndex loads the collection's index from disk
func (c *Collection) LoadIndex(ctx context.Context, path string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return fmt.Errorf("collection is closed")
	}

	// Sharded collections do not support LoadIndex
	if c.shards != nil {
		return fmt.Errorf("LoadIndex is not supported for sharded collections")
	}

	return c.index.LoadFromDisk(ctx, path)
}

// GetIndexMetadata returns metadata about the collection's index
func (c *Collection) GetIndexMetadata() *index.PersistenceMetadata {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return nil
	}

	// Sharded collections do not support GetIndexMetadata
	if c.shards != nil {
		return nil
	}

	// Get HNSW-specific metadata if available
	if hnswIndex, ok := c.index.(interface {
		GetPersistenceMetadata() *index.PersistenceMetadata
	}); ok {
		return hnswIndex.GetPersistenceMetadata()
	}

	return nil
}

// checkAndSwitchIndexType checks if the index type should be changed based on collection size
func (c *Collection) checkAndSwitchIndexType(ctx context.Context) error {
	// Sharded collections do not support auto index switching
	if c.shards != nil {
		return fmt.Errorf("checkAndSwitchIndexType is not supported for sharded collections")
	}

	currentSize := c.index.Size()
	hnswThreshold := c.config.AutoIndexThresholds.HNSWThreshold
	if hnswThreshold == 0 {
		hnswThreshold = DefaultHNSWThreshold
	}
	ivfpqThreshold := c.config.AutoIndexThresholds.IVFPQThreshold
	if ivfpqThreshold == 0 {
		ivfpqThreshold = DefaultIVFPQThreshold
	}
	optimalType := selectOptimalIndexType(currentSize, hnswThreshold, ivfpqThreshold)

	// If the optimal type is different from current, switch
	if optimalType != c.config.IndexType {
		return c.switchIndexType(ctx, optimalType)
	}

	return nil
}

// switchIndexType rebuilds the index with a new type
func (c *Collection) switchIndexType(ctx context.Context, newType IndexType) error {
	// Sharded collections do not support index type switching
	if c.shards != nil {
		return fmt.Errorf("switchIndexType is not supported for sharded collections")
	}

	// Get all vectors from current index
	vectors, err := c.getAllVectors(ctx)
	if err != nil {
		return fmt.Errorf("failed to get vectors for index switch: %w", err)
	}

	provider, _ := c.storage.(interface {
		GetByOrdinal(uint32) ([]float32, error)
		Distance([]float32, uint32) (float32, error)
	})
	updatedConfig := *c.config
	updatedConfig.IndexType = newType
	newIndex, err := buildIndexForEntries(ctx, &updatedConfig, provider, vectors)
	if err != nil {
		return fmt.Errorf("failed to build new index: %w", err)
	}

	// Close old index and switch atomically under the collection lock
	// so concurrent readers always see a consistent index/config pair.
	c.mu.Lock()
	c.index.Close()
	c.index = newIndex
	c.config.IndexType = newType
	c.mu.Unlock()

	return nil
}

// getAllVectors retrieves all vectors from the storage layer
func (c *Collection) getAllVectors(ctx context.Context) ([]*index.VectorEntry, error) {
	var vectors []*index.VectorEntry

	if c.shards != nil {
		for i := range c.shards {
			err := c.shards[i].storage.Iterate(ctx, func(entry *index.VectorEntry) error {
				vectors = append(vectors, entry)
				return nil
			})
			if err != nil {
				return nil, fmt.Errorf("failed to iterate shard %d storage: %w", i, err)
			}
		}
	} else {
		err := c.storage.Iterate(ctx, func(entry *index.VectorEntry) error {
			vectors = append(vectors, entry)
			return nil
		})
		if err != nil {
			return nil, fmt.Errorf("failed to iterate storage: %w", err)
		}
	}

	return vectors, nil
}

func recordFromIndexEntry(entry *index.VectorEntry) Record {
	if entry == nil {
		return Record{}
	}

	return Record{
		ID:       entry.ID,
		Ordinal:  entry.Ordinal,
		Vector:   entry.Vector,
		Metadata: entry.Metadata,
		Version:  entry.Version,
	}
}

func filterEntriesFromRecords(records []Record) []*filter.VectorEntry {
	entries := make([]*filter.VectorEntry, 0, len(records))
	for _, record := range records {
		entries = append(entries, &filter.VectorEntry{
			ID:       record.ID,
			Vector:   record.Vector,
			Metadata: record.Metadata,
		})
	}
	return entries
}

func cloneVector(vector []float32) []float32 {
	if vector == nil {
		return nil
	}
	return append([]float32(nil), vector...)
}

func cloneMetadata(metadata map[string]interface{}) map[string]interface{} {
	if metadata == nil {
		return nil
	}

	cloned := make(map[string]interface{}, len(metadata))
	for k, v := range metadata {
		cloned[k] = cloneMetadataValue(v)
	}
	return cloned
}

func cloneMetadataValue(value interface{}) interface{} {
	switch v := value.(type) {
	case map[string]interface{}:
		out := make(map[string]interface{}, len(v))
		for key, item := range v {
			out[key] = cloneMetadataValue(item)
		}
		return out
	case []interface{}:
		out := make([]interface{}, len(v))
		for i := range v {
			out[i] = cloneMetadataValue(v[i])
		}
		return out
	case []string:
		return append([]string(nil), v...)
	case []byte:
		return append([]byte(nil), v...)
	default:
		return value
	}
}

// RegisterInsertHook adds a hook to be called before a vector is inserted.
// Maximum of 4 hooks can be registered.
func (c *Collection) RegisterInsertHook(hook InsertHook) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if hook == nil {
		return fmt.Errorf("cannot register nil hook")
	}
	if len(c.insertHooks) >= 4 {
		return fmt.Errorf("maximum of 4 insert hooks allowed")
	}
	c.insertHooks = append(c.insertHooks, hook)
	return nil
}

// RegisterDeleteHook adds a callback invoked before a vector is deleted.
//
// Deprecated: delete hooks use the legacy physical graph-transaction path and
// are not part of the combined record/graph WAL transaction. New code should
// use explicit transactional graph operations instead.
func (c *Collection) RegisterDeleteHook(hook DeleteHook) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if hook == nil {
		return fmt.Errorf("cannot register nil hook")
	}
	if len(c.deleteHooks) >= 4 {
		return fmt.Errorf("maximum of 4 delete hooks allowed")
	}
	c.deleteHooks = append(c.deleteHooks, hook)
	return nil
}

// validate checks if the collection configuration is valid
func (config *CollectionConfig) validate() error {
	if config.Dimension < 0 {
		return fmt.Errorf("dimension must be positive, got %d", config.Dimension)
	}
	// dimension == 0 is valid: metadata-only collection (WithMetadataOnly)

	if config.M <= 0 {
		return fmt.Errorf("M must be positive, got %d", config.M)
	}

	if config.EfConstruction <= 0 {
		return fmt.Errorf("EfConstruction must be positive, got %d", config.EfConstruction)
	}

	if config.EfSearch <= 0 {
		return fmt.Errorf("EfSearch must be positive, got %d", config.EfSearch)
	}

	switch config.RawVectorStore {
	case "", "memory", "slabby":
	default:
		return fmt.Errorf("unsupported raw vector store backend: %s", config.RawVectorStore)
	}
	if config.RawVectorStore == "slabby" && config.RawStoreCap <= 0 {
		return fmt.Errorf("slabby raw store capacity must be positive, got %d", config.RawStoreCap)
	}

	// Validate quantization configuration if provided
	if config.Quantization != nil {
		if err := config.Quantization.Validate(); err != nil {
			return fmt.Errorf("invalid quantization config: %w", err)
		}
	}

	// Validate memory configuration
	if config.MemoryLimit < 0 {
		return fmt.Errorf("memory limit must be non-negative, got %d", config.MemoryLimit)
	}

	if config.MemoryConfig != nil {
		if config.MemoryConfig.MaxMemory < 0 {
			return fmt.Errorf("max memory must be non-negative, got %d", config.MemoryConfig.MaxMemory)
		}
		if config.MemoryConfig.MonitorInterval <= 0 {
			return fmt.Errorf("monitor interval must be positive, got %v", config.MemoryConfig.MonitorInterval)
		}
		if config.MemoryConfig.GCThreshold < 0 || config.MemoryConfig.GCThreshold > 1 {
			return fmt.Errorf("GC threshold must be between 0 and 1, got %f", config.MemoryConfig.GCThreshold)
		}
		if config.MemoryConfig.MMapThreshold < 0 {
			return fmt.Errorf("mmap threshold must be non-negative, got %d", config.MemoryConfig.MMapThreshold)
		}
	}

	// Validate metadata schema if provided
	if config.MetadataSchema != nil {
		if err := config.MetadataSchema.Validate(); err != nil {
			return fmt.Errorf("invalid metadata schema: %w", err)
		}
	}

	// Validate indexed fields
	if len(config.IndexedFields) > 0 && config.MetadataSchema != nil {
		for _, field := range config.IndexedFields {
			if _, exists := config.MetadataSchema[field]; !exists {
				return fmt.Errorf("indexed field '%s' not found in metadata schema", field)
			}
		}
	}
	for _, index := range config.JSONIndexes {
		if strings.TrimSpace(index.Name) == "" || strings.TrimSpace(index.Column) == "" || strings.TrimSpace(index.Path) == "" {
			return fmt.Errorf("JSON index requires name, column, and path")
		}
		if config.MetadataSchema != nil {
			fieldType, ok := config.MetadataSchema[index.Column]
			if !ok {
				return fmt.Errorf("JSON index column %q not found in metadata schema", index.Column)
			}
			if fieldType != JSONField && fieldType != JSONBField {
				return fmt.Errorf("JSON index column %q must be JSON or JSONB", index.Column)
			}
		}
	}

	// Apply default batch configuration if not set (for backward compatibility)
	if config.BatchConfig.ChunkSize == 0 {
		config.BatchConfig = DefaultBatchConfig()
	}

	// Validate batch configuration
	if config.BatchConfig.ChunkSize <= 0 {
		return fmt.Errorf("batch chunk size must be positive, got %d", config.BatchConfig.ChunkSize)
	}
	if config.BatchConfig.MaxConcurrency <= 0 {
		return fmt.Errorf("batch max concurrency must be positive, got %d", config.BatchConfig.MaxConcurrency)
	}
	if config.BatchConfig.TimeoutPerChunk <= 0 {
		return fmt.Errorf("batch timeout per chunk must be positive, got %v", config.BatchConfig.TimeoutPerChunk)
	}

	return nil
}

// LookupNodeID looks up a record ID within this collection and returns its underlying
// system-scoped GraphNodeID. Returns an error if the record is not found.
func (c *Collection) LookupNodeID(ctx context.Context, id string) (uint64, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return 0, ErrCollectionClosed
	}
	// For sharded collections, route to the correct shard.
	if len(c.shards) > 0 {
		si := shardForID(id)
		sname := shardName(c.name, si)
		return c.db.storage.GetNodeID(ctx, sname, id)
	}
	return c.db.storage.GetNodeID(ctx, c.name, id)
}
