// Package libravdb provides a high-performance vector database library
// optimized for Go applications with HNSW indexing and LSM storage.
package libravdb

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/xDarkicex/libravdb/internal/catalog"
	"github.com/xDarkicex/libravdb/internal/index"
	"github.com/xDarkicex/libravdb/internal/obs"
	"github.com/xDarkicex/libravdb/internal/quant"
	"github.com/xDarkicex/libravdb/internal/storage"
	"github.com/xDarkicex/libravdb/internal/storage/singlefile"
	"github.com/xDarkicex/memory"
)

// Logger is the logging interface accepted by the database.
// It is compatible with the standard library's log.Printf signature.
type Logger interface {
	Printf(format string, v ...interface{})
}

// Database represents the main vector database instance
type Database struct {
	storage       storage.Engine
	logger        Logger
	collections   map[string]*Collection
	bridge        *indexPersistenceBridge
	metrics       *obs.Metrics
	health        *obs.HealthChecker
	healthMonitor *SystemHealthMonitorImpl
	config        *Config
	scratchPool   *sync.Pool
	catalog       *catalog.Catalog
	quantRegistry *quant.Registry
	// Bounded in-memory query feedback for future calibration aggregation.
	costModelStats    *costModelStats
	activeSnaps       activeSnapshots
	temporalCache     *temporalIndexCache
	sqlPlanCache      *sqlPlanCache
	sqlStats          *sqlStatsCounters
	catalogGeneration atomic.Uint64
	autoIncrementMu   sync.Mutex
	autoIncrementNext map[string]uint64
	defaultGraphMu    sync.Mutex
	defaultGraph      Graph
	mu                sync.RWMutex
	closed            bool
}

// Config holds database-wide configuration
type Config struct {
	Logger               Logger
	StoragePath          string
	MaxCollections       int
	MaxConcurrentWrites  int
	MaxWriteQueueDepth   int
	AsyncIndexQueueDepth int
	AsyncIndexWorkers    int
	MetricsEnabled       bool
	TracingEnabled       bool
	Durability           DurabilityMode
	Temporal             TemporalConfig
	TemporalANN          TemporalANNConfig
	maxWritesExplicit    bool
	writeQueueExplicit   bool
}

// DurabilityMode controls when a successful write may be acknowledged.
type DurabilityMode uint8

const (
	// DurabilitySynchronous acknowledges writes only after the WAL group reaches
	// stable storage through file.Sync.
	DurabilitySynchronous DurabilityMode = iota
	// DurabilityUnsafeNoSync is an explicit benchmark-only mode. A successful
	// write may still be lost after power failure or kernel crash.
	DurabilityUnsafeNoSync
)

// WithTemporalANNCache enables the optional temporal ANN cache with byte/entry limits.
func WithTemporalANNCache(maxBytes int64, maxEntries int) Option {
	return func(c *Config) error {
		c.TemporalANN.MaxBytes = maxBytes
		c.TemporalANN.MaxEntries = maxEntries
		return nil
	}
}

// Open opens a Database at the configured path, creating it if necessary.
func Open(opts ...Option) (*Database, error) {
	config := &Config{
		StoragePath:         "./data",
		MetricsEnabled:      true,
		TracingEnabled:      false,
		MaxCollections:      100,
		MaxConcurrentWrites: defaultMaxConcurrentWrites(),
		// The default queue must absorb normal transaction fan-in (including
		// concurrent epoch commits) without rejecting otherwise valid writes.
		// Callers that need a tighter admission bound can still set it
		// explicitly with WithMaxWriteQueueDepth.
		MaxWriteQueueDepth:   128,
		AsyncIndexQueueDepth: 0,
		AsyncIndexWorkers:    min(4, runtime.GOMAXPROCS(0)),
		Durability:           DurabilitySynchronous,
	}

	// Apply options
	for _, opt := range opts {
		if err := opt(config); err != nil {
			return nil, fmt.Errorf("failed to apply option: %w", err)
		}
	}
	if config.AsyncIndexQueueDepth > 0 {
		if !config.maxWritesExplicit {
			config.MaxConcurrentWrites = 32
		}
		if !config.writeQueueExplicit {
			config.MaxWriteQueueDepth = config.AsyncIndexQueueDepth
		}
	}
	if err := recoverMigrate(config.StoragePath); err != nil {
		return nil, fmt.Errorf("recover interrupted migration: %w", err)
	}

	// Create the index persistence bridge so persisted indexes can be
	// deserialized during recovery (avoiding full rebuild from records).
	bridge := &indexPersistenceBridge{cache: make(map[string]index.Index)}
	storageOptions := []singlefile.Option{
		singlefile.WithIndexSnapshotProvider(bridge),
		singlefile.WithWALSync(config.Durability == DurabilitySynchronous),
	}
	if config.AsyncIndexQueueDepth > 0 {
		storageOptions = append(storageOptions, singlefile.WithWALGroupCommitTarget(min(28, config.MaxConcurrentWrites), 5*time.Millisecond))
	}
	storageEngine, err := singlefile.New(config.StoragePath, storageOptions...)

	if err != nil {
		if errors.Is(err, storage.ErrV1FormatMigrationRequired) {
			if err := Migrate(context.Background(), config.StoragePath); err != nil {
				bridge.closeCachedIndexes()
				return nil, fmt.Errorf("auto-migration failed: %w", err)
			}
			// Retry opening the newly migrated database
			storageEngine, err = singlefile.New(config.StoragePath, storageOptions...)
			if err != nil {
				bridge.closeCachedIndexes()
				return nil, fmt.Errorf("failed to open database after migration: %w", err)
			}
		} else {
			bridge.closeCachedIndexes()
			return nil, fmt.Errorf("failed to initialize storage engine: %w", err)
		}
	}

	// Initialize observability
	var metrics *obs.Metrics
	if config.MetricsEnabled {
		metrics = obs.NewMetrics()
	}

	db := &Database{
		collections:       make(map[string]*Collection),
		storage:           storageEngine,
		bridge:            bridge,
		metrics:           metrics,
		config:            config,
		logger:            config.Logger,
		quantRegistry:     quant.NewRegistry(),
		costModelStats:    newCostModelStats(2048),
		autoIncrementNext: make(map[string]uint64),
		sqlPlanCache:      newSQLPlanCache(256),
		sqlStats:          newSQLStatsCounters(),
		scratchPool: &sync.Pool{
			New: func() interface{} {
				arena, err := memory.NewArena(1024*1024, 64)
				if err != nil {
					panic(fmt.Sprintf("failed to allocate scratch arena: %v", err))
				}
				return arena
			},
		},
	}

	// Restore SQL edge-kind names and direction metadata before collections and
	// queries are exposed. Numeric kinds are stored in the single-file state/WAL;
	// graph instances receive the direction metadata after they are recreated.
	durableEdgeKinds := make(map[string]storage.EdgeKindDefinition)
	if edgeKinds, ok := storageEngine.(storage.EdgeKindDefinitionStore); ok {
		kinds, err := edgeKinds.ListEdgeKindDefinitions()
		if err != nil {
			_ = storageEngine.Close()
			bridge.closeCachedIndexes()
			return nil, fmt.Errorf("load durable graph edge kinds: %w", err)
		}
		for name, definition := range kinds {
			durableEdgeKinds[name] = definition
			if !RegisterEdgeKindWithDirection(name, definition.Kind, definition.Undirected) {
				_ = storageEngine.Close()
				bridge.closeCachedIndexes()
				return nil, fmt.Errorf("restore graph edge kind %q=%d: runtime registry conflict", name, definition.Kind)
			}
		}
	} else if edgeKinds, ok := storageEngine.(storage.EdgeKindStore); ok {
		kinds, err := edgeKinds.ListEdgeKinds()
		if err != nil {
			_ = storageEngine.Close()
			bridge.closeCachedIndexes()
			return nil, fmt.Errorf("load durable graph edge kinds: %w", err)
		}
		for name, kind := range kinds {
			durableEdgeKinds[name] = storage.EdgeKindDefinition{Kind: kind}
			if !RegisterEdgeKind(name, kind) {
				_ = storageEngine.Close()
				bridge.closeCachedIndexes()
				return nil, fmt.Errorf("restore graph edge kind %q=%d: runtime registry conflict", name, kind)
			}
		}
	}

	// Load catalog from storage engine if it was persisted in a previous session.
	// Falls back to the sidecar file, then to an empty catalog for fresh databases.
	if data, ok := db.storage.(interface{ CatalogData() []byte }); ok {
		if catData := data.CatalogData(); len(catData) > 0 {
			if cat, err := catalog.Load(catData, db.quantRegistry); err == nil {
				db.catalog = cat
			}
		}
	}
	if db.catalog == nil {
		// Fresh database — build a valid empty catalog rather than leaving nil.
		// A nil catalog makes every query fail with "catalog not initialized",
		// which breaks e.g. pgx's Ping ("-- ping") on a brand-new database.
		if cat, err := catalog.Load(catalog.NewBuilder().Build(), db.quantRegistry); err == nil {
			db.catalog = cat
		}
	}
	// Catalog objects are immutable snapshots. The generation lets compiled
	// SQL plans be reused only while the schema they were bound against is
	// still current.
	db.catalogGeneration.Store(1)

	// Wire the bridge back to the database so SerializeIndex can access
	// collection indexes during checkpoint.
	bridge.mu.Lock()
	bridge.db = db
	bridge.mu.Unlock()

	// Initialize health checker
	db.health = obs.NewHealthChecker(db)

	// Start the background health monitor. Registers a storage engine
	// liveness check and begins periodic monitoring with callbacks.
	db.healthMonitor = NewSystemHealthMonitor(30 * time.Second)
	db.healthMonitor.RegisterHealthCheck("storage", func(ctx context.Context) (HealthLevel, error) {
		_, err := storageEngine.ListCollections()
		if err != nil {
			return HealthCritical, fmt.Errorf("storage list: %w", err)
		}
		return HealthHealthy, nil
	})
	if err := db.healthMonitor.Start(context.Background()); err != nil {
		_ = storageEngine.Close()
		bridge.closeCachedIndexes()
		return nil, fmt.Errorf("failed to start health monitor: %w", err)
	}

	// Load existing collections from storage, preferring cached indexes
	// that were deserialized or rebuilt during recovery.
	if err := db.loadExistingCollections(context.Background(), bridge); err != nil {
		db.healthMonitor.Stop()
		db.closeDefaultGraph()
		_ = storageEngine.Close()
		bridge.closeCachedIndexes()
		return nil, fmt.Errorf("failed to load existing collections: %w", err)
	}
	for _, col := range db.collections {
		if g := col.GetGraph(); g != nil {
			for _, definition := range durableEdgeKinds {
				g.SetEdgeKindDirection(definition.Kind, definition.Undirected)
			}
		}
	}

	// Wire graph WAL: if the engine supports graph edge persistence, wire it
	// to every collection's graph so Txn.Commit() writes durable edge records.
	if walWriter, ok := storageEngine.(storage.GraphWALWriter); ok {
		for _, col := range db.collections {
			db.wireGraphWAL(col, walWriter)
		}
		// Wire recovery: register every graph-enabled collection as a
		// recovery target so committed graph edge WAL frames route to the
		// correct per-collection graph.
		for _, col := range db.collections {
			if g := col.GetGraph(); g != nil {
				if target, ok := g.(storage.GraphRecoveryTarget); ok {
					storageEngine.SetGraphRecoveryTarget(col.name, target)
				}
			}
		}

	}

	// Initialize temporal ANN cache if configured.
	if db.config.TemporalANN.MaxBytes > 0 || db.config.TemporalANN.MaxEntries > 0 {
		db.temporalCache = newTemporalIndexCache(db, db.config.TemporalANN.MaxBytes, db.config.TemporalANN.MaxEntries)
	}

	return db, nil
}

// defaultGraphForCollection returns a collection-bound view of the one graph
// namespace used by SQL-created graph tables. The graph object is runtime
// state and is reconstructed from persisted collection namespace metadata on
// reopen; transactions retain the collection name for WAL routing.
func (db *Database) defaultGraphForCollection(collection string) (Graph, error) {
	db.defaultGraphMu.Lock()
	defer db.defaultGraphMu.Unlock()
	if db.defaultGraph == nil {
		g, err := NewGraph(GraphConfig{})
		if err != nil {
			return nil, fmt.Errorf("create default graph namespace: %w", err)
		}
		db.defaultGraph = g
	}
	return &collectionGraph{Graph: db.defaultGraph, collection: collection}, nil
}

func (db *Database) graphOverrideForCollection(name string, config *storage.CollectionConfig) (Graph, error) {
	if config == nil || !config.GraphEnabled || config.GraphNamespace != defaultGraphNamespace {
		return nil, nil
	}
	return db.defaultGraphForCollection(name)
}

func (db *Database) closeDefaultGraph() {
	db.defaultGraphMu.Lock()
	g := db.defaultGraph
	db.defaultGraph = nil
	db.defaultGraphMu.Unlock()
	if g != nil {
		_ = g.Close()
	}
}

// createSQLEdgeKind allocates, registers, and durably records a named graph
// edge kind for CREATE EDGE TYPE. The numeric kind is an internal wire/storage
// detail; SQL callers use the stable name.
func (db *Database) createSQLEdgeKind(name string, undirected bool, directionSpecified bool) error {
	if name == "" {
		return fmt.Errorf("edge type name must not be empty")
	}
	db.mu.Lock()
	defer db.mu.Unlock()
	if db.closed {
		return ErrDatabaseClosed
	}
	store, ok := db.storage.(storage.EdgeKindStore)
	if !ok {
		return fmt.Errorf("storage engine does not support durable SQL edge types")
	}
	var kinds map[string]uint8
	definitions := make(map[string]storage.EdgeKindDefinition)
	if definitionStore, ok := db.storage.(storage.EdgeKindDefinitionStore); ok {
		loaded, loadErr := definitionStore.ListEdgeKindDefinitions()
		if loadErr != nil {
			return loadErr
		}
		definitions = loaded
		kinds = make(map[string]uint8, len(loaded))
		for edgeName, definition := range loaded {
			kinds[edgeName] = definition.Kind
		}
	} else {
		loaded, loadErr := store.ListEdgeKinds()
		if loadErr != nil {
			return loadErr
		}
		kinds = loaded
		for edgeName, kind := range loaded {
			definitions[edgeName] = storage.EdgeKindDefinition{Kind: kind}
		}
	}
	if existing, exists := kinds[name]; exists {
		if ResolveEdgeKind(name) != 0 && ResolveEdgeKind(name) != existing {
			return fmt.Errorf("edge type %q is already registered with kind %d", name, ResolveEdgeKind(name))
		}
		existingDefinition := definitions[name]
		if directionSpecified && existingDefinition.Undirected != undirected {
			return fmt.Errorf("edge type %q already has a conflicting direction", name)
		}
		if !RegisterEdgeKindWithDirection(name, existing, existingDefinition.Undirected) {
			return fmt.Errorf("runtime graph registry rejected edge type %q=%d", name, existing)
		}
		for _, col := range db.collections {
			if g := col.GetGraph(); g != nil {
				g.SetEdgeKindDirection(existing, existingDefinition.Undirected)
			}
		}
		return nil
	}

	kind := ResolveEdgeKind(name)
	if kind == 0 {
		for candidate := uint8(1); candidate != 0; candidate++ {
			used := false
			for _, existing := range kinds {
				if existing == candidate {
					used = true
					break
				}
			}
			if !used {
				kind = candidate
				break
			}
		}
	}
	if kind == 0 {
		return fmt.Errorf("no graph edge kinds remain")
	}
	if definitionStore, ok := db.storage.(storage.EdgeKindDefinitionStore); ok {
		if err := definitionStore.CreateEdgeKindDefinition(name, kind, undirected); err != nil {
			return err
		}
	} else {
		if undirected {
			return fmt.Errorf("storage engine does not support durable undirected edge types")
		}
		if err := store.CreateEdgeKind(name, kind); err != nil {
			return err
		}
	}
	if !RegisterEdgeKindWithDirection(name, kind, undirected) {
		return fmt.Errorf("runtime graph registry rejected edge type %q=%d", name, kind)
	}
	for _, col := range db.collections {
		if g := col.GetGraph(); g != nil {
			g.SetEdgeKindDirection(kind, undirected)
		}
	}
	return nil
}

// wireGraphWAL sets the engine WAL writer on a collection's graph so edge
// mutations are durably recorded.
func (db *Database) wireGraphWAL(col *Collection, walWriter storage.GraphWALWriter) {
	g := col.GetGraph()
	if g == nil {
		return
	}
	type walWirer interface {
		SetWALWriter(w storage.GraphWALWriter)
	}
	if w, ok := g.(walWirer); ok {
		w.SetWALWriter(walWriter)
	}
	// Ensure the graph knows its owning collection name so WAL frames
	// carry the correct collection identity for per-collection recovery.
	type collectionNamer interface {
		SetCollectionName(name string)
	}
	if w, ok := g.(collectionNamer); ok {
		w.SetCollectionName(col.name)
	}
}

// SetLogger configures the database logger. It is safe to call concurrently.
func (db *Database) SetLogger(logger Logger) {
	db.mu.Lock()
	db.logger = logger
	db.mu.Unlock()
}

// Catalog returns the current read-only catalog snapshot.
//
// DDL performs copy-on-write catalog swaps, so the returned pointer must not
// be retained across schema changes — it is valid only for the duration of a
// single synchronous operation. This mirrors how the query path reads
// db.catalog under RLock and drops it before returning.
func (db *Database) Catalog() *catalog.Catalog {
	db.mu.RLock()
	defer db.mu.RUnlock()
	return db.catalog
}

// CreateCollection creates a new collection with the specified options
func (db *Database) CreateCollection(ctx context.Context, name string, opts ...CollectionOption) (*Collection, error) {
	db.mu.Lock()
	defer db.mu.Unlock()

	if db.closed {
		return nil, ErrDatabaseClosed
	}

	if _, exists := db.collections[name]; exists {
		return nil, fmt.Errorf("collection %s already exists: %w", name, ErrCollectionExists)
	}

	if len(db.collections) >= db.config.MaxCollections {
		return nil, ErrTooManyCollections
	}

	collection, err := newCollection(ctx, name, db.storage, db.metrics, db.newWriteController(), opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create collection: %w", err)
	}
	collection.db = db
	if g := collection.GetGraph(); g != nil {
		collection.SetGraph(g)
	}
	if err := db.configureAsyncIndex(collection); err != nil {
		return nil, errors.Join(
			fmt.Errorf("failed to configure asynchronous index: %w", err),
			collection.Close(),
			db.storage.DeleteCollection(name),
		)
	}

	db.collections[name] = collection
	db.registerCollectionInCatalog(name, collection.config)

	// Wire graph WAL for newly created collection.
	if walWriter, ok := db.storage.(storage.GraphWALWriter); ok {
		db.wireGraphWAL(collection, walWriter)
	}

	return collection, nil
}

// registerCollectionInCatalog adds a collection's schema to the SQL catalog
// so that binder resolution works without manual mock catalog injection.
// Caller must hold db.mu.
func (db *Database) registerCollectionInCatalog(name string, config *CollectionConfig) {
	builder := catalog.NewBuilderFrom(db.catalog)
	// "id" is always the primary key and non-null.
	columns := []catalog.ColumnInfo{{
		Name:  "id",
		Type:  catalog.TypeString,
		Flags: catalog.ColFlagPrimaryKey | catalog.ColFlagNotNull,
	}}
	// Physical collection configs intentionally omit relational schema on
	// reopen. Preserve the existing catalog columns when a DDL operation (for
	// example CREATE/DROP JSON INDEX) republishes that collection; otherwise a
	// harmless index change would erase all metadata columns.
	if (config == nil || config.MetadataSchema == nil) && db.catalog != nil {
		if table, err := db.catalog.GetTable(catalog.HashIdentifier(name)); err == nil {
			columns = columns[:0]
			for _, existing := range db.catalog.AllColumns(table) {
				columns = append(columns, catalog.ColumnInfo{Name: existing.Name, Type: existing.Type, Flags: existing.Flags})
			}
		}
	}
	if config != nil && config.MetadataSchema != nil {
		for fieldName, fieldType := range config.MetadataSchema {
			var flags uint16
			if config.ColumnConstraints != nil {
				flags = config.ColumnConstraints[fieldName]
			}
			// The physical record key is represented by the catalog's id
			// column. If SQL declared an explicit type for id (for example
			// BIGINT), update that definition instead of appending a duplicate
			// name whose earlier textual definition would win binder lookup.
			if strings.EqualFold(fieldName, "id") {
				columns[0].Type = metadataFieldToCatalogType(fieldType)
				columns[0].Flags |= flags
				continue
			}
			columns = append(columns, catalog.ColumnInfo{
				Name:  fieldName,
				Type:  metadataFieldToCatalogType(fieldType),
				Flags: flags,
			})
		}
	}
	builder.AddTable(name, columns)
	// Vector collections register a vector index named after the collection's
	// vector column ("vector" by convention) so SIMILARITY()/VECTOR_DISTANCE()
	// and pgvector operators bind against it in SQL.
	if config != nil && config.Dimension > 0 {
		builder.AddVectorIndex("vector", uint32(config.Dimension), uint8(config.Metric))
	}
	// Register foreign key constraints in the catalog.
	if config != nil {
		for _, fk := range config.ForeignKeys {
			builder.AddForeignKey(fk)
		}
		// Register CHECK constraints in the catalog.
		for _, chk := range config.CheckConstraints {
			builder.AddCheckConstraint(name, chk.Expression, chk.ColumnName)
		}
		// Register column DEFAULTs in the catalog.
		for colName, defaultVal := range config.ColumnDefaults {
			builder.AddDefaultValue(name, colName, defaultVal)
		}
		jsonIndexes := make([]catalog.JSONIndexInfo, 0, len(config.JSONIndexes))
		for _, jsonIndex := range config.JSONIndexes {
			jsonIndexes = append(jsonIndexes, catalog.JSONIndexInfo{
				Name: jsonIndex.Name, Table: name, Column: jsonIndex.Column,
				Path: jsonIndex.Path, TextResult: jsonIndex.TextResult,
			})
		}
		builder.ReplaceJSONIndexesForTable(name, jsonIndexes)
	}
	data := builder.Build()
	cat, err := catalog.Load(data, db.quantRegistry)
	if err != nil {
		// Catalog registration is best-effort; if it fails, queries will
		// fail at bind time with a clear error message.
		return
	}
	db.catalog = cat
	db.catalogGeneration.Add(1)

	// Push to storage engine for persistence across restarts
	if e, ok := db.storage.(interface{ SetCatalogData([]byte) }); ok {
		e.SetCatalogData(data)
	}
}

// registerVectorColumnInCatalog adds the logical name of a SQL VECTOR column
// to the durable table definition. Vectors live in the collection's physical
// vector slot rather than in metadata, but SQL still needs the declared name
// (for example name_embedding) to bind INSERT and vector expressions.
//
// CREATE GRAPH/TABLE is the only path that can introduce a non-canonical
// vector name today, so this is kept as a narrow catalog publication step
// instead of changing the native collection configuration contract.
func (db *Database) registerVectorColumnInCatalog(name, vectorName string, dimension int, metric DistanceMetric) {
	if strings.TrimSpace(name) == "" || strings.TrimSpace(vectorName) == "" || dimension <= 0 {
		return
	}

	db.mu.Lock()
	defer db.mu.Unlock()
	if db.catalog == nil {
		return
	}

	table, err := db.catalog.GetTable(catalog.HashIdentifier(name))
	if err != nil {
		return
	}
	columns := make([]catalog.ColumnInfo, 0, table.ColumnsCount+1)
	found := false
	for _, column := range db.catalog.AllColumns(table) {
		columns = append(columns, catalog.ColumnInfo{
			Name:  column.Name,
			Type:  column.Type,
			Flags: column.Flags,
		})
		if strings.EqualFold(column.Name, vectorName) {
			found = true
		}
	}
	if !found {
		columns = append(columns, catalog.ColumnInfo{Name: vectorName, Type: catalog.TypeVector})
	}

	builder := catalog.NewBuilderFrom(db.catalog)
	builder.AddTable(name, columns)
	if !found {
		builder.AddVectorIndex(vectorName, uint32(dimension), uint8(metric))
	}
	data := builder.Build()
	cat, err := catalog.Load(data, db.quantRegistry)
	if err != nil {
		return
	}
	db.catalog = cat
	db.catalogGeneration.Add(1)
	if e, ok := db.storage.(interface{ SetCatalogData([]byte) }); ok {
		e.SetCatalogData(data)
	}
}

func (db *Database) vectorColumnName(name string) string {
	if db == nil || strings.TrimSpace(name) == "" {
		return ""
	}
	db.mu.RLock()
	cat := db.catalog
	db.mu.RUnlock()
	if cat == nil {
		return ""
	}
	table, err := cat.GetTable(catalog.HashIdentifier(name))
	if err != nil {
		return ""
	}
	for _, column := range cat.AllColumns(table) {
		if column.Type == catalog.TypeVector {
			return column.Name
		}
	}
	return ""
}

// metadataFieldToCatalogType maps a metadata FieldType to a catalog column type.
func metadataFieldToCatalogType(ft FieldType) uint16 {
	switch ft {
	case IntField:
		return catalog.TypeInt
	case BigIntField:
		return catalog.TypeBigInt
	case FloatField:
		return catalog.TypeFloat
	case StringField:
		return catalog.TypeString
	case BoolField:
		return catalog.TypeBool
	case TimeField:
		return catalog.TypeTimestamp
	case StringArrayField, IntArrayField, FloatArrayField:
		return catalog.TypeString // arrays stored as string representations
	case JSONField:
		return catalog.TypeJSON
	case JSONBField:
		return catalog.TypeJSONB
	default:
		return catalog.TypeString
	}
}

// EnsureCollection gets an existing collection, or creates it with the given options.
// If the collection exists but its dimension differs from the requested dimension,
// it returns a CollectionDimensionMismatchError without modifying the existing
// collection.
func (db *Database) EnsureCollection(ctx context.Context, name string, dimension int, opts ...CollectionOption) (*Collection, error) {
	return db.ensureCollection(ctx, name, dimension, false, opts...)
}

// EnsureCollectionRecreateOnDimensionMismatch gets an existing collection, or
// creates it with the given options. If the collection exists but its dimension
// differs from the requested dimension, it is dropped and recreated.
//
// Prefer EnsureCollection unless destructive recovery from a known-bad schema is
// explicitly intended.
func (db *Database) EnsureCollectionRecreateOnDimensionMismatch(ctx context.Context, name string, dimension int, opts ...CollectionOption) (*Collection, error) {
	return db.ensureCollection(ctx, name, dimension, true, opts...)
}

func (db *Database) ensureCollection(ctx context.Context, name string, dimension int, recreateOnDimensionMismatch bool, opts ...CollectionOption) (*Collection, error) {
	if dimension < 0 {
		return nil, ErrInvalidDimension
	}
	// dimension == 0 is valid: metadata-only collection (no vectors required)

	db.mu.Lock()
	defer db.mu.Unlock()

	if db.closed {
		return nil, ErrDatabaseClosed
	}

	// Fast path: collection exists with correct dimension. EnsureCollection is
	// also a configuration contract: options must either describe the existing
	// collection or be rejected, never silently discarded.
	if col, exists := db.collections[name]; exists {
		if col.Dimension() == dimension {
			if err := ensureExistingCollectionConfig(name, col, dimension, opts...); err != nil {
				return nil, err
			}
			return col, nil
		}
		// Metadata-only ↔ vector mode transitions are destructive — reject
		if col.Dimension() == 0 || dimension == 0 {
			return nil, fmt.Errorf("cannot change collection %q between metadata-only and vector modes", name)
		}
		if !recreateOnDimensionMismatch {
			return nil, newCollectionDimensionMismatchError(name, col.Dimension(), dimension)
		}
		if err := db.deleteCollectionLocked(col, name); err != nil {
			return nil, fmt.Errorf("failed to drop mismatched collection %q: %w", name, err)
		}
		// Fall through to create.
	}

	col, err := db.createCollectionLocked(ctx, name, ensureCollectionOptions(dimension, opts)...)
	if err == nil {
		return col, nil
	}
	if errors.Is(err, ErrCollectionExists) {
		// Another caller raced and created it. Use it if dimension matches.
		if col, getErr := db.getCollectionLocked(name); getErr == nil {
			if col.Dimension() == dimension {
				if configErr := ensureExistingCollectionConfig(name, col, dimension, opts...); configErr != nil {
					return nil, configErr
				}
				return col, nil
			}
			return nil, newCollectionDimensionMismatchError(name, col.Dimension(), dimension)
		}
	}
	return nil, err
}

func ensureExistingCollectionConfig(name string, col *Collection, dimension int, opts ...CollectionOption) error {
	existing := col.Config()
	desired := existing
	for _, opt := range ensureCollectionOptions(dimension, opts) {
		if err := opt(&desired); err != nil {
			return fmt.Errorf("failed to apply EnsureCollection option for %q: %w", name, err)
		}
	}
	if reflect.DeepEqual(existing, desired) {
		return nil
	}
	return newCollectionConfigurationMismatchError(name)
}

func newCollectionDimensionMismatchError(name string, existing, requested int) error {
	return &CollectionDimensionMismatchError{
		Collection:         name,
		ExistingDimension:  existing,
		RequestedDimension: requested,
	}
}

func newCollectionConfigurationMismatchError(name string) error {
	return &CollectionConfigurationMismatchError{Collection: name}
}

func ensureCollectionOptions(dimension int, opts []CollectionOption) []CollectionOption {
	createOpts := make([]CollectionOption, 0, len(opts)+1)
	createOpts = append(createOpts, opts...)
	if dimension > 0 {
		createOpts = append(createOpts, WithDimension(dimension))
	}
	// dimension == 0: metadata-only — WithMetadataOnly() is already in opts
	return createOpts
}

// createCollectionLocked creates a collection. Caller must hold db.mu.
func (db *Database) createCollectionLocked(ctx context.Context, name string, opts ...CollectionOption) (*Collection, error) {
	collection, err := newCollection(ctx, name, db.storage, db.metrics, db.newWriteController(), opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create collection: %w", err)
	}
	collection.db = db
	if err := db.configureAsyncIndex(collection); err != nil {
		return nil, errors.Join(
			fmt.Errorf("failed to configure asynchronous index: %w", err),
			collection.Close(),
			db.storage.DeleteCollection(name),
		)
	}
	db.collections[name] = collection
	db.registerCollectionInCatalog(name, collection.config)
	return collection, nil
}

// deleteCollectionLocked deletes a collection from memory and storage.
// Caller must hold db.mu. Returns error without altering the map on failure.
func (db *Database) deleteCollectionLocked(col *Collection, name string) error {
	delete(db.collections, name)
	if err := col.Close(); err != nil {
		db.collections[name] = col
		return fmt.Errorf("failed to close collection %s: %w", name, err)
	}
	isSharded := col.config.Sharded
	if isSharded {
		for _, shardName := range shardStorageNames(name) {
			if err := db.storage.DeleteCollection(shardName); err != nil {
				return fmt.Errorf("failed to delete shard %s: %w", shardName, err)
			}
		}
		return nil
	}
	if err := db.storage.DeleteCollection(name); err != nil {
		return err
	}
	return nil
}

// getCollectionLocked returns a collection from db.collections without storage lookup.
// Caller must hold db.mu.
func (db *Database) getCollectionLocked(name string) (*Collection, error) {
	if col, exists := db.collections[name]; exists {
		return col, nil
	}
	return nil, ErrCollectionNotFound
}

// GetCollection retrieves an existing collection by name
func (db *Database) GetCollection(name string) (*Collection, error) {
	db.mu.Lock()
	defer db.mu.Unlock()

	if db.closed {
		return nil, ErrDatabaseClosed
	}

	// Check if collection is already loaded
	collection, exists := db.collections[name]
	if exists {
		return collection, nil
	}

	// Try to load collection from storage with configuration.
	if fileEngine, ok := db.storage.(interface {
		GetCollectionWithConfig(name string) (storage.Collection, *storage.CollectionConfig, error)
	}); ok {
		var err error
		collection, err = db.loadCollectionFromStorage(context.Background(), name, fileEngine, db.bridge)
		if err != nil {
			return nil, err
		}
	} else {
		return nil, fmt.Errorf("collection %s not found", name)
	}

	// Cache the collection
	db.collections[name] = collection

	return collection, nil
}

// firstGraphCollection returns the first collection that has an attached graph,
// or nil if none exists. Used for SQL GRAPH_EDGES operations that need a graph
// but don't specify a collection name explicitly.
func (db *Database) firstGraphCollection() *Collection {
	db.mu.RLock()
	defer db.mu.RUnlock()
	for _, col := range db.collections {
		if col.GetGraph() != nil {
			return col
		}
	}
	return nil
}

// graphCollectionNames returns graph-backed collections in stable order. A
// logical GRAPH_NODES reference carries a record ID rather than a collection
// name, so FK validation may need to resolve that ID across the graph-backed
// collections already owned by this database. The collection owning the FK is
// placed first when it is graph-backed, which preserves the natural local
// interpretation and makes duplicate record IDs deterministic.
func (db *Database) graphCollectionNames(preferred string) []string {
	db.mu.RLock()
	names := make([]string, 0, len(db.collections))
	for name, col := range db.collections {
		if col.GetGraph() != nil {
			names = append(names, name)
		}
	}
	db.mu.RUnlock()
	sort.Strings(names)
	if preferred != "" {
		for i, name := range names {
			if name == preferred {
				if i > 0 {
					copy(names[1:i+1], names[0:i])
					names[0] = preferred
				}
				break
			}
		}
	}
	return names
}

// ListCollections returns the names of all collections as a best-effort
// compatibility helper. Use ListCollectionsWithContext when you need explicit
// error reporting from storage-backed discovery.
func (db *Database) ListCollections() []string {
	names, _ := db.ListCollectionsWithContext(context.Background())
	return names
}

// ListCollectionsWithContext returns the names of all persisted collections.
func (db *Database) ListCollectionsWithContext(ctx context.Context) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	db.mu.RLock()
	if db.closed {
		db.mu.RUnlock()
		return nil, ErrDatabaseClosed
	}
	namesMap := make(map[string]struct{}, len(db.collections))
	for name := range db.collections {
		namesMap[name] = struct{}{}
	}
	db.mu.RUnlock()

	names, err := db.storage.ListCollections()
	if err != nil {
		return nil, err
	}
	for _, name := range names {
		// Filter out shard collection names - only return parent collection names
		if _, _, ok := parseShardName(name); !ok {
			namesMap[name] = struct{}{}
		}
	}

	result := make([]string, 0, len(namesMap))
	for name := range namesMap {
		result = append(result, name)
	}
	sort.Strings(result)
	return result, nil
}

// Iterate walks every persisted record in every collection. Records are
// delivered one at a time and callback errors stop iteration immediately.
func (db *Database) Iterate(ctx context.Context, fn func(collection string, record Record) error) error {
	if fn == nil {
		return fmt.Errorf("iterate callback cannot be nil")
	}

	names, err := db.ListCollectionsWithContext(ctx)
	if err != nil {
		return err
	}

	for _, name := range names {
		if err := ctx.Err(); err != nil {
			return err
		}

		collection, err := db.GetCollection(name)
		if err != nil {
			return fmt.Errorf("get collection %q during iteration: %w", name, err)
		}
		if err := collection.Iterate(ctx, func(record Record) error {
			return fn(name, record)
		}); err != nil {
			return fmt.Errorf("iterate collection %q: %w", name, err)
		}
	}

	return nil
}

// DeleteCollection removes a collection and its persisted data.
func (db *Database) DeleteCollection(ctx context.Context, name string) error {
	db.mu.Lock()
	if db.closed {
		db.mu.Unlock()
		return ErrDatabaseClosed
	}

	collection := db.collections[name]
	isSharded := collection != nil && collection.config.Sharded
	delete(db.collections, name)
	db.mu.Unlock()

	if collection != nil {
		if err := collection.Close(); err != nil {
			return fmt.Errorf("failed to close collection %s: %w", name, err)
		}
	}

	// For sharded collections, the parent collection doesn't exist in storage.
	// Only the hidden shard children exist. Delete all shard children.
	if isSharded {
		for _, shardName := range shardStorageNames(name) {
			if err := db.storage.DeleteCollection(shardName); err != nil {
				return fmt.Errorf("failed to delete shard %s: %w", shardName, err)
			}
		}
		return nil
	}

	// Non-sharded collection: delete the normal parent storage collection.
	if err := db.storage.DeleteCollection(name); err != nil {
		return err
	}

	return nil
}

// DeleteCollections removes multiple collections by exact name.
func (db *Database) DeleteCollections(ctx context.Context, names []string) error {
	var errs []error
	for _, name := range names {
		if err := db.DeleteCollection(ctx, name); err != nil {
			errs = append(errs, err)
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("delete collections failed: %v", errs)
	}
	return nil
}

// Health returns the current health status
func (db *Database) Health(ctx context.Context) (*obs.HealthStatus, error) {
	return db.health.Check(ctx)
}

// HealthMonitor returns the background system health monitor. Callers can
// register additional health checks and status-change callbacks.
func (db *Database) HealthMonitor() SystemHealthMonitor {
	return db.healthMonitor
}

// Ping checks if the database is responsive and the storage engine is accessible.
func (db *Database) Ping(ctx context.Context) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.closed {
		return ErrDatabaseClosed
	}

	// A basic check to see if storage responds
	_, err := db.storage.ListCollections()
	return err
}

// Stats returns database statistics
func (db *Database) Stats(ctx context.Context) *DatabaseStats {
	db.mu.RLock()
	defer db.mu.RUnlock()

	stats := &DatabaseStats{
		CollectionCount: len(db.collections),
		Collections:     make(map[string]*CollectionStats),
	}

	var totalMemory int64
	for name, collection := range db.collections {
		collectionStats := collection.Stats(ctx)
		stats.Collections[name] = collectionStats
		totalMemory += collectionStats.MemoryUsage
	}

	stats.MemoryUsage = totalMemory
	return stats
}

// OptimizeCollection performs optimization on a specific collection
func (db *Database) OptimizeCollection(ctx context.Context, name string, options *OptimizationOptions) error {
	collection, err := db.GetCollection(name)
	if err != nil {
		return fmt.Errorf("collection not found: %w", err)
	}

	return collection.OptimizeCollection(ctx, options)
}

// OptimizeAllCollections performs optimization on all collections
func (db *Database) OptimizeAllCollections(ctx context.Context, options *OptimizationOptions) error {
	db.mu.RLock()
	collections := make([]*Collection, 0, len(db.collections))
	for _, collection := range db.collections {
		collections = append(collections, collection)
	}
	db.mu.RUnlock()

	var errors []error
	for _, collection := range collections {
		if err := collection.OptimizeCollection(ctx, options); err != nil {
			errors = append(errors, fmt.Errorf("failed to optimize collection %s: %w", collection.name, err))
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("optimization errors: %v", errors)
	}

	return nil
}

// SetGlobalMemoryLimit sets a memory limit that applies to all collections
func (db *Database) SetGlobalMemoryLimit(bytes int64) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.closed {
		return ErrDatabaseClosed
	}

	// Distribute memory limit across collections
	collectionCount := len(db.collections)
	if collectionCount == 0 {
		return nil
	}

	perCollectionLimit := bytes / int64(collectionCount)

	var errors []error
	for _, collection := range db.collections {
		if err := collection.SetMemoryLimit(perCollectionLimit); err != nil {
			errors = append(errors, fmt.Errorf("failed to set memory limit for collection %s: %w", collection.name, err))
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("memory limit errors: %v", errors)
	}

	return nil
}

// GetGlobalMemoryUsage returns total memory usage across all collections
func (db *Database) GetGlobalMemoryUsage(ctx context.Context) (*GlobalMemoryUsage, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.closed {
		return nil, ErrDatabaseClosed
	}

	usage := &GlobalMemoryUsage{
		Collections: make(map[string]*CollectionMemoryStats),
		Timestamp:   time.Now(),
	}

	for name, collection := range db.collections {
		memUsage, err := collection.GetMemoryUsage(ctx)
		if err != nil {
			continue // Skip collections with errors
		}

		collectionMemStats := &CollectionMemoryStats{
			Total:         memUsage.Total,
			Index:         memUsage.Indices,
			Cache:         memUsage.Caches,
			Quantized:     memUsage.Quantized,
			MemoryMapped:  memUsage.MemoryMapped,
			Limit:         memUsage.Limit,
			Available:     memUsage.Available,
			PressureLevel: calculatePressureLevel(memUsage.Total, memUsage.Limit),
			Timestamp:     memUsage.Timestamp,
		}

		usage.Collections[name] = collectionMemStats
		usage.TotalMemory += memUsage.Total
		usage.TotalIndex += memUsage.Indices
		usage.TotalCache += memUsage.Caches
		usage.TotalQuantized += memUsage.Quantized
		usage.TotalMemoryMapped += memUsage.MemoryMapped
	}

	return usage, nil
}

// TriggerGlobalGC forces garbage collection across all collections
func (db *Database) TriggerGlobalGC() error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.closed {
		return ErrDatabaseClosed
	}

	var errors []error
	for name, collection := range db.collections {
		if err := collection.TriggerGC(); err != nil {
			errors = append(errors, fmt.Errorf("failed to trigger GC for collection %s: %w", name, err))
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("GC errors: %v", errors)
	}

	return nil
}

// loadExistingCollections discovers and loads existing collections from storage
func (db *Database) loadExistingCollections(ctx context.Context, bridge *indexPersistenceBridge) error {
	fileEngine, ok := db.storage.(interface {
		ListCollections() ([]string, error)
		GetCollectionWithConfig(name string) (storage.Collection, *storage.CollectionConfig, error)
	})
	if !ok {
		return nil
	}

	names, err := fileEngine.ListCollections()
	if err != nil {
		return fmt.Errorf("failed to list collections: %w", err)
	}

	// Track which parent collections we've loaded (to avoid loading shards as separate collections)
	loadedParents := make(map[string]bool)
	loadedCollections := make(map[string]*Collection)

	for _, name := range names {
		// Skip shard collection names - they are loaded as part of the parent
		if _, _, ok := parseShardName(name); ok {
			continue
		}

		// Skip if already loaded as a parent (from a previous shard entry)
		if loadedParents[name] {
			continue
		}

		collection, err := db.loadCollectionFromStorage(ctx, name, fileEngine, bridge)
		if err != nil {
			for _, c := range loadedCollections {
				c.Close()
			}
			if bridge != nil {
				bridge.closeCachedIndexes()
			}
			return err
		}

		loadedCollections[name] = collection
		loadedParents[name] = true
	}

	db.mu.Lock()
	for name, collection := range loadedCollections {
		// JSON index definitions live in the durable catalog rather than the
		// physical collection config. Hydrate them before the first query so the
		// derived inverted postings can rebuild lazily from visible records.
		if db.catalog != nil {
			if table, err := db.catalog.GetTable(catalog.HashIdentifier(name)); err == nil {
				columns := db.catalog.AllColumns(table)
				for _, idx := range db.catalog.JSONIndexesForTable(table.NameHash) {
					columnName := ""
					for _, column := range columns {
						if column.NameHash == idx.ColumnHash {
							columnName = column.Name
							break
						}
					}
					if columnName == "" {
						continue
					}
					collection.config.JSONIndexes = append(collection.config.JSONIndexes, JSONIndexDefinition{
						Name:   db.catalog.JSONIndexName(idx),
						Column: columnName,
						Path:   db.catalog.JSONIndexPath(idx), TextResult: idx.TextResult != 0,
					})
				}
			}
		}
		db.collections[name] = collection
		// The persisted catalog is authoritative for SQL schema metadata. A
		// storage collection config contains physical index settings but does
		// not carry relational columns, composite PK order, or FK definitions;
		// rebuilding the catalog here would silently erase those definitions
		// on every reopen. Register only legacy collections that have no table
		// entry yet.
		if db.catalog == nil {
			db.registerCollectionInCatalog(name, collection.config)
		} else if _, err := db.catalog.GetTable(catalog.HashIdentifier(name)); err != nil {
			db.registerCollectionInCatalog(name, collection.config)
		}
	}
	db.mu.Unlock()
	if bridge != nil {
		bridge.closeCachedIndexes()
	}

	return nil
}

func (db *Database) loadCollectionFromStorage(ctx context.Context, name string, engine interface {
	GetCollectionWithConfig(name string) (storage.Collection, *storage.CollectionConfig, error)
}, bridge *indexPersistenceBridge) (*Collection, error) {
	// Check if this is a shard collection name - if so, skip it
	if _, _, ok := parseShardName(name); ok {
		return nil, fmt.Errorf("cannot load shard collection %s directly, load parent collection instead", name)
	}

	// Check if this collection has shards (sharded collection)
	shardNames := shardStorageNames(name)
	firstShardStorage, config, err := engine.GetCollectionWithConfig(shardNames[0])
	if err == nil {
		// Shard 0 exists - this is a sharded collection
		// Load all shards
		shardStorages := make([]storage.Collection, shardCount)
		shardStorages[0] = firstShardStorage

		for i := 1; i < shardCount; i++ {
			shardStorages[i], _, err = engine.GetCollectionWithConfig(shardNames[i])
			if err != nil {
				for j := 0; j < i; j++ {
					shardStorages[j].Close()
				}
				return nil, fmt.Errorf("collection %s is missing shard %d: %w", name, i, err)
			}
		}

		graphOverride, overrideErr := db.graphOverrideForCollection(name, config)
		if overrideErr != nil {
			for j := 0; j < shardCount; j++ {
				shardStorages[j].Close()
			}
			return nil, overrideErr
		}
		collection, err := newShardedCollectionFromStorage(ctx, name, shardStorages, config, db.metrics, db.newWriteController(), graphOverride)
		if err != nil {
			for j := 0; j < shardCount; j++ {
				shardStorages[j].Close()
			}
			return nil, fmt.Errorf("failed to create sharded collection from storage: %w", err)
		}
		collection.db = db
		if g := collection.GetGraph(); g != nil {
			collection.SetGraph(g)
		}
		if err := db.configureAsyncIndex(collection); err != nil {
			_ = collection.Close()
			return nil, fmt.Errorf("failed to configure asynchronous index: %w", err)
		}
		return collection, nil
	}

	// Not a sharded collection - load as single collection
	storageCollection, config, err := engine.GetCollectionWithConfig(name)
	if err != nil {
		return nil, fmt.Errorf("collection %s not found", name)
	}

	// Prefer a cached index that was deserialized or rebuilt during recovery,
	// avoiding an expensive full rebuild from storage records.
	var cachedIndex index.Index
	if bridge != nil {
		cachedIndex = bridge.takeCachedIndex(name)
	}

	graphOverride, overrideErr := db.graphOverrideForCollection(name, config)
	if overrideErr != nil {
		storageCollection.Close()
		return nil, overrideErr
	}
	collection, err := newCollectionFromStorage(ctx, name, storageCollection, db.metrics, config, db.newWriteController(), cachedIndex, graphOverride)
	if err != nil {
		return nil, fmt.Errorf("failed to create collection from storage: %w", err)
	}
	collection.db = db
	if g := collection.GetGraph(); g != nil {
		collection.SetGraph(g)
	}
	if err := db.configureAsyncIndex(collection); err != nil {
		_ = collection.Close()
		return nil, fmt.Errorf("failed to configure asynchronous index: %w", err)
	}

	return collection, nil
}

func (db *Database) configureAsyncIndex(collection *Collection) error {
	if db.config.AsyncIndexQueueDepth == 0 || collection == nil || collection.shards != nil || collection.config.IndexType != HNSW {
		return nil
	}
	queue, err := newAsyncIndexQueue(collection, db.config.AsyncIndexQueueDepth, db.config.AsyncIndexWorkers)
	if err != nil {
		return err
	}
	collection.asyncIndex = queue
	return nil
}

func (db *Database) newWriteController() *writeController {
	return newWriteController(db.config.MaxConcurrentWrites, db.config.MaxWriteQueueDepth)
}

func defaultMaxConcurrentWrites() int {
	procs := runtime.GOMAXPROCS(0)
	if procs < 1 {
		return 1
	}
	if procs > 2 {
		return 2
	}
	return procs
}

// Close gracefully shuts down the database
func (db *Database) Close() error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if db.closed {
		return nil
	}

	var errors []error

	// Stop the background health monitor before tearing down collections
	// so health checks don't access closing state.
	if db.healthMonitor != nil {
		if err := db.healthMonitor.Stop(); err != nil {
			errors = append(errors, fmt.Errorf("health monitor stop: %w", err))
		}
	}

	// Close all collections
	for _, collection := range db.collections {
		if err := collection.Close(); err != nil {
			errors = append(errors, err)
		}
	}
	// SQL graph tables share one namespace. Collection.Close intentionally
	// leaves that runtime alive; the database owns the single final close.
	db.closeDefaultGraph()

	// Close temporal ANN cache before storage.
	if db.temporalCache != nil {
		db.temporalCache.close()
	}
	// Close storage
	if err := db.storage.Close(); err != nil {
		errors = append(errors, err)
	}

	db.closed = true

	if len(errors) > 0 {
		return fmt.Errorf("errors during shutdown: %v", errors)
	}

	return nil
}

// Vacuum reclaims disk space by rewriting the underlying storage file, dropping
// deleted records and obsolete WAL frames. This is a non-blocking operation
// that only briefly pauses the database during the final swap.
func (db *Database) Vacuum(ctx context.Context) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.closed {
		return ErrDatabaseClosed
	}

	if v, ok := db.storage.(interface{ Vacuum(context.Context) error }); ok {
		return v.Vacuum(ctx)
	}

	return fmt.Errorf("underlying storage engine does not support Vacuum")
}

// Backup creates a point-in-time copy of the database to the specified destination
// path. It uses a non-blocking fast-forward design to ensure the copy is consistent
// without interrupting active database operations.
func (db *Database) Backup(ctx context.Context, destPath string) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.closed {
		return ErrDatabaseClosed
	}

	if v, ok := db.storage.(interface {
		Backup(context.Context, string) error
	}); ok {
		return v.Backup(ctx, destPath)
	}

	return fmt.Errorf("underlying storage engine does not support Backup")
}

// Drop completely closes the database and destroys its underlying files from disk.
// Once a database is dropped, it cannot be recovered without a backup.
func (db *Database) Drop(ctx context.Context) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if db.closed {
		return ErrDatabaseClosed
	}

	// Close temporal ANN cache before collections to release HNSW index
	// pages, which hold ShardedFreeList slots tracked by the memory package's
	// shared PID controller.
	if db.temporalCache != nil {
		db.temporalCache.close()
	}

	// Close all collections safely
	for _, collection := range db.collections {
		collection.Close()
	}
	db.collections = make(map[string]*Collection)
	db.closeDefaultGraph()

	// Stop health monitor
	if db.healthMonitor != nil {
		db.healthMonitor.Stop()
	}

	db.closed = true

	if v, ok := db.storage.(interface{ Drop(context.Context) error }); ok {
		return v.Drop(ctx)
	}

	return fmt.Errorf("underlying storage engine does not support Drop")
}

// ResolveNodeID looks up a system-scoped GraphNodeID and returns its corresponding
// collection name and record ID. If the ID is unknown or deleted, it returns an error.
// This is an O(1) operation designed for rapid reverse-lookup during graph traversal.
func (db *Database) ResolveNodeID(ctx context.Context, id uint64) (string, string, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.closed {
		return "", "", ErrDatabaseClosed
	}
	colName, recID, err := db.storage.ResolveNodeID(ctx, id)
	if err != nil {
		return "", "", err
	}
	// Translate internal shard names back to logical collection names.
	if parent, _, ok := parseShardName(colName); ok {
		return parent, recID, nil
	}
	return colName, recID, nil
}

// GetNodeID resolves a (collection, recordID) pair to a system-scoped GraphNodeID.
// This is the forward direction of the node ID mapping; ResolveNodeID is the reverse.
func (db *Database) GetNodeID(ctx context.Context, collection, id string) (uint64, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.closed {
		return 0, ErrDatabaseClosed
	}
	return db.storage.GetNodeID(ctx, collection, id)
}
