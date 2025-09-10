# Storage Layer Design

This document describes the design and implementation of LibraVDB's storage layer, including the LSM-tree architecture, Write-Ahead Log (WAL), and data persistence strategies.

## Storage Architecture Overview

LibraVDB uses a Log-Structured Merge (LSM) tree architecture optimized for write-heavy workloads typical in vector databases. The storage layer provides durability, consistency, and efficient data organization.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Storage Layer                           │
├─────────────────────────────────────────────────────────────────┤
│  LSM Engine  │  Write-Ahead Log  │  Segment Management        │
├─────────────────────────────────────────────────────────────────┤
│  MemTable    │  WAL Entries      │  SSTable Segments          │
│  Immutable   │  Recovery         │  Compaction                │
│  MemTable    │  Replication      │  Bloom Filters             │
├─────────────────────────────────────────────────────────────────┤
│                    File System Interface                       │
└─────────────────────────────────────────────────────────────────┘
```

## LSM-Tree Architecture

### Core Components

```go
type Engine struct {
    basePath    string
    collections map[string]*Collection
    wal         *wal.WAL
    compactor   *Compactor
    config      *EngineConfig
    mu          sync.RWMutex
}

type Collection struct {
    name        string
    config      *CollectionConfig
    memTable    *MemTable
    immutable   []*MemTable
    segments    []*Segment
    walPath     string
    wal         *wal.WAL
    cache       map[string]*index.VectorEntry
    mu          sync.RWMutex
}
```

### MemTable Design

The MemTable is an in-memory data structure that buffers writes before flushing to disk:

```go
type MemTable struct {
    data      map[string]*VectorEntry
    size      int64
    maxSize   int64
    timestamp time.Time
    mu        sync.RWMutex
}

func (mt *MemTable) Insert(entry *VectorEntry) error {
    mt.mu.Lock()
    defer mt.mu.Unlock()
    
    // Calculate entry size
    entrySize := calculateEntrySize(entry)
    
    // Check if MemTable is full
    if mt.size+entrySize > mt.maxSize {
        return ErrMemTableFull
    }
    
    // Insert entry
    mt.data[entry.ID] = entry
    mt.size += entrySize
    
    return nil
}
```

### Segment (SSTable) Design

Segments are immutable, sorted files that store vector data efficiently:

```go
type Segment struct {
    path        string
    level       int
    minKey      string
    maxKey      string
    entryCount  int
    size        int64
    bloomFilter *BloomFilter
    index       *SegmentIndex
    metadata    *SegmentMetadata
}
```

#### Segment File Format

```
┌─────────────────────────────────────────────────────────────┐
│                      File Header                            │
├─────────────────────────────────────────────────────────────┤
│  Magic(4)  │ Version(4) │ Compression(4) │ Checksum(4)    │
├─────────────────────────────────────────────────────────────┤
│                      Data Blocks                            │
├─────────────────────────────────────────────────────────────┤
│  Block 1   │  Block 2   │  ...  │  Block N  │  Padding    │
├─────────────────────────────────────────────────────────────┤
│                      Index Block                            │
├─────────────────────────────────────────────────────────────┤
│  Key Offsets  │  Block Offsets  │  Bloom Filter           │
├─────────────────────────────────────────────────────────────┤
│                      Footer                                 │
├─────────────────────────────────────────────────────────────┤
│  Index Offset(8)  │  Metadata Offset(8)  │  Checksum(4)  │
└─────────────────────────────────────────────────────────────┘
```

#### Data Block Format

```go
type DataBlock struct {
    entries     []*VectorEntry
    compression CompressionType
    checksum    uint32
}

func (db *DataBlock) Serialize() ([]byte, error) {
    var buf bytes.Buffer
    
    // Write entry count
    binary.Write(&buf, binary.LittleEndian, uint32(len(db.entries)))
    
    // Write entries
    for _, entry := range db.entries {
        if err := db.writeEntry(&buf, entry); err != nil {
            return nil, err
        }
    }
    
    // Apply compression if enabled
    data := buf.Bytes()
    if db.compression != NoCompression {
        compressed, err := compress(data, db.compression)
        if err != nil {
            return nil, err
        }
        data = compressed
    }
    
    // Calculate and append checksum
    checksum := crc32.ChecksumIEEE(data)
    checksumBytes := make([]byte, 4)
    binary.LittleEndian.PutUint32(checksumBytes, checksum)
    
    return append(data, checksumBytes...), nil
}
```

## Write-Ahead Log (WAL)

The WAL ensures durability by logging all operations before applying them:

### WAL Entry Format

```go
type WALEntry struct {
    Timestamp time.Time
    Operation OperationType
    ID        string
    Vector    []float32
    Metadata  map[string]interface{}
    Checksum  uint32
}

type OperationType uint8

const (
    OpInsert OperationType = iota
    OpUpdate
    OpDelete
    OpCompact
)
```

### WAL Implementation

```go
type WAL struct {
    file       *os.File
    path       string
    syncPolicy SyncPolicy
    buffer     *bytes.Buffer
    mu         sync.Mutex
}

func (w *WAL) WriteEntry(entry *WALEntry) error {
    w.mu.Lock()
    defer w.mu.Unlock()
    
    // Serialize entry
    data, err := w.serializeEntry(entry)
    if err != nil {
        return err
    }
    
    // Write to buffer
    w.buffer.Write(data)
    
    // Sync based on policy
    switch w.syncPolicy {
    case SyncEveryWrite:
        return w.flush()
    case SyncPeriodic:
        // Handled by background goroutine
        return nil
    case SyncOnClose:
        // Only sync on close
        return nil
    }
    
    return nil
}
```

### WAL Recovery

```go
func (w *WAL) Recover() ([]*WALEntry, error) {
    file, err := os.Open(w.path)
    if err != nil {
        if os.IsNotExist(err) {
            return nil, nil // No WAL file exists
        }
        return nil, err
    }
    defer file.Close()
    
    var entries []*WALEntry
    reader := bufio.NewReader(file)
    
    for {
        entry, err := w.readEntry(reader)
        if err == io.EOF {
            break
        }
        if err != nil {
            // Corruption detected, truncate at this point
            log.Printf("WAL corruption detected, truncating at entry %d", len(entries))
            break
        }
        
        entries = append(entries, entry)
    }
    
    return entries, nil
}
```

## Compaction Strategy

Compaction merges segments to maintain read performance and reclaim space:

### Leveled Compaction

```go
type LeveledCompactor struct {
    levels      []Level
    maxLevels   int
    sizeRatio   int
    maxSize     []int64
    running     bool
    mu          sync.Mutex
}

type Level struct {
    segments    []*Segment
    maxSize     int64
    currentSize int64
}

func (lc *LeveledCompactor) TriggerCompaction() error {
    lc.mu.Lock()
    defer lc.mu.Unlock()
    
    if lc.running {
        return ErrCompactionInProgress
    }
    
    // Find level that needs compaction
    for i, level := range lc.levels {
        if level.currentSize > level.maxSize {
            go lc.compactLevel(i)
            lc.running = true
            return nil
        }
    }
    
    return nil
}
```

### Compaction Process

```go
func (lc *LeveledCompactor) compactLevel(levelNum int) error {
    defer func() {
        lc.mu.Lock()
        lc.running = false
        lc.mu.Unlock()
    }()
    
    level := lc.levels[levelNum]
    nextLevel := lc.levels[levelNum+1]
    
    // Select segments to compact
    segments := lc.selectSegmentsForCompaction(level)
    
    // Find overlapping segments in next level
    overlapping := lc.findOverlappingSegments(nextLevel, segments)
    
    // Merge segments
    newSegments, err := lc.mergeSegments(append(segments, overlapping...))
    if err != nil {
        return err
    }
    
    // Atomically replace segments
    return lc.replaceSegments(levelNum, levelNum+1, segments, overlapping, newSegments)
}
```

## Bloom Filters

Bloom filters reduce unnecessary disk reads by quickly determining if a key might exist in a segment:

```go
type BloomFilter struct {
    bits      []uint64
    size      uint64
    hashFuncs int
}

func NewBloomFilter(expectedItems int, falsePositiveRate float64) *BloomFilter {
    size := calculateOptimalSize(expectedItems, falsePositiveRate)
    hashFuncs := calculateOptimalHashFuncs(size, expectedItems)
    
    return &BloomFilter{
        bits:      make([]uint64, (size+63)/64),
        size:      size,
        hashFuncs: hashFuncs,
    }
}

func (bf *BloomFilter) Add(key string) {
    hash1, hash2 := bf.hash(key)
    
    for i := 0; i < bf.hashFuncs; i++ {
        bit := (hash1 + uint64(i)*hash2) % bf.size
        wordIndex := bit / 64
        bitIndex := bit % 64
        bf.bits[wordIndex] |= 1 << bitIndex
    }
}

func (bf *BloomFilter) MightContain(key string) bool {
    hash1, hash2 := bf.hash(key)
    
    for i := 0; i < bf.hashFuncs; i++ {
        bit := (hash1 + uint64(i)*hash2) % bf.size
        wordIndex := bit / 64
        bitIndex := bit % 64
        
        if bf.bits[wordIndex]&(1<<bitIndex) == 0 {
            return false
        }
    }
    
    return true
}
```

## Caching Strategy

Multi-level caching improves read performance:

```go
type CacheManager struct {
    l1Cache    *LRUCache    // Hot data in memory
    l2Cache    *LRUCache    // Warm data, larger capacity
    blockCache *LRUCache    // Cached data blocks
    config     *CacheConfig
}

type CacheConfig struct {
    L1Size     int64
    L2Size     int64
    BlockSize  int64
    Policy     CachePolicy
}

func (cm *CacheManager) Get(key string) (*VectorEntry, bool) {
    // Try L1 cache first
    if entry, found := cm.l1Cache.Get(key); found {
        return entry.(*VectorEntry), true
    }
    
    // Try L2 cache
    if entry, found := cm.l2Cache.Get(key); found {
        // Promote to L1
        cm.l1Cache.Put(key, entry)
        return entry.(*VectorEntry), true
    }
    
    return nil, false
}
```

## Data Compression

Compression reduces storage space and I/O overhead:

```go
type CompressionType uint8

const (
    NoCompression CompressionType = iota
    LZ4Compression
    ZstdCompression
    SnappyCompression
)

type Compressor interface {
    Compress(data []byte) ([]byte, error)
    Decompress(data []byte) ([]byte, error)
    Type() CompressionType
}

func NewCompressor(cType CompressionType) Compressor {
    switch cType {
    case LZ4Compression:
        return &LZ4Compressor{}
    case ZstdCompression:
        return &ZstdCompressor{}
    case SnappyCompression:
        return &SnappyCompressor{}
    default:
        return &NoOpCompressor{}
    }
}
```

## Error Handling and Recovery

### Corruption Detection

```go
func (s *Segment) ValidateIntegrity() error {
    file, err := os.Open(s.path)
    if err != nil {
        return err
    }
    defer file.Close()
    
    // Read and validate header
    header, err := s.readHeader(file)
    if err != nil {
        return fmt.Errorf("invalid header: %w", err)
    }
    
    // Validate checksums
    if err := s.validateChecksums(file, header); err != nil {
        return fmt.Errorf("checksum validation failed: %w", err)
    }
    
    // Validate index consistency
    if err := s.validateIndex(file, header); err != nil {
        return fmt.Errorf("index validation failed: %w", err)
    }
    
    return nil
}
```

### Recovery Procedures

```go
func (e *Engine) RecoverFromCorruption() error {
    // 1. Identify corrupted segments
    corruptedSegments, err := e.identifyCorruptedSegments()
    if err != nil {
        return err
    }
    
    // 2. Recover from WAL
    walEntries, err := e.wal.Recover()
    if err != nil {
        return err
    }
    
    // 3. Rebuild corrupted segments
    for _, segment := range corruptedSegments {
        if err := e.rebuildSegment(segment, walEntries); err != nil {
            return err
        }
    }
    
    // 4. Verify integrity
    return e.validateAllSegments()
}
```

## Performance Optimizations

### Parallel I/O

```go
func (s *Segment) ReadEntriesParallel(keys []string) ([]*VectorEntry, error) {
    numWorkers := min(len(keys), runtime.NumCPU())
    jobs := make(chan string, len(keys))
    results := make(chan *VectorEntry, len(keys))
    
    // Start workers
    var wg sync.WaitGroup
    for i := 0; i < numWorkers; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for key := range jobs {
                if entry, err := s.ReadEntry(key); err == nil {
                    results <- entry
                } else {
                    results <- nil
                }
            }
        }()
    }
    
    // Send jobs
    for _, key := range keys {
        jobs <- key
    }
    close(jobs)
    
    // Collect results
    go func() {
        wg.Wait()
        close(results)
    }()
    
    var entries []*VectorEntry
    for entry := range results {
        if entry != nil {
            entries = append(entries, entry)
        }
    }
    
    return entries, nil
}
```

### Memory Mapping

```go
func (s *Segment) EnableMemoryMapping() error {
    file, err := os.Open(s.path)
    if err != nil {
        return err
    }
    
    stat, err := file.Stat()
    if err != nil {
        return err
    }
    
    // Memory map the file
    data, err := syscall.Mmap(int(file.Fd()), 0, int(stat.Size()),
        syscall.PROT_READ, syscall.MAP_SHARED)
    if err != nil {
        return err
    }
    
    s.mmapData = data
    s.mmapFile = file
    
    return nil
}
```

## Configuration and Tuning

### Storage Configuration

```go
type EngineConfig struct {
    // MemTable settings
    MemTableSize     int64
    MaxMemTables     int
    
    // Compaction settings
    CompactionStyle  CompactionStyle
    MaxLevels        int
    LevelSizeRatio   int
    
    // I/O settings
    BlockSize        int
    CompressionType  CompressionType
    BloomFilterBits  int
    
    // Cache settings
    CacheSize        int64
    BlockCacheSize   int64
    
    // WAL settings
    WALSyncPolicy    SyncPolicy
    WALBufferSize    int
}
```

### Performance Tuning Guidelines

**For Write-Heavy Workloads**:
- Larger MemTable size
- Less frequent compaction
- Faster compression (LZ4/Snappy)
- Async WAL sync

**For Read-Heavy Workloads**:
- Smaller MemTable size
- More aggressive compaction
- Better compression (Zstd)
- Larger block cache

**For Memory-Constrained Environments**:
- Smaller MemTable and cache sizes
- Enable memory mapping
- Higher compression ratios
- More frequent compaction

This storage layer design provides LibraVDB with a robust, scalable foundation for persistent vector data management while maintaining high performance for both read and write operations.