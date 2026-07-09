package hnsw

import (
	"bufio"
	"context"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/xDarkicex/libravdb/internal/util"
	"github.com/xDarkicex/memory"
)

const (
	// File format constants
	ChunkSize = 1000 // Process nodes in batches
)

var HNSWMagicBytes = []byte("LIBRAHNS")

// Core serialization functions
func (h *Index) saveToDiskImpl(ctx context.Context, path string) error {
	return h.saveToDiskWithoutLock(ctx, path)
}

// saveToDiskWithoutLock saves to disk without acquiring locks (must be called with lock held)
func (h *Index) saveToDiskWithoutLock(ctx context.Context, path string) error {
	// Ensure directory exists
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	return atomicWrite(path, func(file *os.File) error {
		writer := bufio.NewWriter(file)
		defer writer.Flush()

		// Write in order: header, config, nodes, links, metadata
		if err := h.writeHeader(writer); err != nil {
			return fmt.Errorf("failed to write header: %w", err)
		}

		if err := h.writeConfig(writer); err != nil {
			return fmt.Errorf("failed to write config: %w", err)
		}

		if err := h.writeNodes(writer); err != nil {
			return fmt.Errorf("failed to write nodes: %w", err)
		}

		if err := h.writeLinks(writer); err != nil {
			return fmt.Errorf("failed to write links: %w", err)
		}

		if err := h.writeMetadata(writer); err != nil {
			return fmt.Errorf("failed to write metadata: %w", err)
		}

		return nil
	})
}

func (h *Index) loadFromDiskImpl(ctx context.Context, path string) error {
	if err := validateFileFormat(path); err != nil {
		return fmt.Errorf("invalid file format: %w", err)
	}

	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	reader := bufio.NewReader(file)

	// Read in order: header, config, nodes, links, metadata
	if err := h.readHeader(reader); err != nil {
		return fmt.Errorf("failed to read header: %w", err)
	}

	if err := h.readConfig(reader); err != nil {
		return fmt.Errorf("failed to read config: %w", err)
	}

	// Always restore a local raw vector store during load so the persisted
	// index can answer searches even before external storage is populated.
	if h.rawVectorStore == nil {
		switch h.config.RawVectorStore {
		case "", RawVectorStoreMemory:
			h.rawVectorStore = NewInMemoryRawVectorStore(h.config.Dimension)
		case RawVectorStoreSlabby:
			store, err := NewSlabbyRawVectorStore(h.config.Dimension, h.config.RawStoreCap)
			if err != nil {
				return fmt.Errorf("failed to create slabby raw vector store: %w", err)
			}
			h.rawVectorStore = store
		default:
			return fmt.Errorf("unsupported raw vector store backend: %s", h.config.RawVectorStore)
		}
	}

	if err := h.readNodes(ctx, reader); err != nil {
		return fmt.Errorf("failed to read nodes: %w", err)
	}

	if err := h.readLinks(ctx, reader); err != nil {
		return fmt.Errorf("failed to read links: %w", err)
	}

	if err := h.readMetadata(reader); err != nil {
		return fmt.Errorf("failed to read metadata: %w", err)
	}

	// Rebuild internal state after loading
	if err := h.rebuildIndexState(); err != nil {
		return fmt.Errorf("failed to rebuild index state: %w", err)
	}

	return nil
}

func (h *Index) writeHeader(writer io.Writer) error {
	// Magic bytes (8 bytes)
	if _, err := writer.Write(HNSWMagicBytes); err != nil {
		return err
	}

	// Format version (4 bytes)
	if err := binary.Write(writer, binary.LittleEndian, uint32(FormatVersion)); err != nil {
		return err
	}

	// Timestamp (8 bytes)
	timestamp := time.Now().Unix()
	if err := binary.Write(writer, binary.LittleEndian, int64(timestamp)); err != nil {
		return err
	}

	// CRC32 placeholder (4 bytes) - will be updated later
	crc := h.calculateCRC32()
	if err := binary.Write(writer, binary.LittleEndian, crc); err != nil {
		return err
	}

	return nil
}

func (h *Index) writeConfig(writer io.Writer) error {
	// Write index parameters
	if err := binary.Write(writer, binary.LittleEndian, uint32(h.config.M)); err != nil {
		return err
	}
	if err := binary.Write(writer, binary.LittleEndian, uint32(h.config.EfConstruction)); err != nil {
		return err
	}
	if err := binary.Write(writer, binary.LittleEndian, uint32(h.config.EfSearch)); err != nil {
		return err
	}
	if err := binary.Write(writer, binary.LittleEndian, uint32(16)); err != nil { // MaxLevels hardcoded for now
		return err
	}

	// Write dimension
	if err := binary.Write(writer, binary.LittleEndian, uint32(h.config.Dimension)); err != nil {
		return err
	}

	// Write metric type (as uint32)
	if err := binary.Write(writer, binary.LittleEndian, uint32(h.config.Metric)); err != nil {
		return err
	}

	return nil
}

func (h *Index) writeNodes(writer io.Writer) error {
	// Write total node count
	nodeCount := uint32(h.nodes.Len())
	if err := binary.Write(writer, binary.LittleEndian, nodeCount); err != nil {
		return err
	}

	// Write nodes in chunks for memory efficiency
	for i := 0; i < h.nodes.Len(); i += ChunkSize {
		end := i + ChunkSize
		if end > h.nodes.Len() {
			end = h.nodes.Len()
		}

		for j := i; j < end; j++ {
			node := h.nodes.Get(uint32(j))
			if node == nil {
				// Write marker for nil node
				if err := binary.Write(writer, binary.LittleEndian, uint8(0)); err != nil {
					return err
				}
				continue
			}

			// Write marker for valid node
			if err := binary.Write(writer, binary.LittleEndian, uint8(1)); err != nil {
				return err
			}

			if err := binary.Write(writer, binary.LittleEndian, node.Ordinal); err != nil {
				return err
			}

			idBytes := []byte(h.ordinalToID.Get(node.Ordinal))
			if err := binary.Write(writer, binary.LittleEndian, uint32(len(idBytes))); err != nil {
				return err
			}
			if _, err := writer.Write(idBytes); err != nil {
				return err
			}

			vector, err := h.getNodeVectorLocal(node)
			if err != nil {
				return fmt.Errorf("failed to get node vector for persistence: %w", err)
			}

			// Write vector dimension and data
			if err := binary.Write(writer, binary.LittleEndian, uint32(len(vector))); err != nil {
				return err
			}
			if len(vector) > 0 {
				vectorBytes := unsafe.Slice((*byte)(unsafe.Pointer(&vector[0])), len(vector)*4)
				if _, err := writer.Write(vectorBytes); err != nil {
					return err
				}
			}

			// Write level
			if err := binary.Write(writer, binary.LittleEndian, uint32(node.Level)); err != nil {
				return err
			}
		}
	}

	return nil
}

func (h *Index) writeLinks(writer io.Writer) error {
	// Count nodes with links
	nodeCount := 0
	for i := 0; i < h.nodes.Len(); i++ {
		node := h.nodes.Get(uint32(i))
		if node == nil {
			continue
		}
		if node != nil && (node.Level+1) > 0 {
			nodeCount++
		}
	}

	// Write total node count with links
	if err := binary.Write(writer, binary.LittleEndian, uint32(nodeCount)); err != nil {
		return err
	}

	// Write links for each node
	for i := 0; i < h.nodes.Len(); i++ {
		node := h.nodes.Get(uint32(i))
		if node == nil {
			continue
		}
		if node == nil || (node.Level+1) == 0 {
			continue
		}

		// Write node index
		if err := binary.Write(writer, binary.LittleEndian, uint32(i)); err != nil {
			return err
		}

		// Write number of levels
		if err := binary.Write(writer, binary.LittleEndian, uint32((node.Level + 1))); err != nil {
			return err
		}

		// Write each level's connections
		for level := 0; level <= node.Level; level++ {
			connections := h.getNodeLinks(node, level)
			if err := binary.Write(writer, binary.LittleEndian, uint32(level)); err != nil {
				return err
			}
			if err := binary.Write(writer, binary.LittleEndian, uint32(len(connections))); err != nil {
				return err
			}

			for _, connIndex := range connections {
				if err := binary.Write(writer, binary.LittleEndian, connIndex); err != nil {
					return err
				}
			}
		}
	}

	return nil
}

func (h *Index) writeMetadata(writer io.Writer) error {
	// Write entry point
	if h.getEntryPoint() != nil {
		if err := binary.Write(writer, binary.LittleEndian, uint8(1)); err != nil {
			return err
		}
		if err := binary.Write(writer, binary.LittleEndian, h.getEntryPoint().Ordinal); err != nil {
			return err
		}
	} else {
		if err := binary.Write(writer, binary.LittleEndian, uint8(0)); err != nil {
			return err
		}
	}

	return nil
}

func (h *Index) calculateCRC32() uint32 {
	// Simple CRC32 calculation based on key parameters
	crc := crc32.NewIEEE()

	// Include key parameters in CRC
	_ = binary.Write(crc, binary.LittleEndian, uint32(h.config.M))
	_ = binary.Write(crc, binary.LittleEndian, uint32(h.config.EfConstruction))
	_ = binary.Write(crc, binary.LittleEndian, uint32(h.config.Dimension))
	_ = binary.Write(crc, binary.LittleEndian, uint32(h.nodes.Len()))

	return crc.Sum32()
}

// Atomic file operations
func atomicWrite(finalPath string, writeFunc func(*os.File) error) error {
	// 1. Write to temporary file
	tempPath := finalPath + ".tmp"
	file, err := os.Create(tempPath)
	if err != nil {
		return fmt.Errorf("failed to create temp file: %w", err)
	}

	// 2. Write all data
	writeErr := writeFunc(file)

	// 3. Sync and close
	if syncErr := file.Sync(); syncErr != nil && writeErr == nil {
		writeErr = syncErr
	}

	if closeErr := file.Close(); closeErr != nil && writeErr == nil {
		writeErr = closeErr
	}

	// 4. Clean up on error
	if writeErr != nil {
		os.Remove(tempPath)
		return fmt.Errorf("failed to write data: %w", writeErr)
	}

	// 5. Atomic rename
	if err := os.Rename(tempPath, finalPath); err != nil {
		os.Remove(tempPath)
		return fmt.Errorf("failed to rename temp file: %w", err)
	}

	return nil
}

func validateFileFormat(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	// Read and validate magic bytes
	magic := make([]byte, 8)
	if _, err := io.ReadFull(file, magic); err != nil {
		return fmt.Errorf("failed to read magic bytes: %w", err)
	}

	if string(magic) != string(HNSWMagicBytes) {
		return fmt.Errorf("invalid magic number: expected %q, got %q", HNSWMagicBytes, magic)
	}

	// Read and validate version
	var version uint32
	if err := binary.Read(file, binary.LittleEndian, &version); err != nil {
		return fmt.Errorf("failed to read version: %w", err)
	}

	if version != FormatVersion {
		return fmt.Errorf("unsupported format version: expected %d, got %d", FormatVersion, version)
	}

	return nil
}

// Read functions for loading from disk
func (h *Index) readHeader(reader io.Reader) error {
	// Read magic bytes (already validated, but we must consume them from the reader)
	magic := make([]byte, 8)
	if _, err := io.ReadFull(reader, magic); err != nil {
		return err
	}

	// Read version (already validated)
	var version uint32
	if err := binary.Read(reader, binary.LittleEndian, &version); err != nil {
		return err
	}

	// Read timestamp
	var timestamp int64
	if err := binary.Read(reader, binary.LittleEndian, &timestamp); err != nil {
		return err
	}

	// Read CRC32 (for future validation)
	var crc uint32
	if err := binary.Read(reader, binary.LittleEndian, &crc); err != nil {
		return err
	}

	return nil
}

func (h *Index) readConfig(reader io.Reader) error {
	// Read index parameters
	var m, efConstruction, efSearch, maxLevels uint32
	if err := binary.Read(reader, binary.LittleEndian, &m); err != nil {
		return err
	}
	if err := binary.Read(reader, binary.LittleEndian, &efConstruction); err != nil {
		return err
	}
	if err := binary.Read(reader, binary.LittleEndian, &efSearch); err != nil {
		return err
	}
	if err := binary.Read(reader, binary.LittleEndian, &maxLevels); err != nil {
		return err
	}

	// Update parameters
	h.config.M = int(m)
	h.config.EfConstruction = int(efConstruction)
	h.config.EfSearch = int(efSearch)
	// MaxLevels is not stored in config, skip

	// Read dimension
	var dimension uint32
	if err := binary.Read(reader, binary.LittleEndian, &dimension); err != nil {
		return err
	}
	h.config.Dimension = int(dimension)

	// Read metric type
	var metric uint32
	if err := binary.Read(reader, binary.LittleEndian, &metric); err != nil {
		return err
	}
	h.config.Metric = util.DistanceMetric(metric)

	return nil
}

func (h *Index) readNodes(ctx context.Context, reader io.Reader) error {
	// Read total node count
	var nodeCount uint32
	if err := binary.Read(reader, binary.LittleEndian, &nodeCount); err != nil {
		return err
	}

	// Initialize nodes slice
	// h.nodes = newSegmentedNodeArray() handled

	// Use the scratch arena for temporary per-node buffers (id bytes, vector
	// data) so deserialization doesn't allocate on the Go heap.
	arena := h.scratchPool.Get().(*memory.Arena)
	defer func() {
		arena.Reset()
		h.scratchPool.Put(arena)
	}()

	// Read nodes
	for i := uint32(0); i < nodeCount; i++ {
		if i%1024 == 0 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}
		}

		// Read node marker
		var marker uint8
		if err := binary.Read(reader, binary.LittleEndian, &marker); err != nil {
			return err
		}

		if marker == 0 {
			// Nil node
			// nil removed
			continue
		}

		var ordinal uint32
		if err := binary.Read(reader, binary.LittleEndian, &ordinal); err != nil {
			return err
		}

		var idLen uint32
		if err := binary.Read(reader, binary.LittleEndian, &idLen); err != nil {
			return err
		}
		idBytes, err := memory.ArenaSlice[byte](arena, int(idLen))
		if err != nil {
			return fmt.Errorf("arena allocate id bytes: %w", err)
		}
		idBytes = idBytes[:idLen]
		if _, err := io.ReadFull(reader, idBytes); err != nil {
			return err
		}
		// Clone to heap: idBytes is arena-backed and will be invalidated
		// when the arena is Reset+returned to scratchPool below.
		nodeID := strings.Clone(string(idBytes))

		// Read vector dimension
		var vectorLen uint32
		if err := binary.Read(reader, binary.LittleEndian, &vectorLen); err != nil {
			return err
		}

		// Read vector data — arena-backed bulk read instead of
		// binary.Read per element (reflection overhead × dim).
		vector, err := memory.ArenaSlice[float32](arena, int(vectorLen))
		if err != nil {
			return fmt.Errorf("arena allocate vector: %w", err)
		}
		vector = vector[:vectorLen]
		raw := unsafe.Slice((*byte)(unsafe.Pointer(&vector[0])), int(vectorLen)*4)
		if _, err := io.ReadFull(reader, raw); err != nil {
			return err
		}

		// Read level
		var level uint32
		if err := binary.Read(reader, binary.LittleEndian, &level); err != nil {
			return err
		}

		// Create node
		node := &Node{
			Ordinal: ordinal,
			Level:   int(level),
			Slot:    SentinelNodeID,
		}
		if h.rawVectorStore != nil {
			ref, err := h.rawVectorStore.Put(vector)
			if err != nil {
				return fmt.Errorf("failed to restore raw vector into store: %w", err)
			}
			node.Slot = ref.Slot
			if vec, err := h.rawVectorStore.Get(ref); err == nil {
				node.setVector(vec)
			}
		}

		h.nodes.Set(ordinal, node)
		if nodeID != "" {
			node := h.nodes.Get(ordinal)
			h.idToIndex.Put(hashID(nodeID), node)
			h.ordinalToID.Set(ordinal, nodeID)
		}

		// Reset the arena after each node to prevent exhaustion on large files
		arena.Reset()
	}

	return nil
}

func (h *Index) readLinks(ctx context.Context, reader io.Reader) error {
	// Read total node count with links
	var nodeCount uint32
	if err := binary.Read(reader, binary.LittleEndian, &nodeCount); err != nil {
		return err
	}

	// Read links for each node
	for i := uint32(0); i < nodeCount; i++ {
		if i%1024 == 0 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}
		}

		// Read node index
		var nodeIndex uint32
		if err := binary.Read(reader, binary.LittleEndian, &nodeIndex); err != nil {
			return err
		}

		// Validate node index
		if int(nodeIndex) >= h.nodes.Len() || h.nodes.Get(nodeIndex) == nil {
			return fmt.Errorf("invalid node index: %d", nodeIndex)
		}

		node := h.nodes.Get(nodeIndex)

		// Read number of levels
		var levelCount uint32
		if err := binary.Read(reader, binary.LittleEndian, &levelCount); err != nil {
			return err
		}

		// Initialize links for this node. Allocate from SFL (not Go heap)
		// so freeNodeLinks can safely deallocate them during Delete.
		node.Links, node.Backlinks = h.newNodeArrays(node.Level, h.config.M)
		for j := uint32(0); j < levelCount; j++ {
			var level uint32
			if err := binary.Read(reader, binary.LittleEndian, &level); err != nil {
				return err
			}

			var connectionCount uint32
			if err := binary.Read(reader, binary.LittleEndian, &connectionCount); err != nil {
				return err
			}

			if int(level) < (node.Level + 1) {
				maxCount := uint32(linkArrayCapacity(h.config.M, int(level)))
				if connectionCount > maxCount {
					return fmt.Errorf("connection count %d exceeds level %d capacity %d", connectionCount, level, maxCount)
				}
				destSlice := unsafe.Slice(node.Links[level], int(maxCount))
				for k := uint32(0); k < connectionCount; k++ {
					if err := binary.Read(reader, binary.LittleEndian, &destSlice[k]); err != nil {
						return err
					}
				}
				atomic.StoreUint32(&node.LinkCounts[level], connectionCount)
			} else {
				// Skip the bytes if the level is invalid, to keep reader in sync
				for k := uint32(0); k < connectionCount; k++ {
					var dummy uint32
					if err := binary.Read(reader, binary.LittleEndian, &dummy); err != nil {
						return err
					}
				}
			}
		}
	}

	return nil
}

func (h *Index) readMetadata(reader io.Reader) error {
	// Read entry point marker
	var hasEntryPoint uint8
	if err := binary.Read(reader, binary.LittleEndian, &hasEntryPoint); err != nil {
		return err
	}

	if hasEntryPoint == 1 {
		var entryPointOrdinal uint32
		if err := binary.Read(reader, binary.LittleEndian, &entryPointOrdinal); err != nil {
			return err
		}
		if int(entryPointOrdinal) < h.nodes.Len() {
			h.setEntryPoint(h.nodes.Get(entryPointOrdinal))
		}
	}

	return nil
}

// rebuildIndexState reconstructs internal state after loading from disk
func (h *Index) rebuildIndexState() error {
	// Reset state
	h.size.Store(0)
	h.nextOrdinal.Store(0)
	/* maxLevel handled by globalState */
	h.setEntryPoint(nil)

	// Rebuild state from loaded nodes
	for i := 0; i < h.nodes.Len(); i++ {
		node := h.nodes.Get(uint32(i))
		if node == nil {
			continue
		}
		if node != nil {
			h.size.Add(1)
			if h.provider == nil && node.Ordinal >= h.nextOrdinal.Load() {
				h.nextOrdinal.Store(node.Ordinal + 1)
			}
			// Note: idToIndex and ordinalToID are already populated during readNodes

			// Set entry point (highest level node, or first high-level node found)
			if h.getEntryPoint() == nil || node.Level > h.getEntryPoint().Level {
				h.setEntryPoint(node)
			}
		}
	}

	// Rebuild Backlinks
	// We no longer clear Backlinks since newNodeArrays already sets them to SentinelNodeID
	for i := 0; i < h.nodes.Len(); i++ {
		node := h.nodes.Get(uint32(i))
		if node == nil {
			continue
		}
		if node != nil {
			for level := 0; level <= node.Level; level++ {
				links := h.getNodeLinks(node, level)
				for _, linkID := range links {
					if int(linkID) < h.nodes.Len() {
						linkNode := h.nodes.Get(linkID)
						if linkNode != nil && level < (linkNode.Level+1) {
							// Lock-free append to backlink
							h.appendWithSpinlock(linkNode, linkNode.Backlinks[level], uint32(i), h.config.M, level)
						}
					}
				}
			}
		}
	}

	return nil
}
