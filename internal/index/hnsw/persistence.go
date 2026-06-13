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
	h.mu.RLock()
	defer h.mu.RUnlock()
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
	nodeCount := uint32(len(h.nodes))
	if err := binary.Write(writer, binary.LittleEndian, nodeCount); err != nil {
		return err
	}

	// Write nodes in chunks for memory efficiency
	for i := 0; i < len(h.nodes); i += ChunkSize {
		end := i + ChunkSize
		if end > len(h.nodes) {
			end = len(h.nodes)
		}

		for j := i; j < end; j++ {
			node := h.nodes[j]
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

			idBytes := []byte(h.ordinalToID[node.Ordinal])
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
	for _, node := range h.nodes {
		if node != nil && len(node.Links) > 0 {
			nodeCount++
		}
	}

	// Write total node count with links
	if err := binary.Write(writer, binary.LittleEndian, uint32(nodeCount)); err != nil {
		return err
	}

	// Write links for each node
	for i, node := range h.nodes {
		if node == nil || len(node.Links) == 0 {
			continue
		}

		// Write node index
		if err := binary.Write(writer, binary.LittleEndian, uint32(i)); err != nil {
			return err
		}

		// Write number of levels
		if err := binary.Write(writer, binary.LittleEndian, uint32(len(node.Links))); err != nil {
			return err
		}

		// Write each level's connections
		for level, connections := range node.Links {
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
	if h.entryPoint != nil {
		if err := binary.Write(writer, binary.LittleEndian, uint8(1)); err != nil {
			return err
		}
		if err := binary.Write(writer, binary.LittleEndian, h.entryPoint.Ordinal); err != nil {
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
	_ = binary.Write(crc, binary.LittleEndian, uint32(len(h.nodes)))

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
	h.nodes = make([]*Node, nodeCount)

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
			h.nodes[i] = nil
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
		}
		if h.rawVectorStore != nil {
			ref, err := h.rawVectorStore.Put(vector)
			if err != nil {
				return fmt.Errorf("failed to restore raw vector into store: %w", err)
			}
			node.Slot = ref.Slot
		}
		if int(ordinal) >= len(h.nodes) {
			newCap := nextNodeCapacity(len(h.nodes), int(ordinal)+1)
			grown := make([]*Node, int(ordinal)+1, newCap)
			copy(grown, h.nodes)
			h.nodes = grown
		}
		h.nodes[ordinal] = node
		if nodeID != "" {
			h.idToIndex[nodeID] = ordinal
			h.ordinalToID[ordinal] = nodeID
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
		if int(nodeIndex) >= len(h.nodes) || h.nodes[nodeIndex] == nil {
			return fmt.Errorf("invalid node index: %d", nodeIndex)
		}

		node := h.nodes[nodeIndex]

		// Read number of levels
		var levelCount uint32
		if err := binary.Read(reader, binary.LittleEndian, &levelCount); err != nil {
			return err
		}

		// Initialize links for this node. Allocate from SFL (not Go heap)
		// so freeNodeLinks can safely deallocate them during Delete.
		node.Links = make([][]uint32, levelCount)

		// Read each level's connections
		for j := uint32(0); j < levelCount; j++ {
			var level uint32
			if err := binary.Read(reader, binary.LittleEndian, &level); err != nil {
				return err
			}

			var connectionCount uint32
			if err := binary.Read(reader, binary.LittleEndian, &connectionCount); err != nil {
				return err
			}

			// Use the same SFL allocation as newNodeLinks: level 0 uses
			// link0SFL (larger slot for 2×M connections), higher levels
			// use linkSFL.
			var slot []byte
			var slotErr error
			if level == 0 {
				slot, slotErr = h.link0SFL.Allocate()
			} else {
				slot, slotErr = h.linkSFL.Allocate()
			}
			if slotErr != nil {
				return fmt.Errorf("sfl allocate links for node %d level %d: %w", nodeIndex, level, slotErr)
			}

			// Data starts after SFLMetadataOverhead. Capacity matches what
			// newNodeLinks configures (capacity + slack).
			maxCap := (len(slot) - SFLMetadataOverhead) / 4
			if uint32(maxCap) < connectionCount {
				return fmt.Errorf("sfl slot too small for node %d level %d: need %d, have %d",
					nodeIndex, level, connectionCount, maxCap)
			}
			connections := unsafe.Slice((*uint32)(unsafe.Pointer(&slot[SFLMetadataOverhead])), maxCap)[:connectionCount]
			for k := uint32(0); k < connectionCount; k++ {
				if err := binary.Read(reader, binary.LittleEndian, &connections[k]); err != nil {
					return err
				}
			}

			if int(level) < len(node.Links) {
				node.Links[level] = connections
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
		if int(entryPointOrdinal) < len(h.nodes) {
			h.entryPoint = h.nodes[entryPointOrdinal]
		}
	}

	return nil
}

// rebuildIndexState reconstructs internal state after loading from disk
func (h *Index) rebuildIndexState() error {
	// Reset state
	h.size = 0
	h.nextOrdinal = 0
	h.maxLevel = 0
	h.idToIndex = make(map[string]uint32)
	h.entryPoint = nil

	// Rebuild state from loaded nodes
	for i, node := range h.nodes {
		if node != nil {
			h.size++
			if h.provider == nil && node.Ordinal >= h.nextOrdinal {
				h.nextOrdinal = node.Ordinal + 1
			}
			if id, ok := h.ordinalToID[uint32(i)]; ok && id != "" {
				h.idToIndex[id] = uint32(i)
			}

			// Update max level
			if node.Level > h.maxLevel {
				h.maxLevel = node.Level
			}

			// Set entry point (highest level node, or first high-level node found)
			if h.entryPoint == nil || node.Level > h.entryPoint.Level {
				h.entryPoint = node
			}
		}
	}

	// Rebuild Backlinks
	for _, node := range h.nodes {
		if node != nil {
			node.Backlinks = make([][]uint32, len(node.Links))
		}
	}
	for i, node := range h.nodes {
		if node != nil {
			for level, links := range node.Links {
				for _, linkID := range links {
					if int(linkID) < len(h.nodes) {
						linkNode := h.nodes[linkID]
						if linkNode != nil && level < len(linkNode.Backlinks) {
							linkNode.Backlinks[level] = append(linkNode.Backlinks[level], uint32(i))
						}
					}
				}
			}
		}
	}

	return nil
}
