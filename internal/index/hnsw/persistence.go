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
	"time"

	"github.com/xDarkicex/libravdb/internal/util"
)

const (
	// File format constants
	HNSWMagicNumber = 0x484E5357 // "HNSW" in hex
	ChunkSize       = 1000       // Process nodes in batches
)

// Core serialization functions
func (h *Index) saveToDiskImpl(ctx context.Context, path string) error {
	h.mu.RLock()
	defer h.mu.RUnlock()

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

	if err := h.readNodes(reader); err != nil {
		return fmt.Errorf("failed to read nodes: %w", err)
	}

	if err := h.readLinks(reader); err != nil {
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
	// Magic number (4 bytes)
	if err := binary.Write(writer, binary.LittleEndian, uint32(HNSWMagicNumber)); err != nil {
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

			// Write node ID length and ID
			idBytes := []byte(node.ID)
			if err := binary.Write(writer, binary.LittleEndian, uint32(len(idBytes))); err != nil {
				return err
			}
			if _, err := writer.Write(idBytes); err != nil {
				return err
			}

			// Write vector dimension and data
			if err := binary.Write(writer, binary.LittleEndian, uint32(len(node.Vector))); err != nil {
				return err
			}
			for _, val := range node.Vector {
				if err := binary.Write(writer, binary.LittleEndian, val); err != nil {
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
		entryBytes := []byte(h.entryPoint.ID)
		if err := binary.Write(writer, binary.LittleEndian, uint32(len(entryBytes))); err != nil {
			return err
		}
		if _, err := writer.Write(entryBytes); err != nil {
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
	binary.Write(crc, binary.LittleEndian, uint32(h.config.M))
	binary.Write(crc, binary.LittleEndian, uint32(h.config.EfConstruction))
	binary.Write(crc, binary.LittleEndian, uint32(h.config.Dimension))
	binary.Write(crc, binary.LittleEndian, uint32(len(h.nodes)))

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

	// Read and validate magic number
	var magic uint32
	if err := binary.Read(file, binary.LittleEndian, &magic); err != nil {
		return fmt.Errorf("failed to read magic number: %w", err)
	}

	if magic != HNSWMagicNumber {
		return fmt.Errorf("invalid magic number: expected %x, got %x", HNSWMagicNumber, magic)
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
	// Read magic number (already validated)
	var magic uint32
	if err := binary.Read(reader, binary.LittleEndian, &magic); err != nil {
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

func (h *Index) readNodes(reader io.Reader) error {
	// Read total node count
	var nodeCount uint32
	if err := binary.Read(reader, binary.LittleEndian, &nodeCount); err != nil {
		return err
	}

	// Initialize nodes slice
	h.nodes = make([]*Node, nodeCount)

	// Read nodes
	for i := uint32(0); i < nodeCount; i++ {
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

		// Read node ID
		var idLen uint32
		if err := binary.Read(reader, binary.LittleEndian, &idLen); err != nil {
			return err
		}
		idBytes := make([]byte, idLen)
		if _, err := io.ReadFull(reader, idBytes); err != nil {
			return err
		}
		nodeID := string(idBytes)

		// Read vector dimension
		var vectorLen uint32
		if err := binary.Read(reader, binary.LittleEndian, &vectorLen); err != nil {
			return err
		}

		// Read vector data
		vector := make([]float32, vectorLen)
		for j := uint32(0); j < vectorLen; j++ {
			if err := binary.Read(reader, binary.LittleEndian, &vector[j]); err != nil {
				return err
			}
		}

		// Read level
		var level uint32
		if err := binary.Read(reader, binary.LittleEndian, &level); err != nil {
			return err
		}

		// Create node
		node := &Node{
			ID:     nodeID,
			Vector: vector,
			Level:  int(level),
		}
		h.nodes[i] = node
	}

	return nil
}

func (h *Index) readLinks(reader io.Reader) error {
	// Read total node count with links
	var nodeCount uint32
	if err := binary.Read(reader, binary.LittleEndian, &nodeCount); err != nil {
		return err
	}

	// Read links for each node
	for i := uint32(0); i < nodeCount; i++ {
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

		// Initialize links for this node
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

			connections := make([]uint32, connectionCount)
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
		// Read entry point ID
		var idLen uint32
		if err := binary.Read(reader, binary.LittleEndian, &idLen); err != nil {
			return err
		}
		idBytes := make([]byte, idLen)
		if _, err := io.ReadFull(reader, idBytes); err != nil {
			return err
		}
		entryPointID := string(idBytes)

		// Find the entry point node
		for _, node := range h.nodes {
			if node != nil && node.ID == entryPointID {
				h.entryPoint = node
				break
			}
		}
	}

	return nil
}

// rebuildIndexState reconstructs internal state after loading from disk
func (h *Index) rebuildIndexState() error {
	// Reset state
	h.size = 0
	h.maxLevel = 0
	h.idToIndex = make(map[string]uint32)
	h.entryPoint = nil

	// Rebuild state from loaded nodes
	for i, node := range h.nodes {
		if node != nil {
			h.size++
			h.idToIndex[node.ID] = uint32(i)

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

	return nil
}
