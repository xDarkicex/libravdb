package hnsw

import (
	"encoding/binary"
	"fmt"
	"os"

	internalmemory "github.com/xDarkicex/libravdb/internal/memory"
)

func (h *Index) createMmapRawVectorStore(mmapPath string) (*MmapRawVectorStore, error) {
	// Open file for writing sequentially
	file, err := os.OpenFile(mmapPath, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to create vector mmap file: %w", err)
	}
	defer file.Close()

	bytesPerVector := h.config.Dimension * 4
	active := 0

	for _, node := range h.nodes {
		if node == nil {
			// Write zeros for empty slots to maintain O(1) alignment
			zeros := make([]byte, bytesPerVector)
			if _, err := file.Write(zeros); err != nil {
				return nil, err
			}
			continue
		}

		vec, err := h.getNodeVectorLocal(node)
		if err != nil {
			return nil, fmt.Errorf("get vector for ordinal %d: %w", node.Ordinal, err)
		}
		if len(vec) != h.config.Dimension {
			// Write zeros if unavailable (e.g. deleted node)
			zeros := make([]byte, bytesPerVector)
			if _, err := file.Write(zeros); err != nil {
				return nil, err
			}
			continue
		}

		// Write vector
		for _, val := range vec {
			if err := binary.Write(file, binary.LittleEndian, val); err != nil {
				return nil, err
			}
		}
		active++
	}

	// Now memory map the file
	mmap, err := internalmemory.NewMemoryMap(mmapPath, 0, true)
	if err != nil {
		return nil, fmt.Errorf("failed to memory map vectors: %w", err)
	}

	return NewMmapRawVectorStore(h.config.Dimension, active, mmap), nil
}

func (h *Index) createMmapCompressedVectorStore(mmapPath string) (*internalmemory.MemoryMap, error) {
	file, err := os.OpenFile(mmapPath, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to create compressed vector mmap file: %w", err)
	}
	defer file.Close()

	numCodebooks := 1
	if h.config.Quantization != nil {
		numCodebooks = h.config.Quantization.Codebooks
	}
	bytesPerVector := numCodebooks // For typical PQ where 1 codebook = 1 byte
	
	for _, node := range h.nodes {
		if node == nil || node.CompressedVector == nil {
			zeros := make([]byte, bytesPerVector)
			if _, err := file.Write(zeros); err != nil {
				return nil, err
			}
			continue
		}

		if len(node.CompressedVector) != bytesPerVector {
			return nil, fmt.Errorf("compressed vector length %d mismatch expected %d", len(node.CompressedVector), bytesPerVector)
		}

		if _, err := file.Write(node.CompressedVector); err != nil {
			return nil, err
		}
	}

	// Now memory map the file
	mmap, err := internalmemory.NewMemoryMap(mmapPath, 0, true)
	if err != nil {
		return nil, fmt.Errorf("failed to memory map compressed vectors: %w", err)
	}

	// Slice into mmap data
	data := mmap.Data()
	for i, node := range h.nodes {
		if node != nil && node.CompressedVector != nil {
			offset := int64(i) * int64(bytesPerVector)
			node.CompressedVector = data[offset : offset+int64(bytesPerVector)]
		}
	}

	return mmap, nil
}
