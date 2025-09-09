package memory

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"syscall"
	"unsafe"
)

// MemoryMap represents a memory-mapped file
type MemoryMap struct {
	mu       sync.RWMutex
	file     *os.File
	data     []byte
	size     int64
	path     string
	readOnly bool
}

// NewMemoryMap creates a new memory map for the given file
func NewMemoryMap(path string, size int64, readOnly bool) (*MemoryMap, error) {
	// Ensure directory exists
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return nil, fmt.Errorf("failed to create directory: %w", err)
	}

	// Open or create file
	var file *os.File
	var err error

	if readOnly {
		file, err = os.OpenFile(path, os.O_RDONLY, 0644)
	} else {
		file, err = os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0644)
		if err == nil && size > 0 {
			// Ensure file is large enough
			if err := file.Truncate(size); err != nil {
				file.Close()
				return nil, fmt.Errorf("failed to truncate file: %w", err)
			}
		}
	}

	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}

	// Get actual file size if not specified
	if size == 0 {
		stat, err := file.Stat()
		if err != nil {
			file.Close()
			return nil, fmt.Errorf("failed to stat file: %w", err)
		}
		size = stat.Size()
	}

	if size == 0 {
		file.Close()
		return nil, fmt.Errorf("cannot memory map empty file")
	}

	// Memory map the file
	prot := syscall.PROT_READ
	if !readOnly {
		prot |= syscall.PROT_WRITE
	}

	data, err := syscall.Mmap(int(file.Fd()), 0, int(size), prot, syscall.MAP_SHARED)
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("failed to mmap file: %w", err)
	}

	return &MemoryMap{
		file:     file,
		data:     data,
		size:     size,
		path:     path,
		readOnly: readOnly,
	}, nil
}

// Data returns the memory-mapped data
func (m *MemoryMap) Data() []byte {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.data
}

// Size returns the size of the memory-mapped region
func (m *MemoryMap) Size() int64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.size
}

// Path returns the file path
func (m *MemoryMap) Path() string {
	return m.path
}

// IsReadOnly returns true if the mapping is read-only
func (m *MemoryMap) IsReadOnly() bool {
	return m.readOnly
}

// Sync flushes changes to disk
func (m *MemoryMap) Sync() error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.data == nil {
		return fmt.Errorf("memory map is closed")
	}

	if m.readOnly {
		return nil // No need to sync read-only mappings
	}

	// Use msync to flush changes
	_, _, errno := syscall.Syscall(syscall.SYS_MSYNC,
		uintptr(unsafe.Pointer(&m.data[0])),
		uintptr(m.size),
		syscall.MS_SYNC)

	if errno != 0 {
		return fmt.Errorf("msync failed: %v", errno)
	}

	return nil
}

// Close unmaps the memory and closes the file
func (m *MemoryMap) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	var err error

	// Unmap memory
	if m.data != nil {
		if unmapErr := syscall.Munmap(m.data); unmapErr != nil {
			err = fmt.Errorf("failed to unmap memory: %w", unmapErr)
		}
		m.data = nil
	}

	// Close file
	if m.file != nil {
		if closeErr := m.file.Close(); closeErr != nil && err == nil {
			err = fmt.Errorf("failed to close file: %w", closeErr)
		}
		m.file = nil
	}

	return err
}

// Resize changes the size of the memory mapping
func (m *MemoryMap) Resize(newSize int64) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.readOnly {
		return fmt.Errorf("cannot resize read-only mapping")
	}

	if m.data == nil {
		return fmt.Errorf("memory map is closed")
	}

	// Unmap current mapping
	if err := syscall.Munmap(m.data); err != nil {
		return fmt.Errorf("failed to unmap memory: %w", err)
	}

	// Resize file
	if err := m.file.Truncate(newSize); err != nil {
		return fmt.Errorf("failed to truncate file: %w", err)
	}

	// Remap with new size
	data, err := syscall.Mmap(int(m.file.Fd()), 0, int(newSize),
		syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		return fmt.Errorf("failed to remap file: %w", err)
	}

	m.data = data
	m.size = newSize

	return nil
}

// MemoryMapManager manages multiple memory mappings
type MemoryMapManager struct {
	mu       sync.RWMutex
	mappings map[string]*MemoryMap
	basePath string
}

// NewMemoryMapManager creates a new memory map manager
func NewMemoryMapManager(basePath string) *MemoryMapManager {
	return &MemoryMapManager{
		mappings: make(map[string]*MemoryMap),
		basePath: basePath,
	}
}

// CreateMapping creates a new memory mapping
func (m *MemoryMapManager) CreateMapping(name string, size int64, readOnly bool) (*MemoryMap, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.mappings[name]; exists {
		return nil, fmt.Errorf("mapping %s already exists", name)
	}

	path := filepath.Join(m.basePath, name+".mmap")
	mapping, err := NewMemoryMap(path, size, readOnly)
	if err != nil {
		return nil, fmt.Errorf("failed to create mapping: %w", err)
	}

	m.mappings[name] = mapping
	return mapping, nil
}

// GetMapping returns an existing memory mapping
func (m *MemoryMapManager) GetMapping(name string) (*MemoryMap, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	mapping, exists := m.mappings[name]
	return mapping, exists
}

// RemoveMapping removes and closes a memory mapping
func (m *MemoryMapManager) RemoveMapping(name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	mapping, exists := m.mappings[name]
	if !exists {
		return fmt.Errorf("mapping %s not found", name)
	}

	if err := mapping.Close(); err != nil {
		return fmt.Errorf("failed to close mapping: %w", err)
	}

	delete(m.mappings, name)

	// Remove the file
	if err := os.Remove(mapping.Path()); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to remove mapping file: %w", err)
	}

	return nil
}

// TotalSize returns the total size of all mappings
func (m *MemoryMapManager) TotalSize() int64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var total int64
	for _, mapping := range m.mappings {
		total += mapping.Size()
	}
	return total
}

// Close closes all mappings
func (m *MemoryMapManager) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	var firstErr error
	for name, mapping := range m.mappings {
		if err := mapping.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		delete(m.mappings, name)
	}

	return firstErr
}
