//go:build windows

package memory

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"unsafe"

	"golang.org/x/sys/windows"
)

// ErrPlatformUnsupported is returned when mmap is not supported on the platform
var ErrPlatformUnsupported = errors.New("memory mapping is not supported on this platform")

// MemoryMap represents a memory-mapped file on Windows
type MemoryMap struct {
	mu       sync.RWMutex
	file     *os.File
	data     []byte
	size     int64
	path     string
	readOnly bool

	// Windows-specific handle for file mapping
	mappingHandle windows.Handle
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

	// Create file mapping
	var access uint32
	if readOnly {
		access = windows.PAGE_READONLY
	} else {
		access = windows.PAGE_READWRITE
	}

	mappingHandle, err := windows.CreateFileMapping(
		windows.Handle(file.Fd()),
		nil,
		access,
		uint32(size>>32),
		uint32(size),
		nil,
	)
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("failed to create file mapping: %w", err)
	}

	// Map the file into memory
	var mapAccess uint32
	if readOnly {
		mapAccess = windows.FILE_MAP_READ
	} else {
		mapAccess = windows.FILE_MAP_READ | windows.FILE_MAP_WRITE
	}

	addr, err := windows.MapViewOfFile(
		mappingHandle,
		mapAccess,
		0, // offset high
		0, // offset low
		uintptr(size),
	)
	if err != nil {
		windows.CloseHandle(mappingHandle)
		file.Close()
		return nil, fmt.Errorf("failed to map view of file: %w", err)
	}

	// Create slice from the mapped memory
	data := unsafe.Slice((*byte)(unsafe.Pointer(addr)), int(size))

	return &MemoryMap{
		file:          file,
		data:          data,
		size:          size,
		path:          path,
		readOnly:      readOnly,
		mappingHandle: mappingHandle,
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
		return nil
	}

	// Flush view of mapped memory
	if err := windows.FlushViewOfFile(uintptr(unsafe.Pointer(&m.data[0])), uintptr(m.size)); err != nil {
		return fmt.Errorf("failed to flush view: %w", err)
	}

	// Ensure file metadata is flushed
	if m.file != nil {
		if err := m.file.Sync(); err != nil {
			return fmt.Errorf("failed to sync file: %w", err)
		}
	}

	return nil
}

// Close unmaps the memory and closes the file
func (m *MemoryMap) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	var err error

	// Unmap the view
	if m.data != nil {
		if unmapErr := windows.UnmapViewOfFile(uintptr(unsafe.Pointer(&m.data[0]))); unmapErr != nil {
			err = fmt.Errorf("failed to unmap view of file: %w", unmapErr)
		}
		m.data = nil
	}

	// Close the mapping handle
	if m.mappingHandle != 0 {
		if closeErr := windows.CloseHandle(m.mappingHandle); closeErr != nil && err == nil {
			err = fmt.Errorf("failed to close mapping handle: %w", closeErr)
		}
		m.mappingHandle = 0
	}

	// Close the file
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

	// Unmap current view
	if unmapErr := windows.UnmapViewOfFile(uintptr(unsafe.Pointer(&m.data[0]))); unmapErr != nil {
		return fmt.Errorf("failed to unmap view of file: %w", unmapErr)
	}

	// Close current mapping handle
	if m.mappingHandle != 0 {
		windows.CloseHandle(m.mappingHandle)
		m.mappingHandle = 0
	}

	// Resize file
	if err := m.file.Truncate(newSize); err != nil {
		return fmt.Errorf("failed to truncate file: %w", err)
	}

	// Create new mapping with new size
	mappingHandle, err := windows.CreateFileMapping(
		windows.Handle(m.file.Fd()),
		nil,
		windows.PAGE_READWRITE,
		uint32(newSize>>32),
		uint32(newSize),
		nil,
	)
	if err != nil {
		return fmt.Errorf("failed to create new file mapping: %w", err)
	}

	// Map new view
	addr, err := windows.MapViewOfFile(
		mappingHandle,
		windows.FILE_MAP_READ|windows.FILE_MAP_WRITE,
		0,
		0,
		uintptr(newSize),
	)
	if err != nil {
		windows.CloseHandle(mappingHandle)
		return fmt.Errorf("failed to map new view of file: %w", err)
	}

	m.data = unsafe.Slice((*byte)(unsafe.Pointer(addr)), int(newSize))
	m.size = newSize
	m.mappingHandle = mappingHandle

	return nil
}

// MemoryMapManager manages multiple memory mappings on Windows
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