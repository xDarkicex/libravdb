package wal

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sync"
	"time"
)

// WAL implements write-ahead logging for durability
type WAL struct {
	mu     sync.RWMutex
	file   *os.File
	writer *bufio.Writer
	path   string
	offset int64
	closed bool
}

// Entry represents a single WAL entry
type Entry struct {
	Timestamp uint64
	Operation Operation
	ID        string
	Vector    []float32
	Metadata  map[string]interface{}
}

// Operation defines the type of operation
type Operation uint8

const (
	OpInsert Operation = iota
	OpUpdate
	OpDelete
)

// New creates a new WAL instance
func New(path string) (*WAL, error) {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to open WAL file: %w", err)
	}

	// Get current file size
	stat, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("failed to stat WAL file: %w", err)
	}

	wal := &WAL{
		file:   file,
		writer: bufio.NewWriter(file),
		path:   path,
		offset: stat.Size(),
	}

	return wal, nil
}

// Append adds a new entry to the WAL
func (w *WAL) Append(ctx context.Context, entry *Entry) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return fmt.Errorf("WAL is closed")
	}

	// Set timestamp if not provided
	if entry.Timestamp == 0 {
		entry.Timestamp = uint64(time.Now().UnixNano())
	}

	// Serialize entry
	data, err := w.serializeEntry(entry)
	if err != nil {
		return fmt.Errorf("failed to serialize entry: %w", err)
	}

	// Write length prefix
	if err := binary.Write(w.writer, binary.LittleEndian, uint32(len(data))); err != nil {
		return fmt.Errorf("failed to write entry length: %w", err)
	}

	// Write data
	if _, err := w.writer.Write(data); err != nil {
		return fmt.Errorf("failed to write entry data: %w", err)
	}

	// Flush to ensure durability
	if err := w.writer.Flush(); err != nil {
		return fmt.Errorf("failed to flush WAL: %w", err)
	}

	if err := w.file.Sync(); err != nil {
		return fmt.Errorf("failed to sync WAL: %w", err)
	}

	w.offset += int64(4 + len(data))
	return nil
}

// Read reads all entries from the WAL for recovery
func (w *WAL) Read() ([]*Entry, error) {
	w.mu.RLock()
	defer w.mu.RUnlock()

	// Open read-only file handle
	file, err := os.Open(w.path)
	if err != nil {
		return nil, fmt.Errorf("failed to open WAL for reading: %w", err)
	}
	defer file.Close()

	var entries []*Entry
	reader := bufio.NewReader(file)

	for {
		// Read length prefix
		var length uint32
		if err := binary.Read(reader, binary.LittleEndian, &length); err != nil {
			if err == io.EOF {
				break
			}
			return nil, fmt.Errorf("failed to read entry length: %w", err)
		}

		// Read entry data
		data := make([]byte, length)
		if _, err := io.ReadFull(reader, data); err != nil {
			return nil, fmt.Errorf("failed to read entry data: %w", err)
		}

		// Deserialize entry
		entry, err := w.deserializeEntry(data)
		if err != nil {
			return nil, fmt.Errorf("failed to deserialize entry: %w", err)
		}

		entries = append(entries, entry)
	}

	return entries, nil
}

// Truncate removes all entries from the WAL
func (w *WAL) Truncate() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return fmt.Errorf("WAL is closed")
	}

	// Close current file
	if err := w.file.Close(); err != nil {
		return fmt.Errorf("failed to close WAL file: %w", err)
	}

	// Recreate empty file
	file, err := os.Create(w.path)
	if err != nil {
		return fmt.Errorf("failed to recreate WAL file: %w", err)
	}

	w.file = file
	w.writer = bufio.NewWriter(file)
	w.offset = 0

	return nil
}

// Close shuts down the WAL
func (w *WAL) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return nil
	}

	var errors []error

	if err := w.writer.Flush(); err != nil {
		errors = append(errors, err)
	}

	if err := w.file.Sync(); err != nil {
		errors = append(errors, err)
	}

	if err := w.file.Close(); err != nil {
		errors = append(errors, err)
	}

	w.closed = true

	if len(errors) > 0 {
		return fmt.Errorf("errors during WAL close: %v", errors)
	}

	return nil
}

func (w *WAL) serializeEntry(entry *Entry) ([]byte, error) {
	// TODO: true implementation, replace json
	return json.Marshal(entry)
}

func (w *WAL) deserializeEntry(data []byte) (*Entry, error) {
	// TODO: true implementation, replace json
	var entry Entry
	err := json.Unmarshal(data, &entry)
	return &entry, err
}
