package wal

import (
	"bufio"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"sync"
	"time"

	"github.com/xDarkicex/libravdb/internal/util"
)

// WAL implements write-ahead logging for durability
type WAL struct {
	file   *os.File
	writer *bufio.Writer
	path   string
	offset int64
	mu     sync.RWMutex
	closed bool
}

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

	return w.appendEntriesLocked(ctx, []*Entry{entry})
}

// AppendBatch adds multiple entries to the WAL and flushes once for the batch.
func (w *WAL) AppendBatch(ctx context.Context, entries []*Entry) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return fmt.Errorf("WAL is closed")
	}

	return w.appendEntriesLocked(ctx, entries)
}

func (w *WAL) appendEntriesLocked(ctx context.Context, entries []*Entry) error {
	if len(entries) == 0 {
		return nil
	}

	var lenBuf [4]byte
	for _, entry := range entries {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		if entry == nil {
			return fmt.Errorf("WAL entry cannot be nil")
		}

		// Set timestamp if not provided
		if entry.Timestamp == 0 {
			entry.Timestamp = uint64(time.Now().UnixNano())
		}

		// Serialize the entire entry natively
		enc := util.AcquireBinaryEncoder(8 + 1 + 4 + len(entry.ID) + 4 + len(entry.Vector)*4 + util.EstimateMetadataSize(entry.Metadata) + 4 + len(entry.Data))
		enc.WriteUint64(entry.Timestamp)
		enc.WriteByte(byte(entry.Operation))
		enc.WriteString(entry.ID)
		enc.WriteVector(entry.Vector)

		err := enc.WriteMetadata(entry.Metadata)
		if err != nil {
			util.ReleaseBinaryEncoder(enc)
			return fmt.Errorf("failed to encode metadata: %w", err)
		}

		enc.WriteBytes(entry.Data)
		err = nil
		if err != nil {
			util.ReleaseBinaryEncoder(enc)
			return fmt.Errorf("failed to encode metadata: %w", err)
		}

		entryBytes := enc.Bytes()

		if len(entryBytes) > maxWALEntrySize {
			util.ReleaseBinaryEncoder(enc)
			return fmt.Errorf("WAL entry size %d exceeds limit %d", len(entryBytes), maxWALEntrySize)
		}

		// Write length prefix
		binary.LittleEndian.PutUint32(lenBuf[:], uint32(len(entryBytes)))
		if _, err := w.writer.Write(lenBuf[:]); err != nil {
			util.ReleaseBinaryEncoder(enc)
			return fmt.Errorf("failed to write entry length: %w", err)
		}

		if _, err := w.writer.Write(entryBytes); err != nil {
			util.ReleaseBinaryEncoder(enc)
			return fmt.Errorf("failed to write entry data: %w", err)
		}

		w.offset += int64(4 + len(entryBytes))
		util.ReleaseBinaryEncoder(enc)
	}

	// Flush to ensure durability
	if err := w.writer.Flush(); err != nil {
		return fmt.Errorf("failed to flush WAL: %w", err)
	}

	if err := w.file.Sync(); err != nil {
		return fmt.Errorf("failed to sync WAL: %w", err)
	}

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
		if length > maxWALEntrySize {
			return nil, fmt.Errorf("WAL entry size %d exceeds limit %d", length, maxWALEntrySize)
		}
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

func (w *WAL) deserializeEntry(data []byte) (*Entry, error) {
	dec := &util.BinaryDecoder{Data: data}
	entry := &Entry{}
	var err error

	entry.Timestamp, err = dec.ReadUint64()
	if err != nil {
		return nil, fmt.Errorf("read timestamp: %w", err)
	}

	opByte, err := dec.ReadByte()
	if err != nil {
		return nil, fmt.Errorf("read op: %w", err)
	}
	entry.Operation = Operation(opByte)

	entry.ID, err = dec.ReadString()
	if err != nil {
		return nil, fmt.Errorf("read id: %w", err)
	}

	entry.Vector, err = dec.ReadVector()
	if err != nil {
		return nil, fmt.Errorf("read vector: %w", err)
	}

	entry.Metadata, err = dec.ReadMetadata()
	if err != nil {
		return nil, fmt.Errorf("read metadata: %w", err)
	}

	if dec.Off < len(dec.Data) {
		entry.Data, err = dec.ReadBytes()
		if err != nil {
			return nil, fmt.Errorf("read data: %w", err)
		}
	}

	return entry, nil
}
