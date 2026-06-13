package wal

// Entry represents a single WAL entry
type Entry struct {
	Metadata  map[string]interface{}
	ID        string
	Vector    []float32
	Timestamp uint64
	Operation Operation
	Data      []byte // Opaque binary data for graph records
}

// Operation defines the type of operation
type Operation uint8

const (
	OpInsert Operation = iota
	OpUpdate
	OpDelete

	// Graph operations
	OpEdgeAdd      Operation = 0x40
	OpEdgeRemove   Operation = 0x41
	OpNodeEdgeDrop Operation = 0x42
	OpTxnCommit    Operation = 0x4F
)
