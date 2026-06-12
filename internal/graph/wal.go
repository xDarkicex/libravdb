package graph

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"log"
	"math"
	"sync/atomic"
	"time"

	"github.com/xDarkicex/libravdb/internal/storage/wal"
)

// WALEdgeAddRecord represents a fixed-width record for adding an edge (40 bytes total).
type WALEdgeAddRecord struct {
	TxnID  uint64  // 8 bytes - transaction identifier
	From   uint64  // 8 bytes - source node ID
	To     uint64  // 8 bytes - target node ID
	Weight float32 // 4 bytes - edge weight
	Stamp  uint32  // 4 bytes - MVCC stamp
	Kind   uint8   // 1 byte  - edge kind
	_      [3]byte // 3 bytes - padding
	CRC32  uint32  // 4 bytes - record checksum
}

// WALEdgeRemoveRecord represents a fixed-width record for removing an edge (32 bytes total).
type WALEdgeRemoveRecord struct {
	TxnID uint64  // 8 bytes
	From  uint64  // 8 bytes
	To    uint64  // 8 bytes
	Kind  uint8   // 1 byte
	_     [3]byte // 3 bytes - padding
	CRC32 uint32  // 4 bytes
}

// WALNodeEdgeDropRecord represents a fixed-width record for dropping all edges of a node (24 bytes total).
type WALNodeEdgeDropRecord struct {
	TxnID  uint64  // 8 bytes
	NodeID uint64  // 8 bytes
	CRC32  uint32  // 4 bytes
	_      [4]byte // 4 bytes - padding
}

// WALTxnCommitRecord represents a fixed-width record for committing a transaction (16 bytes total).
type WALTxnCommitRecord struct {
	TxnID uint64  // 8 bytes
	CRC32 uint32  // 4 bytes
	_     [4]byte // 4 bytes - padding
}

// SerializeWALEdgeAddRecord serializes the record and computes its CRC32.
func SerializeWALEdgeAddRecord(r *WALEdgeAddRecord) []byte {
	buf := make([]byte, 40)
	binary.LittleEndian.PutUint64(buf[0:8], r.TxnID)
	binary.LittleEndian.PutUint64(buf[8:16], r.From)
	binary.LittleEndian.PutUint64(buf[16:24], r.To)
	binary.LittleEndian.PutUint32(buf[24:28], math.Float32bits(r.Weight))
	binary.LittleEndian.PutUint32(buf[28:32], r.Stamp)
	buf[32] = r.Kind
	// 33,34,35 are padding
	
	r.CRC32 = crc32.ChecksumIEEE(buf[:36])
	binary.LittleEndian.PutUint32(buf[36:40], r.CRC32)
	return buf
}

// DeserializeWALEdgeAddRecord deserializes the record and validates its CRC32.
func DeserializeWALEdgeAddRecord(data []byte) (*WALEdgeAddRecord, error) {
	if len(data) < 40 {
		return nil, fmt.Errorf("WALEdgeAddRecord too short: %d bytes", len(data))
	}
	r := &WALEdgeAddRecord{
		TxnID:  binary.LittleEndian.Uint64(data[0:8]),
		From:   binary.LittleEndian.Uint64(data[8:16]),
		To:     binary.LittleEndian.Uint64(data[16:24]),
		Weight: math.Float32frombits(binary.LittleEndian.Uint32(data[24:28])),
		Stamp:  binary.LittleEndian.Uint32(data[28:32]),
		Kind:   data[32],
		CRC32:  binary.LittleEndian.Uint32(data[36:40]),
	}
	
	expectedCRC := crc32.ChecksumIEEE(data[:36])
	if expectedCRC != r.CRC32 {
		return nil, fmt.Errorf("CRC32 mismatch in WALEdgeAddRecord: expected %d, got %d", expectedCRC, r.CRC32)
	}
	return r, nil
}

// SerializeWALEdgeRemoveRecord serializes the record and computes its CRC32.
func SerializeWALEdgeRemoveRecord(r *WALEdgeRemoveRecord) []byte {
	buf := make([]byte, 32)
	binary.LittleEndian.PutUint64(buf[0:8], r.TxnID)
	binary.LittleEndian.PutUint64(buf[8:16], r.From)
	binary.LittleEndian.PutUint64(buf[16:24], r.To)
	buf[24] = r.Kind
	// 25,26,27 are padding
	
	r.CRC32 = crc32.ChecksumIEEE(buf[:28])
	binary.LittleEndian.PutUint32(buf[28:32], r.CRC32)
	return buf
}

// DeserializeWALEdgeRemoveRecord deserializes the record and validates its CRC32.
func DeserializeWALEdgeRemoveRecord(data []byte) (*WALEdgeRemoveRecord, error) {
	if len(data) < 32 {
		return nil, fmt.Errorf("WALEdgeRemoveRecord too short: %d bytes", len(data))
	}
	r := &WALEdgeRemoveRecord{
		TxnID: binary.LittleEndian.Uint64(data[0:8]),
		From:  binary.LittleEndian.Uint64(data[8:16]),
		To:    binary.LittleEndian.Uint64(data[16:24]),
		Kind:  data[24],
		CRC32: binary.LittleEndian.Uint32(data[28:32]),
	}
	
	expectedCRC := crc32.ChecksumIEEE(data[:28])
	if expectedCRC != r.CRC32 {
		return nil, fmt.Errorf("CRC32 mismatch in WALEdgeRemoveRecord: expected %d, got %d", expectedCRC, r.CRC32)
	}
	return r, nil
}

// SerializeWALNodeEdgeDropRecord serializes the record and computes its CRC32.
func SerializeWALNodeEdgeDropRecord(r *WALNodeEdgeDropRecord) []byte {
	buf := make([]byte, 24)
	binary.LittleEndian.PutUint64(buf[0:8], r.TxnID)
	binary.LittleEndian.PutUint64(buf[8:16], r.NodeID)
	
	r.CRC32 = crc32.ChecksumIEEE(buf[:16])
	binary.LittleEndian.PutUint32(buf[16:20], r.CRC32)
	// 20,21,22,23 are padding
	return buf
}

// DeserializeWALNodeEdgeDropRecord deserializes the record and validates its CRC32.
func DeserializeWALNodeEdgeDropRecord(data []byte) (*WALNodeEdgeDropRecord, error) {
	if len(data) < 24 {
		return nil, fmt.Errorf("WALNodeEdgeDropRecord too short: %d bytes", len(data))
	}
	r := &WALNodeEdgeDropRecord{
		TxnID:  binary.LittleEndian.Uint64(data[0:8]),
		NodeID: binary.LittleEndian.Uint64(data[8:16]),
		CRC32:  binary.LittleEndian.Uint32(data[16:20]),
	}
	
	expectedCRC := crc32.ChecksumIEEE(data[:16])
	if expectedCRC != r.CRC32 {
		return nil, fmt.Errorf("CRC32 mismatch in WALNodeEdgeDropRecord: expected %d, got %d", expectedCRC, r.CRC32)
	}
	return r, nil
}

// SerializeWALTxnCommitRecord serializes the record and computes its CRC32.
func SerializeWALTxnCommitRecord(r *WALTxnCommitRecord) []byte {
	buf := make([]byte, 16)
	binary.LittleEndian.PutUint64(buf[0:8], r.TxnID)
	
	r.CRC32 = crc32.ChecksumIEEE(buf[:8])
	binary.LittleEndian.PutUint32(buf[8:12], r.CRC32)
	// 12,13,14,15 are padding
	return buf
}

// DeserializeWALTxnCommitRecord deserializes the record and validates its CRC32.
func DeserializeWALTxnCommitRecord(data []byte) (*WALTxnCommitRecord, error) {
	if len(data) < 16 {
		return nil, fmt.Errorf("WALTxnCommitRecord too short: %d bytes", len(data))
	}
	r := &WALTxnCommitRecord{
		TxnID: binary.LittleEndian.Uint64(data[0:8]),
		CRC32: binary.LittleEndian.Uint32(data[8:12]),
	}
	
	expectedCRC := crc32.ChecksumIEEE(data[:8])
	if expectedCRC != r.CRC32 {
		return nil, fmt.Errorf("CRC32 mismatch in WALTxnCommitRecord: expected %d, got %d", expectedCRC, r.CRC32)
	}
	return r, nil
}

// ReplayWAL reads WAL records, groups them by transaction, and replays committed transactions.
func ReplayWAL(w *wal.WAL, forwardIndex *graphStore) error {
	start := time.Now()
	entries, err := w.Read()
	if err != nil {
		return fmt.Errorf("failed to read WAL: %w", err)
	}

	// Group records by TxnID
	// TxnID is extracted from the data, but what if it's a vector op?
	// We only care about graph ops. We group graph ops by TxnID.
	txnGroups := make(map[uint64][]*wal.Entry)
	txnCommitted := make(map[uint64]bool)

	for _, entry := range entries {
		switch entry.Operation {
		case wal.OpEdgeAdd:
			if len(entry.Data) >= 8 {
				txnID := binary.LittleEndian.Uint64(entry.Data[0:8])
				txnGroups[txnID] = append(txnGroups[txnID], entry)
			}
		case wal.OpEdgeRemove:
			if len(entry.Data) >= 8 {
				txnID := binary.LittleEndian.Uint64(entry.Data[0:8])
				txnGroups[txnID] = append(txnGroups[txnID], entry)
			}
		case wal.OpNodeEdgeDrop:
			if len(entry.Data) >= 8 {
				txnID := binary.LittleEndian.Uint64(entry.Data[0:8])
				txnGroups[txnID] = append(txnGroups[txnID], entry)
			}
		case wal.OpTxnCommit:
			if len(entry.Data) >= 8 {
				txnID := binary.LittleEndian.Uint64(entry.Data[0:8])
				// Also validate CRC32 for commit record
				if _, err := DeserializeWALTxnCommitRecord(entry.Data); err == nil {
					txnCommitted[txnID] = true
				}
			}
		}
	}

	var maxStamp uint32

	// We should replay them in order. Since we grouped them, we need to sort txnIDs by their commit order?
	// Actually, the WAL preserves order. We can just iterate the WAL again, and if the entry belongs to a committed txn, replay it.
	for _, entry := range entries {
		switch entry.Operation {
		case wal.OpEdgeAdd, wal.OpEdgeRemove, wal.OpNodeEdgeDrop:
			if len(entry.Data) < 8 {
				continue
			}
			txnID := binary.LittleEndian.Uint64(entry.Data[0:8])
			if !txnCommitted[txnID] {
				continue
			}

			switch entry.Operation {
			case wal.OpEdgeAdd:
				record, err := DeserializeWALEdgeAddRecord(entry.Data)
				if err != nil {
					log.Printf("WAL replay warning: skipping corrupt OpEdgeAdd record (txn %d): %v", txnID, err)
					continue
				}
				err = forwardIndex.AddEdgeWithStamp(nil, record.From, record.To, record.Weight, record.Kind, record.Stamp)
				if err != nil {
					return err
				}
				if record.Stamp > maxStamp {
					maxStamp = record.Stamp
				}
			case wal.OpEdgeRemove:
				record, err := DeserializeWALEdgeRemoveRecord(entry.Data)
				if err != nil {
					log.Printf("WAL replay warning: skipping corrupt OpEdgeRemove record (txn %d): %v", txnID, err)
					continue
				}
				err = forwardIndex.RemoveEdge(nil, record.From, record.To, record.Kind)
				if err != nil {
					return err
				}
			case wal.OpNodeEdgeDrop:
				record, err := DeserializeWALNodeEdgeDropRecord(entry.Data)
				if err != nil {
					log.Printf("WAL replay warning: skipping corrupt OpNodeEdgeDrop record (txn %d): %v", txnID, err)
					continue
				}
				err = forwardIndex.DropNodeEdges(nil, record.NodeID)
				if err != nil {
					return err
				}
			}
		}
	}

	// Restore lastFlushedGen
	atomic.StoreUint32(&forwardIndex.lastFlushedGen, maxStamp)

	forwardIndex.metrics.walReplayDuration.Store(time.Since(start).Nanoseconds())

	return nil
}
