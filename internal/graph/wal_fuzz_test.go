package graph

import (
	"testing"
)

func FuzzDeserializeWALEdgeAddRecord(f *testing.F) {
	// Add valid seed corpus
	validRec := &WALEdgeAddRecord{
		TxnID:  123,
		From:   456,
		To:     789,
		Weight: 1.5,
		Kind:   2,
		Stamp:  10,
	}
	f.Add(SerializeWALEdgeAddRecord(validRec))
	f.Add([]byte("too short"))
	f.Add([]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})

	f.Fuzz(func(t *testing.T, data []byte) {
		// Just ensure it doesn't panic
		_, _ = DeserializeWALEdgeAddRecord(data)
	})
}

func FuzzDeserializeWALEdgeRemoveRecord(f *testing.F) {
	validRec := &WALEdgeRemoveRecord{
		TxnID: 123,
		From:  456,
		To:    789,
		Kind:  2,
	}
	f.Add(SerializeWALEdgeRemoveRecord(validRec))

	f.Fuzz(func(t *testing.T, data []byte) {
		_, _ = DeserializeWALEdgeRemoveRecord(data)
	})
}

func FuzzDeserializeWALNodeEdgeDropRecord(f *testing.F) {
	validRec := &WALNodeEdgeDropRecord{
		TxnID:  123,
		NodeID: 456,
	}
	f.Add(SerializeWALNodeEdgeDropRecord(validRec))

	f.Fuzz(func(t *testing.T, data []byte) {
		_, _ = DeserializeWALNodeEdgeDropRecord(data)
	})
}

func FuzzDeserializeWALTxnCommitRecord(f *testing.F) {
	validRec := &WALTxnCommitRecord{
		TxnID: 123,
	}
	f.Add(SerializeWALTxnCommitRecord(validRec))

	f.Fuzz(func(t *testing.T, data []byte) {
		_, _ = DeserializeWALTxnCommitRecord(data)
	})
}
