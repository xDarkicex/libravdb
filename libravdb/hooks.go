package libravdb

// GraphTx defines the methods available to hooks during a transaction.
type GraphTx interface {
	AddEdge(src, tgt uint64, weight float32, kind uint8) error
	RemoveEdge(src, tgt uint64, kind uint8) error
}

// InsertHook is a callback invoked before a vector insertion is committed to the WAL.
type InsertHook func(txn GraphTx, id uint64, vector []float32, metadata map[string]interface{}) error

// DeleteHook is a callback invoked before a vector deletion is committed to the WAL.
type DeleteHook func(txn GraphTx, id uint64) error
