package graph

import "errors"

var (
	// ErrNodeNotFound indicates that the requested node does not exist in the graph.
	ErrNodeNotFound = errors.New("node not found")

	// ErrEdgeNotFound indicates that the requested edge does not exist between the source and target nodes.
	ErrEdgeNotFound = errors.New("edge not found")

	// ErrInvalidKind indicates that an invalid edge kind was provided.
	ErrInvalidKind = errors.New("invalid edge kind")

	// ErrNoTransaction indicates that a required transaction was not provided.
	ErrNoTransaction = errors.New("transaction required")

	// ErrTransactionAborted indicates that the transaction was aborted, usually due to a hook failure.
	ErrTransactionAborted = errors.New("transaction aborted")

	// ErrConcurrentModification indicates that a lock-free operation failed due to concurrent writes and should be retried.
	// Callers should implement exponential backoff when encountering this error.
	ErrConcurrentModification = errors.New("concurrent modification")
)
