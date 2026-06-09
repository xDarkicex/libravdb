package singlefile

// DoS limits (not performance budgets).
// These limits provide 10x-1000x headroom over legitimate use cases.
// Exceeding them implies corruption or malicious files.

const (
	// maxChunkSize is the maximum allowed size for a chunk, such as a snapshot block (2 GiB)
	maxChunkSize = 2 * 1024 * 1024 * 1024

	// maxIndexEntrySize is the maximum allowed size for a single index's serialized state (1 GiB)
	maxIndexEntrySize = 1 * 1024 * 1024 * 1024
)
