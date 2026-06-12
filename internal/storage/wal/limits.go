package wal

// DoS limits (not performance budgets).
// These limits provide 10x-1000x headroom over legitimate use cases.
// Exceeding them implies corruption or malicious files.

const (
	// maxWALEntrySize is the maximum allowed size for a single WAL entry (64 MiB)
	maxWALEntrySize = 64 * 1024 * 1024

	// maxWALMetadataSize is the maximum allowed size for WAL metadata JSON (16 MiB)
	maxWALMetadataSize = 16 * 1024 * 1024
)
