package storage

import "errors"

// ErrChangeCursor indicates a malformed cursor, a different database/history,
// or a position outside the retained feed. Consumers must obtain a new baseline.
var ErrChangeCursor = errors.New("incompatible change cursor")

// ChangeRows is a columnar, owned page of compact WAL mutation references.
// Put is an upsert, not necessarily a creation. MutationLSNs identify exact WAL
// versions; graph endpoints are database-scoped graph node IDs. Empty columns
// contain zero values for operations to which they do not apply.
type ChangeRows struct {
	BeforeLSNs       []uint64 `json:"before_lsns"`
	CommitLSNs       []uint64 `json:"commit_lsns"`
	MutationLSNs     []uint64 `json:"mutation_lsns"`
	Positions        []uint32 `json:"positions"`
	TransactionSizes []uint32 `json:"transaction_sizes"`
	Operations       []string `json:"operations"`
	Collections      []string `json:"collections"`
	RecordIDs        []string `json:"record_ids"`
	Sources          []uint64 `json:"sources"`
	Targets          []uint64 `json:"targets"`
	EdgeKinds        []uint8  `json:"edge_kinds"`
}

// ChangePage covers the committed feed prefix through Next. Watermark is an
// exclusive upper checkpoint captured under the same engine lock. More signals
// remaining rows; TransactionComplete says Next ends at a transaction boundary.
// StartLSN is the baseline before which mutation enumeration is unavailable.
type ChangePage struct {
	Rows                ChangeRows `json:"rows"`
	Next                string     `json:"next"`
	Watermark           string     `json:"watermark"`
	StartLSN            uint64     `json:"start_lsn"`
	More                bool       `json:"more"`
	TransactionComplete bool       `json:"transaction_complete"`
}
