package libravdb

// MetadataLookupStats is process-local diagnostic data for equality metadata
// lookups. It describes the route selected by ListByMetadata and the amount of
// work performed by the derived posting-list path; it is not persisted and has
// no effect on query results.
type MetadataLookupStats struct {
	IndexedLookups       uint64
	FullScanFallbacks    uint64
	IndexRebuilds        uint64
	IndexRebuildRecords  uint64
	CandidateRecordsRead uint64
}

// MetadataLookupStats returns a consistent snapshot of metadata lookup
// counters for this collection.
func (c *Collection) MetadataLookupStats() MetadataLookupStats {
	if c == nil {
		return MetadataLookupStats{}
	}
	return MetadataLookupStats{
		IndexedLookups:       c.metadataLookupIndexed.Load(),
		FullScanFallbacks:    c.metadataLookupFallback.Load(),
		IndexRebuilds:        c.metadataIndexRebuilds.Load(),
		IndexRebuildRecords:  c.metadataIndexRecords.Load(),
		CandidateRecordsRead: c.metadataLookupCandidates.Load(),
	}
}

// ResetMetadataLookupStats clears process-local metadata lookup counters. It
// does not clear or rebuild the actual metadata index.
func (c *Collection) ResetMetadataLookupStats() {
	if c == nil {
		return
	}
	c.metadataLookupIndexed.Store(0)
	c.metadataLookupFallback.Store(0)
	c.metadataIndexRebuilds.Store(0)
	c.metadataIndexRecords.Store(0)
	c.metadataLookupCandidates.Store(0)
}
