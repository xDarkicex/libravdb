package singlefile

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"log"
	"os"
	"syscall"

	"github.com/xDarkicex/libravdb/internal/index"
)

// The store file is shared. Two processes may hold it open and write to it,
// and the engine has to make that safe rather than merely tolerate it.
//
// Three things make it safe, and each is small:
//
//   - A transaction's frames are written as one write() at the end of the
//     file. The commit lock means only one process is at the end of the file
//     at a time, so two transactions cannot interleave.
//
//   - The commit takes an exclusive lock on the file. It covers the LSN
//     allocation and the append that go with it, not the transaction's work:
//     the frames are built inside it, but the index, the vectors and the
//     caller's own work are not. A second writer waits here and then
//     proceeds. It is not refused.
//
//   - Before allocating, the writer brings its own view of the file up to
//     date. A process that has been open since before another writer
//     committed has a catalog, an index and a graph that are behind the file,
//     and an LSN counter that would hand out an LSN another process has
//     already taken.
//
// Without all three, two processes writing the same store lose data: their
// frames interleave at the same file offset, and one process's compaction
// renames the file out from under the other.

// lockFile locks the database inode, then verifies that it is still the inode
// named by the database path. A waiter on an old inode retries on the replacement.
func (e *Engine) lockFile() error {
	for {
		if err := syscall.Flock(int(e.file.Fd()), syscall.LOCK_EX); err != nil {
			return fmt.Errorf("lock store file: %w", err)
		}
		held, err := e.file.Stat()
		if err != nil {
			e.unlockFile()
			return err
		}
		current, err := os.Stat(e.path)
		if err != nil {
			e.unlockFile()
			return err
		}
		if os.SameFile(held, current) {
			return nil
		}
		e.unlockFile()
		replacement, err := os.OpenFile(e.path, os.O_RDWR|oNoFollow, 0)
		if err != nil {
			return err
		}
		old := e.file
		e.file = replacement
		_ = old.Close()
		e.fileChanged = true
	}
}

func (e *Engine) unlockFile() { _ = syscall.Flock(int(e.file.Fd()), syscall.LOCK_UN) }

// replaceLocked hands the commit lock from the original inode to the new inode.
// Both are locked across rename. Waiters must verify path identity after flock.
func (e *Engine) replaceLocked(path string, replacement *os.File) error {
	if err := syscall.Flock(int(replacement.Fd()), syscall.LOCK_EX); err != nil {
		return err
	}
	if err := replaceDatabaseFile(path, e.path); err != nil {
		_ = syscall.Flock(int(replacement.Fd()), syscall.LOCK_UN)
		return err
	}
	old := e.file
	e.file = replacement
	_ = old.Close()
	return nil
}

func (e *Engine) reloadReplacementLocked() error {
	old := e.state
	pendingCatalog, catalogDirty := e.catalogData, e.catalogDirty
	// Live recovery must not rebuild through a provider callback that re-enters
	// engine.mu. Keep the live indexes and advance them from the snapshot diff.
	provider := e.indexProvider
	e.indexProvider = nil
	err := e.openExisting()
	e.indexProvider = provider
	if catalogDirty {
		e.catalogData, e.catalogDirty = pendingCatalog, true
		e.dirty = true
	}
	if err != nil {
		e.disableWrites()
		return err
	}
	if err := e.refreshReplacementIndexesLocked(old); err != nil {
		e.disableWrites()
		return err
	}
	e.initReservedOrdinals()
	for name, col := range old.Collections {
		if next := col.reservedNextOrdinal.Load(); next > 0 {
			advanceReservedOrdinal(e.state.Collections[name], next-1)
		}
		col.freeVectorSFLs()
	}
	e.refreshPending = nil
	e.fileChanged = false
	return nil
}

// refreshReplacementIndexesLocked reconciles live indexes with records folded
// into the replacement snapshot, which are no longer present as record WAL.
func (e *Engine) refreshReplacementIndexesLocked(old *persistedState) error {
	if e.indexProvider == nil {
		return nil
	}
	delta, ok := e.indexProvider.(IncrementalIndexRecoveryProvider)
	canApply := func(name string, col *persistedCollection) bool {
		if !ok {
			return false
		}
		if live, supported := e.indexProvider.(interface{ CanApplyLiveIndexDeltas(string) bool }); supported {
			return live.CanApplyLiveIndexDeltas(name)
		}
		return delta.CanApplyIndexDeltas(name, &col.Config)
	}
	for name, current := range e.state.Collections {
		previous := old.Collections[name]
		if current.Deleted {
			continue
		}
		for id, record := range current.Records {
			var before *recordValue
			if previous != nil {
				before = previous.Records[id]
			}
			if before != nil && before.UpdatedLSN == record.UpdatedLSN && before.Deleted == record.Deleted && before.Ordinal == record.Ordinal {
				continue
			}
			if !canApply(name, current) {
				return fmt.Errorf("index %s requires reopen after database replacement", name)
			}
			if record.Deleted {
				if before != nil && !before.Deleted {
					if err := delta.ApplyIndexDelete(name, id, before.Ordinal, &current.Config); err != nil {
						return err
					}
				}
			} else {
				replace := before != nil && !before.Deleted
				ordinal := record.Ordinal
				if replace {
					ordinal = before.Ordinal
				}
				entry := &index.VectorEntry{ID: id, Vector: record.Vector, Metadata: record.Metadata, Version: record.Version, Ordinal: record.Ordinal}
				if err := delta.ApplyIndexPut(name, entry, replace, ordinal, &current.Config); err != nil {
					return err
				}
			}
		}
	}
	for name, previous := range old.Collections {
		current := e.state.Collections[name]
		if current == nil || current.Deleted {
			if ok {
				delta.DiscardIndex(name)
			}
			continue
		}
		for id, before := range previous.Records {
			if _, exists := current.Records[id]; exists || before.Deleted {
				continue
			}
			if !canApply(name, current) {
				return fmt.Errorf("index %s requires reopen after database replacement", name)
			}
			if err := delta.ApplyIndexDelete(name, id, before.Ordinal, &current.Config); err != nil {
				return err
			}
		}
	}
	return nil
}

// beginCommit takes the commit lock and brings this instance's view of the
// file up to date.
//
// It must be paired with endCommit. The caller holds e.mu.
func (e *Engine) beginCommit() error {
	e.commitDepth++
	if e.commitDepth != 1 {
		// An outer section already holds the lock, and it already refreshed:
		// no other process can have written while it holds it.
		return nil
	}
	if err := e.lockFile(); err != nil {
		e.commitDepth--
		return err
	}
	if e.fileChanged {
		if err := e.reloadReplacementLocked(); err != nil {
			e.unlockFile()
			e.commitDepth--
			return err
		}
	}
	if err := e.refreshFromDiskLocked(); err != nil {
		e.unlockFile()
		e.commitDepth--
		return err
	}
	return nil
}

// endCommit releases the commit lock.
//
// The commit sections nest: a transaction can reach the batch writer or the
// graph flush, and both of those append. The depth counter keeps the inner
// section from releasing the lock the outer one holds.
func (e *Engine) endCommit() {
	e.commitDepth--
	if e.commitDepth == 0 {
		e.unlockFile()
	}
}

// refreshIfBehind brings the instance up to date if another process has
// written to the file.
//
// It is the read path's entry point. The check is a stat, which is cheap;
// the refresh runs when size or inode identity differs.
func (e *Engine) refreshIfBehind() error {
	e.mu.RLock()
	stat, err := e.file.Stat()
	behind := err == nil && stat.Size() != e.walAppliedOffset
	if err == nil {
		current, pathErr := os.Stat(e.path)
		if pathErr != nil {
			err = pathErr
		} else {
			behind = behind || !os.SameFile(stat, current)
		}
	}
	e.mu.RUnlock()
	if err != nil {
		return fmt.Errorf("stat store file: %w", err)
	}
	if !behind {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := e.beginCommit(); err != nil {
		return err
	}
	e.endCommit()
	return nil
}

// refreshFromDiskLocked applies the commits another process made.
//
// It scans the part of the file this instance has not applied, and brings
// the instance's records, index, graph and commit catalog up to date. The
// caller holds e.mu.
//
// The scan is cheap when nothing else has written: it stops at the end of
// the file it already knows. A chunk that is half-written — another process
// mid-append — fails its checksum and is left for the next refresh.
func (e *Engine) refreshFromDiskLocked() error {
	stat, err := e.file.Stat()
	if err != nil {
		return fmt.Errorf("stat store file: %w", err)
	}
	fileSize := stat.Size()
	offset := e.walAppliedOffset
	if fileSize < offset {
		return fmt.Errorf("store file shrank while open: size %d, applied offset %d", fileSize, offset)
	}
	if fileSize == 0 {
		return nil
	}
	header, err := e.readHeader()
	if err != nil {
		return err
	}
	meta, err := e.readMetaPage(header.ActiveMetaPage)
	if err != nil {
		return err
	}
	e.activeMetaPage = header.ActiveMetaPage
	if meta.MetaEpoch > e.metaEpoch {
		e.metaEpoch = meta.MetaEpoch
	}
	if fileSize == offset {
		return nil
	}
	if e.refreshPending == nil {
		e.refreshPending = make(map[uint64][]walRecord)
	}
	pending := e.refreshPending
	touched := make(map[string]struct{})
	refreshedFrames := 0
	deltaProvider, _ := e.indexProvider.(IncrementalIndexRecoveryProvider)

	for fileSize >= 16 && offset >= 0 && offset <= fileSize-16 {
		headerBuf := make([]byte, 16)
		if _, err := e.file.ReadAt(headerBuf, offset); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return err
		}
		chunk := decodeChunkHeader(headerBuf)
		if chunk.Magic != chunkMagic {
			break
		}
		chunkEnd := offset + 16 + int64(chunk.PayloadLen)
		if chunkEnd < offset || chunkEnd > fileSize {
			// A partially written chunk is another process mid-append. It
			// will be complete on the next refresh.
			break
		}
		if chunk.Kind != chunkTypeWAL {
			offset = chunkEnd
			continue
		}
		payload := make([]byte, chunk.PayloadLen)
		if _, err := e.file.ReadAt(payload, offset+16); err != nil {
			return err
		}
		if crc32.Checksum(payload, castagnoli) != chunk.Checksum {
			// A torn write. The frames become visible on the next refresh.
			break
		}
		record, err := decodeWALRecord(payload)
		if err != nil {
			return err
		}
		switch record.Header.RecordType {
		case recordTypeTxBegin:
			pending[record.Header.TxID] = []walRecord{record}
		case recordTypeTxAbort:
			delete(pending, record.Header.TxID)
		case recordTypeTxCommit:
			if record.Header.LSN <= e.lastLSN.Load() {
				delete(pending, record.Header.TxID)
				break
			}
			if len(pending[record.Header.TxID]) == 0 || pending[record.Header.TxID][0].Header.RecordType != recordTypeTxBegin {
				return fmt.Errorf("commit for transaction %d has no begin frame", record.Header.TxID)
			}
			frames := append(pending[record.Header.TxID], record)
			if err := e.applyCommittedFrames(frames, touched, deltaProvider); err != nil {
				return err
			}
			if payload, ok := decodeTxCommitPayload(record.Payload); ok {
				e.recordCommitLocked(record.Header.LSN, payload.Timestamp)
			}
			refreshedFrames += len(frames)
			delete(pending, record.Header.TxID)
			// The counters follow the file, but only as far as a
			// *committed* transaction. Advancing them on a frame whose
			// transaction has not been seen whole would put the next
			// transaction's LSN below the counter, and the frames of a
			// half-read transaction would then be skipped as though they
			// had already been applied.
			if record.Header.LSN > e.lastLSN.Load() {
				e.lastLSN.Store(record.Header.LSN)
			}
			if record.Header.TxID > e.lastTxID.Load() {
				e.lastTxID.Store(record.Header.TxID)
			}
		default:
			pending[record.Header.TxID] = append(pending[record.Header.TxID], record)
		}
		offset = chunkEnd
	}
	e.walAppliedOffset = offset
	if refreshedFrames > 0 && traceEnabled() {
		log.Printf("singlefile: refresh applied %d frames (tx %d)", refreshedFrames, e.lastLSN.Load())
	}
	return nil
}

// persistOrdinalReservationsLocked publishes reservation frontiers as one WAL
// transaction, amortizing durability across a batch. Caller holds e.mu and the
// commit lock. Aborted writes may leave reserved gaps.
func (e *Engine) persistOrdinalReservationsLocked(frontiers map[string]uint32) error {
	if len(frontiers) == 0 {
		return nil
	}
	txID := e.nextTxID()
	begin := e.nextLSN()
	frames := []walRecord{newFrame(recordTypeTxBegin, begin, txID, 0, emptyPayload())}
	prev := begin
	for name, next := range frontiers {
		payload := make([]byte, 4+len(name))
		binary.LittleEndian.PutUint32(payload, next)
		copy(payload[4:], name)
		lsn := e.nextLSN()
		frames = append(frames, newFrame(recordTypeOrdinalReserve, lsn, txID, prev, encodedPayload{bytes: payload}))
		prev = lsn
	}
	commit := e.nextLSN()
	frames = append(frames, e.makeTxCommitFrame(commit, txID, prev))
	written, err := e.appendTransactionLocked(frames)
	if err != nil {
		if isAmbiguousWriteError(err) {
			e.disableWrites()
		}
		return err
	}
	if err := e.syncWALLocked(); err != nil {
		e.disableWrites()
		return err
	}
	for name, next := range frontiers {
		if col := e.state.Collections[name]; next > col.NextOrdinal {
			col.NextOrdinal = next
		}
	}
	e.recordPendingCommitLocked()
	e.markDirtyLocked(written, 1)
	return nil
}
