package singlefile

import (
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"syscall"
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

// holdUsersFile takes the store's shared users lock.
//
// Every process that has the store open holds this lock, shared. The
// compaction needs the file to itself -- it replaces the file -- so it tries
// to take the lock exclusively, and skips the compaction when another process
// still has the store open. Without this, one process's compaction renames
// the file out from under the other, and the other's writes land in an
// orphaned inode and are lost.
func (e *Engine) holdUsersFile() error {
	file, err := os.OpenFile(e.usersPath(), os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return fmt.Errorf("open users lock file: %w", err)
	}
	if err := syscall.Flock(int(file.Fd()), syscall.LOCK_SH); err != nil {
		_ = file.Close()
		return fmt.Errorf("lock users file: %w", err)
	}
	e.usersFile = file
	return nil
}

// usersPath is the path of the sidecar file that records which processes have
// the store open.
func (e *Engine) usersPath() string {
	return e.path + ".users"
}

// releaseUsersFile gives the users lock back.
func (e *Engine) releaseUsersFile() {
	if e.usersFile == nil {
		return
	}
	_ = syscall.Flock(int(e.usersFile.Fd()), syscall.LOCK_UN)
	_ = e.usersFile.Close()
	e.usersFile = nil
}

// compactionAllowed reports whether the store may be compacted.
//
// The compaction replaces the file. That is only safe when no other process
// has the store open: the lock is shared by everyone who has it, and the
// compaction takes it exclusively for the length of the replacement.
func (e *Engine) compactionAllowed() bool {
	if e.usersFile == nil {
		return true
	}
	if err := syscall.Flock(int(e.usersFile.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		// Somebody else has the store open.
		return false
	}
	// Put the shared lock back for the rest of this instance's life.
	_ = syscall.Flock(int(e.usersFile.Fd()), syscall.LOCK_SH)
	return true
}

// lockFile takes the store's exclusive commit lock.
//
// The lock is on the file itself, so it coordinates with other processes. It
// is advisory: a reader never takes it, so a read keeps progressing while a
// write is in flight.
func (e *Engine) lockFile() error {
	if err := syscall.Flock(int(e.file.Fd()), syscall.LOCK_EX); err != nil {
		return fmt.Errorf("lock store file: %w", err)
	}
	return nil
}

// unlockFile releases the store's commit lock.
func (e *Engine) unlockFile() {
	_ = syscall.Flock(int(e.file.Fd()), syscall.LOCK_UN)
}

// beginCommit takes the commit lock and brings this instance's view of the
// file up to date.
//
// It must be paired with endCommit. The caller holds e.mu.
func (e *Engine) beginCommit() error {
	if e.commitDepth == 0 {
		if err := e.lockFile(); err != nil {
			return err
		}
		if err := e.refreshFromDiskLocked(); err != nil {
			e.unlockFile()
			return err
		}
	}
	e.commitDepth++
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
// the refresh only runs when the file has actually grown.
func (e *Engine) refreshIfBehind() error {
	stat, err := e.file.Stat()
	if err != nil {
		return fmt.Errorf("stat store file: %w", err)
	}
	e.mu.RLock()
	behind := stat.Size() > e.walAppliedOffset
	e.mu.RUnlock()
	if !behind {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.refreshFromDiskLocked()
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
	if fileSize <= offset {
		return nil
	}
	pending := make(map[uint64][]walRecord)
	touched := make(map[string]struct{})
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
		if record.Header.LSN <= e.lastLSN.Load() {
			// Already applied.
			offset = chunkEnd
			continue
		}
		switch record.Header.RecordType {
		case recordTypeTxBegin:
			pending[record.Header.TxID] = []walRecord{record}
		case recordTypeTxCommit:
			frames := append(pending[record.Header.TxID], record)
			if err := e.applyCommittedFrames(frames, touched, deltaProvider); err != nil {
				return err
			}
			delete(pending, record.Header.TxID)
		default:
			pending[record.Header.TxID] = append(pending[record.Header.TxID], record)
		}
		// The counters have to follow the file. A stale one hands out an LSN
		// another process has already taken, and two commits at the same LSN
		// leave the catalog ambiguous and the recovery unable to say which
		// of them a snapshot contains.
		if record.Header.LSN > e.lastLSN.Load() {
			e.lastLSN.Store(record.Header.LSN)
		}
		if record.Header.TxID > e.lastTxID.Load() {
			e.lastTxID.Store(record.Header.TxID)
		}
		offset = chunkEnd
	}
	e.walAppliedOffset = offset
	return nil
}
