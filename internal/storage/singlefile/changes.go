package singlefile

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"sort"

	"github.com/xDarkicex/libravdb/internal/storage"
)

type changeMutation struct {
	LSN        uint64
	Source     uint64
	Target     uint64
	Collection string
	ID         string
	Operation  string
	Kind       uint8
}

type changeCommit struct {
	BeforeLSN uint64
	Timestamp int64
	LSN       uint64
	Nonce     [16]byte
	Rows      changeColumns
}

// prepareChanges runs before append releases pooled WAL payloads. Only small
// identifiers are retained; vectors and metadata never enter the feed.
func prepareChanges(frames []walRecord) (changeCommit, error) {
	var result changeCommit
	for _, frame := range frames {
		if frame.Header.RecordType == recordTypeTxCommit {
			result.LSN = frame.Header.LSN
			payload, ok := decodeTxCommitPayload(frame.Payload)
			if !ok && len(frame.Payload) != 0 {
				return result, fmt.Errorf("invalid change commit payload")
			}
			result.Nonce = payload.Nonce
			result.Timestamp = payload.Timestamp
			continue
		}
		mutation, err := decodeChangeMutation(frame)
		if err != nil {
			return result, err
		}
		if mutation.Operation != "" {
			if result.Rows.Operations == nil {
				result.Rows = allocateChangeColumns(max(len(frames)-2, 0))
			}
			result.Rows.append(mutation)
		}
	}
	return result, nil
}

func (e *Engine) stageChanges(frames []walRecord) error {
	commit, err := prepareChanges(frames)
	if err != nil {
		return err
	}
	if commit.LSN != 0 {
		e.pendingChanges = append(e.pendingChanges, commit)
	}
	return nil
}

func (e *Engine) publishChanges(commit changeCommit) {
	if commit.LSN <= e.state.ChangeStartLSN {
		return
	}
	commits := e.state.ChangeCommits
	commit.BeforeLSN = e.state.ChangeStartLSN
	if len(commits) > 0 {
		commit.BeforeLSN = commits[len(commits)-1].LSN
	}
	pos := sort.Search(len(commits), func(i int) bool { return commits[i].LSN >= commit.LSN })
	if pos < len(commits) && commits[pos].LSN == commit.LSN {
		return
	}
	commits = append(commits, changeCommit{})
	copy(commits[pos+1:], commits[pos:])
	commits[pos] = commit
	e.state.ChangeCommits = commits
}

func (e *Engine) replayChanges(frames []walRecord) error {
	commit, err := prepareChanges(frames)
	if err != nil {
		return err
	}
	if commit.Nonce == [16]byte{} {
		if commit.LSN > e.state.ChangeStartLSN {
			e.state.ChangeStartLSN = commit.LSN
			e.state.ChangeCommits = nil
		}
		return nil
	}
	e.publishChanges(commit)
	return nil
}

func changeNonce() [16]byte {
	// rand.Read is guaranteed to fill the buffer or terminate the process.
	var nonce [16]byte
	_, _ = rand.Read(nonce[:])
	return nonce
}

type changePosition struct {
	File   uint64
	Start  uint64
	LSN    uint64
	Offset uint32
	Nonce  [16]byte
}

func encodeChangePosition(p changePosition) string {
	var buf [45]byte
	buf[0] = 1
	binary.LittleEndian.PutUint64(buf[1:9], p.File)
	binary.LittleEndian.PutUint64(buf[9:17], p.Start)
	binary.LittleEndian.PutUint64(buf[17:25], p.LSN)
	binary.LittleEndian.PutUint32(buf[25:29], p.Offset)
	copy(buf[29:], p.Nonce[:])
	return base64.RawURLEncoding.EncodeToString(buf[:])
}
func decodeChangePosition(encoded string) (changePosition, error) {
	var p changePosition
	buf, err := base64.RawURLEncoding.DecodeString(encoded)
	if err != nil || len(buf) != 45 || buf[0] != 1 {
		return p, storage.ErrChangeCursor
	}
	p.File = binary.LittleEndian.Uint64(buf[1:9])
	p.Start = binary.LittleEndian.Uint64(buf[9:17])
	p.LSN = binary.LittleEndian.Uint64(buf[17:25])
	p.Offset = binary.LittleEndian.Uint32(buf[25:29])
	copy(p.Nonce[:], buf[29:])
	return p, nil
}
func (e *Engine) changePosition(index int, offset uint32) changePosition {
	p := changePosition{File: e.fileID, Start: e.state.ChangeStartLSN, LSN: e.state.ChangeStartLSN}
	if index >= 0 {
		commit := e.state.ChangeCommits[index]
		p.LSN, p.Nonce, p.Offset = commit.LSN, commit.Nonce, offset
	}
	return p
}
func (e *Engine) validateChangePosition(p changePosition) (int, error) {
	if p.File != e.fileID || p.Start != e.state.ChangeStartLSN {
		return 0, storage.ErrChangeCursor
	}
	if isChangeStart(p) {
		return 0, nil
	}
	commits := e.state.ChangeCommits
	index := sort.Search(len(commits), func(i int) bool { return commits[i].LSN >= p.LSN })
	if index == len(commits) || commits[index].LSN != p.LSN {
		return 0, storage.ErrChangeCursor
	}
	commit := commits[index]
	if !validChangeOffset(commit, p) {
		return 0, storage.ErrChangeCursor
	}
	if int(p.Offset) == len(commit.Rows.Operations) {
		return index + 1, nil
	}
	return index, nil
}

// ChangeCheckpoint returns the position after all currently committed mutations.
func (e *Engine) ChangeCheckpoint(ctx context.Context) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if err := e.refreshIfBehind(); err != nil {
		return "", err
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed.Load() {
		return "", fmt.Errorf("database is closed")
	}
	return encodeChangePosition(e.changeTail()), nil
}
func (e *Engine) changeTail() changePosition {
	index := len(e.state.ChangeCommits) - 1
	var offset uint32
	if index >= 0 {
		offset = uint32(len(e.state.ChangeCommits[index].Rows.Operations))
	}
	return e.changePosition(index, offset)
}

// ReadChanges returns at most limit mutation references. The empty cursor starts
// at StartLSN, which may be later than database creation after an upgrade.
func (e *Engine) ReadChanges(ctx context.Context, after string, limit int) (storage.ChangePage, error) {
	var page storage.ChangePage
	if err := ctx.Err(); err != nil {
		return page, err
	}
	if limit < 1 || limit > 10000 {
		return page, fmt.Errorf("change limit must be 1..10000")
	}
	if err := e.refreshIfBehind(); err != nil {
		return page, err
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.closed.Load() {
		return page, fmt.Errorf("database is closed")
	}
	position := e.changePosition(-1, 0)
	var err error
	if after != "" {
		position, err = decodeChangePosition(after)
	}
	if err != nil {
		return page, err
	}
	index, err := e.validateChangePosition(position)
	if err != nil {
		return page, err
	}
	return e.changePage(ctx, position, index, limit)
}
func (e *Engine) changePage(ctx context.Context, position changePosition, index, limit int) (storage.ChangePage, error) {
	page := storage.ChangePage{TransactionComplete: true, StartLSN: e.state.ChangeStartLSN, Watermark: encodeChangePosition(e.changeTail())}
	offset := position.Offset
	if index < len(e.state.ChangeCommits) && e.state.ChangeCommits[index].LSN != position.LSN {
		offset = 0
	}
	for scanned := 0; index < len(e.state.ChangeCommits) && scanned < limit; scanned++ {
		if err := ctx.Err(); err != nil {
			return storage.ChangePage{}, err
		}
		commit := e.state.ChangeCommits[index]
		offset = appendChangeWindow(&page.Rows, commit, offset, limit)
		position = e.changePosition(index, offset)
		if int(offset) < len(commit.Rows.Operations) {
			page.TransactionComplete = false
			break
		}
		index++
		offset = 0
		if len(page.Rows.Operations) == limit {
			break
		}
	}
	page.Next = encodeChangePosition(position)
	page.More = page.Next != page.Watermark
	return page, nil
}
func appendChangeWindow(rows *storage.ChangeRows, commit changeCommit, offset uint32, limit int) uint32 {
	if rows.Operations == nil && int(offset) < len(commit.Rows.Operations) {
		*rows = allocateChangeRows(min(limit, 256))
	}
	for int(offset) < len(commit.Rows.Operations) && len(rows.Operations) < limit {
		appendChange(rows, commit, offset)
		offset++
	}
	return offset
}

func validChangeOffset(commit changeCommit, p changePosition) bool {
	return commit.Nonce == p.Nonce && uint64(p.Offset) <= uint64(len(commit.Rows.Operations))
}

func isChangeStart(p changePosition) bool {
	return p.LSN == p.Start && p.Offset == 0 && p.Nonce == [16]byte{}
}
