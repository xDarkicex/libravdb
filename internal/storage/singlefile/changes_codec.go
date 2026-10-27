package singlefile

import (
	"fmt"
	"math"

	"github.com/xDarkicex/libravdb/internal/util"
)

// Compact references survive physical compaction independently of record history.
func encodeChanges(state *persistedState, enc *util.BinaryEncoder) error {
	if len(state.ChangeCommits) > math.MaxUint32 {
		return fmt.Errorf("too many change transactions")
	}
	enc.WriteUint64(state.ChangeStartLSN)
	enc.WriteUint32(uint32(len(state.ChangeCommits)))
	for _, commit := range state.ChangeCommits {
		if err := encodeChangeCommit(commit, enc); err != nil {
			return err
		}
	}
	return nil
}
func encodeChangeCommit(commit changeCommit, enc *util.BinaryEncoder) error {
	if len(commit.Rows.Operations) > math.MaxUint32 {
		return fmt.Errorf("too many transaction changes")
	}
	enc.WriteUint64(commit.LSN)
	enc.WriteBytes(commit.Nonce[:])
	enc.WriteUint32(uint32(len(commit.Rows.Operations)))
	for i := range commit.Rows.Operations {
		encodeChangeReference(enc, commit.Rows.at(uint32(i)))
	}

	return nil
}
func decodeChanges(state *persistedState, dec *util.BinaryDecoder) error {
	start, err := dec.ReadUint64()
	if err != nil {
		return err
	}
	count, err := dec.ReadUint32()
	if err != nil {
		return err
	}
	if uint64(count) > uint64(len(dec.Data)-dec.Off)/32 {
		return fmt.Errorf("invalid change transaction count")
	}
	state.ChangeStartLSN = start
	state.ChangeFeedVersion = 1
	for i := uint32(0); i < count; i++ {
		commit, err := decodeChangeCommit(dec)
		if err != nil {
			return err
		}
		if commit.LSN <= start {
			return fmt.Errorf("unordered change transaction")
		}
		commit.BeforeLSN = start
		state.ChangeCommits = append(state.ChangeCommits, commit)
		start = commit.LSN
	}
	return nil
}
func decodeChangeCommit(dec *util.BinaryDecoder) (changeCommit, error) {
	var commit changeCommit
	var err error
	commit.LSN, err = dec.ReadUint64()
	if err != nil {
		return commit, err
	}

	nonce, err := dec.ReadBytes()
	if err != nil {
		return commit, err
	}
	if len(nonce) != 16 {
		return commit, fmt.Errorf("invalid change nonce")
	}
	copy(commit.Nonce[:], nonce)
	err = decodeChangeColumns(dec, &commit.Rows)
	return commit, err
}
func decodeChangeColumns(dec *util.BinaryDecoder, rows *changeColumns) error {
	count, err := dec.ReadUint32()
	if err != nil {
		return err
	}
	if uint64(count) > uint64(len(dec.Data)-dec.Off)/18 {
		return fmt.Errorf("invalid mutation count")
	}
	for i := uint32(0); i < count; i++ {
		m, err := decodeChangeReference(dec)
		if err != nil {
			return err
		}
		rows.append(m)
	}
	return nil
}

var changeOperations = [...]string{"record_put", "record_delete", "edge_add", "edge_remove", "node_drop", "vertex_label", "collection_create", "collection_delete", "collection_config", "collection_stats", "edge_kind"}

func encodeChangeReference(enc *util.BinaryEncoder, m changeMutation) {
	enc.WriteUint64(m.LSN)
	enc.WriteByte(changeOperationCode(m.Operation))
	enc.WriteString(m.Collection)
	enc.WriteString(m.ID)
	var flags byte
	if m.Source != 0 {
		flags |= 1
	}
	if m.Target != 0 {
		flags |= 2
	}
	if m.Kind != 0 {
		flags |= 4
	}
	enc.WriteByte(flags)
	if flags&1 != 0 {
		enc.WriteUint64(m.Source)
	}
	if flags&2 != 0 {
		enc.WriteUint64(m.Target)
	}
	if flags&4 != 0 {
		enc.WriteByte(m.Kind)
	}
}
func changeOperationCode(operation string) byte {
	for i, name := range changeOperations {
		if operation == name {
			return byte(i)
		}
	}
	return 255
}
func decodeChangeReference(dec *util.BinaryDecoder) (changeMutation, error) {
	var m changeMutation
	var err error
	m.LSN, err = dec.ReadUint64()
	if err != nil {
		return m, err
	}
	code, err := dec.ReadByte()
	if err != nil {
		return m, err
	}
	if int(code) >= len(changeOperations) {
		return m, fmt.Errorf("invalid change operation")
	}
	m.Operation = changeOperations[code]
	m.Collection, err = dec.ReadString()
	if err != nil {
		return m, err
	}
	m.ID, err = dec.ReadString()
	if err != nil {
		return m, err
	}
	err = decodeChangeScalars(dec, &m)
	return m, err
}
func decodeChangeScalars(dec *util.BinaryDecoder, m *changeMutation) error {
	flags, err := dec.ReadByte()
	if err != nil {
		return err
	}
	if flags&^byte(7) != 0 {
		return fmt.Errorf("invalid change flags")
	}
	if flags&1 != 0 {
		m.Source, err = dec.ReadUint64()
		if err != nil {
			return err
		}
	}
	if flags&2 != 0 {
		m.Target, err = dec.ReadUint64()
		if err != nil {
			return err
		}
	}
	if flags&4 != 0 {
		m.Kind, err = dec.ReadByte()
	}
	return err
}
