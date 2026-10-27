package singlefile

import (
	"fmt"
	"strings"

	"github.com/xDarkicex/libravdb/internal/storage"
	"github.com/xDarkicex/libravdb/internal/util"
)

func decodeChangeMutation(frame walRecord) (changeMutation, error) {
	m := changeMutation{LSN: frame.Header.LSN}
	switch frame.Header.RecordType {
	case recordTypeRecordPut, recordTypeRecordDelete:
		return decodeChangeRecord(frame)
	case recordTypeGraphEdgeAdd:
		p, err := decodeGraphEdgeAddPayload(frame.Payload)
		m.Collection, m.Source, m.Target, m.Kind, m.Operation = strings.Clone(p.Collection), p.Src, p.Tgt, p.Kind, "edge_add"
		return m, err
	case recordTypeGraphEdgeRemove:
		p, err := decodeGraphEdgeRemovePayload(frame.Payload)
		m.Collection, m.Source, m.Target, m.Kind, m.Operation = strings.Clone(p.Collection), p.Src, p.Tgt, p.Kind, "edge_remove"
		return m, err
	case recordTypeGraphNodeDrop:
		p, err := decodeGraphNodeDropPayload(frame.Payload)
		m.Collection, m.Source, m.Operation = strings.Clone(p.Collection), p.NodeID, "node_drop"
		return m, err
	case recordTypeGraphVertexLabel:
		p, err := decodeGraphVertexLabelPayload(frame.Payload)
		m.Source, m.ID, m.Operation = p.NodeID, strings.Clone(p.Label), "vertex_label"
		return m, err
	default:
		return decodeChangeCatalog(frame)
	}
}
func decodeChangeRecord(frame walRecord) (changeMutation, error) {
	m := changeMutation{LSN: frame.Header.LSN, Operation: "record_put"}
	dec := &util.BinaryDecoder{Data: frame.Payload}
	version, err := dec.ReadByte()
	if err != nil {
		return m, err
	}
	if version < 1 || version > codecVersion {
		return m, fmt.Errorf("unsupported change payload %d", version)
	}
	collection, err := dec.ReadString()
	if err != nil {
		return m, err
	}
	id, err := dec.ReadString()
	m.Collection, m.ID = strings.Clone(collection), strings.Clone(id)
	if frame.Header.RecordType == recordTypeRecordDelete {
		m.Operation = "record_delete"
	}
	return m, err
}
func decodeChangeCatalog(frame walRecord) (changeMutation, error) {
	m := changeMutation{LSN: frame.Header.LSN}
	switch frame.Header.RecordType {
	case recordTypeCollectionCreate, recordTypeCollectionConfig:
		p, err := decodeCollectionCreatePayloadBinary(frame.Payload)
		m.Collection, m.Operation = strings.Clone(p.Name), "collection_create"
		if frame.Header.RecordType == recordTypeCollectionConfig {
			m.Operation = "collection_config"
		}
		return m, err
	case recordTypeCollectionDelete:
		p, err := decodeCollectionDeletePayloadBinary(frame.Payload)
		m.Collection, m.Operation = strings.Clone(p.Name), "collection_delete"
		return m, err
	case recordTypeCollectionStats:
		p, err := decodeCollectionStatsPayloadBinary(frame.Payload)
		m.Collection, m.Operation = strings.Clone(p.Name), "collection_stats"
		return m, err
	case recordTypeEdgeKindCreate:
		p, err := decodeEdgeKindCreatePayload(frame.Payload)
		m.ID, m.Kind, m.Operation = strings.Clone(p.Name), p.Kind, "edge_kind"
		return m, err
	default:
		return m, nil
	}
}
func appendChange(rows *storage.ChangeRows, commit changeCommit, offset uint32) {
	m := commit.Rows.at(offset)
	rows.BeforeLSNs = append(rows.BeforeLSNs, commit.BeforeLSN)
	rows.CommitLSNs = append(rows.CommitLSNs, commit.LSN)
	rows.MutationLSNs = append(rows.MutationLSNs, m.LSN)
	rows.Positions = append(rows.Positions, offset)
	rows.TransactionSizes = append(rows.TransactionSizes, uint32(len(commit.Rows.Operations)))
	rows.Operations = append(rows.Operations, m.Operation)
	rows.Collections = append(rows.Collections, m.Collection)
	rows.RecordIDs = append(rows.RecordIDs, m.ID)
	rows.Sources = append(rows.Sources, m.Source)
	rows.Targets = append(rows.Targets, m.Target)
	rows.EdgeKinds = append(rows.EdgeKinds, m.Kind)
}

// Only references are retained; hot scalar columns stay separate from strings.
type changeColumns struct {
	LSNs, Sources, Targets       []uint64
	Collections, IDs, Operations []string
	Kinds                        []uint8
}

func (rows *changeColumns) append(m changeMutation) {
	rows.LSNs = append(rows.LSNs, m.LSN)
	rows.Sources = append(rows.Sources, m.Source)
	rows.Targets = append(rows.Targets, m.Target)
	rows.Collections = append(rows.Collections, m.Collection)
	rows.IDs = append(rows.IDs, m.ID)
	rows.Operations = append(rows.Operations, m.Operation)
	rows.Kinds = append(rows.Kinds, m.Kind)
}
func (rows changeColumns) at(i uint32) changeMutation {
	return changeMutation{LSN: rows.LSNs[i], Source: rows.Sources[i], Target: rows.Targets[i], Collection: rows.Collections[i], ID: rows.IDs[i], Operation: rows.Operations[i], Kind: rows.Kinds[i]}
}

func allocateChangeRows(capacity int) storage.ChangeRows {
	return storage.ChangeRows{
		BeforeLSNs: make([]uint64, 0, capacity), CommitLSNs: make([]uint64, 0, capacity), MutationLSNs: make([]uint64, 0, capacity),
		Positions: make([]uint32, 0, capacity), TransactionSizes: make([]uint32, 0, capacity),
		Operations: make([]string, 0, capacity), Collections: make([]string, 0, capacity), RecordIDs: make([]string, 0, capacity),
		Sources: make([]uint64, 0, capacity), Targets: make([]uint64, 0, capacity), EdgeKinds: make([]uint8, 0, capacity),
	}
}
func allocateChangeColumns(capacity int) changeColumns {
	return changeColumns{
		LSNs: make([]uint64, 0, capacity), Sources: make([]uint64, 0, capacity), Targets: make([]uint64, 0, capacity),
		Collections: make([]string, 0, capacity), IDs: make([]string, 0, capacity), Operations: make([]string, 0, capacity), Kinds: make([]uint8, 0, capacity),
	}
}
