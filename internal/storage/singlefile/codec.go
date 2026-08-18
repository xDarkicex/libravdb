package singlefile

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"sort"
	"strings"

	"github.com/xDarkicex/libravdb/internal/storage"
	"github.com/xDarkicex/libravdb/internal/util"
)

const (
	codecVersion byte = 3 // Binary payload encoding (snapshot state, WAL frames, collection records)
)

const snapshotCodecVersion byte = 8 // v4: historical versions; v5: graph tombstones; v6: temporal catalog; v7: edge kinds; v8: edge directionality

var graphConfigFieldMagic = []byte{'G', 'R', 'P', 'H', 1}

type encodedPayload struct {
	encoder *util.BinaryEncoder
	bytes   []byte
}

type collectionStatsPayload struct {
	Name  string
	Stats []byte
}

func emptyPayload() encodedPayload {
	return encodedPayload{}
}

func detachPayload(enc *util.BinaryEncoder) encodedPayload {
	buf := enc.DetachBytes()
	return encodedPayload{
		bytes:   buf,
		encoder: enc,
	}
}

func encodeStateBinary(state *persistedState) ([]byte, error) {
	enc := util.AcquireBinaryEncoder(estimateStateSize(state))
	defer util.ReleaseBinaryEncoder(enc)
	enc.WriteByte(snapshotCodecVersion)
	enc.WriteUint64(state.NextCollectionID)
	enc.WriteUint64(state.NextGraphNodeID)
	enc.WriteUint32(uint32(len(state.TombstonedGraphNodeIDs)))
	for _, id := range state.TombstonedGraphNodeIDs {
		enc.WriteUint64(id)
	}
	if uint64(len(state.CommitCatalog)) > uint64(^uint32(0)) {
		return nil, fmt.Errorf("commit catalog too large: %d", len(state.CommitCatalog))
	}
	enc.WriteUint32(uint32(len(state.CommitCatalog)))
	for _, entry := range state.CommitCatalog {
		enc.WriteUint64(entry.LSN)
		enc.WriteUint64(uint64(entry.Timestamp))
	}
	enc.WriteUint64(state.OldestRetainedLSN)
	if uint64(len(state.EdgeKinds)) > uint64(^uint32(0)) {
		return nil, fmt.Errorf("edge kind registry too large: %d", len(state.EdgeKinds))
	}
	edgeKindNames := make([]string, 0, len(state.EdgeKinds))
	for name := range state.EdgeKinds {
		edgeKindNames = append(edgeKindNames, name)
	}
	sort.Strings(edgeKindNames)
	enc.WriteUint32(uint32(len(edgeKindNames)))
	for _, name := range edgeKindNames {
		enc.WriteString(name)
		_ = enc.WriteByte(state.EdgeKinds[name])
		if state.UndirectedEdgeKinds[name] {
			_ = enc.WriteByte(1)
		} else {
			_ = enc.WriteByte(0)
		}
	}
	names := make([]string, 0, len(state.Collections))
	for name := range state.Collections {
		names = append(names, name)
	}
	sort.Strings(names)
	enc.WriteUint32(uint32(len(names)))
	for _, name := range names {
		collection := state.Collections[name]
		enc.WriteString(name)
		if err := writeCollection(enc, collection); err != nil {
			return nil, err
		}
	}
	return enc.DetachBytes(), nil
}

func decodeStateBinary(data []byte) (*persistedState, error) {
	dec := &util.BinaryDecoder{Data: data}
	version, err := dec.ReadByte()
	if err != nil {
		return nil, err
	}
	if version < 1 || version > snapshotCodecVersion {
		return nil, fmt.Errorf("unsupported snapshot codec version %d (max %d)", version, snapshotCodecVersion)
	}
	nextCollectionID, err := dec.ReadUint64()
	if err != nil {
		return nil, err
	}
	// NextGraphNodeID: present in v3+, absent in v1/v2 (legacy).
	nextGraphNodeID := uint64(1)
	if version >= 3 {
		nextGraphNodeID, err = dec.ReadUint64()
		if err != nil {
			return nil, err
		}
	}
	var tombstonedGraphNodeIDs []uint64
	if version >= 5 {
		count, err := dec.ReadUint32()
		if err != nil {
			return nil, err
		}
		tombstonedGraphNodeIDs = make([]uint64, count)
		for i := range tombstonedGraphNodeIDs {
			tombstonedGraphNodeIDs[i], err = dec.ReadUint64()
			if err != nil {
				return nil, err
			}
		}
	}
	var commitCatalog []commitEntry
	var oldestRetainedLSN uint64
	var edgeKinds map[string]uint8
	var stateUndirectedEdgeKinds map[string]bool
	if version >= 6 {
		count, err := dec.ReadUint32()
		if err != nil {
			return nil, err
		}
		commitCatalog = make([]commitEntry, count)
		for i := range commitCatalog {
			commitCatalog[i].LSN, err = dec.ReadUint64()
			if err != nil {
				return nil, err
			}
			ts, readErr := dec.ReadUint64()
			if readErr != nil {
				return nil, readErr
			}
			commitCatalog[i].Timestamp = int64(ts)
		}
		oldestRetainedLSN, err = dec.ReadUint64()
		if err != nil {
			return nil, err
		}
	}
	if version >= 7 {
		count, err := dec.ReadUint32()
		if err != nil {
			return nil, err
		}
		if count > 0 {
			edgeKinds = make(map[string]uint8, count)
		}
		for i := uint32(0); i < count; i++ {
			name, err := dec.ReadString()
			if err != nil {
				return nil, err
			}
			kind, err := dec.ReadByte()
			if err != nil {
				return nil, err
			}
			edgeKinds[name] = kind
			if version >= 8 {
				undirected, err := dec.ReadByte()
				if err != nil {
					return nil, err
				}
				if undirected != 0 {
					if stateUndirectedEdgeKinds == nil {
						stateUndirectedEdgeKinds = make(map[string]bool, count)
					}
					stateUndirectedEdgeKinds[name] = true
				}
			}
		}
	}
	count, err := dec.ReadUint32()
	if err != nil {
		return nil, err
	}
	state := &persistedState{
		NextCollectionID:       nextCollectionID,
		NextGraphNodeID:        nextGraphNodeID,
		TombstonedGraphNodeIDs: tombstonedGraphNodeIDs,
		CommitCatalog:          commitCatalog,
		OldestRetainedLSN:      oldestRetainedLSN,
		EdgeKinds:              edgeKinds,
		UndirectedEdgeKinds:    stateUndirectedEdgeKinds,
		Collections:            make(map[string]*persistedCollection, count),
	}
	for i := uint32(0); i < count; i++ {
		name, err := dec.ReadString()
		if err != nil {
			return nil, err
		}
		collection, err := readCollection(dec, version)
		if err != nil {
			return nil, err
		}
		state.Collections[name] = collection
	}
	return state, nil
}

func encodeCollectionCreatePayloadBinary(payload collectionCreatePayload) (encodedPayload, error) {
	enc := util.AcquireBinaryEncoder(estimateCollectionCreatePayloadSize(payload))
	enc.WriteByte(codecVersion)
	enc.WriteString(payload.Name)
	if err := writeCollectionConfig(enc, payload.Config); err != nil {
		util.ReleaseBinaryEncoder(enc)
		return encodedPayload{}, err
	}
	return detachPayload(enc), nil
}

func decodeCollectionCreatePayloadBinary(data []byte) (collectionCreatePayload, error) {
	dec := &util.BinaryDecoder{Data: data}
	if err := dec.ExpectVersion(); err != nil {
		return collectionCreatePayload{}, err
	}
	name, err := dec.ReadString()
	if err != nil {
		return collectionCreatePayload{}, err
	}
	config, err := readCollectionConfig(dec)
	if err != nil {
		return collectionCreatePayload{}, err
	}
	return collectionCreatePayload{Name: name, Config: config}, nil
}

func encodeCollectionDeletePayloadBinary(payload collectionDeletePayload) (encodedPayload, error) {
	enc := util.AcquireBinaryEncoder(1 + 4 + len(payload.Name))
	enc.WriteByte(codecVersion)
	enc.WriteString(payload.Name)
	return detachPayload(enc), nil
}

func decodeCollectionDeletePayloadBinary(data []byte) (collectionDeletePayload, error) {
	dec := &util.BinaryDecoder{Data: data}
	if err := dec.ExpectVersion(); err != nil {
		return collectionDeletePayload{}, err
	}
	name, err := dec.ReadString()
	if err != nil {
		return collectionDeletePayload{}, err
	}
	return collectionDeletePayload{Name: name}, nil
}

func encodeCollectionStatsPayloadBinary(payload collectionStatsPayload) (encodedPayload, error) {
	enc := util.AcquireBinaryEncoder(1 + 4 + len(payload.Name) + 4 + len(payload.Stats))
	enc.WriteByte(codecVersion)
	enc.WriteString(payload.Name)
	enc.WriteBytes(payload.Stats)
	return detachPayload(enc), nil
}

func decodeCollectionStatsPayloadBinary(data []byte) (collectionStatsPayload, error) {
	dec := &util.BinaryDecoder{Data: data}
	if err := dec.ExpectVersion(); err != nil {
		return collectionStatsPayload{}, err
	}
	name, err := dec.ReadString()
	if err != nil {
		return collectionStatsPayload{}, err
	}
	stats, err := dec.ReadBytes()
	if err != nil {
		return collectionStatsPayload{}, err
	}
	return collectionStatsPayload{Name: name, Stats: stats}, nil
}

func encodeRecordPutPayloadBinary(payload recordPutPayload) (encodedPayload, error) {
	enc := util.AcquireBinaryEncoder(estimateRecordPutPayloadSize(payload))
	enc.WriteByte(codecVersion)
	enc.WriteString(payload.Collection)
	enc.WriteString(payload.ID)
	enc.WriteUint32(payload.Ordinal)
	enc.WriteUint64(payload.GraphNodeID)
	enc.WriteVector(payload.Vector)
	if err := enc.WriteMetadata(payload.Metadata); err != nil {
		util.ReleaseBinaryEncoder(enc)
		return encodedPayload{}, err
	}
	return detachPayload(enc), nil
}

func decodeRecordPutPayloadBinary(data []byte) (recordPutPayload, error) {
	dec := &util.BinaryDecoder{Data: data}
	version, err := dec.ReadByte()
	if err != nil {
		return recordPutPayload{}, err
	}
	if version < 1 || version > codecVersion {
		return recordPutPayload{}, fmt.Errorf("unsupported record put codec version %d", version)
	}
	collection, err := dec.ReadString()
	if err != nil {
		return recordPutPayload{}, err
	}
	id, err := dec.ReadString()
	if err != nil {
		return recordPutPayload{}, err
	}
	ordinal, err := dec.ReadUint32()
	if err != nil {
		return recordPutPayload{}, err
	}
	// GraphNodeID is present in v3+ WAL payloads, absent in v1/v2.
	graphNodeID := uint64(0)
	if version >= 3 {
		graphNodeID, err = dec.ReadUint64()
		if err != nil {
			return recordPutPayload{}, err
		}
	}
	vector, err := dec.ReadVector()
	if err != nil {
		return recordPutPayload{}, err
	}
	metadata, err := dec.ReadMetadata()
	if err != nil {
		return recordPutPayload{}, err
	}
	return recordPutPayload{
		Collection:  collection,
		ID:          id,
		Ordinal:     ordinal,
		GraphNodeID: graphNodeID,
		Vector:      vector,
		Metadata:    metadata,
	}, nil
}

func encodeRecordDeletePayloadBinary(payload recordDeletePayload) (encodedPayload, error) {
	enc := util.AcquireBinaryEncoder(1 + 8 + len(payload.Collection) + len(payload.ID))
	enc.WriteByte(codecVersion)
	enc.WriteString(payload.Collection)
	enc.WriteString(payload.ID)
	return detachPayload(enc), nil
}

func decodeRecordDeletePayloadBinary(data []byte) (recordDeletePayload, error) {
	dec := &util.BinaryDecoder{Data: data}
	if err := dec.ExpectVersion(); err != nil {
		return recordDeletePayload{}, err
	}
	collection, err := dec.ReadString()
	if err != nil {
		return recordDeletePayload{}, err
	}
	id, err := dec.ReadString()
	if err != nil {
		return recordDeletePayload{}, err
	}
	return recordDeletePayload{Collection: collection, ID: id}, nil
}

// ── Graph edge WAL codec ──────────────────────────────────────────────────

type graphEdgeAddPayload struct {
	Collection string
	Src        uint64
	Tgt        uint64
	Weight     float32
	Kind       uint8
	Properties []byte
}

type graphEdgeRemovePayload struct {
	Collection string
	Src        uint64
	Tgt        uint64
	Kind       uint8
}

type graphNodeDropPayload struct {
	Collection string
	NodeID     uint64
}

type graphVertexLabelPayload struct {
	NodeID uint64
	Label  string
}

type edgeKindCreatePayload struct {
	Name       string
	Kind       uint8
	Undirected bool
}

func encodeEdgeKindCreatePayload(p edgeKindCreatePayload) encodedPayload {
	enc := util.AcquireBinaryEncoder(1 + 4 + len(p.Name) + 2)
	enc.WriteByte(codecVersion)
	enc.WriteString(p.Name)
	_ = enc.WriteByte(p.Kind)
	if p.Undirected {
		_ = enc.WriteByte(1)
	} else {
		_ = enc.WriteByte(0)
	}
	return detachPayload(enc)
}

func decodeEdgeKindCreatePayload(data []byte) (edgeKindCreatePayload, error) {
	dec := &util.BinaryDecoder{Data: data}
	if err := dec.ExpectVersion(); err != nil {
		return edgeKindCreatePayload{}, err
	}
	name, err := dec.ReadString()
	if err != nil {
		return edgeKindCreatePayload{}, err
	}
	kind, err := dec.ReadByte()
	if err != nil {
		return edgeKindCreatePayload{}, err
	}
	// The direction byte was added after the original edge-kind WAL format.
	// Old committed frames remain directed and continue to replay normally.
	undirected := false
	if dec.Off < len(dec.Data) {
		value, readErr := dec.ReadByte()
		if readErr != nil {
			return edgeKindCreatePayload{}, readErr
		}
		undirected = value != 0
	}
	return edgeKindCreatePayload{Name: name, Kind: kind, Undirected: undirected}, nil
}

func encodeGraphVertexLabelPayload(p graphVertexLabelPayload) encodedPayload {
	enc := util.AcquireBinaryEncoder(1 + 8 + 4 + len(p.Label))
	enc.WriteByte(codecVersion)
	enc.WriteUint64(p.NodeID)
	enc.WriteString(p.Label)
	return detachPayload(enc)
}
func decodeGraphVertexLabelPayload(data []byte) (graphVertexLabelPayload, error) {
	dec := &util.BinaryDecoder{Data: data}
	if err := dec.ExpectVersion(); err != nil {
		return graphVertexLabelPayload{}, err
	}
	nodeID, err := dec.ReadUint64()
	if err != nil {
		return graphVertexLabelPayload{}, err
	}
	label, err := dec.ReadString()
	if err != nil {
		return graphVertexLabelPayload{}, err
	}
	return graphVertexLabelPayload{NodeID: nodeID, Label: label}, nil
}

func encodeGraphEdgeAddPayload(p graphEdgeAddPayload) encodedPayload {
	enc := util.AcquireBinaryEncoder(1 + 4 + len(p.Collection) + 8 + 8 + 4 + 1 + 4 + len(p.Properties))
	enc.WriteByte(codecVersion)
	enc.WriteString(p.Collection)
	enc.WriteUint64(p.Src)
	enc.WriteUint64(p.Tgt)
	enc.WriteFloat32(p.Weight)
	enc.WriteByte(p.Kind)
	enc.WriteBytes(p.Properties)
	return detachPayload(enc)
}

func decodeGraphEdgeAddPayload(data []byte) (graphEdgeAddPayload, error) {
	dec := &util.BinaryDecoder{Data: data}
	if err := dec.ExpectVersion(); err != nil {
		return graphEdgeAddPayload{}, err
	}
	collection, err := dec.ReadString()
	if err != nil {
		return graphEdgeAddPayload{}, err
	}
	src, err := dec.ReadUint64()
	if err != nil {
		return graphEdgeAddPayload{}, err
	}
	tgt, err := dec.ReadUint64()
	if err != nil {
		return graphEdgeAddPayload{}, err
	}
	weight, err := dec.ReadFloat32()
	if err != nil {
		return graphEdgeAddPayload{}, err
	}
	kind, err := dec.ReadByte()
	if err != nil {
		return graphEdgeAddPayload{}, err
	}
	// Property bytes were added after the original graph-edge payload fields.
	// Older WAL frames end immediately after Kind and remain valid.
	var properties []byte
	if dec.Off < len(dec.Data) {
		properties, err = dec.ReadBytes()
		if err != nil {
			return graphEdgeAddPayload{}, err
		}
	}
	return graphEdgeAddPayload{Collection: collection, Src: src, Tgt: tgt, Weight: weight, Kind: kind, Properties: properties}, nil
}

func encodeGraphEdgeRemovePayload(p graphEdgeRemovePayload) encodedPayload {
	enc := util.AcquireBinaryEncoder(1 + 4 + len(p.Collection) + 8 + 8 + 1)
	enc.WriteByte(codecVersion)
	enc.WriteString(p.Collection)
	enc.WriteUint64(p.Src)
	enc.WriteUint64(p.Tgt)
	enc.WriteByte(p.Kind)
	return detachPayload(enc)
}

func decodeGraphEdgeRemovePayload(data []byte) (graphEdgeRemovePayload, error) {
	dec := &util.BinaryDecoder{Data: data}
	if err := dec.ExpectVersion(); err != nil {
		return graphEdgeRemovePayload{}, err
	}
	collection, err := dec.ReadString()
	if err != nil {
		return graphEdgeRemovePayload{}, err
	}
	src, err := dec.ReadUint64()
	if err != nil {
		return graphEdgeRemovePayload{}, err
	}
	tgt, err := dec.ReadUint64()
	if err != nil {
		return graphEdgeRemovePayload{}, err
	}
	kind, err := dec.ReadByte()
	if err != nil {
		return graphEdgeRemovePayload{}, err
	}
	return graphEdgeRemovePayload{Collection: collection, Src: src, Tgt: tgt, Kind: kind}, nil
}

func encodeGraphNodeDropPayload(p graphNodeDropPayload) encodedPayload {
	enc := util.AcquireBinaryEncoder(1 + 4 + len(p.Collection) + 8)
	enc.WriteByte(codecVersion)
	enc.WriteString(p.Collection)
	enc.WriteUint64(p.NodeID)
	return detachPayload(enc)
}

func decodeGraphNodeDropPayload(data []byte) (graphNodeDropPayload, error) {
	dec := &util.BinaryDecoder{Data: data}
	if err := dec.ExpectVersion(); err != nil {
		return graphNodeDropPayload{}, err
	}
	collection, err := dec.ReadString()
	if err != nil {
		return graphNodeDropPayload{}, err
	}
	nodeID, err := dec.ReadUint64()
	if err != nil {
		return graphNodeDropPayload{}, err
	}
	return graphNodeDropPayload{Collection: collection, NodeID: nodeID}, nil
}

// ── Commit timestamp catalog codec ────────────────────────────────────────

// txCommitPayload is embedded in every TxCommit WAL frame. Timestamp is UTC
// unix nano; 0 means pre-temporal (replayed from a database written before
// the commit catalog existed).
type txCommitPayload struct {
	Timestamp int64
}

func encodeTxCommitPayload(p txCommitPayload) encodedPayload {
	enc := util.AcquireBinaryEncoder(1 + 8)
	enc.WriteByte(codecVersion)
	enc.WriteUint64(uint64(p.Timestamp))
	return detachPayload(enc)
}

func decodeTxCommitPayload(data []byte) (txCommitPayload, bool) {
	if len(data) == 0 {
		return txCommitPayload{}, false
	}
	dec := &util.BinaryDecoder{Data: data}
	version, err := dec.ReadByte()
	if err != nil {
		return txCommitPayload{}, false
	}
	if version != codecVersion {
		return txCommitPayload{}, false
	}
	ts, err := dec.ReadUint64()
	if err != nil {
		return txCommitPayload{}, false
	}
	return txCommitPayload{Timestamp: int64(ts)}, true
}

func writeCollectionConfig(enc *util.BinaryEncoder, config storage.CollectionConfig) error {
	enc.WriteUint32(uint32(config.Dimension))
	enc.WriteUint32(uint32(config.Metric))
	enc.WriteUint32(uint32(config.IndexType))
	enc.WriteUint32(uint32(config.M))
	enc.WriteUint32(uint32(config.EfConstruction))
	enc.WriteUint32(uint32(config.EfSearch))
	enc.WriteFloat64(config.ML)
	enc.WriteUint32(uint32(config.Version))
	enc.WriteString(config.RawVectorStore)
	enc.WriteUint32(uint32(config.RawStoreCap))
	if config.Version >= 2 {
		// The optional block is length-prefixed so older readers can skip fields
		// they do not understand. CostModelStats is intentionally opaque to the
		// storage layer and is encoded as one length-prefixed optional field.
		// Collection declarations are appended as a second length-prefixed
		// optional field so readers of the previous layout can skip them.
		declarations := encodeCollectionDeclarations(config)
		optSize := uint32(4 + 4 + 4 + 4 + len(config.CostModelStats))
		if len(declarations) > 0 {
			optSize += uint32(4 + len(declarations))
		}
		if config.GraphEnabled {
			// A trailing length-prefixed optional field keeps older readers able
			// to skip this declaration using the existing optSize boundary.
			optSize += 5 // uint32 length + one boolean byte
		}
		enc.WriteUint32(optSize)
	}
	enc.WriteUint32(uint32(config.NClusters))
	enc.WriteUint32(uint32(config.NProbes))
	if config.Version >= 2 {
		enc.WriteUint32(uint32(config.IDMapCapacity))
		enc.WriteBytes(config.CostModelStats)
		if declarations := encodeCollectionDeclarations(config); len(declarations) > 0 {
			enc.WriteBytes(declarations)
		}
		if config.GraphEnabled {
			enc.WriteBytes(graphConfigFieldMagic)
		}
	}
	return nil
}

// encodeCollectionDeclarations is deliberately a small, deterministic blob
// inside the existing optional config block. It carries declarations only;
// metadata posting lists remain derived from records and are rebuilt on load.
func encodeCollectionDeclarations(config storage.CollectionConfig) []byte {
	if len(config.MetadataSchema) == 0 && len(config.IndexedFields) == 0 && len(config.SQLIndexes) == 0 {
		return nil
	}

	fields := make([]string, 0, len(config.MetadataSchema))
	for field := range config.MetadataSchema {
		fields = append(fields, field)
	}
	sort.Strings(fields)

	enc := util.AcquireBinaryEncoder(estimateCollectionDeclarationsSize(config))
	enc.WriteUint32(uint32(len(fields)))
	for _, field := range fields {
		enc.WriteString(field)
		_ = enc.WriteByte(config.MetadataSchema[field])
	}
	enc.WriteUint32(uint32(len(config.IndexedFields)))
	for _, field := range config.IndexedFields {
		enc.WriteString(field)
	}
	enc.WriteUint32(uint32(len(config.SQLIndexedFields)))
	for _, field := range config.SQLIndexedFields {
		enc.WriteString(field)
	}
	indexes := append([]storage.SQLIndexDefinition(nil), config.SQLIndexes...)
	sort.Slice(indexes, func(i, j int) bool {
		return strings.ToLower(indexes[i].Name) < strings.ToLower(indexes[j].Name)
	})
	enc.WriteUint32(uint32(len(indexes)))
	for _, index := range indexes {
		enc.WriteString(index.Name)
		enc.WriteBool(index.Unique)
		enc.WriteUint32(uint32(len(index.Columns)))
		for _, column := range index.Columns {
			enc.WriteString(column)
		}
	}
	data := append([]byte(nil), enc.Bytes()...)
	util.ReleaseBinaryEncoder(enc)
	return data
}

func estimateCollectionDeclarationsSize(config storage.CollectionConfig) int {
	size := 4 + 4
	for field := range config.MetadataSchema {
		size += 4 + len(field) + 1
	}
	for _, field := range config.IndexedFields {
		size += 4 + len(field)
	}
	size += 4
	for _, field := range config.SQLIndexedFields {
		size += 4 + len(field)
	}
	size += 4
	for _, index := range config.SQLIndexes {
		size += 4 + len(index.Name) + 1 + 4
		for _, column := range index.Columns {
			size += 4 + len(column)
		}
	}
	return size
}

func decodeCollectionDeclarations(data []byte) (map[string]uint8, []string, []storage.SQLIndexDefinition, []string, error) {
	if len(data) == 0 {
		return nil, nil, nil, nil, nil
	}
	dec := &util.BinaryDecoder{Data: data}
	schemaCount, err := dec.ReadUint32()
	if err != nil {
		return nil, nil, nil, nil, err
	}
	var schema map[string]uint8
	if schemaCount > 0 {
		schema = make(map[string]uint8, schemaCount)
	}
	for i := uint32(0); i < schemaCount; i++ {
		field, err := dec.ReadString()
		if err != nil {
			return nil, nil, nil, nil, err
		}
		fieldType, err := dec.ReadByte()
		if err != nil {
			return nil, nil, nil, nil, err
		}
		schema[field] = fieldType
	}
	indexedCount, err := dec.ReadUint32()
	if err != nil {
		return nil, nil, nil, nil, err
	}
	var indexed []string
	if indexedCount > 0 {
		indexed = make([]string, 0, indexedCount)
	}
	for i := uint32(0); i < indexedCount; i++ {
		field, err := dec.ReadString()
		if err != nil {
			return nil, nil, nil, nil, err
		}
		indexed = append(indexed, field)
	}
	var sqlIndexes []storage.SQLIndexDefinition
	var sqlIndexedFields []string
	// The SQL index section was appended after the original declaration
	// payload. Older snapshots end here and remain fully readable.
	if dec.Off < len(dec.Data) {
		sqlIndexedCount, readErr := dec.ReadUint32()
		if readErr != nil {
			return nil, nil, nil, nil, readErr
		}
		if sqlIndexedCount > 0 {
			sqlIndexedFields = make([]string, 0, sqlIndexedCount)
		}
		for i := uint32(0); i < sqlIndexedCount; i++ {
			field, readErr := dec.ReadString()
			if readErr != nil {
				return nil, nil, nil, nil, readErr
			}
			sqlIndexedFields = append(sqlIndexedFields, field)
		}
		indexCount, readErr := dec.ReadUint32()
		if readErr != nil {
			return nil, nil, nil, nil, readErr
		}
		if indexCount > 0 {
			sqlIndexes = make([]storage.SQLIndexDefinition, 0, indexCount)
		}
		for i := uint32(0); i < indexCount; i++ {
			name, readErr := dec.ReadString()
			if readErr != nil {
				return nil, nil, nil, nil, readErr
			}
			unique, readErr := dec.ReadBool()
			if readErr != nil {
				return nil, nil, nil, nil, readErr
			}
			columnCount, readErr := dec.ReadUint32()
			if readErr != nil {
				return nil, nil, nil, nil, readErr
			}
			definition := storage.SQLIndexDefinition{Name: name, Unique: unique}
			if columnCount > 0 {
				definition.Columns = make([]string, 0, columnCount)
			}
			for j := uint32(0); j < columnCount; j++ {
				column, readErr := dec.ReadString()
				if readErr != nil {
					return nil, nil, nil, nil, readErr
				}
				definition.Columns = append(definition.Columns, column)
			}
			sqlIndexes = append(sqlIndexes, definition)
		}
	}
	if dec.Off != len(dec.Data) {
		return nil, nil, nil, nil, fmt.Errorf("trailing bytes in collection declarations: %d", len(dec.Data)-dec.Off)
	}
	return schema, indexed, sqlIndexes, sqlIndexedFields, nil
}

func writeCollection(enc *util.BinaryEncoder, collection *persistedCollection) error {
	enc.WriteUint64(collection.ID)
	if err := writeCollectionConfig(enc, collection.Config); err != nil {
		return err
	}
	enc.WriteUint64(collection.CreatedLSN)
	enc.WriteUint64(collection.UpdatedLSN)
	enc.WriteBool(collection.Deleted)
	enc.WriteUint64(collection.LiveCount)
	enc.WriteUint32(collection.NextOrdinal)
	ids := make([]string, 0, len(collection.Records))
	for id := range collection.Records {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	enc.WriteUint32(uint32(len(ids)))
	for _, id := range ids {
		record := collection.Records[id]
		enc.WriteString(id)
		enc.WriteUint64(record.Version)
		enc.WriteUint64(record.CreatedLSN)
		enc.WriteUint64(record.UpdatedLSN)
		enc.WriteBool(record.Deleted)
		enc.WriteUint32(record.Ordinal)
		enc.WriteUint64(record.GraphNodeID)
		enc.WriteVector(record.Vector)
		if err := enc.WriteMetadata(record.Metadata); err != nil {
			return err
		}
	}
	// Historical versions (snapshot codec v4+).
	numHistorical := uint32(len(collection.HistoricalVersions))
	enc.WriteUint32(numHistorical)
	if numHistorical > 0 {
		histIDs := make([]string, 0, len(collection.HistoricalVersions))
		for id := range collection.HistoricalVersions {
			histIDs = append(histIDs, id)
		}
		sort.Strings(histIDs)
		for _, id := range histIDs {
			versions := collection.HistoricalVersions[id]
			enc.WriteString(id)
			enc.WriteUint32(uint32(len(versions)))
			for _, v := range versions {
				enc.WriteUint64(v.BeginLSN)
				enc.WriteUint64(v.EndLSN)
				enc.WriteUint32(v.Ordinal)
				enc.WriteVector(v.Vector)
				if err := enc.WriteMetadata(v.Metadata); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func estimateStateSize(state *persistedState) int {
	size := 1 + 8 + 8 + 4 + len(state.TombstonedGraphNodeIDs)*8 + 4 + len(state.CommitCatalog)*16 + 8 // version + IDs + tombstones + catalog + collection count
	for name, collection := range state.Collections {
		size += 4 + len(name)
		size += estimateCollectionSize(collection)
	}
	return size
}

func estimateCollectionSize(collection *persistedCollection) int {
	if collection == nil {
		return 0
	}
	size := 8 + estimateCollectionConfigSize(collection.Config) + 8 + 8 + 1 + 8 + 4 + 4
	for id, record := range collection.Records {
		size += 4 + len(id)
		size += 8 + 8 + 8 + 1 + 4 + 8 // version, createdLSN, updatedLSN, deleted, ordinal, graphNodeID
		size += 4 + len(record.Vector)*4
		size += util.EstimateMetadataSize(record.Metadata)
	}
	return size
}

func estimateCollectionConfigSize(config storage.CollectionConfig) int {
	size := 4 + 4 + 4 + 4 + 4 + 4 + 8 + 4 + 4 + len(config.RawVectorStore) + 4 + 4 + 4
	if config.Version >= 2 {
		size += 4 + 4 + len(config.CostModelStats) // block length + stats bytes length + payload
		if declarations := encodeCollectionDeclarations(config); len(declarations) > 0 {
			size += 4 + len(declarations)
		}
		if config.GraphEnabled {
			size += 5
		}
	}
	return size
}

func estimateCollectionCreatePayloadSize(payload collectionCreatePayload) int {
	return 1 + 4 + len(payload.Name) + estimateCollectionConfigSize(payload.Config)
}

func estimateRecordPutPayloadSize(payload recordPutPayload) int {
	return 1 + 4 + len(payload.Collection) + 4 + len(payload.ID) + 4 + 8 + 4 + len(payload.Vector)*4 + util.EstimateMetadataSize(payload.Metadata)
}

func readCollectionConfig(dec *util.BinaryDecoder) (storage.CollectionConfig, error) {
	dimension, err := dec.ReadUint32()
	if err != nil {
		return storage.CollectionConfig{}, err
	}
	metric, err := dec.ReadUint32()
	if err != nil {
		return storage.CollectionConfig{}, err
	}
	indexType, err := dec.ReadUint32()
	if err != nil {
		return storage.CollectionConfig{}, err
	}
	m, err := dec.ReadUint32()
	if err != nil {
		return storage.CollectionConfig{}, err
	}
	efConstruction, err := dec.ReadUint32()
	if err != nil {
		return storage.CollectionConfig{}, err
	}
	efSearch, err := dec.ReadUint32()
	if err != nil {
		return storage.CollectionConfig{}, err
	}
	ml, err := dec.ReadFloat64()
	if err != nil {
		return storage.CollectionConfig{}, err
	}
	version, err := dec.ReadUint32()
	if err != nil {
		return storage.CollectionConfig{}, err
	}
	rawVectorStore, err := dec.ReadString()
	if err != nil {
		return storage.CollectionConfig{}, err
	}
	rawStoreCap, err := dec.ReadUint32()
	if err != nil {
		return storage.CollectionConfig{}, err
	}

	var nClusters uint32
	var nProbes uint32
	var idMapCapacity uint32
	var costModelStats []byte
	var metadataSchema map[string]uint8
	var indexedFields []string
	var sqlIndexes []storage.SQLIndexDefinition
	var sqlIndexedFields []string
	var graphEnabled bool

	if version >= 2 {
		if dec.Off+4 <= len(dec.Data) {
			optSize, err := dec.ReadUint32()
			if err != nil {
				return storage.CollectionConfig{}, err
			}
			if optSize >= 4 && dec.Off+4 <= len(dec.Data) {
				nClusters, err = dec.ReadUint32()
				if err != nil {
					return storage.CollectionConfig{}, err
				}
			}
			if optSize >= 8 && dec.Off+4 <= len(dec.Data) {
				nProbes, err = dec.ReadUint32()
				if err != nil {
					return storage.CollectionConfig{}, err
				}
			}
			if optSize >= 12 && dec.Off+4 <= len(dec.Data) {
				idMapCapacity, err = dec.ReadUint32()
				if err != nil {
					return storage.CollectionConfig{}, err
				}
			}
			if optSize >= 16 && dec.Off+4 <= len(dec.Data) {
				costModelStats, err = dec.ReadBytes()
				if err != nil {
					return storage.CollectionConfig{}, err
				}
			}
			// Skip any trailing unknown optional fields based on the length prefix.
			consumed := 12
			if optSize >= 16 {
				consumed += 4 + len(costModelStats)
			}
			if !hasGraphConfigField(dec, optSize, consumed) && int(optSize) >= consumed+4 && dec.Off+4 <= len(dec.Data) {
				declarationBytes, readErr := dec.ReadBytes()
				if readErr != nil {
					return storage.CollectionConfig{}, readErr
				}
				metadataSchema, indexedFields, sqlIndexes, sqlIndexedFields, readErr = decodeCollectionDeclarations(declarationBytes)
				if readErr != nil {
					return storage.CollectionConfig{}, fmt.Errorf("decode collection declarations: %w", readErr)
				}
				consumed += 4 + len(declarationBytes)
			}
			if hasGraphConfigField(dec, optSize, consumed) {
				graphBytes, readErr := dec.ReadBytes()
				if readErr != nil {
					return storage.CollectionConfig{}, readErr
				}
				graphEnabled = bytes.Equal(graphBytes, graphConfigFieldMagic)
				consumed += 4 + len(graphBytes)
			}
			if int(optSize) > consumed {
				dec.Off += int(optSize) - consumed
			}
		}
	} else {
		if dec.Off+4 <= len(dec.Data) {
			nClusters, err = dec.ReadUint32()
			if err != nil {
				return storage.CollectionConfig{}, err
			}
		}

		if dec.Off+4 <= len(dec.Data) {
			nProbes, err = dec.ReadUint32()
			if err != nil {
				return storage.CollectionConfig{}, err
			}
		}
	}

	return storage.CollectionConfig{
		Dimension:        int(dimension),
		Metric:           int(metric),
		IndexType:        int(indexType),
		M:                int(m),
		EfConstruction:   int(efConstruction),
		EfSearch:         int(efSearch),
		NClusters:        int(nClusters),
		NProbes:          int(nProbes),
		ML:               ml,
		Version:          int(version),
		RawVectorStore:   rawVectorStore,
		RawStoreCap:      int(rawStoreCap),
		IDMapCapacity:    int(idMapCapacity),
		CostModelStats:   costModelStats,
		MetadataSchema:   metadataSchema,
		IndexedFields:    indexedFields,
		SQLIndexes:       sqlIndexes,
		SQLIndexedFields: sqlIndexedFields,
		GraphEnabled:     graphEnabled,
	}, nil
}

func hasGraphConfigField(dec *util.BinaryDecoder, optSize uint32, consumed int) bool {
	if int(optSize) < consumed+4 || dec.Off+4 > len(dec.Data) {
		return false
	}
	length := binary.LittleEndian.Uint32(dec.Data[dec.Off : dec.Off+4])
	if length != uint32(len(graphConfigFieldMagic)) || dec.Off+4+int(length) > len(dec.Data) {
		return false
	}
	return bytes.Equal(dec.Data[dec.Off+4:dec.Off+4+int(length)], graphConfigFieldMagic)
}

func readCollection(dec *util.BinaryDecoder, snapshotVersion byte) (*persistedCollection, error) {
	id, err := dec.ReadUint64()
	if err != nil {
		return nil, err
	}
	config, err := readCollectionConfig(dec)
	if err != nil {
		return nil, err
	}
	createdLSN, err := dec.ReadUint64()
	if err != nil {
		return nil, err
	}
	updatedLSN, err := dec.ReadUint64()
	if err != nil {
		return nil, err
	}
	deleted, err := dec.ReadBool()
	if err != nil {
		return nil, err
	}
	liveCount, err := dec.ReadUint64()
	if err != nil {
		return nil, err
	}
	nextOrdinal, err := dec.ReadUint32()
	if err != nil {
		return nil, err
	}
	recordCount, err := dec.ReadUint32()
	if err != nil {
		return nil, err
	}
	records := make(map[string]*recordValue, recordCount)
	for i := uint32(0); i < recordCount; i++ {
		recordID, err := dec.ReadString()
		if err != nil {
			return nil, err
		}
		version, err := dec.ReadUint64()
		if err != nil {
			return nil, err
		}
		recordCreatedLSN, err := dec.ReadUint64()
		if err != nil {
			return nil, err
		}
		recordUpdatedLSN, err := dec.ReadUint64()
		if err != nil {
			return nil, err
		}
		recordDeleted, err := dec.ReadBool()
		if err != nil {
			return nil, err
		}
		ordinal, err := dec.ReadUint32()
		if err != nil {
			return nil, err
		}
		// GraphNodeID is present in v3+ format only.
		graphNodeID := uint64(0)
		if snapshotVersion >= 3 {
			graphNodeID, err = dec.ReadUint64()
			if err != nil {
				return nil, err
			}
		}
		vector, err := dec.ReadVector()
		if err != nil {
			return nil, err
		}
		metadata, err := dec.ReadMetadata()
		if err != nil {
			return nil, err
		}
		records[recordID] = &recordValue{
			Version:     version,
			CreatedLSN:  recordCreatedLSN,
			UpdatedLSN:  recordUpdatedLSN,
			Deleted:     recordDeleted,
			Ordinal:     ordinal,
			GraphNodeID: graphNodeID,
			Vector:      vector,
			Metadata:    metadata,
		}
	}
	collection := &persistedCollection{
		ID:          id,
		Config:      config,
		CreatedLSN:  createdLSN,
		UpdatedLSN:  updatedLSN,
		Deleted:     deleted,
		LiveCount:   liveCount,
		NextOrdinal: nextOrdinal,
		Records:     records,
	}
	for recordID, record := range records {
		if !record.Deleted {
			ensureOrdinalSlot(collection, record.Ordinal, recordID)
		}
	}
	// Historical versions (snapshot codec v4+).
	// Historical versions were introduced in snapshot codec v4. Keep this
	// threshold independent of the current codec version so v7 snapshots
	// remain readable after v8 adds edge direction metadata.
	if snapshotVersion >= 4 {
		numHistorical, err := dec.ReadUint32()
		if err != nil {
			return nil, err
		}
		if numHistorical > 0 {
			collection.HistoricalVersions = make(map[string][]recordVersion, numHistorical)
			for i := uint32(0); i < numHistorical; i++ {
				recID, err := dec.ReadString()
				if err != nil {
					return nil, err
				}
				versionCount, err := dec.ReadUint32()
				if err != nil {
					return nil, err
				}
				versions := make([]recordVersion, versionCount)
				for j := uint32(0); j < versionCount; j++ {
					beginLSN, err := dec.ReadUint64()
					if err != nil {
						return nil, err
					}
					endLSN, err := dec.ReadUint64()
					if err != nil {
						return nil, err
					}
					ordinal, err := dec.ReadUint32()
					if err != nil {
						return nil, err
					}
					vector, err := dec.ReadVector()
					if err != nil {
						return nil, err
					}
					metadata, err := dec.ReadMetadata()
					if err != nil {
						return nil, err
					}
					versions[j] = recordVersion{
						BeginLSN: beginLSN, EndLSN: endLSN, Ordinal: ordinal,
						Vector: vector, Metadata: metadata,
					}
				}
				collection.HistoricalVersions[recID] = versions
			}
		}
	}
	return collection, nil
}
