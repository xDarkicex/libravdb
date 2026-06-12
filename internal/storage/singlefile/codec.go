package singlefile

import (
	"fmt"
	"sort"

	"github.com/xDarkicex/libravdb/internal/storage"
	"github.com/xDarkicex/libravdb/internal/util"
)

const (
	codecVersion byte = 2 // Binary payload encoding (snapshot state, WAL frames, collection records)
)

type encodedPayload struct {
	encoder *util.BinaryEncoder
	bytes   []byte
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
	enc.WriteByte(codecVersion)
	enc.WriteUint64(state.NextCollectionID)
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
	if version != codecVersion {
		return nil, fmt.Errorf("unsupported snapshot codec version %d", version)
	}
	nextCollectionID, err := dec.ReadUint64()
	if err != nil {
		return nil, err
	}
	count, err := dec.ReadUint32()
	if err != nil {
		return nil, err
	}
	state := &persistedState{
		NextCollectionID: nextCollectionID,
		Collections:      make(map[string]*persistedCollection, count),
	}
	for i := uint32(0); i < count; i++ {
		name, err := dec.ReadString()
		if err != nil {
			return nil, err
		}
		collection, err := readCollection(dec)
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

func encodeRecordPutPayloadBinary(payload recordPutPayload) (encodedPayload, error) {
	enc := util.AcquireBinaryEncoder(estimateRecordPutPayloadSize(payload))
	enc.WriteByte(codecVersion)
	enc.WriteString(payload.Collection)
	enc.WriteString(payload.ID)
	enc.WriteUint32(payload.Ordinal)
	enc.WriteVector(payload.Vector)
	if err := enc.WriteMetadata(payload.Metadata); err != nil {
		util.ReleaseBinaryEncoder(enc)
		return encodedPayload{}, err
	}
	return detachPayload(enc), nil
}

func decodeRecordPutPayloadBinary(data []byte) (recordPutPayload, error) {
	dec := &util.BinaryDecoder{Data: data}
	if err := dec.ExpectVersion(); err != nil {
		return recordPutPayload{}, err
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
	vector, err := dec.ReadVector()
	if err != nil {
		return recordPutPayload{}, err
	}
	metadata, err := dec.ReadMetadata()
	if err != nil {
		return recordPutPayload{}, err
	}
	return recordPutPayload{Collection: collection, ID: id, Ordinal: ordinal, Vector: vector, Metadata: metadata}, nil
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
		// Calculate size of optional fields (NClusters, NProbes)
		optSize := uint32(4 + 4)
		enc.WriteUint32(optSize)
	}
	enc.WriteUint32(uint32(config.NClusters))
	enc.WriteUint32(uint32(config.NProbes))
	return nil
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
		enc.WriteVector(record.Vector)
		if err := enc.WriteMetadata(record.Metadata); err != nil {
			return err
		}
	}
	return nil
}

func estimateStateSize(state *persistedState) int {
	size := 1 + 8 + 4
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
		size += 8 + 8 + 8 + 1 + 4
		size += 4 + len(record.Vector)*4
		size += util.EstimateMetadataSize(record.Metadata)
	}
	return size
}

func estimateCollectionConfigSize(config storage.CollectionConfig) int {
	size := 4 + 4 + 4 + 4 + 4 + 4 + 8 + 4 + 4 + len(config.RawVectorStore) + 4 + 4 + 4
	if config.Version >= 2 {
		size += 4 // length prefix
	}
	return size
}

func estimateCollectionCreatePayloadSize(payload collectionCreatePayload) int {
	return 1 + 4 + len(payload.Name) + estimateCollectionConfigSize(payload.Config)
}

func estimateRecordPutPayloadSize(payload recordPutPayload) int {
	return 1 + 4 + len(payload.Collection) + 4 + len(payload.ID) + 4 + 4 + len(payload.Vector)*4 + util.EstimateMetadataSize(payload.Metadata)
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

	if version >= 2 {
		if dec.Off+4 <= len(dec.Data) {
			optSize, err := dec.ReadUint32()
			if err != nil {
				return storage.CollectionConfig{}, err
			}
			// Only read if we have exactly the optSize bytes remaining, or limit the read
			// For now, we know the two fields take 8 bytes.
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
			// Skip any trailing unknown optional fields based on the length prefix
			if optSize > 8 {
				dec.Off += int(optSize) - 8
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
		Dimension:      int(dimension),
		Metric:         int(metric),
		IndexType:      int(indexType),
		M:              int(m),
		EfConstruction: int(efConstruction),
		EfSearch:       int(efSearch),
		NClusters:      int(nClusters),
		NProbes:        int(nProbes),
		ML:             ml,
		Version:        int(version),
		RawVectorStore: rawVectorStore,
		RawStoreCap:    int(rawStoreCap),
	}, nil
}

func readCollection(dec *util.BinaryDecoder) (*persistedCollection, error) {
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
		vector, err := dec.ReadVector()
		if err != nil {
			return nil, err
		}
		metadata, err := dec.ReadMetadata()
		if err != nil {
			return nil, err
		}
		records[recordID] = &recordValue{
			Version:    version,
			CreatedLSN: recordCreatedLSN,
			UpdatedLSN: recordUpdatedLSN,
			Deleted:    recordDeleted,
			Ordinal:    ordinal,
			Vector:     vector,
			Metadata:   metadata,
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
	return collection, nil
}
