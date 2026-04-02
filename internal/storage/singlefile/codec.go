package singlefile

import (
	"encoding/binary"
	"fmt"
	"math"
	"sort"
	"sync"

	"github.com/xDarkicex/libravdb/internal/storage"
)

const (
	codecVersion byte = 1

	minPooledEncoderCapacity = 256
	maxPooledEncoderCapacity = 16 << 10

	valueTypeNil byte = iota
	valueTypeBool
	valueTypeString
	valueTypeInt
	valueTypeInt64
	valueTypeFloat32
	valueTypeFloat64
	valueTypeStringSlice
	valueTypeInterfaceSlice
	valueTypeMap
)

type binaryEncoder struct {
	buf []byte
}

type binaryDecoder struct {
	data []byte
	off  int
}

type encodedPayload struct {
	bytes   []byte
	encoder *binaryEncoder
}

func emptyPayload() encodedPayload {
	return encodedPayload{}
}

var binaryEncoderPool = sync.Pool{
	New: func() interface{} {
		return &binaryEncoder{}
	},
}

func acquireBinaryEncoder(estimated int) *binaryEncoder {
	enc := binaryEncoderPool.Get().(*binaryEncoder)
	enc.buf = enc.buf[:0]
	targetCap := pooledEncoderCapacity(estimated)
	if targetCap > cap(enc.buf) {
		enc.buf = make([]byte, 0, targetCap)
	}
	return enc
}

func releaseBinaryEncoder(enc *binaryEncoder) {
	if enc == nil {
		return
	}
	if cap(enc.buf) > maxPooledEncoderCapacity {
		enc.buf = nil
	} else {
		enc.buf = enc.buf[:0]
	}
	binaryEncoderPool.Put(enc)
}

func pooledEncoderCapacity(estimated int) int {
	if estimated <= 0 {
		return minPooledEncoderCapacity
	}
	if estimated > maxPooledEncoderCapacity {
		return estimated
	}
	capacity := minPooledEncoderCapacity
	for capacity < estimated {
		capacity <<= 1
	}
	return capacity
}

func (enc *binaryEncoder) detachBytes() []byte {
	out := enc.buf
	enc.buf = nil
	return out
}

func (enc *binaryEncoder) detachPayload() encodedPayload {
	buf := enc.detachBytes()
	return encodedPayload{
		bytes:   buf,
		encoder: enc,
	}
}

func encodeStateBinary(state *persistedState) ([]byte, error) {
	enc := acquireBinaryEncoder(estimateStateSize(state))
	defer releaseBinaryEncoder(enc)
	enc.writeByte(codecVersion)
	enc.writeUint64(state.NextCollectionID)
	names := make([]string, 0, len(state.Collections))
	for name := range state.Collections {
		names = append(names, name)
	}
	sort.Strings(names)
	enc.writeUint32(uint32(len(names)))
	for _, name := range names {
		collection := state.Collections[name]
		enc.writeString(name)
		if err := enc.writeCollection(collection); err != nil {
			return nil, err
		}
	}
	return enc.detachBytes(), nil
}

func decodeStateBinary(data []byte) (*persistedState, error) {
	dec := &binaryDecoder{data: data}
	version, err := dec.readByte()
	if err != nil {
		return nil, err
	}
	if version != codecVersion {
		return nil, fmt.Errorf("unsupported snapshot codec version %d", version)
	}
	nextCollectionID, err := dec.readUint64()
	if err != nil {
		return nil, err
	}
	count, err := dec.readUint32()
	if err != nil {
		return nil, err
	}
	state := &persistedState{
		NextCollectionID: nextCollectionID,
		Collections:      make(map[string]*persistedCollection, count),
	}
	for i := uint32(0); i < count; i++ {
		name, err := dec.readString()
		if err != nil {
			return nil, err
		}
		collection, err := dec.readCollection()
		if err != nil {
			return nil, err
		}
		state.Collections[name] = collection
	}
	return state, nil
}

func encodeCollectionCreatePayloadBinary(payload collectionCreatePayload) (encodedPayload, error) {
	enc := acquireBinaryEncoder(estimateCollectionCreatePayloadSize(payload))
	enc.writeByte(codecVersion)
	enc.writeString(payload.Name)
	if err := enc.writeCollectionConfig(payload.Config); err != nil {
		releaseBinaryEncoder(enc)
		return encodedPayload{}, err
	}
	return enc.detachPayload(), nil
}

func decodeCollectionCreatePayloadBinary(data []byte) (collectionCreatePayload, error) {
	dec := &binaryDecoder{data: data}
	if err := dec.expectVersion(); err != nil {
		return collectionCreatePayload{}, err
	}
	name, err := dec.readString()
	if err != nil {
		return collectionCreatePayload{}, err
	}
	config, err := dec.readCollectionConfig()
	if err != nil {
		return collectionCreatePayload{}, err
	}
	return collectionCreatePayload{Name: name, Config: config}, nil
}

func encodeCollectionDeletePayloadBinary(payload collectionDeletePayload) (encodedPayload, error) {
	enc := acquireBinaryEncoder(1 + 4 + len(payload.Name))
	enc.writeByte(codecVersion)
	enc.writeString(payload.Name)
	return enc.detachPayload(), nil
}

func decodeCollectionDeletePayloadBinary(data []byte) (collectionDeletePayload, error) {
	dec := &binaryDecoder{data: data}
	if err := dec.expectVersion(); err != nil {
		return collectionDeletePayload{}, err
	}
	name, err := dec.readString()
	if err != nil {
		return collectionDeletePayload{}, err
	}
	return collectionDeletePayload{Name: name}, nil
}

func encodeRecordPutPayloadBinary(payload recordPutPayload) (encodedPayload, error) {
	enc := acquireBinaryEncoder(estimateRecordPutPayloadSize(payload))
	enc.writeByte(codecVersion)
	enc.writeString(payload.Collection)
	enc.writeString(payload.ID)
	enc.writeUint32(payload.Ordinal)
	enc.writeVector(payload.Vector)
	if err := enc.writeMetadata(payload.Metadata); err != nil {
		releaseBinaryEncoder(enc)
		return encodedPayload{}, err
	}
	return enc.detachPayload(), nil
}

func decodeRecordPutPayloadBinary(data []byte) (recordPutPayload, error) {
	dec := &binaryDecoder{data: data}
	if err := dec.expectVersion(); err != nil {
		return recordPutPayload{}, err
	}
	collection, err := dec.readString()
	if err != nil {
		return recordPutPayload{}, err
	}
	id, err := dec.readString()
	if err != nil {
		return recordPutPayload{}, err
	}
	ordinal, err := dec.readUint32()
	if err != nil {
		return recordPutPayload{}, err
	}
	vector, err := dec.readVector()
	if err != nil {
		return recordPutPayload{}, err
	}
	metadata, err := dec.readMetadata()
	if err != nil {
		return recordPutPayload{}, err
	}
	return recordPutPayload{Collection: collection, ID: id, Ordinal: ordinal, Vector: vector, Metadata: metadata}, nil
}

func encodeRecordDeletePayloadBinary(payload recordDeletePayload) (encodedPayload, error) {
	enc := acquireBinaryEncoder(1 + 8 + len(payload.Collection) + len(payload.ID))
	enc.writeByte(codecVersion)
	enc.writeString(payload.Collection)
	enc.writeString(payload.ID)
	return enc.detachPayload(), nil
}

func decodeRecordDeletePayloadBinary(data []byte) (recordDeletePayload, error) {
	dec := &binaryDecoder{data: data}
	if err := dec.expectVersion(); err != nil {
		return recordDeletePayload{}, err
	}
	collection, err := dec.readString()
	if err != nil {
		return recordDeletePayload{}, err
	}
	id, err := dec.readString()
	if err != nil {
		return recordDeletePayload{}, err
	}
	return recordDeletePayload{Collection: collection, ID: id}, nil
}

func (enc *binaryEncoder) bytes() []byte {
	return enc.buf
}

func (enc *binaryEncoder) writeByte(value byte) {
	enc.buf = append(enc.buf, value)
}

func (enc *binaryEncoder) writeBool(value bool) {
	if value {
		enc.writeByte(1)
		return
	}
	enc.writeByte(0)
}

func (enc *binaryEncoder) writeUint32(value uint32) {
	var buf [4]byte
	binary.LittleEndian.PutUint32(buf[:], value)
	enc.buf = append(enc.buf, buf[:]...)
}

func (enc *binaryEncoder) writeUint64(value uint64) {
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], value)
	enc.buf = append(enc.buf, buf[:]...)
}

func (enc *binaryEncoder) writeFloat64(value float64) {
	enc.writeUint64(math.Float64bits(value))
}

func (enc *binaryEncoder) writeFloat32(value float32) {
	enc.writeUint32(math.Float32bits(value))
}

func (enc *binaryEncoder) writeBytes(value []byte) {
	enc.writeUint32(uint32(len(value)))
	enc.buf = append(enc.buf, value...)
}

func (enc *binaryEncoder) writeString(value string) {
	enc.writeUint32(uint32(len(value)))
	enc.buf = append(enc.buf, value...)
}

func (enc *binaryEncoder) writeVector(vector []float32) {
	enc.writeUint32(uint32(len(vector)))
	for _, value := range vector {
		enc.writeFloat32(value)
	}
}

func (enc *binaryEncoder) writeCollectionConfig(config storage.CollectionConfig) error {
	enc.writeUint32(uint32(config.Dimension))
	enc.writeUint32(uint32(config.Metric))
	enc.writeUint32(uint32(config.IndexType))
	enc.writeUint32(uint32(config.M))
	enc.writeUint32(uint32(config.EfConstruction))
	enc.writeUint32(uint32(config.EfSearch))
	enc.writeFloat64(config.ML)
	enc.writeUint32(uint32(config.Version))
	enc.writeString(config.RawVectorStore)
	enc.writeUint32(uint32(config.RawStoreCap))
	enc.writeUint32(uint32(config.NClusters))
	enc.writeUint32(uint32(config.NProbes))
	return nil
}

func (enc *binaryEncoder) writeCollection(collection *persistedCollection) error {
	enc.writeUint64(collection.ID)
	if err := enc.writeCollectionConfig(collection.Config); err != nil {
		return err
	}
	enc.writeUint64(collection.CreatedLSN)
	enc.writeUint64(collection.UpdatedLSN)
	enc.writeBool(collection.Deleted)
	enc.writeUint64(collection.LiveCount)
	enc.writeUint32(collection.NextOrdinal)
	ids := make([]string, 0, len(collection.Records))
	for id := range collection.Records {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	enc.writeUint32(uint32(len(ids)))
	for _, id := range ids {
		record := collection.Records[id]
		enc.writeString(id)
		enc.writeUint64(record.Version)
		enc.writeUint64(record.CreatedLSN)
		enc.writeUint64(record.UpdatedLSN)
		enc.writeBool(record.Deleted)
		enc.writeUint32(record.Ordinal)
		enc.writeVector(record.Vector)
		if err := enc.writeMetadata(record.Metadata); err != nil {
			return err
		}
	}
	return nil
}

func (enc *binaryEncoder) writeMetadata(metadata map[string]interface{}) error {
	if len(metadata) == 0 {
		enc.writeUint32(0)
		return nil
	}

	if len(metadata) == 1 {
		enc.writeUint32(1)
		for key, value := range metadata {
			enc.writeString(key)
			return enc.writeValue(value)
		}
	}

	if len(metadata) <= 8 {
		var local [8]string
		keys := local[:0]
		for key := range metadata {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		enc.writeUint32(uint32(len(keys)))
		for _, key := range keys {
			enc.writeString(key)
			if err := enc.writeValue(metadata[key]); err != nil {
				return err
			}
		}
		return nil
	}

	keys := make([]string, 0, len(metadata))
	for key := range metadata {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	enc.writeUint32(uint32(len(keys)))
	for _, key := range keys {
		enc.writeString(key)
		if err := enc.writeValue(metadata[key]); err != nil {
			return err
		}
	}
	return nil
}

func (enc *binaryEncoder) writeValue(value interface{}) error {
	switch typed := value.(type) {
	case nil:
		enc.writeByte(valueTypeNil)
	case bool:
		enc.writeByte(valueTypeBool)
		enc.writeBool(typed)
	case string:
		enc.writeByte(valueTypeString)
		enc.writeString(typed)
	case int:
		enc.writeByte(valueTypeInt)
		enc.writeUint64(uint64(typed))
	case int64:
		enc.writeByte(valueTypeInt64)
		enc.writeUint64(uint64(typed))
	case float32:
		enc.writeByte(valueTypeFloat32)
		enc.writeFloat32(typed)
	case float64:
		enc.writeByte(valueTypeFloat64)
		enc.writeFloat64(typed)
	case []string:
		enc.writeByte(valueTypeStringSlice)
		enc.writeUint32(uint32(len(typed)))
		for _, item := range typed {
			enc.writeString(item)
		}
	case []interface{}:
		enc.writeByte(valueTypeInterfaceSlice)
		enc.writeUint32(uint32(len(typed)))
		for _, item := range typed {
			if err := enc.writeValue(item); err != nil {
				return err
			}
		}
	case map[string]interface{}:
		enc.writeByte(valueTypeMap)
		if err := enc.writeMetadata(typed); err != nil {
			return err
		}
	default:
		return fmt.Errorf("unsupported metadata type %T", value)
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
		size += estimateMetadataSize(record.Metadata)
	}
	return size
}

func estimateCollectionConfigSize(config storage.CollectionConfig) int {
	return 4 + 4 + 4 + 4 + 4 + 4 + 8 + 4 + 4 + len(config.RawVectorStore) + 4 + 4 + 4
}

func estimateCollectionCreatePayloadSize(payload collectionCreatePayload) int {
	return 1 + 4 + len(payload.Name) + estimateCollectionConfigSize(payload.Config)
}

func estimateRecordPutPayloadSize(payload recordPutPayload) int {
	return 1 + 4 + len(payload.Collection) + 4 + len(payload.ID) + 4 + 4 + len(payload.Vector)*4 + estimateMetadataSize(payload.Metadata)
}

func estimateMetadataSize(metadata map[string]interface{}) int {
	if len(metadata) == 0 {
		return 4
	}
	size := 4
	for key, value := range metadata {
		size += 4 + len(key)
		size += estimateValueSize(value)
	}
	return size
}

func estimateValueSize(value interface{}) int {
	switch typed := value.(type) {
	case nil:
		return 1
	case bool:
		return 2
	case string:
		return 1 + 4 + len(typed)
	case int, int64:
		return 1 + 8
	case float32:
		return 1 + 4
	case float64:
		return 1 + 8
	case []string:
		size := 1 + 4
		for _, item := range typed {
			size += 4 + len(item)
		}
		return size
	case []interface{}:
		size := 1 + 4
		for _, item := range typed {
			size += estimateValueSize(item)
		}
		return size
	case map[string]interface{}:
		return 1 + estimateMetadataSize(typed)
	default:
		return 1
	}
}

func estimateMetadataValueSize(value interface{}) int64 {
	return int64(estimateValueSize(value))
}

func (dec *binaryDecoder) expectVersion() error {
	version, err := dec.readByte()
	if err != nil {
		return err
	}
	if version != codecVersion {
		return fmt.Errorf("unsupported codec version %d", version)
	}
	return nil
}

func (dec *binaryDecoder) readByte() (byte, error) {
	if dec.off >= len(dec.data) {
		return 0, fmt.Errorf("unexpected end of data")
	}
	value := dec.data[dec.off]
	dec.off++
	return value, nil
}

func (dec *binaryDecoder) readBool() (bool, error) {
	value, err := dec.readByte()
	if err != nil {
		return false, err
	}
	return value != 0, nil
}

func (dec *binaryDecoder) readUint32() (uint32, error) {
	if dec.off+4 > len(dec.data) {
		return 0, fmt.Errorf("unexpected end of data")
	}
	value := binary.LittleEndian.Uint32(dec.data[dec.off : dec.off+4])
	dec.off += 4
	return value, nil
}

func (dec *binaryDecoder) readUint64() (uint64, error) {
	if dec.off+8 > len(dec.data) {
		return 0, fmt.Errorf("unexpected end of data")
	}
	value := binary.LittleEndian.Uint64(dec.data[dec.off : dec.off+8])
	dec.off += 8
	return value, nil
}

func (dec *binaryDecoder) readFloat64() (float64, error) {
	value, err := dec.readUint64()
	if err != nil {
		return 0, err
	}
	return math.Float64frombits(value), nil
}

func (dec *binaryDecoder) readFloat32() (float32, error) {
	value, err := dec.readUint32()
	if err != nil {
		return 0, err
	}
	return math.Float32frombits(value), nil
}

func (dec *binaryDecoder) readBytes() ([]byte, error) {
	size, err := dec.readUint32()
	if err != nil {
		return nil, err
	}
	if dec.off+int(size) > len(dec.data) {
		return nil, fmt.Errorf("unexpected end of data")
	}
	value := dec.data[dec.off : dec.off+int(size)]
	dec.off += int(size)
	return append([]byte(nil), value...), nil
}

func (dec *binaryDecoder) readString() (string, error) {
	value, err := dec.readBytes()
	if err != nil {
		return "", err
	}
	return string(value), nil
}

func (dec *binaryDecoder) readVector() ([]float32, error) {
	size, err := dec.readUint32()
	if err != nil {
		return nil, err
	}
	vector := make([]float32, int(size))
	for i := range vector {
		vector[i], err = dec.readFloat32()
		if err != nil {
			return nil, err
		}
	}
	return vector, nil
}

func (dec *binaryDecoder) readCollectionConfig() (storage.CollectionConfig, error) {
	dimension, err := dec.readUint32()
	if err != nil {
		return storage.CollectionConfig{}, err
	}
	metric, err := dec.readUint32()
	if err != nil {
		return storage.CollectionConfig{}, err
	}
	indexType, err := dec.readUint32()
	if err != nil {
		return storage.CollectionConfig{}, err
	}
	m, err := dec.readUint32()
	if err != nil {
		return storage.CollectionConfig{}, err
	}
	efConstruction, err := dec.readUint32()
	if err != nil {
		return storage.CollectionConfig{}, err
	}
	efSearch, err := dec.readUint32()
	if err != nil {
		return storage.CollectionConfig{}, err
	}
	ml, err := dec.readFloat64()
	if err != nil {
		return storage.CollectionConfig{}, err
	}
	version, err := dec.readUint32()
	if err != nil {
		return storage.CollectionConfig{}, err
	}
	rawVectorStore, err := dec.readString()
	if err != nil {
		return storage.CollectionConfig{}, err
	}
	rawStoreCap, err := dec.readUint32()
	if err != nil {
		return storage.CollectionConfig{}, err
	}

	var nClusters uint32
	if dec.off+4 <= len(dec.data) {
		nClusters, err = dec.readUint32()
		if err != nil {
			return storage.CollectionConfig{}, err
		}
	}

	var nProbes uint32
	if dec.off+4 <= len(dec.data) {
		nProbes, err = dec.readUint32()
		if err != nil {
			return storage.CollectionConfig{}, err
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

func (dec *binaryDecoder) readCollection() (*persistedCollection, error) {
	id, err := dec.readUint64()
	if err != nil {
		return nil, err
	}
	config, err := dec.readCollectionConfig()
	if err != nil {
		return nil, err
	}
	createdLSN, err := dec.readUint64()
	if err != nil {
		return nil, err
	}
	updatedLSN, err := dec.readUint64()
	if err != nil {
		return nil, err
	}
	deleted, err := dec.readBool()
	if err != nil {
		return nil, err
	}
	liveCount, err := dec.readUint64()
	if err != nil {
		return nil, err
	}
	nextOrdinal, err := dec.readUint32()
	if err != nil {
		return nil, err
	}
	recordCount, err := dec.readUint32()
	if err != nil {
		return nil, err
	}
	records := make(map[string]*recordValue, recordCount)
	for i := uint32(0); i < recordCount; i++ {
		recordID, err := dec.readString()
		if err != nil {
			return nil, err
		}
		version, err := dec.readUint64()
		if err != nil {
			return nil, err
		}
		recordCreatedLSN, err := dec.readUint64()
		if err != nil {
			return nil, err
		}
		recordUpdatedLSN, err := dec.readUint64()
		if err != nil {
			return nil, err
		}
		recordDeleted, err := dec.readBool()
		if err != nil {
			return nil, err
		}
		ordinal, err := dec.readUint32()
		if err != nil {
			return nil, err
		}
		vector, err := dec.readVector()
		if err != nil {
			return nil, err
		}
		metadata, err := dec.readMetadata()
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
		ensureOrdinalSlot(collection, record.Ordinal, recordID)
	}
	return collection, nil
}

func (dec *binaryDecoder) readMetadata() (map[string]interface{}, error) {
	count, err := dec.readUint32()
	if err != nil {
		return nil, err
	}
	if count == 0 {
		return nil, nil
	}
	metadata := make(map[string]interface{}, count)
	for i := uint32(0); i < count; i++ {
		key, err := dec.readString()
		if err != nil {
			return nil, err
		}
		value, err := dec.readValue()
		if err != nil {
			return nil, err
		}
		metadata[key] = value
	}
	return metadata, nil
}

func (dec *binaryDecoder) readValue() (interface{}, error) {
	valueType, err := dec.readByte()
	if err != nil {
		return nil, err
	}
	switch valueType {
	case valueTypeNil:
		return nil, nil
	case valueTypeBool:
		return dec.readBool()
	case valueTypeString:
		return dec.readString()
	case valueTypeInt:
		value, err := dec.readUint64()
		return int(value), err
	case valueTypeInt64:
		value, err := dec.readUint64()
		return int64(value), err
	case valueTypeFloat32:
		return dec.readFloat32()
	case valueTypeFloat64:
		return dec.readFloat64()
	case valueTypeStringSlice:
		count, err := dec.readUint32()
		if err != nil {
			return nil, err
		}
		values := make([]string, int(count))
		for i := range values {
			values[i], err = dec.readString()
			if err != nil {
				return nil, err
			}
		}
		return values, nil
	case valueTypeInterfaceSlice:
		count, err := dec.readUint32()
		if err != nil {
			return nil, err
		}
		values := make([]interface{}, int(count))
		for i := range values {
			values[i], err = dec.readValue()
			if err != nil {
				return nil, err
			}
		}
		return values, nil
	case valueTypeMap:
		return dec.readMetadata()
	default:
		return nil, fmt.Errorf("unsupported metadata value type %d", valueType)
	}
}
