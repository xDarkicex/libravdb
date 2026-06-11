package util

import (
	"encoding/binary"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"
)

const (
	MaxCodecBytes     = 16 * 1024 * 1024
	MaxVectorSize     = 64 * 1024 * 1024
	MaxMetadataFields = 65536
	MaxStringLen      = 256 * 1024      // 256 KB per string
	MaxMetadataKeyLen = 1024            // 1 KB per metadata key
)

const (
	codecVersion byte = 1 // Tagged-value wire format (vectors, metadata, value types)

	minPooledEncoderCapacity = 256
	maxPooledEncoderCapacity = 16 << 10

	valueTypeNil byte = iota
	valueTypeBool
	valueTypeString
	valueTypeInt
	valueTypeInt64
	valueTypeUint64
	valueTypeFloat32
	valueTypeFloat64
	valueTypeStringSlice
	valueTypeInterfaceSlice
	valueTypeMap
)

type BinaryEncoder struct {
	Buf []byte
}

type BinaryDecoder struct {
	Data []byte
	Off  int
}

var BinaryEncoderPool = sync.Pool{
	New: func() interface{} {
		return &BinaryEncoder{}
	},
}

func AcquireBinaryEncoder(estimated int) *BinaryEncoder {
	enc := BinaryEncoderPool.Get().(*BinaryEncoder)
	enc.Buf = enc.Buf[:0]
	targetCap := pooledEncoderCapacity(estimated)
	if targetCap > cap(enc.Buf) {
		enc.Buf = make([]byte, 0, targetCap)
	}
	return enc
}

func ReleaseBinaryEncoder(enc *BinaryEncoder) {
	if enc == nil {
		return
	}
	if cap(enc.Buf) > maxPooledEncoderCapacity {
		enc.Buf = nil
	} else {
		enc.Buf = enc.Buf[:0]
	}
	BinaryEncoderPool.Put(enc)
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

func (enc *BinaryEncoder) DetachBytes() []byte {
	out := enc.Buf
	enc.Buf = nil
	return out
}

func (enc *BinaryEncoder) Bytes() []byte {
	return enc.Buf
}

func (enc *BinaryEncoder) WriteByte(value byte) error {
	enc.Buf = append(enc.Buf, value)
	return nil
}

func (enc *BinaryEncoder) WriteBool(value bool) {
	if value {
		enc.WriteByte(1)
		return
	}
	enc.WriteByte(0)
}

func (enc *BinaryEncoder) WriteUint32(value uint32) {
	var buf [4]byte
	binary.LittleEndian.PutUint32(buf[:], value)
	enc.Buf = append(enc.Buf, buf[:]...)
}

func (enc *BinaryEncoder) WriteUint64(value uint64) {
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], value)
	enc.Buf = append(enc.Buf, buf[:]...)
}

func (enc *BinaryEncoder) WriteFloat64(value float64) {
	enc.WriteUint64(math.Float64bits(value))
}

func (enc *BinaryEncoder) WriteFloat32(value float32) {
	enc.WriteUint32(math.Float32bits(value))
}

func (enc *BinaryEncoder) WriteBytes(value []byte) {
	enc.WriteUint32(uint32(len(value)))
	enc.Buf = append(enc.Buf, value...)
}

func (enc *BinaryEncoder) WriteString(value string) {
	if len(value) > MaxStringLen {
		enc.WriteUint32(uint32(MaxStringLen))
		enc.Buf = append(enc.Buf, value[:MaxStringLen]...)
		return
	}
	enc.WriteUint32(uint32(len(value)))
	enc.Buf = append(enc.Buf, value...)
}

func (enc *BinaryEncoder) WriteVector(vector []float32) {
	enc.WriteUint32(uint32(len(vector)))
	for _, value := range vector {
		enc.WriteFloat32(value)
	}
}

func (enc *BinaryEncoder) WriteMetadata(metadata map[string]interface{}) error {
	if len(metadata) == 0 {
		enc.WriteUint32(0)
		return nil
	}
	if len(metadata) > MaxMetadataFields {
		return fmt.Errorf("metadata field count %d exceeds limit %d", len(metadata), MaxMetadataFields)
	}
	for key := range metadata {
		if key == "" {
			return fmt.Errorf("metadata key cannot be empty")
		}
		if len(key) > MaxMetadataKeyLen {
			return fmt.Errorf("metadata key %q exceeds length limit %d", key, MaxMetadataKeyLen)
		}
	}

	if len(metadata) == 1 {
		enc.WriteUint32(1)
		for key, value := range metadata {
			enc.WriteString(key)
			return enc.WriteValue(value)
		}
	}

	if len(metadata) <= 8 {
		var local [8]string
		keys := local[:0]
		for key := range metadata {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		enc.WriteUint32(uint32(len(keys)))
		for _, key := range keys {
			enc.WriteString(key)
			if err := enc.WriteValue(metadata[key]); err != nil {
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
	enc.WriteUint32(uint32(len(keys)))
	for _, key := range keys {
		enc.WriteString(key)
		if err := enc.WriteValue(metadata[key]); err != nil {
			return err
		}
	}
	return nil
}

func (enc *BinaryEncoder) WriteValue(value interface{}) error {
	switch typed := value.(type) {
	case nil:
		enc.WriteByte(valueTypeNil)
	case bool:
		enc.WriteByte(valueTypeBool)
		enc.WriteBool(typed)
	case string:
		enc.WriteByte(valueTypeString)
		enc.WriteString(typed)
	case int:
		enc.WriteByte(valueTypeInt)
		enc.WriteUint64(uint64(typed))
	case int8:
		enc.WriteByte(valueTypeInt64)
		enc.WriteUint64(uint64(int64(typed)))
	case int16:
		enc.WriteByte(valueTypeInt64)
		enc.WriteUint64(uint64(int64(typed)))
	case int32:
		enc.WriteByte(valueTypeInt64)
		enc.WriteUint64(uint64(int64(typed)))
	case time.Time:
		enc.WriteByte(valueTypeString)
		enc.WriteString(typed.Format(time.RFC3339Nano))
	case int64:
		enc.WriteByte(valueTypeInt64)
		enc.WriteUint64(uint64(typed))
	case uint:
		enc.WriteByte(valueTypeUint64)
		enc.WriteUint64(uint64(typed))
	case uint8:
		enc.WriteByte(valueTypeUint64)
		enc.WriteUint64(uint64(typed))
	case uint16:
		enc.WriteByte(valueTypeUint64)
		enc.WriteUint64(uint64(typed))
	case uint32:
		enc.WriteByte(valueTypeUint64)
		enc.WriteUint64(uint64(typed))
	case uint64:
		enc.WriteByte(valueTypeUint64)
		enc.WriteUint64(typed)
	case float32:
		enc.WriteByte(valueTypeFloat32)
		enc.WriteFloat32(typed)
	case float64:
		enc.WriteByte(valueTypeFloat64)
		enc.WriteFloat64(typed)
	case []string:
		enc.WriteByte(valueTypeStringSlice)
		enc.WriteUint32(uint32(len(typed)))
		for _, item := range typed {
			enc.WriteString(item)
		}
	case []interface{}:
		enc.WriteByte(valueTypeInterfaceSlice)
		enc.WriteUint32(uint32(len(typed)))
		for _, item := range typed {
			if err := enc.WriteValue(item); err != nil {
				return err
			}
		}
	case map[string]interface{}:
		enc.WriteByte(valueTypeMap)
		if err := enc.WriteMetadata(typed); err != nil {
			return err
		}
	default:
		return fmt.Errorf("unsupported metadata type %T", value)
	}
	return nil
}

func EstimateValueSize(value interface{}) int {
	switch typed := value.(type) {
	case nil:
		return 1
	case bool:
		return 2
	case string:
		return 1 + 4 + len(typed)
	case time.Time:
		return 1 + 4 + len(typed.Format(time.RFC3339Nano))
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
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
			size += EstimateValueSize(item)
		}
		return size
	case map[string]interface{}:
		return 1 + EstimateMetadataSize(typed)
	default:
		return 1
	}
}

func EstimateMetadataValueSize(value interface{}) int64 {
	return int64(EstimateValueSize(value))
}

func (dec *BinaryDecoder) ExpectVersion() error {
	version, err := dec.ReadByte()
	if err != nil {
		return err
	}
	if version != codecVersion {
		return fmt.Errorf("unsupported codec version %d", version)
	}
	return nil
}

func (dec *BinaryDecoder) ReadByte() (byte, error) {
	if dec.Off >= len(dec.Data) {
		return 0, fmt.Errorf("unexpected end of data")
	}
	value := dec.Data[dec.Off]
	dec.Off++
	return value, nil
}

func (dec *BinaryDecoder) ReadBool() (bool, error) {
	value, err := dec.ReadByte()
	if err != nil {
		return false, err
	}
	return value != 0, nil
}

func (dec *BinaryDecoder) ReadUint32() (uint32, error) {
	if dec.Off+4 > len(dec.Data) {
		return 0, fmt.Errorf("unexpected end of data")
	}
	value := binary.LittleEndian.Uint32(dec.Data[dec.Off : dec.Off+4])
	dec.Off += 4
	return value, nil
}

func (dec *BinaryDecoder) ReadUint64() (uint64, error) {
	if dec.Off+8 > len(dec.Data) {
		return 0, fmt.Errorf("unexpected end of data")
	}
	value := binary.LittleEndian.Uint64(dec.Data[dec.Off : dec.Off+8])
	dec.Off += 8
	return value, nil
}

func (dec *BinaryDecoder) ReadFloat64() (float64, error) {
	value, err := dec.ReadUint64()
	if err != nil {
		return 0, err
	}
	return math.Float64frombits(value), nil
}

func (dec *BinaryDecoder) ReadFloat32() (float32, error) {
	value, err := dec.ReadUint32()
	if err != nil {
		return 0, err
	}
	return math.Float32frombits(value), nil
}

func (dec *BinaryDecoder) ReadBytes() ([]byte, error) {
	size, err := dec.ReadUint32()
	if err != nil {
		return nil, err
	}
	if size > MaxStringLen {
		return nil, fmt.Errorf("string/bytes size %d exceeds limit %d", size, MaxStringLen)
	}
	if dec.Off+int(size) > len(dec.Data) {
		return nil, fmt.Errorf("unexpected end of data")
	}
	value := dec.Data[dec.Off : dec.Off+int(size)]
	dec.Off += int(size)
	return append([]byte(nil), value...), nil
}

func (dec *BinaryDecoder) ReadString() (string, error) {
	value, err := dec.ReadBytes()
	if err != nil {
		return "", err
	}
	return string(value), nil
}

func (dec *BinaryDecoder) ReadVector() ([]float32, error) {
	size, err := dec.ReadUint32()
	if err != nil {
		return nil, err
	}
	if uint64(size)*4 > MaxVectorSize {
		return nil, fmt.Errorf("vector size %d exceeds limit", size)
	}
	vector := make([]float32, int(size))
	for i := range vector {
		vector[i], err = dec.ReadFloat32()
		if err != nil {
			return nil, err
		}
	}
	return vector, nil
}

func (dec *BinaryDecoder) ReadMetadata() (map[string]interface{}, error) {
	count, err := dec.ReadUint32()
	if err != nil {
		return nil, err
	}
	if count == 0 {
		return nil, nil
	}
	metadata := make(map[string]interface{}, count)
	for i := uint32(0); i < count; i++ {
		key, err := dec.ReadString()
		if err != nil {
			return nil, err
		}
		value, err := dec.ReadValue()
		if err != nil {
			return nil, err
		}
		metadata[key] = value
	}
	return metadata, nil
}

func (dec *BinaryDecoder) ReadValue() (interface{}, error) {
	valueType, err := dec.ReadByte()
	if err != nil {
		return nil, err
	}
	switch valueType {
	case valueTypeNil:
		return nil, nil
	case valueTypeBool:
		return dec.ReadBool()
	case valueTypeString:
		return dec.ReadString()
	case valueTypeInt:
		value, err := dec.ReadUint64()
		return int(value), err
	case valueTypeInt64:
		value, err := dec.ReadUint64()
		return int64(value), err
	case valueTypeUint64:
		return dec.ReadUint64()
	case valueTypeFloat32:
		return dec.ReadFloat32()
	case valueTypeFloat64:
		return dec.ReadFloat64()
	case valueTypeStringSlice:
		count, err := dec.ReadUint32()
		if err != nil {
			return nil, err
		}
		values := make([]string, int(count))
		for i := range values {
			values[i], err = dec.ReadString()
			if err != nil {
				return nil, err
			}
		}
		return values, nil
	case valueTypeInterfaceSlice:
		count, err := dec.ReadUint32()
		if err != nil {
			return nil, err
		}
		values := make([]interface{}, int(count))
		for i := range values {
			values[i], err = dec.ReadValue()
			if err != nil {
				return nil, err
			}
		}
		return values, nil
	case valueTypeMap:
		return dec.ReadMetadata()
	default:
		return nil, fmt.Errorf("unsupported metadata value type %d", valueType)
	}
}

func EstimateMetadataSize(metadata map[string]interface{}) int {
	if metadata == nil {
		return 4
	}
	size := 4
	for key, value := range metadata {
		size += 4 + len(key)
		size += int(EstimateValueSize(value))
	}
	return size
}
