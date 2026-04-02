package hnsw

import "fmt"

type VectorEncoding uint8

const (
	VectorEncodingNone VectorEncoding = iota
	VectorEncodingRaw
	VectorEncodingCompressed
)

type VectorRef struct {
	Kind  VectorEncoding
	Slot  uint32
	Bytes uint32
	Valid bool
}

type RawVectorStoreProfile struct {
	Backend             string
	VectorCount         int
	Dimension           int
	BytesPerVector      int
	MemoryUsage         int64
	ReservedBytes       int64
	ReservedDataBytes   int64
	ReservedMetaBytes   int64
	ReservedGuardBytes  int64
	LiveBytes           int64
	FreeBytes           int64
	CapacityUtilization float64
}

type RawVectorStore interface {
	Put(vec []float32) (VectorRef, error)
	Get(ref VectorRef) ([]float32, error)
	Delete(ref VectorRef) error
	Reset() error
	Close() error
	MemoryUsage() int64
	Profile() RawVectorStoreProfile
}

type InMemoryRawVectorStore struct {
	dim     int
	vectors [][]float32
	bytes   int64
	active  int
}

func NewInMemoryRawVectorStore(dim int) *InMemoryRawVectorStore {
	return &InMemoryRawVectorStore{
		dim:     dim,
		vectors: make([][]float32, 0),
	}
}

func (s *InMemoryRawVectorStore) Put(vec []float32) (VectorRef, error) {
	if s == nil {
		return VectorRef{}, fmt.Errorf("raw vector store is nil")
	}
	if s.dim > 0 && len(vec) != s.dim {
		return VectorRef{}, fmt.Errorf("vector dimension mismatch: expected %d, got %d", s.dim, len(vec))
	}

	stored := make([]float32, len(vec))
	copy(stored, vec)
	s.vectors = append(s.vectors, stored)
	s.bytes += int64(len(stored) * 4)
	s.active++

	return VectorRef{
		Kind:  VectorEncodingRaw,
		Slot:  uint32(len(s.vectors) - 1),
		Bytes: uint32(len(stored) * 4),
		Valid: true,
	}, nil
}

func (s *InMemoryRawVectorStore) Get(ref VectorRef) ([]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("raw vector store is nil")
	}
	if !ref.Valid || ref.Kind != VectorEncodingRaw {
		return nil, fmt.Errorf("invalid raw vector reference")
	}
	if int(ref.Slot) >= len(s.vectors) {
		return nil, fmt.Errorf("raw vector slot out of range: %d", ref.Slot)
	}
	vec := s.vectors[ref.Slot]
	if vec == nil {
		return nil, fmt.Errorf("raw vector slot %d is empty", ref.Slot)
	}
	return vec, nil
}

func (s *InMemoryRawVectorStore) Delete(ref VectorRef) error {
	if s == nil || !ref.Valid || ref.Kind != VectorEncodingRaw {
		return nil
	}
	if int(ref.Slot) >= len(s.vectors) {
		return nil
	}
	if vec := s.vectors[ref.Slot]; vec != nil {
		s.bytes -= int64(len(vec) * 4)
		s.vectors[ref.Slot] = nil
		s.active--
	}
	return nil
}

func (s *InMemoryRawVectorStore) Reset() error {
	if s == nil {
		return nil
	}
	s.vectors = nil
	s.bytes = 0
	s.active = 0
	return nil
}

func (s *InMemoryRawVectorStore) MemoryUsage() int64 {
	if s == nil {
		return 0
	}
	return s.bytes
}

func (s *InMemoryRawVectorStore) Close() error {
	return s.Reset()
}

func (s *InMemoryRawVectorStore) Profile() RawVectorStoreProfile {
	return RawVectorStoreProfile{
		Backend:             RawVectorStoreMemory,
		VectorCount:         s.active,
		Dimension:           s.dim,
		BytesPerVector:      s.dim * 4,
		MemoryUsage:         s.bytes,
		ReservedBytes:       s.bytes,
		ReservedDataBytes:   s.bytes,
		ReservedMetaBytes:   0,
		ReservedGuardBytes:  0,
		LiveBytes:           s.bytes,
		FreeBytes:           0,
		CapacityUtilization: 1.0,
	}
}
