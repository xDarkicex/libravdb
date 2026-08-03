package pgwire

import (
	"encoding/binary"
	"fmt"
	"io"
	"unsafe"

	"github.com/xDarkicex/memory"
)

// Default arena slab size: 64KB per connection.
const arenaSize = 64 * 1024

// connArena provides arena-backed buffers for zero-heap-allocation message I/O.
// Each connection gets one arena that is reused across queries.
type connArena struct {
	arena *memory.Arena
}

// newConnArena allocates a new connection arena.
func newConnArena() (*connArena, error) {
	a, err := memory.NewArena(arenaSize, 64)
	if err != nil {
		return nil, fmt.Errorf("allocating pgwire arena: %w", err)
	}
	return &connArena{arena: a}, nil
}

// allocBytes allocates n bytes from the arena. Returns a []byte backed by arena memory.
func (a *connArena) allocBytes(n int) ([]byte, error) {
	if n <= 0 {
		return nil, nil
	}
	ptr, err := a.arena.Alloc(uint64(n))
	if err != nil {
		return nil, fmt.Errorf("arena alloc %d bytes: %w", n, err)
	}
	// Convert the unsafe.Pointer to a []byte via unsafe slice
	return unsafeSlice(ptr, n), nil
}

// reset returns all arena memory to the free list. Call between queries.
func (a *connArena) reset() {
	a.arena.Reset()
}

// close releases the arena entirely. Call on connection close.
func (a *connArena) close() {
	a.arena.Free()
}

// readMessage reads a pgwire message into arena-backed memory.
// Uses arena allocation for the payload to avoid heap escapes.
func readMessageArena(r io.Reader, a *connArena) (byte, []byte, error) {
	var header [5]byte
	if _, err := io.ReadFull(r, header[:1]); err != nil {
		return 0, nil, err
	}
	msgType := header[0]

	if _, err := io.ReadFull(r, header[1:5]); err != nil {
		return 0, nil, fmt.Errorf("reading message length: %w", err)
	}
	length := int(binary.BigEndian.Uint32(header[1:5])) - 4 // subtract self
	if length < 0 || length > 1<<24 {
		return 0, nil, fmt.Errorf("invalid message length: %d", length)
	}

	if length == 0 {
		return msgType, nil, nil
	}

	payload, err := a.allocBytes(length)
	if err != nil {
		return 0, nil, err
	}
	if _, err := io.ReadFull(r, payload); err != nil {
		return 0, nil, fmt.Errorf("reading message payload: %w", err)
	}
	return msgType, payload, nil
}

// unsafeSlice converts an unsafe.Pointer to a []byte without copying.
// The returned slice points directly into the arena's memory region.
func unsafeSlice(ptr unsafe.Pointer, n int) []byte {
	// This is safe because the arena guarantees the memory is valid until
	// the next reset or free. The caller must not retain the slice across
	// arena resets.
	return (*[1 << 30]byte)(ptr)[:n:n]
}
