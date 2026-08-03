// Package pgwire implements the PostgreSQL wire protocol (v3) for direct
// client compatibility. Any PostgreSQL-compatible client (psql, JDBC, psycopg2,
// etc.) can connect to libraVDB without a driver.
package pgwire

import (
	"encoding/binary"
	"fmt"
	"io"
)

// PostgreSQL protocol v3 message types.
const (
	// Frontend (client → server)
	msgPassword  byte = 'p'
	msgQuery     byte = 'Q'
	msgParse     byte = 'P'
	msgBind      byte = 'B'
	msgDescribe  byte = 'D'
	msgExecute   byte = 'E'
	msgSync      byte = 'S'
	msgTerminate byte = 'X'
	msgFlush     byte = 'H'
	msgClose     byte = 'C'
	msgCopyData  byte = 'd' // COPY data from client
	msgCopyDone  byte = 'c' // COPY complete from client
	msgCopyFail  byte = 'f' // COPY failed from client

	// Backend (server → client)
	msgAuth                 byte = 'R'
	msgParameterStatus      byte = 'S'
	msgBackendKeyData       byte = 'K'
	msgReadyForQuery        byte = 'Z'
	msgRowDescription       byte = 'T'
	msgDataRow              byte = 'D'
	msgCommandComplete      byte = 'C'
	msgErrorResponse        byte = 'E'
	msgNoticeResponse       byte = 'N'
	msgEmptyQuery           byte = 'I'
	msgParseComplete        byte = '1'
	msgBindComplete         byte = '2'
	msgCloseComplete        byte = '3'
	msgNoData               byte = 'n'
	msgParameterDescription byte = 't'
	msgCopyInResponse       byte = 'G'
	msgCopyOutResponse      byte = 'H'

	// SSL negotiation
	sslRequestCode int32 = 80877103

	// Authentication types
	authOK        int32 = 0
	authCleartext int32 = 3
)

// Startup protocol version for PostgreSQL 3.0.
const protocolVersion int32 = 196608 // 3.0

// ReadMessage reads a single pgwire message: 1-byte type + 4-byte length (incl. self) + payload.
// Returns the message type byte and payload. On EOF, returns 0, nil, io.EOF.
func ReadMessage(r io.Reader) (byte, []byte, error) {
	var header [5]byte
	if _, err := io.ReadFull(r, header[:1]); err != nil {
		return 0, nil, err
	}
	msgType := header[0]

	if _, err := io.ReadFull(r, header[1:5]); err != nil {
		return 0, nil, fmt.Errorf("reading message length: %w", err)
	}
	length := int(binary.BigEndian.Uint32(header[1:5])) - 4 // subtract self
	if length < 0 || length > 1<<24 {                       // 16MB cap
		return 0, nil, fmt.Errorf("invalid message length: %d", length)
	}

	payload := make([]byte, length)
	if length > 0 {
		if _, err := io.ReadFull(r, payload); err != nil {
			return 0, nil, fmt.Errorf("reading message payload: %w", err)
		}
	}
	return msgType, payload, nil
}

// WriteMessage writes a single pgwire message: 1-byte type + 4-byte length + payload.
func WriteMessage(w io.Writer, msgType byte, payload []byte) error {
	length := 4 + len(payload) // includes self
	var header [5]byte
	header[0] = msgType
	binary.BigEndian.PutUint32(header[1:5], uint32(length))
	if _, err := w.Write(header[:]); err != nil {
		return fmt.Errorf("writing message header: %w", err)
	}
	if len(payload) > 0 {
		if _, err := w.Write(payload); err != nil {
			return fmt.Errorf("writing message payload: %w", err)
		}
	}
	return nil
}

// WriteNullTerminated writes a C-style null-terminated string.
func WriteNullTerminated(buf []byte, s string) []byte {
	buf = append(buf, s...)
	buf = append(buf, 0)
	return buf
}

// ReadNullTerminated reads a null-terminated string from payload starting at offset.
// Returns the string and the new offset.
func ReadNullTerminated(payload []byte, offset int) (string, int) {
	end := offset
	for end < len(payload) && payload[end] != 0 {
		end++
	}
	s := string(payload[offset:end])
	if end < len(payload) {
		end++ // skip null terminator
	}
	return s, end
}
