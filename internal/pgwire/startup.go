package pgwire

import (
	"encoding/binary"
	"fmt"
	"io"

	"github.com/xDarkicex/libravdb/libravdb"
)

// StartupResult holds the outcome of startup negotiation.
type StartupResult struct {
	Database string
	User     string
}

// handleStartup performs the PostgreSQL startup handshake:
// 1. Read startup message (SSLRequest or normal startup)
// 2. If SSLRequest: respond 'N' (no SSL), client falls back to plain
// 3. Authenticate (trust mode — no password required)
// 4. Send server parameters and ReadyForQuery
func handleStartup(rw io.ReadWriter, db *libravdb.Database) (*StartupResult, error) {
	// Read the first message: either SSLRequest or startup packet.
	// Startup is length-prefixed but without a type byte for the first message.
	var lenBuf [4]byte
	if _, err := io.ReadFull(rw, lenBuf[:]); err != nil {
		return nil, fmt.Errorf("reading startup length: %w", err)
	}
	length := int(binary.BigEndian.Uint32(lenBuf[:]))
	if length < 8 || length > 1<<20 {
		return nil, fmt.Errorf("invalid startup packet length: %d", length)
	}

	// Read the rest of the startup packet
	payload := make([]byte, length-4)
	if _, err := io.ReadFull(rw, payload); err != nil {
		return nil, fmt.Errorf("reading startup payload: %w", err)
	}

	// Check for SSLRequest
	if len(payload) >= 4 {
		proto := int32(binary.BigEndian.Uint32(payload[:4]))
		if proto == sslRequestCode {
			// Client wants SSL — we don't support it, respond 'N'
			if _, err := rw.Write([]byte{'N'}); err != nil {
				return nil, fmt.Errorf("sending SSL decline: %w", err)
			}
			// Client will retry without SSL — read the real startup packet
			if _, err := io.ReadFull(rw, lenBuf[:]); err != nil {
				return nil, fmt.Errorf("reading startup after SSL decline: %w", err)
			}
			length = int(binary.BigEndian.Uint32(lenBuf[:]))
			payload = make([]byte, length-4)
			if _, err := io.ReadFull(rw, payload); err != nil {
				return nil, fmt.Errorf("reading startup payload after SSL: %w", err)
			}
		}
	}

	// Verify protocol version
	if len(payload) < 4 {
		return nil, fmt.Errorf("startup payload too short")
	}
	major := int32(binary.BigEndian.Uint32(payload[:4])) >> 16
	if major != 3 {
		return nil, fmt.Errorf("unsupported protocol version %d", major)
	}

	// Parse key=value pairs
	result := &StartupResult{}
	offset := 4
	for offset < len(payload)-1 {
		key, nextOff := ReadNullTerminated(payload, offset)
		offset = nextOff
		if offset >= len(payload) || key == "" {
			break
		}
		val, nextOff := ReadNullTerminated(payload, offset)
		offset = nextOff

		switch key {
		case "database":
			result.Database = val
		case "user":
			result.User = val
		}
	}

	// Trust authentication: no password required
	if err := sendAuthOK(rw); err != nil {
		return nil, err
	}

	// Send server parameters
	if err := sendParameterStatus(rw, "server_version", "libraVDB/0.1"); err != nil {
		return nil, err
	}
	if err := sendParameterStatus(rw, "client_encoding", "UTF8"); err != nil {
		return nil, err
	}
	if err := sendParameterStatus(rw, "server_encoding", "UTF8"); err != nil {
		return nil, err
	}

	// BackendKeyData: PID and secret key (zeroed — no cancel support yet)
	if err := WriteMessage(rw, msgBackendKeyData, []byte{
		0, 0, 0, 0, // PID
		0, 0, 0, 0, // secret key
	}); err != nil {
		return nil, err
	}

	// ReadyForQuery (idle, no transaction)
	if err := WriteMessage(rw, msgReadyForQuery, []byte{'I'}); err != nil {
		return nil, err
	}

	return result, nil
}

// sendAuthOK sends AuthenticationOk to the client.
func sendAuthOK(w io.Writer) error {
	var buf [4]byte
	binary.BigEndian.PutUint32(buf[:], uint32(authOK))
	return WriteMessage(w, msgAuth, buf[:])
}

// sendParameterStatus sends a ParameterStatus message (key=value).
func sendParameterStatus(w io.Writer, key, val string) error {
	var buf []byte
	buf = WriteNullTerminated(buf, key)
	buf = WriteNullTerminated(buf, val)
	return WriteMessage(w, msgParameterStatus, buf)
}

// sendError sends an ErrorResponse for the given error, extracting the
// SQLSTATE code automatically via errorToSQLState.
func sendError(w io.Writer, severity string, err error) error {
	return sendErrorWithCode(w, severity, errorToSQLState(err), err.Error())
}

// sendErrorWithCode sends an ErrorResponse with explicit severity, SQLSTATE, and message.
func sendErrorWithCode(w io.Writer, severity, sqlstate, message string) error {
	var buf []byte
	buf = append(buf, 'S') // Severity
	buf = WriteNullTerminated(buf, severity)
	buf = append(buf, 'C') // SQLSTATE code
	buf = WriteNullTerminated(buf, sqlstate)
	buf = append(buf, 'M') // Message
	buf = WriteNullTerminated(buf, message)
	buf = append(buf, 0) // terminator
	return WriteMessage(w, msgErrorResponse, buf)
}
