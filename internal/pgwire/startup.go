package pgwire

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net"
	"strconv"
	"time"
	"unicode/utf8"

	"github.com/xDarkicex/libravdb/libravdb"
)

const (
	maxStartupPacketBytes = 1 << 20
	maxStartupFieldBytes  = 8 << 10
	maxStartupParameters  = 256
)

// StartupResult holds the outcome of startup negotiation.
type StartupResult struct {
	Database string
	User     string
}

// handleStartup preserves the legacy in-process helper used by tests and
// callers that intentionally run the wire protocol without TLS/auth. The
// network Server uses handleStartupWithConfig below.
func handleStartup(rw io.ReadWriter, db *libravdb.Database) (*StartupResult, error) {
	_, result, err := handleStartupWithConfig(rw, db, nil, false, ServerConfig{})
	return result, err
}

// handleStartupWithConfig performs PostgreSQL startup negotiation. TLS is
// negotiated before the startup packet is parsed, and SCRAM is completed
// before any ReadyForQuery message is emitted. The returned reader/writer is
// the TLS connection when TLS was negotiated; callers must use it for all
// subsequent protocol messages.
func handleStartupWithConfig(rw io.ReadWriter, db *libravdb.Database, tlsConfig *tls.Config, requireTLS bool, config ServerConfig) (io.ReadWriter, *StartupResult, error) {
	return handleStartupWithConfigContext(context.Background(), rw, db, tlsConfig, requireTLS, config)
}

// handleStartupWithConfigContext is the cancellable startup path used by the
// network server. The context reaches external SCRAM credential lookup while
// preserving the legacy helper above for in-process callers and tests.
func handleStartupWithConfigContext(ctx context.Context, rw io.ReadWriter, db *libravdb.Database, tlsConfig *tls.Config, requireTLS bool, config ServerConfig) (io.ReadWriter, *StartupResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	payload, err := readStartupPacket(rw)
	if err != nil {
		return rw, nil, err
	}

	// SSLRequest is a special untyped startup packet. PostgreSQL requires the
	// one-byte response to be sent before the TLS record layer is installed.
	if len(payload) >= 4 && int32(binary.BigEndian.Uint32(payload[:4])) == sslRequestCode {
		if len(payload) != 4 {
			startupErr := fmt.Errorf("invalid SSLRequest packet length: %d", len(payload)+4)
			_ = sendErrorWithCode(rw, "FATAL", "08P01", startupErr.Error())
			return rw, nil, startupErr
		}
		if tlsConfig == nil {
			if requireTLS {
				startupErr := fmt.Errorf("TLS is required but no TLS configuration is available")
				_ = sendErrorWithCode(rw, "FATAL", "08004", startupErr.Error())
				return rw, nil, startupErr
			}
			if _, err := rw.Write([]byte{'N'}); err != nil {
				return rw, nil, fmt.Errorf("sending SSL decline: %w", err)
			}
			payload, err = readStartupPacket(rw)
			if err != nil {
				return rw, nil, fmt.Errorf("reading startup after SSL negotiation: %w", err)
			}
		} else {
			conn, ok := rw.(net.Conn)
			if !ok {
				return rw, nil, fmt.Errorf("TLS startup requires a net.Conn")
			}
			if _, err := conn.Write([]byte{'S'}); err != nil {
				return rw, nil, fmt.Errorf("sending SSL acceptance: %w", err)
			}
			secure := tls.Server(conn, tlsConfig.Clone())
			if err := secure.Handshake(); err != nil {
				return secure, nil, fmt.Errorf("TLS handshake: %w", err)
			}
			rw = secure
			binding, available, bindingErr := tlsServerEndPointBinding(tlsConfig)
			if bindingErr != nil {
				return rw, nil, bindingErr
			}
			config.scramChannelBinding = binding
			config.scramChannelBindingOK = available
			payload, err = readStartupPacket(rw)
			if err != nil {
				return rw, nil, fmt.Errorf("reading startup over TLS: %w", err)
			}
		}
	} else if requireTLS {
		startupErr := fmt.Errorf("TLS is required")
		_ = sendErrorWithCode(rw, "FATAL", "08004", startupErr.Error())
		return rw, nil, startupErr
	}

	result, err := parseStartupPayload(payload)
	if err != nil {
		_ = sendErrorWithCode(rw, "FATAL", "08P01", err.Error())
		return rw, nil, err
	}

	if scramAuthEnabled(config) {
		if result.User == "" {
			startupErr := fmt.Errorf("SCRAM authentication requires a startup user")
			_ = sendErrorWithCode(rw, "FATAL", "28000", startupErr.Error())
			return rw, nil, startupErr
		}
		if config.authMetrics != nil {
			config.authMetrics.attempts.Add(1)
		}
		now := time.Now()
		if config.authLimiter != nil && (!config.authLimiter.allow(config.authClientKey, now) || !config.authLimiter.allowUser(config.authClientKey, result.User, now)) {
			if config.authMetrics != nil {
				config.authMetrics.rateLimited.Add(1)
			}
			startupErr := fmt.Errorf("authentication rate limit exceeded")
			_ = sendErrorWithCode(rw, "FATAL", "28P01", "password authentication failed")
			return rw, nil, startupErr
		}
		admitted := false
		if config.authLimiter != nil {
			if !config.authLimiter.admitAttempt(config.authClientKey, now) {
				if config.authMetrics != nil {
					config.authMetrics.admissionRejected.Add(1)
				}
				startupErr := fmt.Errorf("authentication attempt admission limit exceeded")
				_ = sendErrorWithCode(rw, "FATAL", "28P01", "password authentication failed")
				return rw, nil, startupErr
			}
			admitted = true
			if config.authMetrics != nil {
				config.authMetrics.activeSCRAM.Add(1)
			}
			defer func() {
				if admitted {
					config.authLimiter.releaseAttempt()
					if config.authMetrics != nil {
						config.authMetrics.activeSCRAM.Add(-1)
					}
				}
			}()
		}
		if err := authenticateSCRAMContext(ctx, rw, result.User, config); err != nil {
			if config.authMetrics != nil {
				config.authMetrics.failures.Add(1)
			}
			if shouldRecordAuthFailure(err) {
				config.authLimiter.recordFailureFor(config.authClientKey, result.User, time.Now())
			}
			// Authentication failures terminate startup. Do not emit ReadyForQuery
			// or any normal session messages after a failed exchange.
			// Return the detailed error to the server for diagnostics, but expose
			// one generic protocol error to clients to avoid username/protocol
			// enumeration through startup responses.
			_ = sendErrorWithCode(rw, "FATAL", "28P01", "password authentication failed")
			return rw, nil, err
		}
		if config.authMetrics != nil {
			config.authMetrics.successes.Add(1)
		}
		config.authLimiter.recordSuccessFor(config.authClientKey, result.User)
	} else if err := sendAuthOK(rw); err != nil {
		return rw, nil, err
	}

	if err := sendStartupReady(rw, db); err != nil {
		return rw, nil, err
	}
	return rw, result, nil
}

func shouldRecordAuthFailure(err error) bool {
	if err == nil || errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, net.ErrClosed) || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var networkErr net.Error
	return !errors.As(err, &networkErr)
}

func readStartupPacket(r io.Reader) ([]byte, error) {
	var lenBuf [4]byte
	if _, err := io.ReadFull(r, lenBuf[:]); err != nil {
		return nil, fmt.Errorf("reading startup length: %w", err)
	}
	length := int(binary.BigEndian.Uint32(lenBuf[:]))
	if length < 8 || length > maxStartupPacketBytes {
		return nil, fmt.Errorf("invalid startup packet length: %d", length)
	}
	payload := make([]byte, length-4)
	if _, err := io.ReadFull(r, payload); err != nil {
		return nil, fmt.Errorf("reading startup payload: %w", err)
	}
	return payload, nil
}

func parseStartupPayload(payload []byte) (*StartupResult, error) {
	if len(payload) < 4 {
		return nil, fmt.Errorf("startup payload too short")
	}
	major := int32(binary.BigEndian.Uint32(payload[:4])) >> 16
	if major != 3 {
		return nil, fmt.Errorf("unsupported protocol version %d", major)
	}

	result := &StartupResult{}
	seen := make(map[string]struct{}, 4)
	terminated := false
	offset := 4
	for offset < len(payload) {
		key, next, err := readStartupField(payload, offset)
		if err != nil {
			return nil, err
		}
		offset = next
		if key == "" {
			// The startup parameter list must terminate with an empty key.
			if offset != len(payload) {
				return nil, fmt.Errorf("trailing bytes after startup parameters")
			}
			terminated = true
			break
		}
		if len(seen) >= maxStartupParameters {
			return nil, fmt.Errorf("too many startup parameters")
		}
		if _, exists := seen[key]; exists {
			return nil, fmt.Errorf("duplicate startup parameter %q", key)
		}
		seen[key] = struct{}{}
		if offset >= len(payload) {
			return nil, fmt.Errorf("missing value for startup parameter %q", key)
		}
		val, next, err := readStartupField(payload, offset)
		if err != nil {
			return nil, fmt.Errorf("malformed value for startup parameter %q: %w", key, err)
		}
		offset = next
		if !utf8.ValidString(val) {
			return nil, fmt.Errorf("startup parameter %q is not valid UTF-8", key)
		}
		switch key {
		case "database":
			result.Database = val
		case "user":
			result.User = val
		}
	}
	if !terminated {
		return nil, fmt.Errorf("startup parameters are not NUL terminated")
	}
	if result.User == "" {
		return nil, fmt.Errorf("startup user is required")
	}
	return result, nil
}

func readStartupField(payload []byte, offset int) (string, int, error) {
	if offset < 0 || offset >= len(payload) {
		return "", offset, fmt.Errorf("missing NUL-terminated field")
	}
	relativeEnd := bytes.IndexByte(payload[offset:], 0)
	if relativeEnd < 0 {
		return "", len(payload), fmt.Errorf("missing NUL terminator")
	}
	if relativeEnd > maxStartupFieldBytes {
		return "", len(payload), fmt.Errorf("startup field exceeds %d bytes", maxStartupFieldBytes)
	}
	end := offset + relativeEnd
	return string(payload[offset:end]), end + 1, nil
}

func sendStartupReady(w io.Writer, db *libravdb.Database) error {
	// PostgreSQL exposes the numeric server_version parameter separately from
	// the verbose version() function result. Drivers such as Django's psycopg
	// backend parse this startup value as an integer version.
	if err := sendParameterStatus(w, "server_version", "16.0"); err != nil {
		return err
	}
	if err := sendParameterStatus(w, "client_encoding", "UTF8"); err != nil {
		return err
	}
	if err := sendParameterStatus(w, "server_encoding", "UTF8"); err != nil {
		return err
	}
	// pgx uses this startup parameter before it enables simple-protocol
	// queries (database/sql drivers and ORMs commonly do so for metadata
	// scans). PostgreSQL defaults it to on; advertising the same value keeps
	// the wire contract explicit and prevents clients from rejecting the
	// connection before issuing a query.
	if err := sendParameterStatus(w, "standard_conforming_strings", "on"); err != nil {
		return err
	}
	// This is the exact durable commit position observed when the connection
	// starts. Clients can retain it as a snapshot token and use the existing
	// native SnapshotAtLSN API or the documented temporal SQL surfaces. The
	// SQL function returns the live value when a later refresh is required.
	latestLSN := uint64(0)
	if db != nil {
		if lsn, err := db.LatestCommitLSN(context.Background()); err == nil {
			latestLSN = lsn
		}
	}
	if err := sendParameterStatus(w, "libravdb_latest_commit_lsn", strconv.FormatUint(latestLSN, 10)); err != nil {
		return err
	}
	// BackendKeyData: PID and secret key (zeroed — cancel support is not yet
	// exposed by the server).
	if err := WriteMessage(w, msgBackendKeyData, []byte{
		0, 0, 0, 0,
		0, 0, 0, 0,
	}); err != nil {
		return err
	}
	return WriteMessage(w, msgReadyForQuery, []byte{'I'})
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

// sendErrorWithCode sends an ErrorResponse with explicit severity, SQLSTATE,
// and message.
func sendErrorWithCode(w io.Writer, severity, sqlstate, message string) error {
	var buf []byte
	buf = append(buf, 'S')
	buf = WriteNullTerminated(buf, severity)
	buf = append(buf, 'C')
	buf = WriteNullTerminated(buf, sqlstate)
	buf = append(buf, 'M')
	buf = WriteNullTerminated(buf, message)
	buf = append(buf, 0)
	return WriteMessage(w, msgErrorResponse, buf)
}
