package pgwire

import (
	"context"
	"fmt"
	"io"
	"net"
	"sync"
	"time"

	"github.com/xDarkicex/libravdb/libravdb"
)

// ServerConfig configures the pgwire protocol server.
type ServerConfig struct {
	// Addr is the listen address (e.g., ":5432" or "127.0.0.1:5432").
	Addr string

	// MaxConnections limits concurrent connections. Zero means no limit.
	MaxConnections int
}

// Server is a PostgreSQL wire protocol listener that exposes a libravdb.Database
// to any PostgreSQL-compatible client.
type Server struct {
	db     *libravdb.Database
	config ServerConfig

	mu      sync.Mutex
	ln      net.Listener
	conns   map[net.Conn]struct{}
	connSem chan struct{} // semaphore for MaxConnections
	closed  bool
}

// NewServer creates a pgwire protocol server for the given database.
func NewServer(db *libravdb.Database, config ServerConfig) *Server {
	if config.Addr == "" {
		config.Addr = ":5432"
	}
	s := &Server{
		db:     db,
		config: config,
		conns:  make(map[net.Conn]struct{}),
	}
	if config.MaxConnections > 0 {
		s.connSem = make(chan struct{}, config.MaxConnections)
	}
	return s
}

// Addr returns the address the server is listening on.
// Returns empty string if not listening.
func (s *Server) Addr() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ln == nil {
		return ""
	}
	return s.ln.Addr().String()
}

// Serve starts listening and blocks until the context is cancelled or the
// listener encounters a fatal error. Returns when all connections have
// been drained.
func (s *Server) Serve(ctx context.Context) error {
	lc := net.ListenConfig{}
	ln, err := lc.Listen(ctx, "tcp", s.config.Addr)
	if err != nil {
		return fmt.Errorf("pgwire listen on %s: %w", s.config.Addr, err)
	}

	s.mu.Lock()
	s.ln = ln
	s.mu.Unlock()

	// Accept loop
	var wg sync.WaitGroup
	for {
		conn, err := ln.Accept()
		if err != nil {
			select {
			case <-ctx.Done():
				// Graceful shutdown: wait for connections to drain
				wg.Wait()
				return ctx.Err()
			default:
				return fmt.Errorf("pgwire accept: %w", err)
			}
		}

		// Connection limit
		if s.connSem != nil {
			select {
			case s.connSem <- struct{}{}:
			default:
				conn.Close()
				continue
			}
		}

		s.mu.Lock()
		s.conns[conn] = struct{}{}
		s.mu.Unlock()

		wg.Add(1)
		go func() {
			defer wg.Done()
			defer conn.Close()
			defer func() {
				s.mu.Lock()
				delete(s.conns, conn)
				s.mu.Unlock()
				if s.connSem != nil {
					<-s.connSem
				}
			}()

			s.handleConn(ctx, conn)
		}()
	}
}

// Close stops the server and closes all active connections.
func (s *Server) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	if s.ln != nil {
		if err := s.ln.Close(); err != nil {
			return err
		}
	}
	for conn := range s.conns {
		conn.Close()
	}
	return nil
}

// handleConn processes a single client connection.
func (s *Server) handleConn(ctx context.Context, conn net.Conn) {
	// Set a reasonable deadline for the startup handshake
	if err := conn.SetDeadline(time.Now().Add(30 * time.Second)); err != nil {
		return
	}

	// Startup handshake
	_, err := handleStartup(conn, s.db)
	if err != nil {
		// Already sent error to client in handleStartup
		return
	}

	// Clear deadline — the connection is now in steady state
	if err := conn.SetDeadline(time.Time{}); err != nil {
		return
	}

	// Extended query protocol state and optional epoch transaction.
	state := newConnState()
	// Rollback any active epoch on connection close.
	defer state.rollbackEpoch()

	// Arena-backed buffer pool for zero-heap message I/O
	arena, err := newConnArena()
	if err != nil {
		return
	}
	defer arena.close()

	// Message processing loop
	for {
		// Reset arena between messages — each message is processed immediately,
		// so one message's worth of arena memory is sufficient.
		arena.reset()

		msgType, payload, err := readMessageArena(conn, arena)
		if err != nil {
			if err != io.EOF {
				// Log? For now, just close.
			}
			return
		}

		switch msgType {
		case msgQuery:
			// Simple Query: null-terminated SQL string
			query := ""
			if len(payload) > 0 && payload[len(payload)-1] == 0 {
				query = string(payload[:len(payload)-1])
			} else {
				query = string(payload)
			}
			// Check for COPY ... FROM STDIN — enter copy-in mode
			if isCopyIn(query) {
				if err := handleCopyIn(conn, arena, s.db, query); err != nil {
					return
				}
			} else {
				if err := handleQuery(conn, s.db, state, query); err != nil {
					return
				}
			}

		case msgTerminate:
			return

		case msgParse, msgBind, msgDescribe, msgExecute, msgSync, msgClose, msgFlush:
			cont, err := handleExtendedMessage(conn, s.db, state, msgType, payload)
			if err != nil {
				return
			}
			if !cont {
				return
			}

		default:
			// Unknown message type — ignore and continue
		}
	}
}
