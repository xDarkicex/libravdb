package pgwire

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net"
	"sync"
	"time"

	"github.com/xDarkicex/libravdb/libravdb"
)

const (
	// DefaultMaxConnections bounds the number of sockets and worker goroutines
	// a server accepts when MaxConnections is not explicitly configured.
	DefaultMaxConnections = 256

	// DefaultStartupTimeout bounds the complete pre-authentication exchange,
	// including SSL negotiation, the TLS handshake, startup parsing, and SCRAM.
	DefaultStartupTimeout = 30 * time.Second

	// DefaultPasswordLookupTimeout bounds a context-aware external credential
	// lookup. The callback must honor context cancellation.
	DefaultPasswordLookupTimeout = 5 * time.Second

	// DefaultIdleTimeout bounds the time a steady-state connection may wait for
	// its next frontend message. It is deliberately long enough for normal
	// PostgreSQL connection pools while preventing permanent slot retention.
	DefaultIdleTimeout = 30 * time.Minute

	// These limits are per connection. They are generous for normal PostgreSQL
	// drivers while preventing unbounded extended-protocol state growth.
	DefaultMaxPreparedStatements = 1024
	DefaultMaxPortals            = 1024
	// The byte defaults match the normal frontend message ceiling so common
	// large PostgreSQL parameters remain compatible; deployments may lower them.
	DefaultMaxPreparedStatementBytes = 1 << 24
	DefaultMaxPortalBytes            = 1 << 24
)

// ServerConfig configures the pgwire protocol server.
type ServerConfig struct {
	// Addr is the listen address (e.g., ":5432" or "127.0.0.1:5432").
	Addr string

	// MaxConnections limits concurrent connections. Zero selects
	// DefaultMaxConnections. Negative values are rejected by Serve.
	MaxConnections int

	// StartupTimeout bounds the pre-authentication exchange. Zero selects
	// DefaultStartupTimeout. Negative values are rejected by Serve.
	StartupTimeout time.Duration

	// IdleTimeout bounds the wait for the next frontend message after startup.
	// Zero selects DefaultIdleTimeout. Negative values are rejected by Serve.
	IdleTimeout time.Duration

	// MaxPreparedStatements and MaxPortals bound extended-protocol objects per
	// connection. Zero selects the corresponding default. Existing objects may
	// still be replaced or closed at the limit for driver compatibility.
	MaxPreparedStatements int
	MaxPortals            int
	// MaxPreparedStatementBytes and MaxPortalBytes bound the query text and
	// bind payload for each newly-created object. Zero selects the default.
	MaxPreparedStatementBytes int
	MaxPortalBytes            int

	// TLSConfig enables PostgreSQL SSLRequest negotiation and TLS. The config
	// is cloned before use. If TLSCertificateFile/TLSKeyFile are provided they
	// are loaded into this config at Serve time.
	TLSConfig *tls.Config
	// TLSCertificateFile and TLSKeyFile optionally load a server certificate.
	TLSCertificateFile string
	TLSKeyFile         string
	// RequireTLS rejects plaintext startup packets. It requires a valid TLS
	// certificate/configuration.
	RequireTLS bool
	// AllowInsecure explicitly permits a non-loopback listener without TLS or
	// authentication. Loopback-only development listeners remain compatible by
	// default; public listeners must opt into this unsafe mode deliberately.
	AllowInsecure bool

	// RequireAuthentication enables SCRAM-SHA-256. SCRAM requires RequireTLS
	// so credentials and the authenticated session are not exposed on a plain
	// socket. Credentials are stored as salted SCRAM verifiers, never plaintext
	// passwords. PasswordLookup may be used for an external credential store
	// and must be concurrency-safe.
	RequireAuthentication bool
	// RequireChannelBinding requires clients to select
	// SCRAM-SHA-256-PLUS over TLS. When false, TLS connections advertise both
	// SCRAM-SHA-256-PLUS and SCRAM-SHA-256, allowing PostgreSQL drivers to use
	// their channel_binding=prefer/default policy.
	RequireChannelBinding bool
	Credentials           map[string]SCRAMCredential
	PasswordLookup        func(username string) (SCRAMCredential, bool)
	PasswordLookupContext func(context.Context, string) (SCRAMCredential, bool)
	// PasswordLookupTimeout applies to PasswordLookupContext. Legacy
	// PasswordLookup callbacks cannot be forcefully interrupted and should be
	// migrated when the credential source may block.
	PasswordLookupTimeout time.Duration
	// AuthFailureThreshold and AuthLockoutDuration control per-source failed
	// authentication handling. Zero selects secure defaults.
	AuthFailureThreshold int
	AuthLockoutDuration  time.Duration
	// AuthGlobalFailureLimit and AuthGlobalFailureWindow control the shared
	// circuit breaker for failed authentication across all sources. Zero selects
	// secure defaults.
	AuthGlobalFailureLimit  int
	AuthGlobalFailureWindow time.Duration
	// AuthAttemptBurst and AuthAttemptRefill bound authentication attempts per
	// source before SCRAM work begins. MaxConcurrentAuth bounds active SCRAM
	// handshakes. Zero selects secure defaults.
	AuthAttemptBurst  int
	AuthAttemptRefill time.Duration
	MaxConcurrentAuth int

	// ProxyProtocol enables the opt-in PROXY v1 header before SSLRequest. It is
	// intended only for deployments behind a trusted TCP proxy. Every proxy
	// address must be listed in TrustedProxyCIDRs; arbitrary forwarding headers
	// are never accepted.
	ProxyProtocol     bool
	TrustedProxyCIDRs []string

	// scramUnknownCredential is generated once per server and used for mock
	// SCRAM exchanges for nonexistent users. It is intentionally private so
	// callers cannot provide a verifier that changes the server's anti-enumeration
	// behavior.
	scramUnknownCredential *SCRAMCredential
	scramChannelBinding    []byte
	scramChannelBindingOK  bool
	authLimiter            *authFailureLimiter
	authMetrics            *authMetrics
	trustedProxyNetworks   []*net.IPNet
	authClientKey          string
}

// Server is a PostgreSQL wire protocol listener that exposes a libravdb.Database
// to any PostgreSQL-compatible client.
type Server struct {
	db     *libravdb.Database
	config ServerConfig

	mu          sync.Mutex
	ln          net.Listener
	conns       map[net.Conn]struct{}
	connSem     chan struct{} // semaphore for MaxConnections
	authLimiter *authFailureLimiter
	authMetrics *authMetrics
	closed      bool
}

func configuredLimit(value, fallback int) int {
	if value == 0 {
		return fallback
	}
	return value
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
		authLimiter: newAuthFailureLimiterWithPolicy(config.AuthFailureThreshold, config.AuthLockoutDuration, authAttemptPolicy{
			burst:               config.AuthAttemptBurst,
			refill:              config.AuthAttemptRefill,
			maxConcurrent:       config.MaxConcurrentAuth,
			globalFailureLimit:  config.AuthGlobalFailureLimit,
			globalFailureWindow: config.AuthGlobalFailureWindow,
		}),
		authMetrics: &authMetrics{},
	}
	if config.TLSConfig != nil {
		s.config.TLSConfig = config.TLSConfig.Clone()
	}
	if config.Credentials != nil {
		s.config.Credentials = cloneSCRAMCredentials(config.Credentials)
	}
	if config.TrustedProxyCIDRs != nil {
		s.config.TrustedProxyCIDRs = append([]string(nil), config.TrustedProxyCIDRs...)
	}
	maxConnections := config.MaxConnections
	if maxConnections == 0 {
		maxConnections = DefaultMaxConnections
	}
	if maxConnections > 0 {
		s.connSem = make(chan struct{}, maxConnections)
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

// AuthStats returns a race-free snapshot of authentication events. The
// counters contain no usernames, addresses, or credential material.
func (s *Server) AuthStats() AuthStats {
	s.mu.Lock()
	metrics := s.authMetrics
	s.mu.Unlock()
	return metrics.snapshot()
}

// Serve starts listening and blocks until the context is cancelled or the
// listener encounters a fatal error. Returns when all connections have
// been drained.
func (s *Server) Serve(ctx context.Context) error {
	if s.config.MaxConnections < 0 {
		return fmt.Errorf("pgwire MaxConnections must not be negative")
	}
	if s.config.StartupTimeout < 0 {
		return fmt.Errorf("pgwire StartupTimeout must not be negative")
	}
	if s.config.IdleTimeout < 0 {
		return fmt.Errorf("pgwire IdleTimeout must not be negative")
	}
	if s.config.MaxPreparedStatements < 0 {
		return fmt.Errorf("pgwire MaxPreparedStatements must not be negative")
	}
	if s.config.MaxPortals < 0 {
		return fmt.Errorf("pgwire MaxPortals must not be negative")
	}
	if s.config.MaxPreparedStatementBytes < 0 {
		return fmt.Errorf("pgwire MaxPreparedStatementBytes must not be negative")
	}
	if s.config.MaxPortalBytes < 0 {
		return fmt.Errorf("pgwire MaxPortalBytes must not be negative")
	}
	if s.config.AuthFailureThreshold < 0 {
		return fmt.Errorf("pgwire AuthFailureThreshold must not be negative")
	}
	if s.config.AuthLockoutDuration < 0 {
		return fmt.Errorf("pgwire AuthLockoutDuration must not be negative")
	}
	if s.config.AuthGlobalFailureLimit < 0 {
		return fmt.Errorf("pgwire AuthGlobalFailureLimit must not be negative")
	}
	if s.config.AuthGlobalFailureWindow < 0 {
		return fmt.Errorf("pgwire AuthGlobalFailureWindow must not be negative")
	}
	if s.config.PasswordLookupTimeout < 0 {
		return fmt.Errorf("pgwire PasswordLookupTimeout must not be negative")
	}
	if s.config.RequireChannelBinding && !s.config.RequireTLS {
		return fmt.Errorf("pgwire RequireChannelBinding requires RequireTLS")
	}
	if s.config.AuthAttemptBurst < 0 {
		return fmt.Errorf("pgwire AuthAttemptBurst must not be negative")
	}
	if s.config.AuthAttemptRefill < 0 {
		return fmt.Errorf("pgwire AuthAttemptRefill must not be negative")
	}
	if s.config.MaxConcurrentAuth < 0 {
		return fmt.Errorf("pgwire MaxConcurrentAuth must not be negative")
	}
	if !s.config.ProxyProtocol && len(s.config.TrustedProxyCIDRs) > 0 {
		return fmt.Errorf("TrustedProxyCIDRs requires ProxyProtocol")
	}
	if s.config.ProxyProtocol {
		networks, err := parseTrustedProxyCIDRs(s.config.TrustedProxyCIDRs)
		if err != nil {
			return err
		}
		s.config.trustedProxyNetworks = networks
	}
	if err := validateTransportSecurity(s.config); err != nil {
		return err
	}
	tlsConfig, err := s.buildTLSConfig()
	if err != nil {
		return err
	}
	if s.config.RequireChannelBinding {
		if tlsConfig == nil {
			return fmt.Errorf("pgwire RequireChannelBinding requires a TLS configuration")
		}
		if _, available, err := tlsServerEndPointBinding(tlsConfig); err != nil {
			return fmt.Errorf("pgwire channel binding: %w", err)
		} else if !available {
			return fmt.Errorf("pgwire RequireChannelBinding needs a static TLS server certificate")
		}
	}
	if err := validateSCRAMConfig(s.config); err != nil {
		return err
	}
	if err := ensureSCRAMUnknownCredential(&s.config); err != nil {
		return err
	}
	lc := net.ListenConfig{}
	ln, err := lc.Listen(ctx, "tcp", s.config.Addr)
	if err != nil {
		return fmt.Errorf("pgwire listen on %s: %w", s.config.Addr, err)
	}

	s.mu.Lock()
	s.ln = ln
	s.mu.Unlock()
	listenerDone := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			_ = ln.Close()
		case <-listenerDone:
		}
	}()
	defer close(listenerDone)

	// Accept loop
	var wg sync.WaitGroup
	for {
		conn, err := ln.Accept()
		if err != nil {
			s.mu.Lock()
			closed := s.closed
			s.mu.Unlock()
			if closed {
				s.closeActiveConnections()
				wg.Wait()
				return nil
			}
			select {
			case <-ctx.Done():
				// Graceful shutdown: actively close connections so a client
				// blocked in startup or a query cannot hold Serve open forever.
				s.closeActiveConnections()
				wg.Wait()
				return ctx.Err()
			default:
				// Treat an unexpected accept failure as terminal, but still
				// drain every connection already handed to a worker. Without
				// this cleanup, an EMFILE/resource failure can leave live
				// sockets and goroutines behind after Serve returns.
				_ = ln.Close()
				s.closeActiveConnections()
				wg.Wait()
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

			s.handleConn(ctx, conn, tlsConfig)
		}()
	}
}

func validateTransportSecurity(config ServerConfig) error {
	if config.RequireTLS || scramAuthEnabled(config) || config.AllowInsecure || isLoopbackListenAddress(config.Addr) {
		return nil
	}
	return fmt.Errorf("refusing insecure unauthenticated pgwire listener on non-loopback address %q; configure RequireTLS/authentication or explicitly set AllowInsecure", config.Addr)
}

func isLoopbackListenAddress(addr string) bool {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return false
	}
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// Close stops the server and closes all active connections.
func (s *Server) Close() error {
	s.mu.Lock()
	s.closed = true
	ln := s.ln
	conns := make([]net.Conn, 0, len(s.conns))
	for conn := range s.conns {
		conns = append(conns, conn)
	}
	s.mu.Unlock()

	var closeErr error
	if ln != nil {
		if err := ln.Close(); err != nil && !errors.Is(err, net.ErrClosed) {
			closeErr = err
		}
	}
	for _, conn := range conns {
		_ = conn.Close()
	}
	return closeErr
}

func (s *Server) closeActiveConnections() {
	s.mu.Lock()
	conns := make([]net.Conn, 0, len(s.conns))
	for conn := range s.conns {
		conns = append(conns, conn)
	}
	s.mu.Unlock()
	for _, conn := range conns {
		_ = conn.Close()
	}
}

// handleConn processes a single client connection.
func (s *Server) handleConn(ctx context.Context, conn net.Conn, tlsConfig *tls.Config) {
	startupTimeout := s.config.StartupTimeout
	if startupTimeout == 0 {
		startupTimeout = DefaultStartupTimeout
	}
	// Set one absolute deadline for the complete startup exchange. This limits
	// slowloris clients even when they send a byte just before each read would
	// otherwise time out, and the deadline also covers the TLS handshake.
	if err := conn.SetDeadline(time.Now().Add(startupTimeout)); err != nil {
		return
	}
	startupConn := conn
	if s.config.ProxyProtocol {
		forwardedAddr, err := readPROXYv1(conn, s.config.trustedProxyNetworks)
		if err != nil {
			return
		}
		startupConn = &proxyConn{Conn: conn, remote: forwardedAddr}
	}

	// Startup handshake
	startupConfig := s.config
	startupConfig.authLimiter = s.authLimiter
	startupConfig.authMetrics = s.authMetrics
	startupConfig.authClientKey = authClientKey(startupConn)
	rw, _, err := handleStartupWithConfigContext(ctx, startupConn, s.db, tlsConfig, s.config.RequireTLS, startupConfig)
	if err != nil {
		// Already sent error to client in handleStartup
		return
	}

	// Clear deadline — the connection is now in steady state
	if deadlineConn, ok := rw.(net.Conn); ok {
		if err := deadlineConn.SetDeadline(time.Time{}); err != nil {
			return
		}
	}

	// Extended query protocol state and optional epoch transaction.
	state := newConnState()
	state.maxPreparedStatements = configuredLimit(s.config.MaxPreparedStatements, DefaultMaxPreparedStatements)
	state.maxPortals = configuredLimit(s.config.MaxPortals, DefaultMaxPortals)
	state.maxPreparedStatementBytes = configuredLimit(s.config.MaxPreparedStatementBytes, DefaultMaxPreparedStatementBytes)
	state.maxPortalBytes = configuredLimit(s.config.MaxPortalBytes, DefaultMaxPortalBytes)
	// Rollback any active epoch on connection close.
	defer state.rollbackEpoch()

	// Arena-backed buffer pool for zero-heap message I/O
	arena, err := newConnArena()
	if err != nil {
		return
	}
	defer arena.close()

	// Message processing loop
	readDeadlineConn, hasReadDeadline := rw.(interface{ SetReadDeadline(time.Time) error })
	idleTimeout := s.config.IdleTimeout
	if idleTimeout == 0 {
		idleTimeout = DefaultIdleTimeout
	}
	for {
		// Reset arena between messages — each message is processed immediately,
		// so one message's worth of arena memory is sufficient.
		arena.reset()

		if hasReadDeadline {
			if err := readDeadlineConn.SetReadDeadline(time.Now().Add(idleTimeout)); err != nil {
				return
			}
		}
		msgType, payload, err := readMessageArena(rw, arena)
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
			// Check for COPY ... FROM STDIN / TO STDOUT — enter copy mode
			if isCopy(query) {
				if err := handleCopy(rw, arena, s.db, state, query); err != nil {
					return
				}
			} else {
				if err := handleQuery(rw, s.db, state, query); err != nil {
					return
				}
			}

		case msgTerminate:
			return

		case msgParse, msgBind, msgDescribe, msgExecute, msgSync, msgClose, msgFlush:
			cont, err := handleExtendedMessage(rw, s.db, state, msgType, payload)
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
