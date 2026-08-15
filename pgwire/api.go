// Package pgwire is the public PostgreSQL wire-protocol entry point for
// libraVDB.
//
// The implementation lives in internal/pgwire so protocol internals remain
// private, but applications must be able to start a server without importing
// a Go internal package. This facade is the supported external API.
package pgwire

import (
	"context"

	internalpgwire "github.com/xDarkicex/libravdb/internal/pgwire"
	"github.com/xDarkicex/libravdb/libravdb"
)

// Server is a PostgreSQL wire-protocol listener exposing a libravDB database.
type Server = internalpgwire.Server

// ServerConfig configures a PostgreSQL wire-protocol listener.
type ServerConfig = internalpgwire.ServerConfig

// SCRAMCredential is a salted SCRAM-SHA-256 verifier. It contains no
// plaintext password.
type SCRAMCredential = internalpgwire.SCRAMCredential

// AuthStats is a race-free snapshot of authentication counters.
type AuthStats = internalpgwire.AuthStats

const (
	DefaultMaxConnections            = internalpgwire.DefaultMaxConnections
	DefaultStartupTimeout            = internalpgwire.DefaultStartupTimeout
	DefaultPasswordLookupTimeout     = internalpgwire.DefaultPasswordLookupTimeout
	DefaultIdleTimeout               = internalpgwire.DefaultIdleTimeout
	DefaultMaxPreparedStatements     = internalpgwire.DefaultMaxPreparedStatements
	DefaultMaxPortals                = internalpgwire.DefaultMaxPortals
	DefaultMaxPreparedStatementBytes = internalpgwire.DefaultMaxPreparedStatementBytes
	DefaultMaxPortalBytes            = internalpgwire.DefaultMaxPortalBytes
	DefaultSCRAMIterations           = internalpgwire.DefaultSCRAMIterations

	OIDInt2   = internalpgwire.OIDInt2
	OIDInt4   = internalpgwire.OIDInt4
	OIDInt8   = internalpgwire.OIDInt8
	OIDFloat4 = internalpgwire.OIDFloat4
	OIDFloat8 = internalpgwire.OIDFloat8
	OIDText   = internalpgwire.OIDText
	OIDBool   = internalpgwire.OIDBool
)

// NewServer creates a public PostgreSQL wire-protocol server for db.
func NewServer(db *libravdb.Database, config ServerConfig) *Server {
	return internalpgwire.NewServer(db, config)
}

// NewSCRAMCredential derives a strict PRECIS-prepared SCRAM verifier.
func NewSCRAMCredential(username, password string) (SCRAMCredential, error) {
	return internalpgwire.NewSCRAMCredential(username, password)
}

// NewSCRAMCredentialWithIterations derives a strict PRECIS-prepared SCRAM
// verifier with an explicit PBKDF2 iteration count.
func NewSCRAMCredentialWithIterations(username, password string, iterations int) (SCRAMCredential, error) {
	return internalpgwire.NewSCRAMCredentialWithIterations(username, password, iterations)
}

// Serve starts a PostgreSQL wire-protocol server and blocks until ctx is
// cancelled or the server is closed.
func Serve(ctx context.Context, db *libravdb.Database, config ServerConfig) error {
	return NewServer(db, config).Serve(ctx)
}

// PGTypeName returns the PostgreSQL name for a wire type OID.
func PGTypeName(oid uint32) string {
	return internalpgwire.PGTypeName(oid)
}
