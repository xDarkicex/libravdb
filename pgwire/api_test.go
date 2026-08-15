package pgwire_test

import (
	"testing"

	"github.com/xDarkicex/libravdb/libravdb"
	"github.com/xDarkicex/libravdb/pgwire"
)

func TestPublicServerAPI(t *testing.T) {
	var db *libravdb.Database
	server := pgwire.NewServer(db, pgwire.ServerConfig{Addr: "127.0.0.1:0"})
	if server == nil {
		t.Fatal("NewServer returned nil")
	}
	if got := server.Addr(); got != "" {
		t.Fatalf("unstarted server reported address %q", got)
	}
}
