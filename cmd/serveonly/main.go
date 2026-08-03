package main

import (
	"context"
	"log"

	"github.com/xDarkicex/libravdb/internal/pgwire"
	"github.com/xDarkicex/libravdb/libravdb"
)

// serveonly starts the pgwire server and blocks — used with the raw probe.
func main() {
	db, err := libravdb.Open(libravdb.WithStoragePath(":memory:serveonly"), libravdb.WithMetrics(false))
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	srv := pgwire.NewServer(db, pgwire.ServerConfig{Addr: "127.0.0.1:15432"})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		<-ctx.Done()
	}()
	log.Printf("listening on %s", srv.Addr())
	if err := srv.Serve(ctx); err != nil {
		log.Fatal(err)
	}
}
