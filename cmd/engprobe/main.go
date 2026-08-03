package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/xDarkicex/libravdb/libravdb"
)

func main() {
	db, err := libravdb.Open(libravdb.WithStoragePath(":memory:insert_probe"), libravdb.WithMetrics(false))
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	ctx := context.Background()
	queries := []string{
		`CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)`,
		`INSERT INTO users (id, name, age) VALUES (1, 'alice', 30)`,
		`INSERT INTO users VALUES (2, 'bob', 25)`,
		`INSERT INTO users (id, name, age) VALUES (3, 'carol', 28), (4, 'dave', 35)`,
		`SELECT id, name, age FROM users`,
		`SELECT * FROM users`,
	}
	for _, q := range queries {
		start := time.Now()
		res, err := db.Query(ctx, q)
		fmt.Printf("Query(%q)\n  -> err=%v took=%v total=%d\n", q, err, time.Since(start), resTotal(res))
	}
}

func resTotal(r *libravdb.SearchResults) int {
	if r == nil {
		return -1
	}
	return r.Total
}
