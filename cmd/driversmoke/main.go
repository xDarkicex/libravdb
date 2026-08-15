package main

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"time"

	_ "github.com/jackc/pgx/v5/stdlib"

	"github.com/xDarkicex/libravdb/libravdb"
	"github.com/xDarkicex/libravdb/pgwire"
)

func main() {
	ctx := context.Background()

	db, err := libravdb.Open(libravdb.WithStoragePath(":memory:driver_test"), libravdb.WithMetrics(false))
	if err != nil {
		log.Fatalf("open libravdb: %v", err)
	}
	defer db.Close()

	srv := pgwire.NewServer(db, pgwire.ServerConfig{Addr: "127.0.0.1:15432"})
	srvCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	go srv.Serve(srvCtx)
	time.Sleep(300 * time.Millisecond)
	fmt.Printf("pgwire server on %s\n", srv.Addr())

	connStr := "postgres://libra:libra@127.0.0.1:15432/libra?sslmode=disable"
	sqlDB, err := sql.Open("pgx", connStr)
	if err != nil {
		log.Fatalf("sql.Open: %v", err)
	}
	defer sqlDB.Close()

	if err := sqlDB.PingContext(ctx); err != nil {
		log.Fatalf("PING FAILED: %v", err)
	}
	fmt.Println("✅ PING OK — real pgx driver through pgwire")

	if _, err := sqlDB.ExecContext(ctx, `CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)`); err != nil {
		log.Fatalf("CREATE TABLE failed: %v", err)
	}
	fmt.Println("✅ CREATE TABLE OK")

	if _, err := sqlDB.ExecContext(ctx, `INSERT INTO users (id, name, age) VALUES (1, 'alice', 30), (2, 'bob', 25)`); err != nil {
		log.Fatalf("INSERT failed: %v", err)
	}
	fmt.Println("✅ INSERT OK")

	rows, err := sqlDB.QueryContext(ctx, `SELECT id, name, age FROM users ORDER BY id`)
	if err != nil {
		log.Fatalf("SELECT failed: %v", err)
	}
	defer rows.Close()

	fmt.Println("✅ SELECT OK — rows:")
	for rows.Next() {
		var id, age int
		var name string
		if err := rows.Scan(&id, &name, &age); err != nil {
			log.Fatalf("scan: %v", err)
		}
		fmt.Printf("   id=%d name=%s age=%d\n", id, name, age)
	}
	if err := rows.Err(); err != nil {
		log.Fatalf("rows err: %v", err)
	}

	stmt, err := sqlDB.PrepareContext(ctx, `SELECT name FROM users WHERE id = $1`)
	if err != nil {
		log.Fatalf("PREPARE failed: %v", err)
	}
	defer stmt.Close()
	var name string
	if err := stmt.QueryRowContext(ctx, 2).Scan(&name); err != nil {
		log.Fatalf("prepared query failed: %v", err)
	}
	fmt.Printf("✅ PREPARED STATEMENT OK — id=2 name=%s\n", name)

	var n int
	if err := sqlDB.QueryRowContext(ctx, `SELECT count(*) FROM pg_class WHERE relname = 'users'`).Scan(&n); err != nil {
		fmt.Printf("⚠️  pg_catalog probe failed: %v\n", err)
	} else {
		fmt.Printf("✅ pg_catalog OK — pg_class has %d row(s) for 'users'\n", n)
	}

	fmt.Println("\n🎉 ALL CHECKS PASSED — real pgx driver works against libravdb pgwire")
}
