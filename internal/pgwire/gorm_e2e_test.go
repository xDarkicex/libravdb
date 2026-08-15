package pgwire

import (
	"context"
	"fmt"
	"net"
	"testing"
	"time"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"github.com/xDarkicex/libravdb/libravdb"
)

// gormUser intentionally uses GORM's conventional auto-incrementing integer
// primary key. This exercises the schema and DML shape emitted by ordinary
// GORM applications instead of hiding compatibility issues behind a hand-
// authored SQL schema.
type gormUser struct {
	ID        uint `gorm:"primaryKey"`
	Name      string
	Age       int
	CreatedAt time.Time
	UpdatedAt time.Time
}

func TestGORMRealDriverEndToEnd(t *testing.T) {
	db, err := libravdb.Open(
		libravdb.WithStoragePath(":memory:gorm_e2e"),
		libravdb.WithMetrics(false),
	)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	srv := startTestServer(t, db)
	defer srv.Close()
	host, port, err := net.SplitHostPort(srv.Addr())
	if err != nil {
		t.Fatalf("server address %q: %v", srv.Addr(), err)
	}

	dsn := fmt.Sprintf("host=%s port=%s user=test password=test dbname=test sslmode=disable", host, port)
	gdb, err := gorm.Open(postgres.Open(dsn), &gorm.Config{
		Logger: logger.Default.LogMode(logger.Silent),
	})
	if err != nil {
		t.Fatalf("gorm.Open: %v", err)
	}

	if err := gdb.AutoMigrate(&gormUser{}); err != nil {
		t.Fatalf("AutoMigrate: %v", err)
	}

	user := &gormUser{Name: "alice", Age: 30}
	if err := gdb.Create(user).Error; err != nil {
		t.Fatalf("Create: %v", err)
	}
	if user.ID == 0 {
		t.Fatalf("Create did not populate auto-increment ID")
	}

	var got gormUser
	if err := gdb.First(&got, user.ID).Error; err != nil {
		t.Fatalf("First: %v", err)
	}
	if got.Name != "alice" || got.Age != 30 {
		t.Fatalf("First returned %#v", got)
	}

	if err := gdb.Model(&got).Update("age", 31).Error; err != nil {
		t.Fatalf("Update: %v", err)
	}
	if err := gdb.First(&got, user.ID).Error; err != nil {
		t.Fatalf("First after Update: %v", err)
	}
	if got.Age != 31 {
		t.Fatalf("updated age=%d, want 31", got.Age)
	}

	if err := gdb.Delete(&got).Error; err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if err := gdb.First(&gormUser{}, user.ID).Error; err == nil {
		t.Fatalf("First after Delete unexpectedly succeeded")
	}

	var count int64
	if err := gdb.WithContext(context.Background()).Model(&gormUser{}).Count(&count).Error; err != nil {
		t.Fatalf("post-CRUD Count: %v", err)
	}
	if count != 0 {
		t.Fatalf("post-CRUD row count=%d, want 0", count)
	}
}
