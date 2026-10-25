package libravdb

import (
	"context"
	"fmt"
	"os"
	osexec "os/exec"
	"path/filepath"
	"sync"
	"testing"
)

// concurrentHelperEnv names the environment variable that turns a test process
// into a writer.
const concurrentHelperEnv = "LIBRAVDB_CONCURRENT_HELPER"

// concurrentRecords is how many records each writer inserts.
const concurrentRecords = 250

// TestConcurrentProcessesKeepEveryRecord checks that two processes writing the
// same store both keep their records.
//
// Without the commit lock, the two processes' frames interleave at the same
// file offset, and one process's compaction renames the file out from under the
// other. Measured on the unfixed engine: 400 of 800 records survived, and the
// compaction failed with a rename error.
func TestConcurrentProcessesKeepEveryRecord(t *testing.T) {
	if os.Getenv(concurrentHelperEnv) == "write" {
		concurrentHelper(t)
		return
	}

	dir := filepath.Join(t.TempDir(), "concurrent.libravdb")
	if fixed := os.Getenv("LIBRAVDB_CONCURRENT_DIR"); fixed != "" {
		dir = filepath.Join(fixed, "concurrent.libravdb")
	}
	ctx := context.Background()

	db, err := Open(WithStoragePath(dir))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if _, err := db.CreateCollection(ctx, "nodes", WithMetadataOnly()); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	var wg sync.WaitGroup
	for _, who := range []string{"A", "B"} {
		wg.Add(1)
		go func(who string) {
			defer wg.Done()
			cmd := osexec.Command(os.Args[0], "-test.run=^TestConcurrentProcessesKeepEveryRecord$", "-test.timeout=30s")
			cmd.Env = append(os.Environ(),
				concurrentHelperEnv+"=write",
				"LIBRAVDB_CONCURRENT_PATH="+dir,
				"LIBRAVDB_CONCURRENT_WHO="+who)
			out, err := cmd.CombinedOutput()
			t.Logf("writer %s: err=%v out=%q", who, err, string(out))
			if err != nil {
				t.Errorf("writer %s: %v\n%s", who, err, out)
			}
		}(who)
	}
	wg.Wait()

	db, err = Open(WithStoragePath(dir))
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer func() { _ = db.Close() }()
	col, err := db.GetCollection("nodes")
	if err != nil {
		t.Fatalf("GetCollection: %v", err)
	}
	records, err := col.ListAll(ctx)
	if err != nil {
		t.Fatalf("ListAll: %v", err)
	}
	if len(records) != 2*concurrentRecords {
		seen := map[string]int{}
		for _, r := range records {
			seen[r.ID[:1]]++
		}
		t.Errorf("records = %d, want %d — a writer lost its records (A=%d B=%d)",
			len(records), 2*concurrentRecords, seen["A"], seen["B"])
	}
}

// concurrentHelper is one writer process.
func concurrentHelper(t *testing.T) {
	t.Helper()
	ctx := context.Background()
	db, err := Open(WithStoragePath(os.Getenv("LIBRAVDB_CONCURRENT_PATH")))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer func() { _ = db.Close() }()
	col, err := db.GetCollection("nodes")
	if err != nil {
		t.Fatalf("GetCollection: %v", err)
	}
	who := os.Getenv("LIBRAVDB_CONCURRENT_WHO")
	for i := range concurrentRecords {
		if i == concurrentRecords/2 {
			if err := db.Vacuum(ctx); err != nil {
				t.Fatalf("mid-write vacuum: %v", err)
			}
		}
		id := fmt.Sprintf("%s-%06d", who, i)
		if err := col.Insert(ctx, id, nil, map[string]interface{}{"k": "v"}); err != nil {
			t.Fatalf("insert %s: %v", id, err)
		}
	}
}

func TestLiveIndexFollowsRemoteVacuum(t *testing.T) {
	for _, tc := range []struct {
		name   string
		option CollectionOption
	}{
		{"flat", WithFlat()}, {"hnsw", WithHNSW(16, 100, 50)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			path := filepath.Join(t.TempDir(), "live.db")
			a, err := Open(WithStoragePath(path), WithMetrics(false))
			if err != nil {
				t.Fatal(err)
			}
			defer a.Close()
			ca, err := a.CreateCollection(ctx, "rows", WithDimension(2), tc.option)
			if err != nil {
				t.Fatal(err)
			}
			if err := ca.Insert(ctx, "original", []float32{1, 0}, nil); err != nil {
				t.Fatal(err)
			}
			b, err := Open(WithStoragePath(path), WithMetrics(false))
			if err != nil {
				t.Fatal(err)
			}
			defer b.Close()
			cb, err := b.GetCollection("rows")
			if err != nil {
				t.Fatal(err)
			}
			if err := ca.Insert(ctx, "remote", []float32{0, 1}, nil); err != nil {
				t.Fatal(err)
			}
			if err := a.Vacuum(ctx); err != nil {
				t.Fatal(err)
			}
			// The next write refreshes b's storage and its existing live vector index.
			if err := cb.Insert(ctx, "local", []float32{1, 1}, nil); err != nil {
				t.Fatal(err)
			}
			results, err := cb.Search(ctx, []float32{0, 1}, 3)
			if err != nil {
				t.Fatal(err)
			}
			found := false
			for _, result := range results.Results {
				if result.ID == "remote" {
					found = true
				}
			}
			if !found {
				t.Fatalf("remote record missing from live index after vacuum: %v", results.Results)
			}
		})
	}
}
