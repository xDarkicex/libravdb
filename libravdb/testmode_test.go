package libravdb

import (
	"os"
	"testing"
)

const stressModeEnv = "LIBRAVDB_RUN_STRESS"

// requireStressMode skips tests that are intentionally long-running or soak-style
// unless the caller explicitly opts in.
func requireStressMode(t *testing.T) {
	t.Helper()
	if testing.Short() {
		t.Skip("Skipping stress test in short mode")
	}
	if os.Getenv(stressModeEnv) != "1" {
		t.Skip("Skipping stress test; set LIBRAVDB_RUN_STRESS=1 to run")
	}
}
