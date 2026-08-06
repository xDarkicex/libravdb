package libravdb

// Temporary diagnostic for the M3c exact=0.0 anomaly. DELETE AFTER USE.
import (
	"context"
	"fmt"
	"testing"
	"time"
)

func TestM3cDiag(t *testing.T) {
	for _, cfg := range []struct {
		n, dim int
		sel    float64
	}{
		{1000, 32, 0.50},
		{5000, 32, 0.50},
		{20000, 32, 0.50},
		{20000, 512, 0.50},
	} {
		col, q, cleanup := m3cBuild(t, cfg.n, cfg.dim, cfg.sel)
		records, _ := col.ListAll(context.Background())
		vecCount := 0
		matchCount := 0
		for _, r := range records {
			if len(r.Vector) > 0 {
				vecCount++
			}
			if recordMatchesPredicates(r, m3cPred()) {
				matchCount++
			}
		}
		start := time.Now()
		res := scoreAndSelectTopK(col, records, q, 10)
		allNs := time.Since(start).Nanoseconds()
		fmt.Printf("N=%d dim=%d sel=%.2f: records=%d vecs=%d matches=%d (expect %d) exactTopK=%d allExactNs=%dns\n",
			cfg.n, cfg.dim, cfg.sel, len(records), vecCount, matchCount,
			int(float64(cfg.n)*cfg.sel), len(res.Results), allNs)
		cleanup()
	}
}
