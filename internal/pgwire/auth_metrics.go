package pgwire

import "sync/atomic"

// AuthStats is a point-in-time snapshot of server-side authentication events.
// It contains counters only; usernames, addresses, and credential material are
// never retained in these metrics.
type AuthStats struct {
	Attempts          uint64
	Successes         uint64
	Failures          uint64
	RateLimited       uint64
	AdmissionRejected uint64
	ActiveSCRAM       int64
}

type authMetrics struct {
	attempts          atomic.Uint64
	successes         atomic.Uint64
	failures          atomic.Uint64
	rateLimited       atomic.Uint64
	admissionRejected atomic.Uint64
	activeSCRAM       atomic.Int64
}

func (m *authMetrics) snapshot() AuthStats {
	if m == nil {
		return AuthStats{}
	}
	return AuthStats{
		Attempts:          m.attempts.Load(),
		Successes:         m.successes.Load(),
		Failures:          m.failures.Load(),
		RateLimited:       m.rateLimited.Load(),
		AdmissionRejected: m.admissionRejected.Load(),
		ActiveSCRAM:       m.activeSCRAM.Load(),
	}
}
