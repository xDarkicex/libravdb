package pgwire

import (
	"bytes"
	"encoding/binary"
	"net"
	"strconv"
	"sync"
	"testing"
	"time"
)

func FuzzAuthFailureLimiterNeverPanics(f *testing.F) {
	f.Add("198.51.100.1", int64(0), uint8(2))
	f.Add("", int64(123456789), uint8(1))
	f.Add("malformed-address", int64(-1), uint8(5))
	f.Fuzz(func(t *testing.T, key string, nanos int64, threshold uint8) {
		if threshold == 0 {
			threshold = 1
		}
		now := time.Unix(0, nanos)
		limiter := newAuthFailureLimiter(int(threshold), time.Second)
		for i := 0; i < int(threshold)+3; i++ {
			_ = limiter.allow(key, now)
			_ = limiter.allowUser(key, "fuzz-user", now)
			if limiter.admitAttempt(key, now) {
				limiter.releaseAttempt()
			}
			limiter.recordFailure(key, now)
			limiter.recordFailureFor(key, "fuzz-user", now)
		}
		_ = limiter.allow(key, now.Add(2*time.Second))
		_ = limiter.allowUser(key, "fuzz-user", now.Add(2*time.Second))
		limiter.recordSuccess(key)
		limiter.recordSuccessFor(key, "fuzz-user")
		_ = limiter.allow(key, now.Add(2*time.Second))
	})
}

func TestAuthFailureLimiterEntryBound(t *testing.T) {
	limiter := newAuthFailureLimiter(5, time.Minute)
	now := time.Now()
	for i := 0; i < maxAuthLimiterEntries*3; i++ {
		key := string(rune(i))
		limiter.recordFailureFor(key, "user", now)
		if limiter.admitAttempt(key, now) {
			limiter.releaseAttempt()
		}
	}
	limiter.mu.Lock()
	entries := len(limiter.clients)
	identityEntries := len(limiter.identities)
	attemptEntries := len(limiter.attempts)
	limiter.mu.Unlock()
	if entries > maxAuthLimiterEntries {
		t.Fatalf("auth limiter grew beyond bound: %d > %d", entries, maxAuthLimiterEntries)
	}
	if identityEntries > maxAuthLimiterEntries {
		t.Fatalf("auth identity limiter grew beyond bound: %d > %d", identityEntries, maxAuthLimiterEntries)
	}
	if attemptEntries > maxAuthLimiterEntries {
		t.Fatalf("auth attempt limiter grew beyond bound: %d > %d", attemptEntries, maxAuthLimiterEntries)
	}
}

func TestAuthFailureLimiterCompositeIdentity(t *testing.T) {
	limiter := newAuthFailureLimiter(2, time.Minute)
	now := time.Unix(100, 0)
	limiter.recordFailureFor("198.51.100.20", "alice", now)
	limiter.recordFailureFor("198.51.100.20", "alice", now)
	if limiter.allowUser("198.51.100.20", "alice", now) {
		t.Fatal("composite identity was not blocked after its failure threshold")
	}
	if !limiter.allowUser("198.51.100.20", "bob", now) {
		t.Fatal("independent username was incorrectly blocked")
	}
	limiter.recordSuccessFor("198.51.100.20", "bob")
	if limiter.allowUser("198.51.100.20", "alice", now) {
		t.Fatal("successful bob login cleared alice's composite failure state")
	}
}

func TestAuthAttemptAdmissionBucket(t *testing.T) {
	limiter := newAuthFailureLimiterWithPolicy(5, time.Minute, authAttemptPolicy{
		burst:         2,
		refill:        time.Second,
		maxConcurrent: 1,
	})
	now := time.Unix(100, 0)
	if !limiter.admitAttempt("198.51.100.21", now) {
		t.Fatal("first authentication attempt was rejected")
	}
	if limiter.admitAttempt("198.51.100.22", now) {
		t.Fatal("concurrent authentication limit was not enforced")
	}
	limiter.releaseAttempt()
	if !limiter.admitAttempt("198.51.100.21", now) {
		t.Fatal("second burst token was rejected")
	}
	limiter.releaseAttempt()
	if limiter.admitAttempt("198.51.100.21", now) {
		t.Fatal("attempt burst was not exhausted")
	}
	if !limiter.admitAttempt("198.51.100.21", now.Add(time.Second)) {
		t.Fatal("refilled attempt token was rejected")
	}
	limiter.releaseAttempt()
}

func TestAuthLimiterEvictionPreservesBlockedState(t *testing.T) {
	limiter := newAuthFailureLimiter(2, time.Minute)
	now := time.Unix(100, 0)
	blockedUntil := now.Add(time.Hour)
	limiter.mu.Lock()
	for i := 0; i < maxAuthLimiterEntries; i++ {
		limiter.clients[strconv.Itoa(i)] = authFailureEntry{blockedUntil: blockedUntil, lastSeen: now.Add(time.Duration(i) * time.Nanosecond)}
		limiter.identities[authIdentity{client: strconv.Itoa(i), user: "alice"}] = authFailureEntry{blockedUntil: blockedUntil, lastSeen: now.Add(time.Duration(i) * time.Nanosecond)}
		limiter.attempts[strconv.Itoa(i)] = authTokenBucket{tokens: float64(limiter.attemptPolicy.burst - 1), lastSeen: now.Add(time.Duration(i) * time.Nanosecond)}
	}
	limiter.mu.Unlock()

	limiter.recordFailureFor("new-client", "alice", now)
	limiter.mu.Lock()
	_, sourcePreserved := limiter.clients["0"]
	_, identityPreserved := limiter.identities[authIdentity{client: "0", user: "alice"}]
	limiter.mu.Unlock()
	if !sourcePreserved || !identityPreserved {
		t.Fatal("blocked source or identity state was evicted")
	}
	if limiter.allow("0", now) || limiter.allowUser("0", "alice", now) {
		t.Fatal("blocked state was not enforced after saturation")
	}
	if limiter.allow("new-client", now) {
		t.Fatal("new source was admitted while limiter state was saturated")
	}
	if limiter.admitAttempt("new-client", now) {
		t.Fatal("admission unexpectedly evicted a depleted token bucket")
	}
}

func TestAuthGlobalFailurePolicyConfigurable(t *testing.T) {
	limiter := newAuthFailureLimiterWithPolicy(100, time.Minute, authAttemptPolicy{
		burst:               4,
		refill:              time.Second,
		maxConcurrent:       4,
		globalFailureLimit:  2,
		globalFailureWindow: time.Minute,
	})
	now := time.Unix(100, 0)
	limiter.recordFailure("one", now)
	limiter.recordFailure("two", now)
	if limiter.allow("three", now) {
		t.Fatal("configured global failure limit was not enforced")
	}
}

func TestAuthFailureLimiterConcurrentSafety(t *testing.T) {
	limiter := newAuthFailureLimiter(5, time.Minute)
	now := time.Unix(100, 0)
	var wg sync.WaitGroup
	for worker := 0; worker < 32; worker++ {
		worker := worker
		wg.Add(1)
		go func() {
			defer wg.Done()
			key := "198.51.100." + string(rune('a'+worker))
			for i := 0; i < 100; i++ {
				if limiter.admitAttempt(key, now) {
					limiter.releaseAttempt()
				}
				limiter.allow(key, now)
				limiter.allowUser(key, "alice", now)
				limiter.recordFailureFor(key, "alice", now)
				limiter.recordSuccessFor(key, "alice")
			}
		}()
	}
	wg.Wait()
	limiter.mu.Lock()
	active := limiter.activeAttempts
	limiter.mu.Unlock()
	if active != 0 {
		t.Fatalf("authentication slots leaked: %d", active)
	}
}

func TestAuthFailureLimiterEnforcedInStartup(t *testing.T) {
	credential, err := deriveSCRAMCredential("alice", "secret", []byte("fixed-scram-salt"), DefaultSCRAMIterations)
	if err != nil {
		t.Fatal(err)
	}
	limiter := newAuthFailureLimiter(2, time.Minute)
	config := ServerConfig{
		RequireAuthentication: true,
		Credentials:           map[string]SCRAMCredential{"alice": credential},
		authLimiter:           limiter,
		authClientKey:         "198.51.100.9",
	}

	for attempt := 0; attempt < 2; attempt++ {
		rw := &startupTestReadWriter{reader: bytes.NewReader(malformedAuthStartupWire())}
		if _, _, err := handleStartupWithConfig(rw, nil, nil, false, config); err == nil {
			t.Fatalf("malformed authentication attempt %d unexpectedly succeeded", attempt+1)
		}
	}
	if limiter.allow(config.authClientKey, time.Now()) {
		t.Fatal("client was not blocked after startup authentication failures")
	}

	rw := &startupTestReadWriter{reader: bytes.NewReader(malformedAuthStartupWire())}
	if _, _, err := handleStartupWithConfig(rw, nil, nil, false, config); err == nil {
		t.Fatal("rate-limited startup unexpectedly succeeded")
	}
	msgType, _, err := ReadMessage(bytes.NewReader(rw.writer.Bytes()))
	if err != nil {
		t.Fatalf("read rate-limit response: %v", err)
	}
	if msgType != msgErrorResponse {
		t.Fatalf("expected immediate ErrorResponse for rate-limited startup, got %q", msgType)
	}
}

func TestAuthAttemptAdmissionEnforcedInStartup(t *testing.T) {
	credential, err := deriveSCRAMCredential("alice", "secret", []byte("fixed-scram-salt"), DefaultSCRAMIterations)
	if err != nil {
		t.Fatal(err)
	}
	limiter := newAuthFailureLimiterWithPolicy(100, time.Minute, authAttemptPolicy{
		burst:         1,
		refill:        time.Minute,
		maxConcurrent: 32,
	})
	config := ServerConfig{
		RequireAuthentication: true,
		Credentials:           map[string]SCRAMCredential{"alice": credential},
		authLimiter:           limiter,
		authClientKey:         "198.51.100.10",
	}

	first := &startupTestReadWriter{reader: bytes.NewReader(malformedAuthStartupWire())}
	if _, _, err := handleStartupWithConfig(first, nil, nil, false, config); err == nil {
		t.Fatal("initial malformed authentication attempt unexpectedly succeeded")
	}
	second := &startupTestReadWriter{reader: bytes.NewReader(malformedAuthStartupWire())}
	if _, _, err := handleStartupWithConfig(second, nil, nil, false, config); err == nil {
		t.Fatal("admission-limited authentication attempt unexpectedly succeeded")
	}
	msgType, _, err := ReadMessage(bytes.NewReader(second.writer.Bytes()))
	if err != nil {
		t.Fatalf("read admission-limit response: %v", err)
	}
	if msgType != msgErrorResponse {
		t.Fatalf("expected immediate ErrorResponse for admission-limited startup, got %q", msgType)
	}
}

func malformedAuthStartupWire() []byte {
	payload := binary.BigEndian.AppendUint32(nil, uint32(protocolVersion))
	payload = append(payload, "user\x00alice\x00database\x00test\x00\x00"...)
	wire := binary.BigEndian.AppendUint32(nil, uint32(4+len(payload)))
	wire = append(wire, payload...)
	clientFirst := []byte("n,,broken")
	initial := append([]byte("SCRAM-SHA-256\x00"), 0, 0, 0, 0)
	binary.BigEndian.PutUint32(initial[len("SCRAM-SHA-256")+1:], uint32(len(clientFirst)))
	initial = append(initial, clientFirst...)
	var message bytes.Buffer
	_ = WriteMessage(&message, msgPassword, initial)
	return append(wire, message.Bytes()...)
}

func TestAuthClientKeyNormalization(t *testing.T) {
	cases := []struct {
		name string
		conn net.Conn
		want string
	}{
		{name: "nil", want: "unknown"},
	}
	for _, tc := range cases {
		if got := authClientKey(tc.conn); got != tc.want {
			t.Fatalf("%s: got %q, want %q", tc.name, got, tc.want)
		}
	}
	if got := authClientKey(&limiterTestConn{remote: &net.TCPAddr{IP: net.ParseIP("2001:db8::1"), Port: 54321}}); got != "2001:db8::1" {
		t.Fatalf("IPv6 key normalization: got %q", got)
	}
	if got := authClientKey(&limiterTestConn{remote: &net.TCPAddr{IP: net.ParseIP("::ffff:198.51.100.7"), Port: 54321}}); got != "198.51.100.7" {
		t.Fatalf("IPv4-mapped IPv6 key normalization: got %q", got)
	}
}

type limiterTestConn struct{ remote net.Addr }

func (c *limiterTestConn) Read([]byte) (int, error)         { return 0, nil }
func (c *limiterTestConn) Write(p []byte) (int, error)      { return len(p), nil }
func (c *limiterTestConn) Close() error                     { return nil }
func (c *limiterTestConn) LocalAddr() net.Addr              { return nil }
func (c *limiterTestConn) RemoteAddr() net.Addr             { return c.remote }
func (c *limiterTestConn) SetDeadline(time.Time) error      { return nil }
func (c *limiterTestConn) SetReadDeadline(time.Time) error  { return nil }
func (c *limiterTestConn) SetWriteDeadline(time.Time) error { return nil }
