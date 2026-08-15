package pgwire

import (
	"net"
	"sync"
	"time"
)

const (
	DefaultAuthFailureThreshold = 5
	DefaultAuthLockoutDuration  = time.Minute
	DefaultAuthAttemptBurst     = 32
	DefaultAuthAttemptRefill    = time.Second
	DefaultMaxConcurrentAuth    = 32
	globalAuthFailureLimit      = 100
	globalAuthFailureWindow     = time.Minute
	maxAuthLimiterEntries       = 4096
)

type authFailureEntry struct {
	failures     int
	windowStart  time.Time
	blockedUntil time.Time
	lastSeen     time.Time
}

type authIdentity struct {
	client string
	user   string
}

type authTokenBucket struct {
	tokens     float64
	lastRefill time.Time
	lastSeen   time.Time
}

type authAttemptPolicy struct {
	burst               int
	refill              time.Duration
	maxConcurrent       int
	globalFailureLimit  int
	globalFailureWindow time.Duration
}

// authFailureLimiter limits failed authentication and admits bounded SCRAM
// work without adding a protocol round trip or changing PostgreSQL driver
// behavior. State includes source-address and (source, username) buckets,
// plus bounded per-source/global attempt tokens.
type authFailureLimiter struct {
	mu                  sync.Mutex
	clients             map[string]authFailureEntry
	identities          map[authIdentity]authFailureEntry
	attempts            map[string]authTokenBucket
	threshold           int
	lockout             time.Duration
	attemptPolicy       authAttemptPolicy
	globalFailureLimit  int
	globalFailureWindow time.Duration
	globalAttempts      authTokenBucket
	activeAttempts      int
	globalFailures      int
	globalWindowStart   time.Time
	globalBlockedUntil  time.Time
	clientsSaturated    bool
	identitiesSaturated bool
}

func newAuthFailureLimiter(threshold int, lockout time.Duration) *authFailureLimiter {
	return newAuthFailureLimiterWithPolicy(threshold, lockout, authAttemptPolicy{
		burst:               DefaultAuthAttemptBurst,
		refill:              DefaultAuthAttemptRefill,
		maxConcurrent:       DefaultMaxConcurrentAuth,
		globalFailureLimit:  globalAuthFailureLimit,
		globalFailureWindow: globalAuthFailureWindow,
	})
}

func newAuthFailureLimiterWithPolicy(threshold int, lockout time.Duration, policy authAttemptPolicy) *authFailureLimiter {
	if threshold == 0 {
		threshold = DefaultAuthFailureThreshold
	}
	if lockout == 0 {
		lockout = DefaultAuthLockoutDuration
	}
	if policy.burst == 0 {
		policy.burst = DefaultAuthAttemptBurst
	}
	if policy.refill == 0 {
		policy.refill = DefaultAuthAttemptRefill
	}
	if policy.maxConcurrent == 0 {
		policy.maxConcurrent = DefaultMaxConcurrentAuth
	}
	if policy.globalFailureLimit == 0 {
		policy.globalFailureLimit = globalAuthFailureLimit
	}
	if policy.globalFailureWindow == 0 {
		policy.globalFailureWindow = globalAuthFailureWindow
	}
	return &authFailureLimiter{
		clients:             make(map[string]authFailureEntry),
		identities:          make(map[authIdentity]authFailureEntry),
		attempts:            make(map[string]authTokenBucket),
		threshold:           threshold,
		lockout:             lockout,
		attemptPolicy:       policy,
		globalFailureLimit:  policy.globalFailureLimit,
		globalFailureWindow: policy.globalFailureWindow,
	}
}

func (l *authFailureLimiter) allow(key string, now time.Time) bool {
	if l == nil {
		return true
	}
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.globalBlockedUntil.After(now) {
		return false
	}
	entry, ok := l.clients[key]
	if !ok {
		if l.clientsSaturated {
			if !l.evictOldestLocked(now) {
				return false
			}
			l.clientsSaturated = false
		}
		return true
	}
	if !entry.blockedUntil.After(now) && now.Sub(entry.lastSeen) >= l.globalFailureWindow {
		delete(l.clients, key)
		if len(l.clients) < maxAuthLimiterEntries {
			l.clientsSaturated = false
		}
		return true
	}
	return !entry.blockedUntil.After(now)
}

func (l *authFailureLimiter) allowUser(client, user string, now time.Time) bool {
	if l == nil {
		return true
	}
	l.mu.Lock()
	defer l.mu.Unlock()

	entry, ok := l.identities[makeAuthIdentity(client, user)]
	if !ok {
		if l.identitiesSaturated {
			if !l.evictOldestIdentityLocked(now) {
				return false
			}
			l.identitiesSaturated = false
		}
		return true
	}
	if !entry.blockedUntil.After(now) && now.Sub(entry.lastSeen) >= l.globalFailureWindow {
		delete(l.identities, makeAuthIdentity(client, user))
		if len(l.identities) < maxAuthLimiterEntries {
			l.identitiesSaturated = false
		}
		return true
	}
	return !entry.blockedUntil.After(now)
}

// admitAttempt consumes one source and global admission token and reserves a
// concurrent authentication slot. The caller must call releaseAttempt after
// SCRAM completes, regardless of success or failure.
func (l *authFailureLimiter) admitAttempt(client string, now time.Time) bool {
	if l == nil {
		return true
	}
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.attemptPolicy.maxConcurrent > 0 && l.activeAttempts >= l.attemptPolicy.maxConcurrent {
		return false
	}
	refillTokenBucket(&l.globalAttempts, l.globalFailureLimit, l.globalFailureWindow, now)
	source := l.attempts[client]
	refillTokenBucket(&source, l.attemptPolicy.burst, l.attemptPolicy.refill, now)
	if l.globalAttempts.tokens < 1 || source.tokens < 1 {
		return false
	}
	if _, ok := l.attempts[client]; !ok && len(l.attempts) >= maxAuthLimiterEntries {
		if !l.evictOldestAttemptLocked() {
			return false
		}
	}
	l.globalAttempts.tokens--
	source.tokens--
	source.lastSeen = now
	l.attempts[client] = source
	l.activeAttempts++
	return true
}

func (l *authFailureLimiter) releaseAttempt() {
	if l == nil {
		return
	}
	l.mu.Lock()
	if l.activeAttempts > 0 {
		l.activeAttempts--
	}
	l.mu.Unlock()
}

func (l *authFailureLimiter) recordFailure(key string, now time.Time) {
	l.recordFailureFor(key, "", now)
}

func (l *authFailureLimiter) recordFailureFor(client, user string, now time.Time) {
	if l == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.globalWindowStart.IsZero() || now.Sub(l.globalWindowStart) >= l.globalFailureWindow {
		l.globalWindowStart = now
		l.globalFailures = 0
		l.globalBlockedUntil = time.Time{}
	}
	l.globalFailures++
	if l.globalFailures >= l.globalFailureLimit {
		l.globalBlockedUntil = now.Add(l.globalFailureWindow)
	}

	entry, ok := l.clients[client]
	if !ok || entry.windowStart.IsZero() || now.Sub(entry.windowStart) >= l.globalFailureWindow {
		entry = authFailureEntry{windowStart: now}
	}
	entry.failures++
	entry.lastSeen = now
	if entry.failures >= l.threshold {
		entry.blockedUntil = now.Add(l.lockout)
	}
	if !ok && len(l.clients) >= maxAuthLimiterEntries {
		if !l.evictOldestLocked(now) {
			l.clientsSaturated = true
			return
		}
		l.clientsSaturated = false
	}
	l.clients[client] = entry

	if user != "" {
		identity := makeAuthIdentity(client, user)
		identityEntry, identityOK := l.identities[identity]
		if !identityOK || identityEntry.windowStart.IsZero() || now.Sub(identityEntry.windowStart) >= l.globalFailureWindow {
			identityEntry = authFailureEntry{windowStart: now}
		}
		identityEntry.failures++
		identityEntry.lastSeen = now
		if identityEntry.failures >= l.threshold {
			identityEntry.blockedUntil = now.Add(l.lockout)
		}
		if !identityOK && len(l.identities) >= maxAuthLimiterEntries {
			if !l.evictOldestIdentityLocked(now) {
				l.identitiesSaturated = true
				return
			}
			l.identitiesSaturated = false
		}
		l.identities[identity] = identityEntry
	}
}

func (l *authFailureLimiter) recordSuccess(key string) {
	l.recordSuccessFor(key, "")
}

func (l *authFailureLimiter) recordSuccessFor(client, user string) {
	if l == nil {
		return
	}
	l.mu.Lock()
	delete(l.clients, client)
	if user != "" {
		delete(l.identities, makeAuthIdentity(client, user))
	}
	l.mu.Unlock()
}

func makeAuthIdentity(client, user string) authIdentity {
	if len(user) > maxSCRAMUsernameBytes {
		user = user[:maxSCRAMUsernameBytes]
	}
	return authIdentity{client: client, user: user}
}

func refillTokenBucket(bucket *authTokenBucket, burst int, refill time.Duration, now time.Time) {
	if burst <= 0 || refill <= 0 {
		bucket.tokens = 0
		bucket.lastRefill = now
		bucket.lastSeen = now
		return
	}
	if bucket.lastRefill.IsZero() {
		bucket.tokens = float64(burst)
		bucket.lastRefill = now
		bucket.lastSeen = now
		return
	}
	if now.After(bucket.lastRefill) {
		elapsed := now.Sub(bucket.lastRefill)
		bucket.tokens += elapsed.Seconds() / refill.Seconds()
		if bucket.tokens > float64(burst) {
			bucket.tokens = float64(burst)
		}
		bucket.lastRefill = now
	}
	bucket.lastSeen = now
}

func (l *authFailureLimiter) evictOldestLocked(now time.Time) bool {
	var oldestKey string
	var oldest time.Time
	for key, entry := range l.clients {
		if entry.blockedUntil.After(now) {
			continue
		}
		if oldestKey == "" || entry.lastSeen.Before(oldest) {
			oldestKey = key
			oldest = entry.lastSeen
		}
	}
	if oldestKey != "" {
		delete(l.clients, oldestKey)
		return true
	}
	return false
}

func (l *authFailureLimiter) evictOldestIdentityLocked(now time.Time) bool {
	var oldestKey authIdentity
	var oldest time.Time
	for key, entry := range l.identities {
		if entry.blockedUntil.After(now) {
			continue
		}
		if oldestKey == (authIdentity{}) || entry.lastSeen.Before(oldest) {
			oldestKey = key
			oldest = entry.lastSeen
		}
	}
	if oldestKey != (authIdentity{}) {
		delete(l.identities, oldestKey)
		return true
	}
	return false
}

func (l *authFailureLimiter) evictOldestAttemptLocked() bool {
	var oldestKey string
	var oldest time.Time
	for key, entry := range l.attempts {
		// Never evict a source that has consumed tokens recently. A fresh
		// insertion must not reset an active source's admission budget.
		if entry.tokens < float64(l.attemptPolicy.burst) {
			continue
		}
		if oldestKey == "" || entry.lastSeen.Before(oldest) {
			oldestKey = key
			oldest = entry.lastSeen
		}
	}
	if oldestKey != "" {
		delete(l.attempts, oldestKey)
		return true
	}
	return false
}

func authClientKey(conn net.Conn) string {
	if conn == nil || conn.RemoteAddr() == nil {
		return "unknown"
	}
	address := conn.RemoteAddr().String()
	host, _, err := net.SplitHostPort(address)
	if err == nil && host != "" {
		// Canonicalize IPv4-mapped IPv6 addresses so one source cannot evade
		// the per-client bucket merely by presenting as ::ffff:a.b.c.d.
		if ip := net.ParseIP(host); ip != nil {
			if ipv4 := ip.To4(); ipv4 != nil {
				return ipv4.String()
			}
			return ip.String()
		}
		return host
	}
	return address
}
