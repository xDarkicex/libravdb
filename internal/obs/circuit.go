package obs

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// CircuitState represents the state of a circuit breaker
type CircuitState int

const (
	// CircuitClosed - normal operation, requests are allowed
	CircuitClosed CircuitState = iota
	// CircuitOpen - circuit is open, requests are rejected
	CircuitOpen
	// CircuitHalfOpen - testing if service has recovered
	CircuitHalfOpen
)

// String returns the string representation of circuit state
func (cs CircuitState) String() string {
	switch cs {
	case CircuitClosed:
		return "CLOSED"
	case CircuitOpen:
		return "OPEN"
	case CircuitHalfOpen:
		return "HALF_OPEN"
	default:
		return "UNKNOWN"
	}
}

// CircuitBreakerConfig configures circuit breaker behavior
type CircuitBreakerConfig struct {
	// Name identifies the circuit breaker
	Name string

	// MaxFailures is the number of failures before opening the circuit
	MaxFailures int

	// Timeout is how long to wait before attempting to close the circuit
	Timeout time.Duration

	// MaxRequests is the maximum number of requests allowed in half-open state
	MaxRequests int

	// FailureThreshold is the failure rate threshold (0.0-1.0) to open circuit
	FailureThreshold float64

	// MinRequests is the minimum number of requests before calculating failure rate
	MinRequests int

	// ResetTimeout is how long to wait before resetting failure counts
	ResetTimeout time.Duration
}

// DefaultCircuitBreakerConfig returns sensible defaults
func DefaultCircuitBreakerConfig(name string) CircuitBreakerConfig {
	return CircuitBreakerConfig{
		Name:             name,
		MaxFailures:      5,
		Timeout:          30 * time.Second,
		MaxRequests:      3,
		FailureThreshold: 0.6, // 60% failure rate
		MinRequests:      10,
		ResetTimeout:     60 * time.Second,
	}
}

// CircuitBreaker implements the circuit breaker pattern for fault tolerance
type CircuitBreaker struct {
	mu     sync.RWMutex
	config CircuitBreakerConfig
	state  CircuitState

	// Counters
	failures   int
	successes  int
	requests   int
	generation int64

	// Timing
	lastFailureTime time.Time
	lastSuccessTime time.Time
	expiry          time.Time

	// Callbacks
	onStateChange func(name string, from, to CircuitState)
}

// NewCircuitBreaker creates a new circuit breaker
func NewCircuitBreaker(config CircuitBreakerConfig) *CircuitBreaker {
	return &CircuitBreaker{
		config:     config,
		state:      CircuitClosed,
		expiry:     time.Now().Add(config.ResetTimeout),
		generation: 0,
	}
}

// Execute executes a function with circuit breaker protection
func (cb *CircuitBreaker) Execute(ctx context.Context, fn func() error) error {
	generation, err := cb.beforeRequest()
	if err != nil {
		return err
	}

	defer func() {
		if r := recover(); r != nil {
			cb.afterRequest(generation, fmt.Errorf("panic: %v", r))
			panic(r)
		}
	}()

	err = fn()
	cb.afterRequest(generation, err)
	return err
}

// beforeRequest checks if the request should be allowed
func (cb *CircuitBreaker) beforeRequest() (int64, error) {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	now := time.Now()
	state, generation := cb.currentState(now)

	if state == CircuitOpen {
		return generation, fmt.Errorf("circuit breaker '%s' is open", cb.config.Name)
	}

	if state == CircuitHalfOpen && cb.requests >= cb.config.MaxRequests {
		return generation, fmt.Errorf("circuit breaker '%s' is half-open and max requests exceeded", cb.config.Name)
	}

	cb.requests++
	return generation, nil
}

// afterRequest records the result of a request
func (cb *CircuitBreaker) afterRequest(generation int64, err error) {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	now := time.Now()
	state, currentGeneration := cb.currentState(now)

	if generation != currentGeneration {
		return // Request was from a previous generation, ignore
	}

	if err != nil {
		cb.onFailure(state, now)
	} else {
		cb.onSuccess(state, now)
	}
}

// onFailure handles a failed request
func (cb *CircuitBreaker) onFailure(state CircuitState, now time.Time) {
	cb.failures++
	cb.lastFailureTime = now

	switch state {
	case CircuitClosed:
		if cb.shouldOpen(now) {
			cb.setState(CircuitOpen, now)
		}
	case CircuitHalfOpen:
		cb.setState(CircuitOpen, now)
	}
}

// onSuccess handles a successful request
func (cb *CircuitBreaker) onSuccess(state CircuitState, now time.Time) {
	cb.successes++
	cb.lastSuccessTime = now

	switch state {
	case CircuitHalfOpen:
		if cb.successes >= cb.config.MaxRequests {
			cb.setState(CircuitClosed, now)
		}
	}
}

// shouldOpen determines if the circuit should be opened
func (cb *CircuitBreaker) shouldOpen(now time.Time) bool {
	if cb.failures >= cb.config.MaxFailures {
		return true
	}

	if cb.requests >= cb.config.MinRequests {
		failureRate := float64(cb.failures) / float64(cb.requests)
		return failureRate >= cb.config.FailureThreshold
	}

	return false
}

// currentState returns the current state and generation
func (cb *CircuitBreaker) currentState(now time.Time) (CircuitState, int64) {
	switch cb.state {
	case CircuitClosed:
		if cb.expiry.Before(now) {
			cb.toNewGeneration(now)
		}
	case CircuitOpen:
		if cb.expiry.Before(now) {
			cb.setState(CircuitHalfOpen, now)
		}
	}
	return cb.state, cb.generation
}

// setState changes the circuit breaker state
func (cb *CircuitBreaker) setState(state CircuitState, now time.Time) {
	if cb.state == state {
		return
	}

	prev := cb.state
	cb.state = state

	cb.toNewGeneration(now)

	if cb.onStateChange != nil {
		cb.onStateChange(cb.config.Name, prev, state)
	}
}

// toNewGeneration resets counters and starts a new generation
func (cb *CircuitBreaker) toNewGeneration(now time.Time) {
	cb.generation++
	cb.requests = 0
	cb.failures = 0
	cb.successes = 0

	var timeout time.Duration
	switch cb.state {
	case CircuitClosed:
		timeout = cb.config.ResetTimeout
	case CircuitOpen:
		timeout = cb.config.Timeout
	case CircuitHalfOpen:
		timeout = cb.config.Timeout
	}

	cb.expiry = now.Add(timeout)
}

// State returns the current state of the circuit breaker
func (cb *CircuitBreaker) State() CircuitState {
	cb.mu.RLock()
	defer cb.mu.RUnlock()

	state, _ := cb.currentState(time.Now())
	return state
}

// Counts returns the current failure and success counts
func (cb *CircuitBreaker) Counts() (failures, successes, requests int) {
	cb.mu.RLock()
	defer cb.mu.RUnlock()

	return cb.failures, cb.successes, cb.requests
}

// OnStateChange sets a callback for state changes
func (cb *CircuitBreaker) OnStateChange(fn func(name string, from, to CircuitState)) {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.onStateChange = fn
}

// Reset manually resets the circuit breaker to closed state
func (cb *CircuitBreaker) Reset() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.setState(CircuitClosed, time.Now())
}

// CircuitBreakerManager manages multiple circuit breakers
type CircuitBreakerManager struct {
	mu       sync.RWMutex
	breakers map[string]*CircuitBreaker
}

// NewCircuitBreakerManager creates a new circuit breaker manager
func NewCircuitBreakerManager() *CircuitBreakerManager {
	return &CircuitBreakerManager{
		breakers: make(map[string]*CircuitBreaker),
	}
}

// GetOrCreate gets an existing circuit breaker or creates a new one
func (cbm *CircuitBreakerManager) GetOrCreate(name string, config CircuitBreakerConfig) *CircuitBreaker {
	cbm.mu.Lock()
	defer cbm.mu.Unlock()

	if breaker, exists := cbm.breakers[name]; exists {
		return breaker
	}

	config.Name = name
	breaker := NewCircuitBreaker(config)
	cbm.breakers[name] = breaker
	return breaker
}

// Get retrieves a circuit breaker by name
func (cbm *CircuitBreakerManager) Get(name string) (*CircuitBreaker, bool) {
	cbm.mu.RLock()
	defer cbm.mu.RUnlock()

	breaker, exists := cbm.breakers[name]
	return breaker, exists
}

// Remove removes a circuit breaker
func (cbm *CircuitBreakerManager) Remove(name string) {
	cbm.mu.Lock()
	defer cbm.mu.Unlock()

	delete(cbm.breakers, name)
}

// GetAll returns all circuit breakers
func (cbm *CircuitBreakerManager) GetAll() map[string]*CircuitBreaker {
	cbm.mu.RLock()
	defer cbm.mu.RUnlock()

	result := make(map[string]*CircuitBreaker)
	for name, breaker := range cbm.breakers {
		result[name] = breaker
	}
	return result
}

// GetStates returns the states of all circuit breakers
func (cbm *CircuitBreakerManager) GetStates() map[string]CircuitState {
	cbm.mu.RLock()
	defer cbm.mu.RUnlock()

	result := make(map[string]CircuitState)
	for name, breaker := range cbm.breakers {
		result[name] = breaker.State()
	}
	return result
}

// ResetAll resets all circuit breakers to closed state
func (cbm *CircuitBreakerManager) ResetAll() {
	cbm.mu.RLock()
	defer cbm.mu.RUnlock()

	for _, breaker := range cbm.breakers {
		breaker.Reset()
	}
}
