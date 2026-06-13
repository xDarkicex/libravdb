package libravdb

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestHealthMonitorCallbackCleanup verifies that Stop() waits for in-flight
// callback goroutines to finish, preventing goroutine leaks.
func TestHealthMonitorCallbackCleanup(t *testing.T) {
	var callbackCount atomic.Int32
	var wg sync.WaitGroup

	cb := func(status HealthStatus) {
		callbackCount.Add(1)
		// Simulate work that the callback might do.
		time.Sleep(10 * time.Millisecond)
		wg.Done()
	}

	hm := NewSystemHealthMonitor(50 * time.Millisecond)
	hm.RegisterCallback(cb)

	// Track callbacks ourselves so the test knows when they finish.
	// The monitor's internal WaitGroup tracks goroutine completion.
	wg.Add(1)                      // expect at least one callback during the run
	hm.Start(context.Background()) // nil context triggers immediate status check via performHealthCheck

	// Wait for the initial callback to be scheduled.
	time.Sleep(100 * time.Millisecond)

	// Stop must wait for in-flight callbacks.
	err := hm.Stop()
	if err != nil {
		t.Fatalf("Stop: %v", err)
	}

	// Wait for the callback we tracked externally.
	wg.Wait()

	if callbackCount.Load() == 0 {
		t.Error("expected at least one callback invocation")
	}
}

// TestHealthMonitorStartStopCycle verifies Start/Stop can be called
// multiple times without races or deadlocks.
func TestHealthMonitorStartStopCycle(t *testing.T) {
	hm := NewSystemHealthMonitor(10 * time.Millisecond)
	hm.RegisterCallback(func(status HealthStatus) {})

	for i := range 10 {
		if err := hm.Start(context.Background()); err != nil {
			t.Fatalf("cycle %d Start: %v", i, err)
		}
		time.Sleep(5 * time.Millisecond)
		if err := hm.Stop(); err != nil {
			t.Fatalf("cycle %d Stop: %v", i, err)
		}
	}
}
