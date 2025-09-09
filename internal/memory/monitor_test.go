package memory

import (
	"testing"
	"time"
)

func TestNewMonitor(t *testing.T) {
	monitor := NewMonitor(10, time.Second)

	if monitor == nil {
		t.Fatal("NewMonitor returned nil")
	}

	if monitor.maxSnapshots != 10 {
		t.Errorf("Expected maxSnapshots 10, got %d", monitor.maxSnapshots)
	}

	if monitor.interval != time.Second {
		t.Errorf("Expected interval 1s, got %v", monitor.interval)
	}
}

func TestTakeSnapshot(t *testing.T) {
	monitor := NewMonitor(5, time.Second)

	snapshot := monitor.TakeSnapshot()

	// Basic sanity checks
	if snapshot.Timestamp.IsZero() {
		t.Error("Snapshot timestamp should be set")
	}

	if snapshot.HeapAlloc == 0 {
		t.Error("HeapAlloc should be greater than 0")
	}

	if snapshot.HeapSys == 0 {
		t.Error("HeapSys should be greater than 0")
	}

	// Check that snapshot was stored
	snapshots := monitor.GetSnapshots()
	if len(snapshots) != 1 {
		t.Errorf("Expected 1 snapshot, got %d", len(snapshots))
	}

	if snapshots[0].Timestamp != snapshot.Timestamp {
		t.Error("Stored snapshot should match returned snapshot")
	}
}

func TestSnapshotHistory(t *testing.T) {
	monitor := NewMonitor(3, time.Second) // Max 3 snapshots

	// Take more snapshots than the limit
	for i := 0; i < 5; i++ {
		monitor.TakeSnapshot()
		time.Sleep(10 * time.Millisecond) // Ensure different timestamps
	}

	snapshots := monitor.GetSnapshots()
	if len(snapshots) != 3 {
		t.Errorf("Expected 3 snapshots (max), got %d", len(snapshots))
	}

	// Verify snapshots are in chronological order
	for i := 1; i < len(snapshots); i++ {
		if snapshots[i].Timestamp.Before(snapshots[i-1].Timestamp) {
			t.Error("Snapshots should be in chronological order")
		}
	}
}

func TestGetLatestSnapshot(t *testing.T) {
	monitor := NewMonitor(5, time.Second)

	// Test with no snapshots
	_, exists := monitor.GetLatestSnapshot()
	if exists {
		t.Error("Should return false when no snapshots exist")
	}

	// Take a snapshot
	original := monitor.TakeSnapshot()

	// Get latest snapshot
	latest, exists := monitor.GetLatestSnapshot()
	if !exists {
		t.Error("Should return true when snapshots exist")
	}

	if latest.Timestamp != original.Timestamp {
		t.Error("Latest snapshot should match the one we just took")
	}

	// Take another snapshot
	time.Sleep(10 * time.Millisecond)
	second := monitor.TakeSnapshot()

	// Latest should now be the second one
	latest, exists = monitor.GetLatestSnapshot()
	if !exists {
		t.Error("Should return true when snapshots exist")
	}

	if latest.Timestamp != second.Timestamp {
		t.Error("Latest snapshot should be the most recent one")
	}
}

func TestCalculateMemoryTrend(t *testing.T) {
	monitor := NewMonitor(10, time.Second)

	// Test with no snapshots
	trend := monitor.CalculateMemoryTrend()
	if trend.Direction != TrendUnknown {
		t.Errorf("Expected TrendUnknown with no snapshots, got %v", trend.Direction)
	}

	// Test with one snapshot
	monitor.TakeSnapshot()
	trend = monitor.CalculateMemoryTrend()
	if trend.Direction != TrendUnknown {
		t.Errorf("Expected TrendUnknown with one snapshot, got %v", trend.Direction)
	}

	// Test with two snapshots
	time.Sleep(10 * time.Millisecond)
	monitor.TakeSnapshot()

	trend = monitor.CalculateMemoryTrend()
	// We can't predict the exact trend, but it should be one of the valid values
	validTrends := []TrendDirection{TrendIncreasing, TrendDecreasing, TrendStable}
	found := false
	for _, validTrend := range validTrends {
		if trend.Direction == validTrend {
			found = true
			break
		}
	}

	if !found {
		t.Errorf("Trend direction should be valid, got %v", trend.Direction)
	}

	if trend.Duration <= 0 {
		t.Error("Trend duration should be positive")
	}
}

func TestTrendDirectionString(t *testing.T) {
	tests := []struct {
		direction TrendDirection
		expected  string
	}{
		{TrendUnknown, "unknown"},
		{TrendIncreasing, "increasing"},
		{TrendDecreasing, "decreasing"},
		{TrendStable, "stable"},
	}

	for _, test := range tests {
		result := test.direction.String()
		if result != test.expected {
			t.Errorf("Expected %s, got %s", test.expected, result)
		}
	}
}

func TestGetMemoryPressure(t *testing.T) {
	pressure, err := GetMemoryPressure()
	if err != nil {
		t.Fatalf("GetMemoryPressure failed: %v", err)
	}

	if pressure < 0 || pressure > 1 {
		t.Errorf("Memory pressure should be between 0 and 1, got %f", pressure)
	}
}

func TestForceGC(t *testing.T) {
	// Allocate some memory to potentially free
	data := make([][]byte, 1000)
	for i := range data {
		data[i] = make([]byte, 1024)
	}

	// Clear references
	data = nil

	freed := ForceGC()

	// We can't guarantee memory will be freed, but the function should not panic
	if freed < 0 {
		t.Error("Freed memory should not be negative")
	}
}

func TestMemorySnapshotFields(t *testing.T) {
	monitor := NewMonitor(1, time.Second)
	snapshot := monitor.TakeSnapshot()

	// Test that all fields are populated with reasonable values
	if snapshot.HeapAlloc == 0 {
		t.Error("HeapAlloc should be greater than 0")
	}

	if snapshot.HeapSys == 0 {
		t.Error("HeapSys should be greater than 0")
	}

	if snapshot.HeapInuse == 0 {
		t.Error("HeapInuse should be greater than 0")
	}

	// HeapReleased can be 0, so we don't test it

	// NumGC can be 0 initially, so we don't test it

	// PauseTotalNs can be 0 initially, so we don't test it

	// Custom fields start at 0
	if snapshot.CacheUsage != 0 {
		t.Error("CacheUsage should start at 0")
	}

	if snapshot.IndexUsage != 0 {
		t.Error("IndexUsage should start at 0")
	}

	if snapshot.TotalManaged != 0 {
		t.Error("TotalManaged should start at 0")
	}
}

func TestMonitorConcurrency(t *testing.T) {
	monitor := NewMonitor(100, time.Millisecond)

	// Start multiple goroutines taking snapshots concurrently
	done := make(chan bool, 10)

	for i := 0; i < 10; i++ {
		go func() {
			defer func() { done <- true }()

			for j := 0; j < 10; j++ {
				monitor.TakeSnapshot()
				monitor.GetSnapshots()
				monitor.GetLatestSnapshot()
				monitor.CalculateMemoryTrend()
			}
		}()
	}

	// Wait for all goroutines to complete
	for i := 0; i < 10; i++ {
		<-done
	}

	// Monitor should still be functional
	snapshot := monitor.TakeSnapshot()
	if snapshot.Timestamp.IsZero() {
		t.Error("Monitor should still be functional after concurrent access")
	}
}
