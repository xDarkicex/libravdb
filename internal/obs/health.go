package obs

import (
	"context"
)

// HealthStatus represents the health status of the database
type HealthStatus struct {
	Status string                  `json:"status"`
	Checks map[string]*CheckResult `json:"checks"`
}

// CheckResult represents the result of a single health check
type CheckResult struct {
	Healthy bool   `json:"healthy"`
	Message string `json:"message"`
}

// HealthChecker performs health checks
type HealthChecker struct {
	db interface{}
}

// NewHealthChecker creates health checker
func NewHealthChecker(db interface{}) *HealthChecker {
	return &HealthChecker{db: db}
}

// Pinger is an interface for checking database responsiveness
type Pinger interface {
	Ping(ctx context.Context) error
}

// Check performs health check
func (hc *HealthChecker) Check(ctx context.Context) (*HealthStatus, error) {
	status := &HealthStatus{
		Status: "healthy",
		Checks: map[string]*CheckResult{
			"basic": {
				Healthy: true,
				Message: "System operational",
			},
		},
	}

	if pinger, ok := hc.db.(Pinger); ok {
		if err := pinger.Ping(ctx); err != nil {
			status.Status = "unhealthy"
			status.Checks["db_ping"] = &CheckResult{
				Healthy: false,
				Message: err.Error(),
			}
		} else {
			status.Checks["db_ping"] = &CheckResult{
				Healthy: true,
				Message: "Database responsive",
			}
		}
	} else {
		status.Checks["db_ping"] = &CheckResult{
			Healthy: false,
			Message: "Database instance does not support Ping",
		}
	}

	return status, nil
}
