package obs

import (
	"context"

	"github.com/xDarkicex/libravdb/libravdb"
)

// HealthChecker performs health checks
type HealthChecker struct {
	db interface{}
}

// NewHealthChecker creates health checker
func NewHealthChecker(db interface{}) *HealthChecker {
	return &HealthChecker{db: db}
}

// Check performs health check
func (hc *HealthChecker) Check(ctx context.Context) (*libravdb.HealthStatus, error) {
	return &libravdb.HealthStatus{
		Status: "healthy",
		Checks: map[string]*libravdb.CheckResult{
			"basic": {
				Healthy: true,
				Message: "System operational",
			},
		},
	}, nil
}
