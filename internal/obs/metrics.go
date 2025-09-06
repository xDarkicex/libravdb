package obs

import (
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Metrics holds all metrics
type Metrics struct {
	VectorInserts prometheus.Counter
	SearchQueries prometheus.Counter
	SearchErrors  prometheus.Counter
	SearchLatency prometheus.Histogram
	registry      *prometheus.Registry
}

var (
	globalMetrics *Metrics
	metricsOnce   sync.Once
)

// NewMetrics creates metrics instance (singleton pattern for tests)
func NewMetrics() *Metrics {
	metricsOnce.Do(func() {
		// Create custom registry
		registry := prometheus.NewRegistry()
		factory := promauto.With(registry)

		globalMetrics = &Metrics{
			VectorInserts: factory.NewCounter(prometheus.CounterOpts{
				Name: "libravdb_vector_inserts_total",
				Help: "Total vector insertions",
			}),
			SearchQueries: factory.NewCounter(prometheus.CounterOpts{
				Name: "libravdb_search_queries_total",
				Help: "Total search queries",
			}),
			SearchErrors: factory.NewCounter(prometheus.CounterOpts{
				Name: "libravdb_search_errors_total",
				Help: "Total search errors",
			}),
			SearchLatency: factory.NewHistogram(prometheus.HistogramOpts{
				Name:    "libravdb_search_latency_seconds",
				Help:    "Search latency",
				Buckets: prometheus.DefBuckets,
			}),
			registry: registry,
		}
	})

	return globalMetrics
}

// ResetForTesting resets metrics singleton for testing (call in test cleanup)
func ResetForTesting() {
	globalMetrics = nil
	metricsOnce = sync.Once{}
}
