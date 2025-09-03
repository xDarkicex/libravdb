package obs

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Metrics holds all metrics
type Metrics struct {
	VectorInserts prometheus.Counter
	SearchQueries prometheus.Counter
	SearchErrors  prometheus.Counter
	SearchLatency prometheus.Histogram
}

// NewMetrics creates metrics instance
func NewMetrics() *Metrics {
	return &Metrics{
		VectorInserts: promauto.NewCounter(prometheus.CounterOpts{
			Name: "libravdb_vector_inserts_total",
			Help: "Total vector insertions",
		}),
		SearchQueries: promauto.NewCounter(prometheus.CounterOpts{
			Name: "libravdb_search_queries_total",
			Help: "Total search queries",
		}),
		SearchErrors: promauto.NewCounter(prometheus.CounterOpts{
			Name: "libravdb_search_errors_total",
			Help: "Total search errors",
		}),
		SearchLatency: promauto.NewHistogram(prometheus.HistogramOpts{
			Name: "libravdb_search_latency_seconds",
			Help: "Search latency",
		}),
	}
}
