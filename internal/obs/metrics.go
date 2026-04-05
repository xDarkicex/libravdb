package obs

import (
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Metrics holds all metrics
type Metrics struct {
	VectorInserts   prometheus.Counter
	VectorUpdates   prometheus.Counter
	VectorDeletes   prometheus.Counter
	TxBegins        prometheus.Counter
	TxCommits       prometheus.Counter
	TxRollbacks     prometheus.Counter
	TxConflicts     prometheus.Counter
	CASSuccesses    prometheus.Counter
	CASConflicts    prometheus.Counter
	CASAborts       prometheus.Counter
	TxCommitOps     prometheus.Histogram
	TxCommitLatency prometheus.Histogram
	SearchQueries   prometheus.Counter
	SearchErrors    prometheus.Counter
	SearchLatency   prometheus.Histogram
	registry        *prometheus.Registry
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
			VectorUpdates: factory.NewCounter(prometheus.CounterOpts{
				Name: "libravdb_vector_updates_total",
				Help: "Total vector updates",
			}),
			VectorDeletes: factory.NewCounter(prometheus.CounterOpts{
				Name: "libravdb_vector_deletes_total",
				Help: "Total vector deletions",
			}),
			TxBegins: factory.NewCounter(prometheus.CounterOpts{
				Name: "libravdb_transactions_begun_total",
				Help: "Total transactions begun",
			}),
			TxCommits: factory.NewCounter(prometheus.CounterOpts{
				Name: "libravdb_transactions_committed_total",
				Help: "Total transactions committed",
			}),
			TxRollbacks: factory.NewCounter(prometheus.CounterOpts{
				Name: "libravdb_transactions_rolled_back_total",
				Help: "Total transactions rolled back",
			}),
			TxConflicts: factory.NewCounter(prometheus.CounterOpts{
				Name: "libravdb_transaction_conflicts_total",
				Help: "Total transaction conflicts or validation failures",
			}),
			CASSuccesses: factory.NewCounter(prometheus.CounterOpts{
				Name: "libravdb_cas_success_total",
				Help: "Total successful compare-and-swap writes",
			}),
			CASConflicts: factory.NewCounter(prometheus.CounterOpts{
				Name: "libravdb_cas_conflict_total",
				Help: "Total compare-and-swap version conflicts",
			}),
			CASAborts: factory.NewCounter(prometheus.CounterOpts{
				Name: "libravdb_cas_abort_total",
				Help: "Total transaction aborts caused by compare-and-swap precondition failures",
			}),
			TxCommitOps: factory.NewHistogram(prometheus.HistogramOpts{
				Name:    "libravdb_transaction_commit_ops",
				Help:    "Number of staged operations per committed transaction",
				Buckets: []float64{1, 2, 4, 8, 16, 32, 64, 128, 256},
			}),
			TxCommitLatency: factory.NewHistogram(prometheus.HistogramOpts{
				Name:    "libravdb_transaction_commit_latency_seconds",
				Help:    "Transaction commit latency",
				Buckets: prometheus.DefBuckets,
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
