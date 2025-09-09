#!/bin/bash

# LibraVDB HNSW Persistence Benchmarks
# This script runs comprehensive benchmarks for the persistence functionality

set -e

echo "🚀 Starting LibraVDB HNSW Persistence Benchmarks"
echo "=================================================="

# Create results directory
RESULTS_DIR="benchmark_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "📊 Results will be saved to: $RESULTS_DIR"
echo ""

# Function to run benchmark and save results
run_benchmark() {
    local name=$1
    local pattern=$2
    local output_file="$RESULTS_DIR/${name}.txt"
    
    echo "🔄 Running $name benchmarks..."
    go test -bench="$pattern" -benchmem -count=3 -timeout=30m ./benchmark > "$output_file" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ $name benchmarks completed successfully"
        # Extract key metrics
        echo "   Key Results:"
        grep -E "(MB/s|success_rate|detection_rate|compatibility)" "$output_file" | head -5 | sed 's/^/   /'
    else
        echo "❌ $name benchmarks failed"
        tail -10 "$output_file" | sed 's/^/   /'
    fi
    echo ""
}

# Run basic persistence benchmarks
run_benchmark "Basic_Persistence" "^BenchmarkSave|^BenchmarkLoad"

# Run reliability benchmarks
run_benchmark "Reliability" "^BenchmarkMemoryUsage|^BenchmarkCRC32|^BenchmarkAtomic|^BenchmarkSearch"

# Run compatibility benchmarks
run_benchmark "Compatibility" "^BenchmarkFormat|^BenchmarkZeroData|^BenchmarkConcurrent"

# Run large scale benchmarks (if not in short mode)
if [ "$1" != "--short" ]; then
    echo "🔄 Running large scale benchmarks (this may take a while)..."
    run_benchmark "Large_Scale" "^BenchmarkLargeScale|^BenchmarkProgressive"
else
    echo "⏩ Skipping large scale benchmarks (use without --short to include)"
fi

echo "📈 Benchmark Summary"
echo "==================="

# Generate summary report
SUMMARY_FILE="$RESULTS_DIR/summary.txt"
{
    echo "LibraVDB HNSW Persistence Benchmark Summary"
    echo "Generated: $(date)"
    echo "=========================================="
    echo ""
    
    echo "🎯 Performance Targets:"
    echo "- Save Speed: >10MB/s for vector data"
    echo "- Load Speed: >15MB/s with validation"
    echo "- Memory Usage: <2x index size during save/load"
    echo "- Search Impact: <5% latency increase during save"
    echo ""
    
    echo "🛡️ Reliability Targets:"
    echo "- Corruption Detection: 100% via CRC32"
    echo "- Atomic Operations: 100% success rate"
    echo "- Recovery Rate: 100% from valid backups"
    echo "- Zero Data Loss: 100% in atomic operations"
    echo ""
    
    echo "📊 Actual Results:"
    echo "=================="
    
    # Extract performance metrics
    for file in "$RESULTS_DIR"/*.txt; do
        if [ -f "$file" ]; then
            echo ""
            echo "From $(basename "$file" .txt):"
            grep -E "(MB/s|μs|success_rate|detection_rate|compatibility|integrity)" "$file" | head -10 | sed 's/^/  /'
        fi
    done
    
} > "$SUMMARY_FILE"

# Display summary
cat "$SUMMARY_FILE"

echo ""
echo "✅ All benchmarks completed!"
echo "📁 Detailed results available in: $RESULTS_DIR"
echo "📋 Summary report: $SUMMARY_FILE"

# Check if we meet performance targets
echo ""
echo "🎯 Performance Target Analysis:"
echo "==============================="

# Simple analysis of results
if grep -q "MB/s" "$RESULTS_DIR"/*.txt 2>/dev/null; then
    echo "✅ Throughput metrics found - check detailed results for target compliance"
else
    echo "⚠️  No throughput metrics found - check benchmark execution"
fi

if grep -q "success_rate.*100" "$RESULTS_DIR"/*.txt 2>/dev/null; then
    echo "✅ 100% success rates detected"
else
    echo "⚠️  Check success rates in detailed results"
fi

if grep -q "detection_rate.*100" "$RESULTS_DIR"/*.txt 2>/dev/null; then
    echo "✅ 100% corruption detection rates detected"
else
    echo "⚠️  Check corruption detection rates in detailed results"
fi

echo ""
echo "🏁 Benchmark run complete!"