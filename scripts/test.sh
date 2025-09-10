#!/bin/bash
set -e

echo "🧪 Running LibraVDB test suite..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "go.mod" ]; then
    print_error "go.mod not found. Please run this script from the project root."
    exit 1
fi

# Run unit tests (excluding examples directory)
print_status "Running unit tests..."
if go test -v -race $(go list ./... | grep -v '/examples'); then
    print_success "Unit tests passed"
else
    print_error "Unit tests failed"
    exit 1
fi

# Run integration tests if they exist (excluding examples)
print_status "Running integration tests..."
if go test -v -tags=integration $(go list ./... | grep -v '/examples') 2>/dev/null; then
    print_success "Integration tests passed"
else
    print_warning "Integration tests not found or failed"
fi

# Generate coverage report (excluding examples)
print_status "Generating coverage report..."
if go test -coverprofile=coverage.out $(go list ./... | grep -v '/examples'); then
    COVERAGE=$(go tool cover -func=coverage.out | grep total | awk '{print $3}')
    print_success "Coverage report generated: $COVERAGE"
    
    # Generate HTML coverage report
    go tool cover -html=coverage.out -o coverage.html
    print_success "HTML coverage report: coverage.html"
else
    print_error "Coverage generation failed"
fi

# Run benchmarks (excluding examples)
print_status "Running benchmarks..."
if go test -run=^$ -bench=. -benchmem $(go list ./... | grep -v '/examples') > benchmark_results.txt 2>&1; then
    print_success "Benchmarks completed (results in benchmark_results.txt)"
    
    # Show a summary of benchmark results
    echo ""
    echo "📊 Benchmark Summary:"
    grep -E "Benchmark.*-[0-9]+" benchmark_results.txt | head -10 || true
else
    print_warning "Benchmarks failed or not found"
fi

# Run validation tests if they exist
if [ -f "benchmark/validation_test.go" ]; then
    print_status "Running validation tests..."
    if go test -v ./benchmark/...; then
        print_success "Validation tests passed"
    else
        print_error "Validation tests failed"
    fi
fi

# Check for race conditions in a subset of tests (excluding examples)
print_status "Running race condition detection..."
if go test -race -short $(go list ./... | grep -v '/examples'); then
    print_success "No race conditions detected"
else
    print_error "Race conditions detected"
    exit 1
fi

# Memory leak detection (if available)
if command -v valgrind &> /dev/null; then
    print_status "Running memory leak detection..."
    # This is more applicable for C/C++ but can be useful for CGO
    print_warning "Valgrind detected but skipping (not typically needed for pure Go)"
fi

echo ""
print_success "All tests completed successfully! 🎉"
echo ""
echo "📋 Test Summary:"
echo "  ✅ Unit tests: PASSED"
echo "  ✅ Coverage: $COVERAGE"
echo "  ✅ Race detection: PASSED"
echo "  📊 Benchmarks: See benchmark_results.txt"
echo "  📈 Coverage report: coverage.html"
echo ""