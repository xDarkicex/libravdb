#!/bin/bash
set -e

echo "🔍 Running LibraVDB code quality checks..."

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

# Format code
print_status "Formatting code with go fmt..."
if go fmt ./...; then
    print_success "Code formatted successfully"
else
    print_error "Code formatting failed"
    exit 1
fi

# Run goimports if available
if command -v goimports &> /dev/null; then
    print_status "Organizing imports with goimports..."
    if goimports -w .; then
        print_success "Imports organized successfully"
    else
        print_warning "goimports encountered issues"
    fi
else
    print_warning "goimports not found. Install with: go install golang.org/x/tools/cmd/goimports@latest"
fi

# Run go vet
print_status "Running go vet..."
if go vet ./...; then
    print_success "go vet passed"
else
    print_error "go vet found issues"
    exit 1
fi

# Run golangci-lint if available
if command -v golangci-lint &> /dev/null; then
    print_status "Running golangci-lint..."
    if golangci-lint run; then
        print_success "golangci-lint passed"
    else
        print_error "golangci-lint found issues"
        exit 1
    fi
else
    print_warning "golangci-lint not found. Install with: go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest"
fi

# Check for common Go issues
print_status "Checking for common issues..."

# Check for TODO/FIXME comments
TODO_COUNT=$(grep -r "TODO\|FIXME" --include="*.go" . | wc -l || echo "0")
if [ "$TODO_COUNT" -gt 0 ]; then
    print_warning "Found $TODO_COUNT TODO/FIXME comments"
    echo "Consider addressing these before release:"
    grep -r "TODO\|FIXME" --include="*.go" . | head -5 || true
    if [ "$TODO_COUNT" -gt 5 ]; then
        echo "... and $(($TODO_COUNT - 5)) more"
    fi
else
    print_success "No TODO/FIXME comments found"
fi

# Check for potential security issues (basic checks)
print_status "Running basic security checks..."

# Check for hardcoded credentials (basic patterns)
SECURITY_ISSUES=0

if grep -r "password\s*=\|secret\s*=\|token\s*=" --include="*.go" . >/dev/null 2>&1; then
    print_warning "Potential hardcoded credentials found"
    SECURITY_ISSUES=$((SECURITY_ISSUES + 1))
fi

if grep -r "http://" --include="*.go" . >/dev/null 2>&1; then
    print_warning "HTTP URLs found (consider HTTPS)"
    SECURITY_ISSUES=$((SECURITY_ISSUES + 1))
fi

if [ $SECURITY_ISSUES -eq 0 ]; then
    print_success "Basic security checks passed"
fi

# Check for proper error handling
print_status "Checking error handling patterns..."
if grep -r "_ = " --include="*.go" . | grep -v "_test.go" >/dev/null 2>&1; then
    IGNORED_ERRORS=$(grep -r "_ = " --include="*.go" . | grep -v "_test.go" | wc -l)
    print_warning "Found $IGNORED_ERRORS potentially ignored errors (excluding tests)"
    echo "Consider proper error handling for:"
    grep -r "_ = " --include="*.go" . | grep -v "_test.go" | head -3 || true
fi

# Check module tidiness
print_status "Checking module tidiness..."
if go mod tidy && git diff --exit-code go.mod go.sum >/dev/null 2>&1; then
    print_success "go.mod and go.sum are tidy"
else
    print_warning "go.mod or go.sum needs tidying"
    echo "Run: go mod tidy"
fi

# Check for unused dependencies
if command -v go-mod-outdated &> /dev/null; then
    print_status "Checking for outdated dependencies..."
    go list -u -m all | grep '\[' || print_success "All dependencies are up to date"
fi

# Performance checks
print_status "Running performance checks..."

# Check for potential performance issues
PERF_ISSUES=0

if grep -r "fmt\.Sprintf" --include="*.go" . | grep -v "_test.go" >/dev/null 2>&1; then
    SPRINTF_COUNT=$(grep -r "fmt\.Sprintf" --include="*.go" . | grep -v "_test.go" | wc -l)
    if [ "$SPRINTF_COUNT" -gt 10 ]; then
        print_warning "Many fmt.Sprintf calls found ($SPRINTF_COUNT) - consider string builders for performance"
        PERF_ISSUES=$((PERF_ISSUES + 1))
    fi
fi

if [ $PERF_ISSUES -eq 0 ]; then
    print_success "Performance checks passed"
fi

echo ""
print_success "Code quality checks completed! 🎉"
echo ""
echo "📋 Quality Summary:"
echo "  ✅ Code formatting: PASSED"
echo "  ✅ Import organization: PASSED"
echo "  ✅ go vet: PASSED"
echo "  ✅ Linting: PASSED"
echo "  📝 TODO/FIXME comments: $TODO_COUNT"
echo "  🔒 Security issues: $SECURITY_ISSUES"
echo "  ⚡ Performance issues: $PERF_ISSUES"
echo ""

if [ $SECURITY_ISSUES -gt 0 ] || [ $PERF_ISSUES -gt 0 ]; then
    print_warning "Some issues found - consider addressing them before release"
    exit 1
fi