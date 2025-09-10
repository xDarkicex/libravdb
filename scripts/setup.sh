#!/bin/bash
set -e

echo "🚀 Setting up LibraVDB development environment..."

# Check Go version
if ! command -v go &> /dev/null; then
    echo "❌ Go is not installed. Please install Go 1.25 or later."
    exit 1
fi

GO_VERSION=$(go version | grep -oE 'go[0-9]+\.[0-9]+' | sed 's/go//')
REQUIRED_VERSION="1.25"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$GO_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Go version $GO_VERSION is too old. Please install Go $REQUIRED_VERSION or later."
    exit 1
fi

echo "✅ Go version $GO_VERSION detected"

# Install development tools
echo "📦 Installing development tools..."

echo "  - Installing golangci-lint..."
if ! command -v golangci-lint &> /dev/null; then
    go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
    echo "    ✅ golangci-lint installed"
else
    echo "    ✅ golangci-lint already installed"
fi

echo "  - Installing goimports..."
if ! command -v goimports &> /dev/null; then
    go install golang.org/x/tools/cmd/goimports@latest
    echo "    ✅ goimports installed"
else
    echo "    ✅ goimports already installed"
fi

# Generate test data if script exists
if [ -f "scripts/generate_testdata.go" ]; then
    echo "🔧 Generating test data..."
    go run scripts/generate_testdata.go
    echo "✅ Test data generated"
fi

# Download dependencies
echo "📥 Downloading dependencies..."
go mod download
echo "✅ Dependencies downloaded"

# Verify installation
echo "🔍 Verifying installation..."
go build ./...
echo "✅ Build successful"

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "Available commands:"
echo "  go build ./...              # Build the library"
echo "  go test ./...               # Run tests"
echo "  go test -bench=. ./...      # Run benchmarks"
echo "  ./scripts/test.sh           # Run comprehensive tests"
echo "  ./scripts/lint.sh           # Format and lint code"
echo "  golangci-lint run           # Run linter"
echo ""
echo "Happy coding! 🚀"