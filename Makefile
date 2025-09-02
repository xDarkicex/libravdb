.PHONY: build test lint clean install-tools benchmark

# Build the library
build:
	go build ./...

# Run all tests
test:
	go test -v -race ./...

# Run tests with coverage
test-coverage:
	go test -v -race -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html

# Run benchmarks
benchmark:
	go test -v -run=^$$ -bench=. -benchmem ./...

# Lint the code
lint:
	golangci-lint run

# Format code
fmt:
	go fmt ./...
	goimports -w .

# Install development tools
install-tools:
	go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
	go install golang.org/x/tools/cmd/goimports@latest

# Clean build artifacts
clean:
	go clean ./...
	rm -f coverage.out coverage.html

# Generate test data
generate-testdata:
	go run scripts/generate_testdata.go

# Run integration tests
test-integration:
	go test -v -tags=integration ./...

# Development setup
setup: install-tools generate-testdata
	@echo "Development environment ready!"

# Quick development cycle
dev: fmt lint test
	@echo "Development cycle complete!"
