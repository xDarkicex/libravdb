# Contributing to LibraVDB

Thank you for your interest in contributing to LibraVDB! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Contribution Types](#contribution-types)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Performance Considerations](#performance-considerations)
- [Security](#security)
- [Community](#community)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to help us maintain a welcoming and inclusive community.

## Getting Started

### Prerequisites

- **Go 1.25+**: LibraVDB requires Go 1.25 or later
- **Git**: For version control
- **GitHub Account**: For submitting contributions

### Development Setup

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/libravdb.git
   cd libravdb
   ```

2. **Set Up Development Environment**
   ```bash
   # Run the setup script
   ./scripts/setup.sh
   
   # Verify installation
   go build ./...
   go test ./...
   ```

3. **Configure Git**
   ```bash
   # Add upstream remote
   git remote add upstream https://github.com/xDarkicex/libravdb.git
   
   # Configure your identity
   git config user.name "Your Name"
   git config user.email "your.email@example.com"
   ```

### Project Structure

```
libravdb/
├── libravdb/          # Main library package
├── internal/          # Internal packages
│   ├── index/         # Indexing algorithms (HNSW, IVF-PQ, Flat)
│   ├── storage/       # Storage layer (LSM, WAL, Segments)
│   ├── memory/        # Memory management
│   ├── filter/        # Query filtering
│   ├── quant/         # Quantization algorithms
│   ├── obs/           # Observability (metrics, health)
│   └── util/          # Utilities
├── examples/          # Usage examples
├── tests/             # Integration tests
├── benchmark/         # Performance benchmarks
├── docs/              # Documentation
└── scripts/           # Development scripts
```

## Development Workflow

### 1. Issue First

Before starting work, please:

- **Check existing issues** to avoid duplication
- **Create an issue** for bugs, features, or improvements
- **Discuss your approach** in the issue before implementing
- **Get approval** for significant changes

### 2. Branch Strategy

```bash
# Create a feature branch from main
git checkout main
git pull upstream main
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description

# Or for documentation
git checkout -b docs/documentation-update
```

### 3. Development Cycle

```bash
# Make your changes
# ...

# Run tests frequently
go test ./...

# Run linting and formatting
./scripts/lint.sh

# Run comprehensive tests
./scripts/test.sh

# Commit your changes
git add .
git commit -m "feat: add new feature description"
```

### 4. Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```bash
git commit -m "feat(index): add IVF-PQ quantization support"
git commit -m "fix(storage): resolve WAL corruption on crash recovery"
git commit -m "docs: update API reference for new search options"
git commit -m "perf(hnsw): optimize distance calculations with SIMD"
```

## Contribution Types

### 🐛 Bug Reports

When reporting bugs, please include:

- **Clear description** of the issue
- **Steps to reproduce** the problem
- **Expected vs actual behavior**
- **Environment details** (Go version, OS, etc.)
- **Minimal code example** if applicable
- **Stack traces** or error messages

Use our [bug report template](.github/ISSUE_TEMPLATE/bug_report.md).

### ✨ Feature Requests

For new features:

- **Describe the use case** and motivation
- **Provide examples** of how it would be used
- **Consider alternatives** and explain why this approach is best
- **Discuss performance implications**
- **Check if it aligns** with project goals

Use our [feature request template](.github/ISSUE_TEMPLATE/feature_request.md).

### 📚 Documentation

Documentation improvements are always welcome:

- **API documentation** in code comments
- **User guides** and tutorials
- **Architecture documentation**
- **Performance guides**
- **Example code** and use cases

### 🚀 Performance Improvements

For performance contributions:

- **Benchmark before and after** your changes
- **Profile the code** to identify bottlenecks
- **Consider memory usage** and allocations
- **Test with realistic datasets**
- **Document performance characteristics**

## Pull Request Process

### 1. Pre-submission Checklist

- [ ] **Issue exists** and is referenced in PR
- [ ] **Tests pass** locally (`./scripts/test.sh`)
- [ ] **Code is formatted** (`./scripts/lint.sh`)
- [ ] **Documentation updated** if needed
- [ ] **Benchmarks run** for performance changes
- [ ] **Breaking changes documented**

### 2. Pull Request Template

```markdown
## Description
Brief description of changes

## Related Issue
Fixes #123

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Breaking change

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Benchmarks run (if applicable)

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests pass locally
```

### 3. Review Process

1. **Automated Checks**: CI/CD runs tests, linting, and benchmarks
2. **Code Review**: Maintainers review code quality, design, and tests
3. **Discussion**: Address feedback and make requested changes
4. **Approval**: At least one maintainer approval required
5. **Merge**: Squash and merge to main branch

### 4. After Merge

- **Delete your branch** after successful merge
- **Update your fork** with the latest changes
- **Close related issues** if they're fully resolved

## Coding Standards

### Go Style Guide

We follow the [Go Code Review Comments](https://github.com/golang/go/wiki/CodeReviewComments) and [Effective Go](https://golang.org/doc/effective_go.html).

#### Key Principles

1. **Simplicity**: Prefer simple, readable code over clever solutions
2. **Consistency**: Follow existing patterns in the codebase
3. **Performance**: Consider performance implications of your changes
4. **Error Handling**: Always handle errors appropriately
5. **Documentation**: Document public APIs and complex logic

#### Code Formatting

```bash
# Format code
go fmt ./...

# Organize imports
goimports -w .

# Run linter
golangci-lint run
```

#### Naming Conventions

```go
// Good: Clear, descriptive names
func (c *Collection) InsertVector(ctx context.Context, id string, vector []float32) error

// Bad: Unclear abbreviations
func (c *Collection) InsVec(ctx context.Context, i string, v []float32) error

// Good: Interface names
type VectorSearcher interface {
    Search(ctx context.Context, query []float32, k int) ([]*SearchResult, error)
}

// Good: Error variables
var (
    ErrInvalidDimension = errors.New("invalid vector dimension")
    ErrCollectionClosed = errors.New("collection is closed")
)
```

#### Error Handling

```go
// Good: Wrap errors with context
func (c *Collection) Insert(ctx context.Context, id string, vector []float32) error {
    if err := c.validateVector(vector); err != nil {
        return fmt.Errorf("vector validation failed: %w", err)
    }
    
    if err := c.index.Insert(ctx, &VectorEntry{ID: id, Vector: vector}); err != nil {
        return fmt.Errorf("index insertion failed: %w", err)
    }
    
    return nil
}

// Good: Use structured errors for public APIs
type ValidationError struct {
    Field   string
    Value   interface{}
    Message string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("validation failed for field %s: %s", e.Field, e.Message)
}
```

#### Documentation

```go
// Package documentation
// Package libravdb provides a high-performance vector database library
// for Go applications with support for HNSW indexing and advanced filtering.
package libravdb

// Function documentation with examples
// Insert adds a vector to the collection with optional metadata.
//
// The vector must match the collection's configured dimension.
// Metadata can be used for filtering during search operations.
//
// Example:
//   vector := []float32{0.1, 0.2, 0.3}
//   metadata := map[string]interface{}{"category": "document"}
//   err := collection.Insert(ctx, "doc1", vector, metadata)
//   if err != nil {
//       log.Fatal(err)
//   }
func (c *Collection) Insert(ctx context.Context, id string, vector []float32, metadata map[string]interface{}) error {
    // Implementation...
}
```

## Testing Guidelines

### Test Structure

```go
func TestCollectionInsert(t *testing.T) {
    // Setup
    collection := createTestCollection(t)
    defer collection.Close()
    
    // Test cases
    tests := []struct {
        name     string
        id       string
        vector   []float32
        metadata map[string]interface{}
        wantErr  bool
    }{
        {
            name:    "valid insertion",
            id:      "test1",
            vector:  []float32{1.0, 2.0, 3.0},
            wantErr: false,
        },
        {
            name:    "invalid dimension",
            id:      "test2",
            vector:  []float32{1.0, 2.0}, // Wrong dimension
            wantErr: true,
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            err := collection.Insert(context.Background(), tt.id, tt.vector, tt.metadata)
            if (err != nil) != tt.wantErr {
                t.Errorf("Insert() error = %v, wantErr %v", err, tt.wantErr)
            }
        })
    }
}
```

### Test Categories

1. **Unit Tests**: Test individual functions and methods
   ```bash
   go test ./libravdb -v
   ```

2. **Integration Tests**: Test component interactions
   ```bash
   go test -tags=integration ./tests -v
   ```

3. **Benchmark Tests**: Performance testing
   ```bash
   go test -bench=. -benchmem ./...
   ```

### Test Requirements

- **Coverage**: Aim for >80% test coverage for new code
- **Edge Cases**: Test boundary conditions and error cases
- **Concurrency**: Test concurrent access where applicable
- **Performance**: Include benchmarks for performance-critical code
- **Integration**: Test interactions between components

### Running Tests

```bash
# Run all tests
./scripts/test.sh

# Run specific package tests
go test ./libravdb -v

# Run with race detection
go test -race ./...

# Run benchmarks
go test -bench=. -benchmem ./...

# Generate coverage report
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out
```

## Documentation

### Types of Documentation

1. **Code Documentation**: GoDoc comments for public APIs
2. **User Documentation**: Guides and tutorials in `docs/`
3. **Architecture Documentation**: Design documents in `docs/design/`
4. **Examples**: Working code examples in `examples/`

### Documentation Standards

- **Clear and Concise**: Write for your audience
- **Examples**: Include practical code examples
- **Up-to-date**: Keep documentation synchronized with code
- **Comprehensive**: Cover all public APIs and important concepts

### Writing Guidelines

```go
// Good: Clear, comprehensive documentation
// SearchWithFilter performs a vector similarity search with metadata filtering.
//
// The query vector must match the collection's dimension. The k parameter
// specifies the maximum number of results to return. Filters are applied
// after the vector search to refine results based on metadata.
//
// Returns results sorted by similarity score (lower is more similar for
// distance metrics, higher for similarity metrics).
//
// Example:
//   results, err := collection.SearchWithFilter(ctx, queryVector, 10, 
//       filter.Eq("category", "documents"))
//   if err != nil {
//       return err
//   }
//   for _, result := range results.Results {
//       fmt.Printf("ID: %s, Score: %.3f\n", result.ID, result.Score)
//   }
func (c *Collection) SearchWithFilter(ctx context.Context, query []float32, k int, filters ...filter.Filter) (*SearchResults, error)
```

## Performance Considerations

### Benchmarking

Always benchmark performance-critical changes:

```go
func BenchmarkCollectionInsert(b *testing.B) {
    collection := createBenchmarkCollection(b)
    defer collection.Close()
    
    vectors := generateRandomVectors(b.N, 768)
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        collection.Insert(context.Background(), fmt.Sprintf("vec_%d", i), vectors[i], nil)
    }
}

func BenchmarkCollectionSearch(b *testing.B) {
    collection := createBenchmarkCollection(b)
    defer collection.Close()
    
    // Insert test data
    insertBenchmarkData(b, collection, 100000)
    
    queryVector := generateRandomVector(768)
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        collection.Search(context.Background(), queryVector, 10)
    }
}
```

### Performance Guidelines

1. **Memory Allocations**: Minimize allocations in hot paths
2. **CPU Usage**: Profile CPU-intensive operations
3. **I/O Operations**: Batch I/O operations when possible
4. **Concurrency**: Use goroutines judiciously
5. **Data Structures**: Choose appropriate data structures

### Profiling

```bash
# CPU profiling
go test -cpuprofile=cpu.prof -bench=BenchmarkSearch ./...
go tool pprof cpu.prof

# Memory profiling
go test -memprofile=mem.prof -bench=BenchmarkInsert ./...
go tool pprof mem.prof

# Trace analysis
go test -trace=trace.out -bench=BenchmarkBatch ./...
go tool trace trace.out
```

## Security

### Security Guidelines

1. **Input Validation**: Validate all external inputs
2. **Error Messages**: Don't leak sensitive information in errors
3. **Dependencies**: Keep dependencies updated
4. **Memory Safety**: Avoid buffer overflows and memory leaks

### Reporting Security Issues

Please report security vulnerabilities privately to the maintainers:

- **Email**: security@libravdb.org (if available)
- **GitHub**: Use private vulnerability reporting
- **Response Time**: We aim to respond within 48 hours

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussion
- **Pull Requests**: Code contributions and reviews

### Getting Help

1. **Documentation**: Check the docs first
2. **Examples**: Look at example code
3. **Issues**: Search existing issues
4. **Discussions**: Ask in GitHub Discussions
5. **Stack Overflow**: Tag questions with `libravdb`

### Recognition

Contributors are recognized in:

- **CONTRIBUTORS.md**: List of all contributors
- **Release Notes**: Major contributions highlighted
- **GitHub**: Contributor statistics and graphs

## License

By contributing to LibraVDB, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).

---

Thank you for contributing to LibraVDB! Your contributions help make vector search accessible and performant for the Go community. 🚀