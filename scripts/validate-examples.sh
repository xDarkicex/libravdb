#!/bin/bash
set -e

echo "🔍 Validating example code compilation..."

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

# Find all Go files in examples directory
if [ ! -d "examples" ]; then
    print_warning "No examples directory found"
    exit 0
fi

print_status "Found examples directory, validating individual files..."

# Validate each example file individually
EXAMPLES_VALID=0
EXAMPLES_TOTAL=0

for example_file in examples/*.go; do
    if [ -f "$example_file" ]; then
        EXAMPLES_TOTAL=$((EXAMPLES_TOTAL + 1))
        filename=$(basename "$example_file")
        
        print_status "Validating $filename..."
        
        # Try to build the individual file
        if go build -o /tmp/example_test "$example_file" 2>/dev/null; then
            print_success "$filename compiles successfully"
            EXAMPLES_VALID=$((EXAMPLES_VALID + 1))
            # Clean up the temporary binary
            rm -f /tmp/example_test
        else
            print_error "$filename has compilation errors"
            echo "Attempting to show specific errors:"
            go build -o /tmp/example_test "$example_file" 2>&1 | head -10
        fi
    fi
done

# Summary
echo ""
print_status "Example Validation Summary:"
echo "  ✅ Valid examples: $EXAMPLES_VALID"
echo "  📁 Total examples: $EXAMPLES_TOTAL"

if [ $EXAMPLES_VALID -eq $EXAMPLES_TOTAL ]; then
    print_success "All examples compile successfully! 🎉"
    exit 0
else
    FAILED=$((EXAMPLES_TOTAL - EXAMPLES_VALID))
    print_error "$FAILED examples failed to compile"
    exit 1
fi