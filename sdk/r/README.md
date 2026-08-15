# LibraVDB R SDK

The official R SDK for LibraVDB, providing an elegant and high-performance Object-Oriented (`R6`) interface for the global data science and statistical computing community.

This SDK leverages **Rcpp** to create native C++ extensions that bind directly to the underlying LibraVDB C-Shared library (`libravdb.dylib` / `libravdb.so`).

## Unified SQL Engine

This SDK natively integrates with the LibraVDB Unified SQL Engine. You can execute expressive queries seamlessly across multiple paradigms:
- **Relational SQL**: Standard ANSI SQL data manipulation.
- **Vector SQL**: Order by `VECTOR_DISTANCE` and perform similarity matching.
- **Graph SQL**: Perform Cypher-like graph traversals using `JOIN MATCH (src)-[:EDGE]->(tgt)`.
- **Temporal SQL**: Query historical database snapshots using `AS OF TIMESTAMP`.

## Features
- **Zero-Copy Rcpp Bridge**: Uses `Rcpp` to pass data across the C-Shared boundary without JVM or Python overhead.
- **Data Frame Integration**: Fully utilizes `jsonlite` so that returned SQL rows or vectors are automatically unmarshaled into native R `data.frame` objects and `list()` types.
- **Object-Oriented**: Cleanly namespaced `LibraVDB` and `Collection` classes using R's `R6` ecosystem.

## Installation

Ensure you have the `Rcpp`, `R6`, and `jsonlite` packages installed in your R environment:

```R
install.packages(c("R6", "jsonlite", "Rcpp"))
```

To install the SDK locally from source, navigate to the `sdk/r/libravdb` directory and run:

```bash
R CMD INSTALL .
```
*(Note: Ensure the LibraVDB C-shared library has been compiled via `go build -buildmode=c-shared` in the `cgo/` directory first).*

## Quick Start

```R
library(libravdb)

# Open or create a local single-file database
db <- LibraVDB$new("./my_database")

# Create a collection with vector dimension 3
col <- db$create_collection("docs", 3)

# Insert a vector (metadata lists are automatically converted to JSON)
col$insert("doc1", c(1.0, 2.0, 3.0), list(category = "ai", active = TRUE))

# Unified Vector SQL Query
result <- db$query_with_params(
  "SELECT id FROM docs ORDER BY VECTOR_DISTANCE(embedding, $vec) ASC LIMIT 10",
  list(vec = c(1.0, 2.0, 3.0))
)

# Results are natively returned as R lists / data.frames
print(result$results)

db$close()
```

## Testing
Run the comprehensive integration test suite to verify Relational, Vector, Graph, and Temporal functionality natively in R:

```bash
Rscript tests/test_sql.R
```
