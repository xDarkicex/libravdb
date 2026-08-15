# LibraVDB PHP SDK

A high-performance PHP SDK for LibraVDB powered by the native `FFI` extension (PHP 7.4+).

## Unified SQL Engine

This SDK natively integrates with the LibraVDB Unified SQL Engine. You can execute expressive queries seamlessly across multiple paradigms:
- **Relational SQL**: Standard ANSI SQL data manipulation.
- **Vector SQL**: Order by `VECTOR_DISTANCE` and perform similarity matching.
- **Graph SQL**: Perform Cypher-like graph traversals using `JOIN MATCH (src)-[:EDGE]->(tgt)`.
- **Temporal SQL**: Query historical database snapshots using `AS OF TIMESTAMP`.

## Prerequisites

- PHP 7.4 or higher
- The `ffi` extension must be enabled. Ensure `ffi.enable=true` is set in your `php.ini` or passed via CLI (`php -d ffi.enable=true`).

## Installation

You can include the `src` directory directly or add it to your composer autoload map. 

Ensure `libravdb.dylib` (macOS), `libravdb.so` (Linux), or `libravdb.dll` (Windows) is available in your dynamic linker path, or explicitly set via the `LIBRAVDB_LIBRARY_PATH` environment variable.

## Usage

```php
require_once 'src/Database.php';
require_once 'src/Filter.php';

use LibraVDB\Database;
use LibraVDB\Filter;

// 1. Initialize Database
$db = new Database("./my_db");

// 2. Create or Get a Collection
$col = $db->createCollection("items", 3);

// 3. Insert Vectors
$col->insert("doc1", [0.1, 0.2, 0.3], '{"tag": "A"}');

// 4. Batch Insertion (Zero-Copy FFI Memory Mapping)
$col->insertBatch(
    ["doc2", "doc3"],
    [
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ],
    ['{"tag": "B"}', '{"tag": "C"}']
);

// 5. Query with AST Filtering
$filter = Filter::eq("tag", "A");
$filterJson = Filter::toJson($filter);

$results = $col->search([0.1, 0.2, 0.3], 10, $filterJson);
echo $results . "\n";
```

## Advanced Filtering

LibraVDB PHP SDK supports complex filter trees cleanly using nested arrays mapped natively to JSON:

```php
$filter = Filter::and(
    Filter::eq("status", "active"),
    Filter::in("category", ["electronics", "home"])
);

$jsonStr = Filter::toJson($filter);
```
