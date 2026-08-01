# LibraVDB Perl SDK

A high-performance Perl SDK for LibraVDB powered by `FFI::Platypus`.

## Prerequisites

- Perl 5.20+
- `FFI::Platypus`
- `JSON::PP` (Core)

```bash
cpanm FFI::Platypus
```

## Installation

Ensure `libravdb.dylib` (macOS), `libravdb.so` (Linux), or `libravdb.dll` (Windows) is available in your dynamic linker path or set via the `LIBRAVDB_LIBRARY_PATH` environment variable.

## Usage

```perl
use strict;
use warnings;
use LibraVDB::Database;
use LibraVDB::Filter;

# 1. Initialize Database
my $db = LibraVDB::Database->new("./my_db");

# 2. Create or Get a Collection
my $col = $db->create_collection('items', 3);

# 3. Insert Vectors
$col->insert('doc1', [0.1, 0.2, 0.3], '{"tag": "A"}');

# 4. Batch Insertion
$col->insert_batch(
    ['doc2', 'doc3'],
    [
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ],
    ['{"tag": "B"}', '{"tag": "C"}']
);

# 5. Query with AST Filtering
my $filter = LibraVDB::Filter->eq('tag', 'A');
my $filter_json = LibraVDB::Filter->as_json($filter);

my $results = $col->search([0.1, 0.2, 0.3], 10, $filter_json);
print "$results\n";
```

## Advanced Filtering

LibraVDB Perl SDK supports complex filter trees cleanly using object constructors natively supported by `JSON::PP`:

```perl
my $filter = LibraVDB::Filter->and(
    LibraVDB::Filter->eq('status', 'active'),
    LibraVDB::Filter->in('category', ['electronics', 'home'])
);

my $json_str = LibraVDB::Filter->as_json($filter);
```
