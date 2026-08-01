<?php

require_once __DIR__ . '/../src/Database.php';
require_once __DIR__ . '/../src/Filter.php';

use LibraVDB\Database;
use LibraVDB\Filter;

$dbPath = "./test_db_php_" . getmypid();

// 1. Open Database
echo "Opening DB...\n";
$db = new Database($dbPath);
if (!$db) {
    die("Failed to open DB\n");
}

// 2. Create Collection
echo "Creating Collection...\n";
$col = $db->createCollection("test_col", 3);
if (!$col) {
    die("Failed to create collection\n");
}

// 3. Insert and Search
echo "Inserting...\n";
$col->insert("1", [1.0, 2.0, 3.0], '{"category": "A"}');

$filter = Filter::eq("category", "A");
$filterJson = Filter::toJson($filter);

echo "Searching...\n";
$searchRes = $col->search([1.0, 2.0, 3.0], 10, $filterJson);
if (!str_contains($searchRes, '"id":"1"')) {
    die("Search results missing inserted vector\n");
}

// 4. Batch Insert
echo "Batch Inserting...\n";
$col->insertBatch(
    ["2", "3"],
    [
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ],
    ['{"category": "B"}', '{"category": "C"}']
);

// 5. Scan
echo "Scanning...\n";
$scanRes = $col->scan();
if (!str_contains($scanRes, '"id":"2"') || !str_contains($scanRes, '"id":"3"')) {
    die("Scan results missing batch inserted vectors\n");
}

// Cleanup
$db->close();
system("rm -rf " . escapeshellarg($dbPath));

echo "All PHP integration tests passed!\n";
