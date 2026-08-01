use strict;
use warnings;
use Test::More tests => 5;
use File::Path qw( rmtree );
use lib 'lib';
use LibraVDB::Database;
use LibraVDB::Filter;

my $db_path = "./test_db_perl_$$";

# Cleanup before
rmtree($db_path) if -d $db_path;

# 1. Open Database
my $db = eval { LibraVDB::Database->new($db_path) };
ok( !$@ && defined $db, 'Database opened successfully' ) or diag($@);

# 2. Create Collection
my $col = eval { $db->create_collection('test_col', 3) };
ok( !$@ && defined $col, 'Collection created successfully' ) or diag($@);

# 3. Insert and Search
eval {
    $col->insert('1', [1.0, 2.0, 3.0], '{"category": "A"}');
};
ok( !$@, 'Insert single vector successfully' ) or diag($@);

my $filter = LibraVDB::Filter->eq('category', 'A');
my $filter_json = LibraVDB::Filter->as_json($filter);

my $search_res = eval { $col->search([1.0, 2.0, 3.0], 10, $filter_json) };
ok( !$@ && $search_res =~ /"id":"1"/, 'Search found the vector' ) or diag($@);

# 4. Batch Insert
eval {
    $col->insert_batch(
        ['2', '3'],
        [
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ],
        ['{"category": "B"}', '{"category": "C"}']
    );
};
ok( !$@, 'Batch insert successfully' ) or diag($@);

# Cleanup after
undef $col;
undef $db;
rmtree($db_path) if -d $db_path;
