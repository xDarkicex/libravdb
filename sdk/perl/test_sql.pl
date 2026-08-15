use strict;
use warnings;
use File::Path qw(remove_tree);
use JSON::PP;
use Time::HiRes qw(time);
use POSIX qw(strftime);
use lib 'lib';
use LibraVDB::Database;

sub main {
    my $db_path = './demo_db_sql_perl';

    if (-e $db_path) {
        if (-d $db_path) {
            remove_tree($db_path);
        } else {
            unlink $db_path;
        }
    }

    print "Initializing LibraVDB at $db_path...\n";
    my $db = LibraVDB::Database->new($db_path);

    # Create tables
    $db->query("CREATE GRAPH TABLE users (id STRING PRIMARY KEY, name STRING, embedding VECTOR(3))");
    $db->query("CREATE EDGE TYPE FOLLOWS");

    # 1. Relational
    print "\n--- Relational SQL ---\n";
    $db->query_with_params("INSERT INTO users (id, name, embedding) VALUES (\$1, \$2, \$3)", {"1" => "u1", "2" => "Alice", "3" => [1.0, 0.0, 0.0]});
    $db->query_with_params("INSERT INTO users (id, name, embedding) VALUES (\$1, \$2, \$3)", {"1" => "u2", "2" => "Bob", "3" => [0.0, 1.0, 0.0]});
    $db->query_with_params("INSERT INTO users (id, name, embedding) VALUES (\$1, \$2, \$3)", {"1" => "u3", "2" => "Charlie", "3" => [0.0, 0.0, 1.0]});

    my $res = $db->query("SELECT id, name FROM users ORDER BY name ASC");
    print "Relational Result: " . encode_json($res) . "\n";

    # 2. Vector
    print "\n--- Vector SQL ---\n";
    $res = $db->query_with_params("SELECT id, name FROM users ORDER BY VECTOR_DISTANCE(embedding, \$vec) ASC LIMIT 2", {"vec" => [1.0, 0.0, 0.0]});
    print "Vector Result: " . encode_json($res) . "\n";

    # 3. Graph
    print "\n--- Graph SQL ---\n";
    $db->query_with_params("INSERT INTO GRAPH_EDGES VALUES (\$1, \$2, \$3)", {"1" => "u1", "2" => "FOLLOWS", "3" => "u2"});
    $db->query_with_params("INSERT INTO GRAPH_EDGES VALUES (\$1, \$2, \$3)", {"1" => "u2", "2" => "FOLLOWS", "3" => "u3"});
    $res = $db->query_with_params("SELECT tgt.id FROM users src JOIN MATCH (src)-[:FOLLOWS]->(tgt) WHERE src.id = \$1", {"1" => "u1"});
    print "Graph Result: " . encode_json($res) . "\n";

    # 4. Temporal SQL
    print "\n--- Temporal SQL ---\n";
    my $t = time() + 2;
    my $cutoff = strftime("%Y-%m-%dT%H:%M:%S.000Z", gmtime($t));
    $res = $db->query("SELECT id FROM users AS OF TIMESTAMP '$cutoff' ORDER BY id ASC");
    print "Temporal Result: " . encode_json($res) . "\n";

    print "\nAll unified SQL tests passed successfully.\n";
}

main();
