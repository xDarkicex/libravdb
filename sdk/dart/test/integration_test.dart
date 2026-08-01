import 'dart:io';
import 'package:test/test.dart';
import 'package:libravdb/libravdb.dart';

void main() {
  final dbPath = './demo_db_dart';

  setUpAll(() {
    final dir = Directory(dbPath);
    if (dir.existsSync()) {
      dir.deleteSync(recursive: true);
    }
  });

  tearDownAll(() {
    final dir = Directory(dbPath);
    if (dir.existsSync()) {
      dir.deleteSync(recursive: true);
    }
  });

  test('Full Integration Test', () {
    print('Initializing LibraVDB...');
    final db = LibraVDB(dbPath);

    print('Testing Ping...');
    db.ping();

    print('Setting Memory Limit...');
    db.setMemoryLimit(10 * 1024 * 1024);

    print('Creating Collection docs...');
    final col = db.createCollection('docs', 3);

    final collections = db.listCollections();
    expect(collections.contains('docs'), isTrue);

    print('Testing InsertBatch with 1000 vectors...');
    final ids = <String>[];
    final vectors = <List<double>>[];
    final metadata = <Map<String, dynamic>>[];

    for (int i = 0; i < 1000; i++) {
      ids.add('vec_$i');
      vectors.add([0.1, 0.2, 0.3]);
      metadata.add({'source': 'dart_test', 'index': i, 'active': i % 2 == 0});
    }

    col.insertBatch(ids, vectors, metadata: metadata);

    print('Testing Update...');
    col.update('vec_0', [1.0, 1.0, 1.0], metadata: {'updated': true});

    print('Testing Get...');
    final rec = col.get('vec_0');
    expect(rec, isNotNull);
    expect(rec['id'], equals('vec_0'));

    print('Testing Search...');
    final filter = Filter.eq('active', true);
    final results = col.search([1.0, 1.0, 1.0], 5, filter: filter);
    expect(results is List, isTrue);
    expect(results.length, greaterThan(0));

    print('Testing Scan...');
    final scanned = col.scan(0, 2);
    expect(scanned is List, isTrue);
    expect(scanned.length, equals(2));

    print('Testing DeleteBatch...');
    col.deleteBatch(ids.sublist(0, 500));

    final count = col.count();
    expect(count, equals(500));

    print('Testing Vacuum...');
    db.vacuum();

    print('Testing DropDatabase...');
    db.dropDatabase();
    
    db.close();

    final dirAfter = Directory(dbPath);
    expect(dirAfter.existsSync(), isFalse);

    print('Integration Test Passed!');
  });
}
