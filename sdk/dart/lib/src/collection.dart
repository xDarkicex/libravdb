import 'dart:convert';
import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'bindings.dart' as lib;
import 'filter.dart';
import 'libravdb.dart';

class Collection {
  final int _handle;
  final int _dim;

  Collection(this._handle, this._dim);

  void _checkError(Pointer<Utf8> errPtr, String opName) {
    if (errPtr != nullptr) {
      final msg = errPtr.toDartString();
      lib.freeString(errPtr);
      if (msg.startsWith('error: ')) {
        throw LibraException('$opName failed: ${msg.substring(7)}');
      }
      throw LibraException('$opName failed: $msg');
    }
  }

  dynamic _parseResult(Pointer<Utf8> resPtr, String opName) {
    if (resPtr == nullptr) {
      return null;
    }
    final msg = resPtr.toDartString();
    lib.freeString(resPtr);

    if (msg.isEmpty) {
      return null;
    }

    try {
      final node = jsonDecode(msg);
      if (node is Map && node.containsKey('error')) {
        throw LibraException('$opName failed: ${node['error']}');
      }
      return node;
    } catch (e) {
      if (e is LibraException) rethrow;
      throw LibraException('JSON Parse error: $e');
    }
  }

  void insert(String id, List<double> vector, {Map<String, dynamic>? metadata}) {
    if (vector.length != _dim) throw LibraException('Vector dimension mismatch');
    
    final idPtr = id.toNativeUtf8();
    final metaStr = metadata != null ? jsonEncode(metadata) : '';
    final metaPtr = metaStr.toNativeUtf8();
    
    final vecPtr = malloc.allocate<Float>(_dim * sizeOf<Float>());
    for (int i = 0; i < _dim; i++) {
      vecPtr[i] = vector[i];
    }

    final errPtr = lib.insertVector(_handle, idPtr, vecPtr, _dim, metaPtr);
    
    malloc.free(vecPtr);
    malloc.free(idPtr);
    malloc.free(metaPtr);
    
    _checkError(errPtr, 'Insert');
  }

  void update(String id, List<double> vector, {Map<String, dynamic>? metadata}) {
    if (vector.length != _dim) throw LibraException('Vector dimension mismatch');
    
    final idPtr = id.toNativeUtf8();
    final metaStr = metadata != null ? jsonEncode(metadata) : '';
    final metaPtr = metaStr.toNativeUtf8();
    
    final vecPtr = malloc.allocate<Float>(_dim * sizeOf<Float>());
    for (int i = 0; i < _dim; i++) {
      vecPtr[i] = vector[i];
    }

    final errPtr = lib.updateVector(_handle, idPtr, vecPtr, _dim, metaPtr);
    
    malloc.free(vecPtr);
    malloc.free(idPtr);
    malloc.free(metaPtr);
    
    _checkError(errPtr, 'Update');
  }

  void updateIfVersion(String id, List<double> vector, int expectedVersion, {Map<String, dynamic>? metadata}) {
    if (vector.length != _dim) throw LibraException('Vector dimension mismatch');
    
    final idPtr = id.toNativeUtf8();
    final metaStr = metadata != null ? jsonEncode(metadata) : '';
    final metaPtr = metaStr.toNativeUtf8();
    
    final vecPtr = malloc.allocate<Float>(_dim * sizeOf<Float>());
    for (int i = 0; i < _dim; i++) {
      vecPtr[i] = vector[i];
    }

    final errPtr = lib.updateVectorIfVersion(_handle, idPtr, vecPtr, _dim, metaPtr, expectedVersion);
    
    malloc.free(vecPtr);
    malloc.free(idPtr);
    malloc.free(metaPtr);
    
    _checkError(errPtr, 'UpdateIfVersion');
  }

  dynamic get(String id) {
    final idPtr = id.toNativeUtf8();
    final resPtr = lib.getVector(_handle, idPtr);
    malloc.free(idPtr);
    return _parseResult(resPtr, 'Get');
  }

  dynamic search(List<double> vector, int k, {Filter? filter}) {
    if (vector.length != _dim) throw LibraException('Vector dimension mismatch');
    
    final filterStr = filter != null ? filter.toJsonString() : '';
    final filterPtr = filterStr.toNativeUtf8();
    
    final vecPtr = malloc.allocate<Float>(_dim * sizeOf<Float>());
    for (int i = 0; i < _dim; i++) {
      vecPtr[i] = vector[i];
    }

    final resPtr = lib.queryVector(_handle, vecPtr, _dim, k, filterPtr);
    
    malloc.free(vecPtr);
    malloc.free(filterPtr);
    
    return _parseResult(resPtr, 'Search');
  }

  dynamic scan(int offset, int limit) {
    final resPtr = lib.scanCollection(_handle, offset, limit);
    return _parseResult(resPtr, 'Scan');
  }

  void insertBatch(List<String> ids, List<List<double>> vectors, {List<Map<String, dynamic>>? metadata}) {
    int count = ids.size;
    if (vectors.length != count) throw LibraException('ids and vectors size mismatch');

    final idsPtr = malloc.allocate<Pointer<Utf8>>(count * sizeOf<Pointer<Utf8>>());
    for (int i = 0; i < count; i++) {
      idsPtr[i] = ids[i].toNativeUtf8();
    }

    final vecsPtr = malloc.allocate<Float>(count * _dim * sizeOf<Float>());
    for (int i = 0; i < count; i++) {
      if (vectors[i].length != _dim) throw LibraException('Vector dimension mismatch at index $i');
      for (int j = 0; j < _dim; j++) {
        vecsPtr[i * _dim + j] = vectors[i][j];
      }
    }

    Pointer<Pointer<Utf8>> metasPtr = nullptr;
    if (metadata != null) {
      if (metadata.length != count) throw LibraException('ids and metadata size mismatch');
      metasPtr = malloc.allocate<Pointer<Utf8>>(count * sizeOf<Pointer<Utf8>>());
      for (int i = 0; i < count; i++) {
        metasPtr[i] = jsonEncode(metadata[i]).toNativeUtf8();
      }
    }

    final errPtr = lib.insertBatch(_handle, idsPtr, vecsPtr, count, _dim, metasPtr);

    // Free everything
    for (int i = 0; i < count; i++) {
      malloc.free(idsPtr[i]);
      if (metasPtr != nullptr) {
        malloc.free(metasPtr[i]);
      }
    }
    malloc.free(idsPtr);
    malloc.free(vecsPtr);
    if (metasPtr != nullptr) {
      malloc.free(metasPtr);
    }

    _checkError(errPtr, 'InsertBatch');
  }

  void deleteBatch(List<String> ids) {
    int count = ids.length;
    final idsPtr = malloc.allocate<Pointer<Utf8>>(count * sizeOf<Pointer<Utf8>>());
    for (int i = 0; i < count; i++) {
      idsPtr[i] = ids[i].toNativeUtf8();
    }

    final errPtr = lib.deleteBatch(_handle, idsPtr, count);

    for (int i = 0; i < count; i++) {
      malloc.free(idsPtr[i]);
    }
    malloc.free(idsPtr);

    _checkError(errPtr, 'DeleteBatch');
  }

  int count() {
    final c = lib.getCollectionCount(_handle);
    if (c < 0) throw LibraException('Failed to get collection count');
    return c;
  }

  void enableMemoryMapping(String path) {
    final pathPtr = path.toNativeUtf8();
    final errPtr = lib.enableMemoryMapping(_handle, pathPtr);
    malloc.free(pathPtr);
    _checkError(errPtr, 'EnableMemoryMapping');
  }
}

extension on List {
  int get size => length;
}
