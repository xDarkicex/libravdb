import 'dart:convert';
import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'bindings.dart' as lib;
import 'collection.dart';

class LibraException implements Exception {
  final String message;
  LibraException(this.message);
  @override
  String toString() => 'LibraException: $message';
}

class LibraVDB {
  int _handle = -1;

  LibraVDB(String path) {
    final pathPtr = path.toNativeUtf8();
    _handle = lib.openDB(pathPtr);
    malloc.free(pathPtr);

    if (_handle < 0) {
      throw LibraException('Failed to open database at $path');
    }
  }

  void _checkError(Pointer<Utf8> errPtr, String opName) {
    if (errPtr != nullptr) {
      final msg = errPtr.toDartString();
      lib.freeString(errPtr);
      throw LibraException('$opName failed: $msg');
    }
  }

  String _parseQueryResult(Pointer<Utf8> resPtr, String opName) {
    if (resPtr == nullptr) {
      throw LibraException('$opName failed: null pointer returned');
    }
    final result = resPtr.toDartString();
    lib.freeString(resPtr);

    if (result.startsWith('{"error"')) {
      throw LibraException('$opName failed: $result');
    }
    return result;
  }

  String query(String sql) {
    final sqlPtr = sql.toNativeUtf8();
    final resPtr = lib.databaseQuery(_handle, sqlPtr);
    malloc.free(sqlPtr);
    return _parseQueryResult(resPtr, 'Query');
  }

  String queryWithParams(String sql, [String params = ""]) {
    final sqlPtr = sql.toNativeUtf8();
    final paramsPtr = params.toNativeUtf8();
    final resPtr = lib.databaseQueryWithParams(_handle, sqlPtr, paramsPtr);
    malloc.free(sqlPtr);
    malloc.free(paramsPtr);
    return _parseQueryResult(resPtr, 'QueryWithParams');
  }

  void ping() {
    final errPtr = lib.ping(_handle);
    _checkError(errPtr, 'Ping');
  }

  void setMemoryLimit(int limit) {
    final errPtr = lib.setGlobalMemoryLimit(_handle, limit);
    _checkError(errPtr, 'SetMemoryLimit');
  }

  void vacuum() {
    final errPtr = lib.vacuum(_handle);
    _checkError(errPtr, 'Vacuum');
  }

  void dropDatabase() {
    final errPtr = lib.dropDatabase(_handle);
    _checkError(errPtr, 'DropDatabase');
  }

  List<String> listCollections() {
    final resPtr = lib.listCollections(_handle);
    if (resPtr == nullptr) return [];

    final msg = resPtr.toDartString();
    lib.freeString(resPtr);

    if (msg.isEmpty) return [];

    try {
      final node = jsonDecode(msg);
      if (node is List) {
        return node.map((e) => e.toString()).toList();
      }
      return [];
    } catch (e) {
      return [];
    }
  }

  Collection createCollection(String name, int dimension) {
    final namePtr = name.toNativeUtf8();
    final colHandle = lib.createCollection(_handle, namePtr, dimension);
    malloc.free(namePtr);

    if (colHandle < 0) {
      throw LibraException('Failed to create collection $name');
    }
    return Collection(colHandle, dimension);
  }

  Collection getCollection(String name, int dimension) {
    final namePtr = name.toNativeUtf8();
    final colHandle = lib.getCollection(_handle, namePtr);
    malloc.free(namePtr);

    if (colHandle < 0) {
      throw LibraException('Failed to get collection $name');
    }
    return Collection(colHandle, dimension);
  }

  void close() {
    if (_handle >= 0) {
      lib.closeDB(_handle);
      _handle = -1;
    }
  }
}
