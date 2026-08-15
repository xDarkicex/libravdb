import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';

final DynamicLibrary _lib = Platform.isMacOS
    ? DynamicLibrary.open('../cgo/libravdb.dylib')
    : Platform.isLinux
        ? DynamicLibrary.open('../cgo/libravdb.so')
        : DynamicLibrary.process();

// OpenDB
typedef OpenDBNative = Int32 Function(Pointer<Utf8> path);
typedef OpenDBDart = int Function(Pointer<Utf8> path);
final openDB = _lib.lookupFunction<OpenDBNative, OpenDBDart>('OpenDB');

// CloseDB
typedef CloseDBNative = Int32 Function(Int32 dbID);
typedef CloseDBDart = int Function(int dbID);
final closeDB = _lib.lookupFunction<CloseDBNative, CloseDBDart>('CloseDB');

// Ping
typedef PingNative = Pointer<Utf8> Function(Int32 dbID);
typedef PingDart = Pointer<Utf8> Function(int dbID);
final ping = _lib.lookupFunction<PingNative, PingDart>('Ping');

// Vacuum
typedef VacuumNative = Pointer<Utf8> Function(Int32 dbID);
typedef VacuumDart = Pointer<Utf8> Function(int dbID);
final vacuum = _lib.lookupFunction<VacuumNative, VacuumDart>('Vacuum');

// DropDatabase
typedef DropDatabaseNative = Pointer<Utf8> Function(Int32 dbID);
typedef DropDatabaseDart = Pointer<Utf8> Function(int dbID);
final dropDatabase = _lib.lookupFunction<DropDatabaseNative, DropDatabaseDart>('DropDatabase');

// SetGlobalMemoryLimit
typedef SetMemoryLimitNative = Pointer<Utf8> Function(Int32 dbID, Int64 limit);
typedef SetMemoryLimitDart = Pointer<Utf8> Function(int dbID, int limit);
final setGlobalMemoryLimit = _lib.lookupFunction<SetMemoryLimitNative, SetMemoryLimitDart>('SetGlobalMemoryLimit');

// ListCollections
typedef ListCollectionsNative = Pointer<Utf8> Function(Int32 dbID);
typedef ListCollectionsDart = Pointer<Utf8> Function(int dbID);
final listCollections = _lib.lookupFunction<ListCollectionsNative, ListCollectionsDart>('ListCollections');

// CreateCollection
typedef CreateCollectionNative = Int32 Function(Int32 dbID, Pointer<Utf8> name, Int32 dim);
typedef CreateCollectionDart = int Function(int dbID, Pointer<Utf8> name, int dim);
final createCollection = _lib.lookupFunction<CreateCollectionNative, CreateCollectionDart>('CreateCollection');

// GetCollection
typedef GetCollectionNative = Int32 Function(Int32 dbID, Pointer<Utf8> name);
typedef GetCollectionDart = int Function(int dbID, Pointer<Utf8> name);
final getCollection = _lib.lookupFunction<GetCollectionNative, GetCollectionDart>('GetCollection');

// DatabaseQuery
typedef DatabaseQueryNative = Pointer<Utf8> Function(Int32 dbID, Pointer<Utf8> sql);
typedef DatabaseQueryDart = Pointer<Utf8> Function(int dbID, Pointer<Utf8> sql);
final databaseQuery = _lib.lookupFunction<DatabaseQueryNative, DatabaseQueryDart>('DatabaseQuery');

// DatabaseQueryWithParams
typedef DatabaseQueryWithParamsNative = Pointer<Utf8> Function(Int32 dbID, Pointer<Utf8> sql, Pointer<Utf8> params);
typedef DatabaseQueryWithParamsDart = Pointer<Utf8> Function(int dbID, Pointer<Utf8> sql, Pointer<Utf8> params);
final databaseQueryWithParams = _lib.lookupFunction<DatabaseQueryWithParamsNative, DatabaseQueryWithParamsDart>('DatabaseQueryWithParams');

// InsertVector
typedef InsertVectorNative = Pointer<Utf8> Function(Int32 colID, Pointer<Utf8> id, Pointer<Float> vec, Int32 dim, Pointer<Utf8> metadataJSON);
typedef InsertVectorDart = Pointer<Utf8> Function(int colID, Pointer<Utf8> id, Pointer<Float> vec, int dim, Pointer<Utf8> metadataJSON);
final insertVector = _lib.lookupFunction<InsertVectorNative, InsertVectorDart>('InsertVector');

// UpdateVector
typedef UpdateVectorNative = Pointer<Utf8> Function(Int32 colID, Pointer<Utf8> id, Pointer<Float> vec, Int32 dim, Pointer<Utf8> metadataJSON);
typedef UpdateVectorDart = Pointer<Utf8> Function(int colID, Pointer<Utf8> id, Pointer<Float> vec, int dim, Pointer<Utf8> metadataJSON);
final updateVector = _lib.lookupFunction<UpdateVectorNative, UpdateVectorDart>('UpdateVector');

// UpdateVectorIfVersion
typedef UpdateVectorIfVersionNative = Pointer<Utf8> Function(Int32 colID, Pointer<Utf8> id, Pointer<Float> vec, Int32 dim, Pointer<Utf8> metadataJSON, Int64 expectedVersion);
typedef UpdateVectorIfVersionDart = Pointer<Utf8> Function(int colID, Pointer<Utf8> id, Pointer<Float> vec, int dim, Pointer<Utf8> metadataJSON, int expectedVersion);
final updateVectorIfVersion = _lib.lookupFunction<UpdateVectorIfVersionNative, UpdateVectorIfVersionDart>('UpdateVectorIfVersion');

// GetVector
typedef GetVectorNative = Pointer<Utf8> Function(Int32 colID, Pointer<Utf8> id);
typedef GetVectorDart = Pointer<Utf8> Function(int colID, Pointer<Utf8> id);
final getVector = _lib.lookupFunction<GetVectorNative, GetVectorDart>('GetVector');

// QueryVector
typedef QueryVectorNative = Pointer<Utf8> Function(Int32 colID, Pointer<Float> vec, Int32 dim, Int32 limit, Pointer<Utf8> filterJSON);
typedef QueryVectorDart = Pointer<Utf8> Function(int colID, Pointer<Float> vec, int dim, int limit, Pointer<Utf8> filterJSON);
final queryVector = _lib.lookupFunction<QueryVectorNative, QueryVectorDart>('QueryVector');

// ScanCollection
typedef ScanCollectionNative = Pointer<Utf8> Function(Int32 colID, Int32 offset, Int32 limit);
typedef ScanCollectionDart = Pointer<Utf8> Function(int colID, int offset, int limit);
final scanCollection = _lib.lookupFunction<ScanCollectionNative, ScanCollectionDart>('ScanCollection');

// InsertBatch
typedef InsertBatchNative = Pointer<Utf8> Function(Int32 colID, Pointer<Pointer<Utf8>> ids, Pointer<Float> vectors, Int32 count, Int32 dim, Pointer<Pointer<Utf8>> metas);
typedef InsertBatchDart = Pointer<Utf8> Function(int colID, Pointer<Pointer<Utf8>> ids, Pointer<Float> vectors, int count, int dim, Pointer<Pointer<Utf8>> metas);
final insertBatch = _lib.lookupFunction<InsertBatchNative, InsertBatchDart>('InsertBatch');

// DeleteBatch
typedef DeleteBatchNative = Pointer<Utf8> Function(Int32 colID, Pointer<Pointer<Utf8>> ids, Int32 count);
typedef DeleteBatchDart = Pointer<Utf8> Function(int colID, Pointer<Pointer<Utf8>> ids, int count);
final deleteBatch = _lib.lookupFunction<DeleteBatchNative, DeleteBatchDart>('DeleteBatch');

// GetCollectionCount
typedef GetCollectionCountNative = Int64 Function(Int32 colID);
typedef GetCollectionCountDart = int Function(int colID);
final getCollectionCount = _lib.lookupFunction<GetCollectionCountNative, GetCollectionCountDart>('GetCollectionCount');

// EnableMemoryMapping
typedef EnableMemoryMappingNative = Pointer<Utf8> Function(Int32 colID, Pointer<Utf8> path);
typedef EnableMemoryMappingDart = Pointer<Utf8> Function(int colID, Pointer<Utf8> path);
final enableMemoryMapping = _lib.lookupFunction<EnableMemoryMappingNative, EnableMemoryMappingDart>('EnableMemoryMapping');

// FreeString
typedef FreeStringNative = Void Function(Pointer<Utf8> ptr);
typedef FreeStringDart = void Function(Pointer<Utf8> ptr);
final freeString = _lib.lookupFunction<FreeStringNative, FreeStringDart>('FreeString');
