import koffi from 'koffi';
import * as os from 'os';
import * as path from 'path';

import * as fs from 'fs';

let ext = '.so';
if (os.platform() === 'darwin') {
    ext = '.dylib';
} else if (os.platform() === 'win32') {
    ext = '.dll';
}

let libPath = path.resolve(__dirname, '../ext/libravdb' + ext);
if (!fs.existsSync(libPath)) {
    // Fallback for local development
    libPath = path.resolve(__dirname, '../../cgo/libravdb' + ext);
}

const lib = koffi.load(libPath);

const CString = 'void *';
const CFloatArray = koffi.pointer('float');
const CStringArray = koffi.pointer('str');

export const _lib = {
    OpenDB: lib.func('OpenDB', 'int', ['str']),
    CloseDB: lib.func('CloseDB', 'void', ['int']),
    CreateCollection: lib.func('CreateCollection', 'int', ['int', 'str', 'int']),
    GetCollection: lib.func('GetCollection', 'int', ['int', 'str']),
    InsertVector: lib.func('InsertVector', CString, ['int', 'str', CFloatArray, 'int', 'str']),
    UpsertVector: lib.func('UpsertVector', CString, ['int', 'str', CFloatArray, 'int', 'str']),
    UpdateVector: lib.func('UpdateVector', CString, ['int', 'str', CFloatArray, 'int', 'str']),
    DeleteVector: lib.func('DeleteVector', CString, ['int', 'str']),
    QueryVector: lib.func('QueryVector', CString, ['int', CFloatArray, 'int', 'int', 'str']),
    GetCollectionStats: lib.func('GetCollectionStats', CString, ['int']),
    OptimizeCollection: lib.func('OptimizeCollection', CString, ['int', 'str']),
    ListCollections: lib.func('ListCollections', CString, ['int']),
    DeleteCollection: lib.func('DeleteCollection', CString, ['int', 'str']),
    Vacuum: lib.func('Vacuum', CString, ['int']),
    Backup: lib.func('Backup', CString, ['int', 'str']),
    DropDatabase: lib.func('DropDatabase', CString, ['int']),
    SetGlobalMemoryLimit: lib.func('SetGlobalMemoryLimit', CString, ['int', 'longlong']),
    GetGlobalMemoryUsage: lib.func('GetGlobalMemoryUsage', CString, ['int']),
    TriggerGlobalGC: lib.func('TriggerGlobalGC', CString, ['int']),
    Ping: lib.func('Ping', CString, ['int']),
    GetDatabaseHealth: lib.func('GetDatabaseHealth', CString, ['int']),
    GetDatabaseStats: lib.func('GetDatabaseStats', CString, ['int']),
    InsertBatch: lib.func('InsertBatch', CString, ['int', CStringArray, CFloatArray, 'int', 'int', CStringArray]),
    DeleteBatch: lib.func('DeleteBatch', CString, ['int', CStringArray, 'int']),
    ScanCollection: lib.func('ScanCollection', CString, ['int', 'int', 'int']),
    
    GetVector: lib.func('GetVector', CString, ['int', 'str']),
    GetCollectionCount: lib.func('GetCollectionCount', 'longlong', ['int']),
    UpdateVectorIfVersion: lib.func('UpdateVectorIfVersion', CString, ['int', 'str', CFloatArray, 'int', 'str', 'uint64']),
    DeleteVectorIfVersion: lib.func('DeleteVectorIfVersion', CString, ['int', 'str', 'uint64']),
    SetCollectionMemoryLimit: lib.func('SetCollectionMemoryLimit', CString, ['int', 'longlong']),
    GetCollectionMemoryUsage: lib.func('GetCollectionMemoryUsage', CString, ['int']),
    TriggerCollectionGC: lib.func('TriggerCollectionGC', CString, ['int']),
    EnableMemoryMapping: lib.func('EnableMemoryMapping', CString, ['int', 'str']),
    DisableMemoryMapping: lib.func('DisableMemoryMapping', CString, ['int']),
    SaveIndex: lib.func('SaveIndex', CString, ['int', 'str']),
    LoadIndex: lib.func('LoadIndex', CString, ['int', 'str']),
    
    FreeString: lib.func('FreeString', 'void', [CString]),
};

export function _from_c_string(ptr: any): string | null {
    if (!ptr || koffi.address(ptr) === 0n) return null;
    const str = koffi.decode(ptr, 'char', -1);
    _lib.FreeString(ptr);
    return str as string;
}
