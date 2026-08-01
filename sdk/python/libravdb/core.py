import ctypes
import os
import platform

# Determine the library extension
system = platform.system()
if system == 'Linux':
    ext = 'so'
elif system == 'Darwin':
    ext = 'dylib'
elif system == 'Windows':
    ext = 'dll'
else:
    raise RuntimeError(f"Unsupported OS: {system}")

# Assuming the library is bundled in the ext/ directory.
base_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(base_dir, 'ext', f'libravdb.{ext}')

if not os.path.exists(lib_path):
    # Fallback for local development
    dev_dir = os.path.dirname(os.path.dirname(os.path.dirname(base_dir)))
    lib_path = os.path.join(dev_dir, 'cgo', f'libravdb.{ext}')

try:
    _lib = ctypes.CDLL(lib_path)
except OSError as e:
    raise RuntimeError(f"Failed to load {lib_path}. Did you build the CGO library? Error: {e}")

# Define types
c_char_p = ctypes.c_char_p
c_int = ctypes.c_int
c_float_p = ctypes.POINTER(ctypes.c_float)

# int OpenDB(char* path)
_lib.OpenDB.argtypes = [c_char_p]
_lib.OpenDB.restype = c_int

# int CloseDB(int dbID)
_lib.CloseDB.argtypes = [c_int]
_lib.CloseDB.restype = c_int

# int CreateCollection(int dbID, char* name, int dim)
_lib.CreateCollection.argtypes = [c_int, c_char_p, c_int]
_lib.CreateCollection.restype = c_int

# int GetCollection(int dbID, char* name)
_lib.GetCollection.argtypes = [c_int, c_char_p]
_lib.GetCollection.restype = c_int

# char* InsertVector(int colID, char* id, float* vector, int dim, char* metadataJSON)
_lib.InsertVector.argtypes = [c_int, c_char_p, c_float_p, c_int, c_char_p]
_lib.InsertVector.restype = ctypes.POINTER(ctypes.c_char) # Returns an allocated string, need to free it

# char* UpsertVector(int colID, char* id, float* vector, int dim, char* metadataJSON)
_lib.UpsertVector.argtypes = [c_int, c_char_p, c_float_p, c_int, c_char_p]
_lib.UpsertVector.restype = ctypes.POINTER(ctypes.c_char)

# char* DeleteVector(int colID, char* id)
_lib.DeleteVector.argtypes = [c_int, c_char_p]
_lib.DeleteVector.restype = ctypes.POINTER(ctypes.c_char)

# char* OptimizeCollection(int dbID, char* name)
_lib.OptimizeCollection.argtypes = [c_int, c_char_p]
_lib.OptimizeCollection.restype = ctypes.POINTER(ctypes.c_char)

# char* GetCollectionStats(int colID)
_lib.GetCollectionStats.argtypes = [c_int]
_lib.GetCollectionStats.restype = ctypes.POINTER(ctypes.c_char)

# char* QueryVector(int colID, float* vector, int dim, int limit, char* filterJSON)
_lib.QueryVector.argtypes = [c_int, c_float_p, c_int, c_int, c_char_p]
_lib.QueryVector.restype = ctypes.POINTER(ctypes.c_char)

# char* ScanCollection(int colID, int offset, int limit)
_lib.ScanCollection.argtypes = [c_int, c_int, c_int]
_lib.ScanCollection.restype = ctypes.POINTER(ctypes.c_char)

# char* UpdateVector(int colID, char* id, float* vector, int dim, char* metadataJSON)
_lib.UpdateVector.argtypes = [c_int, c_char_p, c_float_p, c_int, c_char_p]
_lib.UpdateVector.restype = ctypes.POINTER(ctypes.c_char)

# char* InsertBatch(int colID, char** ids, float* vecs, int count, int dim, char** metas)
_lib.InsertBatch.argtypes = [c_int, ctypes.POINTER(c_char_p), c_float_p, c_int, c_int, ctypes.POINTER(c_char_p)]
_lib.InsertBatch.restype = ctypes.POINTER(ctypes.c_char)

# char* DeleteBatch(int colID, char** ids, int count)
_lib.DeleteBatch.argtypes = [c_int, ctypes.POINTER(c_char_p), c_int]
_lib.DeleteBatch.restype = ctypes.POINTER(ctypes.c_char)

# char* ListCollections(int dbID)
_lib.ListCollections.argtypes = [c_int]
_lib.ListCollections.restype = ctypes.POINTER(ctypes.c_char)

# char* DeleteCollection(int dbID, char* name)
_lib.DeleteCollection.argtypes = [c_int, c_char_p]
_lib.DeleteCollection.restype = ctypes.POINTER(ctypes.c_char)

# char* Vacuum(int dbID)
_lib.Vacuum.argtypes = [c_int]
_lib.Vacuum.restype = ctypes.POINTER(ctypes.c_char)

# char* Backup(int dbID, char* dest)
_lib.Backup.argtypes = [c_int, c_char_p]
_lib.Backup.restype = ctypes.POINTER(ctypes.c_char)

# char* DropDatabase(int dbID)
_lib.DropDatabase.argtypes = [c_int]
_lib.DropDatabase.restype = ctypes.POINTER(ctypes.c_char)

# char* SetGlobalMemoryLimit(int dbID, long long limit)
_lib.SetGlobalMemoryLimit.argtypes = [c_int, ctypes.c_longlong]
_lib.SetGlobalMemoryLimit.restype = ctypes.POINTER(ctypes.c_char)

# char* GetGlobalMemoryUsage(int dbID)
_lib.GetGlobalMemoryUsage.argtypes = [c_int]
_lib.GetGlobalMemoryUsage.restype = ctypes.POINTER(ctypes.c_char)

# char* TriggerGlobalGC(int dbID)
_lib.TriggerGlobalGC.argtypes = [c_int]
_lib.TriggerGlobalGC.restype = ctypes.POINTER(ctypes.c_char)

# char* Ping(int dbID)
_lib.Ping.argtypes = [c_int]
_lib.Ping.restype = ctypes.POINTER(ctypes.c_char)

# char* GetDatabaseHealth(int dbID)
_lib.GetDatabaseHealth.argtypes = [c_int]
_lib.GetDatabaseHealth.restype = ctypes.POINTER(ctypes.c_char)

# char* GetDatabaseStats(int dbID)
_lib.GetDatabaseStats.argtypes = [c_int]
_lib.GetDatabaseStats.restype = ctypes.POINTER(ctypes.c_char)

# char* GetVector(int colID, char* id)
_lib.GetVector.argtypes = [c_int, c_char_p]
_lib.GetVector.restype = ctypes.POINTER(ctypes.c_char)

# long long GetCollectionCount(int colID)
_lib.GetCollectionCount.argtypes = [c_int]
_lib.GetCollectionCount.restype = ctypes.c_longlong

# char* UpdateVectorIfVersion(int colID, char* id, float* vector, int dim, char* metadataJSON, unsigned long long expectedVersion)
_lib.UpdateVectorIfVersion.argtypes = [c_int, c_char_p, c_float_p, c_int, c_char_p, ctypes.c_ulonglong]
_lib.UpdateVectorIfVersion.restype = ctypes.POINTER(ctypes.c_char)

# char* DeleteVectorIfVersion(int colID, char* id, unsigned long long expectedVersion)
_lib.DeleteVectorIfVersion.argtypes = [c_int, c_char_p, ctypes.c_ulonglong]
_lib.DeleteVectorIfVersion.restype = ctypes.POINTER(ctypes.c_char)

# char* SetCollectionMemoryLimit(int colID, long long bytes)
_lib.SetCollectionMemoryLimit.argtypes = [c_int, ctypes.c_longlong]
_lib.SetCollectionMemoryLimit.restype = ctypes.POINTER(ctypes.c_char)

# char* GetCollectionMemoryUsage(int colID)
_lib.GetCollectionMemoryUsage.argtypes = [c_int]
_lib.GetCollectionMemoryUsage.restype = ctypes.POINTER(ctypes.c_char)

# char* TriggerCollectionGC(int colID)
_lib.TriggerCollectionGC.argtypes = [c_int]
_lib.TriggerCollectionGC.restype = ctypes.POINTER(ctypes.c_char)

# char* EnableMemoryMapping(int colID, char* path)
_lib.EnableMemoryMapping.argtypes = [c_int, c_char_p]
_lib.EnableMemoryMapping.restype = ctypes.POINTER(ctypes.c_char)

# char* DisableMemoryMapping(int colID)
_lib.DisableMemoryMapping.argtypes = [c_int]
_lib.DisableMemoryMapping.restype = ctypes.POINTER(ctypes.c_char)

# char* SaveIndex(int colID, char* path)
_lib.SaveIndex.argtypes = [c_int, c_char_p]
_lib.SaveIndex.restype = ctypes.POINTER(ctypes.c_char)

# char* LoadIndex(int colID, char* path)
_lib.LoadIndex.argtypes = [c_int, c_char_p]
_lib.LoadIndex.restype = ctypes.POINTER(ctypes.c_char)

# void FreeString(char* str)
_lib.FreeString.argtypes = [ctypes.POINTER(ctypes.c_char)]
_lib.FreeString.restype = None

def _to_c_string(s: str) -> bytes:
    if s is None:
        return None
    return s.encode('utf-8')

def _from_c_string(ptr) -> str:
    if not ptr:
        return None
    
    # Cast pointer to string, then free the memory to avoid leaks
    c_str = ctypes.cast(ptr, c_char_p).value
    if c_str is not None:
        result = c_str.decode('utf-8')
        _lib.FreeString(ptr)
        return result
    return None
