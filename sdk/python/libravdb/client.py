import json
from typing import List, Dict, Any, Optional

from .core import _to_c_string, _from_c_string, _lib, c_float_p
from .filters import Filter
import ctypes

class Collection:
    def __init__(self, handle: int, dim: int):
        self._handle = handle
        self._dim = dim

    def insert(self, id: str, vector: List[float], metadata: Dict[str, Any] = None):
        if len(vector) != self._dim:
            raise ValueError(f"Vector dimension must be {self._dim}, got {len(vector)}")

        # Convert python list to C float array
        float_array = (ctypes.c_float * self._dim)(*vector)
        
        meta_str = json.dumps(metadata) if metadata else ""

        err_ptr = _lib.InsertVector(
            self._handle,
            _to_c_string(id),
            float_array,
            self._dim,
            _to_c_string(meta_str)
        )
        
        err_msg = _from_c_string(err_ptr)
        if err_msg is not None:
            raise RuntimeError(f"Insert failed: {err_msg}")

    def upsert(self, id: str, vector: List[float], metadata: Dict[str, Any] = None):
        if len(vector) != self._dim:
            raise ValueError(f"Vector dimension must be {self._dim}, got {len(vector)}")

        float_array = (ctypes.c_float * self._dim)(*vector)
        meta_str = json.dumps(metadata) if metadata else ""

        err_ptr = _lib.UpsertVector(
            self._handle,
            _to_c_string(id),
            float_array,
            self._dim,
            _to_c_string(meta_str)
        )
        
        err_msg = _from_c_string(err_ptr)
        if err_msg is not None:
            raise RuntimeError(f"Upsert failed: {err_msg}")

    def delete(self, id: str):
        err_ptr = _lib.DeleteVector(self._handle, _to_c_string(id))
        err_msg = _from_c_string(err_ptr)
        if err_msg is not None:
            raise RuntimeError(f"Delete failed: {err_msg}")
            
    def stats(self) -> Dict[str, Any]:
        res_ptr = _lib.GetCollectionStats(self._handle)
        res_json = _from_c_string(res_ptr)
        if not res_json:
            return {}
        if res_json.startswith('{"error"'):
            err = json.loads(res_json)
            raise RuntimeError(f"Stats failed: {err.get('error')}")
        return json.loads(res_json)

    def search(self, vector: List[float], k: int = 10, filter: Optional[Filter] = None) -> List[Dict[str, Any]]:
        if len(vector) != self._dim:
            raise ValueError(f"Vector dimension must be {self._dim}, got {len(vector)}")

        float_array = (ctypes.c_float * self._dim)(*vector)
        
        filter_str = json.dumps(filter.to_json()) if filter else ""
        
        res_ptr = _lib.QueryVector(
            self._handle,
            float_array,
            self._dim,
            k,
            _to_c_string(filter_str)
        )
        
        res_json = _from_c_string(res_ptr)
        if not res_json:
            return []
            
        if res_json.startswith('{"error"'):
            err = json.loads(res_json)
            raise RuntimeError(f"Search failed: {err.get('error')}")
            
        return json.loads(res_json)

    def scan(self, offset: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        res_ptr = _lib.ScanCollection(self._handle, offset, limit)
        res_json = _from_c_string(res_ptr)
        if not res_json:
            return []
        if res_json.startswith('{"error"'):
            err = json.loads(res_json)
            raise RuntimeError(f"Scan failed: {err.get('error')}")
        return json.loads(res_json)

    def update(self, id: str, vector: List[float], metadata: Dict[str, Any] = None):
        if len(vector) != self._dim:
            raise ValueError(f"Vector dimension must be {self._dim}, got {len(vector)}")

        float_array = (ctypes.c_float * self._dim)(*vector)
        meta_str = json.dumps(metadata) if metadata else ""

        err_ptr = _lib.UpdateVector(
            self._handle,
            _to_c_string(id),
            float_array,
            self._dim,
            _to_c_string(meta_str)
        )
        
        err_msg = _from_c_string(err_ptr)
        if err_msg is not None:
            raise RuntimeError(f"Update failed: {err_msg}")

    def insert_batch(self, ids: List[str], vectors: List[List[float]], metadata: List[Dict[str, Any]] = None):
        count = len(ids)
        if len(vectors) != count:
            raise ValueError("ids and vectors must have the same length")
        if metadata and len(metadata) != count:
            raise ValueError("ids and metadata must have the same length")
        
        flat_vectors = []
        for v in vectors:
            if len(v) != self._dim:
                raise ValueError(f"All vectors must have dimension {self._dim}")
            flat_vectors.extend(v)
            
        float_array = (ctypes.c_float * len(flat_vectors))(*flat_vectors)
        
        str_array_type = ctypes.c_char_p * count
        ids_array = str_array_type(*[_to_c_string(i) for i in ids])
        
        metas_array = None
        if metadata:
            metas_array = str_array_type(*[_to_c_string(json.dumps(m)) if m else None for m in metadata])

        err_ptr = _lib.InsertBatch(
            self._handle,
            ids_array,
            float_array,
            count,
            self._dim,
            metas_array
        )
        err_msg = _from_c_string(err_ptr)
        if err_msg is not None:
            raise RuntimeError(f"InsertBatch failed: {err_msg}")

    def delete_batch(self, ids: List[str]):
        count = len(ids)
        str_array_type = ctypes.c_char_p * count
        ids_array = str_array_type(*[_to_c_string(i) for i in ids])

        err_ptr = _lib.DeleteBatch(self._handle, ids_array, count)
        err_msg = _from_c_string(err_ptr)
        if err_msg is not None:
            raise RuntimeError(f"DeleteBatch failed: {err_msg}")

    def get(self, id: str) -> Dict[str, Any]:
        res_ptr = _lib.GetVector(self._handle, _to_c_string(id))
        res_json = _from_c_string(res_ptr)
        if not res_json:
            raise RuntimeError(f"Failed to get vector {id}")
        if res_json.startswith('{"error"'):
            err = json.loads(res_json)
            raise RuntimeError(f"Get failed: {err.get('error')}")
        return json.loads(res_json)

    def count(self) -> int:
        c = _lib.GetCollectionCount(self._handle)
        if c < 0:
            raise RuntimeError("Failed to get collection count")
        return c

    def update_if_version(self, id: str, vector: List[float], expected_version: int, metadata: Dict[str, Any] = None):
        if len(vector) != self._dim:
            raise ValueError(f"Vector dimension must be {self._dim}, got {len(vector)}")

        float_array = (ctypes.c_float * self._dim)(*vector)
        meta_str = json.dumps(metadata) if metadata else ""

        err_ptr = _lib.UpdateVectorIfVersion(
            self._handle,
            _to_c_string(id),
            float_array,
            self._dim,
            _to_c_string(meta_str),
            expected_version
        )
        
        err_msg = _from_c_string(err_ptr)
        if err_msg is not None:
            raise RuntimeError(f"UpdateIfVersion failed: {err_msg}")

    def delete_if_version(self, id: str, expected_version: int):
        err_ptr = _lib.DeleteVectorIfVersion(self._handle, _to_c_string(id), expected_version)
        err_msg = _from_c_string(err_ptr)
        if err_msg is not None:
            raise RuntimeError(f"DeleteIfVersion failed: {err_msg}")

    def set_memory_limit(self, limit: int):
        err_ptr = _lib.SetCollectionMemoryLimit(self._handle, limit)
        err_msg = _from_c_string(err_ptr)
        if err_msg is not None:
            raise RuntimeError(f"SetMemoryLimit failed: {err_msg}")

    def memory_usage(self) -> Dict[str, Any]:
        res_ptr = _lib.GetCollectionMemoryUsage(self._handle)
        res_json = _from_c_string(res_ptr)
        if not res_json:
            return {}
        if res_json.startswith('{"error"'):
            err = json.loads(res_json)
            raise RuntimeError(f"Memory usage failed: {err.get('error')}")
        return json.loads(res_json)

    def trigger_gc(self):
        err_ptr = _lib.TriggerCollectionGC(self._handle)
        err_msg = _from_c_string(err_ptr)
        if err_msg is not None:
            raise RuntimeError(f"TriggerGC failed: {err_msg}")

    def enable_memory_mapping(self, path: str):
        err_ptr = _lib.EnableMemoryMapping(self._handle, _to_c_string(path))
        err_msg = _from_c_string(err_ptr)
        if err_msg is not None:
            raise RuntimeError(f"EnableMemoryMapping failed: {err_msg}")

    def disable_memory_mapping(self):
        err_ptr = _lib.DisableMemoryMapping(self._handle)
        err_msg = _from_c_string(err_ptr)
        if err_msg is not None:
            raise RuntimeError(f"DisableMemoryMapping failed: {err_msg}")

    def save_index(self, path: str):
        err_ptr = _lib.SaveIndex(self._handle, _to_c_string(path))
        err_msg = _from_c_string(err_ptr)
        if err_msg is not None:
            raise RuntimeError(f"SaveIndex failed: {err_msg}")

    def load_index(self, path: str):
        err_ptr = _lib.LoadIndex(self._handle, _to_c_string(path))
        err_msg = _from_c_string(err_ptr)
        if err_msg is not None:
            raise RuntimeError(f"LoadIndex failed: {err_msg}")


class LibraVDB:
    def __init__(self, path: str):
        self._handle = _lib.OpenDB(_to_c_string(path))
        if self._handle < 0:
            raise RuntimeError(f"Failed to open database at {path}")
            

    def close(self):
        if self._handle >= 0:
            _lib.CloseDB(self._handle)
            self._handle = -1

    def create_collection(self, name: str, dimension: int) -> Collection:
        col_handle = _lib.CreateCollection(self._handle, _to_c_string(name), dimension)
        if col_handle < 0:
            raise RuntimeError(f"Failed to create collection {name}")
        return Collection(col_handle, dimension)
        
    def get_collection(self, name: str, dimension: int) -> Collection:
        col_handle = _lib.GetCollection(self._handle, _to_c_string(name))
        if col_handle < 0:
            raise RuntimeError(f"Failed to get collection {name}")
        return Collection(col_handle, dimension)

    def optimize_collection(self, name: str):
        err_ptr = _lib.OptimizeCollection(self._handle, _to_c_string(name))
        err_msg = _from_c_string(err_ptr)
        if err_msg is not None:
            raise RuntimeError(f"Optimize failed: {err_msg}")

    def list_collections(self) -> List[str]:
        res_ptr = _lib.ListCollections(self._handle)
        res_json = _from_c_string(res_ptr)
        if not res_json:
            return []
        if res_json.startswith('{"error"'):
            err = json.loads(res_json)
            raise RuntimeError(f"List collections failed: {err.get('error')}")
        return json.loads(res_json)

    def delete_collection(self, name: str):
        err_ptr = _lib.DeleteCollection(self._handle, _to_c_string(name))
        err_msg = _from_c_string(err_ptr)
        if err_msg is not None:
            raise RuntimeError(f"Delete collection failed: {err_msg}")

    def vacuum(self):
        err_ptr = _lib.Vacuum(self._handle)
        err_msg = _from_c_string(err_ptr)
        if err_msg is not None:
            raise RuntimeError(f"Vacuum failed: {err_msg}")

    def backup(self, dest: str):
        err_ptr = _lib.Backup(self._handle, _to_c_string(dest))
        err_msg = _from_c_string(err_ptr)
        if err_msg is not None:
            raise RuntimeError(f"Backup failed: {err_msg}")

    def drop(self):
        err_ptr = _lib.DropDatabase(self._handle)
        err_msg = _from_c_string(err_ptr)
        if err_msg is not None:
            raise RuntimeError(f"Drop database failed: {err_msg}")

    def set_memory_limit(self, limit: int):
        err_ptr = _lib.SetGlobalMemoryLimit(self._handle, limit)
        err_msg = _from_c_string(err_ptr)
        if err_msg is not None:
            raise RuntimeError(f"Set memory limit failed: {err_msg}")

    def memory_usage(self) -> Dict[str, Any]:
        res_ptr = _lib.GetGlobalMemoryUsage(self._handle)
        res_json = _from_c_string(res_ptr)
        if not res_json:
            return {}
        if res_json.startswith('{"error"'):
            err = json.loads(res_json)
            raise RuntimeError(f"Memory usage failed: {err.get('error')}")
        return json.loads(res_json)

    def trigger_gc(self):
        err_ptr = _lib.TriggerGlobalGC(self._handle)
        err_msg = _from_c_string(err_ptr)
        if err_msg is not None:
            raise RuntimeError(f"Trigger GC failed: {err_msg}")

    def ping(self):
        err_ptr = _lib.Ping(self._handle)
        err_msg = _from_c_string(err_ptr)
        if err_msg is not None:
            raise RuntimeError(f"Ping failed: {err_msg}")

    def health(self) -> Dict[str, Any]:
        res_ptr = _lib.GetDatabaseHealth(self._handle)
        res_json = _from_c_string(res_ptr)
        if not res_json:
            return {}
        if res_json.startswith('{"error"'):
            err = json.loads(res_json)
            raise RuntimeError(f"Health failed: {err.get('error')}")
        return json.loads(res_json)

    def stats(self) -> Dict[str, Any]:
        res_ptr = _lib.GetDatabaseStats(self._handle)
        res_json = _from_c_string(res_ptr)
        if not res_json:
            return {}
        if res_json.startswith('{"error"'):
            err = json.loads(res_json)
            raise RuntimeError(f"Database Stats failed: {err.get('error')}")
        return json.loads(res_json)

    def query(self, sql: str) -> Dict[str, Any]:
        res_ptr = _lib.DatabaseQuery(self._handle, _to_c_string(sql))
        res_json = _from_c_string(res_ptr)
        if not res_json:
            return {}
        if res_json.startswith('{"error"'):
            err = json.loads(res_json)
            raise RuntimeError(f"Query failed: {err.get('error')}")
        return json.loads(res_json)

    def query_with_params(self, sql: str, params: Dict[str, Any]) -> Dict[str, Any]:
        params_str = json.dumps(params) if params else ""
        res_ptr = _lib.DatabaseQueryWithParams(
            self._handle,
            _to_c_string(sql),
            _to_c_string(params_str)
        )
        res_json = _from_c_string(res_ptr)
        if not res_json:
            return {}
        if res_json.startswith('{"error"'):
            err = json.loads(res_json)
            raise RuntimeError(f"QueryWithParams failed: {err.get('error')}")
        return json.loads(res_json)

    def latest_commit_lsn(self) -> int:
        """Return the exact latest durable commit LSN for this database."""
        res_ptr = _lib.DatabaseLatestCommitLSN(self._handle)
        res_json = _from_c_string(res_ptr)
        if not res_json:
            raise RuntimeError("LatestCommitLSN returned no response")
        response = json.loads(res_json)
        if "error" in response:
            raise RuntimeError(f"LatestCommitLSN failed: {response['error']}")
        return int(response["lsn"])

    def __del__(self):
        self.close()
