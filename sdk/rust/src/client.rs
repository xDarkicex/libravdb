use crate::bindings;
use crate::core::{from_c_string, to_c_string};
use crate::filter::Filter;
use serde_json::Value;
use std::os::raw::c_char;

#[derive(Debug)]
pub struct LibraError(pub String);

pub struct Collection {
    handle: i32,
    dim: i32,
}

impl Collection {
    pub(crate) fn new(handle: i32, dim: i32) -> Self {
        Self { handle, dim }
    }

    fn check_error(&self, err_ptr: *mut c_char, op_name: &str) -> Result<(), LibraError> {
        let err_msg = from_c_string(err_ptr);
        if let Some(msg) = err_msg {
            if msg.starts_with("error: ") {
                return Err(LibraError(format!("{} failed: {}", op_name, &msg[7..])));
            }
            return Err(LibraError(format!("{} failed: {}", op_name, msg)));
        }
        Ok(())
    }

    fn parse_result(&self, res_ptr: *mut c_char, op_name: &str) -> Result<Value, LibraError> {
        let json_str = from_c_string(res_ptr);
        if let Some(msg) = json_str {
            if msg.starts_with("{\"error\"") {
                let v: Value = serde_json::from_str(&msg).unwrap_or(Value::Null);
                let err_text = v["error"].as_str().unwrap_or("Unknown error");
                return Err(LibraError(format!("{} failed: {}", op_name, err_text)));
            }
            if msg.is_empty() {
                return Ok(Value::Null);
            }
            let v: Value = serde_json::from_str(&msg)
                .map_err(|e| LibraError(format!("JSON Parse error: {}", e)))?;
            return Ok(v);
        }
        Ok(Value::Null)
    }

    pub fn insert(&self, id: &str, vector: &[f32], metadata: Option<Value>) -> Result<(), LibraError> {
        if vector.len() as i32 != self.dim {
            return Err(LibraError(format!("Vector dimension must be {}", self.dim)));
        }
        
        let c_id = to_c_string(id);
        let meta_str = metadata.map(|m| m.to_string()).unwrap_or_default();
        let c_meta = to_c_string(&meta_str);

        let err_ptr = unsafe {
            bindings::InsertVector(
                self.handle,
                c_id.as_ptr() as *mut c_char,
                vector.as_ptr() as *mut f32,
                self.dim,
                c_meta.as_ptr() as *mut c_char,
            )
        };
        self.check_error(err_ptr, "Insert")
    }

    pub fn update(&self, id: &str, vector: &[f32], metadata: Option<Value>) -> Result<(), LibraError> {
        let c_id = to_c_string(id);
        let meta_str = metadata.map(|m| m.to_string()).unwrap_or_default();
        let c_meta = to_c_string(&meta_str);

        let err_ptr = unsafe {
            bindings::UpdateVector(
                self.handle,
                c_id.as_ptr() as *mut c_char,
                vector.as_ptr() as *mut f32,
                self.dim,
                c_meta.as_ptr() as *mut c_char,
            )
        };
        self.check_error(err_ptr, "Update")
    }

    pub fn get(&self, id: &str) -> Result<Value, LibraError> {
        let c_id = to_c_string(id);
        let res_ptr = unsafe { bindings::GetVector(self.handle, c_id.as_ptr() as *mut c_char) };
        self.parse_result(res_ptr, "Get")
    }

    pub fn search(&self, vector: &[f32], k: i32, filter: Option<&Filter>) -> Result<Value, LibraError> {
        let filter_str = filter.map(|f| serde_json::to_string(f).unwrap_or_default()).unwrap_or_default();
        let c_filter = to_c_string(&filter_str);
        
        let res_ptr = unsafe {
            bindings::QueryVector(
                self.handle,
                vector.as_ptr() as *mut f32,
                self.dim,
                k,
                c_filter.as_ptr() as *mut c_char,
            )
        };
        self.parse_result(res_ptr, "Search")
    }

    pub fn scan(&self, offset: i32, limit: i32) -> Result<Value, LibraError> {
        let res_ptr = unsafe { bindings::ScanCollection(self.handle, offset, limit) };
        self.parse_result(res_ptr, "Scan")
    }

    pub fn insert_batch(&self, ids: &[String], vectors: &[Vec<f32>], metadata: Option<&[Value]>) -> Result<(), LibraError> {
        let count = ids.len();
        if vectors.len() != count {
            return Err(LibraError("ids and vectors must have same length".to_string()));
        }
        
        let mut flat_vectors = Vec::with_capacity(count * self.dim as usize);
        for vec in vectors {
            if vec.len() as i32 != self.dim {
                return Err(LibraError(format!("Vector dimension must be {}", self.dim)));
            }
            flat_vectors.extend_from_slice(vec);
        }

        // Convert Strings to CStrings
        let c_ids: Vec<std::ffi::CString> = ids.iter().map(|id| to_c_string(id)).collect();
        let mut c_id_ptrs: Vec<*mut c_char> = c_ids.iter().map(|c| c.as_ptr() as *mut c_char).collect();
        
        let mut c_meta_ptrs: Vec<*mut c_char> = Vec::new();
        let mut c_metas: Vec<std::ffi::CString> = Vec::new();
        
        let meta_ptr = if let Some(metas) = metadata {
            if metas.len() != count {
                return Err(LibraError("ids and metadata must have same length".to_string()));
            }
            c_metas = metas.iter().map(|m| to_c_string(&m.to_string())).collect();
            c_meta_ptrs = c_metas.iter().map(|c| c.as_ptr() as *mut c_char).collect();
            c_meta_ptrs.as_mut_ptr()
        } else {
            std::ptr::null_mut()
        };

        let err_ptr = unsafe {
            bindings::InsertBatch(
                self.handle,
                c_id_ptrs.as_mut_ptr(),
                flat_vectors.as_mut_ptr(),
                count as i32,
                self.dim,
                meta_ptr,
            )
        };
        self.check_error(err_ptr, "InsertBatch")
    }

    pub fn delete_batch(&self, ids: &[String]) -> Result<(), LibraError> {
        let c_ids: Vec<std::ffi::CString> = ids.iter().map(|id| to_c_string(id)).collect();
        let mut c_id_ptrs: Vec<*mut c_char> = c_ids.iter().map(|c| c.as_ptr() as *mut c_char).collect();
        
        let err_ptr = unsafe { bindings::DeleteBatch(self.handle, c_id_ptrs.as_mut_ptr(), ids.len() as i32) };
        self.check_error(err_ptr, "DeleteBatch")
    }

    pub fn count(&self) -> Result<i64, LibraError> {
        let c = unsafe { bindings::GetCollectionCount(self.handle) };
        if c < 0 {
            return Err(LibraError("Failed to get collection count".to_string()));
        }
        Ok(c)
    }

    pub fn update_if_version(&self, id: &str, vector: &[f32], expected_version: u64, metadata: Option<Value>) -> Result<(), LibraError> {
        let c_id = to_c_string(id);
        let meta_str = metadata.map(|m| m.to_string()).unwrap_or_default();
        let c_meta = to_c_string(&meta_str);

        let err_ptr = unsafe {
            bindings::UpdateVectorIfVersion(
                self.handle,
                c_id.as_ptr() as *mut c_char,
                vector.as_ptr() as *mut f32,
                self.dim,
                c_meta.as_ptr() as *mut c_char,
                expected_version,
            )
        };
        self.check_error(err_ptr, "UpdateIfVersion")
    }

    pub fn enable_memory_mapping(&self, path: &str) -> Result<(), LibraError> {
        let c_path = to_c_string(path);
        let err_ptr = unsafe { bindings::EnableMemoryMapping(self.handle, c_path.as_ptr() as *mut c_char) };
        self.check_error(err_ptr, "EnableMemoryMapping")
    }
}

pub struct LibraVDB {
    handle: i32,
}

impl LibraVDB {
    pub fn new(path: &str) -> Result<Self, LibraError> {
        let c_path = to_c_string(path);
        let handle = unsafe { bindings::OpenDB(c_path.as_ptr() as *mut c_char) };
        if handle < 0 {
            return Err(LibraError(format!("Failed to open database at {}", path)));
        }
        Ok(Self { handle })
    }

    pub fn ping(&self) -> Result<(), LibraError> {
        let err_ptr = unsafe { bindings::Ping(self.handle) };
        let err_msg = from_c_string(err_ptr);
        if err_msg.is_some() {
            return Err(LibraError("Ping failed".to_string()));
        }
        Ok(())
    }

    pub fn set_memory_limit(&self, limit: i64) -> Result<(), LibraError> {
        let err_ptr = unsafe { bindings::SetGlobalMemoryLimit(self.handle, limit) };
        let err_msg = from_c_string(err_ptr);
        if err_msg.is_some() {
            return Err(LibraError("Set memory limit failed".to_string()));
        }
        Ok(())
    }

    pub fn create_collection(&self, name: &str, dimension: i32) -> Result<Collection, LibraError> {
        let c_name = to_c_string(name);
        let col_handle = unsafe { bindings::CreateCollection(self.handle, c_name.as_ptr() as *mut c_char, dimension) };
        if col_handle < 0 {
            return Err(LibraError(format!("Failed to create collection {}", name)));
        }
        Ok(Collection::new(col_handle, dimension))
    }

    pub fn get_collection(&self, name: &str, dimension: i32) -> Result<Collection, LibraError> {
        let c_name = to_c_string(name);
        let col_handle = unsafe { bindings::GetCollection(self.handle, c_name.as_ptr() as *mut c_char) };
        if col_handle < 0 {
            return Err(LibraError(format!("Failed to get collection {}", name)));
        }
        Ok(Collection::new(col_handle, dimension))
    }
    
    pub fn list_collections(&self) -> Result<Vec<String>, LibraError> {
        let res_ptr = unsafe { bindings::ListCollections(self.handle) };
        let json_str = from_c_string(res_ptr);
        if let Some(msg) = json_str {
            let cols: Vec<String> = serde_json::from_str(&msg).unwrap_or_default();
            return Ok(cols);
        }
        Ok(Vec::new())
    }

    pub fn vacuum(&self) -> Result<(), LibraError> {
        let err_ptr = unsafe { bindings::Vacuum(self.handle) };
        let err_msg = from_c_string(err_ptr);
        if err_msg.is_some() {
            return Err(LibraError("Vacuum failed".to_string()));
        }
        Ok(())
    }

    pub fn drop_db(&self) -> Result<(), LibraError> {
        let err_ptr = unsafe { bindings::DropDatabase(self.handle) };
        let err_msg = from_c_string(err_ptr);
        if err_msg.is_some() {
            return Err(LibraError("Drop database failed".to_string()));
        }
        Ok(())
    }

    fn parse_query_result(&self, res_ptr: *mut c_char, op_name: &str) -> Result<Value, LibraError> {
        let json_str = from_c_string(res_ptr);
        if let Some(msg) = json_str {
            if msg.starts_with("{\"error\"") {
                let v: Value = serde_json::from_str(&msg).unwrap_or(Value::Null);
                let err_text = v["error"].as_str().unwrap_or("Unknown error");
                return Err(LibraError(format!("{} failed: {}", op_name, err_text)));
            }
            if msg.is_empty() {
                return Ok(Value::Null);
            }
            let v: Value = serde_json::from_str(&msg)
                .map_err(|e| LibraError(format!("JSON Parse error: {}", e)))?;
            return Ok(v);
        }
        Ok(Value::Null)
    }

    pub fn query(&self, sql: &str) -> Result<Value, LibraError> {
        let c_sql = to_c_string(sql);
        let res_ptr = unsafe { bindings::DatabaseQuery(self.handle, c_sql.as_ptr() as *mut c_char) };
        self.parse_query_result(res_ptr, "Query")
    }

    pub fn query_with_params(&self, sql: &str, params: Option<Value>) -> Result<Value, LibraError> {
        let c_sql = to_c_string(sql);
        let params_str = params.map(|p| p.to_string()).unwrap_or_default();
        let c_params = to_c_string(&params_str);
        let res_ptr = unsafe {
            bindings::DatabaseQueryWithParams(
                self.handle,
                c_sql.as_ptr() as *mut c_char,
                c_params.as_ptr() as *mut c_char,
            )
        };
        self.parse_query_result(res_ptr, "QueryWithParams")
    }
}

impl Drop for LibraVDB {
    fn drop(&mut self) {
        if self.handle >= 0 {
            unsafe { bindings::CloseDB(self.handle) };
        }
    }
}
