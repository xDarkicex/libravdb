use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use crate::bindings;

pub fn from_c_string(ptr: *mut c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    
    // Read the string
    let c_str = unsafe { CStr::from_ptr(ptr) };
    let string_result = c_str.to_string_lossy().into_owned();
    
    // Free the pointer allocated by CGO
    unsafe {
        bindings::FreeString(ptr);
    }
    
    Some(string_result)
}

pub fn to_c_string(s: &str) -> CString {
    CString::new(s).expect("CString::new failed")
}
