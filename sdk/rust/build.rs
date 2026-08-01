use std::env;
use std::path::PathBuf;

fn main() {
    let cgo_dir = PathBuf::from("../cgo")
        .canonicalize()
        .expect("Failed to find cgo directory");

    let header_path = cgo_dir.join("libravdb.h");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    // Create a symlink named liblibravdb.dylib/so to satisfy the rustc linker which expects the 'lib' prefix
    let lib_name = if cfg!(target_os = "macos") {
        "libravdb.dylib"
    } else if cfg!(target_os = "windows") {
        "libravdb.dll"
    } else {
        "libravdb.so"
    };
    
    let prefixed_name = format!("lib{}", lib_name);
    let src_lib = cgo_dir.join(lib_name);
    let dst_lib = out_path.join(prefixed_name);
    
    if dst_lib.exists() {
        std::fs::remove_file(&dst_lib).ok();
    }
    std::os::unix::fs::symlink(&src_lib, &dst_lib).ok();

    println!("cargo:rustc-link-search=native={}", out_path.display());
    println!("cargo:rustc-link-lib=dylib=libravdb");
    
    // Pass rpath so the binary can find the dylib at runtime without DYLD_LIBRARY_PATH
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cgo_dir.display());

    // Re-run if the header changes
    println!("cargo:rerun-if-changed={}", header_path.display());

    let bindings = bindgen::Builder::default()
        .header(header_path.to_str().unwrap())
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
