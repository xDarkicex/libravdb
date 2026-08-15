# LibraVDB SDKs

Welcome to the LibraVDB SDK directory! LibraVDB is written in high-performance Go, but it is designed to be easily accessible from other languages via its CGO (C-Shared) bridge.

The SDKs in this directory provide idiomatic, language-specific wrappers over the C-Shared library, bringing the power of the native Go engine into your favorite ecosystems with zero loss in performance.

## Available SDKs

*   **[Python SDK](./python/README.md)**: A complete, fully-featured Python implementation providing 100% API parity with the native Go engine. Ideal for AI, ML, and Agentic workflows.
*   **[TypeScript SDK](./typescript/README.md)**: A Node.js wrapper using `koffi` for blazing-fast CGO FFI bindings, offering the same 100% API parity for the TypeScript and JavaScript ecosystem.
*   **[Ruby SDK](./ruby/README.md)**: A pure Ruby implementation using the `ffi` gem to securely and performantly bind to the CGO bridge.
*   **[Rust SDK](./rust/README.md)**: An idiomatic, safe Rust wrapper generated via `bindgen`, delivering zero-allocation FFI boundaries for performance-critical AI systems.
*   **[C++ SDK](./cpp/README.md)**: A modern C++17 wrapper using CMake and `nlohmann::json`, offering high-performance memory safety directly on top of the native header.
*   **[Java SDK](./java/README.md)**: A pure Java enterprise wrapper using JNA and Jackson for seamless FFI bindings without requiring JNI compilation.
*   **[Dart SDK](./dart/README.md)**: A high-performance Dart implementation leveraging `dart:ffi` and `malloc`, perfect for Flutter edge-AI or Dart backend services.
*   **[Kotlin SDK](./kotlin/README.md)**: A true Kotlin Native multiplatform implementation utilizing `cinterop` and `memScoped` memory management, compiling to native machine code without the JVM.
*   **[C# / .NET SDK](./csharp/README.md)**: An optimized P/Invoke implementation utilizing `DllImport` and `fixed` pointers to bypass managed-to-unmanaged memory copying.
*   **[Swift SDK](./swift/README.md)**: A native integration utilizing Swift Package Manager and direct C-interoperability for on-device Apple Silicon deployments.
*   **[Odin SDK](./odin/README.md)**: Zero-overhead native `foreign` bindings utilizing direct slice mapping `raw_data(vector)` for seamless memory mapping.
*   **[Perl SDK](./perl/README.md)**: A robust native FFI wrapper utilizing CPAN's `FFI::Platypus` and the Core `JSON::PP` module.
*   **[PHP SDK](./php/README.md)**: Utilizes PHP's native `FFI` extension (PHP 7.4+) enabling zero-copy batching of multi-dimensional PHP float arrays.
*   **[Lua SDK](./lua/README.md)**: A completely native Lua implementation powered by `LuaJIT`'s FFI engine, offering full object-orientation and pure-Lua JSON parsing for embedded workflows and game engines.
*   **[R SDK](./r/README.md)**: A high-performance wrapper tailored for data scientists using `Rcpp` to bind directly to the Go engine's C-Shared library with flawless `jsonlite` `data.frame` mapping.

## The CGO Bridge
All SDKs communicate with the native Go engine via the `libravdb` C-Shared Library. 
*   **[CGO Bridge Documentation](./cgo/README.md)**

If you want to build an SDK for a new language, read the CGO documentation above to understand the FFI (Foreign Function Interface) patterns used by LibraVDB.
