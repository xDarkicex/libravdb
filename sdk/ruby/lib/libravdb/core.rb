require 'ffi'

module LibraVDB
  module Core
    extend FFI::Library
    
    # Try to load the library based on platform
    lib_ext = case RbConfig::CONFIG['host_os']
              when /mswin|msys|mingw|cygwin|bccwin|wince|emc/
                '.dll'
              when /darwin|mac os/
                '.dylib'
              else
                '.so'
              end
              
    ffi_lib [
      File.expand_path("../../ext/libravdb#{lib_ext}", __dir__),
      File.expand_path("../../../../cgo/libravdb#{lib_ext}", __dir__) # Fallback for local dev
    ]
    # Exported Functions
    attach_function :OpenDB, [:string], :int
    attach_function :CloseDB, [:int], :void
    attach_function :CreateCollection, [:int, :string, :int], :int
    attach_function :GetCollection, [:int, :string], :int
    attach_function :DatabaseQuery, [:int, :string], :pointer
    attach_function :DatabaseQueryWithParams, [:int, :string, :string], :pointer
    attach_function :DatabaseLatestCommitLSN, [:int], :pointer
    
    # Vector CRUD
    attach_function :InsertVector, [:int, :string, :pointer, :int, :string], :pointer
    attach_function :UpsertVector, [:int, :string, :pointer, :int, :string], :pointer
    attach_function :UpdateVector, [:int, :string, :pointer, :int, :string], :pointer
    attach_function :DeleteVector, [:int, :string], :pointer
    
    # Query & Search
    attach_function :QueryVector, [:int, :pointer, :int, :int, :string], :pointer
    attach_function :ScanCollection, [:int, :int, :int], :pointer
    
    # Batch
    attach_function :InsertBatch, [:int, :pointer, :pointer, :int, :int, :pointer], :pointer
    attach_function :DeleteBatch, [:int, :pointer, :int], :pointer
    
    # Lifecycle
    attach_function :GetCollectionStats, [:int], :pointer
    attach_function :OptimizeCollection, [:int, :string], :pointer
    attach_function :ListCollections, [:int], :pointer
    attach_function :DeleteCollection, [:int, :string], :pointer
    attach_function :Vacuum, [:int], :pointer
    attach_function :Backup, [:int, :string], :pointer
    attach_function :DropDatabase, [:int], :pointer
    
    # Global Memory
    attach_function :SetGlobalMemoryLimit, [:int, :long_long], :pointer
    attach_function :GetGlobalMemoryUsage, [:int], :pointer
    attach_function :TriggerGlobalGC, [:int], :pointer
    
    # Global Health
    attach_function :Ping, [:int], :pointer
    attach_function :GetDatabaseHealth, [:int], :pointer
    attach_function :GetDatabaseStats, [:int], :pointer
    
    # Phase 4 Features
    attach_function :GetVector, [:int, :string], :pointer
    attach_function :GetCollectionCount, [:int], :long_long
    attach_function :UpdateVectorIfVersion, [:int, :string, :pointer, :int, :string, :uint64], :pointer
    attach_function :DeleteVectorIfVersion, [:int, :string, :uint64], :pointer
    attach_function :SetCollectionMemoryLimit, [:int, :long_long], :pointer
    attach_function :GetCollectionMemoryUsage, [:int], :pointer
    attach_function :TriggerCollectionGC, [:int], :pointer
    attach_function :EnableMemoryMapping, [:int, :string], :pointer
    attach_function :DisableMemoryMapping, [:int], :pointer
    attach_function :SaveIndex, [:int, :string], :pointer
    attach_function :LoadIndex, [:int, :string], :pointer

    # Memory Free
    attach_function :FreeString, [:pointer], :void
    
    def self.from_c_string(ptr)
      return nil if ptr.null?
      str = ptr.read_string
      FreeString(ptr)
      str
    end
  end
end
