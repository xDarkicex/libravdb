require 'json'
require_relative 'core'
require_relative 'filter'

module LibraVDB
  class Collection
    def initialize(handle, dim)
      @handle = handle
      @dim = dim
    end

    def insert(id, vector, metadata = {})
      raise "Vector dimension must be #{@dim}" if vector.size != @dim
      float_array = FFI::MemoryPointer.new(:float, @dim)
      float_array.write_array_of_float(vector)
      
      meta_str = metadata.empty? ? "" : metadata.to_json
      err_ptr = Core.InsertVector(@handle, id, float_array, @dim, meta_str)
      check_error(err_ptr, "Insert")
    end

    def upsert(id, vector, metadata = {})
      raise "Vector dimension must be #{@dim}" if vector.size != @dim
      float_array = FFI::MemoryPointer.new(:float, @dim)
      float_array.write_array_of_float(vector)
      
      meta_str = metadata.empty? ? "" : metadata.to_json
      err_ptr = Core.UpsertVector(@handle, id, float_array, @dim, meta_str)
      check_error(err_ptr, "Upsert")
    end

    def update(id, vector, metadata = {})
      raise "Vector dimension must be #{@dim}" if vector.size != @dim
      float_array = FFI::MemoryPointer.new(:float, @dim)
      float_array.write_array_of_float(vector)
      
      meta_str = metadata.empty? ? "" : metadata.to_json
      err_ptr = Core.UpdateVector(@handle, id, float_array, @dim, meta_str)
      check_error(err_ptr, "Update")
    end

    def delete(id)
      err_ptr = Core.DeleteVector(@handle, id)
      check_error(err_ptr, "Delete")
    end

    def stats
      res_ptr = Core.GetCollectionStats(@handle)
      json_str = parse_result(res_ptr, "Stats")
      json_str ? JSON.parse(json_str) : {}
    end

    def search(vector, k = 10, filter = nil)
      raise "Vector dimension must be #{@dim}" if vector.size != @dim
      float_array = FFI::MemoryPointer.new(:float, @dim)
      float_array.write_array_of_float(vector)
      
      filter_str = filter ? filter.to_hash.to_json : ""
      res_ptr = Core.QueryVector(@handle, float_array, @dim, k, filter_str)
      json_str = parse_result(res_ptr, "Search")
      json_str ? JSON.parse(json_str) : []
    end

    def scan(offset = 0, limit = 100)
      res_ptr = Core.ScanCollection(@handle, offset, limit)
      json_str = parse_result(res_ptr, "Scan")
      json_str ? JSON.parse(json_str) : []
    end

    def insert_batch(ids, vectors, metadata = nil)
      count = ids.size
      raise "ids and vectors must have same length" if vectors.size != count
      raise "ids and metadata must have same length" if metadata && metadata.size != count

      flat_vectors = []
      vectors.each do |vec|
        raise "Vector dimension must be #{@dim}" if vec.size != @dim
        flat_vectors.concat(vec)
      end

      c_float_array = FFI::MemoryPointer.new(:float, count * @dim)
      c_float_array.write_array_of_float(flat_vectors)

      c_id_array = FFI::MemoryPointer.new(:pointer, count)
      id_pointers = ids.map { |id| FFI::MemoryPointer.from_string(id) }
      c_id_array.write_array_of_pointer(id_pointers)

      c_meta_array = nil
      if metadata
        c_meta_array = FFI::MemoryPointer.new(:pointer, count)
        meta_pointers = metadata.map { |m| FFI::MemoryPointer.from_string(m ? m.to_json : "") }
        c_meta_array.write_array_of_pointer(meta_pointers)
      end

      err_ptr = Core.InsertBatch(@handle, c_id_array, c_float_array, count, @dim, c_meta_array)
      check_error(err_ptr, "InsertBatch")
    end

    def delete_batch(ids)
      count = ids.size
      c_id_array = FFI::MemoryPointer.new(:pointer, count)
      id_pointers = ids.map { |id| FFI::MemoryPointer.from_string(id) }
      c_id_array.write_array_of_pointer(id_pointers)

      err_ptr = Core.DeleteBatch(@handle, c_id_array, count)
      check_error(err_ptr, "DeleteBatch")
    end

    def get(id)
      res_ptr = Core.GetVector(@handle, id)
      json_str = parse_result(res_ptr, "Get")
      json_str ? JSON.parse(json_str) : nil
    end

    def count
      c = Core.GetCollectionCount(@handle)
      raise "Failed to get collection count" if c < 0
      c
    end

    def update_if_version(id, vector, expected_version, metadata = {})
      raise "Vector dimension must be #{@dim}" if vector.size != @dim
      float_array = FFI::MemoryPointer.new(:float, @dim)
      float_array.write_array_of_float(vector)
      
      meta_str = metadata.empty? ? "" : metadata.to_json
      err_ptr = Core.UpdateVectorIfVersion(@handle, id, float_array, @dim, meta_str, expected_version)
      check_error(err_ptr, "UpdateIfVersion")
    end

    def delete_if_version(id, expected_version)
      err_ptr = Core.DeleteVectorIfVersion(@handle, id, expected_version)
      check_error(err_ptr, "DeleteIfVersion")
    end

    def set_memory_limit(limit)
      err_ptr = Core.SetCollectionMemoryLimit(@handle, limit)
      check_error(err_ptr, "SetMemoryLimit")
    end

    def memory_usage
      res_ptr = Core.GetCollectionMemoryUsage(@handle)
      json_str = parse_result(res_ptr, "Memory usage")
      json_str ? JSON.parse(json_str) : {}
    end

    def trigger_gc
      err_ptr = Core.TriggerCollectionGC(@handle)
      check_error(err_ptr, "TriggerGC")
    end

    def enable_memory_mapping(path)
      err_ptr = Core.EnableMemoryMapping(@handle, path)
      check_error(err_ptr, "EnableMemoryMapping")
    end

    def disable_memory_mapping
      err_ptr = Core.DisableMemoryMapping(@handle)
      check_error(err_ptr, "DisableMemoryMapping")
    end

    def save_index(path)
      err_ptr = Core.SaveIndex(@handle, path)
      check_error(err_ptr, "SaveIndex")
    end

    def load_index(path)
      err_ptr = Core.LoadIndex(@handle, path)
      check_error(err_ptr, "LoadIndex")
    end

    private

    def check_error(err_ptr, op_name)
      err_msg = Core.from_c_string(err_ptr)
      if err_msg
        if err_msg.start_with?("error: ")
          raise "#{op_name} failed: #{err_msg[7..-1]}"
        else
          raise "#{op_name} failed: #{err_msg}"
        end
      end
    end

    def parse_result(res_ptr, op_name)
      json_str = Core.from_c_string(res_ptr)
      if json_str
        if json_str.start_with?('{"error"')
          err = JSON.parse(json_str)
          raise "#{op_name} failed: #{err['error']}"
        end
      end
      json_str
    end
  end

  class Client
    def initialize(path)
      @handle = Core.OpenDB(path)
      raise "Failed to open database at #{path}" if @handle < 0
    end

    def close
      if @handle >= 0
        Core.CloseDB(@handle)
        @handle = -1
      end
    end

    def create_collection(name, dimension)
      col_handle = Core.CreateCollection(@handle, name, dimension)
      raise "Failed to create collection #{name}" if col_handle < 0
      Collection.new(col_handle, dimension)
    end

    def get_collection(name, dimension)
      col_handle = Core.GetCollection(@handle, name)
      raise "Failed to get collection #{name}" if col_handle < 0
      Collection.new(col_handle, dimension)
    end

    def list_collections
      res_ptr = Core.ListCollections(@handle)
      json_str = parse_result(res_ptr, "List collections")
      json_str ? JSON.parse(json_str) : []
    end

    def delete_collection(name)
      err_ptr = Core.DeleteCollection(@handle, name)
      check_error(err_ptr, "Delete collection")
    end

    def optimize_collection(name)
      err_ptr = Core.OptimizeCollection(@handle, name)
      check_error(err_ptr, "Optimize collection")
    end

    def vacuum
      err_ptr = Core.Vacuum(@handle)
      check_error(err_ptr, "Vacuum")
    end

    def backup(dest)
      err_ptr = Core.Backup(@handle, dest)
      check_error(err_ptr, "Backup")
    end

    def drop
      err_ptr = Core.DropDatabase(@handle)
      check_error(err_ptr, "Drop database")
    end

    def set_memory_limit(limit)
      err_ptr = Core.SetGlobalMemoryLimit(@handle, limit)
      check_error(err_ptr, "Set memory limit")
    end

    def memory_usage
      res_ptr = Core.GetGlobalMemoryUsage(@handle)
      json_str = parse_result(res_ptr, "Memory usage")
      json_str ? JSON.parse(json_str) : {}
    end

    def trigger_gc
      err_ptr = Core.TriggerGlobalGC(@handle)
      check_error(err_ptr, "Trigger GC")
    end

    def ping
      err_ptr = Core.Ping(@handle)
      check_error(err_ptr, "Ping")
    end

    def health
      res_ptr = Core.GetDatabaseHealth(@handle)
      json_str = parse_result(res_ptr, "Health")
      json_str ? JSON.parse(json_str) : {}
    end

    def stats
      res_ptr = Core.GetDatabaseStats(@handle)
      json_str = parse_result(res_ptr, "Stats")
      json_str ? JSON.parse(json_str) : {}
    end

    private

    def check_error(err_ptr, op_name)
      err_msg = Core.from_c_string(err_ptr)
      if err_msg
        if err_msg.start_with?("error: ")
          raise "#{op_name} failed: #{err_msg[7..-1]}"
        else
          raise "#{op_name} failed: #{err_msg}"
        end
      end
    end

    def parse_result(res_ptr, op_name)
      json_str = Core.from_c_string(res_ptr)
      if json_str
        if json_str.start_with?('{"error"')
          err = JSON.parse(json_str)
          raise "#{op_name} failed: #{err['error']}"
        end
      end
      json_str
    end
  end
end
