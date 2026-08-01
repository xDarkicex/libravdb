using System;
using System.Text.Json;
using System.Collections.Generic;

namespace LibraVDB
{
    public class Collection
    {
        private readonly int _colID;
        private readonly string _name;
        private readonly int _dimension;

        internal Collection(int colID, string name, int dimension)
        {
            _colID = colID;
            _name = name;
            _dimension = dimension;
        }

        public void Insert(string id, float[] vector, object metadata = null)
        {
            if (vector.Length != _dimension)
            {
                throw new ArgumentException($"Vector dimension mismatch. Expected {_dimension}, got {vector.Length}");
            }

            string metaStr = metadata != null ? JsonSerializer.Serialize(metadata) : "{}";

            IntPtr errPtr = Bindings.InsertVector(_colID, id, vector, _dimension, metaStr);
            Database.CheckError(errPtr, "Failed to insert vector");
        }
        
        public void UpdateIfVersion(string id, float[] vector, ulong expectedVersion, object metadata = null)
        {
            if (vector.Length != _dimension)
            {
                throw new ArgumentException($"Vector dimension mismatch. Expected {_dimension}, got {vector.Length}");
            }

            string metaStr = metadata != null ? JsonSerializer.Serialize(metadata) : "{}";

            IntPtr errPtr = Bindings.UpdateVectorIfVersion(_colID, id, vector, _dimension, metaStr, expectedVersion);
            Database.CheckError(errPtr, "Failed to update vector");
        }

        public unsafe void InsertBatch(string[] ids, float[][] vectors, object[] metadata = null)
        {
            if (ids.Length != vectors.Length || (metadata != null && ids.Length != metadata.Length))
            {
                throw new ArgumentException("Length of ids, vectors, and metadata must match");
            }

            int count = ids.Length;
            if (count == 0) return;

            string[] metas = new string[count];
            float[] flatVectors = new float[count * _dimension];

            for (int i = 0; i < count; i++)
            {
                if (vectors[i].Length != _dimension)
                {
                    throw new ArgumentException($"Vector at index {i} has dimension {vectors[i].Length}, expected {_dimension}");
                }

                metas[i] = (metadata != null && metadata[i] != null) ? JsonSerializer.Serialize(metadata[i]) : "{}";
                Array.Copy(vectors[i], 0, flatVectors, i * _dimension, _dimension);
            }

            fixed (float* pVectors = flatVectors)
            {
                IntPtr errPtr = Bindings.InsertBatch(_colID, ids, pVectors, count, _dimension, metas);
                Database.CheckError(errPtr, "Failed to insert batch");
            }
        }

        public string Search(float[] vector, int k, Filter filter = null)
        {
            if (vector.Length != _dimension)
            {
                throw new ArgumentException($"Vector dimension mismatch. Expected {_dimension}, got {vector.Length}");
            }

            string filterJson = filter != null ? filter.ToJson() : "{}";

            IntPtr resPtr = Bindings.QueryVector(_colID, vector, _dimension, k, filterJson);
            return Database.ExtractString(resPtr);
        }

        public string Scan(int offset = 0, int limit = 100)
        {
            IntPtr resPtr = Bindings.ScanCollection(_colID, offset, limit);
            return Database.ExtractString(resPtr);
        }

        public string Get(string id)
        {
            IntPtr resPtr = Bindings.GetVector(_colID, id);
            return Database.ExtractString(resPtr);
        }

        public void Delete(string id)
        {
            IntPtr errPtr = Bindings.DeleteVector(_colID, id);
            Database.CheckError(errPtr, "Failed to delete vector");
        }

        public void DeleteBatch(string[] ids)
        {
            IntPtr errPtr = Bindings.DeleteBatch(_colID, ids, ids.Length);
            Database.CheckError(errPtr, "Failed to delete batch");
        }
    }
}
