using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;

namespace LibraVDB
{
    public class LibraVDBException : Exception
    {
        public LibraVDBException(string message) : base(message) { }
    }

    public class Database : IDisposable
    {
        private int _dbID;
        private bool _disposed = false;

        public Database(string path)
        {
            _dbID = Bindings.OpenDB(path);
            if (_dbID < 0)
            {
                throw new LibraVDBException("Failed to open database");
            }
        }

        public Collection CreateCollection(string name, int dimension)
        {
            int colID = Bindings.CreateCollection(_dbID, name, dimension);
            if (colID < 0)
            {
                throw new LibraVDBException("Failed to create collection");
            }
            return new Collection(colID, name, dimension);
        }

        public Collection GetCollection(string name, int dimension)
        {
            int colID = Bindings.GetCollection(_dbID, name);
            if (colID < 0)
            {
                throw new LibraVDBException("Failed to get collection");
            }
            return new Collection(colID, name, dimension);
        }

        public void DropDatabase()
        {
            IntPtr errPtr = Bindings.DropDatabase(_dbID);
            CheckError(errPtr, "Failed to drop database");
        }

        public void Vacuum()
        {
            IntPtr errPtr = Bindings.Vacuum(_dbID);
            CheckError(errPtr, "Failed to vacuum database");
        }

        internal static void CheckError(IntPtr resultPtr, string context)
        {
            if (resultPtr != IntPtr.Zero)
            {
                string errorMsg = Marshal.PtrToStringUTF8(resultPtr) ?? "Unknown native error";
                Bindings.FreeString(resultPtr);
                
                if (errorMsg != "OK" && !errorMsg.StartsWith("{") && !errorMsg.StartsWith("["))
                {
                    throw new LibraVDBException($"{context}: {errorMsg}");
                }
            }
        }

        internal static string ExtractString(IntPtr resultPtr)
        {
            if (resultPtr == IntPtr.Zero)
            {
                return null;
            }

            string result = Marshal.PtrToStringUTF8(resultPtr);
            Bindings.FreeString(resultPtr);
            
            if (result != null && result.StartsWith("ERROR:"))
            {
                throw new LibraVDBException(result);
            }

            return result;
        }

        internal static string ParseQueryResult(IntPtr resultPtr, string context)
        {
            if (resultPtr == IntPtr.Zero)
            {
                throw new LibraVDBException($"{context}: null pointer returned");
            }

            string result = Marshal.PtrToStringUTF8(resultPtr);
            Bindings.FreeString(resultPtr);

            if (result != null && result.StartsWith("{\"error\""))
            {
                throw new LibraVDBException($"{context} failed: {result}");
            }

            return result;
        }

        public string Query(string sql)
        {
            IntPtr resPtr = Bindings.DatabaseQuery(_dbID, sql);
            return ParseQueryResult(resPtr, "Query");
        }

        public string QueryWithParams(string sql, string parameters = "")
        {
            IntPtr resPtr = Bindings.DatabaseQueryWithParams(_dbID, sql, parameters);
            return ParseQueryResult(resPtr, "QueryWithParams");
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                if (_dbID >= 0)
                {
                    Bindings.CloseDB(_dbID);
                    _dbID = -1;
                }
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        ~Database()
        {
            Dispose();
        }
    }
}
