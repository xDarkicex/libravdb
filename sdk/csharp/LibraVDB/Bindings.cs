using System;
using System.Runtime.InteropServices;

namespace LibraVDB
{
    internal static class Bindings
    {
        private const string LibName = "libravdb";

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int OpenDB(string path);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int CloseDB(int dbID);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void FreeString(IntPtr ptr);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int CreateCollection(int dbID, string name, int dim);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int GetCollection(int dbID, string name);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr DatabaseQuery(int dbID, string sql);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr DatabaseQueryWithParams(int dbID, string sql, string paramsStr);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr InsertVector(int colID, string id, float[] vector, int dim, string metadataJson);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr QueryVector(int colID, float[] vector, int dim, int limit, string filterJson);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr ScanCollection(int colID, int offset, int limit);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr UpdateVector(int colID, string id, float[] vector, int dim, string metadataJson);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr UpdateVectorIfVersion(int colID, string id, float[] vector, int dim, string metadataJson, ulong expectedVersion);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr DeleteVector(int colID, string id);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr DeleteBatch(int colID, [In, MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] ids, int count);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr DropDatabase(int dbID);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr GetVector(int colID, string id);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern long GetCollectionCount(int colID);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static unsafe extern IntPtr InsertBatch(
            int colID,
            [In, MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] ids,
            float* vectors,
            int count,
            int dim,
            [In, MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] metadataJson
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr Vacuum(int dbID);
    }
}
