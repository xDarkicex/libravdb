<?php
namespace LibraVDB;

class LibraVDB {
    private static ?\FFI $ffi = null;

    public static function getFFI(): \FFI {
        if (self::$ffi === null) {
            $os = PHP_OS_FAMILY;
            $libExt = 'so';
            if ($os === 'Darwin') {
                $libExt = 'dylib';
            } elseif ($os === 'Windows') {
                $libExt = 'dll';
            }

            $libPath = getenv('LIBRAVDB_LIBRARY_PATH');
            if (!$libPath) {
                $libPath = __DIR__ . "/../../cgo/libravdb." . $libExt;
                if (!file_exists($libPath)) {
                    $libPath = __DIR__ . "/../libravdb." . $libExt; // Try local for tests
                }
            }

            $cdef = "
                int OpenDB(const char* path);
                void CloseDB(int dbID);
                int CreateCollection(int dbID, const char* name, int dimension);
                int GetCollection(int dbID, const char* name);
                void* InsertVector(int colID, const char* id, float* vector, int dimension, const char* metadata);
                void* QueryVector(int colID, float* vector, int dimension, int k, const char* filter);
                void* ScanCollection(int colID, int offset, int limit);
                void* Vacuum(int dbID);
                void* DropDatabase(int dbID);
                void* InsertBatch(int colID, const char** ids, float* vectors, int count, int dimension, const char** metadata);
                void* DeleteVector(int colID, const char* id);
                void* DeleteBatch(int colID, const char** ids, int count);
                void FreeString(void* ptr);
            ";

            self::$ffi = \FFI::cdef($cdef, $libPath);
        }

        return self::$ffi;
    }

    public static function checkError(?\FFI\CData $errPtr): void {
        if ($errPtr === null) {
            return;
        }

        $charPtr = self::getFFI()->cast("char*", $errPtr);
        $err = \FFI::string($charPtr);
        self::getFFI()->FreeString($errPtr);

        if ($err === "OK") {
            return;
        }

        if (str_starts_with(strtolower($err), "error")) {
            throw new \Exception($err);
        }
    }

    public static function extractString(?\FFI\CData $ptr): string {
        if ($ptr === null) {
            return "{}";
        }

        $charPtr = self::getFFI()->cast("char*", $ptr);
        $str = \FFI::string($charPtr);
        self::getFFI()->FreeString($ptr);

        if (str_starts_with(strtolower($str), "error")) {
            throw new \Exception($str);
        }

        return $str;
    }
}
