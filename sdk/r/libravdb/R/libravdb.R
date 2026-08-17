library(R6)
library(jsonlite)

#' @export
Collection <- R6Class("Collection",
    public = list(
        handle = NULL,
        dimension = NULL,

        initialize = function(handle, dimension) {
            self$handle <- handle
            self$dimension <- dimension
        },

        count = function() {
            r_GetCollectionCount(self$handle)
        },

        insert = function(id, vector, metadata = NULL) {
            if (length(vector) != self$dimension) {
                stop("Vector dimension mismatch")
            }
            meta_json <- if (is.null(metadata)) "" else toJSON(metadata, auto_unbox = TRUE)
            r_InsertVector(self$handle, id, vector, self$dimension, meta_json)
        },

        update = function(id, vector, metadata = NULL) {
            meta_json <- if (is.null(metadata)) "" else toJSON(metadata, auto_unbox = TRUE)
            r_UpdateVector(self$handle, id, vector, self$dimension, meta_json)
        },

        update_if_version = function(id, vector, metadata = NULL, expected_version) {
            meta_json <- if (is.null(metadata)) "" else toJSON(metadata, auto_unbox = TRUE)
            r_UpdateVectorIfVersion(self$handle, id, vector, self$dimension, meta_json, expected_version)
        },

        delete = function(id) {
            r_DeleteVector(self$handle, id)
        },

        get = function(id) {
            res <- r_GetVector(self$handle, id)
            if (res == "") return(NULL)
            fromJSON(res)
        },

        search = function(vector, limit = 10, filter = NULL) {
            filter_json <- if (is.null(filter)) "" else toJSON(filter, auto_unbox = TRUE)
            res <- r_QueryVector(self$handle, vector, self$dimension, limit, filter_json)
            if (res == "") return(NULL)
            fromJSON(res)
        },

        scan = function(offset = 0, limit = 10) {
            res <- r_ScanCollection(self$handle, offset, limit)
            if (res == "") return(NULL)
            fromJSON(res)
        }
    )
)

#' @export
LibraVDB <- R6Class("LibraVDB",
    public = list(
        handle = NULL,

        initialize = function(path) {
            self$handle <- r_OpenDB(path)
        },

        ping = function() {
            r_Ping(self$handle)
        },

        set_memory_limit = function(limit) {
            r_SetGlobalMemoryLimit(self$handle, limit)
        },

        vacuum = function() {
            r_Vacuum(self$handle)
        },

        drop_database = function() {
            r_DropDatabase(self$handle)
        },

        list_collections = function() {
            res <- r_ListCollections(self$handle)
            if (res == "") return(list())
            fromJSON(res)
        },

        create_collection = function(name, dimension) {
            col_handle <- r_CreateCollection(self$handle, name, dimension)
            Collection$new(col_handle, dimension)
        },

        get_collection = function(name, dimension) {
            col_handle <- r_GetCollection(self$handle, name)
            Collection$new(col_handle, dimension)
        },

        query = function(sql) {
            res <- r_DatabaseQuery(self$handle, sql)
            if (res == "") return(NULL)
            fromJSON(res)
        },

        query_with_params = function(sql, params = NULL) {
            params_json <- if (is.null(params)) "" else toJSON(params, auto_unbox = TRUE)
            res <- r_DatabaseQueryWithParams(self$handle, sql, params_json)
            if (res == "") return(NULL)
            fromJSON(res)
        },

        latest_commit_lsn = function() {
            res <- r_DatabaseLatestCommitLSN(self$handle)
            if (res == "") return(NULL)
            payload <- fromJSON(res, simplifyVector = FALSE)
            if (is.null(payload$lsn)) stop("LatestCommitLSN failed: response did not contain lsn")
            as.character(payload$lsn)
        },

        close = function() {
            if (!is.null(self$handle) && self$handle >= 0) {
                r_CloseDB(self$handle)
                self$handle <- -1
            }
        }
    )
)
