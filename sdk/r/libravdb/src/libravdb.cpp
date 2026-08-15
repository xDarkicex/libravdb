#include <Rcpp.h>
#include "libravdb.h"
#include <string>
#include <vector>

using namespace Rcpp;

void check_error(char* errPtr, const char* context) {
    if (errPtr != nullptr) {
        std::string msg(errPtr);
        FreeString(errPtr);
        if (msg != "OK" && msg != "") {
            stop(std::string(context) + " failed: " + msg);
        }
    }
}

String parse_query_result(char* resPtr, const char* context) {
    if (resPtr == nullptr) {
        stop(std::string(context) + " failed: null pointer returned");
    }
    std::string result(resPtr);
    FreeString(resPtr);

    if (result.rfind("{\"error\"", 0) == 0) { // starts_with
        stop(std::string(context) + " failed: " + result);
    }
    return String(result);
}

// [[Rcpp::export]]
int r_OpenDB(std::string path) {
    int handle = OpenDB((char*)path.c_str());
    if (handle < 0) stop("Failed to open database");
    return handle;
}

// [[Rcpp::export]]
void r_CloseDB(int dbID) {
    CloseDB(dbID);
}

// [[Rcpp::export]]
void r_Ping(int dbID) {
    check_error(Ping(dbID), "Ping");
}

// [[Rcpp::export]]
void r_SetGlobalMemoryLimit(int dbID, double limit) {
    check_error(SetGlobalMemoryLimit(dbID, (long long)limit), "SetGlobalMemoryLimit");
}

// [[Rcpp::export]]
void r_Vacuum(int dbID) {
    check_error(Vacuum(dbID), "Vacuum");
}

// [[Rcpp::export]]
void r_DropDatabase(int dbID) {
    check_error(DropDatabase(dbID), "DropDatabase");
}

// [[Rcpp::export]]
String r_ListCollections(int dbID) {
    char* res = ListCollections(dbID);
    return parse_query_result(res, "ListCollections");
}

// [[Rcpp::export]]
int r_CreateCollection(int dbID, std::string name, int dim) {
    int handle = CreateCollection(dbID, (char*)name.c_str(), dim);
    if (handle < 0) stop("Failed to create collection");
    return handle;
}

// [[Rcpp::export]]
int r_GetCollection(int dbID, std::string name) {
    int handle = GetCollection(dbID, (char*)name.c_str());
    if (handle < 0) stop("Failed to get collection");
    return handle;
}

// [[Rcpp::export]]
double r_GetCollectionCount(int colID) {
    return (double)GetCollectionCount(colID);
}

// [[Rcpp::export]]
void r_InsertVector(int colID, std::string id, NumericVector vector, int dim, std::string metadataJSON) {
    std::vector<float> vec(vector.begin(), vector.end());
    check_error(InsertVector(colID, (char*)id.c_str(), vec.data(), dim, (char*)metadataJSON.c_str()), "InsertVector");
}

// [[Rcpp::export]]
void r_UpdateVector(int colID, std::string id, NumericVector vector, int dim, std::string metadataJSON) {
    std::vector<float> vec(vector.begin(), vector.end());
    check_error(UpdateVector(colID, (char*)id.c_str(), vec.data(), dim, (char*)metadataJSON.c_str()), "UpdateVector");
}

// [[Rcpp::export]]
void r_UpdateVectorIfVersion(int colID, std::string id, NumericVector vector, int dim, std::string metadataJSON, double expectedVersion) {
    std::vector<float> vec(vector.begin(), vector.end());
    check_error(UpdateVectorIfVersion(colID, (char*)id.c_str(), vec.data(), dim, (char*)metadataJSON.c_str(), (long long)expectedVersion), "UpdateVectorIfVersion");
}

// [[Rcpp::export]]
void r_DeleteVector(int colID, std::string id) {
    check_error(DeleteVector(colID, (char*)id.c_str()), "DeleteVector");
}

// [[Rcpp::export]]
String r_GetVector(int colID, std::string id) {
    char* res = GetVector(colID, (char*)id.c_str());
    return parse_query_result(res, "GetVector");
}

// [[Rcpp::export]]
String r_QueryVector(int colID, NumericVector vector, int dim, int limit, std::string filterJSON) {
    std::vector<float> vec(vector.begin(), vector.end());
    char* res = QueryVector(colID, vec.data(), dim, limit, (char*)filterJSON.c_str());
    return parse_query_result(res, "QueryVector");
}

// [[Rcpp::export]]
String r_ScanCollection(int colID, int offset, int limit) {
    char* res = ScanCollection(colID, offset, limit);
    return parse_query_result(res, "ScanCollection");
}

// [[Rcpp::export]]
String r_DatabaseQuery(int dbID, std::string sql) {
    char* res = DatabaseQuery(dbID, (char*)sql.c_str());
    return parse_query_result(res, "DatabaseQuery");
}

// [[Rcpp::export]]
String r_DatabaseQueryWithParams(int dbID, std::string sql, std::string paramsStr) {
    char* res = DatabaseQueryWithParams(dbID, (char*)sql.c_str(), (char*)paramsStr.c_str());
    return parse_query_result(res, "DatabaseQueryWithParams");
}
