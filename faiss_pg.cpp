// faiss_pg.cpp - versão com índice Faiss salvo em arquivo .faiss

extern "C" {
    #include "postgres.h"
    #include "fmgr.h"
    #include "utils/array.h"
    #include "miscadmin.h"
    #include "utils/memutils.h"
    #include "executor/spi.h"
    #include "access/htup_details.h"
    #include "utils/lsyscache.h"
    #include "catalog/pg_type.h"
    #include "catalog/namespace.h"
    #include "utils/builtins.h"
    #include "utils/timestamp.h"
    #include "funcapi.h"

    PG_MODULE_MAGIC;

    PG_FUNCTION_INFO_V1(faiss_knn_l2);
    Datum faiss_knn_l2(PG_FUNCTION_ARGS);

    PG_FUNCTION_INFO_V1(faiss_knn_l2_table);
    Datum faiss_knn_l2_table(PG_FUNCTION_ARGS);

    PG_FUNCTION_INFO_V1(faiss_build_index);
    Datum faiss_build_index(PG_FUNCTION_ARGS);

    PG_FUNCTION_INFO_V1(faiss_search);
    Datum faiss_search(PG_FUNCTION_ARGS);

    PG_FUNCTION_INFO_V1(faiss_clear_cache);
    Datum faiss_clear_cache(PG_FUNCTION_ARGS);

    PG_FUNCTION_INFO_V1(faiss_clear_all_cache);
    Datum faiss_clear_all_cache(PG_FUNCTION_ARGS);
}

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <regex>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>

using std::string;
using std::vector;

static const char* DEFAULT_INDEX_DIR = "/var/lib/postgresql/faiss_indexes";

// -----------------------------------------------------------------------------
// Utilitários
// -----------------------------------------------------------------------------

static string to_lower_copy(string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return s;
}

static bool starts_with(const string& value, const string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

static string text_arg_to_string(PG_FUNCTION_ARGS, int argno, const char* fallback = "") {
    if (PG_NARGS() <= argno || PG_ARGISNULL(argno)) {
        return string(fallback);
    }
    text* t = PG_GETARG_TEXT_PP(argno);
    char* c = text_to_cstring(t);
    string out(c);
    pfree(c);
    return out;
}

static int get_json_int_param(const string& params, const string& key, int fallback) {
    try {
        std::regex re("\\\"" + key + "\\\"\\s*:\\s*(-?[0-9]+)");
        std::smatch match;
        if (std::regex_search(params, match, re) && match.size() >= 2) {
            return std::stoi(match[1].str());
        }
    } catch (...) {}
    return fallback;
}

static string get_json_string_param(const string& params, const string& key, const string& fallback) {
    try {
        std::regex re("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
        std::smatch match;
        if (std::regex_search(params, match, re) && match.size() >= 2) {
            return match[1].str();
        }
    } catch (...) {}
    return fallback;
}

static string quote_ident_cpp(const string& ident) {
    return string(quote_identifier(ident.c_str()));
}

static string qualified_relation_name(Oid relid) {
    char* relname = get_rel_name(relid);
    if (relname == NULL) {
        ereport(ERROR, (errcode(ERRCODE_UNDEFINED_TABLE), errmsg("relation with oid %u does not exist", relid)));
    }

    Oid nspoid = get_rel_namespace(relid);
    char* nspname = get_namespace_name(nspoid);
    if (nspname == NULL) {
        ereport(ERROR, (errcode(ERRCODE_UNDEFINED_SCHEMA), errmsg("schema for relation oid %u does not exist", relid)));
    }

    return quote_ident_cpp(nspname) + "." + quote_ident_cpp(relname);
}

static string sanitize_index_name(const string& name) {
    string out;
    out.reserve(name.size());
    for (char c : name) {
        if (std::isalnum((unsigned char)c) || c == '_' || c == '-' || c == '.') out.push_back(c);
        else out.push_back('_');
    }
    if (out.empty()) out = "faiss_index";
    return out;
}

static void ensure_dir_exists(const string& dir) {
    struct stat st;
    if (stat(dir.c_str(), &st) == 0) {
        if (!S_ISDIR(st.st_mode)) {
            ereport(ERROR, (errmsg("Faiss index path exists but is not a directory: %s", dir.c_str())));
        }
        return;
    }

    if (mkdir(dir.c_str(), 0700) != 0) {
        int saved_errno = errno;
        const char* saved_error = strerror(saved_errno);
        ereport(ERROR,
                (errmsg("could not create Faiss index directory '%s': %s", dir.c_str(), saved_error),
                 errhint("Create it manually and make it writable by the postgres user.")));
    }
}

static string build_index_path(const string& index_name, const string& params) {
    string dir = get_json_string_param(params, "indexDir", DEFAULT_INDEX_DIR);
    ensure_dir_exists(dir);

    if (!dir.empty() && dir.back() == '/') dir.pop_back();
    return dir + "/" + sanitize_index_name(index_name) + ".faiss";
}

static vector<float> pg_array_to_float_vector(ArrayType *array) {
    if (ARR_NDIM(array) != 1) {
        ereport(ERROR, (errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR), errmsg("expected 1-D real[] array")));
    }

    if (ARR_ELEMTYPE(array) != FLOAT4OID) {
        ereport(ERROR, (errcode(ERRCODE_DATATYPE_MISMATCH), errmsg("expected real[] (float4) array")));
    }

    Datum* elem_values;
    bool* elem_nulls;
    int nelems;

    deconstruct_array(array, FLOAT4OID, sizeof(float4), true, 'i',
                      &elem_values, &elem_nulls, &nelems);

    vector<float> out;
    out.reserve(nelems);

    for (int i = 0; i < nelems; i++) {
        if (elem_nulls[i]) {
            ereport(ERROR, (errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED), errmsg("array elements must not be NULL")));
        }
        out.push_back(DatumGetFloat4(elem_values[i]));
    }

    pfree(elem_values);
    pfree(elem_nulls);
    return out;
}

static ArrayType* int_vector_to_pg_array(const vector<int32>& vals) {
    int nelems = (int) vals.size();
    Datum* elems = (Datum*) palloc(nelems * sizeof(Datum));
    for (int i = 0; i < nelems; i++) elems[i] = Int32GetDatum(vals[i]);

    int16 elmlen;
    bool elmbyval;
    char elmalign;
    get_typlenbyvalalign(INT4OID, &elmlen, &elmbyval, &elmalign);

    ArrayType* result = construct_array(elems, nelems, INT4OID, elmlen, elmbyval, elmalign);
    pfree(elems);
    return result;
}

static int64 get_int64_from_datum(Datum d, Oid type_oid) {
    if (type_oid == INT8OID) return DatumGetInt64(d);
    if (type_oid == INT4OID) return (int64) DatumGetInt32(d);
    ereport(ERROR, (errcode(ERRCODE_DATATYPE_MISMATCH), errmsg("id column must be int4 or int8 (got type oid %u)", type_oid)));
    return 0;
}

static faiss::MetricType parse_metric_type(const string& metric_raw, bool* use_cosine) {
    string metric = to_lower_copy(metric_raw);
    *use_cosine = false;

    if (metric == "cosine" || metric == "cos" || metric == "angular") {
        *use_cosine = true;
        return faiss::METRIC_INNER_PRODUCT;
    }
    if (metric == "ip" || metric == "inner_product" || metric == "inner-product" || metric == "dot" || metric == "dot_product") {
        return faiss::METRIC_INNER_PRODUCT;
    }
    if (metric == "l2" || metric == "euclidean" || metric == "euclid") return faiss::METRIC_L2;

    ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("unsupported metric '%s' (use cosine, ip, or l2)", metric_raw.c_str())));
    return faiss::METRIC_L2;
}

static string canonical_metric_name(const string& metric_raw) {
    string metric = to_lower_copy(metric_raw);
    if (metric == "cosine" || metric == "cos" || metric == "angular") return "cosine";
    if (metric == "ip" || metric == "inner_product" || metric == "inner-product" || metric == "dot" || metric == "dot_product") return "ip";
    if (metric == "l2" || metric == "euclidean" || metric == "euclid") return "l2";
    return metric;
}

static string canonical_index_type(const string& index_type_raw) {
    string t = to_lower_copy(index_type_raw);
    if (t.empty() || t == "flat" || t == "exact") return "flat";
    if (t == "hnsw" || starts_with(t, "hnsw")) return t;
    if (t == "ivf" || t == "ivfflat" || starts_with(t, "ivf")) return t;
    ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("unsupported index_type '%s' (use flat, hnsw32, or ivfflat)", index_type_raw.c_str())));
    return "flat";
}

static int parse_hnsw_m(const string& index_type, const string& params) {
    int m = get_json_int_param(params, "M", 32);
    if (starts_with(index_type, "hnsw") && index_type.size() > 4) {
        try {
            int from_name = std::stoi(index_type.substr(4));
            if (from_name > 0) m = from_name;
        } catch (...) {}
    }
    if (m <= 0) ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("HNSW parameter M must be positive")));
    return m;
}

static faiss::Index* make_base_index(int d, faiss::MetricType metric, const string& index_type, int64 ntotal, const string& params) {
    if (index_type == "flat") {
        if (metric == faiss::METRIC_INNER_PRODUCT) return new faiss::IndexFlatIP(d);
        return new faiss::IndexFlatL2(d);
    }

    if (starts_with(index_type, "hnsw")) {
        int m = parse_hnsw_m(index_type, params);
        int ef_construction = get_json_int_param(params, "efConstruction", 200);
        int ef_search = get_json_int_param(params, "efSearch", 128);

        faiss::IndexHNSWFlat* hnsw = new faiss::IndexHNSWFlat(d, m, metric);
        hnsw->hnsw.efConstruction = ef_construction;
        hnsw->hnsw.efSearch = ef_search;
        return hnsw;
    }

    if (starts_with(index_type, "ivf")) {
        int nlist = get_json_int_param(params, "nlist", 4096);
        int nprobe = get_json_int_param(params, "nprobe", 64);
        if (nlist <= 0) ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("IVF parameter nlist must be positive")));
        if (ntotal > 0 && nlist > ntotal) nlist = (int) ntotal;

        faiss::Index* quantizer = (metric == faiss::METRIC_INNER_PRODUCT)
            ? (faiss::Index*) new faiss::IndexFlatIP(d)
            : (faiss::Index*) new faiss::IndexFlatL2(d);

        faiss::IndexIVFFlat* ivf = new faiss::IndexIVFFlat(quantizer, d, nlist, metric);
        ivf->own_fields = true;
        ivf->nprobe = nprobe;
        return ivf;
    }

    ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("unsupported index_type '%s'", index_type.c_str())));
    return NULL;
}

static faiss::Index* unwrap_id_map(faiss::Index* index) {
    if (faiss::IndexIDMap2* idmap2 = dynamic_cast<faiss::IndexIDMap2*>(index)) return idmap2->index;
    if (faiss::IndexIDMap* idmap = dynamic_cast<faiss::IndexIDMap*>(index)) return idmap->index;
    return index;
}

static void apply_search_params(faiss::Index* index, const string& search_params) {
    faiss::Index* base = unwrap_id_map(index);

    if (faiss::IndexHNSW* hnsw = dynamic_cast<faiss::IndexHNSW*>(base)) {
        int ef_search = get_json_int_param(search_params, "efSearch", hnsw->hnsw.efSearch);
        if (ef_search > 0) hnsw->hnsw.efSearch = ef_search;
    }

    if (faiss::IndexIVF* ivf = dynamic_cast<faiss::IndexIVF*>(base)) {
        int nprobe = get_json_int_param(search_params, "nprobe", (int) ivf->nprobe);
        if (nprobe > 0) ivf->nprobe = nprobe;
    }
}

// -----------------------------------------------------------------------------
// Cache em memória por backend PostgreSQL
// -----------------------------------------------------------------------------

struct CachedFaissIndex {
    faiss::Index* index;
    int dim;
    string metric;
    string index_type;
    bool normalize_vectors;
    string index_path;
    TimestampTz updated_at;
};

struct LoadedIndexInfo {
    faiss::Index* index;
    int dim;
    string metric;
    string index_type;
    bool normalize_vectors;
    string index_path;
};

static std::unordered_map<string, CachedFaissIndex> faiss_cache;

static void erase_cache_entry(const string& index_name) {
    auto it = faiss_cache.find(index_name);
    if (it != faiss_cache.end()) {
        delete it->second.index;
        faiss_cache.erase(it);
    }
}

static void clear_all_cache_entries() {
    for (auto& kv : faiss_cache) delete kv.second.index;
    faiss_cache.clear();
}

static LoadedIndexInfo get_cached_index(const string& index_name) {
    if (SPI_connect() != SPI_OK_CONNECT) ereport(ERROR, (errmsg("SPI_connect failed")));

    const char* cmd =
        "SELECT dim, metric, index_type, normalize_vectors, index_path, updated_at, dirty "
        "FROM faiss_indexes WHERE name = $1";

    Oid argtypes[1] = { TEXTOID };
    Datum values[1] = { CStringGetTextDatum(index_name.c_str()) };
    const char nulls[1] = { ' ' };

    int ret = SPI_execute_with_args(cmd, 1, argtypes, values, nulls, true, 1);
    if (ret != SPI_OK_SELECT || SPI_processed != 1) {
        SPI_finish();
        ereport(ERROR, (errmsg("index '%s' not found in faiss_indexes", index_name.c_str())));
    }

    HeapTuple tup = SPI_tuptable->vals[0];
    TupleDesc tupdesc = SPI_tuptable->tupdesc;
    bool isnull = false;

    int32 dim = DatumGetInt32(SPI_getbinval(tup, tupdesc, 1, &isnull));
    if (isnull) { SPI_finish(); ereport(ERROR, (errmsg("dim is NULL"))); }

    Datum metricDatum = SPI_getbinval(tup, tupdesc, 2, &isnull);
    if (isnull) { SPI_finish(); ereport(ERROR, (errmsg("metric is NULL"))); }
    char* metricC = TextDatumGetCString(metricDatum);
    string metric(metricC);
    pfree(metricC);

    Datum indexTypeDatum = SPI_getbinval(tup, tupdesc, 3, &isnull);
    if (isnull) { SPI_finish(); ereport(ERROR, (errmsg("index_type is NULL"))); }
    char* indexTypeC = TextDatumGetCString(indexTypeDatum);
    string index_type(indexTypeC);
    pfree(indexTypeC);

    bool normalize_vectors = DatumGetBool(SPI_getbinval(tup, tupdesc, 4, &isnull));
    if (isnull) { SPI_finish(); ereport(ERROR, (errmsg("normalize_vectors is NULL"))); }

    Datum pathDatum = SPI_getbinval(tup, tupdesc, 5, &isnull);
    if (isnull) { SPI_finish(); ereport(ERROR, (errmsg("index_path is NULL"))); }
    char* pathC = TextDatumGetCString(pathDatum);
    string index_path(pathC);
    pfree(pathC);

    TimestampTz updated_at = DatumGetTimestampTz(SPI_getbinval(tup, tupdesc, 6, &isnull));
    if (isnull) { SPI_finish(); ereport(ERROR, (errmsg("updated_at is NULL"))); }

    bool dirty = DatumGetBool(SPI_getbinval(tup, tupdesc, 7, &isnull));
    if (!isnull && dirty) {
        SPI_finish();
        ereport(ERROR, (errmsg("index '%s' is dirty; run faiss_build_index again", index_name.c_str())));
    }

    auto cached = faiss_cache.find(index_name);
    if (cached != faiss_cache.end() && cached->second.updated_at == updated_at && cached->second.index_path == index_path) {
        SPI_finish();
        return LoadedIndexInfo{cached->second.index, cached->second.dim, cached->second.metric, cached->second.index_type, cached->second.normalize_vectors, cached->second.index_path};
    }

    SPI_finish();

    faiss::Index* loaded = faiss::read_index(index_path.c_str());
    if (loaded == NULL) ereport(ERROR, (errmsg("failed to read Faiss index from '%s'", index_path.c_str())));

    erase_cache_entry(index_name);
    faiss_cache[index_name] = CachedFaissIndex{loaded, dim, metric, index_type, normalize_vectors, index_path, updated_at};

    return LoadedIndexInfo{loaded, dim, metric, index_type, normalize_vectors, index_path};
}

// -----------------------------------------------------------------------------
// Funções antigas auxiliares
// -----------------------------------------------------------------------------

extern "C" Datum faiss_knn_l2(PG_FUNCTION_ARGS) {
    ArrayType* queryArr = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType* dataArr = PG_GETARG_ARRAYTYPE_P(1);
    int32 n = PG_GETARG_INT32(2);
    int32 d = PG_GETARG_INT32(3);
    int32 k = PG_GETARG_INT32(4);

    if (n <= 0 || d <= 0 || k <= 0) ereport(ERROR, (errmsg("n, d and k must be positive")));

    vector<float> query = pg_array_to_float_vector(queryArr);
    vector<float> data = pg_array_to_float_vector(dataArr);

    if ((int) query.size() != d) ereport(ERROR, (errmsg("query vector length (%d) must equal d (%d)", (int) query.size(), d)));
    if ((int) data.size() != n * d) ereport(ERROR, (errmsg("data length (%d) must equal n * d (%d)", (int) data.size(), n * d)));
    if (k > n) k = n;

    faiss::IndexFlatL2 index(d);
    index.add(n, data.data());

    vector<float> distances(k);
    vector<faiss::idx_t> labels(k);
    index.search(1, query.data(), k, distances.data(), labels.data());

    vector<int32> result_idx;
    result_idx.reserve(k);
    for (int i = 0; i < k; i++) result_idx.push_back((int32) labels[i]);

    PG_RETURN_ARRAYTYPE_P(int_vector_to_pg_array(result_idx));
}

extern "C" Datum faiss_knn_l2_table(PG_FUNCTION_ARGS) {
    ArrayType* queryArr = PG_GETARG_ARRAYTYPE_P(0);
    int32 k = PG_GETARG_INT32(1);
    if (k <= 0) ereport(ERROR, (errmsg("k must be positive")));

    vector<float> query = pg_array_to_float_vector(queryArr);
    int d = (int) query.size();
    if (d <= 0) ereport(ERROR, (errmsg("query vector must not be empty")));

    if (SPI_connect() != SPI_OK_CONNECT) ereport(ERROR, (errmsg("SPI_connect failed")));

    const char* cmd = "SELECT id, embedding FROM faiss_items";
    int ret = SPI_execute(cmd, true, 0);
    if (ret != SPI_OK_SELECT) { SPI_finish(); ereport(ERROR, (errmsg("SPI_execute failed with code %d", ret))); }

    uint64 nrows = SPI_processed;
    if (nrows == 0) {
        SPI_finish();
        vector<int32> empty;
        PG_RETURN_ARRAYTYPE_P(int_vector_to_pg_array(empty));
    }

    TupleDesc tupdesc = SPI_tuptable->tupdesc;
    SPITupleTable* tuptable = SPI_tuptable;
    vector<int32> ids;
    vector<float> data;
    ids.reserve(nrows);
    data.reserve(nrows * d);

    for (uint64 i = 0; i < nrows; i++) {
        HeapTuple tuple = tuptable->vals[i];
        bool isnull;

        Datum idDatum = SPI_getbinval(tuple, tupdesc, 1, &isnull);
        if (isnull) { SPI_finish(); ereport(ERROR, (errmsg("id must not be NULL"))); }
        int32 id = DatumGetInt32(idDatum);

        Datum embDatum = SPI_getbinval(tuple, tupdesc, 2, &isnull);
        if (isnull) { SPI_finish(); ereport(ERROR, (errmsg("embedding must not be NULL"))); }

        vector<float> emb = pg_array_to_float_vector(DatumGetArrayTypeP(embDatum));
        if ((int) emb.size() != d) { SPI_finish(); ereport(ERROR, (errmsg("all embeddings must have dimension %d", d))); }

        ids.push_back(id);
        data.insert(data.end(), emb.begin(), emb.end());
    }

    SPI_finish();

    int n = (int) nrows;
    if (k > n) k = n;

    faiss::IndexFlatL2 index(d);
    index.add(n, data.data());

    vector<float> distances(k);
    vector<faiss::idx_t> labels(k);
    index.search(1, query.data(), k, distances.data(), labels.data());

    vector<int32> result_ids;
    result_ids.reserve(k);
    for (int i = 0; i < k; i++) {
        int idx = (int) labels[i];
        if (idx < 0 || idx >= n) ereport(ERROR, (errmsg("Faiss returned invalid index %d", idx)));
        result_ids.push_back(ids[idx]);
    }

    PG_RETURN_ARRAYTYPE_P(int_vector_to_pg_array(result_ids));
}

// -----------------------------------------------------------------------------
// Construção do índice em arquivo
// -----------------------------------------------------------------------------

extern "C" Datum faiss_build_index(PG_FUNCTION_ARGS) {
    text* indexNameText = PG_GETARG_TEXT_PP(0);
    char* indexNameC = text_to_cstring(indexNameText);
    string indexName(indexNameC);
    pfree(indexNameC);

    bool legacy_mode = (PG_NARGS() == 1);

    string tableSql;
    string idColumn = "id";
    string embeddingColumn = "embedding";
    string metricRaw = legacy_mode ? "l2" : text_arg_to_string(fcinfo, 4, "cosine");
    string indexTypeRaw = legacy_mode ? "flat" : text_arg_to_string(fcinfo, 5, "flat");
    bool normalize_vectors = legacy_mode ? false : PG_GETARG_BOOL(6);
    string params = legacy_mode ? "{}" : text_arg_to_string(fcinfo, 7, "{}");

    if (legacy_mode) {
        tableSql = "faiss_items";
    } else {
        Oid tableOid = PG_GETARG_OID(1);
        Name idName = (Name) PG_GETARG_POINTER(2);
        Name embName = (Name) PG_GETARG_POINTER(3);
        tableSql = qualified_relation_name(tableOid);
        idColumn = string(NameStr(*idName));
        embeddingColumn = string(NameStr(*embName));
    }

    bool cosine_metric = false;
    faiss::MetricType metric = parse_metric_type(metricRaw, &cosine_metric);
    string metricName = canonical_metric_name(metricRaw);
    string indexType = canonical_index_type(indexTypeRaw);
    string indexPath = build_index_path(indexName, params);

    if (cosine_metric && !normalize_vectors) {
        ereport(WARNING, (errmsg("metric='cosine' requested with normalize_vectors=false; results only match cosine if vectors are already normalized")));
    }

    if (SPI_connect() != SPI_OK_CONNECT) ereport(ERROR, (errmsg("SPI_connect failed")));

    string countSql = "SELECT count(*) FROM " + tableSql;
    int ret = SPI_execute(countSql.c_str(), true, 0);
    if (ret != SPI_OK_SELECT || SPI_processed != 1) {
        SPI_finish();
        ereport(ERROR, (errmsg("failed to count rows from %s", tableSql.c_str())));
    }

    bool isnull = false;
    int64 ntotal = DatumGetInt64(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
    if (isnull || ntotal <= 0) {
        SPI_finish();
        ereport(ERROR, (errmsg("%s is empty; nothing to index", tableSql.c_str())));
    }

    string cmd = "SELECT " + quote_ident_cpp(idColumn) + ", " + quote_ident_cpp(embeddingColumn) +
                 " FROM " + tableSql + " ORDER BY " + quote_ident_cpp(idColumn);

    Portal portal = SPI_cursor_open_with_args(NULL, cmd.c_str(), 0, NULL, NULL, NULL, true, 0);
    if (portal == NULL) {
        SPI_finish();
        ereport(ERROR, (errmsg("SPI_cursor_open_with_args failed")));
    }

    int fetch_batch_size = get_json_int_param(params, "fetchBatchSize", 5000);
    if (fetch_batch_size <= 0) fetch_batch_size = 5000;

    vector<float> data;
    vector<faiss::idx_t> ids;
    ids.reserve((size_t) ntotal);

    int d = -1;
    Oid id_type = InvalidOid;

    while (true) {
        SPI_cursor_fetch(portal, true, fetch_batch_size);
        uint64 batch_rows = SPI_processed;
        if (batch_rows == 0) break;

        SPITupleTable* tuptable = SPI_tuptable;
        TupleDesc tupdesc = tuptable->tupdesc;

        if (id_type == InvalidOid) id_type = TupleDescAttr(tupdesc, 0)->atttypid;

        for (uint64 i = 0; i < batch_rows; i++) {
            HeapTuple tup = tuptable->vals[i];
            bool idnull = false;
            bool embnull = false;
            Datum idDatum = SPI_getbinval(tup, tupdesc, 1, &idnull);
            Datum embDatum = SPI_getbinval(tup, tupdesc, 2, &embnull);
            if (idnull || embnull) {
                SPI_cursor_close(portal);
                SPI_finish();
                ereport(ERROR, (errmsg("id/embedding must not be NULL")));
            }

            int64 id64 = get_int64_from_datum(idDatum, id_type);
            vector<float> v = pg_array_to_float_vector(DatumGetArrayTypeP(embDatum));

            if (d < 0) {
                d = (int) v.size();
                if (d <= 0) {
                    SPI_cursor_close(portal);
                    SPI_finish();
                    ereport(ERROR, (errmsg("embedding dimension must be > 0")));
                }
                data.reserve((size_t) ntotal * (size_t) d);
            }

            if ((int) v.size() != d) {
                SPI_cursor_close(portal);
                SPI_finish();
                ereport(ERROR, (errmsg("all embeddings must have same dimension (%d)", d)));
            }

            ids.push_back((faiss::idx_t) id64);
            data.insert(data.end(), v.begin(), v.end());
        }
    }

    SPI_cursor_close(portal);

    if ((int64) ids.size() != ntotal) {
        SPI_finish();
        ereport(ERROR, (errmsg("expected %ld rows but read %ld rows", (long) ntotal, (long) ids.size())));
    }

    if (normalize_vectors) faiss::fvec_renorm_L2(d, (size_t) ntotal, data.data());

    std::unique_ptr<faiss::Index> base(make_base_index(d, metric, indexType, ntotal, params));
    if (!base->is_trained) base->train((faiss::idx_t) ntotal, data.data());

    faiss::IndexIDMap2 index(base.release());
    index.own_fields = true;
    index.add_with_ids((faiss::idx_t) ntotal, data.data(), ids.data());

    string tmpPath = indexPath + ".tmp";

    try {
        faiss::write_index(&index, tmpPath.c_str());
    } catch (const std::exception& e) {
        SPI_finish();
        ereport(ERROR, (errmsg("failed to write Faiss index to '%s': %s", tmpPath.c_str(), e.what())));
    }

    if (rename(tmpPath.c_str(), indexPath.c_str()) != 0) {
        int saved_errno = errno;
        const char* saved_error = strerror(saved_errno);
        unlink(tmpPath.c_str());
        SPI_finish();
        ereport(ERROR, (errmsg("failed to rename '%s' to '%s': %s", tmpPath.c_str(), indexPath.c_str(), saved_error)));
    }

    const char* upsert =
        "INSERT INTO faiss_indexes(name, dim, metric, index_type, normalize_vectors, ntotal, index_path, params, updated_at, dirty) "
        "VALUES($1, $2, $3, $4, $5, $6, $7, $8, now(), false) "
        "ON CONFLICT (name) DO UPDATE SET "
        "dim = EXCLUDED.dim, metric = EXCLUDED.metric, index_type = EXCLUDED.index_type, "
        "normalize_vectors = EXCLUDED.normalize_vectors, ntotal = EXCLUDED.ntotal, "
        "index_path = EXCLUDED.index_path, params = EXCLUDED.params, updated_at = now(), dirty = false";

    Oid argtypes[8] = { TEXTOID, INT4OID, TEXTOID, TEXTOID, BOOLOID, INT8OID, TEXTOID, TEXTOID };
    Datum values[8];
    const char nulls[8] = { ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ' };

    values[0] = CStringGetTextDatum(indexName.c_str());
    values[1] = Int32GetDatum(d);
    values[2] = CStringGetTextDatum(metricName.c_str());
    values[3] = CStringGetTextDatum(indexType.c_str());
    values[4] = BoolGetDatum(normalize_vectors);
    values[5] = Int64GetDatum(ntotal);
    values[6] = CStringGetTextDatum(indexPath.c_str());
    values[7] = CStringGetTextDatum(params.c_str());

    ret = SPI_execute_with_args(upsert, 8, argtypes, values, nulls, false, 0);
    if (ret != SPI_OK_INSERT && ret != SPI_OK_UPDATE) {
        SPI_finish();
        ereport(ERROR, (errmsg("failed to upsert faiss_indexes (%d)", ret)));
    }

    SPI_finish();

    erase_cache_entry(indexName);
    PG_RETURN_VOID();
}

// -----------------------------------------------------------------------------
// Busca
// -----------------------------------------------------------------------------

extern "C" Datum faiss_search(PG_FUNCTION_ARGS) {
    text* indexNameText = PG_GETARG_TEXT_PP(0);
    ArrayType* queryArr = PG_GETARG_ARRAYTYPE_P(1);
    int32 k = PG_GETARG_INT32(2);
    string searchParams = text_arg_to_string(fcinfo, 3, "{}");

    if (k <= 0) ereport(ERROR, (errmsg("k must be positive")));

    char* indexNameC = text_to_cstring(indexNameText);
    string indexName(indexNameC);
    pfree(indexNameC);

    vector<float> query = pg_array_to_float_vector(queryArr);
    int d = (int) query.size();
    if (d <= 0) ereport(ERROR, (errmsg("query must not be empty")));

    LoadedIndexInfo loaded = get_cached_index(indexName);
    faiss::Index* index = loaded.index;

    if (loaded.dim != d) ereport(ERROR, (errmsg("query dim (%d) != index dim (%d)", d, loaded.dim)));
    if (index->d != d) ereport(ERROR, (errmsg("query dim (%d) != faiss index dim (%d)", d, (int) index->d)));

    if (loaded.normalize_vectors) faiss::fvec_renorm_L2(d, 1, query.data());
    if (k > (int32) index->ntotal) k = (int32) index->ntotal;

    apply_search_params(index, searchParams);

    vector<float> distances(k);
    vector<faiss::idx_t> labels(k);
    index->search(1, query.data(), k, distances.data(), labels.data());

    ReturnSetInfo* rsinfo = (ReturnSetInfo*) fcinfo->resultinfo;
    if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo)) {
        ereport(ERROR, (errmsg("set-valued function called in a context that cannot accept a set")));
    }

    rsinfo->returnMode = SFRM_Materialize;

    TupleDesc outdesc;
    if (get_call_result_type(fcinfo, NULL, &outdesc) != TYPEFUNC_COMPOSITE) {
        ereport(ERROR, (errmsg("return type must be a composite type")));
    }

    if (rsinfo->econtext == NULL) ereport(ERROR, (errmsg("no execution context available")));

    MemoryContext oldcontext = MemoryContextSwitchTo(rsinfo->econtext->ecxt_per_query_memory);
    Tuplestorestate* tupstore = tuplestore_begin_heap(true, false, work_mem);

    for (int i = 0; i < k; i++) {
        if (labels[i] < 0) continue;
        Datum outvals[2];
        bool outnulls[2] = { false, false };
        outvals[0] = Int64GetDatum((int64) labels[i]);
        outvals[1] = Float4GetDatum((float4) distances[i]);
        tuplestore_putvalues(tupstore, outdesc, outvals, outnulls);
    }

    rsinfo->setResult = tupstore;
    rsinfo->setDesc = outdesc;
    MemoryContextSwitchTo(oldcontext);

    PG_RETURN_NULL();
}

extern "C" Datum faiss_clear_cache(PG_FUNCTION_ARGS) {
    text* indexNameText = PG_GETARG_TEXT_PP(0);
    char* indexNameC = text_to_cstring(indexNameText);
    string indexName(indexNameC);
    pfree(indexNameC);
    erase_cache_entry(indexName);
    PG_RETURN_VOID();
}

extern "C" Datum faiss_clear_all_cache(PG_FUNCTION_ARGS) {
    clear_all_cache_entries();
    PG_RETURN_VOID();
}
