// faiss_pg.cpp

// Parte C (exposta ao PostgreSQL)
extern "C" {
    #include "postgres.h"
    #include "fmgr.h"
    #include "utils/array.h"
    #include "miscadmin.h" 
    #include "utils/guc.h"
    #include "utils/memutils.h"
    #include "executor/executor.h"
    #include "utils/tuplestore.h"
    #include "executor/spi.h"
    #include "access/htup_details.h"
    #include "utils/lsyscache.h"   // get_typlenbyvalalign
    #include "catalog/pg_type.h"   // FLOAT4OID, INT4OID, INT8OID, TEXTOID, BYTEAOID
    #include "utils/builtins.h"    // text_to_cstring, CStringGetTextDatum
    #include "funcapi.h"          // SRF helpers

    PG_MODULE_MAGIC;

    // Funções já existentes
    PG_FUNCTION_INFO_V1(faiss_knn_l2);
    Datum faiss_knn_l2(PG_FUNCTION_ARGS);

    PG_FUNCTION_INFO_V1(faiss_knn_l2_table);
    Datum faiss_knn_l2_table(PG_FUNCTION_ARGS);

    // Novas implementações
    // - Persistir índice Faiss no PostgreSQL (em bytea)
    // - Buscar retornando uma tabela (id, distance)
    PG_FUNCTION_INFO_V1(faiss_build_index);
    Datum faiss_build_index(PG_FUNCTION_ARGS);

    PG_FUNCTION_INFO_V1(faiss_search);
    Datum faiss_search(PG_FUNCTION_ARGS);
}

// Parte C++ / Faiss
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cstring>

#include <faiss/IndexFlat.h>      // índice exato L2
#include <faiss/IndexIDMap.h>     // mapear ids reais
#include <faiss/index_io.h>       // read_index / write_index
#include <faiss/impl/io.h>

using std::vector;

// Converte um array Postgres de float4 (real) para std::vector<float>
static vector<float> pg_array_to_float_vector(ArrayType *array) {
    if (ARR_NDIM(array) != 1)
        ereport(ERROR,
                (errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
                 errmsg("expected 1-D real[] array")));

    Oid elem_type = ARR_ELEMTYPE(array);
    if (elem_type != FLOAT4OID)
        ereport(ERROR,
                (errcode(ERRCODE_DATATYPE_MISMATCH),
                 errmsg("expected real[] (float4) array")));

    Datum  *elem_values;
    bool   *elem_nulls;
    int     nelems;

    deconstruct_array(array,
                      FLOAT4OID,           // tipo dos elementos
                      sizeof(float4),      // tamanho
                      true,                // float4 é by value
                      'i',                 // alinhamento
                      &elem_values,
                      &elem_nulls,
                      &nelems);

    vector<float> out;
    out.reserve(nelems);

    for (int i = 0; i < nelems; i++) {
        if (elem_nulls[i])
            ereport(ERROR,
                    (errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
                     errmsg("array elements must not be NULL")));

        float4 v = DatumGetFloat4(elem_values[i]);
        out.push_back(v);
    }

    pfree(elem_values);
    pfree(elem_nulls);

    return out;
}

// Constrói um int[] Postgres a partir de um vetor de int32
static ArrayType *int_vector_to_pg_array(const vector<int32> &vals) {
    int nelems = (int)vals.size();
    Datum *elems = (Datum *) palloc(nelems * sizeof(Datum));

    for (int i = 0; i < nelems; i++) {
        elems[i] = Int32GetDatum(vals[i]);
    }

    int16 elmlen;
    bool  elmbyval;
    char  elmalign;

    get_typlenbyvalalign(INT4OID, &elmlen, &elmbyval, &elmalign);

    ArrayType *result = construct_array(elems,
                                        nelems,
                                        INT4OID,
                                        elmlen,
                                        elmbyval,
                                        elmalign);

    pfree(elems);

    return result;
}

// Lê um id que pode ser INT4 ou INT8 e devolve int64
static int64 get_int64_from_datum(Datum d, Oid type_oid) {
    if (type_oid == INT8OID) {
        return DatumGetInt64(d);
    }
    if (type_oid == INT4OID) {
        return (int64) DatumGetInt32(d);
    }
    ereport(ERROR,
            (errcode(ERRCODE_DATATYPE_MISMATCH),
             errmsg("id column must be int4 or int8 (got type oid %u)", type_oid)));
    return 0; // unreachable
}

// Converte um buffer (bytes) em bytea
static bytea* buffer_to_bytea(const uint8_t* data, size_t len) {
    bytea* out = (bytea*) palloc(len + VARHDRSZ);
    SET_VARSIZE(out, len + VARHDRSZ);
    memcpy(VARDATA(out), data, len);
    return out;
}

// query: real[]  (tamanho d)
// data : real[]  (tamanho n * d, achatado)
// n    : nº de vetores
// d    : dimensão
// k    : top-k
extern "C" Datum faiss_knn_l2(PG_FUNCTION_ARGS) {
    // 1) Pegar argumentos
    ArrayType *queryArr = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *dataArr  = PG_GETARG_ARRAYTYPE_P(1);
    int32 n = PG_GETARG_INT32(2);
    int32 d = PG_GETARG_INT32(3);
    int32 k = PG_GETARG_INT32(4);

    if (n <= 0 || d <= 0 || k <= 0)
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                 errmsg("n, d and k must be positive")));

    // 2) Converter arrays para std::vector<float>
    vector<float> query = pg_array_to_float_vector(queryArr);
    vector<float> data  = pg_array_to_float_vector(dataArr);

    if ((int)query.size() != d)
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                 errmsg("query vector length (%d) must equal d (%d)",
                        (int)query.size(), d)));

    if ((int)data.size() != n * d)
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                 errmsg("data length (%d) must equal n * d (%d)",
                        (int)data.size(), n * d)));

    if (k > n)
        k = n; // ajusta k se for maior que n

    // 3) Criar índice Faiss (IndexFlatL2 = busca exata em L2)
    faiss::IndexFlatL2 index(d);  // dimensão d

    // 4) Adicionar dataset ao índice
    index.add(n, data.data());

    // 5) Buscar k vizinhos para 1 query
    vector<float> distances(k);
    vector<faiss::idx_t> labels(k);

    index.search(1,                // número de queries
                 query.data(),     // ponteiro p/ query (1 * d)
                 k,                // top-k
                 distances.data(), // distâncias de saída
                 labels.data());   // índices de saída

    // 6) Converter labels (idx_t) para int32
    vector<int32> result_idx;
    result_idx.reserve(k);
    for (int i = 0; i < k; i++) {
        result_idx.push_back((int32)labels[i]);
    }

    ArrayType *resultArr = int_vector_to_pg_array(result_idx);

    PG_RETURN_ARRAYTYPE_P(resultArr);
}

extern "C" Datum faiss_knn_l2_table(PG_FUNCTION_ARGS) {
    // 1) Ler argumentos da função SQL
    ArrayType *queryArr = PG_GETARG_ARRAYTYPE_P(0);
    int32 k             = PG_GETARG_INT32(1);

    if (k <= 0)
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                 errmsg("k must be positive")));

    // Converte query para vetor de floats e descobre dimensão
    vector<float> query = pg_array_to_float_vector(queryArr);
    int d = (int) query.size();
    if (d <= 0)
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                 errmsg("query vector must not be empty")));

    // 2) Conectar ao SPI (API interna do Postgres pra rodar SQL)
    if (SPI_connect() != SPI_OK_CONNECT)
        ereport(ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("SPI_connect failed")));

    // Vamos ler todos os vetores da tabela faiss_items
    const char *cmd = "SELECT id, embedding FROM faiss_items";

    int ret = SPI_execute(cmd, true /* read-only */, 0 /* sem limite */);
    if (ret != SPI_OK_SELECT) {
        SPI_finish();
        ereport(ERROR,
                (errcode(ERRCODE_INTERNAL_ERROR),
                 errmsg("SPI_execute failed with code %d", ret)));
    }

    uint64 nrows = SPI_processed;

    // Se não tiver linhas, retorna array vazio
    if (nrows == 0) {
        SPI_finish();
        vector<int32> empty;
        ArrayType *resultArr = int_vector_to_pg_array(empty);
        PG_RETURN_ARRAYTYPE_P(resultArr);
    }

    TupleDesc tupdesc = SPI_tuptable->tupdesc;
    SPITupleTable *tuptable = SPI_tuptable;

    // 3) Montar dataset para o Faiss
    vector<int32> ids;      // guarda os ids da tabela
    vector<float> data;     // guarda todos os vetores concatenados

    ids.reserve(nrows);
    data.reserve(nrows * d);

    for (uint64 i = 0; i < nrows; i++) {
        HeapTuple tuple = tuptable->vals[i];
        bool isnull;

        // Coluna 1: id (integer)
        Datum idDatum = SPI_getbinval(tuple, tupdesc, 1, &isnull);
        if (isnull) {
            SPI_finish();
            ereport(ERROR,
                    (errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
                     errmsg("id must not be NULL")));
        }

        int32 id = DatumGetInt32(idDatum);

        // Coluna 2: embedding (real[])
        Datum embDatum = SPI_getbinval(tuple, tupdesc, 2, &isnull);
        if (isnull) {
            SPI_finish();
            ereport(ERROR,
                    (errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
                     errmsg("embedding must not be NULL")));
        }

        ArrayType *embArray = DatumGetArrayTypeP(embDatum);
        vector<float> emb = pg_array_to_float_vector(embArray);

        if ((int)emb.size() != d) {
            SPI_finish();
            ereport(ERROR,
                    (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                     errmsg("all embeddings must have dimension %d (found %d)",
                            d, (int)emb.size())));
        }

        ids.push_back(id);
        // concatena o vetor emb no data
        data.insert(data.end(), emb.begin(), emb.end());
    }

    SPI_finish();  // já montamos tudo, podemos desconectar

    int n = (int) nrows;
    if (k > n)
        k = n;  // não faz sentido k > n

    // 4) Criar índice Faiss em memória
    faiss::IndexFlatL2 index(d);   // índice L2 exato
    index.add(n, data.data());     // adiciona n vetores

    // 5) Fazer a busca k-NN
    vector<float> distances(k);
    vector<faiss::idx_t> labels(k);

    index.search(1,                   // número de queries
                 query.data(),        // ponteiro para a query
                 k,                   // top-k
                 distances.data(),    // distâncias
                 labels.data());      // índices retornados (0..n-1)

    // 6) Converter índices internos do Faiss -> ids da tabela
    vector<int32> result_ids;
    result_ids.reserve(k);

    for (int i = 0; i < k; i++) {
        int idx = (int) labels[i];
        if (idx < 0 || idx >= n) {
            ereport(ERROR,
                    (errcode(ERRCODE_INTERNAL_ERROR),
                     errmsg("Faiss returned invalid index %d", idx)));
        }

        result_ids.push_back(ids[idx]);   // pega o id real
    }

    ArrayType *resultArr = int_vector_to_pg_array(result_ids);
    PG_RETURN_ARRAYTYPE_P(resultArr);
}

// -----------------------------------------------------------------------------
// NOVO: construir (e persistir) o índice Faiss no PostgreSQL (tabela faiss_indexes)
// -----------------------------------------------------------------------------

// SQL esperado:
//
//   SELECT faiss_build_index('meu_indice');
//
// A função lê a tabela faiss_items(id, embedding), monta um índice IndexFlatL2
// com IDMap, serializa e salva em faiss_indexes(name, dim, metric, index_data...).
extern "C" Datum faiss_build_index(PG_FUNCTION_ARGS) {
    text* indexNameText = PG_GETARG_TEXT_PP(0);
    char* indexName = text_to_cstring(indexNameText);

    if (SPI_connect() != SPI_OK_CONNECT)
        ereport(ERROR, (errmsg("SPI_connect failed")));

    const char* cmd = "SELECT id, embedding FROM faiss_items ORDER BY id";
    int ret = SPI_execute(cmd, true /* read-only */, 0 /* no limit */);
    if (ret != SPI_OK_SELECT) {
        SPI_finish();
        ereport(ERROR, (errmsg("SPI_execute failed (%d)", ret)));
    }

    uint64 nrows = SPI_processed;
    if (nrows == 0) {
        SPI_finish();
        ereport(ERROR, (errmsg("faiss_items is empty; nothing to index")));
    }

    SPITupleTable* tuptable = SPI_tuptable;
    TupleDesc tupdesc = tuptable->tupdesc;

    // Aceita id como int4 ou int8
    Oid id_type = TupleDescAttr(tupdesc, 0)->atttypid;

    // Descobrir dimensão d a partir do primeiro embedding
    bool isnull = false;
    Datum firstEmbDatum = SPI_getbinval(tuptable->vals[0], tupdesc, 2, &isnull);
    if (isnull) {
        SPI_finish();
        ereport(ERROR, (errmsg("embedding is NULL")));
    }

    vector<float> emb0 = pg_array_to_float_vector(DatumGetArrayTypeP(firstEmbDatum));
    int d = (int) emb0.size();
    if (d <= 0) {
        SPI_finish();
        ereport(ERROR, (errmsg("embedding dimension must be > 0")));
    }

    vector<float> data;
    data.reserve((size_t)nrows * (size_t)d);

    vector<faiss::idx_t> ids;
    ids.reserve((size_t)nrows);

    for (uint64 i = 0; i < nrows; i++) {
        HeapTuple tup = tuptable->vals[i];

        bool idnull = false, embnull = false;
        Datum idDatum = SPI_getbinval(tup, tupdesc, 1, &idnull);
        Datum embDatum = SPI_getbinval(tup, tupdesc, 2, &embnull);

        if (idnull || embnull) {
            SPI_finish();
            ereport(ERROR, (errmsg("id/embedding must not be NULL")));
        }

        int64 id64 = get_int64_from_datum(idDatum, id_type);

        ArrayType* eArr = DatumGetArrayTypeP(embDatum);
        vector<float> v = pg_array_to_float_vector(eArr);
        if ((int)v.size() != d) {
            SPI_finish();
            ereport(ERROR, (errmsg("all embeddings must have same dimension (%d)", d)));
        }

        ids.push_back((faiss::idx_t) id64);
        data.insert(data.end(), v.begin(), v.end());
    }

    // Monta índice: Flat L2 + IDMap2
    faiss::IndexFlatL2 base(d);
    faiss::IndexIDMap2 index(&base);
    index.add_with_ids((faiss::idx_t)nrows, data.data(), ids.data());

    // Serializa para memória (compatível com mais versões do Faiss)
    faiss::MemoryIOWriter writer;
    faiss::write_index(&index, &writer);

    bytea* indexBytea = buffer_to_bytea((const uint8_t*)writer.data.data(), writer.data.size());

    // Salva/atualiza em faiss_indexes
    const char* upsert =
        "INSERT INTO faiss_indexes(name, dim, metric, index_data, updated_at, dirty) "
        "VALUES($1, $2, 'L2', $3, now(), false) "
        "ON CONFLICT (name) DO UPDATE SET "
        "dim = EXCLUDED.dim, metric = EXCLUDED.metric, index_data = EXCLUDED.index_data, "
        "updated_at = now(), dirty = false";

    Oid argtypes[3] = { TEXTOID, INT4OID, BYTEAOID };
    Datum values[3];
    const char nulls[3] = { ' ', ' ', ' ' };

    values[0] = CStringGetTextDatum(indexName);
    values[1] = Int32GetDatum(d);
    values[2] = PointerGetDatum(indexBytea);

    ret = SPI_execute_with_args(upsert, 3, argtypes, values, nulls, false, 0);
    if (ret != SPI_OK_INSERT && ret != SPI_OK_UPDATE) {
        SPI_finish();
        ereport(ERROR, (errmsg("failed to upsert faiss_indexes (%d)", ret)));
    }

    SPI_finish();
    PG_RETURN_VOID();
}

// -----------------------------------------------------------------------------
// NOVO: busca retornando tabela (id, distance)
// -----------------------------------------------------------------------------

// SQL esperado:
//
//   SELECT * FROM faiss_search('meu_indice', ARRAY[...], 10);
//
// Retorna:
//   id       -> id real salvo com add_with_ids
//   distance -> distância L2
extern "C" Datum faiss_search(PG_FUNCTION_ARGS) {
    text* indexNameText = PG_GETARG_TEXT_PP(0);
    ArrayType* queryArr = PG_GETARG_ARRAYTYPE_P(1);
    int32 k = PG_GETARG_INT32(2);

    if (k <= 0)
        ereport(ERROR, (errmsg("k must be positive")));

    char* indexName = text_to_cstring(indexNameText);
    vector<float> query = pg_array_to_float_vector(queryArr);
    int d = (int) query.size();
    if (d <= 0)
        ereport(ERROR, (errmsg("query must not be empty")));

    if (SPI_connect() != SPI_OK_CONNECT)
        ereport(ERROR, (errmsg("SPI_connect failed")));

    const char* cmd = "SELECT dim, index_data, dirty FROM faiss_indexes WHERE name = $1";
    Oid argtypes[1] = { TEXTOID };
    Datum values[1] = { CStringGetTextDatum(indexName) };
    const char nulls1[1] = { ' ' };

    int ret = SPI_execute_with_args(cmd, 1, argtypes, values, nulls1, true, 1);
    if (ret != SPI_OK_SELECT || SPI_processed != 1) {
        SPI_finish();
        ereport(ERROR, (errmsg("index '%s' not found in faiss_indexes", indexName)));
    }

    HeapTuple tup = SPI_tuptable->vals[0];
    TupleDesc tupdesc = SPI_tuptable->tupdesc;

    bool isnull = false;

    int32 dim = DatumGetInt32(SPI_getbinval(tup, tupdesc, 1, &isnull));
    if (isnull) {
        SPI_finish();
        ereport(ERROR, (errmsg("dim is NULL")));
    }
    if (dim != d) {
        SPI_finish();
        ereport(ERROR, (errmsg("query dim (%d) != index dim (%d)", d, dim)));
    }

    bool dirty = DatumGetBool(SPI_getbinval(tup, tupdesc, 3, &isnull));
    if (!isnull && dirty) {
        SPI_finish();
        ereport(ERROR, (errmsg("index '%s' is dirty; run faiss_build_index('%s')", indexName, indexName)));
    }

    Datum idxDatum = SPI_getbinval(tup, tupdesc, 2, &isnull);
    if (isnull) {
        SPI_finish();
        ereport(ERROR, (errmsg("index_data is NULL")));
    }

    // Copia o bytea para um buffer próprio (evita ponteiros para memória do SPI)
    bytea* b = DatumGetByteaPP(idxDatum);
    Size rawlen = VARSIZE_ANY(b);
    bytea* bcopy = (bytea*) palloc(rawlen);
    memcpy(bcopy, b, rawlen);

    uint8_t* ptr = (uint8_t*) VARDATA_ANY(bcopy);
    size_t len = (size_t) VARSIZE_ANY_EXHDR(bcopy);

    // Carrega índice Faiss enquanto SPI ainda está ativo
    faiss::MemoryIOReader reader(ptr, len);
    faiss::Index* index = faiss::read_index(&reader);

    // Termina SPI cedo (o índice já está em memória)
    SPI_finish();

    if (index == nullptr) {
        ereport(ERROR, (errmsg("failed to read Faiss index")));
    }

    if (index->d != d) {
        delete index;
        ereport(ERROR, (errmsg("query dim (%d) != faiss index dim (%d)", d, (int)index->d)));
    }

    if (index->ntotal == 0) {
        delete index;
        // Retorna conjunto vazio
        ReturnSetInfo* rsinfo0 = (ReturnSetInfo*) fcinfo->resultinfo;
        if (rsinfo0 == NULL || !IsA(rsinfo0, ReturnSetInfo))
            ereport(ERROR, (errmsg("set-valued function called in a context that cannot accept a set")));

        rsinfo0->returnMode = SFRM_Materialize;

        TupleDesc outdesc0;
        if (get_call_result_type(fcinfo, NULL, &outdesc0) != TYPEFUNC_COMPOSITE)
            ereport(ERROR, (errmsg("return type must be a composite type")));

        if (rsinfo0->econtext == NULL)
            ereport(ERROR, (errmsg("no execution context available (rsinfo->econtext is NULL)")));

        MemoryContext old0 = MemoryContextSwitchTo(rsinfo0->econtext->ecxt_per_query_memory);
        Tuplestorestate* tupstore0 = tuplestore_begin_heap(true, false, work_mem);
        rsinfo0->setResult = tupstore0;
        rsinfo0->setDesc = outdesc0;
        MemoryContextSwitchTo(old0);

        delete index;
        PG_RETURN_NULL();
    }

    if (k > (int32)index->ntotal)
        k = (int32)index->ntotal;

    vector<float> distances(k);
    vector<faiss::idx_t> labels(k);
    index->search(1, query.data(), k, distances.data(), labels.data());

    ReturnSetInfo* rsinfo = (ReturnSetInfo*) fcinfo->resultinfo;
    if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo)) {
        delete index;
        ereport(ERROR, (errmsg("set-valued function called in a context that cannot accept a set")));
    }

    rsinfo->returnMode = SFRM_Materialize;

    TupleDesc outdesc;
    if (get_call_result_type(fcinfo, NULL, &outdesc) != TYPEFUNC_COMPOSITE) {
        delete index;
        ereport(ERROR, (errmsg("return type must be a composite type")));
    }

    if (rsinfo->econtext == NULL) {
        delete index;
        ereport(ERROR, (errmsg("no execution context available (rsinfo->econtext is NULL)")));
    }

    MemoryContext oldcontext = MemoryContextSwitchTo(rsinfo->econtext->ecxt_per_query_memory);
    Tuplestorestate* tupstore = tuplestore_begin_heap(true, false, work_mem);

    for (int i = 0; i < k; i++) {
        if (labels[i] < 0) continue; // Faiss usa -1 quando não há vizinho

        Datum outvals[2];
        bool outnulls[2] = { false, false };

        outvals[0] = Int64GetDatum((int64)labels[i]);
        outvals[1] = Float4GetDatum((float4)distances[i]);

        tuplestore_putvalues(tupstore, outdesc, outvals, outnulls);
    }

    rsinfo->setResult = tupstore;
    rsinfo->setDesc = outdesc;

    MemoryContextSwitchTo(oldcontext);

    delete index;
    PG_RETURN_NULL();
}
