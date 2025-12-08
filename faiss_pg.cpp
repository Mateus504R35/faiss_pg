// faiss_pg.cpp

// Parte C (exposta ao PostgreSQL)
extern "C" {
    #include "postgres.h"
    #include "fmgr.h"
    #include "utils/array.h"
    #include "utils/lsyscache.h"   // get_typlenbyvalalign
    #include "catalog/pg_type.h"   // FLOAT4OID, INT4OID

    PG_MODULE_MAGIC;

    PG_FUNCTION_INFO_V1(faiss_knn_l2);
    Datum faiss_knn_l2(PG_FUNCTION_ARGS);
}

// Parte C++ / Faiss
#include <vector>
#include <stdexcept>

#include <faiss/IndexFlat.h> // índice exato L2

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

// query: real[]  (tamanho d)
// data : real[]  (tamanho n * d, achatado)
// n    : nº de vetores
// d    : dimensão
// k    : top-k
Datum faiss_knn_l2(PG_FUNCTION_ARGS) {
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
