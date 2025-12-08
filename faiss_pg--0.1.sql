-- faiss_pg--0.1.sql

\echo Use "CREATE EXTENSION faiss_pg" to load this file. \quit

-- Função: recebe
--  query  : vetor de tamanho d  (real[])
--  data   : dataset achatado com n * d elementos (real[])
--  n      : número de vetores no dataset
--  d      : dimensão de cada vetor
--  k      : número de vizinhos mais próximos
-- retorna:
--  int[] com os índices (0..n-1) dos k vizinhos mais próximos

CREATE FUNCTION faiss_knn_l2(query real[],
                             data real[],
                             n integer,
                             d integer,
                             k integer)
RETURNS int[]
AS 'MODULE_PATHNAME', 'faiss_knn_l2'
LANGUAGE C STRICT;
