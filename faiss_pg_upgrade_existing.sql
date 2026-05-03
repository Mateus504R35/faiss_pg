-- faiss_pg_upgrade_existing.sql
-- Use este arquivo apenas se você NÃO quiser fazer DROP EXTENSION faiss_pg CASCADE.
-- Ele adapta a tabela faiss_indexes antiga e recria as funções SQL apontando para o novo .so.

ALTER TABLE faiss_indexes ADD COLUMN IF NOT EXISTS index_type text NOT NULL DEFAULT 'flat';
ALTER TABLE faiss_indexes ADD COLUMN IF NOT EXISTS normalize_vectors boolean NOT NULL DEFAULT false;
ALTER TABLE faiss_indexes ADD COLUMN IF NOT EXISTS ntotal bigint NOT NULL DEFAULT 0;
ALTER TABLE faiss_indexes ADD COLUMN IF NOT EXISTS params text NOT NULL DEFAULT '{}';

DROP FUNCTION IF EXISTS faiss_build_index(text, regclass, name, name, text, text, boolean, text);
DROP FUNCTION IF EXISTS faiss_search(text, real[], integer, text);
DROP FUNCTION IF EXISTS faiss_search_batch(text, real[], integer, integer);
DROP FUNCTION IF EXISTS faiss_search_batch(text, real[], integer, integer, text);
DROP FUNCTION IF EXISTS faiss_clear_cache(text);
DROP FUNCTION IF EXISTS faiss_clear_all_cache();

CREATE OR REPLACE FUNCTION faiss_build_index(index_name text)
RETURNS void
AS 'MODULE_PATHNAME', 'faiss_build_index'
LANGUAGE C STRICT;

CREATE FUNCTION faiss_build_index(index_name text,
                                  table_name regclass,
                                  id_column name,
                                  embedding_column name,
                                  metric text,
                                  index_type text,
                                  normalize_vectors boolean,
                                  params text)
RETURNS void
AS 'MODULE_PATHNAME', 'faiss_build_index'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION faiss_search(index_name text, query real[], k integer)
RETURNS TABLE(id bigint, distance real)
AS 'MODULE_PATHNAME', 'faiss_search'
LANGUAGE C STRICT;

CREATE FUNCTION faiss_search(index_name text, query real[], k integer, search_params text)
RETURNS TABLE(id bigint, distance real)
AS 'MODULE_PATHNAME', 'faiss_search'
LANGUAGE C STRICT;

CREATE FUNCTION faiss_search_batch(index_name text, queries real[], nq integer, k integer)
RETURNS TABLE(query_no integer, id bigint, distance real)
AS 'MODULE_PATHNAME', 'faiss_search_batch'
LANGUAGE C STRICT;

CREATE FUNCTION faiss_search_batch(index_name text, queries real[], nq integer, k integer, search_params text)
RETURNS TABLE(query_no integer, id bigint, distance real)
AS 'MODULE_PATHNAME', 'faiss_search_batch'
LANGUAGE C STRICT;

CREATE FUNCTION faiss_clear_cache(index_name text)
RETURNS void
AS 'MODULE_PATHNAME', 'faiss_clear_cache'
LANGUAGE C STRICT;

CREATE FUNCTION faiss_clear_all_cache()
RETURNS void
AS 'MODULE_PATHNAME', 'faiss_clear_all_cache'
LANGUAGE C;
