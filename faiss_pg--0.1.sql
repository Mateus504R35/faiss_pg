-- faiss_pg--0.1.sql
\echo Use "CREATE EXTENSION faiss_pg" to load this file. \quit

CREATE FUNCTION faiss_knn_l2(query real[], data real[], n integer, d integer, k integer)
RETURNS int[]
AS 'MODULE_PATHNAME', 'faiss_knn_l2'
LANGUAGE C STRICT;

CREATE FUNCTION faiss_knn_l2_table(query real[], k integer)
RETURNS int[]
AS 'MODULE_PATHNAME', 'faiss_knn_l2_table'
LANGUAGE C STRICT;

-- Agora o índice Faiss fica no sistema de arquivos.
-- O PostgreSQL guarda somente metadados e o caminho do arquivo .faiss.
CREATE TABLE IF NOT EXISTS faiss_indexes (
  name text PRIMARY KEY,
  dim integer NOT NULL,
  metric text NOT NULL DEFAULT 'cosine',
  index_type text NOT NULL DEFAULT 'flat',
  normalize_vectors boolean NOT NULL DEFAULT true,
  ntotal bigint NOT NULL DEFAULT 0,
  index_path text NOT NULL,
  params text NOT NULL DEFAULT '{}',
  updated_at timestamptz NOT NULL DEFAULT now(),
  dirty boolean NOT NULL DEFAULT false
);

CREATE OR REPLACE FUNCTION faiss_mark_dirty()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  UPDATE faiss_indexes SET dirty = true WHERE name = 'faiss_items_l2';
  RETURN NEW;
END;
$$;

DO $$
BEGIN
  IF to_regclass('public.faiss_items') IS NOT NULL THEN
    DROP TRIGGER IF EXISTS trg_faiss_items_dirty ON faiss_items;
    CREATE TRIGGER trg_faiss_items_dirty
    AFTER INSERT OR UPDATE OR DELETE ON faiss_items
    FOR EACH STATEMENT
    EXECUTE FUNCTION faiss_mark_dirty();
  END IF;
END;
$$;

-- Compatibilidade com a versão antiga: indexa faiss_items(id, embedding) como flat/l2.
CREATE FUNCTION faiss_build_index(index_name text)
RETURNS void
AS 'MODULE_PATHNAME', 'faiss_build_index'
LANGUAGE C STRICT;

-- Nova versão parametrizável.
-- params pode conter, por exemplo:
--   '{"indexDir":"/var/lib/postgresql/faiss_indexes","M":32,"efConstruction":200,"efSearch":128}'
--   '{"indexDir":"/var/lib/postgresql/faiss_indexes","nlist":4096,"nprobe":64}'
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

CREATE FUNCTION faiss_search(index_name text, query real[], k integer)
RETURNS TABLE(id bigint, distance real)
AS 'MODULE_PATHNAME', 'faiss_search'
LANGUAGE C STRICT;

CREATE FUNCTION faiss_search(index_name text, query real[], k integer, search_params text)
RETURNS TABLE(id bigint, distance real)
AS 'MODULE_PATHNAME', 'faiss_search'
LANGUAGE C STRICT;

CREATE FUNCTION faiss_clear_cache(index_name text)
RETURNS void
AS 'MODULE_PATHNAME', 'faiss_clear_cache'
LANGUAGE C STRICT;

CREATE FUNCTION faiss_clear_all_cache()
RETURNS void
AS 'MODULE_PATHNAME', 'faiss_clear_all_cache'
LANGUAGE C;
