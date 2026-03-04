-- faiss_pg--0.1.sql
\echo Use "CREATE EXTENSION faiss_pg" to load this file. \quit

-- Funções antigas
CREATE FUNCTION faiss_knn_l2(query real[],
                             data real[],
                             n integer,
                             d integer,
                             k integer)
RETURNS int[]
AS 'MODULE_PATHNAME', 'faiss_knn_l2'
LANGUAGE C STRICT;

CREATE FUNCTION faiss_knn_l2_table(query real[], k integer)
RETURNS int[]
AS 'MODULE_PATHNAME', 'faiss_knn_l2_table'
LANGUAGE C STRICT;

-- Tabela que guarda o índice Faiss serializado
CREATE TABLE IF NOT EXISTS faiss_indexes (
  name text PRIMARY KEY,
  dim integer NOT NULL,
  metric text NOT NULL DEFAULT 'L2',
  index_data bytea NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now(),
  dirty boolean NOT NULL DEFAULT true
);

-- (Opcional) marca índice como "sujo" quando faiss_items muda (nome padrão)
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

-- Constrói e persiste índice
CREATE FUNCTION faiss_build_index(index_name text)
RETURNS void
AS 'MODULE_PATHNAME', 'faiss_build_index'
LANGUAGE C STRICT;

-- Busca no índice persistido e retorna tabela
CREATE FUNCTION faiss_search(index_name text, query real[], k integer)
RETURNS TABLE(id bigint, distance real)
AS 'MODULE_PATHNAME', 'faiss_search'
LANGUAGE C STRICT;
