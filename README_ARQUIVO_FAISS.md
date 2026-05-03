# faiss_pg com índice salvo em arquivo

Esta versão troca a persistência do índice Faiss de `bytea` dentro do PostgreSQL para arquivo `.faiss` no sistema de arquivos.

## Por que mudar?

No dataset CCNEWS/SISAP2025, o índice HNSW fica grande demais para ser salvo como um único `bytea`. O erro típico é:

```text
ERROR: invalid memory alloc request size 1096333395
```

Com esta versão, a tabela `faiss_indexes` guarda apenas metadados e `index_path`.

## Diretório dos índices

Crie um diretório gravável pelo usuário do PostgreSQL:

```bash
sudo mkdir -p /var/lib/postgresql/faiss_indexes
sudo chown postgres:postgres /var/lib/postgresql/faiss_indexes
sudo chmod 700 /var/lib/postgresql/faiss_indexes
```

Também é possível passar outro diretório no `params`:

```sql
'{"indexDir":"/caminho/para/faiss_indexes","M":32,"efConstruction":200,"efSearch":128}'
```

## Instalação

Copie os arquivos para o repositório:

```bash
cp faiss_pg.cpp /caminho/do/repositorio/faiss_pg.cpp
cp faiss_pg--0.1.sql /caminho/do/repositorio/faiss_pg--0.1.sql
cp faiss_pg.control /caminho/do/repositorio/faiss_pg.control
cp Makefile /caminho/do/repositorio/Makefile
```

Compile:

```bash
make clean
make NO_BC=1 with_llvm=
sudo make install NO_BC=1 with_llvm=
```

Reinicie o PostgreSQL:

```bash
sudo systemctl restart postgresql
```

ou:

```bash
sudo pg_ctlcluster 16 main restart
```

Recrie a extensão:

```sql
DROP EXTENSION IF EXISTS faiss_pg CASCADE;
CREATE EXTENSION faiss_pg;
```

## Uso com sua tabela real[]

```sql
SELECT faiss_build_index(
    'ccnews_cosine_hnsw32',
    'embeddings_train_faiss'::regclass,
    'id'::name,
    'embedding'::name,
    'cosine',
    'hnsw32',
    true,
    '{"indexDir":"/var/lib/postgresql/faiss_indexes","M":32,"efConstruction":200,"efSearch":128,"fetchBatchSize":5000}'
);
```

Busca:

```sql
SELECT *
FROM faiss_search(
    'ccnews_cosine_hnsw32',
    (SELECT embedding FROM embeddings_queries_faiss ORDER BY id LIMIT 1),
    10,
    '{"efSearch":128}'
);
```

## Observações

- O arquivo `.faiss` precisa estar acessível pelo processo do PostgreSQL.
- O cache é por backend/conexão. A primeira busca carrega o arquivo; as próximas usam o índice em memória.
- Se recriar o índice, a próxima busca recarrega porque `updated_at` muda.
