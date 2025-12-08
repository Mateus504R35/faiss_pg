# faiss_pg

Extensão experimental do PostgreSQL que integra a biblioteca [Faiss](https://github.com/facebookresearch/faiss) para busca por similaridade em vetores (k-vizinhos mais próximos).

> **Status:** protótipo / estudo de caso para iniciação científica. Ainda não possui índices persistentes nem integração com tabelas reais (usa vetores passados como `real[]`).

---

## Motivação

A ideia deste projeto é mostrar, de forma bem direta, como:

1. Escrever uma extensão nativa para PostgreSQL em C/C++;
2. Linkar essa extensão com a biblioteca Faiss;
3. Expor funções SQL para fazer busca vetorial (k-NN) diretamente via `SELECT`.

Isso serve como base para trabalhos acadêmicos (IC/TCC) e para quem quer entender como integrar um mecanismo de busca vetorial de alta performance dentro do PostgreSQL.

---

## Requisitos

- Linux / WSL (Ubuntu recomendado)
- PostgreSQL (testado com 16, mas deve funcionar em versões próximas)
- Pacotes de desenvolvimento do PostgreSQL:

  ```bash
  sudo apt install postgresql postgresql-server-dev-16
