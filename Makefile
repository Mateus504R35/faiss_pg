EXTENSION = faiss_pg
MODULE_big = faiss_pg
OBJS = faiss_pg.o

# MUITO IMPORTANTE: listar o script SQL aqui
DATA = faiss_pg--0.1.sql

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

# Compilar como C++
CC = g++
CFLAGS += -std=c++17

# Onde estão os headers do Faiss (ajuste se necessário)
PG_CPPFLAGS += -I/usr/local/include

# Linkar com a biblioteca libfaiss.so (ajuste se necessário)
SHLIB_LINK += -lfaiss
