EXTENSION = faiss_pg
MODULE_big = faiss_pg
OBJS = faiss_pg.o

DATA = faiss_pg--0.1.sql

PG_CONFIG = pg_config

# Evita geração do .bc com clang, que costuma falhar por causa de omp.h.
NO_BC = 1

CC = g++
CXX = g++

PG_CPPFLAGS += -I/usr/local/include
CFLAGS += -std=c++17 -fPIC
CXXFLAGS += -std=c++17 -fPIC

SHLIB_LINK += -L/usr/local/lib -lfaiss -lstdc++ -fopenmp

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
