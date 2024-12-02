SOURCES=test_cuckoohash.cpp ../util.c
DEPS=${SOURCES} ../util.h

FLAGS=-I./libcuckoo/install/include -std=c++17 -march=native -mavx -mavx2 -mbmi2 -mlzcnt -Wno-deprecated-declarations -w -Wall -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free

CXX ?= g++

all: test_cuckoohash test_cuckoohash_debug

clean:
	rm test_cuckoohash test_cuckoohash_debug

test_cuckoohash: ${DEPS} Makefile
	${CXX} -o $@ ${FLAGS} -O3 -DNDEBUG ${SOURCES} -ltbb -lpthread 

test_cuckoohash_debug: ${DEPS} Makefile
	${CXX} -o $@ ${FLAGS} -g ${SOURCES} -ltbb -lpthread 

