SOURCES=test_alex.cpp ../util.c
DEPS=${SOURCES} ../util.h

FLAGS=-I./ALEX/src/core -std=c++17 -march=native -mavx -mavx2 -mbmi2 -mlzcnt -Wno-deprecated-declarations -w -Wall -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free

CXX ?= g++

all: test_alex test_alex_debug

clean:
	rm test_alex test_alex_debug

test_alex: ${DEPS} Makefile
	${CXX} -o $@ ${FLAGS} -O3 -DNDEBUG ${SOURCES} -ltbb -lpthread 

test_alex_debug: ${DEPS} Makefile
	${CXX} -o $@ ${FLAGS} -g ${SOURCES} -ltbb -lpthread 

