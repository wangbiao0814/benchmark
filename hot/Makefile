SOURCES=test_hot.cpp ../util.c
DEPS=${SOURCES} ../util.h

FLAGS=-I ./include -std=c++14 -DHOT_TEST -march=native -mavx -mavx2 -mbmi2 -mlzcnt -Wno-deprecated-declarations -Wall -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free

CXX ?= g++

all: test_hot test_hot_debug

clean:
	rm test_hot test_hot_debug

test_hot: ${DEPS} Makefile
	${CXX} -o $@ ${FLAGS} -O3 -DNDEBUG ${SOURCES} -ltbb -ljemalloc -lpthread

test_hot_debug: ${DEPS} Makefile
	${CXX} -o $@ ${FLAGS} -g ${SOURCES} -ltbb -lpthread
