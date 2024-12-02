SOURCES=test_veb.cpp ./veb/*.cc ../util.c
DEPS=${SOURCES} ../util.h

FLAGS=-I./include -I./veb -std=c++17 -march=native -mavx -mavx2 -mbmi2 -mlzcnt -Wno-deprecated-declarations -w -Wall -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free

CXX ?= g++

all: test_veb test_veb_debug

clean:
	rm test_veb test_veb_debug

test_veb: ${DEPS} Makefile
	${CXX} -o $@ ${FLAGS} -O3 -DNDEBUG ${SOURCES} -ltbb -lpthread

test_veb_debug: ${DEPS} Makefile
	${CXX} -o $@ ${FLAGS} -g ${SOURCES} -ltbb -lpthread
