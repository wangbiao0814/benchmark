SOURCES=test_vindex.cpp ./MemoryPool/memorypool.c ./vindex/*.cc ../util.c
SOURCESTHREADSATE = test_vindex.cpp ./threadsafe_vindex/*.cc ../util.c
DEPS=${SOURCES} ../util.h

FLAGS=-I./include -I./vindex -I./MemoryPool -std=c++17 -march=native -mavx -mavx2 -mbmi2 -mlzcnt -Wno-deprecated-declarations -w -Wall -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free

CXX ?= g++

all: test_vindex test_vindex_debug test_threadsafe_vindex test_threadsafe_vindex_debug

clean:
	rm test_vindex test_vindex_debug test_threadsafe_vindex test_threadsafe_vindex_debug

test_threadsafe_vindex: ${DEPS} Makefile
	${CXX} -o $@ ${FLAGS} -O3 -DNDEBUG ${SOURCESTHREADSATE} -ljemalloc -lpthread

test_threadsafe_vindex_debug: ${DEPS} Makefile
	${CXX} -o $@ ${FLAGS} -g ${SOURCESTHREADSATE}  -lpthread 

test_vindex: ${DEPS} Makefile
	${CXX} -o $@ ${FLAGS} -O3 -DNDEBUG ${SOURCES} -lpthread 

test_vindex_debug: ${DEPS} Makefile
	${CXX} -o $@ ${FLAGS} -g ${SOURCES} -lpthread 
