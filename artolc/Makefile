ARTOLC_DEPS=artolc/Epoche.cpp artolc/N.cpp artolc/N16.cpp artolc/N256.cpp artolc/N4.cpp\
            artolc/N48.cpp artolc/Tree.cpp artolc/Epoche.cpp artolc/Epoche.h artolc/Key.h\
            artolc/N.h artolc/Tree.h ../util.h
SOURCES=test_artolc.cpp ../util.c

CXX=g++ -std=gnu++0x
CFLAGS = -g -Wno-invalid-offsetof -mcx16 -DNDEBUG

all: test_artolc test_artolc_debug 

test_artolc: ${ARTOLC_DEPS} ${SOURCES} Makefile
	${CXX} ${CFLAGS} -O3 -o $@ ${SOURCES} artolc/Tree.cpp -Iartolc -lpthread -lm -ltbb 

test_artolc_debug: ${ARTOLC_DEPS} ${SOURCES} Makefile
	${CXX} ${CFLAGS} -o $@ ${SOURCES} artolc/Tree.cpp -Iartolc -lpthread -lm -ltbb

clean:
	rm -rf test_artolc test_artolc_debug 
