#ifndef _VEB_H_
#define _VEB_H_


#define MAX_KEY_LENGTH 16

#include <cstdint>
#include <vector>
#include <string>
#include <gtl/phmap.hpp>
#include "xxhash64.h"

static uint8_t byte_count[256] = {
                                    0,1,2,3,1,2,3,4,2,3,4,5,3,4,5,6,
                                    1,2,3,4,2,3,4,5,3,4,5,6,4,5,6,7,
                                    2,3,4,5,3,4,5,6,4,5,6,7,5,6,7,8,
                                    3,4,5,6,4,5,6,7,5,6,7,8,6,7,8,9,
                                    1,2,3,4,2,3,4,5,3,4,5,6,4,5,6,7,
                                    2,3,4,5,3,4,5,6,4,5,6,7,5,6,7,8,
                                    3,4,5,6,4,5,6,7,5,6,7,8,6,7,8,9,
                                    4,5,6,7,5,6,7,8,6,7,8,9,7,8,9,10,
                                    2,3,4,5,3,4,5,6,4,5,6,7,5,6,7,8,
                                    3,4,5,6,4,5,6,7,5,6,7,8,6,7,8,9,
                                    4,5,6,7,5,6,7,8,6,7,8,9,7,8,9,10,
                                    5,6,7,8,6,7,8,9,7,8,9,10,8,9,10,11,
                                    3,4,5,6,4,5,6,7,5,6,7,8,6,7,8,9,
                                    4,5,6,7,5,6,7,8,6,7,8,9,7,8,9,10,
                                    5,6,7,8,6,7,8,9,7,8,9,10,8,9,10,11,
                                    6,7,8,9,7,8,9,10,8,9,10,11,9,10,11,12};


inline uint8_t ctz_16(uint16_t x)
{
    uint8_t n = 1;
    if((x & 0xFF) == 0) {n += 8; x >>= 8;}
    if((x & 0x0F) == 0) {n += 4; x >>= 4;}
    if((x & 0x03) == 0) {n += 2; x >>= 2;}
    return n - (x & 1);
}

inline uint8_t ctz_32(uint32_t x)
{
    uint8_t n = 1;
    if((x & 0xFFFF) == 0) {n += 16; x >>= 16;}
    if((x & 0x00FF) == 0) {n += 8; x >>= 8;}
    if((x & 0x000F) == 0) {n += 4; x >>= 4;}
    if((x & 0x0003) == 0) {n += 2; x >>= 2;}
    return n - (x & 1);
}

inline uint8_t ctz_64(uint64_t x)
{
    uint8_t n = 1;
    if((x & 0xFFFFFFFF) == 0) {n += 32; x >>= 32;}
    if((x & 0x0000FFFF) == 0) {n += 16; x >>= 16;}
    if((x & 0x000000FF) == 0) {n += 8; x >>= 8;}
    if((x & 0x0000000F) == 0) {n += 4; x >>= 4;}
    if((x & 0x00000003) == 0) {n += 2; x >>= 2;}
    return n - (x & 1);
}

inline bool isUpperLevel(int level)
{
    return (level & 1) ? false : true;
}



/*
extern "C"
{
#include <sys/mman.h> // madvise, mmap
#include <nmmintrin.h> // _mm_crc32_u64
}
template <typename T> struct huge_page_allocator {
  constexpr static std::size_t huge_page_size = 1 << 21; // 2 MiB
  using value_type = T;

  huge_page_allocator() = default;
  template <class U>
  constexpr huge_page_allocator(const huge_page_allocator<U> &) noexcept {}

  size_t round_to_huge_page_size(size_t n) {
    return (((n - 1) / huge_page_size) + 1) * huge_page_size;
  }

  T *allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::bad_alloc();
    }
    auto p = static_cast<T *>(mmap(
        nullptr, round_to_huge_page_size(n * sizeof(T)), PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0));
    if (p == MAP_FAILED) {
      throw std::bad_alloc();
    }
    return p;
  }

  void deallocate(T *p, std::size_t n) {
    munmap(p, round_to_huge_page_size(n));
  }
};

struct hash {
    size_t operator()(size_t h) const noexcept { return _mm_crc32_u64(0, h); }
};

typedef gtl::flat_hash_map<uint64_t, void*, std::hash<uint64_t>, std::equal_to<uint64_t>, huge_page_allocator<size_t>> HashTableRoot;
*/




typedef struct 
{
    int keyLength;
    int valueLength;
    uint8_t kv[];
}vEBKV;

typedef struct _vEBHashKey
{
    bool operator==(const _vEBHashKey &o) const
    { 
        return !memcmp((char*)key, (char*)o.key, end);
    }

    friend size_t hash_value(const _vEBHashKey &p)
    {
        return XXHash64::hash(p.key, p.end, 0xBADBEAF);
    }
    uint8_t key[MAX_KEY_LENGTH - 2];
    uint16_t end;
}vEBHashKey;

typedef gtl::flat_hash_map<vEBHashKey, void*> HashTableRoot;


typedef struct 
{
    void** levelHash;
    uint64_t maxKeyLength;
}vEB;

typedef struct 
{
    void* directPointer[1 << 16];
}vEBDirectPointer;


typedef struct
{
    uint64_t* vNodeKeysDistribution;
}vEBStatistics;



typedef struct 
{
    uint64_t currentLevel;
    void** levelvNodes;
    vEBKV* kv;   
    vEBKV** kvBuf;
}vEBIterator;


inline vEBKV* setLeafTag(vEBKV* leaf, uint16_t tag)
{
    return ((vEBKV*)((uint64_t)leaf | (((uint64_t)tag) << 48)));
}

inline vEBKV* getLeaf(vEBKV* leafTag)
{
    vEBKV* temp = ((vEBKV*)((uint64_t)leafTag & ((((uint64_t)1) << 48) - 1)));
    return ((vEBKV*)((uint64_t)leafTag & ((((uint64_t)1) << 48) - 1)));
}

inline uint16_t getTag(vEBKV* leafTag)
{
    return ((uint64_t)leafTag >> 48); 
}



void InitvEB(vEB* root, int maxKeyLength);
void InsertvEB(vEB* root, vEBKV* kv);
void* GetvEB(vEB* root, vEBKV* kv);

void CollectvEB(vEB* root, vEBStatistics* vs);

#endif