extern "C"{
#include <stdio.h>
#include <time.h>
#include <signal.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
}

#include "xxhash64.h"
#include "../util.h"
#include "libcuckoo/cuckoohash_map.hh"


typedef struct 
{
    int keyLength;
    int valueLength;
    uint8_t kv[];
}cuckooKV;

struct cuckooKVHash
{
public:
	size_t operator()(const cuckooKV* p) const
	{
		return XXHash64::hash(p->kv, p->keyLength, 0xBADBEAF);
	}
};
 
struct cuckooKVEqual
{
public:
	bool operator()(const cuckooKV* p1, const cuckooKV* p2) const
	{
		if(p1->keyLength != p2->keyLength) return false;
		return !memcmp((char*)p1->kv, (char*)p2->kv, p1->keyLength);
	}
};

#ifdef USE_HUGE_PAGES
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


typedef libcuckoo::cuckoohash_map<cuckooKV*, void*, cuckooKVHash, cuckooKVEqual, huge_page_allocator<std::pair<cuckooKV*, void*>>> CuckooHashTable; 
#else
typedef libcuckoo::cuckoohash_map<cuckooKV*, void*, cuckooKVHash, cuckooKVEqual> CuckooHashTable; 
#endif




#define DEFAULT_VALUE_SIZE 8

#define MILLION 1000000
#define DEFAULT_NUM_KEYS (10 * MILLION)
#define DEFAULT_NUM_THREADS 4

#define PID_NO_PROFILER 0

static int maxKeyLength = 0;

pid_t profiler_pid = PID_NO_PROFILER;

// Notify the profiler that the critical section starts, so it should start collecting statistics
void notify_critical_section_start() {
	if (profiler_pid != PID_NO_PROFILER)
		kill(profiler_pid, SIGUSR1);
}

void notify_critical_section_end() {
	if (profiler_pid != PID_NO_PROFILER)
		kill(profiler_pid, SIGUSR1);
}

cuckooKV** read_kvs(dataset_t* dataset, uint64_t value_size) {
	uint64_t i;
	ct_key key;
	dynamic_buffer_t kvs_buf;
	uint8_t* key_buf = (uint8_t*) malloc(MAX_KEY_SIZE);
	uintptr_t* kv_ptrs = (uintptr_t*) malloc(sizeof(struct kv*) * dataset->num_keys);
	key.bytes = key_buf;

	dynamic_buffer_init(&kvs_buf);
	for (i = 0;i < dataset->num_keys;i++) {
		dataset->read_key(dataset, &key);

		uint64_t pos = dynamic_buffer_extend(&kvs_buf, sizeof(cuckooKV) + key.size + value_size);
		cuckooKV* kv = (cuckooKV*) (kvs_buf.ptr + pos);

		kv->keyLength = key.size;
		kv->valueLength = value_size;
		memcpy(kv->kv, key.bytes, key.size);

		memset(kv->kv + kv->keyLength, 0xAB, kv->valueLength);
		kv_ptrs[i] = pos;

		maxKeyLength = kv->keyLength > maxKeyLength ? kv->keyLength : maxKeyLength;
	}

	for (i = 0;i < dataset->num_keys;i++)
		kv_ptrs[i] += (uintptr_t)(kvs_buf.ptr);

	return (cuckooKV**) kv_ptrs;
}


void mem_usage(char* dataset_name) {
	dataset_t dataset;
	int result;
	uint64_t i;
	uint64_t start_mem, end_mem;
	cuckooKV** kv_ptrs;
	uint64_t index_overhead;
	uint64_t keys_size = 0;


	// seed_and_print();
	rand_seed(0);
	result = init_dataset(&dataset, dataset_name, DATASET_ALL_KEYS);
	if (!result) {
		printf("Error creating dataset.\n");
		return;
	}
	kv_ptrs = read_kvs(&dataset, DEFAULT_VALUE_SIZE);

	CuckooHashTable index;
	start_mem = virt_mem_usage();
	for (i = 0;i < dataset.num_keys;i++) {
		index.insert(kv_ptrs[i], kv_ptrs[i]);
	}
	end_mem = virt_mem_usage();
	index_overhead = end_mem - start_mem;

	printf("Keys size: %luKB (%.1fb/key)\n", keys_size / 1024, ((float)keys_size) / dataset.num_keys);
	printf("Index size: %luKB (%.1fb/key)\n", index_overhead / 1024, ((float)index_overhead) / dataset.num_keys);
	printf("RESULT: keys=%lu bytes=%lu\n", dataset.num_keys, index_overhead);


}

void pos_lookup(dataset_t* dataset) {
	const uint64_t num_lookups = 10 * MILLION;
	uint64_t i;
	struct timespec start_time;
	struct timespec end_time;

	cuckooKV** kv_ptrs;
	dynamic_buffer_t workloads_buf;

	kv_ptrs = read_kvs(dataset, DEFAULT_VALUE_SIZE);

	CuckooHashTable index;

	printf("Loading...\n");
	notify_critical_section_start();
	clock_gettime(CLOCK_MONOTONIC, &start_time);
	for (i = 0;i < dataset->num_keys;i++)
		index.insert(kv_ptrs[i], kv_ptrs[i]);
	clock_gettime(CLOCK_MONOTONIC, &end_time);
	notify_critical_section_end();
	float time_took_insert = time_diff(&end_time, &start_time);
	printf("Took %.2fs (%.0fns/key)\n", time_took_insert, time_took_insert / dataset->num_keys * 1.0e9);
	printf("RESULT: ops=%lu ms=%d\n", dataset->num_keys, (int)(time_took_insert * 1000));


	seed_and_print();
	printf("Creating workload...\n");
	dynamic_buffer_init(&workloads_buf);
	for (i = 0;i < num_lookups;i++) {
		cuckooKV* tmpKV = (kv_ptrs[rand_uint64() % dataset->num_keys]);
		uint64_t data_size = sizeof(cuckooKV) + tmpKV->keyLength;
		uint64_t offset = dynamic_buffer_extend(&workloads_buf, data_size);
		cuckooKV* data = (cuckooKV*) (workloads_buf.ptr + offset);
		data->keyLength = tmpKV->keyLength;
		memcpy(data->kv, tmpKV->kv, tmpKV->keyLength);
	}

	printf("Performing lookups...\n");
	uint8_t* buf_pos = workloads_buf.ptr;
	notify_critical_section_start();
	clock_gettime(CLOCK_MONOTONIC, &start_time);
	for (i = 0;i < num_lookups;i++) {
		cuckooKV* targetKey = (cuckooKV*)buf_pos;
		auto result = index.find(targetKey);
		if(result == NULL)
		{
			printf("Error: key not found\n");
			return;	
		}
		buf_pos += sizeof(cuckooKV) + targetKey->keyLength;
		speculation_barrier();
	}
	clock_gettime(CLOCK_MONOTONIC, &end_time);
	notify_critical_section_end();

	float time_took = time_diff(&end_time, &start_time);
	printf("Took %.2fs (%.0fns/key)\n", time_took, time_took / num_lookups * 1.0e9);
	printf("RESULT: ops=%lu ms=%d\n", num_lookups, (int)(time_took * 1000));
}


typedef struct {
	uint64_t num_keys;
	uint8_t* keys;
	CuckooHashTable* vindex;
} mt_lookup_ctx;

void* mt_lookup_thread(void* arg) {
	mt_lookup_ctx* ctx = (mt_lookup_ctx*) arg;
	uint64_t i;
	uint8_t* buf_pos = (uint8_t*)ctx->keys;
	for (i = 0; i < ctx->num_keys; i++) {
		cuckooKV* targetKey = (cuckooKV*)buf_pos;
		auto result = ctx->vindex->find(targetKey);
		if(result == NULL)
		{
			printf("Error: key not found\n");
			return NULL;	
		}
		buf_pos += (sizeof(cuckooKV) + targetKey->keyLength);
		speculation_barrier();
	}

	return NULL;
}

void mt_pos_lookup(char* dataset_name, unsigned int num_threads) {
	const uint64_t lookups_per_thread = 10 * MILLION;
	uint64_t i,j;
	int result;
	cuckooKV** keys;
	dataset_t dataset;
	

	struct timespec start_time;
	struct timespec end_time;
	dynamic_buffer_t workload_data;
	mt_lookup_ctx thread_contexts[num_threads];
	uint64_t total_lookups = num_threads * lookups_per_thread;
	
	seed_and_print();

	result = init_dataset(&dataset, dataset_name, DATASET_ALL_KEYS);
	if (!result) {
		printf("Error creating dataset.\n");
		return;
	}
	keys = read_kvs(&dataset, DEFAULT_VALUE_SIZE);

	CuckooHashTable vindex;
	printf("Loading...\n");
	for (i = 0;i < dataset.num_keys;i++)
		vindex.insert(keys[i], keys[i]);

	printf("Creating workload...\n");
	dynamic_buffer_init(&workload_data);
	for (i = 0; i < num_threads; i++) {
		thread_contexts[i].vindex = &vindex;
		thread_contexts[i].num_keys = lookups_per_thread;
		thread_contexts[i].keys = (uint8_t*) workload_data.pos;

		for (j = 0; j < lookups_per_thread; j++) {
			cuckooKV* kv = keys[rand_uint64() % dataset.num_keys];
			uint64_t data_size = sizeof(cuckooKV) + kv->keyLength;
			uint64_t offset = dynamic_buffer_extend(&workload_data, data_size);
			cuckooKV* data = (cuckooKV*) (workload_data.ptr + offset);
			data->keyLength = kv->keyLength;
			memcpy(data->kv, kv->kv, kv->keyLength);
		}
	}

	for (i = 0; i < num_threads; i++)
		thread_contexts[i].keys += (uintptr_t) workload_data.ptr;

	printf("Performing lookups...\n");
	notify_critical_section_start();
	clock_gettime(CLOCK_MONOTONIC, &start_time);
	run_multiple_threads(mt_lookup_thread, num_threads, thread_contexts, sizeof(mt_lookup_ctx));
	clock_gettime(CLOCK_MONOTONIC, &end_time);
	notify_critical_section_end();

	float time_took = time_diff(&end_time, &start_time);
	report_mt(time_took, lookups_per_thread * num_threads, num_threads);
}

typedef struct {
	int threadid;
	uint64_t num_keys;
	uint8_t* keys;
	CuckooHashTable* vindex;
} mt_insert_ctx;

void* mt_insert_thread(void* arg) {
	mt_insert_ctx* ctx = (mt_insert_ctx*) arg;
	uint64_t i;
	bind_core(ctx->threadid);
	uint8_t* buf_pos = ctx->keys;
	for (i = 0; i < ctx->num_keys; i++) {
		cuckooKV* kv = (cuckooKV*)buf_pos;
		// InsertvIndex(ctx->vindex, kv);
		ctx->vindex->insert(kv, kv);
		buf_pos += (sizeof(cuckooKV) + kv->keyLength + kv->valueLength);
		speculation_barrier();
	}

	return NULL;
}

void mt_insert(char* dataset_name, unsigned int num_threads) {
	uint64_t i;
	int result;
	cuckooKV** keys;
	dataset_t dataset;
	struct timespec start_time;
	struct timespec end_time;
	pthread_t thread_ids[num_threads];
	mt_insert_ctx thread_contexts[num_threads];

	printf("Reading dataset...\n");
	init_dataset(&dataset, dataset_name, DATASET_ALL_KEYS);
	
	keys = read_kvs(&dataset, DEFAULT_VALUE_SIZE);

	CuckooHashTable index;

	for (i = 0;i < num_threads;i++) {
		uint64_t first_kv = dataset.num_keys * i / num_threads;
		uint64_t last_kv = dataset.num_keys * (i+1) / num_threads;

		thread_contexts[i].keys = (uint8_t*) keys[first_kv];
		thread_contexts[i].num_keys = last_kv - first_kv;
		thread_contexts[i].vindex = &index;
		thread_contexts[i].threadid = i + 1;
	}

	printf("Inserting...\n");
	notify_critical_section_start();
	clock_gettime(CLOCK_MONOTONIC, &start_time);
	for (i = 0; i < num_threads; i++) {
		result = pthread_create(&(thread_ids[i]), NULL, mt_insert_thread, &(thread_contexts[i]));
		if (result != 0) {
			printf("Failed to cerate thread\n");
			return;
		}
	}

	for (i = 0; i < num_threads; i++) {
		result = pthread_join(thread_ids[i], NULL);
		if (result != 0) {
			printf("Failed to join thread\n");
			return;
		}
	}
	clock_gettime(CLOCK_MONOTONIC, &end_time);
	float time_took = time_diff(&end_time, &start_time);
	report_mt(time_took, dataset.num_keys, num_threads);
}



const ycsb_workload_spec YCSB_A_SPEC = {{0.5,  0,    0.5,  0,    0,    0  }, 10 * MILLION, DIST_ZIPF};
const ycsb_workload_spec YCSB_B_SPEC = {{0.95, 0,    0.05, 0,    0,    0  }, 10 * MILLION, DIST_ZIPF};
const ycsb_workload_spec YCSB_C_SPEC = {{1.0,  0,    0,    0,    0,    0  }, 10 * MILLION, DIST_ZIPF};
const ycsb_workload_spec YCSB_D_SPEC = {{0,    0.95, 0,    0.05, 0,    0  }, 10 * MILLION, DIST_ZIPF};
const ycsb_workload_spec YCSB_E_SPEC = {{0,    0,    0,    0.05, 0.95, 0  }, 2  * MILLION, DIST_ZIPF};
const ycsb_workload_spec YCSB_F_SPEC = {{0.5,  0,    0,    0,    0,    0.5}, 10 * MILLION, DIST_ZIPF};

const ycsb_workload_spec READ_HEAVY_SPEC = {{0.75,  0,    0,    0.25,    0,    0}, 10 * MILLION, DIST_UNIFORM};
const ycsb_workload_spec READ_WRITE_BALANCED_SPEC = {{0.5,  0,    0,    0.5,    0,    0}, 10 * MILLION, DIST_UNIFORM};
const ycsb_workload_spec WRITE_HEAVY_SPEC = {{0.25,  0,    0,    0.75,    0,    0}, 10 * MILLION, DIST_UNIFORM};

const ycsb_workload_spec WRITE_ONLY_SPEC = {{0,  0,    0,    1.0,    0,    0}, 10 * MILLION, DIST_UNIFORM};


typedef struct ycsb_thread_ctx_struct {
	void* index;  // Either mt_kv_hot_t* or kv_hot_t*
	uint64_t thread_id;
	uint64_t num_threads;
	uint64_t inserts_done;
	struct ycsb_thread_ctx_struct* thread_contexts;
	ycsb_workload workload;
} ycsb_thread_ctx;

template<typename IndexType>
void execute_ycsb_workload(ycsb_thread_ctx* ctx) {
	uint64_t i, j;
	uint64_t inserter_idx;
	uint64_t total_read_latest = 0;
	uint64_t failed_read_latest = 0;
	uint64_t read_latest_from_thread = 0;
	cuckooKV* range_results[100];
	int num_range_results;
	ycsb_thread_ctx* inserter;
	IndexType index = (IndexType) (ctx->index); 

	uint64_t last_inserts_done[ctx->num_threads];
	uint8_t* next_read_latest_key[ctx->num_threads];
	uint8_t** thread_read_latest_blocks[ctx->num_threads];

	for (i = 0;i < ctx->num_threads;i++) {
		last_inserts_done[i] = 0;
		thread_read_latest_blocks[i] = ctx->thread_contexts[i].workload.read_latest_blocks_for_thread[ctx->thread_id];
		next_read_latest_key[i] = thread_read_latest_blocks[i][0];
	}

	for (i = 0;i < ctx->workload.num_ops; i++) {
		ycsb_op* op = &(ctx->workload.ops[i]);
		switch (op->type) {
			case YCSB_READ:{
				cuckooKV* key = (cuckooKV*) (ctx->workload.data_buf + op->data_pos);
				auto result = index->find(key);
				if(result == NULL)
				{
					printf("Error: key not found\n");
					return;	
				}
				speculation_barrier();
			}
			break;

			case YCSB_READ_LATEST:{
				total_read_latest++;
				inserter_idx = read_latest_from_thread;

				cuckooKV* key = (cuckooKV*) next_read_latest_key[inserter_idx];
				

				// Advancing next_read_latest_key must be done before checking whether to
				// move to another block (by comparing inserts_done). Otherwise, in the
				// single-threaded case, we'll advance next_read_latest_key[0] after it was
				// set to the block start, and by an incorrect amount.
				if (key->keyLength != 0xFFFFFFFFU)
					next_read_latest_key[inserter_idx] += sizeof(cuckooKV) + key->keyLength;

				read_latest_from_thread++;
				if (read_latest_from_thread == ctx->num_threads)
					read_latest_from_thread = 0;

				inserter = &(ctx->thread_contexts[read_latest_from_thread]);
				uint64_t inserts_done = __atomic_load_n(&(inserter->inserts_done), __ATOMIC_RELAXED);
				if (inserts_done != last_inserts_done[read_latest_from_thread]) {
					last_inserts_done[read_latest_from_thread] = inserts_done;

					uint8_t* block_start = thread_read_latest_blocks[read_latest_from_thread][inserts_done];
					next_read_latest_key[read_latest_from_thread] = block_start;
					__builtin_prefetch(&(thread_read_latest_blocks[read_latest_from_thread][inserts_done+8]));
				}
				__builtin_prefetch(next_read_latest_key[read_latest_from_thread]);

				if (key->keyLength == 0xFFFFFFFFU) {
					// Reached end-of-block sentinel
					failed_read_latest++;
					break;
				}

				auto result = index->find(key);
				if (result == NULL) {
					printf("Error: key not found\n");
					return;
				}
				speculation_barrier();
			}
			break;

			case YCSB_UPDATE:{
				cuckooKV* updated_kv = (cuckooKV*) (ctx->workload.data_buf + op->data_pos);
				index->insert(updated_kv, updated_kv);
				speculation_barrier();
			}
			break;

			case YCSB_INSERT:{
				cuckooKV* kv = (cuckooKV*) (ctx->workload.data_buf + op->data_pos);
				index->insert(kv, kv);

				// Use atomic_store to make sure that the write isn't reordered with ct_insert,
				// and eventually becomes visible to other threads.
				__atomic_store_n(&(ctx->inserts_done), ctx->inserts_done + 1, __ATOMIC_RELEASE);
				speculation_barrier();
			}
			break;

			case YCSB_RMW:{
				cuckooKV* kv = (cuckooKV*) (ctx->workload.data_buf + op->data_pos);
				// Find existing value
				auto result = index->find(kv);
				if (result == NULL) {
					printf("Error: a key was not found\n");
					return;
				}

				// Insert the new value
				index->insert(kv, kv);
				speculation_barrier();
			}
			break;

			case YCSB_SCAN:{
				// cuckooKV* kv = (cuckooKV*) (ctx->workload.data_buf + op->data_pos);
				// uint64_t range_size = (rand_dword() % 100) + 1; num_range_results = 0;
				// RangevIndex(index, kv, range_size, range_results, &num_range_results);
				// uint64_t checksum = 0;
				// for (j = 0; j < num_range_results; j++)
				// 	checksum += (uint64_t)(range_results[j]);

				// if (checksum == ((uint64_t)-1ULL))
				// 	printf("Impossible!\n");
				speculation_barrier();
			}
			break;

			default:
				abort();
		}
	}

	if (failed_read_latest > 0) {
		printf("Note: %lu / %lu (%.1f%%) of read-latest operations were skipped\n",
			failed_read_latest, total_read_latest,
			((float)failed_read_latest) / total_read_latest * 100.0);
	}
}

void generate_ycsb_workload(dataset_t* dataset, cuckooKV** kvs, ycsb_workload* workload,
						   const ycsb_workload_spec* spec, int thread_id,
						   int num_threads) {
	uint64_t i;
	int data_size;
	cuckooKV* kv;
	uint64_t num_inserts = 0;
	uint64_t insert_offset;
	uint64_t inserts_per_thread;
	uint64_t read_latest_block_size;
	dynamic_buffer_t workload_buf;
	rand_distribution dist;
	rand_distribution backward_dist;

	workload->ops = (ycsb_op*) malloc(sizeof(ycsb_op) * spec->num_ops);
	workload->num_ops = spec->num_ops;

	inserts_per_thread = spec->op_type_probs[YCSB_INSERT] * spec->num_ops;
	workload->initial_num_keys = dataset->num_keys - inserts_per_thread * num_threads;
	insert_offset = workload->initial_num_keys + inserts_per_thread * thread_id;

	// //shuffle for read/write ops
	// std::random_shuffle(kvs + insert_offset, kvs + insert_offset + inserts_per_thread);
	

	read_latest_block_size = spec_read_latest_block_size(spec, num_threads);

	if (spec->distribution == DIST_UNIFORM) {
		rand_uniform_init(&dist, workload->initial_num_keys);
	} else if (spec->distribution == DIST_ZIPF) {
		rand_zipf_init(&dist, workload->initial_num_keys, YCSB_SKEW);
	} else {
		printf("Error: Unknown YCSB distribution\n");
		return;
	}

	if (spec->op_type_probs[YCSB_READ_LATEST] > 0.0) {
		// spec->distribution is meaningless for read-latest. Read offsets for read-latest are
		// always Zipf-distributed.
		assert(spec->distribution == DIST_ZIPF);
		rand_zipf_rank_init(&backward_dist, workload->initial_num_keys, YCSB_SKEW);
	}

	dynamic_buffer_init(&workload_buf);
	for (i = 0; i < spec->num_ops; i++) {
		ycsb_op* op = &(workload->ops[i]);
		op->type = choose_ycsb_op_type(spec->op_type_probs);

		if (num_inserts == inserts_per_thread && op->type == YCSB_INSERT) {
			// Used all keys intended for insertion. Do another op type.
			i--;
			continue;
		}

		switch (op->type) {
			case YCSB_SCAN:
			case YCSB_READ:{
				kv = kvs[rand_dist(&dist)];
				data_size = sizeof(cuckooKV) + kv->keyLength;
				op->data_pos = dynamic_buffer_extend(&workload_buf, data_size);
				cuckooKV* target_key = (cuckooKV*) (workload_buf.ptr + op->data_pos);
				target_key->keyLength = kv->keyLength;
				memcpy(target_key->kv, kv->kv, kv->keyLength);
			}
			break;

			case YCSB_READ_LATEST:
				// Data for read-latest ops is generated separately
				break;

			case YCSB_RMW:
			case YCSB_UPDATE:{
				kv = kvs[rand_dist(&dist)];
				data_size = sizeof(cuckooKV) + kv->keyLength + kv->valueLength;
				op->data_pos = dynamic_buffer_extend(&workload_buf, data_size);

				cuckooKV* newKV = (cuckooKV*) (workload_buf.ptr + op->data_pos);
				newKV->keyLength = kv->keyLength;
				newKV->valueLength = kv->valueLength;
				memcpy(newKV->kv, kv->kv, kv->keyLength);
				memset(newKV->kv + newKV->keyLength, 7, newKV->valueLength);  // Update to a dummy value
			}
			break;

			case YCSB_INSERT:{
				kv = kvs[insert_offset + num_inserts];
				num_inserts++;
				data_size = sizeof(cuckooKV) + kv->keyLength + kv->valueLength;
				op->data_pos = dynamic_buffer_extend(&workload_buf, data_size);

				memcpy(workload_buf.ptr + op->data_pos, kv, data_size);
			}
			break;

			default:
				printf("Error: Unknown YCSB op type\n");
				return;
		}
	}

	// Create the read-latest key blocks
	uint64_t block;
	uint64_t thread;
	for (thread = 0; thread < num_threads; thread++) {
		uint8_t** block_offsets = (uint8_t**) malloc(sizeof(uint64_t) * (num_inserts + 1));
		workload->read_latest_blocks_for_thread[thread] = block_offsets;

		// We have one block for each amount of inserts between 0 and num_inserts, /inclusive/
		for (block = 0; block < num_inserts + 1; block++) {
			for (i = 0; i < read_latest_block_size; i++) {
				uint64_t backwards = rand_dist(&backward_dist);
				if (backwards < block * num_threads) {
					// This read-latest op refers to a key that was inserted during the workload
					backwards /= num_threads;
					kv = kvs[insert_offset + block - backwards - 1];
				} else {
					// This read-latest op refers to a key that was loaded before the workload started
					backwards -= block * num_threads;
					kv = kvs[workload->initial_num_keys - backwards - 1];
				}

				data_size = sizeof(cuckooKV) + kv->keyLength;
				uint64_t data_pos = dynamic_buffer_extend(&workload_buf, data_size);

				cuckooKV* key = (cuckooKV*) (workload_buf.ptr + data_pos);
				key->keyLength = kv->keyLength;
				memcpy(key->kv, kv->kv, key->keyLength);

				if (i == 0)
					block_offsets[block] = (uint8_t*) data_pos;
			}

			uint64_t sentinel_pos = dynamic_buffer_extend(&workload_buf, sizeof(cuckooKV));
			cuckooKV* sentinel = (cuckooKV*) (workload_buf.ptr + sentinel_pos);
			sentinel->keyLength = 0xFFFFFFFFU;
		}
	}

	workload->data_buf = workload_buf.ptr;

	// Now that the final buffer address is known, convert the read-latest offsets to pointers
	for (thread = 0; thread < num_threads; thread++) {
		for (block = 0; block < num_inserts + 1; block++)
			workload->read_latest_blocks_for_thread[thread][block] += (uintptr_t) (workload->data_buf);
	}
}

bool compare(cuckooKV* kv1, cuckooKV* kv2)
{
	return memcmp(kv1->kv, kv2->kv, kv1->keyLength) < 0;
}

void ycsb(char* dataset_name, const ycsb_workload_spec* spec) {
	struct timespec start_time;
	struct timespec end_time;
	ycsb_thread_ctx ctx;
	dataset_t dataset;
	cuckooKV** kv_ptrs;
	int result;
	uint64_t i;

	seed_and_print();
	result = init_dataset(&dataset, dataset_name, DATASET_ALL_KEYS);
	if (!result) {
		printf("Error creating dataset.\n");
		return;
	}

	printf("Generate YCSB workloads\n");
	kv_ptrs = read_kvs(&dataset, DEFAULT_VALUE_SIZE);

	CuckooHashTable index;

	// Create workload
	generate_ycsb_workload(&dataset, kv_ptrs, &(ctx.workload), spec, 0, 1);

	// Initialize context
	ctx.index = &index; 
	ctx.thread_id = 0;
	ctx.num_threads = 1;
	ctx.inserts_done = 0;
	ctx.thread_contexts = &ctx;

	// Fill the tree
	printf("Loading\n");
	std::sort(kv_ptrs, kv_ptrs + ctx.workload.initial_num_keys, compare);
	for (i = 0; i < ctx.workload.initial_num_keys; i++) {
		index.insert(kv_ptrs[i], kv_ptrs[i]);
	}

	// Perform YCSB ops
	printf("Perform YCSB ops\n");
	notify_critical_section_start();
	clock_gettime(CLOCK_MONOTONIC, &start_time);
	execute_ycsb_workload<CuckooHashTable*>(&ctx); 
	clock_gettime(CLOCK_MONOTONIC, &end_time);
	notify_critical_section_end();
	float time_took = time_diff(&end_time, &start_time);
	report(time_took, spec->num_ops);
}


const flag_spec_t FLAGS[] = {
	{ "--profiler-pid", 1},
	{ "--threads", 1},
	{ "--dataset-size", 1},
	{ "--ycsb-uniform-dist", 0},
	{ NULL, 0}
};

int main(int argc, char** argv) {
	int result;
	char* test_name;
	char* dataset_name = NULL;
	int num_threads;
	dataset_t dataset;
	uint64_t dataset_size;
	ycsb_workload_spec ycsb_workload;
	int is_ycsb = 0;
	int is_mt_ycsb = 0;
	args_t* args = parse_args((flag_spec_t*) FLAGS, argc, argv);

	if (args == NULL) {
		printf("Commandline error\n");
		return 1;
	}
	if (args->num_args < 1) {
		printf("Missing test name\n");
		return 1;
	}
	profiler_pid = get_int_flag(args, "--profiler-pid", PID_NO_PROFILER);
	test_name = args->args[0];

	if (!strcmp(test_name, "load-uint64")) {
		int num_keys = DEFAULT_NUM_KEYS;
		if (argc >= 3)
			num_keys = atoi(argv[2]);
		// load_uint64(num_keys);
		return 0;
	}

	if (args->num_args < 2) {
		printf("Missing dataset name\n");
		return 1;
	}
	dataset_name = args->args[1];
	num_threads = get_int_flag(args, "--threads", DEFAULT_NUM_THREADS);
	bind_core(0);
	if (!strcmp(test_name, "pos-lookup")) {
		// seed_and_print();
		dataset_size = get_uint64_flag(args, "--dataset-size", DATASET_ALL_KEYS);
		result = init_dataset(&dataset, dataset_name, dataset_size);
		if (!result) {
			printf("Error creating dataset.\n");
			return 1;
		}
		pos_lookup(&dataset);
		return 0;
	}

	if (!strcmp(test_name, "mt-pos-lookup")) {
		mt_pos_lookup(dataset_name, num_threads);
		return 0;
	}

	if (!strcmp(test_name, "mt-insert")) {
		mt_insert(dataset_name, num_threads);
		return 0;
	}
	
	if (!strcmp(test_name, "ycsb-a")) {
		ycsb_workload = YCSB_A_SPEC;
		is_ycsb = 1;
	}
	if (!strcmp(test_name, "ycsb-b")) {
		ycsb_workload = YCSB_B_SPEC;
		is_ycsb = 1;
	}
	if (!strcmp(test_name, "ycsb-c")) {
		ycsb_workload = YCSB_C_SPEC;
		is_ycsb = 1;
	}
	if (!strcmp(test_name, "ycsb-d")) {
		ycsb_workload = YCSB_D_SPEC;
		is_ycsb = 1;
	}
	if (!strcmp(test_name, "ycsb-e")) {
		ycsb_workload = YCSB_E_SPEC;
		is_ycsb = 1;
	}
	if (!strcmp(test_name, "ycsb-f")) {
		ycsb_workload = YCSB_F_SPEC;
		is_ycsb = 1;
	}

	if (!strcmp(test_name, "read-heavy")) {
		ycsb_workload = READ_HEAVY_SPEC;
		is_ycsb = 1;
	}
	if (!strcmp(test_name, "read-write-balanced"))
	{
		ycsb_workload = READ_WRITE_BALANCED_SPEC;
		is_ycsb = 1;
	}
	if (!strcmp(test_name, "write-heavy"))
	{
		ycsb_workload = WRITE_HEAVY_SPEC;
		is_ycsb = 1;
	}

	if (!strcmp(test_name, "write-only"))
	{
		ycsb_workload = WRITE_ONLY_SPEC;
		is_ycsb = 1;
	}

	if ((is_ycsb) && has_flag(args, "--ycsb-uniform-dist"))
		ycsb_workload.distribution = DIST_UNIFORM;

	if (is_ycsb) {
		ycsb(dataset_name, &ycsb_workload);
		return 0;
	}

	if (!strcmp(test_name, "mem-usage")) {
		mem_usage(dataset_name);
		return 0;
	}

	printf("Unknown test name '%s'\n", test_name);
	return 1;
}
