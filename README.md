# Indexes used for Cuckoo Trie benchmarks


The indexes included are: [HOT](https://github.com/speedskater/hot), [Wormhole](https://github.com/wuxb45/wormhole), [ARTOLC](https://github.com/wangziqi2016/index-microbench/tree/master/ARTOLC), [STX](https://github.com/tlx/tlx/tree/master/tlx/container), [Cuckoohash](https://github.com/efficient/libcuckoo), [ALEX](https://github.com/microsoft/ALEX). For each index the repository contains:
- The version of the index used in the paper.
- A benchmarking program.

### Compilation

Running `make` in the directory of each index will build the index and a benchmarking program named `test_<index>`.

### Usage (all indexes except MlpIndex)

```sh
./test_<index> [flags] BENCHMARK DATASET
```

`BENCHMARK` is one of the available benchmark types and `DATASET` is the path of the file containing the dataset to be used. The special name `rand-<k>` specifies a dataset of 10M random `k`-byte keys.

The seed used for the random generator is printed at the start of each run to allow for debugging benchmarks that only fail for certain seeds.

Most benchmarks have a single-threaded and a multi-threaded version (named `mt-*`). The multi-threaded versions are only available for thread-safe indexes (HOT, Wormhole and ARTOLC). The available benchmark types are:
- `insert`, `mt-insert`: Insert all keys into the index.
- `pos-lookup`, `mt-pos-lookup`: Perform positive lookups for random keys. Positive lookups are ones that succeed (that is, ask for keys that are present in the trie).
- `mem-usage`: Report the memory usage of the index after inserting all keys. To have a fair comparison, the result includes just the index size, without the memory required to store the keys themselves.
- `ycsb-a`, `ycsb-b`, ... `ycsb-f`, `mt-ycsb-a`, ..., `mt-ycsb-f`: Run the appropriate mix of insert and lookup operations from the [YCSB](https://github.com/brianfrankcooper/YCSB/wiki/Core-Workloads) benchmark suite. By default, this runs the benchmark with a Zipfian query distribution, specify `--ycsb-uniform-dist` to use a uniform distribution instead.

The following flags can be used with each benchmark:
- `--dataset-size <N>` (`pos-lookup` only): Use only the first `N` keys of the dataset.
- `--threads <N>` (`mt-*` only): use `N` threads. Each thread is bound to a different core. The default is to use 4 threads.
- `--ycsb-uniform-dist` (`ycsb-*` only): Run YCSB benchmarks with a uniform query distribution. The default is Zipfian distribution.

### Dataset file format

The dataset files used with `benchmark` are binary files with the following format:
- `Number of keys`: a 64-bit little-endian number.
- `Total size of all keys`: a 64-bit little-endian number. This number does not include the size that precedes each key.
- `Keys`: Each key is encoded as a 32-bit little-endian length `L`, followed by `L` key bytes. The keys are not NULL-terminated.

