#!/bin/bash

# for bench in ycsb-a ycsb-b ycsb-c ycsb-d ycsb-e ycsb-f 
# do
#     for dataset in ../wiki_uint64 ../fb_uint64 ../osm_uint64 ../rand_uint64
#     do
#         echo "./test_alex ${bench} ${dataset}"
#         echo "./test_alex ${bench} ${dataset}" >> result_alex.txt
#         ./test_alex ${bench} ${dataset} >> result_alex.txt
#     done
# done

# for bench in read-heavy read-write-balanced write-heavy
# do
#     for dataset in ../wiki_uint64 ../fb_uint64 ../osm_uint64 ../rand_uint64
#     do
#         echo "./test_alex ${bench} ${dataset}"
#         echo "./test_alex ${bench} ${dataset}" >> result_alex.txt
#         ./test_alex ${bench} ${dataset} >> result_alex.txt
#     done
# done

for bench in RW8_2 RW6_4 RW4_6 RW2_8 RW0_10
do
    for dataset in ../wiki_uint64_rand ../fb_uint64_rand ../osm_uint64_rand ../rand_uint64_rand
    do
        echo "./test_alex ${bench} ${dataset}"
        echo "./test_alex ${bench} ${dataset}" >> result_alex.txt
        ./test_alex ${bench} ${dataset} >> result_alex.txt
    done
done