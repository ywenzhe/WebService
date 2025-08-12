#!/bin/bash

source ../../shared.sh

arr_aifm_heap_size=( 11605 13824 16043 18261 20480 )
arr_hashtable_idx_shift=( 28 28 28 28 28 )

sudo pkill -9 main

for ((i=0;i<${#arr_aifm_heap_size[@]};++i)); do
    cur_heap_size=${arr_aifm_heap_size[i]}
    cur_idx_shift=${arr_hashtable_idx_shift[i]}
    local_mem_size=$(( (1 << ($cur_idx_shift - 20)) * 24 + $cur_heap_size ))

    # --- 新增的提示信息开始 ---
    echo "=================================================="
    echo ">>> 开始处理 Local_mem_size: ${local_mem_size} MB <<<"
    echo "=================================================="
    # --- 新增的提示信息结束 ---

    sed "s/constexpr static uint64_t kCacheSize = .*/constexpr static uint64_t kCacheSize = $cur_heap_size * Region::kSize;/g" main.cpp -i
    sed "s/constexpr static uint32_t kLocalHashTableNumEntriesShift = .*/constexpr static uint32_t kLocalHashTableNumEntriesShift = $cur_idx_shift;/g" main.cpp -i    
    make clean
    make -j
    echo ">>> 启动 iokerneld..."
    if ! rerun_local_iokerneld_args simple 1,2,3,4,5,6,7,8,9,11,12,13,14,15; then
        echo "错误：iokerneld 启动失败"
        exit 1
    fi

    echo ">>> 启动 mem_server..."
    if ! rerun_mem_server; then
        echo "错误：mem_server 启动失败"
        exit 1
    fi

    echo ">>> 运行主程序..."
    run_program ./main 2>&1 | tee log.$local_mem_size
done

kill_local_iokerneld
