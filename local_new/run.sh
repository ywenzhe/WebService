#!/bin/bash

# 脚本名称: run_local.sh
# 描述: 编译和运行本地 WebService 基准测试程序

# --- 编译 ---
echo ">>> Compiling main_local.cpp..."

# 编译器和标志
# -std=c++17: 使用 C++17 标准
# -O3:        最高级别优化
# -o main_local: 输出可执行文件名为 main_local
# -I.:        将当前目录加入头文件搜索路径 (用于 helpers.hpp 和 zipf.hpp)
# -ltbb:      链接 Intel TBB 库
# -lcryptopp: 链接 Crypto++ 库
# -lsnappy:   链接 Snappy 库
# -pthread:   链接 POSIX 线程库
g++ -std=c++17 -O3 main.cpp -o main_local -I. \
    -ltbb -lcryptopp -lsnappy -pthread

# 检查编译是否成功
if [ $? -ne 0 ]; then
    echo ">>> Compilation failed!"
    exit 1
fi

echo ">>> Compilation successful."

# --- 运行 ---
echo ">>> Running the benchmark..."
echo ">>> WARNING: This program will consume approximately 26 GB of RAM."
echo ">>> Press Ctrl+C to abort if you do not have enough memory."
sleep 3

# 运行基准测试
time ./main_local

echo ">>> Benchmark finished."

