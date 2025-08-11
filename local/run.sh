#!/bin/bash

# 脚本出错时立即退出
set -e

echo ">>> [Step 1/2] Compiling the project with make..."

# 清理旧的编译结果，确保是全新编译
make clean

# 使用所有可用的 CPU核心进行并行编译
# 对于只有一个 .cpp 文件的项目，-j 选项可能不会带来显著速度提升，但对于多文件项目是好的实践
make -j$(nproc)

echo
echo ">>> [Step 2/2] Running the benchmark..."
echo "=================================================="

# 运行编译好的可执行文件
# 使用 stdbuf -oL 来确保输出是行缓冲的，这样即使重定向到文件也能实时看到输出
stdbuf -oL ./main

echo "=================================================="
echo ">>> Benchmark finished successfully!"
